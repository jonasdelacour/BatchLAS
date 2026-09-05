// Native batched CSR SpMM kernels: gather (transA == NoTrans), plus scale and
// scatter (transposed). spmm_native.hh carries the CSR indexing contract and the
// beta == 0 / alpha == 0 semantics. evidence: docs/perf/spmm.md

#include "spmm_native.hh"

#include "../queue.hh"
#include "device_scalar.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_spmm {

using sycl_device::Cx;
using sycl_device::DevMap;
using sycl_device::dev_conj;
using sycl_device::dev_is_complex_v;
using sycl_device::dev_mul;
using sycl_device::fma_acc;

namespace {

// Body-for-body copy of gemv_wg_ladder (gemv_native.cc), kept diffable -- hence
// the `units_per_wg_shift` parameter that every caller here passes 0 for.
inline int spmm_wg_ladder(int64_t work_units, int max_wg, int cu,
                          int units_per_wg_shift) {
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t per_wg = cand >> units_per_wg_shift;
        if (per_wg < 1) continue;
        const int64_t groups = (work_units + per_wg - 1) / per_wg;
        if (groups >= static_cast<int64_t>(4) * cu) break;
    }
    // No admissible candidate leaves `wg` at its initial 32, which may exceed the
    // device limit: that is an INVALID nd_range, not merely a slow one.
    if (max_wg > 0 && wg > max_wg) wg = max_wg;
    return wg;
}

// 64 bytes of accumulator for every scalar type, so the register block shrinks as
// the scalar widens. evidence: docs/perf/spmm.md#the-gather-window
template <typename D>
inline constexpr int kNCmax = 64 / static_cast<int>(sizeof(D));

static_assert(kNCmax<float> == 16 && kNCmax<double> == 8, "64-byte register block");
static_assert(kNCmax<Cx<float>> == 8 && kNCmax<Cx<double>> == 4, "64-byte register block");

// The complex overload below is two independent scalar atomics, NOT an atomic
// complex add: safe only because nothing reads C between the scatter's first
// atomic and its last.
template <typename R>
inline void spmm_atomic_add(R* p, R v) {
    sycl::atomic_ref<R, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*p);
    ref.fetch_add(v);
}

template <typename R>
inline void spmm_atomic_add(Cx<R>* p, Cx<R> v) {
    spmm_atomic_add(&p->re, v.re);
    spmm_atomic_add(&p->im, v.im);
}

template <typename T, int NC> class SpmmGatherKernel;
template <typename T> class SpmmScaleKernel;
template <typename T> class SpmmScatterKernel;

// Gather arm (transA == NoTrans):
//   C[i, c] = alpha * sum_{p in row i of A} A_val[p] * op(B)[A_ci[p], c]
//             + beta * C[i, c]
// One work-item per (batch item, row i, block of NC output columns), flattened
// rows-fastest. The item owns every element it writes: no atomic, no barrier.
template <typename T, int NC>
Event spmm_gather(Queue& ctx,
                  const MatrixView<T, MatrixFormat::CSR>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B_mat,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha, T beta, Transpose transB) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int m = A.rows();
    const int nrhs = C.cols();
    const int batch = A.batch_size();

    const int64_t cblocks = (static_cast<int64_t>(nrhs) + NC - 1) / NC;
    const int64_t rc = static_cast<int64_t>(m) * cblocks;
    const int64_t items = rc * batch;
    if (items <= 0) return ctx.get_event();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = spmm_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__: LOBPCG passes X/P/R as slices of one buffer and
        // AX/AP/AR of another -- element-disjoint, but aliasing at object level.
        const D* a_val = reinterpret_cast<const D*>(A.data_ptr());
        const int* a_ro = A.row_offsets().data();
        const int* a_ci = A.col_indices().data();

        // Two distinct strides, both widened: b*stride overflows an int.
        const int64_t a_os = A.offset_stride();
        const int64_t a_ms = A.matrix_stride();

        const D* b_ptr = reinterpret_cast<const D*>(B_mat.data_ptr());
        const int64_t ldb = B_mat.ld();
        // Read from the view, never derived as ld*cols.
        const int64_t sb = B_mat.stride();

        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int64_t ldc = C.ld();
        const int64_t sc = C.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        // beta == 0 means C is NOT read -- the contract, not an optimisation:
        // callers pass unzeroed BumpAllocator memory as C.
        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool b_notrans = (transB == Transpose::NoTrans);
        const bool conj_b = (transB == Transpose::ConjTrans);

        const int out_rows = m;
        const int width = nrhs;
        const int64_t rowblocks = rc;
        const int64_t total = items;

        h.parallel_for<SpmmGatherKernel<T, NC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / rowblocks;
                const int64_t rem = gid - b * rowblocks;
                const int64_t cb = rem / out_rows;
                const int i = static_cast<int>(rem - cb * static_cast<int64_t>(out_rows));
                const int c0 = static_cast<int>(cb) * NC;

                // Bases widened by hand: KernelMatrixView::get computes its own
                // in plain int and wraps, so no body here calls get().
                const int64_t ro = b * a_os;   // indexes a_ro ONLY
                const int64_t vb = b * a_ms;   // indexes a_val AND a_ci
                const D* Bb = b_ptr + b * sb;
                D* Cb = c_ptr + b * sc;

                // The only legal bound. Never A.nnz(): that is the per-item
                // capacity, and slots above each item's own count are garbage.
                const int rs = a_ro[ro + i];
                const int re = a_ro[ro + i + 1];

                D acc[NC];
#pragma unroll
                for (int t = 0; t < NC; ++t) acc[t] = D{};

                if (!alpha_zero) {
                    for (int p = rs; p < re; ++p) {
                        const D av = a_val[vb + p];
                        const int j = a_ci[vb + p];  // a COLUMN of A: a row of B
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            const int c = c0 + t;
                            if (c < width) {
                                // op(B)[j,c]: NoTrans -> Bb[c*ldb+j], else
                                // Bb[j*ldb+c] (conjugated for ConjTrans).
                                D bv = b_notrans
                                           ? Bb[static_cast<int64_t>(c) * ldb + j]
                                           : Bb[static_cast<int64_t>(j) * ldb + c];
                                if constexpr (dev_is_complex_v<D>) {
                                    if (conj_b) bv = dev_conj(bv);
                                }
                                fma_acc(acc[t], av, bv);
                            }
                        }
                    }
                }

#pragma unroll
                for (int t = 0; t < NC; ++t) {
                    const int c = c0 + t;
                    if (c < width) {
                        const int64_t ci = static_cast<int64_t>(c) * ldc + i;
                        D out{};
                        fma_acc(out, alpha_d, acc[t]);
                        if (!beta_zero) fma_acc(out, beta_d, Cb[ci]);
                        Cb[ci] = out;
                    }
                }
            });
    });

    return ctx.get_event();
}

// Scale arm: runs only ahead of the scatter, which cannot fold beta into its
// accumulation -- no item owns an output element, so beta*C_old must land exactly
// once before any atomic. At beta == 0 it stores zero rather than reading C.
template <typename T>
Event spmm_scale(Queue& ctx,
                 const MatrixView<T, MatrixFormat::Dense>& C,
                 T beta, int out_rows, int batch) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int nrhs = C.cols();
    const int64_t per = static_cast<int64_t>(out_rows) * nrhs;
    const int64_t elems = per * batch;
    if (elems <= 0) return ctx.get_event();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = spmm_wg_ladder(elems, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (elems + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int64_t ldc = C.ld();
        const int64_t sc = C.stride();

        D beta_d;
        __builtin_memcpy(&beta_d, &beta, sizeof(D));
        const bool beta_zero = (beta == T(0));

        const int rows_out = out_rows;
        const int64_t stride_out = per;
        const int64_t total = elems;

        h.parallel_for<SpmmScaleKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / stride_out;
                const int64_t r = gid - b * stride_out;
                const int c = static_cast<int>(r / rows_out);
                const int j = static_cast<int>(r - static_cast<int64_t>(c) * rows_out);

                D* p = c_ptr + b * sc + static_cast<int64_t>(c) * ldc + j;
                *p = beta_zero ? D{} : dev_mul(beta_d, *p);
            });
    });

    return ctx.get_event();
}

// Scatter arm (transA == Trans or ConjTrans):
//   C[A_ci[p], c] += alpha * conj?(A_val[p]) * op(B)[i, c]
// One work-item per (batch item, row i of the STORED A) -- a column of op(A): it
// reads one row of op(B) and pushes into many rows of C. NCS is fixed at 4; this
// arm has no in-tree callers. The unsigned range guard on j is not optional: out
// of range here is an out-of-range ATOMIC WRITE into uninitialised nnz padding.
template <typename T>
Event spmm_scatter(Queue& ctx,
                   const MatrixView<T, MatrixFormat::CSR>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B_mat,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha, int out_rows, bool conjugate_a, Transpose transB) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    // Rows of the STORED A, which under Trans/ConjTrans is the REDUCTION extent.
    const int red_rows = A.rows();
    const int nrhs = C.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(red_rows) * batch;
    if (items <= 0 || nrhs <= 0) return ctx.get_event();

    // A host-side skip, not an in-kernel early return: the scale has already
    // written C = beta*C, and this body only ever adds the alpha term.
    if (alpha == T(0)) return ctx.get_event();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = spmm_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        const D* a_val = reinterpret_cast<const D*>(A.data_ptr());
        const int* a_ro = A.row_offsets().data();
        const int* a_ci = A.col_indices().data();
        const int64_t a_os = A.offset_stride();
        const int64_t a_ms = A.matrix_stride();

        const D* b_ptr = reinterpret_cast<const D*>(B_mat.data_ptr());
        const int64_t ldb = B_mat.ld();
        const int64_t sb = B_mat.stride();

        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int64_t ldc = C.ld();
        const int64_t sc = C.stride();

        D alpha_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));

        const bool b_notrans = (transB == Transpose::NoTrans);
        const bool conj_b = (transB == Transpose::ConjTrans);
        const bool conj_a = conjugate_a;

        const int rows_in = red_rows;
        const int rows_out = out_rows;
        const int width = nrhs;
        const int64_t total = items;

        constexpr int NCS = 4;

        h.parallel_for<SpmmScatterKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / rows_in;
                const int i = static_cast<int>(gid - b * static_cast<int64_t>(rows_in));

                const int64_t ro = b * a_os;
                const int64_t vb = b * a_ms;
                const D* Bb = b_ptr + b * sb;
                D* Cb = c_ptr + b * sc;

                const int rs = a_ro[ro + i];
                const int re = a_ro[ro + i + 1];

                for (int c0 = 0; c0 < width; c0 += NCS) {
                    D ab[NCS];
#pragma unroll
                    for (int t = 0; t < NCS; ++t) {
                        const int c = c0 + t;
                        ab[t] = D{};
                        if (c < width) {
                            // op(B)[i,c]: NoTrans -> Bb[c*ldb+i], else Bb[i*ldb+c].
                            D bv = b_notrans ? Bb[static_cast<int64_t>(c) * ldb + i]
                                             : Bb[static_cast<int64_t>(i) * ldb + c];
                            if constexpr (dev_is_complex_v<D>) {
                                if (conj_b) bv = dev_conj(bv);
                            }
                            ab[t] = dev_mul(alpha_d, bv);
                        }
                    }

                    for (int p = rs; p < re; ++p) {
                        D av = a_val[vb + p];
                        // ConjTrans conjugates the SPARSE operand.
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj_a) av = dev_conj(av);
                        }
                        const int j = a_ci[vb + p];  // an OUTPUT ROW of C
                        if (static_cast<unsigned>(j) >= static_cast<unsigned>(rows_out)) continue;
#pragma unroll
                        for (int t = 0; t < NCS; ++t) {
                            const int c = c0 + t;
                            if (c < width) {
                                spmm_atomic_add(Cb + static_cast<int64_t>(c) * ldc + j,
                                                dev_mul(av, ab[t]));
                            }
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

}  // namespace

template <typename T>
Event spmm_native_csr(Queue& ctx,
                      const MatrixView<T, MatrixFormat::CSR>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B_mat,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha, T beta, Transpose transA, Transpose transB) {
    using D = typename DevMap<T>::type;

    const bool a_notrans = (transA == Transpose::NoTrans);
    const int out_rows = a_notrans ? A.rows() : A.cols();
    const int nrhs = C.cols();
    const int batch = A.batch_size();

    // Deliberately NOT reference ?GEMV's (alpha == 0 && beta == 1) quick return:
    // at alpha == 0 the answer is still C = beta*C, which the arms below write.
    if (out_rows == 0 || nrhs == 0 || batch <= 0) return ctx.get_event();

    if (a_notrans) {
        // nrhs <= 2 is the lanczos shape; everything else gets the full block.
        if (nrhs <= 2) {
            return spmm_gather<T, 2>(ctx, A, B_mat, C, alpha, beta, transB);
        }
        return spmm_gather<T, kNCmax<D>>(ctx, A, B_mat, C, alpha, beta, transB);
    }

    // Ordered: the scale must complete before the first atomic. The default queue
    // is in_order; an out-of-order one pays one host wait here.
    (void)spmm_scale<T>(ctx, C, beta, out_rows, batch);
    if (!ctx.in_order()) ctx.wait();
    return spmm_scatter<T>(ctx, A, B_mat, C, alpha, out_rows,
                           transA == Transpose::ConjTrans, transB);
}

// Specialised in the same TU as the kernels, so a build that drops this file
// cannot advertise an unlinked kernel. A statement about the BUILD, not the
// device: there is deliberately no is_gpu gate.
template <> bool spmm_gather_available<float>()                { return true; }
template <> bool spmm_gather_available<double>()               { return true; }
template <> bool spmm_gather_available<std::complex<float>>()  { return true; }
template <> bool spmm_gather_available<std::complex<double>>() { return true; }

template <> bool spmm_scatter_available<float>()                { return true; }
template <> bool spmm_scatter_available<double>()               { return true; }
template <> bool spmm_scatter_available<std::complex<float>>()  { return true; }
template <> bool spmm_scatter_available<std::complex<double>>() { return true; }

#define BATCHLAS_SPMM_NATIVE_INSTANTIATE(fp)                                   \
    template Event spmm_native_csr<fp>(                                        \
        Queue&, const MatrixView<fp, MatrixFormat::CSR>&,                      \
        const MatrixView<fp, MatrixFormat::Dense>&,                            \
        const MatrixView<fp, MatrixFormat::Dense>&, fp, fp, Transpose,         \
        Transpose);

BATCHLAS_SPMM_NATIVE_INSTANTIATE(float)
BATCHLAS_SPMM_NATIVE_INSTANTIATE(double)
BATCHLAS_SPMM_NATIVE_INSTANTIATE(std::complex<float>)
BATCHLAS_SPMM_NATIVE_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SPMM_NATIVE_INSTANTIATE

}  // namespace batchlas::sycl_spmm
