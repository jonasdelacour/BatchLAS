// SPDX-License-Identifier: MIT
// Native batched CSR SpMM kernels: gather (transA == NoTrans), plus scale and
// scatter (transposed). CSR indexing and beta == 0 / alpha == 0 semantics:
// evidence: docs/perf/spmm.md#spmm-the-kernel-contract

#include <src/ops/spmm/spmm_native.hh>
#include <src/ops/gemv/gemv_native.hh>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/sparse.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/util/batch_mask.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::spmm {

using sycl_device::Cx;
using sycl_device::DevMap;
using sycl_device::dev_is_complex_v;

namespace {

// The work-group ladder is gemv's (src/ops/gemv/gemv_native.hh); every caller here passes
// units_per_wg_shift 0 (one work-item per unit).
using ops::gemv::gemv_wg_ladder;

// 64 bytes of accumulator for every scalar type, so the register block shrinks as
// the scalar widens. evidence: docs/perf/spmm.md#the-gather-window
template <typename D>
inline constexpr int kNCmax = 64 / static_cast<int>(sizeof(D));

static_assert(kNCmax<float> == 16 && kNCmax<double> == 8, "64-byte register block");
static_assert(kNCmax<Cx<float>> == 8 && kNCmax<Cx<double>> == 4, "64-byte register block");

// The scatter's complex atomic (device::atomic_accumulate) is two independent scalar
// atomics, NOT an atomic complex add: safe only because nothing reads C between
// the scatter's first atomic and its last.

// Pair = one two-component load per complex element (device::PairAccess). As two scalar
// loads, both halves of a missing sector go to L2 separately, and the gather is
// L2-bound: 24% more L2 sectors and 18% slower for cfloat under -ffp-model=precise,
// whose schedule issues every column's pair at once. evidence: docs/perf/blackwell.md#cfloat-spmm-under-the-precise-fp-model

// The pair load needs 2*sizeof(R) alignment, which std::complex<R> does not
// promise: a view built on a raw pointer may be offset by one R.
template <typename D>
inline bool spmm_pair_aligned(const void* a, const void* b, const void* c) {
    const auto mis = [](const void* p) {
        return reinterpret_cast<std::uintptr_t>(p) % sizeof(D) != 0;
    };
    return dev_is_complex_v<D> && !mis(a) && !mis(b) && !mis(c);
}

template <typename T, int NC, bool Pair> class SpmmGatherKernel;
template <typename T> class SpmmScaleKernel;
template <typename T> class SpmmScatterKernel;

// Gather arm (transA == NoTrans):
//   C[i, c] = alpha * sum_{p in row i of A} A_val[p] * op(B)[A_ci[p], c]
//             + beta * C[i, c]
// One work-item per (batch item, row i, block of NC output columns), flattened
// rows-fastest. The item owns every element it writes: no atomic, no barrier.
template <typename T, int NC, bool Pair>
Event spmm_gather(Queue& ctx,
                  const MatrixView<T, MatrixFormat::CSR>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B_mat,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha, T beta, Transpose transB, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;
    using Ar = device::arith_t<device::arith::Fma, D>;  // sycl_device::fma_acc's chain, bit for bit
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
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
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

        batchlas::impl::parallel_for<SpmmGatherKernel<T, NC, Pair>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / rowblocks;
                if (mask != nullptr && mask[b] == 0) return;  // I3: an inactive item's C is untouched
                const int64_t rem = gid - b * rowblocks;
                const int64_t cb = rem / out_rows;
                const int i = static_cast<int>(rem - cb * static_cast<int64_t>(out_rows));
                const int c0 = static_cast<int>(cb) * NC;

                // Bases widened by hand: KernelMatrixView::get computes its own
                // in plain int and wraps, so no body here calls get().
                const int64_t ro = b * a_os;   // indexes a_ro ONLY
                const int64_t vb = b * a_ms;   // indexes a_val AND a_ci

                // Row i of A; its bound is the row offsets, never A.nnz() (the per-item
                // capacity: slots above each item's own count are garbage). A column of A
                // is a row of op(B) = B, B^T or B^H (transB is a kernel argument).
                const auto arow = device::sp_row(device::LoadVec<const D, Pair>{a_val + vb},
                                             device::BasedVec<const int, int64_t>{a_ci, vb},
                                             a_ro[ro + i], a_ro[ro + i + 1]);
                const device::RtOpMat<const D, Pair> opB{b_ptr + b * sb, ldb, b_notrans, conj_b};
                const device::ColMat64<D, Pair> Cb{c_ptr + b * sc, ldc};

                // C(i, c0:c0+NC) = alpha A(i, :) op(B)(:, c0:c0+NC) + beta C: the device spmm's unit.
                device::detail::gather_unit<Ar, NC>(alpha_zero, arow, opB, c0, width, alpha_d, device::MaybeZero{beta_d, beta_zero}, Cb, i);
            });
    });

    return ctx.get_event();
}

template <typename T, int NC>
Event spmm_gather_pair(Queue& ctx,
                       const MatrixView<T, MatrixFormat::CSR>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B_mat,
                       const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha, T beta, Transpose transB, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;
    if constexpr (dev_is_complex_v<D>) {
        if (spmm_pair_aligned<D>(A.data_ptr(), B_mat.data_ptr(), C.data_ptr())) {
            return spmm_gather<T, NC, true>(ctx, A, B_mat, C, alpha, beta, transB, mask);
        }
    }
    return spmm_gather<T, NC, false>(ctx, A, B_mat, C, alpha, beta, transB, mask);
}

// Scale arm: runs only ahead of the scatter, which cannot fold beta into its
// accumulation -- no item owns an output element, so beta*C_old must land exactly
// once before any atomic. At beta == 0 it stores zero rather than reading C.
template <typename T>
Event spmm_scale(Queue& ctx,
                 const MatrixView<T, MatrixFormat::Dense>& C,
                 T beta, int out_rows, int batch, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int nrhs = C.cols();
    const int64_t per = static_cast<int64_t>(out_rows) * nrhs;
    const int64_t elems = per * batch;
    if (elems <= 0) return ctx.get_event();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(elems, max_wg, cu, /*units_per_wg_shift=*/0);
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

        batchlas::impl::parallel_for<SpmmScaleKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / stride_out;
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const int64_t r = gid - b * stride_out;
                const int c = static_cast<int>(r / rows_out);
                const int j = static_cast<int>(r - static_cast<int64_t>(c) * rows_out);

                D* p = c_ptr + b * sc + static_cast<int64_t>(c) * ldc + j;
                *p = beta_zero ? D{} : device::arith_t<device::arith::Fma, D>::mul(beta_d, *p);  // C := beta C
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
                   T alpha, int out_rows, bool conjugate_a, Transpose transB, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;
    using Ar = device::arith_t<device::arith::Fma, D>;  // sycl_device::dev_mul, bit for bit
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
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
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

        batchlas::impl::parallel_for<SpmmScatterKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / rows_in;
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const int i = static_cast<int>(gid - b * static_cast<int64_t>(rows_in));

                const int64_t ro = b * a_os;
                const int64_t vb = b * a_ms;
                const device::RtOpMat<const D> opB{b_ptr + b * sb, ldb, b_notrans, conj_b};
                D* Cb = c_ptr + b * sc;
                const int rs = a_ro[ro + i];
                const int re = a_ro[ro + i + 1];

                // C(A_ci, :) += conj?(A(i, :))^T (alpha op(B)(i, :)), the device spmm's scatter row;
                // ConjTrans conjugates the SPARSE operand.
                device::detail::scatter_row<Ar, NCS>(width, alpha_d, opB, i, rs, re, a_val, vb, conj_a, a_ci, rows_out,
                                                      device::detail::ColPtr<D>{Cb, ldc});
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
    const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("spmm", A, B_mat, C));

    if (a_notrans) {
        // nrhs <= 2 is the lanczos shape; everything else gets the full block.
        if (nrhs <= 2) {
            return spmm_gather_pair<T, 2>(ctx, A, B_mat, C, alpha, beta, transB, mask);
        }
        return spmm_gather_pair<T, kNCmax<D>>(ctx, A, B_mat, C, alpha, beta, transB, mask);
    }

    // Ordered: the scale must complete before the first atomic. The default queue
    // is in_order; an out-of-order one pays one host wait here.
    (void)spmm_scale<T>(ctx, C, beta, out_rows, batch, mask);
    if (!ctx.in_order()) ctx.wait();
    return spmm_scatter<T>(ctx, A, B_mat, C, alpha, out_rows,
                           transA == Transpose::ConjTrans, transB, mask);
}

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

}  // namespace batchlas::ops::spmm
