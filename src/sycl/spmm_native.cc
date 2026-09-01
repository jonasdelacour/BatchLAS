// Native batched CSR SpMM -- the kernel translation unit.
//
// Read src/sycl/spmm_native.hh first: it carries the four invariants (zero
// local memory, zero workspace, no is_gpu gate, all nine transpose
// combinations), the four-part CSR indexing contract, the beta == 0 / alpha == 0
// semantics and the aliasing rule that forbids __restrict__ and pointer arrays.
//
// This file follows src/sycl/gemv_native.cc and src/sycl/trsm_native.cc:
// geometry lives here beside the launchers, the capability flags are full
// explicit specialisations at the bottom of this same file, and one
// instantiation macro covers the four scalar types.

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

// ---------------------------------------------------------------------------
// THE WORK-GROUP LADDER. Body-for-body gemv_wg_ladder (gemv_native.cc:78-101),
// which lives in an anonymous namespace there and so cannot be shared. The
// signature is kept identical, `units_per_wg_shift` included, so the two stay
// diffable -- but EVERY caller in this file passes 0, because no body here uses
// a sub-group and one work-ITEM is always one unit of work.
//
// It exists to prevent this codebase's signature performance defect: a
// decomposition whose only parallel extent is the batch. The 1-D flattening
// below (b = gid / units_per_item) is what stops the work-group count from
// depending on the batch alone; the ladder then takes the LARGEST candidate
// work-group that still leaves at least 4 work-groups per compute unit, and
// falls to the smallest if none does. Falling to 32 is the important half -- it
// is what keeps a small problem spread across the device instead of packed into
// a handful of fat groups.
// ---------------------------------------------------------------------------
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
    // If NO candidate was admissible the loop never assigned and `wg` is still
    // the 32 it was initialised to, which would be larger than the device
    // allows. Unreachable on both devices this was written against (the CUDA GPU
    // reports MAX_WORK_GROUP_SIZE 1024 and the native_cpu device 2048), but it
    // is an INVALID nd_range rather than a slow one, so it is clamped rather
    // than documented away.
    if (max_wg > 0 && wg > max_wg) wg = max_wg;
    return wg;
}

// ---------------------------------------------------------------------------
// THE REGISTER-BLOCK WIDTH, as a constexpr rather than a per-type table.
//
// 64 bytes of accumulator -- 16 32-bit registers -- for ALL four scalar types:
// float 16, double 8, complex<float> 8, complex<double> 4. That is the recorded
// "thread tile must shrink as the scalar widens" residency rule solved once
// instead of four times.
//
// TWO INSTANTIATED VALUES PER TYPE, {2, kNCmax}, and the second one is not
// gold-plating. At nrhs = 2 -- lanczos's shape, and the hottest one this op
// serves -- NC = kNCmax would waste 14 of 16 accumulators, while NC = 1 would
// double `cblocks` and read the whole value+index array of A twice. The A
// arrays are the dominant traffic term at low nnz/row, so that is a 2x on the
// term that matters, paid to save four kernels of device-link time.
// ---------------------------------------------------------------------------
template <typename D>
inline constexpr int kNCmax = 64 / static_cast<int>(sizeof(D));

static_assert(kNCmax<float> == 16 && kNCmax<double> == 8, "64-byte register block");
static_assert(kNCmax<Cx<float>> == 8 && kNCmax<Cx<double>> == 4, "64-byte register block");

// ---------------------------------------------------------------------------
// THE SCATTER'S ATOMIC ADD. Device scope, global address space -- the same form
// src/sycl/gemm/persistent.hh:130 already compiles into this device-link unit
// for `int`, extended to the floating-point element types.
//
// PROBED BEFORE IT WAS WRITTEN, not assumed: float, double and both complex
// pairs compile and run correctly on BOTH device images (nvidia_gpu_sm_89 and
// native_cpu) from one -fsycl-targets invocation, with 1024 concurrent
// fetch_adds per slot reading back exactly. The GPU arm lowers to
// llvm.nvvm.atomic.add.global.f.f32 / .f64 with ZERO cmpxchg, so the cost is one
// hardware reduction per nonzero-column touch and not a CAS retry loop. The FP64
// forms carry a `sycl_used_aspects` requirement of atomic64 that the FP32 forms
// do not -- see the note in spmm_native.hh.
//
// Cx<R> is a plain aggregate {R re; R im;} (device_scalar.hh:35-38) whose layout
// compatibility with std::complex is static_asserted (:62-65), so &p->re and
// &p->im are well-formed and naturally aligned and the complex case is two
// independent scalar atomics. It is NOT an atomic complex add -- the two
// components can be observed at different instants -- which is correct here
// because nothing reads C between the scatter's first and last atomic.
// ---------------------------------------------------------------------------
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

// The kernel names. THREE BODIES, NOT ONE WITH A RUNTIME MODE: the gather owns
// its output and needs no atomic, the scatter does not own its output and needs
// one, and the scale exists only because a scatter cannot fold beta into its
// accumulation. One body would allocate registers for the union of three
// unrelated inner loops.
//
// CONJUGATION AND THE transB LAYOUT, by contrast, ARE runtime: each is one
// branch on a value that is uniform across the entire launch, and making either
// a template parameter would multiply the instantiation count for a sign flip
// and an index swap.
template <typename T, int NC> class SpmmGatherKernel;
template <typename T> class SpmmScaleKernel;
template <typename T> class SpmmScatterKernel;

// ===========================================================================
// BODY 1 -- {Native, Direct}, transA == NoTrans.  THE GATHER.
//
//   C[i, c] = alpha * sum_{p in row i of A} A_val[p] * op(B)[A_ci[p], c]
//             + beta * C[i, c]
//
// ONE WORK-ITEM PER (batch item, row i of A, block of NC output columns), and
// no collective anywhere. The item owns every element it writes, so there is no
// atomic and no barrier.
//
// THE DECOMPOSITION ORDER IS ROWS-FASTEST, AND THE COLUMN BLOCK IS AN AXIS OF
// THE FLATTENED RANGE rather than a loop inside the work-item. Both choices are
// load-bearing:
//
//   * ROWS FASTEST (rc = m*cblocks; b = gid/rc; cb = rem/m; i = rem - cb*m).
//     32 consecutive work-items hold 32 consecutive ROWS of one matrix at one
//     column block, so a_ro[ro+i] and a_ro[ro+i+1] are 32 consecutive ints --
//     one 128 B line each -- and the C stores are 32 consecutive elements of one
//     column. The competing "column fastest" order fragments the C write into
//     roughly nrhs sectors per line. The argument that rows-fastest loses a
//     broadcast on the A value/index reads is WRONG: CSR packs rows back to back
//     within an item's slab (src/matrix.cc:556-565 writes at
//     row_offsets[base+r] + pos into one contiguous region), so the warp's 32
//     runs collectively consume every byte of the sectors fetched. What
//     rows-fastest actually costs is DIVERGENCE -- the warp runs max_row_len
//     iterations, not mean_row_len -- and that is a property of scalar CSR, not
//     of the ordering.
//
//   * COLUMN BLOCK AS AN AXIS. Looping column tiles inside one work-item caps
//     the parallel extent at m*batch. Making the block an axis multiplies it by
//     ceil(nrhs/NC) at identical global A traffic (the same ceil(nrhs/NC) passes
//     either way), which at a LOBPCG-shaped call (m = 4096, batch = 64,
//     nrhs = 50, NC = 16) is 786,432 items rather than 262,144 -- 3072 rather
//     than 1024 work-groups of 256 on a 128-SM part.
//
// THE ONE TERM THAT DOES NOT COALESCE is the op(B) gather itself: one sector per
// (nonzero, column) touch for sizeof(T) useful bytes. No work-item mapping fixes
// it; it follows from B's column-major layout, and cuSPARSE faces it
// identically. The lever that DOES fix it is the caller's: hand the dense block
// in transposed layout (transB = Trans) and the touch becomes
// ceil(nrhs*sizeof(T)/32) sectors per nonzero instead of nrhs of them.
//
// ZERO BYTES OF LOCAL MEMORY: no local_accessor is created in this submit.
// ===========================================================================
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
        // NO __restrict__ ON ANY OF THESE. LOBPCG passes X/P/R as slices of one
        // buffer and AX/AP/AR of another (syevx_lobpcg.cc:331-341); they are
        // element-disjoint but alias at the object level, and __restrict__ is a
        // promise about the object.
        const D* a_val = reinterpret_cast<const D*>(A.data_ptr());
        const int* a_ro = A.row_offsets().data();
        const int* a_ci = A.col_indices().data();

        // TWO STRIDES, NOT ONE, AND BOTH WIDENED TO int64_t. row_offsets is
        // indexed by offset_stride() (== rows+1) and values/col_indices by
        // matrix_stride() (src/matrix.cc:331-333, :561-564); the view returns
        // both as `int`, and b*stride overflows a plain int well inside the
        // batch sizes this library runs.
        const int64_t a_os = A.offset_stride();
        const int64_t a_ms = A.matrix_stride();

        const D* b_ptr = reinterpret_cast<const D*>(B_mat.data_ptr());
        const int64_t ldb = B_mat.ld();
        // READ FROM THE VIEW, NEVER DERIVED AS ld*cols -- that exact derivation
        // passed 232 gemv test cases before it was caught.
        const int64_t sb = B_mat.stride();

        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int64_t ldc = C.ld();
        const int64_t sc = C.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        // Launch-uniform, so these are predictable branches outside the loop and
        // never a per-element test. `alpha_zero` is what keeps A and B unread at
        // alpha == 0; `beta_zero` is what keeps C unread at beta == 0, which is
        // not an optimisation but the contract -- callers pass unzeroed
        // BumpAllocator memory as C.
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

                // FOUR SEPARATE BASES. KernelMatrixView::get computes its own in
                // plain int (matrix.hh:160-161) and wraps at a large batch times
                // a large stride; these are widened, and no body in this file
                // calls get().
                const int64_t ro = b * a_os;   // indexes a_ro ONLY
                const int64_t vb = b * a_ms;   // indexes a_val AND a_ci
                const D* Bb = b_ptr + b * sb;
                D* Cb = c_ptr + b * sc;

                // THE ONLY LEGAL BOUND. Never A.nnz(): that is the per-item
                // CAPACITY, equal to the batch MAXIMUM, and the slots above each
                // item's own count are UNINITIALISED in BOTH arrays. Row offsets
                // are ITEM-LOCAL and start at 0, so rs and re are offsets INTO
                // the item's slab -- add vb when indexing values and indices,
                // never when comparing offsets.
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
                                // op(B)[j, c]: NoTrans   -> B[j,c] = Bb[c*ldb+j]
                                //              Trans     -> B[c,j] = Bb[j*ldb+c]
                                //              ConjTrans -> conj of the Trans one
                                D bv = b_notrans
                                           ? Bb[static_cast<int64_t>(c) * ldb + j]
                                           : Bb[static_cast<int64_t>(j) * ldb + c];
                                // `if constexpr` so a real instantiation emits no
                                // branch at all, and a runtime bool inside it so
                                // complex gets ONE launch-uniform branch rather
                                // than a second set of instantiations.
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
                        // out = alpha*acc + beta*C0, built with fma_acc from an
                        // exact zero rather than a multiply and a separate add:
                        // an fma into a zero addend is the product, rounded once.
                        // beta == 0 MEANS C IS NOT READ -- see the contract note
                        // in the header; the in-tree host arm
                        // (netlib_lapack.cc:254) is the outlier here, not this.
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

// ===========================================================================
// BODY 0 -- the SCALE, launched ONLY on the transposed arm and only ever
// immediately before body 2.
//
//   C[j, c] = beta * C[j, c]      (and C[j, c] = 0 when beta == 0)
//
// A SCATTER CANNOT FOLD beta INTO ITS ACCUMULATION. No single work-item owns an
// output element there, and there is no device-wide barrier inside a SYCL
// kernel, so beta*C_old must land exactly once and it must land before any
// atomic. That is the entire reason this body exists.
//
// At beta == 0 it STORES ZERO rather than reading C, which is what makes the
// transposed arm safe against the uninitialised workspace C that every
// in-library caller passes.
//
// ZERO BYTES OF LOCAL MEMORY. Consecutive gid gives consecutive j, so both the
// read and the write are fully coalesced.
// ===========================================================================
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

// ===========================================================================
// BODY 2 -- {Native, Direct}, transA == Trans or ConjTrans.  THE SCATTER.
//
//   C[A_ci[p], c] += alpha * conj?(A_val[p]) * op(B)[i, c]
//                    for every nonzero p of stored row i
//
// ONE WORK-ITEM PER (batch item, ROW i OF THE STORED A). A row of the stored A
// is a COLUMN of op(A), which is exactly what CSR hands you -- so the transposed
// product needs no transposed storage and no expansion, only a different
// direction of travel: the item reads ONE row of op(B) and pushes into many rows
// of C.
//
// alpha IS FOLDED ONCE PER (row, column block), into ab[], not once per nonzero.
// At nnz/row = 16 that is 16x fewer multiplies by alpha, and it also means the
// inner atomic carries a single dev_mul.
//
// THE UNSIGNED RANGE GUARD ON j IS NOT OPTIONAL. In the gather a bad column
// index is an out-of-range READ -- a wrong answer. Here it is an out-of-range
// ATOMIC WRITE, i.e. heap corruption, and the padding above each item's nnz is
// genuinely uninitialised. One predicate per nonzero next to an atomic is free.
//
// NCS IS FIXED AT 4, deliberately not a template parameter. This arm has ZERO
// in-tree C++ callers today (netlib refuses every transpose,
// netlib_lapack.cc:249, and no in-library caller asks for one), so it is not a
// hot path and does not deserve four more device kernels.
//
// ZERO BYTES OF LOCAL MEMORY. Registers: NCS scalars of staged op(B).
// ===========================================================================
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

    // ALPHA == 0 IS A HOST-SIDE SKIP, not an in-kernel early return: body 0 has
    // already written C = beta*C, this body's only contribution is the alpha
    // term, and there is nothing left for it to do. C is NOT left untouched --
    // that is the difference from reference ?GEMV's quick return, and copying
    // gemv's version here would be a route-dependent wrong answer.
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
                            // op(B)[i, c]: NoTrans -> B[i,c] = Bb[c*ldb+i]
                            //              Trans   -> B[c,i] = Bb[i*ldb+c]
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

// ---------------------------------------------------------------------------
// The public entry. ONE ROUTE, THREE BODIES, chosen here on transA -- gemv's own
// precedent, where {Native, Direct} names two kernels and the pick lives in the
// launcher (gemv_native.cc:1249-1289) rather than in the facade or the route
// table. Body selection is a decomposition, not an algorithm.
// ---------------------------------------------------------------------------
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

    // THE QUICK RETURN, ON THE HOST, BEFORE ANY SUBMIT. There is nothing to
    // write, so C is left completely untouched and no zero-size nd_range is ever
    // enqueued.
    //
    // NOTE WHAT IS **NOT** HERE. Reference ?GEMV also quick-returns at
    // (alpha == 0 && beta == 1), leaving y untouched, and gemv_native.cc:486-489
    // matches it. Copying that here would be wrong: the answer at alpha == 0 is
    // C = beta*C, and skipping it would be a ROUTE-DEPENDENT wrong answer,
    // correct under cuSPARSE and wrong under the native kernel on the same call.
    // A zero REDUCTION extent is likewise not a quick return -- the gather's
    // empty row loop and the scale kernel both produce C = beta*C, which is the
    // right answer.
    if (out_rows == 0 || nrhs == 0 || batch <= 0) return ctx.get_event();

    if (a_notrans) {
        // NC is a host-side pick over the two instantiated widths. nrhs <= 2 is
        // the lanczos shape; everything else gets the full 64-byte block.
        if (nrhs <= 2) {
            return spmm_gather<T, 2>(ctx, A, B_mat, C, alpha, beta, transB);
        }
        return spmm_gather<T, kNCmax<D>>(ctx, A, B_mat, C, alpha, beta, transB);
    }

    // TWO SUBMITS, ORDERED. The scale must be complete before the first atomic
    // lands. Queue's default is in_order = true
    // (include/batchlas/util/sycl-device-queue.hh:254-255, in_order() at :330),
    // which costs nothing here; a caller that built an out-of-order queue pays
    // one host wait. Idiom from src/extensions/getrf_blocked.cc:410.
    (void)spmm_scale<T>(ctx, C, beta, out_rows, batch);
    if (!ctx.in_order()) ctx.wait();
    return spmm_scatter<T>(ctx, A, B_mat, C, alpha, out_rows,
                           transA == Transpose::ConjTrans, transB);
}

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAGS, as full explicit specialisations in the same TU as the
// kernels (route_trsm.hh:62-96's rule). A build that drops this file drops these
// definitions too, so the route table can never advertise a kernel that is not
// linked -- the failure mode this campaign records as "LINKED is not REACHABLE",
// stated the other way round.
//
// ALL FOUR TYPES, BOTH TIERS. The instantiation macro below emits every body for
// every scalar type, so there is no type for which one of these is a lie.
//
// NOTE WHAT THEY DO NOT SAY. These are statements about the BUILD, not about the
// device. There is deliberately no is_gpu notion and no sub-group notion
// anywhere in this file: SPMM_ALL(Backend::NETLIB) is instantiated in a
// vendor-free build, its spmm symbol exists and throws today, and it runs on a
// native_cpu queue -- a device-gated capability flag would leave it throwing and
// move the vendor-free burn-down by exactly zero.
// ---------------------------------------------------------------------------
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
