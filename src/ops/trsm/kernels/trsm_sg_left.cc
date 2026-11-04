// SPDX-License-Identifier: MIT
// The Side::Left sub-group TRSM kernel. It lives in its own TU because it needs
// the raised pragma-unroll threshold (src/CMakeLists.txt): without it the N x QC
// broadcast loop is not unrolled and x[] / nL[] go to a stack frame (520 B at
// cfloat N=32), which V1's kernels in trsm_native.cc do not need.

#include <src/ops/trsm/trsm_native.hh>
#include <src/ops/trsm/trsm_canonical.hh>
#include <src/ops/trsm/trsm_v2.hh>
#include <batchlas/error.hh>

#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/util/batch_mask.hh>
#include <src/kernels/common/dla_v2_scalar.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>

#include <batchlas/device.hh>

namespace batchlas::sycl_trsm {

namespace v2 = batchlas::device;

namespace {

template <typename T, int N, int QC, bool Masked>
class TrsmSgLeftKernel;

// Row bucket: N lanes per matrix, 32/N matrices per sub-group; 0 means no bucket.
inline int sg_left_bucket(int n) {
    if (n <= 4) return 4;
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}
static_assert(kTrsmSgLeftMaxN == 32, "sg_left_bucket's largest bucket is trsm's can_run ceiling");

}  // namespace

// Side::Left CTA solve, one lane per canonical row. Lane r keeps row r of Lc in
// registers and QC right-hand sides; step s broadcasts x[s] from lane s. No SLM,
// no barrier, and a lane exists only for a real (row, rhs-chunk) pair, which is
// what V1's one-thread-per-rhs geometry lacked at small q.
// Masked: several items share a sub-group (I2), so an inactive item folds into okb like a tail slot
// and is never returned; a template so the unmasked instance stays the kernel as it was.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
template <typename T, int N, int QC, bool Masked>
Event trsm_native_sg_left(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha, Uplo uplo, Transpose transA, Diag diag, const std::uint8_t* mask) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(N >= 4 && N <= 32 && 32 % N == 0, "a matrix must tile the sub-group");
    constexpr int kSg = 32;
    constexpr int kMpw = kSg / N;
    constexpr int kSgPerWg = 4;
    static_assert(kSgPerWg * kSg == kTrsmSgLeftWgSize, "trsm's can_run gates on kTrsmSgLeftWgSize");
    // Rolled, nL[] lives in local memory and is read once per step: fewer registers,
    // which wins where the unrolled form's occupancy is lowest.
    // evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
    constexpr bool kRolled = (sycl_device::dev_is_complex_v<D> && N >= 16) || (N == 32 && QC == 16);
    constexpr int kUnrollS = kRolled ? 1 : N;

    const Canonical can = canonicalise(Side::Left, uplo, transA, diag);
    const int n = static_cast<int>(A.rows());
    const int q = static_cast<int>(B.cols());
    const int bs = static_cast<int>(A.batch_size());
    const int chunks = (q + QC - 1) / QC;
    const int64_t nsg = (static_cast<int64_t>(bs) + kMpw - 1) / kMpw * chunks;
    const int64_t nwg = (nsg + kSgPerWg - 1) / kSgPerWg;

    const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
    D* b_ptr = reinterpret_cast<D*>(B.data_ptr());
    const int64_t lda = A.ld(), ldb = B.ld();
    const int64_t stride_a = A.stride(), stride_b = B.stride();
    const bool do_trans = can.do_trans, do_conj = can.do_conj;
    const bool fwd = can.fwd, unit = can.unit;
    D alpha_d;
    __builtin_memcpy(&alpha_d, &alpha, sizeof(D));

    ctx->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<TrsmSgLeftKernel<T, N, QC, Masked>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(nwg) * kSgPerWg * kSg),
                              sycl::range<1>(kSgPerWg * kSg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                using Ar = v2::arith_t<v2::arith::Fma, D>;
                const auto sg = it.get_sub_group();
                const int64_t gsg = static_cast<int64_t>(it.get_group_linear_id()) * kSgPerWg +
                                    static_cast<int64_t>(sg.get_group_linear_id());
                // Uniform per sub-group, so no lane is left alone at a broadcast.
                if (gsg >= nsg) return;
                const v2::Partition<N> ex{sg};
                const int mat = static_cast<int>(sg.get_local_linear_id()) / N;
                // lane % N, not ex.lid() (lane & (N-1)): the AND form shifts ptxas' register
                // allocation in the unrolled N=16 kernels by 3-4% of the instructions.
                const int r = static_cast<int>(sg.get_local_linear_id()) % N;
                const int chunk = static_cast<int>(gsg % chunks);
                const int64_t b = (gsg / chunks) * kMpw + mat;
                bool okb = b < bs;
                if constexpr (Masked) okb = okb && mask[b] != 0;
                const bool okr = okb && r < n;
                const dla::CanonTri<const D, true, int64_t> L{a_ptr + (okb ? b : 0) * stride_a, lda, n,
                                                              fwd, do_trans, do_conj};
                const dla::FlipRows<D, int64_t> Bc{b_ptr + (okb ? b : 0) * stride_b, ldb, n, q, fwd};

                // Lane r: the AXPY multipliers -Lc(r, 0:r); pad rows are identity rows.
                // Built in place (load_row's Frag& out-param): a returned Frag stays live whole
                // until its copy-out in the rolled kernels (REG 44->90, cfloat N=32).
                const auto neg = [](const D& v) BATCHLAS_DLA_BODY { return v2::Arith<D>::neg(v); };
                v2::Frag<D, N> nL;
                v2::load_row(nL, dla::MapCols{L, 0, neg}, r, okr ? r : 0);
                D dd = Ar::one();
                if (okr && !unit) dd = L.diag(r);   // not L(r, r): see CanonTri::diag
                // V1's fallback, per matrix: any non-finite reciprocal in this
                // matrix's N lanes switches all of them to division.
                const auto dg = dla::guarded_recip<Ar>(ex, dd);

                const int c0 = chunk * QC;
                const int live_c = okr ? q - c0 : 0;   // columns c0.. of B this lane loads and stores
                const auto scale = [=](const D& v) BATCHLAS_DLA_BODY { return Ar::mul(alpha_d, v); };
                v2::Frag<D, QC> x;
                v2::load_row(x, dla::MapCols{Bc, c0, scale}, r, live_c);
                // The division path sits behind a sub-group-uniform branch: as a
                // per-lane select it kept QC divisions live at every step (2x at q=64).
                dla::trsm_lower_rowfrag<kUnrollS, Ar>(ex, nL, dg, n, unit, x);
                v2::store_row(x, dla::ColsFrom{Bc, c0}, r, live_c);
            });
    });
    return ctx.get_event();
}

// QC tracks q up to 16: every broadcast is paid for all QC slots.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
template <typename T, int N, bool Masked>
Event trsm_native_sg_left_qc(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             T alpha, Uplo uplo, Transpose transA, Diag diag, const std::uint8_t* mask) {
    const int q = static_cast<int>(B.cols());
    const int qc = q <= 4 ? 4 : (q <= 8 ? 8 : 16);
    if (qc == 4) return trsm_native_sg_left<T, N, 4, Masked>(ctx, A, B, alpha, uplo, transA, diag, mask);
    // complex<double> at QC=16 needs 255 registers and a stack frame.
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return trsm_native_sg_left<T, N, 8, Masked>(ctx, A, B, alpha, uplo, transA, diag, mask);
    } else {
        if (qc == 8) return trsm_native_sg_left<T, N, 8, Masked>(ctx, A, B, alpha, uplo, transA, diag, mask);
        return trsm_native_sg_left<T, N, 16, Masked>(ctx, A, B, alpha, uplo, transA, diag, mask);
    }
}

template <typename T, int N>
Event trsm_native_sg_left_leg(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              T alpha, Uplo uplo, Transpose transA, Diag diag) {
    const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("trsm", A, B));
    if (mask) return trsm_native_sg_left_qc<T, N, true>(ctx, A, B, alpha, uplo, transA, diag, mask);
    return trsm_native_sg_left_qc<T, N, false>(ctx, A, B, alpha, uplo, transA, diag, nullptr);
}

template <typename T>
Event trsm_native_sg_left_dispatch(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   T alpha, Uplo uplo, Transpose transA, Diag diag) {
    switch (sg_left_bucket(static_cast<int>(A.rows()))) {
        case 4:  return trsm_native_sg_left_leg<T, 4>(ctx, A, B, alpha, uplo, transA, diag);
        case 8:  return trsm_native_sg_left_leg<T, 8>(ctx, A, B, alpha, uplo, transA, diag);
        case 16: return trsm_native_sg_left_leg<T, 16>(ctx, A, B, alpha, uplo, transA, diag);
        case 32: return trsm_native_sg_left_leg<T, 32>(ctx, A, B, alpha, uplo, transA, diag);
        default: break;
    }
    throw batchlas::unsupported("BatchLAS: trsm_native_sg_left called with triangular order " +
                                std::to_string(A.rows()) + " > 32, the sub-group width.");
}

#define BATCHLAS_TRSM_SG_INST(T)                                                          \
    template Event trsm_native_sg_left_dispatch<T>(                                       \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                \
        const MatrixView<T, MatrixFormat::Dense>&, T, Uplo, Transpose, Diag);
BATCHLAS_TRSM_SG_INST(float)
BATCHLAS_TRSM_SG_INST(double)
BATCHLAS_TRSM_SG_INST(std::complex<float>)
BATCHLAS_TRSM_SG_INST(std::complex<double>)
#undef BATCHLAS_TRSM_SG_INST

}  // namespace batchlas::sycl_trsm
