// The Side::Left sub-group TRSM kernel. It lives in its own TU because it needs
// the raised pragma-unroll threshold (src/CMakeLists.txt): without it the N x QC
// broadcast loop is not unrolled and x[] / nL[] go to a stack frame (520 B at
// cfloat N=32), which V1's kernels in trsm_native.cc do not need.

#include "trsm_native.hh"
#include "trsm_canonical.hh"
#include <batchlas/error.hh>

#include "../queue.hh"
#include "device_scalar.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace batchlas::sycl_trsm {

namespace {

template <typename T, int N, int QC>
class TrsmSgLeftKernel;

template <typename D>
inline D sg_bcast(const sycl::sub_group& sg, D v, int src) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{sycl::select_from_group(sg, v.re, src), sycl::select_from_group(sg, v.im, src)};
    } else {
        return sycl::select_from_group(sg, v, src);
    }
}

template <typename D>
inline D dev_neg(D v) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{-v.re, -v.im};
    } else {
        return -v;
    }
}

// Row bucket: N lanes per matrix, 32/N matrices per sub-group; 0 means no bucket.
inline int sg_left_bucket(int n) {
    if (n <= 4) return 4;
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

}  // namespace

// Side::Left CTA solve, one lane per canonical row. Lane r keeps row r of Lc in
// registers and QC right-hand sides; step s broadcasts x[s] from lane s. No SLM,
// no barrier, and a lane exists only for a real (row, rhs-chunk) pair, which is
// what V1's one-thread-per-rhs geometry lacked at small q.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
template <typename T, int N, int QC>
Event trsm_native_sg_left(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha, Uplo uplo, Transpose transA, Diag diag) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(N >= 4 && N <= 32 && 32 % N == 0, "a matrix must tile the sub-group");
    constexpr int kSg = 32;
    constexpr int kMpw = kSg / N;
    constexpr int kSgPerWg = 4;
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
        h.parallel_for<TrsmSgLeftKernel<T, N, QC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(nwg) * kSgPerWg * kSg),
                              sycl::range<1>(kSgPerWg * kSg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int64_t gsg = static_cast<int64_t>(it.get_group_linear_id()) * kSgPerWg +
                                    static_cast<int64_t>(sg.get_group_linear_id());
                // Uniform per sub-group, so no lane is left alone at a broadcast.
                if (gsg >= nsg) return;
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int mat = lane / N;
                const int r = lane % N;
                const int chunk = static_cast<int>(gsg % chunks);
                const int64_t b = (gsg / chunks) * kMpw + mat;
                const bool okb = b < bs;
                const bool okr = okb && r < n;
                const D* Ab = a_ptr + (okb ? b : 0) * stride_a;
                D* Bb = b_ptr + (okb ? b : 0) * stride_b;
                const int64_t rs = fwd ? r : (n - 1 - r);

                // nL[t] = -Lc(r,t) for t < r, else 0; pad rows are identity rows.
                D nL[N];
#pragma unroll
                for (int t = 0; t < N; ++t) {
                    D v{};
                    if (okr && t < r) {
                        const int64_t rt = fwd ? t : (n - 1 - t);
                        v = do_trans ? Ab[rt + rs * lda] : Ab[rs + rt * lda];
                        if (do_conj) v = sycl_device::dev_conj(v);
                        v = dev_neg(v);
                    }
                    nL[t] = v;
                }
                D dd = sycl_device::dev_one<D>();
                if (okr && !unit) {
                    dd = Ab[rs + rs * lda];
                    if (do_conj) dd = sycl_device::dev_conj(dd);
                }
                D rinv = sycl_device::dev_recip(dd);
                // V1's fallback, per matrix: any non-finite reciprocal in this
                // matrix's N lanes switches all of them to division.
                int bad = sycl_device::dev_isfinite(rinv) ? 0 : 1;
#pragma unroll
                for (int off = N / 2; off > 0; off /= 2) {
                    bad |= sycl::permute_group_by_xor(sg, bad, off);
                }
                const bool divide = bad != 0;

                const int c0 = chunk * QC;
                D x[QC];
#pragma unroll
                for (int j = 0; j < QC; ++j) {
                    D v{};
                    if (okr && c0 + j < q) {
                        v = sycl_device::dev_mul(alpha_d, Bb[rs + (c0 + j) * ldb]);
                    }
                    x[j] = v;
                }

                const int base = mat * N;
                // The division path sits behind a sub-group-uniform branch: as a
                // per-lane select it kept QC divisions live at every step (2x at q=64).
                const bool any_divide = sycl::any_of_group(sg, divide);
#pragma unroll kUnrollS
                for (int s = 0; s < N; ++s) {
                    // Rows n..N-1 are identity padding. A break stops full unrolling.
                    if (s >= n) {
                        if constexpr (kRolled) break; else continue;
                    }
                    if (r == s && !unit) {
                        if (any_divide) [[unlikely]] {
#pragma unroll
                            for (int j = 0; j < QC; ++j) {
                                x[j] = divide ? sycl_device::dev_div(x[j], dd)
                                              : sycl_device::dev_mul(x[j], rinv);
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < QC; ++j) x[j] = sycl_device::dev_mul(x[j], rinv);
                        }
                    }
                    const D l = nL[s];
#pragma unroll
                    for (int j = 0; j < QC; ++j) {
                        const D xs = sg_bcast(sg, x[j], base + s);
                        sycl_device::fma_acc(x[j], l, xs);
                    }
                }

#pragma unroll
                for (int j = 0; j < QC; ++j) {
                    if (okr && c0 + j < q) Bb[rs + (c0 + j) * ldb] = x[j];
                }
            });
    });
    return ctx.get_event();
}

// QC tracks q up to 16: every broadcast is paid for all QC slots.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
template <typename T, int N>
Event trsm_native_sg_left_qc(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             T alpha, Uplo uplo, Transpose transA, Diag diag) {
    const int q = static_cast<int>(B.cols());
    int qc = q <= 4 ? 4 : (q <= 8 ? 8 : 16);
    if (const char* e = std::getenv("BATCHLAS_DEV_TRSM_SG_QC")) qc = std::atoi(e);
    if (qc == 4) return trsm_native_sg_left<T, N, 4>(ctx, A, B, alpha, uplo, transA, diag);
    // complex<double> at QC=16 needs 255 registers and a stack frame.
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return trsm_native_sg_left<T, N, 8>(ctx, A, B, alpha, uplo, transA, diag);
    } else {
        if (qc == 8) return trsm_native_sg_left<T, N, 8>(ctx, A, B, alpha, uplo, transA, diag);
        return trsm_native_sg_left<T, N, 16>(ctx, A, B, alpha, uplo, transA, diag);
    }
}

template <typename T>
Event trsm_native_sg_left_dispatch(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   T alpha, Uplo uplo, Transpose transA, Diag diag) {
    switch (sg_left_bucket(static_cast<int>(A.rows()))) {
        case 4:  return trsm_native_sg_left_qc<T, 4>(ctx, A, B, alpha, uplo, transA, diag);
        case 8:  return trsm_native_sg_left_qc<T, 8>(ctx, A, B, alpha, uplo, transA, diag);
        case 16: return trsm_native_sg_left_qc<T, 16>(ctx, A, B, alpha, uplo, transA, diag);
        case 32: return trsm_native_sg_left_qc<T, 32>(ctx, A, B, alpha, uplo, transA, diag);
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
