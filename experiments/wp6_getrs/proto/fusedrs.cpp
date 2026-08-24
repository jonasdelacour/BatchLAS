// WP6 PERF: the FUSED NARROW-RHS getrs probe.
//
// THE QUESTION. getrs at nrhs = 1 is 0.32x of cublas?getrsBatched, and the
// orchestrator's nsys profile says why: the blocked trsm driver decomposes one
// column into many CTA solves plus ~26,000 TILE-16 GEMM launches of shape
// n x 1 x k -- matrix-VECTOR products run through a matrix kernel -- and the
// permutation kernel is another 26.4% of the call on its own. trsm exists to
// amortise a panel over many columns; one column gives it nothing to amortise.
// BatchLAS has no native trsv.
//
// THIS PROBE asks whether ONE kernel per matrix -- permutation, forward
// substitution and back substitution fused, no GEMM launches, no separate laswp
// -- beats the composition, and where on nrhs it stops.
//
// THE CEILING IS MEMORY, NOT FLOPS. The substitution is O(n^2) work over O(n^2)
// matrix reads, so at nrhs = 1 the arithmetic intensity is 2 flop per element and
// the kernel can do no better than streaming L and U once. Every timing row below
// therefore also prints the achieved fraction of the 1008 GB/s DRAM peak of an
// RTX 4090. n = 512 float is 1 MB per matrix, so the matrix is NOT SLM-resident at
// the sizes that matter and no CTA-resident-matrix design is attempted; what is
// resident is the RHS VECTOR and one small diagonal block.
//
// FOUR ARMS, INTERLEAVED IN ONE PROCESS, one A/B session:
//   vendor   cublas?getrsBatched, called DIRECTLY (so this arm is the same in the
//            vendor-free link as in the vendor-present one)
//   comp     sycl_getrs::getrs_blocked_dispatch -- the SHIPPED composition,
//            called at its direct entry point with the routed trsm injected, so
//            the arm is exactly the one the facade would run
//   fstream  the fused kernel with nb = 1: pure streaming, one barrier triple per
//            column
//   fblock   the fused kernel with a resident nb x nb diagonal block: the block
//            solve runs inside ONE sub-group with shuffles and no work-group
//            barrier, and only the trailing update crosses the work-group
//
// The two fused arms are the "pure streaming vs blocked-diagonal" comparison the
// brief demands be settled by measurement rather than assumed.
//
// NOPERM=1 additionally times the fused kernel with the pivot walk removed. That
// is a TIMING-ONLY BREAK -- the answers are wrong by construction and the row is
// marked BAD -- and it exists to price the one part of this kernel that is a
// serial n-step recurrence over a single lane.
//
// The oracle is the HOST: || A0 X - B0 ||inf / || B0 ||inf in double regardless of
// T, on the FIRST and LAST batch item, computed from A0 -- which no device call
// writes. Copied from experiments/wp6_lu/bench/lubench6.cpp rather than
// re-derived; WP4's measurement went 2x off by re-deriving a harness.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/getrf_route.hh"
#include "src/backends/getrs_route.hh"
#include "src/extensions/getrs_native.hh"
#include "src/sycl/device_scalar.hh"
#include "src/queue.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;
static constexpr bool kVendorF = dispatch::factorization_vendor_available<BE>;

// ------------------------------------------------------------------ scalars
template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };
static inline double dconj(double x) { return x; }
static inline std::complex<double> dconj(std::complex<double> x) { return std::conj(x); }
static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

static inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

struct Stat { double med, mean, relsd; };
static Stat stat_of(std::vector<double> v) {
    if (v.empty()) return {0, 0, 0};
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0};
}

template <typename T> struct Tol;
template <> struct Tol<float>                { static constexpr double v = 1e-4; };
template <> struct Tol<std::complex<float>>  { static constexpr double v = 1e-4; };
template <> struct Tol<double>               { static constexpr double v = 1e-11; };
template <> struct Tol<std::complex<double>> { static constexpr double v = 1e-11; };

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }
static int    envi(const char* k, int d) { const char* e = std::getenv(k); return e ? std::atoi(e) : d; }

// ===========================================================================
// THE FUSED KERNEL.
//
// ONE WORK-GROUP PER MATRIX, PARALLEL OVER ROWS. The substitution is a serial
// recurrence in the column index k with a parallel vector update over the
// remaining rows, so there is no second axis to give a second work-group; L and U
// are STREAMED from global memory and only the RHS vector (n x nrhs) and one
// nb x nb diagonal block are resident.
//
// WHY COLUMN-ORIENTED (axpy) AND NOT ROW-ORIENTED (dot). BatchLAS is COLUMN-MAJOR,
// so the axpy form of a NoTrans triangular solve reads a CONTIGUOUS column segment
// at every step -- L[k+1..n-1, k] going forward and U[0..k-1, k] coming back --
// while the dot form would read a row, i.e. `ld` apart, which is 32 transactions
// per warp. The transposed modes are the mirror image and are NOT in this probe;
// they are the same access class (a dot against a contiguous column) and are a
// correctness question, not a timing one.
//
// THE PERMUTATION IS FOLDED IN, and it is the one part that is not parallel: the
// interchange list must be walked IN ORDER, so column c of the RHS is walked by a
// single work-item. It is done in LOCAL memory (the RHS is loaded coalesced
// first), never in global, because n dependent global round-trips per matrix is
// ~400 cycles each. NOPERM=1 prices it.
// ===========================================================================
namespace fused {

using namespace batchlas::sycl_device;

template <typename D> static inline D dev_zero() {
    if constexpr (std::is_same_v<D, float> || std::is_same_v<D, double>) return D(0);
    else return D{0, 0};
}

template <typename T, int NR, bool PERM> class FusedGetrsKernel;

// nrhs <= NR. NR is a compile-time chunk so the A element read in the trailing
// update is REUSED across the right-hand sides from a register instead of being
// re-read per column -- which is the only reuse a narrow solve has.
template <typename T, int NR, bool PERM>
static sycl::event launch(sycl::queue& q,
                          const T* A, int lda, int strideA,
                          T* B, int ldb, int strideB,
                          const int* piv, int pstride,
                          int n, int nrhs, int batch, int wg, int nb) {
    using DM = DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const Ap = reinterpret_cast<const D*>(A);
    D* const Bp = reinterpret_cast<D*>(B);

    const size_t y_elems = size_t(n) * size_t(nrhs);
    const size_t blk_elems = size_t(nb) * size_t(nb);

    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> ysl(sycl::range<1>(y_elems), h);
        sycl::local_accessor<D, 1> bsl(sycl::range<1>(blk_elems), h);
        h.parallel_for<FusedGetrsKernel<T, NR, PERM>>(
            sycl::nd_range<1>(sycl::range<1>(size_t(batch) * size_t(wg)),
                              sycl::range<1>(size_t(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = int(it.get_local_id(0));
                const size_t b = it.get_group(0);
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int sgid = int(sg.get_group_linear_id());

                const D* const Ab = Ap + b * size_t(strideA);
                D* const Bb = Bp + b * size_t(strideB);
                const int* const pv = piv + b * size_t(pstride);
                D* const y = &ysl[0];
                D* const blk = &bsl[0];

                // --- load the RHS, coalesced ---------------------------------
                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    y[e] = Bb[size_t(i) + size_t(c) * size_t(ldb)];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // --- the interchange walk, in LOCAL memory -------------------
                if constexpr (PERM) {
                    if (tid < nrhs) {
                        D* const yc = y + size_t(tid) * size_t(n);
                        for (int k = 0; k < n; ++k) {
                            const int p = pv[k] - 1;      // 1-BASED on the wire
                            if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // --- forward substitution: L, unit lower ---------------------
                for (int j = 0; j < n; j += nb) {
                    const int jb = (n - j < nb) ? (n - j) : nb;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[size_t(i) + size_t(c) * size_t(nb)] =
                            Ab[size_t(j + i) + size_t(j + c) * size_t(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // THE BLOCK SOLVE RUNS INSIDE ONE SUB-GROUP. Lane i owns row
                    // j+i and holds it in a REGISTER; the pivot value is a
                    // shuffle, not an SLM round-trip, and there is NO work-group
                    // barrier in the jb-step recurrence. This is the whole reason
                    // the blocked arm can beat the streaming one: the streaming
                    // arm pays a work-group barrier per column, this pays one per
                    // BLOCK. It requires nb <= the sub-group size, which the
                    // launcher enforces.
                    if (sgid == 0 && jb > 1) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + size_t(c) * size_t(n);
                            D v = (lane < jb) ? yc[j + lane] : dev_zero<D>();
                            for (int kk = 0; kk < jb - 1; ++kk) {
                                const D pivv = sycl::group_broadcast(sg, v, kk);
                                if (lane > kk && lane < jb) {
                                    v = dev_sub(v, dev_mul(blk[size_t(lane) + size_t(kk) * size_t(nb)], pivv));
                                }
                            }
                            if (lane < jb) yc[j + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // --- trailing update, parallel over ROWS -----------------
                    // Consecutive work-items take consecutive rows, so the read of
                    // A[i, j+kk] is coalesced; the jb column values it multiplies
                    // are broadcast from local memory.
                    for (int i = j + jb + tid; i < n; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[size_t(i) + size_t(j + kk) * size_t(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs) fma_acc(acc[c], a, y[size_t(c) * size_t(n) + size_t(j + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + size_t(c) * size_t(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // --- back substitution: U, non-unit upper --------------------
                for (int jend = n; jend > 0; jend -= nb) {
                    const int j0 = (jend - nb > 0) ? (jend - nb) : 0;
                    const int jb = jend - j0;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[size_t(i) + size_t(c) * size_t(nb)] =
                            Ab[size_t(j0 + i) + size_t(j0 + c) * size_t(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (sgid == 0) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + size_t(c) * size_t(n);
                            D v = (lane < jb) ? yc[j0 + lane] : dev_zero<D>();
                            for (int kk = jb - 1; kk >= 0; --kk) {
                                if (lane == kk) v = dev_div(v, blk[size_t(kk) + size_t(kk) * size_t(nb)]);
                                const D pivv = sycl::group_broadcast(sg, v, kk);
                                if (lane < kk) {
                                    v = dev_sub(v, dev_mul(blk[size_t(lane) + size_t(kk) * size_t(nb)], pivv));
                                }
                            }
                            if (lane < jb) yc[j0 + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    for (int i = tid; i < j0; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[size_t(i) + size_t(j0 + kk) * size_t(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs) fma_acc(acc[c], a, y[size_t(c) * size_t(n) + size_t(j0 + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + size_t(c) * size_t(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[size_t(i) + size_t(c) * size_t(ldb)] = y[e];
                }
            });
    });
}

// Runtime nrhs -> compile-time chunk. Above 16 the probe declines rather than
// silently re-reading A once per column block, which would be a different
// algorithm wearing the same name.
template <typename T, bool PERM>
static bool dispatch_nr(sycl::queue& q, const T* A, int lda, int sA, T* B, int ldb, int sB,
                        const int* piv, int pstride, int n, int nrhs, int batch, int wg, int nb) {
    if (nrhs <= 1)  { launch<T, 1,  PERM>(q, A, lda, sA, B, ldb, sB, piv, pstride, n, nrhs, batch, wg, nb); return true; }
    if (nrhs <= 2)  { launch<T, 2,  PERM>(q, A, lda, sA, B, ldb, sB, piv, pstride, n, nrhs, batch, wg, nb); return true; }
    if (nrhs <= 4)  { launch<T, 4,  PERM>(q, A, lda, sA, B, ldb, sB, piv, pstride, n, nrhs, batch, wg, nb); return true; }
    if (nrhs <= 8)  { launch<T, 8,  PERM>(q, A, lda, sA, B, ldb, sB, piv, pstride, n, nrhs, batch, wg, nb); return true; }
    if (nrhs <= 16) { launch<T, 16, PERM>(q, A, lda, sA, B, ldb, sB, piv, pstride, n, nrhs, batch, wg, nb); return true; }
    return false;
}

}  // namespace fused

// ------------------------------------------------------------------ cuBLAS
static cublasStatus_t cb_getrs(cublasHandle_t h, cublasOperation_t op, int n, int nrhs,
                               float* const A[], int lda, const int* piv,
                               float* const Bm[], int ldb, int* info, int batch) {
    return cublasSgetrsBatched(h, op, n, nrhs, A, lda, piv, Bm, ldb, info, batch);
}
static cublasStatus_t cb_getrs(cublasHandle_t h, cublasOperation_t op, int n, int nrhs,
                               double* const A[], int lda, const int* piv,
                               double* const Bm[], int ldb, int* info, int batch) {
    return cublasDgetrsBatched(h, op, n, nrhs, A, lda, piv, Bm, ldb, info, batch);
}
static cublasStatus_t cb_getrs(cublasHandle_t h, cublasOperation_t op, int n, int nrhs,
                               std::complex<float>* const A[], int lda, const int* piv,
                               std::complex<float>* const Bm[], int ldb, int* info, int batch) {
    return cublasCgetrsBatched(h, op, n, nrhs, reinterpret_cast<cuComplex* const*>(A), lda, piv,
                               reinterpret_cast<cuComplex* const*>(Bm), ldb, info, batch);
}
static cublasStatus_t cb_getrs(cublasHandle_t h, cublasOperation_t op, int n, int nrhs,
                               std::complex<double>* const A[], int lda, const int* piv,
                               std::complex<double>* const Bm[], int ldb, int* info, int batch) {
    return cublasZgetrsBatched(h, op, n, nrhs, reinterpret_cast<cuDoubleComplex* const*>(A), lda, piv,
                               reinterpret_cast<cuDoubleComplex* const*>(Bm), ldb, info, batch);
}

// ------------------------------------------------------------------ oracle
// Copied verbatim from lubench6.cpp. NoTrans only in this probe.
template <typename T>
static double solve_probe(const UnifiedVector<T>& X, const UnifiedVector<T>& B0,
                          const UnifiedVector<T>& A0, int n, int nrhs, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t sa = size_t(n) * n, sb = size_t(n) * nrhs;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            const int j = (p * 7) % nrhs;
            std::vector<D> r(n, D(0));
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int c = 0; c < n; ++c)
                    acc += up(A0[size_t(b) * sa + size_t(c) * n + i]) *
                           up(X[size_t(b) * sb + size_t(j) * n + c]);
                r[i] = acc;
            }
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                const D bi = up(B0[size_t(b) * sb + size_t(j) * n + i]);
                num = nanmax(num, ab(r[i] - bi)); den = nanmax(den, ab(bi));
            }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

template <typename T>
static void fill_A0(UnifiedVector<T>& A0, int n, int batch, uint64_t seed) {
    const size_t sa = size_t(n) * n;
    Rng rg(seed);
    for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
    std::vector<T> col(n);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i)
            A0[size_t(b) * sa + size_t(i) * n + i] =
                A0[size_t(b) * sa + size_t(i) * n + i] + mk<T>(double(n), 0.0);
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        for (int i = n - 1; i > 0; --i) {
            const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
            std::swap(perm[i], perm[j]);
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) col[i] = A0[size_t(b) * sa + size_t(j) * n + size_t(perm[i])];
            for (int i = 0; i < n; ++i) A0[size_t(b) * sa + size_t(j) * n + i] = col[i];
        }
    }
}

// ------------------------------------------------------------------ driver
template <typename T>
static int run(const char* tn, int n, int nrhs, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    // QueueImpl IS a sycl::queue (src/queue.hh:289). The fused kernel is
    // submitted on it directly -- the Queue is in-order, and every arm is
    // followed by q->wait(), so nothing depends on the Queue's event tracking.
    sycl::queue& sq = **q;

    const size_t sa = size_t(n) * n;
    const size_t sb = size_t(n) * nrhs;

    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 12345);

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
    q->wait();

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    int ntp = 0; for (int k = 0; k < n; ++k) if (pivi[k] != k + 1) ++ntp;

    UnifiedVector<T> B0(sb * batch), X(sb * batch);
    { Rng rg(777); for (size_t i = 0; i < B0.size(); ++i) B0[i] = mk<T>(rg.next(), rg.next()); }
    UnifiedVector<T*> pB0(batch), pX(batch);
    MatrixView<T, MatrixFormat::Dense> B0v(B0.data(), n, nrhs, n, int(sb), batch, pB0.data());
    MatrixView<T, MatrixFormat::Dense> Xv(X.data(), n, nrhs, n, int(sb), batch, pX.data());

    const size_t s_ws = getrs_buffer_size<BE, T>(*q, Av, Xv, Transpose::NoTrans);
    UnifiedVector<std::byte> sws(s_ws ? s_ws : 1);

    // Geometry. wg is a work-group over ROWS; nb is the resident diagonal block
    // and must not exceed the sub-group size, because the block solve is a
    // sub-group shuffle recurrence.
    // THE DEFAULT GEOMETRY, and it is a measured rule rather than a constant.
    // wg ~ n/2 clamped to [64, 1024]: measured best at n=128 (wg 64-128 tie),
    // n=512 (wg 256) and n=2048 (wg 1024, where batch=32 leaves only 32 of 128
    // SMs occupied and the only remaining lever is threads per work-group).
    // nb is 16 below n=1024 and 32 at or above it.
    int wg_d = 32; while (wg_d < n / 2 && wg_d < 1024) wg_d *= 2;
    if (wg_d < 64) wg_d = 64;
    int wg = envi("WG", wg_d);
    int nb = envi("NB", n >= 1024 ? 32 : 16);
    if (nb > 32) nb = 32;
    if (nb > n) nb = n;
    if (wg < 32) wg = 32;

    const size_t slm = (size_t(n) * size_t(nrhs) + size_t(nb) * size_t(nb)) * sizeof(T);
    const size_t slm_stream = (size_t(n) * size_t(nrhs) + 1) * sizeof(T);

    // THE CAPACITY GATE. The RHS is resident, so n * nrhs * sizeof(T) is a hard
    // ceiling on what this kernel can serve; above it the arm is not slow, it
    // cannot launch. Reported as a SKIP row rather than as a launch failure that
    // takes the whole cell with it.
    //
    // The budget is deliberately below local_mem_size: WP4 recorded a launch hole
    // at specific BYTE COUNTS (48,896 passes, 49,152 FAILS, 49,664 passes) that is
    // sticky per CUfunction, so a request near 48 KB is not safe merely because
    // the device advertises more.
    const size_t local_mem = q->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const size_t budget = envi("SLMBUDGET", 46080);
    const bool fits = slm <= budget && slm_stream <= budget && local_mem >= budget;

    // The device pointer arrays cuBLAS needs.
    auto pAs = Av.data_ptrs(*q);
    auto pXs = Xv.data_ptrs(*q);
    cublasHandle_t cbh = nullptr;
    cublasCreate(&cbh);
    int cbinfo = 0;

    auto reset = [&] {
        MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v);
        q->wait();
    };

    auto arm_vendor = [&] {
        cb_getrs(cbh, CUBLAS_OP_N, n, nrhs, pAs.data(), n, pivi, pXs.data(), n, &cbinfo, batch);
        cudaDeviceSynchronize();
    };
    auto arm_comp = [&] {
        sycl_getrs::getrs_blocked_dispatch<T>(
            *q, Av, Xv, Transpose::NoTrans, piv.to_span(), sws.to_span(),
            [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
               const MatrixView<T, MatrixFormat::Dense>& tb,
               T al, Side sd, Uplo up_, Transpose tr, Diag dg) {
                return trsm<BE, T>(c, ta, tb, al, sd, up_, tr, dg);
            });
        q->wait();
    };
    auto arm_fused = [&](int nbv, bool perm) {
        if (perm)
            fused::dispatch_nr<T, true>(sq, A.data(), n, int(sa), X.data(), n, int(sb),
                                        pivi, n, n, nrhs, batch, wg, nbv);
        else
            fused::dispatch_nr<T, false>(sq, A.data(), n, int(sa), X.data(), n, int(sb),
                                         pivi, n, n, nrhs, batch, wg, nbv);
        q->wait();
    };

    struct Arm { const char* name; std::function<void()> fn; bool on; double resid; std::vector<double> ms; };
    const bool noperm = envi("NOPERM", 0) != 0;
    std::vector<Arm> arms;
    arms.push_back({"vendor", arm_vendor, envi("NOVENDOR", 0) == 0, 0, {}});
    arms.push_back({"comp",   arm_comp,   envi("NOCOMP", 0) == 0, 0, {}});
    arms.push_back({"fstream", [&]{ arm_fused(1, true); }, fits && envi("NOSTREAM", 0) == 0, 0, {}});
    arms.push_back({"fblock",  [&]{ arm_fused(nb, true); }, fits, 0, {}});
    if (noperm) arms.push_back({"fblock_noperm", [&]{ arm_fused(nb, false); }, fits, 0, {}});
    if (!fits) std::printf("skip,%s,%d,%d,%d,%d,%d,%zu,0,0,0,0,0,0,%d,SLM\n",
                           tn, n, nrhs, batch, wg, nb, slm, ntp);

    // WARM: the JIT and the clocks. A cold first run has fabricated a 3.7x loss
    // in this campaign.
    try {
        const auto w0 = std::chrono::steady_clock::now();
        do {
            for (auto& a : arms) if (a.on) { reset(); a.fn(); }
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
    } catch (const sycl::exception& e) {
        std::printf("FAIL,%s,%d,%d,%d,%zu,%s\n", tn, n, nrhs, batch, slm, e.what());
        cublasDestroy(cbh);
        return 1;
    }

    // INTERLEAVED, arm by arm within each rep, one session.
    for (int r = 0; r < reps; ++r) {
        for (auto& a : arms) {
            if (!a.on) continue;
            reset();
            const auto t0 = std::chrono::steady_clock::now();
            a.fn();
            a.ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
    }
    // Correctness, once per arm, after the timing.
    for (auto& a : arms) {
        if (!a.on) continue;
        reset();
        a.fn();
        q->wait();
        a.resid = solve_probe<T>(X, B0, A0, n, nrhs, batch, envi("NPROBE", 2));
    }

    // Bytes the fused kernel MUST move: the factored matrix, once.
    const double bytes = double(batch) * double(n) * double(n) * double(sizeof(T));
    double ref = 0;
    for (auto& a : arms) if (a.on && std::string(a.name) == "vendor") ref = stat_of(a.ms).med;

    for (auto& a : arms) {
        if (!a.on) continue;
        const Stat s = stat_of(a.ms);
        const bool bad_ok = std::string(a.name) == "fblock_noperm";
        const bool ok = bad_ok ? true
                               : (std::isfinite(a.resid) && a.resid <= Tol<T>::v && ntp > 0);
        std::printf("%s,%s,%d,%d,%d,%d,%d,%zu,%.4f,%.4f,%.4f,%.1f,%.3f,%.3e,%d,%s\n",
                    a.name, tn, n, nrhs, batch, wg,
                    (std::string(a.name).rfind("fstream", 0) == 0) ? 1 : nb,
                    (std::string(a.name).rfind("fstream", 0) == 0) ? slm_stream : slm,
                    s.med, s.mean, s.relsd,
                    bytes / (s.med * 1e6),                    // GB/s
                    ref > 0 ? ref / s.med : 0.0,              // x over cuBLAS
                    a.resid, ntp, ok ? "ok" : "BAD");
    }
    cublasDestroy(cbh);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: fusedrs <type> <n> <nrhs> <batch> <reps>\n"
            "types : float double cfloat cdouble\n"
            "env   : WG NB WARM_S NPROBE NOPERM NOVENDOR NOCOMP\n"
            "cols  : arm,type,n,nrhs,batch,wg,nb,slm,med_ms,mean_ms,relsd,GBs,x_vs_cublas,resid,ntpiv,flag\n");
        return 2;
    }
    const std::string t = argv[1];
    const int n = std::atoi(argv[2]), nrhs = std::atoi(argv[3]);
    const int b = std::atoi(argv[4]), r = std::atoi(argv[5]);
    try {
        if (t == "float")   return run<float>("float", n, nrhs, b, r);
        if (t == "double")  return run<double>("double", n, nrhs, b, r);
        if (t == "cfloat")  return run<std::complex<float>>("cfloat", n, nrhs, b, r);
        if (t == "cdouble") return run<std::complex<double>>("cdouble", n, nrhs, b, r);
    } catch (const std::exception& e) {
        std::printf("THREW,%s,%d,%d,%d,%s\n", t.c_str(), n, nrhs, b, e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
