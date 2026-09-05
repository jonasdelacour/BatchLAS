// Native batched GETRS -- the fused narrow-RHS tier. One work-group per matrix does
// the row permutation, the forward substitution and the back substitution in a single
// kernel; the RHS and one nb x nb diagonal block are resident, L and U are streamed.
// getrs_native.cc's composed tier serves the wide-nrhs end.
// evidence: docs/perf/lu.md#the-fused-narrow-rhs-getrs
//
// Belongs in EXTENSIONS_FACTORIZATION_SOURCES: calling a getrf CTA device function
// from here fails the device link with `ptxas fatal`.

#include "getrs_native.hh"

#include "../queue.hh"
#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrs {

namespace {

using namespace batchlas::sycl_device;

// nb must not exceed the sub-group size: the block solve is a sub-group shuffle
// recurrence with lane i owning row i.
constexpr int kGetrsFusedNbSmall = 16;
constexpr int kGetrsFusedNbLarge = 32;
constexpr int kGetrsFusedNbMax   = 32;

inline int getrs_fused_nb(int n) {
    const int nb = (n >= 1024) ? kGetrsFusedNbLarge : kGetrsFusedNbSmall;
    return (nb > n) ? n : nb;
}

// The block is padded to nb + 1: the transposed block solve reads blk[s + t*ldb],
// stride ldb across lanes, so an unpadded 16 or 32 is a full bank conflict on every
// step of that recurrence. Inert on this device, kept for portability.
inline int getrs_fused_blk_ld(int nb) { return nb + 1; }

// The register gate. registers-per-work-item x work-group-size must not exceed 65,536
// or the launch ABORTS rather than merely slowing down. The table is per (type, body,
// width) rather than a max over them, and is measured with scripts/register_probe.sh
// -- re-run that probe if ptxas moves a cell by more than kGetrsFusedRegMargin.
constexpr int kGetrsFusedRegMargin = 8;

// The accumulator-width bucket. It MUST agree with fused_dispatch_nr's ladder below:
// a cap computed for one bucket is a cap for a different kernel.
constexpr int getrs_fused_nr_bucket(int nrhs) {
    return (nrhs <= 1) ? 0 : (nrhs <= 2) ? 1 : (nrhs <= 4) ? 2 : 3;
}

template <typename S> struct GetrsFusedRegs;
template <> struct GetrsFusedRegs<float> {
    static constexpr int notrans[4] = {39, 48, 48, 48};
    static constexpr int trans[4]   = {39, 40, 48, 68};
};
template <> struct GetrsFusedRegs<double> {
    static constexpr int notrans[4] = {39, 52, 44, 61};
    static constexpr int trans[4]   = {46, 44, 51, 72};
};
template <> struct GetrsFusedRegs<std::complex<float>> {
    static constexpr int notrans[4] = {40, 40, 40, 48};
    static constexpr int trans[4]   = {42, 43, 48, 56};
};
template <> struct GetrsFusedRegs<std::complex<double>> {
    static constexpr int notrans[4] = {54, 56, 56, 72};
    static constexpr int trans[4]   = {56, 58, 58, 86};
};

template <typename T>
constexpr int getrs_fused_regs_for(int nrhs, bool trans) {
    const int i = getrs_fused_nr_bucket(nrhs);
    return trans ? GetrsFusedRegs<T>::trans[i] : GetrsFusedRegs<T>::notrans[i];
}

// The work-group width: ~ n/2 clamped to [64, 1024], then capped by the register gate.
template <typename T>
inline int getrs_fused_wg(int n, int nrhs, int max_wg, bool trans) {
    int wg = 32;
    while (wg < n / 2 && wg < 1024) wg *= 2;
    if (wg < 64) wg = 64;

    const int regs = getrs_fused_regs_for<T>(nrhs, trans) + kGetrsFusedRegMargin;
    int cap = (65536 / regs) & ~31;          // down to a multiple of the sub-group
    if (cap < 32) cap = 32;
    if (wg > cap) wg = cap;

    if (wg > max_wg) wg = max_wg;
    if (wg < 32) wg = 32;
    return wg;
}

// The 48 KB launch hole, carried verbatim from potrf_cta.cc so the two agree: a dynamic
// local-memory request in (49152 - static_shared, 49152] fails at enqueue with
// CUDA_ERROR_INVALID_VALUE, and is STICKY PER CUfunction, so a larger earlier launch
// hides it from a warm test suite. evidence: docs/perf/lu.md#the-48-kb-launch-hole
constexpr std::size_t kGetrsHoleLo    = 47104;
constexpr std::size_t kGetrsHoleHi    = 49664;
constexpr std::size_t kGetrsHolePadTo = 49920;

constexpr std::size_t getrs_hole_padded(std::size_t bytes) {
    return (bytes > kGetrsHoleLo && bytes <= kGetrsHoleHi) ? kGetrsHolePadTo : bytes;
}

constexpr std::size_t getrs_fused_slm(std::size_t rhs_elems, int nb,
                                      std::size_t scalar_bytes) {
    return getrs_hole_padded(
        (rhs_elems + static_cast<std::size_t>(nb) *
                     static_cast<std::size_t>(getrs_fused_blk_ld(nb))) * scalar_bytes);
}

// A sub-group sum, hand-rolled with shift_group_left rather than
// sycl::reduce_over_group, which puts static shared into the kernel and reopens the
// 48 KB hole. Afterwards only LANE 0 holds the total.
template <typename SG, typename D>
inline D sg_sum(const SG& sg, D v) {
    if constexpr (dev_is_complex_v<D>) {
        auto re = v.re, im = v.im;
        for (int off = 16; off > 0; off >>= 1) {
            re += sycl::shift_group_left(sg, re, off);
            im += sycl::shift_group_left(sg, im, off);
        }
        return D{re, im};
    } else {
        for (int off = 16; off > 0; off >>= 1) v += sycl::shift_group_left(sg, v, off);
        return v;
    }
}

template <typename D>
inline D dev_zero_of() {
    if constexpr (dev_is_complex_v<D>) return D{0, 0};
    else return D(0);
}

template <typename T, int NR> class GetrsFusedNKernel;
template <typename T, int NR> class GetrsFusedTKernel;

// NoTrans: apply F to B, solve L y = F b (unit lower), solve U x = y.
template <typename T, int NR>
Event fused_launch_notrans(Queue& ctx,
                           const T* A, int lda, int strideA,
                           T* B, int ldb, int strideB,
                           const int* piv, int pstride,
                           int n, int nrhs, int batch, int wg, int nb) {
    using DM = DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const Ap = reinterpret_cast<const D*>(A);
    D* const Bp = reinterpret_cast<D*>(B);

    const int bld = getrs_fused_blk_ld(nb);
    const std::size_t rhs_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    const std::size_t slm_elems =
        getrs_fused_slm(rhs_elems, nb, sizeof(D)) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(slm_elems), h);
        h.parallel_for<GetrsFusedNKernel<T, NR>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = static_cast<int>(it.get_local_id(0));
                const std::size_t b = it.get_group(0);
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());

                const D* const Ab = Ap + b * static_cast<std::size_t>(strideA);
                D* const Bb = Bp + b * static_cast<std::size_t>(strideB);
                const int* const pv = piv + b * static_cast<std::size_t>(pstride);
                D* const y = &slm[0];
                D* const blk = &slm[rhs_elems];

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    y[e] = Bb[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // F: the interchange list, walked FORWARDS, in LOCAL memory
                if (tid < nrhs) {
                    D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);
                    for (int k = 0; k < n; ++k) {
                        const int p = pv[k] - 1;          // 1-BASED on the wire
                        if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // L y = F b, unit lower, forward
                for (int j = 0; j < n; j += nb) {
                    const int jb = (n - j < nb) ? (n - j) : nb;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            Ab[static_cast<std::size_t>(j + i) +
                               static_cast<std::size_t>(j + c) * static_cast<std::size_t>(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // The block solve runs inside ONE sub-group: lane i owns row j+i in
                    // a register and the recurrence takes no work-group barrier.
                    // group_broadcast is a collective and must not be called under
                    // divergence -- the lane guards are INSIDE it, not around.
                    if (sgid == 0 && jb > 1) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j + lane] : dev_zero_of<D>();
                            for (int kk = 0; kk < jb - 1; ++kk) {
                                const D pv2 = sycl::group_broadcast(sg, v, kk);
                                if (lane > kk && lane < jb)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(lane) +
                                                               static_cast<std::size_t>(kk) *
                                                               static_cast<std::size_t>(bld)], pv2));
                            }
                            if (lane < jb) yc[j + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Parallel over rows, so the read of A[i, j+kk] is coalesced.
                    for (int i = j + jb + tid; i < n; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[static_cast<std::size_t>(i) +
                                           static_cast<std::size_t>(j + kk) *
                                           static_cast<std::size_t>(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(j + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + static_cast<std::size_t>(c) *
                                                  static_cast<std::size_t>(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // U x = y, non-unit upper, backward
                for (int jend = n; jend > 0; jend -= nb) {
                    const int j0 = (jend - nb > 0) ? (jend - nb) : 0;
                    const int jb = jend - j0;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            Ab[static_cast<std::size_t>(j0 + i) +
                               static_cast<std::size_t>(j0 + c) * static_cast<std::size_t>(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (sgid == 0) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j0 + lane] : dev_zero_of<D>();
                            for (int kk = jb - 1; kk >= 0; --kk) {
                                if (lane == kk)
                                    v = dev_div(v, blk[static_cast<std::size_t>(kk) +
                                                       static_cast<std::size_t>(kk) *
                                                       static_cast<std::size_t>(bld)]);
                                const D pv2 = sycl::group_broadcast(sg, v, kk);
                                if (lane < kk)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(lane) +
                                                               static_cast<std::size_t>(kk) *
                                                               static_cast<std::size_t>(bld)], pv2));
                            }
                            if (lane < jb) yc[j0 + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    for (int i = tid; i < j0; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[static_cast<std::size_t>(i) +
                                           static_cast<std::size_t>(j0 + kk) *
                                           static_cast<std::size_t>(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(j0 + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + static_cast<std::size_t>(c) *
                                                  static_cast<std::size_t>(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)] = y[e];
                }
            });
    });
    return ctx.get_event();
}

// Trans / ConjTrans: solve op(U) z = b, solve op(L) w = z, then w = F^{-1} w -- the SAME
// interchange list walked BACKWARDS, because A^T = U^T L^T F moves the permutation to
// the OUTPUT and reverses it; getting that wrong is the classic silently-wrong getrs.
// Both solves are the DOT form, because op(U)[i][j] = op(U[j][i]): the reduction runs
// down a CONTIGUOUS COLUMN of A.
template <typename T, int NR>
Event fused_launch_trans(Queue& ctx,
                         const T* A, int lda, int strideA,
                         T* B, int ldb, int strideB,
                         const int* piv, int pstride,
                         int n, int nrhs, int batch, int wg, int nb, bool conj) {
    using DM = DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const Ap = reinterpret_cast<const D*>(A);
    D* const Bp = reinterpret_cast<D*>(B);

    const int bld = getrs_fused_blk_ld(nb);
    const std::size_t rhs_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    const std::size_t slm_elems =
        getrs_fused_slm(rhs_elems, nb, sizeof(D)) / sizeof(D);
    const int nsg = wg / 32;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(slm_elems), h);
        h.parallel_for<GetrsFusedTKernel<T, NR>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = static_cast<int>(it.get_local_id(0));
                const std::size_t b = it.get_group(0);
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());

                const D* const Ab = Ap + b * static_cast<std::size_t>(strideA);
                D* const Bb = Bp + b * static_cast<std::size_t>(strideB);
                const int* const pv = piv + b * static_cast<std::size_t>(pstride);
                D* const y = &slm[0];
                D* const blk = &slm[rhs_elems];

                auto ld_a = [&](int i, int c) {
                    const D a = Ab[static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(c) * static_cast<std::size_t>(lda)];
                    return conj ? dev_conj(a) : a;
                };

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    y[e] = Bb[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // op(U) z = b : op(U) is LOWER, non-unit, forward
                for (int j = 0; j < n; j += nb) {
                    const int jb = (n - j < nb) ? (n - j) : nb;

                    // Staged contiguously in i; indexed below as blk[s + t*bld].
                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            ld_a(j + i, j + c);
                    }

                    // The PAST contribution: y[j+t] -= sum_{i<j} op(A[i, j+t]) y[i]. ONE
                    // SUB-GROUP PER COLUMN, so its 32 lanes read 32 consecutive elements.
                    for (int t = sgid; t < jb; t += nsg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int i = lane; i < j; i += 32) {
                            const D a = ld_a(i, j + t);
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(i)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                const D s = sg_sum(sg, acc[c]);
                                if (lane == 0) {
                                    D* const yc = y + static_cast<std::size_t>(c) *
                                                      static_cast<std::size_t>(n);
                                    yc[j + t] = dev_sub(yc[j + t], s);
                                }
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // The diagonal block, by ONE sub-group. Lane t owns row t and
                    // reads blk[s + t*bld], stride bld across lanes -- the pad.
                    if (sgid == 0) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j + lane] : dev_zero_of<D>();
                            for (int s = 0; s < jb; ++s) {
                                if (lane == s)
                                    v = dev_div(v, blk[static_cast<std::size_t>(s) +
                                                       static_cast<std::size_t>(s) *
                                                       static_cast<std::size_t>(bld)]);
                                const D vs = sycl::group_broadcast(sg, v, s);
                                if (lane > s && lane < jb)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(s) +
                                                               static_cast<std::size_t>(lane) *
                                                               static_cast<std::size_t>(bld)], vs));
                            }
                            if (lane < jb) yc[j + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // op(L) w = z : op(L) is UPPER, UNIT, backward
                for (int jend = n; jend > 0; jend -= nb) {
                    const int j0 = (jend - nb > 0) ? (jend - nb) : 0;
                    const int jb = jend - j0;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            ld_a(j0 + i, j0 + c);
                    }

                    for (int t = sgid; t < jb; t += nsg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int i = jend + lane; i < n; i += 32) {
                            const D a = ld_a(i, j0 + t);
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(i)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                const D s = sg_sum(sg, acc[c]);
                                if (lane == 0) {
                                    D* const yc = y + static_cast<std::size_t>(c) *
                                                      static_cast<std::size_t>(n);
                                    yc[j0 + t] = dev_sub(yc[j0 + t], s);
                                }
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // UNIT diagonal: no division, and the recurrence runs
                    // BACKWARDS because op(L) is upper.
                    if (sgid == 0 && jb > 1) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j0 + lane] : dev_zero_of<D>();
                            for (int s = jb - 1; s > 0; --s) {
                                const D vs = sycl::group_broadcast(sg, v, s);
                                if (lane < s)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(s) +
                                                               static_cast<std::size_t>(lane) *
                                                               static_cast<std::size_t>(bld)], vs));
                            }
                            if (lane < jb) yc[j0 + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // F^{-1} on the OUTPUT: the SAME list walked BACKWARDS. Every
                // transposition is its own inverse, so only the ORDER changes.
                if (tid < nrhs) {
                    D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);
                    for (int k = n - 1; k >= 0; --k) {
                        const int p = pv[k] - 1;
                        if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)] = y[e];
                }
            });
    });
    return ctx.get_event();
}

// Runtime nrhs -> the compile-time accumulator width. The ladder must match
// getrs_fused_nr_bucket, and stops at kGetrsFusedMaxRhs (route_getrs.hh).
template <typename T>
Event fused_dispatch_nr(Queue& ctx, bool trans, bool conj,
                        const T* A, int lda, int sA, T* B, int ldb, int sB,
                        const int* piv, int pstride,
                        int n, int nrhs, int batch, int wg, int nb) {
    #define BATCHLAS_GETRS_FUSED_ARM(NRV)                                            \
        if (trans) return fused_launch_trans<T, NRV>(ctx, A, lda, sA, B, ldb, sB,    \
                                                     piv, pstride, n, nrhs, batch,   \
                                                     wg, nb, conj);                  \
        return fused_launch_notrans<T, NRV>(ctx, A, lda, sA, B, ldb, sB,             \
                                            piv, pstride, n, nrhs, batch, wg, nb);
    if (nrhs <= 1) { BATCHLAS_GETRS_FUSED_ARM(1) }
    if (nrhs <= 2) { BATCHLAS_GETRS_FUSED_ARM(2) }
    if (nrhs <= 4) { BATCHLAS_GETRS_FUSED_ARM(4) }
    BATCHLAS_GETRS_FUSED_ARM(8)
    #undef BATCHLAS_GETRS_FUSED_ARM
}

}  // namespace

template <> bool getrs_fused_available<float>()                { return true; }
template <> bool getrs_fused_available<double>()               { return true; }
template <> bool getrs_fused_available<std::complex<float>>()  { return true; }
template <> bool getrs_fused_available<std::complex<double>>() { return true; }

// THE CAPACITY, IN RHS ELEMENTS (n * nrhs). The RHS vector is resident, so this is a
// HARD launch ceiling -- a supports() question and not a preferred() one. The budget is
// asked of the DEVICE, and the largest nb the tier ever uses is charged, not this
// call's. getrs_hole_padded is NOT monotone, so the largest admissible request is the
// budget when it exceeds kGetrsHoleHi and min(budget, kGetrsHoleLo) otherwise.
template <typename T>
std::size_t getrs_fused_max_rhs_elems(std::size_t slm_budget_bytes) {
    using D = typename sycl_device::DevMap<T>::type;
    const std::size_t admissible =
        (slm_budget_bytes > kGetrsHoleHi) ? slm_budget_bytes
                                          : std::min(slm_budget_bytes, kGetrsHoleLo);
    const std::size_t blk_bytes =
        static_cast<std::size_t>(kGetrsFusedNbMax) *
        static_cast<std::size_t>(getrs_fused_blk_ld(kGetrsFusedNbMax)) * sizeof(D);
    if (admissible <= blk_bytes) return 0;
    const std::size_t elems = (admissible - blk_bytes) / sizeof(D);

    // The floor division above can round the implied request BACK DOWN INTO the band,
    // where the pad raises it again and the launch is refused; the exact repair is the
    // request that ends AT kGetrsHoleLo. evidence: docs/perf/lu.md#correctness-findings
    if (getrs_fused_slm(elems, kGetrsFusedNbMax, sizeof(D)) > slm_budget_bytes) {
        if (kGetrsHoleLo <= blk_bytes) return 0;
        return (kGetrsHoleLo - blk_bytes) / sizeof(D);
    }
    return elems;
}

// WORKSPACE: ZERO -- the RHS is permuted and solved in local memory and written back in
// place. Nothing is dereferenced; a measuring pass hands this null data pointers.
template <typename T>
std::size_t getrs_fused_buffer_size(Queue&,
                                    const MatrixView<T, MatrixFormat::Dense>&,
                                    const MatrixView<T, MatrixFormat::Dense>&,
                                    Transpose) {
    return 0;
}

// Every gate RouteTable<Op::getrs,T>::supports() applies is RE-APPLIED here, because
// this entry point is reachable WITHOUT the table: route_resolve.hh falls through to
// automatic() when a forced route is unsupported.
template <typename T>
Event getrs_fused_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           Transpose transA,
                           Span<int64_t> pivots,
                           Span<std::byte> workspace) {
    static_cast<void>(workspace);   // this tier needs none

    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (n < 1 || nrhs < 1 || batch < 1) {
        throw std::invalid_argument("getrs_fused: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getrs_fused: A must be square");
    }
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("getrs_fused: B must have A.rows() rows");
    }
    if (A.batch_size() != B.batch_size()) {
        throw std::invalid_argument("getrs_fused: A and B must agree on batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        throw std::invalid_argument("getrs_fused: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrs_fused: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that property
        // returns sub_group_sizes()[0], so the weak test refuses a {8,16,32} device
        // and ACCEPTS a {64} one -- where this kernel's reqd_sub_group_size(32)
        // block solve is a launch abort.
        throw std::runtime_error(
            "getrs_fused: device does not offer sub-group size 32");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrs_fused: pivot span is shorter than n * batch");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const std::size_t need = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    if (need > getrs_fused_max_rhs_elems<T>(budget)) {
        throw std::invalid_argument(
            "getrs_fused: n * nrhs = " + std::to_string(need) +
            " exceeds this device's resident-RHS capacity (" +
            std::to_string(getrs_fused_max_rhs_elems<T>(budget)) +
            " elements). This is a CAPACITY ceiling, not a speed one: route the "
            "call to Algorithm::Blocked instead.");
    }
    if (nrhs > kGetrsFusedMaxRhs) {
        throw std::invalid_argument(
            "getrs_fused: nrhs = " + std::to_string(nrhs) + " is above the widest "
            "instantiated accumulator (" + std::to_string(kGetrsFusedMaxRhs) +
            "). Route to Algorithm::Blocked.");
    }

    // PACKED 1-BASED int32 -- the format cublas.cc and rocsolver.cc read through
    // pivots.as_span<int>(), and the one a native getrf writes (getrf_native.hh's
    // PIVOT CONTRACT). Every mixture of native and vendor arms is reachable.
    auto piv_i32 = pivots.as_span<int>();
    const bool trans = (transA != Transpose::NoTrans);

    const int nb = getrs_fused_nb(n);
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int wg = getrs_fused_wg<T>(n, nrhs, max_wg, trans);

    return fused_dispatch_nr<T>(
        ctx,
        trans,
        /*conj=*/transA == Transpose::ConjTrans,
        A.data_ptr(), A.ld(), A.stride(),
        B.data_ptr(), B.ld(), B.stride(),
        piv_i32.data(), /*pstride=*/n,
        n, nrhs, batch, wg, nb);
}

#define BATCHLAS_GETRS_FUSED_INSTANTIATE(T)                                                \
    template std::size_t getrs_fused_max_rhs_elems<T>(std::size_t);                        \
    template std::size_t getrs_fused_buffer_size<T>(                                       \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose);                             \
    template Event getrs_fused_dispatch<T>(                                                \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose,                              \
        Span<int64_t>, Span<std::byte>);

BATCHLAS_GETRS_FUSED_INSTANTIATE(float)
BATCHLAS_GETRS_FUSED_INSTANTIATE(double)
BATCHLAS_GETRS_FUSED_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_FUSED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_FUSED_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
