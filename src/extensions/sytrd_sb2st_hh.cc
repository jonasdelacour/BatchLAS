// Stage-2 band -> tridiagonal by Householder bulge chasing, retaining the
// reflectors for the eigenvector back-transform. See sytrd_sb2st_hh.hh for the
// rationale and for the schedule, which is validated in
// playground/sb2st_hh_sequential.py.

#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/env.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include "sytrd_sb2st_hh.hh"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace batchlas {
namespace internal {

namespace {

template <typename U>
inline U conj_if(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline T real_part_as_T(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), typename base_type<T>::type(0));
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type abs2(const T& x) {
    using R = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        return static_cast<R>(x.real()) * static_cast<R>(x.real()) +
               static_cast<R>(x.imag()) * static_cast<R>(x.imag());
    } else {
        return static_cast<R>(x) * static_cast<R>(x);
    }
}

// Group-wide sum that also works for std::complex, which sycl::plus<> does not
// accept directly.
template <typename Group, typename T>
inline T group_sum(Group g, T v) {
    using R = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const R re = sycl::reduce_over_group(g, static_cast<R>(v.real()), sycl::plus<R>());
        const R im = sycl::reduce_over_group(g, static_cast<R>(v.imag()), sycl::plus<R>());
        return T(re, im);
    } else {
        return sycl::reduce_over_group(g, v, sycl::plus<T>());
    }
}

template <Backend B, typename T>
class Sb2stHhChaseKernel;

// Sub-group XOR shuffle that also works for std::complex.
template <typename T>
inline T shuffle_xor(sycl::sub_group sg, T v, uint32_t mask) {
    using R = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const R re = sycl::permute_group_by_xor(sg, static_cast<R>(v.real()), mask);
        const R im = sycl::permute_group_by_xor(sg, static_cast<R>(v.imag()), mask);
        return T(re, im);
    } else {
        return sycl::permute_group_by_xor(sg, v, mask);
    }
}

// The chase is sequential per matrix, so a work-group owns one problem and the
// only parallelism inside it is the <= kd x kd window of the current step. With
// a 32-thread work-group and kd=32 that means each lane loops 32x serially, and
// at batch=128 the whole kernel occupies ~4096 threads on a GPU with ~196k
// slots -- about 2% occupancy, which is what made the chase cost as much as the
// back-transform. kWg=256 with a 2D (row, column-chunk) mapping puts 8 lanes on
// each row of the window instead of 1.
//
// kRowsPar is a pure split of the 256 lanes between window rows and the reduced
// dimension, NOT a bound on kd: windows taller than kRowsPar are walked in
// row-blocks. kTpr must stay <= 32 so the lanes sharing a row are inside one
// sub-group.
constexpr int kWg = 256;
constexpr int kRowsPar = 32;          // window rows resident at once
constexpr int kTpr = kWg / kRowsPar;  // lanes cooperating on one row/column
static_assert(kTpr <= 32 && kRowsPar * kTpr == kWg);

} // namespace

// Single description of sytrd_sb2st_hh's workspace; see workspace_bytes() in
// util/mempool.hh. The expanded working band is wider than the input band
// because transient bulge fill reaches kd rows below it.
template <typename T>
Span<T> sytrd_sb2st_hh_layout(Queue& ctx, BumpAllocator& pool, int32_t n, int32_t kdw, int32_t batch) {
    return pool.allocate<T>(
        ctx, static_cast<size_t>(kdw + 1) * static_cast<size_t>(n) * static_cast<size_t>(batch));
}

template <Backend B, typename T>
size_t sytrd_sb2st_hh_buffer_size(Queue& ctx, int32_t n, int32_t kd, int32_t batch) {
    if (n <= 0 || batch <= 0) return 0;
    const int32_t kdw = sb2st_hh_work_bandwidth(n, kd);
    return workspace_bytes([&](BumpAllocator& pool) {
        return sytrd_sb2st_hh_layout<T>(ctx, pool, n, kdw, batch);
    });
}

// ab_in      : (kd+1) x n  lower band, read-only
// ab_tri_out : 2 x n       row 0 = diagonal, row 1 = *signed* subdiagonal
//              (kept signed so build_phase_from_kd1_band works unchanged)
// d_out/e_out: real diagonal and |subdiagonal|
// v_out      : kd x nrefl  reflector k in column k, v[0] = 1, zero-padded
// tau_out    : nrefl
template <Backend B, typename T>
Event sytrd_sb2st_hh(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                     const MatrixView<T, MatrixFormat::Dense>& ab_tri_out,
                     const VectorView<typename base_type<T>::type>& d_out,
                     const VectorView<typename base_type<T>::type>& e_out,
                     const MatrixView<T, MatrixFormat::Dense>& v_out,
                     const VectorView<T>& tau_out,
                     Uplo uplo,
                     int32_t kd,
                     const Span<std::byte>& ws) {
    using Real = typename base_type<T>::type;

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_sb2st_hh: requires an in-order Queue");
    }
    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_sb2st_hh: only Uplo::Lower is implemented");
    }

    const int32_t n = static_cast<int32_t>(ab_in.cols());
    const int32_t batch = static_cast<int32_t>(ab_in.batch_size());
    if (n <= 0 || batch <= 0) return ctx.get_event();

    const int32_t kd_i = std::max<int32_t>(0, kd);
    const int32_t kdw = sb2st_hh_work_bandwidth(n, kd_i);

    BumpAllocator pool(ws);

    auto abw = sytrd_sb2st_hh_layout<T>(ctx, pool, n, kdw, batch);

    const int32_t ldw = kdw + 1;

    auto ABsrc = ab_in.kernel_view();
    auto ABtri = ab_tri_out.kernel_view();
    auto Vout = v_out.kernel_view();
    const int32_t nrefl = static_cast<int32_t>(v_out.cols());
    const int32_t ldv = static_cast<int32_t>(v_out.ld());
    T* abw_ptr = abw.data();

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> vloc(sycl::range<1>(static_cast<size_t>(std::max(1, kd_i))), h);
        sycl::local_accessor<T, 1> wloc(sycl::range<1>(static_cast<size_t>(std::max(1, kd_i))), h);

        auto Dv = d_out;
        auto Ev = e_out;
        auto TAUv = tau_out;

        h.parallel_for<Sb2stHhChaseKernel<B, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kWg),
                              sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const int32_t b = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t lid = static_cast<int32_t>(it.get_local_linear_id());
                if (b >= batch) return;

                T* AB = abw_ptr + static_cast<size_t>(b) * static_cast<size_t>(ldw) *
                                      static_cast<size_t>(n);
                auto ABs = ABsrc.batch_item(b);
                auto ABt = ABtri.batch_item(b);
                auto Vb = Vout.batch_item(b);

                // --- Load the input band into the expanded working band.
                for (int32_t idx = lid; idx < ldw * n; idx += kWg) {
                    const int32_t col = idx / ldw;
                    const int32_t row = idx - col * ldw;
                    AB[idx] = (row <= kd_i) ? ABs(row, col) : T(0);
                }
                sycl::group_barrier(wg);

                // Hermitian input: force a real diagonal before we start.
                if constexpr (internal::is_complex<T>::value) {
                    for (int32_t j = lid; j < n; j += kWg) {
                        AB[j * ldw] = real_part_as_T(AB[j * ldw]);
                    }
                    sycl::group_barrier(wg);
                }

                auto bget = [&](int32_t i, int32_t j) -> T {
                    if (i >= j) {
                        const int32_t r = i - j;
                        return (r <= kdw) ? AB[r + j * ldw] : T(0);
                    }
                    const int32_t r = j - i;
                    return (r <= kdw) ? conj_if(AB[r + i * ldw]) : T(0);
                };
                auto bset = [&](int32_t i, int32_t j, T val) {
                    if (i >= j) {
                        const int32_t r = i - j;
                        if (r <= kdw) AB[r + j * ldw] = val;
                    } else {
                        const int32_t r = j - i;
                        if (r <= kdw) AB[r + i * ldw] = conj_if(val);
                    }
                };

                // W <- H W H^H on the principal block [a..b], H = I - tau v v^H,
                // v held in vloc with length m. Only the lower triangle is
                // written; the update keeps the diagonal real by construction.
                // 2D lane mapping: ti indexes the row (or column) of the window,
                // ts indexes a chunk of the reduced dimension. Lanes sharing a ti
                // are the kTpr contiguous lanes ti*kTpr .. ti*kTpr+kTpr-1, which
                // sit inside one sub-group, so the partial sums reduce with XOR
                // shuffles rather than through local memory.
                const int32_t ti = lid / kTpr;
                const int32_t ts = lid % kTpr;
                const auto sg = it.get_sub_group();

                // NOTE: this is a sub-group collective, so every lane in the
                // work-group must call it -- out-of-range lanes contribute 0
                // rather than branching around it.
                auto reduce_tpr = [&](T v) -> T {
                    for (uint32_t msk = 1; msk < static_cast<uint32_t>(kTpr); msk <<= 1) {
                        v += shuffle_xor(sg, v, msk);
                    }
                    return v;
                };

                auto two_sided = [&](int32_t a, int32_t bb, T tau) {
                    const int32_t m = bb - a + 1;
                    if (m <= 0) return;

                    // p = W v, one row per ti, reduced over ts. The row-block
                    // count is derived from m (uniform) so every lane makes the
                    // same number of reduce_tpr calls.
                    {
                        const int32_t nrb = (m + kRowsPar - 1) / kRowsPar;
                        for (int32_t rb = 0; rb < nrb; ++rb) {
                            const int32_t r = rb * kRowsPar + ti;
                            T acc = T(0);
                            if (r < m) {
                                for (int32_t j = ts; j < m; j += kTpr) {
                                    acc += bget(a + r, a + j) * vloc[j];
                                }
                            }
                            acc = reduce_tpr(acc);
                            if (r < m && ts == 0) wloc[r] = acc;
                        }
                    }
                    sycl::group_barrier(wg);

                    // kappa = v^H W v
                    T part = T(0);
                    for (int32_t i = lid; i < m; i += kWg) {
                        part += conj_if(vloc[i]) * wloc[i];
                    }
                    const T kappa = group_sum(wg, part);

                    // w = conj(tau) p - (|tau|^2 kappa / 2) v
                    const T coef = tau * conj_if(tau) * kappa * T(Real(0.5));
                    for (int32_t i = lid; i < m; i += kWg) {
                        wloc[i] = conj_if(tau) * wloc[i] - coef * vloc[i];
                    }
                    sycl::group_barrier(wg);

                    // Rank-2 update of the lower triangle; no reduction, so ts
                    // simply strides the inner index.
                    for (int32_t r = ti; r < m; r += kRowsPar) {
                        for (int32_t j = ts; j <= r; j += kTpr) {
                            T val = bget(a + r, a + j) - wloc[r] * conj_if(vloc[j]) -
                                    vloc[r] * conj_if(wloc[j]);
                            if (r == j) val = real_part_as_T(val);
                            bset(a + r, a + j, val);
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // B <- B (I - tau v v^H), v of length (c1-c0+1) in vloc.
                auto right_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
                    if (r0 > r1 || c0 > c1) return;
                    const int32_t nr = r1 - r0 + 1;
                    const int32_t nrb = (nr + kRowsPar - 1) / kRowsPar;
                    for (int32_t rb = 0; rb < nrb; ++rb) {
                        const int32_t rr = rb * kRowsPar + ti;
                        const int32_t i = r0 + rr;
                        T y = T(0);
                        if (rr < nr) {
                            for (int32_t j = c0 + ts; j <= c1; j += kTpr) {
                                y += bget(i, j) * vloc[j - c0];
                            }
                        }
                        y = tau * reduce_tpr(y);
                        if (rr < nr) {
                            for (int32_t j = c0 + ts; j <= c1; j += kTpr) {
                                bset(i, j, bget(i, j) - y * conj_if(vloc[j - c0]));
                            }
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // C <- (I - tau v v^H) C, v of length (r1-r0+1) in vloc.
                auto left_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
                    if (r0 > r1 || c0 > c1) return;
                    const int32_t nc = c1 - c0 + 1;
                    const int32_t ncb = (nc + kRowsPar - 1) / kRowsPar;
                    for (int32_t cb = 0; cb < ncb; ++cb) {
                        const int32_t cc = cb * kRowsPar + ti;
                        const int32_t j = c0 + cc;
                        T z = T(0);
                        if (cc < nc) {
                            for (int32_t i = r0 + ts; i <= r1; i += kTpr) {
                                z += conj_if(vloc[i - r0]) * bget(i, j);
                            }
                        }
                        z = tau * reduce_tpr(z);
                        if (cc < nc) {
                            for (int32_t i = r0 + ts; i <= r1; i += kTpr) {
                                bset(i, j, bget(i, j) - vloc[i - r0] * z);
                            }
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // Build a reflector from column `col`, rows [r0..r1]; write it to
                // vloc, store it at reflector slot `slot`, and overwrite the
                // column with (beta, 0, ..., 0). Returns tau.
                auto make_reflector = [&](int32_t col, int32_t r0, int32_t r1,
                                          int32_t slot) -> T {
                    const int32_t m = r1 - r0 + 1;
                    Real partial = Real(0);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        partial += abs2(bget(r0 + k, col));
                    }
                    const Real ss = sycl::reduce_over_group(wg, partial, sycl::plus<Real>());
                    const Real xnorm = sycl::sqrt(ss);
                    const T alpha = bget(r0, col);
                    const auto res = internal::larfg<T>(alpha, xnorm, m);

                    // v[0] = 1, v[k] = x[k] * scale. Read the column before it is
                    // overwritten below.
                    if (lid == 0) vloc[0] = T(1);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        vloc[k] = (res.tau == T(0)) ? T(0) : bget(r0 + k, col) * res.scale;
                    }
                    sycl::group_barrier(wg);

                    if (lid == 0) bset(r0, col, res.beta);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        bset(r0 + k, col, T(0));
                    }

                    for (int32_t k = lid; k < kd_i; k += kWg) {
                        Vb(k, slot) = (k < m) ? vloc[k] : T(0);
                    }
                    if (lid == 0) TAUv(slot, b) = res.tau;
                    sycl::group_barrier(wg);
                    return res.tau;
                };

                // --- Sequential chase schedule. The reflector counter must match
                // build_sb2st_hh_schedule() exactly, since the back-transform
                // indexes V by the host-side schedule.
                int32_t slot = 0;
                if (kd_i > 1) {
                    for (int32_t st = 0; st + 2 < n; ++st) {
                        int32_t r0 = st + 1;
                        int32_t r1 = (st + kd_i < n - 1) ? (st + kd_i) : (n - 1);
                        if (r1 <= r0) continue;

                        T tau = make_reflector(st, r0, r1, slot++);
                        two_sided(r0, r1, tau);

                        while (true) {
                            const int32_t p0 = r1 + 1;
                            const int32_t p1 = (r1 + kd_i < n - 1) ? (r1 + kd_i) : (n - 1);
                            if (p0 > p1) break;

                            right_apply(p0, p1, r0, r1, tau);
                            tau = make_reflector(r0, p0, p1, slot++);
                            left_apply(p0, p1, r0 + 1, r1, tau);
                            two_sided(p0, p1, tau);

                            r0 = p0;
                            r1 = p1;
                        }
                    }
                }
                (void)nrefl;
                (void)ldv;

                // --- Extract the tridiagonal.
                for (int32_t j = lid; j < n; j += kWg) {
                    const T diag = bget(j, j);
                    ABt(0, j) = real_part_as_T(diag);
                    if constexpr (internal::is_complex<T>::value) {
                        Dv(j, b) = static_cast<Real>(diag.real());
                    } else {
                        Dv(j, b) = static_cast<Real>(diag);
                    }
                }
                for (int32_t j = lid; j < n - 1; j += kWg) {
                    const T sub = bget(j + 1, j);
                    ABt(1, j) = sub;  // signed: the phase pass consumes this
                    Ev(j, b) = internal::abs(sub);
                }
                if (lid == 0 && n > 0) ABt(1, n - 1) = T(0);
            });
    });

    return ctx.get_event();
}

// ---------------------------------------------------------------------------
// Q2 back-transform:  Z := Q2 Z  with  Q2 = H_1 H_2 ... H_m.
//
// Because the product is in generation order, Z := Q2 Z applies the reflectors
// in *reverse* order: H_m first, H_1 last.
//
// Parallelisation: reflectors act on rows, so distinct columns of Z are
// completely independent. Each work-group takes one (batch item, column chunk)
// and walks the entire reflector list itself -- no inter-work-group
// synchronisation and no launch per sweep. Within a work-group the 32 lanes map
// to *rows* of the reflector (row-blocked when kd > 32), which keeps the Z
// accesses contiguous since Z is column-major; the dot product is a sub-group
// reduction.
//
// This is flop-optimal (2n^3 total, versus ~4n^3 for a larft/larfb formulation
// whose V panels are zero-padded) but memory bound. Wang et al. (PPoPP'25)
// found a hand-written BLAS-2 back-transform beat MAGMA's BLAS-3 one by 1.5x on
// A100 for exactly this reason, so this is a reasonable starting point; the
// larft grouping remains available if profiling says otherwise.
namespace {
template <Backend B, typename T>
class Sb2stHhBackKernel;

template <Backend B, typename T, int C>
class Sb2stHhBackTiledKernel;

template <Backend B, typename T, int C, int S>
class Sb2stHhBackWaveKernel;

constexpr int kBackCols = 8;  // columns of Z per work-group (global-memory path)

}

// Wave back-transform: resident Z tile + concurrent application of every
// reflector in a commuting run.
//
// The resident-tile kernel below fixed the traffic on Z but left the *serial
// chain* untouched: one work-group walked all m reflectors one at a time with
// 32 threads. At n=1024/kd=64 that is 8687 dependent steps, and the tile's
// local-memory footprint capped occupancy at ~1 block/SM worth of threads.
//
// But the reflectors of a single chase sweep act on disjoint row ranges (each
// starts one past the previous one's end), so they commute and can all be
// applied at once. build_sb2st_hh_wave_offsets recovers those runs: at
// n=1024/kd=64 the 8687 reflectors form 1022 waves averaging 8.5 reflectors, so
// the chain is 8.5x shorter than the reflector count suggests.
//
// One work-group of S sub-groups owns a C-column tile of Z. Per wave each
// sub-group takes reflectors k = lo + sgid, lo + sgid + S, ...; they touch
// disjoint rows of the tile, so no synchronisation is needed *within* a wave --
// not even between successive reflectors handled by the same sub-group. A
// single work-group barrier separates waves.
//
// This also fixes the occupancy problem for free: the tile costs n*C
// regardless of S, so going from 32 to S*32 threads multiplies threads-per-byte
// of local memory by S.
template <Backend B, typename T, int C, int S>
Event unmqr_hb2st_wave(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& v_in,
                       const VectorView<T>& tau_in,
                       const MatrixView<T, MatrixFormat::Dense>& z_io,
                       int32_t n,
                       const int32_t* starts_p,
                       const int32_t* lens_p,
                       const int32_t* waves_p,
                       int32_t num_waves) {
    constexpr int32_t kSg = 32;
    constexpr int32_t G = kSg / C;
    constexpr int32_t kWg = kSg * S;
    static_assert(C >= 1 && C <= 32 && (kSg % C) == 0);
    static_assert(S >= 1);

    const int32_t batch = static_cast<int32_t>(z_io.batch_size());
    const int32_t ncols = static_cast<int32_t>(z_io.cols());
    const int32_t col_chunks = (ncols + C - 1) / C;
    const int32_t num_wg = batch * col_chunks;

    auto Vv = v_in.kernel_view();
    auto Zv = z_io.kernel_view();

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> Zs(
            sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(C)), h);
        auto TAUv = tau_in;

        h.parallel_for<Sb2stHhBackWaveKernel<B, T, C, S>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wg) * kWg),
                              sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const auto sg = it.get_sub_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t lid = static_cast<int32_t>(it.get_local_linear_id());
                const int32_t sgid = lid / kSg;
                const int32_t lane = lid - sgid * kSg;

                const int32_t b = wg_id / col_chunks;
                const int32_t chunk = wg_id - b * col_chunks;
                if (b >= batch) return;

                const int32_t c0 = chunk * C;
                auto Z = Zv.batch_item(b);
                auto V = Vv.batch_item(b);

                T* Zl = Zs.template get_multi_ptr<sycl::access::decorated::no>().get();

                for (int32_t idx = lid; idx < n * C; idx += kWg) {
                    const int32_t r = idx / C;
                    const int32_t c = idx - r * C;
                    Zl[idx] = (c0 + c < ncols) ? Z(r, c0 + c) : T(0);
                }
                sycl::group_barrier(wg);

                const int32_t col = lane % C;
                const int32_t grp = lane / C;

                // Q = H_1 ... H_m, so Z := Q Z runs the waves in reverse
                // generation order. Order *within* a wave is free.
                for (int32_t wv = num_waves - 1; wv >= 0; --wv) {
                    const int32_t lo = waves_p[wv];
                    const int32_t hi = waves_p[wv + 1];

                    // Every lane of a sub-group shares k, so tau and L are
                    // sub-group uniform and the shuffles below stay uniform.
                    for (int32_t k = lo + sgid; k < hi; k += S) {
                        const T tau = TAUv(k, b);
                        if (tau == T(0)) continue;
                        const int32_t s = starts_p[k];
                        const int32_t L = lens_p[k];

                        T acc = T(0);
                        for (int32_t r = grp; r < L; r += G) {
                            acc += conj_if(V(r, k)) * Zl[(s + r) * C + col];
                        }
                        if constexpr (G > 1) {
                            for (uint32_t m = C; m < static_cast<uint32_t>(kSg); m <<= 1) {
                                acc += shuffle_xor(sg, acc, m);
                            }
                        }
                        for (int32_t r = grp; r < L; r += G) {
                            Zl[(s + r) * C + col] -= tau * V(r, k) * acc;
                        }
                        // No barrier: the next k in this wave is disjoint from
                        // this one, and other sub-groups are on disjoint rows.
                    }
                    sycl::group_barrier(wg);
                }

                for (int32_t idx = lid; idx < n * C; idx += kWg) {
                    const int32_t r = idx / C;
                    const int32_t c = idx - r * C;
                    if (c0 + c < ncols) Z(r, c0 + c) = Zl[idx];
                }
            });
    });

    return ctx.get_event();
}

// Resident-tile back-transform.
//
// The bottleneck is not flops, it is traffic on Z: every reflector reloads its
// kd rows, and each row of Z is touched ~n/2 times over the whole sweep set, so
// the naive form moves ~n^3 elements for 2n^3 flops -- about 2 flops/element.
//
// A larft/larfb formulation cannot fix that here. Two reflectors commute iff
// their row ranges are disjoint, i.e. iff |u - u'| >= kd where u = start-1. The
// BLAS-3-groupable sets (consecutive u, one per sweep at a fixed chase step)
// interleave: for n=32,kd=4 generation order gives u = 0,4,8,1,..., so u=0
// precedes u=4 which precedes u=1, and the groups {0..3} and {4..7} cannot be
// linearised as contiguous units. More generally, in any valid schedule the
// consecutive reflectors are mutually disjoint -- that is what makes them
// schedulable -- so the row-sharing ones are always far apart in the product.
// This is why Wang et al. (PPoPP'25) found a hand-written BLAS-2 back-transform
// beat MAGMA's larft-based one by 1.5x on A100.
//
// So instead of reordering, keep Z resident: one work-group owns a C-column
// tile of Z for one batch item, loads it into local memory once, applies every
// reflector in the exact same order, and writes it back once. Traffic drops
// from ~n^3 to 2*n*ncols per matrix, which is where the BLAS-3-like arithmetic
// intensity actually comes from.
//
// Lane mapping: lane -> (column, row-group), col = lane % C, grp = lane / C,
// with G = 32/C lanes cooperating per column. Lanes with the same column differ
// by multiples of C, so the dot product reduces with XOR masks C, 2C, ... < 32.
// Zs is row-major with C columns, so consecutive lanes touch consecutive
// addresses.
template <Backend B, typename T, int C>
Event unmqr_hb2st_tiled(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& v_in,
                        const VectorView<T>& tau_in,
                        const MatrixView<T, MatrixFormat::Dense>& z_io,
                        int32_t n,
                        int32_t nrefl,
                        const int32_t* starts_p,
                        const int32_t* lens_p) {
    constexpr int32_t kSg = 32;
    constexpr int32_t G = kSg / C;
    static_assert(C >= 1 && C <= 32 && (kSg % C) == 0);

    const int32_t batch = static_cast<int32_t>(z_io.batch_size());
    const int32_t ncols = static_cast<int32_t>(z_io.cols());
    const int32_t col_chunks = (ncols + C - 1) / C;
    const int32_t num_wg = batch * col_chunks;

    auto Vv = v_in.kernel_view();
    auto Zv = z_io.kernel_view();

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> Zs(
            sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(C)), h);
        auto TAUv = tau_in;

        h.parallel_for<Sb2stHhBackTiledKernel<B, T, C>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wg) * kSg),
                              sycl::range<1>(kSg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t lid = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t b = wg_id / col_chunks;
                const int32_t chunk = wg_id - b * col_chunks;
                if (b >= batch) return;

                const int32_t c0 = chunk * C;
                auto Z = Zv.batch_item(b);
                auto V = Vv.batch_item(b);

                T* Zl = Zs.template get_multi_ptr<sycl::access::decorated::no>().get();

                // Load the column tile once.
                for (int32_t idx = lid; idx < n * C; idx += kSg) {
                    const int32_t r = idx / C;
                    const int32_t c = idx - r * C;
                    Zl[idx] = (c0 + c < ncols) ? Z(r, c0 + c) : T(0);
                }
                sycl::group_barrier(sg);

                const int32_t col = lid % C;
                const int32_t grp = lid / C;

                // Q = H_1 ... H_m, so Z := Q Z applies them in reverse order.
                for (int32_t k = nrefl - 1; k >= 0; --k) {
                    const T tau = TAUv(k, b);
                    if (tau == T(0)) continue;
                    const int32_t s = starts_p[k];
                    const int32_t L = lens_p[k];

                    T acc = T(0);
                    for (int32_t r = grp; r < L; r += G) {
                        acc += conj_if(V(r, k)) * Zl[(s + r) * C + col];
                    }
                    if constexpr (G > 1) {
                        for (uint32_t m = C; m < static_cast<uint32_t>(kSg); m <<= 1) {
                            acc += shuffle_xor(sg, acc, m);
                        }
                    }
                    for (int32_t r = grp; r < L; r += G) {
                        Zl[(s + r) * C + col] -= tau * V(r, k) * acc;
                    }
                    sycl::group_barrier(sg);
                }

                for (int32_t idx = lid; idx < n * C; idx += kSg) {
                    const int32_t r = idx / C;
                    const int32_t c = idx - r * C;
                    if (c0 + c < ncols) Z(r, c0 + c) = Zl[idx];
                }
            });
    });

    return ctx.get_event();
}

// True for any spelling a user would plausibly write to mean "off", matched
// case-insensitively. See the call site in unmqr_hb2st for why this is local
// rather than a widening of util/env.hh's env_falsy.
static bool sb2st_wave_disabled(const char* v) {
    if (!v || !*v) return false;  // unset is not "off" -- the default is on
    std::string s(v);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s == "0" || s == "false" || s == "off" || s == "no" || s == "n" ||
           s == "disable" || s == "disabled";
}

template <Backend B, typename T>
Event unmqr_hb2st(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& v_in,
                  const VectorView<T>& tau_in,
                  const MatrixView<T, MatrixFormat::Dense>& z_io,
                  int32_t n,
                  int32_t kd,
                  Span<const int32_t> starts,
                  Span<const int32_t> lens,
                  Span<const int32_t> waves) {
    if (!ctx.in_order()) {
        throw std::runtime_error("unmqr_hb2st: requires an in-order Queue");
    }
    const int32_t nrefl = static_cast<int32_t>(starts.size());
    const int32_t batch = static_cast<int32_t>(z_io.batch_size());
    const int32_t ncols = static_cast<int32_t>(z_io.cols());
    if (nrefl <= 0 || batch <= 0 || ncols <= 0 || n <= 0) return ctx.get_event();

    // Preferred path: resident tile + wave-parallel reflectors. Set
    // BATCHLAS_SB2ST_BACK_WAVE=0 to fall through to the single-sub-group tiled
    // kernel below (kept for comparison and as a fallback if the tile does not
    // fit in local memory).
    //
    // Read against a local case-folded disable set rather than through
    // env_falsy/env_int_or. This knob used to be parsed with a local atoi, under
    // which every non-numeric spelling collapsed to 0, so "=off" and "=false"
    // disabled the wave path; routing it through the shared helpers inverted
    // that, because env_falsy matches only {0,false,FALSE,off,OFF} and
    // env_int_or hands an unparseable value back as the fallback 1 -- so
    // "=False", "=Off" and "=no" silently turned the wave path *on*. The cost of
    // that is not a wasted flag: a wave-vs-tiled A/B driven by such a spelling
    // measures the wave kernel against itself and reports the two paths as
    // identical.
    //
    // env_falsy is deliberately not widened to fix this. Its contract is "exactly
    // the spellings the six parsers it replaced accepted", and this is the only
    // knob that wants more; broadening it would quietly change every other call
    // site's reading of the same strings.
    //
    // Anything not in the set enables the wave path, including a typo. That is
    // the fail-open direction on purpose: a mistyped value must not silently cost
    // the fast path, and the spellings a user would actually write to mean "off"
    // are all here.
    const bool want_wave = !sb2st_wave_disabled(std::getenv("BATCHLAS_SB2ST_BACK_WAVE"));
    if (want_wave) {
        const size_t lmem = ctx->get_device().get_info<sycl::info::device::local_mem_size>();
        const size_t per_col = static_cast<size_t>(n) * sizeof(T);
        const int32_t num_waves = static_cast<int32_t>(waves.size()) - 1;

        // With S sub-groups the tile is shared by 32*S threads, so a wider tile
        // no longer costs occupancy the way it did at S=1; C is chosen to keep
        // the footprint near a budget rather than as small as possible. Measured
        // best at 8 columns for every n tried (see the subs table below).
        // Precedence: the legacy env knob forces a value, then the tuned
        // constant (0 = no opinion), then the budget heuristic below.
        int tile = env_positive_int_or("BATCHLAS_SB2ST_BACK_TILE_W", 0);
        if (tile <= 0) tile = tuning::sb2st_back_tile_for_n(n);
        if (tile <= 0) {
            constexpr size_t kTargetLocalBytes = 32768;
            tile = 1;
            while (tile < 8 && per_col * static_cast<size_t>(tile * 2) <= kTargetLocalBytes) {
                tile <<= 1;
            }
        }
        while (tile > 1 && per_col * static_cast<size_t>(tile) > lmem) tile >>= 1;

        // Sub-groups per work-group. More than a wave holds just leaves
        // sub-groups idle at every barrier, and fewer serialises the wave, so
        // this tracks the mean wave width (~n/2kd, but read off the schedule
        // rather than assumed).
        //
        // Back-transform alone, RTX 4090, float, ms (rows subs, cols the four
        // benchmark shapes; tile=8):
        //           256/1024 512/512 1024/128 1024/256   mean wave
        //   subs=8      17.1    51.0    123.8    245.0    8.5 / 8.5 / 16.5
        //   subs=16     20.6    58.5    103.5    207.1
        // -- 8 wins where waves hold ~8, 16 wins where they hold ~16.
        int subs = env_positive_int_or("BATCHLAS_SB2ST_BACK_SUBS", 0);
        if (subs <= 0) subs = tuning::sb2st_back_subs_for_n(n);
        if (subs <= 0) {
            const int32_t avg = (num_waves > 0) ? (nrefl + num_waves - 1) / num_waves : 1;
            subs = (avg >= 12) ? 16 : (avg >= 6 ? 8 : 4);
        }

        if (num_waves > 0 && per_col * static_cast<size_t>(tile) <= lmem) {
            // Caller-owned, like starts/lens: a buffer allocated here would be
            // freed at return while the kernel is still running.
            const int32_t* wp = waves.data();

            // C x S combinations are instantiated explicitly; anything else
            // falls through to the tiled kernel.
            #define BL_WAVE_CASE(CC, SS)                                              \
                if (tile == (CC) && subs == (SS))                                     \
                    return unmqr_hb2st_wave<B, T, (CC), (SS)>(                        \
                        ctx, v_in, tau_in, z_io, n, starts.data(), lens.data(), wp,   \
                        num_waves);
            BL_WAVE_CASE(1, 8) BL_WAVE_CASE(2, 8) BL_WAVE_CASE(4, 8) BL_WAVE_CASE(8, 8)
            BL_WAVE_CASE(1, 4) BL_WAVE_CASE(2, 4) BL_WAVE_CASE(4, 4) BL_WAVE_CASE(8, 4)
            BL_WAVE_CASE(1, 16) BL_WAVE_CASE(2, 16) BL_WAVE_CASE(4, 16) BL_WAVE_CASE(8, 16)
            #undef BL_WAVE_CASE
        }
    }

    // Prefer the resident-tile kernel: pick the widest column tile whose local
    // footprint fits, so Z is loaded and stored once instead of once per
    // reflector. Falls back to the streaming kernel below only if even a single
    // column does not fit (very large n).
    {
        // Two competing costs set the tile width C.
        //
        // Holding a column of Z resident already gives the full ~n/2 reuse, so
        // reuse does NOT grow with C. What does grow is amortisation of the V
        // reads: V is re-read once per column tile, and at C=1 that alone is
        // ~292 GB at n=1024/batch=128, which dominates. Pushing C up shrinks
        // that but grows the local footprint (n*C*sizeof(T)) and so cuts
        // occupancy -- these are 32-thread work-groups, so a 32 KB tile is
        // ~1 block/SM.
        //
        // Measured on RTX 4090 (back-transform alone, ms):
        //         C=0(stream)  C=1   C=2   C=4   C=8   C=16
        //   n=256/b=1024  39.9  109.2  44.8  25.9  27.1   42.8
        //   n=512/b=512  107.6  184.6 107.0 109.7 174.2  238.3
        //   n=1024/b=128 420.7  379.5 332.2 396.2 553.6 1391.4
        //   n=1024/b=256 913.5  752.0 660.5 786.1 1105.7 2782.9
        //
        // The optimum tracks a ~8 KB footprint, capped at 4 columns.
        constexpr size_t kTargetLocalBytes = 8192;
        constexpr int kMaxTile = 4;
        const size_t lmem = ctx->get_device().get_info<sycl::info::device::local_mem_size>();
        const size_t per_col = static_cast<size_t>(n) * sizeof(T);
        int want = 1;
        while (want < kMaxTile && per_col * static_cast<size_t>(want * 2) <= kTargetLocalBytes) {
            want <<= 1;
        }
        if (const char* ev = std::getenv("BATCHLAS_SB2ST_BACK_TILE")) {
            const int f = std::atoi(ev);
            if (f == 0) goto streaming;   // 0 selects the streaming kernel
            if (f > 0) want = f;
        }
        int tile = 0;
        for (int c = want; c >= 1; c >>= 1) {
            if (per_col * static_cast<size_t>(c) <= lmem) { tile = c; break; }
        }
        switch (tile) {
            case 32: return unmqr_hb2st_tiled<B, T, 32>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            case 16: return unmqr_hb2st_tiled<B, T, 16>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            case 8:  return unmqr_hb2st_tiled<B, T, 8>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            case 4:  return unmqr_hb2st_tiled<B, T, 4>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            case 2:  return unmqr_hb2st_tiled<B, T, 2>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            case 1:  return unmqr_hb2st_tiled<B, T, 1>(ctx, v_in, tau_in, z_io, n, nrefl, starts.data(), lens.data());
            default: break;
        }
    }

streaming:
    const int32_t col_chunks = (ncols + kBackCols - 1) / kBackCols;
    const int32_t num_wg = batch * col_chunks;

    auto Vv = v_in.kernel_view();
    auto Zv = z_io.kernel_view();
    const int32_t* starts_p = starts.data();
    const int32_t* lens_p = lens.data();

    ctx->submit([&](sycl::handler& h) {
        auto TAUv = tau_in;
        h.parallel_for<Sb2stHhBackKernel<B, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wg) * 32),
                              sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t lane = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t b = wg_id / col_chunks;
                const int32_t chunk = wg_id - b * col_chunks;
                if (b >= batch) return;

                const int32_t c0 = chunk * kBackCols;
                const int32_t c1 = sycl::min(c0 + kBackCols, ncols);

                auto Z = Zv.batch_item(b);
                auto V = Vv.batch_item(b);

                for (int32_t k = nrefl - 1; k >= 0; --k) {
                    const T tau = TAUv(k, b);
                    if (tau == T(0)) continue;
                    const int32_t s = starts_p[k];
                    const int32_t L = lens_p[k];

                    // L may exceed the 32 lanes (kd > 32), so walk the reflector
                    // in row-blocks and accumulate across them before the update.
                    for (int32_t c = c0; c < c1; ++c) {
                        T part = T(0);
                        for (int32_t r = lane; r < L; r += 32) {
                            part += conj_if(V(r, k)) * Z(s + r, c);
                        }
                        const T sum = group_sum(sg, part);
                        for (int32_t r = lane; r < L; r += 32) {
                            Z(s + r, c) -= tau * V(r, k) * sum;
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

#define SB2ST_HH_INSTANTIATE(back, fp)                                                  \
    template Event sytrd_sb2st_hh<back, BATCHLAS_UNPAREN fp>(                           \
        Queue&,                                                                          \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&,                \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&,                \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const VectorView<BATCHLAS_UNPAREN fp>&,                                          \
        Uplo,                                                                            \
        int32_t,                                                                         \
        const Span<std::byte>&);                                                         \
    template size_t sytrd_sb2st_hh_buffer_size<back, BATCHLAS_UNPAREN fp>(               \
        Queue&, int32_t, int32_t, int32_t);                                              \
    template Event unmqr_hb2st<back, BATCHLAS_UNPAREN fp>(                               \
        Queue&,                                                                          \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const VectorView<BATCHLAS_UNPAREN fp>&,                                          \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        int32_t,                                                                         \
        int32_t,                                                                         \
        Span<const int32_t>,                                                             \
        Span<const int32_t>,                                                             \
        Span<const int32_t>);

#define SB2ST_HH_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SB2ST_HH_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SB2ST_HH_INSTANTIATE_FOR_BACKEND
#undef SB2ST_HH_INSTANTIATE

} // namespace internal
} // namespace batchlas
