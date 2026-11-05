// SPDX-License-Identifier: MIT
#pragma once
// sb2st: a Hermitian (real: symmetric) matrix in lower band storage -> real tridiagonal by bulge
// chasing, inside the caller's kernel, on storage the caller prepared:
//   sb2st_givens_chase<Rot>(ex, n, kd, AB, ldab, C, WORK)   xSBTRD / xHBTRD's Givens chase on one
//                                                          sub-group partition
//   sb2st_hh_chase(ex, rows, A, n, kd, vl, wl, V, tau)      the two-stage Householder chase
//                                                          (xSYTRD_SB2ST), reflectors kept in V, tau
// Rot supplies the scalar rotation kernels (Rot::lartg, Rot::rot_pair, Rot::lar2v_pair); the
// vector forms are core's largv / lartv / lar2v. sb2st_detail holds what core has no form for:
//   StridedTeams<C>  C strided teams per sub-group (xor masks C, 2C, ...)
//   win_of, fn_vec   an index-form window, and a vector of write-through references
//   HermBandLower    a writable Hermitian matrix in LAPACK lower band storage
//   hemv_band, larf_right   team-per-row loops in lockstep, in T's own operators
//   rot_by<Rot>      one rotation through Rot::rot_pair
// Each element is the C++ expression of the kernel it was lifted from (std::complex operators,
// one expression per FMA chain), so a strict port stays bit-identical.

#include <array>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-invoke.hh>

namespace batchlas::device::inline v2 {

namespace sb2st_detail {

// C strided teams inside each sub-group: lane l is member l / C of team l mod C (SG / C members,
// xor distances C, 2C, ...): the column-tile layout of a resident Z tile. lane() and lid() are the
// member index (core Teams' convention). WG == SG lets the lane skip the modulo; barrier() is the
// sub-group's then, else the work-group's.
template <int C, int WG = 32, int SG = 32> struct StridedTeams {
    static_assert(detail::is_pow2(C) && C <= SG && WG % SG == 0, "sb2st::StridedTeams: C | SG | WG");
    static constexpr int size = SG / C, team_size = SG / C, nteams = C, sg_size = SG;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    sycl::nd_item<1> it;
    int lane_ = (WG == SG) ? static_cast<int>(it.get_local_linear_id()) : static_cast<int>(it.get_local_linear_id()) % SG;
    int grp_ = lane_ / C;
    int col_ = lane_ % C;

    BATCHLAS_DLA_INLINE int lane() const { return grp_; }
    BATCHLAS_DLA_INLINE int lid() const { return grp_; }
    BATCHLAS_DLA_INLINE int team() const { return col_; }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE sycl::sub_group sub() const { return it.get_sub_group(); }
    BATCHLAS_DLA_INLINE void barrier(Mem = Mem::Local) const {
        if constexpr (WG == SG) sycl::group_barrier(it.get_sub_group());
        else sycl::group_barrier(it.get_group());
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return butterfly<C, SG, bfly::Asc>(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return team_reduce(v, op);
    }
};

// Index-form window of any 2-D operand: element (i, j) is a(i0 + i, j0 + j), the pointer never
// moves (for operands without sub(): StridedMat).
template <class M> struct WinOf {
    using value_type = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;
    M a;
    int i0, j0, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(i0 + i, j0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE WinOf<M> win_of(const M& a, int i0, int j0, int m, int n) { return {a, i0, j0, m, n}; }

// x(t) = f(t), t < n, f returning a reference to write through (core fn_vec returns values). Each
// element keeps the caller's address expression, e.g. a band chase's rotation sequence
// AB(r, j1 + t kd1 + dc).
template <class F> struct FnVec {
    using value_type = std::remove_cvref_t<decltype(std::declval<const F&>()(0))>;
    F f;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DLA_INLINE FnVec<F> fn_vec(int n, F f) { return {f, n}; }

// A Hermitian (real: symmetric) matrix in LAPACK lower band storage, half-bandwidth kd:
// A(i, j), i >= j, lives at p[(i - j) + j * ld]. A(i, j) above the diagonal reads conj(A(j, i));
// outside the band it reads 0 and a write is dropped. Elements are BandRef proxies; get(i, j)
// reads a value.
template <class T> struct HermBandLower;
template <class T> struct BandRef {
    HermBandLower<T> b;
    int i, j;
    BATCHLAS_DLA_INLINE operator T() const { return b.get(i, j); }
    BATCHLAS_DLA_INLINE const BandRef& operator=(const T& v) const {
        b.set(i, j, v);
        return *this;
    }
    BATCHLAS_DLA_INLINE const BandRef& operator=(const BandRef& o) const {
        b.set(i, j, static_cast<T>(o));
        return *this;
    }
};
// Rows [i0, i0 + m) x columns [j0, j0 + n) of a band matrix.
template <class T> struct BandWin {
    using value_type = T;
    HermBandLower<T> b;
    int i0, j0, m, n;
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int r, int c) const { return {b, i0 + r, j0 + c}; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
// A(i0 : i0 + n, j): a column segment of a band matrix.
template <class T> struct BandVec {
    using value_type = T;
    HermBandLower<T> b;
    int i0, j, n;
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int k) const { return {b, i0 + k, j}; }
    BATCHLAS_DLA_INLINE int size() const { return n; }
    BATCHLAS_DLA_INLINE BandVec tail(int k) const { return {b, i0 + k, j, n - k}; }
};
template <class T> struct HermBandLower {
    using value_type = T;
    T* p;
    int ld;
    int kd;
    BATCHLAS_DLA_INLINE T get(int i, int j) const {
        if (i >= j) {
            const int r = i - j;
            return (r <= kd) ? p[r + j * ld] : T(0);
        }
        const int r = j - i;
        return (r <= kd) ? arith_t<arith::Native, T>::conj(p[r + i * ld]) : T(0);
    }
    BATCHLAS_DLA_INLINE void set(int i, int j, T v) const {
        if (i >= j) {
            const int r = i - j;
            if (r <= kd) p[r + j * ld] = v;
        } else {
            const int r = j - i;
            if (r <= kd) p[r + i * ld] = arith_t<arith::Native, T>::conj(v);
        }
    }
    BATCHLAS_DLA_INLINE BandRef<T> operator()(int i, int j) const { return {*this, i, j}; }
    BATCHLAS_DLA_INLINE BandWin<T> sub(int i0, int j0, int m, int n) const { return {*this, i0, j0, m, n}; }
    BATCHLAS_DLA_INLINE BandVec<T> col(int j, int i0, int m) const { return {*this, i0, j, m}; }
};
template <class V> inline constexpr bool is_band_win_v = false;
template <class T> inline constexpr bool is_band_win_v<BandWin<T>> = true;

// y := alpha A x + beta y for a window of a Hermitian band matrix (every entry readable through
// the mirror): one team per row, lanes stride the columns, lockstep (team_reduce reached by every
// lane when teams share a sub-group; core gemv<alg::TeamPerRow> stops at each team's bound).
template <Uplo UL, class Ex, class AS, class MA, class X, class BS, class Y>
    requires is_band_win_v<MA>
BATCHLAS_DLA_INLINE void hemv_band(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using T = typename MA::value_type;
    static_assert(UL == Uplo::Lower, "sb2st::hemv_band: lower band storage only");
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    for_each_team_owned<loop::Lockstep>(ex, m, [&](int r, bool live) BATCHLAS_DLA_BODY {
        T acc = T(0);
        if (live) {
            for (int j = ex.lid(); j < n; j += Ex::size) acc += static_cast<T>(a(r, j)) * static_cast<T>(x(j));
        }
        acc = ex.team_reduce(acc);
        if (live && ex.leader()) store(alpha, acc, beta, y(r));
    });
}

// C := C (I - tau v v^H) in T's own operators: each team owns whole rows of C, reduces C v with
// team_reduce in a lockstep loop and updates its own row; no barrier inside (core larf<Right> is
// one row per item, serial over the columns).
template <class Ex, class Tau, class VP, class MC> BATCHLAS_DLA_INLINE void larf_right(const Ex& ex, Tau tau, const VP& vp, const MC& c) {
    using T = typename MC::value_type;
    using N = arith_t<arith::Native, T>;
    detail::require_published<Ex, VP>();
    const auto& v = detail::unpub(vp);
    const int len = c.cols();
    for_each_team_owned<loop::Lockstep>(ex, c.rows(), [&](int o, bool live) BATCHLAS_DLA_BODY {
        T w = T(0);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) w += static_cast<T>(c(o, k)) * static_cast<T>(v(k));
        }
        w = tau * ex.team_reduce(w);
        if (live) {
            for (int k = ex.lid(); k < len; k += Ex::size) c(o, k) = static_cast<T>(c(o, k)) - w * N::conj(static_cast<T>(v(k)));
        }
    });
}

// One rotation (c, s) applied to the vector pair (x, y) (xROT through Rot::rot_pair).
template <class Rot, class Ex, class X, class Y, class Rc, class Sc>
BATCHLAS_DLA_INLINE void rot_by(const Ex& ex, const X& x, const Y& y, Rc c, const Sc& s) {
    for_each_owned(ex, x.size(), [&](int i) BATCHLAS_DLA_BODY { Rot::rot_pair(x(i), y(i), c, s); });
}

}  // namespace sb2st_detail

// xSBTRD / xHBTRD's bulge chase (lower band, kd >= 2) for one matrix on one sub-group partition
// executor pex (SgpPartition<P>): AB the band with leading dimension ldab (global or local), C (real)
// and WORK the rotation scratch of n + kd + 2 elements, zeroed by the caller. On return the band
// holds the tridiagonal in rows 0 and 1 (complex: the off-diagonal not yet made real).
template <class Rot, class Ex, class T, class R>
BATCHLAS_DLA_INLINE void sb2st_givens_chase(const Ex& pex, int n, int kd, T* AB, const int ldab, R* C, T* WORK) {
    const auto& partition = pex.part;
    const int kd1 = kd + 1;
    const int kdm1 = kd - 1;
    const int incx = ldab - 1;
    const int kdn = (n - 1 < kd) ? (n - 1) : kd;

    auto idx = [&](int r, int c) -> int { return r + c * ldab; };
    auto AB_at = [&](int r, int c) -> T& { return AB[idx(r, c)]; };

    int nr = 0;
    int j1 = kdn + 1;
    int j2 = 0;

    // The rotation sequence j = j1 + t kd1, t < cnt, read through at(j): C(j), WORK(j) and the band
    // entries AB(r, j + dc), addressed as the scalar loops address them.
    auto rseq = [&](int cnt, auto at) {
        return sb2st_detail::fn_vec(cnt, [at, j1 = j1, kd1](int t) BATCHLAS_DLA_BODY -> decltype(auto) {
            return at(j1 + t * kd1);
        });
    };
    auto Cj = [&](int j) BATCHLAS_DLA_BODY -> R& { return C[j]; };
    auto Wj = [&](int j) BATCHLAS_DLA_BODY -> T& { return WORK[j]; };
    auto band = [&](int r, int dc) {
        return [&AB_at, r, dc](int j) BATCHLAS_DLA_BODY -> T& { return AB_at(r, j + dc); };
    };

    if constexpr (Arith<T>::is_complex) {
        for_each_owned(pex, n, [&](int j) BATCHLAS_DLA_BODY { AB_at(0, j) = force_real(AB_at(0, j)); });
        group_barrier(partition);
    }

    for (int i = 0; i < n - 2; ++i) {
        for (int k = kdn; k >= 1; --k) {
            // Every lane holds the same scalar state for the partition's matrix.
            j1 += kdn;
            j2 += kdn;

            if (nr > 0) {
                // xLARGV over the rotations.
                largv<Rot>(pex, rseq(nr, band(kd, -kd1)), rseq(nr, Wj), rseq(nr, Cj));
                group_barrier(partition);

                if (nr > 2 * kd - 1) {
                    // xLARTV per band diagonal l (l sequential, rotations parallel); the rotations
                    // are independent, so one barrier after all l.
                    for (int l = 0; l < kd - 1; ++l) {
                        lartv<Rot>(pex, rseq(nr, band(kd - 1 - l, -kd1 + l + 1)), rseq(nr, band(kd - l, -kd1 + l + 1)), rseq(nr, Cj),
                                   rseq(nr, Wj));
                    }
                    group_barrier(partition);
                } else {
                    // Each lane applies its own rotations along the band diagonal (xROT).
                    if (kdm1 > 0) {
                        for_each_owned(pex, nr, [&](int t) BATCHLAS_DLA_BODY {
                            const int j = j1 + t * kd1;
                            sb2st_detail::rot_by<Rot>(Thread{}, vec(AB + idx(kd - 1, j - kd), kdm1, incx), vec(AB + idx(kd, j - kd), kdm1, incx),
                                                      C[j], WORK[j]);
                        });
                    }
                    group_barrier(partition);
                }
            }

            // Scalar bulge creation and state update on the leader; (nr, j1) broadcast.
            const auto nr_j1 = ::batchlas::invoke_one_broadcast(partition, [&]() {
                int nr_out = nr;
                int j1_out = j1;
                if (k > 1) {
                    if (k <= n - i - 1) {
                        const T f = AB_at(k - 1, i);
                        const T g = AB_at(k, i);
                        R ct = R(0);
                        T st = T(0);
                        T rt = T(0);
                        Rot::lartg(f, g, ct, st, rt);
                        C[i + k] = ct;
                        WORK[i + k] = st;
                        AB_at(k - 1, i) = rt;

                        if (k > 2) {
                            const R ct2 = C[i + k];
                            const T st2 = WORK[i + k];
                            sb2st_detail::rot_by<Rot>(Thread{}, vec(AB + idx(k - 2, i + 1), k - 2, ldab - 1),
                                                      vec(AB + idx(k - 1, i + 1), k - 2, ldab - 1), ct2, st2);
                        }
                    }
                    nr_out += 1;
                    j1_out = j1_out - kdn - 1;
                }
                return std::array<int, 2>{nr_out, j1_out};
            });
            nr = nr_j1[0];
            j1 = nr_j1[1];

            if (nr > 0) {
                // xLAR2V: the two-sided rotation of each 2x2 diagonal block.
                lar2v<Rot>(pex, rseq(nr, band(0, -1)), rseq(nr, band(0, 0)), rseq(nr, band(1, -1)), rseq(nr, Cj), rseq(nr, Wj));
                group_barrier(partition);

                // xHBTRD: conjugate WORK(J1) before the right application.
                if constexpr (Arith<T>::is_complex) {
                    const auto ws = rseq(nr, Wj);
                    map(pex, ws, [](const T& w) { return arith_t<arith::Native, T>::conj(w); }, ws);
                    group_barrier(partition);
                }

                if (nr > 2 * kd - 1) {
                    for (int l = 0; l < kd - 1; ++l) {
                        const int nrt = (j2 + l + 2 > n) ? (nr - 1) : nr;
                        if (nrt > 0) {
                            lartv<Rot>(pex, rseq(nrt, band(l + 2, -1)), rseq(nrt, band(l + 1, 0)), rseq(nrt, Cj), rseq(nrt, Wj));
                        }
                    }
                    group_barrier(partition);
                } else {
                    // Each lane applies its own rotations (0 .. nr - 2) from the right (xROT).
                    const int main_nrot = nr - 1;
                    if (kdm1 > 0 && main_nrot > 0) {
                        for_each_owned(pex, main_nrot, [&](int t) BATCHLAS_DLA_BODY {
                            const int j = j1 + t * kd1;
                            sb2st_detail::rot_by<Rot>(Thread{}, vec(AB + idx(2, j - 1), kdm1), vec(AB + idx(1, j), kdm1), C[j], WORK[j]);
                        });
                    }

                    // The tail rotation ("last") has its own length lend: once, on the leader.
                    if (pex.leader()) {
                        const int j1end = j1 + kd1 * (nr - 2);
                        const int lend = (n - j2 - 1 < kdm1) ? (n - j2 - 1) : kdm1;
                        const int last = j1end + kd1;
                        if (lend > 0) {
                            sb2st_detail::rot_by<Rot>(Thread{}, vec(AB + idx(2, last - 1), lend), vec(AB + idx(1, last), lend), C[last],
                                                      WORK[last]);
                        }
                    }
                    group_barrier(partition);
                }
            }

            // State update from j2; (nr, j2) broadcast.
            const auto nr_j2 = ::batchlas::invoke_one_broadcast(partition, [&]() {
                int nr_out = nr;
                int j2_out = j2;
                if (j2_out + kdn >= n) {
                    nr_out -= 1;
                    j2_out = j2_out - kdn - 1;
                }
                return std::array<int, 2>{nr_out, j2_out};
            });
            nr = nr_j2[0];
            j2 = nr_j2[1];

            if (nr > 0) {
                // The next fill: WORK(j + kd) = WORK(j) AB(kd, j), then AB(kd, j) *= C(j).
                for_each_owned(pex, nr, [&](int t) BATCHLAS_DLA_BODY {
                    const int j = j1 + t * kd1;
                    WORK[j + kd] = WORK[j] * AB_at(kd, j);
                    AB_at(kd, j) = T(C[j]) * AB_at(kd, j);
                });
                group_barrier(partition);
            }
        }
    }
}

// The two-stage Householder chase (xSYTRD_SB2ST's sweep schedule, one sweep at a time) for one
// matrix on a work-group: A the working band (bandwidth >= 2 kd, the bulge fill fits), vl / wl
// kd-element local buffers, reflector k stored as V(:, k) = (v; 0) (v(0) = 1) and tau(k). rows: the
// team-per-row executor of the window updates. The reflector count follows the host-side schedule
// (build_sb2st_hh_schedule), which the back-transform indexes V by. Exit: A holds the tridiagonal.
template <class Ex, class Rows, class T, class VL, class WL, class VOut, class TauOut>
BATCHLAS_DLA_INLINE void sb2st_hh_chase(const Ex& ex, const Rows& rows, const sb2st_detail::HermBandLower<T>& A, int n, int kd, const VL& vloc,
                                        const WL& wloc, const VOut& Vb, const TauOut& tau_out) {
    using Real = real_t<T>;
    const int32_t kd_i = kd;

    // W <- H W H^H on the principal block [a..bb], H = I - tau v v^H, v in vloc (zhetd2's update):
    // p = W v, kappa = v^H p, w = conj(tau) p - (|tau|^2 kappa / 2) v, W -= w v^H + v w^H on the
    // lower triangle (diagonal kept real).
    auto two_sided = [&](int32_t a, int32_t bb, T tau) {
        const int32_t m = bb - a + 1;
        if (m <= 0) return;
        const auto W = A.sub(a, a, m, m);
        const auto v = vec(vloc, m);
        const auto w = vec(wloc, m);
        sb2st_detail::hemv_band<Uplo::Lower>(rows, One{}, W, assume_published(v), Zero{}, w);
        sync(ex, Mem::Global);

        const T kappa = dotc<Scope::Auto, arith::Native>(ex, v, w);
        const T coef = tau * ::batchlas::device::conj(tau) * kappa * T(Real(0.5));
        // The loop is written out: through for_each_owned ptxas unrolls it 4x (latch layout).
        for (int i = ex.lid(); i < m; i += Ex::size)
            w(i) = ::batchlas::device::conj(tau) * static_cast<T>(w(i)) - coef * static_cast<T>(v(i));
        sync(ex, Mem::Global);

        her2<Part::Lower, alg::TeamPerRow, arith::Native>(rows, NegOne{}, assume_published(w), assume_published(v), W);
        sync(ex, Mem::Global);
    };

    // B <- B (I - tau v v^H) on rows [r0..r1] x columns [c0..c1].
    auto right_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
        if (r0 > r1 || c0 > c1) return;
        const auto v = assume_published(vec(vloc, c1 - c0 + 1));
        sb2st_detail::larf_right(rows, tau, v, A.sub(r0, c0, r1 - r0 + 1, c1 - c0 + 1));
        sync(ex, Mem::Global);
    };

    // C <- (I - tau v v^H) C on rows [r0..r1] x columns [c0..c1].
    auto left_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
        if (r0 > r1 || c0 > c1) return;
        const auto v = assume_published(vec(vloc, r1 - r0 + 1));
        larf<Side::Left, arith::Native, larf_form::ScaleW, loop::Lockstep>(rows, tau, v, A.sub(r0, c0, r1 - r0 + 1, c1 - c0 + 1));
        sync(ex, Mem::Global);
    };

    // larfg on A(r0:r1, col): v -> vloc and V(:, slot); the column becomes (beta, 0...).
    auto make_reflector_body = [&](int32_t col, int32_t r0, int32_t r1, int32_t slot) -> T {
        const int32_t m = r1 - r0 + 1;
        const auto y = A.col(col, r0, m);  // (alpha; x)
        const auto v = vec(vloc, m);
        // xLARFG scalars from alpha and an unscaled ||x|| (abs2 per element, one reduction).
        const auto x = fn_vec(m - 1, [A, col, r0](int k) BATCHLAS_DLA_BODY { return A.get(r0 + 1 + k, col); });
        const Real xnorm = sycl::sqrt(sumsq<alg::Unfused>(ex, x));
        const T alpha = y(0);
        const auto h = larfg_scalars<conv::Phase>(alpha, xnorm, m);

        // v = (1; scale x), read before the column is overwritten below.
        if (ex.leader()) v(0) = T(1);
        for_each_owned(ex, m - 1, [&](int k) BATCHLAS_DLA_BODY { v(k + 1) = (h.tau == T(0)) ? T(0) : static_cast<T>(y(k + 1)) * h.scale; });
        sync(ex, Mem::Global);

        if (ex.leader()) y(0) = h.beta;
        for_each_owned(ex, m - 1, [&](int k) BATCHLAS_DLA_BODY { y(k + 1) = T(0); });
        const auto vs = mat(Vb).col(slot).head(kd_i);  // V(:, slot) = (v; 0)
        for_each_owned(ex, kd_i, [&](int i) BATCHLAS_DLA_BODY { vs(i) = (i < m) ? static_cast<T>(v(i)) : T(0); });
        if (ex.leader()) tau_out(slot) = h.tau;
        sync(ex, Mem::Global);
        return h.tau;
    };
    // Complex: out of line, as the legacy inliner left it; inlined at both sites it cost REG 70->119.
    auto make_reflector_ool = [&](int32_t col, int32_t r0, int32_t r1, int32_t slot)
                                  __attribute__((noinline)) -> T { return make_reflector_body(col, r0, r1, slot); };
    auto make_reflector = [&](int32_t col, int32_t r0, int32_t r1, int32_t slot) -> T {
        if constexpr (Arith<T>::is_complex) return make_reflector_ool(col, r0, r1, slot);
        else return make_reflector_body(col, r0, r1, slot);
    };

    // The sequential chase schedule; the reflector counter must match build_sb2st_hh_schedule().
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
}

}  // namespace batchlas::device::inline v2
