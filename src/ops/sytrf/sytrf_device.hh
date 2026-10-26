#pragma once

// Bunch-Kaufman LDL^T as group-cooperative device code: LAPACK ?sytf2 (factor) and ?sytrs (solve),
// written once for the lower form. Uplo::Upper runs the same steps on the index-reversed matrix
// (logical i is physical n-1-i): the reversal maps ?sytf2's upper loop onto its lower loop step for
// step, so the stored factor, D and ipiv are LAPACK's for either uplo, and only the referenced
// triangle is ever read or written. evidence: docs/perf/sytrf.md#sytrf-the-reversal-map
//
// Every lane of the group must call these with the same arguments. Each pivot decision is lane 0's,
// broadcast (Group::bcast): a decision that differed between lanes (a NaN compare, items aliased by
// a short stride) would deadlock the next barrier. With it every loop below is bounded by n and
// every lane runs the same barriers, whatever the data. evidence: docs/perf/sytrf.md#sytrf-termination

#include <src/sycl/device_scalar.hh>
#include <batchlas/util/group-collectives.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <limits>

namespace batchlas::sytrf_device {

template <typename D> struct RealOf { using type = D; };
template <typename R> struct RealOf<sycl_device::Cx<R>> { using type = R; };
template <typename D> using real_t = typename RealOf<D>::type;

// LAPACK's pivot metric (csytf2 uses CABS1, not the modulus).
template <typename D>
inline real_t<D> cabs1(D a) {
    if constexpr (sycl_device::dev_is_complex_v<D>) return sycl::fabs(a.re) + sycl::fabs(a.im);
    else return sycl::fabs(a);
}

template <typename D>
inline D add(D a, D b) {
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{a.re + b.re, a.im + b.im};
    else return a + b;
}

template <typename D>
inline D neg(D a) {
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{-a.re, -a.im};
    else return -a;
}

template <typename D>
inline D zero() {
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{0, 0};
    else return D(0);
}

template <typename D>
inline D nan_of() {
    using R = real_t<D>;
    constexpr R q = std::numeric_limits<R>::quiet_NaN();
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{q, q};
    else return q;
}

template <typename D>
inline void swap(D& a, D& b) {
    const D t = a;
    a = b;
    b = t;
}

// The logical lower triangle over storage: (i, j) with i >= j is physical (map(i), map(j)).
template <typename D>
struct Tile {
    D* p;
    std::ptrdiff_t ld;
    int n;
    bool rev;
    int map(int i) const { return rev ? n - 1 - i : i; }
    D& operator()(int i, int j) const {
        return p[static_cast<std::ptrdiff_t>(map(i)) + static_cast<std::ptrdiff_t>(map(j)) * ld];
    }
};

template <typename D>
struct Vec {
    D* p;
    int n;
    bool rev;
    D& operator()(int i) const { return p[rev ? n - 1 - i : i]; }
};

// Group reductions with a deterministic order: a sub-group butterfly, then the sub-group results in
// sub-group order through `nsg` slots of local scratch. Every lane returns the same value.
template <typename R>
struct Group {
    sycl::nd_item<1> it;
    R* sv;
    R* sv2;
    int* si;
    int lid;
    int nl;
    bool sg_only = false;  // nl == 32 run by sub-group 0 of a wider work-group: sub-group collectives

    void bar() const {
        if (sg_only) sycl::group_barrier(it.get_sub_group());
        else sycl::group_barrier(it.get_group());
    }
    int bcast(int v) const {
        if (sg_only) return batchlas::portable::group_broadcast(it.get_sub_group(), v, 0);
        return batchlas::portable::group_broadcast(it.get_group(), v, 0);
    }

    static bool better(R ov, int oi, R bv, int bi, bool prefer_low) {
        return ov > bv || (ov == bv && (prefer_low ? oi < bi : oi > bi));
    }

    void argmax(R& v, int& idx, bool prefer_low) const {
        auto sg = it.get_sub_group();
        for (unsigned off = 16; off > 0; off >>= 1) {
            const R ov = sycl::permute_group_by_xor(sg, v, off);
            const int oi = sycl::permute_group_by_xor(sg, idx, off);
            if (better(ov, oi, v, idx, prefer_low)) { v = ov; idx = oi; }
        }
        const int nsg = nl / 32;
        if (nsg == 1) return;
        if (sg.leader()) {
            sv[sg.get_group_linear_id()] = v;
            si[sg.get_group_linear_id()] = idx;
        }
        bar();
        v = sv[0];
        idx = si[0];
        for (int s = 1; s < nsg; ++s)
            if (better(sv[s], si[s], v, idx, prefer_low)) { v = sv[s]; idx = si[s]; }
        bar();
    }

    R sum_real(R v, R* slots) const {
        auto sg = it.get_sub_group();
        for (unsigned off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
        const int nsg = nl / 32;
        if (nsg == 1) return v;
        if (sg.leader()) slots[sg.get_group_linear_id()] = v;
        bar();
        v = slots[0];
        for (int s = 1; s < nsg; ++s) v += slots[s];
        bar();
        return v;
    }

    template <typename D>
    D sum(D v) const {
        if constexpr (sycl_device::dev_is_complex_v<D>) return D{sum_real(v.re, sv), sum_real(v.im, sv2)};
        else return sum_real(v, sv);
    }
};

template <typename R>
inline R finite_or_inf(R v) {
    return sycl::isnan(v) ? std::numeric_limits<R>::infinity() : v;
}

// ?sytf2's first test, lane 0's verdict: a zero (or NaN) column, a 1 x 1 pivot without interchange,
// or the row search. ?sytf2 and ?lasyf share both tests; only where the columns live differs.
inline constexpr int kNeedRow = 0;
inline constexpr int kZero = 1;
inline constexpr int kKeep = 2;

template <typename R>
inline int pivot_stage(const Group<R>& g, R absakk, R colmax, R alpha) {
    return g.bcast(sycl::fmax(absakk, colmax) == R(0) || sycl::isnan(absakk) ? kZero
                   : absakk >= alpha * colmax                                ? kKeep
                                                                             : kNeedRow);
}

template <typename R>
inline void pivot_choice(const Group<R>& g, R absakk, R colmax, R rowmax, R absimax, R alpha, int k, int imax,
                         int& kp, int& kstep) {
    kp = imax;
    kstep = 1;
    if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
    else if (!(absimax >= alpha * rowmax)) kstep = 2;
    const int code = g.bcast(kp * 2 + kstep - 1);
    kp = code >> 1;
    kstep = (code & 1) + 1;
}

// ?sytf2, lower form, on the logical tile. `piv_out(phys_k, value)` stores LAPACK's ipiv entry
// (1-based; a 2x2 pivot writes -(p+1) at both of its positions); lane 0 calls it. Returns info.
template <typename D, typename R, typename PivOut>
inline int ldl_factor(const Tile<D>& S, int n, bool rev, const Group<R>& g, PivOut&& piv_out) {
    const R alpha = (R(1) + sycl::sqrt(R(17))) / R(8);
    const int lid = g.lid, nl = g.nl;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    int info = 0;
    int k = 0;
    while (k < n) {
        int kstep = 1;
        int kp = k;
        const R absakk = cabs1(S(k, k));
        // ?sytf2's I?AMAX takes the first maximum in storage order: the smallest physical row,
        // which for the reversed (upper) form is the largest logical one.
        R cv = R(-1);
        int ci = -1;
        for (int i = k + 1 + lid; i < n; i += nl) {
            const R v = finite_or_inf(cabs1(S(i, k)));
            if (Group<R>::better(v, i, cv, ci, !rev)) { cv = v; ci = i; }
        }
        g.argmax(cv, ci, !rev);
        const R colmax = cv < R(0) ? R(0) : cv;
        const int imax = ci;
        const int stage = pivot_stage(g, absakk, colmax, alpha);
        if (stage == kNeedRow) {
            R rv = R(-1);
            int ri = -1;
            for (int j = k + lid; j < n; j += nl) {
                if (j == imax) continue;
                const R v = finite_or_inf(cabs1(j > imax ? S(j, imax) : S(imax, j)));
                if (Group<R>::better(v, j, rv, ri, true)) { rv = v; ri = j; }
            }
            g.argmax(rv, ri, true);
            pivot_choice(g, absakk, colmax, rv, cabs1(S(imax, imax)), alpha, k, imax, kp, kstep);
        }
        if (stage == kZero) {
            if (info == 0) info = phys(k) + 1;
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {
                for (int i = kp + 1 + lid; i < n; i += nl) swap(S(i, kk), S(i, kp));
                for (int j = kk + 1 + lid; j < kp; j += nl) swap(S(j, kk), S(kp, j));
                if (lid == 0) {
                    swap(S(kk, kk), S(kp, kp));
                    if (kstep == 2) swap(S(k + 1, k), S(kp, k));
                }
            }
            g.bar();
            if (kstep == 1) {
                const D mr1 = neg(sycl_device::dev_recip(S(k, k)));
                for (int j = k + 1; j < n; ++j) {
                    const D t = sycl_device::dev_mul(mr1, S(j, k));
                    for (int i = j + lid; i < n; i += nl) S(i, j) = add(S(i, j), sycl_device::dev_mul(S(i, k), t));
                }
                g.bar();
                const D r1 = neg(mr1);
                for (int i = k + 1 + lid; i < n; i += nl) S(i, k) = sycl_device::dev_mul(r1, S(i, k));
            } else if (k < n - 2) {
                const D d21 = S(k + 1, k);
                const D d11 = sycl_device::dev_div(S(k + 1, k + 1), d21);
                const D d22 = sycl_device::dev_div(S(k, k), d21);
                const D t = sycl_device::dev_recip(sycl_device::dev_sub(sycl_device::dev_mul(d11, d22),
                                                                        sycl_device::dev_one<D>()));
                const D e21 = sycl_device::dev_div(t, d21);
                auto wk = [&](int j) {
                    return sycl_device::dev_mul(e21, sycl_device::dev_sub(sycl_device::dev_mul(d11, S(j, k)), S(j, k + 1)));
                };
                auto wkp1 = [&](int j) {
                    return sycl_device::dev_mul(e21, sycl_device::dev_sub(sycl_device::dev_mul(d22, S(j, k + 1)), S(j, k)));
                };
                for (int j = k + 2; j < n; ++j) {
                    const D a = wk(j), b = wkp1(j);
                    for (int i = j + lid; i < n; i += nl)
                        S(i, j) = sycl_device::dev_sub(sycl_device::dev_sub(S(i, j), sycl_device::dev_mul(S(i, k), a)),
                                                       sycl_device::dev_mul(S(i, k + 1), b));
                }
                g.bar();
                for (int i = k + 2 + lid; i < n; i += nl) {
                    const D a = wk(i), b = wkp1(i);
                    S(i, k) = a;
                    S(i, k + 1) = b;
                }
            }
        }
        if (lid == 0) {
            if (kstep == 1) {
                piv_out(phys(k), phys(kp) + 1);
            } else {
                piv_out(phys(k), -(phys(kp) + 1));
                piv_out(phys(k + 1), -(phys(kp) + 1));
            }
        }
        g.bar();
        k += kstep;
    }
    return info;
}

// ?lasyf's W, kRank = nb + 1 columns: logical row i at physical row map(i), panel column c at
// physical column c (Lower) or kRank-1-c (Upper), so W's physical column order is the order of the
// panel's physical columns in A and the trailing gemm takes both as plain sub-views.
template <typename D>
struct WTile {
    D* p;
    std::ptrdiff_t ld;
    int n;
    int ncol;
    bool rev;
    D& operator()(int i, int c) const {
        return p[static_cast<std::ptrdiff_t>(rev ? n - 1 - i : i) +
                 static_cast<std::ptrdiff_t>(rev ? ncol - 1 - c : c) * ld];
    }
};

// ?lasyf, lower form, on the uniform panel [k0, k0 + nb) of the logical matrix, starting at
// `start` (k0, or k0 + 1 when the previous panel's last 2 x 2 block took column k0). The step at
// the panel's last column is always taken, so a 2 x 2 block may take column k0 + nb (W's column nb):
// the panel ends at k0 + nb or k0 + nb + 1, and the trailing update has one uniform rank, nb + 1.
// W's columns the item did not compute are zero. Returns the end; *info as ldl_factor's.
// evidence: docs/perf/sytrf.md#sytrf-the-blocked-route
template <typename D, typename R, typename PivOut>
inline int ldl_panel(const Tile<D>& S, const WTile<D>& W, int n, int k0, int nb, int start, bool rev,
                     const Group<R>& g, int& info, PivOut&& piv_out) {
    const R alpha = (R(1) + sycl::sqrt(R(17))) / R(8);
    const int lid = g.lid, nl = g.nl;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    for (int i = k0 + lid; i < n; i += nl)
        for (int c = 0; c <= nb; ++c) W(i, c) = zero<D>();
    g.bar();
    // W(k:n, dst) := source - A(k:n, start:k) W(r, start-k0:k-k0): the panel's finished columns only.
    auto update_col = [&](int k, int dst, int r, auto&& source) {
        for (int i = k + lid; i < n; i += nl) {
            D acc = source(i);
            for (int l = start - k0; l < k - k0; ++l)
                acc = sycl_device::dev_sub(acc, sycl_device::dev_mul(S(i, k0 + l), W(r, l)));
            W(i, dst) = acc;
        }
        g.bar();
    };
    int k = start;
    while (k < k0 + nb) {
        const int c = k - k0;
        update_col(k, c, k, [&](int i) { return S(i, k); });
        int kstep = 1;
        int kp = k;
        const R absakk = cabs1(W(k, c));
        R cv = R(-1);
        int ci = -1;
        for (int i = k + 1 + lid; i < n; i += nl) {
            const R v = finite_or_inf(cabs1(W(i, c)));
            if (Group<R>::better(v, i, cv, ci, !rev)) { cv = v; ci = i; }
        }
        g.argmax(cv, ci, !rev);
        const R colmax = cv < R(0) ? R(0) : cv;
        const int imax = ci;
        const int stage = pivot_stage(g, absakk, colmax, alpha);
        if (stage == kNeedRow) {
            update_col(k, c + 1, imax, [&](int i) { return i < imax ? S(imax, i) : S(i, imax); });
            R rv = R(-1);
            int ri = -1;
            for (int j = k + lid; j < n; j += nl) {
                if (j == imax) continue;
                const R v = finite_or_inf(cabs1(W(j, c + 1)));
                if (Group<R>::better(v, j, rv, ri, true)) { rv = v; ri = j; }
            }
            g.argmax(rv, ri, true);
            pivot_choice(g, absakk, colmax, rv, cabs1(W(imax, c + 1)), alpha, k, imax, kp, kstep);
            if (kstep == 1 && kp != k)
                for (int i = k + lid; i < n; i += nl) W(i, c) = W(i, c + 1);
            g.bar();
        }
        if (stage == kZero) {
            if (info == 0) info = phys(k) + 1;
            for (int i = k + lid; i < n; i += nl) S(i, k) = W(i, c);
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {
                // The non-updated column kk moves to kp; the panel's finished columns and W swap
                // rows kk and kp (ldl_undo restores the columns' ?sytf2 form after the update).
                if (lid == 0) S(kp, kp) = S(kk, kk);
                for (int j = kk + 1 + lid; j < kp; j += nl) S(kp, j) = S(j, kk);
                for (int i = kp + 1 + lid; i < n; i += nl) S(i, kp) = S(i, kk);
                for (int l = start + lid; l < kk; l += nl) swap(S(kk, l), S(kp, l));
                for (int l = start - k0 + lid; l <= kk - k0; l += nl) swap(W(kk, l), W(kp, l));
            }
            g.bar();
            if (kstep == 1) {
                const D r1 = sycl_device::dev_recip(W(k, c));
                for (int i = k + lid; i < n; i += nl) S(i, k) = i == k ? W(k, c) : sycl_device::dev_mul(r1, W(i, c));
            } else {
                const D d21 = W(k + 1, c);
                const D d11 = sycl_device::dev_div(W(k + 1, c + 1), d21);
                const D d22 = sycl_device::dev_div(W(k, c), d21);
                const D t = sycl_device::dev_recip(sycl_device::dev_sub(sycl_device::dev_mul(d11, d22),
                                                                        sycl_device::dev_one<D>()));
                const D e21 = sycl_device::dev_div(t, d21);
                for (int j = k + 2 + lid; j < n; j += nl) {
                    const D wk = W(j, c), wk1 = W(j, c + 1);
                    S(j, k) = sycl_device::dev_mul(e21, sycl_device::dev_sub(sycl_device::dev_mul(d11, wk), wk1));
                    S(j, k + 1) = sycl_device::dev_mul(e21, sycl_device::dev_sub(sycl_device::dev_mul(d22, wk1), wk));
                }
                if (lid == 0) {
                    S(k, k) = W(k, c);
                    S(k + 1, k) = W(k + 1, c);
                    S(k + 1, k + 1) = W(k + 1, c + 1);
                }
            }
        }
        if (lid == 0) {
            if (kstep == 1) {
                piv_out(phys(k), phys(kp) + 1);
            } else {
                piv_out(phys(k), -(phys(kp) + 1));
                piv_out(phys(k + 1), -(phys(kp) + 1));
            }
        }
        g.bar();
        k += kstep;
    }
    // The last step's row search may have left column imax in W(:, nb) though the block was 1 x 1:
    // the uniform-rank gemm reads that column, so it must be zero unless a 2 x 2 block owns it.
    if (k == k0 + nb)
        for (int i = k0 + lid; i < n; i += nl) W(i, nb) = zero<D>();
    return k;
}

// ?lasyf's last loop: undo the panel's interchanges on its own finished columns [start, end), in
// reverse, so they hold ?sytf2's form. `logical_partner(j)` is ipiv's partner of logical column j
// (rows j / j-1 for a 2 x 2 block ending at j) and its sign. Runs after the trailing update.
template <typename D, typename R, typename PivIn>
inline void ldl_undo(const Tile<D>& S, int start, int end, const Group<R>& g, PivIn&& logical_partner) {
    int j = end - 1;
    while (j >= start) {
        const int jj = j;
        bool two = false;
        const int jp = logical_partner(j, two);
        if (two) --j;
        --j;
        if (jp != jj && j >= start)
            for (int l = start + g.lid; l <= j; l += g.nl) swap(S(jp, l), S(jj, l));
        g.bar();
    }
}

// ?sytrs for one right-hand side, lower form. `piv(phys_k)` reads LAPACK's ipiv entry.
template <typename D, typename R, typename PivIn>
inline void ldl_solve(const Tile<D>& S, int n, bool rev, const Vec<D>& b, const Group<R>& g, PivIn&& piv) {
    const int lid = g.lid, nl = g.nl;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    // A list sytrf cannot have written (|ipiv| outside [1, n], an unpaired 2 x 2 entry) would index
    // b out of bounds: that item's X is NaN instead. Every lane walks the list, lane 0 decides.
    bool ok = true;
    for (int k = 0; k < n && ok;) {
        const int raw = piv(phys(k));
        ok = raw != 0 && raw >= -n && raw <= n && (raw > 0 || (k + 1 < n && piv(phys(k + 1)) == raw));
        k += raw > 0 ? 1 : 2;
    }
    if (g.bcast(ok ? 1 : 0) == 0) {
        for (int i = lid; i < n; i += nl) b(i) = nan_of<D>();
        g.bar();
        return;
    }
    int k = 0;
    while (k < n) {
        const int raw = piv(phys(k));
        if (raw > 0) {
            const int kp = phys(raw - 1);
            if (lid == 0 && kp != k) swap(b(k), b(kp));
            g.bar();
            const D bk = b(k);
            g.bar();
            for (int i = k + 1 + lid; i < n; i += nl) b(i) = sycl_device::dev_sub(b(i), sycl_device::dev_mul(S(i, k), bk));
            if (lid == 0) b(k) = sycl_device::dev_mul(bk, sycl_device::dev_recip(S(k, k)));
            g.bar();
            k += 1;
        } else {
            const int kp = phys(-raw - 1);
            if (lid == 0 && kp != k + 1) swap(b(k + 1), b(kp));
            g.bar();
            const D bk = b(k), bk1 = b(k + 1);
            g.bar();
            for (int i = k + 2 + lid; i < n; i += nl)
                b(i) = sycl_device::dev_sub(sycl_device::dev_sub(b(i), sycl_device::dev_mul(S(i, k), bk)),
                                            sycl_device::dev_mul(S(i, k + 1), bk1));
            if (lid == 0) {
                const D akm1k = S(k + 1, k);
                const D akm1 = sycl_device::dev_div(S(k, k), akm1k);
                const D ak = sycl_device::dev_div(S(k + 1, k + 1), akm1k);
                const D denom = sycl_device::dev_sub(sycl_device::dev_mul(akm1, ak), sycl_device::dev_one<D>());
                const D bkm1 = sycl_device::dev_div(bk, akm1k);
                const D bkk = sycl_device::dev_div(bk1, akm1k);
                b(k) = sycl_device::dev_div(sycl_device::dev_sub(sycl_device::dev_mul(ak, bkm1), bkk), denom);
                b(k + 1) = sycl_device::dev_div(sycl_device::dev_sub(sycl_device::dev_mul(akm1, bkk), bkm1), denom);
            }
            g.bar();
            k += 2;
        }
    }
    k = n - 1;
    while (k >= 0) {
        const int raw = piv(phys(k));
        D p0 = zero<D>(), p1 = zero<D>();
        for (int i = k + 1 + lid; i < n; i += nl) {
            p0 = add(p0, sycl_device::dev_mul(S(i, k), b(i)));
            if (raw < 0) p1 = add(p1, sycl_device::dev_mul(S(i, k - 1), b(i)));
        }
        const D s0 = g.sum(p0);
        const D s1 = raw < 0 ? g.sum(p1) : zero<D>();
        g.bar();
        if (lid == 0) {
            b(k) = sycl_device::dev_sub(b(k), s0);
            if (raw < 0) b(k - 1) = sycl_device::dev_sub(b(k - 1), s1);
            const int kp = phys((raw > 0 ? raw : -raw) - 1);
            if (kp != k) swap(b(k), b(kp));
        }
        g.bar();
        k -= raw > 0 ? 1 : 2;
    }
}

}  // namespace batchlas::sytrf_device
