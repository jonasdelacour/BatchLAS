#pragma once

// The native batched GEQRF panel factorisation -- all of its device code.
// `geqr2_panel_device` is LAPACK ?GEQR2 over a `Tile` supplying `at(r, c)`,
// instantiated against a local_accessor (the CTA tier) and against a raw global
// pointer (the blocked tier's panel leaf). See docs/perf/qr.md.
//
// Deliberately NOT internal::larfg: that helper preserves alpha's phase, while
// clarfg/zlarfg -- and the vendors this must be a drop-in for -- return a REAL
// beta, and tau is a contract consumed by ormqr/orgqr/ormbr/sy2sb. It also
// returns tau = 0 for len <= 1 even for complex, where zlarfg returns a nonzero
// tau, so a short-final-panel test on a square REAL matrix guards nothing.
// evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
//
// Overflow is handled by ONE COMMON SCALE, s = max over the column of
// max(|Re|, |Im|), not xLARFG's rescaling loop: tau is scale-invariant, beta is
// s * O(1), and the reciprocal in v = x/(alpha-beta) has a division fallback.
//
// One work-group per matrix; the apply maps TEAM -> trailing column and LANE ->
// row over distinct columns, so it needs -- and must not have -- a work-group
// barrier. The required barriers are marked B1..B3 (B0/B4 are in the launcher).

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::geqrf_native {

using batchlas::sycl_device::Cx;

template <typename D> struct RealOf            { using type = D; };
template <typename R> struct RealOf<Cx<R>>     { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

template <typename D>
inline D dev_zero() {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{real_of<D>(0), real_of<D>(0)};
    } else {
        return D(0);
    }
}

template <typename D>
inline D dev_add(D a, D b) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{a.re + b.re, a.im + b.im};
    } else {
        return a + b;
    }
}

template <typename D>
inline real_of<D> dev_imag(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return a.im;
    } else {
        static_cast<void>(a);
        return real_of<D>(0);
    }
}

// max(|Re|, |Im|) -- NOT |a|, whose square-and-add is the overflow this avoids.
template <typename D>
inline real_of<D> dev_absmax(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return sycl::fmax(sycl::fabs(a.re), sycl::fabs(a.im));
    } else {
        return sycl::fabs(a);
    }
}

// |a/s|^2, formed by dividing FIRST. s is guaranteed positive by the caller.
template <typename D>
inline real_of<D> dev_abs2_scaled(D a, real_of<D> s) {
    using R = real_of<D>;
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        const R x = a.re / s;
        const R y = a.im / s;
        return sycl::fma(x, x, y * y);
    } else {
        const R x = a / s;
        return x * x;
    }
}

// The Householder scalars, LAPACK ?LARFG exactly. `s` scales the WHOLE column
// (alpha included); `ssq` is the TAIL's sum of |x/s|^2. Postcondition:
//   identity == true   ->  tau = 0, H = I, A(j,j) unchanged, v not written.
//   identity == false  ->  A(j,j) := beta (REAL even for complex),
//                          A(j+1:m, j) := x / (alpha - beta) = v's tail.
template <typename D>
struct LarfgScalars {
    D tau;
    D beta;      // written into A(j,j)
    D vfactor;   // multiply v by this, or divide by it -- see use_mul
    bool use_mul;
    bool identity;
};

template <typename D>
inline LarfgScalars<D> geqrf_larfg_scalars(D alpha, real_of<D> s, real_of<D> ssq) {
    using R = real_of<D>;
    LarfgScalars<D> out;
    out.tau = dev_zero<D>();
    out.beta = alpha;
    out.vfactor = batchlas::sycl_device::dev_one<D>();
    out.use_mul = true;
    out.identity = true;

    const R alphi = dev_imag(alpha);

    // xLARFG's early out: a zero tail AND a real alpha means H = I. For complex,
    // a nonzero Im(alpha) still needs a reflector and zlarfg returns a tau here.
    if (ssq == R(0) && alphi == R(0)) return out;
    // Redundant in exact arithmetic; it makes the divisions below safe by
    // construction.
    if (!(s > R(0))) return out;

    const R alphr = batchlas::sycl_device::dev_real(alpha);
    const R ar = alphr / s;
    const R ai = alphi / s;

    const R nrm = sycl::sqrt(sycl::fma(ar, ar, sycl::fma(ai, ai, ssq)));

    // beta = -SIGN(nrm, alphr). Fortran's SIGN returns +|x| for a +0 second
    // argument and alphr == 0 is reachable, so the test is `>= 0`, not `> 0`.
    const R beta_s = (alphr >= R(0)) ? -nrm : nrm;

    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        out.tau = D{(beta_s - ar) / beta_s, -ai / beta_s};
        out.beta = D{beta_s * s, R(0)};           // REAL beta -- the LAPACK convention
    } else {
        out.tau = (beta_s - ar) / beta_s;
        out.beta = beta_s * s;
    }

    // Formed FROM out.beta, not re-derived from beta_s: one source for beta, so
    // changing it cannot silently scale v by another reflector's divisor.
    const D d = batchlas::sycl_device::dev_sub(alpha, out.beta);

    const D r = batchlas::sycl_device::dev_recip(d);
    out.use_mul = batchlas::sycl_device::dev_isfinite(r) &&
                  !batchlas::sycl_device::dev_is_zero(r);
    out.vfactor = out.use_mul ? r : d;
    out.identity = false;
    return out;
}

// Tile accessors -- the only difference between the two residencies.
template <typename D>
struct GeqrfGlobalTile {
    D* p;
    int ld;
    D& at(int r, int c) const {
        return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

template <typename D, typename LocalAcc>
struct GeqrfLocalTile {
    LocalAcc a;
    int ld;
    // Unpadded ld is deliberate: both hot patterns walk consecutive rows within
    // a column, conflict-free in local memory for any ld.
    auto& at(int r, int c) const {
        return a[static_cast<std::size_t>(r) +
                 static_cast<std::size_t>(c) * static_cast<std::size_t>(ld)];
    }
};

// Sub-group sum, replicated to every lane. An XOR butterfly rather than
// sycl::reduce_over_group, whose CUDA path has had float-reduction limits.
template <typename D>
inline D geqrf_sg_sum(const sycl::sub_group& sg, D v) {
    const uint32_t lanes = static_cast<uint32_t>(sg.get_local_linear_range());
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        using R = real_of<D>;
        R re = v.re;
        R im = v.im;
        for (uint32_t off = lanes / 2; off > 0; off >>= 1) {
            re += sycl::permute_group_by_xor(sg, re, off);
            im += sycl::permute_group_by_xor(sg, im, off);
        }
        return D{re, im};
    } else {
        for (uint32_t off = lanes / 2; off > 0; off >>= 1) {
            v += sycl::permute_group_by_xor(sg, v, off);
        }
        return v;
    }
}

// LAPACK ?GEQR2 on an m x n tile, in place. The apply's conj(tau) is zgeqr2's,
// not a transcription slip -- the factorisation applies H^H while ormqr/orgqr
// reconstruct Q -- and dropping it passes every float and double test.
//
// `tau_ptr` is a GLOBAL pointer already offset to this matrix; reflector j lands
// at tau_ptr[j]. kmax lets a caller factor fewer reflectors than min(m, n).
template <typename D, typename Tile>
inline void geqr2_panel_device(sycl::nd_item<1> it, Tile A, int m, int n, int kmax,
                               D* tau_ptr) {
    using R = real_of<D>;

    const auto g = it.get_group();
    const auto sg = it.get_sub_group();
    const int wg = static_cast<int>(it.get_local_range(0));
    const int tid = static_cast<int>(it.get_local_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int nlanes = static_cast<int>(sg.get_local_linear_range());
    const int team = static_cast<int>(sg.get_group_linear_id());
    const int nteams = static_cast<int>(sg.get_group_linear_range());

    for (int j = 0; j < kmax; ++j) {
        // B1 -- the previous reflector's column writes become visible.
        sycl::group_barrier(g);

        const D alpha = A.at(j, j);

        R smax = dev_absmax(alpha);
        for (int r = j + 1 + tid; r < m; r += wg) {
            smax = sycl::fmax(smax, dev_absmax(A.at(r, j)));
        }
        smax = sycl::reduce_over_group(g, smax, sycl::maximum<R>());

        // The collective sits outside the `smax > 0` test: all items reach it.
        R ssq = R(0);
        if (smax > R(0)) {
            for (int r = j + 1 + tid; r < m; r += wg) {
                ssq += dev_abs2_scaled(A.at(r, j), smax);
            }
        }
        ssq = sycl::reduce_over_group(g, ssq, sycl::plus<R>());

        const LarfgScalars<D> h = geqrf_larfg_scalars<D>(alpha, smax, ssq);

        // B2 -- every work-item has read A(j,j) as `alpha`; item 0 overwrites it.
        sycl::group_barrier(g);

        if (!h.identity) {
            if (tid == 0) {
                A.at(j, j) = h.beta;
            }
            // No barrier needed here: work-item t writes exactly the rows it read.
            if (h.use_mul) {
                for (int r = j + 1 + tid; r < m; r += wg) {
                    A.at(r, j) = batchlas::sycl_device::dev_mul(A.at(r, j), h.vfactor);
                }
            } else {
                for (int r = j + 1 + tid; r < m; r += wg) {
                    A.at(r, j) = batchlas::sycl_device::dev_div(A.at(r, j), h.vfactor);
                }
            }
        }
        if (tid == 0) {
            tau_ptr[j] = h.tau;
        }

        // B3 -- beta and v are published to the teams about to read them.
        sycl::group_barrier(g);

        // `h.identity` is work-group-uniform (both its inputs are replicated by
        // the reductions), so this `continue` cannot desync B1/B2/B3.
        if (h.identity) continue;

        const D ctau = batchlas::sycl_device::dev_conj(h.tau);

        // TEAM -> column, LANE -> row. The trip count is sub-group- but NOT
        // work-group-uniform: no work-group barrier may appear in this region.
        for (int c = j + 1 + team; c < n; c += nteams) {
            // w = v^H A(j:m, c); v(j) = 1 implicitly but A(j,j) holds beta.
            D acc = (lane == 0) ? A.at(j, c) : dev_zero<D>();
            for (int r = j + 1 + lane; r < m; r += nlanes) {
                acc = dev_add(acc, batchlas::sycl_device::dev_mul(
                                       batchlas::sycl_device::dev_conj(A.at(r, j)),
                                       A.at(r, c)));
            }
            const D w = geqrf_sg_sum<D>(sg, acc);
            const D f = batchlas::sycl_device::dev_mul(ctau, w);

            if (lane == 0) {
                A.at(j, c) = batchlas::sycl_device::dev_sub(A.at(j, c), f);
            }
            for (int r = j + 1 + lane; r < m; r += nlanes) {
                A.at(r, c) = batchlas::sycl_device::dev_sub(
                    A.at(r, c), batchlas::sycl_device::dev_mul(f, A.at(r, j)));
            }
        }
    }
}

}  // namespace batchlas::geqrf_native
