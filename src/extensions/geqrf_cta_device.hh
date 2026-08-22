#pragma once

// The native batched GEQRF panel factorisation -- ALL of its device code.
//
// ONE algorithm, TWO residencies. `geqr2_panel_device` is an unblocked
// right-looking Householder QR (LAPACK ?GEQR2) written against a `Tile`
// abstraction that supplies `at(r, c)`. It is instantiated twice:
//
//   * against a local_accessor  -> the CTA tier, where the whole m x n matrix is
//                                  staged into local memory once, factorised
//                                  there, and stored once;
//   * against a raw global ptr  -> the BLOCKED tier's panel leaf, for panels too
//                                  tall to hold resident.
//
// Having one body rather than two is the point: the two residencies differ by a
// pointer type, so a correctness fix cannot land in one and miss the other. The
// two tiers are otherwise the two arms of RouteTable<Op::geqrf,T>.
//
// ---------------------------------------------------------------------------
// THE COMPLEX PHASE CONVENTION -- READ THIS BEFORE CHANGING larfg
// ---------------------------------------------------------------------------
// This file does NOT use internal::larfg (src/math-helpers.hh:359), and that is
// deliberate rather than an oversight. internal::larfg's complex branch returns
//
//     beta = -(alpha/|alpha|) * hypot(|alpha|, xnorm)          [COMPLEX beta]
//
// i.e. it PRESERVES alpha's phase. LAPACK's clarfg/zlarfg -- and therefore
// cuSOLVER, rocSOLVER and netlib, every geqrf this library can be asked to be a
// drop-in for -- return
//
//     beta = -sign(Re alpha) * hypot3(Re alpha, Im alpha, xnorm)  [REAL beta]
//
// Both are valid factorisations. They are NOT the same factorisation: the
// phase-preserving one leaves R with a non-real diagonal and produces a
// different tau, so its Q differs from the vendor's by a diagonal of
// unit-modulus phases. WP5's brief argues the phase-preserving form is
// numerically the better one, and that is defensible. It is nonetheless the
// WRONG choice here, for three reasons that outrank it:
//
//   1. geqrf is a CONTRACT, not a private helper. tau and the reflectors are
//      consumed by ormqr, orgqr, ormbr, sy2sb and band_reduction, and by user
//      code. A native geqrf that disagrees with the vendor on the phase is not a
//      drop-in, and the vendor-free build is supposed to be indistinguishable.
//   2. ormqr_cta_tests and ormqr_blocked_tests build their references from
//      NETLIB geqrf and from ormqr_vendor_or_throw. A phase divergence would
//      make those suites go red the day this kernel becomes the default -- and
//      "the test is comparing conventions" is exactly the kind of red that gets
//      silenced rather than diagnosed.
//   3. band_reduction / sy2sb copy R's diagonal into a Hermitian BAND. That band
//      is downstream input to sytrd_sb2st. Changing which representative of the
//      phase class lands there is a change nobody asked for.
//
// The real branch is IDENTICAL in both conventions, so this decision costs
// nothing for float/double.
//
// ONE CONSEQUENCE WORTH NAMING, because it is a real observable difference from
// the rest of this tree: internal::larfg returns tau = 0 whenever len <= 1,
// INCLUDING for complex -- its `if (len <= 1) return result;` sits above the
// complex branch. zlarfg does not: at N = 1 with Im(alpha) != 0 it returns a
// NONZERO tau, precisely to rotate the lone diagonal entry onto the real axis.
// This file follows zlarfg. It is also why WP5's baseline measured that deleting
// the LAST reflector of a SQUARE REAL matrix leaves the residual bit-identical
// (|tau[k-1]| = 0) while the same break turns complex red (|tau[k-1]| = 1.553).
// A short-final-panel test on a square real matrix guards NOTHING.
//
// ---------------------------------------------------------------------------
// OVERFLOW AND UNDERFLOW
// ---------------------------------------------------------------------------
// No larfg call site anywhere else in this tree implements xLARFG's rescaling
// loop, and every one of them forms xnorm by squaring UNSCALED values
// (math-helpers.hh:411, gebrd_blocked.cc:189, gebrd.cc:52,
// sytrd_cta_device.hh:133) -- so they all overflow to inf above ~1e19 in float
// and ~1e154 in double, and flush to zero at the small end. geqrf is the op most
// exposed to that, because unlike a symmetric reduction it is handed arbitrarily
// scaled user data.
//
// This file does not inherit that gap. Instead of xLARFG's iterative rescale it
// uses a SINGLE COMMON SCALE: one work-group pass finds
// s = max over the column of max(|Re|, |Im|), the second pass accumulates the
// tail's sum of squares ALREADY DIVIDED by s, and every scalar below is formed
// from quantities of order 1. tau is scale-INVARIANT and is computed entirely in
// scaled arithmetic, so it never overflows at all; beta is s * O(1).
// Mathematically this agrees with xLARFG (whose rescaling also leaves tau
// unchanged and rescales beta back), it costs one extra reduction, and it
// removes the loop.
//
// The one place a scale can still bite is v = x / (alpha - beta): the
// reciprocal-multiply spelling overflows when alpha - beta is subnormal. So the
// reciprocal is TESTED (dev_isfinite and non-zero) and the code falls back to a
// per-element DIVISION when it fails -- the deliberate asymmetry
// src/sycl/device_scalar.hh:200-209 describes, here decided per reflector rather
// than once per call site.
//
// ---------------------------------------------------------------------------
// THE nd_range, because "parallel over batch only" is this repository's
// recurring performance defect
// ---------------------------------------------------------------------------
// ONE WORK-GROUP PER MATRIX, of 32*CT work-items (CT = 1..8 sub-group teams).
// Inside a work-group:
//   * the two larfg reductions run over ALL work-items, strided by row, so the
//     global reads are coalesced;
//   * the reflector application maps TEAM -> trailing column and LANE -> row, so
//     each 32-lane sub-group walks one column contiguously (coalesced again) and
//     the dot product it needs is a SUB-GROUP butterfly, not a work-group
//     barrier. Different teams touch different columns, so the apply needs no
//     work-group barrier at all.
// That is gebrd_blocked.cc:145's accepted form (one work-group per matrix with
// real intra-matrix width), NOT gebrd.cc:45's defect (one work-ITEM per matrix).
//
// BARRIER AUDIT -- every site, all at the top level of the reflector loop, whose
// trip count (kmax) is kernel-uniform, and none inside a divergent region:
//   B0  after the tile load (CTA tier only, in the launcher).
//   B1  top of the loop: publishes the previous reflector's column updates before
//       column j is read.
//   B2  after the two reductions and before any write: every work-item read
//       A(j,j) as alpha, and work-item 0 is about to overwrite it with beta.
//       The reductions converge control flow but do not order memory.
//   B3  after beta/v are written: the teams are about to read them.
//   (the apply itself contains no work-group barrier -- see above)
//   B4  after the loop, before the store-back (CTA tier only, in the launcher).
// The v-scaling loop needs no barrier of its own: work-item t reads and writes
// exactly the rows it strides over, so there is no cross-item hazard within it.

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::geqrf_native {

using batchlas::sycl_device::Cx;

// The real component type of a DEVICE scalar. DevMap answers the question for a
// SOURCE type (std::complex<R> -> R); a kernel body templated on D cannot see T,
// which is the same reason IsDevComplex exists (device_scalar.hh:55-60).
template <typename D> struct RealOf            { using type = D; };
template <typename R> struct RealOf<Cx<R>>     { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

// The four spellings device_scalar.hh does not carry, kept here rather than
// added to it because only this file needs them and that header is included by
// every GEMM tile. dev_mul / dev_conj / dev_sub / dev_recip / dev_div /
// dev_real / dev_one / dev_isfinite / dev_is_zero all come from there, so this
// file contains NO private complex multiply -- which is the thing the WP5 brief
// forbids, and which latrd_lower_panel.cc:148 and seven other TUs each carry a
// copy of.
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

// max(|Re|, |Im|) -- the scaling functional. NOT |a|: a magnitude squares and
// adds, which is the overflow this scaling exists to avoid.
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

// ---------------------------------------------------------------------------
// The Householder scalars, LAPACK ?LARFG semantics exactly. See the phase note
// at the top of this file for why this is not internal::larfg.
//
// `alpha` is A(j,j); `s` is the common scale (max over max(|Re|,|Im|) of the
// WHOLE column, alpha included); `ssq` is sum over the TAIL of |x/s|^2.
//
// Postcondition, matching xLARFG's contract:
//   identity == true   ->  tau = 0, H = I, A(j,j) unchanged, v not written.
//   identity == false  ->  A(j,j) := beta (REAL even for complex),
//                          A(j+1:m, j) := x / (alpha - beta),
//                          H = I - tau v v^H with v = [1; A(j+1:m,j)].
// ---------------------------------------------------------------------------
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

    // xLARFG's early out, verbatim: a zero tail AND a real alpha means H = I.
    // For a REAL scalar this is the whole zero-column case (alphi is
    // structurally 0), which is why a real geqrf gets tau = 0 on a null column
    // and on the final 1x1 reflector. For a COMPLEX one it is not: a nonzero
    // Im(alpha) still needs a reflector to rotate the diagonal onto the real
    // axis, and clarfg/zlarfg return a nonzero tau there.
    if (ssq == R(0) && alphi == R(0)) return out;
    // s == 0 implies the whole column is zero, which the test above already
    // caught for every representable input; kept because it is what makes the
    // divisions below unconditionally safe rather than safe-by-argument.
    if (!(s > R(0))) return out;

    const R alphr = batchlas::sycl_device::dev_real(alpha);
    const R ar = alphr / s;
    const R ai = alphi / s;

    // Everything from here is O(1): the largest component of the scaled column
    // is 1 by construction, so nrm >= 1 and no product below can overflow.
    const R nrm = sycl::sqrt(sycl::fma(ar, ar, sycl::fma(ai, ai, ssq)));

    // beta = -SIGN(nrm, alphr). Fortran's SIGN returns +|x| when the second
    // argument is +0, and alphr == 0 is reachable here (a pure-imaginary
    // diagonal), so the comparison is `>= 0` and not `> 0`.
    const R beta_s = (alphr >= R(0)) ? -nrm : nrm;

    // tau = (beta - alpha)/beta is SCALE-INVARIANT, so it is formed entirely
    // from the scaled quantities and cannot overflow or underflow whatever the
    // magnitude of the input column. That is the half of xLARFG's rescaling loop
    // the loop exists to protect.
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        out.tau = D{(beta_s - ar) / beta_s, -ai / beta_s};
        out.beta = D{beta_s * s, R(0)};           // REAL beta -- the LAPACK convention
    } else {
        out.tau = (beta_s - ar) / beta_s;
        out.beta = beta_s * s;
    }

    // d = alpha - beta. |d/s| = |ar| + nrm >= 1, so |d| >= s: the only way the
    // reciprocal misbehaves is a subnormal or enormous s, and both are DETECTED
    // rather than assumed away.
    //
    // FORMED FROM out.beta, NOT RE-DERIVED FROM beta_s. Spelling it
    // `D{alphr - beta_s*s, alphi}` is arithmetically identical TODAY and would
    // silently stop being so the moment anyone changed which beta this function
    // returns -- v would then be scaled by a divisor belonging to a different
    // reflector, which is not a different convention but a wrong answer. One
    // source for beta.
    const D d = batchlas::sycl_device::dev_sub(alpha, out.beta);

    const D r = batchlas::sycl_device::dev_recip(d);
    out.use_mul = batchlas::sycl_device::dev_isfinite(r) &&
                  !batchlas::sycl_device::dev_is_zero(r);
    out.vfactor = out.use_mul ? r : d;
    out.identity = false;
    return out;
}

// ---------------------------------------------------------------------------
// Tile accessors. The ONLY difference between the two residencies.
// ---------------------------------------------------------------------------
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
    // No padding on ld, and that is a shape decision rather than an oversight:
    // the two hot access patterns are (lane -> consecutive r within one column)
    // and (team -> different column, lane -> consecutive r), both conflict-free
    // in local memory for ANY ld. A pad would only cost capacity.
    auto& at(int r, int c) const {
        return a[static_cast<std::size_t>(r) +
                 static_cast<std::size_t>(c) * static_cast<std::size_t>(ld)];
    }
};

// Sub-group sum, replicated to every lane. An XOR butterfly rather than
// sycl::reduce_over_group for sytrd_cta_device.hh:73-77's reason -- DPC++'s CUDA
// path for non-uniform group collectives has had limitations for floating-point
// reductions -- and because a complex sum has to be split into two real ones
// either way.
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

// ---------------------------------------------------------------------------
// THE PANEL FACTORISATION. LAPACK ?GEQR2 on an m x n tile, in place.
//
//   for j = 0 .. kmax-1
//       [beta, tau_j, v] = larfg( A(j,j), A(j+1:m, j) )
//       A(j:m, j+1:n) := (I - conj(tau_j) v v^H) A(j:m, j+1:n)
//
// The conj(tau) in the apply is zgeqr2's, not a transcription slip: zgeqr2 calls
// ZLARF with DCONJG(TAU(I)) because the factorisation applies H^H, while
// Q = H_1 H_2 ... H_k is what ormqr/orgqr reconstruct. Getting this wrong is
// invisible for real scalars and wrong-by-a-conjugate for complex -- i.e. it
// would pass every float and double test.
//
// `tau_ptr` is already offset to this matrix; reflector j lands at tau_ptr[j].
// It is a GLOBAL pointer in both instantiations: tau is an output of the op, and
// for the blocked tier the panel's slice of it is not contiguous with the panel.
//
// kmax is passed rather than derived so a caller can factor fewer reflectors than
// min(m, n); both callers pass min(m, n).
// ---------------------------------------------------------------------------
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

        // Pass 1: the common scale, over the WHOLE column (alpha included).
        R smax = dev_absmax(alpha);
        for (int r = j + 1 + tid; r < m; r += wg) {
            smax = sycl::fmax(smax, dev_absmax(A.at(r, j)));
        }
        smax = sycl::reduce_over_group(g, smax, sycl::maximum<R>());

        // Pass 2: the tail's scaled sum of squares. The `smax > 0` test is
        // OUTSIDE the accumulation and the collective is outside the test, so
        // every work-item reaches the reduction on both paths.
        R ssq = R(0);
        if (smax > R(0)) {
            for (int r = j + 1 + tid; r < m; r += wg) {
                ssq += dev_abs2_scaled(A.at(r, j), smax);
            }
        }
        ssq = sycl::reduce_over_group(g, ssq, sycl::plus<R>());

        // Evaluated REDUNDANTLY in every work-item rather than on a leader and
        // broadcast: both inputs are already replicated by the reductions above,
        // so a leader-plus-broadcast would add a barrier and a local slot to save
        // nothing. sytrd_cta_device.hh:135-141 makes the same call for the same
        // reason. It also makes `h.identity` work-group-uniform, which the
        // `continue` below depends on.
        const LarfgScalars<D> h = geqrf_larfg_scalars<D>(alpha, smax, ssq);

        // B2 -- every work-item has read A(j,j) as `alpha`; work-item 0 is about
        // to overwrite it.
        sycl::group_barrier(g);

        if (!h.identity) {
            if (tid == 0) {
                A.at(j, j) = h.beta;
            }
            // No barrier needed inside this loop: work-item t writes exactly the
            // rows it read.
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

        // `h.identity` is work-group-uniform (see above), so this `continue`
        // cannot make B1/B2/B3 divergent on the next iteration.
        if (h.identity) continue;

        const D ctau = batchlas::sycl_device::dev_conj(h.tau);

        // TEAM -> column, LANE -> row. The column loop's trip count depends only
        // on `team`, so it is sub-group-uniform and geqrf_sg_sum below is reached
        // by all lanes; it is NOT work-group-uniform, which is why there is no
        // work-group barrier in this region and why there must not be one.
        // Distinct teams touch distinct columns, so none is needed either.
        for (int c = j + 1 + team; c < n; c += nteams) {
            // w = v^H A(j:m, c), with the implicit v(j) = 1. A(j,j) holds beta,
            // not 1, so the leading term is peeled rather than branched on inside
            // the row loop.
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
