#pragma once

// GESVD's routing table.
//
// gesvd's chooser was already the most careful of the three Provider-based
// ones -- it checked a forced provider against its support predicate, which
// ormqr's did not -- so this is mostly a translation. The one thing it does
// change is where a measured judgement lives.
//
// THE WIDE-BAND RULE WAS A PREFERENCE SITTING INSIDE AN ORDER WALK.
// choose_gesvd_provider's loop carried, in the middle of its Jacobi arm, a
// `wide_band` test that declined Jacobi for REAL input above max(m,n) = 32
// even though gesvd_supports_jacobi accepts it. That is a speed/accuracy
// judgement, not a capability one, and it was indistinguishable from the
// support checks around it. Here it is `preferred`, so it can never make the
// Jacobi route ineligible -- which matters directly for vendor independence:
// with no vendor compiled in, a real 33..64 matrix must still be served, and
// Jacobi is what serves it.
//
// The measurements behind the rule are kept verbatim; see the comment on
// preferred() below.

#include <optional>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// gesvd's routing reads two things OpShape has no field for. Rather than grow
// OpShape into a union of every op's arguments, the op extends it.
struct GesvdShape : OpShape {
    SvdVectors jobu = SvdVectors::All;
    SvdVectors jobvh = SvdVectors::All;
    std::optional<Uplo> hermitian_uplo;   // engaged => Hermitian input

    // Both predicates below want the canonicalised jobs; the caller does that
    // once, before building this, exactly as gesvd_dispatch does.
    bool want_vectors() const {
        return jobu != SvdVectors::None || jobvh != SvdVectors::None;
    }
};

// Jacobi first. That ordering is not arbitrary and its measurement is recorded
// in full on default_order_gesvd (dispatch/env.hh): the one-sided Jacobi kernel
// dominates the older CTA path on accuracy across the whole conditioning sweep,
// and on speed wherever U is requested, and the shared order had buried it
// behind exactly the path it replaces.
inline constexpr Route kGesvdOrder[] = {
    {Origin::Native, Algorithm::Jacobi},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

// Largest max(m, n) gesvdj_cta accepts, per scalar type. Mirrors
// gesvdj_cta_max_dim in src/extensions/gesvdj_cta.cc.
//
// The kernel keeps P = 32 lanes above n = 32 and grows the tile capacity C to
// 64, so each lane owns two rows. The limit is local memory: per problem with
// the V tile resident, C=64 costs 37,952 B for float, 71,744 B for double and
// complex<float>, and 138,816 B for complex<double>, against a measured device
// limit of 101,376 B. Values-only drops the V tile and halves it.
template <typename T>
inline constexpr int64_t gesvd_jacobi_max_dim(bool want_vectors) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return want_vectors ? 32 : 64;
    } else {
        return 64;
    }
}

template <typename T>
struct RouteTable<Op::gesvd, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // The three gesvd_supports_* predicates, transcribed and nothing more.
    static bool supports(Route r, const GesvdShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        const int64_t max_dim = s.m > s.n ? s.m : s.n;

        switch (r.algo) {
            case Algorithm::Jacobi: {
                // gesvdj_cta supports complex GENERAL input natively, unlike
                // the two below, which both return false for non-real T outside
                // the Hermitian branch. That is the Tier 4 coverage gap: complex
                // general SVD on GPU used to fall through to Vendor and throw.
                // Do NOT copy the RealScalar gate here.
                if (s.hermitian_uplo.has_value()) return false;   // no Hermitian shortcut
                if (!s.is_gpu) return false;
                if (s.max_sub_group < 32) return false;
                if (s.m < 1 || s.n < 1 || s.batch < 1) return false;
                if (max_dim > gesvd_jacobi_max_dim<T>(s.want_vectors())) return false;
                // Every job combination is served, Thin included: one-sided
                // Jacobi produces the thin U natively -- it IS the rotated,
                // normalised A -- and the full-U columns are the extra work,
                // manufactured by an in-kernel Gram-Schmidt that a Thin request
                // skips outright.
                return true;
            }
            case Algorithm::CTA: {
                if (!s.is_gpu) return false;
                if (s.max_sub_group < 32) return false;
                if (s.m < 1 || s.n < 1 || s.batch < 1) return false;
                if (max_dim > 32) return false;
                // A genuinely thin factor is out of reach for this route: mode
                // CTA always takes the normal-equations branch, whose
                // patch_zero_left_vectors writes m columns of U unconditionally.
                if (s.jobu == SvdVectors::Thin || s.jobvh == SvdVectors::Thin) return false;
                if (s.hermitian_uplo.has_value()) {
                    if (s.m != s.n) return false;
                    return *s.hermitian_uplo == Uplo::Lower || *s.hermitian_uplo == Uplo::Upper;
                }
                if constexpr (!RealScalar<T>) {
                    return false;
                }
                return true;
            }
            case Algorithm::Blocked: {
                // Real matrices with optional full, thin, or absent U and/or
                // V^H backtransforms via ORMBR. Hermitian support remains
                // square-only, where Thin canonicalises to All anyway.
                if (!s.is_gpu) return false;
                if (s.m < 1 || s.n < 1 || s.batch < 1) return false;
                if (s.hermitian_uplo.has_value()) {
                    if (s.m != s.n) return false;
                    return *s.hermitian_uplo == Uplo::Lower;
                }
                if constexpr (!RealScalar<T>) {
                    return false;
                }
                return true;
            }
            default:
                // Including Algorithm::Auto: gesvd has three native routes, so
                // a bare "native" names none of them. resolve_route walks the
                // order to pick one -- see the note there.
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    static bool preferred(Route r, const GesvdShape& s) {
        if (!is_native(r)) return false;
        if (!supports(r, s)) return false;

        if (r.algo == Algorithm::Jacobi) {
            // THE WIDE-BAND RULE, moved here verbatim from the order walk.
            //
            // The 33..64 band is served by Jacobi only where the alternative
            // cannot serve it at all -- that is, complex GENERAL input, which
            // the blocked route declines, leaving Vendor and a throw.
            //
            // For REAL input in that band the blocked path is the better
            // default, and this is the one place the two disagree. Measured at
            // n=64, batch=4096, float, full vectors: blocked 4.86 us/matrix
            // against Jacobi's 7.25, and at low conditioning blocked is also
            // the more accurate of the two (kappa=1e1: 1.1e-6 vs 1.2e-5).
            //
            // Jacobi wins decisively at HIGH conditioning -- kappa=1e6, n=64:
            // singular-value relative error 6.2e-3 vs 0.526, orthogonality
            // 3.7e-5 vs 0.144, i.e. the blocked path returns no correct digits
            // and a U that is not a basis. But which regime a caller is in is
            // not knowable from the shape, and unlike the n <= 32 case (where
            // the CTA path was worse from kappa=1e2 up) blocked is genuinely
            // better below ~1e4. So the default stays blocked and the accurate
            // route is opt-in via BATCHLAS_GESVD_PROVIDER=jacobi -- which is a
            // FORCED request and so never consults this function.
            //
            // Being un-preferred rather than unsupported is the whole point of
            // the split: with no vendor available, a real 33..64 matrix still
            // has to be served, and Jacobi is what serves it.
            const bool wide_band = (s.m > s.n ? s.m : s.n) > 32;
            if constexpr (RealScalar<T>) {
                return !wide_band;
            } else {
                return true;
            }
        }

        // CTA and Blocked were preferred wherever they were supported: the old
        // loop tested only their support predicates.
        return true;
    }

    static constexpr const Route* order_begin() { return kGesvdOrder; }
    static constexpr const Route* order_end() {
        return kGesvdOrder + (sizeof(kGesvdOrder) / sizeof(kGesvdOrder[0]));
    }
};

template <typename T>
inline Route resolve_gesvd_route(Route forced, const GesvdShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::gesvd, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
