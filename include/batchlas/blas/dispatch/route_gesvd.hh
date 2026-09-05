#pragma once

// GESVD's routing table: order walk, the three native support predicates, and
// the wide-band preference (which lives in preferred(), not supports(), so a
// vendor-free build still serves a real 33..64 matrix).
// evidence: docs/design/vendor-free-status.md#what-has-a-native-kernel-and-what-routes-to-it-by-default

#include <optional>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct GesvdShape : OpShape {
    SvdVectors jobu = SvdVectors::All;
    SvdVectors jobvh = SvdVectors::All;
    std::optional<Uplo> hermitian_uplo;   // engaged => Hermitian input

    // Precondition: the caller canonicalises jobu/jobvh before building this.
    bool want_vectors() const {
        return jobu != SvdVectors::None || jobvh != SvdVectors::None;
    }
};

// Jacobi first: it beats the CTA path it replaces on accuracy, and on speed where U is wanted.
inline constexpr Route kGesvdOrder[] = {
    {Origin::Native, Algorithm::Jacobi},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

// Largest max(m, n) gesvdj_cta accepts; must match gesvdj_cta_max_dim in
// src/extensions/gesvdj_cta.cc. The bound is local memory for the V tile.
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
    static bool supports(Route r, const GesvdShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        const int64_t max_dim = s.m > s.n ? s.m : s.n;

        switch (r.algo) {
            case Algorithm::Jacobi: {
                // gesvdj_cta handles complex GENERAL input; the two arms below
                // do not. Do NOT copy their RealScalar gate here.
                if (s.hermitian_uplo.has_value()) return false;   // no Hermitian shortcut
                if (!s.is_gpu) return false;
                if (s.max_sub_group < 32) return false;
                if (s.m < 1 || s.n < 1 || s.batch < 1) return false;
                if (max_dim > gesvd_jacobi_max_dim<T>(s.want_vectors())) return false;
                // Thin needs no gate: one-sided Jacobi produces the thin U natively.
                return true;
            }
            case Algorithm::CTA: {
                if (!s.is_gpu) return false;
                if (s.max_sub_group < 32) return false;
                if (s.m < 1 || s.n < 1 || s.batch < 1) return false;
                if (max_dim > 32) return false;
                // Thin is unreachable here: mode CTA takes the normal-equations
                // branch, whose patch_zero_left_vectors always writes m U columns.
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
                // Auto included: with three native routes, a bare "native" names none.
                return false;
        }
    }

    static bool preferred(Route r, const GesvdShape& s) {
        if (!is_native(r)) return false;
        if (!supports(r, s)) return false;

        if (r.algo == Algorithm::Jacobi) {
            // Wide-band rule: for REAL input Jacobi is preferred only at
            // max(m, n) <= 32; above that blocked is the better default except at
            // high conditioning. Force it with BATCHLAS_GESVD_PROVIDER=jacobi.
            const bool wide_band = (s.m > s.n ? s.m : s.n) > 32;
            if constexpr (RealScalar<T>) {
                return !wide_band;
            } else {
                return true;
            }
        }

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
