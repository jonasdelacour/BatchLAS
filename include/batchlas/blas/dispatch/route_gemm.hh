#pragma once

// GEMM's routing table, as the three-way split the WP0 spec mandates.
//
// WHY A SPLIT AT ALL
//
// `gemm_use_sycl_custom` (src/backends/gemm_variant.hh) is three different
// things welded into one predicate, and that is what makes it dangerous to move:
//
//   1. an ENVIRONMENT READ   -- it opens by calling gemm_variant_request() and
//                               returns false for Vendor/Native/CuBLASDx;
//   2. a CORRECTNESS GATE    -- gemm_custom_problem_supported(): dimensions
//                               agree, batch sizes agree, homogeneous, default
//                               precision. If this is false the kernel computes
//                               the WRONG ANSWER;
//   3. a MEASURED WINDOW     -- GPU-only, real-types-only, square, batch >= 64,
//                               and the per-type max_dim ranges. If this is
//                               false the kernel is merely SLOWER.
//
// Conflating 2 and 3 is the trap. Move the window into `supports` and a 1024^3
// float GEMM at batch 256 suddenly has no supported route at all, which breaks
// an op that works today the moment vendor is unavailable. Move the env read
// into `supports` and forcing a route stops working. So:
//
//   supports()   == correctness only, and nothing else. Never a speed cutoff.
//   preferred()  == the measured window. Returning false never makes a route
//                   ineligible, only un-preferred.
//   the env read  == lives in the alias table (route_env.hh), not here.
//
// Everything here is PURE -- it reads only its arguments. No getenv, no SYCL
// query. That is what makes `gemm` and `gemm_buffer_size` reach the same route
// by construction rather than by a hand-written comment asking them to.
//
// STATUS: live. gemm_use_sycl_custom (src/backends/gemm_variant.hh) is now a
// two-line adapter over resolve_gemm_route, so cublas.cc, mkl.cc and rocblas.cc
// all route through this. tests/route_gemm_equivalence_tests.cc pins it against
// the behaviour it replaced. See WP0_DISPATCH_SPEC.md S4.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// The candidate order for GEMM. Auto-terminated and of natural length, which
// removes the truncation hazard of the four hand-counted std::array<Provider,6>
// sites (provider.hh:26, env.hh:58/99/111) -- a missed one there silently
// truncates an order rather than failing to compile.
inline constexpr Route kGemmOrder[] = {
    {Origin::Native, Algorithm::RegisterTiled},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::gemm, T> {
    // ---- 2. CORRECTNESS ---------------------------------------------------
    // Verbatim gemm_custom_problem_supported, and nothing else. A route whose
    // supports() is false cannot produce the right answer for this shape.
    static bool supports(Route r, const OpShape& s) {
        if (r.origin == Origin::Vendor) {
            return true;  // the vendor serves everything it is given
        }
        if (r.origin != Origin::Native) {
            return false;
        }
        if (r.algo != Algorithm::RegisterTiled && r.algo != Algorithm::Auto) {
            return false;
        }
        // --- gemm_custom_problem_supported ---
        if (s.precision != ComputePrecision::Default) return false;
        // WP2 C2: heterogeneous batch is no longer a correctness gap. The
        // facade walks the batch (gemm_heterogeneous.hh) and each member is
        // HOMOGENEOUS by construction, so a native route can serve it.
        //
        // It stays rejected by preferred() below, deliberately: the per-item
        // loop is one launch per batch member, which is a cost, not a win. This
        // is exactly the supports()/preferred() split doing its job -- "can
        // produce the right answer" and "is the right choice" are different
        // questions, and conflating them is the defect this vocabulary exists
        // to prevent.
        return s.m > 0 && s.n > 0 && s.k > 0;
    }

    // ---- 3. MEASURED WINDOW ----------------------------------------------
    // Everything from `ctx.device().type != GPU` downward in the original,
    // preserved cell for cell. These are speed judgements, not correctness
    // ones; every threshold here was measured and none should be "tidied".
    static bool preferred(Route r, const OpShape& s) {
        if (r.origin != Origin::Native) return false;
        if (!supports(r, s)) return false;

        if (!s.is_gpu) return false;

        // WP2 C2. This test used to live in supports() and is now stated here
        // explicitly, because loosening supports() would otherwise have let a
        // heterogeneous batch through preferred() and MOVED A VENDOR-PRESENT
        // ROUTE -- a silent perf change riding on a correctness fix. The
        // per-item loop is one launch per batch member; correct, but not a
        // reason to leave the vendor.
        if (s.heterogeneous_batch) return false;

        // COMPLEX IS NOT MERELY UNPREFERRED HERE -- IT HAS NO REGISTER KERNEL.
        // The note that used to sit here said the WP2 measurement shows
        // wide-scalar tiles beating cuBLAS "so this exclusion is expected to be
        // revisited". Acting on that sentence alone is a REGRESSION: the second
        // gate is select_kernel_variant (src/sycl/gemm_kernels.cc:466), whose
        // entire register ladder is inside `if constexpr (is_same_v<T,float>)`,
        // so complex falls to `max_dim <= 64 ? Direct : Tiled16` (:514) --
        // measured at 3.2-7.1x SLOWER than cuBLAS in
        // WP2_WIDE_SCALAR_GEMM_VERDICT.md. Widening here without first porting
        // the 64x64x16 t4x4 tile into src/ and wiring the selector routes
        // complex to Tiled16, not to a register kernel. Order: port ->
        // selector -> predicate. See WP2_GEMM_SPEC.md.
        if constexpr (is_std_complex_v<T>) {
            return false;
        } else {
            const int64_t max_dim = s.max_dim();

            // Square only, and enough batch to fill the device.
            if (s.m != s.n || s.n != s.k || s.batch < 64) return false;

            if constexpr (std::is_same_v<T, float>) {
                if (s.transA != Transpose::NoTrans || s.transB != Transpose::NoTrans) {
                    // ConjTrans is meaningless for a real type and was rejected.
                    if (s.transA == Transpose::ConjTrans || s.transB == Transpose::ConjTrans) {
                        return false;
                    }
                    return s.batch >= 128 && max_dim >= 128 && max_dim <= 512;
                }
                if (max_dim <= 32) return true;
                return max_dim >= 128 && max_dim <= 512;
            } else if constexpr (std::is_same_v<T, double>) {
                return max_dim <= 512;
            } else {
                return false;
            }
        }
    }

    static constexpr const Route* order_begin() { return kGemmOrder; }
    static constexpr const Route* order_end() {
        return kGemmOrder + (sizeof(kGemmOrder) / sizeof(kGemmOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// The body now lives in route_resolve.hh, shared with every other op: the
// forced-bypasses-preferred-but-not-supports rule, the requested-vendor-must-
// exist rule, and the vendor-off degradation are not GEMM-specific. This name
// stays as the spelling GEMM's callers and tests already use.
//
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default differs per op
// and is supplied by the caller -- see legacy_unset_default() in route_env.hh,
// and note that GEMM's is Vendor while the level-3 ops' is Auto.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_gemm_route(Route forced, const OpShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::gemm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
