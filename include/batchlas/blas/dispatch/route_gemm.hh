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
// STATUS: additive. Not yet wired into cublas.cc; tests/route_gemm_equivalence_tests.cc
// pins it against the current behaviour first. See WP0_DISPATCH_SPEC.md S4.

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

// The candidate order for GEMM. Auto-terminated and of natural length, which
// removes the truncation hazard of the four hand-counted std::array<Provider,6>
// sites (provider.hh:26, env.hh:58/99/111) -- a missed one there silently
// truncates an order rather than failing to compile.
inline constexpr Route kGemmOrder[] = {
    {Origin::Native, Algorithm::RegisterTiled},
    {Origin::Vendor, Algorithm::Auto},
};

template <Op O, typename T>
struct RouteTable;

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
        if (s.heterogeneous_batch) return false;
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

        // Complex was excluded outright. NOTE: the WP2 measurement now shows
        // wide-scalar tiles beating both cuBLAS and the in-tree Tiled16
        // fallback for complex, so this exclusion is expected to be revisited
        // -- but changing it is a routing change with its own measurement, not
        // part of a refactor that must preserve behaviour.
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
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default differs per op
// and is supplied by the caller -- see legacy_unset_default() in route_env.hh,
// and note that GEMM's is Vendor while the level-3 ops' is Auto.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_gemm_route(Route forced, const OpShape& s,
                                bool vendor_available = true) {
    using Table = RouteTable<Op::gemm, T>;

    // Where everything that cannot get what it asked for ends up.
    //
    // The obvious body -- "take the first merely SUPPORTED route" -- is wrong,
    // and the equivalence test catches it: the order lists Native first, so an
    // 8x8x8 batch-1 GEMM (supported, but far outside the measured window) would
    // select the native kernel where today it goes to the vendor. That silently
    // inverts GEMM's vendor-by-default.
    //
    // Supported-but-not-preferred means "correct, but not the one we measured
    // as better", so it is reached only when there is no vendor left to fall
    // back to -- exactly the configuration this work package is building
    // toward. Returning Vendor when nothing at all can serve the shape is
    // deliberate: it is the honest "this needs a vendor and there isn't one"
    // signal, and the caller turns it into a diagnostic rather than a wrong
    // answer.
    auto fallback = [&]() -> Route {
        if (!vendor_available) {
            for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
                if (is_native(*r) && Table::supports(*r, s)) return *r;
            }
        }
        return Route{Origin::Vendor, Algorithm::Auto};
    };

    // A forced selection bypasses `preferred` -- that is the entire point of
    // forcing -- but never bypasses `supports`, because that would hand the
    // caller a wrong answer rather than a slow one.
    if (forced.origin != Origin::Auto) {
        // A REQUESTED VENDOR STILL HAS TO EXIST. This arm is not hypothetical:
        // GEMM's unset default IS Vendor (legacy_unset_default), so an ordinary
        // call with nothing set arrives here rather than at the preference walk
        // below. Returning `forced` unconditionally therefore made the
        // vendor-off degradation unreachable through the real call path --
        // provable only at the pure layer, where a test can pass Origin::Auto
        // that the adapter never produces. Caught by GemmTest.RouteAdapter*.
        if (is_vendor(forced)) {
            return vendor_available ? forced : fallback();
        }
        // A forced native route that cannot serve this shape yields rather than
        // computing nonsense. The old code did the same, by returning false
        // from gemm_use_sycl_custom.
        return Table::supports(forced, s) ? forced : fallback();
    }

    // Preference decides. A native route wins only where it is BOTH able to
    // serve the shape and measured to be the better choice for it.
    for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
        if (Table::supports(*r, s) && Table::preferred(*r, s)) return *r;
    }
    return fallback();
}

} // namespace batchlas::dispatch
