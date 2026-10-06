#pragma once

/// @file
/// @brief The routing vocabulary: Route{Origin, Algorithm}, the Op list and OpShape.
///
/// A Route says whose code runs (Origin) and which strategy it uses
/// (Algorithm). The device family (Backend) and the library (BackendLibrary)
/// are separate axes in enums.hh.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-three-axes

#include <cctype>     // std::toupper in op_env_stem; libstdc++'s <string> happens
                      // to pull this in, which is not a guarantee
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

/// @addtogroup dispatch
/// @{

/// @brief Whose code serves a call.
enum class Origin : uint8_t {
    Auto,    ///< no opinion: let the route table decide
    Native,  ///< a kernel in this repository
    Vendor,  ///< third-party math code; the MathDx device libraries count as Vendor
};

/// @brief Which strategy serves a call; meaningful only together with an Origin.
///
/// A route table's order array, never this numeric value, is the walk order.
/// @invariant Enumerators are APPENDED, never inserted: this is an installed header
///            behind a SOVERSION, so every enumerator keeps the number it shipped with.
enum class Algorithm : uint8_t {
    Auto,             ///< no opinion; a bare origin resolves to that origin's first supported route
    Direct,           ///< one vendor call, or one monolithic kernel
    CTA,              ///< one work-group per matrix
    Blocked,          ///< a blocked driver over panel kernels and routed sub-operations
    TwoStage,         ///< two-stage reduction (band, then tridiagonal)
    Jacobi,           ///< Jacobi rotations
    RegisterTiled,    ///< the register-tiled GEMM family (src/sycl/gemm/)
    SplitK,           ///< k-partitioned GEMM
    ExpandGemm,       ///< materialise the structured operand, then batched GEMM
    TriangularTiles,  ///< tile-masked triangular kernel
    GramTiles,        ///< narrow-n single-tile rank-k kernel
    FusedDevice,      ///< one fused device-library kernel (cuBLASDx / cuSolverDx); Vendor origin

    /// Deliberately WRONG, kept only as a measurement baseline: it stores BOTH
    /// triangles, clobbering the half the caller owns. Auto must never select it.
    DiagFullGemm,

    Tiny,             ///< one matrix per sub-group partition, held in registers; first in its tables' walk despite its number
    LPanel,           ///< left-looking fused panel: local memory holds ONE n x NB panel
};

/// @brief A route selection: the {Origin, Algorithm} pair.
///
/// Equality compares `origin` and `algo` only.
/// @note `library` / `library_valid` are documented as a resolver output but
///       nothing writes them, so a resolved Route reads `CBLAS` / `false`.
///       Known and deliberately left; do not read them.
struct Route {
    Origin origin = Origin::Auto;                    ///< whose code
    Algorithm algo = Algorithm::Auto;                ///< which strategy
    BackendLibrary library = BackendLibrary::CBLAS;  ///< never written (see note)
    bool library_valid = false;                      ///< never written (see note)

    friend constexpr bool operator==(const Route& a, const Route& b) {
        return a.origin == b.origin && a.algo == b.algo;
    }
    friend constexpr bool operator!=(const Route& a, const Route& b) { return !(a == b); }
};

/// @brief True for any vendor route, including the MathDx `FusedDevice` routes.
///
/// The vendor gate is phrased through this predicate so a new Algorithm cannot escape it.
inline constexpr bool is_vendor(Origin o) { return o == Origin::Vendor; }
/// @brief True for any vendor route, including the MathDx `FusedDevice` routes.
inline constexpr bool is_vendor(const Route& r) { return is_vendor(r.origin); }

/// @brief True only for the ordinary vendor library call, `{Vendor, Auto}`.
///
/// Use this, not is_vendor(), when the question is "does this call the library":
/// a forced cuBLASDx request is also `Origin::Vendor`.
inline constexpr bool is_plain_vendor(const Route& r) {
    return r.origin == Origin::Vendor && r.algo == Algorithm::Auto;
}
/// @brief True for a kernel in this repository.
inline constexpr bool is_native(Origin o) { return o == Origin::Native; }
/// @brief True for a kernel in this repository.
inline constexpr bool is_native(const Route& r) { return is_native(r.origin); }

/// @brief The dispatchable leaf ops, one per `include/batchlas/blas/functions/*.hh`.
///
/// extensions.hh's entry points are absent on purpose: they have no vendor
/// alternative to choose between. `iluk` is listed but routes through nothing.
/// @invariant APPENDED before `COUNT`, never inserted (installed header behind a SOVERSION).
enum class Op : uint8_t {
    gemm, gemv, trsm, trmm, symm, hemm, syrk, herk, syr2k, her2k,
    potrf, getrf, getrs, getri, geqrf, orgqr, ormqr, syev, gesvd, spmm, iluk,
    // gesv and posv: the two ops with no vendor arm on any backend (route_gesv.hh).
    gesv, posv,
    COUNT
};

/// @brief Runtime tag of the four scalar types.
enum class ScalarKind : uint8_t { F32, F64, C32, C64 };

/// @brief The ScalarKind of `T`; any type that is not float, double or complex<float> maps to C64.
template <typename T>
inline constexpr ScalarKind scalar_kind_of =
    std::is_same_v<T, float>                ? ScalarKind::F32 :
    std::is_same_v<T, double>               ? ScalarKind::F64 :
    std::is_same_v<T, std::complex<float>>  ? ScalarKind::C32 :
                                              ScalarKind::C64;

/// @brief Lower-case spelling, as accepted by the route environment variables.
inline constexpr std::string_view to_string(Origin o) {
    switch (o) {
        case Origin::Auto:   return "auto";
        case Origin::Native: return "native";
        case Origin::Vendor: return "vendor";
    }
    return "?";
}

/// @brief Canonical lower-case spelling, as accepted by the route environment variables.
inline constexpr std::string_view to_string(Algorithm a) {
    switch (a) {
        case Algorithm::Auto:            return "auto";
        case Algorithm::Direct:          return "direct";
        case Algorithm::CTA:             return "cta";
        case Algorithm::Blocked:         return "blocked";
        case Algorithm::TwoStage:        return "two_stage";
        case Algorithm::Jacobi:          return "jacobi";
        case Algorithm::RegisterTiled:   return "register_tiled";
        case Algorithm::SplitK:          return "split_k";
        case Algorithm::ExpandGemm:      return "expand_gemm";
        case Algorithm::TriangularTiles: return "triangular_tiles";
        case Algorithm::GramTiles:       return "gram_tiles";
        case Algorithm::FusedDevice:     return "fused_device";
        case Algorithm::DiagFullGemm:    return "diag_full_gemm";
        case Algorithm::Tiny:            return "tiny";
        case Algorithm::LPanel:          return "lpanel";
    }
    return "?";
}

/// @brief C++ spelling of the scalar type, e.g. `complex<float>`.
inline constexpr std::string_view to_string(ScalarKind s) {
    switch (s) {
        case ScalarKind::F32: return "float";
        case ScalarKind::F64: return "double";
        case ScalarKind::C32: return "complex<float>";
        case ScalarKind::C64: return "complex<double>";
    }
    return "?";
}

/// @brief Lower-case op name, e.g. `gemm`; `?` for `COUNT`.
inline constexpr std::string_view op_name(Op o) {
    switch (o) {
        case Op::gemm:  return "gemm";   case Op::gemv:  return "gemv";
        case Op::trsm:  return "trsm";   case Op::trmm:  return "trmm";
        case Op::symm:  return "symm";   case Op::hemm:  return "hemm";
        case Op::syrk:  return "syrk";   case Op::herk:  return "herk";
        case Op::syr2k: return "syr2k";  case Op::her2k: return "her2k";
        case Op::potrf: return "potrf";  case Op::getrf: return "getrf";
        case Op::getrs: return "getrs";  case Op::getri: return "getri";
        case Op::geqrf: return "geqrf";  case Op::orgqr: return "orgqr";
        case Op::ormqr: return "ormqr";  case Op::syev:  return "syev";
        case Op::gesvd: return "gesvd";  case Op::spmm:  return "spmm";
        case Op::iluk:  return "iluk";
        case Op::gesv:  return "gesv";   case Op::posv:  return "posv";
        case Op::COUNT: return "?";
    }
    return "?";
}

/// @brief The `<OP>` in `BATCHLAS_<OP>_ROUTE`: op_name() upper-cased.
inline std::string op_env_stem(Op o) {
    std::string s(op_name(o));
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/// @brief Everything the routing predicates read, and nothing more.
///
/// Device facts are cached fields rather than SYCL `get_info` queries, which keeps
/// the resolver pure. Ops that route on more derive from it (TrsmShape, GeqrfShape, ...).
/// @trap A derived shape must never shadow an OpShape field: resolve_route() slices
///       the shape to OpShape for coverage, so the shadow is dropped silently.
struct OpShape {
    Op op = Op::COUNT;                    ///< the op being routed
    ScalarKind scalar = ScalarKind::F32;  ///< scalar type of the call
    Backend backend = Backend::AUTO;      ///< often AUTO: the backend is a template parameter at the call site

    int64_t m = 0, n = 0, k = 0;   ///< op-specific extents; square ops set all three
    int64_t batch = 1;             ///< batch count

    Transpose transA = Transpose::NoTrans;                    ///< op() on the first operand
    Transpose transB = Transpose::NoTrans;                    ///< op() on the second operand
    Uplo uplo = Uplo::Lower;                                  ///< triangle, for triangular and symmetric ops
    Side side = Side::Left;                                   ///< side, for trsm-like ops
    Diag diag = Diag::NonUnit;                                ///< unit diagonal flag
    ComputePrecision precision = ComputePrecision::Default;   ///< requested compute precision
    bool heterogeneous_batch = false;                         ///< items carry differing active dimensions

    bool is_gpu = false;       ///< the queue's device is a GPU
    int max_sub_group = 0;     ///< sub_group_sizes()[0], NOT the maximum; see the `has_sg32` fields
    int compute_units = 0;     ///< device compute units

    /// @brief max(m, n, k).
    int64_t max_dim() const { return m > n ? (m > k ? m : k) : (n > k ? n : k); }
    /// @brief min(m, n, k).
    int64_t min_dim() const { return m < n ? (m < k ? m : k) : (n < k ? n : k); }

    /// @brief Human-readable `m=.. n=.. k=.. batch=.. T=..` for diagnostics.
    std::string describe() const {
        return "m=" + std::to_string(m) + " n=" + std::to_string(n) +
               " k=" + std::to_string(k) + " batch=" + std::to_string(batch) +
               " T=" + std::string(to_string(scalar));
    }

    /// @brief Coverage bucket: `floor(log2(max_dim())) << 8 | floor(log2(batch))`,
    ///        so a 10,000-iteration test collapses to a handful of coverage rows.
    uint32_t shape_class() const {
        auto log2b = [](int64_t v) -> uint32_t {
            uint32_t r = 0;
            while (v > 1) { v >>= 1; ++r; }
            return r;
        };
        return (log2b(max_dim()) << 8) | log2b(batch);
    }
};

/// @brief Where a forced selection came from, so a diagnostic can quote what the caller typed.
/// @note Load-bearing: tests/trmm_tests.cc asserts on the literal "BATCHLAS_TRMM_VARIANT".
struct RouteRequestSource {
    std::string variable;   ///< e.g. "BATCHLAS_TRMM_VARIANT" or "BATCHLAS_TRMM_ROUTE"
    std::string value;      ///< the raw value as set
    bool legacy = false;    ///< true when it came from a pre-Route spelling
};

/// @}

} // namespace batchlas::dispatch
