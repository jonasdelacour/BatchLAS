#pragma once

// The routing vocabulary: a Route is an {Origin, Algorithm} pair -- whose code runs,
// and which strategy it uses. Backend (device family) and BackendLibrary are separate
// axes and stay in enums.hh. See docs/design/vendor-independence.md#the-three-axes.

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

enum class Origin : uint8_t {
    Auto,
    Native,  // a kernel in this repository
    Vendor,  // third-party math code; the MathDx device libraries count as Vendor.
};

enum class Algorithm : uint8_t {
    Auto,
    Direct,           // one vendor call, or one monolithic kernel
    CTA,              // one work-group per matrix
    Blocked,
    TwoStage,
    Jacobi,
    RegisterTiled,    // the register-tiled GEMM family (src/sycl/gemm/)
    SplitK,           // k-partitioned GEMM
    ExpandGemm,       // materialise the structured operand, then batched GEMM
    TriangularTiles,  // tile-masked triangular kernel
    GramTiles,        // narrow-n single-tile rank-k kernel
    FusedDevice,      // one fused device-library kernel (cuBLASDx / cuSolverDx)

    // Deliberately WRONG, kept only as a measurement baseline: it stores BOTH
    // triangles, clobbering the half the caller owns. Auto must never select it.
    DiagFullGemm,
};

// A selection is a PAIR: `library` is a resolver output and is excluded from equality.
// Known wrong, deliberately left: nothing writes it, so a resolved Route reads CBLAS/false.
struct Route {
    Origin origin = Origin::Auto;
    Algorithm algo = Algorithm::Auto;
    BackendLibrary library = BackendLibrary::CBLAS;
    bool library_valid = false;

    friend constexpr bool operator==(const Route& a, const Route& b) {
        return a.origin == b.origin && a.algo == b.algo;
    }
    friend constexpr bool operator!=(const Route& a, const Route& b) { return !(a == b); }
};

// The vendor gate question as a predicate, so a new Algorithm cannot escape the gate.
inline constexpr bool is_vendor(Origin o) { return o == Origin::Vendor; }
inline constexpr bool is_vendor(const Route& r) { return is_vendor(r.origin); }

// "The ordinary vendor library call", not merely "some vendor route": MathDx routes are
// Origin::Vendor too, so a forced cuBLASDx request also answers yes to is_vendor().
inline constexpr bool is_plain_vendor(const Route& r) {
    return r.origin == Origin::Vendor && r.algo == Algorithm::Auto;
}
inline constexpr bool is_native(Origin o) { return o == Origin::Native; }
inline constexpr bool is_native(const Route& r) { return is_native(r.origin); }

// The dispatchable leaf ops -- one per include/batchlas/blas/functions/*.hh.
// extensions.hh's entry points are absent on purpose: no vendor alternative to choose.
enum class Op : uint8_t {
    gemm, gemv, trsm, trmm, symm, hemm, syrk, herk, syr2k, her2k,
    potrf, getrf, getrs, getri, geqrf, orgqr, ormqr, syev, gesvd, spmm, iluk,
    COUNT
};

enum class ScalarKind : uint8_t { F32, F64, C32, C64 };

template <typename T>
inline constexpr ScalarKind scalar_kind_of =
    std::is_same_v<T, float>                ? ScalarKind::F32 :
    std::is_same_v<T, double>               ? ScalarKind::F64 :
    std::is_same_v<T, std::complex<float>>  ? ScalarKind::C32 :
                                              ScalarKind::C64;

inline constexpr std::string_view to_string(Origin o) {
    switch (o) {
        case Origin::Auto:   return "auto";
        case Origin::Native: return "native";
        case Origin::Vendor: return "vendor";
    }
    return "?";
}

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
    }
    return "?";
}

inline constexpr std::string_view to_string(ScalarKind s) {
    switch (s) {
        case ScalarKind::F32: return "float";
        case ScalarKind::F64: return "double";
        case ScalarKind::C32: return "complex<float>";
        case ScalarKind::C64: return "complex<double>";
    }
    return "?";
}

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
        case Op::iluk:  return "iluk";   case Op::COUNT: return "?";
    }
    return "?";
}

// The <OP> in BATCHLAS_<OP>_ROUTE.
inline std::string op_env_stem(Op o) {
    std::string s(op_name(o));
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// Everything the routing predicates read, and nothing more. The device facts are
// cached FIELDS rather than SYCL get_info queries, keeping the resolver pure.
struct OpShape {
    Op op = Op::COUNT;
    ScalarKind scalar = ScalarKind::F32;
    Backend backend = Backend::AUTO;

    int64_t m = 0, n = 0, k = 0;   // square ops set all three
    int64_t batch = 1;

    Transpose transA = Transpose::NoTrans;
    Transpose transB = Transpose::NoTrans;
    Uplo uplo = Uplo::Lower;
    Side side = Side::Left;
    Diag diag = Diag::NonUnit;
    ComputePrecision precision = ComputePrecision::Default;
    bool heterogeneous_batch = false;

    bool is_gpu = false;
    int max_sub_group = 0;
    int compute_units = 0;

    int64_t max_dim() const { return m > n ? (m > k ? m : k) : (n > k ? n : k); }
    int64_t min_dim() const { return m < n ? (m < k ? m : k) : (n < k ? n : k); }

    std::string describe() const {
        return "m=" + std::to_string(m) + " n=" + std::to_string(n) +
               " k=" + std::to_string(k) + " batch=" + std::to_string(batch) +
               " T=" + std::string(to_string(scalar));
    }

    // Power-of-two buckets on max(m,n,k) and batch, so a 10,000-iteration test
    // collapses to a handful of coverage rows.
    uint32_t shape_class() const {
        auto log2b = [](int64_t v) -> uint32_t {
            uint32_t r = 0;
            while (v > 1) { v >>= 1; ++r; }
            return r;
        };
        return (log2b(max_dim()) << 8) | log2b(batch);
    }
};

// Where a forced selection came from, so a diagnostic can quote the spelling the caller
// typed. Load-bearing: tests/trmm_tests.cc asserts on the literal "BATCHLAS_TRMM_VARIANT".
struct RouteRequestSource {
    std::string variable;   // "BATCHLAS_TRMM_VARIANT" / "BATCHLAS_TRMM_ROUTE"
    std::string value;
    bool legacy = false;    // true when it came from a pre-Route spelling
};

} // namespace batchlas::dispatch
