#pragma once

// The dispatch vocabulary: two orthogonal axes, one spelling.
//
// WHY THIS EXISTS
//
// `Provider` (dispatch/provider.hh) flattens two independent questions into one
// enum. `Provider::Vendor` and `Provider::Netlib` answer "whose code is it";
// `Provider::BatchLAS_CTA`, `_Blocked`, `_TwoStage`, `_Jacobi` answer "which
// algorithm". Because they share one list, every consumer has to normalise the
// origin values away by hand before it can reason about the algorithm ones --
// `normalize_vendor_like()` in syev.hh, `normalize_gesvd_vendor_like()` in
// gesvd.hh, `normalize_ormqr_vendor_like()` in ormqr.hh, three copies of the
// same fixup for the same reason.
//
// It also could not express the state this whole work package exists to reach.
// "Run natively on an NVIDIA GPU that has no cuBLAS installed" is a statement
// about ORIGIN (native) and about a LIBRARY being absent (cuBLAS). It is not a
// statement about the device, and it is not an algorithm.
//
// Separating the axes:
//
//     Origin     whose code runs            Native | Vendor
//     Algorithm  what the code does         CTA | Blocked | ExpandGemm | ...
//     Backend    which device family        CUDA | ROCM | ... (enums.hh, unchanged)
//     BackendLibrary  which vendor library  CUBLAS | CUSOLVER | ... (enums.hh)
//
// Deliberately NOT added: a `Backend::SYCL` or an `Origin::SYCL`. Every route in
// this library is SYCL; naming one of them "SYCL" would carry no information and
// would collide with the device-family axis. See VENDOR_INDEPENDENCE_PLAN.md §3.1.
//
// STATUS: live. This is the only routing vocabulary in the tree. `Provider`,
// DispatchPolicy and the three dispatch/{provider,env,context}.hh headers are
// gone; every op that has a routing decision now makes it through a
// RouteTable<Op, T> and dispatch::resolve_route. See WP0_DISPATCH_SPEC.md S4.

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// Axis 1 -- ORIGIN: whose code is it.
// ---------------------------------------------------------------------------
enum class Origin : uint8_t {
    Auto,    // let the resolver decide
    Native,  // a kernel in this repository
    Vendor,  // third-party math code: cuBLAS/cuSOLVER/cuSPARSE, roc*, oneMKL,
             // CBLAS/LAPACKE -- and the MathDx device libraries cuBLASDx and
             // cuSolverDx. Those two count as Vendor even though their kernels
             // compile into our .so: the source is NVIDIA's, it ships only for
             // NVIDIA, and so it can never be the portable path. Vendor
             // independence has to be measurable without them.
};

// ---------------------------------------------------------------------------
// Axis 2 -- ALGORITHM: what the code does.
//
// Orthogonal to Origin: the same strategy can in principle be reached natively
// or through a vendor device library, which is exactly why these cannot live in
// one enum with the origins.
// ---------------------------------------------------------------------------
enum class Algorithm : uint8_t {
    Auto,
    Direct,           // one vendor call, or one monolithic kernel
    CTA,              // one work-group per matrix
    Blocked,          // panel factorisation + blocked trailing update
    TwoStage,         // dense -> band -> tridiagonal
    Jacobi,           // one-sided / cyclic Jacobi
    RegisterTiled,    // the register-tiled GEMM family (src/sycl/gemm/)
    SplitK,           // k-partitioned GEMM
    ExpandGemm,       // materialise the structured operand, then batched GEMM
    TriangularTiles,  // tile-masked triangular kernel
    GramTiles,        // narrow-n single-tile rank-k kernel
    FusedDevice,      // one fused device-library kernel (cuBLASDx / cuSolverDx)

    // A deliberately WRONG route kept only so the arithmetic it saves can be
    // measured: it computes and stores BOTH triangles, which is not what SYRK
    // or SYR2K mean -- the half the caller did not name is the caller's
    // storage. See syrk_custom_dispatch.cc. Auto must never select it; it is
    // reachable only by naming it explicitly.
    DiagFullGemm,
};

// ---------------------------------------------------------------------------
// A selection is a PAIR. `library` is an OUTPUT: the resolver fills it in on
// the way back so a caller (or a coverage table, or a throw) can say which
// third-party library a Vendor route actually landed on. It is not part of
// equality, because a request never specifies it.
// ---------------------------------------------------------------------------
struct Route {
    Origin origin = Origin::Auto;
    Algorithm algo = Algorithm::Auto;
    BackendLibrary library = BackendLibrary::CBLAS;  // output; see note above
    bool library_valid = false;                      // false until resolved

    friend constexpr bool operator==(const Route& a, const Route& b) {
        return a.origin == b.origin && a.algo == b.algo;
    }
    friend constexpr bool operator!=(const Route& a, const Route& b) { return !(a == b); }
};

// The question the vendor-independence gate actually asks. Expressed as a
// predicate rather than by enumerating names, so that adding an Algorithm can
// never accidentally escape the gate.
inline constexpr bool is_vendor(Origin o) { return o == Origin::Vendor; }
inline constexpr bool is_vendor(const Route& r) { return is_vendor(r.origin); }

// "The ordinary vendor library call", as distinct from any vendor route.
//
// A TRAP WORTH NAMING. The MathDx device libraries are Origin::Vendor (their
// source is NVIDIA's), so {Vendor, FusedDevice} satisfies is_vendor -- but the
// level-3 dispatchers' old `request == Vendor` tests meant specifically "call
// cublasSsyrk", and a fused-kernel request was emphatically NOT that. Rendering
// those as is_vendor() inverts them: a forced cublasdx request starts answering
// yes to "did the caller ask for the vendor?". Use this instead wherever the
// question is about the plain library call rather than about origin.
inline constexpr bool is_plain_vendor(const Route& r) {
    return r.origin == Origin::Vendor && r.algo == Algorithm::Auto;
}
inline constexpr bool is_native(Origin o) { return o == Origin::Native; }
inline constexpr bool is_native(const Route& r) { return is_native(r.origin); }

// ---------------------------------------------------------------------------
// The dispatchable leaf ops -- one per include/batchlas/blas/functions/*.hh.
//
// extensions.hh's entry points are deliberately absent: they are BatchLAS
// algorithms with no vendor alternative to choose between, and they keep
// dispatching through BATCHLAS_DISPATCH_ON_QUEUE unchanged.
// ---------------------------------------------------------------------------
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

// The uppercase stem of this op's environment variables, i.e. the <OP> in
// BATCHLAS_<OP>_ROUTE.
inline std::string op_env_stem(Op o) {
    std::string s(op_name(o));
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// ---------------------------------------------------------------------------
// Everything the routing predicates in this tree actually read, and nothing
// more. POD, no allocation, cheap to build per call.
//
// The device facts are FIELDS rather than queries. dispatch/context.hh's
// query_caps() performs three SYCL get_info round-trips plus a std::string heap
// allocation on every op invocation; a routing decision has no business paying
// that, and caching it on the Queue makes the resolver pure.
// ---------------------------------------------------------------------------
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

    // Power-of-two bucket on max(m,n,k), power-of-two bucket on batch. A
    // 10,000-iteration test therefore collapses to a handful of coverage rows
    // rather than 10,000.
    uint32_t shape_class() const {
        auto log2b = [](int64_t v) -> uint32_t {
            uint32_t r = 0;
            while (v > 1) { v >>= 1; ++r; }
            return r;
        };
        return (log2b(max_dim()) << 8) | log2b(batch);
    }
};

// ---------------------------------------------------------------------------
// Where a forced selection came from, so a diagnostic can quote the exact
// spelling the caller typed rather than the canonical one. Load-bearing:
// tests/trmm_tests.cc asserts on the literal text "BATCHLAS_TRMM_VARIANT".
// ---------------------------------------------------------------------------
struct RouteRequestSource {
    std::string variable;   // "BATCHLAS_TRMM_VARIANT" / "BATCHLAS_TRMM_ROUTE"
    std::string value;      // "cublasdx"
    bool legacy = false;    // true when it came from a pre-Route spelling
};

} // namespace batchlas::dispatch
