# WP0 — Final Specification: Dispatch Axes, Vendor Decoupling, and the Coverage Instrument

Root: `/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan`. Every line/file citation below was re-read in this session unless marked *(from ground truth)*.

---

## 1. The decision, in five sentences

`Backend` keeps its name and becomes **exactly one thing — the device family a call is compiled for** (`CUDA`, `ROCM`, `INTEL`, `HOST`), with `NETLIB`/`MKL` retained as value-identical aliases so no instantiation and no existing source line changes, and `MAGMA`/`SYCL` deleted. Which *math library* exists is a second, independent axis already present in the tree as `BackendLibrary`, which is now derived from CMake per-library probes (`BATCHLAS_HAS_CUBLAS`, `_CUSOLVER`, …) rather than from the device family. Which *implementation runs for one call* is a third axis, `Route = {Origin, Algorithm}` — `Origin ∈ {Vendor, Native}` answers "whose code", `Algorithm` answers "which strategy" — replacing `Provider`, `GemmVariantRequest`, `SymmVariantRequest`, `SyrkRoute`, `Syr2kRoute` and `TrmmVariantRequest` with one vocabulary. There is deliberately **no** `Backend::SYCL` and **no** `Provider::BatchLAS_SYCL`: "NVIDIA GPU with no cuBLAS" is spelled `Backend::CUDA` + `library_available(CUBLAS) == false`, which is the truth, and it costs zero new template instantiations on a `.so` that is device-link-bound. The public definitions of the 21 leaf ops move out of the vendor TUs into a thin facade so that a build with no vendor library links, and every vendor entry is wrapped by a typed `op_external` that counts, logs and (under `BATCHLAS_NO_VENDOR=1`) throws naming op, library and shape.

Where designs conflicted, the picks and the one-line reasons:

- **Backend as compile-time family, not runtime** (minimal over capability): the capability registry's instantiation collapse is an unmeasured build-time bet on a box with seven measured-dead build fixes, and it is not needed for any WP0 requirement.
- **`Route{Origin, Algorithm}` pair, not a flat `Provider`** (two-axis over minimal): the user's complaint is precisely that ORIGIN and ALGORITHM are mixed, and a flat enum re-mixes them.
- **Public op definitions move to a facade** (two-axis over minimal): verified at `src/backends/cublas.cc:1568-1580` — `gemm<Back,T>` is *defined* in the vendor TU, so `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` cannot link without moving it; no enum can fix this.
- **`with_backend`, `BATCHLAS_DISPATCH_ON_QUEUE` and all 105 sites are untouched** (minimal over capability): 70 of the 105 are in `extensions.hh`, which has no vendor/native choice to make.
- **cuBLASDx/cuSOLVERDx count as `Origin::Vendor`**: they are third-party code shipped in the NVIDIA MathDx package and can never be the portable path, so `NO_VENDOR` and `ENABLE_VENDOR_BLAS=OFF` must exclude them.

---

## 2. Exact C++ type definitions

### 2.1 `include/batchlas/blas/enums.hh` — replaces lines 78-100

```cpp
    // The compiled instantiation family a call is generated for: a DEVICE
    // FAMILY, and nothing else.
    //
    // This is NOT "which math library implements the op" -- that is
    // batchlas::dispatch::Route, chosen per call, and
    // Queue::library_available() for what exists. An NVIDIA GPU with no cuBLAS
    // installed is still Backend::CUDA: it is a CUDA device with no vendor
    // provider.
    //
    // There is deliberately no Backend::SYCL. Every backend here is SYCL, so
    // the value would name nothing, and because Backend is bound once per Queue
    // (queue-dispatch.hh:35) it would also make "cuSOLVER gesvd with a native
    // syrk" -- the state M2 converges through, cell by cell -- inexpressible.
    // MAGMA never had an implementation (queue-dispatch.hh:26-28).
    enum class Backend {
        AUTO,     // a request; Queue::backend() never returns it
        CUDA,     // NVIDIA SYCL device (nvptx64)
        ROCM,     // AMD SYCL device (amdgcn)
        INTEL,    // Intel SYCL device (spir64 / level_zero)
        HOST,     // CPU SYCL device: host-reachable pointers, host-callable BLAS

        // Deprecated spellings, retained as ALIASES with identical values so
        // that every existing `Backend::NETLIB` / `Backend::MKL` in the tree
        // keeps compiling and keeps naming the SAME explicit instantiation.
        // That is the entire reason these are aliases and not new enumerators:
        // an alias adds zero instantiations. Not emitted by to_string().
        NETLIB = HOST,
        MKL    = INTEL,
        // REMOVED: MAGMA, SYCL.
    };

    // Aliases share values, so this switch must name only the five canonical
    // enumerators; adding `case Backend::MKL:` beside `case Backend::INTEL:`
    // is a duplicate-case error. Same rule applies to with_backend().
    inline constexpr std::string_view to_string(Backend v) {
        switch (v) {
            case Backend::AUTO:  return "AUTO";
            case Backend::CUDA:  return "CUDA";
            case Backend::ROCM:  return "ROCM";
            case Backend::INTEL: return "INTEL";
            case Backend::HOST:  return "HOST";
        }
        return "Backend(?)";
    }
```

`BackendLibrary` (`enums.hh:102-113`) gains three enumerators and keeps everything else:

```cpp
    enum class BackendLibrary {
        CUBLAS, CUSPARSE, CUSOLVER,
        ROCBLAS, ROCSPARSE, ROCSOLVER,
        MAGMA, MKL, CBLAS, LAPACKE,
        CUBLASDX,    // NEW: header-only NVIDIA device library, separately absent
        CUSOLVERDX,  // NEW: ditto
        NONE,        // NEW: the implementation is ours
    };
```

and, replacing the comment-only mapping:

```cpp
    // The Backend -> BackendLibrary map, in code for the first time.
    constexpr std::optional<BackendLibrary> vendor_blas_for(Backend b);   // CUBLAS/ROCBLAS/MKL/CBLAS
    constexpr std::optional<BackendLibrary> vendor_solver_for(Backend b); // CUSOLVER/ROCSOLVER/MKL/LAPACKE
    constexpr std::optional<BackendLibrary> vendor_sparse_for(Backend b); // CUSPARSE/ROCSPARSE/nullopt/nullopt
```

`vendor_sparse_for(HOST)` is `nullopt` on purpose: `netlib_lapack.cc:218`'s CSR SpMM is a hand-written host loop, i.e. already native code living in a vendor TU.

### 2.2 `include/batchlas/blas/dispatch/route.hh` — NEW (~180 lines)

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <batchlas/blas/enums.hh>

namespace batchlas::dispatch {

// AXIS: ORIGIN -- whose code is it.
enum class Origin : uint8_t {
    Auto,     // let the resolver decide
    Native,   // a kernel in this repository
    Vendor,   // third-party math code: cuBLAS/cuSOLVER/cuSPARSE, roc*, oneMKL,
              // CBLAS/LAPACKE, and the MathDx device libraries cuBLASDx /
              // cuSOLVERDx. The device libraries are Vendor even though their
              // kernels compile into our .so: they are somebody else's source,
              // they ship only for NVIDIA, and they can never be the portable
              // path -- so vendor independence must be measurable without them.
};

// AXIS: ALGORITHM -- what the code does. Orthogonal to Origin: the same
// strategy name can be reached natively or through a vendor device library.
enum class Algorithm : uint8_t {
    Auto,
    Direct,           // one vendor call / one monolithic kernel
    CTA,              // one work-group per matrix
    Blocked,          // panel factorisation + blocked trailing update
    TwoStage,         // dense -> band -> tridiagonal
    Jacobi,           // one-sided / cyclic Jacobi
    RegisterTiled,    // the 128x128 register-tiled GEMM family (sycl_gemm)
    SplitK,           // k-partitioned GEMM
    ExpandGemm,       // materialise the structured operand, then batched GEMM
    TriangularTiles,  // tile-masked triangular kernel
    GramTiles,        // narrow-k single-tile rank-k kernel
    FusedDevice,      // one fused device-library kernel (cuBLASDx/cuSOLVERDx)
    DiagFullGemm,     // DELIBERATELY WRONG measurement route: computes and
                      // stores BOTH triangles, which is not what SYRK/SYR2K
                      // mean (syrk_custom_dispatch.cc:31-34). Never selected by
                      // Auto; reachable only by explicit force.
};

// A selection is a PAIR. This is the fix for `Provider`, which flattened these
// two axes -- which is why Provider::Netlib (an origin) sat in one list beside
// Provider::BatchLAS_CTA (an algorithm) and had to be normalised away by all
// three consumers.
struct Route {
    Origin    origin = Origin::Auto;
    Algorithm algo   = Algorithm::Auto;
    // Filled in by the resolver on the way out; input value is ignored.
    BackendLibrary library = BackendLibrary::NONE;

    friend constexpr bool operator==(const Route& a, const Route& b) {
        return a.origin == b.origin && a.algo == b.algo;   // library is output
    }
};

// The 21 dispatchable leaf ops -- one per include/batchlas/blas/functions/*.hh.
// extensions.hh's 70 entry points are NOT here: they are BatchLAS algorithms
// with no vendor alternative, and they keep dispatching through
// BATCHLAS_DISPATCH_ON_QUEUE unchanged.
enum class Op : uint8_t {
    gemm, gemv, trsm, trmm, symm, hemm, syrk, herk, syr2k, her2k,
    potrf, getrf, getrs, getri, geqrf, orgqr, ormqr, syev, gesvd, spmm, iluk,
    COUNT
};

enum class ScalarKind : uint8_t { F32, F64, C32, C64 };

template <typename T> inline constexpr ScalarKind scalar_kind_of =
    std::is_same_v<T, float>  ? ScalarKind::F32 :
    std::is_same_v<T, double> ? ScalarKind::F64 :
    std::is_same_v<T, std::complex<float>> ? ScalarKind::C32 : ScalarKind::C64;

constexpr std::string_view to_string(Origin);
constexpr std::string_view to_string(Algorithm);
constexpr std::string_view op_name(Op);        // "gemm", "syev", ...
constexpr std::string_view op_env_stem(Op);    // "GEMM", "SYEV", ...

// Everything any predicate in the tree reads today, and nothing else. POD,
// no allocation. Built by shape_of<Op::X>(...) from the op's own arguments.
struct OpShape {
    Op         op;
    ScalarKind scalar;
    Backend    backend;

    int64_t m = 0, n = 0, k = 0;   // square ops set all three
    int64_t batch = 1;

    Transpose transA = Transpose::NoTrans, transB = Transpose::NoTrans;
    Uplo      uplo   = Uplo::Lower;
    Side      side   = Side::Left;
    Diag      diag   = Diag::NonUnit;
    JobType   job    = JobType::NoEigenVectors;
    ComputePrecision precision = ComputePrecision::Default;
    bool heterogeneous_batch = false;

    // Device facts, cached on the Queue at construction -- NOT queried per call
    // the way query_caps() is today (context.hh:24-47, three SYCL get_info
    // round-trips plus a std::string heap allocation, per op invocation).
    bool is_gpu = false;
    int  max_sub_group = 0;
    int  compute_units = 0;

    std::string describe() const;      // "m=512 n=512 k=512 batch=2048 T=float"
    // Power-of-two bucket on max(m,n,k) x power-of-two bucket on batch, so a
    // 10 000-iteration test yields one coverage row.
    uint32_t shape_class() const;
};

// ---- diagnostics ---------------------------------------------------------
// Where a forced selection came from, so that a throw can quote the exact
// spelling the user typed. tests/trmm_tests.cc:151 asserts on the literal
// string "BATCHLAS_TRMM_VARIANT=cublasdx", so this is load-bearing.
struct RouteRequestSource {
    std::string_view variable;   // "BATCHLAS_TRMM_VARIANT" or "BATCHLAS_TRMM_ROUTE"
    std::string      value;      // "cublasdx"
    bool             legacy = false;
};

class DispatchError : public std::runtime_error {
public:
    Op op; Backend backend; ScalarKind scalar; OpShape shape; Route route;
    using std::runtime_error::runtime_error;
};
class NoRouteError            : public DispatchError {};  // nothing serves this call
class ForcedRouteUnavailable  : public DispatchError { public: RouteRequestSource src; };
class VendorForbidden         : public DispatchError { public: BackendLibrary lib; };

} // namespace batchlas::dispatch
```

### 2.3 `include/batchlas/blas/dispatch/plan.hh` — NEW (~200 lines)

```cpp
#pragma once
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

// The ONE customisation point, and the answer to requirement (f).
//
// The PRIMARY TEMPLATE says: this op has a vendor implementation and nothing
// else. An op with no native kernel therefore needs ZERO new code -- it builds,
// it links, and it works via vendor. Adding a native kernel later is one
// specialisation, and it touches nothing else.
template <Op O, typename T>
struct RouteTable {
    // Hard gate: correctness only. Never a speed cutoff -- speed lives in
    // `preferred`. A route whose `supports` is false CANNOT produce the right
    // answer for this shape.
    static bool supports(Route r, const Queue&, const OpShape&) {
        return r.origin == Origin::Vendor;
    }
    // Soft gate: this route is the measured winner for this shape. Returning
    // false never makes a route ineligible.
    static bool preferred(Route, const Queue&, const OpShape&) { return false; }
    // Candidate order, tried after `preferred`. Auto terminates.
    static std::span<const Route> order() { return kDefaultOrder; }
    // Optional per-op measured routing table, consulted before the order walk.
    // Returns nullopt for "no opinion". syev's saturated grid lives here.
    static std::optional<Route> measured(const Queue&, const OpShape&) { return std::nullopt; }
};

// Auto-terminated candidate lists. `constexpr Route[]` of natural length,
// exposed as std::span -- this removes the truncation hazard of the four
// hand-counted std::array<Provider,6> sites (provider.hh:26, env.hh:58/99/111).
inline constexpr Route kDefaultOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Native, Algorithm::TwoStage},
    {Origin::Native, Algorithm::Jacobi},
    {Origin::Vendor, Algorithm::Auto},
};

// Availability. Purely a build fact; kept separate from `supports` so that
// "not compiled" and "cannot serve this shape" are distinguishable in a throw.
bool route_compiled(Op, Backend, ScalarKind, Route);
BackendLibrary library_for(Op, Backend, Route);

struct RouteConfig {                       // per-op, parsed once; see §4.3
    Route              forced{};
    RouteRequestSource src{};
    bool               strict = false;     // BATCHLAS_<OP>_STRICT=1
};
enum class VendorMode { Allow, PreferNative, Forbid };

// PURE: reads only the arguments, the tables and the cached env config.
// No getenv, no SYCL query, no clock. Therefore xxx_buffer_size and xxx reach
// the same route by construction -- this structurally eliminates the failure
// mode syev.hh:918-923 and :975-978 currently guard by hand-written comment.
template <Op O, typename T>
Route resolve_route(Backend, const Queue&, const OpShape&);

} // namespace batchlas::dispatch
```

### 2.4 `include/batchlas/blas/dispatch/op.hh` — rewritten (see §7 for semantics)

```cpp
#pragma once
#include <batchlas/blas/dispatch/route.hh>
#include <utility>

namespace batchlas {

// A call about to enter third-party code. This is the counter, the log point
// and the throw point -- see §7.
struct ExternalCall {
    dispatch::Op            op;
    BackendLibrary          lib;
    Backend                 backend;
    dispatch::ScalarKind    scalar;
    const dispatch::OpShape* shape;    // may be null only for descriptor setup
    const char*             symbol;    // "cublasGemmStridedBatchedEx", ...
};

// Counts, logs, and throws dispatch::VendorForbidden when BATCHLAS_NO_VENDOR=1.
// Also publishes a thread-local scope that call_backend()/call_backend_nh()
// assert against, so an unattributed vendor call is caught in a debug build.
template <class F>
decltype(auto) op_external(const ExternalCall& c, F&& f);

// Deprecated overload, retained so the 19 existing call sites keep compiling.
// Counts into an "unattributed" bucket and throws the same exception with a
// less useful message. Every site is migrated to the typed form in step S6.
template <class F>
[[deprecated("use op_external(ExternalCall, F)")]]
decltype(auto) op_external(const char* name, F&& f);

} // namespace batchlas
```

---

## 3. Migration table

### 3.1 Types and enumerators

| Old | New | Churn |
|---|---|---|
| `Backend::CUDA`, `Backend::ROCM` | **unchanged** name; meaning narrowed to device family | 0 |
| `Backend::NETLIB` | `Backend::HOST`; `NETLIB = HOST` alias kept | 0 source edits |
| `Backend::MKL` | `Backend::INTEL`; `MKL = INTEL` alias kept | 0, except `to_string` + `with_backend` switch must use `INTEL` |
| `Backend::MAGMA` | **deleted** | `enums.hh:95`, `queue-dispatch.hh:27` comment, `tests/backend_dispatch_tests.cc`, `python/batchlas/bindings/support.hh:409` |
| `Backend::SYCL` | **deleted** | same 4 sites (`support.hh:406`) |
| plan's `Backend::SYCL` (new) | **rejected** — see §1 | — |
| `BackendLibrary` | **unchanged**, +`CUBLASDX`, `CUSOLVERDX`, `NONE`, and now derived from CMake | +40 lines |
| `Provider` (7 values) | **deleted** → `Route{Origin, Algorithm}` | `provider.hh` deleted |
| `Provider::Auto` | `Route{Auto, Auto}` | |
| `Provider::Vendor` | `Route{Vendor, Auto}` | |
| `Provider::Netlib` | `Route{Vendor, Auto}` **on `Backend::HOST` only**; on any other backend the token is now an **error**, not a silent synonym for cuBLAS | fixes a real conflation |
| `Provider::BatchLAS_CTA` | `Route{Native, CTA}` | |
| `Provider::BatchLAS_Blocked` | `Route{Native, Blocked}` | |
| `Provider::BatchLAS_TwoStage` | `Route{Native, TwoStage}` | |
| `Provider::BatchLAS_Jacobi` | `Route{Native, Jacobi}` | |
| plan's `Provider::BatchLAS_SYCL` | **never exists** — every BatchLAS route is SYCL | — |
| `normalize_vendor_like` ×3 (`syev.hh:158`, `gesvd.hh:203`, `ormqr.hh:143`) | **deleted** | 3 sites |
| `DispatchPolicy` | `RouteConfig` | |
| `DispatchPolicy::log` (dead) | `BATCHLAS_DISPATCH_LOG=1`, now actually read | dead → live |
| `DispatchPolicy::require_in_order` (dead) | `BATCHLAS_<OP>_STRICT=1` | dead → live |
| `std::array<Provider,6>` ×4 | `constexpr Route[]` + `std::span<const Route>` | truncation hazard removed |
| `DispatchContext` (`context.hh:17-21`, unused) | **deleted** | |
| `DeviceCaps` (`context.hh:11-15`) | folded into `OpShape`, cached on `Queue` | |
| `GemmVariantRequest::{Vendor,Sycl,Native,CuBLASDx,Auto}` | `Route{Vendor,Auto}` / `{Native,RegisterTiled}` / `{Vendor,Auto}` / `{Vendor,FusedDevice}` / `{Auto,Auto}` | enum deleted |
| `SymmVariantRequest::{Vendor,CuBLASDx,Auto}` | `{Vendor,Auto}` / `{Vendor,FusedDevice}` / `{Auto,Auto}` | enum deleted |
| `SyrkRoute::{Vendor,Fused,Triangular,Gram,Gemm,Auto}` | `{Vendor,Auto}` / `{Vendor,FusedDevice}` / `{Native,TriangularTiles}` / `{Native,GramTiles}` / `{Native,DiagFullGemm}` / `{Auto,Auto}` | enum deleted |
| `Syr2kRoute::*` | as SyrkRoute minus Gram | enum deleted |
| `TrmmVariantRequest::*` + `trmm_triangular_requested()` (two independent readers of one variable) | one `Route`; `triangular\|tiles` → `{Native,TriangularTiles}` | **the one non-literal alias — see §9 risk 3** |
| *(unnamed default route of symm/hemm/trmm, `triangular_expand.hh`)* | `Algorithm::ExpandGemm` — newly nameable, behaviour unchanged | |
| `parse_cublasdx_variant_request` (`route_common.hh:43`) | **deleted** → the legacy alias table | |
| `should_use_cublasdx` (`route_common.hh:73`) | **deleted** → `resolve_route` | |
| `cublasdx_variant_needs_fallback` (`cublasdx_dispatch_common.hh:26`) | becomes an **availability predicate consulted by the resolver**, never a substitution inside the callee | fixes a real defect: `symm_custom_dispatch.cc:168` falls back to our expand+gemm while `trmm_custom_dispatch.cc:203` falls back to cuBLAS, under the same logged route |
| `throw_forced_cublasdx_unavailable` | `ForcedRouteUnavailable` carrying `RouteRequestSource` | message text preserved |
| `gemm_use_sycl_custom`, `gemm_use_cublasdx_custom`, `symm_use_cuda_custom`, `syrk_route_*`, `syr2k_use_cuda_custom`, `trmm_*` | **split**: the correctness half → `RouteTable::supports`, the measured-window half → `RouteTable::preferred`, the env half → the alias table. See §9 risk 2. | 5 files |
| `choose_syev_provider` (`syev.hh:755`) | `RouteTable<Op::syev,T>::{supports,preferred,order,measured}`; the saturated grids (`syev.hh:686-752`) move **verbatim** into `measured()` | |
| `choose_gesvd_provider` (`gesvd.hh:302`) | `RouteTable<Op::gesvd,T>::*`; `default_order_gesvd` (`env.hh:99-106`) and its 30-line measurement comment move verbatim into `order()` | `tests/gesvd_tests.cc:1198,1207,1214` |
| `choose_ormqr_provider` (`ormqr.hh:161`) | `RouteTable<Op::ormqr,T>::*` | |
| `default_order_for_op`'s `string_view(opname)=="GESVD"` compare (`env.hh:111-116`) | **deleted** — trait specialisation | |
| `policy_from_env` (2 `std::string` allocs + `getenv` per call) | `route_config(Op)`, a cached static | net **cheaper** per call |
| `op_external(const char*, F&&)` | typed `op_external(ExternalCall, F&&)`; string overload deprecated | §7 |
| `with_backend`, `BATCHLAS_DISPATCH_ON_QUEUE`, `kProbeBackend`, all 105 sites | **unchanged** in shape; the switch loses `MKL`, gains `INTEL`; `kProbeBackend` becomes `Backend::HOST` unconditionally | ~15 lines |
| `BATCHLAS_INSTANTIATE`, `BATCHLAS_FOR_EACH_*`, the 46 extension ladders | **unchanged** | 1 line (`symm.cc:78` `Backend::MKL` → dropped, see §5) |
| `BackendMatrixHandle` | **unchanged** shape; its cuSPARSE members re-gate from `BATCHLAS_HAS_CUDA_BACKEND` to `BATCHLAS_HAS_CUSPARSE` | `backend_handle_impl.hh` |

### 3.2 CMake / preprocessor names

| Old | New | Note |
|---|---|---|
| `BATCHLAS_HAS_CUDA_BACKEND` | **name unchanged**, meaning becomes "the NVIDIA instantiation family is compiled"; set from `BATCHLAS_CUDA_ENABLED`, not from `find_library(cublas)` | the 46 extension guards are already exactly this |
| `BATCHLAS_HAS_ROCM_BACKEND` | name unchanged, set from `BATCHLAS_DETECTED_AMD_GPU OR BATCHLAS_ENABLE_ROCM` | |
| `BATCHLAS_HAS_HOST_BACKEND` | name unchanged, set from `BATCHLAS_HAS_CPU_TARGET OR BATCHLAS_HAS_LAPACKE` | keeps today's value on this box; does **not** add a family in a `-DBATCHLAS_ENABLE_CPU_TESTS=OFF -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build |
| `BATCHLAS_HAS_MKL_BACKEND` | **split**: `BATCHLAS_HAS_INTEL_BACKEND` (family, hard-wired FALSE in WP0) + `BATCHLAS_HAS_ONEMKL` (library). `linalg-impl.hh:20`'s `#if !BATCHLAS_HAS_MKL_BACKEND` becomes `#if !BATCHLAS_HAS_ONEMKL`, which is what it always meant | kills the wired dead branch |
| *(new)* | `BATCHLAS_HAS_CUBLAS`, `_CUSOLVER`, `_CUSPARSE`, `_CUBLASDX`, `_CUSOLVERDX`, `_ROCBLAS`, `_ROCSOLVER`, `_ROCSPARSE`, `_LAPACKE`, `_CBLAS`, `_ONEMKL` | |
| *(new)* | `BATCHLAS_ENABLE_VENDOR_BLAS` (option, ON), per-library `BATCHLAS_ENABLE_<LIB>` defaulting from it | |
| `BATCHLAS_HAS_GPU_BACKEND` (`backend_config.h.in:44-48`, classes MKL as a GPU backend) | recomputed over `CUDA_BACKEND \|\| ROCM_BACKEND \|\| INTEL_BACKEND` | |
| duplicate `target_compile_definitions` channel (`BatchLASDependencies.cmake:411-430`) | **deleted**; the generated header is the single source | |

### 3.3 Environment variables

Canonical: `BATCHLAS_<OP>_ROUTE` (ordered, comma-separated token list, optional trailing `!` = strict), plus global `BATCHLAS_ROUTE`. `_ROUTE` and not `_IMPL`, because `BATCHLAS_ORMQR_IMPL`, `BATCHLAS_LATRD_IMPL` and `BATCHLAS_SYTRD_IMPL` already exist and mean something else.

Precedence: `BATCHLAS_<OP>_ROUTE` > legacy `BATCHLAS_<OP>_{VARIANT,PROVIDER}` > `BATCHLAS_ROUTE` > built-in.

Tokens: `auto`, `vendor`, `native`, plus every `Algorithm` spelling (`direct`, `cta`, `blocked`, `two_stage`, `jacobi`, `reg128`/`registertiled`, `splitk`, `expand`, `tiles`, `gram`, `fused`, `gemm_both_triangles`), plus every `BackendLibrary` spelling (`cublas`, `cusolver`, `lapacke`, `cublasdx`, …). First namespace that matches wins.

| Legacy variable | value | new `Route` | default when unset |
|---|---|---|---|
| `BATCHLAS_GEMM_VARIANT` | *(unset)*, `vendor`, *(unknown)* | `{Vendor, Auto}` | **`{Vendor,Auto}` — the odd one out, preserved verbatim** (`gemm_variant.hh:53-58,77`) |
| | `sycl`, `custom` | `{Native, RegisterTiled}`, **forced past the speed window but not past `supports`** — this reproduces `gemm_variant.hh:161`, where `request == Sycl` returns true after `gemm_custom_problem_supported` and *before* the GPU check | |
| | `native`, `cuda-native`, `direct-cuda` | `{Vendor, Auto}` — it reaches no distinct route today, it only disables the other two (`gemm_variant.hh:151-154`). The deprecation warning must say so explicitly, because `output/` provenance distinguishes the tags. | |
| | `cublasdx`, `dx` | `{Vendor, FusedDevice}` | |
| | `auto` | `{Auto, Auto}` | |
| `BATCHLAS_SYMM_VARIANT` | `vendor` / `cublasdx\|dx\|custom` / `auto` / *unset* / *unknown* | `{Vendor,Auto}` / `{Vendor,FusedDevice}` / `{Auto,Auto}` / `{Auto,Auto}` / `{Auto,Auto}` | `Auto` (`route_common.hh:48`) |
| `BATCHLAS_SYRK_VARIANT` | + `triangular\|tiles` / `gram\|narrow` / `gemm` | `{Native,TriangularTiles}` / `{Native,GramTiles}` / `{Native,DiagFullGemm}` | `Auto` |
| `BATCHLAS_SYR2K_VARIANT` | + `triangular\|tiles` / `gemm` | `{Native,TriangularTiles}` / `{Native,DiagFullGemm}` | `Auto` |
| `BATCHLAS_TRMM_VARIANT` | + `triangular\|tiles` | `{Native,TriangularTiles}` | `Auto` |
| `BATCHLAS_{SYEV,GESVD,ORMQR}_PROVIDER` | `auto` / `vendor` / `cta`/`blocked`/`two_stage`/`jacobi` + all `-`/`_`/`batchlas_` spellings (`env.hh:28-49`) | `{Auto,Auto}` / `{Vendor,Auto}` / `{Native,<Algo>}` | `Auto` |
| | `netlib` | `{Vendor,Auto}` on `Backend::HOST`; **`ForcedRouteUnavailable` elsewhere** (was: silently cuBLAS) | |
| | *unknown* | `{Auto,Auto}` (`env.hh:47-48`) | |

**Unknown-token policy, deliberately asymmetric.** A new `BATCHLAS_*_ROUTE` with an unrecognised token **throws** at first resolve, listing that op's valid tokens — a silently-ignored typo is exactly how a benchmark run gets recorded against the wrong route. A legacy variable keeps its silent fallback bit-exactly, so recorded provenance stays valid.

Each legacy variable emits one deprecation line to `stderr` on first read, naming the replacement, suppressible with `BATCHLAS_QUIET_DEPRECATIONS=1`.

New: `BATCHLAS_NO_VENDOR ∈ {0,1,warn}`, `BATCHLAS_PREFER_NATIVE=1`, `BATCHLAS_DISPATCH_LOG=1`, `BATCHLAS_COVERAGE_OUT=<path>`, `BATCHLAS_<OP>_STRICT=1`.

**Untouched** (different axis — they select *within* one implementation): `BATCHLAS_ORTHO_GRAM`, `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE`, `BATCHLAS_ORMQR_IMPL`, `BATCHLAS_ORMQR_WY`, `BATCHLAS_LATRD_*`, `BATCHLAS_SYEVX_ALGORITHM`, `BATCHLAS_STEDC_*`, `BATCHLAS_GEMM_SYCL_KERNEL`, `BATCHLAS_EXPAND_ROUTE`, `BATCHLAS_TEST_BACKEND`, `BATCHLAS_SKIP_POINTER_CHECKS`.

---

## 4. Resolution algorithm

### 4.1 Level 1 — `Queue::backend()`, device family only

```cpp
// src/util/queue-impl.cc, replacing :63-113
static bool Queue::backend_available(Backend b) {          // FAMILY compiled? (build query)
    switch (b) {
        case Backend::CUDA:  return BATCHLAS_HAS_CUDA_BACKEND;
        case Backend::ROCM:  return BATCHLAS_HAS_ROCM_BACKEND;
        case Backend::INTEL: return BATCHLAS_HAS_INTEL_BACKEND;
        case Backend::HOST:  return BATCHLAS_HAS_HOST_BACKEND;
        default:             return false;                 // AUTO
    }
}
static bool Queue::library_available(BackendLibrary L);    // NEW: exact per-library
static bool Queue::vendor_available(Backend b);            // NEW: the family's primary BLAS
       bool Queue::device_serves(Backend b) const;         // NEW: query, not an assertion

Backend Queue::backend() const {
    if (backend_ != AUTO) return backend_;
    if (resolved_ != AUTO) return resolved_;
    Backend choice = AUTO;
    if (device_.type == DeviceType::GPU) {
        switch (device_.get_vendor()) {
            // THE DELETED CONJUNCT IS REQUIREMENT (d): backend_available() is no
            // longer and-ed with a vendor library having been found. An NVIDIA
            // GPU resolves to Backend::CUDA whether or not cuBLAS exists.
            case Vendor::NVIDIA: if (backend_available(CUDA))  choice = CUDA;  break;
            case Vendor::AMD:    if (backend_available(ROCM))  choice = ROCM;  break;
            case Vendor::INTEL:  if (backend_available(INTEL)) choice = INTEL; break;
            default: break;
        }
        // THE DELETED FALLBACK: a GPU device NEVER resolves to HOST. Today
        // (queue-impl.cc:105-107) it does, and netlib_lapack.cc:199-214 then
        // runs cblas on the raw pointers -- silently 10-100x slow on
        // malloc_shared, a SEGFAULT on malloc_device, uncatchable by
        // require_device_accessible (queue-impl.cc:144-150 accepts
        // usm::alloc::device and short-circuits only for CPU *devices*).
        if (choice == AUTO) {
            throw std::runtime_error(
                "Queue::backend: this build generates no device code for '" +
                device_.get_name() + "'. Reconfigure with -DBATCHLAS_ENABLE_CUDA=ON, "
                "or construct the Queue on a CPU device.");
        }
    } else {
        if (!backend_available(HOST))
            throw std::runtime_error("Queue::backend: no host backend in this build.");
        choice = HOST;
    }
    resolved_ = choice;
    return choice;
}
```

`set_backend` stays **permissive** (build-availability check only, unchanged). `tests/linalg_layer_tests.cc:245` constructs `Queue(Device::default_device(), Backend::NETLIB)` on a GPU box as a deliberate host reference, and `python/batchlas/_api.py:42` does the same; adding a device/family assertion would break both. `device_serves()` exists for callers that want to ask.

### 4.2 Level 2 — route, per call

```
resolve_route<Op O, typename T>(Backend B, const Queue& q, const OpShape& s) -> Route

  cfg  = route_config(O)              # cached static; see §4.3
  mode = vendor_mode()                # cached static
  sk   = scalar_kind_of<T>

  compiled(r) = route_compiled(O, B, sk, r)
                # Vendor  -> Queue::library_available(library_for(O,B,r))
                #            per-LIBRARY, not per-family: a build with cuBLAS but
                #            no cuSOLVER must report Vendor unavailable for potrf
                #            and available for gemm.
                # FusedDevice -> library_available(CUBLASDX/CUSOLVERDX) AND the
                #            fused kernel exists for this shape/type. This is
                #            cublasdx_variant_needs_fallback, evaluated HERE and
                #            never inside the callee -- an implementation may not
                #            substitute a different route than the one resolved.
                # Native  -> the family is compiled and the kernel exists

  admissible(r):
      if !compiled(r): return false
      if !RouteTable<O,T>::supports(r, q, s): return false
      if r.origin == Vendor and mode == Forbid:
          # NOT a silent skip. A silent skip would slide to native and hide
          # exactly the gap this run exists to enumerate.
          if coverage_active(): coverage_record_miss(O, s, r); return false
          throw VendorForbidden(O, B, sk, s, r, library_for(O,B,r))
      return true

  # ---- 1. explicit force --------------------------------------------------
  if cfg.forced != {Auto,Auto}:
      for r in complete(cfg.forced, RouteTable<O,T>::order()):   # fill an Auto field
          if admissible(r): return with_library(r)
      if cfg.strict or cfg.src.legacy_throws_today:
          # `legacy_throws_today` reproduces trmm/syr2k's current behaviour:
          # BATCHLAS_TRMM_VARIANT=cublasdx on a build without cuBLASDx, or on a
          # non-float type, throws (cublas.cc:1001, route_common.hh:93-98). The
          # message quotes cfg.src.variable + "=" + cfg.src.value verbatim,
          # because tests/trmm_tests.cc:151 asserts on that literal.
          throw ForcedRouteUnavailable(O, B, sk, s, cfg.forced, cfg.src)
      # else fall through to Auto -- today's syev/gesvd/ormqr behaviour

  # ---- 2. measured routing table ------------------------------------------
  if r := RouteTable<O,T>::measured(q, s); r and admissible(*r): return with_library(*r)

  # ---- 3. preference partition --------------------------------------------
  cands = [r in RouteTable<O,T>::order() until Auto if admissible(r)]
  pref  = [r in cands if RouteTable<O,T>::preferred(r, q, s)]
  pool  = pref if pref nonempty else cands
  # DiagFullGemm is in no order() array, so Auto can never reach it -- exactly
  # as today (syrk_custom_dispatch.cc:31-34).
  if pool nonempty:
      if mode == PreferNative: pool = stable_partition(pool, origin != Vendor)
      chosen = pool[0]
      if cfg.log: log(op_name(O), to_string(B), to_string(chosen), s.describe())
      return with_library(chosen)

  # ---- 4. nothing serves this call ----------------------------------------
  throw NoRouteError(O, B, sk, s)
      # "batchlas: potrf on Backend::CUDA has no admissible route
      #  (n=512 batch=2048 T=float); vendor route requires cuSOLVER, which this
      #  build does not link; no native potrf exists."
```

### 4.3 Config caching and the 198 `ScopedEnvVar` sites

`route_config(Op)` and `vendor_mode()` are function-local statics guarded by a **generation counter**. `batchlas::dispatch::reload_env()` bumps it. `ScopedEnvVar`'s constructor and destructor (`tests/test_utils.hh`) call `reload_env()` — one edit, and it covers all 198 sites across `tests/` (15 files, `gemm_tests.cc` alone has 99) and the three committed `benchmarks/gemm_*` binaries. Any other in-process `setenv` must call it; that is a documented contract, not an inference.

Cost, against what it replaces per call: `policy_from_env` (`env.hh:52-56,118`) does two `std::string` allocations plus a `getenv`; `gemm_variant_request()` does a `getenv` plus a lowercase copy; `trmm` does it **twice**; `query_caps` (`context.hh:24-47`) does three SYCL `get_info` round-trips plus a `std::string` heap allocation. The resolver is strictly cheaper than the status quo on every op that dispatches.

### 4.4 What the two switches do, and what an op with no native impl does

| | `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` | `BATCHLAS_NO_VENDOR=1` (vendor-present build) |
|---|---|---|
| mechanism | every `BATCHLAS_HAS_<LIB>` is 0 → no vendor TU compiled, no vendor header included, no vendor library linked → `route_compiled(Vendor route)` is false | `vendor_mode() == Forbid` → `admissible()` throws on any vendor candidate |
| `gemm`, float/double/complex | works: `{Native, RegisterTiled}` via `sycl_gemm::gemm_custom`, which is instantiated for all four scalar types in `src/sycl/gemm_kernels.cc` and lives in the vendor-free `batchlas_sycl` component. Shapes outside `gemm_custom_problem_supported` throw `NoRouteError`. | same, plus `VendorForbidden` where the resolver would have chosen vendor |
| `symm`/`hemm`/`syrk`/`herk`/`syr2k`/`trmm` | **throws `NoRouteError`.** Verified: `symm_custom_dispatch.cc:160` calls `symm_vendor_cuda_raw` (defined `cublas.cc:1035`), `syrk_custom_dispatch.cc:214,235` → `cublas.cc:1046`, `syr2k_custom_dispatch.cc:181,187,197` → `cublas.cc:1056`, `trmm_custom_dispatch.cc:182-234` → `cublas.cc:1067`; all four `#include "cublasdx_dispatch_common.hh"`, which includes `<cuda_runtime_api.h>`; and the herk/her2k expand routes are inline in `cublas.cc:600-680` calling `gemm_vendor`. These are **not** portable today. Freeing them is WP1, not WP0. | same |
| `gemv`, `trsm`, `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `spmm`, `syev`, `gesvd`, `ormqr`, `iluk` | **throws** `NoRouteError` / `VendorForbidden` naming op, backend, scalar and shape. The library still **configures, compiles, links and loads** — that is what the per-library `vendor_absent.cc` stubs are for. | throws `VendorForbidden` |
| an op with **no native implementation at all** | primary `RouteTable` template says only `Origin::Vendor` supports it → step 3 finds no candidate → `NoRouteError` at the call. **Zero new code was required for that op.** Adding a native kernel later is one `RouteTable` specialisation plus one `case` in its facade. | identical, with `reason = "BATCHLAS_NO_VENDOR=1"` |
| `BATCHLAS_NO_VENDOR=warn` | n/a | never throws; records a coverage miss and lets the vendor route run, so **one** test run enumerates **all** gaps instead of aborting at the first |

**Honest statement of WP0's ceiling, to be repeated in the plan:** `ctest` under `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is **not** green when WP0 lands, and no naming scheme or dispatch mechanism can make it so — `ortho` alone pulls in `potrf`, `trsm`, `geqrf`, `orgqr` and `gemv`, and `ortho` sits under `syevx`, `lobpcg` and `lanczos`. WP0's deliverable is that the build **exists** and that the gap is **enumerated**, not closed. The plan's M1 must be restated accordingly.

---

## 5. CMake changes, variable by variable

### 5.1 `cmake/BatchLASOptions.cmake` (after line 153)

```cmake
option(BATCHLAS_ENABLE_VENDOR_BLAS
    "Build against vendor math libraries (cuBLAS/cuSOLVER/cuSPARSE, roc*, oneMKL, netlib, MathDx)" ON)
foreach(_lib CUBLAS CUSOLVER CUSPARSE CUBLASDX CUSOLVERDX
             ROCBLAS ROCSOLVER ROCSPARSE LAPACKE CBLAS ONEMKL)
    option(BATCHLAS_ENABLE_${_lib} "" ${BATCHLAS_ENABLE_VENDOR_BLAS})
    set(BATCHLAS_HAS_${_lib} FALSE)          # axis 3: library found
endforeach()

# axis 1: device families. NOT set from any find_library().
set(BATCHLAS_HAS_HOST_BACKEND  FALSE)
set(BATCHLAS_HAS_CUDA_BACKEND  FALSE)
set(BATCHLAS_HAS_ROCM_BACKEND  FALSE)
set(BATCHLAS_HAS_INTEL_BACKEND FALSE)        # replaces BATCHLAS_HAS_MKL_BACKEND
set(BATCHLAS_HAS_CPU_TARGET    FALSE)        # unchanged
```

`BATCHLAS_HAS_MKL_BACKEND` is deleted as a name. Where it currently means "oneMKL supplies cblas.h" (`linalg-impl.hh:20`), it becomes `BATCHLAS_HAS_ONEMKL`.

### 5.2 `cmake/BatchLASDependencies.cmake`

**The decoupling, in three lines.** Delete the assignment at `:118-123`:

```cmake
    if(CUBLAS_LIBRARY)
        set(BATCHLAS_HAS_CUDA_BACKEND TRUE PARENT_SCOPE)   # <-- THE DEFECT
```

replace with `set(BATCHLAS_HAS_CUBLAS TRUE PARENT_SCOPE)`, and set the family from the hardware:

```cmake
# BatchLASDetectSYCL.cmake:180-184 already states in a comment that these are
# two different questions ("The NVPTX codegen flags below are keyed on the
# *hardware*, not on the CUDA backend option ... BATCHLAS_ENABLE_CUDA only
# governs the cuBLAS/cuSOLVER backend"). This promotes that comment to code.
if(BATCHLAS_CUDA_ENABLED)
    set(BATCHLAS_HAS_CUDA_BACKEND TRUE)
endif()
if(BATCHLAS_DETECTED_AMD_GPU OR BATCHLAS_ENABLE_ROCM)
    set(BATCHLAS_HAS_ROCM_BACKEND TRUE)
endif()
if(BATCHLAS_HAS_CPU_TARGET OR BATCHLAS_HAS_LAPACKE)
    set(BATCHLAS_HAS_HOST_BACKEND TRUE)
endif()
# BATCHLAS_HAS_INTEL_BACKEND stays FALSE in WP0. Today, setting it TRUE opens
# `case Backend::MKL:` in with_backend (queue-dispatch.hh:44-46), makes
# backend_available() true (queue-impl.cc:68) and resolves Intel GPUs to it
# (queue-impl.cc:101-103), while src/backends/CMakeLists.txt:15-19 compiles
# mkl.cc -- the only TU with Backend::MKL instantiations -- not at all. Wiring
# an enum value to nothing is the exact defect this WP exists to remove, and
# fixing it properly needs Intel hardware nobody here has.
if(BATCHLAS_HAS_ONEMKL)
    message(WARNING "oneMKL found but Backend::INTEL is not built in this release; "
                    "oneMKL will be used only as the host CBLAS provider.")
endif()
```

**Per-library probes.** Wrap `find_nvidia_libs()`, `find_rocm_libs()`, `find_onemkl_libs()`, `find_netlib_libs()` in `if(NOT BATCHLAS_ENABLE_VENDOR_BLAS) return() endif()`, and probe each library separately:

- cuBLAS (existing `find_library` at `:86-116`) → `BATCHLAS_HAS_CUBLAS`
- **NEW** `find_library(CUSOLVER_LIBRARY NAMES cusolver)` → `BATCHLAS_HAS_CUSOLVER`
- **NEW** `find_library(CUSPARSE_LIBRARY NAMES cusparse)` → `BATCHLAS_HAS_CUSPARSE`
  (today they are *never probed* — pulled in blind at `:298-305` on a flag they did not influence)
- MathDx presence (`BATCHLAS_MATHDX_TARGETS` / `__has_include(<cublasdx.hpp>)`) → `BATCHLAS_HAS_CUBLASDX`, `_CUSOLVERDX`
- hipBLAS (`:169`) → `BATCHLAS_HAS_ROCBLAS`; **NEW** separate probes → `_ROCSOLVER`, `_ROCSPARSE` (today `rocsolver`/`hipsparse` are appended to the link list even when `-NOTFOUND`, `:166`)
- `:232-240` → `BATCHLAS_HAS_LAPACKE` and `BATCHLAS_HAS_CBLAS` independently; the `else()` branch **no longer sets `BATCHLAS_HAS_HOST_BACKEND FALSE`** — a missing netlib removes implementations, not the host device
- `:3-21` and `:200-205` (two independent setters) → one setter for `BATCHLAS_HAS_ONEMKL`

**Link libraries**, replacing the en-bloc `:298-305`:

```cmake
set(BATCHLAS_CUDA_LINK_LIBRARIES CUDA::cudart)
if(BATCHLAS_HAS_CUBLAS)   list(APPEND BATCHLAS_CUDA_LINK_LIBRARIES CUDA::cublas)   endif()
if(BATCHLAS_HAS_CUSOLVER) list(APPEND BATCHLAS_CUDA_LINK_LIBRARIES CUDA::cusolver) endif()
if(BATCHLAS_HAS_CUSPARSE) list(APPEND BATCHLAS_CUDA_LINK_LIBRARIES CUDA::cusparse) endif()
```

`enable_language(CUDA)` stays under the family flag (the `.cu` TUs need it), but all six `.cu` files are MathDx wrappers and are additionally vendor-gated.

**Delete** the duplicate definition channel at `:411-430` (`target_compile_definitions(... BATCHLAS_HAS_X=1)`); the generated header becomes the single source.

### 5.3 `cmake/backend_config.h.in`

```c
/* axis 1 -- device families */
#cmakedefine01 BATCHLAS_HAS_HOST_BACKEND
#cmakedefine01 BATCHLAS_HAS_CUDA_BACKEND
#cmakedefine01 BATCHLAS_HAS_ROCM_BACKEND
#cmakedefine01 BATCHLAS_HAS_INTEL_BACKEND
#cmakedefine01 BATCHLAS_HAS_CPU_TARGET
/* axis 3 -- vendor libraries present */
#cmakedefine01 BATCHLAS_HAS_CUBLAS
#cmakedefine01 BATCHLAS_HAS_CUSOLVER
#cmakedefine01 BATCHLAS_HAS_CUSPARSE
#cmakedefine01 BATCHLAS_HAS_CUBLASDX
#cmakedefine01 BATCHLAS_HAS_CUSOLVERDX
#cmakedefine01 BATCHLAS_HAS_ROCBLAS
#cmakedefine01 BATCHLAS_HAS_ROCSOLVER
#cmakedefine01 BATCHLAS_HAS_ROCSPARSE
#cmakedefine01 BATCHLAS_HAS_LAPACKE
#cmakedefine01 BATCHLAS_HAS_CBLAS
#cmakedefine01 BATCHLAS_HAS_ONEMKL

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND || BATCHLAS_HAS_INTEL_BACKEND
  #define BATCHLAS_HAS_GPU_BACKEND 1
#else
  #define BATCHLAS_HAS_GPU_BACKEND 0
#endif
#if BATCHLAS_HAS_CUBLAS || BATCHLAS_HAS_ROCBLAS || BATCHLAS_HAS_ONEMKL || BATCHLAS_HAS_LAPACKE
  #define BATCHLAS_HAS_ANY_VENDOR 1
#else
  #define BATCHLAS_HAS_ANY_VENDOR 0
#endif
```

### 5.4 `src/backends/CMakeLists.txt`

```cmake
set(BACKEND_COMMON_SOURCES backend_handle_instantiations.cc)

if(BATCHLAS_HAS_LAPACKE AND BATCHLAS_HAS_CBLAS)
    list(APPEND BACKEND_HOST_SOURCES netlib_lapack.cc)
endif()

if(BATCHLAS_HAS_CUBLAS)   list(APPEND BACKEND_CUDA_SOURCES cublas.cc)   endif()
if(BATCHLAS_HAS_CUSOLVER) list(APPEND BACKEND_CUDA_SOURCES cusolver.cc) endif()
if(BATCHLAS_HAS_CUSPARSE) list(APPEND BACKEND_CUDA_SOURCES cusparse.cc) endif()
if(BATCHLAS_HAS_CUBLASDX AND BATCHLAS_ENABLE_CUBLASDX_WRAPPER)
    list(APPEND BACKEND_CUDA_SOURCES gemm_cublasdx.cu gemm_cublasdx_dispatch.cc
        symm_cublasdx_fused.cu syrk_cublasdx_fused.cu
        syr2k_cublasdx_fused.cu trmm_cublasdx_fused.cu)
endif()
if(BATCHLAS_HAS_CUSOLVERDX) list(APPEND BACKEND_CUDA_SOURCES cusolverdx.cc cusolverdx_kernels.cu) endif()

# The four *_custom_dispatch.cc call symm/syrk/syr2k/trmm_vendor_cuda_raw
# (cublas.cc:1035-1067) and include <cuda_runtime_api.h> transitively. They are
# NOT portable today and stay gated on cuBLAS until WP1 rewrites their fallback
# path. Gating them on the family would produce undefined references.
if(BATCHLAS_HAS_CUBLAS)
    list(APPEND BACKEND_CUDA_SOURCES symm_custom_dispatch.cc syrk_custom_dispatch.cc
        syr2k_custom_dispatch.cc trmm_custom_dispatch.cc)
endif()

if(BATCHLAS_HAS_ROCBLAS)   list(APPEND BACKEND_ROCM_SOURCES rocblas.cc)   endif()
if(BATCHLAS_HAS_ROCSOLVER) list(APPEND BACKEND_ROCM_SOURCES rocsolver.cc) endif()
if(BATCHLAS_HAS_ROCSPARSE) list(APPEND BACKEND_ROCM_SOURCES rocsparse.cc) endif()
# mkl.cc is DELETED from the tree; the commented-out block goes with it.
```

`src/dispatch/absent/` is added in the same file, one TU per library, each compiled only when its family is on and its library is off:

```cmake
if(BATCHLAS_HAS_CUDA_BACKEND AND NOT BATCHLAS_HAS_CUBLAS)
    list(APPEND DISPATCH_SOURCES absent/cublas_absent.cc)
endif()
# ... _CUSOLVER, _CUSPARSE, _ROCBLAS, _ROCSOLVER, _ROCSPARSE, _LAPACKE
```

This is **one block per library, not 2ⁿ per family**, which is the fix for the combinatorial objection: a build with cuBLAS but no cuSOLVER compiles exactly `cusolver_absent.cc`.

### 5.5 `src/CMakeLists.txt`

- `add_library(batchlas_backends_cuda_obj OBJECT)` gate changes from `BATCHLAS_HAS_CUDA_BACKEND` to `BATCHLAS_HAS_CUBLAS OR BATCHLAS_HAS_CUSOLVER OR BATCHLAS_HAS_CUSPARSE OR BATCHLAS_HAS_CUBLASDX OR BATCHLAS_HAS_CUSOLVERDX`; `CUDA_ARCHITECTURES` properties stay keyed on the family.
- `target_link_libraries(batchlas_backends_cuda PRIVATE ${BATCHLAS_CUDA_LINK_LIBRARIES})` moves under `if(BATCHLAS_HAS_CUBLAS OR ...)`; netlib link (`:205-218`) moves to `BATCHLAS_HAS_LAPACKE`; rocm to `BATCHLAS_HAS_ROCBLAS`.
- **NEW** `add_library(batchlas_dispatch_obj OBJECT)` → `batchlas_dispatch` SHARED, appended to `BATCHLAS_OBJECT_LIBS` and `BATCHLAS_COMPONENT_TARGETS`. It holds the resolver, the env parser, the coverage emitter, the entry-point facade and the `absent/` stubs. It contains **no kernels**, so it adds nothing to the device link. The new cross-`.so` edge (facade → `batchlas_backends_cuda`) is a 15th cycle in a graph that already has 14 and is already covered by the global `-Wl,--no-as-needed` at `src/CMakeLists.txt:284-296`.
- The commented-out MKL block at `:259-263` is deleted.

### 5.6 Acceptance for requirement (c), build half

```
cmake -B build-novendor -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON .
```
configures with `BATCHLAS_HAS_CUDA_BACKEND=1` and every `BATCHLAS_HAS_<lib>=0`, compiles zero vendor sources, includes zero vendor headers, links zero vendor libraries, dispatches on an NVIDIA GPU, and runs native GEMM. Mechanically checkable: `grep -L cublas $(ldd libbatchlas_backends_cuda.so)` and `nm -D` on the components.

---

## 6. File-by-file change list, ordered so each step compiles

### S1 — additive CMake probes. **Mechanical.** ~120 lines.
`cmake/BatchLASOptions.cmake`, `cmake/BatchLASDependencies.cmake`, `cmake/backend_config.h.in`. Introduce every `BATCHLAS_ENABLE_<LIB>` / `BATCHLAS_HAS_<LIB>` and emit them into the generated header; probe cuSOLVER/cuSPARSE/rocSOLVER/rocSPARSE/MathDx separately. **Family flags keep their current derivation.** Nothing in C++ reads the new macros yet; the build is bit-identical.

### S2 — re-key vendor headers and vendor types onto library macros. **Judgement.** ~250 lines.
`src/linalg-impl.hh` (the include block at `:10-42` and all 45 `#if BATCHLAS_HAS_*_BACKEND` sites), `src/backends/backend_handle_impl.hh` (`:6-11` includes, `:29-33`, `:41-60`, the descriptor members), `src/backends/cusolverdx.hh/.cc`. Requires:
- an **empty primary `LinalgHandle<Backend::CUDA>`** for the no-cuBLAS case, because `src/extensions/ortho.cc:104` declares `static LinalgHandle<B> handle;` in a *native* TU and `linalg-impl.hh:58` declares the template with no primary definition;
- `BackendMatrixHandle`'s cuSPARSE descriptor members and `initialize_cuda_descriptors` gated on `BATCHLAS_HAS_CUSPARSE` (it is compiled unconditionally into `batchlas_backends_obj`, so this is the one vendor-touching TU outside the vendor object library);
- `#if !BATCHLAS_HAS_MKL_BACKEND` → `#if !BATCHLAS_HAS_ONEMKL` at `linalg-impl.hh:20`.
Compiles unchanged because library flags currently equal family flags. This step is the largest under-count in every input design and the reason S2 is separate.

### S3 — per-library source and link gating. **Mechanical.** ~80 lines.
`src/backends/CMakeLists.txt`, `src/CMakeLists.txt`. Still bit-identical on a full-vendor box.

### S4 — the dispatch vocabulary. **Judgement.** ~700 lines, ~250 of them moved.
New `include/batchlas/blas/dispatch/{route.hh,plan.hh}`; rewritten `op.hh`; new `src/dispatch/{route_env.cc,resolve.cc}`; delete `provider.hh`, `env.hh`, `context.hh`. Rewrite the three choosers into `RouteTable` specialisations in `syev.hh`, `gesvd.hh`, `ormqr.hh`, moving `syev_saturated_provider_for_n*` and `default_order_gesvd`'s measurement comment **verbatim**. Delete the five private route enums and the two shared parsers in `gemm_variant.hh`, `route_common.hh`, and the four `*_custom_dispatch.cc`, splitting each predicate into `supports` (correctness) and `preferred` (measured window). `tests/test_utils.hh` gains the `reload_env()` call; `tests/gesvd_tests.cc:1198-1214` and `tests/backend_dispatch_tests.cc` updated; `python/batchlas/bindings/support.hh:399-410` drops `sycl`/`magma`, accepts `host`/`intel`, and `features["has_cuda_backend"]` is renamed with the old key retained. Behaviour-preserving; verified by re-running `tests/{gemm,symm,syrk,syr2k,trmm,herk,hemm,her2k,gesvd,sytrd_*}_tests` which already sweep every legacy env spelling.

### S5 — move the 21 public op definitions into the facade. **Mechanical, large.** ~900 lines, almost all relocation.
New `src/dispatch/entry_points/{level3,level2,factorization,eigen,sparse}.cc`. From `src/backends/cublas.cc:1568-1766`, `cusolver.cc`, `cusparse.cc`, `rocblas.cc`, `rocsolver.cc`, `rocsparse.cc`, `netlib_lapack.cc`: remove the public `gemm/gemv/trsm/...` definitions and their instantiation blocks; add `sig::*_vendor` aliases beside the existing `sig::` aliases in `include/batchlas/blas/functions/*.hh`; instantiate `backend::*_vendor<B,T>` in each vendor TU. `Backend::MKL` instantiations in `src/extensions/{symm,syrk,syr2k,trmm,ritz_values}.cc` and `src/extra/norm.cc:88` are deleted with `mkl.cc`. Each op's move must be atomic across all four vendor TUs — see §9 risk 1.

### S6 — the two gates. **Judgement.** ~450 lines.
`src/dispatch/absent/*.cc` (7 stub TUs); the typed `op_external` at every vendor leaf; the debug assertion inside `call_backend`/`call_backend_nh`; `src/util/queue-impl.cc` resolver (delete the GPU→HOST fallback, add `library_available`/`vendor_available`/`device_serves`); `include/batchlas/blas/queue-dispatch.hh` (`MKL`→`INTEL` case, `kProbeBackend` → `HOST`, `static_assert` message). After this step `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` configures, builds, links and runs.

### S7 — the coverage instrument. **Mechanical.** ~200 lines.
`src/dispatch/coverage.cc` + a `ctest` wrapper target `batchlas_coverage`. See §7.

### S8 — retire the four "our kernel is wired here" sites. **Judgement.** ~40 lines.
`src/extensions/ormqr_blocked.cc:101-112` (`route_has_tile_kernel = (B == Backend::CUDA)`, whose own comment says *"Not a statement about CUDA the vendor"*), `src/extensions/sytrd_blocked.cc:814-816`, `src/extensions/ortho.cc:171-175`, and `syev.hh:778`'s measured-grid guard. Each becomes `dispatch::route_compiled(Op::syrk, B, sk, {Native, GramTiles}) && RouteTable<...>::supports(...)` — the question those sites were always asking. Without this step the whole exercise is cosmetic; with it the separation is real. Also `src/extensions/ortho.cc:109` and `stedc.cc:24-32`, which use `B == Backend::NETLIB` to mean "no device kernels here": re-express as `!ctx.device().is_gpu()`, preserving today's outcome exactly.

**Untouched throughout:** the 46 extension instantiation ladders, the 105 `BATCHLAS_DISPATCH_ON_QUEUE` sites, `with_backend`, `template-instantiations.hh`, `BackendMatrixHandle`'s shape, every kernel body, every tuning header. **Instantiation-count delta: zero.**

---

## 7. The instrumentation

### 7.1 `op_external` as counter and throw point

Today `op_external(const char*, F&&)` is a no-op with 19 call sites, none of them on the level-3 or factorisation paths (verified: `cusolver`×4, `cublas`×2, `rocsolver`×4, `cusolverdx`×4, `netlib`×5). It cannot be the primary gate. It becomes the **backstop and the counter**, and the resolver stays the primary gate.

```cpp
template <class F>
decltype(auto) op_external(const ExternalCall& c, F&& f) {
    coverage::record_external(c);                     // op x lib x scalar x shape_class
    if (dispatch::vendor_mode() == VendorMode::Forbid && !coverage::warn_mode())
        throw dispatch::VendorForbidden(c);           // names op, library, backend, shape
    if (dispatch::log_enabled()) log_external(c);
    detail::ExternalScope scope(c);                   // thread-local, for the assertion below
    return std::forward<F>(f)();
}
```

Placement, mechanically: at the top of every `backend::*_vendor_impl` / `*_vendor` leaf in `cublas.cc`, `cusolver.cc`, `cusparse.cc`, `rocblas.cc`, `rocsolver.cc`, `rocsparse.cc`, `netlib_lapack.cc`, `cusolverdx.cc` — ~60 sites, each already beginning `static LinalgHandle<Back> handle; handle.setStream(ctx);`, so the insertion is one line above an existing line. Plus `backend_handle_impl.hh:52-60` for the cuSPARSE descriptor creation, which no leaf covers.

**Completeness check, not faith.** `call_backend` and `call_backend_nh` (`linalg-impl.hh:626,652,...`) already carry `BackendLibrary BL` as a template parameter and are the actual vendor invocation for 47 of the 62 vendor calls. In a `-UNDEBUG` build they assert that an `ExternalScope` is active, so any vendor call that escaped a leaf wrapper fails a debug test rather than going silently uncounted. `cusparse.cc` uses zero `call_backend` (it calls `cusparseSpMM` directly), so its leaves are wrapped explicitly.

**Why this catches what the resolver cannot.** Five extension sites call the vendor entry point directly, bypassing the public entry and therefore the resolver: `src/extensions/syevx_lobpcg.cc:652,1282` (`backend::syev_vendor` in the Rayleigh–Ritz solve, every iteration), `syev_two_stage.cc:372`, `syevx_direct_subset.cc:380` (`backend::ormqr_vendor`), `src/extra/norm.cc:45`, `cond.cc:52`. These are exactly the "native algorithm secretly reaching a vendor" cases the coverage table exists to find, and `op_external` sees all of them with full op/library/shape.

**Do not put the gate in `setStream` or `submit_host_task`.** `src/extensions/ortho.cc:104-106` calls `handle.setStream(ctx)` at the top of a native function, before any routing decision; gating there would make every `ortho` call throw under `BATCHLAS_NO_VENDOR=1` regardless of route, poisoning the very burn-down list the run produces.

### 7.2 The coverage table

```
BATCHLAS_NO_VENDOR=warn BATCHLAS_COVERAGE_OUT=cov.csv ctest --test-dir build
```

`warn` mode never throws: `admissible()` records the miss and lets the vendor route run, so **one** run enumerates **all** gaps instead of aborting at the first. The counters are a fixed-size array indexed by `(Op, ScalarKind, Backend, BackendLibrary, shape_class)` — `shape_class()` buckets `max(m,n,k)` and `batch` by power of two, so a 10 000-iteration test yields one row. Emitted from an `atexit` handler that touches **no SYCL object** (the standing static-destruction rule).

Columns: `op, scalar, backend, shape_class, m, n, k, batch, chosen_origin, chosen_algo, library, calls, native_route_existed, native_route_supported`.

The last two columns distinguish the three states WP3–WP8 must burn down: *no native kernel exists*, *a native kernel exists but does not support this shape/type*, and *a native kernel exists and was simply not preferred*. A second, **static** table is generated at build time by iterating `route_compiled` over `(Op × Backend × ScalarKind × Route)` with no kernel run — it says what is *linked*, which is a different and equally necessary fact from what a test run *reached*.

Two additional gates to run in CI:
- `BATCHLAS_NO_VENDOR=1 ctest -R <op>_tests` per op — proves the throw carries op and shape.
- `ctest` under `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` — expected red; the *set* of failures is recorded as the WP3–WP8 burn-down baseline, and any change to that set is a reviewable diff.

---

## 8. Explicitly out of scope for WP0

1. **Relocating the four `*_custom_dispatch.cc` out of the CUDA object library (WP1).** They call `symm/syrk/syr2k/trmm_vendor_cuda_raw` (`cublas.cc:1035-1067`) and include `<cuda_runtime_api.h>`; that is a rewrite, not a re-gate.
2. **Freeing the herk/her2k expand routes** (`cublas.cc:600-680`, which call `gemm_vendor` directly) and `symmetric_product_fold.hh`. Same reason; same work package.
3. **Any new numerical kernel.** WP0 builds the instrument, not the coverage.
4. **Making `Backend` a runtime parameter / collapsing the instantiation matrix.** Its payoff is an unmeasured build-time win on a box where seven standard build-time fixes are already measured dead, and it multiplies risk across 654 declarations for no WP0 requirement.
5. **Flipping `BATCHLAS_GEMM_VARIANT`'s `Vendor` default.** It is a measurement deliverable (WP2), and flipping it inside a refactor makes any regression unattributable.
6. **The four per-site tuning knobs** (`BATCHLAS_ORTHO_GRAM`, `BATCHLAS_SYTRD_FUSE_PANEL_UPDATE`, `BATCHLAS_ORMQR_IMPL`, `BATCHLAS_SYEVX_ALGORITHM`, `WyPin`). They select *within* one implementation; folding them into `Route` would repeat exactly the category error `Provider` makes.
7. **`SyevxAlgorithm`, `OrthoAlgorithm` and the other public algorithm-request enums.** They change the semantics of the *answer*, not the implementation of the same math.
8. **`Backend::INTEL` / oneMKL revival.** It cannot be tested here; WP0 only removes the dead branch that produces undefined references today.
9. **The USM host-reachability blind spot** (`queue-impl.cc:144-150` accepts `usm::alloc::device`). A real bug, orthogonal, and the deleted GPU→HOST fallback removes its worst consequence.
10. **`with_backend`, `BATCHLAS_DISPATCH_ON_QUEUE`, `extensions.hh`'s 70 entry points, `spmm`'s `MatrixFormat` axis, oneDPL, the `.so` merge.** All correct as-is or unrelated.

---

## 9. The three biggest risks in this spec

### Risk 1 — S5 edits ROCm and oneMKL code that cannot be compiled on this machine, and each op's move must be atomic across four TUs.

`gemm<Backend::CUDA,float>` is defined at `cublas.cc:1568-1580` and instantiated at `:1818`. Removing it from `cublas.cc` and adding it to the facade must land in the same commit as the identical edit to `rocblas.cc` and `netlib_lapack.cc`, or the build has either duplicate or undefined symbols. Neither `rocblas.cc`/`rocsolver.cc`/`rocsparse.cc` nor `mkl.cc` compiles here.

**Mitigation.** (a) Delete `mkl.cc` and all `Backend::MKL` instantiations outright in S5 — it is not compiled today (`src/backends/CMakeLists.txt:15-19`), so nothing regresses and one of the four TUs disappears. (b) Add a CI job that runs `clang++ -fsyntax-only -DBATCHLAS_HAS_ROCM_BACKEND=1 -DBATCHLAS_HAS_ROCBLAS=1 …` against the three ROCm TUs using the ROCm headers from a container image; this catches every error S5 can introduce, since the edits are declaration/instantiation-level. (c) Do S5 **one op at a time**, `gemm` first, so a mistake is one op wide. (d) Land the ROCm edits in the same commit as the CUDA ones, never after.

### Risk 2 — the "verbatim heuristics" are env readers, and splitting them wrong silently changes the default route of the hottest op.

`gemm_use_sycl_custom` (`gemm_variant.hh:142-198`) *begins* with `gemm_variant_request()` and returns false for `Vendor`; only then does it check `gemm_custom_problem_supported`, then `request == Sycl → true` (bypassing the GPU check and the window), then the shape envelope. It is three things in one function: an env read, a correctness gate, and a measured window. Moving it "verbatim" into `supports()` breaks `BATCHLAS_GEMM_ROUTE=native`; moving the window into `supports()` makes a 1024³ float GEMM at batch 256 have *no supported route* under vendor-off, breaking an op that works today. The same three-way split is needed for `symm_prefer_cuda_custom_heuristic` (`symm_custom_dispatch.cc:52-68`) and `syrk_prefer_{gram,triangular}_tiles`.

**Mitigation.** The split is specified, not left to the implementer: **`supports` = `gemm_custom_problem_supported` and nothing else; `preferred` = everything from `ctx.device().type != GPU` downward including the complex exclusion and the `128 ≤ max_dim ≤ 512, batch ≥ 64` window; the env read moves to the alias table.** Acceptance: run `benchmarks/gemm_128x32x32_family_benchmark`, `gemm_128x64x32_family_benchmark`, `gemm_heterogeneous_benchmark` and `tests/gemm_tests.cc` (99 `ScopedEnvVar` cases) before and after S4 and diff the *chosen route* under `BATCHLAS_DISPATCH_LOG=1` for every case — not the timing, the route. Any differing case is a bug, not a tuning question. `order_gemm` must also carry an explicit heterogeneous-batch arm, or `gemm_heterogeneous_vendor_impl` (`cublas.cc:60,171`) becomes unreachable.

### Risk 3 — legacy env provenance: one non-literal alias, and a cache that can go stale mid-process.

`BATCHLAS_TRMM_VARIANT=triangular` is read by **two independent parsers** today: `parse_cublasdx_variant_request` maps it to `Auto` while `trmm_triangular_requested()` returns true, so the net state at `trmm_custom_dispatch.cc:133-139` is "tile kernel pinned, cuBLASDx heuristic still live". Collapsing that to one forced `{Native, TriangularTiles}` is the one translation in §3.3 that is not a literal renaming. Separately, `route_config` is cached, while 198 `ScopedEnvVar` sites and three committed benchmark binaries set these variables *mid-process*.

**Mitigation.** (a) The `trmm` alias is gated on an explicit A/B: run `tests/trmm_tests.cc` and the trmm benchmark under `=triangular`, `=tiles`, `=vendor`, `=cublasdx` and unset, before and after, and diff both the chosen route and the numbers; if the pinned state differs, the alias becomes a two-token list `tiles,fused` that reproduces it exactly. (b) `ScopedEnvVar`'s ctor and dtor call `dispatch::reload_env()`; a test asserts that setting a variable after a first resolve changes the next resolve. (c) `ForcedRouteUnavailable` carries `RouteRequestSource` and quotes `variable=value` verbatim, so `tests/trmm_tests.cc:151`'s assertion on the literal `"BATCHLAS_TRMM_VARIANT=cublasdx"` keeps passing, and a forced-but-unavailable legacy request keeps **throwing** rather than silently falling through to cuBLAS under a cuBLASDx label. (d) The `native`→`vendor` collapse in `BATCHLAS_GEMM_VARIANT` gets its own explicit deprecation sentence, because `output/` provenance distinguishes tags that were always the same measurement.