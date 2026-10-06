// trmm transcriber (temporary: docs/design/flat-select-l3/trmm.md, tuned/README.md).
//
// Evaluates the ff340fc6 trmm routing -- the level3.cc entry gate, trmm_custom_dispatch.cc,
// level3_vendor_fallback.cc and cublas.cc's trmm_vendor / trmm_vendor_impl -- on stub types,
// with the predicate bodies copied VERBATIM (each block below names its source lines;
// trmm_gate.py --fidelity diffs them against `git show ff340fc6:<file>`). Stubs: a GPU queue,
// every pin Auto, the expansion always fits (capacities unlimited; can_run re-applies them),
// the cuBLASDx fused kernel absent (MathDx is absent on both boxes, so it never ran).
//
// Build (host only, no BatchLAS headers), once per build flavour:
//   g++ -std=c++20 -O1 -DVENDOR_BUILD=1 tools/transcribe/trmm_transcribe.cc -o /tmp/trmm_tr_v
//   g++ -std=c++20 -O1 -DVENDOR_BUILD=0 tools/transcribe/trmm_transcribe.cc -o /tmp/trmm_tr_vf
// Grid (vendor flavour):  /tmp/trmm_tr_v grid > tuned/transcribed/trmm.csv
//   python3 scripts/sweep_to_table.py --transcribe tuned/transcribed/trmm.csv --sha ff340fc6
// Points: /tmp/trmm_tr_v points <file>, lines "dtype side order q batch", prints the old Auto
// choice (triangular | expand | vendor, or throw) after each.
//
// A row lists the old vendor-present Auto choice, then every other candidate that is
// structurally runnable at the cell (homogeneous GPU batch, unlimited capacity) in candidate
// order, vendor last: Left `triangular|expand|vendor`, Right `expand|vendor`, every dtype.

#include <complex>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef VENDOR_BUILD
#error "build with -DVENDOR_BUILD=0 or 1"
#endif
#define BATCHLAS_HAS_CUBLAS VENDOR_BUILD

namespace batchlas {

enum class Side { Left, Right };
enum class Uplo { Lower, Upper };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Diag { NonUnit, Unit };
enum class Backend { CUDA };
enum class Op { trmm };
enum class MatrixFormat { Dense };

struct unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
struct NoRoute : std::runtime_error { using std::runtime_error::runtime_error; };

template <typename T, MatrixFormat F>
struct MatrixView {
    int r = 0, c = 0, b = 1;
    bool het = false;
    int rows() const { return r; }
    int cols() const { return c; }
    int batch_size() const { return b; }
    bool is_heterogeneous() const { return het; }
};

struct Queue { bool gpu = true; };
struct Event { std::string route; };

namespace select {
template <Backend B>
inline constexpr bool level3_vendor_available = bool(VENDOR_BUILD);
template <Backend B>
inline constexpr const char* kLevel3Library = "cuBLAS";
template <typename T>
[[noreturn]] inline void throw_no_vendor_route(Op, Backend, const char*) { throw NoRoute("no vendor"); }
}  // namespace select

namespace backend {
namespace detail {
enum class Level3Pin { Auto, Native, Vendor, Cublasdx, Expand, Triangular, Gram };
inline Level3Pin level3_pin(const char*, std::initializer_list<Level3Pin>) { return Level3Pin::Auto; }
inline bool is_gpu_queue(const Queue& q) { return q.gpu; }
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };
struct Level3Variant { Uplo uplo; Side side; Diag diag; Transpose transA; };
inline void record_level3_route(Op, const char*, long, long, long, long, int, Level3Variant = {}) {}
[[noreturn]] inline void throw_forced_cublasdx_unavailable(const char*, const std::string& r) {
    throw unsupported(r);
}
struct FusedResult {
    enum class Outcome { Ran, NoKernel, Declined };
    Outcome outcome = Outcome::NoKernel;
    Event event;
};
template <class... A>
FusedResult trmm_fused_try(A&&...) { return {}; }  // MathDx absent: never Ran
template <typename T>
Event trmm_triangular_tiles(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                            const MatrixView<T, MatrixFormat::Dense>&,
                            const MatrixView<T, MatrixFormat::Dense>&, T, Uplo, Transpose, Diag) {
    return {"triangular"};
}
template <typename T>
std::size_t expanded_workspace_bytes(Queue&, int, int) { return 0; }
inline bool expansion_fits(const Queue&, int, int, std::size_t) { return true; }  // capacity unlimited
[[noreturn]] inline void no_vendor(Op) { throw NoRoute("no vendor"); }

// VERBATIM src/backends/trmm_triangular_tiles.hh:352-373 (ff340fc6)
template <typename T>
bool trmm_tiles_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          const MatrixView<T, MatrixFormat::Dense>& C,
                          Side side) {
    if (side != Side::Left) {
        return false;
    }
    if (A.rows() != A.cols() || A.rows() != C.rows()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    if (B.rows() != C.rows() || B.cols() != C.cols()) {
        return false;
    }
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    return C.rows() > 0 && C.cols() > 0;
}
// END VERBATIM
}  // namespace detail

// cublas.cc's trmm_vendor_impl, reduced to its route decision.
template <Backend Back, typename T>
Event trmm_vendor_impl(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>&,
                       T, Side side, Uplo, Transpose, Diag) {
    const int k = side == Side::Left ? B.rows() : B.cols();
// VERBATIM src/backends/cublas.cc:661-662 (ff340fc6)
        const std::size_t expansion_bytes = detail::expanded_workspace_bytes<T>(ctx, k, A.batch_size());
        if (detail::expansion_fits(ctx, k, A.batch_size(), expansion_bytes)) {
// END VERBATIM
        return {"expand"};
    }
    return {"vendor"};
}

// VERBATIM src/backends/cublas.cc:785-795 (ff340fc6)
    Event trmm_vendor_cuda_raw(Queue& ctx,
                               const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               float alpha,
                               Side side,
                               Uplo uplo,
                               Transpose transA,
                               Diag diag) {
        return trmm_vendor_impl<Backend::CUDA, float>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
// END VERBATIM

namespace detail {
// VERBATIM src/backends/level3_vendor_fallback.cc:86-102 (ff340fc6)
Event trmm_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag) {
#if BATCHLAS_HAS_CUBLAS
    return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
#else
    (void)ctx; (void)A; (void)B; (void)C; (void)alpha; (void)side; (void)uplo;
    (void)transA; (void)diag;
    no_vendor(Op::trmm);
#endif
}
// END VERBATIM
}  // namespace detail

namespace {
// VERBATIM src/backends/trmm_custom_dispatch.cc:21-56 (ff340fc6)
constexpr int kTrmmCublasDxTile = 32;

// BATCHLAS_TRMM_ROUTE: vendor, the expansion plus GEMM; triangular or native, the
// tile kernel; cublasdx, the fused MathDx kernel (throws when it cannot run).
detail::Level3Pin trmm_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("trmm", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Triangular,
                                       Level3Pin::Cublasdx});
}

// Every left-side float problem with a homogeneous batch. The kernel indexes
// both operands as base + batch * stride, which a batch of unrelated pointers
// or differing shapes is out of reach of; everything else it handles, because
// uplo, trans and diag are loop bounds and a staging mask rather than separate
// kernels.
bool trmm_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               Side side) {
    if (side != Side::Left) {
        return false;
    }
    if (A.rows() != A.cols() || A.rows() != C.rows()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    if (B.rows() != C.rows() || B.cols() != C.cols()) {
        return false;
    }
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    return C.rows() > 0 && C.cols() > 0;
}
// END VERBATIM
// VERBATIM src/backends/trmm_custom_dispatch.cc:78-110 (ff340fc6)
bool trmm_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Side side,
                            Uplo uplo,
                            Transpose transA) {
    if (side != Side::Left || uplo != Uplo::Lower || transA != Transpose::NoTrans) {
        return false;
    }
    if (A.rows() != A.cols()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    return A.rows() == B.rows() && B.rows() == C.rows() && B.cols() == C.cols() && A.rows() > 0 && B.cols() > 0;
}

bool trmm_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& B) {
    if (A.rows() < kTrmmCublasDxTile || B.cols() < kTrmmCublasDxTile) {
        return false;
    }

    const int output_tile_rows = detail::ceil_div(A.rows(), kTrmmCublasDxTile);
    const int output_tile_cols = detail::ceil_div(B.cols(), kTrmmCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_cols;
    return tiled_work >= 8;
}

[[noreturn]] void throw_forced_trmm_unavailable(const std::string& reason) {
    detail::throw_forced_cublasdx_unavailable("trmm", reason);
}
// END VERBATIM
}  // namespace

// VERBATIM src/backends/trmm_custom_dispatch.cc:114-207 (ff340fc6)
bool trmm_route_prefers_vendor() {
    return trmm_pin() == detail::Level3Pin::Vendor;
}

bool trmm_cuda_custom_forced() {
    return trmm_pin() == detail::Level3Pin::Cublasdx;
}

bool trmm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag) {
    // `vendor` keeps meaning the vendor even though the tile kernel is the default:
    // it is the "before" a measurement is taken against.
    using detail::Level3Pin;
    const Level3Pin pin = trmm_pin();
    if (pin == Level3Pin::Cublasdx || pin == Level3Pin::Triangular) return true;
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx)) return false;
    if (trmm_triangular_supported(A, B, C, side)) return true;
    return pin == Level3Pin::Auto && trmm_problem_supported(A, B, C, side, uplo, transA) &&
           trmm_prefer_cuda_custom_heuristic(A, B);
}

Event trmm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       Side side,
                       Uplo uplo,
                       Transpose transA,
                       Diag diag) {
    // uplo/diag are carried into the coverage key: trmm has a prior incident where
    // the tempting fix was the wrong-answer one, and a row that cannot tell uplo
    // apart cannot catch that coming back.
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::trmm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, diag, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = trmm_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            throw_forced_trmm_unavailable("the active queue is not a GPU queue");
        }
        if (pin == Level3Pin::Triangular) {
            throw std::invalid_argument("trmm: BATCHLAS_TRMM_ROUTE=triangular needs a GPU queue");
        }
        rec("vendor", false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
    // The tile kernel is the only route that respects the triangle rather than
    // expanding it, so it is what the automatic choice takes wherever it fits.
    if (!forced && trmm_triangular_supported(A, B, C, side)) {
        rec("triangular", true);
        return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (pin == Level3Pin::Triangular) {
        throw std::invalid_argument("trmm: BATCHLAS_TRMM_ROUTE=triangular serves only left-side "
                                    "problems with a homogeneous batch");
    }
    if (!trmm_problem_supported(A, B, C, side, uplo, transA)) {
        if (forced) {
            throw_forced_trmm_unavailable("only left/lower/notrans float problems with matching dense batches are currently supported");
        }
        rec("vendor", false);
        return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    if (pin == Level3Pin::Auto || forced) {
        auto fused = detail::trmm_fused_try(ctx, A, B, C, alpha, side, uplo, transA, diag);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        if (forced) {
            if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
                throw_forced_trmm_unavailable("no compatible fused kernel is available in this build for the requested problem");
            }
            throw_forced_trmm_unavailable("the current device or matrix layout does not satisfy the fused kernel requirements");
        }
    }

    rec("vendor", true);
    return detail::trmm_vendor_fallback(ctx, A, B, C, alpha, side, uplo, transA, diag);
}
// END VERBATIM

// VERBATIM src/backends/cublas.cc:712-751 (ff340fc6)
    template <Backend Back, typename T>
    Event trmm_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      Side side,
                      Uplo uplo,
                      Transpose transA,
                      Diag diag) {
        if constexpr (Back == Backend::CUDA) {
            if (trmm_cuda_custom_forced()) {
                if constexpr (std::is_same_v<T, float>) {
                    return trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
                } else {
                    throw batchlas::unsupported("BATCHLAS_TRMM_ROUTE=cublasdx only supports float");
                }
            }
                // WP1 S6: the float custom-route gate moved to the facade
                // (src/ops/level3/level3.cc). It has to run BEFORE
                // the vendor-available test, and this TU is compiled only when
                // cuBLAS exists -- so leaving it here made the tile kernels
                // linkable everywhere but callable nowhere.
            //
            // The NON-float tile route below stays, and is reachable only from
            // here -- see the syrk note and WP1 S7.
            if constexpr (!std::is_same_v<T, float>) {
                // The tile kernel is type-generic; only its routing was ever
                // float. The alternative for double and complex is the same
                // expansion-plus-GEMM as for float, which is strictly more work
                // than the GEMM it wraps, so there is nothing to weigh here.
                if (detail::is_gpu_queue(ctx) && !trmm_route_prefers_vendor() &&
                    detail::trmm_tiles_supported(A, B, C, side)) {
                    return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
                }
            }
        }

        return trmm_vendor_impl<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
// END VERBATIM
}  // namespace backend

// VERBATIM src/ops/level3/level3.cc:178-206 (ff340fc6)
template <Backend Back, typename T>
Event trmm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::trmm_use_cuda_custom(ctx, A, B, C, side, uplo, transA, diag)) {
            return backend::trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            Op::trmm, "vendor",
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown, {uplo, side, diag, transA});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::trmm, Back, select::kLevel3Library<Back>);
    } else {
        return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
}
// END VERBATIM

}  // namespace batchlas

using namespace batchlas;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// The old public trmm at one homogeneous cell on a GPU queue: its route, or "throw".
template <class T>
std::string old_auto(char side_c, int order, int q, int batch) {
    const Side side = side_c == 'L' ? Side::Left : Side::Right;
    const int m = side == Side::Left ? order : q, n = side == Side::Left ? q : order;
    Queue ctx;
    const MV<T> A{order, order, batch}, B{m, n, batch}, C{m, n, batch};
    try {
        return trmm<Backend::CUDA, T>(ctx, A, B, C, T(1), side, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit).route;
    } catch (const NoRoute&) {
        return "throw";
    }
}

template <class T>
std::string ranked(char side_c, int order, int q, int batch) {
    const Side side = side_c == 'L' ? Side::Left : Side::Right;
    const int m = side == Side::Left ? order : q, n = side == Side::Left ? q : order;
    const MV<T> A{order, order, batch}, B{m, n, batch}, C{m, n, batch};
    const std::string first = old_auto<T>(side_c, order, q, batch);
    // Structurally runnable (the new can_run, capacities unlimited): triangular Left only.
    std::vector<std::string> runnable;
    if (backend::detail::trmm_tiles_supported(A, B, C, side)) runnable.push_back("triangular");
    runnable.push_back("expand");
    std::string out = first;
    for (const auto& c : runnable)
        if (c != first) out += "|" + c;
    if (first != "vendor") out += "|vendor";
    return out;
}

template <class T>
std::string dispatch(const std::string& dt, char s, int o, int q, int b, bool rank) {
    return rank ? ranked<T>(s, o, q, b) : old_auto<T>(s, o, q, b);
}
std::string by_dtype(const std::string& dt, char s, int o, int q, int b, bool rank) {
    if (dt == "float") return dispatch<float>(dt, s, o, q, b, rank);
    if (dt == "double") return dispatch<double>(dt, s, o, q, b, rank);
    if (dt == "cfloat") return dispatch<std::complex<float>>(dt, s, o, q, b, rank);
    return dispatch<std::complex<double>>(dt, s, o, q, b, rank);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "grid") {
        // src/ops/trmm/choice.hh: grid_order, grid_q, grid_batch.
        const int orders[] = {1, 16, 32, 64, 128, 256, 512, 1024, 2048};
        const int qs[] = {1, 16, 128, 1024, 4096};
        const int batches[] = {1, 128, 1024, 32768};
        std::printf("op,dtype,device,side,order,q,batch,ranked\n");
        for (const char* dev : {"sm_89", "sm_120"})
            for (const char* dt : {"float", "double", "cfloat", "cdouble"})
                for (char s : {'L', 'R'})
                    for (int o : orders)
                        for (int q : qs)
                            for (int b : batches)
                                std::printf("trmm,%s,%s,%c,%d,%d,%d,%s\n", dt, dev, s, o, q, b,
                                            by_dtype(dt, s, o, q, b, true).c_str());
        return 0;
    }
    if (mode == "points" && argc > 2) {
        std::ifstream in(argv[2]);
        std::string dt;
        char s;
        int o, q, b;
        while (in >> dt >> s >> o >> q >> b)
            std::printf("%s %c %d %d %d %s\n", dt.c_str(), s, o, q, b, by_dtype(dt, s, o, q, b, false).c_str());
        return 0;
    }
    std::fprintf(stderr, "usage: %s grid | points <file>\n", argv[0]);
    return 1;
}
