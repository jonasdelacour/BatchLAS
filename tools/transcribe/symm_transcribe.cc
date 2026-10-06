// symm transcriber (temporary; docs/design/flat-select-l3/symm.md). Evaluates the old
// hand-written symm router of ff340fc6 -- verbatim copies of its predicate bodies, checked
// byte for byte by symm_gate.py --fidelity -- over stub types, at every grid cell
// (grid mode) or at given points (point mode).
//
// Build (host only, no SYCL):  g++ -std=c++20 -O2 -o symm_transcribe symm_transcribe.cc
// Grid:   ./symm_transcribe grid tuned/transcribed/symm.csv
// Points: ./symm_transcribe points <in: dtype m n batch side> <out>
//
// Copied verbatim (ff340fc6):
//   src/backends/symm_custom_dispatch.cc:36-72    symm_problem_supported, symm_prefer_cuda_custom_heuristic
//   src/backends/symm_custom_dispatch.cc:137-150  symm_use_cuda_custom
//   src/backends/symm_custom_dispatch.cc:152-191  symm_cuda_custom
//   src/backends/triangular_expand.hh:45-63       kExpandMinBatch, kExpandMinDim, expansion_preferred
//   src/ops/level3/level3.cc:47-64                the public symm's float/CUDA gate and vendor tail
// Stubs model the build both boxes have: no MathDx (symm_fused_try -> NoKernel, the
// level3_fused_absent.cc / unavailable-header outcome), BATCHLAS_SYMM_ROUTE and
// BATCHLAS_EXPAND_ROUTE unset, a GPU queue. Capacities are unlimited (the old rule had none).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace batchlas {

enum class MatrixFormat { Dense };
enum class Side { Left, Right };
enum class Uplo { Lower, Upper };
enum class Diag { NonUnit, Unit };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Op { symm };

// A Backend value carries the device family and whether the vendor library is linked, so the
// verbatim `Back == Backend::CUDA` and `select::level3_vendor_available<Back>` both work.
struct Backend {
    int family;
    bool vendor;
    constexpr bool operator==(const Backend& o) const { return family == o.family; }
    static const Backend CUDA;
};
inline constexpr Backend Backend::CUDA{0, true};
inline constexpr Backend kCudaVendor{0, true};
inline constexpr Backend kCudaVendorFree{0, false};

using Event = std::string;  // the route a call ends in

struct Queue {
    bool gpu = true;
};

template <class T, MatrixFormat F>
struct MatrixView {
    int r = 0, c = 0, b = 0;
    int rows() const { return r; }
    int cols() const { return c; }
    int batch_size() const { return b; }
};

struct NoRoute {};

struct NullSetting {
    const char* get() const { return nullptr; }
};
struct Selection {
    NullSetting expand_route;
};
struct Settings {
    Selection selection;
};
inline const Settings& settings() {
    static Settings s;
    return s;
}

namespace select {
template <Backend B>
inline constexpr bool level3_vendor_available = B.vendor;
template <Backend B>
inline constexpr const char* kLevel3Library = "cuBLAS";
template <class T>
[[noreturn]] void throw_no_vendor_route(Op, Backend, const char*) {
    throw NoRoute{};
}
}  // namespace select

namespace backend {
namespace detail {

enum class Level3Pin { Auto, Native, Vendor, Cublasdx, Expand, Triangular, Gram };
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };
struct Level3Variant {
    Uplo uplo;
    Side side;
    Diag diag;
    Transpose transA;
};
inline void record_level3_route(Op, const char*, long, long, long, long, int, Level3Variant = {}) {}
inline bool is_gpu_queue(const Queue& q) { return q.gpu; }
[[noreturn]] inline void throw_forced_cublasdx_unavailable(const char*, const char*) {
    throw std::logic_error("unreachable: no cublasdx pin in the transcription");
}

struct FusedResult {
    enum class Outcome { Ran, NoKernel, DeviceUnsupported };
    Event event{};
    Outcome outcome = Outcome::NoKernel;
};
template <class... A>
FusedResult symm_fused_try(A&&...) {
    return FusedResult{Event{}, FusedResult::Outcome::NoKernel};
}
template <class... A>
Event symm_vendor_fallback(A&&...) {
    return "vendor";
}

// --- src/backends/triangular_expand.hh:45-63 ---
constexpr int kExpandMinBatch = 4;
constexpr int kExpandMinDim = 256;

// BATCHLAS_EXPAND_ROUTE pins the choice to "expand" or "loop", so a test can
// reach whichever route the shape would not have picked. An expansion still has
// to fit before it can be built, so this only ever narrows expansion_fits.
inline bool expansion_preferred(int max_dim, int batch) {
    // Same Settings field as expansion_budget.hh's expansion_route_pin(), so the
    // two independent parsers can no longer be handed different strings.
    if (const char* route = batchlas::settings().selection.expand_route.get()) {
        if (std::string_view(route) == "expand") {
            return true;
        }
        if (std::string_view(route) == "loop") {
            return false;
        }
    }
    return batch >= kExpandMinBatch || max_dim >= kExpandMinDim;
}
// --- end ---

}  // namespace detail

template <Backend Back, class T, class... A>
Event symm_vendor(A&&...) {
    return "vendor";
}

namespace {

detail::Level3Pin symm_pin() { return detail::Level3Pin::Auto; }

template <class... A>
Event symm_expand_gemm(A&&...) {
    return "expand";
}

// --- src/backends/symm_custom_dispatch.cc:36-72 ---
bool symm_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Side side) {
    if (A.rows() != A.cols()) {
        return false;
    }

    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    const int m = C.rows();
    const int n = C.cols();
    const int expected_a = side == Side::Left ? B.rows() : B.cols();
    return A.rows() == expected_a && B.rows() == m && B.cols() == n && m > 0 && n > 0;
}

bool symm_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& B,
                                       const MatrixView<float, MatrixFormat::Dense>& C,
                                       Side side) {
    const int m = C.rows();
    const int n = C.cols();
    const int k = A.rows();
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    const bool squareish = min_dim * 2 >= max_dim;
    const int shared_dim = side == Side::Left ? B.rows() : B.cols();
    if (!squareish || shared_dim != k) {
        return false;
    }

    // Skewed shapes are excluded above because the expansion always costs a
    // full k x k pass, which stops paying for itself once k dwarfs m and n.
    return detail::expansion_preferred(max_dim, A.batch_size());
}
// --- end ---

}  // namespace

// --- src/backends/symm_custom_dispatch.cc:137-150 ---
bool symm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo) {
    using detail::Level3Pin;
    const Level3Pin pin = symm_pin();
    if (pin == Level3Pin::Cublasdx) return true;
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) || !symm_problem_supported(A, B, C, side)) {
        return false;
    }
    return pin != Level3Pin::Auto || symm_prefer_cuda_custom_heuristic(A, B, C, side);
}
// --- end ---

// --- src/backends/symm_custom_dispatch.cc:152-191 ---
Event symm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Side side,
                       Uplo uplo) {
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::symm, taken,
                                    C.rows(), C.cols(), A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    };

    using detail::Level3Pin;
    const Level3Pin pin = symm_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!symm_problem_supported(A, B, C, side)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("symm", "the problem shape is unsupported");
        }
        rec("vendor", false);
        return detail::symm_vendor_fallback(ctx, A, B, C, alpha, beta, side, uplo);
    }

    if (pin == Level3Pin::Auto || forced) {
        auto fused = detail::symm_fused_try(ctx, A, B, C, alpha, beta, side, uplo);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("symm", "no fused kernel ran for this problem");
        }
    }

    rec("expand", true);
    return symm_expand_gemm(ctx, A, B, C, alpha, beta, side, uplo);
}
// --- end ---

}  // namespace backend

template <Backend Back, class T>
Event old_symm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
               const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side, Uplo uplo) {
    // --- src/ops/level3/level3.cc:47-64 ---
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::symm_use_cuda_custom(ctx, A, B, C, side, uplo)) {
            return backend::symm_cuda_custom(ctx, A, B, C, alpha, beta, side, uplo);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            Op::symm, "vendor",
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown,
            {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::symm, Back, select::kLevel3Library<Back>);
    } else {
        return backend::symm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }
    // --- end ---
}

}  // namespace batchlas

using namespace batchlas;

// One old call at C = m x n, batch b, on the given side: A is the order the side asks for.
template <Backend Back, class T>
std::string run_old(int m, int n, int b, Side side) {
    const int k = side == Side::Left ? m : n;
    Queue q;
    const MatrixView<T, MatrixFormat::Dense> A{k, k, b}, Bm{m, n, b}, C{m, n, b};
    try {
        return old_symm<Back, T>(q, A, Bm, C, T(1), T(0), side, Uplo::Lower);
    } catch (const NoRoute&) {
        return "throw";
    }
}

std::string form_of(long a, long b) {
    return 2 * std::min(a, b) >= std::max(a, b) ? "sq" : (a > 2 * b ? "tall" : "wide");
}

// The vendor-present Auto choice, required to be side-invariant.
template <class T>
std::string old_auto(int m, int n, int b) {
    const std::string l = run_old<kCudaVendor, T>(m, n, b, Side::Left);
    const std::string r = run_old<kCudaVendor, T>(m, n, b, Side::Right);
    if (l != r) {
        std::fprintf(stderr, "side-variant old choice at m=%d n=%d b=%d: %s vs %s\n", m, n, b, l.c_str(), r.c_str());
        std::exit(2);
    }
    return l;
}

// The cell's representative: itself, or R_f when (m, n) contradicts the form.
std::pair<int, int> representative(const std::string& form, int m, int n) {
    if (form_of(m, n) == form) return {m, n};
    const int M = std::max(m, n), h = std::max(1, (M - 1) / 2);
    if (form == "sq") return {M, M};
    if (form == "tall") return {M, h};
    return {h, M};
}

// Old Auto first, then every other candidate structurally runnable at the cell (both are, on a
// homogeneous GPU call at unlimited capacity), vendor last unless it is the Auto choice.
std::string ranked(const std::string& first) { return first == "expand" ? "expand|vendor" : "vendor|expand"; }

const int kMN[] = {1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096};
const int kBatch[] = {1, 2, 3, 4, 8, 128, 1024, 8192, 32768};
const char* kForms[] = {"sq", "tall", "wide"};

int grid(const char* out_path) {
    std::ofstream out(out_path);
    out << "op,dtype,device,form,m,n,batch,ranked\n";
    long rows = 0;
    for (const char* dtype : {"float", "double"})
        for (const char* dev : {"sm_89", "sm_120"})
            for (const char* form : kForms)
                for (int m : kMN)
                    for (int n : kMN)
                        for (int b : kBatch) {
                            const auto [rm, rn] = representative(form, m, n);
                            const std::string first = std::string(dtype) == "float" ? old_auto<float>(rm, rn, b)
                                                                                     : old_auto<double>(rm, rn, b);
                            out << "symm," << dtype << ',' << dev << ',' << form << ',' << m << ',' << n << ','
                                << b << ',' << ranked(first) << '\n';
                            ++rows;
                        }
    std::fprintf(stderr, "wrote %ld rows to %s\n", rows, out_path);
    return 0;
}

int points(const char* in_path, const char* out_path) {
    std::ifstream in(in_path);
    std::ofstream out(out_path);
    std::string dtype, side;
    int m, n, b;
    while (in >> dtype >> m >> n >> b >> side) {
        const Side s = side == "L" ? Side::Left : Side::Right;
        std::string vp, vf;
        if (dtype == "float") {
            vp = run_old<kCudaVendor, float>(m, n, b, s);
            vf = run_old<kCudaVendorFree, float>(m, n, b, s);
        } else {
            vp = run_old<kCudaVendor, double>(m, n, b, s);
            vf = run_old<kCudaVendorFree, double>(m, n, b, s);
        }
        out << dtype << ' ' << m << ' ' << n << ' ' << b << ' ' << side << ' ' << vp << ' ' << vf << '\n';
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "grid") return grid(argv[2]);
    if (argc == 4 && std::string(argv[1]) == "points") return points(argv[2], argv[3]);
    std::fprintf(stderr, "usage: %s grid <out.csv> | points <in> <out>\n", argv[0]);
    return 1;
}
