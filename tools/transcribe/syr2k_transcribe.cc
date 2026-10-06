// syr2k transcriber (temporary; flat-select level-3 wave, docs/design/flat-select-l3/syr2k.md).
// Evaluates ff340fc6's hand-written syr2k routing at every grid cell of src/ops/syr2k/choice.hh
// (grid mode) or at given points (point mode). The three blocks between the BEGIN/END VERBATIM
// markers are byte-for-byte copies of
//   src/backends/syr2k_custom_dispatch.cc:18-167   (pin parse, predicates, gate, dispatcher)
//   src/backends/cublas.cc:613-638                 (syr2k_vendor: the cublasdx-forced hook)
//   src/ops/level3/level3.cc:147-176               (the public entry's gate)
// all at ff340fc6; syr2k_gate.py --fidelity re-extracts them from git and diffs them. Everything
// else is a host stub: views carry only extents, batch and heterogeneity; a kernel or library
// call returns the name of the route it stands for.
// Build: g++ -std=c++20 -O1 -DOLD_VENDOR=1 syr2k_transcribe.cc -o syr2k_transcribe_vp
//        g++ -std=c++20 -O1 -DOLD_VENDOR=0 syr2k_transcribe.cc -o syr2k_transcribe_vf
// Grid:  syr2k_transcribe_vp grid > tuned/transcribed/syr2k.csv
// Point: syr2k_transcribe_v{p,f} points FILE   (lines: dtype n k batch trans[N|T|C] [het 0|1])

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef OLD_VENDOR
#error "build with -DOLD_VENDOR=0 (vendor-free) or -DOLD_VENDOR=1 (cuBLAS present)"
#endif

namespace batchlas {

enum class Backend { CUDA, ROCM, NETLIB };
enum class MatrixFormat { Dense };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Uplo { Upper, Lower };
enum class Side { Left, Right };
enum class Diag { NonUnit, Unit };
enum class Op { syr2k };
template <class T>
concept RealScalar = std::is_floating_point_v<T>;
using Event = std::string;  // the route a terminal stands for
struct unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
struct NoRoute : std::runtime_error { using std::runtime_error::runtime_error; };

template <class T, MatrixFormat F>
struct MatrixView {
    int r = 0, c = 0, b = 1;
    bool het = false;
    int rows() const { return r; }
    int cols() const { return c; }
    int batch_size() const { return b; }
    bool is_heterogeneous() const { return het; }
};
struct Queue { bool gpu = true; };

namespace select {
template <Backend B>
inline constexpr bool level3_vendor_available = B == Backend::CUDA ? bool(OLD_VENDOR) : true;
template <Backend B>
inline constexpr const char* kLevel3Library = "cuBLAS";
template <class T>
[[noreturn]] void throw_no_vendor_route(Op, Backend, const char*) { throw NoRoute("no route"); }
}  // namespace select

namespace backend {
namespace detail {
enum class Level3Pin { Auto, Native, Vendor, Cublasdx, Expand, Triangular, Gram };
inline Level3Pin g_pin = Level3Pin::Auto;  // the transcription is the Auto choice
inline Level3Pin level3_pin(std::string_view, std::initializer_list<Level3Pin>) { return g_pin; }
inline bool is_gpu_queue(const Queue& q) { return q.gpu; }
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };
struct Level3Variant { Uplo uplo; Side side; Diag diag; Transpose transA; };
inline void record_level3_route(Op, const char*, long, long, long, long, int, Level3Variant = {}) {}
struct FusedResult {
    enum class Outcome { Ran, NoKernel, DeviceUnsupported };
    Event event{};
    Outcome outcome = Outcome::NoKernel;
};
template <class... A>
FusedResult syr2k_fused_try(A&&...) { return {}; }  // MathDx is absent on both boxes
[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view, const std::string& why) {
    throw unsupported(why);
}
template <class... A>
Event syr2k_triangular_tiles(A&&...) { return "triangular"; }
template <class... A>
Event syr2k_vendor_fallback(A&&...) {
    if (!OLD_VENDOR) throw NoRoute("no vendor");  // level3_vendor_fallback.cc without cuBLAS
    return "vendor";
}
}  // namespace detail
template <Backend Back, class T, class... A>
Event syr2k_vendor_impl(A&&...) { return "vendor"; }
}  // namespace backend
}  // namespace batchlas

namespace batchlas::backend {
// ---- BEGIN VERBATIM src/backends/syr2k_custom_dispatch.cc:18-167 ----
namespace {

// BATCHLAS_SYR2K_ROUTE: vendor; triangular or native, the tile-masked kernel that
// computes only the requested half of C; cublasdx, the fused MathDx kernel (throws
// when it cannot run).
detail::Level3Pin syr2k_pin() {
    using detail::Level3Pin;
    return detail::level3_pin("syr2k", {Level3Pin::Native, Level3Pin::Vendor, Level3Pin::Triangular,
                                        Level3Pin::Cublasdx});
}

bool syr2k_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                             const MatrixView<float, MatrixFormat::Dense>& B,
                             const MatrixView<float, MatrixFormat::Dense>& C,
                             Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        return false;
    }
    if (C.rows() != C.cols()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }

    const int n = C.rows();
    const int a_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    const int b_n = transA == Transpose::NoTrans ? B.rows() : B.cols();
    const int a_k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int b_k = transA == Transpose::NoTrans ? B.cols() : B.rows();
    return a_n == n && b_n == n && a_k == b_k && n > 0 && a_k > 0;
}

// The tile-masked kernel indexes every operand as base + batch * stride, so a
// batch whose members differ in shape or live at unrelated pointers is out of
// reach.
bool syr2k_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                                const MatrixView<float, MatrixFormat::Dense>& B,
                                const MatrixView<float, MatrixFormat::Dense>& C) {
    return !A.is_heterogeneous() && !B.is_heterogeneous() && !C.is_heterogeneous();
}

// Where the fused kernel beats the vendor. The vendor route is a host loop over
// cublasSsyr2k, one launch per batch member, against one launch for the whole
// batch here, so the two are only ever close at a batch of one and the vendor
// pays double from two members up.
//
// Measured on RTX 4090 / sm_89 in float over n in 8..3072 x k in 4..2048 x
// batch in 1..1024. From batch 2 the kernel won every shape in the grid: 1.06x
// at n = 3072, 1.12x at n = 1024, 1.3-1.4x through the middle, and up to 226x
// where n is small enough that the whole cost is the launch. Neither n nor k
// nor the tile count enters, because none of them changes which side of that
// per-launch difference a shape falls on.
//
// A batch of one does not sort by anything: the vendor wins by 1.18-1.60x below
// n = 1280 and again by 1.16x at n = 3072, the kernel wins by 1.02-1.71x
// between, and by 4-10x the vendor wins on a deep k with a small n, where the
// kernel has a single block and cuBLAS splits the reduction. There is no
// threshold in n to be had, so the batch of one is left with the vendor.
bool syr2k_prefer_triangular_tiles(const MatrixView<float, MatrixFormat::Dense>& A) {
    return A.batch_size() >= 2;
}

} // namespace

bool syr2k_cuda_custom_forced() {
    return syr2k_pin() == detail::Level3Pin::Cublasdx;
}

bool syr2k_use_cuda_custom(const Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           Uplo,
                           Transpose transA) {
    using detail::Level3Pin;
    const Level3Pin pin = syr2k_pin();
    if (pin != Level3Pin::Auto && pin != Level3Pin::Vendor) {
        return true;
    }
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) ||
        !syr2k_problem_supported(A, B, C, transA) || !syr2k_triangular_supported(A, B, C)) {
        return false;
    }
    // The tile-masked kernel is the only custom route that respects the
    // triangle, so it is the only one the automatic choice may leave the vendor
    // for, and its own threshold is the whole decision.
    return syr2k_prefer_triangular_tiles(A);
}

Event syr2k_cuda_custom(Queue& ctx,
                        const MatrixView<float, MatrixFormat::Dense>& A,
                        const MatrixView<float, MatrixFormat::Dense>& B,
                        const MatrixView<float, MatrixFormat::Dense>& C,
                        float alpha,
                        float beta,
                        Uplo uplo,
                        Transpose transA) {
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::syr2k, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = syr2k_pin();
    const bool forced = pin == Level3Pin::Cublasdx;
    if (!detail::is_gpu_queue(ctx)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("syr2k", "the active queue is not a GPU queue");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }
    if (!syr2k_problem_supported(A, B, C, transA)) {
        if (forced) {
            detail::throw_forced_cublasdx_unavailable("syr2k", "the problem shape or transpose mode is unsupported");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    if (!forced) {
        if (syr2k_triangular_supported(A, B, C)) {
            rec("triangular", true);
            return detail::syr2k_triangular_tiles(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        if (pin == Level3Pin::Triangular) {
            throw std::invalid_argument("syr2k: BATCHLAS_SYR2K_ROUTE=triangular cannot run a heterogeneous batch");
        }
        rec("vendor", false);
        return detail::syr2k_vendor_fallback(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    // There is no uplo-respecting fallback for a fused kernel that did not run, so a
    // cublasdx pin it cannot serve throws rather than overwrite the caller's triangle.
    auto fused = detail::syr2k_fused_try(ctx, A, B, C, alpha, beta, uplo, transA);
    if (fused.outcome == detail::FusedResult::Outcome::Ran) {
        rec("cublasdx", true);
        return std::move(fused.event);
    }
    if (fused.outcome == detail::FusedResult::Outcome::NoKernel) {
        detail::throw_forced_cublasdx_unavailable(
            "syr2k", "no compatible fused kernel is available in this build for the requested problem");
    }
    detail::throw_forced_cublasdx_unavailable(
        "syr2k", "the fused kernel exists but this device refused to launch it");
}
// ---- END VERBATIM ----
}  // namespace batchlas::backend

namespace batchlas::backend {
// ---- BEGIN VERBATIM src/backends/cublas.cc:613-638 ----
    template <Backend Back, RealScalar T>
    Event syr2k_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha,
                       T beta,
                       Uplo uplo,
                       Transpose transA) {
        if constexpr (Back == Backend::CUDA) {
            if (syr2k_cuda_custom_forced()) {
                if constexpr (std::is_same_v<T, float>) {
                    return syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
                } else {
                    throw batchlas::unsupported("BATCHLAS_SYR2K_ROUTE=cublasdx only supports float");
                }
            }
                // WP1 S6: the float custom-route gate moved to the facade
                // (src/ops/level3/level3.cc). It has to run BEFORE
                // the vendor-available test, and this TU is compiled only when
                // cuBLAS exists -- so leaving it here made the tile kernels
                // linkable everywhere but callable nowhere.
        }

        return syr2k_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
// ---- END VERBATIM ----
}  // namespace batchlas::backend

namespace batchlas {
// ---- BEGIN VERBATIM src/ops/level3/level3.cc:147-176 ----
template <Backend Back, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syr2k_use_cuda_custom(ctx, A, B, C, uplo, transA)) {
            return backend::syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            Op::syr2k, "vendor",
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::syr2k, Back, select::kLevel3Library<Back>);
    } else {
        return backend::syr2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}
// ---- END VERBATIM ----
}  // namespace batchlas

namespace {

using namespace batchlas;

// src/ops/syr2k/choice.hh's grid (tuned_tables_tests checks the table rows equal it).
const int kGridN[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
const int kGridK[] = {1, 8, 64, 512, 4096};
const int kGridBatch[] = {1, 2, 3, 128, 1024, 8192, 32768};

// The old public syr2k<CUDA, T> on a GPU queue, Auto: the route name, or "throw".
template <class T>
std::string old_route(int n, int k, int batch, Transpose t, bool het) {
    const bool nt = t == Transpose::NoTrans;
    MatrixView<T, MatrixFormat::Dense> A{nt ? n : k, nt ? k : n, batch, het};
    MatrixView<T, MatrixFormat::Dense> B{nt ? n : k, nt ? k : n, batch, het};
    MatrixView<T, MatrixFormat::Dense> C{n, n, batch, het};
    Queue q{true};
    try {
        return syr2k<Backend::CUDA, T>(q, A, B, C, T(1), T(0), Uplo::Lower, t);
    } catch (const std::exception&) {
        return "throw";
    }
}

// The transcribed row: the old Auto choice, then every other candidate of the dtype that can
// run the (homogeneous, GPU, NoTrans) cell, in candidate order with vendor last.
std::string ranked(const std::string& dtype, const std::string& first) {
    if (dtype == "double") return "vendor";
    return first == "triangular" ? "triangular|vendor" : "vendor|triangular";
}

int grid() {
    std::printf("op,dtype,device,n,k,batch,ranked\n");
    for (const char* dev : {"sm_89", "sm_120"})
        for (const std::string dtype : {"float", "double"})
            for (int n : kGridN)
                for (int k : kGridK)
                    for (int b : kGridBatch) {
                        const std::string r = dtype == "float" ? old_route<float>(n, k, b, Transpose::NoTrans, false)
                                                               : old_route<double>(n, k, b, Transpose::NoTrans, false);
                        if (r == "throw") {
                            std::fprintf(stderr, "old router threw at %s n=%d k=%d batch=%d\n", dtype.c_str(), n, k, b);
                            return 1;
                        }
                        std::printf("syr2k,%s,%s,%d,%d,%d,%s\n", dtype.c_str(), dev, n, k, b, ranked(dtype, r).c_str());
                    }
    return 0;
}

int points(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string dtype, ts;
        int n = 0, k = 0, b = 0, het = 0;
        if (!(ss >> dtype >> n >> k >> b >> ts)) continue;
        ss >> het;
        const Transpose t = ts == "N" ? Transpose::NoTrans : ts == "T" ? Transpose::Trans : Transpose::ConjTrans;
        const std::string r = dtype == "float" ? old_route<float>(n, k, b, t, het != 0)
                                               : old_route<double>(n, k, b, t, het != 0);
        std::printf("%s %d %d %d %s %d %s\n", dtype.c_str(), n, k, b, ts.c_str(), het, r.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "grid") return grid();
    if (argc >= 3 && std::string(argv[1]) == "points") return points(argv[2]);
    std::fprintf(stderr, "usage: %s grid | points FILE\n", argv[0]);
    return 2;
}
