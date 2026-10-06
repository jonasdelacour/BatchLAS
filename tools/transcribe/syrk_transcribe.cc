// TEMPORARY transcriber (deleted at integration, like phase 5's tools/transcribe/): syrk's
// hand-written Auto rule at ff340fc6, evaluated at every cell of src/ops/syrk/choice.hh's grid.
// Every block between VERBATIM markers is a byte-for-byte copy of the ff340fc6 source named on
// the marker (tools/transcribe/syrk_gate.py --fidelity checks it); everything else is a shim:
// Event is the name of the route the old code returned, the queue is a GPU, no pin is set.
//
// Build (repo root): g++ -std=c++20 -O2 -I. -o syrk_transcribe tools/transcribe/syrk_transcribe.cc
// Grid:  ./syrk_transcribe grid > tuned/transcribed/syrk.csv
// Point: ./syrk_transcribe points FILE   (lines "dtype n k batch") -> "vendor_present vendor_free"
#ifndef SYRK_FACADE_PASS

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace batchlas {

enum class Backend { CUDA, ROCM, NETLIB };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Uplo { Lower, Upper };
enum class Side { Left, Right };
enum class Diag { NonUnit, Unit };
enum class MatrixFormat { Dense };
enum class Op { syrk };
using Event = std::string;
struct Queue {};
struct NoRoute {};
template <class T>
concept RealScalar = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <class T, MatrixFormat F>
struct MatrixView {
    int r = 0, c = 0, b = 1;
    int rows() const { return r; }
    int cols() const { return c; }
    int batch_size() const { return b; }
    bool is_heterogeneous() const { return false; }
};

namespace backend {
namespace detail {

// VERBATIM src/backends/triangular_tiles.hh 126 136
inline constexpr int kTriangularTile = 128;
inline constexpr int kTriangularTileK = 8;

inline int triangular_tiles_per_side(int n) {
    return (n + kTriangularTile - 1) / kTriangularTile;
}

inline int triangular_tile_count(int n) {
    const int t = triangular_tiles_per_side(n);
    return t * (t + 1) / 2;
}
// END VERBATIM

// VERBATIM src/backends/syrk_gram_tiles.hh 65 65
inline constexpr int kGramMaxTile = 128;
// END VERBATIM

// VERBATIM src/backends/syrk_gram_tiles.hh 314 338
template <typename T>
bool syrk_gram_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Transpose transA,
                         bool conjugated) {
    if (C.rows() != C.cols() || C.rows() <= 0 || C.rows() > kGramMaxTile) {
        return false;
    }
    if (A.batch_size() != C.batch_size()) {
        return false;
    }
    if (A.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    // SYRK spells A*A^T and must not be handed a ConjTrans; HERK spells A*A^H
    // and must not be handed a plain Trans, which would be complex-symmetric.
    if (conjugated ? (transA != Transpose::NoTrans && transA != Transpose::ConjTrans)
                   : (transA == Transpose::ConjTrans)) {
        return false;
    }
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    return expected_n == n && k > 0;
}
// END VERBATIM

inline int ceil_div(int v, int d) { return (v + d - 1) / d; }
inline bool is_gpu_queue(const Queue&) { return true; }
enum class Level3Pin { Auto, Native, Vendor, Cublasdx, Expand, Triangular, Gram };
inline const char* level3_pin_word(Level3Pin) { return "auto"; }
struct Level3Variant {
    Uplo uplo;
    Side side;
    Diag diag;
    Transpose transA;
};
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };
inline void record_level3_route(Op, const char*, long, long, long, long, int, Level3Variant = {}) {}
[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view, const std::string& r) {
    throw std::runtime_error(r);
}
struct FusedResult {
    enum class Outcome { Ran, Declined };
    Outcome outcome = Outcome::Declined;
    Event event;
};
template <class... A>
FusedResult syrk_fused_try(A&&...) { return {}; }
template <class... A>
Event syrk_vendor_fallback(A&&...) { return "vendor"; }
template <class T = void, bool Conj = false, class... A>
Event syrk_gram_tiles(A&&...) { return "gram"; }
template <class... A>
Event syrk_triangular_tiles(A&&...) { return "triangular"; }

}  // namespace detail

template <Backend B, class T, class... A>
Event syrk_vendor_impl(A&&...) { return "vendor"; }
inline bool syrk_route_prefers_vendor() { return false; }

namespace {

// VERBATIM src/backends/syrk_custom_dispatch.cc 22 22
constexpr int kSyrkCublasDxTile = 32;
// END VERBATIM

detail::Level3Pin syrk_pin() { return detail::Level3Pin::Auto; }

// VERBATIM src/backends/syrk_custom_dispatch.cc 34 117
bool syrk_problem_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            Transpose transA) {
    if (transA == Transpose::ConjTrans) {
        return false;
    }
    if (C.rows() != C.cols()) {
        return false;
    }
    if (A.batch_size() != C.batch_size()) {
        return false;
    }

    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    return expected_n == n && n > 0 && k > 0;
}

// The tile-masked kernel indexes both operands as base + batch * stride, so a
// batch whose members differ in shape or live at unrelated pointers is out of
// reach.
bool syrk_triangular_supported(const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& C) {
    return !A.is_heterogeneous() && !C.is_heterogeneous();
}

// Where skipping the tiles outside the triangle starts paying for the
// tile-masked kernel's lower per-tile rate. Two conditions, both measured on
// RTX 4090 / sm_89 in float over n in 64..2048 x batch in 1..512, against the
// full n x n batched GEMM this replaces:
//
//   - n has to be past 256. A tile grid narrower than three 128-wide tiles a
//     side is more than half diagonal, and a diagonal tile is computed whole
//     and then masked, so at n = 256 only one tile in four is saved. That does
//     not cover the gap to cuBLAS per tile: n = 256 measured anywhere between
//     0.84x and 1.22x depending on where its grid happened to fall against a
//     wave boundary, which is no win at all. From n = 384 up every saturated
//     shape won, and the win grows with n as the diagonal thins out -- 1.45x
//     at n = 512 batch 512, 1.63x at n = 1024 batch 64, 1.71x at n = 2048
//     batch 16.
//   - the grid has to fill the device. The 128 SMs hold two of these
//     256-thread blocks apiece, and below ~160 blocks the triangular route
//     lost (1.14x slower at 144 blocks, 1.25x at 136) where from 168 up it won
//     (0.71x).
//
// k does not enter: it only deepens each block's reduction, which moves both
// routes together.
bool syrk_prefer_triangular_tiles(const MatrixView<float, MatrixFormat::Dense>& A,
                                  const MatrixView<float, MatrixFormat::Dense>& C,
                                  Transpose transA) {
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    if (detail::triangular_tiles_per_side(n) < 3 || k < detail::kTriangularTileK) {
        return false;
    }
    return static_cast<long long>(A.batch_size()) * detail::triangular_tile_count(n) >= 160;
}

// The single-tile kernel's whole premise is that the tile is sized to n, so it
// serves exactly the range the triangular grid cannot: n no wider than one
// tile. Inside that range it is not a close call and there is no threshold to
// tune -- the alternative is a host loop over cublasSsyrk, which at large batch
// is one to two orders of magnitude off anything batched.
bool syrk_prefer_gram_tiles(const MatrixView<float, MatrixFormat::Dense>& C) {
    return C.rows() <= detail::kGramMaxTile;
}

bool syrk_prefer_cuda_custom_heuristic(const MatrixView<float, MatrixFormat::Dense>& A,
                                       const MatrixView<float, MatrixFormat::Dense>& C,
                                       Transpose transA) {
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int max_dim = std::max(n, k);
    const int min_dim = std::min(n, k);
    if (n < 16) {
        return false;
    }

    const int output_tile_rows = detail::ceil_div(n, kSyrkCublasDxTile);
    const int reduction_tiles = detail::ceil_div(k, kSyrkCublasDxTile);
    const int tiled_work = A.batch_size() * output_tile_rows * output_tile_rows * reduction_tiles;
    return min_dim * 2 >= max_dim && tiled_work >= 8;
}
// END VERBATIM

}  // namespace

// VERBATIM src/backends/syrk_custom_dispatch.cc 129 209
bool syrk_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Uplo,
                          Transpose transA) {
    using detail::Level3Pin;
    const Level3Pin pin = syrk_pin();
    if (pin != Level3Pin::Auto && pin != Level3Pin::Vendor) {
        return true;
    }
    if (pin == Level3Pin::Vendor || !detail::is_gpu_queue(ctx) ||
        !syrk_problem_supported(A, C, transA) || !syrk_triangular_supported(A, C)) {
        return false;
    }
    // The two tile-masked kernels are the only custom routes that respect the
    // triangle, so they are the only ones the automatic choice may leave the
    // vendor for. Between them they cover the range: `gram` below one tile,
    // `triangular` from three tiles a side up. Its own threshold says where it
    // beats the full n x n GEMM; below that the question is instead whether it
    // beats a host loop over cublasSsyrk, which the cuBLASDx heuristic already
    // answers -- one launch per batch member costs about 9 us, so anything with
    // a batch at all is better off here even where the tile grid is half
    // diagonal.
    return syrk_prefer_gram_tiles(C) ||
        syrk_prefer_triangular_tiles(A, C, transA) ||
        syrk_prefer_cuda_custom_heuristic(A, C, transA);
}

Event syrk_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Uplo uplo,
                       Transpose transA) {
    const auto rec = [&](const char* taken, bool native_supported) {
        detail::record_level3_route(Op::syrk, taken,
                                    C.rows(), C.cols(),
                                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                                    A.batch_size(), native_supported,
                                    {uplo, Side::Left, Diag::NonUnit, transA});
    };

    using detail::Level3Pin;
    const Level3Pin pin = syrk_pin();
    if (!syrk_problem_supported(A, C, transA)) {
        if (pin == Level3Pin::Cublasdx) {
            detail::throw_forced_cublasdx_unavailable("syrk", "the problem shape or transpose mode is unsupported");
        }
        rec("vendor", false);
        return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
    }

    if (pin == Level3Pin::Cublasdx) {
        auto fused = detail::syrk_fused_try(ctx, A, C, alpha, beta, uplo, transA);
        if (fused.outcome == detail::FusedResult::Outcome::Ran) {
            rec("cublasdx", true);
            return std::move(fused.event);
        }
        detail::throw_forced_cublasdx_unavailable("syrk", "no fused kernel ran for this problem");
    }

    const bool any_tile = pin == Level3Pin::Auto || pin == Level3Pin::Native;
    if (syrk_triangular_supported(A, C)) {
        // A narrow C is one tile wide, so the triangular grid has nothing to skip and
        // would charge a full 128-wide tile for it. Auto splits the range there.
        if (pin == Level3Pin::Gram || (any_tile && syrk_prefer_gram_tiles(C))) {
            rec("gram", true);
            return detail::syrk_gram_tiles(ctx, A, C, alpha, beta, uplo, transA);
        }
        rec("triangular", true);
        return detail::syrk_triangular_tiles(ctx, A, C, alpha, beta, uplo, transA);
    }
    if (!any_tile) {
        throw std::invalid_argument("syrk: BATCHLAS_SYRK_ROUTE=" + std::string(detail::level3_pin_word(pin)) +
                                    " cannot run a heterogeneous batch");
    }
    // A heterogeneous batch: the tile kernels index base + batch * stride.
    rec("vendor", false);
    return detail::syrk_vendor_fallback(ctx, A, C, alpha, beta, uplo, transA);
}
// END VERBATIM

// VERBATIM src/backends/cublas.cc 544 580
    template <Backend Back, RealScalar T>
    Event syrk_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Uplo uplo,
                      Transpose transA) {
        if constexpr (Back == Backend::CUDA) {
                // WP1 S6: the float custom-route gate moved to the facade
                // (src/ops/level3/level3.cc). It has to run BEFORE
                // the vendor-available test, and this TU is compiled only when
                // cuBLAS exists -- so leaving it here made the tile kernels
                // linkable everywhere but callable nowhere.
            //
            // The NON-float gram route below stays: it is reachable only from
            // here, so double and complex syrk still have no native route in a
            // vendor-free build. That is why WP1 S7 refuses to flip
            // level3_tile_kernels_compiled to a bare `true`.
            if constexpr (!std::is_same_v<T, float>) {
                // Everything that is not float reaches the single-tile Gram
                // kernel only. It is the one route here whose staging and
                // fragment loads are not written around a 128-bit packet, so it
                // is the one that generalises; the 128x128 triangular kernel
                // stays float. Below kGramMaxTile the alternative is
                // syrk_vendor_impl's host loop over one cublasXsyrk per batch
                // member, which at large batch is two orders of magnitude off
                // anything batched, so there is no threshold to tune.
                if (detail::is_gpu_queue(ctx) && !syrk_route_prefers_vendor() &&
                    detail::syrk_gram_supported(A, C, transA, /*conjugated=*/false)) {
                    return detail::syrk_gram_tiles<T, false>(ctx, A, C, alpha, beta, uplo, transA);
                }
            }
        }

        return syrk_vendor_impl<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
// END VERBATIM

}  // namespace backend

namespace vp {
inline constexpr bool kFacadeVendor = true;
#define SYRK_FACADE_PASS 1
#include __FILE__
#undef SYRK_FACADE_PASS
}  // namespace vp
namespace vf {
inline constexpr bool kFacadeVendor = false;
#define SYRK_FACADE_PASS 1
#include __FILE__
#undef SYRK_FACADE_PASS
}  // namespace vf

}  // namespace batchlas

namespace {

using namespace batchlas;

// src/ops/syrk/choice.hh's grid and form rule, spelled again here.
const std::array<int, 30> kN{1,   2,   4,   8,   16,  32,  64,   128,  129,  256,  257,  384,  385,  512,  513,
                             640, 641, 768, 769, 896, 897, 1024, 1025, 1152, 1153, 1536, 1537, 2176, 2177, 4096};
const std::array<int, 6> kK{1, 7, 8, 64, 512, 4096};
const std::array<int, 17> kBatch{1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 26, 27, 128, 1024, 32768};

std::string form_of(long a, long b) {
    return 2 * std::min(a, b) >= std::max(a, b) ? "sq" : (a > 2 * b ? "tall" : "wide");
}

// The form representative R_f: a cell whose (n, k) contradicts its form holds R_f's decision.
int rep_k(const std::string& form, int n) {
    if (form == "sq") return n;
    if (form == "tall") return std::max(1, (n - 1) / 2);
    return 2 * n + 1;
}

template <class T>
std::string old_auto(int n, int k, int batch, bool vendor) {
    Queue q;
    const MatrixView<T, MatrixFormat::Dense> A{n, k, batch}, C{n, n, batch};
    try {
        if (vendor) return vp::syrk<Backend::CUDA, T>(q, A, C, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
        return vf::syrk<Backend::CUDA, T>(q, A, C, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
    } catch (const NoRoute&) {
        return "throw";
    }
}

// Old choice first; then every other candidate that can run a homogeneous GPU cell with
// unlimited capacity, in candidate order; vendor last unless it is the old choice.
template <class T>
std::string ranked(int n, int k, int batch) {
    const std::string first = old_auto<T>(n, k, batch, true);
    const MatrixView<T, MatrixFormat::Dense> A{n, k, batch}, C{n, n, batch};
    std::vector<std::string> out{first};
    const bool gram = backend::detail::syrk_gram_supported(A, C, Transpose::NoTrans, false);
    if (first != "gram" && gram) out.push_back("gram");
    if (std::is_same_v<T, float> && first != "triangular") out.push_back("triangular");
    if (first != "vendor") out.push_back("vendor");
    std::string s;
    for (const auto& c : out) s += (s.empty() ? "" : "|") + c;
    return s;
}

template <class T>
void grid(const char* dtype) {
    for (const char* dev : {"sm_89", "sm_120"})
        for (const std::string form : {"sq", "tall", "wide"})
            for (int n : kN)
                for (int k : kK)
                    for (int b : kBatch) {
                        const int ke = form_of(n, k) == form ? k : rep_k(form, n);
                        std::printf("syrk,%s,%s,%s,%d,%d,%d,%s\n", dtype, dev, form.c_str(), n, k, b,
                                    ranked<T>(n, ke, b).c_str());
                    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "grid") {
        std::printf("op,dtype,device,form,n,k,batch,ranked\n");
        grid<float>("float");
        grid<double>("double");
        return 0;
    }
    if (mode == "points" && argc > 2) {
        std::ifstream in(argv[2]);
        std::string dtype;
        int n, k, b;
        while (in >> dtype >> n >> k >> b) {
            const bool f = dtype == "float";
            std::printf("%s %s\n", (f ? old_auto<float>(n, k, b, true) : old_auto<double>(n, k, b, true)).c_str(),
                        (f ? old_auto<float>(n, k, b, false) : old_auto<double>(n, k, b, false)).c_str());
        }
        return 0;
    }
    std::fprintf(stderr, "usage: %s grid | points FILE\n", argv[0]);
    return 2;
}

#else  // SYRK_FACADE_PASS: the facade, once with a vendor library (vp) and once without (vf).

namespace select {
template <Backend>
inline constexpr bool level3_vendor_available = kFacadeVendor;
template <Backend>
inline constexpr const char* kLevel3Library = "cuBLAS";
template <class T>
[[noreturn]] void throw_no_vendor_route(Op, Backend, const char*) { throw NoRoute{}; }
}  // namespace select

// VERBATIM src/ops/level3/level3.cc 117 145
template <Backend Back, RealScalar T>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syrk_use_cuda_custom(ctx, A, C, uplo, transA)) {
            return backend::syrk_cuda_custom(ctx, A, C, alpha, beta, uplo, transA);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            Op::syrk, "vendor",
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::syrk, Back, select::kLevel3Library<Back>);
    } else {
        return backend::syrk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
}
// END VERBATIM

#endif
