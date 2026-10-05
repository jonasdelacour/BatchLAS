// gemm transcriber (flat-kernel-selection-phase3-plan.md §2 option D, §3 gemm grid). Evaluates the
// OLD gemm decision at every grid cell of src/ops/gemm/choice.hh and prints its preference order
// as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// The old decision had two halves, and both are the real code, not a restatement:
//   1. vendor vs native: RouteTable<Op::gemm,T>::preferred via resolve_route (route_gemm.hh), the
//      predicate cublas.cc's gemm_use_sycl_custom re-route consulted (Auto, vendor present);
//   2. which native kernel: sycl_gemm::select_kernel_variant<T>, exported from libbatchlas_sycl.
// So this links against a BUILT tree that still has both (424a45bc, the sha in the tables):
//   OLD=<that checkout, built>
//   clang++ -fsycl -std=c++20 -I$OLD/include -I$OLD/build/include -I$OLD/src -isystem /opt/include \
//     tools/transcribe/gemm_transcribe.cc -L$OLD/build/src -lbatchlas_sycl -lbatchlas_core -o /tmp/gt
//   CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$OLD/build/src:$LD_LIBRARY_PATH /tmp/gt sm_89 \
//     > tuned/transcribed/gemm.sm_89.csv
// No BATCHLAS_GEMM_* variable may be set: select_kernel_variant honours BATCHLAS_GEMM_SYCL_KERNEL.
//
// Layout: `packed` cells are evaluated on contiguous views with 4 KiB-aligned bases (every
// aligned fast-path predicate then sees only the tile divisibility); `strided` cells on
// ld = rows + 1 and an odd batch stride, which fails every such predicate. A strided call whose
// ld happens to be a multiple of 4 could pass can_use_128x128_fast_path in the old code; the new
// key does not distinguish it (plan §1.3: the leg is derived in the launcher, never a gate).
//
// Edge rows (transcription only, not in the tuner grid): the old preferred() had edges below the
// grid -- batch < 64 and double k < 2 went to the vendor, and float was native only on NN squares
// up to 48. Nearest lookup would carry the grid's native rows across them, so real types also get
// batch {1, 63, 64}, double gets k {1, 2} at every (form, layout, m, n), and float NN gets the
// squares 1, 2, 4, 40, 49, 56 plus one-axis-off neighbours of the small squares.
//
// Ranked list: the old first choice, then the old code's forced-name fallback for that kernel
// (register/wide tiles fell back to Tiled16, SmallBatched to Direct), then the other of
// tiled/direct, then small for a real max(m, n, k) <= 64 (it alone survives batch > 65535);
// vendor is first when the old route was the vendor, else last. Old variants
// never returned by Auto (the four pin-only register variants, the five experimental ones)
// cannot appear.

#include <batchlas/blas/dispatch/route_gemm.hh>
#include <batchlas/blas/matrix.hh>

#include "sycl/gemm_kernels.hh"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using namespace batchlas;
using batchlas::sycl_gemm::KernelVariant;

// src/ops/gemm/choice.hh's grid, spelled again (this TU builds against the OLD tree).
constexpr std::array<int, 14> kSquare{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
constexpr std::array<int, 7> kPanelMn{32, 64, 128, 256, 512, 1024, 2048};
constexpr std::array<int, 6> kPanelK{8, 16, 32, 64, 96, 128};
constexpr std::array<int, 6> kSkinnyMn{64, 128, 256, 512, 1024, 2048};
constexpr std::array<int, 2> kSkinnyK{256, 1024};
constexpr std::array<int, 3> kBatch{128, 2048, 32768};
constexpr std::array<int, 6> kRealBatch{1, 63, 64, 128, 2048, 32768};
constexpr int kPackedPanelMin = 128;
constexpr std::array<int, 6> kSmallSquares{8, 16, 24, 32, 40, 48};
constexpr std::array<int, 11> kOffAxis{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64};
constexpr std::array<int, 6> kEdgeSquares{1, 2, 4, 40, 49, 56};

struct Form {
    char a, b;
};
constexpr std::array<Form, 4> kRealForms{{{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'T', 'T'}}};
constexpr std::array<Form, 3> kRealPanelForms{{{'N', 'N'}, {'N', 'T'}, {'T', 'N'}}};
constexpr std::array<Form, 6> kCplxForms{{{'N', 'N'}, {'N', 'T'}, {'T', 'N'}, {'N', 'C'}, {'C', 'N'}, {'C', 'T'}}};
constexpr std::array<Form, 5> kCplxPanelForms{{{'N', 'N'}, {'N', 'T'}, {'N', 'C'}, {'T', 'N'}, {'C', 'N'}}};

Transpose trans_of(char c) {
    return c == 'N' ? Transpose::NoTrans : c == 'T' ? Transpose::Trans : Transpose::ConjTrans;
}

std::string spelling(KernelVariant v) {
    switch (v) {
        case KernelVariant::Direct: return "direct";
        case KernelVariant::Tiled16: return "tiled";
        case KernelVariant::SmallBatched: return "small";
        case KernelVariant::Tiled32x32Register: return "reg:m=32:n=32:k=8:u=1";
        case KernelVariant::Tiled64x64Register: return "reg:m=64:n=64:k=8:u=1";
        case KernelVariant::Tiled64x64RegisterK16:
        case KernelVariant::Tiled64x64RegisterK16TN:
        case KernelVariant::Tiled64x64RegisterK16NT:
        case KernelVariant::Tiled64x64RegisterK16TT: return "reg:m=64:n=64:k=16:u=1";
        case KernelVariant::Tiled128x32RegisterK16:
        case KernelVariant::Tiled128x32RegisterK16TN:
        case KernelVariant::Tiled128x32RegisterK16NT:
        case KernelVariant::Tiled128x32RegisterK16TT: return "reg:m=128:n=32:k=16:u=1";
        case KernelVariant::Tiled128x32RegisterK32TN:
        case KernelVariant::Tiled128x32RegisterK32NT:
        case KernelVariant::Tiled128x32RegisterK32TT:
        case KernelVariant::Tiled128x32RegisterK32S2U1:
        case KernelVariant::Tiled128x32RegisterK32S2U1Aligned:
        case KernelVariant::Tiled128x32RegisterK32S2U1Generic: return "reg:m=128:n=32:k=32:u=1";
        case KernelVariant::Tiled128x64RegisterK16TN:
        case KernelVariant::Tiled128x64RegisterK16NT:
        case KernelVariant::Tiled128x64RegisterK16TT: return "reg:m=128:n=64:k=16:u=1";
        case KernelVariant::Tiled128x64RegisterK32Large: return "reg:m=128:n=64:k=32:u=4";
        case KernelVariant::Tiled128x64RegisterK32LargeU2: return "reg:m=128:n=64:k=32:u=2";
        case KernelVariant::Tiled128x128RegisterK8: return "reg:m=128:n=128:k=8:u=1";
        case KernelVariant::Tiled32x128RegisterK16:
        case KernelVariant::Tiled32x128RegisterK16TN:
        case KernelVariant::Tiled32x128RegisterK16TT: return "reg:m=32:n=128:k=16:u=1";
        case KernelVariant::Tiled64x64RegisterK16Wide:
        case KernelVariant::Tiled64x64RegisterK16WideCN:
        case KernelVariant::Tiled64x64RegisterK16WideNC: return "wide:m=64:n=64:k=16";
        case KernelVariant::Tiled128x32RegisterK16WideNC: return "wide:m=128:n=32:k=16";
        case KernelVariant::Tiled32x128RegisterK16WideCN: return "wide:m=32:n=128:k=16";
        case KernelVariant::Tiled32x32RegisterK16Wide: return "wide:m=32:n=32:k=16";
        case KernelVariant::Tiled16x16RegisterK16Wide: return "wide:m=16:n=16:k=16";
        default: break;
    }
    std::fprintf(stderr, "gemm_transcribe: Auto returned a deleted variant %d\n", static_cast<int>(v));
    std::exit(2);
}

// Distinct 4 KiB-aligned fake bases: the selector reads addresses, never data.
template <typename T>
MatrixView<T, MatrixFormat::Dense> view(int which, int rows, int cols, int batch, bool packed) {
    T* base = reinterpret_cast<T*>(static_cast<std::uintptr_t>(which + 1) << 40);
    const int ld = packed ? rows : rows + 1;
    const int stride = packed ? ld * cols : ld * cols + 1;
    return MatrixView<T, MatrixFormat::Dense>(base, rows, cols, ld, stride, batch);
}

template <typename T>
bool old_route_native(Transpose ta, Transpose tb, int m, int n, int k, int batch) {
    dispatch::OpShape s;
    s.op = dispatch::Op::gemm;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = k;
    s.batch = batch;
    s.transA = ta;
    s.transB = tb;
    s.is_gpu = true;
    const dispatch::Route r = dispatch::resolve_route_uninstrumented<dispatch::Op::gemm, T>(
        dispatch::Route{dispatch::Origin::Auto, dispatch::Algorithm::Auto}, s, true);
    return dispatch::is_native(r);
}

template <typename T>
std::string ranked(Form f, bool packed, int m, int n, int k, int batch) {
    const Transpose ta = trans_of(f.a), tb = trans_of(f.b);
    const auto A = ta == Transpose::NoTrans ? view<T>(0, m, k, batch, packed) : view<T>(0, k, m, batch, packed);
    const auto B = tb == Transpose::NoTrans ? view<T>(1, k, n, batch, packed) : view<T>(1, n, k, batch, packed);
    const auto C = view<T>(2, m, n, batch, packed);
    const std::string first = spelling(sycl_gemm::select_kernel_variant<T>(A, B, C, ta, tb));
    std::vector<std::string> natives{first};
    if (first == "small") natives.push_back("direct");
    for (const char* s : {"tiled", "direct"})
        if (std::find(natives.begin(), natives.end(), s) == natives.end()) natives.push_back(s);
    // small is the one 1-D launch: it still runs where batch > 65535 refuses every other native.
    constexpr bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
    if (!cplx && std::max({m, n, k}) <= 64 && natives.front() != "small") natives.push_back("small");
    std::string out = old_route_native<T>(ta, tb, m, n, k, batch) ? "" : "vendor";
    for (const auto& s : natives) out += (out.empty() ? "" : "|") + s;
    if (out.rfind("vendor", 0) != 0) out += "|vendor";
    return out;
}

template <typename T, std::size_t NF, std::size_t NP>
std::vector<std::tuple<Form, bool, int, int, int>> cells(const std::array<Form, NF>& forms,
                                                          const std::array<Form, NP>& panel_forms) {
    std::vector<std::tuple<Form, bool, int, int, int>> out;
    std::set<std::tuple<char, char, bool, int, int, int>> seen;
    auto add = [&](Form f, bool packed, int m, int n, int k) {
        if (seen.insert({f.a, f.b, packed, m, n, k}).second) out.push_back({f, packed, m, n, k});
    };
    for (Form f : forms) {
        for (bool packed : {false, true})
            for (int s : kSquare) add(f, packed, s, s, s);
        bool panel = false;
        for (Form p : panel_forms) panel = panel || (p.a == f.a && p.b == f.b);
        if (!panel) continue;
        for (int m : kPanelMn)
            for (int n : kPanelMn)
                for (int k : kPanelK) {
                    add(f, false, m, n, k);
                    if (m >= kPackedPanelMin && n >= kPackedPanelMin) add(f, true, m, n, k);
                }
        for (int mn : kSkinnyMn)
            for (int k : kSkinnyK) {
                add(f, false, mn, 32, k);
                add(f, false, 32, mn, k);
            }
    }
    if constexpr (std::is_same_v<T, double>) {
        const auto grid = out;
        for (const auto& [f, packed, m, n, k] : grid)
            for (int kk : {1, 2}) add(f, packed, m, n, kk);
    }
    if constexpr (std::is_same_v<T, float>) {
        const Form nn{'N', 'N'};
        for (bool packed : {false, true}) {
            for (int s : kEdgeSquares) add(nn, packed, s, s, s);
            for (int s : kSmallSquares)
                for (int v : kOffAxis) {
                    add(nn, packed, v, s, s);
                    add(nn, packed, s, v, s);
                    add(nn, packed, s, s, v);
                }
        }
    }
    return out;
}

template <typename T>
void emit(const char* dtype, const char* device) {
    constexpr bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
    const auto cs = cplx ? cells<T>(kCplxForms, kCplxPanelForms) : cells<T>(kRealForms, kRealPanelForms);
    const std::vector<int> batches = cplx ? std::vector<int>(kBatch.begin(), kBatch.end())
                                          : std::vector<int>(kRealBatch.begin(), kRealBatch.end());
    for (const auto& [f, packed, m, n, k] : cs)
        for (int batch : batches)
            std::printf("gemm,%s,%s,%c,%c,%s,%d,%d,%d,%d,%s\n", dtype, device, f.a, f.b,
                        packed ? "packed" : "strided", m, n, k, batch, ranked<T>(f, packed, m, n, k, batch).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    for (const char* v : {"BATCHLAS_GEMM_SYCL_KERNEL", "BATCHLAS_GEMM_ROUTE", "BATCHLAS_GEMM_VARIANT"})
        if (std::getenv(v)) {
            std::fprintf(stderr, "gemm_transcribe: unset %s first\n", v);
            return 2;
        }
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,ta,tb,layout,m,n,k,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
