// posv transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::posv,T> + resolve_route, at every grid cell of src/ops/posv/choice.hh and prints
// its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_posv.hh (7e71a6e0, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/posv_transcribe.cc -o /tmp/pt
//   /tmp/pt sm_89 > tuned/transcribed/posv.sm_89.csv
//
// The order is DERIVED, not written down: resolve the cell, record the winner, mark it unsupported
// and resolve again, until the walk returns no native route. Capacities (the tiny ceilings, the
// fused solve's SLM capacity) are unlimited here because can_run re-applies them at run time; the
// device is a GPU with sub-group 32. The list stops after `blocked`, which can_run never refuses.

#include <batchlas/blas/dispatch/route_posv.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kPosvOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kPosvOrder); ++i)
        if (kPosvOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::posv, T>;
    static bool supports(Route r, const PosvShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const PosvShape& s) { return Real::preferred(r, s); }
    static bool native_tier_preferred(Route r, const PosvShape& s) { return Real::native_tier_preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/posv/choice.hh's grid (potrf's grid_n within [1,1024]).
constexpr std::array<int, 33> kGridN{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64,
                                     80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024};
constexpr std::array<int, 6> kGridNrhs{1, 2, 4, 8, 16, 64};
constexpr std::array<int, 5> kGridBatch{128, 512, 2048, 8192, 32768};

std::string spelling(Route r) {
    switch (r.algo) {
        case Algorithm::Tiny: return "tiny";
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
std::string ranked(Uplo uplo, int n, int nrhs, int batch) {
    PosvShape s;
    s.op = Op::posv;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = s.k = n;
    s.n = nrhs;
    s.batch = batch;
    s.uplo = uplo;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.tiny_max_n = s.tiny_max_nrhs = s.fused_max_nrhs = s.fused_max_rhs_elems = INT64_MAX / 4;
    s.composed_available = true;
    std::string out;
    g_excluded = 0;
    for (;;) {
        const Route r = resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto}, s, false);
        if (!is_native(r)) break;
        out += (out.empty() ? "" : "|") + spelling(r);
        if (r.algo == Algorithm::Blocked) break;
        g_excluded |= bit_of(r);
    }
    return out;
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper})
        for (int n : kGridN)
            for (int nrhs : kGridNrhs)
                for (int batch : kGridBatch)
                    std::printf("posv,%s,%s,%s,%d,%d,%d,%s\n", dtype, device, u == Uplo::Lower ? "L" : "U", n, nrhs,
                                batch, ranked<T>(u, n, nrhs, batch).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,uplo,n,nrhs,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
