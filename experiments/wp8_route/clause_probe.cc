// PURE-LAYER PROBE for the four preferred() tables this pass owns.
//
// WHY IT EXISTS. A clause is transcribed from a CSV, and the thing that has to
// match the CSV is not the source text -- it is the DECISION the resolver
// returns. Reading `s.nrhs() >= 64` and believing it admits what you meant is
// how a predicate written on the wrong extent survives review; that error was
// caught twice in WP7. This binary asks the real resolver, cell by cell, on the
// same (type, order, nrhs, batch) / (type, out_len, red_len, batch) grids the
// sweeps used, and prints the resolved route so the admitted set can be diffed
// against the measured set MECHANICALLY.
//
// No SYCL and no device -- but NOT header-only. resolve_route calls
// coverage::record_if_enabled, so the one TU that defines it has to come along
// or the link fails on g_dynamic_enabled and coverage::record. Verified to
// build and run from the worktree root exactly as written:
//   g++ -std=c++20 -I include -I build/include \
//       experiments/wp8_route/clause_probe.cc src/dispatch/coverage.cc \
//       -o experiments/wp8_route/clause_probe
//   ./experiments/wp8_route/clause_probe getri|getrf|getrs|gemv
// (An earlier version of this comment omitted coverage.cc and did not link,
// which means the committed binary was built some other way and nobody could
// reproduce it from the file. A build line that does not build is a defect in
// the record.)
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/dispatch/route_gemv.hh>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

using namespace batchlas;
using namespace batchlas::dispatch;
using cf = std::complex<float>;
using cd = std::complex<double>;

static const char* rs(Route r) {
    if (is_vendor(r)) return "vendor:auto";
    switch (r.algo) {
        case Algorithm::CTA:     return "native:cta";
        case Algorithm::Blocked: return "native:blocked";
        case Algorithm::Direct:  return "native:direct";
        default:                 return "native:?";
    }
}

static GetrfShape rf(int64_t n, int64_t b) {
    GetrfShape s; s.op = Op::getrf; s.scalar = ScalarKind::F32; s.backend = Backend::CUDA;
    s.m = s.n = s.k = n; s.batch = b; s.is_gpu = true; s.has_sg32 = true;
    s.cta_max_n = 155; s.blocked_available = true; return s;
}
static GetriShape ri(int64_t n, int64_t b) {
    GetriShape s; s.op = Op::getri; s.scalar = ScalarKind::F32; s.backend = Backend::CUDA;
    s.m = s.n = s.k = n; s.batch = b; s.is_gpu = true; s.has_sg32 = true;
    s.blocked_available = true; return s;
}
static GetrsShape rr(int64_t n, int64_t nrhs, int64_t b) {
    GetrsShape s; s.op = Op::getrs; s.scalar = ScalarKind::F32; s.backend = Backend::CUDA;
    s.m = n; s.n = nrhs; s.k = n; s.batch = b; s.transA = Transpose::NoTrans;
    s.is_gpu = true; s.has_sg32 = true; s.blocked_available = true;
    s.fused_max_elems = 23264; s.fused_max_nrhs = 8; return s;
}
static GemvShape gv(int64_t out, int64_t red, int64_t b, Transpose t) {
    GemvShape s; s.op = Op::gemv; s.scalar = ScalarKind::F32; s.backend = Backend::CUDA;
    // Under Trans/ConjTrans out_len == n == cols and red_len == m == rows.
    s.m = red; s.n = out; s.k = red; s.batch = b; s.transA = t;
    s.is_gpu = true; s.has_sg32 = true;
    s.direct_available = true; s.cta_available = true; return s;
}

template <typename T> static const char* nm();
template <> const char* nm<float>()  { return "float"; }
template <> const char* nm<double>() { return "double"; }
template <> const char* nm<cf>()     { return "cfloat"; }
template <> const char* nm<cd>()     { return "cdouble"; }
template <typename T> static ScalarKind sk();
template <> ScalarKind sk<float>()  { return ScalarKind::F32; }
template <> ScalarKind sk<double>() { return ScalarKind::F64; }
template <> ScalarKind sk<cf>()     { return ScalarKind::C32; }
template <> ScalarKind sk<cd>()     { return ScalarKind::C64; }

static const std::vector<int64_t> kN{32, 64, 128, 155, 156, 256, 512, 1024, 2048};
static const std::vector<int64_t> kB{1, 2, 4, 32, 64, 128, 192, 256, 320, 384,
                                     512, 1024, 2048, 4096, 8192, 16384};

template <typename T> static void do_getri() {
    for (auto n : kN) for (auto b : kB) {
        auto s = ri(n, b); s.scalar = sk<T>();
        std::printf("getri,%s,%lld,1,%lld,%s\n", nm<T>(), (long long)n, (long long)b,
                    rs(resolve_getri_route<T>(Route{}, s, true)));
    }
}
template <typename T> static void do_getrf() {
    for (auto n : kN) for (auto b : kB) {
        auto s = rf(n, b); s.scalar = sk<T>();
        std::printf("getrf,%s,%lld,1,%lld,%s\n", nm<T>(), (long long)n, (long long)b,
                    rs(resolve_getrf_route<T>(Route{}, s, true)));
    }
}
template <typename T> static void do_getrs() {
    for (auto n : {int64_t(64), int64_t(128), int64_t(512), int64_t(1024)})
      for (auto q : {int64_t(1), int64_t(2), int64_t(4), int64_t(8), int64_t(16),
                     int64_t(32), int64_t(63), int64_t(64), int64_t(127), int64_t(128)})
        for (auto b : kB) {
            auto s = rr(n, q, b); s.scalar = sk<T>();
            std::printf("getrs,%s,%lld,%lld,%lld,%s\n", nm<T>(), (long long)n,
                        (long long)q, (long long)b,
                        rs(resolve_getrs_route<T>(Route{}, s, true)));
        }
}
template <typename T> static void do_gemv() {
    const Transpose trs[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};
    for (auto tr : trs)
      for (auto o : {int64_t(1), int64_t(64), int64_t(128), int64_t(192), int64_t(256),
                     int64_t(512), int64_t(767), int64_t(768), int64_t(1024), int64_t(2048)})
        for (auto r : {int64_t(1), int64_t(32), int64_t(48), int64_t(63), int64_t(64),
                       int64_t(128), int64_t(256), int64_t(352), int64_t(353),
                       int64_t(512), int64_t(1024)})
          for (auto b : kB) {
            auto s = gv(o, r, b, tr); s.scalar = sk<T>();
            std::printf("gemv,%s,%d,%lld,%lld,%lld,%s\n", nm<T>(), int(tr),
                        (long long)o, (long long)r, (long long)b,
                        rs(resolve_gemv_route<T>(Route{}, s, true)));
        }
}

int main(int argc, char** argv) {
    const std::string op = argc > 1 ? argv[1] : "all";
    if (op == "getri" || op == "all") { do_getri<float>(); do_getri<double>(); do_getri<cf>(); do_getri<cd>(); }
    if (op == "getrf" || op == "all") { do_getrf<float>(); do_getrf<double>(); do_getrf<cf>(); do_getrf<cd>(); }
    if (op == "getrs" || op == "all") { do_getrs<float>(); do_getrs<double>(); do_getrs<cf>(); do_getrs<cd>(); }
    if (op == "gemv"  || op == "all") { do_gemv<float>();  do_gemv<double>();  do_gemv<cf>();  do_gemv<cd>(); }
    return 0;
}
