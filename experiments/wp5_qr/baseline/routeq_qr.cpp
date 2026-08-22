// Ask the RESOLVER what geqrf's WY trailing-update GEMMs will route to, per
// scalar type, vendor-present and vendor-free.
//
// A blocked right-looking geqrf at panel start j0 with block width ib on an
// N x N parent issues three GEMM-shaped operations, with m1 = N - j0 the panel
// height and n2 = N - j0 - ib the trailing width:
//
//   G1   W   = V^H A22    m=ib, n=n2, k=m1   transA = Trans/ConjTrans
//   G2   W   = T^H W      m=ib, n=n2, k=ib   transA = Trans/ConjTrans
//   G3   A22 = A22 - V W  m=m1, n=n2, k=ib   NN
//
// G1 and G3 are the ones that carry the arithmetic. Both operands of both are
// SUB-VIEWS of the parent at the parent ld -- which the resolver does not see,
// but the KernelVariant selector does, so the Route half is answered here and
// the variant half by the traced run in run_gemmtrace.sh.
//
// Pure host code: route_gemm.hh reads only its arguments. Stub the coverage
// recorder, exactly as experiments/wp4_potrf/phase2_ab/routeq.cpp does.
#include <batchlas/blas/dispatch/route_gemm.hh>

#include <complex>
#include <cstdio>
#include <string>
#include <vector>

namespace batchlas::dispatch::coverage {
bool g_dynamic_enabled = false;
void record(Op, ScalarKind, Backend, const OpShape&, Route, bool, int) {}
}  // namespace batchlas::dispatch::coverage

using namespace batchlas;
using namespace batchlas::dispatch;

static const char* orig(Origin o) {
    switch (o) { case Origin::Auto: return "Auto"; case Origin::Native: return "Native";
                 case Origin::Vendor: return "Vendor"; }
    return "?";
}
static const char* alg(Algorithm a) {
    switch (a) { case Algorithm::Auto: return "Auto"; case Algorithm::Direct: return "Direct";
                 case Algorithm::RegisterTiled: return "RegisterTiled"; case Algorithm::CTA: return "CTA";
                 case Algorithm::Blocked: return "Blocked"; default: return "other"; }
    return "?";
}

template <typename T>
static void row(const char* tn, const char* tag, int64_t m, int64_t n, int64_t k,
                int64_t batch, Transpose tA, Transpose tB) {
    OpShape s;
    s.op = Op::gemm;
    s.m = m; s.n = n; s.k = k; s.batch = batch;
    s.transA = tA; s.transB = tB;
    s.is_gpu = true;
    for (int va = 1; va >= 0; --va) {
        const Route r = resolve_gemm_route<T>(Route{}, s, va != 0);
        std::printf("%s,%s,%lld,%lld,%lld,%lld,%s,%s,%s:%s\n", tag, tn,
                    (long long)m, (long long)n, (long long)k, (long long)batch,
                    tA == Transpose::NoTrans ? "N" : (tA == Transpose::Trans ? "T" : "C"),
                    va ? "vendor_present" : "vendor_free", orig(r.origin), alg(r.algo));
    }
}

struct Cell { int N, nb, batch; };

template <typename T>
static void emit_type(const char* tn, bool complex_type) {
    // The block widths the SHIPPED tuning table returns for ormqr, which is what
    // a geqrf driver would inherit unless WP5 adds its own constant:
    //   n<=64 -> 16, n<=128 -> 16, n<=256 -> 24, n<=512 -> 48, else 56.
    const std::vector<Cell> cells = {
        {256, 24, 2048}, {512, 48, 512}, {1024, 56, 128}, {2048, 56, 32}};
    const Transpose tA = complex_type ? Transpose::ConjTrans : Transpose::Trans;
    for (const Cell& c : cells) {
        // Three panel positions: the first panel, the middle, and the SHORT
        // FINAL panel (N % nb), which is the shape class that produced the
        // sy2sb silent failure.
        const int last_full = ((c.N - 1) / c.nb) * c.nb;
        const int tail = c.N - last_full;
        const int js[3] = {0, (c.N / 2 / c.nb) * c.nb, last_full - c.nb};
        for (int p = 0; p < 3; ++p) {
            const int j0 = js[p];
            const int ib = c.nb;
            const int m1 = c.N - j0, n2 = c.N - j0 - ib;
            if (n2 <= 0) continue;
            char tag[96];
            std::snprintf(tag, sizeof tag, "G1_N%d_nb%d_j%d", c.N, c.nb, j0);
            row<T>(tn, tag, ib, n2, m1, c.batch, tA, Transpose::NoTrans);
            std::snprintf(tag, sizeof tag, "G3_N%d_nb%d_j%d", c.N, c.nb, j0);
            row<T>(tn, tag, m1, n2, ib, c.batch, Transpose::NoTrans, Transpose::NoTrans);
        }
        // The short final panel itself: ib = N % nb (or nb when it divides).
        const int ib = (tail == 0) ? c.nb : tail;
        const int j0 = last_full;
        const int m1 = c.N - j0, n2 = c.N - j0 - ib;
        char tag[96];
        std::snprintf(tag, sizeof tag, "G1_TAIL_N%d_nb%d_ib%d", c.N, c.nb, ib);
        if (n2 > 0) row<T>(tn, tag, ib, n2, m1, c.batch, tA, Transpose::NoTrans);
        std::snprintf(tag, sizeof tag, "G3_TAIL_N%d_nb%d_ib%d", c.N, c.nb, ib);
        if (n2 > 0) row<T>(tn, tag, m1, n2, ib, c.batch, Transpose::NoTrans, Transpose::NoTrans);
    }
}

int main() {
    std::printf("shape,type,m,n,k,batch,transA,vendor,route\n");
    emit_type<float>("float", false);
    emit_type<double>("double", false);
    emit_type<std::complex<float>>("cfloat", true);
    emit_type<std::complex<double>>("cdouble", true);
    return 0;
}
