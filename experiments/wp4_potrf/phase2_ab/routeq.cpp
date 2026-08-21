// Ask the RESOLVER directly what potrf's trailing-update and panel-solve shapes
// resolve to, with and without a vendor.
//
// WHY NOT scripts/route_diff.sh, which Open Question 6 names. That script runs
// the whole ctest suite and diffs the routes the TESTS reach. No test in the
// tree issues potrf's trailing-update shape -- Phase 2 has not been written --
// so a capture would contain zero rows for the question being asked. It is also
// blind to KernelVariant by construction (it records resolver Routes only), and
// KernelVariant is precisely where the ConjTrans short-circuit lives. This
// program answers the Route half exactly; the trace in routes.txt answers the
// KernelVariant half.
//
// Pure host code: route_gemm.hh and route_trsm.hh read only their arguments.
#include <batchlas/blas/dispatch/route_gemm.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>

#include <cstdio>
#include <string>

// The route headers call the coverage recorder. Stub it: this program is not
// linked against the library's dispatch TU and records nothing.
namespace batchlas::dispatch::coverage {
bool g_dynamic_enabled = false;
void record(Op, ScalarKind, Backend, const OpShape&, Route, bool, int) {}
}  // namespace batchlas::dispatch::coverage

using namespace batchlas;
using namespace batchlas::dispatch;

static const char* orig(Origin o) {
    switch (o) {
        case Origin::Auto: return "Auto";
        case Origin::Native: return "Native";
        case Origin::Vendor: return "Vendor";
    }
    return "?";
}
static const char* alg(Algorithm a) {
    switch (a) {
        case Algorithm::Auto: return "Auto";
        case Algorithm::Direct: return "Direct";
        case Algorithm::RegisterTiled: return "RegisterTiled";
        case Algorithm::CTA: return "CTA";
        case Algorithm::Blocked: return "Blocked";
        default: return "other";
    }
}

template <typename T>
static void gemm_row(const char* tn, int64_t m, int64_t n, int64_t k, int64_t batch,
                     Transpose tB) {
    OpShape s;
    s.op = Op::gemm;
    s.m = m; s.n = n; s.k = k; s.batch = batch;
    s.transA = Transpose::NoTrans; s.transB = tB;
    s.is_gpu = true;
    for (int va = 1; va >= 0; --va) {
        const Route r = resolve_gemm_route<T>(Route{}, s, va != 0);
        std::printf("gemm,%s,%lld,%lld,%lld,%lld,N%c,%s,%s:%s\n", tn,
                    (long long)m, (long long)n, (long long)k, (long long)batch,
                    tB == Transpose::Trans ? 'T' : 'C',
                    va ? "vendor_present" : "vendor_free",
                    orig(r.origin), alg(r.algo));
    }
}

template <typename T>
static void trsm_row(const char* tn, int64_t order, int64_t q, int64_t batch,
                     int cta_max_n, bool blocked_available) {
    TrsmShape s;
    s.op = Op::trsm;
    s.m = q; s.n = order; s.k = order;
    s.batch = batch;
    s.side = Side::Right; s.uplo = Uplo::Lower;
    s.transA = Transpose::ConjTrans; s.diag = Diag::NonUnit;
    s.is_gpu = true;
    s.cta_max_n = cta_max_n;
    s.blocked_available = blocked_available;
    for (int va = 1; va >= 0; --va) {
        const Route r = resolve_trsm_route<T>(Route{}, s, va != 0);
        std::printf("trsm,%s,order=%lld,q=%lld,batch=%lld,%s,%s:%s\n", tn,
                    (long long)order, (long long)q, (long long)batch,
                    va ? "vendor_present" : "vendor_free",
                    orig(r.origin), alg(r.algo));
    }
}

int main() {
    std::printf("op,type,...,vendor,route\n");
    struct Cfg { const char* tn; int ib; int m1; int m2; };
    const Cfg cfgs[] = {{"float", 155, 869, 1893},
                        {"double", 109, 915, 1939},
                        {"cfloat", 109, 915, 1939},
                        {"cdouble", 77, 947, 1971}};
    for (const Cfg& c : cfgs) {
        const std::string t = c.tn;
        auto emit = [&](int64_t m, int64_t n, int64_t k, Transpose tB) {
            if (t == "float")   gemm_row<float>(c.tn, m, n, k, 128, tB);
            if (t == "double")  gemm_row<double>(c.tn, m, n, k, 128, tB);
            if (t == "cfloat")  gemm_row<std::complex<float>>(c.tn, m, n, k, 128, tB);
            if (t == "cdouble") gemm_row<std::complex<double>>(c.tn, m, n, k, 128, tB);
        };
        emit(c.m1, 128, c.ib, Transpose::ConjTrans);   // below-diagonal rectangle
        emit(128, 128, c.ib, Transpose::ConjTrans);    // diagonal WxW block
        emit(c.m2, 128, c.ib, Transpose::ConjTrans);
        emit(c.m1, c.m1, c.ib, Transpose::ConjTrans);  // whole-A22 (rejected form)
        emit(c.m1, 128, c.ib, Transpose::Trans);       // the real-type alternative
    }
    for (const Cfg& c : cfgs) {
        const std::string t = c.tn;
        auto emit = [&](int64_t order, int64_t q, int64_t batch) {
            if (t == "float")   trsm_row<float>(c.tn, order, q, batch, 32, true);
            if (t == "double")  trsm_row<double>(c.tn, order, q, batch, 32, true);
            if (t == "cfloat")  trsm_row<std::complex<float>>(c.tn, order, q, batch, 32, true);
            if (t == "cdouble") trsm_row<std::complex<double>>(c.tn, order, q, batch, 32, true);
        };
        emit(c.ib, c.m1, 128);
        emit(c.ib, c.m1, 64);
        emit(c.ib, c.m1, 8);
    }
    return 0;
}
