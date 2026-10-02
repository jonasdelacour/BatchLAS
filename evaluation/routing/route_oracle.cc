// route_oracle: transcribe an op's hand-written RouteTable policy into ranked candidate lists,
// for evaluation/routing/compile_rules.py --op getrs. Needs no GPU.
//
//   route_oracle getrs   stdout: "<dtype> <trans> <batch> <nrhs> <n0> <n1|inf> <rank>", runs
//                        over n; each axis is dense at small values, then powers of two, and a
//                        sample covers up to the next sample (the compiler's band semantics)
//
// The rank is legality-free: preferred() and native_tier_preferred() are asked of a shape whose
// capacity fields are unbounded, in walk order: preferred hits, then the vendor, then the
// tie-break hits, then every native tier. The engine's first-LEGAL pass over that list is the
// resolver's walk exactly whenever preferred()/native_tier_preferred() read no capacity field
// (true for getrs); capacity stays where it belongs, in legal().

#include <batchlas/blas/dispatch/route_getrs.hh>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace batchlas;
using namespace batchlas::dispatch;

namespace {

const char* kTier[] = {"native:fused", "native:blocked", "vendor"};

int tier_index(Route r) {
    if (is_vendor(r)) return 2;
    return r.algo == Algorithm::CTA ? 0 : 1;
}

// Up to three tiers, 2 bits each, plus a count.
template <class T>
unsigned rank_code(int64_t n, int64_t nrhs, int64_t batch, Transpose tr) {
    using Tbl = RouteTable<Op::getrs, T>;
    GetrsShape s;
    s.op = Op::getrs;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = n;
    s.n = nrhs;
    s.k = n;
    s.batch = batch;
    s.transA = tr;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = true;
    s.fused_max_elems = INT64_MAX / 4;
    s.fused_max_nrhs = INT64_MAX / 4;
    unsigned code = 0, cnt = 0, seen = 0;
    auto push = [&](Route r) {
        const int t = tier_index(r);
        if (seen & (1u << t)) return;
        seen |= 1u << t;
        code |= static_cast<unsigned>(t) << (2 * cnt);
        ++cnt;
    };
    for (const Route* r = Tbl::order_begin(); r != Tbl::order_end(); ++r) {
        if (is_native(*r) && Tbl::preferred(*r, s)) push(*r);
    }
    push(Route{Origin::Vendor, Algorithm::Auto});
    for (const Route* r = Tbl::order_begin(); r != Tbl::order_end(); ++r) {
        if (is_native(*r) && Tbl::native_tier_preferred(*r, s)) push(*r);
    }
    for (const Route* r = Tbl::order_begin(); r != Tbl::order_end(); ++r) {
        if (is_native(*r)) push(*r);
    }
    return code | (cnt << 8);
}

std::string rank_text(unsigned code) {
    std::string j;
    for (unsigned i = 0; i < (code >> 8); ++i) {
        j += (i ? " > " : "");
        j += kTier[(code >> (2 * i)) & 3];
    }
    return j;
}

std::vector<int64_t> axis(int dense, int pow2_max) {
    std::vector<int64_t> v;
    for (int i = 1; i <= dense; ++i) v.push_back(i);
    for (int64_t p = 1; p <= (int64_t(1) << pow2_max); p <<= 1) {
        if (p > dense) v.push_back(p);
    }
    return v;
}

template <class T>
void dump(const char* dt) {
    const char* trn[] = {"N", "T", "C"};
    const Transpose trv[] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};
    const auto ns = axis(160, 13), rs = axis(160, 13), bs = axis(160, 20);
    for (int t = 0; t < 3; ++t) {
        for (int64_t b : bs) {
            for (int64_t r : rs) {
                std::size_t i0 = 0;
                unsigned cur = rank_code<T>(ns[0], r, b, trv[t]);
                for (std::size_t i = 1; i <= ns.size(); ++i) {
                    const unsigned lab = (i < ns.size()) ? rank_code<T>(ns[i], r, b, trv[t]) : ~0u;
                    if (lab == cur) continue;
                    const std::string hi = (i < ns.size()) ? std::to_string(ns[i] - 1) : "inf";
                    std::printf("%s %s %lld %lld %lld %s %s\n", dt, trn[t],
                                static_cast<long long>(b), static_cast<long long>(r),
                                static_cast<long long>(ns[i0]), hi.c_str(), rank_text(cur).c_str());
                    i0 = i;
                    cur = lab;
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "getrs") {
        std::fprintf(stderr, "usage: route_oracle getrs\n");
        return 2;
    }
    dump<float>("float");
    dump<double>("double");
    dump<std::complex<float>>("cfloat");
    dump<std::complex<double>>("cdouble");
    return 0;
}
