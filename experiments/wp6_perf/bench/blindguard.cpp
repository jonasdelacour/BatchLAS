// IS `RouteGetrs.PreferredIsFalseEverywhereAndAbsentDriverIsUnsupported` ABLE TO
// SEE A preferred() WINDOW ON THE FUSED TIER AT ALL?
//
// The routing proposal in this directory's README turns on {Native, CTA}. The
// suite's route-neutrality assertion for getrs
// (tests/route_vocabulary_tests.cc:2195-2211) sweeps order/nrhs/batch and asserts
// `is_vendor(resolve_getrs_route<float>(kGetrsAuto, s, true))` -- but it builds
// its shapes with the file's own `getrs_shape()` helper (:1565-1581), which sets
// blocked_available and NOTHING ELSE, leaving fused_max_elems and fused_max_nrhs
// at their default 0.
//
// If supports({Native, CTA}) is false on such a shape, resolve_route never
// consults preferred() for CTA at all, and the assertion cannot fail no matter
// what window lands there. That is this repository's blind-guard class, and it
// is checked HERE rather than asserted in prose, because the whole point of the
// class is that it looks healthy.
//
// PURE HEADER ONLY. No device, no queue, no library link -- route_resolve.hh
// reads only its arguments by construction, which is what makes this checkable
// in nine lines instead of a GPU run.
//
// build: /opt/dpcpp-cuda/bin/clang++ -std=c++20 -I<repo>/include \
//        -I<repo>/build/include blindguard.cpp -o blindguard
#include <batchlas/blas/dispatch/route_getrs.hh>

#include <cstdio>

using namespace batchlas::dispatch;

static GetrsShape helper_shape(int64_t order, int64_t nrhs, int64_t batch) {
    // tests/route_vocabulary_tests.cc:1565-1581, field for field.
    GetrsShape s;
    s.op = Op::getrs;
    s.scalar = ScalarKind::F32;
    s.backend = batchlas::Backend::AUTO;
    s.m = order;
    s.n = nrhs;
    s.k = order;
    s.batch = batch;
    s.transA = batchlas::Transpose::NoTrans;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = true;
    return s;
}

int main() {
    using Tbl = RouteTable<Op::getrs, float>;
    const Route cta{Origin::Native, Algorithm::CTA};
    int seen = 0, supported = 0;
    for (int64_t order : {int64_t(1), int64_t(32), int64_t(128), int64_t(2048)}) {
        for (int64_t nrhs : {int64_t(1), int64_t(8), int64_t(64)}) {
            for (int64_t batch : {int64_t(1), int64_t(128), int64_t(8192)}) {
                const auto s = helper_shape(order, nrhs, batch);
                ++seen;
                if (Tbl::supports(cta, s)) ++supported;
            }
        }
    }
    std::printf("route_vocabulary_tests' own getrs_shape(): %d shapes swept, "
                "supports({Native,CTA}) true on %d of them\n", seen, supported);
    std::printf("fused_max_elems=%lld fused_max_nrhs=%lld on that helper's shape\n",
                (long long)helper_shape(128, 1, 128).fused_max_elems,
                (long long)helper_shape(128, 1, 128).fused_max_nrhs);
    // And the same shape WITH the capacities a real device query fills in.
    auto real = helper_shape(128, 1, 128);
    real.fused_max_elems = 11104;
    real.fused_max_nrhs = 8;
    std::printf("with the capacities set: supports({Native,CTA}) = %s\n",
                Tbl::supports(cta, real) ? "true" : "false");
    std::printf("VERDICT: the sweep %s see a preferred() window on the fused tier.\n",
                supported == 0 ? "CANNOT" : "can");
    return supported == 0 ? 0 : 1;
}
