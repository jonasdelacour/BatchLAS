#pragma once
#include <util/minibench.hh>

namespace bench_utils {

inline void SquareSizes(minibench::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s});
    }
}

inline void SquareBatchSizes(minibench::Benchmark* b) {
    for (int s : {64, 128, 256}) {
        b->Args({s, s, 1});
    }
    for (int s : {64, 128, 256}) {
        b->Args({s, s, 10});
    }
    for (int s : {64, 128, 256, 512}) {
        b->Args({s, s, 100});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, 1000});
    }
}

inline void CubeSizes(minibench::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s, s});
    }
}

inline void CubeBatchSizes(minibench::Benchmark* b) {
    for (int s : {64, 128, 256}) {
        b->Args({s, s, s, 1});
    }
    for (int s : {64, 128, 256}) {
        b->Args({s, s, s, 10});
    }
    for (int s : {64, 128, 256, 512}) {
        b->Args({s, s, s, 100});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, s, 1000});
    }
}

} // namespace bench_utils
