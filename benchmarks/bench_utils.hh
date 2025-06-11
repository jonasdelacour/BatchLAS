#pragma once
#include <benchmark/benchmark.h>

namespace bench_utils {

inline void SquareSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s});
    }
}

inline void SquareBatchSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256}) {
        b->Args({s, s, 10});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, 100});
    }
}

inline void CubeSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, s});
    }
}

inline void CubeBatchSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256}) {
        b->Args({s, s, s, 10});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, s, 100});
    }
}

} // namespace bench_utils
