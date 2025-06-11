#pragma once
#include <benchmark/benchmark.h>

namespace bench_utils {

inline void SquareSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s});
    }
}

inline void SquareBatchSizes(benchmark::internal::Benchmark* b) {
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

inline void CubeSizes(benchmark::internal::Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s, s});
    }
}

inline void CubeBatchSizes(benchmark::internal::Benchmark* b) {
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
