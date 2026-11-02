// SPDX-License-Identifier: MIT
// SASS lab, device LA v2 parity: hand vs lean vs v2 for each op at WG 64 and 256 (dot/nrm2
// also at 32 = one sub-group). Kernel names lab::KW<Op><T, N, V, WG>; sass_stats.py keys them
// as "<Op>@<WG>". The hand and lean bodies are the lab's (ops_*.cc), with WG a parameter.
#pragma once

#include <benchmarks/sass_lab/lab.hh>

#include <batchlas/blas/device/v2/dla.hh>

namespace lab {

constexpr int kV2 = 9;        // device LA v2 (include/batchlas/blas/device/v2)
constexpr int kV2Scaled = 10; // nrm2 only: alg::Scaled (two-pass lassq)
constexpr int kV2Blue = 11;   // nrm2 only: alg::Blue (three accumulators, one exchange)
constexpr int kV2Scratch = 12; // dotN2 only: dotN<2> with Scratch (one packed exchange)

namespace dv2 = batchlas::device::v2;

template <class T, int N, int V, int WG> class KWAxpy;
template <class T, int N, int V, int WG> class KWDot;
template <class T, int N, int V, int WG> class KWNrm2;
template <class T, int N, int V, int WG> class KWDotN2;
template <class T, int N, int V, int WG> class KWGemvN;
template <class T, int N, int V, int WG> class KWGemvT;
template <class T, int N, int V, int WG> class KWSymv;
template <class T, int N, int V, int WG> class KWGer;
template <class T, int N, int V, int WG> class KWLatrd;

template <class T, int N, int V, class F> inline void addw(const char* op, int wg, bool inplace, F f) {
    const std::string name = std::string(op) + "@" + std::to_string(wg);
    registry().push_back(Entry{name, tname<T>(), N, V, inplace, [f](sycl::queue& q, void* d, int n) {
        return f(q, *static_cast<Data<T>*>(d), n);
    }});
}

}  // namespace lab
