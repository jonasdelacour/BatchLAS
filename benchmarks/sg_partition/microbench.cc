// Standalone microbenchmark for the chunked sub-group partition layer on NVPTX.
//
// Built by hand (not part of the CMake tree):
//   clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_89 -O3 -std=c++20 \
//       -I src/extensions benchmarks/sg_partition/microbench.cc -o sgp_microbench
//   ONEAPI_DEVICE_SELECTOR=cuda:1 ./sgp_microbench [log2_lanes=24] [iters=64] [reps=30]
//
// Variants, all doing the same per-chunk work:
//   masked    batchlas::SubGroupPartition<P, true>   (per-chunk member mask)
//   lockstep  batchlas::SubGroupPartition<P, false>  (constant full mask)
//   native    DPC++ ext::oneapi::experimental::chunked_partition<P>
//   emulated  the pre-sg_partition path: plain sub-group collectives at base + lane
// Each kernel runs `iters` iterations of one collective pattern at saturation.
// Every kernel's inner loop carries `#pragma unroll 1` so its SASS loop body is
// exactly one iteration (see the report's instruction counts).
//
// "div" rows give every chunk its own trip count (iters - 8 * (chunk % 4)), so
// the chunks of a warp diverge; only masked and native are legal there.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sg_partition/sg_partition.hh"

namespace syclex = sycl::ext::oneapi::experimental;

enum Var : int { kMasked = 0, kLockstep = 1, kNative = 2, kEmulated = 3, kRedux = 4 };
// kPerm is a hand-written permute_group_by_xor butterfly (one front-end call
// per step); kReduce is reduce_over_group (native/emulated have no float
// reduce on CUDA, so they run the same butterfly by hand).
enum Op : int { kPerm = 0, kBcast = 1, kShift = 2, kVote = 3, kReduce = 4, kReduceDiv = 5, kSelectExit = 6 };

static const char* op_name(int o) {
    static const char* n[] = {"xor-perm", "select", "shift-left", "any", "reduce", "reduce-div", "select-exit"};
    return n[o];
}

// One kernel type per (variant, op, P, T), so each has its own SASS function.
template <int V, int O, uint32_t P, typename T>
struct Kernel {
    T* data;
    const int* trips;
    int iters;

    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const size_t gid = it.get_global_id(0);
        const uint32_t sgl = static_cast<uint32_t>(sg.get_local_linear_id());
        const uint32_t base = sgl & ~(P - 1u);
        const uint32_t lid = sgl & (P - 1u);
        T v = data[gid];
        // select-exit: odd chunks of every warp leave at once, so the warp is
        // never whole again (the masked fast path never applies).
        if constexpr (O == kSelectExit && P < 32) {
            if ((sgl / P) & 1u) return;
        }
        int n = iters;
        if constexpr (O == kReduceDiv) n = trips[gid / P];

        auto part_m = batchlas::make_partition<P, true>(sg);
        auto part_l = batchlas::make_partition<P, false>(sg);
#ifdef __SYCL_DEVICE_ONLY__
        auto chunk = syclex::chunked_partition<P>(sg);
#endif
        auto xr = [&](T x, uint32_t m) -> T {
            if constexpr (V == kMasked) return batchlas::permute_group_by_xor(part_m, x, m);
            else if constexpr (V == kLockstep) return batchlas::permute_group_by_xor(part_l, x, m);
            else if constexpr (V == kEmulated) return sycl::permute_group_by_xor(sg, x, m);
            else {
#ifdef __SYCL_DEVICE_ONLY__
                return sycl::permute_group_by_xor(chunk, x, m);
#else
                return x;
#endif
            }
        };
        auto sel = [&](T x, uint32_t src) -> T {
            if constexpr (V == kMasked) return batchlas::select_from_group(part_m, x, src);
            else if constexpr (V == kLockstep) return batchlas::select_from_group(part_l, x, src);
            else if constexpr (V == kEmulated) return sycl::select_from_group(sg, x, base + src);
            else {
#ifdef __SYCL_DEVICE_ONLY__
                return sycl::select_from_group(chunk, x, src);
#else
                return x;
#endif
            }
        };
        auto shl = [&](T x) -> T {
            if constexpr (V == kMasked) return batchlas::shift_group_left(part_m, x, 1u);
            else if constexpr (V == kLockstep) return batchlas::shift_group_left(part_l, x, 1u);
            else if constexpr (V == kEmulated) return sycl::shift_group_left(sg, x, 1u);
            else {
#ifdef __SYCL_DEVICE_ONLY__
                return sycl::shift_group_left(chunk, x, 1u);
#else
                return x;
#endif
            }
        };
        auto any = [&](bool p) -> bool {
            if constexpr (V == kMasked) return batchlas::any_of_group(part_m, p);
            else if constexpr (V == kLockstep) return batchlas::any_of_group(part_l, p);
            else if constexpr (V == kEmulated) {
                // The old layer had no vote: an OR butterfly over the chunk.
                uint32_t b = p ? 1u : 0u;
                for (uint32_t m = 1; m < P; m <<= 1) b |= sycl::permute_group_by_xor(sg, b, m);
                return b != 0u;
            } else {
#ifdef __SYCL_DEVICE_ONLY__
                return sycl::any_of_group(chunk, p);
#else
                return p;
#endif
            }
        };

        const T half = T(0.5);
        const T inv = T(1) / T(P);
#pragma unroll 1
        for (int k = 0; k < n; ++k) {
            if constexpr ((O == kReduce || O == kReduceDiv) && V == kMasked) {
                v = batchlas::reduce_over_group(part_m, v, sycl::plus<T>()) * inv + T(1);
            } else if constexpr (O == kReduce && V == kLockstep) {
                v = batchlas::reduce_over_group(part_l, v, sycl::plus<T>()) * inv + T(1);
            } else if constexpr (O == kPerm || O == kReduce || O == kReduceDiv) {
                T r = v;
#pragma unroll
                for (uint32_t m = 1; m < P; m <<= 1) r += xr(r, m);
                v = r * inv + T(1);
            } else if constexpr (O == kBcast || O == kSelectExit) {
                v = sel(v, (static_cast<uint32_t>(k) + lid) & (P - 1u)) * half + v * half;
            } else if constexpr (O == kShift) {
                const T s = shl(v);
                v = (lid + 1u < P ? s : v) * half + T(1);
            } else if constexpr (O == kVote) {
                v = any(v > T(3) + T(lid)) ? v * half : v + T(1);
            }
        }
        data[gid] = v;
    }
};

// Integer reduction: reduce_over_group (redux.sync at P = 32, butterfly below)
// vs a direct redux.sync with the chunk mask.
template <int V, uint32_t P>
struct IntKernel {
    uint32_t* data;
    int iters;
    [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const size_t gid = it.get_global_id(0);
        const uint32_t base = static_cast<uint32_t>(sg.get_local_linear_id()) & ~(P - 1u);
        uint32_t v = data[gid];
        auto part_m = batchlas::make_partition<P, true>(sg);
        auto part_l = batchlas::make_partition<P, false>(sg);
#pragma unroll 1
        for (int k = 0; k < iters; ++k) {
            uint32_t r;
            if constexpr (V == kMasked) {
                r = batchlas::reduce_over_group(part_m, v, sycl::plus<uint32_t>());
            } else if constexpr (V == kLockstep) {
                r = batchlas::reduce_over_group(part_l, v, sycl::plus<uint32_t>());
            } else {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
                const uint32_t mask = P >= 32 ? ~0u : (((1u << (P & 31u)) - 1u) << base);
                r = __nvvm_redux_sync_add(v, mask);
#else
                r = v + base;
#endif
            }
            v = (r >> 2) + static_cast<uint32_t>(k);
        }
        data[gid] = v;
    }
};

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

template <typename K>
static double time_kernel(sycl::queue& q, size_t n, const K& k, int reps) {
    const sycl::nd_range<1> r{sycl::range<1>{n}, sycl::range<1>{256}};
    if (reps <= 0) return 0.0;  // reps = 0: one launch per kernel, for ncu counting
    for (int i = 0; i < 3; ++i) q.parallel_for(r, k).wait();  // JIT + clocks
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        auto e = q.parallel_for(r, k);
        e.wait();
        t.push_back((e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
                     e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6);
    }
    return median(t);
}

struct Ctx {
    sycl::queue q;
    size_t n;
    int iters, reps;
    int* trips;
};

template <typename T>
static const char* tname() {
    return sizeof(T) == 4 ? "float" : "double";
}

template <uint32_t P, typename T, int V, int O>
static double run_one(Ctx& c, std::vector<T>& out, const std::vector<T>& init) {
    T* d = sycl::malloc_device<T>(c.n, c.q);
    Kernel<V, O, P, T> k{d, c.trips, c.iters};
    c.q.memcpy(d, init.data(), c.n * sizeof(T)).wait();
    // Correctness run on a fresh input, then time (timing iterates in place).
    c.q.parallel_for(sycl::nd_range<1>{c.n, 256}, k).wait();
    out.resize(c.n);
    c.q.memcpy(out.data(), d, c.n * sizeof(T)).wait();
    const double ms = time_kernel(c.q, c.n, k, c.reps);
    sycl::free(d, c.q);
    return ms;
}

template <uint32_t P, typename T, int O>
static void run_op(Ctx& c, const std::vector<T>& init) {
    std::vector<T> ref, out;
    double ms[4] = {-1, -1, -1, -1};
    double err = 0;
    auto diff = [&] {
        for (size_t i = 0; i < c.n; ++i) err = std::max(err, double(std::abs(out[i] - ref[i])));
    };
    ms[kMasked] = run_one<P, T, kMasked, O>(c, ref, init);
    if constexpr (O != kReduceDiv) {
        ms[kLockstep] = run_one<P, T, kLockstep, O>(c, out, init);
        diff();
        ms[kEmulated] = run_one<P, T, kEmulated, O>(c, out, init);
        diff();
    }
    ms[kNative] = run_one<P, T, kNative, O>(c, out, init);
    diff();
    std::printf("| %2u | %-6s | %-14s | %8.4f | %8.4f | %8.4f | %8.4f | %5.2f | %5.2f | %5.2f | %.1e |\n", P,
                tname<T>(), op_name(O), ms[0], ms[1], ms[2], ms[3], ms[2] / ms[0], ms[1] > 0 ? ms[1] / ms[0] : 0.0,
                ms[3] > 0 ? ms[3] / ms[0] : 0.0, err);
    std::fflush(stdout);
}

template <uint32_t P, typename T>
static void run_p(Ctx& c) {
    std::vector<T> init(c.n);
    for (size_t i = 0; i < c.n; ++i) init[i] = T((i * 2654435761u) % 1000) / T(250);
    run_op<P, T, kPerm>(c, init);
    run_op<P, T, kReduce>(c, init);
    run_op<P, T, kBcast>(c, init);
    run_op<P, T, kShift>(c, init);
    run_op<P, T, kVote>(c, init);
    run_op<P, T, kReduceDiv>(c, init);
    run_op<P, T, kSelectExit>(c, init);
}

template <uint32_t P, int V>
static double run_int(Ctx& c, std::vector<uint32_t>& out) {
    std::vector<uint32_t> init(c.n);
    for (size_t i = 0; i < c.n; ++i) init[i] = static_cast<uint32_t>(i * 2654435761u);
    uint32_t* d = sycl::malloc_device<uint32_t>(c.n, c.q);
    c.q.memcpy(d, init.data(), c.n * 4).wait();
    IntKernel<V, P> k{d, c.iters};
    c.q.parallel_for(sycl::nd_range<1>{c.n, 256}, k).wait();
    out.resize(c.n);
    c.q.memcpy(out.data(), d, c.n * 4).wait();
    const double ms = time_kernel(c.q, c.n, k, c.reps);
    sycl::free(d, c.q);
    return ms;
}

template <uint32_t P>
static void run_int_p(Ctx& c) {
    std::vector<uint32_t> a, b, r;
    const double tm = run_int<P, kMasked>(c, a), tl = run_int<P, kLockstep>(c, b), tr = run_int<P, kRedux>(c, r);
    size_t bad = 0;
    for (size_t i = 0; i < c.n; ++i) bad += (a[i] != b[i]) + (a[i] != r[i]);
    std::printf("| %2u | u32 | reduce plus | %8.4f | %8.4f | %8.4f | %5.2f | mismatches %zu |\n", P, tm, tl, tr,
                tr / tm, bad);
}

int main(int argc, char** argv) {
    const int lg = argc > 1 ? std::atoi(argv[1]) : 24;
    Ctx c{sycl::queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}}, size_t(1) << lg,
          argc > 2 ? std::atoi(argv[2]) : 64, argc > 3 ? std::atoi(argv[3]) : 30, nullptr};
    std::printf("# device: %s  lanes=2^%d iters=%d reps=%d (median ms)\n",
                c.q.get_device().get_info<sycl::info::device::name>().c_str(), lg, c.iters, c.reps);
    std::vector<int> trips(c.n);
    for (size_t i = 0; i < c.n; ++i) trips[i] = c.iters - 8 * int(i % 4);
    c.trips = sycl::malloc_device<int>(c.n, c.q);
    c.q.memcpy(c.trips, trips.data(), c.n * sizeof(int)).wait();

    std::printf("| P | type | op | masked | lockstep | native | emulated | native/masked | lockstep/masked | "
                "emulated/masked | max|diff| |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    run_p<4, float>(c);
    run_p<8, float>(c);
    run_p<16, float>(c);
    run_p<32, float>(c);
    run_p<4, double>(c);
    run_p<8, double>(c);
    run_p<16, double>(c);
    run_p<32, double>(c);
    std::printf("\n| P | type | op | masked | lockstep | redux.sync(chunk mask) | redux/masked | check |\n|---|---|---|---|---|---|---|---|\n");
    run_int_p<4>(c);
    run_int_p<8>(c);
    run_int_p<16>(c);
    run_int_p<32>(c);
    sycl::free(c.trips, c.q);
    return 0;
}
