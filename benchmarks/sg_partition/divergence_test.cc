// Standalone chunk-divergence test for the sub-group partition layer.
//
//   clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_89 -O3 -std=c++20 \
//       -I src/extensions -I build/include benchmarks/sg_partition/divergence_test.cc -o sgp_divergence_test
//   ONEAPI_DEVICE_SELECTOR=cuda:1 ./sgp_divergence_test
//
// Every chunk runs its own data-dependent trip count (some chunks return before
// the loop, some break out of it early) and inside the loop uses every
// collective: select, xor, shift left/right, reduce, inclusive scan, ballot,
// any/all, a local-memory exchange across group_barrier, a collective under a
// chunk-uniform branch, and collectives in both arms of an if/else that splits
// the chunks of a warp. The host replays each chunk.
// Values stay integer-valued, so floating-point sums are exact in any order.
//
// Masked partitions run the divergent kernel; lockstep partitions run the same
// body on converged control flow only (fixed trip count, no early exit, no
// chunk-dependent branch), which is all a lockstep partition promises.
// Both run on 1-D work-groups of 64 and 96 and a 2-D work-group of 4 x 16.

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "sg_partition/sg_partition.hh"
#include "../../src/sycl/kernel_attrs.hh"

namespace {

template <typename T>
struct is_cplx : std::false_type {};
template <typename R>
struct is_cplx<std::complex<R>> : std::true_type {};

template <typename T>
double re(const T& v) {
    if constexpr (is_cplx<T>::value) return static_cast<double>(v.real());
    else return static_cast<double>(v);
}

template <typename T>
T norm1000(const T& v) {
    if constexpr (is_cplx<T>::value) {
        using R = typename T::value_type;
        return T(norm1000<R>(v.real()), norm1000<R>(v.imag()));
    } else if constexpr (std::is_integral_v<T>) {
        return ((v % 1000) + 1000) % 1000;
    } else {
        T r = sycl::fmod(v, T(1000));
        return r < T(0) ? r + T(1000) : r;
    }
}

inline uint32_t hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

template <typename T>
T init_value(uint32_t L) {
    const uint32_t h = hash(L * 2654435761u + 17u);
    if constexpr (is_cplx<T>::value) {
        using R = typename T::value_type;
        return T(R(h % 1000u), R((h >> 10) % 1000u));
    } else {
        return T(h % 1000u);
    }
}

// Chunk trip count: -1 = leave before the loop.
inline int trips_of(uint32_t chunk, bool divergent, int fixed) {
    return divergent ? static_cast<int>(hash(chunk + 99u) % 10u) - 1 : fixed;
}

inline uint32_t log2u(uint32_t p) {
    uint32_t l = 0;
    while ((1u << l) < p) ++l;
    return l;
}

// One chunk's step, shared by host and device through the Coll interface.
template <uint32_t P, typename T, typename Coll>
T step(Coll& c, T v, int k, bool divergent, bool& stop) {
    const uint32_t lid = c.lid();
    const uint32_t m = P > 1 ? (1u << (static_cast<uint32_t>(k) % log2u(P))) : 0u;
    const T x = c.select(v, (static_cast<uint32_t>(k) + 1u) % P);
    const T y = c.xr(v, m);
    const T s0 = c.shl(v);
    const T s = lid + 1u < P ? s0 : v;
    const T u0 = c.shr(v);
    const T u = lid >= 1u ? u0 : v;
    const T r = c.reduce(v);
    const T sc = c.scan(v);
    const bool pred = (static_cast<int64_t>(re(v)) + k) % 3 == 0;
    const uint32_t b = c.ballot(pred);
    const bool any = c.any(pred), all = c.all(pred);
    const T nb = c.exchange(v);
    T e = T(0);
    if (divergent && any && !all) e = c.select(v, P - 1u);  // collective under a chunk-uniform branch
    // Chunks of one warp split across if/else arms that both hold a collective
    // (the second pair identical, the shape a compiler may merge or hoist).
    T w = T(0), w2 = T(0);
    if (divergent) {
        if (c.cid() & 1u) w = c.select(v, 0u);
        else w = c.select(v, P - 1u);
        if (c.cid() & 2u) w2 = c.xr(v, m);
        else w2 = c.xr(v, m);
    }
    const T out = norm1000<T>(x + y + s + u + r + sc + T(static_cast<int>(__builtin_popcount(b))) +
                              T(all ? 7 : 0) + nb + e + w + w2 + T(k));
    stop = divergent && static_cast<int64_t>(re(r)) % 5 == 0;
    return out;
}

// Device collectives on a SubGroupPartition.
template <uint32_t P, bool Masked, typename T, typename Local>
struct DevColl {
    batchlas::SubGroupPartition<P, Masked> part;
    Local slot;  // this sub-group's P-aligned local-memory window base
    uint32_t sg_off;

    uint32_t lid() const { return part.get_local_linear_id(); }
    uint32_t cid() const { return part.get_group_linear_id(); }
    T select(T v, uint32_t s) { return batchlas::select_from_group(part, v, s); }
    T xr(T v, uint32_t m) { return batchlas::permute_group_by_xor(part, v, m); }
    T shl(T v) { return batchlas::shift_group_left(part, v, 1u); }
    T shr(T v) { return batchlas::shift_group_right(part, v, 1u); }
    T reduce(T v) { return batchlas::reduce_over_group(part, v, sycl::plus<T>()); }
    T scan(T v) { return batchlas::inclusive_scan_over_group(part, v, sycl::plus<T>()); }
    uint32_t ballot(bool p) { return batchlas::ballot(part, p); }
    bool any(bool p) { return batchlas::any_of_group(part, p); }
    bool all(bool p) { return batchlas::all_of_group(part, p); }
    T exchange(T v) {
        const uint32_t me = sg_off + part.base + lid();
        slot[me] = v;
        batchlas::group_barrier(part);
        const T got = slot[sg_off + part.base + (lid() + 1u) % P];
        batchlas::group_barrier(part);  // before the next write to the slot
        return got;
    }
};

// Host replay of one chunk: every call is made for all P lanes at once.
template <uint32_t P, typename T>
struct HostChunk {
    std::vector<T> vals;  // current input of each lane, set before a lane's step
    uint32_t cur = 0;
    uint32_t chunk_in_sg = 0;  // work-groups are multiples of 32, so (L % 32) / P
    // The host replays lane by lane, so each collective reads `vals` (the
    // values every lane passed in) rather than a live exchange.
    uint32_t lid() const { return cur; }
    uint32_t cid() const { return chunk_in_sg; }
    T select(T, uint32_t s) { return vals[s]; }
    T xr(T, uint32_t m) { return vals[cur ^ m]; }
    T shl(T v) { return cur + 1 < P ? vals[cur + 1] : v; }
    T shr(T v) { return cur >= 1 ? vals[cur - 1] : v; }
    T reduce(T) {
        T s = T(0);
        for (auto& x : vals) s += x;
        return s;
    }
    T scan(T) {
        T s = T(0);
        for (uint32_t i = 0; i <= cur; ++i) s += vals[i];
        return s;
    }
    std::vector<int> preds;
    uint32_t ballot(bool) {
        uint32_t b = 0;
        for (uint32_t i = 0; i < P; ++i) b |= preds[i] ? (1u << i) : 0u;
        return b;
    }
    bool any(bool p) { return ballot(p) != 0; }
    bool all(bool p) { return ballot(p) == (P >= 32 ? ~0u : ((1u << P) - 1u)); }
    T exchange(T) { return vals[(cur + 1) % P]; }
};

template <uint32_t P, bool Masked, typename T, int Dims>
struct TestKernel;

template <uint32_t P, bool Masked, typename T, int Dims>
int run_case(sycl::queue& q, bool divergent, sycl::range<Dims> local, const char* tname) {
    constexpr int kFixed = 6;
    const size_t wg = local.size();
    const size_t n_wg = 64;
    const size_t n = wg * n_wg;
    sycl::range<Dims> global = local;
    global[0] *= n_wg;

    T* out = sycl::malloc_shared<T>(n, q);
    uint32_t* lane_of = sycl::malloc_shared<uint32_t>(n, q);
    for (size_t i = 0; i < n; ++i) out[i] = T(-1), lane_of[i] = ~0u;

    q.submit([&](sycl::handler& h) {
         sycl::local_accessor<T, 1> slot(sycl::range<1>(wg + 32), h);
         h.parallel_for<TestKernel<P, Masked, T, Dims>>(
             sycl::nd_range<Dims>(global, local),
             [=](sycl::nd_item<Dims> it) BATCHLAS_REQD_SG_SIZE(32) {
                 auto sg = it.get_sub_group();
                 const uint32_t sgl = static_cast<uint32_t>(sg.get_local_linear_id());
                 const uint32_t sgi = static_cast<uint32_t>(sg.get_group_linear_id());
                 const uint32_t sgr = static_cast<uint32_t>(sg.get_max_local_range()[0]);
                 const uint32_t wgi = static_cast<uint32_t>(it.get_group_linear_id());
                 // Logical lane: independent of how the work-group linearises.
                 const uint32_t L = wgi * static_cast<uint32_t>(it.get_local_range().size()) + sgi * sgr + sgl;
                 lane_of[L] = L;
                 T v = init_value<T>(L);
                 const int trips = trips_of(L / P, divergent, kFixed);
                 if (trips < 0) {
                     out[L] = v;
                     return;
                 }
                 DevColl<P, Masked, T, decltype(slot)> c{batchlas::make_partition<P, Masked>(sg), slot, sgi * sgr};
                 for (int k = 0; k < trips; ++k) {
                     bool stop = false;
                     v = step<P, T>(c, v, k, divergent, stop);
                     if (stop) break;
                 }
                 out[L] = v;
             });
     }).wait();

    int bad = 0;
    for (size_t ch = 0; ch < n / P; ++ch) {
        HostChunk<P, T> hc;
        hc.vals.resize(P);
        hc.preds.resize(P);
        hc.chunk_in_sg = static_cast<uint32_t>((ch * P) % 32u) / P;
        for (uint32_t i = 0; i < P; ++i) hc.vals[i] = init_value<T>(static_cast<uint32_t>(ch * P + i));
        const int trips = trips_of(static_cast<uint32_t>(ch), divergent, kFixed);
        for (int k = 0; k < trips; ++k) {
            std::vector<T> next(P);
            bool stop = false;
            for (uint32_t i = 0; i < P; ++i)
                hc.preds[i] = (static_cast<int64_t>(re(hc.vals[i])) + k) % 3 == 0;
            for (uint32_t i = 0; i < P; ++i) {
                hc.cur = i;
                next[i] = step<P, T>(hc, hc.vals[i], k, divergent, stop);
            }
            hc.vals = next;
            if (stop) break;
        }
        for (uint32_t i = 0; i < P; ++i) {
            const size_t L = ch * P + i;
            if (lane_of[L] != L || !(out[L] == hc.vals[i])) {
                if (bad < 3)
                    std::printf("  MISMATCH P=%u %s L=%zu got %g want %g\n", P, tname, L, re(out[L]), re(hc.vals[i]));
                ++bad;
            }
        }
    }
    std::printf("%-8s P=%2u %-9s %-10s local=%zu%s : %s (%d bad)\n", batchlas::SubGroupPartition<P, Masked>::kMasked ? "masked" : "lockstep",
                P, tname, divergent ? "divergent" : "converged", wg, Dims == 2 ? " (2-D)" : "", bad ? "FAIL" : "ok", bad);
    sycl::free(out, q);
    sycl::free(lane_of, q);
    return bad ? 1 : 0;
}

template <uint32_t P, typename T>
int run_p(sycl::queue& q, const char* tname) {
    int f = 0;
    f += run_case<P, true, T, 1>(q, true, sycl::range<1>(64), tname);
    f += run_case<P, true, T, 1>(q, true, sycl::range<1>(96), tname);
    f += run_case<P, true, T, 2>(q, true, sycl::range<2>(4, 16), tname);
    f += run_case<P, true, T, 1>(q, false, sycl::range<1>(64), tname);
    f += run_case<P, false, T, 1>(q, false, sycl::range<1>(64), tname);
    f += run_case<P, false, T, 2>(q, false, sycl::range<2>(4, 16), tname);
    return f;
}

template <typename T>
int run_t(sycl::queue& q, const char* tname) {
    return run_p<1, T>(q, tname) + run_p<2, T>(q, tname) + run_p<4, T>(q, tname) + run_p<8, T>(q, tname) +
           run_p<16, T>(q, tname) + run_p<32, T>(q, tname);
}

} // namespace

int main() {
    // Unbuffered: a broken masked path deadlocks, and the last line printed
    // names the case that hung.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("# device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0;
    fails += run_t<float>(q, "float");
    fails += run_t<double>(q, "double");
    fails += run_t<int32_t>(q, "int");
    fails += run_t<std::complex<double>>(q, "cdouble");
    std::printf("%s: %d failing cases\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
