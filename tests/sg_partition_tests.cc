// Correctness suite for the chunked sub-group partition layer
// (src/extensions/sg_partition/): SubGroupPartition<P, Masked> and every
// collective of its front end, on whatever backend the device pass selects.
//
// Every kernel launches kNumWg work-groups of kSgPerWg sub-groups, so a
// cross-sub-group or cross-work-group mix-up shows up as a wrong value. Each
// result is checked against a host model; values carry their chunk in their
// seed, so a lane that read another chunk's data gets a wrong answer.
//
// Matrix: sub-group size SG in {8, 16, 32, 64} (forced with
// reqd_sub_group_size; a size the device lacks is skipped), P in {1..SG}
// powers of two, Masked in {true, false}, and ten value types. SG = 32 runs
// every type with every op a backend may map to its own instruction; the
// other sizes run a representative subset.
//
// Kernel count is the budget: the OpenCL CPU JIT costs ~0.1 s per kernel, so
// ops share kernels wherever they can (one kernel per type and config).
//
// Masked partitions additionally get divergence tests: chunks of one
// sub-group with different trip counts, early returns, and different
// branches, all running collectives.
//
// This binary links no BatchLAS library: it compiles the header layer only.

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <batchlas/util/group-invoke.hh>

#include "../src/extensions/sg_partition/sg_partition.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// The SG != 32 kernels are compiled for NVPTX too, with empty bodies (see
// launch()); the attribute there is meaningless but harmless.
#pragma clang diagnostic ignored "-Wincorrect-sub-group-size"

namespace {

using batchlas::SubGroupPartition;

constexpr size_t kSgPerWg = 4;
constexpr size_t kNumWg = 6;
constexpr size_t kLocalWords = 4;  // uint64 words of local memory per work-item

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

struct S12 {
    uint32_t a, b, c;
    friend bool operator==(const S12& x, const S12& y) { return x.a == y.a && x.b == y.b && x.c == y.c; }
    friend std::ostream& operator<<(std::ostream& os, const S12& s) {
        return os << "{" << s.a << "," << s.b << "," << s.c << "}";
    }
};
static_assert(sizeof(S12) == 12);

// Commutative: component-wise wrapping add.
struct S12Add {
    S12 operator()(const S12& x, const S12& y) const { return {x.a + y.a, x.b + y.b, x.c + y.c}; }
};

// Associative, NOT commutative: composition of affine maps t -> a*t + b,
// plus a "leftmost operand" tag in c. A scan that swaps operands fails.
struct S12Affine {
    S12 operator()(const S12& x, const S12& y) const { return {x.a * y.a, x.b * y.a + y.b, x.c}; }
};

inline uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline uint32_t seed(uint32_t item, uint32_t salt) { return mix(item * 0x9e3779b9u + salt * 0x85ebca6bu + 1u); }

// Integer-valued floating point keeps every sum and order of summation exact.
template <typename T>
inline T gen(uint32_t s) {
    const uint32_t h = mix(s);
    if constexpr (std::is_same_v<T, bool>) {
        return (h & 1u) != 0u;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<uint8_t>(h);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<int32_t>(h % 2001u) - 1000;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return h;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return (static_cast<int64_t>(h) << 20) - static_cast<int64_t>(mix(s + 7u));
    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return static_cast<T>(static_cast<int32_t>(h % 2001u) - 1000);
    } else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        using R = typename T::value_type;
        return T(gen<R>(s), gen<R>(s ^ 0x5bd1e995u));
    } else {
        static_assert(std::is_same_v<T, S12>);
        return S12{h, mix(s + 1u), mix(s + 2u)};
    }
}

template <typename T>
std::string show(const T& v) {
    std::ostringstream os;
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
        os << static_cast<int>(v);
    } else {
        os << v;
    }
    return os.str();
}

template <typename T>
bool same(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0 || a == b;
}

// Collects mismatches; reports the first few individually and a total.
struct Checker {
    std::string ctx;
    size_t bad = 0, checked = 0;
    template <typename T>
    void eq(const T& got, const T& want, const std::string& what, size_t item) {
        ++checked;
        if (same(got, want)) return;
        if (bad++ < 6)
            ADD_FAILURE() << ctx << " " << what << " item " << item << ": got " << show(got) << " want "
                          << show(want);
    }
    ~Checker() {
        if (bad) ADD_FAILURE() << ctx << ": " << bad << " of " << checked << " values wrong";
    }
};

template <size_t SG, size_t P, bool M>
std::string cfg(const char* type) {
    std::ostringstream os;
    os << "[SG=" << SG << " P=" << P << (M ? " masked" : " lockstep") << " T=" << type << "]";
    return os.str();
}

template <typename T> const char* tname() {
    if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
    else if constexpr (std::is_same_v<T, int32_t>) return "int32";
    else if constexpr (std::is_same_v<T, uint32_t>) return "uint32";
    else if constexpr (std::is_same_v<T, int64_t>) return "int64";
    else if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "cfloat";
    else if constexpr (std::is_same_v<T, std::complex<double>>) return "cdouble";
    else return "S12";
}

// ---------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------

sycl::queue& queue() {
    static sycl::queue q{sycl::default_selector_v};
    return q;
}

bool supports_sg(size_t sg) {
    const auto sizes = queue().get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sizes.begin(), sizes.end(), sg) != sizes.end();
}

bool supports_fp64() { return queue().get_device().has(sycl::aspect::fp64); }

template <size_t SG>
constexpr size_t kItems = SG * kSgPerWg * kNumWg;

// Device array of n Ts, downloaded on request.
template <typename T>
struct DevArr {
    T* d = nullptr;
    size_t n = 0;
    explicit DevArr(size_t n_) : n(n_) {
        d = sycl::malloc_device<T>(n, queue());
        queue().memset(d, 0, n * sizeof(T)).wait();
    }
    ~DevArr() { sycl::free(d, queue()); }
    DevArr(const DevArr&) = delete;
    std::unique_ptr<T[]> get() const {
        std::unique_ptr<T[]> h(new T[n]);
        queue().memcpy(h.get(), d, n * sizeof(T)).wait();
        return h;
    }
};

// Launches f(item, local_mem) over kItems<SG> work-items with the sub-group
// size forced to SG. Returns false (nothing launched) if the device lacks SG.
// The NVPTX pass compiles only SG = 32 bodies: CUDA has no other size, and
// empty bodies keep the build fast. The empty body still has to capture what
// the host pass captures, or the two passes disagree on the lambda's layout.
template <typename... A>
inline void keep_captured(const A&...) {}

template <size_t SG, typename F>
bool launch(F f) {
    if (!supports_sg(SG)) return false;
    constexpr size_t wg = SG * kSgPerWg;
    queue()
        .submit([&](sycl::handler& h) {
            sycl::local_accessor<uint64_t, 1> lm(sycl::range<1>(wg * kLocalWords), h);
            h.parallel_for(sycl::nd_range<1>(kItems<SG>, wg),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
                               if constexpr (SG != 32) {
                                   keep_captured(f, lm);
                                   return;
                               } else
#endif
                               {
                                   f(it, lm.template get_multi_ptr<sycl::access::decorated::no>().get());
                               }
                           });
        })
        .wait_and_throw();
    return true;
}

// P filters. Word handling does not depend on P and P handling does not
// depend on the type, so a few types sweep every P and the rest take the
// edges: the trivial chunk, a small chunk, and the 32- and SG-wide ones.
constexpr bool every_p(size_t) { return true; }
constexpr bool edge_p(size_t P) { return P == 1 || P == 4 || P == 32 || P == 64; }

template <typename T>
inline constexpr bool sweeps_every_p_v =
    std::is_same_v<T, uint32_t> || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, bool>;

// Calls f(integral_constant<P>, bool_constant<Masked>) for every P <= SG that
// Keep accepts (P == SG is always kept).
template <size_t SG, auto Keep = every_p, typename F>
void for_each_config(F&& f) {
    [&]<size_t... L>(std::index_sequence<L...>) {
        (
            [&] {
                constexpr size_t P = size_t{1} << L;
                if constexpr (P <= SG && (P == SG || Keep(P))) {
                    f(std::integral_constant<size_t, P>{}, std::true_type{});
                    f(std::integral_constant<size_t, P>{}, std::false_type{});
                }
            }(),
            ...);
    }(std::make_index_sequence<7>{});
}

template <typename T>
bool type_runnable() {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) return supports_fp64();
    return true;
}

// ---------------------------------------------------------------------------
// Partition identity, votes, lockstep hooks, invoke_one, and group_barrier
// ordering a local-memory exchange: one kernel per (SG, P, Masked).
// ---------------------------------------------------------------------------

// Five predicate shapes by chunk: random, all, none, one lane, all but one
// lane. The odd lane moves through the chunk, so with P = 64 it also lands in
// the upper 32 lanes.
inline bool vote_pred(uint32_t P, uint32_t chunk, uint32_t lid) {
    const uint32_t mode = chunk % 5u, odd = P - 1u - (chunk / 5u) % P;
    switch (mode) {
    case 0: return (mix(chunk * P + lid) & 1u) != 0u;
    case 1: return true;
    case 2: return false;
    case 3: return lid == odd;
    default: return lid != odd;
    }
}

constexpr size_t kMiscSlots = 12;
constexpr uint32_t kBarrierRounds = 3;

template <size_t SG, size_t P, bool M>
bool run_misc() {
    static_assert(sizeof(S12) <= kLocalWords * 8);
    constexpr size_t N = kItems<SG>, K = kMiscSlots, R = kBarrierRounds;
    DevArr<uint32_t> out(N * K), calls(N / P), xu(N * R);
    DevArr<S12> xs(N * R);
    uint32_t *o = out.d, *cl = calls.d, *xo = xu.d;
    S12* so = xs.d;
    const bool ran = launch<SG>([=](sycl::nd_item<1> it, uint64_t* lm) {
        const auto sg = it.get_sub_group();
        const auto part = batchlas::make_partition<P, M>(sg);
        const uint32_t gid = static_cast<uint32_t>(it.get_global_linear_id());
        const uint32_t wl = static_cast<uint32_t>(it.get_local_linear_id());
        const uint32_t chunk = gid / P;
        const uint32_t lid = part.get_local_linear_id();
        uint32_t* r = o + size_t(gid) * K;
        r[0] = lid;
        r[1] = part.get_local_linear_range();
        r[2] = part.get_group_linear_id();
        r[3] = part.get_group_linear_range();
        r[4] = part.leader() ? 1u : 0u;
        r[5] = static_cast<uint32_t>(part.get_local_id()[0] + 100u * part.get_local_range()[0]);
        const bool pred = vote_pred(P, chunk, lid);
        if constexpr (P <= 32) r[6] = ballot(part, pred);
        r[7] = any_of_group(part, pred);
        r[8] = all_of_group(part, pred);
        r[9] = none_of_group(part, pred);
        // Chunk-uniform inputs, as the lockstep hooks require.
        r[10] = batchlas::lockstep_any(part, mix(chunk) % 3u == 0u);
        r[11] = static_cast<uint32_t>(batchlas::lockstep_max(part, static_cast<int32_t>(mix(chunk) % 1000u) - 500));
        batchlas::invoke_one(part, [&] {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> a(cl[chunk]);
            a.fetch_add(gid + 1u);
        });
        // Several rounds, so a missing write-after-read barrier also shows.
        // Disjoint regions: bytes [0, 4 wg) and [8 wg, 20 wg) of the 32 wg.
        const size_t wg = it.get_local_range(0);
        uint32_t* su = reinterpret_cast<uint32_t*>(lm);
        S12* ss = reinterpret_cast<S12*>(lm + wg);
        for (uint32_t k = 0; k < R; ++k) {
            const uint32_t src = wl - lid + (lid + k + 1u) % uint32_t(P);
            su[wl] = gen<uint32_t>(seed(gid, 10 + k));
            group_barrier(part);
            xo[size_t(gid) * R + k] = su[src];
            group_barrier(part);
            ss[wl] = gen<S12>(seed(gid, 20 + k));
            group_barrier(part);
            so[size_t(gid) * R + k] = ss[src];
            group_barrier(part);
        }
    });
    if (!ran) return false;
    const auto h = out.get();
    const auto hc = calls.get();
    const auto hx = xu.get();
    const auto hs = xs.get();
    Checker c{cfg<SG, P, M>("-")};
    for (uint32_t gid = 0; gid < N; ++gid) {
        const uint32_t lane = gid % SG, lid = gid % P, chunk = gid / P, sgid = gid / SG, base = gid - lid;
        const uint32_t* r = h.get() + size_t(gid) * K;
        c.eq(r[0], lid, "get_local_linear_id", gid);
        c.eq(r[1], uint32_t(P), "get_local_linear_range", gid);
        c.eq(r[2], lane / uint32_t(P), "get_group_linear_id", gid);
        c.eq(r[3], uint32_t(SG / P), "get_group_linear_range", gid);
        c.eq(r[4], uint32_t(lid == 0), "leader", gid);
        c.eq(r[5], uint32_t(lid + 100u * P), "get_local_id/get_local_range", gid);
        uint32_t bal = 0;
        bool any = false, all = true;
        for (uint32_t j = 0; j < P; ++j) {
            const bool p = vote_pred(P, chunk, j);
            if (p && j < 32) bal |= 1u << j;
            any |= p;
            all &= p;
        }
        if (P <= 32) c.eq(r[6], bal, "ballot", gid);
        c.eq(r[7], uint32_t(any), "any_of_group", gid);
        c.eq(r[8], uint32_t(all), "all_of_group", gid);
        c.eq(r[9], uint32_t(!any), "none_of_group", gid);
        // Lockstep hooks: own chunk when masked, whole sub-group when lockstep.
        const uint32_t per_sg = uint32_t(SG / P);
        const uint32_t lo = M ? chunk : sgid * per_sg, hi = M ? chunk + 1 : (sgid + 1) * per_sg;
        bool la = false;
        int32_t lmax = std::numeric_limits<int32_t>::min();
        for (uint32_t ch = lo; ch < hi; ++ch) {
            la |= mix(ch) % 3u == 0u;
            lmax = std::max(lmax, static_cast<int32_t>(mix(ch) % 1000u) - 500);
        }
        c.eq(r[10], uint32_t(la), "lockstep_any", gid);
        c.eq(r[11], static_cast<uint32_t>(lmax), "lockstep_max", gid);
        if (lid == 0) c.eq(hc[chunk], base + 1u, "invoke_one: called once, by the leader", gid);
        for (uint32_t k = 0; k < R; ++k) {
            const uint32_t src = base + (lid + k + 1u) % uint32_t(P);
            c.eq(hx[size_t(gid) * R + k], gen<uint32_t>(seed(src, 10 + k)), "barrier exchange uint32", gid);
            c.eq(hs[size_t(gid) * R + k], gen<S12>(seed(src, 20 + k)), "barrier exchange S12", gid);
        }
    }
    return true;
}

template <size_t SG>
void misc_all() {
    bool ran = true;
    for_each_config<SG, (SG == 64 ? &edge_p : &every_p)>([&](auto p, auto m) { ran &= run_misc<SG, decltype(p)::value, decltype(m)::value>(); });
    if (!ran) GTEST_SKIP() << "device lacks sub-group size " << SG;
}

TEST(SgPartition, IdentityVotesBarrierSG8) { misc_all<8>(); }
TEST(SgPartition, IdentityVotesBarrierSG16) { misc_all<16>(); }
TEST(SgPartition, IdentityVotesBarrierSG32) { misc_all<32>(); }
TEST(SgPartition, IdentityVotesBarrierSG64) { misc_all<64>(); }

// ---------------------------------------------------------------------------
// Per value type: data movement (select, xor with every mask, shifts by every
// delta, broadcasts, the group-invoke.hh helpers) and reductions/scans.
// ---------------------------------------------------------------------------

// One reduction case: reduce, inclusive scan, exclusive scan from `init`.
// Non-commutative ops only check the scans: reduce is a GENERALIZED_SUM (any
// order), scans are GENERALIZED_NONCOMMUTATIVE_SUMs.
template <typename T, typename Op>
struct Case {
    Op op;
    T init;
    const char* name;
    bool commutative;
};

template <typename T, typename Op>
Case<T, Op> make_case(Op op, const char* name, bool commutative = true) {
    return Case<T, Op>{op, gen<T>(0xabcdefu), name, commutative};
}

// The op every type gets at every (SG, P, Masked).
template <typename T>
auto core_cases() {
    if constexpr (std::is_same_v<T, bool>) {
        return std::make_tuple(make_case<T>(sycl::bit_or<T>(), "bit_or"));
    } else if constexpr (std::is_same_v<T, S12>) {
        return std::make_tuple(make_case<T>(S12Add{}, "S12Add"),
                               make_case<T>(S12Affine{}, "S12Affine(non-commutative)", false));
    } else {
        return std::make_tuple(make_case<T>(sycl::plus<T>(), "plus"));
    }
}

// Every further (type, op) a backend may map to its own instruction; SG = 32.
template <typename T>
auto extra_cases() {
    if constexpr (std::is_same_v<T, bool>) {
        return std::make_tuple(make_case<T>(sycl::bit_and<T>(), "bit_and"));
    } else if constexpr (std::is_same_v<T, S12>) {
        return std::tuple<>();
    } else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return std::make_tuple(make_case<T>(sycl::plus<>(), "plus<void>"));
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return std::make_tuple(make_case<T>(sycl::minimum<T>(), "minimum"), make_case<T>(sycl::maximum<T>(), "maximum"),
                               make_case<T>(sycl::bit_or<T>(), "bit_or"), make_case<T>(sycl::bit_and<T>(), "bit_and"),
                               make_case<T>(sycl::bit_xor<T>(), "bit_xor"), make_case<T>(sycl::plus<>(), "plus<void>"),
                               make_case<T>(sycl::maximum<>(), "maximum<void>"));
    } else if constexpr (std::is_same_v<T, uint32_t>) {  // multiplies wraps defined only unsigned
        return std::make_tuple(make_case<T>(sycl::minimum<T>(), "minimum"), make_case<T>(sycl::maximum<T>(), "maximum"),
                               make_case<T>(sycl::bit_and<T>(), "bit_and"), make_case<T>(sycl::bit_xor<T>(), "bit_xor"),
                               make_case<T>(sycl::multiplies<T>(), "multiplies"));
    } else if constexpr (std::is_integral_v<T>) {  // int64, uint8
        return std::make_tuple(make_case<T>(sycl::minimum<T>(), "minimum"), make_case<T>(sycl::maximum<T>(), "maximum"),
                               make_case<T>(sycl::bit_or<T>(), "bit_or"));
    } else {  // float, double
        return std::make_tuple(make_case<T>(sycl::minimum<T>(), "minimum"), make_case<T>(sycl::maximum<T>(), "maximum"),
                               make_case<T>(sycl::plus<>(), "plus<void>"));
    }
}

template <size_t P>
constexpr size_t kMoveSlots = 6 + 3 * (P - 1);

// Core cases check reduce, inclusive scan and exclusive scan; the extra cases
// (Move == false) skip the exclusive scan, which only composes the other two.
template <size_t SG, size_t P, bool M, typename T, bool Move, typename... Cs>
bool run_ops(const std::tuple<Cs...>& cases) {
    constexpr size_t S = Move ? 3 : 2;
    constexpr size_t N = kItems<SG>, NM = Move ? kMoveSlots<P> : 0, K = NM + S * sizeof...(Cs);
    if constexpr (K == 0) {
        return true;
    } else {
        DevArr<T> out(N * K);
        T* o = out.d;
        const bool ran = launch<SG>([=](sycl::nd_item<1> it, uint64_t*) {
            const auto part = batchlas::make_partition<P, M>(it.get_sub_group());
            const uint32_t gid = static_cast<uint32_t>(it.get_global_linear_id());
            const uint32_t lid = part.get_local_linear_id();
            const T x = gen<T>(seed(gid, 1));
            T* r = o + size_t(gid) * K;
            if constexpr (Move) {
                r[0] = select_from_group(part, x, (lid * 3u + 1u) & uint32_t(P - 1));
                r[1] = group_broadcast(part, x, uint32_t(P / 2));
                r[2] = group_broadcast(part, x);
                r[3] = sg_leader_broadcast(part, x);
                r[4] = batchlas::broadcast_from_leader(part, x);
                r[5] = batchlas::invoke_one_broadcast(part, [&] { return gen<T>(seed(gid, 2)); });
                for (uint32_t d = 1; d < P; ++d) {
                    r[5 + d] = permute_group_by_xor(part, x, d);
                    r[5 + (P - 1) + d] = shift_group_left(part, x, d);
                    r[5 + 2 * (P - 1) + d] = shift_group_right(part, x, d);
                }
            }
            std::apply(
                [&](const auto&... cs) {
                    size_t k = NM;
                    auto one = [&](const auto& cs1) {
                        r[k] = reduce_over_group(part, x, cs1.op);
                        r[k + 1] = inclusive_scan_over_group(part, x, cs1.op);
                        if constexpr (Move) r[k + 2] = exclusive_scan_over_group(part, x, cs1.init, cs1.op);
                        k += S;
                    };
                    (one(cs), ...);
                },
                cases);
        });
        if (!ran) return false;
        const auto h = out.get();
        Checker c{cfg<SG, P, M>(tname<T>())};
        for (uint32_t gid = 0; gid < N; ++gid) {
            const uint32_t lid = gid % P, base = gid - lid;
            auto X = [&](uint32_t l) { return gen<T>(seed(base + l, 1)); };
            const T* r = h.get() + size_t(gid) * K;
            if constexpr (Move) {
                c.eq(r[0], X((lid * 3u + 1u) & uint32_t(P - 1)), "select_from_group", gid);
                c.eq(r[1], X(uint32_t(P / 2)), "group_broadcast(P/2)", gid);
                c.eq(r[2], X(0), "group_broadcast()", gid);
                c.eq(r[3], X(0), "sg_leader_broadcast", gid);
                c.eq(r[4], X(0), "broadcast_from_leader", gid);
                c.eq(r[5], gen<T>(seed(base, 2)), "invoke_one_broadcast", gid);
                for (uint32_t d = 1; d < P; ++d) {
                    const std::string ds = std::to_string(d);
                    c.eq(r[5 + d], X(lid ^ d), "permute_group_by_xor " + ds, gid);
                    if (lid + d < P) c.eq(r[5 + (P - 1) + d], X(lid + d), "shift_group_left " + ds, gid);
                    if (lid >= d) c.eq(r[5 + 2 * (P - 1) + d], X(lid - d), "shift_group_right " + ds, gid);
                }
            }
            std::apply(
                [&](const auto&... cs) {
                    size_t k = NM;
                    auto one = [&](const auto& cs1) {
                        T red = X(0), inc = X(0), exc = cs1.init;
                        for (uint32_t j = 1; j < P; ++j) red = cs1.op(red, X(j));
                        for (uint32_t j = 1; j <= lid; ++j) inc = cs1.op(inc, X(j));
                        for (uint32_t j = 0; j < lid; ++j) exc = cs1.op(exc, X(j));
                        const std::string n = cs1.name;
                        if (cs1.commutative) c.eq(r[k], red, "reduce_over_group " + n, gid);
                        c.eq(r[k + 1], inc, "inclusive_scan_over_group " + n, gid);
                        if (Move) c.eq(r[k + 2], exc, "exclusive_scan_over_group " + n, gid);
                        k += S;
                    };
                    (one(cs), ...);
                },
                cases);
        }
        return true;
    }
}

template <size_t SG, typename T, bool Move, auto Keep, typename Cases>
void ops_all(const Cases& cases) {
    if (!type_runnable<T>()) GTEST_SKIP() << "device lacks fp64";
    bool ran = true;
    for_each_config<SG, Keep>(
        [&](auto p, auto m) { ran &= run_ops<SG, decltype(p)::value, decltype(m)::value, T, Move>(cases); });
    if (!ran) GTEST_SKIP() << "device lacks sub-group size " << SG;
}

template <typename T>
class SgPartitionTyped : public ::testing::Test {};
using AllTypes = ::testing::Types<int32_t, uint32_t, int64_t, float, double, std::complex<float>,
                                  std::complex<double>, S12, uint8_t, bool>;
TYPED_TEST_SUITE(SgPartitionTyped, AllTypes);

TYPED_TEST(SgPartitionTyped, MoveReduceScanSG32) {
    constexpr auto keep = sweeps_every_p_v<TypeParam> ? &every_p : &edge_p;
    ops_all<32, TypeParam, true, keep>(core_cases<TypeParam>());
}
// int32 has the most native mappings, so it sweeps every P.
TYPED_TEST(SgPartitionTyped, ExtraOpsSG32) {
    if constexpr (std::tuple_size_v<decltype(extra_cases<TypeParam>())> == 0) GTEST_SKIP() << "no extra ops";
    constexpr auto keep = std::is_same_v<TypeParam, int32_t> ? &every_p : &edge_p;
    ops_all<32, TypeParam, false, keep>(extra_cases<TypeParam>());
}

// The other sub-group sizes (Intel SIMD8/16, wave64): 1 and 4 words and sub-word.
template <typename T>
class SgPartitionCore : public ::testing::Test {};
using CoreTypes = ::testing::Types<uint32_t, std::complex<double>, bool>;
TYPED_TEST_SUITE(SgPartitionCore, CoreTypes);

TYPED_TEST(SgPartitionCore, MoveReduceScanSG8) {
    constexpr auto keep = std::is_same_v<TypeParam, uint32_t> ? &every_p : &edge_p;
    ops_all<8, TypeParam, true, keep>(core_cases<TypeParam>());
}
TYPED_TEST(SgPartitionCore, MoveReduceScanSG16) {
    constexpr auto keep = std::is_same_v<TypeParam, uint32_t> ? &every_p : &edge_p;
    ops_all<16, TypeParam, true, keep>(core_cases<TypeParam>());
}
TYPED_TEST(SgPartitionCore, MoveReduceScanSG64) { ops_all<64, TypeParam, true, &edge_p>(core_cases<TypeParam>()); }

// ---------------------------------------------------------------------------
// Divergence (Masked only): the chunks of one sub-group run different
// data-dependent trip counts, different branches, and some return at once,
// and every collective still sees exactly its own chunk.
// ---------------------------------------------------------------------------

constexpr uint32_t kMaxTrips = 6;
constexpr size_t kDivT = 9;   // T-valued results per iteration
constexpr size_t kDivU = 3;   // uint32 results per iteration

inline uint32_t div_trips(uint32_t chunk) { return mix(chunk * 7u + 1u) % kMaxTrips; }
inline bool div_early_exit(uint32_t chunk) { return mix(chunk ^ 0xdeadu) % 5u == 0u; }
inline bool div_pred(uint32_t chunk, uint32_t lid, uint32_t i) {
    return (mix(chunk * 131u + lid * 7u + i) % 3u) == 0u;
}
inline uint32_t div_lim(uint32_t chunk, uint32_t lid) { return mix(chunk * 17u + lid) % 5u; }

template <typename T>
struct AddFor {
    using type = sycl::plus<T>;
};
template <>
struct AddFor<S12> {
    using type = S12Add;
};

template <size_t SG, size_t P, typename T>
bool run_divergence() {
    constexpr size_t N = kItems<SG>;
    using Add = typename AddFor<T>::type;
    DevArr<T> outT(N * kMaxTrips * kDivT);
    DevArr<uint32_t> outU(N * kMaxTrips * kDivU);
    T* ot = outT.d;
    uint32_t* ou = outU.d;
    const bool ran = launch<SG>([=](sycl::nd_item<1> it, uint64_t* lm) {
        const auto part = batchlas::make_partition<P, true>(it.get_sub_group());
        const uint32_t gid = static_cast<uint32_t>(it.get_global_linear_id());
        const uint32_t wl = static_cast<uint32_t>(it.get_local_linear_id());
        const uint32_t lid = part.get_local_linear_id();
        const uint32_t chunk = gid / P;
        if (div_early_exit(chunk)) return;
        T* slot = reinterpret_cast<T*>(lm);
        const uint32_t trips = div_trips(chunk);
        for (uint32_t i = 0; i < trips; ++i) {
            const T x = gen<T>(seed(gid, 100 + i));
            T* rt = ot + (size_t(gid) * kMaxTrips + i) * kDivT;
            uint32_t* ru = ou + (size_t(gid) * kMaxTrips + i) * kDivU;
            // Chunks take different branches, each with a different collective.
            if (chunk & 1u) {
                rt[0] = reduce_over_group(part, x, Add{});
            } else {
                rt[0] = select_from_group(part, x, (lid + i) % uint32_t(P));
            }
            if constexpr (P > 1) rt[1] = permute_group_by_xor(part, x, (i % uint32_t(P - 1)) + 1u);
            rt[2] = shift_group_left(part, x, 1u);
            rt[3] = shift_group_right(part, x, 1u);
            rt[4] = inclusive_scan_over_group(part, x, Add{});
            slot[wl] = x;
            group_barrier(part);
            rt[5] = slot[wl - lid + (lid + i + 1u) % uint32_t(P)];
            group_barrier(part);
            rt[6] = batchlas::invoke_one_broadcast(part, [&] { return x; });
            if (mix(chunk + i * 17u) & 1u) {
                rt[7] = batchlas::broadcast_from_leader(part, x);
            } else {
                rt[7] = group_broadcast(part, x, uint32_t(P - 1));
            }
            if constexpr (P <= 32) ru[0] = ballot(part, div_pred(chunk, lid, i));
            // A convergence loop: the chunk iterates while any lane still wants to.
            const uint32_t lim = div_lim(chunk, lid);
            uint32_t k = 0;
            T acc = x;
            while (any_of_group(part, k < lim)) {
                acc = Add{}(acc, select_from_group(part, x, k % uint32_t(P)));
                ++k;
            }
            rt[8] = acc;
            ru[1] = k;
            ru[2] = all_of_group(part, div_pred(chunk, lid, i) || (lid & 1u));
        }
    });
    if (!ran) return false;
    const auto ht = outT.get();
    const auto hu = outU.get();
    Checker c{cfg<SG, P, true>(tname<T>()) + " divergent"};
    for (uint32_t gid = 0; gid < N; ++gid) {
        const uint32_t lid = gid % P, base = gid - lid, chunk = gid / P;
        if (div_early_exit(chunk)) continue;
        for (uint32_t i = 0; i < div_trips(chunk); ++i) {
            auto X = [&](uint32_t l) { return gen<T>(seed(base + l, 100 + i)); };
            const T* rt = ht.get() + (size_t(gid) * kMaxTrips + i) * kDivT;
            const uint32_t* ru = hu.get() + (size_t(gid) * kMaxTrips + i) * kDivU;
            const std::string at = " iter " + std::to_string(i);
            if (chunk & 1u) {
                T s = X(0);
                for (uint32_t j = 1; j < P; ++j) s = Add{}(s, X(j));
                c.eq(rt[0], s, "reduce in branch" + at, gid);
            } else {
                c.eq(rt[0], X((lid + i) % uint32_t(P)), "select in branch" + at, gid);
            }
            if (P > 1) c.eq(rt[1], X(lid ^ ((i % uint32_t(P > 1 ? P - 1 : 1)) + 1u)), "xor" + at, gid);
            if (lid + 1 < P) c.eq(rt[2], X(lid + 1), "shift_left" + at, gid);
            if (lid >= 1) c.eq(rt[3], X(lid - 1), "shift_right" + at, gid);
            T inc = X(0);
            for (uint32_t j = 1; j <= lid; ++j) inc = Add{}(inc, X(j));
            c.eq(rt[4], inc, "inclusive_scan" + at, gid);
            c.eq(rt[5], X((lid + i + 1u) % uint32_t(P)), "barrier exchange" + at, gid);
            c.eq(rt[6], X(0), "invoke_one_broadcast" + at, gid);
            c.eq(rt[7], (mix(chunk + i * 17u) & 1u) ? X(0) : X(uint32_t(P - 1)), "branch broadcast" + at, gid);
            uint32_t bal = 0, K = 0;
            bool all = true;
            for (uint32_t j = 0; j < P; ++j) {
                if (div_pred(chunk, j, i) && j < 32) bal |= 1u << j;
                K = std::max(K, div_lim(chunk, j));
                all &= div_pred(chunk, j, i) || (j & 1u);
            }
            T acc = X(lid);
            for (uint32_t k = 0; k < K; ++k) acc = Add{}(acc, X(k % uint32_t(P)));
            if (P <= 32) c.eq(ru[0], bal, "ballot" + at, gid);
            c.eq(rt[8], acc, "any_of convergence loop value" + at, gid);
            c.eq(ru[1], K, "any_of convergence loop trips" + at, gid);
            c.eq(ru[2], uint32_t(all), "all_of" + at, gid);
        }
    }
    return true;
}

// Explicit isolation check: every lane's value is tagged with its chunk, and
// each collective result's tag must be the lane's own chunk.
template <size_t SG, size_t P>
bool run_isolation() {
    constexpr size_t N = kItems<SG>;
    DevArr<uint32_t> foreign(1), total(1);
    uint32_t *fo = foreign.d, *to = total.d;
    const bool ran = launch<SG>([=](sycl::nd_item<1> it, uint64_t*) {
        const auto part = batchlas::make_partition<P, true>(it.get_sub_group());
        const uint32_t gid = static_cast<uint32_t>(it.get_global_linear_id());
        const uint32_t lid = part.get_local_linear_id();
        const uint32_t chunk = gid / P;
        const uint32_t tag = chunk << 8;
        uint32_t bad = 0, n = 0;
        auto chk = [&](uint32_t v) { ++n; bad += (v & ~0xffu) != tag; };
        // Chunk c runs c % 5 iterations: neighbours are in different loop trips.
        for (uint32_t i = 0; i < chunk % 5u; ++i) {
            const uint32_t x = tag | ((lid + i) & 0xffu);
            chk(select_from_group(part, x, (lid + i) % uint32_t(P)));
            for (uint32_t m = 1; m < P; ++m) chk(permute_group_by_xor(part, x, m));
            chk(reduce_over_group(part, x, sycl::maximum<uint32_t>()));
            chk(reduce_over_group(part, x, sycl::minimum<uint32_t>()));
            chk(reduce_over_group(part, x, sycl::bit_or<uint32_t>()));
            chk(inclusive_scan_over_group(part, x, sycl::maximum<uint32_t>()));
            chk(sg_leader_broadcast(part, x));
            // Every lane shifts (a collective); only in-chunk sources are checked.
            const uint32_t sl = shift_group_left(part, x, 1u), sr = shift_group_right(part, x, 1u);
            if (lid + 1 < P) chk(sl);
            if (lid >= 1) chk(sr);
        }
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*fo).fetch_add(bad);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*to).fetch_add(n);
    });
    if (!ran) return false;
    const uint32_t f = foreign.get()[0], t = total.get()[0];
    EXPECT_EQ(f, 0u) << cfg<SG, P, true>("uint32") << " read another chunk's data in " << f << " of " << t
                     << " collective results";
    EXPECT_GT(t, 0u);
    return true;
}

template <size_t SG, typename T>
void divergence_all() {
    if (!type_runnable<T>()) GTEST_SKIP() << "device lacks fp64";
    bool ran = true;
    for_each_config<SG, (std::is_same_v<T, uint32_t> && SG != 64 ? &every_p : &edge_p)>([&](auto p, auto m) {
        if constexpr (decltype(m)::value) ran &= run_divergence<SG, decltype(p)::value, T>();
    });
    if (!ran) GTEST_SKIP() << "device lacks sub-group size " << SG;
}

template <typename T>
class SgPartitionDivergence : public ::testing::Test {};
using DivTypes = ::testing::Types<uint32_t, double, std::complex<double>, S12>;
TYPED_TEST_SUITE(SgPartitionDivergence, DivTypes);

TYPED_TEST(SgPartitionDivergence, MaskedSG32) { divergence_all<32, TypeParam>(); }
TYPED_TEST(SgPartitionDivergence, MaskedSG16) { divergence_all<16, TypeParam>(); }

TEST(SgPartition, DivergenceMaskedSG8) { divergence_all<8, uint32_t>(); }
TEST(SgPartition, DivergenceMaskedSG64) { divergence_all<64, uint32_t>(); }

template <size_t SG>
void isolation_all() {
    bool ran = true;
    for_each_config<SG>([&](auto p, auto m) {
        if constexpr (decltype(m)::value) ran &= run_isolation<SG, decltype(p)::value>();
    });
    if (!ran) GTEST_SKIP() << "device lacks sub-group size " << SG;
}
TEST(SgPartition, ChunkIsolationSG16) { isolation_all<16>(); }
TEST(SgPartition, ChunkIsolationSG32) { isolation_all<32>(); }

} // namespace
