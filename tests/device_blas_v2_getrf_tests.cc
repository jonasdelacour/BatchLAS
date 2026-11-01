// SPDX-License-Identifier: MIT
// Device LA v2, LU family (include/batchlas/blas/device/v2/ext/getrf.hh) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld with poisoned pad rows,
// rectangular m > n and m < n, complex data through the POD device scalar (as the kernels run
// it), batch 1024 with period-3 data where every item must be bit-identical to its twin (SLM
// slots, local tiles), pivots forced with a 4x margin so the host oracle is independent of
// rounding, an EXACT zero pivot plus a pre-set info (first failure wins), ties and NaN for the
// pivot search, the sfmin (divide) arm of rscl, Smith division past |d|^2 overflow, three
// executors per op (run-time work-group, run-time sub-group, static), and 3 sub-groups (a
// non-power-of-two team count) for the slot scan.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/getrf.hh>
#include <src/kernels/common/dla_v2_partition.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;
// The device scalar the kernels use: std::complex is re-typed to the POD Cx at the pointer.
template <class T> using D_t = std::conditional_t<is_cplx<T>, sycl_device::Cx<real_of<T>>, T>;
template <class T> D_t<T>* dv(T* p) { return reinterpret_cast<D_t<T>*>(p); }

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> T val(std::uint64_t i) {
    if constexpr (is_cplx<T>) return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else return uval<T>(i);
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> long double cabs1(LD<T> v) {
    if constexpr (is_cplx<T>) return std::fabs(v.real()) + std::fabs(v.imag());
    else return std::fabs(v);
}

struct alignas(16) Blk {
    unsigned char b[16];
};

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(h.size(), q());
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    // `groups` work-groups of WG; f(it, local bytes, 16-byte aligned).
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<Blk, 1> lm(sycl::range<1>(std::max<std::size_t>((local_bytes + 15) / 16, 1)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 f(it, reinterpret_cast<unsigned char*>(lm.template get_multi_ptr<sycl::access::decorated::no>().get()));
             });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GetrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GetrfTest, Types);

// ---- LU inputs and the host oracle ----------------------------------------------------------

constexpr float kPadPoison = 1234.5f;

// A = P0 L0 U0 with |L0(i,j)| <= 0.24 below a unit diagonal and |U0(j,j)| >= 1: partial pivoting
// must pick P0's rows with a 4x margin (cabs1 included), so host and device agree on every
// pivot whatever the rounding. `singular` zeroes U0's column kmax-1: that Schur column stays
// EXACTLY zero, the last step's pivot is 0 and info fires. Pad rows [m, ld) hold kPadPoison.
template <class T> std::vector<T> lu_input(int m, int n, int ld, std::uint64_t seed, bool singular) {
    using R = real_of<T>;
    const int kmax = std::min(m, n);
    std::vector<LD<T>> L(std::size_t(m) * kmax, LD<T>(0)), U(std::size_t(kmax) * n, LD<T>(0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < std::min(i + 1, kmax); ++j)
            L[i + std::size_t(j) * m] = i == j ? LD<T>(1) : LD<T>(val<T>(seed + 7u * i + 1000u * j)) * (long double)0.17;
    for (int j = 0; j < kmax; ++j)
        for (int c = j; c < n; ++c) {
            LD<T> u = LD<T>(val<T>(seed + 99991u + 13u * j + 1777u * c));
            if (c == j) u = LD<T>(T(R(1.5) + R(0.5) * uval<R>(seed + 31u * j))) + u * (long double)0.1;
            U[j + std::size_t(c) * kmax] = (singular && c == kmax - 1) ? LD<T>(0) : u;
        }
    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = m - 1; i > 0; --i) std::swap(perm[i], perm[mix(seed + 555u + i) % std::uint64_t(i + 1)]);
    std::vector<T> a(std::size_t(ld) * n, T(R(kPadPoison)));
    for (int i = 0; i < m; ++i)
        for (int c = 0; c < n; ++c) {
            LD<T> s = 0;
            for (int j = 0; j < kmax; ++j) s += L[i + std::size_t(j) * m] * U[j + std::size_t(c) * kmax];
            a[perm[i] + std::size_t(c) * ld] = static_cast<T>(s);
        }
    return a;
}

// kBatch items, item b a copy of representative b % kPeriod; representatives 1 and 2 singular.
template <class T> std::vector<T> lu_batch(int m, int n, int ld, std::uint64_t seed) {
    const std::size_t per = std::size_t(ld) * n;
    std::vector<T> h(per * kBatch);
    for (int r = 0; r < kPeriod; ++r) {
        const auto a = lu_input<T>(m, n, ld, seed + 104729u * r, r != 0);
        std::copy(a.begin(), a.end(), h.begin() + per * r);
    }
    for (int b = kPeriod; b < kBatch; ++b) std::copy_n(h.data() + per * (b % kPeriod), per, h.data() + per * b);
    return h;
}
// Representative 2 enters with a failure already recorded: first failure wins.
inline std::vector<std::int32_t> info_in() {
    std::vector<std::int32_t> v(kBatch);
    for (int b = 0; b < kBatch; ++b) v[b] = (b % kPeriod == 2) ? 7 : 0;
    return v;
}

// LAPACK ?GETF2 in long double with the cabs1 metric: pivots 1-based plus base, info.
template <class T> void host_getf2(const T* a0, int m, int n, int ld, int base, std::vector<int>& ipiv, int& info) {
    const int kmax = std::min(m, n);
    std::vector<LD<T>> a(std::size_t(ld) * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = LD<T>(a0[i]);
    auto at = [&](int i, int j) -> LD<T>& { return a[i + std::size_t(j) * ld]; };
    ipiv.assign(kmax, 0);
    for (int k = 0; k < kmax; ++k) {
        int p = k;
        long double best = -1;
        for (int i = k; i < m; ++i)
            if (cabs1<T>(at(i, k)) > best) best = cabs1<T>(at(i, k)), p = i;
        ipiv[k] = base + p + 1;
        for (int c = 0; c < n; ++c) std::swap(at(k, c), at(p, c));
        const LD<T> d = at(k, k);
        if (d == LD<T>(0)) {
            if (info == 0) info = base + k + 1;
            continue;
        }
        for (int i = k + 1; i < m; ++i) at(i, k) /= d;
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < m; ++i) at(i, j) -= at(i, k) * at(k, j);
    }
}

// max_ij |P A - L U|_ij / (|L| |U|)_ij from the DEVICE factors f and pivots (componentwise).
template <class T> double lu_resid(const T* a0, const T* f, const int* ipiv, int m, int n, int ld, int base) {
    const int kmax = std::min(m, n);
    std::vector<LD<T>> pa(std::size_t(m) * n);
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < m; ++i) pa[i + std::size_t(c) * m] = LD<T>(a0[i + std::size_t(c) * ld]);
    for (int k = 0; k < kmax; ++k) {
        const int p = ipiv[k] - base - 1;
        if (p < k || p >= m) return std::numeric_limits<double>::infinity();
        for (int c = 0; c < n; ++c) std::swap(pa[k + std::size_t(c) * m], pa[p + std::size_t(c) * m]);
    }
    double worst = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            LD<T> lu = 0;
            long double s = 0;
            for (int t = 0; t < std::min({i + 1, j + 1, kmax}); ++t) {
                const LD<T> l = t == i ? LD<T>(1) : LD<T>(f[i + std::size_t(t) * ld]);
                const LD<T> u = LD<T>(f[t + std::size_t(j) * ld]);
                lu += l * u, s += std::abs(l) * std::abs(u);
            }
            const long double d = std::abs(pa[i + std::size_t(j) * m] - lu);
            const double r = static_cast<double>(d / std::max<long double>(s, std::numeric_limits<real_of<T>>::min()));
            worst = std::max(worst, std::isnan(r) ? std::numeric_limits<double>::infinity() : r);
        }
    return worst;
}

// Every output of item b is bit-identical to its representative's; representatives match the
// host pivots and info, and factor P A within Check::factorization; pad rows are untouched.
template <class T>
void check_lu(const std::vector<T>& h, const std::vector<T>& g, const std::vector<int>& piv, const std::vector<std::int32_t>& info,
              int m, int n, int ld, int base, const char* what) {
    const int kmax = std::min(m, n);
    const std::size_t per = std::size_t(ld) * n;
    int bitdiff = 0;
    for (int b = kPeriod; b < kBatch; ++b) {
        const int r = b % kPeriod;
        for (std::size_t e = 0; e < per; ++e) bitdiff += !same_bits(g[per * b + e], g[per * r + e]);
        for (int k = 0; k < kmax; ++k) bitdiff += piv[std::size_t(b) * kmax + k] != piv[std::size_t(r) * kmax + k];
        bitdiff += info[b] != info[r];
    }
    EXPECT_EQ(bitdiff, 0) << what << ": items differ from their representative";
    for (int r = 0; r < kPeriod; ++r) {
        std::vector<int> hp;
        int hi = (r == 2) ? 7 : 0;
        host_getf2(h.data() + per * r, m, n, ld, base, hp, hi);
        for (int k = 0; k < kmax; ++k) EXPECT_EQ(piv[std::size_t(r) * kmax + k], hp[k]) << what << " rep " << r << " ipiv[" << k << "]";
        EXPECT_EQ(info[r], hi) << what << " rep " << r << " info";
        EXPECT_EQ(info[r], r == 0 ? 0 : (r == 1 ? base + kmax : 7)) << what << " rep " << r << " info, expected by construction";
        EXPECT_VERIFY(T, Check::factorization, kmax, lu_resid(h.data() + per * r, g.data() + per * r, piv.data() + std::size_t(r) * kmax, m, n, ld, base))
            << what << " rep " << r;
        int pad = 0;
        for (int c = 0; c < n; ++c)
            for (int i = m; i < ld; ++i) pad += !same_bits(g[per * r + i + std::size_t(c) * ld], T(real_of<T>(kPadPoison)));
        EXPECT_EQ(pad, 0) << what << " rep " << r << ": pad rows written";
    }
}

// ---- ?GETF2, explicit swaps: run-time work-group (global), run-time sub-group (local tile),
// static WorkGroup (global) -------------------------------------------------------------------

TYPED_TEST(DeviceBlasV2GetrfTest, Getf2ExplicitSwapEveryExecutor) {
    using T = TypeParam;
    using D = D_t<T>;
    using R = real_of<T>;
    struct Shape {
        int m, n, ld, base;
    };
    // m > n on a 128-wide group; m < n in a 21-row local tile per sub-group; m > n, static 64.
    const Shape sh[3] = {{45, 29, 48, 5}, {19, 23, 22, 0}, {70, 33, 75, 2}};
    for (int v = 0; v < 3; ++v) {
        const int m = sh[v].m, n = sh[v].n, ld = sh[v].ld, base = sh[v].base;
        const int kmax = std::min(m, n);
        const auto h = lu_batch<T>(m, n, ld, 17u + v);
        std::vector<int> hp(std::size_t(kBatch) * kmax, -9);
        T* a = this->dev.up(h);
        int* piv = this->dev.up(hp);
        std::int32_t* inf = this->dev.up(info_in());
        D* ad = dv(a);
        const std::size_t per = std::size_t(ld) * n;
        constexpr std::size_t slot_bytes = 32 * (sizeof(R) + sizeof(int));
        if (v == 0) {
            this->dev.template run<128>(kBatch, slot_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const v2::DynGroup<v2::Scope::Group> ex{it};
                R* vs = reinterpret_cast<R*>(lm);
                int* is = reinterpret_cast<int*>(lm + 32 * sizeof(R));
                std::int32_t info = ex.leader() ? inf[b] : 0;
                info = v2::getf2(ex, v2::mat(ad + per * b, m, n, ld), kmax, v2::Ipiv{piv + std::size_t(b) * kmax, base}, info, vs, is);
                if (ex.leader()) inf[b] = info;
            });
        } else if (v == 1) {
            const int ldt = 21;  // odd, below ld: the tile has its own layout
            this->dev.template run<64>(kBatch / 2, 2 * std::size_t(ldt) * n * sizeof(D), [=](sycl::nd_item<1> it, unsigned char* lm) {
                const v2::DynGroup<v2::Scope::Sub> ex{it};
                const int b = static_cast<int>(it.get_group_linear_id()) * 2 + ex.team();
                const auto At = v2::mat<v2::Mem::Local>(reinterpret_cast<D*>(lm) + std::size_t(ex.team()) * ldt * n, m, n, ldt);
                const auto Ag = v2::mat(ad + per * b, m, n, ld);
                v2::lacpy<v2::alg::Flat, std::size_t>(ex, Ag, At);
                ex.barrier();
                std::int32_t info = ex.leader() ? inf[b] : 0;
                info = v2::getf2(ex, At, kmax, v2::Ipiv{piv + std::size_t(b) * kmax, base}, info, static_cast<R*>(nullptr),
                                 static_cast<int*>(nullptr));
                if (ex.leader()) inf[b] = info;
                ex.barrier();
                v2::lacpy<v2::alg::Flat, std::size_t>(ex, At, Ag);
            });
        } else {
            this->dev.template run<64>(kBatch, slot_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const v2::WorkGroup<64> ex{it};
                R* vs = reinterpret_cast<R*>(lm);
                int* is = reinterpret_cast<int*>(lm + 32 * sizeof(R));
                std::int32_t info = ex.leader() ? inf[b] : 0;
                info = v2::getf2(ex, v2::mat(ad + per * b, m, n, ld), kmax, v2::Ipiv{piv + std::size_t(b) * kmax, base}, info, vs, is);
                if (ex.leader()) inf[b] = info;
            });
        }
        const char* what[3] = {"DynGroup<Group> global", "DynGroup<Sub> local tile", "WorkGroup<64> global"};
        check_lu<T>(h, this->dev.down(a, h.size()), this->dev.down(piv, hp.size()), this->dev.down(inf, std::size_t(kBatch)), m, n, ld,
                    base, what[v]);
    }
}

// ---- ?GETF2 by lazy relabel on a RowFrag --------------------------------------------------------

// A test-local pivot policy for one matrix per partition (getrf_tiny.cc's shape): the key is
// (order << 8) | lane, so the winner's lane comes back with its row; XorUp butterfly.
template <int N, class Px> struct KeyPivot {
    const Px& px;
    int n, lane, base;
    int* ipiv;
    std::int32_t& info;
    template <class R> v2::PivotWinner argmax(int j, R mag, int rowid) const {
        const bool cand = rowid >= j && rowid < n;
        v2::MaxLoc<R> a{(cand && (mag == mag)) ? mag : R(-1), ((cand ? rowid : rowid + N) << 8) | lane};
        v2::maxloc_team<v2::XorUp>(px, a);
        return {a.i >> 8, a.i & 0xff};
    }
    void record(int j, v2::PivotWinner w) const {
        if (lane == 0) ipiv[j] = base + w.row + 1;
    }
    void on_zero(bool zero, int j) const {
        if (zero && info == 0) info = base + j + 1;
    }
};

// (a) the register panel's shape: DynGroup<Group> of 64 items over m = 45 rows (19 pad items),
// PivotRowsGroup + BcastRowLocal; (b) the tiny tier's shape: Partition<16> per matrix, key
// pivot + BcastShuffle. Both must equal the host AND, bit for bit, v2::getf2 on the same data
// (the same arithmetic in the same order).
TYPED_TEST(DeviceBlasV2GetrfTest, Getf2RelabelMatchesExplicitSwapBitForBit) {
    using T = TypeParam;
    using D = D_t<T>;
    using R = real_of<T>;
    constexpr int NC = 16;
    for (int v = 0; v < 2; ++v) {
        const int m = v == 0 ? 45 : 13, n = 13, ld = v == 0 ? 50 : 15, base = v == 0 ? 3 : 0, kmax = 13;
        const auto h = lu_batch<T>(m, n, ld, 401u + v);
        const std::size_t per = std::size_t(ld) * n;
        std::vector<int> hp(std::size_t(kBatch) * kmax, -9);
        T* a = this->dev.up(h);
        T* a2 = this->dev.up(h);
        int* piv = this->dev.up(hp);
        int* piv2 = this->dev.up(hp);
        std::int32_t* inf = this->dev.up(info_in());
        std::int32_t* inf2 = this->dev.up(info_in());
        D* ad = dv(a);
        D* ad2 = dv(a2);
        constexpr std::size_t lbytes = 32 * (sizeof(R) + sizeof(int)) + NC * sizeof(D) + 16;
        if (v == 0) {
            this->dev.template run<64>(kBatch, lbytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const v2::DynGroup<v2::Scope::Group> ex{it};
                D* sx = reinterpret_cast<D*>(lm);
                R* vs = reinterpret_cast<R*>(lm + NC * sizeof(D));
                int* is = reinterpret_cast<int*>(lm + NC * sizeof(D) + 32 * sizeof(R));
                D* A = ad + per * b;
                const int tid = ex.lid();
                const bool live = tid < m;
                v2::Frag<D, NC> rA;
                for (int c = 0; c < NC; ++c) {
                    D x = v2::Arith<D>::zero();
                    if (live && c < n) x = A[tid + std::size_t(c) * ld];
                    rA[c] = x;
                }
                int rowid = live ? tid : -1;
                std::int32_t info = ex.leader() ? inf[b] : 0;
                auto pv = v2::pivot_rows_group(ex, vs, is, m, v2::Ipiv{piv + std::size_t(b) * kmax, base}, info);
                auto bc = v2::bcast_row_local(ex, sx);
                v2::getf2_relabel(rA, rowid, kmax, n, pv, bc);
                ex.barrier();  // every item has read A before rows land at their relabelled place
                if (live)
                    for (int c = 0; c < n; ++c) A[rowid + std::size_t(c) * ld] = rA[c];
                if (ex.leader()) inf[b] = info;
            });
        } else {
            this->dev.template run<64>(kBatch / 4, 0, [=](sycl::nd_item<1> it, unsigned char*) {
                const auto sg = it.get_sub_group();
                const v2::SgpPartition<16> px(sg);
                const int lane = px.lid();
                const int b = static_cast<int>(it.get_group_linear_id()) * 4 + static_cast<int>(sg.get_group_linear_id()) * 2 +
                              static_cast<int>(px.part.get_group_linear_id());
                D* A = ad + per * b;
                v2::Frag<D, NC> rA;
                for (int c = 0; c < NC; ++c) {
                    D x = c == lane ? v2::Arith<D>::one() : v2::Arith<D>::zero();  // identity pad rows/cols
                    if (lane < n && c < n) x = A[lane + std::size_t(c) * ld];
                    rA[c] = x;
                }
                int rowid = lane;
                std::int32_t info = inf[b];  // partition-uniform
                const KeyPivot<16, decltype(px)> pv{px, n, lane, base, piv + std::size_t(b) * kmax, info};
                const auto bc = v2::bcast_shuffle(px);
                v2::getf2_relabel(rA, rowid, kmax, n, pv, bc);
                sycl::group_barrier(sg);
                if (lane < n)
                    for (int c = 0; c < n; ++c) A[rowid + std::size_t(c) * ld] = rA[c];
                if (px.leader()) inf[b] = info;
            });
        }
        // The explicit-swap twin on the same data.
        this->dev.template run<64>(kBatch, 32 * (sizeof(R) + sizeof(int)), [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::DynGroup<v2::Scope::Group> ex{it};
            std::int32_t info = ex.leader() ? inf2[b] : 0;
            info = v2::getf2(ex, v2::mat(ad2 + per * b, m, n, ld), kmax, v2::Ipiv{piv2 + std::size_t(b) * kmax, base}, info,
                             reinterpret_cast<R*>(lm), reinterpret_cast<int*>(lm + 32 * sizeof(R)));
            if (ex.leader()) inf2[b] = info;
        });
        const char* what = v == 0 ? "relabel DynGroup<Group> + local row" : "relabel Partition<16> + shuffles";
        const auto g = this->dev.down(a, h.size()), g2 = this->dev.down(a2, h.size());
        const auto p = this->dev.down(piv, hp.size()), p2 = this->dev.down(piv2, hp.size());
        const auto i1 = this->dev.down(inf, std::size_t(kBatch)), i2 = this->dev.down(inf2, std::size_t(kBatch));
        check_lu<T>(h, g, p, i1, m, n, ld, base, what);
        int twin = 0;
        for (std::size_t e = 0; e < g.size(); ++e) twin += !same_bits(g[e], g2[e]);
        for (std::size_t e = 0; e < p.size(); ++e) twin += p[e] != p2[e];
        for (int b = 0; b < kBatch; ++b) twin += i1[b] != i2[b];
        EXPECT_EQ(twin, 0) << what << ": differs from v2::getf2 on the same data";
    }
}

// ---- pivot search, rscl, ger_flat, fp ----------------------------------------------------------

// iamax_loc on 96-wide groups (3 sub-groups: a non-power-of-two slot scan), one sub-group, a
// static WorkGroup and a Thread: a tie on one owner and across owners (lowest index wins),
// NaN skipped, all-NaN and empty -> n.
TYPED_TEST(DeviceBlasV2GetrfTest, IamaxLocTiesNanEveryExecutor) {
    using T = TypeParam;
    using D = D_t<T>;
    using R = real_of<T>;
    constexpr int n = 250, ncase = 4, nexec = 4;
    const R nan = std::numeric_limits<R>::quiet_NaN();
    std::vector<T> hx(std::size_t(ncase) * n);
    for (int i = 0; i < n; ++i) hx[i] = hx[n + i] = val<T>(i) * R(0.5);
    hx[200] = hx[200 - 96] = hx[200 - 64] = T(R(3));  // 104 and 200 share owner 8, 136 is owner 40: 104 wins
    if constexpr (is_cplx<T>) hx[136] = T(R(1), R(2));  // still a cabs1 tie (3); its modulus is 2.24
    hx[n + 3] = T(nan);
    hx[n + 77] = T(R(5));
    hx[n + 78] = T(R(5));
    for (int i = 0; i < n; ++i) hx[2 * n + i] = T(nan);
    std::vector<int> ho(std::size_t(ncase) * nexec, -7);
    T* x = this->dev.up(hx);
    int* o = this->dev.up(ho);
    D* xd = dv(x);
    this->dev.template run<96>(ncase, 64 * (sizeof(R) + sizeof(int)), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int c = static_cast<int>(it.get_group_linear_id());
        R* vs = reinterpret_cast<R*>(lm);                       // 64 slots: [0, 32) and [32, 64)
        int* is = reinterpret_cast<int*>(lm + 64 * sizeof(R));  // 64 slots
        const auto xv = v2::vec(xd + std::size_t(c) * n, c == 3 ? 0 : n);
        const v2::DynGroup<v2::Scope::Group> eg{it};
        const int ig = v2::iamax_loc(eg, xv, vs, is);
        eg.barrier();
        const int isub = v2::iamax_loc(v2::DynGroup<v2::Scope::Sub>{it}, xv, vs, is);
        const int iwg = v2::iamax_loc(v2::WorkGroup<96>{it}, xv, vs + 32, is + 32);
        int ith = -5;
        if (eg.leader()) ith = v2::iamax_loc(v2::Thread{}, xv, vs, is);
        if (eg.leader()) o[c * nexec] = ig, o[c * nexec + 1] = isub, o[c * nexec + 2] = iwg, o[c * nexec + 3] = ith;
    });
    const auto g = this->dev.down(o, ho.size());
    const int want[ncase] = {104, 77, n, 0};
    for (int c = 0; c < ncase; ++c)
        for (int e = 0; e < nexec; ++e) EXPECT_EQ(g[c * nexec + e], want[c]) << "case " << c << " executor " << e;
}

// rscl: a normal d takes the reciprocal; a subnormal d (1/d overflows) takes the divide arm;
// complex Smith recip/div at |d| ~ max/4, where |d|^2 overflows. ger_flat NegOne and One on a
// 96-wide DynGroup and a static WorkGroup, ld padded and poisoned.
TYPED_TEST(DeviceBlasV2GetrfTest, RsclArmsSmithAndGerFlat) {
    using T = TypeParam;
    using D = D_t<T>;
    using R = real_of<T>;
    constexpr int n = 77, m = 37, k = 21, ld = m + 6;
    const R sub = std::numeric_limits<R>::min() / R(16), big = std::numeric_limits<R>::max() / R(4);
    std::vector<T> hx(3 * n);
    for (int i = 0; i < n; ++i) hx[i] = val<T>(10 + i), hx[n + i] = val<T>(500 + i) * sub * R(4), hx[2 * n + i] = val<T>(900 + i) * big;
    const T d0 = val<T>(5) + T(R(2)), d1 = T(sub);
    T d2 = T(big);
    if constexpr (is_cplx<T>) d2 = T(big, big / R(3));
    std::vector<T> hd = {d0, d1, d2};
    std::vector<int> hflag(2, -1);
    auto ha = std::vector<T>(std::size_t(ld) * k * 2, T(R(kPadPoison)));
    std::vector<T> hy(k), hv(m);
    for (int i = 0; i < m; ++i) hv[i] = val<T>(3000 + i);
    for (int c = 0; c < k; ++c) hy[c] = val<T>(4000 + c);
    for (int c = 0; c < k; ++c)
        for (int i = 0; i < m; ++i) ha[i + std::size_t(c) * ld] = ha[std::size_t(ld) * k + i + std::size_t(c) * ld] = val<T>(5000 + i + 97 * c);
    T* x = this->dev.up(hx);
    T* dd = this->dev.up(hd);
    int* flag = this->dev.up(hflag);
    T* a = this->dev.up(ha);
    T* y = this->dev.up(hy);
    T* vv = this->dev.up(hv);
    D *xd = dv(x), *ddd = dv(dd), *aad = dv(a), *yd = dv(y), *vd = dv(vv);
    this->dev.template run<96>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::DynGroup<v2::Scope::Group> eg{it};
        v2::rscl(eg, ddd[0], v2::vec(xd, n));
        v2::rscl(eg, ddd[1], v2::vec(xd + n, n));
        v2::rscl(eg, ddd[2], v2::vec(xd + 2 * n, n));
        if (eg.leader()) flag[0] = v2::fp::is_zero(ddd[1]) ? 1 : 0, flag[1] = v2::fp::isfinite(v2::fp::recip(ddd[1])) ? 1 : 0;
        v2::ger_flat(eg, v2::NegOne{}, v2::assume_published(v2::vec(vd, m)), v2::assume_published(v2::vec(yd, k)), v2::mat(aad, m, k, ld));
        const v2::WorkGroup<96> ew{it};
        v2::ger_flat(ew, v2::One{}, v2::assume_published(v2::vec(vd, m)), v2::assume_published(v2::vec(yd, k)),
                     v2::mat(aad + std::size_t(ld) * k, m, k, ld));
    });
    const auto gx = this->dev.down(x, hx.size());
    const auto gf = this->dev.down(flag, 2);
    const auto ga = this->dev.down(a, ha.size());
    double w_mul = 0, w_div = 0, w_big = 0;
    auto rel = [](T got, LD<T> ref) {
        const long double d = std::abs(LD<T>(got) - ref), s = std::max<long double>(std::abs(ref), std::numeric_limits<real_of<T>>::min());
        return std::isnan(static_cast<double>(d)) ? std::numeric_limits<double>::infinity() : static_cast<double>(d / s);
    };
    for (int i = 0; i < n; ++i) {
        w_mul = std::max(w_mul, rel(gx[i], LD<T>(hx[i]) / LD<T>(d0)));
        w_div = std::max(w_div, rel(gx[n + i], LD<T>(hx[n + i]) / LD<T>(d1)));
        w_big = std::max(w_big, rel(gx[2 * n + i], LD<T>(hx[2 * n + i]) / LD<T>(d2)));
    }
    EXPECT_VERIFY(T, Check::blas, 2, w_mul) << "rscl, reciprocal arm";
    EXPECT_VERIFY(T, Check::blas, 4, w_big) << "rscl at |d| ~ max/4 (Smith, no |d|^2)";
    if (gf[0] == 1) {
        GTEST_LOG_(INFO) << "the device flushes subnormals: the sfmin arm is unreachable here";
    } else {
        EXPECT_EQ(gf[1], 0) << "1/d of a subnormal must overflow, or this case does not reach the divide arm";
        EXPECT_VERIFY(T, Check::blas, 2, w_div) << "rscl, divide (sfmin) arm";
    }
    double w_ger = 0;
    int pad = 0;
    for (int s = 0; s < 2; ++s)
        for (int c = 0; c < k; ++c) {
            for (int i = 0; i < m; ++i) {
                const LD<T> a0 = LD<T>(ha[std::size_t(ld) * k * s + i + std::size_t(c) * ld]), p = LD<T>(hv[i]) * LD<T>(hy[c]);
                const LD<T> ref = s == 0 ? a0 - p : a0 + p;
                const long double sc = std::abs(a0) + std::abs(p);
                const LD<T> got = LD<T>(ga[std::size_t(ld) * k * s + i + std::size_t(c) * ld]);
                w_ger = std::max(w_ger, static_cast<double>(std::abs(got - ref) / sc));
            }
            for (int i = m; i < ld; ++i) pad += !same_bits(ga[std::size_t(ld) * k * s + i + std::size_t(c) * ld], T(R(kPadPoison)));
        }
    EXPECT_VERIFY(T, Check::blas, 1, w_ger) << "ger_flat";
    EXPECT_EQ(pad, 0) << "ger_flat wrote the ld pad";
}

// ---- ?LASWP and the 2-D traversals -----------------------------------------------------------

// laswp Forward then Backward on one column (Thread) restores it bit for bit; the Mat form over
// a DynGroup and a static WorkGroup matches the host; the INT form (the gathers' composition);
// lacpy under alg::Flat int / size_t and alg::Stepped leaves the destination pad untouched.
TYPED_TEST(DeviceBlasV2GetrfTest, LaswpDirectionsAndTraversals) {
    using T = TypeParam;
    using D = D_t<T>;
    using R = real_of<T>;
    constexpr int m = 53, nc = 37, ld = m + 5, k0 = 4, k1 = 31;
    std::vector<int> hp(m);
    for (int k = 0; k < m; ++k) hp[k] = k + 1 + static_cast<int>(mix(800 + k) % std::uint64_t(m - k));  // 1-based, in [k, m)
    std::vector<T> ha(std::size_t(ld) * nc * 3, T(R(kPadPoison)));
    for (int c = 0; c < nc; ++c)
        for (int i = 0; i < m; ++i)
            for (int s = 0; s < 3; ++s) ha[std::size_t(ld) * nc * s + i + std::size_t(c) * ld] = val<T>(7000 + i + 131 * c);
    std::vector<int> hidx(m, -1);
    T* a = this->dev.up(ha);
    int* ip = this->dev.up(hp);
    int* idx = this->dev.up(hidx);
    D* ad = dv(a);
    const std::size_t blk = std::size_t(ld) * nc;
    this->dev.template run<96>(1, m * sizeof(int), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const v2::DynGroup<v2::Scope::Group> eg{it};
        const auto to = [ip](int k) { return ip[k] - 1; };
        // block 0: Forward then Backward per column, one item per column
        if (eg.lid() < nc) {
            const auto col = v2::vec(ad + std::size_t(eg.lid()) * ld, m);
            v2::laswp<v2::LaswpDir::Forward>(v2::Thread{}, col, to, k0, k1);
            v2::laswp<v2::LaswpDir::Backward>(v2::Thread{}, col, to, k0, k1);
        }
        // block 1: the Mat form on the DynGroup; block 2: on a static WorkGroup<96>
        v2::laswp<v2::LaswpDir::Forward>(eg, v2::mat(ad + blk, m, nc, ld), to, k0, k1);
        v2::laswp<v2::LaswpDir::Forward>(v2::WorkGroup<96>{it}, v2::mat(ad + 2 * blk, m, nc, ld), to, k0, k1);
        // the gathers' composition: idx := P iota in local memory, then out
        int* li = reinterpret_cast<int*>(lm);
        v2::for_each_owned(eg, m, [&](int i) { li[i] = i; });
        eg.barrier();
        if (eg.leader()) v2::laswp<v2::LaswpDir::Forward>(v2::Thread{}, v2::vec<v2::Mem::Local>(li, m), to, k0, k1);
        eg.barrier();
        v2::for_each_owned(eg, m, [&](int i) { idx[i] = li[i]; });
    });
    const auto g = this->dev.down(a, ha.size());
    const auto gi = this->dev.down(idx, hidx.size());
    std::vector<T> ref(ha.begin(), ha.begin() + blk);
    std::vector<int> rid(m);
    std::iota(rid.begin(), rid.end(), 0);
    for (int k = k0; k < k1; ++k) {
        const int p = hp[k] - 1;
        for (int c = 0; c < nc; ++c) std::swap(ref[k + std::size_t(c) * ld], ref[p + std::size_t(c) * ld]);
        std::swap(rid[k], rid[p]);
    }
    int d0 = 0, d1 = 0, d2 = 0, di = 0;
    for (std::size_t e = 0; e < blk; ++e) {
        d0 += !same_bits(g[e], ha[e]);
        d1 += !same_bits(g[blk + e], ref[e]);
        d2 += !same_bits(g[2 * blk + e], ref[e]);
    }
    for (int i = 0; i < m; ++i) di += gi[i] != rid[i];
    EXPECT_EQ(d0, 0) << "Backward must undo Forward";
    EXPECT_EQ(d1, 0) << "laswp Mat form on DynGroup";
    EXPECT_EQ(d2, 0) << "laswp Mat form on WorkGroup<96>";
    EXPECT_EQ(di, 0) << "laswp on the int index vector";

    // lacpy under each traversal: 41 x 29 from ld 47 into ld 44 (poisoned pad), 96 wide.
    constexpr int cm = 41, cn = 29, lds = 47, ldd = 44;
    std::vector<T> hs(std::size_t(lds) * cn), hd(std::size_t(ldd) * cn * 3, T(R(kPadPoison)));
    for (std::size_t e = 0; e < hs.size(); ++e) hs[e] = val<T>(9000 + e);
    T* s = this->dev.up(hs);
    T* dsts = this->dev.up(hd);
    const std::size_t dblk = std::size_t(ldd) * cn;
    this->dev.template run<96>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const v2::DynGroup<v2::Scope::Group> eg{it};
        const auto src = v2::mat(s, cm, cn, lds);
        v2::lacpy<v2::alg::Flat, int>(eg, src, v2::mat(dsts, cm, cn, ldd));
        v2::lacpy<v2::alg::Flat, std::size_t>(v2::WorkGroup<96>{it}, src, v2::mat(dsts + dblk, cm, cn, ldd));
        v2::lacpy<v2::alg::Stepped>(eg, src, v2::mat(dsts + 2 * dblk, cm, cn, ldd));
    });
    const auto gd = this->dev.down(dsts, hd.size());
    for (int t = 0; t < 3; ++t) {
        int bad = 0;
        for (int c = 0; c < cn; ++c)
            for (int i = 0; i < ldd; ++i) {
                const T want = i < cm ? hs[i + std::size_t(c) * lds] : T(R(kPadPoison));
                bad += !same_bits(gd[dblk * t + i + std::size_t(c) * ldd], want);
            }
        EXPECT_EQ(bad, 0) << "lacpy traversal " << t;
    }
}

static_assert(!v2::is_dyn_group_v<v2::WorkGroup<64>> && v2::is_dyn_group_v<v2::DynGroup<v2::Scope::Sub>>);
static_assert(v2::DynGroup<v2::Scope::Sub>::natural_scope == v2::Scope::Sub && v2::DynGroup<v2::Scope::Sub>::sub_whole);

}  // namespace
