// SPDX-License-Identifier: MIT
// The trsm-private device LA ops (src/ops/trsm/trsm_v2.hh) against host residuals, driven
// through both kernels' op sequences. Shared ops (arith::Fma, RtGroup, Partition::bcast, Frag
// load_row / store_row) are covered by device_blas_v2_tests.
// Guards (docs/developer/agent-guide.md section 8): every (side, uplo, trans, diag) fold; lda and
// ldb above the natural ones; complex data; the unreferenced triangle, a unit diagonal and the ld
// gaps NaN-poisoned, B's gaps checked unchanged; pad rows (n < N) and dead lanes / chunks; batch
// 1024 with period-3 data where every item must be bit-identical to its representative; a
// RtGroup launch at V1's ceiling (WG 256); the division fallback on a subnormal diagonal;
// sycl_device::Cx bit-identical to std::complex.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/arith.hh>
#include <src/sycl/device_scalar.hh>
#include <src/ops/trsm/trsm_canonical.hh>
#include <src/ops/trsm/trsm_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
using batchlas::sycl_trsm::Canonical;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
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
template <class T> T nan_of() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> conj_h(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 f(it, batchlas::util::get_raw_ptr(lm));
             });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

struct Case {
    Side side;
    Uplo uplo;
    Transpose trans;
    Diag diag;
    std::string name() const {
        return std::string(side == Side::Left ? "L" : "R") + (uplo == Uplo::Lower ? "lo" : "up") +
               (trans == Transpose::NoTrans ? "N" : trans == Transpose::Trans ? "T" : "C") + (diag == Diag::Unit ? "U" : "D");
    }
};
std::vector<Case> every_fold(Side side) {
    std::vector<Case> cs;
    for (Uplo u : {Uplo::Lower, Uplo::Upper})
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
            for (Diag d : {Diag::NonUnit, Diag::Unit}) cs.push_back({side, u, t, d});
    return cs;
}

// One trsm problem per item: A (n x n, lda = n + 3) with only its stored triangle finite, B
// (Left n x q, Right q x n) with NaN ld gaps. tiny >= 0 (Left Lower NoTrans only): that row of
// A is d = 1e-309 on the diagonal and 0 to its left, and B's row is d * O(1): the reciprocal
// overflows while the solution stays O(1).
template <class T> struct Problem {
    Case c;
    int n, q, lda, ldb, brows, bcols;
    T alpha;
    std::vector<T> a, b;
    int a_elems() const { return lda * n; }
    int b_elems() const { return ldb * bcols; }
};
template <class T> Problem<T> make_problem(const Case& c, int n, int q, std::uint64_t seed, int tiny = -1) {
    Problem<T> P{c, n, q, n + 3, 0, 0, 0, tiny >= 0 ? T(1) : val<T>(seed + 99), {}, {}};
    P.brows = c.side == Side::Left ? n : q;
    P.bcols = c.side == Side::Left ? q : n;
    P.ldb = P.brows + (c.side == Side::Left ? 5 : 2);
    P.a.resize(std::size_t(kBatch) * P.a_elems());
    P.b.resize(std::size_t(kBatch) * P.b_elems());
    for (int it = 0; it < kPeriod; ++it) {
        T* a = P.a.data() + std::size_t(it) * P.a_elems();
        T* b = P.b.data() + std::size_t(it) * P.b_elems();
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < P.lda; ++i) {
                const bool tri = i < n && (c.uplo == Uplo::Lower ? i >= j : i <= j);
                T v = val<T>(seed + std::uint64_t(it) * 1000003u + std::uint64_t(j) * P.lda + i);
                if (!tri || (i == j && c.diag == Diag::Unit)) v = nan_of<T>();
                else if (i == j) v = v + T(device::Arith<T>::re(v) >= 0 ? 1.5 : -1.5);
                else v = v * T(0.5);
                if (i == tiny && j < tiny) v = T(0);
                if (i == tiny && j == tiny) v = T(static_cast<real_of<T>>(1e-309));
                a[i + j * P.lda] = v;
            }
        for (int j = 0; j < P.bcols; ++j)
            for (int i = 0; i < P.ldb; ++i) {
                T v = i < P.brows ? val<T>(seed + 77777u + std::uint64_t(it) * 1000003u + std::uint64_t(j) * P.ldb + i) : nan_of<T>();
                if (i == tiny) v = v * T(static_cast<real_of<T>>(1e-309));
                b[i + j * P.ldb] = v;
            }
    }
    for (int it = kPeriod; it < kBatch; ++it) {
        std::copy_n(P.a.data() + std::size_t(it % kPeriod) * P.a_elems(), P.a_elems(), P.a.data() + std::size_t(it) * P.a_elems());
        std::copy_n(P.b.data() + std::size_t(it % kPeriod) * P.b_elems(), P.b_elems(), P.b.data() + std::size_t(it) * P.b_elems());
    }
    return P;
}

template <class T> double rel(LD<T> r, long double scale) {
    const long double d = std::abs(r);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Componentwise backward-error residual |op(A) X - alpha B| / (|op(A)| |X| + |alpha B|) of the
// representatives, the twins' bits and B's untouched gaps.
template <class T> void check_solution(const Problem<T>& P, const std::vector<T>& x, const std::string& where) {
    const Case& c = P.c;
    long long bitdiff = 0, gapdiff = 0;
    for (int it = 0; it < kBatch; ++it) {
        const T* g = x.data() + std::size_t(it) * P.b_elems();
        const T* g0 = x.data() + std::size_t(it % kPeriod) * P.b_elems();
        for (int e = 0; e < P.b_elems(); ++e) {
            bitdiff += !same_bits(g[e], g0[e]);
            if (e % P.ldb >= P.brows) gapdiff += !same_bits(g[e], P.b[std::size_t(it) * P.b_elems() + e]);
        }
    }
    EXPECT_EQ(bitdiff, 0) << where << ": items differ from their representative";
    EXPECT_EQ(gapdiff, 0) << where << ": an ld gap of B was written";
    double worst = 0;
    for (int it = 0; it < kPeriod; ++it) {
        const T* a = P.a.data() + std::size_t(it) * P.a_elems();
        const T* b = P.b.data() + std::size_t(it) * P.b_elems();
        const T* xs = x.data() + std::size_t(it) * P.b_elems();
        auto tri = [&](int i, int j) -> LD<T> {
            if (i == j && c.diag == Diag::Unit) return LD<T>(1);
            if (c.uplo == Uplo::Lower ? i < j : i > j) return LD<T>(0);
            return LD<T>(a[i + j * P.lda]);
        };
        auto op = [&](int i, int j) -> LD<T> {
            if (c.trans == Transpose::NoTrans) return tri(i, j);
            return c.trans == Transpose::Trans ? tri(j, i) : conj_h<T>(tri(j, i));
        };
        auto X = [&](int i, int j) { return LD<T>(xs[i + j * P.ldb]); };
        const LD<T> al = LD<T>(P.alpha);
        for (int u = 0; u < P.q; ++u)
            for (int i = 0; i < P.n; ++i) {
                const LD<T> rhs = al * LD<T>(c.side == Side::Left ? b[i + u * P.ldb] : b[u + i * P.ldb]);
                LD<T> r = -rhs;
                long double s = std::abs(rhs);
                for (int k = 0; k < P.n; ++k) {
                    const LD<T> t = c.side == Side::Left ? op(i, k) * X(k, u) : X(u, k) * op(k, i);
                    r += t;
                    s += std::abs(c.side == Side::Left ? op(i, k) : op(k, i)) * std::abs(c.side == Side::Left ? X(k, u) : X(u, k));
                }
                worst = std::max(worst, rel<T>(r, s));
            }
    }
    EXPECT_VERIFY(T, Check::solve, P.n, worst) << where;
}

// V1's shape: RtGroup (width read from the launch), packed Lc in SLM, one rhs per item,
// staged (TR-row tiles) or direct. Returns X; flags[item] = the division-fallback flag.
template <class D, int N, int TR, bool Staged, int WG, class T>
std::vector<T> packed_solve(Dev& dev, const Problem<T>& P, std::vector<int>* flags = nullptr) {
    static_assert(sizeof(D) == sizeof(T));
    const Canonical can = sycl_trsm::canonicalise(P.c.side, P.c.uplo, P.c.trans, P.c.diag);
    const D* a = reinterpret_cast<const D*>(dev.up(P.a));
    T* bt = dev.up(P.b);
    D* b = reinterpret_cast<D*>(bt);
    int* fl = dev.up(std::vector<int>(kBatch, -1));
    const int n = P.n, q = P.q, lda = P.lda, ldb = P.ldb, sa = P.a_elems(), sb = P.b_elems();
    const bool fwd = can.fwd, trans = can.do_trans, conj = can.do_conj, unit = can.unit;
    D alpha;
    std::memcpy(&alpha, &P.alpha, sizeof(D));
    constexpr int kTri = N * (N + 1) / 2;
    const std::size_t bytes = sizeof(D) * (kTri + N + (Staged ? (TR + 1) * WG : 0)) + sizeof(int);
    auto body = [=](sycl::nd_item<1> it, unsigned char* lm, auto left) {
        constexpr bool kLeft = decltype(left)::value;
        using Ar = device::arith_t<device::arith::Fma, D>;
        const int item = static_cast<int>(it.get_group_linear_id());
        const device::RtGroup<> ex{it};
        const int lane = ex.lid();
        const bool live = lane < q;
        D* sLc = reinterpret_cast<D*>(lm);
        D* sRd = sLc + kTri;
        D* sTile = sRd + N;
        int* sDiv = reinterpret_cast<int*>(sTile + (Staged ? (TR + 1) * ex.width() : 0));
        D* Bb = b + std::ptrdiff_t(item) * sb;
        const sycl_trsm::dla::CanonTri<const D, kLeft> L{a + std::ptrdiff_t(item) * sa, lda, n, fwd, trans, conj};
        if (lane == 0) sDiv[0] = 0;
        sycl_trsm::dla::tp_pack_lower<N>(ex, L, n, sLc);
        ex.barrier(device::Mem::Global);
        sycl_trsm::dla::tp_diag_recip<N, Ar>(ex, sLc, n, unit, sRd, sDiv);
        ex.barrier(device::Mem::Global);
        const sycl_trsm::dla::TpDiag<D> dg{sLc, sRd, unit, sDiv[0] != 0};
        if (lane == 0) fl[item] = sDiv[0];
        device::Frag<D, N> x;
        if constexpr (Staged) {
            const sycl_trsm::dla::FlipRows<D> Bc{Bb, ldb, n, q, fwd};
            D* my = sTile + lane * (TR + 1);
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < (N + TR - 1) / TR; ++k) {
                ex.barrier(device::Mem::Global);
                sycl_trsm::dla::lacpy_tile_in<TR, TR + 1>(ex, Bc, k * TR, 0, sTile);
                ex.barrier(device::Mem::Global);
                sycl_trsm::dla::trsv_lower_rows<N, TR, Ar>(x, k * TR, sLc, dg, [&](int s) {
                    D r = Ar::zero();
                    if (live && s < n) r = Ar::mul(alpha, my[s - k * TR]);
                    return r;
                });
            }
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < (N + TR - 1) / TR; ++k) {
                ex.barrier(device::Mem::Global);
                sycl_trsm::dla::frag_copy_out<TR>(x, k * TR, my);
                ex.barrier(device::Mem::Global);
                sycl_trsm::dla::lacpy_tile_out<TR, TR + 1>(ex, sTile, Bc, k * TR, 0);
            }
        } else {
            const auto bu = sycl_trsm::dla::CanonRhs<D, kLeft>(Bb, n, ldb, fwd).rhs(lane);
            sycl_trsm::dla::trsv_lower_rows<N, N, Ar>(x, 0, sLc, dg, [&](int s) {
                D r = Ar::zero();
                if (live && s < n) r = Ar::mul(alpha, bu(s));
                return r;
            });
            device::store(x, bu, live ? n : 0);
        }
    };
    if (P.c.side == Side::Left) {
        dev.template run<WG>(kBatch, bytes, [=](sycl::nd_item<1> it, unsigned char* lm) { body(it, lm, std::true_type{}); });
    } else if constexpr (!Staged) {
        dev.template run<WG>(kBatch, bytes, [=](sycl::nd_item<1> it, unsigned char* lm) { body(it, lm, std::false_type{}); });
    } else {
        ADD_FAILURE() << "the staged tile is Side::Left only";
    }
    if (flags) *flags = dev.down(fl, kBatch);
    return dev.down(bt, P.b.size());
}

// trsm_sg_left's shape: Partition<P> per item (32/P items per sub-group), lane r holds row r,
// QC rhs per chunk; Unroll = 1 is the rolled step loop.
template <class D, int P, int QC, int Unroll, class T>
std::vector<T> rowfrag_solve(Dev& dev, const Problem<T>& Pb, std::vector<int>* flags = nullptr) {
    constexpr int kWg = 128, kMpw = 32 / P;
    const Canonical can = sycl_trsm::canonicalise(Side::Left, Pb.c.uplo, Pb.c.trans, Pb.c.diag);
    const D* a = reinterpret_cast<const D*>(dev.up(Pb.a));
    T* bt = dev.up(Pb.b);
    D* b = reinterpret_cast<D*>(bt);
    int* fl = dev.up(std::vector<int>(kBatch, -1));
    const int n = Pb.n, q = Pb.q, sa = Pb.a_elems(), sb = Pb.b_elems();
    const std::int64_t lda = Pb.lda, ldb = Pb.ldb;
    const bool fwd = can.fwd, trans = can.do_trans, conj = can.do_conj, unit = can.unit;
    D alpha;
    std::memcpy(&alpha, &Pb.alpha, sizeof(D));
    const int chunks = (q + QC - 1) / QC;
    const int groups = kBatch / kMpw * chunks / (kWg / 32);
    dev.template run<kWg>(groups, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        using Ar = device::arith_t<device::arith::Fma, D>;
        const auto sg = it.get_sub_group();
        const int gsg = static_cast<int>(it.get_group_linear_id()) * (kWg / 32) + static_cast<int>(sg.get_group_linear_id());
        const device::Partition<P> ex{sg};
        const int r = ex.lid();
        const int chunk = gsg % chunks;
        const int item = (gsg / chunks) * kMpw + static_cast<int>(sg.get_local_linear_id()) / P;
        const bool okr = r < n;
        const sycl_trsm::dla::CanonTri<const D, true, std::int64_t> L{a + std::ptrdiff_t(item) * sa, lda, n, fwd, trans, conj};
        const sycl_trsm::dla::FlipRows<D, std::int64_t> Bc{b + std::ptrdiff_t(item) * sb, ldb, n, q, fwd};
        const auto neg = [](const D& v) { return device::Arith<D>::neg(v); };
        device::Frag<D, P> nL;
        device::load_row(nL, sycl_trsm::dla::MapCols{L, 0, neg}, r, okr ? r : 0);
        D dd = Ar::one();
        if (okr && !unit) dd = L.diag(r);
        const auto dg = sycl_trsm::dla::guarded_recip<Ar>(ex, dd);
        if (r == 0 && chunk == 0) fl[item] = dg.divide ? 1 : 0;
        const int c0 = chunk * QC;
        const int live_c = okr ? q - c0 : 0;
        const auto scale = [=](const D& v) { return Ar::mul(alpha, v); };
        device::Frag<D, QC> x;
        device::load_row(x, sycl_trsm::dla::MapCols{Bc, c0, scale}, r, live_c);
        sycl_trsm::dla::trsm_lower_rowfrag<Unroll, Ar>(ex, nL, dg, n, unit, x);
        device::store_row(x, sycl_trsm::dla::ColsFrom{Bc, c0}, r, live_c);
    });
    if (flags) *flags = dev.down(fl, kBatch);
    return dev.down(bt, Pb.b.size());
}

template <class T> class DeviceBlasV2TrsmTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "trsm kernel tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2TrsmTest, Types);

// n = 13 in the N = 16 bucket (3 pad rows), q = 59 of 64 lanes live; TR = 8 gives two rounds.
TYPED_TEST(DeviceBlasV2TrsmTest, PackedThreadSolveEveryFold) {
    using T = TypeParam;
    for (Side side : {Side::Left, Side::Right})
        for (const Case& c : every_fold(side)) {
            const auto P = make_problem<T>(c, 13, 59, 11);
            check_solution(P, packed_solve<T, 16, 8, false, 64>(this->dev, P), c.name() + " direct");
            if (side == Side::Left) check_solution(P, packed_solve<T, 16, 8, true, 64>(this->dev, P), c.name() + " staged");
        }
}

// V1's widest rung (kTrsmV1MaxWg = 256): 251 live lanes, the staged tile at its largest.
TYPED_TEST(DeviceBlasV2TrsmTest, PackedThreadSolveAtTheCeiling) {
    using T = TypeParam;
    for (const Case& c : {Case{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit},
                          Case{Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::Unit},
                          Case{Side::Right, Uplo::Upper, Transpose::Trans, Diag::NonUnit}}) {
        const auto P = make_problem<T>(c, 13, 251, 23);
        check_solution(P, packed_solve<T, 16, 8, false, 256>(this->dev, P), c.name() + " direct WG256");
        if (c.side == Side::Left) check_solution(P, packed_solve<T, 16, 8, true, 256>(this->dev, P), c.name() + " staged WG256");
    }
}

// Partition<8>, n = 7 (one pad lane), q = 6 in chunks of 4 (the second half-empty); the step
// loop unrolled and rolled.
TYPED_TEST(DeviceBlasV2TrsmTest, RowFragPartitionSolveEveryFold) {
    using T = TypeParam;
    for (const Case& c : every_fold(Side::Left)) {
        const auto P = make_problem<T>(c, 7, 6, 31);
        check_solution(P, rowfrag_solve<T, 8, 4, 8>(this->dev, P), c.name() + " rowfrag unrolled");
        check_solution(P, rowfrag_solve<T, 8, 4, 1>(this->dev, P), c.name() + " rowfrag rolled");
    }
}

// 1 / 1e-309 overflows: the multiply path would give inf. Float types skip: their subnormal
// diagonal may be flushed to zero.
TYPED_TEST(DeviceBlasV2TrsmTest, DivisionFallbackOnSubnormalDiagonal) {
    using T = TypeParam;
    if constexpr (sizeof(real_of<T>) < 8) {
        GTEST_SKIP() << "needs a subnormal whose reciprocal overflows";
    } else {
        const Case c{Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit};
        std::vector<int> flags;
        const auto P = make_problem<T>(c, 13, 59, 41, 5);
        check_solution(P, packed_solve<T, 16, 8, false, 64>(this->dev, P, &flags), "packed tiny diagonal");
        EXPECT_EQ(std::count(flags.begin(), flags.end(), 1), kBatch) << "packed: the division flag was not raised";
        const auto Q = make_problem<T>(c, 7, 6, 43, 3);
        check_solution(Q, rowfrag_solve<T, 8, 4, 8>(this->dev, Q, &flags), "rowfrag tiny diagonal");
        EXPECT_EQ(std::count(flags.begin(), flags.end(), 1), kBatch) << "rowfrag: the division flag was not raised";
        const auto R = make_problem<T>(c, 13, 59, 41);
        (void)packed_solve<T, 16, 8, false, 64>(this->dev, R, &flags);
        EXPECT_EQ(std::count(flags.begin(), flags.end(), 0), kBatch) << "packed: the flag rose on a normal diagonal";
    }
}

// The kernels re-type std::complex as sycl_device::Cx: the private ops (TpDiag, guarded_recip,
// core Arith::isfinite) under arith::Fma must give the same bits on Cx.
TYPED_TEST(DeviceBlasV2TrsmTest, CxMatchesStdComplex) {
    using T = TypeParam;
    if constexpr (!is_cplx<T>) {
        GTEST_SKIP() << "complex only";
    } else {
        using Cx = sycl_device::Cx<real_of<T>>;
        const Case c{Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit};
        const auto P = make_problem<T>(c, 13, 59, 51);
        const auto Q = make_problem<T>(c, 7, 6, 53);
        const auto pairs = {std::pair{packed_solve<T, 16, 8, false, 64>(this->dev, P), packed_solve<Cx, 16, 8, false, 64>(this->dev, P)},
                            std::pair{rowfrag_solve<T, 8, 4, 8>(this->dev, Q), rowfrag_solve<Cx, 8, 4, 8>(this->dev, Q)}};
        for (const auto& [s, x] : pairs) {
            ASSERT_EQ(s.size(), x.size());
            long long diff = 0;
            for (std::size_t i = 0; i < s.size(); ++i) diff += !same_bits(s[i], x[i]);
            EXPECT_EQ(diff, 0) << "Cx and std::complex differ";
        }
    }
}

}  // namespace
