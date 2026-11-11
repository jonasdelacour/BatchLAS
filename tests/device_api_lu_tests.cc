// SPDX-License-Identifier: MIT
// device_api_lu_tests: the Blocked tier of the public device LU family (lapack/getrf.hh, getrs.hh,
// gesv.hh, getri.hh, inv.hh; v22 B2) called from a user-style kernel, one matrix per work-group
// of tier_info::wg_size items, against a long-double ?GETF2 oracle and the host blocked routes
// (pinned and reached) with the op's tolerance and info parity.
// Guards (agent-guide section 8): orders straddling NB = 32 (31|32|33, 64|65, 96|97) and the last
// Cta tier (128|129, the with_tier switch launched); ld = rows + 3 and a batch stride past
// ld * cols, the pads poisoned with a large finite value that must survive; complex data with
// nonzero imaginary parts; NoTrans, Trans, ConjTrans; nrhs 1 and off the tile grid; three operand
// kinds (Mat, ItemMat, l3::Global); a singular representative (U's last column zero, so A's is,
// exactly: info = n by construction); 1024-item batches of period-3 data, every item
// bit-identical to its representative; pivots against the oracle on inputs with a 4x pivot
// margin; a non-default tier (NB = 16 on 64 x 64 tiles); with_tier's picks, filters and throws;
// buffer sizes by hand; inv bit-identical to getrf + getri on a copy, A untouched; and a
// deliberate break (a dead item is the ONLY mismatch against the host route).
#include <tests/device_core_common.hh>
#include <tests/test_support/coverage_rows.hh>

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/device/lapack/gesv.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getri.hh>
#include <batchlas/device/lapack/getrs.hh>
#include <batchlas/device/lapack/inv.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/gesv/choice.hh>
#include <src/ops/getrf/choice.hh>
#include <src/ops/getri/choice.hh>
#include <src/ops/getrs/choice.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace device_core_test;
using batchlas::Transpose;
using batchlas::verify::Check;

namespace {

using Tier = batchlas::device::tier::Blocked<32>;
template <class T> constexpr int kWG = batchlas::device::tier_info<batchlas::device::routine::getrf, Tier, T>::wg_size;
static_assert(kWG<float> == 256 && kWG<std::complex<double>> == 256);
constexpr int kPeriod = 3;
constexpr int kSaturating = 1024;
constexpr Transpose kNT = Transpose::NoTrans, kTr = Transpose::Trans, kCT = Transpose::ConjTrans;

template <class T> T poison() { return T(real_of<T>(1234.5)); }
template <class T> LD<T> cj(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> long double cabs1(LD<T> v) {
    if constexpr (is_cplx<T>) return std::fabs(v.real()) + std::fabs(v.imag());
    else return std::fabs(v);
}
template <class T> long double sq(LD<T> v) {
    const long double m = std::abs(v);
    return m * m;
}
std::int32_t* i32(std::int64_t* p) { return reinterpret_cast<std::int32_t*>(p); }

// items column-major rows x cols matrices, ld = rows + 3, stride = ld * cols + 5; every element
// outside the matrices holds poison().
template <class T> struct Batch {
    int rows, cols, ld, items;
    std::size_t stride;
    std::vector<T> h;
    Batch(int r, int c, int n_items)
        : rows(r), cols(c), ld(r + 3), items(n_items), stride(std::size_t(r + 3) * c + 5), h(stride * n_items, poison<T>()) {}
    std::size_t off(int q, int i, int j) const { return std::size_t(q) * stride + i + std::size_t(j) * ld; }
    T& at(int q, int i, int j) { return h[off(q, i, j)]; }
    const T& at(int q, int i, int j) const { return h[off(q, i, j)]; }
    LD<T> w(int q, int i, int j) const { return LD<T>(at(q, i, j)); }
    bool inside(std::size_t e) const {
        const std::size_t r = e % stride;
        return r < std::size_t(ld) * cols && int(r % ld) < rows;
    }
    void replicate() {  // items >= kPeriod become copies of their representative q % kPeriod
        for (int q = kPeriod; q < items; ++q)
            std::copy_n(h.begin() + std::ptrdiff_t(stride) * (q % kPeriod), stride, h.begin() + std::ptrdiff_t(stride) * q);
    }
};

template <class T> struct DevBatch {
    T* p;
    int rows, cols, ld, items;
    std::size_t stride;
    T* item(int q) const { return p + std::ptrdiff_t(q) * std::ptrdiff_t(stride); }
    auto mat(int q) const { return batchlas::device::mat_wide(item(q), rows, cols, ld); }
    batchlas::device::l3::Global<T> global(int q) const { return {item(q), ld, rows, cols}; }
    batchlas::KernelMatrixView<T, batchlas::MatrixFormat::Dense> kview() const {
        return batchlas::KernelMatrixView<T, batchlas::MatrixFormat::Dense>(p, rows, cols, ld, int(stride), items);
    }
    batchlas::MatrixView<T, batchlas::MatrixFormat::Dense> view() const {
        return batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>(p, rows, cols, ld, int(stride), items);
    }
};
template <class T> DevBatch<T> to_dev(Dev& d, const Batch<T>& b) { return {d.up(b.h), b.rows, b.cols, b.ld, b.items, b.stride}; }
template <class T> Batch<T> downloaded(Dev& d, const Batch<T>& like, const T* p) {
    Batch<T> r = like;
    r.h = d.down(p, like.h.size());
    return r;
}

// A = P0 L0 U0, |L0(i, j)| <= 0.17 below a unit diagonal, |U0(j, j)| >= 1: partial pivoting picks
// P0's rows with a 4x margin (cabs1 included). U0's off-diagonal is scaled by min(1, 2 / sqrt(n)):
// unscaled, cond(U0) grows exponentially (complex n = 129: ~1e9) and float LU loses the pivots.
// `singular` zeroes U0's last column: A's last column is then exactly zero, stays so through
// every update, and info = n.
template <class T> std::vector<T> lu_input(int n, std::uint64_t seed, bool singular) {
    using R = real_of<T>;
    std::vector<LD<T>> L(std::size_t(n) * n, LD<T>(0)), U(std::size_t(n) * n, LD<T>(0));
    const long double us = std::min(1.0L, 2.0L / std::sqrt(static_cast<long double>(n)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            L[i + std::size_t(j) * n] = i == j ? LD<T>(1) : LD<T>(val<T>(seed + 7u * i + 1000u * j)) * (long double)0.17;
    for (int j = 0; j < n; ++j)
        for (int c = j; c < n; ++c) {
            LD<T> u = LD<T>(val<T>(seed + 99991u + 13u * j + 1777u * c));
            if (c == j) u = LD<T>(T(R(1.5) + R(0.5) * uval<R>(seed + 31u * j))) + u * (long double)0.1;
            else u *= us;
            U[j + std::size_t(c) * n] = (singular && c == n - 1) ? LD<T>(0) : u;
        }
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[mix(seed + 555u + i) % std::uint64_t(i + 1)]);
    std::vector<T> a(std::size_t(n) * n);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < n; ++c) {
            LD<T> s = 0;
            for (int j = 0; j < n; ++j) s += L[i + std::size_t(j) * n] * U[j + std::size_t(c) * n];
            a[perm[i] + std::size_t(c) * n] = static_cast<T>(s);
        }
    return a;
}
template <class T> void fill_lu(Batch<T>& a, std::uint64_t seed, bool singular_rep) {
    for (int r = 0; r < kPeriod && r < a.items; ++r) {
        const auto m = lu_input<T>(a.rows, seed + 104729u * r, singular_rep && r == 1);
        for (int j = 0; j < a.cols; ++j)
            for (int i = 0; i < a.rows; ++i) a.at(r, i, j) = m[i + std::size_t(j) * a.rows];
    }
    a.replicate();
}
template <class T> void fill_vals(Batch<T>& b, std::uint64_t seed) {
    for (int r = 0; r < kPeriod && r < b.items; ++r)
        for (int j = 0; j < b.cols; ++j)
            for (int i = 0; i < b.rows; ++i) b.at(r, i, j) = val<T>(seed * 1000003u + b.off(r, i, j));
    b.replicate();
}

// LAPACK ?GETF2 in long double, cabs1 metric: item q's 1-based pivots and info.
template <class T> std::pair<std::vector<int>, int> oracle(const Batch<T>& A0, int q) {
    const int n = A0.rows;
    std::vector<LD<T>> a(std::size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a[i + std::size_t(j) * n] = A0.w(q, i, j);
    auto at = [&](int i, int j) -> LD<T>& { return a[i + std::size_t(j) * n]; };
    std::vector<int> ipiv(n, 0);
    int info = 0;
    for (int k = 0; k < n; ++k) {
        int p = k;
        long double best = -1;
        for (int i = k; i < n; ++i)
            if (cabs1<T>(at(i, k)) > best) best = cabs1<T>(at(i, k)), p = i;
        ipiv[k] = p + 1;
        for (int c = 0; c < n; ++c) std::swap(at(k, c), at(p, c));
        const LD<T> dk = at(k, k);
        if (dk == LD<T>(0)) {
            if (info == 0) info = k + 1;
            continue;
        }
        for (int i = k + 1; i < n; ++i) at(i, k) /= dk;
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < n; ++i) at(i, j) -= at(i, k) * at(k, j);
    }
    return {ipiv, info};
}

template <class T> long double fro2(const Batch<T>& a, int q) {
    long double s = 0;
    for (int j = 0; j < a.cols; ++j)
        for (int i = 0; i < a.rows; ++i) s += sq<T>(a.w(q, i, j));
    return s;
}
// ||P A - L U||_F / ||A||_F of item q (F: getrf's factor, piv: its pivots).
template <class T> double factor_residual(const Batch<T>& A0, const Batch<T>& F, int q, const std::int32_t* piv) {
    const int n = A0.rows;
    std::vector<LD<T>> pa(std::size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) pa[i + std::size_t(j) * n] = A0.w(q, i, j);
    for (int k = 0; k < n; ++k) {
        const int p = piv[k] - 1;
        if (p < k || p >= n) return std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j < n; ++j) std::swap(pa[k + std::size_t(j) * n], pa[p + std::size_t(j) * n]);
    }
    long double num = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            LD<T> s = 0;
            for (int k = 0; k <= std::min(i, j); ++k) s += (k == i ? LD<T>(1) : F.w(q, i, k)) * F.w(q, k, j);
            num += sq<T>(pa[i + std::size_t(j) * n] - s);
        }
    return static_cast<double>(std::sqrt(num / fro2(A0, q)));
}
// ||op(A) X - B||_F / (||A||_F ||X||_F) of item q.
template <class T> double solve_residual(const Batch<T>& A0, const Batch<T>& X, const Batch<T>& B0, int q, Transpose tr) {
    const int n = A0.rows;
    long double num = 0;
    for (int c = 0; c < B0.cols; ++c)
        for (int i = 0; i < n; ++i) {
            LD<T> s = -B0.w(q, i, c);
            for (int k = 0; k < n; ++k) {
                const LD<T> a = tr == kNT ? A0.w(q, i, k) : tr == kTr ? A0.w(q, k, i) : cj<T>(A0.w(q, k, i));
                s += a * X.w(q, k, c);
            }
            num += sq<T>(s);
        }
    return static_cast<double>(std::sqrt(num / (fro2(A0, q) * fro2(X, q))));
}
// ||A C - I||_F / (||A||_F ||C||_F) of item q.
template <class T> double inverse_residual(const Batch<T>& A0, const Batch<T>& C, int q) {
    const int n = A0.rows;
    long double num = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            LD<T> s = i == j ? LD<T>(-1) : LD<T>(0);
            for (int k = 0; k < n; ++k) s += A0.w(q, i, k) * C.w(q, k, j);
            num += sq<T>(s);
        }
    return static_cast<double>(std::sqrt(num / (fro2(A0, q) * fro2(C, q))));
}
// ||Y - Z||_F / ||Z||_F of item q.
template <class T> double rel_diff(const Batch<T>& Y, const Batch<T>& Z, int q) {
    long double num = 0;
    for (int j = 0; j < Z.cols; ++j)
        for (int i = 0; i < Z.rows; ++i) num += sq<T>(Y.w(q, i, j) - Z.w(q, i, j));
    return static_cast<double>(std::sqrt(num / fro2(Z, q)));
}

// Items (not `skip`) whose bytes, pads included, differ from their representative's.
template <class T> int differ_from_rep(const Batch<T>& b, int skip = -1) {
    int bad = 0;
    for (int q = kPeriod; q < b.items; ++q)
        if (q != skip)
            bad += std::memcmp(b.h.data() + std::size_t(q) * b.stride, b.h.data() + std::size_t(q % kPeriod) * b.stride, sizeof(T) * b.stride) != 0;
    return bad;
}
template <class V> int differ_from_rep(const std::vector<V>& v, int per, int skip = -1) {
    int bad = 0;
    for (std::size_t e = std::size_t(kPeriod) * per; e < v.size(); ++e) {
        const int q = int(e / per);
        if (q != skip) bad += v[e] != v[std::size_t(q % kPeriod) * per + e % per];
    }
    return bad;
}
// Elements outside the matrices that lost their poison.
template <class T> int pads_changed(const Batch<T>& before, const Batch<T>& after) {
    int bad = 0;
    for (std::size_t e = 0; e < after.h.size(); ++e) bad += !before.inside(e) && !same_bits(before.h[e], after.h[e]);
    return bad;
}

bool host_gpu(Dev& d) { return d.ctx->device().type == DeviceType::GPU; }

// The user's kernel for tier Tr of routine R: items work-groups of tier_info::wg_size items at
// sub-group 32, the tier's local bytes, f(ex, b, workspace slice of b).
template <class R, class Tr, class T, class F> void launch(Dev& d, int items, int n, std::byte* gws, F f) {
    using I = batchlas::device::tier_info<R, Tr, T>;
    constexpr int WG = I::wg_size;
    const batchlas::device::WorkspaceSize ws = I::buffer_size(n);
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<std::byte, 1> lm(sycl::range<1>(std::max<std::size_t>(ws.local_bytes, 16)), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(items) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const batchlas::device::WorkGroup<WG, 32> ex{it};
             const int b = static_cast<int>(it.get_group_linear_id());
             const batchlas::device::Workspace w{lm.template get_multi_ptr<sycl::access::decorated::yes>(),
                                   gws == nullptr ? nullptr : batchlas::device::global_slice(gws, ws, b)};
             f(ex, b, w);
         });
     }).wait_and_throw();
}

// getrf<Tr> in place on Mat operands; `dead` passes live = false and keeps its info slot.
template <class Tr, class T> void device_getrf(Dev& d, const DevBatch<T>& a, std::int32_t* piv, std::int32_t* info, int dead = -1) {
    const int n = a.cols;
    launch<batchlas::device::routine::getrf, Tr, T>(d, a.items, n, nullptr, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        const std::int32_t r = batchlas::device::getrf<Tr>(ex, a.mat(b), piv + std::ptrdiff_t(b) * n, w, b != dead);
        if (b != dead && ex.leader()) info[b] = r;
    });
}
// getrs<TR> on ItemMat operands.
template <Transpose TR, class T> void device_getrs(Dev& d, const DevBatch<T>& a, const DevBatch<T>& b, const std::int32_t* piv) {
    const auto av = a.kview(), bv = b.kview();
    const int n = a.cols;
    launch<batchlas::device::routine::getrs, Tier, T>(d, a.items, n, nullptr, [=](const auto& ex, int q, const batchlas::device::Workspace& w) {
        (void)batchlas::device::getrs<TR, Tier>(ex, batchlas::device::item(av, q), batchlas::device::item(bv, q), piv + std::ptrdiff_t(q) * n, w);
    });
}
template <class T> void device_getrs(Dev& d, Transpose tr, const DevBatch<T>& a, const DevBatch<T>& b, const std::int32_t* piv) {
    if (tr == kNT) device_getrs<kNT>(d, a, b, piv);
    else if (tr == kTr) device_getrs<kTr>(d, a, b, piv);
    else device_getrs<kCT>(d, a, b, piv);
}
// gesv on Mat operands.
template <class T> void device_gesv(Dev& d, const DevBatch<T>& a, const DevBatch<T>& b, std::int32_t* piv, std::int32_t* info) {
    const int n = a.cols;
    launch<batchlas::device::routine::gesv, Tier, T>(d, a.items, n, nullptr, [=](const auto& ex, int q, const batchlas::device::Workspace& w) {
        const std::int32_t r = batchlas::device::gesv<Tier>(ex, a.mat(q), b.mat(q), piv + std::ptrdiff_t(q) * n, w);
        if (ex.leader()) info[q] = r;
    });
}
// getri on l3::Global operands.
template <class T> void device_getri(Dev& d, const DevBatch<T>& a, const DevBatch<T>& c, const std::int32_t* piv, std::int32_t* info) {
    const int n = a.cols;
    launch<batchlas::device::routine::getri, Tier, T>(d, a.items, n, nullptr, [=](const auto& ex, int q, const batchlas::device::Workspace& w) {
        const std::int32_t r = batchlas::device::getri<Tier>(ex, a.global(q), c.global(q), piv + std::ptrdiff_t(q) * n, w);
        if (ex.leader()) info[q] = r;
    });
}
template <class T> void device_inv(Dev& d, const DevBatch<T>& a, const DevBatch<T>& c, std::byte* gws, std::int32_t* info) {
    launch<batchlas::device::routine::inv, Tier, T>(d, a.items, a.cols, gws, [=](const auto& ex, int q, const batchlas::device::Workspace& w) {
        const std::int32_t r = batchlas::device::inv<Tier>(ex, a.mat(q), c.mat(q), w);
        if (ex.leader()) info[q] = r;
    });
}

// The host op under a strict pin of its blocked route, which must be reached.
template <class C, class Run> void host_blocked(Dev& d, const char* op, Run run) {
    const auto rows = test_support::reached_rows([&] {
        const batchlas::select::ScopedPin<C> pin(op, "blocked", batchlas::select::StrictPin{});
        run();
        d.ctx->wait();
    });
    bool reached = false;
    for (const auto& r : rows) reached = reached || (r.op == op && r.spelling == "blocked");
    EXPECT_TRUE(reached) << op << ": the pinned host blocked route was not reached";
}

// The device factor on its own: every item bit-identical to its representative (factor, pads,
// pivots, info), the representatives' pivots and info the oracle's, the backward error bounded.
template <class T>
void check_factor(const Batch<T>& A0, const Batch<T>& F, const std::vector<std::int32_t>& P, const std::vector<std::int32_t>& I,
                  const std::string& what) {
    const int n = A0.cols;
    EXPECT_EQ(differ_from_rep(F), 0) << what << " n=" << n << ": factors differ from their representative";
    EXPECT_EQ(differ_from_rep(P, n) + differ_from_rep(I, 1), 0) << what << " n=" << n << ": pivots or info differ";
    EXPECT_EQ(pads_changed(A0, F), 0) << what << " n=" << n;
    double worst = 0;
    for (int r = 0; r < kPeriod; ++r) {
        const auto [hp, hi] = oracle(A0, r);
        int pbad = 0;
        for (int k = 0; k < n; ++k) pbad += P[std::size_t(r) * n + k] != hp[k];
        EXPECT_EQ(pbad, 0) << what << " n=" << n << " rep " << r << ": pivots differ from the oracle";
        EXPECT_EQ(I[r], hi) << what << " n=" << n << " rep " << r;
        EXPECT_EQ(I[r], r == 1 ? n : 0) << what << " n=" << n << " rep " << r << " info by construction";
        worst = batchlas::verify::nanmax(worst, factor_residual(A0, F, r, P.data() + std::size_t(r) * n));
    }
    EXPECT_VERIFY(T, Check::factorization, n, worst) << what;
}

// Items where the device and host factors disagree: pivots, info, or a factor difference past
// the factorization bound.
template <class T>
std::vector<int> lu_mismatches(const Batch<T>& F, const std::vector<std::int32_t>& P, const std::vector<std::int32_t>& I, const Batch<T>& H,
                               const std::vector<std::int32_t>& PH, const std::vector<std::int32_t>& IH) {
    const int n = F.cols;
    std::vector<int> bad;
    for (int q = 0; q < F.items; ++q) {
        bool same = I[q] == IH[q] && batchlas::verify::within<T>(Check::factorization, n, rel_diff(F, H, q));
        for (int k = 0; k < n; ++k) same = same && P[std::size_t(q) * n + k] == PH[std::size_t(q) * n + k];
        if (!same) bad.push_back(q);
    }
    return bad;
}

template <class T> class DeviceApiLuTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceApiLuTest, AllTypes);

// getrf Blocked<32> across the NB and Cta boundaries; on a GPU also the host blocked route.
TYPED_TEST(DeviceApiLuTest, GetrfBlockedMatchesTheOracleAndTheHostBlockedRoute) {
    using T = TypeParam;
    auto& d = this->dev;
    for (int n : {2, 31, 32, 33, 64, 65, 97, 129}) {
        const int items = n <= 65 ? kSaturating : 96;
        Batch<T> A0(n, n, items);
        fill_lu(A0, 0x10u + n, true);
        const auto a = to_dev(d, A0), ah = to_dev(d, A0);
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int64_t* pivh = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infoh = d.template alloc<std::int32_t>(items, -12345);
        device_getrf<Tier>(d, a, i32(piv), info);
        const Batch<T> F = downloaded(d, A0, a.p);
        const auto P = d.down(i32(piv), std::size_t(n) * items);
        const auto I = d.down(info, items);
        check_factor(A0, F, P, I, "Blocked<32>");
        if (!host_gpu(d)) continue;
        host_blocked<batchlas::ops::getrf::GetrfChoice>(d, "getrf", [&] {
            (void)batchlas::getrf(*d.ctx, ah.view(), Span<std::int64_t>(pivh, std::size_t(n) * items), Span<std::int32_t>(infoh, items));
        });
        const Batch<T> H = downloaded(d, A0, ah.p);
        const auto bad = lu_mismatches(F, P, I, H, d.down(i32(pivh), std::size_t(n) * items), d.down(infoh, items));
        EXPECT_TRUE(bad.empty()) << "n=" << n << ": " << bad.size() << " items differ from the host blocked route, first " << (bad.empty() ? -1 : bad[0]);
    }
}

// with_tier over {Cta<128>, Blocked<32>} launches Cta at n = 128 (where it fits) and Blocked at
// 129; a non-default Blocked<16, 64 x 64 tiles> factors as the oracle says.
TYPED_TEST(DeviceApiLuTest, WithTierSwitchesToBlockedAndAnyConfigurationFactors) {
    using T = TypeParam;
    auto& d = this->dev;
    using List = batchlas::device::tier::list<batchlas::device::tier::Cta<128>, Tier>;
    const auto lim = batchlas::device::DeviceLimits::of(d.q().get_device());
    for (int n : {128, 129}) {
        Batch<T> A0(n, n, 6);
        fill_lu(A0, 0x77u + n, true);
        const auto a = to_dev(d, A0);
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * 6, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(6, -12345);
        std::string picked;
        batchlas::device::with_tier<batchlas::device::routine::getrf, T, List>(lim, n, [&]<class Tr>(Tr) {
            picked = batchlas::device::getrf_detail::is_blocked_v<Tr> ? "Blocked" : "Cta";
            device_getrf<Tr>(d, a, i32(piv), info);
        });
        const bool cta = n <= 128 && batchlas::device::getrf_buffer_size<batchlas::device::tier::Cta<128>, T>(n).local_bytes <= lim.local_mem_bytes;
        EXPECT_EQ(picked, cta ? "Cta" : "Blocked") << "n=" << n;
        check_factor(A0, downloaded(d, A0, a.p), d.down(i32(piv), std::size_t(n) * 6), d.down(info, 6), "with_tier " + picked);
    }
    using Tier16 = batchlas::device::tier::Blocked<16, batchlas::device::l3::cfg::RegTiled<T, 64, 64, 8, 4, 4>>;
    static_assert(batchlas::device::tier_info<batchlas::device::routine::getrf, Tier16, T>::wg_size == 256);
    const int n = 97;
    Batch<T> A0(n, n, 6);
    fill_lu(A0, 0x16u, true);
    const auto a = to_dev(d, A0);
    std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * 6, -1);
    std::int32_t* info = d.template alloc<std::int32_t>(6, -12345);
    device_getrf<Tier16>(d, a, i32(piv), info);
    check_factor(A0, downloaded(d, A0, a.p), d.down(i32(piv), std::size_t(n) * 6), d.down(info, 6), "Blocked<16, 64x64>");
}

// getrs<TR> on the device factor: the solve bound for the device and the host blocked getrs on
// the same factor and pivots; the 1024-item case bit-identical per representative; pads survive.
TYPED_TEST(DeviceApiLuTest, GetrsBlockedSolvesAsTheHostBlockedRoute) {
    using T = TypeParam;
    auto& d = this->dev;
    const int shapes[3][3] = {{33, 37, kSaturating}, {70, 1, 48}, {70, 5, 48}};
    for (const auto& s : shapes) {
        const int n = s[0], nrhs = s[1], items = s[2];
        Batch<T> A0(n, n, items), B0(n, nrhs, items);
        fill_lu(A0, 0x2000u + n, false);
        fill_vals(B0, 0x3000u + nrhs);
        const auto a = to_dev(d, A0);
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
        device_getrf<Tier>(d, a, i32(piv), info);
        for (Transpose tr : {kNT, kTr, kCT}) {
            const auto b = to_dev(d, B0), bh = to_dev(d, B0);
            device_getrs(d, tr, a, b, i32(piv));
            const Batch<T> X = downloaded(d, B0, b.p);
            const std::string what = "getrs n=" + std::to_string(n) + " nrhs=" + std::to_string(nrhs) + " tr=" + std::to_string(int(tr));
            EXPECT_EQ(differ_from_rep(X), 0) << what;
            EXPECT_EQ(pads_changed(B0, X), 0) << what;
            double worst = 0;
            for (int r = 0; r < kPeriod; ++r) worst = batchlas::verify::nanmax(worst, solve_residual(A0, X, B0, r, tr));
            EXPECT_VERIFY(T, Check::solve, n, worst) << what;
            if (!host_gpu(d)) continue;
            host_blocked<batchlas::ops::getrs::GetrsChoice>(d, "getrs", [&] {
                (void)batchlas::getrs(*d.ctx, a.view(), bh.view(), Span<std::int64_t>(piv, std::size_t(n) * items), batchlas::GetrsOptions{tr});
            });
            const Batch<T> XH = downloaded(d, B0, bh.p);
            double worst_h = 0;
            for (int r = 0; r < kPeriod; ++r) worst_h = batchlas::verify::nanmax(worst_h, solve_residual(A0, XH, B0, r, tr));
            EXPECT_VERIFY(T, Check::solve, n, worst_h) << what << " (host)";
        }
    }
}

// gesv: the factor half as getrf, X within the solve bound (non-singular representatives), and
// pivots and info equal to the host blocked gesv's on every item.
TYPED_TEST(DeviceApiLuTest, GesvBlockedMatchesTheHostBlockedRoute) {
    using T = TypeParam;
    auto& d = this->dev;
    for (int n : {31, 65, 129}) {
        const int nrhs = 5, items = n == 65 ? kSaturating : 48;
        Batch<T> A0(n, n, items), B0(n, nrhs, items);
        fill_lu(A0, 0x4000u + n, true);
        fill_vals(B0, 0x5000u + n);
        const auto a = to_dev(d, A0), ah = to_dev(d, A0), b = to_dev(d, B0), bh = to_dev(d, B0);
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int64_t* pivh = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infoh = d.template alloc<std::int32_t>(items, -12345);
        device_gesv(d, a, b, i32(piv), info);
        const auto P = d.down(i32(piv), std::size_t(n) * items);
        const auto I = d.down(info, items);
        check_factor(A0, downloaded(d, A0, a.p), P, I, "gesv");
        const Batch<T> X = downloaded(d, B0, b.p);
        EXPECT_EQ(differ_from_rep(X), 0) << "gesv n=" << n;
        EXPECT_EQ(pads_changed(B0, X), 0) << "gesv n=" << n;
        const double worst = batchlas::verify::nanmax(solve_residual(A0, X, B0, 0, kNT), solve_residual(A0, X, B0, 2, kNT));
        EXPECT_VERIFY(T, Check::solve, n, worst) << "gesv n=" << n;
        if (!host_gpu(d)) continue;
        host_blocked<batchlas::ops::gesv::GesvChoice>(d, "gesv", [&] {
            const std::size_t bytes = batchlas::gesv_buffer_size(*d.ctx, ah.view(), bh.view());
            std::byte* ws = d.template alloc<std::byte>(std::max<std::size_t>(bytes, 1), std::byte{0});
            (void)batchlas::gesv(*d.ctx, ah.view(), bh.view(), Span<std::int64_t>(pivh, std::size_t(n) * items), Span<std::byte>(ws, bytes),
                                 Span<std::int32_t>(infoh, items));
        });
        EXPECT_EQ(d.down(i32(pivh), std::size_t(n) * items), P) << "gesv n=" << n << ": pivots differ from the host route";
        EXPECT_EQ(d.down(infoh, items), I) << "gesv n=" << n << ": info differs from the host route";
        const Batch<T> XH = downloaded(d, B0, bh.p);
        EXPECT_VERIFY(T, Check::solve, n, batchlas::verify::nanmax(solve_residual(A0, XH, B0, 0, kNT), solve_residual(A0, XH, B0, 2, kNT)))
            << "gesv n=" << n << " (host)";
    }
}

// getri out of place on the device factor (l3::Global operands): A untouched, C within the solve
// bound, info (= n for the singular representative) equal to the host blocked getri's.
TYPED_TEST(DeviceApiLuTest, GetriBlockedMatchesTheHostBlockedRoute) {
    using T = TypeParam;
    auto& d = this->dev;
    for (int n : {32, 33, 97}) {
        const int items = n == 33 ? kSaturating : 48;
        Batch<T> A0(n, n, items), C0(n, n, items);
        fill_lu(A0, 0x6000u + n, true);
        const auto a = to_dev(d, A0), c = to_dev(d, C0), ch = to_dev(d, C0);
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infoc = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infoh = d.template alloc<std::int32_t>(items, -12345);
        device_getrf<Tier>(d, a, i32(piv), info);
        const auto LU = d.down(a.p, A0.h.size());
        device_getri(d, a, c, i32(piv), infoc);
        EXPECT_EQ(std::memcmp(LU.data(), d.down(a.p, A0.h.size()).data(), sizeof(T) * LU.size()), 0) << "getri n=" << n << ": A changed";
        const Batch<T> C = downloaded(d, C0, c.p);
        const auto IC = d.down(infoc, items);
        EXPECT_EQ(differ_from_rep(C), 0) << "getri n=" << n;
        EXPECT_EQ(differ_from_rep(IC, 1), 0) << "getri n=" << n;
        EXPECT_EQ(pads_changed(C0, C), 0) << "getri n=" << n;
        for (int r = 0; r < kPeriod; ++r) EXPECT_EQ(IC[r], r == 1 ? n : 0) << "getri n=" << n << " rep " << r;
        EXPECT_VERIFY(T, Check::solve, n, batchlas::verify::nanmax(inverse_residual(A0, C, 0), inverse_residual(A0, C, 2))) << "getri n=" << n;
        if (!host_gpu(d)) continue;
        host_blocked<batchlas::ops::getri::GetriChoice>(d, "getri", [&] {
            (void)batchlas::getri(*d.ctx, a.view(), ch.view(), Span<std::int64_t>(piv, std::size_t(n) * items), Span<std::int32_t>(infoh, items));
        });
        EXPECT_EQ(d.down(infoh, items), IC) << "getri n=" << n << ": info differs from the host route";
        const Batch<T> CH = downloaded(d, C0, ch.p);
        EXPECT_VERIFY(T, Check::solve, n, batchlas::verify::nanmax(inverse_residual(A0, CH, 0), inverse_residual(A0, CH, 2)))
            << "getri n=" << n << " (host)";
    }
}

// inv is getrf on a copy in the global workspace, then getri: bit-identical to that composition
// run by the caller on its own copy (ld = n + 3 against the workspace's n), A untouched.
TYPED_TEST(DeviceApiLuTest, InvBlockedIsGetrfThenGetriOnACopy) {
    using T = TypeParam;
    auto& d = this->dev;
    for (int n : {40, 100}) {
        const int items = 12;
        Batch<T> A0(n, n, items), C0(n, n, items);
        fill_lu(A0, 0x7000u + n, true);
        const auto a = to_dev(d, A0), w = to_dev(d, A0), c = to_dev(d, C0), cw = to_dev(d, C0);
        const batchlas::device::WorkspaceSize ws = batchlas::device::inv_buffer_size<Tier, T>(n);
        std::byte* gws = d.template alloc<std::byte>(ws.global_bytes * items, std::byte{0x5a});
        std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
        std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infow = d.template alloc<std::int32_t>(items, -12345);
        std::int32_t* infoc = d.template alloc<std::int32_t>(items, -12345);
        device_inv(d, a, c, gws, info);
        device_getrf<Tier>(d, w, i32(piv), infow);
        device_getri(d, w, cw, i32(piv), infoc);
        const Batch<T> A1 = downloaded(d, A0, a.p), C = downloaded(d, C0, c.p), CW = downloaded(d, C0, cw.p);
        EXPECT_EQ(std::memcmp(A1.h.data(), A0.h.data(), sizeof(T) * A0.h.size()), 0) << "inv n=" << n << ": A changed";
        EXPECT_EQ(std::memcmp(C.h.data(), CW.h.data(), sizeof(T) * C.h.size()), 0) << "inv n=" << n << ": not getrf + getri bit for bit";
        const auto I = d.down(info, items);
        EXPECT_EQ(I, d.down(infow, items)) << "inv n=" << n << ": info is not getrf's";
        for (int r = 0; r < kPeriod; ++r) EXPECT_EQ(I[r], r == 1 ? n : 0) << "inv n=" << n << " rep " << r;
        EXPECT_VERIFY(T, Check::solve, n, batchlas::verify::nanmax(inverse_residual(A0, C, 0), inverse_residual(A0, C, 2))) << "inv n=" << n;
    }
}

// The deliberate break: a matrix the device op did not factor (live = false) is the ONLY item
// that differs from the host blocked route, so the comparison can fail.
TYPED_TEST(DeviceApiLuTest, DeliberateBreakADeadItemIsTheOnlyMismatch) {
    using T = TypeParam;
    auto& d = this->dev;
    if (!host_gpu(d)) GTEST_SKIP() << "the host blocked route is GPU only";
    const int n = 40, items = kSaturating, dead = 517;
    Batch<T> A0(n, n, items);
    fill_lu(A0, 0xdeadu, true);
    const auto a = to_dev(d, A0), ah = to_dev(d, A0);
    std::int64_t* piv = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
    std::int64_t* pivh = d.template alloc<std::int64_t>(std::size_t(n) * items, -1);
    std::int32_t* info = d.template alloc<std::int32_t>(items, -12345);
    std::int32_t* infoh = d.template alloc<std::int32_t>(items, -12345);
    device_getrf<Tier>(d, a, i32(piv), info, dead);
    host_blocked<batchlas::ops::getrf::GetrfChoice>(d, "getrf", [&] {
        (void)batchlas::getrf(*d.ctx, ah.view(), Span<std::int64_t>(pivh, std::size_t(n) * items), Span<std::int32_t>(infoh, items));
    });
    EXPECT_EQ(lu_mismatches(downloaded(d, A0, a.p), d.down(i32(piv), std::size_t(n) * items), d.down(info, items), downloaded(d, A0, ah.p),
                            d.down(i32(pivh), std::size_t(n) * items), d.down(infoh, items)),
              std::vector<int>{dead});
}

// Host only: every LU routine past getrf offers Blocked<32> for any n >= 1, with_tier's filters
// and throws, and the plans by hand.
template <class R> std::string picked(const batchlas::device::DeviceLimits& lim, int n) {
    std::string s;
    try {
        batchlas::device::with_tier<R, float>(lim, n, [&]<class Tr>(Tr) { s = "Blocked<" + std::to_string(Tr::nb) + ">"; });
    } catch (const batchlas::invalid_argument& e) {
        s = e.what();
    }
    return s;
}
template <class R> std::string picked_double(const batchlas::device::DeviceLimits& lim, int n) {
    std::string s;
    try {
        batchlas::device::with_tier<R, double>(lim, n, [&]<class Tr>(Tr) { s = "Blocked<" + std::to_string(Tr::nb) + ">"; });
    } catch (const batchlas::invalid_argument& e) {
        s = e.what();
    }
    return s;
}
template <class R> void expect_any_n(const std::string& name) {
    const batchlas::device::DeviceLimits full{99 * 1024, {32}, 1024, true};
    EXPECT_EQ(picked<R>(full, 1), "Blocked<32>") << name;
    EXPECT_EQ(picked<R>(full, 5000), "Blocked<32>") << name;
    EXPECT_EQ(picked<R>(full, 0), name + ": no device tier admits n=0");
    EXPECT_EQ(picked<R>(batchlas::device::DeviceLimits{99 * 1024, {16}, 1024, true}, 5), name + ": no device tier admits n=5") << "sub-group 32 only";
    EXPECT_EQ(picked<R>(batchlas::device::DeviceLimits{99 * 1024, {32}, 128, true}, 5), name + ": no device tier admits n=5") << "256 items";
    EXPECT_EQ(picked<R>(batchlas::device::DeviceLimits{2 * 1024, {32}, 1024, true}, 5), name + ": no device tier admits n=5") << "2464 local bytes";
    EXPECT_EQ(picked_double<R>(batchlas::device::DeviceLimits{99 * 1024, {32}, 1024, false}, 5), name + ": no device tier admits n=5") << "no fp64";
}

TEST(DeviceApiLuTier, EveryRoutineOffersBlockedForAnyN) {
    expect_any_n<batchlas::device::routine::getrs>("getrs");
    expect_any_n<batchlas::device::routine::gesv>("gesv");
    expect_any_n<batchlas::device::routine::getri>("getri");
    expect_any_n<batchlas::device::routine::inv>("inv");
    static_assert(std::is_same_v<batchlas::device::resolve_tier_t<batchlas::device::routine::gesv, float, batchlas::device::tier::upto<7>>, Tier>);
}

TEST(DeviceApiLuTier, BufferSizeIsThePlanByHand) {
    using batchlas::device::WorkspaceSize;
    // L3 stages 1 * (33 * 8 + 9 * 32) = 552 elements, then 32 real and 32 int pivot slots.
    EXPECT_EQ((batchlas::device::getrf_buffer_size<Tier, float>(500)), (WorkspaceSize{2464, 0, 16}));
    EXPECT_EQ((batchlas::device::getrf_buffer_size<Tier, double>(1)), (WorkspaceSize{4800, 0, 16}));
    EXPECT_EQ((batchlas::device::getrf_buffer_size<Tier, std::complex<float>>(7)), (WorkspaceSize{4672, 0, 16}));
    EXPECT_EQ((batchlas::device::getrf_buffer_size<Tier, std::complex<double>>(4000)), (WorkspaceSize{9216, 0, 16}));
    EXPECT_EQ((batchlas::device::getrs_buffer_size<kCT, Tier, float>(500, 7)), (WorkspaceSize{2464, 0, 16}));
    EXPECT_EQ((batchlas::device::gesv_buffer_size<Tier, double>(9, 2)), (WorkspaceSize{4800, 0, 16}));
    EXPECT_EQ((batchlas::device::getri_buffer_size<Tier, std::complex<float>>(3)), (WorkspaceSize{4672, 0, 16}));
    // inv: the copy (n * n) then n int32 pivots at a 16-byte offset: float n = 10: 400 + 40 -> 448.
    EXPECT_EQ((batchlas::device::inv_buffer_size<Tier, float>(10)), (WorkspaceSize{2464, 448, 16}));
    // complex<double> n = 3: 144 + 12 -> 160.
    EXPECT_EQ((batchlas::device::inv_buffer_size<Tier, std::complex<double>>(3)), (WorkspaceSize{9216, 160, 16}));
    // 64 x 64 tiles of 4 x 4: 65 * 8 + 9 * 64 = 1096 elements.
    using Tier16 = batchlas::device::tier::Blocked<16, batchlas::device::l3::cfg::RegTiled<float, 64, 64, 8, 4, 4>>;
    EXPECT_EQ((batchlas::device::getrf_buffer_size<Tier16, float>(1)), (WorkspaceSize{4640, 0, 16}));
    static_assert(batchlas::device::tier_info<batchlas::device::routine::inv, Tier, float>::wg_size == 256);
    static_assert(!batchlas::device::tier_info<batchlas::device::routine::getri, Tier, float>::admits(0));
}

}  // namespace
