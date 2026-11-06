// SPDX-License-Identifier: MIT
// Public device API of the util family (v22 design C.10), each op called from a user-style kernel:
//   norm<NT>      (lapack/norm.hh)         vs the host norm (a thin launcher): bit-identical
//   lascl         (core lapack_aux.hh)     vs the host lascl (a thin launcher): bit-identical
//   ritz_values   (lapack/ritz_values.hh)  the tier-free form and tier::Blocked<NB> (in-group gemm)
//                                          vs a long-double reference
//   ortho         (lapack/ortho.hh)        CGS2 Cta and Blocked (block CGS2 on the in-group gemm),
//                                          Cholesky / Chol2 / ShiftChol3 Cta (herk + potrf + trsm) vs
//                                          the long-double Gram-Schmidt Q (unique: R's diagonal > 0)
//                                          and, where the host runs the algorithm (GPU), its route
// Guards (agent-guide section 8): batch 1024 of period-3 data, every item bit-identical to its
// representative; ld and batch stride non-natural; complex data with nonzero imaginary parts;
// padding poisoned with a large finite value that no op may read or write; the global workspaces
// all-ones bytes (NaN: a beta = Zero product or an unwritten Gram triangle must never be read);
// norm at both of the host's work-group sizes (m n <= 1024 and above); lascl on both sides of the
// one-step threshold; ortho at the Cta ceiling (k = N), past it (k = N + 1: -2, A untouched), k
// across NB multiples (k = NB + 1, 3 NB, NB < k), m off the 32-row tile grid, two trsm diagonal
// blocks (k = 36); near-dependent columns that one Gram-Schmidt or CholeskyQR pass leaves
// non-orthogonal; a zero column (info = its index + 1, A untouched); workspace sizes against bytes
// written out here. Deliberate breaks this catches (red sets, wave 1 measured): norm reading ld
// padding (Max, all types); ritz_values_quotient with dot for dotc (the complex types); ortho with
// one pass (OrthoNearDependent, all types); ortho Cta without the k > N refusal (OrthoPastCeiling,
// all). Expected red, to record on the GPU gate (wave 2): bcgs2 projecting once
// (OrthoNearDependent Blocked<4> at 1e-4); Chol2 without its second pass (OrthoNearDependent);
// cholqr_pass returning 0 for potrf's info (OrthoCholeskyInfoAndCeiling); the Blocked
// ritz_values reading v.col(j) for v.col(c0 + j) (RitzValuesBlocked, Blocked<3>).
#include <tests/device_core_common.hh>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device/lapack/norm.hh>
#include <batchlas/device/lapack/ortho.hh>
#include <batchlas/device/lapack/ritz_values.hh>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using namespace device_core_test;
using batchlas::verify::Check;

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;

template <class T> T poison() { return T(static_cast<real_of<T>>(1.0e6)); }

// batch m x n matrices at (ld, stride), item b a copy of representative b % kPeriod, padding poisoned.
template <class T> struct Host {
    int m, n, ld, stride;
    std::vector<T> h;
    T& at(int b, int i, int j) { return h[std::size_t(b) * stride + std::size_t(j) * ld + i]; }
    const T& at(int b, int i, int j) const { return h[std::size_t(b) * stride + std::size_t(j) * ld + i]; }
};
template <class T> Host<T> make(int m, int n, std::uint64_t seed) {
    Host<T> a{m, n, m + 3, (m + 3) * n + 5, std::vector<T>(std::size_t((m + 3) * n + 5) * kBatch, poison<T>())};
    for (int b = 0; b < kPeriod; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) a.at(b, i, j) = val<T>(seed * 7919u + std::uint64_t(b) * 1000003u + std::uint64_t(j) * m + i);
    for (int b = kPeriod; b < kBatch; ++b)
        std::memcpy(&a.h[std::size_t(b) * a.stride], &a.h[std::size_t(b % kPeriod) * a.stride], sizeof(T) * a.stride);
    return a;
}
template <class T> Matrix<T, MatrixFormat::Dense> to_matrix(Dev& dev, const Host<T>& a) {
    Matrix<T, MatrixFormat::Dense> M(a.m, a.n, kBatch, a.ld, a.stride);
    dev.q().memcpy(M.data_ptr(), a.h.data(), a.h.size() * sizeof(T)).wait();
    return M;
}
template <class T> std::vector<T> from_matrix(Dev& dev, const Matrix<T, MatrixFormat::Dense>& M, std::size_t count) {
    std::vector<T> h(count);
    dev.q().memcpy(h.data(), M.data_ptr(), count * sizeof(T)).wait();
    return h;
}
template <class T> int bit_twins(const std::vector<T>& h, std::size_t per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (std::size_t i = 0; i < per_item; ++i) bad += !same_bits(h[b * per_item + i], h[(b % kPeriod) * per_item + i]);
    return bad;
}
inline sycl::nd_range<1> one_wg_per_item(int wg) { return sycl::nd_range<1>(std::size_t(kBatch) * wg, std::size_t(wg)); }

template <class T> class DeviceApiUtil : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceApiUtil, AllTypes);

// ---- norm -----------------------------------------------------------------------------------
// The host's launch: one WorkGroup<WG> per item, WG = 32 for m n <= 1024, else 256.
template <NormType NT, int WG, class T> std::vector<real_of<T>> device_norm(Dev& dev, const T* a, const Host<T>& s) {
    auto* out = dev.alloc<real_of<T>>(kBatch, real_of<T>(-1));
    const int m = s.m, n = s.n, ld = s.ld, stride = s.stride;
    dev.q().parallel_for(one_wg_per_item(WG), [=](sycl::nd_item<1> it) {
        const dv::WorkGroup<WG> ex{it};
        const int b = int(it.get_group_linear_id());
        const auto r = dv::norm<NT>(ex, dv::mat(a + std::size_t(b) * stride, m, n, ld));
        if (ex.leader()) out[b] = r;
    }).wait_and_throw();
    return dev.down(out, kBatch);
}
template <class T> long double ref_norm(NormType nt, const Host<T>& a, int b) {
    long double mx = 0, sq = 0, one = 0;
    std::vector<long double> row(a.m, 0.0L);
    for (int j = 0; j < a.n; ++j) {
        long double col = 0;
        for (int i = 0; i < a.m; ++i) {
            const long double v = std::abs(wide(a.at(b, i, j)));
            mx = std::max(mx, v);
            sq += v * v;
            col += v;
            row[i] += v;
        }
        one = std::max(one, col);
    }
    if (nt == NormType::Max) return mx;
    if (nt == NormType::One) return one;
    if (nt == NormType::Inf) return *std::max_element(row.begin(), row.end());
    return std::sqrt(sq);
}
template <NormType NT, int WG, class T> void check_norm(Dev& dev, const Host<T>& a) {
    SCOPED_TRACE(::testing::Message() << "NormType " << int(NT) << " m " << a.m << " n " << a.n << " WG " << WG);
    auto M = to_matrix(dev, a);
    const auto host = norm(*dev.ctx, M.view(), NT);
    const auto got = device_norm<NT, WG>(dev, M.data_ptr(), a);
    int differ = 0;
    for (int b = 0; b < kBatch; ++b) differ += !same_bits(got[b], host[b]);
    EXPECT_EQ(differ, 0) << "device norm is not bit-identical to the host route";
    EXPECT_EQ(bit_twins(got, 1), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const long double ref = ref_norm(NT, a, b);
        worst = std::max(worst, double(std::abs(got[b] - ref) / ref));
    }
    EXPECT_VERIFY(T, Check::blas, a.m * a.n, worst);
}
template <int WG, class T> void check_norms(Dev& dev, const Host<T>& a) {
    check_norm<NormType::Max, WG>(dev, a);
    check_norm<NormType::One, WG>(dev, a);
    check_norm<NormType::Inf, WG>(dev, a);
    check_norm<NormType::Frobenius, WG>(dev, a);
}

TYPED_TEST(DeviceApiUtil, NormIsTheHostRoute) {
    using T = TypeParam;
    check_norms<32>(this->dev, make<T>(13, 11, 1));  // 143 <= 1024: the host's WG 32
    if (this->dev.ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE) < 256) GTEST_SKIP() << "host uses WG 32 only";
    check_norms<256>(this->dev, make<T>(37, 29, 2));  // 1073 > 1024: WG 256
}

// ---- lascl ----------------------------------------------------------------------------------
template <class T> void check_lascl(Dev& dev, const Host<T>& a, T cfrom, T cto) {
    SCOPED_TRACE(::testing::Message() << "cfrom " << cfrom << " cto " << cto);
    auto H = to_matrix(dev, a);
    (void)lascl<MatrixFormat::Dense, T>(*dev.ctx, H.view(), cfrom, cto);
    dev.ctx->wait();
    T* d = dev.up(a.h);
    const int m = a.m, n = a.n, ld = a.ld, stride = a.stride;
    dev.q().parallel_for(one_wg_per_item(64), [=](sycl::nd_item<1> it) {
        const dv::WorkGroup<64> ex{it};
        dv::lascl(ex, cfrom, cto, dv::mat(d + std::size_t(it.get_group_linear_id()) * stride, m, n, ld));
    }).wait_and_throw();
    const auto got = dev.down(d, a.h.size());
    const auto host = from_matrix(dev, H, a.h.size());
    EXPECT_EQ(std::memcmp(got.data(), host.data(), got.size() * sizeof(T)), 0) << "device lascl differs from the host route (pads included)";
    double worst = 0;
    const long double f = static_cast<long double>(cto) / static_cast<long double>(cfrom);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const long double ref = f * static_cast<long double>(a.at(0, i, j));
            worst = std::max(worst, double(std::abs(static_cast<long double>(got[std::size_t(j) * ld + i]) - ref) / std::abs(ref)));
        }
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

TYPED_TEST(DeviceApiUtil, LasclIsTheHostRoute) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "the host lascl is real only";
    } else {
        const auto a = make<T>(19, 7, 3);
        const T bignum = T(1) / dv::safmin<T>();
        check_lascl(this->dev, a, T(3), T(-0.5));       // one step
        check_lascl(this->dev, a, T(1), bignum / T(2));  // cto / cfrom just under bignum: one step
        check_lascl(this->dev, a, T(1), bignum * T(2));  // just over: bignum, then 2
    }
}

// ---- ritz_values ----------------------------------------------------------------------------
// The in-group Level 3 preset of the 128-item work-groups below: 32 x 32 tiles, 4 x 2 per item.
template <class T> using L3Cfg = dv::l3::cfg::RegTiled<T, 32, 32, 8, 4, 2>;
static_assert(L3Cfg<float>::threads == 128 && L3Cfg<std::complex<double>>::threads == 128);
inline std::size_t r16(std::size_t bytes) { return (bytes + 15) / 16 * 16; }
// A global workspace of batch * bytes, every byte 0xFF (a NaN in every T).
inline std::byte* nan_workspace(Dev& dev, std::size_t bytes) {
    return dev.alloc<std::byte>(std::max<std::size_t>(bytes * kBatch, 1), std::byte{0xFF});
}

// The worst error of w against the long-double quotient, relative to sum |v_r| |A_rc| |v_c| / v^H v.
template <class T> double ritz_worst(const std::vector<real_of<T>>& got, const Host<T>& A, const Host<T>& V) {
    const int n = A.m, k = V.n;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int j = 0; j < k; ++j) {
            LD<T> num = 0;
            long double den = 0, scale = 0;
            for (int r = 0; r < n; ++r) {
                LD<T> av = 0;
                long double abs_av = 0;
                for (int c = 0; c < n; ++c) {
                    av += wide(A.at(b, r, c)) * wide(V.at(b, c, j));
                    abs_av += std::abs(wide(A.at(b, r, c))) * std::abs(wide(V.at(b, c, j)));
                }
                const LD<T> vr = wide(V.at(b, r, j));
                if constexpr (is_cplx<T>) num += std::conj(vr) * av;
                else num += vr * av;
                den += std::norm(vr);
                scale += std::abs(vr) * abs_av;
            }
            const long double ref = std::real(num) / den;
            worst = std::max(worst, double(std::abs(got[std::size_t(b) * k + j] - ref) / (scale / den)));
        }
    return worst;
}

TYPED_TEST(DeviceApiUtil, RitzValuesMatchesReference) {
    using T = TypeParam;
    using R = real_of<T>;
    const int n = 33, k = 5;
    const auto A = make<T>(n, n, 4);
    const auto V = make<T>(n, k, 5);
    const T* ad = this->dev.up(A.h);
    const T* vd = this->dev.up(V.h);
    R* w = this->dev.template alloc<R>(std::size_t(kBatch) * k, R(-1));
    const int lda = A.ld, sa = A.stride, ldv = V.ld, sv = V.stride;
    this->dev.q().parallel_for(one_wg_per_item(128), [=](sycl::nd_item<1> it) {
        const dv::WorkGroup<128> ex{it};
        const int b = int(it.get_group_linear_id());
        const int info = dv::ritz_values(ex, dv::mat(ad + std::size_t(b) * sa, n, n, lda), dv::mat(vd + std::size_t(b) * sv, n, k, ldv),
                                         dv::vec(w + std::size_t(b) * k, k));
        if (ex.leader() && info != 0) w[std::size_t(b) * k] = R(-7);
    }).wait_and_throw();
    const auto got = this->dev.down(w, std::size_t(kBatch) * k);
    EXPECT_EQ(bit_twins(got, k), 0);
    EXPECT_VERIFY(T, Check::blas, n, ritz_worst(got, A, V));
}

// The Blocked form: NB columns of V per in-group gemm into the NaN workspace (beta = Zero).
template <class Tier, class T> std::vector<real_of<T>> device_ritz_blocked(Dev& dev, const T* ad, const Host<T>& A, const T* vd, const Host<T>& V) {
    using R = real_of<T>;
    const int n = A.m, k = V.n;
    const auto sz = dv::ritz_values_buffer_size<Tier, T, L3Cfg<T>>(n, k);
    std::byte* gws = nan_workspace(dev, sz.global_bytes);
    R* w = dev.alloc<R>(std::size_t(kBatch) * k, R(-1));
    const int lda = A.ld, sa = A.stride, ldv = V.ld, sv = V.stride;
    dev.q().submit([&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(std::max<std::size_t>(sz.local_bytes, 16)), h);
        h.parallel_for(one_wg_per_item(128), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
            const dv::WorkGroup<128, 32> ex{it};
            const int b = int(it.get_group_linear_id());
            const dv::Workspace ws{slm.template get_multi_ptr<sycl::access::decorated::yes>(), dv::global_slice(gws, sz, b)};
            const int info = dv::ritz_values<Tier, L3Cfg<T>>(ex, dv::mat(ad + std::size_t(b) * sa, n, n, lda),
                                                             dv::mat(vd + std::size_t(b) * sv, n, k, ldv), dv::vec(w + std::size_t(b) * k, k), ws);
            if (ex.leader() && info != 0) w[std::size_t(b) * k] = R(-7);
        });
    }).wait_and_throw();
    return dev.down(w, std::size_t(kBatch) * k);
}

// n off the 32-row tile grid; k = 2 NB + 1 (a one-column tail block) and k = NB - 1 (one block).
TYPED_TEST(DeviceApiUtil, RitzValuesBlocked) {
    using T = TypeParam;
    const int n = 45, k = 7;
    const auto A = make<T>(n, n, 10);
    const auto V = make<T>(n, k, 11);
    const T* ad = this->dev.up(A.h);
    const T* vd = this->dev.up(V.h);
    {
        SCOPED_TRACE("Blocked<3>");
        const auto got = device_ritz_blocked<dv::tier::Blocked<3>>(this->dev, ad, A, vd, V);
        EXPECT_EQ(bit_twins(got, k), 0);
        EXPECT_VERIFY(T, Check::blas, n, ritz_worst(got, A, V));
    }
    {
        SCOPED_TRACE("Blocked<8>");
        const auto got = device_ritz_blocked<dv::tier::Blocked<8>>(this->dev, ad, A, vd, V);
        EXPECT_EQ(bit_twins(got, k), 0);
        EXPECT_VERIFY(T, Check::blas, n, ritz_worst(got, A, V));
    }
}

// ---- ortho ----------------------------------------------------------------------------------
// The long-double CGS2 of the host op, in place on representative b. Gram-Schmidt, CholeskyQR and
// block CGS2 give the same Q in exact arithmetic (the Q of A = Q R with R's diagonal > 0).
template <class T> std::vector<LD<T>> ref_cgs2(const Host<T>& a, int b) {
    const int m = a.m, k = a.n;
    std::vector<LD<T>> q(std::size_t(m) * k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) q[std::size_t(j) * m + i] = wide(a.at(b, i, j));
    std::vector<LD<T>> h(k);
    for (int i = 0; i < k; ++i) {
        LD<T>* ai = &q[std::size_t(i) * m];
        for (int pass = 0; pass < 2 && i > 0; ++pass) {
            for (int c = 0; c < i; ++c) {
                h[c] = 0;
                for (int r = 0; r < m; ++r) {
                    if constexpr (is_cplx<T>) h[c] += std::conj(q[std::size_t(c) * m + r]) * ai[r];
                    else h[c] += q[std::size_t(c) * m + r] * ai[r];
                }
            }
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < i; ++c) ai[r] -= q[std::size_t(c) * m + r] * h[c];
        }
        long double s = 0;
        for (int r = 0; r < m; ++r) s += std::norm(ai[r]);
        const long double nr = std::sqrt(s);
        for (int r = 0; r < m; ++r) ai[r] /= nr;
    }
    return q;
}

// One WorkGroup<128, 32> per item (Cfg::threads for the Level 3 forms; potrf Cta's 32-wide
// sub-group), the global workspace NaN.
template <OrthoAlgorithm AL, class Tier, class Cfg = void, class T> std::vector<int> device_ortho(Dev& dev, T* a, const Host<T>& s, int k) {
    constexpr auto TR = Transpose::NoTrans;
    const auto sz = dv::ortho_buffer_size<TR, AL, Tier, T, Cfg>(k);
    std::byte* gws = nan_workspace(dev, sz.global_bytes);
    auto* info = dev.alloc<int>(kBatch, -99);
    const int m = s.m, ld = s.ld, stride = s.stride;
    dev.q().submit([&](sycl::handler& h) {
        sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(std::max<std::size_t>(sz.local_bytes, 16)), h);
        h.parallel_for(one_wg_per_item(128), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
            const dv::WorkGroup<128, 32> ex{it};
            const int b = int(it.get_group_linear_id());
            const dv::Workspace ws{slm.template get_multi_ptr<sycl::access::decorated::yes>(), dv::global_slice(gws, sz, b)};
            const int r = dv::ortho<TR, AL, Tier, Cfg>(ex, dv::mat(a + std::size_t(b) * stride, m, k, ld), ws);
            if (ex.leader()) info[b] = r;
        });
    }).wait_and_throw();
    return dev.down(info, kBatch);
}

template <class T> double orth_error(const std::vector<T>& q, const Host<T>& s, int b) {
    double worst = 0;
    for (int c = 0; c < s.n; ++c)
        for (int d = 0; d < s.n; ++d) {
            LD<T> g = 0;
            for (int r = 0; r < s.m; ++r) {
                const LD<T> x = wide(q[std::size_t(b) * s.stride + std::size_t(c) * s.ld + r]);
                const LD<T> y = wide(q[std::size_t(b) * s.stride + std::size_t(d) * s.ld + r]);
                if constexpr (is_cplx<T>) g += std::conj(x) * y;
                else g += x * y;
            }
            worst = std::max(worst, double(std::abs(g - LD<T>(c == d ? 1 : 0))));
        }
    return worst;
}
template <class T> int pads_changed(const std::vector<T>& q, const Host<T>& s) {
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (std::size_t e = 0; e < std::size_t(s.stride); ++e) {
            const std::size_t j = e / s.ld, i = e % s.ld;
            if (j < std::size_t(s.n) && i < std::size_t(s.m)) continue;
            bad += !same_bits(q[std::size_t(b) * s.stride + e], poison<T>());
        }
    return bad;
}

template <OrthoAlgorithm AL, class Tier, class Cfg = void, class T> void check_ortho(Dev& dev, const Host<T>& a) {
    SCOPED_TRACE(::testing::Message() << "algo " << int(AL) << " m " << a.m << " k " << a.n);
    T* d = dev.up(a.h);
    const auto info = device_ortho<AL, Tier, Cfg>(dev, d, a, a.n);
    const auto got = dev.down(d, a.h.size());
    int bad_info = 0;
    for (int b = 0; b < kBatch; ++b) bad_info += info[b] != 0;
    EXPECT_EQ(bad_info, 0);
    EXPECT_EQ(bit_twins(got, std::size_t(a.stride)), 0);
    EXPECT_EQ(pads_changed(got, a), 0);
    double fwd = 0, orth = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const auto ref = ref_cgs2(a, b);
        for (int j = 0; j < a.n; ++j)
            for (int i = 0; i < a.m; ++i)
                fwd = std::max(fwd, err(got[std::size_t(b) * a.stride + std::size_t(j) * a.ld + i], ref[std::size_t(j) * a.m + i]));
        orth = std::max(orth, orth_error(got, a, b));
    }
    EXPECT_VERIFY(T, Check::orthogonality, a.m, orth);
    EXPECT_VERIFY(T, Check::factorization, a.m, fwd);
    // Where the host op runs AL (a GPU; elsewhere it forces Householder), against its route.
    if (dev.ctx->device().type != DeviceType::GPU) return;
    auto H = to_matrix(dev, a);
    UnifiedVector<std::byte> hws(ortho_buffer_size(*dev.ctx, H.view(), Transpose::NoTrans, AL));
    (void)ortho(*dev.ctx, H.view(), Transpose::NoTrans, hws.to_span(), AL);
    dev.ctx->wait();
    const auto host = from_matrix(dev, H, a.h.size());
    double vs_host = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int j = 0; j < a.n; ++j)
            for (int i = 0; i < a.m; ++i) {
                const std::size_t e = std::size_t(b) * a.stride + std::size_t(j) * a.ld + i;
                vs_host = std::max(vs_host, err(got[e], wide(host[e])));
            }
    EXPECT_VERIFY(T, Check::factorization, a.m, vs_host);
}

TYPED_TEST(DeviceApiUtil, OrthoCgs2Tiers) {
    using T = TypeParam;
    constexpr auto AL = OrthoAlgorithm::CGS2;
    const auto a = make<T>(40, 12, 6);
    check_ortho<AL, dv::tier::Cta<16>>(this->dev, a);
    check_ortho<AL, dv::tier::Blocked<32>, L3Cfg<T>>(this->dev, a);  // k < NB: one block, cgs2 only
    check_ortho<AL, dv::tier::Blocked<4>, L3Cfg<T>>(this->dev, a);   // k = 3 NB: two projected blocks
    check_ortho<AL, dv::tier::Blocked<11>, L3Cfg<T>>(this->dev, a);  // k = NB + 1: a one-column block
    check_ortho<AL, dv::tier::Cta<16>>(this->dev, make<T>(40, 16, 7));  // k = N: the Cta ceiling
}

// m = 40 and 70 (one and two partial 32-row tiles); k = 36 on Cta<40>: herk over 2 x 2 tiles and
// trsm over two diagonal blocks (32 + 4, the L3 update between them).
TYPED_TEST(DeviceApiUtil, OrthoCholeskyTiers) {
    using T = TypeParam;
    const auto a = make<T>(40, 12, 12);
    check_ortho<OrthoAlgorithm::Cholesky, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, a);
    check_ortho<OrthoAlgorithm::Chol2, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, a);
    check_ortho<OrthoAlgorithm::ShiftChol3, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, a);
    check_ortho<OrthoAlgorithm::Chol2, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, make<T>(70, 16, 13));  // k = N
    check_ortho<OrthoAlgorithm::Chol2, dv::tier::Cta<40>, L3Cfg<T>>(this->dev, make<T>(70, 36, 14));
}

// Columns c0 + dep r_j: one Gram-Schmidt pass loses orthogonality as eps * cond^2, and one
// CholeskyQR pass as eps * cond(A)^2 too; CGS2, block CGS2, Chol2 and ShiftChol3 keep eps.
template <class T> Host<T> near_dependent(int m, int k, std::uint64_t seed, double dep) {
    auto a = make<T>(m, k, seed);
    for (int b = 0; b < kBatch; ++b)
        for (int j = 1; j < a.n; ++j)
            for (int i = 0; i < a.m; ++i) a.at(b, i, j) = a.at(b, i, 0) + T(real_of<T>(dep)) * a.at(b, i, j);
    return a;
}
template <OrthoAlgorithm AL, class Tier, class Cfg = void, class T> void check_orthogonal(Dev& dev, const Host<T>& a) {
    SCOPED_TRACE(::testing::Message() << "algo " << int(AL));
    T* d = dev.up(a.h);
    const auto info = device_ortho<AL, Tier, Cfg>(dev, d, a, a.n);
    const auto got = dev.down(d, a.h.size());
    EXPECT_EQ(info[0], 0);
    EXPECT_EQ(bit_twins(got, std::size_t(a.stride)), 0);
    double orth = 0;
    for (int b = 0; b < kPeriod; ++b) orth = std::max(orth, orth_error(got, a, b));
    EXPECT_VERIFY(T, Check::orthogonality, a.m, orth);
}

TYPED_TEST(DeviceApiUtil, OrthoNearDependent) {
    using T = TypeParam;
    constexpr auto AL = OrthoAlgorithm::CGS2;
    const auto a = near_dependent<T>(48, 10, 8, 1e-2);
    check_orthogonal<AL, dv::tier::Blocked<32>, L3Cfg<T>>(this->dev, a);
    check_orthogonal<AL, dv::tier::Blocked<4>, L3Cfg<T>>(this->dev, a);
    check_orthogonal<OrthoAlgorithm::Chol2, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, a);
    check_orthogonal<OrthoAlgorithm::ShiftChol3, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, a);
    // cond ~ 1e5: one inter-block projection leaves eps * ||a|| / ||a_perp|| ~ 1e4 eps (block CGS2: eps).
    check_orthogonal<AL, dv::tier::Blocked<4>, L3Cfg<T>>(this->dev, near_dependent<T>(48, 10, 9, 1e-4));
}

TYPED_TEST(DeviceApiUtil, OrthoPastCeilingAndWorkspace) {
    using T = TypeParam;
    constexpr auto TR = Transpose::NoTrans;
    constexpr auto AL = OrthoAlgorithm::CGS2;
    const auto a = make<T>(40, 17, 9);  // k = N + 1 for tier::Cta<16>
    T* d = this->dev.up(a.h);
    const auto info = device_ortho<AL, dv::tier::Cta<16>>(this->dev, d, a, a.n);
    int not_refused = 0;
    for (int b = 0; b < kBatch; ++b) not_refused += info[b] != -2;
    EXPECT_EQ(not_refused, 0);
    const auto got = this->dev.down(d, a.h.size());
    EXPECT_EQ(std::memcmp(got.data(), a.h.data(), got.size() * sizeof(T)), 0) << "a refused call wrote A";
    // Each slot rounded up to the 16-byte alignment, in its space. The Level 3 stages of a
    // 32 x 32 x 8 tile: (TileM + 1) TileK + (TileK + 1) TileN = 264 + 288 values.
    const std::size_t stages = r16(552 * sizeof(T));
    const auto cta = dv::ortho_buffer_size<TR, AL, dv::tier::Cta<16>, T>(12);
    EXPECT_EQ(cta.local_bytes, r16(12 * sizeof(T)));
    EXPECT_EQ(cta.global_bytes, 0u);
    const auto blk = dv::ortho_buffer_size<TR, AL, dv::tier::Blocked<32>, T, L3Cfg<T>>(12);
    EXPECT_EQ(blk.local_bytes, stages);
    EXPECT_EQ(blk.global_bytes, r16(r16(12 * 32 * sizeof(T)) + 32 * sizeof(T)));  // H (k x NB), h (NB)
    const auto chol = dv::ortho_buffer_size<TR, OrthoAlgorithm::Chol2, dv::tier::Cta<16>, T, L3Cfg<T>>(12);
    EXPECT_EQ(chol.local_bytes, stages + dv::potrf_buffer_size<Uplo::Lower, dv::tier::Cta<16>, T>(12).local_bytes);
    EXPECT_EQ(chol.global_bytes, r16(12 * 12 * sizeof(T)));  // the Gram matrix
    const auto rz = dv::ritz_values_buffer_size<dv::tier::Blocked<3>, T, L3Cfg<T>>(45, 7);
    EXPECT_EQ(rz.local_bytes, stages);
    EXPECT_EQ(rz.global_bytes, r16(45 * 3 * sizeof(T)));  // A V(:, blk), n x NB
}

// A zero column j: the Gram matrix's leading minor j + 1 is singular, so every item returns
// info = j + 1 from the first pass with A untouched; k = N + 1 is refused (-2), A untouched.
TYPED_TEST(DeviceApiUtil, OrthoCholeskyInfoAndCeiling) {
    using T = TypeParam;
    auto a = make<T>(40, 12, 15);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < a.m; ++i) a.at(b, i, 5) = T(0);
    T* d = this->dev.up(a.h);
    const auto info = device_ortho<OrthoAlgorithm::Cholesky, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, d, a, a.n);
    int wrong = 0;
    for (int b = 0; b < kBatch; ++b) wrong += info[b] != 6;
    EXPECT_EQ(wrong, 0) << "info[0] = " << info[0];
    const auto got = this->dev.down(d, a.h.size());
    EXPECT_EQ(std::memcmp(got.data(), a.h.data(), got.size() * sizeof(T)), 0) << "a failed CholeskyQR pass wrote A";
    const auto past = make<T>(40, 17, 16);
    T* p = this->dev.up(past.h);
    const auto refused = device_ortho<OrthoAlgorithm::Chol2, dv::tier::Cta<16>, L3Cfg<T>>(this->dev, p, past, past.n);
    int not_refused = 0;
    for (int b = 0; b < kBatch; ++b) not_refused += refused[b] != -2;
    EXPECT_EQ(not_refused, 0);
    const auto kept = this->dev.down(p, past.h.size());
    EXPECT_EQ(std::memcmp(kept.data(), past.h.data(), kept.size() * sizeof(T)), 0) << "a refused call wrote A";
}

}  // namespace
