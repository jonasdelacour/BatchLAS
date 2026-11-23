// SPDX-License-Identifier: MIT
// device_lapack_launch-run_tests: launch<R, Tier, T>::run and the routine_call of every LAPACK
// routine tag (lapack_run.hh).
// Guards (agent-guide section 8): every routine_call is instantiated at compile time against its
// routine's operands (a binding that names the wrong function, drops a mode or puts the tier in
// the wrong place does not compile); at run time L::run is bit-identical to the direct call for
// potrf<Upper> on the packed Reg<16> tier (odd batch, dead tail slot, ld = n + 3), Cta<64> and
// Blocked<32>, real and complex, and for getrs<Trans> on getrf's factors. Upper and Trans are the
// non-default modes, so a binding that ignored its tag's mode would differ from the direct call.
// Red set (each break made once, 2026-11-23, and this suite failed): routine_call<potrf<UL>>
// forwarding Uplo::Lower; routine_call<getrs<TR>> forwarding Transpose::NoTrans.
#include <tests/device_core_common.hh>

#include <batchlas/device/lapack.hh>

#include <algorithm>
#include <complex>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

using namespace device_core_test;

namespace {

namespace dv = batchlas::device;
namespace rt = batchlas::device::routine;
using dv::tier::Blocked;
using dv::tier::Cta;
using dv::tier::Reg;

using F = float;
using M = decltype(dv::mat(static_cast<F*>(nullptr), 1, 1, 1));
using V = decltype(dv::vec(static_cast<F*>(nullptr), 1));
using P = decltype(dv::vec(static_cast<int*>(nullptr), 1));
using W = const dv::Workspace&;
template <class X> using C = const X&;
template <class Cs> using L3 = dv::l3::cfg::RegTiled<Cs, 32, 32, 8, 4, 2>;

template <class R, class Tier, class... A>
constexpr bool runs = std::is_same_v<
    decltype(dv::launch<R, Tier, F>::run(std::declval<C<typename dv::launch<R, Tier, F>::executor>>(), std::declval<A>()...)), int>;
template <class List> struct front;
template <class H, class... Rest> struct front<dv::tier::list<H, Rest...>> {
    using type = H;
};
// The routine's first listed tier: every listed tier is valid at compile time, whatever n admits.
template <class R, class... A> constexpr bool runs_any = runs<R, typename front<dv::tiers_t<R, F>>::type, A...>;

static_assert(runs_any<rt::bdsdc, C<V>, C<V>, C<V>, C<M>, C<M>, W>);
static_assert(runs_any<rt::bdsqr, C<V>, C<V>, C<V>, C<M>, C<M>, W>);
static_assert(runs_any<rt::gebrd, C<sycl::nd_item<1>>, C<M>, C<V>, C<V>, C<V>, C<V>, W>);  // Cta: the nd_item too
static_assert(runs_any<rt::geqrf, C<M>, C<V>, W>);
static_assert(runs_any<rt::gesv, C<M>, C<P>, C<M>, W>);
static_assert(runs_any<rt::getrf, C<M>, C<P>, W>);
static_assert(runs_any<rt::getri, C<M>, C<P>, C<M>, W>);
static_assert(runs_any<rt::inv, C<M>, C<M>, W>);
static_assert(runs_any<rt::orgqr, C<M>, C<V>, W>);
static_assert(runs_any<rt::stebz, C<V>, C<V>, C<V>, int&, W, C<dv::StebzOptions<F>>>);
static_assert(runs_any<rt::stein, C<V>, C<V>, C<V>, int, C<M>, W, C<dv::SteinOptions<F>>>);
static_assert(runs_any<rt::potrf<Uplo::Upper>, C<M>, W>);
static_assert(runs_any<rt::potrs<Uplo::Upper>, C<M>, C<M>, W>);
static_assert(runs_any<rt::posv<Uplo::Upper>, C<M>, C<M>, W>);
static_assert(runs_any<rt::potri<Uplo::Upper>, C<M>, W>);
static_assert(runs_any<rt::lauum<Uplo::Upper>, C<M>, W>);
static_assert(runs_any<rt::trtri<Uplo::Upper, Diag::Unit>, C<M>, W>);
static_assert(runs_any<rt::sytrf<Uplo::Upper>, C<M>, C<P>, W>);
static_assert(runs_any<rt::sytrs<Uplo::Upper>, C<M>, C<P>, C<M>, W>);
static_assert(runs_any<rt::sysv<Uplo::Upper>, C<M>, C<P>, C<M>, W>);
static_assert(runs_any<rt::sytrd<Uplo::Upper>, C<M>, C<V>, C<V>, C<V>, W>);
static_assert(runs_any<rt::sy2sb<Uplo::Lower, 32>, C<M>, C<M>, C<V>, int, W>);
static_assert(runs_any<rt::sb2st<Uplo::Lower, 32>, C<M>, C<V>, C<V>, C<V>, int, W>);
static_assert(runs_any<rt::getrs<Transpose::Trans>, C<M>, C<P>, C<M>, W>);
static_assert(runs_any<rt::ormqr<Side::Right, Transpose::Trans>, C<M>, C<V>, C<M>, W>);
static_assert(runs_any<rt::ormbr<dv::Vect::P, Side::Right, Transpose::Trans>, C<M>, C<V>, C<M>, W>);
static_assert(runs_any<rt::sygst<dv::Itype::AB, Uplo::Upper>, C<M>, C<M>, W>);
static_assert(runs_any<rt::syev<JobType::EigenVectors, Uplo::Upper>, C<M>, C<V>, W, C<dv::SyevOptions<F>>>);
static_assert(runs_any<rt::stedc<JobType::EigenVectors>, C<V>, C<V>, C<V>, C<M>, W, C<dv::StedcOptions<F>>>);
static_assert(runs_any<rt::steqr<JobType::EigenVectors>, C<V>, C<V>, C<M>, W>);
static_assert(runs_any<rt::gesvd<SvdVectors::All, SvdVectors::None>, C<M>, C<V>, C<M>, C<M>, W>);
static_assert(runs<rt::ortho<Transpose::NoTrans, L3<F>, OrthoAlgorithm::Chol2>, Cta<16>, C<M>, W>);
static_assert(runs<rt::ritz_values<L3<F>>, Blocked<3>, C<M>, C<M>, C<V>, W>);

// SPD matrices A = n I + a symmetric Hilbert-like perturbation, column-major at ld; the entries
// outside A (ld padding) are a poison value both runs must leave alone.
template <class T> std::vector<T> spd(int n, int ld, int batch) {
    std::vector<T> a(std::size_t(ld) * n * batch, T(-99));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const float re = (i == j ? float(n) : 0.f) + 1.f / float(1 + i + j + b % 5);
                const float im = i == j ? 0.f : (i < j ? 0.25f : -0.25f) / float(1 + i + j);
                if constexpr (std::is_same_v<T, float>) a[(std::size_t(b) * n + j) * ld + i] = T(re);
                else a[(std::size_t(b) * n + j) * ld + i] = T(re, im);
            }
    return a;
}

template <bool ViaRun, class Tier, class T> std::vector<T> potrf_upper(Dev& d, const std::vector<T>& host, int n, int ld, int batch) {
    using L = dv::launch<rt::potrf<Uplo::Upper>, Tier, T>;
    T* a = d.alloc<T>(host.size(), T(0));
    d.q().copy(host.data(), a, host.size()).wait();
    const auto sz = L::buffer_size(n);
    std::byte* gws = sz.global_bytes ? d.alloc<std::byte>(sz.global_bytes * batch, std::byte{0}) : nullptr;
    int* info = d.alloc<int>(batch, -1);
    const long stride = long(ld) * n;
    d.q().submit([&](sycl::handler& h) {
         auto slm = dv::local_workspace<L>(h, sz);
         h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
             const auto ex = L::exec(it);
             const auto s = L::slot(it, batch);
             const auto A = dv::mat(a + (s.live ? s.b : 0) * stride, n, n, ld);
             const auto ws = L::workspace(it, slm, gws, sz);
             int r;
             if constexpr (ViaRun) r = L::run(ex, A, ws, s.live);
             else r = dv::potrf<Uplo::Upper, typename L::tier_type>(ex, A, ws, s.live);
             if (s.live && ex.leader()) info[s.b] = r;
         });
     }).wait_and_throw();
    for (int v : d.down(info, batch)) EXPECT_EQ(v, 0);
    return d.down(a, host.size());
}

template <class T> bool same_bits(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

template <class T> class LaunchRun : public ::testing::Test {};
using Scalars = ::testing::Types<float, std::complex<float>>;
TYPED_TEST_SUITE(LaunchRun, Scalars);

// potrf<Upper> through L::run against the direct call, on each kind of tier. Reg<16> packs two
// matrices per work-group, so batch 1025 leaves one dead slot.
TYPED_TEST(LaunchRun, PotrfUpperMatchesTheDirectCallBitForBit) {
    using T = TypeParam;
    Dev d;
    if (!d.ctx) GTEST_SKIP() << "no SYCL device";
    {
        const int n = 13, ld = n + 3, batch = 1025;
        const auto host = spd<T>(n, ld, batch);
        const auto got = potrf_upper<true, Reg<16>>(d, host, n, ld, batch);
        EXPECT_FALSE(same_bits(got, host)) << "Reg<16>: nothing was factored";
        EXPECT_TRUE(same_bits(got, potrf_upper<false, Reg<16>>(d, host, n, ld, batch))) << "Reg<16>";
    }
    {
        const int n = 45, ld = n + 3, batch = 129;
        const auto host = spd<T>(n, ld, batch);
        EXPECT_TRUE(same_bits(potrf_upper<true, Cta<64>>(d, host, n, ld, batch), potrf_upper<false, Cta<64>>(d, host, n, ld, batch)))
            << "Cta<64>";
    }
    {
        const int n = 70, ld = n + 3, batch = 128;
        const auto host = spd<T>(n, ld, batch);
        EXPECT_TRUE(same_bits(potrf_upper<true, Blocked<32>>(d, host, n, ld, batch), potrf_upper<false, Blocked<32>>(d, host, n, ld, batch)))
            << "Blocked<32>";
    }
}

// getrf then getrs<Trans> (A^T X = B) on one launch, the solve through L::run or the direct call.
template <bool ViaRun, class T> std::vector<T> lu_solve_trans(Dev& d, const std::vector<T>& ha, const std::vector<T>& hb, int n, int nrhs, int ld, int batch) {
    using Tier = Blocked<32>;
    using L = dv::launch<rt::getrs<Transpose::Trans>, Tier, T>;
    T* a = d.alloc<T>(ha.size(), T(0));
    T* b = d.alloc<T>(hb.size(), T(0));
    d.q().copy(ha.data(), a, ha.size()).wait();
    d.q().copy(hb.data(), b, hb.size()).wait();
    int* piv = d.alloc<int>(std::size_t(n) * batch, 0);
    const auto sz = L::buffer_size(n, nrhs);
    const auto fz = dv::buffer_size<rt::getrf, Tier, T>(n);
    const dv::WorkspaceSize big{std::max(sz.local_bytes, fz.local_bytes), std::max(sz.global_bytes, fz.global_bytes), sz.align};
    std::byte* gws = big.global_bytes ? d.alloc<std::byte>(big.global_bytes * batch, std::byte{0}) : nullptr;
    const long sa = long(ld) * n, sb = long(ld) * nrhs;
    d.q().submit([&](sycl::handler& h) {
         auto slm = dv::local_workspace<L>(h, big);
         h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
             const auto ex = L::exec(it);
             const auto s = L::slot(it, batch);
             const auto A = dv::mat(a + s.b * sa, n, n, ld);
             const auto B = dv::mat(b + s.b * sb, n, nrhs, ld);
             const auto Pv = dv::vec(piv + s.b * n, n);
             const auto ws = L::workspace(it, slm, gws, big);
             dv::getrf<Tier>(ex, A, Pv, ws, s.live);
             if constexpr (ViaRun) L::run(ex, A, Pv, B, ws, s.live);
             else dv::getrs<Transpose::Trans, Tier>(ex, A, Pv, B, ws, s.live);
         });
     }).wait_and_throw();
    return d.down(b, hb.size());
}

TYPED_TEST(LaunchRun, GetrsTransMatchesTheDirectCallBitForBit) {
    using T = TypeParam;
    Dev d;
    if (!d.ctx) GTEST_SKIP() << "no SYCL device";
    const int n = 37, nrhs = 3, ld = n + 3, batch = 128;
    auto ha = spd<T>(n, ld, batch);
    for (int b = 0; b < batch; ++b) ha[(std::size_t(b) * n + n - 1) * ld] += T(3);  // A != A^T
    std::vector<T> hb(std::size_t(ld) * nrhs * batch, T(-99));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) hb[(std::size_t(b) * nrhs + j) * ld + i] = T(float(1 + (i * 7 + j * 3 + b) % 11));
    const auto via = lu_solve_trans<true>(d, ha, hb, n, nrhs, ld, batch);
    EXPECT_FALSE(same_bits(via, hb)) << "nothing was solved";
    EXPECT_TRUE(same_bits(via, lu_solve_trans<false>(d, ha, hb, n, nrhs, ld, batch)));
}

}  // namespace
