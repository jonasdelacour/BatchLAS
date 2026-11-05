// SPDX-License-Identifier: MIT
// device_api_qr_tests: the public in-kernel QR ops (include/batchlas/device/lapack/{geqrf,ormqr,
// orgqr}.hh, design C.2-C.10) called from user-style kernels.
//   geqrf<Reg<N>>, geqrf<Cta<N>> and ormqr<SD, TR, Cta<P>> are the bodies of the host tiny / cta /
//   ormqx_cta launchers: the device call must be BIT-IDENTICAL to the host route (memcmp of the
//   whole buffer, pads included, and tau), the host route reached by its direct entry point
//   (geqrf_tiny_dispatch, geqrf_cta_dispatch, ormqx_cta), never through Auto.
//   orgqr<Cta<P>> has no host route of its own: it is checked against the host orgqr with a
//   tolerance and for orthogonality.
// Guards (agent-guide section 8): batch 1027 (a dead tail partition in every packed kernel), ld =
// rows + 3 and stride = ld * cols + 5, pads poisoned and checked unchanged, complex data with
// nonzero imaginary parts everywhere, n at every Reg tier's ceiling and just above the previous
// one, the Cta tier at both executor scopes (sub-group packed, work-group), a one-ulp input change
// in one matrix that must change that matrix and no other (the comparator is not vacuous, and no
// matrix leaks into its neighbour through local memory), with_tier at every tier boundary with a
// 32 KB local-memory limit straddled, and buffer_size equal to the bytes the op carves.
// Red set: not yet recorded (written without a build); the first GPU run records it.
#include <tests/device_core_common.hh>

#include <batchlas/device/lapack/geqrf.hh>
#include <batchlas/device/lapack/orgqr.hh>
#include <batchlas/device/lapack/ormqr.hh>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/ops/geqrf/geqrf_native.hh>

#include <cmath>
#include <string>

using namespace device_core_test;

namespace {

template <class T> using DevT = std::conditional_t<is_cplx<T>, dv::Cx<real_of<T>>, T>;
template <class T> DevT<T>* dptr(T* p) { return reinterpret_cast<DevT<T>*>(p); }
template <class T> T poison() {
    if constexpr (is_cplx<T>) return T(real_of<T>(-7.75e3), real_of<T>(3.5e3));
    else return T(-7.75e3);
}

// A batch of m x n matrices at ld, stride with every element outside the matrices poisoned.
template <class T> struct Batch {
    int m, n, ld, stride, batch;
    std::vector<T> h;
    Batch(int m_, int n_, int batch_, std::uint64_t seed) : m(m_), n(n_), ld(m_ + 3), stride((m_ + 3) * n_ + 5), batch(batch_) {
        h.assign(static_cast<std::size_t>(stride) * batch, poison<T>());
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) h[idx(b, i, j)] = val<T>(seed * 7919u + idx(b, i, j));
    }
    std::size_t idx(int b, int i, int j) const { return static_cast<std::size_t>(b) * stride + static_cast<std::size_t>(j) * ld + i; }
    bool in_matrix(std::size_t o) const {
        const std::size_t r = o % static_cast<std::size_t>(stride);
        return r < static_cast<std::size_t>(ld) * n && static_cast<int>(r % ld) < m;
    }
};

// First offset where a and b differ in bits, or -1.
template <class T> long first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    for (std::size_t o = 0; o < a.size(); ++o)
        if (!same_bits(a[o], b[o])) return static_cast<long>(o);
    return -1;
}

inline bool host_gpu(Dev& d) { return d.ctx->device().type == DeviceType::GPU; }

// ---- user-style kernels ---------------------------------------------------------------------
// geqrf<Reg<N>>: 64 / N matrices per 64-item work-group, one per N-lane partition; a dead
// partition (the batch tail) still calls the op with live = false.
template <class T, int N> void geqrf_reg(Dev& d, T* a, T* tau, int n, int ld, int stride, int batch, int* info) {
    using D = DevT<T>;
    using Tier = dv::tier::Reg<N>;
    constexpr std::size_t LB = dv::geqrf_buffer_size<Tier, D>(N).local_bytes;
    constexpr int WG = 64, M = WG / N;
    D* const ap = dptr(a);
    D* const tp = dptr(tau);
    const int groups = (batch + M - 1) / M;
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<std::uint64_t, 1> lm(sycl::range<1>(M * LB / 8 + 1), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const auto sg = it.get_sub_group();
             const dv::SgpPartition<N> ex{sg};
             const int slot = static_cast<int>(sg.get_group_linear_id()) * (32 / N) + static_cast<int>(ex.part.get_group_linear_id());
             const int b = static_cast<int>(it.get_group_linear_id()) * M + slot;
             const bool live = b < batch;
             const int bb = live ? b : 0;
             std::byte* const base = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(lm)) + std::size_t(slot) * LB;
             const dv::Workspace ws{
                 sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base)};
             const int r = dv::geqrf<Tier>(ex, dv::mat_wide(ap + std::ptrdiff_t(bb) * stride, n, n, ld),
                                          dv::vec(tp + std::ptrdiff_t(bb) * n, n), ws, live);
             if (live && ex.leader()) info[b] = r;
         });
     }).wait_and_throw();
}

// geqrf<Cta<64>> on RtGroup<S>: Sub = one matrix per sub-group (two per work-group), Group =
// one matrix per work-group of wg items.
template <class T, dv::Scope S> void geqrf_cta(Dev& d, T* a, T* tau, int m, int n, int ld, int stride, int batch, int wg, int* info) {
    using D = DevT<T>;
    using Tier = dv::tier::Cta<64>;
    const std::size_t lb = dv::geqrf_buffer_size<Tier, D>(m, n).local_bytes;
    const int per = S == dv::Scope::Sub ? wg / 32 : 1;
    const int k = std::min(m, n);
    D* const ap = dptr(a);
    D* const tp = dptr(tau);
    const int groups = (batch + per - 1) / per;
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<std::uint64_t, 1> lm(sycl::range<1>(per * lb / 8 + 1), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, wg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const dv::RtGroup<S> ex{it};
             const int slot = S == dv::Scope::Sub ? static_cast<int>(it.get_sub_group().get_group_linear_id()) : 0;
             const int b = static_cast<int>(it.get_group_linear_id()) * per + slot;
             if (b >= batch) return;  // sub-group-uniform (Sub) or work-group-uniform (Group)
             std::byte* const base = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(lm)) + std::size_t(slot) * lb;
             const dv::Workspace ws{
                 sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base)};
             const int r = dv::geqrf<Tier>(ex, dv::mat_wide(ap + std::ptrdiff_t(b) * stride, m, n, ld),
                                          dv::vec(tp + std::ptrdiff_t(b) * k, k), ws);
             if (ex.leader()) info[b] = r;
         });
     }).wait_and_throw();
}

// ormqr<SD, TR, Cta<P>> (host scalar T, arith::Native: the host route's arithmetic) and
// orgqr<Cta<P>> (device scalar, arith::Explicit), 64 / P matrices per 64-item work-group.
template <class T, Side SD, Transpose TR, int P>
void ormqr_cta(Dev& d, T* a, T* c, T* tau, int n, int k, int ld, int stride, int batch, int* info) {
    using Tier = dv::tier::Cta<P>;
    constexpr std::size_t LB = dv::ormqr_buffer_size<SD, Tier, T>(P).local_bytes;
    constexpr int WG = 64, M = WG / P;
    const int groups = (batch + M - 1) / M;
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<std::uint64_t, 1> lm(sycl::range<1>(M * LB / 8 + 1), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const auto sg = it.get_sub_group();
             const dv::SgpPartition<P> ex{sg};
             const int slot = static_cast<int>(sg.get_group_linear_id()) * (32 / P) + static_cast<int>(ex.part.get_group_linear_id());
             const int b = static_cast<int>(it.get_group_linear_id()) * M + slot;
             const bool live = b < batch;
             const int bb = live ? b : 0;
             std::byte* const base = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(lm)) + std::size_t(slot) * LB;
             const dv::Workspace ws{
                 sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base)};
             const int r = dv::ormqr<SD, TR, Tier>(ex, dv::mat_wide(a + std::ptrdiff_t(bb) * stride, n, n, ld),
                                                   dv::mat_wide(c + std::ptrdiff_t(bb) * stride, n, n, ld),
                                                   dv::vec(tau + std::ptrdiff_t(bb) * k, k), ws, k, live);
             if (live && ex.leader()) info[b] = r;
         });
     }).wait_and_throw();
}

template <class T, int P> void orgqr_cta(Dev& d, T* a, T* tau, int m, int n, int ld, int stride, int batch, int* info) {
    using D = DevT<T>;
    using Tier = dv::tier::Cta<P>;
    constexpr std::size_t LB = dv::orgqr_buffer_size<Tier, D>(P).local_bytes;
    constexpr int WG = 64, M = WG / P;
    D* const ap = dptr(a);
    D* const tp = dptr(tau);
    const int groups = (batch + M - 1) / M;
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<std::uint64_t, 1> lm(sycl::range<1>(M * LB / 8 + 1), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const auto sg = it.get_sub_group();
             const dv::SgpPartition<P> ex{sg};
             const int slot = static_cast<int>(sg.get_group_linear_id()) * (32 / P) + static_cast<int>(ex.part.get_group_linear_id());
             const int b = static_cast<int>(it.get_group_linear_id()) * M + slot;
             const bool live = b < batch;
             const int bb = live ? b : 0;
             std::byte* const base = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(lm)) + std::size_t(slot) * LB;
             const dv::Workspace ws{
                 sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(base)};
             const int r = dv::orgqr<Tier>(ex, dv::mat_wide(ap + std::ptrdiff_t(bb) * stride, m, n, ld),
                                           dv::vec(tp + std::ptrdiff_t(bb) * n, n), ws, live);
             if (live && ex.leader()) info[b] = r;
         });
     }).wait_and_throw();
}

template <class T> class DeviceApiQr : public DeviceCoreTest<T> {
  protected:
    void SetUp() override {
        DeviceCoreTest<T>::SetUp();
        if (this->IsSkipped()) return;
        if (!host_gpu(this->dev)) GTEST_SKIP() << "the host routes compared against run on GPU queues only";
    }

  public:
    Queue& ctx() { return *this->dev.ctx; }

    // Runs the device op and the host route on two copies of p; returns {device, host} buffers
    // and taus. host_run returns false when the host route declines the shape (skip).
    template <class DevRun, class HostRun>
    bool both(const Batch<T>& p, int k, DevRun dev_run, HostRun host_run, std::vector<T> out[2], std::vector<T> tau[2]) {
        const std::size_t ntau = static_cast<std::size_t>(k) * p.batch;
        T* ad = this->dev.up(p.h);
        T* ah = this->dev.up(p.h);
        T* td = this->dev.alloc(ntau, poison<T>());
        T* th = this->dev.alloc(ntau, poison<T>());
        int* info = this->dev.alloc(p.batch, -99);
        try {
            host_run(ah, th);
            ctx().wait();
        } catch (const batchlas::invalid_argument&) {
            return false;
        } catch (const batchlas::unsupported&) {
            return false;
        }
        dev_run(ad, td, info);
        out[0] = this->dev.down(ad, p.h.size());
        out[1] = this->dev.down(ah, p.h.size());
        tau[0] = this->dev.down(td, ntau);
        tau[1] = this->dev.down(th, ntau);
        const auto inf = this->dev.down(info, p.batch);
        for (int b = 0; b < p.batch; ++b) EXPECT_EQ(inf[b], 0) << "info of matrix " << b;
        return true;
    }

    void expect_identical(const Batch<T>& p, const std::vector<T> out[2], const std::vector<T> tau[2], const std::string& what) {
        const long o = first_diff(out[0], out[1]);
        EXPECT_EQ(o, -1) << what << ": device op and host route differ at offset " << o << " (matrix " << (o / p.stride) << ")";
        const long t = first_diff(tau[0], tau[1]);
        EXPECT_EQ(t, -1) << what << ": tau differs at " << t;
        for (std::size_t q = 0; q < out[0].size(); ++q)
            if (!p.in_matrix(q)) ASSERT_TRUE(same_bits(out[0][q], poison<T>())) << what << ": pad written at offset " << q;
    }
};

TYPED_TEST_SUITE(DeviceApiQr, AllTypes);

template <class T, int N> void reg_case(DeviceApiQr<T>& t, Dev& d, int n) {
    Batch<T> p(n, n, 1027, 11u + unsigned(n));
    std::vector<T> out[2], tau[2];
    const bool ran = t.both(
        p, n, [&](T* a, T* tau_, int* info) { geqrf_reg<T, N>(d, a, tau_, n, p.ld, p.stride, p.batch, info); },
        [&](T* a, T* tau_) {
            (void)sycl_geqrf::geqrf_tiny_dispatch<T>(*d.ctx, MatrixView<T, MatrixFormat::Dense>(a, n, n, p.ld, p.stride, p.batch),
                                                     Span<T>(tau_, std::size_t(n) * p.batch), Span<std::byte>());
        },
        out, tau);
    if (!ran) return;  // above this type's tiny ceiling on this device
    t.expect_identical(p, out, tau, "geqrf Reg<" + std::to_string(N) + "> n=" + std::to_string(n));
}

// Reg<N> is the tiny launcher's body: bit-identical at every bucket's ceiling and just above the
// previous one (n = 1 is the degenerate reflector).
TYPED_TEST(DeviceApiQr, GeqrfRegIsTheTinyRouteBitForBit) {
    using T = TypeParam;
    for (int n : {1, 5, 8}) reg_case<T, 8>(*this, this->dev, n);
    for (int n : {9, 16}) reg_case<T, 16>(*this, this->dev, n);
    if constexpr (!(is_cplx<T> && std::is_same_v<real_of<T>, double>))
        for (int n : {17, 29, 32}) reg_case<T, 32>(*this, this->dev, n);
}

template <class T, dv::Scope S> void cta_case(DeviceApiQr<T>& t, Dev& d, int m, int n, int wg) {
    Batch<T> p(m, n, 1024, 23u + unsigned(m * 64 + n));
    const int k = std::min(m, n);
    std::vector<T> out[2], tau[2];
    const bool ran = t.both(
        p, k, [&](T* a, T* tau_, int* info) { geqrf_cta<T, S>(d, a, tau_, m, n, p.ld, p.stride, p.batch, wg, info); },
        [&](T* a, T* tau_) {
            (void)sycl_geqrf::geqrf_cta_dispatch<T>(*d.ctx, MatrixView<T, MatrixFormat::Dense>(a, m, n, p.ld, p.stride, p.batch),
                                                    Span<T>(tau_, std::size_t(k) * p.batch), Span<std::byte>());
        },
        out, tau);
    if (!ran) return;  // the panel does not fit this device's CTA budget
    t.expect_identical(p, out, tau, "geqrf Cta m=" + std::to_string(m) + " n=" + std::to_string(n));
}

// The host packs an m, n <= 32 panel one per sub-group only when two fit the occupancy slice of
// local memory (geqrf_cta.cc's geqrf_leaf_launch); on a 32 KiB device (Metal) a complex<float>
// 27 x 20 or 12 x 31 panel runs on a work-group instead, so the comparison takes that scope.
template <class T> void packed_cta_case(DeviceApiQr<T>& t, Dev& d, int m, int n) {
    namespace rc = ::batchlas::resident;
    const auto dev = d.ctx->device();
    const std::size_t slice = rc::occupancy_budget(rc::device_slm_budget(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE)));
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    if (rc::pack_matrices_per_wg(std::size_t(m) * std::size_t(n) * sizeof(T), 32, slice, max_wg) > 1)
        cta_case<T, dv::Scope::Sub>(t, d, m, n, 64);
    else
        cta_case<T, dv::Scope::Group>(t, d, m, n, dv::geqrf_detail::cta_wg(n));
}

// Cta<N> is the resident leaf's body: the host packs m, n <= 32 panels one per sub-group (when
// two fit) and runs larger ones on a work-group of 32 * min(8, pow2 >= n) items; the device call
// at the same scope and width is bit-identical (tall, wide, and the n = 33 step out of the band).
TYPED_TEST(DeviceApiQr, GeqrfCtaIsTheCtaRouteBitForBit) {
    using T = TypeParam;
    packed_cta_case<T>(*this, this->dev, 27, 20);
    packed_cta_case<T>(*this, this->dev, 12, 31);
    cta_case<T, dv::Scope::Group>(*this, this->dev, 37, 34, dv::geqrf_detail::cta_wg(34));
    cta_case<T, dv::Scope::Group>(*this, this->dev, 30, 33, dv::geqrf_detail::cta_wg(33));
}

// One ulp in one input element of matrix 517 changes matrix 517 and nothing else: the bitwise
// comparator above can fail, and no matrix reaches its neighbour's local slot.
TYPED_TEST(DeviceApiQr, OneUlpInOneMatrixChangesOnlyThatMatrix) {
    using T = TypeParam;
    constexpr int n = 13;
    Batch<T> p(n, n, 1027, 77u);
    Batch<T> q = p;
    T& e = q.h[q.idx(517, 6, 4)];
    if constexpr (is_cplx<T>) e = T(std::nextafter(e.real(), real_of<T>(2)), e.imag());
    else e = std::nextafter(e, T(2));
    T* a0 = this->dev.up(p.h);
    T* a1 = this->dev.up(q.h);
    T* t0 = this->dev.alloc(std::size_t(n) * p.batch, poison<T>());
    T* t1 = this->dev.alloc(std::size_t(n) * p.batch, poison<T>());
    int* info = this->dev.alloc(p.batch, -99);
    geqrf_reg<T, 16>(this->dev, a0, t0, n, p.ld, p.stride, p.batch, info);
    geqrf_reg<T, 16>(this->dev, a1, t1, n, p.ld, p.stride, p.batch, info);
    const auto r0 = this->dev.down(a0, p.h.size());
    const auto r1 = this->dev.down(a1, p.h.size());
    int changed = 0;
    for (int b = 0; b < p.batch; ++b) {
        bool diff = false;
        for (int o = 0; o < p.stride; ++o) diff = diff || !same_bits(r0[std::size_t(b) * p.stride + o], r1[std::size_t(b) * p.stride + o]);
        if (diff) {
            ++changed;
            EXPECT_EQ(b, 517) << "matrix " << b << " changed with an input of matrix 517";
        }
    }
    EXPECT_EQ(changed, 1) << "the one-ulp change was not seen (a vacuous comparator) or leaked";
}

template <class T, Side SD, Transpose TR, int P> void ormqr_case(DeviceApiQr<T>& t, Dev& d, int n, int k) {
    Batch<T> p(n, n, 1027, 31u + unsigned(n));
    Batch<T> cm(n, n, 1027, 41u + unsigned(n));
    std::vector<T> out[2], tau[2];
    const bool ran = t.both(
        p, k,
        [&](T* a, T* tau_, int* info) {
            T* c = d.up(cm.h);
            d.q().memcpy(tau_, d.up(vals<T>(std::size_t(k) * p.batch, 5u)), sizeof(T) * k * p.batch).wait();
            ormqr_cta<T, SD, TR, P>(d, a, c, tau_, n, k, p.ld, p.stride, p.batch, info);
            d.q().memcpy(a, c, sizeof(T) * cm.h.size()).wait();  // compare C through the A slot
        },
        [&](T* a, T* tau_) {
            T* c = d.up(cm.h);
            d.q().memcpy(tau_, d.up(vals<T>(std::size_t(k) * p.batch, 5u)), sizeof(T) * k * p.batch).wait();
            (void)batchlas::ormqx_cta(*d.ctx, MatrixView<T, MatrixFormat::Dense>(a, n, n, p.ld, p.stride, p.batch),
                            VectorView<T>(tau_, k, p.batch, 1, k), MatrixView<T, MatrixFormat::Dense>(c, n, n, p.ld, p.stride, p.batch),
                            Uplo::Upper, SD, TR, k, Span<std::byte>(), std::size_t(1));
            d.ctx->wait();
            d.q().memcpy(a, c, sizeof(T) * cm.h.size()).wait();
        },
        out, tau);
    if (!ran) return;
    t.expect_identical(cm, out, tau, std::string("ormqr ") + (SD == Side::Left ? "L" : "R") + " n=" + std::to_string(n));
}

// ormqr<SD, TR, Cta<P>> is ormqx_cta's body (QR order): bit-identical for both sides, both
// transposes, k < n, at P's ceiling and below it.
TYPED_TEST(DeviceApiQr, OrmqrCtaIsTheOrmqxCtaRouteBitForBit) {
    using T = TypeParam;
    constexpr Transpose H = is_cplx<T> ? Transpose::ConjTrans : Transpose::Trans;
    ormqr_case<T, Side::Left, Transpose::NoTrans, 16>(*this, this->dev, 13, 11);
    ormqr_case<T, Side::Left, H, 16>(*this, this->dev, 16, 16);
    ormqr_case<T, Side::Right, Transpose::NoTrans, 32>(*this, this->dev, 29, 23);
    ormqr_case<T, Side::Right, H, 8>(*this, this->dev, 8, 5);
}

template <class T, int P> void orgqr_case(DeviceApiQr<T>& t, Dev& d, int m, int n) {
    Batch<T> p(m, n, 1027, 53u + unsigned(m * 64 + n));
    T* a = d.up(p.h);
    T* tau = d.alloc(std::size_t(n) * p.batch, poison<T>());
    const MatrixView<T, MatrixFormat::Dense> A(a, m, n, p.ld, p.stride, p.batch);
    {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, batchlas::geqrf_buffer_size(*d.ctx, A, Span<T>(tau, std::size_t(n) * p.batch))));
        (void)batchlas::geqrf(*d.ctx, A, Span<T>(tau, std::size_t(n) * p.batch), ws.to_span());
        d.ctx->wait();
    }
    const auto factored = d.down(a, p.h.size());
    T* ah = d.up(factored);
    int* info = d.alloc(p.batch, -99);
    orgqr_cta<T, P>(d, a, tau, m, n, p.ld, p.stride, p.batch, info);
    const MatrixView<T, MatrixFormat::Dense> Ah(ah, m, n, p.ld, p.stride, p.batch);
    UnifiedVector<std::byte> ws(std::max<std::size_t>(1, batchlas::orgqr_buffer_size(*d.ctx, Ah, Span<T>(tau, std::size_t(n) * p.batch))));
    (void)batchlas::orgqr(*d.ctx, Ah, Span<T>(tau, std::size_t(n) * p.batch), ws.to_span());
    d.ctx->wait();
    const auto qd = d.down(a, p.h.size());
    const auto qh = d.down(ah, p.h.size());
    const auto inf = d.down(info, p.batch);
    const double eps = static_cast<double>(std::numeric_limits<real_of<T>>::epsilon());
    double worst = 0, worst_orth = 0;
    for (int b = 0; b < p.batch; ++b) {
        EXPECT_EQ(inf[b], 0) << b;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) worst = std::max(worst, static_cast<double>(std::abs(qd[p.idx(b, i, j)] - qh[p.idx(b, i, j)])));
        if (b % 97 != 0) continue;  // Q^H Q - I in long double on a sample
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < n; ++l) {
                LD<T> s = 0;
                for (int i = 0; i < m; ++i) {
                    const LD<T> x = wide(qd[p.idx(b, i, j)]);
                    const LD<T> y = wide(qd[p.idx(b, i, l)]);
                    if constexpr (is_cplx<T>) s += std::conj(x) * y;
                    else s += x * y;
                }
                worst_orth = std::max(worst_orth, static_cast<double>(std::abs(s - LD<T>(j == l ? 1 : 0))));
            }
    }
    EXPECT_LE(worst, 16.0 * m * eps) << "orgqr Cta<" << P << "> m=" << m << " n=" << n << " vs the host orgqr";
    EXPECT_LE(worst_orth, 16.0 * m * eps) << "orgqr Cta<" << P << "> m=" << m << " n=" << n << ": Q^H Q - I";
    for (std::size_t q = 0; q < qd.size(); ++q)
        if (!p.in_matrix(q)) ASSERT_TRUE(same_bits(qd[q], poison<T>())) << "pad written at offset " << q;
}

// orgqr<Cta<P>> (xORG2R in registers) forms the host orgqr's Q within 16 m eps, square and tall,
// at P's ceiling (m = P) and below it.
TYPED_TEST(DeviceApiQr, OrgqrCtaMatchesTheHostOrgqr) {
    using T = TypeParam;
    orgqr_case<T, 16>(*this, this->dev, 16, 16);
    orgqr_case<T, 16>(*this, this->dev, 13, 9);
    orgqr_case<T, 32>(*this, this->dev, 31, 17);
}

// ---- host side: with_tier and the workspace plans --------------------------------------------
dv::DeviceLimits limits(std::size_t local, bool fp64 = true) { return {local, {32}, 1024, fp64}; }

template <class T, class Routine> std::string tier_for(const dv::DeviceLimits& lim, int n) {
    std::string got;
    try {
        dv::with_tier<Routine, T>(lim, n, [&]<class Tier>(Tier) {
            if constexpr (dv::geqrf_detail::is_reg_v<Tier>) got = "reg" + std::to_string(Tier::n_max);
            else got = "cta" + std::to_string(Tier::n_max);
        });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string("throw:") + e.what();
    }
    return got;
}

TEST(DeviceApiQrHost, WithTierPicksTheSmallestAdmittingGeqrfTier) {
    using R = dv::routine::geqrf;
    const auto big = limits(std::size_t(100) << 10);
    EXPECT_EQ((tier_for<float, R>(big, 8)), "reg8");
    EXPECT_EQ((tier_for<float, R>(big, 9)), "reg16");
    EXPECT_EQ((tier_for<float, R>(big, 17)), "reg32");
    EXPECT_EQ((tier_for<float, R>(big, 33)), "cta64");
    EXPECT_EQ((tier_for<float, R>(big, 65)), "cta128");
    EXPECT_EQ((tier_for<float, R>(big, 129)), "throw:geqrf: no device tier admits n=129");
    // complex<double> has no Reg<32>: n = 17 goes to the Cta tier.
    EXPECT_EQ((tier_for<std::complex<double>, R>(big, 17)), "cta64");
    // A 32 KB device: Cta<128> float holds n = 90 (32400 B) and not n = 91 (33124 B).
    const auto metal = limits(std::size_t(32) << 10);
    EXPECT_EQ((tier_for<float, R>(metal, 90)), "cta128");
    EXPECT_EQ((tier_for<float, R>(metal, 91)).rfind("throw:", 0), 0u);
    // No fp64: no double tier at all.
    EXPECT_EQ((tier_for<double, R>(limits(std::size_t(100) << 10, false), 4)).rfind("throw:", 0), 0u);
}

TEST(DeviceApiQrHost, WithTierPicksTheOrmqrAndOrgqrPartitionWidth) {
    using L = dv::routine::ormqr<Side::Left, Transpose::NoTrans>;
    const auto big = limits(std::size_t(100) << 10);
    EXPECT_EQ((tier_for<float, L>(big, 4)), "cta4");
    EXPECT_EQ((tier_for<float, L>(big, 5)), "cta8");
    EXPECT_EQ((tier_for<float, L>(big, 32)), "cta32");
    EXPECT_EQ((tier_for<float, L>(big, 33)).rfind("throw:ormqr", 0), 0u);
    EXPECT_EQ((tier_for<double, dv::routine::orgqr>(big, 17)), "cta32");
    EXPECT_EQ((tier_for<double, dv::routine::orgqr>(big, 33)).rfind("throw:orgqr", 0), 0u);
}

// buffer_size is the bytes the op carves: Reg's tile N x (C + 1) then y (C), Cta's m * n
// stepped over the CUDA launch hole (47104, 49664] (straddled), ormqr Right's tile then v.
TEST(DeviceApiQrHost, BufferSizeIsTheCarvedPlan) {
    using D = dv::Cx<float>;
    constexpr auto reg = dv::geqrf_detail::RegPlan<D, 16, 8>{};
    static_assert(reg.tile.offset == 0 && reg.y.offset == 16 * 9 * sizeof(D));
    static_assert(dv::geqrf_buffer_size<dv::tier::Reg<16>, D, 8>(16).local_bytes == (16 * 9 + 8) * sizeof(D));
    static_assert(dv::geqrf_buffer_size<dv::tier::Reg<32>, float>(32).local_bytes == (32 * 33 + 32) * sizeof(float));
    EXPECT_EQ((dv::geqrf_buffer_size<dv::tier::Cta<128>, float>(108, 108).local_bytes), 108u * 108u * 4u);  // 46656
    EXPECT_EQ((dv::geqrf_buffer_size<dv::tier::Cta<128>, float>(110, 110).local_bytes), 49920u);           // 48400 -> pad
    EXPECT_EQ((dv::geqrf_buffer_size<dv::tier::Cta<128>, float>(112, 112).local_bytes), 112u * 112u * 4u);  // 50176
    static_assert(dv::ormqr_buffer_size<Side::Right, dv::tier::Cta<8>, double>(8).local_bytes == (64 + 8) * sizeof(double));
    static_assert(dv::ormqr_buffer_size<Side::Left, dv::tier::Cta<8>, double>(8).local_bytes == 8 * sizeof(double));
    static_assert(dv::orgqr_buffer_size<dv::tier::Cta<4>, float>(4).local_bytes == 16);
}

}  // namespace
