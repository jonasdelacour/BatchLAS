// SPDX-License-Identifier: MIT
// Device LA v2 ops private to the krylov-ortho-ritz family (src/ops/ortho/krylov-ortho-ritz_v2.hh:
// Generated, laqge) against exact host references; the core ops those kernels also call are
// covered by device_blas_v2_tests. Guards (docs/developer/agent-guide.md section 8): a WG that
// is not a power of two and a launch at the device ceiling (laqge's contract: WG == rows);
// non-natural ld and inc; complex data with nonzero imaginary parts; padding poisoned with a
// large finite value the op must not touch; batch 1024 with period-3 data where every item must
// be bit-identical to its representative (the SLM publish path). Deliberate breaks this catches:
// dropping r(i) or c(j) from the scale, reading c unpublished before the barrier, a row loop
// that strides past rows into the padding.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/ops/ortho/krylov-ortho-ritz_v2.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kPeriod = 3;
constexpr double kPoison = 1.0e6;  // in range, finite: an op that writes padding changes it

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

// kPeriod representatives copied to every item b from representative b % kPeriod.
template <class T> std::vector<T> batched(int batch, int per_item, std::uint64_t seed) {
    std::vector<T> h(static_cast<std::size_t>(batch) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < batch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T poison() { return T(static_cast<real_of<T>>(kPoison)); }

// Items whose results differ from their representative's.
template <class T> int bit_twins(const std::vector<T>& h, int batch, std::size_t per_item) {
    int bad = 0;
    for (int b = kPeriod; b < batch; ++b)
        for (std::size_t i = 0; i < per_item; ++i) bad += !same_bits(h[b * per_item + i], h[(b % kPeriod) * per_item + i]);
    return bad;
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    int max_wg() const { return static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)); }
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
    // A run-time local range and no required sub-group size: the launch shape of the ortho kernels.
    template <class F> void run(int batch, int wg, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2KrylovTest : public ::testing::Test {
protected:
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2KrylovTest, Types);

// ---- laqge on RtGroup (WG = rows), c Generated or Published ----------------------------------

// E := diag(r) E diag(c) with r a strided global vector (inc 2) and c Generated; F := D F D with
// D staged in SLM and published (the SVQB D C D kernel). k = 37 at batch 1024, and k = the device
// ceiling (capped at 1024) at a small batch. ld = k + 2, padding poisoned. Every scale r(i) c(j)
// is one rounding in R and the scaling one more, both reproduced on the host, so the comparison
// is bit for bit.
TYPED_TEST(DeviceBlasV2KrylovTest, LaqgeGeneratedAndPublished) {
    using T = TypeParam;
    using R = real_of<T>;
    auto cgen = [](int j) { return R(1) + R(j % 64) * R(0.125); };
    for (const int k : {37, std::min(1024, this->dev.max_wg())}) {
        const int batch = k == 37 ? 1024 : 2 * kPeriod;
        const int ld = k + 2, per = ld * k, rinc = 2;
        auto pad = [&](std::vector<T> h) {
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < k; ++j)
                    for (int i = k; i < ld; ++i) h[std::size_t(b) * per + i + j * ld] = poison<T>();
            return h;
        };
        const auto hE = pad(batched<T>(batch, per, 32)), hF = pad(batched<T>(batch, per, 33));
        auto hd = batched<R>(batch, k * rinc, 34);
        for (auto& v : hd) v = R(0.5) + std::abs(v);  // positive scales, as rsqrt(diag) gives
        T* E = this->dev.up(hE);
        T* F = this->dev.up(hF);
        R* dg = this->dev.up(hd);
        this->dev.run(batch, k, sizeof(R) * k, [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::RtGroup<> ex{it};
            const auto r = v2::vec(dg + std::ptrdiff_t(b) * k * rinc, k, rinc);
            krylov_v2::laqge(ex, r, krylov_v2::generated(k, cgen), v2::mat(E + std::ptrdiff_t(b) * per, k, k, ld));
            const auto d_slm = v2::vec<v2::Mem::Local>(reinterpret_cast<R*>(lm), k);
            v2::for_each_owned_exact(ex, k, [&](int i) { d_slm(i) = r(i); });
            const auto D = ex.publish(d_slm);
            krylov_v2::laqge(ex, D, D, v2::mat(F + std::ptrdiff_t(b) * per, k, k, ld));
        });
        const auto gE = this->dev.down(E, hE.size()), gF = this->dev.down(F, hF.size());
        EXPECT_EQ(bit_twins(gE, batch, per) + bit_twins(gF, batch, per), 0) << "k " << k << ": items differ from their representative";
        auto scaled = [](T a, R sc) {
            if constexpr (is_cplx<T>) return T(a.real() * sc, a.imag() * sc);
            else return T(a * sc);
        };
        int wrong = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const std::size_t o = std::size_t(b) * per;
            const R* db = hd.data() + std::size_t(b) * k * rinc;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < ld; ++i) {
                    const std::size_t e = o + i + j * ld;
                    if (i >= k) {
                        wrong += !same_bits(gE[e], poison<T>()) + !same_bits(gF[e], poison<T>());
                        continue;
                    }
                    const R se = db[i * rinc] * cgen(j), sf = db[i * rinc] * db[j * rinc];
                    wrong += !same_bits(gE[e], scaled(hE[e], se));
                    wrong += !same_bits(gF[e], scaled(hF[e], sf));
                }
        }
        EXPECT_EQ(wrong, 0) << "k " << k << ": laqge (exact) wrong or padding touched";
    }
}

}  // namespace
