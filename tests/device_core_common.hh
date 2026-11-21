// SPDX-License-Identifier: MIT
#pragma once
// Shared harness of the device_core_<header>_tests: a queue, USM up/down, a one-WG-per-item
// launcher at sub-group size 32, deterministic data with nonzero imaginary parts, and the
// fixture that skips without a device, without sub-group size 32, or for an FP64 type on a device
// without aspect::fp64. References are scalar host code in each test, never the op under test.

#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/queue.hh>

namespace device_core_test {

using namespace batchlas;

template <class Scalar> using real_of = typename device::Arith<Scalar>::real_t;
template <class Scalar> constexpr bool is_cplx = device::Arith<Scalar>::is_complex;

inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// Uniform in [-1, 1): seeded, reproducible, independent of the op under test.
template <class Real> Real uval(std::uint64_t i) { return static_cast<Real>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class Scalar> Scalar val(std::uint64_t i) {
    if constexpr (is_cplx<Scalar>) return Scalar(uval<real_of<Scalar>>(2 * i), uval<real_of<Scalar>>(2 * i + 1));
    else return uval<Scalar>(i);
}
template <class Scalar> std::vector<Scalar> vals(std::size_t n, std::uint64_t seed) {
    std::vector<Scalar> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = val<Scalar>(seed * 1000003u + i);
    return v;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible() ? std::optional<Queue>(std::in_place, Device::default_device())
                                                                 : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(std::max<std::size_t>(h.size(), 1), q());
        if (!h.empty()) q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> T* alloc(std::size_t n, const T& fill) { return up(std::vector<T>(n, fill)); }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    // batch work-groups of WG items, sub-group size 32; f(it, local bytes).
    template <int WG, class F> void run(int batch, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 f(it, batchlas::util::get_raw_ptr(lm));
             });
         }).wait_and_throw();
    }
    // True when the device flushes a subnormal Real operand to zero (Metal). Probed by a
    // kernel: AdaptiveCpp reports fp_config::denorm on every device.
    template <class Real> bool flushes_subnormals() {
        Real* p = up(std::vector<Real>{std::numeric_limits<Real>::denorm_min() * Real(4), Real(1)});
        run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
            if (it.get_local_linear_id() == 0) p[0] = p[0] * p[1];
        });
        return down(p, 1)[0] == Real(0);
    }
    ~Dev() {
        if (ctx)
            for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class Scalar> class DeviceCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        const auto sgs = dev.q().get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sgs.begin(), sgs.end(), std::size_t(32)) == sgs.end()) GTEST_SKIP() << "device_core tests launch sub-group size 32";
        if (test_utils::lacks_fp64<Scalar>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using AllTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

// The host-side scalar reference arithmetic in long double (complex parts separately).
template <class Scalar> using LD = std::conditional_t<is_cplx<Scalar>, std::complex<long double>, long double>;
template <class Scalar> LD<Scalar> wide(Scalar v) { return LD<Scalar>(v); }
template <class Scalar> double err(Scalar got,
                                   LD<Scalar> ref) { return static_cast<double>(std::abs(wide(got) - ref)); }
template <class Scalar> double tol(double scale = 1.0) { return scale * 64.0 * static_cast<double>(std::numeric_limits<real_of<Scalar>>::epsilon()); }

}  // namespace device_core_test
