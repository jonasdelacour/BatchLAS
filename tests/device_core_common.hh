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
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/queue.hh>

namespace device_core_test {

using namespace batchlas;
namespace dv = batchlas::device;

template <class T> using real_of = typename dv::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = dv::Arith<T>::is_complex;

inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// Uniform in [-1, 1): seeded, reproducible, independent of the op under test.
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> T val(std::uint64_t i) {
    if constexpr (is_cplx<T>) return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else return uval<T>(i);
}
template <class T> std::vector<T> vals(std::size_t n, std::uint64_t seed) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = val<T>(seed * 1000003u + i);
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
    ~Dev() {
        if (ctx)
            for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceCoreTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        const auto sgs = dev.q().get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sgs.begin(), sgs.end(), std::size_t(32)) == sgs.end()) GTEST_SKIP() << "device_core tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using AllTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

// The host-side scalar reference arithmetic in long double (complex parts separately).
template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> double err(T got, LD<T> ref) { return static_cast<double>(std::abs(wide(got) - ref)); }
template <class T> double tol(double scale = 1.0) { return scale * 64.0 * static_cast<double>(std::numeric_limits<real_of<T>>::epsilon()); }

}  // namespace device_core_test
