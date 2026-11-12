// SPDX-License-Identifier: MIT
#pragma once

#include <sycl/sycl.hpp>
#include <batchlas/util/group-collectives.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/scalar.hh>

#include <complex>

#include <src/queue.hh>
#include <batchlas/device/detail/sscp_target.hh>


namespace batchlas {
    namespace detail {
        template <typename T>
        using base_float_t = typename base_type<T>::type;

        template <typename T>
        struct is_complex : std::false_type {};

        template <>
        struct is_complex<std::complex<float>> : std::true_type {};

        template <>
        struct is_complex<std::complex<double>> : std::true_type {};

        template <typename T>
        inline constexpr base_float_t<T> abs(const T& value) {
            if constexpr (is_complex<T>::value) {
                return sycl::hypot(value.real(), value.imag());
            } else {
                return sycl::fabs(value);
            }
        }

        template <typename T>
        inline constexpr base_float_t<T> norm_squared(const T& value) {
            if constexpr (is_complex<T>::value) {
                return std::real(std::conj(value) * value); // For complex numbers, return the squared norm
            } else {
                return value * value;
            }
        }

        template <typename T>
        inline base_float_t<T> real_part(const T& x) {
            if constexpr (is_complex<T>::value) {
                return static_cast<base_float_t<T>>(x.real());
            } else {
                return static_cast<base_float_t<T>>(x);
            }
        }

        // re*re + im*im in explicit real arithmetic; not interchangeable with norm_squared bit for bit.
        template <typename T>
        inline base_float_t<T> abs2_if_complex(const T& x) {
            using Real = base_float_t<T>;
            if constexpr (is_complex<T>::value) {
                const Real re = x.real();
                const Real im = x.imag();
                return re * re + im * im;
            } else {
                return x * x;
            }
        }

        template <typename Real>
        inline Real sign_nonzero_real(Real x) {
            return (sycl::signbit(x) ? Real(-1) : Real(1));
        }

        // x/|x| (complex) or +-1 by sign bit (real); never 0, and 1 for a complex zero.
        template <typename T>
        inline T sign_nonzero(const T& x) {
            using Real = base_float_t<T>;
            if constexpr (is_complex<T>::value) {
                const Real a = sycl::hypot(x.real(), x.imag());
                if (a == Real(0)) return T(1);
                return x / a;
            } else {
                return T(sign_nonzero_real(static_cast<Real>(x)));
            }
        }

        template <typename K>
        inline constexpr K ceil_div(K num, K denom) {
            return (num + denom - 1) / denom;
        }

        template <typename T, typename Op = std::plus<T>>
        Event scan_inclusive_inplace(Queue& ctx, const Span<T>& data, Op op = Op()) {
            ctx -> submit([&](sycl::handler& cgh) {
                auto max_wg_size = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
                auto bsize = data.size() > max_wg_size ? max_wg_size : data.size();
                batchlas::impl::parallel_for(cgh, sycl::nd_range<1>(sycl::range<1>(bsize), sycl::range<1>(bsize)), [=](sycl::nd_item<1> item) {
                    auto cta = item.get_group();
                    sycl::joint_inclusive_scan(cta, data.data(), data.data() + data.size(), data.data(), op);
                });
            });
            return ctx.get_event();
        }


        using ::batchlas::device::safmin;
        using ::batchlas::device::safmax;

    }



}