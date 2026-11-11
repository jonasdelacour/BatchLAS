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

        template <RealScalar T>
        inline constexpr T ssfmin(){
            return std::sqrt(safmin<T>()) / (::batchlas::device::lamch<'P', T>() * ::batchlas::device::lamch<'P', T>());
        }

        template <RealScalar T>
        inline constexpr T ssfmax(){
            return std::sqrt(safmax<T>()) / T(3.0);
        }

        //Matches LAPACK's "eps2" used in DLAED2/DSTEQR: dlamch('E') squared
        template <RealScalar T>
        inline constexpr T eps2(){
            return ::batchlas::device::lamch<'E', T>() * ::batchlas::device::lamch<'E', T>();
        }

        // Device LA scalar lartg; complex<float> works in double except on Metal (no fp64,
        // chosen at JIT time) to cut rounding noise in the downstream bulge chase.
        template <typename T>
        __attribute__((always_inline)) inline ::batchlas::device::LartgResult<T> lartg(const T& f, const T& g)
        {
            if constexpr (is_complex<T>::value && std::is_same_v<base_float_t<T>, float>) {
                if (sycl_impl::on_metal()) return ::batchlas::device::lartg<float>(f, g);
                return ::batchlas::device::lartg<double>(f, g);
            } else {
                return ::batchlas::device::lartg(f, g);
            }
        }

        template <typename T>
        __attribute__((always_inline)) inline auto eigenvalues_2x2(const T& a, const T& b, const T& c) {
            return ::batchlas::device::lae2(a, b, c);
        }

        // The eigenvalue of [[a, b], [b, c]] closest to c.
        template <typename T>
        inline T wilkinson_shift(const T& a, const T& b, const T& c) {
            const auto [lambda1, lambda2] = eigenvalues_2x2(a, b, c);
            return std::abs(lambda1 - c) < std::abs(lambda2 - c) ? lambda1 : lambda2;
        }

        // Binds core laev2's operands by reference: a using-declaration (by value) costs the steqr
        // legacy / wg Francis kernels 3 LDG and 4 REG on sm_89. Goes when core laev2 takes const R&.
        template <typename T>
        __attribute__((always_inline)) inline auto laev2(const T& a, const T& b, const T& c) {
            return ::batchlas::device::laev2(a, b, c);
        }

    }



}