// SPDX-License-Identifier: MIT
#pragma once

#include <sycl/sycl.hpp>
#include <batchlas/util/group-collectives.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/util/group-invoke.hh>
#include <batchlas/blas/device/v2/scalar.hh>

#include <complex>

#include <src/queue.hh>
#include <src/sycl/sscp_target.hh>


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
        inline constexpr bool is_numerically_zero(const T& value) {
            return abs(value) < std::numeric_limits<base_float_t<T>>::epsilon();
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


        template <typename T>
        inline constexpr auto nrm2(const sycl::group<1>& cta, const VectorView<T>& v)
        {
            using R = base_float_t<T>;

            const auto tid  = cta.get_local_linear_id();
            const auto bdim = cta.get_local_range()[0];
            const auto bid  = cta.get_group_linear_id();
            const int  N    = static_cast<int>(v.size());

            // ---- Blue's scaling constants (match dnrm2.f90 formulas) ----
            const int radix = std::numeric_limits<R>::radix;      // typically 2
            const int p     = std::numeric_limits<R>::digits;     // mantissa bits (e.g. 53)
            const int emin  = std::numeric_limits<R>::min_exponent; // min exponent for normalized
            const int emax  = std::numeric_limits<R>::max_exponent; // max exponent

            auto floor_div = [](int a, int b) {
                int q = a / b;
                // ensure mathematical floor for negatives
                if (((a ^ b) < 0) && (a % b)) --q;
                return q;
            };
            auto ceil_div = [](int a, int b) {
                int q = a / b;
                // ensure mathematical ceil
                if (((a ^ b) >= 0) && (a % b)) ++q;
                return q;
            };

            const int e_tsml = ceil_div(emin - 1, 2);
            const int e_tbig = floor_div(emax - p + 1, 2);
            const int e_ssml = -floor_div(emin - p, 2);
            const int e_sbig = -ceil_div(emax + p - 1, 2);

            const R tsml = sycl::pown(static_cast<R>(radix), e_tsml);
            const R tbig = sycl::pown(static_cast<R>(radix), e_tbig);
            const R ssml = sycl::pown(static_cast<R>(radix), e_ssml);
            const R sbig = sycl::pown(static_cast<R>(radix), e_sbig);

            // ---- Three accumulators per work-item ----
            struct Three { R abig; R amed; R asml; };
            sycl::vec<R, 3> part{R(0), R(0), R(0)};

            for (int i = tid; i < N; i += bdim) {
                const T xi = v(i, bid);
                const R ax = sycl::fabs(static_cast<R>(detail::abs(xi)));
                if (ax > tbig) {
                    const R t = ax * sbig;
                    part[0] += t * t;
                } else if (ax < tsml) {
                    const R t = ax * ssml;
                    part[2] += t * t;
                } else {
                    part[1] += ax * ax;
                }
            }

            sycl::vec<R, 3> total = batchlas::portable::reduce_over_group(cta, part, sycl::plus<sycl::vec<R, 3>>());

            // ---- Final combination (verbatim logic from dnrm2) ----
            const R zero = R(0);
            const R one  = R(1);
            const R maxN = std::numeric_limits<R>::max();

            R scl = one;
            R sumsq = zero;

            if (total[0] > zero) {
                // combine abig and amed
                if ((total[1] > zero) || (total[1] > maxN) || (total[1] != total[1])) {
                    const R tmp = (total[1] * sbig) * sbig;
                    total[0] += tmp;
                }
                scl = one / sbig;
                sumsq = total[0];
            } else if (total[2] > zero) {
                // combine amed and asml
                if ((total[1] > zero) || (total[1] > maxN) || (total[1] != total[1])) {
                    const R am = sycl::sqrt(total[1]);
                    const R as = sycl::sqrt(total[2]) / ssml;
                    const R ymin = sycl::fmin(am, as);
                    const R ymax = sycl::fmax(am, as);
                    scl   = one;
                    sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
                } else {
                    scl   = one / ssml;
                    sumsq = total[2];
                }
            } else {
                // all mids
                scl   = one;
                sumsq = total[1];
            }

            return scl * sycl::sqrt(sumsq);
        }

        template <typename T>
        inline constexpr T ipow(T base, int e) {
            T result = T(1);
            if (e < 0) {
                base = T(1) / base;
                e = -e;
            }
            while (e > 0) {
                if (e & 1) result *= base;
                base *= base;
                e >>= 1;
            }
            return result;
        }

        template <RealScalar T>
        inline constexpr T safmin(){ return ::batchlas::device::v2::safmin<T>(); }

        template <RealScalar T>
        inline constexpr T safmax(){ return ::batchlas::device::v2::safmax<T>(); }

        template <RealScalar T>
        inline constexpr T ssfmin(){
            return std::sqrt(safmin<T>()) / (std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon());
        }

        template <RealScalar T>
        inline constexpr T ssfmax(){
            return std::sqrt(safmax<T>()) / T(3.0);
        }

        //Matches LAPACK's DLAMCH/SLAMCH 'E' option
        template <RealScalar T>
        inline constexpr T rounding_eps(){
            return std::numeric_limits<T>::epsilon() / T(2);
        }

        //Matches LAPACK's "eps2" used in DLAED2/DSTEQR
        template <RealScalar T>
        inline constexpr T eps2(){
            return rounding_eps<T>() * rounding_eps<T>();
        }

        template <typename T>
        inline constexpr T sbig() {
            constexpr T e = std::ceil( (std::numeric_limits<T>::max_exponent + std::numeric_limits<T>::digits - 1) * T(0.5) );
            return ipow(std::numeric_limits<T>::radix, -static_cast<int>(e));
        }

        template <typename T>
        inline constexpr T smlnum() {
            return safmin<T>() / std::numeric_limits<T>::epsilon();
        }

        template <typename T>
        inline constexpr T ssml(){
            constexpr T e = std::floor( (std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits) * T(0.5) );
            return ipow(std::numeric_limits<T>::radix, -static_cast<int>(e));
        }

        template <typename T>
        inline constexpr T tbig() {
            constexpr T e = std::floor( (std::numeric_limits<T>::max_exponent - std::numeric_limits<T>::digits + 1) * T(0.5) );
            return ipow(std::numeric_limits<T>::radix, static_cast<int>(e));
        }

        template <typename T>
        using lartg_result = ::batchlas::device::v2::LartgResult<T>;

        // Three fields, not v2::LarfgScalars (five): invoke_one_broadcast moves it whole.
        template <typename T>
        struct larfg_result {
            T tau;
            T beta;
            T scale;
        };

        // Device LA v2 scalar lartg; complex<float> works in double except on Metal (no fp64,
        // chosen at JIT time) to cut rounding noise in the downstream bulge chase.
        template <typename T>
        __attribute__((always_inline)) inline lartg_result<T> lartg(const T& f, const T& g)
        {
            if constexpr (is_complex<T>::value && std::is_same_v<base_float_t<T>, float>) {
                if (sycl_impl::on_metal()) return ::batchlas::device::v2::lartg<float>(f, g);
                return ::batchlas::device::v2::lartg<double>(f, g);
            } else {
                return ::batchlas::device::v2::lartg(f, g);
            }
        }

        // The phase-keeping larfg convention (device::v2::conv::Phase).
        template <typename T>
        __attribute__((always_inline)) inline larfg_result<T> larfg(const T& alpha, base_float_t<T> xnorm, int len)
        {
            const auto s = ::batchlas::device::v2::larfg_scalars<::batchlas::device::v2::conv::Phase>(alpha, xnorm, len);
            return {s.tau, s.beta, s.scale};
        }

        template <typename T>
        __attribute__((always_inline)) inline auto eigenvalues_2x2(const T& a, const T& b, const T& c) {
            return ::batchlas::device::v2::lae2(a, b, c);
        }

        // The eigenvalue of [[a, b], [b, c]] closest to c.
        template <typename T>
        inline T wilkinson_shift(const T& a, const T& b, const T& c) {
            const auto [lambda1, lambda2] = eigenvalues_2x2(a, b, c);
            return std::abs(lambda1 - c) < std::abs(lambda2 - c) ? lambda1 : lambda2;
        }

        template <typename T>
        __attribute__((always_inline)) inline std::array<T, 2> las2(const T& f, const T& g, const T& h) {
            return ::batchlas::device::v2::las2(f, g, h);
        }

        template <typename T>
        __attribute__((always_inline)) inline auto laev2(const T& a, const T& b, const T& c) {
            return ::batchlas::device::v2::laev2(a, b, c);
        }

    }



}