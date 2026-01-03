#include <sycl/sycl.hpp>
#include <complex>
#include <blas/enums.hh>
#include "queue.hh"


namespace batchlas {
    namespace internal {
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
                return std::hypot(value.real(), value.imag());
            } else {
                return std::fabs(value);
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

        template <typename K>
        inline constexpr K ceil_div(K num, K denom) {
            return (num + denom - 1) / denom;
        }

        template <typename T, typename Op = std::plus<T>>
        Event scan_exclusive_inplace(Queue& ctx, const Span<T>& data, T init = T(0), Op op = Op()) {
            ctx -> submit([&](sycl::handler& cgh) {
                auto max_wg_size = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
                auto bsize = data.size() > max_wg_size ? max_wg_size : data.size();
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(bsize), sycl::range<1>(bsize)), [=](sycl::nd_item<1> item) {
                    auto cta = item.get_group();
                    sycl::joint_exclusive_scan(cta, data.data(), data.data() + data.size(), data.data(), init, op);
                });
            });
            return ctx.get_event();
        }

        template <typename T, typename Op = std::plus<T>>
        Event scan_inclusive_inplace(Queue& ctx, const Span<T>& data, Op op = Op()) {
            ctx -> submit([&](sycl::handler& cgh) {
                auto max_wg_size = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
                auto bsize = data.size() > max_wg_size ? max_wg_size : data.size();
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(bsize), sycl::range<1>(bsize)), [=](sycl::nd_item<1> item) {
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
                const R ax = sycl::fabs(static_cast<R>(internal::abs(xi)));
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

            sycl::vec<R, 3> total = sycl::reduce_over_group(cta, part, sycl::plus<sycl::vec<R, 3>>());

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
        
        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
        inline constexpr T safmin(){
            constexpr auto eps = std::numeric_limits<T>::epsilon();
            constexpr auto s1 = std::numeric_limits<T>::min();
            constexpr auto s2 = T(1) / std::numeric_limits<T>::max();
            return s2 > s1 ? s2 * (T(1) + eps) : s1;
        }

        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
        inline constexpr T safmax(){
            return T(1) / safmin<T>();
        }

        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
        inline constexpr T ssfmin(){
            return std::sqrt(safmin<T>()) / (std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon());
        }

        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
        inline constexpr T ssfmax(){
            return std::sqrt(safmax<T>()) / T(3.0);
        }

        //Matches LAPACK's DLAMCH/SLAMCH 'E' option
        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
        inline constexpr T rounding_eps(){
            return std::numeric_limits<T>::epsilon() / T(2);
        }

        //Matches LAPACK's "eps2" used in DLAED2/DSTEQR
        template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
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
        inline constexpr auto lartg(const T& f, const T& g)
        {
            using R = base_float_t<T>;

            R c = R(0);
            R s = R(0);
            R r = R(0);
            R rtmin = std::sqrt(safmin<R>());
            R rtmax = std::sqrt(safmax<R>() / R(2));

            R f1 = std::abs(static_cast<R>(f));
            R g1 = std::abs(static_cast<R>(g));

            if (g == T(0)) {
                c = R(1);
                s = R(0);
                r = static_cast<R>(f);
            } else if (f == T(0)) {
                c = R(0);
                s = std::copysign(R(1), g);
                r = g1;
            } else if (f1 > rtmin && f1 < rtmax &&
                        g1 > rtmin && g1 < rtmax) {
                auto d = std::sqrt(f * f + g * g);
                c = f1 / d;
                r = std::copysign(d, f);
                s = g / r;
            } else {
                auto u = std::min( safmax<R>(), std::max( safmin<R>(), std::max(f1, g1) ) );
                R fs = f / u;
                R gs = g / u;
                auto d = std::sqrt(fs * fs + gs * gs);
                c = std::abs(fs) / d;
                r = std::copysign(d, f);
                s = gs / r;
                r = r * u;
            }

            return std::array{c, s, r};
        }

        template <typename T>
        inline constexpr auto eigenvalues_2x2(const T& a, const T& b, const T& c) {
            // LAPACK SLAE2/DLAE2-style stable computation of eigenvalues of a 2x2
            // symmetric matrix [[A, B], [B, C]]. If (b != c), we symmetrize.
            const auto one  = T(1);
            const auto two  = T(2);
            const auto zero = T(0);
            const auto half = T(0.5);

            const auto sm  = a + c;          // sum (trace)
            const auto df  = a - c;          // difference
            const auto adf = std::abs(df);
            const auto tb  = b + b;          // 2*B
            const auto ab  = std::abs(tb);

            // Select larger/smaller of A and C by magnitude as in LAPACK
            const auto acmx = (std::abs(a) > std::abs(c)) ? a : c;
            const auto acmn = (std::abs(a) > std::abs(c)) ? c : a;

            T rt; // sqrt(adf^2 + (2B)^2) without overflow/underflow
            if (adf > ab) {
                rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
            } else if (adf < ab) {
                rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
            } else { // includes case ab = adf = 0
                rt = ab * std::sqrt(two);
            }

            T rt1, rt2;
            if (sm < zero) {
                rt1 = half * (sm - rt);
                // Order of operations important for an accurate smaller eigenvalue
                rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
            } else if (sm > zero) {
                rt1 = half * (sm + rt);
                rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
            } else {
                // includes case rt1 = rt2 = 0
                rt1 = half * rt;
                rt2 = -half * rt;
            }

            return std::array{rt1, rt2};
        }

        template <typename T>
        inline constexpr auto laev2(const T& a, const T& b, const T& c) {
            // LAPACK SLAEV2-style stable computation of eigenvalues and eigenvector
            // of a 2x2 symmetric matrix [[A, B], [B, C]]. Returns {rt1, rt2, cs1, sn1},
            // where rt1 >= rt2 are the eigenvalues and [cs1, sn1]^T is the normalized
            // eigenvector corresponding to rt1.

            const T one  = T(1);
            const T two  = T(2);
            const T zero = T(0);
            const T half = T(0.5);

            // Local scalars
            int sgn1 = 0;
            int sgn2 = 0;

            const T sm  = a + c;
            const T df  = a - c;
            const T adf = std::abs(df);
            const T tb  = b + b;      // 2*B
            const T ab  = std::abs(tb);

            T acmx, acmn;
            if (std::abs(a) > std::abs(c)) {
                acmx = a;
                acmn = c;
            } else {
                acmx = c;
                acmn = a;
            }

            T rt;
            if (adf > ab) {
                rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
            } else if (adf < ab) {
                rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
            } else {
                // Includes case ab = adf = 0
                rt = ab * std::sqrt(two);
            }

            T rt1, rt2;
            if (sm < zero) {
                rt1 = half * (sm - rt);
                sgn1 = -1;
                // Order of operations important for the smaller eigenvalue
                rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
            } else if (sm > zero) {
                rt1 = half * (sm + rt);
                sgn1 = 1;
                // Order of operations important for the smaller eigenvalue
                rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
            } else {
                // Includes case rt1 = rt2 = 0
                rt1 = half * rt;
                rt2 = -half * rt;
                sgn1 = 1;
            }

            // Compute the eigenvector corresponding to rt1
            T cs;
            if (df >= zero) {
                cs   = df + rt;
                sgn2 = 1;
            } else {
                cs   = df - rt;
                sgn2 = -1;
            }

            const T acs = std::abs(cs);
            T cs1, sn1;

            if (acs > ab) {
                const T ct = -tb / cs;
                sn1        = one / std::sqrt(one + ct * ct);
                cs1        = ct * sn1;
            } else {
                if (ab == zero) {
                    cs1 = one;
                    sn1 = zero;
                } else {
                    const T tn = -cs / tb;
                    cs1        = one / std::sqrt(one + tn * tn);
                    sn1        = tn * cs1;
                }
            }

            if (sgn1 == sgn2) {
                const T tn = cs1;
                cs1        = -sn1;
                sn1        = tn;
            }

            return std::array{rt1, rt2, cs1, sn1};
        }

    }



}