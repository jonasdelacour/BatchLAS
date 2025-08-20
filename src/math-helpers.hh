#include <sycl/sycl.hpp>
#include <complex>
#include <blas/enums.hh>

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
        base_float_t<T> abs(const T& value) {
            if constexpr (is_complex<T>::value) {
                return std::hypot(value.real(), value.imag());
            } else {
                return std::fabs(value);
            }
        }

        template <typename T>
        bool is_numerically_zero(const T& value) {
            return abs(value) < std::numeric_limits<base_float_t<T>>::epsilon();
        }

        template <typename T>
        base_float_t<T> norm_squared(const T& value) {
            if constexpr (is_complex<T>::value) {
                return std::real(std::conj(value) * value); // For complex numbers, return the squared norm
            } else {
                return value * value;
            }
        }

        template <typename K>
        K ceil_div(K num, K denom) {
            return (num + denom - 1) / denom;
        }
    }



}