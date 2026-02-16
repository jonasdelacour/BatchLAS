#include <blas/linalg.hh>
#include "../math-helpers.hh"

namespace batchlas {

template <MatrixFormat MType, typename T>
Event lascl(Queue& ctx, const MatrixView<T, MType>& mat, T cfrom, T cto) {
    constexpr T zero = T(0);
    constexpr T one  = T(1);
    const T smlnum = internal::smlnum<T>();
    const T bignum = T(1) / smlnum;

    T cfromc = cfrom;
    T ctoc   = cto;

    T cfrom1, cto1, mul;
    bool done = false;
    Event last_event;

    while (!done) {
        cfrom1 = cfromc * smlnum;
        cto1   = ctoc / bignum;

        if (std::abs(cfrom1) > std::abs(ctoc) && ctoc != zero) {
            mul    = smlnum;
            done   = false;
            cfromc = cfrom1;
        } else if (std::abs(cto1) > std::abs(cfromc)) {
            mul  = bignum;
            done = false;
            ctoc = cto1;
        } else {
            mul  = ctoc / cfromc;
            done = true;
        }

        // Scale the matrix by mul
        last_event = scale(ctx, mul, mat);
    }

    return last_event;
}

template Event lascl<MatrixFormat::Dense, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, float, float);
template Event lascl<MatrixFormat::Dense, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, double, double);
//template Event lascl<MatrixFormat::Dense, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, std::complex<float>, std::complex<float>);
//template Event lascl<MatrixFormat::Dense, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, std::complex<double>, std::complex<double>);

}