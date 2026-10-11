// SPDX-License-Identifier: MIT
// The host dgemm probe behind batchlas::verify's double LAPACKE references (verify/reference.hh).
#include <batchlas/verify/reference.hh>

#if BATCHLAS_VERIFY_HAVE_LAPACKE
#include <cblas.h>

#include "../backends/host_blas_health.hh"

namespace batchlas::verify::detail {

std::string host_dgemm_defect() {
    static const std::string defect = [] {
        const auto health = batchlas::detail::probe_dgemm([](int n, const double* a, const double* b, double* c) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
        });
        return health.ok ? std::string() : batchlas::detail::host_blas_double_health_message(health);
    }();
    return defect;
}

}  // namespace batchlas::verify::detail
#endif
