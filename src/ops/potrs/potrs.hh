// SPDX-License-Identifier: MIT
#pragma once

// The solve with a Cholesky factor (kernels/potrs_trsm.cc), shared by potrs, posv's cta and blocked
// and potri. evidence: docs/perf/potri.md#potrs-and-potri

#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/getrs/getrs_native.hh>
#include <src/select/select.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::ops::potrs {

// Correctness only: can the one-launch fused solve take an \f$n \times nrhs\f$ system here? Never a speed term.
template <class Scalar>
bool fused_fits(const select::Device& d, int64_t n, int64_t nrhs) {
    const auto cap = ops::getrs::getrs_fused_max_rhs_elems<Scalar>(static_cast<std::size_t>(d.slm_budget));
    return select::natives_run_on(d) && d.has_sg32 && n >= 1 && nrhs >= 1 && nrhs <= ops::getrs::kGetrsFusedMaxRhs &&
           static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs) <= cap;
}

template <Backend B, class Scalar>
Event solve(Queue& q, const MatrixView<Scalar, MatrixFormat::Dense>& A, const MatrixView<Scalar,
            MatrixFormat::Dense>& Bm, Uplo uplo, bool fused);

}  // namespace batchlas::ops::potrs
