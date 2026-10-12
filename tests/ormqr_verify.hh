#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/tolerance.hh>

#include <cstddef>
#include <vector>

namespace test_utils {

// Worst componentwise |C - op(Q) C0| / (|op(Q)||C0|) (Side::Right: C0 op(Q)) over the batch, with Q
// formed by MatrixView::form_q from the stored reflectors, independently of any ormqr.
template <typename T>
double ormqr_apply_error(batchlas::Queue& ctx, const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& A_fact,
                         const batchlas::VectorView<T>& tau, const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C0,
                         const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C, batchlas::Side side, batchlas::Transpose trans) {
    using MV = batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>;
    using D = typename MV::promoted_type;
    using DV = batchlas::MatrixView<D, batchlas::MatrixFormat::Dense>;
    const int n = A_fact.rows();
    double worst = 0;
    for (int b = 0; b < C.batch_size(); ++b) {
        const auto Q = MV::form_q(ctx, A_fact.view(), tau, b, n);
        const auto C0b = C0.view().batch_item(b).template astype<D>();
        const auto Cb = C.view().batch_item(b).template astype<D>();
        const double err = side == batchlas::Side::Left
            ? DV::gemm_backward_error(ctx, Q.view(), trans, C0b.view(), batchlas::Transpose::NoTrans, C0b.view(), Cb.view(), D(1), D(0))
            : DV::gemm_backward_error(ctx, C0b.view(), batchlas::Transpose::NoTrans, Q.view(), trans, C0b.view(), Cb.view(), D(1), D(0));
        worst = batchlas::verify::nanmax(worst, err);
    }
    return worst;
}

// ormqr applied to the identity returns op(Q) (either side). Worst ||A0 - Q triu(R)||_F / ||A0||_F over
// the batch with that Q (conjugate-transposed back for a transposed op) and R from the geqrf factor F.
// Not componentwise: Q's near-zero entries make an elementwise relative error meaningless.
template <typename T>
double ormqr_q_error(batchlas::Queue& ctx, const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& A0,
                     const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& F,
                     const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C, batchlas::Transpose trans) {
    using MV = batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>;
    const int n = A0.rows();
    batchlas::Matrix<T, batchlas::MatrixFormat::Dense> qh(n, n, 1);
    double worst = 0;
    for (int b = 0; b < C.batch_size(); ++b) {
        MV Q = C.view().batch_item(b);
        if (trans != batchlas::Transpose::NoTrans) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) qh(i, j, 0) = MV::conj(C.view()(j, i, b));
            Q = qh.view();
        }
        worst = batchlas::verify::nanmax(worst, MV::qr_reconstruction(ctx, A0.view().batch_item(b), Q, F.view().batch_item(b)));
    }
    return worst;
}

} // namespace test_utils
