#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/residuals.hh>

#include <cstddef>
#include <vector>

namespace test_utils {

// Worst componentwise |C - op(Q) C0| / (|op(Q)||C0|) (Side::Right: C0 op(Q)) over the batch, with Q
// formed on the host by batchlas::verify from the stored reflectors, independently of any ormqr.
template <typename T>
double ormqr_apply_error(const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& A_fact, const batchlas::VectorView<T>& tau,
                         const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C0,
                         const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C, batchlas::Side side, batchlas::Transpose trans) {
    using D = batchlas::verify::promoted_t<T>;
    using batchlas::verify::Shape;
    const int n = A_fact.rows();
    double worst = 0;
    for (int b = 0; b < C.batch_size(); ++b) {
        const auto Q = batchlas::verify::form_q(A_fact.view(), tau, b, n);
        const auto Qv = batchlas::verify::view(Q.data(), n, n, n);
        const auto C0b = batchlas::verify::view(C0.view().data_ptr() + static_cast<std::size_t>(b) * C0.view().stride(), n, n, C0.view().ld());
        const auto Cb = batchlas::verify::view(C.view().data_ptr() + static_cast<std::size_t>(b) * C.view().stride(), n, n, C.view().ld());
        const double err = side == batchlas::Side::Left
            ? batchlas::verify::gemm_backward_error(Qv, Shape::general, trans, C0b, Shape::general, batchlas::Transpose::NoTrans, C0b, Cb,
                                                    Shape::general, D(1), D(0))
            : batchlas::verify::gemm_backward_error(C0b, Shape::general, batchlas::Transpose::NoTrans, Qv, Shape::general, trans, C0b, Cb,
                                                    Shape::general, D(1), D(0));
        worst = batchlas::verify::nanmax(worst, err);
    }
    return worst;
}

// ormqr applied to the identity returns op(Q) (either side). Worst ||A0 - Q triu(R)||_F / ||A0||_F over
// the batch with that Q (conjugate-transposed back for a transposed op) and R from the geqrf factor F.
// Not componentwise: Q's near-zero entries make an elementwise relative error meaningless.
template <typename T>
double ormqr_q_error(const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& A0,
                     const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& F,
                     const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& C, batchlas::Transpose trans) {
    const int n = A0.rows();
    const auto item = [](const batchlas::Matrix<T, batchlas::MatrixFormat::Dense>& M, int b) {
        return M.view().data_ptr() + static_cast<std::size_t>(b) * M.view().stride();
    };
    std::vector<T> qh(static_cast<std::size_t>(n) * n);
    double worst = 0;
    for (int b = 0; b < C.batch_size(); ++b) {
        const T* c = item(C, b);
        const int ldc = C.view().ld();
        auto Q = batchlas::verify::view(c, n, n, ldc);
        if (trans != batchlas::Transpose::NoTrans) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    qh[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n] =
                        batchlas::verify::conj(c[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * ldc]);
            Q = batchlas::verify::view(static_cast<const T*>(qh.data()), n, n, n);
        }
        worst = batchlas::verify::nanmax(
            worst, batchlas::verify::qr_reconstruction(batchlas::verify::view(item(A0, b), n, n, A0.view().ld()), Q,
                                                       batchlas::verify::view(item(F, b), n, n, F.view().ld())));
    }
    return worst;
}

} // namespace test_utils
