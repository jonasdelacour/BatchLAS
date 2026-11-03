// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/dla.hh>

#include <src/math-helpers.hh>
#include <src/ops/gesvd/gebrd/gebrd_v2.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <stdexcept>

namespace batchlas {

namespace {

namespace v2 = batchlas::device::v2;

// dgebd2 on one matrix (one work-item): H(i) from the left, then G(i) from the right. A function,
// not the kernel lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename MA, typename RV, typename TV>
BATCHLAS_DLA_INLINE void gebd2_real(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    const v2::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols(), k = std::min(m, n);
    for (int32_t i = 0; i < k; ++i) {
        // H(i) annihilates A(i+1:m, i), then A(i:m, i+1:n) := H(i) A(i:m, i+1:n).
        const auto xq = Ab.col(i).tail(i + 1);
        const T sigma = v2::dot(th, xq, xq);
        const auto hl = v2::larfg_scalars<v2::conv::Phase>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
        if (hl.tau != T(0)) {
            Ab(i, i) = T(1);
            v2::scal(th, hl.scale, xq);
            v2::larf<Side::Left>(th, hl.tau, v2::unit_head(Ab.col(i).tail(i), 0), Ab.sub(i, i + 1, m - i, n - i - 1));
        }
        Ab(i, i) = hl.beta;
        D(i, b) = hl.beta;
        TAUQ(i, b) = hl.tau;

        if (i >= n - 1) {
            TAUP(i, b) = T(0);
            continue;
        }

        // G(i) annihilates A(i, i+2:n), then A(i+1:m, i+1:n) := A(i+1:m, i+1:n) G(i).
        const auto xp = Ab.row(i).tail(i + 2);
        const T sigma_r = v2::dot(th, xp, xp);
        const auto hr = v2::larfg_scalars<v2::conv::Phase>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
        if (hr.tau != T(0)) {
            Ab(i, i + 1) = T(1);
            v2::scal(th, hr.scale, xp);
            v2::larf<Side::Right>(th, hr.tau, v2::unit_head(Ab.row(i).tail(i + 1), 0), Ab.sub(i + 1, i + 1, m - i - 1, n - i - 1));
        }
        Ab(i, i + 1) = hr.beta;
        E(i, b) = hr.beta;
        TAUP(i, b) = hr.tau;
    }

    for (int32_t i = k; i < TAUQ.size(); ++i) {
        TAUQ(i, b) = T(0);
    }
    for (int32_t i = std::max<int32_t>(0, k - 1); i < TAUP.size(); ++i) {
        TAUP(i, b) = T(0);
    }
}

// zgebd2 on one matrix: d, e real (ZLARFG's real beta); row i holds conj(u_i) on exit (unmbr 'P'
// order). H(i) is applied as H^H = I - conj(tau) v v^H, with v(0) = 1 stored.
template <typename T, typename MA, typename RV, typename TV>
BATCHLAS_DLA_INLINE void gebd2_complex(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    using Real = typename base_type<T>::type;
    const v2::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols();
    for (int32_t i = 0; i < n; ++i) {
        const auto xq = Ab.col(i).tail(i + 1);
        const Real sigma = v2::sumsq<v2::alg::Unfused>(th, xq);
        const auto lq = v2::larfg_scalars<v2::conv::RealBeta>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
        v2::scal(th, lq.scale, xq);
        Ab(i, i) = T(1);
        v2::larf<Side::Left>(th, v2::Arith<T>::conj(lq.tau), Ab.col(i).tail(i), Ab.sub(i, i + 1, m - i, n - i - 1));
        D(i, b) = detail::real_part(lq.beta);
        Ab(i, i) = lq.beta;
        TAUQ(i, b) = lq.tau;
        if (i >= n - 1) {
            TAUP(i, b) = T(0);
            continue;
        }
        // Row i conjugated (lacgv); G(i) annihilates A(i, i+2:n) (u(0) = 1 stored); A := A G.
        const auto row = Ab.row(i).tail(i + 1);
        const auto xp = Ab.row(i).tail(i + 2);
        gebrd_v2::lacgv(th, row);
        const Real sigma_r = v2::sumsq<v2::alg::Unfused>(th, xp);
        const auto rp = v2::larfg_scalars<v2::conv::RealBeta>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
        v2::scal(th, rp.scale, xp);
        Ab(i, i + 1) = T(1);
        v2::larf<Side::Right>(th, rp.tau, row, Ab.sub(i + 1, i + 1, m - i - 1, n - i - 1));
        gebrd_v2::lacgv(th, xp);
        E(i, b) = detail::real_part(rp.beta);
        Ab(i, i + 1) = rp.beta;
        TAUP(i, b) = rp.tau;
    }
}

// One work-item per matrix. The launch sits in a function per type, as in the legacy file, so
// the kernels keep their names.
template <typename T>
Event gebrd_unblocked_real(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<typename base_type<T>::type>& d,
                           const VectorView<typename base_type<T>::type>& e,
                           const VectorView<T>& tauq,
                           const VectorView<T>& taup,
                           const std::uint8_t* mask) {
    static_assert(!detail::is_complex<T>::value, "gebrd_unblocked_real expects real scalar type");
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto D = d;
        auto E = e;
        auto TAUQ = tauq;
        auto TAUP = taup;

        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3: an inactive item is untouched
            gebd2_real<T>(v2::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld()), b, D, E, TAUQ, TAUP);
        });
    });

    return ctx.get_event();
}

template <typename T>
Event gebrd_unblocked_complex(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<typename base_type<T>::type>& d,
                              const VectorView<typename base_type<T>::type>& e,
                              const VectorView<T>& tauq,
                              const VectorView<T>& taup,
                              const std::uint8_t* mask) {
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto D = d;
        auto E = e;
        auto TAUQ = tauq;
        auto TAUP = taup;
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3: an inactive item is untouched
            gebd2_complex<T>(v2::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld()), b, D, E, TAUQ, TAUP);
        });
    });
    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
Event gebrd_unblocked(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& a,
                      const VectorView<typename base_type<T>::type>& d,
                      const VectorView<typename base_type<T>::type>& e,
                      const VectorView<T>& tauq,
                      const VectorView<T>& taup) {
    const std::uint8_t* const mask =
        ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("gebrd_unblocked", a, d, e, tauq, taup));
    static_cast<void>(B);

    if (a.rows() < a.cols()) {
        throw batchlas::invalid_argument("gebrd_unblocked: current implementation requires rows >= cols");
    }
    if constexpr (detail::is_complex<T>::value) {
        return gebrd_unblocked_complex<T>(ctx, a, d, e, tauq, taup, mask);
    } else {
        return gebrd_unblocked_real<T>(ctx, a, d, e, tauq, taup, mask);
    }
}

#define GEBRD_INSTANTIATE(back, fp) \
    template Event gebrd_unblocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_INSTANTIATE)

#undef GEBRD_INSTANTIATE

} // namespace batchlas
