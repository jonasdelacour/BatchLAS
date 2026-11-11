// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gebrd.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <stdexcept>

namespace batchlas {

namespace {


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
            device::gebd2_real<T>(device::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld()), b, D, E, TAUQ, TAUP);
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
            device::gebd2_complex<T>(device::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld()), b, D, E, TAUQ, TAUP);
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
