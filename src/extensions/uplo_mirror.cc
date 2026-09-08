#include "uplo_mirror.hh"

#include <sycl/sycl.hpp>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

namespace {
template <Backend B, typename T> class UploMirrorUpperToLowerKernel;
} // namespace

template <Backend B, typename T>
Event mirror_upper_to_lower(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& a) {
    const int n = static_cast<int>(a.rows());
    const int batch = static_cast<int>(a.batch_size());
    const int ld = static_cast<int>(a.ld());
    const int64_t stride = a.stride();
    T* ptr = a.data().data();

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<UploMirrorUpperToLowerKernel<B, T>>(
            sycl::range<3>(static_cast<size_t>(batch),
                           static_cast<size_t>(n),
                           static_cast<size_t>(n)),
            [=](sycl::id<3> idx) {
                const int b = static_cast<int>(idx[0]);
                const int r = static_cast<int>(idx[1]);
                const int c = static_cast<int>(idx[2]);
                // Only strictly-upper source entries do work; each writes its mirror.
                if (c <= r) return;
                T* A = ptr + b * stride;
                const T v = A[r + c * ld];          // (r, c) with r < c: upper triangle
                if constexpr (internal::is_complex<T>::value) {
                    A[c + r * ld] = T(v.real(), -v.imag());
                } else {
                    A[c + r * ld] = v;
                }
            });
    });
    return ctx.get_event();
}

#define UPLO_MIRROR_INSTANTIATE(back, fp) \
    template Event mirror_upper_to_lower<back, BATCHLAS_UNPAREN fp>( \
        Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(UPLO_MIRROR_INSTANTIATE)

#undef UPLO_MIRROR_INSTANTIATE

} // namespace batchlas
