// potri's element-wise launches (potri_native.hh). No shared or sub-group state, so no
// work-group size to derive and nothing CTA-capped.

#include <src/ops/potri/potri_native.hh>

#include <src/sycl/device_scalar.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_potri {

namespace {

template <typename T> class PotriDiagInfoKernel;
template <typename T> class PotriStoreKernel;

}  // namespace

template <typename T>
Event potri_diag_info(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<int32_t> info) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    const D* const ap = reinterpret_cast<const D*>(A.data_ptr());
    const int n = static_cast<int>(A.rows());
    const std::ptrdiff_t lda = A.ld(), stride = A.stride();
    int32_t* const out = info.data();
    const std::size_t batch = static_cast<std::size_t>(A.batch_size());

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotriDiagInfoKernel<T>>(sycl::range<1>(batch), [=](sycl::item<1> it) {
            const D* const Ab = ap + static_cast<std::ptrdiff_t>(it.get_id(0)) * stride;
            int32_t v = 0;
            // EXACT zero, as ?POTRI reports: the first diagonal entry that is a true binary zero.
            for (int i = 0; i < n && v == 0; ++i)
                if (sycl_device::dev_is_zero(Ab[static_cast<std::ptrdiff_t>(i) * lda + i])) v = i + 1;
            out[it.get_id(0)] = v;
        });
    });
    return ctx.get_event();
}

template <typename T>
Event potri_store(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& X,
                  const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, bool full) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    const D* const xp = reinterpret_cast<const D*>(X.data_ptr());
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    const int n = static_cast<int>(A.rows());
    const std::ptrdiff_t ldx = X.ld(), sx = X.stride(), lda = A.ld(), sa = A.stride();
    const bool lower = uplo == Uplo::Lower;
    const std::size_t batch = static_cast<std::size_t>(A.batch_size());
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotriStoreKernel<T>>(sycl::range<2>(batch, elems), [=](sycl::item<2> it) {
            const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_id(0));
            const std::size_t e = it.get_id(1);
            const int r = static_cast<int>(e % static_cast<std::size_t>(n));
            const int c = static_cast<int>(e / static_cast<std::size_t>(n));
            const bool owned = lower ? r >= c : r <= c;
            const D* const Xb = xp + b * sx;
            D* const Ab = ap + b * sa;
            if (owned) {
                const D v = Xb[static_cast<std::ptrdiff_t>(c) * ldx + r];
                // The inverse of a Hermitian matrix has a real diagonal: drop the rounding noise in its imaginary part.
                Ab[static_cast<std::ptrdiff_t>(c) * lda + r] = r == c ? sycl_device::dev_from_real<D>(sycl_device::dev_real(v)) : v;
            } else if (full) {
                Ab[static_cast<std::ptrdiff_t>(c) * lda + r] =
                    sycl_device::dev_conj(Xb[static_cast<std::ptrdiff_t>(r) * ldx + c]);
            }
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_POTRI_INSTANTIATE(T)                                                                    \
    template Event potri_diag_info<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<int32_t>); \
    template Event potri_store<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                     \
                                  const MatrixView<T, MatrixFormat::Dense>&, Uplo, bool);

BATCHLAS_POTRI_INSTANTIATE(float)
BATCHLAS_POTRI_INSTANTIATE(double)
BATCHLAS_POTRI_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRI_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRI_INSTANTIATE

}  // namespace batchlas::sycl_potri
