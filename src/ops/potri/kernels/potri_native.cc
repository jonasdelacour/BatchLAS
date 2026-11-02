// potri's triangle finish (potri_native.hh): element-wise, grid-strided; each element is one
// v2 hermitian_fill_at (include/batchlas/blas/device/v2/ext/potri.hh).

#include <src/ops/potri/potri_native.hh>
#include <batchlas/error.hh>

#include <src/sycl/device_scalar.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <batchlas/blas/device/v2/ext/potri.hh>
#include <src/queue.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_potri {

namespace {

namespace v2 = batchlas::device::v2;

template <typename T> class PotriFinishKernel;

}  // namespace

template <typename T>
Event potri_finish(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, bool full, const int32_t* skip) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    const int n = static_cast<int>(A.rows());
    const std::ptrdiff_t lda = A.ld(), sa = A.stride();
    const bool lower = uplo == Uplo::Lower;
    // Without full only the diagonal is touched: n entries per item, not n^2.
    const std::size_t elems = full ? static_cast<std::size_t>(n) * static_cast<std::size_t>(n) : static_cast<std::size_t>(n);
    const std::size_t total = static_cast<std::size_t>(A.batch_size()) * elems;
    const auto [global_size, local_size] = compute_nd_range_sizes(total, ctx.device(), KernelType::ELEMENTWISE);
    const std::uint8_t* const mask = detail::mask_ptr(detail::call_mask("potri", A));

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotriFinishKernel<T>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
            for (std::size_t f = it.get_global_id(0); f < total; f += it.get_global_range(0)) {
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(f / elems);
                if ((skip && skip[b] != 0) || (mask && mask[b] == 0)) continue;
                const std::size_t e = f % elems;
                const int r = full ? static_cast<int>(e % static_cast<std::size_t>(n)) : static_cast<int>(e);
                const int c = full ? static_cast<int>(e / static_cast<std::size_t>(n)) : r;
                D* const Ab = ap + b * sa;
                const auto Am =
                    v2::mat_fn(n, n, [&](int i, int j) -> D& { return Ab[static_cast<std::ptrdiff_t>(j) * lda + i]; });
                v2::hermitian_fill_at(lower, Am, r, c);
            }
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_POTRI_INSTANTIATE(T) \
    template Event potri_finish<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, bool, const int32_t*);

BATCHLAS_POTRI_INSTANTIATE(float)
BATCHLAS_POTRI_INSTANTIATE(double)
BATCHLAS_POTRI_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRI_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRI_INSTANTIATE

}  // namespace batchlas::sycl_potri
