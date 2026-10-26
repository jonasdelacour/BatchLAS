// potri's small launches (potri_native.hh): the info scan (one 32-wide work-group per item) and
// the triangle finish (element-wise, grid-strided).

#include <src/ops/potri/potri_native.hh>
#include <batchlas/error.hh>

#include <src/sycl/device_scalar.hh>
#include <src/queue.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <src/util/template-instantiations.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

namespace batchlas::sycl_potri {

namespace {

template <typename T> class PotriDiagInfoKernel;
template <typename T> class PotriFinishKernel;

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
    // The kernel writes out[0, batch) unguarded: refuse a short span here, not only at the entry point.
    if (info.size() < batch)
        throw batchlas::internal_error("potri: info span of " + std::to_string(info.size()) +
                                       " entries reached the info kernel for a batch of " + std::to_string(batch));

    // One 32-wide work-group per item: the diagonal's loads overlap instead of chaining per item.
    constexpr int kWg = 32;
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotriDiagInfoKernel<T>>(sycl::nd_range<1>(batch * kWg, kWg), [=](sycl::nd_item<1> it) {
            const std::size_t b = it.get_group_linear_id();
            const D* const Ab = ap + static_cast<std::ptrdiff_t>(b) * stride;
            int first = n;
            // EXACT zero, as ?POTRI reports: the first diagonal entry that is a true binary zero.
            for (int i = static_cast<int>(it.get_local_linear_id()); i < n; i += kWg)
                if (sycl_device::dev_is_zero(Ab[static_cast<std::ptrdiff_t>(i) * lda + i])) first = sycl::min(first, i);
            first = sycl::reduce_over_group(it.get_group(), first, sycl::minimum<int>());
            if (it.get_local_linear_id() == 0) out[b] = first == n ? 0 : first + 1;
        });
    });
    return ctx.get_event();
}

template <typename T>
Event potri_finish(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, bool full) {
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

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotriFinishKernel<T>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
            for (std::size_t f = it.get_global_id(0); f < total; f += it.get_global_range(0)) {
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(f / elems);
                const std::size_t e = f % elems;
                const int r = full ? static_cast<int>(e % static_cast<std::size_t>(n)) : static_cast<int>(e);
                const int c = full ? static_cast<int>(e / static_cast<std::size_t>(n)) : r;
                D* const Ab = ap + b * sa;
                if (r == c) {
                    D& d = Ab[static_cast<std::ptrdiff_t>(c) * lda + r];
                    d = sycl_device::dev_from_real<D>(sycl_device::dev_real(d));
                } else if (lower ? r < c : r > c) {
                    Ab[static_cast<std::ptrdiff_t>(c) * lda + r] =
                        sycl_device::dev_conj(Ab[static_cast<std::ptrdiff_t>(r) * lda + c]);
                }
            }
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_POTRI_INSTANTIATE(T)                                                                    \
    template Event potri_diag_info<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<int32_t>); \
    template Event potri_finish<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, bool);

BATCHLAS_POTRI_INSTANTIATE(float)
BATCHLAS_POTRI_INSTANTIATE(double)
BATCHLAS_POTRI_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRI_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRI_INSTANTIATE

}  // namespace batchlas::sycl_potri
