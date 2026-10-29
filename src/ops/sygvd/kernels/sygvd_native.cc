// sygvd's element-wise launches (sygvd_native.hh). No shared or sub-group state.

#include <src/ops/sygvd/sygvd_native.hh>

#include <src/sycl/device_scalar.hh>
#include <src/queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_sygvd {

namespace {

template <typename T> class SygvdIdentityFailedKernel;
class SygvdMergeInfoKernel;

}  // namespace

template <typename T>
Event sygvd_identity_failed(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                            Span<const int32_t> potrf_info, const std::uint8_t* mask) {
    using D = typename sycl_device::DevMap<T>::type;
    using R = typename base_type<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    D* const bp = reinterpret_cast<D*>(B.data_ptr());
    const int n = B.rows();
    const std::ptrdiff_t ldb = B.ld(), stride = B.stride();
    const int32_t* const fail = potrf_info.data();
    const bool lower = uplo == Uplo::Lower;
    const std::size_t batch = static_cast<std::size_t>(B.batch_size());
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<SygvdIdentityFailedKernel<T>>(sycl::range<2>(batch, elems), [=](sycl::item<2> it) {
            const std::size_t b = it.get_id(0);
            if (mask && !mask[b]) return;  // I3
            if (fail[b] == 0) return;
            const std::size_t e = it.get_id(1);
            const int r = static_cast<int>(e % static_cast<std::size_t>(n));
            const int c = static_cast<int>(e / static_cast<std::size_t>(n));
            if (lower ? r < c : r > c) return;
            bp[static_cast<std::ptrdiff_t>(b) * stride + static_cast<std::ptrdiff_t>(c) * ldb + r] =
                sycl_device::dev_from_real<D>(R(r == c ? 1 : 0));
        });
    });
    return ctx.get_event();
}

Event sygvd_merge_info(Queue& ctx, int n, Span<const int32_t> potrf_info, Span<int32_t> info,
                       const std::uint8_t* mask) {
    const int32_t* const fail = potrf_info.data();
    int32_t* const out = info.data();
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<SygvdMergeInfoKernel>(sycl::range<1>(info.size()), [=](sycl::item<1> it) {
            if (mask && !mask[it.get_id(0)]) return;  // I3
            const int32_t f = fail[it.get_id(0)];
            if (f != 0) out[it.get_id(0)] = n + f;
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_SYGVD_INSTANTIATE(T)                                                                          \
    template Event sygvd_identity_failed<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo,                \
                                            Span<const int32_t>, const std::uint8_t*);

BATCHLAS_SYGVD_INSTANTIATE(float)
BATCHLAS_SYGVD_INSTANTIATE(double)
BATCHLAS_SYGVD_INSTANTIATE(std::complex<float>)
BATCHLAS_SYGVD_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SYGVD_INSTANTIATE

}  // namespace batchlas::sycl_sygvd
