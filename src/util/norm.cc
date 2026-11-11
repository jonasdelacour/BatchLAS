// SPDX-License-Identifier: MIT
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/util/group-collectives.hh>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <src/util/batch_mask.hh>
#include <src/linalg-impl.hh>
#include <src/queue.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/norm.hh>
#include <batchlas/device/detail/sscp_target.hh>

namespace batchlas
{
    template <typename T, MatrixFormat MF, NormType NT, int WG> struct NormKernel;
    template <Backend B, typename T> struct NormSpectralKernel;

    template <Backend B, typename T>
    Event norm_spectral_vendor_impl(Queue &ctx,
                                   const MatrixView<T, MatrixFormat::Dense> &A,
                                   const Span<float_t<T>> norms) {
        if (A.rows() != A.cols()) {
            throw batchlas::invalid_argument("norm: Spectral norm requires square symmetric/Hermitian matrices");
        }

        using Real = typename base_type<T>::type;
        const int n = A.rows();
        const int batch_size = A.batch_size();

        const auto mask = detail::call_mask("norm", A);
        const std::uint8_t* mk = detail::mask_ptr(mask);
        const bool masked = mk != nullptr;  // a captured null pointer can read as non-null on AdaptiveCpp Metal
        Matrix<T, MatrixFormat::Dense> A_copy_store(A.rows(), A.cols(), A.batch_size(), A.ld(), A.stride());
        // The scratch copy carries the mask, so the copy, the mirror and syev skip the same items.
        const auto A_copy = A_copy_store.view().with_batch_mask(mask);
        Event copy_e = MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy, A);
        ctx.enqueue(copy_e);
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            A_copy.hermitize(ctx, Uplo::Lower).wait();
        } else {
            A_copy.symmetrize(ctx, Uplo::Lower).wait();
        }

        UnifiedVector<Real> eigenvalues(static_cast<size_t>(batch_size) * n);
        auto eigen_span = eigenvalues.to_span();

        // The public syev, so selection decides vendor vs native (a vendor-free build has no vendor).
        auto workspace = ctx.workspace(
            syev_buffer_size<B, T>(ctx, A_copy, eigen_span, JobType::NoEigenVectors, Uplo::Lower));
        Event e = syev<B, T>(ctx, A_copy, eigen_span, JobType::NoEigenVectors, Uplo::Lower, workspace.span());
        ctx.enqueue(e);

        ctx->parallel_for<NormSpectralKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> idx) {
            const size_t b = idx[0];
            if (masked && !mk[b]) return;
            // Hand-written, not an absrange op: behind any call boundary this loop's blocks are laid out after the
            // body, PTX gets the opposite latch polarity and ptxas unrolls it 4x more (LDG 14 -> 70).
            const Real* eig = eigen_span.data() + b * n;
            Real max_val = Real(0);
            for (int i = 0; i < n; ++i) {
                max_val = sycl::fmax(max_val, sycl::fabs(eig[i]));
            }
            norms[b] = max_val;
        });

        // A_copy, eigenvalues and the workspace lease die with this frame: finish before they do.
        ctx.wait();
        return ctx.get_event();
    }

    template <typename T>
    Event norm_spectral_impl(Queue &ctx,
                             const MatrixView<T, MatrixFormat::Dense> &A,
                             const Span<float_t<T>> norms) {
        const auto dev = ctx.device();
        const bool is_gpu = dev.type == DeviceType::GPU;
        const auto vendor = dev.get_vendor();

        #if BATCHLAS_HAS_CUDA_BACKEND
        if (is_gpu && vendor == Vendor::NVIDIA) {
            return norm_spectral_vendor_impl<Backend::CUDA>(ctx, A, norms);
        }
        #endif

        #if BATCHLAS_HAS_ROCM_BACKEND
        if (is_gpu && vendor == Vendor::AMD) {
            return norm_spectral_vendor_impl<Backend::ROCM>(ctx, A, norms);
        }
        #endif

        #if BATCHLAS_HAS_SYCL_BACKEND
        if (is_gpu && ctx.backend() == Backend::SYCL) {
            if constexpr (detail::is_fp64_scalar_v<T> && !BATCHLAS_HAS_DEVICE_FP64) {
                detail::throw_no_device_fp64(ctx, "norm");
            } else {
                return norm_spectral_vendor_impl<Backend::SYCL>(ctx, A, norms);
            }
        }
        #endif

        #if BATCHLAS_HAS_HOST_BACKEND
        return norm_spectral_vendor_impl<Backend::NETLIB>(ctx, A, norms);
        #else
        throw batchlas::unsupported("norm: Spectral norm requires a vendor backend (CUDA/ROCM/NETLIB)");
        #endif
    }

    // One work-group per item over the public device norm (a thin launcher); NT and WG are types,
    // so each norm is its own lange body.
    template <NormType NT, int WG, typename T, MatrixFormat MF>
    void norm_launch(Queue &ctx, const MatrixView<T, MF> &A, const Span<float_t<T>> norms, const std::uint8_t* mk) {
        const bool masked = mk != nullptr;  // a captured null pointer can read as non-null on AdaptiveCpp Metal
        ctx -> submit([&](sycl::handler &cgh) {
            auto [data, batch_size, rows, cols, ld, stride] =
                std::make_tuple(A.data_ptr(), A.batch_size(), A.rows(), A.cols(), A.ld(), A.stride());
            batchlas::impl::parallel_for<NormKernel<T, MF, NT, WG>>(cgh, sycl::nd_range<1>(size_t(batch_size) * WG, WG), [=](sycl::nd_item<1> item) {
                const auto b = item.get_group_linear_id();
                if (masked && !mk[b]) return;  // I1: uniform per work-group, before any barrier
                const device::WorkGroup<WG> ex{item};
                const auto r = device::norm<NT>(ex, device::mat(data + b * stride, rows, cols, ld));
                if (ex.leader()) norms[b] = r;
            });
        });
    }

    template <NormType NT, typename T, MatrixFormat MF>
    void norm_launch(Queue &ctx, const MatrixView<T, MF> &A, const Span<float_t<T>> norms, const std::uint8_t* mk) {
        const auto max_wg = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
        if (static_cast<int64_t>(A.rows()) * A.cols() <= 1024 || max_wg < 256) norm_launch<NT, 32>(ctx, A, norms, mk);
        else norm_launch<NT, 256>(ctx, A, norms, mk);
    }

    template <typename T, MatrixFormat MF>
    Event norm_impl(Queue &ctx,
                   const MatrixView<T, MF> &A,
                   const NormType norm_type,
                   const Span<float_t<T>> norms){
        const std::uint8_t* mk = detail::mask_ptr(detail::call_mask("norm", A));
        switch (norm_type) {
            case NormType::Frobenius: norm_launch<NormType::Frobenius>(ctx, A, norms, mk); break;
            case NormType::One: norm_launch<NormType::One>(ctx, A, norms, mk); break;
            case NormType::Inf: norm_launch<NormType::Inf>(ctx, A, norms, mk); break;
            case NormType::Max: norm_launch<NormType::Max>(ctx, A, norms, mk); break;
            case NormType::Spectral: throw batchlas::invalid_argument("norm: Spectral is not a lange norm");
        }
        return ctx.get_event();
    }

    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms)
    {
        (void)detail::call_mask("norm", A);
        detail::require_device_fp64<T>(ctx, "norm");
        if (norm_type == NormType::Spectral) {
            if constexpr (MF != MatrixFormat::Dense) {
                throw batchlas::unsupported("norm: Spectral norm only supported for dense symmetric/Hermitian matrices");
            } else {
                return norm_spectral_impl(ctx, A, norms);
            }
        }
        return norm_impl(ctx, A, norm_type, norms);
    }

    template <typename T, MatrixFormat MF>
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type)
    {
        (void)detail::call_mask("norm", A);
        detail::require_device_fp64<T>(ctx, "norm");
        // Allocate memory for the results
        UnifiedVector<float_t<T>> norms(A.batch_size());
        if (norm_type == NormType::Spectral) {
            if constexpr (MF != MatrixFormat::Dense) {
                throw batchlas::unsupported("norm: Spectral norm only supported for dense symmetric/Hermitian matrices");
            } else {
                norm_spectral_impl(ctx, A, norms.to_span()).wait();
            }
        } else {
            norm_impl(ctx, A, norm_type, norms.to_span()).wait();
        }
        return norms;
    }

    #define NORM_INSTANTIATE(fp, fmt) \
    template Event norm<fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType,\
        const Span<float_t<fp>>);\
    template UnifiedVector<float_t<fp>> norm<fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        const NormType);

    NORM_INSTANTIATE(float, MatrixFormat::Dense)
    NORM_INSTANTIATE(double, MatrixFormat::Dense)
    NORM_INSTANTIATE(std::complex<float>, MatrixFormat::Dense)
    NORM_INSTANTIATE(std::complex<double>, MatrixFormat::Dense)

} // namespace batchlas