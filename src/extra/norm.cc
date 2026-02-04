#include <blas/extra.hh>
#include <blas/functions/syev.hh>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include "../linalg-impl.hh"
#include "../queue.hh"
#include "../math-helpers.hh"

namespace batchlas
{   
    template <typename T, MatrixFormat MF> struct NormKernel;
    template <Backend B, typename T> struct NormSpectralKernel;

    template <Backend B, typename T>
    Event norm_spectral_vendor_impl(Queue &ctx,
                                   const MatrixView<T, MatrixFormat::Dense> &A,
                                   const Span<float_t<T>> norms) {
        if (A.rows() != A.cols()) {
            throw std::runtime_error("norm: Spectral norm requires square symmetric/Hermitian matrices");
        }

        using Real = typename base_type<T>::type;
        const int n = A.rows();
        const int batch_size = A.batch_size();

        Matrix<T, MatrixFormat::Dense> A_copy(A.rows(), A.cols(), A.batch_size(), A.ld(), A.stride());
        Event copy_e = MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy.view(), A);
        ctx.enqueue(copy_e);
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            A_copy.view().hermitize(ctx, Uplo::Lower).wait();
        } else {
            A_copy.view().symmetrize(ctx, Uplo::Lower).wait();
        }

        UnifiedVector<Real> eigenvalues(static_cast<size_t>(batch_size) * n);
        auto eigen_span = eigenvalues.to_span();

        const size_t ws_bytes = backend::syev_vendor_buffer_size<B, T>(ctx,
                                                                        A_copy.view(),
                                                                        eigen_span,
                                                                        JobType::NoEigenVectors,
                                                                        Uplo::Lower);
        UnifiedVector<std::byte> workspace(ws_bytes);
        Event e = backend::syev_vendor<B, T>(ctx,
                                             A_copy.view(),
                                             eigen_span,
                                             JobType::NoEigenVectors,
                                             Uplo::Lower,
                                             workspace.to_span());
        ctx.enqueue(e);

        ctx->parallel_for<NormSpectralKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> idx) {
            const size_t b = idx[0];
            const Real* eig = eigen_span.data() + b * n;
            Real max_val = Real(0);
            for (int i = 0; i < n; ++i) {
                max_val = sycl::fmax(max_val, sycl::fabs(eig[i]));
            }
            norms[b] = max_val;
        });

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

        #if BATCHLAS_HAS_MKL_BACKEND
        if (!is_gpu && vendor == Vendor::INTEL) {
            return norm_spectral_vendor_impl<Backend::MKL>(ctx, A, norms);
        }
        #endif

        #if BATCHLAS_HAS_HOST_BACKEND
        return norm_spectral_vendor_impl<Backend::NETLIB>(ctx, A, norms);
        #else
        throw std::runtime_error("norm: Spectral norm requires a vendor backend (CUDA/ROCM/MKL/NETLIB)");
        #endif
    }

    template <typename T, MatrixFormat MF>
    Event norm_impl(Queue &ctx,
                   const MatrixView<T, MF> &A,
                   const NormType norm_type,
                   const Span<float_t<T>> norms){
        //Zero extra memory allocation kernels
        ctx -> submit([&](sycl::handler &cgh) {
            // Get the data pointers and sizes
            auto [data, batch_size, rows, cols, ld, stride] = 
                std::make_tuple(A.data_ptr(), A.batch_size(), A.rows(), A.cols(), A.ld(), A.stride());

            auto wgs = std::min(size_t(rows * cols), get_kernel_max_wg_size<NormKernel<T, MF>>(ctx));
            auto local_mem = sycl::local_accessor<float_t<T>>(  norm_type == NormType::One ? cols :
                                                                norm_type == NormType::Inf ? rows : 0, cgh);
            // Use a parallel_for to compute the norms
            // All kernels assume that matrices are stored in column-major order
            cgh.parallel_for<NormKernel<T, MF>>(sycl::nd_range<1>(batch_size * wgs, wgs), [=](sycl::nd_item<1> item) {
                auto batch_idx = item.get_group_linear_id();
                auto local_idx = item.get_local_linear_id();
                auto local_size = item.get_local_range()[0];
                auto cta = item.get_group();
                float_t<T> temp = 0;
                float_t<T> result = 0;
                auto data_span = Span<T>(data + batch_idx * stride, cols * ld);
                if (norm_type == NormType::Frobenius) {
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % rows;
                        temp += internal::norm_squared(data_span[col * ld + row]);
                    }
                    result = sycl::sqrt(sycl::reduce_over_group(cta, temp, sycl::plus<float_t<T>>()));
                } else if (norm_type == NormType::One) {
                    // Initialize local memory with zeros for column sums
                    if (local_idx < cols) {
                        local_mem[local_idx] = 0;
                    }
                    sycl::group_barrier(cta);

                    // Sum absolute values for each column across all rows
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % rows;
                        sycl::atomic_ref<float_t<T>, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group>
                            atomic_local_mem(local_mem[col]);
                        atomic_local_mem += internal::abs(data_span[col * ld + row]);
                    }
                    sycl::group_barrier(cta);

                    // Find the maximum column sum (the One norm)
                    result = sycl::joint_reduce(cta, local_mem.begin(), local_mem.end(), float_t<T>(0), sycl::maximum<float_t<T>>());
                } else if (norm_type == NormType::Inf) {
                    // Initialize local memory with zeros for row sums
                    if (local_idx < rows) {
                        local_mem[local_idx] = 0;
                    }
                    sycl::group_barrier(cta);

                    // Sum absolute values for each row across all columns
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % rows;
                        sycl::atomic_ref<float_t<T>, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group>
                            atomic_local_mem(local_mem[row]);
                        atomic_local_mem += internal::abs(data_span[col * ld + row]);
                    }
                    sycl::group_barrier(cta);

                    // Find the maximum row sum (the Inf norm)
                    result = sycl::joint_reduce(cta, local_mem.begin(), local_mem.end(), float_t<T>(0), sycl::maximum<float_t<T>>());
                } else if (norm_type == NormType::Max) {
                    // Find the maximum absolute value in the matrix
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % cols;
                        temp = std::max(internal::abs(data_span[col * ld + row]), temp);
                    }
                    result = sycl::reduce_over_group(cta, temp, sycl::maximum<float_t<T>>());
                }

                norms[batch_idx] = result;
            });
        }); 

        return ctx.get_event();
    }

    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms)
    {
        if (norm_type == NormType::Spectral) {
            if constexpr (MF != MatrixFormat::Dense) {
                throw std::runtime_error("norm: Spectral norm only supported for dense symmetric/Hermitian matrices");
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
        // Allocate memory for the results
        UnifiedVector<float_t<T>> norms(A.batch_size());
        if (norm_type == NormType::Spectral) {
            if constexpr (MF != MatrixFormat::Dense) {
                throw std::runtime_error("norm: Spectral norm only supported for dense symmetric/Hermitian matrices");
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