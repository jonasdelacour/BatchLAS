#include <blas/extra.hh>
#include <cstddef>
#include "../queue.hh"

namespace batchlas
{   
    template <typename T, MatrixFormat MF> struct NormKernel;

    template <typename T, MatrixFormat MF>
    Event norm_impl(Queue &ctx,
                   const MatrixView<T, MF> &A,
                   const NormType norm_type,
                   const Span<T>& norms){
        //Zero extra memory allocation kernels
        ctx -> submit([&](sycl::handler &cgh) {
            // Get the data pointers and sizes
            auto [data, batch_size, rows, cols, ld, stride] = 
                std::make_tuple(A.data_ptr(), A.batch_size(), A.rows(), A.cols(), A.ld(), A.stride());

            auto wgs = std::min(rows * cols, get_kernel_max_wg_size<NormKernel<T, MF>>(ctx));
            auto local_mem = sycl::local_accessor<T>(norm_type == NormType::One ? cols :
                                                     norm_type == NormType::Inf ? rows : 0, cgh);
            // Use a parallel_for to compute the norms
            // All kernels assume that matrices are stored in column-major order
            cgh.parallel_for<NormKernel<T, MF>>(sycl::nd_range<1>(batch_size * wgs, wgs), [=](sycl::nd_item<1> item) {
                auto batch_idx = item.get_group_linear_id();
                auto local_idx = item.get_local_linear_id();
                auto local_size = item.get_local_range()[0];
                auto cta = item.get_group();
                T temp = 0;
                T norm = 0;
                auto data_span = Span<T>(data + batch_idx * stride, cols * ld);
                if (norm_type == NormType::Frobenius) {
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j % cols;
                        auto row = j / cols;
                        temp += data_span[col * ld + row] * data_span[col * ld + row];
                    }
                    norm = sycl::sqrt(sycl::reduce_over_group(cta, temp));
                } else if (norm_type == NormType::One) {
                    // Initialize local memory with zeros for column sums
                    if (local_idx < cols) {
                        local_mem[local_idx] = 0;
                    }
                    sycl::group_barrier(cta);

                    // Sum absolute values for each column across all rows
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % cols;
                        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group>
                            atomic_local_mem(local_mem[col]);
                        atomic_local_mem += sycl::fabs(data_span[col * ld + row]);
                    }
                    sycl::group_barrier(cta);

                    // Find the maximum column sum (the One norm)
                    norm = sycl::reduce_over_group(cta, local_mem.begin(), local_mem.end(), T(0), sycl::maximum<T>());
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
                        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group>
                            atomic_local_mem(local_mem[row]);
                        atomic_local_mem += sycl::fabs(data_span[col * ld + row]);
                    }
                    sycl::group_barrier(cta);

                    // Find the maximum row sum (the Inf norm)
                    norm = sycl::reduce_over_group(cta, local_mem.begin(), local_mem.end(), T(0), sycl::maximum<T>());
                } else if (norm_type == NormType::Max) {
                    // Find the maximum absolute value in the matrix
                    for (int j = local_idx; j < rows * cols; j += local_size) {
                        auto col = j / rows;
                        auto row = j % cols;
                        temp = sycl::fmax(sycl::fabs(data_span[col * ld + row]), temp);
                    }
                    norm = sycl::reduce_over_group(cta, norm);
                }

                norms[batch_idx] = norm;
            });
        });

        return ctx.get_event();
    }

    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> norms)
    {
        norm_impl(ctx, A, norm_type, norms);
    }

    template <typename T, MatrixFormat MF>
    UnifiedVector<T> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type)
    {
        // Allocate memory for the results
        UnifiedVector<T> norms(A.batch_size());
        norm_impl(ctx, A, norm_type, norms);
        return norms;
    }
} // namespace batchlas