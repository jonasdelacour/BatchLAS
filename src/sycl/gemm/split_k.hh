#pragma once

#include "../gemm_kernels.hh"

#include "register_launchers.hh"

#include <memory>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace {

template <typename T>
struct GemmSplitKReduceKernel;

constexpr int kExperimentalSplitKPartitions = 4;

template <typename T>
inline bool can_use_split_k_128x32x32_experimental(const MatrixView<T, MatrixFormat::Dense>& A,
                                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                                   Transpose transA,
                                                   Transpose transB) {
    if constexpr (!std::is_same_v<T, float>) {
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(transA);
        static_cast<void>(transB);
        return false;
    } else {
        if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
            return false;
        }

        const int m = A.rows();
        const int k = A.cols();
        const int n = B.cols();
        if (m < 256 || n < 256 || k < 256) {
            return false;
        }
        if ((m % 128) != 0 || (n % 32) != 0 || (k % (32 * kExperimentalSplitKPartitions)) != 0) {
            return false;
        }

        return can_use_aligned_nn_fast_path<T, 128, 32, 32, 4, 4>(A, B, C);
    }
}

template <typename T>
Event reduce_split_k_partials(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& partials,
                              const MatrixView<T, MatrixFormat::Dense>& C,
                              int split_k_partitions,
                              T alpha,
                              T beta) {
    const int rows = C.rows();
    const int cols = C.cols();
    const int batch_size = C.batch_size();
    constexpr size_t local_size = 256;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(batch_size);
    const size_t global_size = ((total + local_size - 1) / local_size) * local_size;

    ctx->submit([&](sycl::handler& h) {
        const T* partial_ptr = partials.data_ptr();
        T* c_ptr = C.data_ptr();
        const int partial_ld = partials.ld();
        const int partial_stride = partials.stride();
        const int c_ld = C.ld();
        const int c_stride = C.stride();

        h.parallel_for<GemmSplitKReduceKernel<T>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            const size_t global_id = item.get_global_id(0);
            const size_t total_work_items = item.get_global_range(0);
            const size_t matrix_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);

            for (size_t flat = global_id; flat < total; flat += total_work_items) {
                const int bid = static_cast<int>(flat / matrix_elems);
                const size_t rem = flat % matrix_elems;
                const int col = static_cast<int>(rem / static_cast<size_t>(rows));
                const int row = static_cast<int>(rem % static_cast<size_t>(rows));

                T sum = T(0);
                for (int split_idx = 0; split_idx < split_k_partitions; ++split_idx) {
                    const int partial_batch = split_idx * batch_size + bid;
                    sum += partial_ptr[partial_batch * partial_stride + col * partial_ld + row];
                }

                const int c_index = bid * c_stride + col * c_ld + row;
                c_ptr[c_index] = LinearEpilogue<T>::apply(alpha, beta, sum, c_ptr[c_index]);
            }
        });
    });

    return ctx.get_event();
}

template <typename T>
Event launch_register_128x32_k32_split_k4(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A,
                                          const MatrixView<T, MatrixFormat::Dense>& B,
                                          const MatrixView<T, MatrixFormat::Dense>& C,
                                          T alpha,
                                          T beta,
                                          Transpose transA,
                                          Transpose transB,
                                          const char* (*kernel_trace_name)(KernelVariant)) {
    if constexpr (!std::is_same_v<T, float>) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(alpha);
        static_cast<void>(beta);
        static_cast<void>(transA);
        static_cast<void>(transB);
        static_cast<void>(kernel_trace_name);
        throw std::runtime_error("Experimental split-K GEMM is currently only implemented for float");
    } else {
        if (!can_use_split_k_128x32x32_experimental(A, B, C, transA, transB)) {
            throw std::runtime_error("Experimental split-K GEMM requires large aligned NN float inputs with K divisible by 128");
        }

        BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(KernelVariant::Tiled128x32RegisterK32SplitK4));

        const int rows = C.rows();
        const int cols = C.cols();
        const int batch_size = C.batch_size();
        const int k = A.cols();

        auto partials = std::make_shared<Matrix<T, MatrixFormat::Dense>>(rows, cols, batch_size * kExperimentalSplitKPartitions);
        auto partials_view = partials->view();
        const size_t partial_batch_stride = static_cast<size_t>(partials_view.stride());

        for (int split_idx = 0; split_idx < kExperimentalSplitKPartitions; ++split_idx) {
            const int k_begin = (k * split_idx) / kExperimentalSplitKPartitions;
            const int k_end = (k * (split_idx + 1)) / kExperimentalSplitKPartitions;

            auto a_slice = A(Slice(), Slice(k_begin, k_end));
            auto b_slice = B(Slice(k_begin, k_end), Slice());
            MatrixView<T, MatrixFormat::Dense> partial_view(
                partials_view.data_ptr() + static_cast<size_t>(split_idx) * static_cast<size_t>(batch_size) * partial_batch_stride,
                rows,
                cols,
                partials_view.ld(),
                partials_view.stride(),
                batch_size);

            launch_register_128x32_k32_s2_u2(ctx, a_slice, b_slice, partial_view, T(1), T(0), kernel_trace_name);
        }

        reduce_split_k_partials(ctx, partials_view, C, kExperimentalSplitKPartitions, alpha, beta);

        ctx->submit([partials](sycl::handler& h) {
            h.host_task([partials]() {});
        });

        return ctx.get_event();
    }
}

} // namespace

} // namespace batchlas::sycl_gemm