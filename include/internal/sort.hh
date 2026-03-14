#pragma once
#include <sycl/sycl.hpp>
#include <util/mempool.hh>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include "../../src/queue.hh"
#include <util/kernel-heuristics.hh>

namespace batchlas {

template <typename T, typename K>
Event permute(Queue& ctx, VectorView<T> data, VectorView<K> indices){
    auto n = data.size();
    auto batch_size = data.batch_size();
    // Since we use sycl::ext::oneapi::experimental::gather, we need to ensure data and indices use unit increments.
    if(data.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("permute: data and indices must have unit increment (inc=1)");
    }

    ctx -> submit([&](sycl::handler& h) {
        auto temp_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range()[0];
            auto tid = item.get_local_linear_id();
            for (int i = tid; i < n; i += bdim) temp_mem[i] = data(indices(i, bid), bid);
            sycl::group_barrier(item.get_group());
            for (int i = tid; i < n; i += bdim) data(i, bid) = temp_mem[i];
        });
    });
    return ctx.get_event();
}

template <typename T, typename K>
class ActiveArgsortKernel;

template <typename T, typename K>
Event argsort_active(Queue& ctx,
                     VectorView<T> data,
                     VectorView<K> indices,
                     Span<int32_t> active_lengths,
                     SortOrder order,
                     bool fill_indices) {
    const auto n = data.size();
    const auto batch_size = data.batch_size();
    if (data.inc() != 1 || indices.inc() != 1) {
        throw std::runtime_error("argsort_active: data and indices must have unit increment (inc=1)");
    }
    if (static_cast<int64_t>(active_lengths.size()) < batch_size) {
        throw std::runtime_error("argsort_active: active_lengths must cover every batch item");
    }

    ctx->submit([&](sycl::handler& h) {
        auto local_indices = sycl::local_accessor<K, 1>(sycl::range<1>(n), h);
        auto local_values = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);

        h.parallel_for<ActiveArgsortKernel<T, K>>(sycl::nd_range<1>(batch_size * 128, 128), [=](sycl::nd_item<1> item) {
            const auto bid = item.get_group_linear_id();
            const auto tid = item.get_local_linear_id();
            const auto bdim = item.get_local_range()[0];
            const auto cta = item.get_group();
            const int32_t active_n = std::max<int32_t>(0, std::min<int32_t>(active_lengths[bid], n));
            K* batch_indices = indices.batch_item(bid).data_ptr();

            if (fill_indices) {
                for (int i = tid; i < n; i += bdim) {
                    local_indices[i] = static_cast<K>(i);
                    local_values[i] = data(i, bid);
                }
            } else {
                for (int i = tid; i < n; i += bdim) {
                    local_indices[i] = batch_indices[i];
                    local_values[i] = data(local_indices[i], bid);
                }
            }
            sycl::group_barrier(cta);

            auto should_swap = [&](int lhs, int rhs) {
                const auto left = local_values[lhs];
                const auto right = local_values[rhs];
                if (order == SortOrder::Descending) {
                    if (left < right) return true;
                    if (right < left) return false;
                } else {
                    if (right < left) return true;
                    if (left < right) return false;
                }
                return local_indices[rhs] < local_indices[lhs];
            };

            for (int pass = 0; pass < active_n; ++pass) {
                const int start = pass & 1;
                for (int i = start + 2 * tid; i + 1 < active_n; i += 2 * bdim) {
                    if (should_swap(i, i + 1)) {
                        auto tmp_index = local_indices[i];
                        local_indices[i] = local_indices[i + 1];
                        local_indices[i + 1] = tmp_index;

                        auto tmp_value = local_values[i];
                        local_values[i] = local_values[i + 1];
                        local_values[i + 1] = tmp_value;
                    }
                }
                sycl::group_barrier(cta);
            }

            for (int i = tid; i < active_n; i += bdim) {
                batch_indices[i] = local_indices[i];
            }
            for (int i = active_n + tid; i < n; i += bdim) {
                batch_indices[i] = static_cast<K>(i);
            }
        });
    });
    return ctx.get_event();
}

struct PermutedCopyParams {
    std::pair<int32_t, int32_t> work_group_size_range = {-1, -1}; // Use default
};

template <typename T, typename K>
Event permuted_copy(Queue& ctx, const Matrix<T, MatrixFormat::Dense>& src, const Matrix<T, MatrixFormat::Dense>& dst, const Vector<K>& indices, const PermutedCopyParams& params = {}){
    return permuted_copy(ctx, src.view(), dst.view(), VectorView<K>(indices), params);
}

template <typename T, typename K>
struct PermutedCopyKernel {};

template <typename T, typename K>
Event permuted_copy(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& src, const MatrixView<T, MatrixFormat::Dense>& dst, const VectorView<K>& indices, const PermutedCopyParams& params = {}){
    auto n = src.rows();
    auto batch_size = src.batch_size();
    if(src.inc() != 1 || dst.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("permute: data and indices must have unit increment (inc=1)");
    }
    if(src.rows() != dst.rows() || src.cols() != dst.cols() || src.batch_size() != dst.batch_size() || src.batch_size() != indices.batch_size()){
        throw std::runtime_error("permute: src, dst and indices must have the same dimensions");
    }
    auto total_work_items = src.rows() * src.cols() * batch_size;
    bool use_default_work_group_size = (params.work_group_size_range.first == -1 && params.work_group_size_range.second == -1);
    auto [global_size, local_size] = use_default_work_group_size ?
        compute_nd_range_sizes(
            total_work_items,
            ctx.device(), KernelType::MEMORY_BOUND) :
        std::pair<size_t, size_t>(params.work_group_size_range.first, params.work_group_size_range.second);

    ctx -> submit([&](sycl::handler& h) {
        auto src_view = src.kernel_view();
        auto dst_view = dst.kernel_view();
        auto cols = src.cols();
        h.parallel_for<PermutedCopyKernel<T, K>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            auto gid = item.get_global_linear_id();
            for (int index = gid; index < total_work_items; index += item.get_global_range()[0]) {
                int batch_id = index / (cols * n);
                int rem = index % (cols * n);
                int col = rem / n;
                int row = rem % n;
                dst_view(row, col, batch_id) = src_view(row, indices(col, batch_id), batch_id);
            }
        });
    });
    return ctx.get_event();
}

template <typename T, typename K>
struct ActivePermutedCopyKernel {};

template <typename T, typename RowK, typename ColK>
struct ActivePermutedCopy2DKernel {};

template <typename T, typename K>
Event permuted_copy_active(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& src,
                           const MatrixView<T, MatrixFormat::Dense>& dst,
                           const VectorView<K>& indices,
                           Span<int32_t> active_lengths,
                           const PermutedCopyParams& params = {}) {
    auto n = src.rows();
    auto batch_size = src.batch_size();
    if (src.inc() != 1 || dst.inc() != 1 || indices.inc() != 1) {
        throw std::runtime_error("permuted_copy_active: src, dst and indices must have unit increment");
    }
    if (src.rows() != dst.rows() || src.cols() != dst.cols() || src.batch_size() != dst.batch_size() || src.batch_size() != indices.batch_size()) {
        throw std::runtime_error("permuted_copy_active: src, dst and indices must have the same dimensions");
    }
    if (static_cast<int64_t>(active_lengths.size()) < batch_size) {
        throw std::runtime_error("permuted_copy_active: active_lengths must cover every batch item");
    }

    auto total_work_items = src.rows() * src.cols() * batch_size;
    bool use_default_work_group_size = (params.work_group_size_range.first == -1 && params.work_group_size_range.second == -1);
    auto [global_size, local_size] = use_default_work_group_size ?
        compute_nd_range_sizes(
            total_work_items,
            ctx.device(), KernelType::MEMORY_BOUND) :
        std::pair<size_t, size_t>(params.work_group_size_range.first, params.work_group_size_range.second);

    ctx->submit([&](sycl::handler& h) {
        auto src_view = src.kernel_view();
        auto dst_view = dst.kernel_view();
        auto cols = src.cols();
        h.parallel_for<ActivePermutedCopyKernel<T, K>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            auto gid = item.get_global_linear_id();
            for (int index = gid; index < total_work_items; index += item.get_global_range()[0]) {
                int batch_id = index / (cols * n);
                int rem = index % (cols * n);
                int col = rem / n;
                int row = rem % n;
                const int32_t active_n = std::max<int32_t>(0, std::min<int32_t>(active_lengths[batch_id], n));
                if (col < active_n) {
                    dst_view(row, col, batch_id) = src_view(row, indices(col, batch_id), batch_id);
                }
            }
        });
    });
    return ctx.get_event();
}

template <typename T, typename RowK, typename ColK>
Event permuted_copy_active_2d(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& src,
                              const MatrixView<T, MatrixFormat::Dense>& dst,
                              const VectorView<RowK>& row_indices,
                              const VectorView<ColK>& col_indices,
                              Span<int32_t> active_lengths,
                              const PermutedCopyParams& params = {}) {
    auto rows = src.rows();
    auto cols = src.cols();
    auto batch_size = src.batch_size();
    if (src.inc() != 1 || dst.inc() != 1 || row_indices.inc() != 1 || col_indices.inc() != 1) {
        throw std::runtime_error("permuted_copy_active_2d: src, dst, and index vectors must have unit increment");
    }
    if (src.rows() != dst.rows() || src.cols() != dst.cols() || src.batch_size() != dst.batch_size() ||
        src.batch_size() != row_indices.batch_size() || src.batch_size() != col_indices.batch_size()) {
        throw std::runtime_error("permuted_copy_active_2d: src, dst, row_indices and col_indices must have matching dimensions");
    }
    if (row_indices.size() < rows || col_indices.size() < cols) {
        throw std::runtime_error("permuted_copy_active_2d: index vectors must cover the active matrix dimensions");
    }
    if (static_cast<int64_t>(active_lengths.size()) < batch_size) {
        throw std::runtime_error("permuted_copy_active_2d: active_lengths must cover every batch item");
    }

    auto total_work_items = rows * cols * batch_size;
    bool use_default_work_group_size = (params.work_group_size_range.first == -1 && params.work_group_size_range.second == -1);
    auto [global_size, local_size] = use_default_work_group_size ?
        compute_nd_range_sizes(
            total_work_items,
            ctx.device(), KernelType::MEMORY_BOUND) :
        std::pair<size_t, size_t>(params.work_group_size_range.first, params.work_group_size_range.second);

    ctx->submit([&](sycl::handler& h) {
        auto src_view = src.kernel_view();
        auto dst_view = dst.kernel_view();
        h.parallel_for<ActivePermutedCopy2DKernel<T, RowK, ColK>>(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            auto gid = item.get_global_linear_id();
            for (int index = gid; index < total_work_items; index += item.get_global_range()[0]) {
                int batch_id = index / (cols * rows);
                int rem = index % (cols * rows);
                int col = rem / rows;
                int row = rem % rows;
                const int32_t active_n = std::max<int32_t>(0, std::min<int32_t>(active_lengths[batch_id], rows));
                if (row < active_n && col < active_n) {
                    dst_view(row, col, batch_id) = src_view(row_indices(row, batch_id), col_indices(col, batch_id), batch_id);
                }
            }
        });
    });
    return ctx.get_event();
}

//Out of place permutation
template <typename T, typename K>
Event permute(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& data, const MatrixView<T, MatrixFormat::Dense>& temp_storage, const VectorView<K>& indices, const PermutedCopyParams& params = {}){
    MatrixView<T>::copy(ctx, temp_storage, data);
    return permuted_copy(ctx, temp_storage, data, indices, params);
}

template <typename T, typename K>
Event permute_active(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& data,
                     const MatrixView<T, MatrixFormat::Dense>& temp_storage,
                     const VectorView<K>& indices,
                     Span<int32_t> active_lengths,
                     const PermutedCopyParams& params = {}) {
    MatrixView<T>::copy(ctx, temp_storage, data);
    return permuted_copy_active(ctx, temp_storage, data, indices, active_lengths, params);
}

template <typename T, typename K>
Event permute(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& data, const VectorView<K>& indices){
    auto n = data.rows();
    auto batch_size = data.batch_size();
    if(data.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("permute: data and indices must have unit increment (inc=1)");
    }

    ctx -> submit([&](sycl::handler& h) {
        auto temp_indices = sycl::local_accessor<K, 1>(sycl::range<1>(n), h);
        auto temp_vec = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        auto data_view = data.kernel_view();
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto cta = item.get_group();
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range()[0];
            auto tid = item.get_local_linear_id();
            K* batch_indices = indices.batch_item(bid).data_ptr();
            // Use cycle decomposition to perform in-place permutation of each column
            for (int i = tid; i < n; i += bdim)
                temp_indices[i] = batch_indices[i];
            sycl::group_barrier(cta);
            
            for (int i = 0; i < n; ++i) {
                if (temp_indices[i] < 0) continue;               // already processed
                int curr_ix = i;
                int next_ix = -1;

                // Preserve the first column of this cycle
                sycl::group_barrier(cta);
                for (int j = tid; j < n; j += bdim)
                    temp_vec[j] = data_view(j, curr_ix, bid);
                sycl::group_barrier(cta);

                while (true) {
                    next_ix = temp_indices[curr_ix];

                    // If the next index closes the cycle, write the saved column and finish
                    if (next_ix == i) {
                        for (int j = tid; j < n; j += bdim)
                            data_view(j, curr_ix, bid) = temp_vec[j];
                        temp_indices[curr_ix] = -1;
                        break;
                    }

                    // Move the column that belongs to the current position
                    for (int j = tid; j < n; j += bdim)
                        data_view(j, curr_ix, bid) = data_view(j, next_ix, bid);

                    sycl::group_barrier(cta);
                    temp_indices[curr_ix] = -1;                   // mark this slot done
                    curr_ix = next_ix;
                }

                sycl::group_barrier(cta); // ensure all threads sync before the next outer iteration
            }
        });
    });
    return ctx.get_event();
}

template <typename T, typename K>
class ActivePermuteVectorKernel;

template <typename T, typename K>
Event permute_active(Queue& ctx,
                     VectorView<T> data,
                     VectorView<K> indices,
                     Span<int32_t> active_lengths) {
    const auto n = data.size();
    const auto batch_size = data.batch_size();
    if (data.inc() != 1 || indices.inc() != 1) {
        throw std::runtime_error("permute_active: data and indices must have unit increment (inc=1)");
    }
    if (static_cast<int64_t>(active_lengths.size()) < batch_size) {
        throw std::runtime_error("permute_active: active_lengths must cover every batch item");
    }

    ctx->submit([&](sycl::handler& h) {
        auto temp_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        h.parallel_for<ActivePermuteVectorKernel<T, K>>(sycl::nd_range<1>(batch_size * 128, 128), [=](sycl::nd_item<1> item) {
            const auto bid = item.get_group_linear_id();
            const auto bdim = item.get_local_range()[0];
            const auto tid = item.get_local_linear_id();
            const int32_t active_n = std::max<int32_t>(0, std::min<int32_t>(active_lengths[bid], n));
            for (int i = tid; i < active_n; i += bdim) {
                temp_mem[i] = data(indices(i, bid), bid);
            }
            sycl::group_barrier(item.get_group());
            for (int i = tid; i < active_n; i += bdim) {
                data(i, bid) = temp_mem[i];
            }
        });
    });
    return ctx.get_event();
}

template <typename T, typename K>
Event argsort(Queue& ctx, VectorView<T> data, VectorView<K> indices, SortOrder order, bool fill_indices){
    auto n = data.size();
    auto batch_size = data.batch_size();
    if(data.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("argsort: data and indices must have unit increment (inc=1)");
    }

    ctx -> submit([&](sycl::handler& h) {
        auto local_indices = sycl::local_accessor<K, 1>(sycl::range<1>(n), h);
        auto local_values = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto tid = item.get_local_linear_id();
            auto bdim = item.get_local_range()[0];
            auto cta = item.get_group();
            K* batch_indices = indices.batch_item(bid).data_ptr();

            if (fill_indices) {
                for (int i = tid; i < n; i += bdim) {
                    local_indices[i] = static_cast<K>(i);
                    local_values[i] = data(i, bid);
                }
            } else {
                for (int i = tid; i < n; i += bdim) {
                    local_indices[i] = batch_indices[i];
                    local_values[i] = data(local_indices[i], bid);
                }
            }
            sycl::group_barrier(cta);

            auto should_swap = [&](int lhs, int rhs) {
                const auto left = local_values[lhs];
                const auto right = local_values[rhs];
                if (order == SortOrder::Descending) {
                    if (left < right) return true;
                    if (right < left) return false;
                } else {
                    if (right < left) return true;
                    if (left < right) return false;
                }
                return local_indices[rhs] < local_indices[lhs];
            };

            for (int pass = 0; pass < n; ++pass) {
                const int start = pass & 1;
                for (int i = start + 2 * tid; i + 1 < n; i += 2 * bdim) {
                    if (should_swap(i, i + 1)) {
                        auto tmp_index = local_indices[i];
                        local_indices[i] = local_indices[i + 1];
                        local_indices[i + 1] = tmp_index;

                        auto tmp_value = local_values[i];
                        local_values[i] = local_values[i + 1];
                        local_values[i + 1] = tmp_value;
                    }
                }
                sycl::group_barrier(cta);
            }

            for (int i = tid; i < n; i += bdim) {
                batch_indices[i] = local_indices[i];
            }
        });
    });
    return ctx.get_event();
}

template <typename T>
Event sort(Queue& ctx, const VectorView<T>& eigs, const MatrixView<T, MatrixFormat::Dense>& eigvects, JobType jobz, SortOrder order, Span<std::byte> workspace){
    auto pool = BumpAllocator(workspace);
    // Ensure contiguous unit-inc views for argsort
    VectorView<T> eigs_unit = eigs;
    if (eigs.inc() != 1) {
        auto tmp = Vector<T>(eigs.size(), eigs.batch_size(), eigs.size(), 1);
        VectorView<T>::copy(ctx, tmp, eigs);
        eigs_unit = VectorView<T>(tmp);
    }
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, eigs.size() * eigs.batch_size()).data(), eigs.size(), eigs.batch_size(), 1, eigs.size());
    argsort(ctx, eigs_unit, permutation, order, true);
    permute(ctx, eigs, permutation);
    if (jobz == JobType::EigenVectors){
        auto temp_eigvects = MatrixView<T, MatrixFormat::Dense>(pool.allocate<T>(ctx, eigvects.rows() * eigvects.cols() * eigvects.batch_size()).data(), eigvects.rows(), eigvects.cols(), eigvects.rows(), eigvects.rows() * eigvects.cols(), eigvects.batch_size());
        permute(ctx, eigvects, temp_eigvects, permutation);
    }
    return ctx.get_event();
}

template <typename T>
Event sort(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz, SortOrder order, Span<std::byte> workspace) {
    const auto batch_size = V.batch_size();
    const auto n_total = W.size();
    if (batch_size <= 0 || (n_total % batch_size) != 0) {
        throw std::runtime_error("sort: invalid batch layout for eigenvalues span");
    }
    const auto n = n_total / batch_size;
    return sort(ctx, VectorView<T>(W.data(), static_cast<int>(n), batch_size, 1, static_cast<int>(n)), V, jobz, order, workspace);
}

template <typename T>
Event sort(Queue& ctx, const VectorView<T>& data, SortOrder order){
    auto ws = UnifiedVector<std::byte>(sort_buffer_size<T>(ctx, data.data(), MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0), JobType::NoEigenVectors));
    sort<T>(ctx, data, MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0), JobType::NoEigenVectors, order, ws).wait();
    return ctx.get_event();
}

template <typename T>
Event sort(Queue& ctx, const VectorView<T>& data, const MatrixView<T, MatrixFormat::Dense>& eigvects, SortOrder order){
    auto ws = UnifiedVector<std::byte>(sort_buffer_size<T>(ctx, data.data(), eigvects, JobType::EigenVectors));
    sort<T>(ctx, data, eigvects, JobType::EigenVectors, order, ws).wait();
    return ctx.get_event();
}


template <typename T>
size_t sort_buffer_size(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz) {
    // Workspace used by `sort(...)` in this header:
    // - permutation indices: int32_t[n * batch]
    // - optional temp eigenvectors matrix: T[rows * cols * batch]
    //
    // Workspace used by sort is sized by (n, batch) and does not depend on any padding
    // that may exist in the underlying eigenvalue storage.
    //
    // Many callers provide `W` via `VectorView::data()`, whose Span length is the
    // *required span length* for a possibly-strided batched layout. In that case,
    // `W.size()` may be larger than `n * batch` and not divisible by `batch`.
    //
    // Prefer deriving (n, batch) from the eigenvector view when available.
    int64_t batch = V.batch_size();
    int64_t n_total = static_cast<int64_t>(W.size());
    if (batch > 0 && V.rows() > 0) {
        n_total = static_cast<int64_t>(V.rows()) * static_cast<int64_t>(batch);
    } else {
        // Fallback: assume W is a packed flat span for a single batch.
        batch = 1;
    }
    if (batch <= 0 || n_total < 0) {
        throw std::runtime_error("sort_buffer_size: invalid batch layout for eigenvalues span");
    }

    size_t size = 0;
    size += BumpAllocator::allocation_size<int32_t>(ctx, static_cast<size_t>(n_total));
    if (jobz == JobType::EigenVectors) {
        size += BumpAllocator::allocation_size<T>(ctx,
                                                  static_cast<size_t>(V.rows()) * static_cast<size_t>(V.cols()) * static_cast<size_t>(batch));
    }
    return size;
}

} // namespace batchlas
