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

//Out of place permutation
template <typename T, typename K>
Event permute(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& data, const MatrixView<T, MatrixFormat::Dense>& temp_storage, const VectorView<K>& indices, const PermutedCopyParams& params = {}){
    MatrixView<T>::copy(ctx, temp_storage, data);
    return permuted_copy(ctx, temp_storage, data, indices, params);
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