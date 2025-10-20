#pragma once
#include <sycl/sycl.hpp>
#include <util/mempool.hh>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include "../../src/queue.hh"

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
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
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
Event permuted_copy(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& src, const MatrixView<T, MatrixFormat::Dense>& dst, const VectorView<K>& indices){
    auto n = src.rows();
    auto batch_size = src.batch_size();
    if(src.inc() != 1 || dst.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("permute: data and indices must have unit increment (inc=1)");
    }
    if(src.rows() != dst.rows() || src.cols() != dst.cols() || src.batch_size() != dst.batch_size() || src.batch_size() != indices.batch_size()){
        throw std::runtime_error("permute: src, dst and indices must have the same dimensions");
    }

    ctx -> submit([&](sycl::handler& h) {
        auto src_view = src.kernel_view();
        auto dst_view = dst.kernel_view();
        auto cols = src.cols();
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range()[0];
            auto tid = item.get_local_linear_id();
            for (int col = 0; col < cols; col++) {
                auto src_ix = indices(col, bid);
                for (int i = tid; i < n; i += bdim) {
                    dst_view(i, col, bid) = src_view(i, src_ix, bid);
                }
            }
        });
    });
    return ctx.get_event();
}

//Out of place permutation
template <typename T, typename K>
Event permute(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& data, const MatrixView<T, MatrixFormat::Dense>& temp_storage, const VectorView<K>& indices){
    MatrixView<T>::copy(ctx, temp_storage, data);
    return permuted_copy(ctx, temp_storage, data, indices);
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
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
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
    // Implement the argsort functionality here
    auto n = data.size();
    auto batch_size = data.batch_size();
    // Since we use sycl::ext::oneapi::experimental::joint_sort, we need to ensure data and indices use unit increments.
    if(data.inc() != 1 || indices.inc() != 1){
        throw std::runtime_error("argsort: data and indices must have unit increment (inc=1)");
    }

    ctx -> submit([&](sycl::handler& h) {
        // Calculate memory required for joint_sort operation
        // We need more memory for larger arrays
        size_t sort_mem_size = sycl::ext::oneapi::experimental::default_sorters::joint_sorter<std::less<>>::memory_required<T>(
                sycl::memory_scope::work_group,  n);
        
        sycl::local_accessor<std::byte, 1> scratch(sycl::range<1>(sort_mem_size), h);
        
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto tid = item.get_local_linear_id();
            auto bdim = item.get_local_range()[0];
            auto cta = item.get_group();
            
            // Set up pointers to the batch-specific data
            T* batch_data = data.batch_item(bid).data_ptr();
            K* batch_indices = indices.batch_item(bid).data_ptr();

            // Create group helper with scratch memory
            auto group_helper = sycl::ext::oneapi::experimental::group_with_scratchpad(
                cta, sycl::span<std::byte>(scratch.get_pointer(), scratch.size()));
            
            // Initialize indices - each thread handles a portion
            if (fill_indices) {
                for (int i = tid; i < n; i += bdim) {
                    batch_indices[i] = i;
                }
            }
            
            sycl::group_barrier(cta);
            
            // Use joint_sort with a custom comparator that also swaps indices
            // Only one thread needs to start the sort
            
            sycl::ext::oneapi::experimental::joint_sort(
                group_helper, 
                batch_indices, 
                batch_indices + n,
                [batch_data, order](auto idx_a, auto idx_b) {
                    // Sort by value in ascending order
                    if (order == SortOrder::Descending) {
                        return batch_data[idx_a] > batch_data[idx_b];
                    } else {
                        return batch_data[idx_a] < batch_data[idx_b];
                    }
                }
            );
        });
    });
    return ctx.get_event();
}

template <typename T>
Event sort(Queue& ctx, const VectorView<T>& eigs, const MatrixView<T, MatrixFormat::Dense>& eigvects, JobType jobz, SortOrder order, Span<std::byte> workspace){
    auto pool = BumpAllocator(workspace);
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, eigs.size() * eigs.batch_size()).data(), eigs.size(), 1, eigs.size(), eigs.batch_size());
    argsort(ctx, eigs, permutation, order, true);
    permute(ctx, eigs, permutation);
    if (jobz == JobType::EigenVectors){
        auto temp_eigvects = MatrixView<T, MatrixFormat::Dense>(pool.allocate<T>(ctx, eigvects.rows() * eigvects.cols() * eigvects.batch_size()).data(), eigvects.rows(), eigvects.cols(), eigvects.rows(), eigvects.rows() * eigvects.cols(), eigvects.batch_size());
        permute(ctx, eigvects, temp_eigvects, permutation);
    }
    return ctx.get_event();
}

template <typename T>
Event sort(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz, SortOrder order, Span<std::byte> workspace) {
    return sort(ctx, VectorView<T>(W.data(), W.size(), 1, W.size(), V.batch_size()), V, jobz, order, workspace);
}

template <typename T>
size_t sort_buffer_size(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz) {
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, W.size() * V.batch_size()); // For eigenvalues
    size += BumpAllocator::allocation_size<T>(ctx, jobz == JobType::EigenVectors ? V.rows() * V.batch_size() : 0); // For eigenvectors if needed
    size += jobz == JobType::EigenVectors ? BumpAllocator::allocation_size<T>(ctx, V.rows() * V.cols() * V.batch_size()) : 0; // Temporary storage for eigenvectors
    return size;
}

} // namespace batchlas