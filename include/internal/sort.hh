#pragma once
#include <sycl/sycl.hpp>
#include <util/mempool.hh>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include "../../src/queue.hh"

using namespace batchlas;

template <typename T>
Event sort(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz, SortOrder order, Span<std::byte> workspace) {
    auto pool = BumpAllocator(workspace);
    auto n = V.rows();
    auto batch_size = V.batch_size();
    ctx->submit([&](sycl::handler& h) {
        // Calculate memory required for joint_sort operation
        // We need more memory for larger arrays
        size_t sort_mem_size = sycl::ext::oneapi::experimental::default_sorters::joint_sorter<std::less<>>::memory_required<T>(
                sycl::memory_scope::work_group,  n);
        
        sycl::local_accessor<std::byte, 1> scratch(sycl::range<1>(sort_mem_size), h);
        sycl::local_accessor<T, 1> temp_eigenvalues(n, h);
        
        // Allocate global memory for indices array used during sorting
        auto indices_mem = pool.allocate<int>(ctx, n * batch_size);
        auto temp_vec_mem = pool.allocate<T>(ctx, jobz == JobType::EigenVectors ? n * batch_size : 0);
        
        auto Vstride = V.stride();
        auto Vdata = V.data_ptr();
        auto Vld = V.ld();

        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto tid = item.get_local_linear_id();
            auto bdim = item.get_local_range()[0];
            auto cta = item.get_group();
            
            // Set up pointers to the batch-specific data
            typename base_type<T>::type* batch_W = W.data() + bid * n;
            int* batch_indices = indices_mem.data() + bid * n;
            T* batch_temp_vec = temp_vec_mem.data() + bid * n;
            
            // Create group helper with scratch memory
            auto group_helper = sycl::ext::oneapi::experimental::group_with_scratchpad(
                cta, sycl::span<std::byte>(scratch.get_pointer(), scratch.size()));
            
            if (jobz == JobType::EigenVectors) {
                // Initialize indices - each thread handles a portion
                for (int i = tid; i < n; i += bdim) {
                    batch_indices[i] = i;
                }
                
                sycl::group_barrier(cta);
                
                // Use joint_sort with a custom comparator that also swaps indices
                // Only one thread needs to start the sort
                
                sycl::ext::oneapi::experimental::joint_sort(
                    group_helper, 
                    batch_indices, 
                    batch_indices + n,
                    [batch_W, batch_indices, order](auto idx_a, auto idx_b) {
                        // Sort by value in ascending order
                        if (order == SortOrder::Descending) {
                            return batch_W[idx_a] > batch_W[idx_b];
                        } else {
                            return batch_W[idx_a] < batch_W[idx_b];
                        }
                    }
                );

                for (int i = tid; i < n; i += bdim) {
                    temp_eigenvalues[i] = batch_W[batch_indices[i]];
                }
                sycl::group_barrier(cta);

                for (int i = tid; i < n; i += bdim) {
                    batch_W[i] = temp_eigenvalues[i];
                }
                
                
                
                // Reorder eigenvectors based on the sorted indices
                T* batch_V_out = const_cast<T*>(Vdata) + bid * Vstride;
                
                // In‑place permutation of the eigenvector columns using cycle‑decomposition.
                for (int i = 0; i < n; ++i) {
                    if (batch_indices[i] < 0) continue;               // already processed
                    int curr_ix = i;
                    int next_ix = -1;

                    // Preserve the first column of this cycle
                    sycl::group_barrier(cta);
                    for (int j = tid; j < n; j += bdim)
                        batch_temp_vec[j] = batch_V_out[curr_ix * Vld + j];
                    sycl::group_barrier(cta);

                    while (true) {
                        next_ix = batch_indices[curr_ix];

                        // If the next index closes the cycle, write the saved column and finish
                        if (next_ix == i) {
                            for (int j = tid; j < n; j += bdim)
                                batch_V_out[curr_ix * Vld + j] = batch_temp_vec[j];
                            batch_indices[curr_ix] = -1;
                            break;
                        }

                        // Move the column that belongs to the current position
                        for (int j = tid; j < n; j += bdim)
                            batch_V_out[curr_ix * Vld + j] = batch_V_out[next_ix * Vld + j];

                        sycl::group_barrier(cta);
                        batch_indices[curr_ix] = -1;                   // mark this slot done
                        curr_ix = next_ix;
                    }

                    sycl::group_barrier(cta); // ensure all threads sync before the next outer iteration
                }
            } else {
                // If only eigenvalues are needed, just sort them directly
                sycl::ext::oneapi::experimental::joint_sort(
                    group_helper, 
                    batch_W, 
                    batch_W + n,
                    [order](auto a, auto b) {
                        // Sort by value in ascending order
                        if (order == SortOrder::Descending) {
                            return a > b;
                        } else {
                            return a < b;
                        }
                    }
                );
                
            }
        });
    });
    return ctx.get_event();
}

template <typename T>
size_t sort_buffer_size(Queue& ctx, Span<T> W, const MatrixView<T, MatrixFormat::Dense>& V, JobType jobz) {
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, W.size() * V.batch_size()); // For eigenvalues
    size += BumpAllocator::allocation_size<T>(ctx, jobz == JobType::EigenVectors ? V.rows() * V.batch_size() : 0); // For eigenvectors if needed
    return size;
}