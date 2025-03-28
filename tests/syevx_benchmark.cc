#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using namespace batchlas;

// Helper function to generate sparse matrices with controlled sparsity
template <typename T>
SparseMatHandle<T, Format::CSR, BatchType::Batched> generate_sparse_matrices(Queue& ctx,
    size_t n, size_t batch_size, double sparsity, Span<std::byte> external_memory) {
    auto pool = BumpAllocator(external_memory);
    
    // Initialize random number generator
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    
    // Calculate expected non-zeros per row based on sparsity
    size_t nnz_per_row = static_cast<size_t>(n * sparsity);
    size_t total_nnz_per_matrix = n * nnz_per_row;
    
    auto values = pool.allocate<T>(ctx, total_nnz_per_matrix * batch_size);
    auto col_indices = pool.allocate<int>(ctx, total_nnz_per_matrix * batch_size);
    auto row_offsets = pool.allocate<int>(ctx, (n + 1) * batch_size);
    
    size_t row_offsets_idx = 0;
    
    for (size_t b = 0; b < batch_size; ++b) {
        size_t values_idx = 0;
        // Start row pointers for this matrix
        size_t row_start = b * (n + 1);
        row_offsets[row_offsets_idx++] = 0; // First row offset
        
        for (size_t i = 0; i < n; ++i) {
            // Generate random column indices for this row
            std::vector<int> cols;
            cols.reserve(nnz_per_row);
            
            // Always include diagonal for SPD property
            cols.push_back(i);
            
            // Add random off-diagonal elements
            for (size_t j = 1; j < nnz_per_row; ++j) {
                int col;
                do {
                    col = rng() % n;
                } while (col == i || std::find(cols.begin(), cols.end(), col) != cols.end());
                cols.push_back(col);
            }
            
            // Sort columns for CSR format
            std::sort(cols.begin(), cols.end());
            
            // Add to CSR data
            for (int col : cols) {
                // Make matrix symmetric and positive definite
                T val = (col == i) ? static_cast<T>(n + dist(rng)) : static_cast<T>(dist(rng) * 0.1);
                values[total_nnz_per_matrix*b + values_idx] = val;
                col_indices[total_nnz_per_matrix*b + values_idx] = col;
                values_idx++;
            }
            
            // Set row pointer
            row_offsets[row_offsets_idx++] = values_idx;
        }
    }
    // Create the sparse matrix handle
    return SparseMatHandle<T, Format::CSR, BatchType::Batched>(
        values.data(), row_offsets.data(), col_indices.data(),
        total_nnz_per_matrix, n, n, total_nnz_per_matrix, (n+1), batch_size);
}

// Benchmark function
template <typename T>
void benchmark_syevx(Queue& ctx, size_t n, size_t batch_size, size_t neigs, double sparsity) {
    std::cout << "Benchmarking syevx: n=" << n 
              << ", batch_size=" << batch_size 
              << ", neigs=" << neigs 
              << ", sparsity=" << sparsity << std::endl;
    
    // Generate sparse matrices
    auto external_memory = UnifiedVector<std::byte>(static_cast<size_t>(n * sparsity)*n*batch_size * (sizeof(T) + sizeof(int)) + sizeof(int) * (n + 1)*batch_size + sizeof(T) * n * batch_size);
    auto sparse_matrix = generate_sparse_matrices<T>(ctx, n, batch_size, sparsity, external_memory);

    // Allocate memory for eigenvalues
    UnifiedVector<typename base_type<T>::type> W(batch_size * neigs);
    
    // Configure parameters
    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 50;
    params.extra_directions = 0;
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;
    
    // Get buffer size and allocate workspace
    size_t buffer_size = syevx_buffer_size<Backend::CUDA>(
        ctx, sparse_matrix, Span<typename base_type<T>::type>(W.data(), W.size()),
        neigs, JobType::NoEigenVectors, DenseMatView<T, BatchType::Batched>(), params);
    
    UnifiedVector<std::byte> workspace(buffer_size);
    
    // Warmup run
    syevx<Backend::CUDA>(
        ctx, sparse_matrix, Span<typename base_type<T>::type>(W.data(), W.size()),
        neigs, Span<std::byte>(workspace.data(), workspace.size()),
        JobType::NoEigenVectors, DenseMatView<T, BatchType::Batched>(), params);
    ctx.wait();
    
    // Benchmark run
    auto start = std::chrono::high_resolution_clock::now();
    
    syevx<Backend::CUDA>(
        ctx, sparse_matrix, Span<typename base_type<T>::type>(W.data(), W.size()),
        neigs, Span<std::byte>(workspace.data(), workspace.size()),
        JobType::NoEigenVectors, DenseMatView<T, BatchType::Batched>(), params);
    
    ctx.wait();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Report results
    std::cout << "  Time: " << elapsed.count() << " seconds" << std::endl;
    
    // Print first few eigenvalues from first matrix
    std::cout << "  First few eigenvalues: ";
    for (size_t i = 0; i < std::min(3UL, neigs); ++i) {
        std::cout << W[i] << " ";
    }
    std::cout << std::endl;
}

// Main benchmark routine
void run_benchmarks() {
    std::cout << "\n=== SYEVX Benchmarks ===" << std::endl;
    Queue ctx(Device::default_device());
    
    // Different matrix sizes
    std::vector<size_t> sizes = {128, 256, 512, 1024};
    
    // Different batch sizes
    std::vector<size_t> batches = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    // Different sparsity levels
    std::vector<double> sparsities = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5};
    
    // Different numbers of eigenvalues
    std::vector<size_t> neigs_values = {5, 10, 15, 20};
    
    // Run benchmarks with varying parameters
    for (auto n : sizes) {
        for (auto batch : batches) {
            // Skip larger combinations which would be too slow or memory-intensive
            if (n >= 512 && batch >= 50) continue;
            
            for (auto sparsity : sparsities) {
                for (auto neigs : neigs_values) {
                    // Skip if asking for too many eigenvalues
                    if (neigs > n/4) continue;
                    
                    // Run benchmark
                    benchmark_syevx<float>(ctx, n, batch, neigs, sparsity);
                    
                    // Add separator
                    std::cout << "------------------------------" << std::endl;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    try {
        run_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
