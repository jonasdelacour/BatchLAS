#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

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
    std::vector<int> cols;
    cols.reserve(nnz_per_row);

    for (size_t b = 0; b < 1; ++b) {
        size_t values_idx = 0;
        // Start row pointers for this matrix
        size_t row_start = b * (n + 1);
        row_offsets[row_offsets_idx++] = 0; // First row offset
        
        for (size_t i = 0; i < n; ++i) {
            // Generate random column indices for this row
            cols.clear();
            
            
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
    //For speed let's just reuse the same matrix for all batches
    for (size_t b = 1; b < batch_size; ++b) {
        std::memcpy(row_offsets.data() + b * (n + 1), row_offsets.data(), (n + 1) * sizeof(int));
        std::memcpy(values.data() + b * total_nnz_per_matrix, values.data(), total_nnz_per_matrix * sizeof(T));
        std::memcpy(col_indices.data() + b * total_nnz_per_matrix, col_indices.data(), total_nnz_per_matrix * sizeof(int));
    }


    // Create the sparse matrix handle
    return SparseMatHandle<T, Format::CSR, BatchType::Batched>(
        values.data(), row_offsets.data(), col_indices.data(),
        total_nnz_per_matrix, n, n, total_nnz_per_matrix, (n+1), batch_size);
}

// Benchmark function
template <typename T>
double benchmark_syevx(Queue& ctx, size_t n, size_t batch_size, size_t neigs, double sparsity, std::ofstream& csv_file) {
    std::cout << "Benchmarking syevx: n=" << n 
              << ", batch_size=" << batch_size 
              << ", neigs=" << neigs 
              << ", sparsity=" << sparsity << std::endl;
    
    // Generate sparse matrices
    auto T0 = std::chrono::high_resolution_clock::now();
    auto external_memory = UnifiedVector<std::byte>(static_cast<size_t>(n * sparsity)*n*batch_size * (sizeof(T) + sizeof(int)) + sizeof(int) * (n + 1)*batch_size + sizeof(T) * n * batch_size);
    auto elapsed_mem_secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - T0).count();
    T0 = std::chrono::high_resolution_clock::now();
    auto sparse_matrix = generate_sparse_matrices<T>(ctx, n, batch_size, sparsity, external_memory);
    auto elapsed_gen_secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - T0).count();


    // Allocate memory for eigenvalues
    UnifiedVector<typename base_type<T>::type> W(batch_size * neigs);
    
    // Configure parameters
    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 10;
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
    T0 = std::chrono::high_resolution_clock::now();
    syevx<Backend::CUDA>(
        ctx, sparse_matrix, Span<typename base_type<T>::type>(W.data(), W.size()),
        neigs, Span<std::byte>(workspace.data(), workspace.size()),
        JobType::NoEigenVectors, DenseMatView<T, BatchType::Batched>(), params);
    ctx.wait();
    auto elapsed_warmup_secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - T0).count();
    std::cout << "  Warmup time: " << elapsed_warmup_secs << " seconds" << std::endl;
    
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
    double time_seconds = elapsed.count();
    std::cout << "  Time: " << time_seconds << " seconds" << std::endl;
    std::cout << "  Matrix generation time: " << elapsed_gen_secs << " seconds" << std::endl;
    std::cout << "  Memory allocation time: " << elapsed_mem_secs << " seconds" << std::endl;
    std::cout << "  Total time: " << (elapsed_gen_secs + elapsed_mem_secs + time_seconds) << " seconds" << std::endl;
    
    // Print first few eigenvalues from first matrix
    std::cout << "  First few eigenvalues: ";
    for (size_t i = 0; i < std::min(3UL, neigs); ++i) {
        std::cout << W[i] << " ";
    }
    std::cout << std::endl;
    
    // Write to CSV if file is open
    if (csv_file.is_open()) {
        csv_file << n << "," << batch_size << "," << neigs << "," 
                 << sparsity << "," << time_seconds;
                 
        // Add first few eigenvalues to CSV
        for (size_t i = 0; i < std::min(3UL, neigs); ++i) {
            csv_file << "," << W[i];
        }
        csv_file << std::endl;
    }
    
    return time_seconds;
}

// Parse command line arguments
void parse_args(int argc, char **argv, 
                std::vector<size_t>& sizes,
                std::vector<size_t>& batches,
                std::vector<double>& sparsities,
                std::vector<size_t>& neigs_values,
                std::string& csv_path) {
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--sizes" || arg == "-s") {
            if (++i < argc) {
                sizes.clear();
                std::string sizes_str = argv[i];
                std::stringstream ss(sizes_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    sizes.push_back(std::stoul(item));
                }
            }
        }
        else if (arg == "--batches" || arg == "-b") {
            if (++i < argc) {
                batches.clear();
                std::string batches_str = argv[i];
                std::stringstream ss(batches_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    batches.push_back(std::stoul(item));
                }
            }
        }
        else if (arg == "--sparsity" || arg == "-p") {
            if (++i < argc) {
                sparsities.clear();
                std::string sparsity_str = argv[i];
                std::stringstream ss(sparsity_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    sparsities.push_back(std::stod(item));
                }
            }
        }
        else if (arg == "--neigs" || arg == "-n") {
            if (++i < argc) {
                neigs_values.clear();
                std::string neigs_str = argv[i];
                std::stringstream ss(neigs_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    neigs_values.push_back(std::stoul(item));
                }
            }
        }
        else if (arg == "--output" || arg == "-o") {
            if (++i < argc) {
                csv_path = argv[i];
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --sizes, -s SIZE1,SIZE2,...    Matrix sizes to benchmark (e.g., 128,256,512)" << std::endl;
            std::cout << "  --batches, -b B1,B2,...        Batch sizes to benchmark (e.g., 1,2,4,8,16)" << std::endl;
            std::cout << "  --sparsity, -p S1,S2,...       Sparsity levels to benchmark (e.g., 0.01,0.05,0.1)" << std::endl;
            std::cout << "  --neigs, -n N1,N2,...          Numbers of eigenvalues to compute (e.g., 5,10,15)" << std::endl;
            std::cout << "  --output, -o FILE              CSV output file path" << std::endl;
            std::cout << "  --help, -h                     Show this help message" << std::endl;
            exit(0);
        }
    }
}

// Main benchmark routine
void run_benchmarks(int argc, char **argv) {
    std::cout << "\n=== SYEVX Benchmarks ===" << std::endl;
    Queue ctx(Device::default_device());
    
    // Default parameters
    std::vector<size_t> sizes = {128, 256, 512};
    std::vector<size_t> batches = {1, 2, 4, 8, 16};
    std::vector<double> sparsities = {0.01, 0.05, 0.1};
    std::vector<size_t> neigs_values = {5, 10, 15};
    std::string csv_path;
    
    // Parse command line arguments
    parse_args(argc, argv, sizes, batches, sparsities, neigs_values, csv_path);
    
    // Open CSV file if specified
    std::ofstream csv_file;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open CSV file: " << csv_path << std::endl;
        } else {
            // Write header
            csv_file << "matrix_size,batch_size,neigs,sparsity,time_seconds,eigenvalue1,eigenvalue2,eigenvalue3" << std::endl;
        }
    }
    
    // Run benchmarks with varying parameters
    for (auto n : sizes) {
        for (auto batch : batches) {
            for (auto sparsity : sparsities) {
                for (auto neigs : neigs_values) {
                    if (neigs > n / 4) continue; // Skip invalid configurations
                    benchmark_syevx<float>(ctx, n, batch, neigs, sparsity, csv_file);
                    std::cout << "------------------------------" << std::endl;
                }
            }
        }
    }
    
    // Close CSV file
    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "Results written to: " << csv_path << std::endl;
    }
}

int main(int argc, char **argv) {
    try {
        run_benchmarks(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
