#include <blas/linalg.hh>
#include <util/sycl-vector.hh>
#include <util/mempool.hh>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <execution>

using namespace batchlas;

// Function to generate a CSV header row
std::string generate_csv_header() {
    return "matrix_size,batch_size,algorithm,time_seconds,orthonormality_error,transpose";
}

// Function to measure orthonormality error (A^T * A - I)
template <typename T>
double measure_orthonormality_error(
    Queue& ctx, 
    const DenseMatView<T, BatchType::Single>& A, 
    Transpose transA) {
    
    // Get dimensions
    int m = A.rows_;
    int n = A.cols_;
    int k = (transA == Transpose::NoTrans) ? n : m;
    
    // Create identity matrix for comparison
    UnifiedVector<T> identity_data(k * k, 0.0);
    for (int i = 0; i < k; i++) {
        identity_data[i * k + i] = 1.0;
    }
    
    // Create workspace for A^T * A
    UnifiedVector<T> result_data(k * k, 0.0);
    
    // Create views
    DenseMatView<T, BatchType::Single> I(identity_data.data(), k, k, k);
    DenseMatView<T, BatchType::Single> ATA(result_data.data(), k, k, k);
    
    // Compute A^T * A or A * A^T depending on transpose flag
    Transpose inv_trans = (transA == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
    gemm<Backend::CUDA>(ctx, A, A, ATA, T(1.0), T(0.0), inv_trans, transA);
    ctx.wait();
    
    // Compute Frobenius norm of (A^T * A - I)
    double error = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = std::abs(result_data[i * k + j] - expected);
            error += diff * diff;
        }
    }
    
    return std::sqrt(error);
}

// Structure to hold benchmark results
struct BenchmarkResult {
    int matrix_size;
    int batch_size;
    OrthoAlgorithm algorithm;
    double time_seconds;
    double orthonormality_error;
    Transpose transpose;
    
    std::string to_csv() const {
        std::stringstream ss;
        ss << matrix_size << "," 
           << batch_size << ","
           << static_cast<int>(algorithm) << ","
           << time_seconds << ","
           << orthonormality_error << ","
           << static_cast<int>(transpose);
        return ss.str();
    }
};

// Function to run benchmark for a specific configuration
template <typename T, BatchType BT>
BenchmarkResult run_single_benchmark(
    Queue& ctx,
    int matrix_size,
    int num_vectors,
    int batch_size,
    OrthoAlgorithm algorithm,
    Transpose transpose,
    int warmup_iterations = 2,
    int timing_iterations = 5) {
    
    // Create randomized matrices
    int rows = (transpose == Transpose::NoTrans) ? matrix_size : num_vectors;
    int cols = (transpose == Transpose::NoTrans) ? num_vectors : matrix_size;
    int ld = rows;
    
    // Create matrices for each batch
    UnifiedVector<T> matrices_data(rows * cols * batch_size);
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    
    std::transform(matrices_data.begin(), matrices_data.end(), matrices_data.begin(),
        [&gen, &dist](T) { return dist(gen); });
    
    
    // Create batched view
    // Use if constexpr (C++17) for compile-time conditional initialization
    DenseMatHandle<T, BT> matrices_handle = [&]() {
        if constexpr (BT == BatchType::Single) {
            return DenseMatHandle<T, BT>(matrices_data.data(), rows, cols, ld);
        } else {
            return DenseMatHandle<T, BT>(matrices_data.data(), rows, cols, ld, ld * cols, batch_size);
        }
    }();
    
    // Allocate workspace
    UnifiedVector<std::byte> workspace(ortho_buffer_size<Backend::CUDA>(
        ctx, matrices_handle(), transpose, algorithm));
    
    // Warmup iterations
    for (int i = 0; i < warmup_iterations; ++i) {
        // Reset matrices to random data
        for (size_t i = 0; i < matrices_data.size(); ++i) {
            matrices_data[i] = dist(gen);
        }
        
        // Run orthogonalization
        ortho<Backend::CUDA>(ctx, matrices_handle(), transpose, workspace, algorithm);
        ctx.wait();
    }
    
    // Timing iterations
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < timing_iterations; ++i) {
        // Reset matrices to random data
        for (size_t i = 0; i < matrices_data.size(); ++i) {
            matrices_data[i] = dist(gen);
        }
        
        // Run orthogonalization
        ortho<Backend::CUDA>(ctx, matrices_handle(), transpose, workspace, algorithm);
        ctx.wait();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / timing_iterations;
    
    // Measure orthonormality error
    double error = 0.0;
    if constexpr (BT == BatchType::Single) {
        error = measure_orthonormality_error(ctx, matrices_handle(), transpose);
    } else {
        // For batched, we need to check each batch
        for (int i = 0; i < batch_size; ++i) {
            const auto & batch_view = matrices_handle();
            error += measure_orthonormality_error(ctx, batch_view[i], transpose);
        }
        error /= batch_size;
    }
        
    
    // Return benchmark result
    return BenchmarkResult{
        matrix_size,
        batch_size,
        algorithm,
        avg_time,
        error,
        transpose
    };
}

// Print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -s, --sizes SIZES      Comma-separated matrix sizes (default: 1024,2048,4096)" << std::endl;
    std::cout << "  -v, --vectors VECTORS  Comma-separated number of vectors (default: 16,32,64)" << std::endl;
    std::cout << "  -b, --batches BATCHES  Comma-separated batch sizes (default: 1,4,16,64)" << std::endl;
    std::cout << "  -a, --algos ALGOS      Comma-separated algorithm IDs (default: 0,1,2,4,5)" << std::endl;
    std::cout << "                         0=Cholesky, 1=Chol2, 2=ShiftChol3, 3=Householder, 4=CGS2, 5=SVQB" << std::endl;
    std::cout << "  -t, --transpose FLAG   Transpose flag (0=NoTrans, 1=Trans, default: 0)" << std::endl;
    std::cout << "  -o, --output FILE      Output CSV file (default: ortho_benchmark_results.csv)" << std::endl;
    std::cout << "  -w, --warmup N         Number of warmup iterations (default: 2)" << std::endl;
    std::cout << "  -i, --iterations N     Number of timing iterations (default: 5)" << std::endl;
    std::cout << "  -h, --help             Show this help message" << std::endl;
}

// Parse comma-separated integers
std::vector<int> parse_comma_separated_ints(const std::string& str) {
    std::vector<int> values;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stoi(token));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing value: " << token << std::endl;
        }
    }
    
    return values;
}

// Convert algorithm ID to OrthoAlgorithm enum
OrthoAlgorithm algorithm_from_id(int id) {
    switch (id) {
        case 0: return OrthoAlgorithm::Cholesky;
        case 1: return OrthoAlgorithm::Chol2;
        case 2: return OrthoAlgorithm::ShiftChol3;
        case 3: return OrthoAlgorithm::Householder;
        case 4: return OrthoAlgorithm::CGS2;
        case 5: return OrthoAlgorithm::SVQB;
        default: return OrthoAlgorithm::Chol2;
    }
}

// Get string representation of algorithm
std::string algorithm_to_string(OrthoAlgorithm algo) {
    switch (algo) {
        case OrthoAlgorithm::Cholesky: return "Cholesky";
        case OrthoAlgorithm::Chol2: return "Chol2";
        case OrthoAlgorithm::ShiftChol3: return "ShiftChol3";
        case OrthoAlgorithm::Householder: return "Householder";
        case OrthoAlgorithm::CGS2: return "CGS2";
        case OrthoAlgorithm::SVQB: return "SVQB";
        default: return "Unknown";
    }
}

// Get string representation of transpose flag
std::string transpose_to_string(Transpose trans) {
    return (trans == Transpose::NoTrans) ? "NoTrans" : "Trans";
}

int main(int argc, char** argv) {
    // Default parameters
    std::vector<int> matrix_sizes = {1024, 2048, 4096};
    std::vector<int> num_vectors = {16, 32, 64};
    std::vector<int> batch_sizes = {1, 4, 16, 64};
    std::vector<int> algorithm_ids = {0, 1, 2, 4};
    Transpose transpose = Transpose::NoTrans;
    std::string output_file = "ortho_benchmark_results.csv";
    int warmup_iterations = 2;
    int timing_iterations = 5;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-s" || arg == "--sizes") {
            if (i + 1 < argc) {
                matrix_sizes = parse_comma_separated_ints(argv[++i]);
            }
        } else if (arg == "-v" || arg == "--vectors") {
            if (i + 1 < argc) {
                num_vectors = parse_comma_separated_ints(argv[++i]);
            }
        } else if (arg == "-b" || arg == "--batches") {
            if (i + 1 < argc) {
                batch_sizes = parse_comma_separated_ints(argv[++i]);
            }
        } else if (arg == "-a" || arg == "--algos") {
            if (i + 1 < argc) {
                algorithm_ids = parse_comma_separated_ints(argv[++i]);
            }
        } else if (arg == "-t" || arg == "--transpose") {
            if (i + 1 < argc) {
                transpose = (std::stoi(argv[++i]) != 0) ? Transpose::Trans : Transpose::NoTrans;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) {
                warmup_iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) {
                timing_iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Create context
    auto ctx = std::make_shared<Queue>(Device::default_device());
    
    // Open output file
    std::ofstream csv_file(output_file);
    if (!csv_file) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }
    
    // Write CSV header
    csv_file << generate_csv_header() << std::endl;
    
    // Print benchmark configuration
    std::cout << "==========================================" << std::endl;
    std::cout << "Orthogonalization Benchmark" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Matrix sizes: ";
    for (auto size : matrix_sizes) std::cout << size << " ";
    std::cout << std::endl;
    
    std::cout << "Vector counts: ";
    for (auto count : num_vectors) std::cout << count << " ";
    std::cout << std::endl;
    
    std::cout << "Batch sizes: ";
    for (auto size : batch_sizes) std::cout << size << " ";
    std::cout << std::endl;
    
    std::cout << "Algorithms: ";
    for (auto id : algorithm_ids) std::cout << algorithm_to_string(algorithm_from_id(id)) << " ";
    std::cout << std::endl;
    
    std::cout << "Transpose mode: " << transpose_to_string(transpose) << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Timing iterations: " << timing_iterations << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Run benchmarks
    size_t total_benchmarks = matrix_sizes.size() * num_vectors.size() * 
                            batch_sizes.size() * algorithm_ids.size();
    size_t completed_benchmarks = 0;
    
    for (auto size : matrix_sizes) {
        for (auto vecs : num_vectors) {
            if (vecs > size) continue; // Skip invalid configurations
            
            for (auto batch : batch_sizes) {
                for (auto algo_id : algorithm_ids) {
                    OrthoAlgorithm algorithm = algorithm_from_id(algo_id);
                    
                    // Print progress
                    std::cout << "Running benchmark " << (completed_benchmarks + 1) 
                              << "/" << total_benchmarks << ": "
                              << "Size=" << size << ", Vectors=" << vecs
                              << ", Batch=" << batch 
                              << ", Algorithm=" << algorithm_to_string(algorithm) 
                              << std::endl;
                    
                    // Run the benchmark
                    BenchmarkResult result;
                    if (batch == 1) {
                        result = run_single_benchmark<float, BatchType::Single>(
                            *ctx, size, vecs, batch, algorithm, transpose,
                            warmup_iterations, timing_iterations);
                    } else {
                        result = run_single_benchmark<float, BatchType::Batched>(
                            *ctx, size, vecs, batch, algorithm, transpose,
                            warmup_iterations, timing_iterations);
                    }
                    
                    // Write result to CSV
                    csv_file << result.to_csv() << std::endl;
                    
                    // Print result summary
                    std::cout << "  Time: " << std::fixed << std::setprecision(6) 
                              << result.time_seconds << " seconds, Error: " 
                              << std::scientific << std::setprecision(3) 
                              << result.orthonormality_error << std::endl;
                    
                    completed_benchmarks++;
                }
            }
        }
    }
    
    csv_file.close();
    std::cout << "Benchmark completed. Results saved to " << output_file << std::endl;
    
    return 0;
}
