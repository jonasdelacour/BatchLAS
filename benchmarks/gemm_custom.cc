#include <blas/linalg.hh>
using namespace batchlas;

int main (int argc, char **argv) {
    auto m = argc > 1 ? std::atoi(argv[1]) : 100;
    auto n = argc > 2 ? std::atoi(argv[2]) : 100;
    auto k = argc > 3 ? std::atoi(argv[3]) : 100;

    auto A = Matrix<float>::Random(m, k);
    auto B = Matrix<float>::Random(k, n);
    auto C = Matrix<float>::Zeros(m, n);

    Queue queue("gpu");

    std::vector<float> times(10);
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        gemm<Backend::CUDA>(queue, A.view(), B.view(), C.view(), 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
        queue.wait();
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<float, std::milli>(end - start).count();
        }
        float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        std::cout << "Average GEMM time: " << avg_time << " ms" << std::endl;
        
        // Calculate GFLOPS: (2*m*n*k operations) / (time in seconds * 10^9)
        float ops = 2.0f * m * n * k;
        float seconds = avg_time / 1000.0f; // Convert ms to seconds
        float gflops = (ops / seconds) / 1e9;
        std::cout << "GFLOPS: " << gflops << std::endl;
}