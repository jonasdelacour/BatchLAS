// Time the SHIPPED potrf public API. No harness-local re-implementation, and no
// forced route: whatever this build resolves is what gets timed. Built twice --
// once against build/ (vendor present) and once against build-novendor/ -- so
// "vendor-free" means the vendor-free BUILD, not a forced route inside a build
// that still has cuSOLVER linked into it.
//
// WHY THIS EXISTS. experiments/wp4_potrf/phase2_ab/phase2.cpp's `blocked` mode
// times a `Blocked<T>` class defined inside the harness, not
// src/extensions/potrf_blocked.cc, and it calls backend::trsm_vendor directly
// so it cannot be linked against a vendor-free build at all. It was the right
// instrument for the Phase 2 DESIGN study, which ran before the driver existed;
// it cannot answer "is the shipped vendor-free potrf faster than cuSOLVER".
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace batchlas;

template <typename T>
static int run(const char* tn, int n, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const size_t st = size_t(n) * n;
    UnifiedVector<T> A(st * batch), A0(st * batch);
    UnifiedVector<int32_t> info(size_t(batch), 0);

    // Diagonally dominant SPD, condition number close to 1, so any non-zero
    // info or large residual is the implementation and not the input.
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const double v = (r == c) ? double(n) + 1.0
                                          : 0.5 / (1.0 + std::abs(r - c));
                A0[size_t(b) * st + size_t(c) * n + r] = T(v);
            }
    std::memcpy(A.data(), A0.data(), A.size() * sizeof(T));

    // A VIEW OVER THE CALLER'S BUFFER, not a Matrix. Matrix's (const T*, ...)
    // constructor COPIES into its own storage, so factorising a Matrix built
    // that way leaves the array this program checks untouched -- which reads as
    // "info == 0 and a residual of 1e+03", i.e. a wrong-answer report for a
    // correct kernel.
    // Each view gets its OWN pointer array: the vendor batched path calls
    // data_ptrs(ctx) and throws "data_ptrs target is null" on a view built
    // without one, and sharing one array between two views is the
    // matrix.hh:1140 trap.
    const size_t nb_ptrs = size_t(batch);
    UnifiedVector<T*> pa(nb_ptrs);
    UnifiedVector<T*> pa0(nb_ptrs);
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(st), batch, pa.data());
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(st), batch, pa0.data());

    const size_t wsz = potrf_buffer_size<Backend::CUDA, T>(*q, Av, Uplo::Lower);
    UnifiedVector<std::byte> ws(wsz ? wsz : size_t(1));

    auto reset = [&] {
        MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
        q->wait();
    };

    // Warm the JIT and the clocks, and discard: a cold first run has fabricated
    // a 3.7x result in this repository before.
    const double warm_s = std::getenv("WARM_S") ? std::atof(std::getenv("WARM_S")) : 1.5;
    const auto w0 = std::chrono::steady_clock::now();
    do {
        reset();
        potrf<Backend::CUDA, T>(*q, Av, Uplo::Lower, ws.to_span(), info.to_span());
        q->wait();
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s);

    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        reset();
        const auto t0 = std::chrono::steady_clock::now();
        potrf<Backend::CUDA, T>(*q, Av, Uplo::Lower, ws.to_span(), info.to_span());
        q->wait();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    double mean = 0;
    for (double v : ms) mean += v;
    mean /= double(ms.size());
    double sd = 0;
    for (double v : ms) sd += (v - mean) * (v - mean);
    sd = std::sqrt(sd / double(ms.size()));

    // Correctness in the same process, so a fast WRONG answer cannot be
    // reported as a win -- which is exactly how five apparent wins entered the
    // Phase 2 record. Worst ||L L^T - A||inf / ||A||inf over items 0 and
    // batch-1; item 0 alone is blind to a wrong sub-view stride.
    double worst = 0;
    for (int b : {0, batch - 1}) {
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                double acc = 0;
                for (int k = 0; k <= j; ++k)
                    acc += double(A[size_t(b) * st + size_t(k) * n + i]) *
                           double(A[size_t(b) * st + size_t(k) * n + j]);
                const double a = double(A0[size_t(b) * st + size_t(j) * n + i]);
                num = std::max(num, std::abs(acc - a));
                den = std::max(den, std::abs(a));
            }
        worst = std::max(worst, num / den);
    }
    int bad = 0;
    for (int i = 0; i < batch; ++i)
        if (info[size_t(i)] != 0) ++bad;

    std::printf("%s,%d,%d,%.4f,%.4f,%.4f,%.3e,%d\n", tn, n, batch, med, mean,
                sd / mean, worst, bad);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: realpotrf <type> <n> <batch> <reps>\n"
                             "prints: type,n,batch,median_ms,mean_ms,rel_sd,residual,info_nonzero\n");
        return 2;
    }
    const std::string t = argv[1];
    const int n = std::atoi(argv[2]), b = std::atoi(argv[3]), r = std::atoi(argv[4]);
    if (t == "float") return run<float>("float", n, b, r);
    if (t == "double") return run<double>("double", n, b, r);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
