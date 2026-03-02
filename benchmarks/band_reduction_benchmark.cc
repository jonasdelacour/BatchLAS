#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

std::vector<int32_t> g_block_size_seq;
std::vector<int32_t> g_d_seq;

enum class BandReductionMode {
    Phase,
    Sequence
};

BandReductionMode g_mode = BandReductionMode::Phase;
int32_t g_phase_reduce = 0;

inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

inline std::vector<int32_t> parse_int_list(const std::string& csv) {
    std::vector<int32_t> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(static_cast<int32_t>(std::stoi(item)));
    }
    return out;
}

inline BandReductionMode parse_mode_or_throw(const std::string& mode) {
    if (mode == "phase") return BandReductionMode::Phase;
    if (mode == "sequence") return BandReductionMode::Sequence;
    throw std::runtime_error("--mode must be one of: phase, sequence");
}

inline void resolve_schedule_or_throw(int32_t fallback_nb,
                                      int32_t fallback_d,
                                      std::vector<int32_t>& block_size_seq,
                                      std::vector<int32_t>& d_seq) {
    block_size_seq = g_block_size_seq;
    d_seq = g_d_seq;

    if (block_size_seq.empty() && d_seq.empty()) {
        block_size_seq = {std::max<int32_t>(1, fallback_nb)};
        d_seq = {std::max<int32_t>(0, fallback_d)};
        return;
    }

    if (block_size_seq.empty()) {
        block_size_seq.assign(d_seq.size(), std::max<int32_t>(1, fallback_nb));
    }
    if (d_seq.empty()) {
        d_seq.assign(block_size_seq.size(), std::max<int32_t>(0, fallback_d));
    }
    if (block_size_seq.size() != d_seq.size()) {
        throw std::runtime_error("--nb-seq and --d-seq must have the same length");
    }

    for (size_t i = 0; i < block_size_seq.size(); ++i) {
        if (block_size_seq[i] <= 0) {
            throw std::runtime_error("--nb-seq values must be > 0");
        }
        if (d_seq[i] < 0) {
            throw std::runtime_error("--d-seq values must be >= 0");
        }
    }
}

template <typename Benchmark>
inline void BandReductionBenchSizes(Benchmark* b) {
    // n, kd, batch, nb_target
    for (int n : {64, 128, 256, 512}) {
        for (int kd : {8, 16, 32}) {
            if (kd >= n) continue; // bandwidth must be less than matrix size
            for (int batch : {1, 8, 32, 64}) {
                for (int nb : {8, 16, 32}) {
                    if (nb > kd) continue; // block size shouldn't exceed bandwidth
                    b->Args({n, kd, batch, nb});
                }
            }
        }
    }
}

template <typename Benchmark>
inline void BandReductionBenchSizesNetlib(Benchmark* b) {
    // Smaller sizes for CPU backend
    for (int n : {64, 128, 256}) {
        for (int kd : {8, 16}) {
            if (kd >= n) continue;
            for (int batch : {1, 4, 8}) {
                for (int nb : {8, 16}) {
                    if (nb > kd) continue;
                    b->Args({n, kd, batch, nb});
                }
            }
        }
    }
}

} // namespace

// Unified benchmark:
// - mode=phase (default): run one sweep only
// - mode=sequence: run a sequence of sweep configurations
template <typename T, Backend B>
static void BM_BAND_REDUCTION(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t kd = state.range(1);
    const size_t batch = state.range(2);
    const int nb_target = static_cast<int>(state.range(3));
    const Uplo uplo = Uplo::Lower;

    if (g_mode == BandReductionMode::Phase && (!g_block_size_seq.empty() || !g_d_seq.empty())) {
        throw std::runtime_error("--nb-seq/--d-seq are only supported with --mode=sequence");
    }

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu", /*in_order=*/true);

    // Create input band matrix (kd+1 rows, n columns, in lower-band format)
    auto ab_in = Matrix<T>::Random(kd + 1, n, /*hermitian=*/false, batch, /*seed=*/2027);
    
    using Real = typename base_type<T>::type;
    auto d = Vector<Real>::zeros(n, batch);
    auto e = Vector<Real>::zeros(n - 1, batch);
    auto tau = Vector<T>::zeros(n - 1, batch);

    SytrdBandReductionParams params;
    if (g_mode == BandReductionMode::Phase) {
        params.block_size_seq = {std::max<int32_t>(1, static_cast<int32_t>(nb_target))};
        params.d_seq = {std::max<int32_t>(0, g_phase_reduce)};
        params.max_sweeps = 1;
    } else {
        resolve_schedule_or_throw(static_cast<int32_t>(std::max(1, nb_target)),
                                  /*fallback_d=*/0,
                                  params.block_size_seq,
                                  params.d_seq);
        // In sequence mode, run exactly the requested sweep schedule length.
        params.max_sweeps = static_cast<int32_t>(params.block_size_seq.size());
    }
    params.kd_work = std::min(static_cast<int>(kd * 3 / 2), static_cast<int>(n - 1));

    int64_t sweep_count = (g_mode == BandReductionMode::Phase) ? 1 : std::max<int64_t>(1, static_cast<int64_t>(params.max_sweeps));
    const double total_flops = double(n) * double(kd) * double(kd) * double(sweep_count) * double(batch);

    const size_t ws_bytes = sytrd_band_reduction_buffer_size<B, T>(*q,
                                                                   ab_in.view(),
                                                                   VectorView<Real>(d),
                                                                   VectorView<Real>(e),
                                                                   VectorView<T>(tau),
                                                                   uplo,
                                                                   kd,
                                                                   params);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(ab_in),
        d, e, tau, uplo, static_cast<int32_t>(kd), ws, params,
        [](Queue& q, auto&&... xs) {
            sytrd_band_reduction<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

// Register benchmarks
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_BAND_REDUCTION, BandReductionBenchSizes);

int main(int argc, char** argv) {
    std::vector<std::string> forwarded;
    forwarded.reserve(static_cast<size_t>(argc));
    forwarded.emplace_back(argv[0]);

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (starts_with(arg, "--nb-seq=")) {
                g_block_size_seq = parse_int_list(arg.substr(std::string("--nb-seq=").size()));
            } else if (starts_with(arg, "--d-seq=")) {
                g_d_seq = parse_int_list(arg.substr(std::string("--d-seq=").size()));
            } else if (starts_with(arg, "--mode=")) {
                g_mode = parse_mode_or_throw(arg.substr(std::string("--mode=").size()));
            } else if (starts_with(arg, "--phase-reduce=")) {
                g_phase_reduce = static_cast<int32_t>(std::stoi(arg.substr(std::string("--phase-reduce=").size())));
                if (g_phase_reduce < 0) {
                    throw std::runtime_error("--phase-reduce must be >= 0");
                }
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Band reduction benchmark options:\n"
                          << "  --mode=phase|sequence  Benchmark mode (default: phase)\n"
                          << "  --phase-reduce=d       Diagonal reduction target in phase mode\n"
                          << "  --nb-seq=v1,v2,...   Sequence of block sizes per sweep\n"
                          << "  --d-seq=v1,v2,...    Sequence of diagonals-per-sweep\n"
                          << "\n"
                          << "Notes:\n"
                          << "  - phase mode runs one sweep only (default).\n"
                          << "  - sequence mode runs one sweep per sequence element.\n"
                          << "  - If one sequence is omitted, it is filled with defaults of matching length.\n"
                          << "  - Provide positional minibench args to override default size grid, e.g.\n"
                          << "      ./band_reduction_benchmark 256 64 1 32 --mode=phase --phase-reduce=64\n"
                          << "      ./band_reduction_benchmark 256 64 1 32 --mode=sequence --nb-seq=32,16 --d-seq=16,8\n"
                          << "    (args: n kd batch nb).\n\n";
                forwarded.push_back(arg);
            } else {
                forwarded.push_back(arg);
            }
        }

        if (g_mode == BandReductionMode::Phase && (!g_block_size_seq.empty() || !g_d_seq.empty())) {
            throw std::runtime_error("--nb-seq/--d-seq require --mode=sequence");
        }

        std::vector<char*> forwarded_argv;
        forwarded_argv.reserve(forwarded.size());
        for (auto& s : forwarded) {
            forwarded_argv.push_back(s.data());
        }
        return minibench::MiniBenchMain(static_cast<int>(forwarded_argv.size()), forwarded_argv.data());
    } catch (const std::exception& ex) {
        std::cerr << "band_reduction_benchmark: " << ex.what() << "\n";
        return 2;
    }
}
