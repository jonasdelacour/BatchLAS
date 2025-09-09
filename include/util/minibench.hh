#pragma once
// Helpers to generate unique variable names in macros
#define MINI_BENCH_CONCAT_INNER(x, y) x##y
#define MINI_BENCH_CONCAT(x, y) MINI_BENCH_CONCAT_INNER(x, y)
#define MINI_BENCH_UNIQUE_NAME(prefix) MINI_BENCH_CONCAT(prefix, __COUNTER__)

// Macro to register a benchmark and apply a sizing function at static-init time
// The benchmark name is derived from the function name.
#define MINI_BENCHMARK_REGISTER_SIZES(FUNC, SIZER)                           \
    static int MINI_BENCH_UNIQUE_NAME(_reg_sizes_) = ([]() {                 \
        SIZER(minibench::RegisterBenchmark(#FUNC, FUNC));                    \
        return 0;                                                            \
    })();
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <blas/enums.hh>

namespace minibench {

constexpr const char* kColorReset  = "\033[0m";
constexpr const char* kColorHeader = "\033[1;36m";
constexpr const char* kColorName   = "\033[1m";

struct Config {
    size_t warmup_runs = 2;              // number of warmup executions
    size_t warmup_internal_iterations = 1; // internal iterations for warmup
    size_t min_iters = 5;                // minimum measurement iterations
    size_t max_iters = 100;              // safety cap on iterations
    double min_time_ms = 200.0;          // target minimum total measurement time
};

struct Result {
    std::string name;                       // benchmark name
    std::vector<int> args;                  // input arguments
    double avg_ms = 0.0;                    // average time per iteration
    double stddev_ms = 0.0;                 // standard deviation of times
    size_t iterations = 0;                  // number of measured iterations
    std::unordered_map<std::string, double> metrics; // all metrics
};

enum MetricType {
    Rate,        // divide by time
    Reciprocal,  // multiply by time (reciprocal rate)
    Normal       // no special handling
};

struct Metric {
    double value = 0.0;
    MetricType type = Normal; // type of metric handling
};

// Helper to parse a single integer or a range/list specification of the form
// "start:end:num" or "v1,v2,v3". Returned values are rounded to int.
inline std::vector<int> parse_range_or_list(const std::string& str) {
    std::vector<int> vals;
    if (str.find(':') != std::string::npos) {
        std::vector<std::string> parts;
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ':')) parts.push_back(item);
        if (parts.size() >= 2) {
            double start = std::stod(parts[0]);
            double end   = std::stod(parts[1]);
            size_t num   = parts.size() >= 3 ? std::stoul(parts[2]) : 5;
            if (num < 2) num = 2;
            for (size_t i = 0; i < num; ++i) {
                double v = start + (end - start) * static_cast<double>(i) / (num - 1);
                vals.push_back(static_cast<int>(std::lround(v)));
            }
        }
    } else if (str.find(',') != std::string::npos) {
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) vals.push_back(std::stoi(item));
        }
    } else if (!str.empty()) {
        vals.push_back(std::stoi(str));
    }
    return vals;
}

inline std::vector<std::string> parse_list(const std::string& str) {
    std::vector<std::string> vals;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) vals.push_back(item);
    }
    return vals;
}

inline std::vector<int> make_range(int start, int end, int step = 1) {
    std::vector<int> vals;
    if (step == 0) step = 1;
    if (end < start && step > 0) step = -step;
    for (int v = start; (step > 0 ? v <= end : v >= end); v += step) {
        vals.push_back(v);
    }
    return vals;
}

using MetricsFunc = std::function<void(Result&)>;

struct State {
    explicit State(const std::vector<int>& ranges) : ranges_(ranges) {}

    std::vector<int> ranges_;
    size_t internal_iterations_ = 1;  // number of internal iterations per benchmark call

    using clock = std::chrono::steady_clock;
    clock::time_point start_;
    std::chrono::duration<double> elapsed_{0};
    bool running_ = false;

    std::unordered_map<std::string, Metric> metrics_;
    MetricsFunc metrics_fn_;

    // Optional: user can register a single "kernel" (work unit) to run.
    // If provided, the framework will: call the benchmark once to perform setup
    // and register the kernel, then execute warmups and measurements by calling
    // only this kernel repeatedly (avoiding re-doing setup each time).
    std::function<void()> kernel_once_;
    std::function<void()> batch_end_fn_;

    struct iterator {
        size_t i;
        size_t limit;
        State* state;
        bool operator!=(const iterator& other) const { return i != other.i; }
        void operator++() { ++i; }
        State& operator*() { return *state; }
    };

    iterator begin() { return {0, internal_iterations_, this}; }
    iterator end() { return {internal_iterations_, internal_iterations_, this}; }

    int range(size_t idx) const { return idx < ranges_.size() ? ranges_[idx] : 0; }

    void PauseTiming() {
        if (running_) {
            auto now = clock::now();
            elapsed_ += now - start_;
            running_ = false;
        }
    }

    void ResumeTiming() {
        if (!running_) {
            start_ = clock::now();
            running_ = true;
        }
    }

    void ResetTiming() {
        elapsed_ = std::chrono::duration<double>(0);
        running_ = false;
    }

    double StopTiming() {
        if (running_) {
            auto now = clock::now();
            elapsed_ += now - start_;
            running_ = false;
        }
        return std::chrono::duration<double, std::milli>(elapsed_).count();
    }

    void SetMetric(const std::string& name, double value, MetricType type = Normal) {
        metrics_[name] = {value, type};
    }

    void SetInternalIterations(size_t iterations) {
        internal_iterations_ = iterations;
    }

    void SetMetricsFunc(MetricsFunc fn) { metrics_fn_ = std::move(fn); }

    // Structured mode: register the kernel body and optional end-of-batch hook
    // (e.g., to insert a single queue.wait() after a batch of internal iters).
    void SetKernel(std::function<void()> kernel_once) { kernel_once_ = std::move(kernel_once); }
    void SetBatchEnd(std::function<void()> fn) { batch_end_fn_ = std::move(fn); }
    bool HasKernel() const { return static_cast<bool>(kernel_once_); }
    void RunKernelOnce() { if (kernel_once_) kernel_once_(); }
    void RunBatchEnd() { if (batch_end_fn_) batch_end_fn_(); }
};

// Benchmark representation and registry
using BenchFunc = std::function<void(State&)>;

struct Benchmark {
    std::string name;
    BenchFunc func;
    std::vector<std::vector<int>> args_list;

    Benchmark(const std::string& n, BenchFunc f) : name(n), func(std::move(f)) {}

    Benchmark* Args(const std::vector<int>& a) {
        args_list.push_back(a);
        return this;
    }

    Benchmark* ArgRange(int start, int end, int step = 1) {
        if (step == 0) step = 1;
        if (end < start && step > 0) step = -step;
        for (int v = start; (step > 0 ? v <= end : v >= end); v += step) {
            args_list.push_back({v});
        }
        return this;
    }

    Benchmark* ArgsProduct(const std::vector<std::vector<int>>& ranges) {
        std::vector<std::vector<int>> combos{{}};
        for (const auto& r : ranges) {
            std::vector<std::vector<int>> next;
            for (const auto& c : combos) {
                for (int v : r) {
                    auto tmp = c;
                    tmp.push_back(v);
                    next.push_back(std::move(tmp));
                }
            }
            combos = std::move(next);
        }
        for (auto& c : combos) args_list.push_back(std::move(c));
        return this;
    }
};

inline std::vector<Benchmark>& registry() {
    static std::vector<Benchmark> bench_list;
    return bench_list;
}

inline Benchmark* RegisterBenchmark(const std::string& name, BenchFunc func) {
    registry().emplace_back(name, std::move(func));
    return &registry().back();
}

#define MINI_BENCHMARK(fn)                                                \
    static void fn(minibench::State&);                                    \
    static minibench::Benchmark* BENCHMARK_##fn =                         \
        minibench::RegisterBenchmark(#fn, fn);                            \
    static void fn(minibench::State& state)

// Run a single benchmark function with specified arguments and configuration
inline Result run_benchmark(const Benchmark& b,
                            const std::vector<int>& args,
                            const Config& cfg) {
    Result res;
    res.name = b.name;
    res.args = args;

    State state(args);

    // First call gives the benchmark a chance to perform setup and optionally
    // register a kernel for structured execution. This call is not timed here.
    b.func(state);

    std::vector<double> times;
    double total_ms = 0.0;

    if (state.HasKernel()) {
        // Structured mode: warm up and measure by calling the registered kernel
        // repeatedly without re-running setup, which is critical for USM.

        // Warmup calls (not timed). Ensure completion after each warmup batch
        // and once more before starting timed measurement to avoid contamination
        // from any outstanding async work (e.g., USM migrations).
        for (size_t i = 0; i < cfg.warmup_runs; ++i) {
            for (size_t j = 0; j < cfg.warmup_internal_iterations; ++j) {
                state.RunKernelOnce();
            }
            state.RunBatchEnd();
        }

        // Determine optimal internal iterations for measurement by sampling
        size_t measurement_internal_iters = 1;
        state.ResetTiming();
        state.ResumeTiming();
        state.RunKernelOnce();
        state.RunBatchEnd();
        double sample_time = state.StopTiming();

        while (sample_time < 1.0 && measurement_internal_iters < 10000) {
            measurement_internal_iters *= 10;
            state.ResetTiming();
            state.ResumeTiming();
            for (size_t i = 0; i < measurement_internal_iters; ++i) state.RunKernelOnce();
            state.RunBatchEnd();
            sample_time = state.StopTiming();
        }

        // Measurements
        while (res.iterations < cfg.max_iters &&
               (res.iterations < cfg.min_iters || total_ms < cfg.min_time_ms)) {
            state.ResetTiming();
            state.ResumeTiming();
            for (size_t i = 0; i < measurement_internal_iters; ++i) state.RunKernelOnce();
            state.RunBatchEnd();
            double t = state.StopTiming();
            double time_per_iter = t / measurement_internal_iters;
            times.push_back(time_per_iter);
            total_ms += time_per_iter;
            ++res.iterations;
        }
    } else {
        // Legacy mode: the benchmark function controls timing and iterations.
        // We perform warmup and measurement by calling the function as before.

        // Warmup with internal iterations
        for (size_t i = 0; i < cfg.warmup_runs; ++i) {
            state.internal_iterations_ = cfg.warmup_internal_iterations;
            state.ResetTiming();
            state.ResumeTiming();
            b.func(state);
            state.StopTiming();
        }

        // Determine optimal internal iterations for measurement
        size_t measurement_internal_iters = 1;
        state.internal_iterations_ = measurement_internal_iters;
        state.ResetTiming();
        state.ResumeTiming();
        b.func(state);
        double sample_time = state.StopTiming();

        while (sample_time < 1.0 && measurement_internal_iters < 10000) {
            measurement_internal_iters *= 10;
            state.internal_iterations_ = measurement_internal_iters;
            state.ResetTiming();
            state.ResumeTiming();
            b.func(state);
            sample_time = state.StopTiming();
        }

        // Measurements
        while (res.iterations < cfg.max_iters &&
               (res.iterations < cfg.min_iters || total_ms < cfg.min_time_ms)) {
            state.internal_iterations_ = measurement_internal_iters;
            state.ResetTiming();
            state.ResumeTiming();
            b.func(state);
            double t = state.StopTiming();
            double time_per_iter = t / measurement_internal_iters;
            times.push_back(time_per_iter);
            total_ms += time_per_iter;
            ++res.iterations;
        }
    }

    if (!times.empty()) {
        double sum = 0.0;
        for (double t : times) sum += t;
        res.avg_ms = sum / times.size();
        double var = 0.0;
        for (double t : times) var += (t - res.avg_ms) * (t - res.avg_ms);
        res.stddev_ms = std::sqrt(var / times.size());
    }

    double secs = res.avg_ms / 1000.0;
    for (const auto& kv : state.metrics_) {
        double v = kv.second.value;
        if (kv.second.type == MetricType::Rate && secs > 0.0)
            v /= secs;
        else if (kv.second.type == MetricType::Reciprocal && secs > 0.0)
            v *= secs;
        res.metrics[kv.first] = v;
    }
    if (state.metrics_fn_) {
        state.metrics_fn_(res);
    }

    return res;
}

inline void print_header(const std::vector<std::string>& metric_names,
                        size_t max_args) {
    std::ios orig_state(nullptr);
    orig_state.copyfmt(std::cout);
    std::cout << std::showpoint << std::setprecision(5);

    constexpr int name_width = 50;
    std::cout << kColorHeader << std::left << std::setw(name_width) << "Name";
    for (size_t i = 0; i < max_args; ++i) {
        std::cout << std::setw(8) << ("Arg" + std::to_string(i));
    }
    std::cout << std::setw(12) << "Iter"
              << std::setw(12) << "Avg(ms)"
              << std::setw(12) << "Std(ms)";
    for (const auto& m : metric_names) {
        std::cout << std::setw(12) << m;
    }
    std::cout << kColorReset << '\n';

    std::cout.copyfmt(orig_state);
}

inline void print_row(const Result& r,
                      const std::vector<std::string>& metric_names,
                      size_t max_args) {
    std::ios orig_state(nullptr);
    orig_state.copyfmt(std::cout);
    std::cout << std::showpoint << std::setprecision(5);

    constexpr int name_width = 50;
    std::cout << kColorName << std::left << std::setw(name_width) << r.name << kColorReset;
    for (size_t i = 0; i < max_args; ++i) {
        if (i < r.args.size())
            std::cout << std::setw(8) << r.args[i];
        else
            std::cout << std::setw(8) << "";
    }
    std::cout << std::setw(12) << r.iterations
              << std::setw(12) << r.avg_ms
              << std::setw(12) << r.stddev_ms;
    for (const auto& m : metric_names) {
        auto it = r.metrics.find(m);
        if (it != r.metrics.end()) {
            double v = it->second;
            if (std::fabs(v - std::round(v)) < 1e-9)
                std::cout << std::setw(12)
                          << static_cast<long long>(std::llround(v));
            else
                std::cout << std::setw(12) << v;
        } else {
            std::cout << std::setw(12) << "";
        }
    }
    std::cout << '\n';

    std::cout.copyfmt(orig_state);
}

inline void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open CSV file: " << path << '\n';
        return;
    }
    size_t max_args = 0;
    std::vector<std::string> metric_names;
    std::unordered_map<std::string, bool> metric_seen;
    for (const auto& r : results) {
        max_args = std::max(max_args, r.args.size());
        for (const auto& kv : r.metrics) {
            if (!metric_seen[kv.first]) {
                metric_names.push_back(kv.first);
                metric_seen[kv.first] = true;
            }
        }
    }
    f << "name";
    for (size_t i = 0; i < max_args; ++i) f << ",arg" << i;
    f << ",iterations,avg_ms,stddev_ms";
    for (const auto& m : metric_names) f << ',' << m;
    f << '\n';
    for (const auto& r : results) {
        f << r.name;
        for (size_t i = 0; i < max_args; ++i) {
            if (i < r.args.size()) f << ',' << r.args[i];
            else f << ',';
        }
        f << ',' << r.iterations
          << ',' << r.avg_ms
          << ',' << r.stddev_ms;
        for (const auto& m : metric_names) {
            auto it = r.metrics.find(m);
            if (it != r.metrics.end()) f << ',' << it->second; else f << ',';
        }
        f << '\n';
    }
}

inline bool match_filter(const Benchmark& b,
                         const std::vector<std::string>& backends,
                         const std::vector<std::string>& types) {
    if (!backends.empty()) {
        bool ok = false;
        for (const auto& be : backends) {
            if (b.name.find("Backend::" + be) != std::string::npos) {
                ok = true;
                break;
            }
        }
        if (!ok) return false;
    }
    if (!types.empty()) {
        bool ok = false;
        for (const auto& t : types) {
            if (t == "float" && b.name.find("float") != std::string::npos)
                ok = true;
            else if (t == "double" && b.name.find("double") != std::string::npos)
                ok = true;
            else if ((t == "cfloat" || t == "complex<float>") &&
                     b.name.find("complex<float>") != std::string::npos)
                ok = true;
            else if ((t == "cdouble" || t == "complex<double>") &&
                     b.name.find("complex<double>") != std::string::npos)
                ok = true;
            if (ok) break;
        }
        if (!ok) return false;
    }
    return true;
}

inline int RunRegisteredBenchmarks(const Config& cfg = {},
                                   const std::string& csv_path = "",
                                   const std::vector<std::string>& backends = {},
                                   const std::vector<std::string>& types = {}) {
    std::vector<Result> results;
    std::vector<std::string> metric_names;
    std::unordered_map<std::string, bool> metric_seen;
    size_t max_args = 0;
    bool header_printed = false;

    auto update_header = [&](const Result& r) {
        bool changed = false;
        max_args = std::max(max_args, r.args.size());
        for (const auto& kv : r.metrics) {
            if (!metric_seen[kv.first]) {
                metric_names.push_back(kv.first);
                metric_seen[kv.first] = true;
                changed = true;
            }
        }
        if (!header_printed || changed) {
            if (header_printed) std::cout << "\n";
            print_header(metric_names, max_args);
            header_printed = true;
        }
    };

    for (const auto& b : registry()) {
        if (!match_filter(b, backends, types))
            continue;
        if (b.args_list.empty()) {
            auto r = run_benchmark(b, {}, cfg);
            update_header(r);
            print_row(r, metric_names, max_args);
            results.push_back(r);
        } else {
            for (const auto& a : b.args_list) {
                auto r = run_benchmark(b, a, cfg);
                update_header(r);
                print_row(r, metric_names, max_args);
                results.push_back(r);
            }
        }
    }

    if (!csv_path.empty()) write_csv(csv_path, results);
    return 0;
}

struct CliOptions {
    Config cfg;
    std::vector<std::vector<int>> args_list;
    std::string csv_file;
    std::vector<std::string> backends;
    std::vector<std::string> types;
};

inline CliOptions ParseCommandLine(int argc, char** argv) {
    CliOptions opt;
    std::vector<std::vector<int>> arg_values;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s.rfind("--warmup=", 0) == 0) {
            opt.cfg.warmup_runs = std::stoul(s.substr(9));
        } else if (s.rfind("--warmup_internal=", 0) == 0) {
            opt.cfg.warmup_internal_iterations = std::stoul(s.substr(18));
        } else if (s.rfind("--min_iters=", 0) == 0) {
            opt.cfg.min_iters = std::stoul(s.substr(12));
        } else if (s.rfind("--max_iters=", 0) == 0) {
            opt.cfg.max_iters = std::stoul(s.substr(12));
        } else if (s.rfind("--min_time=", 0) == 0) {
            opt.cfg.min_time_ms = std::stod(s.substr(11));
        } else if (s.rfind("--csv=", 0) == 0) {
            opt.csv_file = s.substr(6);
        } else if (s.rfind("--backend=", 0) == 0) {
            opt.backends = parse_list(s.substr(10));
        } else if (s.rfind("--type=", 0) == 0) {
            opt.types = parse_list(s.substr(7));
        } else if (s == "--help" || s == "-h") {
            std::cout << "Usage: benchmark [options] [ARGS...]\n";
            std::cout << "Options:\n";
            std::cout << "  --warmup=N           warmup iterations (default 2)\n";
            std::cout << "  --warmup_internal=N  internal iterations for warmup (default 1)\n";
            std::cout << "  --min_iters=N        minimum measured iterations\n";
            std::cout << "  --max_iters=N        maximum measured iterations\n";
            std::cout << "  --min_time=MS        minimum total measurement time\n";
            std::cout << "  --csv=FILE           write results to CSV file\n";
            std::cout << "  --backend=LIST       comma separated backends to run\n";
            std::cout << "  --type=LIST          comma separated floating point types\n";
            std::cout << "  ARGS can be integers, comma lists or start:end:num ranges\n";
            exit(0);
        } else {
            arg_values.push_back(parse_range_or_list(s));
        }
    }

    if (!arg_values.empty()) {
        std::vector<std::vector<int>> combos{{}};
        for (const auto& vals : arg_values) {
            std::vector<std::vector<int>> next;
            for (const auto& c : combos) {
                for (int v : vals) {
                    auto tmp = c;
                    tmp.push_back(v);
                    next.push_back(std::move(tmp));
                }
            }
            combos = std::move(next);
        }
        opt.args_list = std::move(combos);
    }

    return opt;
}

inline int MiniBenchMain(int argc, char** argv) {
    auto opts = ParseCommandLine(argc, argv);
    for (auto& b : registry()) {
        if (!opts.args_list.empty()) {
            b.args_list = opts.args_list;
        }
    }
    return RunRegisteredBenchmarks(opts.cfg, opts.csv_file,
                                   opts.backends, opts.types);
}

#define MINI_BENCHMARK_MAIN()                          \
    int main(int argc, char** argv) {                  \
        return minibench::MiniBenchMain(argc, argv); \
    }


template <typename Benchmark>
inline void SquareSizes(Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s});
    }
}

template <typename Benchmark>
inline void SquareBatchSizes(Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
            b->Args({s, s, bs});
        }
    }
}

// Reduced size set for CPU/NETLIB benchmarks
template <typename Benchmark>
inline void SquareBatchSizesNetlib(Benchmark* b) {
    for (int s : {16, 32, 64, 128}) {
        for (int bs : {1, 10, 100}) {
            b->Args({s, s, bs});
        }
    }
}

template <typename Benchmark>
inline void CubeSizes(Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        b->Args({s, s, s});
    }
}

template <typename Benchmark>
inline void CubeBatchSizes(Benchmark* b) {
    for (int s : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
            b->Args({s, s, s, bs});
        }
    }
}

// Reduced 3D sizes for CPU/NETLIB benchmarks
template <typename Benchmark>
inline void CubeBatchSizesNetlib(Benchmark* b) {
    for (int s : {16, 32, 64, 128}) {
        for (int bs : {1, 10, 100}) {
            b->Args({s, s, s, bs});
        }
    }
}

template <typename Benchmark>
inline void SyevxBenchSizes(Benchmark* b) {
    for (int n : {128, 256, 512}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
            for (int ne : {5, 10, 15}) {
                b->Args({n, bs, ne});
            }
        }
    }
}

template <typename Benchmark>
inline void SyevxBenchSizesNetlib(Benchmark* b) {
    for (int n : {64, 128, 256}) {
        for (int bs : {1, 10, 100}) {
            for (int ne : {2, 4, 8}) {
                b->Args({n, bs, ne});
            }
        }
    }
}


template <typename Benchmark>
inline void OrthoBenchSizes(Benchmark* b) {
    for (int algo = 0; algo < static_cast<int>(batchlas::OrthoAlgorithm::NUM_ALGORITHMS); ++algo) {
        for (int m : {64, 128, 256, 512, 1024}) {
            for (int n : {64, 128, 256, 512, 1024}) {
                for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
                    if (m >= n) {
                        b->Args({m, n, bs, algo});
                    }
                }
            }
        }
    }
}

template <typename Benchmark>
inline void OrthoBenchSizesNetlib(Benchmark* b) {
    for (int algo = 0; algo < static_cast<int>(batchlas::OrthoAlgorithm::NUM_ALGORITHMS); ++algo) {
        for (int m : {16, 32, 64, 128}) {
            for (int n : {16, 32, 64, 128}) {
                for (int bs : {1, 10, 100}) {
                    if (m >= n) {
                        b->Args({m, n, bs, algo});
                    }
                }
            }
        }
    }
}

} // namespace minibench
