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

namespace minibench {

constexpr const char* kColorReset  = "\033[0m";
constexpr const char* kColorHeader = "\033[1;36m";
constexpr const char* kColorName   = "\033[1m";

struct Config {
    size_t warmup_runs = 2;          // number of warmup executions
    size_t min_iters = 5;            // minimum measurement iterations
    size_t max_iters = 100;          // safety cap on iterations
    double min_time_ms = 200.0;      // target minimum total measurement time
};

struct Result {
    std::string name;                       // benchmark name
    std::vector<int> args;                  // input arguments
    double avg_ms = 0.0;                    // average time per iteration
    double stddev_ms = 0.0;                 // standard deviation of times
    size_t iterations = 0;                  // number of measured iterations
    std::unordered_map<std::string, double> metrics; // all metrics
};

struct Metric {
    double value = 0.0;
    bool rate = false;  // divide by time if true
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
    size_t iter_limit_ = 1;

    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;
    std::chrono::duration<double> elapsed_{0};
    bool running_ = false;

    std::unordered_map<std::string, Metric> metrics_;
    MetricsFunc metrics_fn_;

    struct iterator {
        size_t i;
        size_t limit;
        State* state;
        bool operator!=(const iterator& other) const { return i != other.i; }
        void operator++() { ++i; }
        State& operator*() { return *state; }
    };

    iterator begin() { return {0, iter_limit_, this}; }
    iterator end() { return {iter_limit_, iter_limit_, this}; }

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

    void SetMetric(const std::string& name, double value, bool rate = false) {
        metrics_[name] = {value, rate};
    }

    void SetMetricsFunc(MetricsFunc fn) { metrics_fn_ = std::move(fn); }
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

    for (size_t i = 0; i < cfg.warmup_runs; ++i) {
        state.iter_limit_ = 1;
        state.ResetTiming();
        state.ResumeTiming();
        b.func(state);
        state.StopTiming();
    }

    std::vector<double> times;
    double total_ms = 0.0;
    while (res.iterations < cfg.max_iters &&
           (res.iterations < cfg.min_iters || total_ms < cfg.min_time_ms)) {
        state.iter_limit_ = 1;
        state.ResetTiming();
        state.ResumeTiming();
        b.func(state);
        double t = state.StopTiming();
        times.push_back(t);
        total_ms += t;
        ++res.iterations;
    }

    if (!times.empty()) {
        double sum = 0.0;
        for (double t : times) sum += t;
        res.avg_ms = sum / times.size();
        double var = 0.0;
        for (double t : times) var += (t - res.avg_ms) * (t - res.avg_ms);
        res.stddev_ms = std::sqrt(var / times.size());
    }

    res.metrics["time_ms"] = res.avg_ms;
    res.metrics["iterations"] = static_cast<double>(res.iterations);
    double secs = res.avg_ms / 1000.0;
    for (const auto& kv : state.metrics_) {
        double v = kv.second.value;
        if (kv.second.rate && secs > 0.0)
            v /= secs;
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
    std::cout << std::setprecision(5);

    std::cout << kColorHeader << std::left << std::setw(16) << "Name";
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
    std::cout << std::setprecision(5);

    std::cout << kColorName << std::left << std::setw(16) << r.name << kColorReset;
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

inline int RunRegisteredBenchmarks(const Config& cfg = {},
                                   const std::string& csv_path = "") {
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
};

inline CliOptions ParseCommandLine(int argc, char** argv) {
    CliOptions opt;
    std::vector<std::vector<int>> arg_values;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s.rfind("--warmup=", 0) == 0) {
            opt.cfg.warmup_runs = std::stoul(s.substr(9));
        } else if (s.rfind("--min_iters=", 0) == 0) {
            opt.cfg.min_iters = std::stoul(s.substr(12));
        } else if (s.rfind("--max_iters=", 0) == 0) {
            opt.cfg.max_iters = std::stoul(s.substr(12));
        } else if (s.rfind("--min_time=", 0) == 0) {
            opt.cfg.min_time_ms = std::stod(s.substr(11));
        } else if (s.rfind("--csv=", 0) == 0) {
            opt.csv_file = s.substr(6);
        } else if (s == "--help" || s == "-h") {
            std::cout << "Usage: benchmark [options] [ARGS...]\n";
            std::cout << "Options:\n";
            std::cout << "  --warmup=N       warmup iterations (default 2)\n";
            std::cout << "  --min_iters=N    minimum measured iterations\n";
            std::cout << "  --max_iters=N    maximum measured iterations\n";
            std::cout << "  --min_time=MS    minimum total measurement time\n";
            std::cout << "  --csv=FILE       write results to CSV file\n";
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
        if (!opts.args_list.empty() && b.args_list.empty()) {
            b.args_list = opts.args_list;
        }
    }
    return RunRegisteredBenchmarks(opts.cfg, opts.csv_file);
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
    for (int s : {64, 128, 256}) {
        b->Args({s, s, 1});
    }
    for (int s : {64, 128, 256}) {
        b->Args({s, s, 10});
    }
    for (int s : {64, 128, 256, 512}) {
        b->Args({s, s, 100});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, 1000});
    }
}

// Reduced size set for CPU/NETLIB benchmarks
template <typename Benchmark>
inline void SquareBatchSizesNetlib(Benchmark* b) {
    for (int s : {16, 32, 64}) {
        b->Args({s, s, 1});
    }
    for (int s : {16, 32, 64}) {
        b->Args({s, s, 10});
    }
    for (int s : {16, 32, 64, 128}) {
        b->Args({s, s, 100});
    }
    for (int s : {16, 32, 64, 128, 256}) {
        b->Args({s, s, 1000});
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
    for (int s : {64, 128, 256}) {
        b->Args({s, s, s, 1});
    }
    for (int s : {64, 128, 256}) {
        b->Args({s, s, s, 10});
    }
    for (int s : {64, 128, 256, 512}) {
        b->Args({s, s, s, 100});
    }
    for (int s : {64, 128, 256, 512, 1024}) {
        b->Args({s, s, s, 1000});
    }
}

// Reduced 3D sizes for CPU/NETLIB benchmarks
template <typename Benchmark>
inline void CubeBatchSizesNetlib(Benchmark* b) {
    for (int s : {16, 32, 64}) {
        b->Args({s, s, s, 1});
    }
    for (int s : {16, 32, 64}) {
        b->Args({s, s, s, 10});
    }
    for (int s : {16, 32, 64, 128}) {
        b->Args({s, s, s, 100});
    }
    for (int s : {16, 32, 64, 128, 256}) {
        b->Args({s, s, s, 1000});
    }
}

} // namespace minibench

