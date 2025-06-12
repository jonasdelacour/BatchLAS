#pragma once
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace simple_bench {

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
};

inline std::vector<Benchmark>& registry() {
    static std::vector<Benchmark> bench_list;
    return bench_list;
}

inline Benchmark* RegisterBenchmark(const std::string& name, BenchFunc func) {
    registry().emplace_back(name, std::move(func));
    return &registry().back();
}

#define SIMPLE_BENCHMARK(fn)                                              \
    static void fn(simple_bench::State&);                                 \
    static simple_bench::Benchmark* BENCHMARK_##fn =                      \
        simple_bench::RegisterBenchmark(#fn, fn);                         \
    static void fn(simple_bench::State& state)

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

inline void print_result(const Result& r) {
    std::cout << r.name;
    if (!r.args.empty()) {
        std::cout << " [";
        for (size_t i = 0; i < r.args.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << r.args[i];
        }
        std::cout << "]";
    }
    std::cout << "\n";
    std::cout << "Iterations: " << r.iterations
              << "  Avg(ms): " << r.avg_ms
              << "  StdDev(ms): " << r.stddev_ms << "\n";
    for (const auto& kv : r.metrics) {
        std::cout << "  " << kv.first << ": " << kv.second << "\n";
    }
}

inline int RunRegisteredBenchmarks(const Config& cfg = {}) {
    for (const auto& b : registry()) {
        if (b.args_list.empty()) {
            auto r = run_benchmark(b, {}, cfg);
            print_result(r);
        } else {
            for (const auto& a : b.args_list) {
                auto r = run_benchmark(b, a, cfg);
                print_result(r);
            }
        }
    }
    return 0;
}

struct CliOptions {
    Config cfg;
    std::vector<int> args;
};

inline CliOptions ParseCommandLine(int argc, char** argv) {
    CliOptions opt;
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
        } else {
            opt.args.push_back(std::stoi(s));
        }
    }
    return opt;
}

inline int SimpleBenchMain(int argc, char** argv) {
    auto opts = ParseCommandLine(argc, argv);
    for (auto& b : registry()) {
        if (!opts.args.empty() && b.args_list.empty()) {
            b.args_list.push_back(opts.args);
        }
    }
    return RunRegisteredBenchmarks(opts.cfg);
}

#define SIMPLE_BENCHMARK_MAIN()                        \
    int main(int argc, char** argv) {                  \
        return simple_bench::SimpleBenchMain(argc, argv); \
    }

} // namespace simple_bench

