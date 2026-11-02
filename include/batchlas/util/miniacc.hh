// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/// @file
/// @brief miniacc: a header-only accuracy-benchmark harness (register cases, sample, summarise, CSV).
///
/// Registered with MINI_ACC_BENCHMARK_REGISTER_SIZES; a program built with MINI_ACC_MAIN()
/// parses `--samples`, `--seed`, `--log10-cond`, `--csv`, `--benchmark_filter`,
/// `--backend`, `--type` and positional argument lists. Installed because it is
/// under include/batchlas; not a stable interface.
/// @ingroup api_internal_helpers

// Helpers to generate unique variable names in macros.
#define MINI_ACC_CONCAT_INNER(x, y) x##y
#define MINI_ACC_CONCAT(x, y) MINI_ACC_CONCAT_INNER(x, y)
#define MINI_ACC_UNIQUE_NAME(prefix) MINI_ACC_CONCAT(prefix, __COUNTER__)

// Register a benchmark and apply a sizing function at static-init time.
#define MINI_ACC_BENCHMARK_REGISTER_SIZES(FUNC, SIZER)                         \
    static int MINI_ACC_UNIQUE_NAME(_reg_sizes_) = ([]() {                     \
        SIZER(miniacc::RegisterBenchmark(#FUNC, FUNC));                        \
        return 0;                                                               \
    })();

namespace miniacc {

constexpr const char* kColorReset = "\033[0m";
constexpr const char* kColorHeader = "\033[1;36m";
constexpr const char* kColorName = "\033[1m";

/// @brief Run-wide settings parsed from the command line.
/// @ingroup api_internal_helpers
struct Config {
    size_t samples = 1024;
    unsigned int seed = 1234u;
    std::vector<double> log10_cond_values;
};

struct SampleRow {
    size_t sample_index = 0;
    bool ok = true;
    int info_code = 0;
    std::string failure_reason;
    std::unordered_map<std::string, double> metrics;
    std::unordered_map<std::string, std::string> tags;
};

struct MetricSummary {
    size_t count = 0;
    size_t finite_count = 0;
    size_t nan_count = 0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

struct Result {
    std::string name;
    std::vector<double> args;
    double target_log10_cond = std::numeric_limits<double>::quiet_NaN();
    size_t sample_count = 0;
    size_t failure_count = 0;
    double failure_rate = 0.0;
    std::unordered_map<std::string, MetricSummary> summary;
    std::vector<SampleRow> sample_rows;
};

/// @brief Parses a scalar, a comma list `v1,v2,...` or a range `start:end[:num]` into values.
///
/// A range gives \f$\mathrm{num}\f$ evenly spaced values from start to end inclusive
/// (default 5, at least 2).
/// @param[in] str  the specification; empty, or a range with fewer than two fields, gives no values
/// @return the parsed values, in order
/// @throws std::invalid_argument or std::out_of_range if a field is not a representable number (from `std::stod` / `std::stoul`)
inline std::vector<double> parse_range_or_list_double(const std::string& str) {
    std::vector<double> vals;
    if (str.empty()) return vals;

    if (str.find(':') != std::string::npos) {
        std::vector<std::string> parts;
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ':')) parts.push_back(item);
        if (parts.size() >= 2) {
            const double start = std::stod(parts[0]);
            const double end = std::stod(parts[1]);
            size_t num = parts.size() >= 3 ? std::stoul(parts[2]) : 5;
            if (num < 2) num = 2;
            for (size_t i = 0; i < num; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(num - 1);
                vals.push_back(start + (end - start) * t);
            }
        }
        return vals;
    }

    if (str.find(',') != std::string::npos) {
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) vals.push_back(std::stod(item));
        }
        return vals;
    }

    vals.push_back(std::stod(str));
    return vals;
}

/// @brief Splits a comma-separated list, dropping empty items.
/// @param[in] str  the list, e.g. `float,double`
/// @return the non-empty items, in order
inline std::vector<std::string> parse_list(const std::string& str) {
    std::vector<std::string> vals;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) vals.push_back(item);
    }
    return vals;
}

/// @brief Drops every value within @p eps of an earlier kept value.
/// @param[in] input  values to deduplicate
/// @param[in] eps    absolute tolerance: \f$v\f$ is dropped if \f$\lvert u - v \rvert \le \mathrm{eps}\f$ for a kept \f$u\f$
/// @return the kept values, in their original order
inline std::vector<double> dedup_nearly_equal(const std::vector<double>& input,
                                              double eps = 1e-12) {
    std::vector<double> out;
    out.reserve(input.size());
    for (double v : input) {
        bool seen = false;
        for (double u : out) {
            if (std::abs(u - v) <= eps) {
                seen = true;
                break;
            }
        }
        if (!seen) out.push_back(v);
    }
    return out;
}

/// @brief Per-case context handed to a benchmark: its arguments, the run settings, and the sample sink.
/// @ingroup api_internal_helpers
class State {
public:
    /// @brief Creates the context for one (case, argument list, condition target) run.
    /// @param[in] args               the case's argument list, read by arg() / arg_int()
    /// @param[in] cfg                run-wide settings
    /// @param[in] target_log10_cond  requested \f$\log_{10} \kappa\f$; NaN when none was asked for
    /// @param[in] benchmark_name     the registered case name
    State(std::vector<double> args,
          Config cfg,
          double target_log10_cond,
          std::string benchmark_name)
        : args_(std::move(args)),
          cfg_(std::move(cfg)),
          target_log10_cond_(target_log10_cond),
          benchmark_name_(std::move(benchmark_name)) {}

    /// @brief Argument @p idx of the case's argument list.
    /// @param[in] idx  0-based position in the argument list
    /// @return the argument, or 0.0 when @p idx is past the end
    double arg(size_t idx) const {
        return idx < args_.size() ? args_[idx] : 0.0;
    }

    /// @brief Argument @p idx rounded to the nearest integer.
    /// @param[in] idx  0-based position in the argument list
    /// @return the rounded argument, or 0 when @p idx is past the end
    int arg_int(size_t idx) const {
        return static_cast<int>(std::lround(arg(idx)));
    }

    /// @brief Number of samples the case should record (`--samples`).
    /// @return Config::samples
    size_t samples() const { return cfg_.samples; }
    /// @brief Base RNG seed (`--seed`).
    /// @return Config::seed
    unsigned int seed() const { return cfg_.seed; }
    /// @brief Requested \f$\log_{10} \kappa\f$ of this run.
    /// @return the target, or NaN when `--log10-cond` was not given
    double target_log10_cond() const { return target_log10_cond_; }
    /// @brief Registered name of the running case.
    /// @return the name
    const std::string& benchmark_name() const { return benchmark_name_; }

    /// @brief Sets a case-level tag copied into every sample recorded after this call.
    /// @param[in] key    tag name (a `tag_<key>` CSV column)
    /// @param[in] value  tag value; replaces an earlier value for @p key
    void SetTag(const std::string& key, const std::string& value) {
        case_tags_[key] = value;
    }

    /// @brief Appends one sample row.
    /// @param[in] metrics         named metric values; non-finite values count as missing in the summary
    /// @param[in] ok              false marks the sample as a failure
    /// @param[in] failure_reason  free-text reason for a failure
    /// @param[in] info_code       status code of the call under test (e.g. LAPACK `info`)
    /// @param[in] tags            per-sample tags; they override case tags of the same name
    void RecordSample(const std::unordered_map<std::string, double>& metrics,
                      bool ok = true,
                      const std::string& failure_reason = "",
                      int info_code = 0,
                      const std::unordered_map<std::string, std::string>& tags = {}) {
        SampleRow row;
        row.sample_index = rows_.size();
        row.ok = ok;
        row.info_code = info_code;
        row.failure_reason = failure_reason;
        row.metrics = metrics;
        row.tags = case_tags_;
        for (const auto& kv : tags) {
            row.tags[kv.first] = kv.second;
        }
        rows_.push_back(std::move(row));
    }

    /// @brief Samples recorded so far.
    /// @return the rows, in recording order
    const std::vector<SampleRow>& rows() const { return rows_; }

private:
    std::vector<double> args_;
    Config cfg_;
    double target_log10_cond_;
    std::string benchmark_name_;
    std::unordered_map<std::string, std::string> case_tags_;
    std::vector<SampleRow> rows_;
};

using BenchFunc = std::function<void(State&)>;

/// @brief A registered case: name, function, and its argument lists (Args, ArgRange, ArgsProduct).
/// @ingroup api_internal_helpers
struct Benchmark {
    std::string name;
    BenchFunc func;
    std::vector<std::vector<double>> args_list;

    /// @brief Creates a case with no argument lists.
    /// @param[in] n  case name
    /// @param[in] f  function run once per argument list and condition target
    Benchmark(const std::string& n, BenchFunc f) : name(n), func(std::move(f)) {}

    /// @brief Appends one argument list.
    /// @param[in] a  the argument list
    /// @return this, for chaining
    Benchmark* Args(const std::vector<double>& a) {
        args_list.push_back(a);
        return this;
    }

    /// @brief Appends @p num single-argument lists evenly spaced over \f$[\mathrm{start}, \mathrm{end}]\f$.
    ///
    /// List \f$i\f$ holds \f$\mathrm{start} + (\mathrm{end} - \mathrm{start}) \, \frac{i}{\mathrm{num} - 1}\f$
    /// for \f$i = 0, \dots, \mathrm{num} - 1\f$.
    /// @param[in] start  first value
    /// @param[in] end    last value
    /// @param[in] num    number of values; below 2 is raised to 2
    /// @return this, for chaining
    Benchmark* ArgRange(double start, double end, size_t num = 5) {
        if (num < 2) num = 2;
        for (size_t i = 0; i < num; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(num - 1);
            args_list.push_back({start + (end - start) * t});
        }
        return this;
    }

    /// @brief Appends the Cartesian product of @p ranges, one argument list per combination.
    /// @param[in] ranges  candidate values per argument position; the last position varies fastest
    /// @return this, for chaining
    Benchmark* ArgsProduct(const std::vector<std::vector<double>>& ranges) {
        std::vector<std::vector<double>> combos{{}};
        for (const auto& r : ranges) {
            std::vector<std::vector<double>> next;
            for (const auto& c : combos) {
                for (double v : r) {
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

/// @brief The process-wide list of registered cases.
/// @return the registry, in registration order
inline std::vector<Benchmark>& registry() {
    static std::vector<Benchmark> bench_list;
    return bench_list;
}

/// @brief Registers a case.
/// @param[in] name  case name, matched by `--benchmark_filter`, `--backend` and `--type`
/// @param[in] func  function run once per argument list and condition target
/// @return the new case, for adding argument lists; valid until the next registration
inline Benchmark* RegisterBenchmark(const std::string& name, BenchFunc func) {
    registry().emplace_back(name, std::move(func));
    return &registry().back();
}

/// @brief Mean, 95th percentile and maximum of one metric over the finite samples.
///
/// The percentile is the value at 0-based index \f$\lceil 0.95 \, (n - 1) \rceil\f$ of the
/// \f$n\f$ finite values sorted ascending.
/// @param[in] rows         recorded samples
/// @param[in] metric_name  metric to summarise; a row without it counts as NaN
/// @return the summary; mean, p95 and max are NaN when no sample is finite
inline MetricSummary summarize_metric(const std::vector<SampleRow>& rows,
                                      const std::string& metric_name) {
    MetricSummary out;
    out.count = rows.size();

    std::vector<double> finite_values;
    finite_values.reserve(rows.size());

    for (const auto& r : rows) {
        auto it = r.metrics.find(metric_name);
        if (it == r.metrics.end() || !std::isfinite(it->second)) {
            ++out.nan_count;
            continue;
        }
        finite_values.push_back(it->second);
    }

    out.finite_count = finite_values.size();
    if (finite_values.empty()) return out;

    std::sort(finite_values.begin(), finite_values.end());
    double sum = 0.0;
    for (double v : finite_values) sum += v;
    out.mean = sum / static_cast<double>(finite_values.size());
    out.max = finite_values.back();

    const size_t n = finite_values.size();
    const size_t p95_idx = (n == 1) ? 0 : static_cast<size_t>(std::ceil(0.95 * static_cast<double>(n - 1)));
    out.p95 = finite_values[p95_idx];

    return out;
}

/// @brief Runs one case on one argument list and condition target and summarises every metric.
/// @param[in] b                  the case
/// @param[in] args               its argument list
/// @param[in] cfg                run-wide settings
/// @param[in] target_log10_cond  requested \f$\log_{10} \kappa\f$, or NaN
/// @param[in] keep_samples       keep the per-sample rows in Result::sample_rows (for the CSV)
/// @return the result
inline Result run_benchmark(const Benchmark& b,
                            const std::vector<double>& args,
                            const Config& cfg,
                            double target_log10_cond,
                            bool keep_samples) {
    State state(args, cfg, target_log10_cond, b.name);
    b.func(state);

    Result res;
    res.name = b.name;
    res.args = args;
    res.target_log10_cond = target_log10_cond;
    res.sample_count = state.rows().size();

    for (const auto& row : state.rows()) {
        if (!row.ok) ++res.failure_count;
    }
    if (res.sample_count > 0) {
        res.failure_rate = static_cast<double>(res.failure_count) / static_cast<double>(res.sample_count);
    }

    std::unordered_map<std::string, bool> seen_metrics;
    std::vector<std::string> metric_names;
    for (const auto& row : state.rows()) {
        for (const auto& kv : row.metrics) {
            if (!seen_metrics[kv.first]) {
                metric_names.push_back(kv.first);
                seen_metrics[kv.first] = true;
            }
        }
    }
    for (const auto& m : metric_names) {
        res.summary[m] = summarize_metric(state.rows(), m);
    }

    if (keep_samples) {
        res.sample_rows = state.rows();
    }

    return res;
}

/// @brief Whether a case passes the command-line filters; an empty filter passes everything.
/// @param[in] b                 the case
/// @param[in] benchmark_filter  glob on the case name, with `*` and `?` wildcards
/// @param[in] backends          backend names; the name must contain `Backend::<name>` for one of them
/// @param[in] types             `float`, `double`, `cfloat` / `complex<float>`, `cdouble` / `complex<double>`
/// @return true if all three filters pass
inline bool match_filter(const Benchmark& b,
                         const std::string& benchmark_filter,
                         const std::vector<std::string>& backends,
                         const std::vector<std::string>& types) {
    if (!benchmark_filter.empty()) {
        const auto glob_match = [](const std::string& pattern, const std::string& text) {
            size_t p = 0;
            size_t t = 0;
            size_t star = std::string::npos;
            size_t match = 0;
            while (t < text.size()) {
                if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
                    ++p;
                    ++t;
                } else if (p < pattern.size() && pattern[p] == '*') {
                    star = p++;
                    match = t;
                } else if (star != std::string::npos) {
                    p = star + 1;
                    t = ++match;
                } else {
                    return false;
                }
            }
            while (p < pattern.size() && pattern[p] == '*') ++p;
            return p == pattern.size();
        };

        if (!glob_match(benchmark_filter, b.name)) return false;
    }

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
            if (t == "float" &&
                b.name.find("<float") != std::string::npos &&
                b.name.find("complex<float>") == std::string::npos)
                ok = true;
            else if (t == "double" &&
                     b.name.find("<double") != std::string::npos &&
                     b.name.find("complex<double>") == std::string::npos)
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

/// @brief Formats a summary cell with 6 significant digits.
/// @param[in] v  the value
/// @return the text, or an empty string for a non-finite @p v
inline std::string format_value(double v) {
    if (!std::isfinite(v)) return "";
    std::ostringstream oss;
    oss << std::setprecision(6) << std::showpoint << v;
    return oss.str();
}

/// @brief Mean of the first of @p keys with a finite mean in @p r.
/// @param[in] r     a result
/// @param[in] keys  metric names in order of preference (aliases of one quantity)
/// @return the mean, or NaN when none is finite
inline double metric_mean_or_nan(const Result& r,
                                 const std::initializer_list<const char*>& keys) {
    for (const char* k : keys) {
        auto it = r.summary.find(k);
        if (it != r.summary.end() && std::isfinite(it->second.mean)) {
            return it->second.mean;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

struct SummaryMetricColumns {
    bool show_r = false;
    bool show_o = false;
    bool show_relerr = false;
    bool show_iterations_done = false;
    bool show_total_time_sec = false;
    bool show_final_best_max = false;
    bool show_converged_fraction = false;
    bool show_fill_ratio = false;
    bool show_iter_ratio = false;
    bool show_time_ratio = false;
    bool show_residual_ratio = false;
};

/// @brief Whether @p r has a finite mean for one of @p keys.
/// @param[in] r     a result
/// @param[in] keys  metric names (aliases of one quantity)
/// @return true if metric_mean_or_nan() is finite
inline bool has_any_metric(const Result& r,
                           const std::initializer_list<const char*>& keys) {
    return std::isfinite(metric_mean_or_nan(r, keys));
}

/// @brief Chooses the optional summary columns: those with a finite mean in at least one result.
/// @param[in] results  all results of the run
/// @return the column flags
inline SummaryMetricColumns detect_summary_columns(const std::vector<Result>& results) {
    SummaryMetricColumns cols;
    for (const auto& r : results) {
        cols.show_r = cols.show_r || has_any_metric(r, {"R"});
        cols.show_o = cols.show_o || has_any_metric(r, {"O", "orthogonality"});
        cols.show_relerr = cols.show_relerr || has_any_metric(r, {"max_relerr", "relerr"});
        cols.show_iterations_done = cols.show_iterations_done || has_any_metric(r, {"iterations_done"});
        cols.show_total_time_sec = cols.show_total_time_sec || has_any_metric(r, {"total_time_sec", "solve_time_sec"});
        cols.show_final_best_max = cols.show_final_best_max || has_any_metric(r, {"final_best_max", "final_best_mean"});
        cols.show_converged_fraction = cols.show_converged_fraction || has_any_metric(r, {"converged_fraction"});
        cols.show_fill_ratio = cols.show_fill_ratio || has_any_metric(r, {"fill_ratio"});
        cols.show_iter_ratio = cols.show_iter_ratio || has_any_metric(r, {"iter_ratio_vs_baseline"});
        cols.show_time_ratio = cols.show_time_ratio || has_any_metric(r, {"time_ratio_vs_baseline"});
        cols.show_residual_ratio = cols.show_residual_ratio || has_any_metric(r, {"residual_ratio_vs_baseline"});
    }
    return cols;
}

/// @brief Prints the summary table's header line to `std::cout`.
/// @param[in] name_width  width of the name column
/// @param[in] max_args    number of argument columns
/// @param[in] cols        optional columns to print
inline void print_summary_header(size_t name_width,
                                 size_t max_args,
                                 const SummaryMetricColumns& cols) {
    std::cout << kColorHeader
              << std::left << std::setw(static_cast<int>(name_width)) << "Name";
    for (size_t i = 0; i < max_args; ++i) {
        std::ostringstream col;
        col << "Arg" << i;
        std::cout << std::setw(10) << col.str();
    }
    std::cout << std::setw(12) << "log10cond"
              << std::setw(10) << "Samples"
              << std::setw(10) << "Fail%";
    if (cols.show_r) std::cout << std::setw(28) << "||AZ - Z\\Lambda||_F / n";
    if (cols.show_o) std::cout << std::setw(24) << "||Z^T Z - I||_F / n";
    if (cols.show_relerr) std::cout << std::setw(16) << "max_relerr";
    if (cols.show_iterations_done) std::cout << std::setw(14) << "iters";
    if (cols.show_total_time_sec) std::cout << std::setw(14) << "time_s";
    if (cols.show_final_best_max) std::cout << std::setw(14) << "final_res";
    if (cols.show_converged_fraction) std::cout << std::setw(14) << "conv_frac";
    if (cols.show_fill_ratio) std::cout << std::setw(14) << "fill_ratio";
    if (cols.show_iter_ratio) std::cout << std::setw(14) << "iter_ratio";
    if (cols.show_time_ratio) std::cout << std::setw(14) << "time_ratio";
    if (cols.show_residual_ratio) std::cout << std::setw(14) << "resid_ratio";
    std::cout << kColorReset << '\n';
}

/// @brief Prints one result as a summary table row to `std::cout`.
/// @param[in] r           the result
/// @param[in] name_width  width of the name column
/// @param[in] max_args    number of argument columns; missing arguments print empty
/// @param[in] cols        optional columns to print, as for print_summary_header()
inline void print_summary_row(const Result& r,
                              size_t name_width,
                              size_t max_args,
                              const SummaryMetricColumns& cols) {
    const auto arg_at = [&](size_t i) -> double {
        return i < r.args.size() ? r.args[i] : std::numeric_limits<double>::quiet_NaN();
    };

    const double avg_log10_cond = metric_mean_or_nan(r, {"log10_cond"});
    const double mean_r = metric_mean_or_nan(r, {"R"});
    const double mean_o = metric_mean_or_nan(r, {"O", "orthogonality"});
    const double mean_relerr = metric_mean_or_nan(r, {"max_relerr", "relerr"});
    const double mean_iterations_done = metric_mean_or_nan(r, {"iterations_done"});
    const double mean_total_time_sec = metric_mean_or_nan(r, {"total_time_sec", "solve_time_sec"});
    const double mean_final_best_max = metric_mean_or_nan(r, {"final_best_max", "final_best_mean"});
    const double mean_converged_fraction = metric_mean_or_nan(r, {"converged_fraction"});
    const double mean_fill_ratio = metric_mean_or_nan(r, {"fill_ratio"});
    const double mean_iter_ratio = metric_mean_or_nan(r, {"iter_ratio_vs_baseline"});
    const double mean_time_ratio = metric_mean_or_nan(r, {"time_ratio_vs_baseline"});
    const double mean_residual_ratio = metric_mean_or_nan(r, {"residual_ratio_vs_baseline"});
    const double fail_pct = r.sample_count > 0 ? 100.0 * r.failure_rate : 0.0;

    std::cout << kColorName << std::left << std::setw(static_cast<int>(name_width)) << r.name << kColorReset
              << std::left;
    for (size_t i = 0; i < max_args; ++i) {
        std::cout << std::setw(10) << format_value(arg_at(i));
    }
    std::cout << std::setw(12) << format_value(std::isfinite(avg_log10_cond) ? avg_log10_cond : r.target_log10_cond)
              << std::setw(10) << r.sample_count
              << std::setw(10) << format_value(fail_pct);
    if (cols.show_r) std::cout << std::setw(28) << format_value(mean_r);
    if (cols.show_o) std::cout << std::setw(24) << format_value(mean_o);
    if (cols.show_relerr) std::cout << std::setw(16) << format_value(mean_relerr);
    if (cols.show_iterations_done) std::cout << std::setw(14) << format_value(mean_iterations_done);
    if (cols.show_total_time_sec) std::cout << std::setw(14) << format_value(mean_total_time_sec);
    if (cols.show_final_best_max) std::cout << std::setw(14) << format_value(mean_final_best_max);
    if (cols.show_converged_fraction) std::cout << std::setw(14) << format_value(mean_converged_fraction);
    if (cols.show_fill_ratio) std::cout << std::setw(14) << format_value(mean_fill_ratio);
    if (cols.show_iter_ratio) std::cout << std::setw(14) << format_value(mean_iter_ratio);
    if (cols.show_time_ratio) std::cout << std::setw(14) << format_value(mean_time_ratio);
    if (cols.show_residual_ratio) std::cout << std::setw(14) << format_value(mean_residual_ratio);
    std::cout << '\n';
}

/// @brief Writes one CSV row per kept sample: name, arguments, target, status, tags, metrics.
///
/// On an open failure prints a message to `std::cerr` and writes nothing.
/// @param[in] path     output file; overwritten
/// @param[in] results  results run with `keep_samples`; only their Result::sample_rows are written
inline void write_samples_csv(const std::string& path,
                              const std::vector<Result>& results) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open CSV file: " << path << '\n';
        return;
    }

    size_t max_args = 0;
    std::vector<std::string> metric_names;
    std::unordered_map<std::string, bool> metric_seen;
    std::vector<std::string> tag_names;
    std::unordered_map<std::string, bool> tag_seen;

    for (const auto& r : results) {
        max_args = std::max(max_args, r.args.size());
        for (const auto& row : r.sample_rows) {
            for (const auto& kv : row.metrics) {
                if (!metric_seen[kv.first]) {
                    metric_names.push_back(kv.first);
                    metric_seen[kv.first] = true;
                }
            }
            for (const auto& kv : row.tags) {
                if (!tag_seen[kv.first]) {
                    tag_names.push_back(kv.first);
                    tag_seen[kv.first] = true;
                }
            }
        }
    }

    f << "name";
    for (size_t i = 0; i < max_args; ++i) f << ",arg" << i;
    f << ",target_log10_cond,sample,ok,info_code,failure_reason";
    for (const auto& t : tag_names) f << ",tag_" << t;
    for (const auto& m : metric_names) f << ',' << m;
    f << '\n';

    for (const auto& r : results) {
        for (const auto& row : r.sample_rows) {
            f << '"' << r.name << '"';
            for (size_t i = 0; i < max_args; ++i) {
                if (i < r.args.size()) f << ',' << r.args[i];
                else f << ',';
            }
            f << ',' << r.target_log10_cond
              << ',' << row.sample_index
              << ',' << (row.ok ? 1 : 0)
              << ',' << row.info_code
              << ',' << '"' << row.failure_reason << '"';

            for (const auto& t : tag_names) {
                auto it = row.tags.find(t);
                if (it != row.tags.end()) f << ',' << '"' << it->second << '"';
                else f << ',';
            }

            for (const auto& m : metric_names) {
                auto it = row.metrics.find(m);
                if (it != row.metrics.end()) f << ',' << it->second;
                else f << ',';
            }
            f << '\n';
        }
    }
}

/// @brief Runs every matching case on each argument list and condition target, prints the summary, writes the CSV.
/// @param[in] cfg               run-wide settings; an empty Config::log10_cond_values runs one NaN target
/// @param[in] csv_path          per-sample CSV file; empty writes none and drops the samples
/// @param[in] benchmark_filter  glob on case names, see match_filter()
/// @param[in] backends          backend filter, see match_filter()
/// @param[in] types             scalar-type filter, see match_filter()
/// @param[in] args_override     argument lists to run instead of each case's own; empty keeps them
/// @return 0
inline int RunRegisteredBenchmarks(const Config& cfg,
                                   const std::string& csv_path,
                                   const std::string& benchmark_filter,
                                   const std::vector<std::string>& backends,
                                   const std::vector<std::string>& types,
                                   const std::vector<std::vector<double>>& args_override) {
    std::vector<Result> results;

    std::vector<double> cond_targets = cfg.log10_cond_values;
    if (cond_targets.empty()) {
        cond_targets.push_back(std::numeric_limits<double>::quiet_NaN());
    }

    const bool keep_samples = !csv_path.empty();
    size_t max_args = 0;

    for (auto& b : registry()) {
        if (!match_filter(b, benchmark_filter, backends, types)) continue;

        const auto* active_args = &b.args_list;
        if (!args_override.empty()) active_args = &args_override;

        const bool has_args = !active_args->empty();
        const std::vector<std::vector<double>> empty_case{{}};
        const auto& to_run = has_args ? *active_args : empty_case;

        for (const auto& args : to_run) {
            for (double log10_cond : cond_targets) {
                Result r = run_benchmark(b, args, cfg, log10_cond, keep_samples);
                max_args = std::max(max_args, r.args.size());
                results.push_back(std::move(r));
            }
        }
    }

    const SummaryMetricColumns cols = detect_summary_columns(results);
    size_t name_width = 50;
    for (const auto& r : results) {
        name_width = std::max(name_width, r.name.size() + 2);
    }

    print_summary_header(name_width, max_args, cols);
    for (const auto& r : results) {
        print_summary_row(r, name_width, max_args, cols);
    }

    if (!csv_path.empty()) {
        write_samples_csv(csv_path, results);
    }

    return 0;
}

struct CliOptions {
    Config cfg;
    std::vector<std::vector<double>> args_list;
    std::string csv_file;
    std::string benchmark_filter;
    std::vector<std::string> backends;
    std::vector<std::string> types;

    /// `--help` was seen: usage has been printed and there is nothing to run.
    /// MiniAccMain() turns it into exit status 0.
    // A flag, never std::exit: this installed header must not terminate its host process.
    bool help_requested = false;
};

/// @brief Parses the miniacc command line; positional arguments combine as a Cartesian product.
///
/// On `--help` / `-h` prints usage to `std::cout` and returns with CliOptions::help_requested set.
/// @param[in] argc  argument count, as passed to `main`
/// @param[in] argv  argument vector, as passed to `main`; `argv[0]` is skipped
/// @return the parsed options
/// @throws std::invalid_argument or std::out_of_range on a value that is not a representable number (from `std::stod` / `std::stoul`)
inline CliOptions ParseCommandLine(int argc, char** argv) {
    CliOptions opt;
    std::vector<std::vector<double>> arg_values;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s.rfind("--samples=", 0) == 0) {
            opt.cfg.samples = std::stoul(s.substr(10));
        } else if (s.rfind("--seed=", 0) == 0) {
            opt.cfg.seed = static_cast<unsigned int>(std::stoul(s.substr(7)));
        } else if (s.rfind("--log10-cond=", 0) == 0) {
            opt.cfg.log10_cond_values = dedup_nearly_equal(parse_range_or_list_double(s.substr(13)));
        } else if (s.rfind("--csv=", 0) == 0) {
            opt.csv_file = s.substr(6);
        } else if (s.rfind("--benchmark_filter=", 0) == 0) {
            opt.benchmark_filter = s.substr(19);
        } else if (s.rfind("--backend=", 0) == 0) {
            opt.backends = parse_list(s.substr(10));
        } else if (s.rfind("--type=", 0) == 0) {
            opt.types = parse_list(s.substr(7));
        } else if (s == "--help" || s == "-h") {
            std::cout << "Usage: miniacc [options] [ARGS...]\n";
            std::cout << "Options:\n";
            std::cout << "  --samples=N          number of samples available to each case (default 1024)\n";
            std::cout << "  --seed=N             base RNG seed (default 1234)\n";
            std::cout << "  --log10-cond=SPEC    scalar, v1,v2,... or start:end:num\n";
            std::cout << "  --csv=FILE           emit per-sample rows to CSV (summary always to terminal)\n";
            std::cout << "  --benchmark_filter=GLOB  benchmark-name filter with * and ? wildcards\n";
            std::cout << "  --backend=LIST       comma separated backends to run\n";
            std::cout << "  --type=LIST          comma separated floating point types\n";
            std::cout << "  ARGS can be scalars, comma lists or start:end:num ranges\n";
            // Return, do not exit; MiniAccMain turns the flag into exit status 0.
            opt.help_requested = true;
            return opt;
        } else {
            arg_values.push_back(parse_range_or_list_double(s));
        }
    }

    if (!arg_values.empty()) {
        std::vector<std::vector<double>> combos{{}};
        for (const auto& vals : arg_values) {
            std::vector<std::vector<double>> next;
            for (const auto& c : combos) {
                for (double v : vals) {
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

/// @brief Parses the command line and runs every registered case that matches; the body of MINI_ACC_MAIN().
/// @param[in] argc  argument count, as passed to `main`
/// @param[in] argv  argument vector, as passed to `main`
/// @return 0, including after `--help`
/// @ingroup api_internal_helpers
inline int MiniAccMain(int argc, char** argv) {
    const CliOptions opts = ParseCommandLine(argc, argv);
    // The one place that decides what --help means for the PROCESS.
    if (opts.help_requested) return 0;
    return RunRegisteredBenchmarks(opts.cfg,
                                   opts.csv_file,
                                   opts.benchmark_filter,
                                   opts.backends,
                                   opts.types,
                                   opts.args_list);
}

#define MINI_ACC_MAIN()                        \
    int main(int argc, char** argv) {          \
        return miniacc::MiniAccMain(argc, argv); \
    }

} // namespace miniacc
