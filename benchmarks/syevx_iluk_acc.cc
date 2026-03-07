#include <batchlas/backend_config.h>
#include <blas/linalg.hh>
#include <util/miniacc.hh>

#include "acc_utils.hh"
#include "miniacc_accuracy_common.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename Real>
struct CliMetricSample {
    double iterations_done = std::numeric_limits<double>::quiet_NaN();
    double solve_time_sec = std::numeric_limits<double>::quiet_NaN();
    double total_time_sec = std::numeric_limits<double>::quiet_NaN();
    double final_best_max = std::numeric_limits<double>::quiet_NaN();
    double converged_fraction = std::numeric_limits<double>::quiet_NaN();
    double fill_ratio = std::numeric_limits<double>::quiet_NaN();
    double factor_time_sec = std::numeric_limits<double>::quiet_NaN();
    double iter_ratio_vs_baseline = std::numeric_limits<double>::quiet_NaN();
    double time_ratio_vs_baseline = std::numeric_limits<double>::quiet_NaN();
    double residual_ratio_vs_baseline = std::numeric_limits<double>::quiet_NaN();
    double converged_fraction_delta = std::numeric_limits<double>::quiet_NaN();
};

struct CliConfigKey {
    double density = 0.0;
    double diagonal_boost = 0.0;
    double fill_factor = 0.0;
    double drop_tolerance = 0.0;

    bool operator<(const CliConfigKey& other) const {
        return std::tie(density, diagonal_boost, fill_factor, drop_tolerance)
            < std::tie(other.density, other.diagonal_boost, other.fill_factor, other.drop_tolerance);
    }
};

struct CliConfigSummary {
    std::vector<double> iter_ratio_vs_baseline;
    std::vector<double> time_ratio_vs_baseline;
    std::vector<double> residual_ratio_vs_baseline;
};

struct CliImplSummary {
    std::vector<double> iterations_done;
    std::vector<double> solve_time_sec;
    std::vector<double> total_time_sec;
    std::vector<double> final_best_max;
    std::vector<double> converged_fraction;
    std::vector<double> fill_ratio;
    std::vector<double> factor_time_sec;
    std::vector<double> iter_ratio_vs_baseline;
    std::vector<double> time_ratio_vs_baseline;
    std::vector<double> residual_ratio_vs_baseline;
    std::vector<double> converged_fraction_delta;
    std::size_t success_count = 0;
    std::size_t failure_count = 0;
    std::size_t iter_win_count = 0;
    std::size_t time_win_count = 0;
    std::size_t residual_win_count = 0;
    std::map<CliConfigKey, CliConfigSummary> config_summaries;
};

inline void push_if_finite(std::vector<double>& values, double value) {
    if (std::isfinite(value)) {
        values.push_back(value);
    }
}

inline double median_of(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](double value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    const double upper = *mid;
    if ((values.size() & 1U) != 0U) {
        return upper;
    }
    const auto lower_mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2 - 1);
    std::nth_element(values.begin(), lower_mid, values.end());
    return 0.5 * (*lower_mid + upper);
}

inline double mean_of(const std::vector<double>& values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (double value : values) {
        if (!std::isfinite(value)) {
            continue;
        }
        sum += value;
        ++count;
    }
    return count == 0 ? std::numeric_limits<double>::quiet_NaN() : (sum / static_cast<double>(count));
}

inline std::string format_number(double value, int precision = 4) {
    if (!std::isfinite(value)) {
        return "n/a";
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

inline std::string format_percent(double value) {
    if (!std::isfinite(value)) {
        return "n/a";
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << (100.0 * value) << '%';
    return os.str();
}

inline std::string impl_label(const std::string& impl_name) {
    if (impl_name == "syevx_sparse_baseline") return "baseline";
    if (impl_name == "syevx_sparse_iluk_k2") return "iluk_k2";
    if (impl_name == "syevx_sparse_iluk_k3") return "iluk_k3";
    if (impl_name == "syevx_sparse_iluk_k4") return "iluk_k4";
    return impl_name;
}

class CliSummary {
public:
    static CliSummary& instance() {
        static CliSummary summary;
        return summary;
    }

    template <typename Real>
    void record_success(const std::string& impl_name,
                        double density,
                        double diagonal_boost,
                        double fill_factor,
                        double drop_tolerance,
                        const CliMetricSample<Real>& sample) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& impl = impl_summaries_[impl_name];
        ++impl.success_count;
        push_if_finite(impl.iterations_done, sample.iterations_done);
        push_if_finite(impl.solve_time_sec, sample.solve_time_sec);
        push_if_finite(impl.total_time_sec, sample.total_time_sec);
        push_if_finite(impl.final_best_max, sample.final_best_max);
        push_if_finite(impl.converged_fraction, sample.converged_fraction);
        push_if_finite(impl.fill_ratio, sample.fill_ratio);
        push_if_finite(impl.factor_time_sec, sample.factor_time_sec);
        push_if_finite(impl.iter_ratio_vs_baseline, sample.iter_ratio_vs_baseline);
        push_if_finite(impl.time_ratio_vs_baseline, sample.time_ratio_vs_baseline);
        push_if_finite(impl.residual_ratio_vs_baseline, sample.residual_ratio_vs_baseline);
        push_if_finite(impl.converged_fraction_delta, sample.converged_fraction_delta);

        if (std::isfinite(sample.iter_ratio_vs_baseline) && sample.iter_ratio_vs_baseline < 1.0) ++impl.iter_win_count;
        if (std::isfinite(sample.time_ratio_vs_baseline) && sample.time_ratio_vs_baseline < 1.0) ++impl.time_win_count;
        if (std::isfinite(sample.residual_ratio_vs_baseline) && sample.residual_ratio_vs_baseline < 1.0) ++impl.residual_win_count;

        if (std::isfinite(sample.iter_ratio_vs_baseline) || std::isfinite(sample.time_ratio_vs_baseline) || std::isfinite(sample.residual_ratio_vs_baseline)) {
            CliConfigKey key{density, diagonal_boost, fill_factor, drop_tolerance};
            auto& config = impl.config_summaries[key];
            push_if_finite(config.iter_ratio_vs_baseline, sample.iter_ratio_vs_baseline);
            push_if_finite(config.time_ratio_vs_baseline, sample.time_ratio_vs_baseline);
            push_if_finite(config.residual_ratio_vs_baseline, sample.residual_ratio_vs_baseline);
        }
    }

    void record_failure(const std::string& impl_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++impl_summaries_[impl_name].failure_count;
    }

    void print() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (printed_ || impl_summaries_.empty()) {
            return;
        }
        printed_ = true;

        std::cout << "\n[SYEVX ILUK summary]\n";

        const auto baseline_it = impl_summaries_.find("syevx_sparse_baseline");
        if (baseline_it != impl_summaries_.end()) {
            const auto& baseline = baseline_it->second;
            std::cout << "  baseline: samples=" << baseline.success_count
                      << ", failed=" << baseline.failure_count
                      << ", median_iter=" << format_number(median_of(baseline.iterations_done), 2)
                      << ", median_solve_s=" << format_number(median_of(baseline.solve_time_sec), 6)
                      << ", median_final_res=" << format_number(median_of(baseline.final_best_max), 6)
                      << ", mean_conv_frac=" << format_number(mean_of(baseline.converged_fraction), 3)
                      << "\n";
        }

        for (const auto& [impl_name, summary] : impl_summaries_) {
            if (impl_name == "syevx_sparse_baseline") {
                continue;
            }

            const double iter_trials = static_cast<double>(summary.iter_ratio_vs_baseline.size());
            const double time_trials = static_cast<double>(summary.time_ratio_vs_baseline.size());
            const double residual_trials = static_cast<double>(summary.residual_ratio_vs_baseline.size());

            std::cout << "  " << impl_label(impl_name)
                      << ": samples=" << summary.success_count
                      << ", failed=" << summary.failure_count
                      << ", median_iter_ratio=" << format_number(median_of(summary.iter_ratio_vs_baseline), 3)
                      << " (wins=" << format_percent(iter_trials > 0.0 ? static_cast<double>(summary.iter_win_count) / iter_trials : std::numeric_limits<double>::quiet_NaN()) << ")"
                      << ", median_time_ratio=" << format_number(median_of(summary.time_ratio_vs_baseline), 3)
                      << " (wins=" << format_percent(time_trials > 0.0 ? static_cast<double>(summary.time_win_count) / time_trials : std::numeric_limits<double>::quiet_NaN()) << ")"
                      << ", median_residual_ratio=" << format_number(median_of(summary.residual_ratio_vs_baseline), 3)
                      << " (wins=" << format_percent(residual_trials > 0.0 ? static_cast<double>(summary.residual_win_count) / residual_trials : std::numeric_limits<double>::quiet_NaN()) << ")"
                      << ", mean_conv_delta=" << format_number(mean_of(summary.converged_fraction_delta), 3)
                      << ", median_fill_ratio=" << format_number(median_of(summary.fill_ratio), 3)
                      << ", median_factor_s=" << format_number(median_of(summary.factor_time_sec), 6)
                      << "\n";

            if (!summary.config_summaries.empty()) {
                const CliConfigKey* best_residual_key = nullptr;
                const CliConfigKey* worst_time_key = nullptr;
                double best_residual_value = std::numeric_limits<double>::infinity();
                double worst_time_value = -std::numeric_limits<double>::infinity();

                for (const auto& [key, config] : summary.config_summaries) {
                    const double residual_med = median_of(config.residual_ratio_vs_baseline);
                    if (std::isfinite(residual_med) && residual_med < best_residual_value) {
                        best_residual_value = residual_med;
                        best_residual_key = &key;
                    }
                    const double time_med = median_of(config.time_ratio_vs_baseline);
                    if (std::isfinite(time_med) && time_med > worst_time_value) {
                        worst_time_value = time_med;
                        worst_time_key = &key;
                    }
                }

                if (best_residual_key != nullptr) {
                    std::cout << "    best_residual_cfg: density=" << format_number(best_residual_key->density, 3)
                              << ", diag_boost=" << format_number(best_residual_key->diagonal_boost, 2)
                              << ", fill_factor=" << format_number(best_residual_key->fill_factor, 2)
                              << ", drop_tol=" << format_number(best_residual_key->drop_tolerance, 5)
                              << ", median_residual_ratio=" << format_number(best_residual_value, 3)
                              << "\n";
                }
                if (worst_time_key != nullptr) {
                    std::cout << "    worst_time_cfg: density=" << format_number(worst_time_key->density, 3)
                              << ", diag_boost=" << format_number(worst_time_key->diagonal_boost, 2)
                              << ", fill_factor=" << format_number(worst_time_key->fill_factor, 2)
                              << ", drop_tol=" << format_number(worst_time_key->drop_tolerance, 5)
                              << ", median_time_ratio=" << format_number(worst_time_value, 3)
                              << "\n";
                }
            }
        }
    }

private:
    std::mutex mutex_;
    std::map<std::string, CliImplSummary> impl_summaries_;
    bool printed_ = false;
};

struct CliSummaryPrinter {
    ~CliSummaryPrinter() {
        CliSummary::instance().print();
    }
};

CliSummaryPrinter g_cli_summary_printer;

template <typename Benchmark>
void SparseSyevxIlukSizes(Benchmark* b) {
    for (double n : {128.0, 256.0}) {
        for (double density_percent : {2.0, 5.0}) {
            for (double diagonal_boost_x10 : {5.0, 20.0, 40.0}) {
                for (double fill_factor_x10 : {10.0, 20.0, 40.0}) {
                    for (double drop_tol_neg_log10 : {2.0, 4.0}) {
                        b->Args({n, density_percent, diagonal_boost_x10, fill_factor_x10, drop_tol_neg_log10});
                    }
                }
            }
        }
    }
}

template <typename Benchmark>
void SparseSyevxIlukSizesNetlib(Benchmark* b) {
    SparseSyevxIlukSizes(b);
}

template <typename Real>
struct SparseRunBuffers {
    UnifiedVector<Real> eigvals;
    UnifiedVector<Real> best_hist;
    UnifiedVector<Real> current_hist;
    UnifiedVector<Real> rate_hist;
    UnifiedVector<Real> ritz_hist;
    UnifiedVector<int32_t> iters_done;

    SparseRunBuffers(int neigs, int batch, int max_iters)
        : eigvals(static_cast<std::size_t>(neigs * batch), Real(0)),
          best_hist(static_cast<std::size_t>(neigs * batch * max_iters), std::numeric_limits<Real>::quiet_NaN()),
          current_hist(static_cast<std::size_t>(neigs * batch * max_iters), std::numeric_limits<Real>::quiet_NaN()),
          rate_hist(static_cast<std::size_t>(neigs * batch * max_iters), std::numeric_limits<Real>::quiet_NaN()),
          ritz_hist(static_cast<std::size_t>(neigs * batch * max_iters), std::numeric_limits<Real>::quiet_NaN()),
          iters_done(static_cast<std::size_t>(batch), 0) {}
};

template <typename Real>
struct MatrixMetrics {
    double iterations_done = std::numeric_limits<double>::quiet_NaN();
    double final_best_mean = std::numeric_limits<double>::quiet_NaN();
    double final_best_max = std::numeric_limits<double>::quiet_NaN();
    double avg_rate_mean = std::numeric_limits<double>::quiet_NaN();
    double converged_fraction = std::numeric_limits<double>::quiet_NaN();
    double first_tol_iter_mean = std::numeric_limits<double>::quiet_NaN();
    double solve_time_sec = std::numeric_limits<double>::quiet_NaN();
    bool converged_all = false;
};

template <typename Real>
std::vector<MatrixMetrics<Real>> summarize_sparse_run(const SparseRunBuffers<Real>& buffers,
                                                      int neigs,
                                                      int batch,
                                                      int max_iters,
                                                      Real tolerance,
                                                      double solve_time_sec) {
    std::vector<MatrixMetrics<Real>> metrics(static_cast<std::size_t>(batch));
    for (int b = 0; b < batch; ++b) {
        auto& out = metrics[static_cast<std::size_t>(b)];
        out.iterations_done = static_cast<double>(buffers.iters_done[static_cast<std::size_t>(b)]);
        out.solve_time_sec = solve_time_sec / static_cast<double>(std::max(batch, 1));

        double best_sum = 0.0;
        double rate_sum = 0.0;
        double first_tol_sum = 0.0;
        int best_count = 0;
        int rate_count = 0;
        int first_tol_count = 0;
        int converged_count = 0;
        double best_max = 0.0;

        for (int eig = 0; eig < neigs; ++eig) {
            Real final_best = std::numeric_limits<Real>::quiet_NaN();
            double eig_rate_sum = 0.0;
            int eig_rate_count = 0;
            int first_tol_iter = -1;

            for (int it = 0; it < max_iters; ++it) {
                const std::size_t idx = static_cast<std::size_t>(it) * static_cast<std::size_t>(batch * neigs)
                                      + static_cast<std::size_t>(b * neigs + eig);
                const Real best = buffers.best_hist[idx];
                const Real rate = buffers.rate_hist[idx];
                if (std::isfinite(best)) {
                    final_best = best;
                    if (first_tol_iter < 0 && best <= tolerance) {
                        first_tol_iter = it;
                    }
                }
                if (std::isfinite(rate) && rate > Real(0)) {
                    eig_rate_sum += static_cast<double>(rate);
                    ++eig_rate_count;
                }
            }

            if (std::isfinite(final_best)) {
                best_sum += static_cast<double>(final_best);
                best_max = std::max(best_max, static_cast<double>(final_best));
                ++best_count;
                if (final_best <= tolerance) {
                    ++converged_count;
                }
            }
            if (eig_rate_count > 0) {
                rate_sum += eig_rate_sum / static_cast<double>(eig_rate_count);
                ++rate_count;
            }
            if (first_tol_iter >= 0) {
                first_tol_sum += static_cast<double>(first_tol_iter);
                ++first_tol_count;
            }
        }

        if (best_count > 0) {
            out.final_best_mean = best_sum / static_cast<double>(best_count);
            out.final_best_max = best_max;
        }
        if (rate_count > 0) {
            out.avg_rate_mean = rate_sum / static_cast<double>(rate_count);
        }
        if (first_tol_count > 0) {
            out.first_tol_iter_mean = first_tol_sum / static_cast<double>(first_tol_count);
        }
        out.converged_fraction = static_cast<double>(converged_count) / static_cast<double>(std::max(neigs, 1));
        out.converged_all = (converged_count == neigs);
    }
    return metrics;
}

template <Backend B, typename Real>
std::vector<MatrixMetrics<Real>> run_sparse_syevx_once(Queue& q,
                                                        const MatrixView<Real, MatrixFormat::CSR>& csr_view,
                                                        int neigs,
                                                        int extra_dirs,
                                                        int max_iters,
                                                        Real tolerance,
                                                        const ILUKPreconditioner<Real>* preconditioner) {
    SparseRunBuffers<Real> buffers(neigs, csr_view.batch_size(), max_iters);

    SyevxInstrumentation<Real> instrumentation;
    instrumentation.best_residual_history = buffers.best_hist;
    instrumentation.current_residual_history = buffers.current_hist;
    instrumentation.convergence_rate_history = buffers.rate_hist;
    instrumentation.ritz_value_history = buffers.ritz_hist;
    instrumentation.max_iterations = static_cast<std::size_t>(max_iters);
    instrumentation.store_every = 1;
    instrumentation.store_current_residual = true;
    instrumentation.store_convergence_rate = true;
    instrumentation.store_ritz_values = true;
    instrumentation.iterations_done = buffers.iters_done.data();

    SyevxParams<Real> params{};
    params.algorithm = OrthoAlgorithm::ShiftChol3;
    params.iterations = static_cast<std::size_t>(max_iters);
    params.extra_directions = static_cast<std::size_t>(extra_dirs);
    params.find_largest = true;
    params.absolute_tolerance = tolerance;
    params.relative_tolerance = tolerance;
    params.preconditioner = preconditioner;
    params.instrumentation = &instrumentation;

    UnifiedVector<std::byte> ws(
        syevx_buffer_size<B>(q,
                             csr_view,
                             buffers.eigvals.to_span(),
                             static_cast<std::size_t>(neigs),
                             JobType::NoEigenVectors,
                             MatrixView<Real, MatrixFormat::Dense>(),
                             params));

    const auto start = std::chrono::steady_clock::now();
    syevx<B>(q,
             csr_view,
             buffers.eigvals.to_span(),
             static_cast<std::size_t>(neigs),
             ws.to_span(),
             JobType::NoEigenVectors,
             MatrixView<Real, MatrixFormat::Dense>(),
             params);
    q.wait_and_throw();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    return summarize_sparse_run(buffers, neigs, csr_view.batch_size(), max_iters, tolerance, elapsed);
}

template <int Level, typename Real, Backend B>
void run_sparse_syevx_iluk(miniacc::State& state, const char* impl_name) {
    static_assert(Level >= -1, "invalid ILUK level");

    const int n = std::max(16, state.arg_int(0));
    const double density_percent = state.arg(1);
    const double diagonal_boost_x10 = state.arg(2);
    const double fill_factor_x10 = state.arg(3);
    const double drop_tol_neg_log10 = state.arg(4);
    const Real density = static_cast<Real>(density_percent / 100.0);
    const Real diagonal_boost = static_cast<Real>(diagonal_boost_x10 / 10.0);
    const Real fill_factor = static_cast<Real>(std::max(1.0, fill_factor_x10 / 10.0));
    const Real drop_tolerance = static_cast<Real>(std::pow(10.0, -drop_tol_neg_log10));
    const int neigs = std::min(5, std::max(1, n / 16));
    const int extra_dirs = neigs;
    const int max_iters = 30;
    const Real tolerance = static_cast<Real>(1e-6);
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples(), 16);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetTag("impl", impl_name);
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());
    state.SetTag("density", std::to_string(static_cast<double>(density)));
    state.SetTag("diag_boost", std::to_string(static_cast<double>(diagonal_boost)));
    state.SetTag("fill_factor", std::to_string(static_cast<double>(fill_factor)));
    state.SetTag("drop_tolerance", std::to_string(static_cast<double>(drop_tolerance)));
    state.SetTag("neigs", std::to_string(neigs));
    state.SetTag("max_iters", std::to_string(max_iters));
    state.SetTag("iluk_level", Level >= 0 ? std::to_string(Level) : std::string("none"));

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto csr = csr_generators::random_sparse_hermitian_csr<Real>(
            n,
            static_cast<float>(density),
            cur_batch,
            seed,
            diagonal_boost,
            true);

        std::vector<MatrixMetrics<Real>> baseline_metrics;
        try {
            baseline_metrics = run_sparse_syevx_once<B>(*q,
                                                        csr.view(),
                                                        neigs,
                                                        extra_dirs,
                                                        max_iters,
                                                        tolerance,
                                                        static_cast<const ILUKPreconditioner<Real>*>(nullptr));
        } catch (const std::exception& ex) {
            CliSummary::instance().record_failure(impl_name);
            miniacc_acc::record_failed_samples(state,
                                               n,
                                               neigs,
                                               cur_batch,
                                               std::numeric_limits<double>::quiet_NaN(),
                                               std::string("baseline_exception:") + ex.what());
            produced += static_cast<std::size_t>(cur_batch);
            continue;
        }

        if constexpr (Level < 0) {
            for (int b = 0; b < cur_batch; ++b) {
                const auto& metric = baseline_metrics[static_cast<std::size_t>(b)];
                CliMetricSample<Real> cli_sample;
                cli_sample.iterations_done = metric.iterations_done;
                cli_sample.solve_time_sec = metric.solve_time_sec;
                cli_sample.total_time_sec = metric.solve_time_sec;
                cli_sample.final_best_max = metric.final_best_max;
                cli_sample.converged_fraction = metric.converged_fraction;
                CliSummary::instance().record_success(impl_name,
                                                      static_cast<double>(density),
                                                      static_cast<double>(diagonal_boost),
                                                      static_cast<double>(fill_factor),
                                                      static_cast<double>(drop_tolerance),
                                                      cli_sample);
                state.RecordSample({
                    {"density", static_cast<double>(density)},
                    {"diag_boost", static_cast<double>(diagonal_boost)},
                    {"fill_factor", static_cast<double>(fill_factor)},
                    {"drop_tolerance", static_cast<double>(drop_tolerance)},
                    {"nnz", static_cast<double>(csr.nnz())},
                    {"iterations_done", metric.iterations_done},
                    {"solve_time_sec", metric.solve_time_sec},
                    {"total_time_sec", metric.solve_time_sec},
                    {"final_best_mean", metric.final_best_mean},
                    {"final_best_max", metric.final_best_max},
                    {"avg_rate_mean", metric.avg_rate_mean},
                    {"converged_fraction", metric.converged_fraction},
                    {"first_tol_iter_mean", metric.first_tol_iter_mean},
                    {"converged_all", metric.converged_all ? 1.0 : 0.0}
                });
            }
            produced += static_cast<std::size_t>(cur_batch);
            continue;
        }

        ILUKParams<Real> iluk_params{};
        iluk_params.levels_of_fill = Level;
        iluk_params.fill_factor = fill_factor;
        iluk_params.drop_tolerance = drop_tolerance;

        std::optional<ILUKPreconditioner<Real>> preconditioner;
        double factor_time_sec = std::numeric_limits<double>::quiet_NaN();
        double fill_ratio = std::numeric_limits<double>::quiet_NaN();
        double precond_nnz = std::numeric_limits<double>::quiet_NaN();

        try {
            const auto factor_start = std::chrono::steady_clock::now();
            preconditioner = iluk_factorize<B>(*q, csr.view(), iluk_params);
            q->wait_and_throw();
            factor_time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - factor_start).count();
            precond_nnz = static_cast<double>(preconditioner->lu.nnz());
            fill_ratio = precond_nnz / static_cast<double>(std::max(1, csr.nnz()));
        } catch (const std::exception& ex) {
            CliSummary::instance().record_failure(impl_name);
            for (int b = 0; b < cur_batch; ++b) {
                const auto& baseline = baseline_metrics[static_cast<std::size_t>(b)];
                state.RecordSample({
                        {"density", static_cast<double>(density)},
                        {"diag_boost", static_cast<double>(diagonal_boost)},
                    {"fill_factor", static_cast<double>(fill_factor)},
                    {"drop_tolerance", static_cast<double>(drop_tolerance)},
                        {"nnz", static_cast<double>(csr.nnz())},
                        {"baseline_iterations_done", baseline.iterations_done},
                        {"baseline_solve_time_sec", baseline.solve_time_sec},
                        {"baseline_final_best_max", baseline.final_best_max}
                    },
                    false,
                    std::string("factorization_exception:") + ex.what());
            }
            produced += static_cast<std::size_t>(cur_batch);
            continue;
        }

        std::vector<MatrixMetrics<Real>> iluk_metrics;
        try {
            iluk_metrics = run_sparse_syevx_once<B>(*q,
                                                    csr.view(),
                                                    neigs,
                                                    extra_dirs,
                                                    max_iters,
                                                    tolerance,
                                                    &(*preconditioner));
        } catch (const std::exception& ex) {
            CliSummary::instance().record_failure(impl_name);
            for (int b = 0; b < cur_batch; ++b) {
                const auto& baseline = baseline_metrics[static_cast<std::size_t>(b)];
                state.RecordSample({
                        {"density", static_cast<double>(density)},
                        {"diag_boost", static_cast<double>(diagonal_boost)},
                    {"fill_factor", static_cast<double>(fill_factor)},
                    {"drop_tolerance", static_cast<double>(drop_tolerance)},
                        {"nnz", static_cast<double>(csr.nnz())},
                        {"fill_ratio", fill_ratio},
                        {"precond_nnz", precond_nnz},
                        {"factor_time_sec", factor_time_sec / static_cast<double>(cur_batch)},
                        {"baseline_iterations_done", baseline.iterations_done},
                        {"baseline_solve_time_sec", baseline.solve_time_sec},
                        {"baseline_final_best_max", baseline.final_best_max}
                    },
                    false,
                    std::string("solver_exception:") + ex.what());
            }
            produced += static_cast<std::size_t>(cur_batch);
            continue;
        }

        for (int b = 0; b < cur_batch; ++b) {
            const auto& baseline = baseline_metrics[static_cast<std::size_t>(b)];
            const auto& metric = iluk_metrics[static_cast<std::size_t>(b)];
            const double factor_time_per_matrix = factor_time_sec / static_cast<double>(cur_batch);
            const double total_time_sec = factor_time_per_matrix + metric.solve_time_sec;

            const double iter_ratio_vs_baseline = (std::isfinite(baseline.iterations_done) && baseline.iterations_done > 0.0)
                ? (metric.iterations_done / baseline.iterations_done)
                : std::numeric_limits<double>::quiet_NaN();
            const double time_ratio_vs_baseline = (std::isfinite(baseline.solve_time_sec) && baseline.solve_time_sec > 0.0)
                ? (total_time_sec / baseline.solve_time_sec)
                : std::numeric_limits<double>::quiet_NaN();
            const double residual_ratio_vs_baseline = (std::isfinite(baseline.final_best_max) && baseline.final_best_max > 0.0)
                ? (metric.final_best_max / baseline.final_best_max)
                : std::numeric_limits<double>::quiet_NaN();

            CliMetricSample<Real> cli_sample;
            cli_sample.iterations_done = metric.iterations_done;
            cli_sample.solve_time_sec = metric.solve_time_sec;
            cli_sample.total_time_sec = total_time_sec;
            cli_sample.final_best_max = metric.final_best_max;
            cli_sample.converged_fraction = metric.converged_fraction;
            cli_sample.fill_ratio = fill_ratio;
            cli_sample.factor_time_sec = factor_time_per_matrix;
            cli_sample.iter_ratio_vs_baseline = iter_ratio_vs_baseline;
            cli_sample.time_ratio_vs_baseline = time_ratio_vs_baseline;
            cli_sample.residual_ratio_vs_baseline = residual_ratio_vs_baseline;
            cli_sample.converged_fraction_delta = metric.converged_fraction - baseline.converged_fraction;
            CliSummary::instance().record_success(impl_name,
                                                  static_cast<double>(density),
                                                  static_cast<double>(diagonal_boost),
                                                  static_cast<double>(fill_factor),
                                                  static_cast<double>(drop_tolerance),
                                                  cli_sample);

            state.RecordSample({
                {"density", static_cast<double>(density)},
                {"diag_boost", static_cast<double>(diagonal_boost)},
                {"fill_factor", static_cast<double>(fill_factor)},
                {"drop_tolerance", static_cast<double>(drop_tolerance)},
                {"nnz", static_cast<double>(csr.nnz())},
                {"precond_nnz", precond_nnz},
                {"fill_ratio", fill_ratio},
                {"factor_time_sec", factor_time_per_matrix},
                {"solve_time_sec", metric.solve_time_sec},
                {"total_time_sec", total_time_sec},
                {"iterations_done", metric.iterations_done},
                {"final_best_mean", metric.final_best_mean},
                {"final_best_max", metric.final_best_max},
                {"avg_rate_mean", metric.avg_rate_mean},
                {"converged_fraction", metric.converged_fraction},
                {"first_tol_iter_mean", metric.first_tol_iter_mean},
                {"converged_all", metric.converged_all ? 1.0 : 0.0},
                {"baseline_iterations_done", baseline.iterations_done},
                {"baseline_solve_time_sec", baseline.solve_time_sec},
                {"baseline_final_best_max", baseline.final_best_max},
                {"iter_delta_vs_baseline", baseline.iterations_done - metric.iterations_done},
                {"iter_ratio_vs_baseline", iter_ratio_vs_baseline},
                {"time_ratio_vs_baseline", time_ratio_vs_baseline},
                {"residual_ratio_vs_baseline", residual_ratio_vs_baseline},
                {"converged_fraction_delta", metric.converged_fraction - baseline.converged_fraction}
            });
        }

        produced += static_cast<std::size_t>(cur_batch);
    }
}

template <typename Real, Backend B>
static void ACC_SYEVX_SPARSE_BASELINE(miniacc::State& state) {
    run_sparse_syevx_iluk<-1, Real, B>(state, "syevx_sparse_baseline");
}

template <typename Real, Backend B>
static void ACC_SYEVX_SPARSE_ILUK_K2(miniacc::State& state) {
    run_sparse_syevx_iluk<2, Real, B>(state, "syevx_sparse_iluk_k2");
}

template <typename Real, Backend B>
static void ACC_SYEVX_SPARSE_ILUK_K3(miniacc::State& state) {
    run_sparse_syevx_iluk<3, Real, B>(state, "syevx_sparse_iluk_k3");
}

template <typename Real, Backend B>
static void ACC_SYEVX_SPARSE_ILUK_K4(miniacc::State& state) {
    run_sparse_syevx_iluk<4, Real, B>(state, "syevx_sparse_iluk_k4");
}

}  // namespace

BATCHLAS_ACC_CUDA(ACC_SYEVX_SPARSE_BASELINE, SparseSyevxIlukSizes)
BATCHLAS_ACC_ROCM(ACC_SYEVX_SPARSE_BASELINE, SparseSyevxIlukSizes)
BATCHLAS_ACC_CUDA(ACC_SYEVX_SPARSE_ILUK_K2, SparseSyevxIlukSizes)
BATCHLAS_ACC_ROCM(ACC_SYEVX_SPARSE_ILUK_K2, SparseSyevxIlukSizes)
BATCHLAS_ACC_CUDA(ACC_SYEVX_SPARSE_ILUK_K3, SparseSyevxIlukSizes)
BATCHLAS_ACC_ROCM(ACC_SYEVX_SPARSE_ILUK_K3, SparseSyevxIlukSizes)
BATCHLAS_ACC_CUDA(ACC_SYEVX_SPARSE_ILUK_K4, SparseSyevxIlukSizes)
BATCHLAS_ACC_ROCM(ACC_SYEVX_SPARSE_ILUK_K4, SparseSyevxIlukSizes)

MINI_ACC_MAIN()