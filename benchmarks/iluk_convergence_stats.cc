#include <batchlas/backend_config.h>
#include <blas/linalg.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

#if BATCHLAS_HAS_CUDA_BACKEND
constexpr Backend kGpuBackend = Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
constexpr Backend kGpuBackend = Backend::ROCM;
#elif BATCHLAS_HAS_MKL_BACKEND
constexpr Backend kGpuBackend = Backend::MKL;
#endif

struct Options {
    int n = 256;
    int batch = 1;
    int neigs = 5;
    int extra_directions = 5;
    int iterations = 30;
    int samples_per_bucket = 16;
    float tolerance = 1e-6f;
    unsigned seed = 1234u;
    std::string densities = "0.02,0.04,0.06";
    std::string diagonal_boosts = "0.5,2.0,4.0";
    std::string levels = "2,3,4,5";
    std::string output_prefix = "output/accuracy/iluk_convergence_stats";
};

struct RunSpec {
    std::string label;
    int level = -1;
};

struct RunBuffers {
    RunSpec spec;
    UnifiedVector<float> eigvals;
    UnifiedVector<float> best_hist;
    UnifiedVector<float> current_hist;
    UnifiedVector<float> rate_hist;
    UnifiedVector<float> ritz_hist;
    UnifiedVector<int32_t> iters_done;

    RunBuffers(const RunSpec& run_spec, size_t history_size, size_t eigval_size, size_t batch_size)
        : spec(run_spec),
          eigvals(eigval_size, 0.0f),
          best_hist(history_size, std::nanf("")),
          current_hist(history_size, std::nanf("")),
          rate_hist(history_size, std::nanf("")),
          ritz_hist(history_size, std::nanf("")),
          iters_done(batch_size, 0) {}
};

struct EigenpairMetrics {
    float first_best = std::nanf("");
    float final_best = std::nanf("");
    float final_ritz = std::nanf("");
    float avg_rate = std::nanf("");
    float log10_reduction = std::nanf("");
    int first_tol_iter = -1;
    int last_iter = -1;
    int iterations_done = 0;
    int valid_rate_count = 0;
    bool converged = false;
};

struct AggregateStats {
    double final_best_sum = 0.0;
    double ratio_sum = 0.0;
    double log10_ratio_sum = 0.0;
    int eigenpair_count = 0;
    int converged_count = 0;
    int ratio_count = 0;
    int win_count = 0;
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

template <typename T>
std::vector<T> parse_csv_list(const std::string& text) {
    std::vector<T> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        std::stringstream item_stream(item);
        T value{};
        item_stream >> value;
        if (item_stream.fail()) {
            throw std::invalid_argument("Failed to parse list item: " + item);
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::invalid_argument("Expected a non-empty comma-separated list");
    }
    return values;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto get_value = [&](const std::string& token) -> std::string {
            const auto pos = token.find('=');
            if (pos != std::string::npos) return token.substr(pos + 1);
            if (i + 1 < argc) return std::string(argv[++i]);
            return "";
        };

        if (starts_with(arg, "--n")) {
            opt.n = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--batch")) {
            opt.batch = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--neigs")) {
            opt.neigs = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--extra-directions")) {
            opt.extra_directions = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--iterations")) {
            opt.iterations = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--samples-per-bucket")) {
            opt.samples_per_bucket = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--tolerance")) {
            opt.tolerance = std::stof(get_value(arg));
        } else if (starts_with(arg, "--seed")) {
            opt.seed = static_cast<unsigned>(std::stoul(get_value(arg)));
        } else if (starts_with(arg, "--densities")) {
            opt.densities = get_value(arg);
        } else if (starts_with(arg, "--diagonal-boosts")) {
            opt.diagonal_boosts = get_value(arg);
        } else if (starts_with(arg, "--levels")) {
            opt.levels = get_value(arg);
        } else if (starts_with(arg, "--output-prefix")) {
            opt.output_prefix = get_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "ILU(k) convergence statistics benchmark\n\n"
                << "Produces:\n"
                << "  <prefix>_trace.csv   per-iteration per-eigenpair traces\n"
                << "  <prefix>_summary.csv per-eigenpair summary metrics\n\n"
                << "Options:\n"
                << "  --n N\n"
                << "  --batch B\n"
                << "  --neigs N\n"
                << "  --extra-directions N\n"
                << "  --iterations N\n"
                << "  --samples-per-bucket N\n"
                << "  --tolerance X\n"
                << "  --seed SEED\n"
                << "  --densities A,B,C\n"
                << "  --diagonal-boosts A,B,C\n"
                << "  --levels 2,3,4,5\n"
                << "  --output-prefix PATH\n";
            std::exit(0);
        }
    }
    return opt;
}

std::vector<EigenpairMetrics> summarize_run(const RunBuffers& run,
                                            int max_iters,
                                            int neigs,
                                            int batch,
                                            float tolerance) {
    std::vector<EigenpairMetrics> metrics(static_cast<size_t>(batch * neigs));
    for (int b = 0; b < batch; ++b) {
        for (int eig = 0; eig < neigs; ++eig) {
            auto& metric = metrics[static_cast<size_t>(b * neigs + eig)];
            double rate_sum = 0.0;
            int rate_count = 0;
            for (int it = 0; it < max_iters; ++it) {
                const size_t idx = static_cast<size_t>(it) * static_cast<size_t>(batch * neigs)
                                 + static_cast<size_t>(b * neigs + eig);
                const float best = run.best_hist[idx];
                const float rate = run.rate_hist[idx];
                const float ritz = run.ritz_hist[idx];

                if (std::isfinite(best)) {
                    if (!std::isfinite(metric.first_best)) {
                        metric.first_best = best;
                    }
                    metric.final_best = best;
                    metric.final_ritz = ritz;
                    metric.last_iter = it;
                    if (!metric.converged && best <= tolerance) {
                        metric.converged = true;
                        metric.first_tol_iter = it;
                    }
                }
                if (std::isfinite(rate) && rate > 0.0f) {
                    rate_sum += static_cast<double>(rate);
                    ++rate_count;
                }
            }

            metric.iterations_done = run.iters_done[static_cast<size_t>(b)];
            metric.valid_rate_count = rate_count;
            if (rate_count > 0) {
                metric.avg_rate = static_cast<float>(rate_sum / static_cast<double>(rate_count));
            }
            if (std::isfinite(metric.first_best) && std::isfinite(metric.final_best)
                && metric.first_best > 0.0f && metric.final_best > 0.0f) {
                metric.log10_reduction = std::log10(metric.first_best / metric.final_best);
            }
        }
    }
    return metrics;
}

void run_case(Queue& queue,
              const MatrixView<float, MatrixFormat::CSR>& csr_view,
              int neigs,
              int extra_dirs,
              int max_iters,
              const RunSpec& spec,
              RunBuffers& buffers) {
    std::fill(buffers.best_hist.begin(), buffers.best_hist.end(), std::nanf(""));
    std::fill(buffers.current_hist.begin(), buffers.current_hist.end(), std::nanf(""));
    std::fill(buffers.rate_hist.begin(), buffers.rate_hist.end(), std::nanf(""));
    std::fill(buffers.ritz_hist.begin(), buffers.ritz_hist.end(), std::nanf(""));
    std::fill(buffers.iters_done.begin(), buffers.iters_done.end(), 0);
    std::fill(buffers.eigvals.begin(), buffers.eigvals.end(), 0.0f);

    std::optional<ILUKPreconditioner<float>> preconditioner;
    const ILUKPreconditioner<float>* precond_ptr = nullptr;
    if (spec.level >= 0) {
        ILUKParams<float> iluk_params;
        iluk_params.levels_of_fill = spec.level;
        preconditioner = iluk_factorize<kGpuBackend>(queue, csr_view, iluk_params);
        precond_ptr = &(*preconditioner);
    }

    SyevxInstrumentation<float> instrumentation;
    instrumentation.best_residual_history = buffers.best_hist;
    instrumentation.current_residual_history = buffers.current_hist;
    instrumentation.convergence_rate_history = buffers.rate_hist;
    instrumentation.ritz_value_history = buffers.ritz_hist;
    instrumentation.max_iterations = max_iters;
    instrumentation.store_every = 1;
    instrumentation.store_current_residual = true;
    instrumentation.store_convergence_rate = true;
    instrumentation.store_ritz_values = true;
    instrumentation.iterations_done = buffers.iters_done.data();

    SyevxParams<float> params;
    params.iterations = max_iters;
    params.extra_directions = extra_dirs;
    params.find_largest = true;
    params.absolute_tolerance = 1e-6f;
    params.relative_tolerance = 1e-6f;
    params.preconditioner = precond_ptr;
    params.instrumentation = &instrumentation;

    UnifiedVector<std::byte> workspace(
        syevx_buffer_size<kGpuBackend>(queue,
                           csr_view,
                           buffers.eigvals,
                           static_cast<size_t>(neigs),
                           JobType::NoEigenVectors,
                           MatrixView<float, MatrixFormat::Dense>(),
                           params));

        syevx<kGpuBackend>(queue,
                   csr_view,
                   buffers.eigvals,
                   static_cast<size_t>(neigs),
                   workspace,
                   JobType::NoEigenVectors,
                   MatrixView<float, MatrixFormat::Dense>(),
                   params);
    queue.wait_and_throw();
}

std::string bucket_label(float density, float diagonal_boost) {
    std::ostringstream os;
    os << "d" << density << "_b" << diagonal_boost;
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
#if !BATCHLAS_HAS_GPU_BACKEND
    (void)argc;
    (void)argv;
    throw std::runtime_error("iluk_convergence_stats requires a configured GPU backend");
#else
    const Options opt = parse_args(argc, argv);
    if (opt.n <= 0 || opt.batch <= 0 || opt.neigs <= 0 || opt.iterations <= 0 || opt.samples_per_bucket <= 0) {
        throw std::invalid_argument("All size/count options must be positive");
    }

    const auto densities = parse_csv_list<float>(opt.densities);
    const auto diagonal_boosts = parse_csv_list<float>(opt.diagonal_boosts);
    const auto levels = parse_csv_list<int>(opt.levels);

    std::vector<RunSpec> run_specs;
    run_specs.push_back({"baseline", -1});
    for (int level : levels) {
        run_specs.push_back({"iluk_k" + std::to_string(level), level});
    }

    auto queue = std::make_shared<Queue>(Device::default_device());

    const size_t history_size = static_cast<size_t>(opt.iterations)
                              * static_cast<size_t>(opt.batch)
                              * static_cast<size_t>(opt.neigs);
    const size_t eigval_size = static_cast<size_t>(opt.neigs) * static_cast<size_t>(opt.batch);

    std::vector<RunBuffers> runs;
    runs.reserve(run_specs.size());
    for (const auto& spec : run_specs) {
        runs.emplace_back(spec, history_size, eigval_size, static_cast<size_t>(opt.batch));
    }

    const std::filesystem::path trace_path = opt.output_prefix + "_trace.csv";
    const std::filesystem::path summary_path = opt.output_prefix + "_summary.csv";
    std::filesystem::create_directories(trace_path.parent_path());

    std::ofstream trace(trace_path);
    std::ofstream summary(summary_path);
    if (!trace.good() || !summary.good()) {
        throw std::runtime_error("Failed to open ILUK convergence output files");
    }

    trace << "sample_id,bucket_label,density,diag_boost,seed,level,label,iter,batch,eig,best_res,current_res,rate,ritz,converged\n";
    summary << "sample_id,bucket_label,density,diag_boost,seed,level,label,batch,eig,first_best,final_best,final_ritz,avg_rate,log10_reduction,first_tol_iter,last_iter,iterations_done,converged,ratio_vs_baseline,log10_ratio_vs_baseline\n";
    trace << std::setprecision(9);
    summary << std::setprecision(9);

    std::vector<AggregateStats> aggregates(run_specs.size());
    int sample_id = 0;

    std::cout << "[ILUK convergence stats] n=" << opt.n
              << " neigs=" << opt.neigs
              << " extra_directions=" << opt.extra_directions
              << " iterations=" << opt.iterations
              << " samples_per_bucket=" << opt.samples_per_bucket
              << " buckets=" << (densities.size() * diagonal_boosts.size())
              << " total_matrices=" << (densities.size() * diagonal_boosts.size() * static_cast<size_t>(opt.samples_per_bucket))
              << "\n";

    for (float density : densities) {
        for (float diagonal_boost : diagonal_boosts) {
            const auto label = bucket_label(density, diagonal_boost);
            std::cout << "  bucket=" << label << "\n";
            for (int sample = 0; sample < opt.samples_per_bucket; ++sample, ++sample_id) {
                const unsigned sample_seed = opt.seed
                    + static_cast<unsigned>(sample_id * 17)
                    + static_cast<unsigned>(sample * 101);
                auto csr = csr_generators::random_sparse_hermitian_csr<float>(
                    opt.n,
                    density,
                    opt.batch,
                    sample_seed,
                    diagonal_boost,
                    true);

                for (auto& run : runs) {
                    run_case(*queue, csr.view(), opt.neigs, opt.extra_directions, opt.iterations, run.spec, run);
                }

                std::vector<std::vector<EigenpairMetrics>> per_run_metrics;
                per_run_metrics.reserve(runs.size());
                for (const auto& run : runs) {
                    per_run_metrics.push_back(summarize_run(run, opt.iterations, opt.neigs, opt.batch, opt.tolerance));
                }

                const auto& baseline_metrics = per_run_metrics.front();
                for (size_t run_idx = 0; run_idx < runs.size(); ++run_idx) {
                    const auto& run = runs[run_idx];
                    const auto& metrics = per_run_metrics[run_idx];

                    for (int it = 0; it < opt.iterations; ++it) {
                        for (int b = 0; b < opt.batch; ++b) {
                            for (int eig = 0; eig < opt.neigs; ++eig) {
                                const size_t idx = static_cast<size_t>(it) * static_cast<size_t>(opt.batch * opt.neigs)
                                                 + static_cast<size_t>(b * opt.neigs + eig);
                                const float best = run.best_hist[idx];
                                const int converged = (std::isfinite(best) && best <= opt.tolerance) ? 1 : 0;
                                trace << sample_id << ',' << label << ',' << density << ',' << diagonal_boost << ','
                                      << sample_seed << ',' << run.spec.level << ',' << run.spec.label << ','
                                      << it << ',' << b << ',' << eig << ','
                                      << run.best_hist[idx] << ',' << run.current_hist[idx] << ','
                                      << run.rate_hist[idx] << ',' << run.ritz_hist[idx] << ','
                                      << converged << '\n';
                            }
                        }
                    }

                    for (int b = 0; b < opt.batch; ++b) {
                        for (int eig = 0; eig < opt.neigs; ++eig) {
                            const size_t metric_idx = static_cast<size_t>(b * opt.neigs + eig);
                            const auto& metric = metrics[metric_idx];
                            const auto& baseline = baseline_metrics[metric_idx];
                            float ratio_vs_baseline = std::nanf("");
                            float log10_ratio_vs_baseline = std::nanf("");
                            if (run_idx > 0 && std::isfinite(metric.final_best) && metric.final_best > 0.0f
                                && std::isfinite(baseline.final_best) && baseline.final_best > 0.0f) {
                                ratio_vs_baseline = metric.final_best / baseline.final_best;
                                log10_ratio_vs_baseline = std::log10(ratio_vs_baseline);
                            }

                            auto& agg = aggregates[run_idx];
                            if (std::isfinite(metric.final_best)) {
                                agg.final_best_sum += metric.final_best;
                                ++agg.eigenpair_count;
                            }
                            if (metric.converged) {
                                ++agg.converged_count;
                            }
                            if (std::isfinite(ratio_vs_baseline)) {
                                agg.ratio_sum += ratio_vs_baseline;
                                agg.log10_ratio_sum += log10_ratio_vs_baseline;
                                ++agg.ratio_count;
                                if (ratio_vs_baseline < 1.0f) {
                                    ++agg.win_count;
                                }
                            }

                            summary << sample_id << ',' << label << ',' << density << ',' << diagonal_boost << ','
                                    << sample_seed << ',' << run.spec.level << ',' << run.spec.label << ','
                                    << b << ',' << eig << ','
                                    << metric.first_best << ',' << metric.final_best << ',' << metric.final_ritz << ','
                                    << metric.avg_rate << ',' << metric.log10_reduction << ','
                                    << metric.first_tol_iter << ',' << metric.last_iter << ','
                                    << metric.iterations_done << ',' << (metric.converged ? 1 : 0) << ','
                                    << ratio_vs_baseline << ',' << log10_ratio_vs_baseline << '\n';
                        }
                    }
                }
            }
        }
    }

    std::cout << "[ILUK aggregate summary]\n";
    for (size_t run_idx = 0; run_idx < runs.size(); ++run_idx) {
        const auto& run = runs[run_idx];
        const auto& agg = aggregates[run_idx];
        const double mean_final_best = agg.eigenpair_count > 0
            ? agg.final_best_sum / static_cast<double>(agg.eigenpair_count)
            : std::numeric_limits<double>::quiet_NaN();
        const double converged_fraction = agg.eigenpair_count > 0
            ? static_cast<double>(agg.converged_count) / static_cast<double>(agg.eigenpair_count)
            : std::numeric_limits<double>::quiet_NaN();
        std::cout << "  " << run.spec.label
                  << ": mean_final_best=" << mean_final_best
                  << " converged_fraction=" << converged_fraction;
        if (agg.ratio_count > 0) {
            std::cout << " mean_ratio_vs_baseline=" << (agg.ratio_sum / static_cast<double>(agg.ratio_count))
                      << " mean_log10_ratio_vs_baseline=" << (agg.log10_ratio_sum / static_cast<double>(agg.ratio_count))
                      << " win_rate=" << (static_cast<double>(agg.win_count) / static_cast<double>(agg.ratio_count));
        }
        std::cout << "\n";
    }

    std::cout << "[ILUK output] trace=" << trace_path << "\n";
    std::cout << "[ILUK output] summary=" << summary_path << "\n";
    return 0;
#endif
}