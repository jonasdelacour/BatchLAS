#include <blas/functions.hh>
#include <blas/extra.hh>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>
#include "../src/queue.hh"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

struct Options {
    std::string impl = "all";
    std::string backend = "CUDA";
    std::string dtype = "float";
    int n = 32;
    int batch = 128;
    int samples = 4096;
    double log10_cond_min = 0.0;
    double log10_cond_max = 10.0;
    unsigned int seed = 1234u;
    size_t max_sweeps = 100;
    SteqrUpdateScheme scheme = SteqrUpdateScheme::PG;
    std::string cta_shift = "wilkinson";
    std::string output = "output/accuracy/orthogonality_accuracy.csv";
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

SteqrShiftStrategy parse_shift_strategy(const std::string& value) {
    const auto key = to_lower(value);
    if (key == "lapack") return SteqrShiftStrategy::Lapack;
    if (key == "wilkinson") return SteqrShiftStrategy::Wilkinson;
    throw std::invalid_argument("Invalid --cta-shift value (use lapack or wilkinson)");
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto get_value = [&](const std::string& a) -> std::string {
            auto pos = a.find('=');
            if (pos != std::string::npos) return a.substr(pos + 1);
            if (i + 1 < argc) return std::string(argv[++i]);
            return "";
        };

        if (starts_with(arg, "--impl")) {
            opt.impl = to_lower(get_value(arg));
        } else if (starts_with(arg, "--scheme")) {
            const auto val = to_lower(get_value(arg));
            if (val == "pg") {
                opt.scheme = SteqrUpdateScheme::PG;
            } else if (val == "exp") {
                opt.scheme = SteqrUpdateScheme::EXP;
            } else {
                throw std::invalid_argument("Invalid --scheme value (use pg or exp)");
            }
        } else if (starts_with(arg, "--backend")) {
            opt.backend = get_value(arg);
        } else if (starts_with(arg, "--type")) {
            opt.dtype = get_value(arg);
        } else if (starts_with(arg, "--n")) {
            opt.n = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--batch")) {
            opt.batch = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--samples")) {
            opt.samples = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--log10-cond-min")) {
            opt.log10_cond_min = std::stod(get_value(arg));
        } else if (starts_with(arg, "--log10-cond-max")) {
            opt.log10_cond_max = std::stod(get_value(arg));
        } else if (starts_with(arg, "--seed")) {
            opt.seed = static_cast<unsigned int>(std::stoul(get_value(arg)));
        } else if (starts_with(arg, "--max-sweeps")) {
            opt.max_sweeps = static_cast<size_t>(std::stoul(get_value(arg)));
        } else if (starts_with(arg, "--cta-shift")) {
            opt.cta_shift = get_value(arg);
        } else if (starts_with(arg, "--output")) {
            opt.output = get_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Orthogonality accuracy sampler\n\n"
                      << "Options:\n"
                      << "  --impl all|ortho_all|eig_all|ortho_chol2|ortho_cholesky|ortho_shiftchol3|ortho_householder|ortho_cgs2|ortho_svqb|ortho_svqb2|syev|steqr|steqr_cta|stedc\n"
                      << "  --scheme pg|exp\n"
                      << "  --backend CUDA|ROCM|MKL|NETLIB\n"
                      << "  --type float|double\n"
                      << "  --n N\n"
                      << "  --batch B\n"
                      << "  --samples S\n"
                      << "  --log10-cond-min X\n"
                      << "  --log10-cond-max X\n"
                      << "  --seed SEED\n"
                      << "  --max-sweeps N\n"
                      << "  --cta-shift lapack|wilkinson\n"
                      << "  --output PATH\n";
            std::exit(0);
        }
    }
    return opt;
}

bool run_impl(const std::string& selected, const std::string& name, const std::string& group) {
    return selected == "all" || selected == name || selected == group;
}

template <Backend B, typename Real>
UnifiedVector<typename batchlas::base_type<Real>::type> orthogonality_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& Q) {
    const int n = Q.rows();
    const int batch = Q.batch_size();
    auto qtq_minus_i = Matrix<Real>::Identity(n, batch);
    gemm<B, Real>(q,
                  Q.view(),
                  Q.view(),
                  qtq_minus_i.view(),
                  Real(1),
                  Real(-1),
                  Transpose::Trans,
                  Transpose::NoTrans);
    q.wait();
    return norm(q, qtq_minus_i.view(), NormType::Frobenius);
}

template <typename Real>
void extract_tridiagonal(Queue& q,
                         const MatrixView<Real, MatrixFormat::Dense>& dense,
                         Vector<Real>& d,
                         Vector<Real>& e) {
    const int n = dense.rows();
    const int batch = dense.batch_size();
    auto a_view = dense.kernel_view();
    auto d_ptr = d.data_ptr();
    auto e_ptr = e.data_ptr();
    const int d_inc = d.inc();
    const int e_inc = e.inc();
    const int d_stride = d.stride();
    const int e_stride = e.stride();

    q->parallel_for(sycl::range<1>(static_cast<size_t>(batch * n)), [=](sycl::id<1> idx) {
        const int linear = static_cast<int>(idx[0]);
        const int b = linear / n;
        const int i = linear - b * n;
        d_ptr[b * d_stride + i * d_inc] = a_view(i, i, b);
        if (i < n - 1) {
            e_ptr[b * e_stride + i * e_inc] = a_view(i + 1, i, b);
        }
    });
    q.wait();
}

template <Backend B, typename Real>
int run_accuracy(const Options& opt) {
    const int n = opt.n;
    const int total_samples = opt.samples;
    const int batch = std::max(1, opt.batch);

    const bool run_ortho_chol2 = run_impl(opt.impl, "ortho_chol2", "ortho_all");
    const bool run_ortho_cholesky = run_impl(opt.impl, "ortho_cholesky", "ortho_all");
    const bool run_ortho_shiftchol3 = run_impl(opt.impl, "ortho_shiftchol3", "ortho_all");
    const bool run_ortho_householder = run_impl(opt.impl, "ortho_householder", "ortho_all");
    const bool run_ortho_cgs2 = run_impl(opt.impl, "ortho_cgs2", "ortho_all");
    const bool run_ortho_svqb = run_impl(opt.impl, "ortho_svqb", "ortho_all");
    const bool run_ortho_svqb2 = run_impl(opt.impl, "ortho_svqb2", "ortho_all");

    const bool run_syev = run_impl(opt.impl, "syev", "eig_all");
    const bool run_steqr = run_impl(opt.impl, "steqr", "eig_all");
    const bool run_steqr_cta = run_impl(opt.impl, "steqr_cta", "eig_all");
    const bool run_stedc = run_impl(opt.impl, "stedc", "eig_all");

    const double log10_span = opt.log10_cond_max - opt.log10_cond_min;
    const int bins = std::max(1, static_cast<int>(std::ceil(std::max(0.0, log10_span) * 10.0)));
    const double bin_width = (bins > 0) ? (log10_span / static_cast<double>(bins)) : 0.0;
    const int base_per_bin = static_cast<int>(std::ceil(static_cast<double>(total_samples) / static_cast<double>(bins)));
    const int per_bin = static_cast<int>(std::ceil(static_cast<double>(base_per_bin) / static_cast<double>(batch))) * batch;

    std::filesystem::path out_path(opt.output);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(opt.output);
    if (!out) {
        std::cerr << "Failed to open output file: " << opt.output << "\n";
        return 4;
    }
    out << "sample,n,impl,backend,dtype,target_log10_cond,cond,log10_cond,orthogonality,log10_orthogonality\n";
    out << std::setprecision(12);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto emit_rows = [&](int sample_base,
                         const std::vector<double>& target_log10s,
                         const UnifiedVector<Real>& conds,
                         const UnifiedVector<typename batchlas::base_type<Real>::type>& ortho_vals,
                         const char* impl_name,
                         int cur_batch) {
        for (int b = 0; b < cur_batch; ++b) {
            const double cond = static_cast<double>(conds[static_cast<size_t>(b)]);
            const double log10_cond = std::log10(std::max(cond, 1e-300));
            const double orth = static_cast<double>(ortho_vals[static_cast<size_t>(b)]);
            const double log10_orth = std::log10(std::max(orth, 1e-300));

            out << (sample_base + b) << ","
                << n << ","
                << impl_name << ","
                << opt.backend << ","
                << opt.dtype << ","
                << target_log10s[static_cast<size_t>(b)] << ","
                << cond << ","
                << log10_cond << ","
                << orth << ","
                << log10_orth << "\n";
        }
    };

    int sample_id = 0;
    for (int bin_idx = 0; bin_idx < bins; ++bin_idx) {
        const double target_log10 = (bins == 1)
                                        ? opt.log10_cond_min
                                        : (opt.log10_cond_min + (static_cast<double>(bin_idx) + 0.5) * bin_width);

        for (int bin_sample = 0; bin_sample < per_bin; bin_sample += batch) {
            const int cur_batch = batch;
            std::vector<double> target_log10s(static_cast<size_t>(cur_batch), target_log10);

            auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, Real>(
                *q,
                n,
                static_cast<Real>(target_log10),
                NormType::Spectral,
                cur_batch,
                opt.seed + static_cast<unsigned int>(sample_id));
            q->wait();

            const auto conds = cond<B>(*q, dense_A.view(), NormType::Spectral);
            q->wait();

            auto run_ortho_case = [&](OrthoAlgorithm algo, const char* impl_name) {
                auto Q = dense_A.clone();
                UnifiedVector<std::byte> ws(ortho_buffer_size<B, Real>(*q, Q.view(), Transpose::NoTrans, algo));
                ortho<B, Real>(*q, Q.view(), Transpose::NoTrans, ws.to_span(), algo);
                q->wait();
                const auto ortho_vals = orthogonality_residuals<B, Real>(*q, Q);
                emit_rows(sample_id, target_log10s, conds, ortho_vals, impl_name, cur_batch);
            };

            if (run_ortho_chol2) run_ortho_case(OrthoAlgorithm::Chol2, "ortho_chol2");
            if (run_ortho_cholesky) run_ortho_case(OrthoAlgorithm::Cholesky, "ortho_cholesky");
            if (run_ortho_shiftchol3) run_ortho_case(OrthoAlgorithm::ShiftChol3, "ortho_shiftchol3");
            if (run_ortho_householder) run_ortho_case(OrthoAlgorithm::Householder, "ortho_householder");
            if (run_ortho_cgs2) run_ortho_case(OrthoAlgorithm::CGS2, "ortho_cgs2");
            if (run_ortho_svqb) run_ortho_case(OrthoAlgorithm::SVQB, "ortho_svqb");
            if (run_ortho_svqb2) run_ortho_case(OrthoAlgorithm::SVQB2, "ortho_svqb2");

            if (run_syev) {
                auto A = dense_A.clone();
                UnifiedVector<Real> eigvals(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));
                UnifiedVector<std::byte> ws(
                    syev_buffer_size<B, Real>(*q, A.view(), eigvals.to_span(), JobType::EigenVectors, Uplo::Lower));
                syev<B, Real>(*q, A.view(), eigvals.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
                q->wait();
                const auto ortho_vals = orthogonality_residuals<B, Real>(*q, A);
                emit_rows(sample_id, target_log10s, conds, ortho_vals, "syev", cur_batch);
            }

            if (run_steqr || run_steqr_cta || run_stedc) {
                SteqrParams<Real> steqr_params{};
                steqr_params.max_sweeps = opt.max_sweeps;
                steqr_params.sort = true;
                steqr_params.transpose_working_vectors = false;
                steqr_params.sort_order = SortOrder::Ascending;
                steqr_params.cta_shift_strategy = parse_shift_strategy(opt.cta_shift);

                if (run_steqr) {
                    Vector<Real> d_work(n, Real(0), cur_batch);
                    Vector<Real> e_work(std::max(0, n - 1), Real(0), cur_batch);
                    extract_tridiagonal(*q, dense_A.view(), d_work, e_work);
                    auto eigvals = Vector<Real>::zeros(n, cur_batch);
                    auto eigvects = Matrix<Real>::Identity(n, cur_batch);
                    UnifiedVector<std::byte> ws(
                        steqr_buffer_size<Real>(*q, d_work, e_work, eigvals, JobType::EigenVectors, steqr_params));
                    steqr<B, Real>(*q,
                                   d_work,
                                   e_work,
                                   eigvals,
                                   ws.to_span(),
                                   JobType::EigenVectors,
                                   steqr_params,
                                   eigvects);
                    q->wait();
                    const auto ortho_vals = orthogonality_residuals<B, Real>(*q, eigvects);
                    emit_rows(sample_id, target_log10s, conds, ortho_vals, "steqr", cur_batch);
                }

                if (run_steqr_cta) {
                    Vector<Real> d_work(n, Real(0), cur_batch);
                    Vector<Real> e_work(std::max(0, n - 1), Real(0), cur_batch);
                    extract_tridiagonal(*q, dense_A.view(), d_work, e_work);
                    auto eigvals = Vector<Real>::zeros(n, cur_batch);
                    auto eigvects = Matrix<Real>::Identity(n, cur_batch);
                    steqr_params.cta_update_scheme = opt.scheme;
                    const char* cta_name = opt.scheme == SteqrUpdateScheme::PG ? "steqr_cta_pg" : "steqr_cta_exp";
                    UnifiedVector<std::byte> ws(
                        steqr_cta_buffer_size<Real>(*q, d_work, e_work, eigvals, JobType::EigenVectors, steqr_params));
                    steqr_cta<B, Real>(*q,
                                       d_work,
                                       e_work,
                                       eigvals,
                                       ws.to_span(),
                                       JobType::EigenVectors,
                                       steqr_params,
                                       eigvects);
                    q->wait();
                    const auto ortho_vals = orthogonality_residuals<B, Real>(*q, eigvects);
                    emit_rows(sample_id, target_log10s, conds, ortho_vals, cta_name, cur_batch);
                }

                if (run_stedc) {
                    Vector<Real> d_work(n, Real(0), cur_batch);
                    Vector<Real> e_work(std::max(0, n - 1), Real(0), cur_batch);
                    extract_tridiagonal(*q, dense_A.view(), d_work, e_work);
                    auto eigvals = Vector<Real>::zeros(n, cur_batch);
                    auto eigvects = Matrix<Real>::Identity(n, cur_batch);
                    StedcParams<Real> stedc_params{};
                    UnifiedVector<std::byte> ws(
                        stedc_workspace_size<B, Real>(*q, n, cur_batch, JobType::EigenVectors, stedc_params));
                    stedc<B, Real>(*q,
                                   d_work,
                                   e_work,
                                   eigvals,
                                   ws.to_span(),
                                   JobType::EigenVectors,
                                   stedc_params,
                                   eigvects);
                    q->wait();
                    const auto ortho_vals = orthogonality_residuals<B, Real>(*q, eigvects);
                    emit_rows(sample_id, target_log10s, conds, ortho_vals, "stedc", cur_batch);
                }
            }

            sample_id += cur_batch;
        }
    }

    return 0;
}

int dispatch(const Options& opt) {
    if (opt.dtype != "float" && opt.dtype != "double") {
        std::cerr << "Unsupported dtype for orthogonality accuracy: " << opt.dtype << " (use float/double)\n";
        return 1;
    }

    if (opt.backend == "CUDA") {
#if BATCHLAS_HAS_CUDA_BACKEND
        if (opt.dtype == "float") return run_accuracy<Backend::CUDA, float>(opt);
        return run_accuracy<Backend::CUDA, double>(opt);
#else
        std::cerr << "CUDA backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "ROCM") {
#if BATCHLAS_HAS_ROCM_BACKEND
        if (opt.dtype == "float") return run_accuracy<Backend::ROCM, float>(opt);
        return run_accuracy<Backend::ROCM, double>(opt);
#else
        std::cerr << "ROCM backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "MKL") {
#if BATCHLAS_HAS_MKL_BACKEND
        if (opt.dtype == "float") return run_accuracy<Backend::MKL, float>(opt);
        return run_accuracy<Backend::MKL, double>(opt);
#else
        std::cerr << "MKL backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "NETLIB") {
#if BATCHLAS_HAS_HOST_BACKEND
        if (opt.dtype == "float") return run_accuracy<Backend::NETLIB, float>(opt);
        return run_accuracy<Backend::NETLIB, double>(opt);
#else
        std::cerr << "NETLIB backend not available in this build\n";
        return 1;
#endif
    }

    std::cerr << "Unknown backend: " << opt.backend << "\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    auto opt = parse_args(argc, argv);

    if (opt.log10_cond_max < opt.log10_cond_min) {
        std::swap(opt.log10_cond_min, opt.log10_cond_max);
    }

    if (opt.samples <= 0 || opt.n <= 0 || opt.batch <= 0) {
        std::cerr << "Invalid --samples, --n, or --batch value\n";
        return 1;
    }

    return dispatch(opt);
}
