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
#include <lapacke.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

struct Options {
    std::string impl = "all"; // all|steqr_cta|stedc|syev_cta|syev_blocked|syevx
    std::string backend = "CUDA"; // CUDA|ROCM|MKL|NETLIB
    std::string dtype = "float"; // float|double
    int n = 32;
    int batch = 128;
    int samples = 4096;
    double log10_cond_min = 0.0;
    double log10_cond_max = 10.0;
    unsigned int seed = 1234u;
    size_t max_sweeps = 100;
    SteqrUpdateScheme scheme = SteqrUpdateScheme::PG;
    std::string cta_shift = "wilkinson"; // lapack|wilkinson
    int sytrd_block_size = 32;
    int ormqr_block_size = 32;
    int syevx_iterations = 10;
    int syevx_extra_directions = 0;
    int syevx_neigs = -1;
    bool syevx_find_largest = false;
    std::string output = "output/accuracy/eigensolver_accuracy.csv";
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
        } else if (starts_with(arg, "--sytrd-block-size")) {
            opt.sytrd_block_size = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--ormqr-block-size")) {
            opt.ormqr_block_size = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--syevx-iterations")) {
            opt.syevx_iterations = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--syevx-extra-directions")) {
            opt.syevx_extra_directions = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--syevx-neigs")) {
            opt.syevx_neigs = std::stoi(get_value(arg));
        } else if (starts_with(arg, "--syevx-find-largest")) {
            opt.syevx_find_largest = (to_lower(get_value(arg)) == "1" || to_lower(get_value(arg)) == "true");
        } else if (starts_with(arg, "--output")) {
            opt.output = get_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Eigensolver accuracy benchmark\n\n"
                << "Reports per-matrix:\n"
                << "  R = ||AZ - Z\\Lambda|| / (||A|| * n)\n"
                << "  O = ||Z^T Z - I|| / n\n"
                << "  max relative eigenvalue error vs LAPACKE fp64 STERF\n\n"
                << "Options:\n"
                << "  --impl all|steqr_cta|stedc|syev_cta|syev_blocked|syevx\n"
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
                << "  --sytrd-block-size N\n"
                << "  --ormqr-block-size N\n"
                << "  --syevx-iterations N\n"
                << "  --syevx-extra-directions N\n"
                << "  --syevx-neigs N\n"
                << "  --syevx-find-largest 0|1\n"
                << "  --output PATH\n";
            std::exit(0);
        }
    }
    return opt;
}

bool run_impl(const std::string& selected, const std::string& name) {
    return selected == "all" || selected == name;
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
UnifiedVector<typename base_type<Real>::type> orthogonality_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& Z) {
    const int m = Z.cols();
    const int batch = Z.batch_size();
    auto ztz_minus_i = Matrix<Real>::Identity(m, batch);
    gemm<B, Real>(q,
                  Z.view(),
                  Z.view(),
                  ztz_minus_i.view(),
                  Real(1),
                  Real(-1),
                  Transpose::Trans,
                  Transpose::NoTrans);
    q.wait();
    return norm(q, ztz_minus_i.view(), NormType::Frobenius);
}

template <Backend B, typename Real>
UnifiedVector<typename base_type<Real>::type> residual_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& A,
    const Matrix<Real, MatrixFormat::Dense>& Z,
    const VectorView<Real>& evals) {
    const int n = A.rows();
    const int m = Z.cols();
    const int batch = A.batch_size();
    auto R = Matrix<Real>::Zeros(n, m, batch);

    gemm<B, Real>(q,
                  A.view(),
                  Z.view(),
                  R.view(),
                  Real(1),
                  Real(0),
                  Transpose::NoTrans,
                  Transpose::NoTrans);

    auto r_view = R.kernel_view();
    auto z_view = Z.view().kernel_view();
    auto lambda_ptr = evals.data_ptr();
    const int lambda_inc = evals.inc();
    const int lambda_stride = evals.stride();

    q->parallel_for(sycl::range<3>(
                        static_cast<size_t>(batch),
                        static_cast<size_t>(n),
                        static_cast<size_t>(m)),
                    [=](sycl::id<3> idx) {
                        const int b = static_cast<int>(idx[0]);
                        const int i = static_cast<int>(idx[1]);
                        const int j = static_cast<int>(idx[2]);
                        const Real lambda = lambda_ptr[b * lambda_stride + j * lambda_inc];
                        r_view(i, j, b) -= z_view(i, j, b) * lambda;
                    });
    q.wait();

    return norm(q, R.view(), NormType::Frobenius);
}

double max_relative_eig_error(const std::vector<double>& ref_sorted,
                              const std::vector<double>& est_unsorted) {
    if (ref_sorted.empty() || ref_sorted.size() != est_unsorted.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> est_sorted = est_unsorted;
    std::sort(est_sorted.begin(), est_sorted.end());

    double max_rel = 0.0;
    const double tiny = std::numeric_limits<double>::min();
    for (size_t i = 0; i < ref_sorted.size(); ++i) {
        const double ref = ref_sorted[i];
        const double est = est_sorted[i];
        if (!std::isfinite(ref) || !std::isfinite(est)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double denom = std::max(std::abs(ref), tiny);
        const double rel = std::abs(est - ref) / denom;
        if (!std::isfinite(rel)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        max_rel = std::max(max_rel, rel);
    }
    return max_rel;
}

template <Backend B, typename Real>
void emit_metrics_rows(std::ofstream& out,
                       Queue& q,
                       const char* impl_name,
                       int neigs,
                       bool compare_largest,
                       const Options& opt,
                       int sample_base,
                       double target_log10_cond,
                       const Matrix<Real, MatrixFormat::Dense>& A,
                       const VectorView<Real>& evals,
                       const Matrix<Real, MatrixFormat::Dense>& Z,
                       const UnifiedVector<Real>& conds,
                       const std::vector<std::vector<double>>& ref_eigs_sorted,
                       const std::vector<char>& ref_ok) {
    using Scalar = typename base_type<Real>::type;
    const int n = A.rows();
    const int m = evals.size();
    const int batch = A.batch_size();

    const auto residual_num = residual_residuals<B, Real>(q, A, Z, evals);
    const auto a_norm = norm(q, A.view(), NormType::Spectral);
    q.wait();
    const auto ortho_num = orthogonality_residuals<B, Real>(q, Z);

    const double n_scale = static_cast<double>(n);

    for (int b = 0; b < batch; ++b) {
        const double cond = static_cast<double>(conds[static_cast<size_t>(b)]);
        const double log10_cond = std::log10(std::max(cond, 1e-300));
        const double a_den = std::max(static_cast<double>(a_norm[static_cast<size_t>(b)]) * n_scale, 0.0);
        const double r_num = static_cast<double>(residual_num[static_cast<size_t>(b)]);
        const double o_num = static_cast<double>(ortho_num[static_cast<size_t>(b)]);

        double R = std::numeric_limits<double>::quiet_NaN();
        if (a_den > 0.0 && std::isfinite(r_num)) {
            R = r_num / a_den;
        }

        double O = std::numeric_limits<double>::quiet_NaN();
        if (n_scale > 0.0 && std::isfinite(o_num)) {
            O = o_num / n_scale;
        }

        std::vector<double> est_vals(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) {
            est_vals[static_cast<size_t>(i)] = static_cast<double>(evals(i, b));
        }

        double max_rel = std::numeric_limits<double>::quiet_NaN();
        if (ref_ok[static_cast<size_t>(b)]) {
            std::vector<double> ref_subset;
            const auto& ref_full = ref_eigs_sorted[static_cast<size_t>(b)];
            if (m == n) {
                ref_subset = ref_full;
            } else if (m > 0 && m <= n) {
                ref_subset.resize(static_cast<size_t>(m));
                if (compare_largest) {
                    for (int i = 0; i < m; ++i) {
                        ref_subset[static_cast<size_t>(i)] = ref_full[static_cast<size_t>(n - m + i)];
                    }
                } else {
                    for (int i = 0; i < m; ++i) {
                        ref_subset[static_cast<size_t>(i)] = ref_full[static_cast<size_t>(i)];
                    }
                }
            }
            max_rel = max_relative_eig_error(ref_subset, est_vals);
        }

        const double log10_R = std::log10(std::max(std::abs(R), 1e-300));
        const double log10_O = std::log10(std::max(std::abs(O), 1e-300));
        const double log10_rel = std::log10(std::max(std::abs(max_rel), 1e-300));

        out << (sample_base + b) << ","
            << n << ","
            << neigs << ","
            << impl_name << ","
            << opt.backend << ","
            << opt.dtype << ","
            << target_log10_cond << ","
            << cond << ","
            << log10_cond << ","
            << r_num << ","
            << a_den << ","
            << o_num << ","
            << n_scale << ","
            << R << ","
            << O << ","
            << max_rel << ","
            << log10_R << ","
            << log10_O << ","
            << log10_rel << "\n";
    }
}

template <Backend B, typename Real>
int run_accuracy(const Options& opt) {
#if !BATCHLAS_HAS_HOST_BACKEND
    (void)opt;
    std::cerr << "Host backend is required for LAPACKE fp64 STERF reference.\n";
    return 3;
#else
    const int n = opt.n;
    const int total_samples = opt.samples;
    const int batch = std::max(1, opt.batch);

    const bool run_steqr_cta = run_impl(opt.impl, "steqr_cta");
    const bool run_stedc = run_impl(opt.impl, "stedc");
    const bool run_syev_cta = run_impl(opt.impl, "syev_cta");
    const bool run_syev_blocked = run_impl(opt.impl, "syev_blocked");
    const bool run_syevx = run_impl(opt.impl, "syevx");

    if (!run_steqr_cta && !run_stedc && !run_syev_cta && !run_syev_blocked && !run_syevx) {
        std::cerr << "No impl selected. Use --impl all|steqr_cta|stedc|syev_cta|syev_blocked|syevx\n";
        return 2;
    }

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

    out << "sample,n,neigs,impl,backend,dtype,target_log10_cond,cond,log10_cond,res_num,res_denom,ortho_num,ortho_denom,R,O,max_relerr,log10_R,log10_O,log10_relerr\n";
    out << std::setprecision(12);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    int sample_id = 0;
    for (int bin_idx = 0; bin_idx < bins; ++bin_idx) {
        const double target_log10 = (bins == 1)
                                        ? opt.log10_cond_min
                                        : (opt.log10_cond_min + (static_cast<double>(bin_idx) + 0.5) * bin_width);

        for (int bin_sample = 0; bin_sample < per_bin; bin_sample += batch) {
            const int cur_batch = batch;

            auto A = random_hermitian_tridiagonal_with_log10_cond_metric<B, Real>(
                *q,
                n,
                static_cast<Real>(target_log10),
                NormType::Spectral,
                cur_batch,
                opt.seed + static_cast<unsigned int>(sample_id));
            q->wait();

            const auto conds = cond<B>(*q, A.view(), NormType::Spectral);
            q->wait();

            Vector<Real> d_ref(n, Real(0), cur_batch);
            Vector<Real> e_ref(std::max(0, n - 1), Real(0), cur_batch);
            extract_tridiagonal(*q, A.view(), d_ref, e_ref);

            std::vector<std::vector<double>> ref_eigs_sorted(static_cast<size_t>(cur_batch), std::vector<double>(static_cast<size_t>(n), std::numeric_limits<double>::quiet_NaN()));
            std::vector<char> ref_ok(static_cast<size_t>(cur_batch), 0);
            for (int b = 0; b < cur_batch; ++b) {
                std::vector<double> d_work(static_cast<size_t>(n));
                std::vector<double> e_work(static_cast<size_t>(std::max(0, n - 1)));
                for (int i = 0; i < n; ++i) {
                    d_work[static_cast<size_t>(i)] = static_cast<double>(d_ref(i, b));
                    if (i < n - 1) e_work[static_cast<size_t>(i)] = static_cast<double>(e_ref(i, b));
                }
                const int info = LAPACKE_dsterf(static_cast<lapack_int>(n), d_work.data(), e_work.data());
                if (info == 0) {
                    std::sort(d_work.begin(), d_work.end());
                    ref_eigs_sorted[static_cast<size_t>(b)] = std::move(d_work);
                    ref_ok[static_cast<size_t>(b)] = 1;
                }
            }

            if (run_steqr_cta) {
                try {
                    Vector<Real> d_work(n, Real(0), cur_batch);
                    Vector<Real> e_work(std::max(0, n - 1), Real(0), cur_batch);
                    extract_tridiagonal(*q, A.view(), d_work, e_work);

                    auto eigvals = Vector<Real>::zeros(n, cur_batch);
                    auto eigvects = Matrix<Real>::Identity(n, cur_batch);

                    SteqrParams<Real> params{};
                    params.max_sweeps = opt.max_sweeps;
                    params.sort = true;
                    params.transpose_working_vectors = false;
                    params.sort_order = SortOrder::Ascending;
                    params.cta_update_scheme = opt.scheme;
                    params.cta_shift_strategy = parse_shift_strategy(opt.cta_shift);

                    UnifiedVector<std::byte> ws(
                        steqr_cta_buffer_size<Real>(*q, d_work, e_work, eigvals, JobType::EigenVectors, params));
                    steqr_cta<B, Real>(*q,
                                       d_work,
                                       e_work,
                                       eigvals,
                                       ws.to_span(),
                                       JobType::EigenVectors,
                                       params,
                                       eigvects);
                    q->wait();

                    const char* impl_name = (opt.scheme == SteqrUpdateScheme::PG) ? "steqr_cta_pg" : "steqr_cta_exp";
                    emit_metrics_rows<B, Real>(out,
                                               *q,
                                               impl_name,
                                               n,
                                               false,
                                               opt,
                                               sample_id,
                                               target_log10,
                                               A,
                                               VectorView<Real>(eigvals),
                                               eigvects,
                                               conds,
                                               ref_eigs_sorted,
                                               ref_ok);
                } catch (const std::exception& ex) {
                    std::cerr << "steqr_cta failed: " << ex.what() << "\n";
                }
            }

            if (run_stedc) {
                try {
                    Vector<Real> d_work(n, Real(0), cur_batch);
                    Vector<Real> e_work(std::max(0, n - 1), Real(0), cur_batch);
                    extract_tridiagonal(*q, A.view(), d_work, e_work);

                    auto eigvals = Vector<Real>::zeros(n, cur_batch);
                    auto eigvects = Matrix<Real>::Identity(n, cur_batch);
                    StedcParams<Real> params{};
                    UnifiedVector<std::byte> ws(
                        stedc_workspace_size<B, Real>(*q, n, cur_batch, JobType::EigenVectors, params));
                    stedc<B, Real>(*q,
                                   d_work,
                                   e_work,
                                   eigvals,
                                   ws.to_span(),
                                   JobType::EigenVectors,
                                   params,
                                   eigvects);
                    q->wait();

                    emit_metrics_rows<B, Real>(out,
                                               *q,
                                               "stedc",
                                               n,
                                               false,
                                               opt,
                                               sample_id,
                                               target_log10,
                                               A,
                                               VectorView<Real>(eigvals),
                                               eigvects,
                                               conds,
                                               ref_eigs_sorted,
                                               ref_ok);
                } catch (const std::exception& ex) {
                    std::cerr << "stedc failed: " << ex.what() << "\n";
                }
            }

            if (run_syev_cta) {
                try {
                    auto A_work = A.clone();
                    UnifiedVector<Real> eigvals(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

                    SteqrParams<Real> params{};
                    params.max_sweeps = opt.max_sweeps;
                    params.cta_update_scheme = opt.scheme;
                    params.cta_shift_strategy = parse_shift_strategy(opt.cta_shift);

                    UnifiedVector<std::byte> ws(
                        syev_cta_buffer_size<B, Real>(*q, A_work.view(), JobType::EigenVectors, params));
                    syev_cta<B, Real>(*q,
                                      A_work.view(),
                                      eigvals.to_span(),
                                      JobType::EigenVectors,
                                      Uplo::Lower,
                                      ws.to_span(),
                                      params);
                    q->wait();

                    VectorView<Real> evals_view(eigvals.to_span(), n, cur_batch, 1, n);
                    const char* impl_name = (opt.scheme == SteqrUpdateScheme::PG) ? "syev_cta_pg" : "syev_cta_exp";
                    emit_metrics_rows<B, Real>(out,
                                               *q,
                                               impl_name,
                                               n,
                                               false,
                                               opt,
                                               sample_id,
                                               target_log10,
                                               A,
                                               evals_view,
                                               A_work,
                                               conds,
                                               ref_eigs_sorted,
                                               ref_ok);
                } catch (const std::exception& ex) {
                    std::cerr << "syev_cta failed: " << ex.what() << "\n";
                }
            }

            if (run_syev_blocked) {
                try {
                    auto A_work = A.clone();
                    UnifiedVector<Real> eigvals(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

                    UnifiedVector<std::byte> ws(
                        syev_blocked_buffer_size<B, Real>(*q,
                                                          A_work.view(),
                                                          JobType::EigenVectors,
                                                          Uplo::Lower,
                                                          opt.sytrd_block_size,
                                                          opt.ormqr_block_size));
                    syev_blocked<B, Real>(*q,
                                          A_work.view(),
                                          eigvals.to_span(),
                                          JobType::EigenVectors,
                                          Uplo::Lower,
                                          ws.to_span(),
                                          opt.sytrd_block_size,
                                          opt.ormqr_block_size);
                    q->wait();

                    VectorView<Real> evals_view(eigvals.to_span(), n, cur_batch, 1, n);
                    emit_metrics_rows<B, Real>(out,
                                               *q,
                                               "syev_blocked",
                                               n,
                                               false,
                                               opt,
                                               sample_id,
                                               target_log10,
                                               A,
                                               evals_view,
                                               A_work,
                                               conds,
                                               ref_eigs_sorted,
                                               ref_ok);
                } catch (const std::exception& ex) {
                    std::cerr << "syev_blocked failed: " << ex.what() << "\n";
                }
            }

            if (run_syevx) {
                try {
                    auto A_work = A.clone();
                    const int neigs = (opt.syevx_neigs > 0)
                                          ? std::min(opt.syevx_neigs, n)
                                          : std::max(1, n / 4);
                    UnifiedVector<Real> eigvals(static_cast<size_t>(neigs) * static_cast<size_t>(cur_batch));
                    auto V = Matrix<Real>::Zeros(n, neigs, cur_batch);

                    SyevxParams<Real> params{};
                    params.algorithm = OrthoAlgorithm::Chol2;
                    params.iterations = std::max(1, opt.syevx_iterations);
                    params.extra_directions = std::max(0, opt.syevx_extra_directions);
                    params.find_largest = opt.syevx_find_largest;
                    params.absolute_tolerance = static_cast<Real>(1e-6);
                    params.relative_tolerance = static_cast<Real>(1e-6);

                    UnifiedVector<std::byte> ws(
                        syevx_buffer_size<B>(*q,
                                             A_work.view(),
                                             eigvals.to_span(),
                                             static_cast<size_t>(neigs),
                                             JobType::EigenVectors,
                                             V.view(),
                                             params));

                    syevx<B>(*q,
                             A_work.view(),
                             eigvals.to_span(),
                             static_cast<size_t>(neigs),
                             ws.to_span(),
                             JobType::EigenVectors,
                             V.view(),
                             params);
                    q->wait();

                    VectorView<Real> evals_view(eigvals.to_span(), neigs, cur_batch, 1, neigs);
                    emit_metrics_rows<B, Real>(out,
                                               *q,
                                               "syevx",
                                               neigs,
                                               opt.syevx_find_largest,
                                               opt,
                                               sample_id,
                                               target_log10,
                                               A,
                                               evals_view,
                                               V,
                                               conds,
                                               ref_eigs_sorted,
                                               ref_ok);
                } catch (const std::exception& ex) {
                    std::cerr << "syevx failed: " << ex.what() << "\n";
                }
            }

            sample_id += cur_batch;
        }
    }

    return 0;
#endif
}

int dispatch(const Options& opt) {
    if (opt.dtype != "float" && opt.dtype != "double") {
        std::cerr << "Unsupported dtype for eigensolver accuracy: " << opt.dtype << " (use float/double)\n";
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
