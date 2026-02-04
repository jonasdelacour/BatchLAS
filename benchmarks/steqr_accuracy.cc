#include <blas/linalg.hh>
#include <blas/extra.hh>
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
#include <lapacke.h>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

struct Options {
    std::string impl = "steqr"; // steqr | steqr_cta | cuda_syev | netlib_syev32 | netlib_steqr | netlib_sterf | netlib_stedc | both | all
    std::string backend = "CUDA"; // CUDA | ROCM | MKL | NETLIB
    std::string dtype = "float"; // float | double
    int n = 32;
    int batch = 128;
    int samples = 4096;
    double log10_cond_min = 0.0;
    double log10_cond_max = 10.0;
    unsigned int seed = 1234u;
    size_t max_sweeps = 100;
    SteqrUpdateScheme scheme = SteqrUpdateScheme::PG;
    std::string cta_shift = "wilkinson"; // lapack | wilkinson
    std::string output = "output/accuracy/steqr_accuracy.csv";
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
    if (key == "lapack") {
        return SteqrShiftStrategy::Lapack;
    }
    if (key == "wilkinson") {
        return SteqrShiftStrategy::Wilkinson;
    }
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
            opt.impl = get_value(arg);
        } else if (starts_with(arg, "--scheme")) {
            const auto val = to_lower(get_value(arg));
            if (val == "pg") {
                opt.scheme = SteqrUpdateScheme::PG;
            } else if (val == "exp") {
                opt.scheme = SteqrUpdateScheme::EXP;
            } else {
                throw std::invalid_argument("Invalid --scheme value (use pg, exp, or pg_guarded)");
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
            std::cout << "STEQR accuracy sampler\n\n"
                      << "Options:\n"
                      << "  --impl steqr|steqr_cta|cuda_syev|netlib_syev32|netlib_steqr|netlib_sterf|netlib_stedc|stedc|both|all\n"
                      << "  --scheme pg|exp|pg_guarded\n"
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

#if BATCHLAS_HAS_HOST_BACKEND
template <typename OutType, typename InType>
UnifiedVector<OutType> netlib_ref_eigs(const MatrixView<InType, MatrixFormat::Dense>& A) {
    const int n = A.rows();
    const int batch = A.batch_size();
    Queue ctx_cpu("cpu");
    auto A_conv = A.template astype<OutType>();
    UnifiedVector<OutType> ref_eigs(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    UnifiedVector<std::byte> ws(
        syev_buffer_size<Backend::NETLIB, OutType>(ctx_cpu, A_conv.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev<Backend::NETLIB, OutType>(ctx_cpu, A_conv.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span()).wait();
    ctx_cpu.wait();
    return ref_eigs;
}

template <typename Real>
int call_lapack_steqr(int n, Real* d, Real* e) {
    if constexpr (std::is_same_v<Real, float>) {
        return LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(n), d, e, nullptr, 1);
    } else {
        return LAPACKE_dsteqr(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(n), d, e, nullptr, 1);
    }
}

template <typename Real>
int call_lapack_sterf(int n, Real* d, Real* e) {
    if constexpr (std::is_same_v<Real, float>) {
        return LAPACKE_ssterf(static_cast<lapack_int>(n), d, e);
    } else {
        return LAPACKE_dsterf(static_cast<lapack_int>(n), d, e);
    }
}

template <typename Real>
int call_lapack_stedc(int n, Real* d, Real* e) {
    if constexpr (std::is_same_v<Real, float>) {
        return LAPACKE_sstedc(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(n), d, e, nullptr, 1);
    } else {
        return LAPACKE_dstedc(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(n), d, e, nullptr, 1);
    }
}

bool should_run_impl(const std::string& opt_impl, const char* impl_name, bool type_matches = true) {
    const std::string name(impl_name);
    return type_matches && (opt_impl == name || opt_impl == "both" || opt_impl == "all");
}
#endif

template <Backend B, typename Real>
int run_accuracy(const Options& opt) {
    constexpr bool is_float = std::is_same_v<Real, float>;
    const bool run_steqr = should_run_impl(opt.impl, "steqr");
    const bool run_cta = should_run_impl(opt.impl, "steqr_cta");
    const bool run_cuda_syev = should_run_impl(opt.impl, "cuda_syev");
    const bool run_netlib32 = should_run_impl(opt.impl, "netlib_syev32", is_float);
    const bool run_netlib_steqr = should_run_impl(opt.impl, "netlib_steqr");
    const bool run_netlib_sterf = should_run_impl(opt.impl, "netlib_sterf");
    const bool run_netlib_stedc = should_run_impl(opt.impl, "netlib_stedc");
    const bool run_stedc = should_run_impl(opt.impl, "stedc");

    if (run_cuda_syev && B != Backend::CUDA) {
        std::cerr << "cuda_syev requires CUDA backend\n";
        return 2;
    }
    if (opt.impl == "netlib_syev32" && !is_float) {
        std::cerr << "netlib_syev32 requires --type=float\n";
        return 2;
    }

#if !BATCHLAS_HAS_HOST_BACKEND
    std::cerr << "Host backend required for NETLIB reference eigenvalues.\n";
    return 3;
#else
    const int n = opt.n;
    const int total_samples = opt.samples;
    const int batch = std::max(1, opt.batch);
    const double log10_span = opt.log10_cond_max - opt.log10_cond_min;
    const int bins = std::max(1, static_cast<int>(std::ceil(std::max(0.0, log10_span) * 10.0)));
    const double bin_width = (bins > 0) ? (log10_span / static_cast<double>(bins)) : 0.0;
    const int base_per_bin = static_cast<int>(std::ceil(static_cast<double>(total_samples) / static_cast<double>(bins)));
    const int per_bin = static_cast<int>(std::ceil(static_cast<double>(base_per_bin) / static_cast<double>(batch))) * batch;
    const auto scheme = opt.scheme;

    const char* cta_impl_name = "steqr_cta";
    switch (scheme) {
        case SteqrUpdateScheme::PG:
            cta_impl_name = "steqr_cta_pg";
            break;
        case SteqrUpdateScheme::EXP:
            cta_impl_name = "steqr_cta_exp";
            break;
    }

    std::filesystem::path out_path(opt.output);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(opt.output);
    if (!out) {
        std::cerr << "Failed to open output file: " << opt.output << "\n";
        return 4;
    }

    out << "sample,n,impl,backend,dtype,target_log10_cond,cond,log10_cond,relerr,log10_relerr\n";
    out << std::setprecision(12);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    int sample_id = 0;
    for (int bin_idx = 0; bin_idx < bins; ++bin_idx) {
        const double target_log10 = (bins == 1) ? opt.log10_cond_min
                                                : (opt.log10_cond_min + (static_cast<double>(bin_idx) + 0.5) * bin_width);
        for (int bin_sample = 0; bin_sample < per_bin; bin_sample += batch) {
            const int cur_batch = batch;

        Vector<Real> d(n, Real(0), cur_batch);
        Vector<Real> e(n - 1, Real(0), cur_batch);
        Vector<Real> eigs_steqr = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_cta = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_cuda = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_netlib_steqr = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_netlib_sterf = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_netlib_stedc = Vector<Real>::zeros(n, cur_batch);
        Vector<Real> eigs_stedc = Vector<Real>::zeros(n, cur_batch);
        auto eigvects = Matrix<Real>::Zeros(n, n, cur_batch);

        std::vector<double> target_log10s(static_cast<std::size_t>(cur_batch), 0.0);

        auto dense_A = Matrix<Real>::Zeros(n, n, cur_batch);
        auto dense_A_view = dense_A.view();
        for (int b = 0; b < cur_batch; ++b) {
            target_log10s[static_cast<std::size_t>(b)] = target_log10;
            auto A_b = random_hermitian_tridiagonal_with_log10_cond_metric<B, Real>(
                *q, n, static_cast<Real>(target_log10), NormType::Spectral, 1, opt.seed + static_cast<unsigned int>(sample_id + b));
            MatrixView<Real, MatrixFormat::Dense>::copy(*q, dense_A_view.batch_item(b), A_b.view().batch_item(0)).wait();
        }
        q->wait();

        auto extract_diag = [&](Vector<Real>& d_out, Vector<Real>& e_out) {
            auto a_view = dense_A_view.kernel_view();
            auto d_ptr = d_out.data_ptr();
            auto e_ptr = e_out.data_ptr();
            const int d_inc = d_out.inc();
            const int e_inc = e_out.inc();
            const int d_stride = d_out.stride();
            const int e_stride = e_out.stride();
            (*q)->parallel_for(sycl::range<1>(static_cast<size_t>(cur_batch * n)), [=](sycl::id<1> idx) {
                const int linear = static_cast<int>(idx[0]);
                const int b = linear / n;
                const int i = linear - b * n;
                d_ptr[b * d_stride + i * d_inc] = a_view(i, i, b);
                if (i < n - 1) {
                    e_ptr[b * e_stride + i * e_inc] = a_view(i + 1, i, b);
                }
            });
            q->wait();
        };
        extract_diag(d, e);

        SteqrParams<Real> params = {};
        params.max_sweeps = opt.max_sweeps;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = parse_shift_strategy(opt.cta_shift);

        if (run_steqr) {
            params.max_sweeps = 10;
            UnifiedVector<std::byte> ws(
                steqr_buffer_size<Real>(*q, d, e, eigs_steqr, JobType::EigenVectors, params));
            steqr<B, Real>(*q, d, e, eigs_steqr, ws.to_span(), JobType::EigenVectors, params, eigvects);
            q->wait();
        }
        if (run_cta) {
            params.max_sweeps = opt.max_sweeps;
            params.cta_update_scheme = scheme;
            UnifiedVector<std::byte> ws(
                steqr_cta_buffer_size<Real>(*q, d, e, eigs_cta, JobType::EigenVectors, params));
            steqr_cta<B, Real>(*q, d, e, eigs_cta, ws.to_span(), JobType::EigenVectors, params, eigvects);
            q->wait();
        }
        if (run_cuda_syev) {
            auto dense_vendor = dense_A.clone();
            auto eigs_span = eigs_cuda.data();
            UnifiedVector<std::byte> ws(
                backend::syev_vendor_buffer_size<Backend::CUDA, Real>(*q, dense_vendor.view(), eigs_span, JobType::NoEigenVectors, Uplo::Lower));
            backend::syev_vendor<Backend::CUDA, Real>(*q, dense_vendor.view(), eigs_span, JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        }
        if (run_stedc) {
            UnifiedVector<std::byte> ws(
                stedc_workspace_size<B, Real>(*q, n, cur_batch, JobType::EigenVectors, StedcParams<Real>{}));
            stedc<B, Real>(*q, d, e, eigs_stedc, ws.to_span(), JobType::EigenVectors, StedcParams<Real>{}, eigvects);
        }
        q->wait();

        auto call_lapack_variant = [&](Vector<Real>& out, const char* func_name, 
                                        int (*func_ptr)(int, Real*, Real*)) {
            const Real nan_val = std::numeric_limits<Real>::quiet_NaN();
            for (int b = 0; b < cur_batch; ++b) {
                auto d_work = std::vector<Real>(static_cast<std::size_t>(n));
                auto e_work = std::vector<Real>(static_cast<std::size_t>(std::max(0, n - 1)));
                for (int i = 0; i < n; ++i) {
                    d_work[static_cast<std::size_t>(i)] = d(i, b);
                    if (i < n - 1) e_work[static_cast<std::size_t>(i)] = e(i, b);
                }
                const int info = func_ptr(n, d_work.data(), e_work.data());
                for (int i = 0; i < n; ++i) {
                    out(i, b) = (info == 0) ? d_work[static_cast<std::size_t>(i)] : nan_val;
                }
                if (info != 0) std::cerr << func_name << " failed (info=" << info << ")\n";
            }
        };

        if (run_netlib_steqr) call_lapack_variant(eigs_netlib_steqr, "LAPACKE_xsteqr", call_lapack_steqr<Real>);
        if (run_netlib_sterf) call_lapack_variant(eigs_netlib_sterf, "LAPACKE_xsterf", call_lapack_sterf<Real>);
        if (run_netlib_stedc) call_lapack_variant(eigs_netlib_stedc, "LAPACKE_xstedc", call_lapack_stedc<Real>);

        const auto ref_eigs = netlib_ref_eigs<double>(dense_A.view());
        UnifiedVector<float> ref_eigs_f;
        if constexpr (std::is_same_v<Real, float>) {
            if (run_netlib32) {
                ref_eigs_f = netlib_ref_eigs<float>(dense_A.view());
            }
        }
        const auto conds = cond<B>(*q, dense_A.view(), NormType::Spectral);
        q->wait();

        for (int b = 0; b < cur_batch; ++b) {
            std::vector<double> ref_vals(static_cast<std::size_t>(n));
            const double cond = static_cast<double>(conds[b]);
            const double log10_cond = std::log10(std::max(cond, 1e-300));
            for (int i = 0; i < n; ++i) {
                const double ref = ref_eigs[static_cast<std::size_t>(i + b * n)];
                ref_vals[static_cast<std::size_t>(i)] = ref;
            }
            std::sort(ref_vals.begin(), ref_vals.end());

            auto write_row = [&](const VectorView<Real>& eigs, const char* impl_name) {
                std::vector<double> est_vals(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    est_vals[static_cast<std::size_t>(i)] = static_cast<double>(eigs(i, b));
                }
                std::sort(est_vals.begin(), est_vals.end());

                double max_rel = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double r = ref_vals[static_cast<std::size_t>(i)];
                    const double denom = std::max(1.0, std::abs(r));
                    const double rel = std::abs(est_vals[static_cast<std::size_t>(i)] - r) / denom;
                    if (std::isfinite(rel)) {
                        max_rel = std::max(max_rel, rel);
                    }
                }

                const double log10_rel = std::log10(std::max(max_rel, 1e-300));

                out << (sample_id + b) << ","
                    << n << ","
                    << impl_name << ","
                    << opt.backend << ","
                    << opt.dtype << ","
                    << target_log10s[static_cast<std::size_t>(b)] << ","
                    << cond << ","
                    << log10_cond << ","
                    << max_rel << ","
                    << log10_rel << "\n";
            };

            if (run_steqr) {
                write_row(VectorView<Real>(eigs_steqr), "steqr");
            }
            if (run_cta) {
                write_row(VectorView<Real>(eigs_cta), cta_impl_name);
            }
            if (run_cuda_syev) {
                write_row(VectorView<Real>(eigs_cuda), "cuda_syev");
            }
            if (run_netlib_steqr) {
                write_row(VectorView<Real>(eigs_netlib_steqr), "netlib_steqr");
            }
            if (run_netlib_sterf) {
                write_row(VectorView<Real>(eigs_netlib_sterf), "netlib_sterf");
            }
            if (run_netlib_stedc) {
                write_row(VectorView<Real>(eigs_netlib_stedc), "netlib_stedc");
            }
            if (run_stedc) {
                write_row(VectorView<Real>(eigs_stedc), "stedc");
            }
            if constexpr (std::is_same_v<Real, float>) {
                if (run_netlib32) {
                    VectorView<float> eigs32(ref_eigs_f.to_span(), /*size=*/n, /*batch_size=*/cur_batch, /*inc=*/1, /*stride=*/n);
                    write_row(eigs32, "netlib_syev32");
                }
            }
        }

            sample_id += cur_batch;
        }
    }

    return 0;
#endif
}

int dispatch(const Options& opt) {
    const std::string& dtype = opt.dtype;
    if (dtype != "float" && dtype != "double") {
        std::cerr << "Unsupported dtype for steqr accuracy: " << opt.dtype << " (use float/double)\n";
        return 1;
    }

    if (opt.backend == "CUDA") {
#if BATCHLAS_HAS_CUDA_BACKEND
        if (dtype == "float") return run_accuracy<Backend::CUDA, float>(opt);
        return run_accuracy<Backend::CUDA, double>(opt);
#else
        std::cerr << "CUDA backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "ROCM") {
#if BATCHLAS_HAS_ROCM_BACKEND
        if (dtype == "float") return run_accuracy<Backend::ROCM, float>(opt);
        return run_accuracy<Backend::ROCM, double>(opt);
#else
        std::cerr << "ROCM backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "MKL") {
#if BATCHLAS_HAS_MKL_BACKEND
        if (dtype == "float") return run_accuracy<Backend::MKL, float>(opt);
        return run_accuracy<Backend::MKL, double>(opt);
#else
        std::cerr << "MKL backend not available in this build\n";
        return 1;
#endif
    }
    if (opt.backend == "NETLIB") {
#if BATCHLAS_HAS_HOST_BACKEND
        if (dtype == "float") return run_accuracy<Backend::NETLIB, float>(opt);
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
    if (opt.dtype != "float" && opt.dtype != "double") {
        std::cerr << "Unsupported dtype for steqr accuracy: " << opt.dtype << " (use float/double)\n";
        return 1;
    }

    if (opt.log10_cond_max < opt.log10_cond_min) {
        std::swap(opt.log10_cond_min, opt.log10_cond_max);
    }

    if (opt.samples <= 0 || opt.n <= 0) {
        std::cerr << "Invalid --samples or --n value\n";
        return 1;
    }

    return dispatch(opt);
}
