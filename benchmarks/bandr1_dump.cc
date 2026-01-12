#include <blas/extensions.hh>
#include <blas/matrix.hh>

#include <batchlas/backend_config.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace batchlas;

namespace {

template <typename T>
inline void fill_lower_band_from_dense(const MatrixView<T, MatrixFormat::Dense>& A,
                                       MatrixView<T, MatrixFormat::Dense> AB,
                                       int n,
                                       int kd) {
    // AB is (kd+1) x n, lower band: AB(r,j) = A(j+r, j)
    for (int j = 0; j < n; ++j) {
        const int rmax = std::min(kd, n - 1 - j);
        for (int r = 0; r <= rmax; ++r) {
            AB(r, j, 0) = A(j + r, j, 0);
        }
        for (int r = rmax + 1; r <= kd; ++r) {
            AB(r, j, 0) = T(0);
        }
    }
}

inline void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --dump-dir <path> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --n <int>            Matrix size (default 64)\n";
    std::cerr << "  --kd <int>           Input semibandwidth (default 8)\n";
    std::cerr << "  --kd-work <int>      Working semibandwidth (default 0 => min(n-1, 3*kd))\n";
    std::cerr << "  --block-size <int>   BANDR1 panel size (default 16)\n";
    std::cerr << "  --d <int>            Diagonals to eliminate per sweep (default 0 => impl default)\n";
    std::cerr << "  --max-sweeps <int>   Max sweeps (default -1 => impl default)\n";
    std::cerr << "  --batch <int>        Batch size (default 1)\n";
    std::cerr << "  --seed <int>         RNG seed (default 123)\n";
    std::cerr << "  --type <str>         f32|f64|c64|c128 (default f32)\n";
    std::cerr << "  --device <str>       Queue device string (default gpu)\n";
    std::cerr << "  --out-order          Use out-of-order queue (default in-order)\n";
    std::cerr << "  --abw-only           Only dump ABw (faster / smaller output)\n";
    std::cerr << "  --dump-step          Enable BANDR1 dump hooks\n";
    std::cerr << "\n";
    std::cerr << "This program runs sytrd_band_reduction once and relies on\n";
    std::cerr << "BANDR1 dump hooks to write sweep_XXX/step_YYYY CSV outputs.\n";
}

inline bool is_flag(const char* s, const char* name) {
    return s && name && std::strcmp(s, name) == 0;
}

inline void set_env(const char* key, const std::string& value) {
    ::setenv(key, value.c_str(), 1);
}

template <Backend B, typename T>
int run_one(const std::string& dump_dir,
            int n,
            int kd,
            int kd_work,
            int block_size,
            int diagonals_per_sweep,
            int max_sweeps,
            int batch,
            int seed,
            const std::string& device,
            bool in_order,
            bool dump_step,
            bool abw_only) {
    // Drive dumping via env vars used by src/extensions/band_reduction.cc.
    set_env("BATCHLAS_DUMP_BANDR1_DIR", dump_dir);
    if (dump_step) {
        set_env("BATCHLAS_DUMP_BANDR1_STEP", "1");
    }
    if (abw_only) {
        set_env("BATCHLAS_DUMP_BANDR1_ABW_ONLY", "1");
    }
    // Default: dump batch 0 only (plotting expects a single batch unless asked).
    set_env("BATCHLAS_DUMP_BANDR1_BATCH", "0");

    auto q = std::make_shared<Queue>(device.c_str(), /*in_order=*/in_order);

    Matrix<T, MatrixFormat::Dense> A0 =
        Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, /*batch=*/1, /*seed=*/seed);

    // Enforce banded input.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (std::abs(i - j) > kd) {
                A0(i, j, 0) = T(0);
            }
        }
    }

    auto AB = Matrix<T, MatrixFormat::Dense>::Zeros(kd + 1, n, batch);
    {
        Matrix<T, MatrixFormat::Dense> AB0(kd + 1, n, /*batch=*/1);
        fill_lower_band_from_dense<T>(A0.view(), AB0.view(), n, kd);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int r = 0; r <= kd; ++r) {
                    AB(r, j, b) = AB0(r, j, 0);
                }
            }
        }
    }

    using Real = typename base_type<T>::type;
    auto dvec = Vector<Real>::zeros(n, batch);
    auto e = Vector<Real>::zeros(n > 0 ? n - 1 : 0, batch);
    auto tau = Vector<T>::zeros(n > 0 ? n - 1 : 0, batch);

    SytrdBandReductionParams params;
    params.block_size = block_size;
    params.d = diagonals_per_sweep;
    params.max_sweeps = max_sweeps;
    params.kd_work = kd_work;

    const size_t ws_bytes = sytrd_band_reduction_buffer_size<B, T>(*q,
                                                                   AB.view(),
                                                                   VectorView<Real>(dvec),
                                                                   VectorView<Real>(e),
                                                                   VectorView<T>(tau),
                                                                   Uplo::Lower,
                                                                   kd,
                                                                   params);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});
    sytrd_band_reduction<B, T>(*q, AB.view(), dvec, e, tau, Uplo::Lower, kd, ws.to_span(), params).wait();

    return 0;
}

} // namespace

int main(int argc, char** argv) {
#if !BATCHLAS_HAS_CUDA_BACKEND
    std::cerr << "bandr1_dump requires a GPU backend (CUDA) enabled build.\n";
    return 2;
#else
    std::string dump_dir;
    int n = 64;
    int kd = 8;
    int kd_work = 0;
    int block_size = 16;
    int d = 0;
    int max_sweeps = -1;
    int batch = 1;
    int seed = 123;
    std::string type = "f32";
    std::string device = "gpu";
    bool in_order = true;
    bool dump_step = true;
    bool abw_only = true;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };

        if (is_flag(a, "--help") || is_flag(a, "-h")) {
            usage(argv[0]);
            return 0;
        } else if (is_flag(a, "--dump-dir")) {
            dump_dir = need("--dump-dir");
        } else if (is_flag(a, "--n")) {
            n = std::stoi(need("--n"));
        } else if (is_flag(a, "--kd")) {
            kd = std::stoi(need("--kd"));
        } else if (is_flag(a, "--kd-work")) {
            kd_work = std::stoi(need("--kd-work"));
        } else if (is_flag(a, "--block-size")) {
            block_size = std::stoi(need("--block-size"));
        } else if (is_flag(a, "--d")) {
            d = std::stoi(need("--d"));
        } else if (is_flag(a, "--max-sweeps")) {
            max_sweeps = std::stoi(need("--max-sweeps"));
        } else if (is_flag(a, "--batch")) {
            batch = std::stoi(need("--batch"));
        } else if (is_flag(a, "--seed")) {
            seed = std::stoi(need("--seed"));
        } else if (is_flag(a, "--type")) {
            type = need("--type");
        } else if (is_flag(a, "--device")) {
            device = need("--device");
        } else if (is_flag(a, "--out-order")) {
            in_order = false;
        } else if (is_flag(a, "--in-order")) {
            in_order = true;
        } else if (is_flag(a, "--abw-only")) {
            abw_only = true;
        } else if (is_flag(a, "--dump-step")) {
            dump_step = true;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (dump_dir.empty()) {
        std::cerr << "--dump-dir is required\n";
        usage(argv[0]);
        return 2;
    }
    if (n <= 0 || kd <= 0 || kd >= n || block_size <= 0 || batch <= 0) {
        std::cerr << "Invalid sizes: n=" << n << " kd=" << kd << " block_size=" << block_size << " batch=" << batch
                  << "\n";
        return 2;
    }
    if (kd_work > 0 && kd_work < kd) {
        std::cerr << "Invalid --kd-work=" << kd_work << " (must be 0 or >= kd=" << kd << ")\n";
        return 2;
    }
    if (d < 0) {
        std::cerr << "Invalid --d=" << d << " (must be >= 0)\n";
        return 2;
    }

    if (type == "f32") return run_one<Backend::CUDA, float>(dump_dir, n, kd, kd_work, block_size, d, max_sweeps, batch, seed, device, in_order, dump_step, abw_only);
    if (type == "f64") return run_one<Backend::CUDA, double>(dump_dir, n, kd, kd_work, block_size, d, max_sweeps, batch, seed, device, in_order, dump_step, abw_only);
    if (type == "c64") return run_one<Backend::CUDA, std::complex<float>>(dump_dir, n, kd, kd_work, block_size, d, max_sweeps, batch, seed, device, in_order, dump_step, abw_only);
    if (type == "c128") return run_one<Backend::CUDA, std::complex<double>>(dump_dir, n, kd, kd_work, block_size, d, max_sweeps, batch, seed, device, in_order, dump_step, abw_only);

    std::cerr << "Unknown --type " << type << " (expected f32|f64|c64|c128)\n";
    return 2;
#endif
}
