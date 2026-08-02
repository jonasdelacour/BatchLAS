// 1. Getting started with the BatchLAS C++ interface
//
// This example covers the basics: discovering what the build and the machine
// can do, running your first batched operation, and the conventions every
// other call in the library follows.
//
// Covered: backends and devices, Queue, Matrix/MatrixView, column-major
// storage, the batching convention, alpha/beta, the workspace contract,
// scalar types, transposes, and asynchrony.
//
// Every check prints [ok  ] or [FAIL], so a clean run doubles as a smoke test.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"

using namespace batchlas;
using examples::report;
using examples::report_check;
using examples::report_error;
using examples::section;

namespace {

// Largest absolute difference between a matrix and a host reference buffer.
// The reference buffer is column-major with leading dimension `rows`.
template <typename T>
double max_abs_diff(const Matrix<T, MatrixFormat::Dense>& A, const std::vector<T>& reference) {
    double worst = 0.0;
    for (int b = 0; b < A.batch_size(); ++b) {
        for (int j = 0; j < A.cols(); ++j) {
            for (int i = 0; i < A.rows(); ++i) {
                const T got = A(i, j, b);
                const T want = reference[static_cast<size_t>(b) * A.rows() * A.cols() + j * A.rows() + i];
                worst = std::max(worst, static_cast<double>(std::abs(got - want)));
            }
        }
    }
    return worst;
}

// Plain column-major reference product, computed on the host: C = alpha*A*B + beta*C.
template <typename T>
void reference_gemm(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int m, int n,
                    int k, T alpha = T(1), T beta = T(0)) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T acc = T(0);
            for (int p = 0; p < k; ++p) acc += A[p * m + i] * B[j * k + p];
            C[j * m + i] = alpha * acc + beta * C[j * m + i];
        }
    }
}

// ---------------------------------------------------------------------------
// The body of the example. `B` — the backend — is a *compile-time* template
// parameter here, unlike Python where `backend=` is a runtime string. Pick it
// once at the top of your own program and thread it through the same way.
// ---------------------------------------------------------------------------
template <Backend B>
void run(Queue& ctx) {
    // -----------------------------------------------------------------------
    // A single matrix product
    //
    // `Matrix<T>` owns its storage in unified (USM shared) memory, so you can
    // fill it from the host, hand it to a kernel, and read the result back
    // without an explicit transfer. Storage is COLUMN-MAJOR: element (i, j) of
    // batch item b lives at data[b*stride + j*ld + i].
    // -----------------------------------------------------------------------
    section("A single matrix product");

    const std::vector<double> a_host = {1.0, 3.0, 2.0, 4.0};  // [[1, 2], [3, 4]] column-major
    const std::vector<double> b_host = {5.0, 7.0, 6.0, 8.0};  // [[5, 6], [7, 8]] column-major

    // (data, rows, cols, ld, stride, batch_size) — the data is copied in.
    Matrix<double, MatrixFormat::Dense> A(a_host.data(), 2, 2, /*ld=*/2, /*stride=*/0, /*batch_size=*/1);
    Matrix<double, MatrixFormat::Dense> Bmat(b_host.data(), 2, 2, 2, 0, 1);
    auto C = Matrix<double, MatrixFormat::Dense>::Zeros(2, 2);

    gemm<B>(ctx, A, Bmat, C, /*alpha=*/1.0, /*beta=*/0.0, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    std::vector<double> c_ref(4, 0.0);
    reference_gemm(a_host, b_host, c_ref, 2, 2, 2);
    report_error("gemm error vs reference", max_abs_diff(C, c_ref), 1e-12);

    // Element access on the host works directly — no copy-back step.
    report_check("C(0,0) == 19", C(0, 0, 0) == 19.0);

    // -----------------------------------------------------------------------
    // The batching convention
    //
    // A Matrix carries a batch_size. Every routine takes the whole batch in one
    // call; there is no per-matrix loop and no separate "batched" entry point.
    // Batch item b starts at offset b*stride, which defaults to ld*cols.
    // -----------------------------------------------------------------------
    section("The batching convention");

    const int batch = 3;
    std::vector<double> batch_a(4 * batch);
    std::vector<double> batch_b(4 * batch);
    for (int b = 0; b < batch; ++b) {
        for (int e = 0; e < 4; ++e) {
            batch_a[b * 4 + e] = (b + 1) * a_host[e];  // A, 2A, 3A
            batch_b[b * 4 + e] = b_host[e];
        }
    }

    Matrix<double, MatrixFormat::Dense> Ab(batch_a.data(), 2, 2, /*ld=*/2, /*stride=*/4, /*batch_size=*/batch);
    Matrix<double, MatrixFormat::Dense> Bb(batch_b.data(), 2, 2, 2, 4, batch);
    auto Cb = Matrix<double, MatrixFormat::Dense>::Zeros(2, 2, batch);

    gemm<B>(ctx, Ab, Bb, Cb, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    std::vector<double> cb_ref(4 * batch, 0.0);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> ai(batch_a.begin() + b * 4, batch_a.begin() + (b + 1) * 4);
        std::vector<double> bi(batch_b.begin() + b * 4, batch_b.begin() + (b + 1) * 4);
        std::vector<double> ci(4, 0.0);
        reference_gemm(ai, bi, ci, 2, 2, 2);
        std::copy(ci.begin(), ci.end(), cb_ref.begin() + b * 4);
    }

    report("batched shape", std::to_string(Cb.batch_size()) + " x " + std::to_string(Cb.rows()) + " x " +
                                std::to_string(Cb.cols()));
    report_error("batched gemm error", max_abs_diff(Cb, cb_ref), 1e-12);
    report_check("stride defaults to ld*cols", Cb.stride() == Cb.ld() * Cb.cols());

    // -----------------------------------------------------------------------
    // Scaling factors
    //
    // BLAS-style routines compute C <- alpha*op(A)*op(B) + beta*C. Because beta
    // multiplies the existing contents of C, C is an input as well as the
    // destination — there is no "allocate the output for me" overload.
    // -----------------------------------------------------------------------
    section("Scaling factors: C = alpha * A * B + beta * C");

    auto Cs = Matrix<double, MatrixFormat::Dense>::Ones(2, 2);
    std::vector<double> cs_ref(4, 1.0);
    gemm<B>(ctx, A, Bmat, Cs, /*alpha=*/2.0, /*beta=*/0.5, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();
    reference_gemm(a_host, b_host, cs_ref, 2, 2, 2, /*alpha=*/2.0, /*beta=*/0.5);
    report_error("alpha/beta error", max_abs_diff(Cs, cs_ref), 1e-12);

    // -----------------------------------------------------------------------
    // Views: MatrixView and slicing
    //
    // Routines take a MatrixView — a non-owning descriptor (pointer, rows,
    // cols, ld, stride, batch_size). Matrix converts to one implicitly, so
    // passing a Matrix works, but a view lets you operate on part of a matrix,
    // or on a single item of a batch, without copying anything.
    // -----------------------------------------------------------------------
    section("Views and slicing");

    MatrixView<double, MatrixFormat::Dense> Ab_view = Ab.view();
    report_check("view shares storage", Ab_view.data_ptr() == Ab.data().data());

    MatrixView<double, MatrixFormat::Dense> second = Ab_view[1];  // batch item 1
    report_check("Ab_view[1] is one matrix", second.batch_size() == 1 && second.rows() == 2);
    report_check("Ab_view[1] sees 2*A", second.at(0, 0) == 2.0 * a_host[0]);

    MatrixView<double, MatrixFormat::Dense> first_col = Ab_view(Slice{}, Slice{0, 1});
    report_check("column slice keeps ld", first_col.cols() == 1 && first_col.ld() == Ab_view.ld());

    // -----------------------------------------------------------------------
    // The workspace contract
    //
    // This is the main thing the Python facade does for you. Routines that need
    // scratch memory never allocate it: you ask how much they need, allocate a
    // byte buffer once, and reuse it across calls. The pattern is always
    //
    //     size_t bytes = foo_buffer_size<B>(ctx, ...same arguments...);
    //     UnifiedVector<std::byte> workspace(bytes);
    //     foo<B>(ctx, ..., workspace.to_span());
    //
    // Here it is with syev, the symmetric eigensolver. syev overwrites A with
    // the eigenvectors and writes ascending eigenvalues into `W`.
    // -----------------------------------------------------------------------
    section("The workspace contract");

    const int n = 6;
    UnifiedVector<double> diag(n);
    for (int i = 0; i < n; ++i) diag[i] = static_cast<double>(i + 1);
    auto D = Matrix<double, MatrixFormat::Dense>::Diagonal(diag.to_span(), /*batch_size=*/2);

    UnifiedVector<double> W(static_cast<size_t>(n) * 2);
    const size_t bytes = syev_buffer_size<B>(ctx, D.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower);
    report("syev workspace bytes", bytes);

    UnifiedVector<std::byte> workspace(bytes);
    syev<B>(ctx, D.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, workspace.to_span());
    ctx.wait();

    double eig_error = 0.0;
    for (int b = 0; b < 2; ++b) {
        for (int i = 0; i < n; ++i) {
            eig_error = std::max(eig_error, std::abs(W[static_cast<size_t>(b) * n + i] - diag[i]));
        }
    }
    report_error("eigenvalues of diag(1..6)", eig_error, 1e-10);

    // The same buffer can be reused for any call whose requirement it covers.
    auto D2 = Matrix<double, MatrixFormat::Dense>::Diagonal(diag.to_span(), 2);
    syev<B>(ctx, D2.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, workspace.to_span());
    ctx.wait();
    report_check("workspace is reusable", std::abs(W[0] - diag[0]) < 1e-10);

    // -----------------------------------------------------------------------
    // Transposes
    //
    // Transpose::{NoTrans, Trans, ConjTrans} are folded into the kernel — the
    // transposed operand is never materialised.
    // -----------------------------------------------------------------------
    section("Transposes without materialising a copy");

    const std::vector<double> m_host = {0.0, 3.0, 1.0, 4.0, 2.0, 5.0};  // 2x3, column-major
    Matrix<double, MatrixFormat::Dense> M(m_host.data(), 2, 3, /*ld=*/2, 0, 1);
    auto MtM = Matrix<double, MatrixFormat::Dense>::Zeros(3, 3);

    gemm<B>(ctx, M, M, MtM, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
    ctx.wait();

    std::vector<double> mtm_ref(9, 0.0);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            double acc = 0.0;
            for (int p = 0; p < 2; ++p) acc += m_host[i * 2 + p] * m_host[j * 2 + p];
            mtm_ref[j * 3 + i] = acc;
        }
    }
    report_error("trans_a error", max_abs_diff(MtM, mtm_ref), 1e-12);

    // -----------------------------------------------------------------------
    // Asynchrony
    //
    // Every routine returns an Event and enqueues work rather than running it.
    // Wait on the Event, or on the Queue, before touching the results from the
    // host. Consecutive calls on an in-order Queue are ordered for you, so you
    // only need one wait at the end of a chain.
    // -----------------------------------------------------------------------
    section("Asynchrony");

    auto Cchain = Matrix<double, MatrixFormat::Dense>::Zeros(2, 2);
    gemm<B>(ctx, A, Bmat, Cchain, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    Event e = gemm<B>(ctx, A, Cchain, Cchain, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    e.wait();  // equivalent here to ctx.wait()

    std::vector<double> chain_ref(4, 0.0);
    reference_gemm(a_host, b_host, chain_ref, 2, 2, 2);
    std::vector<double> chain_ref2(4, 0.0);
    reference_gemm(a_host, chain_ref, chain_ref2, 2, 2, 2);
    report_error("chained gemm error", max_abs_diff(Cchain, chain_ref2), 1e-12);
    report_check("queue is in-order", ctx.in_order());
}

// ---------------------------------------------------------------------------
// Scalar types. Every routine is templated on the scalar type; float, double,
// std::complex<float> and std::complex<double> are instantiated for each
// backend. This runs the same gemm for all four.
// ---------------------------------------------------------------------------
template <Backend B, typename T>
void check_scalar_type(Queue& ctx, const std::string& name) {
    const T two = T(2);
    auto I = Matrix<T, MatrixFormat::Dense>::Identity(3);
    auto Out = Matrix<T, MatrixFormat::Dense>::Zeros(3, 3);

    gemm<B>(ctx, I, I, Out, two, T(0), Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    double worst = 0.0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const T want = (i == j) ? two : T(0);
            worst = std::max(worst, static_cast<double>(std::abs(Out(i, j, 0) - want)));
        }
    }
    report_error(name + " gemm error", worst, 1e-6);
}

template <Backend B>
void run_scalar_types(Queue& ctx) {
    section("Supported scalar types");
    check_scalar_type<B, float>(ctx, "float");
    check_scalar_type<B, double>(ctx, "double");
    check_scalar_type<B, std::complex<float>>(ctx, "complex<float>");
    check_scalar_type<B, std::complex<double>>(ctx, "complex<double>");
}

const char* device_type_name(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::GPU: return "gpu";
        case DeviceType::ACCELERATOR: return "accelerator";
        case DeviceType::HOST: return "host";
        default: return "unknown";
    }
}

void describe_build() {
    section("What this build can do");

    std::string backends;
    auto add = [&backends](const char* name) {
        if (!backends.empty()) backends += ", ";
        backends += name;
    };
#if BATCHLAS_HAS_HOST_BACKEND
    add("NETLIB (host BLAS/LAPACK)");
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    add("CUDA (cuBLAS/cuSOLVER)");
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    add("ROCM (rocBLAS/rocSOLVER)");
#endif
#if BATCHLAS_HAS_MKL_BACKEND
    add("MKL (oneMKL)");
#endif
    report("compiled backends", backends.empty() ? std::string("none") : backends);

    for (auto type : {DeviceType::GPU, DeviceType::CPU}) {
        for (const auto& dev : Device::get_devices(type)) {
            report("device", dev.get_name() + " (type=" + device_type_name(dev.type) + ")");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    examples::header("1. Getting started with the BatchLAS C++ interface");
    describe_build();

    // Backend selection is a compile-time decision, device selection a runtime
    // one, and the two have to agree: a GPU backend needs a GPU queue, NETLIB
    // needs the host. Prefer a GPU backend when this build and this machine
    // both have one, and fall back to the host otherwise.
    // Pass "cpu" on the command line to force the host path.
    const bool force_cpu = (argc > 1 && std::string(argv[1]) == "cpu");
    const bool have_gpu = !force_cpu && !Device::get_devices(DeviceType::GPU).empty();

#if BATCHLAS_HAS_GPU_BACKEND
#if BATCHLAS_HAS_CUDA_BACKEND
    constexpr Backend gpu_backend = Backend::CUDA;
    const char* gpu_backend_name = "CUDA";
#elif BATCHLAS_HAS_ROCM_BACKEND
    constexpr Backend gpu_backend = Backend::ROCM;
    const char* gpu_backend_name = "ROCM";
#else
    constexpr Backend gpu_backend = Backend::MKL;
    const char* gpu_backend_name = "MKL";
#endif
    if (have_gpu) {
        section("Queue and backend");
        Queue ctx("gpu", /*in_order=*/true);
        report("backend", std::string(gpu_backend_name) + " (compile-time)");
        report("device", ctx.device().get_name());
        run<gpu_backend>(ctx);
        run_scalar_types<gpu_backend>(ctx);
        return examples::exit_code();
    }
#endif

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    section("Queue and backend");
    Queue ctx("cpu", /*in_order=*/true);
    report("backend", std::string("NETLIB (compile-time)"));
    report("device", ctx.device().get_name());
    (void)have_gpu;
    run<Backend::NETLIB>(ctx);
    run_scalar_types<Backend::NETLIB>(ctx);
    return examples::exit_code();
#else
    (void)have_gpu;
    std::cout << "\nNo usable backend/device combination in this build.\n";
    return 0;
#endif
}
