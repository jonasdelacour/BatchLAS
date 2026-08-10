// 1. Getting started with the BatchLAS C++ interface
//
// The basics: what the build and the machine can do, running your first
// batched operation, and the conventions every other call follows.

#include <complex>
#include <cstddef>
#include <iostream>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using examples::print;
using examples::section;

namespace {

// ---------------------------------------------------------------------------
// The body of the example. `B` — the backend — is a *compile-time* template
// parameter, unlike Python's `backend=` string. Pick it once at the top of your
// own program and thread it through the same way.
// ---------------------------------------------------------------------------
template <Backend B>
void run(Queue& ctx) {
    // -----------------------------------------------------------------------
    // A single matrix product
    //
    // `Matrix<T>` owns its storage in unified (USM shared) memory, so you can
    // fill it from the host, hand it to a kernel and read the result back with
    // no explicit transfer. Storage is COLUMN-MAJOR: element (i, j) of batch
    // item b lives at data[b*stride + j*ld + i].
    // -----------------------------------------------------------------------
    section("A single matrix product");

    const double a_host[] = {1.0, 3.0, 2.0, 4.0};  // [[1, 2], [3, 4]], column-major
    const double b_host[] = {5.0, 7.0, 6.0, 8.0};  // [[5, 6], [7, 8]]

    // (data, rows, cols, ld, stride, batch_size) — the data is copied in.
    Matrix<double> A(a_host, 2, 2, /*ld=*/2, /*stride=*/0, /*batch_size=*/1);
    Matrix<double> Bm(b_host, 2, 2, 2, 0, 1);
    auto C = Matrix<double>::Zeros(2, 2);

    gemm<B>(ctx, A, Bm, C, /*alpha=*/1.0, /*beta=*/0.0, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    std::cout << "A * B =\n";
    C.print();

    // Element access on the host works directly — no copy-back step.
    print("C(0,0)", C(0, 0, 0));

    // -----------------------------------------------------------------------
    // The batching convention
    //
    // A Matrix carries a batch_size, and every routine takes the whole batch in
    // one call. There is no per-matrix loop and no separate "batched" entry
    // point. Item b starts at offset b*stride, which defaults to ld*cols.
    // -----------------------------------------------------------------------
    section("The batching convention");

    const int batch = 4;
    auto Ab = Matrix<double>::Random(3, 3, /*hermitian=*/false, batch, /*seed=*/42);
    auto Bb = Matrix<double>::Identity(3, batch);
    auto Cb = Matrix<double>::Zeros(3, 3, batch);

    gemm<B>(ctx, Ab, Bb, Cb, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    print("batch_size", Cb.batch_size());
    print("ld", Cb.ld());
    print("stride (defaults to ld*cols)", Cb.stride());
    std::cout << "A * I for batch item 2:\n";
    Cb.view()[2].print();

    // -----------------------------------------------------------------------
    // Scaling factors
    //
    // BLAS-style routines compute C <- alpha*op(A)*op(B) + beta*C. Because beta
    // multiplies the existing contents of C, C is an input as well as the
    // destination — there is no "allocate the output for me" overload.
    // -----------------------------------------------------------------------
    section("Scaling factors: C = alpha * A * B + beta * C");

    auto Cs = Matrix<double>::Ones(2, 2);
    gemm<B>(ctx, A, Bm, Cs, /*alpha=*/2.0, /*beta=*/0.5, Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();
    std::cout << "2*(A*B) + 0.5*ones =\n";
    Cs.print();

    // -----------------------------------------------------------------------
    // Views
    //
    // Routines take a `MatrixView` — a non-owning descriptor (pointer, rows,
    // cols, ld, stride, batch_size). `Matrix` converts implicitly, but a view
    // also lets you address part of a matrix, or one item of a batch, without
    // copying anything.
    // -----------------------------------------------------------------------
    section("Views");

    MatrixView<double> whole = Ab.view();
    MatrixView<double> one_item = whole[1];                     // batch item 1
    MatrixView<double> first_col = whole(Slice{}, Slice{0, 1});  // first column

    print("view batch_size", whole.batch_size());
    print("whole[1] batch_size", one_item.batch_size());
    print("first column shape", std::to_string(first_col.rows()) + "x" + std::to_string(first_col.cols()));
    print("a slice keeps the parent's ld", first_col.ld() == whole.ld());

    // -----------------------------------------------------------------------
    // The workspace contract
    //
    // This is the main thing the Python facade does for you. Routines that need
    // scratch memory never allocate it: you ask how much they need, allocate a
    // byte buffer once, and reuse it. The pattern is always
    //
    //     size_t bytes = foo_buffer_size<B>(ctx, ...the same arguments...);
    //     UnifiedVector<std::byte> workspace(bytes);
    //     foo<B>(ctx, ..., workspace.to_span());
    //
    // Here it is with syev, the symmetric eigensolver. It overwrites A with the
    // eigenvectors and writes ascending eigenvalues into `W`.
    // -----------------------------------------------------------------------
    section("The workspace contract");

    const int n = 6;
    auto S = Matrix<double>::Random(n, n, /*hermitian=*/true, 2, 7);
    UnifiedVector<double> W(static_cast<size_t>(n) * 2);

    const size_t bytes = syev_buffer_size<B>(ctx, S.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(bytes);
    syev<B>(ctx, S.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, workspace.to_span());
    ctx.wait();

    print("workspace bytes", bytes);
    examples::print_values("eigenvalues (ascending)", W.to_span(), n);

    // -----------------------------------------------------------------------
    // Scalar types
    //
    // Every routine is templated on the scalar type; float, double,
    // std::complex<float> and std::complex<double> are all instantiated. Note
    // that eigenvalues and singular values stay real for complex input — the
    // signatures say `Span<float_t<T>>` for those.
    // -----------------------------------------------------------------------
    section("Scalar types");

    auto Z = Matrix<std::complex<double>>::Random(4, 4, /*hermitian=*/true, 1, 3);
    UnifiedVector<double> Wz(4);  // real, even though Z is complex
    UnifiedVector<std::byte> ws_z(
        syev_buffer_size<B>(ctx, Z.view(), Wz.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev<B>(ctx, Z.view(), Wz.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_z.to_span());
    ctx.wait();
    examples::print_values("Hermitian eigenvalues", Wz.to_span(), 4);

    // -----------------------------------------------------------------------
    // Transposes
    //
    // Transpose::{NoTrans, Trans, ConjTrans} are folded into the kernel; the
    // transposed operand is never materialised.
    // -----------------------------------------------------------------------
    section("Transposes without materialising a copy");

    auto M = Matrix<double>::Random(2, 3, false, 1, 5);
    auto MtM = Matrix<double>::Zeros(3, 3);
    gemm<B>(ctx, M, M, MtM, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
    ctx.wait();
    std::cout << "M^T M =\n";
    MtM.print();

    // -----------------------------------------------------------------------
    // Asynchrony
    //
    // Routines enqueue work and return an `Event` rather than running it. Wait
    // on the Event, or on the Queue, before touching results from the host.
    // Consecutive calls on an in-order Queue are ordered for you, so one wait
    // at the end of a chain is enough.
    // -----------------------------------------------------------------------
    section("Asynchrony");

    auto D = Matrix<double>::Zeros(2, 2);
    gemm<B>(ctx, A, Bm, D, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    Event e = gemm<B>(ctx, A, D, D, 1.0, 0.0, Transpose::NoTrans, Transpose::NoTrans);
    e.wait();  // equivalent here to ctx.wait()

    print("queue is in-order", ctx.in_order());
    std::cout << "A * (A * B) =\n";
    D.print();

    // -----------------------------------------------------------------------
    // Where to go next
    // -----------------------------------------------------------------------
    section("Where to go next");
    std::cout << "  02  the rest of the dense BLAS, including heterogeneous batches\n"
                 "  06  the symmetric eigensolvers, the richest part of the library\n"
                 "  12  measuring the batching speed-up and picking between variants\n";
}

const char* device_type_name(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::GPU: return "gpu";
        case DeviceType::ACCELERATOR: return "accelerator";
        default: return "host";
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
    print("compiled backends", backends.empty() ? std::string("none") : backends);

    for (auto type : {DeviceType::GPU, DeviceType::CPU}) {
        for (const auto& dev : Device::get_devices(type)) {
            print("device", dev.get_name() + " (" + device_type_name(dev.type) + ")");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    examples::header("1. Getting started with the BatchLAS C++ interface");
    describe_build();

    // Backend selection is a compile-time decision, device selection a runtime
    // one, and the two have to agree: a GPU backend needs a GPU queue, NETLIB
    // needs the host. Prefer a GPU when this build and this machine both have
    // one. Pass "cpu" to force the host path.
    const bool force_cpu = (argc > 1 && std::string(argv[1]) == "cpu");

#if BATCHLAS_HAS_GPU_BACKEND
    if (!force_cpu && !Device::get_devices(DeviceType::GPU).empty()) {
        section("Queue and backend");
        Queue ctx("gpu", /*in_order=*/true);
        print("backend", std::string(examples::gpu_backend_name) + " (chosen at compile time)");
        print("device", ctx.device().get_name());
        run<examples::gpu_backend>(ctx);
        return 0;
    }
#else
    (void)force_cpu;
#endif

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    section("Queue and backend");
    Queue ctx("cpu", /*in_order=*/true);
    print("backend", std::string("NETLIB (chosen at compile time)"));
    print("device", ctx.device().get_name());
    run<Backend::NETLIB>(ctx);
    return 0;
#else
    std::cout << "\nNo usable backend/device combination in this build.\n";
    return 0;
#endif
}
