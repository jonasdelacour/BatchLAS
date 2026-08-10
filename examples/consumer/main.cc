// A batched gemm, checked against a hand-computed reference, from a standalone
// consumer project that knows nothing about BatchLAS's build tree.
//
// This file exists as much to document the data-in/data-out contract as to
// compute anything, because three things below are load-bearing and are stated
// nowhere else:
//
//   1. LAYOUT. Dense matrices are COLUMN-MAJOR and batched by a stride. Element
//      (i, j) of batch item b lives at
//
//          view.data_ptr()[b * view.stride() + j * view.ld() + i]
//
//      `ld()` is the leading dimension (>= rows, chosen by the library when you
//      let it allocate) and `stride()` is the element distance between batch
//      items (>= ld * cols). Never assume ld == rows or stride == rows * cols:
//      the allocating constructor is free to pad, and every loop below indexes
//      through the accessors instead of hardcoding the packing.
//
//   2. MEMORY. Pointers handed to a MatrixView must be device-accessible (USM).
//      A plain `std::vector<float>` compiles fine and then aborts the process
//      with CUDA_ERROR_ILLEGAL_ADDRESS on a GPU backend, while returning the
//      right answer on a CPU backend -- i.e. it survives a CPU prototype. The
//      owning `Matrix` allocates USM shared memory, so filling it from the host
//      as we do here is fine, and no explicit copy is needed either way.
//
//   3. SYNCHRONISATION. Calls are asynchronous. ctx.wait() before reading the
//      output, or you read whatever was in the buffer beforehand (typically
//      zeros, i.e. a silent wrong answer rather than a crash).
//
// See README.md next to this file for the build recipe; the compiler choice and
// LD_LIBRARY_PATH are load-bearing too.

#include <blas/linalg.hh>

#include <cmath>
#include <cstdio>
#include <exception>

using batchlas::Matrix;
using batchlas::MatrixFormat;

namespace {

constexpr int kN = 3;      // square, for brevity; gemm handles m/n/k
constexpr int kBatch = 4;
constexpr float kAlpha = 2.0f;

// Deterministic, batch-dependent operands. Distinct per (i, j, b) so that a
// transposed, packed-instead-of-padded or batch-collapsed read shows up as a
// mismatch rather than as a coincidence.
float a_value(int i, int j, int b) { return float(1 + i + 2 * j + 3 * b); }
float b_value(int i, int j, int b) { return float(2 + 3 * i + j - b); }

// The one place the layout is spelled out. Everything else goes through it.
template <typename View>
float& at(const View& v, int i, int j, int b) {
    return v.data_ptr()[b * v.stride() + j * v.ld() + i];
}

}  // namespace

int main() try {
    // Owning, USM-backed operands. (rows, cols, batch_size); ld and stride are
    // chosen by the library, which is exactly why we read them back below.
    Matrix<float, MatrixFormat::Dense> A(kN, kN, kBatch);
    Matrix<float, MatrixFormat::Dense> B(kN, kN, kBatch);
    Matrix<float, MatrixFormat::Dense> C(kN, kN, kBatch);

    auto a = A.view();
    auto b = B.view();
    auto c = C.view();

    for (int k = 0; k < kBatch; ++k) {
        for (int j = 0; j < kN; ++j) {
            for (int i = 0; i < kN; ++i) {
                at(a, i, j, k) = a_value(i, j, k);
                at(b, i, j, k) = b_value(i, j, k);
                // beta defaults to 0, so C is written not accumulated -- but the
                // allocation is uninitialised, and 0 * NaN is NaN on some paths.
                at(c, i, j, k) = 0.0f;
            }
        }
    }

    Queue ctx(Device::default_device());  // backend resolved from the device
    std::printf("backend = %d\n", static_cast<int>(ctx.backend()));

    // C := alpha * A * B  (+ beta * C, beta = 0). Option-struct spelling; the
    // scalar type is fixed by A, so the designated initialiser needs no cast
    // beyond matching float.
    batchlas::gemm(ctx, a, b, c, {.alpha = kAlpha});
    ctx.wait();  // REQUIRED before reading c; without it you read the zeros above

    double max_err = 0.0;
    for (int k = 0; k < kBatch; ++k) {
        for (int j = 0; j < kN; ++j) {
            for (int i = 0; i < kN; ++i) {
                float ref = 0.0f;
                for (int p = 0; p < kN; ++p) {
                    ref += a_value(i, p, k) * b_value(p, j, k);
                }
                ref *= kAlpha;
                max_err = std::fmax(max_err, std::fabs(double(at(c, i, j, k)) - double(ref)));
            }
        }
    }

    const bool ok = max_err < 1e-4;
    std::printf("ld = %d, stride = %d (rows = %d, cols = %d, batch = %d)\n",
                c.ld(), c.stride(), c.rows(), c.cols(), c.batch_size());
    std::printf("max err = %g -> %s\n", max_err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    // No device, no adapter, no driver: nothing to say about the packaging, so
    // report CTest's skip code rather than a failure. consumer_test.sh maps it.
    std::fprintf(stderr, "consumer example could not run: %s\n", e.what());
    return 77;
}
