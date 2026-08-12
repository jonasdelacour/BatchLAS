// A batched gemm, checked against a hand-computed reference, from a standalone
// consumer project that knows nothing about BatchLAS's build tree.
//
// Three rules govern everything below. docs/cpp-api.md states them in full.
//
//   1. LAYOUT. Dense matrices are COLUMN-MAJOR and batched by a stride. Element
//      (i, j) of batch item b lives at
//
//          view.data_ptr()[b * view.stride() + j * view.ld() + i]
//
//      `ld()` is the leading dimension and `stride()` is the element distance
//      between batch items. Matrix(rows, cols, batch) packs: ld() == rows and
//      stride() == rows * cols. A matrix built over someone else's buffer can
//      carry a larger ld or stride, so index through the accessors, as the loops
//      below do, and the same code reads both.
//
//   2. MEMORY. Pointers handed to a MatrixView must be device-accessible (USM).
//      Passing a plain `std::vector<float>` compiles, and the entry points that
//      take their backend from the Queue -- the ones used here -- then throw
//      std::invalid_argument naming the argument. On a CPU backend host memory
//      is what the kernels read and nothing is rejected, so run the check on the
//      device you will ship on. The owning `Matrix` allocates USM shared memory,
//      so filling it from the host as we do here is fine on every backend, and
//      no explicit copy is needed either way.
//
//   3. SYNCHRONISATION. Calls are asynchronous. ctx.wait() before reading the
//      output, or you read whatever was in the buffer beforehand. A fresh
//      Matrix is uninitialised, so that is a silent wrong answer, not a crash.
//
// See README.md next to this file for the build recipe; the compiler choice and
// LD_LIBRARY_PATH are load-bearing too.

// <batchlas.hh> is the umbrella header; individual headers are reachable as
// <batchlas/...>. Both spellings work.
#include <batchlas.hh>
#include <batchlas/blas/linalg.hh>

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
// transposed or batch-collapsed read shows up as a mismatch rather than as a
// coincidence.
float a_value(int i, int j, int b) { return float(1 + i + 2 * j + 3 * b); }
float b_value(int i, int j, int b) { return float(2 + 3 * i + j - b); }

// The one place the layout is spelled out. Everything else goes through it.
template <typename View>
float& at(const View& v, int i, int j, int b) {
    return v.data_ptr()[b * v.stride() + j * v.ld() + i];
}

}  // namespace

int main() try {
    // Owning, USM-backed operands. (rows, cols, batch_size) packs them: ld is
    // kN and stride is kN * kN, which the printout at the end confirms.
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
                // A fresh Matrix is uninitialised; zero C before the call.
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
    // No device, no adapter, no driver. 77 is CTest's skip code.
    std::fprintf(stderr, "consumer example could not run: %s\n", e.what());
    return 77;
}
