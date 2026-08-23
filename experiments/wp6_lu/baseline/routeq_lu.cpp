// WP6 question 4: what will a blocked LU's trailing update and triangular solve
// ROUTE to, per scalar type, vendor-present and vendor-free?
//
// A right-looking blocked getrf at panel start j0, block width nb, on an N x N
// parent, with m1 = N - j0 the panel height and n2 = N - j0 - nb the trailing
// width, issues exactly three shaped operations after the panel getf2:
//
//   S   laswp on the trailing columns             (no route -- a kernel of ours)
//   T   U12 = L11^-1 A12   trsm Left/Lower/Unit   order nb, nrhs n2
//   G   A22 -= L21 U12     gemm NN                m = m1-nb, n = n2, k = nb
//
// G is where the arithmetic is, and it is the SHALLOW-K NN shape -- k = nb is
// tens, m and n are hundreds. That is not the square shape a gemm benchmark
// measures, and it is the reason to ask rather than assume.
//
// Both halves are printed: the ROUTE (RouteTable, pure metadata) and the
// KernelVariant (the selector inside the native gemm, which reads POINTERS and
// leading dimensions, so it must be asked on real sub-views of a real parent).
//
// SUB-VIEWS ARE BUILT EXPLICITLY with the parent ld AND stride AND batch. Never
// operator()(Slice,Slice): matrix.hh:1140 propagates the parent pointer ARRAY,
// a known open bug, and every blocked driver in this tree works around it the
// same way.
//
// THE PARENT IS ALLOCATED AT THE REAL BATCH, and that is not a detail.
// select_kernel_variant reads more of the view than "m, n, k and a transpose":
// the wide-scalar relaxation at gemm_kernels.cc:695-707 is gated on a CTA COUNT
// that multiplies by A.batch_size(), and can_use_64x64_k16_wide_fast_path
// (register_64x64_k16_wide.hh:201-207) reads data_ptr(), ld() AND stride(). A
// first version of this program used batch = 1 parents and reported Tiled16 for
// every complex trailing update -- which is the answer the brief predicted, and
// it was an artefact of the harness, not the library.
#include <batchlas/blas/dispatch/route_gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/trsm_route.hh"
#include "src/sycl/gemm_kernels.hh"

#include <complex>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace batchlas;
using namespace batchlas::dispatch;

static constexpr Backend BE = Backend::CUDA;

static const char* orig(Origin o) {
    switch (o) { case Origin::Auto: return "Auto"; case Origin::Native: return "Native";
                 case Origin::Vendor: return "Vendor"; }
    return "?";
}
static const char* alg(Algorithm a) {
    switch (a) { case Algorithm::Auto: return "Auto"; case Algorithm::Direct: return "Direct";
                 case Algorithm::RegisterTiled: return "RegisterTiled"; case Algorithm::CTA: return "CTA";
                 case Algorithm::Blocked: return "Blocked"; default: return "other"; }
}
static const char* kv(sycl_gemm::KernelVariant v) {
    using K = sycl_gemm::KernelVariant;
    switch (v) {
        case K::Direct: return "Direct";
        case K::Tiled16: return "Tiled16";
        case K::Tiled32x32Register: return "Tiled32x32Register";
        case K::Tiled64x64Register: return "Tiled64x64Register";
        case K::Tiled64x64RegisterK16: return "Tiled64x64RegisterK16";
        case K::Tiled64x64RegisterK16Wide: return "Tiled64x64RegisterK16Wide";
        case K::Tiled128x32RegisterK16: return "Tiled128x32RegisterK16";
        case K::Tiled128x32RegisterK32: return "Tiled128x32RegisterK32";
        case K::Tiled128x32RegisterK32S2U1Aligned: return "Tiled128x32K32S2U1Aligned";
        case K::Tiled128x32RegisterK32S2U1Generic: return "Tiled128x32K32S2U1Generic";
        case K::Tiled128x32RegisterK32S1U1: return "Tiled128x32K32S1U1";
        case K::Tiled128x32RegisterK32S2U1: return "Tiled128x32K32S2U1";
        case K::Tiled128x32RegisterK32S2U2: return "Tiled128x32K32S2U2";
        case K::Tiled128x64RegisterK32Large: return "Tiled128x64K32Large";
        case K::Tiled128x64RegisterK32LargeU2: return "Tiled128x64K32LargeU2";
        case K::Tiled128x128RegisterK8: return "Tiled128x128RegisterK8";
        case K::Tiled32x128RegisterK16: return "Tiled32x128RegisterK16";
        default: return "other";
    }
}

struct Cell { int N, nb, batch; };
static const std::vector<Cell> kCells = {
    {128, 32, 4096}, {256, 32, 2048}, {512, 64, 512}, {1024, 64, 128}, {2048, 64, 32}};

template <typename T>
static void emit(Queue& q, const char* tn) {
    for (const Cell& c : kCells) {
        const int ld = c.N;
        const size_t st = size_t(c.N) * size_t(c.N);
        // Real extents, real batch, real stride. Not filled: nothing below
        // dereferences the data, only its address and alignment are read.
        UnifiedVector<T> P(st * size_t(c.batch));
        auto sub = [&](int i0, int j0, int rows, int cols, int lda, T** pa) {
            return MatrixView<T, MatrixFormat::Dense>(
                P.data() + size_t(j0) * size_t(lda) + size_t(i0), rows, cols, lda,
                int(st), c.batch, pa);
        };
        // First panel, middle panel, and the SHORT FINAL panel -- the shape class
        // that produced the sy2sb silent failure.
        const int last_full = ((c.N - 1) / c.nb) * c.nb;
        const int js[3] = {0, (c.N / 2 / c.nb) * c.nb, last_full - c.nb};
        for (int p = 0; p < 3; ++p) {
            const int j0 = js[p];
            if (j0 < 0) continue;
            const int nb = c.nb;
            const int m1 = c.N - j0, n2 = c.N - j0 - nb;
            if (n2 <= 0) continue;
            const int gm = m1 - nb;
            if (gm <= 0) continue;

            // --- G: A22 -= L21 U12, NN, k = nb
            OpShape s;
            s.op = Op::gemm; s.scalar = scalar_kind_of<T>; s.backend = BE;
            s.m = gm; s.n = n2; s.k = nb; s.batch = c.batch;
            s.transA = Transpose::NoTrans; s.transB = Transpose::NoTrans;
            s.is_gpu = true;
            UnifiedVector<T*> pa(c.batch), pb(c.batch), pc(c.batch);
            auto A21 = sub(j0 + nb, j0, gm, nb, ld, pa.data());
            auto U12 = sub(j0, j0 + nb, nb, n2, ld, pb.data());
            auto A22 = sub(j0 + nb, j0 + nb, gm, n2, ld, pc.data());
            const auto var = sycl_gemm::select_kernel_variant<T>(
                A21, U12, A22, Transpose::NoTrans, Transpose::NoTrans);
            for (int va = 1; va >= 0; --va) {
                const Route r = resolve_gemm_route<T>(Route{}, s, va != 0);
                std::printf("G,%s,%d,%d,%d,%d,%d,%d,%d,%s,%s:%s,%s\n", tn,
                            c.N, nb, j0, c.batch, gm, n2, nb,
                            va ? "vendor_present" : "vendor_free",
                            orig(r.origin), alg(r.algo), kv(var));
            }

            // --- T: U12 = L11^-1 A12, Left/Lower/Unit, order nb, nrhs n2
            UnifiedVector<T*> pl(c.batch), pu(c.batch);
            auto L11 = sub(j0, j0, nb, nb, ld, pl.data());
            auto A12 = sub(j0, j0 + nb, nb, n2, ld, pu.data());
            for (int va = 1; va >= 0; --va) {
                const Route r = backend::trsm_route<T>(q, L11, A12, Side::Left, Uplo::Lower,
                                                       Transpose::NoTrans, Diag::Unit, va != 0);
                std::printf("T,%s,%d,%d,%d,%d,%d,%d,%d,%s,%s:%s,-\n", tn,
                            c.N, nb, j0, c.batch, nb, n2, nb,
                            va ? "vendor_present" : "vendor_free",
                            orig(r.origin), alg(r.algo));
            }
        }

        // --- getrs / getri: the two full-order solves, at nrhs = 1 and nrhs = N.
        for (int nrhs : {1, c.N}) {
            UnifiedVector<T*> pa(c.batch), pb(c.batch);
            auto A = sub(0, 0, c.N, c.N, ld, pa.data());
            UnifiedVector<T> B(size_t(c.N) * size_t(nrhs) * size_t(c.batch));
            MatrixView<T, MatrixFormat::Dense> Bv(B.data(), c.N, nrhs, c.N,
                                                 c.N * nrhs, c.batch, pb.data());
            for (int va = 1; va >= 0; --va) {
                const Route rl = backend::trsm_route<T>(q, A, Bv, Side::Left, Uplo::Lower,
                                                        Transpose::NoTrans, Diag::Unit, va != 0);
                const Route ru = backend::trsm_route<T>(q, A, Bv, Side::Left, Uplo::Upper,
                                                        Transpose::NoTrans, Diag::NonUnit, va != 0);
                std::printf("SOLVE_L,%s,%d,-,-,%d,%d,%d,%d,%s,%s:%s,-\n", tn,
                            c.N, c.batch, c.N, nrhs, c.N,
                            va ? "vendor_present" : "vendor_free", orig(rl.origin), alg(rl.algo));
                std::printf("SOLVE_U,%s,%d,-,-,%d,%d,%d,%d,%s,%s:%s,-\n", tn,
                            c.N, c.batch, c.N, nrhs, c.N,
                            va ? "vendor_present" : "vendor_free", orig(ru.origin), alg(ru.algo));
            }
        }
    }
}

int main() {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    std::printf("# level3_vendor_available=%d factorization_vendor_available=%d\n",
                int(dispatch::level3_vendor_available<BE>),
                int(dispatch::factorization_vendor_available<BE>));
    std::printf("op,type,N,nb,j0,batch,m,n,k,vendor,route,variant\n");
    emit<float>(*q, "float");
    emit<double>(*q, "double");
    emit<std::complex<float>>(*q, "cfloat");
    emit<std::complex<double>>(*q, "cdouble");
    return 0;
}
