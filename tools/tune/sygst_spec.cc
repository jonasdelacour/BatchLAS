// sygst for the tuner (docs/design/flat-kernel-selection.md §6.1): a Hermitian A0 (fill_random) and an
// SPD B factored once, untimed, by the public potrf under Auto, at every itype and triangle (table keys).
// Each run is checked on the default items by ||P - F Q F^H|| / ||P|| (MatrixView::similarity_residual,
// Check::factorization), with C the result mirrored: itype 1 P = A0, Q = C and F = L or U^H; itypes 2
// and 3 P = C, Q = A0 and F = L^H or U. A conjugate-transposed F is written into A (on the host, for the
// checked items only) once C is copied out; A is restored from A0 before every run. evidence: docs/perf/sygvd.md#sygst-measured

#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/sygst/choice.hh>
#include <tools/tune/cell_runner.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace I = ops::sygst;

template <class T>
struct SygstProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    Queue& q;
    int n, batch, ld, itype;
    Uplo uplo;
    std::size_t stride;
    UnifiedVector<T> A0, L, A, X;
    UnifiedVector<T*> pA0, pL, pA, pX;
    MV A0v, Lv, Av, Xv;
    UnifiedVector<std::int32_t> info;

    SygstProblem(Queue& q_, int n_, int batch_, int ld_pad, int itype_, Uplo uplo_)
        : q(q_), n(n_), batch(batch_), ld(n_ + ld_pad), itype(itype_), uplo(uplo_),
          stride(std::size_t(ld) * std::size_t(n_)), A0(stride * std::size_t(batch_)), L(stride * std::size_t(batch_)),
          A(stride * std::size_t(batch_)), X(stride * std::size_t(batch_)), pA0(std::size_t(batch_)),
          pL(std::size_t(batch_)), pA(std::size_t(batch_)), pX(std::size_t(batch_)),
          A0v(A0.data(), n, n, ld, int(stride), batch, pA0.data()), Lv(L.data(), n, n, ld, int(stride), batch, pL.data()),
          Av(A.data(), n, n, ld, int(stride), batch, pA.data()), Xv(X.data(), n, n, ld, int(stride), batch, pX.data()),
          info(std::size_t(batch_), 0) {
        (void)A0v.fill_random(q, true, 4242);
        (void)Lv.fill_random_spd(q, 12345);
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, potrf_buffer_size<kBackend, T>(q, Lv, uplo)));
        (void)potrf<kBackend, T>(q, Lv, uplo, fws.to_span(), info.to_span());
        (void)Lv.triangularize(q, uplo, Diag::NonUnit);
        q.wait();
        for (int b = 0; b < batch; ++b)
            if (info[std::size_t(b)] != 0) throw std::runtime_error("sygst input: potrf reported a non-SPD item");
    }
    // F: the stored factor (itype 1 Lower, itypes 2, 3 Upper) or its conjugate transpose, written into A.
    const MV& factor() {
        if ((itype == 1) == (uplo == Uplo::Lower)) return Lv;
        for (int b : A0v.default_items())
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const std::size_t o = std::size_t(b) * stride;
                    const T v = L[o + std::size_t(i) * ld + j];
                    if constexpr (is_std_complex_v<T>) A[o + std::size_t(j) * ld + i] = std::conj(v);
                    else A[o + std::size_t(j) * ld + i] = v;
                }
        return Av;
    }
    std::size_t workspace() { return sygst_buffer_size<kBackend, T>(q, Av, Lv, itype, uplo); }
    void reset() {
        (void)MV::copy(q, Av, A0v);
        q.wait();
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)sygst<kBackend, T>(q, Av, Lv, itype, uplo, ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        (void)MV::copy(q, Xv, Av);
        (void)Xv.hermitize(q, uplo);
        q.wait();
        if (itype == 1) return {MV::similarity_residual(q, A0v, factor(), Xv), 0};
        return {MV::similarity_residual(q, Xv, factor(), A0v), 0};
    }
};

class SygstSpec final : public OpSpec {
public:
    std::string op() const override { return "sygst"; }
    std::vector<std::string> key_names() const override { return {I::key_names.begin(), I::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(I::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        std::vector<std::string> n, b;
        for (int v : I::grid_n) n.push_back(std::to_string(v));
        for (int v : I::grid_batch) b.push_back(std::to_string(v));
        return {{"itype", {"1", "2", "3"}}, {"uplo", {"L", "U"}}, {"n", n}, {"batch", b}};
    }
    double bytes(const std::string& dtype, const CellKey& k) const override {
        const double n = double(key_int(k, "n"));
        return 4.0 * n * n * double(key_int(k, "batch")) * dtype_bytes(dtype);  // A0, L, A, X
    }
    FrontierPlan frontier() const override { return {1}; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            "src/sycl/kernel_attrs.hh",
            "src/ops/sygst/choice.hh",  // vendor (LAPACKE, host only) has no GPU source of its own
            // family: blocked
            "src/ops/sygst/sygst_native.hh",
            "src/ops/sygst/kernels/sygst_diag.cc",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/sygst/sygst.cc"
        // twosided calls the public trsm (itype 1) and trmm (2, 3): their native kernels count too.
        // family: twosided "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh"
        // blocked also calls the public gemm, her2k/syr2k and trmm.
        // family: blocked "src/ops/trsm/kernels/trsm_native.cc" "src/ops/trsm/trsm_native.hh" "src/ops/trsm/kernels/trsm_sg_left.cc" "src/ops/trsm/trsm_canonical.hh" "src/ops/trsm/choice.hh" "src/ops/her2k/her2k.cc" "src/ops/her2k/choice.hh" "src/ops/syr2k/syr2k.cc" "src/ops/syr2k/choice.hh" "src/ops/syr2k/kernels/syr2k_triangular_tiles.hh" "src/ops/trmm/trmm.cc" "src/ops/trmm/choice.hh" "src/ops/trmm/kernels/trmm_triangular_tiles.hh" "src/kernels/common/triangular_expand.hh" "src/kernels/common/triangular_tiles.hh" "src/ops/gemm/choice.hh" "src/ops/gemm/gemm.cc" "src/ops/gemm/kernels/gemm_kernels.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/sygst_spec.cc"; }
    std::vector<ArmOutcome> run_cell(const CellRequest& req, Queue& q) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            const Uplo uplo = *key_get(req.key, "uplo") == "U" ? Uplo::Upper : Uplo::Lower;
            SygstProblem<T> p(q, int(key_int(req.key, "n")), int(key_int(req.key, "batch")), req.ld_pad,
                              int(key_int(req.key, "itype")), uplo);
            const double tol = verify::bound<T>(verify::Check::factorization, p.n);
            if (req.mode == "race") return run_race<I::SygstChoice>("sygst", p, req, tol);
            return run_arms<I::SygstChoice>("sygst", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SygstSpec)
