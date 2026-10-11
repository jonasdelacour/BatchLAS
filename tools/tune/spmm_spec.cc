// spmm for the tuner (GPU tables only; the cpu tables stay transcribed): a square m x m CSR A with
// min(m, 8) sorted, scattered columns per row (a hashed start, stride m / r; items cycle through four
// patterns, item 0 and batch-1 differ), values, B and C0 from MatrixView::fill_random (complex parts
// both); alpha and beta with imaginary parts, beta != 0; B and C at ld = rows + ld_pad. trans=T times
// ConjTrans for a complex scalar (the key folds C into T). Verification is
// MatrixView::spmm_backward_error, |C - Cref| / (|alpha| |op(A)| |op(B)| + |beta| |C0|) componentwise on
// the default items (Check::blas, docs/design/verification.md). No nnz key (choice.hh): 8 per row sits
// between the lanczos (3) and LOBPCG (16) cells of docs/perf/spmm.md.

#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include "../../src/ops/spmm/choice.hh"
#include "cell_runner.hh"

#include <algorithm>
#include <complex>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace batchlas::tune {
namespace {

namespace P = ops::spmm;

constexpr int kVariants = 4;
constexpr int kNnzPerRow = 8;

inline int variant_of(int b, int batch) { return b == 0 ? 0 : b == batch - 1 ? 1 : 2 + (b & 1); }

Transpose trans_of(const std::string& w, bool complex) {
    if (w == "N") return Transpose::NoTrans;
    return complex ? Transpose::ConjTrans : Transpose::Trans;
}

template <class T>
struct SpmmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using SV = MatrixView<T, MatrixFormat::CSR>;
    Queue& q;
    Transpose ta, tb;
    int m, nrhs, batch, r, nnz, br, bc, ldb, ldc;
    std::size_t sb, sc;
    T alpha, beta;
    UnifiedVector<T> vals, B, C0, C;
    UnifiedVector<int> offs, cols;
    SV Av;
    MV Bv, C0v, Cv;

    SpmmProblem(Queue& q_, Transpose ta_, Transpose tb_, int m_, int nrhs_, int batch_, int pad)
        : q(q_), ta(ta_), tb(tb_), m(m_), nrhs(nrhs_), batch(batch_), r(std::min(m_, kNnzPerRow)), nnz(m_ * r),
          br(tb_ == Transpose::NoTrans ? m_ : nrhs_), bc(tb_ == Transpose::NoTrans ? nrhs_ : m_), ldb(br + pad),
          ldc(m_ + pad), sb(std::size_t(ldb) * std::size_t(bc)), sc(std::size_t(ldc) * std::size_t(nrhs_)),
          alpha(MV::make(1.25, -0.5)), beta(MV::make(0.75, 0.25)), vals(std::size_t(nnz) * std::size_t(batch_)),
          B(sb * std::size_t(batch_)), C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)),
          offs(std::size_t(m_ + 1) * std::size_t(batch_)), cols(std::size_t(nnz) * std::size_t(batch_)),
          Av(vals.data(), offs.data(), cols.data(), m_, m_, NonZeros{nnz}, nnz, m_ + 1, batch_),
          Bv(B.data(), br, bc, ldb, int(sb), batch_), C0v(C0.data(), m_, nrhs_, ldc, int(sc), batch_),
          Cv(C.data(), m_, nrhs_, ldc, int(sc), batch_) {
        std::vector<std::vector<int>> vc(kVariants, std::vector<int>(std::size_t(nnz)));
        const int step = std::max(1, m / r);
        for (int v = 0; v < kVariants; ++v)
            for (int i = 0; i < m; ++i) {  // a fixed hash of (v, i): the pattern is the cell's structure
                const int h = int((std::uint64_t(i) * 2654435761ULL + std::uint64_t(v) * 40503ULL) % std::uint64_t(m));
                int* row = vc[v].data() + std::size_t(i) * r;
                for (int d = 0; d < r; ++d) row[d] = (h + d * step) % m;
                std::sort(row, row + r);
            }
        for (int b = 0; b < batch; ++b) {
            const int v = variant_of(b, batch);
            std::copy(vc[v].begin(), vc[v].end(), cols.data() + std::size_t(b) * nnz);
            for (int i = 0; i <= m; ++i) offs[std::size_t(b) * (m + 1) + i] = i * r;
        }
        (void)MV(vals.data(), nnz, 1, nnz, nnz, batch).fill_random(q, false, 3000);
        (void)Bv.fill_random(q, false, 4000);
        (void)C0v.fill_random(q, false, 4001);
        q.wait();
    }
    void reset() {  // beta != 0 reads C
        (void)MV::copy(q, Cv, C0v);
        q.wait();
    }
    // A `vendor` pin that can_run refuses (spmm.cc's CUDA vendor terms) is refused by the tuner's
    // strict pin. evidence: docs/design/known-defects.md#defect-17-cusparse-spmm-shapes-refused-in-can_run-and-the-alignment-waiver
    std::size_t workspace() {
        return spmm_buffer_size<kBackend, T, MatrixFormat::CSR>(q, Av, Bv, Cv, alpha, beta, ta, tb);
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)spmm<kBackend, T, MatrixFormat::CSR>(q, Av, Bv, Cv, alpha, beta, ta, tb, ws);
        q.wait();
    }
    std::pair<double, int> verify() {
        return {MV::spmm_backward_error(q, Av, ta, Bv, tb, C0v, Cv, MV::up(alpha), MV::up(beta)), 0};
    }
};

std::vector<std::string> ints(const auto& xs) {
    std::vector<std::string> out;
    for (int v : xs) out.push_back(std::to_string(v));
    return out;
}

class SpmmSpec final : public OpSpec {
public:
    std::string op() const override { return "spmm"; }
    std::vector<std::string> key_names() const override { return {P::key_names.begin(), P::key_names.end()}; }
    std::vector<std::string> candidates(const std::string& dtype) const override {
        return with_dtype(dtype, []<class T>() { return spellings(P::candidates<T>()); });
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> axes() const override {
        return {{"transA", {"N", "T"}}, {"transB", {"N", "T"}}, {"m", ints(P::grid_m)},
                {"nrhs", ints(P::grid_nrhs)}, {"batch", ints(P::grid_batch)}};
    }
    double bytes(const std::string& dtype, const CellKey& key) const override {  // A (values, indices, offsets), B, C0, C
        const double m = double(key_int(key, "m")), q = double(key_int(key, "nrhs")), s = dtype_bytes(dtype);
        const double nnz = m * std::min(m, double(kNnzPerRow));
        return (nnz * (s + 4.0) + (m + 1.0) * 4.0 + 3.0 * m * q * s) * double(key_int(key, "batch"));
    }
    std::string refine_key() const override { return "m"; }
    std::vector<std::string> kernel_sources() const override {
        // kernel-sources-begin
        return {
            // family: direct
            "src/sycl/spmm_native.cc",
            "src/sycl/spmm_native.hh",
            // common
            "src/ops/spmm/choice.hh",
        };
        // kernel-sources-end
        // kernel-deps-begin
        // common "src/ops/spmm/spmm.cc"
        // kernel-deps-end
    }
    std::string spec_file() const override { return "tools/tune/spmm_spec.cc"; }
    std::string normalize_route(const std::string& origin, const std::string& algo) const override {
        return tune::normalize_route<P::SpmmChoice>(origin, algo);
    }
    std::vector<ArmOutcome> run_cell(const CellRequest& req) const override {
        return with_dtype(req.dtype, [&]<class T>() {
            auto q = std::make_shared<Queue>(Device("gpu"), kBackend);
            constexpr bool kComplex = !std::is_same_v<T, typename base_type<T>::type>;
            SpmmProblem<T> p(*q, trans_of(*key_get(req.key, "transA"), kComplex),
                             trans_of(*key_get(req.key, "transB"), kComplex), int(key_int(req.key, "m")),
                             int(key_int(req.key, "nrhs")), int(key_int(req.key, "batch")), req.ld_pad);
            const double tol = verify::bound<T>(verify::Check::blas, p.m);
            if (req.mode == "race") return run_race<P::SpmmChoice>("spmm", p, req, tol);
            return run_arms<P::SpmmChoice>("spmm", p, req, tol);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SpmmSpec)
