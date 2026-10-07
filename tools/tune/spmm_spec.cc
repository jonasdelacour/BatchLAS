// spmm for the tuner (GPU tables only; the cpu tables stay transcribed): a square m x m CSR A with
// min(m, 8) sorted, scattered columns per row (a hashed start, stride m / r), values in [-1, 1)
// with both complex parts; B, C0 random; alpha and beta with imaginary parts, beta != 0; B and C
// at ld = rows + ld_pad. trans=T times ConjTrans for a complex scalar (the key folds C into T).
// Items cycle through four patterns and value sets; item 0 and batch-1 differ. Verification is the
// componentwise error |C - Cref| / (|alpha| |op(A)| |op(B)| + |beta| |C0|) against a double host
// reference on items 0 and batch-1, every row of up to 64 sampled columns. No nnz key (choice.hh):
// 8 per row sits between the lanczos (3) and LOBPCG (16) cells of docs/perf/spmm.md.

#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/ops/spmm/choice.hh"
#include "cell_runner.hh"
#include "host_reference.hh"
#include "residuals.hh"

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
constexpr int kMaxCheckedCols = 64;

inline int variant_of(int b, int batch) { return b == 0 ? 0 : b == batch - 1 ? 1 : 2 + (b & 1); }

Transpose trans_of(const std::string& w, bool complex) {
    if (w == "N") return Transpose::NoTrans;
    return complex ? Transpose::ConjTrans : Transpose::Trans;
}

template <class T>
struct SpmmProblem {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    using SV = MatrixView<T, MatrixFormat::CSR>;
    using D = typename Prom<T>::type;
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
          alpha(mk<T>(1.25, -0.5)), beta(mk<T>(0.75, 0.25)), vals(std::size_t(nnz) * std::size_t(batch_)),
          B(sb * std::size_t(batch_)), C0(sc * std::size_t(batch_)), C(sc * std::size_t(batch_)),
          offs(std::size_t(m_ + 1) * std::size_t(batch_)), cols(std::size_t(nnz) * std::size_t(batch_)),
          Av(vals.data(), offs.data(), cols.data(), m_, m_, NonZeros{nnz}, nnz, m_ + 1, batch_),
          Bv(B.data(), br, bc, ldb, int(sb), batch_), C0v(C0.data(), m_, nrhs_, ldc, int(sc), batch_),
          Cv(C.data(), m_, nrhs_, ldc, int(sc), batch_) {
        std::vector<std::vector<int>> vc(kVariants, std::vector<int>(std::size_t(nnz)));
        std::vector<std::vector<T>> vv(kVariants, std::vector<T>(std::size_t(nnz)));
        const int step = std::max(1, m / r);
        for (int v = 0; v < kVariants; ++v) {
            Rng rg(3000 + std::uint64_t(v));
            for (int i = 0; i < m; ++i) {
                const int h = int((rg.next() + 1.0) * 0.5 * double(m)) % m;
                int* row = vc[v].data() + std::size_t(i) * r;
                for (int d = 0; d < r; ++d) row[d] = (h + d * step) % m;
                std::sort(row, row + r);
                for (int d = 0; d < r; ++d) vv[v][std::size_t(i) * r + d] = mk<T>(rg.next(), rg.next());
            }
        }
        for (int b = 0; b < batch; ++b) {
            const int v = variant_of(b, batch);
            std::copy(vc[v].begin(), vc[v].end(), cols.data() + std::size_t(b) * nnz);
            std::copy(vv[v].begin(), vv[v].end(), vals.data() + std::size_t(b) * nnz);
            for (int i = 0; i <= m; ++i) offs[std::size_t(b) * (m + 1) + i] = i * r;
        }
        Rng rg(4000);
        for (auto* x : {&B, &C0})
            for (std::size_t i = 0; i < x->size(); ++i) (*x)[i] = mk<T>(rg.next(), rg.next());
    }
    void reset() {  // beta != 0 reads C
        (void)MV::copy(q, Cv, C0v);
        q.wait();
    }
    // A `vendor` pin that can_run refuses falls back to Auto (a class word, not an error), so its
    // times would be direct's under vendor's name. These are spmm.cc's two CUDA vendor terms.
    // evidence: docs/design/known-defects.md#17-cusparse-spmm-shapes-refused-in-can_run-and-the-alignment-waiver
    std::size_t workspace() {
        constexpr bool cx = !std::is_same_v<T, typename base_type<T>::type>;
        constexpr bool zz = std::is_same_v<T, std::complex<double>>;
        const bool nn = ta == Transpose::NoTrans && tb == Transpose::NoTrans;
        std::string src;
        if (kBackend == Backend::CUDA && select::detail::pin_text("spmm", &src) == std::optional<std::string>("vendor") &&
            ((cx && tb == Transpose::ConjTrans && br == 1) || (zz && nn && nrhs == 1)))
            throw std::invalid_argument("spmm vendor: can_run refuses this shape (known defect 17)");
        return spmm_buffer_size<kBackend, T, MatrixFormat::CSR>(q, Av, Bv, Cv, alpha, beta, ta, tb);
    }
    void clear_info() {}
    void run(Span<std::byte> ws) {
        (void)spmm<kBackend, T, MatrixFormat::CSR>(q, Av, Bv, Cv, alpha, beta, ta, tb, ws);
        q.wait();
    }
    D opb(std::size_t o, int p, int j) const {  // op(B)(p, j)
        const D x = up(B[o + (tb == Transpose::NoTrans ? std::size_t(j) * ldb + p : std::size_t(p) * ldb + j)]);
        return tb == Transpose::ConjTrans ? cj(x) : x;
    }
    std::pair<double, int> verify() {
        const std::vector<int> js = sample_indices(nrhs, kMaxCheckedCols);
        double worst = 0;
        const double aal = ab(up(alpha)), abe = ab(up(beta));
        for (int b : {0, batch - 1}) {
            const std::size_t oa = std::size_t(b) * nnz, ob = std::size_t(b) * sb, oc = std::size_t(b) * sc;
            std::vector<D> acc(std::size_t(m) * js.size(), D(0));
            std::vector<double> mag(acc.size(), 0.0);
            for (int p = 0; p < m; ++p)
                for (int e = offs[std::size_t(b) * (m + 1) + p]; e < offs[std::size_t(b) * (m + 1) + p + 1]; ++e) {
                    const int c = cols[oa + e];
                    D a = up(vals[oa + e]);
                    if (ta == Transpose::ConjTrans) a = cj(a);
                    // NoTrans: C(p, j) += a(p, c) op(B)(c, j); else C(c, j) += op(a)(c, p) op(B)(p, j).
                    const int row = ta == Transpose::NoTrans ? p : c, k = ta == Transpose::NoTrans ? c : p;
                    for (std::size_t t = 0; t < js.size(); ++t) {
                        const D x = a * opb(ob, k, js[t]);
                        acc[t * m + row] += x;
                        mag[t * m + row] += ab(x);
                    }
                }
            for (std::size_t t = 0; t < js.size(); ++t)
                for (int i = 0; i < m; ++i) {
                    const std::size_t at = oc + std::size_t(js[t]) * ldc + i;
                    const D c0 = up(C0[at]), want = up(alpha) * acc[t * m + i] + up(beta) * c0;
                    const double num = ab(up(C[at]) - want), den = aal * mag[t * m + i] + abe * ab(c0);
                    worst = nanmax(worst, den > 0 ? num / den : num);
                }
        }
        return {worst, 0};
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
            if (req.mode == "race") return run_race<P::SpmmChoice>("spmm", p, req, Tol<T>::v);
            return run_arms<P::SpmmChoice>("spmm", p, req, Tol<T>::v);
        });
    }
};

}  // namespace
}  // namespace batchlas::tune

BATCHLAS_TUNE_REGISTER(batchlas::tune::SpmmSpec)
