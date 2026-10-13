// SPDX-License-Identifier: MIT
// MatrixView's verification checks (docs/design/verification.md). Operands are promoted with
// astype(items); shapes, identities, differences, masks and Frobenius norms are the library's own
// utilities; products, the componentwise backward error, reflector application and row interchanges
// are this file's kernels, so no check routes through the op it judges.
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "../queue.hh"
#include "../sycl/device_scalar.hh"

namespace batchlas {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void bad(const char* who, const char* what) {
    throw batchlas::invalid_argument(std::string("MatrixView::") + who + ": " + what);
}

double nanmax(double a, double b) { return std::isnan(a) || std::isnan(b) ? kNaN : (a > b ? a : b); }

// den > 0 is false for a NaN den, so test NaN first or a poisoned norm reads as "no scale".
double quot(double num, double den) {
    if (std::isnan(num) || std::isnan(den)) return kNaN;
    return den > 0 ? num / den : num;
}

template <class V> std::vector<int> pick(const V& A, std::span<const int> items) {
    return items.empty() ? A.default_items() : std::vector<int>(items.begin(), items.end());
}

// Items processed together: as many as keep the largest operand under 2^25 elements.
std::vector<std::span<const int>> chunks(std::span<const int> items, std::size_t per_item) {
    const std::size_t cap = std::max<std::size_t>(1, (std::size_t(1) << 25) / std::max<std::size_t>(per_item, 1));
    std::vector<std::span<const int>> out;
    for (std::size_t i = 0; i < items.size(); i += cap) out.push_back(items.subspan(i, std::min(cap, items.size() - i)));
    return out;
}

template <class P> using MV = MatrixView<P, MatrixFormat::Dense>;

// Kernels see the POD device scalar, never std::complex (Annex-G operator*).
// evidence: docs/perf/gemm.md#gemm-the-pod-device-scalar
template <class P> using Dev = typename sycl_device::DevMap<P>::type;

// Element (i, j) of item b of a dense view, as the device scalar.
template <class P> Dev<P>* dev_ptr(const MV<P>& A) { return reinterpret_cast<Dev<P>*>(A.data_ptr()); }

template <class D> D dev_op(const D* p, int ld, long long stride, int b, Transpose t, int i, int l) {
    if (t == Transpose::NoTrans) return p[b * stride + static_cast<long long>(l) * ld + i];
    const D x = p[b * stride + static_cast<long long>(i) * ld + l];
    return t == Transpose::ConjTrans ? sycl_device::dev_conj(x) : x;
}

// A packed view of the first rows x cols of each item of a packed (ld = ld) promoted batch.
template <class P> MV<P> window(const MV<P>& A, int rows, int cols) {
    return MV<P>(A.data_ptr(), rows, cols, A.ld(), A.stride(), A.batch_size());
}

template <class P> void frobenius(Queue& ctx, const MV<P>& A, double* out) {
    if (std::size_t(A.rows()) * A.cols() == 0) { std::fill(out, out + A.batch_size(), 0.0); return; }
    (void)norm(ctx, A, NormType::Frobenius, Span<double>(out, std::size_t(A.batch_size())));
}

// R := R - a T, both masked to `mask` when given; then ||R||_F and ||T||_F per item into num, den.
template <class P> void difference(Queue& ctx, const MV<P>& R, const MV<P>& T, P a, std::optional<Uplo> mask, double* num, double* den) {
    if (std::size_t(R.rows()) * R.cols() != 0) {
        (void)linalg::axpby_into(ctx, P(1), R, -a, T, R);
        if (mask) {
            (void)linalg::triangular_mask_into(ctx, R, R, *mask);
            (void)linalg::triangular_mask_into(ctx, T, T, *mask);
        }
    }
    frobenius(ctx, R, num);
    frobenius(ctx, T, den);
}

// The Hermitian (real: symmetric) matrix from the lower triangle, in place.
template <class P> void mirror_lower(Queue& ctx, const MV<P>& A) {
    if constexpr (is_std_complex_v<P>) (void)A.hermitize(ctx, Uplo::Lower);
    else (void)A.symmetrize(ctx, Uplo::Lower);
}

template <class P> Matrix<P> identity(Queue& ctx, int rows, int cols, int batch) {
    Matrix<P> I(rows, cols, batch);
    if (std::size_t(rows) * cols * batch) (void)I.view().fill_identity(ctx);
    return I;
}

// A batch of row/column vectors as an n x 1 matrix of promoted entries (host-read through inc and stride).
template <class P, class E> Matrix<P> column_of(const VectorView<E>& v, int size, const std::vector<int>& items) {
    Matrix<P> out(size, 1, int(items.size()));
    for (std::size_t g = 0; g < items.size(); ++g)
        for (int i = 0; i < size; ++i) out.data()[g * size + i] = P(v(i, items[g]));
    return out;
}

// 0-based pivots of each item (identity, and nan_items set, for an item with one outside [1, range]).
std::vector<std::int32_t> pivots(const VectorView<std::int32_t>& piv, std::span<const int> grp, int count, int range,
                                 std::vector<bool>& nan_items) {
    std::vector<std::int32_t> p0;
    nan_items.assign(grp.size(), false);
    for (std::size_t it = 0; it < grp.size(); ++it) {
        for (int k = 0; k < count; ++k) {
            const int p = piv(k, grp[it]);
            if (p < 1 || p > range) { nan_items[it] = true; break; }
        }
        for (int k = 0; k < count; ++k) p0.push_back(nan_items[it] ? k : piv(k, grp[it]) - 1);
    }
    return p0;
}

// Row interchanges k <-> piv[k] of each item of M, k in [0, count), in order or reversed.
template <class P> void row_swaps(Queue& ctx, const MV<P>& M, const std::vector<std::int32_t>& p0, int count, bool forward) {
    UnifiedVector<std::int32_t> piv(p0.size());
    std::copy(p0.begin(), p0.end(), piv.data());
    const std::int32_t* p = piv.data();
    const auto m = M.kernel_view();
    const int n = M.cols();
    const std::size_t total = std::size_t(M.batch_size()) * n;
    if (total) ctx->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int it = int(id[0] / std::size_t(n)), c = int(id[0] % std::size_t(n));
        for (int s = 0; s < count; ++s) {
            const int k = forward ? s : count - 1 - s, ip = p[std::size_t(it) * count + k];
            if (ip != k) { const P t = m(k, c, it); m(k, c, it) = m(ip, c, it); m(ip, c, it) = t; }
        }
    });
    ctx.wait();  // piv is freed on return
}

// x <- H_0 ... H_{k-1} x for each column of X; reflector i below F's diagonal, tau a k x 1 batch.
template <class P> void apply_q(Queue& ctx, const MV<P>& X, const MV<P>& F, const MV<P>& tau, int k) {
    using D = Dev<P>;
    namespace sd = sycl_device;
    D* x = dev_ptr(X);
    const D *f = dev_ptr(F), *t = dev_ptr(tau);
    const int m = X.rows(), cols = X.cols(), ldx = X.ld(), ldf = F.ld();
    const long long sx = X.stride(), sf = F.stride(), st = tau.stride();
    const std::size_t total = std::size_t(X.batch_size()) * cols;
    if (total) (void)ctx->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const long long it = static_cast<long long>(id[0] / std::size_t(cols));
        const int j = int(id[0] % std::size_t(cols));
        D* v = x + it * sx + static_cast<long long>(j) * ldx;
        const D* fi = f + it * sf;
        for (int i = k - 1; i >= 0; --i) {
            D s = v[i];
            for (int r = i + 1; r < m; ++r) sd::fma_acc(s, sd::dev_conj(fi[static_cast<long long>(i) * ldf + r]), v[r]);
            s = sd::dev_mul(s, t[it * st + i]);
            v[i] = sd::dev_sub(v[i], s);
            for (int r = i + 1; r < m; ++r) v[r] = sd::dev_sub(v[r], sd::dev_mul(s, fi[static_cast<long long>(i) * ldf + r]));
        }
    });
}

// The componentwise ratio (or denominator) of each element, nanmax'ed on the host.
template <class P> double componentwise(Queue& ctx, const MV<P>& A, Transpose ta, const MV<P>& B, Transpose tb, const MV<P>* C0,
                                        const MV<P>* C, std::optional<Uplo> tri, P alpha, P beta, bool c0_real_diag, bool den_only) {
    using D = Dev<P>;
    using R = typename sycl_device::DevMap<P>::real;
    namespace sd = sycl_device;
    const int m = ta == Transpose::NoTrans ? A.rows() : A.cols(), kk = ta == Transpose::NoTrans ? A.cols() : A.rows();
    const int n = tb == Transpose::NoTrans ? B.cols() : B.rows(), g = A.batch_size();
    const std::size_t per = std::size_t(m) * n, total = per * g;
    UnifiedVector<double> out(std::max<std::size_t>(total, 1));
    double* o = out.data();
    const D *a = dev_ptr(A), *b = dev_ptr(B), *c0 = C0 ? dev_ptr(*C0) : nullptr, *c = C ? dev_ptr(*C) : nullptr;
    const int lda = A.ld(), ldb = B.ld(), ldc0 = C0 ? C0->ld() : 1, ldc = C ? C->ld() : 1;
    const long long sa = A.stride(), sb = B.stride(), sc0 = C0 ? C0->stride() : 0, sc = C ? C->stride() : 0;
    const bool lo = tri == Uplo::Lower, hi = tri == Uplo::Upper;
    const double aa = std::abs(alpha), ab = std::abs(beta);
    const D al = *reinterpret_cast<const D*>(&alpha), be = *reinterpret_cast<const D*>(&beta);
    if (total) ctx->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int it = int(id[0] / per), i = int(id[0] % per % std::size_t(m)), j = int(id[0] % per / std::size_t(m));
        if ((lo && i < j) || (hi && i > j)) { o[id[0]] = 0; return; }
        D acc = sd::dev_from_real<D>(R(0));
        double magsum = 0;
        for (int l = 0; l < kk; ++l) {
            const D u = dev_op(a, lda, sa, it, ta, i, l), w = dev_op(b, ldb, sb, it, tb, l, j);
            sd::fma_acc(acc, u, w);
            magsum += double(sd::dev_abs(u)) * double(sd::dev_abs(w));
        }
        D want = sd::dev_mul(al, acc);
        double den = aa * magsum;
        if (c0) {
            D z = c0[it * sc0 + static_cast<long long>(j) * ldc0 + i];
            if (c0_real_diag && i == j) z = sd::dev_from_real<D>(sd::dev_real(z));
            sd::fma_acc(want, be, z);
            den += ab * double(sd::dev_abs(z));
        }
        if (den_only) { o[id[0]] = den; return; }
        const double num = double(sd::dev_abs(sd::dev_sub(c[it * sc + static_cast<long long>(j) * ldc + i], want)));
        o[id[0]] = sycl::isnan(num) || sycl::isnan(den) ? kNaN : (den > 0 ? num / den : num);
    });
    ctx.wait();
    double worst = 0;
    for (std::size_t x = 0; x < total; ++x) worst = nanmax(worst, o[x]);
    return worst;
}

// Double kernels need fp64: without it the check runs on a leaked per-thread CPU queue.
// evidence: docs/design/verification.md#verification-no-fp64-device
Queue& check_queue(Queue& ctx) {
    if (ctx.device().supports_fp64()) return ctx;
    ctx.wait();
    static thread_local Queue* cpu = new Queue(Device("cpu"));
    return *cpu;
}

std::pair<int, int> op_dims(int rows, int cols, Transpose t) {
    return t == Transpose::NoTrans ? std::pair{rows, cols} : std::pair{cols, rows};
}

}  // namespace

template <typename T, MatrixFormat MType>
Event MatrixView<T, MType>::product(const Queue& ctx, const MatrixView& A, Transpose ta, const MatrixView& B, Transpose tb,
                                   const MatrixView& C) {
    detail::require_device_fp64<T>(ctx, "MatrixView::product");
    const auto [m, kk] = op_dims(A.rows(), A.cols(), ta);
    const auto [kb, n] = op_dims(B.rows(), B.cols(), tb);
    if (kk != kb || C.rows() != m || C.cols() != n) bad("product", "dimension mismatch");
    // An operand with batch 1 is shared by every item of C.
    if ((A.batch_size() != 1 && A.batch_size() != C.batch_size()) || (B.batch_size() != 1 && B.batch_size() != C.batch_size()))
        bad("product", "batch sizes differ");
    const std::size_t per = std::size_t(m) * n, total = per * C.batch_size();
    if (total == 0) return ctx.get_event();
    using D = Dev<T>;
    using R = typename sycl_device::DevMap<T>::real;
    namespace sd = sycl_device;
    const D *a = reinterpret_cast<const D*>(A.data_ptr()), *b = reinterpret_cast<const D*>(B.data_ptr());
    D* c = reinterpret_cast<D*>(C.data_ptr());
    const int lda = A.ld(), ldb = B.ld(), ldc = C.ld();
    const long long sa = A.batch_size() == 1 ? 0 : A.stride(), sb = B.batch_size() == 1 ? 0 : B.stride(), sc = C.stride();
    ctx->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int it = int(id[0] / per), i = int(id[0] % per % std::size_t(m)), j = int(id[0] % per / std::size_t(m));
        D acc = sd::dev_from_real<D>(R(0));
        for (int l = 0; l < kk; ++l) sd::fma_acc(acc, dev_op(a, lda, sa, it, ta, i, l), dev_op(b, ldb, sb, it, tb, l, j));
        c[it * sc + static_cast<long long>(j) * ldc + i] = acc;
    });
    return ctx.get_event();
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::potrf_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& F, Uplo uplo, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int n = F.rows();
    if (F.cols() != n || A0.rows() != n || A0.cols() != n) bad("potrf_residual", "square matrices of equal order only");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(n) * n)) {
        const int g = int(grp.size());
        auto f = F.template astype<P>(grp);
        (void)f.view().triangularize(ctx, uplo, Diag::NonUnit);
        Matrix<P> R(n, n, g);
        const bool lower = uplo == Uplo::Lower;
        (void)MV<P>::product(ctx, f.view(), lower ? Transpose::NoTrans : Transpose::ConjTrans, f.view(),
                             lower ? Transpose::ConjTrans : Transpose::NoTrans, R.view());
        auto t = A0.template astype<P>(grp);
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, R.view(), t.view(), P(1), uplo, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::similarity_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& Q, const MatrixView& B,
                                                 std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int n = A0.rows();
    if (A0.cols() != n || Q.rows() != n || Q.cols() != n || B.rows() != n || B.cols() != n)
        bad("similarity_residual", "square matrices of equal order only");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(n) * n)) {
        const int g = int(grp.size());
        auto q = Q.template astype<P>(grp), b = B.template astype<P>(grp), t = A0.template astype<P>(grp);
        Matrix<P> qb(n, n, g), R(n, n, g);
        (void)MV<P>::product(ctx, q.view(), Transpose::NoTrans, b.view(), Transpose::NoTrans, qb.view());
        (void)MV<P>::product(ctx, qb.view(), Transpose::NoTrans, q.view(), Transpose::ConjTrans, R.view());
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, R.view(), t.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::getrf_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& F, const VectorView<std::int32_t>& piv,
                                            std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = F.rows(), n = F.cols(), mn = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) bad("getrf_residual", "A0 and F differ in shape");
    if (piv.size() < mn) bad("getrf_residual", "fewer than min(m, n) pivots per item");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(m) * n)) {
        const int g = int(grp.size());
        std::vector<bool> nan_items;
        const auto p0 = pivots(piv, grp, mn, m, nan_items);
        auto pa = A0.template astype<P>(grp);
        row_swaps(ctx, pa.view(), p0, mn, true);
        auto l = F.template astype<P>(grp), u = F.template astype<P>(grp);
        const auto L = window(l.view(), m, mn), U = window(u.view(), mn, n);
        (void)L.triangularize(ctx, Uplo::Lower, Diag::Unit);
        (void)U.triangularize(ctx, Uplo::Upper, Diag::NonUnit);
        Matrix<P> R(m, n, g);
        (void)MV<P>::product(ctx, L, Transpose::NoTrans, U, Transpose::NoTrans, R.view());
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, R.view(), pa.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, nan_items[std::size_t(it)] ? kNaN : quot(nm[it], nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::qr_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& F, const VectorView<T>& tau,
                                         std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = F.rows(), n = F.cols(), k = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) bad("qr_residual", "A0 and F differ in shape");
    if (tau.size() < k) bad("qr_residual", "fewer than min(m, n) tau entries per item");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(m) * n)) {
        const int g = int(grp.size());
        const std::vector<int> it_list(grp.begin(), grp.end());
        auto f = F.template astype<P>(grp), x = F.template astype<P>(grp), t = A0.template astype<P>(grp);
        auto tv = column_of<P>(tau, k, it_list);
        // Column j of R: rows r <= j, which also satisfy r < k (j < n, and k = m when m <= n).
        (void)x.view().triangularize(ctx, Uplo::Upper, Diag::NonUnit);
        apply_q(ctx, x.view(), f.view(), tv.view(), k);
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, x.view(), t.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
Matrix<typename MatrixView<T, MType>::promoted_type, MType> MatrixView<T, MType>::form_q(Queue& ctx_in, const MatrixView& F,
                                                                                      const VectorView<T>& tau, int item, int cols) {
    Queue& ctx = check_queue(ctx_in);
    const int one[] = {item};
    return form_q(ctx, F, tau, std::span<const int>(one), cols);
}

template <typename T, MatrixFormat MType>
Matrix<typename MatrixView<T, MType>::promoted_type, MType> MatrixView<T, MType>::form_q(Queue& ctx_in, const MatrixView& F,
                                                                                      const VectorView<T>& tau,
                                                                                      std::span<const int> items, int cols) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = F.rows(), k = std::min(m, F.cols());
    if (cols < 0 || cols > m) bad("form_q", "cols must lie in [0, rows]");
    if (tau.size() < k) bad("form_q", "fewer than min(m, n) tau entries per item");
    const auto all = pick(F, items);
    auto f = F.template astype<P>(all);
    auto tv = column_of<P>(tau, k, all);
    auto Q = identity<P>(ctx, m, cols, int(all.size()));
    apply_q(ctx, Q.view(), f.view(), tv.view(), k);
    ctx.wait();
    return Q;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::svd_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& U, const VectorView<float_t<T>>& s,
                                          const MatrixView& Vh, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = A0.rows(), n = A0.cols(), k = std::min(m, n);
    if (U.rows() != m || U.cols() < k || Vh.cols() != n || Vh.rows() < k)
        bad("svd_residual", "need A0 m x n, U m x (at least min(m, n)), Vh (at least min(m, n)) x n");
    if (s.size() < k) bad("svd_residual", "fewer than min(m, n) singular values per item");
    const auto all = pick(A0, items);
    double worst = 0;
    const std::size_t per = std::max({std::size_t(m) * n, std::size_t(m) * U.cols(), std::size_t(Vh.rows()) * n});
    for (auto grp : chunks(all, per)) {
        const int g = int(grp.size());
        const std::vector<int> it_list(grp.begin(), grp.end());
        auto u = U.template astype<P>(grp), v = Vh.template astype<P>(grp), t = A0.template astype<P>(grp);
        auto sv = column_of<P>(s, k, it_list);
        const auto Vk = window(v.view(), k, n);
        using D = Dev<P>;
        D* x = dev_ptr(Vk);
        const D* d = dev_ptr(sv.view());
        const int ldv = Vk.ld();
        const long long strv = Vk.stride(), strs = sv.view().stride();
        const std::size_t total = std::size_t(g) * k * n;
        if (total) ctx->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {  // row l of Vh times s_l
            const long long it = static_cast<long long>(id[0] / (std::size_t(k) * n));
            const std::size_t r = id[0] % (std::size_t(k) * n);
            const int l = int(r % std::size_t(k)), j = int(r / std::size_t(k));
            D& e = x[it * strv + static_cast<long long>(j) * ldv + l];
            e = sycl_device::dev_mul(e, d[it * strs + l]);
        });
        Matrix<P> X(m, n, g);
        (void)MV<P>::product(ctx, window(u.view(), m, k), Transpose::NoTrans, Vk, Transpose::NoTrans, X.view());
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, X.view(), t.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::qr_reconstruction(Queue& ctx_in, const MatrixView& A0, const MatrixView& Q, const MatrixView& R,
                                               std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = A0.rows(), n = A0.cols(), k = Q.cols();
    if (Q.rows() != m || R.cols() != n || R.rows() < k) bad("qr_reconstruction", "need A0 m x n, Q m x k, R at least k x n");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(std::max(m, R.rows())) * std::max(n, k))) {
        const int g = int(grp.size());
        auto q = Q.template astype<P>(grp), r = R.template astype<P>(grp), t = A0.template astype<P>(grp);
        const auto Rk = window(r.view(), k, n);
        (void)Rk.triangularize(ctx, Uplo::Upper, Diag::NonUnit);
        Matrix<P> X(m, n, g);
        (void)MV<P>::product(ctx, q.view(), Transpose::NoTrans, Rk, Transpose::NoTrans, X.view());
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, X.view(), t.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[g + it]));
    }
    return worst;
}

namespace {

// ||op(M) xv - bv||_F / (||M||_F ||xv||_F) per item, xv = X V and bv = B0 V for 8 seeded probe columns
// V above full_limit multiply-adds per item (V = I below it).
template <class T, class P>
double solve_core(Queue& ctx, const MV<P>& M, Transpose ta, const MV<P>& X, const MV<P>& B, double full_limit,
                  const std::vector<bool>& nan_items) {
    const int g = M.batch_size();
    const auto [m, k] = op_dims(M.rows(), M.cols(), ta);
    const int nc = X.cols();
    const bool full = double(m) * double(k) * double(nc) <= full_limit;
    const int nv = full ? nc : 8;
    std::optional<Matrix<P>> xv_store, bv_store;
    MV<P> xv = X, bv = B;
    if (!full) {
        Matrix<T> v(nc, nv, 1);
        (void)v.view().fill_random(ctx, false, 4242);
        ctx.wait();
        auto V = v.view().template astype<P>();
        xv = xv_store.emplace(k, nv, g).view();
        bv = bv_store.emplace(m, nv, g).view();
        (void)MV<P>::product(ctx, X, Transpose::NoTrans, V.view(), Transpose::NoTrans, xv);
        (void)MV<P>::product(ctx, B, Transpose::NoTrans, V.view(), Transpose::NoTrans, bv);
        ctx.wait();  // V is freed on leaving this block
    }
    Matrix<P> R(m, nv, g);
    (void)MV<P>::product(ctx, M, ta, xv, Transpose::NoTrans, R.view());
    UnifiedVector<double> nm(4 * std::size_t(g));
    difference(ctx, R.view(), bv, P(1), std::nullopt, nm.data(), nm.data() + g);
    frobenius(ctx, M, nm.data() + 2 * g);
    frobenius(ctx, xv, nm.data() + 3 * g);
    ctx.wait();
    double worst = 0;
    for (int it = 0; it < g; ++it) worst = nanmax(worst, nan_items[std::size_t(it)] ? kNaN : quot(nm[it], nm[2 * g + it] * nm[3 * g + it]));
    return worst;
}

}  // namespace

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::solve_residual(Queue& ctx_in, const MatrixView& A0, Transpose ta, const MatrixView& X, const MatrixView& B0,
                                            std::span<const int> items, double full_limit) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const auto [m, k] = op_dims(A0.rows(), A0.cols(), ta);
    const bool identity_rhs = B0.rows() == 0;
    if (X.rows() != k || (!identity_rhs && (B0.rows() != m || B0.cols() != X.cols()))) bad("solve_residual", "dimension mismatch");
    const auto all = pick(A0, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(std::max(m, k)) * std::max({m, k, X.cols()}))) {
        const int g = int(grp.size());
        auto a = A0.template astype<P>(grp), x = X.template astype<P>(grp);
        auto b = identity_rhs ? identity<P>(ctx, m, X.cols(), g) : B0.template astype<P>(grp);
        worst = nanmax(worst, solve_core<T>(ctx, a.view(), ta, x.view(), b.view(), full_limit, std::vector<bool>(grp.size(), false)));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::solve_residual(Queue& ctx_in, const MatrixView& A0, const MatrixView& X, const MatrixView& B0,
                                            std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    return solve_residual(ctx, A0, Transpose::NoTrans, X, B0, items);
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::lu_solve_residual(Queue& ctx_in, const MatrixView& F, const VectorView<std::int32_t>& piv, Transpose ta,
                                               const MatrixView& X, const MatrixView& B0, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int n = F.rows();
    if (F.cols() != n) bad("lu_solve_residual", "square factors only");
    if (piv.size() < n) bad("lu_solve_residual", "fewer than n pivots per item");
    const bool identity_rhs = B0.rows() == 0;
    if (X.rows() != n || (!identity_rhs && (B0.rows() != n || B0.cols() != X.cols()))) bad("solve_residual", "dimension mismatch");
    const auto all = pick(F, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(n) * std::max(n, X.cols()))) {
        const int g = int(grp.size());
        std::vector<bool> nan_items;
        const auto p0 = pivots(piv, grp, n, n, nan_items);
        auto l = F.template astype<P>(grp), u = F.template astype<P>(grp);
        (void)l.view().triangularize(ctx, Uplo::Lower, Diag::Unit);
        (void)u.view().triangularize(ctx, Uplo::Upper, Diag::NonUnit);
        Matrix<P> M(n, n, g);
        (void)MV<P>::product(ctx, l.view(), Transpose::NoTrans, u.view(), Transpose::NoTrans, M.view());
        row_swaps(ctx, M.view(), p0, n, false);
        auto x = X.template astype<P>(grp);
        auto b = identity_rhs ? identity<P>(ctx, n, X.cols(), g) : B0.template astype<P>(grp);
        worst = nanmax(worst, solve_core<T>(ctx, M.view(), ta, x.view(), b.view(), double(1 << 28), nan_items));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::gemm_backward_error(Queue& ctx_in, const MatrixView& A, Transpose ta, const MatrixView& B, Transpose tb,
                                                 const MatrixView& C0, const MatrixView& C, promoted_type alpha, promoted_type beta,
                                                 std::span<const int> items, std::optional<Uplo> c_triangle) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const auto [am, ak] = op_dims(A.rows(), A.cols(), ta);
    const auto [bk, bn] = op_dims(B.rows(), B.cols(), tb);
    if (am != C.rows() || bn != C.cols() || ak != bk) bad("gemm_backward_error", "dimension mismatch");
    const bool use_c0 = beta != P(0);
    if (use_c0 && (C0.rows() != C.rows() || C0.cols() != C.cols())) bad("gemm_backward_error", "C0 and C differ in shape");
    const auto all = pick(C, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(std::max(am, ak)) * std::max(ak, bn))) {
        auto a = A.template astype<P>(grp), b = B.template astype<P>(grp), c = C.template astype<P>(grp);
        std::optional<Matrix<P>> c0;
        if (use_c0) c0.emplace(C0.template astype<P>(grp));
        const MV<P> c0v = c0 ? c0->view() : c.view(), cv = c.view();
        worst = nanmax(worst, componentwise(ctx, a.view(), ta, b.view(), tb, use_c0 ? &c0v : nullptr, &cv, c_triangle, alpha, beta,
                                            false, false));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::gemm_max_denominator(Queue& ctx_in, const MatrixView& A, Transpose ta, const MatrixView& B, Transpose tb,
                                                  const MatrixView& C0, promoted_type alpha, promoted_type beta, int item) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const auto [am, ak] = op_dims(A.rows(), A.cols(), ta);
    const auto [bk, bn] = op_dims(B.rows(), B.cols(), tb);
    if (ak != bk) bad("gemm_max_denominator", "dimension mismatch");
    const bool use_c0 = beta != P(0);
    if (use_c0 && (C0.rows() != am || C0.cols() != bn)) bad("gemm_max_denominator", "C0 and op(A) op(B) differ in shape");
    const int one[] = {item};
    auto a = A.template astype<P>(one), b = B.template astype<P>(one);
    std::optional<Matrix<P>> c0;
    if (use_c0) c0.emplace(C0.template astype<P>(one));
    const MV<P> c0v = c0 ? c0->view() : a.view();
    return componentwise<P>(ctx, a.view(), ta, b.view(), tb, use_c0 ? &c0v : nullptr, nullptr, std::nullopt, alpha, beta, false, true);
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::gemv_backward_error(Queue& ctx_in, const MatrixView& A, Transpose ta, const VectorView<T>& x,
                                                 const VectorView<T>& y0, const VectorView<T>& y, promoted_type alpha,
                                                 promoted_type beta, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const auto [am, ak] = op_dims(A.rows(), A.cols(), ta);
    if (x.size() != ak || y.size() != am) bad("gemv_backward_error", "dimension mismatch");
    const bool use_c0 = beta != P(0);
    if (use_c0 && y0.size() != y.size()) bad("gemv_backward_error", "y0 and y differ in size");
    const std::vector<int> all = items.empty() ? std::vector<int>{0, y.batch_size() / 2, y.batch_size() - 1} : std::vector<int>(items.begin(), items.end());
    std::vector<int> it(all);
    it.erase(std::unique(it.begin(), it.end()), it.end());
    auto a = A.template astype<P>(it);
    auto xv = column_of<P>(x, x.size(), it), yv = column_of<P>(y, y.size(), it);
    std::optional<Matrix<P>> y0v;
    if (use_c0) y0v.emplace(column_of<P>(y0, y0.size(), it));
    const MV<P> c0 = y0v ? y0v->view() : yv.view(), c = yv.view();
    return componentwise(ctx, a.view(), ta, xv.view(), Transpose::NoTrans, use_c0 ? &c0 : nullptr, &c, std::nullopt, alpha, beta, false,
                         false);
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::trsm_residual(Queue& ctx_in, const MatrixView& A, Side side, Uplo uplo, Transpose ta, Diag diag, const MatrixView& X,
                                           const MatrixView& B0, promoted_type alpha, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const bool left = side == Side::Left;
    const int na = A.rows();
    if (A.cols() != na) bad("trsm_residual", "A must be square");
    if ((left ? X.rows() : X.cols()) != na || B0.rows() != X.rows() || B0.cols() != X.cols()) bad("trsm_residual", "dimension mismatch");
    const auto all = pick(X, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(std::max(na, X.rows())) * std::max(na, X.cols()))) {
        const int g = int(grp.size());
        auto a = A.template astype<P>(grp), x = X.template astype<P>(grp), b = B0.template astype<P>(grp);
        (void)a.view().triangularize(ctx, uplo, diag);
        Matrix<P> R(X.rows(), X.cols(), g);
        if (left) (void)MV<P>::product(ctx, a.view(), ta, x.view(), Transpose::NoTrans, R.view());
        else (void)MV<P>::product(ctx, x.view(), Transpose::NoTrans, a.view(), ta, R.view());
        UnifiedVector<double> nm(4 * std::size_t(g));
        difference(ctx, R.view(), b.view(), alpha, std::nullopt, nm.data(), nm.data() + g);
        frobenius(ctx, a.view(), nm.data() + 2 * g);
        frobenius(ctx, x.view(), nm.data() + 3 * g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[2 * g + it] * nm[3 * g + it] + std::abs(alpha) * nm[g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::spmm_backward_error(Queue& ctx_in, const MatrixView<T, MatrixFormat::CSR>& A, Transpose ta, const MatrixView& B,
                                                 Transpose tb, const MatrixView& C0, const MatrixView& C, promoted_type alpha,
                                                 promoted_type beta, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    if (A.is_heterogeneous()) bad("spmm_backward_error", "heterogeneous views are not supported");
    const auto [am, ak] = op_dims(A.rows(), A.cols(), ta);
    const auto [bk, bn] = op_dims(B.rows(), B.cols(), tb);
    if (am != C.rows() || bn != C.cols() || ak != bk) bad("spmm_backward_error", "dimension mismatch");
    const bool use_c0 = beta != P(0);
    if (use_c0 && (C0.rows() != C.rows() || C0.cols() != C.cols())) bad("spmm_backward_error", "C0 and C differ in shape");
    const auto it = pick(C, items);
    // Densified in the promoted type, so duplicate entries sum as they always have.
    Matrix<P> a(A.rows(), A.cols(), int(it.size()));
    std::fill(a.data().data(), a.data().data() + std::size_t(A.rows()) * A.cols() * it.size(), P(0));
    bool bad_index = false;
    for (std::size_t g = 0; g < it.size(); ++g) {
        if (it[g] < 0 || it[g] >= A.batch_size()) throw batchlas::out_of_range("MatrixView::spmm_backward_error: batch item out of range");
        const int* ro = A.row_offsets().data() + static_cast<long long>(it[g]) * A.offset_stride();
        const int* ci = A.col_indices().data() + static_cast<long long>(it[g]) * A.matrix_stride();
        const T* v = A.data_ptr() + static_cast<long long>(it[g]) * A.matrix_stride();
        for (int r = 0; r < A.rows(); ++r)
            for (int p = ro[r]; p < ro[r + 1]; ++p) {
                if (ci[p] < 0 || ci[p] >= A.cols()) { bad_index = true; continue; }
                a.data()[g * A.rows() * A.cols() + std::size_t(ci[p]) * A.rows() + r] += P(v[p]);
            }
    }
    auto b = B.template astype<P>(it), c = C.template astype<P>(it);
    std::optional<Matrix<P>> c0;
    if (use_c0) c0.emplace(C0.template astype<P>(it));
    const MV<P> c0v = c0 ? c0->view() : c.view(), cv = c.view();
    const double e = it.empty() ? 0.0
                                : componentwise(ctx, a.view(), ta, b.view(), tb, use_c0 ? &c0v : nullptr, &cv, std::nullopt, alpha, beta,
                                                false, false);
    return bad_index ? kNaN : e;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::rank2k_backward_error(Queue& ctx_in, const MatrixView& A, const MatrixView& B, Transpose trans,
                                                   const MatrixView& C0, const MatrixView& C, Uplo uplo, promoted_type alpha,
                                                   promoted_type beta, bool hermitian, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    if constexpr (is_std_complex_v<T>) {
        if (hermitian && trans == Transpose::Trans) bad("rank2k_backward_error", "her2k takes NoTrans or ConjTrans");
        if (!hermitian && trans == Transpose::ConjTrans) bad("rank2k_backward_error", "syr2k takes NoTrans or Trans");
        if (hermitian && std::imag(beta) != 0.0) bad("rank2k_backward_error", "her2k takes a real beta");
    }
    const auto [an, ak] = op_dims(A.rows(), A.cols(), trans);
    const auto [bn, bk] = op_dims(B.rows(), B.cols(), trans);
    const int n = C.rows(), k = ak;
    if (C.cols() != n || an != n || bn != n || ak != bk) bad("rank2k_backward_error", "dimension mismatch");
    const bool use_c0 = beta != P(0);
    if (use_c0 && (C0.rows() != n || C0.cols() != n)) bad("rank2k_backward_error", "C0 and C differ in shape");
    const P a2 = hermitian ? MV<P>::conj(alpha) : alpha;
    const auto all = pick(C, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(n) * std::max(2 * k, n))) {
        const int g = int(grp.size());
        auto a = A.template astype<P>(grp), b = B.template astype<P>(grp), c = C.template astype<P>(grp);
        // [op(A) op(B)] . [alpha back(op(B)); a2 back(op(A))] has exactly the rank-2k denominator.
        Matrix<P> Pm(n, 2 * k, g), Qm(2 * k, n, g);
        const auto op = [&](const Matrix<P>& M, int rows, int i, int l, int it) {
            const P* p = M.data().data() + std::size_t(it) * M.rows() * M.cols();
            if (trans == Transpose::NoTrans) return p[std::size_t(l) * rows + i];
            const P x = p[std::size_t(i) * rows + l];
            return trans == Transpose::ConjTrans ? MV<P>::conj(x) : x;
        };
        for (int it = 0; it < g; ++it)
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < n; ++i) {
                    const P va = op(a, A.rows(), i, l, it), vb = op(b, B.rows(), i, l, it);
                    Pm(i, l, it) = va;
                    Pm(i, k + l, it) = vb;
                    Qm(l, i, it) = alpha * (hermitian ? MV<P>::conj(vb) : vb);
                    Qm(k + l, i, it) = a2 * (hermitian ? MV<P>::conj(va) : va);
                }
        std::optional<Matrix<P>> c0;
        if (use_c0) c0.emplace(C0.template astype<P>(grp));
        const MV<P> c0v = c0 ? c0->view() : c.view(), cv = c.view();
        worst = nanmax(worst, componentwise(ctx, Pm.view(), Transpose::NoTrans, Qm.view(), Transpose::NoTrans, use_c0 ? &c0v : nullptr, &cv,
                                            uplo, P(1), beta, hermitian, false));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::orthogonality(Queue& ctx_in, const MatrixView& Q, std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int m = Q.rows(), c = Q.cols();
    const auto all = pick(Q, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(std::max(m, c)) * c)) {
        const int g = int(grp.size());
        auto q = Q.template astype<P>(grp);
        Matrix<P> G(c, c, g);
        (void)MV<P>::product(ctx, q.view(), Transpose::ConjTrans, q.view(), Transpose::NoTrans, G.view());
        auto I = identity<P>(ctx, c, c, g);
        UnifiedVector<double> nm(2 * std::size_t(g));
        difference(ctx, G.view(), I.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, nm[it]);
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::eigen_residual(Queue& ctx_in, const MatrixView& A, const MatrixView& V, const VectorView<float_t<T>>& w,
                                            std::span<const int> items) {
    Queue& ctx = check_queue(ctx_in);
    using P = promoted_type;
    const int n = V.rows(), nv = V.cols();
    if (A.rows() != A.cols() || A.rows() != n || w.size() < nv) bad("eigen_residual", "dimension mismatch");
    const auto all = pick(V, items);
    double worst = 0;
    for (auto grp : chunks(all, std::size_t(n) * std::max(n, nv))) {
        const int g = int(grp.size());
        auto a = A.template astype<P>(grp), v = V.template astype<P>(grp);
        mirror_lower(ctx, a.view());
        Matrix<P> W(nv, nv, g);
        std::fill(W.data().data(), W.data().data() + std::size_t(nv) * nv * g, P(0));
        for (int it = 0; it < g; ++it)
            for (int j = 0; j < nv; ++j) W(j, j, it) = P(double(w(j, grp[std::size_t(it)])));
        Matrix<P> R(n, nv, g), Tm(n, nv, g);
        (void)MV<P>::product(ctx, a.view(), Transpose::NoTrans, v.view(), Transpose::NoTrans, R.view());
        (void)MV<P>::product(ctx, v.view(), Transpose::NoTrans, W.view(), Transpose::NoTrans, Tm.view());
        UnifiedVector<double> nm(3 * std::size_t(g));
        difference(ctx, R.view(), Tm.view(), P(1), std::nullopt, nm.data(), nm.data() + g);
        frobenius(ctx, a.view(), nm.data() + 2 * g);
        ctx.wait();
        for (int it = 0; it < g; ++it) worst = nanmax(worst, quot(nm[it], nm[2 * g + it]));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::values_error(const VectorView<float_t<T>>& w, const std::vector<std::vector<double>>& ref, double scale,
                                          std::span<const int> items) {
    std::vector<int> all = items.empty() ? std::vector<int>{0, w.batch_size() / 2, w.batch_size() - 1} : std::vector<int>(items.begin(), items.end());
    if (items.empty()) all.erase(std::unique(all.begin(), all.end()), all.end());
    double worst = 0;
    for (int b : all) {
        if (b < 0 || b >= w.batch_size()) throw batchlas::out_of_range("MatrixView::values_error: batch item out of range");
        if (b >= int(ref.size()) || int(ref[std::size_t(b)].size()) != w.size()) bad("values_error", "ref has no matching item");
        for (int i = 0; i < w.size(); ++i) worst = nanmax(worst, quot(std::fabs(double(w(i, b)) - ref[std::size_t(b)][std::size_t(i)]), scale));
    }
    return worst;
}

template <typename T, MatrixFormat MType>
double MatrixView<T, MType>::pivot_ratio(const MatrixView& F, std::span<const int> items) {
    const int m = F.rows(), k = std::min(m, F.cols());
    double worst = 0;
    for (int b : pick(F, items)) {
        if (b < 0 || b >= F.batch_size()) throw batchlas::out_of_range("MatrixView::pivot_ratio: batch item out of range");
        for (int j = 0; j < k; ++j) {
            const T ukk = F(j, j, b);
            const double den = cabs1(ukk);
            if (den == 0.0) continue;
            for (int i = j + 1; i < m; ++i) {
                const auto prod = up(F(i, j, b)) * up(ukk);
                worst = nanmax(worst, MatrixView<promoted_type, MType>::cabs1(prod) / den);
            }
        }
    }
    return worst;
}

template <typename T, MatrixFormat MType>
bool MatrixView<T, MType>::pivots_valid(const VectorView<std::int32_t>& piv, int n) {
    for (int b = 0; b < piv.batch_size(); ++b)
        for (int k = 0; k < piv.size(); ++k)
            if (piv(k, b) < 1 || piv(k, b) > n) return false;
    return true;
}

#define BATCHLAS_MATRIX_CHECKS(T)                                                                                               \
    template Event MatrixView<T, MatrixFormat::Dense>::product(const Queue&, const MatrixView&, Transpose, const MatrixView&,   \
                                                               Transpose, const MatrixView&);                                  \
    template double MatrixView<T, MatrixFormat::Dense>::potrf_residual(Queue&, const MatrixView&, const MatrixView&, Uplo,     \
                                                                       std::span<const int>);                                  \
    template double MatrixView<T, MatrixFormat::Dense>::similarity_residual(Queue&, const MatrixView&, const MatrixView&,       \
                                                                            const MatrixView&, std::span<const int>);         \
    template double MatrixView<T, MatrixFormat::Dense>::getrf_residual(Queue&, const MatrixView&, const MatrixView&,           \
                                                                       const VectorView<std::int32_t>&, std::span<const int>); \
    template double MatrixView<T, MatrixFormat::Dense>::qr_residual(Queue&, const MatrixView&, const MatrixView&,              \
                                                                    const VectorView<T>&, std::span<const int>);               \
    template Matrix<MatrixView<T, MatrixFormat::Dense>::promoted_type, MatrixFormat::Dense>                                    \
    MatrixView<T, MatrixFormat::Dense>::form_q(Queue&, const MatrixView&, const VectorView<T>&, int, int);                     \
    template Matrix<MatrixView<T, MatrixFormat::Dense>::promoted_type, MatrixFormat::Dense>                                    \
    MatrixView<T, MatrixFormat::Dense>::form_q(Queue&, const MatrixView&, const VectorView<T>&, std::span<const int>, int);    \
    template double MatrixView<T, MatrixFormat::Dense>::svd_residual(Queue&, const MatrixView&, const MatrixView&,             \
                                                                     const VectorView<float_t<T>>&, const MatrixView&,         \
                                                                     std::span<const int>);                                    \
    template double MatrixView<T, MatrixFormat::Dense>::qr_reconstruction(Queue&, const MatrixView&, const MatrixView&,        \
                                                                          const MatrixView&, std::span<const int>);           \
    template double MatrixView<T, MatrixFormat::Dense>::solve_residual(Queue&, const MatrixView&, Transpose, const MatrixView&, \
                                                                       const MatrixView&, std::span<const int>, double);       \
    template double MatrixView<T, MatrixFormat::Dense>::solve_residual(Queue&, const MatrixView&, const MatrixView&,           \
                                                                       const MatrixView&, std::span<const int>);               \
    template double MatrixView<T, MatrixFormat::Dense>::lu_solve_residual(Queue&, const MatrixView&,                           \
                                                                          const VectorView<std::int32_t>&, Transpose,          \
                                                                          const MatrixView&, const MatrixView&,                \
                                                                          std::span<const int>);                               \
    template double MatrixView<T, MatrixFormat::Dense>::gemm_backward_error(                                                   \
        Queue&, const MatrixView&, Transpose, const MatrixView&, Transpose, const MatrixView&, const MatrixView&, promoted_type, \
        promoted_type, std::span<const int>, std::optional<Uplo>);                                                             \
    template double MatrixView<T, MatrixFormat::Dense>::gemm_max_denominator(Queue&, const MatrixView&, Transpose,             \
                                                                             const MatrixView&, Transpose, const MatrixView&,  \
                                                                             promoted_type, promoted_type, int);               \
    template double MatrixView<T, MatrixFormat::Dense>::gemv_backward_error(Queue&, const MatrixView&, Transpose,              \
                                                                            const VectorView<T>&, const VectorView<T>&,        \
                                                                            const VectorView<T>&, promoted_type,               \
                                                                            promoted_type, std::span<const int>);              \
    template double MatrixView<T, MatrixFormat::Dense>::trsm_residual(Queue&, const MatrixView&, Side, Uplo, Transpose, Diag,  \
                                                                      const MatrixView&, const MatrixView&, promoted_type,     \
                                                                      std::span<const int>);                                   \
    template double MatrixView<T, MatrixFormat::Dense>::spmm_backward_error(                                                   \
        Queue&, const MatrixView<T, MatrixFormat::CSR>&, Transpose, const MatrixView&, Transpose, const MatrixView&,           \
        const MatrixView&, promoted_type, promoted_type, std::span<const int>);                                                \
    template double MatrixView<T, MatrixFormat::Dense>::rank2k_backward_error(                                                 \
        Queue&, const MatrixView&, const MatrixView&, Transpose, const MatrixView&, const MatrixView&, Uplo, promoted_type,    \
        promoted_type, bool, std::span<const int>);                                                                            \
    template double MatrixView<T, MatrixFormat::Dense>::orthogonality(Queue&, const MatrixView&, std::span<const int>);        \
    template double MatrixView<T, MatrixFormat::Dense>::eigen_residual(Queue&, const MatrixView&, const MatrixView&,           \
                                                                       const VectorView<float_t<T>>&, std::span<const int>);   \
    template double MatrixView<T, MatrixFormat::Dense>::values_error(const VectorView<float_t<T>>&,                            \
                                                                     const std::vector<std::vector<double>>&, double,          \
                                                                     std::span<const int>);                                    \
    template double MatrixView<T, MatrixFormat::Dense>::pivot_ratio(const MatrixView&, std::span<const int>);                   \
    template bool MatrixView<T, MatrixFormat::Dense>::pivots_valid(const VectorView<std::int32_t>&, int);

BATCHLAS_MATRIX_CHECKS(float)
BATCHLAS_MATRIX_CHECKS(double)
BATCHLAS_MATRIX_CHECKS(std::complex<float>)
BATCHLAS_MATRIX_CHECKS(std::complex<double>)

}  // namespace batchlas
