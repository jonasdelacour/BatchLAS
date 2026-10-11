// SPDX-License-Identifier: MIT
// libbatchlas_verify: the checks of batchlas::verify on a SYCL device
// (docs/design/verification.md#verification-device-engine). Checked items are packed and promoted
// on the host into shared USM of the lane's queue; shapes, differences, masks and Frobenius norms
// are library utilities; products, the componentwise backward error, reflector application and
// row swaps are this file's kernels (a check cannot route through the ops it judges).
#include <batchlas/verify/engine.hh>
#include <batchlas/verify/scalar.hh>

#include <batchlas/blas/extra.hh>
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/sycl_interop.hh>

#include <sycl/sycl.hpp>

#if BATCHLAS_VERIFY_HAVE_LAPACKE
#include <cblas.h>

#include "../backends/host_blas_health.hh"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace batchlas::verify::detail {
namespace {

using cd = std::complex<double>;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void bad(const char* who, const char* what) {
    throw std::invalid_argument(std::string("batchlas::verify::") + who + ": " + what);
}

// Written-out complex<double> for this file's kernels (layout-compatible with std::complex<double>):
// no std::complex arithmetic in device code (Annex G __muldc3, and AdaptiveCpp).
struct zd {
    double re, im;
};
inline zd operator+(zd a, zd b) { return {a.re + b.re, a.im + b.im}; }
inline zd operator-(zd a, zd b) { return {a.re - b.re, a.im - b.im}; }
inline zd operator*(zd a, zd b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
inline double mag(double x) { return sycl::fabs(x); }
inline double mag(zd x) { return sycl::hypot(x.re, x.im); }
inline double conj(double x) { return x; }
inline zd conj(zd x) { return {x.re, -x.im}; }
inline double real_part(double x) { return x; }
inline zd real_part(zd x) { return {x.re, 0.0}; }

template <class D> using K = std::conditional_t<std::is_same_v<D, double>, double, zd>;
template <class D> K<D>* kp(D* p) { return reinterpret_cast<K<D>*>(p); }
template <class D> const K<D>* kp(const D* p) { return reinterpret_cast<const K<D>*>(p); }
template <class X> X kzero() { if constexpr (std::is_same_v<X, double>) return 0.0; else return zd{0, 0}; }
template <class X> X kmake(cd z) { if constexpr (std::is_same_v<X, double>) return z.real(); else return zd{z.real(), z.imag()}; }

inline double hquot(double num, double den) {
    if (std::isnan(num) || std::isnan(den)) return kNaN;
    return den > 0 ? num / den : num;
}

// One device's batchlas::Queue and its grow-only scratch slots, in shared USM of that queue's
// context so the host packs straight into them and reads results back after the check's one wait
// (a synchronization costs ~1 ms on the CUDA adapter here).
// evidence: docs/design/verification.md#verification-device-engine
class Lane {
public:
    explicit Lane(Queue* Q) : Q_(Q), slots_(24) {}
    Queue& Q() { return *Q_; }
    sycl::queue& q() { return sycl_queue(*Q_); }

    template <class X> X* slot(int id, std::size_t count) {
        auto& s = slots_.at(std::size_t(id));
        const std::size_t bytes = std::max<std::size_t>(count * sizeof(X), 1);
        if (s.second < bytes) {
            q().wait();  // queued work may still use the old slot
            if (s.first) sycl::free(s.first, q());
            s.first = sycl::malloc_shared(bytes, q());
            if (!s.first) throw std::runtime_error("batchlas::verify: allocation failed");
            s.second = bytes;
        }
        return static_cast<X*>(s.first);
    }

private:
    Queue* Q_;
    std::vector<std::pair<void*, std::size_t>> slots_;
};

// Checks above kGpuWork multiply-adds run on the first GPU, the rest on the first CPU device: both
// lanes run the same kernels. A device qualifies with fp64 and a probe kernel that runs (a build
// without its image fails the probe). Leaked: destroying SYCL objects after the runtime is gone hangs.
class Engine {
public:
    static constexpr double kGpuWork = double(1 << 24);
    static Engine& get() {
        static Engine* e = new Engine();
        return *e;
    }
    std::mutex& mutex() { return mu_; }
    Lane& lane(double work) {
        Lane& l = gpu_ && (work > kGpuWork || !cpu_) ? *gpu_ : *cpu_;
        l.Q().attach_to_current_thread();
        return l;
    }

private:
    static Lane* probe(DeviceType type) {
        for (const Device& d : Device::get_devices(type)) {
            try {
                auto Q = std::make_unique<Queue>(d, true);
                sycl::queue& q = sycl_queue(*Q);
                if (!q.get_device().has(sycl::aspect::fp64)) continue;
                double* p = sycl::malloc_device<double>(1, q);
                q.single_task([=] { p[0] = 1.0; }).wait_and_throw();
                sycl::free(p, q);
                return new Lane(Q.release());
            } catch (const std::exception&) {
            }
        }
        return nullptr;
    }
    Engine() : gpu_(probe(DeviceType::GPU)), cpu_(probe(DeviceType::CPU)) {
        if (!gpu_ && !cpu_) throw std::runtime_error("batchlas::verify: no SYCL device with fp64 runs the verification kernels");
    }
    std::mutex mu_;
    Lane* gpu_;
    Lane* cpu_;
};

enum Slot { kA, kB, kC, kD, kE, kF, kR, kT, kU, kV, kW, kOut, kNorm, kPiv };

void check_items(const Operand& o, std::span<const int> items, const char* who) {
    for (int b : items)
        if (b < 0 || b >= o.batch) bad(who, "batch item out of range");
}

template <class D> D read_as(const Operand& o, long long off) {
    double re = 0, im = 0;
    switch (o.dtype) {
        case Dtype::f32: re = static_cast<const float*>(o.data)[off]; break;
        case Dtype::f64: re = static_cast<const double*>(o.data)[off]; break;
        case Dtype::c32: re = static_cast<const float*>(o.data)[2 * off]; im = static_cast<const float*>(o.data)[2 * off + 1]; break;
        case Dtype::c64: re = static_cast<const double*>(o.data)[2 * off]; im = static_cast<const double*>(o.data)[2 * off + 1]; break;
        case Dtype::i32: re = static_cast<const std::int32_t*>(o.data)[off]; break;
    }
    if constexpr (std::is_same_v<D, double>) { (void)im; return re; }
    else return D(re, im);
}

// The rows x cols top-left window of each item in `items`, promoted and packed (ld = rows).
template <class D> D* pack(Lane& e, int slot, const Operand& o, std::span<const int> items, int rows, int cols) {
    const std::size_t per = std::size_t(rows) * std::size_t(cols);
    D* out = e.slot<D>(slot, per * items.size());
    for (std::size_t g = 0; g < items.size(); ++g) {
        const long long base = static_cast<long long>(items[g]) * o.stride;
        for (int c = 0; c < cols; ++c)
            for (int r = 0; r < rows; ++r)
                out[g * per + std::size_t(c) * rows + r] =
                    read_as<D>(o, base + static_cast<long long>(c) * o.ld + static_cast<long long>(r) * o.inc);
    }
    return out;
}

template <class D> MatrixView<D, MatrixFormat::Dense> mv(D* p, int rows, int cols, int g) {
    return MatrixView<D, MatrixFormat::Dense>(p, rows, cols, std::max(rows, 1), std::max(rows * cols, 1), g);
}

// The Shape applied in place by the library's transforms (the unread triangle is overwritten).
// Their Events are dropped throughout: the lane's queue is in-order and each check waits once.
template <class D> void shape(Lane& e, D* p, int rows, int cols, int g, Shape s) {
    if (s == Shape::general || rows == 0 || cols == 0) return;
    const auto v = mv(p, rows, cols, g);
    switch (s) {
        case Shape::lower: (void)v.triangularize(e.Q(), Uplo::Lower, Diag::NonUnit); break;
        case Shape::upper: (void)v.triangularize(e.Q(), Uplo::Upper, Diag::NonUnit); break;
        case Shape::unit_lower: (void)v.triangularize(e.Q(), Uplo::Lower, Diag::Unit); break;
        case Shape::unit_upper: (void)v.triangularize(e.Q(), Uplo::Upper, Diag::Unit); break;
        case Shape::hermitian_lower:
        case Shape::hermitian_upper: {
            const Uplo u = s == Shape::hermitian_lower ? Uplo::Lower : Uplo::Upper;
            if constexpr (std::is_same_v<D, double>) (void)v.symmetrize(e.Q(), u);  // real: hermitize is not instantiated
            else (void)v.hermitize(e.Q(), u);
            break;
        }
        case Shape::symmetric_lower: (void)v.symmetrize(e.Q(), Uplo::Lower); break;
        case Shape::symmetric_upper: (void)v.symmetrize(e.Q(), Uplo::Upper); break;
        case Shape::general: break;
    }
}

template <class D> D* operand(Lane& e, int slot, const Operand& o, std::span<const int> items, Shape s) {
    D* p = pack<D>(e, slot, o, items, o.rows, o.cols);
    shape(e, p, o.rows, o.cols, int(items.size()), s);
    return p;
}

template <class F> void launch(Lane& e, std::size_t n, F f) {
    if (n == 0) return;
    e.q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { f(std::size_t(id[0])); });
}

// op(X)(i, l) of a packed rows x cols item.
template <class X> X op_at(const X* p, int rows, Transpose t, int i, int l) {
    if (t == Transpose::NoTrans) return p[std::size_t(l) * rows + i];
    const X x = p[std::size_t(i) * rows + l];
    return t == Transpose::ConjTrans ? conj(x) : x;
}

// R = op(P) op(Q) per item (packed m x n), the inner sum in order.
template <class D> D* product(Lane& e, int slot, const D* P, int pr, int pc, Transpose tp, const D* Q, int qr, int qc, Transpose tq, int g) {
    const int m = tp == Transpose::NoTrans ? pr : pc, kk = tp == Transpose::NoTrans ? pc : pr;
    const int n = tq == Transpose::NoTrans ? qc : qr;
    D* R = e.slot<D>(slot, std::size_t(g) * m * n);
    const auto *p = kp(P), *q = kp(Q);
    auto* r = kp(R);
    const std::size_t per = std::size_t(m) * n, sp = std::size_t(pr) * pc, sq = std::size_t(qr) * qc;
    launch(e, std::size_t(g) * per, [=](std::size_t x) {
        const std::size_t it = x / per, rc = x % per;
        const int i = int(rc % std::size_t(m)), j = int(rc / std::size_t(m));
        auto acc = kzero<K<D>>();
        for (int l = 0; l < kk; ++l) acc = acc + op_at(p + it * sp, pr, tp, i, l) * op_at(q + it * sq, qr, tq, l, j);
        r[x] = acc;
    });
    return R;
}

// Frobenius norm of each item of a packed m x n batch, into norms[0, g) (valid after the wait).
template <class D> void frob(Lane& e, D* p, int m, int n, int g, double* norms) {
    if (std::size_t(m) * n == 0) { std::fill(norms, norms + g, 0.0); return; }
    (void)batchlas::norm(e.Q(), mv(p, m, n, g), NormType::Frobenius, Span<double>(norms, std::size_t(g)));
}

// R := R - a T (T null: the identity), both masked to `mask`'s triangle; then ||R||_F and ||T||_F per item.
template <class D> void difference(Lane& e, D* R, D* T, D a, int m, int n, int g, Shape mask, double* num, double* den) {
    if (std::size_t(m) * n * g == 0) { std::fill(num, num + g, 0.0); std::fill(den, den + g, 0.0); return; }
    if (!T) {
        T = e.slot<D>(kU, std::size_t(g) * m * n);
        (void)mv(T, m, n, g).fill_identity(e.Q());
    }
    const auto r = mv(R, m, n, g), t = mv(T, m, n, g);
    (void)linalg::axpby_into(e.Q(), D(1), r, -a, t, r);
    if (mask == Shape::lower || mask == Shape::upper) {
        const Uplo u = mask == Shape::lower ? Uplo::Lower : Uplo::Upper;
        (void)linalg::triangular_mask_into(e.Q(), r, r, u);
        (void)linalg::triangular_mask_into(e.Q(), t, t, u);
    }
    frob(e, R, m, n, g, num);
    frob(e, T, m, n, g, den);
}

// Items processed together: as many as keep the largest buffer under 2^25 elements.
std::vector<std::span<const int>> groups(std::span<const int> items, std::size_t per_item) {
    const std::size_t cap = std::max<std::size_t>(1, (std::size_t(1) << 25) / std::max<std::size_t>(per_item, 1));
    std::vector<std::span<const int>> out;
    for (std::size_t i = 0; i < items.size(); i += cap) out.push_back(items.subspan(i, std::min(cap, items.size() - i)));
    return out;
}

template <class F> auto dispatch(bool complex, F f) { return complex ? f(cd{}) : f(double{}); }

std::vector<std::int32_t> read_pivots(const Operand& p, int item, int count) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    const auto* d = static_cast<const std::int32_t*>(p.data);
    for (int k = 0; k < count; ++k) out[std::size_t(k)] = d[static_cast<long long>(item) * p.stride + static_cast<long long>(k) * p.inc];
    return out;
}

// 0-based pivots of each item of `grp` (count each, identity for an item with one out of [1, range]).
std::vector<std::int32_t> pivots(const Operand& piv, std::span<const int> grp, int count, int range, std::vector<bool>& nan_items) {
    std::vector<std::int32_t> p0;
    nan_items.assign(grp.size(), false);
    for (std::size_t it = 0; it < grp.size(); ++it) {
        const auto p = read_pivots(piv, grp[it], count);
        for (int k = 0; k < count; ++k)
            if (p[std::size_t(k)] < 1 || p[std::size_t(k)] > range) { nan_items[it] = true; break; }
        for (int k = 0; k < count; ++k) p0.push_back(nan_items[it] ? k : p[std::size_t(k)] - 1);
    }
    return p0;
}

// The row interchanges k <-> piv[k] of each item of M (m x n), k in [0, count), in order or reversed.
template <class D> void row_swaps(Lane& e, D* M, const std::vector<std::int32_t>& piv0, int g, int m, int n, int count, bool forward) {
    auto* piv = e.slot<std::int32_t>(kPiv, piv0.size());
    std::copy(piv0.begin(), piv0.end(), piv);
    auto* mm = kp(M);
    launch(e, std::size_t(g) * n, [=](std::size_t x) {
        const std::size_t it = x / std::size_t(n);
        auto* col = mm + it * std::size_t(m) * n + (x % std::size_t(n)) * m;
        const std::int32_t* p = piv + it * std::size_t(count);
        for (int s = 0; s < count; ++s) {
            const int k = forward ? s : count - 1 - s, ip = p[k];
            if (ip != k) { const auto t = col[k]; col[k] = col[ip]; col[ip] = t; }
        }
    });
}

// x <- H_0 ... H_{k-1} x for each column of X (m x cols per item); reflector i below F's (m x n) diagonal.
template <class D> void apply_q(Lane& e, D* X, const D* F, const D* tau, int g, int m, int n, int k, int cols) {
    auto* xx = kp(X);
    const auto *f = kp(F), *t = kp(tau);
    launch(e, std::size_t(g) * cols, [=](std::size_t x) {
        const std::size_t it = x / std::size_t(cols);
        auto* v = xx + it * std::size_t(m) * cols + (x % std::size_t(cols)) * m;
        const auto* fi = f + it * std::size_t(m) * n;
        for (int i = k - 1; i >= 0; --i) {
            auto s = v[i];
            for (int r = i + 1; r < m; ++r) s = s + conj(fi[std::size_t(i) * m + r]) * v[r];
            s = s * t[it * std::size_t(k) + i];
            v[i] = v[i] - s;
            for (int r = i + 1; r < m; ++r) v[r] = v[r] - s * fi[std::size_t(i) * m + r];
        }
    });
}

// The componentwise ratio (or denominator) per element of each item, nanmax'ed on the host.
template <class D> double componentwise_core(Lane& e, const D* A, int ar, int ac, Transpose ta, const D* B, int br, int bc, Transpose tb,
                                             const D* C0, const D* C, int g, Shape sc, cd alpha, cd beta, bool c0_real_diag, bool den_only) {
    const int m = ta == Transpose::NoTrans ? ar : ac, kk = ta == Transpose::NoTrans ? ac : ar, n = tb == Transpose::NoTrans ? bc : br;
    const std::size_t per = std::size_t(m) * n, total = std::size_t(g) * per, sa = std::size_t(ar) * ac, sb = std::size_t(br) * bc;
    double* out = e.slot<double>(kOut, total);
    using X = K<D>;
    const X al = kmake<X>(alpha), be = kmake<X>(beta);
    const double aa = std::abs(alpha), ab = std::abs(beta);
    const auto *a = kp(A), *b = kp(B), *c0 = kp(C0), *c = kp(C);
    launch(e, total, [=](std::size_t x) {
        const std::size_t it = x / per, rc = x % per;
        const int i = int(rc % std::size_t(m)), j = int(rc / std::size_t(m));
        const bool lo = sc == Shape::lower, hi = sc == Shape::upper;
        if ((lo && i < j) || (hi && i > j)) { out[x] = 0; return; }
        auto acc = kzero<X>();
        double magsum = 0;
        for (int l = 0; l < kk; ++l) {
            const X u = op_at(a + it * sa, ar, ta, i, l), w = op_at(b + it * sb, br, tb, l, j);
            acc = acc + u * w;
            magsum += mag(u) * mag(w);
        }
        X want = al * acc;
        double den = aa * magsum;
        if (c0) {
            X z = c0[x];
            if (c0_real_diag && i == j) z = real_part(z);
            want = want + be * z;
            den += ab * mag(z);
        }
        if (den_only) { out[x] = den; return; }
        const double num = mag(c[x] - want);
        out[x] = sycl::isnan(num) || sycl::isnan(den) ? kNaN : (den > 0 ? num / den : num);
    });
    e.q().wait();
    double worst = 0;
    for (std::size_t x = 0; x < total; ++x) worst = nanmax(worst, out[x]);
    return worst;
}

}  // namespace

double potrf(const Operand& A0, const Operand& F, bool upper, bool complex, std::span<const int> items) {
    check_items(A0, items, "potrf_residual");
    check_items(F, items, "potrf_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int n = F.rows;
    Lane& e = eng.lane(double(n) * n * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * n)) {
            const int g = int(grp.size());
            const Shape s = upper ? Shape::upper : Shape::lower;
            D* f = operand<D>(e, kA, F, grp, s);
            D* R = upper ? product(e, kR, f, n, n, Transpose::ConjTrans, f, n, n, Transpose::NoTrans, g)
                         : product(e, kR, f, n, n, Transpose::NoTrans, f, n, n, Transpose::ConjTrans, g);
            D* T = operand<D>(e, kT, A0, grp, Shape::general);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference(e, R, T, D(1), n, n, g, s, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, hquot(nm[it], nm[g + it]));
        }
        return worst;
    });
}

double similarity(const Operand& A0, const Operand& Q, const Operand& B, bool complex, std::span<const int> items) {
    for (const Operand* o : {&A0, &Q, &B}) check_items(*o, items, "similarity_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int n = A0.rows;
    Lane& e = eng.lane(2.0 * n * n * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * n)) {
            const int g = int(grp.size());
            D* q = operand<D>(e, kA, Q, grp, Shape::general);
            D* b = operand<D>(e, kB, B, grp, Shape::general);
            D* qb = product(e, kD, q, n, n, Transpose::NoTrans, b, n, n, Transpose::NoTrans, g);
            D* R = product(e, kR, qb, n, n, Transpose::NoTrans, q, n, n, Transpose::ConjTrans, g);
            D* T = operand<D>(e, kT, A0, grp, Shape::general);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference(e, R, T, D(1), n, n, g, Shape::general, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, hquot(nm[it], nm[g + it]));
        }
        return worst;
    });
}

double getrf(const Operand& A0, const Operand& F, const Operand& piv, bool complex, std::span<const int> items) {
    check_items(A0, items, "getrf_residual");
    check_items(F, items, "getrf_residual");
    check_items(piv, items, "getrf_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int m = F.rows, n = F.cols, mn = std::min(m, n);
    Lane& e = eng.lane(double(m) * mn * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(m) * n)) {
            const int g = int(grp.size());
            std::vector<bool> nan_items;
            const auto p0 = pivots(piv, grp, mn, m, nan_items);
            D* PA = operand<D>(e, kT, A0, grp, Shape::general);
            row_swaps(e, PA, p0, g, m, n, mn, true);
            D* L = pack<D>(e, kA, F, grp, m, mn);
            shape(e, L, m, mn, g, Shape::unit_lower);
            D* U = pack<D>(e, kB, F, grp, mn, n);
            shape(e, U, mn, n, g, Shape::upper);
            D* R = product(e, kR, L, m, mn, Transpose::NoTrans, U, mn, n, Transpose::NoTrans, g);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference(e, R, PA, D(1), m, n, g, Shape::general, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, nan_items[std::size_t(it)] ? kNaN : hquot(nm[it], nm[g + it]));
        }
        return worst;
    });
}

double qr(const Operand& A0, const Operand& F, const Operand& tau, bool complex, std::span<const int> items) {
    check_items(A0, items, "qr_residual");
    check_items(F, items, "qr_residual");
    check_items(tau, items, "qr_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int m = F.rows, n = F.cols, k = std::min(m, n);
    Lane& e = eng.lane(2.0 * m * n * k * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(m) * n)) {
            const int g = int(grp.size());
            D* f = pack<D>(e, kA, F, grp, m, n);
            D* t = pack<D>(e, kB, tau, grp, k, 1);
            // Column j of R: rows r <= j, which also satisfy r < k (j < n, and k = m when m <= n).
            D* X = operand<D>(e, kR, F, grp, Shape::upper);
            apply_q(e, X, f, t, g, m, n, k, n);
            D* T = operand<D>(e, kT, A0, grp, Shape::general);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference(e, X, T, D(1), m, n, g, Shape::general, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, hquot(nm[it], nm[g + it]));
        }
        return worst;
    });
}

std::vector<double> form_q(const Operand& F, const Operand& tau, bool complex, int item, int cols) {
    const int one[] = {item};
    check_items(F, one, "form_q");
    check_items(tau, one, "form_q");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int m = F.rows, n = F.cols, k = std::min(m, n);
    Lane& e = eng.lane(2.0 * m * cols * k);
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        D* f = pack<D>(e, kA, F, one, m, n);
        D* t = pack<D>(e, kB, tau, one, k, 1);
        D* X = e.slot<D>(kR, std::size_t(m) * cols);
        if (std::size_t(m) * cols) (void)mv(X, m, cols, 1).fill_identity(e.Q());
        apply_q(e, X, f, t, 1, m, n, k, cols);
        e.q().wait();
        const auto* x = reinterpret_cast<const double*>(X);
        return std::vector<double>(x, x + std::size_t(m) * cols * (sizeof(D) / sizeof(double)));
    });
}

double qr_reconstruction(const Operand& A0, const Operand& Q, const Operand& R, bool complex, std::span<const int> items) {
    for (const Operand* o : {&A0, &Q, &R}) check_items(*o, items, "qr_reconstruction");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int m = A0.rows, n = A0.cols, k = Q.cols;
    Lane& e = eng.lane(double(m) * k * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(m, k)) * std::max(n, k))) {
            const int g = int(grp.size());
            D* q = operand<D>(e, kA, Q, grp, Shape::general);
            D* r = pack<D>(e, kB, R, grp, k, n);
            shape(e, r, k, n, g, Shape::upper);
            D* P = product(e, kR, q, m, k, Transpose::NoTrans, r, k, n, Transpose::NoTrans, g);
            D* T = operand<D>(e, kT, A0, grp, Shape::general);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference(e, P, T, D(1), m, n, g, Shape::general, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, hquot(nm[it], nm[g + it]));
        }
        return worst;
    });
}

double solve(const Operand& A, Shape shape_a, Transpose trans, const Operand* piv, const Operand& X, const Operand& B0, bool complex,
             std::span<const int> items, double full_limit) {
    check_items(A, items, "solve_residual");
    check_items(X, items, "solve_residual");
    if (piv) check_items(*piv, items, "lu_solve_residual");
    const bool identity = B0.rows == 0;
    if (!identity) check_items(B0, items, "solve_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const bool nt = trans == Transpose::NoTrans;
    const int ar = A.rows, ac = A.cols, m = nt ? ar : ac, k = nt ? ac : ar, nc = X.cols;
    const bool full = double(m) * double(k) * double(nc) <= full_limit;
    const int nv = full ? nc : 8;
    Lane& e = eng.lane((double(m) * k * nv + double(k) * nc * nv + (piv ? double(ar) * ar * ar : 0.0)) * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(m, k)) * std::max({m, k, nc}))) {
            const int g = int(grp.size());
            std::vector<bool> nan_items(grp.size(), false);
            D* M;
            if (piv) {
                const auto p0 = pivots(*piv, grp, ar, ar, nan_items);
                D* L = operand<D>(e, kA, A, grp, Shape::unit_lower);
                D* U = operand<D>(e, kB, A, grp, Shape::upper);
                M = product(e, kD, L, ar, ar, Transpose::NoTrans, U, ar, ar, Transpose::NoTrans, g);
                row_swaps(e, M, p0, g, ar, ar, ar, false);
            } else {
                M = operand<D>(e, kD, A, grp, shape_a);
            }
            D* x = operand<D>(e, kE, X, grp, Shape::general);
            D* b;
            if (identity) {
                b = e.slot<D>(kF, std::size_t(g) * m * nc);
                if (std::size_t(m) * nc) (void)mv(b, m, nc, g).fill_identity(e.Q());
            } else {
                b = operand<D>(e, kF, B0, grp, Shape::general);
            }
            D *xv = x, *bv = b;
            if (!full) {
                // Probe columns: factor_bench's seeded draws, rounded to A's precision as draw<T> rounds them.
                D* V = e.slot<D>(kV, std::size_t(g) * nc * nv);
                Rng rng(4242);
                const bool single = A.dtype == Dtype::f32 || A.dtype == Dtype::c32, cx = A.dtype == Dtype::c32 || A.dtype == Dtype::c64;
                for (std::size_t z = 0; z < std::size_t(nc) * nv; ++z) {
                    const double re = rng.next(), im = rng.next();
                    const cd w(single ? double(float(re)) : re, cx ? (single ? double(float(im)) : im) : 0.0);
                    if constexpr (std::is_same_v<D, double>) V[z] = w.real();
                    else V[z] = w;
                }
                for (int it = 1; it < g; ++it) std::copy(V, V + std::size_t(nc) * nv, V + std::size_t(it) * nc * nv);
                xv = product(e, kW, x, k, nc, Transpose::NoTrans, V, nc, nv, Transpose::NoTrans, g);
                bv = product(e, kT, b, m, nc, Transpose::NoTrans, V, nc, nv, Transpose::NoTrans, g);
            }
            D* R = product(e, kR, M, ar, ac, trans, xv, k, nv, Transpose::NoTrans, g);
            double* nm = e.slot<double>(kNorm, 4 * std::size_t(g));
            difference(e, R, bv, D(1), m, nv, g, Shape::general, nm, nm + g);
            frob(e, M, ar, ac, g, nm + 2 * g);
            frob(e, xv, k, nv, g, nm + 3 * g);
            e.q().wait();
            for (int it = 0; it < g; ++it)
                worst = nanmax(worst, nan_items[std::size_t(it)] ? kNaN : hquot(nm[it], nm[2 * g + it] * nm[3 * g + it]));
        }
        return worst;
    });
}

double componentwise(const Operand& A, Shape sa, Transpose ta, const Operand& B, Shape sb, Transpose tb, const Operand* C0, const Operand* C,
                     Shape sc, std::complex<double> alpha, std::complex<double> beta, bool complex, std::span<const int> items,
                     bool denominator_only) {
    check_items(A, items, "gemm_backward_error");
    check_items(B, items, "gemm_backward_error");
    const bool use_c0 = beta != cd(0);
    if (use_c0) check_items(*C0, items, "gemm_backward_error");
    if (!denominator_only) check_items(*C, items, "gemm_backward_error");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const bool ntA = ta == Transpose::NoTrans, ntB = tb == Transpose::NoTrans;
    const int m = ntA ? A.rows : A.cols, kk = ntA ? A.cols : A.rows, n = ntB ? B.cols : B.rows;
    Lane& e = eng.lane(double(m) * kk * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(m, kk)) * std::max(kk, n))) {
            const int g = int(grp.size());
            D* a = operand<D>(e, kA, A, grp, sa);
            D* b = operand<D>(e, kB, B, grp, sb);
            D* c0 = use_c0 ? operand<D>(e, kC, *C0, grp, Shape::general) : nullptr;
            D* c = denominator_only ? nullptr : operand<D>(e, kD, *C, grp, Shape::general);
            worst = nanmax(worst, componentwise_core(e, a, A.rows, A.cols, ta, b, B.rows, B.cols, tb, c0, c, g, sc, alpha, beta, false,
                                                     denominator_only));
        }
        return worst;
    });
}

double rank2k(const Operand& A, const Operand& B, Transpose trans, const Operand* C0, const Operand& C, Uplo uplo, std::complex<double> alpha,
              std::complex<double> beta, bool hermitian, bool complex, std::span<const int> items) {
    check_items(A, items, "rank2k_backward_error");
    check_items(B, items, "rank2k_backward_error");
    check_items(C, items, "rank2k_backward_error");
    const bool use_c0 = beta != cd(0);
    if (use_c0) check_items(*C0, items, "rank2k_backward_error");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const bool nt = trans == Transpose::NoTrans;
    const int n = C.rows, k = nt ? A.cols : A.rows;
    Lane& e = eng.lane(2.0 * n * k * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        using X = K<D>;
        const X a1 = kmake<X>(alpha), a2 = hermitian ? conj(a1) : a1;
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * std::max(2 * k, n))) {
            const int g = int(grp.size());
            const auto* a = kp(operand<D>(e, kA, A, grp, Shape::general));
            const auto* b = kp(operand<D>(e, kB, B, grp, Shape::general));
            // [op(A) op(B)] . [a1 back(op(B)); a2 back(op(A))] has exactly the rank-2k denominator.
            D* P = e.slot<D>(kR, std::size_t(g) * n * 2 * k);
            D* Q = e.slot<D>(kT, std::size_t(g) * 2 * k * n);
            auto *p = kp(P), *q = kp(Q);
            const int ar = A.rows, br = B.rows;
            const std::size_t per = std::size_t(n) * k, sa = std::size_t(A.rows) * A.cols, sb = std::size_t(B.rows) * B.cols;
            launch(e, std::size_t(g) * per, [=](std::size_t x) {
                const std::size_t it = x / per, rc = x % per;
                const int i = int(rc % std::size_t(n)), l = int(rc / std::size_t(n));
                const X va = op_at(a + it * sa, ar, trans, i, l), vb = op_at(b + it * sb, br, trans, i, l);
                X* pp = p + it * std::size_t(n) * 2 * k;
                X* qq = q + it * std::size_t(2 * k) * n;
                pp[std::size_t(l) * n + i] = va;
                pp[std::size_t(k + l) * n + i] = vb;
                qq[std::size_t(i) * 2 * k + l] = a1 * (hermitian ? conj(vb) : vb);
                qq[std::size_t(i) * 2 * k + k + l] = a2 * (hermitian ? conj(va) : va);
            });
            D* c0 = use_c0 ? operand<D>(e, kC, *C0, grp, Shape::general) : nullptr;
            D* c = operand<D>(e, kD, C, grp, Shape::general);
            worst = nanmax(worst, componentwise_core(e, P, n, 2 * k, Transpose::NoTrans, Q, 2 * k, n, Transpose::NoTrans, c0, c, g,
                                                     uplo == Uplo::Lower ? Shape::lower : Shape::upper, cd(1), beta, hermitian, false));
        }
        return worst;
    });
}

double trsm(const Operand& A, Shape s, Transpose ta, bool left, const Operand& X, const Operand& B0, std::complex<double> alpha, bool complex,
            std::span<const int> items) {
    for (const Operand* o : {&A, &X, &B0}) check_items(*o, items, "trsm_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int na = A.rows, m = X.rows, n = X.cols;
    Lane& e = eng.lane(double(m) * na * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(na, m)) * std::max(na, n))) {
            const int g = int(grp.size());
            D* a = operand<D>(e, kA, A, grp, s);
            D* x = operand<D>(e, kB, X, grp, Shape::general);
            D* b = operand<D>(e, kC, B0, grp, Shape::general);
            D* R = left ? product(e, kR, a, na, na, ta, x, m, n, Transpose::NoTrans, g)
                        : product(e, kR, x, m, n, Transpose::NoTrans, a, na, na, ta, g);
            double* nm = e.slot<double>(kNorm, 4 * std::size_t(g));
            const cd al = alpha;
            D ad;
            if constexpr (std::is_same_v<D, double>) ad = al.real();
            else ad = al;
            difference(e, R, b, ad, m, n, g, Shape::general, nm, nm + g);
            frob(e, a, na, na, g, nm + 2 * g);
            frob(e, x, m, n, g, nm + 3 * g);
            e.q().wait();
            for (int it = 0; it < g; ++it)
                worst = nanmax(worst, hquot(nm[it], nm[2 * g + it] * nm[3 * g + it] + std::abs(alpha) * nm[g + it]));
        }
        return worst;
    });
}

double orthogonality(const Operand& Q, bool complex, std::span<const int> items) {
    check_items(Q, items, "orthogonality");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int m = Q.rows, c = Q.cols;
    Lane& e = eng.lane(double(c) * m * c * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(m, c)) * c)) {
            const int g = int(grp.size());
            D* q = operand<D>(e, kA, Q, grp, Shape::general);
            D* G = product(e, kR, q, m, c, Transpose::ConjTrans, q, m, c, Transpose::NoTrans, g);
            double* nm = e.slot<double>(kNorm, 2 * std::size_t(g));
            difference<D>(e, G, nullptr, D(1), c, c, g, Shape::general, nm, nm + g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, nm[it]);
        }
        return worst;
    });
}

double eigen(const Operand& A, const Operand& V, const Operand& w, bool complex, std::span<const int> items) {
    for (const Operand* o : {&A, &V, &w}) check_items(*o, items, "eigen_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int n = V.rows, nv = V.cols;
    Lane& e = eng.lane(double(n) * nv * (n + nv) * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * std::max(n, nv))) {
            const int g = int(grp.size());
            D* a = operand<D>(e, kA, A, grp, Shape::hermitian_lower);
            D* v = operand<D>(e, kB, V, grp, Shape::general);
            // diag(w) per item, written by the host; V diag(w) is then one more product.
            D* W = e.slot<D>(kC, std::size_t(g) * nv * nv);
            std::fill(W, W + std::size_t(g) * nv * nv, D(0));
            for (int it = 0; it < g; ++it)
                for (int j = 0; j < nv; ++j)
                    W[std::size_t(it) * nv * nv + std::size_t(j) * (nv + 1)] =
                        D(read_as<cd>(w, static_cast<long long>(grp[std::size_t(it)]) * w.stride + static_cast<long long>(j) * w.inc).real());
            D* R = product(e, kR, a, n, n, Transpose::NoTrans, v, n, nv, Transpose::NoTrans, g);
            D* T = product(e, kT, v, n, nv, Transpose::NoTrans, W, nv, nv, Transpose::NoTrans, g);
            double* nm = e.slot<double>(kNorm, 3 * std::size_t(g));
            difference(e, R, T, D(1), n, nv, g, Shape::general, nm, nm + g);
            frob(e, a, n, n, g, nm + 2 * g);
            e.q().wait();
            for (int it = 0; it < g; ++it) worst = nanmax(worst, hquot(nm[it], nm[2 * g + it]));
        }
        return worst;
    });
}

#if BATCHLAS_VERIFY_HAVE_LAPACKE
std::string host_dgemm_defect() {
    static const std::string defect = [] {
        const auto health = batchlas::detail::probe_dgemm([](int n, const double* a, const double* b, double* c) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
        });
        return health.ok ? std::string() : batchlas::detail::host_blas_double_health_message(health);
    }();
    return defect;
}
#endif

}  // namespace batchlas::verify::detail
