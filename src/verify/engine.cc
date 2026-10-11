// SPDX-License-Identifier: MIT
// libbatchlas_verify: the checks of batchlas::verify as SYCL kernels on the library's own queue
// (docs/design/verification.md#verification-device-engine). Checked items are packed on the host in
// their own precision, promoted on the device, and every reduction is finished on the host in the
// host loops' order from per-element partials, so no device reduction or group algorithm is used.
#include <batchlas/verify/engine.hh>
#include <batchlas/verify/scalar.hh>

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

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void bad(const char* who, const char* what) {
    throw std::invalid_argument(std::string("batchlas::verify::") + who + ": " + what);
}

// Written-out complex<double>: no std::complex in device code (Annex G __muldc3, and AdaptiveCpp).
struct zd {
    double re, im;
};
inline zd operator+(zd a, zd b) { return {a.re + b.re, a.im + b.im}; }
inline zd operator-(zd a, zd b) { return {a.re - b.re, a.im - b.im}; }
inline zd operator*(zd a, zd b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
inline zd operator*(zd a, double s) { return {a.re * s, a.im * s}; }
inline double mag(double x) { return sycl::fabs(x); }
inline double mag(zd x) { return sycl::hypot(x.re, x.im); }
inline double conj(double x) { return x; }
inline zd conj(zd x) { return {x.re, -x.im}; }
inline double real_part(double x) { return x; }
inline zd real_part(zd x) { return {x.re, 0.0}; }
inline bool nan_of(double x) { return sycl::isnan(x); }

template <class D> D make_d(double re, double im) {
    if constexpr (std::is_same_v<D, double>) { (void)im; return re; }
    else return zd{re, im};
}
template <class D> D make_d(std::complex<double> z) { return make_d<D>(z.real(), z.imag()); }

inline double dquot(double num, double den) {
    if (nan_of(num) || nan_of(den)) return kNaN;
    return den > 0 ? num / den : num;
}
inline double hmax(double a, double b) { return nanmax(a, b); }
inline double hquot(double num, double den) {
    if (std::isnan(num) || std::isnan(den)) return kNaN;
    return den > 0 ? num / den : num;
}

std::size_t elem_bytes(Dtype t) { return t == Dtype::f32 || t == Dtype::i32 ? 4 : t == Dtype::c64 ? 16 : 8; }

template <class D> D load(const void* p, Dtype dt, std::size_t i) {
    double re = 0, im = 0;
    switch (dt) {
        case Dtype::f32: re = static_cast<const float*>(p)[i]; break;
        case Dtype::f64: re = static_cast<const double*>(p)[i]; break;
        case Dtype::c32: re = static_cast<const float*>(p)[2 * i]; im = static_cast<const float*>(p)[2 * i + 1]; break;
        case Dtype::c64: re = static_cast<const double*>(p)[2 * i]; im = static_cast<const double*>(p)[2 * i + 1]; break;
        case Dtype::i32: re = static_cast<const std::int32_t*>(p)[i]; break;
    }
    return make_d<D>(re, im);
}

// shaped() of residuals.hh on a packed item (ld = its rows): only the named triangle is read.
template <class D> D shaped(const void* raw, Dtype dt, std::size_t base, int ld, Shape s, int i, int j) {
    const auto g = [&](int r, int c) { return load<D>(raw, dt, base + std::size_t(c) * std::size_t(ld) + std::size_t(r)); };
    const D zero = make_d<D>(0, 0), one = make_d<D>(1, 0);
    switch (s) {
        case Shape::general: return g(i, j);
        case Shape::lower: return i >= j ? g(i, j) : zero;
        case Shape::upper: return i <= j ? g(i, j) : zero;
        case Shape::unit_lower: return i == j ? one : (i > j ? g(i, j) : zero);
        case Shape::unit_upper: return i == j ? one : (i < j ? g(i, j) : zero);
        case Shape::hermitian_lower: return i == j ? real_part(g(i, i)) : (i > j ? g(i, j) : conj(g(j, i)));
        case Shape::hermitian_upper: return i == j ? real_part(g(i, i)) : (i < j ? g(i, j) : conj(g(j, i)));
        case Shape::symmetric_lower: return i >= j ? g(i, j) : g(j, i);
        case Shape::symmetric_upper: return i <= j ? g(i, j) : g(j, i);
    }
    return zero;
}

inline bool masked(Shape s, int i, int j) {
    const bool lo = s == Shape::lower || s == Shape::unit_lower || s == Shape::hermitian_lower || s == Shape::symmetric_lower;
    const bool hi = s == Shape::upper || s == Shape::unit_upper || s == Shape::hermitian_upper || s == Shape::symmetric_upper;
    return (lo && i < j) || (hi && i > j);
}

// One device's in-order queue, its device slots (grow-only) and a pinned host arena. A check enqueues
// every copy and kernel and waits once: a synchronization costs ~1 ms on the CUDA adapter here.
// evidence: docs/design/verification.md#verification-device-engine
class Lane {
public:
    explicit Lane(sycl::queue* q) : q_(q), slots_(16) {}
    sycl::queue& q() { return *q_; }

    template <class X> X* slot(int id, std::size_t count) {
        auto& s = slots_.at(std::size_t(id));
        const std::size_t bytes = std::max<std::size_t>(count * sizeof(X), 1);
        if (s.second < bytes) {
            q_->wait();  // queued work may still use the old slot
            if (s.first) sycl::free(s.first, *q_);
            s.first = sycl::malloc_device(bytes, *q_);
            if (!s.first) throw std::runtime_error("batchlas::verify: device allocation failed");
            s.second = bytes;
        }
        return static_cast<X*>(s.first);
    }

    // The queue is idle between checks: fold last check's chunks into one.
    void begin() {
        if (chunks_.size() > 1) {
            std::size_t total = 0;
            for (auto& c : chunks_) {
                total += c.second;
                sycl::free(c.first, *q_);
            }
            chunks_.clear();
            add_chunk(total);
        }
        used_ = 0;
    }

    // Pinned host memory that stays valid (and in place) until the next begin().
    void* take(std::size_t bytes) {
        bytes = (bytes + 63) & ~std::size_t(63);
        if (chunks_.empty() || used_ + bytes > chunks_.back().second) {
            add_chunk(std::max({bytes, std::size_t(1) << 20, chunks_.empty() ? 0 : 2 * chunks_.back().second}));
            used_ = 0;
        }
        void* p = static_cast<unsigned char*>(chunks_.back().first) + used_;
        used_ += bytes;
        return p;
    }

private:
    void add_chunk(std::size_t bytes) {
        void* p = sycl::malloc_host(bytes, *q_);
        if (!p) throw std::runtime_error("batchlas::verify: host staging allocation failed");
        chunks_.emplace_back(p, bytes);
    }
    sycl::queue* q_;
    std::vector<std::pair<void*, std::size_t>> slots_;
    std::vector<std::pair<void*, std::size_t>> chunks_;
    std::size_t used_ = 0;
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
        l.begin();
        return l;
    }

private:
    static Lane* probe(sycl::info::device_type type) {
        for (const auto& d : sycl::device::get_devices(type)) {
            if (!d.has(sycl::aspect::fp64)) continue;
            try {
                auto q = std::make_unique<sycl::queue>(d, sycl::property::queue::in_order{});
                double* p = sycl::malloc_device<double>(1, *q);
                q->single_task([=] { p[0] = 1.0; }).wait_and_throw();
                sycl::free(p, *q);
                return new Lane(q.release());
            } catch (const sycl::exception&) {
            }
        }
        return nullptr;
    }
    Engine() : gpu_(probe(sycl::info::device_type::gpu)), cpu_(probe(sycl::info::device_type::cpu)) {
        if (!gpu_ && !cpu_) throw std::runtime_error("batchlas::verify: no SYCL device with fp64 runs the verification kernels");
    }
    std::mutex mu_;
    Lane* gpu_;
    Lane* cpu_;
};

enum Slot { kRawA, kRawB, kRawC, kRawD, kA, kB, kC, kD, kR, kT, kP1, kP2, kP3, kV, kPiv, kAux };

void check_items(const Operand& o, std::span<const int> items, const char* who) {
    for (int b : items)
        if (b < 0 || b >= o.batch) bad(who, "batch item out of range");
}

// Packs rows x cols of each item in `items` back to back (ld = rows) in the operand's own precision
// and copies them to `slot`.
const void* upload(Lane& e, int slot, const Operand& o, std::span<const int> items, int rows, int cols) {
    const std::size_t es = elem_bytes(o.dtype), per = std::size_t(rows) * std::size_t(cols) * es;
    const std::size_t bytes = per * items.size();
    auto* dst = e.slot<unsigned char>(slot, bytes);
    if (bytes == 0) return dst;
    auto* h = static_cast<unsigned char*>(e.take(bytes));
    const auto* src = static_cast<const unsigned char*>(o.data);
    for (std::size_t g = 0; g < items.size(); ++g) {
        const long long base = static_cast<long long>(items[g]) * o.stride;
        for (int c = 0; c < cols; ++c) {
            unsigned char* out = h + g * per + std::size_t(c) * std::size_t(rows) * es;
            const long long col = base + static_cast<long long>(c) * o.ld;
            if (o.inc == 1) std::memcpy(out, src + std::size_t(col) * es, std::size_t(rows) * es);
            else
                for (int r = 0; r < rows; ++r)
                    std::memcpy(out + std::size_t(r) * es, src + std::size_t(col + static_cast<long long>(r) * o.inc) * es, es);
        }
    }
    e.q().memcpy(dst, h, bytes);
    return dst;
}

// n doubles of device memory, copied to the arena: valid after the check's wait.
struct Fetched {
    const double* p;
    std::size_t n;
};
Fetched fetch(Lane& e, const double* d, std::size_t n) {
    auto* h = static_cast<double*>(e.take(std::max<std::size_t>(n, 1) * sizeof(double)));
    if (n) e.q().memcpy(h, d, n * sizeof(double));
    return {h, n};
}

template <class F> void launch(Lane& e, std::size_t n, F f) {
    if (n == 0) return;
    e.q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { f(std::size_t(id[0])); });
}

// out (g items of out_rows x out_cols, packed) = op(shaped(raw item)), raw packed with ld = src_rows.
template <class D>
D* expand(Lane& e, int slot, const void* raw, Dtype dt, int g, int src_rows, int src_cols, Shape s, Transpose t, int out_rows,
          int out_cols) {
    D* out = e.slot<D>(slot, std::size_t(g) * out_rows * out_cols);
    const std::size_t per_out = std::size_t(out_rows) * out_cols, per_src = std::size_t(src_rows) * src_cols;
    launch(e, std::size_t(g) * per_out, [=](std::size_t x) {
        const std::size_t it = x / per_out, rc = x % per_out;
        const int i = int(rc % std::size_t(out_rows)), j = int(rc / std::size_t(out_rows));
        const std::size_t base = it * per_src;
        D v;
        if (t == Transpose::NoTrans) v = shaped<D>(raw, dt, base, src_rows, s, i, j);
        else {
            v = shaped<D>(raw, dt, base, src_rows, s, j, i);
            if (t == Transpose::ConjTrans) v = conj(v);
        }
        out[x] = v;
    });
    return out;
}

template <class D> D* expand(Lane& e, int raw_slot, int slot, const Operand& o, std::span<const int> items, Shape s, Transpose t,
                             int out_rows, int out_cols) {
    const void* raw = upload(e, raw_slot, o, items, o.rows, o.cols);
    return expand<D>(e, slot, raw, o.dtype, int(items.size()), o.rows, o.cols, s, t, out_rows, out_cols);
}

// R = P Q per item, the l sum in order.
template <class D> D* product(Lane& e, int slot, const D* P, const D* Q, int g, int m, int kk, int n) {
    D* R = e.slot<D>(slot, std::size_t(g) * m * n);
    const std::size_t per = std::size_t(m) * n;
    launch(e, std::size_t(g) * per, [=](std::size_t x) {
        const std::size_t it = x / per, rc = x % per;
        const int i = int(rc % std::size_t(m)), j = int(rc / std::size_t(m));
        const D* p = P + it * std::size_t(m) * kk;
        const D* q = Q + it * std::size_t(kk) * n + std::size_t(j) * kk;
        D acc = make_d<D>(0, 0);
        for (int l = 0; l < kk; ++l) acc = acc + p[std::size_t(l) * m + i] * q[l];
        R[x] = acc;
    });
    return R;
}

// diff2 = |R - a T|^2 and tgt2 = |T|^2 per element; masked elements contribute 0. A null T is the identity.
template <class D> void partials(Lane& e, const D* R, const D* T, D a, int g, int m, int n, Shape mask, double*& diff2, double*& tgt2) {
    const std::size_t total = std::size_t(g) * m * n, per = std::size_t(m) * n;
    double* d2 = e.slot<double>(kP1, total);
    double* t2 = e.slot<double>(kP2, total);
    launch(e, total, [=](std::size_t x) {
        const std::size_t rc = x % per;
        const int i = int(rc % std::size_t(m)), j = int(rc / std::size_t(m));
        if (masked(mask, i, j)) { d2[x] = 0; t2[x] = 0; return; }
        const D z = T ? T[x] : make_d<D>(i == j ? 1 : 0, 0);
        const double d = mag(R[x] - a * z), t = mag(z);
        d2[x] = d * d;
        t2[x] = t * t;
    });
    diff2 = d2;
    tgt2 = t2;
}

template <class D> double* sumsq(Lane& e, int slot, const D* M, std::size_t total) {
    double* s = e.slot<double>(slot, total);
    launch(e, total, [=](std::size_t x) { const double t = mag(M[x]); s[x] = t * t; });
    return s;
}

// Host sums per item, in column-major order (the host loops' order).
std::vector<double> per_item_sums(const Fetched& v, int g) {
    std::vector<double> s(std::size_t(g), 0.0);
    const std::size_t per = g ? v.n / std::size_t(g) : 0;
    for (int it = 0; it < g; ++it)
        for (std::size_t x = 0; x < per; ++x) s[std::size_t(it)] += v.p[std::size_t(it) * per + x];
    return s;
}

// Items processed together: as many as keep the largest buffer under 2^25 elements.
std::vector<std::span<const int>> groups(std::span<const int> items, std::size_t per_item) {
    const std::size_t cap = std::max<std::size_t>(1, (std::size_t(1) << 25) / std::max<std::size_t>(per_item, 1));
    std::vector<std::span<const int>> out;
    for (std::size_t i = 0; i < items.size(); i += cap) out.push_back(items.subspan(i, std::min(cap, items.size() - i)));
    return out;
}

// sqrt(sum diff2) / sqrt(sum tgt2) per item of a Frobenius ratio.
template <class D> double frob_ratio(Lane& e, const D* R, const D* T, D a, int g, int m, int n, Shape mask, double worst,
                                     const std::vector<bool>* nan_items = nullptr) {
    double *d2, *t2;
    partials(e, R, T, a, g, m, n, mask, d2, t2);
    const std::size_t total = std::size_t(g) * m * n;
    const Fetched f2 = fetch(e, d2, total), ft = fetch(e, t2, total);
    e.q().wait();
    const auto num = per_item_sums(f2, g), den = per_item_sums(ft, g);
    for (int it = 0; it < g; ++it)
        worst = hmax(worst, nan_items && (*nan_items)[std::size_t(it)] ? kNaN : hquot(std::sqrt(num[std::size_t(it)]), std::sqrt(den[std::size_t(it)])));
    return worst;
}

template <class F> auto dispatch(bool complex, F f) { return complex ? f(zd{}) : f(double{}); }

std::vector<std::int32_t> read_pivots(const Operand& p, int item, int count) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    const auto* d = static_cast<const std::int32_t*>(p.data);
    for (int k = 0; k < count; ++k) out[std::size_t(k)] = d[static_cast<long long>(item) * p.stride + static_cast<long long>(k) * p.inc];
    return out;
}

// Applies the row swaps of each item (k in [0, count), in `forward` order or reversed) to M (m x n).
template <class D> void row_swaps(Lane& e, D* M, const std::vector<std::int32_t>& piv0, int g, int m, int n, int count, bool forward) {
    auto* piv = e.slot<std::int32_t>(kPiv, piv0.size());
    if (!piv0.empty()) {
        void* h = e.take(piv0.size() * sizeof(std::int32_t));
        std::memcpy(h, piv0.data(), piv0.size() * sizeof(std::int32_t));
        e.q().memcpy(piv, h, piv0.size() * sizeof(std::int32_t));
    }
    launch(e, std::size_t(g) * n, [=](std::size_t x) {
        const std::size_t it = x / std::size_t(n);
        const int c = int(x % std::size_t(n));
        D* col = M + it * std::size_t(m) * n + std::size_t(c) * m;
        const std::int32_t* p = piv + it * std::size_t(count);
        for (int s = 0; s < count; ++s) {
            const int k = forward ? s : count - 1 - s, ip = p[k];
            if (ip != k) { const D t = col[k]; col[k] = col[ip]; col[ip] = t; }
        }
    });
}

// x <- H_0 ... H_{k-1} x for each column of X (m x cols per item): reflector i below F's diagonal.
template <class D> void apply_q(Lane& e, D* X, const void* rawF, Dtype fdt, const void* rawTau, Dtype tdt, int g, int m, int n, int k,
                                int cols) {
    launch(e, std::size_t(g) * cols, [=](std::size_t x) {
        const std::size_t it = x / std::size_t(cols);
        const int j = int(x % std::size_t(cols));
        D* v = X + it * std::size_t(m) * cols + std::size_t(j) * m;
        const std::size_t fb = it * std::size_t(m) * n, tb = it * std::size_t(k);
        for (int i = k - 1; i >= 0; --i) {
            D s = v[i];
            for (int r = i + 1; r < m; ++r) s = s + conj(load<D>(rawF, fdt, fb + std::size_t(i) * m + r)) * v[r];
            s = s * load<D>(rawTau, tdt, tb + i);
            v[i] = v[i] - s;
            for (int r = i + 1; r < m; ++r) v[r] = v[r] - s * load<D>(rawF, fdt, fb + std::size_t(i) * m + r);
        }
    });
}

template <class D> D* fill_identity(Lane& e, int slot, int g, int m, int n) {
    D* out = e.slot<D>(slot, std::size_t(g) * m * n);
    const std::size_t per = std::size_t(m) * n;
    launch(e, std::size_t(g) * per, [=](std::size_t x) {
        const std::size_t rc = x % per;
        out[x] = make_d<D>(rc % std::size_t(m) == rc / std::size_t(m) ? 1 : 0, 0);
    });
    return out;
}

// The componentwise ratio (or denominator) per element of each item; host nanmax.
template <class D> double componentwise_core(Lane& e, const D* opA, const D* opB, const D* C0, const D* C, int g, int m, int kk, int n,
                                             Shape sc, D alpha, D beta, bool c0_real_diag, bool den_only, double worst) {
    const std::size_t per = std::size_t(m) * n, total = std::size_t(g) * per;
    double* out = e.slot<double>(kP1, total);
    const double aa = mag(alpha), ab = mag(beta);
    launch(e, total, [=](std::size_t x) {
        const std::size_t it = x / per, rc = x % per;
        const int i = int(rc % std::size_t(m)), j = int(rc / std::size_t(m));
        if (masked(sc, i, j)) { out[x] = 0; return; }
        const D* a = opA + it * std::size_t(m) * kk;
        const D* b = opB + it * std::size_t(kk) * n + std::size_t(j) * kk;
        D acc = make_d<D>(0, 0);
        double magsum = 0;
        for (int l = 0; l < kk; ++l) {
            const D al = a[std::size_t(l) * m + i];
            acc = acc + al * b[l];
            magsum += mag(al) * mag(b[l]);
        }
        D want = alpha * acc;
        double den = aa * magsum;
        if (C0) {
            D z = C0[x];
            if (c0_real_diag && i == j) z = real_part(z);
            want = want + beta * z;
            den += ab * mag(z);
        }
        out[x] = den_only ? den : dquot(mag(C[x] - want), den);
    });
    const Fetched f = fetch(e, out, total);
    e.q().wait();
    for (std::size_t x = 0; x < f.n; ++x) worst = hmax(worst, f.p[x]);
    return worst;
}

inline Shape mask_of(Uplo u) { return u == Uplo::Lower ? Shape::lower : Shape::upper; }

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
            const void* rawF = upload(e, kRawB, F, grp, n, n);
            const Shape s = upper ? Shape::upper : Shape::lower;
            D* X = expand<D>(e, kA, rawF, F.dtype, g, n, n, s, upper ? Transpose::ConjTrans : Transpose::NoTrans, n, n);
            D* Y = expand<D>(e, kB, rawF, F.dtype, g, n, n, s, upper ? Transpose::NoTrans : Transpose::ConjTrans, n, n);
            D* R = product<D>(e, kR, X, Y, g, n, n, n);
            D* T = expand<D>(e, kRawA, kT, A0, grp, Shape::general, Transpose::NoTrans, n, n);
            worst = frob_ratio<D>(e, R, T, make_d<D>(1, 0), g, n, n, s, worst);
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
            const void* rawQ = upload(e, kRawB, Q, grp, n, n);
            D* Qd = expand<D>(e, kA, rawQ, Q.dtype, g, n, n, Shape::general, Transpose::NoTrans, n, n);
            D* Qh = expand<D>(e, kB, rawQ, Q.dtype, g, n, n, Shape::general, Transpose::ConjTrans, n, n);
            D* Bd = expand<D>(e, kRawC, kC, B, grp, Shape::general, Transpose::NoTrans, n, n);
            D* QB = product<D>(e, kD, Qd, Bd, g, n, n, n);
            D* R = product<D>(e, kR, QB, Qh, g, n, n, n);
            D* T = expand<D>(e, kRawA, kT, A0, grp, Shape::general, Transpose::NoTrans, n, n);
            worst = frob_ratio<D>(e, R, T, make_d<D>(1, 0), g, n, n, Shape::general, worst);
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
            std::vector<std::int32_t> p0;
            std::vector<bool> nan_items(grp.size(), false);
            for (std::size_t it = 0; it < grp.size(); ++it) {
                auto p = read_pivots(piv, grp[it], mn);
                for (int k = 0; k < mn; ++k) {
                    if (p[std::size_t(k)] < 1 || p[std::size_t(k)] > m) { nan_items[it] = true; break; }
                }
                for (int k = 0; k < mn; ++k) p0.push_back(nan_items[it] ? k : p[std::size_t(k)] - 1);
            }
            D* PA = expand<D>(e, kRawA, kT, A0, grp, Shape::general, Transpose::NoTrans, m, n);
            row_swaps<D>(e, PA, p0, g, m, n, mn, true);
            const void* rawF = upload(e, kRawB, F, grp, m, n);
            D* L = expand<D>(e, kA, rawF, F.dtype, g, m, n, Shape::unit_lower, Transpose::NoTrans, m, mn);
            D* U = expand<D>(e, kB, rawF, F.dtype, g, m, n, Shape::upper, Transpose::NoTrans, mn, n);
            D* R = product<D>(e, kR, L, U, g, m, mn, n);
            worst = frob_ratio<D>(e, R, PA, make_d<D>(1, 0), g, m, n, Shape::general, worst, &nan_items);
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
            const void* rawF = upload(e, kRawB, F, grp, m, n);
            const void* rawT = upload(e, kRawC, tau, grp, k, 1);
            // Column j of R: rows r <= j, which also satisfy r < k (j < n, and k = m when m <= n).
            D* X = expand<D>(e, kR, rawF, F.dtype, g, m, n, Shape::upper, Transpose::NoTrans, m, n);
            apply_q<D>(e, X, rawF, F.dtype, rawT, tau.dtype, g, m, n, k, n);
            D* T = expand<D>(e, kRawA, kT, A0, grp, Shape::general, Transpose::NoTrans, m, n);
            worst = frob_ratio<D>(e, X, T, make_d<D>(1, 0), g, m, n, Shape::general, worst);
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
        const void* rawF = upload(e, kRawB, F, one, m, n);
        const void* rawT = upload(e, kRawC, tau, one, k, 1);
        D* X = fill_identity<D>(e, kR, 1, m, cols);
        apply_q<D>(e, X, rawF, F.dtype, rawT, tau.dtype, 1, m, n, k, cols);
        const std::size_t total = std::size_t(m) * cols * (sizeof(D) / sizeof(double));
        const Fetched f = fetch(e, reinterpret_cast<const double*>(X), total);
        e.q().wait();
        return std::vector<double>(f.p, f.p + f.n);
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
        for (auto grp : groups(items, std::size_t(std::max(m, R.rows)) * std::max(n, k))) {
            const int g = int(grp.size());
            D* Qd = expand<D>(e, kRawA, kA, Q, grp, Shape::general, Transpose::NoTrans, m, k);
            const void* rawR = upload(e, kRawB, R, grp, R.rows, n);
            D* Rd = expand<D>(e, kB, rawR, R.dtype, g, R.rows, n, Shape::upper, Transpose::NoTrans, k, n);
            D* P = product<D>(e, kR, Qd, Rd, g, m, k, n);
            D* T = expand<D>(e, kRawC, kT, A0, grp, Shape::general, Transpose::NoTrans, m, n);
            worst = frob_ratio<D>(e, P, T, make_d<D>(1, 0), g, m, n, Shape::general, worst);
        }
        return worst;
    });
}

double solve(const Operand& A, Shape shape, Transpose trans, const Operand* piv, const Operand& X, const Operand& B0, bool complex,
             std::span<const int> items, double full_limit) {
    check_items(A, items, "solve_residual");
    check_items(X, items, "solve_residual");
    if (piv) check_items(*piv, items, "lu_solve_residual");
    const bool identity = B0.rows == 0;
    if (!identity) check_items(B0, items, "solve_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const bool nt = trans == Transpose::NoTrans;
    const int m = nt ? A.rows : A.cols, k = nt ? A.cols : A.rows, nc = X.cols;
    const bool full = double(m) * double(k) * double(nc) <= full_limit;
    const int nv = full ? nc : 8;
    Lane& e = eng.lane((double(m) * k * nv + double(k) * nc * nv + (piv ? double(A.rows) * A.rows * A.rows : 0.0)) * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        // Probe columns: factor_bench's seeded draws, rounded to A's precision as draw<T> rounds them.
        std::vector<D> V;
        if (!full) {
            Rng rng(4242);
            V.resize(std::size_t(nc) * nv);
            for (D& v : V) {
                const double re = rng.next(), im = rng.next();
                const bool single = A.dtype == Dtype::f32 || A.dtype == Dtype::c32;
                const bool cx = A.dtype == Dtype::c32 || A.dtype == Dtype::c64;
                v = make_d<D>(single ? double(float(re)) : re, cx ? (single ? double(float(im)) : im) : 0.0);
            }
        }
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(m, k)) * std::max({m, k, nc}))) {
            const int g = int(grp.size());
            std::vector<bool> nan_items(grp.size(), false);
            D* M;
            if (piv) {
                const int n = A.rows;
                const void* rawF = upload(e, kRawA, A, grp, n, n);
                D* L = expand<D>(e, kA, rawF, A.dtype, g, n, n, Shape::unit_lower, Transpose::NoTrans, n, n);
                D* U = expand<D>(e, kB, rawF, A.dtype, g, n, n, Shape::upper, Transpose::NoTrans, n, n);
                D* LU = product<D>(e, kD, L, U, g, n, n, n);
                std::vector<std::int32_t> p0;
                for (std::size_t it = 0; it < grp.size(); ++it) {
                    auto p = read_pivots(*piv, grp[it], n);
                    for (int q = 0; q < n; ++q)
                        if (p[std::size_t(q)] < 1 || p[std::size_t(q)] > n) { nan_items[it] = true; break; }
                    for (int q = 0; q < n; ++q) p0.push_back(nan_items[it] ? q : p[std::size_t(q)] - 1);
                }
                row_swaps<D>(e, LU, p0, g, n, n, n, false);
                M = e.slot<D>(kA, std::size_t(g) * n * n);
                const std::size_t per = std::size_t(n) * n;
                launch(e, std::size_t(g) * per, [=](std::size_t x) {
                    const std::size_t it = x / per, rc = x % per;
                    const std::size_t r = rc % std::size_t(n), l = rc / std::size_t(n);
                    const D* src = LU + it * per;
                    M[x] = trans == Transpose::NoTrans ? src[l * n + r] : trans == Transpose::ConjTrans ? conj(src[r * n + l]) : src[r * n + l];
                });
            } else {
                M = expand<D>(e, kRawA, kA, A, grp, shape, trans, m, k);
            }
            D* Xd = expand<D>(e, kRawB, kB, X, grp, Shape::general, Transpose::NoTrans, k, nc);
            D* Bd = identity ? fill_identity<D>(e, kC, g, m, nc) : expand<D>(e, kRawC, kC, B0, grp, Shape::general, Transpose::NoTrans, m, nc);
            D *xv = Xd, *bv = Bd;
            if (!full) {
                std::vector<D> Vg;
                for (int it = 0; it < g; ++it) Vg.insert(Vg.end(), V.begin(), V.end());
                D* Vd = e.slot<D>(kV, Vg.size());
                void* h = e.take(Vg.size() * sizeof(D));
                std::memcpy(h, Vg.data(), Vg.size() * sizeof(D));
                e.q().memcpy(Vd, h, Vg.size() * sizeof(D));
                xv = product<D>(e, kD, Xd, Vd, g, k, nc, nv);
                bv = product<D>(e, kT, Bd, Vd, g, m, nc, nv);
            }
            D* R = product<D>(e, kR, M, xv, g, m, k, nv);
            double *d2, *t2;
            partials(e, R, bv, make_d<D>(1, 0), g, m, nv, Shape::general, d2, t2);
            const Fetched fnum = fetch(e, d2, std::size_t(g) * m * nv);
            const Fetched fnx = fetch(e, sumsq(e, kP2, xv, std::size_t(g) * k * nv), std::size_t(g) * k * nv);
            const Fetched fna = fetch(e, sumsq(e, kP3, M, std::size_t(g) * m * k), std::size_t(g) * m * k);
            e.q().wait();
            const auto num = per_item_sums(fnum, g), nx = per_item_sums(fnx, g), na = per_item_sums(fna, g);
            for (int it = 0; it < g; ++it) {
                const std::size_t u = std::size_t(it);
                worst = hmax(worst, nan_items[u] ? kNaN : hquot(std::sqrt(num[u]), std::sqrt(na[u]) * std::sqrt(nx[u])));
            }
        }
        return worst;
    });
}

double componentwise(const Operand& A, Shape sa, Transpose ta, const Operand& B, Shape sb, Transpose tb, const Operand* C0, const Operand* C,
                     Shape sc, std::complex<double> alpha, std::complex<double> beta, bool complex, std::span<const int> items,
                     bool denominator_only) {
    check_items(A, items, "gemm_backward_error");
    check_items(B, items, "gemm_backward_error");
    const bool use_c0 = beta != std::complex<double>(0);
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
            D* opA = expand<D>(e, kRawA, kA, A, grp, sa, ta, m, kk);
            D* opB = expand<D>(e, kRawB, kB, B, grp, sb, tb, kk, n);
            D* c0 = use_c0 ? expand<D>(e, kRawC, kC, *C0, grp, Shape::general, Transpose::NoTrans, m, n) : nullptr;
            D* c = denominator_only ? nullptr : expand<D>(e, kRawD, kD, *C, grp, Shape::general, Transpose::NoTrans, m, n);
            worst = componentwise_core<D>(e, opA, opB, c0, c, g, m, kk, n, sc, make_d<D>(alpha), make_d<D>(beta), false, denominator_only, worst);
        }
        return worst;
    });
}

double rank2k(const Operand& A, const Operand& B, Transpose trans, const Operand* C0, const Operand& C, Uplo uplo, std::complex<double> alpha,
              std::complex<double> beta, bool hermitian, bool complex, std::span<const int> items) {
    check_items(A, items, "rank2k_backward_error");
    check_items(B, items, "rank2k_backward_error");
    check_items(C, items, "rank2k_backward_error");
    const bool use_c0 = beta != std::complex<double>(0);
    if (use_c0) check_items(*C0, items, "rank2k_backward_error");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const bool nt = trans == Transpose::NoTrans;
    const int n = C.rows, k = nt ? A.cols : A.rows;
    Lane& e = eng.lane(2.0 * n * k * n * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        const D a1 = make_d<D>(alpha), a2 = hermitian ? conj(a1) : a1;
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * std::max(2 * k, n))) {
            const int g = int(grp.size());
            D* opA = expand<D>(e, kRawA, kA, A, grp, Shape::general, trans, n, k);
            D* opB = expand<D>(e, kRawB, kB, B, grp, Shape::general, trans, n, k);
            // [op(A) op(B)] . [a1 back(op(B)); a2 back(op(A))] has exactly the rank-2k denominator.
            D* P = e.slot<D>(kR, std::size_t(g) * n * 2 * k);
            D* Q = e.slot<D>(kT, std::size_t(g) * 2 * k * n);
            const std::size_t per = std::size_t(n) * k;
            launch(e, std::size_t(g) * per, [=](std::size_t x) {
                const std::size_t it = x / per, rc = x % per;
                const int i = int(rc % std::size_t(n)), l = int(rc / std::size_t(n));
                const D va = opA[x], vb = opB[x];
                D* p = P + it * std::size_t(n) * 2 * k;
                D* q = Q + it * std::size_t(2 * k) * n;
                p[std::size_t(l) * n + i] = va;
                p[std::size_t(k + l) * n + i] = vb;
                q[std::size_t(i) * 2 * k + l] = a1 * (hermitian ? conj(vb) : vb);
                q[std::size_t(i) * 2 * k + k + l] = a2 * (hermitian ? conj(va) : va);
            });
            D* c0 = use_c0 ? expand<D>(e, kRawC, kC, *C0, grp, Shape::general, Transpose::NoTrans, n, n) : nullptr;
            D* c = expand<D>(e, kRawD, kD, C, grp, Shape::general, Transpose::NoTrans, n, n);
            worst = componentwise_core<D>(e, P, Q, c0, c, g, n, 2 * k, n, mask_of(uplo), make_d<D>(1, 0), make_d<D>(beta), hermitian, false,
                                          worst);
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
        const D a = make_d<D>(alpha);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(std::max(na, m)) * std::max(na, n))) {
            const int g = int(grp.size());
            D* opA = expand<D>(e, kRawA, kA, A, grp, s, ta, na, na);
            D* Xd = expand<D>(e, kRawB, kB, X, grp, Shape::general, Transpose::NoTrans, m, n);
            D* Bd = expand<D>(e, kRawC, kC, B0, grp, Shape::general, Transpose::NoTrans, m, n);
            D* R = left ? product<D>(e, kR, opA, Xd, g, m, na, n) : product<D>(e, kR, Xd, opA, g, m, na, n);
            double *d2, *t2;
            partials(e, R, Bd, a, g, m, n, Shape::general, d2, t2);
            const std::size_t total = std::size_t(g) * m * n;
            const Fetched fnum = fetch(e, d2, total), fnb = fetch(e, t2, total);
            const Fetched fA = fetch(e, sumsq(e, kP3, opA, std::size_t(g) * na * na), std::size_t(g) * na * na);
            const Fetched fX = fetch(e, sumsq(e, kV, Xd, total), total);
            e.q().wait();
            const auto num = per_item_sums(fnum, g), nb = per_item_sums(fnb, g), nA = per_item_sums(fA, g), nX = per_item_sums(fX, g);
            for (int it = 0; it < g; ++it) {
                const std::size_t u = std::size_t(it);
                worst = hmax(worst, hquot(std::sqrt(num[u]), std::sqrt(nA[u]) * std::sqrt(nX[u]) + std::abs(alpha) * std::sqrt(nb[u])));
            }
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
            const void* raw = upload(e, kRawA, Q, grp, m, c);
            D* Qh = expand<D>(e, kA, raw, Q.dtype, g, m, c, Shape::general, Transpose::ConjTrans, c, m);
            D* Qd = expand<D>(e, kB, raw, Q.dtype, g, m, c, Shape::general, Transpose::NoTrans, m, c);
            D* G = product<D>(e, kR, Qh, Qd, g, c, m, c);
            double *d2, *t2;
            partials<D>(e, G, nullptr, make_d<D>(1, 0), g, c, c, Shape::general, d2, t2);
            const Fetched f = fetch(e, d2, std::size_t(g) * c * c);
            e.q().wait();
            for (double s : per_item_sums(f, g)) worst = hmax(worst, std::sqrt(s));
        }
        return worst;
    });
}

double eigen(const Operand& A, const Operand& V, const Operand& w, bool complex, std::span<const int> items) {
    for (const Operand* o : {&A, &V, &w}) check_items(*o, items, "eigen_residual");
    auto& eng = Engine::get();
    std::lock_guard<std::mutex> lock(eng.mutex());
    const int n = V.rows, nv = V.cols;
    Lane& e = eng.lane(double(n) * n * nv * double(items.size()));
    return dispatch(complex, [&](auto tag) {
        using D = decltype(tag);
        double worst = 0;
        for (auto grp : groups(items, std::size_t(n) * std::max(n, nv))) {
            const int g = int(grp.size());
            D* full = expand<D>(e, kRawA, kA, A, grp, Shape::hermitian_lower, Transpose::NoTrans, n, n);
            D* Vd = expand<D>(e, kRawB, kB, V, grp, Shape::general, Transpose::NoTrans, n, nv);
            D* lam = expand<D>(e, kRawC, kC, w, grp, Shape::general, Transpose::NoTrans, nv, 1);
            D* R = product<D>(e, kR, full, Vd, g, n, n, nv);
            D* T = e.slot<D>(kT, std::size_t(g) * n * nv);
            const std::size_t per = std::size_t(n) * nv;
            launch(e, std::size_t(g) * per, [=](std::size_t x) {
                const std::size_t it = x / per, j = (x % per) / std::size_t(n);
                if constexpr (std::is_same_v<D, double>) T[x] = Vd[x] * lam[it * nv + j];
                else T[x] = Vd[x] * lam[it * nv + j].re;
            });
            double *d2, *t2;
            partials(e, R, T, make_d<D>(1, 0), g, n, nv, Shape::general, d2, t2);
            const Fetched fnum = fetch(e, d2, std::size_t(g) * per);
            const Fetched fden = fetch(e, sumsq(e, kP3, full, std::size_t(g) * n * n), std::size_t(g) * n * n);
            e.q().wait();
            const auto num = per_item_sums(fnum, g), den = per_item_sums(fden, g);
            for (int it = 0; it < g; ++it) worst = hmax(worst, hquot(std::sqrt(num[std::size_t(it)]), std::sqrt(den[std::size_t(it)])));
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
