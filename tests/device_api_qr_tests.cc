// SPDX-License-Identifier: MIT
// device_api_qr_tests: the public device geqrf / orgqr / ormqr in tier::Blocked<NB> (v22 design C.7,
// B3: include/batchlas/device/lapack/{geqrf,orgqr,ormqr,larfb}.hh) called from a user-style
// kernel, one matrix per work-group of 256 items, against long-double LAPACK references (xGEQR2 with
// xLARFG's convention; op(Q) applied reflector by reflector) and, on a GPU, against the host blocked
// routes pinned and reached, within the library's calibrated bounds (never bit for bit: the host
// panel leaf and gemm kernels differ).
// Guards (agent-guide section 8): orders across the block width (NB - 1, NB, NB + 1, 2 NB + 1, and
// a wide m < n for geqrf); all four (Side, Trans) of ormqr; complex data with Im(diag) != 0; ld =
// m + 3 and a batch stride past ld * n, the pads holding a large finite value that must stay; the
// global workspace NaN-poisoned (each slot written before it is read); orgqr / ormqr fed
// T-rounded reference factors, so their references are exact for the inputs they get; inputs
// 4 I + 0.2 G (cond <= ~5), so a forward comparison has the bound's headroom; a 1024-item batch of
// period-3 data bit-identical to its representative; with_tier's pick, its sub-group /
// local-memory / work-group / fp64 filters and its op-named throw; buffer_size against the layout
// by hand; the -i argument returns; and a deliberate break (op(Q) checked against the OTHER
// Trans's reference must fail the same bound, so the comparison can fail).
// Red set: not recorded yet (the file was written without a GPU run).
#include <tests/device_core_common.hh>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/device/lapack/geqrf.hh>
#include <batchlas/device/lapack/orgqr.hh>
#include <batchlas/device/lapack/ormqr.hh>
#include <batchlas/error.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/geqrf/choice.hh>
#include <src/ops/orgqr/choice.hh>
#include <src/ops/ormqr/choice.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace device_core_test;
using batchlas::Side;
using batchlas::Transpose;
using batchlas::verify::Check;

namespace {

constexpr int kWG = 256;
constexpr int kItems = 3;
constexpr int kBig = 1024;
constexpr int kPeriod = 3;
constexpr int kTail = 7;  // poisoned elements past the last tau / workspace slice
template <class T> constexpr int kNB = dv::larfb_default_nb<T>();
template <class T> using Blk = dv::tier::Blocked<kNB<T>>;
static_assert(dv::tier_info<dv::routine::geqrf, Blk<float>, float>::wg_size == kWG);
static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::geqrf, double, dv::tier::upto<4096>>, dv::tier::Blocked<16>>);
static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::orgqr, float, dv::tier::upto<1>>, dv::tier::Blocked<32>>);

struct alignas(16) Chunk16 {
    unsigned char b[16];
};

template <class T> T pad_value() { return T(real_of<T>(1e30)); }
template <class T> LD<T> cj(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

// A batch of column-major rows x cols matrices, ld = rows + 3, stride = ld * cols + 5; every element
// outside them is pad_value(). Item q holds item q % period's data: shift * I + scale * G.
template <class T> struct Batch {
    int rows, cols, ld, items;
    std::size_t stride;
    std::vector<T> h;
    Batch(int r, int c, int n_items, std::uint64_t seed, double shift, double scale, int period = 1 << 30)
        : rows(r), cols(c), ld(r + 3), items(n_items), stride(std::size_t(r + 3) * c + 5), h(stride * n_items, pad_value<T>()) {
        for (int q = 0; q < items; ++q)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i) {
                    const T g = val<T>(seed * 1000003u + off(q % period, i, j));
                    at(q, i, j) = T(real_of<T>(scale)) * g + (i == j ? T(real_of<T>(shift)) : T(0));
                }
    }
    std::size_t off(int q, int i, int j) const { return std::size_t(q) * stride + i + std::size_t(j) * ld; }
    T& at(int q, int i, int j) { return h[off(q, i, j)]; }
    const T& at(int q, int i, int j) const { return h[off(q, i, j)]; }
    bool inside(std::size_t e) const {
        const std::size_t r = e % stride;
        return e < stride * items && r < std::size_t(ld) * cols && int(r % ld) < rows;
    }
    std::vector<LD<T>> wide_item(int q) const {
        std::vector<LD<T>> w(std::size_t(rows) * cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) w[i + std::size_t(j) * rows] = wide(at(q, i, j));
        return w;
    }
    void set_item(int q, const std::vector<LD<T>>& w) {
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) at(q, i, j) = static_cast<T>(w[i + std::size_t(j) * rows]);
    }
};

// ---- long-double references ------------------------------------------------------------------

// xGEQR2: xLARFG (beta = -sign(Re alpha) ||(alpha, x)||, tau = (beta - alpha) / beta, x /= alpha -
// beta; tau = 0 when x = 0 and Im alpha = 0), then H^H = I - conj(tau) v v^H from the left.
template <class T> void geqr2_ref(std::vector<LD<T>>& a, int m, int n, std::vector<LD<T>>& tau) {
    using L = long double;
    const int k = std::min(m, n);
    tau.assign(k, LD<T>(0));
    for (int j = 0; j < k; ++j) {
        const LD<T> alpha = a[j + std::size_t(j) * m];
        L xn2 = 0;
        for (int i = j + 1; i < m; ++i) xn2 += std::norm(a[i + std::size_t(j) * m]);
        const L ar = std::real(alpha), ai = std::imag(alpha);
        if (xn2 == 0 && ai == 0) continue;
        const L beta = -std::copysign(std::sqrt(ar * ar + ai * ai + xn2), ar);
        if constexpr (is_cplx<T>) tau[j] = LD<T>((beta - ar) / beta, -ai / beta);
        else tau[j] = (beta - ar) / beta;
        const LD<T> s = LD<T>(1) / (alpha - LD<T>(beta));
        for (int i = j + 1; i < m; ++i) a[i + std::size_t(j) * m] *= s;
        a[j + std::size_t(j) * m] = LD<T>(beta);
        const LD<T> ct = cj<T>(tau[j]);
        for (int c = j + 1; c < n; ++c) {
            LD<T> w = a[j + std::size_t(c) * m];
            for (int i = j + 1; i < m; ++i) w += cj<T>(a[i + std::size_t(j) * m]) * a[i + std::size_t(c) * m];
            const LD<T> f = ct * w;
            a[j + std::size_t(c) * m] -= f;
            for (int i = j + 1; i < m; ++i) a[i + std::size_t(c) * m] -= f * a[i + std::size_t(j) * m];
        }
    }
}

// x (rows x cols) := op(Q) x (left) or x op(Q), Q = H_0 ... H_{k-1} of the factor f (order nq, ld
// nq), op = ^H when conj_q. H_i = I - tau_i v v^H, v = [0 .. 0, 1, f(i + 1 :, i)].
template <class T>
void apply_q_ref(const std::vector<LD<T>>& f, const std::vector<LD<T>>& tau, int nq, int k, bool left, bool conj_q, std::vector<LD<T>>& x,
                 int rows, int cols) {
    auto v = [&](int i, int r) { return r < i ? LD<T>(0) : (r == i ? LD<T>(1) : f[r + std::size_t(i) * nq]); };
    // Left Q x and right x Q^H apply H_{k-1} first; left Q^H x and right x Q apply H_0 first.
    const bool from_last = left != conj_q;
    for (int s = 0; s < k; ++s) {
        const int i = from_last ? k - 1 - s : s;
        const LD<T> t = conj_q ? cj<T>(tau[i]) : tau[i];
        if (left) {
            for (int c = 0; c < cols; ++c) {
                LD<T> w = 0;
                for (int r = i; r < nq; ++r) w += cj<T>(v(i, r)) * x[r + std::size_t(c) * rows];
                for (int r = i; r < nq; ++r) x[r + std::size_t(c) * rows] -= t * v(i, r) * w;
            }
        } else {
            for (int r = 0; r < rows; ++r) {
                LD<T> w = 0;
                for (int c = i; c < nq; ++c) w += x[r + std::size_t(c) * rows] * v(i, c);
                for (int c = i; c < nq; ++c) x[r + std::size_t(c) * rows] -= t * w * cj<T>(v(i, c));
            }
        }
    }
}

// ||got(q) - ref||_F / ||ref||_F over a rows x cols item; NaN-propagating.
template <class T> double rel_err(const Batch<T>& layout, const std::vector<T>& got, int q, const std::vector<LD<T>>& ref) {
    long double num = 0, den = 0;
    for (int j = 0; j < layout.cols; ++j)
        for (int i = 0; i < layout.rows; ++i) {
            const LD<T> r = ref[i + std::size_t(j) * layout.rows];
            num += std::norm(wide(got[layout.off(q, i, j)]) - r);
            den += std::norm(r);
        }
    return static_cast<double>(std::sqrt(num) / std::sqrt(den));
}
template <class T> double rel_err_vec(const T* got, const std::vector<LD<T>>& ref) {
    long double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += std::norm(wide(got[i]) - ref[i]);
        den += std::norm(ref[i]);
    }
    return den == 0 ? static_cast<double>(std::sqrt(num)) : static_cast<double>(std::sqrt(num) / std::sqrt(den));
}
template <class T> int pads_changed(const Batch<T>& before, const std::vector<T>& after) {
    int bad = 0;
    for (std::size_t e = 0; e < after.size(); ++e)
        if (!before.inside(e) && !same_bits(before.h[e], after[e])) ++bad;
    return bad;
}

// ---- the user kernel ---------------------------------------------------------------------------

// items work-groups of kWG at sub-group 32, a 16-byte aligned local slab of local_bytes, the global
// workspace of items * global_bytes (+ a poisoned tail) NaN-filled; f(ex, b, ws) returns info[b].
template <class T> struct Launch {
    Dev& d;
    dv::WorkspaceSize sz;
    std::byte* gws = nullptr;
    int* info = nullptr;
    int items;
    Launch(Dev& dev, dv::WorkspaceSize s, int n_items) : d(dev), sz(s), items(n_items) {
        const std::size_t elems = (sz.global_bytes * items) / sizeof(T) + kTail;
        gws = reinterpret_cast<std::byte*>(d.alloc<T>(elems, dv::qnan<T>()));
        info = d.alloc<int>(items, 12345);
    }
    template <class F> std::vector<int> run(F f) {
        const std::size_t l16 = std::max<std::size_t>((sz.local_bytes + 15) / 16, 1);
        const dv::WorkspaceSize s = sz;
        std::byte* const g = gws;
        int* const inf = info;
        d.q().submit([&](sycl::handler& h) {
             sycl::local_accessor<Chunk16, 1> slm(sycl::range<1>(l16), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(items) * kWG, kWG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 const dv::WorkGroup<kWG, 32> ex{it};
                 const int b = static_cast<int>(it.get_group_linear_id());
                 auto* raw = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(slm));
                 const dv::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(raw),
                                        dv::global_slice(g, s, b)};
                 const int r = f(ex, b, ws);
                 if (ex.leader()) inf[b] = r;
             });
         }).wait_and_throw();
        return d.down(info, std::size_t(items));
    }
};

template <class T> batchlas::KernelMatrixView<T, batchlas::MatrixFormat::Dense> kview(T* p, const Batch<T>& b) {
    return batchlas::KernelMatrixView<T, batchlas::MatrixFormat::Dense>(p, b.rows, b.cols, b.ld, int(b.stride), b.items);
}
template <class T> batchlas::MatrixView<T, batchlas::MatrixFormat::Dense> hview(T* p, const Batch<T>& b) {
    return batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>(p, b.rows, b.cols, b.ld, int(b.stride), b.items);
}

struct Shape {
    int m, n;
};
// Across NB (n = NB - 1, NB, NB + 1, 2 NB + 1, tall) and one wide geqrf (m < n).
template <class T> std::vector<Shape> shapes(bool with_wide) {
    constexpr int nb = kNB<T>;
    std::vector<Shape> s{{nb + 6, nb - 1}, {nb + 7, nb}, {nb + 8, nb + 1}, {2 * nb + 9, 2 * nb + 1}};
    if (with_wide) s.push_back({nb + 1, 2 * nb + 3});
    return s;
}

// Device geqrf on a copy of A; returns (factors, tau, info).
template <class T> struct QrOut {
    std::vector<T> f, tau;
    std::vector<int> info;
};
template <class T> QrOut<T> device_geqrf(Dev& d, const Batch<T>& A) {
    const int k = std::min(A.rows, A.cols);
    T* a = d.up(A.h);
    T* tau = d.alloc<T>(std::size_t(k) * A.items + kTail, pad_value<T>());
    const auto kv = kview(a, A);
    Launch<T> L(d, dv::geqrf_buffer_size<Blk<T>, T>(A.rows, A.cols), A.items);
    auto info = L.run([=](const auto& ex, int b, const dv::Workspace& ws) {
        return dv::geqrf<Blk<T>>(ex, dv::item(kv, b), dv::vec(tau + std::ptrdiff_t(b) * k, k), ws);
    });
    return {d.down(a, A.h.size()), d.down(tau, std::size_t(k) * A.items + kTail), info};
}

// The T-rounded long-double factors of A (geqr2_ref), as orgqr's / ormqr's inputs.
template <class T> std::vector<T> ref_factors(Batch<T>& F, const Batch<T>& A) {
    const int k = std::min(A.rows, A.cols);
    std::vector<T> tau(std::size_t(k) * A.items + kTail, pad_value<T>());
    for (int q = 0; q < A.items; ++q) {
        auto w = A.wide_item(q);
        std::vector<LD<T>> t;
        geqr2_ref<T>(w, A.rows, A.cols, t);
        F.set_item(q, w);
        for (int i = 0; i < k; ++i) tau[std::size_t(q) * k + i] = static_cast<T>(t[i]);
    }
    return tau;
}
template <class T> std::vector<LD<T>> wide_tau(const std::vector<T>& tau, int q, int k) {
    std::vector<LD<T>> t(k);
    for (int i = 0; i < k; ++i) t[i] = wide(tau[std::size_t(q) * k + i]);
    return t;
}

template <class T> bool on_gpu(Dev& d) { return d.ctx->device().type == batchlas::DeviceType::GPU; }

template <class Choice, class F> bool reached(const char* op, F&& run) {
    const auto rows = test_support::reached_rows([&] {
        const batchlas::select::ScopedPin<Choice> pin(op, "blocked", batchlas::select::StrictPin{});
        run();
    });
    bool hit = false;
    for (const auto& r : rows) hit = hit || (r.op == op && r.spelling == "blocked");
    return hit;
}

template <class T> class DeviceApiQrTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceApiQrTest, AllTypes);

// ---- geqrf ---------------------------------------------------------------------------------------

TYPED_TEST(DeviceApiQrTest, GeqrfBlockedMatchesReferenceAndHostRoute) {
    using T = TypeParam;
    Dev& d = this->dev;
    for (const Shape s : shapes<T>(true)) {
        SCOPED_TRACE("m=" + std::to_string(s.m) + " n=" + std::to_string(s.n));
        const int k = std::min(s.m, s.n);
        const Batch<T> A(s.m, s.n, kItems, 21, 4.0, 0.2);
        const auto out = device_geqrf(d, A);
        for (int q = 0; q < kItems; ++q) EXPECT_EQ(out.info[q], 0);
        EXPECT_EQ(pads_changed(A, out.f), 0) << "geqrf wrote outside the matrices";
        for (int e = 0; e < kTail; ++e) EXPECT_TRUE(same_bits(out.tau[std::size_t(k) * kItems + e], pad_value<T>()));
        for (int q = 0; q < kItems; ++q) {
            auto ref = A.wide_item(q);
            std::vector<LD<T>> tref;
            geqr2_ref<T>(ref, s.m, s.n, tref);
            EXPECT_VERIFY(T, Check::factorization, s.m, rel_err(A, out.f, q, ref)) << "factors, item " << q;
            EXPECT_VERIFY(T, Check::factorization, s.m, rel_err_vec(out.tau.data() + std::size_t(q) * k, tref)) << "tau, item " << q;
        }
        if (!on_gpu<T>(d)) continue;
        T* ha = d.up(A.h);
        T* ht = d.alloc<T>(std::size_t(k) * kItems, T(0));
        EXPECT_TRUE(reached<batchlas::ops::geqrf::GeqrfChoice>("geqrf", [&] {
            (void)batchlas::geqrf(*d.ctx, hview(ha, A), batchlas::Span<T>(ht, std::size_t(k) * kItems));
            d.ctx->wait();
        })) << "the pinned host blocked geqrf was not reached";
        const auto hf = d.down(ha, A.h.size()), htau = d.down(ht, std::size_t(k) * kItems);
        for (int q = 0; q < kItems; ++q) {
            std::vector<LD<T>> hw(std::size_t(s.m) * s.n), ht_w(k);
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i) hw[i + std::size_t(j) * s.m] = wide(hf[A.off(q, i, j)]);
            for (int i = 0; i < k; ++i) ht_w[i] = wide(htau[std::size_t(q) * k + i]);
            EXPECT_VERIFY(T, Check::factorization, s.m, rel_err(A, out.f, q, hw)) << "device vs host factors, item " << q;
            EXPECT_VERIFY(T, Check::factorization, s.m, rel_err_vec(out.tau.data() + std::size_t(q) * k, ht_w)) << "device vs host tau";
        }
    }
}

// ---- orgqr ---------------------------------------------------------------------------------------

TYPED_TEST(DeviceApiQrTest, OrgqrBlockedMatchesReferenceAndHostRoute) {
    using T = TypeParam;
    Dev& d = this->dev;
    for (const Shape s : shapes<T>(false)) {
        SCOPED_TRACE("m=" + std::to_string(s.m) + " n=" + std::to_string(s.n));
        const Batch<T> A(s.m, s.n, kItems, 31, 4.0, 0.2);
        Batch<T> F = A;
        const auto tau = ref_factors(F, A);
        T* a = d.up(F.h);
        T* tp = d.up(tau);
        const auto kv = kview(a, F);
        const int n = s.n;
        Launch<T> L(d, dv::orgqr_buffer_size<Blk<T>, T>(s.m, s.n), kItems);
        const auto info = L.run([=](const auto& ex, int b, const dv::Workspace& ws) {
            return dv::orgqr<Blk<T>>(ex, dv::item(kv, b), tp + std::ptrdiff_t(b) * n, ws);
        });
        const auto got = d.down(a, F.h.size());
        for (int q = 0; q < kItems; ++q) EXPECT_EQ(info[q], 0);
        EXPECT_EQ(pads_changed(F, got), 0) << "orgqr wrote outside the matrices";
        for (int q = 0; q < kItems; ++q) {
            std::vector<LD<T>> qref(std::size_t(s.m) * n, LD<T>(0));
            for (int j = 0; j < n; ++j) qref[j + std::size_t(j) * s.m] = 1;
            apply_q_ref<T>(F.wide_item(q), wide_tau(tau, q, n), s.m, n, true, false, qref, s.m, n);
            EXPECT_VERIFY(T, Check::orthogonality, s.m, rel_err(F, got, q, qref)) << "Q, item " << q;
        }
        if (!on_gpu<T>(d)) continue;
        T* ha = d.up(F.h);
        T* ht = d.up(tau);
        EXPECT_TRUE(reached<batchlas::ops::orgqr::OrgqrChoice>("orgqr", [&] {
            const batchlas::select::ScopedPin<batchlas::ops::ormqr::OrmqrChoice> apply("ormqr", "blocked", batchlas::select::StrictPin{});
            (void)batchlas::orgqr(*d.ctx, hview(ha, F), batchlas::Span<T>(ht, std::size_t(n) * kItems));
            d.ctx->wait();
        })) << "the pinned host blocked orgqr was not reached";
        const auto hq = d.down(ha, F.h.size());
        for (int q = 0; q < kItems; ++q) {
            std::vector<LD<T>> hw(std::size_t(s.m) * n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < s.m; ++i) hw[i + std::size_t(j) * s.m] = wide(hq[F.off(q, i, j)]);
            EXPECT_VERIFY(T, Check::orthogonality, s.m, rel_err(F, got, q, hw)) << "device vs host Q, item " << q;
        }
    }
}

// ---- ormqr ---------------------------------------------------------------------------------------

template <class T, Side SD, Transpose TR> void check_ormqr(Dev& d, const Shape s) {
    constexpr bool left = SD == Side::Left;
    constexpr int kQ = 45;  // C's other extent: off the 32-tile grid
    const int k = std::min(s.m, s.n);
    const Batch<T> A(s.m, s.n, kItems, 41, 4.0, 0.2);
    Batch<T> F = A;
    const auto tau = ref_factors(F, A);
    const Batch<T> C(left ? s.m : kQ, left ? kQ : s.m, kItems, 43, 0.0, 1.0);
    T* a = d.up(F.h);
    T* tp = d.up(tau);
    T* c = d.up(C.h);
    const auto av = kview(a, F);
    const auto cv = kview(c, C);
    Launch<T> L(d, dv::ormqr_buffer_size<SD, TR, Blk<T>, T>(C.rows, C.cols), kItems);
    const auto info = L.run([=](const auto& ex, int b, const dv::Workspace& ws) {
        return dv::ormqr<SD, TR, Blk<T>>(ex, dv::item(av, b), dv::item(cv, b), dv::vec(tp + std::ptrdiff_t(b) * k, k), ws);
    });
    const auto got = d.down(c, C.h.size());
    EXPECT_EQ(pads_changed(C, got), 0) << "ormqr wrote outside C";
    EXPECT_EQ(pads_changed(F, d.down(a, F.h.size())), 0);
    const bool conj_q = TR != Transpose::NoTrans;
    for (int q = 0; q < kItems; ++q) {
        EXPECT_EQ(info[q], 0);
        auto ref = C.wide_item(q), other = C.wide_item(q);
        apply_q_ref<T>(F.wide_item(q), wide_tau(tau, q, k), s.m, k, left, conj_q, ref, C.rows, C.cols);
        EXPECT_VERIFY(T, Check::factorization, s.m, rel_err(C, got, q, ref)) << "op(Q) C, item " << q;
        // The deliberate break: the same result against the other Trans's reference must fail.
        apply_q_ref<T>(F.wide_item(q), wide_tau(tau, q, k), s.m, k, left, !conj_q, other, C.rows, C.cols);
        EXPECT_FALSE(batchlas::verify::within<T>(Check::factorization, s.m, rel_err(C, got, q, other)))
            << "the comparison cannot tell Q from Q^H";
    }
    if (!on_gpu<T>(d)) return;
    T* ha = d.up(F.h);
    T* hc = d.up(C.h);
    T* ht = d.up(tau);
    EXPECT_TRUE(reached<batchlas::ops::ormqr::OrmqrChoice>("ormqr", [&] {
        (void)batchlas::ormqr(*d.ctx, hview(ha, F), hview(hc, C), batchlas::Span<T>(ht, std::size_t(k) * kItems),
                              batchlas::OrmqrOptions{SD, TR, kNB<T>});
        d.ctx->wait();
    })) << "the pinned host blocked ormqr was not reached";
    const auto hgot = d.down(hc, C.h.size());
    for (int q = 0; q < kItems; ++q) {
        std::vector<LD<T>> hw(std::size_t(C.rows) * C.cols);
        for (int j = 0; j < C.cols; ++j)
            for (int i = 0; i < C.rows; ++i) hw[i + std::size_t(j) * C.rows] = wide(hgot[C.off(q, i, j)]);
        EXPECT_VERIFY(T, Check::factorization, s.m, rel_err(C, got, q, hw)) << "device vs host, item " << q;
    }
}

TYPED_TEST(DeviceApiQrTest, OrmqrBlockedAllSidesAndTrans) {
    using T = TypeParam;
    for (const Shape s : shapes<T>(false)) {
        SCOPED_TRACE("m=" + std::to_string(s.m) + " n=" + std::to_string(s.n));
        check_ormqr<T, Side::Left, Transpose::NoTrans>(this->dev, s);
        check_ormqr<T, Side::Left, Transpose::ConjTrans>(this->dev, s);
        check_ormqr<T, Side::Right, Transpose::NoTrans>(this->dev, s);
        check_ormqr<T, Side::Right, Transpose::ConjTrans>(this->dev, s);
    }
}

// ---- batch bit-identity (local memory reused across tiles) ---------------------------------------

template <class T> int items_differing(const Batch<T>& layout, const std::vector<T>& got) {
    int bad = 0;
    for (int q = kPeriod; q < layout.items; ++q) {
        bool same = true;
        for (int j = 0; j < layout.cols && same; ++j)
            for (int i = 0; i < layout.rows && same; ++i) same = same_bits(got[layout.off(q, i, j)], got[layout.off(q % kPeriod, i, j)]);
        bad += !same;
    }
    return bad;
}

TYPED_TEST(DeviceApiQrTest, BatchOf1024IsBitIdenticalPerRepresentative) {
    using T = TypeParam;
    Dev& d = this->dev;
    const int m = kNB<T> + 8, n = kNB<T> + 1, k = n;
    const Batch<T> A(m, n, kBig, 51, 4.0, 0.2, kPeriod);
    const auto qr = device_geqrf(d, A);
    EXPECT_EQ(items_differing(A, qr.f), 0) << "geqrf";
    int tau_bad = 0;
    for (int q = kPeriod; q < kBig; ++q) tau_bad += std::memcmp(&qr.tau[std::size_t(q) * k], &qr.tau[std::size_t(q % kPeriod) * k], sizeof(T) * k) != 0;
    EXPECT_EQ(tau_bad, 0) << "geqrf tau";

    T* a = d.up(qr.f);
    T* tp = d.up(qr.tau);
    const Batch<T> C(m, 37, kBig, 53, 0.0, 1.0, kPeriod);
    T* c = d.up(C.h);
    const auto av = kview(a, A);
    const auto cv = kview(c, C);
    Launch<T> Lm(d, dv::ormqr_buffer_size<Side::Left, Transpose::ConjTrans, Blk<T>, T>(C.rows, C.cols), kBig);
    Lm.run([=](const auto& ex, int b, const dv::Workspace& ws) {
        return dv::ormqr<Side::Left, Transpose::ConjTrans, Blk<T>>(ex, dv::item(av, b), dv::item(cv, b), tp + std::ptrdiff_t(b) * k, ws);
    });
    EXPECT_EQ(items_differing(C, d.down(c, C.h.size())), 0) << "ormqr";

    Launch<T> Lo(d, dv::orgqr_buffer_size<Blk<T>, T>(m, n), kBig);
    Lo.run([=](const auto& ex, int b, const dv::Workspace& ws) { return dv::orgqr<Blk<T>>(ex, dv::item(av, b), tp + std::ptrdiff_t(b) * k, ws); });
    EXPECT_EQ(items_differing(A, d.down(a, A.h.size())), 0) << "orgqr";
}

// ---- argument errors (returned before any barrier, uniform) --------------------------------------

TEST(DeviceApiQrArgs, BadArgumentsReturnMinusI) {
    using T = float;
    Dev d;
    if (!d.ctx) GTEST_SKIP() << "no SYCL device visible";
    const auto sgs = d.q().get_device().get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sgs.begin(), sgs.end(), std::size_t(32)) == sgs.end()) GTEST_SKIP() << "sub-group size 32";
    const int m = 40, n = 33;
    const Batch<T> A(m, n, 1, 61, 4.0, 0.2);
    T* a = d.up(A.h);
    T* tp = d.alloc<T>(n, 0.f);
    const auto av = kview(a, A);
    const dv::WorkspaceSize sz = dv::orgqr_buffer_size<Blk<T>, T>(m, n);
    Launch<T> L(d, sz, 1);
    int* out = d.alloc<int>(6, 0);
    L.run([=](const auto& ex, int, const dv::Workspace& ws) {
        const dv::Workspace none{ws.local, nullptr};
        const auto flat = dv::mat(a, n, m, n);  // n x m: orgqr's m < n
        const int r0 = dv::geqrf<Blk<T>>(ex, dv::item(av, 0), dv::vec(tp, n - 1), ws);
        const int r1 = dv::geqrf<Blk<T>>(ex, dv::item(av, 0), tp, none);
        const int r2 = dv::orgqr<Blk<T>>(ex, flat, tp, ws);
        const int r3 = dv::orgqr<Blk<T>>(ex, dv::item(av, 0), dv::vec(tp, n - 1), ws);
        const int r4 = dv::ormqr<Side::Left, Transpose::NoTrans, Blk<T>>(ex, dv::item(av, 0), flat, tp, ws);
        const int r5 = dv::ormqr<Side::Right, Transpose::NoTrans, Blk<T>>(ex, dv::item(av, 0), dv::item(av, 0), dv::vec(tp, n - 1), ws);
        if (ex.leader()) {
            out[0] = r0, out[1] = r1, out[2] = r2, out[3] = r3, out[4] = r4, out[5] = r5;
        }
        return 0;
    });
    const auto r = d.down(out, 6);
    EXPECT_EQ(r[0], -2) << "geqrf: tau shorter than k";
    EXPECT_EQ(r[1], -3) << "geqrf: no global workspace";
    EXPECT_EQ(r[2], -1) << "orgqr: m < n";
    EXPECT_EQ(r[3], -2) << "orgqr: tau shorter than n";
    EXPECT_EQ(r[4], -1) << "ormqr: A.rows() is not C.rows()";
    EXPECT_EQ(r[5], -1) << "ormqr Right: A.rows() is not C.cols()";
    const auto after = d.down(a, A.h.size());
    int changed = 0;
    for (std::size_t e = 0; e < after.size(); ++e) changed += !same_bits(after[e], A.h[e]);
    EXPECT_EQ(changed, 0) << "a rejected call wrote A";
}

// ---- with_tier and buffer_size (host only) ------------------------------------------------------

template <class Routine, class T> std::string pick(const dv::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        dv::with_tier<Routine, T>(lim, n, [&]<class Tier>(Tier) { got = "Blocked<" + std::to_string(Tier::nb) + ">"; });
    } catch (const batchlas::invalid_argument& e) {
        got = e.what();
    }
    return got;
}

TEST(DeviceApiQrTier, WithTierPicksBlockedAndFilters) {
    const dv::DeviceLimits ok{48 * 1024, {16, 32}, 1024, true};
    for (int n : {0, 1, 31, 33, 4096}) {
        EXPECT_EQ((pick<dv::routine::geqrf, float>(ok, n)), "Blocked<32>") << n;
        EXPECT_EQ((pick<dv::routine::geqrf, double>(ok, n)), "Blocked<16>") << n;
        EXPECT_EQ((pick<dv::routine::orgqr, std::complex<double>>(ok, n)), "Blocked<32>") << n;
        EXPECT_EQ((pick<dv::routine::ormqr<Side::Right, Transpose::ConjTrans>, std::complex<float>>(ok, n)), "Blocked<32>") << n;
    }
    auto starts = [](const std::string& s, const char* p) { return s.rfind(p, 0) == 0; };
    dv::DeviceLimits no_sg32 = ok;
    no_sg32.sub_group_sizes = {8, 16};
    EXPECT_TRUE(starts(pick<dv::routine::geqrf, float>(no_sg32, 40), "geqrf: no device tier admits n=40"));
    dv::DeviceLimits no_fp64 = ok;
    no_fp64.fp64 = false;
    EXPECT_TRUE(starts(pick<dv::routine::orgqr, double>(no_fp64, 40), "orgqr: no device tier admits n=40"));
    EXPECT_EQ((pick<dv::routine::orgqr, float>(no_fp64, 40)), "Blocked<32>");
    using Cfg = dv::larfb_cfg_t<dv::alg::Default, std::complex<double>>;
    dv::DeviceLimits small_slm = ok;
    small_slm.local_mem_bytes = dv::l3::smem_bytes<Cfg>() - 1;
    EXPECT_TRUE(starts(pick<dv::routine::ormqr<Side::Left, Transpose::NoTrans>, std::complex<double>>(small_slm, 40),
                       "ormqr: no device tier admits n=40"));
    small_slm.local_mem_bytes = dv::l3::smem_bytes<Cfg>();
    EXPECT_EQ((pick<dv::routine::ormqr<Side::Left, Transpose::NoTrans>, std::complex<double>>(small_slm, 40)), "Blocked<32>")
        << "a launch at the local-memory ceiling is admitted";
    dv::DeviceLimits small_wg = ok;
    small_wg.max_work_group_size = 128;
    EXPECT_TRUE(starts(pick<dv::routine::geqrf, float>(small_wg, 40), "geqrf: no device tier admits n=40"));
}

// buffer_size against the host blocked routes' layout, written out by hand (16-byte slots).
TEST(DeviceApiQrTier, BufferSizeIsTheLayout) {
    auto r16 = [](std::size_t b) { return (b + 15) / 16 * 16; };
    using T = std::complex<float>;
    constexpr std::size_t e = sizeof(T);
    const std::size_t smem = r16(dv::l3::smem_bytes<dv::larfb_cfg_t<dv::alg::Default, T>>());
    const int m = 70, n = 41, nb = 32;
    const auto g = dv::geqrf_buffer_size<Blk<T>, T>(m, n);
    EXPECT_EQ(g.local_bytes, smem);
    EXPECT_EQ(g.global_bytes, r16(m * nb * e) + r16(nb * nb * e) + r16(2 * nb * (n - nb) * e));
    const auto gw = dv::geqrf_buffer_size<Blk<T>, T>(5, 9);  // k = 5 < NB: nb = 5, W spans n - 5
    EXPECT_EQ(gw.global_bytes, r16(5 * nb * e) + r16(nb * nb * e) + r16(2 * nb * (9 - 5) * e));
    const auto l = dv::ormqr_buffer_size<Side::Left, Transpose::ConjTrans, Blk<T>, T>(m, 45);
    EXPECT_EQ(l.global_bytes, r16(m * nb * e) + r16(nb * nb * e) + r16(2 * nb * 45 * e));
    const auto rr = dv::ormqr_buffer_size<Side::Right, Transpose::NoTrans, Blk<T>, T>(45, m);
    EXPECT_EQ(rr.global_bytes, r16(m * nb * e) + r16(nb * nb * e) + r16(2 * 45 * nb * e));
    const auto o = dv::orgqr_buffer_size<Blk<T>, T>(m, n);
    EXPECT_EQ(o.local_bytes, smem);
    EXPECT_EQ(o.global_bytes, r16(m * n * e) + r16(m * nb * e) + r16(nb * nb * e) + r16(2 * nb * n * e));
    EXPECT_EQ((dv::tier_info<dv::routine::geqrf, Blk<T>, T>::buffer_size(50)), (dv::geqrf_buffer_size<Blk<T>, T>(50, 50)));
}

}  // namespace
