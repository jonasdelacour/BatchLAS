// SPDX-License-Identifier: MIT
// device_api_chol_tests: the Cholesky family's tier::Blocked (v22 design C.7, B1) called from a
// user-style kernel, one matrix per WorkGroup<256, 32>: device::potrf, potrs, posv, potri and
// trtri (lapack/{potrf,potrs,posv,potri}.hh) against the host routes PINNED to their multi-kernel
// families (potrf blocked, potrs trsm, posv blocked over potrf blocked, potri blocked:nb=64) on the
// same data: info for every item equal, the stored triangle within the op's verify bound of the
// host's, and an independent residual against the input (potrf_residual, solve_residual).
// Guards (agent-guide section 8): ld = rows + 3 and a padded batch stride, B at ld = n + 2 with
// nrhs = 3; a finite poison in the unread triangle and every pad that must survive bit for bit
// (the factor handed to potrs / potri is re-poisoned outside its triangle); complex data, Im(diag)
// poisoned on potrf / posv input (ignored by contract) and checked zero on the potri output; both
// triangles; n = 1, NB - 1, NB, NB + 1, 2 NB + 1, 8 NB + 1 and Blocked<8> at 9, 23, 37 (many
// panels); three representatives (SPD, an indefinite pivot at n / 2, a NaN last pivot; potri: an
// exactly zero diagonal entry) with the failing items left unchanged where the contract says so;
// a 1024-item batch of one repeated problem bit-identical to item 0 (local-memory kernels);
// arguments -2 / -3 / -4 (misaligned local, missing global) with A untouched; with_tier on a
// starved local memory, sub-group 16 and no fp64; buffer_size == the carved plan.
// Red set: not recorded (edit-only round). Intended breaks: potrf_blocked_op's herk alpha One{}
// (PotrfMatchesHostBlocked, n > NB); its Upper trsm Transpose::NoTrans (Upper only);
// potrs_blocked_solve's Lower trsm order swapped (Potrs, Lower only); trtri_blocked_body's trsm
// alpha One{} (Potri, n > NB); lauum_blocked_body without the `rest > 0` herk (Potri, n > NB);
// blocked_ws_ok without the alignment test (ArgumentsAreMinusI).
// Timing (informative, not a gate): DISABLED_TimingAgainstTheHostRoutes, n 64-512, batch 1024.
#include <tests/device_core_common.hh>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device/lapack/posv.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/lapack/potri.hh>
#include <batchlas/device/lapack/potrs.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/posv/choice.hh>
#include <src/ops/potrf/choice.hh>
#include <src/ops/potri/choice.hh>
#include <src/ops/potrs/choice.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace device_core_test;
using batchlas::verify::Check;

namespace {

constexpr int kWg = batchlas::device::potrf_detail::kBlockedWg;
constexpr int kPeriod = 3;
constexpr int kBatch = 6;
constexpr int kNrhs = 3;
template <class T> constexpr int kNb = batchlas::device::potrf_detail::reg_max_n<T> >= 32 ? 32 : 16;
template <class T> using Big = batchlas::device::tier::Blocked<kNb<T>>;  // the tier potrf's and posv's lists end with
using Small = batchlas::device::tier::Blocked<8>;

struct alignas(16) Slab {
    unsigned char b[16];
};

template <class T> T poison() {
    if constexpr (is_cplx<T>) return T(real_of<T>(1e6), real_of<T>(-1e6));
    else return T(1e6);
}
template <class T> real_of<T> scale_of(int b) { return real_of<T>(1) + real_of<T>(b % 7) * real_of<T>(0.25); }

// rows x cols per item at ld = rows + pad, stride = ld * cols + 5.
struct Shape {
    int rows, cols, ld, stride, batch;
    std::size_t elems() const { return std::size_t(stride) * batch; }
    std::size_t at(int b, int i, int j) const { return std::size_t(b) * stride + i + std::size_t(j) * ld; }
};
Shape shape_of(int rows, int cols, int batch, int pad) { return {rows, cols, rows + pad, (rows + pad) * cols + 5, batch}; }

enum class Region { Full, Lower, Upper, StrictLower, StrictUpper };
Region region_of(Uplo ul) { return ul == Uplo::Lower ? Region::Lower : Region::Upper; }
bool in_region(Region r, int i, int j) {
    switch (r) {
        case Region::Full: return true;
        case Region::Lower: return i >= j;
        case Region::Upper: return i <= j;
        case Region::StrictLower: return i > j;
        case Region::StrictUpper: return i < j;
    }
    return false;
}
// The (i, j) of element e of item b's slab, or false when e is a pad.
bool slab_index(const Shape& s, int e, int& i, int& j) {
    i = e % s.ld;
    j = e / s.ld;
    return i < s.rows && j < s.cols;
}

template <class T> std::vector<T> spd(int n, std::uint64_t seed) {
    using R = real_of<T>;
    std::vector<T> m(std::size_t(n) * n), a(std::size_t(n) * n);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = val<T>(seed * 7919u + i);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            LD<T> s{};
            for (int k = 0; k < n; ++k) {
                const LD<T> x = wide(m[i + std::size_t(k) * n]), y = wide(m[j + std::size_t(k) * n]);
                if constexpr (is_cplx<T>) s += x * std::conj(y);
                else s += x * y;
            }
            a[i + std::size_t(j) * n] = static_cast<T>(s / static_cast<long double>(n));
        }
    for (int i = 0; i < n; ++i) a[i + std::size_t(i) * n] = T(std::real(a[i + std::size_t(i) * n]) + R(2));
    return a;
}

// Item b is representative b % 3 scaled by scale_of(b) (repeat: representative 0 unscaled on
// every item). plant: representative 1 gets the pivot -1000 at n / 2, representative 2 a NaN last.
template <class T> struct Spd {
    Shape s;
    Uplo uplo;
    bool repeat;
    std::vector<T> ref;  // kPeriod full n x n Hermitian matrices, real diagonal
    std::vector<T> h;    // the stored triangle in the s layout, poison elsewhere
    T full(int b, int i, int j) const {
        const int n = s.rows;
        const T v = ref[std::size_t(repeat ? 0 : b % kPeriod) * n * n + i + std::size_t(j) * n];
        return repeat ? v : v * scale_of<T>(b);
    }
};
template <class T> Spd<T> make_spd(int n, Uplo uplo, int batch, bool plant, bool repeat = false) {
    using R = real_of<T>;
    Spd<T> p{shape_of(n, n, batch, 3), uplo, repeat, {}, {}};
    for (int r = 0; r < kPeriod; ++r) {
        auto a = spd<T>(n, 17u + 3u * r + 101u * n);
        const int k = r == 1 ? n / 2 : n - 1;
        if (plant && r == 1) a[k + std::size_t(k) * n] = T(R(-1000));
        if (plant && r == 2) a[k + std::size_t(k) * n] = T(std::numeric_limits<R>::quiet_NaN());
        p.ref.insert(p.ref.end(), a.begin(), a.end());
    }
    p.h.assign(p.s.elems(), poison<T>());
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (!in_region(region_of(uplo), i, j)) continue;
                T v = p.full(b, i, j);
                if constexpr (is_cplx<T>)
                    if (i == j) v = T(v.real(), R(0.75));  // Im(diag) is ignored by contract
                p.h[p.s.at(b, i, j)] = v;
            }
    return p;
}
int expected_info(int n, int b, bool plant) {
    if (!plant) return 0;
    const int r = b % kPeriod;
    return r == 0 ? 0 : r == 1 ? n / 2 + 1 : n;
}

// src's region of every item over a fresh poison buffer (the other triangle re-poisoned).
template <class T> std::vector<T> keep_region(const std::vector<T>& src, const Shape& s, Region r) {
    std::vector<T> out(src.size(), poison<T>());
    for (int b = 0; b < s.batch; ++b)
        for (int j = 0; j < s.cols; ++j)
            for (int i = 0; i < s.rows; ++i)
                if (in_region(r, i, j)) out[s.at(b, i, j)] = src[s.at(b, i, j)];
    return out;
}

template <class T> MatrixView<T, MatrixFormat::Dense> view(T* p, const Shape& s) {
    return MatrixView<T, MatrixFormat::Dense>(p, s.rows, s.cols, s.ld, s.stride, s.batch);
}
template <class T> KernelMatrixView<T, MatrixFormat::Dense> kview(T* p, const Shape& s) {
    return KernelMatrixView<T, MatrixFormat::Dense>(p, s.rows, s.cols, s.ld, s.stride, s.batch);
}

// The user kernel: one matrix per WorkGroup<256, 32>, the tier's local bytes from a 16-byte
// aligned local_accessor, its global bytes per item from one buffer (gbuf, else allocated here);
// f(ex, b, ws) -> info.
template <class F> void run_blocked(Dev& d, int batch, const batchlas::device::WorkspaceSize& sz, std::int32_t* info, F f, std::byte* gbuf = nullptr) {
    std::byte* const g =
        gbuf != nullptr || sz.global_bytes == 0 ? gbuf : d.alloc<std::byte>(sz.global_bytes * std::size_t(batch), std::byte{0});
    d.q().submit([&](sycl::handler& h) {
         sycl::local_accessor<Slab, 1> slm(sycl::range<1>(std::max<std::size_t>((sz.local_bytes + 15) / 16, 1)), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * kWg, kWg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const batchlas::device::WorkGroup<kWg, 32> ex{it};
             const int b = static_cast<int>(it.get_group_linear_id());
             auto* raw = reinterpret_cast<std::byte*>(batchlas::util::get_raw_ptr(slm));
             const batchlas::device::Workspace w{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(raw),
                                   g != nullptr ? batchlas::device::global_slice(g, sz, b) : nullptr};
             const int r = f(ex, b, w);
             if (ex.leader()) info[b] = r;
         });
     }).wait_and_throw();
}

// ---- the host routes, pinned -------------------------------------------------------------------
// A host-route workspace at the BumpAllocator's base alignment (MEM_BASE_ADDR_ALIGN): AdaptiveCpp's
// Metal device USM reports 512 bytes but does not honour it (malloc_device), so align by hand.
inline std::byte* host_ws(Dev& d, std::size_t bytes) {
    const std::size_t al = BumpAllocator::alignment<std::byte>(d.ctx->device());
    std::byte* const p = d.alloc<std::byte>(std::max<std::size_t>(bytes, 1) + al, std::byte{0});
    return p + (al - reinterpret_cast<std::uintptr_t>(p) % al) % al;
}
template <class T> std::vector<std::int32_t> host_potrf(Dev& d, T* a, const Shape& s, Uplo uplo) {
    const select::ScopedPin<ops::potrf::PotrfChoice> pin("potrf", ops::potrf::PotrfChoice{ops::potrf::Blocked{}});
    Queue& ctx = *d.ctx;
    const auto V = view(a, s);
    const std::size_t bytes = batchlas::potrf_buffer_size(ctx, V, uplo);
    std::byte* ws = host_ws(d, bytes);
    std::int32_t* info = d.alloc<std::int32_t>(s.batch, -7);
    (void)batchlas::potrf(ctx, V, uplo, Span<std::byte>(ws, bytes), Span<std::int32_t>(info, s.batch));
    ctx.wait();
    return d.down(info, s.batch);
}
template <class T> void host_potrs(Dev& d, T* a, const Shape& sa, T* b, const Shape& sb, Uplo uplo) {
    const select::ScopedPin<ops::potrs::PotrsChoice> pin("potrs", ops::potrs::PotrsChoice{ops::potrs::Trsm{}});
    Queue& ctx = *d.ctx;
    (void)batchlas::potrs(ctx, view(a, sa), view(b, sb), uplo);
    ctx.wait();
}
template <class T> std::vector<std::int32_t> host_posv(Dev& d, T* a, const Shape& sa, T* b, const Shape& sb, Uplo uplo) {
    const select::ScopedPin<ops::posv::PosvChoice> pin("posv", ops::posv::PosvChoice{ops::posv::Blocked{}});
    const select::ScopedPin<ops::potrf::PotrfChoice> pin_potrf("potrf", ops::potrf::PotrfChoice{ops::potrf::Blocked{}});
    Queue& ctx = *d.ctx;
    const auto A = view(a, sa);
    const auto B = view(b, sb);
    const std::size_t bytes = batchlas::posv_buffer_size(ctx, A, B, uplo);
    std::byte* ws = host_ws(d, bytes);
    std::int32_t* info = d.alloc<std::int32_t>(sa.batch, -7);
    (void)batchlas::posv(ctx, A, B, uplo, Span<std::byte>(ws, bytes), Span<std::int32_t>(info, sa.batch));
    ctx.wait();
    return d.down(info, sa.batch);
}
template <class T> std::vector<std::int32_t> host_potri(Dev& d, T* a, const Shape& s, Uplo uplo) {
    const select::ScopedPin<ops::potri::PotriChoice> pin("potri", ops::potri::PotriChoice{ops::potri::Blocked{64}});
    Queue& ctx = *d.ctx;
    const auto V = view(a, s);
    const std::size_t bytes = batchlas::potri_buffer_size(ctx, V, uplo);
    std::byte* ws = host_ws(d, bytes);
    std::int32_t* info = d.alloc<std::int32_t>(s.batch, -7);
    (void)batchlas::potri(ctx, V, uplo, Span<std::byte>(ws, bytes), Span<std::int32_t>(info, s.batch));
    ctx.wait();
    return d.down(info, s.batch);
}

// ---- comparisons -----------------------------------------------------------------------------
// max over `items` of ||x - y||_F / ||y||_F over each item's region (NaN-propagating).
template <class T> double rel_diff(const std::vector<T>& x, const std::vector<T>& y, const Shape& s, Region r, const std::vector<int>& items) {
    double worst = 0;
    for (int b : items) {
        long double num = 0, den = 0;
        for (int j = 0; j < s.cols; ++j)
            for (int i = 0; i < s.rows; ++i) {
                if (!in_region(r, i, j)) continue;
                const LD<T> dx = wide(x[s.at(b, i, j)]) - wide(y[s.at(b, i, j)]);
                num += std::norm(dx);
                den += std::norm(wide(y[s.at(b, i, j)]));
            }
        worst = batchlas::verify::nanmax(worst, static_cast<double>(std::sqrt(num / (den > 0 ? den : 1))));
    }
    return worst;
}
// Elements of `got` outside each item's region (pads included) that differ from `before`.
template <class T> int outside_changed(const std::vector<T>& got, const std::vector<T>& before, const Shape& s, Region r) {
    int bad = 0;
    for (int b = 0; b < s.batch; ++b)
        for (int e = 0; e < s.stride; ++e) {
            int i, j;
            const bool inside = slab_index(s, e, i, j) && in_region(r, i, j);
            const std::size_t at = std::size_t(b) * s.stride + e;
            if (!inside) bad += !same_bits(got[at], before[at]);
        }
    return bad;
}
// Items whose whole slab differs from `before` (an item the contract leaves unchanged).
template <class T> int items_changed(const std::vector<T>& got, const std::vector<T>& before, const Shape& s, const std::vector<int>& items) {
    int bad = 0;
    for (int b : items)
        for (int e = 0; e < s.stride; ++e) bad += !same_bits(got[std::size_t(b) * s.stride + e], before[std::size_t(b) * s.stride + e]);
    return bad;
}
template <class T> std::vector<int> spd_items(const Spd<T>& p, bool plant) {
    std::vector<int> v;
    for (int b = 0; b < p.s.batch; ++b)
        if (expected_info(p.s.rows, b, plant) == 0) v.push_back(b);
    return v;
}

// The full input matrices of `items` (n x n, ld n) and a slab copy of `got`'s items, in unified
// memory (the verify residuals read on the host).
template <class T> struct Residual {
    UnifiedVector<T> a0, x, b0;
};
template <class T> void fill_a0(const Spd<T>& p, const std::vector<int>& items, UnifiedVector<T>& a0) {
    const int n = p.s.rows;
    a0 = UnifiedVector<T>(std::size_t(n) * n * items.size());
    for (std::size_t q = 0; q < items.size(); ++q)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) a0.data()[q * n * n + i + std::size_t(j) * n] = p.full(items[q], i, j);
}
template <class T> void fill_slabs(const std::vector<T>& src, const Shape& s, const std::vector<int>& items, UnifiedVector<T>& out) {
    out = UnifiedVector<T>(std::size_t(s.stride) * items.size());
    for (std::size_t q = 0; q < items.size(); ++q)
        std::copy(src.begin() + std::size_t(items[q]) * s.stride, src.begin() + std::size_t(items[q] + 1) * s.stride, out.data() + q * s.stride);
}
template <class T> std::vector<int> iota_items(std::size_t k) {
    std::vector<int> v(k);
    for (std::size_t q = 0; q < k; ++q) v[q] = int(q);
    return v;
}

template <class T> std::string tag(const char* op, Uplo ul, int n, int nb) {
    return std::string(op) + (ul == Uplo::Lower ? " lower" : " upper") + " n=" + std::to_string(n) + " NB=" + std::to_string(nb);
}

// ---- the cases -------------------------------------------------------------------------------
template <class T, Uplo UL, class Tier> void potrf_case(Dev& d, int n) {
    const auto p = make_spd<T>(n, UL, kBatch, true);
    T* dd = d.up(p.h);
    T* dh = d.up(p.h);
    std::int32_t* info = d.alloc<std::int32_t>(kBatch, -7);
    const auto kv = kview(dd, p.s);
    run_blocked(d, kBatch, batchlas::device::potrf_buffer_size<UL, Tier, T>(n), info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potrf<UL, Tier>(ex, batchlas::device::item(kv, b), w); });
    const auto hinfo = host_potrf(d, dh, p.s, UL);
    const auto got = d.down(dd, p.h.size()), host = d.down(dh, p.h.size());
    const auto gi = d.down(info, kBatch);
    const std::string where = tag<T>("potrf", UL, n, Tier::nb);
    int infobad = 0;
    for (int b = 0; b < kBatch; ++b) infobad += gi[b] != hinfo[b] || gi[b] != expected_info(n, b, true);
    EXPECT_EQ(infobad, 0) << where << ": info differs from the host route or from the planted pivot";
    EXPECT_EQ(outside_changed(got, p.h, p.s, region_of(UL)), 0) << where << ": an element outside the stored triangle changed";
    const auto items = spd_items(p, true);
    EXPECT_VERIFY(T, Check::factorization, n, rel_diff(got, host, p.s, region_of(UL), items)) << where << ": vs the host blocked route";
    Residual<T> r;
    fill_a0(p, items, r.a0);
    fill_slabs(got, p.s, items, r.x);
    const double res = MatrixView<T, MatrixFormat::Dense>::potrf_residual(
        *d.ctx, MatrixView<T, MatrixFormat::Dense>(r.a0.data(), n, n, n, n * n, int(items.size())),
        MatrixView<T, MatrixFormat::Dense>(r.x.data(), n, n, p.s.ld, p.s.stride, int(items.size())), UL, iota_items<T>(items.size()));
    EXPECT_VERIFY(T, Check::factorization, n, res) << where << ": residual";
}

template <class T> Shape rhs_shape(int n, int batch) { return shape_of(n, kNrhs, batch, 2); }
template <class T> std::vector<T> make_rhs(const Shape& s, std::uint64_t seed) {
    std::vector<T> h(s.elems(), poison<T>());
    for (int b = 0; b < s.batch; ++b)
        for (int j = 0; j < s.cols; ++j)
            for (int i = 0; i < s.rows; ++i) h[s.at(b, i, j)] = val<T>(seed * 1000003u + s.at(b, i, j));
    return h;
}
template <class T> double solve_res(Dev& d, const Spd<T>& p, const std::vector<T>& x, const std::vector<T>& b0, const Shape& sb,
                                    const std::vector<int>& items) {
    Residual<T> r;
    fill_a0(p, items, r.a0);
    fill_slabs(x, sb, items, r.x);
    fill_slabs(b0, sb, items, r.b0);
    const int n = p.s.rows, k = int(items.size());
    return MatrixView<T, MatrixFormat::Dense>::solve_residual(
        *d.ctx, MatrixView<T, MatrixFormat::Dense>(r.a0.data(), n, n, n, n * n, k),
        MatrixView<T, MatrixFormat::Dense>(r.x.data(), n, sb.cols, sb.ld, sb.stride, k),
        MatrixView<T, MatrixFormat::Dense>(r.b0.data(), n, sb.cols, sb.ld, sb.stride, k), iota_items<T>(k));
}

template <class T, Uplo UL, class Tier> void potrs_case(Dev& d, int n) {
    const auto p = make_spd<T>(n, UL, kBatch, false);
    T* df = d.up(p.h);
    (void)host_potrf(d, df, p.s, UL);
    const auto factor = keep_region(d.down(df, p.h.size()), p.s, region_of(UL));
    const Shape sb = rhs_shape<T>(n, kBatch);
    const auto b0 = make_rhs<T>(sb, 31u + n);
    T* da = d.up(factor);
    T* db = d.up(b0);
    T* dbh = d.up(b0);
    std::int32_t* info = d.alloc<std::int32_t>(kBatch, -7);
    const auto ka = kview(da, p.s);
    const auto kb = kview(db, sb);
    run_blocked(d, kBatch, batchlas::device::potrs_buffer_size<UL, Tier, T>(n), info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::potrs<UL, Tier>(ex, batchlas::device::item(ka, b), batchlas::device::item(kb, b), w);
    });
    host_potrs(d, df, p.s, dbh, sb, UL);
    const auto x = d.down(db, b0.size()), xh = d.down(dbh, b0.size());
    const auto gi = d.down(info, kBatch);
    const std::string where = tag<T>("potrs", UL, n, Tier::nb);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch) << where << ": info";
    EXPECT_EQ(items_changed(d.down(da, factor.size()), factor, p.s, iota_items<T>(kBatch)), 0) << where << ": A changed";
    EXPECT_EQ(outside_changed(x, b0, sb, Region::Full), 0) << where << ": a pad of B changed";
    const auto items = iota_items<T>(kBatch);
    EXPECT_VERIFY(T, Check::solve, n, rel_diff(x, xh, sb, Region::Full, items)) << where << ": vs the host trsm route";
    EXPECT_VERIFY(T, Check::solve, n, solve_res(d, p, x, b0, sb, items)) << where << ": residual";
}

template <class T, Uplo UL, class Tier> void posv_case(Dev& d, int n) {
    const auto p = make_spd<T>(n, UL, kBatch, true);
    const Shape sb = rhs_shape<T>(n, kBatch);
    const auto b0 = make_rhs<T>(sb, 37u + n);
    T* da = d.up(p.h);
    T* dah = d.up(p.h);
    T* db = d.up(b0);
    T* dbh = d.up(b0);
    std::int32_t* info = d.alloc<std::int32_t>(kBatch, -7);
    const auto ka = kview(da, p.s);
    const auto kb = kview(db, sb);
    run_blocked(d, kBatch, batchlas::device::posv_buffer_size<UL, Tier, T>(n), info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::posv<UL, Tier>(ex, batchlas::device::item(ka, b), batchlas::device::item(kb, b), w);
    });
    const auto hinfo = host_posv(d, dah, p.s, dbh, sb, UL);
    const auto a = d.down(da, p.h.size()), ah = d.down(dah, p.h.size());
    const auto x = d.down(db, b0.size()), xh = d.down(dbh, b0.size());
    const auto gi = d.down(info, kBatch);
    const std::string where = tag<T>("posv", UL, n, Tier::nb);
    int infobad = 0;
    std::vector<int> failed;
    for (int b = 0; b < kBatch; ++b) {
        infobad += gi[b] != hinfo[b] || gi[b] != expected_info(n, b, true);
        if (expected_info(n, b, true) != 0) failed.push_back(b);
    }
    EXPECT_EQ(infobad, 0) << where << ": info differs from the host route or from the planted pivot";
    EXPECT_EQ(items_changed(x, b0, sb, failed), 0) << where << ": B of a failed item changed";
    EXPECT_EQ(outside_changed(a, p.h, p.s, region_of(UL)), 0) << where << ": an element outside the stored triangle changed";
    EXPECT_EQ(outside_changed(x, b0, sb, Region::Full), 0) << where << ": a pad of B changed";
    const auto items = spd_items(p, true);
    EXPECT_VERIFY(T, Check::factorization, n, rel_diff(a, ah, p.s, region_of(UL), items)) << where << ": factor vs the host route";
    EXPECT_VERIFY(T, Check::solve, n, rel_diff(x, xh, sb, Region::Full, items)) << where << ": X vs the host route";
    EXPECT_VERIFY(T, Check::solve, n, solve_res(d, p, x, b0, sb, items)) << where << ": residual";
}

// potri's input: the host factor of SPD items, the other triangle re-poisoned, representative 1
// with an exactly zero diagonal entry at n / 2 (info n / 2 + 1, the item unchanged).
template <class T, Uplo UL, class Tier> void potri_case(Dev& d, int n) {
    const auto p = make_spd<T>(n, UL, kBatch, false);
    T* df = d.up(p.h);
    (void)host_potrf(d, df, p.s, UL);
    auto factor = keep_region(d.down(df, p.h.size()), p.s, region_of(UL));
    std::vector<int> zero_items, items;
    for (int b = 0; b < kBatch; ++b) {
        if (b % kPeriod == 1) {
            factor[p.s.at(b, n / 2, n / 2)] = T(0);
            zero_items.push_back(b);
        } else {
            items.push_back(b);
        }
    }
    T* da = d.up(factor);
    T* dah = d.up(factor);
    std::int32_t* info = d.alloc<std::int32_t>(kBatch, -7);
    const auto ka = kview(da, p.s);
    run_blocked(d, kBatch, batchlas::device::potri_buffer_size<UL, Tier, T>(n), info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potri<UL, Tier>(ex, batchlas::device::item(ka, b), w); });
    const auto hinfo = host_potri(d, dah, p.s, UL);
    const auto got = d.down(da, factor.size()), host = d.down(dah, factor.size());
    const auto gi = d.down(info, kBatch);
    const std::string where = tag<T>("potri", UL, n, Tier::nb);
    int infobad = 0, imag = 0;
    for (int b = 0; b < kBatch; ++b) infobad += gi[b] != hinfo[b] || gi[b] != (b % kPeriod == 1 ? n / 2 + 1 : 0);
    if constexpr (is_cplx<T>)
        for (int b : items)
            for (int i = 0; i < n; ++i) imag += got[p.s.at(b, i, i)].imag() != real_of<T>(0);
    EXPECT_EQ(infobad, 0) << where << ": info differs from the host route or from the planted zero";
    EXPECT_EQ(imag, 0) << where << ": a diagonal element of A^-1 is not real";
    EXPECT_EQ(items_changed(got, factor, p.s, zero_items), 0) << where << ": a zero-diagonal item changed";
    EXPECT_EQ(outside_changed(got, factor, p.s, region_of(UL)), 0) << where << ": an element outside the stored triangle changed";
    EXPECT_VERIFY(T, Check::solve, n, rel_diff(got, host, p.s, region_of(UL), items)) << where << ": vs the host blocked route";
    // ||A0 X - I|| / (||A0|| ||X||) with X the stored triangle mirrored.
    std::vector<T> mirrored = got;
    for (int b : items)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (!in_region(region_of(UL), i, j)) {
                    const T v = got[p.s.at(b, j, i)];
                    if constexpr (is_cplx<T>) mirrored[p.s.at(b, i, j)] = std::conj(v);
                    else mirrored[p.s.at(b, i, j)] = v;
                }
    Residual<T> r;
    fill_a0(p, items, r.a0);
    fill_slabs(mirrored, p.s, items, r.x);
    const int k = int(items.size());
    const double res = MatrixView<T, MatrixFormat::Dense>::solve_residual(
        *d.ctx, MatrixView<T, MatrixFormat::Dense>(r.a0.data(), n, n, n, n * n, k),
        MatrixView<T, MatrixFormat::Dense>(r.x.data(), n, n, p.s.ld, p.s.stride, k), MatrixView<T, MatrixFormat::Dense>{}, iota_items<T>(k));
    EXPECT_VERIFY(T, Check::solve, n, res) << where << ": inverse residual";
}

// trtri with an implied unit diagonal (poisoned, never read) against a long-double substitution;
// the strict triangle is O(1 / 2n), so T^-1 is well conditioned.
template <class T, Uplo UL, class Tier> void trtri_unit_case(Dev& d, int n) {
    const Shape s = shape_of(n, n, kBatch, 3);
    const Region tri = region_of(UL);
    std::vector<T> h(s.elems(), poison<T>());
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (i != j && in_region(tri, i, j)) h[s.at(b, i, j)] = val<T>(911u + s.at(b, i, j)) * (real_of<T>(0.5) / real_of<T>(n));
    T* da = d.up(h);
    std::int32_t* info = d.alloc<std::int32_t>(kBatch, -7);
    const auto ka = kview(da, s);
    run_blocked(d, kBatch, batchlas::device::trtri_buffer_size<UL, Diag::Unit, Tier, T>(n), info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::trtri<UL, Diag::Unit, Tier>(ex, batchlas::device::item(ka, b), w);
    });
    const auto got = d.down(da, h.size());
    const auto gi = d.down(info, kBatch);
    // X = T^-1 column by column: T x_c = e_c (Lower forward, Upper backward), unit diagonal.
    std::vector<T> ref = h;
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < n; ++c) {
            std::vector<LD<T>> x(n, LD<T>(0));
            x[c] = 1;
            for (int t = 0; t < n; ++t) {
                const int i = UL == Uplo::Lower ? t : n - 1 - t;
                LD<T> v = i == c ? LD<T>(1) : LD<T>(0);
                for (int k = 0; k < n; ++k)
                    if (k != i && in_region(tri, i, k)) v -= wide(h[s.at(b, i, k)]) * x[k];
                x[i] = v;
            }
            for (int i = 0; i < n; ++i)
                if (i != c && in_region(tri, i, c)) ref[s.at(b, i, c)] = static_cast<T>(x[i]);
        }
    const std::string where = tag<T>("trtri unit", UL, n, Tier::nb);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch) << where << ": info";
    int diag = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) diag += !same_bits(got[s.at(b, i, i)], h[s.at(b, i, i)]);
    EXPECT_EQ(diag, 0) << where << ": the (unit) diagonal was written";
    EXPECT_EQ(outside_changed(got, h, s, tri), 0) << where << ": an element outside the triangle changed";
    const Region strict = UL == Uplo::Lower ? Region::StrictLower : Region::StrictUpper;
    EXPECT_VERIFY(T, Check::solve, n, rel_diff(got, ref, s, strict, iota_items<T>(kBatch))) << where << ": vs the substitution";
}

template <class T> class DeviceApiCholTest : public DeviceCoreTest<T> {
  protected:
    void SetUp() override {
        DeviceCoreTest<T>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (this->dev.ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the pinned host blocked routes are GPU kernels";
    }
};
TYPED_TEST_SUITE(DeviceApiCholTest, AllTypes);

template <class T> std::vector<int> big_orders() {
    constexpr int nb = kNb<T>;
    return {1, nb - 1, nb, nb + 1, 2 * nb + 1, 8 * nb + 1};
}

TYPED_TEST(DeviceApiCholTest, PotrfMatchesHostBlocked) {
    using T = TypeParam;
    for (int n : big_orders<T>()) {
        potrf_case<T, Uplo::Lower, Big<T>>(this->dev, n);
        potrf_case<T, Uplo::Upper, Big<T>>(this->dev, n);
    }
    for (int n : {9, 23, 37}) {
        potrf_case<T, Uplo::Lower, Small>(this->dev, n);
        potrf_case<T, Uplo::Upper, Small>(this->dev, n);
    }
}

TYPED_TEST(DeviceApiCholTest, PotrsMatchesHostTrsm) {
    using T = TypeParam;
    for (int n : {1, 31, 33, 97}) {
        potrs_case<T, Uplo::Lower, batchlas::device::tier::Blocked<32>>(this->dev, n);
        potrs_case<T, Uplo::Upper, batchlas::device::tier::Blocked<32>>(this->dev, n);
    }
}

TYPED_TEST(DeviceApiCholTest, PosvMatchesHostBlocked) {
    using T = TypeParam;
    for (int n : {1, kNb<T> + 1, 3 * kNb<T> + 1}) {
        posv_case<T, Uplo::Lower, Big<T>>(this->dev, n);
        posv_case<T, Uplo::Upper, Big<T>>(this->dev, n);
    }
}

TYPED_TEST(DeviceApiCholTest, PotriMatchesHostBlocked) {
    using T = TypeParam;
    for (int n : {1, 31, 32, 33, 65, 129}) {
        potri_case<T, Uplo::Lower, batchlas::device::tier::Blocked<32>>(this->dev, n);
        potri_case<T, Uplo::Upper, batchlas::device::tier::Blocked<32>>(this->dev, n);
    }
    for (int n : {9, 37}) {
        potri_case<T, Uplo::Lower, Small>(this->dev, n);
        potri_case<T, Uplo::Upper, Small>(this->dev, n);
    }
}

TYPED_TEST(DeviceApiCholTest, TrtriUnitDiagonal) {
    using T = TypeParam;
    for (int n : {1, 8, 9, 37}) {
        trtri_unit_case<T, Uplo::Lower, Small>(this->dev, n);
        trtri_unit_case<T, Uplo::Upper, Small>(this->dev, n);
    }
}

// One repeated problem on 1024 work-groups: every item bit-identical to item 0 (potrf, posv,
// potri on potrf's factor); item 0 itself is held to the host by the cases above.
TYPED_TEST(DeviceApiCholTest, SaturatingBatchIsBitIdentical) {
    using T = TypeParam;
    constexpr int kSat = 1024;
    constexpr Uplo UL = Uplo::Lower;
    const int n = 2 * kNb<T> + 6;
    const auto p = make_spd<T>(n, UL, kSat, false, true);
    const Shape sb = rhs_shape<T>(n, kSat);
    std::vector<T> b0(sb.elems(), poison<T>());
    for (int b = 0; b < kSat; ++b)
        for (int j = 0; j < kNrhs; ++j)
            for (int i = 0; i < n; ++i) b0[sb.at(b, i, j)] = val<T>(53u + i + 1000u * j);
    T* da = this->dev.up(p.h);
    T* dpos = this->dev.up(p.h);
    T* db = this->dev.up(b0);
    std::int32_t* info = this->dev.template alloc<std::int32_t>(kSat, -7);
    const auto ka = kview(da, p.s);
    const auto kp = kview(dpos, p.s);
    const auto kb = kview(db, sb);
    auto same_as_item0 = [&](const std::vector<T>& v, const Shape& s) {
        int bad = 0;
        for (int b = 1; b < s.batch; ++b)
            for (int e = 0; e < s.stride; ++e) bad += !same_bits(v[std::size_t(b) * s.stride + e], v[e]);
        return bad;
    };
    auto all_zero = [&]() {
        const auto gi = this->dev.down(info, kSat);
        return int(std::count(gi.begin(), gi.end(), 0));
    };
    run_blocked(this->dev, kSat, batchlas::device::potrf_buffer_size<UL, Big<T>, T>(n), info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potrf<UL, Big<T>>(ex, batchlas::device::item(ka, b), w); });
    EXPECT_EQ(all_zero(), kSat) << "potrf info";
    EXPECT_EQ(same_as_item0(this->dev.down(da, p.h.size()), p.s), 0) << "potrf: an item differs from item 0";
    run_blocked(this->dev, kSat, batchlas::device::posv_buffer_size<UL, Big<T>, T>(n), info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::posv<UL, Big<T>>(ex, batchlas::device::item(kp, b), batchlas::device::item(kb, b), w);
    });
    EXPECT_EQ(all_zero(), kSat) << "posv info";
    EXPECT_EQ(same_as_item0(this->dev.down(db, b0.size()), sb), 0) << "posv: an X differs from item 0's";
    run_blocked(this->dev, kSat, batchlas::device::potri_buffer_size<UL, Big<T>, T>(n), info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potri<UL, Big<T>>(ex, batchlas::device::item(ka, b), w); });
    EXPECT_EQ(all_zero(), kSat) << "potri info";
    EXPECT_EQ(same_as_item0(this->dev.down(da, p.h.size()), p.s), 0) << "potri: an item differs from item 0";
}

// -i for argument i, A and B untouched: a non-square A, B's rows, a misaligned local slice, a
// missing global slice (potri).
TYPED_TEST(DeviceApiCholTest, ArgumentsAreMinusI) {
    using T = TypeParam;
    using Tier = Small;
    constexpr Uplo UL = Uplo::Lower;
    constexpr int n = 20;
    const auto p = make_spd<T>(n, UL, 2, false);
    const Shape sb = rhs_shape<T>(n, 2);
    const auto b0 = make_rhs<T>(sb, 5u);
    T* da = this->dev.up(p.h);
    T* db = this->dev.up(b0);
    std::int32_t* info = this->dev.template alloc<std::int32_t>(2, -7);
    const auto ka = kview(da, p.s);
    const auto kb = kview(db, sb);
    auto off4 = [](const batchlas::device::Workspace& w) {
        return batchlas::device::Workspace{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(w.local.get_raw() + 4),
                             w.global};
    };
    auto expect = [&](int want, const char* what) {
        const auto gi = this->dev.down(info, 2);
        EXPECT_EQ(gi[0], want) << what;
        EXPECT_EQ(gi[1], want) << what;
    };
    const auto potrf_ws = batchlas::device::potrf_buffer_size<UL, Tier, T>(n);
    run_blocked(this->dev, 2, potrf_ws, info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potrf<UL, Tier>(ex, batchlas::device::item(ka, b, n, n - 1), w); });
    expect(-2, "potrf, A not square");
    run_blocked(this->dev, 2, potrf_ws, info,
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potrf<UL, Tier>(ex, batchlas::device::item(ka, b), off4(w)); });
    expect(-3, "potrf, misaligned ws.local");
    run_blocked(this->dev, 2, potrf_ws, info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::potrs<UL, Tier>(ex, batchlas::device::item(ka, b), batchlas::device::item(kb, b, n - 1, kNrhs), w);
    });
    expect(-3, "potrs, B rows != n");
    run_blocked(this->dev, 2, potrf_ws, info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::potrs<UL, Tier>(ex, batchlas::device::item(ka, b), batchlas::device::item(kb, b), off4(w));
    });
    expect(-4, "potrs, misaligned ws.local");
    run_blocked(this->dev, 2, potrf_ws, info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) {
        return batchlas::device::posv<UL, Tier>(ex, batchlas::device::item(ka, b), batchlas::device::item(kb, b, n, 0), off4(w));
    });
    expect(-4, "posv nrhs = 0, misaligned ws.local");
    run_blocked(this->dev, 2, potrf_ws, info,  // no global bytes in potrf's plan: w.global is null
                [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potri<UL, Tier>(ex, batchlas::device::item(ka, b), w); });
    expect(-3, "potri, no ws.global");
    EXPECT_EQ(items_changed(this->dev.down(da, p.h.size()), p.h, p.s, {0, 1}), 0) << "A changed";
    EXPECT_EQ(items_changed(this->dev.down(db, b0.size()), b0, sb, {0, 1}), 0) << "B changed";
}

template <class Routine, class T> std::string pick(const batchlas::device::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        batchlas::device::with_tier<Routine, T>(lim, n, [&]<class Tier>(Tier) { got = "blocked" + std::to_string(Tier::nb); });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string(e.what()).find(std::string(Routine::name) + ": no device tier admits n=" + std::to_string(n)) != std::string::npos
                  ? "throw"
                  : "bad message";
    }
    return got;
}

TYPED_TEST(DeviceApiCholTest, WithTierFilters) {
    using T = TypeParam;
    using Potrs = batchlas::device::routine::potrs<Uplo::Upper>;
    using Posv = batchlas::device::routine::posv<Uplo::Lower>;
    using Potri = batchlas::device::routine::potri<Uplo::Lower>;
    const std::string nb = "blocked" + std::to_string(kNb<T>);
    const batchlas::device::DeviceLimits big{std::size_t(1) << 20, {16, 32}, 1024, true};
    for (int n : {0, 1, 4000}) {
        EXPECT_EQ((pick<Potrs, T>(big, n)), "blocked32") << n;
        EXPECT_EQ((pick<Posv, T>(big, n)), nb) << n;
        EXPECT_EQ((pick<Potri, T>(big, n)), "blocked32") << n;
    }
    const std::size_t potrs_local = batchlas::device::potrs_buffer_size<Uplo::Upper, batchlas::device::tier::Blocked<32>, T>(1).local_bytes;
    const batchlas::device::DeviceLimits starved{potrs_local - 1, {32}, 1024, true};
    EXPECT_EQ((pick<Potrs, T>(starved, 64)), "throw");
    const batchlas::device::DeviceLimits sg16{std::size_t(1) << 20, {16}, 1024, true};
    EXPECT_EQ((pick<Posv, T>(sg16, 64)), "throw");
    const batchlas::device::DeviceLimits narrow{std::size_t(1) << 20, {32}, 128, true};
    EXPECT_EQ((pick<Potri, T>(narrow, 64)), "throw");
    const batchlas::device::DeviceLimits no_fp64{std::size_t(1) << 20, {32}, 1024, false};
    EXPECT_EQ((pick<Potri, T>(no_fp64, 64)), (std::is_same_v<real_of<T>, double> ? "throw" : "blocked32"));
    static_assert(batchlas::device::tier_info<Posv, batchlas::device::tier::Blocked<32>, T>::admits(64) == (kNb<T> == 32));
}

// The plan the ops carve is the size the host allocates: slots inside it, aligned, disjoint;
// posv's potrs carves potrf's first slot; potri's global panel is n x NB.
TYPED_TEST(DeviceApiCholTest, BufferSizeIsTheCarvedPlan) {
    using T = TypeParam;
    using D = batchlas::device::potrf_detail::dev_scalar_t<T>;
    using Cfg = batchlas::device::potrf_detail::BlockedCfg<T>;
    constexpr auto pl = batchlas::device::potrf_detail::BlockedPlan<T, 16>::make();
    const auto sz = batchlas::device::potrf_buffer_size<Uplo::Lower, batchlas::device::tier::Blocked<16>, T>(1000);
    EXPECT_EQ(sz, pl.size);
    EXPECT_EQ(pl.smem.offset, 0u);
    EXPECT_EQ(pl.smem.count, std::size_t(Cfg::smem_elems));
    EXPECT_EQ(pl.blk.count, 16u * 16u);
    EXPECT_LE(pl.smem.count * sizeof(T), pl.blk.offset);
    EXPECT_LE(pl.blk.offset + pl.blk.count * sizeof(D), pl.diag.offset);
    EXPECT_LE(pl.diag.offset + pl.diag.count * sizeof(real_of<T>), pl.fail.offset);
    EXPECT_LE(pl.fail.offset + sizeof(int), sz.local_bytes);
    EXPECT_EQ(pl.blk.offset % 16 + pl.diag.offset % 16 + pl.fail.offset % 16, 0u);
    EXPECT_EQ(sz.global_bytes, 0u);
    EXPECT_EQ((batchlas::device::posv_buffer_size<Uplo::Upper, batchlas::device::tier::Blocked<16>, T>(7)), sz);
    const auto rs = batchlas::device::potrs_buffer_size<Uplo::Upper, batchlas::device::tier::Blocked<16>, T>(7);
    EXPECT_EQ(rs.local_bytes, (batchlas::device::l3::smem_bytes<Cfg>() + 15) / 16 * 16);
    EXPECT_EQ(rs.global_bytes, 0u);
    for (int n : {1, 33, 1000}) {
        const auto ps = batchlas::device::potri_buffer_size<Uplo::Lower, batchlas::device::tier::Blocked<32>, T>(n);
        EXPECT_EQ(ps.local_bytes, rs.local_bytes) << n;
        EXPECT_EQ(ps.global_bytes, (std::size_t(n) * 32 * sizeof(T) + 15) / 16 * 16) << n;
        EXPECT_EQ((batchlas::device::trtri_buffer_size<Uplo::Upper, Diag::Unit, batchlas::device::tier::Blocked<32>, T>(n)), ps) << n;
        EXPECT_EQ((batchlas::device::lauum_buffer_size<Uplo::Upper, batchlas::device::tier::Blocked<32>, T>(n)), ps) << n;
    }
}

// Informative only (run with --gtest_also_run_disabled_tests): the device Blocked tier in one
// kernel against the pinned multi-kernel host routes, n 64-512, batch 1024, median of 3 after a
// warm-up. Prints one line per (op, n); asserts nothing beyond the runs succeeding.
template <class F> double median_ms(Queue& ctx, F&& f) {
    f();
    ctx.wait();
    std::vector<double> t;
    for (int r = 0; r < 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        ctx.wait();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[1];
}
TYPED_TEST(DeviceApiCholTest, DISABLED_TimingAgainstTheHostRoutes) {
    using T = TypeParam;
    constexpr int kTimeBatch = 1024;
    constexpr Uplo UL = Uplo::Lower;
    Dev& d = this->dev;
    Queue& ctx = *d.ctx;
    for (int n : {64, 128, 256, 512}) {
        const auto p = make_spd<T>(n, UL, kTimeBatch, false, true);
        T* da = d.up(p.h);
        T* dfac = d.up(p.h);
        std::int32_t* info = d.alloc<std::int32_t>(kTimeBatch, -7);
        const auto ka = kview(da, p.s);
        const auto V = view(da, p.s);
        const std::size_t bytes = p.h.size() * sizeof(T);
        auto reset = [&](const T* from) { d.q().memcpy(da, from, bytes).wait(); };
        const auto potrf_ws = batchlas::device::potrf_buffer_size<UL, Big<T>, T>(n);
        const auto potri_ws = batchlas::device::potri_buffer_size<UL, Big<T>, T>(n);
        std::byte* gbuf = d.alloc<std::byte>(std::max<std::size_t>(potri_ws.global_bytes * kTimeBatch, 1), std::byte{0});
        T* const h0 = d.up(p.h);
        const double dev_potrf = median_ms(ctx, [&]() {
            reset(h0);
            run_blocked(d, kTimeBatch, potrf_ws, info,
                        [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potrf<UL, Big<T>>(ex, batchlas::device::item(ka, b), w); });
        });
        d.q().memcpy(dfac, da, bytes).wait();
        double host_potrf_ms = 0, host_potri_ms = 0;
        {
            const select::ScopedPin<ops::potrf::PotrfChoice> pin("potrf", ops::potrf::PotrfChoice{ops::potrf::Blocked{}});
            const std::size_t wsb = batchlas::potrf_buffer_size(ctx, V, UL);
            std::byte* ws = host_ws(d, wsb);
            host_potrf_ms = median_ms(ctx, [&]() {
                reset(h0);
                (void)batchlas::potrf(ctx, V, UL, Span<std::byte>(ws, wsb), Span<std::int32_t>(info, kTimeBatch));
            });
        }
        const double dev_potri = median_ms(ctx, [&]() {
            reset(dfac);
            run_blocked(
                d, kTimeBatch, potri_ws, info, [=](const auto& ex, int b, const batchlas::device::Workspace& w) { return batchlas::device::potri<UL, Big<T>>(ex, batchlas::device::item(ka, b), w); },
                gbuf);
        });
        {
            const select::ScopedPin<ops::potri::PotriChoice> pin("potri", ops::potri::PotriChoice{ops::potri::Blocked{64}});
            const std::size_t wsb = batchlas::potri_buffer_size(ctx, V, UL);
            std::byte* ws = host_ws(d, wsb);
            host_potri_ms = median_ms(ctx, [&]() {
                reset(dfac);
                (void)batchlas::potri(ctx, V, UL, Span<std::byte>(ws, wsb), Span<std::int32_t>(info, kTimeBatch));
            });
        }
        std::printf("chol-blocked %s n=%d batch=%d: potrf device %.3f ms host %.3f ms (%.2fx); potri device %.3f ms host %.3f ms (%.2fx)"
                    " [one input reset copy included in every timing]\n",
                    batchlas::verify::detail::dtype_name<T>(), n, kTimeBatch, dev_potrf, host_potrf_ms, dev_potrf / host_potrf_ms, dev_potri,
                    host_potri_ms, dev_potri / host_potri_ms);
        const auto gi = d.down(info, kTimeBatch);
        EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kTimeBatch) << "n=" << n;
    }
}

}  // namespace
