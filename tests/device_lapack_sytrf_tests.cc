// SPDX-License-Identifier: MIT
// device_lapack_sytrf_tests: the public in-kernel sytrf / sytrs / sysv (batchlas/device/lapack/
// {sytrf,sytrs,sysv}.hh, v22 design C) called from user-style kernels, against the host route
// pinned to the same body: sytrf Cta = `cta`, Blocked<1> = `wg`; sytrs Cta<32/64/128> = `cta`,
// Blocked<1> = `wg`; sysv Cta = `cta`, Blocked<1> = `compose` over sytrf `wg` and sytrs `wg`.
// Guards (agent-guide section 8): every comparison is a memcmp over the whole padded storage
// (ld n + 3 and n + 2, item gaps, the unreferenced triangle poisoned), the int32 pivots and info,
// at batch 1024 with singular, NaN, 2 x 2-forcing and dominant items mixed in (each kind asserted
// to occur); complex data is symmetric with Im(diag) != 0; each host route is pinned and its
// `reached` row checked; n straddles every tier boundary (1, 31, 32 | 33, 64 | 65, 128 | 129) and
// with_tier's local-memory filter is straddled by one byte; every item's return value is compared
// (info uniform across the executor), on whole-sub-group and multi-sub-group executors and an
// RtGroup; a canary past buffer_size(n) in local memory must survive every tier, and a deliberately
// shifted workspace must break it (the guard's red case, run every time); argument errors return
// the same code on every item and leave A untouched.
// Blocked<31> (the in-group blocked tiers, wave 2) runs another summation order than the host
// `blocked` routes, so it is compared within tolerance: every item's info (uniform), the ipiv and
// info that pairs / singular items force by exact zeros (an oracle, also held against the host),
// and each solve's residual against the full matrix; n straddles every panel end and a 2 x 2 block
// crosses the first; a 1024-item batch of one problem is bit-identical to item 0; a global canary
// after each matrix's slice must survive, and a slice shifted into it must break it (red case).
// Red set: not yet recorded (written without a build, waves 1 and 2); the gate round makes each
// break once (Blocked<31>: the trailing update's rank e - k0 forced to NB; lasyf_undo dropped).
#include <gtest/gtest.h>

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/sysv.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/test_support/coverage_rows.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <src/ops/sysv/choice.hh>
#include <src/ops/sytrf/choice.hh>
#include <src/ops/sytrf/sytrf_native.hh>
#include <src/ops/sytrs/choice.hh>
#include <src/queue.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

constexpr int kBatch = 1024;
constexpr std::size_t kCanary = 64;
constexpr int32_t kPivSentinel = 0x0BADBEEF;

// ---- the plans, pinned (each tier's buffer_size is what its body carves) ----------------------
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sytrf<Uplo::Lower>, float, device::tier::upto<32>>, device::tier::Cta<32>>);
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sytrf<Uplo::Lower>, float, device::tier::upto<33>>, device::tier::Blocked<1>>);
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sytrs<Uplo::Upper>, double, device::tier::upto<50>>, device::tier::Cta<64>>);
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sytrs<Uplo::Upper>, double, device::tier::upto<129>>, device::tier::Blocked<1>>);
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sysv<Uplo::Lower>, float, device::tier::upto<20>>, device::tier::Cta<32>>);
static_assert(device::sytrf_buffer_size<Uplo::Lower, device::tier::Cta<32>, float>(7).local_bytes == 33 * 32 * 4);
static_assert(device::sytrf_buffer_size<Uplo::Lower, device::tier::Cta<32>, float>(7).global_bytes == 0);
static_assert(device::sytrf_buffer_size<Uplo::Upper, device::tier::Blocked<1>, double>(1000).local_bytes == 8 * 8 + 8 * 4);
static_assert(device::sytrs_buffer_size<Uplo::Lower, device::tier::Cta<32>, float>(5).local_bytes == 128 + 64);  // 30 floats, 16 ints
static_assert(device::sytrs_buffer_size<Uplo::Lower, device::tier::Blocked<1>, std::complex<float>>(9).local_bytes == 16 * 4);
static_assert(device::sysv_buffer_size<Uplo::Upper, device::tier::Cta<32>, double>(3).local_bytes == 9232);  // 1088 doubles, 130 ints
static_assert(device::sysv_buffer_size<Uplo::Lower, device::tier::Blocked<1>, float>(3).local_bytes == 64);
// Blocked<31>: local = the arg-max scratch (8 R, 8 int) then the L3 tile (552 T); global = W.
static_assert(device::sytrf_buffer_size<Uplo::Lower, device::tier::Blocked<31>, float>(100) == device::WorkspaceSize{2272, 100 * 32 * 4, 16});
static_assert(device::sytrf_buffer_size<Uplo::Upper, device::tier::Blocked<31>, std::complex<double>>(10) == device::WorkspaceSize{8928, 5120, 16});
// sytrs: local = flag, the L3 tile; global = L (n n), y (n nrhs), five int vectors (n each).
static_assert(device::sytrs_buffer_size<Uplo::Lower, device::tier::Blocked<31>, float>(10, 3) == device::WorkspaceSize{2224, 736, 16});
static_assert(device::sytrs_buffer_size<Uplo::Lower, device::tier::Blocked<31>, float>(1000, 1).local_bytes == 2224);
static_assert(device::sysv_buffer_size<Uplo::Upper, device::tier::Blocked<31>, float>(10, 3) == device::WorkspaceSize{2272, 1280, 16});
// The default lists never hold Blocked<NB> (no measured win; design F6): upto stays on Blocked<1>.
static_assert(std::is_same_v<device::resolve_tier_t<device::routine::sysv<Uplo::Upper>, float, device::tier::upto<4096>>, device::tier::Blocked<1>>);

// ---- data ---------------------------------------------------------------------------------------
template <typename T>
T draw(std::mt19937& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double re = u(g), im = u(g);
    return MV<T>::make(re, im);
}

// Item b of the full symmetric batch (complex: symmetric, not Hermitian). b % 16 picks its kind:
// 5 a zero row and column (singular: info > 0), 9 a NaN off the diagonal, 12 a dominant diagonal
// (1 x 1 pivots, no interchange), 14 a zero diagonal (2 x 2 pivots); the rest uniform in [-1, 1].
template <typename T>
Matrix<T> symmetric_batch(int n, unsigned seed) {
    Matrix<T> full(n, n, kBatch);
    std::mt19937 g(seed);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int b = 0; b < kBatch; ++b) {
        const int kind = b % 16;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                T v = draw<T>(g);
                if (kind == 5 && (i == n / 2 || j == n / 2)) v = T{};
                if (kind == 12 && i == j) v = MV<T>::make((j % 2 ? -1.0 : 1.0) * (n + 2), 0.5);
                if (kind == 14 && i == j) v = T{};
                full(i, j, b) = full(j, i, b) = v;
            }
        if (kind == 9 && n >= 2) full(1, 0, b) = full(0, 1, b) = MV<T>::make(nan, nan);
    }
    return full;
}

// The uplo triangle of `full` in padded storage (ld n + 3, item gap 7), poison elsewhere.
template <typename T>
Matrix<T> stored(const Matrix<T>& full, Uplo uplo) {
    Matrix<T> a = ts::padded<T>(full.rows(), full.cols(), full.batch_size(), 3, 7);
    ts::store<T>(a, full, Live::triangle(uplo));
    return a;
}

template <typename T>
Matrix<T> rhs(int n, int nrhs, unsigned seed, int batch = kBatch) {
    Matrix<T> src(n, nrhs, batch);
    std::mt19937 g(seed);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) src(i, c, b) = draw<T>(g);
    Matrix<T> m = ts::padded<T>(n, nrhs, batch, 2, 5);
    ts::store<T>(m, src);
    return m;
}

// Blocked<NB> data, item b's kind from b % 8. pairs: logical 2 x 2 pairs [[0, c], [c, 0]] at
// (2i, 2i + 1), nothing else (odd n: the last diagonal n + 2), so every even logical k takes a
// 2 x 2 block without interchange, one of them at 30 (across the first panel's end for NB = 31);
// singular: a dominant leading logical m x m block, the rest exactly zero, so no step interchanges
// and logical column m is the first zero one (m from singular_m: inside a panel, at its last
// column, past it); nan: random with a NaN at logical (1, 0); random: uniform in [-1, 1]. pairs and
// singular decide every pivot by an exact zero, so their ipiv and info hold whatever the rounding.
enum class BKind { random, pairs, singular, nan };
inline BKind kind_of(int b) {
    const int k = b % 8;
    return k == 3 ? BKind::pairs : k == 5 ? BKind::singular : k == 7 ? BKind::nan : BKind::random;
}
inline int singular_m(int b, int n) {
    const int choice[4] = {n / 3, 30, 31, 62};
    return std::min(choice[(b / 8) % 4], n - 1);
}
inline int phys_of(Uplo uplo, int n, int i) { return uplo == Uplo::Upper ? n - 1 - i : i; }

template <typename T>
Matrix<T> blocked_batch(int n, int batch, Uplo uplo, unsigned seed) {
    Matrix<T> full(n, n, batch);
    std::mt19937 g(seed);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int b = 0; b < batch; ++b) {
        const BKind kd = kind_of(b);
        const int m = singular_m(b, n);
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {  // logical lower triangle
                T v = draw<T>(g);
                if (kd == BKind::pairs) {
                    const bool pair = i == j + 1 && j % 2 == 0, last = i == j && j == n - 1 && n % 2 == 1;
                    v = pair ? MV<T>::make(n + 2.0 + j, 0.5) : last ? MV<T>::make(n + 2.0, 0.5) : T{};
                }
                if (kd == BKind::singular) {
                    if (i >= m || j >= m) v = T{};
                    else if (i == j) v = MV<T>::make((j % 2 ? -1.0 : 1.0) * (n + 2), 0.5);
                }
                const int pi = phys_of(uplo, n, i), pj = phys_of(uplo, n, j);
                full(pi, pj, b) = full(pj, pi, b) = v;
            }
        if (kd == BKind::nan && n >= 2) {
            const int p1 = phys_of(uplo, n, 1), p0 = phys_of(uplo, n, 0);
            full(p1, p0, b) = full(p0, p1, b) = MV<T>::make(nan, nan);
        }
    }
    return full;
}

// The ipiv a pairs or singular item forces, in LAPACK's physical order.
inline std::vector<int32_t> forced_ipiv(BKind kd, int n, Uplo uplo) {
    std::vector<int32_t> p(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) p[phys_of(uplo, n, k)] = phys_of(uplo, n, k) + 1;
    if (kd == BKind::pairs)
        for (int j = 0; j + 1 < n; j += 2) p[phys_of(uplo, n, j)] = p[phys_of(uplo, n, j + 1)] = -(phys_of(uplo, n, j + 1) + 1);
    return p;
}

// ---- comparisons --------------------------------------------------------------------------------
template <typename T>
bool has_nan(const T& v) {
    if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
    else return std::isnan(v.real()) || std::isnan(v.imag());
}

template <typename T>
::testing::AssertionResult same_storage(const Matrix<T>& x, const Matrix<T>& y, const char* what) {
    if (x.stride() != y.stride() || x.batch_size() != y.batch_size()) return ::testing::AssertionFailure() << what << ": layouts differ";
    const std::size_t count = static_cast<std::size_t>(x.stride()) * x.batch_size();
    for (std::size_t e = 0; e < count; ++e)
        if (std::memcmp(x.data_ptr() + e, y.data_ptr() + e, sizeof(T)) != 0)
            return ::testing::AssertionFailure() << what << " differs at element " << e << " (item " << e / x.stride() << ")";
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult same_ints(const int32_t* x, const int32_t* y, std::size_t count, const char* what) {
    for (std::size_t e = 0; e < count; ++e)
        if (x[e] != y[e]) return ::testing::AssertionFailure() << what << " differs at " << e << ": " << x[e] << " vs " << y[e];
    return ::testing::AssertionSuccess();
}

// Every item of work-group b returned the same value, equal to want[b] (want null: 0).
::testing::AssertionResult uniform_returns(const UnifiedVector<int32_t>& r, int wg, const int32_t* want) {
    const int batch = static_cast<int>(r.size() / std::size_t(wg));
    for (int b = 0; b < batch; ++b) {
        const int32_t w = want ? want[b] : 0;
        for (int l = 0; l < wg; ++l)
            if (r.data()[std::size_t(b) * wg + l] != w)
                return ::testing::AssertionFailure() << "item " << b << " lane " << l << " returned " << r.data()[std::size_t(b) * wg + l]
                                                     << ", expected " << w;
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult canary_intact(const UnifiedVector<unsigned char>& c) {
    const int batch = static_cast<int>(c.size() / kCanary);
    for (int b = 0; b < batch; ++b)
        for (std::size_t k = 0; k < kCanary; ++k)
            if (c.data()[std::size_t(b) * kCanary + k] != static_cast<unsigned char>(0xA5 ^ k))
                return ::testing::AssertionFailure() << "local byte buffer_size + " << k << " of item " << b << " was written";
    return ::testing::AssertionSuccess();
}

// The run left a `reached,<op>,...,<spelling>` coverage row (linked is not reached).
template <class Run>
::testing::AssertionResult reached(const std::vector<std::pair<std::string, std::string>>& want, Run&& run) {
    const auto rows = ts::reached_rows(std::forward<Run>(run));
    for (const auto& [op, spelling] : want) {
        bool hit = false;
        for (const auto& r : rows) hit = hit || (r.op == op && r.spelling == spelling);
        if (!hit) return ::testing::AssertionFailure() << "no reached row " << op << " / " << spelling;
    }
    return ::testing::AssertionSuccess();
}

// ---- the user kernels ---------------------------------------------------------------------------
// `batch` work-groups of wg items at sub-group size 32 with `local` bytes of local memory.
template <class F>
void launch(sycl::queue& q, int wg, std::size_t local, F f, int batch = kBatch) {
    q.submit([&](sycl::handler& h) {
         sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local, 16)), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, std::size_t(wg)), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             f(it, batchlas::util::get_raw_ptr(lm));
         });
     }).wait_and_throw();
}

inline device::Workspace local_ws(unsigned char* lm, std::size_t offset, std::byte* global = nullptr) {
    return {sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                reinterpret_cast<std::byte*>(lm + offset)),
            global};
}

// A caller's global workspace: `bytes` (buffer_size's global part) per matrix, each slice followed
// by kCanary canary bytes (a slice stride the caller may choose: it stays 16-byte aligned). `shift`
// moves every matrix's slice that far into its canary (the guard's red case).
struct Gws {
    std::byte* base = nullptr;
    std::size_t stride = 0;
    std::size_t shift = 0;
    std::byte* at(std::size_t b) const { return base == nullptr ? nullptr : base + b * stride + shift; }
};
struct GlobalWs {
    std::size_t bytes, stride;
    int batch;
    UnifiedVector<std::byte> mem;
    GlobalWs(std::size_t bytes_, int batch_)
        : bytes(bytes_), stride(bytes_ + kCanary), batch(batch_), mem(std::max<std::size_t>(stride * std::size_t(batch_), 1)) {
        for (int b = 0; b < batch; ++b)
            for (std::size_t k = 0; k < kCanary; ++k) mem.data()[b * stride + bytes + k] = static_cast<std::byte>(0x5A ^ k);
    }
    Gws view(std::size_t shift = 0) { return {mem.data(), stride, shift}; }
    ::testing::AssertionResult intact() const {
        for (int b = 0; b < batch; ++b)
            for (std::size_t k = 0; k < kCanary; ++k)
                if (mem.data()[b * stride + bytes + k] != static_cast<std::byte>(0x5A ^ k))
                    return ::testing::AssertionFailure() << "global byte buffer_size + " << k << " of item " << b << " was written";
        return ::testing::AssertionSuccess();
    }
};

// The canary: kCanary bytes right after the op's buffer_size, set before and read back after.
inline void canary_set(sycl::nd_item<1> it, unsigned char* lm, std::size_t need) {
    for (std::size_t k = it.get_local_linear_id(); k < kCanary; k += it.get_local_range(0)) lm[need + k] = static_cast<unsigned char>(0xA5 ^ k);
    sycl::group_barrier(it.get_group());
}
inline void canary_get(sycl::nd_item<1> it, const unsigned char* lm, std::size_t need, unsigned char* out) {
    sycl::group_barrier(it.get_group());
    const std::size_t b = it.get_group_linear_id();
    for (std::size_t k = it.get_local_linear_id(); k < kCanary; k += it.get_local_range(0)) out[b * kCanary + k] = lm[need + k];
}

// The executor a user kernel builds: WorkGroup<WG, 32> for WG > 0, an RtGroup for WG == 0.
template <int WG, class F>
void with_exec(sycl::nd_item<1> it, F&& f) {
    if constexpr (WG == 0) f(device::RtGroup<>(it));
    else f(device::WorkGroup<WG, 32>{it});
}

struct Outs {
    int32_t* ret;           // every item's return value, [b * wg + lid]
    unsigned char* canary;  // kCanary bytes per work-group
};

template <Uplo UL, class Tier, int WG, typename T>
void dev_sytrf(sycl::queue& q, const Matrix<T>& a, int32_t* ipiv, Outs o, int wg, std::size_t ws_offset = 0, Gws gw = {}) {
    const int n = a.rows();
    const std::size_t need = device::sytrf_buffer_size<UL, Tier, T>(n).local_bytes;
    const auto av = a.kernel_view();
    launch(
        q, wg, need + kCanary,
        [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            canary_set(it, lm, need);
            with_exec<WG>(it, [&](const auto& ex) {
                o.ret[std::size_t(b) * wg + it.get_local_linear_id()] =
                    device::sytrf<UL, Tier>(ex, device::item(av, b, n, n), ipiv + std::size_t(b) * n, local_ws(lm, ws_offset, gw.at(b)));
            });
            canary_get(it, lm, need, o.canary);
        },
        a.batch_size());
}

template <Uplo UL, class Tier, int WG, typename T>
void dev_sytrs(sycl::queue& q, const Matrix<T>& a, const Matrix<T>& bm, const int32_t* ipiv, Outs o, int wg, Gws gw = {}) {
    const int n = a.rows(), nrhs = bm.cols();
    const std::size_t need = device::sytrs_buffer_size<UL, Tier, T>(n, nrhs).local_bytes;
    const auto av = a.kernel_view();
    const auto bv = bm.kernel_view();
    launch(
        q, wg, need + kCanary,
        [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            canary_set(it, lm, need);
            with_exec<WG>(it, [&](const auto& ex) {
                o.ret[std::size_t(b) * wg + it.get_local_linear_id()] = device::sytrs<UL, Tier>(
                    ex, device::item(av, b, n, n), device::item(bv, b, n, nrhs), ipiv + std::size_t(b) * n, local_ws(lm, 0, gw.at(b)));
            });
            canary_get(it, lm, need, o.canary);
        },
        a.batch_size());
}

template <Uplo UL, class Tier, int WG, typename T>
void dev_sysv(sycl::queue& q, const Matrix<T>& a, const Matrix<T>& bm, int32_t* ipiv, Outs o, int wg, Gws gw = {}) {
    const int n = a.rows(), nrhs = bm.cols();
    const std::size_t need = device::sysv_buffer_size<UL, Tier, T>(n, nrhs).local_bytes;
    const auto av = a.kernel_view();
    const auto bv = bm.kernel_view();
    launch(
        q, wg, need + kCanary,
        [=](sycl::nd_item<1> it, unsigned char* lm) {
            const int b = static_cast<int>(it.get_group_linear_id());
            canary_set(it, lm, need);
            with_exec<WG>(it, [&](const auto& ex) {
                o.ret[std::size_t(b) * wg + it.get_local_linear_id()] = device::sysv<UL, Tier>(
                    ex, device::item(av, b, n, n), device::item(bv, b, n, nrhs), ipiv + std::size_t(b) * n, local_ws(lm, 0, gw.at(b)));
            });
            canary_get(it, lm, need, o.canary);
        },
        a.batch_size());
}

// The host routes' work-group width for `wg` / Blocked<1> parity (sytrf_kernels.cc group_width).
int host_width(int n) { return std::max(32, std::min(256, (n + 31) / 32 * 32)); }

template <typename Config>
class DeviceApiSytrfTest : public test_utils::BatchLASTest<Config> {
  protected:
    using T = typename Config::ScalarType;
    static constexpr Backend Bk = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (::testing::Test::IsSkipped()) return;
        if constexpr (Bk == Backend::NETLIB) GTEST_SKIP() << "the device API's host twins are the GPU routes";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "the device API runs at sub-group size 32";
        if (test_utils::lacks_fp64<T>(sycl_queue(*this->ctx).get_device())) GTEST_SKIP() << "device has no fp64";
    }

    sycl::queue& q() { return sycl_queue(*this->ctx); }

    void host_sytrf(Matrix<T>& a, Uplo uplo, UnifiedVector<int64_t>& piv, UnifiedVector<int32_t>& info) {
        const size_t need = sytrf_buffer_size<Bk, T>(*this->ctx, a, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sytrf<Bk, T>(*this->ctx, a, uplo, piv.to_span(), Span<std::byte>(ws.data(), need), info.to_span());
        this->ctx->wait();
    }
    void host_sytrs(const Matrix<T>& a, Matrix<T>& bm, Uplo uplo, UnifiedVector<int64_t>& piv) {
        const size_t need = sytrs_buffer_size<Bk, T>(*this->ctx, a, bm, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sytrs<Bk, T>(*this->ctx, a, bm, uplo, piv.to_span(), Span<std::byte>(ws.data(), need));
        this->ctx->wait();
    }
    void host_sysv(Matrix<T>& a, Matrix<T>& bm, Uplo uplo, UnifiedVector<int64_t>& piv, UnifiedVector<int32_t>& info) {
        const size_t need = sysv_buffer_size<Bk, T>(*this->ctx, a, bm, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sysv<Bk, T>(*this->ctx, a, bm, uplo, piv.to_span(), Span<std::byte>(ws.data(), need), info.to_span());
        this->ctx->wait();
    }

    // The batch's mix reached every branch: a singular item, and (n >= 2) a 2 x 2 block.
    static void expect_mix(const UnifiedVector<int32_t>& info, const int32_t* ipiv, int n) {
        bool singular = false, block = false;
        for (int b = 0; b < kBatch; ++b) singular = singular || info.data()[b] > 0;
        for (std::size_t e = 0; e < std::size_t(n) * kBatch; ++e) block = block || ipiv[e] < 0;
        EXPECT_TRUE(singular) << "no singular item: the info branch went untested";
        if (n >= 2) EXPECT_TRUE(block) << "no 2 x 2 block: the kstep 2 branch went untested";
    }

    // Host sytrf on `route` vs the device call on the tier, bit for bit.
    template <Uplo UL, class Tier, int WG>
    void sytrf_case(int n, const ops::sytrf::SytrfChoice& route, const char* spelling, int wg = WG) {
        const Matrix<T> full = symmetric_batch<T>(n, 1000u + unsigned(n));
        Matrix<T> ah = stored(full, UL), ad = ah.clone();
        UnifiedVector<int64_t> hp(std::size_t(n) * kBatch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(kBatch, -12345);
        ASSERT_TRUE(reached({{"sytrf", spelling}}, [&] {
            const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", route);
            host_sytrf(ah, UL, hp, hi);
        }));
        UnifiedVector<int32_t> dp(std::size_t(n) * kBatch, kPivSentinel), ret(std::size_t(kBatch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(kBatch) * kCanary, 0);
        dev_sytrf<UL, Tier, WG>(q(), ad, dp.data(), Outs{ret.data(), can.data()}, wg);
        const int32_t* hpi = reinterpret_cast<const int32_t*>(hp.data());
        EXPECT_TRUE(same_storage(ah, ad, "A")) << "n " << n;
        EXPECT_TRUE(same_ints(hpi, dp.data(), std::size_t(n) * kBatch, "ipiv")) << "n " << n;
        EXPECT_TRUE(uniform_returns(ret, wg, hi.data())) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
        expect_mix(hi, hpi, n);
    }

    // Host sytrs on `route` vs the device call, both on one host `wg` factor, bit for bit.
    template <Uplo UL, class Tier, int WG>
    void sytrs_case(int n, int nrhs, const ops::sytrs::SytrsChoice& route, const char* spelling, int wg = WG) {
        const Matrix<T> full = symmetric_batch<T>(n, 2000u + unsigned(n));
        Matrix<T> a = stored(full, UL);
        UnifiedVector<int64_t> hp(std::size_t(n) * kBatch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(kBatch, -12345);
        {
            const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
            host_sytrf(a, UL, hp, hi);
        }
        const Matrix<T> a0 = a.clone();
        Matrix<T> bh = rhs<T>(n, nrhs, 3000u + unsigned(n)), bd = bh.clone();
        ASSERT_TRUE(reached({{"sytrs", spelling}}, [&] {
            const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", route);
            host_sytrs(a, bh, UL, hp);
        }));
        UnifiedVector<int32_t> ret(std::size_t(kBatch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(kBatch) * kCanary, 0);
        dev_sytrs<UL, Tier, WG>(q(), a, bd, reinterpret_cast<const int32_t*>(hp.data()), Outs{ret.data(), can.data()}, wg);
        EXPECT_TRUE(same_storage(bh, bd, "B")) << "n " << n;
        EXPECT_TRUE(same_storage(a0, a, "A (read only)")) << "n " << n;
        EXPECT_TRUE(uniform_returns(ret, wg, nullptr)) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
    }

    // Host sysv on `route` (with the sytrf / sytrs pins `inner`) vs the device call, bit for bit.
    template <Uplo UL, class Tier, int WG>
    void sysv_case(int n, int nrhs, const ops::sysv::SysvChoice& route, const char* spelling, bool compose_wg, int wg = WG) {
        const Matrix<T> full = symmetric_batch<T>(n, 4000u + unsigned(n));
        Matrix<T> ah = stored(full, UL), ad = ah.clone();
        Matrix<T> bh = rhs<T>(n, nrhs, 5000u + unsigned(n)), bd = bh.clone();
        UnifiedVector<int64_t> hp(std::size_t(n) * kBatch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(kBatch, -12345);
        std::vector<std::pair<std::string, std::string>> want{{"sysv", spelling}};
        if (compose_wg) {
            want.emplace_back("sytrf", "wg");
            want.emplace_back("sytrs", "wg");
        }
        ASSERT_TRUE(reached(want, [&] {
            const select::ScopedPin<ops::sysv::SysvChoice> pin("sysv", route);
            const select::ScopedPin<ops::sytrf::SytrfChoice> pf("sytrf", ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
            const select::ScopedPin<ops::sytrs::SytrsChoice> ps("sytrs", ops::sytrs::SytrsChoice{ops::sytrs::Wg{}});
            host_sysv(ah, bh, UL, hp, hi);
        }));
        UnifiedVector<int32_t> dp(std::size_t(n) * kBatch, kPivSentinel), ret(std::size_t(kBatch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(kBatch) * kCanary, 0);
        dev_sysv<UL, Tier, WG>(q(), ad, bd, dp.data(), Outs{ret.data(), can.data()}, wg);
        const int32_t* hpi = reinterpret_cast<const int32_t*>(hp.data());
        EXPECT_TRUE(same_storage(ah, ad, "A")) << "n " << n;
        EXPECT_TRUE(same_storage(bh, bd, "B")) << "n " << n;
        EXPECT_TRUE(same_ints(hpi, dp.data(), std::size_t(n) * kBatch, "ipiv")) << "n " << n;
        EXPECT_TRUE(uniform_returns(ret, wg, hi.data())) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
        expect_mix(hi, hpi, n);
    }

    // Whether sytrs `cta` takes order n on this device (sytrs.hh's can_run).
    bool cta_fits(int n) {
        const auto& d = select::describe(this->ctx->device(), Bk, false);
        return n <= sycl_sytrf::kSytrsCtaMaxN && static_cast<int64_t>(sycl_sytrf::sytrs_cta_slm_bytes<T>(n)) <= d.slm_budget;
    }

    // ---- Blocked<NB> (NB > 1): a different summation order from the host `blocked` route, so the
    // comparison is info and forced-ipiv parity plus each solve's residual against the full matrix.

    // Item b's ipiv (int32, physical) and info as the kind forces them; false for random / nan.
    static bool forced(int b, int n, Uplo uplo, std::vector<int32_t>& ipiv, int32_t& info) {
        const BKind kd = kind_of(b);
        if (kd != BKind::pairs && kd != BKind::singular) return false;
        ipiv = forced_ipiv(kd, n, uplo);
        const int m = singular_m(b, n);
        info = kd == BKind::singular ? (uplo == Uplo::Lower ? m + 1 : n - m) : 0;
        return true;
    }

    static ::testing::AssertionResult forced_parity(const int32_t* ipiv, const int32_t* info, int n, int batch, Uplo uplo, const char* who) {
        std::vector<int32_t> want;
        int32_t want_info = 0;
        for (int b = 0; b < batch; ++b) {
            if (!forced(b, n, uplo, want, want_info)) continue;
            if (info[b] != want_info)
                return ::testing::AssertionFailure() << who << " item " << b << ": info " << info[b] << ", forced " << want_info;
            for (int k = 0; k < n; ++k)
                if (ipiv[std::size_t(b) * n + k] != want[k])
                    return ::testing::AssertionFailure() << who << " item " << b << ": ipiv[" << k << "] " << ipiv[std::size_t(b) * n + k]
                                                         << ", forced " << want[k];
        }
        return ::testing::AssertionSuccess();
    }

    // Items with info 0 and finite data: the ones whose solve has a residual to check.
    static std::vector<int> solvable(const int32_t* info, int batch, int skip = -1) {
        std::vector<int> r;
        for (int b = 0; b < batch; ++b)
            if (info[b] == 0 && kind_of(b) != BKind::nan && b != skip) r.push_back(b);
        return r;
    }

    double residual(const Matrix<T>& full, const Matrix<T>& x, const Matrix<T>& b0, const std::vector<int>& items) {
        return MV<T>::solve_residual(*this->ctx, full, x, b0, items);
    }

    // Host sytrf `blocked` vs the device call: every item's info (uniform over the executor), the
    // forced ipiv, A's poison, both canaries, and the device factor solving through sytrs `wg`.
    template <Uplo UL, int NB>
    void sytrf_blocked_case(int n, int batch) {
        using Tier = device::tier::Blocked<NB>;
        constexpr int wg = device::tier_info<device::routine::sytrf<UL>, Tier, T>::wg_size;
        const Matrix<T> full = blocked_batch<T>(n, batch, UL, 8000u + unsigned(n));
        Matrix<T> ah = stored(full, UL), ad = ah.clone();
        UnifiedVector<int64_t> hp(std::size_t(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(batch, -12345);
        ASSERT_TRUE(reached({{"sytrf", "blocked"}}, [&] {
            const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", ops::sytrf::SytrfChoice{ops::sytrf::Blocked{}});
            host_sytrf(ah, UL, hp, hi);
        }));
        UnifiedVector<int32_t> dp(std::size_t(n) * batch, kPivSentinel), ret(std::size_t(batch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(batch) * kCanary, 0);
        GlobalWs gw(device::sytrf_buffer_size<UL, Tier, T>(n).global_bytes, batch);
        dev_sytrf<UL, Tier, wg>(q(), ad, dp.data(), Outs{ret.data(), can.data()}, wg, 0, gw.view());
        const int32_t* hpi = reinterpret_cast<const int32_t*>(hp.data());
        EXPECT_TRUE(uniform_returns(ret, wg, hi.data())) << "n " << n;
        EXPECT_TRUE(forced_parity(dp.data(), hi.data(), n, batch, UL, "device")) << "n " << n;
        EXPECT_TRUE(forced_parity(hpi, hi.data(), n, batch, UL, "host blocked")) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
        EXPECT_TRUE(gw.intact()) << "n " << n;
        EXPECT_TRUE(ts::unwritten_outside<T>(ad, Live::triangle(UL))) << "n " << n;
        // The device factor solves: sytrs `wg` (Blocked<1>'s twin) on it, residual vs the full A.
        UnifiedVector<int64_t> dp64(std::size_t(n) * batch, 0);
        std::copy(dp.data(), dp.data() + dp.size(), reinterpret_cast<int32_t*>(dp64.data()));
        Matrix<T> x = rhs<T>(n, 2, 9000u + unsigned(n), batch);
        const Matrix<T> b0 = x.clone();
        {
            const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", ops::sytrs::SytrsChoice{ops::sytrs::Wg{}});
            host_sytrs(ad, x, UL, dp64);
        }
        const auto ok = solvable(hi.data(), batch);
        ASSERT_FALSE(ok.empty());
        EXPECT_VERIFY(T, verify::Check::solve, n, residual(full, x, b0, ok)) << "n " << n;
    }

    // Host sytrs `blocked` vs the device call on one host `wg` factor: A read only, B's poison, an
    // invalid list (item 1) all NaN on both sides, the residual of both solves.
    template <Uplo UL, int NB>
    void sytrs_blocked_case(int n, int nrhs, int batch) {
        using Tier = device::tier::Blocked<NB>;
        constexpr int wg = device::tier_info<device::routine::sytrs<UL>, Tier, T>::wg_size;
        const Matrix<T> full = blocked_batch<T>(n, batch, UL, 6000u + unsigned(n));
        Matrix<T> a = stored(full, UL);
        UnifiedVector<int64_t> hp(std::size_t(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(batch, -12345);
        {
            const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
            host_sytrf(a, UL, hp, hi);
        }
        int32_t* const hpi = reinterpret_cast<int32_t*>(hp.data());
        std::fill(hpi + n, hpi + 2 * n, 0);  // item 1: a list sytrf cannot write
        const Matrix<T> a0 = a.clone();
        Matrix<T> bh = rhs<T>(n, nrhs, 7000u + unsigned(n), batch), bd = bh.clone();
        const Matrix<T> b0 = bh.clone();
        ASSERT_TRUE(reached({{"sytrs", "blocked"}}, [&] {
            const select::ScopedPin<ops::sytrs::SytrsChoice> pin("sytrs", ops::sytrs::SytrsChoice{ops::sytrs::Blocked{}});
            host_sytrs(a, bh, UL, hp);
        }));
        UnifiedVector<int32_t> ret(std::size_t(batch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(batch) * kCanary, 0);
        GlobalWs gw(device::sytrs_buffer_size<UL, Tier, T>(n, nrhs).global_bytes, batch);
        dev_sytrs<UL, Tier, wg>(q(), a, bd, hpi, Outs{ret.data(), can.data()}, wg, gw.view());
        EXPECT_TRUE(same_storage(a0, a, "A (read only)")) << "n " << n;
        EXPECT_TRUE(ts::unwritten_outside<T>(bd, Live::block())) << "n " << n;
        EXPECT_TRUE(uniform_returns(ret, wg, nullptr)) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
        EXPECT_TRUE(gw.intact()) << "n " << n;
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) {
                EXPECT_TRUE(has_nan(bd(i, c, 1))) << "device: item 1 row " << i << " col " << c << " not NaN";
                EXPECT_TRUE(has_nan(bh(i, c, 1))) << "host: item 1 row " << i << " col " << c << " not NaN";
            }
        const auto ok = solvable(hi.data(), batch, 1);
        ASSERT_FALSE(ok.empty());
        EXPECT_VERIFY(T, verify::Check::solve, n, residual(full, bd, b0, ok)) << "device n " << n << " nrhs " << nrhs;
        EXPECT_VERIFY(T, verify::Check::solve, n, residual(full, bh, b0, ok)) << "host n " << n << " nrhs " << nrhs;
    }

    // Host sysv `compose` over sytrf / sytrs `blocked` vs the device call: info parity, forced
    // ipiv, a singular item's B as given (bit for bit), the residual of every solvable item.
    template <Uplo UL, int NB>
    void sysv_blocked_case(int n, int nrhs, int batch) {
        using Tier = device::tier::Blocked<NB>;
        constexpr int wg = device::tier_info<device::routine::sysv<UL>, Tier, T>::wg_size;
        const Matrix<T> full = blocked_batch<T>(n, batch, UL, 10000u + unsigned(n));
        Matrix<T> ah = stored(full, UL), ad = ah.clone();
        Matrix<T> bh = rhs<T>(n, nrhs, 11000u + unsigned(n), batch), bd = bh.clone();
        const Matrix<T> b0 = bh.clone();
        UnifiedVector<int64_t> hp(std::size_t(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL));
        UnifiedVector<int32_t> hi(batch, -12345);
        ASSERT_TRUE(reached({{"sysv", "compose"}, {"sytrf", "blocked"}, {"sytrs", "blocked"}}, [&] {
            const select::ScopedPin<ops::sysv::SysvChoice> pin("sysv", ops::sysv::SysvChoice{ops::sysv::Compose{}});
            const select::ScopedPin<ops::sytrf::SytrfChoice> pf("sytrf", ops::sytrf::SytrfChoice{ops::sytrf::Blocked{}});
            const select::ScopedPin<ops::sytrs::SytrsChoice> ps("sytrs", ops::sytrs::SytrsChoice{ops::sytrs::Blocked{}});
            host_sysv(ah, bh, UL, hp, hi);
        }));
        UnifiedVector<int32_t> dp(std::size_t(n) * batch, kPivSentinel), ret(std::size_t(batch) * wg, -12345);
        UnifiedVector<unsigned char> can(std::size_t(batch) * kCanary, 0);
        GlobalWs gw(device::sysv_buffer_size<UL, Tier, T>(n, nrhs).global_bytes, batch);
        dev_sysv<UL, Tier, wg>(q(), ad, bd, dp.data(), Outs{ret.data(), can.data()}, wg, gw.view());
        EXPECT_TRUE(uniform_returns(ret, wg, hi.data())) << "n " << n;
        EXPECT_TRUE(forced_parity(dp.data(), hi.data(), n, batch, UL, "device")) << "n " << n;
        EXPECT_TRUE(canary_intact(can)) << "n " << n;
        EXPECT_TRUE(gw.intact()) << "n " << n;
        EXPECT_TRUE(ts::unwritten_outside<T>(ad, Live::triangle(UL))) << "n " << n;
        EXPECT_TRUE(ts::unwritten_outside<T>(bd, Live::block())) << "n " << n;
        std::vector<int> singular;
        for (int b = 0; b < batch; ++b)
            if (hi.data()[b] != 0) singular.push_back(b);
        ASSERT_FALSE(singular.empty());
        EXPECT_TRUE(ts::items_bit_identical_to<T>(bd, b0, singular)) << "n " << n << ": a singular item's B changed";
        const auto ok = solvable(hi.data(), batch);
        ASSERT_FALSE(ok.empty());
        EXPECT_VERIFY(T, verify::Check::solve, n, residual(full, bd, b0, ok)) << "device n " << n;
        EXPECT_VERIFY(T, verify::Check::solve, n, residual(full, bh, b0, ok)) << "host n " << n;
    }
};

using DeviceApiSytrfTypes = typename test_utils::backend_types<ts::Config>::type;
TYPED_TEST_SUITE(DeviceApiSytrfTest, DeviceApiSytrfTypes);

TYPED_TEST(DeviceApiSytrfTest, SytrfCtaIsTheHostCtaRouteBitForBit) {
    const ops::sytrf::SytrfChoice cta{ops::sytrf::Cta{}};
    for (int n : {1, 2, 17, 31, 32}) {
        this->template sytrf_case<Uplo::Lower, device::tier::Cta<32>, 32>(n, cta, "cta");
        this->template sytrf_case<Uplo::Upper, device::tier::Cta<32>, 32>(n, cta, "cta");
    }
    // Two sub-groups: team 0 factors, the info is broadcast; and an RtGroup of 96.
    this->template sytrf_case<Uplo::Lower, device::tier::Cta<32>, 64>(32, cta, "cta");
    this->template sytrf_case<Uplo::Upper, device::tier::Cta<32>, 0>(29, cta, "cta", 96);
}

TYPED_TEST(DeviceApiSytrfTest, SytrfBlocked1IsTheHostWgRouteBitForBit) {
    const ops::sytrf::SytrfChoice wg{ops::sytrf::Wg{}};
    for (int n : {33, 64, 129, 257}) {
        this->template sytrf_case<Uplo::Lower, device::tier::Blocked<1>, 256>(n, wg, "wg");
        this->template sytrf_case<Uplo::Upper, device::tier::Blocked<1>, 0>(n, wg, "wg", host_width(n));
    }
    this->template sytrf_case<Uplo::Lower, device::tier::Blocked<1>, 32>(5, wg, "wg");  // one sub-group
}

TYPED_TEST(DeviceApiSytrfTest, SytrsCtaIsTheHostCtaRouteBitForBit) {
    const ops::sytrs::SytrsChoice cta{ops::sytrs::Cta{}};
    // A pinned route that cannot run throws: every order is checked against this device's local memory.
    this->template sytrs_case<Uplo::Lower, device::tier::Cta<32>, 32>(1, 2, cta, "cta");
    this->template sytrs_case<Uplo::Upper, device::tier::Cta<32>, 32>(32, 3, cta, "cta");
    if (this->cta_fits(33)) this->template sytrs_case<Uplo::Lower, device::tier::Cta<64>, 64>(33, 5, cta, "cta");  // 2 columns per sub-group
    if (this->cta_fits(64)) this->template sytrs_case<Uplo::Upper, device::tier::Cta<64>, 0>(64, 2, cta, "cta", 128);
    for (int n : {65, 100, 128})
        if (this->cta_fits(n)) {
            this->template sytrs_case<Uplo::Lower, device::tier::Cta<128>, 32>(n, 3, cta, "cta");
            this->template sytrs_case<Uplo::Upper, device::tier::Cta<128>, 32>(n, 1, cta, "cta");
        }
}

TYPED_TEST(DeviceApiSytrfTest, SytrsBlocked1IsTheHostWgRouteBitForBit) {
    const ops::sytrs::SytrsChoice wg{ops::sytrs::Wg{}};
    // The sums reduce over the executor's width: the host's width is the parity width.
    for (int n : {3, 40, 129, 300}) {
        this->template sytrs_case<Uplo::Lower, device::tier::Blocked<1>, 0>(n, 2, wg, "wg", host_width(n));
        this->template sytrs_case<Uplo::Upper, device::tier::Blocked<1>, 0>(n, 3, wg, "wg", host_width(n));
    }
    this->template sytrs_case<Uplo::Lower, device::tier::Blocked<1>, 256>(300, 1, wg, "wg");
}

TYPED_TEST(DeviceApiSytrfTest, SysvCtaIsTheHostCtaRouteBitForBit) {
    const ops::sysv::SysvChoice cta{ops::sysv::Cta{}};
    for (int n : {1, 7, 32}) {
        this->template sysv_case<Uplo::Lower, device::tier::Cta<32>, 32>(n, 3, cta, "cta", false);
        this->template sysv_case<Uplo::Upper, device::tier::upto<32>, 128>(n, 5, cta, "cta", false);  // four sub-groups
    }
}

TYPED_TEST(DeviceApiSytrfTest, SysvBlocked1IsTheHostComposeOverWgBitForBit) {
    const ops::sysv::SysvChoice compose{ops::sysv::Compose{}};
    for (int n : {33, 150}) {
        this->template sysv_case<Uplo::Lower, device::tier::Blocked<1>, 0>(n, 2, compose, "compose", true, host_width(n));
        this->template sysv_case<Uplo::Upper, device::tier::Blocked<1>, 0>(n, 1, compose, "compose", true, host_width(n));
    }
}

// ---- Blocked<NB>: the in-group blocked tiers, NB = the host route's 31 -------------------------
constexpr int kNb = sycl_sytrf::kSytrfBlockedNb;
static_assert(kNb == 31, "the n lists below straddle panels of 31");
constexpr int kBlockedBatch = 128;

// n straddles every panel end (31 | 32, 62 | 63 | 64, 93 | 94) and n <= NB (the ?sytf2 tail only).
TYPED_TEST(DeviceApiSytrfTest, SytrfBlockedAgreesWithTheHostBlockedRoute) {
    for (int n : {5, 31, 32, 33, 62, 63, 64, 94, 150}) {
        this->template sytrf_blocked_case<Uplo::Lower, kNb>(n, kBlockedBatch);
        this->template sytrf_blocked_case<Uplo::Upper, kNb>(n, kBlockedBatch);
    }
}

TYPED_TEST(DeviceApiSytrfTest, SytrsBlockedAgreesWithTheHostBlockedRoute) {
    // trsm's diagonal blocks are NB wide: n straddles them; nrhs past one 32-wide C tile.
    for (int n : {5, 31, 32, 33, 64, 100}) {
        this->template sytrs_blocked_case<Uplo::Lower, kNb>(n, 3, kBlockedBatch);
        this->template sytrs_blocked_case<Uplo::Upper, kNb>(n, 1, kBlockedBatch);
    }
    this->template sytrs_blocked_case<Uplo::Upper, kNb>(70, 40, 32);
}

TYPED_TEST(DeviceApiSytrfTest, SysvBlockedAgreesWithTheHostComposeOverBlocked) {
    for (int n : {5, 32, 63, 120}) {
        this->template sysv_blocked_case<Uplo::Lower, kNb>(n, 2, kBlockedBatch);
        this->template sysv_blocked_case<Uplo::Upper, kNb>(n, 3, kBlockedBatch);
    }
}

// A saturating batch of one random problem (three panels and the tail): every item bit-identical
// to item 0, A, ipiv and the return value.
TYPED_TEST(DeviceApiSytrfTest, SytrfBlockedSaturatingBatchIsBitIdenticalToItemZero) {
    using T = typename TestFixture::T;
    using Tier = device::tier::Blocked<kNb>;
    const int n = 94;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        Matrix<T> one = blocked_batch<T>(n, 1, uplo, 4242u);
        Matrix<T> full(n, n, kBatch);
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) full(i, j, b) = one(i, j, 0);
        Matrix<T> a = stored(full, uplo);
        UnifiedVector<int32_t> dp(std::size_t(n) * kBatch, kPivSentinel), ret(std::size_t(kBatch) * 256, -12345);
        UnifiedVector<unsigned char> can(std::size_t(kBatch) * kCanary, 0);
        if (uplo == Uplo::Lower) {
            GlobalWs gw(device::sytrf_buffer_size<Uplo::Lower, Tier, T>(n).global_bytes, kBatch);
            dev_sytrf<Uplo::Lower, Tier, 256>(this->q(), a, dp.data(), Outs{ret.data(), can.data()}, 256, 0, gw.view());
            EXPECT_TRUE(gw.intact());
        } else {
            GlobalWs gw(device::sytrf_buffer_size<Uplo::Upper, Tier, T>(n).global_bytes, kBatch);
            dev_sytrf<Uplo::Upper, Tier, 256>(this->q(), a, dp.data(), Outs{ret.data(), can.data()}, 256, 0, gw.view());
            EXPECT_TRUE(gw.intact());
        }
        EXPECT_TRUE(ts::items_bit_identical<T>(a)) << ts::uplo_name(uplo);
        for (int b = 1; b < kBatch; ++b)
            ASSERT_TRUE(same_ints(dp.data(), dp.data() + std::size_t(b) * n, std::size_t(n), "ipiv vs item 0")) << "item " << b;
        EXPECT_TRUE(uniform_returns(ret, 256, nullptr)) << ts::uplo_name(uplo);
        EXPECT_TRUE(canary_intact(can));
    }
}

// The global canary's own red case: W carved 64 bytes late runs into it, so it must fail.
TYPED_TEST(DeviceApiSytrfTest, ShiftedGlobalWorkspaceBreaksTheCanary) {
    using T = typename TestFixture::T;
    using Tier = device::tier::Blocked<kNb>;
    const int n = 40, batch = 32;  // n > NB: the first panel zero-fills all of W
    Matrix<T> a = stored(blocked_batch<T>(n, batch, Uplo::Lower, 78u), Uplo::Lower);
    UnifiedVector<int32_t> dp(std::size_t(n) * batch, kPivSentinel), ret(std::size_t(batch) * 256, -12345);
    UnifiedVector<unsigned char> can(std::size_t(batch) * kCanary, 0);
    GlobalWs gw(device::sytrf_buffer_size<Uplo::Lower, Tier, T>(n).global_bytes, batch);
    dev_sytrf<Uplo::Lower, Tier, 256>(this->q(), a, dp.data(), Outs{ret.data(), can.data()}, 256, 0, gw.view(kCanary));
    EXPECT_FALSE(gw.intact()) << "the global canary did not see a write past buffer_size";
}

// The canary guard's own red case: the tile carved 64 bytes late runs into it, so it must fail.
TYPED_TEST(DeviceApiSytrfTest, ShiftedWorkspaceBreaksTheCanary) {
    using T = typename TestFixture::T;
    const int n = 32;  // the tile's last lower element sits in the tile's last 64 bytes
    Matrix<T> a = stored(symmetric_batch<T>(n, 77u), Uplo::Lower);
    UnifiedVector<int32_t> dp(std::size_t(n) * kBatch, kPivSentinel), ret(std::size_t(kBatch) * 32, -12345);
    UnifiedVector<unsigned char> can(std::size_t(kBatch) * kCanary, 0);
    dev_sytrf<Uplo::Lower, device::tier::Cta<32>, 32>(this->q(), a, dp.data(), Outs{ret.data(), can.data()}, 32, kCanary);
    EXPECT_FALSE(canary_intact(can)) << "the canary did not see a write past buffer_size";
}

// Argument errors: LAPACK's numbers, the same on every item, nothing written.
TYPED_TEST(DeviceApiSytrfTest, ArgumentErrorsAreUniformAndWriteNothing) {
    using T = typename TestFixture::T;
    const int n = 17, nrhs = 2;
    Matrix<T> a = stored(symmetric_batch<T>(n, 91u), Uplo::Lower);
    const Matrix<T> a0 = a.clone();
    Matrix<T> bm = rhs<T>(n, nrhs, 92u);
    const Matrix<T> b0 = bm.clone();
    const auto av = a.kernel_view();
    const auto bv = bm.kernel_view();
    UnifiedVector<int32_t> dp(std::size_t(n) * kBatch, kPivSentinel);
    int32_t* const ip = dp.data();
    auto run = [&](int wg, auto body) {
        UnifiedVector<int32_t> ret(std::size_t(kBatch) * wg, -12345);
        int32_t* const r = ret.data();
        launch(this->q(), wg, 4096, [=](sycl::nd_item<1> it, unsigned char* lm) {
            r[it.get_global_linear_id()] = body(it, static_cast<int>(it.get_group_linear_id()), local_ws(lm, 0));
        });
        const int32_t first = r[0];
        for (std::size_t e = 0; e < ret.size(); ++e)
            if (r[e] != first) ADD_FAILURE() << "item " << e << " returned " << r[e] << ", item 0 " << first;
        return first;
    };
    // n = 17 above Cta<16>: -2 (N).
    EXPECT_EQ(run(32, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  return device::sytrf<Uplo::Lower, device::tier::Cta<16>>(device::WorkGroup<32, 32>{it}, device::item(av, b, n, n), ip + b * n, w);
              }),
              -2);
    // ld = n - 1: -4 (LDA).
    EXPECT_EQ(run(32, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  const device::Mat<T> m{av.data() + std::ptrdiff_t(b) * av.stride(), device::Dyn{n}, device::Dyn{n}, device::Dyn{n - 1}};
                  return device::sytrf<Uplo::Lower, device::tier::Cta<32>>(device::WorkGroup<32, 32>{it}, m, ip + b * n, w);
              }),
              -4);
    // B with n - 1 rows: -7 (B), sytrs and sysv alike.
    EXPECT_EQ(run(32, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  return device::sytrs<Uplo::Lower, device::tier::Cta<32>>(device::WorkGroup<32, 32>{it}, device::item(av, b, n, n),
                                                                   device::item(bv, b, n - 1, nrhs), ip + b * n, w);
              }),
              -7);
    EXPECT_EQ(run(32, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  return device::sysv<Uplo::Upper, device::tier::Cta<32>>(device::WorkGroup<32, 32>{it}, device::item(av, b, n, n),
                                                                  device::item(bv, b, n - 1, nrhs), ip + b * n, w);
              }),
              -7);
    // A run-time width of 48 (a partial sub-group): -1 (the executor), before any collective.
    EXPECT_EQ(run(48, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  return device::sytrf<Uplo::Lower, device::tier::Blocked<1>>(device::RtGroup<>(it), device::item(av, b, n, n), ip + b * n, w);
              }),
              -1);
    // Blocked<31> checks first as well: ld = n - 1 (-4), B with n - 1 rows (-7), no workspace touched.
    EXPECT_EQ(run(256, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  const device::Mat<T> m{av.data() + std::ptrdiff_t(b) * av.stride(), device::Dyn{n}, device::Dyn{n}, device::Dyn{n - 1}};
                  return device::sytrf<Uplo::Upper, device::tier::Blocked<31>>(device::WorkGroup<256, 32>{it}, m, ip + b * n, w);
              }),
              -4);
    EXPECT_EQ(run(256, [=](sycl::nd_item<1> it, int b, device::Workspace w) {
                  return device::sysv<Uplo::Lower, device::tier::Blocked<31>>(device::WorkGroup<256, 32>{it}, device::item(av, b, n, n),
                                                                     device::item(bv, b, n - 1, nrhs), ip + b * n, w);
              }),
              -7);
    EXPECT_TRUE(same_storage(a0, a, "A"));
    EXPECT_TRUE(same_storage(b0, bm, "B"));
    for (std::size_t e = 0; e < dp.size(); ++e) ASSERT_EQ(dp.data()[e], kPivSentinel) << "ipiv written at " << e;
}

// ---- with_tier: host side, mock device limits ----------------------------------------------------
device::DeviceLimits big_limits() { return {std::size_t(100) << 10, {16, 32}, 1024, true}; }

template <class Rt, class T, class List = device::tiers_t<Rt, T>>
std::string pick(const device::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        device::with_tier<Rt, T, List>(lim, n, [&]<class Tier>(Tier) {
            if constexpr (requires { Tier::nb; }) got = "blocked" + std::to_string(Tier::nb);
            else got = "cta" + std::to_string(Tier::n_max);
        });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string("throw:") + e.what();
    }
    return got;
}

using SytrfL = device::routine::sytrf<Uplo::Lower>;
using SytrsU = device::routine::sytrs<Uplo::Upper>;
using SysvL = device::routine::sysv<Uplo::Lower>;

TEST(DeviceApiSytrfTiers, WithTierPicksTheFirstAdmittingTierAtEveryBoundary) {
    const auto lim = big_limits();
    EXPECT_EQ((pick<SytrfL, float>(lim, 1)), "cta32");
    EXPECT_EQ((pick<SytrfL, float>(lim, 32)), "cta32");
    EXPECT_EQ((pick<SytrfL, float>(lim, 33)), "blocked1");
    EXPECT_EQ((pick<SytrfL, std::complex<double>>(lim, 5000)), "blocked1");
    EXPECT_EQ((pick<SytrsU, float>(lim, 32)), "cta32");
    EXPECT_EQ((pick<SytrsU, float>(lim, 33)), "cta64");
    EXPECT_EQ((pick<SytrsU, float>(lim, 64)), "cta64");
    EXPECT_EQ((pick<SytrsU, float>(lim, 65)), "cta128");
    EXPECT_EQ((pick<SytrsU, float>(lim, 128)), "cta128");
    EXPECT_EQ((pick<SytrsU, float>(lim, 129)), "blocked1");
    EXPECT_EQ((pick<SysvL, double>(lim, 32)), "cta32");
    EXPECT_EQ((pick<SysvL, double>(lim, 33)), "blocked1");
}

TEST(DeviceApiSytrfTiers, EachDeviceFilterRemovesItsTier) {
    // sytrs Cta<128> float at n = 100: fits its buffer_size exactly, not one byte less.
    const std::size_t need = device::sytrs_buffer_size<Uplo::Upper, device::tier::Cta<128>, float>(100).local_bytes;
    device::DeviceLimits slm = big_limits();
    slm.local_mem_bytes = need;
    EXPECT_EQ((pick<SytrsU, float>(slm, 100)), "cta128");
    slm.local_mem_bytes = need - 1;
    EXPECT_EQ((pick<SytrsU, float>(slm, 100)), "blocked1");
    // No fp64: no double tier at all; float unaffected.
    device::DeviceLimits nofp64 = big_limits();
    nofp64.fp64 = false;
    EXPECT_EQ((pick<SytrfL, float>(nofp64, 10)), "cta32");
    EXPECT_EQ((pick<SytrfL, double>(nofp64, 10)), "throw:sytrf: no device tier admits n=10");
    // No sub-group size 32: nothing.
    device::DeviceLimits sg16 = big_limits();
    sg16.sub_group_sizes = {16};
    EXPECT_EQ((pick<SytrsU, float>(sg16, 5)), "throw:sytrs: no device tier admits n=5");
    // A 128-item ceiling drops Blocked<1> (256 items), not Cta (32).
    device::DeviceLimits narrow = big_limits();
    narrow.max_work_group_size = 128;
    EXPECT_EQ((pick<SysvL, float>(narrow, 20)), "cta32");
    EXPECT_EQ((pick<SysvL, float>(narrow, 40)), "throw:sysv: no device tier admits n=40");
}

// Blocked<31> is the caller's to list: first after the small tiers, and filtered by its 256 items
// and its local part (one byte short drops it).
TEST(DeviceApiSytrfTiers, WithTierTakesBlockedFromACallerList) {
    using SytrfList = device::tier::list<device::tier::Cta<32>, device::tier::Blocked<31>>;
    using SytrsList = device::tier::list<device::tier::Cta<32>, device::tier::Cta<64>, device::tier::Cta<128>, device::tier::Blocked<31>>;
    const auto lim = big_limits();
    EXPECT_EQ((pick<SytrfL, float, SytrfList>(lim, 32)), "cta32");
    EXPECT_EQ((pick<SytrfL, float, SytrfList>(lim, 33)), "blocked31");
    EXPECT_EQ((pick<SytrfL, std::complex<double>, SytrfList>(lim, 5000)), "blocked31");
    EXPECT_EQ((pick<SytrsU, float, SytrsList>(lim, 129)), "blocked31");
    EXPECT_EQ((pick<SysvL, double, SytrfList>(lim, 33)), "blocked31");
    device::DeviceLimits narrow = big_limits();
    narrow.max_work_group_size = 128;
    EXPECT_EQ((pick<SytrfL, float, SytrfList>(narrow, 33)), "throw:sytrf: no device tier admits n=33");
    const std::size_t need = device::sytrf_buffer_size<Uplo::Lower, device::tier::Blocked<31>, float>(33).local_bytes;
    device::DeviceLimits slm = big_limits();
    slm.local_mem_bytes = need;
    EXPECT_EQ((pick<SytrfL, float, SytrfList>(slm, 33)), "blocked31");
    slm.local_mem_bytes = need - 1;
    EXPECT_EQ((pick<SytrfL, float, SytrfList>(slm, 33)), "throw:sytrf: no device tier admits n=33");
}

}  // namespace
