// Native batched sytrs from Bunch-Kaufman factors, LAPACK ?sytrs2 form (sytrs_device.hh): `cta`
// (one work-group per item, the converted factor in local memory, one sub-group per right-hand
// side) and `blocked` (the converted factor and the gathered B in workspace per chunk of items,
// then two public trsm around a D^-1 kernel, and the scatter). evidence: docs/perf/sytrf.md
//
// Belongs in EXTENSIONS_FACTORIZATION_SOURCES with sytrf_kernels.cc: it shares no device symbol with
// any other cluster.

#include <src/ops/sytrf/sytrf_native.hh>
#include <src/ops/sytrs/sytrs_device.hh>

#include <src/queue.hh>
#include <src/select/grid_chunk.hh>
#include <src/select/select.hh>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/util/batch_mask.hh>

#include <batchlas/error.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

namespace batchlas {
namespace sycl_sytrf {

template <typename T, int R> class SytrsCtaKernel;
template <typename T> class SytrsPrepKernel;
template <typename T> class SytrsDiagKernel;
template <typename T> class SytrsScatterKernel;

namespace {

using sytrf_device::Staged;
using sytrf_device::Tile;
using sytrf_device::Vec;

template <typename T>
using Dev = typename sycl_device::DevMap<T>::type;

void check_device(Queue& ctx, const char* who) {
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) throw batchlas::invalid_argument(std::string(who) + ": GPU queues only");
    if (!dev.supports_sub_group_size(32))
        throw batchlas::unsupported(std::string(who) + ": device does not offer sub-group size 32");
}

template <typename T>
void check_solve_args(const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                      Uplo uplo, Span<int64_t> pivots, const char* who) {
    const auto n = A.rows();
    if (n < 1 || A.cols() != n || A.batch_size() < 1)
        throw batchlas::invalid_argument(std::string(who) + ": A must be square with n >= 1 and batch >= 1");
    if (A.is_heterogeneous() || B.is_heterogeneous())
        throw batchlas::invalid_argument(std::string(who) + ": heterogeneous batch is not supported");
    if (B.rows() != n || B.batch_size() != A.batch_size() || B.cols() < 1)
        throw batchlas::invalid_argument(std::string(who) + ": B must be n x nrhs, nrhs >= 1, with A's batch");
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) throw batchlas::invalid_argument(std::string(who) + ": invalid uplo");
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(A.batch_size()))
        throw batchlas::invalid_argument(std::string(who) + ": pivot span is shorter than n * batch");
}

template <typename T, int R>
Event cta_launch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 bool rev, const int* piv, const int32_t* skip, const std::uint8_t* mask, int nsg) {
    using D = Dev<T>;
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const int nrhs = static_cast<int>(B.cols());
    const int ldt = n | 1;
    const int wg = 32 * nsg;
    const D* const ap = reinterpret_cast<const D*>(A.data_ptr());
    D* const bp = reinterpret_cast<D*>(B.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stra = A.stride(), ldb = B.ld(), strb = B.stride();
    const std::size_t dbytes = sytrs_cta_slm_bytes<T>(n) - (3 * static_cast<std::size_t>(n) + 1) * sizeof(int);
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> lm(sycl::range<1>(dbytes / sizeof(D)), h);
        sycl::local_accessor<int, 1> im(sycl::range<1>(3 * static_cast<std::size_t>(n) + 1), h);
        batchlas::impl::parallel_for<SytrsCtaKernel<T, R>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                if (mask != nullptr && mask[b] == 0) return;  // I1: before the first barrier
                if (skip != nullptr && skip[b] != 0) return;
                const int lid = static_cast<int>(it.get_local_id(0));
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());
                const Staged<D> S{&lm[0], ldt, n, &lm[static_cast<std::size_t>(n) * ldt], &im[0], &im[n], &im[2 * n]};
                const Tile<D> G{const_cast<D*>(ap) + b * stra, lda, n, rev};
                for (int idx = lid; idx < n * n; idx += wg) {
                    const int i = idx % n, j = idx / n;
                    if (i >= j) S(i, j) = G(i, j);
                }
                const int* const gp = piv + b * n;
                for (int k = lid; k < n; k += wg) S.ps[k] = sytrf_device::logical_piv(gp[rev ? n - 1 - k : k], n, rev);
                auto bar = [&] { sycl::group_barrier(it.get_group()); };
                bar();
                const bool ok = sytrf_device::ldl_convert<D>(S, n, S.ps, S.perm, S.kind, S.e, lid, wg, &im[3 * n], bar);
                sytrf_device::solve_columns<R>(sg, lane, sgid, nsg, S, ok, rev, bp + b * strb, ldb, nrhs);
            });
    });
    return ctx.get_event();
}

// The blocked tier's scratch, one part's worth: the converted factor and B, n x n and n x nrhs per
// item with ld n; the logical ipiv, perm and kind; one pointer array per trsm operand.
template <typename T>
struct BlockedWs {
    Span<T> lw, bw;
    Span<int> ps, perm, kind, sig, inv;
    Span<T*> pl, pb;
};

template <typename T>
BlockedWs<T> blocked_layout(Queue& ctx, BumpAllocator& pool, int n, int nrhs, int part) {
    const auto p = static_cast<std::size_t>(part);
    const auto un = static_cast<std::size_t>(n);
    BlockedWs<T> ws;
    ws.lw = pool.allocate<T>(ctx, p * un * un);
    ws.bw = pool.allocate<T>(ctx, p * un * static_cast<std::size_t>(nrhs));
    ws.ps = pool.allocate<int>(ctx, p * un);
    ws.perm = pool.allocate<int>(ctx, p * un);
    ws.kind = pool.allocate<int>(ctx, p * un);
    ws.sig = pool.allocate<int>(ctx, p * un);
    ws.inv = pool.allocate<int>(ctx, p * un);
    ws.pl = pool.allocate<T*>(ctx, p);
    ws.pb = pool.allocate<T*>(ctx, p);
    return ws;
}

// Items per pass: getrf's L2 rule on the per-item workspace, then capped by one call's scratch
// lease and the grid. The query and the call must agree. Untimed.
template <typename T>
int blocked_chunk(Queue& ctx, int n, int nrhs, int batch) {
    const auto dev = ctx.device();
    const int64_t item = (static_cast<int64_t>(n) * n + static_cast<int64_t>(n) * nrhs) * static_cast<int64_t>(sizeof(T)) +
                         5 * static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(int));
    int64_t chunk = tuning::blocked_chunk_items(
        batch, item, static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
    const int64_t lease = select::describe(dev, ctx.backend(), false).scratch_budget;
    if (lease > 0) chunk = std::max<int64_t>(1, std::min<int64_t>(chunk, lease / 2 / item));
    return select::grid_capped_chunk(ctx, static_cast<int>(std::min<int64_t>(chunk, batch)), batch);
}

inline int blocked_part(int chunk, int batch) { return batch < 2 * chunk ? batch : chunk + batch % chunk; }

// One work-group per item: stage and convert the factor into lw, gather B into bw (P^T B).
template <typename T>
void prep_launch(Queue& ctx, const T* a, int lda, int stra, const T* b, int ldb, int strb, const int* piv, int n,
                 int nrhs, int cb, bool rev, int wg, const BlockedWs<T>& ws, const int32_t* skip,
                 const std::uint8_t* mask) {
    using D = Dev<T>;
    const D* const ap = reinterpret_cast<const D*>(a);
    const D* const bp = reinterpret_cast<const D*>(b);
    D* const lw = reinterpret_cast<D*>(ws.lw.data());
    D* const bw = reinterpret_cast<D*>(ws.bw.data());
    int* const psw = ws.ps.data();
    int* const pw = ws.perm.data();
    int* const kw = ws.kind.data();
    int* const sw = ws.sig.data();
    int* const iw = ws.inv.data();
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> flag(sycl::range<1>(1), h);
        batchlas::impl::parallel_for<SytrsPrepKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(cb) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const std::ptrdiff_t i0 = static_cast<std::ptrdiff_t>(it.get_group(0));
                if (mask != nullptr && mask[i0] == 0) return;
                if (skip != nullptr && skip[i0] != 0) return;
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n) * n, nb = static_cast<std::ptrdiff_t>(n) * nrhs;
                const Tile<D> G{const_cast<D*>(ap) + i0 * stra, lda, n, rev};
                const Tile<D> W{lw + i0 * nn, n, n, rev};
                int* const ps = psw + i0 * n;
                int* const perm = pw + i0 * n;
                int* const kind = kw + i0 * n;
                for (int k = lid; k < n; k += wg) ps[k] = sytrf_device::logical_piv(piv[i0 * n + (rev ? n - 1 - k : k)], n, rev);
                auto bar = [&] { sycl::group_barrier(it.get_group()); };
                bar();
                if (lid == 0) {
                    const bool valid = sytrf_device::piv_list_valid(ps, n);
                    flag[0] = valid ? 1 : 0;
                    if (valid) sytrf_device::build_perm(ps, n, perm, kind);
                }
                bar();
                const bool ok = flag[0] != 0;
                if (ok) {
                    sytrf_device::convert_gather<D>(W, G, n, ps, kind, sw + i0 * n, iw + i0 * n, lid, wg, bar);
                } else {
                    for (std::ptrdiff_t idx = lid; idx < nn; idx += wg) {
                        const int i = static_cast<int>(idx % n), j = static_cast<int>(idx / n);
                        if (i > j) W(i, j) = G(i, j);
                    }
                    for (int t = lid; t < n; t += wg) {
                        perm[t] = t;
                        kind[t] = 0;
                    }
                    bar();
                }
                D* const y = bw + i0 * nb;
                const D* const src = bp + i0 * strb;
                for (std::ptrdiff_t idx = lid; idx < nb; idx += wg) {
                    const int t = static_cast<int>(idx % n), c = static_cast<int>(idx / n);
                    const int lt = rev ? n - 1 - t : t;
                    const int s = rev ? n - 1 - perm[lt] : perm[lt];
                    y[idx] = ok ? src[s + static_cast<std::ptrdiff_t>(c) * ldb] : sytrf_device::nan_of<D>();
                }
            });
    });
}

// bw := D^-1 bw, one work-item per (row, rhs, item); the first row of a 2 x 2 block writes both.
template <typename T>
void diag_launch(Queue& ctx, const T* a, int lda, int stra, int n, int nrhs, int cb, bool rev, const BlockedWs<T>& ws,
                 const int32_t* skip, const std::uint8_t* mask) {
    using D = Dev<T>;
    const D* const ap = reinterpret_cast<const D*>(a);
    D* const bw = reinterpret_cast<D*>(ws.bw.data());
    const int* const kw = ws.kind.data();
    const std::size_t total = static_cast<std::size_t>(cb) * nrhs * n;
    constexpr std::size_t kWg = 256;
    ctx->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<SytrsDiagKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>((total + kWg - 1) / kWg * kWg), sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) {
                const std::size_t g = it.get_global_id(0);
                if (g >= total) return;
                const int t = static_cast<int>(g % n);
                const std::size_t rest = g / n;
                const std::ptrdiff_t i0 = static_cast<std::ptrdiff_t>(rest / nrhs);
                if (mask != nullptr && mask[i0] == 0) return;
                if (skip != nullptr && skip[i0] != 0) return;
                const int kd = kw[i0 * n + t];
                if (kd == 2) return;
                const Tile<D> G{const_cast<D*>(ap) + i0 * stra, lda, n, rev};
                const Vec<D> x{bw + static_cast<std::ptrdiff_t>(rest) * n, n, rev};
                if (kd == 0) {
                    x(t) = sytrf_device::d_solve_row(0, x(t), x(t), G(t, t), G(t, t), G(t, t));
                    return;
                }
                const D x0 = x(t), x1 = x(t + 1), d0 = G(t, t), d1 = G(t + 1, t + 1), off = G(t + 1, t);
                x(t) = sytrf_device::d_solve_row(1, x0, x1, d0, d1, off);
                x(t + 1) = sytrf_device::d_solve_row(2, x1, x0, d1, d0, off);
            });
    });
}

// B := P bw: row t of the solved system goes back to the row it was gathered from.
template <typename T>
void scatter_launch(Queue& ctx, T* b, int ldb, int strb, int n, int nrhs, int cb, bool rev, const BlockedWs<T>& ws,
                    const int32_t* skip, const std::uint8_t* mask) {
    using D = Dev<T>;
    D* const bp = reinterpret_cast<D*>(b);
    const D* const bw = reinterpret_cast<const D*>(ws.bw.data());
    const int* const pw = ws.perm.data();
    const std::size_t total = static_cast<std::size_t>(cb) * nrhs * n;
    constexpr std::size_t kWg = 256;
    ctx->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<SytrsScatterKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>((total + kWg - 1) / kWg * kWg), sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) {
                const std::size_t g = it.get_global_id(0);
                if (g >= total) return;
                const int t = static_cast<int>(g % n);
                const std::size_t rest = g / n;
                const std::ptrdiff_t i0 = static_cast<std::ptrdiff_t>(rest / nrhs);
                const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(rest % nrhs);
                if (mask != nullptr && mask[i0] == 0) return;
                if (skip != nullptr && skip[i0] != 0) return;
                const int lt = rev ? n - 1 - t : t;
                const int s = rev ? n - 1 - pw[i0 * n + lt] : pw[i0 * n + lt];
                bp[i0 * strb + c * ldb + s] = bw[g];
            });
    });
}

}  // namespace

// sytrs.hh's can_run(Cta) mirrors these checks; keep the two in step.
template <typename T>
Event sytrs_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                         const int32_t* skip_info) {
    check_solve_args(A, B, uplo, pivots, "sytrs_cta");
    check_device(ctx, "sytrs_cta");
    const int n = static_cast<int>(A.rows());
    if (n > kSytrsCtaMaxN)
        throw batchlas::unsupported("sytrs_cta: order " + std::to_string(n) + " is above the register tile's 128");
    const auto dev = ctx.device();
    const std::size_t slm = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    if (sytrs_cta_slm_bytes<T>(n) + 4096 > slm)
        throw batchlas::unsupported("sytrs_cta: order " + std::to_string(n) + " does not fit local memory");
    const int max_sg = std::max(1, std::min<int>(kSytrsCtaMaxSg,
        static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) / 32));
    const int nsg = std::min<int>(max_sg, static_cast<int>(B.cols()));
    const bool rev = uplo == Uplo::Upper;
    const int* const piv = pivots.as_span<int>().data();
    const std::uint8_t* const mask = detail::mask_ptr(detail::call_mask("sytrs_cta", A, B));
    if (n <= 32) return cta_launch<T, 1>(ctx, A, B, rev, piv, skip_info, mask, nsg);
    if (n <= 64) return cta_launch<T, 2>(ctx, A, B, rev, piv, skip_info, mask, nsg);
    return cta_launch<T, 4>(ctx, A, B, rev, piv, skip_info, mask, nsg);
}

template <typename T>
std::size_t sytrs_blocked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B) {
    const int n = static_cast<int>(A.rows()), nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());
    if (n < 1 || nrhs < 1 || batch < 1) return 0;
    const int part = blocked_part(blocked_chunk<T>(ctx, n, nrhs, batch), batch);
    return workspace_bytes([&](BumpAllocator& p) { return blocked_layout<T>(ctx, p, n, nrhs, part); });
}

// sytrs.hh's can_run(Blocked) mirrors these checks; keep the two in step.
template <typename T>
Event sytrs_blocked_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                             Span<std::byte> workspace, const int32_t* skip_info, SytrsSolveTrsm<T> solve_trsm) {
    check_solve_args(A, B, uplo, pivots, "sytrs_blocked");
    check_device(ctx, "sytrs_blocked");
    if (!solve_trsm)
        throw batchlas::invalid_argument("sytrs_blocked: the solve seam is empty; inject the public trsm");
    const int n = static_cast<int>(A.rows()), nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());
    const bool rev = uplo == Uplo::Upper;
    const int max_wg = static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int wg = std::max(32, std::min({kSytrfWgMax, (max_wg / 32) * 32, ((n + 31) / 32) * 32}));
    const int chunk = blocked_chunk<T>(ctx, n, nrhs, batch);
    BumpAllocator pool(workspace);
    const auto ws = blocked_layout<T>(ctx, pool, n, nrhs, blocked_part(chunk, batch));
    const int* const piv = pivots.as_span<int>().data();
    auto sync = [&] { if (!ctx.in_order()) ctx.wait(); };
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const Span<const std::uint8_t> mask = detail::call_mask("sytrs_blocked", A, B);
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * chunk ? batch - b0 : chunk;
        const T* const a = A.data_ptr() + static_cast<std::ptrdiff_t>(b0) * A.stride();
        T* const bb = B.data_ptr() + static_cast<std::ptrdiff_t>(b0) * B.stride();
        const int32_t* const skip = skip_info ? skip_info + b0 : nullptr;
        // The drop hazard: the kernels and the trsm views are rebuilt per part, so each takes its slice.
        const Span<const std::uint8_t> part_mask = mask.empty() ? mask : mask.subspan(b0, cb);
        const std::uint8_t* const pm = detail::mask_ptr(part_mask);
        prep_launch<T>(ctx, a, A.ld(), A.stride(), bb, B.ld(), B.stride(), piv + static_cast<std::ptrdiff_t>(b0) * n, n,
                       nrhs, cb, rev, wg, ws, skip, pm);
        sync();
        const MV Lv = MV(ws.lw.data(), n, n, n, n * n, cb, ws.pl.data()).with_batch_mask(part_mask);
        const MV Bv = MV(ws.bw.data(), n, nrhs, n, n * nrhs, cb, ws.pb.data()).with_batch_mask(part_mask);
        (void)solve_trsm(ctx, Lv, Bv, T(1), Side::Left, uplo, Transpose::NoTrans, Diag::Unit);
        sync();
        diag_launch<T>(ctx, a, A.ld(), A.stride(), n, nrhs, cb, rev, ws, skip, pm);
        sync();
        (void)solve_trsm(ctx, Lv, Bv, T(1), Side::Left, uplo, Transpose::Trans, Diag::Unit);
        sync();
        scatter_launch<T>(ctx, bb, B.ld(), B.stride(), n, nrhs, cb, rev, ws, skip, pm);
        sync();
    }
    return ctx.get_event();
}

#define BATCHLAS_SYTRS_INSTANTIATE(T)                                                                             \
    template Event sytrs_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                       \
                                         const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>,          \
                                         const int32_t*);                                                         \
    template std::size_t sytrs_blocked_buffer_size<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,          \
                                                      const MatrixView<T, MatrixFormat::Dense>&);                 \
    template Event sytrs_blocked_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                   \
                                             const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>,      \
                                             Span<std::byte>, const int32_t*, SytrsSolveTrsm<T>);

BATCHLAS_SYTRS_INSTANTIATE(float)
BATCHLAS_SYTRS_INSTANTIATE(double)
BATCHLAS_SYTRS_INSTANTIATE(std::complex<float>)
BATCHLAS_SYTRS_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SYTRS_INSTANTIATE

}  // namespace sycl_sytrf
}  // namespace batchlas
