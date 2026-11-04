// Native batched Bunch-Kaufman LDL^T: the CTA tier (one sub-group per matrix, the logical lower
// triangle resident in local memory, n <= 32; with B it is sysv's fused factor-and-solve), the
// work-group tier (one work-group per matrix, unblocked ?sytf2 in place in global memory, any n),
// and sytrs `wg` (one work-group per right-hand side). All run the public device routines of
// batchlas/device/lapack/sytrf.hh and sytrs.hh; sytrs `cta` and `blocked` are sytrs_kernels.cc.
// evidence: docs/perf/sytrf.md
//
// Belongs in EXTENSIONS_FACTORIZATION_SOURCES: it shares no device symbol with any other cluster.

#include <src/ops/sytrf/sytrf_native.hh>
#include <batchlas/device/lapack/sytrf.hh>
#include <batchlas/device/lapack/sytrs.hh>

#include <src/kernels/common/info_span.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
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

template <typename T> class SytrfCtaKernel;
template <typename T> class SytrfWgKernel;
template <typename T> class SytrsKernel;
template <typename T> class SytrfBlockedStepKernel;
template <typename T> class SytrfBlockedTrailKernel;

namespace {

namespace v2 = batchlas::device;
template <typename D> using Tile = v2::sytrf_detail::RevMat<D>;
template <typename D> using Vec = v2::sytrf_detail::RevVec<D>;
using v2::sytrf_detail::ArgMaxScratch;
using v2::sytrf_detail::rev_mat;

template <typename T>
using Dev = typename sycl_device::DevMap<T>::type;
template <typename T>
using Real = typename sycl_device::DevMap<T>::real;

void check_device(Queue& ctx, const char* who) {
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) throw batchlas::invalid_argument(std::string(who) + ": GPU queues only");
    // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports entry [0].
    if (!dev.supports_sub_group_size(32))
        throw batchlas::unsupported(std::string(who) + ": device does not offer sub-group size 32");
}

// A multiple of 32 no wider than the device takes or the order needs.
int group_width(Queue& ctx, int n) {
    const int max_wg = static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cap = std::min(kSytrfWgMax, (max_wg / 32) * 32);
    return std::max(32, std::min(cap, ((n + 31) / 32) * 32));
}

template <typename T>
void check_factor_args(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots, const char* who) {
    const auto n = A.rows();
    if (n < 1 || A.cols() != n || A.batch_size() < 1)
        throw batchlas::invalid_argument(std::string(who) + ": A must be square with n >= 1 and batch >= 1");
    if (A.is_heterogeneous()) throw batchlas::invalid_argument(std::string(who) + ": heterogeneous batch is not supported");
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) throw batchlas::invalid_argument(std::string(who) + ": invalid uplo");
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(A.batch_size()))
        throw batchlas::invalid_argument(std::string(who) + ": pivot span is shorter than n * batch");
}

template <typename T>
Span<int32_t> info_layout(Queue& ctx, BumpAllocator& pool, Span<int32_t> info_out, int batch) {
    if (info_out.size() >= static_cast<std::size_t>(batch)) return info_out;
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

// One launch for sytrf's `cta` and sysv's `cta`: B is null for the factor alone. Sub-group 0
// factors (sub-group collectives only, so the factor is the same with or without B); sysv's
// work-group has one sub-group per right-hand side (at most kSytrsCtaMaxSg), which then solve in
// parallel with sytrs `cta`'s converted solve (v2::sytrs_cols).
template <typename T>
Event cta_launch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, int* piv, int32_t* info,
                 const MatrixView<T, MatrixFormat::Dense>* B, const std::uint8_t* mask) {
    using D = Dev<T>;
    using R = Real<T>;
    constexpr int kLdt = kSytrfCtaMaxN + 1;
    constexpr int kN = kSytrfCtaMaxN;
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const bool upper = uplo == Uplo::Upper;
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stra = A.stride();
    D* const bp = B ? reinterpret_cast<D*>(B->data_ptr()) : nullptr;
    const std::ptrdiff_t ldb = B ? B->ld() : 0, strb = B ? B->stride() : 0;
    const int nrhs = B ? static_cast<int>(B->cols()) : 0;
    const int max_sg = std::max(1, std::min<int>(kSytrsCtaMaxSg,
        static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) / 32));
    const int nsg = B ? std::min(max_sg, nrhs) : 1;
    const int wg = 32 * nsg;
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(kLdt * kN + kN), h);
        sycl::local_accessor<int, 1> slm_piv(sycl::range<1>(4 * kN + 2), h);
        sycl::local_accessor<R, 1> red(sycl::range<1>(2), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(1), h);
        batchlas::impl::parallel_for<SytrfCtaKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                if (mask != nullptr && mask[b] == 0) return;  // I1: before the first barrier
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());
                D* const tp = &slm[0];
                int* const pp = &slm_piv[0];
                const v2::RtGroup<> all(it);
                const v2::SubGroup<kSytrfCtaWg> g{sg};  // sub-group 0 factors
                const Tile<D> G = rev_mat(ap + b * stra, lda, n, upper);
                const Tile<D> S = rev_mat(tp, kLdt, n, false);
                v2::lacpy<v2::Part::Lower, Transpose::NoTrans, v2::alg::DivMod>(all, G, S);
                v2::sync<v2::scope::Group>(all, v2::Mem::Global);
                if (sgid == 0) {
                    int* const gp = piv + b * n;
                    const int st = v2::sytf2(S, n, upper, g, ArgMaxScratch<R>{&red[0], &redi[0]}, [&](int k, int v) {
                        gp[k] = v;
                        pp[k] = v;
                    });
                    v2::lacpy<v2::Part::Lower, Transpose::NoTrans, v2::alg::DivMod>(g, S, G);
                    if (lane == 0) {
                        info[b] = st;
                        pp[4 * kN + 1] = st;
                    }
                }
                if (bp == nullptr) return;
                v2::sync<v2::scope::Group>(all, v2::Mem::Global);
                if (pp[4 * kN + 1] != 0) return;
                const v2::sytrs_detail::Staged<D> L{tp, kLdt, n, tp + kLdt * kN, pp + kN, pp + 2 * kN, pp + 3 * kN};
                v2::for_each_owned(all, n, [&](int k) BATCHLAS_DLA_BODY {
                    L.ps[k] = v2::sytrs_detail::logical_piv(pp[upper ? n - 1 - k : k], n, upper);
                });
                v2::sync<v2::scope::Group>(all, v2::Mem::Global);
                const bool ok = v2::syconv<D>(L, n, L.ps, L.perm, L.kind, L.e, all, pp + 4 * kN);
                v2::sytrs_cols<1>(sg, lane, sgid, nsg, L, ok, upper, bp + b * strb, ldb, nrhs);
            });
    });
    return ctx.get_event();
}

}  // namespace

template <typename T>
std::size_t sytrf_native_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& p) { return info_layout<T>(ctx, p, Span<int32_t>{}, batch); });
}

// sytrf.cc's can_run(Cta) mirrors these checks; keep the two in step.
template <typename T>
Event sytrf_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                         Span<std::byte> workspace, Span<int32_t> info_out) {
    check_factor_args(A, uplo, pivots, "sytrf_cta");
    if (A.rows() > kSytrfCtaMaxN)
        throw batchlas::unsupported("sytrf_cta: order " + std::to_string(A.rows()) + " is above the sub-group width 32");
    check_device(ctx, "sytrf_cta");
    BumpAllocator pool(workspace);
    auto info = info_layout<T>(ctx, pool, info_out, static_cast<int>(A.batch_size()));
    return cta_launch<T>(ctx, A, uplo, pivots.as_span<int>().data(), info.data(), nullptr,
                         detail::mask_ptr(detail::call_mask("sytrf_cta", A)));
}

template <typename T>
Event sysv_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                        Span<std::byte> workspace, Span<int32_t> info_out) {
    check_factor_args(A, uplo, pivots, "sysv_cta");
    if (A.rows() > kSytrfCtaMaxN)
        throw batchlas::unsupported("sysv_cta: order " + std::to_string(A.rows()) + " is above the sub-group width 32");
    if (B.rows() != A.rows() || B.batch_size() != A.batch_size() || B.cols() < 1 || B.is_heterogeneous())
        throw batchlas::invalid_argument("sysv_cta: B must be a homogeneous n x nrhs batch, nrhs >= 1, matching A");
    check_device(ctx, "sysv_cta");
    BumpAllocator pool(workspace);
    auto info = info_layout<T>(ctx, pool, info_out, static_cast<int>(A.batch_size()));
    return cta_launch<T>(ctx, A, uplo, pivots.as_span<int>().data(), info.data(), &B,
                         detail::mask_ptr(detail::call_mask("sysv_cta", A, B)));
}

template <typename T>
Event sytrf_wg_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                        Span<std::byte> workspace, Span<int32_t> info_out) {
    using D = Dev<T>;
    using R = Real<T>;
    check_factor_args(A, uplo, pivots, "sytrf_wg");
    check_device(ctx, "sytrf_wg");
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const int wg = group_width(ctx, n);
    BumpAllocator pool(workspace);
    int32_t* const info = info_layout<T>(ctx, pool, info_out, batch).data();
    int* const piv = pivots.as_span<int>().data();
    const bool upper = uplo == Uplo::Upper;
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stra = A.stride();
    const std::uint8_t* const mask = detail::mask_ptr(detail::call_mask("sytrf_wg", A));
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> red(sycl::range<1>(2 * (wg / 32)), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(wg / 32), h);
        batchlas::impl::parallel_for<SytrfWgKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                if (mask != nullptr && mask[b] == 0) return;
                const v2::RtGroup<> g(it);
                const Tile<D> G = rev_mat(ap + b * stra, lda, n, upper);
                int* const gp = piv + b * n;
                const int st = v2::sytf2(G, n, upper, g, ArgMaxScratch<R>{&red[0], &redi[0]},
                                                        [&](int k, int v) { gp[k] = v; });
                if (lid == 0) info[b] = st;
            });
    });
    return ctx.get_event();
}

template <typename T>
Event sytrs_native_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                            const int32_t* skip_info) {
    using D = Dev<T>;
    using R = Real<T>;
    check_factor_args(A, uplo, pivots, "sytrs");
    if (B.rows() != A.rows() || B.batch_size() != A.batch_size() || B.is_heterogeneous())
        throw batchlas::invalid_argument("sytrs: B must be a homogeneous n x nrhs batch matching A");
    check_device(ctx, "sytrs");
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const int nrhs = static_cast<int>(B.cols());
    if (nrhs < 1) return ctx.create_event_after_external_work();
    const int wg = group_width(ctx, n);
    const int* const piv = pivots.as_span<int>().data();
    const bool upper = uplo == Uplo::Upper;
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    D* const bp = reinterpret_cast<D*>(B.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stra = A.stride(), ldb = B.ld(), strb = B.stride();
    const std::uint8_t* const mask = detail::mask_ptr(detail::call_mask("sytrs", A, B));
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> red(sycl::range<1>(2 * (wg / 32)), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(wg / 32), h);
        batchlas::impl::parallel_for<SytrsKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * nrhs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const std::ptrdiff_t grp = static_cast<std::ptrdiff_t>(it.get_group(0));
                const std::ptrdiff_t b = grp / nrhs, c = grp % nrhs;
                if (mask != nullptr && mask[b] == 0) return;
                if (skip_info != nullptr && skip_info[b] != 0) return;
                const v2::RtGroup<> g(it);
                const Tile<D> G = rev_mat(ap + b * stra, lda, n, upper);
                const Vec<D> x{bp + b * strb + c * ldb, n, upper};
                const int* const gp = piv + b * n;
                v2::sytrs_col(G, n, upper, x, g, v2::sytrf_detail::SumScratch<R>{&red[0], &red[wg / 32]},
                                        [&](int k) { return gp[k]; });
            });
    });
    return ctx.get_event();
}

namespace {

// The blocked tier's scratch, one part's worth: W is n x (nb + 1) per item; `state` is each item's
// {first, end} of its last panel, which the next step launch undoes and continues from.
template <typename T>
struct BlockedWs {
    Span<int32_t> info;
    Span<int> state;
    Span<T> w;
    Span<T*> pl, pw, pc;  // one pointer array per gemm role (init_data_ptr_array rebases per view)
};

template <typename T>
BlockedWs<T> blocked_layout(Queue& ctx, BumpAllocator& pool, int n, int part) {
    const auto p = static_cast<std::size_t>(part);
    BlockedWs<T> ws;
    ws.info = pool.allocate<int32_t>(ctx, p);
    ws.state = pool.allocate<int>(ctx, 2 * p);
    ws.w = pool.allocate<T>(ctx, p * static_cast<std::size_t>(n) * (kSytrfBlockedNb + 1));
    ws.pl = pool.allocate<T*>(ctx, p);
    ws.pw = pool.allocate<T*>(ctx, p);
    ws.pc = pool.allocate<T*>(ctx, p);
    return ws;
}

// getrf's and potrf's L2 rule (untimed for sytrf); the query and the call must agree.
template <typename T>
int blocked_chunk(Queue& ctx, int n, int batch) {
    if (n <= kSytrfBlockedNb) return batch;
    const auto dev = ctx.device();
    return static_cast<int>(tuning::blocked_chunk_items(
        batch, static_cast<int64_t>(n) * n * static_cast<int64_t>(sizeof(T)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS))));
}

// One launch per panel: the previous panel's ldl_undo, then this panel's ldl_panel, or for the tail
// ?sytf2 on the trailing sub-matrix from the item's own first unfinished column.
template <typename T>
void blocked_step(Queue& ctx, T* a, int lda, int stra, T* w, int n, int cb, bool upper, int wg, int k0, bool tail,
                  int* state, int* piv, int32_t* info, const std::uint8_t* mask) {
    using D = Dev<T>;
    using R = Real<T>;
    constexpr int nb = kSytrfBlockedNb;
    D* const ap = reinterpret_cast<D*>(a);
    D* const wp = reinterpret_cast<D*>(w);
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> red(sycl::range<1>(2 * (wg / 32)), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(wg / 32), h);
        batchlas::impl::parallel_for<SytrfBlockedStepKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(cb) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                if (mask != nullptr && mask[b] == 0) return;  // I1: before the first barrier
                const v2::RtGroup<> g(it);
                const ArgMaxScratch<R> am{&red[0], &redi[0]};
                D* const item = ap + b * stra;
                const Tile<D> S = rev_mat(item, lda, n, upper);
                int* const gp = piv + b * n;
                const int first = state[2 * b], end = state[2 * b + 1];
                int st = info[b];
                v2::sync<v2::scope::Group>(g, v2::Mem::Global);
                v2::lasyf_undo(S, first, end, g, [&](int j, bool& two) {
                    const int raw = gp[upper ? n - 1 - j : j];
                    two = raw < 0;
                    const int p = (raw < 0 ? -raw : raw) - 1;
                    return upper ? n - 1 - p : p;
                });
                if (tail) {
                    const int sub = n - end;
                    const int off = upper ? 0 : end;
                    const Tile<D> T2 = rev_mat(item + off + static_cast<std::ptrdiff_t>(off) * lda, lda, sub, upper);
                    const int t = sub > 0 ? v2::sytf2(T2, sub, upper, g, am, [&](int k, int v) {
                        gp[k + off] = v > 0 ? v + off : v - off;
                    }) : 0;
                    if (st == 0 && t > 0) st = t + off;
                } else {
                    const Tile<D> W = v2::sytrf_detail::w_tile(wp + b * static_cast<std::ptrdiff_t>(n) * (nb + 1), n, n, nb + 1, upper);
                    const int e = v2::lasyf(S, W, n, k0, nb, end, upper, g, am, st,
                                                          [&](int k, int v) { gp[k] = v; });
                    if (lid == 0) {
                        state[2 * b] = end;
                        state[2 * b + 1] = e;
                    }
                }
                if (lid == 0) info[b] = st;
            });
    });
}

// The trailing update's native part: the lower triangle of each ub-wide diagonal block of
// [kt, n), and column k0 + nb of an item whose panel ended at k0 + nb (it is still trailing). The
// rank is the item's own (nb, or nb + 1 after a 2 x 2 block took column k0 + nb).
template <typename T>
void blocked_trail(Queue& ctx, T* a, int lda, int stra, T* w, int n, int cb, bool upper, int wg, int k0,
                   const int* state, const std::uint8_t* mask) {
    using D = Dev<T>;
    using A = v2::arith_t<v2::arith::Fma, D>;
    constexpr int nb = kSytrfBlockedNb, ub = kSytrfBlockedUb;
    const int kt = k0 + nb + 1;
    const int nblk = (n - kt + ub - 1) / ub + 1;
    D* const ap = reinterpret_cast<D*>(a);
    D* const wp = reinterpret_cast<D*>(w);
    ctx->submit([&](sycl::handler& h) {
        batchlas::impl::parallel_for<SytrfBlockedTrailKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(cb) * nblk * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const std::ptrdiff_t grp = static_cast<std::ptrdiff_t>(it.get_group(0));
                const std::ptrdiff_t b = grp / nblk;
                if (mask != nullptr && mask[b] == 0) return;
                const int t = static_cast<int>(grp % nblk);
                const v2::RtGroup<> ex(it);
                const Tile<D> S = rev_mat(ap + b * stra, lda, n, upper);
                const Tile<D> W = v2::sytrf_detail::w_tile(wp + b * static_cast<std::ptrdiff_t>(n) * (nb + 1), n, n, nb + 1, upper);
                const int rank = state[2 * b + 1] - k0;
                // C(i, j) -= sum_l L(i, l) W(j, l), one term at a time from C (?gemmt, alpha -1).
                auto upd = [&](int i, int j) {
                    D acc = S(i, j);
                    for (int l = 0; l < rank; ++l) acc = A::sub(acc, A::mul(S(i, k0 + l), W(j, l)));
                    S(i, j) = acc;
                };
                if (t == nblk - 1) {
                    if (rank == nb) v2::for_each_owned(ex, v2::Range{kt - 1, n}, [&](int i) BATCHLAS_DLA_BODY { upd(i, kt - 1); });
                    return;
                }
                const int j0 = kt + t * ub, jb = sycl::min(ub, n - j0);
                v2::for_each_owned_2d<v2::alg::DivMod>(ex, jb, jb, [&](int r, int c) BATCHLAS_DLA_BODY {
                    const int i = j0 + r, j = j0 + c;
                    if (i >= j) upd(i, j);
                });
            });
    });
}

}  // namespace

template <typename T>
std::size_t sytrf_blocked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = static_cast<int>(A.batch_size());
    const int n = static_cast<int>(A.rows());
    if (batch < 1 || n < 1) return 0;
    const int chunk = blocked_chunk<T>(ctx, n, batch);
    return workspace_bytes([&](BumpAllocator& p) { return blocked_layout<T>(ctx, p, n, chunk + batch % chunk); });
}

// sytrf.cc's can_run(Blocked) mirrors these checks; keep the two in step.
template <typename T>
Event sytrf_blocked_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                             Span<std::byte> workspace, Span<int32_t> info_out, SytrfTrailingGemm<T> trailing_gemm) {
    check_factor_args(A, uplo, pivots, "sytrf_blocked");
    check_device(ctx, "sytrf_blocked");
    if (!trailing_gemm)
        throw batchlas::invalid_argument("sytrf_blocked: the trailing-update gemm seam is empty; inject the public gemm");
    constexpr int nb = kSytrfBlockedNb, ub = kSytrfBlockedUb, r = nb + 1;
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const int wg = group_width(ctx, n);
    const bool upper = uplo == Uplo::Upper;
    const int lda = A.ld(), stra = A.stride();
    const int chunk = blocked_chunk<T>(ctx, n, batch);
    BumpAllocator pool(workspace);
    auto ws = blocked_layout<T>(ctx, pool, n, chunk + batch % chunk);
    const bool info_requested = info_out.size() >= static_cast<std::size_t>(batch);
    const Span<const std::uint8_t> mask = detail::call_mask("sytrf_blocked", A);
    auto sync = [&] { if (!ctx.in_order()) ctx.wait(); };
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * chunk ? batch - b0 : chunk;
        T* const a = A.data_ptr() + static_cast<std::ptrdiff_t>(b0) * stra;
        int* const piv = pivots.as_span<int>().data() + static_cast<std::ptrdiff_t>(b0) * n;
        int32_t* const info = info_requested ? info_out.data() + b0 : ws.info.data();
        // The drop hazard: every view and pointer below is rebuilt from data_ptr, so each part
        // carries its own slice of the mask.
        const Span<const std::uint8_t> part_mask = mask.empty() ? mask : mask.subspan(b0, cb);
        const std::uint8_t* const pm = detail::mask_ptr(part_mask);
        detail::info_clear(ctx, Span<int32_t>(info, static_cast<std::size_t>(cb)), cb, part_mask);
        ctx->fill(ws.state.data(), 0, 2 * static_cast<std::size_t>(cb));
        sync();
        int k0 = 0;
        for (; n - k0 > nb; k0 += nb) {
            blocked_step<T>(ctx, a, lda, stra, ws.w.data(), n, cb, upper, wg, k0, false, ws.state.data(), piv, info, pm);
            sync();
            blocked_trail<T>(ctx, a, lda, stra, ws.w.data(), n, cb, upper, wg, k0, ws.state.data(), pm);
            sync();
            // A22 -= L21 W21^T below each diagonal block, in physical coordinates (Upper: above).
            const int kt = k0 + r;
            for (int j0 = kt; j0 < n; j0 += ub) {
                const int jb = std::min(ub, n - j0), r0 = j0 + jb, m2 = n - r0;
                if (m2 <= 0) break;
                const int crow = upper ? 0 : r0;
                const int ccol = upper ? n - j0 - jb : j0;
                const int lcol = upper ? n - k0 - r : k0;
                const int wrow = upper ? n - j0 - jb : j0;
                using MV = MatrixView<T, MatrixFormat::Dense>;
                const MV L = MV(a + crow + static_cast<std::ptrdiff_t>(lcol) * lda, m2, r, lda, stra, cb, ws.pl.data())
                                 .with_batch_mask(part_mask);
                const MV Wv(ws.w.data() + wrow, jb, r, n, n * r, cb, ws.pw.data());  // scratch: adopts the mask
                const MV C = MV(a + crow + static_cast<std::ptrdiff_t>(ccol) * lda, m2, jb, lda, stra, cb, ws.pc.data())
                                 .with_batch_mask(part_mask);
                (void)trailing_gemm(ctx, L, Wv, C, T(-1), T(1), Transpose::NoTrans, Transpose::Trans,
                                    ComputePrecision::Default);
                sync();
            }
        }
        blocked_step<T>(ctx, a, lda, stra, ws.w.data(), n, cb, upper, wg, k0, true, ws.state.data(), piv, info, pm);
        sync();
    }
    return ctx.get_event();
}

#define BATCHLAS_SYTRF_NATIVE_INSTANTIATE(T)                                                                     \
    template std::size_t sytrf_blocked_buffer_size<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&);        \
    template Event sytrf_blocked_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo,            \
                                             Span<int64_t>, Span<std::byte>, Span<int32_t>, SytrfTrailingGemm<T>); \
    template std::size_t sytrf_native_buffer_size<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&);         \
    template Event sytrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>, \
                                         Span<std::byte>, Span<int32_t>);                                        \
    template Event sytrf_wg_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>,  \
                                        Span<std::byte>, Span<int32_t>);                                         \
    template Event sytrs_native_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                   \
                                            const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>,      \
                                            const int32_t*);                                                     \
    template Event sysv_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                       \
                                        const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>,          \
                                        Span<std::byte>, Span<int32_t>);

BATCHLAS_SYTRF_NATIVE_INSTANTIATE(float)
BATCHLAS_SYTRF_NATIVE_INSTANTIATE(double)
BATCHLAS_SYTRF_NATIVE_INSTANTIATE(std::complex<float>)
BATCHLAS_SYTRF_NATIVE_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SYTRF_NATIVE_INSTANTIATE

}  // namespace sycl_sytrf
}  // namespace batchlas
