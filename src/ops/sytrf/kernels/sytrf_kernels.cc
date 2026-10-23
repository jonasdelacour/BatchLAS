// Native batched Bunch-Kaufman LDL^T: the CTA tier (one sub-group per matrix, the logical lower
// triangle resident in local memory, n <= 32; with B it is sysv's fused factor-and-solve), the
// work-group tier (one work-group per matrix, unblocked ?sytf2 in place in global memory, any n),
// and sytrs (one work-group per right-hand side). All three run sytrf_device.hh's one algorithm.
// evidence: docs/perf/sytrf.md
//
// Belongs in EXTENSIONS_FACTORIZATION_SOURCES: it shares no device symbol with any other cluster.

#include <src/ops/sytrf/sytrf_native.hh>
#include <src/ops/sytrf/sytrf_device.hh>

#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/error.hh>
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

namespace {

using sytrf_device::Group;
using sytrf_device::Tile;
using sytrf_device::Vec;

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

// One launch for sytrf's `cta` and sysv's `cta`: B is null for the factor alone.
template <typename T>
Event cta_launch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, int* piv, int32_t* info,
                 const MatrixView<T, MatrixFormat::Dense>* B) {
    using D = Dev<T>;
    using R = Real<T>;
    constexpr int kLdt = kSytrfCtaMaxN + 1;
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    const bool upper = uplo == Uplo::Upper;
    D* const ap = reinterpret_cast<D*>(A.data_ptr());
    const std::ptrdiff_t lda = A.ld(), stra = A.stride();
    D* const bp = B ? reinterpret_cast<D*>(B->data_ptr()) : nullptr;
    const std::ptrdiff_t ldb = B ? B->ld() : 0, strb = B ? B->stride() : 0;
    const int nrhs = B ? static_cast<int>(B->cols()) : 0;
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(kLdt * kSytrfCtaMaxN + kSytrfCtaMaxN), h);
        sycl::local_accessor<int, 1> slm_piv(sycl::range<1>(kSytrfCtaMaxN), h);
        sycl::local_accessor<R, 1> red(sycl::range<1>(2), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(1), h);
        batchlas::impl::parallel_for<SytrfCtaKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * kSytrfCtaWg),
                              sycl::range<1>(kSytrfCtaWg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                D* const tp = &slm[0];
                int* const pp = &slm_piv[0];
                const Group<R> g{it, &red[0], &red[1], &redi[0], lid, kSytrfCtaWg};
                const Tile<D> G{ap + b * stra, lda, n, upper};
                const Tile<D> S{tp, kLdt, n, false};
                for (int idx = lid; idx < n * n; idx += kSytrfCtaWg) {
                    const int i = idx % n, j = idx / n;
                    if (i >= j) S(i, j) = G(i, j);
                }
                g.bar();
                int* const gp = piv + b * n;
                const int st = sytrf_device::ldl_factor(S, n, upper, g, [&](int k, int v) {
                    gp[k] = v;
                    pp[k] = v;
                });
                for (int idx = lid; idx < n * n; idx += kSytrfCtaWg) {
                    const int i = idx % n, j = idx / n;
                    if (i >= j) G(i, j) = S(i, j);
                }
                if (lid == 0) info[b] = st;
                if (bp == nullptr || st != 0) return;
                D* const bv = tp + kLdt * kSytrfCtaMaxN;
                for (int c = 0; c < nrhs; ++c) {
                    D* const col = bp + b * strb + static_cast<std::ptrdiff_t>(c) * ldb;
                    for (int i = lid; i < n; i += kSytrfCtaWg) bv[i] = col[i];
                    g.bar();
                    sytrf_device::ldl_solve(S, n, upper, Vec<D>{bv, n, upper}, g, [&](int k) { return pp[k]; });
                    for (int i = lid; i < n; i += kSytrfCtaWg) col[i] = bv[i];
                    g.bar();
                }
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
    return cta_launch<T>(ctx, A, uplo, pivots.as_span<int>().data(), info.data(), nullptr);
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
    return cta_launch<T>(ctx, A, uplo, pivots.as_span<int>().data(), info.data(), &B);
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
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> red(sycl::range<1>(2 * (wg / 32)), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(wg / 32), h);
        batchlas::impl::parallel_for<SytrfWgKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group(0));
                const Group<R> g{it, &red[0], &red[wg / 32], &redi[0], lid, wg};
                const Tile<D> G{ap + b * stra, lda, n, upper};
                int* const gp = piv + b * n;
                const int st = sytrf_device::ldl_factor(G, n, upper, g, [&](int k, int v) { gp[k] = v; });
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
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> red(sycl::range<1>(2 * (wg / 32)), h);
        sycl::local_accessor<int, 1> redi(sycl::range<1>(wg / 32), h);
        batchlas::impl::parallel_for<SytrsKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * nrhs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                const int lid = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t grp = static_cast<std::ptrdiff_t>(it.get_group(0));
                const std::ptrdiff_t b = grp / nrhs, c = grp % nrhs;
                if (skip_info != nullptr && skip_info[b] != 0) return;
                const Group<R> g{it, &red[0], &red[wg / 32], &redi[0], lid, wg};
                const Tile<D> G{ap + b * stra, lda, n, upper};
                const Vec<D> x{bp + b * strb + c * ldb, n, upper};
                const int* const gp = piv + b * n;
                sytrf_device::ldl_solve(G, n, upper, x, g, [&](int k) { return gp[k]; });
            });
    });
    return ctx.get_event();
}

#define BATCHLAS_SYTRF_NATIVE_INSTANTIATE(T)                                                                     \
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
