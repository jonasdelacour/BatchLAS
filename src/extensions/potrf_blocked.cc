// Native batched POTRF: the blocked right-looking driver, Uplo::Lower only. Per panel j:
// the leaf potrf on the ib x ib diagonal block, an info-merge/quench fixup, the panel solve
// L21 = A21 L11^{-H}, and A22 -= L21 L21^H. See docs/perf/potrf.md#the-blocked-driver

#include "potrf_launch_plan.hh"
#include "potrf_native.hh"
#include "symmetric_product_fold.hh"

#include "../sycl/gemm_kernels.hh"
#include "../sycl/trsm_native.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <batchlas/settings.hh>

namespace batchlas {
namespace sycl_potrf {

namespace {

template <typename T> struct PotrfIsComplex : std::false_type {};
template <> struct PotrfIsComplex<std::complex<float>>  : std::true_type {};
template <> struct PotrfIsComplex<std::complex<double>> : std::true_type {};

// Complex: A22 -= L21 L21^H is Hermitian, so ConjTrans here is correctness, not tuning.
template <typename T>
constexpr Transpose kTrailingTransB =
    PotrfIsComplex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

// Blocking overrides only, never routing; read once so the sizing query and the
// call agree. The function-local statics are kept for exactly that reason: the
// settings() snapshot is re-readable, and a reload landing between
// potrf_buffer_size() and the matching potrf would desynchronise the allocated
// workspace from the block width actually used. 0 means "unset" on both fields,
// which is what potrf_plan::BlockedConst falls back to.
inline int potrf_nb_env() { static const int v = batchlas::settings().geometry.potrf_nb; return v; }
inline int potrf_w_env()  { static const int v = batchlas::settings().geometry.potrf_w;  return v; }

// nb, W and the occupancy target live in potrf_launch_plan.hh (blocked_params), as does the
// panel schedule (blocked_schedule): the driver below and the cost model walk the same one.
using PotrfBlockedParams = potrf_plan::BlockedParams;

template <typename T>
PotrfBlockedParams potrf_blocked_params(Queue& ctx, int n) {
    // From THIS device's SLM: the hardcoded potrf_cta_max_n<T>() can name a block the leaf
    // refuses. The leaf is launched at the same occupancy target, so the two cannot disagree.
    return potrf_plan::blocked_params<T>(
        n, potrf_device_facts(ctx.device()).local_mem_bytes, sycl_trsm::trsm_cta_max_n<T>(),
        potrf_nb_env(), potrf_w_env());
}

template <typename T>
struct PotrfBlockedWs {
    Span<int32_t> info;       // driver-owned info fallback
    Span<int32_t> leaf_info;  // what the leaf writes, per panel, before merging
    Span<T*>      a11_ptrs;   // pointer array for the L11 role
    Span<T*>      a21_ptrs;   // pointer array for the panel role
    Span<T>       product;    // W x W x batch scratch for the diagonal-block gemm
    Span<std::byte> leaf_ws;  // handed straight to potrf_cta_dispatch
};

template <typename T>
PotrfBlockedWs<T> potrf_blocked_layout(Queue& ctx, BumpAllocator& pool,
                                       int n, int nb, int batch, int W,
                                       std::size_t leaf_bytes) {
    PotrfBlockedWs<T> ws;
    const std::size_t b = static_cast<std::size_t>(batch);

    ws.info = pool.allocate<int32_t>(ctx, b);

    // Separate from the driver's info: the leaf re-zeroes per launch, so sharing loses failures.
    ws.leaf_info = pool.allocate<int32_t>(ctx, b);

    // One pointer array PER ROLE: init_data_ptr_array recomputes from each view's base.
    ws.a11_ptrs = pool.allocate<T*>(ctx, b);
    ws.a21_ptrs = pool.allocate<T*>(ctx, b);

    // The fold has no alpha (C = product + beta*C), so the feeding gemm carries the -1.
    ws.product = (n > nb) ? pool.allocate<T>(ctx, static_cast<std::size_t>(W) *
                                                  static_cast<std::size_t>(W) * b)
                          : Span<T>{};

    // An explicit draw, never pool.remaining(): a sizing pool has no tail and throws.
    ws.leaf_ws = pool.allocate<std::byte>(ctx, leaf_bytes);

    return ws;
}

}  // namespace

template <typename T>
std::size_t potrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Uplo uplo) {
    static_cast<void>(uplo);

    const int batch = static_cast<int>(A.batch_size());
    const int n = static_cast<int>(A.rows());
    if (batch < 1 || n < 1) return 0;

    const auto p = potrf_blocked_params<T>(ctx, n);

    const std::size_t leaf_bytes = potrf_cta_buffer_size<T>(ctx, A);

    return workspace_bytes([&](BumpAllocator& pool) {
        return potrf_blocked_layout<T>(ctx, pool, n, p.nb, batch, p.W, leaf_bytes);
    });
}

// The plan's params as given: W is already clamped to n - nb, so the scratch is W_eff^2*batch.
template <typename T>
std::size_t potrf_blocked_buffer_size_planned(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& A,
                                              const potrf_plan::BlockedParams& p) {
    const int batch = static_cast<int>(A.batch_size());
    const int n = static_cast<int>(A.rows());
    if (batch < 1 || n < 1) return 0;
    const std::size_t leaf_bytes = potrf_cta_buffer_size<T>(ctx, A);
    return workspace_bytes([&](BumpAllocator& pool) {
        return potrf_blocked_layout<T>(ctx, pool, n, p.nb, batch, p.W, leaf_bytes);
    });
}

void potrf_blocked_overrides(int& nb_env, int& w_env) {
    nb_env = potrf_nb_env();
    w_env = potrf_w_env();
}

template <typename T>
unsigned potrf_blocked_debug_params(Queue& ctx, int n) {
    const auto p = potrf_blocked_params<T>(ctx, n);
    return (static_cast<unsigned>(p.W) << 16) | static_cast<unsigned>(p.nb);
}

namespace {

template <typename T> class PotrfBlockedFixupKernel;

// Runs after the leaf and BEFORE the panel solve, which would otherwise divide by the failed
// pivot. Merges the leaf's 1-based sub-view-local index as j + leaf[b], first failure wins,
// and quenches a failed item to the IDENTITY diagonal (zeroing the panel alone gives 0/0).
template <typename T>
Event potrf_blocked_panel_fixup(Queue& ctx,
                                T* a_ptr, int ld, int stride,
                                int j, int ib, int m2, int batch,
                                int32_t* info_ptr, const int32_t* leaf_ptr,
                                int wg) {
    const int rows = ib + m2;          // the whole column panel A(j:n, j:j+ib)
    const T one = T(1);
    const T zero = T(0);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PotrfBlockedFixupKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());

                // Every work-item must read info[b] before any writes it, or the merge races.
                const int32_t prev = info_ptr[b];
                const int32_t li = leaf_ptr[b];
                sycl::group_barrier(it.get_group());
                if (tid == 0 && prev == 0 && li != 0) {
                    info_ptr[b] = j + li;
                }

                const bool dead = (prev != 0) || (li != 0);
                if (!dead) return;

                T* base = a_ptr + static_cast<std::ptrdiff_t>(b) * stride +
                          static_cast<std::ptrdiff_t>(j) * ld + j;
                const std::size_t total = static_cast<std::size_t>(rows) *
                                          static_cast<std::size_t>(ib);
                for (std::size_t e = static_cast<std::size_t>(tid); e < total;
                     e += static_cast<std::size_t>(wg)) {
                    const int c = static_cast<int>(e / static_cast<std::size_t>(rows));
                    const int r = static_cast<int>(e % static_cast<std::size_t>(rows));
                    if (r < ib) {
                        if (r < c) continue;                 // upper triangle: untouched
                        base[static_cast<std::ptrdiff_t>(c) * ld + r] = (r == c) ? one : zero;
                    } else {
                        base[static_cast<std::ptrdiff_t>(c) * ld + r] = zero;
                    }
                }
            });
    });
    return ctx.get_event();
}

}  // namespace

namespace {
template <typename T>
Event potrf_blocked_run(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                        Span<std::byte> workspace, Span<int32_t> info_out,
                        PotrfTrailingGemm<T> trailing_gemm, PotrfPanelSolve<T> panel_solve,
                        const PotrfBlockedParams* planned);
}  // namespace

template <typename T>
Event potrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Uplo uplo,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             PotrfTrailingGemm<T> trailing_gemm,
                             PotrfPanelSolve<T> panel_solve) {
    return potrf_blocked_run<T>(ctx, A, uplo, workspace, info_out, std::move(trailing_gemm),
                                std::move(panel_solve), nullptr);
}

template <typename T>
Event potrf_blocked_dispatch_planned(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     Uplo uplo,
                                     Span<std::byte> workspace,
                                     Span<int32_t> info_out,
                                     const potrf_plan::BlockedParams& p,
                                     PotrfTrailingGemm<T> trailing_gemm,
                                     PotrfPanelSolve<T> panel_solve) {
    if (!trailing_gemm || !panel_solve) {
        throw batchlas::internal_error("potrf_blocked (planned): sub-op seams must be injected");
    }
    return potrf_blocked_run<T>(ctx, A, uplo, workspace, info_out, std::move(trailing_gemm),
                                std::move(panel_solve), &p);
}

namespace {
template <typename T>
Event potrf_blocked_run(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                        Span<std::byte> workspace, Span<int32_t> info_out,
                        PotrfTrailingGemm<T> trailing_gemm, PotrfPanelSolve<T> panel_solve,
                        const PotrfBlockedParams* planned) {
    // Both seams default to the NATIVE kernels; the facade injects the ROUTED ones.
    if (!trailing_gemm) {
        trailing_gemm = [](Queue& c,
                           const MatrixView<T, MatrixFormat::Dense>& ga,
                           const MatrixView<T, MatrixFormat::Dense>& gb,
                           const MatrixView<T, MatrixFormat::Dense>& gc,
                           T galpha, T gbeta, Transpose gta, Transpose gtb,
                           ComputePrecision gp) {
            return sycl_gemm::gemm_custom<T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
        };
    }
    if (!panel_solve) {
        panel_solve = [](Queue& c,
                         const MatrixView<T, MatrixFormat::Dense>& ta,
                         const MatrixView<T, MatrixFormat::Dense>& tb,
                         T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
            // V2, not V1: it degenerates to a single V1 solve when the order fits the CTA.
            return sycl_trsm::trsm_native_blocked<T>(c, ta, tb, talpha, tside, tuplo,
                                                     ttrans, tdiag);
        };
    }

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("potrf_blocked: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw batchlas::invalid_argument("potrf_blocked: degenerate extents");
    }
    if (uplo != Uplo::Lower) {
        throw batchlas::invalid_argument(
            "potrf_blocked: Uplo::Upper is not implemented; the driver factors the "
            "lower triangle only; see RouteTable<Op::potrf, T>::supports, Blocked arm)");
    }
    if (A.is_heterogeneous()) {
        throw batchlas::invalid_argument("potrf_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("potrf_blocked: GPU queues only");
    }

    const auto p = planned ? *planned : potrf_blocked_params<T>(ctx, n);
    const int nb = p.nb;
    const int W = p.W;

    const std::size_t leaf_bytes = potrf_cta_buffer_size<T>(ctx, A);
    BumpAllocator pool(workspace);
    auto ws = potrf_blocked_layout<T>(ctx, pool, n, nb, batch, W, leaf_bytes);

    // An empty OR SHORT caller span means "not requested"; zero THIS span, not info_out.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : ws.info;

    // Not optional: the merge reads info[b] for an earlier panel's failure; info_out is dirty.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));

    // Not hygiene: the gemm epilogue reads prior even at beta == 0, so arena poison gives NaN.
    // evidence: docs/perf/potrf.md#correctness-findings
    if (!ws.product.empty()) {
        ctx->fill(ws.product.data(), T(0), ws.product.size());
    }

    if (!ctx.in_order()) ctx.wait();

    const int ld = A.ld();
    const int stride = A.stride();
    T* const a_ptr = A.data_ptr();

    // The explicit 6-arg constructor, never operator()(Slice, Slice): the constructor defaults
    // stride to ld*cols when 0 is passed, so a sub-view would read the wrong matrix past item 0.
    auto sub = [&](int r0, int nr, int c0, int nc, T** ptrs) {
        return MatrixView<T, MatrixFormat::Dense>(
            a_ptr + static_cast<std::ptrdiff_t>(c0) * ld + r0,
            nr, nc, ld, stride, batch, ptrs);
    };

    T* const prod_ptr = ws.product.data();

    const int fixup_wg = std::min<int>(
        128, std::max<int>(32, static_cast<int>(
                                   dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE))));

    potrf_plan::blocked_schedule(
        n, nb, W,
        [&](int j, int ib, int m2) {
            const auto A11 = sub(j, ib, j, ib, ws.a11_ptrs.data());
            // (void) on an Event: deliberate. This Queue is in-order, so the next submission
            // is already ordered after this one and the Event carries nothing the caller needs.
            // The SAME occupancy target nb was clamped against: the leaf's gate and the block
            // width are one decision, and a mismatch makes the leaf throw on a legal block.
            (void)potrf_cta_dispatch<T>(ctx, A11, Uplo::Lower, ws.leaf_ws, ws.leaf_info,
                                        p.leaf_min_blocks);

            // Unguarded: stale leaf_info, and a solve dividing by a pivot the quench has not replaced.
            if (!ctx.in_order()) ctx.wait();

            (void)potrf_blocked_panel_fixup<T>(ctx, a_ptr, ld, stride, j, ib, m2, batch,
                                               info.data(), ws.leaf_info.data(), fixup_wg);
        },
        [&](int j, int ib, int m2) {
            if (!ctx.in_order()) ctx.wait();

            const auto A11 = sub(j, ib, j, ib, ws.a11_ptrs.data());
            const auto A21 = sub(j + ib, m2, j, ib, ws.a21_ptrs.data());
            (void)panel_solve(ctx, A11, A21, T(1), Side::Right, Uplo::Lower,
                              Transpose::ConjTrans, Diag::NonUnit);

            if (!ctx.in_order()) ctx.wait();
        },
        // A plain square gemm over A22 would write the upper triangle, which potrf(Lower) must
        // leave untouched; the W x W block is folded in instead. Only a poisoned-upper test sees
        // a lost fold.
        [&](int j, int ib, int c, int w, int mr) {
            const auto Lrow = sub(j + ib + c, w, j, ib, nullptr);
            const auto Cd = sub(j + ib + c, w, j + ib + c, w, nullptr);
            const MatrixView<T, MatrixFormat::Dense> Sc(prod_ptr, w, w, W, W * W, batch);

            (void)trailing_gemm(ctx, Lrow, Lrow, Sc, T(-1), T(0),
                                Transpose::NoTrans, kTrailingTransB<T>,
                                ComputePrecision::Default);

            // RAW: out of order the fold reads stale scratch -- a wrong factor with info == 0.
            if (!ctx.in_order()) ctx.wait();

            (void)::batchlas::detail::fold_symmetric_product_into_triangle<T>(
                ctx, Cd, Sc, T(1), Uplo::Lower);

            if (mr > 0) {
                const auto Lr = sub(j + ib + c + w, mr, j, ib, nullptr);
                const auto Cr = sub(j + ib + c + w, mr, j + ib + c, w, nullptr);
                (void)trailing_gemm(ctx, Lr, Lrow, Cr, T(-1), T(1),
                                    Transpose::NoTrans, kTrailingTransB<T>,
                                    ComputePrecision::Default);
            }

            // WAR: the next panel's gemm overwrites the scratch this fold still reads.
            if (!ctx.in_order()) ctx.wait();
        },
        [&] {
            if (!ctx.in_order()) ctx.wait();
        });

    return ctx.get_event();
}
}  // namespace

template <> bool potrf_blocked_available<float>()                { return true; }
template <> bool potrf_blocked_available<double>()               { return true; }
template <> bool potrf_blocked_available<std::complex<float>>()  { return true; }
template <> bool potrf_blocked_available<std::complex<double>>() { return true; }

#define BATCHLAS_POTRF_BLOCKED_INSTANTIATE(T)                                                 \
    template unsigned potrf_blocked_debug_params<T>(Queue&, int);                              \
    template std::size_t potrf_blocked_buffer_size<T>(                                        \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo);                             \
    template Event potrf_blocked_dispatch<T>(                                                 \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<std::byte>,             \
        Span<int32_t>, PotrfTrailingGemm<T>, PotrfPanelSolve<T>);                             \
    template std::size_t potrf_blocked_buffer_size_planned<T>(                                \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, const potrf_plan::BlockedParams&); \
    template Event potrf_blocked_dispatch_planned<T>(                                         \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<std::byte>,             \
        Span<int32_t>, const potrf_plan::BlockedParams&, PotrfTrailingGemm<T>,                \
        PotrfPanelSolve<T>);

BATCHLAS_POTRF_BLOCKED_INSTANTIATE(float)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(double)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_BLOCKED_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
