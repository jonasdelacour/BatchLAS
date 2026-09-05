// Native batched POTRF: the blocked right-looking driver, Uplo::Lower only. Per panel j:
// the leaf potrf on the ib x ib diagonal block, an info-merge/quench fixup, the panel solve
// L21 = A21 L11^{-H}, and A22 -= L21 L21^H. See docs/perf/potrf.md#the-blocked-driver

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

// nb is the diagonal block order -- the leaf's order and the trailing update's k -- and W
// the trailing panel width; neither is potrf_cta_max_n<T>(). evidence: docs/perf/potrf.md#nb-and-w
template <typename T> struct PotrfBlockedConst;
template <> struct PotrfBlockedConst<float>                { static constexpr int NB = 128; static constexpr int W = 128; };
template <> struct PotrfBlockedConst<double>               { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct PotrfBlockedConst<std::complex<float>>  { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct PotrfBlockedConst<std::complex<double>> { static constexpr int NB = 64;  static constexpr int W = 16; };

// Blocking overrides only, never routing; read once so the sizing query and the call agree.
inline int potrf_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return 0;
    const int v = std::atoi(raw);
    return v > 0 ? v : 0;
}
inline int potrf_nb_env() { static const int v = potrf_env_int("BATCHLAS_POTRF_NB"); return v; }
inline int potrf_w_env()  { static const int v = potrf_env_int("BATCHLAS_POTRF_W");  return v; }

struct PotrfBlockedParams {
    int nb;  // diagonal block order, and the trailing update's k
    int W;   // trailing-update column-panel width
};

template <typename T>
PotrfBlockedParams potrf_blocked_params(Queue& ctx, int n) {
    using C = PotrfBlockedConst<T>;

    // From THIS device's SLM: the hardcoded potrf_cta_max_n<T>() can name a block the leaf refuses.
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int ceiling = potrf_cta_max_n_for_slm<T>(local_mem > 4096 ? local_mem - 4096 : 0);

    const int want = potrf_nb_env() ? potrf_nb_env() : C::NB;
    int nb = std::min(want, std::max(ceiling, 1));
    if (n > 0) nb = std::min(nb, n);

    // Rounded to whole trsm_cta_max_n<T>() blocks, so a hand-set BATCHLAS_POTRF_NB stays measured.
    const int leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    if (leaf_trsm > 0 && nb >= leaf_trsm) {
        nb = (nb / leaf_trsm) * leaf_trsm;
    }
    if (nb < 1) nb = 1;

    int W = potrf_w_env() ? potrf_w_env() : C::W;
    if (W < 1) W = 1;

    return {nb, W};
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

template <typename T>
Event potrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Uplo uplo,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             PotrfTrailingGemm<T> trailing_gemm,
                             PotrfPanelSolve<T> panel_solve) {
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
        throw std::invalid_argument("potrf_blocked: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("potrf_blocked: degenerate extents");
    }
    if (uplo != Uplo::Lower) {
        throw std::invalid_argument(
            "potrf_blocked: Uplo::Upper is not implemented; the driver factors the "
            "lower triangle only (route_potrf.hh:270-278)");
    }
    if (A.is_heterogeneous()) {
        throw std::invalid_argument("potrf_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("potrf_blocked: GPU queues only");
    }

    const auto p = potrf_blocked_params<T>(ctx, n);
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

    for (int j = 0; j < n; j += nb) {
        // ib < nb implies m2 == 0, so a short final block issues no panel solve or trailing update.
        const int ib = std::min(nb, n - j);
        const int m2 = n - j - ib;

        const auto A11 = sub(j, ib, j, ib, ws.a11_ptrs.data());
        potrf_cta_dispatch<T>(ctx, A11, Uplo::Lower, ws.leaf_ws, ws.leaf_info);

        // Unguarded: stale leaf_info, and a solve dividing by a pivot the quench has not replaced.
        if (!ctx.in_order()) ctx.wait();

        potrf_blocked_panel_fixup<T>(ctx, a_ptr, ld, stride, j, ib, m2, batch,
                                     info.data(), ws.leaf_info.data(), fixup_wg);

        if (m2 == 0) break;

        if (!ctx.in_order()) ctx.wait();

        const auto A21 = sub(j + ib, m2, j, ib, ws.a21_ptrs.data());
        panel_solve(ctx, A11, A21, T(1), Side::Right, Uplo::Lower,
                    Transpose::ConjTrans, Diag::NonUnit);

        if (!ctx.in_order()) ctx.wait();

        // A plain square gemm over A22 would write the upper triangle, which potrf(Lower) must leave
        // untouched; the W x W block is folded in instead. Only a poisoned-upper test sees a lost fold.
        for (int c = 0; c < m2; c += W) {
            const int w = std::min(W, m2 - c);

            const auto Lrow = sub(j + ib + c, w, j, ib, nullptr);
            const auto Cd = sub(j + ib + c, w, j + ib + c, w, nullptr);
            const MatrixView<T, MatrixFormat::Dense> Sc(prod_ptr, w, w, W, W * W, batch);

            trailing_gemm(ctx, Lrow, Lrow, Sc, T(-1), T(0),
                          Transpose::NoTrans, kTrailingTransB<T>,
                          ComputePrecision::Default);

            // RAW: out of order the fold reads stale scratch -- a wrong factor with info == 0.
            if (!ctx.in_order()) ctx.wait();

            ::batchlas::detail::fold_symmetric_product_into_triangle<T>(
                ctx, Cd, Sc, T(1), Uplo::Lower);

            const int mr = m2 - c - w;
            if (mr > 0) {
                const auto Lr = sub(j + ib + c + w, mr, j, ib, nullptr);
                const auto Cr = sub(j + ib + c + w, mr, j + ib + c, w, nullptr);
                trailing_gemm(ctx, Lr, Lrow, Cr, T(-1), T(1),
                              Transpose::NoTrans, kTrailingTransB<T>,
                              ComputePrecision::Default);
            }

            // WAR: the next panel's gemm overwrites the scratch this fold still reads.
            if (!ctx.in_order()) ctx.wait();
        }

        if (!ctx.in_order()) ctx.wait();
    }

    return ctx.get_event();
}

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
        Span<int32_t>, PotrfTrailingGemm<T>, PotrfPanelSolve<T>);

BATCHLAS_POTRF_BLOCKED_INSTANTIATE(float)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(double)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_BLOCKED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_BLOCKED_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
