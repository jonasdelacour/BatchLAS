// Native batched GETRI. With A = F^{-1} L U (F the interchange sequence applied
// FORWARDS), A^-1 = U^-1 L^-1 F: set C := F, then two ROUTED triangular solves.
// F is traced straight into C from ipiv, so there is no permutation kernel, no
// perm[] array and no workspace. preferred() is false for every shape.
// evidence: docs/perf/lu.md#getri-window-evidence

#include "getri_native.hh"

#include "../sycl/device_scalar.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getri {

namespace {

template <typename T> class GetriZeroKernel;
template <typename T> class GetriPermKernel;

// C := 0 over the LOGICAL rows x cols x batch region only, NOT a fill over
// stride*batch: C may be a view into a larger buffer, and writing its padding or
// a neighbour's rows is a corruption no residual on C could see.
template <typename T>
Event getri_zero_c_launch(Queue& ctx, T* c_ptr, int ldc, int stride_c,
                          int n, int batch) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const cp = reinterpret_cast<D*>(c_ptr);
    const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<GetriZeroKernel<T>>(
            sycl::range<2>(static_cast<std::size_t>(batch), elems),
            [=](sycl::item<2> it) {
                const int b = static_cast<int>(it.get_id(0));
                const std::size_t e = it.get_id(1);
                const int r = static_cast<int>(e % static_cast<std::size_t>(n));
                const int c = static_cast<int>(e / static_cast<std::size_t>(n));
                cp[static_cast<std::ptrdiff_t>(b) * stride_c +
                   static_cast<std::ptrdiff_t>(c) * ldc + r] = D{};
            });
    });
    return ctx.get_event();
}

// C := F (the ones of the permutation matrix) and, if requested, info. One
// work-group per matrix, because info is a min-reduction over the diagonal.
template <typename T>
Event getri_perm_launch(Queue& ctx,
                        const T* a_ptr, int lda, int stride_a,
                        T* c_ptr, int ldc, int stride_c,
                        int n, int batch,
                        const int* piv, int piv_stride,
                        int32_t* info_ptr, bool want_info,
                        int wg) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const ap = reinterpret_cast<const D*>(a_ptr);
    D* const cp = reinterpret_cast<D*>(c_ptr);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> slot(sycl::range<1>(1), h);
        h.parallel_for<GetriPermKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());
                const int lwg = static_cast<int>(it.get_local_range(0));

                if (tid == 0) slot[0] = std::numeric_limits<int>::max();
                sycl::group_barrier(it.get_group());

                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;
                const D* const Ab = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
                D* const Cb = cp + static_cast<std::ptrdiff_t>(b) * stride_c;

                for (int i = tid; i < n; i += lwg) {
                    // Walk position i back through the interchange list from
                    // the LAST transposition to the first; each is its own
                    // inverse, so only the ORDER reverses.
                    int r = i;
                    for (int k = n - 1; k >= 0; --k) {
                        const int p = ip[k] - 1;        // 1-BASED on the wire
                        if (r == k) {
                            r = p;
                        } else if (r == p) {
                            r = k;
                        }
                    }
                    // F[i, perm[i]] = 1, column-major.
                    Cb[static_cast<std::ptrdiff_t>(r) * ldc + i] =
                        sycl_device::dev_one<D>();

                    if (want_info) {
                        // EXACT zero, no epsilon: ?GETRI reports the first
                        // U(i,i) that is a true binary zero; |u| < eps would
                        // diverge from the vendor invisibly.
                        const D u = Ab[static_cast<std::ptrdiff_t>(i) * lda + i];
                        if (sycl_device::dev_is_zero(u)) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>(
                                slot[0])
                                .fetch_min(i + 1);      // 1-BASED, LAPACK info
                        }
                    }
                }

                if (want_info) {
                    sycl::group_barrier(it.get_group());
                    if (tid == 0) {
                        const int v = slot[0];
                        info_ptr[b] =
                            (v == std::numeric_limits<int>::max()) ? 0 : static_cast<int32_t>(v);
                    }
                }
            });
    });
    return ctx.get_event();
}

}  // namespace

// True for all four types. Defined beside the driver so that "the flag is true"
// and "the file is compiled" are the same fact.
template <> bool getri_blocked_available<float>()                { return true; }
template <> bool getri_blocked_available<double>()               { return true; }
template <> bool getri_blocked_available<std::complex<float>>()  { return true; }
template <> bool getri_blocked_available<std::complex<double>>() { return true; }

// Workspace: none. Called from inside a layout function under measuring(), so
// per mempool.hh:180-186 it must not dereference A.data_ptr(); it reads nothing.
template <typename T>
std::size_t getri_blocked_buffer_size(Queue&, const MatrixView<T, MatrixFormat::Dense>&) {
    return 0;
}

template <typename T>
Event getri_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info_out,
                             GetriSolveTrsm<T> solve_trsm) {
    static_cast<void>(workspace);   // this arm needs none

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // Every RouteTable<Op::getri,T>::supports() gate is re-applied, because this
    // entry point is reachable WITHOUT the table and route_resolve.hh:165 falls
    // through to automatic() when a forced route is unsupported -- so a gate that
    // is wrong here makes a pinned-route test silently measure cuBLAS and pass.
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("getri_blocked: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getri_blocked: A must be square");
    }
    if (A.is_heterogeneous() || C.is_heterogeneous()) {
        throw std::invalid_argument("getri_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getri_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // This file's kernels need no sub-group size; the routed trsm does.
        throw std::runtime_error("getri_blocked: device does not offer sub-group size 32");
    }

    // Not expressible in GetriShape and so ungateable by the route:
    // getri_buffer_size(ctx, A) has no C, so the shape is a function of A alone.
    if (C.rows() != n || C.cols() != n) {
        throw std::invalid_argument("getri_blocked: C must be square of A's order");
    }
    if (C.batch_size() != A.batch_size()) {
        throw std::invalid_argument("getri_blocked: A and C must agree on batch size");
    }
    if (C.data_ptr() == A.data_ptr()) {
        // Refused, matching cuBLAS, and a hard requirement here: C is zeroed
        // before the solves read A's triangles, so aliasing destroys the factor.
        throw std::invalid_argument(
            "getri_blocked: in-place (C aliasing A) is not supported, as it is not by "
            "cublas<t>getriBatched");
    }

    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getri_blocked: pivot span is shorter than n * batch");
    }
    if (!solve_trsm) {
        // Deliberately no native fallback: the router, not this file, chooses
        // the trsm arm, so an empty seam throws rather than reaching for one.
        throw std::invalid_argument(
            "getri_blocked: the solve seam is empty. Inject the ROUTED batchlas::trsm "
            "(the facade does; a direct caller must too) -- this driver deliberately has "
            "no native fallback, so that the router, and not this file, chooses the trsm "
            "arm.");
    }

    // PACKED 1-BASED int32 inside the caller's int64 span, the format cuBLAS,
    // rocSOLVER and native getrf agree on; getrf and getri route independently,
    // so every mixture of arms is reachable and must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    const bool want_info = info_out.size() >= static_cast<std::size_t>(batch);

    // Derived from n rather than hardcoded as a portability choice, not a
    // performance claim: it measures as noise (docs/perf/lu.md#negative-results).
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    int wg = 32;
    while (wg < n && wg < 256) wg <<= 1;
    while (wg > max_wg && wg > 32) wg >>= 1;

    (void)getri_zero_c_launch<T>(ctx, C.data_ptr(), C.ld(), C.stride(), n, batch);
    // In-order queues give the ordering for free; an out-of-order one does not.
    if (!ctx.in_order()) ctx.wait();

    (void)getri_perm_launch<T>(ctx, A.data_ptr(), A.ld(), A.stride(),
                               C.data_ptr(), C.ld(), C.stride(),
                               n, batch, piv_i32.data(), /*piv_stride=*/n,
                               want_info ? info_out.data() : nullptr, want_info, wg);
    if (!ctx.in_order()) ctx.wait();

    // C := L^-1 C, then C := U^-1 C. alpha comes THIRD in the public trsm.
    (void)solve_trsm(ctx, A, C, T(1), Side::Left, Uplo::Lower,
                     Transpose::NoTrans, Diag::Unit);
    if (!ctx.in_order()) ctx.wait();
    return solve_trsm(ctx, A, C, T(1), Side::Left, Uplo::Upper,
                      Transpose::NoTrans, Diag::NonUnit);
}

#define BATCHLAS_GETRI_INSTANTIATE(T)                                                      \
    template std::size_t getri_blocked_buffer_size<T>(                                     \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&);                                \
    template Event getri_blocked_dispatch<T>(                                              \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&,                                         \
        Span<int64_t>, Span<std::byte>, Span<int32_t>, GetriSolveTrsm<T>);

BATCHLAS_GETRI_INSTANTIATE(float)
BATCHLAS_GETRI_INSTANTIATE(double)
BATCHLAS_GETRI_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRI_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRI_INSTANTIATE

}  // namespace sycl_getri
}  // namespace batchlas
