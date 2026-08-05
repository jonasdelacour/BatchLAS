// syevx_direct: partial symmetric eigensolve by full decomposition + selection.
//
// This is the correct choice whenever the requested fraction of the spectrum is
// large enough that an iterative method cannot amortize its matvecs, and for small
// n where a subset solver cannot beat the CTA-resident full solver at all.
// See SYEVX_PLAN.md §2 (cost model) and §3.1.

#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <blas/functions/syev.hh>
#include "../util/template-instantiations.hh"

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxDirectSelectKernel;

namespace {

// Shape of the private copy of A that syev consumes. Packed (ld == n,
// stride == n*n) regardless of the input view's own layout.
template <typename T>
inline MatrixView<T, MatrixFormat::Dense> packed_copy_view(T* data,
                                                           int64_t n,
                                                           int64_t batch_size,
                                                           T** ptr_array) {
    return MatrixView<T, MatrixFormat::Dense>(data,
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              static_cast<int>(n),
                                              n * n,
                                              static_cast<int>(batch_size),
                                              ptr_array);
}

// Number of eigenvalues <= x in an ASCENDING array, i.e. the index of the first
// entry strictly greater than x (std::upper_bound, not lower_bound).
//
// The `<=` here is load-bearing and must not be relaxed to `<`: it is exactly the
// predicate `stebz`'s Sturm count implements (src/extensions/stebz.cc:34-59), and
// matching it is what makes `Direct` and `DirectSubset` agree on m for the same
// half-open interval (vl, vu]. A boundary within rounding distance of an
// eigenvalue can still make the two differ by one -- they compute the count by
// genuinely different means -- but nothing else may.
template <typename R>
inline int64_t count_le(const R* asc, int64_t n, R x) {
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (asc[mid] <= x) lo = mid + 1;
        else               hi = mid;
    }
    return lo;
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_direct(Queue& ctx,
                   const MatrixView<T, MFormat>& A,
                   Span<typename base_type<T>::type> W,
                   Span<int32_t> m,
                   size_t neigs,
                   Span<std::byte> workspace,
                   JobType jobz,
                   const MatrixView<T, MatrixFormat::Dense>& V,
                   const SyevxParams<T>& params) {
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)m; (void)neigs; (void)workspace;
        (void)jobz; (void)V; (void)params;
        throw std::runtime_error("syevx_direct: only dense matrices are supported");
    } else {
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        if (A.rows() != A.cols()) {
            throw std::runtime_error("syevx_direct: A must be square");
        }
        // `neigs` is a capacity, so exceeding n is harmless -- the tail of W and V
        // simply goes unwritten. It is clamped inside syevx_resolve_range and is
        // deliberately no longer rejected. (This is the one place an existing
        // call's outcome changed from a throw to a success; see the doc comment on
        // syevx_direct in extensions.hh.)
        if (!m.empty() && static_cast<int64_t>(m.size()) < batch_size) {
            throw std::runtime_error("syevx_direct: m must cover every batch item");
        }
        // The RANGE, on the other hand, is checked here and not only in the public
        // `syevx`. This function is a public entry point in its own right -- the
        // test suite calls it directly, and so may a caller who wants to pin the
        // algorithm -- and an out-of-range index block would otherwise reach a
        // device kernel that indexes an n-entry eigenvalue array with il..iu. The
        // resolver clamps as a second line of defence, so this throw is about
        // telling the caller rather than about memory safety; both are wanted. The
        // wording matches validate_syevx_range_params in syevx.cc, and so does the
        // exception type (std::invalid_argument, not the std::runtime_error the
        // shape checks above use).
        if (params.select == SyevxSelect::Index) {
            const int64_t iu = (params.iu < 0) ? (n - 1) : params.iu;
            if (params.il < 0 || iu >= n || params.il > iu) {
                throw std::invalid_argument(
                    "syevx_direct: SyevxSelect::Index requires 0 <= il <= iu < n (iu < 0 means "
                    "n-1); an empty block is expressed with neigs == 0, not with il > iu");
            }
        }
        if (params.select == SyevxSelect::Value && !(params.vl < params.vu)) {
            throw std::invalid_argument(
                "syevx_direct: SyevxSelect::Value requires vl < vu for the half-open interval "
                "(vl, vu]; an empty or inverted interval is almost always swapped arguments");
        }

        auto pool = BumpAllocator(workspace);
        auto a_copy_data = pool.allocate<T>(ctx, static_cast<size_t>(n * n * batch_size));
        auto a_copy_ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch_size));
        auto lambdas = pool.allocate<float_type>(ctx, static_cast<size_t>(n * batch_size));

        auto A_copy = packed_copy_view<T>(a_copy_data.data(), n, batch_size, a_copy_ptrs.data());

        // syev overwrites its input; syevx must leave A intact.
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, A_copy, A);

        auto syev_ws = pool.allocate<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, lambdas, jobz, Uplo::Lower));
        syev<B>(ctx, A_copy, lambdas, {.jobz = jobz}, syev_ws);

        // syev returns eigenvalues ascending, so every range reduces to picking a
        // contiguous block out of `lambdas` -- for an index block statically, for a
        // value interval by two searches over that ascending array.
        const auto rr = syevx_resolve_range(n, neigs, params);

        // Three distinct quantities that all used to be one `k`. Conflating them is
        // how a batched solver silently writes item b's answers into item b+1's
        // slots, so they are named separately and never mixed.
        const int64_t capacity = static_cast<int64_t>(neigs);  // stride of W, columns of V
        const bool value_range = rr.value_range;
        const bool reverse = rr.reverse;
        const int64_t range_il = rr.il;                        // index ranges only
        const int64_t range_count = rr.max_count;              // index ranges only
        const float_type vl = params.vl;
        const float_type vu = params.vu;

        const size_t wg = std::min<size_t>(256, static_cast<size_t>(std::max<int64_t>(n, 1)));
        const auto* lam_ptr = lambdas.data();
        auto* w_ptr = W.data();
        const T* src_ptr = A_copy.data_ptr();
        T* dst_ptr = want_eigenvectors ? V.data_ptr() : nullptr;
        const int64_t dst_ld = want_eigenvectors ? V.ld() : 0;
        const int64_t dst_stride = want_eigenvectors ? V.stride() : 0;
        // How many columns of V may be written at all. Normally the capacity, but
        // a caller is allowed to declare a capacity above n (it is clamped, not
        // rejected), and V is then legitimately narrower than `neigs`; the zeroing
        // pass below must not run off the end of it.
        const int64_t dst_cols = want_eigenvectors
                                     ? std::min<int64_t>(static_cast<int64_t>(neigs), V.cols())
                                     : 0;
        int32_t* m_ptr = m.empty() ? nullptr : m.data();

        ctx->submit([&](sycl::handler& h) {
            // first index of the block, and its true (untruncated) size.
            auto block = sycl::local_accessor<int64_t, 1>(sycl::range<1>(2), h);

            h.parallel_for<SyevxDirectSelectKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch_size) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                    const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));
                    auto cta = item.get_group();

                    const auto* lam = lam_ptr + bid * n;

                    int64_t first;
                    int64_t count;
                    if (value_range) {
                        // Two O(log n) searches, done once per group rather than
                        // redundantly per work-item; same shape as stebz's
                        // count-then-barrier.
                        if (tid == 0) {
                            const int64_t c_lo = count_le<float_type>(lam, n, vl);
                            const int64_t c_hi = count_le<float_type>(lam, n, vu);
                            block[0] = c_lo;
                            // Cannot go negative for vl < vu on an ascending array,
                            // but m[b] is consumed as a loop bound downstream and an
                            // unsigned wrap there would be an enormous write.
                            block[1] = (c_hi > c_lo) ? (c_hi - c_lo) : int64_t(0);
                        }
                        sycl::group_barrier(cta);
                        first = block[0];
                        count = block[1];
                    } else {
                        first = range_il;
                        count = range_count;
                    }

                    // Truncation policy: keep the LOWEST `capacity` eigenvalues of
                    // the block, so the answer does not depend on the requested
                    // output order. `reverse` then only flips within what was kept.
                    const int64_t write = (count < capacity) ? count : capacity;
                    const int64_t hi_src = first + write - 1;

                    auto* w = w_ptr + bid * capacity;
                    for (int64_t i = tid; i < write; i += local_size) {
                        const int64_t src = reverse ? (hi_src - i) : (first + i);
                        w[i] = lam[src];
                    }

                    if (dst_ptr != nullptr) {
                        const auto* vsrc = src_ptr + bid * n * n;
                        auto* vdst = dst_ptr + bid * dst_stride;
                        for (int64_t linear = tid; linear < n * write; linear += local_size) {
                            const int64_t row = linear % n;
                            const int64_t col = linear / n;
                            const int64_t src_col = reverse ? (hi_src - col) : (first + col);
                            vdst[row + col * dst_ld] = vsrc[row + src_col * n];
                        }
                        // Columns [write, dst_cols) are written as EXACTLY zero
                        // rather than left holding whatever the caller's buffer
                        // had. This is not tidiness: syevx_direct_subset cannot
                        // avoid zeroing them (stein zeroes the unused columns so
                        // that its uniform-width back-transforms have something
                        // inert to transform, and an orthogonal map keeps zero at
                        // zero), and `Auto` picks between the two paths on (n,
                        // batch). Leaving them stale here would mean the identical
                        // Value-range call returned zeros at one shape and the
                        // caller's previous contents -- or uninitialized bytes
                        // decoding as NaN -- at another. One contract, both paths;
                        // see the note on V in syevx's doc comment.
                        //
                        // Unused W slots are NOT zeroed, deliberately: W's contract
                        // is "untouched past m[b]", the sentinel tests depend on it,
                        // and unlike V there is no downstream uniform-width kernel
                        // that would consume them.
                        for (int64_t linear = tid; linear < n * (dst_cols - write);
                             linear += local_size) {
                            const int64_t row = linear % n;
                            const int64_t col = write + linear / n;
                            vdst[row + col * dst_ld] = T(0);
                        }
                    }

                    if (m_ptr != nullptr && tid == 0) {
                        // The TRUE count, not `write`: m[b] > capacity is the
                        // caller's truncation signal.
                        m_ptr[bid] = static_cast<int32_t>(count);
                    }
                });
        });

        return ctx.get_event();
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_direct_buffer_size(Queue& ctx,
                                const MatrixView<T, MFormat>& A,
                                Span<typename base_type<T>::type> W,
                                size_t neigs,
                                JobType jobz,
                                const MatrixView<T, MatrixFormat::Dense>& V,
                                const SyevxParams<T>& params) {
    using float_type = typename base_type<T>::type;

    if constexpr (MFormat != MatrixFormat::Dense) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)jobz; (void)V; (void)params;
        return 0;
    } else {
        // Range-independent by construction, and worth stating because this is the
        // first place a reader chasing a value-range memory question will look: the
        // workspace is a private copy of A, the full eigenvalue array and syev's own
        // scratch, all sized on n and batch alone. `neigs` -- and therefore the
        // capacity semantics that come with a Value range -- never enters.
        (void)W; (void)V; (void)params; (void)neigs;
        const int64_t n = A.rows();
        const int64_t batch_size = A.batch_size();

        size_t work_size = 0;
        work_size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n * n * batch_size));
        work_size += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch_size));
        work_size += BumpAllocator::allocation_size<float_type>(ctx, static_cast<size_t>(n * batch_size));

        auto A_copy = packed_copy_view<T>(nullptr, n, batch_size, nullptr);
        work_size += BumpAllocator::allocation_size<std::byte>(
            ctx, syev_buffer_size<B>(ctx, A_copy, Span<float_type>(), jobz, Uplo::Lower));

        return work_size;
    }
}

#define SYEVX_DIRECT_INSTANTIATE(back, fp, fmt) \
    template Event syevx_direct<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        Span<int32_t>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    /* The m-less forwarder is inline and would not need an instantiation to be \
       CALLED, but instantiating it keeps the symbol this library exported before \
       `m` was added, so an object file built against the old header still links. */\
    template Event syevx_direct<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_direct_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_DIRECT_INSTANTIATE, back, fp)

#define SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
#endif

#undef SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND
#undef SYEVX_DIRECT_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_DIRECT_INSTANTIATE

} // namespace batchlas
