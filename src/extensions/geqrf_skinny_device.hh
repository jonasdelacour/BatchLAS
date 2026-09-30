#pragma once

// The CTA tier's skinny leg: ?GEQR2 on a thin m x n panel held entirely in one sub-group's
// registers. P lanes own one matrix (P = 8, 16 or 32, so 32/P matrices share a sub-group) and
// row r lives on lane r % P, slot r / P. No local memory and no barriers; the n-1-j column
// reductions of a reflector are independent butterflies. The Householder scalars are
// geqrf_larfg_scalars, so tau/beta follow the LAPACK convention of the resident body.
// evidence: docs/perf/blackwell.md#geqrf-the-skinny-register-leg

#include "geqrf_cta_device.hh"

#include <sycl/sycl.hpp>

#include <cstddef>

namespace batchlas::geqrf_native {

// Butterflies over one P-lane partition: an xor offset below P never leaves an aligned block
// of P lanes, so 32/P matrices reduce side by side without a partition object.
template <int P, typename D>
inline D geqrf_part_sum(const sycl::sub_group& sg, D v) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        using R = real_of<D>;
        R re = v.re;
        R im = v.im;
#pragma unroll
        for (int off = P / 2; off > 0; off >>= 1) {
            re += sycl::permute_group_by_xor(sg, re, static_cast<uint32_t>(off));
            im += sycl::permute_group_by_xor(sg, im, static_cast<uint32_t>(off));
        }
        return D{re, im};
    } else {
#pragma unroll
        for (int off = P / 2; off > 0; off >>= 1) {
            v += sycl::permute_group_by_xor(sg, v, static_cast<uint32_t>(off));
        }
        return v;
    }
}

template <int P, typename R>
inline R geqrf_part_max(const sycl::sub_group& sg, R v) {
#pragma unroll
    for (int off = P / 2; off > 0; off >>= 1) {
        v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, static_cast<uint32_t>(off)));
    }
    return v;
}

// The kernel's shape bounds, shared by the launcher's fit predicate and the tests.
inline constexpr int kGeqrfSkinnyMaxCols = 8;
inline constexpr int kGeqrfSkinnyWg = 128;

// In place on A (column-major, leading dimension ld, one matrix); tau[j] is written by the
// partition's lane j. Every lane of the sub-group must call this (the collectives are
// sub-group wide); a dead partition passes live = false and stores nothing.
// Requires m >= n, n <= N <= P, m <= P * RP.
template <typename D, int N, int RP, int P>
inline void geqr2_skinny_device(const sycl::sub_group& sg, D* A, std::ptrdiff_t ld, int m,
                                int n, D* tau, bool live) {
    namespace sd = batchlas::sycl_device;
    using R = real_of<D>;
    static_assert(N <= P && P <= 32 && (32 % P) == 0, "row j must sit in slot 0");

    const int sl = static_cast<int>(sg.get_local_linear_id());
    const int lane = sl % P;
    const int base = sl - lane;

    // Top-level array, fully unrolled indexing only: a dynamic index sends it to the stack.
    D rA[RP][N];
#pragma unroll
    for (int s = 0; s < RP; ++s) {
        const int r = lane + P * s;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            rA[s][c] = (r < m && c < n) ? A[r + static_cast<std::ptrdiff_t>(c) * ld]
                                        : dev_zero<D>();
        }
    }

#pragma unroll
    for (int j = 0; j < N; ++j) {
        // `continue`, not `break`: a data-dependent trip count declines the unroll. n is
        // launch-uniform (the entry point rejects heterogeneous batches).
        if (j >= n) continue;

        const D alpha = sycl::select_from_group(sg, rA[0][j], static_cast<uint32_t>(base + j));

        R smax = (lane == j) ? dev_absmax<D>(rA[0][j]) : R(0);
#pragma unroll
        for (int s = 0; s < RP; ++s) {
            const int r = lane + P * s;
            if (r > j && r < m) smax = sycl::fmax(smax, dev_absmax<D>(rA[s][j]));
        }
        smax = geqrf_part_max<P, R>(sg, smax);

        R ssq = R(0);
        if (smax > R(0)) {
#pragma unroll
            for (int s = 0; s < RP; ++s) {
                const int r = lane + P * s;
                if (r > j && r < m) ssq += dev_abs2_scaled<D>(rA[s][j], smax);
            }
        }
        ssq = geqrf_part_sum<P, R>(sg, ssq);

        const LarfgScalars<D> h = geqrf_larfg_scalars<D>(alpha, smax, ssq);
        if (live && lane == j) tau[j] = h.tau;

        // identity is per PARTITION, so it cannot branch around the butterflies below:
        // it zeroes the reflector instead (tau = 0, v = 0, A unchanged).
        const bool apply = !h.identity;
#pragma unroll
        for (int s = 0; s < RP; ++s) {
            const int r = lane + P * s;
            if (!apply) continue;
            if (r == j) {
                rA[s][j] = h.beta;
            } else if (r > j && r < m) {
                rA[s][j] = h.use_mul ? sd::dev_mul(rA[s][j], h.vfactor)
                                     : sd::dev_div(rA[s][j], h.vfactor);
            }
        }
        const D ctau = apply ? sd::dev_conj(h.tau) : dev_zero<D>();

        D w[N];
#pragma unroll
        for (int c = 0; c < N; ++c) {
            w[c] = dev_zero<D>();
            if (c <= j || c >= n) continue;
#pragma unroll
            for (int s = 0; s < RP; ++s) {
                const int r = lane + P * s;
                const D v = (!apply || r < j || r >= m)
                                ? dev_zero<D>()
                                : (r == j ? sd::dev_one<D>() : rA[s][j]);
                w[c] = dev_add(w[c], sd::dev_mul(sd::dev_conj(v), rA[s][c]));
            }
        }
#pragma unroll
        for (int c = 0; c < N; ++c) {
            if (c <= j || c >= n) continue;
            w[c] = sd::dev_mul(ctau, geqrf_part_sum<P, D>(sg, w[c]));
        }
#pragma unroll
        for (int s = 0; s < RP; ++s) {
            const int r = lane + P * s;
            const D v = (!apply || r < j || r >= m) ? dev_zero<D>()
                                                    : (r == j ? sd::dev_one<D>() : rA[s][j]);
#pragma unroll
            for (int c = 0; c < N; ++c) {
                if (c <= j || c >= n) continue;
                rA[s][c] = sd::dev_sub(rA[s][c], sd::dev_mul(w[c], v));
            }
        }
    }

    if (!live) return;
#pragma unroll
    for (int s = 0; s < RP; ++s) {
        const int r = lane + P * s;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            if (r < m && c < n) A[r + static_cast<std::ptrdiff_t>(c) * ld] = rA[s][c];
        }
    }
}

}  // namespace batchlas::geqrf_native
