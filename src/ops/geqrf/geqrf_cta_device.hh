// SPDX-License-Identifier: MIT
#pragma once

// The native batched GEQRF panel's device code: ?GEQR2 over a `Tile` (local_accessor or raw
// pointer). Scalars are v2::larfg_scalars<conv::Lapack>, NOT detail::larfg (keeps alpha's phase;
// clarfg/zlarfg and the vendors return a REAL beta, and tau is read by ormqr/orgqr/ormbr/sy2sb).
// evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention

#include <src/sycl/device_scalar.hh>
#include <src/ops/geqrf/geqrf_v2.hh>

#include <batchlas/blas/device/v2/dla.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::geqrf_native {

using batchlas::sycl_device::Cx;

template <typename D> struct RealOf            { using type = D; };
template <typename R> struct RealOf<Cx<R>>     { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

template <typename D>
inline D dev_zero() {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{real_of<D>(0), real_of<D>(0)};
    } else {
        return D(0);
    }
}

template <typename D>
struct GeqrfGlobalTile {
    D* p;
    int ld;
    D& at(int r, int c) const {
        return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// Raw-pointer tile: G panels share one local_accessor, so each needs its own base; ld unpadded.
template <typename D> using GeqrfRawTile = GeqrfGlobalTile<D>;

// Which group the barriers synchronise and the two column reductions use. SubGroup makes
// G-packing legal (sub-group barriers, butterfly reductions, no cross-sub-group sync) and drops
// reduce_over_group's static shared allocation, which trips the 48 KB hole.
// evidence: docs/perf/qr.md#the-48-kib-launch-hole
enum class GeqrfScope { SubGroup, WorkGroup };

// LAPACK ?GEQR2 on an m x n tile, in place; conj(tau) in the apply is zgeqr2's (real-typed tests
// cannot see it). Reflector j's tau lands at tau_ptr[j] (global, this matrix's), kmax <= min(m, n).
// Absolute rows and columns: a relative loop over a moved base costs 3-4 REG on sm_89.
template <typename D, GeqrfScope SC, typename Tile>
inline void geqr2_panel_device(sycl::nd_item<1> it, Tile A, int m, int n, int kmax,
                               D* tau_ptr) {
    namespace v2 = batchlas::device::v2;
    namespace q2 = batchlas::geqrf_v2;
    using R = real_of<D>;
    using F = v2::FmaArith<D>;  // arith::Fma: sycl_device::dev_mul / dev_div bit for bit
    constexpr bool sub = (SC == GeqrfScope::SubGroup);
    const v2::RtGroup<sub ? v2::Scope::Sub : v2::Scope::Group> ex{it};
    const int lanes = ex.team_size();
    const auto a = v2::mat_wide(A.p, m, n, A.ld);

    for (int j = 0; j < kmax; ++j) {
        ex.barrier(v2::Mem::Global);  // B1 -- the previous reflector's column writes are visible.
        const D alpha = q2::by_parts(a(j, j));
        R smax = q2::absmax_ri(alpha);
        v2::for_each_owned(ex, v2::Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { smax = sycl::fmax(smax, q2::absmax_ri(a(r, j))); });
        // Scope::Sub: the run-time xor butterfly; Scope::Group: portable::reduce_over_group.
        if constexpr (sub) smax = q2::xor_down_rt(ex.sub(), smax, q2::Fmax());
        else smax = ex.group_reduce(smax, sycl::maximum<R>());
        R ssq = R(0);  // the collective sits outside the `smax > 0` test: all items reach it
        if (smax > R(0)) {
            v2::for_each_owned(ex, v2::Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { ssq += q2::abs2_scaled(a(r, j), smax); });
        }
        if constexpr (sub) ssq = q2::xor_down_rt(ex.sub(), ssq, sycl::plus<R>());
        else ssq = ex.group_reduce(ssq, sycl::plus<R>());
        const v2::LarfgScalars<D> h = v2::larfg_scalars<v2::conv::Lapack>(alpha, v2::ScaledSsq<R>{smax, ssq});

        ex.barrier(v2::Mem::Global);  // B2 -- every item has read A(j,j) as `alpha`; item 0 overwrites it.
        if (!h.identity) {
            if (ex.leader()) a(j, j) = h.beta;
            // No barrier needed here: each item writes exactly the rows it read.
            if (h.use_mul) {
                v2::for_each_owned(ex, v2::Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { a(r, j) = F::mul(a(r, j), h.scale); });
            } else {
                v2::for_each_owned(ex, v2::Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { a(r, j) = F::div(a(r, j), h.scale); });
            }
        }
        if (ex.leader()) tau_ptr[j] = h.tau;
        ex.barrier(v2::Mem::Global);  // B3 -- beta and v are published to the teams about to read them.

        // Uniform over the barriers' scope (two collectives there): cannot desync B1/B2/B3.
        if (h.identity) continue;
        q2::larf_left_head(ex, lanes, v2::Arith<D>::conj(h.tau), a, j, m, n);
    }
}

}  // namespace batchlas::geqrf_native
