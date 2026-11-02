// SPDX-License-Identifier: MIT
#pragma once

// LAPACK ?GEQR2 over a `Tile` (CTA tier and blocked panel leaf) as device LA v2 ops. NOT
// detail::larfg: zlarfg's REAL beta / tau is a contract ormqr/orgqr/ormbr/sy2sb read.
// evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention

#include <src/sycl/device_scalar.hh>
#include <src/kernels/common/dla_v2_scalar.hh>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/geqrf.hh>

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

template <typename D> using LarfgScalars = batchlas::device::v2::LarfgScalars<D>;

template <typename D>
struct GeqrfGlobalTile {
    D* p;
    int ld;
    D& at(int r, int c) const {
        return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// G panels share one local_accessor, so each gets a raw base. Unpadded ld is deliberate:
// both hot patterns walk rows within a column, conflict-free at any ld.
template <typename D> using GeqrfRawTile = GeqrfGlobalTile<D>;

// The phase barriers' and column reductions' scope. SubGroup makes G-packing legal (sub-group
// barriers, butterflies, no static shared from reduce_over_group: the 48 KB hole).
// evidence: docs/perf/qr.md#the-48-kib-launch-hole
enum class GeqrfScope { SubGroup, WorkGroup };

// In place on an m x n tile; reflector j's tau lands at tau_ptr[j] (global, this matrix's),
// kmax <= min(m, n). conj(tau) in the apply is zgeqr2's: real-typed tests cannot see it.
template <typename D, GeqrfScope SC, typename Tile>
inline void geqr2_panel_device(sycl::nd_item<1> it, Tile A, int m, int n, int kmax,
                               D* tau_ptr) {
    namespace v2 = batchlas::device::v2;
    using Ex = std::conditional_t<SC == GeqrfScope::SubGroup, v2::DynSubGroup, v2::DynWorkGroup>;
    const Ex ex{it};
    const auto teams = v2::teams_of(ex);  // no work-group barrier member, by type
    const v2::WideMat<D> a{A.p, m, n, A.ld};

    for (int j = 0; j < kmax; ++j) {
        ex.barrier();  // B1 -- the previous reflector's column writes become visible.
        const D alpha = a(j, j);
        const auto x = a.col(j).tail(j + 1);
        const LarfgScalars<D> h = v2::larfg_scalars<v2::Conv::Lapack>(ex, alpha, x);
        ex.barrier();  // B2 -- every item has read A(j,j) as `alpha`; item 0 overwrites it.
        if (!h.identity) {
            if (ex.leader()) a(j, j) = h.beta;
            v2::scal_or_div(ex, h, x);  // each item rewrites exactly the rows it read
        }
        if (ex.leader()) tau_ptr[j] = h.tau;
        ex.barrier();  // B3 -- beta and v are published to the teams about to read them.
        // Uniform over the barriers' scope: derived from two collectives at that scope.
        if (h.identity) continue;
        v2::larf<Side::Left, v2::SgPerCol>(teams, v2::Arith<D>::conj(h.tau),
                                           v2::unit_head(a.col(j).tail(j), 0),
                                           a.sub(j, j + 1, m - j, n - j - 1));
    }
}

}  // namespace batchlas::geqrf_native
