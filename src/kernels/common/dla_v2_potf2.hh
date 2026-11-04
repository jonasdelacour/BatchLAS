// SPDX-License-Identifier: MIT
#pragma once
// v2::potf2_lower: the unblocked lower Cholesky of one matrix per Partition<N>, lane r holding
// row r of L in a RowFrag. Shared by potrf_tiny.cc and posv_tiny.cc. src-side because the
// update is sycl_device::dev_mul's explicit fma (bit-identity with both kernels' history).
// evidence: docs/perf/potrf.md#potrf-the-tiny-tier

#include <src/kernels/common/dla_v2_partition.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/kernels/common/tiny_device.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/device/operand.hh>

#include <cstdint>

namespace batchlas::device::inline v2 {

// How column j of L reaches the lanes below the pivot.
struct Shuffle {};  // one indexed shuffle per element: no local memory, no barrier
// Every lane stores its element, one sub-group barrier, 16-byte vector loads; double-buffered
// by parity. slm[((j & 1) * Mpw + pidx) * NV] is this partition's buffer for column j.
template <int Mpw, class Acc> struct Slm {
    static constexpr int mpw = Mpw;
    Acc slm;
    int pidx;
};

// How the order guard j < n acts. Predicate: every column runs, the effects are masked
// (n may vary across the sub-group). Skip: columns j >= n skip their collectives, legal only
// when n is uniform over the sub-group.
enum class OrderGuard { Predicate, Skip };

// Returns the 1-based first failing column (0: success), partition-uniform. `!(akk > 0)`
// rejects NaN as LAPACK does; every failure effect is a value substitution, never control
// flow, because `live` is partition-uniform but not sub-group-uniform. rA[k] for k > lane
// stays exactly zero (the `k <= lane` predicate), so the identity padding stays inert.
template <class D, int N, int NC, OrderGuard G = OrderGuard::Predicate, class Part, class Bcast>
BATCHLAS_DLA_INLINE std::int32_t potf2_lower(const Part& part, Frag<D, NC, RowFrag>& rA, int n, bool live,
                                             Bcast bc) {
    namespace sd = ::batchlas::sycl_device;
    namespace tn = ::batchlas::tiny_native;
    using R = real_t<D>;
    constexpr bool kSkip = (G == OrderGuard::Skip);
    const int lane = part.lid();
    bool alive = live;
    std::int32_t linfo = 0;

    // A runtime break defeats the unroll at N >= 16, after which rA[j] is a dynamic index.
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < NC; ++j) {
        if constexpr (kSkip) {
            if (j >= n) continue;
        }
        const bool col_live = (j < n);
        const R akk = part.bcast(sd::dev_real(rA[j]), static_cast<std::uint32_t>(j));
        const bool bad = !(akk > R(0));
        if (alive && col_live && bad) {
            linfo = j + 1;  // sticky: the first failure wins
            alive = false;
        }
        const bool act = alive && col_live;
        const R dkk = act ? sycl::sqrt(akk) : R(1);
        const R rinv = R(1) / dkk;  // NOT rsqrt: rsqrt.approx is not the reference

        rA[j] = select(act, select(lane == j, sd::dev_from_real<D>(dkk), sd::dev_mul_real(rA[j], rinv)), rA[j]);

        if constexpr (std::is_same_v<Bcast, Shuffle>) {
BATCHLAS_UNROLL_FULL
            for (int k = j + 1; k < NC; ++k) {
                if constexpr (kSkip) {
                    if (k >= n) continue;
                }
                const D vk = part.bcast(rA[j], static_cast<std::uint32_t>(k));  // guard inside
                const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                rA[k] = select(act && k < n && k <= lane, upd, rA[k]);
            }
        } else {
            static_assert(kSkip, "v2::potf2_lower: the Slm broadcast skips columns >= n");
            using R4 = tn::tiny_r4<R>;
            constexpr int kCpx = static_cast<int>(sizeof(D) / sizeof(R));
            constexpr int kW = tn::tiny_r4_width<D>();
            constexpr int kNV = (N + kW - 1) / kW;
            R4* const sp = &bc.slm[((j & 1) * Bcast::mpw + bc.pidx) * kNV];
            R* const rp = reinterpret_cast<R*>(sp);
            if constexpr (kCpx == 2) {
                rp[2 * lane] = rA[j].re;
                rp[2 * lane + 1] = rA[j].im;
            } else {
                rp[lane] = rA[j];
            }
            sycl::group_barrier(part.sub());
BATCHLAS_UNROLL_FULL
            for (int v = 0; v < kNV; ++v) {
                if ((v + 1) * kW <= j + 1) continue;
                if (v * kW >= n || v * kW >= NC) continue;
                const R4 x = sp[v];
BATCHLAS_UNROLL_FULL
                for (int e = 0; e < kW; ++e) {
                    const int k = v * kW + e;
                    if (k <= j || k >= NC || k >= n) continue;
                    const D vk = tn::tiny_r4_get<D, R>(x, e);
                    const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                    rA[k] = select(act && k < n && k <= lane, upd, rA[k]);
                }
            }
        }
    }
    return linfo;
}

}  // namespace batchlas::device
