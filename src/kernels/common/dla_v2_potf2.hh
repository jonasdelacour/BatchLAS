// SPDX-License-Identifier: MIT
#pragma once
// The tiny tiers' potf2 distribution: one matrix per Partition<N>, lane r holding row r of L in a
// RowFrag, run by the one potf2 loop (include/batchlas/device/lapack/potrf.hh). Shared by
// potrf_tiny.cc and posv_tiny.cc. src-side until core has its scalar forms: the arithmetic is
// sycl_device's (dev_mul_real is a * s, core scale_real is s * a) and the SLM broadcast reads
// tiny_r4 vectors (v22 core requests). evidence: docs/perf/potrf.md#potrf-the-tiny-tier

#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <src/kernels/common/tiny_device.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/device/operand.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::tiny_native {

// How the order guard j < n acts. Predicate: every column runs, the effects are masked
// (n may vary across the sub-group). Skip: columns j >= n skip their collectives, legal only
// when n is uniform over the sub-group.
enum class OrderGuard { Predicate, Skip };

// Bcast (device::bcast_policy): how column j of L reaches the lanes below the pivot. The loop's
// failure effects are value substitutions, never control flow, because `live` is partition-uniform
// but not sub-group-uniform. rA[k] for k > lane stays exactly zero (the `k <= lane` predicate),
// so the identity padding stays inert.
template <class D, int N, int NC, OrderGuard G, class Part, class Bcast> struct Potf2RowFrag {
    using R = ::batchlas::device::real_t<D>;
    static constexpr int static_cols = NC;
    static constexpr bool skips = (G == OrderGuard::Skip);
    const Part& part;
    ::batchlas::device::Frag<D, NC, ::batchlas::device::RowFrag>& rA;
    int n;
    const Bcast& bc;
    int lane;

    BATCHLAS_DLA_INLINE bool skip(int j) const { return j >= n; }
    BATCHLAS_DLA_INLINE bool col_live(int j) const { return j < n; }
    BATCHLAS_DLA_INLINE R pivot(int j) const {
        return part.bcast(sycl_device::dev_real(rA[j]), static_cast<std::uint32_t>(j));
    }
    BATCHLAS_DLA_INLINE void scale(int j, R akk, bool act) const {
        namespace sd = ::batchlas::sycl_device;
        const R dkk = act ? sycl::sqrt(akk) : R(1);
        const R rinv = R(1) / dkk;  // NOT rsqrt: rsqrt.approx is not the reference
        rA[j] = ::batchlas::device::select(
            act, ::batchlas::device::select(lane == j, sd::dev_from_real<D>(dkk), sd::dev_mul_real(rA[j], rinv)),
            rA[j]);
    }
    BATCHLAS_DLA_INLINE void update(int j, bool act) const {
        namespace sd = ::batchlas::sycl_device;
        if constexpr (std::is_same_v<Bcast, ::batchlas::device::bcast_policy::Shuffle>) {
BATCHLAS_UNROLL_FULL
            for (int k = j + 1; k < NC; ++k) {
                if constexpr (skips) {
                    if (k >= n) continue;
                }
                const D vk = part.bcast(rA[j], static_cast<std::uint32_t>(k));  // guard inside
                const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                rA[k] = ::batchlas::device::select(act && k < n && k <= lane, upd, rA[k]);
            }
        } else {
            static_assert(skips, "potf2_lower: the Slm broadcast skips columns >= n");
            using R4 = tiny_r4<R>;
            constexpr int kCpx = static_cast<int>(sizeof(D) / sizeof(R));
            constexpr int kW = tiny_r4_width<D>();
            constexpr int kNV = (N + kW - 1) / kW;
            R4* const sp = &bc.slm[bc.buffer(j, kNV)];
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
                    const D vk = tiny_r4_get<D, R>(x, e);
                    const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                    rA[k] = ::batchlas::device::select(act && k < n && k <= lane, upd, rA[k]);
                }
            }
        }
    }
};

// Returns the 1-based first failing column (0: success), partition-uniform.
template <class D, int N, int NC, OrderGuard G = OrderGuard::Predicate, class Part, class Bcast>
BATCHLAS_DLA_INLINE std::int32_t potf2_lower(const Part& part,
                                             ::batchlas::device::Frag<D, NC, ::batchlas::device::RowFrag>& rA, int n,
                                             bool live, Bcast bc) {
    const Potf2RowFrag<D, N, NC, G, Part, Bcast> d{part, rA, n, bc, part.lid()};
    return ::batchlas::device::potf2<Uplo::Lower>(d, live).info;
}

}  // namespace batchlas::tiny_native
