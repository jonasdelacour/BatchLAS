// SPDX-License-Identifier: MIT
#pragma once

// The native batched POTRF CTA kernel: one matrix staged whole into local memory, factorised by a
// right-looking blocked recurrence of device LA v2 ops (potf2 / trsm / herk on ArithFma, the exact
// device_scalar.hh expression forms), touching global memory twice. evidence: docs/perf/potrf.md
// Invariants: phase barriers follow `Scope` (potrf_cta_launch_params derives it; the wrong one races
// into a plausible wrong factor); each phase barrier is top-level, outside its phase's `if`, and
// failure is a predicated skip, never a `break`; the ops' Frag and tile arrays stay in registers
// (always-inline, constant indices only). evidence: docs/perf/potrf.md#register-gate

#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/blas/device/v2/ext/potrf-cta-lpanel.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::potrf_native {

enum class PotrfScope { SubGroup, WorkGroup };

template <PotrfScope SC>
inline void potrf_phase_barrier(const sycl::nd_item<1>& it) {
    if constexpr (SC == PotrfScope::SubGroup) {
        sycl::group_barrier(it.get_sub_group());
    } else {
        sycl::group_barrier(it.get_group());
    }
}

// ONE matrix (S, diag, fail, off, Ag already offset). `upper` is a load/store transform: A = U^H U
// is the same recurrence on S(i,c) = conj(A(c,i)), so Uplo costs no extra instantiation.
template <typename D, typename R, int NB, int TS, PotrfScope SC>
inline void potrf_cta_body(const sycl::nd_item<1>& it,
                           const sycl::sub_group& sg,
                           int tid, int L, bool p1_active,
                           D* __restrict S, int lda,
                           R* __restrict diag,
                           int* __restrict fail,
                           int* __restrict off,
                           D* __restrict Ag, int ldg,
                           int n, bool upper) {
    namespace v2 = batchlas::device::v2;
    using Ar = v2::ArithFma<D>;
    const v2::DynItems ex{tid, L};
    const auto A = v2::mat<v2::Mem::Local>(S, lda, n, lda);  // lda rows: the pad rows too
    const v2::HermLower<D> Ah{Ag, ldg, upper};
    const v2::Vec<R, v2::Ext<NB>, v2::Unit, v2::Mem::Local> dg{diag, {}};
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;

    // Writes every element once, pad rows too: herk_tiles reads through its `< m2` guards.
    v2::lacpy_herm_lower<Ar>(ex, Ah, A, n);
    v2::fill(ex, dg, R(0));
    // Published once at B0, never rewritten per panel: hence TriTilePrefix's dR rescale.
    v2::tri_tile_prefix_init(ex, off, Rt0);
    if (ex.leader()) *fail = 0;

    potrf_phase_barrier<SC>(it);  // B0

    bool ok = true;
    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const int m2 = n - j - ib;
        const auto A11 = A.sub(j, j, ib, ib);
        const auto A21 = A.sub(j + ib, j, m2, ib);

        if (ok && p1_active) {  // (P1) sub-group barriers only: may run inside `if (sg_id == 0)`
            v2::potf2_sg_slm<NB, Ar>(v2::SubGroup<32>{sg}, A11, ib, dg, fail, j);
        }
        potrf_phase_barrier<SC>(it);  // B1
        ok = (*fail == 0);

        if (ok && m2 > 0) {  // (P2) L21 := A21 L11^-H; rows are independent, no barrier
            v2::trsm_rowfrag<Side::Right, Uplo::Lower, Transpose::ConjTrans, NB, Ar>(ex, A11, dg, A21, ib);
        }
        potrf_phase_barrier<SC>(it);  // B2

        if (ok && m2 > 0) {  // (P3) A22 -= L21 L21^H, lower, trimmed at tile granularity
            const int Rt = (m2 + TS - 1) / TS;
            v2::herk_tiles<Uplo::Lower, TS, Ar>(ex, v2::NegOne{}, A21, v2::One{}, A.sub(j + ib, j + ib, m2, m2),
                                                v2::TriTilePrefix{off, Rt, Rt0 - Rt});
        }
        potrf_phase_barrier<SC>(it);  // B3
    }

    // The other triangle is neither READ nor WRITTEN, which ortho.cc depends on.
    v2::lacpy_lower(ex, A, Ah, n);
}

}  // namespace batchlas::potrf_native
