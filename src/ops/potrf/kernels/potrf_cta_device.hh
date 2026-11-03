// SPDX-License-Identifier: MIT
#pragma once

// The native batched POTRF CTA kernel: one matrix staged whole into local memory, factorised by a
// right-looking blocked recurrence of device LA v2 ops (potf2 / trsm / herk on arith::Fma, the
// exact device_scalar.hh expression forms), touching global memory twice. evidence: docs/perf/potrf.md
// Invariants: phase barriers follow `Scope` (potrf_cta_launch_params derives it; the wrong one races
// into a plausible wrong factor); each phase barrier is top-level, outside its phase's `if`, and
// failure is a predicated skip, never a `break`; the ops' Frag and tile arrays stay in registers
// (inlined, constant indices only). evidence: docs/perf/potrf.md#register-gate

#include <src/ops/potrf/potrf_cta_lpanel_v2.hh>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/blas/device/v2/lapack.hh>

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
// is the same recurrence on S(i,c) = conj(A(c,i)). The scope's items are the executor: L == 32 is
// one sub-group (SubGroup scope), otherwise the whole G == 1 work-group, so its width is L.
template <typename D, typename R, int NB, int TS, PotrfScope SC>
inline void potrf_cta_body(const sycl::nd_item<1>& it,
                           const sycl::sub_group& sg,
                           bool p1_active,
                           D* __restrict S, int lda,
                           R* __restrict diag,
                           int* __restrict fail,
                           int* __restrict off,
                           D* __restrict Ag, int ldg,
                           int n, bool upper) {
    namespace v2 = batchlas::device::v2;
    const v2::RtGroup<(SC == PotrfScope::SubGroup ? v2::Scope::Sub : v2::Scope::Group)> ex(it);
    const auto A = dla::slm_mat(S, lda, n, lda);  // lda rows: the pad rows too
    const v2::Vec<R, v2::Ext<NB>, v2::Unit, v2::Mem::Local> dg{diag, {}};
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;

    // Writes every element once, pad rows too: herk_tiles reads through its `< m2` guards.
    dla::lacpy_herm_lower(ex, dla::HermLower<D>{Ag, ldg, upper}, A, n);
    v2::scal(ex, v2::Zero{}, dg);
    // Published once at B0, never rewritten per panel: hence TriTilePrefix's dR rescale.
    dla::tri_tile_prefix_init(ex, off, Rt0);
    if (ex.leader()) *fail = 0;

    potrf_phase_barrier<SC>(it);  // B0

    bool ok = true;
    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const int m2 = n - j - ib;
        const auto A11 = A.sub(j, j, ib, ib);
        const auto A21 = A.sub(j + ib, j, m2, ib);

        if (ok && p1_active) {  // (P1) sub-group barriers only: may run inside `if (sg_id == 0)`
            dla::potf2_sg_slm<NB>(v2::SubGroup<32>{sg}, A11, ib, dg, fail, j);
        }
        potrf_phase_barrier<SC>(it);  // B1
        ok = (*fail == 0);

        if (ok && m2 > 0) {  // (P2) L21 := A21 L11^-H; rows are independent, no barrier
            dla::trsm_rowfrag<Side::Right, Uplo::Lower, Transpose::ConjTrans, NB>(ex, A11, dg, A21, ib);
        }
        potrf_phase_barrier<SC>(it);  // B2

        if (ok && m2 > 0) {  // (P3) A22 -= L21 L21^H, lower, trimmed at tile granularity
            const int Rt = (m2 + TS - 1) / TS;
            dla::herk_tiles<Uplo::Lower, TS>(ex, v2::NegOne{}, A21, v2::One{}, A.sub(j + ib, j + ib, m2, m2),
                                             dla::TriTilePrefix{off, Rt, Rt0 - Rt});
        }
        potrf_phase_barrier<SC>(it);  // B3
    }

    // The other triangle is neither READ nor WRITTEN, which ortho.cc depends on.
    const auto G = v2::mat_wide(Ag, n, n, ldg);
    if (upper) v2::lacpy<v2::Part::Upper, Transpose::ConjTrans>(ex, A.sub(0, 0, n, n), G);
    else v2::lacpy<v2::Part::Lower>(ex, A.sub(0, 0, n, n), G);
}

}  // namespace batchlas::potrf_native
