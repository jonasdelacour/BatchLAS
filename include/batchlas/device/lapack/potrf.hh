// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xPOTRF / xPOTF2 inside a kernel (v22 design A.3, C.4-C.7). Public entry:
//   int potrf<UL, Tier>(ex, A, ws)               A = L L^H (Lower) or U^H U (Upper) in place;
//                                               returns LAPACK info, uniform over ex
//   WorkspaceSize potrf_buffer_size<UL, Tier, T>(n)
//   tier::Reg<N>  ex = Partition<N> (one matrix per N lanes of a 32-wide sub-group, lane r row r)
//   tier::Cta<N>  ex = WorkGroup<WG, 32>, WG = tier_info<...>::wg_size or any multiple of 32;
//                 the matrix in local memory (ws.local, potrf_buffer_size bytes)
//   tier::Blocked<NB>  ex = WorkGroup<256, 32>, any n: A stays in global memory; per NB-wide
//                 panel potf2 (SgSlm) on the local diagonal block, then the in-group L3 trsm and
//                 herk (level3/group.hh); ws.local = L3 stages + the block, no global bytes
// The host tiny and CTA routes launch the SAME bodies (potrf_reg_run, potrf_cta_run) with a run-time uplo.
// Collective: every item of ex calls with the same arguments; the op ends with a barrier (the
// output is visible to every item). info = k > 0: the leading minor of order k is not positive
// definite; -2: n > the tier's bound or A not square; -3: ws.local missing or misaligned.
//
// potf2 is ONE column loop owning LAPACK's control: the pivot test `!(akk > 0)` (NaN fails),
// sticky, so the first failing column k records info = k + 1 and every later column stops
// changing. Per column k it asks the distribution: skip(k) (Dist::skips: past a uniform
// run-time order, no collective), col_live(k), pivot(k) -> Re A(k, k) on every item,
// scale(k, akk, act), update(k, act), optional fail(k). Dist::static_cols > 0 is a compile-time
// column count (unrolled register forms), 0 a run-time d.cols(). The control stays here, not in
// a hook: hook state lives in memory until inlined, and the tiny tiers' SASS is pinned to the
// form simplified with SSA state. Distributions:
//   RegRows      Partition, lane r holds row r in a RowFrag; columns by shuffle or SLM vectors
//   SgSlm        SubGroup, lane r holds row r in a Frag, column k published to local memory
//   PanelRows    any executor, one item per row of a local n x kb panel, caller's barrier

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::device {

namespace detail {
struct Potf2Result {
    std::int32_t info;  // 1-based first failing column, 0: success
    bool alive;         // live and no failure
};
}  // namespace detail

// Column k. A macro, not a function: the sticky state must stay SSA in the function that holds
// the loop (a by-value step function changes PotrfTiny's SASS), and the two loop headers differ
// only in the unroll pragma (unroll_count(1) on the run-time loop would add PTX "nounroll").
#define BATCHLAS_POTF2_COLUMN(k)                                                          \
    if constexpr (Dist::skips) {                                                          \
        if (d.skip(k)) continue;                                                          \
    }                                                                                     \
    const bool col_live = d.col_live(k);                                                  \
    const R akk = d.pivot(k);                                                             \
    const bool bad = !(akk > R(0));                                                       \
    if (alive && col_live && bad) {                                                       \
        info = k + 1;                                                                     \
        alive = false;                                                                    \
        if constexpr (requires { d.fail(k); }) d.fail(k); /* a Dist that records info */  \
    }                                                                                     \
    const bool act = alive && col_live;                                                   \
    d.scale(k, akk, act);                                                                 \
    d.update(k, act);

template <Uplo UL, class Dist> BATCHLAS_DEVICE_INLINE detail::Potf2Result potf2(const Dist& d, bool live = true) {
    static_assert(UL == Uplo::Lower, "potf2: only Uplo::Lower (an Upper caller factors the conjugate transpose)");
    using R = typename Dist::R;
    bool alive = live;
    std::int32_t info = 0;
    if constexpr (Dist::static_cols > 0) {
BATCHLAS_UNROLL_FULL
        for (int k = 0; k < Dist::static_cols; ++k) {
            BATCHLAS_POTF2_COLUMN(k)
        }
    } else {
        for (int k = 0; k < d.cols(); ++k) {
            BATCHLAS_POTF2_COLUMN(k)
        }
    }
    return {info, alive};
}
#undef BATCHLAS_POTF2_COLUMN

namespace detail {

// potf2's Rsqrt policy for the SLM forms: core inv_sqrt (rsqrt.approx + one Newton step on NVPTX
// float, decision 10).
struct InvSqrt {
    template <class R> BATCHLAS_DEVICE_INLINE static R inv_sqrt(R a, R sqrt_a) { return ::batchlas::device::inv_sqrt(a, sqrt_a); }
};

// ---- the register tier: one matrix per Partition<N>, lane r holds row r of L ---------------
// How the order guard j < n acts. Predicate: every column runs, the effects are masked (n may
// vary across the sub-group). Skip: columns j >= n skip their collectives, legal only when n is
// uniform over the sub-group.
enum class OrderGuard { Predicate, Skip };

// The register tier's scalar steps: arith::Fma's values bit for bit, in the shape the tiny
// kernels' SASS is pinned to. Each is a PLAIN inline call (folded early, the SLM vector load
// splits into scalar LDS) and reads complex fields directly (a copy into Arith's by-value operand
// reorders the field reads, and with them the scale and the Upper load).
template <class D> struct RegScalar {
    using A = arith_t<arith::Fma, D>;
    using R = real_t<D>;
    static R re(D a) {
        if constexpr (A::is_complex) return a.re;
        else return a;
    }
    static D from_real(R x) { return A::make(x, R(0)); }
    static D mul_real(D a, R s) {  // a * s, a first (the hand form's operand order)
        if constexpr (A::is_complex) return A::make(a.re * s, a.im * s);
        else return a * s;
    }
    static D mul(D a, D b) {  // A::mul: two fma
        if constexpr (A::is_complex) return A::make(sycl::fma(a.re, b.re, -a.im * b.im), sycl::fma(a.re, b.im, a.im * b.re));
        else return a * b;
    }
    static D sub(D a, D b) {
        if constexpr (A::is_complex) return A::make(a.re - b.re, a.im - b.im);
        else return a - b;
    }
    static D conj(D a) {
        if constexpr (A::is_complex) return A::make(a.re, -a.im);
        else return a;
    }
    template <class V> static D vec_elem(V x, int e) {  // e: compile-time once unrolled
        if constexpr (A::is_complex) return A::make(e == 0 ? x.x : x.z, e == 0 ? x.y : x.w);
        else return e == 0 ? x.x : e == 1 ? x.y : e == 2 ? x.z : x.w;
    }
};

// Bcast (bcast_policy): how column j of L reaches the lanes below the pivot. The failure effects
// are value substitutions, never control flow, because `live` is partition-uniform but not
// sub-group-uniform. rA[k] for k > lane stays exactly zero (the `k <= lane` predicate), so the
// identity padding stays inert. Scale is a * (1 / L(j, j)).
template <class D, int N, int NC, OrderGuard G, class Part, class Bcast> struct RegRows {
    using A = arith_t<arith::Fma, D>;
    using S = RegScalar<D>;
    using R = real_t<D>;
    static constexpr int static_cols = NC;
    static constexpr bool skips = (G == OrderGuard::Skip);
    const Part& part;
    Frag<D, NC, RowFrag>& rA;
    int n;
    const Bcast& bc;
    int lane;

    BATCHLAS_DEVICE_INLINE bool skip(int j) const { return j >= n; }
    BATCHLAS_DEVICE_INLINE bool col_live(int j) const { return j < n; }
    BATCHLAS_DEVICE_INLINE R pivot(int j) const { return part.bcast(S::re(rA[j]), static_cast<std::uint32_t>(j)); }
    BATCHLAS_DEVICE_INLINE void scale(int j, R akk, bool act) const {
        const R dkk = act ? sycl::sqrt(akk) : R(1);
        const R rinv = ::batchlas::device::inv_sqrt<InvSqrtEdge::Exact>(akk, dkk);  // NOT rsqrt: rsqrt.approx is not the reference
        rA[j] = ::batchlas::device::select(act, ::batchlas::device::select(lane == j, S::from_real(dkk), S::mul_real(rA[j], rinv)), rA[j]);
    }
    BATCHLAS_DEVICE_INLINE void update(int j, bool act) const {
        if constexpr (std::is_same_v<Bcast, bcast_policy::Shuffle>) {
BATCHLAS_UNROLL_FULL
            for (int k = j + 1; k < NC; ++k) {
                if constexpr (skips) {
                    if (k >= n) continue;
                }
                const D vk = part.bcast(rA[j], static_cast<std::uint32_t>(k));  // guard inside
                const D upd = S::sub(rA[k], S::mul(rA[j], S::conj(vk)));
                rA[k] = ::batchlas::device::select(act && k < n && k <= lane, upd, rA[k]);
            }
        } else {
            static_assert(skips, "potrf: the Slm broadcast skips columns >= n");
            using R4 = std::remove_cvref_t<decltype(bc.slm[0])>;  // a 16-byte vector of R
            constexpr int kCpx = static_cast<int>(sizeof(D) / sizeof(R));
            constexpr int kW = A::is_complex ? 2 : 4;
            constexpr int kNV = (N + kW - 1) / kW;
            R4* const sp = &bc.slm[bc.buffer(j, kNV)];
            R* const rp = reinterpret_cast<R*>(sp);
            if constexpr (kCpx == 2) {
                rp[2 * lane] = rA[j].re;  // the fields, as RegScalar reads them
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
                    const D vk = S::vec_elem(x, e);
                    const D upd = S::sub(rA[k], S::mul(rA[j], S::conj(vk)));
                    rA[k] = ::batchlas::device::select(act && k < n && k <= lane, upd, rA[k]);
                }
            }
        }
    }
};

// potf2 on the RowFrag rA; returns the 1-based first failing column (0: success), partition-uniform.
template <class D, int N, int NC, OrderGuard G = OrderGuard::Predicate, class Part, class Bcast>
BATCHLAS_DEVICE_INLINE std::int32_t potf2_reg(const Part& part, Frag<D, NC, RowFrag>& rA, int n, bool live, Bcast bc) {
    const RegRows<D, N, NC, G, Part, Bcast> d{part, rA, n, bc, part.lid()};
    return potf2<Uplo::Lower>(d, live).info;
}

// Row `lane` of the stored triangle into rA (pointer, never a reference to the Frag). A row that
// is not `row_live` carries the IDENTITY, which lets the unrolled bodies run with no lane guard;
// `c <= lane` keeps the other triangle unread (ortho.cc's contract). Upper is a LOAD TRANSFORM:
// A = U^H U is the same recurrence on S(i, c) = conj(A(c, i)). real_diag: Im(diag) is ignored.
template <class D, int N> inline void reg_pad_identity(D* rA, int lane) {
    const D one = Arith<D>::one();
    const D zero = D{};
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) rA[c] = ::batchlas::device::select(c == lane, one, zero);
}
template <class D, int N>
inline void reg_load_lower(D* rA, const D* __restrict a, int ld, int lane, bool row_live, bool real_diag) {
    using A = Arith<D>;
    reg_pad_identity<D, N>(rA, lane);
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) {
            D v = a[lane + static_cast<std::ptrdiff_t>(c) * ld];
            if (real_diag && c == lane) v = A::make(A::re(v), real_t<D>(0));
            rA[c] = v;
        }
    }
}
template <class D, int N>
inline void reg_load_upper(D* rA, const D* __restrict a, int ld, int lane, bool row_live, bool real_diag) {
    using A = Arith<D>;
    reg_pad_identity<D, N>(rA, lane);
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) {
            D v = RegScalar<D>::conj(a[c + static_cast<std::ptrdiff_t>(lane) * ld]);
            if (real_diag && c == lane) v = A::make(A::re(v), real_t<D>(0));
            rA[c] = v;
        }
    }
}
template <class D, int N> inline void reg_store_lower(const D* rA, D* __restrict a, int ld, int lane, bool row_live) {
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) a[lane + static_cast<std::ptrdiff_t>(c) * ld] = rA[c];
    }
}
template <class D, int N> inline void reg_store_upper(const D* rA, D* __restrict a, int ld, int lane, bool row_live) {
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) a[c + static_cast<std::ptrdiff_t>(lane) * ld] = RegScalar<D>::conj(rA[c]);
    }
}

// The tier's body: load, potf2, store of the matrix at a (order n <= N, leading dimension ld) on
// the partition `part` (lane = its lane, live = this partition's matrix exists). Upper at run
// time: the host route launches one kernel for both triangles.
template <class D, int N, class Part>
BATCHLAS_DEVICE_INLINE std::int32_t potrf_reg_run(const Part& part, D* a, int ld, int lane, int n, bool live, bool upper) {
    constexpr bool real_diag = Arith<D>::is_complex;
    const bool row_live = live && (lane < n);
    Frag<D, N> rA;
    if (upper) reg_load_upper<D, N>(rA.v, a, ld, lane, row_live, real_diag);
    else reg_load_lower<D, N>(rA.v, a, ld, lane, row_live, real_diag);
    const std::int32_t linfo = potf2_reg<D, N, N>(part, rA, n, live, bcast_policy::Shuffle{});
    if (upper) reg_store_upper<D, N>(rA.v, a, ld, lane, row_live);
    else reg_store_lower<D, N>(rA.v, a, ld, lane, row_live);
    return linfo;
}

// ---- the SLM forms: potf2 on one sub-group (CTA) and on a row-per-item panel (LPANEL) --------
// The ib x ib block `a` (ib <= NB <= SG) on one sub-group: lane r holds row r in `d`; column k is
// written to `a` (local) and read back after one sub-group barrier; diag(k) := L(k, k). The NB
// barriers are unconditional.
template <int NB, class Rsqrt, int SG, class MA, class DV> struct SgSlm {
    using D = typename MA::value_type;
    using A = arith_t<arith::Fma, D>;
    using R = real_t<D>;
    static constexpr int static_cols = NB;
    static constexpr bool skips = false;
    const SubGroup<SG>& ex;
    const MA& a;
    int ib;
    const DV& diag;
    int* info;
    int info_base;
    int lane;
    Frag<D, NB>& d;

    BATCHLAS_DEVICE_INLINE bool col_live(int k) const { return k < ib; }
    BATCHLAS_DEVICE_INLINE void fail(int k) const {
        if (lane == 0) *info = info_base + k + 1;
    }
    BATCHLAS_DEVICE_INLINE R pivot(int k) const {
        return sycl::select_from_group(ex.sg, A::re(d[k]), static_cast<std::uint32_t>(k));
    }
    BATCHLAS_DEVICE_INLINE void scale(int k, R akk, bool active) const {
        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = Rsqrt::inv_sqrt(active ? akk : R(1), dkk);
        if (active) {
            if (lane == k) d[k] = A::make(dkk, R(0));
            else if (lane > k && lane < ib) d[k] = A::scale_real(r, d[k]);
            if (lane < ib && lane >= k) a(lane, k) = d[k];
            if (lane == 0) diag(k) = dkk;
        }
        sycl::group_barrier(ex.sg);
    }
    BATCHLAS_DEVICE_INLINE void update(int k, bool active) const {
        if (active) {
            BATCHLAS_UNROLL_FULL
            for (int c = k + 1; c < NB; ++c) {
                if (c < ib && lane < ib && lane >= c) d[c] = A::sub(d[c], A::mul(d[k], A::conj(a(c, k))));
            }
        }
    }
};

// The first bad pivot sets *info := info_base + k + 1 (lane 0, SgSlm::fail).
template <int NB, class Rsqrt, int SG, class MA, class DV>
BATCHLAS_DEVICE_HELPER void potf2_sg_slm_body(SubGroup<SG> ex, MA a, int ib, DV diag, int* info, int info_base) {
    using D = typename MA::value_type;
    static_assert(NB <= SG, "potf2_sg_slm: one lane per row");
    const int lane = ex.lid();
    Frag<D, NB> d;
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < NB; ++c) d[c] = (lane < ib && c < ib && c <= lane) ? a(lane, c) : D{};
    const SgSlm<NB, Rsqrt, SG, MA, DV> dist{ex, a, ib, diag, info, info_base, lane, d};
    potf2<Uplo::Lower>(dist);
}

// The n x kb panel P whose diagonal block starts at row c0, one item per row (row = ex.lid();
// rows outside (c0, n) idle). `bar` is the caller's barrier over every item reading P, three per
// column: pivot read / its overwrite, scale, rank-1 update.
template <class Rsqrt, class PM, class Bar> struct PanelRows {
    using D = typename PM::value_type;
    using A = arith_t<arith::Fma, D>;
    using R = real_t<D>;
    static constexpr int static_cols = 0;
    static constexpr bool skips = false;
    const PM& p;
    int c0;
    int kb;
    int n;
    Bar& bar;
    int row;
    bool leader;
    int* info;

    BATCHLAS_DEVICE_INLINE int cols() const { return kb; }
    BATCHLAS_DEVICE_INLINE bool col_live(int) const { return true; }
    BATCHLAS_DEVICE_INLINE void fail(int i) const {
        if (leader) *info = c0 + i + 1;
    }
    BATCHLAS_DEVICE_INLINE R pivot(int i) const { return A::re(p(c0 + i, i)); }
    BATCHLAS_DEVICE_INLINE void scale(int i, R d, bool act) const {
        bar();  // WAR: every item read the pivot cell above, item c0 + i overwrites it below
        if (act) {
            const R dkk = sycl::sqrt(d);
            const R rs = Rsqrt::inv_sqrt(d, dkk);  // act: d > 0, not NaN
            if (row == c0 + i) p(row, i) = A::make(dkk, R(0));
            else if (row > c0 + i && row < n) p(row, i) = A::scale_real(rs, p(row, i));
        }
        bar();
    }
    BATCHLAS_DEVICE_INLINE void update(int i, bool act) const {
        if (row > c0 + i && act && row < n) {  // this operand order keeps PotrfLpanel's SASS
            const D lri = p(row, i);
            for (int c = i + 1; c < kb; ++c) {
                const D ljc = p(c0 + c, i);
                p(row, c) = A::sub(p(row, c), A::mul(lri, A::conj(ljc)));
            }
        }
        bar();
    }
};
}  // namespace detail

// potf2 of the ib x ib local block `a` on one sub-group (SgSlm above). Real T: the body is a
// by-value phase op the inliner takes after simplifying it alone (its column-0..3 reciprocal
// shapes depend on it). Complex T: inlined at the call; out of line the by-value operands spill.
template <int NB, class Rsqrt, int SG, class MA, class DV>
BATCHLAS_DEVICE_INLINE void potf2_sg_slm(const SubGroup<SG>& ex, const MA& a, int ib, const DV& diag, int* info,
                                      int info_base) {
    if constexpr (arith_t<arith::Fma, typename MA::value_type>::is_complex) {
        [[clang::always_inline]] detail::potf2_sg_slm_body<NB, Rsqrt>(ex, a, ib, diag, info, info_base);
    } else {
        detail::potf2_sg_slm_body<NB, Rsqrt>(ex, a, ib, diag, info, info_base);
    }
}

// potf2 of the panel P, one item per row (PanelRows above). `ok` is sticky: returns it, and the
// first bad pivot sets *info := c0 + i + 1 (leader).
template <class Rsqrt, class Ex, class PM, class Bar>
BATCHLAS_DEVICE_INLINE bool potf2_panel_rows(const Ex& ex, const PM& p, int c0, int kb, int n, bool ok, int* info,
                                          Bar&& bar) {
    using B = std::remove_reference_t<Bar>;
    const detail::PanelRows<Rsqrt, PM, B> dist{p, c0, kb, n, bar, ex.lid(), ex.leader(), info};
    return potf2<Uplo::Lower>(dist, ok).alive;
}

// ---- the CTA (right-looking, SLM-resident) and LPANEL (left-looking) recurrences' pieces -------
// Every op reproduces the expression tree of the hand-written code it replaced: arith::Fma is
// bit-equal to the src/sycl/device_scalar.hh forms, and sub(mul()) stays two roundings.
// The BLAS / auxiliary ones (herk_tiles, trsm_rowfrag, gemv_n_slm_bcast, lacpy_*, HermLower,
// slm_mat, TriTilePrefix, frag_*_row_herm_lower) wait here for their core forms (v22 core requests).
namespace detail {

// A Hermitian matrix read through its lower triangle (i >= c), uplo at run time: p[i + c ld], or
// conj(p[c + i ld]) when upper. A = U^H U is the lower recurrence on this view.
template <class T> struct HermLower {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int ld;
    bool upper;
    BATCHLAS_DEVICE_INLINE value_type operator()(int i, int c) const {
        return upper ? arith_t<arith::Fma, value_type>::conj(p[c + static_cast<std::ptrdiff_t>(i) * ld]) : p[i + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// A local tile through a local-space pointer, so every access stays LDS/STS even where a body is
// not inlined (the slot's tile is &acc[0] + slot * size: core's accessor base cannot carry it).
template <class T> using local_ptr_t = sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>;
template <class T> using SlmMat = Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, local_ptr_t<T>, std::ptrdiff_t>;
template <int M, int N, class T> using SlmTile = Mat<T, Ext<M>, Ext<N>, Ext<M>, Mem::Local, Cyclic, local_ptr_t<T>, int>;

template <class T> BATCHLAS_DEVICE_INLINE local_ptr_t<T> local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <class T> BATCHLAS_DEVICE_INLINE SlmMat<T> slm_mat(T* p, int m, int n, int ld) {
    return {local_ptr(p), Dyn{m}, Dyn{n}, Dyn{ld}};
}
template <int M, int N, class T> BATCHLAS_DEVICE_INLINE SlmTile<M, N, T> slm_tile(T* p) { return {local_ptr(p), {}, {}, {}}; }

// dst(i, c) := src(i, c) for c <= i < n, diagonal made real (Hermitian input), and 0 in every
// other row i < dst.rows(): the whole dst.rows() x n block (pad rows too) is written once.
template <class Ex, class MS, class MD> BATCHLAS_DEVICE_INLINE void lacpy_herm_lower(const Ex& ex, const MS& src, const MD& dst, int n) {
    using D = typename MD::value_type;
    using A = arith_t<arith::Fma, D>;
    for (int c = 0; c < n; ++c) {
        for_each_owned(ex, dst.rows(), [&](int i) BATCHLAS_DEVICE_BODY {
            D v{};
            if (i < n && i >= c) {
                v = src(i, c);
                if (i == c) v = A::make(A::re(v), real_t<D>(0));
            }
            dst(i, c) = v;
        });
    }
}

// dst := conj(src) zero-padded to dst's static NB x NB (src r x c, r, c <= NB); the NB*NB
// elements are distributed flat (e -> (e mod NB, e / NB)).
template <int NB, class Ex, class MS, class MD> BATCHLAS_DEVICE_INLINE void lacpy_conj_pad(const Ex& ex, const MS& src, const MD& dst) {
    using D = typename MD::value_type;
    const int r = src.rows(), c = src.cols();
    for_each_owned(ex, NB * NB, [&](int e) BATCHLAS_DEVICE_BODY {
        const int i = e % NB;
        const int kk = e / NB;
        D v{};
        if (i < r && kk < c) v = arith_t<arith::Fma, D>::conj(src(i, kk));
        dst(i, kk) = v;
    });
}

// X := X L^{-H} for the m x kb block X, one row at a time in a Frag<NB> (rows Cyclic over ex, no
// barrier); L is kb x kb lower (kb <= NB) whose diagonal is read from the real vector `ldiag`.
// Divides by the diagonal, as reference trsm does (potf2 scales by a reciprocal).
template <Side S, Uplo UL, Transpose TR, int NB, class Ex, class ML, class DV, class MX>
BATCHLAS_DEVICE_HELPER void trsm_rowfrag(const Ex& ex, const ML& l, const DV& ldiag, const MX& x, int kb) {
    static_assert(S == Side::Right && UL == Uplo::Lower && TR == Transpose::ConjTrans,
                  "trsm_rowfrag: only Right / Lower / ConjTrans is implemented");
    using D = typename MX::value_type;
    using A = arith_t<arith::Fma, D>;
    const int m = x.rows();
    for (int row = ex.lid(); row < m; row += detail::size_of(ex)) {
        Frag<D, NB> xr;
        load_row(xr, x, row, kb);
        BATCHLAS_UNROLL_FULL
        for (int c = 0; c < NB; ++c) {
            if (c < kb) {
                D s = xr[c];
                BATCHLAS_UNROLL_FULL
                for (int q = 0; q < NB; ++q) {
                    if (q < c) s = A::sub(s, A::mul(xr[q], A::conj(l(c, q))));
                }
                xr[c] = A::div_real(s, ldiag(c));
            }
        }
        store_row(xr, x, row, kb);
    }
}

// Column-major enumeration of the lower tiles (rt >= ct) of an Rt x Rt tile grid through a prefix
// table off[c] = c Rt0 - c(c-1)/2 built once for Rt0 >= Rt and rescaled by dR = Rt0 - Rt, so a
// shrinking trailing matrix reuses it. Decoding is a binary search over the tile columns.
struct TileRC {
    int rt;
    int ct;
};
struct TriTilePrefix {
    const int* off;
    int Rt;
    int dR;
    BATCHLAS_DEVICE_INLINE int count() const { return Rt * (Rt + 1) / 2; }
    BATCHLAS_DEVICE_INLINE TileRC operator()(int t) const {
        int lo = 0, hi = Rt - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (off[mid] - mid * dR <= t) lo = mid;
            else hi = mid - 1;
        }
        return {lo + (t - (off[lo] - lo * dR)), lo};
    }
};
template <class Ex> BATCHLAS_DEVICE_INLINE void tri_tile_prefix_init(const Ex& ex, int* off, int Rt0) {
    for (int c = ex.lid(); c <= Rt0; c += detail::size_of(ex)) off[c] = c * Rt0 - ((c * (c - 1)) >> 1);
}

// C := C - A A^H on the lower triangle (A m x k, C m x m) in TS x TS register tiles distributed
// Cyclic over ex in map's order; diagonal tiles are trimmed element-wise (r >= c), so the strict
// upper triangle is never written. Complex: the diagonal is stored real (LAPACK herk). No barrier.
template <Uplo UL, int TS, class Ex, class AS, class MA, class BS, class MC, class Map>
BATCHLAS_DEVICE_HELPER void herk_tiles(const Ex& ex, AS, const MA& a, BS, const MC& c, const Map& map) {
    static_assert(UL == Uplo::Lower, "herk_tiles: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<AS, NegOne> && std::is_same_v<BS, One>, "herk_tiles: alpha = NegOne, beta = One");
    using D = typename MC::value_type;
    using A = arith_t<arith::Fma, D>;
    const int m = a.rows(), k = a.cols();
    const int ntiles = map.count();
    for (int t = ex.lid(); t < ntiles; t += detail::size_of(ex)) {
        const TileRC rc = map(t);
        const int r0 = rc.rt * TS;
        const int c0 = rc.ct * TS;
        D acc[TS][TS];
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) acc[i][j] = D{};
        }
        for (int p = 0; p < k; ++p) {
            D va[TS], vb[TS];
            BATCHLAS_UNROLL_FULL
            for (int i = 0; i < TS; ++i) va[i] = (r0 + i < m) ? a(r0 + i, p) : D{};
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) vb[j] = (c0 + j < m) ? a(c0 + j, p) : D{};
            BATCHLAS_UNROLL_FULL
            for (int i = 0; i < TS; ++i) {
                BATCHLAS_UNROLL_FULL
                for (int j = 0; j < TS; ++j) A::mac(acc[i][j], va[i], A::conj(vb[j]));
            }
        }
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TS; ++i) {
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TS; ++j) {
                const int ri = r0 + i;
                const int cj = c0 + j;
                if (ri < m && cj < m && ri >= cj) {
                    D v = A::sub(c(ri, cj), acc[i][j]);
                    if constexpr (A::is_complex) {
                        if (ri == cj) v = A::make(A::re(v), real_t<D>(0));
                    }
                    c(ri, cj) = v;
                }
            }
        }
    }
}

// f(i) := a(row, c0 + i) where live, i < kb and the element is in the lower triangle (row >= c0+i),
// diagonal made real; 0 elsewhere (the strict upper triangle is never read).
template <int NB, class MA, class F> BATCHLAS_DEVICE_INLINE void frag_load_row_herm_lower(F& f, const MA& a, int row, int c0, int kb, bool live) {
    using D = typename MA::value_type;
    using A = arith_t<arith::Fma, D>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        D v{};
        if (live && i < kb && row >= c0 + i) {
            v = a(row, c0 + i);
            if (row == c0 + i) v = A::make(A::re(v), real_t<D>(0));
        }
        f[i] = v;
    }
}

// p(row, i) := g(i) for i < kb where row >= c0 + i (diagonal made real), 0 above the diagonal:
// column i of p is global column c0 + i of a lower-stored Hermitian block.
template <int NB, class PM, class G> BATCHLAS_DEVICE_INLINE void frag_store_row_herm_lower(const PM& p, int row, int c0, int kb, G&& g) {
    using D = typename PM::value_type;
    using A = arith_t<arith::Fma, D>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < NB; ++i) {
        if (i < kb) {
            D v{};
            if (row >= c0 + i) {
                v = g(i);
                if (row == c0 + i) v = A::make(A::re(v), real_t<D>(0));  // a mac(x, conj(x)) residue
            }
            p(row, i) = v;
        }
    }
}

// y(i) += sum_kk B(i, kk) x(kk) for the NB x NB column-major tile B in local memory, read as
// 16-byte vectors (every item reads the same column: a broadcast); each y(i) sums kk ascending.
template <int NB, class D, class FY, class FX> BATCHLAS_DEVICE_INLINE void gemv_n_slm_bcast(FY& y, const FX& x, D* b) {
    using V = sycl::vec<float, 4>;
    constexpr int VW = 16 / static_cast<int>(sizeof(D));
    static_assert((NB * sizeof(D)) % 16 == 0, "gemv_n_slm_bcast: a column must be whole 16-byte vectors");
    const auto bv = local_ptr(reinterpret_cast<V*>(b));
    BATCHLAS_UNROLL_FULL
    for (int kk = 0; kk < NB; ++kk) {
        D col[NB];
        BATCHLAS_UNROLL_FULL
        for (int v = 0; v < NB / VW; ++v) {
            const V t = bv[(kk * NB) / VW + v];
            __builtin_memcpy(&col[v * VW], &t, 16);
        }
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < NB; ++i) arith_t<arith::Fma, D>::mac(y[i], x[kk], col[i]);
    }
}

// ---- the CTA tier body -------------------------------------------------------------------------
// One matrix staged whole into local memory, factorised by a right-looking blocked recurrence
// (potf2 on sub-group 0, trsm_rowfrag, herk_tiles), touching global memory twice. Phase barriers
// follow SC (Sub: one sub-group per matrix; Group: the whole work-group; the wrong one races
// into a plausible wrong factor); each is top-level, outside its phase's `if`, and failure is a
// predicated skip, never a `break`. evidence: docs/perf/potrf.md#register-gate
template <class D> struct CtaConst {
    static constexpr int nb = 8;  // the panel width and the d[] / x[] register arrays
    static constexpr int ts = (Arith<D>::is_complex && std::is_same_v<real_t<D>, double>) ? 2 : 4;  // the thread tile
};

// The CTA width ladder: work-items follow the first trailing update's elements (m2_0 = n - NB),
// 24 per item, 32 .. 256. evidence: docs/perf/potrf.md#the-l-ladder
inline constexpr int kCtaMaxItems = 256;
inline constexpr int kCtaElemsPerItem = 24;
constexpr int cta_items(int n, int nb, int ts) {
    const int m2_0 = (n > nb) ? (n - nb) : 0;
    const int rt0 = (m2_0 + ts - 1) / ts;
    const long long work = static_cast<long long>(rt0) * (rt0 + 1) / 2 * static_cast<long long>(ts) * ts;
    int want = 32;
    while (want < kCtaMaxItems && static_cast<long long>(want) * kCtaElemsPerItem < work) want <<= 1;
    return want;
}

// The public tier's local memory, one plan for size and carve: the lda x n tile (lda = n | 1,
// odd: conflict-free row reads), the NB diagonal, the info cell and the tile prefix table.
template <class D> struct PotrfCtaPlan {
    int lda;
    int rt0;
    ws::Slot<D> tile;
    ws::Slot<real_t<D>> diag;
    ws::Slot<int> fail;
    ws::Slot<int> off;
    WorkspaceSize size;
    static constexpr PotrfCtaPlan make(int n) {
        using C = CtaConst<D>;
        const int lda = n | 1;
        const int m2_0 = (n > C::nb) ? (n - C::nb) : 0;
        const int rt0 = (m2_0 + C::ts - 1) / C::ts;
        Plan p;
        const auto t = p.take<D>(ws::Space::Local, static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
        const auto d = p.take<real_t<D>>(ws::Space::Local, static_cast<std::size_t>(C::nb));
        const auto f = p.take<int>(ws::Space::Local, 1);
        const auto o = p.take<int>(ws::Space::Local, static_cast<std::size_t>(rt0 + 1));
        return {lda, rt0, t, d, f, o, p.size()};
    }
};

template <Scope SC> inline void cta_barrier(const sycl::nd_item<1>& it) {
    if constexpr (SC == Scope::Sub) {
        sycl::group_barrier(it.get_sub_group());
    } else {
        sycl::group_barrier(it.get_group());
    }
}

// ONE matrix (S, diag, fail, off, Ag already offset). `upper` is a load/store transform: A = U^H U
// is the same recurrence on S(i,c) = conj(A(c,i)). The scope's items are the executor.
template <typename D, typename R, int NB, int TS, Scope SC>
inline void potrf_cta_run(const sycl::nd_item<1>& it, const sycl::sub_group& sg, bool p1_active, D* __restrict S, int lda,
                     R* __restrict diag, int* __restrict fail, int* __restrict off, D* __restrict Ag, int ldg, int n,
                     bool upper) {
    const RtGroup<SC> ex(it);
    const auto A = slm_mat(S, lda, n, lda);  // lda rows: the pad rows too
    const Vec<R, Ext<NB>, Unit, Mem::Local> dg{diag, {}};
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;

    // Writes every element once, pad rows too: herk_tiles reads through its `< m2` guards.
    lacpy_herm_lower(ex, HermLower<D>{Ag, ldg, upper}, A, n);
    scal(ex, Zero{}, dg);
    // Published once at B0, never rewritten per panel: hence TriTilePrefix's dR rescale.
    tri_tile_prefix_init(ex, off, Rt0);
    if (ex.leader()) *fail = 0;

    cta_barrier<SC>(it);  // B0

    bool ok = true;
    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const int m2 = n - j - ib;
        const auto A11 = A.sub(j, j, ib, ib);
        const auto A21 = A.sub(j + ib, j, m2, ib);

        if (ok && p1_active) {  // (P1) sub-group barriers only: may run inside `if (sg_id == 0)`
            potf2_sg_slm<NB, InvSqrt>(SubGroup<32>{sg}, A11, ib, dg, fail, j);
        }
        cta_barrier<SC>(it);  // B1
        ok = (*fail == 0);

        if (ok && m2 > 0) {  // (P2) L21 := A21 L11^-H; rows are independent, no barrier
            trsm_rowfrag<Side::Right, Uplo::Lower, Transpose::ConjTrans, NB>(ex, A11, dg, A21, ib);
        }
        cta_barrier<SC>(it);  // B2

        if (ok && m2 > 0) {  // (P3) A22 -= L21 L21^H, lower, trimmed at tile granularity
            const int Rt = (m2 + TS - 1) / TS;
            herk_tiles<Uplo::Lower, TS>(ex, NegOne{}, A21, One{}, A.sub(j + ib, j + ib, m2, m2), TriTilePrefix{off, Rt, Rt0 - Rt});
        }
        cta_barrier<SC>(it);  // B3
    }

    // The other triangle is neither READ nor WRITTEN, which ortho.cc depends on.
    const auto G = mat_wide(Ag, n, n, ldg);
    if (upper) lacpy<Part::Upper, Transpose::ConjTrans>(ex, A.sub(0, 0, n, n), G);
    else lacpy<Part::Lower>(ex, A.sub(0, 0, n, n), G);
}

// ---- the public op's plumbing ------------------------------------------------------------------
// std::complex operands are re-typed to the POD Cx at the pointer boundary (no Annex-G operator*).
template <class T> struct dev_scalar {
    using type = T;
};
template <class R> struct dev_scalar<std::complex<R>> {
    using type = Cx<R>;
};
template <class T> using dev_scalar_t = typename dev_scalar<std::remove_cv_t<T>>::type;
template <class MA> BATCHLAS_DEVICE_INLINE dev_scalar_t<typename MA::value_type>* base_ptr(const MA& a) {
    return reinterpret_cast<dev_scalar_t<typename MA::value_type>*>(&a(0, 0));
}
template <class MA> BATCHLAS_DEVICE_INLINE int ld_of(const MA& a) { return static_cast<int>(a.ld); }

// The register tier's order ceiling per type (complex<double>: a 32-row RowFrag spills).
template <class T> inline constexpr int reg_max_n = (Arith<T>::is_complex && std::is_same_v<real_t<T>, double>) ? 16 : 32;


template <Uplo UL, int N, class Ex, class MA> BATCHLAS_DEVICE_INLINE int potrf_reg_op(const Ex& ex, const MA& a) {
    using T = typename MA::value_type;
    static_assert(Ex::size == N, "potrf: tier::Reg<N> runs on a Partition<N> (one lane per row)");
    static_assert(N <= reg_max_n<T>, "potrf: tier::Reg<32> does not hold a complex<double> row; use Reg<16> or Cta");
    const int n = a.rows();
    if (n < 0 || n > N || a.cols() != n) return -2;
    if (n == 0) return 0;
    const std::int32_t info = potrf_reg_run<dev_scalar_t<T>, N>(ex, base_ptr(a), ld_of(a), ex.lid(), n, true, UL == Uplo::Upper);
    ex.barrier();
    return info;
}

template <Uplo UL, int N, class Alg, int WG, int SG, class MA>
BATCHLAS_DEVICE_INLINE int potrf_cta_op(const WorkGroup<WG, SG>& ex, const MA& a, const Workspace& ws) {
    static_assert(SG == 32, "potrf: tier::Cta runs potf2 on one 32-wide sub-group");
    static_assert(std::is_same_v<Alg, alg::Default>, "potrf: tier::Cta has one algorithm");
    using D = dev_scalar_t<typename MA::value_type>;
    using C = CtaConst<D>;
    const int n = a.rows();
    if (n < 0 || n > N || a.cols() != n) return -2;
    if (n == 0) return 0;
    std::byte* const lb = ws.local.get_raw();
    if (lb == nullptr || reinterpret_cast<std::uintptr_t>(lb) % 16 != 0) return -3;
    const PotrfCtaPlan<D> pl = PotrfCtaPlan<D>::make(n);
    int* const fl = carve_local(ws, pl.fail).get_raw();
    constexpr Scope kScope = (WG == SG) ? Scope::Sub : Scope::Group;
    const bool p1_active = (WG == SG) || ex.sub().get_group_linear_id() == 0;
    potrf_cta_run<D, real_t<D>, C::nb, C::ts, kScope>(ex.it, ex.sub(), p1_active, carve_local(ws, pl.tile).get_raw(), pl.lda,
                                                 carve_local(ws, pl.diag).get_raw(), fl, carve_local(ws, pl.off).get_raw(),
                                                 base_ptr(a), ld_of(a), n, UL == Uplo::Upper);
    ex.barrier(Mem::Global);
    return *fl;
}

// ---- the Blocked tier (and potrs / posv / potri's): any n on the in-group L3 -------------------
// The L3 configuration of every Cholesky-family Blocked tier: 32 x 32 tiles, 8-deep slabs, 2 x 2
// per item, so the executor is WorkGroup<256, 32>. Compile-time from T only (design B.3).
template <class T> using PotrfBlockedCfg = l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;
inline constexpr int kBlockedWg = 256;
static_assert(PotrfBlockedCfg<float>::threads == kBlockedWg);


// One plan for size and carve. The L3 stage slot is first in every Cholesky-family plan, so
// posv's potrs carves the same bytes potrf did (potrs.hh static_asserts it).
template <class T, int NB> struct PotrfBlockedPlan {
    using D = dev_scalar_t<T>;
    ws::Slot<T> smem;     // BlockedCfg<T>'s stages
    ws::Slot<D> blk;      // the NB x NB diagonal block, ld NB
    ws::Slot<real_t<D>> diag;
    ws::Slot<int> fail;   // info of the current panel, written by lane 0 of sub-group 0
    WorkspaceSize size;
    static constexpr PotrfBlockedPlan make() {
        Plan p;
        const auto s = p.take<T>(ws::Space::Local, static_cast<std::size_t>(PotrfBlockedCfg<T>::smem_elems));
        const auto b = p.take<D>(ws::Space::Local, static_cast<std::size_t>(NB) * NB);
        const auto d = p.take<real_t<D>>(ws::Space::Local, static_cast<std::size_t>(NB));
        const auto f = p.take<int>(ws::Space::Local, 1);
        return {s, b, d, f, p.size()};
    }
};

// The Workspace a Blocked tier was handed: a 16-byte aligned local slice, and a global one when
// the plan has global bytes.
BATCHLAS_DEVICE_INLINE bool blocked_ws_ok(const Workspace& ws, bool need_global) {
    const std::byte* const lb = ws.local.get_raw();
    if (lb == nullptr || reinterpret_cast<std::uintptr_t>(lb) % 16 != 0) return false;
    return !need_global || (ws.global != nullptr && reinterpret_cast<std::uintptr_t>(ws.global) % 16 == 0);
}

// Right-looking over NB-wide panels (LAPACK xPOTRF): the diagonal block is staged Hermitian-lower
// into local memory and factored by potf2 on sub-group 0 (SgSlm, the CTA tier's leaf), written
// back, then L21 := A21 L11^-H and A22 -= L21 L21^H on the in-group L3. Upper is the transposed
// schedule: U12 := U11^-H A12, A22 -= U12^H U12. A failing panel stops the loop after its write
// back (LAPACK: no later column changes); info is local, so it is uniform.
template <Uplo UL, int NB, class Alg, int WG, int SG, class MA>
BATCHLAS_DEVICE_INLINE int potrf_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const Workspace& ws) {
    using T = typename MA::value_type;
    using D = dev_scalar_t<T>;
    using R = real_t<D>;
    using Cfg = PotrfBlockedCfg<T>;
    static_assert(std::is_same_v<Alg, alg::Default>, "potrf: tier::Blocked has one algorithm");
    static_assert(NB >= 1 && NB <= reg_max_n<T>, "potrf: tier::Blocked<NB> needs 1 <= NB <= 32 (complex<double>: 16)");
    static_assert(WG == Cfg::threads && SG == 32, "potrf: tier::Blocked runs on WorkGroup<256, 32>");
    constexpr bool upper = UL == Uplo::Upper;
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (n == 0) return 0;
    if (!blocked_ws_ok(ws, false)) return -3;
    constexpr PotrfBlockedPlan<T, NB> pl = PotrfBlockedPlan<T, NB>::make();
    const l3::Smem<Cfg> sm{carve_local(ws, pl.smem).get_raw()};
    D* const S = carve_local(ws, pl.blk).get_raw();
    int* const fl = carve_local(ws, pl.fail).get_raw();
    const Vec<R, Ext<NB>, Unit, Mem::Local> dg{carve_local(ws, pl.diag).get_raw(), {}};
    T* const p = &a(0, 0);
    const int ld = static_cast<int>(a.ld);
    const auto G = mat(p, n, n, ld);
    const auto Gd = mat(reinterpret_cast<D*>(p), n, n, ld);
    const bool sg0 = ex.sub().get_group_linear_id() == 0;
    if (ex.leader()) *fl = 0;
    int info = 0;
    for (int j = 0; j < n; j += NB) {
        const int ib = n - j < NB ? n - j : NB;
        const int m2 = n - j - ib;
        const auto S11 = slm_mat(S, ib, ib, NB);
        lacpy_herm_lower(ex, HermLower<D>{&Gd(j, j), ld, upper}, slm_mat(S, NB, NB, NB), ib);
        ex.barrier();
        if (sg0) potf2_sg_slm<NB, InvSqrt>(SubGroup<32>{ex.sub()}, S11, ib, dg, fl, j);
        ex.barrier();
        info = *fl;
        if constexpr (upper) lacpy<Part::Upper, Transpose::ConjTrans>(ex, S11, Gd.sub(j, j, ib, ib));
        else lacpy<Part::Lower>(ex, S11, Gd.sub(j, j, ib, ib));
        ex.barrier(Mem::Global);
        if (info != 0 || m2 == 0) break;
        if constexpr (upper) {
            const auto U12 = G.sub(j, j + ib, ib, m2);
            trsm<Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, G.sub(j, j, ib, ib), U12, One{}, sm);
            herk<Uplo::Upper, Transpose::ConjTrans, Cfg>(ex, U12, G.sub(j + ib, j + ib, m2, m2), NegOne{}, One{}, sm);
        } else {
            const auto L21 = G.sub(j + ib, j, m2, ib);
            trsm<Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, G.sub(j, j, ib, ib), L21, One{}, sm);
            herk<Uplo::Lower, Transpose::NoTrans, Cfg>(ex, L21, G.sub(j + ib, j + ib, m2, m2), NegOne{}, One{}, sm);
        }
    }
    return info;
}
}  // namespace detail

// ---- the public op -----------------------------------------------------------------------------
namespace routine {
template <Uplo UL> struct potrf {
    static constexpr const char* name = "potrf";
    static constexpr Uplo uplo = UL;
};
}  // namespace routine

template <Uplo UL, int N, bool Exact, class T> struct tier_info<routine::potrf<UL>, tier::Reg<N, Exact>, T> {
    static constexpr int wg_size = 32;  // one sub-group: 32 / N matrices
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && n <= N && N <= detail::reg_max_n<T>; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <Uplo UL, int N, class Alg, class T> struct tier_info<routine::potrf<UL>, tier::Cta<N, Alg>, T> {
    using D = detail::dev_scalar_t<T>;
    static constexpr int wg_size = detail::cta_items(N, detail::CtaConst<D>::nb, detail::CtaConst<D>::ts);
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::PotrfCtaPlan<D>::make(n).size; }
};
template <Uplo UL, int NB, class Alg, class T> struct tier_info<routine::potrf<UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && NB <= detail::reg_max_n<T>; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::PotrfBlockedPlan<T, NB>::make().size; }
};
template <Uplo UL, class T> struct tiers<routine::potrf<UL>, T> {
    using type = std::conditional_t<(detail::reg_max_n<T> >= 32),
                                    tier::list<tier::Reg<8>, tier::Reg<16>, tier::Reg<32>, tier::Cta<64>, tier::Cta<128>, tier::Blocked<32>>,
                                    tier::list<tier::Reg<8>, tier::Reg<16>, tier::Cta<64>, tier::Cta<128>, tier::Blocked<16>>>;
};

// The workspace of one matrix (host and device; Reg: none). Tier may be tier::upto<NMax>.
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize potrf_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::potrf<UL>, T, Tier>;
    return tier_info<routine::potrf<UL>, Tr, T>::buffer_size(n);
}

// A := L L^H (Lower) / U^H U (Upper) of the n x n Hermitian positive definite A (its UL triangle
// read and overwritten, the other neither read nor written); returns info (see the file head).
template <Uplo UL, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int potrf(const Ex& ex, const MA& A, const Workspace& ws = {}) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::potrf<UL>, T, Tier>;
    if constexpr (tier::is_reg_v<Tr>) {
        (void)ws;
        return detail::potrf_reg_op<UL, Tr::n_max>(ex, A);
    } else if constexpr (tier::is_cta_v<Tr>) {
        return detail::potrf_cta_op<UL, Tr::n_max, typename Tr::alg_type>(ex, A, ws);
    } else {
        static_assert(tier::is_blocked_v<Tr>, "potrf: tier::Reg<N>, tier::Cta<N> or tier::Blocked<NB>");
        return detail::potrf_blocked_op<UL, Tr::nb, typename Tr::alg_type>(ex, A, ws);
    }
}

}  // namespace batchlas::device
