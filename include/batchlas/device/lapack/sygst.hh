// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_eig
/// @brief ?SYGST / ?HEGST in a kernel: the reduction of a generalized Hermitian-definite
/// eigenproblem to a standard one, for one matrix, from B's Cholesky factor.
///
/// The problems are \f$A x = \lambda B x\f$ (`Itype::AinvB`, itype 1), \f$A B x = \lambda x\f$
/// (`Itype::AB`, 2) and \f$B A x = \lambda x\f$ (`Itype::BA`, 3). With F the factor (L for
/// `Uplo::Lower`, \f$U^H\f$ for `Uplo::Upper`), itype 1 gives \f$C = F^{-1} A F^{-H}\f$ and
/// itypes 2 and 3 give \f$C = F^H A F\f$, on the `UL` triangles only. The routine is sygst(); its
/// routine tag is routine::sygst. Its tiers, smallest first, are `tier::Cta<8>`, `tier::Cta<16>`,
/// `tier::Cta<32>`, `tier::Blocked<32>` and `tier::Blocked<16>`. The diagonal-block kernel
/// sygs2_group() is in @ref api_device_lowlevel.
// sygs2_group<Inverse, KB, NF>(...) is ?sygs2 on ipw diagonal blocks per work-group: the one
// body; sygst's host blocked route launches it for every kb x kb diagonal block.
// One work-item per column of a block, the column in registers; F and a transpose buffer per
// block in local memory, F stored so each pass reads its lines as aligned 16-byte packs. Each pass
// is a triangular solve with F (itype 1) or a product with F^H (2, 3), applied to A and then to
// the conjugate transpose of the result, which is the Hermitian C itself.
// Arithmetic is arith::Fma, bit-equal to the hand kernel's four-fma helpers.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <cstddef>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device {

namespace detail {

/// F's leading dimension at order kb: whole 16-byte packs per line, plus one pack against bank
/// conflicts; and the local elements of one block (F, then the transpose buffer of ld kb + 1).
template <class Scalar> constexpr int sygs2_ldf(int kb) {
    const int v = sizeof(Scalar) >= 16 ? 1 : static_cast<int>(16 / sizeof(Scalar));
    return (kb + v - 1) / v * v + v;
}
template <class Scalar> constexpr std::size_t sygs2_block_elems(int kb) {
    return static_cast<std::size_t>(kb) * static_cast<std::size_t>(sygs2_ldf<Scalar>(kb) + kb + 1);
}

BATCHLAS_DEVICE_INLINE bool in_triangle(Uplo uplo, int i, int j) { return uplo == Uplo::Lower ? i >= j : i <= j; }

// A group of `size` items over `tiles` m x m tiles, column by column: item lid owns row lid % m
// of tile columns lid / m, lid / m + size / m, ...; f(tile, row, col). When m does not divide
// size, the last items revisit columns another item covers: f must be idempotent.
template <class F> BATCHLAS_DEVICE_INLINE void for_each_tile_col(int lid, int size, int m, int tiles, F&& f) {
    const int row = lid % m, step = size / m;
    for (int idx = lid / m; idx < tiles * m; idx += step) {
        const int g = idx / m, c = idx - g * m;
        f(g, row, c);
    }
}

// Every element of the 16-byte-aligned local packs is aligned to min(16, sizeof(D)): said through
// the operand base, since a view's pointer member drops the alignment derived from the local base
// (a complex S column would go out as two 8-byte shared stores per element instead of one).
template <class D> inline constexpr int slm_align = sizeof(D) < 16 ? static_cast<int>(sizeof(D)) : 16;
template <class D> using Slm = Aligned<D, slm_align<D>>;

template <int KB> using KbExt = std::conditional_t<(KB > 0), Ext<KB>, Dyn>;
template <int KB> inline KbExt<KB> kb_ext(int kb) {
    if constexpr (KB > 0) return {};
    else return Dyn{kb};
}

// device::load(f, x, live), except on DPC++'s SPIR targets: there a rolled loop of guarded loads.
// The unrolled form came out wrong on the OpenCL CPU backend for complex<double> at a run-time
// order (results O(1) off, right at -cl-opt-disable or with the loop rolled; 2026-11-21, opencl:cpu,
// sygvd_tests' SygstBlocked cases). The same values; other targets compile exactly as before.
template <class D, int NF, class X> BATCHLAS_DEVICE_INLINE void load_live(Frag<D, NF>& f, const X& x, int live) {
#if defined(__SPIR__) && !defined(__ACPP__)
#pragma nounroll
    for (int k = 0; k < NF; ++k) {
        if (k < live) f[k] = static_cast<D>(x(k));
        else f[k] = Arith<D>::zero();
    }
#else
    load(f, x, live);
#endif
}
// F(r, q) as stored: F^T column-major for the solve (column i of the store is row i of F),
// F column-major for the product.
template <bool Inverse, class M> inline auto& fat(const M& fs, int r, int q) { return Inverse ? fs(q, r) : fs(r, q); }

// F^-1 c (itype 1, ?TRSV with the stored F^T) or F^H c (2, 3, ?TRMV). Plain `static inline`, the
// internal linkage of the hand kernel's anonymous-namespace helper: the compiler's inlining
// choice for the two calls per pass is unchanged.
template <bool Inverse, int KB, int NF, class D>
static inline Frag<D, NF> apply(Frag<D, NF> c, const D* f, int ldf, int kb) {
    const KbExt<KB> kn = kb_ext<KB>(kb);
    const Mat<const D, KbExt<KB>, KbExt<KB>, Dyn, Mem::Local> fs{f, kn, kn, Dyn{ldf}};
    if constexpr (Inverse) trsv<Uplo::Upper, Transpose::Trans, Diag::NonUnit, alg::RealDiv, alg::SplitDot<4>, arith::Fma>(Thread{}, fs, c, fs.cols());
    else trmv<Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, arith::Fma, alg::SplitDot<4>>(Thread{}, fs, c, fs.cols());
    return c;
}

// The local memory of one Cta-tier matrix: one block, in whole 16-byte packs.
template <class D> struct SygstCtaPlan {
    Plan plan;
    WsSlot<Pack16<D>> block;
};
template <class D> constexpr SygstCtaPlan<D> sygst_cta_plan(int n) {
    SygstCtaPlan<D> p{Plan(16), {}};
    const std::size_t v = static_cast<std::size_t>(pack16_elems<D>);
    p.block = p.plan.template take<Pack16<D>>(WsSpace::Local, (sygs2_block_elems<D>(n) + v - 1) / v);
    return p;
}

}  // namespace detail

/// @ingroup api_device_lowlevel
/// @brief ?SYGS2 on `ipw` diagonal blocks of order `kb` (<= NF <= 32) per work-group.
///
/// Not covered by the compatibility promise: this is the diagonal-block body of sygst() and of
/// the host blocked route. Block gi is matrix first + gi (first = `grp` * `ipw`), its column j
/// owned by item gi * `kb` + j of the `wg` items. The group copies A's and B's uplo triangles
/// into local memory (coalesced across the blocks), then each item reduces its column; C (real
/// diagonal) goes back to A's uplo triangle and, through `full`, to wherever the caller keeps the
/// whole Hermitian block. Every item of `ex` reaches the four barriers.
/// @tparam Inverse  true: itype 1 (\f$F^{-1} A F^{-H}\f$); false: itypes 2 and 3 (\f$F^H A F\f$)
/// @tparam KB       a compile-time order (`kb` == KB), or 0 for any `kb` <= NF
/// @tparam NF       the bound on `kb`, at most 32
/// @param[in] ex     the work-group
/// @param[in] t      this item's index in the work-group
/// @param[in] kb     the block order
/// @param[in] lower  true: the lower triangles hold A and F
/// @param[in] grp    the work-group's index
/// @param[in] ipw    the number of blocks per work-group
/// @param[in] fbase  `ipw` * (`fper` + `sper`) local elements, 16-byte aligned: the F stores, then
///                   the S buffers
/// @param[in] fper   `kb` * `ldf` local elements per F store
/// @param[in] a_of   `a_of(itm)`: matrix itm's A operand (read and written)
/// @param[in] sper   `kb` * `lds` local elements per S buffer
/// @param[in] lds    `kb` + 1
/// @param[in] wg     the number of items in the work-group
/// @param[in] dead   `dead(itm)`: true for a matrix no item may load or store (uniform per block)
/// @param[in] b_of   `b_of(itm)`: matrix itm's B operand (read)
/// @param[in] ldf    `detail::sygs2_ldf<D>(kb)`
/// @param[in] full   `full(itm, row, col, value)`: stores an element of the whole Hermitian block
// The argument order is the hand kernel's capture order (its parameter order).
template <bool Inverse, int KB, int NF = 32, class Ex, class D, class AOf, class Dead, class BOf, class Full>
BATCHLAS_DEVICE_INLINE void sygs2_group(const Ex& ex, int t, int kb, bool lower, std::size_t grp, int ipw, D* fbase,
                                     int fper, const AOf& a_of, int sper, int lds, int wg, const Dead& dead,
                                     const BOf& b_of, int ldf, const Full& full) {
    static_assert(NF >= 1 && NF <= 32, "sygs2_group: the column fragment holds at most 32 rows");
    using SMat = Mat<D, detail::KbExt<KB>, detail::KbExt<KB>, Dyn, Mem::Local, Cyclic, detail::Slm<D>>;
    constexpr int A16 = detail::slm_align<D>;
    const int live = KB ? KB : kb;
    const detail::KbExt<KB> kn = detail::kb_ext<KB>(kb);
    const Uplo ul = lower ? Uplo::Lower : Uplo::Upper;
    const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(grp) * ipw;
    D* const sbase = fbase + ipw * fper;
    auto s_of = [=](int gi) { return mat_aligned<A16, Mem::Local>(sbase + gi * sper, kb, kb, lds); };
    // ?LACPY of A's uplo triangle into S and of B's into F (conjugate-transposed from U), column-wise
    // and coalesced. I2: a dead block is never loaded or stored.
    detail::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DEVICE_BODY {
        const std::ptrdiff_t itm = first + gi;
        if (row >= kb || dead(itm) || !detail::in_triangle(ul, row, c)) return;
        s_of(gi)(row, c) = a_of(itm)(row, c);
        const D bv = b_of(itm)(row, c);
        const auto fs = mat_aligned<A16, Mem::Local>(fbase + gi * fper, kb, kb, ldf);
        if (lower) detail::fat<Inverse>(fs, row, c) = bv;
        else detail::fat<Inverse>(fs, c, row) = Arith<D>::conj(bv);
    });
    // Every item runs the two passes, so no barrier sits beside a per-item branch: AdaptiveCpp's
    // Metal code repeats such a branch's barriers on both of its sides, and a partly live sub-group
    // then reaches each one twice and frees the others a phase early (wrong blocks from the first
    // one in that sub-group). A dead block's items work in its own slot, which nothing loads from or
    // stores to global memory; an item past the ipw * kb columns repeats item t % (ipw * kb), so it
    // stores the same values to the same column.
    const int tc = t % (ipw * live);
    const int g = tc / kb, j = tc - g * kb;
    const D* const f = fbase + g * fper;
    const SMat s{detail::Slm<D>{sbase + g * sper}, kn, kn, Dyn{lds}};
    sync(ex);
    // X = op(F) A, column j: the Hermitian A's column, then F^-1 or F^H.
    Frag<D, NF> col{};
    detail::load_live(col, herm_view(s, ul).col(j), live);
    sync(ex);
    col = detail::apply<Inverse, KB>(col, f, ldf, kb);
    store(col, s.col(j), live);
    sync(ex);
    // C = op(F) X^H: X's conjugate transpose, row j of X as column j.
    detail::load_live(col, conj_view(s.row(j)), live);
    sync(ex);
    col = detail::apply<Inverse, KB>(col, f, ldf, kb);
    store(col, s.col(j), live);
    sync(ex);
    // The Hermitian C (real diagonal) to `full` and its uplo triangle back to A.
    detail::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DEVICE_BODY {
        const std::ptrdiff_t itm = first + gi;
        if (row >= kb || dead(itm)) return;
        const D v = herm_view(s_of(gi), ul)(row, c);
        full(itm, row, c, v);
        if (detail::in_triangle(ul, row, c)) a_of(itm)(row, c) = v;
    });
}

namespace routine {
/// @ingroup api_device_lapack_eig
/// @brief The routine tag of sygst() for the problem type `IT` and triangle `UL`: names the
/// routine to `with_tier`, `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Itype IT, Uplo UL> struct sygst {
    static constexpr const char* name = "sygst";  ///< The routine's name, "sygst".
};
}  // namespace routine

/// @ingroup api_device_lapack_eig
/// @brief sygst()'s tiers for every `Scalar`, smallest first: `Cta<8>`, `Cta<16>`, `Cta<32>`,
/// `Blocked<32>`, `Blocked<16>`.
///
/// `tier::Cta<N>`, N <= 32, holds the column of the block in one item's registers;
/// `tier::Blocked<NB>` takes any n. `Blocked<16>` is the fallback where `Blocked<32>`'s diagonal
/// block does not fit (`std::complex<double>` on 32 KB of local memory).
template <Itype IT, Uplo UL, class Scalar> struct tiers<routine::sygst<IT, UL>, Scalar> {
    using type = tier::list<tier::Cta<8>, tier::Cta<16>, tier::Cta<32>, tier::Blocked<32>, tier::Blocked<16>>;
};

namespace detail {
template <class Tier> inline constexpr bool sygst_is_cta_v = false;
template <int N, class Alg> inline constexpr bool sygst_is_cta_v<tier::Cta<N, Alg>> = N >= 1 && N <= 32;
template <class Tier> inline constexpr bool sygst_is_blocked_v = false;
template <int NB, class Alg> inline constexpr bool sygst_is_blocked_v<tier::Blocked<NB, Alg>> = NB >= 1 && NB <= 32;
template <class Tier> inline constexpr bool is_tier_v = sygst_is_cta_v<Tier> || sygst_is_blocked_v<Tier>;

// The Blocked tier's Level 3 tile: device_lapack_l3's 32 x 32 x 8 preset, 256 items.
template <class D> using blocked_cfg_t = l3::cfg::RegTiled<D, 32, 32, 8, 2, 2>;

// One Blocked matrix: local, the ?sygs2 block of order min(n, NB) and the L3 staging, both at
// offset 0 (never live at once); global (n > NB only), H, the whole Hermitian diagonal block
// (NB x NB, ld NB), and for itypes 2, 3 W, the n x NB panel product of the out-of-place trmm.
template <class D> struct SygstBlockedPlan {
    WorkspaceSize size;
    WsSlot<D> h, w;
};
template <class D, int NB, Itype IT> constexpr SygstBlockedPlan<D> sygst_blocked_plan(int n) {
    Plan p(16);
    const int kb = n < NB ? n : NB;
    const std::size_t blk = sygs2_block_elems<D>(kb) * sizeof(D), stage = l3::smem_bytes<blocked_cfg_t<D>>();
    p.take<std::byte>(WsSpace::Local, blk > stage ? blk : stage);
    SygstBlockedPlan<D> r{};
    if (n > NB) {
        r.h = p.take<D>(WsSpace::Global, static_cast<std::size_t>(NB) * NB);
        if (IT != Itype::AinvB) r.w = p.take<D>(WsSpace::Global, static_cast<std::size_t>(n) * NB);
    }
    r.size = p.size();
    return r;
}
}  // namespace detail

/// @ingroup api_device_lapack_eig
/// @brief sygst()'s local-memory tier `tier::Cta<N>`, N <= 32: one matrix per work-group, item j
/// owning column j.
///
/// The executor is `WorkGroup<wg_size, 32>`, `wg_size` N rounded up to a multiple of 32.
/// `admits(n)` is 0 <= n <= N. `buffer_size(n)` is local memory only: one ?SYGS2 block in whole
/// 16-byte packs; no global workspace.
template <Itype IT, Uplo UL, int N, class Alg, class Scalar> struct tier_info<routine::sygst<IT, UL>, tier::Cta<N, Alg>, Scalar> {
    static_assert(N >= 1 && N <= 32, "sygst: tier::Cta<N>, N <= 32");
    static constexpr int wg_size = (N + 31) / 32 * 32, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int n) {
        return detail::sygst_cta_plan<std::remove_cv_t<Scalar>>(n).plan.size();
    }
};
/// @ingroup api_device_lapack_eig
/// @brief sygst()'s blocked tier `tier::Blocked<NB>`, NB <= 32: any n, LAPACK's ?SYGST schedule in
/// one 256-item work-group, `A` and `B` in global memory.
///
/// The NB x NB diagonal blocks are reduced by sygs2_group(), the rest by the in-group level 3.
/// The executor is `WorkGroup<256, 32>`. `admits(n)` is n >= 0. `buffer_size(n)` has a local part
/// (the ?SYGS2 block and the level 3 stages, never live at once) and, for n > NB, a global part
/// per matrix: the whole Hermitian diagonal block, and for itypes 2 and 3 an n x NB panel.
template <Itype IT, Uplo UL, int NB, class Alg, class Scalar> struct tier_info<routine::sygst<IT, UL>, tier::Blocked<NB, Alg>, Scalar> {
    static_assert(NB >= 1 && NB <= 32, "sygst: tier::Blocked<NB>, NB <= 32 (the ?sygs2 block)");
    static constexpr int wg_size = detail::blocked_cfg_t<std::remove_cv_t<Scalar>>::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) {
        return detail::sygst_blocked_plan<std::remove_cv_t<Scalar>, NB, IT>(n).size;
    }
};

namespace detail {

// One ?sygs2 block of order kb <= NF at f, A's and B's leading kb x kb blocks (ipw 1): sygs2_group
// on ex when ex is one sub-group, else on sub-group 0 alone (its 32 items cover the kb columns;
// sygs2_group's passes have no per-item branch, so no sub-group meets a barrier twice).
template <bool Inverse, int NF, class Ex, class D, class MA, class MB, class Full>
BATCHLAS_DEVICE_INLINE void sygs2_one(const Ex& ex, int kb, bool lower, D* f, const MA& a, const MB& b, const Full& full) {
    const int ldf = sygs2_ldf<D>(kb), lds = kb + 1;
    auto run = [&](const auto& x) BATCHLAS_DEVICE_BODY {
        sygs2_group<Inverse, 0, NF>(
            x, x.lid(), kb, lower, std::size_t(0), 1, f, kb * ldf, [&](std::ptrdiff_t) BATCHLAS_DEVICE_BODY { return a; }, kb * lds,
            lds, std::remove_cvref_t<decltype(x)>::size, [](std::ptrdiff_t) BATCHLAS_DEVICE_BODY { return false; },
            [&](std::ptrdiff_t) BATCHLAS_DEVICE_BODY { return b; }, ldf, full);
    };
    if constexpr (Ex::size <= Ex::sg_size) run(ex);
    else if (ex.sub().get_group_linear_id() == 0) run(SubGroup<Ex::sg_size>{ex.sub()});
}

template <class Scalar> BATCHLAS_DEVICE_INLINE l3::Global<Scalar> gsub(const l3::Global<Scalar>& g, int r0, int c0,
                                                                       int m, int n) {
    return {g.p + r0 + c0 * g.ld, g.ld, m, n};
}

// LAPACK's dsygst / zhegst on the UL triangles of one matrix in global memory, the step order of
// the host blocked route (kernels/sygst_blocked.cc) at nb = NB: per block the ?sygs2 diagonal
// reduction (sygs2_group, one block, in `lmem`), then trsm (itype 1) or trmm (2, 3), the gemm
// with the mirrored diagonal block H (LAPACK's hemm) and her2k (syr2k for a real T) on the panel and trailing
// parts. Every L3 op ends on a global barrier; each diagonal step is followed by one.
template <Itype IT, Uplo UL, int NB, class Ex, class D, class DB>
BATCHLAS_DEVICE_INLINE void blocked(const Ex& ex, const l3::Global<D>& a, const l3::Global<DB>& b, const Workspace& sm, D* h, D* w) {
    using Cfg = blocked_cfg_t<D>;
    using Ar = Arith<D>;
    using R = typename Ar::real_t;
    constexpr bool lower = UL == Uplo::Lower;
    constexpr Transpose TH = Ar::is_complex ? Transpose::ConjTrans : Transpose::Trans;
    constexpr Transpose NT = Transpose::NoTrans;
    D* const lmem = reinterpret_cast<D*>(sm.local.get_raw());  // sygs2's block, under the L3 stage
    auto lift = [](R x) BATCHLAS_DEVICE_BODY {
        if constexpr (Ar::is_complex) return Ar::make(x, R(0));
        else return D(x);
    };
    const D half = lift(R(0.5)), mhalf = lift(R(-0.5));
    const int n = a.n;
    // Reduce the kb x kb diagonal block at `at`; with `full`, the whole Hermitian result goes to H.
    auto diag = [&](int at, int kb, bool full) BATCHLAS_DEVICE_BODY {
        const auto ab = mat(a.p + at + at * a.ld, kb, kb, a.ld);
        const auto bb = mat(b.p + at + at * b.ld, kb, kb, b.ld);
        sygs2_one<IT == Itype::AinvB, NB>(ex, kb, lower, lmem, ab, bb, [&](std::ptrdiff_t, int r, int c, const D& v) BATCHLAS_DEVICE_BODY {
            if (full) h[r + c * NB] = v;
        });
        sync(ex, Mem::Global);
    };
    // H := the whole Hermitian kb x kb block whose UL triangle A holds at `at` (real diagonal).
    auto mirror = [&](int at, int kb) BATCHLAS_DEVICE_BODY {
        lacpy(ex, herm_view<UL, true>(mat(a.p + at + at * a.ld, kb, kb, a.ld)), mat(h, kb, kb, NB));
        sync(ex, Mem::Global);
    };
    for (int k = 0; k < n; k += NB) {
        const int kb = n - k < NB ? n - k : NB, m = n - k - kb, r = k + kb;
        const l3::Global<D> hm{h, NB, kb, kb};
        if constexpr (IT == Itype::AinvB) {
            diag(k, kb, m > 0);
            if (m == 0) break;
            const auto f11 = gsub(b, k, k, kb, kb);
            const auto f22 = gsub(b, r, r, m, m);
            const auto a22 = gsub(a, r, r, m, m);
            if constexpr (lower) {  // A21 := L22^-1 (A21 L11^-H - L21 C11), A22 -= P L21^H + L21 P^H
                const auto p = gsub(a, r, k, m, kb);
                const auto l21 = gsub(b, r, k, m, kb);
                trsm<Side::Right, UL, TH, Diag::NonUnit, Cfg>(ex, One{}, f11, p, sm);
                gemm<NT, NT, Cfg>(ex, mhalf, l21, hm, One{}, p, sm);
                her2k<UL, NT, Cfg>(ex, NegOne{}, p, l21, One{}, a22, sm);
                gemm<NT, NT, Cfg>(ex, mhalf, l21, hm, One{}, p, sm);
                trsm<Side::Left, UL, NT, Diag::NonUnit, Cfg>(ex, One{}, f22, p, sm);
            } else {  // A12 := (U11^-H A12 - C11 U12) U22^-1, A22 -= P^H U12 + U12^H P
                const auto p = gsub(a, k, r, kb, m);
                const auto u12 = gsub(b, k, r, kb, m);
                trsm<Side::Left, UL, TH, Diag::NonUnit, Cfg>(ex, One{}, f11, p, sm);
                gemm<NT, NT, Cfg>(ex, mhalf, hm, u12, One{}, p, sm);
                her2k<UL, TH, Cfg>(ex, NegOne{}, p, u12, One{}, a22, sm);
                gemm<NT, NT, Cfg>(ex, mhalf, hm, u12, One{}, p, sm);
                trsm<Side::Right, UL, NT, Diag::NonUnit, Cfg>(ex, One{}, f22, p, sm);
            }
            continue;
        }
        if (k > 0) {  // itypes 2, 3: extend the reduced leading k x k block by this block row/column
            const auto a11 = gsub(a, 0, 0, k, k);
            const auto f11 = gsub(b, 0, 0, k, k);
            const auto f22 = gsub(b, k, k, kb, kb);
            mirror(k, kb);
            if constexpr (lower) {  // W = A21 L11 + A22 L21 / 2, A11 += W^H L21 + L21^H W, A21 = L22^H (W + A22 L21 / 2)
                const auto p = gsub(a, k, 0, kb, k);
                const auto l21 = gsub(b, k, 0, kb, k);
                const l3::Global<D> wm{w, NB, kb, k};
                trmm_out<Side::Right, UL, NT, Diag::NonUnit, Cfg>(ex, One{}, f11, p, wm, sm);
                gemm<NT, NT, Cfg>(ex, half, hm, l21, One{}, wm, sm);
                her2k<UL, TH, Cfg>(ex, One{}, wm, l21, One{}, a11, sm);
                gemm<NT, NT, Cfg>(ex, half, hm, l21, One{}, wm, sm);
                trmm_out<Side::Left, UL, TH, Diag::NonUnit, Cfg>(ex, One{}, f22, wm, p, sm);
            } else {  // W = U11 A12 + U12 A22 / 2, A11 += W U12^H + U12 W^H, A12 = (W + U12 A22 / 2) U22^H
                const auto p = gsub(a, 0, k, k, kb);
                const auto u12 = gsub(b, 0, k, k, kb);
                const l3::Global<D> wm{w, n, k, kb};
                trmm_out<Side::Left, UL, NT, Diag::NonUnit, Cfg>(ex, One{}, f11, p, wm, sm);
                gemm<NT, NT, Cfg>(ex, half, u12, hm, One{}, wm, sm);
                her2k<UL, NT, Cfg>(ex, One{}, wm, u12, One{}, a11, sm);
                gemm<NT, NT, Cfg>(ex, half, u12, hm, One{}, wm, sm);
                trmm_out<Side::Right, UL, TH, Diag::NonUnit, Cfg>(ex, One{}, f22, wm, p, sm);
            }
        }
        diag(k, kb, false);
    }
}

}  // namespace detail

/// @ingroup api_device_lapack_eig
/// @brief ?SYGST / ?HEGST in a kernel: \f$C := F^{-1} A F^{-H}\f$ (`Itype::AinvB`) or
/// \f$C := F^H A F\f$ (`Itype::AB`, `Itype::BA`) for one matrix, in `a`'s `UL` triangle.
///
/// The other triangle of `a` is not touched. `b` holds F, potrf()'s factor of B, in its `UL`
/// triangle and is read only; its diagonal must be real (the trsm and trmm steps read it as
/// stored, as LAPACK's do).
/// - `tier::Cta<N>`, N <= 32: `ex` a `WorkGroup` of at least N items (item j < n owns column j);
///   n <= N.
/// - `tier::Blocked<NB>`, NB <= 32: `ex` a `WorkGroup` of `tier_info::wg_size` (256) items; any n;
///   `a` and `b` in global memory (a `Mat` over `T*`, `device::item` or `l3::Global`).
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::sygst<IT, UL>, T>` that admits N.
///
/// Workspace: `ws` holds `buffer_size<routine::sygst<IT, UL>, Tier, T>(n)`, 16-byte aligned
/// (`tier::Blocked` with n > NB: a global slice too). The workspace is not checked.
///
/// @pre Collective over a `WorkGroup`: every item of `ex` calls sygst() with the same arguments;
/// `a` and `b` visible to every item on entry. The result in `a` is visible to every item on
/// return.
/// @tparam IT    `Itype::AinvB`, `Itype::AB` or `Itype::BA`: the generalized problem
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle that holds A, F and the result
/// @tparam Tier  `tier::Cta<N>`, `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the work-group that owns this matrix
/// @param[in,out] a   the n x n Hermitian A; its `UL` triangle is overwritten with C
/// @param[in]     b   B's Cholesky factor F in its `UL` triangle, at least n x n
/// @param[in]     ws  this matrix's workspace slice
/// @return 0 or a negative argument number, the same on every item of `ex`:
/// @retval 0   success
/// @retval -2  `a` is not square, or (`tier::Cta`) n > N
/// @retval -3  `b` is smaller than n x n
template <Itype IT, Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int sygst(const Ex& ex, const MA& a, const MB& b, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using Tr = resolve_tier_t<routine::sygst<IT, UL>, D, Tier>;
    static_assert(detail::is_tier_v<Tr>, "sygst: tier::Cta<N> or tier::Blocked<NB>, N, NB <= 32");
    if constexpr (detail::sygst_is_cta_v<Tr>) {
        constexpr int N = Tr::n_max;
        static_assert(is_work_group_v<Ex> && Ex::size >= N, "sygst<tier::Cta<N>>: a WorkGroup of at least N items");
        const int n = a.cols();
        if (a.rows() != n || n > N) return -2;
        if (b.rows() < n || b.cols() < n) return -3;
        if (n == 0) return 0;
        D* const f = reinterpret_cast<D*>(carve_local(ws, detail::sygst_cta_plan<D>(n).block).get_raw());
        detail::sygs2_one<IT == Itype::AinvB, N>(ex, n, UL == Uplo::Lower, f, a, b,
                                               [](std::ptrdiff_t, int, int, const D&) BATCHLAS_DEVICE_BODY {});
        sync(ex, Mem::Global);  // A's result visible to every item
        return 0;
    } else {
        static_assert(is_work_group_v<Ex> && Ex::size == detail::blocked_cfg_t<D>::threads,
                      "sygst<tier::Blocked<NB>>: a WorkGroup of tier_info's wg_size items");
        const int n = a.cols();
        if (a.rows() != n) return -2;
        if (b.rows() < n || b.cols() < n) return -3;
        if (n == 0) return 0;
        const auto plan = detail::sygst_blocked_plan<D, Tr::nb, IT>(n);
        detail::blocked<IT, UL, Tr::nb>(ex, l3::global_of(a), l3::global_of(b), ws,
                                                 carve_global(ws, plan.h), carve_global(ws, plan.w));
        sync(ex, Mem::Global);
        return 0;
    }
}

}  // namespace batchlas::device
