// SPDX-License-Identifier: MIT
#pragma once
// LAPACK ?sygst / ?hegst in a kernel: the reduction of A x = lambda B x (itype 1), A B x = lambda x
// (2) and B A x = lambda x (3) to a standard Hermitian problem, from B's Cholesky factor F (L for
// Lower, U^H for Upper): itype 1 C = F^-1 A F^-H, itypes 2, 3 C = F^H A F, uplo triangles only.
//   sygst<Itype, UL, tier::Cta<N>>(ex, A, B, ws)  one matrix per WorkGroup, n <= N <= 32
//   sygs2_group<Inverse, KB, NF>(...)             ?sygs2 on ipw diagonal blocks per work-group:
//                                                 the one body; sygst's host blocked route
//                                                 launches it for every kb x kb diagonal block
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

namespace batchlas::device::inline v2 {

/// F's leading dimension at order kb: whole 16-byte packs per line, plus one pack against bank
/// conflicts; and the local elements of one block (F, then the transpose buffer of ld kb + 1).
template <class T> constexpr int sygs2_ldf(int kb) {
    const int v = sizeof(T) >= 16 ? 1 : static_cast<int>(16 / sizeof(T));
    return (kb + v - 1) / v * v + v;
}
template <class T> constexpr std::size_t sygs2_block_elems(int kb) {
    return static_cast<std::size_t>(kb) * static_cast<std::size_t>(sygs2_ldf<T>(kb) + kb + 1);
}

namespace sygst_detail {

// One aligned 16-byte load of consecutive elements (one element when T is 16 bytes).
template <class T> struct alignas(16) Pack16 {
    T e[16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1];
};
template <class T> inline constexpr int pack16_elems = static_cast<int>(sizeof(Pack16<T>::e) / sizeof(T));

BATCHLAS_DLA_INLINE bool in_triangle(Uplo uplo, int i, int j) { return uplo == Uplo::Lower ? i >= j : i <= j; }

template <class H> struct HermCol {
    H h;
    int j;
    using value_type = typename H::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return h(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return h.rows(); }
};

// The Hermitian (or symmetric) matrix whose uplo triangle M stores (uplo at run time): the
// diagonal read as real, the other triangle as the conjugate of its mirror, one load.
template <class M> struct HermView {
    M a;
    Uplo uplo;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const {
        const bool own = in_triangle(uplo, i, j);  // one load, its address selected
        const value_type x = a(own ? i : j, own ? j : i);
        return i == j ? force_real(x) : own ? x : Arith<value_type>::conj(x);
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
    BATCHLAS_DLA_INLINE HermCol<HermView> col(int j) const { return {*this, j}; }
};
template <class M> BATCHLAS_DLA_INLINE HermView<M> herm(M a, Uplo uplo) { return {a, uplo}; }

// A group of `size` items over `tiles` m x m tiles, column by column: item lid owns row lid % m
// of tile columns lid / m, lid / m + size / m, ...; f(tile, row, col). When m does not divide
// size, the last items revisit columns another item covers: f must be idempotent.
template <class F> BATCHLAS_DLA_INLINE void for_each_tile_col(int lid, int size, int m, int tiles, F&& f) {
    const int row = lid % m, step = size / m;
    for (int idx = lid / m; idx < tiles * m; idx += step) {
        const int g = idx / m, c = idx - g * m;
        f(g, row, c);
    }
}

// x_i combines column i of A with the other entries of x (Below: p < i, else p > i); term p
// goes to accumulator p mod W, summed as a tree. Solve: x_i = (x_i - sum) / a_ii, else
// x_i = a_ii x_i + sum. In place, so i is visited before every entry whose sum still needs it.
template <bool Below, bool Solve, bool Conj, int W, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void coldot_tri(const MA& a, Frag<T, N, L>& x) {
    static_assert(W > 0 && (W & (W - 1)) == 0, "sygst trsv_packed/trmv_packed: W is a power of two");
    using A = Arith<T>;
    using Ar = arith_t<arith::Fma, T>;
    constexpr int V = pack16_elems<T>;
    constexpr int K = static_extent_v<std::remove_cv_t<decltype(a.n)>>;  // -1: a run-time extent
    constexpr int Trip = K > 0 ? K : N;
    static_assert(Trip <= N, "sygst trsv_packed/trmv_packed: the extent exceeds the fragment");
    constexpr bool ascending = Below == Solve;
    const int n = a.cols();
    auto step = [&](T& acc, const T& aij, const T& xj) BATCHLAS_DLA_BODY {
        if constexpr (Solve) Ar::msub(acc, Conj ? A::conj(aij) : aij, xj);  // acc - op(a) x
        else if constexpr (Conj) Ar::mac_conj(acc, aij, xj);
        else Ar::mac(acc, aij, xj);
    };
    BATCHLAS_UNROLL_FULL
    for (int s = 0; s < Trip; ++s) {
        const int i = ascending ? s : Trip - 1 - s;
        if (i < n) {
            T acc[W];
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < W; ++k) acc[k] = A::zero();
            const T* const line = &a(0, i);  // column i of A
            if constexpr (Below) {
                BATCHLAS_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            } else {
                BATCHLAS_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < Trip; p0 += V) {
                    if (p0 >= n) continue;
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < n) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            }
            BATCHLAS_UNROLL_FULL
            for (int w = W / 2; w >= 1; w /= 2) {
                BATCHLAS_UNROLL_FULL
                for (int k = 0; k < w; ++k) acc[k] = A::add(acc[2 * k], acc[2 * k + 1]);
            }
            const auto d = A::re(line[i]);
            if constexpr (Solve) x[i] = div_real(A::add(x[i], acc[0]), d);
            else x[i] = A::add(scale_real(d, x[i]), acc[0]);
        }
    }
}

// x := op(A)^-1 x in place, op = Trans or ConjTrans, A n x n triangular (UL stored) with a real
// diagonal; n is A's extent (compile-time for Ext<K>). Every column start of A is 16-byte
// aligned. x is the caller's Frag: a by-value copy hoists every load of A (REG 37 -> 70).
template <Uplo UL, Transpose TR, int W = 4, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void trsv_packed(const Thread&, const MA& a, Frag<T, N, L>& x) {
    static_assert(TR != Transpose::NoTrans, "sygst trsv_packed: the column form is op(A) = A^T or A^H");
    coldot_tri<UL == Uplo::Upper, true, TR == Transpose::ConjTrans, W>(a, x);
}

// x := op(A) x in place, op = Trans or ConjTrans, A as for trsv_packed.
template <Uplo UL, Transpose TR, int W = 4, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void trmv_packed(const Thread&, const MA& a, Frag<T, N, L>& x) {
    static_assert(TR != Transpose::NoTrans, "sygst trmv_packed: the column form is op(A) = A^T or A^H");
    coldot_tri<UL == Uplo::Upper, false, TR == Transpose::ConjTrans, W>(a, x);
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
    if constexpr (Inverse) trsv_packed<Uplo::Upper, Transpose::Trans>(Thread{}, fs, c);
    else trmv_packed<Uplo::Lower, Transpose::ConjTrans>(Thread{}, fs, c);
    return c;
}

// The local memory of one Cta-tier matrix: one block, in whole 16-byte packs.
template <class D> struct CtaPlan {
    Plan plan;
    ws::Slot<Pack16<D>> block;
};
template <class D> constexpr CtaPlan<D> cta_plan(int n) {
    CtaPlan<D> p{Plan(16), {}};
    const std::size_t v = static_cast<std::size_t>(pack16_elems<D>);
    p.block = p.plan.template take<Pack16<D>>(ws::Space::Local, (sygs2_block_elems<D>(n) + v - 1) / v);
    return p;
}

}  // namespace sygst_detail

/// ?SYGS2 on ipw diagonal blocks of order kb (<= NF <= 32) per work-group: block gi is matrix
/// first + gi (first = grp * ipw), its column j owned by item gi * kb + j of the wg items. The
/// group copies A's and B's uplo triangles into local memory (coalesced across the blocks), then
/// each item reduces its column; C (real diagonal) goes back to A's uplo triangle and, through
/// full(itm, row, col, value), to wherever the caller keeps the whole Hermitian block.
///   fbase   ipw * (fper + sper) local elements, 16-byte aligned: the F stores, then the S buffers
///   fper    kb * ldf, ldf = sygs2_ldf<D>(kb);  sper = kb * lds, lds = kb + 1
///   a_of(itm), b_of(itm)  matrix itm's A (read, written) and B (read) operands
///   dead(itm)             true for a matrix no item may load, compute or store (out of range or
///                         masked; uniform per block)
/// KB > 0 is a compile-time order (kb == KB); KB = 0 any kb <= NF. Every item of ex reaches the
/// four barriers. The argument order is the hand kernel's capture order (its parameter order).
template <bool Inverse, int KB, int NF = 32, class Ex, class D, class AOf, class Dead, class BOf, class Full>
BATCHLAS_DLA_INLINE void sygs2_group(const Ex& ex, int t, int kb, bool lower, std::size_t grp, int ipw, D* fbase,
                                     int fper, const AOf& a_of, int sper, int lds, int wg, const Dead& dead,
                                     const BOf& b_of, int ldf, const Full& full) {
    namespace sd = sygst_detail;
    static_assert(NF >= 1 && NF <= 32, "sygs2_group: the column fragment holds at most 32 rows");
    using SMat = Mat<D, sd::KbExt<KB>, sd::KbExt<KB>, Dyn, Mem::Local, Cyclic, sd::Slm<D>>;
    constexpr int A16 = sd::slm_align<D>;
    const int live = KB ? KB : kb;
    const sd::KbExt<KB> kn = sd::kb_ext<KB>(kb);
    const Uplo ul = lower ? Uplo::Lower : Uplo::Upper;
    const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(grp) * ipw;
    D* const sbase = fbase + ipw * fper;
    auto s_of = [=](int gi) { return mat_aligned<A16, Mem::Local>(sbase + gi * sper, kb, kb, lds); };
    // ?LACPY of A's uplo triangle into S and of B's into F (conjugate-transposed from U), column-wise
    // and coalesced. I2: a dead block is never loaded, computed or stored.
    sd::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DLA_BODY {
        const std::ptrdiff_t itm = first + gi;
        if (row >= kb || dead(itm) || !sd::in_triangle(ul, row, c)) return;
        s_of(gi)(row, c) = a_of(itm)(row, c);
        const D bv = b_of(itm)(row, c);
        const auto fs = mat_aligned<A16, Mem::Local>(fbase + gi * fper, kb, kb, ldf);
        if (lower) sd::fat<Inverse>(fs, row, c) = bv;
        else sd::fat<Inverse>(fs, c, row) = Arith<D>::conj(bv);
    });
    const int g = t / kb, j = t - g * kb;
    const bool active = g < ipw && !dead(first + g);
    const D* const f = fbase + (active ? g * fper : 0);
    const SMat s{sd::Slm<D>{sbase + (active ? g * sper : 0)}, kn, kn, Dyn{lds}};
    sync(ex);
    // X = op(F) A, column j: the Hermitian A's column, then F^-1 or F^H.
    Frag<D, NF> col{};
    if (active) load(col, sd::herm(s, ul).col(j), live);
    sync(ex);
    if (active) col = sd::apply<Inverse, KB>(col, f, ldf, kb);
    if (active) store(col, s.col(j), live);
    sync(ex);
    // C = op(F) X^H: X's conjugate transpose, row j of X as column j.
    if (active) load(col, conj_view(s.row(j)), live);
    sync(ex);
    if (active) col = sd::apply<Inverse, KB>(col, f, ldf, kb);
    if (active) store(col, s.col(j), live);
    sync(ex);
    // The Hermitian C (real diagonal) to `full` and its uplo triangle back to A.
    sd::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DLA_BODY {
        const std::ptrdiff_t itm = first + gi;
        if (row >= kb || dead(itm)) return;
        const D v = sd::herm(s_of(gi), ul)(row, c);
        full(itm, row, c, v);
        if (sd::in_triangle(ul, row, c)) a_of(itm)(row, c) = v;
    });
}

namespace routine {
template <int Itype, Uplo UL> struct sygst {
    static_assert(Itype >= 1 && Itype <= 3, "sygst: itype is 1, 2 or 3");
    static constexpr const char* name = "sygst";
};
}  // namespace routine

// One tier: tier::Cta<N>, N <= 32 (the column of the block in one item's registers). A larger
// order is the host blocked route today (design C.7 B5: the Blocked tier is not written yet).
template <int Itype, Uplo UL, class T> struct tiers<routine::sygst<Itype, UL>, T> {
    using type = tier::list<tier::Cta<8>, tier::Cta<16>, tier::Cta<32>>;
};

namespace sygst_detail {
template <class Tier> inline constexpr bool is_cta_v = false;
template <int N, class Alg> inline constexpr bool is_cta_v<tier::Cta<N, Alg>> = N >= 1 && N <= 32;
}  // namespace sygst_detail

/// Local and global bytes of one matrix of order n <= N (global: none); Tier may be tier::upto.
template <int Itype, Uplo UL, class Tier, class T> constexpr WorkspaceSize sygst_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::sygst<Itype, UL>, std::remove_cv_t<T>, Tier>;
    static_assert(sygst_detail::is_cta_v<Tr>, "sygst: tier::Cta<N>, N <= 32, is the only tier");
    return sygst_detail::cta_plan<std::remove_cv_t<T>>(n).plan.size();
}

template <int Itype, Uplo UL, int N, class Alg, class T> struct tier_info<routine::sygst<Itype, UL>, tier::Cta<N, Alg>, T> {
    static_assert(N >= 1 && N <= 32, "sygst: tier::Cta<N>, N <= 32");
    static constexpr int wg_size = (N + 31) / 32 * 32, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int n) { return sygst_buffer_size<Itype, UL, tier::Cta<N, Alg>, T>(n); }
};

/// C := F^-1 A F^-H (Itype 1) or F^H A F (2, 3) on one matrix, in A's UL triangle (the other
/// triangle is not touched); B holds F in its UL triangle (read only). Tier: tier::Cta<N> or
/// tier::upto<N>. COLLECTIVE over a WorkGroup of at least N items (item j < n owns column j);
/// A and B visible to every item on entry, A's result on return. ws.local:
/// sygst_buffer_size<Itype, UL, Tier, T>(n).local_bytes, 16-byte aligned. Returns 0, or -2 (A not
/// square or n > N), -3 (B smaller than n x n); uniform over the executor.
template <int Itype, Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE int sygst(const Ex& ex, const MA& a, const MB& b, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using Tr = resolve_tier_t<routine::sygst<Itype, UL>, D, Tier>;
    static_assert(sygst_detail::is_cta_v<Tr>, "sygst: tier::Cta<N>, N <= 32, is the only tier");
    constexpr int N = Tr::n_max;
    static_assert(is_work_group_v<Ex> && Ex::size >= N, "sygst<tier::Cta<N>>: a WorkGroup of at least N items");
    const int n = a.cols();
    if (a.rows() != n || n > N) return -2;
    if (b.rows() < n || b.cols() < n) return -3;
    if (n == 0) return 0;
    const int ldf = sygs2_ldf<D>(n), lds = n + 1;
    D* const f = reinterpret_cast<D*>(carve_local(ws, sygst_detail::cta_plan<D>(n).block).get_raw());
    sygs2_group<Itype == 1, 0, N>(
        ex, ex.lid(), n, UL == Uplo::Lower, std::size_t(0), 1, f, n * ldf, [&](std::ptrdiff_t) BATCHLAS_DLA_BODY { return a; },
        n * lds, lds, Ex::size, [](std::ptrdiff_t) BATCHLAS_DLA_BODY { return false; },
        [&](std::ptrdiff_t) BATCHLAS_DLA_BODY { return b; }, ldf, [](std::ptrdiff_t, int, int, const D&) BATCHLAS_DLA_BODY {});
    sync(ex, Mem::Global);  // A's result visible to every item
    return 0;
}

}  // namespace batchlas::device
