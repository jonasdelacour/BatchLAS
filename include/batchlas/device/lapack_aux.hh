// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-kernel LAPACK auxiliaries on an executor (@ref api_device_lapack_aux).
///
/// 2-D copies and fills on a triangle (`lacpy`, `laset`, `mirror`), row interchanges (`laswp`,
/// `lapmr`), plane rotations (`rot`, `lasr`, `lasr_rows`, `largv`, `lartv`, `lar2v`), Householder
/// reflectors (`larfg`, `larf`), norms (`lanst`, `lange`), scaling (`laqge`, `lascl`, `lacgv`),
/// packed and triangular copies (`trttp`, `tpttr`, `trimask`, `tri_expand`) and `geam`.
///
/// Ownership is the executor's cyclic layout (2-D: flat column-major, as `for_each_owned_2d`),
/// so operations on the same executor compose without a barrier unless one reads an element
/// another item wrote; each operation says when that happens. Pivots are 1-based `int32`, as in
/// LAPACK. Included by `<batchlas/device.hh>`.
/// @ingroup api_device_lapack_aux

#include <cstddef>
#include <type_traits>

#include <batchlas/device/level1.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device {

namespace alg {
/// @brief `lacpy` edge policy: `op(A)` covers `B`.
/// @ingroup api_device_lapack_aux
struct EdgeNone {};
/// @brief `lacpy` edge policy: `B`'s elements outside `op(A)` are not written.
/// @ingroup api_device_lapack_aux
struct EdgeGuard {};
/// @brief `lacpy` edge policy: `B` is written everywhere, 0 outside `op(A)` or outside the part (a padded tile load).
/// @ingroup api_device_lapack_aux
struct EdgeZeroPad {};
/// @brief `lacpy` algorithm: the Hermitian completion of a stored triangle, one row per item.
/// @ingroup api_device_lapack_aux
struct RowMirror {};
}  // namespace alg
/// @brief `B := op(A)` on part `P` of `B` (LAPACK lacpy).
///
/// `B` outside `P` is not written and `A` outside the mirrored part is not read. `B` must not
/// alias `A` when `OP` is not `NoTrans`. A conjugated source without a transpose is
/// `conj_view(A)`. Elementwise under `for_each_owned_2d` ownership, no barrier, except with
/// `alg::RowMirror`.
/// @ingroup api_device_lapack_aux
/// @tparam P the part of `B` written: `Part::Full` (default), `Part::Lower`, `Part::Upper` (or the
///   strict parts).
/// @tparam OP `Transpose::NoTrans` (default), `Trans` or `ConjTrans` (reads `A(j, i)`).
/// @tparam Alg `for_each_owned_2d`'s walk (default `alg::Walk`; `alg::RowOwner` copies whole rows),
///   or `alg::RowMirror`: with `P` `Lower` or `Upper`, `NoTrans`, `alg::EdgeZeroPad` and
///   `DiagFill::Real`, `B :=` the Hermitian matrix whose `P` triangle `A` stores. Item `r` reads row
///   `r` of the triangle and writes it and its conjugate mirror, so a barrier must precede reads
///   of other rows.
/// @tparam Edge how `op(A)`'s extent meets `B`'s: `alg::EdgeNone` (default), `alg::EdgeGuard` or
///   `alg::EdgeZeroPad`.
/// @tparam DF what `B`'s diagonal gets inside `P`: `DiagFill::Keep` (default, copied),
///   `DiagFill::Real` (`(re, 0)`, a Hermitian diagonal) or `DiagFill::Unit` (1, `A` not read).
/// @param ex the executor.
/// @param a the source `A`.
/// @param[out] b the destination `B`; its extents are the copy's.
/// @param live `false` (a masked-off item): `op(A)` is empty, so only `alg::EdgeZeroPad` writes (zeros).
template <Part P = Part::Full, Transpose OP = Transpose::NoTrans, class Alg = alg::Walk, class Edge = alg::EdgeNone,
          DiagFill DF = DiagFill::Keep, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b, bool live = true) {
    using T = typename MB::value_type;
    if constexpr (std::is_same_v<Alg, alg::RowMirror>) {
        static_assert((P == Part::Lower || P == Part::Upper) && OP == Transpose::NoTrans && std::is_same_v<Edge, alg::EdgeZeroPad> &&
                          DF == DiagFill::Real,
                      "lacpy: alg::RowMirror is the Hermitian completion of a Lower / Upper triangle (EdgeZeroPad, DiagFill::Real)");
        using A = Arith<T>;
        const int n = a.rows(), r = ex.lid();
        if (r < 0 || r >= b.rows()) return;
        if (live && r < n) {
            for (int c = 0; c < n; ++c) {
                if (P == Part::Upper ? r > c : r < c) continue;
                T v = a(r, c);
                if (r == c) {
                    if constexpr (A::is_complex) v = A::make(A::re(v), real_t<T>(0));
                } else {
                    b(c, r) = A::conj(v);
                }
                b(r, c) = v;
            }
            for (int c = n; c < b.cols(); ++c) b(r, c) = A::zero();
        } else {
            for (int c = 0; c < b.cols(); ++c) b(r, c) = A::zero();
        }
    } else if constexpr (std::is_same_v<Edge, alg::EdgeNone> && DF == DiagFill::Keep) {
        if (!live) return;
        for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) {
            if (!detail::in_part<P>(i, j)) return;
            if constexpr (OP == Transpose::NoTrans) b(i, j) = a(i, j);
            else if constexpr (OP == Transpose::Trans) b(i, j) = a(j, i);
            else b(i, j) = Arith<T>::conj(static_cast<T>(a(j, i)));
        });
    } else {
        using A = Arith<T>;
        static_assert(std::is_same_v<Edge, alg::EdgeNone> || std::is_same_v<Edge, alg::EdgeGuard> || std::is_same_v<Edge, alg::EdgeZeroPad>,
                      "lacpy: alg::EdgeNone, Guard or ZeroPad");
        static_assert(DF == DiagFill::Keep || DF == DiagFill::Real || DF == DiagFill::Unit,
                      "lacpy: DiagFill::Keep, Real or Unit");
        const int ar = OP == Transpose::NoTrans ? a.rows() : a.cols();
        const int ac = OP == Transpose::NoTrans ? a.cols() : a.rows();
        for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) {
            const bool inside = live && (std::is_same_v<Edge, alg::EdgeNone> || (i < ar && j < ac));
            if (!detail::in_part<P>(i, j) || !inside) {
                if constexpr (std::is_same_v<Edge, alg::EdgeZeroPad>) b(i, j) = A::zero();
                return;
            }
            if constexpr (DF == DiagFill::Unit) {
                if (i == j) {
                    b(i, j) = A::one();
                    return;
                }
            }
            T v;
            if constexpr (OP == Transpose::NoTrans) v = a(i, j);
            else if constexpr (OP == Transpose::Trans) v = a(j, i);
            else v = A::conj(static_cast<T>(a(j, i)));
            if constexpr (DF == DiagFill::Real && A::is_complex)
                if (i == j) v = A::make(A::re(v), real_t<T>(0));
            b(i, j) = v;
        });
    }
}

/// @brief `A := alpha` off the diagonal and `beta` on it, on part `P` (LAPACK laset).
///
/// Tags fold: `laset` of `Zero, Zero` is a store of zero with no select. Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam P the part written (default `Part::Full`).
/// @tparam Alg `for_each_owned_2d`'s walk (default `alg::Walk`).
/// @param ex the executor.
/// @param alpha_ the off-diagonal value: a value or a tag.
/// @param beta_ the diagonal value: a value or a tag.
/// @param[out] a the matrix.
template <Part P = Part::Full, class Alg = alg::Walk, class Ex, class AS, class BS, class MA>
BATCHLAS_DEVICE_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const T off = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T dia = tag_mul(as_tag(beta_), Arith<T>::one());
    for_each_owned_2d<Alg>(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i, j)) return;
        if constexpr (std::is_same_v<tag_t<AS>, tag_t<BS>> && !std::is_same_v<tag_t<AS>, Scalar<T>>) a(i, j) = off;
        else a(i, j) = select(i == j, dia, off);
    });
}
/// @brief `laset` on a block of a larger matrix: the diagonal and `P` are the parent's.
///
/// Element `(i, j)` of the block is the parent's `(i + off, j)` up to a common shift.
/// @ingroup api_device_lapack_aux
/// @tparam P the part of the parent written (default `Part::Full`).
/// @tparam Alg `for_each_owned_2d`'s walk (default `alg::Walk`).
/// @param ex the executor.
/// @param alpha_ the off-diagonal value: a value or a tag.
/// @param beta_ the diagonal value: a value or a tag.
/// @param[out] a the block.
/// @param off the block's row origin minus its column origin in the parent.
template <Part P = Part::Full, class Alg = alg::Walk, class Ex, class AS, class BS, class MA>
BATCHLAS_DEVICE_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a, int off) {
    using T = typename MA::value_type;
    const T o = tag_mul(as_tag(alpha_), Arith<T>::one());
    const T d = tag_mul(as_tag(beta_), Arith<T>::one());
    for_each_owned_2d<Alg>(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (!detail::in_part<P>(i + off, j)) return;
        a(i, j) = select(i + off == j, d, o);
    });
}
/// @brief Register form: one column of laset, `f[r] := beta` at `r == j` and `alpha` elsewhere.
/// @ingroup api_device_lapack_aux
/// @param[out] f the register fragment.
/// @param alpha_ the off-diagonal value: a value or a tag.
/// @param beta_ the diagonal value: a value or a tag.
/// @param j the column, so the diagonal's row.
template <class Scalar, int N, class L, class AS, class BS> BATCHLAS_DEVICE_INLINE void laset(Frag<Scalar, N, L>& f, AS alpha_, BS beta_, int j) {
    const Scalar o = tag_mul(as_tag(alpha_), Arith<Scalar>::one());
    const Scalar d = tag_mul(as_tag(beta_), Arith<Scalar>::one());
#pragma unroll
    for (int r = 0; r < N; ++r) f[r] = select(r == j, d, o);
}
/// @brief Cyclic register form: the item's share of a column of `N * size` rows, `beta` at row `j` and `alpha` elsewhere.
///
/// `f[k]` is row `lid + k * size`.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param alpha_ the off-diagonal value: a value or a tag.
/// @param beta_ the diagonal value: a value or a tag.
/// @param[out] f the item's fragment.
/// @param j the diagonal's row.
template <class Ex, class Scalar, int N, class AS, class BS>
BATCHLAS_DEVICE_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, Frag<Scalar, N, Cyclic>& f, int j) {
    const Scalar o = tag_mul(as_tag(alpha_), Arith<Scalar>::one());
    const Scalar d = tag_mul(as_tag(beta_), Arith<Scalar>::one());
    for_each_owned_frag<N * Ex::size>(ex, [&](int k, int r) { f[k] = select(r == j, d, o); });
}
/// @brief Register tile form: `t(i, j) := beta` where `i - j == off`, `alpha` elsewhere.
/// @ingroup api_device_lapack_aux
/// @param[out] t the register tile.
/// @param alpha_ the off-diagonal value: a value or a tag.
/// @param beta_ the diagonal value: a value or a tag.
/// @param off the diagonal's offset (default 0, the main diagonal).
template <class Scalar, int R, int C, class AS, class BS> BATCHLAS_DEVICE_INLINE void laset(Tile<Scalar, R, C>& t, AS alpha_, BS beta_, int off = 0) {
    const Scalar o = tag_mul(as_tag(alpha_), Arith<Scalar>::one());
    const Scalar d = tag_mul(as_tag(beta_), Arith<Scalar>::one());
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = select(i - j == off, d, o);
}

/// @brief Fills the strict triangle opposite `FROM` from `FROM`: `A(i, j) := A(j, i)`, conjugated under `ConjTrans`.
///
/// Each write reads only the `FROM` triangle, so the items do not race; a later operation that
/// reads the mirrored triangle across owners needs a barrier.
/// @ingroup api_device_lapack_aux
/// @tparam FROM the triangle read.
/// @tparam TR `Transpose::Trans` (default, symmetric) or `ConjTrans` (Hermitian: the mirror is
///   conjugated and the diagonal's imaginary part set to 0).
/// @tparam Alg `for_each_owned_2d`'s walk (default `alg::Walk`).
/// @param ex the executor.
/// @param[in,out] a the n x n matrix.
template <Uplo FROM, Transpose TR = Transpose::Trans, class Alg = alg::Walk, class Ex, class MA>
BATCHLAS_DEVICE_INLINE void mirror(const Ex& ex, const MA& a) {
    static_assert(TR != Transpose::NoTrans, "mirror: TR is Trans or ConjTrans");
    constexpr bool Herm = TR == Transpose::ConjTrans;
    using T = typename MA::value_type;
    using A = Arith<T>;
    const int n = a.rows();
    for_each_owned_2d<Alg>(ex, n, n, [&](int i, int j) {
        const bool dst = FROM == Uplo::Lower ? i < j : i > j;
        if (dst) a(i, j) = Herm ? A::conj(static_cast<T>(a(j, i))) : static_cast<T>(a(j, i));
        else if constexpr (Herm && A::is_complex) {
            if (i == j) a(i, j) = A::make(A::re(static_cast<T>(a(i, j))), real_t<T>(0));
        }
    });
}

/// @brief The order in which a sequence of interchanges or rotations is applied.
///
/// `Forward` applies them in ascending index order, `Backward` in descending order.
/// @ingroup api_device_lapack_aux
enum class Direct { Forward, Backward };
/// @brief Row interchanges (LAPACK laswp): for each `k` in `[k1, k2)`, swap rows `k` and `ipiv(k) - 1` of `A`.
///
/// Columns are owned cyclically and each item runs the whole interchange sequence on its
/// columns: no barrier, no race.
/// @ingroup api_device_lapack_aux
/// @tparam Dir `Direct::Forward` (default, ascending `k`) or `Direct::Backward`.
/// @param ex the executor.
/// @param[in,out] a the matrix.
/// @param ipiv the pivots, 1-based `int32` (LAPACK's convention), read at `[k1, k2)`.
/// @param k1 the first interchange.
/// @param k2 one past the last interchange.
template <Direct Dir = Direct::Forward, class Ex, class MA, class P>
BATCHLAS_DEVICE_INLINE void laswp(const Ex& ex, const MA& a, const P& ipiv, int k1, int k2) {
    using T = typename MA::value_type;
    for_each_owned(ex, a.cols(), [&](int j) {
        auto step = [&](int k) {
            const int p = static_cast<int>(ipiv(k)) - 1;
            if (p != k) {
                const T t = a(k, j);
                a(k, j) = a(p, j);
                a(p, j) = t;
            }
        };
        if constexpr (Dir == Direct::Forward)
            for (int k = k1; k < k2; ++k) step(k);
        else
            for (int k = k2 - 1; k >= k1; --k) step(k);
    });
}
/// @brief Where row `i` of `A` ends up after `laswp<Dir>(ex, A, ipiv, k1, k2)`: the interchanges
/// followed on one index, no memory touched.
///
/// `laswp_index<Direct::Backward>` of a position is the row that lands there under `Forward`.
/// @ingroup api_device_lapack_aux
/// @tparam Dir `Direct::Forward` (default) or `Direct::Backward`.
/// @param ipiv the pivots, 1-based.
/// @param k1 the first interchange.
/// @param k2 one past the last interchange.
/// @param i the row followed.
/// @return the row's final position, 0-based.
template <Direct Dir = Direct::Forward, class P> BATCHLAS_DEVICE_INLINE int laswp_index(const P& ipiv, int k1, int k2, int i) {
    int r = i;
    auto step = [&](int k) {
        const int p = static_cast<int>(ipiv(k)) - 1;
        if (r == k) r = p;
        else if (r == p) r = k;
    };
    if constexpr (Dir == Direct::Forward)
        for (int k = k1; k < k2; ++k) step(k);
    else
        for (int k = k2 - 1; k >= k1; --k) step(k);
    return r;
}
/// @brief `B := laswp<Dir>(A)` out of place: every element one gather, `for_each_owned_2d` ownership.
///
/// `B(i, j) = A(r, j)` with `r` the row that lands at `i`; there is no per-column interchange
/// chain. `B` must not alias `A`.
/// @ingroup api_device_lapack_aux
/// @tparam Dir `Direct::Forward` (default) or `Direct::Backward`.
/// @param ex the executor.
/// @param a the source `A`.
/// @param ipiv the pivots, 1-based.
/// @param k1 the first interchange.
/// @param k2 one past the last interchange.
/// @param[out] b the destination `B`.
template <Direct Dir = Direct::Forward, class Ex, class MA, class P, class MB>
BATCHLAS_DEVICE_INLINE void laswp_gather(const Ex& ex, const MA& a, const P& ipiv, int k1, int k2, const MB& b) {
    constexpr Direct Inv = Dir == Direct::Forward ? Direct::Backward : Direct::Forward;
    for_each_owned_2d(ex, b.rows(), b.cols(), [&](int i, int j) { b(i, j) = a(laswp_index<Inv>(ipiv, k1, k2, i), j); });
}
/// @brief Row permutation out of place (LAPACK lapmr) with a permutation `k`.
///
/// `Forward`: `B(i, :) = A(k(i) - Base, :)`. `Backward`: `B(k(i) - Base, :) = A(i, :)`. `B` must
/// not alias `A`.
/// @ingroup api_device_lapack_aux
/// @tparam Dir `Direct::Forward` (default) or `Direct::Backward`.
/// @tparam Alg `for_each_owned_2d`'s walk over `B` (default `alg::Walk`).
/// @tparam Base the index base of `k`: 1 (default, LAPACK's) or 0.
/// @param ex the executor.
/// @param a the source `A`.
/// @param k the permutation.
/// @param[out] b the destination `B`.
template <Direct Dir = Direct::Forward, class Alg = alg::Walk, int Base = 1, class Ex, class MA, class K, class MB>
BATCHLAS_DEVICE_INLINE void lapmr(const Ex& ex, const MA& a, const K& k, const MB& b) {
    static_assert(Base == 0 || Base == 1, "lapmr: Base is 0 or 1");
    for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int j) BATCHLAS_DEVICE_BODY {
        if constexpr (Dir == Direct::Forward) {
            if constexpr (Base == 0) b(i, j) = a(k(i), j);
            else b(i, j) = a(static_cast<int>(k(i)) - Base, j);
        } else {
            b(static_cast<int>(k(i)) - Base, j) = a(i, j);
        }
    });
}
/// @brief The pivot sink of a panel factorization: step `k`'s 0-based row is stored 1-based plus `base`.
///
/// With `base` the panel's first global row, a panel writes global pivots with no fix-up pass.
/// @ingroup api_device_lowlevel
struct IpivSink {
    /// The pivot array.
    int* p;
    /// The offset added to every pivot.
    int base;
    /// @brief `p[k] := base + row + 1`.
    /// @param k the step.
    /// @param row the step's 0-based pivot row within the panel.
    BATCHLAS_DEVICE_INLINE void set(int k, int row) const { p[k] = base + row + 1; }
};

// ---- plane rotations ----------------------------------------------------------------------------
namespace alg {
/// @brief Rotation convention of LAPACK zrot and lartg: `(x, y) := (c x + s y, c y - conj(s) x)`.
/// @ingroup api_device_lapack_aux
struct RotLapack {};
/// @brief Rotation convention of steqr and dlasr (pivot V): `(x, y) := (c x - s y, s x + c y)`, `s` real.
/// @ingroup api_device_lapack_aux
struct RotTransposed {};
/// @brief A 2-element Householder reflector `(v0, v1) = (g.c, g.s)`, real with unit norm: `t = x v0 + y v1`, then `x -= 2 v0 t` and `y -= 2 v1 t`.
/// @ingroup api_device_lapack_aux
struct RotReflector2 {};
}  // namespace alg
/// @brief `true` when `C` is a rotation convention (`alg::RotLapack`, `RotTransposed` or `RotReflector2`).
/// @ingroup api_device_lowlevel
template <class C> inline constexpr bool is_rot_conv_v =
    std::is_same_v<C, alg::RotLapack> || std::is_same_v<C, alg::RotTransposed> || std::is_same_v<C, alg::RotReflector2>;
namespace detail {
// zrot / drot in Scalar's own operators (std::complex keeps its Annex-G multiply), plain inline: the
// band-reduction chase's scalar kernel (arith::Native).
template <class Scalar, class Real, class S> inline void rot_native(Scalar& x, Scalar& y, Real c, const S& s_in) {
    if constexpr (Arith<Scalar>::is_complex) {
        const Scalar xt = x;
        const Scalar yt = y;
        x = Scalar(c) * xt + s_in * yt;
        y = Scalar(c) * yt - ::batchlas::device::conj(s_in) * xt;
    } else {
        const Real s = static_cast<Real>(s_in);
        const Real xt = static_cast<Real>(x);
        const Real yt = static_cast<Real>(y);
        x = Scalar(c * xt + s * yt);
        y = Scalar(c * yt - s * xt);
    }
}
}  // namespace detail
/// @brief Applies the plane rotation `g` to one pair `(x, y)`, in convention `Conv`.
/// @ingroup api_device_lapack_aux
/// @tparam Conv `alg::RotLapack` (default), `alg::RotTransposed` (real `s`) or `alg::RotReflector2`
///   (real `c` and `s`; `Scalar`'s own operators).
/// @tparam Ops the arithmetic of the complex products: `arith::Explicit` (default, `Arith<Scalar>`)
///   or `arith::Native` (`Scalar`'s own operators, `alg::RotLapack` only).
/// @param g the rotation `{c, s}`.
/// @param[in,out] x the first element.
/// @param[in,out] y the second element.
template <class Conv = alg::RotLapack, class Ops = arith::Explicit, class Real, class S, class Scalar>
BATCHLAS_DEVICE_INLINE void rot(const Givens<Real, S>& g, Scalar& x, Scalar& y) {
    if constexpr (detail::is_native_v<Ops>) {
        static_assert(std::is_same_v<Conv, alg::RotLapack>, "rot: arith::Native is alg::RotLapack");
        detail::rot_native(x, y, g.c, g.s);
    } else {
        using A = arith_t<Ops, Scalar>;
        const Scalar a = x, b = y;
        if constexpr (std::is_same_v<Conv, alg::RotLapack>) {
            if constexpr (A::is_complex) {
                const Scalar sb = A::mul(static_cast<Scalar>(g.s), b), sa = A::mul(A::conj(static_cast<Scalar>(g.s)),
                                                                                   a);
                x = A::make(g.c * A::re(a) + A::re(sb), g.c * A::im(a) + A::im(sb));
                y = A::make(g.c * A::re(b) - A::re(sa), g.c * A::im(b) - A::im(sa));
            } else {
                x = g.c * a + g.s * b;
                y = g.c * b - g.s * a;
            }
        } else if constexpr (std::is_same_v<Conv, alg::RotReflector2>) {
            static_assert(std::is_same_v<Real, S>, "rot: alg::RotReflector2 takes a real (v0, v1)");
            const Scalar t = x * g.c + y * g.s;
            x -= Real(2) * g.c * t;
            y -= Real(2) * g.s * t;
        } else {
            static_assert(std::is_same_v<Conv, alg::RotTransposed> && std::is_same_v<Real, S>, "rot: alg::RotTransposed takes a real s");
            if constexpr (A::is_complex) {
                x = A::make(g.c * A::re(a) - g.s * A::re(b), g.c * A::im(a) - g.s * A::im(b));
                y = A::make(g.s * A::re(a) + g.c * A::re(b), g.s * A::im(a) + g.c * A::im(b));
            } else {
                x = g.c * a - g.s * b;
                y = g.s * a + g.c * b;
            }
        }
    }
}
/// @brief Applies one plane rotation to every pair `(x(i), y(i))` (BLAS rot); elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam Conv the convention, as for the scalar form (default `alg::RotLapack`).
/// @tparam Ops the arithmetic, as for the scalar form.
/// @param ex the executor.
/// @param g the rotation `{c, s}`.
/// @param[in,out] x the first vector; its size is the trip count.
/// @param[in,out] y the second vector.
template <class Conv = alg::RotLapack, class Ops = arith::Explicit, class Ex, class Real, class S, class X, class Y>
BATCHLAS_DEVICE_INLINE void rot(const Ex& ex, const Givens<Real, S>& g, const X& x, const Y& y) {
    if constexpr (detail::is_native_v<Ops>) {
        for_each_owned(ex, x.size(), [&](int i) BATCHLAS_DEVICE_BODY { rot<Conv, Ops>(g, x(i), y(i)); });
    } else {
        for_each_owned(ex, x.size(), [&](int i) {
            auto& xi = x(i);
            auto& yi = y(i);
            rot<Conv, Ops>(g, xi, yi);
        });
    }
}
/// @brief Which pair a `lasr` rotation acts on (LAPACK's PIVOT).
///
/// For rotation `k`: `Variable` acts on `(k, k + 1)`, `Top` on `(0, k + 1)` and `Bottom` on
/// `(k, n - 1)`.
/// @ingroup api_device_lapack_aux
enum class Pivot { Variable, Top, Bottom };
/// @brief A sequence of plane rotations on one vector (LAPACK lasr on a row, SIDE = R, or a column, SIDE = L).
///
/// Rotation `k` of `rots(0)` to `rots(n - 2)` acts on the pair `PV` names, in convention `Conv`.
/// One vector is one dependence chain, so this runs on `Thread`.
/// @ingroup api_device_lapack_aux
/// @tparam PV `Pivot::Variable`, `Top` or `Bottom`.
/// @tparam Dir `Direct::Forward` (default: `k = 0, 1, ...`) or `Direct::Backward`.
/// @tparam Conv the convention (default `alg::RotTransposed`).
/// @param n the vector's length.
/// @param rots the rotations, called as `rots(k)`, returning a `Givens`.
/// @param[in,out] x the vector.
template <Pivot PV, Direct Dir = Direct::Forward, class Conv = alg::RotTransposed, class Rots, class X>
BATCHLAS_DEVICE_INLINE void lasr(const Thread&, int n, const Rots& rots, const X& x) {
    // One loop per direction, no step lambda: Forward is the pre-existing Variable loop verbatim.
    if constexpr (Dir == Direct::Forward) {
        for (int k = 0; k < n - 1; ++k) {
            const auto g = rots(k);  // a copy: the table is not re-read after a store
            auto& p = x(PV == Pivot::Top ? 0 : k);
            auto& q = x(PV == Pivot::Bottom ? n - 1 : k + 1);
            rot<Conv>(g, p, q);
        }
    } else {
        for (int k = n - 2; k >= 0; --k) {
            const auto g = rots(k);
            auto& p = x(PV == Pivot::Top ? 0 : k);
            auto& q = x(PV == Pivot::Bottom ? n - 1 : k + 1);
            rot<Conv>(g, p, q);
        }
    }
}
/// @brief `x := x G(0) G(1) ... G(n - 2)`: `lasr<Pivot::Variable, Direct::Forward, Conv>`.
/// @ingroup api_device_lapack_aux
/// @tparam Conv the convention (default `alg::RotTransposed`).
/// @param ex the `Thread` executor.
/// @param n the vector's length.
/// @param rots the rotations.
/// @param[in,out] x the vector.
template <class Conv = alg::RotTransposed, class Rots, class X> BATCHLAS_DEVICE_INLINE void lasr(const Thread& ex, int n, const Rots& rots, const X& x) {
    lasr<Pivot::Variable, Direct::Forward, Conv>(ex, n, rots, x);
}
/// @brief `A := A G(0) ... G(n - 2)` on every owned row (LAPACK lasr, SIDE = R, PIVOT = V, DIRECT = F).
///
/// Rows are owned cyclically; each row runs the serial chain over its columns. No barrier.
/// @ingroup api_device_lapack_aux
/// @tparam Conv the convention (default `alg::RotTransposed`).
/// @param ex the executor.
/// @param rots the `a.cols() - 1` rotations.
/// @param[in,out] a the matrix.
template <class Conv = alg::RotTransposed, class Ex, class Rots, class MA>
BATCHLAS_DEVICE_INLINE void lasr_rows(const Ex& ex, const Rots& rots, const MA& a) {
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int j = 0; j < a.cols() - 1; ++j) {
            const auto g = rots(j);
            auto& p = a(r, j);
            auto& q = a(r, j + 1);
            rot<Conv>(g, p, q);
        }
    });
}

// ---- Householder -------------------------------------------------------------------------------
namespace detail {
// |a / s|^2, dividing first; s > 0.
template <class Scalar> BATCHLAS_DEVICE_INLINE real_t<Scalar> abs2_scaled(Scalar a, real_t<Scalar> s) {
    using A = Arith<Scalar>;
    using R = real_t<Scalar>;
    if constexpr (A::is_complex) {
        const R x = A::re(a) / s;
        const R y = A::im(a) / s;
        return sycl::fma(x, x, y * y);
    } else {
        const R x = a / s;
        return x * x;
    }
}
}  // namespace detail

/// @brief Generates a Householder reflector (LAPACK larfg) for `(alpha, x)`, `x` distributed cyclically over `ex`.
///
/// On return `alpha` holds `beta` and `x` holds `v(1:)` (the unit head `v(0) = 1` is implied),
/// with `H = I - tau v v^H` and `H^H (alpha, x) = (beta, 0)`. Two reductions over scope `S` (the
/// maximum, then the scaled sum of squares), then the scaling of `arith::Fma`. No barrier: the
/// caller owns the one before `x` is read across owners. COLLECTIVE.
/// @ingroup api_device_lapack_aux
/// @tparam Conv `alg::LarfgLapack` (default, the only form; `larfg_lanes` takes `alg::Phase` and
///   `alg::RealBeta`).
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param[in,out] alpha the leading element, uniform; overwritten by `beta`.
/// @param[in,out] x the trailing vector; overwritten by `v(1:)`.
/// @return the `LarfgScalars` (`tau`, `beta`, whether `H` is the identity, and the scale applied).
template <class Conv = alg::LarfgLapack, Scope S = Scope::Own, class Ex, class Scalar, class X>
BATCHLAS_DEVICE_INLINE LarfgScalars<Scalar> larfg(const Ex& ex, Scalar& alpha, const X& x) {
    static_assert(std::is_same_v<Conv, alg::LarfgLapack>, "larfg: alg::LarfgLapack (larfg_lanes takes alg::Phase / RealBeta)");
    using A = Arith<Scalar>;
    using R = real_t<Scalar>;
    LarfgScalars<Scalar> h;
    auto amax_ri = [](const Scalar& v) {
        if constexpr (A::is_complex) return sycl::fmax(sycl::fabs(A::re(v)), sycl::fabs(A::im(v)));
        else return sycl::fabs(v);
    };
    R smax = amax_ri(alpha);
    for_each_owned(ex, x.size(), [&](int i) { smax = sycl::fmax(smax, amax_ri(static_cast<Scalar>(x(i)))); });
    smax = detail::scope_reduce<S>(ex, smax, sycl::maximum<R>());
    R ssq = R(0);  // the reduction stays outside the test: every item reaches it
    if (smax > R(0)) {
        for_each_owned(ex, x.size(), [&](int i) { ssq += detail::abs2_scaled(static_cast<Scalar>(x(i)), smax); });
    }
    ssq = detail::scope_reduce<S>(ex, ssq);
    h = larfg_scalars<alg::LarfgLapack>(alpha, ScaledSsq<R>{smax, ssq});
    alpha = h.beta;
    if (!h.identity) {
        if (h.use_mul) for_each_owned(ex, x.size(), [&](int i) { x(i) = FmaArith<Scalar>::mul(static_cast<Scalar>(x(i)), h.scale); });
        else for_each_owned(ex, x.size(),
                            [&](int i) { x(i) = FmaArith<Scalar>::div(static_cast<Scalar>(x(i)), h.scale); });
    }
    return h;
}

namespace alg {
/// @brief `larf` form: `C -= v (tau w)`, as LAPACK.
/// @ingroup api_device_lapack_aux
struct ScaleW {};
/// @brief `larf` form: `C -= (tau v) w` (`Side::Left` only).
/// @ingroup api_device_lapack_aux
struct ScaleV {};
}  // namespace alg
/// @brief Applies a Householder reflector (LAPACK larf): `C := H C` (`Side::Left`) or `C H`
/// (`Side::Right`), `H = I - tau v v^H`.
///
/// Pass `conj(tau)` to apply `H^H`. `tau == 0` returns at once. Every item reads all of `v`.
/// `Side::Left` on a team executor: team `t` owns columns `t mod nteams` and its lanes stride the
/// rows, with team reductions only (no work-group barrier). `Side::Right`: each item owns rows,
/// serial over the columns, or with `Loop = alg::Lockstep` team `t` owns rows `t mod nteams` and
/// its lanes stride the columns. On `Thread` both are serial. `C` is updated in place; the caller
/// owns any barrier before another owner reads it.
/// @ingroup api_device_lapack_aux
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam Ops the arithmetic policy.
/// @tparam Form `alg::ScaleW` (default) or `alg::ScaleV` (`Side::Left` only).
/// @tparam Loop `alg::Diverge` (default; team `t` walks its own columns to its own bound) or
///   `alg::Lockstep` (every team runs the same trip count, so its team reduction is reached by
///   every lane when teams share a sub-group).
/// @param ex the executor.
/// @param tau the scalar `tau`.
/// @param vp the vector `v`: a `UnitHead` (`v(head) = 1` implied, zero above) or any `Published`
///   vector, of `C`'s row count (Left) or column count (Right).
/// @param[in,out] c the matrix `C`.
template <Side SD, class Ops = arith::Explicit, class Form = alg::ScaleW, class Loop = alg::Diverge, class Ex, class Scalar, class V,
          class MC>
BATCHLAS_DEVICE_INLINE void larf(const Ex& ex, Scalar tau, const V& vp, const MC& c) {
    using A = arith_t<Ops, Scalar>;
    const auto& v = detail::unpub(vp);
    const int m = c.rows(), n = c.cols();
    if (A::is_zero(tau)) return;
    constexpr bool scale_w = std::is_same_v<Form, alg::ScaleW>;
    static_assert(scale_w || (std::is_same_v<Form, alg::ScaleV> && SD == Side::Left), "larf: ScaleV is Side::Left only");
    if constexpr (SD == Side::Left && !(scale_w && std::is_same_v<Loop, alg::Diverge>)) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex);
        for_each_team_owned<Loop>(ex, n, [&](int j, bool live) BATCHLAS_DEVICE_BODY {
            Scalar w = A::zero();
            if (live)
                for (int r = lane; r < m; r += tsize) A::mac_conj(w, static_cast<Scalar>(v(r)),
                                                                  static_cast<Scalar>(c(r, j)));
            w = ex.team_reduce(w);
            const Scalar tw = scale_w ? A::mul(tau, w) : w;
            if (live)
                for (int r = lane; r < m; r += tsize) {
                    Scalar cij = c(r, j);
                    if constexpr (scale_w) A::msub(cij, static_cast<Scalar>(v(r)), tw);
                    else A::msub(cij, A::mul(tau, static_cast<Scalar>(v(r))), tw);
                    c(r, j) = cij;
                }
        });
    } else if constexpr (SD == Side::Left) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex), nt = detail::nteams_of(ex);
        const int t0 = is_thread_v<Ex> ? 0 : ex.team();
        static_assert(Ex::sub_whole || is_work_group_v<Ex> || is_runtime_v<Ex>, "larf<Left>: a team or work-group executor");
        for (int j = t0; j < n; j += nt) {
            Scalar w = A::zero();
            for (int r = lane; r < m; r += tsize) A::mac_conj(w, static_cast<Scalar>(v(r)),
                                                              static_cast<Scalar>(c(r, j)));
            w = ex.team_reduce(w);
            const Scalar tw = A::mul(tau, w);
            for (int r = lane; r < m; r += tsize) {
                Scalar cij = c(r, j);
                A::msub(cij, static_cast<Scalar>(v(r)), tw);
                c(r, j) = cij;
            }
        }
    } else if constexpr (std::is_same_v<Loop, alg::Lockstep>) {
        const int lane = is_thread_v<Ex> ? 0 : ex.lane();
        const int tsize = detail::team_size_of(ex);
        for_each_team_owned<alg::Lockstep>(ex, m, [&](int r, bool live) BATCHLAS_DEVICE_BODY {
            Scalar w = A::zero();
            if (live)
                for (int j = lane; j < n; j += tsize) A::mac(w, static_cast<Scalar>(c(r, j)),
                                                             static_cast<Scalar>(v(j)));
            const Scalar tw = A::mul(tau, ex.team_reduce(w));
            if (live)
                for (int j = lane; j < n; j += tsize) {
                    Scalar crj = c(r, j);
                    A::msub(crj, tw, A::conj(static_cast<Scalar>(v(j))));
                    c(r, j) = crj;
                }
        });
    } else {
        for_each_owned(ex, m, [&](int r) {
            Scalar w = A::zero();
            for (int j = 0; j < n; ++j) A::mac(w, static_cast<Scalar>(c(r, j)), static_cast<Scalar>(v(j)));
            const Scalar tw = A::mul(w, tau);
            for (int j = 0; j < n; ++j) {
                Scalar crj = c(r, j);
                A::msub(crj, tw, A::conj(static_cast<Scalar>(v(j))));
                c(r, j) = crj;
            }
        });
    }
}

/// @brief Register column form: `c := (I - tau v v^H) c` for a column held in a register fragment.
///
/// Element `r` of the column is `c[r]`, `r < N`. `v` is read at every `r < N` with no predicate, so
/// it must be zero outside its support.
/// @ingroup api_device_lapack_aux
/// @tparam SD `Side::Left`, the only form.
/// @tparam Ops the arithmetic policy.
/// @param[in,out] c the column.
/// @param vp the vector `v`, `Published` (or a `unit_head` of one).
/// @param tau the scalar `tau`.
template <Side SD, class Ops = arith::Explicit, class Scalar, int N, class L, class V>
BATCHLAS_DEVICE_INLINE void larf(Frag<Scalar, N, L>& c, const V& vp, Scalar tau) {
    static_assert(SD == Side::Left, "larf (Frag): Side::Left, a column");
    using A = arith_t<Ops, Scalar>;
    const auto& v = detail::unpub(vp);
    Scalar dot = A::zero();
#pragma unroll
    for (int r = 0; r < N; ++r) A::mac_conj(dot, static_cast<Scalar>(v(r)), c[r]);
    const Scalar gamma = A::mul(tau, dot);
#pragma unroll
    for (int r = 0; r < N; ++r) A::msub(c[r], static_cast<Scalar>(v(r)), gamma);
}

/// @brief Generates a Householder reflector (LAPACK larfg) for a vector held one element per item.
///
/// Item `alpha_item` holds `alpha` and the items with `x_active` hold `x`. `||x||` (scaled by a
/// power of two, so no square under- or overflows) and `alpha` are made uniform, then every item
/// forms the scalars. On return `alpha_item` holds `beta`, the `x` items hold `v` (the unit head
/// is implied), and every item gets `tau`. COLLECTIVE.
/// @ingroup api_device_lapack_aux
/// @pre `n` is at most the executor's size.
/// @tparam Conv the scalars' convention: `alg::Phase` (default), `alg::PhaseN1` or `alg::RealBeta`
///   (LAPACK zlarfg: `beta` real); see `larfg_scalars`.
/// @tparam Ops the arithmetic policy of the scaling.
/// @param ex the executor (a sub-group or partition).
/// @param n the reflector's order (`alpha` plus `n - 1` elements of `x`); `n <= 1` gives `tau = 0`.
/// @param alpha_item the item that holds `alpha`.
/// @param[in,out] alpha `alpha` on `alpha_item`; `beta` there on return.
/// @param[in,out] x this item's element of `x`; `v` on return.
/// @param x_active `true` on the items that hold an element of `x`.
/// @return `tau`, the same on every item.
template <class Conv = alg::Phase, class Ops = arith::Explicit, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE Scalar larfg_lanes(const Ex& ex, int n, int alpha_item, Scalar& alpha, Scalar& x,
                                          bool x_active) {
    using Ar = arith_t<Ops, Scalar>;
    using A = Arith<Scalar>;
    using R = real_t<Scalar>;
    // ||x|| as LAPACK's nrm2 scales it: by 2^-e, e the exponent of max |x|, so no square under- or
    // overflows (Metal flushes |x| < 2^-63's square to 0: the reflector lost x(1) of a graded
    // matrix and was no longer orthogonal). A power of two scales exactly: without under- or
    // overflow the bits are the unscaled sum's.
    const R mag = x_active ? sycl::fmax(sycl::fabs(A::re(x)), sycl::fabs(A::im(x))) : R(0);
    const R m = detail::scope_reduce<Scope::Own>(ex, mag, sycl::maximum<R>());
    const int e = (m > R(0) && sycl::isfinite(m)) ? sycl::clamp(static_cast<int>(sycl::ilogb(m)), -120, 120) : 0;
    const R sc = sycl::ldexp(R(1), -e);
    const R xsq = x_active ? abs2(A::make(A::re(x) * sc, A::im(x) * sc)) : R(0);
    const R xnorm = sycl::sqrt(detail::scope_reduce<Scope::Own>(ex, xsq)) * sycl::ldexp(R(1), e);
    const Scalar alpha_b = ex.bcast(alpha, alpha_item);
    const auto s = larfg_scalars<Conv>(alpha_b, xnorm, n);
    if (ex.lid() == alpha_item) alpha = s.beta;
    else if (x_active && s.tau != Ar::zero()) x = Ar::mul(x, s.scale);
    return s.tau;
}

// ---- vectors of plane rotations (LAPACK xLARGV / xLARTV / xLAR2V), elementwise over ex ----------
// largv: lartg<W>'s W (e.g. alg::Promote); lartv / lar2v: alg::RotLapack, with Ops =
// arith::Native for T's own operators (the band-reduction chase).
namespace detail {
// lartg<W> into out-parameters, plain inline (the band-reduction chase's form).
template <class W, class Scalar, class Real> inline void lartg_out(const Scalar& f, const Scalar& g, Real& c, Scalar& s,
                                                                   Scalar& r) {
    auto [ct, st, rt] = lartg<W>(f, g);
    c = ct;
    s = st;
    r = rt;
}
}  // namespace detail
/// @brief A vector of plane rotations (LAPACK largv): rotation `i` from `(x(i), y(i))` by `lartg<W>`.
///
/// On return `x(i) := r`, `y(i) := s` and `c(i) := c`. Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam W `lartg`'s algorithm: `void` (default, `alg::LartgExact`), `alg::Promote` or `alg::LartgFast`.
/// @param ex the executor.
/// @param[in,out] x the first components; overwritten by `r`.
/// @param[in,out] y the second components; overwritten by `s`.
/// @param[out] c the cosines.
template <class W = void, class Ex, class X, class Y, class C> BATCHLAS_DEVICE_INLINE void largv(const Ex& ex, const X& x, const Y& y, const C& c) {
    using T = elem_t<X>;
    using R = real_t<T>;
    for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY {
        const T f = x(t);
        const T g = y(t);
        R ct = R(0);
        T st = T(0);
        T rt = T(0);
        detail::lartg_out<W>(f, g, ct, st, rt);
        c(t) = ct;
        y(t) = st;
        x(t) = rt;
    });
}
/// @brief Applies a vector of plane rotations (LAPACK lartv): `(x(i), y(i))` by rotation `i`.
///
/// `x(i) := c(i) x(i) + s(i) y(i)` and `y(i) := c(i) y(i) - conj(s(i)) x(i)`. Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam Rot `alg::RotLapack`, the only convention.
/// @tparam Ops `arith::Explicit` (default) or `arith::Native` (the element type's own operators).
/// @param ex the executor.
/// @param[in,out] x the first vector; its size is the trip count.
/// @param[in,out] y the second vector.
/// @param c the cosines (real).
/// @param s the sines.
template <class Rot = alg::RotLapack, class Ops = arith::Explicit, class Ex, class X, class Y, class C, class S>
BATCHLAS_DEVICE_INLINE void lartv(const Ex& ex, const X& x, const Y& y, const C& c, const S& s) {
    static_assert(std::is_same_v<Rot, alg::RotLapack>, "lartv: alg::RotLapack");
    if constexpr (detail::is_native_v<Ops>) {
        for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY { detail::rot_native(x(t), y(t), c(t), s(t)); });
    } else {
        for_each_owned(ex, x.size(), [&](int i) {
            using T = elem_t<X>;
            T xi = x(i), yi = y(i);
            rot<alg::RotLapack, Ops>(Givens<real_t<T>, std::remove_cvref_t<decltype(s(0))>>{c(i), s(i)}, xi, yi);
            x(i) = xi;
            y(i) = yi;
        });
    }
}
namespace detail {
template <class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DEVICE_INLINE void lar2v_lapack(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    using T = elem_t<Z>;
    using A = Arith<T>;
    using R = real_t<T>;
    for_each_owned(ex, x.size(), [&](int i) {
        const R xi = A::re(static_cast<T>(x(i))), yi = A::re(static_cast<T>(y(i)));
        const R ci = c(i);
        if constexpr (A::is_complex) {
            const T zi = z(i), si = s(i);
            const R zir = A::re(zi), zii = A::im(zi), sir = A::re(si), sii = A::im(si);
            const R t1r = sir * zir - sii * zii, t1i = sir * zii + sii * zir;
            const T t2 = A::make(ci * zir, ci * zii);
            const T t3 = A::make(A::re(t2) - sir * xi, A::im(t2) + sii * xi);
            const T t4 = A::make(A::re(t2) + sir * yi, -A::im(t2) + sii * yi);
            const R t5 = ci * xi + t1r, t6 = ci * yi - t1r;
            x(i) = A::make(ci * t5 + (sir * A::re(t4) + sii * A::im(t4)), R(0));
            y(i) = A::make(ci * t6 - (sir * A::re(t3) - sii * A::im(t3)), R(0));
            z(i) = A::make(ci * A::re(t3) + (sir * t6 + sii * t1i), ci * A::im(t3) + (sir * t1i - sii * t6));
        } else {
            const R zi = z(i), si = s(i);
            const R t1 = si * zi, t2 = ci * zi, t3 = t2 - si * xi, t4 = t2 + si * yi, t5 = ci * xi + t1, t6 = ci * yi - t1;
            x(i) = ci * t5 + si * t4;
            y(i) = ci * t6 - si * t3;
            z(i) = ci * t4 - si * t5;
        }
    });
}
// Netlib CLAR2V / ZLAR2V (DLAR2V for real Scalar) on one element, expression for expression in Scalar's
// own operators, plain inline (arith::Native: the band-reduction chase's scalar kernel).
template <class Scalar, class Real> inline void lar2v_native(Scalar& x, Scalar& y, Scalar& z, Real ci,
                                                             const Scalar& si) {
    if constexpr (Arith<Scalar>::is_complex) {
        const Real xi = static_cast<Real>(x.real());
        const Real yi = static_cast<Real>(y.real());
        const Scalar zi = z;
        const Real zir = static_cast<Real>(zi.real());
        const Real zii = static_cast<Real>(zi.imag());
        const Real sir = static_cast<Real>(si.real());
        const Real sii = static_cast<Real>(si.imag());
        const Real t1r = sir * zir - sii * zii;
        const Real t1i = sir * zii + sii * zir;
        const Scalar t2 = Scalar(ci * zir, ci * zii);
        const Scalar t3 = t2 - ::batchlas::device::conj(si) * Scalar(xi, Real(0));
        const Scalar t4 = ::batchlas::device::conj(t2) + si * Scalar(yi, Real(0));
        const Real t5 = ci * xi + t1r;
        const Real t6 = ci * yi - t1r;
        const Real xnew = ci * t5 + (sir * static_cast<Real>(t4.real()) + sii * static_cast<Real>(t4.imag()));
        const Real ynew = ci * t6 - (sir * static_cast<Real>(t3.real()) - sii * static_cast<Real>(t3.imag()));
        const Scalar znew = Scalar(ci) * t3 + ::batchlas::device::conj(si) * Scalar(t6, t1i);
        x = Scalar(xnew, Real(0));
        y = Scalar(ynew, Real(0));
        z = znew;
    } else {
        const Real xi = static_cast<Real>(x);
        const Real yi = static_cast<Real>(y);
        const Real zi = static_cast<Real>(z);
        const Real sr = static_cast<Real>(si);
        const Real t1 = sr * zi;
        const Real t2 = ci * zi;
        const Real t3 = t2 - sr * xi;
        const Real t4 = t2 + sr * yi;
        const Real t5 = ci * xi + t1;
        const Real t6 = ci * yi - t1;
        x = Scalar(ci * t5 + sr * t4);
        y = Scalar(ci * t6 - sr * t3);
        z = Scalar(ci * t4 - sr * t5);
    }
}
}  // namespace detail
/// @brief Applies a vector of plane rotations from both sides to 2 x 2 Hermitian matrices (LAPACK lar2v).
///
/// For each `i`: `[x z; conj(z) y] := G [x z; conj(z) y] G^H` with `G = [c conj(s); -s c]`, `x`
/// and `y` the real diagonals, in the expressions of LAPACK zlar2v and dlar2v, in their order.
/// Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam Rot `alg::RotLapack`, the only convention.
/// @tparam Ops `arith::Explicit` (default, explicit real arithmetic) or `arith::Native` (the
///   element type's own operators).
/// @param ex the executor.
/// @param[in,out] x the first diagonals; its size is the trip count.
/// @param[in,out] y the second diagonals.
/// @param[in,out] z the off-diagonals.
/// @param c the cosines (real).
/// @param s the sines.
template <class Rot = alg::RotLapack, class Ops = arith::Explicit, class Ex, class X, class Y, class Z, class C, class S>
BATCHLAS_DEVICE_INLINE void lar2v(const Ex& ex, const X& x, const Y& y, const Z& z, const C& c, const S& s) {
    static_assert(std::is_same_v<Rot, alg::RotLapack>, "lar2v: alg::RotLapack");
    if constexpr (detail::is_native_v<Ops>) {
        for_each_owned(ex, x.size(), [&](int t) BATCHLAS_DEVICE_BODY { detail::lar2v_native(x(t), y(t), z(t), c(t), s(t)); });
    } else {
        detail::lar2v_lapack(ex, x, y, z, c, s);
    }
}

// ---- norms ------------------------------------------------------------------------------------
/// @brief A norm of the symmetric tridiagonal matrix `(d, e)` (LAPACK lanst).
///
/// `e(i)` couples `d(i)` and `d(i + 1)`. COLLECTIVE; the result is uniform over scope `S`.
/// @ingroup api_device_lapack_aux
/// @tparam NT the norm:
///   - `NormType::Max`: the largest `|d(i)|` or `|e(i)|` by `fmax`, so a NaN entry is ignored (the
///     steqr rule). On `Thread`, `n >= 1`.
///   - `NormType::One` or `NormType::Inf` (equal for a symmetric matrix): the largest
///     `|e(i - 1)| + |d(i)| + |e(i)|`.
///   - `NormType::Frobenius`: `sqrt(sum d^2 + 2 sum e^2)`, unscaled.
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param d the n diagonal elements (real).
/// @param e the n - 1 off-diagonal elements (real).
/// @return the norm.
template <NormType NT, Scope S = Scope::Own, class Ex, class D, class E> BATCHLAS_DEVICE_INLINE auto lanst(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    static_assert(!Arith<R>::is_complex, "lanst: d and e are real");
    const int n = d.size();
    if constexpr (NT == NormType::Max && is_thread_v<Ex>) {
        // Serial: seeded with |d(n-1)| (n >= 1, dsteqr's block), no per-step i < n-1 test.
        R acc = sycl::fabs(d(n - 1));
        for (int i = 0; i < n - 1; ++i) acc = sycl::fmax(acc, sycl::fmax(sycl::fabs(d(i)), sycl::fabs(e(i))));
        return acc;
    } else if constexpr (NT == NormType::Max) {
        R m = R(0);
        for_each_owned(ex, n, [&](int i) {
            m = sycl::fmax(m, sycl::fabs(d(i)));
            if (i < n - 1) m = sycl::fmax(m, sycl::fabs(e(i)));
        });
        return allreduce<S>(ex, m, sycl::maximum<R>());  // partials are never NaN (fmax from 0)
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        R m = R(0);
        for_each_owned(ex, n, [&](int i) {
            const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
            const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
            m = sycl::max(m, sycl::fabs(d(i)) + left + right);
        });
        return allreduce<S>(ex, m, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "lanst: Max, One, Inf or Frobenius");
        R ss = R(0);
        for_each_owned(ex, n, [&](int i) {
            ss += d(i) * d(i);
            if (i < n - 1) ss += R(2) * e(i) * e(i);
        });
        return static_cast<R>(sycl::sqrt(allreduce<S>(ex, ss)));
    }
}
/// @brief The max norm of a symmetric tridiagonal held one element of `d` and `e` per item, by an ascending butterfly.
///
/// COLLECTIVE; uniform result.
/// @ingroup api_device_lapack_aux
/// @tparam NT `NormType::Max`, the only form.
/// @param ex the executor (a sub-group or partition).
/// @param d the diagonal as a lane vector.
/// @param e the off-diagonal as a lane vector.
/// @return the largest `|d(i)|` or `|e(i)|`.
template <NormType NT, class Ex, class Real> BATCHLAS_DEVICE_INLINE Real lanst(const Ex& ex, const LaneVec<Real>& d, const LaneVec<Real>& e) {
    static_assert(NT == NormType::Max, "lanst (LaneVec): NormType::Max");
    Real acc = Real(0);
    if (d.owns(ex)) acc = sycl::fabs(d.v);
    if (e.owns(ex)) acc = sycl::fmax(acc, sycl::fabs(e.v));
    return detail::butterfly_inline<1, Ex::size, alg::BflyAsc>(ex, acc, reduce_op::FMax{});
}

/// @brief A norm of a general m x n matrix (LAPACK lange); `|a|` is the modulus.
///
/// COLLECTIVE; the result is uniform over scope `S`.
/// @ingroup api_device_lapack_aux
/// @tparam NT the norm:
///   - `NormType::Max`: the largest `|a(i, j)|`, over the flat column-major ownership; keeps
///     `std::max(|a|, m)`'s NaN behaviour.
///   - `NormType::One`: the largest column sum; each item sums whole columns, then one maximum.
///   - `NormType::Inf`: the largest row sum; each item sums whole rows.
///   - `NormType::Frobenius`: `sqrt(sum |a|^2)`, as `Alg` says.
/// @tparam Alg the Frobenius algorithm: `alg::Unscaled` (default, one reduction; overflows past
///   `sqrt(max)`) or `alg::Scaled` (the two-pass lassq).
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param a the matrix.
/// @return the norm, of the real type of `a`'s elements.
template <NormType NT, class Alg = alg::Unscaled, Scope S = Scope::Own, class Ex, class MA>
BATCHLAS_DEVICE_INLINE auto lange(const Ex& ex, const MA& a) {
    using T = typename MA::value_type;
    using R = real_t<T>;
    const int m = a.rows(), n = a.cols();
    if constexpr (NT == NormType::Max) {
        R mx = R(0);
        for_each_owned_2d(ex, m, n, [&](int i, int j) {
            const R v = modulus(static_cast<T>(a(i, j)));
            mx = v < mx ? mx : v;
        });
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        constexpr bool one = NT == NormType::One;
        const int outer = one ? n : m, inner = one ? m : n;
        R mx = R(0);
        for (int k = ex.lid(); k < outer; k += Ex::size) {  // caller-owned shape (core 2.2 rule)
            R s = R(0);
            for (int l = 0; l < inner; ++l) s += modulus(static_cast<T>(one ? a(l, k) : a(k, l)));
            mx = sycl::maximum<R>()(mx, s);
        }
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "lange: Max, One, Inf or Frobenius");
        if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) { ss += abs2(static_cast<T>(a(i, j))); });
            return static_cast<R>(sycl::sqrt(detail::scope_reduce<S>(ex, ss)));
        } else {
            static_assert(std::is_same_v<Alg, alg::Scaled>, "lange: Frobenius is alg::Unscaled or alg::Scaled");
            using A = Arith<T>;
            R sc = R(0);
            auto upd = [&](R v) { sc = (v > sc || sycl::isnan(v)) ? v : sc; };
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                upd(sycl::fabs(A::re(v)));
                if constexpr (A::is_complex) upd(sycl::fabs(A::im(v)));
            });
            const R scale = detail::scope_reduce<S>(ex, sc, sycl::maximum<R>());
            if (sycl::isinf(scale)) return scale;
            const R inv = scale > R(0) ? R(1) / scale : R(1);
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                const R tr = A::re(v) * inv;
                ss += tr * tr;
                if constexpr (A::is_complex) {
                    const R ti = A::im(v) * inv;
                    ss += ti * ti;
                }
            });
            ss = detail::scope_reduce<S>(ex, ss);
            return scale > R(0) ? scale * sycl::sqrt(ss) : sycl::sqrt(ss);
        }
    }
}

// ---- equilibration, scaling, conjugation, packed copies ----------------------------------------
/// @brief Equilibrates `A` (LAPACK laqge): scales its rows by `r` and / or its columns by `c` when the ratios call for it.
///
/// Rows are scaled unless `rowcnd >= 0.1` and `small <= amax <= large` (`small = safmin / eps`,
/// `large = 1 / small`); columns unless `colcnd >= 0.1`. A scaled element is
/// `A(i, j) := c(j) r(i) A(i, j)` in that order; a real scale on a complex `A` scales both parts.
/// Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param[in,out] a the matrix.
/// @param r the row scale factors (real).
/// @param c the column scale factors (real).
/// @param rowcnd the ratio of the smallest to the largest `r(i)`.
/// @param colcnd the ratio of the smallest to the largest `c(j)`.
/// @param amax the largest absolute element of `A`.
/// @return the scaling done, uniform: `'N'` (none), `'R'` (rows), `'C'` (columns) or `'B'` (both).
template <class Ex, class MA, class RV, class CV, class Real>
BATCHLAS_DEVICE_INLINE char laqge(const Ex& ex, const MA& a, const RV& r, const CV& c, Real rowcnd, Real colcnd,
                                  Real amax) {
    using T = typename MA::value_type;
    constexpr Real thresh = Real(0.1);
    constexpr Real small = lamch<'S', Real>() / lamch<'P', Real>();
    constexpr Real large = Real(1) / small;
    const bool rows = !(rowcnd >= thresh && amax >= small && amax <= large);
    const bool cols = !(colcnd >= thresh);
    if (!rows && !cols) return 'N';
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        Real s;
        if (rows && cols) s = static_cast<Real>(c(j)) * static_cast<Real>(r(i));
        else if (rows) s = static_cast<Real>(r(i));
        else s = static_cast<Real>(c(j));
        a(i, j) = Arith<T>::scale_real(s, static_cast<T>(a(i, j)));
    });
    return rows ? (cols ? 'B' : 'R') : 'C';
}

namespace detail {
// A core fn_vec is computed on read, never stored: any item may read any index unpublished.
template <class V> inline constexpr bool is_fn_vec_v = false;
template <class F> inline constexpr bool is_fn_vec_v<device::FnVec<F>> = true;
}  // namespace detail
/// @brief `A := diag(r) A diag(c)` unconditionally, one row per item.
///
/// Unlike LAPACK laqge, no ratios are tested. Every item reads all of `c`.
/// @ingroup api_device_lowlevel
/// @pre the executor has exactly `a.rows()` items (`for_each_owned_exact`).
/// @param ex the executor.
/// @param r the row scale factors (real).
/// @param cp the column scale factors (real), `Published` or an `fn_vec`.
/// @param[in,out] a the matrix, a `Mat` (its pointer, offset and leading dimension are read).
template <class Ex, class Rv, class Cv, class MA> BATCHLAS_DEVICE_INLINE void laqge(const Ex& ex, const Rv& r, const Cv& cp, const MA& a) {
    using Ar = device::Arith<typename MA::value_type>;
    static_assert(device::is_published_v<Cv> || detail::is_fn_vec_v<Cv>, "laqge: c is read across owners; publish it");
    const auto& c = device::detail::unpub(cp);
    const int n = a.cols(), ld = int(a.ld);
    device::for_each_owned_exact(ex, a.rows(), [&](int i) {
        const auto ri = r(i);
        auto* const ai = a.p + a.off + i;  // row i moved once; a per-element i + j*ld blocks the unroll
        for (int j = 0; j < n; ++j) {
            const auto s = ri * c(j);
            auto& e = ai[j * ld];
            e = Ar::scale_real(s, e);
        }
    });
}

/// @brief `A := (cto / cfrom) A` on part `P` (LAPACK lascl), in steps that never over- or underflow.
///
/// Every item computes the same steps; each step is elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @tparam P the part scaled (default `Part::Full`).
/// @param ex the executor.
/// @param cfrom the denominator; must be nonzero.
/// @param cto the numerator.
/// @param[in,out] a the matrix.
template <Part P = Part::Full, class Ex, class Real, class MA> BATCHLAS_DEVICE_INLINE void lascl(const Ex& ex, Real cfrom, Real cto, const MA& a) {
    using T = typename MA::value_type;
    const Real smlnum = safmin<Real>(), bignum = Real(1) / smlnum;
    Real cfromc = cfrom, ctoc = cto;
    bool done = false;
    while (!done) {
        const Real cfrom1 = cfromc * smlnum;
        Real mul;
        if (cfrom1 == cfromc) {  // cfromc is Inf: a NaN or the exact quotient
            mul = ctoc / cfromc;
            done = true;
        } else {
            const Real cto1 = ctoc / bignum;
            if (cto1 == ctoc) {  // ctoc is 0 or Inf
                mul = ctoc;
                done = true;
                cfromc = Real(1);
            } else if (sycl::fabs(cfrom1) > sycl::fabs(ctoc) && ctoc != Real(0)) {
                mul = smlnum;
                cfromc = cfrom1;
            } else if (sycl::fabs(cto1) > sycl::fabs(cfromc)) {
                mul = bignum;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
            if (detail::in_part<P>(i, j)) a(i, j) = Arith<T>::scale_real(mul, static_cast<T>(a(i, j)));
        });
    }
}

/// @brief `x := conj(x)` (LAPACK lacgv); nothing for a real `x`. Elementwise, no barrier.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param[in,out] x the vector.
template <class Ex, class X> BATCHLAS_DEVICE_INLINE void lacgv(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    if constexpr (Arith<T>::is_complex) {
        map(ex, x, [](const T& v) { return Arith<T>::conj(v); }, x);
    }
}

// ---- triangle masks, geam and the SLM tile moves of transpose / mirror (src/util/elementwise.cc,
// transpose.cc, src/kernels/common/triangular_expand.hh) -----------------------------------------
// Uplo / Diag: a UploC (view.hh) / DiagC tag folds at compile time, a plain enum is read at run
// time (the legacy spelling, so a 1:1 port stays 1:1).
/// @brief A compile-time `Diag` tag for `tri_expand`: folds where a plain `Diag` is read at run time.
/// @ingroup api_device_lapack_aux
/// @tparam D the diagonal kind.
template <Diag D> struct DiagC {
    /// The diagonal kind.
    static constexpr Diag value = D;
};
/// @brief The compile-time `Uplo::Lower` tag (`UploC<Uplo::Lower>`).
/// @ingroup api_device_lapack_aux
using LowerC = UploC<Uplo::Lower>;
/// @brief The compile-time `Uplo::Upper` tag (`UploC<Uplo::Upper>`).
/// @ingroup api_device_lapack_aux
using UpperC = UploC<Uplo::Upper>;

namespace detail {
template <class U> BATCHLAS_DEVICE_INLINE constexpr bool is_lower([[maybe_unused]] U u) {
    if constexpr (std::is_same_v<U, Uplo>) return u == Uplo::Lower;
    else return U::value == Uplo::Lower;
}
template <class D> BATCHLAS_DEVICE_INLINE constexpr bool is_unit([[maybe_unused]] D d) {
    if constexpr (std::is_same_v<D, Diag>) return d == Diag::Unit;
    else return D::value == Diag::Unit;
}
}  // namespace detail

/// @brief A triangle with an offset, for `trimask`: `Upper` keeps `j - i >= k`, `Lower` keeps `j - i <= k`.
///
/// `k = 0` is the triangle with its diagonal.
/// @ingroup api_device_lapack_aux
/// @tparam U `Uplo` (read at run time, default) or a `UploC` tag.
template <class U = Uplo> struct Tri {
    /// The triangle.
    U uplo{};
    /// The diagonal offset.
    int k = 0;
};
/// @brief The `Tri` of `uplo` with offset `k`.
/// @ingroup api_device_lapack_aux
/// @param uplo `Uplo` or a `UploC` tag.
/// @param k the diagonal offset (default 0).
template <class U> BATCHLAS_DEVICE_INLINE Tri<U> tri(U uplo, int k = 0) { return {uplo, k}; }

/// @brief `C := A` inside the triangle `r`, 0 outside (tril / triu with an offset). `A` may alias `C`.
///
/// Over `C`'s extents, `for_each_owned_2d` ownership, no barrier.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param r the triangle (`tri(uplo, k)`).
/// @param a the source.
/// @param[out] c the destination.
template <class Ex, class U, class MA, class MC> BATCHLAS_DEVICE_INLINE void trimask(const Ex& ex, const Tri<U>& r, const MA& a, const MC& c) {
    using T = typename MC::value_type;
    const bool lower = detail::is_lower(r.uplo);
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        const bool keep = lower ? j - i <= r.k : j - i >= r.k;
        c(i, j) = keep ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}

/// @brief `O :=` the dense form of a triangular `A` stored in `uplo`: the stored strict triangle,
/// zeros opposite, and `A`'s diagonal or ones.
///
/// Storage outside the referenced triangle is never read. Over `O`'s extents,
/// `for_each_owned_2d` ownership, no barrier.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param uplo the stored triangle: `Uplo` (run time) or a `UploC` tag (compile time).
/// @param diag `Diag::Unit` (ones; the diagonal is never read) or `Diag::NonUnit`: a `Diag` or a `DiagC` tag.
/// @param a the triangular source.
/// @param[out] o the dense destination.
template <class Ex, class U, class D, class MA, class MO>
BATCHLAS_DEVICE_INLINE void tri_expand(const Ex& ex, U uplo, D diag, const MA& a, const MO& o) {
    using T = typename MO::value_type;
    const bool lower = detail::is_lower(uplo);
    const bool unit = detail::is_unit(diag);
    for_each_owned_2d(ex, o.rows(), o.cols(), [&](int i, int j) {
        T value;
        if (i == j) {
            value = unit ? Arith<T>::one() : static_cast<T>(a(i, j));
        } else if (lower ? (i > j) : (i < j)) {
            value = a(i, j);
        } else {
            value = Arith<T>::zero();
        }
        o(i, j) = value;
    });
}

/// @brief `C := alpha A + beta B` (BLAS-extension geam) in explicit arithmetic; `C` may alias `A` or `B`.
///
/// A `Zero` tag for `beta` never reads `B`. Over `C`'s extents, `for_each_owned_2d` ownership,
/// no barrier.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param alpha_ the scalar `alpha`: a value or a tag.
/// @param a the matrix `A`.
/// @param beta_ the scalar `beta`: a value or a tag.
/// @param b the matrix `B`.
/// @param[out] c the matrix `C`.
template <class Ex, class AS, class MA, class BS, class MB, class MC>
BATCHLAS_DEVICE_INLINE void geam(const Ex& ex, AS alpha_, const MA& a, BS beta_, const MB& b, const MC& c) {
    using T = typename MC::value_type;
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        if constexpr (std::is_same_v<tag_t<BS>, Zero>) c(i, j) = tag_mul(alpha, static_cast<T>(a(i, j)));
        else c(i, j) = Arith<T>::add(tag_mul(alpha, static_cast<T>(a(i, j))), tag_mul(beta, static_cast<T>(b(i, j))));
    });
}

// ---- SLM tiles on a Grid2D with a static group shape ----------------------------------------
// The item owns tile row r = lrow() and columns c = lcol() + s * GC < TN. The tile t is any
// Mat (Mem::Local); elements past the global operand's edge load as zero and are not stored.
// The caller owns the barrier between a load and a store of other items' elements.
namespace detail {
template <int TN, class Ex, class F> BATCHLAS_DEVICE_INLINE void tile_cols(const Ex& ex, F&& f) {
    constexpr int GC = Ex::static_group_cols;
    static_assert(GC > 0, "glue tile ops: Grid2D needs a static group shape");
    if constexpr (TN == GC) {
        f(ex.lcol());
    } else {
        for (int c = ex.lcol(); c < TN; c += GC) f(c);
    }
}
}  // namespace detail

/// @brief Local-memory tile load on a `Grid2D` with a static group shape: `t(r, c) = A(i0 + r, j0 + c)`, zero past `A`'s edge.
///
/// The item owns tile row `r = lrow()` and columns `c = lcol() + s * GC < TN`. The caller owns
/// the barrier between a load and a store of other items' elements.
/// @ingroup api_device_lowlevel
/// @tparam TN the tile's column count.
/// @param ex the `Grid2D` executor (static group columns `GC`).
/// @param a the global operand.
/// @param i0 the tile's first row in `A`.
/// @param j0 the tile's first column in `A`.
/// @param t the local tile (a `Mat` in `Mem::Local`).
template <int TN, class Ex, class MA, class MT> BATCHLAS_DEVICE_INLINE void tile_load(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        t(r, c) = (i < a.rows() && j < a.cols()) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}
/// @brief `tile_load` stored transposed: `t(c, r) = A(i0 + r, j0 + c)`.
/// @ingroup api_device_lowlevel
/// @tparam TN the tile's column count.
/// @param ex the `Grid2D` executor.
/// @param a the global operand.
/// @param i0 the tile's first row in `A`.
/// @param j0 the tile's first column in `A`.
/// @param t the local tile.
template <int TN, class Ex, class MA, class MT> BATCHLAS_DEVICE_INLINE void tile_load_trans(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        T v = Arith<T>::zero();
        if (i < a.rows() && j < a.cols()) v = a(i, j);
        t(c, r) = v;
    });
}
/// @brief Local-memory tile store: `B(i0 + r, j0 + c) = t(r, c)` inside `B`.
/// @ingroup api_device_lowlevel
/// @tparam TN the tile's column count.
/// @param ex the `Grid2D` executor.
/// @param t the local tile.
/// @param[out] b the global operand.
/// @param i0 the tile's first row in `B`.
/// @param j0 the tile's first column in `B`.
template <int TN, class Ex, class MT, class MB> BATCHLAS_DEVICE_INLINE void tile_store(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        if (i < b.rows() && j < b.cols()) b(i, j) = t(r, c);
    });
}
/// @brief The off-diagonal tile pair of a mirror in one column loop: `B(i0 + r, j0 + c) = t(r, c)` and `B(j0 + r, i0 + c) = op(t(c, r))`.
/// @ingroup api_device_lowlevel
/// @tparam TR `Transpose::Trans` (`t^T`) or `ConjTrans` (`t^H`).
/// @tparam TN the tile's column count.
/// @param ex the `Grid2D` executor.
/// @param t the local tile, square.
/// @param[out] b the global operand.
/// @param i0 the tile's first row.
/// @param j0 the tile's first column.
template <Transpose TR, int TN, class Ex, class MT, class MB>
BATCHLAS_DEVICE_INLINE void tile_store_pair(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    static_assert(TR != Transpose::NoTrans, "tile_store_pair: TR is Trans or ConjTrans");
    constexpr bool conj = TR == Transpose::ConjTrans;
    using T = typename MB::value_type;
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        if (i0 + r < b.rows() && j0 + c < b.cols()) b(i0 + r, j0 + c) = t(r, c);
        if (j0 + r < b.rows() && i0 + c < b.cols()) b(j0 + r, i0 + c) = conj_if<conj>(static_cast<T>(t(c, r)));
    });
}
/// @brief A diagonal tile (`i0 == j0`) of a mirror: the referenced member of each pair, `op(t(c, r))` opposite.
///
/// With `ConjTrans` the diagonal is forced real.
/// @ingroup api_device_lowlevel
/// @tparam TR `Transpose::Trans` or `ConjTrans`.
/// @tparam TN the tile's column count.
/// @param ex the `Grid2D` executor.
/// @param uplo the referenced triangle: `Uplo` or a `UploC` tag.
/// @param t the local tile.
/// @param[out] b the global operand.
/// @param i0 the tile's first row and column.
template <Transpose TR, int TN, class Ex, class U, class MT, class MB>
BATCHLAS_DEVICE_INLINE void tile_store_diag(const Ex& ex, U uplo, const MT& t, const MB& b, int i0) {
    static_assert(TR != Transpose::NoTrans, "tile_store_diag: TR is Trans or ConjTrans");
    constexpr bool conj = TR == Transpose::ConjTrans;
    using T = typename MB::value_type;
    const bool lower = detail::is_lower(uplo);
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = i0 + c;
        if (i >= b.rows() || j >= b.cols()) return;
        const bool referenced = lower ? (r >= c) : (r <= c);
        T value = referenced ? static_cast<T>(t(r, c)) : conj_if<conj>(static_cast<T>(t(c, r)));
        if constexpr (conj) {
            if (i == j) value = force_real(value);
        }
        b(i, j) = value;
    });
}

/// @brief Copies columns `[c0, c1)` of the `ul` triangle of an n x n matrix to packed storage
/// (LAPACK trttp, `to_packed`) or from it (tpttr).
///
/// The packed storage is by columns (`PackedTri<ul>`). `ul` and the direction are run-time values,
/// so one kernel serves both. Column by column, each column's rows cyclic over `ex`, no barrier;
/// elements outside `ul` are untouched.
/// @ingroup api_device_lapack_aux
/// @param ex the executor.
/// @param ul the triangle.
/// @param to_packed `true`: `a` to `ap`; `false`: `ap` to `a`.
/// @param n the order.
/// @param a the full matrix, `a(r, c)` a reference.
/// @param ap the packed storage: a pointer (64-bit offsets) or an `ap(k)` operand.
/// @param c0 the first column copied.
/// @param c1 one past the last column copied.
template <class Ex, class MA, class AP>
BATCHLAS_DEVICE_INLINE void trttp(const Ex& ex, Uplo ul, bool to_packed, int n, const MA& a, const AP& ap, int c0, int c1) {
    const bool lower = ul == Uplo::Lower;
    for (int c = c0; c < c1; ++c) {
        const std::ptrdiff_t cc = c;
        const std::ptrdiff_t off = lower ? cc * n - cc * (cc - 1) / 2 - cc : cc * (cc + 1) / 2;
        for_each_owned(ex, Range{lower ? c : 0, lower ? n : c + 1}, [&](int r) BATCHLAS_DEVICE_BODY {
            auto&& g = a(r, c);
            auto&& p = [&]() -> decltype(auto) {
                if constexpr (std::is_pointer_v<AP>) return ap[off + r];
                else return ap(static_cast<int>(off + r));
            }();
            if (to_packed)
                p = g;
            else
                g = p;
        });
    }
}
/// @brief Full triangle to packed storage (LAPACK trttp): `ap :=` the `UL` triangle of the square `a`.
/// @ingroup api_device_lapack_aux
/// @tparam UL the triangle.
/// @param ex the executor.
/// @param a the square matrix.
/// @param[out] ap the packed storage, `n (n + 1) / 2` elements.
template <Uplo UL, class Ex, class MA, class AP> BATCHLAS_DEVICE_INLINE void trttp(const Ex& ex, const MA& a, const AP& ap) {
    trttp(ex, UL, true, a.rows(), a, ap, 0, a.rows());
}
/// @brief Packed storage to the full triangle (LAPACK tpttr): the `UL` triangle of the square `a := ap`.
/// @ingroup api_device_lapack_aux
/// @tparam UL the triangle.
/// @param ex the executor.
/// @param ap the packed storage.
/// @param[out] a the square matrix; elements outside `UL` are untouched.
template <Uplo UL, class Ex, class AP, class MA> BATCHLAS_DEVICE_INLINE void tpttr(const Ex& ex, const AP& ap, const MA& a) {
    trttp(ex, UL, false, a.rows(), a, ap, 0, a.rows());
}

// ---- trsm's row-packed lower triangle (local memory) -------------------------------------------
/// @brief `ap := diag(L(0:n, 0:n), I)` as an `N` x `N` lower triangle packed by rows; no barrier.
///
/// Element `(s, t <= s)` is at `s (s + 1) / 2 + t` (`PackedTri<Uplo::Lower, PackOrder::ByRows>`).
/// @ingroup api_device_lowlevel
/// @tparam N the packed order.
/// @param ex the executor.
/// @param l the lower triangle `L`, read at `(s, t)` with `t <= s < n`.
/// @param n the live order, at most `N`.
/// @param[out] ap the packed storage, `N (N + 1) / 2` elements.
template <int N, class Ex, class L, class Scalar> BATCHLAS_DEVICE_INLINE void tp_pack_lower(const Ex& ex, const L& l, int n, Scalar* ap) {
    using Ar = Arith<Scalar>;
    constexpr std::size_t elems = static_cast<std::size_t>(N) * (N + 1) / 2;
    const std::size_t stride = static_cast<std::size_t>(detail::size_of(ex));
    for (std::size_t idx = ex.lid(); idx < elems; idx += stride) {
        int s = 0;
        while (PackedTri<Uplo::Lower, PackOrder::ByRows>::index(s + 1, 0, 0) <= static_cast<int>(idx)) ++s;
        const int t = static_cast<int>(idx) - PackedTri<Uplo::Lower, PackOrder::ByRows>::index(s, 0, 0);
        Scalar v;
        if (s >= n || t >= n) v = (s == t) ? Ar::one() : Ar::zero();
        else v = l(s, t);
        ap[idx] = v;
    }
}

/// @brief `rd(s) := 1 / ap(s, s)` for `s < n` of a row-packed lower triangle (1 when `unit` or `s >= n`); no barrier.
///
/// A non-finite reciprocal is replaced by 1 and stores 1 into `*bad` (a work-group-scope relaxed
/// atomic in local memory).
/// @ingroup api_device_lowlevel
/// @tparam N the packed order.
/// @tparam Ar the arithmetic of the reciprocal.
/// @param ex the executor.
/// @param ap the packed triangle.
/// @param n the live order.
/// @param unit `true`: a unit diagonal, every `rd(s)` is 1.
/// @param[out] rd the `N` reciprocals.
/// @param bad a local flag set to 1 on a non-finite reciprocal.
template <int N, class Ar, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE void tp_diag_recip(const Ex& ex, const Scalar* ap, int n, bool unit, Scalar* rd, int* bad) {
    const int stride = detail::size_of(ex);
    for (int s = ex.lid(); s < N; s += stride) {
        Scalar r = Ar::one();
        if (s < n && !unit) {
            const Scalar d = ap[PackedTri<Uplo::Lower, PackOrder::ByRows>::index(s, s, 0)];
            r = Ar::recip(d);
            if (!Ar::isfinite(r)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(bad[0])
                    .store(1);
                r = Ar::one();
            }
        }
        rd[s] = r;
    }
}

// ---- Side::Left staging tile ------------------------------------------------------------------
/// @brief `tile(r, c) := src(r0 + r, c0 + c)` inside `src`, else 0, for `r < TR` and `c` below the executor's size.
///
/// `tile` is column-major with leading dimension `LDT`; the flat index is distributed cyclically.
/// @ingroup api_device_lowlevel
/// @tparam TR the tile's row count.
/// @tparam LDT the tile's leading dimension.
/// @param ex the executor.
/// @param src the source matrix.
/// @param r0 the tile's first row in `src`.
/// @param c0 the tile's first column in `src`.
/// @param[out] tile the tile.
template <int TR, int LDT, class Ex, class Src, class Scalar>
BATCHLAS_DEVICE_INLINE void lacpy_tile_in(const Ex& ex, const Src& src, int r0, int c0, Scalar* tile) {
    const int stride = detail::size_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        Scalar v = Arith<Scalar>::zero();
        if (s < src.rows() && col < src.cols()) v = src(s, col);
        tile[c * LDT + r] = v;
    }
}

/// @brief `dst(r0 + r, c0 + c) := tile(r, c)` where that element is inside `dst`; the inverse of `lacpy_tile_in`.
/// @ingroup api_device_lowlevel
/// @tparam TR the tile's row count.
/// @tparam LDT the tile's leading dimension.
/// @param ex the executor.
/// @param tile the tile.
/// @param[out] dst the destination matrix.
/// @param r0 the tile's first row in `dst`.
/// @param c0 the tile's first column in `dst`.
template <int TR, int LDT, class Ex, class Scalar, class Dst>
BATCHLAS_DEVICE_INLINE void lacpy_tile_out(const Ex& ex, const Scalar* tile, const Dst& dst, int r0, int c0) {
    const int stride = detail::size_of(ex);
    for (int i = ex.lid(); i < TR * stride; i += stride) {
        const int r = i % TR;
        const int c = i / TR;
        const int s = r0 + r;
        const int col = c0 + c;
        if (s < dst.rows() && col < dst.cols()) dst(s, col) = tile[c * LDT + r];
    }
}

/// @brief `p[j] := x[s0 + j]` for `j < TR` with `s0 + j < N`: a window of a register fragment.
/// @ingroup api_device_lowlevel
/// @tparam TR the window's length.
/// @param x the fragment.
/// @param s0 the window's first slot.
/// @param[out] p the destination.
template <int TR, int N, class Scalar> BATCHLAS_DEVICE_INLINE void frag_copy_out(const Frag<Scalar, N>& x, int s0,
                                                                                 Scalar* p) {
    BATCHLAS_UNROLL_FULL
    for (int j = 0; j < TR; ++j) {
        const int s = s0 + j;
        if (s < N) p[j] = x[s];
    }
}

}  // namespace batchlas::device
