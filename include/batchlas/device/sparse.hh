// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-kernel sparse operations: CSR operands, the sparse-dense product `spmm` and incomplete
/// LU, `iluk_factorize` and `iluk_apply` (@ref api_device_sparse).
///
///   - `spmm<TA, TB, Tier>(ex, alpha, A, B, beta, C[, ws])`: `C := alpha op(A) op(B) + beta C`.
///   - `iluk_factorize<Tier>(ex, A, plan, M, opts[, ws])`: `M :=` the ILU(k) factor of `A` in the
///     plan's pattern.
///   - `iluk_apply<Tier>(ex, M, rhs, y[, ws])`: `y := (L U)^{-1} rhs`, one column.
///
/// Each operation runs one batch item on one executor for any extent; the host's sparse routes
/// launch the same bodies, so the results match the host's. The tiers are `tier::Blocked<NC>` for
/// `spmm` (`NC` the column block of one item's accumulators) and `tier::Blocked<1>` for the two ILU
/// operations (level-scheduled, a barrier per level). All admit any n, read and write their
/// operands where the caller put them (global memory, or local memory the caller sized), and need
/// no workspace: `device::buffer_size<R, Tier, T>(n)` is empty.
///
/// Neither umbrella header includes this one: include `<batchlas/device/sparse.hh>`.
/// @ingroup api_device_sparse
// Element expressions match the hand kernels they replaced (int64 offsets where those used
// them), so the host kernels keep their SASS.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::device {

/// @brief One scalar access, or with `Pair` a complex element as one 2-wide vector access.
/// @ingroup api_device_lowlevel
/// @tparam Pair `true`: a complex element is read and written as one `sycl::vec<real, 2>`, which
///   needs `2 * sizeof(real)` alignment.
template <bool Pair> struct PairAccess {
    /// @brief `*p`.
    /// @param p the element.
    template <class Scalar> BATCHLAS_DEVICE_INLINE static std::remove_cv_t<Scalar> load(Scalar* p) {
        using U = std::remove_cv_t<Scalar>;
        using A = Arith<U>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            const sycl::vec<R, 2> v = *reinterpret_cast<const sycl::vec<R, 2>*>(p);
            return A::make(v[0], v[1]);
        } else {
            return *p;
        }
    }
    /// @brief `*p := x`.
    /// @param p the element.
    /// @param x the value.
    template <class Scalar> BATCHLAS_DEVICE_INLINE static void store(Scalar* p, Scalar x) {
        using A = Arith<Scalar>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            *reinterpret_cast<sycl::vec<R, 2>*>(p) = sycl::vec<R, 2>(A::re(x), A::im(x));
        } else {
            *p = x;
        }
    }
};

/// @brief A relaxed device-scope atomic `*p += v` in global memory.
///
/// A complex value is two independent real atomics, not an atomic complex add: correct only while
/// nothing reads `*p` between the first and the last update.
/// @ingroup api_device_lowlevel
/// @param p the element, in global memory.
/// @param v the value added.
template <class Scalar> BATCHLAS_DEVICE_INLINE void atomic_accumulate(Scalar* p, Scalar v) {
    using A = Arith<Scalar>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        atomic_accumulate(reinterpret_cast<R*>(p), A::re(v));
        atomic_accumulate(reinterpret_cast<R*>(p) + 1, A::im(v));
    } else {
        sycl::atomic_ref<Scalar, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*p);
        ref.fetch_add(v);
    }
}

// ---- operands ------------------------------------------------------------------------------
/// @brief A vector with element `i` at `p[base + i]`, the sum formed in `Off`.
/// @ingroup api_device_lowlevel
/// @tparam T the element type (`const` for read-only).
/// @tparam Off the offset type: `int` (default) or `std::int64_t` where `base = b * stride` can pass 2^31.
template <class T, class Off = int> struct BasedVec {
    /// The element type without cv.
    using value_type = std::remove_cv_t<T>;
    /// The base pointer.
    T* p;
    /// The offset of element 0.
    Off base;
    /// @brief Element `i`, `p[base + i]`.
    /// @param i the index.
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const { return p[base + i]; }
};

/// @brief A read-only vector: element `i` is `PairAccess<Pair>::load(p + i)`.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
/// @tparam Pair `PairAccess`'s pair flag (default `false`).
template <class Scalar, bool Pair = false> struct LoadVec {
    /// The element type without cv.
    using value_type = std::remove_cv_t<Scalar>;
    /// The first element.
    Scalar* p;
    /// @brief Element `i`, by value.
    /// @param i the index.
    BATCHLAS_DEVICE_INLINE value_type operator()(int i) const { return PairAccess<Pair>::load(p + i); }
};

/// @brief Batch-interleaved (slot-major) storage: slot `s` of item `off` at `p[s * stride + off]`.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
// stride and off stay int and widen at each use (as size_t they cost ILU eliminate +3..5 REG).
template <class Scalar> struct InterleavedVec {
    /// The element type without cv.
    using value_type = std::remove_cv_t<Scalar>;
    /// The storage.
    Scalar* p;
    /// The distance between slots.
    int stride;
    /// The item's offset within a slot.
    int off;
    /// @brief Slot `s` of the item, `p[s * stride + off]` with the product formed in `std::size_t`.
    /// @param s the slot.
    BATCHLAS_DEVICE_INLINE Scalar& operator()(int s) const { return p[static_cast<std::size_t>(s) * stride + off]; }
};

/// @brief One CSR row segment: entries `p` in `[begin, end)` are `(val(p), idx(p))`, positions absolute.
/// @ingroup api_device_lowlevel
/// @tparam VV the values' vector type.
/// @tparam IV the column indices' vector type.
template <class VV, class IV> struct SpRow {
    /// The value type.
    using value_type = typename VV::value_type;
    /// The values, indexed by absolute position.
    VV vals;
    /// The column indices, indexed by absolute position.
    IV cols;
    /// The row's first position.
    int begin;
    /// One past the row's last position.
    int end;
    /// @brief The value at position `p`.
    /// @param p an absolute position in `[begin, end)`.
    BATCHLAS_DEVICE_INLINE value_type val(int p) const { return vals(p); }
    /// @brief The column index at position `p`.
    /// @param p an absolute position in `[begin, end)`.
    BATCHLAS_DEVICE_INLINE int idx(int p) const { return cols(p); }
};
/// @brief The row segment `[begin, end)` of `(vals, cols)`.
/// @ingroup api_device_lowlevel
/// @param vals the values.
/// @param cols the column indices.
/// @param begin the first position.
/// @param end one past the last position.
template <class VV, class IV> BATCHLAS_DEVICE_INLINE SpRow<VV, IV> sp_row(VV vals, IV cols, int begin, int end) {
    return {vals, cols, begin, end};
}

/// @brief One batch item of a CSR batch, the `A` operand of `spmm`; build it with `csr_item`.
///
/// Values and column indices start at `nz_base = b * matrix_stride` and the item-local row
/// offsets at `ro_base = b * offset_stride`; the extents are m x n as stored. Row `r` is
/// `[ro(r), ro(r + 1))`; nothing at or past `ro(m)` is read.
/// @ingroup api_device_sparse
/// @tparam Scalar the element type (`const` for read-only).
/// @tparam Off the offset type (default `std::int64_t`).
template <class Scalar, class Off = std::int64_t> struct CsrItem {
    /// The element type without cv.
    using value_type = std::remove_cv_t<Scalar>;
    /// The batch's values.
    Scalar* val;
    /// The batch's column indices.
    const int* col;
    /// The batch's row offsets.
    const int* ro;
    /// The item's offset into `val` and `col`.
    Off nz_base;
    /// The item's offset into `ro`.
    Off ro_base;
    /// @{
    /// The row and column counts.
    int m, n;
    /// @}
    /// @brief The row count `m`.
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    /// @brief The column count `n`.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
    /// @brief Row `r` as an `SpRow`, positions relative to the item.
    /// @tparam Pair `PairAccess`'s pair flag for the values (default `false`).
    /// @param r the row, in `[0, m)`.
    template <bool Pair = false> BATCHLAS_DEVICE_INLINE auto row(int r) const {
        return sp_row(LoadVec<Scalar, Pair>{val + nz_base}, BasedVec<const int, Off>{col, nz_base}, ro[ro_base + r], ro[ro_base + r + 1]);
    }
};
/// @brief Batch item `b` of a host CSR view, as a read-only `CsrItem`.
/// @ingroup api_device_sparse
/// @param a the host view (`KernelMatrixView<Scalar, MatrixFormat::CSR>`), captured by value.
/// @param b the batch index.
/// @return the item's operand.
template <class Scalar> BATCHLAS_DEVICE_INLINE CsrItem<const Scalar> csr_item(const KernelMatrixView<Scalar, MatrixFormat::CSR>& a, std::int64_t b) {
    return {a.data_, a.col_indices_, a.row_offsets_, b * a.matrix_stride_, b * a.offset_stride_, a.rows_, a.cols_};
}

/// @brief `op(B)(r, c)` of a column-major `B` whose transposition is a run-time value (one instantiation).
///
/// `notrans`: `p[c * ld + r]`; otherwise `p[r * ld + c]`, conjugated when `conj`.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
/// @tparam Pair `PairAccess`'s pair flag (default `false`).
template <class Scalar, bool Pair = false> struct RtOpMat {
    /// The element type without cv.
    using value_type = std::remove_cv_t<Scalar>;
    /// `B`'s first element.
    Scalar* p;
    /// `B`'s leading dimension.
    std::int64_t ld;
    /// `true` for `op(B) = B`.
    bool notrans;
    /// `true` to conjugate (with `notrans == false`).
    bool conj;
    /// @brief Element `(r, c)` of `op(B)`.
    /// @param r the row.
    /// @param c the column.
    BATCHLAS_DEVICE_INLINE value_type operator()(int r, int c) const {
        value_type v = PairAccess<Pair>::load(notrans ? p + static_cast<std::int64_t>(c) * ld + r
                                                      : p + static_cast<std::int64_t>(r) * ld + c);
        if constexpr (Arith<value_type>::is_complex) {
            if (conj) v = Arith<value_type>::conj(v);
        }
        return v;
    }
};

/// @brief A column-major `C` with a 64-bit leading dimension whose elements read and write through `PairAccess`.
///
/// `C(r, c)` is a proxy reference to `p + (c * ld + r)`, so `store_epi` can take it as its output.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
/// @tparam Pair `PairAccess`'s pair flag (default `false`).
template <class Scalar, bool Pair = false> struct ColMat64 {
    /// The element type without cv.
    using value_type = std::remove_cv_t<Scalar>;
    /// @brief A proxy reference to one element.
    struct Ref {
        /// The element.
        Scalar* q;
        /// @brief Reads the element.
        BATCHLAS_DEVICE_INLINE operator value_type() const { return PairAccess<Pair>::load(q); }
        /// @brief Writes the element.
        /// @param x the value.
        BATCHLAS_DEVICE_INLINE void operator=(value_type x) const { PairAccess<Pair>::store(q, x); }
    };
    /// `C`'s first element.
    Scalar* p;
    /// `C`'s leading dimension.
    std::int64_t ld;
    /// @brief The address of `C(r, c)`.
    /// @param r the row.
    /// @param c the column.
    BATCHLAS_DEVICE_INLINE Scalar* ptr(int r, int c) const { return p + (static_cast<std::int64_t>(c) * ld + r); }
    /// @brief The proxy reference to `C(r, c)`.
    /// @param r the row.
    /// @param c the column.
    BATCHLAS_DEVICE_INLINE Ref operator()(int r, int c) const { return {ptr(r, c)}; }
};

// ---- CSR row products ----------------------------------------------------------------------
/// @brief `acc - sum a.val(p) y(a.idx(p))` over the row's entries with column below `bound`, left to right.
///
/// The row is sorted, so the first column at or past `bound` ends it; nothing at or past it is
/// read. One item (`Thread`).
/// @ingroup api_device_lowlevel
/// @tparam Ar the arithmetic of the multiply-subtracts.
/// @param acc the starting value.
/// @param a the sorted CSR row (`SpRow`).
/// @param y the dense vector.
/// @param bound the column bound.
/// @return the updated value.
template <class Ar, class Row, class Y, class Scalar>
BATCHLAS_DEVICE_INLINE Scalar usdot_sub_below(const Thread&, Scalar acc, const Row& a, const Y& y, int bound) {
    for (int p = a.begin; p < a.end; ++p) {
        const int j = a.idx(p);
        if (j >= bound) break;
        Ar::msub(acc, a.val(p), y(j));
    }
    return acc;
}

/// @brief One output row of a sparse-dense product: `acc[t] += sum_p a.val(p) b(a.idx(p), c0 + t)` for `c0 + t < width`.
/// @ingroup api_device_lowlevel
/// @tparam Ar the arithmetic of the multiply-adds.
/// @param a the CSR row.
/// @param b the dense operand, `b(row, col)`.
/// @param c0 the column block's first column.
/// @param width the column count.
/// @param[in,out] acc the `NC` accumulators.
template <class Ar, class Row, class B, class Scalar, int NC, class L>
BATCHLAS_DEVICE_INLINE void usmm_row(const Thread&, const Row& a, const B& b, int c0, int width, Frag<Scalar, NC,
                                     L>& acc) {
    for (int p = a.begin; p < a.end; ++p) {
        const Scalar av = a.val(p);
        const int j = a.idx(p);
        BATCHLAS_UNROLL_FULL
        for (int t = 0; t < NC; ++t) {
            const int c = c0 + t;
            if (c < width) Ar::mac(acc[t], av, b(j, c));
        }
    }
}

/// @brief `C(r, c0 + t) := alpha acc[t] + beta C(r, c0 + t)` (`store_epi<epi::MacChain, Ops>`) for `c0 + t < width`.
///
/// A `beta` with its zero flag set never reads `C`.
/// @ingroup api_device_lowlevel
/// @tparam Ops the arithmetic policy.
/// @param alpha the product's scale.
/// @param acc the `NC` accumulators.
/// @param beta `C`'s scale, with its zero flag.
/// @param C the output matrix.
/// @param r the row.
/// @param c0 the column block's first column.
/// @param width the column count.
template <class Ops, class Scalar, int NC, class L, class CM>
BATCHLAS_DEVICE_INLINE void store_cols(const Thread&, Scalar alpha, const Frag<Scalar, NC, L>& acc, MaybeZero<Scalar> beta, const CM& C, int r,
                                    int c0, int width) {
    BATCHLAS_UNROLL_FULL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        if (c < width) store_epi<epi::MacChain, Ops>(alpha, acc[t], beta, C(r, c));
    }
}

/// @brief `w[t] := alpha b(r, c0 + t)` for `c0 + t < width`, else 0: a scaled row block of `op(B)`.
/// @ingroup api_device_lowlevel
/// @tparam Ar the arithmetic of the products.
/// @param alpha the scale.
/// @param b the dense operand.
/// @param r the row.
/// @param c0 the block's first column.
/// @param width the column count.
/// @param[out] w the `NC` values.
template <class Ar, class B, class Scalar, int NC, class L>
BATCHLAS_DEVICE_INLINE void load_cols_scaled(const Thread&, Scalar alpha, const B& b, int r, int c0, int width, Frag<Scalar, NC, L>& w) {
    BATCHLAS_UNROLL_FULL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        w[t] = Arith<Scalar>::zero();
        if (c < width) w[t] = Ar::mul(alpha, b(r, c));
    }
}

// ---- spmm ----------------------------------------------------------------------------------
namespace routine {
/// @brief The routine tag of `spmm`, for `launch`, `with_tier` and `tiers_t`.
/// @ingroup api_device_sparse
/// @tparam TA `op(A)`.
/// @tparam TB `op(B)`.
template <Transpose TA, Transpose TB> struct spmm {
    /// The routine's name.
    static constexpr const char* name = "spmm";
};
/// @brief The routine tag of `iluk_apply`.
/// @ingroup api_device_sparse
struct iluk_apply {
    /// The routine's name.
    static constexpr const char* name = "iluk_apply";
};
/// @brief The routine tag of `iluk_factorize`.
/// @ingroup api_device_sparse
struct iluk_factorize {
    /// The routine's name.
    static constexpr const char* name = "iluk_factorize";
};
}  // namespace routine

/// @brief `spmm`'s column block: 64 bytes of accumulators per item, so the block shrinks as the scalar widens.
/// @ingroup api_device_sparse
/// @tparam Scalar the element type.
template <class Scalar> inline constexpr int spmm_block_v = 64 / static_cast<int>(sizeof(Scalar));

/// @brief `spmm`'s tier list: one tier, `tier::Blocked<spmm_block_v<Scalar>>`.
/// @ingroup api_device_sparse
template <Transpose TA, Transpose TB, class Scalar> struct tiers<routine::spmm<TA, TB>, Scalar> {
    /// The tier list.
    using type = tier::list<tier::Blocked<spmm_block_v<Scalar>>>;
};
/// @brief `spmm`'s `tier::Blocked<NC>`: any n, no workspace.
///
/// `wg_size` and `sg_size` are the launch the tier is tested at; the body runs on any executor.
/// @ingroup api_device_sparse
template <Transpose TA, Transpose TB, int NC, class Scalar> struct tier_info<routine::spmm<TA, TB>, tier::Blocked<NC>, Scalar> {
    /// @{
    /// The tested work-group size and sub-group size.
    static constexpr int wg_size = 128, sg_size = 32;
    /// @}
    /// The executor `launch` builds.
    using executor = WorkGroup<wg_size, sg_size>;
    /// Batch items per work-group.
    static constexpr int per_group = 1;
    /// @brief Every `n >= 0` is admitted.
    /// @param n the extent.
    static constexpr bool admits(int n) { return n >= 0; }
    /// @brief No workspace.
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};

namespace detail {
// One gather unit (A NoTrans): row r of A times op(B)'s column block c0 .. c0 + NC into C(r, :).
// Arguments in the order the body reads them: a launcher's capture layout follows it.
template <class Ar, int NC, class Ops = arith::Fma, class Row, class OB, class Scalar, class CM>
BATCHLAS_DEVICE_INLINE void gather_unit(bool alpha_zero, const Row& a, const OB& b, int c0, int width, Scalar alpha, MaybeZero<Scalar> beta,
                                     const CM& C, int r) {
    Frag<Scalar, NC> acc;
    fill(acc, Scalar{});
    if (!alpha_zero) usmm_row<Ar>(Thread{}, a, b, c0, width, acc);
    store_cols<Ops>(Thread{}, alpha, acc, beta, C, r, c0, width);
}

// Address of the scatter's target C(r, c): a raw column-major pointer (the host route), or a
// matrix operand whose C(r, c) is an lvalue.
template <class Scalar> struct ColPtr {
    Scalar* p;
    std::int64_t ld;
    BATCHLAS_DEVICE_INLINE Scalar* at(int r, int c) const { return p + static_cast<std::int64_t>(c) * ld + r; }
};
template <class CM> struct RefPtr {
    CM m;
    BATCHLAS_DEVICE_INLINE auto* at(int r, int c) const { return &m(r, c); }
};

// One scatter row (A (Conj)Trans): stored row i of A, entries [rs, re), pushes
// conj?(A(i, j)) alpha op(B)(i, :) into row j of C by atomics; j outside [0, rows_out) skipped.
// The row is read in the body, not through a SpRow (sass_parity strict: an int64-based operand
// loses GEP inbounds under LICM, +4..5% instructions).
template <class Ar, int NCS, class Scalar, class OB, class CP>
BATCHLAS_DEVICE_INLINE void scatter_row(int width, Scalar alpha, const OB& b, int i, int rs, int re, const Scalar* a_val, std::int64_t vb, bool conj_a,
                                     const int* a_ci, int rows_out, const CP& C) {
    for (int c0 = 0; c0 < width; c0 += NCS) {
        Frag<Scalar, NCS> ab;
        load_cols_scaled<Ar>(Thread{}, alpha, b, i, c0, width, ab);
        for (int p = rs; p < re; ++p) {
            Scalar av = a_val[vb + p];
            if constexpr (Arith<Scalar>::is_complex) {
                if (conj_a) av = Arith<Scalar>::conj(av);
            }
            const int j = a_ci[vb + p];
            if (static_cast<unsigned>(j) >= static_cast<unsigned>(rows_out)) continue;
            BATCHLAS_UNROLL_FULL
            for (int t = 0; t < NCS; ++t) {
                const int c = c0 + t;
                if (c < width) atomic_accumulate(C.at(j, c), Ar::mul(av, ab[t]));
            }
        }
    }
}
}  // namespace detail

/// @brief `C := alpha op(A) op(B) + beta C` for one batch item, `A` sparse (CSR) and `B`, `C` dense.
///
/// Runs on any executor size and any extents. `beta == 0` never reads `C`; `alpha == 0` never
/// reads `A` or `B`. With `NoTrans` `A`, each item owns its (row, column block) units of the host
/// route's gather, so the result is bit-identical to the host's. With `Trans` / `ConjTrans` `A`:
/// `C := beta C`, a barrier, then a scatter by relaxed atomics in global memory, whose order is
/// unspecified (results may differ in the last bits from run to run). COLLECTIVE: the inputs are
/// visible on entry, and `C` is visible to the executor on return.
/// @ingroup api_device_sparse
/// @tparam TA `op(A)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam TB `op(B)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam Tier `tier::Blocked<NC>`, `NC` the column block of one item's accumulators
///   (`spmm_block_v<Scalar>` is the listed tier).
/// @param ex the executor.
/// @param alpha the scalar `alpha`, of `A`'s value type.
/// @param A the sparse operand, a `CsrItem` (`csr_item(view, b)`); `op(A)` is m x k.
/// @param B the dense operand; `op(B)` is k x n.
/// @param beta the scalar `beta`.
/// @param[in,out] C the m x n dense output with `cols()`; `C(r, c)` assignable, and an lvalue in
///   global memory for a transposed `A`.
/// @param ws the workspace; none is needed (default empty).
/// @retval 0 always.
template <Transpose TA, Transpose TB, class Tier, class Ex, class AM, class BM, class CM, class Scalar>
BATCHLAS_DEVICE_HELPER int spmm(const Ex& ex, Scalar alpha, const AM& A, const BM& B, Scalar beta, const CM& C, const Workspace& ws = {}) {
    static_assert(std::is_same_v<typename AM::value_type, Scalar>, "spmm: alpha, beta and A share one scalar type");
    using Ti = resolve_tier_t<routine::spmm<TA, TB>, Scalar, Tier>;
    static_assert(requires { Ti::nb; }, "spmm: the tier is tier::Blocked<NC>");
    constexpr int NC = Ti::nb;
    using Ar = arith_t<arith::Fma, Scalar>;
    const bool alpha_zero = Arith<Scalar>::is_zero(alpha), beta_zero = Arith<Scalar>::is_zero(beta);
    const auto opB = op_view<TB>(B);
    const int width = C.cols();
    if constexpr (TA == Transpose::NoTrans) {
        const int m = A.rows();
        for_each_owned(ex, m * ((width + NC - 1) / NC), [&](int u) BATCHLAS_DEVICE_BODY {
            const int cb = u / m;
            const int r = u - cb * m;
            detail::gather_unit<Ar, NC>(alpha_zero, A.row(r), opB, cb * NC, width, alpha, MaybeZero<Scalar>{beta, beta_zero}, C, r);
        });
    } else {
        const int rows_out = A.cols();
        for_each_owned(ex, rows_out * width, [&](int e) BATCHLAS_DEVICE_BODY {
            const int c = e / rows_out;
            const int r = e - c * rows_out;
            C(r, c) = beta_zero ? Scalar{} : Ar::mul(beta, static_cast<Scalar>(C(r, c)));
        });
        sync(ex, Mem::Global);
        if (!alpha_zero) {
            const detail::RefPtr<CM> dst{C};
            for_each_owned(ex, A.rows(), [&](int i) BATCHLAS_DEVICE_BODY {
                detail::scatter_row<Ar, 4>(width, alpha, opB, i, A.ro[A.ro_base + i], A.ro[A.ro_base + i + 1], A.val, A.nz_base,
                                                TA == Transpose::ConjTrans, A.col, rows_out, dst);
            });
        }
    }
    sync(ex, Mem::Global);
    return 0;
}

// ---- iluk_apply ----------------------------------------------------------------------------
/// @brief One batch item's ILU(k) factor, the operand of `iluk_apply`.
///
/// The layout `iluk_factorize` and the host's ILU(k) (`ILUKView`) store: `L` (strict lower, unit
/// diagonal) and `U` (upper, with its diagonal) in one CSR matrix with sorted rows, each row's
/// absolute diagonal position `diag[diag_base + i]`, and the batch-shared level schedules of the
/// two triangular solves. Get one from `IlukBatch::item` or `IlukPlan::factor`.
/// @ingroup api_device_sparse
/// @tparam Scalar the element type.
// Fields in the order the solve reads them (a launcher's capture layout follows it).
template <class Scalar> struct IlukFactor {
    /// The factor's values; `base` is the item's offset.
    BasedVec<const Scalar> vals;  // base = b * matrix_stride
    /// The factor's column indices, same offset.
    BasedVec<const int> cols;
    /// The number of levels of the `L` solve.
    int l_levels;
    /// The `L` levels: level `lv` is rows `l_rows[l_level_ptr[lv]]` to `l_rows[l_level_ptr[lv + 1] - 1]`.
    const int* l_level_ptr;
    /// The rows of the `L` levels.
    const int* l_rows;
    /// The row offsets of the batch.
    const int* ro;
    /// The item's offset into `ro`.
    int ro_base;  // b * offset_stride
    /// The number of levels of the `U` solve.
    int u_levels;
    /// The `U` levels, as `l_level_ptr`.
    const int* u_level_ptr;
    /// The rows of the `U` levels.
    const int* u_rows;
    /// Each row's absolute diagonal position in `vals`.
    const int* diag;
    /// The item's offset into `diag`.
    int diag_base;  // b * n
    /// The diagonal shift the solve applies to an unusable pivot.
    Scalar diag_shift;
};

/// @brief A batch of ILU(k) factors: trivially copyable, built on the host and captured by value; `item(b)` is item `b`'s factor.
///
/// Build it from a host factor with `iluk_batch(M.view())`.
/// @ingroup api_device_sparse
/// @tparam Scalar the element type.
template <class Scalar> struct IlukBatch {
    /// The values of every item.
    const Scalar* val;
    /// The column indices of every item.
    const int* col;
    /// The row offsets of every item.
    const int* ro;
    /// The absolute diagonal positions, `n` per item.
    const int* diag;
    /// The `L` levels, shared by the batch.
    const int* l_level_ptr;
    /// The rows of the `L` levels.
    const int* l_rows;
    /// The `U` levels, shared by the batch.
    const int* u_level_ptr;
    /// The rows of the `U` levels.
    const int* u_rows;
    /// @{
    /// The value stride and row-offset stride between items, the order, and the two level counts.
    int matrix_stride, offset_stride, n, l_levels, u_levels;
    /// @}
    /// The diagonal shift.
    Scalar diag_shift;
    /// @brief Batch item `b`'s factor.
    /// @param b the batch index.
    BATCHLAS_DEVICE_INLINE IlukFactor<Scalar> item(int b) const {
        return {{val, b * matrix_stride}, {col, b * matrix_stride}, l_levels, l_level_ptr, l_rows, ro, b * offset_stride,
                u_levels, u_level_ptr, u_rows, diag, b * n, diag_shift};
    }
};
/// @brief The `IlukBatch` of a host ILU(k) factor's view (host code).
/// @ingroup api_device_sparse
/// @param M an `ILUKView` (`ILUKPreconditioner::view()`); its storage must outlive the kernel.
/// @return the batch, to capture by value.
template <class V> auto iluk_batch(const V& M) {
    using T = std::remove_cv_t<decltype(M.diagonal_shift)>;
    const auto kv = M.lu.kernel_view();
    return IlukBatch<T>{kv.data_,          kv.col_indices_,      kv.row_offsets_,   M.diag_positions.data(),
                        M.l_level_ptr.data(), M.l_rows.data(),    M.u_level_ptr.data(), M.u_rows.data(),
                        kv.matrix_stride_, kv.offset_stride_, M.n, M.l_levels, M.u_levels, M.diagonal_shift};
}

/// @brief `iluk_apply`'s tier list: one tier, `tier::Blocked<1>`.
/// @ingroup api_device_sparse
template <class Scalar> struct tiers<routine::iluk_apply, Scalar> {
    /// The tier list.
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief `iluk_apply`'s `tier::Blocked<1>`: any n, no workspace, tested at a `WorkGroup<128, 32>`.
/// @ingroup api_device_sparse
template <class Scalar> struct tier_info<routine::iluk_apply, tier::Blocked<1>, Scalar> {
    /// @{
    /// The tested work-group size and sub-group size.
    static constexpr int wg_size = 128, sg_size = 32;
    /// @}
    /// The executor `launch` builds.
    using executor = WorkGroup<wg_size, sg_size>;
    /// Batch items per work-group.
    static constexpr int per_group = 1;
    /// @brief Every `n >= 0` is admitted.
    /// @param n the order.
    static constexpr bool admits(int n) { return n >= 0; }
    /// @brief No workspace.
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};

namespace detail {
// |x| for the drop and pivot tests of the factorization and the solve: std::abs for a real Scalar,
// hypot of the parts read through the reference for complex (not std::abs: a libm cabs call the
// native_cpu device link cannot resolve). sass_parity strict: core modulus's sycl::fabs folds
// the real comparisons differently, and its by-value parts swap complex registers.
template <class Scalar> inline real_t<Scalar> pivot_modulus(const Scalar& x) {
    if constexpr (Arith<Scalar>::is_complex) return sycl::hypot(x.real(), x.imag());
    else return static_cast<real_t<Scalar>>(std::abs(x));
}

// U's diagonal as the solve uses it (host and device): the stored pivot when it beats the shift,
// else pivot + shift when that does, else the shift; with no shift a zero pivot sets *status
// (when given) and returns 1. The factorization established usability, so the solve passes null.
template <class Scalar> inline Scalar shifted_pivot(const Scalar& pivot, const Scalar& diagonal_shift,
                                                    std::int32_t* status) {
    using R = real_t<Scalar>;
    const R shift_mag = pivot_modulus(diagonal_shift);
    const R pivot_mag = pivot_modulus(pivot);
    if (pivot_mag > shift_mag) return pivot;
    if (shift_mag > R(0)) {
        const Scalar shifted = pivot + diagonal_shift;
        if (pivot_modulus(shifted) > shift_mag) return shifted;
        return diagonal_shift;
    }
    if (status != nullptr) *status = 1;
    return Scalar(1);
}
}  // namespace detail

/// @brief `y := U^{-1} L^{-1} rhs` for one column of one batch item with its ILU(k) factor.
///
/// Level by level, with a barrier after each level of each solve, on a work-group executor. In
/// the `U` solve an unusable pivot is replaced by the factor's shifted pivot. Native arithmetic
/// (Annex-G complex), as the host route. COLLECTIVE: `y` is visible to the work-group on return.
/// @ingroup api_device_sparse
/// @tparam Tier `tier::Blocked<1>` (or a tier list naming it).
/// @param ex the work-group executor.
/// @param M the item's factor (`IlukBatch::item(b)` or `IlukPlan::factor`).
/// @param rhs the right-hand side, a vector `rhs(i)`.
/// @param[out] y the solution, a vector `y(i)` distinct from `rhs`.
/// @param ws the workspace; none is needed (default empty).
/// @retval 0 always.
template <class Tier, class Ex, class Scalar, class RV, class YV>
BATCHLAS_DEVICE_HELPER int iluk_apply(const Ex& ex, const IlukFactor<Scalar>& M, const RV& rhs, const YV& y, const Workspace& ws = {}) {
    static_assert(std::is_same_v<resolve_tier_t<routine::iluk_apply, Scalar, Tier>, tier::Blocked<1>>, "iluk_apply: the tier is tier::Blocked<1>");
    using Ar = arith_t<arith::Native, Scalar>;
    for (int lv = 0; lv < M.l_levels; ++lv) {
        const int begin = M.l_level_ptr[lv];
        for_each_owned(ex, M.l_level_ptr[lv + 1] - begin, [&](int k) {
            const int i = M.l_rows[begin + k];
            const Scalar sum = rhs(i);
            const auto row = sp_row(M.vals, M.cols, M.ro[M.ro_base + i], M.ro[M.ro_base + i + 1]);
            y(i) = usdot_sub_below<Ar>(Thread{}, sum, row, y, i);
        });
        sync(ex, Mem::Global);
    }
    // The U row residual is a hand loop in this body (sass_parity strict: in a for_each_owned
    // lambda its exit block moves after the loop, the latch flips and ptxas unrolls it 4x).
    for (int lv = 0; lv < M.u_levels; ++lv) {
        const int begin = M.u_level_ptr[lv];
        for (int k = ex.lid(), nk = M.u_level_ptr[lv + 1] - begin; k < nk; k += detail::size_of(ex)) {
            const int i = M.u_rows[begin + k];
            const Scalar sum = y(i);
            const int diag_abs = M.diag[M.diag_base + i];
            const int re = M.ro[M.ro_base + i + 1];
            const Scalar diag = detail::shifted_pivot(M.vals.p[diag_abs], M.diag_shift, nullptr);
            const auto row = sp_row(M.vals, M.cols, diag_abs - M.vals.base + 1, re);
            Scalar acc = sum;
            for (int p = row.begin; p < row.end; ++p) Ar::msub(acc, row.val(p), y(row.idx(p)));
            y(i) = Ar::div(acc, diag);
        }
        sync(ex, Mem::Global);
    }
    return 0;
}

// ---- iluk_factorize ------------------------------------------------------------------------
/// @brief The numeric rules of an ILU(k) factorization: the host's `ILUKParams` without `levels_of_fill`, which the plan fixes.
/// @ingroup api_device_sparse
/// @tparam Scalar the element type.
template <class Scalar> struct IlukOptions {
    /// The shift added to (or replacing) an unusable pivot; 0 for none.
    Scalar diagonal_shift;
    /// An off-diagonal entry at or below `drop_tolerance * max(row scale, 1)` is dropped.
    real_t<Scalar> drop_tolerance;
    /// Each row keeps its largest `ceil(fill_factor * row_nnz) - 1` off-diagonal entries.
    real_t<Scalar> fill_factor;
    /// A pivot below `diag_pivot_threshold * max(row scale, 1)` is unusable.
    real_t<Scalar> diag_pivot_threshold;
    /// `true`: modified ILU, dropped entries are added to the diagonal.
    bool modified_ilu;
};
/// @brief The `IlukOptions` of a host `ILUKParams`-like object (host code).
/// @ingroup api_device_sparse
/// @param p an object with the fields `diagonal_shift`, `drop_tolerance`, `fill_factor`,
///   `diag_pivot_threshold` and `modified_ilu`.
template <class P> auto iluk_options(const P& p) {
    using T = std::remove_cv_t<decltype(p.diagonal_shift)>;
    return IlukOptions<T>{p.diagonal_shift, p.drop_tolerance, p.fill_factor, p.diag_pivot_threshold, p.modified_ilu};
}

/// @brief The batch-shared symbolic plan of ILU(k) on one sorted CSR pattern with every diagonal stored.
///
/// Plain device-visible index arrays, captured by value. The library's host symbolic phase builds
/// it for its ILU(k) route; that builder is not an installed header, so a caller outside the
/// library fills the arrays to this layout itself. Slots `0` to `nnz - 1` are item-local positions
/// in the factor's pattern (`ro`, `col`, `diag`: `L` strict lower and `U` upper with its diagonal,
/// rows sorted). Row `i` runs the steps `[step_ptr[i], step_ptr[i + 1])`: step `e` divides slot
/// `step_target[e]` by the diagonal slot `step_pivot[e]`, giving `l`, then for each `u` in
/// `[step_upd_ptr[e], step_upd_ptr[e + 1])` updates slot `upd_dst[u] -= l * slot upd_src[u]`.
/// @ingroup api_device_sparse
struct IlukPlan {
    /// @{
    /// The order, `A`'s entry count, the factor's slot count, and the elimination and `U`-solve level counts.
    int n, a_nnz, nnz, levels, u_levels;
    /// @}
    /// The factor's row offsets (`n + 1`).
    const int* ro;
    /// The factor's column indices (`nnz`).
    const int* col;
    /// Each row's diagonal slot (`n`).
    const int* diag;
    /// `A`'s entry count per row, the fill quota's base (`n`).
    const int* row_nnz;
    /// The slot `A`'s entry `p` lands in, or -1 (`a_nnz`).
    const int* a_to_slot;
    /// Each row's first step (`n + 1`).
    const int* step_ptr;
    /// Each step's target slot.
    const int* step_target;
    /// Each step's diagonal slot.
    const int* step_pivot;
    /// Each step's first update (steps + 1).
    const int* step_upd_ptr;
    /// Each update's source slot.
    const int* upd_src;
    /// Each update's destination slot.
    const int* upd_dst;
    /// The elimination levels, also the `L`-solve levels (`levels + 1`).
    const int* level_ptr;
    /// The rows of the elimination levels.
    const int* level_rows;
    /// The `U`-solve levels (`u_levels + 1`).
    const int* u_level_ptr;
    /// The rows of the `U`-solve levels.
    const int* u_rows;
    /// @brief The factor `iluk_factorize` left at `lu` (one item's `nnz` slots), as `iluk_apply`'s operand.
    /// @param lu the item's factor slots.
    /// @param diag_shift the diagonal shift the solve applies.
    template <class Scalar> BATCHLAS_DEVICE_INLINE IlukFactor<Scalar> factor(const Scalar* lu,
                                                                             const Scalar& diag_shift) const {
        return {{lu, 0}, {col, 0}, levels, level_ptr, level_rows, ro, 0, u_levels, u_level_ptr, u_rows, diag, 0, diag_shift};
    }
};

/// @brief `iluk_factorize`'s tier list: one tier, `tier::Blocked<1>`.
/// @ingroup api_device_sparse
template <class Scalar> struct tiers<routine::iluk_factorize, Scalar> {
    /// The tier list.
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief `iluk_factorize`'s `tier::Blocked<1>`: any n, no workspace, tested at a `WorkGroup<128, 32>`.
/// @ingroup api_device_sparse
template <class Scalar> struct tier_info<routine::iluk_factorize, tier::Blocked<1>, Scalar> {
    /// @{
    /// The tested work-group size and sub-group size.
    static constexpr int wg_size = 128, sg_size = 32;
    /// @}
    /// The executor `launch` builds.
    using executor = WorkGroup<wg_size, sg_size>;
    /// Batch items per work-group.
    static constexpr int per_group = 1;
    /// @brief Every `n >= 0` is admitted.
    /// @param n the order.
    static constexpr bool admits(int n) { return n >= 0; }
    /// @brief No workspace.
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};

namespace detail {
// A usable pivot (kernel-safe: an unusable one sets *status and returns 1, nothing throws): the
// pivot when it clears max(|shift|, diag_pivot_threshold * max(scale, 1)) and is nonzero, else
// pivot + shift, else the shift, else the threshold itself.
template <class Scalar>
inline Scalar stabilize_pivot(const Scalar& pivot, real_t<Scalar> scale, const Scalar& diagonal_shift, real_t<Scalar> diag_pivot_threshold, std::int32_t* status) {
    using R = real_t<Scalar>;
    const auto shift_mag = pivot_modulus(diagonal_shift);
    const auto pivot_mag = pivot_modulus(pivot);
    const auto one = R(1);
    const auto scaled = diag_pivot_threshold * (scale > one ? scale : one);
    const auto threshold = shift_mag > scaled ? shift_mag : scaled;

    if (pivot_mag >= threshold && pivot_mag > R(0)) return pivot;
    if (shift_mag > R(0)) {
        const Scalar shifted = pivot + diagonal_shift;
        if (pivot_modulus(shifted) >= threshold && pivot_modulus(shifted) > R(0)) return shifted;
        if (shift_mag >= threshold) return diagonal_shift;
    }
    if (threshold > R(0)) return Scalar(threshold);
    if (status != nullptr) *status = 1;
    return Scalar(1);
}

// Row i of the factorization in place on the slots at(s) (lvalues; rows of earlier levels done):
// IKJ elimination by the plan's steps, the relative drop (MILU folds dropped entries into the
// diagonal), the fill quota (largest ceil(fill_factor * row_nnz[i]) - 1 off-diagonals survive;
// on a magnitude tie the later slot drops) and the stabilised pivot. Native arithmetic, hand
// loops over absolute slots (sass_parity strict: a for_each_owned loop from 0 moved LLVM's runtime
// unroll remainder, LDG 38->66, and arith::Native msub as a call costs complex<float> +4 REG).
// Arguments in the order the body reads them: a launcher's capture layout follows it.
template <class Scalar, class W>
BATCHLAS_DEVICE_INLINE void eliminate_row(const W& at, int i, const int* step_ptr, const int* step_target, const int* step_pivot,
                                       const int* step_upd_ptr, const int* upd_dst, const int* upd_src, const int* sym_ro,
                                       const int* diag_abs, real_t<Scalar> drop_tolerance, bool modified_ilu, real_t<Scalar> fill_factor,
                                       const int* orig_nnz, const Scalar& diagonal_shift, real_t<Scalar> diag_pivot_threshold, std::int32_t* st) {
    using R = real_t<Scalar>;
    for (int e = step_ptr[i]; e < step_ptr[i + 1]; ++e) {
        const Scalar lij = at(step_target[e]) / at(step_pivot[e]);
        at(step_target[e]) = lij;
        for (int u = step_upd_ptr[e]; u < step_upd_ptr[e + 1]; ++u) {
            at(upd_dst[u]) -= lij * at(upd_src[u]);
        }
    }

    const int rs = sym_ro[i];
    const int re = sym_ro[i + 1];
    const int dg = diag_abs[i];
    R scale = R(0);
    for (int p = rs; p < re; ++p) {
        const auto m = modulus(at(p));
        if (m > scale) scale = m;
    }
    const auto one = R(1);
    const auto threshold = drop_tolerance * (scale > one ? scale : one);

    int kept = 0;
    for (int p = rs; p < re; ++p) {
        if (p == dg) continue;
        if (pivot_modulus(at(p)) <= threshold) {
            if (modified_ilu) at(dg) += at(p);
            at(p) = Scalar(0);
        } else {
            ++kept;
        }
    }

    // Dropping the smallest survivor until the quota holds is sorting and truncating without
    // per-row scratch; at the default fill_factor the loop does not run.
    const int quota_raw = static_cast<int>(sycl::ceil(fill_factor * static_cast<R>(orig_nnz[i]))) - 1;
    const int quota = quota_raw > 0 ? quota_raw : 0;
    while (kept > quota) {
        int victim = -1;
        R smallest = R(0);
        for (int p = rs; p < re; ++p) {
            if (p == dg) continue;
            const auto m = pivot_modulus(at(p));
            if (m == R(0)) continue;
            if (victim < 0 || m < smallest || (m == smallest && p > victim)) {
                victim = p;
                smallest = m;
            }
        }
        if (victim < 0) break;
        if (modified_ilu) at(dg) += at(victim);
        at(victim) = Scalar(0);
        --kept;
    }

    R final_scale = R(0);
    for (int p = rs; p < re; ++p) {
        const auto m = modulus(at(p));
        if (m > final_scale) final_scale = m;
    }
    at(dg) = stabilize_pivot(at(dg), final_scale, diagonal_shift, diag_pivot_threshold, st);
}
}  // namespace detail

/// @brief `M :=` the ILU(k) factor of one batch item's `A` in the plan's pattern, on any n.
///
/// Zeroes `M`, scatters `A` into it, then eliminates one level per barrier, the rows of a level
/// split over the items. Every slot of the pattern is written and a dropped entry is +0, so `M` is
/// the host route's factor expanded to the symbolic pattern, bit for bit (the host keeps the union
/// of the items' surviving entries). Elimination continues past an unusable pivot with pivot 1.
/// COLLECTIVE over a work-group executor: `M` is visible to it on return.
/// @ingroup api_device_sparse
/// @tparam Tier `tier::Blocked<1>` (or a tier list naming it).
/// @param ex the work-group executor.
/// @param A the item's values in `A`'s own CSR order, `A(p)` for `p < plan.a_nnz`.
/// @param plan the batch-shared symbolic plan; the pattern is the plan's.
/// @param[out] M the `plan.nnz` factor slots `M(s)`, lvalues distinct from `A` (global memory, or
///   local memory of `nnz` elements).
/// @param opts the numeric rules.
/// @param ws the workspace; none is needed (default empty).
/// @retval 0 every pivot was usable.
/// @retval k the smallest 1-based row whose pivot was unusable (no shift and a zero pivot; the host
///   route throws `convergence_error` there), the same on every item.
template <class Tier, class Ex, class AV, class MV, class Scalar>
BATCHLAS_DEVICE_HELPER int iluk_factorize(const Ex& ex, const AV& A, const IlukPlan& plan, const MV& M, const IlukOptions<Scalar>& opts,
                                       const Workspace& ws = {}) {
    static_assert(std::is_same_v<resolve_tier_t<routine::iluk_factorize, Scalar, Tier>, tier::Blocked<1>>,
                  "iluk_factorize: the tier is tier::Blocked<1>");
    for_each_owned(ex, plan.nnz, [&](int s) { M(s) = Scalar(0); });
    sync(ex, Mem::Global);
    for_each_owned(ex, plan.a_nnz, [&](int p) {
        const int s = plan.a_to_slot[p];
        if (s >= 0) M(s) = static_cast<Scalar>(A(p));
    });
    sync(ex, Mem::Global);
    int bad = std::numeric_limits<int>::max();
    for (int lv = 0; lv < plan.levels; ++lv) {
        const int begin = plan.level_ptr[lv];
        for_each_owned(ex, plan.level_ptr[lv + 1] - begin, [&](int t) {
            const int i = plan.level_rows[begin + t];
            std::int32_t st = 0;
            detail::eliminate_row(M, i, plan.step_ptr, plan.step_target, plan.step_pivot, plan.step_upd_ptr, plan.upd_dst,
                                       plan.upd_src, plan.ro, plan.diag, opts.drop_tolerance, opts.modified_ilu, opts.fill_factor,
                                       plan.row_nnz, opts.diagonal_shift, opts.diag_pivot_threshold, &st);
            if (st != 0 && i + 1 < bad) bad = i + 1;
        });
        sync(ex, Mem::Global);
    }
    bad = ex.group_reduce(bad, sycl::minimum<int>());
    return bad == std::numeric_limits<int>::max() ? 0 : bad;
}

}  // namespace batchlas::device
