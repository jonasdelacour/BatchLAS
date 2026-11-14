// SPDX-License-Identifier: MIT
#pragma once
// Device LA sparse operands, CSR row products (Sparse BLAS naming: us*) and the public in-kernel
// sparse ops: spmm, iluk_factorize and iluk_apply (v22 design C, decision F-D5). Each op runs one
// batch item on one executor for any extent; the host routes launch the same bodies (detail
// units, the detail row elimination, the iluk_apply solve). Element expressions match the
// hand kernels they replaced (int64 offsets where those used them), so the host kernels keep
// their SASS.
//   spmm<TA, TB, Tier>(ex, A, B, C, alpha, beta[, ws])     C := alpha op(A) op(B) + beta C
//   iluk_factorize<Tier>(ex, A, plan, M, opts[, ws])       M := ILU(k) of A in plan's pattern
//   iluk_apply<Tier>(ex, M, rhs, out[, ws])                out := (L U)^{-1} rhs, one column
// Tiers: tier::Blocked<NC> for spmm (NC = the column block of one item's accumulators) and
// tier::Blocked<1> for the two ILU ops (level-scheduled, a barrier per level); all admit any n,
// read and write their operands where the caller put them (global, or local memory the caller
// sized) and need no workspace (*_buffer_size == {}).

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

// One scalar access, or a complex as one 2-wide vector access (needs 2 * sizeof(real) alignment).
template <bool Pair> struct PairAccess {
    template <class T> BATCHLAS_DEVICE_INLINE static std::remove_cv_t<T> load(T* p) {
        using U = std::remove_cv_t<T>;
        using A = Arith<U>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            const sycl::vec<R, 2> v = *reinterpret_cast<const sycl::vec<R, 2>*>(p);
            return A::make(v[0], v[1]);
        } else {
            return *p;
        }
    }
    template <class T> BATCHLAS_DEVICE_INLINE static void store(T* p, T x) {
        using A = Arith<T>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            *reinterpret_cast<sycl::vec<R, 2>*>(p) = sycl::vec<R, 2>(A::re(x), A::im(x));
        } else {
            *p = x;
        }
    }
};

// Relaxed device-scope atomic *p += v. A complex is two independent real atomics, NOT an atomic
// complex add: correct only while nothing reads *p between the first and the last update.
template <class T> BATCHLAS_DEVICE_INLINE void atomic_accumulate(T* p, T v) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        atomic_accumulate(reinterpret_cast<R*>(p), A::re(v));
        atomic_accumulate(reinterpret_cast<R*>(p) + 1, A::im(v));
    } else {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*p);
        ref.fetch_add(v);
    }
}

// ---- operands ------------------------------------------------------------------------------
// Element i at p[base + i], the sum in Off (int64_t where base = b * stride passes 2^31).
template <class T, class Off = int> struct BasedVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    Off base;
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const { return p[base + i]; }
};

// Read-only element i = PairAccess<Pair>::load(p + i).
template <class T, bool Pair = false> struct LoadVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    BATCHLAS_DEVICE_INLINE value_type operator()(int i) const { return PairAccess<Pair>::load(p + i); }
};

// Batch-interleaved (slot-major) storage: slot s of item `off` at p[size_t(s) * stride + off].
// stride and off stay int and widen at each use (as size_t they cost ILU eliminate +3..5 REG).
template <class T> struct InterleavedVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int stride;
    int off;
    BATCHLAS_DEVICE_INLINE T& operator()(int s) const { return p[static_cast<std::size_t>(s) * stride + off]; }
};

// One CSR row segment: entries p in [begin, end) are (val(p), idx(p)), positions absolute.
template <class VV, class IV> struct SpRow {
    using value_type = typename VV::value_type;
    VV vals;
    IV cols;
    int begin;
    int end;
    BATCHLAS_DEVICE_INLINE value_type val(int p) const { return vals(p); }
    BATCHLAS_DEVICE_INLINE int idx(int p) const { return cols(p); }
};
template <class VV, class IV> BATCHLAS_DEVICE_INLINE SpRow<VV, IV> sp_row(VV vals, IV cols, int begin, int end) {
    return {vals, cols, begin, end};
}

// Batch item b of a CSR batch (KernelMatrixView<T, CSR>): values and column indices at
// nz_base = b * matrix_stride, item-local row offsets at ro_base = b * offset_stride; m x n as
// stored. Row r is [ro(r), ro(r + 1)); nothing at or past ro(m) is read.
template <class T, class Off = std::int64_t> struct CsrItem {
    using value_type = std::remove_cv_t<T>;
    T* val;
    const int* col;
    const int* ro;
    Off nz_base;
    Off ro_base;
    int m, n;
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
    template <bool Pair = false> BATCHLAS_DEVICE_INLINE auto row(int r) const {
        return sp_row(LoadVec<T, Pair>{val + nz_base}, BasedVec<const int, Off>{col, nz_base}, ro[ro_base + r], ro[ro_base + r + 1]);
    }
};
template <class T> BATCHLAS_DEVICE_INLINE CsrItem<const T> csr_item(const KernelMatrixView<T, MatrixFormat::CSR>& a, std::int64_t b) {
    return {a.data_, a.col_indices_, a.row_offsets_, b * a.matrix_stride_, b * a.offset_stride_, a.rows_, a.cols_};
}

// op(B)(r, c) of a column-major B (int64 ld) whose transposition is a kernel argument: NoTrans
// p[c * ld + r], else p[r * ld + c], conjugated when conj (one instantiation, as the hand kernel).
template <class T, bool Pair = false> struct RtOpMat {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::int64_t ld;
    bool notrans;
    bool conj;
    BATCHLAS_DEVICE_INLINE value_type operator()(int r, int c) const {
        value_type v = PairAccess<Pair>::load(notrans ? p + static_cast<std::int64_t>(c) * ld + r
                                                      : p + static_cast<std::int64_t>(r) * ld + c);
        if constexpr (Arith<value_type>::is_complex) {
            if (conj) v = Arith<value_type>::conj(v);
        }
        return v;
    }
};

// Column-major C (int64 ld): C(r, c) at p + (c * ld + r), an element reference that reads and
// writes through PairAccess, so core store_epi() can take it as its y.
template <class T, bool Pair = false> struct ColMat64 {
    using value_type = std::remove_cv_t<T>;
    struct Ref {
        T* q;
        BATCHLAS_DEVICE_INLINE operator value_type() const { return PairAccess<Pair>::load(q); }
        BATCHLAS_DEVICE_INLINE void operator=(value_type x) const { PairAccess<Pair>::store(q, x); }
    };
    T* p;
    std::int64_t ld;
    BATCHLAS_DEVICE_INLINE T* ptr(int r, int c) const { return p + (static_cast<std::int64_t>(c) * ld + r); }
    BATCHLAS_DEVICE_INLINE Ref operator()(int r, int c) const { return {ptr(r, c)}; }
};

// ---- CSR row products ----------------------------------------------------------------------
// acc - sum a.val(p) y(a.idx(p)) over the entries with idx < bound, left to right: the row is
// sorted, so the first idx >= bound ends it; nothing at or past it is read.
template <class Ar, class Row, class Y, class T>
BATCHLAS_DEVICE_INLINE T usdot_sub_below(const Thread&, T acc, const Row& a, const Y& y, int bound) {
    for (int p = a.begin; p < a.end; ++p) {
        const int j = a.idx(p);
        if (j >= bound) break;
        Ar::msub(acc, a.val(p), y(j));
    }
    return acc;
}

// usmm, one output row: acc[t] += sum_p a.val(p) b(a.idx(p), c0 + t) for c0 + t < width.
template <class Ar, class Row, class B, class T, int NC, class L>
BATCHLAS_DEVICE_INLINE void usmm_row(const Thread&, const Row& a, const B& b, int c0, int width, Frag<T, NC, L>& acc) {
    for (int p = a.begin; p < a.end; ++p) {
        const T av = a.val(p);
        const int j = a.idx(p);
        BATCHLAS_UNROLL_FULL
        for (int t = 0; t < NC; ++t) {
            const int c = c0 + t;
            if (c < width) Ar::mac(acc[t], av, b(j, c));
        }
    }
}

// C(r, c0 + t) := store_epi<epi::MacChain, Ops>(alpha, acc[t], beta, C(r, c0 + t)) for c0 + t < width;
// a set beta flag never reads C.
template <class Ops, class T, int NC, class L, class CM>
BATCHLAS_DEVICE_INLINE void store_cols(const Thread&, T alpha, const Frag<T, NC, L>& acc, MaybeZero<T> beta, const CM& C, int r,
                                    int c0, int width) {
    BATCHLAS_UNROLL_FULL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        if (c < width) store_epi<epi::MacChain, Ops>(alpha, acc[t], beta, C(r, c));
    }
}

// w[t] := alpha b(r, c0 + t) for c0 + t < width, else 0: a scaled row block of op(B).
template <class Ar, class B, class T, int NC, class L>
BATCHLAS_DEVICE_INLINE void load_cols_scaled(const Thread&, T alpha, const B& b, int r, int c0, int width, Frag<T, NC, L>& w) {
    BATCHLAS_UNROLL_FULL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        w[t] = Arith<T>::zero();
        if (c < width) w[t] = Ar::mul(alpha, b(r, c));
    }
}

// ---- spmm ----------------------------------------------------------------------------------
namespace routine {
template <Transpose TA, Transpose TB> struct spmm {
    static constexpr const char* name = "spmm";
};
struct iluk_apply {
    static constexpr const char* name = "iluk_apply";
};
struct iluk_factorize {
    static constexpr const char* name = "iluk_factorize";
};
}  // namespace routine

// 64 bytes of accumulators per item: the column block shrinks as the scalar widens.
template <class T> inline constexpr int spmm_block_v = 64 / static_cast<int>(sizeof(T));

template <Transpose TA, Transpose TB, class T> struct tiers<routine::spmm<TA, TB>, T> {
    using type = tier::list<tier::Blocked<spmm_block_v<T>>>;
};
// wg_size / sg_size: the launch the tier is tested at; the body runs on any executor.
template <Transpose TA, Transpose TB, int NC, class T> struct tier_info<routine::spmm<TA, TB>, tier::Blocked<NC>, T> {
    static constexpr int wg_size = 128, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <Transpose TA, Transpose TB, class Tier, class T> constexpr WorkspaceSize spmm_buffer_size(int n) {
    using R = routine::spmm<TA, TB>;
    return tier_info<R, resolve_tier_t<R, T, Tier>, T>::buffer_size(n);
}

namespace detail {
// One gather unit (A NoTrans): row r of A times op(B)'s column block c0 .. c0 + NC into C(r, :).
// Arguments in the order the body reads them: a launcher's capture layout follows it.
template <class Ar, int NC, class Ops = arith::Fma, class Row, class OB, class T, class CM>
BATCHLAS_DEVICE_INLINE void gather_unit(bool alpha_zero, const Row& a, const OB& b, int c0, int width, T alpha, MaybeZero<T> beta,
                                     const CM& C, int r) {
    Frag<T, NC> acc;
    fill(acc, T{});
    if (!alpha_zero) usmm_row<Ar>(Thread{}, a, b, c0, width, acc);
    store_cols<Ops>(Thread{}, alpha, acc, beta, C, r, c0, width);
}

// Address of the scatter's target C(r, c): a raw column-major pointer (the host route), or a
// matrix operand whose C(r, c) is an lvalue.
template <class T> struct ColPtr {
    T* p;
    std::int64_t ld;
    BATCHLAS_DEVICE_INLINE T* at(int r, int c) const { return p + static_cast<std::int64_t>(c) * ld + r; }
};
template <class CM> struct RefPtr {
    CM m;
    BATCHLAS_DEVICE_INLINE auto* at(int r, int c) const { return &m(r, c); }
};

// One scatter row (A (Conj)Trans): stored row i of A, entries [rs, re), pushes
// conj?(A(i, j)) alpha op(B)(i, :) into row j of C by atomics; j outside [0, rows_out) skipped.
// The row is read in the body, not through a SpRow (sass_parity strict: an int64-based operand
// loses GEP inbounds under LICM, +4..5% instructions).
template <class Ar, int NCS, class T, class OB, class CP>
BATCHLAS_DEVICE_INLINE void scatter_row(int width, T alpha, const OB& b, int i, int rs, int re, const T* a_val, std::int64_t vb, bool conj_a,
                                     const int* a_ci, int rows_out, const CP& C) {
    for (int c0 = 0; c0 < width; c0 += NCS) {
        Frag<T, NCS> ab;
        load_cols_scaled<Ar>(Thread{}, alpha, b, i, c0, width, ab);
        for (int p = rs; p < re; ++p) {
            T av = a_val[vb + p];
            if constexpr (Arith<T>::is_complex) {
                if (conj_a) av = Arith<T>::conj(av);
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

// C := alpha op(A) op(B) + beta C for one batch item on executor ex (any size, any extents).
//   A  CsrItem (csr_item(view, b)); op(A) is m x k.   B  matrix operand, op(B) k x n.
//   C  matrix operand with cols(); C(r, c) assignable (an lvalue for (Conj)Trans A).
// beta == 0 never reads C; alpha == 0 never reads A or B. NoTrans A: the host route's gather
// units, each item owning its (row, column block) units: bit-identical to the host. (Conj)Trans:
// C := beta C, a barrier, then the scatter by relaxed atomics, whose order is unspecified.
// Collective: inputs visible on entry; returns 0 after a barrier (C visible to the executor).
template <Transpose TA, Transpose TB, class Tier, class Ex, class AM, class BM, class CM, class T>
BATCHLAS_DEVICE_HELPER int spmm(const Ex& ex, const AM& A, const BM& B, const CM& C, T alpha, T beta, const Workspace& = {}) {
    static_assert(std::is_same_v<typename AM::value_type, T>, "spmm: alpha, beta and A share one scalar type");
    using Ti = resolve_tier_t<routine::spmm<TA, TB>, T, Tier>;
    static_assert(requires { Ti::nb; }, "spmm: the tier is tier::Blocked<NC>");
    constexpr int NC = Ti::nb;
    using Ar = arith_t<arith::Fma, T>;
    const bool alpha_zero = Arith<T>::is_zero(alpha), beta_zero = Arith<T>::is_zero(beta);
    const auto opB = op_view<TB>(B);
    const int width = C.cols();
    if constexpr (TA == Transpose::NoTrans) {
        const int m = A.rows();
        for_each_owned(ex, m * ((width + NC - 1) / NC), [&](int u) BATCHLAS_DEVICE_BODY {
            const int cb = u / m;
            const int r = u - cb * m;
            detail::gather_unit<Ar, NC>(alpha_zero, A.row(r), opB, cb * NC, width, alpha, MaybeZero<T>{beta, beta_zero}, C, r);
        });
    } else {
        const int rows_out = A.cols();
        for_each_owned(ex, rows_out * width, [&](int e) BATCHLAS_DEVICE_BODY {
            const int c = e / rows_out;
            const int r = e - c * rows_out;
            C(r, c) = beta_zero ? T{} : Ar::mul(beta, static_cast<T>(C(r, c)));
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
// One batch item's ILU(k) factor as iluk_factorize stores it (ILUKView): L (strict lower, unit
// diagonal) and U (upper with diagonal) in one CSR LU with sorted rows, each row's absolute
// diagonal position diag[diag_base + i], and the batch-shared level schedules of the two solves.
// Fields in the order the solve reads them (a launcher's capture layout follows it).
template <class T> struct IlukFactor {
    BasedVec<const T> vals;  // base = b * matrix_stride
    BasedVec<const int> cols;
    int l_levels;
    const int* l_level_ptr;
    const int* l_rows;
    const int* ro;
    int ro_base;  // b * offset_stride
    int u_levels;
    const int* u_level_ptr;
    const int* u_rows;
    const int* diag;
    int diag_base;  // b * n
    T diag_shift;
};

// The batch: trivially copyable, built on the host from an ILUKView (iluk_batch(M.view())) and
// captured by value; item(b) is batch item b's factor.
template <class T> struct IlukBatch {
    const T* val;
    const int* col;
    const int* ro;
    const int* diag;
    const int* l_level_ptr;
    const int* l_rows;
    const int* u_level_ptr;
    const int* u_rows;
    int matrix_stride, offset_stride, n, l_levels, u_levels;
    T diag_shift;
    BATCHLAS_DEVICE_INLINE IlukFactor<T> item(int b) const {
        return {{val, b * matrix_stride}, {col, b * matrix_stride}, l_levels, l_level_ptr, l_rows, ro, b * offset_stride,
                u_levels, u_level_ptr, u_rows, diag, b * n, diag_shift};
    }
};
template <class V> auto iluk_batch(const V& M) {
    using T = std::remove_cv_t<decltype(M.diagonal_shift)>;
    const auto kv = M.lu.kernel_view();
    return IlukBatch<T>{kv.data_,          kv.col_indices_,      kv.row_offsets_,   M.diag_positions.data(),
                        M.l_level_ptr.data(), M.l_rows.data(),    M.u_level_ptr.data(), M.u_rows.data(),
                        kv.matrix_stride_, kv.offset_stride_, M.n, M.l_levels, M.u_levels, M.diagonal_shift};
}

template <class T> struct tiers<routine::iluk_apply, T> {
    using type = tier::list<tier::Blocked<1>>;
};
template <class T> struct tier_info<routine::iluk_apply, tier::Blocked<1>, T> {
    static constexpr int wg_size = 128, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <class Tier, class T> constexpr WorkspaceSize iluk_apply_buffer_size(int n) {
    return tier_info<routine::iluk_apply, resolve_tier_t<routine::iluk_apply, T, Tier>, T>::buffer_size(n);
}

namespace detail {
// |x| for the drop and pivot tests of the factorization and the solve: std::abs for a real T,
// hypot of the parts read through the reference for complex (not std::abs: a libm cabs call the
// native_cpu device link cannot resolve). sass_parity strict: core modulus's sycl::fabs folds
// the real comparisons differently, and its by-value parts swap complex registers.
template <class T> inline real_t<T> pivot_modulus(const T& x) {
    if constexpr (Arith<T>::is_complex) return sycl::hypot(x.real(), x.imag());
    else return static_cast<real_t<T>>(std::abs(x));
}

// U's diagonal as the solve uses it (host and device): the stored pivot when it beats the shift,
// else pivot + shift when that does, else the shift; with no shift a zero pivot sets *status
// (when given) and returns 1. The factorization established usability, so the solve passes null.
template <class T> inline T shifted_pivot(const T& pivot, const T& diagonal_shift, std::int32_t* status) {
    using R = real_t<T>;
    const R shift_mag = pivot_modulus(diagonal_shift);
    const R pivot_mag = pivot_modulus(pivot);
    if (pivot_mag > shift_mag) return pivot;
    if (shift_mag > R(0)) {
        const T shifted = pivot + diagonal_shift;
        if (pivot_modulus(shifted) > shift_mag) return shifted;
        return diagonal_shift;
    }
    if (status != nullptr) *status = 1;
    return T(1);
}
}  // namespace detail

// out := U^{-1} L^{-1} rhs for one column of one batch item, level by level (a barrier between
// levels), on a work-group executor; rhs and out are vectors (v(i)), distinct. Collective; returns
// 0 after the last barrier. Native arithmetic (Annex-G complex), as the host route.
template <class Tier, class Ex, class T, class RV, class YV>
BATCHLAS_DEVICE_HELPER int iluk_apply(const Ex& ex, const IlukFactor<T>& M, const RV& rhs, const YV& y, const Workspace& = {}) {
    static_assert(std::is_same_v<resolve_tier_t<routine::iluk_apply, T, Tier>, tier::Blocked<1>>, "iluk_apply: the tier is tier::Blocked<1>");
    using Ar = arith_t<arith::Native, T>;
    for (int lv = 0; lv < M.l_levels; ++lv) {
        const int begin = M.l_level_ptr[lv];
        for_each_owned(ex, M.l_level_ptr[lv + 1] - begin, [&](int k) {
            const int i = M.l_rows[begin + k];
            const T sum = rhs(i);
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
            const T sum = y(i);
            const int diag_abs = M.diag[M.diag_base + i];
            const int re = M.ro[M.ro_base + i + 1];
            const T diag = detail::shifted_pivot(M.vals.p[diag_abs], M.diag_shift, nullptr);
            const auto row = sp_row(M.vals, M.cols, diag_abs - M.vals.base + 1, re);
            T acc = sum;
            for (int p = row.begin; p < row.end; ++p) Ar::msub(acc, row.val(p), y(row.idx(p)));
            y(i) = Ar::div(acc, diag);
        }
        sync(ex, Mem::Global);
    }
    return 0;
}

// ---- iluk_factorize ------------------------------------------------------------------------
// The numeric rules of an ILU(k) factorization: ILUKParams without levels_of_fill (the plan's).
template <class T> struct IlukOptions {
    T diagonal_shift;
    real_t<T> drop_tolerance;
    real_t<T> fill_factor;
    real_t<T> diag_pivot_threshold;
    bool modified_ilu;
};
template <class P> auto iluk_options(const P& p) {
    using T = std::remove_cv_t<decltype(p.diagonal_shift)>;
    return IlukOptions<T>{p.diagonal_shift, p.drop_tolerance, p.fill_factor, p.diag_pivot_threshold, p.modified_ilu};
}

// The batch-shared symbolic plan of ILU(k) on one sorted CSR pattern with every diagonal stored,
// as the host's symbolic phase builds it (the iluk host route; src/ops/iluk/iluk_plan.hh). Plain
// device-visible index arrays, captured by value. Slots 0 .. nnz are item-local positions in the
// factor's pattern (ro, col, diag: L strict lower and U upper with diagonal, rows sorted).
//   a_to_slot[p]  the slot A's entry p lands in (-1: none);  row_nnz[i]  A's row count (fill quota)
//   row i runs steps [step_ptr[i], step_ptr[i + 1]): step e divides slot step_target[e] by the
//   diagonal slot step_pivot[e], then updates slot upd_dst[u] -= l * slot upd_src[u] for u in
//   [step_upd_ptr[e], step_upd_ptr[e + 1])
//   level_ptr / level_rows  the elimination levels (also the L-solve levels); u_*  the U-solve levels
struct IlukPlan {
    int n, a_nnz, nnz, levels, u_levels;
    const int* ro;
    const int* col;
    const int* diag;
    const int* row_nnz;
    const int* a_to_slot;
    const int* step_ptr;
    const int* step_target;
    const int* step_pivot;
    const int* step_upd_ptr;
    const int* upd_src;
    const int* upd_dst;
    const int* level_ptr;
    const int* level_rows;
    const int* u_level_ptr;
    const int* u_rows;
    // The factor iluk_factorize left at lu (one item's nnz slots) as iluk_apply's operand.
    template <class T> BATCHLAS_DEVICE_INLINE IlukFactor<T> factor(const T* lu, const T& diag_shift) const {
        return {{lu, 0}, {col, 0}, levels, level_ptr, level_rows, ro, 0, u_levels, u_level_ptr, u_rows, diag, 0, diag_shift};
    }
};

template <class T> struct tiers<routine::iluk_factorize, T> {
    using type = tier::list<tier::Blocked<1>>;
};
template <class T> struct tier_info<routine::iluk_factorize, tier::Blocked<1>, T> {
    static constexpr int wg_size = 128, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
template <class Tier, class T> constexpr WorkspaceSize iluk_factorize_buffer_size(int n) {
    return tier_info<routine::iluk_factorize, resolve_tier_t<routine::iluk_factorize, T, Tier>, T>::buffer_size(n);
}

namespace detail {
// A usable pivot (kernel-safe: an unusable one sets *status and returns 1, nothing throws): the
// pivot when it clears max(|shift|, diag_pivot_threshold * max(scale, 1)) and is nonzero, else
// pivot + shift, else the shift, else the threshold itself.
template <class T>
inline T stabilize_pivot(const T& pivot, real_t<T> scale, const T& diagonal_shift, real_t<T> diag_pivot_threshold, std::int32_t* status) {
    using R = real_t<T>;
    const auto shift_mag = pivot_modulus(diagonal_shift);
    const auto pivot_mag = pivot_modulus(pivot);
    const auto one = R(1);
    const auto scaled = diag_pivot_threshold * (scale > one ? scale : one);
    const auto threshold = shift_mag > scaled ? shift_mag : scaled;

    if (pivot_mag >= threshold && pivot_mag > R(0)) return pivot;
    if (shift_mag > R(0)) {
        const T shifted = pivot + diagonal_shift;
        if (pivot_modulus(shifted) >= threshold && pivot_modulus(shifted) > R(0)) return shifted;
        if (shift_mag >= threshold) return diagonal_shift;
    }
    if (threshold > R(0)) return T(threshold);
    if (status != nullptr) *status = 1;
    return T(1);
}

// Row i of the factorization in place on the slots at(s) (lvalues; rows of earlier levels done):
// IKJ elimination by the plan's steps, the relative drop (MILU folds dropped entries into the
// diagonal), the fill quota (largest ceil(fill_factor * row_nnz[i]) - 1 off-diagonals survive;
// on a magnitude tie the later slot drops) and the stabilised pivot. Native arithmetic, hand
// loops over absolute slots (sass_parity strict: a v2-owned loop from 0 moved LLVM's runtime
// unroll remainder, LDG 38->66, and arith::Native msub as a call costs complex<float> +4 REG).
// Arguments in the order the body reads them: a launcher's capture layout follows it.
template <class T, class W>
BATCHLAS_DEVICE_INLINE void eliminate_row(const W& at, int i, const int* step_ptr, const int* step_target, const int* step_pivot,
                                       const int* step_upd_ptr, const int* upd_dst, const int* upd_src, const int* sym_ro,
                                       const int* diag_abs, real_t<T> drop_tolerance, bool modified_ilu, real_t<T> fill_factor,
                                       const int* orig_nnz, const T& diagonal_shift, real_t<T> diag_pivot_threshold, std::int32_t* st) {
    using R = real_t<T>;
    for (int e = step_ptr[i]; e < step_ptr[i + 1]; ++e) {
        const T lij = at(step_target[e]) / at(step_pivot[e]);
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
            at(p) = T(0);
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
        at(victim) = T(0);
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

// M := the ILU(k) factor of one batch item's A in plan's pattern, on executor ex (any n).
//   A  the item's values in A's own CSR order (A(p), p < plan.a_nnz; the pattern is plan's).
//   M  plan.nnz slots M(s), lvalues, distinct from A (global, or local memory of nnz elements).
// Zeroes M, scatters A, then one level per barrier: the rows of a level split over the items.
// Every slot of the pattern is written; a dropped entry is +0, so M is the host route's factor
// expanded to the symbolic pattern (the host keeps the union of the items' survivors), bit for
// bit: both run detail::eliminate_row. Returns the smallest 1-based row whose pivot was
// unusable (no shift and a zero pivot: the host route throws convergence_error), else 0,
// uniform across the executor; elimination continues past it with pivot 1. Collective.
template <class Tier, class Ex, class AV, class MV, class T>
BATCHLAS_DEVICE_HELPER int iluk_factorize(const Ex& ex, const AV& A, const IlukPlan& plan, const MV& M, const IlukOptions<T>& opts,
                                       const Workspace& = {}) {
    static_assert(std::is_same_v<resolve_tier_t<routine::iluk_factorize, T, Tier>, tier::Blocked<1>>,
                  "iluk_factorize: the tier is tier::Blocked<1>");
    for_each_owned(ex, plan.nnz, [&](int s) { M(s) = T(0); });
    sync(ex, Mem::Global);
    for_each_owned(ex, plan.a_nnz, [&](int p) {
        const int s = plan.a_to_slot[p];
        if (s >= 0) M(s) = static_cast<T>(A(p));
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
