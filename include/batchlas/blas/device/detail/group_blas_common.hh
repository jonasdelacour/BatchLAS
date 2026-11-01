// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

#include <batchlas/device_limits.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-collectives.hh>

#include <sycl/sycl.hpp>

namespace batchlas::device {

/// @addtogroup api_device_blas
/// @{

/// @brief Which implementation family a device BLAS call may use.
///
/// Level-1 operations, `ger` and `trmv` have one implementation and ignore it.
/// @see @ref design_device_group_blas
enum class DeviceBlasPolicy {
    Auto,        ///< any eligible path; the sub-group fast paths accept sub-group size 16 or 32
    Generic,     ///< the portable group-reduction path only; never stages through local memory
    Subgroup16,  ///< fast paths only when the actual sub-group size is 16
    Subgroup32,  ///< fast paths only when the actual sub-group size is 32
};

/// @brief Kind of launch a DeviceBlasLaunchInfo describes.
enum class DeviceBlasLaunchKind {
    Group,     ///< the call receives a `sycl::group` (or `sycl::sub_group`)
    NdItem1D,  ///< the call receives a `sycl::nd_item<1>`
    NdItem3D,  ///< the call receives a `sycl::nd_item<3>` whose dimensions 1 and 2 index output tiles
};

/// @brief Host-side description of a launch, for the `*_workspace_elements` queries.
///
/// A query evaluates the same eligibility predicates as the call, so the
/// description must match the launch actually made. Build it with
/// make_group_launch_info(), make_nd_item_1d_launch_info() or
/// make_nd_item_3d_launch_info().
struct DeviceBlasLaunchInfo {
    int local_size = 1;     ///< work-items per work-group (per sub-group for a sub-group executor)
    int subgroup_size = 1;  ///< sub-group size of the kernel; 1 for DeviceBlasLaunchKind::Group
    DeviceBlasLaunchKind kind = DeviceBlasLaunchKind::Group;  ///< executor the call receives
};

/// @brief Launch description for a call that receives a `sycl::group`.
/// @param[in] local_size  work-items in the executor; clamped to at least 1
/// @return a DeviceBlasLaunchInfo of kind DeviceBlasLaunchKind::Group with sub-group size 1
/// @ingroup api_device_blas_lowlevel
inline constexpr DeviceBlasLaunchInfo make_group_launch_info(int local_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), 1, DeviceBlasLaunchKind::Group};
}

/// @brief Launch description for a call that receives a `sycl::nd_item<1>`.
/// @param[in] local_size     work-group size; clamped to at least 1
/// @param[in] subgroup_size  sub-group size the kernel runs with; clamped to at least 1
/// @return a DeviceBlasLaunchInfo of kind DeviceBlasLaunchKind::NdItem1D
/// @ingroup api_device_blas_lowlevel
inline constexpr DeviceBlasLaunchInfo make_nd_item_1d_launch_info(int local_size, int subgroup_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), std::max(1, subgroup_size), DeviceBlasLaunchKind::NdItem1D};
}

/// @brief Launch description for a call that receives a `sycl::nd_item<3>` (tiled output).
/// @param[in] local_size     work-group size; clamped to at least 1; the register-tiled paths require 256
/// @param[in] subgroup_size  sub-group size the kernel runs with; clamped to at least 1
/// @return a DeviceBlasLaunchInfo of kind DeviceBlasLaunchKind::NdItem3D
/// @ingroup api_device_blas_lowlevel
inline constexpr DeviceBlasLaunchInfo make_nd_item_3d_launch_info(int local_size, int subgroup_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), std::max(1, subgroup_size), DeviceBlasLaunchKind::NdItem3D};
}

/// @brief Operands of a matrix-vector operation: \f$ y := \alpha\,\mathrm{op}(A)\,x + \beta\,y \f$.
/// @see make_matvec_operand()
template <typename T>
struct MatrixVectorOperand {
    VectorView<T> x{};   ///< input vector
    VectorView<T> y{};   ///< output vector; read (scaled by \f$\beta\f$) and overwritten
    T alpha = T(1);      ///< \f$\alpha\f$, scale of the product
    T beta = T(0);       ///< \f$\beta\f$, scale of the input \f$y\f$; \f$\beta = 0\f$ still reads \f$y\f$
};

/// @brief Operands of a matrix-matrix operation: \f$ C := \alpha\,(\ldots B \ldots) + \beta\,C \f$.
/// @see make_matmat_operand()
template <typename T>
struct MatrixMatrixOperand {
    KernelMatrixView<T, MatrixFormat::Dense> b{};  ///< second input matrix
    KernelMatrixView<T, MatrixFormat::Dense> c{};  ///< output matrix; read (scaled by \f$\beta\f$) and overwritten
    T alpha = T(1);                                ///< \f$\alpha\f$, scale of the product
    T beta = T(0);                                 ///< \f$\beta\f$, scale of the input \f$C\f$; \f$\beta = 0\f$ still reads \f$C\f$
};

/// @brief Operands of a rank-k update: \f$ C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(A)^{T|H} + \beta\,C \f$.
/// @see make_rankk_operand()
template <typename T>
struct RankKOperand {
    KernelMatrixView<T, MatrixFormat::Dense> c{};  ///< square output; only the `uplo` triangle is touched
    T alpha = T(1);                                ///< \f$\alpha\f$, scale of the product
    T beta = T(0);                                 ///< \f$\beta\f$, scale of the input \f$C\f$; \f$\beta = 0\f$ still reads \f$C\f$
};

/// @brief Operands of a rank-1 update: \f$ A := A + \alpha\,x\,y^{T} \f$ (conjugations per call).
/// @see make_rank1_update_operand()
template <typename T>
struct Rank1UpdateOperand {
    VectorView<T> y{};                             ///< right vector, length `a.cols()`
    KernelMatrixView<T, MatrixFormat::Dense> a{};  ///< matrix updated in place
    T alpha = T(1);                                ///< \f$\alpha\f$, scale of the outer product
};

/// @brief Runtime transpose mode of a matrix-vector operation.
struct MatrixVectorTransform {
    Transpose trans = Transpose::NoTrans;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
};

/// @brief Compile-time transpose mode of a matrix-vector operation.
template <Transpose TransV>
struct MatrixVectorTransformTag {
    static constexpr Transpose trans = TransV;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
};

/// @brief Runtime transpose modes of a general matrix-matrix product.
struct GeneralMatrixTransform {
    Transpose trans_a = Transpose::NoTrans;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
    Transpose trans_b = Transpose::NoTrans;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$B\f$
};

/// @brief Compile-time transpose modes of a general matrix-matrix product.
template <Transpose TransAV, Transpose TransBV>
struct GeneralMatrixTransformTag {
    static constexpr Transpose trans_a = TransAV;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
    static constexpr Transpose trans_b = TransBV;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$B\f$
};

/// @brief Mode of a triangular product (trmv, trmm).
struct TriangularTransform {
    Side side = Side::Left;                ///< \f$A\f$ multiplies from the left or the right
    Uplo uplo = Uplo::Upper;               ///< which triangle of \f$A\f$ is stored
    Transpose trans = Transpose::NoTrans;  ///< \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
    Diag diag = Diag::NonUnit;             ///< Unit: \f$a_{ii} = 1\f$ is assumed and the diagonal is not read
};

/// @brief Mode of a symmetric or Hermitian product (symv, hemv, symm).
struct SymmetricTransform {
    Side side = Side::Left;   ///< \f$A\f$ multiplies from the left or the right
    Uplo uplo = Uplo::Upper;  ///< which triangle of \f$A\f$ is stored and read
    bool hermitian = false;   ///< mirror with conjugation
};

/// @brief Conjugation mode of a rank-1 update (ger, geru, gerc).
struct OuterProductTransform {
    bool conjugate_x = false;  ///< use \f$\bar{x}\f$
    bool conjugate_y = false;  ///< use \f$\bar{y}\f$
};

/// @brief Compile-time conjugation mode of a rank-1 update.
template <bool ConjugateXV, bool ConjugateYV>
struct OuterProductTransformTag {
    static constexpr bool conjugate_x = ConjugateXV;  ///< use \f$\bar{x}\f$
    static constexpr bool conjugate_y = ConjugateYV;  ///< use \f$\bar{y}\f$
};

/// @brief Mode of a symmetric or Hermitian rank-2k update (syr2k, her2k).
struct SymmetricRank2kTransform {
    Uplo uplo = Uplo::Upper;               ///< triangle of \f$C\f$ that is updated
    Transpose trans = Transpose::NoTrans;  ///< NoTrans: \f$A B^{T}\f$; otherwise \f$A^{T} B\f$
    bool hermitian = false;                ///< conjugate transposes and \f$\bar{\alpha}\f$ on the second term
};

/// @brief Mode of a symmetric or Hermitian rank-k update (syrk, herk).
struct SymmetricRankKTransform {
    Uplo uplo = Uplo::Upper;               ///< triangle of \f$C\f$ that is updated
    Transpose trans = Transpose::NoTrans;  ///< NoTrans: \f$A A^{T}\f$; otherwise \f$A^{T} A\f$
    bool hermitian = false;                ///< use conjugate transposes
};

/// @brief Builds a MatrixVectorOperand.
///
/// Only copies the views; nothing is read or written until the operand is passed to a call.
/// @param[in] x      input vector \f$x\f$ of the later call
/// @param[in] y      output vector \f$y\f$ of the later call, which reads and overwrites it
/// @param[in] alpha  \f$\alpha\f$, scale of the product
/// @param[in] beta   \f$\beta\f$, scale of the input \f$y\f$
/// @return `MatrixVectorOperand<T>{x, y, alpha, beta}`
/// @tparam T  scalar type of the views and scales
/// @ingroup api_device_blas_lowlevel
template <typename T>
inline constexpr MatrixVectorOperand<T> make_matvec_operand(const VectorView<T>& x,
                                                            const VectorView<T>& y,
                                                            T alpha = T(1),
                                                            T beta = T(0)) {
    return MatrixVectorOperand<T>{x, y, alpha, beta};
}

/// @brief Builds a MatrixMatrixOperand.
///
/// Only copies the views; nothing is read or written until the operand is passed to a call.
/// @param[in] b      second input matrix \f$B\f$ of the later call
/// @param[in] c      output matrix \f$C\f$ of the later call, which reads and overwrites it
/// @param[in] alpha  \f$\alpha\f$, scale of the product
/// @param[in] beta   \f$\beta\f$, scale of the input \f$C\f$
/// @return `MatrixMatrixOperand<T>{b, c, alpha, beta}`
/// @tparam T  scalar type of the views and scales
/// @ingroup api_device_blas_lowlevel
template <typename T>
inline constexpr MatrixMatrixOperand<T> make_matmat_operand(const KernelMatrixView<T, MatrixFormat::Dense>& b,
                                                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                                                            T alpha = T(1),
                                                            T beta = T(0)) {
    return MatrixMatrixOperand<T>{b, c, alpha, beta};
}

/// @brief Builds a RankKOperand.
///
/// Only copies the view; nothing is read or written until the operand is passed to a call.
/// @param[in] c      square output matrix \f$C\f$ of the later call, whose `uplo` triangle it updates
/// @param[in] alpha  \f$\alpha\f$, scale of the product
/// @param[in] beta   \f$\beta\f$, scale of the input \f$C\f$
/// @return `RankKOperand<T>{c, alpha, beta}`
/// @tparam T  scalar type of the view and scales
/// @ingroup api_device_blas_lowlevel
template <typename T>
inline constexpr RankKOperand<T> make_rankk_operand(const KernelMatrixView<T, MatrixFormat::Dense>& c,
                                                    T alpha = T(1),
                                                    T beta = T(0)) {
    return RankKOperand<T>{c, alpha, beta};
}

/// @brief Builds a Rank1UpdateOperand.
///
/// Only copies the views; nothing is read or written until the operand is passed to a call.
/// @param[in] y      right vector \f$y\f$ of the later call, length `a.cols()`
/// @param[in] a      matrix \f$A\f$ the later call updates in place
/// @param[in] alpha  \f$\alpha\f$, scale of the outer product
/// @return `Rank1UpdateOperand<T>{y, a, alpha}`
/// @tparam T  scalar type of the views and scale
/// @ingroup api_device_blas_lowlevel
template <typename T>
inline constexpr Rank1UpdateOperand<T> make_rank1_update_operand(const VectorView<T>& y,
                                                                 const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                 T alpha = T(1)) {
    return Rank1UpdateOperand<T>{y, a, alpha};
}

/// @}

namespace detail {

template <typename Item>
concept NdItemLike = requires(const Item& item) {
    item.get_group();
    item.get_sub_group();
    item.get_local_linear_id();
    item.get_local_range().size();
};

/// @brief Linear index of the calling work-item within its work-group.
/// @param[in] item  `nd_item` of the calling work-item
/// @return the local linear id as `int`
template <NdItemLike Item>
inline constexpr int item_local_linear_id(const Item& item) {
    return static_cast<int>(item.get_local_linear_id());
}

/// @brief Number of work-items in the work-group of the calling work-item.
/// @param[in] item  `nd_item` of the calling work-item
/// @return the work-group size as `int`
template <NdItemLike Item>
inline constexpr int item_local_linear_range(const Item& item) {
    return static_cast<int>(item.get_local_range().size());
}

template <typename T, typename Workspace>
inline constexpr std::size_t workspace_elements_v = sizeof(Workspace) / sizeof(T);

/// @brief Reinterprets a scalar workspace pointer as a pointer to a workspace struct.
/// @param[in] workspace  start of the caller's workspace; not dereferenced here
/// @return `workspace` reinterpreted as `Workspace*`
template <typename Workspace, typename T>
inline constexpr Workspace* workspace_ptr_cast(T* workspace) {
    static_assert(sizeof(Workspace) % sizeof(T) == 0);
    return reinterpret_cast<Workspace*>(workspace);
}

/// @brief Sums a value over a group; complex values are reduced per component.
///
/// Group collective: every work-item of `group` must call it.
/// @param[in] group  group (or sub-group) to reduce over
/// @param[in] value  this work-item's contribution
/// @return the sum of `value` over all work-items of `group`
template <typename Group, typename T>
inline constexpr T reduce_sum_group(const Group& group, const T& value) {
    if constexpr (ComplexScalar<T>) {
        using Real = typename T::value_type;
        const Real re = batchlas::portable::reduce_over_group(group, value.real(), sycl::plus<Real>());
        const Real im = batchlas::portable::reduce_over_group(group, value.imag(), sycl::plus<Real>());
        return T(re, im);
    } else {
        return batchlas::portable::reduce_over_group(group, value, sycl::plus<T>());
    }
}

/// @brief Sums a value over the work-group of an `nd_item`.
///
/// Work-group collective: every work-item of the work-group must call it.
/// @param[in] item   `nd_item` of the calling work-item
/// @param[in] value  this work-item's contribution
/// @return the sum of `value` over the work-group
template <NdItemLike Item, typename T>
inline constexpr T reduce_sum_group(const Item& item, const T& value) {
    return reduce_sum_group(item.get_group(), value);
}

/// @brief Complex conjugate that is the identity on real scalars.
/// @param[in] value  scalar \f$v\f$
/// @return \f$\bar{v}\f$ for a complex `T`, otherwise \f$v\f$
template <typename T>
inline constexpr T conj(const T& value) {
    if constexpr (ComplexScalar<T>) {
        return std::conj(value);
    } else {
        return value;
    }
}

/// @brief Linear index of the calling work-item within a group.
/// @param[in] group  group (or sub-group) executor
/// @return the local linear id as `int`
template <typename Group>
inline constexpr int group_local_linear_id(const Group& group) {
    return static_cast<int>(group.get_local_linear_id());
}

/// @brief Linear index of the calling work-item within the work-group of an `nd_item`.
/// @param[in] item  `nd_item` of the calling work-item
/// @return the local linear id as `int`
template <NdItemLike Item>
inline constexpr int group_local_linear_id(const Item& item) {
    return item_local_linear_id(item);
}

/// @brief Number of work-items in a group.
/// @param[in] group  group (or sub-group) executor
/// @return the group size as `int`
template <typename Group>
inline constexpr int group_local_linear_range(const Group& group) {
    return static_cast<int>(group.get_local_range().size());
}

/// @brief Number of work-items in the work-group of an `nd_item`.
/// @param[in] item  `nd_item` of the calling work-item
/// @return the work-group size as `int`
template <NdItemLike Item>
inline constexpr int group_local_linear_range(const Item& item) {
    return item_local_linear_range(item);
}

/// @brief Executor size recorded in a host-side launch description.
/// @param[in] launch  launch description of the call
/// @return `launch.local_size`, clamped to at least 1
inline constexpr int group_local_linear_range(const DeviceBlasLaunchInfo& launch) {
    return std::max(1, launch.local_size);
}

/// @brief Whether the calling work-item is the first of its executor.
/// @param[in] group  group, sub-group or `nd_item` executor
/// @return true when the local linear id is 0
template <typename Group>
inline constexpr bool group_is_leader(const Group& group) {
    return group_local_linear_id(group) == 0;
}

template <typename T, typename Op>
concept MatrixVectorOperandFor = std::same_as<std::remove_cvref_t<Op>, MatrixVectorOperand<T>>;

template <typename T, typename Op>
concept MatrixMatrixOperandFor = std::same_as<std::remove_cvref_t<Op>, MatrixMatrixOperand<T>>;

template <typename T, typename Op>
concept VectorOperandFor = std::same_as<std::remove_cvref_t<Op>, VectorView<T>>;

/// @brief Conjugates a scalar when a runtime flag asks for it.
/// @param[in] value      scalar \f$v\f$
/// @param[in] conjugate  whether to conjugate
/// @return \f$\bar{v}\f$ when `conjugate` is true, otherwise \f$v\f$
template <typename T>
inline constexpr T maybe_conjugate(const T& value, bool conjugate) {
    return conjugate ? conj(value) : value;
}

/// @brief Whether the address spans of two vectors intersect.
///
/// Compares the span from the first to the last element (either increment sign);
/// interleaved vectors whose elements never coincide still count as overlapping.
/// @param[in] lhs  first vector
/// @param[in] rhs  second vector
/// @return true when both are non-empty and their spans intersect
template <typename T>
inline constexpr bool views_overlap(const VectorView<T>& lhs,
                                    const VectorView<T>& rhs) {
    if (lhs.size() <= 0 || rhs.size() <= 0) {
        return false;
    }

    auto* lhs_begin = lhs.data_ptr();
    auto* lhs_end = lhs_begin + (lhs.size() - 1) * lhs.inc();
    if (lhs_end < lhs_begin) {
        std::swap(lhs_begin, lhs_end);
    }

    auto* rhs_begin = rhs.data_ptr();
    auto* rhs_end = rhs_begin + (rhs.size() - 1) * rhs.inc();
    if (rhs_end < rhs_begin) {
        std::swap(rhs_begin, rhs_end);
    }

    return !(lhs_end < rhs_begin || rhs_end < lhs_begin);
}

/// @brief Whether the address spans of two column-major matrices intersect.
///
/// Compares the span from the first to the last element; disjoint sub-blocks of one
/// allocation whose spans interleave still count as overlapping.
/// @param[in] lhs  first matrix
/// @param[in] rhs  second matrix
/// @return true when both are non-empty and their spans intersect
template <typename T>
inline constexpr bool views_overlap(const KernelMatrixView<T, MatrixFormat::Dense>& lhs,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& rhs) {
    if (lhs.rows() <= 0 || lhs.cols() <= 0 || rhs.rows() <= 0 || rhs.cols() <= 0) {
        return false;
    }

    auto* lhs_begin = lhs.data();
    auto* lhs_end = lhs_begin + (lhs.cols() - 1) * lhs.ld() + (lhs.rows() - 1);
    if (lhs_end < lhs_begin) {
        std::swap(lhs_begin, lhs_end);
    }

    auto* rhs_begin = rhs.data();
    auto* rhs_end = rhs_begin + (rhs.cols() - 1) * rhs.ld() + (rhs.rows() - 1);
    if (rhs_end < rhs_begin) {
        std::swap(rhs_begin, rhs_end);
    }

    return !(lhs_end < rhs_begin || rhs_end < lhs_begin);
}

/// @brief Entry \f$(i, j)\f$ of \f$\mathrm{op}(A)\f$ for a runtime transpose mode.
/// @param[in] a      matrix \f$A\f$
/// @param[in] row    row index \f$i\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] col    column index \f$j\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
/// @return \f$a_{ij}\f$, \f$a_{ji}\f$ or \f$\bar{a}_{ji}\f$
template <typename T>
inline constexpr T matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                int row,
                                int col,
                                Transpose trans) {
    if (trans == Transpose::NoTrans) {
        return a(row, col);
    }
    if (trans == Transpose::Trans) {
        return a(col, row);
    }
    return conj(a(col, row));
}

/// @brief Entry \f$(i, j)\f$ of \f$\mathrm{op}(A)\f$ for a compile-time transpose mode `TransV`.
/// @param[in] a    matrix \f$A\f$
/// @param[in] row  row index \f$i\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] col  column index \f$j\f$ in \f$\mathrm{op}(A)\f$
/// @return \f$a_{ij}\f$, \f$a_{ji}\f$ or \f$\bar{a}_{ji}\f$
template <Transpose TransV, typename T>
inline constexpr T matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                int row,
                                int col) {
    if constexpr (TransV == Transpose::NoTrans) {
        return a(row, col);
    } else if constexpr (TransV == Transpose::Trans) {
        return a(col, row);
    } else {
        return conj(a(col, row));
    }
}

/// @brief Length of the vector \f$\mathrm{op}(A)\f$ multiplies (its column count).
/// @param[in] a      matrix \f$A\f$
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
/// @return `a.cols()` for NoTrans, otherwise `a.rows()`
template <typename T>
inline constexpr int input_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.cols() : a.rows();
}

/// @brief Column count of \f$\mathrm{op}(A)\f$ from the extents of \f$A\f$.
/// @param[in] rows   row count of \f$A\f$
/// @param[in] cols   column count of \f$A\f$
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
/// @return `cols` for NoTrans, otherwise `rows`
inline constexpr int input_size(int rows, int cols, Transpose trans) {
    return trans == Transpose::NoTrans ? cols : rows;
}

/// @brief Column count of \f$\mathrm{op}(A)\f$ for a compile-time transpose mode `TransV`.
/// @param[in] a  matrix \f$A\f$
/// @return `a.cols()` for NoTrans, otherwise `a.rows()`
template <Transpose TransV, typename T>
inline constexpr int input_size(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    if constexpr (TransV == Transpose::NoTrans) {
        return a.cols();
    } else {
        return a.rows();
    }
}

/// @brief Length of the vector \f$\mathrm{op}(A)\f$ produces (its row count).
/// @param[in] a      matrix \f$A\f$
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
/// @return `a.rows()` for NoTrans, otherwise `a.cols()`
template <typename T>
inline constexpr int output_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.rows() : a.cols();
}

/// @brief Row count of \f$\mathrm{op}(A)\f$ from the extents of \f$A\f$.
/// @param[in] rows   row count of \f$A\f$
/// @param[in] cols   column count of \f$A\f$
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
/// @return `rows` for NoTrans, otherwise `cols`
inline constexpr int output_size(int rows, int cols, Transpose trans) {
    return trans == Transpose::NoTrans ? rows : cols;
}

/// @brief Row count of \f$\mathrm{op}(A)\f$ for a compile-time transpose mode `TransV`.
/// @param[in] a  matrix \f$A\f$
/// @return `a.rows()` for NoTrans, otherwise `a.cols()`
template <Transpose TransV, typename T>
inline constexpr int output_size(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    if constexpr (TransV == Transpose::NoTrans) {
        return a.rows();
    } else {
        return a.cols();
    }
}

/// @brief Asserts (debug builds only) that a matrix view holds a single problem.
/// @param[in] a      matrix \f$A\f$; must have batch size 1
/// @param[in] trans  \f$\mathrm{op}(\cdot)\f$ of the call; currently unused
template <typename T>
inline constexpr void validate_single_problem(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                              Transpose trans) {
    (void)trans;
    assert(a.batch_size() == 1 && "device BLAS expects a single logical matrix; pass batch_item() for batched data");
}

/// @brief Asserts (debug builds only) the shape contract of a gemv call.
/// @param[in] a        matrix \f$A\f$
/// @param[in] operand  \f$x\f$, \f$y\f$ and scales of the call; only shapes are checked
/// @param[in] trans    \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$
template <typename T>
inline constexpr void validate_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                       const MatrixVectorOperand<T>& operand,
                                       Transpose trans) {
    validate_single_problem(a, trans);
    assert(operand.x.batch_size() == 1 && "device::gemv expects a single logical input vector; pass batch_item() for batched data");
    assert(operand.y.batch_size() == 1 && "device::gemv expects a single logical output vector; pass batch_item() for batched data");
    assert(operand.x.size() >= input_size(a, trans) && "device::gemv input vector is too small for the requested transpose mode");
    assert(operand.y.size() >= output_size(a, trans) && "device::gemv output vector is too small for the requested transpose mode");
}

/// @brief Asserts (debug builds only) the shape contract of a gemm call.
/// @param[in] a          matrix \f$A\f$
/// @param[in] operand    \f$B\f$, \f$C\f$ and scales of the call; only shapes are checked
/// @param[in] transform  \f$\mathrm{op}(\cdot)\f$ applied to \f$A\f$ and \f$B\f$
template <typename T>
inline constexpr void validate_gemm_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                            const MatrixMatrixOperand<T>& operand,
                                            GeneralMatrixTransform transform) {
    validate_single_problem(a, transform.trans_a);
    assert(operand.b.batch_size() == 1 && "device::gemm expects a single logical input matrix; pass batch_item() for batched data");
    assert(operand.c.batch_size() == 1 && "device::gemm expects a single logical output matrix; pass batch_item() for batched data");

    const int m = output_size(a, transform.trans_a);
    const int k = input_size(a, transform.trans_a);
    const int b_rows = output_size(operand.b, transform.trans_b);
    const int n = input_size(operand.b, transform.trans_b);

    assert(b_rows == k && "device::gemm operand contract dimension mismatch");
    assert(operand.c.rows() == m && "device::gemm output matrix row count does not match op(A)");
    assert(operand.c.cols() == n && "device::gemm output matrix column count does not match op(B)");
}

/// @brief Asserts (debug builds only) the shape contract of a trmv call.
/// @param[in] a          square triangular matrix \f$A\f$
/// @param[in] operand    \f$x\f$, \f$y\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the product; `side` must be Side::Left
template <typename T>
inline constexpr void validate_triangular_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                  const MatrixVectorOperand<T>& operand,
                                                  TriangularTransform transform) {
    validate_single_problem(a, transform.trans);
    assert(transform.side == Side::Left && "device::trmv currently supports only Side::Left");
    assert(a.rows() == a.cols() && "device::trmv expects a square triangular matrix");
    assert(operand.x.batch_size() == 1 && "device::trmv expects a single logical input vector; pass batch_item() for batched data");
    assert(operand.y.batch_size() == 1 && "device::trmv expects a single logical output vector; pass batch_item() for batched data");
    assert(operand.x.size() >= a.rows() && "device::trmv input vector is too small for the triangular matrix");
    assert(operand.y.size() >= a.rows() && "device::trmv output vector is too small for the triangular matrix");
}

/// @brief Asserts (debug builds only) the shape contract of a symv or hemv call.
/// @param[in] a          square symmetric or Hermitian matrix \f$A\f$
/// @param[in] operand    \f$x\f$, \f$y\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the product; `side` must be Side::Left
template <typename T>
inline constexpr void validate_symmetric_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                 const MatrixVectorOperand<T>& operand,
                                                 SymmetricTransform transform) {
    validate_single_problem(a, Transpose::NoTrans);
    assert(transform.side == Side::Left && "device::symv currently supports only Side::Left");
    assert(a.rows() == a.cols() && "device::symv expects a square symmetric matrix");
    assert(operand.x.batch_size() == 1 && "device::symv expects a single logical input vector; pass batch_item() for batched data");
    assert(operand.y.batch_size() == 1 && "device::symv expects a single logical output vector; pass batch_item() for batched data");
    assert(operand.x.size() >= a.rows() && "device::symv input vector is too small for the symmetric matrix");
    assert(operand.y.size() >= a.rows() && "device::symv output vector is too small for the symmetric matrix");
}

/// @brief Asserts (debug builds only) the shape contract of a trmm call.
/// @param[in] a          square triangular matrix \f$A\f$
/// @param[in] operand    \f$B\f$, \f$C\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the product; `side` selects which extent of \f$B\f$ must match \f$A\f$
template <typename T>
inline constexpr void validate_triangular_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                  const MatrixMatrixOperand<T>& operand,
                                                  TriangularTransform transform) {
    validate_single_problem(a, transform.trans);
    assert(operand.b.batch_size() == 1 && "device::trmm expects a single logical input matrix; pass batch_item() for batched data");
    assert(operand.c.batch_size() == 1 && "device::trmm expects a single logical output matrix; pass batch_item() for batched data");
    assert(operand.b.rows() == operand.c.rows() && "device::trmm input and output matrices must have matching row counts");
    assert(operand.b.cols() == operand.c.cols() && "device::trmm input and output matrices must have matching column counts");
    assert(a.rows() == a.cols() && "device::trmm expects a square triangular matrix");
    if (transform.side == Side::Left) {
        assert(a.rows() == operand.b.rows() && "device::trmm left-side triangular matrix must match input rows");
    } else {
        assert(a.rows() == operand.b.cols() && "device::trmm right-side triangular matrix must match input columns");
    }
}

/// @brief Asserts (debug builds only) the shape contract of a symm or hemm call.
/// @param[in] a          square symmetric or Hermitian matrix \f$A\f$
/// @param[in] operand    \f$B\f$, \f$C\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the product; `side` selects which extent of \f$B\f$ must match \f$A\f$
template <typename T>
inline constexpr void validate_symmetric_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                 const MatrixMatrixOperand<T>& operand,
                                                 SymmetricTransform transform) {
    validate_single_problem(a, Transpose::NoTrans);
    assert(a.rows() == a.cols() && "device::symm expects a square symmetric matrix");
    assert(operand.b.batch_size() == 1 && "device::symm expects a single logical input matrix; pass batch_item() for batched data");
    assert(operand.c.batch_size() == 1 && "device::symm expects a single logical output matrix; pass batch_item() for batched data");
    assert(operand.b.rows() == operand.c.rows() && "device::symm input and output matrices must have matching row counts");
    assert(operand.b.cols() == operand.c.cols() && "device::symm input and output matrices must have matching column counts");
    if (transform.side == Side::Left) {
        assert(a.rows() == operand.b.rows() && "device::symm left-side symmetric matrix must match input rows");
    } else {
        assert(a.rows() == operand.b.cols() && "device::symm right-side symmetric matrix must match input columns");
    }
}

/// @brief Asserts (debug builds only) the shape contract of a ger call.
/// @param[in] x        left vector \f$x\f$, length `operand.a.rows()`
/// @param[in] operand  \f$y\f$, \f$A\f$ and \f$\alpha\f$ of the call; only shapes are checked
template <typename T>
inline constexpr void validate_rank1_operand(const VectorView<T>& x,
                                             const Rank1UpdateOperand<T>& operand) {
    assert(x.batch_size() == 1 && "device::ger expects a single logical input vector; pass batch_item() for batched data");
    assert(operand.y.batch_size() == 1 && "device::ger expects a single logical input vector; pass batch_item() for batched data");
    validate_single_problem(operand.a, Transpose::NoTrans);
    assert(operand.a.rows() == x.size() && "device::ger matrix row count must match x size");
    assert(operand.a.cols() == operand.y.size() && "device::ger matrix column count must match y size");
}

/// @brief Asserts (debug builds only) the shape contract of a syr2k or her2k call.
/// @param[in] a          first input matrix \f$A\f$
/// @param[in] operand    \f$B\f$ (same shape as \f$A\f$), \f$C\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the update; `trans` fixes the order of \f$C\f$
template <typename T>
inline constexpr void validate_rank2k_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                              const MatrixMatrixOperand<T>& operand,
                                              SymmetricRank2kTransform transform) {
    validate_single_problem(a, transform.trans);
    assert(operand.b.batch_size() == 1 && "device::syr2k expects a single logical input matrix; pass batch_item() for batched data");
    assert(operand.c.batch_size() == 1 && "device::syr2k expects a single logical output matrix; pass batch_item() for batched data");
    assert(a.rows() == operand.b.rows() && "device::syr2k input matrices must have matching row counts");
    assert(a.cols() == operand.b.cols() && "device::syr2k input matrices must have matching column counts");
    const int extent = output_size(a, transform.trans);
    assert(operand.c.rows() == extent && "device::syr2k output matrix row count does not match op(A)");
    assert(operand.c.cols() == extent && "device::syr2k output matrix column count does not match op(A)");
}

/// @brief Asserts (debug builds only) the shape contract of a syrk or herk call.
/// @param[in] a          input matrix \f$A\f$
/// @param[in] operand    \f$C\f$ and scales of the call; only shapes are checked
/// @param[in] transform  mode of the update; `trans` fixes the order of \f$C\f$
template <typename T>
inline constexpr void validate_rankk_operand(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                             const RankKOperand<T>& operand,
                                             SymmetricRankKTransform transform) {
    validate_single_problem(a, transform.trans);
    assert(operand.c.batch_size() == 1 && "device::syrk expects a single logical output matrix; pass batch_item() for batched data");
    const int extent = output_size(a, transform.trans);
    assert(operand.c.rows() == extent && "device::syrk output matrix row count does not match op(A)");
    assert(operand.c.cols() == extent && "device::syrk output matrix column count does not match op(A)");
}

/// @brief Asserts (debug builds only) that a vector view holds a single problem.
/// @param[in] x        vector; must have batch size 1
/// @param[in] op_name  name of the calling operation; currently unused
template <typename T>
inline constexpr void validate_vector_operand(const VectorView<T>& x,
                                              const char* op_name) {
    (void)op_name;
    assert(x.batch_size() == 1 && "device vector BLAS expects a single logical vector; pass batch_item() for batched data");
}

/// @brief Asserts (debug builds only) that two vectors are single problems of equal length.
/// @param[in] x        first vector
/// @param[in] y        second vector
/// @param[in] op_name  name of the calling operation; currently unused
template <typename T>
inline constexpr void validate_vector_operands(const VectorView<T>& x,
                                               const VectorView<T>& y,
                                               const char* op_name) {
    validate_vector_operand(x, op_name);
    validate_vector_operand(y, op_name);
    assert(x.size() == y.size() && "device vector BLAS operands must have matching sizes");
}

/// @brief Asserts (debug builds only) that three vectors are single problems of equal length.
/// @param[in] x        first vector
/// @param[in] y        second vector
/// @param[in] z        third vector
/// @param[in] op_name  name of the calling operation; currently unused
template <typename T>
inline constexpr void validate_vector_operands(const VectorView<T>& x,
                                               const VectorView<T>& y,
                                               const VectorView<T>& z,
                                               const char* op_name) {
    validate_vector_operands(x, y, op_name);
    validate_vector_operand(z, op_name);
    assert(x.size() == z.size() && "device vector BLAS operands must have matching sizes");
}

/// @brief Asserts (debug builds only) that a Hadamard output and its inputs are single problems of equal length.
/// @param[in] z        output vector of the call; only its shape is checked
/// @param[in] op_name  name of the calling operation; currently unused
/// @param[in] inputs   one or more input vectors, each of length `z.size()`
template <typename T, typename... Views>
    requires(sizeof...(Views) > 0 && (VectorOperandFor<T, Views> && ...))
inline constexpr void validate_hadamard_operands(const VectorView<T>& z,
                                                 const char* op_name,
                                                 const Views&... inputs) {
    validate_vector_operand(z, op_name);
    ((validate_vector_operand(inputs, op_name),
      assert(inputs.size() == z.size() && "device hadamard operands must have matching sizes")),
     ...);
}

template <Side SideV, Uplo UploV, Transpose TransposeV, Diag DiagV>
struct TriangularTransformTag {
    static constexpr Side side = SideV;
    static constexpr Uplo uplo = UploV;
    static constexpr Transpose trans = TransposeV;
    static constexpr Diag diag = DiagV;
};

template <Side SideV, Uplo UploV, bool HermitianV>
struct SymmetricTransformTag {
    static constexpr Side side = SideV;
    static constexpr Uplo uplo = UploV;
    static constexpr bool hermitian = HermitianV;
};

template <Uplo UploV, Transpose TransposeV, bool HermitianV>
struct SymmetricRankTransformTag {
    static constexpr Uplo uplo = UploV;
    static constexpr Transpose trans = TransposeV;
    static constexpr bool hermitian = HermitianV;
};

/// @brief Whether entry \f$(i, j)\f$ lies in the stored triangle `Tag::uplo` (diagonal included).
/// @param[in] row  row index \f$i\f$
/// @param[in] col  column index \f$j\f$
/// @return \f$i \ge j\f$ for Uplo::Lower, otherwise \f$i \le j\f$
template <typename Tag>
inline constexpr bool triangular_storage_contains(int row, int col) {
    if constexpr (Tag::uplo == Uplo::Lower) {
        return row >= col;
    }
    return row <= col;
}

/// @brief Whether entry \f$(i, j)\f$ lies in a stored triangle (diagonal included).
/// @param[in] uplo  stored triangle
/// @param[in] row   row index \f$i\f$
/// @param[in] col   column index \f$j\f$
/// @return \f$i \ge j\f$ for Uplo::Lower, otherwise \f$i \le j\f$
inline constexpr bool triangular_storage_contains(Uplo uplo, int row, int col) {
    return uplo == Uplo::Lower ? row >= col : row <= col;
}

/// @brief Entry \f$(i, j)\f$ of \f$\mathrm{op}(A)\f$ for a triangular \f$A\f$ described by a compile-time `Tag`.
///
/// Reads only the stored triangle; the unit diagonal is not read.
/// @param[in] a    triangular matrix \f$A\f$
/// @param[in] row  row index \f$i\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] col  column index \f$j\f$ in \f$\mathrm{op}(A)\f$
/// @return the entry, 1 on a unit diagonal, 0 outside the stored triangle
template <typename Tag, typename T>
inline constexpr T triangular_matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                           int row,
                                           int col) {
    int src_row = row;
    int src_col = col;
    if constexpr (Tag::trans != Transpose::NoTrans) {
        src_row = col;
        src_col = row;
    }
    if constexpr (Tag::diag == Diag::Unit) {
        if (src_row == src_col) {
            return T(1);
        }
    }
    if (!triangular_storage_contains<Tag>(src_row, src_col)) {
        return T(0);
    }

    T value = a(src_row, src_col);
    if constexpr (Tag::trans == Transpose::ConjTrans) {
        value = conj(value);
    }
    return value;
}

/// @brief Entry \f$(i, j)\f$ of \f$\mathrm{op}(A)\f$ for a triangular \f$A\f$ with a runtime mode.
///
/// Reads only the stored triangle; the unit diagonal is not read.
/// @param[in] a          triangular matrix \f$A\f$
/// @param[in] row        row index \f$i\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] col        column index \f$j\f$ in \f$\mathrm{op}(A)\f$
/// @param[in] transform  `uplo`, `trans` and `diag` of \f$A\f$; `side` is ignored
/// @return the entry, 1 on a unit diagonal, 0 outside the stored triangle
template <typename T>
inline constexpr T triangular_matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                           int row,
                                           int col,
                                           TriangularTransform transform) {
    int src_row = row;
    int src_col = col;
    if (transform.trans != Transpose::NoTrans) {
        src_row = col;
        src_col = row;
    }
    if (src_row == src_col && transform.diag == Diag::Unit) {
        return T(1);
    }
    if (!triangular_storage_contains(transform.uplo, src_row, src_col)) {
        return T(0);
    }

    T value = a(src_row, src_col);
    if (transform.trans == Transpose::ConjTrans) {
        value = conj(value);
    }
    return value;
}

/// @brief Entry \f$(i, j)\f$ of a symmetric or Hermitian \f$A\f$ described by a compile-time `Tag`.
///
/// Reads only the stored triangle `Tag::uplo` and mirrors the other, conjugating when `Tag::hermitian`.
/// @param[in] a    symmetric or Hermitian matrix \f$A\f$
/// @param[in] row  row index \f$i\f$
/// @param[in] col  column index \f$j\f$
/// @return \f$a_{ij}\f$ from the stored triangle, or its mirror \f$a_{ji}\f$ (\f$\bar{a}_{ji}\f$ if Hermitian)
template <typename Tag, typename T>
inline constexpr T symmetric_matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          int row,
                                          int col) {
    if constexpr (Tag::uplo == Uplo::Lower) {
        if (row >= col) {
            return a(row, col);
        }
        const T value = a(col, row);
        if constexpr (Tag::hermitian) {
            return conj(value);
        }
        return value;
    }
    if (row <= col) {
        return a(row, col);
    }
    const T value = a(col, row);
    if constexpr (Tag::hermitian) {
        return conj(value);
    }
    return value;
}

/// @brief Entry \f$(i, j)\f$ of a symmetric or Hermitian \f$A\f$ with a runtime mode.
///
/// Reads only the stored triangle and mirrors the other, conjugating when `transform.hermitian`.
/// @param[in] a          symmetric or Hermitian matrix \f$A\f$
/// @param[in] row        row index \f$i\f$
/// @param[in] col        column index \f$j\f$
/// @param[in] transform  `uplo` and `hermitian` of \f$A\f$; `side` is ignored
/// @return \f$a_{ij}\f$ from the stored triangle, or its mirror \f$a_{ji}\f$ (\f$\bar{a}_{ji}\f$ if Hermitian)
template <typename T>
inline constexpr T symmetric_matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          int row,
                                          int col,
                                          SymmetricTransform transform) {
    if (transform.uplo == Uplo::Lower) {
        if (row >= col) {
            return a(row, col);
        }
        const T value = a(col, row);
        return transform.hermitian ? conj(value) : value;
    }
    if (row <= col) {
        return a(row, col);
    }
    const T value = a(col, row);
    return transform.hermitian ? conj(value) : value;
}

/// @brief Stores one output entry: \f$c_{ij} := \alpha\,v + \beta\,c_{ij}\f$.
/// @param[in,out] operand  operand whose \f$C\f$ entry is read and overwritten through its view
/// @param[in]     row      row index \f$i\f$ in \f$C\f$
/// @param[in]     col      column index \f$j\f$ in \f$C\f$
/// @param[in]     value    unscaled product entry \f$v\f$
template <typename T>
inline constexpr void write_matrix_output(const MatrixMatrixOperand<T>& operand,
                                          int row,
                                          int col,
                                          const T& value) {
    operand.c(row, col) = operand.alpha * value + operand.beta * operand.c(row, col);
}

/// @brief Accumulates one rank-1 update entry: \f$a_{ij} := a_{ij} + \alpha\,v\f$.
/// @param[in,out] operand  operand whose \f$A\f$ entry is read and overwritten through its view
/// @param[in]     row      row index \f$i\f$ in \f$A\f$
/// @param[in]     col      column index \f$j\f$ in \f$A\f$
/// @param[in]     value    unscaled outer-product entry \f$v\f$
template <typename T>
inline constexpr void accumulate_rank1_output(const Rank1UpdateOperand<T>& operand,
                                              int row,
                                              int col,
                                              const T& value) {
    operand.a(row, col) += operand.alpha * value;
}

/// @brief Transpose applied to the right factor of a rank-2k product for a compile-time `Tag`.
/// @return ConjTrans or Trans when `Tag::trans` is NoTrans (Hermitian or not), otherwise NoTrans
template <typename Tag>
inline constexpr Transpose rank2k_rhs_transform() {
    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

/// @brief Transpose applied to the right factor of a rank-2k product.
/// @param[in] transform  mode of the update
/// @return ConjTrans or Trans when `transform.trans` is NoTrans (Hermitian or not), otherwise NoTrans
inline constexpr Transpose rank2k_rhs_transform(SymmetricRank2kTransform transform) {
    if (transform.trans == Transpose::NoTrans) {
        return transform.hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

/// @brief Transpose applied to the right factor of a rank-k product for a compile-time `Tag`.
/// @return ConjTrans or Trans when `Tag::trans` is NoTrans (Hermitian or not), otherwise NoTrans
template <typename Tag>
inline constexpr Transpose rankk_rhs_transform() {
    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

/// @brief Transpose applied to the right factor of a rank-k product.
/// @param[in] transform  mode of the update
/// @return ConjTrans or Trans when `transform.trans` is NoTrans (Hermitian or not), otherwise NoTrans
inline constexpr Transpose rankk_rhs_transform(SymmetricRankKTransform transform) {
    if (transform.trans == Transpose::NoTrans) {
        return transform.hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

/// @brief Scale of the second term of a rank-2k update for a compile-time `Tag`.
/// @param[in] alpha  \f$\alpha\f$ of the call
/// @return \f$\bar{\alpha}\f$ when `Tag::hermitian`, otherwise \f$\alpha\f$
template <typename Tag, typename T>
inline constexpr T secondary_rank2k_alpha(const T& alpha) {
    if constexpr (Tag::hermitian) {
        return conj(alpha);
    }
    return alpha;
}

/// @brief Scale of the second term of a rank-2k update.
/// @param[in] alpha      \f$\alpha\f$ of the call
/// @param[in] hermitian  whether the update is Hermitian (her2k)
/// @return \f$\bar{\alpha}\f$ when `hermitian`, otherwise \f$\alpha\f$
template <typename T>
inline constexpr T secondary_rank2k_alpha(const T& alpha, bool hermitian) {
    return hermitian ? conj(alpha) : alpha;
}

/// @brief Copy of a symmetric mode with `hermitian` set to whether `T` is complex.
/// @param[in] transform  mode of the call; its `hermitian` flag is replaced
/// @return `transform` with `hermitian = ComplexScalar<T>`
template <typename T>
inline constexpr SymmetricTransform canonical_hermitian_transform(SymmetricTransform transform) {
    return SymmetricTransform{
        .side = transform.side,
        .uplo = transform.uplo,
        .hermitian = ComplexScalar<T>,
    };
}

/// @brief Copy of a rank-2k mode with `hermitian` set to whether `T` is complex.
/// @param[in] transform  mode of the call; its `hermitian` flag is replaced
/// @return `transform` with `hermitian = ComplexScalar<T>`
template <typename T>
inline constexpr SymmetricRank2kTransform canonical_hermitian_transform(SymmetricRank2kTransform transform) {
    return SymmetricRank2kTransform{
        .uplo = transform.uplo,
        .trans = transform.trans,
        .hermitian = ComplexScalar<T>,
    };
}

/// @brief Copy of a rank-k mode with `hermitian` set to whether `T` is complex.
/// @param[in] transform  mode of the call; its `hermitian` flag is replaced
/// @return `transform` with `hermitian = ComplexScalar<T>`
template <typename T>
inline constexpr SymmetricRankKTransform canonical_hermitian_transform(SymmetricRankKTransform transform) {
    return SymmetricRankKTransform{
        .uplo = transform.uplo,
        .trans = transform.trans,
        .hermitian = ComplexScalar<T>,
    };
}

} // namespace detail

} // namespace batchlas::device