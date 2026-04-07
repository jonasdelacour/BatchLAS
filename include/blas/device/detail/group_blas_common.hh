#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

#include <batchlas/device_limits.hh>

#include <blas/enums.hh>
#include <blas/matrix.hh>

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/sycl.hpp>

namespace batchlas::device {

enum class DeviceBlasPolicy {
    Auto,
    Generic,
    Subgroup16,
    Subgroup32,
};

enum class DeviceBlasLaunchKind {
    Group,
    NdItem1D,
    NdItem3D,
};

struct DeviceBlasLaunchInfo {
    int local_size = 1;
    int subgroup_size = 1;
    DeviceBlasLaunchKind kind = DeviceBlasLaunchKind::Group;
};

inline constexpr DeviceBlasLaunchInfo make_group_launch_info(int local_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), 1, DeviceBlasLaunchKind::Group};
}

inline constexpr DeviceBlasLaunchInfo make_nd_item_1d_launch_info(int local_size, int subgroup_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), std::max(1, subgroup_size), DeviceBlasLaunchKind::NdItem1D};
}

inline constexpr DeviceBlasLaunchInfo make_nd_item_3d_launch_info(int local_size, int subgroup_size) {
    return DeviceBlasLaunchInfo{std::max(1, local_size), std::max(1, subgroup_size), DeviceBlasLaunchKind::NdItem3D};
}

template <typename T>
struct MatrixVectorOperand {
    VectorView<T> x{};
    VectorView<T> y{};
    T alpha = T(1);
    T beta = T(0);
};

template <typename T>
struct MatrixMatrixOperand {
    KernelMatrixView<T, MatrixFormat::Dense> b{};
    KernelMatrixView<T, MatrixFormat::Dense> c{};
    T alpha = T(1);
    T beta = T(0);
};

template <typename T>
struct RankKOperand {
    KernelMatrixView<T, MatrixFormat::Dense> c{};
    T alpha = T(1);
    T beta = T(0);
};

template <typename T>
struct Rank1UpdateOperand {
    VectorView<T> y{};
    KernelMatrixView<T, MatrixFormat::Dense> a{};
    T alpha = T(1);
};

struct MatrixVectorTransform {
    Transpose trans = Transpose::NoTrans;
};

template <Transpose TransV>
struct MatrixVectorTransformTag {
    static constexpr Transpose trans = TransV;
};

struct GeneralMatrixTransform {
    Transpose trans_a = Transpose::NoTrans;
    Transpose trans_b = Transpose::NoTrans;
};

template <Transpose TransAV, Transpose TransBV>
struct GeneralMatrixTransformTag {
    static constexpr Transpose trans_a = TransAV;
    static constexpr Transpose trans_b = TransBV;
};

struct TriangularTransform {
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    Transpose trans = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
};

struct SymmetricTransform {
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    bool hermitian = false;
};

struct OuterProductTransform {
    bool conjugate_x = false;
    bool conjugate_y = false;
};

template <bool ConjugateXV, bool ConjugateYV>
struct OuterProductTransformTag {
    static constexpr bool conjugate_x = ConjugateXV;
    static constexpr bool conjugate_y = ConjugateYV;
};

struct SymmetricRank2kTransform {
    Uplo uplo = Uplo::Upper;
    Transpose trans = Transpose::NoTrans;
    bool hermitian = false;
};

struct SymmetricRankKTransform {
    Uplo uplo = Uplo::Upper;
    Transpose trans = Transpose::NoTrans;
    bool hermitian = false;
};

template <typename T>
using GemvOperand = MatrixVectorOperand<T>;

template <typename T>
inline constexpr MatrixVectorOperand<T> make_matvec_operand(const VectorView<T>& x,
                                                            const VectorView<T>& y,
                                                            T alpha = T(1),
                                                            T beta = T(0)) {
    return MatrixVectorOperand<T>{x, y, alpha, beta};
}

template <typename T>
inline constexpr MatrixMatrixOperand<T> make_matmat_operand(const KernelMatrixView<T, MatrixFormat::Dense>& b,
                                                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                                                            T alpha = T(1),
                                                            T beta = T(0)) {
    return MatrixMatrixOperand<T>{b, c, alpha, beta};
}

template <typename T>
inline constexpr RankKOperand<T> make_rankk_operand(const KernelMatrixView<T, MatrixFormat::Dense>& c,
                                                    T alpha = T(1),
                                                    T beta = T(0)) {
    return RankKOperand<T>{c, alpha, beta};
}

template <typename T>
inline constexpr Rank1UpdateOperand<T> make_rank1_update_operand(const VectorView<T>& y,
                                                                 const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                 T alpha = T(1)) {
    return Rank1UpdateOperand<T>{y, a, alpha};
}

template <typename T>
inline constexpr GemvOperand<T> make_gemv_operand(const VectorView<T>& x,
                                                  const VectorView<T>& y,
                                                  T alpha = T(1),
                                                  T beta = T(0)) {
    return make_matvec_operand(x, y, alpha, beta);
}

namespace detail {

template <typename Item>
concept NdItemLike = requires(const Item& item) {
    item.get_group();
    item.get_sub_group();
    item.get_local_linear_id();
    item.get_local_range().size();
};

template <NdItemLike Item>
inline constexpr int item_local_linear_id(const Item& item) {
    return static_cast<int>(item.get_local_linear_id());
}

template <NdItemLike Item>
inline constexpr int item_local_linear_range(const Item& item) {
    return static_cast<int>(item.get_local_range().size());
}

template <typename T, typename Workspace>
inline constexpr std::size_t workspace_elements_v = sizeof(Workspace) / sizeof(T);

template <typename Workspace, typename T>
inline constexpr Workspace* workspace_ptr_cast(T* workspace) {
    static_assert(sizeof(Workspace) % sizeof(T) == 0);
    return reinterpret_cast<Workspace*>(workspace);
}

template <typename Group, typename T>
inline constexpr T reduce_sum_group(const Group& group, const T& value) {
    if constexpr (ComplexScalar<T>) {
        using Real = typename T::value_type;
        const Real re = sycl::reduce_over_group(group, value.real(), sycl::plus<Real>());
        const Real im = sycl::reduce_over_group(group, value.imag(), sycl::plus<Real>());
        return T(re, im);
    } else {
        return sycl::reduce_over_group(group, value, sycl::plus<T>());
    }
}

template <NdItemLike Item, typename T>
inline constexpr T reduce_sum_group(const Item& item, const T& value) {
    return reduce_sum_group(item.get_group(), value);
}

template <typename T>
inline constexpr T conj(const T& value) {
    if constexpr (ComplexScalar<T>) {
        return std::conj(value);
    } else {
        return value;
    }
}

template <typename Group>
inline constexpr int group_local_linear_id(const Group& group) {
    return static_cast<int>(group.get_local_linear_id());
}

template <NdItemLike Item>
inline constexpr int group_local_linear_id(const Item& item) {
    return item_local_linear_id(item);
}

template <typename Group>
inline constexpr int group_local_linear_range(const Group& group) {
    return static_cast<int>(group.get_local_range().size());
}

template <NdItemLike Item>
inline constexpr int group_local_linear_range(const Item& item) {
    return item_local_linear_range(item);
}

inline constexpr int group_local_linear_range(const DeviceBlasLaunchInfo& launch) {
    return std::max(1, launch.local_size);
}

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

template <typename T>
inline constexpr T maybe_conjugate(const T& value, bool conjugate) {
    return conjugate ? conj(value) : value;
}

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

template <typename T>
inline constexpr int input_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.cols() : a.rows();
}

inline constexpr int input_size(int rows, int cols, Transpose trans) {
    return trans == Transpose::NoTrans ? cols : rows;
}

template <Transpose TransV, typename T>
inline constexpr int input_size(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    if constexpr (TransV == Transpose::NoTrans) {
        return a.cols();
    } else {
        return a.rows();
    }
}

template <typename T>
inline constexpr int output_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.rows() : a.cols();
}

inline constexpr int output_size(int rows, int cols, Transpose trans) {
    return trans == Transpose::NoTrans ? rows : cols;
}

template <Transpose TransV, typename T>
inline constexpr int output_size(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    if constexpr (TransV == Transpose::NoTrans) {
        return a.rows();
    } else {
        return a.cols();
    }
}

template <typename T>
inline constexpr void validate_single_problem(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                              Transpose trans) {
    (void)trans;
    assert(a.batch_size() == 1 && "device BLAS expects a single logical matrix; pass batch_item() for batched data");
}

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

template <typename T>
inline constexpr void validate_rank1_operand(const VectorView<T>& x,
                                             const Rank1UpdateOperand<T>& operand) {
    assert(x.batch_size() == 1 && "device::ger expects a single logical input vector; pass batch_item() for batched data");
    assert(operand.y.batch_size() == 1 && "device::ger expects a single logical input vector; pass batch_item() for batched data");
    validate_single_problem(operand.a, Transpose::NoTrans);
    assert(operand.a.rows() == x.size() && "device::ger matrix row count must match x size");
    assert(operand.a.cols() == operand.y.size() && "device::ger matrix column count must match y size");
}

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

template <typename T>
inline constexpr void validate_vector_operand(const VectorView<T>& x,
                                              const char* op_name) {
    (void)op_name;
    assert(x.batch_size() == 1 && "device vector BLAS expects a single logical vector; pass batch_item() for batched data");
}

template <typename T>
inline constexpr void validate_vector_operands(const VectorView<T>& x,
                                               const VectorView<T>& y,
                                               const char* op_name) {
    validate_vector_operand(x, op_name);
    validate_vector_operand(y, op_name);
    assert(x.size() == y.size() && "device vector BLAS operands must have matching sizes");
}

template <typename T>
inline constexpr void validate_vector_operands(const VectorView<T>& x,
                                               const VectorView<T>& y,
                                               const VectorView<T>& z,
                                               const char* op_name) {
    validate_vector_operands(x, y, op_name);
    validate_vector_operand(z, op_name);
    assert(x.size() == z.size() && "device vector BLAS operands must have matching sizes");
}

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

template <typename Group, typename T, std::size_t... I>
inline constexpr void reduce_partials_impl(const Group& group,
                                           std::array<T, sizeof...(I)>& values,
                                           std::index_sequence<I...>) {
    ((values[I] = reduce_sum_group(group, values[I])), ...);
}

template <typename Group, typename T, typename... Ops>
inline constexpr void reduce_partials(const Group& group,
                                      std::array<T, sizeof...(Ops)>& values,
                                      const std::tuple<Ops...>&) {
    reduce_partials_impl(group, values, std::index_sequence_for<Ops...>{});
}

template <typename T, typename Tuple, std::size_t... I>
inline constexpr void validate_operands_impl(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                             const Tuple& operands,
                                             Transpose trans,
                                             std::index_sequence<I...>) {
    (validate_operand(a, std::get<I>(operands), trans), ...);
}

template <typename T, typename... Ops>
inline constexpr void validate_operands(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                        const std::tuple<Ops...>& operands,
                                        Transpose trans) {
    validate_operands_impl(a, operands, trans, std::index_sequence_for<Ops...>{});
}

template <typename T, typename Tuple, std::size_t... I>
inline constexpr void validate_triangular_operands_impl(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                        const Tuple& operands,
                                                        TriangularTransform transform,
                                                        std::index_sequence<I...>) {
    (validate_triangular_operand(a, std::get<I>(operands), transform), ...);
}

template <typename T, typename... Ops>
inline constexpr void validate_triangular_operands(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                   const std::tuple<Ops...>& operands,
                                                   TriangularTransform transform) {
    validate_triangular_operands_impl(a, operands, transform, std::index_sequence_for<Ops...>{});
}

template <typename T, typename Tuple, std::size_t... I>
inline constexpr void validate_symmetric_operands_impl(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                       const Tuple& operands,
                                                       SymmetricTransform transform,
                                                       std::index_sequence<I...>) {
    (validate_symmetric_operand(a, std::get<I>(operands), transform), ...);
}

template <typename T, typename... Ops>
inline constexpr void validate_symmetric_operands(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                  const std::tuple<Ops...>& operands,
                                                  SymmetricTransform transform) {
    validate_symmetric_operands_impl(a, operands, transform, std::index_sequence_for<Ops...>{});
}

template <typename T, typename Tuple, std::size_t... I>
inline constexpr void accumulate_impl(std::array<T, sizeof...(I)>& partials,
                                      const Tuple& operands,
                                      const T& a_ij,
                                      int input_index,
                                      std::index_sequence<I...>) {
    ((partials[I] += a_ij * std::get<I>(operands).x(input_index)), ...);
}

template <typename T, typename... Ops>
inline constexpr void accumulate(std::array<T, sizeof...(Ops)>& partials,
                                 const std::tuple<Ops...>& operands,
                                 const T& a_ij,
                                 int input_index) {
    accumulate_impl(partials, operands, a_ij, input_index, std::index_sequence_for<Ops...>{});
}

template <typename T, typename Tuple, std::size_t... I>
inline constexpr void write_outputs_impl(const Tuple& operands,
                                         int output_index,
                                         const std::array<T, sizeof...(I)>& values,
                                         std::index_sequence<I...>) {
    ((std::get<I>(operands).y(output_index) =
          std::get<I>(operands).alpha * values[I] + std::get<I>(operands).beta * std::get<I>(operands).y(output_index)),
     ...);
}

template <typename T, typename... Ops>
inline constexpr void write_outputs(const std::tuple<Ops...>& operands,
                                    int output_index,
                                    const std::array<T, sizeof...(Ops)>& values) {
    write_outputs_impl(operands, output_index, values, std::index_sequence_for<Ops...>{});
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

template <typename Tag>
inline constexpr bool triangular_storage_contains(int row, int col) {
    if constexpr (Tag::uplo == Uplo::Lower) {
        return row >= col;
    }
    return row <= col;
}

inline constexpr bool triangular_storage_contains(Uplo uplo, int row, int col) {
    return uplo == Uplo::Lower ? row >= col : row <= col;
}

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

template <typename T>
inline constexpr void write_matrix_output(const MatrixMatrixOperand<T>& operand,
                                          int row,
                                          int col,
                                          const T& value) {
    operand.c(row, col) = operand.alpha * value + operand.beta * operand.c(row, col);
}

template <typename T>
inline constexpr void accumulate_rank1_output(const Rank1UpdateOperand<T>& operand,
                                              int row,
                                              int col,
                                              const T& value) {
    operand.a(row, col) += operand.alpha * value;
}

template <typename Tag>
inline constexpr int triangular_begin(int index, int extent) {
    if constexpr (Tag::side == Side::Left) {
        if constexpr (Tag::trans == Transpose::NoTrans) {
            return Tag::uplo == Uplo::Lower ? 0 : index;
        }
        return Tag::uplo == Uplo::Lower ? index : 0;
    }

    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::uplo == Uplo::Lower ? index : 0;
    }
    return Tag::uplo == Uplo::Lower ? 0 : index;
}

template <typename Tag>
inline constexpr int triangular_end(int index, int extent) {
    if constexpr (Tag::side == Side::Left) {
        if constexpr (Tag::trans == Transpose::NoTrans) {
            return Tag::uplo == Uplo::Lower ? index + 1 : extent;
        }
        return Tag::uplo == Uplo::Lower ? extent : index + 1;
    }

    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::uplo == Uplo::Lower ? extent : index + 1;
    }
    return Tag::uplo == Uplo::Lower ? index + 1 : extent;
}

template <typename Tag>
inline constexpr Transpose rank2k_rhs_transform() {
    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

inline constexpr Transpose rank2k_rhs_transform(SymmetricRank2kTransform transform) {
    if (transform.trans == Transpose::NoTrans) {
        return transform.hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

template <typename Tag>
inline constexpr Transpose rankk_rhs_transform() {
    if constexpr (Tag::trans == Transpose::NoTrans) {
        return Tag::hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

inline constexpr Transpose rankk_rhs_transform(SymmetricRankKTransform transform) {
    if (transform.trans == Transpose::NoTrans) {
        return transform.hermitian ? Transpose::ConjTrans : Transpose::Trans;
    }
    return Transpose::NoTrans;
}

template <typename Tag, typename T>
inline constexpr T secondary_rank2k_alpha(const T& alpha) {
    if constexpr (Tag::hermitian) {
        return conj(alpha);
    }
    return alpha;
}

template <typename T>
inline constexpr T secondary_rank2k_alpha(const T& alpha, bool hermitian) {
    return hermitian ? conj(alpha) : alpha;
}

template <typename Fn>
inline constexpr decltype(auto) dispatch_triangular_transform(TriangularTransform transform, Fn&& fn) {
    switch (transform.side) {
    case Side::Left:
        switch (transform.uplo) {
        case Uplo::Lower:
            switch (transform.trans) {
            case Transpose::NoTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>{});
            case Transpose::Trans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::Trans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::Trans, Diag::NonUnit>{});
            case Transpose::ConjTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::ConjTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit>{});
            }
            break;
        case Uplo::Upper:
            switch (transform.trans) {
            case Transpose::NoTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>{});
            case Transpose::Trans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::Trans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::Trans, Diag::NonUnit>{});
            case Transpose::ConjTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit>{});
            }
            break;
        }
        break;
    case Side::Right:
        switch (transform.uplo) {
        case Uplo::Lower:
            switch (transform.trans) {
            case Transpose::NoTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit>{});
            case Transpose::Trans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::Trans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::Trans, Diag::NonUnit>{});
            case Transpose::ConjTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit>{});
            }
            break;
        case Uplo::Upper:
            switch (transform.trans) {
            case Transpose::NoTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>{});
            case Transpose::Trans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::Trans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::Trans, Diag::NonUnit>{});
            case Transpose::ConjTrans:
                return transform.diag == Diag::Unit
                    ? std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::ConjTrans, Diag::Unit>{})
                    : std::forward<Fn>(fn)(TriangularTransformTag<Side::Right, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit>{});
            }
            break;
        }
        break;
    }
    assert(false && "unreachable triangular transform");
    return std::forward<Fn>(fn)(TriangularTransformTag<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit>{});
}

template <typename Fn>
inline constexpr decltype(auto) dispatch_symmetric_transform(SymmetricTransform transform, Fn&& fn) {
    switch (transform.side) {
    case Side::Left:
        switch (transform.uplo) {
        case Uplo::Lower:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricTransformTag<Side::Left, Uplo::Lower, true>{})
                : std::forward<Fn>(fn)(SymmetricTransformTag<Side::Left, Uplo::Lower, false>{});
        case Uplo::Upper:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricTransformTag<Side::Left, Uplo::Upper, true>{})
                : std::forward<Fn>(fn)(SymmetricTransformTag<Side::Left, Uplo::Upper, false>{});
        }
        break;
    case Side::Right:
        switch (transform.uplo) {
        case Uplo::Lower:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricTransformTag<Side::Right, Uplo::Lower, true>{})
                : std::forward<Fn>(fn)(SymmetricTransformTag<Side::Right, Uplo::Lower, false>{});
        case Uplo::Upper:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricTransformTag<Side::Right, Uplo::Upper, true>{})
                : std::forward<Fn>(fn)(SymmetricTransformTag<Side::Right, Uplo::Upper, false>{});
        }
        break;
    }
    assert(false && "unreachable symmetric transform");
    return std::forward<Fn>(fn)(SymmetricTransformTag<Side::Left, Uplo::Upper, false>{});
}

template <typename Fn>
inline constexpr decltype(auto) dispatch_rank_transform(SymmetricRank2kTransform transform, Fn&& fn) {
    switch (transform.uplo) {
    case Uplo::Lower:
        switch (transform.trans) {
        case Transpose::NoTrans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::NoTrans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::NoTrans, false>{});
        case Transpose::Trans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::Trans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::Trans, false>{});
        case Transpose::ConjTrans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::ConjTrans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Lower, Transpose::ConjTrans, false>{});
        }
        break;
    case Uplo::Upper:
        switch (transform.trans) {
        case Transpose::NoTrans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::NoTrans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::NoTrans, false>{});
        case Transpose::Trans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::Trans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::Trans, false>{});
        case Transpose::ConjTrans:
            return transform.hermitian
                ? std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::ConjTrans, true>{})
                : std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::ConjTrans, false>{});
        }
        break;
    }
    assert(false && "unreachable rank transform");
    return std::forward<Fn>(fn)(SymmetricRankTransformTag<Uplo::Upper, Transpose::NoTrans, false>{});
}

template <typename Fn>
inline constexpr decltype(auto) dispatch_rank_transform(SymmetricRankKTransform transform, Fn&& fn) {
    return dispatch_rank_transform(SymmetricRank2kTransform{.uplo = transform.uplo, .trans = transform.trans, .hermitian = transform.hermitian},
                                   std::forward<Fn>(fn));
}

template <typename T>
inline constexpr SymmetricTransform canonical_hermitian_transform(SymmetricTransform transform) {
    return SymmetricTransform{
        .side = transform.side,
        .uplo = transform.uplo,
        .hermitian = ComplexScalar<T>,
    };
}

template <typename T>
inline constexpr SymmetricRank2kTransform canonical_hermitian_transform(SymmetricRank2kTransform transform) {
    return SymmetricRank2kTransform{
        .uplo = transform.uplo,
        .trans = transform.trans,
        .hermitian = ComplexScalar<T>,
    };
}

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