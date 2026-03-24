#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

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

struct MatrixVectorTransform {
    Transpose trans = Transpose::NoTrans;
};

struct GeneralMatrixTransform {
    Transpose trans_a = Transpose::NoTrans;
    Transpose trans_b = Transpose::NoTrans;
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
inline constexpr GemvOperand<T> make_gemv_operand(const VectorView<T>& x,
                                                  const VectorView<T>& y,
                                                  T alpha = T(1),
                                                  T beta = T(0)) {
    return make_matvec_operand(x, y, alpha, beta);
}

namespace detail {

template <typename T>
inline constexpr T conjugate_if_needed(const T& value) {
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

template <typename Group>
inline constexpr int group_local_linear_range(const Group& group) {
    return static_cast<int>(group.get_local_range().size());
}

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

template <typename T, typename Op>
concept MatrixVectorOperandFor = std::same_as<std::remove_cvref_t<Op>, MatrixVectorOperand<T>>;

template <typename T, typename Op>
concept MatrixMatrixOperandFor = std::same_as<std::remove_cvref_t<Op>, MatrixMatrixOperand<T>>;

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
    return conjugate_if_needed(a(col, row));
}

template <typename T>
inline constexpr int input_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.cols() : a.rows();
}

template <typename T>
inline constexpr int output_size(const KernelMatrixView<T, MatrixFormat::Dense>& a, Transpose trans) {
    return trans == Transpose::NoTrans ? a.rows() : a.cols();
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

template <typename Group, typename T, std::size_t... I>
inline constexpr void reduce_partials_impl(const Group& group,
                                           std::array<T, sizeof...(I)>& values,
                                           std::index_sequence<I...>) {
    ((values[I] = sycl::reduce_over_group(group, values[I], sycl::plus<T>())), ...);
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

inline constexpr bool triangular_storage_contains(Uplo uplo, int row, int col) {
    return uplo == Uplo::Lower ? row >= col : row <= col;
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
        value = conjugate_if_needed(value);
    }
    return value;
}

template <typename T>
inline constexpr T symmetric_matrix_entry(const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          int row,
                                          int col,
                                          SymmetricTransform transform) {
    if (transform.uplo == Uplo::Lower) {
        return row >= col ? a(row, col) : a(col, row);
    }
    return row <= col ? a(row, col) : a(col, row);
}

template <typename T>
inline constexpr void write_matrix_output(const MatrixMatrixOperand<T>& operand,
                                          int row,
                                          int col,
                                          const T& value) {
    operand.c(row, col) = operand.alpha * value + operand.beta * operand.c(row, col);
}

} // namespace detail

#include <blas/device/detail/group_blas_generic.hh>
#include <blas/device/detail/group_blas_subgroup.hh>

template <typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           GeneralMatrixTransform transform) {
    detail::validate_gemm_operand(a, operand, transform);
    detail::generic::gemm(group, a, operand, transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void gemm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           GeneralMatrixTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    detail::validate_gemm_operand(a, operand, transform);
    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);
    auto* gemm_workspace =
        sycl::ext::oneapi::group_local_memory_for_overwrite<detail::subgroup::GemmWorkspace<T>>(item.get_group()).get();
    if (detail::subgroup::can_use_matrix_aligned_nn_large_fast_path<T>(item, a, operand, transform, policy)) {
        detail::subgroup::gemm_aligned_nn_large(item, a, operand, gemm_workspace);
        return;
    }
    if (detail::subgroup::can_use_matrix_register_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::gemm_register_tiled(item, a, operand, transform, gemm_workspace);
        return;
    }
    if (detail::subgroup::can_use_matrix_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::gemm(item, a, operand, transform);
        return;
    }
    detail::generic::gemm(item.get_group(), a, operand, transform);
}

template <typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           GeneralMatrixTransform transform) {
    gemm(group, a, make_matmat_operand(b, c, alpha, beta), transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void gemm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           GeneralMatrixTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    gemm(item, a, make_matmat_operand(b, c, alpha, beta), transform, policy);
}

template <typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Transpose trans_a = Transpose::NoTrans,
                           Transpose trans_b = Transpose::NoTrans) {
    gemm(group, a, b, c, alpha, beta, GeneralMatrixTransform{.trans_a = trans_a, .trans_b = trans_b});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void gemm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Transpose trans_a = Transpose::NoTrans,
                           Transpose trans_b = Transpose::NoTrans,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    gemm(item,
         a,
         b,
         c,
         alpha,
         beta,
         GeneralMatrixTransform{.trans_a = trans_a, .trans_b = trans_b},
         policy);
}

template <typename Group, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixVectorTransform transform,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_operands(a, operands, transform.trans);
    detail::generic::gemxv(group, a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixVectorTransform transform,
                            DeviceBlasPolicy policy,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_operands(a, operands, transform.trans);
    if constexpr (sizeof...(Ops) == 1) {
        const auto& operand = std::get<0>(operands);
        if (transform.trans == Transpose::NoTrans &&
            detail::subgroup::can_use_dense_gemv_no_trans_fast_path<T>(item, a, policy)) {
            detail::subgroup::gemv_dense_no_trans(item, a, operand);
            return;
        }
    }
    if (detail::subgroup::can_use_vector_fast_path<T, Ops...>(item, a, transform, policy)) {
        detail::subgroup::gemxv(item, a, transform, operands);
        return;
    }
    detail::generic::gemxv(item.get_group(), a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixVectorTransform transform,
                            const Ops&... ops) {
    gemxv(item, a, transform, DeviceBlasPolicy::Auto, ops...);
}

template <typename Group, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            Transpose trans,
                            const Ops&... ops) {
    gemxv(group, a, MatrixVectorTransform{.trans = trans}, ops...);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            Transpose trans,
                            DeviceBlasPolicy policy,
                            const Ops&... ops) {
    gemxv(item, a, MatrixVectorTransform{.trans = trans}, policy, ops...);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void gemxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            Transpose trans,
                            const Ops&... ops) {
    gemxv(item, a, MatrixVectorTransform{.trans = trans}, DeviceBlasPolicy::Auto, ops...);
}

template <typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           MatrixVectorTransform transform) {
    gemxv(group, a, transform, make_matvec_operand(x, y, alpha, beta));
}

template <detail::NdItemLike Item, typename T>
inline constexpr void gemv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           MatrixVectorTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    gemxv(item, a, transform, policy, make_matvec_operand(x, y, alpha, beta));
}

template <typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Transpose trans = Transpose::NoTrans) {
    gemv(group, a, x, y, alpha, beta, MatrixVectorTransform{.trans = trans});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void gemv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Transpose trans = Transpose::NoTrans,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    gemv(item, a, x, y, alpha, beta, MatrixVectorTransform{.trans = trans}, policy);
}

template <typename Group, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void trmxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            TriangularTransform transform,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_triangular_operands(a, operands, transform);
    detail::generic::trmxv(group, a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void trmxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            TriangularTransform transform,
                            DeviceBlasPolicy policy,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_triangular_operands(a, operands, transform);
    if constexpr (sizeof...(Ops) == 1) {
        const auto& operand = std::get<0>(operands);
        if (detail::subgroup::can_use_trmv_no_trans_column_sweep_fast_path<T>(item, a, transform, policy)) {
            detail::subgroup::trmv_no_trans_column_sweep(item, a, operand, transform);
            return;
        }
        if (detail::subgroup::can_use_trmv_transpose_subgroup_dot_fast_path<T>(item, a, transform, policy)) {
            detail::subgroup::trmv_transpose_subgroup_dots(item, a, operand, transform);
            return;
        }
    }
    if (detail::subgroup::can_use_vector_fast_path<T, Ops...>(item, a, MatrixVectorTransform{.trans = transform.trans}, policy)) {
        detail::subgroup::trmxv(item, a, transform, operands);
        return;
    }
    detail::generic::trmxv(item.get_group(), a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void trmxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            TriangularTransform transform,
                            const Ops&... ops) {
    trmxv(item, a, transform, DeviceBlasPolicy::Auto, ops...);
}

template <typename Group, typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           TriangularTransform transform) {
    trmxv(group, a, transform, make_matvec_operand(x, y, alpha, beta));
}

template <detail::NdItemLike Item, typename T>
inline constexpr void trmv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           TriangularTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    trmxv(item, a, transform, policy, make_matvec_operand(x, y, alpha, beta));
}

template <typename Group, typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Uplo uplo = Uplo::Upper,
                           Transpose trans = Transpose::NoTrans,
                           Diag diag = Diag::NonUnit) {
    trmv(group,
         a,
         x,
         y,
         alpha,
         beta,
         TriangularTransform{.side = Side::Left, .uplo = uplo, .trans = trans, .diag = diag});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void trmv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Uplo uplo = Uplo::Upper,
                           Transpose trans = Transpose::NoTrans,
                           Diag diag = Diag::NonUnit,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    trmv(item,
         a,
         x,
         y,
         alpha,
         beta,
         TriangularTransform{.side = Side::Left, .uplo = uplo, .trans = trans, .diag = diag},
         policy);
}

template <typename Group, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void symxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            SymmetricTransform transform,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_symmetric_operands(a, operands, transform);
    detail::generic::symxv(group, a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void symxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            SymmetricTransform transform,
                            DeviceBlasPolicy policy,
                            const Ops&... ops) {
    const auto operands = std::forward_as_tuple(ops...);
    detail::validate_symmetric_operands(a, operands, transform);
    if constexpr (sizeof...(Ops) == 1) {
        const auto& operand = std::get<0>(operands);
        if (detail::subgroup::can_use_symv_no_trans_column_sweep_fast_path<T>(item, a, policy)) {
            detail::subgroup::symv_no_trans_column_sweep(item, a, operand, transform);
            return;
        }
    }
    if (detail::subgroup::can_use_vector_fast_path<T, Ops...>(item, a, MatrixVectorTransform{.trans = Transpose::NoTrans}, policy)) {
        detail::subgroup::symxv(item, a, transform, operands);
        return;
    }
    detail::generic::symxv(item.get_group(), a, transform, operands);
}

template <detail::NdItemLike Item, typename T, typename... Ops>
    requires(sizeof...(Ops) > 0 && (detail::MatrixVectorOperandFor<T, Ops> && ...))
inline constexpr void symxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            SymmetricTransform transform,
                            const Ops&... ops) {
    symxv(item, a, transform, DeviceBlasPolicy::Auto, ops...);
}

template <typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           SymmetricTransform transform) {
    symxv(group, a, transform, make_matvec_operand(x, y, alpha, beta));
}

template <detail::NdItemLike Item, typename T>
inline constexpr void symv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha,
                           T beta,
                           SymmetricTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    symxv(item, a, transform, policy, make_matvec_operand(x, y, alpha, beta));
}

template <typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Uplo uplo = Uplo::Upper) {
    symv(group, a, x, y, alpha, beta, SymmetricTransform{.side = Side::Left, .uplo = uplo});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void symv(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           Uplo uplo = Uplo::Upper,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    symv(item, a, x, y, alpha, beta, SymmetricTransform{.side = Side::Left, .uplo = uplo}, policy);
}

template <typename Group, typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           TriangularTransform transform) {
    detail::validate_triangular_operand(a, operand, transform);
    detail::generic::trmm(group, a, operand, transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void trmm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           TriangularTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    detail::validate_triangular_operand(a, operand, transform);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
    if (detail::subgroup::can_use_matrix_register_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::trmm_register_tiled(item, a, operand, transform);
        return;
    }
    if (detail::subgroup::can_use_matrix_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::trmm(item, a, operand, transform);
        return;
    }
    detail::generic::trmm(item.get_group(), a, operand, transform);
}

template <typename Group, typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           TriangularTransform transform) {
    trmm(group, a, make_matmat_operand(b, c, alpha, beta), transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void trmm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           TriangularTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    trmm(item, a, make_matmat_operand(b, c, alpha, beta), transform, policy);
}

template <typename Group, typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Side side = Side::Left,
                           Uplo uplo = Uplo::Upper,
                           Transpose trans = Transpose::NoTrans,
                           Diag diag = Diag::NonUnit) {
    trmm(group, a, b, c, alpha, beta, TriangularTransform{.side = side, .uplo = uplo, .trans = trans, .diag = diag});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void trmm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Side side = Side::Left,
                           Uplo uplo = Uplo::Upper,
                           Transpose trans = Transpose::NoTrans,
                           Diag diag = Diag::NonUnit,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    trmm(item,
         a,
         b,
         c,
         alpha,
         beta,
         TriangularTransform{.side = side, .uplo = uplo, .trans = trans, .diag = diag},
         policy);
}

template <typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           SymmetricTransform transform) {
    detail::validate_symmetric_operand(a, operand, transform);
    detail::generic::symm(group, a, operand, transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void symm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           SymmetricTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    detail::validate_symmetric_operand(a, operand, transform);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
    if (detail::subgroup::can_use_matrix_register_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::symm_register_tiled(item, a, operand, transform);
        return;
    }
    if (detail::subgroup::can_use_matrix_fast_path<T>(item, row_extent, col_extent, contract_extent, policy)) {
        detail::subgroup::symm(item, a, operand, transform);
        return;
    }
    detail::generic::symm(item.get_group(), a, operand, transform);
}

template <typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           SymmetricTransform transform) {
    symm(group, a, make_matmat_operand(b, c, alpha, beta), transform);
}

template <detail::NdItemLike Item, typename T>
inline constexpr void symm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha,
                           T beta,
                           SymmetricTransform transform,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    symm(item, a, make_matmat_operand(b, c, alpha, beta), transform, policy);
}

template <typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Side side = Side::Left,
                           Uplo uplo = Uplo::Upper) {
    symm(group, a, b, c, alpha, beta, SymmetricTransform{.side = side, .uplo = uplo});
}

template <detail::NdItemLike Item, typename T>
inline constexpr void symm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           Side side = Side::Left,
                           Uplo uplo = Uplo::Upper,
                           DeviceBlasPolicy policy = DeviceBlasPolicy::Auto) {
    symm(item, a, b, c, alpha, beta, SymmetricTransform{.side = side, .uplo = uplo}, policy);
}

} // namespace batchlas::device
