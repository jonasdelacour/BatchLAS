namespace detail::generic {

template <typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           GeneralMatrixTransform transform) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);

    for (int col = 0; col < col_extent; ++col) {
        for (int row = 0; row < row_extent; ++row) {
            T partial{};

            for (int k = local_id; k < contract_extent; k += local_size) {
                partial += detail::matrix_entry(a, row, k, transform.trans_a) *
                    detail::matrix_entry(operand.b, k, col, transform.trans_b);
            }

            partial = sycl::reduce_over_group(group, partial, sycl::plus<T>());
            if (group.leader()) {
                detail::write_matrix_output(operand, row, col, partial);
            }
        }
    }
}

template <typename Group, typename T, typename... Ops>
inline constexpr void gemxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixVectorTransform transform,
                            const std::tuple<Ops...>& operands) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int inner_extent = detail::input_size(a, transform.trans);
    const int outer_extent = detail::output_size(a, transform.trans);

    for (int output_index = 0; output_index < outer_extent; ++output_index) {
        std::array<T, sizeof...(Ops)> partials{};

        for (int input_index = local_id; input_index < inner_extent; input_index += local_size) {
            const T a_ij = detail::matrix_entry(a, output_index, input_index, transform.trans);
            detail::accumulate(partials, operands, a_ij, input_index);
        }

        detail::reduce_partials(group, partials, operands);
        if (group.leader()) {
            detail::write_outputs(operands, output_index, partials);
        }
    }
}

template <typename Group, typename T, typename... Ops>
inline constexpr void trmxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            TriangularTransform transform,
                            const std::tuple<Ops...>& operands) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows();

    for (int output_index = 0; output_index < extent; ++output_index) {
        std::array<T, sizeof...(Ops)> partials{};

        for (int input_index = local_id; input_index < extent; input_index += local_size) {
            const T a_ij = detail::triangular_matrix_entry(a, output_index, input_index, transform);
            detail::accumulate(partials, operands, a_ij, input_index);
        }

        detail::reduce_partials(group, partials, operands);
        if (group.leader()) {
            detail::write_outputs(operands, output_index, partials);
        }
    }
}

template <typename Group, typename T, typename... Ops>
inline constexpr void symxv(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            SymmetricTransform transform,
                            const std::tuple<Ops...>& operands) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows();

    for (int output_index = 0; output_index < extent; ++output_index) {
        std::array<T, sizeof...(Ops)> partials{};

        for (int input_index = local_id; input_index < extent; input_index += local_size) {
            const T a_ij = detail::symmetric_matrix_entry(a, output_index, input_index, transform);
            detail::accumulate(partials, operands, a_ij, input_index);
        }

        detail::reduce_partials(group, partials, operands);
        if (group.leader()) {
            detail::write_outputs(operands, output_index, partials);
        }
    }
}

template <typename Group, typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           TriangularTransform transform) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();

    for (int col = 0; col < col_extent; ++col) {
        for (int row = 0; row < row_extent; ++row) {
            T partial{};

            for (int k = local_id; k < contract_extent; k += local_size) {
                if (transform.side == Side::Left) {
                    partial += detail::triangular_matrix_entry(a, row, k, transform) * operand.b(k, col);
                } else {
                    partial += operand.b(row, k) * detail::triangular_matrix_entry(a, k, col, transform);
                }
            }

            partial = sycl::reduce_over_group(group, partial, sycl::plus<T>());
            if (group.leader()) {
                detail::write_matrix_output(operand, row, col, partial);
            }
        }
    }
}

template <typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           SymmetricTransform transform) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();

    for (int col = 0; col < col_extent; ++col) {
        for (int row = 0; row < row_extent; ++row) {
            T partial{};

            for (int k = local_id; k < contract_extent; k += local_size) {
                if (transform.side == Side::Left) {
                    partial += detail::symmetric_matrix_entry(a, row, k, transform) * operand.b(k, col);
                } else {
                    partial += operand.b(row, k) * detail::symmetric_matrix_entry(a, k, col, transform);
                }
            }

            partial = sycl::reduce_over_group(group, partial, sycl::plus<T>());
            if (group.leader()) {
                detail::write_matrix_output(operand, row, col, partial);
            }
        }
    }
}

} // namespace detail::generic
