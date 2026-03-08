#pragma once

#include "../gemm_kernels.hh"

#include <complex>

namespace batchlas::sycl_gemm {

template <typename T>
inline T maybe_conj(const T& value, bool conjugate) {
    static_cast<void>(conjugate);
    return value;
}

template <typename T>
inline std::complex<T> maybe_conj(const std::complex<T>& value, bool conjugate) {
    return conjugate ? std::conj(value) : value;
}

template <typename T>
inline T operand_value(const T* ptr,
                       int ld,
                       int batch_offset,
                       int row,
                       int col,
                       Transpose trans) {
    const bool transpose = trans != Transpose::NoTrans;
    const bool conjugate = trans == Transpose::ConjTrans;
    const int source_row = transpose ? col : row;
    const int source_col = transpose ? row : col;
    return maybe_conj(ptr[batch_offset + source_col * ld + source_row], conjugate);
}

template <typename T, Transpose Op>
struct OperandAccessor {
    static T load(const T* ptr, int ld, int batch_offset, int row, int col) {
        return operand_value(ptr, ld, batch_offset, row, col, Op);
    }
};

} // namespace batchlas::sycl_gemm