#pragma once

#define BATCHLAS_UNPAREN(...) __VA_ARGS__

#define BATCHLAS_FOR_EACH_REAL_TYPE(INVOKE) \
    INVOKE((float)) \
    INVOKE((double))

#define BATCHLAS_FOR_EACH_REAL_TYPE_1(INVOKE, arg1) \
    INVOKE(arg1, (float)) \
    INVOKE(arg1, (double))

#define BATCHLAS_FOR_EACH_SCALAR_TYPE(INVOKE) \
    BATCHLAS_FOR_EACH_REAL_TYPE(INVOKE) \
    INVOKE((std::complex<float>)) \
    INVOKE((std::complex<double>))

#define BATCHLAS_FOR_EACH_SCALAR_TYPE_1(INVOKE, arg1) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(INVOKE, arg1) \
    INVOKE(arg1, (std::complex<float>)) \
    INVOKE(arg1, (std::complex<double>))

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_1(INVOKE, arg1) \
    INVOKE(arg1, MatrixFormat::Dense) \
    INVOKE(arg1, MatrixFormat::CSR)

#define BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(INVOKE, arg1, arg2) \
    INVOKE(arg1, arg2, MatrixFormat::Dense) \
    INVOKE(arg1, arg2, MatrixFormat::CSR)