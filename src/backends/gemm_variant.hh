#pragma once

#include "../linalg-impl.hh"

#include <complex>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace batchlas::backend {

enum class GemmVariantRequest {
    Vendor,
    Sycl,
    Auto,
};

inline GemmVariantRequest gemm_variant_request() {
    const char* raw = std::getenv("BATCHLAS_GEMM_VARIANT");
    if (!raw) {
        return GemmVariantRequest::Vendor;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "sycl" || value == "custom") {
        return GemmVariantRequest::Sycl;
    }
    if (value == "auto") {
        return GemmVariantRequest::Auto;
    }
    return GemmVariantRequest::Vendor;
}

template <typename T>
inline bool gemm_sycl_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Transpose transA,
                                Transpose transB,
                                ComputePrecision precision) {
    if (precision != ComputePrecision::Default) {
        return false;
    }

    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    if (k != k_b) {
        return false;
    }

    return m == C.rows() && n == C.cols() && m > 0 && n > 0 && k > 0;
}

template <typename T>
inline bool gemm_use_sycl_custom(const Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 Transpose transA,
                                 Transpose transB,
                                 ComputePrecision precision) {
    const auto request = gemm_variant_request();
    if (request == GemmVariantRequest::Vendor) {
        return false;
    }

    if (!gemm_sycl_supported(A, B, C, transA, transB, precision)) {
        return false;
    }

    if (request == GemmVariantRequest::Sycl) {
        return true;
    }

    if (ctx.device().type != DeviceType::GPU) {
        return false;
    }

    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return false;
    }

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    const int max_dim = std::max({m, n, k});
    if (m != n || n != k || A.batch_size() < 64) {
        return false;
    }

    if constexpr (std::is_same_v<T, float>) {
        if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
            if (transA == Transpose::ConjTrans || transB == Transpose::ConjTrans) {
                return false;
            }
            return A.batch_size() >= 128 && max_dim >= 128 && max_dim <= 512;
        }
        if (max_dim <= 32) {
            return true;
        }
        return max_dim >= 128 && max_dim <= 512;
    }

    if constexpr (std::is_same_v<T, double>) {
        return max_dim <= 512;
    }

    return false;
}

} // namespace batchlas::backend