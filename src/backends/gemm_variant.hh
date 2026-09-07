#pragma once

#include "../linalg-impl.hh"

#include <complex>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace batchlas::backend {

template <typename T>
inline bool gemm_has_heterogeneous_batch(const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         const MatrixView<T, MatrixFormat::Dense>& C) {
    return A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous();
}

template <typename T>
inline bool gemm_batch_dimensions_compatible(const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Transpose transA,
                                             Transpose transB) {
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        if (k != k_b) {
            return false;
        }
        if (C.rows(batch_index) != m || C.cols(batch_index) != n) {
            return false;
        }
        if (m < 0 || n < 0 || k < 0) {
            return false;
        }
    }

    return true;
}

// The host loop a batch whose members do not share a shape falls back to.
//
// No vendor GEMM -- strided-batched or pointer-batched -- can be handed such a
// batch at all, because every one of them takes a single m/n/k for the whole
// call, so the batch becomes one single-matrix GEMM per member. cuBLAS, rocBLAS
// and oneMKL each wrote that loop out; what actually differs between them is
// only two things, and both are parameters here rather than something this
// helper decides:
//
//   gemm_one   -- which single-matrix GEMM to issue. The backends do not agree:
//                 cuBLAS calls its vendor_impl directly (recursing through
//                 gemm_vendor would re-run the route selection per member),
//                 while rocBLAS and MKL recurse into gemm_vendor on purpose so
//                 that a member can still reach the SYCL kernel.
//   on_empty   -- which Event a batch that launched nothing hands back. cuBLAS
//                 and rocBLAS fabricate one with
//                 create_event_after_external_work() because their work leaves
//                 the SYCL queue; MKL, whose GEMM is submitted to the queue,
//                 hands back the queue's own get_event(). Unifying the two
//                 would change what a caller may wait on, so it stays a
//                 per-backend decision.
//
// A member with m or n zero has nothing to compute and is skipped outright; a
// member with k zero is a pure scaling of C, which no GEMM spells, so it goes
// to scale(). Both are carried over verbatim from the three loops this
// replaces, including the "launched nothing at all" case they all guard.
template <typename T, typename GemmOne, typename OnEmpty>
inline Event gemm_over_heterogeneous_batch(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           const MatrixView<T, MatrixFormat::Dense>& B,
                                           const MatrixView<T, MatrixFormat::Dense>& C,
                                           T beta,
                                           Transpose transA,
                                           Transpose transB,
                                           GemmOne&& gemm_one,
                                           OnEmpty&& on_empty) {
    Event last_event;
    bool launched = false;
    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        static_cast<void>(k_b);
        if (m == 0 || n == 0) {
            continue;
        }
        if (k == 0) {
            last_event = scale(ctx, beta, C.batch_item(batch_index));
            launched = true;
            continue;
        }
        last_event = gemm_one(A.batch_item(batch_index),
                              B.batch_item(batch_index),
                              C.batch_item(batch_index));
        launched = true;
    }

    if (launched) {
        return std::move(last_event);
    }
    return on_empty();
}

enum class GemmVariantRequest {
    Vendor,
    Sycl,
    Native,
    CuBLASDx,
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
    if (value == "native" || value == "cuda-native" || value == "direct-cuda") {
        return GemmVariantRequest::Native;
    }
    if (value == "cublasdx" || value == "dx") {
        return GemmVariantRequest::CuBLASDx;
    }
    if (value == "auto") {
        return GemmVariantRequest::Auto;
    }
    return GemmVariantRequest::Vendor;
}

template <typename T>
inline bool gemm_custom_problem_supported(const MatrixView<T, MatrixFormat::Dense>& A,
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

    if (gemm_has_heterogeneous_batch(A, B, C)) {
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
inline bool gemm_sycl_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Transpose transA,
                                Transpose transB,
                                ComputePrecision precision) {
    return gemm_custom_problem_supported(A, B, C, transA, transB, precision);
}

template <typename T>
inline bool gemm_use_cublasdx_custom(const Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     const MatrixView<T, MatrixFormat::Dense>& C,
                                     Transpose transA,
                                     Transpose transB,
                                     ComputePrecision precision) {
    const auto request = gemm_variant_request();
    if (request != GemmVariantRequest::CuBLASDx) {
        return false;
    }

    if (ctx.device().type != DeviceType::GPU) {
        return false;
    }

    if (precision != ComputePrecision::Default) {
        return false;
    }

    return gemm_batch_dimensions_compatible(A, B, C, transA, transB);
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
    if (request == GemmVariantRequest::Vendor || request == GemmVariantRequest::Native ||
        request == GemmVariantRequest::CuBLASDx) {
        return false;
    }

    if (!gemm_custom_problem_supported(A, B, C, transA, transB, precision)) {
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