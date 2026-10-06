#pragma once

// Must stay includable from the vendor-free facade: portable headers only, never
// ../linalg-impl.hh (it reaches <cuda_runtime.h>).
// evidence: docs/perf/gemm.md#gemm-the-route-adapter-and-its-environment-readers
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_gemm.hh>

#include <complex>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>
#include <batchlas/settings.hh>

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

// Per-member host loop for a heterogeneous batch, with the empty-batch Event as
// a parameter (on_empty). Same skip / k == 0 semantics as
// detail::gemm_heterogeneous_loop, which is what the backends call today.
// evidence: docs/perf/gemm.md#gemm-the-heterogeneous-batch-loop
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

// TRAP: BATCHLAS_GEMM_VARIANT has two readers, two vocabularies and two unset
// defaults -- this one (Vendor) and parse_route_env(Op::gemm) ({Auto,Auto}). Both
// read the same captured string; unifying the defaults would change what a bare
// gemm() runs. evidence: docs/perf/gemm.md#gemm-the-route-adapter-and-its-environment-readers
inline GemmVariantRequest gemm_variant_request() {
    const char* raw =
        batchlas::settings().routing.legacy_route(dispatch::Op::gemm).get();
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

// The Route adapter: views + environment -> the pure inputs of
// dispatch::resolve_gemm_route(), and nothing else. The decision lives in
// route_gemm.hh. evidence: docs/perf/gemm.md#gemm-the-route-adapter-and-its-environment-readers

// nullopt when the three views disagree (OpShape cannot say so); such a call
// takes the vendor.
template <typename T>
inline std::optional<dispatch::OpShape> gemm_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& B,
    const MatrixView<T, MatrixFormat::Dense>& C,
    Transpose transA,
    Transpose transB,
    ComputePrecision precision) {
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return std::nullopt;
    }

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    if (k != k_b || m != C.rows() || n != C.cols()) {
        return std::nullopt;
    }

    dispatch::OpShape s;
    s.op = dispatch::Op::gemm;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.m = m;
    s.n = n;
    s.k = k;
    s.batch = A.batch_size();
    s.transA = transA;
    s.transB = transB;
    s.precision = precision;
    s.heterogeneous_batch = gemm_has_heterogeneous_batch(A, B, C);
    s.is_gpu = ctx.device().type == DeviceType::GPU;
    return s;
}

// What the environment asked for, in the canonical vocabulary; unset falls to
// dispatch::legacy_unset_default(Op::gemm).
inline dispatch::Route gemm_route_request() {
    const auto parsed = dispatch::parse_route_env(dispatch::Op::gemm);
    return parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::gemm);
}

template <typename T>
inline dispatch::Route gemm_route(const Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B,
                                  const MatrixView<T, MatrixFormat::Dense>& C,
                                  Transpose transA,
                                  Transpose transB,
                                  ComputePrecision precision,
                                  bool vendor_available = true) {
    const auto shape = gemm_op_shape<T>(ctx, A, B, C, transA, transB, precision);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    return dispatch::resolve_gemm_route<T>(gemm_route_request(), *shape, vendor_available);
}

template <typename T>
inline bool gemm_use_sycl_custom(const Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 Transpose transA,
                                 Transpose transB,
                                 ComputePrecision precision) {
    return dispatch::is_native(
        gemm_route<T>(ctx, A, B, C, transA, transB, precision));
}

} // namespace batchlas::backend