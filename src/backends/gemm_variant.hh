#pragma once

// WP1 S5: ../linalg-impl.hh was the ONLY thing in this header that reached
// CUDA (its line 23 includes <cuda_runtime.h> under BATCHLAS_HAS_CUDA_BACKEND).
// Nothing here needs it: MatrixView, get_effective_dims, Queue and DeviceType
// all come from the three portable headers below. Dropping it makes the whole
// Route adapter -- gemm_op_shape, gemm_route_request, gemm_route,
// gemm_use_sycl_custom -- includable from the vendor-independent facade, which
// is what lets the facade's gemm gain a native arm without duplicating any
// routing logic.
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

// ---------------------------------------------------------------------------
// The Route adapter.
//
// Everything below turns the views + the environment into the two pure inputs
// dispatch::resolve_gemm_route() wants, and nothing else. The decision itself
// now lives in include/batchlas/blas/dispatch/route_gemm.hh, split three ways
// (env read / correctness / measured window) per WP0_DISPATCH_SPEC.md S4, and
// is proven route-identical to the code this replaces by
// tests/route_gemm_equivalence_tests.cc.
//
// It is deliberately wired HERE rather than at the call sites: mkl.cc:64 and
// rocblas.cc:62 call gemm_use_sycl_custom too, and neither TU can be compiled
// on this machine. Substituting at the one definition moves all three call
// sites at once and leaves the two unbuildable ones textually untouched.
// ---------------------------------------------------------------------------

// The shape, or nullopt when the three views cannot describe one GEMM at all.
//
// OpShape is a POD of scalars, so it cannot represent "these views disagree
// with each other" -- and disagreement is precisely what the batch-size, k==k_b
// and m==C.rows() checks inside gemm_custom_problem_supported were testing.
// Absence of a shape is the honest encoding, and it reaches the same outcome:
// the old predicate returned false, and a caller with no shape takes the vendor.
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

// What the environment asked for, in the canonical vocabulary. GEMM's unset
// default is Vendor, unlike the four level-3 ops' Auto -- see the note on
// dispatch::legacy_unset_default.
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