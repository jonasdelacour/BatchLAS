// The four cuBLASDx fused level-3 tails, lifted verbatim out of the dispatchers.
//
// This is the ONLY level-3 TU that reaches a CUDA header. It is compiled only
// when BATCHLAS_HAS_CUBLAS (src/backends/CMakeLists.txt); the vendor-free build
// gets level3_fused_absent.cc instead, which answers NoKernel.
//
// Lifted verbatim on purpose. Every one of these tails is unreachable in this
// build -- MathDx is absent, so *_cublasdx::available() is false and
// cublasdx_variant_needs_fallback is unconditionally true -- which makes moving
// them zero-route-risk but also means a mistake here would not be caught by any
// test on this machine. So nothing is "tidied" in transit: the descriptor
// fields, the trace scope names and the three exits are exactly as they were.
//
// What does NOT move is each op's REACTION to a failed launch, because the four
// disagree: syr2k throws where the others fall back, trmm throws only when
// forced, and symm/syrk fall back to different shims. Those stay in the
// dispatchers, driven by FusedResult::Outcome.

#include "level3_fused.hh"

#include "gemm_cublasdx_dispatch.hh"
#include "cublasdx_dispatch_common.hh"
#include "symm_cublasdx_fused.hh"
#include "syrk_cublasdx_fused.hh"
#include "syr2k_cublasdx_fused.hh"
#include "trmm_cublasdx_fused.hh"

#include "../util/kernel-trace.hh"

#include <stdexcept>
#include <string>

namespace batchlas::backend::detail {

namespace {

// The two non-Ran exits, so each op's tail reads as one shape.
FusedResult no_kernel()  { return FusedResult{Event{}, FusedResult::Outcome::NoKernel}; }
FusedResult unsupported(){ return FusedResult{Event{}, FusedResult::Outcome::DeviceUnsupported}; }

} // namespace

FusedResult symm_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Side side,
                           Uplo uplo) {
    const auto variant = cublasdx_gemm_select_variant(side == Side::Left ? A : B,
                                                      side == Side::Left ? B : A,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (cublasdx_variant_needs_fallback(variant, symm_cublasdx::available())) {
        return no_kernel();
    }

    symm_cublasdx::SymmLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.m = C.rows();
    desc.n = C.cols();
    desc.k = A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("symm_cuda_custom.fused");
    const cudaError_t status = symm_cublasdx::launch_float(variant,
                                                           desc,
                                                           side,
                                                           uplo,
                                                           cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return unsupported();
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYMM launch failed: ") + cudaGetErrorString(status));
    }
    return FusedResult{ctx.create_event_after_external_work(), FusedResult::Outcome::Ran};
}

FusedResult syrk_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Uplo uplo,
                           Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    const auto variant = cublasdx_gemm_select_variant(A, A, C, transA, transB);
    if (cublasdx_variant_needs_fallback(variant, syrk_cublasdx::available())) {
        return no_kernel();
    }

    syrk_cublasdx::SyrkLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_c = C.stride();
    desc.n = C.rows();
    desc.k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.fused");
    const cudaError_t status = syrk_cublasdx::launch_float(variant,
                                                           desc,
                                                           uplo,
                                                           transA,
                                                           cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return unsupported();
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYRK launch failed: ") + cudaGetErrorString(status));
    }
    return FusedResult{ctx.create_event_after_external_work(), FusedResult::Outcome::Ran};
}

FusedResult syr2k_fused_try(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            float alpha,
                            float beta,
                            Uplo uplo,
                            Transpose transA) {
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
    const auto variant = cublasdx_gemm_select_variant(A, B, C, transA, transB);
    if (cublasdx_variant_needs_fallback(variant, syr2k_cublasdx::available())) {
        return no_kernel();
    }

    syr2k_cublasdx::Syr2kLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.n = C.rows();
    desc.k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;

    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.fused");
    const cudaError_t status = syr2k_cublasdx::launch_float(variant,
                                                            desc,
                                                            uplo,
                                                            transA,
                                                            cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return unsupported();
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused SYR2K launch failed: ") + cudaGetErrorString(status));
    }
    return FusedResult{ctx.create_event_after_external_work(), FusedResult::Outcome::Ran};
}

FusedResult trmm_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag) {
    static_cast<void>(side);
    static_cast<void>(uplo);
    static_cast<void>(transA);

    const auto variant = cublasdx_gemm_select_variant(A,
                                                      B,
                                                      C,
                                                      Transpose::NoTrans,
                                                      Transpose::NoTrans);
    if (cublasdx_variant_needs_fallback(variant, trmm_cublasdx::available())) {
        return no_kernel();
    }

    trmm_cublasdx::TrmmLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.m = C.rows();
    desc.n = C.cols();
    desc.batch = A.batch_size();
    desc.alpha = alpha;

    BATCHLAS_KERNEL_TRACE_SCOPE("trmm_cuda_custom.fused");
    const cudaError_t status = trmm_cublasdx::launch_float(variant,
                                                           desc,
                                                           diag,
                                                           cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return unsupported();
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx fused TRMM launch failed: ") + cudaGetErrorString(status));
    }
    return FusedResult{ctx.create_event_after_external_work(), FusedResult::Outcome::Ran};
}

} // namespace batchlas::backend::detail
