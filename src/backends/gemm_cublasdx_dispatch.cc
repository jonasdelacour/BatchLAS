#include "gemm_cublasdx_dispatch.hh"

#include "gemm_cublasdx.hh"
#include "gemm_variant.hh"

#include "../linalg-impl.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace batchlas::backend {

namespace {

inline int max_batch_dimension(const MatrixView<float, MatrixFormat::Dense>& mat,
                               Transpose trans,
                               bool use_rows) {
    int max_dim = 0;
    for (int batch_index = 0; batch_index < mat.batch_size(); ++batch_index) {
        const auto [rows, cols] = get_effective_dims(mat, trans, batch_index);
        max_dim = std::max(max_dim, use_rows ? rows : cols);
    }
    return max_dim;
}

inline const int* effective_rows_ptr(const MatrixView<float, MatrixFormat::Dense>& mat,
                                     Transpose trans) {
    if (!mat.is_heterogeneous()) {
        return nullptr;
    }
    auto dims = trans == Transpose::NoTrans ? mat.active_rows() : mat.active_cols();
    return dims.empty() ? nullptr : dims.data();
}

inline const int* effective_cols_ptr(const MatrixView<float, MatrixFormat::Dense>& mat,
                                     Transpose trans) {
    if (!mat.is_heterogeneous()) {
        return nullptr;
    }
    auto dims = trans == Transpose::NoTrans ? mat.active_cols() : mat.active_rows();
    return dims.empty() ? nullptr : dims.data();
}

inline int batch_dim(const int* batch_dims, int fallback, int batch_index) {
    return batch_dims ? batch_dims[batch_index] : fallback;
}

inline bool is_squareish_shape(int m, int n, int k) {
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    return min_dim * 2 >= max_dim;
}

inline std::string lowered_env(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return {};
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

inline bool variant_matches_name(cublasdx_gemm::CuBLASDxGemmVariant variant, const std::string& name) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback:
        return false;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
        return name == "cublasdx_nn" || name == "cublasdx32x32x32nn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
        return name == "cublasdx_tn" || name == "cublasdx32x32x32tn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
        return name == "cublasdx_nt" || name == "cublasdx32x32x32nt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
        return name == "cublasdx_tt" || name == "cublasdx32x32x32tt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        return name == "cublasdx64_nn" || name == "cublasdx_64x64x32_nn" || name == "cublasdx64x64x32nn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
        return name == "cublasdx64_tn" || name == "cublasdx_64x64x32_tn" || name == "cublasdx64x64x32tn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
        return name == "cublasdx64_nt" || name == "cublasdx_64x64x32_nt" || name == "cublasdx64x64x32nt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return name == "cublasdx64_tt" || name == "cublasdx_64x64x32_tt" || name == "cublasdx64x64x32tt";
    }

    return false;
}

inline bool variant_compatible_with_ops(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                        Transpose transA,
                                        Transpose transB) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback:
        return true;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        return transA == Transpose::NoTrans && transB == Transpose::NoTrans;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
        return transA == Transpose::Trans && transB == Transpose::NoTrans;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
        return transA == Transpose::NoTrans && transB == Transpose::Trans;
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return transA == Transpose::Trans && transB == Transpose::Trans;
    }

    return false;
}

inline bool supports_aligned_packet_loads(const float* ptr, int ld, int stride) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return (address % 16u) == 0u && (ld % 4) == 0 && (stride % 4) == 0;
}

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

} // namespace

const char* cublasdx_gemm_trace_name(cublasdx_gemm::CuBLASDxGemmVariant variant) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback:
        return "gemm_cuda_vendor_fallback";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN:
        return "gemm_cuda_cublasdx_32x32x32_nn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN:
        return "gemm_cuda_cublasdx_32x32x32_tn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT:
        return "gemm_cuda_cublasdx_32x32x32_nt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT:
        return "gemm_cuda_cublasdx_32x32x32_tt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN:
        return "gemm_cuda_cublasdx_64x64x32_nn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN:
        return "gemm_cuda_cublasdx_64x64x32_tn";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT:
        return "gemm_cuda_cublasdx_64x64x32_nt";
    case cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TT:
        return "gemm_cuda_cublasdx_64x64x32_tt";
    }

    return "gemm_cuda_cublasdx_unknown";
}

bool cublasdx_gemm_has_forced_variant() {
    const char* raw = std::getenv("BATCHLAS_GEMM_CUBLASDX_KERNEL");
    return raw && raw[0] != '\0';
}

bool cublasdx_gemm_variant_available(cublasdx_gemm::CuBLASDxGemmVariant variant) {
    switch (variant) {
    case cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback:
        return true;
    default:
        return cublasdx_gemm::available();
    }
}

cublasdx_gemm::CuBLASDxGemmVariant forced_cublasdx_gemm_variant() {
    const std::string name = lowered_env("BATCHLAS_GEMM_CUBLASDX_KERNEL");
    if (name.empty()) {
        return cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback;
    }

    for (cublasdx_gemm::CuBLASDxGemmVariant variant : {
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT,
             cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TT,
         }) {
        if (variant_matches_name(variant, name)) {
            return variant;
        }
    }

    return cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback;
}

cublasdx_gemm::CuBLASDxGemmVariant cublasdx_gemm_select_variant(
    const MatrixView<float, MatrixFormat::Dense>& A,
    const MatrixView<float, MatrixFormat::Dense>& B,
    const MatrixView<float, MatrixFormat::Dense>& C,
    Transpose transA,
    Transpose transB) {
    if (cublasdx_gemm_has_forced_variant()) {
        return forced_cublasdx_gemm_variant();
    }

    const bool heterogeneous = gemm_has_heterogeneous_batch(A, B, C);
    const int m = heterogeneous ? max_batch_dimension(A, transA, true) : get_effective_dims(A, transA).first;
    const int k = heterogeneous ? max_batch_dimension(A, transA, false) : get_effective_dims(A, transA).second;
    const int n = heterogeneous ? max_batch_dimension(B, transB, false) : get_effective_dims(B, transB).second;

    if (transA == Transpose::ConjTrans || transB == Transpose::ConjTrans) {
        return cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback;
    }

    if (heterogeneous) {
        if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
            return cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN;
        }
        if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
            return cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN;
        }
        if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
            return cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT;
        }
        if (transA == Transpose::Trans && transB == Transpose::Trans) {
            return cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT;
        }
    }

    const bool prefer_large_family = is_squareish_shape(m, n, k) && m >= 256 && n >= 256 && k >= 256;
    if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
        return prefer_large_family ? cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NN
                                   : cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NN;
    }
    if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
        return prefer_large_family ? cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TN
                                   : cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TN;
    }
    if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
        return prefer_large_family ? cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32NT
                                   : cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32NT;
    }
    if (transA == Transpose::Trans && transB == Transpose::Trans) {
        return prefer_large_family ? cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx64x64x32TT
                                   : cublasdx_gemm::CuBLASDxGemmVariant::CuBLASDx32x32x32TT;
    }

    return cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback;
}

Event gemm_cublasdx(Queue& ctx,
                    const MatrixView<float, MatrixFormat::Dense>& A,
                    const MatrixView<float, MatrixFormat::Dense>& B,
                    const MatrixView<float, MatrixFormat::Dense>& C,
                    float alpha,
                    float beta,
                    Transpose transA,
                    Transpose transB,
                    ComputePrecision precision) {
    if (precision != ComputePrecision::Default) {
        return gemm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }

    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        throw std::runtime_error("cuBLASDx GEMM path requires matching batch sizes");
    }

    if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
        throw std::runtime_error("cuBLASDx GEMM path received incompatible matrix dimensions");
    }

    Event last_event;
    bool launched_host_work = false;
    int max_m = 0;
    int max_n = 0;
    int max_k = 0;
    bool has_positive_work = false;
    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        static_cast<void>(k_b);
        if (m == 0 || n == 0) {
            continue;
        }
        if (k == 0) {
            last_event = scale(ctx, beta, C.batch_item(batch_index));
            launched_host_work = true;
            continue;
        }
        max_m = std::max(max_m, m);
        max_n = std::max(max_n, n);
        max_k = std::max(max_k, k);
        has_positive_work = true;
    }

    if (!has_positive_work) {
        if (launched_host_work) {
            return std::move(last_event);
        }
        return ctx.create_event_after_external_work();
    }

    const auto variant = cublasdx_gemm_select_variant(A, B, C, transA, transB);
    if (variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback) {
        return gemm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }

    if (!cublasdx_gemm_variant_available(variant)) {
        return gemm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }

    if (!variant_compatible_with_ops(variant, transA, transB)) {
        throw std::runtime_error("Requested cuBLASDx GEMM variant is incompatible with the transpose operands");
    }

    cublasdx_gemm::GemmLaunchDescriptor desc{};
    desc.a_ptr = A.data_ptr();
    desc.b_ptr = B.data_ptr();
    desc.c_ptr = C.data_ptr();
    desc.lda = A.ld();
    desc.ldb = B.ld();
    desc.ldc = C.ld();
    desc.stride_a = A.stride();
    desc.stride_b = B.stride();
    desc.stride_c = C.stride();
    desc.m_batch = effective_rows_ptr(A, transA);
    if (!desc.m_batch) {
        desc.m_batch = C.is_heterogeneous() ? C.active_rows().data() : nullptr;
    }
    desc.n_batch = effective_cols_ptr(B, transB);
    if (!desc.n_batch) {
        desc.n_batch = C.is_heterogeneous() ? C.active_cols().data() : nullptr;
    }
    desc.k_batch = effective_cols_ptr(A, transA);
    if (!desc.k_batch) {
        desc.k_batch = effective_rows_ptr(B, transB);
    }
    desc.m = max_m;
    desc.n = max_n;
    desc.k = max_k;
    desc.batch = A.batch_size();
    desc.alpha = alpha;
    desc.beta = beta;
    desc.heterogeneous = gemm_has_heterogeneous_batch(A, B, C);
    desc.packet_a = supports_aligned_packet_loads(A.data_ptr(), A.ld(), A.stride());
    desc.packet_b = supports_aligned_packet_loads(B.data_ptr(), B.ld(), B.stride());
    desc.aligned_fast_path = false;

    BATCHLAS_KERNEL_TRACE_SCOPE(cublasdx_gemm_trace_name(variant));
    const cudaError_t status = cublasdx_gemm::launch_float(variant, desc, cuda_stream_from_queue(ctx));
    if (status == cudaErrorNotSupported) {
        return gemm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cuBLASDx GEMM launch failed: ") + cudaGetErrorString(status));
    }

    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend