#include "gemm_cublasdx_dispatch.hh"

#include "gemm_cublasdx.hh"
#include "gemm_variant.hh"

#include "../linalg-impl.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <batchlas/settings.hh>

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

inline bool is_squareish_shape(int m, int n, int k) {
    const int max_dim = std::max({m, n, k});
    const int min_dim = std::min({m, n, k});
    return min_dim * 2 >= max_dim;
}

inline std::string lowered(const char* raw) {
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

} // namespace

bool cublasdx_gemm_has_forced_variant() {
    const char* raw = batchlas::settings().selection.gemm_cublasdx_kernel.get();
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
    // Same field the presence check above reads, so the two cannot disagree.
    const std::string name =
        lowered(batchlas::settings().selection.gemm_cublasdx_kernel.get());
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

} // namespace batchlas::backend