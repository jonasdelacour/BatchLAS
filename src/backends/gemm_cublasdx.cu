#include "gemm_cublasdx.hh"

// Only available() is left: the cuBLASDx GEMM kernels went with gemm_cublasdx() in phase 5
// (gemm never reaches cuBLASDx since P3.4). The level-3 fused kernels gate on this answer.
namespace batchlas::backend::cublasdx_gemm {

bool available() {
#if defined(BATCHLAS_ENABLE_CUBLASDX_WRAPPER) && __has_include(<cublasdx.hpp>)
    return true;
#else
    return false;
#endif
}

} // namespace batchlas::backend::cublasdx_gemm
