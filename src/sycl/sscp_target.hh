#pragma once
// The target device code is lowered to, as a value. DPC++ runs one device pass per target, so
// these are compile-time constants. AdaptiveCpp generic (SSCP) compiles one IR for every target
// and defines neither __SYCL_DEVICE_ONLY__ nor __NVPTX__; there they are JIT-reflection constants,
// and the JIT deletes the dead branch before lowering. AdaptiveCpp multipass CUDA
// (--acpp-targets=cuda:sm_XX) has a real device pass: there they are constants of that pass.
// TRAP (SSCP): a target intrinsic or inline asm must also sit inside __acpp_if_target_sscp(...);
// on_ptx() alone does not keep it out of the x86 host object, where isel rejects it.

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

namespace batchlas::sycl_impl {

#if BATCHLAS_SYCL_IMPL_ACPP

__attribute__((always_inline)) inline bool on_ptx() {
    bool r = false;
    __acpp_if_target_sscp(
        r = sycl::AdaptiveCpp_jit::reflect<sycl::AdaptiveCpp_jit::reflection_query::compiler_backend>() == sycl::AdaptiveCpp_jit::compiler_backend::ptx;)
    __acpp_if_target_cuda(r = true;)
    return r;
}

// Metal has no double, no vector types and no static local memory; see on_metal() callers.
// 4 is compiler_backend::metal, spelled as a number: acpp 25.10 has no such enumerator.
__attribute__((always_inline)) inline bool on_metal() {
    namespace jit = sycl::AdaptiveCpp_jit;
    bool r = false;
    __acpp_if_target_sscp(
        r = static_cast<int>(jit::reflect<jit::reflection_query::compiler_backend>()) == 4;)
    return r;
}

// sm_XY as XY (sm_89 -> 89, sm_120 -> 120); meaningful only when on_ptx().
__attribute__((always_inline)) inline int target_arch() {
    int r = 0;
    __acpp_if_target_sscp(r = sycl::AdaptiveCpp_jit::reflect<sycl::AdaptiveCpp_jit::reflection_query::target_arch>();)
    __acpp_if_target_cuda(r = __CUDA_ARCH__ / 10;)
    return r;
}

#else

inline constexpr bool on_ptx() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return true;
#else
    return false;
#endif
}

inline constexpr bool on_metal() { return false; }

inline constexpr int target_arch() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SYCL_CUDA_ARCH__)
    return __SYCL_CUDA_ARCH__ / 10;
#else
    return 0;
#endif
}

#endif

} // namespace batchlas::sycl_impl
