#pragma once
// AdaptiveCpp generic (SSCP) backend. Stage 1 compiles one target-agnostic IR, so the NVPTX choice
// moves from the preprocessor to the JIT: on_ptx() is a JIT-reflection constant. The PTX leg is
// NvptxBackend unchanged; the __nvvm_* names it calls are defined below over the LLVM intrinsics.
// Every other JIT target takes GenericBackend, whose sycl:: collectives assume a converged
// sub-group (acpp's PTX ones use a constant full mask: a diverged chunk hangs).
//
// TRAP: every intrinsic call sits inside __acpp_if_target_sscp(...). SSCP also compiles the kernel
// body into the x86 host object, where an llvm.nvvm.* call fails isel, also at -O0.
// TRAP: inline PTX asm compiles too, but an asm statement is not `convergent`; the intrinsics are.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "../../sycl/sscp_target.hh"

// Asm labels bind these to the LLVM intrinsics, so LLVM attaches the intrinsic's own attributes
// (convergent, inaccessiblemem). A plain __nvvm_* declaration is left unresolved at ptxas.
extern "C" {
int batchlas_sscp_nvvm_shfl_idx(unsigned, int, int, int) __asm__("llvm.nvvm.shfl.sync.idx.i32");
int batchlas_sscp_nvvm_shfl_bfly(unsigned, int, int, int) __asm__("llvm.nvvm.shfl.sync.bfly.i32");
int batchlas_sscp_nvvm_shfl_down(unsigned, int, int, int) __asm__("llvm.nvvm.shfl.sync.down.i32");
int batchlas_sscp_nvvm_shfl_up(unsigned, int, int, int) __asm__("llvm.nvvm.shfl.sync.up.i32");
unsigned batchlas_sscp_nvvm_ballot(unsigned, bool) __asm__("llvm.nvvm.vote.ballot.sync");
unsigned batchlas_sscp_nvvm_activemask() __asm__("llvm.nvvm.activemask");
void batchlas_sscp_nvvm_bar_warp_sync(unsigned) __asm__("llvm.nvvm.bar.warp.sync");
int batchlas_sscp_nvvm_redux_add(int, unsigned) __asm__("llvm.nvvm.redux.sync.add");
int batchlas_sscp_nvvm_redux_min(int, unsigned) __asm__("llvm.nvvm.redux.sync.min");
int batchlas_sscp_nvvm_redux_max(int, unsigned) __asm__("llvm.nvvm.redux.sync.max");
int batchlas_sscp_nvvm_redux_umin(int, unsigned) __asm__("llvm.nvvm.redux.sync.umin");
int batchlas_sscp_nvvm_redux_umax(int, unsigned) __asm__("llvm.nvvm.redux.sync.umax");
int batchlas_sscp_nvvm_redux_and(int, unsigned) __asm__("llvm.nvvm.redux.sync.and");
int batchlas_sscp_nvvm_redux_or(int, unsigned) __asm__("llvm.nvvm.redux.sync.or");
int batchlas_sscp_nvvm_redux_xor(int, unsigned) __asm__("llvm.nvvm.redux.sync.xor");
}

// The spellings backend_nvptx.hh calls. Each body is dead outside the SSCP device IR.
#define BATCHLAS_SSCP_NVVM inline __attribute__((always_inline, convergent))
BATCHLAS_SSCP_NVVM int __nvvm_shfl_sync_idx_i32(unsigned m, int v, int s, int c) {
    int r = v;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_shfl_idx(m, v, s, c);)
    return r;
}
BATCHLAS_SSCP_NVVM int __nvvm_shfl_sync_bfly_i32(unsigned m, int v, int s, int c) {
    int r = v;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_shfl_bfly(m, v, s, c);)
    return r;
}
BATCHLAS_SSCP_NVVM int __nvvm_shfl_sync_down_i32(unsigned m, int v, int s, int c) {
    int r = v;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_shfl_down(m, v, s, c);)
    return r;
}
BATCHLAS_SSCP_NVVM int __nvvm_shfl_sync_up_i32(unsigned m, int v, int s, int c) {
    int r = v;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_shfl_up(m, v, s, c);)
    return r;
}
BATCHLAS_SSCP_NVVM unsigned __nvvm_vote_ballot_sync(unsigned m, bool p) {
    unsigned r = 0;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_ballot(m, p);)
    return r;
}
BATCHLAS_SSCP_NVVM unsigned __nvvm_activemask() {
    unsigned r = 0;
    __acpp_if_target_sscp(r = batchlas_sscp_nvvm_activemask();)
    return r;
}
BATCHLAS_SSCP_NVVM void __nvvm_bar_warp_sync(unsigned m) {
    __acpp_if_target_sscp(batchlas_sscp_nvvm_bar_warp_sync(m);)
}
#undef BATCHLAS_SSCP_NVVM

#include "backend_generic.hh"
#include "backend_nvptx.hh"

namespace batchlas::sgp {

namespace sscp_detail {
// Full-warp redux.sync; reached only when on_ptx() && target_arch() >= 80.
template <typename T, typename Op>
__attribute__((always_inline, convergent)) inline T redux(T v, Op) {
    using nvptx_detail::is_op_v;
    constexpr bool kSigned = std::is_same_v<T, int32_t>;
    const int x = static_cast<int>(v);
    int r = x;
    __acpp_if_target_sscp(
        if constexpr (is_op_v<Op, sycl::plus<T>> || is_op_v<Op, sycl::plus<>>) r = batchlas_sscp_nvvm_redux_add(x, ~0u);
        else if constexpr (is_op_v<Op, sycl::minimum<T>> || is_op_v<Op, sycl::minimum<>>)
            r = kSigned ? batchlas_sscp_nvvm_redux_min(x, ~0u) : batchlas_sscp_nvvm_redux_umin(x, ~0u);
        else if constexpr (is_op_v<Op, sycl::maximum<T>> || is_op_v<Op, sycl::maximum<>>)
            r = kSigned ? batchlas_sscp_nvvm_redux_max(x, ~0u) : batchlas_sscp_nvvm_redux_umax(x, ~0u);
        else if constexpr (is_op_v<Op, sycl::bit_and<T>> || is_op_v<Op, sycl::bit_and<>>) r = batchlas_sscp_nvvm_redux_and(x, ~0u);
        else if constexpr (is_op_v<Op, sycl::bit_or<T>> || is_op_v<Op, sycl::bit_or<>>) r = batchlas_sscp_nvvm_redux_or(x, ~0u);
        else r = batchlas_sscp_nvvm_redux_xor(x, ~0u);)
    return static_cast<T>(r);
}
} // namespace sscp_detail

template <uint32_t P, bool Masked>
struct SscpBackend {
    static constexpr const char* name = "sscp";
    static constexpr bool masked_by_default = true;
    using Nv = NvptxBackend<P, Masked>;
    using Gen = GenericBackend<P, Masked>;

    static uint32_t shfl_idx(const sycl::sub_group& sg, uint32_t b, uint32_t v, uint32_t s) {
        return sycl_impl::on_ptx() ? Nv::shfl_idx(sg, b, v, s) : Gen::shfl_idx(sg, b, v, s);
    }
    static uint32_t shfl_xor(const sycl::sub_group& sg, uint32_t b, uint32_t v, uint32_t m) {
        return sycl_impl::on_ptx() ? Nv::shfl_xor(sg, b, v, m) : Gen::shfl_xor(sg, b, v, m);
    }
    static uint32_t shfl_down(const sycl::sub_group& sg, uint32_t b, uint32_t v, uint32_t d) {
        return sycl_impl::on_ptx() ? Nv::shfl_down(sg, b, v, d) : Gen::shfl_down(sg, b, v, d);
    }
    static uint32_t shfl_up(const sycl::sub_group& sg, uint32_t b, uint32_t v, uint32_t d) {
        return sycl_impl::on_ptx() ? Nv::shfl_up(sg, b, v, d) : Gen::shfl_up(sg, b, v, d);
    }
    static uint32_t ballot(const sycl::sub_group& sg, uint32_t b, bool pred) {
        return sycl_impl::on_ptx() ? Nv::ballot(sg, b, pred) : Gen::ballot(sg, b, pred);
    }
    static void barrier(const sycl::sub_group& sg, uint32_t b) {
        if (sycl_impl::on_ptx()) Nv::barrier(sg, b);
        else Gen::barrier(sg, b);
    }

    template <typename F>
    static decltype(auto) region(F&& f) {
        if (sycl_impl::on_ptx()) return Nv::region(f);
        return f(Gen{});
    }

    // NvptxBackend's redux leg keys on __SYCL_CUDA_ARCH__, which SSCP never defines, so the P = 32
    // integer redux.sync is gated here on the JIT target instead.
    static constexpr bool kRedux32 = P >= 32;
    template <typename T, typename Op>
    static constexpr bool has_native_reduce =
        Nv::template has_native_reduce<T, Op> || (kRedux32 && nvptx_detail::redux_ok_v<T, Op>);

    template <typename T, typename Op>
    static T reduce(const sycl::sub_group& sg, uint32_t b, T v, Op op) {
        if constexpr (kRedux32 && nvptx_detail::redux_ok_v<T, Op>) {
            if (sycl_impl::on_ptx() && sycl_impl::target_arch() >= 80) return sscp_detail::redux(v, op);
        }
        if (sycl_impl::on_ptx()) {
            if constexpr (Nv::template has_native_reduce<T, Op>) return Nv::reduce(sg, b, v, op);
            else return Nv::Full::butterfly(b, v, op);
        }
        constexpr uint32_t kW = static_cast<uint32_t>((sizeof(T) + 3) / 4);
#pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) {
            uint32_t w[kW] = {};
            std::memcpy(w, &v, sizeof(T));
#pragma unroll
            for (uint32_t i = 0; i < kW; ++i) w[i] = Gen::shfl_xor(sg, b, w[i], m);
            T other;
            std::memcpy(&other, w, sizeof(T));
            v = op(v, other);
        }
        return v;
    }
};

} // namespace batchlas::sgp
