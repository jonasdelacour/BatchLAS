// The gemm launchers and their kernels, shared by gemm_kernels.cc and the shard TUs that
// instantiate them (gemm_rows.hh). Everything here is in an unnamed namespace: each TU that
// includes it gets its own copy, and each kernel is instantiated by exactly one TU.
#pragma once

#include <src/ops/gemm/gemm_kernels.hh>

#include <src/ops/gemm/kernels/accessors.hh>
#include <src/ops/gemm/kernels/register_128x128.hh>
#include <src/ops/gemm/kernels/register_256x128.hh>
#include <src/ops/gemm/kernels/register_64x64_k16_wide.hh>
#include <src/ops/gemm/kernels/register_launchers.hh>
#include <src/ops/gemm/kernels/register_wide_transposed.hh>
#include <src/ops/gemm/kernels/small_batched.hh>
#include <src/ops/gemm/kernels/tiled_general.hh>
#include <src/sycl/local_mem.hh>

#include <src/linalg-impl.hh>
#include <src/ops/gemm/choice.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace {


namespace og = ::batchlas::ops::gemm;

static_assert(og::kSmallMaxDim == sycl_gemm_small::kSmallMaxDim && og::kSmallWg == sycl_gemm_small::kSmallWg &&
                  og::kSmallTiledMaxDim == sycl_gemm_small::kSmallTiledMaxDim,
              "choice.hh's small limits must match small_batched.hh");
template <typename T>
constexpr bool small_slm_matches() {
    namespace gs = sycl_gemm_small;
    constexpr std::int64_t s = sizeof(T);
    return og::small_max_dim<T>() == gs::kSmallMaxDimFor<T> &&
           og::small_slm_bytes<T>(false, 8) == s * gs::kSmallBatchedSlmElems<8> &&
           og::small_slm_bytes<T>(false, 16) == s * gs::kSmallBatchedSlmElems<16> &&
           og::small_slm_bytes<T>(true, 32) == s * gs::kSmallBatchedSlmElems<32> &&
           og::small_slm_bytes<T>(false, 64) == s * gs::kSmallBatchedSlmElems<64> &&
           (!std::is_same_v<T, float> || (og::small_slm_bytes<T>(true, 48) == s * gs::kSmallTiledSlmElems<48> &&
                                          og::small_slm_bytes<T>(true, 56) == s * gs::kSmallTiledSlmElems<56>));
}
static_assert(small_slm_matches<float>() && small_slm_matches<double>() && small_slm_matches<std::complex<float>>() &&
                  small_slm_matches<std::complex<double>>(),
              "choice.hh's small_max_dim and small_slm_bytes must match small_batched.hh");

template <typename T>
class GemmDirectKernel;

template <typename T>
inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline const char* kernel_trace_name(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::Direct: return "gemm_sycl_direct";
    case KernelVariant::Tiled16: return "gemm_sycl_tiled16";
    case KernelVariant::Tiled32x32Register: return "gemm_sycl_register_32x32";
    case KernelVariant::Tiled64x64Register: return "gemm_sycl_register_64x64";
    case KernelVariant::Tiled64x64RegisterK16: return "gemm_sycl_register_64x64_k16";
    case KernelVariant::Tiled64x64RegisterK16TN: return "gemm_sycl_register_64x64_k16_tn";
    case KernelVariant::Tiled64x64RegisterK16NT: return "gemm_sycl_register_64x64_k16_nt";
    case KernelVariant::Tiled64x64RegisterK16TT: return "gemm_sycl_register_64x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK16: return "gemm_sycl_register_128x32_k16";
    case KernelVariant::Tiled128x32RegisterK16TN: return "gemm_sycl_register_128x32_k16_tn";
    case KernelVariant::Tiled128x32RegisterK16NT: return "gemm_sycl_register_128x32_k16_nt";
    case KernelVariant::Tiled128x32RegisterK16TT: return "gemm_sycl_register_128x32_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32TN: return "gemm_sycl_register_128x32_k32_tn";
    case KernelVariant::Tiled128x32RegisterK32NT: return "gemm_sycl_register_128x32_k32_nt";
    case KernelVariant::Tiled128x32RegisterK32TT: return "gemm_sycl_register_128x32_k32_tt";
    case KernelVariant::Tiled128x64RegisterK16TN: return "gemm_sycl_register_128x64_k16_tn";
    case KernelVariant::Tiled128x64RegisterK16NT: return "gemm_sycl_register_128x64_k16_nt";
    case KernelVariant::Tiled128x64RegisterK16TT: return "gemm_sycl_register_128x64_k16_tt";
    case KernelVariant::Tiled128x32RegisterK32S2U1Aligned: return "gemm_sycl_register_128x32_k32_s2_u1_aligned";
    case KernelVariant::Tiled128x32RegisterK32S2U1Generic: return "gemm_sycl_register_128x32_k32_s2_u1_generic";
    case KernelVariant::Tiled128x64RegisterK32Large: return "gemm_sycl_register_128x64_k32_large";
    case KernelVariant::Tiled128x64RegisterK32LargeU2: return "gemm_sycl_register_128x64_k32_large_u2";
    case KernelVariant::Tiled128x128RegisterK8: return "gemm_sycl_register_128x128_k8";
    case KernelVariant::Tiled128x128RegisterK8NT: return "gemm_sycl_register_128x128_k8_nt";
    case KernelVariant::Tiled128x128RegisterK8TN: return "gemm_sycl_register_128x128_k8_tn";
    case KernelVariant::Tiled128x128RegisterK8TT: return "gemm_sycl_register_128x128_k8_tt";
    case KernelVariant::Tiled64x64RegisterK16Wide: return "gemm_sycl_register_64x64_k16_wide";
    case KernelVariant::Tiled64x64RegisterK16WideCN: return "gemm_sycl_register_64x64_k16_wide_cn";
    case KernelVariant::Tiled64x64RegisterK16WideNC: return "gemm_sycl_register_64x64_k16_wide_nc";
    case KernelVariant::Tiled128x32RegisterK16WideNC: return "gemm_sycl_register_128x32_k16_wide_nc";
    case KernelVariant::Tiled32x128RegisterK16WideCN: return "gemm_sycl_register_32x128_k16_wide_cn";
    case KernelVariant::Tiled32x128RegisterK16: return "gemm_sycl_register_32x128_k16";
    case KernelVariant::Tiled32x128RegisterK16TN: return "gemm_sycl_register_32x128_k16_tn";
    case KernelVariant::Tiled32x128RegisterK16TT: return "gemm_sycl_register_32x128_k16_tt";
    case KernelVariant::Tiled32x32RegisterK16Wide: return "gemm_sycl_register_32x32_k16_wide";
    case KernelVariant::Tiled16x16RegisterK16Wide: return "gemm_sycl_register_16x16_k16_wide";
    case KernelVariant::Tiled64x64RegisterK16WideNT: return "gemm_sycl_register_64x64_k16_wide_nt";
    case KernelVariant::Tiled64x64RegisterK16WideTN: return "gemm_sycl_register_64x64_k16_wide_tn";
    case KernelVariant::Tiled64x64RegisterK16WideTT: return "gemm_sycl_register_64x64_k16_wide_tt";
    case KernelVariant::Tiled64x64RegisterK16WideTC: return "gemm_sycl_register_64x64_k16_wide_tc";
    case KernelVariant::Tiled64x64RegisterK16WideCT: return "gemm_sycl_register_64x64_k16_wide_ct";
    case KernelVariant::Tiled64x64RegisterK16WideCC: return "gemm_sycl_register_64x64_k16_wide_cc";
    case KernelVariant::Tiled128x32RegisterK16WideNT: return "gemm_sycl_register_128x32_k16_wide_nt";
    case KernelVariant::Tiled128x32RegisterK16WideTN: return "gemm_sycl_register_128x32_k16_wide_tn";
    case KernelVariant::Tiled128x32RegisterK16WideCN: return "gemm_sycl_register_128x32_k16_wide_cn";
    case KernelVariant::Tiled32x128RegisterK16WideNT: return "gemm_sycl_register_32x128_k16_wide_nt";
    case KernelVariant::Tiled32x128RegisterK16WideTN: return "gemm_sycl_register_32x128_k16_wide_tn";
    case KernelVariant::Tiled32x128RegisterK16WideNC: return "gemm_sycl_register_32x128_k16_wide_nc";
    case KernelVariant::Tiled32x32RegisterK16WideNT: return "gemm_sycl_register_32x32_k16_wide_nt";
    case KernelVariant::Tiled32x32RegisterK16WideNC: return "gemm_sycl_register_32x32_k16_wide_nc";
    case KernelVariant::Tiled32x32RegisterK16WideTN: return "gemm_sycl_register_32x32_k16_wide_tn";
    case KernelVariant::Tiled32x32RegisterK16WideCN: return "gemm_sycl_register_32x32_k16_wide_cn";
    case KernelVariant::Tiled32x32RegisterK16WideCT: return "gemm_sycl_register_32x32_k16_wide_ct";
    case KernelVariant::Tiled16x16RegisterK16WideNT: return "gemm_sycl_register_16x16_k16_wide_nt";
    case KernelVariant::Tiled16x16RegisterK16WideNC: return "gemm_sycl_register_16x16_k16_wide_nc";
    case KernelVariant::Tiled16x16RegisterK16WideTN: return "gemm_sycl_register_16x16_k16_wide_tn";
    case KernelVariant::Tiled16x16RegisterK16WideCN: return "gemm_sycl_register_16x16_k16_wide_cn";
    case KernelVariant::Tiled16x16RegisterK16WideCT: return "gemm_sycl_register_16x16_k16_wide_ct";
    case KernelVariant::Tiled64x32RegisterK16Wide: return "gemm_sycl_register_64x32_k16_wide";
    case KernelVariant::Tiled64x32RegisterK16WideNT: return "gemm_sycl_register_64x32_k16_wide_nt";
    case KernelVariant::Tiled64x32RegisterK16WideNC: return "gemm_sycl_register_64x32_k16_wide_nc";
    case KernelVariant::Tiled64x32RegisterK16WideTN: return "gemm_sycl_register_64x32_k16_wide_tn";
    case KernelVariant::Tiled64x32RegisterK16WideCN: return "gemm_sycl_register_64x32_k16_wide_cn";
    case KernelVariant::Tiled64x32RegisterK16WideCT: return "gemm_sycl_register_64x32_k16_wide_ct";
    case KernelVariant::Tiled32x64RegisterK16Wide: return "gemm_sycl_register_32x64_k16_wide";
    case KernelVariant::Tiled32x64RegisterK16WideNT: return "gemm_sycl_register_32x64_k16_wide_nt";
    case KernelVariant::Tiled32x64RegisterK16WideNC: return "gemm_sycl_register_32x64_k16_wide_nc";
    case KernelVariant::Tiled32x64RegisterK16WideTN: return "gemm_sycl_register_32x64_k16_wide_tn";
    case KernelVariant::Tiled32x64RegisterK16WideCN: return "gemm_sycl_register_32x64_k16_wide_cn";
    case KernelVariant::Tiled32x64RegisterK16WideCT: return "gemm_sycl_register_32x64_k16_wide_ct";
    case KernelVariant::Tiled64x64RegisterK8Wide: return "gemm_sycl_register_64x64_k8_wide";
    case KernelVariant::Tiled64x64RegisterK8WideNT: return "gemm_sycl_register_64x64_k8_wide_nt";
    case KernelVariant::Tiled64x64RegisterK8WideNC: return "gemm_sycl_register_64x64_k8_wide_nc";
    case KernelVariant::Tiled64x64RegisterK8WideTN: return "gemm_sycl_register_64x64_k8_wide_tn";
    case KernelVariant::Tiled64x64RegisterK8WideTT: return "gemm_sycl_register_64x64_k8_wide_tt";
    case KernelVariant::Tiled64x64RegisterK8WideTC: return "gemm_sycl_register_64x64_k8_wide_tc";
    case KernelVariant::Tiled64x64RegisterK8WideCN: return "gemm_sycl_register_64x64_k8_wide_cn";
    case KernelVariant::Tiled64x64RegisterK8WideCT: return "gemm_sycl_register_64x64_k8_wide_ct";
    case KernelVariant::Tiled64x64RegisterK8WideCC: return "gemm_sycl_register_64x64_k8_wide_cc";
    case KernelVariant::Tiled128x128RegisterK8S4: return "gemm_sycl_register_128x128_k8_s4";
    case KernelVariant::Tiled128x128RegisterK8S4NT: return "gemm_sycl_register_128x128_k8_s4_nt";
    case KernelVariant::Tiled128x128RegisterK8S4TN: return "gemm_sycl_register_128x128_k8_s4_tn";
    case KernelVariant::Tiled128x128RegisterK8S4TT: return "gemm_sycl_register_128x128_k8_s4_tt";
    case KernelVariant::Tiled128x128RegisterK8Wide: return "gemm_sycl_register_128x128_k8_wide";
    case KernelVariant::Tiled128x128RegisterK8WideNT: return "gemm_sycl_register_128x128_k8_wide_nt";
    case KernelVariant::Tiled128x128RegisterK8WideNC: return "gemm_sycl_register_128x128_k8_wide_nc";
    case KernelVariant::Tiled128x128RegisterK8WideTN: return "gemm_sycl_register_128x128_k8_wide_tn";
    case KernelVariant::Tiled128x128RegisterK8WideTT: return "gemm_sycl_register_128x128_k8_wide_tt";
    case KernelVariant::Tiled128x128RegisterK8WideTC: return "gemm_sycl_register_128x128_k8_wide_tc";
    case KernelVariant::Tiled128x128RegisterK8WideCN: return "gemm_sycl_register_128x128_k8_wide_cn";
    case KernelVariant::Tiled128x128RegisterK8WideCT: return "gemm_sycl_register_128x128_k8_wide_ct";
    case KernelVariant::Tiled128x128RegisterK8WideCC: return "gemm_sycl_register_128x128_k8_wide_cc";
    case KernelVariant::Tiled32x32RegisterK16WideTT: return "gemm_sycl_register_32x32_k16_wide_tt";
    case KernelVariant::Tiled32x32RegisterK16WideTC: return "gemm_sycl_register_32x32_k16_wide_tc";
    case KernelVariant::Tiled32x32RegisterK16WideCC: return "gemm_sycl_register_32x32_k16_wide_cc";
    case KernelVariant::Tiled16x16RegisterK16WideTT: return "gemm_sycl_register_16x16_k16_wide_tt";
    case KernelVariant::Tiled16x16RegisterK16WideTC: return "gemm_sycl_register_16x16_k16_wide_tc";
    case KernelVariant::Tiled16x16RegisterK16WideCC: return "gemm_sycl_register_16x16_k16_wide_cc";
    case KernelVariant::Tiled64x32RegisterK16WideCC: return "gemm_sycl_register_64x32_k16_wide_cc";
    case KernelVariant::Tiled32x64RegisterK16WideCC: return "gemm_sycl_register_32x64_k16_wide_cc";
    case KernelVariant::Tiled128x32RegisterK16WideCC: return "gemm_sycl_register_128x32_k16_wide_cc";
    case KernelVariant::Tiled32x128RegisterK16WideCC: return "gemm_sycl_register_32x128_k16_wide_cc";
    case KernelVariant::Tiled256x128RegisterK8S4: return "gemm_sycl_register_256x128_k8_s4";
    case KernelVariant::Tiled256x128RegisterK8S4NT: return "gemm_sycl_register_256x128_k8_s4_nt";
    case KernelVariant::Tiled256x128RegisterK8S4TN: return "gemm_sycl_register_256x128_k8_s4_tn";
    case KernelVariant::Tiled256x128RegisterK8S4TT: return "gemm_sycl_register_256x128_k8_s4_tt";
    case KernelVariant::Tiled128x256RegisterK8S4: return "gemm_sycl_register_128x256_k8_s4";
    case KernelVariant::Tiled128x256RegisterK8S4NT: return "gemm_sycl_register_128x256_k8_s4_nt";
    case KernelVariant::Tiled128x256RegisterK8S4TN: return "gemm_sycl_register_128x256_k8_s4_tn";
    case KernelVariant::Tiled128x256RegisterK8S4TT: return "gemm_sycl_register_128x256_k8_s4_tt";
    case KernelVariant::Count: break;
    }
    return "gemm_sycl_unknown";
}

// A split-k body's name: its host instantiation's, suffixed (static storage for the trace scope).
const char* splitk_trace_name(KernelVariant variant) {
    constexpr int kCount = static_cast<int>(KernelVariant::Count);
    static const std::array<std::string, kCount> names = [] {
        std::array<std::string, kCount> out;
        for (int i = 0; i < kCount; ++i) out[i] = std::string(kernel_trace_name(KernelVariant(i))) + "_splitk";
        return out;
    }();
    return names[static_cast<int>(variant)].c_str();
}

// The unpredicated leg's name where both legs used to share one, so a fast-leg predicate stuck
// at false shows in the kernel trace (the 128x32x32 NN legs have always been named apart).
const char* aligned_trace_name(KernelVariant variant) {
    switch (variant) {
    case KernelVariant::Tiled128x64RegisterK32Large: return "gemm_sycl_register_128x64_k32_large_aligned";
    case KernelVariant::Tiled128x64RegisterK32LargeU2: return "gemm_sycl_register_128x64_k32_large_u2_aligned";
    case KernelVariant::Tiled128x128RegisterK8: return "gemm_sycl_register_128x128_k8_aligned";
    case KernelVariant::Tiled128x128RegisterK8NT: return "gemm_sycl_register_128x128_k8_nt_aligned";
    case KernelVariant::Tiled128x128RegisterK8TN: return "gemm_sycl_register_128x128_k8_tn_aligned";
    case KernelVariant::Tiled128x128RegisterK8TT: return "gemm_sycl_register_128x128_k8_tt_aligned";
    case KernelVariant::Tiled64x64RegisterK16Wide: return "gemm_sycl_register_64x64_k16_wide_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideNT: return "gemm_sycl_register_64x64_k16_wide_nt_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideNC: return "gemm_sycl_register_64x64_k16_wide_nc_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideTN: return "gemm_sycl_register_64x64_k16_wide_tn_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideTT: return "gemm_sycl_register_64x64_k16_wide_tt_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideTC: return "gemm_sycl_register_64x64_k16_wide_tc_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideCN: return "gemm_sycl_register_64x64_k16_wide_cn_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideCT: return "gemm_sycl_register_64x64_k16_wide_ct_aligned";
    case KernelVariant::Tiled64x64RegisterK16WideCC: return "gemm_sycl_register_64x64_k16_wide_cc_aligned";
    case KernelVariant::Tiled64x64RegisterK8Wide: return "gemm_sycl_register_64x64_k8_wide_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideNT: return "gemm_sycl_register_64x64_k8_wide_nt_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideNC: return "gemm_sycl_register_64x64_k8_wide_nc_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideTN: return "gemm_sycl_register_64x64_k8_wide_tn_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideTT: return "gemm_sycl_register_64x64_k8_wide_tt_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideTC: return "gemm_sycl_register_64x64_k8_wide_tc_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideCN: return "gemm_sycl_register_64x64_k8_wide_cn_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideCT: return "gemm_sycl_register_64x64_k8_wide_ct_aligned";
    case KernelVariant::Tiled64x64RegisterK8WideCC: return "gemm_sycl_register_64x64_k8_wide_cc_aligned";
    case KernelVariant::Tiled128x128RegisterK8Wide: return "gemm_sycl_register_128x128_k8_wide_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideNT: return "gemm_sycl_register_128x128_k8_wide_nt_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideNC: return "gemm_sycl_register_128x128_k8_wide_nc_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideTN: return "gemm_sycl_register_128x128_k8_wide_tn_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideTT: return "gemm_sycl_register_128x128_k8_wide_tt_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideTC: return "gemm_sycl_register_128x128_k8_wide_tc_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideCN: return "gemm_sycl_register_128x128_k8_wide_cn_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideCT: return "gemm_sycl_register_128x128_k8_wide_ct_aligned";
    case KernelVariant::Tiled128x128RegisterK8WideCC: return "gemm_sycl_register_128x128_k8_wide_cc_aligned";
    case KernelVariant::Tiled256x128RegisterK8S4: return "gemm_sycl_register_256x128_k8_s4_aligned";
    case KernelVariant::Tiled256x128RegisterK8S4NT: return "gemm_sycl_register_256x128_k8_s4_nt_aligned";
    case KernelVariant::Tiled256x128RegisterK8S4TT: return "gemm_sycl_register_256x128_k8_s4_tt_aligned";
    case KernelVariant::Tiled128x256RegisterK8S4: return "gemm_sycl_register_128x256_k8_s4_aligned";
    case KernelVariant::Tiled128x256RegisterK8S4NT: return "gemm_sycl_register_128x256_k8_s4_nt_aligned";
    case KernelVariant::Tiled128x256RegisterK8S4TT: return "gemm_sycl_register_128x256_k8_s4_tt_aligned";
    default: return kernel_trace_name(variant);
    }
}

// Wide v2 128x128's cp.async legs: the register-prefetch leg's name, suffixed _s3 (static storage).
const char* stages3_trace_name(KernelVariant variant, bool vec) {
    constexpr int kCount = static_cast<int>(KernelVariant::Count);
    static const std::array<std::string, 2 * kCount> names = [] {
        std::array<std::string, 2 * kCount> out;
        for (int i = 0; i < kCount; ++i) {
            out[2 * i] = std::string(kernel_trace_name(KernelVariant(i))) + "_s3";
            out[2 * i + 1] = std::string(aligned_trace_name(KernelVariant(i))) + "_s3";
        }
        return out;
    }();
    return names[2 * static_cast<int>(variant) + (vec ? 1 : 0)].c_str();
}

// The transposed wide tiles' cp.async leg: suffixed _async (element copies) or _async_vec (16-byte
// granules along i), after _splitk for a split body.
const char* async_trace_name(KernelVariant variant, bool split, bool vec) {
    constexpr int kCount = static_cast<int>(KernelVariant::Count);
    static const std::array<std::string, 4 * kCount> names = [] {
        std::array<std::string, 4 * kCount> out;
        for (int i = 0; i < 4 * kCount; ++i)
            out[i] = std::string(kernel_trace_name(KernelVariant(i / 4))) + (i % 4 >= 2 ? "_splitk" : "") +
                "_async" + (i % 2 ? "_vec" : "");
        return out;
    }();
    return names[4 * static_cast<int>(variant) + (split ? 2 : 0) + (vec ? 1 : 0)].c_str();
}

template <typename T>
Event launch_direct(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha,
                    T beta,
                    Transpose transA,
                    Transpose transB) {
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_direct");

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    constexpr int workgroup = 8;

    const sycl::range<3> local(1, workgroup, workgroup);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(ceil_div<T>(m, workgroup) * workgroup),
                                static_cast<size_t>(ceil_div<T>(n, workgroup) * workgroup));

    ctx->submit([&](sycl::handler& h) {
        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const Transpose op_a = transA;
        const Transpose op_b = transB;

        batchlas::impl::parallel_for<GemmDirectKernel<T>>(h, sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int row = static_cast<int>(item.get_global_id(1));
            const int col = static_cast<int>(item.get_global_id(2));
            if (bid >= batch || row >= m || col >= n) {
                return;
            }

            T sum = T(0);
            // Rebased per item in 64 bits (bid * stride passes 2^31); offsets within an item stay int.
            const T* const a_item = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
            const T* const b_item = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
            T* const c_item = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
            constexpr int batch_a = 0, batch_b = 0, batch_c = 0;
            for (int kk = 0; kk < k; ++kk) {
                const T a_val = operand_value(a_item, lda, batch_a, row, kk, op_a);
                const T b_val = operand_value(b_item, ldb, batch_b, kk, col, op_b);
                sum += a_val * b_val;
            }
            LinearEpilogue<T>::store(c_item + batch_c + col * ldc + row, alpha, beta, sum);
        });
    });

    return ctx.get_event();
}

template <typename T, int Tile>
Event launch_tiled(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB) {
    if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::NoTrans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::NoTrans && transB == Transpose::ConjTrans) {
        return launch_tiled_general<T, Tile, Transpose::NoTrans, Transpose::ConjTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::Trans && transB == Transpose::ConjTrans) {
        return launch_tiled_general<T, Tile, Transpose::Trans, Transpose::ConjTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::ConjTrans && transB == Transpose::NoTrans) {
        return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::NoTrans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }
    if (transA == Transpose::ConjTrans && transB == Transpose::Trans) {
        return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::Trans>(
            ctx, A, B, C, alpha, beta, kernel_trace_name);
    }

    return launch_tiled_general<T, Tile, Transpose::ConjTrans, Transpose::ConjTrans>(
        ctx, A, B, C, alpha, beta, kernel_trace_name);
}


// ---- register-tiled (float) ------------------------------------------------------------

enum Form { kNN, kNT, kTN, kTT };

// The trace identity of reg_configs[I] at a form: every (config, form) pair the table
// instantiates is one of the old named variants. `aligned` names the NN unpredicated leg.
constexpr KernelVariant reg_variant(std::size_t i, Form f, bool aligned) {
    using K = KernelVariant;
    switch (i) {
    case 0: return K::Tiled32x32Register;
    case 1: return K::Tiled64x64Register;
    case 2: return f == kNN ? K::Tiled64x64RegisterK16 : f == kTN ? K::Tiled64x64RegisterK16TN
                   : f == kNT ? K::Tiled64x64RegisterK16NT : K::Tiled64x64RegisterK16TT;
    case 3: return f == kNN ? K::Tiled128x32RegisterK16 : f == kTN ? K::Tiled128x32RegisterK16TN
                   : f == kNT ? K::Tiled128x32RegisterK16NT : K::Tiled128x32RegisterK16TT;
    case 4: return f == kNN ? (aligned ? K::Tiled128x32RegisterK32S2U1Aligned : K::Tiled128x32RegisterK32S2U1Generic)
                   : f == kTN ? K::Tiled128x32RegisterK32TN : f == kNT ? K::Tiled128x32RegisterK32NT
                              : K::Tiled128x32RegisterK32TT;
    case 5: return f == kTN ? K::Tiled128x64RegisterK16TN : f == kNT ? K::Tiled128x64RegisterK16NT
                            : K::Tiled128x64RegisterK16TT;
    case 6: return f == kNN ? K::Tiled32x128RegisterK16 : f == kTN ? K::Tiled32x128RegisterK16TN
                            : K::Tiled32x128RegisterK16TT;
    case 7: return K::Tiled128x64RegisterK32Large;
    case 8: return K::Tiled128x64RegisterK32LargeU2;
    case 9: return f == kNN ? K::Tiled128x128RegisterK8 : f == kTN ? K::Tiled128x128RegisterK8TN  // == Body::kVariant
                   : f == kNT ? K::Tiled128x128RegisterK8NT : K::Tiled128x128RegisterK8TT;
    case 10: return f == kNN ? K::Tiled128x128RegisterK8S4 : f == kTN ? K::Tiled128x128RegisterK8S4TN
                    : f == kNT ? K::Tiled128x128RegisterK8S4NT : K::Tiled128x128RegisterK8S4TT;
    case 11: return f == kNN ? K::Tiled256x128RegisterK8S4 : f == kTN ? K::Tiled256x128RegisterK8S4TN
                    : f == kNT ? K::Tiled256x128RegisterK8S4NT : K::Tiled256x128RegisterK8S4TT;
    case 12: return f == kNN ? K::Tiled128x256RegisterK8S4 : f == kTN ? K::Tiled128x256RegisterK8S4TN
                    : f == kNT ? K::Tiled128x256RegisterK8S4NT : K::Tiled128x256RegisterK8S4TT;
    default: return K::Direct;  // a row without a name; launch_reg_cfg static_asserts it is never read
    }
}

template <int I, Form F>
Event launch_reg_cfg(Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                     const MatrixView<float, MatrixFormat::Dense>& B, const MatrixView<float, MatrixFormat::Dense>& C,
                     float alpha, float beta) {
    constexpr og::RegCfg c = og::reg_configs[I];
    constexpr Transpose OA = (F == kTN || F == kTT) ? Transpose::Trans : Transpose::NoTrans;
    constexpr Transpose OB = (F == kNT || F == kTT) ? Transpose::Trans : Transpose::NoTrans;
    if constexpr (c.m == 128 && c.n == 128) {
        static_assert(c.threads() == 256, "the 128x128 kernel runs 256 threads");
        using Staged = Gemm128x128Body<float, false, true, OA, OB, c.stages>;
        static_assert(Staged::kVariant == reg_variant(I, F, false), "the body's trace name must match reg_variant");
        static_assert(c.slm_bytes() == std::int64_t(sizeof(float)) * (Staged::TileAElems + Staged::TileBElems),
                      "choice.hh's slm_bytes must match the 128x128 kernel's largest leg");
        // The leg is derived: the unpredicated path whenever the layout allows it. Every depth's
        // aligned leg is the two-stage body (a 16-byte copy_async aligned leg lost to it, WP-N2).
        if (can_use_128x128_fast_path<float>(A, B, C, OA, OB))
            return launch_register_128x128_k8<float, true, OA, OB>(ctx, A, B, C, alpha, beta, aligned_trace_name);
        return launch_register_128x128_k8<float, false, OA, OB, c.stages>(ctx, A, B, C, alpha, beta,
                                                                           kernel_trace_name);
    } else if constexpr ((c.m == 256 && c.n == 128) || (c.m == 128 && c.n == 256)) {
        using Body = GemmK8S4LargeBody<c.tr, c.tc, OA, OB, false>;
        static_assert(Body::TileM == c.m && Body::TileN == c.n && Body::Stages == c.stages && c.threads() == 256,
                      "choice.hh's row must match the 256x128 / 128x256 kernel");
        static_assert(Body::kVariant == reg_variant(I, F, false), "the body's trace name must match reg_variant");
        static_assert(c.slm_bytes() == std::int64_t(sizeof(float)) * Body::Elems,
                      "choice.hh's slm_bytes must match the kernel's local accessor");
        return launch_register_k8_s4_large<c.tr, c.tc, OA, OB>(ctx, A, B, C, alpha, beta, kernel_trace_name,
                                                               aligned_trace_name);
    } else {
        static_assert(reg_variant(I, F, false) != KernelVariant::Direct, "reg_variant has no name for this row");
        constexpr bool aligned_leg = F == kNN && c.aligned_leg;
        constexpr RegTile P{c.m, c.n, c.k, c.tr, c.tc, 4, 4, c.u, c.stages, OA, OB, aligned_leg};
        using Pol = RegisterTilePolicy<P.M, P.N, P.K, P.TR, P.TC>;
        static_assert(Pol::ThreadsPerGroup == c.threads(), "choice.hh's thread count must match the tile");
        static_assert(c.slm_bytes() == std::int64_t(sizeof(float)) * P.Stages * (Pol::StageASize + Pol::StageBSize),
                      "choice.hh's slm_bytes must match the tile's local accessors");
        return launch_reg<float, P>(ctx, A, B, C, alpha, beta, kernel_trace_name(reg_variant(I, F, false)),
                                    aligned_leg ? aligned_trace_name(reg_variant(I, F, true)) : nullptr);
    }
}

// ---- wide (every scalar) ---------------------------------------------------------------

// The trace identity of wide_configs[i] at form 3 op(A) + op(B), op over N, T, C. Direct marks
// a form the row does not instantiate; launch_wide_cfg static_asserts it is never read.
constexpr KernelVariant wide_variant(std::size_t i, int form) {
    using K = KernelVariant;
    constexpr K x = K::Direct;
    constexpr K table[9][9] = {
        {K::Tiled64x64RegisterK16Wide, K::Tiled64x64RegisterK16WideNT, K::Tiled64x64RegisterK16WideNC,
         K::Tiled64x64RegisterK16WideTN, K::Tiled64x64RegisterK16WideTT, K::Tiled64x64RegisterK16WideTC,
         K::Tiled64x64RegisterK16WideCN, K::Tiled64x64RegisterK16WideCT, K::Tiled64x64RegisterK16WideCC},
        {x, K::Tiled128x32RegisterK16WideNT, K::Tiled128x32RegisterK16WideNC, K::Tiled128x32RegisterK16WideTN, x, x,
         K::Tiled128x32RegisterK16WideCN, x, K::Tiled128x32RegisterK16WideCC},
        {x, K::Tiled32x128RegisterK16WideNT, K::Tiled32x128RegisterK16WideNC, K::Tiled32x128RegisterK16WideTN, x, x,
         K::Tiled32x128RegisterK16WideCN, x, K::Tiled32x128RegisterK16WideCC},
        {K::Tiled32x32RegisterK16Wide, K::Tiled32x32RegisterK16WideNT, K::Tiled32x32RegisterK16WideNC,
         K::Tiled32x32RegisterK16WideTN, K::Tiled32x32RegisterK16WideTT, K::Tiled32x32RegisterK16WideTC,
         K::Tiled32x32RegisterK16WideCN, K::Tiled32x32RegisterK16WideCT, K::Tiled32x32RegisterK16WideCC},
        {K::Tiled16x16RegisterK16Wide, K::Tiled16x16RegisterK16WideNT, K::Tiled16x16RegisterK16WideNC,
         K::Tiled16x16RegisterK16WideTN, K::Tiled16x16RegisterK16WideTT, K::Tiled16x16RegisterK16WideTC,
         K::Tiled16x16RegisterK16WideCN, K::Tiled16x16RegisterK16WideCT, K::Tiled16x16RegisterK16WideCC},
        {K::Tiled64x32RegisterK16Wide, K::Tiled64x32RegisterK16WideNT, K::Tiled64x32RegisterK16WideNC,
         K::Tiled64x32RegisterK16WideTN, x, x,
         K::Tiled64x32RegisterK16WideCN, K::Tiled64x32RegisterK16WideCT, K::Tiled64x32RegisterK16WideCC},
        {K::Tiled32x64RegisterK16Wide, K::Tiled32x64RegisterK16WideNT, K::Tiled32x64RegisterK16WideNC,
         K::Tiled32x64RegisterK16WideTN, x, x,
         K::Tiled32x64RegisterK16WideCN, K::Tiled32x64RegisterK16WideCT, K::Tiled32x64RegisterK16WideCC},
        {K::Tiled64x64RegisterK8Wide, K::Tiled64x64RegisterK8WideNT, K::Tiled64x64RegisterK8WideNC,
         K::Tiled64x64RegisterK8WideTN, K::Tiled64x64RegisterK8WideTT, K::Tiled64x64RegisterK8WideTC,
         K::Tiled64x64RegisterK8WideCN, K::Tiled64x64RegisterK8WideCT, K::Tiled64x64RegisterK8WideCC},
        {K::Tiled128x128RegisterK8Wide, K::Tiled128x128RegisterK8WideNT, K::Tiled128x128RegisterK8WideNC,
         K::Tiled128x128RegisterK8WideTN, K::Tiled128x128RegisterK8WideTT, K::Tiled128x128RegisterK8WideTC,
         K::Tiled128x128RegisterK8WideCN, K::Tiled128x128RegisterK8WideCT, K::Tiled128x128RegisterK8WideCC},
    };
    static_assert(std::size(table) == og::wide_configs.size(), "one trace row per wide_configs row");
    return table[i][form];
}

constexpr Transpose kWideOps[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};

template <typename T, int I, Transpose OA, Transpose OB>
Event launch_wide_cfg(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha, T beta) {
    constexpr og::WideCfg c = og::wide_configs[I];
    constexpr int fa = static_cast<int>(OA == Transpose::Trans) + 2 * static_cast<int>(OA == Transpose::ConjTrans);
    constexpr int fb = static_cast<int>(OB == Transpose::Trans) + 2 * static_cast<int>(OB == Transpose::ConjTrans);
    static_assert(og::wide_compiles<T>(c, fa, fb), "wide_configs does not list this form and scalar for this row");
    if constexpr (og::wide_v2_row<T>(c)) {
        constexpr KernelVariant v = wide_variant(I, 3 * fa + fb);
        using Body = GemmWide64Body<T, OA, OB, true, c.k, c.m, c.n>;
        static_assert((c.k == 16 || c.k == 8) && c.ttm == Body::TTM && c.ttn == Body::TTN,
                      "wide v2 is a k = 16 or 8 slab over a 16 x 16 lane grid");
        using BodyE = GemmWide64Body<T, OA, OB, false, c.k, c.m, c.n>;
        constexpr std::int64_t padded = std::int64_t(2) * c.k * (c.m + c.n + 8 / Body::CF) * Body::CF * 4;
        constexpr std::int64_t vec = std::int64_t(2) * (Body::BufA + Body::BufB) * 4;
        static_assert(og::wide_slm_bytes<T>(c) == (Body::CF == 4 ? vec : padded) &&
                          og::wide_slm_bytes<T>(c) >= std::int64_t(2) * (BodyE::BufA + BodyE::BufB) * 4,
                      "can_run's wide_slm_bytes must bound wide v2's two double-buffered tiles on both legs");
        return launch_wide64_v2<T, OA, OB, c.k, c.m, c.n>(ctx, A, B, C, alpha, beta, kernel_trace_name(v),
                                                          aligned_trace_name(v), stages3_trace_name(v, false),
                                                          stages3_trace_name(v, true));
    } else if constexpr (fa == 0 && fb == 0 && c.m == 64 && c.n == 64 && c.k == 16) {
        // The double NN 64x64x16 tile is its own kernel; its leg is derived from the layout.
        // Double 64x64x8 runs launch_wide_transposed in every form, NN included.
        if (can_use_64x64_k16_wide_fast_path<T>(A, B, C))
            return launch_register_64x64_k16_wide<T, true>(ctx, A, B, C, alpha, beta, aligned_trace_name);
        return launch_register_64x64_k16_wide<T, false>(ctx, A, B, C, alpha, beta, kernel_trace_name);
    } else {
        constexpr KernelVariant v = wide_variant(I, 3 * fa + fb);
        return launch_wide_transposed<T, WideTile{c.m, c.n, c.k, c.ttm, c.ttn, OA, OB, og::wide_stages<T>(c)}>(
            ctx, A, B, C, alpha, beta, kernel_trace_name(v), {}, async_trace_name(v, false, false),
            async_trace_name(v, false, true));
    }
}

// Form F = 3 op(A) + op(B) of wide_configs[I], launched when it serves the call. A real scalar
// compiles only the N/C forms, and its C build serves a real T (og::wide_serves).
// evidence: docs/perf/gemm.md#gemm-wide-instantiations-in-every-transpose-form
// Split: the split-k body of a host row (always launch_wide_transposed), C the partials.
template <typename T, int I, int F, bool Split = false>
void launch_wide_form(std::optional<Event>& out, Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha, T beta, Transpose transA, Transpose transB, WideSplit split = {}) {
    constexpr og::WideCfg c = og::wide_configs[I];
    constexpr int fa = F / 3, fb = F % 3;
    if constexpr (og::wide_compiles<T>(c, fa, fb) && (!Split || og::splitk_host(c))) {
        constexpr Transpose OA = kWideOps[fa], OB = kWideOps[fb];
        if (out || !wide_trans_matches<T>(transA, OA) || !wide_trans_matches<T>(transB, OB)) return;
        if constexpr (Split)
            out = launch_wide_transposed<T, WideTile{c.m, c.n, c.k, c.ttm, c.ttn, OA, OB}, true>(
                ctx, A, B, C, alpha, beta, splitk_trace_name(wide_variant(I, F)), split,
                async_trace_name(wide_variant(I, F), true, false), async_trace_name(wide_variant(I, F), true, true));
        else
            out = launch_wide_cfg<T, I, OA, OB>(ctx, A, B, C, alpha, beta);
    }
}
} // namespace

// One half of a wide_configs row (Half 0: forms 0-4, Half 1: forms 5-8) or one reg_configs row, each
// explicitly instantiated in exactly one generated shard TU (BATCHLAS_GEMM_SHARDS,
// src/ops/CMakeLists.txt), so the device link splits gemm into many modules.
// evidence: docs/developer/agent-guide.md section 7 (Build Performance)
template <typename T, int I, bool Split, int Half>
void gemm_wide_row(std::optional<Event>& out, Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB,
                   WideSplit split);

template <int I>
void gemm_reg_row(std::optional<Event>& out, Queue& ctx, const MatrixView<float, MatrixFormat::Dense>& A,
                  const MatrixView<float, MatrixFormat::Dense>& B,
                  const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta, bool ta, bool tb);

} // namespace batchlas::sycl_gemm
