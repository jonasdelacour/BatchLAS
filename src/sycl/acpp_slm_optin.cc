// BATCHLAS_ACPP_SLM_OPTIN only: libbatchlas_acpp_slm_optin.so. AdaptiveCpp 25.10 never raises a
// kernel's dynamic local memory limit, so on CUDA every launch above 48 KiB fails (CU:1). This
// library interposes the driver's cuLaunchKernel (ELF symbol interposition: it is DT_NEEDED
// ahead of libcuda.so.1, which acpp's CUDA backend resolves against the global scope) and opts
// the CUfunction in first. local_mem.hh then reports the driver's opt-in maximum as the budget.
// evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
#include <cuda.h>
#include <dlfcn.h>

#include <mutex>
#include <unordered_map>

namespace {

constexpr unsigned kDefaultDynamicLimit = 48u * 1024u;

using LaunchFn = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                              unsigned, CUstream, void**, void**);

// Largest limit already set per function: cuFuncSetAttribute is a driver call, a launch is hot.
CUresult opt_in(CUfunction f, unsigned shmem) {
    static auto* m = new std::mutex;
    static auto* done = new std::unordered_map<CUfunction, unsigned>;
    std::lock_guard<std::mutex> lock(*m);
    unsigned& have = (*done)[f];
    if (have >= shmem) return CUDA_SUCCESS;
    const CUresult r =
        cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(shmem));
    if (r == CUDA_SUCCESS) have = shmem;
    return r;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy,
                                                               unsigned gz, unsigned bx, unsigned by,
                                                               unsigned bz, unsigned shmem, CUstream s,
                                                               void** params, void** extra) {
    static const auto real = reinterpret_cast<LaunchFn>(dlsym(RTLD_NEXT, "cuLaunchKernel"));
    if (real == nullptr) return CUDA_ERROR_NOT_FOUND;
    if (shmem > kDefaultDynamicLimit) {
        const CUresult r = opt_in(f, shmem);
        if (r != CUDA_SUCCESS) return r;
    }
    return real(f, gx, gy, gz, bx, by, bz, shmem, s, params, extra);
}

// Opt-in maximum dynamic local memory per block of CUDA device `ordinal`, or 0 if unknown.
__attribute__((visibility("default"))) int batchlas_acpp_slm_optin_bytes(int ordinal) {
    CUdevice dev{};
    int bytes = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, ordinal) != CUDA_SUCCESS) return 0;
    if (cuDeviceGetAttribute(&bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev) != CUDA_SUCCESS)
        return 0;
    return bytes;
}

}  // extern "C"
