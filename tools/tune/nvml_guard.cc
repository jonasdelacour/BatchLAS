// SPDX-License-Identifier: MIT
#include <tools/tune/nvml_guard.hh>

#include <dlfcn.h>

#include <chrono>
#include <cstdlib>

namespace batchlas::tune::nvml {

namespace {

// Layouts from nvml.h (API 13; unchanged since driver 510). v2 and v3 share nvmlProcessInfo_v2_t.
using Ret = int;
constexpr Ret kSuccess = 0, kNotFound = 6, kInsufficientSize = 7;
struct DeviceSt;
using Device = DeviceSt*;
struct ProcessInfo {
    unsigned pid;
    unsigned long long used_memory;
    unsigned gpu_instance, compute_instance;
};
struct ProcessInfoV1 {
    unsigned pid;
    unsigned long long used_memory;
};
struct Utilization {
    unsigned gpu, memory;
};
struct ProcessSample {
    unsigned pid;
    unsigned long long time_stamp;
    unsigned sm, mem, enc, dec;
};

struct Lib {
    std::string error;
    Ret (*by_index)(unsigned, Device*) = nullptr;
    Ret (*procs)(Device, unsigned*, ProcessInfo*) = nullptr;
    Ret (*procs_v1)(Device, unsigned*, ProcessInfoV1*) = nullptr;
    Ret (*util)(Device, Utilization*) = nullptr;
    Ret (*proc_util)(Device, ProcessSample*, unsigned*, unsigned long long) = nullptr;
    Ret (*uuid)(Device, char*, unsigned) = nullptr;
    Ret (*clock_info)(Device, int, unsigned*) = nullptr;               // optional: clocks()
    Ret (*event_reasons)(Device, unsigned long long*) = nullptr;       // optional: clocks()
};

template <typename F>
bool sym(void* h, const char* name, F& f) {
    f = reinterpret_cast<F>(::dlsym(h, name));
    return f != nullptr;
}

// Leaked on purpose: no nvmlShutdown or dlclose from a static destructor.
const Lib& lib() {
    static const Lib* l = [] {
        auto* l = new Lib;
        void* h = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            const char* e = ::dlerror();
            l->error = std::string("dlopen libnvidia-ml.so.1: ") + (e ? e : "failed");
            return l;
        }
        Ret (*init)() = nullptr;
        sym(h, "nvmlDeviceGetClockInfo", l->clock_info);
        if (!sym(h, "nvmlDeviceGetCurrentClocksEventReasons", l->event_reasons))
            sym(h, "nvmlDeviceGetCurrentClocksThrottleReasons", l->event_reasons);
        if (!sym(h, "nvmlDeviceGetComputeRunningProcesses_v3", l->procs))
            sym(h, "nvmlDeviceGetComputeRunningProcesses_v2", l->procs);
        if (!l->procs) sym(h, "nvmlDeviceGetComputeRunningProcesses", l->procs_v1);
        if (!sym(h, "nvmlInit_v2", init) || !sym(h, "nvmlDeviceGetHandleByIndex_v2", l->by_index) ||
            !sym(h, "nvmlDeviceGetUtilizationRates", l->util) || !sym(h, "nvmlDeviceGetProcessUtilization", l->proc_util) ||
            !sym(h, "nvmlDeviceGetUUID", l->uuid) || (!l->procs && !l->procs_v1))
            l->error = "libnvidia-ml.so.1 lacks a symbol the guard needs";
        else if (const Ret r = init(); r != kSuccess)
            l->error = "nvmlInit_v2 returned " + std::to_string(r);
        return l;
    }();
    return *l;
}

std::optional<Device> device(int gpu) {
    Device d = nullptr;
    if (!load_error().empty() || gpu < 0 || lib().by_index(unsigned(gpu), &d) != kSuccess) return std::nullopt;
    return d;
}

template <typename Info, typename Fn>
std::optional<std::vector<std::string>> pids_with(Fn fn, Device d) {
    for (unsigned cap = 64; cap <= 65536; cap *= 4) {
        std::vector<Info> buf(cap);
        unsigned n = cap;
        const Ret r = fn(d, &n, buf.data());
        if (r == kInsufficientSize) continue;
        if (r != kSuccess) return std::nullopt;
        std::vector<std::string> out;
        for (unsigned i = 0; i < n && i < cap; ++i) out.push_back(std::to_string(buf[i].pid));
        return out;
    }
    return std::nullopt;
}

}  // namespace

const std::string& load_error() { return lib().error; }

std::optional<std::vector<std::string>> compute_pids(int gpu) {
    const auto d = device(gpu);
    if (!d) return std::nullopt;
    return lib().procs ? pids_with<ProcessInfo>(lib().procs, *d) : pids_with<ProcessInfoV1>(lib().procs_v1, *d);
}

std::optional<double> utilization(int gpu) {
    const auto d = device(gpu);
    Utilization u{};
    if (!d || lib().util(*d, &u) != kSuccess) return std::nullopt;
    return double(u.gpu);
}

std::optional<std::vector<ProcUtil>> process_util(int gpu, unsigned long long since_us) {
    const auto d = device(gpu);
    if (!d) return std::nullopt;
    for (int tries = 0; tries < 4; ++tries) {
        unsigned n = 0;
        Ret r = lib().proc_util(*d, nullptr, &n, since_us);  // the documented size probe
        if (r == kNotFound || (r == kSuccess && n == 0)) return std::vector<ProcUtil>{};
        if (r != kInsufficientSize && r != kSuccess) return std::nullopt;  // some drivers answer the probe SUCCESS, n > 0
        std::vector<ProcessSample> buf(n + 8);
        n = unsigned(buf.size());
        r = lib().proc_util(*d, buf.data(), &n, since_us);
        if (r == kNotFound) return std::vector<ProcUtil>{};
        if (r == kInsufficientSize) continue;  // processes appeared between the calls
        if (r != kSuccess) return std::nullopt;
        std::vector<ProcUtil> out;
        for (unsigned i = 0; i < n && i < buf.size(); ++i) out.push_back({std::to_string(buf[i].pid), double(buf[i].sm)});
        return out;
    }
    return std::nullopt;
}

std::optional<std::string> uuid(int gpu) {
    const auto d = device(gpu);
    char buf[96] = {};  // NVML_DEVICE_UUID_V2_BUFFER_SIZE
    if (!d || lib().uuid(*d, buf, sizeof(buf)) != kSuccess) return std::nullopt;
    return std::string(buf);
}

std::optional<Clocks> clocks(int gpu) {
    constexpr int kClockSm = 1;  // NVML_CLOCK_SM
    const auto d = device(gpu);
    Clocks c;
    if (!d || !lib().clock_info || !lib().event_reasons || lib().clock_info(*d, kClockSm, &c.sm_mhz) != kSuccess ||
        lib().event_reasons(*d, &c.reasons) != kSuccess)
        return std::nullopt;
    return c;
}

int visible_gpu() {
    const char* v = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!v || !*v) return -1;
    char* end = nullptr;
    const long g = std::strtol(v, &end, 10);
    return *end == '\0' && g >= 0 ? int(g) : -1;
}

unsigned long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace batchlas::tune::nvml
