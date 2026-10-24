// SPDX-License-Identifier: MIT
#pragma once

// The guard's NVML source (dlopen, no nvml.h, no CUDA context); nullopt = the query failed.
// evidence: docs/design/tiered-tuning.md#engine-guard-retries-a-failed-nvidia-smi-query

#include <tools/tune/tune_core.hh>

#include <optional>
#include <string>
#include <vector>

namespace batchlas::tune::nvml {

const std::string& load_error();  // "" once loaded and initialised; the first call loads (never unloaded)
std::optional<std::vector<std::string>> compute_pids(int gpu);
std::optional<double> utilization(int gpu);  // percent, NVML's last sample period
std::optional<std::vector<ProcUtil>> process_util(int gpu, unsigned long long since_us);  // since_us: epoch microseconds, as now_us()
std::optional<std::string> uuid(int gpu);

struct Clocks {
    unsigned sm_mhz = 0;
    unsigned long long reasons = 0;  // nvmlClocksEventReason* bits
};
inline constexpr unsigned long long kSwPowerCap = 0x4;  // nvmlClocksEventReasonSwPowerCap
std::optional<Clocks> clocks(int gpu);  // SM clock and the active clock-event reasons now
// The NVML index of this process's one GPU: CUDA_VISIBLE_DEVICES as a single index (the driver
// sets it with CUDA_DEVICE_ORDER=PCI_BUS_ID, NVML's order), else -1.
int visible_gpu();
unsigned long long now_us();

}  // namespace batchlas::tune::nvml
