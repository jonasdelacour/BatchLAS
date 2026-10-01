#pragma once

// A route's LAUNCH PLAN at one shape, and the analytic cost it feeds. SYCL-free and constexpr
// so the launcher that decides the geometry, the shape builder and the plan-dump tool read one
// definition. The constants are fitted per (profile, op, route, scalar) by
// evaluation/routing/fit.py. evidence: docs/design/routing-cost-model.md

#include "resident_capacity.hh"

#include <cstddef>
#include <cstdint>

namespace batchlas::launch_plan {

// What a plan reads of the device. The launchers fill only the first two: geometry never
// depends on the rest, which feed the occupancy estimate and the cost. 0 means unknown.
// NOT dispatch::DeviceFacts (routing facts, public, memoized): merge when routing reads plans.
struct DeviceFacts {
    std::size_t local_mem_bytes = 0;   // DeviceProperty::LOCAL_MEM_SIZE
    int max_wg_size = 0;               // DeviceProperty::MAX_WORK_GROUP_SIZE
    int compute_units = 0;
    int max_threads_per_cu = 0;
    int max_groups_per_cu = 0;
};

struct LaunchPlan {
    bool fits = false;
    int launches = 0;               // submissions per call, fills included
    int wave_launches = 0;          // of those, the ones carrying the batch grid below
    std::int64_t groups = 0;        // work-groups per wave launch
    int wg_size = 0;
    std::size_t slm_per_group = 0;  // bytes, after any hole pad
    int regs_per_item = 0;          // probed demand from the profile data; 0 = unknown
    int resident_groups_per_cu = 0;
    double flops = 0;               // what the schedule executes
    double useful_flops = 0;        // LAPACK's count for the order, x4 for complex
    double bytes = 0;               // global traffic
    std::int64_t serial_steps = 0;  // dependent steps on one group's critical path, per call
    double slot_chain = 0;          // per-lane barrier chain for the slot term; 0 = no slot term
    bool batch_wide_work = false;   // flops/bytes are sub-op work spread over the whole GPU
};

// Groups one compute unit can hold: the tightest of the SLM, thread, register and group caps
// that are known. The register cap is resident::'s per-sub-partition rule, not regs x wg.
constexpr int resident_groups_per_cu(const DeviceFacts& d, std::size_t slm_per_group,
                                     int wg_size, int regs_per_item) {
    if (wg_size < 1) return 0;
    long long best = -1;
    auto cap = [&best](long long v) { best = (best < 0 || v < best) ? v : best; };
    if (slm_per_group > 0) {
        cap(static_cast<long long>(resident::device_slm_budget(d.local_mem_bytes) / slm_per_group));
    }
    if (d.max_threads_per_cu > 0) cap(d.max_threads_per_cu / wg_size);
    if (regs_per_item > 0) {
        const int alloc = resident::sm89_alloc_regs(regs_per_item);
        const long long warps_per_cu = static_cast<long long>(resident::kPartitionsPerBlock) *
            (resident::kRegsPerPartition / (resident::kLanesPerWarp * alloc));
        const long long warps = (wg_size + resident::kLanesPerWarp - 1) / resident::kLanesPerWarp;
        cap(warps_per_cu / warps);
    }
    if (d.max_groups_per_cu > 0) cap(d.max_groups_per_cu);
    if (best < 0) return 1;  // nothing known binds: one group is the honest floor
    return best < 1 ? 1 : static_cast<int>(best);
}

// Inverse rates, so the cost is linear in every constant except through the max.
struct CostConstants {
    double t_launch = 0;      // seconds per submission
    double s_per_flop = 0;    // seconds per flop for ONE resident group (1 / F_route)
    double s_per_byte = 0;    // seconds per byte for ONE resident group (1 / B_route)
    double t_step = 0;        // seconds per serial step of one group
    double s_per_slot = 0;    // seconds per warp-slot of a CU's lane chain
};

// The plan-dependent half of the cost, one coefficient per constant. The fitter reads these
// (potrf_plan_dump prints them) so the wave arithmetic exists only here.
struct CostTerms {
    double launch = 0;   // x t_launch
    double flop = 0;     // x s_per_flop, inside the max
    double byte = 0;     // x s_per_byte, inside the max
    double step = 0;     // x t_step
    double slot = 0;     // x s_per_slot, inside the max
};

// Throughput is per CU and residency-free: a CU executes load = wave_launches*ceil(groups/CUs)
// groups, so flop and byte are that load times one group's share (or the batch-wide totals for
// sub-op work). Latency is per wave: step = waves * steps per wave launch, waves counted with
// the register-aware residency. slot = load * warps per group * the lane chain.
// evidence: docs/perf/potrf.md#launch-plans
constexpr CostTerms cost_terms(const LaunchPlan& p, const DeviceFacts& d) {
    CostTerms t;
    t.launch = p.launches;
    if (p.wave_launches < 1 || p.groups < 1) return t;
    const long long cus = d.compute_units > 0 ? d.compute_units : 1;
    const long long res = p.resident_groups_per_cu > 0 ? p.resident_groups_per_cu : 1;
    const long long per_wave = res * cus;
    const double waves =
        static_cast<double>(p.wave_launches) * static_cast<double>((p.groups + per_wave - 1) / per_wave);
    const double load =
        static_cast<double>(p.wave_launches) * static_cast<double>((p.groups + cus - 1) / cus);
    const double denom = static_cast<double>(p.groups) * p.wave_launches;
    t.flop = p.batch_wide_work ? p.flops : load * p.flops / denom;
    t.byte = p.batch_wide_work ? p.bytes : load * p.bytes / denom;
    t.step = waves * static_cast<double>(p.serial_steps) / p.wave_launches;
    const double warps = static_cast<double>((p.wg_size + 31) / 32);
    t.slot = load * warps * p.slot_chain;
    return t;
}

// t = t_launch*launch + max(s_per_flop*flop, s_per_byte*byte, s_per_slot*slot) + t_step*step.
// evaluation/routing/fit.py's `combine` is this line; potrf_plan_tests pins the two together.
constexpr double combine(const CostTerms& t, const CostConstants& c) {
    const double tf = t.flop * c.s_per_flop;
    const double tb = t.byte * c.s_per_byte;
    const double ts = t.slot * c.s_per_slot;
    const double m = tf > tb ? tf : tb;
    return c.t_launch * t.launch + (m > ts ? m : ts) + c.t_step * t.step;
}

constexpr double cost(const LaunchPlan& p, const DeviceFacts& d, const CostConstants& c) {
    return combine(cost_terms(p, d), c);
}

// The box a constant set was fitted on (profile.json "support"). Constants borrowed by a
// fallback carry an empty box (rows == 0), so every prediction from them is extrapolated.
struct SupportRegion {
    std::int64_t n_min = 0, n_max = -1;
    std::int64_t batch_min = 0, batch_max = -1;
    int rows = 0;
};

struct Prediction {
    double seconds = 0;
    bool extrapolated = true;
};

constexpr bool extrapolated(const SupportRegion& s, std::int64_t n, std::int64_t batch) {
    return s.rows < 1 || n < s.n_min || n > s.n_max || batch < s.batch_min || batch > s.batch_max;
}

constexpr Prediction predict(const LaunchPlan& p, const DeviceFacts& d, const CostConstants& c,
                             const SupportRegion& s, std::int64_t n, std::int64_t batch) {
    return {cost(p, d, c), extrapolated(s, n, batch)};
}

}  // namespace batchlas::launch_plan
