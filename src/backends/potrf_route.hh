#pragma once

// POTRF shape builder: device and environment queries live here so the route table reads
// only its arguments. Do not add src/queue.hh or <sycl/sycl.hpp> -- the vendor-free facade
// includes this. evidence: docs/perf/potrf.md#what-ships

#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_potrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/potrf_launch_plan.hh"
#include "../extensions/potrf_native.hh"
#include "../sycl/trsm_native.hh"
#include "potrf_profile_constants.hh"

#include <complex>
#include <type_traits>

#include <cstddef>
#include <optional>

namespace batchlas::backend {

// Everything but the device queries, so tools/potrf_plan_dump builds the SAME shape from a
// described device. evidence: docs/perf/potrf.md#launch-plans
template <Backend B, typename T>
inline dispatch::PotrfShape potrf_op_shape_from_facts(const launch_plan::DeviceFacts& d,
                                                      int64_t rows, int64_t cols, int64_t batch,
                                                      Uplo uplo, bool is_gpu, bool has_sg32,
                                                      bool heterogeneous, bool blocked_available) {
    dispatch::PotrfShape s;
    s.op = dispatch::Op::potrf;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;

    // m and n stay separate so the `m == n` gate is representable; k is the order.
    s.m = rows;
    s.n = cols;
    s.k = rows;
    s.batch = batch;
    s.uplo = uplo;
    s.is_gpu = is_gpu;
    s.has_sg32 = has_sg32;  // enumerated, not `>= 32`

    // Not dead code: this becomes a correctness gate the moment the CTA kernel lands.
    s.heterogeneous_batch = heterogeneous;

    // Query THIS device; a hardcoded budget admits a route that cannot launch.
    // evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
    const std::size_t budget = resident::device_slm_budget(d.local_mem_bytes);
    s.cta_max_n = sycl_potrf::potrf_cta_max_n_for_slm<T>(budget);
    s.lpanel_max_n = sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, d.max_wg_size);  // SLM and max WG

    // No budget argument: the tiny tier holds the matrix in REGISTERS and allocates no
    // local memory, so its ceiling is a compile-time constant of the type alone.
    s.tiny_max_n = sycl_potrf::potrf_tiny_max_n<T>();
    s.blocked_available = blocked_available;
    return s;
}

template <typename T>
constexpr int potrf_dtype_index() {
    if constexpr (std::is_same_v<T, float>) return 0;
    else if constexpr (std::is_same_v<T, double>) return 1;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return 2;
    else return 3;
}

// Prices every kPotrfOrder route with the SAME plan functions and cost_terms/combine that
// evaluation/routing/fit.py fitted, when the shape's profile passed its ship gate. `d` carries
// the queried facts (local memory, max work-group, CUs); the thread/group caps and register
// counts come from the profile. Leaves model_enabled false -- the hand windows -- otherwise,
// and also when the vendor has no constants. evidence: docs/perf/potrf.md#the-cost-model-route-choice
template <typename T>
inline void potrf_price_routes(dispatch::PotrfShape& s, launch_plan::DeviceFacts d,
                               int leaf_trsm, int nb_env, int w_env) {
    const potrf_profile::Profile* prof = potrf_profile::profile_for(s.profile);
    if (!prof || !prof->model_enabled || !s.is_gpu || s.m != s.n || s.order() < 1 ||
        s.batch < 1) {
        return;
    }
    constexpr int ti = potrf_dtype_index<T>();
    const int ui = (s.uplo == Uplo::Upper) ? 1 : 0;
    d.max_threads_per_cu = prof->max_threads_per_cu;
    d.max_groups_per_cu = prof->max_groups_per_cu;
    const potrf_plan::KernelRegs& regs = prof->regs[ti];
    const int n = static_cast<int>(s.order());
    const std::int64_t b = s.batch;
    const launch_plan::LaunchPlan plans[5] = {
        potrf_plan::tiny_plan<T>(n, b, d, regs),
        potrf_plan::cta_plan<T>(n, b, d, regs),
        potrf_plan::lpanel_plan<T>(n, b, d, regs),
        potrf_plan::blocked_plan<T>(n, b, d, leaf_trsm, regs, nb_env, w_env),
        potrf_plan::vendor_pseudo_plan<T>(n, b),
    };
    for (int i = 0; i < 5; ++i) {
        const potrf_profile::Entry& e = prof->e[i][ti][ui];
        if (!e.present || !plans[i].fits) continue;
        const auto pr = launch_plan::predict(plans[i], d, e.c, e.box, n, b);
        s.model_cost[i] = pr.seconds;
        s.model_extrapolated[i] = pr.extrapolated;
    }
    if (s.model_cost[4] < 0) return;
    s.model_enabled = true;
    s.model_margin = prof->margin;
}

template <Backend B, typename T>
inline std::optional<dispatch::PotrfShape> potrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo) {

    if (A.rows() != A.cols()) return std::nullopt;
    auto s = potrf_op_shape_from_facts<B, T>(
        sycl_potrf::potrf_device_facts(ctx.device()), A.rows(), A.cols(), A.batch_size(), uplo,
        ctx.device().type == DeviceType::GPU, ctx.device().supports_sub_group_size(32),
        A.is_heterogeneous(), sycl_potrf::potrf_blocked_available<T>());
    batchlas::dispatch::fill_device_facts(s, ctx);
    launch_plan::DeviceFacts d = sycl_potrf::potrf_device_facts(ctx.device());
    d.compute_units = s.compute_units;
    int nb_env = 0, w_env = 0;
    sycl_potrf::potrf_blocked_overrides(nb_env, w_env);
    potrf_price_routes<T>(s, d, sycl_trsm::trsm_cta_max_n<T>(), nb_env, w_env);
    return s;
}

// The only env read on this path, and potrf / potrf_buffer_size must call it with
// IDENTICAL arguments or the reported size will not match the call.
template <Backend B, typename T>
inline dispatch::Route potrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo,
    bool vendor_available) {

    const auto shape = potrf_op_shape<B, T>(ctx, A, uplo);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::potrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::potrf);
    return dispatch::resolve_potrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
