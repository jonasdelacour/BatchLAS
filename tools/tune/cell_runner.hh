#pragma once

// The child side of one cell (docs/design/flat-kernel-selection.md §6.1, §6.3): probe every
// arm through the public sizing call under its ScopedPin (a refused pin is "skipped", never
// timed), then warm up and time the arms interleaved, rotating the order every rep, then verify
// each arm from one more untimed run. ONE CELL PER PROCESS: the SLM carve-out is sticky per
// CUfunction (benchmarks/factor_bench.cc header), so the driver forks a child per cell.
//
// Problem<T> supplies: std::size_t workspace() (the public *_buffer_size; throws
// std::invalid_argument when the current pin cannot run), void reset() (restores the inputs,
// synchronously), void clear_info(), void run(Span<std::byte>) (the public op, then wait), and
// std::pair<double, int> verify() (residual on items 0 and batch-1, nonzero info count).

#include "spec.hh"

#include <batchlas/backend_config.h>
#include <batchlas/util/sycl-vector.hh>

#include "../../src/select/select.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace batchlas::tune {

// The vendor arm is whichever GPU library this build links. Host (cpu) tables: phase 4.
#if BATCHLAS_HAS_CUDA_BACKEND
inline constexpr Backend kBackend = Backend::CUDA;
#else
inline constexpr Backend kBackend = Backend::ROCM;
#endif

template <class F>
decltype(auto) with_dtype(const std::string& dtype, F&& f) {
    if (dtype == "float") return f.template operator()<float>();
    if (dtype == "double") return f.template operator()<double>();
    if (dtype == "cfloat") return f.template operator()<std::complex<float>>();
    if (dtype == "cdouble") return f.template operator()<std::complex<double>>();
    throw std::invalid_argument("unknown dtype '" + dtype + "'");
}

template <class Choice, std::size_t N>
std::vector<std::string> spellings(const std::array<Choice, N>& candidates) {
    std::vector<std::string> out;
    for (const Choice& c : candidates) out.push_back(select::to_string(c));
    return out;
}

// A coverage `reached` row (chosen_origin, chosen_algo) -> a pin spelling for this op. Old
// RouteTable rows read native:<algo> or vendor:auto; flat-selection rows carry the spelling.
template <class Choice>
std::string normalize_route(const select::Rules& rules, const std::string& origin, const std::string& algo) {
    if (origin == "vendor") return "vendor";
    const std::string legacy = origin + ":" + algo;
    for (const select::Alias& a : rules.aliases)
        if (legacy == a.name) return std::string(a.spelling);
    if (auto c = select::parse<Choice>(algo)) return select::to_string(*c);
    return legacy;
}

template <class Choice, class Problem>
std::vector<ArmOutcome> run_arms(const std::string& op, Problem& p, const CellRequest& req, double tol) {
    using clock = std::chrono::steady_clock;
    std::vector<ArmOutcome> arms;
    for (const std::string& w : req.arms) arms.push_back({w, "ok", "", {}, {}, 0.0, 0});
    auto fail = [](ArmOutcome& a, const char* status, const std::string& why) {
        a.status = status;
        a.reason = why;
    };
    // Every call goes through here: a refused pin is "skipped", anything else "error".
    auto guarded = [&](ArmOutcome& a, auto&& f) {
        if (a.status != "ok") return false;
        try {
            select::ScopedPin<Choice> pin(op, a.arm);
            f();
            return true;
        } catch (const std::invalid_argument& e) {
            fail(a, "skipped", std::string("pin refused: ") + e.what());
        } catch (const std::exception& e) {
            fail(a, "error", e.what());
        }
        return false;
    };

    std::size_t wneed = 1;
    for (ArmOutcome& a : arms) guarded(a, [&] { wneed = std::max(wneed, p.workspace()); });
    UnifiedVector<std::byte> ws(wneed);
    auto once = [&](ArmOutcome& a) {
        return guarded(a, [&] {
            p.reset();
            p.run(ws.to_span());
        });
    };

    if (req.mode == "jit" || req.mode == "probe") {
        for (ArmOutcome& a : arms) once(a);
        return arms;
    }

    std::vector<std::size_t> base(arms.size());
    for (std::size_t i = 0; i < base.size(); ++i) base[i] = req.reverse ? base.size() - 1 - i : i;
    // Warm-up interleaved in the timed order: a per-arm warm-up made arm 0's first timed rep
    // 2.2x slow (evidence: docs/perf/small-n-baseline.md#warm-up-order-and-the-variance-gate).
    const std::size_t live = std::count_if(arms.begin(), arms.end(), [](const auto& a) { return a.status == "ok"; });
    const auto w0 = clock::now();
    do {
        for (std::size_t i : base) once(arms[i]);
    } while (std::chrono::duration<double>(clock::now() - w0).count() < req.warm_s * double(live));

    for (int r = 0; r < req.reps; ++r) {
        const auto order = rep_order(arms.size(), r, req.reverse);
        for (std::size_t pos = 0; pos < order.size(); ++pos) {
            ArmOutcome& a = arms[order[pos]];
            double ms = 0;
            const bool ran = guarded(a, [&] {
                p.reset();
                const auto t0 = clock::now();
                p.run(ws.to_span());
                ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
            });
            if (!ran) continue;
            a.ms.push_back(ms);
            a.slot.push_back(static_cast<int>(pos));
        }
    }

    // Correctness in the same process, from one more untimed run: a fast wrong answer is "bad".
    for (ArmOutcome& a : arms) {
        p.clear_info();
        if (!once(a)) continue;
        const auto [res, info] = p.verify();
        a.residual = res;
        a.info_nonzero = info;
        if (!std::isfinite(res)) fail(a, "bad", "residual_nonfinite");
        else if (res > tol) fail(a, "bad", "residual");
        else if (info != 0) fail(a, "bad", "info");
    }
    return arms;
}

}  // namespace batchlas::tune
