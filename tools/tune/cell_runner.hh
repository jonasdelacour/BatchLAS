#pragma once

// The child side of one cell (docs/design/flat-kernel-selection.md §6.1, §6.3): probe every
// arm through the public sizing call under its ScopedPin (a refused pin is "skipped", never
// timed), then warm up and time the arms interleaved, rotating the order every rep, then verify
// each arm from one more untimed run. The SLM carve-out is sticky per CUfunction
// (benchmarks/factor_bench.cc header), so a process that has run a larger launch can accept one a
// fresh process refuses: the custom protocol forks a child per cell, and the tiered worker takes
// its cells in ascending bytes and is audited against fresh children
// (evidence: docs/design/tiered-tuning.md#engine-persistent-workers-and-the-carve-out-audit).
//
// Problem<T> supplies: std::size_t workspace() (the public *_buffer_size; throws
// std::invalid_argument when the current pin cannot run), void reset() (restores the inputs,
// synchronously), void clear_info(), void run(Span<std::byte>) (the public op, then wait), and
// std::pair<double, int> verify() (residual on items 0 and batch-1, nonzero info count).

#include "race.hh"
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
#include <memory>
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

// symm, syrk and syr2k are real-only (RealScalar): no complex instantiation and no complex table.
template <class F>
decltype(auto) with_real_dtype(const std::string& op, const std::string& dtype, F&& f) {
    if (dtype == "float") return f.template operator()<float>();
    if (dtype == "double") return f.template operator()<double>();
    throw std::invalid_argument(op + " is real-only: no '" + dtype + "' (float, double)");
}

// A non-lattice grid() (form derived from the extents) under `--grid name=v1:v2`: the overrides
// FILTER the declared cells, as gemm's do.
inline std::vector<CellKey> filter_cells(const OpSpec& s, std::vector<CellKey> cells,
                                         const std::map<std::string, std::vector<std::string>>& overrides) {
    const auto ax = s.axes();
    for (const auto& [name, values] : overrides)
        if (std::none_of(ax.begin(), ax.end(), [&](const auto& a) { return a.first == name; }))
            throw std::invalid_argument(s.op() + " has no grid axis '" + name + "'");
    std::vector<CellKey> out;
    for (CellKey& key : cells)
        if (std::all_of(key.begin(), key.end(), [&](const auto& kv) {
                const auto it = overrides.find(kv.name);
                return it == overrides.end() || std::find(it->second.begin(), it->second.end(), kv.value) != it->second.end();
            }))
            out.push_back(std::move(key));
    if (out.empty()) throw std::invalid_argument(s.op() + ": the --grid filters leave no cell of the declared grid");
    return out;
}

template <class Choice, std::size_t N>
std::vector<std::string> spellings(const std::array<Choice, N>& candidates) {
    std::vector<std::string> out;
    for (const Choice& c : candidates) out.push_back(select::to_string(c));
    return out;
}

// A coverage `reached` row (chosen_origin, chosen_algo) -> a pin spelling for this op.
// Flat-selection rows carry the spelling; a pre-flat-selection binary (--old-csv) wrote
// native:<algo> or vendor:auto, and its bare `lpanel` is the only algo that is not a spelling.
template <class Choice>
std::string normalize_route(const std::string& origin, const std::string& algo) {
    if (origin == "vendor") return "vendor";
    const std::string text = algo == "lpanel" ? "lpanel:panel=8" : algo;
    if (auto c = select::parse<Choice>(text)) return select::to_string(*c);
    return origin + ":" + algo;
}

// The arms of one cell in one process: probe, warm-up, timed runs and verification, shared by
// the fixed-rep protocol (run_arms) and the race (run_race).
template <class Choice, class Problem>
class ArmBench {
public:
    using clock = std::chrono::steady_clock;
    std::vector<ArmOutcome> arms;

    ArmBench(const std::string& op, Problem& p, const std::vector<std::string>& names) : op_(op), p_(p) {
        for (const std::string& w : names) arms.push_back({w, "ok", "", {}, {}, 0.0, 0});
        std::size_t wneed = 1;
        for (ArmOutcome& a : arms) guarded(a, [&] { wneed = std::max(wneed, p_.workspace()); });
        ws_ = std::make_unique<UnifiedVector<std::byte>>(wneed);
    }

    std::size_t live() const {
        return std::count_if(arms.begin(), arms.end(), [](const auto& a) { return a.status == "ok"; });
    }

    bool once(ArmOutcome& a) {
        return guarded(a, [&] {
            p_.reset();
            p_.run(ws_->to_span());
        });
    }

    // Warm-up interleaved in the timed order: a per-arm warm-up made arm 0's first timed rep
    // 2.2x slow (evidence: docs/perf/small-n-baseline.md#warm-up-order-and-the-variance-gate).
    void warm(const std::vector<std::size_t>& order, double seconds) {
        const auto w0 = clock::now();
        do {
            for (std::size_t i : order) once(arms[i]);
        } while (std::chrono::duration<double>(clock::now() - w0).count() < seconds);
    }

    bool timed(ArmOutcome& a, int slot) {
        double ms = 0;
        const bool ran = guarded(a, [&] {
            p_.reset();
            const auto t0 = clock::now();
            p_.run(ws_->to_span());
            ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        });
        if (ran) {
            a.ms.push_back(ms);
            a.slot.push_back(slot);
        }
        return ran;
    }

    // Correctness in the same process, from one more untimed run: a fast wrong answer is "bad".
    void verify(ArmOutcome& a, double tol) {
        p_.clear_info();
        if (!once(a)) return;
        const auto [res, info] = p_.verify();
        a.residual = res;
        a.info_nonzero = info;
        if (!std::isfinite(res)) fail(a, "bad", "residual_nonfinite");
        else if (res > tol) fail(a, "bad", "residual");
        else if (info != 0) fail(a, "bad", "info");
    }

private:
    const std::string& op_;
    Problem& p_;
    std::unique_ptr<UnifiedVector<std::byte>> ws_;

    static void fail(ArmOutcome& a, const char* status, const std::string& why) {
        a.status = status;
        a.reason = why;
    }
    // Every call goes through here: a refused pin is "skipped", anything else "error".
    template <class F>
    bool guarded(ArmOutcome& a, F&& f) {
        if (a.status != "ok") return false;
        try {
            select::ScopedPin<Choice> pin(op_, a.arm);
            f();
            return true;
        } catch (const std::invalid_argument& e) {
            fail(a, "skipped", std::string("pin refused: ") + e.what());
        } catch (const std::exception& e) {
            fail(a, "error", e.what());
        }
        return false;
    }
};

template <class Choice, class Problem>
std::vector<ArmOutcome> run_arms(const std::string& op, Problem& p, const CellRequest& req, double tol) {
    ArmBench<Choice, Problem> b(op, p, req.arms);
    auto& arms = b.arms;
    if (req.mode == "jit" || req.mode == "probe") {
        for (ArmOutcome& a : arms) b.once(a);
        return arms;
    }
    std::vector<std::size_t> base(arms.size());
    for (std::size_t i = 0; i < base.size(); ++i) base[i] = req.reverse ? base.size() - 1 - i : i;
    b.warm(base, req.warm_s * double(b.live()));
    for (int r = 0; r < req.reps; ++r) {
        const auto order = rep_order(arms.size(), r, req.reverse);
        for (std::size_t pos = 0; pos < order.size(); ++pos) b.timed(arms[order[pos]], static_cast<int>(pos));
    }
    for (ArmOutcome& a : arms) b.verify(a, tol);
    return arms;
}

// The race (docs/design/tiered-tuning.md, the per-cell algorithm): rounds of one timed run per
// live arm, race_step after each, until race_over. Every raced arm is then verified, the
// eliminated too: one that fails is "bad" and leaves the row; one that passes keeps its median.
template <class Choice, class Problem>
std::vector<ArmOutcome> run_race(const std::string& op, Problem& p, const CellRequest& req, double tol) {
    std::vector<std::string> names;
    for (const std::string& s : req.seed_order)
        if (std::find(req.arms.begin(), req.arms.end(), s) != req.arms.end() &&
            std::find(names.begin(), names.end(), s) == names.end())
            names.push_back(s);
    for (const std::string& s : req.arms)
        if (std::find(names.begin(), names.end(), s) == names.end()) names.push_back(s);
    ArmBench<Choice, Problem> b(op, p, names);
    auto& arms = b.arms;
    std::vector<std::size_t> idx;  // the arms that entered the race
    for (std::size_t i = 0; i < arms.size(); ++i)
        if (arms[i].status == "ok") idx.push_back(i);
    b.warm(idx, req.warm_s * double(idx.size()));

    TierParams tp{};
    tp.min_reps = req.min_reps;
    tp.max_reps = req.max_reps;
    tp.confidence = req.confidence;
    RaceState s;
    for (std::size_t i : idx) s.cands.push_back(arms[i].arm);
    s.ms.assign(idx.size(), {});
    s.alive.assign(idx.size(), true);
    std::vector<bool> failed(idx.size(), false);
    for (int r = 0; !idx.empty(); ++r) {
        std::vector<std::size_t> alive;
        for (std::size_t c = 0; c < idx.size(); ++c)
            if (s.alive[c]) alive.push_back(c);
        std::vector<double> round(idx.size(), std::nan(""));
        const auto order = rep_order(alive.size(), r, req.alternate_reverse && r % 2);
        for (std::size_t pos = 0; pos < order.size(); ++pos) {
            const std::size_t c = alive[order[pos]];
            ArmOutcome& a = arms[idx[c]];
            if (b.timed(a, static_cast<int>(pos))) round[c] = a.ms.back();
            else failed[c] = true, s.alive[c] = false;
        }
        for (std::size_t c = 0; c < idx.size(); ++c) s.ms[c].push_back(round[c]);
        if (std::find(s.alive.begin(), s.alive.end(), true) == s.alive.end()) break;
        if (race_over(race_step(s, tp, 0.03), r + 1, tp)) break;
    }
    for (std::size_t c = 0; c < idx.size(); ++c) {
        if (failed[c]) continue;
        ArmOutcome& a = arms[idx[c]];
        b.verify(a, tol);
        if (s.alive[c] || a.status != "ok") continue;
        a.status = "eliminated";
        a.reason = "round " + std::to_string(a.ms.size());
    }
    std::vector<ArmOutcome> out;  // request order, like run_arms
    for (const std::string& w : req.arms)
        for (const ArmOutcome& a : arms)
            if (a.arm == w) out.push_back(a);
    return out;
}

}  // namespace batchlas::tune
