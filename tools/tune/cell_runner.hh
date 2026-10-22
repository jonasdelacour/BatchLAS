// SPDX-License-Identifier: MIT
#pragma once

// The child side of one cell (docs/design/flat-kernel-selection.md §6.1, §6.3): probe every
// arm through the public sizing call under its ScopedPin (a refused pin is "skipped", never
// timed), then warm up and time the arms interleaved, rotating the order every rep, then verify
// each arm from one more untimed run. The workspace is leased from the worker's Queue, whose arena
// outlives the cell; ~ArmBench drains it with wait_and_throw, so no async error reaches the next
// cell, and trims the arena's slack. The SLM carve-out is sticky per CUfunction
// (benchmarks/factor_bench.cc header), so a process that has run a larger launch can accept one a
// fresh process refuses: the custom protocol forks a child per cell, and the tiered worker takes
// its cells in ascending bytes and is audited against fresh children
// (evidence: docs/design/tiered-tuning.md#engine-persistent-workers-and-the-carve-out-audit).
//
// Problem<T> supplies: std::size_t workspace() (the public *_buffer_size; throws
// std::invalid_argument when the current pin cannot run), void reset() (restores the inputs,
// synchronously), void clear_info(), void run(Span<std::byte>) (the public op, then wait), and
// std::pair<double, int> verify() (residual on items 0 and batch-1, nonzero info count).

#include <tools/tune/race.hh>
#include <tools/tune/spec.hh>

#include <batchlas/backend_config.h>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/select/select.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {

// The vendor arm is whichever GPU library this build links. Host (cpu) tables: phase 4.
// Backend::SYCL (AdaptiveCpp Metal) links none: its vendor arm is refused by the strict pin,
// and it has double kernels only with device fp64.
#if BATCHLAS_HAS_CUDA_BACKEND
inline constexpr Backend kBackend = Backend::CUDA;
#define BATCHLAS_TUNE_FP64 1
#elif BATCHLAS_HAS_ROCM_BACKEND
inline constexpr Backend kBackend = Backend::ROCM;
#define BATCHLAS_TUNE_FP64 1
#else
inline constexpr Backend kBackend = Backend::SYCL;
#define BATCHLAS_TUNE_FP64 BATCHLAS_HAS_DEVICE_FP64
#endif

template <typename F>
decltype(auto) with_dtype(const std::string& dtype, F&& f) {
    if (dtype == "float") return f.template operator()<float>();
    if (dtype == "cfloat") return f.template operator()<std::complex<float>>();
#if BATCHLAS_TUNE_FP64
    if (dtype == "double") return f.template operator()<double>();
    if (dtype == "cdouble") return f.template operator()<std::complex<double>>();
#else
    if (dtype == "double" || dtype == "cdouble")
        throw batchlas::invalid_argument("dtype '" + dtype + "' needs device fp64, which this build lacks");
#endif
    throw batchlas::invalid_argument("unknown dtype '" + dtype + "'");
}

// symm, syrk and syr2k are real-only (RealScalar): no complex instantiation and no complex table.
template <typename F>
decltype(auto) with_real_dtype(const std::string& op, const std::string& dtype, F&& f) {
    if (dtype == "float") return f.template operator()<float>();
#if BATCHLAS_TUNE_FP64
    if (dtype == "double") return f.template operator()<double>();
#else
    if (dtype == "double") throw batchlas::invalid_argument(op + ": 'double' needs device fp64, which this build lacks");
#endif
    throw batchlas::invalid_argument(op + " is real-only: no '" + dtype + "' (float, double)");
}

// A non-lattice grid() (form derived from the extents) under `--grid name=v1:v2`: the overrides
// FILTER the declared cells, as gemm's do.
inline std::vector<CellKey> filter_cells(const OpSpec& s, std::vector<CellKey> cells,
                                         const std::map<std::string, std::vector<std::string>>& overrides) {
    const auto ax = s.axes();
    for (const auto& [name, values] : overrides)
        if (std::none_of(ax.begin(), ax.end(), [&](const auto& a) { return a.first == name; }))
            throw batchlas::invalid_argument(s.op() + " has no grid axis '" + name + "'");
    std::vector<CellKey> out;
    for (CellKey& key : cells)
        if (std::all_of(key.begin(), key.end(), [&](const auto& kv) {
                const auto it = overrides.find(kv.name);
                return it == overrides.end() || std::find(it->second.begin(), it->second.end(), kv.value) != it->second.end();
            }))
            out.push_back(std::move(key));
    if (out.empty()) throw batchlas::invalid_argument(s.op() + ": the --grid filters leave no cell of the declared grid");
    return out;
}

template <typename Choice, std::size_t N>
std::vector<std::string> spellings(const std::array<Choice, N>& candidates) {
    std::vector<std::string> out;
    for (const Choice& c : candidates) out.push_back(select::to_string(c));
    return out;
}

// The arms of one cell in one process: probe, warm-up, timed runs and verification, shared by
// the fixed-rep protocol (run_arms) and the race (run_race).
template <typename Choice, typename Problem>
class ArmBench {
public:
    using clock = std::chrono::steady_clock;
    std::vector<ArmOutcome> arms;

    ArmBench(const std::string& op, Problem& p, const std::vector<std::string>& names) : op_(op), p_(p) {
        for (const std::string& w : names) arms.push_back({w, "ok", "", {}, {}, 0.0, 0});
        for (ArmOutcome& a : arms) guarded(a, [&] { wneed_ = std::max(wneed_, p_.workspace()); });
        ws_ = p_.q.workspace(wneed_);
    }
    ArmBench(const ArmBench&) = delete;
    ArmBench& operator=(const ArmBench&) = delete;
    ~ArmBench() {
        ws_.release();
        try {  // DPC++'s plain wait() leaves an async error pending on the Queue for a later cell
            p_.q.wait_and_throw();
            if (p_.q.workspace_capacity() > 2 * wneed_) (void)p_.q.trim_workspace();  // blocks are never freed
        } catch (const std::exception& e) {
            std::fprintf(stderr, "batchlas_tune: %s: an async error at the cell's end: %s\n", op_.c_str(), e.what());
        }
    }

    std::size_t live() const {
        return std::count_if(arms.begin(), arms.end(), [](const auto& a) { return a.status == "ok"; });
    }

    bool once(ArmOutcome& a) {
        return guarded(a, [&] {
            p_.reset();
            p_.run(ws_.span());
        });
    }

    // Warm-up interleaved in the timed order: a per-arm warm-up made arm 0's first timed rep
    // 2.2x slow (evidence: docs/perf/small-n-baseline.md#warm-up-order-and-the-variance-gate).
    void warm(const std::vector<std::size_t>& order, double seconds) {
        const auto w0 = clock::now();
        do {
            for (std::size_t i : order) once(arms[i]);
        } while (std::chrono::duration<double>(clock::now() - w0).count() < seconds);
        spent(&Phases::warm, w0);
    }

    // The lean warm-up: interleaved rounds until two consecutive round totals (run spans only)
    // agree, at most kWarmCapPerArmS per arm. Returns each arm's run ms over the last two rounds.
    std::vector<std::vector<double>> warm_until_stable(const std::vector<std::size_t>& order) {
        const auto w0 = clock::now();
        std::vector<std::vector<double>> last(order.size());
        double prev = 0;
        for (int r = 1;; ++r) {
            double total = 0;
            bool ran = false;
            for (std::size_t i = 0; i < order.size(); ++i) {
                double ms = 0;
                if (guarded(arms[order[i]], [&] { ms = reset_and_run(); })) total += ms, ran = true;
                if (last[i].size() == 2) last[i].erase(last[i].begin());
                last[i].push_back(ms);
            }
            if (!ran || warm_converged(prev, total, r) ||
                seconds_since(w0) >= kWarmCapPerArmS * double(order.size()))
                break;
            prev = total;
        }
        spent(&Phases::warm, w0);
        return last;
    }

    bool timed(ArmOutcome& a, int slot) {
        double ms = 0;
        const auto c0 = clock::now();
        const bool ran = guarded(a, [&] { ms = reset_and_run(); });
        if (ran) {
            a.ms.push_back(ms);
            a.slot.push_back(slot);
        }
        spent(&Phases::timed, c0);  // the call's wall, reset included: more than ms
        return ran;
    }

    // Correctness in the same process, from one more untimed run: a fast wrong answer is "bad".
    void verify(ArmOutcome& a, double tol) {
        const auto v0 = clock::now();
        p_.clear_info();
        const bool ran = once(a);
        const auto [res, info] = ran ? p_.verify() : std::pair<double, int>{0.0, 0};
        spent(&Phases::verify, v0);
        if (!ran) return;
        a.residual = res;
        a.info_nonzero = info;
        if (!std::isfinite(res)) fail(a, "bad", "residual_nonfinite");
        else if (res > tol) fail(a, "bad", "residual");
        else if (info != 0) fail(a, "bad", "info");
    }

private:
    const std::string& op_;
    Problem& p_;
    std::size_t wneed_ = 1;
    WorkspaceLease ws_;

    double reset_and_run() {  // ms of the run alone
        p_.reset();
        const auto t0 = clock::now();
        p_.run(ws_.span());
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }

    static void spent(double Phases::*f, clock::time_point t0) { cell_phases().add(f, seconds_since(t0)); }
    static void fail(ArmOutcome& a, const char* status, const std::string& why) {
        a.status = status;
        a.reason = why;
    }
    // Every call goes through here: a refused pin is "skipped", anything else "error". The pin is
    // strict: `vendor` is also a class word, which would otherwise time Auto under vendor's name.
    template <typename F>
    bool guarded(ArmOutcome& a, F&& f) {
        if (a.status != "ok") return false;
        try {
            select::ScopedPin<Choice> pin(op_, a.arm, select::StrictPin{});
            f();
            return true;
        } catch (const std::invalid_argument& e) {
            const bool strict = std::string_view(e.what()).find("(strict)") != std::string_view::npos;
            fail(a, "skipped", std::string(strict ? "pin refused (strict): " : "pin refused: ") + e.what());
        } catch (const std::exception& e) {
            fail(a, "error", e.what());
        }
        return false;
    }
};

template <typename Choice, typename Problem>
std::vector<ArmOutcome> run_arms(const std::string& op, Problem& p, const CellRequest& req, double tol) {
    ArmBench<Choice, Problem> b(op, p, req.arms);
    auto& arms = b.arms;
    if (req.mode == "jit" || req.mode == "probe") {
        for (ArmOutcome& a : arms) b.once(a);
        return arms;
    }
    if (req.mode == "verify") {
        for (ArmOutcome& a : arms) b.verify(a, tol);
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

// The race (docs/design/tiered-tuning.md#engine-the-lean-cell): one untimed run per arm, two
// timed rounds and the gross cut, the survivors' samples dropped and warm_until_stable over them,
// then rounds of one timed run per live arm, race_step after each, until race_over under the
// lean_max_reps cap. Then only the arms that can rank first (verify_next) and req.verify are
// verified: one that fails is "bad" and leaves the row, the rest keep their medians (a gross loser:
// of its two cut rounds), unverified arms flagged so.
template <typename Choice, typename Problem>
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
    const auto w0 = std::chrono::steady_clock::now();
    for (std::size_t i : idx) b.once(arms[i]);  // JIT and module load stay out of the gross cut
    cell_phases().add(&Phases::warm, seconds_since(w0));

    TierParams tp{};
    tp.min_reps = req.min_reps;
    tp.max_reps = req.max_reps;
    tp.confidence = req.confidence;
    RaceState s;
    for (std::size_t i : idx) s.cands.push_back(arms[i].arm);
    s.ms.assign(idx.size(), {});
    s.alive.assign(idx.size(), true);
    std::vector<bool> failed(idx.size(), false);
    auto fail = [&](std::size_t c) { failed[c] = true, s.alive[c] = false; };
    for (std::size_t c = 0; c < idx.size(); ++c)
        if (arms[idx[c]].status != "ok") fail(c);
    auto round = [&](int r) {  // false once no arm is alive
        std::vector<std::size_t> alive;
        for (std::size_t c = 0; c < idx.size(); ++c)
            if (s.alive[c]) alive.push_back(c);
        std::vector<double> got(idx.size(), std::nan(""));
        const auto order = rep_order(alive.size(), r, req.alternate_reverse && r % 2);
        for (std::size_t pos = 0; pos < order.size(); ++pos) {
            const std::size_t c = alive[order[pos]];
            ArmOutcome& a = arms[idx[c]];
            if (b.timed(a, static_cast<int>(pos))) got[c] = a.ms.back();
            else fail(c);
        }
        for (std::size_t c = 0; c < idx.size(); ++c) s.ms[c].push_back(got[c]);
        return std::find(s.alive.begin(), s.alive.end(), true) != s.alive.end();
    };
    // Two pairs are below every tier's min_reps, so only the gross rule fires here.
    const bool live = round(0) && round(1);
    if (live) race_step(s, tp, 0.03);
    std::vector<std::size_t> warm;
    for (std::size_t c = 0; c < idx.size(); ++c) {
        s.ms[c].clear();
        if (!s.alive[c]) continue;
        arms[idx[c]].ms.clear();
        arms[idx[c]].slot.clear();
        warm.push_back(idx[c]);
    }
    if (live) {
        const auto last = b.warm_until_stable(warm);
        double slowest = 0;
        for (std::size_t c = 0, w = 0; c < idx.size(); ++c) {
            if (!s.alive[c]) continue;
            if (arms[idx[c]].status != "ok") fail(c);
            else slowest = std::max(slowest, median(last[w]));
            ++w;
        }
        tp.max_reps = lean_max_reps(slowest, tp);
        tp.min_reps = std::min(tp.min_reps, tp.max_reps);
    }
    for (int r = 0; live && round(r); ++r)
        if (race_over(race_step(s, tp, 0.03), r + 1, tp)) break;
    for (std::size_t c = 0; c < idx.size(); ++c) {
        if (failed[c]) continue;
        ArmOutcome& a = arms[idx[c]];
        a.verified = false;
        if (s.alive[c]) continue;
        a.status = "eliminated";
        a.reason = "round " + std::to_string(a.ms.size());
    }
    for (auto want = verify_next(arms, req.verify); !want.empty(); want = verify_next(arms, req.verify))
        for (ArmOutcome& a : arms) {
            if (std::find(want.begin(), want.end(), a.arm) == want.end()) continue;
            const std::string kept = a.status;
            a.status = "ok";  // guarded() runs "ok" arms only
            b.verify(a, tol);
            if (a.status == "ok") a.status = kept;
            a.verified = true;
        }
    std::vector<ArmOutcome> out;  // request order, like run_arms
    for (const std::string& w : req.arms)
        for (const ArmOutcome& a : arms)
            if (a.arm == w) out.push_back(a);
    return out;
}

}  // namespace batchlas::tune
