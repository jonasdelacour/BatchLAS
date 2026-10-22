// The tuner's GPU side (docs/design/tiered-tuning.md): the child-side race against the fixed-rep
// protocol at one cell, the persistent worker's round trip, and the carve-out guard (a fresh child
// agrees with a worker that ran a larger launch first).

#include <tools/tune/cell_runner.hh>
#include <tools/tune/grid.hh>
#include <tools/tune/schedule.hh>
#include <tools/tune/spec.hh>
#include <tools/tune/tune_core.hh>
#include <tools/tune/worker.hh>

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace batchlas::tune;
namespace fs = std::filesystem;

namespace {

batchlas::Queue& gpu_queue() {  // the worker's one Queue; leaked, as a static SYCL object must be
    static auto* q = new batchlas::Queue(batchlas::Device("gpu"), kBackend);
    return *q;
}

CellRequest request(const std::string& op, const std::string& dtype, const std::string& key, const std::string& mode) {
    const OpSpec* spec = find_spec(op);
    CellRequest r;
    r.dtype = dtype;
    r.key = *parse_key_arg(key);
    r.arms = spec->candidates(dtype);
    r.mode = mode;
    r.warm_s = 0.2;
    r.min_reps = 3;
    r.max_reps = 6;
    r.confidence = 0.80;
    r.tier = "preview";
    return r;
}

const ArmOutcome* find(const std::vector<ArmOutcome>& v, const std::string& arm) {
    for (const ArmOutcome& a : v)
        if (a.arm == arm) return &a;
    return nullptr;
}

std::string winner(const std::vector<ArmOutcome>& v, const std::vector<std::string>& order) {
    std::map<std::string, double> ok;
    for (const ArmOutcome& a : v)
        if (a.status == "ok") ok[a.arm] = median(a.ms);
    return ok.empty() ? "" : rank(ok, order).front();
}

}  // namespace

TEST(TuneRaceGpu, PreviewRaceAgreesWithSixteenTimedReps) {
    const std::string key = "uplo=L,n=32,batch=8192";
    CellRequest r = request("potrf", "float", key, "race");
    const auto race = find_spec("potrf")->run_cell(r, gpu_queue());
    r.mode = "time";
    r.reps = 16;
    const auto timed = find_spec("potrf")->run_cell(r, gpu_queue());
    ASSERT_EQ(race.size(), r.arms.size());
    int raced = 0, eliminated = 0;
    for (const ArmOutcome& a : race) {
        const ArmOutcome* t = find(timed, a.arm);
        ASSERT_NE(t, nullptr) << a.arm;
        std::printf("%-16s race %-10s reps %zu median %.4g | time %-7s median %.4g\n", a.arm.c_str(), a.status.c_str(),
                    a.ms.size(), median(a.ms), t->status.c_str(), median(t->ms));
        EXPECT_EQ(a.status == "skipped", t->status == "skipped") << a.arm;
        if (a.status == "skipped" || a.status == "error") continue;
        ++raced;
        eliminated += a.status == "eliminated";
        EXPECT_GE(a.ms.size(), a.status == "eliminated" ? 2u : 3u) << a.arm << ": a gross loser keeps its 2 cut runs";
        EXPECT_LE(a.ms.size(), std::size_t(kRepCap)) << a.arm;
        if (a.status == "eliminated") {
            EXPECT_EQ(t->status, "ok") << a.arm;
            EXPECT_FALSE(a.verified) << a.arm << ": outside the tie set and not in CellRequest::verify";
            EXPECT_EQ(a.residual, 0.0) << a.arm;
        } else {
            EXPECT_EQ(a.status, t->status) << a.arm;
        }
    }
    EXPECT_GE(raced, 2) << "the cell must race something";
    EXPECT_GE(eliminated, 1) << "tiny is about 2x faster than every other arm here";
    const std::string rw = winner(race, r.arms), tw = winner(timed, r.arms);
    ASSERT_FALSE(rw.empty());
    ASSERT_FALSE(tw.empty());
    EXPECT_LE(median(find(timed, rw)->ms), 1.03 * median(find(timed, tw)->ms))
        << "race winner " << rw << " vs 16-rep winner " << tw << " (" << eliminated << " eliminated)";
}

// An arm raced alone (a benched arm's fresh child) has no rival to beat: race_step calls it the
// winner after one round, and run_race still gives it min_reps rounds.
TEST(TuneRaceGpu, AnArmRacedAloneGetsMinReps) {
    CellRequest r = request("potrf", "float", "uplo=L,n=32,batch=8192", "race");
    r.arms = {"tiny"};
    const auto race = find_spec("potrf")->run_cell(r, gpu_queue());
    ASSERT_EQ(race.size(), 1u);
    EXPECT_EQ(race[0].status, "ok") << race[0].reason;
    EXPECT_EQ(race[0].ms.size(), std::size_t(r.min_reps));
    EXPECT_GT(race[0].residual, 0.0);
}

namespace {

// Arms that sleep instead of launching, named by the pin: "fast" 5 ms, "mid" 15 ms (3x: below the
// gross cut), "slow" 50 ms (a gross loser), "silent" 5 ms that writes nothing to the output (clear_info
// poisons it, as the specs do), "wobble" 5 and 10 ms alternately (its warm-up never converges),
// "dies" 5 ms that throws from its 4th run on (the first warm-up run), "wrongslow" a slow that writes nothing.
struct SleepProblem {
    batchlas::Queue& q;
    std::map<std::string, int> runs;
    double out = 0;
    std::size_t workspace() { return std::size_t(1) << 20; }
    void reset() {}
    void clear_info() { out = 1e6; }
    void run(batchlas::Span<std::byte>) {
        std::string src;
        const std::string arm = batchlas::select::detail::pin_text("sleepop", &src).value_or("");
        const int ms = arm == "slow" || arm == "wrongslow" ? 50 : arm == "mid" ? 15 : arm == "wobble" && runs[arm] % 2 ? 10 : 5;
        ++runs[arm];
        if (arm == "dies" && runs[arm] >= 4) throw std::runtime_error("dies");
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        if (arm != "silent" && arm != "wrongslow") out = 42;
    }
    std::pair<double, int> verify() { return {std::fabs(out - 42), 0}; }
};

}  // namespace

// The lean cell: a gross loser leaves after one untimed and two timed runs, with no warm-up share
// and no verification; a 3x arm is not cut there, it warms and races to min_reps; the survivors
// warm at least kWarmMinRounds rounds, stop once stable (sleeps are), and keep only the samples after
// it; the tie set is verified, each arm from its own run, so "silent" cannot pass on "slow"'s output.
TEST(TuneRaceGpu, LeanRaceCutsAGrossLoserBeforeTheWarmUp) {
    batchlas::Queue q(batchlas::Device("gpu"), kBackend);
    SleepProblem p{q};
    CellRequest r = request("potrf", "float", "uplo=L,n=32,batch=8192", "race");
    r.arms = {"fast", "slow", "silent", "mid"};
    const auto out = run_race<int>("sleepop", p, r, 1e-3);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[1].status, "eliminated") << out[1].reason;
    EXPECT_EQ(out[1].ms.size(), 2u);
    EXPECT_EQ(p.runs["slow"], 3) << "untimed + 2 cut rounds: no warm-up, no reps, no verify";
    EXPECT_FALSE(out[1].verified);
    EXPECT_FALSE(out[3].verified) << "eliminated, outside the tie set";
    EXPECT_TRUE(out[0].verified);
    EXPECT_EQ(out[3].status, "eliminated") << out[3].reason;
    EXPECT_GE(out[3].ms.size(), std::size_t(r.min_reps)) << "3x is no gross loser: eliminated by the paired rule";
    EXPECT_GE(p.runs["mid"], 3 + kWarmMinRounds + r.min_reps) << "mid warmed with the survivors (and is not verified)";
    EXPECT_EQ(out[0].status, "ok") << out[0].reason;
    EXPECT_GE(out[0].ms.size(), std::size_t(r.min_reps));
    EXPECT_LE(out[0].ms.size(), std::size_t(kRepCap));
    const int warm_rounds = p.runs["fast"] - 4 - int(out[0].ms.size());
    EXPECT_GE(warm_rounds, kWarmMinRounds) << "warm rounds, untimed";
    EXPECT_LT(warm_rounds, 12) << "stable sleeps converge long before the cap (~24 rounds); oversleep can cost a round or two";
    EXPECT_EQ(out[2].status, "bad") << "silent wrote nothing; the poison, not slow's output, is verified";
    EXPECT_EQ(out[2].reason, "residual");
    EXPECT_GE(q.workspace_capacity(), std::size_t(1) << 20) << "the workspace is leased from the cell's Queue";
}

// The bucket's first sight of an arm (CellRequest::verify) verifies it outside the tie set: a gross
// loser with a wrong answer is caught there; without the request it stays unverified, not bad.
TEST(TuneRaceGpu, LeanRaceVerifiesTheRequestedArmsOutsideTheTieSet) {
    batchlas::Queue q(batchlas::Device("gpu"), kBackend);
    SleepProblem p{q};
    CellRequest r = request("potrf", "float", "uplo=L,n=32,batch=8192", "race");
    r.arms = {"fast", "wrongslow", "slow"};
    auto out = run_race<int>("sleepop", p, r, 1e-3);
    EXPECT_EQ(out[1].status, "eliminated") << out[1].reason;
    EXPECT_FALSE(out[1].verified);
    r.verify = {"wrongslow", "slow"};
    out = run_race<int>("sleepop", p, r, 1e-3);
    EXPECT_EQ(out[1].status, "bad");
    EXPECT_EQ(out[1].reason, "residual");
    EXPECT_TRUE(out[1].verified);
    EXPECT_EQ(out[2].status, "eliminated") << "verified and passed: keeps its cut median";
    EXPECT_TRUE(out[2].verified);
    EXPECT_EQ(p.runs["slow"], 3 + 4) << "this race verified it once more";
}

// A warm-up whose round totals alternate 10 and 15 ms never agrees within 2%: the cap ends it, at
// kWarmCapPerArmS per survivor (plus the untimed first runs and one round's overshoot).
TEST(TuneRaceGpu, LeanWarmUpStopsAtTheCapWhenItNeverConverges) {
    batchlas::Queue q(batchlas::Device("gpu"), kBackend);
    SleepProblem p{q};
    CellRequest r = request("potrf", "float", "uplo=L,n=32,batch=8192", "race");
    r.arms = {"fast", "wobble"};
    cell_phases() = Phases{};
    const auto out = run_race<int>("sleepop", p, r, 1e-3);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_NE(out[1].status, "error") << out[1].reason;
    const double cap = kWarmCapPerArmS * 2;
    EXPECT_GE(cell_phases().warm, cap);
    EXPECT_LE(cell_phases().warm, cap + 0.1);
}

// Once no survivor runs, the warm-up ends at once instead of waiting out the cap on the clock.
TEST(TuneRaceGpu, LeanWarmUpEndsWhenEverySurvivorFails) {
    batchlas::Queue q(batchlas::Device("gpu"), kBackend);
    SleepProblem p{q};
    CellRequest r = request("potrf", "float", "uplo=L,n=32,batch=8192", "race");
    r.arms = {"dies"};
    cell_phases() = Phases{};
    const auto out = run_race<int>("sleepop", p, r, 1e-3);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].status, "error");
    EXPECT_EQ(p.runs["dies"], 4) << "untimed + 2 cut rounds + the throwing warm run";
    EXPECT_LT(cell_phases().warm, 0.5 * kWarmCapPerArmS);
}

#ifdef BATCHLAS_TUNE_IMPL
namespace {

// The worker inherits this process's CUDA_VISIBLE_DEVICES (ctest's GPU slot).
void start_worker(WorkerProcess& w, const std::string& log) {
    ASSERT_TRUE(w.start({BATCHLAS_TUNE_IMPL, "--worker", "--warm-s", "0.5"}, {}, log));
}

std::vector<ArmOutcome> on_worker(WorkerProcess& w, const std::string& op, const CellRequest& r) {
    std::vector<Record> recs;
    EXPECT_TRUE(w.send(request_line(op, r)));
    EXPECT_EQ(w.receive(600, &recs), WorkerProcess::Got::done);
    return outcomes_from_records(recs);
}

std::vector<ArmOutcome> fresh_child(const std::string& op, const CellRequest& r, const fs::path& dir) {
    const fs::path res = dir / "fresh.jsonl";
    std::string cmd = std::string(BATCHLAS_TUNE_IMPL) + " --cell " + op + " --dtype " + r.dtype + " --key " +
                      key_arg(r.key) + " --arms ";
    for (std::size_t i = 0; i < r.arms.size(); ++i) cmd += (i ? "," : "") + r.arms[i];
    cmd += " --mode race --warm 0.2 --min-reps 3 --max-reps 6 --confidence 0.8 --result " + res.string() + " > " +
           (dir / "fresh.log").string() + " 2>&1";
    EXPECT_EQ(std::system(cmd.c_str()), 0) << cmd;
    return outcomes_from_records(read_records(res.string()));
}

struct Tmp {
    fs::path path = fs::temp_directory_path() / ("tune_race_gpu_" + std::to_string(::getpid()));
    Tmp() { fs::create_directories(path); }
    ~Tmp() { fs::remove_all(path); }
};

}  // namespace

TEST(TuneRaceGpu, WorkerAnswersTwoCellsInOrder) {
    Tmp tmp;
    WorkerProcess w;
    start_worker(w, (tmp.path / "worker.log").string());
    CellRequest a = request("potrf", "float", "uplo=L,n=16,batch=1024", "race");
    a.arms = {"tiny", "cta"};
    CellRequest b = request("potrf", "float", "uplo=U,n=24,batch=512", "race");
    b.arms = {"cta", "lpanel:panel=8", "vendor"};
    ASSERT_TRUE(w.send(request_line("potrf", a)));
    ASSERT_TRUE(w.send(request_line("potrf", b)));
    for (const CellRequest* r : {&a, &b}) {
        std::vector<Record> recs;
        ASSERT_EQ(w.receive(600, &recs), WorkerProcess::Got::done);
        const auto got = outcomes_from_records(recs);
        ASSERT_EQ(got.size(), r->arms.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_EQ(got[i].arm, r->arms[i]);
            EXPECT_TRUE(got[i].status == "ok" || got[i].status == "eliminated" || got[i].status == "skipped")
                << got[i].arm << " " << got[i].status << " " << got[i].reason;
        }
    }
    w.stop();
    EXPECT_FALSE(w.alive());
}

// The carve-out: each SLM-heavy candidate first runs a cell above 48 KB on the worker, then a cell
// in or just below the 48 KB launch hole (docs/perf/potrf.md#potrf-the-48-kb-launch-hole). A fresh
// child must agree on every candidate's feasibility; a mismatch is the defect the run audit hunts.
TEST(TuneRaceGpu, FreshChildAgreesWithAWorkerAfterALargerLaunch) {
    struct Case {
        std::string op, dtype, big, small;
    };
    const std::vector<Case> cases{
        {"potrf", "float", "uplo=L,n=128,batch=512", "uplo=L,n=110,batch=512"},
        {"potrf", "double", "uplo=L,n=96,batch=512", "uplo=L,n=78,batch=512"},
        {"posv", "float", "uplo=L,n=128,nrhs=1,batch=512", "uplo=L,n=110,nrhs=1,batch=512"},
        {"trsm", "float", "side=L,trans=N,order=128,q=128,batch=512,uplo=L,diag=N",
         "side=L,trans=N,order=104,q=104,batch=512,uplo=L,diag=N"},
    };
    Tmp tmp;
    for (const Case& c : cases) {
        if (!find_spec(c.op)) continue;
        WorkerProcess w;
        start_worker(w, (tmp.path / "worker.log").string());
        (void)on_worker(w, c.op, request(c.op, c.dtype, c.big, "race"));
        const CellRequest small = request(c.op, c.dtype, c.small, "race");
        const auto warm = on_worker(w, c.op, small);
        w.stop();
        const auto fresh = fresh_child(c.op, small, tmp.path);
        const AuditResult a = audit_compare(warm, fresh, small.arms);
        std::printf("%s %s %s after %s: %s\n", c.op.c_str(), c.dtype.c_str(), c.small.c_str(), c.big.c_str(), a.verdict.c_str());
        EXPECT_EQ(a.verdict.rfind("mismatch:feasibility", 0), std::string::npos) << c.op << " " << c.small << ": " << a.verdict;
    }
}
#else
TEST(TuneRaceGpu, WorkerAnswersTwoCellsInOrder) {
    GTEST_SKIP() << "needs batchlas_tune_impl: configure with -DBATCHLAS_BUILD_BENCHMARKS=ON";
}
TEST(TuneRaceGpu, FreshChildAgreesWithAWorkerAfterALargerLaunch) {
    GTEST_SKIP() << "needs batchlas_tune_impl: configure with -DBATCHLAS_BUILD_BENCHMARKS=ON";
}
#endif

// The real specs' batch-ladder hooks, on sm_120's 188 SMs (host-only: no GPU is touched).
// evidence: docs/design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder
TEST(TuneSpecLadder, RealSpecHooksGiveTheDocumentedLadders) {
    using B = std::vector<std::int64_t>;
    const std::vector<std::pair<std::string, B>> want{
        {"gemm", {376, 3008, 24064}},           {"ormqr", {376, 3008, 24064}},
        {"syrk", {1, 2, 4, 8, 47, 376, 3008, 24064}}, {"symm", {1, 2, 4, 8, 47, 376, 3008, 24064}},
        {"syr2k", {1, 2, 3, 47, 376, 3008, 24064}},   {"trmm", {1, 47, 376, 3008, 24064}},
        {"potrf", {47, 376, 3008, 24064}},      {"trsm", {47, 376, 3008, 24064}},
        {"posv", {47, 376, 3008, 24064}}};
    for (const auto& [op, ladder] : want) {
        const OpSpec* s = find_spec(op);
        ASSERT_NE(s, nullptr) << op;
        EXPECT_EQ(batch_ladder(188, s->ladder_classes(default_batch_classes()), s->ladder_extra_batches()), ladder) << op;
    }
}

TEST(TuneSpecLadder, SyrkKeepsItsSingleBatchVendorLine) {
    const OpSpec* s = find_spec("syrk");
    ASSERT_NE(s, nullptr);
    const auto ladder = batch_strings(batch_ladder(188, default_batch_classes(), s->ladder_extra_batches()));
    std::set<std::string> c_batches, other;
    for (const CellKey& k : rebatch(s->grid("float", {}), ladder))
        (*key_get(k, "trans") == "C" ? c_batches : other).insert(*key_get(k, "batch"));
    EXPECT_EQ(other, std::set<std::string>(ladder.begin(), ladder.end()));
    if (!c_batches.empty()) EXPECT_EQ(c_batches, std::set<std::string>{"1"}) << "trans=C is declared at batch 1 only";
}
