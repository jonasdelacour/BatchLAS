// The tuner's GPU side (docs/design/tiered-tuning.md): the child-side race against the fixed-rep
// protocol at one cell, the persistent worker's round trip, and the carve-out guard (a fresh child
// agrees with a worker that ran a larger launch first).

#include "../tools/tune/schedule.hh"
#include "../tools/tune/spec.hh"
#include "../tools/tune/tune_core.hh"
#include "../tools/tune/worker.hh"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace batchlas::tune;
namespace fs = std::filesystem;

namespace {

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
    const auto race = find_spec("potrf")->run_cell(r);
    r.mode = "time";
    r.reps = 16;
    const auto timed = find_spec("potrf")->run_cell(r);
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
        EXPECT_GE(a.ms.size(), 3u) << a.arm;
        EXPECT_LE(a.ms.size(), 6u) << a.arm;
        if (a.status == "eliminated") EXPECT_EQ(t->status, "ok") << a.arm << " (eliminated arms are not verified)";
        else EXPECT_EQ(a.status, t->status) << a.arm;
    }
    EXPECT_GE(raced, 2) << "the cell must race something";
    EXPECT_GE(eliminated, 1) << "tiny is about 2x faster than every other arm here";
    const std::string rw = winner(race, r.arms), tw = winner(timed, r.arms);
    ASSERT_FALSE(rw.empty());
    ASSERT_FALSE(tw.empty());
    EXPECT_LE(median(find(timed, rw)->ms), 1.03 * median(find(timed, tw)->ms))
        << "race winner " << rw << " vs 16-rep winner " << tw << " (" << eliminated << " eliminated)";
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
