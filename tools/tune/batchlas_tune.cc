// batchlas_tune: measures every candidate of an op over its declared grid and writes the
// per-device selection tables (docs/design/flat-kernel-selection.md §6). Usage, the raw JSONL
// schema and the protocol are in tools/tune/README.md.
//
//   batchlas_tune potrf,trsm --tier preview --dtype float --devices 1 --out tuned   (tiered_driver.cc)
//   batchlas_tune potrf --dtype float --devices 1 --reps 16 --raw benchmarks/results/tuning   (custom)
//   batchlas_tune potrf --devices 1 --gate --old-csv old.csv --gate-csv gate.csv
//
// THE DRIVER NEVER RUNS A KERNEL. The custom protocol forks and execs this binary in --cell mode
// once per (cell, pass), because the SLM carve-out attribute is sticky per CUfunction and an
// earlier, larger launch in the same process changes what a later one does
// (benchmarks/factor_bench.cc header). A tiered run keeps one --worker per GPU instead, fed its
// cells in ascending bytes and audited against fresh --cell --mode race children
// (docs/design/tiered-tuning.md#engine-persistent-workers-and-the-carve-out-audit); --no-worker
// races every cell in a fresh child. Candidates are interleaved inside the process either way.
//
// MULTI-GPU (--devices 1,2,3) runs one child per GPU at a time, each GPU held under a flock for
// the whole run, cells sharded round-robin and both passes of a cell kept on one GPU. This
// departs from docs/developer/agent-guide.md §10's one-measuring-process-per-box rule exactly as the potrf and
// posv seed sweeps did (benchmarks/results/routing/README.md); use one device when the box is
// shared or when a verdict hinges on a few percent.
//
// NO CONTEXT IN THE DRIVER. Linking libbatchlas enumerates devices during static init, which
// retains a CUDA primary context on every visible GPU. So this is batchlas_tune_impl, started by
// the SYCL-free launcher (launcher.cc) with CUDA_VISIBLE_DEVICES="" for driver modes; children
// get their GPU back explicitly. guard() dies if the driver's pid ever shows up on a GPU.

#include <batchlas/util/sycl-device-queue.hh>
#include <sycl/sycl.hpp>

#include "../../src/select/select.hh"
#include "cell_runner.hh"
#include "spec.hh"
#include "tiered_driver.hh"
#include "tune_core.hh"
#include "worker.hh"

#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

#ifndef BATCHLAS_TUNE_SOURCE_DIR
#define BATCHLAS_TUNE_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using namespace batchlas::tune;

namespace {

constexpr int kSchema = 1;
constexpr double kTie = 0.03;  // §6.3; scripts/sweep_to_table.py TIE

struct Opts {
    std::string op;
    std::vector<std::string> ops;  // the op argument: a comma list or "all"
    std::optional<Tier> tier;      // --tier; the protocol flags below make the run tier custom
    std::string custom_flag;       // the first protocol flag seen (custom tier, the two-pass path)
    double budget_h = 0, overhead_s = kChildOverheadS;
    bool plan = false, status = false;
    int progress_fd = -1;
    std::string ledger, import_raw, device_key;
    std::vector<std::string> dtypes{"float"};
    std::vector<int> devices;  // required: no default lands on a display GPU (docs/developer/agent-guide.md §13)
    std::string repo = BATCHLAS_TUNE_SOURCE_DIR;
    std::string raw, out;
    int reps = 16, passes = 2, ld_pad = 0;
    double warm = 1.5, remeasure = 0.10, refine_ratio = 1.1, cap_gib = 4.0, cell_timeout = 1800, guard_wait = 300;
    double util_ceiling = 5;  // benchmarks/gpu_guard.sh UTIL_CEILING
    bool jit = true, refine = true, guard = true, devices_given = false, dtype_given = false;
    bool allow_idle_foreign = false;
    bool worker = true;          // tiered: one persistent --worker per GPU (--no-worker: fresh children)
    double audit_fraction = -1;  // tiered: < 0 is the tier's
    std::string lock_dir = "/tmp";
    std::map<std::string, std::vector<std::string>> grid;
    bool gate = false;
    std::string old_csv, parent_bin, gate_csv;
    double gate_limit = 1.05;
    std::vector<std::string> argv;
};

std::string g_tmp;  // the driver's scratch directory, removed on die() too

[[noreturn]] void die(const std::string& why) {
    std::fprintf(stderr, "batchlas_tune: %s\n", why.c_str());
    std::error_code ec;
    if (!g_tmp.empty()) fs::remove_all(g_tmp, ec);
    std::exit(2);
}

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

std::string fmt(double v, const char* f = "%.4g") {
    char b[64];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

std::string capture(const std::string& cmd) {
    std::string out;
    if (FILE* p = ::popen(cmd.c_str(), "r")) {
        char buf[512];
        while (std::fgets(buf, sizeof(buf), p)) out += buf;
        ::pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

std::string self_exe() {
    std::error_code ec;
    const auto p = fs::read_symlink("/proc/self/exe", ec);
    if (ec) die("cannot resolve /proc/self/exe");
    return p.string();
}

// ---- processes ---------------------------------------------------------------------------

struct Spawned {
    pid_t pid = -1;
    int status = -1;  // exit code, or 128 + signal
    bool timed_out = false;
};

Spawned spawn_wait(const std::vector<std::string>& argv, const std::vector<std::pair<std::string, std::string>>& env,
                   const std::string& log, double timeout_s) {
    Spawned r;
    r.pid = ::fork();
    if (r.pid < 0) die("fork failed");
    if (r.pid == 0) {
        for (const auto& [k, v] : env) ::setenv(k.c_str(), v.c_str(), 1);
        if (!log.empty()) {
            const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) {
                ::dup2(fd, 1);
                ::dup2(fd, 2);
                ::close(fd);
            }
        }
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        ::execv(args[0], args.data());
        std::fprintf(stderr, "execv %s: %s\n", args[0], std::strerror(errno));
        ::_exit(127);
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        int st = 0;
        const pid_t w = ::waitpid(r.pid, &st, WNOHANG);
        if (w == r.pid) {
            r.status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            return r;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            ::kill(r.pid, SIGKILL);
            ::waitpid(r.pid, &st, 0);
            r.timed_out = true;
            r.status = 128 + SIGKILL;
            return r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::string tail(const std::string& path, std::size_t bytes = 600) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    if (s.size() > bytes) s = s.substr(s.size() - bytes);
    std::replace(s.begin(), s.end(), '\n', ' ');
    return s;
}

// ---- child modes -------------------------------------------------------------------------

int child_main(const std::vector<std::string>& a) {
    CellRequest req;
    std::string op, result;
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto val = [&] { return i + 1 < a.size() ? a[++i] : std::string(); };
        if (a[i] == "--cell") op = val();
        else if (a[i] == "--dtype") req.dtype = val();
        else if (a[i] == "--key") req.key = parse_key_arg(val()).value_or(CellKey{});
        else if (a[i] == "--arms") req.arms = split(val(), ',');
        else if (a[i] == "--mode") req.mode = val();
        else if (a[i] == "--reps") req.reps = std::stoi(val());
        else if (a[i] == "--warm") req.warm_s = std::stod(val());
        else if (a[i] == "--reverse") req.reverse = true;
        else if (a[i] == "--ld-pad") req.ld_pad = std::stoi(val());
        else if (a[i] == "--tier") req.tier = val();
        else if (a[i] == "--min-reps") req.min_reps = std::stoi(val());
        else if (a[i] == "--max-reps") req.max_reps = std::stoi(val());
        else if (a[i] == "--confidence") req.confidence = std::stod(val());
        else if (a[i] == "--alternate-reverse") req.alternate_reverse = true;
        else if (a[i] == "--seed-order") req.seed_order = split(val(), ',');
        else if (a[i] == "--result") result = val();
        else die("--cell: unknown argument " + a[i]);
    }
    const OpSpec* spec = find_spec(op);
    if (!spec || req.key.empty() || req.arms.empty() || result.empty()) die("--cell needs a known op, --key, --arms, --result");
    std::ofstream out(result);
    out << outcome_text(spec->run_cell(req));
    return out ? 0 : 1;
}

// Clock warm-up at worker start: a plain FMA loop, so no library kernel runs before the first cell.
void worker_warm(double seconds) {
    if (seconds <= 0) return;
    try {
        sycl::queue q{sycl::gpu_selector_v};
        constexpr std::size_t n = std::size_t(1) << 20;
        float* out = sycl::malloc_device<float>(n, q);
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                 float x = float(i[0]), y = 1.0001f;
                 for (int k = 0; k < 4096; ++k) x = sycl::fma(x, y, 0.5f);
                 out[i] = x;
             }).wait();
        }
        sycl::free(out, q);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "batchlas_tune --worker: warm-up failed: %s\n", e.what());
    }
}

// One request line per cell on stdin, its records and a "done" line on the saved stdout (worker.hh).
// Library output goes to stderr (the driver's log), so stray prints never reach the record stream.
int worker_main(const std::vector<std::string>& a) {
    double warm_s = 3;
    for (std::size_t i = 1; i < a.size(); ++i) {
        if (a[i] == "--warm-s" && i + 1 < a.size()) warm_s = std::stod(a[++i]);
        else die("--worker: unknown argument " + a[i]);
    }
    const int out = ::dup(1);
    if (out < 0 || ::dup2(2, 1) < 0) die("--worker: cannot redirect stdout");
    worker_warm(warm_s);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::string err;
        const auto req = parse_request(line, &err);
        if (!req) die("--worker: bad request (" + err + "): " + line);
        const OpSpec* spec = find_spec(req->first);
        if (!spec) die("--worker: unknown op " + req->first);
        const std::string text = outcome_text(spec->run_cell(req->second)) + Json().str("kind", "done").line();
        for (std::size_t done = 0; done < text.size();) {
            const ssize_t n = ::write(out, text.data() + done, text.size() - done);
            if (n <= 0) return 1;
            done += static_cast<std::size_t>(n);
        }
    }
    return 0;
}

int info_main(const std::string& result) {
    batchlas::Queue q(batchlas::Device("gpu"), kBackend);
    const auto& d = batchlas::select::device_of<kBackend>(q);
    std::ofstream(result) << Json().str("kind", "info").str("device", d.key).str("name", q.device().get_name()).line();
    return 0;
}

// ---- the driver --------------------------------------------------------------------------

// The live worker per GPU: our own compute process, so the guard does not count it as foreign.
std::mutex g_worker_mu;
std::map<int, long> g_worker_pid;

void set_worker_pid(int gpu, long pid) {
    std::lock_guard<std::mutex> lock(g_worker_mu);
    if (pid > 0) g_worker_pid[gpu] = pid;
    else g_worker_pid.erase(gpu);
}

long worker_pid(int gpu) {
    std::lock_guard<std::mutex> lock(g_worker_mu);
    const auto it = g_worker_pid.find(gpu);
    return it == g_worker_pid.end() ? -1 : it->second;
}


struct Cell {
    CellKey key;
    int round = 0, gpu = 0;
    std::string skip;  // why the cell was not measured
    std::vector<Attempt> attempts;
};

struct ChildOut {
    bool ok = false;
    bool guard = false;  // a foreign process was on the GPU when the child finished
    std::string error;
    std::vector<std::string> tolerated;  // idle foreign pids accepted before the child
    std::vector<Record> records;
    pid_t pid = -1;
};

class Driver {
public:
    // One driver per op (tiered runs take several): each gets its own subdirectory of g_tmp.
    Driver(const Opts& o, const OpSpec& spec) : o_(o), spec_(spec) {
        static int instances = 0;
        g_tmp = (fs::temp_directory_path() / ("batchlas_tune." + std::to_string(::getpid()))).string();
        tmp_ = g_tmp + "/" + std::to_string(instances++);
        fs::create_directories(tmp_);
        self_ = self_exe();
    }
    ~Driver() {
        std::error_code ec;
        fs::remove_all(tmp_, ec);
        fs::remove(g_tmp, ec);  // only once empty
    }

    // reps/warm < 0: the --reps/--warm options; `extra` arguments go to the child as they are.
    ChildOut child(int gpu, const std::string& bin, const std::string& dtype, const CellKey& key,
                   const std::vector<std::string>& arms, const std::string& mode, bool reverse,
                   const std::string& coverage = "", int reps = -1, double warm = -1,
                   const std::vector<std::string>& extra = {}) {
        const std::vector<std::string> tolerated = guard(gpu);
        const std::string id = std::to_string(gpu) + "_" + std::to_string(counter_++);
        const std::string res = tmp_ + "/r" + id + ".jsonl", log = tmp_ + "/l" + id + ".log";
        std::vector<std::string> argv{bin, "--cell", spec_.op(), "--dtype", dtype, "--key", key_arg(key),
                                      "--arms", join(arms, ","), "--mode", mode,
                                      "--reps", std::to_string(reps < 0 ? o_.reps : reps),
                                      "--warm", fmt(warm < 0 ? o_.warm : warm, "%g"), "--ld-pad", std::to_string(o_.ld_pad),
                                      "--result", res};
        if (reverse) argv.push_back("--reverse");
        argv.insert(argv.end(), extra.begin(), extra.end());
        std::vector<std::pair<std::string, std::string>> env{{"CUDA_DEVICE_ORDER", "PCI_BUS_ID"},
                                                             {"CUDA_VISIBLE_DEVICES", std::to_string(gpu)}};
        if (!coverage.empty()) env.push_back({"BATCHLAS_COVERAGE_OUT", coverage});
        const Spawned s = spawn_wait(argv, env, log, o_.cell_timeout);
        ChildOut c;
        c.pid = s.pid;
        c.tolerated = tolerated;
        if (s.status != 0) {
            c.error = (s.timed_out ? "timeout" : "exit " + std::to_string(s.status)) + ": " + tail(log);
            return c;
        }
        // gpu_guard.sh's after-check: a process that landed on the GPU mid-child voids its numbers.
        // Strict mode tolerated nothing, so any foreign entry is new.
        if (const auto fresh = o_.guard ? guard_new_foreign(apps(gpu), tolerated) : std::vector<std::string>{};
            !fresh.empty()) {
            c.guard = true;
            c.error = "guard: compute processes [" + join(fresh, ",") + "] on GPU " + std::to_string(gpu) + " after the child; numbers discarded";
            return c;
        }
        try {
            c.records = read_records(res);
            c.ok = true;
        } catch (const std::exception& e) {
            c.error = e.what();
        }
        std::error_code ec;
        fs::remove(res, ec);
        fs::remove(log, ec);
        return c;
    }

    // child() once more after a failure: a transient crash or a foreign process must not sink a cell.
    ChildOut child_retry(int gpu, const std::string& bin, const std::string& dtype, const CellKey& key,
                         const std::vector<std::string>& arms, const std::string& mode, bool reverse,
                         const std::string& coverage = "", int reps = -1, double warm = -1,
                         const std::vector<std::string>& extra = {}) {
        ChildOut c = child(gpu, bin, dtype, key, arms, mode, reverse, coverage, reps, warm, extra);
        if (c.ok) return c;
        std::printf("[gpu%d] %s %s %s: retrying after %s\n", gpu, spec_.op().c_str(), key_text(key).c_str(),
                    mode.c_str(), c.error.c_str());
        std::fflush(stdout);
        if (raw_.is_open())
            write(Json().str("kind", "retry").str("op", spec_.op()).str("dtype", dtype).key(key).str("mode", mode)
                      .str("error", c.error).line());
        return child(gpu, bin, dtype, key, arms, mode, reverse, coverage, reps, warm, extra);
    }

    // The spelling a binary's Auto picks at a cell, from its coverage `reached` row.
    std::string probe(int gpu, const std::string& bin, const std::string& dtype, const CellKey& key) {
        const std::string cov = tmp_ + "/cov" + std::to_string(gpu) + "_" + std::to_string(counter_++);
        const ChildOut c = child_retry(gpu, bin, dtype, key, {"auto"}, "probe", false, cov);
        const std::string file = cov + "." + std::to_string(c.pid);
        std::ifstream f(file);
        std::stringstream ss;
        ss << f.rdbuf();
        std::error_code ec;
        fs::remove(file, ec);
        const auto route = reached_route(ss.str(), spec_.op());
        if (!c.ok || !route) return "error:" + (c.ok ? std::string("no reached row") : c.error);
        return spec_.normalize_route(route->first, route->second);
    }

    std::string device_key(int gpu, std::string* name) {
        const std::string res = tmp_ + "/info" + std::to_string(gpu) + ".jsonl", log = res + ".log";
        const Spawned s = spawn_wait({self_, "--info", "--result", res},
                                     {{"CUDA_DEVICE_ORDER", "PCI_BUS_ID"}, {"CUDA_VISIBLE_DEVICES", std::to_string(gpu)}},
                                     log, 300);
        if (s.status != 0) die("device query on GPU " + std::to_string(gpu) + " failed: " + tail(log));
        const auto r = read_records(res).at(0);
        *name = r.get("name");
        return r.get("device");
    }

    int tune();
    int gate();
    std::string start(std::string* device) {
        const std::string name = preflight();
        *device = device_;
        return name;
    }
    ArmBatch race_fresh(const CellJob& j);
    void start_worker(WorkerProcess& w, int gpu);
    WorkerTry race_worker(WorkerProcess& w, const CellJob& j, bool full_guard);

private:
    const Opts& o_;
    const OpSpec& spec_;
    std::string tmp_, self_, dtype_, device_;
    std::atomic<int> counter_{0};
    std::mutex out_mu_;
    std::ofstream raw_;
    std::vector<std::string> cands_;

    std::string tolerated_at_start_;  // "gpu:pid(user),..;.." for the meta record
    CellRequest race_request(const CellJob& j) const;
    std::vector<std::string> race_args(const CellJob& j) const;
    AppScan apps(int gpu);
    double utilization(int gpu);
    std::vector<std::string> guard(int gpu);
    std::string preflight();
    void write(const std::string& line) {
        std::lock_guard<std::mutex> lock(out_mu_);
        raw_ << line;
        raw_.flush();
    }
    PassData timed(Cell& c, int pass, int attempt);
    void run_shard(int gpu, const std::vector<Cell*>& cells);
    void run_round(std::vector<Cell*> cells);
    std::string meta_line(const std::string& name) const;
    void convert(const std::string& jsonl);
};

AppScan Driver::apps(int gpu) {
    const std::string id = "nvidia-smi --id=" + std::to_string(gpu);
    std::string out = capture(id + " --query-compute-apps=pid --format=csv,noheader 2>/dev/null; echo rc=$?");
    const auto rc = out.rfind("rc=");
    if (rc == std::string::npos || out.substr(rc) != "rc=0")
        die("cannot query GPU " + std::to_string(gpu) + " with nvidia-smi (--no-guard measures without the guard)");
    out.resize(rc);
    AppScan scan = scan_compute_apps(out, long(::getpid()));
    if (scan.self) die("the driver holds a CUDA context on GPU " + std::to_string(gpu) + ": start it via the launcher");
    const std::string w = std::to_string(worker_pid(gpu));
    scan.foreign.erase(std::remove(scan.foreign.begin(), scan.foreign.end(), w), scan.foreign.end());
    return scan;
}

double Driver::utilization(int gpu) {
    const std::string util = capture("nvidia-smi --id=" + std::to_string(gpu) +
                                     " --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null");
    if (util.empty() || util.find_first_not_of("0123456789") != std::string::npos)
        die("cannot read GPU " + std::to_string(gpu) + " utilization ('" + util + "'); --no-guard skips the guard");
    return std::stod(util);
}

// The foreign pids this child runs beside (always empty in strict mode).
std::vector<std::string> Driver::guard(int gpu) {
    if (!o_.guard) return {};
    for (double waited = 0;; waited += 1) {
        const AppScan scan = apps(gpu);
        const GuardCheck g = guard_before(scan, utilization(gpu), o_.util_ceiling, o_.allow_idle_foreign);
        if (g.refuse.empty()) return g.tolerated;
        if (waited >= o_.guard_wait)
            die("GPU " + std::to_string(gpu) + " is busy (" + g.refuse + "); refusing to measure");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// One device key for the run (tables are per device), and the §10 / §13 warnings.
std::string Driver::preflight() {
    std::string name;
    for (int gpu : o_.devices) {
        std::string n;
        const std::string k = device_key(gpu, &n);
        if (device_.empty()) device_ = k, name = n;
        else if (k != device_)
            die("--devices mixes " + device_ + " and " + k + " (GPU " + std::to_string(gpu) + "): one device per run");
        const std::string q = "nvidia-smi --id=" + std::to_string(gpu);
        if (capture(q + " --query-gpu=display_active --format=csv,noheader 2>/dev/null") == "Enabled")
            std::fprintf(stderr, "batchlas_tune: warning: GPU %d drives a display, which slows L2-resident cells "
                                 "(docs/developer/agent-guide.md §13); prefer a headless GPU\n", gpu);
        if (!o_.allow_idle_foreign || !o_.guard) continue;
        std::vector<std::string> who;
        for (const std::string& pid : apps(gpu).foreign) {
            const std::string user = pid.find_first_not_of("0123456789") == std::string::npos
                                         ? capture("ps -o user= -p " + pid + " 2>/dev/null") : "";
            who.push_back(pid + "(" + (user.empty() ? "?" : user) + ")");
        }
        std::fprintf(stderr, "batchlas_tune: warning: --allow-idle-foreign: GPU %d tolerates idle foreign compute "
                             "processes [%s] while utilization <= %g%%\n", gpu, join(who, ",").c_str(), o_.util_ceiling);
        tolerated_at_start_ += (tolerated_at_start_.empty() ? "" : ";") + std::to_string(gpu) + ":" + join(who, ",");
    }
    for (const std::string& idx : split(capture("nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null"), '\n')) {
        if (idx.find_first_not_of("0123456789") != std::string::npos ||
            std::find(o_.devices.begin(), o_.devices.end(), std::stoi(idx)) != o_.devices.end())
            continue;
        const auto s = scan_compute_apps(
            capture("nvidia-smi --id=" + idx + " --query-compute-apps=pid --format=csv,noheader 2>/dev/null"), -1);
        if (!s.foreign.empty())
            std::fprintf(stderr, "batchlas_tune: warning: GPU %s has compute processes [%s]; another measurement on "
                                 "the box can skew these numbers (docs/developer/agent-guide.md §10)\n", idx.c_str(), join(s.foreign, ",").c_str());
    }
    return name;
}

PassData Driver::timed(Cell& c, int pass, int attempt) {
    const bool reverse = pass % 2 == 1;
    ChildOut ch = child_retry(c.gpu, self_, dtype_, c.key, cands_, "time", reverse);
    // Still failing: find the arms that crash a child on their own and time the rest without them.
    std::map<std::string, std::string> alone;
    if (!ch.ok && !ch.guard) {
        std::printf("[gpu%d] %s %s pass %d: running each arm alone to find the one that fails\n", c.gpu,
                    spec_.op().c_str(), key_text(c.key).c_str(), pass + 1);
        std::vector<std::string> keep;
        for (const auto& cand : cands_) {
            const ChildOut one = child(c.gpu, self_, dtype_, c.key, {cand}, "jit", false);
            if (one.ok) keep.push_back(cand);
            else alone[cand] = one.error;
        }
        if (!alone.empty() && !keep.empty()) ch = child(c.gpu, self_, dtype_, c.key, keep, "time", reverse);
    }
    PassData p;
    auto base = [&](const char* kind) {
        Json j;
        j.str("kind", kind).str("op", spec_.op()).str("dtype", dtype_).str("device", device_).key(c.key)
            .integer("pass", pass + 1).integer("attempt", attempt).boolean("reverse", reverse).integer("gpu", c.gpu);
        if (o_.allow_idle_foreign) j.str("tolerated_foreign", join(ch.tolerated, ","));
        return j;
    };
    for (const auto& cand : cands_) {
        const bool crashed = alone.count(cand) != 0;
        if (ch.ok && !crashed) continue;
        p.arms[cand] = {"error", crashed ? "crashed alone: " + alone[cand] : "child: " + ch.error, 0};
        write(base("pass").str("cand", cand).str("status", "error").str("reason", p.arms[cand].reason).line());
    }
    if (!ch.ok) p.error = ch.error;
    for (const Record& r : ch.records) {
        const std::string arm = r.get("arm");
        if (r.get("kind") == "rep") {
            write(base("rep").str("cand", arm).integer("rep", std::stoll(r.get("rep")))
                      .integer("slot", std::stoll(r.get("slot"))).num("ms", r.number("ms")).line());
            continue;
        }
        p.arms[arm] = {r.get("status"), r.get("reason"), r.number("median_ms")};
        write(base("pass").str("cand", arm).str("status", r.get("status")).str("reason", r.get("reason"))
                  .num("median_ms", r.number("median_ms")).num("residual", r.number("residual"))
                  .integer("info_nonzero", std::stoll(r.get("info_nonzero", "0")))
                  .integer("reps", std::stoll(r.get("reps", "0"))).line());
    }
    std::string summary;
    std::vector<std::pair<double, std::string>> ok;
    for (const auto& [cand, a] : p.arms)
        if (a.status == "ok") ok.push_back({a.median, cand});
    std::sort(ok.begin(), ok.end());
    for (const auto& [ms, cand] : ok) summary += " | " + cand + " " + fmt(ms);
    std::printf("[gpu%d] %s %s pass %d%s: %s%s\n", c.gpu, spec_.op().c_str(), key_text(c.key).c_str(), pass + 1,
                attempt ? " (re-measure)" : "", ch.ok ? "" : ("ERROR " + ch.error).c_str(),
                summary.empty() ? " (nothing ran)" : summary.c_str());
    std::fflush(stdout);
    return p;
}

CellRequest Driver::race_request(const CellJob& j) const {
    CellRequest r;
    r.dtype = j.dtype;
    r.key = j.key;
    r.arms = j.arms;
    r.mode = "race";
    r.warm_s = j.p.warm_topup_s;
    r.ld_pad = o_.ld_pad;
    r.tier = to_string(j.tier);
    r.min_reps = j.p.min_reps;
    r.max_reps = j.p.max_reps;
    r.confidence = j.p.confidence;
    r.alternate_reverse = j.p.alternate_reverse;
    r.seed_order = j.arms;
    return r;
}

std::vector<std::string> Driver::race_args(const CellJob& j) const {
    std::vector<std::string> a{"--tier", to_string(j.tier), "--min-reps", std::to_string(j.p.min_reps), "--max-reps",
                               std::to_string(j.p.max_reps), "--confidence", fmt(j.p.confidence, "%.17g"),
                               "--seed-order", join(j.arms, ",")};
    if (j.p.alternate_reverse) a.push_back("--alternate-reverse");
    return a;
}

// One fresh `--cell --mode race` child (the audit, --no-worker, the worker's fallback). Arms that
// crash a child even alone are `error`; the rest race without them.
ArmBatch Driver::race_fresh(const CellJob& j) {
    const auto extra = race_args(j);
    const double warm = j.p.warm_topup_s;
    ChildOut ch = child_retry(j.gpu, self_, j.dtype, j.key, j.arms, "race", false, "", 1, warm, extra);
    std::map<std::string, std::string> alone;
    if (!ch.ok && !ch.guard) {
        std::vector<std::string> keep;
        for (const auto& a : j.arms) {
            const ChildOut one = child(j.gpu, self_, j.dtype, j.key, {a}, "jit", false);
            if (one.ok) keep.push_back(a);
            else alone[a] = one.error;
        }
        if (!alone.empty() && !keep.empty()) ch = child(j.gpu, self_, j.dtype, j.key, keep, "race", false, "", 1, warm, extra);
    }
    std::map<std::string, ArmOutcome> got;
    for (ArmOutcome& o : outcomes_from_records(ch.records)) got[o.arm] = std::move(o);
    ArmBatch b;
    if (!ch.ok) b.error = ch.error;
    for (const auto& a : j.arms) {
        if (alone.count(a)) b.arms.push_back({a, "error", "crashed alone: " + alone[a], {}, {}, 0, 0});
        else if (got.count(a) && !got[a].status.empty()) b.arms.push_back(got[a]);
        else b.arms.push_back({a, "error", "child: " + ch.error, {}, {}, 0, 0});
    }
    return b;
}

// The full guard (utilization too) runs before the worker starts.
void Driver::start_worker(WorkerProcess& w, int gpu) {
    guard(gpu);
    const std::string log = tmp_ + "/worker" + std::to_string(gpu) + "_" + std::to_string(counter_++) + ".log";
    if (!w.start({self_, "--worker", "--warm-s", "3"},
                 {{"CUDA_DEVICE_ORDER", "PCI_BUS_ID"}, {"CUDA_VISIBLE_DEVICES", std::to_string(gpu)}}, log))
        die("cannot start the worker for GPU " + std::to_string(gpu));
    set_worker_pid(gpu, long(w.pid()));
}

// One cell on the running worker. Between cells the guard checks for foreign compute processes;
// `full_guard` first idles 1 s so the worker's own last cell leaves the utilization sample.
WorkerTry Driver::race_worker(WorkerProcess& w, const CellJob& j, bool full_guard) {
    WorkerTry r;
    std::vector<std::string> tolerated;
    if (full_guard && o_.guard) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        tolerated = guard(j.gpu);
    }
    for (double waited = 0; o_.guard && !full_guard; waited += 1) {
        const GuardCheck g = guard_before(apps(j.gpu), 0, o_.util_ceiling, o_.allow_idle_foreign);
        if (g.refuse.empty()) {
            tolerated = g.tolerated;
            break;
        }
        if (waited >= o_.guard_wait) die("GPU " + std::to_string(j.gpu) + " is busy (" + g.refuse + "); refusing to measure");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::vector<Record> recs;
    WorkerProcess::Got got = WorkerProcess::Got::eof;
    if (w.send(request_line(spec_.op(), race_request(j)))) got = w.receive(o_.cell_timeout, &recs);
    if (got != WorkerProcess::Got::done) {
        r.error = std::string(got == WorkerProcess::Got::timeout ? "worker timeout" : "worker exited") + ": " +
                  tail(w.log());
        return r;
    }
    if (const auto fresh = o_.guard ? guard_new_foreign(apps(j.gpu), tolerated) : std::vector<std::string>{};
        !fresh.empty()) {
        r.guard = true;
        r.error = "guard: compute processes [" + join(fresh, ",") + "] on GPU " + std::to_string(j.gpu) +
                  " during the cell; numbers discarded";
        return r;
    }
    r.arms = outcomes_from_records(recs);
    for (const std::string& a : j.arms)
        if (std::none_of(r.arms.begin(), r.arms.end(), [&](const ArmOutcome& o) { return o.arm == a && !o.status.empty(); })) {
            r.error = "worker: no outcome for " + a;
            return r;
        }
    r.ok = true;
    return r;
}

void Driver::run_shard(int gpu, const std::vector<Cell*>& cells) {
    for (Cell* c : cells) c->gpu = gpu;
    if (o_.jit)
        for (Cell* c : cells) {
            const ChildOut ch = child(gpu, self_, dtype_, c->key, cands_, "jit", false);
            if (!ch.ok) std::printf("[gpu%d] %s jit: ERROR %s\n", gpu, key_text(c->key).c_str(), ch.error.c_str());
        }
    for (Cell* c : cells) c->attempts.assign(1, std::vector<PassData>(std::size_t(o_.passes)));
    for (int p = 0; p < o_.passes; ++p)
        for (Cell* c : cells) c->attempts[0][std::size_t(p)] = timed(*c, p, 0);
    std::vector<Cell*> again;
    for (Cell* c : cells)
        if (attempt_needs_remeasure(c->attempts[0], cands_, o_.remeasure)) again.push_back(c);
    for (Cell* c : again) c->attempts.emplace_back(std::size_t(o_.passes));
    for (int p = 0; p < o_.passes; ++p)
        for (Cell* c : again) c->attempts[1][std::size_t(p)] = timed(*c, p, 1);
}

void Driver::run_round(std::vector<Cell*> cells) {
    std::vector<std::vector<Cell*>> shards(o_.devices.size());
    for (std::size_t i = 0; i < cells.size(); ++i) shards[i % shards.size()].push_back(cells[i]);
    std::vector<std::thread> workers;
    for (std::size_t g = 0; g < shards.size(); ++g)
        workers.emplace_back([this, g, &shards] { run_shard(o_.devices[g], shards[g]); });
    for (auto& w : workers) w.join();
}

std::string Driver::meta_line(const std::string& name) const {
    std::string missing;
    const auto hash = kernel_hash(o_.repo, spec_.kernel_sources(), &missing);
    if (!hash) die("kernel source " + missing + " not found under --repo " + o_.repo);
    std::string sha = capture("git -C '" + o_.repo + "' rev-parse --short=8 HEAD 2>/dev/null");
    if (!capture("git -C '" + o_.repo + "' status --porcelain --untracked-files=no 2>/dev/null").empty()) sha += "-dirty";
    char date[16];
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&now));
    std::vector<std::string> devs;
    for (int d : o_.devices) devs.push_back(std::to_string(d));
    return Json().str("kind", "meta").integer("schema", kSchema).str("op", spec_.op()).str("dtype", dtype_)
        .str("device", device_).str("device_name", name).str("batchlas", sha.empty() ? "unknown" : sha)
        .str("kernels", *hash).str("kernel_sources", join(spec_.kernel_sources(), "|")).str("date", date)
        .str("keys", join(spec_.key_names(), " ")).str("candidates", join(cands_, "|"))
        .integer("reps", o_.reps).num("warm_s", o_.warm).integer("passes", o_.passes).num("tie", kTie)
        .num("remeasure", o_.remeasure).num("refine_ratio", o_.refine ? o_.refine_ratio : 0.0)
        .num("cap_gib", o_.cap_gib).integer("ld_pad", o_.ld_pad).str("devices", join(devs, ","))
        .boolean("allow_idle_foreign", o_.allow_idle_foreign).str("tolerated_foreign", tolerated_at_start_)
        .str("argv", join(o_.argv, " ")).line();
}

void Driver::convert(const std::string& jsonl) {
    const std::string script = o_.repo + "/scripts/sweep_to_table.py";
    const Spawned s = spawn_wait({"/usr/bin/env", "python3", script, "--tuner", jsonl, "--out", o_.out}, {}, "", 600);
    if (s.status != 0) die("table conversion failed: python3 " + script + " --tuner " + jsonl);
}

int Driver::tune() {
    const std::string name = preflight();
    const double cap = o_.cap_gib * 1024.0 * 1024.0 * 1024.0;
    const std::string refine_key = spec_.refine_key();
    for (const std::string& dtype : o_.dtypes) {
        dtype_ = dtype;
        cands_ = spec_.candidates(dtype);
        const std::string path = o_.raw + "/" + spec_.op() + "." + dtype + "." + device_ + ".jsonl";
        fs::create_directories(o_.raw);
        raw_.open(path, std::ios::trunc);
        if (!raw_) die("cannot write " + path);
        write(meta_line(name));
        std::map<CellKey, Cell> cells;
        std::set<std::string> stalled;
        std::vector<CellKey> next = spec_.grid(dtype, o_.grid);
        for (int round = 0; !next.empty(); ++round) {
            std::vector<Cell*> todo;
            for (const CellKey& k : next) {
                Cell& c = cells[k];
                c.key = k;
                c.round = round;
                if (spec_.bytes(dtype, k) > cap) c.skip = "cap: " + fmt(spec_.bytes(dtype, k) / cap * o_.cap_gib) + " GiB";
                else todo.push_back(&c);
            }
            std::printf("== %s %s %s round %d: %zu cells (%zu over the cap)\n", spec_.op().c_str(), dtype.c_str(),
                        device_.c_str(), round, todo.size(), next.size() - todo.size());
            run_round(todo);
            next.clear();
            if (!o_.refine) break;
            std::map<CellKey, std::map<std::string, double>> measured;
            for (const auto& [k, c] : cells)
                measured[k] = c.skip.empty() ? final_times(c.attempts, cands_) : std::map<std::string, double>{};
            RefineRound r = refine_round(measured, refine_key, cands_, kTie, o_.refine_ratio);
            next = std::move(r.next);
            stalled.insert(r.stalled.begin(), r.stalled.end());
        }
        int ok = 0, none = 0, skipped = 0;
        for (const auto& [k, c] : cells) {
            int used = -1;
            const auto t = final_times(c.attempts, cands_, &used);
            const std::string status = !c.skip.empty() ? "skipped" : t.empty() ? "none" : "ok";
            (status == "ok" ? ok : status == "none" ? none : skipped)++;
            write(Json().str("kind", "cell").str("op", spec_.op()).str("dtype", dtype).str("device", device_).key(k)
                      .integer("round", c.round).boolean("refined", c.round > 0).str("status", status)
                      .str("reason", c.skip).integer("final_attempt", used)
                      .str("ranked", t.empty() ? "" : join(rank(t, cands_, kTie), "|")).line());
        }
        for (const std::string& s : stalled) {
            write(Json().str("kind", "stalled").str("op", spec_.op()).str("dtype", dtype).str("note", s).line());
            std::printf("== refinement stalled: %s\n", s.c_str());
        }
        raw_.close();
        std::printf("== wrote %s: %d cells ranked, %d with no candidate timed in every pass, %d over the cap, "
                    "%zu stalled edges\n", path.c_str(), ok, none, skipped, stalled.size());
        // The ledger records a protocol-flag run as tier custom (docs/design/tiered-tuning.md, driver interface).
        try {
            std::vector<std::string> fams;
            for (const std::string& c : cands_)
                if (std::find(fams.begin(), fams.end(), c.substr(0, c.find(':'))) == fams.end())
                    fams.push_back(c.substr(0, c.find(':')));
            const auto fh = family_hashes(o_.repo, spec_.kernel_block(o_.repo), fams);
            import_schema1(path, o_.ledger, fh, kernel_hash(o_.repo, spec_.kernel_sources()).value_or(""), Tier::custom,
                           spec_.axes());
            std::printf("== recorded as a custom run under %s\n", ledger_dir(o_.ledger, spec_.op(), dtype, device_).c_str());
        } catch (const std::exception& e) {
            die(std::string("ledger: ") + e.what());
        }
        if (!o_.out.empty()) convert(path);
    }
    return 0;
}

// §10.3 / plan §4: pin the old choice against Auto, interleaved, two passes in reversed order.
int Driver::gate() {
    preflight();
    struct GateCell {
        std::string dtype;
        CellKey key;
        std::string old;
    };
    std::vector<GateCell> cells;
    std::vector<std::string> names;
    for (const std::string& k : spec_.key_names()) names.push_back(split(k, ':').front());
    if (!o_.old_csv.empty()) {
        std::ifstream f(o_.old_csv);
        if (!f) die("cannot read " + o_.old_csv);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string err;
        const auto rows = parse_old_csv(ss.str(), names, o_.dtype_given ? o_.dtypes : std::vector<std::string>{}, &err);
        if (!rows) die(o_.old_csv + ": " + err);
        // An old binary wrote native:<algo> (e.g. "native:lpanel"): compare canonical forms.
        for (const OldChoice& r : *rows) {
            const auto [origin, algo] = split_origin(r.old);
            cells.push_back({r.dtype, r.key, spec_.normalize_route(origin, algo)});
        }
        std::printf("== gate: %zu cells from %s\n", cells.size(), o_.old_csv.c_str());
    } else {
        for (const auto& dtype : o_.dtypes)
            for (const CellKey& k : spec_.grid(dtype, o_.grid)) cells.push_back({dtype, k, ""});
    }
    if (!o_.raw.empty()) {
        fs::create_directories(o_.raw);
        raw_.open(o_.raw + "/gate." + spec_.op() + "." + device_ + ".jsonl", std::ios::trunc);
        if (raw_.is_open())
            write(Json().str("kind", "guard").boolean("guard", o_.guard)
                      .boolean("allow_idle_foreign", o_.allow_idle_foreign)
                      .str("tolerated_foreign", tolerated_at_start_).line());
    }
    const double cap = o_.cap_gib * 1024.0 * 1024.0 * 1024.0;
    std::vector<std::string> rows(cells.size());
    GateCounts counts;
    std::mutex count_mu;
    auto finish = [&](std::size_t i, int gpu, std::string row, const std::string& verdict) {
        rows[i] = std::move(row) + "," + verdict;
        std::lock_guard<std::mutex> lock(count_mu);
        gate_count(counts, verdict);
        std::printf("[gpu%d] gate %s\n", gpu, rows[i].c_str());
        std::fflush(stdout);
    };
    auto work = [&](std::size_t g) {
        const int gpu = o_.devices[g];
        for (std::size_t i = g; i < cells.size(); i += o_.devices.size()) {
            GateCell& c = cells[i];
            std::string row = c.dtype;
            for (const KV& f : c.key) row += "," + csv_field(f.value);
            if (spec_.bytes(c.dtype, c.key) > cap) {
                finish(i, gpu, row + "," + csv_field(c.old) + ",,,,,", "cap");
                continue;
            }
            if (c.old.empty()) c.old = probe(gpu, o_.parent_bin, c.dtype, c.key);
            const std::string now = probe(gpu, self_, c.dtype, c.key);
            row += "," + csv_field(c.old) + "," + csv_field(now);
            if (c.old.rfind("error:", 0) == 0 || now.rfind("error:", 0) == 0) {
                finish(i, gpu, row + ",,,,", "ERROR");
                continue;
            }
            if (c.old == now) {
                finish(i, gpu, row + ",,,,", "same");
                continue;
            }
            double ratio[2] = {NAN, NAN}, old_ms = NAN, new_ms = NAN;
            bool bad = false;
            for (int p = 0; p < 2; ++p) {
                const ChildOut ch = child_retry(gpu, self_, c.dtype, c.key, {"auto", c.old}, "time", p == 1);
                std::map<std::string, double> med;
                for (const Record& r : ch.records) {
                    if (r.get("kind") != "arm") continue;
                    bad = bad || r.get("status") != "ok";
                    med[r.get("arm")] = r.number("median_ms");
                    if (raw_.is_open())
                        write(Json().str("kind", "gate").str("op", spec_.op()).str("dtype", c.dtype).key(c.key)
                                  .integer("pass", p + 1).str("arm", r.get("arm")).str("status", r.get("status"))
                                  .str("reason", r.get("reason")).num("median_ms", r.number("median_ms")).line());
                }
                bad = bad || !ch.ok || med.size() != 2;
                ratio[p] = med.count("auto") && med.count(c.old) ? med["auto"] / med[c.old] : NAN;
                if (p == 0) {
                    old_ms = med.count(c.old) ? med[c.old] : NAN;
                    new_ms = med.count("auto") ? med["auto"] : NAN;
                }
            }
            finish(i, gpu,
                   row + "," + fmt(old_ms, "%.6g") + "," + fmt(new_ms, "%.6g") + "," + fmt(ratio[0], "%.3f") + "," +
                       fmt(ratio[1], "%.3f"),
                   gate_verdict(ratio[0], ratio[1], bad, o_.gate_limit));
        }
    };
    std::vector<std::thread> workers;
    for (std::size_t g = 0; g < o_.devices.size(); ++g) workers.emplace_back(work, g);
    for (auto& w : workers) w.join();
    std::ofstream out(o_.gate_csv);
    out << "dtype," << join(names, ",") << ",old,new,old_ms_p1,new_ms_p1,ratio_p1,ratio_p2,verdict\n";
    for (const auto& r : rows) out << r << "\n";
    const int rc = gate_exit_code(counts);
    std::printf("== gate: %zu cells: %s -> %s\n", cells.size(), gate_summary(counts).c_str(), o_.gate_csv.c_str());
    if (rc == 3) std::printf("== gate INCOMPLETE (exit 3): ERROR/BAD_ROW rows, or nothing was gated\n");
    return rc;
}

// One flock per GPU for the whole run: two tuners never share a device.
std::vector<int> lock_devices(const Opts& o) {
    std::vector<int> fds;
    for (int d : o.devices) {
        const std::string path = o.lock_dir + "/batchlas_tune_gpu" + std::to_string(d) + ".lock";
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd < 0) die("cannot open " + path);
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            std::printf("waiting for %s (another tuner holds GPU %d)\n", path.c_str(), d);
            std::fflush(stdout);
            ::flock(fd, LOCK_EX);
        }
        fds.push_back(fd);
    }
    return fds;
}

// The tiered seam (tiered_driver.hh CellMeasurer): one Driver per op, one persistent worker per GPU
// shared by every op, restarted before a cell with a smaller per-item footprint than it has run.
// Failures follow race_on_worker (worker.hh).
class TieredMeasurer : public CellMeasurer {
public:
    TieredMeasurer(const Opts& o, const std::vector<std::string>& ops) : o_(o) {
        for (const std::string& op : ops) drivers_[op] = std::make_unique<Driver>(o, *find_spec(op));
        for (int gpu : o.devices) workers_[gpu] = std::make_unique<WorkerProcess>(), gates_[gpu] = WorkerGate(60);
    }
    ~TieredMeasurer() override {
        for (auto& [gpu, w] : workers_) stop(gpu);
    }
    Driver& first() { return *drivers_.begin()->second; }
    bool persistent() const override { return o_.worker; }
    ArmBatch measure_fresh(const CellJob& j) override { return drivers_.at(j.spec->op())->race_fresh(j); }
    ArmBatch measure(const CellJob& j) override {
        if (!o_.worker) return measure_fresh(j);
        Driver& d = *drivers_.at(j.spec->op());
        WorkerProcess& w = *workers_.at(j.gpu);
        WorkerGate& gate = gates_.at(j.gpu);
        if (w.alive() && gate.restart_before(j.footprint)) stop(j.gpu);  // the carve-out order
        auto attempt = [&](const std::vector<std::string>& arms) {
            CellJob part = j;
            part.arms = arms;
            if (!w.alive()) {
                d.start_worker(w, j.gpu);
                gate.started();
                gate.guarded(now());
            }
            const bool full = gate.full_guard_due(now());
            WorkerTry r = d.race_worker(w, part, full);
            if (full) gate.guarded(now());
            if (r.ok) gate.ran(j.footprint);
            for (const ArmOutcome& a : r.arms)
                if (a.status == "error") r.error = "arm " + a.arm + " error: " + a.reason;
            if (!r.error.empty())
                std::printf("[gpu%d] %s %s: worker: %s\n", j.gpu, j.spec->op().c_str(), key_text(j.key).c_str(),
                            r.error.c_str());
            std::fflush(stdout);
            return r;
        };
        auto fresh = [&](const std::vector<std::string>& arms) {
            CellJob part = j;
            part.arms = arms;
            return measure_fresh(part);
        };
        return race_on_worker(j.arms, errors(j.spec->op() + "." + j.dtype), attempt, [&] { stop(j.gpu); }, fresh);
    }

private:
    const Opts& o_;
    std::map<std::string, std::unique_ptr<Driver>> drivers_;
    std::map<int, std::unique_ptr<WorkerProcess>> workers_;
    std::map<int, WorkerGate> gates_;
    std::mutex errors_mu_;
    std::map<std::string, ArmErrors> errors_;  // per op.dtype; GPU threads share it
    const std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();

    double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
    ArmErrors& errors(const std::string& od) {
        std::lock_guard<std::mutex> lock(errors_mu_);
        return errors_[od];
    }
    void stop(int gpu) {
        workers_.at(gpu)->stop();
        set_worker_pid(gpu, -1);
    }
};

// --plan names the device from nvidia-smi's compute capability (no CUDA context), as select.cc does.
std::string plan_device_key(const Opts& o) {
    if (!o.device_key.empty()) return o.device_key;
    const std::string gpu = std::to_string(o.devices.empty() ? 0 : o.devices[0]);
    const std::string cc = capture("nvidia-smi --id=" + gpu + " --query-gpu=compute_cap --format=csv,noheader 2>/dev/null");
    const auto dot = cc.find('.');
    if (dot == std::string::npos) die("--plan: cannot read GPU " + gpu + "'s compute capability; pass --device-key sm_NN");
    return "sm_" + cc.substr(0, dot) + cc.substr(dot + 1);
}

void usage() {
    std::puts(
        "usage: batchlas_tune <op>[,<op>..|all] --tier preview|coarse|deep --devices N[,M..] [options]   ledger + tables\n"
        "       batchlas_tune <op>[,..] --tier T --plan [--devices N | --device-key sm_NN]   plan and estimate, no GPU\n"
        "       batchlas_tune --status | --import-raw RAW.jsonl | --list\n"
        "       batchlas_tune <op> --devices N --gate (--old-csv F | --parent-bin B) --gate-csv OUT [options]\n"
        "tiered: --budget H --progress-fd N --ledger DIR --out DIR --cell-overhead-s 0.49 --no-worker --audit-fraction F\n"
        "custom (expert protocol, the two-pass path, schema-1 raw): --reps 16 --warm 1.5 --passes 2\n"
        "         --remeasure 0.10 --refine-ratio 1.1 --no-refine --no-jit --ld-pad 0 --raw DIR\n"
        "options: --dtype float,double,cfloat,cdouble --cap-gib 4 --cell-timeout 1800 --no-guard --guard-wait 300\n"
        "         --util-ceiling 5 --allow-idle-foreign\n"
        "         --lock-dir /tmp --repo DIR  grid: --n-list --batches --nrhs-list --uplo  --grid key=v1:v2\n"
        "         gate: --gate-limit 1.05\n"
        "see tools/tune/README.md");
}

int list_main(const std::string& repo) {
    for (const OpSpec* s : all_specs()) {
        std::string missing;
        const auto hash = kernel_hash(repo, s->kernel_sources(), &missing);
        std::printf("%s keys: %s kernels=%s\n", s->op().c_str(), join(s->key_names(), " ").c_str(),
                    hash ? hash->c_str() : ("missing " + missing).c_str());
        for (const char* d : {"float", "double", "cfloat", "cdouble"})
            std::printf("  %-7s %s\n", d, join(s->candidates(d), " ").c_str());
    }
    return 0;
}

std::vector<std::string> csv_list(const std::string& v) { return split(v, ','); }

TieredOpts tiered_opts(const Opts& o) {
    TieredOpts t;
    t.ops = o.ops;
    t.dtypes = o.dtypes;
    t.devices = o.devices;
    t.tier = *o.tier;
    t.budget_h = o.budget_h;
    t.cap_gib = o.cap_gib;
    t.overhead_s = o.overhead_s;
    t.audit_fraction = o.audit_fraction;
    t.plan = o.plan;
    t.progress_fd = o.progress_fd;
    t.repo = o.repo;
    t.ledger_root = o.ledger;
    t.out = o.out;
    t.argv = join(o.argv, " ");
    t.grid = o.grid;
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    if (a.empty() || a[0] == "--help" || a[0] == "-h") {
        usage();
        return a.empty() ? 2 : 0;
    }
    if (a[0] == "--cell") return child_main(a);
    if (a[0] == "--worker") return worker_main(a);
    if (a[0] == "--info") return info_main(a.size() > 2 ? a[2] : "/dev/stdout");
    Opts o;
    o.argv.assign(argv, argv + argc);
    bool list = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto val = [&] {
            if (i + 1 >= a.size()) die(a[i] + " needs a value");
            return a[++i];
        };
        const std::string& f = a[i];
        // An expert protocol flag: the run is tier custom, on the two-pass path.
        auto custom = [&, flag = f] {
            if (o.custom_flag.empty()) o.custom_flag = flag;
        };
        if (f == "--list") list = true;
        else if (f == "--dtype") o.dtypes = csv_list(val()), o.dtype_given = true;
        else if (f == "--devices") {
            o.devices.clear();
            o.devices_given = true;
            for (const auto& d : csv_list(val())) o.devices.push_back(std::stoi(d));
        } else if (f == "--repo") o.repo = val();
        else if (f == "--raw") o.raw = val(), custom();
        else if (f == "--out") o.out = val();
        else if (f == "--reps") o.reps = std::stoi(val()), custom();
        else if (f == "--warm") o.warm = std::stod(val()), custom();
        else if (f == "--passes") o.passes = std::stoi(val()), custom();
        else if (f == "--remeasure") o.remeasure = std::stod(val()), custom();
        else if (f == "--refine-ratio") o.refine_ratio = std::stod(val()), custom();
        else if (f == "--no-refine") o.refine = false, custom();
        else if (f == "--no-jit") o.jit = false, custom();
        else if (f == "--ld-pad") o.ld_pad = std::stoi(val()), custom();
        else if (f == "--tier") {
            const auto t = parse_tier(val());
            if (!t || *t == Tier::custom || *t == Tier::transcribed) die("--tier takes preview, coarse or deep");
            o.tier = t;
        } else if (f == "--budget") o.budget_h = std::stod(val());
        else if (f == "--plan") o.plan = true;
        else if (f == "--status") o.status = true;
        else if (f == "--progress-fd") o.progress_fd = std::stoi(val());
        else if (f == "--ledger") o.ledger = val();
        else if (f == "--import-raw") o.import_raw = val();
        else if (f == "--device-key") o.device_key = val();
        else if (f == "--cell-overhead-s") o.overhead_s = std::stod(val());
        else if (f == "--no-worker") o.worker = false;
        else if (f == "--audit-fraction") o.audit_fraction = std::stod(val());
        else if (f == "--cap-gib") o.cap_gib = std::stod(val());
        else if (f == "--cell-timeout") o.cell_timeout = std::stod(val());
        else if (f == "--no-guard") o.guard = false;
        else if (f == "--guard-wait") o.guard_wait = std::stod(val());
        else if (f == "--util-ceiling") o.util_ceiling = std::stod(val());
        else if (f == "--allow-idle-foreign") o.allow_idle_foreign = true;
        else if (f == "--lock-dir") o.lock_dir = val();
        else if (f == "--n-list") o.grid["n"] = csv_list(val());
        else if (f == "--batches") o.grid["batch"] = csv_list(val());
        else if (f == "--nrhs-list") o.grid["nrhs"] = csv_list(val());
        else if (f == "--uplo") o.grid["uplo"] = csv_list(val());
        else if (f == "--grid") {
            const std::string g = val();
            const auto eq = g.find('=');
            if (eq == std::string::npos) die("--grid wants key=v1:v2:...");
            o.grid[g.substr(0, eq)] = split(g.substr(eq + 1), ':');
        } else if (f == "--gate") o.gate = true;
        else if (f == "--old-csv") o.old_csv = val();
        else if (f == "--parent-bin") o.parent_bin = val();
        else if (f == "--gate-csv") o.gate_csv = val();
        else if (f == "--gate-limit") o.gate_limit = std::stod(val());
        else if (!f.empty() && f[0] != '-' && o.op.empty()) o.op = f;
        else die("unknown argument " + f + " (--help)");
    }
    if (list) return list_main(o.repo);
    if (o.ledger.empty()) o.ledger = o.repo + "/benchmarks/results/tuning/ledger";
    try {
        if (o.status) return status_main(o.repo, o.ledger, o.repo + "/tuned");
        if (!o.import_raw.empty()) return import_raw_main(o.repo, o.ledger, o.import_raw);
    } catch (const std::exception& e) {
        die(e.what());
    }
    if (o.op == "all")
        for (const OpSpec* s : all_specs()) o.ops.push_back(s->op());
    else o.ops = csv_list(o.op);
    if (o.ops.empty()) die("no op given (--list)");
    for (const std::string& op : o.ops)
        if (!find_spec(op)) die("unknown op '" + op + "' (--list)");
    o.ops = op_order(o.ops);
    const bool tiered = !o.gate && o.custom_flag.empty();
    if (!o.gate && !o.custom_flag.empty() && o.tier)
        die(o.custom_flag + " sets tier custom (the two-pass protocol); drop it or --tier");
    if (tiered && !o.tier)
        die("give --tier preview|coarse|deep, or a protocol flag (--reps, --warm, --passes, --remeasure, --refine-ratio, "
            "--no-refine, --no-jit, --ld-pad, --raw) for a custom two-pass run");
    if (!tiered && o.ops.size() != 1) die("--gate and the custom protocol take one op");
    if (!tiered && (o.plan || o.budget_h > 0 || o.progress_fd >= 0)) die("--plan, --budget and --progress-fd need --tier");
    for (const std::string& op : o.ops) {
        const OpSpec* s = find_spec(op);
        for (const auto& d : o.dtypes) (void)s->candidates(d);
        std::ifstream f(o.repo + "/" + s->spec_file());
        std::stringstream ss;
        ss << f.rdbuf();
        if (!f) std::fprintf(stderr, "batchlas_tune: %s not found under --repo; kernel list unchecked\n", s->spec_file().c_str());
        else if (parse_kernel_list(ss.str()) != s->kernel_sources())
            die(s->spec_file() + ": its kernel-sources block differs from this binary's list (rebuild)");
    }
    if (!std::getenv("BATCHLAS_TUNE_LAUNCHED"))
        die("start the tuner as batchlas_tune (launcher.cc), not batchlas_tune_impl: the launcher keeps the driver "
            "off every GPU");
    if (o.plan) {
        try {
            return run_tiered(tiered_opts(o), {plan_device_key(o), ""}, nullptr);
        } catch (const std::exception& e) {
            die(e.what());
        }
    }
    std::optional<std::string> parent_visible;
    if (const char* v = std::getenv("BATCHLAS_TUNE_PARENT_CVD")) parent_visible = v;
    if (const auto why = devices_problem(o.devices, o.devices_given, parent_visible)) die(*why);
    if (o.reps < 1 || o.passes < 1) die("need --reps >= 1, --passes >= 1");
    if (std::set<int>(o.devices.begin(), o.devices.end()).size() != o.devices.size())
        die("--devices lists a GPU twice: one child per GPU at a time");
    for (const char* var : {"ROUTE"})
        for (char** e = environ; *e; ++e)
            if (std::strncmp(*e, "BATCHLAS_", 9) == 0 && std::strstr(*e, (std::string("_") + var + "=").c_str()))
                std::fprintf(stderr, "batchlas_tune: warning: %s is set and reaches the op's children\n", *e);
    const auto locks = lock_devices(o);
    if (tiered) {
        TieredMeasurer m(o, o.ops);
        RunIdentity id;
        id.device_name = m.first().start(&id.device);
        try {
            return run_tiered(tiered_opts(o), id, &m);
        } catch (const std::exception& e) {
            die(e.what());
        }
    }
    const OpSpec* spec = find_spec(o.ops[0]);
    Driver drv(o, *spec);
    if (o.gate) {
        if (o.gate_csv.empty() || (o.old_csv.empty() == o.parent_bin.empty()))
            die("--gate needs --gate-csv and exactly one of --old-csv, --parent-bin");
        return drv.gate();
    }
    if (o.raw.empty()) o.raw = o.repo + "/benchmarks/results/tuning";
    return drv.tune();
}
