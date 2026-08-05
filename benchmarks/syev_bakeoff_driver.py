#!/usr/bin/env python3
"""SYEV small-n bake-off + mid-range block-size retune, with HARD resource caps.

Saturation = the knee: the batch at which steady-state compute dominates launch
overhead and timing is measurable. NOT the asymptote, NOT "all available memory".

Caps (ceilings, not targets):
  * batch chosen so estimated device footprint <= FOOTPRINT_BUDGET (2 GB)
  * absolute batch ceiling BATCH_MAX
  * per-invocation wall timeout INVOCATION_TIMEOUT
  * climbing stops as soon as a 4x batch increase buys < KNEE_TOL
Results are appended to a JSONL as they are produced, so a kill loses nothing.
"""
import csv, json, os, subprocess, sys, time

BUILD = "/home/jonaslacour/BatchLAS/.claude/worktrees/syev-perf-ideation/build/presets/cuda"
OUT = "/home/jonaslacour/.claude/jobs/00fd876a/tmp/results2"
DEV = "1"

FOOTPRINT_BUDGET = 1.5e9     # bytes of device memory we allow ourselves
BATCH_MAX = 16384            # absolute ceiling regardless of how small n is
BUDGET_MIB = 1500            # measured device MiB we allow one invocation to add
HARD_ABORT_MIB = 3000        # kill the invocation outright above this
BATCH_MIN = 64
KNEE_TOL = 0.05              # a 4x batch step buying less than this = knee reached
INVOCATION_TIMEOUT = 120     # seconds; a shape needing more is over-sized
REPEATS = 5                  # process-level repeats for final numbers

os.makedirs(OUT, exist_ok=True)
JSONL = open(os.path.join(OUT, "measurements.jsonl"), "a", buffering=1)
LOG = open(os.path.join(OUT, "driver.log"), "a", buffering=1)


def log(msg):
    LOG.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


# MEASURED: the CTA kernels' EIGENVECTOR path carries ~1.8 MB of workspace per matrix at
# n=32 -- ~450x the 4 KB of actual data -- and it scales linearly above batch 512:
#     batch 512 -> 445 MiB | 1024 -> 2237 MiB | 4096 -> 7505 MiB | 16384 -> 24083 MiB (card)
# The timing at 16384 also REGRESSES (0.592 vs 0.192 us/matrix at 4096), i.e. memory pressure
# rather than compute. Eigenvalues-only is unaffected (~343 MiB at batch 16384). So the
# eigenvector cells are capped by workspace footprint, and that cap is itself a finding.
BATCH_MAX_BY_JOBZ = {0: 16384, 1: 1024}   # jobz=1 ceiling keeps worst-case peak ~2.2 GB


def batch_cap(n, dtype, jobz=0):
    """Largest batch whose estimated footprint stays inside the budget.

    ~6 arrays of n*n (A, Z, workspace and friends) per batch item. Deliberately
    pessimistic: over-estimating the footprint costs a little timing precision,
    under-estimating it takes down the machine.
    """
    elem = 4 if dtype == "float" else 8
    # At large n the matrices dominate (~6 arrays of n*n per batch item) and this model is
    # accurate. At small n it is irrelevant, because the jobz ceiling binds first.
    cap = int(FOOTPRINT_BUDGET / (6.0 * n * n * elem))
    return max(BATCH_MIN, min(cap, BATCH_MAX_BY_JOBZ.get(jobz, BATCH_MAX)))


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", DEV],
                             capture_output=True, timeout=10)
        return int(out.stdout.decode().strip().splitlines()[0])
    except Exception:
        return 0


BASELINE_MIB = None


def run(binary, args, dtype, env_extra=None, min_iters=3, max_iters=5, want=None,
        watch_mem=False):
    """One invocation, one cell. Returns (us_per_matrix, avg_ms, name[, peak_mib]) or None.

    With watch_mem=True the peak device memory attributable to THIS run is sampled while it
    executes. Modelling the footprint from n*n was tried and was wrong by ~360x at n=32 (it
    predicted 402 MB, reality was 24 GB), because these kernels carry a large fixed workspace
    per batch item. Measure it, do not model it.
    """
    csv_path = f"/tmp/bo_{os.getpid()}.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = DEV
    env.pop("ONEAPI_DEVICE_SELECTOR", None)   # setting this hides the CPU device -> abort
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    cmd = [f"{BUILD}/benchmarks/{binary}", f"--type={dtype}", "--backend=CUDA",
           "--warmup=1", f"--min_iters={min_iters}", f"--max_iters={max_iters}",
           f"--csv={csv_path}"]
    if want:
        cmd.append(f"--name={want}")
    cmd += [str(a) for a in args]
    peak = 0
    try:
        if watch_mem:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            t0 = time.time()
            while proc.poll() is None:
                peak = max(peak, gpu_used_mib())
                if peak - (BASELINE_MIB or 0) > HARD_ABORT_MIB:
                    proc.kill()
                    log(f"ABORT (mem {peak} MiB) {binary} {args} {dtype}")
                    return None
                if time.time() - t0 > INVOCATION_TIMEOUT:
                    proc.kill()
                    log(f"TIMEOUT {binary} {args} {dtype}")
                    return None
                time.sleep(0.05)
            proc.wait()
            class _P:
                returncode = proc.returncode
                stderr = b""
            p = _P()
        else:
            p = subprocess.run(cmd, capture_output=True, timeout=INVOCATION_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {binary} {args} {dtype} {env_extra}")
        return None
    if p.returncode != 0 or not os.path.exists(csv_path):
        log(f"FAIL rc={p.returncode} {binary} {args} {dtype}: {p.stderr[-200:]!r}")
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    # Several binaries register MORE THAN ONE benchmark (syev_cta_fused_benchmark has
    # BM_SYEV_CTA_FUSED *and* BM_SYEV_CTA_PIPELINED; syev_jacobi_cta_benchmark has
    # BM_SYEV_JACOBI_CTA *and* BM_SYEV_CTA_TRIDIAG_REF). --name is a substring filter, so
    # SELECT the intended one from the CSV 'name' column rather than trusting the flag.
    if want:
        rows = [r for r in rows if want in r["name"]]
        if not rows:
            log(f"NO ROWS for want={want} {binary} {args}")
            return None
    names = {r["name"] for r in rows}
    if len(names) != 1:
        log(f"NAME CONTAMINATION {binary} {args} want={want}: {names}")
        return None
    r = rows[-1]
    tcol = [k for k in r if "matrix" in k][0]
    if watch_mem:
        return float(r[tcol]), float(r["avg_ms"]), r["name"], peak
    return float(r[tcol]), float(r["avg_ms"]), r["name"]


def knee_batch(binary, argfn, n, dtype, want=None, jobz=0, hard_max=None):
    """Climb in 4x steps until the step buys < KNEE_TOL, or we hit a cap. Return
    (batch, curve, reason). Never climbs past batch_cap."""
    cap = batch_cap(n, dtype, jobz)
    if hard_max:
        cap = min(cap, hard_max)
    curve, b, prev = [], BATCH_MIN, None
    while True:
        # Memory is NOT a gate here. The sampled peak is dominated by a transient JIT/context
        # spike (~2 GB) that does not scale with batch, so gating on it aborted climbs at
        # batch 64 -- an unsaturated, meaningless point. Safety is structural instead:
        # batch_cap() bounds large n by the n*n footprint, and BATCH_MAX_BY_JOBZ bounds the
        # eigenvector path, whose CTA workspace was MEASURED at ~1.8 MB/matrix at n=32.
        got = run(binary, argfn(b), dtype, want=want)
        if got is None:
            return prev_b, curve, "invocation failed"
        us, ms, _ = got
        curve.append({"batch": b, "us_per_matrix": us, "avg_ms": ms})
        if prev is not None:
            gain = (prev - us) / prev if prev > 0 else 0.0
            if gain < KNEE_TOL:
                return b, curve, f"knee: 4x step bought {gain*100:.1f}%"
        prev, prev_b = us, b
        if b >= cap:
            return b, curve, f"hit cap {cap} (jobz={jobz})"
        b = min(b * 4, cap)


def median_iqr(xs):
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return med, s[max(0, n // 4)], s[min(n - 1, (3 * n) // 4)]


def measure(tag, binary, args, dtype, batch, env_extra=None, meta=None, want=None):
    xs, peak = [], 0
    for i in range(REPEATS):
        # Watch memory on the FIRST repeat of every cell. The 24 GB incident came from a
        # contender measured at a batch whose knee had been established on a DIFFERENT
        # kernel -- so the knee scan being safe does not make the contender safe.
        got = run(binary, args, dtype, env_extra, min_iters=5, max_iters=10, want=want,
                  watch_mem=(i == 0))
        if got is None:
            break
        xs.append(got[0])
        if i == 0 and len(got) > 3:
            peak = got[3]   # recorded for provenance only -- NOT a gate, see knee_batch()
    if not xs:
        rec = {"tag": tag, "status": "failed", "binary": binary, "args": args,
               "type": dtype, "batch": batch, "env": env_extra or {}, **(meta or {})}
        JSONL.write(json.dumps(rec) + "\n")
        return None
    med, lo, hi = median_iqr(xs)
    rec = {"tag": tag, "status": "ok", "binary": binary, "args": args, "type": dtype,
           "batch": batch, "env": env_extra or {}, "median_us": med, "iqr": [lo, hi],
           "samples": xs, "peak_mib": peak, **(meta or {})}
    JSONL.write(json.dumps(rec) + "\n")
    return med


# --------------------------------------------------------------------------
# PHASE A -- small-n bake-off, n = 4..32. The user's explicit question:
# which kernel should Auto route to?
# --------------------------------------------------------------------------
SMALL_N = [4, 8, 16, 32]
CONTENDERS = [
    # (label, binary, argfn(n,batch,jobz,wg), exact benchmark name)
    ("cta",       "syev_cta_benchmark",        lambda n, b, j, w: [n, b, j, 0, w], "BM_SYEV_CTA<"),
    ("cta_fused", "syev_cta_fused_benchmark",  lambda n, b, j, w: [n, b, j, w],    "BM_SYEV_CTA_FUSED"),
    ("pipelined", "syev_cta_fused_benchmark",  lambda n, b, j, w: [n, b, j, w],    "BM_SYEV_CTA_PIPELINED"),
    ("jacobi",    "syev_jacobi_cta_benchmark", lambda n, b, j, w: [n, b, j, w],    "BM_SYEV_JACOBI_CTA"),
    ("tridiag_ref","syev_jacobi_cta_benchmark",lambda n, b, j, w: [n, b, j, w],    "BM_SYEV_CTA_TRIDIAG_REF"),
]
WGS = [1, 2, 4]


def phase_a():
    log("=== PHASE A: small-n bake-off ===")
    for dtype in ["float", "double"]:
        for n in SMALL_N:
            for jobz in [0, 1]:
                # Knee is a property of the shape; establish it once on cta at wg=1.
                b, curve, reason = knee_batch(
                    "syev_cta_benchmark",
                    lambda bb: [n, bb, jobz, 0, 1], n, dtype, want="BM_SYEV_CTA<", jobz=jobz)
                if b is None:
                    log(f"A skip n={n} {dtype} jobz={jobz}: knee failed")
                    continue
                JSONL.write(json.dumps({
                    "tag": "knee", "n": n, "type": dtype, "jobz": jobz,
                    "batch": b, "reason": reason, "curve": curve,
                    "cap": batch_cap(n, dtype, jobz)}) + "\n")
                log(f"A knee n={n} {dtype} jobz={jobz} -> batch={b} ({reason})")
                for label, binary, argfn, want in CONTENDERS:
                    for w in WGS:
                        measure(f"smalln:{label}", binary, argfn(n, b, jobz, w), dtype, b,
                                meta={"kernel": label, "n": n, "jobz": jobz, "wg": w},
                                want=want)
                # cuSOLVER reference. syev_benchmark hardcodes JobType::EigenVectors
                # (benchmarks/syev_benchmark.cc:53,59) and takes (n,batch,nb,fuse) --
                # so there is NO eigenvalues-only vendor route in the suite. Record the
                # gap rather than pretending jobz=0 was covered.
                if jobz == 1:
                    measure("smalln:vendor", "syev_benchmark", [n, b, 16, 0], dtype, b,
                            env_extra={"BATCHLAS_SYEV_PROVIDER": "vendor"},
                            meta={"kernel": "vendor", "n": n, "jobz": 1, "wg": 0},
                            want="BM_SYEV<")
                else:
                    JSONL.write(json.dumps({
                        "tag": "smalln:vendor", "status": "unavailable", "n": n,
                        "type": dtype, "jobz": 0,
                        "reason": "syev_benchmark hardcodes EigenVectors; no vendor "
                                  "eigenvalues-only route exists in the benchmark suite"}) + "\n")


# --------------------------------------------------------------------------
# PHASE B -- what IS the optimum block size per n, measured in the syev context?
# The committed values came from a standalone ormqr_blocked microbenchmark at
# batch 8192 with a search space capped at [4,8,12,16] for n<=64, so 16 was the
# ceiling of the search rather than an optimum. This is the measurement that was
# never taken.
# --------------------------------------------------------------------------
MID_N = [32, 64, 128, 256, 512]
ORMQR_VALUES = [8, 16, 32, 64, 128]
SYTRD_VALUES = [8, 16, 24, 32, 48, 64]


def phase_b():
    log("=== PHASE B: per-n block-size sweep (blocked provider) ===")
    for dtype in ["float", "double"]:
        for n in MID_N:
            for jobz in [0, 1]:
                # syev_blocked carries a large per-item workspace too. MEASURED: jobz=0 at
                # batch 16384 (n=32), 15258 (n=64), 3814 (n=128) and 953 (n=256) each
                # exceeded 3 GB and were aborted, losing those cells. Phase B therefore uses
                # its own, much lower ceiling. That costs a little saturation at small n --
                # reported as such -- and the alternative is an OOM.
                b, curve, reason = knee_batch(
                    "syev_blocked_benchmark",
                    lambda bb: [n, bb, jobz], n, dtype, want="BM_SYEV_BLOCKED", jobz=jobz,
                    hard_max=512)
                if b is None:
                    log(f"B skip n={n} {dtype} jobz={jobz}: knee failed")
                    continue
                JSONL.write(json.dumps({
                    "tag": "knee_blocked", "n": n, "type": dtype, "jobz": jobz,
                    "batch": b, "reason": reason, "curve": curve,
                    "cap": batch_cap(n, dtype, jobz)}) + "\n")
                log(f"B knee n={n} {dtype} jobz={jobz} -> batch={b} ({reason})")
                for v in ORMQR_VALUES:
                    measure("ormqr", "syev_blocked_benchmark", [n, b, jobz], dtype, b,
                            env_extra={"BATCHLAS_TUNE_ORMQR_BLOCK_SIZE": v},
                            meta={"knob": "ormqr", "value": v, "n": n, "jobz": jobz},
                            want="BM_SYEV_BLOCKED")
                for v in SYTRD_VALUES:
                    measure("sytrd", "syev_blocked_benchmark", [n, b, jobz], dtype, b,
                            env_extra={"BATCHLAS_TUNE_SYTRD_BLOCK_SIZE": v},
                            meta={"knob": "sytrd", "value": v, "n": n, "jobz": jobz},
                            want="BM_SYEV_BLOCKED")
                # cuSOLVER reference at the same shape (eigenvectors only -- see above)
                if jobz == 1:
                    measure("vendor_ref", "syev_benchmark", [n, b, 16, 0], dtype, b,
                            env_extra={"BATCHLAS_SYEV_PROVIDER": "vendor"},
                            meta={"knob": "vendor", "n": n, "jobz": 1}, want="BM_SYEV<")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    BASELINE_MIB = gpu_used_mib()
    globals()["BASELINE_MIB"] = BASELINE_MIB
    log(f"START {which} pid={os.getpid()} baseline={BASELINE_MIB}MiB "
        f"budget={BUDGET_MIB}MiB abort={HARD_ABORT_MIB}MiB batch<={BATCH_MAX}")
    if which in ("all", "a"):
        phase_a()
    if which in ("all", "b"):
        phase_b()
    log("DONE")
