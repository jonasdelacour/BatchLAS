"""The Tuning tab: the tuner's ledgers and tables, and runs of tools/tune/batchlas_tune.

Everything that reads a ledger goes through scripts/sweep_to_table.py, the Python port of the
ledger rules the tables are written with (best record per cell, staleness, nearest-row lookup),
so the page shows what the next table would hold and needs no GPU or build. Only planning and
measuring call the tuner. Spec: docs/design/tiered-tuning.md#benchviz-tuning-tab-interface-for-sub-project-3
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sweep_to_table as stt  # noqa: E402

DEFAULT_LEDGER = REPO / "benchmarks" / "results" / "tuning" / "ledger"
TUNED = REPO / "tuned"
TIERS = list(stt.LEDGER_TIERS)
DTYPES = ["float", "double", "cfloat", "cdouble"]
TIMED = set(stt.LEDGER_STATUS_TIMED)
TIER_NOTES = {
    "preview": "every 2nd lattice point, short races: a first look, not a table",
    "coarse": "the full lattice, refined to 1.1x; the default fill",
    "deep": "the full lattice, long races at 0.98 confidence; the shipped tables",
}


def _signature(d: Path) -> tuple:
    try:
        return tuple(sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(d)
                            if e.name.endswith(".jsonl")))
    except OSError:
        return ()


def _read(d: Path) -> tuple:
    """(runs, cells, audits) of one ledger directory. Unlike the table writer this never fails:
    a bad line is skipped, since a run may be appending to the last one right now."""
    runs, cells, audits = {}, [], {}
    for name in sorted(n for n in os.listdir(d) if n.endswith(".jsonl") and not n.endswith(".reps.jsonl")):
        with open(d / name) as f:
            for line in f:
                if not line.strip() or line.startswith(stt.LFS_POINTER):
                    continue
                try:
                    rec = json.loads(line)
                    kind = rec.get("kind")
                    if kind == "cell":
                        cells.append(stt.read_cell(rec))
                    elif kind == "run":
                        runs[rec["run_id"]] = rec
                    elif kind == "audit":
                        audits[rec["key"]] = rec.get("verdict", "")
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    return list(runs.values()), cells, audits


def key_text(key) -> str:
    return ",".join(f"{k}={v}" for k, v in key)


class LedgerView:
    """One <op>.<dtype>.<device> ledger with the current kernel hashes applied."""

    def __init__(self, root: Path, op: str, dtype: str, device: str):
        self.op, self.dtype, self.device = op, dtype, device
        self.spec = stt.OP_BY_NAME[op]
        self.dir = root / f"{op}.{dtype}.{device}"
        self.runs, self.cells, self.audits = _read(self.dir)
        newest = max(self.runs, key=lambda r: (r.get("date", ""), r["run_id"])) if self.runs else {}
        cands = [c for c in (newest.get("candidates") or "|".join(self.spec.candidate_order)).split("|") if c]
        self.cands = cands
        self.families = list(dict.fromkeys(stt.spelling_family(c) for c in cands))
        block = stt.parse_kernel_block(stt.read_spec_source(self.spec))
        self.family_hash = stt.family_hashes(stt.REPO, block, self.families)
        self.best = stt.best_records(self.cells, self.family_hash)
        self.newest = newest

    def freshness(self, cell) -> str:
        return stt.freshness(cell, self.family_hash)


class Ledgers:
    """LedgerView per directory, reloaded only when a file in it changes."""

    def __init__(self, root: Path = DEFAULT_LEDGER, tuned: Path = TUNED):
        self.root, self.tuned = Path(root), Path(tuned)
        self._views: dict = {}
        self._ops: dict = {}
        self._lock = threading.Lock()

    def idents(self) -> list:
        out = []
        if self.root.is_dir():
            for e in sorted(os.scandir(self.root), key=lambda e: e.name):
                parts = e.name.split(".")
                if e.is_dir() and len(parts) == 3 and parts[0] in stt.OP_BY_NAME:
                    out.append(tuple(parts))
        return out

    def view(self, op: str, dtype: str, device: str) -> LedgerView | None:
        if op not in stt.OP_BY_NAME:
            return None
        d = self.root / f"{op}.{dtype}.{device}"
        if not d.is_dir():
            return None
        sig = _signature(d)
        with self._lock:
            hit = self._views.get((op, dtype, device))
            if hit and hit[0] == sig:
                return hit[1]
        v = LedgerView(self.root, op, dtype, device)
        with self._lock:
            self._views[(op, dtype, device)] = (sig, v)
        return v

    def table_header(self, op: str, dtype: str, device: str) -> dict | None:
        p = self.tuned / f"{op}.{dtype}.{device}.txt"
        if not p.exists():
            return None
        header, rows = {}, 0
        with open(p) as f:
            for line in f:
                if line.startswith("#"):
                    for tok in line[1:].split():
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            header.setdefault(k, v)
                elif line.strip():
                    rows += 1
        src = header.get("source", "")
        return {"source": src.split(":", 1)[0] or "?", "tiers": header.get("tiers", ""), "date": header.get("date", ""),
                "rows": rows, "batchlas": header.get("batchlas", "")}

    def status(self) -> dict:
        """The op x dtype x device matrix: tier mix, coverage, staleness, audits, age, the shipped table."""
        rows, devices = [], set()
        now = time.time()
        for op, dtype, device in self.idents():
            devices.add(device)
            v = self.view(op, dtype, device)
            tiers = {t: 0 for t in TIERS}
            partial = 0
            for c in v.best.values():
                tiers[c["tier"]] += 1
                partial += v.freshness(c) == "partly_stale"
            keys = {c["key"] for c in v.cells}
            audits = {"ok": 0, "mismatch": 0, "inconclusive": 0}
            for verdict in v.audits.values():
                k = verdict.split(":", 1)[0]
                audits[k if k in audits else "mismatch"] += 1
            newest = max((r.get("date", "") for r in v.runs), default="")
            age = None
            if newest:
                try:
                    age = (now - time.mktime(time.strptime(newest[:10], "%Y-%m-%d"))) / 86400
                except ValueError:
                    pass
            rows.append({"op": op, "dtype": dtype, "device": device, "runs": len(v.runs), "cells": len(keys),
                         "best": len(v.best), "stale": len(keys) - len(v.best), "partial": partial, "tiers": tiers,
                         "refined": sum(1 for c in v.best.values() if c["round"] > 0), "audits": audits,
                         "newest": newest, "age_days": age, "table": self.table_header(op, dtype, device)})
        if self.tuned.is_dir():  # devices with tables but no ledger yet (sm_89's transcribed tables)
            for p in self.tuned.glob("*.txt"):
                parts = p.stem.split(".")
                if len(parts) == 3 and parts[0] in stt.OP_BY_NAME:
                    devices.add(parts[2])
                    if not any(r["op"] == parts[0] and r["dtype"] == parts[1] and r["device"] == parts[2] for r in rows):
                        rows.append({"op": parts[0], "dtype": parts[1], "device": parts[2], "runs": 0, "cells": 0,
                                     "best": 0, "stale": 0, "partial": 0, "tiers": {t: 0 for t in TIERS}, "refined": 0,
                                     "audits": {"ok": 0, "mismatch": 0, "inconclusive": 0}, "newest": "",
                                     "age_days": None, "table": self.table_header(*parts)})
        return {"ledger": str(self.root), "tuned": str(self.tuned), "devices": sorted(devices), "rows": rows}

    def op_view(self, op: str, dtype: str, device: str) -> dict:
        """_op_view, kept until the ledger directory or the shipped table changes."""
        table = self.tuned / f"{op}.{dtype}.{device}.txt"
        sig = (_signature(self.root / f"{op}.{dtype}.{device}"), table.stat().st_mtime_ns if table.exists() else 0)
        with self._lock:
            hit = self._ops.get((op, dtype, device))
        if hit and hit[0] == sig:
            return hit[1]
        val = self._op_view(op, dtype, device)
        with self._lock:
            self._ops[(op, dtype, device)] = (sig, val)
        return val

    def _op_view(self, op: str, dtype: str, device: str) -> dict:
        """Every best record of one ledger, compact, plus the shipped table's pick at it.

        cells[i] = [key values in '# keys:' order, tier, round, winner, runner-up margin,
        freshness, audit, shipped pick, predicted speedup]. The pick is the first entry of the
        nearest shipped row that the cell's record did not refuse (status skipped is can_run
        false); the speedup is that candidate's median over the new pick's, both at this cell.
        """
        v = self.view(op, dtype, device)
        spec = stt.OP_BY_NAME.get(op)
        if spec is None:
            return {"error": f"unknown op {op}"}
        keyspec = stt.parse_keys(spec.keys)
        names = [n for n, _, _ in keyspec]
        out = {"op": op, "dtype": dtype, "device": device, "tiers": TIERS,
               "keys": [{"name": n, "log": is_log, "weight": w} for n, is_log, w in keyspec],
               "cands": v.cands if v else list(spec.candidate_order), "families": v.families if v else [],
               "cells": [], "table": self.table_header(op, dtype, device)}
        if v is None:
            return out
        cand_ix = {c: i for i, c in enumerate(v.cands)}
        cells = list(v.best.values())
        tkeys = []
        for c in cells:
            try:
                tkeys.append(stt.key_tuple(spec, keyspec, c["key"]))
            except SystemExit:
                tkeys.append(None)
        shipped = self._shipped_rows(op, dtype, device)
        if shipped and [n for n, _, _ in shipped[0]] != names:
            out["table_error"] = f"the shipped table's keys {[n for n, _, _ in shipped[0]]} are not {names}"
            shipped = None
        new_rows = self._new_rows(v, spec, keyspec)
        old_pick = Nearest(shipped[1], shipped[0]).lookup(tkeys) if shipped else [None] * len(cells)
        new_pick = Nearest([(k, e) for k, (_, e) in new_rows.items()], keyspec).lookup(tkeys)
        for c, tk, orow, nrow in zip(cells, tkeys, old_pick, new_pick):
            if tk is None:
                continue
            med = {r["cand"]: float(r["median"]) for r in c["cands"]
                   if r["status"] in TIMED and r["median"] and r["median"] > 0}
            ranked = [r for r in c["ranked"] if r in med]
            margin = med[ranked[1]] / med[ranked[0]] - 1 if len(ranked) > 1 else None
            win = c["ranked"][0] if c["ranked"] else None
            refused = {r["cand"] for r in c["cands"] if r["status"] == "skipped"}
            old = next((s for s, _ in orow if s not in refused), None) if orow else None
            new = next((s for s, _ in nrow if s not in refused), None) if nrow else None
            sp = med[old] / med[new] if old in med and new in med else None
            out["cells"].append([list(tk), TIERS.index(c["tier"]), c["round"], cand_ix.get(win, -1),
                                 None if margin is None else round(margin, 4), v.freshness(c) == "partly_stale",
                                 v.audits.get(key_text(c["key"]), ""), cand_ix.get(old, -1) if old else -1,
                                 None if sp is None else round(sp, 4), cand_ix.get(new, -1) if new else -1])
        out["names"] = names
        return out

    def _shipped_rows(self, op, dtype, device):
        p = self.tuned / f"{op}.{dtype}.{device}.txt"
        if not p.exists():
            return None
        _, keyspec, rows = stt.parse_table(p.read_text())
        return keyspec, rows

    def _new_rows(self, v: LedgerView, spec, keyspec) -> dict:
        """The rows `sweep_to_table.py --ledger` would write now: {key: (tier, [(spelling, ms)])}."""
        led = stt.Ledger(runs=v.runs, cells=v.cells)
        old = stt.read_old_transcribed(str(self.tuned / f"{v.op}.{v.dtype}.{v.device}.txt"))
        return stt.ledger_rows(spec, keyspec, led, v.family_hash, old)

    def cell(self, op: str, dtype: str, device: str, key: list) -> dict:
        """Every record of one cell (all tiers and runs), every candidate with its interval."""
        v = self.view(op, dtype, device)
        if v is None:
            return {"records": []}
        spec = stt.OP_BY_NAME[op]
        keyspec = stt.parse_keys(spec.keys)
        want = tuple(key)
        recs = []
        for c in v.cells:
            try:
                if stt.key_tuple(spec, keyspec, c["key"]) != want:
                    continue
            except SystemExit:
                continue
            best = v.best.get(c["key"]) is c
            recs.append({"run_id": c["run_id"], "tier": c["tier"], "round": c["round"], "date": c["date"],
                         "best": best, "fresh": v.freshness(c), "ranked": c["ranked"], "key": key_text(c["key"]),
                         "audit": v.audits.get(key_text(c["key"]), ""),
                         "cands": [{k: r.get(k) for k in ("cand", "status", "reason", "median", "lo", "hi", "reps")}
                                   | {"current": r["status"] == "skipped" or v.family_hash.get(stt.spelling_family(r["cand"])) == r["hash"]}
                                   for r in c["cands"]]})
        recs.sort(key=lambda r: (not r["best"], TIERS.index(r["tier"]), r["date"]), reverse=False)
        return {"records": recs}


class Nearest:
    """sweep_to_table.nearest() for many queries at once: per exact-key prefix, a numpy distance
    over the log keys with the same 1e-9 tie window and the smaller-keys tie break."""

    def __init__(self, rows, keyspec):
        self.keyspec = keyspec
        self.exact = [i for i, (_, is_log, _) in enumerate(keyspec) if not is_log]
        self.logs = [i for i, (_, is_log, _) in enumerate(keyspec) if is_log]
        self.w = np.array([keyspec[i][2] for i in self.logs], dtype=float)
        self.rows = list(rows or [])
        self.pools: dict = {}
        for p in range(len(self.exact), -1, -1):
            for ri, (k, _) in enumerate(self.rows):
                self.pools.setdefault((p, tuple(k[i] for i in self.exact[:p])), []).append(ri)
        self.arr = {pk: (np.array(ix), np.log2(np.array([[self.rows[r][0][i] for i in self.logs] for r in ix], dtype=float))
                         .reshape(len(ix), len(self.logs)))
                    for pk, ix in self.pools.items()}

    def lookup(self, keys) -> list:
        out = [None] * len(keys)
        if not self.rows:
            return out
        by_pool: dict = {}
        for qi, k in enumerate(keys):
            if k is None:
                continue
            for p in range(len(self.exact), -1, -1):
                pk = (p, tuple(k[i] for i in self.exact[:p]))
                if pk in self.arr:
                    by_pool.setdefault(pk, []).append(qi)
                    break
        for pk, qs in by_pool.items():
            ix, lr = self.arr[pk]
            q = np.log2(np.maximum(1, np.array([[keys[qi][i] for i in self.logs] for qi in qs], dtype=float)))
            q = q.reshape(len(qs), len(self.logs))
            d = (np.abs(lr[None, :, :] - q[:, None, :]) * self.w).sum(axis=2) if self.logs else np.zeros((len(qs), len(ix)))
            m = d.min(axis=1)
            for j, qi in enumerate(qs):
                tied = ix[np.nonzero(d[j] <= m[j] + 1e-9)[0]]
                r = min(tied, key=lambda r: (tuple(self.rows[r][0][i] for i in self.logs), r))
                out[qi] = self.rows[r][1]
        return out


# ---------------------------------------------------------------- runs

def tuner_binary(build_dirs) -> Path | None:
    for d in build_dirs or []:
        p = Path(d) / "tools" / "tune" / "batchlas_tune"
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def tuner_ops() -> list:
    return [{"name": s.op, "keys": s.keys, "cands": list(s.candidate_order)} for s in stt.OPS]


def run_argv(binary: Path, req: dict, ledger: Path) -> list:
    """The batchlas_tune command line for a launch request; raises ValueError on a bad one."""
    ops = [o for o in req.get("ops", []) if o in stt.OP_BY_NAME]
    dtypes = [t for t in req.get("dtypes", []) if t in DTYPES]
    tier = req.get("tier", "")
    devices = sorted({int(g) for g in req.get("devices", [])})
    if not ops or not dtypes:
        raise ValueError("choose at least one op and one precision")
    if tier not in TIER_NOTES:
        raise ValueError(f"tier must be one of {', '.join(TIER_NOTES)}")
    if not devices:
        raise ValueError("choose at least one GPU")
    argv = [str(binary), ",".join(ops), "--tier", tier, "--dtype", ",".join(dtypes),
            "--devices", ",".join(map(str, devices)), "--repo", str(REPO), "--ledger", str(ledger)]
    if req.get("max_dim"):
        argv += ["--max-dim", str(int(req["max_dim"]))]
    if req.get("budget_h"):
        argv += ["--budget", str(float(req["budget_h"]))]
    if req.get("allow_idle_foreign", True):
        argv += ["--allow-idle-foreign"]
    if req.get("guard_wait_s"):
        argv += ["--guard-wait", str(int(req["guard_wait_s"]))]
    if req.get("out"):
        argv += ["--out", str(req["out"])]
    return argv


def plan(binary: Path, req: dict, ledger: Path, timeout: float = 300) -> dict:
    """`batchlas_tune --plan`: the per-(op, dtype) plan and the estimate, no GPU work."""
    argv = run_argv(binary, req, ledger) + ["--plan"]
    r, w = os.pipe()
    try:
        p = subprocess.Popen(argv + ["--progress-fd", str(w)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             pass_fds=(w,), text=True, cwd=str(REPO))
    finally:
        os.close(w)
    with os.fdopen(r) as f:
        lines = f.read().splitlines()
    try:
        err = p.communicate(timeout=timeout)[1]
    except subprocess.TimeoutExpired:
        p.kill()
        return {"error": "--plan timed out"}
    evs = [json.loads(ln) for ln in lines if ln.strip().startswith("{")]
    total = next((e for e in evs if e.get("ev") == "plan"), None)
    if p.returncode or total is None:
        return {"error": (err or "").strip().splitlines()[-1:] or [f"--plan exited {p.returncode}"], "argv": argv}
    return {"jobs": [e for e in evs if e.get("ev") == "plan_job"], "total": total, "argv": argv}


def runs_dir(root: Path) -> Path:
    return Path(root) / "_tuning"


def launch(root: Path, binary: Path, req: dict, ledger: Path, name: str | None = None) -> str:
    """Start a detached run; its events, log and exit code land in <root>/_tuning/<name>/."""
    argv = run_argv(binary, req, ledger)
    name = "".join(ch for ch in (name or "") if ch.isalnum() or ch in "-_.") or \
        time.strftime(f"{req['tier']}-%Y%m%d-%H%M%S")
    d = runs_dir(root) / name
    if (d / "run.json").exists():
        raise ValueError(f"a tuning run named {name} exists; resuming starts a new run, which skips what it holds")
    d.mkdir(parents=True)
    # sh keeps the exit status after the server is gone; fd 3 is the progress stream.
    cmd = ["sh", "-c", 'd="$0"; exec 3>>"$d/events.jsonl"; "$@" --progress-fd 3 >>"$d/run.log" 2>&1; echo $? >"$d/exit"',
           str(d), *argv]
    p = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    _children[p.pid] = p
    (d / "run.json").write_text(json.dumps({"name": name, "argv": argv, "request": req, "pid": p.pid,
                                            "started": time.time(), "ledger": str(ledger)}, indent=1))
    return name


def list_runs(root: Path) -> list:
    d = runs_dir(root)
    if not d.is_dir():
        return []
    rs = [x for x in d.iterdir() if (x / "run.json").exists()]
    return [x.name for x in sorted(rs, key=lambda x: x.stat().st_mtime, reverse=True)]


_children: dict = {}  # pid -> Popen of runs this process started; polling reaps them


def _alive(pid: int) -> bool:
    if pid in _children:  # our own child: a dead one is a zombie, which os.kill still finds
        return _children[pid].poll() is None
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def stop(root: Path, name: str) -> bool:
    d = runs_dir(root) / name
    try:
        meta = json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        return False
    (d / "stop").write_text(str(time.time()))
    try:  # the run's own session: the shell, the driver and its per-GPU workers
        os.killpg(int(meta["pid"]), signal.SIGTERM)
    except (OSError, KeyError, ValueError):
        return False
    return True


class RunState:
    """A run's progress, folded incrementally from its events.jsonl."""

    FEED = 40

    def __init__(self, d: Path):
        self.d, self.off, self.partial = d, 0, b""
        self.plan, self.jobs, self.gpus = None, {}, {}
        self.feed, self.done_t = [], []
        self.elim: dict = {}
        self.ended = False

    def _job(self, e):
        k = f"{e.get('op')}.{e.get('dtype')}"
        return self.jobs.setdefault(k, {"op": e.get("op"), "dtype": e.get("dtype"), "planned": 0, "measure": 0,
                                        "refine_est": 0, "est_s": 0.0, "done": 0, "refined": 0, "errors": 0,
                                        "eliminated": 0, "audit_ok": 0, "audit_bad": 0, "restarts": 0,
                                        "ld_failed": 0, "capped": False, "mix": ""})

    def _keytext(self, e) -> str:
        skip = {"ev", "op", "dtype", "gpu", "ranked", "tier", "round", "error", "cand", "verdict", "fresh_ms",
                "warm_ms", "fresh", "restarts", "fallback", "reason", "t", "device"}
        return " ".join(f"{k}={v}" for k, v in e.items() if k not in skip)

    def fold(self, e: dict):
        ev, t = e.get("ev"), e.get("t")
        if ev == "plan":
            self.plan = e
        elif ev == "plan_job":
            j = self._job(e)
            j.update(planned=e.get("cells", 0), measure=e.get("measure", 0), refine_est=e.get("refine_cells", 0),
                     est_s=e.get("est_s", 0) + e.get("est_refine_s", 0), mix=e.get("mix", ""))
        elif ev == "cell_start":
            self.gpus[e.get("gpu")] = {"op": e.get("op"), "dtype": e.get("dtype"), "key": self._keytext(e), "t": t}
        elif ev == "cell_done":
            j, kt = self._job(e), self._keytext(e)
            j["done"] += 1
            j["refined"] += (e.get("round") or 0) > 0
            j["errors"] += bool(e.get("error"))
            for g, cur in list(self.gpus.items()):
                if cur["key"] == kt and cur["op"] == e.get("op") and cur["dtype"] == e.get("dtype"):
                    del self.gpus[g]
            n_el = self.elim.pop((e.get("op"), e.get("dtype"), kt), 0)
            ranked = [r for r in (e.get("ranked") or "").split("|") if r]
            self._push({"t": t, "kind": "error" if e.get("error") else "cell", "op": e.get("op"),
                        "dtype": e.get("dtype"), "key": kt, "win": ranked[0] if ranked else "",
                        "second": ranked[1] if len(ranked) > 1 else "", "eliminated": n_el, "round": e.get("round"),
                        "text": e.get("error", "")})
            if t:
                self.done_t.append(t)
        elif ev == "eliminated":
            self._job(e)["eliminated"] += 1
            k = (e.get("op"), e.get("dtype"), self._keytext(e))
            self.elim[k] = self.elim.get(k, 0) + 1
        elif ev == "audit":
            j = self._job(e)
            ok = e.get("verdict") in ("ok", "inconclusive")
            j["audit_ok" if ok else "audit_bad"] += 1
            if not ok:
                self._push({"t": t, "kind": "audit", "op": e.get("op"), "dtype": e.get("dtype"),
                            "key": self._keytext(e), "text": e.get("verdict", "")})
        elif ev == "worker_restart":
            self._job(e)["restarts"] += 1
        elif ev == "ld_audit":
            self._job(e)["ld_failed"] += 1
            self._push({"t": t, "kind": "ld_audit", "op": e.get("op"), "dtype": e.get("dtype"), "key": self._keytext(e),
                        "text": f"{e.get('cand')}: {e.get('reason', '')}"})
        elif ev == "refine_cap":
            self._job(e)["capped"] = True
            self._push({"t": t, "kind": "cap", "op": e.get("op"), "dtype": e.get("dtype"), "key": "",
                        "text": f"refinement cap: {e.get('dropped')} cells dropped"})
        elif ev == "done":
            self.ended = True

    def _push(self, item):
        self.feed.append(item)
        del self.feed[:-self.FEED]

    def update(self):
        p = self.d / "events.jsonl"
        try:
            with open(p, "rb") as f:
                f.seek(self.off)
                data = f.read()
        except OSError:
            return
        self.off += len(data)
        data = self.partial + data
        lines = data.split(b"\n")
        self.partial = lines.pop()
        for ln in lines:
            try:
                self.fold(json.loads(ln))
            except ValueError:
                continue

    def snapshot(self) -> dict:
        self.update()
        try:
            meta = json.loads((self.d / "run.json").read_text())
        except (OSError, ValueError):
            meta = {}
        code = None
        if (self.d / "exit").exists():
            try:
                code = int((self.d / "exit").read_text().strip() or "0")
            except ValueError:
                code = -1
        if code is None:  # a stop kills the wrapping shell too, before it can write the exit code
            state = "running" if _alive(meta.get("pid")) else "stopped" if (self.d / "stop").exists() else "interrupted"
        elif (self.d / "stop").exists():
            state = "stopped"
        else:
            state = "finished" if code == 0 else "failed"
        done = sum(j["done"] for j in self.jobs.values())
        plan_cells = (self.plan or {}).get("cells", 0)
        refine = (self.plan or {}).get("refine_cells", 0)
        total = max(done, plan_cells + refine)
        now = time.time()
        recent = [t for t in self.done_t[-400:] if now - t < 900] if state == "running" else []
        rate = 60.0 * (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 5 and recent[-1] > recent[0] else None
        eta = (total - done) / (rate / 60.0) if rate and total > done else None
        tail = []
        log = self.d / "run.log"
        if log.exists():
            with open(log, "rb") as fh:
                fh.seek(max(0, log.stat().st_size - 8000))
                tail = fh.read().decode(errors="replace").splitlines()[-60:]
        return {"name": self.d.name, "state": state, "exit": code, "request": meta.get("request", {}),
                "argv": meta.get("argv", []), "started": meta.get("started"), "plan": self.plan,
                "jobs": sorted(self.jobs.values(), key=lambda j: (j["op"], DTYPES.index(j["dtype"]) if j["dtype"] in DTYPES else 9)),
                "gpus": [{"gpu": g, **c} for g, c in sorted(self.gpus.items(), key=lambda x: (x[0] is None, x[0]))] if state == "running" else [],
                "done": done, "total": total, "rate_per_min": rate, "eta_s": eta,
                "est_total_s": (self.plan or {}).get("est_total_s"), "feed": self.feed[::-1], "log": tail}


_states: dict = {}
_states_lock = threading.Lock()


def run_state(root: Path, name: str) -> dict:
    d = runs_dir(root) / name
    if not (d / "run.json").exists():
        return {"name": name, "missing": True}
    with _states_lock:
        st = _states.get(d)
        if st is None:
            st = _states[d] = RunState(d)
        return st.snapshot()


def live_run(root: Path) -> str | None:
    """The newest run whose process is still alive, if any: one tuning run at a time per box."""
    for name in list_runs(root):
        d = runs_dir(root) / name
        try:
            meta = json.loads((d / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if not (d / "exit").exists() and _alive(meta.get("pid")):
            return name
    return None
