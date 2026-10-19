"""The Tuning tab: the tuner's ledgers and tables, and runs of tools/tune/batchlas_tune.

Everything that reads a ledger or a table goes through scripts/sweep_to_table.py, the code the
tables are written with (read_ledger with its rehash and dirty-run rules, best record, freshness,
ledger_rows, parse_table, the kernel hashes), so the page shows what the next table would hold and
needs no GPU or build. Only planning and measuring call the tuner. Files:
docs/design/tuning-data.md; the tab: docs/design/tiered-tuning.md#benchviz-tuning-tab-interface-for-sub-project-3
"""
from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sweep_to_table as stt  # noqa: E402

DEFAULT_LEDGER = REPO / stt.LEDGER_ROOT
TUNED = REPO / stt.TUNED
TIERS = list(stt.LEDGER_TIERS)
DTYPES = list(stt.DTYPES)
TIMED = set(stt.LEDGER_STATUS_TIMED)
TIER_NOTES = {
    "preview": "every 2nd lattice point plus flip-frontier rounds, short races: a first look, not a table",
    "calibrate": "every 4th lattice point raced against the nearest shipped table (prior), measured only where "
                 "the device disagrees; the prior fills the rest: a new GPU's first table",
    "coarse": "every 2nd lattice point plus flip-frontier rounds (gemm, geqrf, getri, orgqr, spmm: the full "
              "lattice, refined to 1.1x), 0.90 confidence; the default fill",
    "deep": "every 2nd lattice point plus flip-frontier rounds (the five stride-1 ops: the full lattice), long races "
            "at 0.98 confidence; the shipped tables",
    "custom": "exactly the cells listed below, every candidate, no lattice: a rehash sample",
}
# Flags this tab passes; a tuner whose --help lacks one predates the current tuner.
TUNER_FLAGS = ("--tier", "--plan", "--progress-fd", "--ledger", "--device-key", "--remeasure-keys", "--allow-dirty")


def _stat(p: Path):
    try:
        st = os.stat(p)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def key_text(key) -> str:
    return ",".join(f"{k}={v}" for k, v in key)


def spec_ops(repo: Path = REPO) -> list:
    """The ops the tuner has a spec for (tools/tune/<op>_spec.cc)."""
    d = Path(repo) / "tools" / "tune"
    return sorted(p.name[:-len("_spec.cc")] for p in d.glob("*_spec.cc")) if d.is_dir() else []


_GEN = itertools.count(1)  # cache generations: a reloaded file, table or view gets a new one
_hashes: dict = {}
HASH_TTL_S = 2.0


def current_hashes(op: str, families) -> dict:
    """Hash v2 of each family's kernel sources in this checkout, {} when the spec is unreadable.
    Re-hashed after HASH_TTL_S, so a kernel edit shows up while the server runs."""
    k = (op, tuple(families))
    hit = _hashes.get(k)
    if hit and time.time() - hit[0] < HASH_TTL_S:
        return hit[1]
    try:
        val = stt.family_hashes(stt.parse_kernel_block(stt.read_spec_source(op)), list(families))
    except SystemExit:
        val = {}
    _hashes[k] = (time.time(), val)
    return val


def _audits(path: Path) -> dict:
    """{(dtype, key text): verdict} of the audit lines; read_ledger has no use for them."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if '"audit"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("kind") == "audit":
                        out[(rec.get("dtype"), rec["key"])] = rec.get("verdict", "")
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return out


class LedgerFile:
    """One <op>.<device>.jsonl as stt.read_ledger reads it: rehash lines applied, dirty runs left out."""

    def __init__(self, path: Path, op: str, device: str):
        self.path, self.op, self.device, self.error, self.gen = Path(path), op, device, None, next(_GEN)
        try:
            led = stt.read_ledger(str(path))
        except SystemExit as e:  # an LFS pointer, or a malformed line before the last
            self.error, led = str(e), stt.Ledger()
        self.runs, self.warnings = led.runs, [w.replace(str(path) + ": ", "") for w in led.warnings]
        self.cells = {d: [c for c in led.cells if c["dtype"] == d] for d in DTYPES}
        self.heads = {d: stt.newest(rs) for d in DTYPES if (rs := [r for r in led.runs if r["dtype"] == d])}
        self.audits = _audits(path) if not self.error else {}
        self.families = list(dict.fromkeys(stt.spelling_family(c) for d in self.heads for c in self.cands(d)))

    def cands(self, dtype: str) -> list:
        return [c for c in self.heads.get(dtype, {}).get("candidates", "").split("|") if c]

    def keys(self, dtype: str) -> str:
        return self.heads.get(dtype, {}).get("keys", "")


class TableFile:
    """One tuned/<op>.<device>.txt through stt.parse_table, split per dtype like select::parse_tables."""

    def __init__(self, path: Path):
        text, self.gen = path.read_text(), next(_GEN)
        self.header, keyspec, rows = stt.parse_table(text)
        self.ok = bool(keyspec) and keyspec[0][:2] == ("dtype", False)
        self.keyspec = keyspec[1:] if self.ok else keyspec
        self.rows = {d: [(k[1:], e) for k, e in rows if k[0] == d] for d in DTYPES} if self.ok else {}
        self.tiers = {d: {t: 0 for t in TIERS} for d in DTYPES}
        for line in text.splitlines():  # the per-row '# tier' tag, which parse_table drops
            if line.startswith("dtype=") and "#" in line:
                d, t = line.split()[0][len("dtype="):], line.rsplit("#", 1)[1].strip()
                if d in self.tiers and t in self.tiers[d]:
                    self.tiers[d][t] += 1
        self.judge = stt.header_hashes(self.header)

    def info(self, dtype: str) -> dict:
        src = self.header.get("source", "")
        return {"source": src.split(":", 1)[0] or "?", "ledger": src.split(":", 1)[1] if ":" in src else "",
                "date": self.header.get("date", ""), "batchlas": self.header.get("batchlas", ""),
                "kernels": self.header.get("kernels", ""), "rows": len(self.rows.get(dtype, [])),
                "tiers": self.tiers.get(dtype, {})}


class LedgerView:
    """One dtype of a ledger file, judged by the kernel sources in this checkout."""

    def __init__(self, f: LedgerFile, dtype: str, now: dict):
        self.op, self.dtype, self.device, self.file, self.gen = f.op, dtype, f.device, f, next(_GEN)
        self.cells, self.cands = f.cells.get(dtype, []), f.cands(dtype)
        self.keyspec = stt.parse_keys(f.keys(dtype)) if f.keys(dtype) else []
        self.families = list(dict.fromkeys(stt.spelling_family(c) for c in self.cands))
        # Records and rows are judged as ledger_text judges them (every family of the file);
        # partly stale only by this dtype's: stale_families counts a judged family a record lacks
        # (gemm's float-only reg would mark every double cell partly stale).
        self.judge = now
        self.family_hash = {k: now[k] for k in self.families if k in now}
        self.best = stt.best_records(self.cells, self.judge)
        self.latest = {}
        for c in self.cells:
            self.latest[c["key"]] = c
        self.runs = [r for r in f.runs if r["dtype"] == dtype]
        self.error, self._stale = None, None
        try:
            self.rows = stt.ledger_rows(self.keyspec, self.cells, self.judge) if self.keyspec else {}
        except SystemExit as e:  # a varied hidden axis: sweep_to_table refuses this dtype
            self.rows, self.error = {}, str(e)

    def freshness(self, cell) -> str:
        return stt.freshness(cell, self.family_hash)

    def stale_by_family(self) -> dict:
        """Per family, the keys the next run re-races for it: stale families of each key's best
        record, or of its latest record where no record is usable."""
        if self._stale is None:
            self._stale = {f: 0 for f in self.families}
            for key, c in self.latest.items():
                for fam in stt.stale_families(self.best.get(key, c), self.family_hash):
                    self._stale[fam] = self._stale.get(fam, 0) + 1
        return self._stale

    def audit(self, cell) -> str:
        return self.file.audits.get((self.dtype, key_text(cell["key"])), "")


def table_sync(view: LedgerView, table: TableFile) -> int | None:
    """Rows of this dtype where the shipped table differs from the ledger, judged with the header's
    hashes as `sweep_to_table.py --check` judges (times compared at fmt_ms precision)."""
    if view.error or not table.ok or not view.keyspec or [n for n, _, _ in table.keyspec] != [n for n, _, _ in view.keyspec]:
        return None
    try:
        new = view.rows if table.judge == view.judge else stt.ledger_rows(view.keyspec, view.cells, table.judge)
    except SystemExit:
        return None
    norm = {k: [(c, None if ms is None else float(stt.fmt_ms(ms))) for c, ms in e] for k, (_, e) in new.items()}
    shipped = dict(table.rows.get(view.dtype, []))
    return sum(1 for k in set(norm) | set(shipped) if norm.get(k) != shipped.get(k))


class Ledgers:
    """Ledger files, tables and per-dtype views, each reloaded only when its file (or, for a view,
    the kernel sources' hashes) changes."""

    def __init__(self, root: Path = DEFAULT_LEDGER, tuned: Path = TUNED):
        self.root, self.tuned = Path(root), Path(tuned)
        self._files: dict = {}
        self._tables: dict = {}
        self._views: dict = {}
        self._syncs: dict = {}
        self._ops: dict = {}
        self._lock = threading.Lock()
        self._load = threading.RLock()

    def _cached(self, store: dict, key, sig, make):
        with self._lock:
            hit = store.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        with self._load:  # one loader at a time: the warm-up thread and the first page share a file's load
            hit = store.get(key)
            if hit and hit[0] == sig:
                return hit[1]
            val = make()
            with self._lock:
                store[key] = (sig, val)
        return val

    def idents(self) -> list:
        """(op, device) of every ledger file and every table."""
        out = set()
        for d, suffix in ((self.root, ".jsonl"), (self.tuned, ".txt")):
            if d.is_dir():
                out |= {i for i in (stt.split_name(n, suffix) for n in os.listdir(d)) if i}
        return sorted(out)

    def file(self, op: str, device: str) -> LedgerFile | None:
        p = self.root / f"{op}.{device}.jsonl"
        sig = _stat(p)
        return None if sig is None else self._cached(self._files, (op, device), sig, lambda: LedgerFile(p, op, device))

    def table(self, op: str, device: str) -> TableFile | None:
        p = self.tuned / f"{op}.{device}.txt"
        sig = _stat(p)
        if sig is None:
            return None

        def make():
            try:
                return TableFile(p)
            except (OSError, ValueError, KeyError):
                return None
        return self._cached(self._tables, (op, device), sig, make)

    def view(self, op: str, dtype: str, device: str) -> LedgerView | None:
        f = self.file(op, device)
        if f is None or dtype not in f.heads:
            return None
        now = current_hashes(op, f.families)
        sig = (f.gen, tuple(sorted(now.items())))
        return self._cached(self._views, (op, dtype, device), sig, lambda: LedgerView(f, dtype, now))

    def sync(self, v: LedgerView | None, t: TableFile | None) -> int | None:
        if v is None or t is None:
            return None
        return self._cached(self._syncs, (v.op, v.dtype, v.device), (v.gen, t.gen), lambda: table_sync(v, t))

    def ops(self, load: bool = True) -> list:
        """Every op with a tuner spec or a ledger: its keys and candidates, from the newest run lines.
        load=False reads no ledger: keys from a table's '# keys:' line, candidates only once loaded."""
        idents = self.idents()
        out = []
        for op in sorted(set(spec_ops()) | {o for o, _ in idents}):
            devs = [dev for o, dev in idents if o == op]
            files = [f for dev in devs if (f := self.file(op, dev) if load else self._loaded(op, dev))]
            heads = [h for f in files for h in f.heads.values()]
            keys = stt.newest(heads).get("keys", "") if heads else next(filter(None, (self._table_keys(op, d) for d in devs)), "")
            cands = list(dict.fromkeys(c for h in heads for c in h.get("candidates", "").split("|") if c))
            out.append({"name": op, "keys": keys, "cands": cands})
        return out

    def _loaded(self, op: str, device: str) -> LedgerFile | None:
        with self._lock:
            hit = self._files.get((op, device))
        return hit[1] if hit and hit[0] == _stat(self.root / f"{op}.{device}.jsonl") else None

    def _table_keys(self, op: str, device: str) -> str:
        try:
            with open(self.tuned / f"{op}.{device}.txt") as f:
                line = next((ln for ln in itertools.islice(f, 8) if ln.startswith("# keys:")), "")
        except OSError:
            return ""
        return " ".join(t for t in line[len("# keys:"):].split() if not t.startswith("dtype:"))

    def families(self, op: str, device: str, views: list) -> list:
        """Per family: the table header's hash, the hash of the sources now, keys stale in it per dtype."""
        f, t = self.file(op, device), self.table(op, device)
        names = list(dict.fromkeys((f.families if f else []) + (list(t.judge) if t else [])))
        now = current_hashes(op, names)
        stale = {v.dtype: v.stale_by_family() for v in views}
        return [{"name": n, "table": t.judge.get(n) if t else None, "now": now.get(n),
                 "stale": {d: s.get(n, 0) for d, s in stale.items() if s.get(n)}} for n in names]

    def status(self) -> dict:
        """The op x dtype x device matrix, plus one entry per (op, device) for the table and its families."""
        rows, files, devices = [], {}, set()
        now = time.time()
        for op, device in self.idents():
            devices.add(device)
            f, t = self.file(op, device), self.table(op, device)
            views = [v for d in DTYPES if (v := self.view(op, d, device))]
            entry = {"op": op, "device": device, "ledger": bool(f), "error": f.error if f else None,
                     "warnings": f.warnings if f else [], "table": None, "table_warnings": [],
                     "families": self.families(op, device, views)}
            if t:
                entry["table"] = {k: v for k, v in t.info("").items() if k not in ("rows", "tiers")} | {"ok": t.ok}
                entry["table_warnings"] = stt.stale_kernel_warnings(str(self.tuned / f"{op}.{device}.txt"), str(REPO))
            files[f"{op}.{device}"] = entry
            dts = [d for d in DTYPES if (f and d in f.heads) or (t and t.rows.get(d))]
            for d in dts:
                rows.append(self._status_row(op, d, device, f, t, self.view(op, d, device), now))
        return {"ledger": str(self.root), "tuned": str(self.tuned), "devices": sorted(devices), "rows": rows,
                "files": files, "tiers": TIERS, "ops": self.ops()}

    def _status_row(self, op, dtype, device, f, t, v, now) -> dict:
        tiers = {x: 0 for x in TIERS}
        audits = {"ok": 0, "mismatch": 0, "inconclusive": 0}
        row = {"op": op, "dtype": dtype, "device": device, "runs": 0, "cells": 0, "best": 0, "stale": 0, "partial": 0,
               "tiers": tiers, "rows": {x: 0 for x in TIERS}, "refined": 0, "audits": audits, "newest": "",
               "age_days": None, "error": None, "table": None}
        if t:
            # The exact row diff (sync) costs a ledger_rows per dtype: the op view computes it. Here
            # a table is behind when its header does not name the newest run, as ledger_text would.
            top = stt.newest(f.runs) if f and f.runs else {}
            row["table"] = t.info(dtype) | {"behind": bool(top) and (t.header.get("batchlas"), t.header.get("date"))
                                            != (top.get("batchlas"), top.get("date"))}
        if v is None:
            return row
        for c in v.best.values():
            tiers[c["tier"]] += 1
        for tier, _ in v.rows.values():
            row["rows"][tier] += 1
        for (d, _), verdict in f.audits.items():
            if d == dtype:
                k = verdict.split(":", 1)[0]
                audits[k if k in audits else "mismatch"] += 1
        newest = max((r.get("date", "") for r in v.runs), default="")
        try:
            row["age_days"] = (now - time.mktime(time.strptime(newest[:10], "%Y-%m-%d"))) / 86400 if newest else None
        except ValueError:
            pass
        row.update(runs=len(v.runs), cells=len(v.latest), best=len(v.best), stale=len(v.latest) - len(v.best),
                   partial=sum(v.freshness(c) == "partly_stale" for c in v.best.values()),
                   refined=sum(1 for c in v.best.values() if c["round"] > 0 and c["tier"] not in stt.UNHASHED), newest=newest, error=v.error)
        return row

    def op_view(self, op: str, dtype: str, device: str) -> dict:
        """_op_view, kept until the ledger, the shipped table or the kernel hashes change."""
        v, t = self.view(op, dtype, device), self.table(op, device)
        sig = (v.gen if v else 0, t.gen if t else 0)
        return self._cached(self._ops, (op, dtype, device), sig, lambda: self._op_view(op, dtype, device, v, t))

    def _op_view(self, op, dtype, device, v: LedgerView | None, t: TableFile | None) -> dict:
        """Every best record of one dtype, compact, plus the shipped table's pick at it.

        cells[i] = [key values in '# keys:' order, tier, round, winner, runner-up margin,
        partly stale, audit, shipped pick, predicted speedup, new pick, saved ms, within noise,
        vendor over best native]. A pick is the first entry of the nearest row (shipped, or the rows
        ledger_rows gives now) that the cell's record did not refuse (status skipped is can_run
        false); the speedup is the shipped pick's median over the new pick's, both at this cell,
        and saved ms their difference. Within noise: the two picks' intervals overlap.

        rows / table_rows = [[key values, [candidate index of each entry, -1 if not a candidate],
        [median ms of each entry or None], tier index (rows only)]], the next and the shipped table,
        so the map can show what either picks where nothing was measured. grid = the newest run's
        planned lattice per log key ({} before the run line carried it).
        """
        keyspec = v.keyspec if v else (t.keyspec if t and t.ok else [])
        names = [n for n, _, _ in keyspec]
        out = {"op": op, "dtype": dtype, "device": device, "tiers": TIERS, "names": names,
               "keys": [{"name": n, "log": is_log, "weight": w} for n, is_log, w in keyspec],
               "cands": v.cands if v else [], "families": v.families if v else [], "cells": [],
               "table": t.info(dtype) if t else None, "error": v.error if v else None,
               "family_status": self.families(op, device, [v] if v else []), "rows": [], "table_rows": []}
        if t and not t.ok:
            out["table_error"] = "the shipped table's first key is not dtype:exact"
        if v is None:
            return out
        out["next_rows"] = len(v.rows)
        out["grid"] = run_grid(stt.newest(v.runs).get("grid", "")) if v.runs else {}
        cand_ix = {c: i for i, c in enumerate(v.cands)}
        cells = list(v.best.values())
        tkeys = [stt.key_tuple(keyspec, c["key"]) for c in cells]
        shipped = t.rows.get(dtype, []) if t and t.ok else []
        if t and t.ok and [n for n, _, _ in t.keyspec] != names:
            out["table_error"] = f"the shipped table's keys {[n for n, _, _ in t.keyspec]} are not {names}"
            shipped = []
        if t:
            out["table"]["differ"] = self.sync(v, t)
        out["rows"] = [[list(k), [cand_ix.get(c, -1) for c, _ in e], [ms for _, ms in e], TIERS.index(tier)]
                       for k, (tier, e) in v.rows.items()]
        out["table_rows"] = [[list(k), [cand_ix.get(c, -1) for c, _ in e], [ms for _, ms in e]] for k, e in shipped]
        old_pick = Nearest(shipped, keyspec).lookup(tkeys) if shipped else [None] * len(cells)
        new_pick = Nearest([(k, e) for k, (_, e) in v.rows.items()], keyspec).lookup(tkeys)
        for c, tk, orow, nrow in zip(cells, tkeys, old_pick, new_pick):
            timed = [r for r in c["cands"] if r["status"] in TIMED and r["median"] and r["median"] > 0]
            med = {r["cand"]: float(r["median"]) for r in timed}
            ival = {r["cand"]: (r["lo"] or r["median"], r["hi"] or r["median"]) for r in timed}
            ranked = [r for r in c["ranked"] if r in med]
            margin = med[ranked[1]] / med[ranked[0]] - 1 if len(ranked) > 1 else None
            win = c["ranked"][0] if c["ranked"] else None
            refused = {r["cand"] for r in c["cands"] if r["status"] == "skipped" and r.get("reason") != "unraced:calibrate"}
            old = next((s for s, _ in orow if s not in refused), None) if orow else None
            new = next((s for s, _ in nrow if s not in refused), None) if nrow else None
            both = old in med and new in med
            sp = med[old] / med[new] if both else None
            noise = both and old != new and ival[old][0] <= ival[new][1] and ival[new][0] <= ival[old][1]
            native = min((t for s, t in med.items() if stt.spelling_family(s) != "vendor"), default=None)
            vr = med["vendor"] / native if "vendor" in med and native else None
            out["cells"].append([list(tk), TIERS.index(c["tier"]), c["round"], cand_ix.get(win, -1),
                                 None if margin is None else round(margin, 4), v.freshness(c) == "partly_stale",
                                 v.audit(c), cand_ix.get(old, -1) if old else -1,
                                 None if sp is None else round(sp, 4), cand_ix.get(new, -1) if new else -1,
                                 round(med[old] - med[new], 6) if both else None, noise,
                                 None if vr is None else round(vr, 4)])
        return out

    def cell(self, op: str, dtype: str, device: str, key: list) -> dict:
        """Every record of one cell (all tiers and runs), every candidate with its interval."""
        v = self.view(op, dtype, device)
        if v is None:
            return {"records": []}
        want = tuple(key)
        recs = []
        for c in v.cells:
            if stt.key_tuple(v.keyspec, c["key"]) != want:
                continue
            recs.append({"run_id": c["run_id"], "tier": c["tier"], "round": c["round"], "date": c["date"],
                         "best": v.best.get(c["key"]) is c, "fresh": v.freshness(c), "ranked": c["ranked"],
                         "key": key_text(c["key"]), "audit": v.audit(c),
                         "cands": [{k: r.get(k) for k in ("cand", "status", "reason", "median", "lo", "hi", "reps")}
                                   | {"current": c["tier"] in stt.UNHASHED or r["status"] == "skipped"
                                      or v.judge.get(stt.spelling_family(r["cand"])) == r["hash"]}
                                   for r in c["cands"]]})
        recs.sort(key=lambda r: (not r["best"], TIERS.index(r["tier"]), r["date"]))
        return {"records": recs}


def run_grid(text: str) -> dict:
    """A run line's `grid` ("n=1,2,4 batch=8,16") as {key: [ints]}; malformed words are skipped."""
    out = {}
    for word in text.split():
        name, _, vals = word.partition("=")
        try:
            out[name] = sorted(int(x) for x in vals.split(",") if x)
        except ValueError:
            continue
    return out


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

def tuner_problem(binary: Path) -> str | None:
    """Why this batchlas_tune cannot take the arguments this tab passes, or None."""
    try:
        r = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"{binary} --help failed: {e}"
    missing = [f for f in TUNER_FLAGS if f not in r.stdout + r.stderr]
    return f"{binary} predates the current tuner (no {', '.join(missing)}): rebuild batchlas_tune" if missing else None


def find_tuner(build_dirs) -> tuple:
    """(binary, None) for the first build whose batchlas_tune takes this tab's arguments, else
    (None, why): no build tree, no binary in it, or only binaries built before the current tuner."""
    dirs = [Path(d) for d in build_dirs or []]
    if not dirs:
        return None, "no build tree found; configure one and build the batchlas_tune target"
    problems = []
    for d in dirs:
        p = d / "tools" / "tune" / "batchlas_tune"
        if p.is_file() and os.access(p, os.X_OK):
            why = tuner_problem(p)
            if why is None:
                return p, None
            problems.append(why)
    return None, "; ".join(problems) or \
        f"no build has tools/tune/batchlas_tune ({', '.join(map(str, dirs))}); build the batchlas_tune target"


def tuner_binary(build_dirs) -> Path | None:
    return find_tuner(build_dirs)[0]


def tree_dirty(repo: Path = REPO) -> bool | None:
    """Tracked files with uncommitted changes, which the tuner refuses without --allow-dirty."""
    r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip()) if r.returncode == 0 else None


def remeasure_lines(req: dict, ops: list, dtypes: list) -> list:
    """The request's '<op> <dtype> <name=v,...>' lines; every op and dtype named must be in the run."""
    lines = [ln.strip() for ln in str(req.get("remeasure_keys") or "").splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        part = ln.split("#", 1)[0].split()
        if len(part) != 3 or "=" not in part[2]:
            raise ValueError(f"remeasure line '{ln}' is not '<op> <dtype> <name=v,...>'")
        if part[0] not in ops or part[1] not in dtypes:
            raise ValueError(f"remeasure line '{ln}' names an op or precision outside the run")
    return lines


def run_argv(binary: Path, req: dict, ledger: Path, keys_file: Path | None = None, plan: bool = False) -> list:
    """The batchlas_tune command line for a launch request; raises ValueError on a bad one.
    keys_file receives the request's remeasure lines (--remeasure-keys)."""
    known = set(spec_ops())
    ops = [o for o in req.get("ops", []) if o in known]
    dtypes = [t for t in req.get("dtypes", []) if t in DTYPES]
    tier = req.get("tier", "")
    devices = sorted({int(g) for g in req.get("devices", [])})
    device_key = str(req.get("device_key") or "")
    if not ops or not dtypes:
        raise ValueError("choose at least one op and one precision")
    if tier not in TIER_NOTES:
        raise ValueError(f"tier must be one of {', '.join(TIER_NOTES)}")
    if not devices and not (plan and device_key):
        raise ValueError("choose at least one GPU")
    keys = remeasure_lines(req, ops, dtypes)
    if tier == "custom" and not keys:
        raise ValueError("tier custom measures exactly the listed cells: give at least one '<op> <dtype> <key>' line")
    argv = [str(binary), ",".join(ops), "--tier", tier, "--dtype", ",".join(dtypes),
            "--repo", str(REPO), "--ledger", str(ledger)]
    if devices:
        argv += ["--devices", ",".join(map(str, devices))]
    if plan and device_key:
        argv += ["--device-key", device_key]
    if keys:
        if keys_file is None:
            raise ValueError("remeasure lines need a file")
        keys_file.write_text("\n".join(keys) + "\n")
        argv += ["--remeasure-keys", str(keys_file)]
    if req.get("max_dim"):
        argv += ["--max-dim", str(int(req["max_dim"]))]
    if req.get("budget_h"):
        argv += ["--budget", str(float(req["budget_h"]))]
    if req.get("allow_idle_foreign", True):
        argv += ["--allow-idle-foreign"]
    if req.get("allow_dirty"):
        argv += ["--allow-dirty"]
    if req.get("guard_wait_s"):
        argv += ["--guard-wait", str(int(req["guard_wait_s"]))]
    if req.get("out"):
        argv += ["--out", str(req["out"])]
    return argv


def plan(binary: Path, req: dict, ledger: Path, timeout: float = 300) -> dict:
    """`batchlas_tune --plan`: the per-(op, dtype) plan and the estimate, no GPU work."""
    with tempfile.TemporaryDirectory(prefix="benchviz-plan-") as tmp:
        argv = run_argv(binary, req, ledger, Path(tmp) / "remeasure-keys.txt", plan=True) + ["--plan"]
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
    name = "".join(ch for ch in (name or "") if ch.isalnum() or ch in "-_.") or \
        time.strftime(f"{req.get('tier', 'run')}-%Y%m%d-%H%M%S")
    d = runs_dir(root) / name
    if (d / "run.json").exists():
        raise ValueError(f"a tuning run named {name} exists; resuming starts a new run, which skips what it holds")
    d.mkdir(parents=True, exist_ok=True)
    try:
        argv = run_argv(binary, req, ledger, d / "remeasure-keys.txt")
    except ValueError:
        if not any(d.iterdir()):
            d.rmdir()
        raise
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
