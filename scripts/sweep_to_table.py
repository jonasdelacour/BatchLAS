#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write and verify the tuned selection tables (tuned/<op>.<device>.txt) from the tuning ledgers.

Spec: docs/design/tuning-data.md (files, ledger lines, generation, the check);
docs/design/tiered-tuning.md "Engine: the ledger and table generation" (best record, freshness,
tier bracketing); docs/design/flat-kernel-selection.md section 5.4 (table format, nearest row).

    scripts/sweep_to_table.py                      # regenerate every tuned/<op>.<device>.txt from its ledger
    scripts/sweep_to_table.py --op potrf --device sm_120 --out DIR
    scripts/sweep_to_table.py --check              # CI: re-derive every table byte for byte
    scripts/sweep_to_table.py --self-test
    scripts/sweep_to_table.py --emit-sample OUT --op potrf --device sm_120 [--cells N]
    scripts/sweep_to_table.py --rehash-moved BASE [--op OP] [--device DEV]
    scripts/sweep_to_table.py --rehash-renamed BASE [--op OP] [--device DEV]

The one input is benchmarks/results/tuning/ledger/<op>.<device>.jsonl (--ledger DIR names another
root). Every line carries "dtype" except `rehash`. Keys and candidates come from each dtype's newest
`run` line (date, then run id), never from this script. Per dtype and cell the best record wins:
tier deep > coarse > calibrate > preview > custom > converted > prior > transcribed, then newest
date, run id, later line; a record whose winner's family hash is stale, or whose every candidate is
"error", is no record. converted, prior and transcribed records carry no hash and are never stale.
A lower-tier row is dropped when a higher-tier row has equal exact keys and lies within the tier's
stride (deep and coarse 1, the rest 2, in index steps of the per-key lattice of the dtype's current
measured cells) on every log key; converted and transcribed rows are dropped near any measured row,
kept or not, transcribed rows near a prior row too; a prior row (calibrate's copy of a borrowed
table) only where a measured record has its own key.

Hashes: kernel hash v2 over the spec's kernel-sources block, sha256 of "<sha256(normalized file)>
<path>\\n" lines, first 8 hex; normalized = /* */ blocks removed (an unterminated one runs to the
end), then // to end of line, then every whitespace byte. A `rehash` line maps a family's stored
hash `from` to `to`. A run whose batchlas ends in -dirty is excluded with a warning.

--check judges freshness with the hashes in each table's header, so it proves table == f(ledger)
whatever the sources say today; header hashes that differ from the sources are a warning (the op
needs a retune), never a failure. It also checks the '# keys:' line, the vocabulary, the row kinds
and that every row looks up to itself. Python 3 standard library only.
"""

import argparse
import bisect
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNED = "tuned"
LEDGER_ROOT = "benchmarks/results/tuning/ledger"
DTYPES = ("float", "double", "cfloat", "cdouble")

LEDGER = "ledger:"
LEDGER_TIERS = ("deep", "coarse", "calibrate", "preview", "custom", "converted", "prior", "transcribed")  # highest first
LEDGER_STRIDE = {"deep": 1, "coarse": 1, "calibrate": 2, "preview": 2, "custom": 2, "converted": 2, "prior": 0,
                 "transcribed": 2}  # 0: its own key only
PRIOR_ABOVE_STRIDE = 2  # a coarse or deep row brackets a prior row as it does a transcribed one
UNHASHED = ("converted", "prior", "transcribed")
LEDGER_STATUS_TIMED = ("ok", "eliminated")
TIE = 0.03  # the tuner's tie band (tune_core.cc rank())
LFS_POINTER = "version https://git-lfs.github.com/spec/v1"  # the first line of an unfetched LFS file
HASH_VECTOR = "tests/data/kernel_hash_v2"
PROTOCOL = "lean-cold"  # tools/tune/ledger.hh kProtocol: a measured record of another protocol ranks below
MONOTONE_DROP = 5.0  # a whole-cell time this many times lower at a larger batch flags both cells


def parse_keys(spec):
    """'# keys:' text -> [(name, is_log, weight)]; a :log weight defaults to 1."""
    out = []
    for tok in spec.split():
        part = tok.split(":")
        is_log = part[1] == "log"
        out.append((part[0], is_log, float(part[2]) if is_log and len(part) > 2 else 1.0))
    return out


def fmt_ms(t):
    """Four significant figures in plain decimal (no exponent), for any parser."""
    if t <= 0:
        raise SystemExit(f"non-positive time {t}")
    decimals = max(0, 3 - int(math.floor(math.log10(t))))
    return f"{t:.{decimals}f}"


def quoted_paths(line):
    """Every "..." pair on the line, like the C++ quoted_paths (an unmatched quote ends it)."""
    out, a = [], line.find('"')
    while a != -1:
        b = line.find('"', a + 1)
        if b == -1:
            break
        out.append(line[a + 1:b])
        a = line.find('"', b + 1)
    return out


def family_marker(line):
    """'// family: name' -> name, else ''; anchored at the start of the line like the C++ parser."""
    m = re.match(r'\s*//\s*family:[ ]*([^ "]*)', line)
    return m.group(1) if m else ""


def parse_kernel_block(text):
    """tools/tune/tune_core.cc parse_kernel_block: the kernel-sources block split by family
    ('// family: x' opens a section, '// common' returns to common) plus the kernel-deps comment
    block ('// family: x "path" ...'; '// common "path" ...' for every family). Returns
    {'all', 'common', 'family', 'deps', 'deps_common'}."""
    out = {"all": [], "common": [], "family": defaultdict(list), "deps": defaultdict(list), "deps_common": []}
    state, section = "none", ""
    for line in text.split("\n"):
        if "kernel-sources-begin" in line:
            state, section = "sources", ""
            continue
        if "kernel-sources-end" in line:
            state = "none"
            continue
        if "kernel-deps-begin" in line:
            state = "deps"
            continue
        if "kernel-deps-end" in line:
            break
        if state == "none":
            continue
        fam = family_marker(line)
        if state == "deps":
            if fam:
                out["deps"][fam] += quoted_paths(line)
            elif re.fullmatch(r'\s*//\s*common(\s*"[^"]*")*\s*', line):
                out["deps_common"] += quoted_paths(line)
            continue
        if fam:
            section = fam
        elif line.strip() == "// common":
            section = ""
        for path in quoted_paths(line):
            out["all"].append(path)
            (out["family"][section] if section else out["common"]).append(path)
    return out


def normalize_source(data):
    """Hash v2's normalization, on bytes: /* */ blocks (an unterminated one to the end), then //
    to end of line, then every whitespace byte. Naive: string literals are not special."""
    data = re.sub(rb"/\*.*?(?:\*/|\Z)", b"", data, flags=re.S)
    data = re.sub(rb"//[^\n]*", b"", data)
    return re.sub(rb"[ \t\n\r\f\v]+", b"", data)


def manifest_hash(files):
    """Hash v2 of [(path, bytes)]: first 8 hex of sha256 over '<sha256(normalized file)>  path' lines."""
    manifest = "".join(hashlib.sha256(normalize_source(data)).hexdigest() + "  " + p + "\n" for p, data in files)
    return hashlib.sha256(manifest.encode()).hexdigest()[:8]


def kernel_hash(paths, repo=REPO):
    """Hash v2 of the working-tree files."""
    files = []
    for p in paths:
        try:
            with open(os.path.join(repo, p), "rb") as f:
                files.append((p, f.read()))
        except OSError:
            raise SystemExit(f"kernel source missing: {p}")
    return manifest_hash(files)


def family_paths(block, family):
    return block["common"] + block["family"].get(family, []) + block["deps_common"] + block["deps"].get(family, [])


def family_hashes(block, families, repo=REPO):
    """Per family: kernel_hash over common + the family's files + deps_common + its deps."""
    return {f: kernel_hash(family_paths(block, f), repo) for f in families}


def read_spec_source(op, repo=REPO):
    path = os.path.join(repo, "tools", "tune", f"{op}_spec.cc")
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        raise SystemExit(f"{path}: cannot read the kernel-sources block of {op}")


def spelling_family(spelling):
    return spelling.split(":", 1)[0]


@dataclass
class Ledger:
    runs: list = field(default_factory=list)   # newest line per (run id, dtype); dirty runs left out
    cells: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    rehash: dict = field(default_factory=dict)  # family -> {from: to}


def read_cell(rec):
    if rec["tier"] not in LEDGER_TIERS:
        raise ValueError(f"unknown tier '{rec['tier']}'")
    key = tuple(tuple(kv.split("=", 1)) for kv in rec["key"].split(","))
    names = [n for n in rec["cands"].split("|") if n]
    cands = [{"cand": n, "hash": rec.get(f"h{i}", ""), "status": rec.get(f"s{i}", ""),
              "reason": rec.get(f"r{i}") or "", "median": rec.get(f"m{i}"), "lo": rec.get(f"lo{i}"),
              "hi": rec.get(f"hi{i}"), "reps": rec.get(f"n{i}"), "verified": rec.get(f"v{i}", 1) != 0}
             for i, n in enumerate(names)]
    return {"run_id": rec["run_id"], "dtype": rec["dtype"], "tier": rec["tier"], "key": key,
            "date": rec.get("date", ""), "round": int(rec.get("round", 0)),
            "ranked": [r for r in rec.get("ranked", "").split("|") if r], "cands": cands}


def read_ledger(path, dtype=None, raw=False):
    """One <op>.<device>.jsonl, lines in file order (only `dtype`'s when given). A malformed last
    line is a warning, an earlier one fatal. rehash lines rewrite stored hashes (not when raw); a
    -dirty run is dropped with its cells."""
    led = Ledger()
    with open(path) as f:
        lines = [(i, ln) for i, ln in enumerate(f.read().split("\n"), start=1) if ln.strip()]
    if lines and lines[0][1].startswith(LFS_POINTER):
        raise SystemExit(f"{path}: Git LFS pointer: run git lfs pull")
    runs, rehash = {}, defaultdict(dict)
    for pos, (no, line) in enumerate(lines):
        try:
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "rehash":
                rehash[rec["family"]][rec["from"]] = rec["to"]
                continue
            if dtype is not None and rec["dtype"] != dtype:
                continue
            if kind == "run":  # a repeated (run id, dtype) (a worker-mode update) replaces the earlier line
                ident = (rec["run_id"], rec["dtype"])
                runs.pop(ident, None)
                runs[ident] = rec
            elif kind == "cell":
                led.cells.append(read_cell(rec))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
            where = f"{path}:{no}: {e}"
            if pos + 1 != len(lines):
                raise SystemExit(where)
            led.warnings.append(f"skipping truncated last line, {where}")
    dirty = {ident for ident, run in runs.items() if run.get("batchlas", "").endswith("-dirty")}
    for ident in sorted(dirty):
        led.warnings.append(f"{path}: run {ident[0]} ({ident[1]}) is {runs[ident]['batchlas']}: excluded from the table")
    led.runs = [r for ident, r in runs.items() if ident not in dirty]
    led.cells = [c for c in led.cells if (c["run_id"], c["dtype"]) not in dirty]
    for c in led.cells:
        c["protocol"] = runs.get((c["run_id"], c["dtype"]), {}).get("protocol", "")
    led.rehash = dict(rehash)
    for c in [] if raw else led.cells:
        for r in c["cands"]:
            r["hash"] = follow_rehash(rehash.get(spelling_family(r["cand"]), {}), r["hash"])
    return led


def follow_rehash(chain, h):
    """A stored hash through every rehash step (a normalized step, then a sample step, ...); a cycle stops."""
    seen = {h}
    while h in chain and chain[h] not in seen:
        h = chain[h]
        seen.add(h)
    return h


def stale_families(cell, family_hash):
    if cell["tier"] in UNHASHED:
        return set()
    seen, out = set(), set()
    for c in cell["cands"]:
        fam = spelling_family(c["cand"])
        seen.add(fam)
        # A candidate that could not run there (skipped) cannot change the ranking: its hash is ignored.
        # A dominated skip (the removed carry-forward) was never tried there: stale, as in C++.
        if c["status"] == "skipped" and c.get("reason", "").startswith("dominated:"):
            out.add(fam)
        elif c["status"] != "skipped" and family_hash.get(fam) != c["hash"]:
            out.add(fam)
    return out | (set(family_hash) - seen)


def freshness(cell, family_hash):
    """'current' | 'partly_stale' | 'stale', the C++ freshness(): only the winner's staleness
    voids a record; an empty ranking has no winner; converted and transcribed are never stale."""
    if cell["tier"] in UNHASHED:
        return "current"
    if cell["ranked"]:
        win = next((c for c in cell["cands"] if c["cand"] == cell["ranked"][0]), None)
        if win is None or family_hash.get(spelling_family(win["cand"])) != win["hash"]:
            return "stale"
    return "partly_stale" if stale_families(cell, family_hash) else "current"


def all_error(cell):
    """Every candidate `error`: a failed child, not a result. Counts as no record (C++ all_error)."""
    return bool(cell["cands"]) and all(c["status"] == "error" for c in cell["cands"])


def usable(cell, family_hash):
    return not all_error(cell) and freshness(cell, family_hash) != "stale"


def best_records(cells, family_hash):
    """Per key, skipping all-error and stale records: a measured PROTOCOL record first, then the
    highest tier, newest date, larger run id, then the later line (the C++ best_records)."""
    def order(c):
        cold = c["tier"] not in UNHASHED and c.get("protocol") == PROTOCOL
        return (cold, LEDGER_TIERS[::-1].index(c["tier"]), c["date"], c["run_id"])
    best = {}
    for c in cells:
        if not usable(c, family_hash):
            continue
        if c["key"] not in best or order(c) >= order(best[c["key"]]):
            best[c["key"]] = c
    return best


def key_tuple(keyspec, key):
    """A record's (name, value) pairs -> the table key tuple (log keys int). Extra fields are hidden
    grid axes (trsm's uplo, diag); hidden_axes_problem refuses one that varied."""
    given = dict(key)
    missing = [n for n, _, _ in keyspec if n not in given]
    if missing:
        raise SystemExit(f"ledger key {key} lacks the '# keys:' fields {missing}")
    return tuple(int(given[n]) if is_log else given[n] for n, is_log, _ in keyspec)


def hidden_axes_problem(keyspec, cells):
    """A field a cell key carries beyond the '# keys:' must hold one value over the dtype's cells:
    a run that varied it (an A/B) would fold several cells into one row."""
    names = {n for n, _, _ in keyspec}
    seen = defaultdict(set)
    for c in cells:
        for k, v in c["key"]:
            if k not in names:
                seen[k].add(v)
    varied = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    return f"hidden grid axes varied {varied}: an A/B run is raw data, not a table" if varied else None


def lattice_index(values, v):
    """Position of v in the sorted distinct values; off the lattice, interpolated in log2 space."""
    if v in values:
        return float(values.index(v))
    hi = next((i for i, x in enumerate(values) if x > v), len(values))
    if hi == 0:
        return -math.log2(values[0] / v)
    if hi == len(values):
        return len(values) - 1 + math.log2(v / values[-1])
    lo = hi - 1
    return lo + math.log2(v / values[lo]) / math.log2(values[hi] / values[lo])


def ledger_rows(keyspec, cells, family_hash):
    """One dtype's {key: (tier, [(spelling, ms or None)])}: each cell's best record, minus rows of a
    lower tier that a higher-tier row brackets (equal exact keys, index distance < the tier's stride
    on every log key); converted and transcribed rows are bracketed by any measured row, transcribed rows
    by a prior row too, and a prior row by a measured record at its own key or a coarse or deep row
    within PRIOR_ABOVE_STRIDE (a frontier run measures every 2nd cell and reads the rest from them)."""
    problem = hidden_axes_problem(keyspec, cells)
    if problem:
        raise SystemExit(problem)
    measured = [c for c in cells if c["tier"] not in UNHASHED and usable(c, family_hash)]
    log_pos = [i for i, (_, is_log, _) in enumerate(keyspec) if is_log]
    exact_pos = [i for i, (_, is_log, _) in enumerate(keyspec) if not is_log]
    lattice = {i: sorted({key_tuple(keyspec, c["key"])[i] for c in measured}) for i in log_pos}
    cand_rows = []
    # Measured and unhashed records compete separately: an unhashed row still stands where the
    # measured record's ranking is empty (it emits no row), as when transcribed rows came from the old table.
    best = list(best_records([c for c in cells if c["tier"] not in UNHASHED], family_hash).items()) + \
        list(best_records([c for c in cells if c["tier"] in UNHASHED], family_hash).items())
    for key, c in best:
        entries = []
        for name in c["ranked"]:
            res = next((r for r in c["cands"] if r["cand"] == name), None)
            # An eliminated candidate with no median cannot print a time: left out of the row.
            if res and res["status"] in LEDGER_STATUS_TIMED and res["median"] and res["median"] > 0:
                entries.append((name, float(res["median"])))
        # A ranking with no time at all (transcribed, or a single runnable candidate never timed) is an untimed row.
        if not entries and c["ranked"]:
            entries = [(name, None) for name in c["ranked"]]
        if entries:
            cand_rows.append((key_tuple(keyspec, key), c["tier"], entries))
    if not measured:  # no current measured cell: transcribed rows are bracketed by prior rows on the prior's lattice
        lattice = {i: sorted({k[i] for k, t, _ in cand_rows if t == "prior"}) for i in log_pos}
    pos = {}

    def index_of(k):
        if k not in pos:
            pos[k] = tuple(lattice_index(lattice[i], k[i]) for i in log_pos)
        return pos[k]

    def bracketed_by(against, stride):
        """near(a, b) = equal exact keys and index distance < stride on every log key. Two such keys
        sit in the same or an adjacent floor(index / stride) bucket, so only those are compared."""
        if not against:
            return lambda key: False
        buckets = defaultdict(list)
        for k in against:
            p = index_of(k)
            buckets[(tuple(k[i] for i in exact_pos), tuple(math.floor(x / stride) for x in p))].append(p)

        def hit(key):
            exact, p = tuple(key[i] for i in exact_pos), index_of(key)
            home = [math.floor(x / stride) for x in p]
            for off in itertools.product((-1, 0, 1), repeat=len(p)):
                for q in buckets.get((exact, tuple(h + o for h, o in zip(home, off))), ()):
                    if all(abs(a - b) < stride for a, b in zip(p, q)):
                        return True
            return False
        return hit

    kept = {}
    for tier in LEDGER_TIERS:
        # Measured rows are bracketed by kept higher rows; an unhashed row by any measured row,
        # kept or dropped, since a dropped preview row still says the region was measured.
        if tier in UNHASHED:
            below_prior = LEDGER_TIERS.index(tier) > LEDGER_TIERS.index("prior")
            against = [k for k, t, _ in cand_rows if t not in UNHASHED or (below_prior and t == "prior")]
        else:
            against = list(kept)
        near = (lambda key, own=set(against): key in own) if LEDGER_STRIDE[tier] == 0 else \
            bracketed_by(against, LEDGER_STRIDE[tier])
        if tier == "prior":
            above = bracketed_by([k for k, t, _ in cand_rows if LEDGER_TIERS.index(t) < LEDGER_TIERS.index("calibrate")],
                                 PRIOR_ABOVE_STRIDE)
            near = (lambda key, own=near, wide=above: own(key) or wide(key))
        for key, t, entries in cand_rows:
            if t != tier or key in kept:
                continue
            if not near(key):
                kept[key] = (tier, entries)
    return kept


def winner_ms(cell):
    """The whole-cell time of a record: its first ranked candidate's median, None when untimed."""
    for name in cell["ranked"]:
        res = next((r for r in cell["cands"] if r["cand"] == name), None)
        if res and res["status"] in LEDGER_STATUS_TIMED and res["median"] and res["median"] > 0:
            return float(res["median"])
    return None


def monotonicity(op, dtype, cells, family_hash):
    """--remeasure-keys lines for the cells around a drop: at fixed other key fields, a cell whose
    whole-cell time exceeds MONOTONE_DROP x a larger batch's, and that larger batch. Whole-cell time
    cannot fall as the batch grows, so one of the two is a bad measurement (co-tenant time-slicing,
    a throttled run, a warm-L2 time). Reports only: nothing is re-measured here."""
    groups = defaultdict(list)
    for key, c in best_records([c for c in cells if c["tier"] not in UNHASHED], family_hash).items():
        fields = dict(key)
        ms = winner_ms(c)
        if "batch" not in fields or ms is None or not fields["batch"].isdigit():
            continue
        groups[tuple(kv for kv in key if kv[0] != "batch")].append((int(fields["batch"]), ms, key))
    flagged = set()
    for row in groups.values():
        row.sort()
        worst = None  # the slowest smaller batch so far
        for b, ms, key in row:
            if worst and worst[0] > MONOTONE_DROP * ms:
                flagged.update((worst[1], key))
            if worst is None or ms > worst[0]:
                worst = (ms, key)
    return [f"{op} {dtype} " + ",".join(f"{k}={v}" for k, v in key) for key in sorted(flagged)]


def newest(runs):
    return max(runs, key=lambda r: (r.get("date", ""), r["run_id"]))


def fmt_key(keyspec, key):
    return " ".join(f"{name}={v}" for (name, _, _), v in zip(keyspec, key))


def rel_path(path):
    """A path as a table header names it: repo-relative when inside the repo."""
    rel = os.path.relpath(os.path.abspath(path), REPO)
    return os.path.abspath(path) if rel.startswith("..") else rel


def split_name(path, suffix):
    """(op, device) from <op>.<device><suffix>, else None."""
    base = os.path.basename(path)
    if not base.endswith(suffix):
        return None
    parts = base[:-len(suffix)].split(".")
    return tuple(parts) if len(parts) == 2 and all(parts) else None


@dataclass
class Derived:
    text: str
    counts: dict
    warnings: list
    vocab: dict          # dtype -> candidate spellings of its newest run
    keys: str            # the run lines' '# keys:' text
    remeasure: list = field(default_factory=list)  # monotonicity(): --remeasure-keys lines


def ledger_text(op, device, ledger_path, judge=None, kernels=None, repo=REPO):
    """The table for one ledger. judge: family -> hash to judge freshness with (default: hash v2 of
    tools/tune/<op>_spec.cc's sources now); kernels: the op-level hash to print (default likewise)."""
    led = read_ledger(ledger_path)
    if not led.runs:
        raise SystemExit(f"{ledger_path}: no clean 'run' line")
    by_dtype = defaultdict(list)
    for r in led.runs:
        by_dtype[r["dtype"]].append(r)
    unknown = sorted(set(by_dtype) - set(DTYPES))
    if unknown:
        raise SystemExit(f"{ledger_path}: unknown dtypes {unknown}")
    dtypes = [d for d in DTYPES if d in by_dtype]
    heads = {d: newest(by_dtype[d]) for d in dtypes}
    keys = {heads[d].get("keys", "") for d in dtypes}
    if len(keys) != 1 or not next(iter(keys)):
        raise SystemExit(f"{ledger_path}: the dtypes' newest run lines disagree on keys {sorted(keys)}")
    keys = keys.pop()
    vocab = {d: [c for c in heads[d].get("candidates", "").split("|") if c] for d in dtypes}
    families = list(dict.fromkeys(spelling_family(c) for d in dtypes for c in vocab[d]))
    if judge is None or kernels is None:
        block = parse_kernel_block(read_spec_source(op, repo))
        judge = family_hashes(block, families, repo) if judge is None else judge
        kernels = kernel_hash(block["all"], repo) if kernels is None else kernels
    keyspec = parse_keys(keys)
    cells_by = defaultdict(list)
    for c in led.cells:
        cells_by[c["dtype"]].append(c)
    counts = {t: 0 for t in LEDGER_TIERS}
    body, remeasure = [], []
    for d in dtypes:
        remeasure += monotonicity(op, d, cells_by[d], judge)
        rows = ledger_rows(keyspec, cells_by[d], judge)
        for key in sorted(rows):
            tier, entries = rows[key]
            bad = [c for c, _ in entries if c not in vocab[d]]
            if bad:
                raise SystemExit(f"{ledger_path}: {d} row {key} names {bad}, not candidates of its newest run")
            counts[tier] += 1
            text = " | ".join(f"{c} {'-' if ms is None else fmt_ms(ms)}" for c, ms in entries)
            body.append(f"dtype={d} {fmt_key(keyspec, key)} | {text} # {tier}")
    top = newest(led.runs)
    lines = [
        f"# op={op} device={device} batchlas={top['batchlas']} kernels={kernels} "
        f"family_kernels={','.join(f + ':' + judge.get(f, '') for f in families)} date={top['date']}",
        f"# source={LEDGER}{rel_path(ledger_path)} tiers={','.join(f'{t}:{counts[t]}' for t in LEDGER_TIERS)}",
        f"# keys: dtype:exact {keys}",
    ]
    return Derived("\n".join(lines + body) + "\n", counts, led.warnings, vocab, keys, remeasure)


def parse_table(text):
    """Parse the section 5.4 format into (header, keyspec, [(key, [(choice, ms or None)])]).

    Header comment words k=v go into header (first one wins); '-' parses as an untimed entry.
    """
    header, keyspec, rows = {}, [], []
    for line in text.splitlines():
        if line.startswith("# keys:"):
            keyspec = parse_keys(line[len("# keys:"):])
            continue
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    header.setdefault(k, v)
            continue
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        parts = [p.strip() for p in body.split("|")]
        given = dict(tok.split("=", 1) for tok in parts[0].split())
        key = tuple(int(given[n]) if is_log else given[n] for n, is_log, _ in keyspec)
        ranked = []
        for p in parts[1:]:
            choice, ms = p.split()
            ranked.append((choice, None if ms == "-" else float(ms)))
        rows.append((key, ranked))
    return header, keyspec, rows


def nearest(rows, key, keyspec):
    """Section 5.4 lookup, the same rule as Table::nearest (src/select/select.cc).

    Keep the rows matching the longest prefix of exact keys in '# keys:' order (exact keys
    drop from the right); among those, min sum w*|log2(row/key)| over the log keys; a tie
    goes to the smaller log keys compared in '# keys:' order (n first).
    """
    exact = [i for i, (_, is_log, _) in enumerate(keyspec) if not is_log]
    pool = rows
    for p in range(len(exact), -1, -1):
        pool = [r for r in rows if all(r[0][i] == key[i] for i in exact[:p])]
        if pool:
            break
    logs = [(i, w) for i, (_, is_log, w) in enumerate(keyspec) if is_log]
    # Same arithmetic and 1e-9 tie window as Table::nearest: with log2(row / key) the two
    # sides of a geometric midpoint differ by an ulp and the tie is lost.
    lk = {i: math.log2(max(1, key[i])) for i, _ in logs}
    best, best_d, best_t = None, 0.0, None
    for r in pool:
        d = 0.0
        for i, w in logs:
            d += w * abs(math.log2(r[0][i]) - lk[i])
        t = tuple(r[0][i] for i, _ in logs)
        tie = best is not None and abs(d - best_d) <= 1e-9
        if best is None or (not tie and d < best_d) or (tie and t < best_t):
            best, best_d, best_t = r, d, t
    return best


def row_kind_problems(header, parsed):
    """parse_table()'s (src/select/select.cc) rules: a row is all-timed or all-untimed, and an
    untimed row needs a ledger: source."""
    out = []
    for key, ranked in parsed:
        kinds = {ms is None for _, ms in ranked}
        if len(kinds) > 1:
            out.append(f"row {key} mixes timed and untimed ('-') entries")
        elif kinds == {True} and not header.get("source", "").startswith(LEDGER):
            out.append(f"untimed ('-') row {key} needs a 'source={LEDGER}<file>' header")
    return out


def header_hashes(header):
    """family_kernels=a:h1,b:h2 -> {a: h1, b: h2}."""
    out = {}
    for item in header.get("family_kernels", "").split(","):
        if ":" in item:
            f, h = item.split(":", 1)
            out[f] = h
    return out


def check_table(path, failures, notes, repo=REPO, remeasure=None):
    """Re-derive one table from the ledger its source= names, judged by its header's hashes, plus
    the format checks; stale header hashes go to notes."""
    rel = os.path.relpath(path, repo)
    ident = split_name(path, ".txt")
    if ident is None:
        failures.append(f"{rel}: not named <op>.<device>.txt")
        return None
    with open(path) as f:
        disk = f.read()
    header, keyspec, parsed = parse_table(disk)
    source = header.get("source", "")
    if not source.startswith(LEDGER):
        failures.append(f"{rel}: source={source or '(none)'} is not a ledger")
        return None
    src = source[len(LEDGER):]
    ledger = src if os.path.isabs(src) else os.path.join(repo, src)
    if not os.path.isfile(ledger):
        failures.append(f"{rel}: ledger {src} is missing, cannot re-derive")
        return None
    judge = header_hashes(header)
    der = ledger_text(*ident, ledger, judge, header.get("kernels", ""), repo)
    notes += [f"{rel}: {w}" for w in der.warnings]
    notes += [f"{rel}: monotonicity: {line}" for line in der.remeasure]
    if remeasure is not None:
        remeasure += der.remeasure
    if disk != der.text:
        failures.append(f"{rel}: differs from {src} (re-run scripts/sweep_to_table.py)")
    if not keyspec or keyspec[0][:2] != ("dtype", False) or keyspec[1:] != parse_keys(der.keys):
        failures.append(f"{rel}: '# keys:' is not 'dtype:exact {der.keys}'")
    if len({k for k, _ in parsed}) != len(parsed):
        failures.append(f"{rel}: duplicate row keys")
    failures += [f"{rel}: {p}" for p in row_kind_problems(header, parsed)]
    # A row's own exact keys match in full, so its lookup pool is the rows sharing them; at distance 0
    # only a row with the same first log key can tie or beat it, so the pool narrows to those (same answer).
    pool = [i for i, (_, is_log, _) in enumerate(keyspec) if not is_log]
    pool += [i for i, (_, is_log, _) in enumerate(keyspec) if is_log][:1]
    groups = defaultdict(list)
    for row in parsed:
        groups[tuple(row[0][i] for i in pool)].append(row)
    for key, ranked in parsed:
        hit = nearest(groups[tuple(key[i] for i in pool)], key, keyspec)
        if hit is None or hit[0] != key:
            failures.append(f"{rel}: nearest{key} -> {hit and hit[0]}")
        bad = [c for c, _ in ranked if c not in der.vocab.get(key[0], [])]
        if bad or len({c for c, _ in ranked}) != len(ranked):
            failures.append(f"{rel}: row {key} has unknown or repeated spellings {bad}")
    notes += stale_kernel_warnings(path, repo)
    print(f"check {rel}: {len(parsed)} rows re-derived from {src}")
    return ledger


def stale_kernel_warnings(table_path, repo=REPO):
    """Header hashes against tools/tune/<op>_spec.cc now: a kernel edit since the table was made.
    A table with no measured row (tiers= counts only unhashed tiers) carries no hash and is
    never stale; a missing or unparsable tiers= is checked. A missing spec is a note, not a warning."""
    rel = os.path.relpath(table_path, repo)
    with open(table_path) as f:
        header = parse_table(f.read().split("\n# keys:", 1)[0])[0]
    tokens = [t.split(":", 1) for t in header.get("tiers", "").split(",")]
    if all(len(t) == 2 and t[1].isdigit() for t in tokens):
        counts = dict(tokens)
        if not any(int(counts.get(t, "0")) > 0 for t in LEDGER_TIERS if t not in UNHASHED):
            return []
    op, judge = os.path.basename(table_path).split(".")[0], header_hashes(header)
    try:
        block = parse_kernel_block(read_spec_source(op, repo))
        now = family_hashes(block, list(judge), repo)
        op_now = kernel_hash(block["all"], repo)
    except SystemExit as e:
        return [f"note: {rel}: kernel sources unreadable ({e}); staleness not checked"]
    fams = [f for f, h in judge.items() if now.get(f) != h]
    out = []
    if fams or header.get("kernels") != op_now:
        out.append(f"warning: {rel} is stale: kernels {header.get('kernels')} -> {op_now}, families "
                   + (",".join(f"{f}:{judge[f]}->{now[f]}" for f in fams) or "none") + " (retune the op)")
    return out


def check(tuned_dir, ledger_root, repo=REPO, remeasure=None):
    failures, notes = [], []
    names = sorted(f for f in os.listdir(tuned_dir) if not f.startswith("."))
    covered = set()
    for name in names:
        if name == "README.md" or os.path.isdir(os.path.join(tuned_dir, name)):
            continue
        led = check_table(os.path.join(tuned_dir, name), failures, notes, repo, remeasure)
        if led:
            covered.add(os.path.realpath(led))
    for name in sorted(os.listdir(ledger_root)):
        path = os.path.join(ledger_root, name)
        if name.endswith(".jsonl") and os.path.realpath(path) not in covered:
            failures.append(f"{os.path.relpath(path, repo)}: no table in {os.path.relpath(tuned_dir, repo)} names it")
    return failures, notes


def ledger_files(root, op=None, device=None):
    out = []
    for name in sorted(os.listdir(root)):
        ident = split_name(name, ".jsonl")
        if ident and (op is None or ident[0] == op) and (device is None or ident[1] == device):
            out.append((ident, os.path.join(root, name)))
    return out


def write_tables(root, out_dir, op=None, device=None, remeasure=None):
    os.makedirs(out_dir, exist_ok=True)
    found = ledger_files(root, op, device)
    if not found:
        raise SystemExit(f"{root}: no <op>.<device>.jsonl ledger matches op={op} device={device}")
    for (o, d), path in found:
        der = ledger_text(o, d, path)
        for w in der.warnings:
            print(f"warning: {w}", file=sys.stderr)
        dest = os.path.join(out_dir, f"{o}.{d}.txt")
        with open(dest, "w") as f:
            f.write(der.text)
        print(f"wrote {os.path.relpath(dest)}: " + ", ".join(f"{t}={n}" for t, n in der.counts.items()))
        for line in der.remeasure:
            print(f"monotonicity: {line}")
        if remeasure is not None:
            remeasure += der.remeasure


SAMPLE_MIN_CELLS = 64


def rehash_sample(ledger_path, op, run_id, families=None, at=None, repo=REPO, write=True):
    """The sample proof of a rehash (docs/design/tuning-data.md#tuning-data-kernel-hash): run_id is a
    run measured with today's kernels. Every sampled cell is compared with the best record usable
    under the OLD hashes: it agrees when the winners match or the old winner is within TIE of the
    sample's winner in the sample's own times. All agree: one rehash line per stale family is
    appended. Returns (ok, report lines, rehash records)."""
    led = read_ledger(ledger_path)
    heads = {}
    for r in led.runs:
        heads.setdefault(r["dtype"], []).append(r)
    cands = [c for d in heads.values() for c in newest(d).get("candidates", "").split("|") if c]
    block = parse_kernel_block(read_spec_source(op, repo))
    now = family_hashes(block, list(dict.fromkeys(spelling_family(c) for c in cands)), repo)
    sample = [c for c in led.cells if c["run_id"] == run_id]
    old_cells = [c for c in led.cells if c["run_id"] != run_id and c["tier"] not in UNHASHED]
    if not sample:
        return False, [f"run {run_id} has no cells in {ledger_path}"], []
    stale_in_sample = sorted({spelling_family(r["cand"]) for c in sample for r in c["cands"]
                              if r["status"] != "skipped" and r["hash"] and now.get(spelling_family(r["cand"])) != r["hash"]})
    if stale_in_sample:
        return False, [f"run {run_id} was not measured with today's kernels (families {stale_in_sample})"], []
    stored = defaultdict(set)  # family -> stale stored hashes in the old records
    for c in old_cells:
        for r in c["cands"]:
            f = spelling_family(r["cand"])
            if r["hash"] and f in now and r["hash"] != now[f] and (families is None or f in families):
                stored[f].add(r["hash"])
    if not stored:
        return False, ["no stale family to rehash"], []
    ambiguous = {f: sorted(h) for f, h in stored.items() if len(h) > 1}
    if ambiguous:
        return False, [f"several stale hashes per family {ambiguous}: rehash them one at a time"], []
    old_judge = dict(now, **{f: next(iter(h)) for f, h in stored.items()})
    old_best = {(c["dtype"], c["key"]): c for c in
                [b for d in heads for b in best_records([c for c in old_cells if c["dtype"] == d], old_judge).values()]}
    keys = {d: parse_keys(newest(rs)["keys"]) for d, rs in heads.items()}
    def exact(d, key):
        return tuple((n, v) for n, v in key if n in {k for k, is_log, _ in keys[d] if not is_log})
    want = {(d, exact(d, key)) for d, key in old_best}
    got = {(c["dtype"], exact(c["dtype"], c["key"])) for c in sample}
    report = []
    if len(sample) < SAMPLE_MIN_CELLS:
        report.append(f"refused: {len(sample)} sampled cells, fewer than {SAMPLE_MIN_CELLS}")
    missing = sorted(want - got)
    if missing:
        report.append(f"refused: the sample misses {len(missing)} exact-key combinations, e.g. {missing[:3]}")
    if report:
        return False, report, []
    # A disagreeing cell is settled once a deep record measured with today's kernels holds it.
    remeasured = {(c["dtype"], c["key"]) for c in old_cells if c["tier"] == "deep" and usable(c, now)
                  and all(not r["hash"] or now.get(spelling_family(r["cand"])) == r["hash"] for r in c["cands"])}
    agree, settled, bad = 0, 0, []
    for c in sample:
        old = old_best.get((c["dtype"], c["key"]))
        if old is None or not old["ranked"] or not c["ranked"]:
            continue
        times = {r["cand"]: r["median"] for r in c["cands"] if r["status"] in LEDGER_STATUS_TIMED and r["median"]}
        win, ow = c["ranked"][0], old["ranked"][0]
        if win == ow or (ow in times and win in times and times[ow] <= times[win] * (1.0 + TIE)):
            agree += 1
        elif (c["dtype"], c["key"]) in remeasured:
            settled += 1
        else:
            bad.append(f"{op} {c['dtype']} {','.join(f'{n}={v}' for n, v in c['key'])}   # old {ow}, now {win}")
    if bad or agree == 0:
        return False, [f"{len(bad)} of {agree + settled + len(bad)} sampled cells disagree (re-measure them with "
                       "--remeasure-keys at deep):"] + bad, []
    at = at or subprocess.run(["git", "-C", repo, "rev-parse", "--short=8", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "unknown"
    lines = [{"kind": "rehash", "family": f, "from": next(iter(h)), "to": now[f], "at": at,
              "proof": f"sample:{run_id}", "why": f"{agree} cells agree, {settled} re-measured at deep"}
             for f, h in sorted(stored.items())]
    if write:
        with open(ledger_path, "a") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
    return True, [f"{agree} sampled cells agree, {settled} re-measured at deep: rehashed "
                  f"{', '.join(x['family'] for x in lines)}"], lines


def emit_sample(ledger_path, op, cells=SAMPLE_MIN_CELLS):
    """--remeasure-keys lines '<op> <dtype> <key>' for a --rehash-sample run: per dtype, every
    combination of exact keys the measured records hold, each with the cells at the min, median and
    max of every log key; then cells spread evenly over the rest up to `cells`. Only keys of measured
    records that are not all-error (the ones --rehash-sample compares with), so every cell sampled
    is on the tuner's grid and has an old record."""
    led = read_ledger(ledger_path)
    heads = defaultdict(list)
    for r in led.runs:
        heads[r["dtype"]].append(r)
    chosen, rest = [], []
    for d in [t for t in DTYPES if t in heads]:
        logs = [n for n, is_log, _ in parse_keys(newest(heads[d])["keys"]) if is_log]
        keys = sorted({c["key"] for c in led.cells if c["dtype"] == d and c["tier"] not in UNHASHED
                       and not all_error(c)},
                      key=lambda k: tuple(int(v) if n in logs else 0 for n, v in k) + tuple(k))
        groups = defaultdict(list)
        for k in keys:
            groups[tuple((n, v) for n, v in k if n not in logs)].append(k)
        for g in groups.values():
            pick = []
            for name in logs:
                vals = sorted({int(dict(k)[name]) for k in g})
                for v in dict.fromkeys((vals[0], vals[len(vals) // 2], vals[-1])):
                    at = [k for k in g if int(dict(k)[name]) == v]
                    pick.append(at[len(at) // 2])
            pick = list(dict.fromkeys(pick))
            chosen += [(d, k) for k in pick]
            rest += [(d, k) for k in g if k not in pick]
    need = min(max(0, cells - len(chosen)), len(rest))
    chosen += [rest[i * len(rest) // need] for i in range(need)]
    return [f"{op} {d} {','.join(f'{n}={v}' for n, v in k)}" for d, k in chosen]


def git_out(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if p.returncode:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.decode().strip()}")
    return p.stdout


class GitTree:
    """One commit's files, read from git."""

    def __init__(self, repo, rev):
        self.repo, self.rev, self.blobs = repo, rev, {}
        self.paths = set(git_out(repo, "ls-tree", "-r", "-z", "--name-only", rev).decode().split("\0")) - {""}

    def read(self, path):
        if path not in self.paths:
            return None
        if path not in self.blobs:
            self.blobs[path] = git_out(self.repo, "show", f"{self.rev}:{path}")
        return self.blobs[path]


def git_renames(repo, base, head):
    """{base path: head path} for every rename git finds between the two commits."""
    tok, out, i = git_out(repo, "diff", "-M", "--name-status", "-z", base, head).decode().split("\0"), {}, 0
    while i < len(tok) and tok[i]:
        if tok[i][0] in "RC":
            if tok[i][0] == "R":
                out[tok[i + 1]] = tok[i + 2]
            i += 3
        else:
            i += 2
    return out


INCLUDE = re.compile(rb'^([ \t]*#[ \t]*include[ \t]*)(?:"([^"\n]*)"|<([^>\n]*)>)', re.M)
ANY_INCLUDE = re.compile(rb"^[ \t]*(?:#|%:)[ \t]*include", re.M)
# A file a move can retarget without the include text changing: refused, never modelled.
UNMODELLED = re.compile(rb'__has_include|include_next|R"[^(\s"]{0,16}\([^)]*?#[ \t]*include')


def unmodelled_include(data, path):
    """Why `data`'s includes are outside the moved proof's model, or None."""
    if UNMODELLED.search(data):
        return f"{path}: __has_include, #include_next or an include inside a raw string"
    if len(ANY_INCLUDE.findall(data)) != len(INCLUDE.findall(data)):
        return f"{path}: an #include that is neither \"x\" nor <x> (macro, comment or digraph)"
    return None


def canonical_includes(data, path, tree, rename=None):
    """Every include -> the repo path it resolves to, mapped through rename: "x" against its own
    directory, then the repo root, then include/; <x> against the repo root, then include/ (the
    build's -I order). An unresolved one (a system header) is kept verbatim, marked."""
    def sub(m):
        quoted = m.group(2) is not None
        target = (m.group(2) if quoted else m.group(3)).decode()
        roots = ([os.path.dirname(path)] if quoted else []) + ["", "include"]
        for cand in (os.path.normpath(os.path.join(r, target)) for r in roots):
            if cand in tree.paths:
                return m.group(1) + b'"@' + (rename or {}).get(cand, cand).encode() + b'"'
        return m.group(0) + b" ?"
    return INCLUDE.sub(sub, data)


def moved_difference(base_paths, head_paths, base, head, rename):
    """Why base_paths at commit `base` is not head_paths at `head` up to a pure move, or None."""
    if len(base_paths) != len(head_paths):
        return f"source list length {len(base_paths)} -> {len(head_paths)}"
    for b, h in zip(base_paths, head_paths):
        if rename.get(b, b) != h or os.path.basename(b) != os.path.basename(h):
            return f"{b} -> {h} is not a rename"
        old, new = base.read(b), head.read(h)
        if old is None or new is None:
            return f"{b if old is None else h} is missing"
        why = unmodelled_include(old, b) or unmodelled_include(new, h)
        if why:
            return why
        if normalize_source(canonical_includes(old, b, base, rename)) != \
                normalize_source(canonical_includes(new, h, head)):
            return f"{b} -> {h} differs beyond comments, whitespace and include paths"
    return None


def rehash_moved(ledger_path, op, base, at=None, repo=REPO, write=True):
    """The moved proof of a rehash (docs/design/tuning-data.md#tuning-data-kernel-hash): per family
    of the measured records, its source list at `base` maps file by file onto HEAD's under git
    renames, equal modulo comments, whitespace and include paths (each resolved, then renamed). One
    line per stored hash whose effective hash is the family's hash at `base`. Returns (report, records)."""
    renames = {}

    def prove(bp, hp, bt, ht):
        if not renames:
            renames["r"] = git_renames(repo, base, "HEAD")
        why = moved_difference(bp, hp, bt, ht, renames["r"])
        moved = sum(b != h for b, h in zip(bp, hp))
        return why, f"{moved} files moved, equal modulo comments, whitespace and include paths"
    return rehash_by_proof(ledger_path, op, base, "moved", prove, at, repo, write)


CXX_KEYWORDS = frozenset("""alignas alignof asm auto bool break case catch char char8_t char16_t char32_t class
concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete
do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable
namespace new noexcept nullptr operator private protected public register reinterpret_cast requires return short
signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef
typeid typename union unsigned using virtual void volatile wchar_t while""".split())
DIGRAPHS = ("%:%:", "<%", "%>", "<:", ":>", "%:")
CXX_PUNCT = DIGRAPHS + ("<=>", "<<=", ">>=", "...", "->*", "::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=",
                        "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ".*", "##")
ALT_TOKENS = frozenset("and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq".split())
# the tokens after which an identifier ending a template parameter is that parameter's name
NAME_AFTER = frozenset("class typename auto bool char char8_t char16_t char32_t wchar_t short int long signed unsigned "
                       "float double ... * & && > >> )".split())
DECL_FOLLOW = ("}", ";", "::", "~")  # besides an identifier and `[[`, what can follow a function body
NOT_DECL_START = ALT_TOKENS | {"catch", "requires"}
# keywords a macro may not hide at the top level of a template declaration
STRUCTURAL = frozenset("template requires class struct union enum operator try typename concept using typedef "
                       "namespace friend catch".split()) | ALT_TOKENS
INCLUDES = ("include", "include_next", "import", "embed")
CONDITIONALS = {"if": 1, "ifdef": 1, "ifndef": 1, "elif": 0, "elifdef": 0, "elifndef": 0, "else": 0, "endif": -1}
CXX_IDENT = re.compile(r"[A-Za-z_$\x80-\xff][A-Za-z0-9_$\x80-\xff]*")
CXX_NUMBER = re.compile(r"\.?[0-9](?:[eEpP][+-]|'[A-Za-z0-9_]|[A-Za-z0-9_.])*")
CXX_RAW = re.compile(r'"([^ ()\\\t\v\f\n]{0,16})\(')
CXX_MACRO_LIKE = re.compile(r"[A-Z][A-Z0-9_]+")
RESERVED = re.compile(r"__|_[A-Z]")  # __restrict, _Atomic: the implementation's, never a parameter's name
# lowercase names the C, C++ and POSIX libraries or a compiler's predefines may make macros, and
# contextual keywords: a renamed parameter takes or leaves none of them
LIBRARY_MACROS = frozenset("""assert errno offsetof setjmp sigsetjmp va_arg va_copy va_end va_start stdin stdout stderr
unix linux i386 sun major minor makedev alloca htons htonl ntohs ntohl st_atime st_mtime st_ctime timeradd timersub
timerclear timerisset timercmp howmany roundup powerof2 nitems isset isclr setbit clrbit isnan isinf isfinite isnormal
signbit fpclassify isgreater isgreaterequal isless islessequal islessgreater isunordered complex imaginary I getc putc
getchar putchar min max small near far interface final override import module""".split())
# A renamed parameter's names: no macro of the headers the hashed files include has this shape (the
# -dM dump of every system, BLAS/LAPACKE and SYCL header they include, docs/design/tuning-data.md)
NEW_NAME = re.compile(r"[A-Z][a-z][A-Za-z0-9]*")
OLD_NAME = re.compile(r"[A-Z]|[A-Z][a-z][A-Za-z0-9]*")
# the names of that shape some macOS SDK header #defines (grep of the SDK, libc++, clang's, OpenBLAS's
# and AdaptiveCpp's include trees): perl, apache, zlib, ncurses and MacTypes.h, none included here
CAMEL_MACROS = frozenset("""BerkelyFS Byte Bytef CreateLSTMBreakEngine CreateLSTMData CreateLSTMDataForScript
Debugger68k DebugStr68k DeleteLSTMData Divide Egaddr Egcall Egcmp Egmemload Egmemstore End Enddata EndOfStream Entry
ExtBoolname ExtNumname ExtStrname Gaddr Gcall Gcmp Gload Gmemload Gmemstore Gotoff Gotoff2 Gotoff3 Gpop Gpush Gstack
IoFLUSH Length Lgcmp Lgmemload Lgmemstore Lgotoff Lgotoff2 Lgotoff3 MgObjIV MgTypeExt MpDirFLAGS MpDirNONE
MpDirSeenFLAGS MpDirSeenNONE MpDirSeenType MpDirSeenUNSET MpDirType MpDirUNSET MpHandlerANON MpHandlerAUTOLOAD
MpHandlerDYNAMIC MpHandlerFAKE MpHandlerFLAGS MpHandlerMETHOD MpHandlerNONE MpHandlerOBJECT MpHandlerPARSED
MpHandlerType MpInterpBASE MpInterpCLONED MpInterpFLAGS MpInterpNONE MpInterpType MpReqFLAGS MpReqNONE MpReqType
MpSrvACCESS MpSrvAUTHEN MpSrvAUTHZ MpSrvAUTOLOAD MpSrvCLEANUP MpSrvCLONE MpSrvENABLE MpSrvFIXUP MpSrvFLAGS MpSrvLOG
MpSrvNONE MpSrvPARENT MpSrvRESPONSE MpSrvTRANS MpSrvType MpSrvTYPE MpSrvUNSET Newx Newxz NeXTBSD Nil String StrLength
SvCLASS SvObjIV Times Value""".split())
# what spells a function's signature, template parameter names included ("void f() [T = int]"): glibc's
# assert passes __PRETTY_FUNCTION__, a source_location default argument its caller's
SOURCE_LOCATION = frozenset(("source_location", "__builtin_source_location"))
PRETTY_NAMES = SOURCE_LOCATION | frozenset("__PRETTY_FUNCTION__ __FUNCSIG__ __builtin_FUNCSIG assert".split())
# the tokens after which a '[' starts a lambda, never a subscript, `operator[]`, `delete[]` or `[[`
LAMBDA_AFTER = frozenset("( , = { ; ? : ! ~ + - * / % ^ & | && || < <= >= == != <=> += -= *= /= %= &= |= ^= <<= >>= "
                         "<< return throw co_return co_yield co_await and or not".split())
# what a base of only these can name: hidden from a parameter by no other member
INTEGRAL_CONSTANT_BASES = ("true_type", "false_type", "bool_constant", "integral_constant")
INTEGRAL_CONSTANT_MEMBERS = frozenset(("value", "value_type", "type") + INTEGRAL_CONSTANT_BASES)
STD_CLASS_TEMPLATES = frozenset("complex pair tuple array vector integral_constant numeric_limits".split())
STD_TEMPLATES = STD_CLASS_TEMPLATES | frozenset("""bool_constant remove_cv_t remove_cvref_t remove_reference_t
remove_const_t remove_pointer_t add_const_t add_pointer_t decay_t conditional_t enable_if_t common_type_t make_signed_t
make_unsigned_t underlying_type_t invoke_result_t type_identity_t""".split())
# <cmath>'s classification names are functions in C++ ([c.math.fpclass]; libc++ and libstdc++ #undef C's macros)
CMATH_FUNCTIONS = frozenset("""isnan isinf isfinite isnormal signbit fpclassify isgreater isgreaterequal isless
islessequal islessgreater isunordered""".split())
TYPE_KEYWORDS = tuple("auto bool char char8_t char16_t char32_t wchar_t short int long signed unsigned float double "
                      "void".split())
LAMBDA_SPECIFIERS = ("mutable", "constexpr", "consteval", "noexcept", "static")
ATTRIBUTE_CALLS = ("__attribute__", "__declspec", "_Pragma")
PRAGMA_OPS = ("_Pragma", "__pragma")
MEMBER_ACCESS = (".", "->", "::", ".*", "->*")
CLASS_KEYS = ("class", "struct", "union")
REQUIRES_EXPR_AFTER = ("requires", "&&", "||", "!", "(", ",", "=", "and", "or", "not", "return")


def cxx_tokens(data):
    """The C++ tokens of bytes `data`: [(kind, text)], kind id, num, str (string or character literal,
    prefix and suffix included), punct, or pp (one whole directive, its tokens joined by spaces).
    Line splices are joined first; comments and whitespace vanish. A splice inside a raw string,
    which C++ keeps there, raises ValueError."""
    text, parts, splices, last, size = data.decode("latin-1"), [], [], 0, 0
    for m in re.finditer(r"\\\r?\n", text):
        parts.append(text[last:m.start()])
        size, last = size + m.start() - last, m.end()
        splices.append(size)
    s = "".join(parts) + text[last:]
    out, pp, i, n, bol, last = [], None, 0, len(s), True, -1

    def quoted(j):
        q, j = s[j], j + 1
        while j < n and s[j] not in (q, "\n"):
            j += 2 if s[j] == "\\" else 1
        return min(j + 1, n)
    while i < n:
        c, kind = s[i], "punct"
        if c == "\n" or c in " \t\r\f\v":
            if c == "\n":
                if pp is not None:
                    out.append(("pp", " ".join(pp)))
                pp, bol = None, True
            i += 1
            continue
        if s.startswith("//", i) or s.startswith("/*", i):
            j = s.find("\n" if s[i + 1] == "/" else "*/", i + 2)
            i = n if j < 0 else j + (0 if s[i + 1] == "/" else 2)
            continue
        if bol and (c == "#" or s.startswith("%:", i)):
            pp, j = [], i + (1 if c == "#" else 2)
        elif pp is not None and c == "<" and pp[1:] in (["include"], ["include_next"], ["import"]):
            j = s.find(">", i)
            j = n if j < 0 or "\n" in s[i:j] else j + 1
        elif CXX_IDENT.match(s, i):
            j, kind = CXX_IDENT.match(s, i).end(), "id"
            word, raw = s[i:j], CXX_RAW.match(s, j)
            if word in ("R", "LR", "uR", "UR", "u8R") and raw:
                end = s.find(")" + raw.group(1) + '"', raw.end())
                j, kind = n if end < 0 else end + len(raw.group(1)) + 2, "str"
                nxt = bisect.bisect_right(splices, raw.start())
                if nxt < len(splices) and splices[nxt] < j:
                    raise ValueError(f"a line splice inside the raw string at offset {i}")
            elif word in ("L", "u", "U", "u8") and j < n and s[j] in "\"'":
                j, kind = quoted(j), "str"
        elif c.isdigit() or (c == "." and s[i + 1:i + 2].isdigit()):
            j, kind = CXX_NUMBER.match(s, i).end(), "num"
        elif c in "\"'":
            j, kind = quoted(i), "str"
        elif s.startswith("<::", i) and s[i + 3:i + 4] not in (":", ">"):
            j = i + 1  # [lex.pptoken]: '<' then '::', not the digraph '<:'
        else:
            j = i + next((len(p) for p in CXX_PUNCT if s.startswith(p, i)), 1)
        if kind == "str" and CXX_IDENT.match(s, j):
            j = CXX_IDENT.match(s, j).end()
        if pp is None:
            out.append((kind, s[i:j]))
        elif pp[1:2] == ["define"] and len(pp) == 3 and c == "(" and i == last:
            pp[2] += "("  # a function-like macro: its parameters are its own
        else:
            pp.append(s[i:j])
        bol, i, last = False, j, j
    if pp is not None:
        out.append(("pp", " ".join(pp)))
    return out


def code_of(toks):
    """The non-directive tokens of cxx_tokens `toks`: (their indices, texts, kinds)."""
    code = [k for k, x in enumerate(toks) if x[0] != "pp"]
    return code, [toks[k][1] for k in code], [toks[k][0] for k in code]


def close_of(t, q):
    """The index of the bracket closing t[q], one of '(', '[' or '{'; ValueError when unbalanced."""
    depth = []
    for r in range(q, len(t)):
        if t[r] in ("(", "[", "{"):
            depth.append(t[r])
        elif t[r] in (")", "]", "}"):
            if not depth or "([{"[")]}".index(t[r])] != depth.pop():
                raise ValueError(f"unbalanced '{t[r]}' at token {r}")
            if not depth:
                return r
    raise ValueError(f"unclosed '{t[q]}' at token {q}")


def open_of(t, q):
    """The index of the bracket opening t[q], one of ')', ']' or '}'; ValueError when unbalanced."""
    depth = []
    for r in range(q, -1, -1):
        if t[r] in (")", "]", "}"):
            depth.append(t[r])
        elif t[r] in ("(", "[", "{"):
            if not depth or ")]}"["([{".index(t[r])] != depth.pop():
                raise ValueError(f"unbalanced '{t[r]}' at token {r}")
            if not depth:
                return r
    raise ValueError(f"unopened '{t[q]}' at token {q}")


def angle_close(t, q):
    """The index of the '>' closing the '<' at t[q] (brackets skipped); ValueError when there is none."""
    angle, r = 0, q
    while r < len(t):
        if t[r] in ("(", "["):
            r = close_of(t, r)
        elif t[r] in ("<", ">", ">>"):
            angle += 1 if t[r] == "<" else -len(t[r])
            if angle <= 0:
                if angle < 0:
                    raise ValueError(f"'>>' closes the '<' at token {q} and more")
                return r
        elif t[r] in (";", "{", "}", ")", "]"):
            break
        r += 1
    raise ValueError(f"unclosed '<' at token {q}")


def angle_open(t, q):
    """The index of the '<' opening the '>' at t[q] (brackets skipped); ValueError when there is none."""
    angle, r = 0, q
    while r >= 0:
        if t[r] in (")", "]"):
            r = open_of(t, r)
        elif t[r] in ("<", ">", ">>"):
            angle += -1 if t[r] == "<" else len(t[r])
            if angle == 0:
                return r
        elif t[r] in (";", "{", "}", "(", "["):
            break
        r -= 1
    raise ValueError(f"unopened '>' at token {q}")


def template_scopes(t, kinds, std_ok=False, macros=frozenset()):
    """Every template head of the code token texts `t` (code_of) as (first, last, [(token index,
    parameter name or None)], declarator qualifiers): from `template` to the end of the declaration
    it introduces, or, for a template template parameter, to the end of its own list. The declaration
    ends at its `;` at bracket depth 0, after an `=` at its `;`, or at the `}` closing its body (the
    `;` after it included): a class's `{`, a function's first `{` at depth 0, or the `{` after a
    constructor's initializer list, which must read as `name(...)`/`name{...}` items. A `requires {`
    or `requires (...) {` expression is not the body. The qualifiers are the class names a qualified
    declarator names (S in `S<T>::f`), "?" when a parenthesised declarator may hide one (may_be_type;
    `macros` are the names some #define makes macros). Raises ValueError for a head, a declaration or
    a `{` it cannot place: a `{` after `]` or `try`, a class body not followed by `;`, an unreadable
    initializer list."""
    scopes, nested, opener, top = [], set(), {}, namespace_scope(t, kinds)

    def head_at(p):
        return t[p] == "template" and t[p + 1:p + 2] == ["<"] and (p == 0 or t[p - 1] not in MEMBER_ACCESS)

    def param_name(group):
        """The declared name of one parameter: its last token before a default when that is a top-level
        identifier after a type keyword, class key, `typename`, `auto`, `...`, `*`, `&`, `&&`, `>`, `)`
        or another identifier, else None (unnamed, or a form not modelled: `Bar::*`, `const Foo`,
        `void (*F)()`, `int A[3]`)."""
        decl = [e for e in itertools.takewhile(lambda e: not (e[1] and t[e[0]] == "="), group)]
        if len(decl) < 2 or not decl[-1][1]:
            return None
        q, prev = decl[-1][0], t[decl[-2][0]]
        word = kinds[q] == "id" and t[q] not in CXX_KEYWORDS | ALT_TOKENS and not RESERVED.match(t[q])
        if not word or not (prev in NAME_AFTER or kinds[decl[-2][0]] == "id" and prev not in CXX_KEYWORDS and
                            prev not in ALT_TOKENS):
            return None
        return q

    def less_than(q, group, names, angle):
        """May the '<' at t[q], outside brackets in a template head, be a less-than (which would end
        the head later than a compiler does)? After a non-identifier, after a parameter in `names`,
        inside a template argument list (`class T = C<a < b>>`) but after `std::` and one of
        STD_TEMPLATES (when std_ok, std_trusted), or in the default argument of a parameter not written
        `class|typename [...] [name]`. A type-id's own `<` opens a template argument list: in a
        well-formed type default only an argument list holds an expression."""
        std = std_ok and t[q - 1] in STD_TEMPLATES and t[q - 3:q - 1] == ["std", "::"] and t[q - 4:q - 3] != ["::"]
        if angle > 1 and not std or kinds[q - 1] != "id" or t[q - 1] in CXX_KEYWORDS | ALT_TOKENS or t[q - 1] in names:
            return True
        decl = [t[e] for e, _ in itertools.takewhile(lambda e: not (e[1] and t[e[0]] == "="), group)]
        rest = decl[1 + (decl[1:2] == ["..."]):]
        return len(decl) < len(group) and not (decl[:1] in (["class"], ["typename"]) and len(rest) <= 1 and
                                               all(CXX_IDENT.fullmatch(w) for w in rest))

    def parse_list(p, outer=frozenset()):
        params, group, depth, angle, q = [], [], [], 1, p + 2
        outer = outer | {t[n] for f, e, ps, _ in scopes if f < p <= e for n in ps if n is not None}
        while True:
            if q >= len(t):
                raise ValueError(f"unclosed template head at token {p}")
            x = t[q]
            names = outer | {t[n] for g in params for n in [param_name(g)]
                             if n is not None and t[g[0][0]] != "template"}
            if not depth and (x in (";", "operator") or x == "<" and less_than(q, group, names, angle)):
                raise ValueError(f"cannot delimit the template head at token {p}: '{x}' at token {q}")
            if not depth and head_at(q):
                end, inner = parse_list(q, names)
                nested.add(q)
                scopes.append((q, end, inner, set()))
                group += [(r, False) for r in range(q, end + 1)]
                q = end + 1
                continue
            top = not depth and angle == 1
            if x in ("(", "[", "{"):
                depth.append(x)
            elif x in (")", "]", "}"):
                if not depth or "([{"[")]}".index(x)] != depth.pop():
                    raise ValueError(f"unbalanced '{x}' in a template head at token {q}")
            elif not depth and x in ("<", ">", ">>"):
                angle += 1 if x == "<" else -len(x)
                if angle < 0:
                    raise ValueError(f"'>>' closes a template head at token {q}")
                if angle == 0:
                    params.append(group)
                    return q, [param_name(g) for g in params if g]
            elif top and x == ",":
                params.append(group)
                group, q = [], q + 1
                continue
            group.append((q, top))
            q += 1

    def after_heads(q):
        while t[q:q + 2] == ["template", "<"]:
            q = angle_close(t, q + 1) + 1
        return q

    def mem_init_body(q):
        """The index of the body '{' after the constructor initializer list starting at t[q]."""
        while True:
            r = q
            while r < len(t) and (t[r] in ("::", "template") or kinds[r] == "id" and t[r] not in CXX_KEYWORDS):
                r = angle_close(t, r + 1) + 1 if kinds[r] == "id" and t[r + 1:r + 2] == ["<"] else r + 1
            if r == q or t[r:r + 1] not in (["("], ["{"]):
                raise ValueError(f"cannot read a constructor's initializer list at token {min(r, len(t) - 1)}")
            r = close_of(t, r) + 1
            r += t[r:r + 1] == ["..."]
            if t[r:r + 1] == ["{"]:
                return r
            if t[r:r + 1] != [","]:
                raise ValueError(f"cannot read a constructor's initializer list at token {min(r, len(t) - 1)}")
            q = r + 1

    def requires_expr(q, declarator):
        """Does the ')' at t[q] close a requires-expression's parameter list? After the `declarator`
        (a function's parameter list, `->` or `operator`), `&& requires (...) {` is ambiguous."""
        o = opener.get(q, 0)
        if declarator and o >= 2 and t[o - 1] == "requires" and t[o - 2] in ("&&", "and"):
            raise ValueError(f"cannot tell a requires-expression from a ref-qualifier's clause at token {o - 1}")
        return o >= 2 and t[o - 1] == "requires" and t[o - 2] in REQUIRES_EXPR_AFTER

    def is_class_decl(q):
        """Does a class key precede the declaration's first `{`, `;`, `=` or `:` outside brackets?"""
        q = after_heads(q)
        while q < len(t) and t[q] not in ("{", ";", "=", ":"):
            if t[q] in CLASS_KEYS + ("enum",):
                return True
            q = close_of(t, q) + 1 if t[q] in ("(", "[") else q + 1
        return False

    def decl_end(q0):
        is_class = is_class_decl(q0)
        opener.clear()
        depth, init, declarator, q = [], False, False, after_heads(q0)
        while q < len(t):
            x, prev = t[q], t[q - 1]
            if not depth and (x in ("->", "operator") or x == "(" and prev not in REQUIRES_EXPR_AFTER and
                              prev not in ("requires", "decltype", "sizeof", "alignof")):
                declarator = True
            if x in ("(", "["):
                depth.append(q)
            elif x in (")", "]"):
                if not depth or t[depth[-1]] != "(["[")]".index(x)]:
                    raise ValueError(f"unbalanced '{x}' at token {q}")
                opener[q] = depth.pop()
            elif x == "{" and (depth or init or prev == "requires" or
                               prev == ")" and requires_expr(q - 1, declarator)):
                q = close_of(t, q)
            elif x == "{":
                if prev == "]":
                    raise ValueError(f"cannot tell a body from an initializer at token {q}")
                close = close_of(t, q)
                if t[close + 1:close + 2] == [";"]:
                    return close + 1
                if is_class:
                    raise ValueError(f"a class body not followed by ';' at token {close}")
                nxt = close + 1 < len(t) and t[close + 1]
                attr = nxt == "[" and t[close + 2:close + 3] == ["["]
                if nxt and not attr and nxt not in DECL_FOLLOW and (kinds[close + 1] != "id" or
                                                                    nxt in NOT_DECL_START):
                    raise ValueError(f"'{nxt}' after the body at token {q} (a brace in template arguments?)")
                return close
            elif not depth and x == "try":
                raise ValueError(f"cannot delimit a function-try-block at token {q}")
            elif x == "}":
                raise ValueError(f"the enclosing block ends inside a template declaration at token {q}")
            elif not depth and x == ";":
                return q
            elif not depth and x == "=" and prev != "operator":
                init = True
            elif not depth and x == ":" and not init and not is_class:
                return close_of(t, mem_init_body(q + 1))
            q += 1
        raise ValueError("a template declaration runs off the end of the file")

    def quals_before(p):
        """The class names qualifying the declarator name that ends at t[p]."""
        ops = [r for r in range(max(p - 3, 0), p + 1) if t[r] == "operator"]
        if ops:
            p = ops[-1] - 1
        else:
            p = angle_open(t, p) - 1 if t[p] == ">" else p
            if p < 0 or kinds[p] != "id" or t[p] in CXX_KEYWORDS:
                return set()
            p -= 1 + (p > 0 and t[p - 1] in ("~", "compl"))
        p -= p >= 0 and t[p] == "template"
        out = set()
        while p >= 1 and t[p] == "::":
            p -= 1
            if t[p] == ">":
                p = angle_open(t, p) - 1
            if p < 0 or kinds[p] != "id" or t[p] in CXX_KEYWORDS:
                break
            out.add(t[p])
            p -= 1
        return out

    def declarator_quals(q0, end):
        """quals_before at every name the declarator may end in, the qualifier of a top-level
        `::operator` (a conversion to a template-id, `S<T>::operator V<int>()`) included; "?" when a
        parenthesised group holding `::` may be the declarator (`int (S<T>::f)()`, `Int (S<T>::v) = 1`):
        one not after an identifier or `operator` and kin, one followed by `(` or `[`, or one after a
        name that may_be_type, unless `->` follows."""
        out, depth, q = set(), 0, after_heads(q0)
        while q <= end:
            x = t[q]
            if not depth and (x in ("{", ";", "->", ":") or x == "=" and t[q - 1] != "operator"):
                return out | quals_before(q - 1 - (t[q - 1] == "final"))
            if not depth and x == "operator":
                out |= quals_before(q)
            if not depth and x == "(":
                out |= quals_before(q - 1)
                close = close_of(t, q)
                after = t[close + 1:close + 3]
                word = kinds[q - 1] == "id" and t[q - 1] not in CXX_KEYWORDS
                named = word or t[q - 1] in (
                    "operator", "requires", "decltype", "sizeof", "alignof", "noexcept", "alignas", "throw")
                subscript = after[:1] == ["["] and after != ["[", "["]
                if "::" in t[q:close] and (not named or after[:1] == ["("] or subscript or
                                           word and after[:1] != ["->"] and may_be_type(q - 1, q0)):
                    out.add("?")
            depth += (x in ("(", "[")) - (x in (")", "]"))
            q += 1
        return out

    def may_be_type(p, q0):
        """May the name ending at t[p], before a top-level `(`, be the declaration's type, making the
        `(...)` a parenthesised declarator (`Int (S<T>::v) = 1;`)? Not for a constructor or destructor
        (`S<T>::S`, `~S`, unqualified in a class) nor after what ends a type: an identifier that is not
        ALL_CAPS or a `macros` name, a type keyword, `*`, `&`, `&&`, a template-id's `>` or `decltype(...)`,
        each maybe then `const` or `volatile`."""
        r, parts = p, [t[p]]
        if t[p - 1] in ("~", "compl"):
            return False
        try:
            while r >= 2 and t[r - 1] == "::":
                r = angle_open(t, r - 2) - 1 if t[r - 2] == ">" else r - 2
                if r < 0 or kinds[r] != "id" or t[r] in CXX_KEYWORDS:
                    return True
                parts.append(t[r])
        except ValueError:
            return True
        if parts[1:2] == parts[:1] or len(parts) == 1 and not top[q0]:
            return False
        b = r - 1 - (t[r - 1:r] == ["template"])
        while b >= 0 and t[b] in ("const", "volatile"):
            b -= 1
        if b < 0 or b == after_heads(q0) - 1:
            return True
        if kinds[b] == "id" and t[b] not in CXX_KEYWORDS:
            return bool(CXX_MACRO_LIKE.fullmatch(t[b])) or t[b] in macros
        if t[b] == ")":
            return t[open_of(t, b) - 1] != "decltype"
        return t[b] not in TYPE_KEYWORDS + ("*", "&", "&&", ">", ">>")
    for p in range(len(t)):
        if head_at(p) and p not in nested:
            close, params = parse_list(p)
            end = decl_end(close + 1)
            scopes.append((p, end, params, declarator_quals(close + 1, end)))
    return [(a, b, [(q, t[q]) if q is not None else (None, None) for q in ps], qs) for a, b, ps, qs in scopes]


def class_definitions(t, kinds, opaque=frozenset()):
    """Every class, struct or union definition of the code token texts `t` (code_of) as (name, first,
    last, bases): the class key, its closing '}', and each base specifier as a list of token indices.
    An `opaque` macro (opaque_macros) in a class head or base clause raises ValueError."""
    out = []
    for i, x in enumerate(t):
        if x not in CLASS_KEYS or i and t[i - 1] in ("<", ",", "enum"):
            continue
        r, name = i + 1, None
        while r < len(t):
            if t[r] in opaque:
                raise ValueError(f"macro {t[r]} in the head of a class at token {r}")
            if t[r] == "alignas" or t[r] == "[":
                r = close_of(t, r + (t[r] == "alignas")) + 1
            elif t[r] == "::" or kinds[r] == "id" and t[r] not in CXX_KEYWORDS:
                name = t[r] if t[r] not in ("::", "final") else name
                if t[r + 1:r + 2] == ["("] and (CXX_MACRO_LIKE.fullmatch(t[r]) or t[r].startswith("__")):
                    r = close_of(t, r + 1)
                elif t[r + 1:r + 2] == ["<"] and t[r] != "::":
                    r = angle_close(t, r + 1)
                r += 1
            else:
                break
        if name is None or t[r:r + 1] not in (["{"], [":"]):
            continue
        bases, group, angle = [], [], 0
        if t[r] == ":":
            r += 1
            while r < len(t) and not (t[r] == "{" and angle == 0):
                if t[r] in ("(", "["):
                    group += range(r, close_of(t, r) + 1)
                    r = close_of(t, r) + 1
                    continue
                if t[r] in (";", "}", ")", "]", "{") or t[r] in opaque:
                    raise ValueError(f"cannot read the base clause of class {name} at token {r}")
                angle += (t[r] == "<") - (t[r] == ">") - 2 * (t[r] == ">>")
                if t[r] == "," and angle == 0:
                    bases, group = bases + [group], []
                else:
                    group.append(r)
                r += 1
            bases.append(group)
        out.append((name, i, close_of(t, r), bases))
    return out


def dependent_base(t, base, bound):
    """Is the base specifier `base` (token indices into the code token texts `t`) dependent as
    modelled: after access specifiers, `virtual` and attributes, a parameter in `bound` alone or before
    `::`, `<` or `...`? Any other base (an alias template's `Al<T>`, `decltype(T())`, `::T`) counts as
    non-dependent, so its members may hide a parameter."""
    xs = [t[q] for q in base]
    while xs and (xs[0] in ("public", "protected", "private", "virtual") or xs[:2] == ["[", "["]):
        xs = xs[close_of(xs, 0) + 1 if xs[0] == "[" else 1:]
    return bool(xs) and xs[0] in bound and xs[1:2] in ([], ["::"], ["<"], ["..."])


def namespace_scope(t, kinds):
    """For each code token text of `t` (code_of): is every bracket enclosing it a namespace's `{`?"""
    out, stack, pending = [], [], None
    for q, x in enumerate(t):
        if x == "namespace" and (not q or t[q - 1] != "using"):
            r = q + 1
            while r < len(t) and (t[r] in ("::", "inline") or kinds[r] == "id" and t[r] not in CXX_KEYWORDS):
                r += 1
            pending = r if t[r:r + 1] == ["{"] else pending
        out.append(all(stack))
        if x in ("(", "[", "{"):
            stack.append(x == "{" and pending == q)
        elif x in (")", "]", "}") and stack:
            stack.pop()
    return out


def std_trusted(t, kinds):
    """Does `std` name only the global namespace std: is every `std` a `std::` (in a namespace
    definition's name only first, at the top level), a top-level `namespace std {` or a `using
    namespace std;`? Any other `std` (`struct std`, `typedef X std;`, `namespace a::std`) may declare one.
    Nor may the code specialize one of INTEGRAL_CONSTANT_BASES (inside `namespace std`, or `struct
    std::integral_constant<...>`), which would give it members of its own."""
    paths = namespace_paths(t, kinds)

    def chain_start(q):
        while q >= 2 and t[q - 1] in ("::", "inline") and (t[q - 1] == "inline" or kinds[q - 2] == "id"):
            q -= 1 if t[q - 1] == "inline" else 2
        return q
    if paths is not None and any(x in INTEGRAL_CONSTANT_BASES and (paths[q][:1] == ("std",) or t[q - 2:q] == [
            "std", "::"] and t[max(q - 3 - (t[q - 3:q - 2] == ["::"]), 0)] in CLASS_KEYS) for q, x in enumerate(t)):
        return False
    return paths is not None and all(
        x != "std" or t[q + 1:q + 2] == ["::"] and (t[chain_start(q) - 1:chain_start(q)] != ["namespace"] or
                                                   chain_start(q) == q and paths[q] == ()) or
        t[q - 1:q] == ["namespace"] and t[q + 1:q + 2] == ["{"] and paths[q] == () or
        t[q - 2:q] == ["using", "namespace"] and t[q + 1:q + 2] == [";"] for q, x in enumerate(t))


def class_templates(t, kinds, scopes):
    """{(namespace path, name): first token} of every class template whose head (template_scopes) sits
    at namespace scope: `template <...> [requires ...] struct N` and partial or explicit specializations."""
    out, paths, top = {}, namespace_paths(t, kinds), namespace_scope(t, kinds)
    for first, _, _, _ in scopes or []:
        if paths is None or not top[first] or t[first - 1:first] in (["<"], [","]):  # a template template parameter
            continue
        q = angle_close(t, first + 1) + 1
        if t[q:q + 1] == ["requires"]:
            while q < len(t) and t[q] not in CLASS_KEYS + (";", "{", "="):
                q = close_of(t, q) + 1 if t[q] in ("(", "[") else q + 1
        if t[q:q + 1] and t[q] in CLASS_KEYS and q + 1 < len(t) and kinds[q + 1] == "id" and \
                t[q + 1] not in CXX_KEYWORDS and t[q + 2:q + 3] in (["{"], [";"], [":"], ["<"], ["final"]):
            out.setdefault((paths[first], t[q + 1]), first)
    return out


def namespace_names(t, kinds):
    """{(namespace path, name)} of every identifier of the code token texts `t` not followed by `<` (what
    may declare a name there other than as a template-id's use) and "using namespace" where a
    using-directive is; None without readable namespaces."""
    paths = namespace_paths(t, kinds)
    return None if paths is None else {(paths[q], "using namespace" if t[q:q + 2] == ["using", "namespace"] else x)
                                       for q, x in enumerate(t) if kinds[q] == "id" and t[q + 1:q + 2] != ["<"]}


def base_hides(t, kinds, base, bound, here, known, std_ok, elsewhere=None):
    """The names the base specifier `base` (token indices into `t`) can hide from a template parameter
    of the class at namespace path `here` (None: any name). Nothing for a dependent_base or a template-id
    of a class template that `known` ({(path, name): first token}, class_templates of earlier code at
    namespace scope) or std holds with a dependent argument (a parameter in `bound`, or such a
    template-id); INTEGRAL_CONSTANT_MEMBERS for std::true_type, std::bool_constant<...> and kin. An
    unqualified name finds a template of an enclosing namespace when no namespace inside that one on the
    way to here (nor one nested in it: inline and unnamed ones) holds the name or a using-directive in
    `elsewhere` (namespace_names of every scanned file; None: never); `q::N` names here's q::N."""
    if dependent_base(t, base, bound):
        return frozenset()
    xs = [t[q] for q in base]
    while xs and (xs[0] in ("public", "protected", "private", "virtual") or xs[:2] == ["[", "["]):
        xs = xs[close_of(xs, 0) + 1 if xs[0] == "[" else 1:]
    xs = [w for x in xs[xs[:1] == ["::"]:] for w in ([">", ">"] if x == ">>" else [x])]
    if std_ok and xs[:2] == ["std", "::"] and (xs[2:] in (["true_type"], ["false_type"]) or xs[2:3] in (
            ["bool_constant"], ["integral_constant"]) and xs[3:4] == ["<"] and xs[-1] == ">"):
        return INTEGRAL_CONSTANT_MEMBERS

    def dependent(ys):
        if ys in ([p] for p in bound) or len(ys) == 2 and ys[0] in bound and ys[1] == "...":
            return True
        if "<" not in ys or ys[-1] != ">":
            return False
        o = ys.index("<")
        quals, name = ys[:o - 1:2], ys[o - 1]
        if ys[1:o - 1:2] != ["::"] * (o // 2) or any(kinds_of.get(w) != "id" for w in quals + [name]):
            return False
        lib = std_ok and quals == ["std"] and name in STD_CLASS_TEMPLATES
        if not lib and not visible(quals, name):
            return False
        args, group, depth = [], [], 0
        for w in ys[o + 1:-1]:
            depth += (w in ("(", "[", "{", "<")) - (w in (")", "]", "}", ">")) - 2 * (w == ">>")
            if w == "," and depth == 0:
                args, group = args + [group], []
            else:
                group.append(w)
        return any(dependent(a) for a in args + [group])

    def visible(quals, name):
        if here is None or quals:
            return here is not None and (here + tuple(quals), name) in known
        at = next((i for i in range(len(here), -1, -1) if (here[:i], name) in known), None)
        return at == len(here) or at is not None and elsewhere is not None and not any(
            w in (name, "using namespace") and len(p) > at and p[:at + 1] == here[:at + 1] for p, w in elsewhere)
    kinds_of = {t[q]: kinds[q] for q in base}
    return frozenset() if dependent(xs) else None


def namespace_paths(t, kinds):
    """The names of the namespaces enclosing each code token text of `t` (code_of), "" for an unnamed
    one, or None when some namespace definition is not `namespace a::inline b {` or an alias."""
    out, stack, path, pending = [], [], (), None
    for q, x in enumerate(t):
        if x == "namespace" and (not q or t[q - 1] != "using"):
            r, names = q + 1, []
            while r < len(t) and (t[r] in ("::", "inline") or kinds[r] == "id" and t[r] not in CXX_KEYWORDS):
                names += [] if t[r] in ("::", "inline") else [t[r]]
                r += 1
            if t[r:r + 1] == ["{"]:
                pending = (r, tuple(names) or ("",))
            elif t[r:r + 1] != ["="] or not names:
                return None
        if x == "{":
            stack.append(path)
            path += pending[1] if pending and pending[0] == q else ()
        elif x == "}":
            if not stack:
                return None
            path = stack.pop()
        out.append(path)
    return out


def included(toks, first, last):
    """The first #include, #import or #embed among cxx_tokens toks[first:last + 1], else None."""
    return next((k for k in range(first, last + 1) if toks[k][0] == "pp" and
                 " ".join(toks[k][1].split(" ")[1:2]) in INCLUDES), None)


def ids_in(toks, first, last):
    """The identifiers of cxx_tokens toks[first:last + 1], directives' included."""
    return ({x for k, x in toks[first:last + 1] if k == "id"} |
            {w for k, x in toks[first:last + 1] if k == "pp" for w in CXX_IDENT.findall(x)})


def class_table(toks, opaque=frozenset(), std_ok=False, elsewhere=None):
    """The class definitions of cxx_tokens `toks` as {name: [(identifiers it uses but the parameters of
    the templates enclosing it, the names its bases can hide (class_hiding), its namespace_paths
    path)]}; without readable template scopes every base hides any name, without readable namespaces
    the path is None.
    A conditional directive reaching a class's top level or an #include inside one raises ValueError."""
    code, t, kinds = code_of(toks)
    defs, groups = class_definitions(t, kinds, opaque), conditional_groups(toks)[1]
    try:
        scopes = template_scopes(t, kinds, std_ok)
    except ValueError:
        scopes = None
    out, paths = defaultdict(list), namespace_paths(t, kinds)
    level = list(itertools.accumulate(((x in ("(", "[", "{")) - (x in (")", "]", "}")) for x in t), initial=0))
    for (name, first, last, _), hides in zip(defs, class_hiding(t, kinds, defs, scopes, std_ok, elsewhere)):
        if straddling(groups, code[first], code[last], level[first]) is not None:
            raise ValueError(f"a conditional directive reaches the top level of class {name}")
        if included(toks, code[first], code[last]) is not None:
            raise ValueError(f"an #include inside class {name}")
        bound = {x for f, l, ps, _ in scopes or [] if f <= first <= l for _, x in ps if x}
        out[name].append((ids_in(toks, code[first], code[last]) - bound, hides, paths and paths[first]))
    return out


def class_hiding(t, kinds, defs, scopes, std_ok, elsewhere=None):
    """Per class_definitions entry of `defs`, the names its bases can hide from a template parameter
    (base_hides; None: any name, so too without readable template scopes)."""
    if scopes is None:
        return [None if bases else frozenset() for _, _, _, bases in defs]
    try:
        known = class_templates(t, kinds, scopes)
    except ValueError:
        known = {}
    paths, top, out = namespace_paths(t, kinds), namespace_scope(t, kinds), []
    for _, first, _, bases in defs:
        bound = {x for f, l, ps, _ in scopes if f <= first <= l for _, x in ps if x}
        here = paths[first] if paths is not None and top[first] else None
        earlier = {k for k, at in known.items() if at < first}
        hides = [base_hides(t, kinds, base, bound, here, earlier, std_ok, elsewhere) for base in bases]
        out.append(None if None in hides else frozenset().union(*hides))
    return out


def macro_names(toks):
    """The identifiers that #define and #undef directives of cxx_tokens `toks` mention (a function-like
    macro's own parameters aside), the macro names they define and each #define's (parameters, body
    words), as (mentioned, defined, bodies, directives, attrs): directives lists each #define and #undef
    as (token index, define or undef, name, the identifiers of its parameters and body, "##" when it
    pastes), each #include as (token index, "include", None, set()), and each `#pragma pop_macro` and
    `_Pragma`/`__pragma` (in code or a body) that may pop one as (token index, "pop", None, the identifiers of its
    literal, or {"?"} for an operand that is not one string literal); attrs maps each defined name to
    whether each #define is object-like and attribute_only. A macro body with unbalanced brackets raises
    ValueError."""
    mentioned, defined, bodies, directives, attrs = set(), set(), defaultdict(list), [], defaultdict(list)
    for k, (kind, text) in enumerate(toks):
        words = text.split(" ") if kind == "pp" else [x for _, x in toks[k:k + 4]] if text in PRAGMA_OPS else [text]
        if kind == "pp" and words[1:2] and words[1] in INCLUDES:
            directives.append((k, "include", None, set()))
        for q in [q for q, w in enumerate(words) if w in PRAGMA_OPS and (kind == "pp" or q == 0)]:
            lit = words[q + 1:q + 2] == ["("] and words[q + 3:q + 4] == [")"] and '"' in "".join(words[q + 2:q + 3])
            if not lit or "pop_macro" in words[q + 2]:
                directives.append((k, "pop", None, set(CXX_IDENT.findall(words[q + 2])) if lit else {"?"}))
        if kind == "pp" and words[1:2] == ["pragma"] and "pop_macro" in text:
            lit = words[2:] == ["pop_macro", "(", words[4] if words[4:] else "", ")"] and '"' in words[4]
            directives.append((k, "pop", None, set(CXX_IDENT.findall(words[4])) if lit else {"?"}))
        if kind != "pp" or words[1:2] not in (["define"], ["undef"]) or len(words) < 3:
            continue
        fn = words[2].endswith("(") and ")" in words
        own = words[3:words.index(")")] if fn else []
        body = words[words.index(")") + 1 if fn else 3:]
        brackets = [w for w in body if w in ("(", ")", "[", "]", "{", "}")]
        try:
            nested = close_of(["("] + brackets + [")"], 0) == len(brackets) + 1
        except ValueError:
            nested = False
        if not nested:  # `} int k() {` balances by count, not by nesting
            raise ValueError(f"#define {words[2].rstrip('(')} has unbalanced brackets")
        defined.add(words[2].rstrip("("))
        mentioned.update(set(CXX_IDENT.findall(text)) - set(own))
        directives.append((k, words[1], words[2].rstrip("("), set(CXX_IDENT.findall(" ".join(words[3:]))) |
                           ({"##"} if "##" in words[3:] else set())))
        if words[1] == "define":
            own = set(own) | ({"__VA_ARGS__", "__VA_OPT__"} if "..." in own else set())
            bodies[words[2].rstrip("(")].append((own, body))
            attrs[words[2].rstrip("(")].append(not words[2].endswith("(") and attribute_only(body))
    return mentioned, defined, bodies, directives, attrs


def attribute_only(body):
    """Is the macro body `body` (words) only attributes (`__attribute__((...))`, `__declspec(...)`,
    `[[...]]`) and LAMBDA_SPECIFIERS, nothing a lambda's `-> T` could follow as an operand?"""
    q = 0
    while q < len(body):
        if body[q] in ("__attribute__", "__declspec") and body[q + 1:q + 2] == ["("]:
            q = close_of(body, q + 1) + 1
        elif body[q:q + 2] == ["[", "["]:
            q = close_of(body, q) + 1
        elif body[q] in LAMBDA_SPECIFIERS:
            q += 1
        else:
            return False
    return True


def opaque_macros(bodies):
    """The macros of `bodies` (macro_names) whose expansion can change where a declaration ends: a body
    whose top level, outside its own (...) and [...], holds anything but identifiers, literals, `*`
    and `&`, or holds a structural keyword, a parameter of its own or another such macro."""
    opaque = set()
    while True:
        more = {name for name, defs in bodies.items() if name not in opaque and any(
            not inert(own, body, opaque) for own, body in defs)}
        if not more:
            return opaque
        opaque |= more


def inert(own, body, opaque):
    """Is the macro body `body` (parameters `own`) one opaque_macros lets through?"""
    depth = 0
    for w in body:
        depth += (w in ("(", "[")) - (w in (")", "]"))
        if depth or w in (")", "]"):
            continue
        word = CXX_IDENT.fullmatch(w)
        if not (word and w not in STRUCTURAL and w not in own and w not in opaque or w in ("*", "&", "&&") or
                w[0].isdigit() or "\"" in w or "'" in w):
            return False
    return True


def conditional_groups(toks):
    """The conditional directives of cxx_tokens `toks` as (why not modelled or None, [(lowest bracket
    depth, [directive token indices])]): one entry per #if..#endif, the lowest depth any of its
    branches reaches and where its #if, #elif, #else and #endif are. Each branch must leave the depth
    it found, no token may be a digraph (<% %> <: :> %: %:%:, a directive's leading %: aside) and no
    #include may sit inside brackets, where the parser cannot see what it adds to a class or namespace."""
    depth, open_, groups = 0, [], []
    for k, (kind, text) in enumerate(toks):
        words = text.split(" ")
        if kind == "pp" and any(w in DIGRAPHS for w in words[1:]) or kind != "pp" and text in DIGRAPHS:
            return f"a digraph at token {k}", groups
        if depth and included(toks, k, k) is not None:
            return f"an #include inside brackets at token {k}", groups
        step = CONDITIONALS.get(words[1] if kind == "pp" and len(words) > 1 else None)
        if step == 1:
            open_.append([[k], depth, depth])
        elif step is not None and (not open_ or open_[-1][1] != depth):
            return f"the brackets of the #if branch ending at token {k} do not balance", groups
        elif step is not None:
            open_[-1][0].append(k)
        if step == -1:
            at, _, low = open_.pop()
            groups.append((low, at))
            if open_:
                open_[-1][2] = min(open_[-1][2], low)
        depth += (kind != "pp" and text in ("(", "[", "{")) - (kind != "pp" and text in (")", "]", "}"))
        if open_:
            open_[-1][2] = min(open_[-1][2], depth)
    return ("an unterminated #if" if open_ else None), groups


def straddling(groups, first, last, depth):
    """The first directive inside tokens first..last of a conditional group (conditional_groups) that
    reaches `depth`: a group wrapping the whole range, or kept below `depth`, is harmless."""
    return next((k for low, at in groups if low <= depth for k in at if first < k < last), None)


def macro_problem(names, k0, k1, here, directives, toks):
    """Why a macro of the hashed files may change what one of `names` means in cxx_tokens toks[k0:k1 + 1]
    of the file `here`, else None; `directives` maps each file to its macro_names directives. A macro
    named like one must be #undef'd later in its own file with no #include between, and not be in
    effect in the range (an #undef needs an earlier #define in its file), and no pop_macro anywhere may
    restore one (nor any whose operand is computed); a macro whose parameters or body name one, a
    PRETTY_NAMES name or such a macro, or that pastes with ##, is neither defined nor expanded in the
    range."""
    for f, dirs in directives.items():
        for i, (k, what, name, ids) in enumerate(dirs):
            if what == "pop" and ids & (names | {"?"}):
                return f"a pop_macro in {f} may restore a macro named {' or '.join(sorted(names))}"
            if name not in names:
                continue
            if what == "undef":
                if not any(w == "define" and n == name for _, w, n, _ in dirs[:i]):
                    return f"#undef {name} in {f} without an earlier #define there"
                continue
            end = next((e for e, w, n, _ in dirs[i + 1:] if w == "undef" and n == name), None)
            if end is None or any(w == "include" and e < end for e, w, _, _ in dirs[i + 1:]):
                return f"#define {name} in {f} may stay in effect beyond that file"
            if f == here and k < k1 and end > k0:
                return f"#define {name} is in effect inside the scope"
    uses = defaultdict(set)
    for dirs in directives.values():
        for _, what, name, ids in dirs:
            uses[name] |= ids if what == "define" else set()
    bad = {n for n, ids in uses.items() if ids & (names | PRETTY_NAMES) or "##" in ids}
    while bad:
        more = {n for n, ids in uses.items() if n not in bad and ids & bad}
        if not more:
            break
        bad |= more
    defs = {k: n for k, w, n, _ in directives.get(here, []) if w == "define"}
    hit = next((toks[k][1] if k not in defs else defs[k] for k in range(k0, k1 + 1) if
                toks[k][0] == "id" and toks[k][1] in bad or defs.get(k) in bad), None)
    return None if hit is None else (f"macro {hit}, which may name {' or '.join(sorted(names))}, paste a name or "
                                     f"spell the signature, inside the scope")


def renamed_tokens(a, b, directives=None, here="", defined=frozenset(), classes=None, opaque=frozenset(),
                   std_ok=False, attr_macros=frozenset(), elsewhere=None):
    """Why cxx_tokens `b` is not `a` up to template-parameter renames, else None, as (why, renames).
    Within each template's scope (template_scopes) a parameter may be renamed by a bijection used at
    every reference in that scope (a token of the head before the parameter is none), to a name absent
    from the old scope, from the parameters of every enclosing template and, for an out-of-line member,
    from the class's definition in the same namespace (in `classes`, class_table; the default is a's
    own), whose old name is absent from the new scope and from that definition; a member access (x.T,
    B::T) is not a reference, nor is a lambda's trailing return type after LAMBDA_SPECIFIERS,
    attributes or `attr_macros`. Every other token must be equal. Each new name is NEW_NAME and each
    old one OLD_NAME, neither in LIBRARY_MACROS or CAMEL_MACROS, nor in the head before the
    declaration, nor met by a
    macro of `directives` ({file: macro_names directives}, the default a's own as file `here`;
    macro_problem); neither may sit inside an attribute or a call of a `defined`, ALL_CAPS or library
    macro, nor be referenced inside a class whose bases may hide it (class_hiding). A renaming scope
    may not hold an #include or, at its top level (outside brackets), an `opaque` macro
    (opaque_macros); no conditional group may reach the top level of a renamed scope or of a class
    from inside it (straddling), nor may a renaming scope name one of PRETTY_NAMES."""
    if len(a) != len(b):
        return f"{len(a)} -> {len(b)} tokens", 0
    if any((x[0] == "pp" or y[0] == "pp") and x != y for x, y in zip(a, b)):
        return "a directive differs", 0
    code, t, kinds = code_of(a)
    _, tb, kb = code_of(b)
    try:
        directives = {here: macro_names(a)[3]} if directives is None else directives
        sa, sb = template_scopes(t, kinds, std_ok, defined), template_scopes(tb, kb, std_ok, defined)
        local = class_definitions(t, kinds, opaque)
        classes = class_table(a, opaque, std_ok, elsewhere) if classes is None else classes
        hides = class_hiding(t, kinds, local, sa, std_ok, elsewhere)
    except ValueError as e:
        return f"cannot scope the templates: {e}", 0
    _, groups = conditional_groups(a)
    level = list(itertools.accumulate(((x in ("(", "[", "{")) - (x in (")", "]", "}")) for x in t), initial=0))
    for name, f, l, _ in local:
        if straddling(groups, code[f], code[l], level[f]) is not None:
            return f"cannot scope the templates: an #if reaches the top level of class {name}", 0
        if included(a, code[f], code[l]) is not None:
            return f"cannot scope the templates: an #include inside class {name}", 0
    if [(f, e, [q for q, _ in ps]) for f, e, ps, _ in sa] != [(f, e, [q for q, _ in ps]) for f, e, ps, _ in sb]:
        return "the template structure differs", 0
    hiding = [(name, f, l, h) for (name, f, l, _), h in zip(local, hides) if h != frozenset()]
    paths = namespace_paths(t, kinds)
    maps = []
    for s, ((first, last, pa, quals), (_, _, pb, _)) in enumerate(zip(sa, sb)):
        m = {x: y for (_, x), (_, y) in zip(pa, pb) if x != y}
        olds, news = ids_in(a, code[first], code[last]), ids_in(b, code[first], code[last])
        where = f"template at token {code[first]}"
        if len(set(m.values())) != len(m):
            return f"{where}: {m} is not a bijection", 0
        if m and olds & PRETTY_NAMES:
            return f"{where}: {min(olds & PRETTY_NAMES)} spells the renamed parameters", 0
        cond = straddling(groups, code[first], code[last], level[first]) if m else None
        if cond is not None:
            return f"{where}: {a[cond][1][:20]} reaches the top level of the scope of a renamed parameter", 0
        for k in range(code[first], code[last]) if m else ():
            words, top = a[k][1].split(" "), level[bisect.bisect_left(code, k)] == level[first]
            if a[k][0] == "pp" and words[1:2] and words[1] in INCLUDES:
                return f"{where}: {' '.join(words[:2])} inside the scope of a renamed parameter", 0
            if a[k][0] != "pp" and top and a[k][1] in opaque:
                return f"{where}: macro {a[k][1]} at the top level of the scope of a renamed parameter", 0
        outer = [e for e in range(len(sa)) if e != s and sa[e][0] <= first and last <= sa[e][1]]
        outer_params = {x for e in outer for _, x in sa[e][2] + sb[e][2] if x}
        quals = quals.union(*(sa[e][3] for e in outer))
        for (at, x), (_, y) in zip(pa, pb):
            if x == y:
                continue
            if y in olds or x in news:
                return f"{where}: {x} -> {y}, but " + (
                    f"{y} occurs in the old scope" if y in olds else f"{x} occurs in the new scope"), 0
            if y in outer_params:
                return f"{where}: {x} -> {y}, but an enclosing template has a parameter {y}", 0
            if x in t[first:at] or y in tb[first:at]:
                return f"{where}: {x} -> {y}, but {x if x in t[first:at] else y} occurs in the head before {x}", 0
            odd = next((z for z, shape in ((x, OLD_NAME), (y, NEW_NAME)) if not shape.fullmatch(z) or
                        z in LIBRARY_MACROS | CAMEL_MACROS), None)
            if odd:
                return (f"{where}: {x} -> {y}, but {odd} is not {'one capital or ' if odd == x else ''}[A-Z][a-z]"
                        f"[A-Za-z0-9]* or is a library macro's name: a header outside the hashed files may "
                        f"define it"), 0
            why = macro_problem({x, y}, code[first], code[last], here, directives, a)
            if why:
                return f"{where}: {x} -> {y}, but {why}", 0
            for name, f, l, h in hiding:
                if (h is None or x in h or y in h) and any(
                        t[q] == x and kinds[q] == "id" for q in range(max(f, first), min(l, last) + 1)):
                    return (f"{where}: {x} -> {y}, but class {name} has a base whose members may hide {x} "
                            f"or {y}"), 0
            if "?" in quals:
                return f"{where}: {x} -> {y}, but a parenthesised declarator may name a class member", 0
            for q in sorted(quals):
                defs, ns = classes.get(q, []), paths and paths[first]
                if None in classes or ns is None or not any(path == ns for _, _, path in defs) or any(
                        x in used or y in used or h is None or x in h or y in h for used, h, _ in defs):
                    return (f"{where}: {x} -> {y} in an out-of-line member of {q}, whose definition is not in the "
                            f"hashed files in this namespace, names {x} or {y} or has a base that may hide "
                            f"them"), 0
        maps.append(m)
    bound = {x for _, _, ps, _ in sa for _, x in ps if x}
    declared = [{x: at for at, x in ps if x} for _, _, ps, _ in sa]
    open_, opener, macro_depth, in_attr = [], {}, 0, set()
    try:
        for j in range(len(t)):
            if j not in in_attr and (t[j:j + 2] == ["[", "["] or t[j] in ATTRIBUTE_CALLS and t[j + 1:j + 2] == ["("]):
                in_attr.update(range(j, close_of(t, j + (t[j] != "[")) + 1))
    except ValueError as e:
        return f"cannot scope the templates: {e}", 0

    def member_access(j):
        """Is t[j] named by `.`, `->`, `::` (or those + `template`)? `) -> T` of a lambda is not: a
        `[...](...)` whose `[` is not `[[` and starts the code or follows LAMBDA_AFTER, then
        LAMBDA_SPECIFIERS, `[[...]]`, `__attribute__((...))` and `attr_macros` (each #define of them
        attribute_only)."""
        p = j - 1 - (t[j - 1:j] == ["template"])
        if p < 0 or t[p] not in MEMBER_ACCESS:
            return False
        q = p - 1
        while q >= 0:
            if t[q] in LAMBDA_SPECIFIERS or t[q] in attr_macros:
                q -= 1
            elif t[q] == "]" and q in opener and t[q - 1:q] == ["]"] and t[opener[q] + 1:opener[q] + 2] == ["["]:
                q = opener[q] - 1
            elif t[q] == ")" and q in opener and t[opener[q] - 1:opener[q]] in (["__attribute__"], ["__declspec"]):
                q = opener[q] - 2
            else:
                break
        if t[p] != "->" or q < 0 or q not in opener or t[opener[q] - 1:opener[q]] != ["]"]:
            return True
        lb = opener.get(opener[q] - 1, -1)
        if lb < 0 or t[lb] != "[":
            return True
        return t[lb + 1] == "[" or lb > 0 and t[lb - 1] not in LAMBDA_AFTER
    for j, text in enumerate(t):
        kind, k, prev = kinds[j], code[j], t[j - 1] if j else ""
        if text in ("(", "["):
            like = text == "(" and j > 0 and kinds[j - 1] == "id" and prev not in bound and bool(
                CXX_MACRO_LIKE.fullmatch(prev) or prev in defined or
                prev in LIBRARY_MACROS - CMATH_FUNCTIONS and t[j - 2:j - 1] != ["::"])
            open_.append((j, like))
            macro_depth += like
        elif text in (")", "]") and open_:
            opener[j], like = open_.pop()
            macro_depth -= like
        member = member_access(j)
        binder = None
        if kind == "id" and text in bound and not member:
            binder = max((s for s in range(len(sa)) if sa[s][0] <= j <= sa[s][1] and
                          declared[s].get(text, j + 1) <= j), key=lambda s: sa[s][0], default=None)
        want = text if binder is None else maps[binder].get(text, text)
        if (kb[j], tb[j]) != (kind, want):
            if kind != "id" or kb[j] != "id":
                return f"token {k}: {text} -> {tb[j]}", 0
            if binder is not None:
                return f"token {k}: {text} -> {tb[j]}, an inconsistent rename (expected {want})", 0
            return f"token {k}: {text} -> {tb[j]} is not a template parameter" + (
                " (a member access)" if member else ""), 0
        if want != text and macro_depth:
            return f"token {k}: {text} -> {want} inside a macro call", 0
        if want != text and j in in_attr:  # [[no_unique_address]] names an attribute, never a parameter
            return f"token {k}: {text} -> {want} inside an attribute", 0
    return None, sum(len(m) for m in maps)


def alpha_problem(files):
    """Why some (path, base bytes, head bytes) of `files` is not its base up to template-parameter
    renames (renamed_tokens, with the macros and classes of every base and head file), else None, as
    (why, renames, files renamed). std names the global namespace only when no file says otherwise
    (std_trusted); an attribute macro is ALL_CAPS with every #define of it attribute_only."""
    try:
        toks = {(p, w): cxx_tokens(data) for p, old, new in files for w, data in (("b", old), ("h", new))}
        named = {key: macro_names(x) for key, x in toks.items()}
    except ValueError as e:
        return f"cannot read the sources: {e}", 0, 0
    odd = next(((p, why) for (p, _), x in toks.items() for why in [conditional_groups(x)[0]] if why), None)
    if odd:
        return f"{odd[0]}: {odd[1]}", 0, 0
    defined = set().union(*(x[1] for x in named.values()))
    bodies, attrs = defaultdict(list), defaultdict(list)
    for _, _, b, _, at in named.values():
        for name, defs in b.items():
            bodies[name] += defs
        for name, flags in at.items():
            attrs[name] += flags
    opaque = opaque_macros(bodies)
    attr_macros = {name for name, flags in attrs.items() if CXX_MACRO_LIKE.fullmatch(name) and all(flags)}
    directives = {p: named[(p, "b")][3] for p, _, _ in files}
    std_ok = all(std_trusted(*code_of(x)[1:]) for x in toks.values())
    names = [namespace_names(*code_of(x)[1:]) for x in toks.values()]
    elsewhere = None if None in names else set().union(*names)
    classes = defaultdict(list)
    for p, _, _ in files:
        try:
            for name, defs in class_table(toks[(p, "b")], opaque, std_ok, elsewhere).items():
                classes[name] += defs
        except ValueError:
            classes[None] = [(set(), None, None)]
    renames = files_renamed = 0
    for p, old, new in files:
        if normalize_source(old) == normalize_source(new):
            continue
        why, n = renamed_tokens(toks[(p, "b")], toks[(p, "h")], directives, p, defined, classes, opaque, std_ok,
                                attr_macros, elsewhere)
        if why:
            return f"{p}: {why}", 0, 0
        renames, files_renamed = renames + n, files_renamed + (n > 0)
    spelt = next(((p, min(w)) for (p, _), x in toks.items() for w in [SOURCE_LOCATION & ids_in(x, 0, len(x) - 1)]
                  if renames and w), None)
    if spelt:  # a default argument spells its caller's signature wherever it is declared
        return f"{spelt[0]}: {spelt[1]} may spell a renamed parameter", 0, 0
    return None, renames, files_renamed


def renamed_difference(base_paths, head_paths, base, head):
    """Why every head file is not its base file at the same path up to template-parameter renames
    (alpha_problem, after v2 normalization), else None, as (why, renames, files renamed)."""
    if base_paths != head_paths:
        return "the source list changed (a move is the moved proof's job)", 0, 0
    files = [(p, base.read(p), head.read(p)) for p in head_paths]
    gone = [p for p, old, new in files if old is None or new is None]
    if gone:
        return f"{gone[0]} is missing", 0, 0
    return alpha_problem(files)


def rehash_renamed(ledger_path, op, base, at=None, repo=REPO, write=True):
    """The renamed proof of a rehash (docs/design/tuning-data.md#tuning-data-kernel-hash): per family
    of the measured records, the same source list at `base` and HEAD, every file equal up to
    template-parameter renames (renamed_difference). One line per stored hash whose effective hash is
    the family's hash at `base`. Returns (report, records)."""
    def prove(bp, hp, bt, ht):
        why, n, files = renamed_difference(bp, hp, bt, ht)
        return why, f"{n} template parameters renamed in {files} files, otherwise equal modulo comments and whitespace"
    return rehash_by_proof(ledger_path, op, base, "renamed", prove, at, repo, write)


def rehash_by_proof(ledger_path, op, base, proof, prove, at=None, repo=REPO, write=True):
    """The git proofs' driver: prove(base paths, head paths, base tree, head tree) -> (why not, why)
    per family whose records sit at its `base` hash; one '<proof>:<base>' line per stored hash."""
    led = read_ledger(ledger_path, raw=True)
    stored = defaultdict(set)
    for c in led.cells:
        for r in c["cands"] if c["tier"] not in UNHASHED else []:
            if r["hash"]:
                stored[spelling_family(r["cand"])].add(r["hash"])
    if not stored:
        return ["no measured records"], []
    bt, ht = GitTree(repo, base), GitTree(repo, "HEAD")
    spec = f"tools/tune/{op}_spec.cc"
    if bt.read(spec) is None:
        return [f"{f}: no proof: {spec} is missing at {base}" for f in sorted(stored)], []
    bb, hb = parse_kernel_block(bt.read(spec).decode()), parse_kernel_block(read_spec_source(op, repo))
    hashed = sorted({spec} | {p for f in stored for p in family_paths(hb, f)})
    if git_out(repo, "status", "--porcelain", "--", *hashed).strip():
        raise SystemExit(f"{op}: uncommitted changes to its hashed sources: commit them first")
    short = [git_out(repo, "rev-parse", "--short=8", r).decode().strip() for r in (base, "HEAD")]
    report, lines = [], []
    for f in sorted(stored):
        bp, hp = family_paths(bb, f), family_paths(hb, f)
        if any(bt.read(p) is None for p in bp):
            report.append(f"{f}: no proof: a source is missing at {base}")
            continue
        h_base, h_now = manifest_hash([(p, bt.read(p)) for p in bp]), kernel_hash(hp, repo)
        eff = {h: follow_rehash(led.rehash.get(f, {}), h) for h in stored[f]}
        todo = sorted(h for h, e in eff.items() if e == h_base != h_now)
        if not todo:
            report.append(f"{f}: nothing to rehash (records at {','.join(sorted(set(eff.values())))}, "
                          f"{base} {h_base}, now {h_now})")
            continue
        why, because = prove(bp, hp, bt, ht)
        if why:
            report.append(f"{f}: no proof: {why}")
            continue
        lines += [{"kind": "rehash", "family": f, "from": h, "to": h_now, "at": at or short[1],
                   "proof": f"{proof}:{short[0]}", "why": because} for h in todo]
        report.append(f"{f}: rehashed {','.join(todo)} -> {h_now}")
    if write and lines:
        with open(ledger_path, "a") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
    return report, lines


def hash_vector_problem(repo=REPO):
    """The shared hash v2 vector (also checked by tune_tests and check_tuned_tables.py)."""
    with open(os.path.join(repo, HASH_VECTOR, "expected.txt")) as f:
        want, path = f.read().split()
    got = kernel_hash([path], repo)
    return None if got == want else f"kernel hash v2 of {path}: {got} != {want}"


def test_git(repo, *args):
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], check=True, capture_output=True)


def test_put(repo, path, data):
    os.makedirs(os.path.dirname(os.path.join(repo, path)), exist_ok=True)
    with open(os.path.join(repo, path), "wb") as f:
        f.write(data)


TEST_SPEC = '// kernel-sources-begin\n"{k}",{extra}\n// family: tiny\n"{t}",\n// kernel-sources-end\n'


def moved_self_test(d, run_line, cell, cands):
    """--rehash-moved on throwaway git repos: a pure move, an include that now names another file,
    a content edit, a longer source list, an uncommitted change and the unmodelled include forms."""
    bad, git, put, spec = [], test_git, test_put, TEST_SPEC
    filler = "".join(f"int f{i}(int);\n" for i in range(24)).encode()

    def case(name, variant):
        repo = os.path.join(d, "moved-" + name)
        git(d, "init", "-q", repo)
        put(repo, "tools/tune/posv_spec.cc", spec.format(k="src/a/k.cc", t="src/a/t.cc", extra="").encode())
        odd = {"macro": b'#define HDR "util.hh"\n#include HDR\n', "has": b'#if __has_include("util.hh")\n#endif\n'}
        odd = odd.get(variant, b"")
        put(repo, "src/a/k.cc", b'#include "util.hh"\n#include "batchlas/p.hh"\n' + odd + filler + b"int k; // kernel\n")
        put(repo, "src/a/t.cc", b'#include "util.hh"\n' + filler + b"int t;\n")
        put(repo, "src/a/util.hh", filler + b"int u;\n")
        put(repo, "include/batchlas/p.hh", b"int p;\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "base")
        base = {f: kernel_hash(family_paths(parse_kernel_block(read_spec_source("posv", repo)), f), repo) for f in cands}
        os.makedirs(os.path.join(repo, "src", "ops", "k"))
        git(repo, "mv", "src/a/k.cc", "src/ops/k/k.cc")
        git(repo, "mv", "src/a/t.cc", "src/ops/k/t.cc")
        inc = b'#include "../../a/util.hh"\n'
        if variant != "other":
            os.makedirs(os.path.join(repo, "src", "common"))
            git(repo, "mv", "src/a/util.hh", "src/common/util.hh")
            inc = b"#include <src/common/util.hh>\n" if variant == "root" else b'#include "../../common/util.hh"\n'
        else:
            put(repo, "src/ops/k/util.hh", filler + b"int u;\n")
            inc = b'#include "util.hh"\n'
        body = b"int k2;\n" if variant == "edit" else b"int k; /* moved */\n"
        put(repo, "src/ops/k/k.cc", inc + b'  #include "batchlas/p.hh"\n' + odd + filler + body)
        put(repo, "src/ops/k/t.cc", inc + filler + b"int t;\n")
        extra = '\n"src/common/util.hh",' if variant == "length" else ""
        put(repo, "tools/tune/posv_spec.cc", spec.format(k="src/ops/k/k.cc", t="src/ops/k/t.cc", extra=extra).encode())
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "move")
        old = cell("20261001T000000-h-1", "deep", 128, hashes=dict(base, tiny="v1tiny"))
        lines = [run_line("20261001T000000-h-1", "2026-10-01"), cell("20261001T000000-h-1", "deep", 64, hashes=base), old,
                 {"kind": "rehash", "family": "tiny", "from": "v1tiny", "to": base["tiny"], "at": "x", "proof": "normalized"}]
        path = os.path.join(d, f"moved-{name}.jsonl")
        with open(path, "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
        return repo, path, base

    def expect(cond, what):
        if not cond:
            bad.append(f"self-test: --rehash-moved: {what}")

    repo, path, base = case("ok", "ok")
    report, recs = rehash_moved(path, "posv", "HEAD~1", repo=repo)
    now = family_hashes(parse_kernel_block(read_spec_source("posv", repo)), cands, repo)
    short = git_out(repo, "rev-parse", "--short=8", "HEAD~1").decode().strip()
    expect(sorted((r["family"], r["from"]) for r in recs) == sorted(
        [("tiny", base["tiny"]), ("tiny", "v1tiny"), ("cta", base["cta"]), ("blocked", base["blocked"])])
        and all(r["to"] == now[r["family"]] != base[r["family"]] and r["proof"] == f"moved:{short}" for r in recs)
        and "2 files moved" in recs[-1]["why"], f"a pure move with rewritten includes -> {report} {recs}")
    cells = read_ledger(path).cells
    expect(all(freshness(c, now) == "current" for c in cells), "the moved records are current")
    expect(rehash_moved(path, "posv", "HEAD~1", repo=repo)[1] == [], "a second run writes nothing")
    with open(os.path.join(repo, "src", "ops", "k", "k.cc"), "ab") as f:
        f.write(b"int dirty;\n")
    try:
        rehash_moved(path, "posv", "HEAD~1", repo=repo, write=False)
        expect(False, "an uncommitted change to a hashed file was accepted")
    except SystemExit as e:
        expect("uncommitted" in str(e), f"uncommitted change message {e}")
    repo, path, _ = case("root", "root")
    expect(len(rehash_moved(path, "posv", "HEAD~1", repo=repo)[1]) == 4, "a move to root-based <src/...> includes")
    for name, want in (("other", "differs"), ("edit", "differs"), ("length", "source list length"),
                       ("macro", "neither"), ("has", "__has_include")):
        repo, path, _ = case(name, name)
        report, recs = rehash_moved(path, "posv", "HEAD~1", repo=repo)
        expect(not recs and any("no proof" in r and want in r for r in report), f"{name} -> {report}")
    return bad


RENAMED_BASE_K = (b'#include "util.hh"\ntemplate <typename T, int N>\nstruct Tile { T v[N]; T get(int i) const '
                  b'{ return v[i]; } };\ntemplate <typename T>\nT twice(T x) { int r = 2; return x * r; }  // kernel\n'
                  b'const char* name = "T";\nint k;\n')
RENAMED_HEAD_K = {
    "ok": b'#include "util.hh"\ntemplate <typename Real, int N>\nstruct Tile { Real v[N]; Real get(int i) const '
          b'{ return v[i]; } };\ntemplate <typename Scalar>\nScalar twice(Scalar x) { int r = 2; return x * r; }\n'
          b'const char* name = "T";\nint k;\n',
    "cosmetic": b'#include "util.hh"\n/* Real, not T */\ntemplate <typename Real,\n          int N>\nstruct Tile {\n'
                b'    Real v[N];\n    Real get(int i) const { return v[i]; }  // one element\n};\n'
                b'template <typename Scalar>\nScalar twice(Scalar x) { int r = 2; return x * r; }\n'
                b'const char* name = "T";\nint k;\n',
    "used": b'#include "util.hh"\ntemplate <typename Real, int N>\nstruct Tile { Real v[N]; Real get(int i) const '
            b'{ return v[i]; } };\ntemplate <typename r>\nr twice(r x) { int r = 2; return x * r; }\n'
            b'const char* name = "T";\nint k;\n',
    "inconsistent": b'#include "util.hh"\ntemplate <typename Real, int N>\nstruct Tile { Real v[N]; Scalar get(int i) '
                    b'const { return v[i]; } };\ntemplate <typename T>\nT twice(T x) { int r = 2; return x * r; }\n'
                    b'const char* name = "T";\nint k;\n',
    "nonparam": b'#include "util.hh"\ntemplate <typename Real, int N>\nstruct Tile { Real v[N]; Real get(int i) const '
                b'{ return v[i]; } };\ntemplate <typename Scalar>\nScalar twice(Scalar x) { int two = 2; return x * two; }\n'
                b'const char* name = "T";\nint k;\n',
    "edit": b'#include "util.hh"\ntemplate <typename Real, int N>\nstruct Tile { Real v[N]; Real get(int i) const '
            b'{ return v[i]; } };\ntemplate <typename Scalar>\nScalar twice(Scalar x) { int r = 2; return x * r + 0; }\n'
            b'const char* name = "T";\nint k;\n',
}


RENAMED_MEMBER = b"template <class T> struct A : T { T x; typename B::T y; };\n"
RENAMED_S = b"template <class T> struct S { using Rv = int; S(); void f(); T m, n; };\n"
RENAMED_ALIAS = b"struct Base { using Zt = double; };\ntemplate <class> using Al = Base;\n"
RENAMED_COL = b"#define COL(k) R a = R(k);\ntemplate <class D> int g() { using R = int; COL(1) return a; }\n"
RENAMED_TI = b"template <class Rt, class T> struct Ti;\ntemplate <class T> struct Ti<int, T> { int v; };\n"
RENAMED_STD = b"namespace bl { namespace std { struct true_type { using Up = int; }; }\n"
RENAMED_INT = b"using Int = int;\ntemplate <class T> struct S { using Up = char; static Int v; };\n"
# unnamed template parameters naming W: renaming W is never a parameter rename
UNNAMED_FORMS = ("void (W::*)()", "int W::*", "W (&)[3]", "W[3]", "enum W", "struct W", "class W::V", "class = W",
                 "typename = W", "W*", "W&", "const W", "W const", "typename W::type", "W", "W<int>", "decltype(W())",
                 "W (*)(int)", "int (*)(W)", "template <class> class = W", "template <class W> class", "auto = W{}",
                 "int = W::v", "class... ")
RENAMED_FIXTURES = (  # (base, head, None for a proof or a substring of the refusal)
    (RENAMED_MEMBER, b"template <class Rv> struct A : Rv { Rv x; typename B::T y; };", "T occurs in the new"),
    (RENAMED_MEMBER, b"template <class Rv> struct A : Rv { Rv x; typename B::Rv y; };", "a member access"),
    (b'template <class T> void f() { g("T", R"x(T)x", T(1)); }',
     b'template <class Up> void f() { g("Up", R"x(T)x", Up(1)); }', '"T" -> "Up"'),
    (b"template <class T> void f() { CHECK(T(1)); }", b"template <class Up> void f() { CHECK(Up(1)); }", "macro call"),
    (b"template <class T> void f() { assert(T(1)); }", b"template <class Up> void f() { assert(Up(1)); }",
     "assert spells"),
    (b"template <class T> void f() { va_end(T(1)); }", b"template <class Up> void f() { va_end(Up(1)); }", "macro call"),
    (b"#define M Up\ntemplate <class T> void f() { T(); }", b"#define M Up\ntemplate <class Up> void f() { Up(); }",
     None),
    (b"#define M Up\ntemplate <class T> void f() { M(); T(); }",
     b"#define M Up\ntemplate <class Up> void f() { M(); Up(); }", "macro M, which may name T or Up"),
    (b"#define I(T) template void f<T>();\ntemplate <class T> void f() { auto g = [](int k) mutable -> T "
     b"{ return T(k); }; }", b"#define I(T) template void f<T>();\ntemplate <class Up> void f() "
     b"{ auto g = [](int k) mutable -> Up { return Up(k); }; }", None),
    (b"template <class T> void f(P* p) { p->T(); }", b"template <class Up> void f(P* p) { p->Up(); }",
     "a member access"),
    (b"#define cat(a, b) a##b\nstruct T1 {}; struct Up1 {};\ntemplate <class T> void f() { cat(T, 1) z; }",
     b"#define cat(a, b) a##b\nstruct T1 {}; struct Up1 {};\ntemplate <class Up> void f() { cat(Up, 1) z; }",
     "macro cat"),
    (b"#define s(x) #x\ntemplate <class T> void f() { g(s(T)); }",
     b"#define s(x) #x\ntemplate <class Up> void f() { g(s(Up)); }", "macro call"),
    (b"#define OPEN g(\ntemplate <class T> void f() { T x; }", b"#define OPEN g(\ntemplate <class Up> void f() { Up x; }",
     "unbalanced brackets"),
    (b'template <class T> void f() { g(R"x(a\\\nb)x"); }', b'template <class T> void f() { g(R"x(ab)x"); }',
     "line splice inside the raw string"),
    (b"template <class T> auto f(T x) -> E<c ? g() : h(), int> { return 0; }\nT y;",
     b"template <class Up> auto f(Up x) -> E<c ? g() : h(), int> { return 0; }\nUp y;", "initializer list"),
    (b"template <class T> S<T>::S() : m(0) MACRO { }\nT y;", b"template <class Up> S<Up>::S() : m(0) MACRO { }\nUp y;",
     "initializer list"),
    (b"template <class T> void f() try { T x; } catch (...) { }",
     b"template <class Up> void f() try { Up x; } catch (...) { }", "function-try-block"),
    (b"template <class T> constexpr T v = T{1} + Up{2};", b"template <class Up> constexpr Up v = Up{1} + Up{2};",
     "Up occurs in the old scope"),
    (b"template <class T> void f() { a[0]()->T(); }", b"template <class Up> void f() { a[0]()->Up(); }",
     "a member access"),
    (b"template <class T> struct A { template <class Up> void f(Up); T t; };",
     b"template <class T> struct A { template <class T> void f(T); T t; };", "an enclosing template has a parameter T"),
    (RENAMED_S + b"template <class T> void S<T>::f() { T x; }",
     RENAMED_S + b"template <class Rv> void S<Rv>::f() { Rv x; }", "out-of-line member of S"),
    (RENAMED_S + b"template <class T> void S<T>::f() { T x; }",
     RENAMED_S + b"template <class Vt> void S<Vt>::f() { Vt x; }", None),
    (RENAMED_S + b"template <class T> S<T>::S() : m{0}, n(1) { T z; }",
     RENAMED_S + b"template <class Real> S<Real>::S() : m{0}, n(1) { Real z; }", None),
    (b"template <class T> void S<T>::f() { T x; }", b"template <class Vt> void S<Vt>::f() { Vt x; }",
     "out-of-line member of S"),
    (b"struct B { using Rv = int; };\ntemplate <class T> struct S : B { T x; };",
     b"struct B { using Rv = int; };\ntemplate <class Rv> struct S : B { Rv x; };", "may hide"),
    (b"template <class T> requires requires (T a) { a + a; } T f(T x) { return x; }",
     b"template <class Up> requires requires (Up a) { a + a; } Up f(Up x) { return x; }", None),
    (b"template <int N> void f() requires (N > 1) { int a[N]; }",
     b"template <int Mv> void f() requires (Mv > 1) { int a[Mv]; }", None),
    (b"template <int Bar::* = nullptr> int f() { return sizeof(Bar); }",
     b"template <int Baz::* = nullptr> int f() { return sizeof(Baz); }", "Bar -> Baz is not a template parameter"),
    (b"template <class T, double Bar::*> int f() { return sizeof(Bar); }",
     b"template <class T, double Baz::*> int f() { return sizeof(Baz); }", "Bar -> Baz is not a template parameter"),
    (b"template <class T, void (*F)(T), int A[3]> void f() { F(A[0]); }",
     b"template <class T, void (*G)(T), int A[3]> void f() { G(A[0]); }", "F -> G is not a template parameter"),
    (b"template <class T, int A[3]> void f() { g(A); }", b"template <class T, int B[3]> void f() { g(B); }",
     "A -> B is not a template parameter"),
    (b"template <class T, const Foo, typename T::Bar, struct Baz> int f() { return sizeof(Foo); }",
     b"template <class T, const Qux, typename T::Bar, struct Baz> int f() { return sizeof(Qux); }",
     "Foo -> Qux is not a template parameter"),
    (b"template <class T, int* __restrict> void f() { T x; }", b"template <class T, int* X> void f() { T x; }",
     "the template structure differs"),
    (b"template <typename E> requires std::is_enum_v<E> && requires (E e) { g(e); } void f(E e) { g(e); }",
     b"template <typename En> requires std::is_enum_v<En> && requires (En e) { g(e); } void f(En e) { g(e); }", None),
    (b"template <class T, int* P, std::size_t N, C<int> V, template <class> class Tt, class... Ts, class = T>\n"
     b"void f() { T a; *P; N; Tt<V> b; g<::h, Ts...>(); }",
     b"template <class Tv, int* Qp, std::size_t Mv, C<int> Wv, template <class> class Ut, class... Us, class = Tv>\n"
     b"void f() { Tv a; *Qp; Mv; Ut<Wv> b; g<::h, Us...>(); }", None),
    (b"struct T { int a; };\ntemplate <class T> int f(T x) <% x; return T(x); %>",
     b"struct T { int a; };\ntemplate <class Up> int f(Up x) <% x; return T(x); %>", "a digraph"),
    (b"#define B <%\ntemplate <class T> int f(T x) { return 0; }", b"#define B <%\ntemplate <class Up> int f(Up x) "
     b"{ return 0; }", "a digraph"),
    (b"template <class T> auto f() -> int&& requires (sizeof(T) > 1) { return 0; }\nT y;",
     b"template <class Up> auto f() -> int&& requires (sizeof(Up) > 1) { return 0; }\nUp y;", "ref-qualifier"),
    (RENAMED_S + b"template <class T> void S<T>::f() and requires (sizeof(T) > 1) { }\nT y;",
     RENAMED_S + b"template <class Vt> void S<Vt>::f() and requires (sizeof(Vt) > 1) { }\nVt y;", "ref-qualifier"),
    (b"template <class T> X<T{}> f() { return T(); }", b"template <class Up> X<Up{}> f() { return T(); }",
     "after the body"),
    (b"template <class T> auto f() -> X<T{}[0]> { return T(); }",
     b"template <class Up> auto f() -> X<Up{}[0]> { return T(); }", "after the body"),
    (RENAMED_S + b"template <class T> S<T>::S() try : m(0) { } catch (T&) { }",
     RENAMED_S + b"template <class Vt> S<Vt>::S() try : m(0) { } catch (T&) { }", "function-try-block"),
    (b"template <class T> struct S { using Up = int; ~S(); };\ntemplate <class T> S<T>::compl S() { T y; }",
     b"template <class T> struct S { using Up = int; ~S(); };\ntemplate <class Up> S<Up>::compl S() { Up y; }",
     "out-of-line member of S"),
    (b"template <class T>\n#if A\nint f(T);\n#else\nint f(T) { return 0; }\n#endif\n",
     b"template <class Up>\n#if A\nint f(Up);\n#else\nint f(T) { return 0; }\n#endif\n", "# if A reaches the top level"),
    (b"template <class T> int f(T x) {\n#include \"body.inc\"\n}",
     b"template <class Up> int f(Up x) {\n#include \"body.inc\"\n}", "#include inside brackets"),
    (b"template <class T>\n#include \"decl.inc\"\nint f(T);", b"template <class Up>\n#include \"decl.inc\"\nint f(Up);",
     "# include inside the scope"),
    (b"template <class T> int f(T x) {\n#if A\n}\n#endif\nreturn T(x); }",
     b"template <class Up> int f(Up x) {\n#if A\n}\n#endif\nreturn T(x); }", "do not balance"),
    (b"template <class T> T f(T x) {\n#if A\nreturn x;\n#else\nreturn T(1);\n#endif\n}",
     b"template <class Up> Up f(Up x) {\n#if A\nreturn x;\n#else\nreturn Up(1);\n#endif\n}", None),
    (b"template <class T> int f(T x) { if (x) { g();\n#if A\n} else if (h()) {\n#endif\n} return T(x); }",
     b"template <class Up> int f(Up x) { if (x) { g();\n#if A\n} else if (h()) {\n#endif\n} return Up(x); }", None),
    (b"#if A\ntemplate <class T> struct S { T a;\n#ifdef B\nint b;\n#endif\n};\n#endif\n",
     b"#if A\ntemplate <class Up> struct S { Up a;\n#ifdef B\nint b;\n#endif\n};\n#endif\n", None),
    (b"template <class T> int f(T x) { g();\n#if A\n} int k() {\n#endif\nreturn T(x); }",
     b"template <class Up> int f(Up x) { g();\n#if A\n} int k() {\n#endif\nreturn T(x); }", "reaches the top level"),
    (b"template <class T> struct S { void f();\n#if A\n}; struct S2 {\n#endif\nusing Up = int; };\n"
     b"template <class T> void S<T>::f() { T y; }", b"template <class T> struct S { void f();\n#if A\n}; struct "
     b"S2 {\n#endif\nusing Up = int; };\ntemplate <class Up> void S<Up>::f() { Up y; }", "top level of class S"),
    (b"#define BODY { return 1; }\ntemplate <class T> int f() BODY int g() { return T(); }",
     b"#define BODY { return 1; }\ntemplate <class Up> int f() BODY int g() { return Up(); }", "macro BODY at the top"),
    (b"#define ID(x) x\n#define ID2 ID\ntemplate <class T> int f() ID2({ return 1; }) int g() { return T(); }",
     b"#define ID(x) x\n#define ID2 ID\ntemplate <class Up> int f() ID2({ return 1; }) int g() { return Up(); }",
     "macro ID2 at the top"),
    (b"#define BASE : B\nstruct B { using Rv = int; };\ntemplate <class T> struct S BASE { T x; };",
     b"#define BASE : B\nstruct B { using Rv = int; };\ntemplate <class Rv> struct S BASE { Rv x; };", "macro BASE"),
    (b"template <class T, auto F = [](T a) { return a > 0; }, int N = int{3}>\n[[x::y({;})]] bool g(T x) "
     b"{ return F(x); }", b"template <class Up, auto F = [](Up a) { return a > 0; }, int N = int{3}>\n[[x::y({;})]] "
     b"bool g(Up x) { return F(x); }", None),
    (b"#define INL(n) inline __attribute__((n))\ntemplate <class T> INL(hot) T f(T x) { return x; }",
     b"#define INL(n) inline __attribute__((n))\ntemplate <class Up> INL(hot) Up f(Up x) { return x; }", None),
    (b"template <class X> struct A { struct B { char c[100]; }; int f(); };\n"
     b"template <class B> int A<B>::f() { return sizeof(B); }",
     b"template <class X> struct A { struct B { char c[100]; }; int f(); };\n"
     b"template <class Zt> int A<Zt>::f() { return sizeof(Zt); }", "out-of-line member of A"),
    (RENAMED_ALIAS + b"template <class T> struct S : Al<T> { int g() { return sizeof(T); } };",
     RENAMED_ALIAS + b"template <class Zt> struct S : Al<Zt> { int g() { return sizeof(Zt); } };", "may hide"),
    (RENAMED_ALIAS.replace(b"Zt", b"T") + b"template <class T> struct S : Al<T> { int g() { return sizeof(T); } };",
     RENAMED_ALIAS.replace(b"Zt", b"T") + b"template <class Zt> struct S : Al<Zt> { int g() { return sizeof(Zt); } };",
     "may hide"),
    (b"template <class T, class... Ts> struct S : public T, [[a]] virtual T::B, Ts... { int g() { return T(); } };",
     b"template <class Up, class... Ups> struct S : public Up, [[a]] virtual Up::B, Ups... { int g() { return Up(); } };",
     None),
    (b"template <class Ab> struct S { [[Ab]] Ab e; int i; };", b"template <class Zt> struct S { [[Zt]] Zt e; int i; };",
     "inside an attribute"),
    (b"template <class T> struct S { int a __attribute__((aligned(sizeof(T)))); T b; };",
     b"template <class Up> struct S { int a __attribute__((aligned(sizeof(Up)))); Up b; };", "inside an attribute"),
    (b"#include <cassert>\ntemplate <class T> int f(int x) { (void)T(x); return 1; }",
     b"#include <cassert>\ntemplate <class assert> int f(int x) { (void)assert(x); return 1; }", "assert is not"),
    (b"template <class errno> int f() { return errno(); }", b"template <class En> int f() { return En(); }", "errno is"),
    (b"template <int N> int f() { return N; }", b"template <int NB> int f() { return NB; }", "NB is not"),
    (b"namespace a { template <class X> struct S { int f(); }; }\nnamespace b {\n"
     b"template <class B> int S<B>::f() { return sizeof(B); } }", b"namespace a { template <class X> struct S "
     b"{ int f(); }; }\nnamespace b {\ntemplate <class Zt> int S<Zt>::f() { return sizeof(Zt); } }",
     "in this namespace"),
    (b"namespace a {\n" + RENAMED_S + b"template <class T> void S<T>::f() { T x; }\n}",
     b"namespace a {\n" + RENAMED_S + b"template <class Vt> void S<Vt>::f() { Vt x; }\n}", None),
    (b"namespace [[d]] a {\n" + RENAMED_S + b"template <class T> void S<T>::f() { T x; }\n}",
     b"namespace [[d]] a {\n" + RENAMED_S + b"template <class Vt> void S<Vt>::f() { Vt x; }\n}", "in this namespace"),
    (b"#define CO } int k() {\ntemplate <class T> int f() { g(); CO return T(); }",
     b"#define CO } int k() {\ntemplate <class Up> int f() { g(); CO return Up(); }", "unbalanced brackets"),
    (b"struct S\n#include \"base.inc\"\n{ template <class T> int f() { return sizeof(T); } };",
     b"struct S\n#include \"base.inc\"\n{ template <class Up> int f() { return sizeof(Up); } };",
     "#include inside class S"),
    (b"int N = 5, M = 7;\ntemplate <int N = 1 < 2> int f() { return N; }\nint g() { return N; }\nbool b = 3 > 2;",
     b"int N = 5, M = 7;\ntemplate <int M = 1 < 2> int f() { return M; }\nint g() { return M; }\nbool b = 3 > 2;",
     "cannot delimit the template head"),
    (b"int a = 1, b = 2;\ntemplate <int N = a < b> int f() { return N; }\nint g() { return N; }\nbool c = 3 > 2;",
     b"int a = 1, b = 2;\ntemplate <int M = a < b> int f() { return M; }\nint g() { return M; }\nbool c = 3 > 2;",
     "cannot delimit the template head"),
    (b"template <int N, class T = C<N < 4>> int f() { return sizeof(T); }\nint g() { return sizeof(T); }\nbool b = 3 > 2;",
     b"template <int N, class Up = C<N < 4>> int f() { return sizeof(Up); }\nint g() { return sizeof(Up); }\n"
     b"bool b = 3 > 2;", "cannot delimit the template head"),
    (b"template <auto F = &X::operator()> int f() { return F(); }",
     b"template <auto Gn = &X::operator()> int f() { return Gn(); }", "cannot delimit the template head"),
    # a token of the head before its parameter's declaration names the outer entity
    (b"struct Tp { char c[3]; }; struct Vv { char c[7]; };\ntemplate <class A = Tp, class Tp = int> int f() "
     b"{ return sizeof(A); }", b"struct Tp { char c[3]; }; struct Vv { char c[7]; };\ntemplate <class A = Vv, "
     b"class Vv = int> int f() { return sizeof(A); }", "Tp occurs in the head before Tp"),
    (b"int N = 5, Mv = 9;\ntemplate <int A = N, int N = 2> int f() { return A; }",
     b"int N = 5, Mv = 9;\ntemplate <int A = Mv, int Mv = 2> int f() { return A; }", "N occurs in the head before N"),
    (b"template <class Tp, class A = Tp> int f() { return sizeof(A); }",
     b"template <class Vv, class A = Vv> int f() { return sizeof(A); }", None),
    # a '<' inside a template argument list of the head may be a less-than: `>>` would end it
    (b"int a = 1, b = 2;\ntemplate <bool> struct C {};\ntemplate <class T = C<a < b>> int f() { return 0; }\n"
     b"bool c = 3 > 2;", b"int a = 1, b = 2;\ntemplate <bool> struct C {};\ntemplate <class Up = C<a < b>> int f() "
     b"{ return 0; }\nbool c = 3 > 2;", "cannot delimit the template head"),
    (b"template <class T = A<B<int>>> int f() { return sizeof(T); }",
     b"template <class Up = A<B<int>>> int f() { return sizeof(Up); }", "cannot delimit the template head"),
    (b"template <class T = std::complex<float>, class L = tiers_t<T, int>> T f() { return T(); }",
     b"template <class Up = std::complex<float>, class L = tiers_t<Up, int>> Up f() { return Up(); }", None),
    # a conversion to a template-id and a parenthesised declarator name a member of S
    (b"template <class> struct V {};\ntemplate <class T> struct S { using Up = int; operator V<int>(); };\n"
     b"template <class T> S<T>::operator V<int>() { T x; return {}; }", b"template <class> struct V {};\ntemplate "
     b"<class T> struct S { using Up = int; operator V<int>(); };\ntemplate <class Up> S<Up>::operator V<int>() "
     b"{ Up x; return {}; }", "out-of-line member of S"),
    (b"template <class> struct V {};\ntemplate <class T> struct S { operator V<int>(); };\n"
     b"template <class T> S<T>::operator V<int>() { T x; return {}; }", b"template <class> struct V {};\ntemplate "
     b"<class T> struct S { operator V<int>(); };\ntemplate <class Up> S<Up>::operator V<int>() { Up x; return {}; }",
     None),
    (b"template <class T> struct S { using Up = int; int f(); };\ntemplate <class T> int (S<T>::f)() "
     b"{ return sizeof(T); }", b"template <class T> struct S { using Up = int; int f(); };\ntemplate <class Up> "
     b"int (S<Up>::f)() { return sizeof(Up); }", "parenthesised declarator"),
    (b"template <class T> int (f)(T x) { return sizeof(T); }",
     b"template <class Up> int (f)(Up x) { return sizeof(Up); }", None),
    # name shapes: a header outside the hashed files may make a lowercase name or one capital a macro
    (b"#include <csignal>\ntemplate <class T> int f(int x) { return T(x); }",
     b"#include <csignal>\ntemplate <class sigmask> int f(int x) { return sigmask(x); }", "sigmask is not"),
    (b"template <class value> int f() { return value(); }", b"template <class Vt> int f() { return Vt(); }",
     "value is not one capital"),
    (b"template <class T> int f() { return T(); }", b"template <class I> int f() { return I(); }", "I is not"),
    (b"template <class Real> int f() { return Real(); }", b"template <class Scalar> int f() { return Scalar(); }",
     None),
    # a lambda's trailing return type after an attribute or an attribute-only macro
    (b"#define BODY __attribute__((always_inline))\ntemplate <class T> void f() { auto g = [&](int j) BODY -> T& "
     b"{ return h<T>(j); }; auto k = [=]() __attribute__((hot)) [[gnu::cold]] noexcept -> T { return T(); }; }",
     b"#define BODY __attribute__((always_inline))\ntemplate <class Up> void f() { auto g = [&](int j) BODY -> Up& "
     b"{ return h<Up>(j); }; auto k = [=]() __attribute__((hot)) [[gnu::cold]] noexcept -> Up { return Up(); }; }",
     None),
    (b"#define BODY mutable p\ntemplate <class T> void f() { auto g = [&](int j) BODY -> T& { return h(j); }; }",
     b"#define BODY mutable p\ntemplate <class Up> void f() { auto g = [&](int j) BODY -> Up& { return h(j); }; }",
     "a member access"),
    (b"template <class T> void f() { auto g = [&](int j) BODY -> T& { return h(j); }; }",
     b"template <class Up> void f() { auto g = [&](int j) BODY -> Up& { return h(j); }; }", "a member access"),
    (b"#ifdef X\n#define BODY [[gnu::hot]]\n#else\n#define BODY .p\n#endif\n"
     b"template <class T> void f() { auto g = [&](int j) BODY -> T& { return h(j); }; }",
     b"#ifdef X\n#define BODY [[gnu::hot]]\n#else\n#define BODY .p\n#endif\n"
     b"template <class Up> void f() { auto g = [&](int j) BODY -> Up& { return h(j); }; }", "a member access"),
    # bases that hide only integral_constant's members, and dependent template-id bases
    (b"template <class T> struct Is : std::false_type {};\ntemplate <class R> struct Is<std::complex<R>> : "
     b"std::true_type {};\ntemplate <class T> struct Fp : std::bool_constant<sizeof(T) == 4> { T x; };\n"
     b"auto sz(P* p) -> std::size_t;",
     b"template <class Scalar> struct Is : std::false_type {};\ntemplate <class Real> struct Is<std::complex<Real>> : "
     b"std::true_type {};\ntemplate <class Scalar> struct Fp : std::bool_constant<sizeof(Scalar) == 4> { Scalar x; };\n"
     b"auto sz(P* p) -> std::size_t;",
     None),
    (RENAMED_STD + b"template <class T> struct S : std::true_type { T x; }; }",
     RENAMED_STD + b"template <class Up> struct S : std::true_type { Up x; }; }", "may hide"),
    (b"template <class T, class P> struct Ca { using Up = int; };\ntemplate <class T> struct A : Ca<T, int> { T x; };\n"
     b"template <class R> struct B : Ca<std::complex<R>, Ca<R, int>> { R y; };",
     b"template <class T, class P> struct Ca { using Up = int; };\ntemplate <class Up> struct A : Ca<Up, int> "
     b"{ Up x; };\ntemplate <class Real> struct B : Ca<std::complex<Real>, Ca<Real, int>> { Real y; };", None),
    (b"template <class T> struct A : Ca<T, int> { T x; };\ntemplate <class T, class P> struct Ca { using Up = int; };",
     b"template <class Up> struct A : Ca<Up, int> { Up x; };\ntemplate <class T, class P> struct Ca { using Up = int; };",
     "may hide"),
    (b"namespace n { template <class T> struct Ca { using Up = int; }; }\ntemplate <class T> struct A : n::Ca<int> "
     b"{ T x; };", b"namespace n { template <class T> struct Ca { using Up = int; }; }\ntemplate <class Up> struct A : "
     b"n::Ca<int> { Up x; };", "may hide"),
    (b"template <template <class> class Ca> struct X {};\ntemplate <class T> struct A : Ca<T> { T x; };",
     b"template <template <class> class Ca> struct X {};\ntemplate <class Up> struct A : Ca<Up> { Up x; };", "may hide"),
    (RENAMED_TI + b"template <class T> struct Ti<long, T> : Ti<int, T> { T x; };",
     RENAMED_TI + b"template <class Scalar> struct Ti<long, Scalar> : Ti<int, Scalar> { Scalar x; };", None),
    (b"namespace l3 { template <int A> struct Geom {}; namespace cfg {\ntemplate <class T, int A> struct Tiled : "
     b"Geom<A> { T x; }; } }", b"namespace l3 { template <int A> struct Geom {}; namespace cfg {\ntemplate <class Scalar, "
     b"int A> struct Tiled : Geom<A> { Scalar x; }; } }", None),
    (b"namespace l3 { template <int A> struct Geom {}; namespace cfg { template <int> using Geom = Base;\n"
     b"template <class T, int A> struct Tiled : Geom<A> { T x; }; } }", b"namespace l3 { template <int A> struct Geom "
     b"{}; namespace cfg { template <int> using Geom = Base;\ntemplate <class Scalar, int A> struct Tiled : Geom<A> "
     b"{ Scalar x; }; } }", "may hide"),
    (b"namespace l3 { template <int A> struct Geom {}; namespace cfg { using namespace other;\ntemplate <class T, int A> "
     b"struct Tiled : Geom<A> { T x; }; } }", b"namespace l3 { template <int A> struct Geom {}; namespace cfg { using "
     b"namespace other;\ntemplate <class Scalar, int A> struct Tiled : Geom<A> { Scalar x; }; } }", "may hide"),
    # a '<' after std:: and a std template inside a head's argument list opens its own
    (b"template <class M, class Re = Rt<std::remove_cv_t<typename M::value_type>>> Re f(M m) { return Re(); }",
     b"template <class Mv, class Re = Rt<std::remove_cv_t<typename Mv::value_type>>> Re f(Mv m) { return Re(); }",
     None),
    (b"template <class M, class Re = Rt<std::my_t<typename M::value_type>>> Re f(M m) { return Re(); }",
     b"template <class Mv, class Re = Rt<std::my_t<typename Mv::value_type>>> Re f(Mv m) { return Re(); }",
     "cannot delimit the template head"),
    # <cmath>'s classification names are functions; other library macros take no renamed argument
    (b"template <class T> struct A { static bool isfinite(const T& a) { return sycl::isfinite(a); } };",
     b"template <class Scalar> struct A { static bool isfinite(const Scalar& a) { return sycl::isfinite(a); } };",
     None),
    (b"template <class T> int f() { return offsetof(T, x); }", b"template <class Scalar> int f() "
     b"{ return offsetof(Scalar, x); }", "macro call"),
    # macros: parameters and bodies matter only where expanded or defined, a name only where in effect
    (b"#define INST(T) template int f<T>();\ntemplate <class T> int f() { return sizeof(T); }\nINST(float)",
     b"#define INST(T) template int f<T>();\ntemplate <class Scalar> int f() { return sizeof(Scalar); }\nINST(float)",
     None),
    (RENAMED_COL + b"#undef COL\ntemplate <class R> R h(R x) { return x; }",
     RENAMED_COL + b"#undef COL\ntemplate <class Real> Real h(Real x) { return x; }", None),
    (RENAMED_COL + b"template <class R> R h(R x) { COL(1) return x; }\n#undef COL",
     RENAMED_COL + b"template <class Real> Real h(Real x) { COL(1) return x; }\n#undef COL", "macro COL"),
    (b"#define CAT(a, b) a##b\n#define COL R\ntemplate <class R> R h(R x) { return CAT(CO, L)(x); }",
     b"#define CAT(a, b) a##b\n#define COL R\ntemplate <class Real> Real h(Real x) { return CAT(CO, L)(x); }",
     "macro CAT"),
    (b"#define Real double\nReal g();\n#undef Real\ntemplate <class R> R h(R x) { return x; }",
     b"#define Real double\nReal g();\n#undef Real\ntemplate <class Real> Real h(Real x) { return x; }", None),
    (b"#define Real double\ntemplate <class R> R h(R x) { return x; }\n#undef Real",
     b"#define Real double\ntemplate <class Real> Real h(Real x) { return x; }\n#undef Real", "in effect inside"),
    (b"#define Real double\ntemplate <class R> R h(R x) { return x; }",
     b"#define Real double\ntemplate <class Real> Real h(Real x) { return x; }", "beyond that file"),
    (b"#define Real double\n#include \"x.hh\"\n#undef Real\ntemplate <class R> R h(R x) { return x; }",
     b"#define Real double\n#include \"x.hh\"\n#undef Real\ntemplate <class Real> Real h(Real x) { return x; }",
     "beyond that file"),
    (b"#undef Real\ntemplate <class R> R h(R x) { return x; }",
     b"#undef Real\ntemplate <class Real> Real h(Real x) { return x; }", "without an earlier #define"),
    # a '[' after `operator`, `delete` or as `[[` starts no lambda: `-> T` there is a member access
    (b"struct E { int T, Up; }; struct A { E* operator[](int); };\ntemplate <class T> int f(A& a) "
     b"{ return a.operator[](0)->T; }", b"struct E { int T, Up; }; struct A { E* operator[](int); };\n"
     b"template <class Up> int f(A& a) { return a.operator[](0)->Up; }", "a member access"),
    (b"template <class T> void f(P* p) { delete [] (p)->T; }", b"template <class Up> void f(P* p) "
     b"{ delete [] (p)->Up; }", "a member access"),
    (b"template <class T> void f(P* p) { [[likely]] (p)->T = 7; }", b"template <class Up> void f(P* p) "
     b"{ [[likely]] (p)->Up = 7; }", "a member access"),
    (b"template <class T> T f() { return [](int k) -> T { return T(k); }(1) + g([&](int j) noexcept -> T "
     b"{ return T(j); }); }", b"template <class Up> Up f() { return [](int k) -> Up { return Up(k); }(1) + "
     b"g([&](int j) noexcept -> Up { return Up(j); }); }", None),
    # a pasting macro may build any name, so it is never expanded in a renamed scope
    (b"#define CAT(a, b) a##b\nstruct Up { static int v(); };\ntemplate <class T> int f() { return CAT(U, p)::v(); }",
     b"#define CAT(a, b) a##b\nstruct Up { static int v(); };\ntemplate <class Up> int f() { return CAT(U, p)::v(); }",
     "macro CAT"),
    (b"#define UPNAME U##p\nstruct Up { static int v(); };\ntemplate <class T> int f() { return UPNAME::v(); }",
     b"#define UPNAME U##p\nstruct Up { static int v(); };\ntemplate <class Up> int f() { return UPNAME::v(); }",
     "macro UPNAME"),
    (b"#define CAT(a, b) a##b\nint CAT(x, 1);\ntemplate <class T> int f() { return T(); }",
     b"#define CAT(a, b) a##b\nint CAT(x, 1);\ntemplate <class Up> int f() { return Up(); }", None),
    # pop_macro restores what #undef removed
    (b'#define Up(x) (x + 41)\n#pragma push_macro("Up")\n#undef Up\n#pragma pop_macro("Up")\n'
     b'template <class T> int f() { return T(1); }', b'#define Up(x) (x + 41)\n#pragma push_macro("Up")\n#undef Up\n'
     b'#pragma pop_macro("Up")\ntemplate <class Up> int f() { return Up(1); }', "pop_macro"),
    (b'#define Up(x) (x + 41)\n#pragma push_macro("Up")\n#undef Up\n_Pragma("pop_macro(\\"Up\\")")\n'
     b'template <class T> int f() { return T(1); }', b'#define Up(x) (x + 41)\n#pragma push_macro("Up")\n#undef Up\n'
     b'_Pragma("pop_macro(\\"Up\\")")\ntemplate <class Up> int f() { return Up(1); }', "pop_macro"),
    (b"#define PRAGMA(x) _Pragma(#x)\ntemplate <class T> int f() { return T(1); }",
     b"#define PRAGMA(x) _Pragma(#x)\ntemplate <class Up> int f() { return Up(1); }", "pop_macro"),
    (b'#pragma push_macro("Wx")\n#pragma pop_macro("Wx")\n#define UNROLL _Pragma("unroll")\n'
     b'template <class T> int f() { UNROLL for (;;) {} return T(1); }', b'#pragma push_macro("Wx")\n'
     b'#pragma pop_macro("Wx")\n#define UNROLL _Pragma("unroll")\ntemplate <class Up> int f() '
     b'{ UNROLL for (;;) {} return Up(1); }', None),
    # a parenthesised declarator after its type: `Int (S<T>::v)` defines S's v, where S's Up hides
    (RENAMED_INT + b"template <class T> Int (S<T>::v) = sizeof(T);",
     RENAMED_INT + b"template <class Up> Int (S<Up>::v) = sizeof(Up);", "parenthesised declarator"),
    (RENAMED_INT + b"template <class T> Int (S<T>::v){sizeof(T)};",
     RENAMED_INT + b"template <class Up> Int (S<Up>::v){sizeof(Up)};", "parenthesised declarator"),
    (RENAMED_INT + b"template <class T> [[maybe_unused]] Int (S<T>::v) = 1;\nT y;",
     RENAMED_INT + b"template <class Up> [[maybe_unused]] Int (S<Up>::v) = 1;\nUp y;", "parenthesised declarator"),
    (RENAMED_INT + b"template <class T> INL Int (S<T>::v) = 1;\nT y;",
     RENAMED_INT + b"template <class Up> INL Int (S<Up>::v) = 1;\nUp y;", "parenthesised declarator"),
    (RENAMED_S + b"template <class T> void S<T>::f(std::size_t n) { T x; }\ntemplate <class T> T const at(std::size_t "
     b"i) { return T(i); }\ntemplate <class T> decltype(auto) g(std::size_t i) { return T(i); }\n"
     b"template <class T> S(std::vector<T>) -> S<T>;\ntemplate <class T> Int h(std::size_t i) { return T(i); }",
     RENAMED_S + b"template <class Vt> void S<Vt>::f(std::size_t n) { Vt x; }\ntemplate <class Vt> Vt const "
     b"at(std::size_t i) { return Vt(i); }\ntemplate <class Vt> decltype(auto) g(std::size_t i) { return Vt(i); }\n"
     b"template <class Vt> S(std::vector<Vt>) -> S<Vt>;\ntemplate <class Vt> Int h(std::size_t i) { return Vt(i); }",
     None),
    # the signature's spelling holds the parameters' names
    (b"template <class T> void f() { puts(__PRETTY_FUNCTION__); }",
     b"template <class Up> void f() { puts(__PRETTY_FUNCTION__); }", "__PRETTY_FUNCTION__ spells"),
    (b"#define WHERE __PRETTY_FUNCTION__\ntemplate <class T> void f() { T x; puts(WHERE); }",
     b"#define WHERE __PRETTY_FUNCTION__\ntemplate <class Up> void f() { Up x; puts(WHERE); }", "macro WHERE"),
    (b"void log(std::source_location l = std::source_location::current());\ntemplate <class T> void f() { log(); T x; }",
     b"void log(std::source_location l = std::source_location::current());\ntemplate <class Up> void f() "
     b"{ log(); Up x; }", "source_location may spell"),
    (b"template <class T> void f() { T x; puts(__func__); }", b"template <class Up> void f() { Up x; puts(__func__); }",
     None),
    (b"#include <cassert>\ntemplate <class T> void f(int n) { T x; assert(n > 0); }\nvoid g(int n) { assert(n); }",
     b"#include <cassert>\ntemplate <class Up> void f(int n) { Up x; assert(n > 0); }\nvoid g(int n) { assert(n); }",
     "assert spells"),
    (b"#include <cassert>\nvoid g(int n) { assert(n); }\ntemplate <class T> void f(int n) { T x; }",
     b"#include <cassert>\nvoid g(int n) { assert(n); }\ntemplate <class Up> void f(int n) { Up x; }", None),
    # a specialization of integral_constant may give it any member
    (b"namespace std { template <> struct integral_constant<My*, nullptr> { using Up = char; }; }\ntemplate <class T> "
     b"struct S : std::integral_constant<My*, nullptr> { T x; };", b"namespace std { template <> struct "
     b"integral_constant<My*, nullptr> { using Up = char; }; }\ntemplate <class Up> struct S : "
     b"std::integral_constant<My*, nullptr> { Up x; };", "may hide"),
    (b"template <> struct std::integral_constant<My*, nullptr> { using Up = char; };\ntemplate <class T> "
     b"struct S : std::integral_constant<My*, nullptr> { T x; };", b"template <> struct std::integral_constant<My*, "
     b"nullptr> { using Up = char; };\ntemplate <class Up> struct S : std::integral_constant<My*, nullptr> { Up x; };",
     "may hide"),
    (b"template <class T> int f() { return T(); }", b"template <class Value> int f() { return Value(); }",
     "Value is not"),
) + tuple((f"template <class T, {form}> int f() {{ return sizeof(W); }}".encode(),
           f"template <class T, {form.replace('W', 'Qt')}> int f() {{ return sizeof(Qt); }}".encode(),
           "W -> Qt is not a template parameter") for form in UNNAMED_FORMS)


def renamed_self_test(d, run_line, cell, cands):
    """--rehash-renamed on throwaway git repos (a pure rename, one with comment and whitespace edits,
    a rename to a name the scope uses, an inconsistent one, a renamed local, an extra token, an
    uncommitted change) and renamed_tokens on member access, macros, literals and the C++ lexicon."""
    bad = []

    def case(name):
        repo = os.path.join(d, "renamed-" + name)
        test_git(d, "init", "-q", repo)
        test_put(repo, "tools/tune/posv_spec.cc", TEST_SPEC.format(k="src/a/k.cc", t="src/a/t.cc", extra="").encode())
        test_put(repo, "src/a/k.cc", RENAMED_BASE_K)
        test_put(repo, "src/a/t.cc", b"template <class T>\nT third(T x) { return x / 3; }\nint t;\n")
        test_put(repo, "src/a/util.hh", b"int u;\n")
        test_git(repo, "add", "-A")
        test_git(repo, "commit", "-qm", "base")
        base = {f: kernel_hash(family_paths(parse_kernel_block(read_spec_source("posv", repo)), f), repo) for f in cands}
        test_put(repo, "src/a/k.cc", RENAMED_HEAD_K[name])
        test_put(repo, "src/a/t.cc", b"template <class Real>\nReal third(Real x) { return x / 3; }\nint t;\n")
        test_git(repo, "commit", "-qam", "rename")
        rid = "20261001T000000-h-1"
        lines = [run_line(rid, "2026-10-01"), cell(rid, "deep", 64, hashes=base),
                 cell(rid, "deep", 128, hashes=dict(base, tiny="v1tiny")),
                 {"kind": "rehash", "family": "tiny", "from": "v1tiny", "to": base["tiny"], "at": "x", "proof": "normalized"}]
        path = os.path.join(d, f"renamed-{name}.jsonl")
        with open(path, "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
        return repo, path, base

    def expect(cond, what):
        if not cond:
            bad.append(f"self-test: --rehash-renamed: {what}")

    for name in ("ok", "cosmetic"):
        repo, path, base = case(name)
        report, recs = rehash_renamed(path, "posv", "HEAD~1", repo=repo)
        now = family_hashes(parse_kernel_block(read_spec_source("posv", repo)), cands, repo)
        short = git_out(repo, "rev-parse", "--short=8", "HEAD~1").decode().strip()
        why = {r["family"]: r["why"] for r in recs}
        expect(sorted((r["family"], r["from"]) for r in recs) == sorted(
            [("tiny", base["tiny"]), ("tiny", "v1tiny"), ("cta", base["cta"]), ("blocked", base["blocked"])])
            and all(r["to"] == now[r["family"]] != base[r["family"]] and r["proof"] == f"renamed:{short}" for r in recs)
            and why.get("tiny", "").startswith("3 template parameters renamed in 2 files")
            and why.get("cta", "").startswith("2 template parameters renamed in 1 files"),
            f"{name}: a template-parameter rename proves -> {report} {recs}")
        expect(all(freshness(c, now) == "current" for c in read_ledger(path).cells), f"{name}: the records are current")
        expect(rehash_renamed(path, "posv", "HEAD~1", repo=repo)[1] == [], f"{name}: a second run writes nothing")
    with open(os.path.join(repo, "src", "a", "k.cc"), "ab") as f:
        f.write(b"int dirty;\n")
    try:
        rehash_renamed(path, "posv", "HEAD~1", repo=repo, write=False)
        expect(False, "an uncommitted change to a hashed file was accepted")
    except SystemExit as e:
        expect("uncommitted" in str(e), f"uncommitted change message {e}")
    for name, want in (("used", "r occurs in the old scope"), ("inconsistent", "inconsistent rename"),
                       ("nonparam", "r -> two is not a template parameter"), ("edit", "tokens")):
        repo, path, _ = case(name)
        report, recs = rehash_renamed(path, "posv", "HEAD~1", repo=repo)
        expect(not recs and any("no proof" in r and want in r for r in report), f"{name} -> {report}")
    for old, new, want in RENAMED_FIXTURES:
        why = alpha_problem([("f.cc", old, new)])[0]
        expect(why is None if want is None else why and want in why, f"alpha_problem({new!r}) -> {why}")
    # another file's macros: an attribute macro defined there, a macro naming R expanded only there
    core = b"#define BODY __attribute__((always_inline))\n" + RENAMED_COL + b"#undef COL\n"
    lam = (b"template <class R> void f() { auto g = [&](int j) BODY -> R& { return h(j); }; }",
           b"template <class Real> void f() { auto g = [&](int j) BODY -> Real& { return h(j); }; }")
    why = [alpha_problem([("k.cc",) + lam] + ([("core.hh", core, core)] if two else []))[0] for two in (0, 1)]
    expect(why[0] and "a member access" in why[0] and why[1] is None, f"BODY defined in another file -> {why}")
    lexed = cxx_tokens(b"a>>=b; x // c\n/* d */ 'q' u8\"s\"_x R\"(p)\"\n %:define Q \\\n  1\n1'000.5e+3")
    expect(lexed == [("id", "a"), ("punct", ">>="), ("id", "b"), ("punct", ";"), ("id", "x"), ("str", "'q'"),
                     ("str", 'u8"s"_x'), ("str", 'R"(p)"'), ("pp", "%: define Q 1"), ("num", "1'000.5e+3")],
           f"cxx_tokens -> {lexed}")
    return bad


def self_test():
    """Table generation from synthetic ledgers, the shared fixtures and the hash vector."""
    bad = []

    def expect(cond, what):
        if not cond:
            bad.append(f"self-test: {what}")

    expect(hash_vector_problem() is None, hash_vector_problem())
    for text, want in ((b"a /* x */ b // y\n c", b"abc"), (b"a/* open", b"a"), (b"/* // */x", b"x"),
                       (b"// /* \nq */r", b""), (b"\"s // t\"", b"\"s")):
        expect(normalize_source(text) == want, f"normalize_source({text!r}) -> {normalize_source(text)!r}")
    keys = "uplo:exact n:log:3 nrhs:log batch:log"
    keyspec = parse_keys(keys)
    cands = ["tiny", "cta", "blocked"]
    fh = {"tiny": "htiny", "cta": "hcta", "blocked": "hblocked"}

    def run_line(rid, date, tier="deep", dtype="float", batchlas=None):
        return {"kind": "run", "dtype": dtype, "run_id": rid, "host": "h", "device": "sm_0", "device_name": "d",
                "batchlas": batchlas or "bl" + rid[-1], "argv": "", "date": date, "tier": tier, "keys": keys,
                "candidates": "|".join(cands)}

    def cell(rid, tier, n, ranked=("tiny", "cta"), date="2026-10-01", hashes=None, dtype="float", uplo="L", batch=128):
        r = {"kind": "cell", "dtype": dtype, "run_id": rid, "tier": tier, "key": f"uplo={uplo},n={n},nrhs=1,batch={batch}",
             "round": 0, "date": date, "ranked": "|".join(ranked), "cands": "|".join(cands)}
        for i, c in enumerate(cands):
            r.update({f"h{i}": (hashes or fh)[c], f"s{i}": "ok" if c in ranked else "skipped", f"r{i}": "",
                      f"m{i}": 1.0 + i if c in ranked else None, f"lo{i}": None, f"hi{i}": None, f"n{i}": 4})
        return r

    def build(lines, d, name="posv.sm_0.jsonl"):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
        return path

    def rows_of(path, judge=fh):
        return ledger_rows(keyspec, read_ledger(path, "float").cells, judge)

    def tiers(rows):
        return {k[1]: t for k, (t, _) in rows.items()}

    with tempfile.TemporaryDirectory() as d:
        r1, r2, r3, r4 = "20261001T000000-h-1", "20261002T000000-h-2", "20261003T000000-h-3", "20261004T000000-h-4"
        # lattice n=8..256; deep@64 and coarse@32 bracket preview@16 (1 from coarse) and preview@128 (1 from deep).
        base = [run_line(r1, "2026-10-01"), cell(r1, "deep", 64), run_line(r2, "2026-10-02", "coarse"),
                cell(r2, "coarse", 32), run_line(r3, "2026-10-03", "preview")] + \
               [cell(r3, "preview", n) for n in (8, 16, 128, 256)]
        led = build(base, d)
        expect(tiers(rows_of(led)) == {8: "preview", 32: "coarse", 64: "deep", 256: "preview"},
               f"gap fill drops a lower tier inside a higher bracket -> {tiers(rows_of(led))}")
        # a measured PROTOCOL record outranks another protocol's at a higher tier
        warm = dict(run_line(r1, "2026-10-01"), protocol="lean")
        cold = dict(run_line(r2, "2026-10-02", "preview"), protocol=PROTOCOL)
        got = tiers(rows_of(build([warm, cell(r1, "deep", 64), cold, cell(r2, "preview", 64)], d, "cold.jsonl")))
        expect(got == {64: "preview"}, f"a cold preview record under a warm deep one -> {got}")
        got = tiers(rows_of(build([dict(warm, protocol=PROTOCOL), cell(r1, "deep", 64), dict(cold, protocol="lean"),
                                   cell(r2, "preview", 64)], d, "cold.jsonl")))
        expect(got == {64: "deep"}, f"a warm preview record under a cold deep one -> {got}")

        # monotonicity: whole-cell time 10 ms at batch 128 but 1 ms at 1024 flags both; 2x and n=32 do not
        def timed_cell(n, batch, ms):
            c = cell(r1, "deep", n, batch=batch)
            for i in range(len(cands)):
                if c[f"m{i}"] is not None:
                    c[f"m{i}"] = ms
            return c
        mono = build([warm] + [timed_cell(64, b, ms) for b, ms in ((32, 0.5), (128, 10.0), (1024, 1.0), (4096, 4.0))] +
                     [timed_cell(32, b, ms) for b, ms in ((128, 2.0), (1024, 1.0))], d, "mono.jsonl")
        got = monotonicity("posv", "float", read_ledger(mono, "float").cells, fh)
        want = ["posv float uplo=L,n=64,nrhs=1,batch=1024", "posv float uplo=L,n=64,nrhs=1,batch=128"]
        expect(got == want, f"monotonicity -> {got}")
        # phase timers, the run's protocol and an audit's seconds leave the table byte-identical
        plain = ledger_text("posv", "sm_0", build(base, d, "ph.jsonl"), fh, "k").text
        phases = {"ph_setup": 0.5, "ph_warm": 0.2, "ph_timed": 1.25, "ph_verify": 0.1, "ph_ld": 0.3, "ph_guard": 0.05,
                  "ph_ipc": 0.01, "ph_start": 3.5, "ph_mode": "worker"}
        timed = [dict(x, **phases) if x["kind"] == "cell" else dict(x, protocol="full") for x in base]
        audit = {"kind": "audit", "dtype": "float", "run_id": r1, "key": "uplo=L,n=64,nrhs=1,batch=128",
                 "verdict": "ok", "fresh_ms": 1.0, "warm_ms": 1.0, "audit_s": 2.5}
        expect(ledger_text("posv", "sm_0", build(timed + [audit], d, "ph.jsonl"), fh, "k").text == plain,
               "ph_* fields changed the table")
        # v<i> = 0 (timed, never verified: the lean race's non-tie arms) stays in the row; absent = verified
        unver = [dict(x, v1=0) if x["kind"] == "cell" else x for x in base]
        vpath = build(unver, d, "ph.jsonl")
        expect(ledger_text("posv", "sm_0", vpath, fh, "k").text == plain, "v<i> = 0 changed the table")
        vc = read_ledger(vpath, "float").cells[0]["cands"]
        expect([c["verified"] for c in vc] == [True, False, True], f"v<i> read back -> {[c['verified'] for c in vc]}")
        # an empty ranking emits no row; a stale winner is skipped so the lower tier stands
        led2 = build([run_line(r3, "2026-10-03", "preview"), cell(r3, "preview", 8), cell(r3, "preview", 256),
                      run_line(r4, "2026-10-04"), cell(r4, "deep", 8, ranked=()),
                      cell(r4, "deep", 256, hashes=dict(fh, tiny="deadbeef"))], d, "b.jsonl")
        expect(tiers(rows_of(led2)) == {256: "preview"}, f"empty ranking / stale winner -> {tiers(rows_of(led2))}")
        # transcribed rows survive only away from measured rows (1024 is off the lattice), kept or dropped
        tr = [dict(cell("transcribed-2b46acab", "transcribed", n, date="2026-09-01"), **{f"h{i}": "" for i in range(3)},
                   **{f"m{i}": None for i in range(3)}) for n in (16, 32, 1024)]
        led3 = build(base + tr, d, "c.jsonl")
        got3 = rows_of(led3)
        expect(tiers(got3) == {8: "preview", 32: "coarse", 64: "deep", 256: "preview", 1024: "transcribed"},
               f"transcribed survive away from measured -> {tiers(got3)}")
        expect(got3[("L", 1024, 1, 128)][1] == [("tiny", None), ("cta", None)], "a transcribed row is untimed")
        # a measured record with an empty ranking emits no row and does not hide the transcribed one there
        tru = dict(tr[0], key="uplo=U,n=64,nrhs=1,batch=128")
        got3b = rows_of(build(base + [tru, cell(r4, "deep", 64, ranked=(), date="2026-10-04", uplo="U")], d, "c2.jsonl"))
        expect(got3b.get(("U", 64, 1, 128), ("",))[0] == "transcribed",
               f"empty measured ranking keeps the transcribed row -> {got3b.get(('U', 64, 1, 128))}")
        # prior rows give way only to a measured record at their own key (16: a dropped preview record), never
        # to a neighbour (512); transcribed rows give way to them as to a measured row (512 same key, 1024 near)
        pr = [dict(cell("20261005T000000-h-5", "prior", n, date="2026-10-05"), **{f"h{i}": "" for i in range(3)},
                   **{f"m{i}": None for i in range(3)}) for n in (16, 512)]
        tr6 = [dict(t, key=t["key"].replace("n=16,", f"n={n},")) for t in tr[:1] for n in (512, 1024, 4096)]
        got6 = rows_of(build(base + pr + tr6, d, "p.jsonl"))
        expect(tiers(got6) == {8: "preview", 32: "coarse", 64: "deep", 256: "preview", 512: "prior", 4096: "transcribed"},
               f"prior rows -> {tiers(got6)}")
        expect(got6[("L", 512, 1, 128)][1] == [("tiny", None), ("cta", None)], "a prior row is untimed")
        # a dtype with only prior and transcribed records (every measured winner stale) still derives
        stale = [run_line(r4, "2026-10-04"), cell(r4, "deep", 64, hashes=dict(fh, tiny="deadbeef"))]
        got6b = rows_of(build(stale + pr + tr6, d, "p2.jsonl"))
        expect(tiers(got6b) == {16: "prior", 512: "prior", 4096: "transcribed"}, f"prior-only lattice -> {tiers(got6b)}")
        # after calibrate (probes 8, 128; prior rows between), a stride-2 deep run (8, 32, 128) leaves no prior
        # row between its rows: a coarse or deep row brackets a prior row within 2 steps, a calibrate row only at its key
        rc, rd = "20261006T000000-h-6", "20261007T000000-h-7"
        cal = [run_line(rc, "2026-10-06", "calibrate")] + [cell(rc, "calibrate", n, date="2026-10-06") for n in (8, 128)]
        pr7 = [dict(cell(rc, "prior", n, date="2026-10-06"), **{f"h{i}": "" for i in range(3)},
                    **{f"m{i}": None for i in range(3)}) for n in (16, 32, 64)]
        got7 = rows_of(build(cal + pr7, d, "q.jsonl"))
        expect(tiers(got7) == {8: "calibrate", 16: "prior", 32: "prior", 64: "prior", 128: "calibrate"},
               f"calibrate keeps its prior rows -> {tiers(got7)}")
        deep7 = [run_line(rd, "2026-10-07")] + [cell(rd, "deep", n, date="2026-10-07") for n in (8, 32, 128)]
        got7b = rows_of(build(cal + pr7 + deep7, d, "q2.jsonl"))
        expect(tiers(got7b) == {8: "deep", 32: "deep", 128: "deep"}, f"a stride-2 deep run drops prior rows -> {tiers(got7b)}")
        # converted rows are timed, never stale, and give way to a measured row the same way
        cv = cell("converted-x", "converted", 64, hashes={c: "" for c in cands}, date="2026-09-01")
        cv2 = dict(cell("converted-x", "converted", 2048, hashes={c: "" for c in cands}, date="2026-09-01"))
        got4 = rows_of(build(base + [cv, cv2], d, "e.jsonl"))
        expect(tiers(got4).get(2048) == "converted" and tiers(got4).get(64) == "deep"
               and got4[("L", 2048, 1, 128)][1] == [("tiny", 1.0), ("cta", 2.0)], f"converted rows -> {tiers(got4)}")
        # a single runnable candidate, ranked but untimed, is an untimed measured row
        single = cell(r4, "deep", 64, ranked=("cta",), uplo="U")
        single["m1"] = None
        got5 = rows_of(build([run_line(r4, "2026-10-04"), single], d, "f.jsonl"))
        expect(got5 == {("U", 64, 1, 128): ("deep", [("cta", None)])}, f"single candidate untimed row -> {got5}")
        # hidden axes: a constant extra field projects away, a varied one is refused
        hid = [dict(cell(r1, "deep", n), key=f"uplo=L,n={n},nrhs=1,batch=128,diag={g}") for n, g in ((8, "N"), (16, "N"))]
        expect(len(rows_of(build([run_line(r1, "2026-10-01")] + hid, d, "g.jsonl"))) == 2, "constant hidden axis")
        hid[1]["key"] = hid[1]["key"].replace("diag=N", "diag=U")
        try:
            rows_of(build([run_line(r1, "2026-10-01")] + hid, d, "h.jsonl"))
            expect(False, "a varied hidden axis was accepted")
        except SystemExit as e:
            expect("diag" in str(e), f"varied hidden axis message {e}")
        # a skipped candidate whose family changed leaves the cell current; a ranked one does not
        expect(freshness(read_cell(cell(r1, "deep", 64, hashes=dict(fh, blocked="old"))), fh) == "current"
               and freshness(read_cell(cell(r1, "deep", 64, hashes=dict(fh, cta="old"))), fh) == "partly_stale",
               "skipped candidate hash is ignored")
        # rehash: a stored hash 'from' reads as 'to', for that family only
        old = cell(r1, "deep", 64, hashes=dict(fh, tiny="v1tiny"))
        rh = {"kind": "rehash", "family": "tiny", "from": "v1tiny", "to": "htiny", "at": "abc", "why": "t"}
        expect(rows_of(build([run_line(r1, "2026-10-01"), old], d, "i.jsonl")) == {}, "v1 hash is stale before rehash")
        got6 = rows_of(build([run_line(r1, "2026-10-01"), old, rh], d, "j.jsonl"))
        expect(tiers(got6) == {64: "deep"}, f"rehash makes the record current -> {got6}")
        rh_other = dict(rh, family="cta")
        expect(rows_of(build([run_line(r1, "2026-10-01"), old, rh_other], d, "k.jsonl")) == {}, "rehash is per family")
        chain = [dict(rh, to="v2old"), dict(rh, **{"from": "v2old", "to": "htiny"})]
        got7 = rows_of(build([run_line(r1, "2026-10-01"), old] + chain, d, "k2.jsonl"))
        expect(tiers(got7) == {64: "deep"}, f"a rehash chain v1 -> v2old -> now is followed -> {got7}")
        # a dirty run is excluded with a warning, per (run id, dtype)
        dl = [run_line(r1, "2026-10-01", batchlas="abc-dirty"), cell(r1, "deep", 64),
              run_line(r1, "2026-10-01", dtype="double"), cell(r1, "deep", 64, dtype="double")]
        ld = read_ledger(build(dl, d, "l.jsonl"))
        expect([c["dtype"] for c in ld.cells] == ["double"] and len(ld.runs) == 1
               and "abc-dirty: excluded" in " ".join(ld.warnings), f"dirty run {ld}")
        # the dtype filter, a repeated run line and a truncated last line
        mixed = base + [run_line(r1, "2026-10-01", dtype="double"), cell(r1, "deep", 64, dtype="double")]
        pm = build(mixed, d, "o.jsonl")
        expect(len(read_ledger(pm, "double").cells) == 1 and len(read_ledger(pm).cells) == 7, "dtype filter")
        with open(pm, "a") as f:
            f.write('{"kind": "cell", "dty')
        expect(len(read_ledger(pm).warnings) == 1, "a torn last line warns")
        # the shared fixtures (tune_tests reads the same files): all-error and dominated records
        fix = os.path.join(REPO, "tests", "data", "ledger_all_error", "posv.sm_0.jsonl")
        got7 = ledger_rows(keyspec, read_ledger(fix, "float").cells, fh)
        expect(got7 == {("L", 64, 1, 128): ("preview", [("tiny", 1.0), ("cta", 2.0)])},
               f"all-error record counts as no record -> {got7}")
        dom = read_ledger(os.path.join(REPO, "tests", "data", "ledger_dominated", "posv.sm_0.jsonl"), "float").cells
        gotd = [(sorted(stale_families(c, fh)), freshness(c, fh)) for c in dom]
        expect(gotd == [(["blocked"], "partly_stale"), ([], "current")], f"dominated skip -> {gotd}")
        # an unfetched LFS pointer is named
        lfs = build([], d, "p.jsonl")
        with open(lfs, "w") as f:
            f.write(LFS_POINTER + "\noid sha256:00\nsize 12\n")
        try:
            read_ledger(lfs)
            expect(False, "an LFS pointer was read")
        except SystemExit as e:
            expect("git lfs pull" in str(e), f"LFS pointer message {e}")
        # a whole table: header, dtype key, byte-reproducible --check, a hand edit and a header hash judged
        repo = os.path.join(d, "repo")
        os.makedirs(os.path.join(repo, "tools", "tune"))
        os.makedirs(os.path.join(repo, "src"))
        os.makedirs(os.path.join(repo, "tuned"))
        os.makedirs(os.path.join(repo, "ledger"))
        with open(os.path.join(repo, "src", "k.cc"), "w") as f:
            f.write("int k; // kernel\n")
        with open(os.path.join(repo, "tools", "tune", "posv_spec.cc"), "w") as f:
            f.write('// kernel-sources-begin\n"src/k.cc",\n// kernel-sources-end\n')
        hk = kernel_hash(["src/k.cc"], repo)
        full = [{**x, **{f"h{i}": hk for i in range(3)}} if x["kind"] == "cell" else x for x in mixed]
        lp = build(full, os.path.join(repo, "ledger"), "posv.sm_0.jsonl")
        der = ledger_text("posv", "sm_0", lp, repo=repo)
        lines = der.text.splitlines()
        expect(lines[0] == f"# op=posv device=sm_0 batchlas=bl3 kernels={hk} family_kernels=tiny:{hk},cta:{hk},"
               f"blocked:{hk} date=2026-10-03", f"header line 1 -> {lines[0]!r}")
        expect(lines[1].endswith(" tiers=deep:2,coarse:1,calibrate:0,preview:2,custom:0,converted:0,prior:0,transcribed:0"),
               f"line 2 {lines[1]!r}")
        expect(lines[2] == f"# keys: dtype:exact {keys}", f"line 3 {lines[2]!r}")
        expect(lines[3].startswith("dtype=float ") and lines[-1].startswith("dtype=double uplo=L n=64 "), "dtype order")
        tab = os.path.join(repo, "tuned", "posv.sm_0.txt")
        for text, want_fail in ((der.text, False), (der.text.replace("tiny 1.000", "tiny 1.001", 1), True)):
            with open(tab, "w") as f:
                f.write(text)
            failures, notes = check(os.path.join(repo, "tuned"), os.path.join(repo, "ledger"), repo)
            expect(bool(failures) == want_fail, f"check on {'an edited' if want_fail else 'a fresh'} table -> {failures}")
            expect(not notes, f"fresh sources give no stale warning -> {notes}")
        with open(tab, "w") as f:
            f.write(der.text)
        with open(os.path.join(repo, "src", "k.cc"), "w") as f:
            f.write("int k;  /* comments and spaces do not count */\n")
        failures, notes = check(os.path.join(repo, "tuned"), os.path.join(repo, "ledger"), repo)
        expect(not failures and not notes, f"a comment-only edit changes nothing -> {failures} {notes}")
        with open(os.path.join(repo, "src", "k.cc"), "w") as f:
            f.write("int k2;\n")
        failures, notes = check(os.path.join(repo, "tuned"), os.path.join(repo, "ledger"), repo)
        expect(not failures and len(notes) == 1 and "stale" in notes[0],
               f"a kernel edit is a stale warning, not a failure -> {failures} {notes}")
        # staleness by tier: an all-transcribed table is never stale; a measured one names its stale family
        os.makedirs(os.path.join(repo, "fix"))
        hk2 = kernel_hash(["src/k.cc"], repo)
        one = "deep:{},coarse:{},calibrate:0,preview:{},custom:{},converted:0,prior:0,transcribed:5".format
        for name, fams, tier_counts, want in (
                ("posv.sm_t.txt", "tiny:deadbeef",
                 "deep:0,coarse:0,calibrate:0,preview:0,custom:0,converted:4,prior:2,transcribed:3", []),
                ("posv.sm_d.txt", f"tiny:{hk2},cta:deadbeef", one(5, 0, 0, 0), ["cta:deadbeef"]),
                ("posv.sm_c.txt", f"tiny:{hk2},cta:deadbeef", one(0, 2, 0, 0), ["cta:deadbeef"]),
                ("posv.sm_p.txt", f"tiny:{hk2},cta:deadbeef", one(0, 0, 2, 0), ["cta:deadbeef"]),
                ("posv.sm_u.txt", f"tiny:{hk2},cta:deadbeef", one(0, 0, 0, 2), ["cta:deadbeef"]),
                ("posv.sm_m.txt", f"tiny:{hk2},cta:deadbeef", None, ["cta:deadbeef"]),
                ("posv.sm_x.txt", f"tiny:{hk2},cta:deadbeef", "deep:x,transcribed:5", ["cta:deadbeef"])):
            fp = os.path.join(repo, "fix", name)
            with open(fp, "w") as f:
                f.write(f"# op=posv device=x batchlas=b kernels=deadbeef family_kernels={fams} date=d\n"
                        f"# source=ledger:x{'' if tier_counts is None else ' tiers=' + tier_counts}\n"
                        "# keys: dtype:exact n:log\n")
            got = stale_kernel_warnings(fp, repo)
            expect(len(got) == len(want) and all(w in g and "tiny:" not in g for w, g in zip(want, got)),
                   f"stale_kernel_warnings({name}) -> {got}")
        fp = os.path.join(repo, "fix", "gemv.sm_d.txt")
        with open(fp, "w") as f:
            f.write("# op=gemv device=x kernels=deadbeef family_kernels=a:b date=d\n# source=ledger:x tiers=deep:1\n")
        got = stale_kernel_warnings(fp, repo)
        expect(len(got) == 1 and got[0].startswith("note:"), f"an op without a spec is a note -> {got}")
        os.remove(tab)
        failures, _ = check(os.path.join(repo, "tuned"), os.path.join(repo, "ledger"), repo)
        expect(any("no table" in x for x in failures), f"a ledger without a table fails -> {failures}")
        # the sample proof of a rehash: agree, a tie within TIE, a disagreement, too few cells, a missing combination
        with open(os.path.join(repo, "src", "k.cc"), "w") as f:
            f.write("int k; // kernel\n")

        def sample_ledger(name, sample_ns, flip=None, upper=False, remeasured=()):
            old = {f"h{i}": "v1old" for i in range(3)}
            lines = [run_line(r1, "2026-10-01"), run_line("S", "2026-10-05", tier="custom")]
            lines += [dict(cell(r1, "deep", n), **old) for n in range(1, 65)]
            if remeasured:
                lines.append(run_line("D", "2026-10-06"))
            lines += [{**cell("D", "deep", n, date="2026-10-06"), **{f"h{i}": hk for i in range(3)}} for n in remeasured]
            if upper:
                lines.append(dict(cell(r1, "deep", 1, uplo="U"), **old))
            for n in sample_ns:
                c = {**cell("S", "custom", n, date="2026-10-05"), **{f"h{i}": hk for i in range(3)}}
                if n in (flip or {}):  # the sample's own times: cta first by the given ratio
                    c.update(ranked="cta|tiny", m0=flip[n], m1=1.0)
                lines.append(c)
            return build(lines, os.path.join(repo, "ledger"), name)

        ok, msg, recs = rehash_sample(sample_ledger("s1.jsonl", range(1, 65), {7: 1.02}), "posv", "S", repo=repo, write=False)
        expect(ok and len(recs) == 3 and recs[0]["proof"] == "sample:S" and recs[0]["from"] == "v1old"
               and recs[0]["to"] == hk and "64 cells agree" in recs[0]["why"], f"rehash sample agrees (tie at n=7) -> {msg}")
        ok, msg, _ = rehash_sample(sample_ledger("s2.jsonl", range(1, 65), {7: 1.5}), "posv", "S", repo=repo)
        expect(not ok and any("n=7" in m and "old tiny, now cta" in m for m in msg), f"rehash sample disagrees -> {msg}")
        ok, msg, recs = rehash_sample(sample_ledger("s2b.jsonl", range(1, 65), {7: 1.5}, remeasured=(7,)), "posv", "S",
                                      repo=repo, write=False)
        expect(ok and "63 cells agree, 1 re-measured at deep" in recs[0]["why"],
               f"rehash sample: a disagreeing cell re-measured at deep is settled -> {msg}")
        ok, msg, _ = rehash_sample(sample_ledger("s3.jsonl", range(1, 11)), "posv", "S", repo=repo)
        expect(not ok and "fewer than 64" in msg[0], f"rehash sample too small -> {msg}")
        ok, msg, _ = rehash_sample(sample_ledger("s4.jsonl", range(1, 65), upper=True), "posv", "S", repo=repo)
        expect(not ok and "misses 1 exact-key" in " ".join(msg), f"rehash sample misses uplo=U -> {msg}")
        s5 = sample_ledger("s5.jsonl", range(1, 65))
        ok, _, _ = rehash_sample(s5, "posv", "S", repo=repo)
        rows5 = ledger_rows(keyspec, read_ledger(s5, "float").cells, family_hashes(parse_kernel_block(
            read_spec_source("posv", repo)), cands, repo))
        expect(ok and rows5[("L", 5, 1, 128)][0] == "deep", f"after the rehash the deep records stand -> {ok}")
        # the stratified sample: posv keys under op potrf, 4 dtypes x 2 uplo x n=1..40 x batch 32|128|1024,
        # stale by one hash
        with open(os.path.join(repo, "tools", "tune", "potrf_spec.cc"), "w") as f:
            f.write('// kernel-sources-begin\n"src/k.cc",\n// kernel-sources-end\n')
        old = {f"h{i}": "v1old" for i in range(3)}
        pl = [run_line(r1, "2026-10-01", dtype=t) for t in DTYPES]
        pl += [dict(cell(r1, "deep", n, dtype=t, uplo=u, batch=b), **old)
               for t in DTYPES for u in "LU" for n in range(1, 41) for b in (32, 128, 1024)]
        pp = build(pl, os.path.join(repo, "ledger"), "potrf.sm_s.jsonl")
        sample = emit_sample(pp, "potrf")
        got = [ln.split() for ln in sample]
        kv = [(t, dict(x.split("=") for x in k.split(","))) for o, t, k in got]
        seen = defaultdict(set)  # the stratified picks alone (no fill) must reach every end
        for t, k in [(t, dict(x.split("=") for x in k.split(","))) for _, t, k in
                     (ln.split() for ln in emit_sample(pp, "potrf", 0))]:
            seen[t, k["uplo"]] |= {("n", k["n"]), ("batch", k["batch"])}
        ends = {("n", "1"), ("n", "40"), ("batch", "32"), ("batch", "1024")}
        expect(len(set(sample)) == len(sample) >= SAMPLE_MIN_CELLS and all(o == "potrf" for o, _, _ in got)
               and set(seen) == {(t, u) for t in DTYPES for u in "LU"} and all(ends <= v for v in seen.values()),
               f"emit_sample covers 4 dtypes x 2 uplo, both ends of n and batch, >= 64 distinct cells -> {len(sample)} "
               f"{sorted((g, sorted(ends - v)) for g, v in seen.items())}")
        expect(len(emit_sample(pp, "potrf", 200)) == 200, "emit_sample --cells 200")
        pl.append(run_line("S", "2026-10-05", tier="custom"))
        for t, k in kv:
            pl.append({**cell("S", "custom", int(k["n"]), date="2026-10-05", dtype=t, uplo=k["uplo"], batch=int(k["batch"])),
                       **{f"h{i}": hk for i in range(3)}})
        ok, msg, _ = rehash_sample(build(pl, os.path.join(repo, "ledger"), "potrf.sm_s.jsonl"), "potrf", "S",
                                   repo=repo, write=False)
        expect(ok, f"--rehash-sample accepts the emitted sample -> {msg}")
        bad += moved_self_test(d, run_line, cell, cands)
        bad += renamed_self_test(d, run_line, cell, cands)
        # row kinds
        expect(row_kind_problems({"source": "ledger:x"}, [(("L",), [("a", None), ("b", 1.0)])]) != [], "mixed row")
        expect(row_kind_problems({"source": "x"}, [(("L",), [("a", None)])]) != [], "untimed row needs ledger:")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="re-derive every table in tuned/ from its ledger")
    ap.add_argument("--self-test", action="store_true", help="run the generator self-test only")
    ap.add_argument("--ledger", default=os.path.join(REPO, LEDGER_ROOT), help="ledger root (default %(default)s)")
    ap.add_argument("--op", help="only this op's ledgers")
    ap.add_argument("--device", help="only this device's ledgers")
    ap.add_argument("--out", default=os.path.join(REPO, TUNED), help="table directory (default tuned/)")
    ap.add_argument("--rehash-sample", metavar="RUN_ID",
                    help="with --op and --device: rehash the stale families if this sample run agrees with the old records")
    ap.add_argument("--rehash-moved", metavar="BASE",
                    help="rehash every family whose sources only moved since commit BASE (all ledgers, or --op/--device)")
    ap.add_argument("--rehash-renamed", metavar="BASE",
                    help="rehash every family whose sources only renamed template parameters since commit BASE "
                         "(all ledgers, or --op/--device)")
    ap.add_argument("--family", nargs="+", help="--rehash-sample: only these families")
    ap.add_argument("--at", help="--rehash-sample, --rehash-moved, --rehash-renamed: the commit to record (default HEAD)")
    ap.add_argument("--emit-sample", metavar="OUT",
                    help="with --op and --device: write a stratified --remeasure-keys file for a --rehash-sample run")
    ap.add_argument("--remeasure-out", metavar="FILE",
                    help="with --check or a table write: the cells monotonicity() flags, as a --remeasure-keys file")
    ap.add_argument("--cells", type=int, default=SAMPLE_MIN_CELLS,
                    help="--emit-sample: this many cells (default %(default)s); a ledger with fewer exits 1")
    args = ap.parse_args()
    if args.emit_sample:
        if not args.op or not args.device:
            raise SystemExit("--emit-sample needs --op and --device")
        lines = emit_sample(os.path.join(args.ledger, f"{args.op}.{args.device}.jsonl"), args.op, args.cells)
        with open(args.emit_sample, "w") as fh:
            fh.write("".join(ln + "\n" for ln in lines))
        print(f"wrote {args.emit_sample}: {len(lines)} cells")
        if len(lines) < SAMPLE_MIN_CELLS:
            print(f"error: fewer than {SAMPLE_MIN_CELLS} measured cells; --rehash-sample would refuse this sample")
            return 1
        return 0
    if args.rehash_sample:
        if not args.op or not args.device:
            raise SystemExit("--rehash-sample needs --op and --device")
        path = os.path.join(args.ledger, f"{args.op}.{args.device}.jsonl")
        ok, msg, _ = rehash_sample(path, args.op, args.rehash_sample, args.family, args.at)
        print("\n".join(msg))
        return 0 if ok else 1
    if args.rehash_moved or args.rehash_renamed:
        if args.rehash_moved and args.rehash_renamed:
            raise SystemExit("--rehash-moved and --rehash-renamed are separate proofs: run one at a time")
        prove, base = (rehash_moved, args.rehash_moved) if args.rehash_moved else (rehash_renamed, args.rehash_renamed)
        refused = 0
        for (o, d), path in ledger_files(args.ledger, args.op, args.device):
            report, lines = prove(path, o, base, args.at)
            refused += sum("no proof" in r for r in report)
            print(f"{o}.{d}: {len(lines)} rehash lines\n" + "".join(f"  {r}\n" for r in report), end="")
        return 1 if refused else 0
    if args.self_test:
        bad = self_test()
        print("\n".join(bad + [f"--self-test: {'FAILED' if bad else 'OK'}"]))
        return 1 if bad else 0
    remeasure = []

    def write_remeasure():
        if args.remeasure_out:
            with open(args.remeasure_out, "w") as fh:
                fh.write("".join(ln + "\n" for ln in remeasure))
            print(f"wrote {args.remeasure_out}: {len(remeasure)} cells (batchlas_tune --remeasure-keys)")
    if args.check:
        failures, notes = check(args.out, args.ledger, remeasure=remeasure)
        for n in notes:
            print(n)
        for f in failures:
            print("FAIL:", f)
        write_remeasure()
        print(f"\n--check: {'FAILED, ' + str(len(failures)) + ' problems' if failures else 'OK'}")
        return 1 if failures else 0
    write_tables(args.ledger, args.out, args.op, args.device, remeasure)
    write_remeasure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
