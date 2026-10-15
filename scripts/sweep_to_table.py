#!/usr/bin/env python3
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

The one input is benchmarks/results/tuning/ledger/<op>.<device>.jsonl (--ledger DIR names another
root). Every line carries "dtype" except `rehash`. Keys and candidates come from each dtype's newest
`run` line (date, then run id), never from this script. Per dtype and cell the best record wins:
tier deep > coarse > preview > custom > converted > transcribed, then newest date, run id, later
line; a record whose winner's family hash is stale, or whose every candidate is "error", is no
record. converted and transcribed records carry no hash and are never stale. A lower-tier row is
dropped when a higher-tier row has equal exact keys and lies within the tier's stride (deep and
coarse 1, the rest 2, in index steps of the per-key lattice of the dtype's current measured cells)
on every log key; converted and transcribed rows are dropped near any measured row, kept or not.

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
LEDGER_TIERS = ("deep", "coarse", "preview", "custom", "converted", "transcribed")  # precedence, highest first
LEDGER_STRIDE = {"deep": 1, "coarse": 1, "preview": 2, "custom": 2, "converted": 2, "transcribed": 2}
UNHASHED = ("converted", "transcribed")
LEDGER_STATUS_TIMED = ("ok", "eliminated")
TIE = 0.03  # the tuner's tie band (tune_core.cc rank())
LFS_POINTER = "version https://git-lfs.github.com/spec/v1"  # the first line of an unfetched LFS file
HASH_VECTOR = "tests/data/kernel_hash_v2"


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
              "hi": rec.get(f"hi{i}"), "reps": rec.get(f"n{i}")}
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
    """Per key, skipping all-error and stale records, the highest tier, then newest date, then
    larger run id, then the later line (the C++ best_records)."""
    def order(c):
        return (LEDGER_TIERS[::-1].index(c["tier"]), c["date"], c["run_id"])
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
    on every log key); converted and transcribed rows are bracketed by any measured row."""
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
            against = [k for k, t, _ in cand_rows if t not in UNHASHED]
        else:
            against = list(kept)
        near = bracketed_by(against, LEDGER_STRIDE[tier])
        for key, t, entries in cand_rows:
            if t != tier or key in kept:
                continue
            if not near(key):
                kept[key] = (tier, entries)
    return kept


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
    body = []
    for d in dtypes:
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
    return Derived("\n".join(lines + body) + "\n", counts, led.warnings, vocab, keys)


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


def check_table(path, failures, notes, repo=REPO):
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
    A table with no measured row (tiers= counts only converted/transcribed) carries no hash and is
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


def check(tuned_dir, ledger_root, repo=REPO):
    failures, notes = [], []
    names = sorted(f for f in os.listdir(tuned_dir) if not f.startswith("."))
    covered = set()
    for name in names:
        if name == "README.md" or os.path.isdir(os.path.join(tuned_dir, name)):
            continue
        led = check_table(os.path.join(tuned_dir, name), failures, notes, repo)
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


def write_tables(root, out_dir, op=None, device=None):
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


INCLUDE = re.compile(rb'^([ \t]*#[ \t]*include[ \t]*)"([^"\n]*)"', re.M)
ANGLE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*<([^>\n]*)>', re.M)
ANY_INCLUDE = re.compile(rb'^[ \t]*(?:#|%:)[ \t]*include', re.M)
# A file a move can retarget without the "x" text changing: refused, never modelled.
UNMODELLED = re.compile(rb'__has_include|include_next|R"[^(\s"]{0,16}\([^)]*?#[ \t]*include')


def unmodelled_include(data, path, rename):
    """Why `data`'s includes are outside the moved proof's model, or None."""
    if UNMODELLED.search(data):
        return f"{path}: __has_include, #include_next or an include inside a raw string"
    if len(ANY_INCLUDE.findall(data)) != len(INCLUDE.findall(data)) + len(ANGLE.findall(data)):
        return f"{path}: an #include that is neither \"x\" nor <x> (macro, comment or digraph)"
    for m in ANGLE.findall(data):
        if os.path.join("include", m.decode()) in rename:
            return f"{path}: <{m.decode()}> names a moved header"
    return None


def canonical_includes(data, path, tree, rename=None):
    """Every #include "x" -> the repo path it resolves to (its own directory, then include/), mapped
    through rename; an unresolved one is kept verbatim, marked."""
    def sub(m):
        target = m.group(2).decode()
        for cand in (os.path.join(os.path.dirname(path), target), os.path.join("include", target)):
            cand = os.path.normpath(cand)
            if cand in tree.paths:
                return m.group(1) + b'"@' + (rename or {}).get(cand, cand).encode() + b'"'
        return m.group(1) + b'"?' + m.group(2) + b'"'
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
        why = unmodelled_include(old, b, rename) or unmodelled_include(new, h, rename)
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
    led = read_ledger(ledger_path, raw=True)
    stored = defaultdict(set)
    for c in led.cells:
        for r in c["cands"] if c["tier"] not in UNHASHED else []:
            if r["hash"]:
                stored[spelling_family(r["cand"])].add(r["hash"])
    if not stored:
        return ["no measured records"], []
    bt, ht = GitTree(repo, base), GitTree(repo, "HEAD")
    rename = git_renames(repo, base, "HEAD")
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
        why = moved_difference(bp, hp, bt, ht, rename)
        if why:
            report.append(f"{f}: no proof: {why}")
            continue
        moved = sum(b != h for b, h in zip(bp, hp))
        lines += [{"kind": "rehash", "family": f, "from": h, "to": h_now, "at": at or short[1],
                   "proof": f"moved:{short[0]}",
                   "why": f"{moved} files moved, equal modulo comments, whitespace and include paths"} for h in todo]
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


def moved_self_test(d, run_line, cell, cands):
    """--rehash-moved on throwaway git repos: a pure move, an include that now names another file,
    a content edit, a longer source list, an uncommitted change and the unmodelled include forms."""
    bad = []
    filler = "".join(f"int f{i}(int);\n" for i in range(24)).encode()
    spec = '// kernel-sources-begin\n"{k}",{extra}\n// family: tiny\n"{t}",\n// kernel-sources-end\n'

    def git(repo, *args):
        subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        *args], check=True, capture_output=True)

    def put(repo, path, data):
        os.makedirs(os.path.dirname(os.path.join(repo, path)), exist_ok=True)
        with open(os.path.join(repo, path), "wb") as f:
            f.write(data)

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
            inc = b'#include "../../common/util.hh"\n'
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
    for name, want in (("other", "differs"), ("edit", "differs"), ("length", "source list length"),
                       ("macro", "neither"), ("has", "__has_include")):
        repo, path, _ = case(name, name)
        report, recs = rehash_moved(path, "posv", "HEAD~1", repo=repo)
        expect(not recs and any("no proof" in r and want in r for r in report), f"{name} -> {report}")
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
        expect(lines[1].endswith(" tiers=deep:2,coarse:1,preview:2,custom:0,converted:0,transcribed:0"), f"line 2 {lines[1]!r}")
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
        one = "deep:{},coarse:{},preview:{},custom:{},converted:0,transcribed:5".format
        for name, fams, tier_counts, want in (
                ("posv.sm_t.txt", "tiny:deadbeef", "deep:0,coarse:0,preview:0,custom:0,converted:4,transcribed:3", []),
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
    ap.add_argument("--family", nargs="+", help="--rehash-sample: only these families")
    ap.add_argument("--at", help="--rehash-sample, --rehash-moved: the commit to record (default HEAD)")
    ap.add_argument("--emit-sample", metavar="OUT",
                    help="with --op and --device: write a stratified --remeasure-keys file for a --rehash-sample run")
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
    if args.rehash_moved:
        refused = 0
        for (o, d), path in ledger_files(args.ledger, args.op, args.device):
            report, lines = rehash_moved(path, o, args.rehash_moved, args.at)
            refused += sum("no proof" in r for r in report)
            print(f"{o}.{d}: {len(lines)} rehash lines\n" + "".join(f"  {r}\n" for r in report), end="")
        return 1 if refused else 0
    if args.self_test:
        bad = self_test()
        print("\n".join(bad + [f"--self-test: {'FAILED' if bad else 'OK'}"]))
        return 1 if bad else 0
    if args.check:
        failures, notes = check(args.out, args.ledger)
        for n in notes:
            print(n)
        for f in failures:
            print("FAIL:", f)
        print(f"\n--check: {'FAILED, ' + str(len(failures)) + ' problems' if failures else 'OK'}")
        return 1 if failures else 0
    write_tables(args.ledger, args.out, args.op, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
