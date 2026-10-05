#!/usr/bin/env python3
"""Write and verify the tuned selection tables (tuned/<op>.<dtype>.<device>.txt).

Spec: docs/design/flat-kernel-selection.md section 7 (conversion rules), section 6.3
(tie rule), section 5.4 (file format and nearest-point lookup) and section 13 (transcribed
tables); flat-kernel-selection-phase3-plan.md section 2 option D.

    scripts/sweep_to_table.py                        # convert every op's sweeps, write tuned/
    scripts/sweep_to_table.py --transcribe CSV...    # write untimed tables from transcriber CSVs
    scripts/sweep_to_table.py --check                # re-derive and diff every table, check nearest()
    scripts/sweep_to_table.py --check --check-points pts.txt
    scripts/sweep_to_table.py --self-test            # converter rules on real rows (also run by --check)
    scripts/sweep_to_table.py --tuner JSONL... [--out DIR]   # tables from batchlas_tune raw output

A table comes from one of three sources:

* converted: forced-route sweeps (JSONL) in benchmarks/results/routing/, one timed ranked
  list per measured cell, "uplo=L n=64 batch=8192 | lpanel:panel=8 0.302 | vendor 0.490".
* transcribed: the old router's preference order, evaluated at every grid cell by a per-op
  C++ transcriber, "uplo=L n=8 nrhs=1 batch=128 | tiny - | cta - | blocked -". The header
  carries source=transcribed:<sha> and transcriber_csv=<path>.
* tuner: tools/tune/batchlas_tune's raw JSONL (schema in tools/tune/README.md). The header
  carries kernels=<hash> from the run and source=tuner:<jsonl>. Rows are formatted by the
  same code as converted ones (table_text), so tuner and converted tables differ only in
  their header. The tuner pins each candidate with select::ScopedPin, and a concrete pin
  either runs or throws (rule R6), so its rows need no reached-route readback.

--check is the pure-data half of the acceptance gate (section 10.2). Every file in tuned/
must belong to a registered op and equal what its source produces (a transcribed table
whose CSV is absent is only self-checked), every spelling must be in the op's vocabulary,
and the section 5.4 lookup must return each row's own key, with the measured winner first
on a timed row. Row kinds follow the C++ loader: a row is all-timed or all-untimed, an
untimed row needs source=transcribed:<hex sha>, and a table may hold both kinds. For potrf
it also prints where the section 10.3 off-grid points land; a points file holds one "n" or
"n batch" per line. --self-test checks these rules and the sweep reader against rows copied
from the live sweeps (tests/data/sweep_to_table_rows.jsonl).

THE FRAMEWORK. Everything op-specific is one OpSpec in OPS:
  op               table name prefix and the header's op=
  keys             the '# keys:' line, identical to the op's choice.hh key_names
  row_ops          sweep-row "op" values accepted (e.g. potrf, potrf_upper)
  row_key          sweep row -> key tuple in '# keys:' order (exact keys str, log keys int),
                   or None when malformed
  arm_spelling     sweep "arm" -> choice spelling (conversion rule 4)
  arm_route        sweep "arm" -> the reached-route spellings that confirm it (rule 1)
  route_field      the sweep field holding the reached route ("route"; posv's driver: "reached")
  keep_not_ok      ok=false rows still measured, kept and marked # noisy (posv: relsd_only)
  candidate_order  choice.hh candidate-list order: the section 6.3 tie order and the vocabulary
  sources          converted inputs per device; a source whose first file is absent is pending,
                   one with adopted=False is read and validated but writes no table yet
  review           optional extra --check report
  tuner_key        tuner "pass" record -> key tuple (default: the '# keys:' fields by name)

To add an op: write its OpSpec (copy POSV), append it to OPS, run without flags to write the
converted tables or --transcribe for untimed ones, then --check.

SWEEP JSONL (converted input), one JSON object per line. Fields read:
  op        str    one of row_ops; "<op>_upper" means uplo=U when "uplo" is absent
  dtype     str    float | double | cfloat | cdouble (complex<float>/complex<double> accepted)
  uplo      str    "L" | "U" (also "Lower" | "Upper"; optional, see op)
  n, batch  int    and for posv nrhs (int); potrf rows also carry m, which must equal n
  arm       str    the pinned arm: potrf "route:native:<tier>" | "vendor"; posv
                   "route:native:<tier>" or the bare factor_bench arm "<tier>"
  pass      int    pass number; (file, cell, arm, pass) must be unique
  time_ms   float  median ms per call for the whole batch
  ok        bool   false rows record a refused or failed arm and are dropped, except those
                   the op's keep_not_ok accepts. posv keeps reason == "relsd" exactly (its
                   only failed gate was rel_sd > 0.10) with info_nonzero 0 and a finite
                   residual; fallback, unsupported, residual, info and error rows still drop.
                   The reached route must still equal the pinned arm.
  route     str    reached coverage spelling chosen_origin:chosen_algo, e.g. "native:tiny";
                   posv rows carry it as "reached" (the field is the op's route_field)
  kernel_current bool  sm_89 archive only (rule 2)
A cell's time per arm is the mean of its pass medians; >10% spread marks the row "# noisy",
and so does any kept ok=false row. A non-pending source that yields no cell, or any row
whose key is malformed, is a hard failure: a drifted contract must not convert to nothing.
An unterminated last line (a sweep still appending) is skipped with a note.

TUNER JSONL (--tuner input), read by read_tuner: the first line is the "meta" record (op,
dtype, device, batchlas, kernels, date, keys, candidates, passes, reps, warm_s, ld_pad); the
"pass" records carry the op's key fields, cand, status, pass, attempt and median_ms. Per cell
only the highest attempt that timed anything counts (a re-measured cell replaces its first
measurement, unless every re-measure child failed); a
candidate counts when it is status "ok" in all `passes` passes of that attempt, and its time
is the mean of those pass medians, ">10% spread" marking the row "# noisy" as above. OpSpec
field tuner_key turns a record into the key tuple; the default reads each '# keys:' name.

TRANSCRIBER CSV (--transcribe input), with a header row:
  op,dtype,device,<one column per '# keys:' name>,ranked
  e.g.  posv,float,sm_89,L,8,1,128,tiny|cta|blocked
ranked is the old router's preference order, '|'-separated choice spellings, most
preferred first; it should list every candidate that router could pick at the cell.
One CSV may hold several (op, dtype, device) tables. The sha in the headers is --sha
(default: the repository HEAD), resolved with git rev-parse to an 8-digit hex sha, and
should name the commit whose router was transcribed.

Python 3 standard library only.
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTING = "benchmarks/results/routing"
TUNED = "tuned"

DTYPES = {
    "float": "float", "double": "double", "cfloat": "cfloat", "cdouble": "cdouble",
    "complex<float>": "cfloat", "complex<double>": "cdouble",
}

TIE = 0.03
NOISY = 0.10
TRANSCRIBED = "transcribed:"
HEX = re.compile(r"[0-9a-fA-F]+")  # the C++ parser's isxdigit rule for transcribed:<sha>


@dataclass
class Source:
    device: str
    files: list
    batchlas: str
    note: str
    current_only: bool = False
    adopted: bool = True


@dataclass
class OpSpec:
    op: str
    keys: str
    row_ops: tuple
    row_key: Callable
    arm_spelling: dict
    arm_route: dict
    candidate_order: list
    sources: list = field(default_factory=list)
    route_field: str = "route"
    keep_not_ok: Callable = lambda r: False
    review: Optional[Callable] = None
    tuner_key: Optional[Callable] = None


UPLO = {"L": "L", "Lower": "L", "lower": "L", "U": "U", "Upper": "U", "upper": "U"}


def uplo_of(r):
    if "uplo" in r:
        return UPLO.get(r["uplo"], r["uplo"])
    return "U" if r["op"].endswith("_upper") else "L"


def potrf_key(r):
    if r["m"] != r["n"]:
        return None
    return (uplo_of(r), int(r["n"]), int(r["batch"]))


def posv_key(r):
    try:
        key = (uplo_of(r), int(r["n"]), int(r["nrhs"]), int(r["batch"]))
    except (KeyError, TypeError, ValueError):
        return None
    return key if key[0] in ("L", "U") and min(key[1:]) >= 1 else None


# potrf: n weighs 3 because the work grows as n^3 and linearly in batch, so the distance
# approximates log-cost. The coverage spellings are the RouteTable era's, then flat
# selection's, whose chosen_algo is the choice spelling.
POTRF = OpSpec(
    op="potrf",
    keys="uplo:exact n:log:3 batch:log",
    row_ops=("potrf", "potrf_upper"),
    row_key=potrf_key,
    arm_spelling={
        "route:native:tiny": "tiny",
        "route:native:cta": "cta",
        "route:native:lpanel": "lpanel:panel=8",
        "route:native:blocked": "blocked",
        "vendor": "vendor",
    },
    arm_route={
        "route:native:tiny": ("native:tiny",),
        "route:native:cta": ("native:cta",),
        "route:native:lpanel": ("native:lpanel", "native:lpanel:panel=8"),
        "route:native:blocked": ("native:blocked",),
        "vendor": ("vendor:auto", "vendor:vendor"),
    },
    candidate_order=["tiny", "cta", "lpanel:panel=8", "lpanel:panel=16", "blocked", "vendor"],
    sources=[
        Source("sm_120", ["sm120_potrf_sweep.jsonl", "sm120_potrf_sweep_edges.jsonl"],
               "a1063892", "converted, passes 1+2"),
        Source("sm_89", ["sm89_potrf_archive.jsonl"], "unknown",
               "converted, kernel_current rows only; repeated medians averaged", current_only=True),
    ],
)

def relsd_only(r):
    """factor_bench's ok also fails rel_sd > 0.10 (launch-bound n=1 cells); its reasons join
    with '+', so reason == "relsd" means residual and info passed: still a measurement."""
    res = r.get("residual")
    return (r.get("reason") == "relsd" and r.get("info_nonzero") == 0
            and isinstance(res, (int, float)) and math.isfinite(res))


# posv (plan section 1.1): flops n^3/3 + 2 n^2 nrhs. No vendor family. sm_89 is transcribed.
POSV_TIERS = ("tiny", "cta", "blocked")
POSV = OpSpec(
    op="posv",
    keys="uplo:exact n:log:3 nrhs:log batch:log",
    row_ops=("posv", "posv_upper"),
    row_key=posv_key,
    arm_spelling={**{f"route:native:{t}": t for t in POSV_TIERS}, **{t: t for t in POSV_TIERS}},
    arm_route={**{f"route:native:{t}": (f"native:{t}",) for t in POSV_TIERS},
               **{t: (f"native:{t}",) for t in POSV_TIERS}},
    candidate_order=list(POSV_TIERS),
    # TODO(P3.1): once the sweep has finished, set batchlas (the sweep binary is a snapshot
    # of 886537e8) and adopted=True, then write the tables.
    sources=[Source("sm_120", ["sm120_posv_sweep.jsonl"], "unknown", "converted, passes 1+2",
                    adopted=False)],
    route_field="reached",
    keep_not_ok=relsd_only,
)


def trsm_key(r):
    try:
        key = (str(r["side"]), str(r["trans"]), int(r["order"]), int(r["q"]), int(r["batch"]))
    except (KeyError, TypeError, ValueError):
        return None
    return key if key[0] in ("L", "R") and key[1] in ("N", "T") and min(key[2:]) >= 1 else None


def trsm_tuner_key(r):
    """uplo and diag are hidden tuner axes for the invariance A/B (tools/tune/trsm_spec.cc):
    a run that varied them would fold several cells into one row, so it never becomes a table."""
    if r.get("uplo", "L") != "L" or r.get("diag", "N") != "N":
        raise SystemExit(f"trsm tuner record with uplo={r.get('uplo')} diag={r.get('diag')}: "
                         "an uplo/diag A/B run is raw data, not a table")
    return (str(r["side"]), str(r["trans"]), int(r["order"]), int(r["q"]), int(r["batch"]))


# trsm (plan section 1.2): work ~ order^2 q batch. ConjTrans folds to T; uplo and diag are not
# keys. No sweep source: sm_89 is transcribed, sm_120 comes from the tuner (--tuner).
TRSM_CHOICES = ("cta", "sg_left", "blocked", "vendor")
TRSM = OpSpec(
    op="trsm",
    keys="side:exact trans:exact order:log:2 q:log batch:log",
    row_ops=("trsm",),
    row_key=trsm_key,
    arm_spelling={c: c for c in TRSM_CHOICES},
    arm_route={c: (("vendor:vendor",) if c == "vendor" else (f"native:{c}",)) for c in TRSM_CHOICES},
    candidate_order=list(TRSM_CHOICES),
    tuner_key=trsm_tuner_key,
)


def orgqr_key(r):
    try:
        key = (int(r["m"]), int(r["n"]))
    except (KeyError, TypeError, ValueError):
        return None
    return key if min(key) >= 1 else None


# orgqr (phase 5): work ~ m n^2; the old predicates read m and n only (no batch, no arch), so
# sm_89 and sm_120 carry the same transcription. No sweep source.
ORGQR_CHOICES = ("blocked", "vendor")
ORGQR = OpSpec(
    op="orgqr",
    keys="m:log n:log:2",
    row_ops=("orgqr",),
    row_key=orgqr_key,
    arm_spelling={c: c for c in ORGQR_CHOICES},
    arm_route={c: (("vendor:vendor",) if c == "vendor" else (f"native:{c}",)) for c in ORGQR_CHOICES},
    candidate_order=list(ORGQR_CHOICES),
)


def parse_keys(spec):
    """'# keys:' text -> [(name, is_log, weight)]; a :log weight defaults to 1."""
    out = []
    for tok in spec.split():
        part = tok.split(":")
        is_log = part[1] == "log"
        out.append((part[0], is_log, float(part[2]) if is_log and len(part) > 2 else 1.0))
    return out


def load_rows(path):
    """JSONL rows. A last line without its newline is a sweep still appending: skipped."""
    with open(path) as f:
        text = f.read()
    lines = text.split("\n")
    partial = lines.pop()
    if partial.strip():
        print(f"note {path}: unterminated last line skipped (sweep running?)")
    out = []
    for i, line in enumerate(lines, start=1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: {e}")
    return out


def source_pending(src):
    return not os.path.exists(os.path.join(REPO, ROUTING, src.files[0]))


def collect(spec, src, stats, noisy_keys, root=ROUTING):
    """Return {(dtype, *key): {spelling: [median, ...]}} for one source.

    Keys holding a kept ok=false row are added to noisy_keys.

    Each list entry is one independent median: a pass on sm_120, an archived
    measurement on sm_89. Rule 2's drop of current-only cells with < 2 arms happens here.
    """
    cells = defaultdict(lambda: defaultdict(list))
    seen = set()
    for fname in src.files:
        for r in load_rows(os.path.join(REPO, root, fname)):
            if r["op"] not in spec.row_ops:
                stats["dropped_op"] += 1
                continue
            if src.current_only and not r.get("kernel_current"):
                stats["dropped_not_current"] += 1
                continue
            arm = r["arm"]
            if arm not in spec.arm_spelling:
                stats["dropped_unpinned_arm"] += 1
                continue
            kept_not_ok = not r["ok"] and spec.keep_not_ok(r)
            if not r["ok"] and not kept_not_ok:
                stats["dropped_not_ok"] += 1
                continue
            if r.get(spec.route_field) not in spec.arm_route[arm]:
                stats["dropped_route_mismatch"] += 1
                continue
            key = spec.row_key(r)
            if key is None or not r.get("time_ms"):
                stats["dropped_malformed"] += 1
                continue
            key = (DTYPES[r["dtype"]],) + key
            if "pass" in r:
                ident = (fname, key, arm, r["pass"])
                if ident in seen:
                    raise SystemExit(f"duplicate pass row: {ident}")
                seen.add(ident)
            if kept_not_ok:
                stats["kept_not_ok"] += 1
                noisy_keys.add(key)
            cells[key][spec.arm_spelling[arm]].append(float(r["time_ms"]))
    if src.current_only:
        for key in [k for k, arms in cells.items() if len(arms) < 2]:
            stats["dropped_cells_lt2_arms"] += 1
            del cells[key]
    return cells


def rank(times, order):
    """Section 6.3: within 3% of the best ties, ties in list order, the rest by time."""
    best = min(times.values())
    tied = [c for c in times if times[c] <= best * (1.0 + TIE)]
    rest = [c for c in times if c not in tied]
    tied.sort(key=order.index)
    rest.sort(key=lambda c: (times[c], order.index(c)))
    return tied + rest


def fmt_ms(t):
    """Four significant figures in plain decimal (no exponent), for any parser."""
    if t <= 0:
        raise SystemExit(f"non-positive time {t}")
    decimals = max(0, 3 - int(math.floor(math.log10(t))))
    return f"{t:.{decimals}f}"


def fmt_key(spec, key):
    return " ".join(f"{name}={v}" for (name, _, _), v in zip(parse_keys(spec.keys), key))


def table_text(spec, dtype, device, batchlas, kernels, date, source, rows):
    """The section 5.4 text of a timed table; rows maps key -> (times, noisy). The one
    formatter for converted and tuner tables."""
    lines = [
        f"# op={spec.op} dtype={dtype} device={device} batchlas={batchlas} "
        f"kernels={kernels} date={date}",
        f"# source={source}",
        f"# keys: {spec.keys}",
    ]
    for key in sorted(rows):
        times, noisy = rows[key]
        entries = " | ".join(f"{c} {fmt_ms(times[c])}" for c in rank(times, spec.candidate_order))
        lines.append(f"{fmt_key(spec, key)} | {entries}" + ("   # noisy" if noisy else ""))
    return "\n".join(lines) + "\n"


def converted_text(spec, src, dtype, rows, date):
    source = " + ".join(f"{ROUTING}/{f}" for f in src.files)
    return table_text(spec, dtype, src.device, src.batchlas, "unknown", date,
                      f"{source} ({src.note})", rows)


TUNER = "tuner:"
TUNER_SCHEMA = 1


def tuner_rel(path):
    """The JSONL path as a tuner table header names it: repo-relative when inside the repo."""
    rel = os.path.relpath(os.path.abspath(path), REPO)
    return os.path.abspath(path) if rel.startswith("..") else rel


def read_tuner(path):
    """Return (spec, meta, {key: (times, noisy)}) from one batchlas_tune raw JSONL file."""
    recs = load_rows(path)
    if not recs or recs[0].get("kind") != "meta":
        raise SystemExit(f"{path}: the first record is not a tuner 'meta' record")
    meta = recs[0]
    if meta.get("schema") != TUNER_SCHEMA:
        raise SystemExit(f"{path}: tuner schema {meta.get('schema')} != {TUNER_SCHEMA}")
    spec = OP_BY_NAME.get(meta.get("op"))
    if spec is None:
        raise SystemExit(f"{path}: op '{meta.get('op')}' has no OpSpec")
    if meta.get("keys") != spec.keys:
        raise SystemExit(f"{path}: keys '{meta.get('keys')}' != the {spec.op} OpSpec '{spec.keys}'")
    cands = meta.get("candidates", "").split("|")
    if [c for c in spec.candidate_order if c in cands] != cands:
        raise SystemExit(f"{path}: candidates {cands} are not in the OpSpec order {spec.candidate_order}")
    keyspec = parse_keys(spec.keys)
    key_of = spec.tuner_key or (lambda r: tuple(int(r[n]) if is_log else str(r[n]) for n, is_log, _ in keyspec))
    passes = int(meta["passes"])
    by_cell = defaultdict(list)
    for r in recs[1:]:
        if r.get("kind") == "pass":
            by_cell[key_of(r)].append(r)
    rows = {}
    for key, rs in by_cell.items():
        for final in sorted({int(r["attempt"]) for r in rs}, reverse=True):
            meds = defaultdict(list)
            for r in rs:
                if int(r["attempt"]) == final and r["status"] == "ok" and r.get("median_ms"):
                    meds[r["cand"]].append(float(r["median_ms"]))
            times = {c: sum(v) / len(v) for c, v in meds.items() if len(v) == passes}
            if times:
                rows[key] = (times, any(max(meds[c]) / min(meds[c]) - 1.0 > NOISY for c in times))
                break
    return spec, meta, rows


def tuner_text(spec, meta, rows, jsonl):
    note = (f"tuner: {meta['passes']} passes x {meta['reps']} reps, warm {meta['warm_s']} s, "
            f"ld_pad {meta.get('ld_pad', 0)}")
    return table_text(spec, meta["dtype"], meta["device"], meta["batchlas"], meta["kernels"],
                      meta["date"], f"{TUNER}{tuner_rel(jsonl)} ({note})", rows)


def write_tuner_tables(paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        spec, meta, rows = read_tuner(p)
        if not rows:
            raise SystemExit(f"{p}: no cell has a candidate timed in every pass")
        dest = os.path.join(out_dir, f"{spec.op}.{meta['dtype']}.{meta['device']}.txt")
        with open(dest, "w") as f:
            f.write(tuner_text(spec, meta, rows, p))
        noisy = sum(1 for _, nz in rows.values() if nz)
        print(f"wrote {dest}: {len(rows)} rows, {noisy} noisy (kernels={meta['kernels']})")


def build_converted(date, stats_by_source, problems):
    """Return {(op, dtype, device): (text, rows, spec, src)}; rows maps key -> (times, noisy).

    Sources with adopted=False are validated (problems, stats) but produce no entry.
    """
    out = {}
    for spec in OPS:
        for src in spec.sources:
            if source_pending(src):
                stats_by_source[(spec.op, src.device)] = {"pending": f"{ROUTING}/{src.files[0]} absent"}
                continue
            stats = defaultdict(int)
            stats_by_source[(spec.op, src.device)] = stats
            by_dtype = defaultdict(dict)
            noisy_keys = set()
            cells = collect(spec, src, stats, noisy_keys)
            for (dtype, *key), arms in cells.items():
                times = {c: sum(v) / len(v) for c, v in arms.items()}
                noisy = (dtype, *key) in noisy_keys or any(
                    max(v) / min(v) - 1.0 > NOISY for v in arms.values())
                by_dtype[dtype][tuple(key)] = (times, noisy)
            stats["cells"] = len(cells)
            where = f"{spec.op} {src.device} ({ROUTING}/{src.files[0]})"
            if not cells:
                problems.append(f"{where}: no cell survived conversion (contract drift?)")
            if stats["dropped_malformed"]:
                problems.append(f"{where}: {stats['dropped_malformed']} rows with a malformed key")
            if not src.adopted:
                stats["not_adopted"] = 1
                continue
            for dtype, rows in by_dtype.items():
                out[(spec.op, dtype, src.device)] = (
                    converted_text(spec, src, dtype, rows, date), rows, spec, src)
    return out


def read_transcriber_csv(path):
    """Return {(op, dtype, device): {key: [spelling, ...]}}; raises SystemExit on bad input."""
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for line_no, r in enumerate(csv.DictReader(f), start=2):
            where = f"{path}:{line_no}"
            spec = OP_BY_NAME.get(r.get("op", ""))
            if spec is None:
                raise SystemExit(f"{where}: op '{r.get('op')}' has no OpSpec")
            key = []
            for name, is_log, _ in parse_keys(spec.keys):
                v = (r.get(name) or "").strip()
                if not v:
                    raise SystemExit(f"{where}: missing key column '{name}'")
                if is_log and (not v.isdigit() or int(v) <= 0):
                    raise SystemExit(f"{where}: log key {name}='{v}' is not a positive integer")
                key.append(int(v) if is_log else v)
            ranked = [s.strip() for s in (r.get("ranked") or "").split("|") if s.strip()]
            bad = [s for s in ranked if s not in spec.candidate_order]
            if not ranked or bad or len(set(ranked)) != len(ranked):
                raise SystemExit(f"{where}: ranked '{r.get('ranked')}' is empty, has a duplicate "
                                 f"or names a spelling outside {spec.candidate_order}")
            if r.get("dtype") not in DTYPES or not r.get("device"):
                raise SystemExit(f"{where}: bad dtype '{r.get('dtype')}' or empty device")
            rows = out[(spec.op, DTYPES[r["dtype"]], r["device"])]
            if tuple(key) in rows:
                raise SystemExit(f"{where}: duplicate cell {key}")
            rows[tuple(key)] = ranked
    return out


def transcribed_text(spec, dtype, device, rows, sha, date, csv_rel):
    lines = [
        f"# op={spec.op} dtype={dtype} device={device} batchlas={sha} kernels=unknown "
        f"date={date} source={TRANSCRIBED}{sha}",
        f"# transcriber_csv={csv_rel} (the old router's preference order per grid cell, untimed)",
        f"# keys: {spec.keys}",
    ]
    for key in sorted(rows):
        lines.append(f"{fmt_key(spec, key)} | " + " | ".join(f"{c} -" for c in rows[key]))
    return "\n".join(lines) + "\n"


def table_path(op, dtype, device):
    return os.path.join(REPO, TUNED, f"{op}.{dtype}.{device}.txt")


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


def support_envelope(rows):
    """Largest measured n per (uplo, choice): a stand-in for can_run when reviewing guesses."""
    env = defaultdict(int)
    env[("L", "blocked")] = env[("L", "vendor")] = env[("U", "vendor")] = sys.maxsize
    for (uplo, n, _), ranked in rows:
        for c, _ in ranked:
            env[(uplo, c)] = max(env[(uplo, c)], n)
    return env


DEFAULT_OFFGRID_N = [44, 72, 104, 144, 208, 272, 352, 480, 704, 1152]
DEFAULT_OFFGRID_BATCH = [512, 8192, 32768]


def potrf_offgrid(texts, points):
    """texts: {(dtype, device): table text} for the op's converted tables."""
    points = points or [(n, None) for n in DEFAULT_OFFGRID_N]
    print("\noff-grid lookup (uplo=L): n batch -> row n/batch : first entry"
          " [first entry within the measured n envelope, if different]")
    for (dtype, device) in sorted(texts, key=lambda k: (k[1], k[0])):
        _, keyspec, parsed = parse_table(texts[(dtype, device)])
        env = support_envelope(parsed)
        print(f"-- {dtype} {device}")
        for n, b in points:
            cells = []
            for batch in ([b] if b else DEFAULT_OFFGRID_BATCH):
                (_, rn, rb), ranked = nearest(parsed, ("L", n, batch), keyspec)
                first = ranked[0][0]
                fit = next((c for c, _ in ranked if env[("L", c)] >= n), "none")
                extra = f" [{fit}]" if fit != first else ""
                cells.append(f"b{batch}->{rn}/{rb}:{first}{extra}")
            print(f"   n={n:<5} " + "  ".join(cells))


POTRF.review = potrf_offgrid
OPS = [POTRF, POSV, TRSM, ORGQR]
OP_BY_NAME = {s.op: s for s in OPS}


def read_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            tok = line.split("#", 1)[0].split()
            if tok:
                pts.append((int(tok[0]), int(tok[1]) if len(tok) > 1 else None))
    return pts


def row_kind_problems(header, parsed):
    """parse_table()'s (src/select/select.cc) timed/untimed rules: a row is all-timed or
    all-untimed; an untimed row needs source=transcribed:<hex sha>; a table may mix rows."""
    out = []
    source = header.get("source", "")
    if source.startswith(TRANSCRIBED) and not HEX.fullmatch(source[len(TRANSCRIBED):]):
        out.append(f"source={source}: the commit after '{TRANSCRIBED}' must be a hex sha")
    for key, ranked in parsed:
        kinds = {ms is None for _, ms in ranked}
        if len(kinds) > 1:
            out.append(f"row {key} mixes timed and untimed ('-') entries")
        elif kinds == {True} and not source.startswith(TRANSCRIBED):
            out.append(f"untimed ('-') row {key} needs a 'source={TRANSCRIBED}<sha>' header")
    return out


def check_parsed(path, spec, header, keyspec, parsed, failures):
    """Format checks shared by both sources; returns whether the rows are untimed."""
    if keyspec != parse_keys(spec.keys):
        failures.append(f"{path}: '# keys:' differs from the {spec.op} OpSpec '{spec.keys}'")
    if len({k for k, _ in parsed}) != len(parsed):
        failures.append(f"{path}: duplicate row keys")
    failures += [f"{path}: {p}" for p in row_kind_problems(header, parsed)]
    untimed = any(ms is None for _, ranked in parsed for _, ms in ranked)
    for key, ranked in parsed:
        bad = [c for c, _ in ranked if c not in spec.candidate_order]
        if bad or len({c for c, _ in ranked}) != len(ranked):
            failures.append(f"{path}: row {key} has unknown or repeated spellings {bad}")
    return untimed


def check(converted, points):
    failures = []
    tuned_dir = os.path.join(REPO, TUNED)
    on_disk = sorted(f for f in os.listdir(tuned_dir) if f.endswith(".txt"))
    expected = sorted(os.path.basename(table_path(*k)) for k in converted)
    disk_converted = []
    tuner_tables = set()  # a tuned table supersedes a converted source for its (op, dtype, device)
    review_texts = defaultdict(dict)
    for fname in on_disk:
        path = os.path.join(tuned_dir, fname)
        rel = os.path.relpath(path, REPO)
        op, dtype, device = (fname[:-len(".txt")].split(".") + ["", "", ""])[:3]
        spec = OP_BY_NAME.get(op)
        if spec is None:
            failures.append(f"{rel}: op '{op}' has no OpSpec in scripts/sweep_to_table.py")
            continue
        with open(path) as f:
            disk = f.read()
        header, keyspec, parsed = parse_table(disk)
        untimed = check_parsed(rel, spec, header, keyspec, parsed, failures)
        if header.get("source", "").startswith(TRANSCRIBED):
            check_transcribed(rel, spec, (op, dtype, device), disk, header, failures)
            for key, _ in parsed:
                hit = nearest(parsed, key, keyspec)
                if hit is None or hit[0] != key:
                    failures.append(f"{rel}: nearest{key} -> {hit and hit[0]}")
            print(f"check {rel}: {len(parsed)} transcribed rows looked up")
            continue
        if header.get("source", "").startswith(TUNER):
            check_tuner(rel, header, disk, parsed, keyspec, failures)
            tuner_tables.add(fname)
            review_texts[op][(dtype, device)] = disk
            continue
        disk_converted.append(fname)
        if untimed or (op, dtype, device) not in converted:
            continue
        _, rows, _, src = converted[(op, dtype, device)]
        if disk != converted_text(spec, src, dtype, rows, header.get("date", "")):
            failures.append(f"{path}: differs from the sweeps (re-run without --check)")
        for key, (times, _) in rows.items():
            hit = nearest(parsed, key, keyspec)
            if hit is None or hit[0] != key:
                failures.append(f"{path}: nearest{key} -> {hit and hit[0]}")
                continue
            winner = rank(times, spec.candidate_order)[0]
            if hit[1][0][0] != winner:
                failures.append(f"{path}: {key} first {hit[1][0][0]} != winner {winner}")
        review_texts[op][(dtype, device)] = disk
        print(f"check {rel}: {len(rows)} measured cells looked up")
    expected = [f for f in expected if f not in tuner_tables]
    if disk_converted != expected:
        failures.append(f"tuned/ files {disk_converted} != derived {expected}")
    for spec in OPS:
        if spec.review and review_texts[spec.op]:
            spec.review(review_texts[spec.op], points)
    return failures


def check_tuner(rel, header, disk, parsed, keyspec, failures):
    """Re-derive a tuner table from the raw JSONL its source= names, when that file exists;
    either way every row must look up to itself (section 5.4)."""
    for key, _ in parsed:
        hit = nearest(parsed, key, keyspec)
        if hit is None or hit[0] != key:
            failures.append(f"{rel}: nearest{key} -> {hit and hit[0]}")
    jsonl = header["source"][len(TUNER):]
    path = jsonl if os.path.isabs(jsonl) else os.path.join(REPO, jsonl)
    if not os.path.exists(path):
        print(f"note {rel}: {jsonl} absent, re-derive skipped (self-check only)")
        return
    spec, meta, rows = read_tuner(path)
    if disk != tuner_text(spec, meta, rows, path):
        failures.append(f"{rel}: differs from {jsonl} (re-run --tuner)")
    print(f"check {rel}: {len(parsed)} tuned rows re-derived from {jsonl}")


def check_transcribed(rel, spec, ident, disk, header, failures):
    """Re-derive a transcribed table from the CSV its header names, when that CSV exists."""
    csv_rel = header.get("transcriber_csv", "")
    sha = header["source"][len(TRANSCRIBED):]
    if not csv_rel:
        failures.append(f"{rel}: transcribed table without a transcriber_csv= header")
        return
    if not os.path.exists(os.path.join(REPO, csv_rel)):
        print(f"note {rel}: {csv_rel} absent, re-derive skipped (self-check only)")
        return
    if header.get("batchlas") != sha:
        failures.append(f"{rel}: batchlas={header.get('batchlas')} != transcribed sha {sha}")
    rows = read_transcriber_csv(os.path.join(REPO, csv_rel)).get(ident)
    if rows is None:
        failures.append(f"{rel}: {csv_rel} has no {ident} rows")
    elif disk != transcribed_text(spec, ident[1], ident[2], rows, sha, header.get("date", ""), csv_rel):
        failures.append(f"{rel}: differs from {csv_rel} (re-run --transcribe)")


def transcribe(csv_paths, sha, date, converted):
    for p in csv_paths:
        csv_rel = os.path.relpath(os.path.abspath(p), REPO)
        if csv_rel.startswith(".."):
            raise SystemExit(f"{p}: the transcriber CSV must live inside the repository")
        for (op, dtype, device), rows in sorted(read_transcriber_csv(p).items()):
            if (op, dtype, device) in converted:
                raise SystemExit(f"{op}.{dtype}.{device}: already converted from a sweep; "
                                 "a device's table has one source")
            path = table_path(op, dtype, device)
            with open(path, "w") as f:
                f.write(transcribed_text(OP_BY_NAME[op], dtype, device, rows, sha, date, csv_rel))
            print(f"wrote {os.path.relpath(path, REPO)}: {len(rows)} transcribed rows")


def resolve_sha(rev):
    """Any commit-ish -> its 8-digit hex sha; the C++ loader rejects a non-hex transcribed:<sha>."""
    r = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", "--short=8",
                        f"{rev}^{{commit}}"], capture_output=True, text=True)
    sha = r.stdout.strip()
    if r.returncode != 0 or not HEX.fullmatch(sha):
        raise SystemExit(f"--sha {rev}: not a commit in this repository")
    return sha


FIXTURE = "tests/data/sweep_to_table_rows.jsonl"


def self_test():
    """Converter rules on rows copied verbatim from the live sweeps (FIXTURE: 6 posv rows of
    the sm_120 posv driver, 5 potrf rows of sm120_potrf_sweep.jsonl) and on mutations of them."""
    bad = []

    def expect(cond, what):
        if not cond:
            bad.append(f"self-test: {what}")

    keys = "# keys: uplo:exact n:log batch:log\n"
    tr, sw = "# op=potrf source=transcribed:2b46acab\n", "# op=potrf source=sweep.jsonl\n"
    for text, want in [
        (tr + keys + "uplo=L n=1 batch=1 | tiny - | cta -\n", None),
        (tr + keys + "uplo=L n=1 batch=1 | tiny -\nuplo=L n=2 batch=1 | tiny 1 | cta 2\n", None),
        (sw + keys + "uplo=L n=1 batch=1 | tiny 1\n", None),
        (tr + keys + "uplo=L n=1 batch=1 | tiny - | cta 1\n", "mixes timed and untimed"),
        (tr + keys + "uplo=L n=1 batch=1 | tiny -\nuplo=L n=2 batch=1 | tiny 1 | cta -\n", "mixes"),
        (sw + keys + "uplo=L n=1 batch=1 | tiny 1\nuplo=L n=2 batch=1 | tiny -\n", "needs a 'source="),
        (keys + "uplo=L n=1 batch=1 | tiny -\n", "needs a 'source="),
        ("# source=transcribed:\n" + keys + "uplo=L n=1 batch=1 | tiny -\n", "must be a hex sha"),
        ("# source=transcribed:HEAD\n" + keys + "uplo=L n=1 batch=1 | tiny -\n", "must be a hex sha"),
    ]:
        header, _, parsed = parse_table(text)
        got = row_kind_problems(header, parsed)
        expect(got == [] if want is None else len(got) == 1 and want in got[0], f"{text!r}: {got}")
    expect(HEX.fullmatch(resolve_sha("HEAD")) is not None, "resolve_sha(HEAD) is not hex")
    for rev in ("not-a-commit", "transcribed", ""):
        try:
            resolve_sha(rev)
            expect(False, f"resolve_sha({rev!r}) accepted")
        except SystemExit:
            pass

    def run(spec, rows, partial=""):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "rows.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows) + partial)
            stats, noisy = defaultdict(int), set()
            return collect(spec, Source("fixture", ["rows.jsonl"], "x", "x"), stats, noisy, d), stats, noisy

    real = load_rows(os.path.join(REPO, FIXTURE))
    posv = [r for r in real if r["op"] in POSV.row_ops]
    potrf = [r for r in real if r["op"] in POTRF.row_ops]
    expect(len(posv) == 6 and len(potrf) == 5, f"fixture holds {len(posv)} posv, {len(potrf)} potrf rows")
    cells, stats, noisy = run(POSV, real, partial='{"op": "posv", "dtype": "flo')
    cell = ("float", "L", 1, 1, 128)
    expect(set(cells) == {cell}, f"posv cells {sorted(cells)}")
    expect(sorted(cells.get(cell, {})) == ["blocked", "cta", "tiny"], f"posv arms {dict(cells.get(cell, {}))}")
    expect(noisy == {cell} and stats["kept_not_ok"] == 1, f"relsd row not kept as noisy: {dict(stats)}")
    expect(stats["dropped_not_ok"] == 3 and stats["dropped_op"] == 5, f"posv stats {dict(stats)}")
    relsd = next(r for r in posv if r["reason"] == "relsd")
    for change, what in [({"reason": "residual+relsd"}, "residual"), ({"info_nonzero": 1}, "info"),
                         ({"residual": None}, "no residual"), ({"reached": "native:blocked"}, "fallback")]:
        cells, stats, _ = run(POSV, [dict(relsd, **change)])
        expect(not cells, f"relsd row with {what} kept: {dict(stats)}")
    upper = [dict(r, op="posv_upper", uplo="Upper") for r in posv[:3]]
    cells, _, _ = run(POSV, upper + [dict(r, **{"pass": 2}) for r in upper])
    ucell = ("float", "U", 1, 1, 128)
    expect(set(cells) == {ucell} and all(len(v) == 2 for v in cells[ucell].values()),
           f"Upper/pass 2 cells {dict(cells)}")
    cells, stats, _ = run(POTRF, real)
    expect(set(cells) == {("float", "L", 2, 32768), ("float", "U", 8, 8192)}, f"potrf cells {sorted(cells)}")
    expect(stats["dropped_not_ok"] == 3 and stats["dropped_op"] == 6, f"potrf stats {dict(stats)}")
    cells, _, _ = run(POTRF, [dict(potrf[0], route=None, reached=potrf[0]["route"])])
    expect(not cells, "potrf read its route from 'reached'")
    bad += self_test_tuner()
    return bad


def self_test_tuner():
    """read_tuner: the final attempt replaces the first, a candidate must be ok in every pass."""
    meta = {"kind": "meta", "schema": TUNER_SCHEMA, "op": "posv", "dtype": "float", "device": "sm_0",
            "batchlas": "x", "kernels": "y", "date": "d", "keys": POSV.keys,
            "candidates": "tiny|cta|blocked", "passes": 2, "reps": 1, "warm_s": 0}
    cell = {"kind": "pass", "uplo": "L", "n": 8, "nrhs": 2, "batch": 128}
    rows = [dict(cell, cand=c, status=s, attempt=a, median_ms=t, **{"pass": p}) for c, s, a, p, t in [
        ("tiny", "ok", 0, 1, 9.0), ("tiny", "ok", 0, 2, 1.0), ("tiny", "ok", 1, 1, 1.0), ("tiny", "ok", 1, 2, 1.02),
        ("cta", "ok", 1, 1, 0.5), ("cta", "error", 1, 2, None), ("blocked", "ok", 1, 1, 2.0),
        ("blocked", "ok", 1, 2, 2.0)]]
    # A re-measure whose children all failed leaves the complete first attempt in force.
    lost = dict(cell, n=16)
    rows += [dict(lost, cand=c, status=s, attempt=a, median_ms=t, **{"pass": p}) for c, s, a, p, t in [
        ("cta", "ok", 0, 1, 3.0), ("cta", "ok", 0, 2, 3.6), ("cta", "error", 1, 1, None),
        ("cta", "error", 1, 2, None)]]
    bad = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        with open(path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in [meta] + rows))
        spec, meta_read, got = read_tuner(path)
        # check_tuner re-derives a written table from its JSONL and catches a hand edit.
        text = tuner_text(spec, meta_read, got, path)
        for disk, want_fail in ((text, False), (text.replace("tiny 1.010", "tiny 1.011"), True)):
            header, _, parsed = parse_table(disk)
            failures = []
            check_tuner("t.txt", header, disk, parsed, parse_keys(spec.keys), failures)
            if bool(failures) != want_fail:
                bad.append(f"self-test: check_tuner on {'an edited' if want_fail else 'a fresh'} table -> {failures}")
    want = {("L", 8, 2, 128): ({"tiny": 1.01, "blocked": 2.0}, False), ("L", 16, 2, 128): ({"cta": 3.3}, True)}
    ok = got.keys() == want.keys() and all(
        abs(got[k][0][c] - v) < 1e-12 and set(got[k][0]) == set(want[k][0]) and got[k][1] == want[k][1]
        for k in want for c, v in want[k][0].items())
    return bad + ([] if ok else [f"self-test: read_tuner -> {got}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="diff tuned/ against the sources")
    ap.add_argument("--check-points", help="file of off-grid 'n [batch]' points for --check")
    ap.add_argument("--transcribe", nargs="+", metavar="CSV", help="transcriber CSVs to write")
    ap.add_argument("--sha", help="--transcribe: the transcribed router's commit (default HEAD)")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--self-test", action="store_true", help="run the converter self-test only")
    ap.add_argument("--tuner", nargs="+", metavar="JSONL", help="batchlas_tune raw files to write")
    ap.add_argument("--out", default=os.path.join(REPO, TUNED), help="--tuner: table directory")
    args = ap.parse_args()

    if args.tuner:
        write_tuner_tables(args.tuner, args.out)
        return 0
    if args.self_test:
        bad = self_test()
        print("\n".join(bad + [f"--self-test: {'FAILED' if bad else 'OK'}"]))
        return 1 if bad else 0
    if args.sha:
        resolve_sha(args.sha)  # rejected here for --check too, as the C++ loader would
    stats, problems = {}, []
    converted = build_converted(args.date, stats, problems)
    if args.check:
        points = read_points(args.check_points) if args.check_points else None
        failures = self_test() + problems + check(converted, points)
        for (op, device), s in stats.items():
            if s.get("not_adopted"):
                print(f"source {op} {device}: validated, not adopted yet: "
                      + ", ".join(f"{k}={v}" for k, v in sorted(s.items()) if k != "not_adopted"))
        for f in failures:
            print("FAIL:", f)
        print(f"\n--check: {'FAILED, ' + str(len(failures)) + ' problems' if failures else 'OK'}")
        return 1 if failures else 0

    if problems:
        raise SystemExit("\n".join(problems))
    os.makedirs(os.path.join(REPO, TUNED), exist_ok=True)
    if args.transcribe:
        transcribe(args.transcribe, resolve_sha(args.sha or "HEAD"), args.date, converted)
        return 0
    for (op, device), s in stats.items():
        print(f"{op} {device}: " + ", ".join(f"{k}={v}" for k, v in sorted(s.items())))
    for (op, dtype, device), (text, rows, _, _) in sorted(
            converted.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        path = table_path(op, dtype, device)
        with open(path, "w") as f:
            f.write(text)
        noisy = sum(1 for _, nz in rows.values() if nz)
        upper = sum(1 for k in rows if k[0] == "U")
        print(f"wrote {os.path.relpath(path, REPO)}: {len(rows)} rows ({upper} Upper), {noisy} noisy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
