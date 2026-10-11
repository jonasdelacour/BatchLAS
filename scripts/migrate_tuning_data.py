#!/usr/bin/env python3
"""One-off: the per-dtype tables, per-run ledger directories and converted/transcribed tables become
benchmarks/results/tuning/ledger/<op>.<device>.jsonl (docs/design/tuning-data.md#tuning-data-migration).

    scripts/migrate_tuning_data.py write      # new ledgers from the old layout in the work tree
    scripts/migrate_tuning_data.py accept DIR # acceptance against tables regenerated into DIR

`write` also verifies and writes the rehash lines from git. `accept` checks (a) every old
run-file line is in the new ledger, in file-name-then-line order per old directory, with "dtype"
added, and (b) every old per-dtype table equals the regenerated table restricted to that dtype.
Run once, then deleted (provenance: tuned/README.md).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_to_table as stt  # noqa: E402

REPO = stt.REPO
OLD_LEDGER = os.path.join(REPO, "benchmarks/results/tuning/ledger")
NEW_LEDGER = OLD_LEDGER
TUNED = os.path.join(REPO, "tuned")
BASE = "189cbc21"            # the commit whose sources "now" means (the tuning branch before this PR's data)
PRE_DEEP = "aad906cb^"       # the sm_120 tables before the deep run replaced them: their transcribed shas
DTYPES = stt.DTYPES
WHY_REHASH = "hash v2; sources identical modulo comments"


def git(*args, binary=False):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode()


def v1_hash_at(commit, paths):
    manifest = ""
    for p in paths:
        data = git("show", f"{commit}:{p}", binary=True)
        if data is None:
            return None
        manifest += hashlib.sha256(data).hexdigest() + "  " + p + "\n"
    return hashlib.sha256(manifest.encode()).hexdigest()[:8]


def block_at(commit, op):
    text = git("show", f"{commit}:tools/tune/{op}_spec.cc")
    return stt.parse_kernel_block(text) if text is not None else None


def with_dtype(line, dtype):
    """The old line, byte for byte, with "dtype" inserted after "kind"."""
    m = re.match(r'\{"kind": "[a-z]+", ', line)
    if not m:
        raise SystemExit(f"unexpected line start: {line[:60]}")
    return line[:m.end()] + f'"dtype": "{dtype}", ' + line[m.end():]


def old_dirs():
    """{(op, device): [(dtype, dir)]} in DTYPES order."""
    out = defaultdict(list)
    for name in sorted(os.listdir(OLD_LEDGER)):
        parts = name.split(".")
        if len(parts) == 3 and os.path.isdir(os.path.join(OLD_LEDGER, name)):
            out[(parts[0], parts[2])].append((parts[1], os.path.join(OLD_LEDGER, name)))
    for k in out:
        out[k].sort(key=lambda x: DTYPES.index(x[0]))
    return out


def old_tables():
    """{(op, device): {dtype: path}} for tuned/<op>.<dtype>.<device>.txt."""
    out = defaultdict(dict)
    for name in sorted(os.listdir(TUNED)):
        parts = name.split(".")
        if len(parts) == 4 and parts[3] == "txt":
            out[(parts[0], parts[2])][parts[1]] = os.path.join(TUNED, name)
    return out


def table_rows(text):
    """[(body, tag)] of a table's rows: body is the text before '#', tag the comment after it."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        body, _, tag = line.partition("#")
        out.append((body.strip(), tag.strip()))
    return out


def untimed_cell(run_id, dtype, tier, key_text, spellings, date, times=None):
    rec = {"kind": "cell", "dtype": dtype, "run_id": run_id, "tier": tier, "key": key_text, "round": 0,
           "date": date, "ranked": "|".join(spellings), "cands": "|".join(spellings)}
    for i, c in enumerate(spellings):
        rec.update({f"h{i}": "", f"s{i}": "ok", f"r{i}": "", f"m{i}": None if times is None else times[i],
                    f"lo{i}": None, f"hi{i}": None, f"n{i}": 0})
    return json.dumps(rec)


def run_line(run_id, dtype, device, batchlas, date, tier, keys, candidates, argv):
    return json.dumps({"kind": "run", "dtype": dtype, "run_id": run_id, "host": "", "device": device,
                       "device_name": "", "batchlas": batchlas, "argv": argv, "date": date, "tier": tier,
                       "keys": keys, "candidates": candidates})


def row_cells(body_rows, keys, run_id, dtype, tier, date):
    """Table rows -> cell lines; a timed row keeps its times."""
    names = [n for n, _, _ in stt.parse_keys(keys)]
    out = []
    for body in body_rows:
        parts = [p.strip() for p in body.split("|")]
        given = dict(tok.split("=", 1) for tok in parts[0].split())
        key_text = ",".join(f"{n}={given[n]}" for n in names)
        ents = [p.split() for p in parts[1:]]
        timed = ents[0][1] != "-"
        out.append(untimed_cell(run_id, dtype, tier, key_text, [e[0] for e in ents], date,
                                [float(e[1]) for e in ents] if timed else None))
    return out


def header_of(text):
    h, keys = {}, ""
    for line in text.splitlines():
        if line.startswith("# keys:"):
            keys = line[len("# keys:"):].strip()
        elif line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    h.setdefault(k, v)
    return h, keys


def newest_runs(dirs):
    """{dtype: newest run record} over the old run files (date, then run id)."""
    out = {}
    for dtype, d in dirs:
        runs = []
        for name in sorted(os.listdir(d)):
            for line in open(os.path.join(d, name)):
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("kind") == "run":
                        runs.append(rec)
        out[dtype] = max(runs, key=lambda r: (r["date"], r["run_id"]))
    return out


def verify_rehash(op, fam, commits, now_block):
    """None when, at every commit, the v1 hash of fam's sources is the stored one (checked by the
    caller) and the normalized sources equal today's; else the reason."""
    now_paths = stt.family_paths(now_block, fam)
    for c in sorted(commits):
        block = block_at(c, op)
        if block is None:
            return f"no spec at {c}"
        paths = stt.family_paths(block, fam)
        if paths != now_paths:
            return f"source list at {c} differs from today's"
        for p in paths:
            then = git("show", f"{c}:{p}", binary=True)
            with open(os.path.join(REPO, p), "rb") as f:
                now = f.read()
            if then is None or stt.normalize_source(then) != stt.normalize_source(now):
                return f"{p} differs modulo comments between {c} and today"
    return None


def migrate_sm120(op, device, dirs, tables, report):
    lines = []
    # (1) every old run-file line, per directory in file-name-then-line order, plus dtype
    for dtype, d in dirs:
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name)) as f:
                for line in f.read().split("\n"):
                    if line.strip():
                        json.loads(line)  # a torn line would become a fatal middle line
                        lines.append(with_dtype(line, dtype))
    heads = newest_runs(dirs)
    # (2) the deep tables' transcribed rows, under the transcription that produced them
    for dtype, _ in dirs:
        rows = [b for b, t in table_rows(open(tables[dtype]).read()) if t == "transcribed"]
        if not rows:
            continue
        pre = git("show", f"{PRE_DEEP}:tuned/{op}.{dtype}.{device}.txt")
        h, keys = header_of(pre or "")
        src = h.get("source", "")
        if not src.startswith("transcribed:"):
            raise SystemExit(f"{op}.{dtype}.{device}: {len(rows)} transcribed rows but the pre-deep table says source={src}")
        sha = src[len("transcribed:"):]
        rid = f"transcribed-{sha}"
        lines.append(run_line(rid, dtype, device, sha, h["date"], "transcribed", heads[dtype]["keys"],
                              heads[dtype]["candidates"], f"transcribed table tuned/{op}.{dtype}.{device}.txt at {PRE_DEEP}"))
        lines += row_cells(rows, heads[dtype]["keys"], rid, dtype, "transcribed", h["date"])
        report["transcribed_rows"][f"{op}.{dtype}.{device}"] = (len(rows), sha)
    # (3) rehash lines, verified from git; -dirty runs are left out
    cells_by = {}
    for dtype, _ in dirs:
        cells_by[dtype] = [stt.read_cell(json.loads(l)) for l in lines
                           if f'"dtype": "{dtype}"' in l[:40] and l.startswith('{"kind": "cell"')]
    run_sha = {}
    extra = []
    for line in lines:
        rec = json.loads(line)
        if rec["kind"] == "run" and rec["tier"] != "transcribed":
            if rec["batchlas"].endswith("-dirty"):  # excluded by every reader; never rehashed
                report["dirty_runs"][f"{op}.{rec['dtype']}.{device} {rec['run_id']}"] = rec["batchlas"]
                continue
            run_sha[(rec["run_id"], rec["dtype"])] = rec["batchlas"]
    now_block = stt.parse_kernel_block(stt.read_spec_source(op))
    produced = defaultdict(lambda: defaultdict(set))  # family -> stored hash -> commits
    for dtype, cells in cells_by.items():
        for c in cells:
            commit = run_sha.get((c["run_id"], dtype))
            if commit is None:
                continue
            for r in c["cands"]:
                if r["hash"]:
                    produced[stt.spelling_family(r["cand"])][r["hash"]].add(commit)
    for fam in sorted(produced):
        to = stt.kernel_hash(stt.family_paths(now_block, fam))
        for frm in sorted(produced[fam]):
            commits = produced[fam][frm]
            bad = [c for c in sorted(commits) if v1_hash_at(c, stt.family_paths(block_at(c, op) or now_block, fam)) != frm]
            why = f"stored hash is not the v1 hash at {bad}" if bad else verify_rehash(op, fam, commits, now_block)
            if why:
                report["rehash_refused"][f"{op}.{device} {fam} {frm}"] = why
                continue
            extra.append(json.dumps({"kind": "rehash", "family": fam, "from": frm, "to": to, "at": BASE,
                                     "why": WHY_REHASH}))
            report["rehash"][f"{op}.{device} {fam}"] = report["rehash"].get(f"{op}.{device} {fam}", []) + [f"{frm}->{to}"]
    return lines + extra


def migrate_other(op, device, tables, heads, report):
    """sm_89 and cpu: transcribed tables -> transcribed cells, the converted potrf tables -> converted cells."""
    lines = []
    for dtype in DTYPES:
        if dtype not in tables:
            continue
        text = open(tables[dtype]).read()
        h, keys = header_of(text)
        src = h.get("source", "")
        rows = table_rows(text)
        if heads.get(dtype) is None:
            raise SystemExit(f"{op}.{dtype}: no sm_120 run line to take the candidates from")
        if keys != heads[dtype]["keys"]:
            raise SystemExit(f"{op}.{dtype}.{device}: keys '{keys}' != choice.hh '{heads[dtype]['keys']}'")
        if src.startswith("transcribed:"):
            sha, tier = src[len("transcribed:"):], "transcribed"
            rid, batchlas = f"transcribed-{sha}", sha
            argv = f"transcribed table tuned/{op}.{dtype}.{device}.txt ({h.get('transcriber_csv', '')})"
        else:
            tier, rid, batchlas = "converted", "converted-" + os.path.basename(src).split(".")[0], h.get("batchlas", "unknown")
            argv = f"converted table tuned/{op}.{dtype}.{device}.txt (source={src})"
            report["noisy_dropped"][f"{op}.{dtype}.{device}"] = sum(1 for _, t in rows if t == "noisy")
        lines.append(run_line(rid, dtype, device, batchlas, h["date"], tier, keys, heads[dtype]["candidates"], argv))
        lines += row_cells([b for b, _ in rows], keys, rid, dtype, tier, h["date"])
        report[f"{tier}_tables"][f"{op}.{dtype}.{device}"] = len(rows)
    return lines


def write():
    report = defaultdict(dict)
    dirs, tables = old_dirs(), old_tables()
    heads_by_op = {op: newest_runs(ds) for (op, dev), ds in dirs.items() if dev == "sm_120"}
    out = {}
    for (op, device), tabs in sorted(tables.items()):
        if device == "sm_120":
            out[(op, device)] = migrate_sm120(op, device, dirs[(op, device)], tabs, report)
        else:
            out[(op, device)] = migrate_other(op, device, tabs, heads_by_op[op], report)
    for (op, device), lines in out.items():
        with open(os.path.join(NEW_LEDGER, f"{op}.{device}.jsonl"), "w") as f:
            f.write("".join(l + "\n" for l in lines))
    print(json.dumps(report, indent=1, sort_keys=True))


def accept(regen):
    """(a) the old lines in order per directory, plus dtype; (b) per dtype, the old rows."""
    problems, stats = [], defaultdict(int)
    for (op, device), ds in sorted(old_dirs().items()):
        new = [l for l in open(os.path.join(NEW_LEDGER, f"{op}.{device}.jsonl")).read().split("\n") if l]
        it = iter(new)
        for dtype, d in ds:
            for name in sorted(os.listdir(d)):
                for line in open(os.path.join(d, name)).read().split("\n"):
                    if not line.strip():
                        continue
                    want = with_dtype(line, dtype)
                    if next(it, None) != want:
                        problems.append(f"(a) {op}.{device}: {d}/{name} line out of order or changed")
                        break
                    stats["a_lines"] += 1
                    a, b = json.loads(line), json.loads(want)
                    if b.pop("dtype") != dtype or a != b:
                        problems.append(f"(a) {op}.{device}: {name} line differs beyond dtype")
    for (op, device), tabs in sorted(old_tables().items()):
        path = os.path.join(regen, f"{op}.{device}.txt")
        new_rows = table_rows(open(path).read())
        for dtype, tpath in tabs.items():
            old = table_rows(open(tpath).read())
            mine = [(b[len(f"dtype={dtype} "):], t) for b, t in new_rows if b.startswith(f"dtype={dtype} ")]
            h, _ = header_of(open(tpath).read())
            src = h.get("source", "")
            if src.startswith("ledger:"):
                ok = mine == old
            else:
                kind = "transcribed" if src.startswith("transcribed:") else "converted"
                ok = [b for b, _ in mine] == [b for b, _ in old] and all(t == kind for _, t in mine)
            stats["b_tables"] += 1
            stats["b_rows"] += len(old)
            if (op, device) == ("gemv", "sm_120"):  # controller ruling: stale until the gemv re-run
                print(f"(b) exempt {op}.{dtype}.{device}: {len(old)} rows before, {len(mine)} after, "
                      f"tiers after {sorted(set(t for _, t in mine))}, equal={ok}")
                continue
            if not ok:
                diff = [(o, m) for o, m in zip(old, mine) if o[0] != m[0] or (src.startswith("ledger:") and o != m)]
                problems.append(f"(b) {op}.{dtype}.{device}: {len(old)} old rows, {len(mine)} new; first diff {diff[:2]}")
    print(json.dumps(stats))
    print("\n".join(problems) if problems else "acceptance (a) and (b): OK")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["write"]:
        write()
    elif sys.argv[1:2] == ["accept"]:
        sys.exit(accept(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
