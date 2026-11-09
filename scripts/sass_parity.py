#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SASS-parity gate: compare the sm_89 code of kernel pairs between two builds.

    scripts/sass_parity.py --base ../main/build --cand build --lib batchlas_extensions_latrd
    scripts/sass_parity.py --base build --cand build --lib batchlas_extensions_latrd \\
        --pair 'LatrdLowerPanelKernelLegacy<=LatrdLowerPanelKernel<' --filter '<float, 128, false>'
    scripts/sass_parity.py --snapshot-write tests/sass_snapshots/batchlas_extensions_posv.json \\
        --cand build --lib batchlas_extensions_posv --select '^batchlas::PosvTinyKernel<'
    scripts/sass_parity.py --snapshot-check tests/sass_snapshots/batchlas_extensions_posv.json --cand build

A device-LA migration must not make a kernel's machine code worse. Per kernel it records
REG, STACK, LMEM, static SMEM, instructions (excluding NOP), a class histogram and the PTX
call targets, then applies the strict (1:1 port) or algo (algorithm change) thresholds.

Pairing: without --pair a kernel is compared with the same demangled name in the other
build. With --pair 'B=C' a base kernel matching B pairs with the cand kernel matching C
whose name is equal once each match is cut out (Legacy<float, 128, false> pairs with
Kernel<float, 128, false>). Base and cand may be the same build dir.

Occupancy: the launch WG comes from the entry's PTX .reqntid/.maxntid, else the --wg table
(default scripts/sass_parity_wg.txt), else it is unknown: the rule then only warns, and is
never inferred from the name. Resident blocks/SM = min(by REG, by static + dynamic SMEM
(dynamic from --smem, default scripts/sass_parity_smem.txt), by warps, 24). strict: any change of blocks/SM is verdict "A/B";
algo: a decrease fails and an increase is "A/B". "A/B" means a runtime A/B decides.

Total: one-sided. A shrink beyond the band passes, annotated "shrink", when every hard class
(BAR SHFL LDS STS LDG STG LDgen LDL/STL) is <= base, REG <= base + 2 and blocks/SM is unchanged
(at every legal WG when the WG is unknown). With an unknown WG, a shrink whose only occupancy
change comes from a REG decrease (blocks/SM can only rise) is "A/B". Growth beyond the band fails. A generic LD/ST
decrease is annotated "conversion"; an increase is a hard fail.

Table files (--wg, --smem): one `<value> <regex>` per line, first match on the demangled name
wins; the value is an integer or `\\N`, capture group N of the regex. An --smem value may be a
comma list of representative sizes for an SLM sized at run time (the verdict is the worst over
them). --wg/--smem also take a plain integer for every kernel, and `none` disables the table.

Identity (--identical): each kernel's SASS text (every instruction line with its encoding, NOPs
included) is hashed; the verdict becomes "every pair byte-identical and, without --pair, the same
kernel set in both builds" (exit 0, else 1), with the strict/algo report still printed. It is the
proof behind the tuned-table SASS-identity rehash (v22 decision 18, F-D1).

Snapshots: a compact per-kernel record of one library for an explicit kernel list,
checked with the algo thresholds. Exit codes: 0 pass, 1 fail (or nothing paired), 2 "A/B"
and no fail, 77 no NVPTX image, no cuobjdump or no image for the snapshot's arch.
evidence: ~/dla-v2-notes/synthesis.txt section 4, wf3_synthesis.txt 3.1, 3.4 and 3.9.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_device_calls as cdc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WG_TABLE = os.path.join(REPO, "scripts", "sass_parity_wg.txt")
DEFAULT_SMEM_TABLE = os.path.join(REPO, "scripts", "sass_parity_smem.txt")

# Opcode classes, as in benchmarks/sass_lab/sass_stats.py (wip/dblas-sass-lab), whose
# numbers the calibration reproduces. First matching class wins.
CLASSES = collections.OrderedDict([
    ("FMA", {"FFMA", "DFMA", "HFMA2", "FFMA32I"}),
    ("LDG", {"LDG"}), ("STG", {"STG"}), ("LDS", {"LDS"}), ("STS", {"STS"}),
    ("LDgen", {"LD", "ST"}), ("LDL/STL", {"LDL", "STL"}),
    ("BAR", {"BAR", "MEMBAR", "ERRBAR", "CCTL"}),
    ("SHFL", {"SHFL", "REDUX", "VOTE", "VOTEU", "MATCH"}),
    ("BRA", {"BRA", "BRX", "JMP", "JMX"}),
    ("CVT", {"I2F", "F2I", "I2FP", "F2IP"}),
])
EXTRA = ("IMADW", "CALL")
RES_KEYS = ("REG", "STACK", "LMEM", "SMEM")
HARD = ("BAR", "SHFL", "LDS", "STS", "LDG", "STG", "LDgen", "LDL/STL")
BAND = {"strict": 0.03, "algo": 0.10}
BANNED_CALL = re.compile(r"^_ZN8batchlas6device|^__(mul|div)[sd]c3$")

ENCODING = re.compile(r"\s*/\* 0x[0-9a-f]{16} \*/\s*$")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
NTID = re.compile(r"\.(reqntid|maxntid)\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?")
SM89 = {"subpartitions": 4, "regs_per_subpartition": 16384, "max_warps": 48, "max_blocks": 24,
        "smem_per_sm": 102400, "smem_reserved": 1024, "smem_granule": 128}


def opcode(text):
    tok = text.split()
    if tok and tok[0].startswith("@"):
        tok = tok[1:]
    return tok[0] if tok else ""


def parse_sass(text, digests=None):
    """{function: opcode Counter}; digests, if given, gets {function: sha256 of its SASS text}."""
    funcs, cur, hashes = {}, None, {}
    for line in text.splitlines():
        if "Function :" in line:
            cur = line.split("Function :", 1)[1].strip()
            funcs[cur], hashes[cur] = collections.Counter(), hashlib.sha256()
            continue
        if cur and (INSN.search(line) or ENCODING.match(line)):
            hashes[cur].update(line.strip().encode() + b"\n")
        m = INSN.search(line) if cur else None
        if not m:
            continue
        full = opcode(m.group(1))
        base = full.split(".")[0]
        if base == "NOP":
            continue
        c = funcs[cur]
        c["total"] += 1
        c["IMADW"] += full.startswith("IMAD.WIDE")
        c["CALL"] += base == "CALL"
        for cls, ops in CLASSES.items():
            if base in ops:
                c[cls] += 1
                break
    if digests is not None:
        digests.update({f: h.hexdigest() for f, h in hashes.items()})
    return funcs


def parse_res(text):
    res, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            cur = m.group(1)
        elif cur and "REG:" in line:
            kv = dict(re.findall(r"(\w+):(\d+)", line))
            res[cur] = {"REG": int(kv.get("REG", 0)), "STACK": int(kv.get("STACK", 0)),
                        "LMEM": int(kv.get("LOCAL", 0)), "SMEM": int(kv.get("SHARED", 0))}
            cur = None
    return res


def parse_ptx_launch(ptx):
    """{entry: {"ptx_wg": int|None, "ptx_wg_src": str|None, "dyn_smem": bool}} from PTX headers."""
    out = {}
    for kind, name, body in cdc.split_functions(ptx):
        if kind != "entry":
            continue
        header = body.split("{", 1)[0]
        wg, src = None, None
        for m in NTID.finditer(header):
            n = math.prod(int(x) for x in m.groups()[1:] if x)
            if m.group(1) == "reqntid" or src is None:
                wg, src = n, m.group(1)
        dyn = re.search(r"\b" + re.escape(name) + r"_shared_mem\b", body.split("{", 1)[-1]) is not None
        out[name] = {"ptx_wg": wg, "ptx_wg_src": src, "dyn_smem": dyn}
    return out


def demangle(names):
    tool = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not names or not tool:
        return {n: n for n in names}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    if len(out) != len(names):
        return {n: n for n in names}
    return {n: d.replace("typeinfo name for ", "", 1) for n, d in zip(names, out)}


def cuda_path_of(build_dir):
    for f in sorted(glob.glob(os.path.join(build_dir, "src", "CMakeFiles", "*", "link.txt")))[:1] + \
            [os.path.join(build_dir, "CMakeCache.txt")]:
        try:
            m = re.search(r"cuda-path=([^\s;\"]+)", open(f, errors="replace").read())
        except OSError:
            continue
        if m:
            return m.group(1)
    return None


def find_cuobjdump(explicit, build_dir):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    roots = [os.environ.get(e) for e in ("CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT")]
    roots += [cuda_path_of(build_dir), "/usr/local/cuda", "/usr/local/cuda-12.9"]
    for r in roots:
        if r and os.path.isfile(os.path.join(r, "bin", "cuobjdump")):
            return os.path.join(r, "bin", "cuobjdump")
    return shutil.which("cuobjdump")


def resolve_lib(build_dir, lib):
    if os.path.isfile(lib):
        return lib
    name = os.path.basename(lib)
    stem = name if name.startswith("lib") else "lib" + name
    for cand in (os.path.join(build_dir, "src", stem + ".so"), os.path.join(build_dir, "src", stem + ".dylib"),
                 os.path.join(build_dir, lib)):
        if os.path.isfile(cand):
            return cand
    return None


def cuobjdump(tool, flag, blob):
    with tempfile.NamedTemporaryFile(suffix=".fatbin", delete=False) as t:
        t.write(blob)
    try:
        run = subprocess.run([tool, flag, t.name], capture_output=True, text=True)
    finally:
        os.unlink(t.name)
    if run.returncode != 0:
        raise RuntimeError(f"cuobjdump {flag} failed: {run.stderr.strip()[:300]}")
    return run.stdout


def records(lib, tool):
    """{demangled: record} for every .entry in the library's NVPTX images; None if no image."""
    blobs = list(cdc.fatbins(lib))
    if not blobs:
        return None
    out = {}
    for blob in blobs:
        sass_text = cuobjdump(tool, "-sass", blob)
        digests = {}
        sass, res = parse_sass(sass_text, digests), parse_res(cuobjdump(tool, "-res-usage", blob))
        arch = ",".join(sorted(set(re.findall(r"arch = (sm_\d+)", sass_text)))) or "?"
        ptx = cuobjdump(tool, "-ptx", blob)
        calls, launch = dict(cdc.scan_ptx(ptx)), parse_ptx_launch(ptx)
        for name in sass:
            if name in out or name not in res:
                continue
            r = dict(res[name])
            r["total"] = sass[name]["total"]
            for k in tuple(CLASSES) + EXTRA:
                r[k] = sass[name][k]
            r["ptx_calls"] = dict(collections.Counter(calls.get(name, [])))
            r.update(launch.get(name, {"ptx_wg": None, "ptx_wg_src": None, "dyn_smem": False}))
            r["arch"], r["sass_sha256"] = arch, digests[name]
            out[name] = r
    dem = demangle(sorted(out))
    for n, r in out.items():
        r["mangled"], r["name"] = n, dem[n]
    return {r["name"]: r for r in out.values()}


def read_table(spec, what):
    """None (no table), an int (every kernel) or [(regex, value text)] from a table file."""
    if spec is None or spec == "none":
        return None
    if re.fullmatch(r"\d+", spec):
        return int(spec)
    if not os.path.isfile(spec):
        raise SystemExit(f"sass_parity: --{what} {spec}: not an integer and not a file")
    rows = []
    for ln, line in enumerate(open(spec), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"\d+(?:,\d+)*|\\\d", parts[0]):
            raise SystemExit(f"sass_parity: {spec}:{ln}: expected '<int[,int...]|\\N> <regex>'")
        rows.append((re.compile(parts[1]), parts[0]))
    return rows


def lookup(table, name):
    if table is None:
        return None
    if isinstance(table, int):
        return table
    for rx, val in table:
        m = rx.search(name)
        if m:
            if val.startswith("\\"):
                return int(m.group(int(val[1:])))
            return [int(v) for v in val.split(",")] if "," in val else int(val)
    return None


def launch_of(rec, name, wg_table, smem_table):
    """(wg, wg source, dynamic SMEM bytes, warnings) for one kernel record."""
    warns = []
    wg, src = rec.get("ptx_wg"), rec.get("ptx_wg_src")
    if wg is None:
        wg = lookup(wg_table, name)
        src = "table" if wg is not None else "unknown"
    dyn = lookup(smem_table, name)
    if dyn is None:
        dyn = 0
        if rec.get("dyn_smem"):
            warns.append("dynamic SMEM in use but not in --smem: SMEM limit counts static only")
    return wg, src, dyn, warns


def occupancy(regs, smem, wg):
    """Resident work-groups per SM and its limiters: the §11 register gate generalised."""
    warps = math.ceil(wg / 32)
    per_sub = math.ceil(warps / SM89["subpartitions"]) * 32 * (math.ceil(max(regs, 1) / 8) * 8)
    by_reg = SM89["regs_per_subpartition"] // per_sub
    alloc = math.ceil(smem / SM89["smem_granule"]) * SM89["smem_granule"] + SM89["smem_reserved"]
    by_smem = SM89["smem_per_sm"] // alloc
    by_warps = SM89["max_warps"] // warps
    return {"blocks": min(by_reg, by_smem, by_warps, SM89["max_blocks"]),
            "by_reg": by_reg, "by_smem": by_smem, "by_warps": by_warps}


def blocks_per_sm(regs, wg, smem=0):
    return occupancy(regs, smem, wg)["blocks"]


def occupancy_steps(b, c, dyn):
    """WGs (multiples of 32 up to 1024) at which blocks/SM differs between b and c."""
    return [wg for wg in range(32, 1025, 32)
            if blocks_per_sm(b["REG"], wg, b["SMEM"] + dyn) != blocks_per_sm(c["REG"], wg, c["SMEM"] + dyn)]


def compare(b, c, mode, wg, dyn=0):
    """{verdict, fails, ab, notes, warns, occ}: verdict FAIL > A/B > pass. A list dyn (an --smem
    row of representative sizes for a run-time-sized SLM) is the worst verdict over its values."""
    if not isinstance(dyn, list):
        return compare_one(b, c, mode, wg, dyn)
    runs = [(d, compare_one(b, c, mode, wg, d)) for d in dyn]
    out = {"verdict": max((r["verdict"] for _, r in runs), key=("pass", "A/B", "FAIL").index),
           "occ": runs[0][1]["occ"]}
    for key in ("fails", "ab", "notes", "warns"):
        seen = []
        for d, r in runs:
            seen += [m if key in ("notes", "warns") else f"{m} [dyn SMEM {d}]" for m in r[key]
                     if not (key in ("notes", "warns") and m in seen)]
        out[key] = seen
    return out


def compare_one(b, c, mode, wg, dyn=0):
    fails, ab, notes, warns = [], [], [], []

    def fail(cond, msg):
        if cond:
            fails.append(msg)

    for k in ("LDL/STL", "STACK", "LMEM"):
        fail(c[k] > b[k], f"spill {k} {b[k]}->{c[k]}")
    fail(c["SMEM"] > b["SMEM"], f"SMEM {b['SMEM']}->{c['SMEM']}")
    fail(c["LDgen"] > b["LDgen"], f"generic LD/ST {b['LDgen']}->{c['LDgen']}")
    if c["LDgen"] < b["LDgen"]:
        notes.append(f"conversion: generic LD/ST {b['LDgen']}->{c['LDgen']}")
    for callee, n in c.get("ptx_calls", {}).items():
        banned = BANNED_CALL.search(callee) or cdc.BANNED_CALLEE.search(callee)
        fail(banned and n > b.get("ptx_calls", {}).get(callee, 0), f"PTX call {callee} x{n}")

    occ = None
    if wg:
        ob, oc = occupancy(b["REG"], b["SMEM"] + dyn, wg), occupancy(c["REG"], c["SMEM"] + dyn, wg)
        occ = {"base": ob, "cand": oc}
        step = (f"occupancy WG={wg}: {ob['blocks']}->{oc['blocks']} blocks/SM "
                f"(REG {b['REG']}->{c['REG']}, SMEM {b['SMEM'] + dyn}->{c['SMEM'] + dyn})")
        if oc["blocks"] < ob["blocks"] and mode == "algo":
            fails.append(step)
        elif oc["blocks"] != ob["blocks"]:
            ab.append(step)
        occ_same = oc["blocks"] == ob["blocks"]
    else:
        steps = occupancy_steps(b, c, dyn)
        occ_same = not steps
        if steps:
            warns.append(f"launch WG unknown: blocks/SM changes at WG {steps[0]}..{steps[-1]} "
                         f"({len(steps)} sizes); add the kernel to --wg")
        else:
            warns.append("launch WG unknown: blocks/SM is equal at every WG")

    dt = (c["total"] - b["total"]) / max(b["total"], 1)
    band = BAND[mode]
    over = [k for k in HARD if c[k] > b[k]]
    shrink = dt < -band and not over and c["REG"] <= b["REG"] + 2 and occ_same
    tot = f"total {b['total']}->{c['total']} ({dt:+.1%})"
    if shrink:
        notes.append(f"shrink: {tot}")
    elif dt < -band and not over and c["REG"] <= b["REG"] + 2 and wg:
        notes.append(f"shrink: {tot}, but blocks/SM changes, so the A/B decides")
    elif dt < -band and not over and c["REG"] < b["REG"] and c["SMEM"] <= b["SMEM"]:
        # Unknown WG and the only occupancy change is a REG decrease: blocks/SM can only rise.
        steps = occupancy_steps(b, c, dyn)
        notes.append(f"shrink: {tot}, but blocks/SM changes, so the A/B decides")
        ab.append(f"launch WG unknown: REG {b['REG']}->{c['REG']} raises blocks/SM at WG "
                  f"{steps[0]}..{steps[-1]} ({len(steps)} sizes)")
    elif dt > band:
        fails.append(f"{tot}, > {band:.0%}")
    elif dt < -band:
        why = (f"{'/'.join(over)} up" if over else f"REG {b['REG']}->{c['REG']}" if c["REG"] > b["REG"] + 2
               else "blocks/SM changes")
        fails.append(f"{tot}, < -{band:.0%} and not a shrink ({why})")
    if mode == "strict":
        for k in ("BAR", "SHFL", "LDS", "STS", "LDG", "STG"):
            fail(c[k] > b[k] or (c[k] < b[k] and not shrink), f"{k} {b[k]}->{c[k]}")
        fail(c["REG"] > b["REG"] + 2, f"REG {b['REG']}->{c['REG']} (> +2)")
        fail(c["FMA"] < b["FMA"], f"FMA {b['FMA']}->{c['FMA']}")
    else:
        fail(c["BAR"] > b["BAR"], f"BAR {b['BAR']}->{c['BAR']}")
        fail(c["REG"] > b["REG"] + max(2, math.floor(0.05 * b["REG"])), f"REG {b['REG']}->{c['REG']}")
        fail(c["FMA"] < 0.9 * b["FMA"], f"FMA {b['FMA']}->{c['FMA']} (< 0.9x)")
        if 0.05 < dt <= band:
            warns.append(f"total {dt:+.1%}")
        warns.append("algo mode: the runtime A/B rule (agent-guide §10) is still required")
    for k in ("IMADW", "CVT", "BRA"):
        if c[k] != b[k]:
            warns.append(f"{k} {b[k]}->{c[k]}")
    verdict = "FAIL" if fails else "A/B" if ab else "pass"
    return {"verdict": verdict, "fails": fails, "ab": ab, "notes": notes, "warns": warns, "occ": occ}


def exit_code(verdicts, empty=False):
    if empty or "FAIL" in verdicts:
        return 1
    return 2 if "A/B" in verdicts else 0


def short(name, width=78):
    name = name.replace("(anonymous namespace)::", "")
    return name if len(name) <= width else name[:width - 3] + "..."


def strip_match(rx, name):
    m = rx.search(name)
    return None if not m else name[:m.start()] + "\0" + name[m.end():]


def pair_kernels(base, cand, pairs, filt):
    keep = (lambda n: filt.search(n)) if filt else (lambda n: True)
    if not pairs:
        common = [n for n in sorted(base) if n in cand and keep(n)]
        return [(n, n) for n in common], []
    out, empty = [], []
    for spec in pairs:
        b_rx, c_rx = (re.compile(s) for s in spec.split("=", 1))
        bk = {k: n for n in base if keep(n) for k in [strip_match(b_rx, n)] if k}
        ck = {k: n for n in cand if keep(n) for k in [strip_match(c_rx, n)] if k}
        got = [(bk[k], ck[k]) for k in sorted(bk) if k in ck]
        if not got and len(bk) == 1 and len(ck) == 1:
            got = [(next(iter(bk.values())), next(iter(ck.values())))]
        out += got
        if not got:
            empty.append(spec)
    return out, empty


COLS = ("REG", "STACK", "LMEM", "SMEM", "total", "FMA", "LDG", "STG", "LDS", "STS", "BAR", "SHFL",
        "LDL/STL", "LDgen", "IMADW", "BRA", "CALL")


def report(bn, cn, b, c, wg, wg_src, r):
    occ = f", {r['occ']['base']['blocks']}->{r['occ']['cand']['blocks']} blocks/SM" if r["occ"] else ""
    same = {True: ", SASS identical", False: ", SASS differs"}.get(r.get("identical"), "")
    print(f"\n{r['verdict']:4}  {short(bn)}\n      vs {short(cn)}   (WG {wg or '?'} {wg_src}{occ}{same})")
    print("  " + "".join(k.rjust(10) for k in COLS))
    print("  " + "".join(f"{b[k]}/{c[k]}".rjust(10) for k in COLS))
    calls = sorted(set(b.get("ptx_calls", {})) | set(c.get("ptx_calls", {})))
    if calls:
        print("  ptx calls: " + ", ".join(f"{k} {b['ptx_calls'].get(k, 0)}/{c['ptx_calls'].get(k, 0)}"
                                          for k in calls))
    for key, tag in (("fails", "fail"), ("ab", "A/B"), ("notes", "note"), ("warns", "warn")):
        for m in r[key]:
            print(f"  {tag}: {m}")


def identical_report(results, only, empty):
    """The --identical verdict: 0 when every pair's SASS text is byte-identical and no kernel exists
    in one build only, else 1. This is the proof the tuned-table SASS-identity rehash needs."""
    same = [r for r in results if r["identical"]]
    for r in results:
        if not r["identical"]:
            print(f"  differs: {short(r['cand'])}")
    one = sum(map(len, only.values()))
    status = 1 if empty or one or len(same) != len(results) else 0
    print(f"sass_parity: identical {len(same)}/{len(results)} pairs, {len(results) - len(same)} differ, "
          f"{one} in one build only -> exit {status}")
    return status


def toolchain_id(build_dir, tool, arch):
    cxx = None
    try:
        m = re.search(r"^CMAKE_CXX_COMPILER:\w+=(.+)$", open(os.path.join(build_dir, "CMakeCache.txt")).read(), re.M)
        cxx = m.group(1).strip() if m else None
    except OSError:
        pass
    ver = "?"
    if cxx and os.path.isfile(cxx):
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()
        ver = " ".join(ln.strip() for ln in out if ln.strip() and not ln.startswith(
            ("Target:", "Thread model:", "InstalledDir:", "Build config:", "Configuration file:")))
    cuo = subprocess.run([tool, "--version"], capture_output=True, text=True).stdout.splitlines()
    rel = [ln.strip() for ln in cuo if "release" in ln]
    return {"cxx": ver, "cuda": rel[0] if rel else "?", "arch": arch}


SNAP_KEYS = RES_KEYS + ("total",) + tuple(CLASSES) + EXTRA


def snapshot_write(a, rec, lib, tool):
    sel = [re.compile(s) for s in a.select]
    if not sel:
        print("sass_parity: --snapshot-write needs at least one --select regex")
        return 1
    wg_t, smem_t = read_table(a.wg, "wg"), read_table(a.smem, "smem")
    kernels, archs = {}, set()
    for name in sorted(rec):
        if not any(rx.search(name) for rx in sel):
            continue
        r = rec[name]
        wg, src, dyn, warns = launch_of(r, name, wg_t, smem_t)
        k = {key: r[key] for key in SNAP_KEYS}
        k.update({"WG": wg, "WG_src": src, "dyn_SMEM": dyn,
                  "blocks_per_sm": ([blocks_per_sm(r["REG"], wg, r["SMEM"] + d) for d in dyn] if isinstance(dyn, list)
                                    else blocks_per_sm(r["REG"], wg, r["SMEM"] + dyn)) if wg else None,
                  "ptx_calls": r["ptx_calls"]})
        kernels[name] = k
        archs.add(r["arch"])
        for w in warns:
            print(f"sass_parity: warn {short(name)}: {w}")
    if not kernels:
        print(f"sass_parity: --select matched no kernel in {lib}")
        return 1
    snap = {"lib": os.path.basename(lib).split(".")[0][3:], "select": a.select,
            "toolchain": toolchain_id(a.cand, tool, ",".join(sorted(archs)))}
    body = ",\n".join(f"  {json.dumps(n)}: {json.dumps(k, sort_keys=True)}" for n, k in kernels.items())
    head = json.dumps(snap, indent=1)[:-2]
    with open(a.snapshot_write, "w") as fh:
        fh.write(f"{head},\n \"kernels\": {{\n{body}\n }}\n}}\n")
    print(f"sass_parity: wrote {len(kernels)} kernels of {snap['lib']} to {a.snapshot_write}")
    return 0


def snapshot_check(a, tool):
    status, verdicts, skipped = 0, [], 0
    for path in a.snapshot_check:
        snap = json.load(open(path))
        lib = resolve_lib(a.cand, snap["lib"])
        if not lib:
            print(f"sass_parity: {snap['lib']} not built in {a.cand}: skip {path}")
            skipped += 1
            continue
        rec = records(lib, tool)
        arch = snap["toolchain"]["arch"]
        if rec is None or not any(r["arch"] == arch for r in rec.values()):
            print(f"sass_parity: no {arch} image in {lib}: skip {path}")
            skipped += 1
            continue
        tc = toolchain_id(a.cand, tool, arch)
        if tc != snap["toolchain"]:
            print(f"sass_parity: warn: toolchain {tc} differs from the snapshot's {snap['toolchain']}")
        sel = [re.compile(s) for s in snap["select"]]
        new = [n for n in rec if any(rx.search(n) for rx in sel) and n not in snap["kernels"]]
        for n in new:
            print(f"sass_parity: warn: {short(n)} matches the snapshot's select but is not in it")
        npass = 0
        for name, b in snap["kernels"].items():
            c = rec.get(name)
            if c is None:
                print(f"\nFAIL  {short(name)}: in the snapshot, not in {lib}")
                verdicts.append("FAIL")
                continue
            r = compare(b, c, "algo", b["WG"], b["dyn_SMEM"])
            r["warns"] = [w for w in r["warns"] if not w.startswith("algo mode")]
            verdicts.append(r["verdict"])
            npass += r["verdict"] == "pass"
            if r["verdict"] != "pass" or a.verbose:
                report(name, name, b, c, b["WG"], b["WG_src"], r)
        print(f"sass_parity: {path}: {len(snap['kernels'])} kernels, {npass} pass")
    if skipped == len(a.snapshot_check):
        return 77
    status = exit_code(verdicts)
    print(f"sass_parity: snapshot check -> exit {status}")
    return status


def build(dirs, target):
    for d in dict.fromkeys(dirs):
        rc = subprocess.run([os.path.join(REPO, "scripts", "b"), "-B", d, target]).returncode
        if rc != 0:
            print(f"sass_parity: scripts/b -B {d} {target} exited {rc}")
            return rc
    return 0


def self_test():
    """The rules on synthetic records; each case is a deliberate break of one rule."""
    def rec(**kw):
        r = {k: 0 for k in SNAP_KEYS}
        r.update({"REG": 62, "LDgen": 1, "total": 1000, "FMA": 100, "LDG": 10, "STG": 10, "LDS": 20, "STS": 20,
                  "BAR": 4, "SHFL": 8, "ptx_calls": {}})
        r.update(kw)
        return r

    base = rec()
    cases = [
        ("identical", base, "strict", 256, 0, "pass", None),
        ("REG +2, same step", rec(REG=64), "strict", 256, 0, "pass", None),
        ("REG down, step up (strict)", rec(REG=48), "strict", 256, 0, "A/B", "occupancy"),
        ("REG down, step up (algo)", rec(REG=48), "algo", 256, 0, "A/B", "occupancy"),
        ("REG up, step down (algo)", rec(REG=72), "algo", 256, 0, "FAIL", "occupancy"),
        ("SLM-capped: no step", rec(REG=48), "strict", 256, 40000, "pass", None),
        ("shrink", rec(total=900, LDS=18), "strict", 256, 0, "pass", "shrink"),
        ("shrink, LDS up", rec(total=900, LDS=21), "strict", 256, 0, "FAIL", "LDS up"),
        ("shrink, step up", rec(total=900, REG=48), "strict", 256, 0, "A/B", "A/B decides"),
        ("shrink, unknown WG, no step", rec(total=900, REG=63), "strict", None, 0, "pass", "shrink"),
        ("shrink, unknown WG, REG down", rec(total=900, REG=40), "strict", None, 0, "A/B", "raises blocks/SM"),
        ("shrink, unknown WG, REG down, LDS down", rec(total=900, REG=40, LDS=18), "strict", None, 0, "FAIL", "LDS 20->18"),
        ("dyn SMEM list: SLM binds at one size", rec(REG=48), "strict", 256, [0, 40000], "A/B", "[dyn SMEM 0]"),
        ("dyn SMEM list: SLM binds at every size", rec(REG=48), "strict", 256, [40000, 45000], "pass", None),
        ("growth", rec(total=1040), "strict", 256, 0, "FAIL", "total"),
        ("in-band class drop", rec(LDS=19), "strict", 256, 0, "FAIL", "LDS 20->19"),
        ("conversion", rec(LDgen=0), "strict", 256, 0, "pass", "conversion"),
        ("generic up", rec(LDgen=2), "strict", 256, 0, "FAIL", "generic"),
        ("SMEM up", rec(SMEM=16), "strict", 256, 0, "FAIL", "SMEM"),
        ("algo growth in band", rec(total=1080), "algo", 256, 0, "pass", None),
        ("algo shrink", rec(total=800, LDG=5), "algo", 256, 0, "pass", "shrink"),
        ("banned call", rec(ptx_calls={"__mulsc3": 1}), "algo", 256, 0, "FAIL", "PTX call"),
    ]
    bad = 0
    cases.append(("REG up, step down (strict)", rec(REG=66), "strict", 32, 0, "A/B", "8->7", rec(REG=64)))
    cases.append(("shrink, unknown WG, REG up 2, step", rec(total=900, REG=65), "strict", None, 0, "FAIL", "blocks/SM",
                  rec(REG=63)))
    for label, c, mode, wg, dyn, want, needle, *b in cases:
        r = compare(b[0] if b else base, c, mode, wg, dyn)
        text = " | ".join(r["fails"] + r["ab"] + r["notes"] + r["warns"])
        ok = r["verdict"] == want and (needle is None or needle in text)
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {label}: {r['verdict']} (want {want}) {text}")
    hdr = (".weak .entry K(\n.param .u32 K_param_0\n)\n.reqntid 128, 1, 1\n{\nld.shared.u32 %r1, [K_shared_mem];\n}\n"
           ".weak .entry M(\n.param .u32 M_param_0\n)\n.maxntid 64, 2, 1\n.minnctapersm 1\n{\nret;\n}\n"
           ".weak .entry U(\n)\n{\nret;\n}\n")
    got = parse_ptx_launch(hdr)
    want = {"K": {"ptx_wg": 128, "ptx_wg_src": "reqntid", "dyn_smem": True},
            "M": {"ptx_wg": 128, "ptx_wg_src": "maxntid", "dyn_smem": False},
            "U": {"ptx_wg": None, "ptx_wg_src": None, "dyn_smem": False}}
    bad += got != want
    print(f"  {'ok ' if got == want else 'BAD'} PTX .reqntid/.maxntid: {got}")
    table = [(re.compile(r"LatrdLowerPanelKernel(?:Legacy|Grid)?<[^,]+, (\d+),"), "\\1"),
             (re.compile(r"^batchlas::PosvTinyKernel<"), "64"), (re.compile(r"Tridiag<"), "416,800")]
    lk = [lookup(table, n) for n in ("batchlas::LatrdLowerPanelKernelLegacy<float, 256, true>",
                                     "batchlas::PosvTinyKernel<double, 8, 8, 1, 1, true>", "batchlas::Other<128>",
                                     "batchlas::Tridiag<float>")]
    want_lk = [256, 64, None, [416, 800]]
    bad += lk != want_lk
    print(f"  {'ok ' if lk == want_lk else 'BAD'} table lookup: {lk} (want {want_lk})")
    sass = ("\t\tFunction : _Z1Kv\n        /*0000*/  MOV R1, c[0x0][0x28] ;  /* 0x00000a0000017a02 */\n"
            "                                    /* 0x000fe40000000f00 */\n"
            "        /*0010*/  NOP ;  /* 0x0000000000007918 */\n                    /* 0x000fc00000000000 */\n")
    variants = {"same": sass, "encoding": sass.replace("0x000fe40000000f00", "0x000fe40000000f01"),
                "nop": sass.replace("NOP", "BRA"), "indent": sass.replace("        /*0010*/", "  /*0010*/")}
    dig = {}
    for k, text in variants.items():
        d = {}
        parse_sass(text, d)
        dig[k] = d["_Z1Kv"]
    want_dig = {"encoding": False, "nop": False, "indent": True}
    got_dig = {k: dig[k] == dig["same"] for k in want_dig}
    bad += got_dig != want_dig
    print(f"  {'ok ' if got_dig == want_dig else 'BAD'} SASS digest equal to base: {got_dig} (want {want_dig})")
    rs = [{"identical": True, "cand": "A"}, {"identical": False, "cand": "B"}]
    idc = [identical_report(rs[:1], {}, False), identical_report(rs, {}, False),
           identical_report(rs[:1], {"base": ["C"], "cand": []}, False), identical_report([], {}, True)]
    bad += idc != [0, 1, 1, 1]
    print(f"  {'ok ' if idc == [0, 1, 1, 1] else 'BAD'} --identical exit codes: {idc} (want [0, 1, 1, 1])")
    codes = [exit_code(v) for v in (["pass"], ["pass", "A/B"], ["A/B", "FAIL"])]
    bad += codes != [0, 2, 1]
    print(f"  {'ok ' if codes == [0, 2, 1] else 'BAD'} exit codes: {codes} (want [0, 2, 1])")
    print(f"sass_parity: self-test {'FAILED' if bad else 'passed'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="build dir at the merge-base")
    ap.add_argument("--cand", help="candidate build dir (may equal --base)")
    ap.add_argument("--lib", help="library target (batchlas_extensions_latrd) or .so path")
    ap.add_argument("--pair", action="append", default=[], help="'BaseRegex=CandRegex' on demangled names")
    ap.add_argument("--filter", help="regex both demangled names must match (e.g. '<float, 128, false>')")
    ap.add_argument("--mode", choices=("strict", "algo"), default="strict")
    ap.add_argument("--wg", default=DEFAULT_WG_TABLE, help="launch WG: a table file, an integer, or none")
    ap.add_argument("--smem", default=DEFAULT_SMEM_TABLE,
                    help="dynamic SMEM bytes per WG: a table file, an integer, or none")
    ap.add_argument("--build", action="store_true", help="scripts/b the --lib target in both trees first")
    ap.add_argument("--cuobjdump")
    ap.add_argument("--json", help="output path (default: <cand>/logs/sass_parity/<stamp>-<lib>.json)")
    ap.add_argument("--snapshot-write", metavar="FILE", help="write a snapshot of the --select kernels of --lib")
    ap.add_argument("--select", action="append", default=[], help="snapshot kernel regex (repeatable)")
    ap.add_argument("--snapshot-check", metavar="FILE", action="append",
                    help="compare --cand against a snapshot with the algo thresholds (repeatable)")
    ap.add_argument("--identical", action="store_true",
                    help="verdict = byte-identical SASS text of every pair (and equal kernel sets when unpaired)")
    ap.add_argument("--self-test", action="store_true", help="run the rule self-test and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="snapshot check: print passing kernels too")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.cand:
        ap.error("--cand is required")
    tool = find_cuobjdump(a.cuobjdump, a.cand)
    if a.snapshot_check:
        if not tool:
            print("sass_parity: cuobjdump not found (set CUDA_PATH or pass --cuobjdump): skip")
            return 77
        return snapshot_check(a, tool)
    if not a.lib or not (a.base or a.snapshot_write):
        ap.error("--lib and --base (or --snapshot-write) are required")

    if a.build and build([d for d in (a.base, a.cand) if d], os.path.basename(a.lib)) != 0:
        return 1
    libs = {}
    for side in ("base", "cand") if a.base else ("cand",):
        libs[side] = resolve_lib(getattr(a, side), a.lib)
        if not libs[side]:
            print(f"sass_parity: {a.lib} not found in {getattr(a, side)} (build it, or pass --build)")
            return 1
    if not tool:
        print("sass_parity: cuobjdump not found (set CUDA_PATH or pass --cuobjdump): skip")
        return 77
    rec = {}
    for side in libs:
        rec[side] = rec["base"] if side == "cand" and libs["cand"] == libs.get("base") else records(libs[side], tool)
        if rec[side] is None:
            print(f"sass_parity: no NVPTX image in {libs[side]}: skip")
            return 77
    if a.snapshot_write:
        return snapshot_write(a, rec["cand"], libs["cand"], tool)

    wg_t, smem_t = read_table(a.wg, "wg"), read_table(a.smem, "smem")
    filt = re.compile(a.filter) if a.filter else None
    pairs, empty = pair_kernels(rec["base"], rec["cand"], a.pair, filt)
    for spec in empty:
        print(f"sass_parity: --pair '{spec}' paired no kernels")
    print(f"sass_parity: mode={a.mode} base={libs['base']} cand={libs['cand']} cuobjdump={tool}")
    toolkits = [os.path.realpath(p) if p else None for p in (cuda_path_of(a.base), cuda_path_of(a.cand))]
    if toolkits[0] != toolkits[1]:
        print(f"sass_parity: warn: base and cand use different CUDA toolkits {toolkits[0]} vs {toolkits[1]}"
              " (--cuda-path): the delta includes the ptxas change")
    results = []
    for bn, cn in pairs:
        b, c = rec["base"][bn], rec["cand"][cn]
        wg, src, dyn, lwarns = launch_of(c, cn, wg_t, smem_t)
        r = compare(b, c, a.mode, wg, dyn)
        r["warns"] = lwarns + r["warns"]
        r["identical"] = b["sass_sha256"] == c["sass_sha256"]
        results.append(dict(r, base=bn, cand=cn, wg=wg, wg_src=src, dyn_smem=dyn, base_rec=b, cand_rec=c))
        report(bn, cn, b, c, wg, src, r)
    only = {} if a.pair else {side: sorted(set(rec[side]) - set(rec[other]))
                              for side, other in (("base", "cand"), ("cand", "base"))}
    if any(only.values()):
        print(f"\nsass_parity: {sum(map(len, only.values()))} kernels exist in one build only (not compared)")
        if a.identical:
            for side, names in only.items():
                for n in names:
                    print(f"  {side} only: {n}")
    status = exit_code([r["verdict"] for r in results], bool(empty) or not pairs)
    count = collections.Counter(r["verdict"] for r in results)
    print(f"\nsass_parity: {len(results)} pairs, {count['pass']} pass, {count['A/B']} A/B, "
          f"{count['FAIL']} fail" + ("" if a.identical else f" -> exit {status}"))
    if a.identical:
        status = identical_report(results, only, bool(empty) or not pairs)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = a.json or os.path.join(a.cand, "logs", "sass_parity",
                                  f"{stamp}-{os.path.basename(libs['cand']).split('.')[0]}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"args": vars(a), "libs": libs, "cuobjdump": tool, "exit": status, "empty_pairs": empty,
                   "one_side_only": only, "pairs": results}, fh, indent=1)
    print(f"sass_parity: json {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
