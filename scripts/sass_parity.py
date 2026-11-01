#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SASS-parity gate: compare the sm_89 code of kernel pairs between two builds.

    scripts/sass_parity.py --base ../main/build --cand build --lib batchlas_extensions_latrd
    scripts/sass_parity.py --base build --cand build --lib batchlas_extensions_latrd \\
        --pair 'LatrdLowerPanelKernelLegacy<=LatrdLowerPanelKernel<' --filter '<float, 128, false>'

A device-LA migration must not make a kernel's machine code worse. Per kernel it records
REG, STACK, LMEM, static SMEM, instructions (excluding NOP), a class histogram and the PTX
call targets, then applies the strict (1:1 port) or algo (algorithm change) thresholds.

Pairing: without --pair a kernel is compared with the same demangled name in the other
build. With --pair 'B=C' a base kernel matching B pairs with the cand kernel matching C
whose name is equal once each match is cut out (Legacy<float, 128, false> pairs with
Kernel<float, 128, false>). Base and cand may be the same build dir.

Exit codes: 0 pass, 1 fail (or nothing paired), 77 no NVPTX image or no cuobjdump.
evidence: ~/dla-v2-notes/synthesis.txt section 4; class map from the sass_lab harness.
"""
import argparse
import collections
import datetime
import glob
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
TABLE_COLS = RES_KEYS + ("total",) + tuple(CLASSES) + EXTRA
BANNED_CALL = re.compile(r"^_ZN8batchlas6device|^__(mul|div)[sd]c3$")

INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
SM89 = {"subpartitions": 4, "regs_per_subpartition": 16384, "max_warps": 48, "max_blocks": 24}


def opcode(text):
    tok = text.split()
    if tok and tok[0].startswith("@"):
        tok = tok[1:]
    return tok[0] if tok else ""


def parse_sass(text):
    funcs, cur = {}, None
    for line in text.splitlines():
        if "Function :" in line:
            cur = line.split("Function :", 1)[1].strip()
            funcs[cur] = collections.Counter()
            continue
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
    """{mangled: record} for every .entry in the library's sm_89 images; None if no image."""
    blobs = list(cdc.fatbins(lib))
    if not blobs:
        return None
    out = {}
    for blob in blobs:
        sass, res = parse_sass(cuobjdump(tool, "-sass", blob)), parse_res(cuobjdump(tool, "-res-usage", blob))
        calls = dict(cdc.scan_ptx(cuobjdump(tool, "-ptx", blob)))
        for name in sass:
            if name in out or name not in res:
                continue
            r = dict(res[name])
            r["total"] = sass[name]["total"]
            for k in tuple(CLASSES) + EXTRA:
                r[k] = sass[name][k]
            r["ptx_calls"] = dict(collections.Counter(calls.get(name, [])))
            out[name] = r
    dem = demangle(sorted(out))
    for n, r in out.items():
        r["mangled"], r["name"] = n, dem[n]
    return {r["name"]: r for r in out.values()}


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


def launch_wg(name, explicit):
    if explicit:
        return explicit
    ints = [int(x) for x in re.findall(r"(?<![\w.])(\d+)(?=[,>])", name)]
    wg = [x for x in ints if x % 32 == 0 and 32 <= x <= 1024]
    return max(wg) if wg else None


def blocks_per_sm(regs, wg):
    """Resident work-groups per SM: the §11 register gate generalised to a block count."""
    warps = math.ceil(wg / 32)
    per_sub = math.ceil(warps / SM89["subpartitions"]) * 32 * (math.ceil(max(regs, 1) / 8) * 8)
    by_reg = SM89["regs_per_subpartition"] // per_sub
    return min(by_reg, SM89["max_warps"] // warps, SM89["max_blocks"])


def compare(b, c, mode, wg):
    fails, warns = [], []

    def fail(cond, msg):
        if cond:
            fails.append(msg)

    for k in ("LDL/STL", "STACK", "LMEM"):
        fail(c[k] > b[k], f"spill {k} {b[k]}->{c[k]}")
    fail(c["SMEM"] > b["SMEM"], f"SMEM {b['SMEM']}->{c['SMEM']}")
    fail(c["LDgen"] > b["LDgen"], f"generic LD/ST {b['LDgen']}->{c['LDgen']}")
    for callee, n in c["ptx_calls"].items():
        banned = BANNED_CALL.search(callee) or cdc.BANNED_CALLEE.search(callee)
        fail(banned and n > b["ptx_calls"].get(callee, 0), f"PTX call {callee} x{n}")
    if wg:
        ob, oc = blocks_per_sm(b["REG"], wg), blocks_per_sm(c["REG"], wg)
        fail(oc < ob, f"occupancy step WG={wg}: {ob}->{oc} blocks/SM (REG {b['REG']}->{c['REG']})")
    else:
        warns.append("launch WG unknown: occupancy rule skipped (pass --wg)")
    dt = (c["total"] - b["total"]) / max(b["total"], 1)
    if mode == "strict":
        for k in ("BAR", "SHFL", "LDS", "STS", "LDG", "STG"):
            fail(c[k] != b[k], f"{k} {b[k]}->{c[k]}")
        fail(c["REG"] > b["REG"] + 2, f"REG {b['REG']}->{c['REG']} (> +2)")
        fail(abs(dt) > 0.03, f"total {b['total']}->{c['total']} ({dt:+.1%}, > 3%)")
        fail(c["FMA"] < b["FMA"], f"FMA {b['FMA']}->{c['FMA']}")
    else:
        fail(c["BAR"] > b["BAR"], f"BAR {b['BAR']}->{c['BAR']}")
        fail(c["REG"] > b["REG"] + max(2, math.floor(0.05 * b["REG"])), f"REG {b['REG']}->{c['REG']}")
        fail(c["FMA"] < 0.9 * b["FMA"], f"FMA {b['FMA']}->{c['FMA']} (< 0.9x)")
        fail(abs(dt) > 0.10, f"total {b['total']}->{c['total']} ({dt:+.1%}, > 10%)")
        if 0.05 < abs(dt) <= 0.10:
            warns.append(f"total {dt:+.1%}")
        warns.append("algo mode: the runtime A/B rule (agent-guide §10) is still required")
    for k in ("IMADW", "CVT", "BRA"):
        if c[k] != b[k]:
            warns.append(f"{k} {b[k]}->{c[k]}")
    return fails, warns


def short(name, width=78):
    name = name.replace("(anonymous namespace)::", "")
    return name if len(name) <= width else name[:width - 3] + "..."


def build(dirs, target):
    for d in dict.fromkeys(dirs):
        rc = subprocess.run([os.path.join(REPO, "scripts", "b"), "-B", d, target]).returncode
        if rc != 0:
            print(f"sass_parity: scripts/b -B {d} {target} exited {rc}")
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="build dir at the merge-base")
    ap.add_argument("--cand", required=True, help="candidate build dir (may equal --base)")
    ap.add_argument("--lib", required=True, help="library target (batchlas_extensions_latrd) or .so path")
    ap.add_argument("--pair", action="append", default=[], help="'BaseRegex=CandRegex' on demangled names")
    ap.add_argument("--filter", help="regex both demangled names must match (e.g. '<float, 128, false>')")
    ap.add_argument("--mode", choices=("strict", "algo"), default="strict")
    ap.add_argument("--wg", type=int, help="launch work-group size (default: largest 32k..1024 template int)")
    ap.add_argument("--build", action="store_true", help="scripts/b the --lib target in both trees first")
    ap.add_argument("--cuobjdump")
    ap.add_argument("--json", help="output path (default: <cand>/logs/sass_parity/<stamp>-<lib>.json)")
    a = ap.parse_args()

    if a.build and build([a.base, a.cand], os.path.basename(a.lib)) != 0:
        return 1
    libs = {}
    for side in ("base", "cand"):
        libs[side] = resolve_lib(getattr(a, side), a.lib)
        if not libs[side]:
            print(f"sass_parity: {a.lib} not found in {getattr(a, side)} (build it, or pass --build)")
            return 1
    tool = find_cuobjdump(a.cuobjdump, a.cand)
    if not tool:
        print("sass_parity: cuobjdump not found (set CUDA_PATH or pass --cuobjdump): skip")
        return 77
    rec = {}
    for side in ("base", "cand"):
        rec[side] = rec["base"] if side == "cand" and libs["cand"] == libs["base"] else records(libs[side], tool)
        if rec[side] is None:
            print(f"sass_parity: no NVPTX image in {libs[side]}: skip")
            return 77

    filt = re.compile(a.filter) if a.filter else None
    pairs, empty = pair_kernels(rec["base"], rec["cand"], a.pair, filt)
    for spec in empty:
        print(f"sass_parity: --pair '{spec}' paired no kernels")
    status = 1 if empty or not pairs else 0
    cols = ("REG", "STACK", "LMEM", "SMEM", "total", "FMA", "LDG", "STG", "LDS", "STS", "BAR", "SHFL",
            "LDL/STL", "LDgen", "IMADW", "BRA", "CALL")
    print(f"sass_parity: mode={a.mode} base={libs['base']} cand={libs['cand']} cuobjdump={tool}")
    results = []
    for bn, cn in pairs:
        b, c = rec["base"][bn], rec["cand"][cn]
        wg = launch_wg(cn, a.wg)
        fails, warns = compare(b, c, a.mode, wg)
        status |= 1 if fails else 0
        results.append({"base": bn, "cand": cn, "wg": wg, "verdict": "FAIL" if fails else "pass",
                        "fails": fails, "warns": warns, "base_rec": b, "cand_rec": c})
        print(f"\n{'FAIL' if fails else 'pass'}  {short(bn)}\n      vs {short(cn)}   (WG {wg})")
        print("  " + "".join(k.rjust(10) for k in cols))
        print("  " + "".join(f"{b[k]}/{c[k]}".rjust(10) for k in cols))
        calls = sorted(set(b["ptx_calls"]) | set(c["ptx_calls"]))
        if calls:
            print("  ptx calls: " + ", ".join(f"{k} {b['ptx_calls'].get(k, 0)}/{c['ptx_calls'].get(k, 0)}"
                                              for k in calls))
        for f in fails:
            print(f"  fail: {f}")
        for w in warns:
            print(f"  warn: {w}")
    if not a.pair:
        only = sorted(set(rec["base"]) ^ set(rec["cand"]))
        if only:
            print(f"\nsass_parity: {len(only)} kernels exist in one build only (not compared)")
    npass = sum(r["verdict"] == "pass" for r in results)
    print(f"\nsass_parity: {len(results)} pairs, {npass} pass, {len(results) - npass} fail -> exit {status}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = a.json or os.path.join(a.cand, "logs", "sass_parity",
                                  f"{stamp}-{os.path.basename(libs['cand']).split('.')[0]}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"args": vars(a), "libs": libs, "cuobjdump": tool, "exit": status, "empty_pairs": empty,
                   "pairs": results}, fh, indent=1)
    print(f"sass_parity: json {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
