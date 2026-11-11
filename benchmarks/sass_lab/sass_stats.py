#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tabulate cuobjdump -sass / -res-usage output per kernel and per hand/dblas pair.

  sass_stats.py --sass all.sass --res all.res            # lab pairs (lab::K<op><T, N, V>)
  sass_stats.py --sass x.sass --res x.res --match REGEX  # any kernels whose demangled name matches
"""
import argparse
import collections
import re
import shutil
import subprocess
import sys

VARIANTS = {0: "hand", 1: "dblas", 2: "dblas_ws", 3: "lean", 4: "lean_rtwg", 5: "lean_bv",
            6: "dblas_reqd", 7: "dblas_static", 8: "hand_reqd", 9: "device", 10: "device_scaled",
            11: "device_blue", 12: "device_scratch"}

CLASSES = collections.OrderedDict([
    ("FMA", {"FFMA", "DFMA", "HFMA2", "FFMA32I"}),
    ("FPALU", {"FADD", "FMUL", "DADD", "DMUL", "FMNMX", "DMNMX", "FSETP", "DSETP", "MUFU", "FSEL", "FCHK"}),
    ("LDG", {"LDG"}), ("STG", {"STG"}), ("LDS", {"LDS"}), ("STS", {"STS"}),
    ("LDgen", {"LD", "ST"}), ("LDL/STL", {"LDL", "STL"}), ("LDC", {"LDC", "ULDC"}),
    ("BAR", {"BAR", "MEMBAR", "ERRBAR", "CCTL"}),
    ("SHFL", {"SHFL", "REDUX", "VOTE", "VOTEU", "MATCH"}),
    ("BRA", {"BRA", "BRX", "JMP", "JMX"}),
    ("CTRL", {"BSSY", "BSYNC", "WARPSYNC", "CALL", "RET", "BMOV", "BREAK", "YIELD"}),
    ("INT", {"IMAD", "IADD3", "LEA", "LOP3", "SHF", "ISETP", "IMNMX", "IABS", "SEL", "SGXT", "PRMT",
             "UIMAD", "UIADD3", "ULEA", "ULOP3", "USHF", "UISETP", "USEL", "UMOV", "IMUL", "FLO", "POPC",
             "BREV", "UPRMT", "ULDC.64"}),
    ("CVT", {"I2F", "F2I", "F2F", "I2I", "FRND", "I2FP", "F2IP", "DSETP_CVT"}),
    ("MOV", {"MOV", "S2R", "S2UR", "CS2R", "R2UR", "P2R", "R2P", "PLOP3"}),
])
COLS = ["REG", "STK", "LMEM", "SMEM", "total"] + list(CLASSES) + ["IMADW", "CALL", "EXIT"]

INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")


def opcode(text):
    tok = text.split()
    if tok and tok[0].startswith("@"):
        tok = tok[1:]
    return tok[0] if tok else ""


def parse_sass(path):
    funcs, cur = {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Function :" in line:
                cur = line.split("Function :", 1)[1].strip()
                funcs[cur] = collections.Counter()
                continue
            if cur is None:
                continue
            m = INSN.search(line)
            if not m:
                continue
            full = opcode(m.group(1))
            base = full.split(".")[0]
            if base == "NOP":
                continue
            c = funcs[cur]
            c["total"] += 1
            c["op:" + base] += 1
            if full.startswith("IMAD.WIDE"):
                c["IMADW"] += 1
            if base == "CALL":
                c["CALL"] += 1
            if base == "EXIT":
                c["EXIT"] += 1
            for cls, ops in CLASSES.items():
                if base in ops:
                    c[cls] += 1
                    break
    return funcs


def parse_res(path):
    res, cur = {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.match(r"\s*Function (\S+):", line)
            if m:
                cur = m.group(1)
                continue
            if cur and "REG:" in line:
                kv = dict(re.findall(r"(\w+):(\d+)", line))
                res[cur] = {"REG": int(kv.get("REG", 0)), "STK": int(kv.get("STACK", 0)),
                            "LMEM": int(kv.get("LOCAL", 0)), "SMEM": int(kv.get("SHARED", 0))}
                cur = None
    return res


def demangle(names):
    tool = shutil.which("llvm-cxxfilt") or "/opt/dpcpp-cuda/bin/llvm-cxxfilt"
    if not shutil.which(tool):
        tool = shutil.which("c++filt") or "c++filt"
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def row(stats, res):
    d = dict(res or {})
    for k in COLS:
        d.setdefault(k, stats.get(k, 0))
    return d


def fmt_table(title, rows, base=None):
    lines = [title, "  " + "variant".ljust(14) + "".join(c.rjust(8) for c in COLS) + "  d_total"]
    for name, r in rows:
        dt = "" if base is None else f"{r['total'] - base['total']:+d}"
        lines.append("  " + name.ljust(14) + "".join(str(r.get(c, 0)).rjust(8) for c in COLS) + dt.rjust(9))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--match", default=None, help="regex on demangled names; prints one row per kernel")
    ap.add_argument("--csv", default=None, help="also write all rows as CSV here")
    ap.add_argument("--ops", action="store_true", help="print the full opcode histogram per kernel")
    a = ap.parse_args()
    funcs, res = parse_sass(a.sass), parse_res(a.res)
    names = sorted(set(funcs) | set(res))
    dem = demangle(names)
    csv_rows = []
    if a.match:
        rx = re.compile(a.match)
        rows = [(dem[n], row(funcs.get(n, {}), res.get(n))) for n in names if rx.search(dem[n])]
        for nm, r in rows:
            print(fmt_table(nm, [("kernel", r)]))
            if a.ops:
                hist = sorted(((k[3:], v) for k, v in funcs.get(next(x for x in names if dem[x] == nm), {}).items()
                               if k.startswith("op:")), key=lambda kv: -kv[1])
                print("  ops: " + " ".join(f"{k}={v}" for k, v in hist))
        return
    # lab::K<Op><T, N, V> (ops_l*.cc) and lab::KW<Op><T, N, V, WG> (ops_device_*.cc, keyed "<Op>@<WG>")
    rx = re.compile(r"lab::K(\w+)<(float|double), (-?\d+), (\d+)(?:, (\d+))?>")
    groups = collections.defaultdict(dict)
    for n in names:
        m = rx.search(dem[n])
        if not m:
            continue
        op, t, N, V = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if m.group(5):
            op = f"{op}@{m.group(5)}"
        groups[(op, t, N)][V] = row(funcs.get(n, {}), res.get(n))
    for (op, t, N) in sorted(groups):
        g = groups[(op, t, N)]
        base = g.get(0)
        rows = [(VARIANTS.get(v, str(v)), g[v]) for v in sorted(g)]
        print(fmt_table(f"{op} {t} N={'rt' if N == 0 else N}", rows, base))
        print()
        for v in sorted(g):
            csv_rows.append([op, t, N, VARIANTS.get(v, v)] + [g[v].get(c, 0) for c in COLS])
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write(",".join(["op", "type", "N", "variant"] + COLS) + "\n")
            for r in csv_rows:
                fh.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
