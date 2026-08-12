#!/usr/bin/env python3
"""Pull (kernel, registers, spill stores, spill loads, stack) out of a
-Xcuda-ptxas -v build log. Reports every entry, including zero-spill ones,
because 'a candidate that spills is not automatically dead, but the spill
must be reported'."""
import re, sys, subprocess

def demangle(names):
    try:
        p = subprocess.run(["c++filt"], input="\n".join(names), text=True,
                           capture_output=True)
        return p.stdout.splitlines()
    except Exception:
        return names

def parse(path):
    txt = open(path, errors="replace").read()
    # ptxas emits: "Compiling entry function 'X' for 'sm_89'" then
    # "Function properties for X: N bytes stack frame, N bytes spill stores, N bytes spill loads"
    # then "Used N registers"
    rows = []
    cur = None
    for line in txt.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = {"name": m.group(1), "stack": 0, "sst": 0, "sld": 0, "regs": None}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            cur["stack"], cur["sst"], cur["sld"] = map(int, m.groups())
        m = re.search(r"Used (\d+) registers", line)
        if m:
            cur["regs"] = int(m.group(1))
    return rows

for path in sys.argv[1:]:
    rows = parse(path)
    names = demangle([r["name"] for r in rows])
    print(f"##### {path}  ({len(rows)} entries)")
    for r, dn in zip(rows, names):
        # compress the SYCL mangling down to something readable
        short = dn
        short = re.sub(r"sycl::_V1::", "", short)
        short = short[:190]
        flag = "  <<< SPILLS" if (r["sst"] or r["sld"]) else ""
        print(f"  regs={r['regs']:>4}  spill_st={r['sst']:>5}  spill_ld={r['sld']:>5}  stack={r['stack']:>5}  {short}{flag}")
    sp = [r for r in rows if r["sst"] or r["sld"]]
    print(f"  -- {len(sp)}/{len(rows)} entries spill; max regs = {max((r['regs'] or 0) for r in rows)}")
    print()
