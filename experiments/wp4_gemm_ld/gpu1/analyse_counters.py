import csv, sys, re
base = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp4_gemm_ld/gpu1/"

FILES = [
    ("native ld=rows", "raw-b1-p0.csv"),
    ("native ld=512", "raw-b1-p384.csv"),
    ("native B-only ld=512", "raw-b1-Bonly384.csv"),
    ("cuBLAS ld=rows", "vraw-b1-p0.csv"),
    ("cuBLAS ld=512", "vraw-b1-p384.csv"),
]


def load(f):
    rows = list(csv.reader(open(base + f)))
    hdr = rows[0]
    units = rows[1]
    data = rows[-1]
    d = {}
    for h, u, v in zip(hdr, units, data):
        try:
            val = float(v.replace(",", ""))
        except ValueError:
            val = None
        d[h] = (val, u)
    return d


DS = [(n, load(f)) for n, f in FILES]

MULT = {"": 1, "%": 1, "sector": 1, "request": 1, "us": 1e3, "ms": 1e6, "ns": 1,
        "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
        "byte/s": 1, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9, "Tbyte/s": 1e12,
        "cycle": 1, "warp": 1, "block": 1, "thread": 1, "register/thread": 1,
        "inst": 1, "Kinst": 1e3, "Minst": 1e6, "Ginst": 1e9,
        "Ksector": 1e3, "Msector": 1e6, "Gsector": 1e9,
        "Krequest": 1e3, "Mrequest": 1e6, "Grequest": 1e9,
        "Kcycle": 1e3, "Mcycle": 1e6, "Gcycle": 1e9,
        "Kbyte/cycle": 1e3, "byte/cycle": 1, "sector/s": 1, "Gsector/s": 1e9,
        "Ksector/s": 1e3, "Msector/s": 1e6,
        "%/s": 1, "ratio": 1, "cycle/s": 1, "Kcycle/s": 1e3, "Mcycle/s": 1e6, "Gcycle/s": 1e9,
        "Tcycle/s": 1e12, "inst/cycle": 1, "warp/cycle": 1, "cycle/inst": 1, "cycle/warp": 1,
        "sector/request": 1, "byte/request": 1, "Kbyte/request": 1e3,
        }


def g(d, key):
    v, u = d.get(key, (None, ""))
    if v is None:
        return None
    m = MULT.get(u.strip())
    if m is None:
        sys.stderr.write(f"unknown unit {u!r} for {key}\n")
        m = 1
    return v * m


def row(label, fn, fmt="{:.3f}"):
    out = []
    for n, d in DS:
        try:
            v = fn(d)
        except Exception:
            v = None
        out.append("n/a" if v is None else fmt.format(v))
    print("| " + label + " | " + " | ".join(out) + " |")


print("| metric | " + " | ".join(n for n, _ in FILES) + " |")
print("|---" * (len(FILES) + 1) + "|")

row("duration (us)", lambda d: g(d, "gpu__time_duration.sum") / 1e3)
row("registers/thread", lambda d: g(d, "launch__registers_per_thread"), "{:.0f}")
row("local ld sectors (spill)", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"), "{:.0f}")
row("local st sectors (spill)", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"), "{:.0f}")
row("achieved occupancy %", lambda d: g(d, "sm__warps_active.avg.pct_of_peak_sustained_active"), "{:.1f}")
row("shared mem/block (B)", lambda d: g(d, "launch__shared_mem_per_block_static") or 0, "{:.0f}")

print()
row("global LD requests", lambda d: g(d, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"), "{:.0f}")
row("global LD sectors", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"), "{:.0f}")
row("** LD sectors/request", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") / g(d, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))
row("global ST requests", lambda d: g(d, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"), "{:.0f}")
row("global ST sectors", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"), "{:.0f}")
row("** ST sectors/request", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") / g(d, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"))
row("global RED sectors", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum"), "{:.0f}")
row("global ATOM sectors", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum"), "{:.0f}")
row("ST sector lookup-hit", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum"), "{:.0f}")
row("ST sector lookup-miss", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum"), "{:.0f}")
row("LD sector lookup-hit", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"), "{:.0f}")
row("LD sector lookup-miss", lambda d: g(d, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"), "{:.0f}")

print()
row("DRAM read sectors", lambda d: g(d, "dram__sectors_read.sum"), "{:.0f}")
row("DRAM write sectors", lambda d: g(d, "dram__sectors_write.sum"), "{:.0f}")
row("DRAM throughput %peak", lambda d: g(d, "dram__throughput.avg.pct_of_peak_sustained_elapsed"), "{:.1f}")
row("DRAM read GB/s", lambda d: g(d, "dram__bytes_read.sum.per_second") / 1e9, "{:.1f}")
row("DRAM write GB/s", lambda d: g(d, "dram__bytes_write.sum.per_second") / 1e9, "{:.1f}")
row("L2 (lts) throughput %peak", lambda d: g(d, "lts__throughput.avg.pct_of_peak_sustained_elapsed"), "{:.1f}")
row("L1/TEX throughput %peak", lambda d: g(d, "l1tex__throughput.avg.pct_of_peak_sustained_active"), "{:.1f}")
row("LSU wavefronts %peak", lambda d: g(d, "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed"), "{:.1f}")
row("SM busy %", lambda d: g(d, "sm__throughput.avg.pct_of_peak_sustained_elapsed"), "{:.1f}")
row("compute (SM) pipe %peak", lambda d: g(d, "sm__inst_executed.avg.pct_of_peak_sustained_elapsed"), "{:.1f}")

print()
row("shared LD bank conflicts", lambda d: g(d, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"), "{:.0f}")
row("shared ST bank conflicts", lambda d: g(d, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"), "{:.0f}")
row("shared LD wavefronts", lambda d: g(d, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"), "{:.0f}")
row("shared ST wavefronts", lambda d: g(d, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"), "{:.0f}")
row("shared LD instructions", lambda d: g(d, "smsp__inst_executed_op_shared_ld.sum"), "{:.0f}")
row("shared ST instructions", lambda d: g(d, "smsp__inst_executed_op_shared_st.sum"), "{:.0f}")
