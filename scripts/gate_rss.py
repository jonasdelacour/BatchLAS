#!/usr/bin/env python3
"""Per-ctest-entry peak memory for scripts/gate.sh (Linux /proc only).

  gate_rss.py --pid PID --log CTEST_LOG --out TSV [--interval S]
  gate_rss.py --summary TSV [--exact FILE]

Sampling mode polls until PID (ctest, or a wrapper around it) exits. Each sample sums VmRSS
over PID's descendants (ctest itself excluded), takes their largest VmHWM, and charges both to
the entries LOG shows as started and not finished ("a+b" when several overlap). It also tracks
the box's lowest MemAvailable. A process that peaks and exits between samples is missed; the
gate cross-checks the largest HWM against /usr/bin/time's exact maximum (--exact).
"""
import argparse
import os
import re
import sys
import time

START = re.compile(r"^\s*Start\s+(\d+): (\S+)")
DONE = re.compile(r"^\s*\d+/\d+ Test\s+#(\d+): ")


def proc_children():
    kids = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        rest = stat[stat.rfind(")") + 2:].split()
        if rest[0] != "Z":
            kids.setdefault(int(rest[1]), []).append(int(name))
    return kids


def status_kb(pid, keys):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in keys:
                    out[k] = int(v.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


def mem_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


class LogTail:
    def __init__(self, path):
        self.path, self.pos, self.part, self.active = path, 0, "", {}

    def poll(self):
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return self.active
        lines = (self.part + data).split("\n")
        self.part = lines.pop()
        for line in lines:
            m = START.match(line)
            if m:
                self.active[m.group(1)] = m.group(2)
                continue
            m = DONE.match(line)
            if m:
                self.active.pop(m.group(1), None)
        return self.active


def sample(args):
    tail = LogTail(args.log)
    peaks = {}
    tree_peak = 0
    min_avail = None
    while alive(args.pid):
        kids = proc_children()
        stack, rss, hwm = list(kids.get(args.pid, [])), 0, 0
        while stack:
            pid = stack.pop()
            stack.extend(kids.get(pid, []))
            try:
                with open(f"/proc/{pid}/comm") as f:
                    if f.read().strip() == "ctest":
                        continue
            except OSError:
                continue
            s = status_kb(pid, ("VmRSS", "VmHWM"))
            rss += s.get("VmRSS", 0)
            hwm = max(hwm, s.get("VmHWM", 0))
        avail = mem_available_kb()
        min_avail = avail if min_avail is None else min(min_avail, avail)
        tree_peak = max(tree_peak, rss)
        active = tail.poll()
        if active and rss:
            key = "+".join(sorted(active.values()))
            p = peaks.setdefault(key, [0, 0, 0])
            p[0], p[1], p[2] = max(p[0], rss), max(p[1], hwm), p[2] + 1
        time.sleep(args.interval)
    with open(args.out, "w") as f:
        f.write("# entry\tpeak_sum_rss_mb\tpeak_proc_hwm_mb\tsamples\n")
        for key, (r, h, n) in sorted(peaks.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{key}\t{r // 1024}\t{h // 1024}\t{n}\n")
        f.write(f"# tree_peak_sum_rss_mb\t{tree_peak // 1024}\n")
        f.write(f"# box_min_mem_available_mb\t{(min_avail or 0) // 1024}\n")
        f.write(f"# interval_s\t{args.interval}\n")


def summary(args):
    rows, meta = [], {}
    with open(args.summary) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if line.startswith("# ") and len(parts) == 2:
                meta[parts[0][2:]] = parts[1]
            elif not line.startswith("#") and len(parts) == 4:
                rows.append((parts[0], int(parts[1]), int(parts[2])))
    top = ", ".join(f"{n} {r}" for n, r, _ in rows[:3]) or "none"
    hwm = max((h for _, _, h in rows), default=0)
    exact = ""
    if args.exact and os.path.isfile(args.exact):
        with open(args.exact) as f:
            m = re.search(r"maxrss_kb (\d+)", f.read())
        if m:
            exact = f", exact largest process {int(m.group(1)) // 1024} MB"
    print(f"peak RSS MB: top {top}; tree peak {meta.get('tree_peak_sum_rss_mb', '?')}, "
          f"largest sampled process {hwm}{exact}; box min MemAvailable "
          f"{meta.get('box_min_mem_available_mb', '?')} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pid", type=int)
    ap.add_argument("--log")
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--summary")
    ap.add_argument("--exact")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    elif args.pid and args.log and args.out:
        if not os.path.isdir("/proc/self"):
            sys.exit("gate_rss: no /proc")
        sample(args)
    else:
        ap.error("--pid, --log and --out, or --summary")


if __name__ == "__main__":
    main()
