#!/usr/bin/env python3
"""The two CEILING breaks: supports()' capacity gates, and the direct entry
point's re-check of them. Both target FusedGetrsHandsBackAtBothCeilings, which
every numerical break correctly leaves green."""
import os, shutil, sys

ROOT = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/"
TMP = "/home/jonaslacour/.claude/jobs/20812aa0/tmp/"
FILES = {
    "table": ROOT + "include/batchlas/blas/dispatch/route_getrs.hh",
    "kernel": ROOT + "src/extensions/getrs_fused.cc",
}
BAKS = {k: TMP + os.path.basename(v) + ".orig3" for k, v in FILES.items()}


def sub(text, old, new, count=1):
    n = text.count(old)
    assert n == count, "anchor matched %d, expected %d" % (n, count)
    return text.replace(old, new)


def backup():
    for k, v in FILES.items():
        if not os.path.exists(BAKS[k]):
            shutil.copyfile(v, BAKS[k])


def restore():
    for k, v in FILES.items():
        if os.path.exists(BAKS[k]):
            shutil.copyfile(BAKS[k], v)


def b_supports_gates():
    # supports() STOPS CHECKING THE FUSED TIER'S TWO CAPACITY CEILINGS, so the
    # table advertises a route the launcher refuses.
    p = FILES["table"]
    t = open(p).read()
    t = sub(t, "                if (s.order() * s.nrhs() > s.fused_max_elems) return false;\n"
               "                if (s.nrhs() > s.fused_max_nrhs) return false;\n", "")
    open(p, "w").write(t)


def b_dispatch_gates():
    # THE DIRECT ENTRY POINT STOPS RE-APPLYING THEM. It is reachable WITHOUT the
    # table, so a pinned caller would walk into an unlaunchable configuration.
    p = FILES["kernel"]
    t = open(p).read()
    i = t.index("    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);")
    j = t.index("    // PACKED 1-BASED int32", i)
    keep = ("    const std::size_t local_mem = dev.get_property("
            "DeviceProperty::LOCAL_MEM_SIZE);\n"
            "    static_cast<void>(local_mem);\n\n")
    t = t[:i] + keep + t[j:]
    open(p, "w").write(t)


BREAKS = {"supports_gates": b_supports_gates, "dispatch_gates": b_dispatch_gates}

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "restore":
        restore()
        print("restored")
        sys.exit(0)
    backup()
    restore()
    BREAKS[what]()
    print("patched:", what)
