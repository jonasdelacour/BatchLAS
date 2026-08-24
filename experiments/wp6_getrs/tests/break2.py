#!/usr/bin/env python3
"""Breaks that live OUTSIDE the kernel file: the facade arm and the tier
tie-break. Same discipline -- anchors are asserted, and `restore` puts both
files back."""
import os, shutil, sys

ROOT = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/"
TMP = "/home/jonaslacour/.claude/jobs/20812aa0/tmp/"
FILES = {
    "facade": ROOT + "src/dispatch/entry_points/factorization.cc",
    "table": ROOT + "include/batchlas/blas/dispatch/route_getrs.hh",
}
BAKS = {k: TMP + os.path.basename(v) + ".orig2" for k, v in FILES.items()}


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


ARM = """        if (route.algo == dispatch::Algorithm::CTA) {
            // THE FUSED NARROW-RHS TIER. It injects NOTHING -- no trsm, no gemm,
            // no laswp -- because it calls no other BLAS operation at all: the
            // permutation and both substitutions are one kernel. That is why this
            // arm has no seam where the Blocked one below has one.
            return sycl_getrs::getrs_fused_dispatch<T>(
                ctx, A, B, transA, pivots, work_space);
        }
"""


def b_facade_arm():
    # THE FACADE'S CTA ARM DELETED. The route still resolves to {Native, CTA};
    # the call falls through to the COMPOSED tier, which returns a numerically
    # fine answer that is not bit-identical to the fused kernel's.
    p = FILES["facade"]
    t = open(p).read()
    t = sub(t, ARM, "")
    t = sub(t, "        if (route.algo == dispatch::Algorithm::Blocked) {\n"
               "            // BOTH TRIANGULAR SOLVES GO THROUGH THE ROUTER.",
               "        if (route.algo == dispatch::Algorithm::Blocked ||\n"
               "            route.algo == dispatch::Algorithm::CTA) {\n"
               "            // BOTH TRIANGULAR SOLVES GO THROUGH THE ROUTER.")
    open(p, "w").write(t)


def b_tier_pref():
    # native_tier_preferred INVERTED: the vendor-free walk falls back to the
    # composed tier for every shape, which is the routing state this tier landed
    # to change.
    p = FILES["table"]
    t = open(p).read()
    t = sub(t, "            case Algorithm::CTA:     return true;\n"
               "            case Algorithm::Blocked: return false;",
               "            case Algorithm::CTA:     return false;\n"
               "            case Algorithm::Blocked: return true;")
    open(p, "w").write(t)


BREAKS = {"facade_arm": b_facade_arm, "tier_pref": b_tier_pref}

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
