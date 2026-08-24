#!/usr/bin/env python3
"""Corrupt one guarded property of src/extensions/getrs_fused.cc, in place.

Usage:  break.py <name>     -> patch
        break.py restore    -> put the pristine copy back

The pristine copy is taken once, on the first patch, and kept beside the
scratch directory. Every patch asserts that its anchor matched exactly the
expected number of times, so a break that silently no-ops is impossible.
"""
import os, shutil, sys, re

ROOT = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/"
SRC = ROOT + "src/extensions/getrs_fused.cc"
BAK = "/home/jonaslacour/.claude/jobs/20812aa0/tmp/getrs_fused.cc.orig"


def sub(text, old, new, count):
    n = text.count(old)
    assert n == count, "anchor matched %d times, expected %d:\n%s" % (n, count, old[:200])
    return text.replace(old, new)


def b_piv_base(t):
    # THE PIVOT BASE. The wire format is 1-BASED; read it 0-based.
    return sub(t, "const int p = pv[k] - 1;", "const int p = pv[k];", 2)


def b_unit_u(t):
    # A UNIT-DIAGONAL ASSUMPTION ON U, in both kernels' non-unit solve.
    t = sub(t, """                                if (lane == kk)
                                    v = dev_div(v, blk[static_cast<std::size_t>(kk) +
                                                       static_cast<std::size_t>(kk) *
                                                       static_cast<std::size_t>(bld)]);
""", "", 1)
    t = sub(t, """                                if (lane == s)
                                    v = dev_div(v, blk[static_cast<std::size_t>(s) +
                                                       static_cast<std::size_t>(s) *
                                                       static_cast<std::size_t>(bld)]);
""", "", 1)
    return t


def b_last_row(t):
    # AN OFF-BY-ONE AT THE LAST ROW of the NoTrans forward trailing update.
    return sub(t, "for (int i = j + jb + tid; i < n; i += wg) {",
                  "for (int i = j + jb + tid; i < n - 1; i += wg) {", 1)


def b_conj(t):
    # ConjTrans STOPS CONJUGATING.
    return sub(t, "return conj ? dev_conj(a) : a;", "return a;", 1)


def b_trans_perm_forward(t):
    # THE TRANSPOSED OUTPUT PERMUTATION WALKED FORWARDS instead of backwards.
    return sub(t, "for (int k = n - 1; k >= 0; --k) {\n                        const int p = pv[k] - 1;",
                  "for (int k = 0; k < n; ++k) {\n                        const int p = pv[k] - 1;", 1)


TRANS_PERM = """                if (tid < nrhs) {
                    D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);
                    for (int k = n - 1; k >= 0; --k) {
                        const int p = pv[k] - 1;
                        if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
"""


def b_perm_wrong_side(t):
    # THE PERMUTATION APPLIED ON THE WRONG SIDE FOR Trans/ConjTrans: moved from
    # the OUTPUT (after both solves, walked backwards) to the INPUT (before
    # them), which is what the NoTrans arm correctly does and the transposed arm
    # correctly must not.
    t = sub(t, TRANS_PERM, "", 1)
    anchor = """                // ---- op(U) z = b : op(U) is LOWER, non-unit, forward --------
"""
    return sub(t, anchor, TRANS_PERM + anchor, 1)


def _between(t, start, end):
    i = t.index(start)
    j = t.index(end, i)
    return t[i:j]


def b_swap_solves(t):
    # THE TWO SUBSTITUTIONS SWAPPED in the NoTrans kernel: U before L.
    s1 = "                // ---- L y = F b, unit lower, forward -------------------------\n"
    s2 = "                // ---- U x = y, non-unit upper, backward ----------------------\n"
    s3 = "                for (int e = tid; e < n * nrhs; e += wg) {\n                    const int i = e % n, c = e / n;\n                    Bb[static_cast<std::size_t>(i) +"
    first = _between(t, s1, s2)
    second = _between(t, s2, s3)
    assert t.count(first) == 1 and t.count(second) == 1
    return t.replace(first + second, second + first, 1)


def b_reg_cap(t):
    # THE REGISTER CAP REMOVED. registers x work-group must stay under 65,536 or
    # the launch ABORTS; float/Trans/NR=8 at n=1428 picks wg=1024 without it.
    return sub(t, """    const int regs = getrs_fused_regs_for(nrhs) + kGetrsFusedRegMargin;
    int cap = (65536 / regs) & ~31;          // down to a multiple of the sub-group
    if (cap < 32) cap = 32;
    if (wg > cap) wg = cap;
""", "", 1)


def b_cap_inversion(t):
    # THE CAPACITY-INVERSION REPAIR REVERTED: the floor division alone, which can
    # round the implied request back INTO the pad band.
    return sub(t, """    if (getrs_fused_slm(elems, kGetrsFusedNbMax, sizeof(D)) > slm_budget_bytes) {
        if (kGetrsHoleLo <= blk_bytes) return 0;
        return (kGetrsHoleLo - blk_bytes) / sizeof(D);
    }
    return elems;""", "    return elems;", 1)


def b_cap_band(t):
    # THE HOLE BAND REMOVED FROM THE CAPACITY QUERY: `admissible` becomes the raw
    # budget, so a request landing in the band is advertised and then padded past
    # the budget at launch.
    return sub(t, """        (slm_budget_bytes > kGetrsHoleHi) ? slm_budget_bytes
                                          : std::min(slm_budget_bytes, kGetrsHoleLo);""",
               "        slm_budget_bytes;", 1)


def b_rhs_ld(t):
    # THE RHS WRITE-BACK IGNORES ldb and uses n, so the pad and the inter-item
    # gap are overwritten.
    return sub(t, """                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)] = y[e];
                }""", """                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(n)] = y[e];
                }""", 2)


def b_hole_pad(t):
    # THE 48 KB PAD REMOVED FROM THE LAUNCHER (the implementer's B5).
    return sub(t, "    return (bytes > kGetrsHoleLo && bytes <= kGetrsHoleHi) ? kGetrsHolePadTo : bytes;",
               "    return bytes;", 1)


BREAKS = {
    "piv_base": b_piv_base,
    "unit_u": b_unit_u,
    "last_row": b_last_row,
    "conj": b_conj,
    "trans_perm_forward": b_trans_perm_forward,
    "perm_wrong_side": b_perm_wrong_side,
    "swap_solves": b_swap_solves,
    "reg_cap": b_reg_cap,
    "cap_inversion": b_cap_inversion,
    "cap_band": b_cap_band,
    "rhs_ld": b_rhs_ld,
    "hole_pad": b_hole_pad,
}

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "restore":
        shutil.copyfile(BAK, SRC)
        print("restored")
        sys.exit(0)
    if not os.path.exists(BAK):
        shutil.copyfile(SRC, BAK)
    shutil.copyfile(BAK, SRC)
    t = open(SRC).read()
    t = BREAKS[what](t)
    open(SRC, "w").write(t)
    print("patched:", what)
