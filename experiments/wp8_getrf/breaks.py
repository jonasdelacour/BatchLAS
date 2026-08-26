#!/usr/bin/env python3
"""GATE-D breaks for the deferred left-hand interchange.

Each break is an ACTUAL EDIT, applied to the tree, rebuilt, run, shown RED, and
reverted. The anchor rule this directory inherits from experiments/wp6_lu:
EVERY ANCHOR MUST MATCH EXACTLY ONCE IN BOTH DIRECTIONS. An 8-space anchor that
is a substring of a 12-space line once left two permutation walks inverted in
the tree, so the count is asserted, not assumed.

usage:  breaks.py list
        breaks.py apply <name>
        breaks.py revert <name>
"""
import sys, os

W = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan"
LASWP = os.path.join(W, "src/extensions/lu_laswp.hh")
DRIVER = os.path.join(W, "src/extensions/getrf_blocked.cc")

BREAKS = {
    # B1 -- the deferred pass never runs. This is D1's recorded break B9 in its
    # new spelling: the left-hand interchange is simply absent, so L's rows do
    # not travel with the later pivots and P A != L U.
    "no_deferred": (DRIVER,
                    "    if (mode != LeftLaswp::InLoop) {",
                    "    if (false && mode != LeftLaswp::InLoop) {"),

    # B2 -- the deferred pass walks the transposition list BACKWARDS. Every
    # transposition is its own inverse, so only the ORDER changes; the same list
    # applied in decreasing k composes P^T, not P. The getrs family has the
    # identical recorded direction break, so the class is known live.
    "reverse_k": (LASWP,
                  "                    for (int i = 0; i < R; ++i) {\n"
                  "                        const int p = ips[i];",
                  "                    for (int i = R - 1; i >= 0; --i) {\n"
                  "                        const int p = ips[i];"),

    # B3 -- the FALLBACK branch (the `defer_walk` spelling, and the only branch
    # reachable when the staging tile does not fit) applies ONE panel's list
    # instead of the whole suffix. Aimed at the new test rather than at the old
    # one: only LeftInterchangeSpellingsAgreeBitForBit can see it, because the
    # gather arm stays correct and the residual bound alone would not separate
    # the two arms.
    "fallback_one_panel": (DRIVER,
                           "piv_ptr, /*piv_stride=*/n, /*k0=*/k0, /*k1=*/n, /*forward=*/true);",
                           "piv_ptr, /*piv_stride=*/n, /*k0=*/k0, "
                           "/*k1=*/std::min(k0 + ib, n), /*forward=*/true);"),
}


def edit(path, old, new):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    if n != 1:
        raise SystemExit("anchor matched %d times (need exactly 1) in %s:\n%r" % (n, path, old))
    if src.count(new) != 0:
        raise SystemExit("replacement already present in %s" % path)
    with open(path, "w") as f:
        f.write(src.replace(old, new))


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for k in BREAKS:
            print(k)
        return
    act, name = sys.argv[1], sys.argv[2]
    path, old, new = BREAKS[name]
    if act == "apply":
        edit(path, old, new)
        print("applied %s to %s" % (name, path))
    elif act == "revert":
        edit(path, new, old)
        print("reverted %s in %s" % (name, path))
    else:
        raise SystemExit("unknown action")


if __name__ == "__main__":
    main()
