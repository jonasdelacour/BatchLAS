#!/usr/bin/env python3
# Assemble getrsab.cpp = lubench6.cpp's SCAFFOLDING + getrsab_body.inc.
#
# The scaffolding is taken VERBATIM, by script, up to (but not including) the
# `run()` driver. That is deliberate: WP4's measurement ended up 2x off its own
# shipped numbers because a harness was re-derived rather than copied, and the
# pieces this A/B leans on -- the diagonally dominant then ROW-PERMUTED matrix
# (without which ipiv is the identity and the permutation is untested by
# construction), the NaN-propagating nanmax, Tol<T>, solve_probe's transposed
# reference, rstr -- are all in that region.
#
# The one addition is the getrs_native.hh include, for the spelling-debug query.
import pathlib
import sys

W = pathlib.Path("/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan")
SRC = W / "experiments/wp6_lu/bench/lubench6.cpp"
BODY = W / "experiments/wp8_getrs/getrsab_body.inc"
OUT = W / "experiments/wp8_getrs/getrsab.cpp"

text = SRC.read_text()
marker = "// --------------------------------------------------------------- driver"
idx = text.index(marker)
head = text[:idx]

# The spelling-debug query lives in getrs_native.hh; lubench6 includes only
# getrf_native.hh.
anchor = '#include "src/extensions/getrf_native.hh"'
assert head.count(anchor) == 1, "include anchor must match exactly once"
head = head.replace(anchor, anchor + '\n#include "src/extensions/getrs_native.hh"')

OUT.write_text(head + BODY.read_text())
sys.stderr.write(f"wrote {OUT} ({len(head.splitlines())} scaffolding lines copied)\n")
