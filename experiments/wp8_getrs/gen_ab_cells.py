#!/usr/bin/env python3
# GATE-B cells for the walk-vs-gather A/B.
#
# ONE SATURATING BATCH PER n, read off the ladder in this directory rather than
# guessed: the ladder's us/item columns stop improving at 8192 (n=64), 4096
# (n=128), 2048 (n=256), 1024 (n=512) and 256 (n=1024). Below those the arms are
# measuring launch amortisation, and an A/B of two PERMUTATION kernels taken
# there would be reporting the composition's overhead rather than the
# permutation's cost.
#
# THE nrhs AXIS IS WALKED DOWN TO 1, not just across the wide end. The whole
# question a boundary answers is where the gather STOPS paying, and a grid that
# only samples the wide end cannot see that -- campaign trap 8. nrhs = 1 is also
# the ONLY width the library itself issues (linalg::solve), so it is the cell the
# default must not regress.
BATCH_FOR_N = {64: 8192, 128: 4096, 256: 2048, 512: 1024, 1024: 256}
TYPES = ["float", "double", "cfloat", "cdouble"]
NRHS = [1, 4, 16, 32, 64, 128]
NS = [64, 128, 256, 512, 1024]

if __name__ == "__main__":
    for t in TYPES:
        for n in NS:
            for r in NRHS:
                print(f"getrs:{t}:{n}:{r}:{BATCH_FOR_N[n]}")
