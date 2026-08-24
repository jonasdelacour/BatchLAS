#!/usr/bin/env python3
"""Every cell list this directory measures, generated in ONE place so each
sweep's batch schedule is a written-down decision and not a side effect
(experiments/wp6_lu/bench/gen_cells.py's rule, and its reason: WP5 published an
"order crossover" that was the batch axis wearing the order axis's clothes).

SCHED is the saturating batch schedule -- the top of each rung of wp6_lu's
SAT_LADDER, and identical to the schedule wp6_lu/bench/getrs_cells.txt used at
every order it shares, so the BEFORE column here and the BEFORE column there are
the same cell and not merely the same label. n = 32 and n = 128 are new (wp6_lu's
getrs table skipped them) and take their rung's top too, capped at 8192 and 4096
so that the widest nrhs still fits 24 GB for cdouble.

CAVEAT CARRIED FORWARD, not silently corrected: wp6_lu/bench/README.md §2
measured that cuBLAS does NOT saturate at n >= 1024 on any ladder this box can
hold. The schedule is therefore saturating for NATIVE at every rung and for the
vendor only below n = 1024; run_flat.sh re-reads the deciding cells across the
whole ladder for exactly that reason.
"""
import sys

SCHED = {32: 8192, 64: 8192, 128: 4096, 256: 2048, 512: 512, 1024: 128, 2048: 32}
ORDERS = [32, 64, 128, 256, 512, 1024, 2048]
TYPES = ["float", "double", "cfloat", "cdouble"]
NRHS = [1, 2, 4, 16, 64, 128]

# The BATCH ladders for the flatness check. Straight out of wp6_lu's SAT_LADDER
# so a rung here is a rung there. Three orders, one per regime.
FLAT_LADDER = {64: [1024, 2048, 4096, 8192, 16384],
               512: [64, 128, 256, 512, 1024],
               2048: [4, 8, 16, 32, 64]}

# The widths the headline grid skipped. nrhs = 4 and 8 are where the fused tier
# stops beating cuBLAS, and 8 is the LAST width it is instantiated for, so the
# routing window's whole boundary lives in these two columns. All seven orders,
# because the grid measured 8 at three orders only and a window written from
# three points is the one-cell over-fit this campaign keeps paying for.
W8_NRHS = [4, 8]


def main(what):
    out = []
    if what == "grid":
        for t in TYPES:
            for n in ORDERS:
                for r in NRHS:
                    out.append("getrs:%s:%d:%d:%d" % (t, n, r, SCHED[n]))
    elif what == "w8":
        for t in TYPES:
            for n in ORDERS:
                for r in W8_NRHS:
                    out.append("getrs:%s:%d:%d:%d" % (t, n, r, SCHED[n]))
    elif what == "flat":
        for t in TYPES:
            for n, ladder in sorted(FLAT_LADDER.items()):
                for r in (1, 4, 8):
                    for b in ladder:
                        out.append("getrs:%s:%d:%d:%d" % (t, n, r, b))
    elif what == "nrhs":
        # The nrhs axis at FIXED order and FIXED batch, dense through the
        # crossover, including the widths the headline grid skips (8, 32).
        for t in TYPES:
            for n, b in ((64, 8192), (512, 512), (2048, 32)):
                for r in (1, 2, 4, 8, 16, 32, 64, 128):
                    out.append("getrs:%s:%d:%d:%d" % (t, n, r, b))
    elif what == "flat2":
        # THE SECOND FLATNESS PASS, and it exists because the first one measured
        # nrhs in {1, 4, 8} and the window that came out of it turns on nrhs = 2.
        # nrhs = 2 is NOT bracketed by 1 and 4: at nrhs = 1 every ladder is a flat
        # win and at nrhs = 4 half of them cross, so 2 has to be measured and not
        # interpolated.
        #
        # It also adds two ORDERS the first pass did not visit (128 and 1024) on
        # the two clauses the window would actually carry -- nrhs = 1 for every
        # type, and nrhs = 4 for float -- so neither clause rests on three orders.
        for t in TYPES:
            for n, ladder in sorted(FLAT_LADDER.items()):
                for b in ladder:
                    out.append("getrs:%s:%d:2:%d" % (t, n, b))
        extra = {128: [256, 512, 1024, 2048, 4096, 8192], 1024: [16, 32, 64, 128, 256]}
        for t in TYPES:
            for n, ladder in sorted(extra.items()):
                for b in ladder:
                    out.append("getrs:%s:%d:1:%d" % (t, n, b))
        for n, ladder in sorted(extra.items()):
            for b in ladder:
                out.append("getrs:float:%d:4:%d" % (n, b))
    elif what == "flat3":
        # THE THIRD FLATNESS PASS. The candidate window's widest clause is
        # "nrhs <= 4 and n >= 128 and batch >= 16" for the NON-float types, and
        # after flat2 that clause rested on full batch ladders at n = 512 and
        # n = 2048 only -- every other order in it was a single batch. The
        # batch >= 16 term in particular came off the n = 2048 ladder alone. So:
        # the two interior orders, full ladders, the three types the clause is
        # about.
        extra = {128: [256, 512, 1024, 2048, 4096, 8192], 1024: [16, 32, 64, 128, 256]}
        for t in ("double", "cfloat", "cdouble"):
            for n, ladder in sorted(extra.items()):
                for b in ladder:
                    out.append("getrs:%s:%d:4:%d" % (t, n, b))
    elif what == "flat4":
        # THE FOURTH FLATNESS PASS, and it exists because the review of the C3
        # proposal found that the window's own stated MINIMUM (1.123x, cdouble
        # n=32 nrhs=2) came from a cell measured at ONE batch. Three gaps, all
        # inside the proposed window and all previously single-point:
        #   * nrhs = 2 at n = 128 and n = 1024 -- flat2 laddered nrhs = 2 at
        #     64/512/2048 and nrhs = 1 at 128/1024, so the two interior orders of
        #     clause A were never laddered at the width the clause turns on.
        #   * n = 32 and n = 256 at nrhs = 1 and 2 -- NO order-32 or order-256
        #     ladder exists anywhere in this directory, at any width, so the whole
        #     small-n end of clause A rested on the saturating point alone.
        #   * float nrhs = 4 at n = 32 and n = 256, clause B's two unladdered
        #     orders.
        # A window written from one batch is the defect this campaign keeps
        # paying for; a window whose MINIMUM is written from one batch is that
        # defect at the exact cell that decides whether the window ships.
        extra = {32:   [1024, 2048, 4096, 8192, 16384],
                 128:  [256, 512, 1024, 2048, 4096, 8192],
                 256:  [256, 512, 1024, 2048, 4096],
                 1024: [16, 32, 64, 128, 256]}
        for t in TYPES:
            for n in (32, 256):
                for r in (1, 2):
                    for b in extra[n]:
                        out.append("getrs:%s:%d:%d:%d" % (t, n, r, b))
            for n in (128, 1024):
                for b in extra[n]:
                    out.append("getrs:%s:%d:2:%d" % (t, n, b))
        for n in (32, 256):
            for b in extra[n]:
                out.append("getrs:float:%d:4:%d" % (n, b))
    elif what == "lu":
        # The getrf/getri regression check: wp6_lu/bench's own order32 and
        # order1024 cells, verbatim, so the AFTER can be diffed against the
        # recorded BEFORE cell by cell.
        for op in ("getrf", "getri"):
            for t in TYPES:
                for n in ORDERS:
                    out.append("%s:%s:%d:1:32" % (op, t, n))
        for op in ("getrf", "getri"):
            for t in TYPES:
                for n in [32, 64, 128, 256, 512]:
                    out.append("%s:%s:%d:1:1024" % (op, t, n))
    else:
        raise SystemExit("unknown list " + what)
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
