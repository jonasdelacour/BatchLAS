#!/usr/bin/env python3
"""Every cell list this directory measures, generated in one place so the batch
schedule of each sweep is a written-down decision rather than a side effect.

WHY THE AXES ARE SEPARATED. WP5 published "order crossovers" from a sweep whose
batch schedule varied with n; the crossover it found was the batch axis wearing
the order axis's clothes. So:

  sat    -- BATCH axis, ORDER fixed. The full ladder per n. Establishes whether
            each arm saturates before any ratio is quoted.
  order  -- ORDER axis, BATCH fixed, twice (32 and 1024), so a crossover that
            moves between the two is visible as a batch effect.
  grid   -- the headline A/B at the SATURATING schedule, which necessarily varies
            batch with n and is therefore NOT a crossover table.
  nrhs   -- getrs's own axis, order and batch fixed.

MEMORY. n^2 * batch * sizeof(T) per array, and getri holds three (A0, A, C).
cdouble n=2048 is 67 MB/item, so batch 32 is the ceiling on a 24 GB 4090 for the
three-array op. Every list below is feasible at every type it names.
"""
import sys

SAT_LADDER = {
    32:   [512, 1024, 2048, 4096, 8192, 16384],
    64:   [512, 1024, 2048, 4096, 8192, 16384],
    128:  [256, 512, 1024, 2048, 4096, 8192],
    256:  [128, 256, 512, 1024, 2048, 4096],
    512:  [64, 128, 256, 512, 1024],
    1024: [16, 32, 64, 128, 256],
    2048: [4, 8, 16, 32, 64],
}

ORDERS = [32, 64, 128, 256, 512, 1024, 2048]
TYPES = ["float", "double", "cfloat", "cdouble"]


def emit(cells):
    for c in cells:
        print(c)


def sat(ops, types, orders=None):
    out = []
    for op in ops:
        for t in types:
            for n in sorted(SAT_LADDER):
                if orders and n not in orders:
                    continue
                for b in SAT_LADDER[n]:
                    out.append("%s:%s:%d:1:%d" % (op, t, n, b))
    return out


def order_fixed_batch(ops, types, batch, orders):
    return ["%s:%s:%d:1:%d" % (op, t, n, batch)
            for op in ops for t in types for n in orders]


def grid(sched, ops, types):
    return ["%s:%s:%d:1:%d" % (op, t, n, b)
            for op in ops for t in types for n, b in sched]


def nrhs_sweep(types, n, batch, nrhs_list):
    return ["getrs:%s:%d:%d:%d" % (t, n, r, batch) for t in types for r in nrhs_list]


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "sat":
        emit(sat(["getrf", "getri"], ["float", "cdouble"]))
    elif what == "sat2":
        # the two types the first saturation sweep did not cover. THE FULL LADDER,
        # not a subset: with all four types resolved against batch, every A/B ratio
        # in this directory can be read at a common batch AND at each arm's own
        # ceiling, and the difference between those two readings is the whole
        # honesty question at n >= 1024.
        emit(sat(["getrf", "getri"], ["double", "cfloat"]))
    elif what == "order32":
        emit(order_fixed_batch(["getrf", "getri"], TYPES, 32, ORDERS))
    elif what == "order1024":
        # n <= 512 only: cdouble n=1024 at batch 1024 is 17 GB for one array
        emit(order_fixed_batch(["getrf", "getri"], TYPES, 1024, [32, 64, 128, 256, 512]))
    elif what == "nrhs":
        emit(nrhs_sweep(TYPES, 512, 256, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    elif what == "nrhs2048":
        emit(nrhs_sweep(TYPES, 2048, 32, [1, 8, 64, 256, 2048]))
    else:
        raise SystemExit("unknown list: %s" % what)
