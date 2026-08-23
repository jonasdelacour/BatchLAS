#!/usr/bin/env python3
"""WHERE THE n >= 1024 RATIOS COME FROM, AND WHY THE RAW ONES ARE NOT SPEEDUPS.

At n >= 1024 cuBLAS's us/item is still falling at the top of the batch ladder, so
a ratio taken there compares against a routine that is not using the machine.
Three readings are printed side by side and they can differ by 4x:

  ratio_at_common_batch  the grid's reading -- the batch schedule the A/B used
  ratio_at_own_ceilings  each arm at ITS OWN best measured batch
  real_gflops columns    what each arm actually achieved, in REAL flops, against
                         this box's roofline

THE FIRST TWO ARE RATIOS; ONLY THE THIRD IS PHYSICS. A linear wall = c0 + c1*b
fit was tried first and DISCARDED: at cdouble n=2048 it returns a marginal cost
of 1.96 ms/item, which is 10.2 real TFLOP/s FP64 on a card whose FP64 peak is
1.29 -- impossible, so the fit was extrapolating across a regime change, not
measuring a marginal. The roofline column is what replaced it, because a number
that cannot exceed the hardware cannot fabricate a win.

FLOP CONVENTION. The harness's GFLOPs column counts 2/3 n^3 for getrf and
4/3 n^3 for getri REGARDLESS OF TYPE, which is the LAPACK convention. Real
hardware flops are 4x that for the two complex types (one complex multiply-add is
four real multiply-adds plus four adds; the 4x is the standard count).

ROOFLINE REFERENCES for one RTX 4090, stated as references and not as targets:
  FP32  82.6 TFLOP/s theoretical; ~47 TFLOP/s is the rate measured for GEMM on
        this box (the 80 TFLOP/s figure quoted elsewhere is TF32, not FP32).
  FP64  1.29 TFLOP/s theoretical (1/64 rate).
"""
import csv
import os

D = os.path.dirname(os.path.abspath(__file__))
FP32_PEAK, FP64_PEAK = 82.6e3, 1.29e3          # GFLOP/s
CX = {"float": 1.0, "double": 1.0, "cfloat": 4.0, "cdouble": 4.0}
PEAK = {"float": FP32_PEAK, "cfloat": FP32_PEAK,
        "double": FP64_PEAK, "cdouble": FP64_PEAK}
COEF = {"getrf": 2.0 / 3.0, "getri": 4.0 / 3.0}


def load(paths):
    rows = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.reader(f):
                if not r or r[0] == "op" or len(r) < 12:
                    continue
                if r[5] in ("TIMEOUT_OR_THROW", "THREW"):
                    continue
                try:
                    rows[(r[0], r[1], int(r[2]), int(r[4]))] = (float(r[5]), r[-1])
                except ValueError:
                    pass
    return rows


V = load([os.path.join(D, "sat_vendor.csv"), os.path.join(D, "sat2_vendor.csv"),
          os.path.join(D, "tail_vendor.csv")])
N = load([os.path.join(D, "sat_native.csv"), os.path.join(D, "sat2_native.csv"),
          os.path.join(D, "tail_native.csv")])

GRID = {32: 8192, 64: 8192, 128: 4096, 256: 2048, 512: 512, 1024: 128, 2048: 32}

print("op,type,n,grid_batch,ratio_at_common_batch,"
      "v_best_batch,v_best_us_item,n_best_batch,n_best_us_item,ratio_at_own_ceilings,"
      "v_real_GFLOPs_at_best,v_pct_roofline,n_real_GFLOPs_at_best,n_pct_roofline,"
      "vendor_wall_flat_to_batch")
keys = sorted({k[:3] for k in V} & {k[:3] for k in N})
for op, t, n in keys:
    if op not in COEF:
        continue
    vp = sorted((k[3], V[k][0]) for k in V if k[:3] == (op, t, n))
    np_ = sorted((k[3], N[k][0]) for k in N if k[:3] == (op, t, n))
    if not vp or not np_:
        continue
    gb = GRID.get(n)
    vg = dict(vp).get(gb)
    ng = dict(np_).get(gb)
    common = ("%.3f" % (vg / ng)) if (vg and ng) else "n/a"

    vb, vw = min(vp, key=lambda p: p[1] * 1000.0 / p[0])
    nb, nw = min(np_, key=lambda p: p[1] * 1000.0 / p[0])
    vu, nu = vw * 1000.0 / vb, nw * 1000.0 / nb

    def gf(us_item):
        return COEF[op] * n ** 3 * CX[t] / (us_item * 1e3)   # GFLOP/s, real flops

    vgf, ngf = gf(vu), gf(nu)

    # the largest batch at which the vendor's WALL is still within 5% of its
    # smallest-batch wall -- i.e. the batch up to which batch is FREE because the
    # call is a latency chain, not a throughput limit
    flat = vp[0][0]
    for b, w in vp:
        if w <= 1.05 * vp[0][1]:
            flat = b
    print("%s,%s,%d,%s,%s,%d,%.2f,%d,%.2f,%.3f,%.1f,%.1f%%,%.1f,%.1f%%,%d" %
          (op, t, n, gb, common, vb, vu, nb, nu, vu / nu,
           vgf, 100.0 * vgf / PEAK[t], ngf, 100.0 * ngf / PEAK[t], flat))
