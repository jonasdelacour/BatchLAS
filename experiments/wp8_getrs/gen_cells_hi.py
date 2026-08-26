#!/usr/bin/env python3
# THE HIGH-BATCH HALF OF THE LADDER, and why it is not optional.
#
# The brief's grid stops at batch 512. The RECORDED wide-nrhs cells -- the 9 and
# 4 losses this pass has to adjudicate -- were all taken at ONE saturating batch
# per n, and those batches are 8192 (n=32,64), 4096 (n=128), 2048 (n=256), 512
# (n=512), 128 (n=1024). So at n = 64, 128 and 256 the brief's grid lives
# ENTIRELY BELOW the batch the loss was measured at, and a ladder that stops
# short of the cell it is meant to explain cannot explain it. Campaign trap 8 in
# its own terms: a grid that cannot reach a regime is not evidence about it.
#
# This file adds batch {1024, 2048, 4096, 8192} at n {64, 128, 256, 512} under
# the same 12 GiB clamp. n = 1024 is not extended: cdouble n=1024 batch=1024 is
# 34 GiB and even float is 8.6 GiB, and n=1024's recorded saturating batch (128)
# is already inside the brief's grid.
from gen_cells import bytes_of, BUDGET

TYPES = ["float", "double", "cfloat", "cdouble"]
NS = [64, 128, 256, 512]
NRHS = [32, 64, 128]   # nrhs=16 is dropped: the base grid measures it a LOSS at every
                      # (type, n, batch) cell it reaches, so a high-batch ladder
                      # under it cannot change any clause. Recorded, not assumed.
BATCH = [1024, 2048, 4096, 8192]

if __name__ == "__main__":
    import sys
    cells, skipped = [], []
    for t in TYPES:
        for n in NS:
            for r in NRHS:
                for b in BATCH:
                    need = bytes_of(t, n, r, b)
                    (cells if need <= BUDGET else skipped).append(
                        (f"getrs:{t}:{n}:{r}:{b}", need))
    for c, _ in cells:
        print(c)
    for c, need in skipped:
        sys.stderr.write(f"SKIP {c}  needs {need / (1 << 30):.2f} GiB > 12.00 GiB\n")
    sys.stderr.write(f"{len(cells)} cells, {len(skipped)} skipped\n")
