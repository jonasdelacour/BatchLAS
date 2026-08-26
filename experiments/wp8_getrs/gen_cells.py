#!/usr/bin/env python3
# WP8-GETRS (G3) THE MISSING LADDER.
#
# D2's finding: no batch ladder exists ANYWHERE at nrhs >= 16 in this tree. Every
# wide-nrhs getrs cell that has ever been measured here comes from grid_cells.txt
# or nrhs_cells.txt, both of which carry ONE saturating batch per n. That is why
# the "window-shaped opportunity" at nrhs 64/128 is not a window.
#
# THE GRID IS THE BRIEF'S: nrhs {16,32,64,128} x n {64,128,256,512,1024} x
# batch {32,128,256,512} x 4 types = 320 cells per arm.
#
# MEMORY CLAMP. lubench6 allocates A0 AND A (2 * n*n*batch) plus B0 AND X
# (2 * n*nrhs*batch), all in unified memory, so a cell's device footprint is
# sizeof(T) * 2 * (n*n + n*nrhs) * batch. The card is 24 GB with ~1.3 GB already
# resident from a foreign graphics context, and the getrs workspace and the
# blocked trsm's own scratch sit on top. Budget 12 GB; every skipped cell is
# PRINTED, never silently dropped -- a grid that cannot reach a regime is not
# evidence about it, and a grid that silently shrinks is worse.
BUDGET = 12 << 30
SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
TYPES = ["float", "double", "cfloat", "cdouble"]
NS = [64, 128, 256, 512, 1024]
NRHS = [16, 32, 64, 128]
BATCH = [32, 128, 256, 512]


def bytes_of(t, n, nrhs, b):
    return SZ[t] * 2 * (n * n + n * nrhs) * b


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
