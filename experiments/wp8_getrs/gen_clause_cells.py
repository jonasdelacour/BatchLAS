#!/usr/bin/env python3
# THE CLAUSE CELLS: the composition WITH THE GATHER against the vendor, on the
# saturated part of the ladder only.
#
# WHICH BATCHES ARE SATURATED is read off experiments/wp8_getrs/ladder.py's
# us/item columns for the WALK ladder (both arms' cost per item stops improving,
# or turns up). Those rungs do not move when the permutation gets cheaper -- the
# vendor arm is unchanged and the native arm only gets faster -- so the same
# rungs are the admissible ones after the change. The rung BELOW the first
# saturated one is kept at every (type, n) so the transition is visible in the
# CSV rather than asserted.
#
# float gets THREE nrhs and every rung, because it is the candidate clause and
# GATE-C wants a batch ladder on every axis the clause names. The other three
# types get the two wide widths, which is all a REFUTATION needs: one loss inside
# a candidate set kills it, and the losses are not marginal.
LADDER = {                       # n -> batches, lowest = the rung below saturation
    64:   [2048, 4096, 8192],
    128:  [1024, 2048, 4096],
    256:  [1024, 2048, 4096],
    512:  [512, 1024, 2048],
    1024: [128, 256, 512],
}
SZ = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
BUDGET = 12 << 30

if __name__ == "__main__":
    import sys
    seen = set()
    skipped = []
    for t in ["float", "double", "cfloat", "cdouble"]:
        nrhs_list = [32, 64, 128] if t == "float" else [64, 128]
        for n, bs in LADDER.items():
            for r in nrhs_list:
                for b in bs:
                    need = SZ[t] * 2 * (n * n + n * r) * b
                    c = f"getrs:{t}:{n}:{r}:{b}"
                    if need > BUDGET:
                        skipped.append((c, need))
                        continue
                    if c in seen:
                        continue
                    seen.add(c)
                    print(c)
    for c, need in skipped:
        sys.stderr.write(f"SKIP {c}  needs {need / (1 << 30):.2f} GiB\n")
    sys.stderr.write(f"{len(seen)} cells, {len(skipped)} skipped\n")
