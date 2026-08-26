#!/usr/bin/env python3
"""The DECISION cells for the LU clauses, and only those.

WHY THIS LIST EXISTS, AND IT IS A MEASUREMENT-HYGIENE FINDING RATHER THAN A
CONVENIENCE. The first LU sweep ran on device 1 while a gemv sweep ran on device
0. Both cards showed foreign == 0 -- correctly, because `nvidia-smi
--query-compute-apps` is PER DEVICE and neither process was on the other's card
-- and both sweeps looked clean. They were not: re-running getrf float n=256
batch=128 by hand gave 5.51 ms against the 1.006 ms WP8-I1 recorded on an idle
box, and the RATIO moved from 1.254 (two agreeing passes of I1's) to 1.764. Some
cells agreed to 0.5% and others were 2x out, so it is not a uniform scale factor
that a ratio divides away.

TWO RTX 4090s IN ONE BOX ARE NOT TWO INDEPENDENT MACHINES. They share a NUMA
node, a CPU affinity mask (0-19 for both, per `nvidia-smi topo -m`) and the
UVM driver, and lubench6 runs on managed memory. The per-row foreign() guard
cannot see this and neither can rel_sd -- the contaminated rows have rel_sd
0.0004-0.017. It belongs in the campaign's measurement-hygiene note beside the
display-GPU finding: SERIALISE, or the guard you added is guarding the wrong
thing.

So the clean passes run ALONE, and the cell list is cut to the cells that decide
a clause or bracket one of its boundaries.
"""
C = []
def add(op, t, n, q, bs):
    for b in bs: C.append(f"{op}:{t}:{n}:{q}:{b}")

# ---- getri: the two candidate windows, their boundaries, and the refutations
add('getri', 'float',  64, 1, [2048, 8192, 16384])          # below the boundary
add('getri', 'float', 128, 1, [1, 2, 32, 256, 1024, 4096, 16384])
add('getri', 'float', 256, 1, [128, 1024, 4096])
add('getri', 'float', 512, 1, [1, 32, 128, 1024, 2048])
add('getri', 'float', 1024, 1, [128, 512, 1024])
add('getri', 'float', 2048, 1, [128, 256])
add('getri', 'cfloat', 128, 1, [512, 2048])                 # below the boundary
add('getri', 'cfloat', 256, 1, [128, 512, 2048, 4096])
add('getri', 'cfloat', 512, 1, [1, 32, 128, 1024])
add('getri', 'cfloat', 1024, 1, [128, 512])
add('getri', 'cfloat', 2048, 1, [128])
add('getri', 'double', 256, 1, [512, 4096])                 # refutation
add('getri', 'double', 512, 1, [128, 1024])
add('getri', 'double', 1024, 1, [128, 512])
add('getri', 'double', 2048, 1, [128])
add('getri', 'cdouble', 512, 1, [128, 1024])                # refutation
add('getri', 'cdouble', 1024, 1, [128, 256])

# ---- getrs: the narrowed clause (nrhs >= 128, batch >= 128) and both edges
for t in ('float', 'double'):
    add('getrs', t,   64, 128, [32, 64, 128, 512, 4096])
    add('getrs', t,  128, 128, [32, 64, 128, 1024, 4096])
    add('getrs', t,  256, 128, [64, 128, 1024, 4096])
    add('getrs', t,  512, 128, [64, 128, 512, 2048])
    add('getrs', t, 1024, 128, [64, 128, 512, 1024])
# the cell that refuted nrhs >= 64, and its neighbours on the same ladder
add('getrs', 'float', 1024, 64, [128, 512, 1024])
add('getrs', 'float',  512, 64, [512, 2048])
add('getrs', 'float',  256, 64, [1024, 4096])
# nrhs 64 for double and the two complex types at one saturated rung each,
# so "cfloat and cdouble earn nothing" is a statement with cells behind it
for t in ('cfloat', 'cdouble'):
    add('getrs', t, 128, 128, [1024])
    add('getrs', t, 512, 128, [512])

# ---- getrf: I1's recommended clause, its boundaries, and the refutations
add('getrf', 'float',  128, 1, [128, 512])                  # native:cta, must lose
add('getrf', 'float',  256, 1, [128, 512, 1024])
add('getrf', 'float',  512, 1, [128, 512, 1024])
add('getrf', 'float', 1024, 1, [128, 512])
add('getrf', 'float', 2048, 1, [128])
add('getrf', 'cfloat', 128, 1, [128, 512])
add('getrf', 'cfloat', 256, 1, [128, 512, 1024])            # I1 put the edge ABOVE this
add('getrf', 'cfloat', 512, 1, [128, 512, 1024])
add('getrf', 'cfloat', 1024, 1, [128, 512])
add('getrf', 'cfloat', 2048, 1, [128])
add('getrf', 'double', 512, 1, [128, 1024])                 # refutation
add('getrf', 'double', 1024, 1, [128])
add('getrf', 'cdouble', 1024, 1, [128])

print("\n".join(C))
