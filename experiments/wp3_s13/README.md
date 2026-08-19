# WP3 step 13 — win at every order 8..512, both sides, float and complex\<float\>

## Result

Worst clean cell per order (relative sd ≤ 10%), vendor_ms / native_ms, >1 = native wins:

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| float Left | 1.59 | 1.72 | 1.77 | 1.48 | 1.18 | **0.86** | **0.76** |
| float Right | 1.61 | 2.72 | 3.36 | 2.31 | 1.49 | 1.55 | 1.23 |
| complex\<float\> Left | 1.17 | 1.37 | 3.35 | 4.09 | 12.25 | 8.03 | 17.76 |
| complex\<float\> Right | 1.01 | 1.02 | 1.35 | 1.66 | 4.90 | 5.23 | 8.11 |

**Three of the four (type, side) combinations win at every order 8–512.** The
fourth, float `Side::Left`, wins at every order too — but only below a work
threshold. Above it, orders ≥ 256 lose and the router sends them to the vendor,
so BatchLAS's trsm is never slower than cuBLAS anywhere in the grid.

## What changed

**V2's outer block width is no longer the CTA capacity.** It was
`nb = trsm_cta_max_n<T>() = 32`, which at n=512 means 16 iterations, 31
serialized launches, and every trailing GEMM with one dimension pinned at 32.
GEMM arithmetic intensity with a dimension pinned at w tends to `2w/sizeof(T)`
flop/byte — 16 for float, against an RTX 4090 machine balance of ~42 — so 93.75%
of the solve's flops ran bandwidth-bound *by construction*, on a problem that at
n=512 is intrinsically compute-bound (51 flop/byte). The left-looking re-read
factor `(p-1)/2` compounds it: 7.5× at n=512.

Now the trailing update runs at `OUTER_NB` and each panel is solved by the old
nb=32 loop against its own, much shorter prefix. `BATCHLAS_TRSM_OUTER_NB`
overrides it for sweeps.

**And the width is side-dependent, which is measured, not aesthetic.** Sweeping
{32, 64, 128, 256} on float, worst cell per order:

```
          Side::Left                     Side::Right
order   nb32  nb64 nb128 nb256    nb32  nb64 nb128 nb256
  128   1.18  1.10  1.20  1.17    1.00  0.96  1.00  0.98
  256   0.75  0.78  0.87  0.75    1.01  0.94  0.83  1.01
  512   0.58  0.74  0.76  0.75    1.07  0.92  0.82  0.91
```

Widening helps Left at every large order and **hurts Right at every large
order**, turning two winning Right cells into losses. The sides put the width on
different GEMM dimensions (Left's update is `C(nb × q)`, Right's is `C(q × nb)`)
and so land in different clauses of `select_kernel_variant`; widening also
shortens the inner updates' `k`, and float's transposed fast paths require
`k ≥ 128`. So Right keeps the single-level schedule. One number for both sides
would have to be 32, discarding everything the two-level driver buys.

`nb256` degenerating to `nb32` at n=256 (0.75 vs 0.75, 1.01 vs 1.01) is the
internal consistency check that the knob does what it says.

## The routing rule, and why it is a work threshold

float `Side::Left` after the change:

```
order 256:  q*b=32768 1.12x   131072 1.40-1.49x   524288 0.90-0.92x   2097152 0.86-0.87x
order 512:  q*b=32768 1.23x   131072 1.05-1.10x   524288 0.76-0.77x
```

Order 512 **wins** at small work and order 256 **loses** at large work, so an
order cap cannot express the boundary. `preferred()` now reads
`order <= 128 || q*batch < 524288`. Neither side is bandwidth-bound in these
cells (both run at 11–26% of DRAM peak), so this is the re-read amplification
escaping L2, not a bandwidth wall.

## Two things tried and abandoned, both by measurement

**Raising the CTA bucket to N=64.** This is the highest-leverage lever left —
the traffic model at n=512, in B-elements per batch item, units of q:

| schedule | traffic | vs ideal (1024) |
|---|---|---|
| NB=32 (old) | 5824 | 5.7× |
| NB=128, nb=32 (now) | 4096 | 4.0× |
| NB=128, nb=64 | 3328 | 3.3× |
| NB=128, nb=128 | 2560 | 2.5× |

The N=64 rejection predated the `Side::Left` staging tile, which had cut float
Left from 114 registers to 53 — so the arithmetic that killed it no longer
described the kernel and it was worth re-testing. **It still fails**, and by more
than before: 456 B stack frame on Left, 256 B on Right, zero spill. Left is
*worse* than Right because the tile's own live state competes with the
accumulator rather than paying for it. Reverted.

So the remaining gap is bounded by the CTA capacity, and closing it needs an
inner block of 64+ that a one-solve-per-work-item design cannot hold. The route
is a cooperative solve — W work-items per solve exchanging `x_s` by sub-group
broadcast, so each holds nb/W elements — which is a redesign, not an increment.

**`OUTER_NB = 128` everywhere.** See the table above; it regresses Right.

## Two facts that change what these ratios mean

**The complex "vendor" is not cuBLAS.** `src/backends/cublas.cc:1111` diverts
both complex types to a hand-written SYCL substitution kernel — a sequential
per-RHS back-substitution — on the strength of an uncited comment about NaNs
under SYCL/USM interop. `cublasCtrsmBatched`/`cublasZtrsmBatched` at :1220 are
unreachable. So every complex ratio here is native vs *another BatchLAS kernel*,
which is why complex "wins" 8–26× at large order. This is worth settling
separately; it is not a vendor-independence result.

**The marginal complex cells are roofline ties, not defects.** complex\<float\>
`Side::Right` at orders 8 and 16 measures 1.01–1.05×, and in exactly those cells
native runs at **88.5–90.5% of the 1008 GB/s DRAM peak**. The ceiling there is
1.12–1.18×, so no kernel change buys a factor. The same orders at small q·batch
fit in L2, are not DRAM-bound, and win 1.25–2.04×.

## Verified

`trsm_tests` 86/86 including five new suites that cross a panel boundary — every
pre-existing blocked test stopped at order 100, so with `OUTER_NB=128` they all
took a single panel and would have passed without ever entering the two-level
path. Both alpha-placement mutations fail (outer beta → 4 failures, inner
beta → 9). `ctest -L 'blas|ortho'` 20/20. Full vendor-present 52/53 (documented
`lanczos_tests` baseline). Vendor-free `trsm_tests` 49 → **54** passing, failing
set byte-identical. Route diff `wp3-s12 → wp3-s13`: **identical**, 3053
decisions unmoved.

## Data

`baseline-before.csv` — before the change; `baseline.csv` — after;
`nb_sweep.csv` — the OUTER_NB sweep. Each row is one (type, side, route, n, q,
batch) cell with its mean and relative standard deviation, aggregated from the
per-invocation CSVs the drivers write (those are regenerable and not committed).
`baseline.py`, `nb_sweep.py` are the drivers; both are serial by construction and
noise-gated, because a concurrent second copy of a sweep script is invisible to
`gpu_guard.sh` and produced 22 of 180 bad cells in step 12.
