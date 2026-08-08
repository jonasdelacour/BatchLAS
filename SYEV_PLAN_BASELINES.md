# SYEV plan — pre-change baselines

Measured 2026-08-08 on this worktree's own build (`build-cuda`), configured identically to
the build behind `SYEV_PERF_RESEARCH.md`: CUDA on, `sm_89`, `RelWithDebInfo`, shared ccache,
plus tests. Device 1 of two RTX 4090s, one process, GPU otherwise idle
(`CUDA_VISIBLE_DEVICES=1`), clocks warmed by a discarded JIT run first.

All figures are **µs/matrix**, `--warmup=2 --min_iters=5`. Benchmark positional arguments are
`n batch nb fuse jobz uplo`.

## Rig validation

Before trusting any of the numbers below, the rig was checked against two figures published in
the research document at the same shape (n = 512, batch 512, eigenvectors, two-stage):

| | this build | research doc | agreement |
|---|---|---|---|
| float | 334.28 | 332.5 | 0.5% |
| cfloat | 971.17 | 972.4 | 0.1% |

Standard deviations across the sweeps below run 0.02–0.5%, so differences under ~1% are noise.

## WP1 (A2) — values-only, `BATCHLAS_SYEV_PROVIDER=blocked`, batch 1024

The regime A2 attacks: `blocked` owns n ≤ 320 in values mode, and 28.3% of the float solve at
n = 256 is an eigenvector divide-and-conquer whose output is discarded.

| n | float | cfloat |
|---|---|---|
| 64 | 1.1614 | 1.5055 |
| 128 | 4.5231 | 7.3078 |
| 192 | 11.566 | 21.486 |
| 256 | **20.774** | **47.218** |
| 320 | 42.804 | 101.45 |

**Gate:** ≥ 1.2× on float at n = 256, i.e. **≤ 17.3 µs/matrix**.

Note the expected asymmetry: stedc runs in real arithmetic, so its *absolute* cost is the same
for both types. It is 28.3% of the float solve but only ~12% of the cfloat solve at n = 256
(scaling by 20.774/47.218), so float should gain substantially more than cfloat here. A result
where cfloat gains as much as float would be suspicious rather than pleasing.

## WP3 (A1) — eigenvectors, `provider=blocked`, n = 256, batch 1024

| | µs/matrix |
|---|---|
| float | 31.657 |
| cfloat | **66.628** |

**Gate:** ≥ 1.05× on cfloat, and float unchanged (it takes the same code path — assert, do not
assume).

## WP4 (A3) — eigenvectors, `provider=two_stage`, n = 512, batch 512

The A3 optimum was re-measured here directly through the env knobs, independently of the code
change, by forcing `BATCHLAS_SB2ST_BACK_TILE_W=2 BATCHLAS_SB2ST_BACK_SUBS=4`:

| | shipped constants | tile=2 subs=4 | effect |
|---|---|---|---|
| cfloat | 971.17 | **852.77** | **1.139× faster** |
| float | 334.28 | **384.30** | **0.87× — i.e. 1.15× SLOWER** |

Two things follow, and the second is the more important one:

1. The claimed 1.14× for complex reproduces (research: 855.75, here 852.77).
2. **The same constants cost float 15%.** This is direct measured evidence that the geometry
   must be selected per scalar type rather than changed globally — which is exactly what WP4
   specifies. A global flip would trade a 1.14× complex win for a 1.15× float loss on the
   provider float is actually routed to at this n.

**Gate:** ≥ 1.10× on cfloat *and* float bit-identical (same instantiation selected).

## WP2 (B3) — eigenvectors, `provider=blocked`, batch 2048

The n range the `cta-large-n` port targets (its measured local-memory limits are n = 128 for
float, 64 for cfloat). These are the numbers a CTA-resident solve has to beat.

| n | float | cfloat |
|---|---|---|
| 33 | 0.65996 | 0.80367 |
| 48 | 0.97778 | 1.3450 |
| 64 | 1.5123 | 2.2372 |
| 96 | 3.4767 | 5.6691 |
| 128 | 6.5491 | 10.982 |

**Gate:** CTA beats these by ≥ 1.15× at any n in 33–128 → proceed to the routing gate;
otherwise record and close the branch.

## Reproducing

```bash
cd .claude/worktrees/syev-perf-implementation-plan
CUDA_VISIBLE_DEVICES=1 BATCHLAS_SYEV_PROVIDER=blocked \
  ./build-cuda/benchmarks/syev_benchmark --backend=CUDA --type=float,cfloat \
  --warmup=2 --min_iters=5 64,128,192,256,320 1024 0 0 0 0
```

Discard the first run of a fresh process: SYCL JIT has fabricated a 3.7× loss on this box
before. Check `nvidia-smi` first — contention between the two cards' processes has produced
spurious 3.6× "wins" here.
