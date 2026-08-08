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

## WP3 (A1) — the primitive-level gate, settled

The plan required this before the solver was touched: *"measure the primitive first. If
`her2k` does not win at the panel's shapes, this package stops."* The concern was that
`her2k_gemm_preferred`'s crossover was measured on general rank-k shapes, while the panel loop
produces a narrow update (k = ib ∈ {16,24,32} against n₂ up to 480) where the fold's extra
n₂²·batch traffic could eat the halved arithmetic.

Measured at n₂ = 480, batch 512, cfloat — `her2k` against the GEMM pair it replaces
(one `gemm` of m=n=480, k=ib, doubled):

| k = ib | her2k | one GEMM | the pair | her2k vs pair |
|---|---|---|---|---|
| 16 | 3.2401 ms | 2.1314 ms | 4.2628 ms | **1.32×** |
| 24 | 3.2703 ms | 2.1603 ms | 4.3206 ms | **1.32×** |
| 32 | 3.3010 ms | 2.1933 ms | 4.3866 ms | **1.33×** |

**her2k wins, and the package proceeds.** But note the shape of the result: her2k's time is
almost independent of k (3.24 → 3.30 ms as k doubles), which is precisely the signature the
concern predicted — it is dominated by the n₂² product-buffer traffic, not by arithmetic. The
GEMM pair is nearly k-independent too, so the ratio holds; but it means this win will *not*
grow with a larger `nb`, and it caps what A1 can deliver.

Propagating 1.33× through the measured phase share (the trailing update is roughly half of the
34.6% that vendor GEMM occupies in the cfloat solve at n = 256) predicts about **1.04×**
end to end — the low end of the plan's 1.05–1.12× estimate, not the middle.

### The fallback, quantified

Forcing the host-loop route (`BATCHLAS_EXPAND_ROUTE=loop`) at n₂=480, k=32, batch 512:

| route | time | vs GEMM pair |
|---|---|---|
| her2k, GEMM+fold | 3.3010 ms | 1.33× faster |
| GEMM pair (today) | 4.3866 ms | — |
| her2k, per-batch host loop | 5.3924 ms | **1.23× slower** |

So the call-site guard is genuinely necessary — but the downside is 1.23×, **not** the 7.8×
that the analogous real-`syr2k` comment in `sytrd_blocked.cc` warns about. That comment was
measured in double, where the vendor loop is far worse. Recording the milder number so nobody
over-engineers the guard on the strength of the wrong precedent.

## WP5 (B1) — the counter premise, independently confirmed

WP5 is the largest package and its entire case rests on the counters in §3 of the research
document. Re-measured here with `ncu` on `LatrdLowerPanelKernelLegacy<float, 256, 0>`,
n = 512, batch 256, ib = 32, j₀ = 0:

| metric | this build | research doc |
|---|---|---|
| DRAM bytes | 4,845 MB | 4,850 MB |
| L2 (`lts__t_bytes`) | 10,050 MB | 10,069 MB |
| L1TEX bytes | **52,499 MB** | **52,499 MB** |
| SM throughput | 10.84% | 10.8% |
| achieved occupancy | 33.19% | 33.2% |
| kernel duration | 11.444 ms | 11.37 ms |

Identical to the byte in the L1 figure. So the 12.2× L1 over-fetch, the 2.34× L2 double-read,
and the 10.8% SM idle all stand, and with them the 2.7× headroom.

The profile also confirms the structural constraint the redesign must respect: the launch is
grid (256,1,1) with 256-thread blocks at batch 256 — **one work-group per matrix**, exactly as
WP5's design section assumes.

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
