# WP2 E5 — non-square, measured on the shapes the library actually issues

## Why this exists

`preferred()` required `m == n == k`, so **no non-square GEMM had ever routed native** — while
the WP2 E2-prep demand analysis says that is where nearly all of BatchLAS's own GEMM lives.
The dominant internal shape is a **panel update**: large m, large n, small k, with k the
blocking factor clustered at 8/32/48/96/136.

So this sweep does not walk a shape cross-product. It measures the shapes the demand table
says the library issues: 992×992×32, 480×480×32, 288×288×32, 224×224×32, 248×248×8,
312×312×8, in the NN/NT/TN forms those call sites use.

## Result: a completely clean split by type

`e5_double.csv` / `e5_float.csv` — batch 128, both betas, median of 3, `gpu_guard`:

| type | cells | verdict |
|---|---|---|
| **double** | 36 / 36 | **native wins, 1.10–1.41×** |
| **float** | 0 / 36 | **native loses, 0.22–0.51×** |

Float needs no change — its window is already `max_dim <= 32` after E4, so these shapes were
never eligible. E5 is therefore a double-only widening.

## Where the widened window stops

`e5_edges.csv` — the extremes, because a predicate belongs where the evidence stops:

| shape | ratio |
|---|---|
| 1024³ | 1.13× |
| 2048³ | 1.14× |
| 4096×64×64 | 1.04–1.06× |
| 64×4096×64 | 1.04–1.05× |
| 992×992×8 | 1.39–1.46× |
| **512×512×1** | **0.49×** ← the only loss |

`e5_k.csv` places that boundary exactly. At 512×512×k, sweeping k = 1, 2, 3, 4, 6, 8, 12, 16:

| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| ratio | **0.49×** | 1.64× | 1.62× | 1.58× | 1.53× | 1.49× | 1.34× | 1.09× |

**k = 1 is the only losing case in the entire work package for double.** A rank-1 update is
barely a GEMM and cuBLAS has a dedicated path for it — note its advantage is β=0 only (cuBLAS
230 → 114 GFLOP/s when β=1, while native is 112 either way). k=2 already wins 1.64×, so the
predicate is `k >= 2` rather than a rounder number. This is not a corner case: k=1 is 761
calls in the demand table.

## The one place this reaches past its measurements

The widened window has **no upper size bound**, and 2048³ is the largest shape measured. That
is deliberate and argued rather than overlooked: the limit here is an **FP64 issue rate** — a
4090 is 1/64 FP32, ceiling ~1.44 TFLOP/s — and native sits at 88–98% of it while cuBLAS sits
at 78–87%, at *every* size from 4 to 2048. A gap produced by a size-independent mechanism does
not need a size cutoff, and a cliff at 2048 would itself be the unjustified number.
