# WP2 E4 — is float's `preferred()` window in the right place?

Raw data and scripts. Every number in `WP2_GEMM_SPEC.md`'s E4 section comes from here.

## Why this exists

`RouteTable<Op::gemm, float>::preferred()` claimed three windows:

- NN, `max_dim <= 32`
- NN, `128 <= max_dim <= 512`
- transposed (non-ConjTrans), `batch >= 128` and `128 <= max_dim <= 512`

The flip (E6) would act on all three. The spec anticipated E4 might be *"a narrowing as much
as a widening"*. It is a narrowing: **two of the three windows are measured losses**, and
nothing anywhere argued to widen.

## Conditions

RTX 4090 / sm_89, dedicated GPU via `experiments/gpu_guard.sh`, `--warmup=5`, median of 3,
both β=0 and β=1. Sanity anchor: peak vendor SGEMM 47.8–53.9 TFLOP/s — real FP32 (a number
near 80 would mean TF32).

## Files

| file | what it answers |
|---|---|
| `e4_nn.csv` | float NN, n=8..1024, across all four regions of the window |
| `e4_batch.csv` | is the NN result a batch artefact? batch 128/512/1024 at n=128..512 |
| `e4_trans.csv` | float TN/NT/TT, n=64..768 — the transposed window |
| `e4_n192.csv` | native-vs-native at the worst cell: 8 kernels at n=160/192/224/320 |
| `e4_large.csv` | does the predicated-128x128 win hold at n=544..1056? |
| `e4_ld.csv` | aligned dims, UNALIGNED ld — the case `gemm_tests` pins |

`e4_which.sh` maps n to the kernel `select_kernel_variant` actually lands on; a float ratio is
meaningless without it, since the float branch has ~10 exits.

## Results

**1. NN `max_dim <= 32` — correct, keep.** Native wins: n=8 1.46×, n=16 1.31×, n=32 1.08×.

**2. NN `128..512` — removed.** Loses in every cell: n=128 0.97×, n=192 0.40×, n=256 0.87×,
n=384 0.79×, n=512 0.91×. Flat across batch 128/512/1024, so not an unsaturated artefact.

**3. Transposed `128..512` — removed.** Loses in *all 30 cells*, 0.34–0.55×, across TN, NT and
TT. Not a fallback effect: TN runs its dedicated `register_128x32_k32_tn` kernel (traced). The
transposed register family plateaus near 15–18 TFLOP/s while cuBLAS SGEMM reaches 45+.

**4. A selector fix worth more than the window.** `select_kernel_variant` sent squareish
float that could not use the 128×128 fast path to the *generic* 128×32×32 route, with a
comment saying the 128×128 predicated path "has not been benchmarked against the generic
route". Benchmarked now, it wins everywhere:

| case | generic | predicated | gain |
|---|---|---|---|
| n=192 | 9 781 | 18 000 | 1.84× |
| n=320 | 12 188 | 25 288 | 2.07× |
| n=1056 | 15 065 | 33 314 | 2.21× |
| n=256, ld+2 | 7 237 | 23 966 | **3.31×** |
| n=512, ld+2 | 8 399 | 36 862 | **4.39×** |

The unaligned-**ld** case gains most, and it is the one that matters: a panel is a sub-view
carrying its parent's ld, so unaligned ld is what the factorisations hand to `gemm`.

## One in-tree claim that aged out

`src/sycl/gemm/register_128x128.hh` records "43.6 TFLOP/s against cuBLAS SGEMM's 43.9" at
512³ b512 — parity. Re-measured: native **43.5** (reproduces exactly), cuBLAS **47.3**. The
native half held; the vendor half moved, presumably a cuBLAS upgrade. The parity claim did not
survive it.

## Scope

Square, real-typed float. Says nothing about complex or about non-square shapes.
