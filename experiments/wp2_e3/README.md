# WP2 E3 — is the native route actually faster than cuBLAS DGEMM?

Raw data and the scripts that produced it. Every number in `WP2_GEMM_SPEC.md`'s E3 section
comes from these files.

## Why this exists

`RouteTable<Op::gemm, double>::preferred()` accepts square, batch >= 64, `max_dim <= 512` —
i.e. everything from n=4 to n=512. Flipping the unset default (E6) would route all of it off
cuBLAS, and **no measurement of that window existed**: the only `Tiled16`-vs-cuBLAS numbers in
the tree were at 256³ and above, and 585 of the 666 real double calls the flip would move land
on `Tiled16`, not on the wide-scalar tile WP2 measured.

## Conditions

RTX 4090 / sm_89, one dedicated GPU via `experiments/gpu_guard.sh`, `--warmup=5` (SYCL JIT has
fabricated a 3.7× regression here before), median of 3 repeats, **both β=0 and β=1**.

Sanity anchors, both of which held: a 4090 is 1/64 FP64, so DGEMM must not exceed ~1450
GFLOP/s — peak observed 1246. And the endpoints reproduce `WP2_WIDE_SCALAR_GEMM_VERDICT.md`
independently (vendor 1244 vs its 1232–1247; native 1411 vs its 1413–1415).

## Files

| file | what it answers |
|---|---|
| `e3_double.csv` | main sweep, n=32..512, batch 512, both betas, 3 reps |
| `e3_small.csv` | the bottom of the window, n=4..32 at batch 4096 — where real demand sits |
| `e3_sat.csv` | saturation: batch 64→8192 at n=48 and n=136. The ratio must not shrink |
| `e3_n32.csv` | Direct vs Tiled16 either side of the `max_dim <= 32` boundary |
| `e3_bound.csv` | that boundary again with 3 reps and both betas, before changing code |
| `e3_after.csv` | n=24 and n=32 after moving the boundary to 24 |

`e3_sweep.sh`, `e3_small.sh`, `e3_sat.sh`, `e3_bound.sh` regenerate them; `e3_report.py`
tabulates `e3_double.csv` with the spread beside every ratio, because a win inside the spread
is not a win. `which_kernel.sh` reports which kernel `select_kernel_variant` actually lands on
per n — the comparison only means what it claims if the native arm runs the kernel named.

## Result

Native wins **every cell** from n=4 to n=512, 1.05×–4.51×, at saturation, both betas, with
run-to-run spreads of 0.0–0.3%. The one exception found — n=32 losing at 0.92× — was a
misplaced `Direct`/`Tiled16` boundary, not a routing problem; moving it to `max_dim <= 24`
turned that cell into a 1.14–1.23× win.

Scope: square, NN, aligned, real-typed. Says nothing about complex, transposed or ragged.
