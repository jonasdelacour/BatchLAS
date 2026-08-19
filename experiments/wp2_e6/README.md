# WP2 E6 — the flip, and the unmeasured region the prediction caught

`legacy_unset_default(Op::gemm)` went from `{Vendor, Auto}` to `{Auto, Auto}`, removing the
asymmetry where GEMM was the one op whose native kernel never ran by default.

## The prediction caught a gap

The route-diff discipline says a routing step enumerates its intended moves *before* it makes
them. `e6_predict.py` does that against the pre-flip capture — and the prediction is what
found the problem: **`preferred()`'s `double` branch is a bare `max_dim <= 512` with no
transpose test at all.** So the flip would route double TN / NT / TT — and ConjTrans, which
float's branch explicitly rejects — natively, and E3 had measured NN only.

That was not a safe assumption to carry: E4 had just measured float's transposed window at
**0.34–0.55×** of cuBLAS. Had double behaved the same way, the flip would have shipped a 2–3×
regression on every transposed double GEMM.

## What the measurement said

`e6_dtrans.csv` — double, square, batch 512, n=32..512, TN/NT/TT/CN, both betas, median of 3,
`gpu_guard`:

| form | n=32 | n=64 | n=128 | n=256 | n=512 |
|---|---|---|---|---|---|
| TN | 1.01–1.02× | 1.09–1.12× | 1.10–1.11× | 1.11× | 1.11× |
| NT | 1.06–1.08× | 1.09–1.11× | 1.10–1.11× | 1.11× | 1.11× |
| TT | 1.04–1.06× | 1.12× | 1.11–1.12× | 1.12× | 1.12× |
| CN | 1.01–1.03× | 1.10–1.11× | 1.10–1.11× | 1.11× | 1.11× |

**No losses anywhere**, ConjTrans included. Spreads 0.0–4.2%.

The contrast with float is a ceiling argument, not luck: cuBLAS DGEMM sits at ~86% of the
4090's ~1.44 TFLOP/s FP64 ceiling and `Tiled16` reaches ~96%, so there is room to win. cuBLAS
SGEMM is strong and the *transposed* register family plateaus near 15–18 TFLOP/s, so there is
not.

## What the flip turns on

Everything below is GPU-only, square, `batch >= 64`, homogeneous, default precision:

- **double**, n=4..512, all transpose forms — 1.01–4.51× over cuBLAS DGEMM
- **float**, NN only, `max_dim <= 32` — 1.03–1.46× over cuBLAS SGEMM

and nothing else. Complex is refused outright by `preferred()`.

## What it does not change

- **The vendor-free build.** `resolve_route` already fell back to any *supported* native route
  when no vendor exists (`route_resolve.hh:60-62`), so the Vendor default was never reached
  there.
- **Explicit requests.** `BATCHLAS_GEMM_VARIANT=vendor` still means vendor — the escape hatch
  if a future cuBLAS turns any of these cells around, which is not hypothetical: see the
  parity claim that aged out in `experiments/wp2_e4/README.md`.
