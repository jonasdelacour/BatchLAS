# WP3 step 9 — the TRSM measurement grid, and the flip it justified

Raw data behind `RouteTable<Op::trsm, T>::preferred()`. Every number quoted in
that predicate's comment comes from a CSV in this directory.

## What was run

| script | what | output |
|---|---|---|
| `sweep.sh` | the saturated ortho grid, vendor vs native, GPU 0 exclusive | `{right,left}-{vendor,native}.csv` |
| `starved.sh` | batch ∈ {1,8,32}, q ∈ {32,128} — **profile only, never ranked** | `starved-*.csv` |
| `ortho_ab.sh` | end-to-end through the real caller, forced routes | `ortho-{vendor,native}.csv` |
| — | end-to-end with the route **unset**, i.e. testing `preferred()` itself | `ortho-default.csv` |
| `analyse.py` | joins the pairs, enforces saturation, applies the kill criterion | — |

Protocol: one GPU held exclusive by `gpu_guard.sh` (no run reported a foreign
process); harness warmup ahead of every timed iteration, so no cold-JIT number;
`bench::pristine(B)` restored between iterations because trsm is in-place; the
two legs of each pair differ **only** in `BATCHLAS_TRSM_ROUTE`.

## The headline

Ratios are `vendor_ms / native_ms`, so **>1 means native is faster**. Quoted only
where the `(type, side, n, q)` family had stopped scaling with batch.

| type | cells won | worst | best |
|---|---|---|---|
| `double` | 32/32 | 1.39× | 9.62× |
| `complex<double>` | 30/30 | 1.20× | 4.66× |
| `complex<float>` | 30/30 | 1.01× | 21.91× |
| `float`, `Side::Right` | 18/18 | 1.54× | 4.59× |
| `float`, `Side::Left` | 6/16 | **0.57×** | 3.58× |

End-to-end through `ortho` (m ∈ {1024,4096}, k ∈ {16..256}, batch ∈ {128,512},
Chol2 and ShiftChol3): **80/80 cells at or above parity, 1.15×–2.72×**, with the
route unset — i.e. selected by `preferred()`, not forced. Within 4.4% of the
forced-native leg.

## Five things the measurement corrected

**1. The spec's grid is not physical.** §10 asks for n ∈ {8..256} × q ∈
{256,1024,4096} × batch ∈ {128,512,2048}. **Nine of those 54 cells exceed this
box's 24 GB**, the largest asking 70.9 GB. The grid here is capped at 6 GB and
*prints* every dropped cell; a grid that shrinks quietly reads exactly like one
that covered everything.

**2. The starvation constant is refuted, not merely unimplementable.** §10
proposed `batch*q < 8*CU*32 → vendor`. At batch=8, q=32 that product is 256
against a threshold of 32,768 — and native wins those cells **2.2–2.4×**. The
guard would have handed every one of them back to the vendor. (It is *also*
unimplementable as written, since `OpShape::compute_units` still has no writer
and reads 0, but it dies on the measurement first.)

**3. The kill criterion fired the other way.** It was stated in advance: if
native real TRSM exceeds `1.10 × vendor` at the saturated ortho shape, "real
stays vendor-first and only complex flips". Real did **not** lose. `double` wins
every cell on both sides; `float` wins every cell on `Side::Right`. The
predicted outcome — complex-only — is wrong, and the per-cell flip that §11
anticipated is what the data actually supports.

**4. The one losing region is exactly where §3.4 said it would be.** `float`,
`Side::Left`, and the cliff is sharp:

```
order    8    16    32    64   128   256
ratio  3.58  1.47  0.73  0.80  0.77  0.61
```

flat across q and across batch, so not an unsaturated artefact. For `Side::Left`
the q independent solves run down B's *columns*, so consecutive work-items read
addresses `ld` apart — the coalescing problem whose SLM staging tile was
deliberately not built. `double` does not show the cliff at all (its Left column
is 1.39–6.37×, monotone) because cuBLAS's *double* triangular path is weak
enough that the over-fetch never decides the race. **Same kernel, same access
pattern, opposite verdict** — which is why this is a per-type predicate and not
one number. Building the staging tile is now measured work with a known prize,
not a guess.

**5. The test suite cannot validate this flip.** Every trsm call the suite makes
runs at batch ≤ 5, below the batch floor of 8, so `preferred()` leaves all of
them on the vendor. The route diff `wp3-cx → wp3-s9` moved **zero library
decisions**; the only two rows that changed are `route_vocabulary_tests`
recording its own `resolve_trsm_route` calls. That is why the ortho A/B exists —
without it this flip would have shipped with no evidence that it does anything.

## Reading the CSVs

`name,arg0,arg1,arg2,iterations,avg_ms,stddev_ms,GFLOPS,Time (us) / matrix`
where `arg0 = n` (triangular order), `arg1 = q`, `arg2 = batch`. GFLOPS uses the
real-arithmetic convention `n²q` for all four types, so complex understates by
4× by construction — compare `avg_ms` across types, not GFLOPS.
