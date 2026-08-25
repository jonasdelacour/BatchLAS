# WP7 — native batched GEMV: the experiment index

Four phases, four directories, in the order they happened. **Read them in this
order**; each one corrects a claim the previous one made.

| directory | what it establishes | the claim it CORRECTED |
|---|---|---|
| [`baseline/`](baseline/) | cuBLAS `gemvStridedBatched` is at **94–105% of the ~950 GB/s achievable DRAM roof** on 90 of 92 reproducing cells, over all four scalar types and both `transA`. **There is no bandwidth to take.** The one exception is `complex<double>` + `Trans`, 310–380 GB/s. | — (recon) |
| [`ab/`](ab/README.md) | the native kernel at parity on 84 default-route cells; the `complex<double>` prize mapped at a fixed ~1 GB footprint; `preferred()` ships **all-false**, with an `n*batch` rule built, tested out-of-sample and **refuted**. | — |
| [`audit/`](audit/README.md) | 2,052 timed rows. **The parity gate as first shipped FAILED**: 15 cells below B6's 0.50× line, worst 0.08×, 13 of them one family that `ab/`'s grid could not reach. B5's flattening verified by ncu. `ConjTrans` measured as a full peer. | `ab/`'s “no cell is below 0.50×” and its 24-cell rectangle |
| [`repair/`](repair/README.md) | that family **FIXED** — `src/sycl/gemv_native.cc` body 4 — and re-measured on the audit's own harness over two fresh passes. Blockers **15 → 2**. | the audit's “not urgent, state it as a weakness” |

## The one-paragraph result

`gemv` is **vendor-free at parity**. cuBLAS is at the DRAM roof almost everywhere
and a batched `gemv` reads A once for two flops, so there is no arithmetic to hide
behind and no reuse to exploit: parity *is* the achievable outcome, and it is the
same headline WP6 shipped. `preferred()` is **all-false**, so a vendor-present
build sends `gemv` nowhere new — verified by `route_diff`, 0 moved decisions.

Two things are not parity, and both are real:

* **The prize is real and is NOT taken by default.** `complex<double>`
  transposed at large footprint runs **1.9×–3.1×** faster natively — the dip is
  cuBLAS's and is type-exclusive (at matched bytes the native CTA body reads
  936–941 GB/s for *all four* types while cuBLAS reads 323 for `complex<double>`
  and 940–946 for the rest). It does not ship as a `preferred()` clause because
  the win is **batch-conditional** and no predicate over
  `(scalar, transA, m, n, batch)` separates win from loss without moving cells
  that are already at the roof. It is one environment variable away:
  `BATCHLAS_GEMV_ROUTE=native:cta`, on a transposed GPU shape.
* **One weakness remains stated rather than fixed**, per B6: `complex<double>`
  transposed with `red_len() <= 64` measures 0.43–0.46×. Mechanism, named fix and
  measured boundary are in [`repair/README.md`](repair/README.md) and in
  `include/batchlas/blas/dispatch/route_gemv.hh`.

## The methodological result worth carrying to the next work package

**A grid that cannot reach a regime is not evidence about it.** `ab/`'s parity
gate reported "ZERO cells below 0.50×" over 84 cells and was *correct about those
84 cells*. Its minimum `out_len` was 64; the whole failing family lived below 32.
The audit found it by defining cells in **(`out_len`, `red_len`)** rather than in
(m, n) — so that a skinny cell stays skinny when the operation is transposed
instead of silently becoming its own opposite — and by walking `out_len` down to 1.

## Reproducing

Everything runs against the **vendor-present** build, so both arms are reachable
from one process and can be interleaved inside one session.

```bash
experiments/wp7_gemv/ab/build.sh          # REBUILD THIS after any preferred() change
GPU=1 experiments/wp7_gemv/audit/parity.sh    # the (out_len, red_len) parity ladder
GPU=1 experiments/wp7_gemv/audit/prize.sh     # the cdouble region, batch-resolved
```

Campaign rules that these harnesses encode, and that cost time when skipped:

* **Pin `CUDA_VISIBLE_DEVICES` to a device verified idle** with
  `nvidia-smi --query-compute-apps`, not to device 0 by convention. This box has
  two RTX 4090s. A contended row can have a **low** relative standard deviation —
  the `rel_sd` guard does not catch it — so every parity and prize row carries a
  count of foreign compute processes on the target device.
* **Pin the route explicitly**: `native:cta` or `native:direct`, never a bare
  `native`, which resolves to the first *supported* route.
* **Print the resolved route as a column.** A kernel being linked is not evidence
  it ran. The harness resolves the route in its own TU, so it must be rebuilt
  after any `preferred()` change or the column lies.
* Interleave arms **within one cell**, so a clock drift has to hit all of them.
