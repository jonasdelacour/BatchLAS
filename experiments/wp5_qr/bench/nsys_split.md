# WP5 — where the time goes (nsys, vendor-free build)

Derived from `kernsum/*_kern.txt` (`cuda_gpu_kern_sum`). The captures themselves
(`*.nsys-rep`, `*.sqlite`) are **not committed** — see `.gitignore` here — and
must not be quoted as timings: nsys inflates wall time, and these runs use
`WARM_S=0.2` and 2 reps because they are a SPLIT, not a measurement. The wall
times come from `order.csv`, unprofiled. Where the two can be compared they agree
to ~4% (float geqrf n=1024: 53.5 ms/call under nsys against 51.4 ms measured).

Commands: `bash run_nsys.sh`, `bash run_nsys_orgqr.sh`, `bash run_nsys_losers.sh`.

---

## 1. `geqrf`, float, n = 1024, batch = 128 — `native:blocked`

4 calls in the capture (32 panels each; 92 resident-leaf + 36 global-leaf
launches = 128 panels). Per call:

| kernel | % | ms/call | what it is |
|---|---|---|---|
| `GemmTiledGeneralKernel<float,16,…>` | **46.6** | 24.96 | **Tiled16** — the two TRANSPOSED trailing GEMMs, G1 `W1 = Vᴴ A22` and G2 `W2 = Tᴴ W1` |
| `GemmRegister128x128Kernel<float,0>` | 24.3 | 13.01 | the NN trailing GEMM G3 `A22 -= V W2` |
| `LarftKernelName<GeqrfWyTag,float,…>` | 15.2 | 8.13 | WY `T` construction (256- and 128-wide) |
| `GeqrfPanelResidentKernel<float>` | 4.7 | 2.51 | panel leaf, local-memory resident |
| `GeqrfPanelGlobalKernel<float>` | 4.6 | 2.46 | panel leaf, global |
| `PackVKernelName<GeqrfWyTag,float>` | 4.5 | 2.43 | packing `V` |
| `GemmRegisterTiled`, `GemmDirect` | 0.05 | 0.03 | the last, tiny, panels |

Rolled up: **panel factorisation 9.3%**, WY construction (larft + pack_v)
**19.7%**, trailing GEMM **71.0%** — and two thirds of that GEMM time is the
Tiled16 transposed pair.

## 2. `geqrf`, complex<double>, n = 1024, batch = 128 — `native:blocked`

3 calls (96 panels). Per call:

| kernel | % | ms/call |
|---|---|---|
| `GemmTiledGeneralKernel<complex<double>,16,…>` | **69.7** | 967.1 |
| `GemmRegister64x64K16WideKernel<complex<double>,0>` | 21.4 | 296.4 |
| `LarftKernelName<GeqrfWyTag,complex<double>,…>` | 6.1 | 83.9 |
| `GeqrfPanelGlobalKernel` + `…ResidentKernel` | 2.5 | 34.9 |
| `PackVKernelName` | 0.4 | 5.4 |

**Panel factorisation is 2.5% of a vendor-free complex<double> geqrf.**

## 3. `geqrf`, float and complex<double>, n = 64, batch = 8192 — `native:cta`

`GeqrfPanelResidentKernel` is **100.0%** of the capture, for both types. The CTA
tier is one kernel and there is nothing else to attribute.

---

## 3b. The LOSING cells, profiled — because a split from a winning cell does not explain a loss

`run_nsys_losers.sh`. The captures in §1, §2 and §4 are cells where native WINS
(4.3x and 41x). Saying "the cdouble loss at n <= 256 is the transposed GEMM" on
the strength of an n = 1024 capture would be an inference wearing a
measurement's clothes, so the losing cells were profiled directly.

### `geqrf`, complex<double>, n = 256, batch = 2048 — `native:blocked`, **0.84x**

3 calls (8 panels each: 15 resident-leaf + 9 global-leaf launches = 24).

| kernel | % |
|---|---|
| `GemmTiledGeneralKernel<complex<double>,16,…>` | **51.3** |
| `LarftKernelName<GeqrfWyTag,complex<double>,128,…>` | **20.3** |
| `GemmRegister64x64K16WideKernel<complex<double>,0>` | 15.6 |
| `GeqrfPanelResidentKernel` + `…GlobalKernel` | 10.6 |
| `PackVKernelName<GeqrfWyTag,…>` | 2.0 |
| `GemmDirectKernel` | 0.3 |

Same mechanism as the winning n = 1024 cell — the transposed GEMM is the largest
single kernel — but `larft` has grown from 6.1% to 20.3% and the panel from 2.5%
to 10.6%. **At the loss, GEMM is 67.2% and the panel is 10.6%.**

### `geqrf`, double, n = 64, batch = 8192 — `native:cta`, **0.53x**

`GeqrfPanelResidentKernel<double>` is **100.0%** of the capture. The CTA tier has
no trailing update and there is nothing else to attribute: this loss is the panel
kernel at FP64 rate, full stop, and no GEMM change can touch it. (§4 of
`README.md` shows the blocked tier is already 1.09x faster at this cell, so part
of it is a tier-boundary error rather than a kernel limit.)

---

## 4. `orgqr`, float, n = 1024, batch = 128 — `native:blocked`

`SYNTH=1`, so no `geqrf` call contaminates the capture (see §6). 3 calls;
per-call figures use the MEDIAN for the two kernels whose first instance is a
first-touch outlier (`OrgqrIdentityKernel` 32.0 ms on call 1 against 3.46 ms
after; it is a unified-memory page-migration cost, not a kernel cost).

| kernel | % | ms/call |
|---|---|---|
| `GemmTiledGeneralKernel<float,16,…>` | **41.9** | 35.7 |
| `GemmRegister128x128Kernel<float,0>` | 17.5 | 15.0 |
| `LarftKernelName<OrmqrWyTag,float,…>` | 9.7 | 8.4 |
| `GemmRegister128x128Kernel<float,1>` | 6.3 | 5.4 |
| `OrgqrCopyBackKernel<float>` | — | 4.67 |
| `OrgqrIdentityKernel<float>` | — | 3.46 (median) |
| `PackVKernelName<OrmqrWyTag,float>` | 2.5 | 2.1 |
| `TrmmTriangularTilesKernel<float,32>` | 1.3 | 1.1 |

The two kernels that exist ONLY because orgqr is ormqr-on-an-identity — the
identity fill and the copy-back — cost **8.1 ms of 74**, about 11%.

## 5. `orgqr`, complex<double>, n = 1024, batch = 128 and float, n = 64, batch = 8192

| complex<double> n=1024 | % |
|---|---|
| `GemmTiledGeneralKernel<complex<double>,16,…>` | **69.2** |
| `GemmRegister64x64K16WideKernel<…,0>` + `<…,1>` | 24.5 |
| `LarftKernelName<OrmqrWyTag,…>` | 4.1 |
| `OrgqrIdentityKernel` + `OrgqrCopyBackKernel` | 1.9 |

| float n=64 b=8192 | % |
|---|---|
| `LarftKernelName<OrmqrWyTag,float,128,…>` | **49.0** |
| `OrgqrIdentityKernel` | 13.6 |
| `GemmTiledGeneralKernel<float,16,…>` | 10.9 |
| `OrgqrCopyBackKernel` | 8.3 |
| `PackVKernelName<OrmqrWyTag,float>` | 7.7 |
| `GemmRegisterTiledKernel<float,32,32,8,…>` | 6.7 |
| `TrmmTriangularTilesKernel<float,32>` | 3.6 |

The bottleneck is a DIFFERENT kernel at each end of the range: `larft` at n = 64,
the transposed GEMM at n = 1024. A single `preferred()` clause cannot be
motivated by one of them.

---

## 6. The first orgqr capture was WRONG and is not in the tables above

`qrbench_nv orgqr` builds its factor with an **untimed** `geqrf` call before it
times anything. At n = 1024 that call issues 32 panels × (panel + pack_v + larft
+ 3 GEMMs). `cuda_gpu_kern_sum` aggregates by kernel NAME, and the gemm kernels
carry no tag naming their caller — so `GemmTiledGeneralKernel<float,16,…>` in the
first capture is the SUM of orgqr's applies and geqrf's trailing updates.

It was caught because the larft and pack_v rows DO separate, by tag: the first
capture showed `LarftKernelName<GeqrfWyTag,…>` *and*
`LarftKernelName<OrmqrWyTag,…>` in a profile of orgqr, and 32 `GeqrfPanel*`
launches that orgqr does not make. `SYNTH=1` (host-fabricated reflectors,
`H_i = I − τ v vᴴ` with `τ = 2/(vᴴv)`, so the product is unitary and the ortho
probe still discriminates) removes the geqrf call entirely. ormqr's cost is a
function of the shape, not of the reflector values, so the profiled work is the
work the real call does.

Numbers, for the record: the contaminated float n=1024 capture attributed 33.0%
to Tiled16 and 13.2% to the identity fill; the clean one says 41.9% and 15.2%.
The contaminated cdouble capture said 69.2% Tiled16 — which happens to be right,
because at that shape geqrf and orgqr are both Tiled16-dominated. A profile can
be contaminated and still land on the right headline; that is not a defence.
