# WP7 repair pass — the parity blocker, fixed and re-measured

The [audit](../audit/README.md) found that **WP7's parity gate failed as first
shipped**: 15 of 192 cells below B6's 0.50× blocker line, worst **0.08×**. Thirteen
were one family — `Algorithm::Direct`, `transA = NoTrans`, `out_len < 32` — and the
audit's recommendation was to state it as a known weakness rather than fix it.

**It was fixed instead.** This directory is the measurement.

| | before | after |
|---|---|---|
| cells below 0.50× (of 192) | **15** | **2** |
| worst cell | **0.08×** | **0.44×** |
| the `NoTrans`, `out_len < 32` family (24 cells) | min 0.08, median 0.38, **13 blockers** | min 0.60, median **1.16**, max **4.09**, **0 blockers** |
| whole grid, worst of two passes | — | min 0.444, median 1.007, 181/192 ≥ 0.85× |

The two remaining blockers are the **already-stated** `complex<double>`
short-reduction weakness on the CTA arm (`out_len=64, red_len=64, batch=512`,
0.444× / 0.456×), not the family that was fixed. They are a different kernel and a
different mechanism; see `route_gemv.hh`'s weakness block.

---

## Why the original gate said "ZERO cells below 0.50×" and was wrong

It was **correct about its own 84 cells**. The lesson is about the grid, not the
arithmetic:

> **The minimum `out_len` anywhere in `../ab/`'s seven shapes is 64, and the whole
> failing family lives below 32.** A grid that cannot reach a regime is not
> evidence about it.

The audit found it by defining cells in **(`out_len`, `red_len`)** rather than in
(m, n) — so a skinny cell stays skinny when the operation is transposed instead of
silently becoming its own opposite — and by walking `out_len` down to 1.

## The mechanism: two effects, both stopping exactly at 32 lanes

From `../audit/mechanism.csv` (body 1) and `mechanism_body4.csv` (body 4), ncu:

* **Coalescing.** B5's flattening `b = gid/out_len, i = gid%out_len` makes a warp
  straddle `32/out_len` batch items, whose rows are `stride_a` apart. Sectors per
  global load, `out_len` = 1, 2, 4, 8, 16, 24, 31, 32, 48, 64:
  **32.0, 16.0, 12.0, 10.0, 9.0, 9.0, 9.5, 8.5, 8.67, 8.5.**
* **Parallelism.** `out_len * batch` is body 1's ONLY extent. At `out_len = 1,
  batch = 512` the launch is **16 work-groups on a 128-SM box** — 2.08% achieved
  occupancy, 7.03% of DRAM.

Two controls that make it a warp story rather than a bytes story, both from the
audit: float turns at the **same** `out_len = 32` despite a 4× narrower scalar, and
padding `ld` moves nothing (at `out_len` 16, `ld` 16/17/24 gives sectors
9.00/9.50/9.00).

## The fix: `src/sycl/gemv_native.cc` **body 4**

With `W = gemv_seg_width(out_len)` — the largest power of two with
`W*out_len <= 32` — lane `l` of a 32-lane sub-group takes

```
i = l % out_len        (which output)
jsub = l / out_len     (which slice of the reduction)
```

and walks `j = jsub, jsub+W, jsub+2W, ...`. Lanes `l` and `l+out_len` hold partial
sums of the **same** output and are folded by a `log2(W)`-step shuffle at stride
`out_len`. That restores 32 contiguous elements per warp when `ld == m`, and
multiplies the launch extent by `32/out_len`.

Measured effect on the two ncu quantities it targets, float, `out_len = 1`:
sectors per load **32.00 → 4.00**, occupancy **2.08% → 8.13%**.

**It is not a new route.** `{Native, Direct}` on a `NoTrans` shape now names two
kernels, chosen on `out_len` and on whether the device **enumerates** a sub-group
size of 32. Putting that in `supports()` would be a speed cutoff in the predicate
documented to carry correctness only; and on the `native_cpu` queue the
enumeration is false, so the 20 `Backend::NETLIB` rows keep body 1 and WP7's
no-GPU-gate deliverable is untouched.

## THE RESULT WORTH CARRYING FORWARD: `W` had to be a template parameter

The first version of body 4 carried `W` as a runtime `const int`. **It fixed
`out_len <= 4` and REGRESSED `out_len >= 8` below the body it replaced.**

GB/s, float, `n = 2048`, batch chosen for a ~128 MB footprint
(`outlen_body1.csv`, `outlen_body4.csv`):

| `out_len` | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| body 1 | 235.1 | 335.4 | 517.3 | 730.5 | 692.9 | 827.2 |
| body 4, `W` runtime | 906.5 | 707.9 | 576.1 | 607.5 | **373.8** | **461.1** |
| body 4, `W` `constexpr` | **934.9** | **921.4** | **913.2** | **903.0** | 624.7 | **861.1** |
| cuBLAS | 901.3 | 1281.0 | 1112.4 | 1012.8 | 897.6 | 962.0 |

**ncu says it was not the memory system.** At `out_len = 16`, float, the
runtime-`W` version already had sectors-per-load **2.50** (ideal) and **8.27%**
occupancy against body 1's 3.00 and 4.12% — and still ran at **26%** of DRAM where
body 1 reached **69%**. Better coalescing *and* better occupancy, slower kernel:
the loop, not the traffic. With `W` a compile-time constant the trip count
`(red_len - jsub + W - 1)/W` and the address stride `W*lda` are both known, the
loop unrolls, and the same shapes run at 90–98% of the vendor.

The cost is 5 instantiations per scalar type — 40 extra entry functions — and it is
**free at link**: same TU touched, same procedure, idle box, **58.03 s with body 4
absent vs 57.59 s with it present.**

`out_len = 12` is the one place body 1 is still ahead by more than noise (0.77×
against body 4's 0.70×): `W = 2` there leaves 8 of 32 lanes idle because 12 does
not divide 32. A documented residual, not a blocker, and not worth a
non-power-of-two lane packing.

## Method

Both passes ran on the audit's own harness (`../audit/parity.sh`, unmodified),
against `../ab/gemvab_v` **rebuilt after the kernel change** (campaign trap 2 — it
resolves the printed route in its own TU), on **device 1 verified idle**, with the
arms interleaved inside each cell and the resolved route asserted against the pin
on every row.

* `relerr` is **exactly 0** on all 1,152 timed rows across both passes.
* The route column agrees with the pin on **384 of 384** compared cells; the
  analysis asserts it rather than eyeballing it.
* **Zero foreign compute processes** on the target device on every row.
* Cross-pass ratio spread: **median 1.0046**. Eight of 192 cells exceed 1.10, and
  every one is a small L2-resident footprint where **cuBLAS** is the unstable arm
  (its GB/s is above the ~1008 GB/s DRAM peak there). The tables above quote the
  **worst** of the two passes, which is the conservative direction.

## Files

| file | what it is |
|---|---|
| `parity_r1.csv`, `parity_r2.csv` | 640 rows each — the audit's (`out_len`, `red_len`) parity ladder, re-run against the repaired kernel, two independent passes |
| `analyse_parity.py` | the blocker/parity report; asserts `relerr == 0` and route-column agreement |
| `outlen_body1.csv`, `outlen_body4.csv`, `outlen_sweep.sh`, `outlen_compare.py` | the body-1-vs-body-4 comparison across `out_len` 1..16 at two footprints (body 1 captured by temporarily disabling the gate) |
| `mechanism_body4.csv`, `mechanism_body4.sh` | ncu sectors / occupancy / DRAM for body 4, same metrics and shapes as `../audit/mechanism.sh` used for body 1 |
| `breaks_kernel.py` | the 14-break campaign against `src/sycl/gemv_native.cc` |
| `breaks_route.py` | the 7-break arming proof for the new `route_vocabulary_tests.cc` gemv section |

Reproduce:

```bash
experiments/wp7_gemv/ab/build.sh
OUT=experiments/wp7_gemv/repair/parity_r1.csv GPU=1 experiments/wp7_gemv/audit/parity.sh
python3 experiments/wp7_gemv/repair/analyse_parity.py repair/parity_r1.csv repair/parity_r2.csv
```
