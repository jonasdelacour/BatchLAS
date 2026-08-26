# WP8 / I3 — the gemv short-reduction fix (G5)

What this directory measured, in the order it was measured, including the two
things it got wrong and had to go back for.

Everything ran on this box's two RTX 4090s. **Device 0 drives the display** and
that turns out to matter; see "the device note" below. Every sweep records the
number of foreign compute processes on its target device before and after each
cell and keeps the max as the last CSV column; it is `0` on every admitted row
in every file here.

---

## 0. The device note, and why half the tables say GPU 1

The brief said to pin `CUDA_VISIBLE_DEVICES=0` and verify the device idle.
Device 0 has **no compute processes** but does host Xorg, gnome-shell and
firefox. That is invisible to `--query-compute-apps` and it is invisible in the
DRAM-resident numbers — but it is **not** invisible in the L2-resident ones:

| out=64, red=64, batch=512, cdouble, T (32 MB, L2-resident) | vendor GB/s | native:cta GB/s |
|---|---|---|
| recorded in `wp7_gemv/repair/parity_r1.csv` (device 1) | 1448.7 | 642.7 |
| device 1, this pass | 1366.2 | 631.5 |
| **device 0, this pass** | **760.6** | 641.7 |

The **native** arm reproduces to within 1.6% on both devices, because it streams
one pass over A from DRAM. The **vendor** arm is 1.8x slower on device 0,
because what cuBLAS does at that shape is convert L2 residency into bandwidth
(1400 GB/s is above the ~1008 GB/s DRAM peak) and the display is evicting the
L2 out from under it.

So: **native-vs-native A/B runs on device 0** (it is unaffected, and the brief
asked for it), **vendor-facing tables run on device 1** (so they are comparable
with the numbers already in the tree). Each file below says which.

---

## 1. Reproducing the defect  (`repro_p{1,2}.csv`, device 0)

cdouble, `native:cta` vs vendor, red_len ∈ {32, 64, 128} × batch ∈ {128, 512},
out_len ∈ {64, 256, 512, 2048}, both transposed spellings, 11 reps, 2 passes.

Reproduced, and the native arm's own throughput is the cleaner statement of it:

    red_len      32        64       128        (roof ~950)
    native   257-435   471-708   771-962  GB/s

Against the vendor the whole red_len = 32 column loses (0.47x–0.69x). Cross-pass
spread: median 1.005, worst 1.052, none above 1.10.

## 2. The mechanism  (`typecurve.csv`, `ncu_precheck.csv`, device 1)

`route_gemv.hh` recorded the cause as "the shuffle ladder … 5 steps, doubled to
10 for a complex scalar". **That reading is refuted.** It predicts `double`
(5 shuffles of a 64-bit value = 10 hardware shuffles) and `complex<float>`
(10 shuffles of a 32-bit value = 10 hardware shuffles) are hurt equally.

Body 3, out_len 2048, batch 512, Trans, DRAM-resident, GB/s:

    red_len       32      64     128    2048 (each type's own roof)
    float      833.8   924.2   931.0    952.3
    double     547.5   928.6   932.2    953.8
    cfloat     921.2   925.4   932.7    954.0
    cdouble    434.5   708.5   932.5    952.2

double and cfloat are **1.68x apart at identical bytes and identical shuffle
count**. `ncu` on the same launches names the discriminator:

    sm__pipe_fp64_cycles_active, % peak     dram__throughput, % peak
    red_len      32      64     128          32      64     128
    cdouble   85.55   86.11   84.69       40.61   66.09   95.08
    double    85.01   82.74   58.33       50.61   95.78   95.39
    cfloat     0.00    0.00    0.00       95.30   95.15   95.27
    float      0.00    0.00    0.00       79.09   95.59   95.63

Occupancy holds at 79–93% and sectors-per-load is ideal (16.00 for cdouble,
8.00/4.00 for the 8- and 4-byte types) throughout. **The fold is FP64 work on a
1/64-rate GeForce part**: `sg_sum` runs log2(32) = 5 add steps on all 32 lanes,
i.e. 160 double-adds per output for `double` and 320 for `complex<double>`,
against only red_len useful FMAs.

Body 5 cuts that to L·log2(L) — 8 adds at L = 4.

## 3. Choosing W  (`wchoice_p1.csv`, `wfine_p1.csv`, device 0)

`gemvsegab.cpp` interleaves body 5 and body 3 **rep by rep inside one process**,
reads each arm's resolved kernel back from `gemv_seg_trans_width_debug` (both
`wA` and `wB` are CSV columns, and a row with `wA == wB` is refused as comparing
an arm with itself), and checks both arms against the same in-process host
oracle. The two arms are *not* bit-identical and are not asserted to be: they
sum the same products in a different order.

The plan predicted a sector floor would bound W (`L·sizeof(T) ≥ 32`, giving
W ≤ 4 for float). **Measured, that is wrong below red_len ≈ 32**: float at
out_len 2048 runs 5.16x–5.91x at W = 8 (16-byte runs — *half* a sector) against
3.34x–3.40x at W = 4. Below the warp width the kernel is not sector-bound;
body 3 is idling 32 − red_len of its 32 lanes and recovering them dominates. The
floor does re-assert itself at the long end, which is where the shipped table
turns to W = 4.

## 4. The three gates, transcribed

    gate 1  red_len ≤ 32 (float) / 16 (cfloat) / 48 (double) / 64 (cdouble)
    gate 2  W = 8 up to red_len 24 / 16 / 32 / 32, then W = 4
    gate 3  out_len·batch ≥ 16·CU  (W = 8 band)  /  64·CU  (W = 4 band)

Gate 1 is where body 3 stops being materially short of the roof, per type.
Gate 3 exists because **body 5 launches W times fewer sub-groups than body 3**.

## 5. Both sides of gate 1  (`above_p{1,2}.csv`, device 0)

`WS="4 8"` forces W and bypasses all three gates, so these rows say what body 5
*would* do above the gate. Just above each type's gate, at DRAM-resident
footprint, it measures 0.983–0.996 — a **revert** under GATE-B, which is what
makes the gate load-bearing rather than decorative.

The same sweep also shows an **unclaimed window**: at out_len = 256, batch = 512
(33–67 MB against a 72 MB L2) body 5 at W = 4 measures 1.40x–2.09x for cfloat at
red_len 24–64, 2.62x for double at red_len 64, and 1.22x–1.71x for float at
red_len 48–128 — all above their gates. Separating those from the 0.99x cells at
the same red_len needs a **footprint** term, which is the L2-residency reasoning
`route_gemv.hh:279-284` forbids and which would be no better founded in a
launcher than in `preferred()`. Left open and stated, not taken.

## 6. THE MISTAKE, AND HOW IT WAS CAUGHT  (`skinny_p{1,2}.csv`)

`plane_cells.txt`'s minimum out_len is **64**. On that grid body 5 scored 83
admitted cells, geomean 3.28x, **zero below 1.00x** — and that was wrong about
the regime it could not reach. Re-running the WP7 audit's parity grid, whose
minimum out_len is **1**, showed the native arm 3–6% *slower* than before the
change at out_len = 1.

Walking the output axis down (`skinny_cells.txt`, out_len 1…64) found **sixteen
losing cells**, worst **0.891x**, every one at out_len·batch ≤ 4096:

    out_len·batch   128    512   1024   2048   4096   ≥ 8192
    losing cells      4      7      2      2      1        0
    worst          0.891  0.957  0.978  0.983  0.998     n/a

That is campaign trap 8 — "a grid that cannot reach a regime is not evidence
about it" — committed by this pass and caught by an instrument that already
existed. Gate 3 is the fix.

A first cut at a single floor of 8·CU left five cells at 0.976–0.998. A **third
and fourth pass at 31 reps** (`resid_p{3,4}.csv`) separated them: `double`
recovered (0.986/1.002, noise) but **three float cells reproduced below 1.00 in
both passes**, all in the W = 4 band. Hence the two-row floor. After it
(`skinny3_p{1,2}.csv`): 30 admitted cells, geomean 1.566x, **MIN 1.037x, zero
below 1.00x**; 74 declined cells at 0.977–1.009.

The cost is seven small wins (1.02x–1.36x) on shapes whose whole launch is a few
thousand outputs. The alternative was shipping a reproduced 0.976x.

## 7. The shipped result  (`planeF_p{1,2}.csv`, `conjF_p{1,2}.csv`, device 0)

| | cells | geomean (worse pass) | min | max | below 1.00 |
|---|---|---|---|---|---|
| Trans, admitted | 83 | 3.286 | 1.070 | 10.47 | 0 |
| Trans, declined | 53 | — | 0.993 | 1.003 | — |
| ConjTrans, admitted | 36 | 2.746 | 1.074 | 10.44 | 0 |
| ConjTrans, declined | 16 | — | 0.985 | 1.002 | — |
| skinny (out_len 1…64), admitted | 30 | 1.566 | 1.037 | 3.04 | 0 |

## 8. Odd `ld`  (`oddld_p1.csv`, device 0)

A run starts at `b·stride + j·ld + s`, so an `ld` that is not a multiple of the
run length straddles an extra 32-byte sector. It costs body 5 something and
**never inverts the sign**: cdouble red_len 8 goes 7.32x → 6.65x, double red_len
8 goes 9.92x → 5.59x, cfloat red_len 8 goes 4.06x → 2.13x, float red_len 32 goes
1.076x → 1.062x. Every admitted odd-`ld` cell stays at or above 1.06x. The suite
already exercises `ld = 79` at `m = 70`.

## 9. Registers and local memory  (`regs.csv`, device 1)

The build emits PTX rather than a cubin into `libbatchlas_sycl.so`, so this is a
runtime fact and `ncu` is the instrument. **Every `GemvSegTKernel<T,W>` uses
exactly the registers `GemvCtaTKernel<T>` uses, at every W:**

| type | regs | static smem | dynamic smem | max wg (65536/regs) | ladder's max wg |
|---|---|---|---|---|---|
| float | 28 | 0 | 0 | 2336 | 256 |
| double | 36 | 0 | 0 | 1792 | 256 |
| complex\<float\> | 38 | 0 | 0 | 1696 | 256 |
| complex\<double\> | 40 | 0 | 0 | 1632 | 256 |

Zero bytes of local memory, static and dynamic, on all twelve new entry
functions — so the recorded 48 KB launch hole stays structurally unreachable in
this TU. `regs·wg` is at most 40·256 = 10240 of 65536.

## 10. The vendor-facing parity gate  (`parity_w8_p{1,2}.csv`, device 1)

`wp7_gemv/audit/parity.sh` re-run unchanged, two passes, scored against
`wp7_gemv/repair/parity_r{1,2}.csv` by `parity_score.py`.

**The two recorded sub-0.50x blockers clear the line**, which is the honest
claim — they do not reach parity, and cannot: the vendor is at 1400 GB/s there,
above the DRAM peak.

    cdouble out=64 red=64 batch=512 T   0.450 -> 0.862
    cdouble out=64 red=64 batch=512 C   0.472 -> 0.861

Cells below 0.50x after: **0**.

## 11. The route diff  (`.route-diff/wp8i3-after-{v,nv}`)

Body 5 is a kernel choice **inside** `{Native, CTA}`, so by construction it
cannot move a route decision — `preferred()` and `supports()` are untouched.
"By construction" is exactly the reasoning this campaign distrusts, so it was
captured and diffed against I2's snapshot anyway:

| | added | removed | ops touched | non-gemv rows moved |
|---|---|---|---|---|
| `wp8i2-after-v` → `wp8i3-after-v` | 60 | 0 | gemv only | 0 |
| `wp8i2-after-nv` → `wp8i3-after-nv` | 60 | 0 | gemv only | 0 |

**Zero removed rows** is the claim: no decision changed. The 60 additions are new
`(scalar, shape_class)` buckets that the new body-5 test cases *reach* and
nothing reached before — new coverage, not new choices. Four of them in the
vendor-present build resolve `native:cta`; those are
`SegTransCasesAreReachable`'s deliberate `vendor_available = false` query, which
is the route line it prints.

The health checks the script does not do (`rd_health.sh`):

    label              reached  linked  miss  decisions  gemv  getrf  getrs  getri  shards
    wp8i2-after-v         4214      40     0       3794   174     56     68     48      53
    wp8i3-after-v         4274      40     0       3854   234     56     68     48      53
    wp8i2-after-nv        3974      40    55       3653   174     48     68     48      53
    wp8i3-after-nv        4034      40    55       3713   234     48     68     48      53

`linked` holds at 40, `miss` holds at 55 in the vendor-free capture (a **zero**
there is defect 4's signature — the gate-declined half unrecorded — and the
script does not catch it), 53 shards merged, and the three LU ops' row counts
are unchanged.

## 12. Device-link delta  (`linktime.sh`)

Twelve new entry functions (4 types × W ∈ {2,4,8}). Arms alternated, idle box,
`batchlas_sycl` target, 3 reps each:

    present  59.096  58.259  58.658   median 58.658
    stubbed  58.915  58.096  58.396   median 58.396

**+0.26 s, +0.45%** — inside the present arm's own 0.84 s spread. Free, as body
4's twenty instantiations were (58.03 vs 57.59).

## Files

| file | what |
|---|---|
| `sweep.sh` | generic (out_len, red_len, batch) sweep, arms pinned, foreign count per row |
| `segab.sh` + `gemvsegab.cpp` + `build_ab.sh` | the GATE-B A/B: body 5 vs body 3, interleaved rep by rep in one process |
| `analyse.py` / `plane.py` / `wtable.py` / `parity_score.py` | scorers; each refuses and NAMES bad rows |
| `ncu_precheck.sh` | the FP64-pipe pre-check |
| `regs.sh` | registers / shared memory per entry function |
| `oddld.sh` | packed vs odd `ld` |
| `breaks_body5.py` | the GATE-D break campaign (build-novendor) |
| `correctness.sh` / `reach.sh` | relerr across every W and shape; which kernel actually launched |
