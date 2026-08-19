# WP3 TRSM — corrections to the spec, before any code is written

`WP3_TRSM_SPEC.md` states that "every line citation below was re-read from source at `aa827f5`".
**`aa827f5` predates WP1 and WP2**, both of which have since landed and changed the dispatch
architecture. This document records what a verification pass against `b02e43e` found. Read it
alongside the spec; where they disagree, this wins.

Method: six independent readers, one per claim-cluster, each finding then adversarially refuted
by a separate reader instructed to default to "does not hold". 27 findings survived. Every
`wrong-edit` item below was additionally re-checked by hand — including two where the
verification pass itself was wrong (marked ✱).

---

## What survives, and is not re-litigated here

- **The rejection of diagonal-block inversion at every tier** (§2.4). Untouched by WP1/WP2, and
  its argument — that the "free for ortho" licence does not survive restatement in orthogonality
  currency — stands.
- **The V1/V2 composition.** V2's dependency is intact: `sycl_gemm::gemm_custom` still has the
  9-argument signature §2.3 calls, instantiated for all four scalars
  (`src/sycl/gemm_kernels.hh:66-74`, `gemm_kernels.cc:801,810,819,828`). Note it takes a
  `ComputePrecision` the spec does not mention; pass `ComputePrecision::Default`.
- **The 24-case canonicalisation** (§5.2), and §9.3's argument that the two in-tree references
  are *one* implementation so the test oracle must be an independent multiply-back
  (`netlib_lapack.cc:445-449` and `cublas.cc:1134-1137` still fold identically).
- **§3.3's conclusion** that the grid must be `batch × ceil(q/WG)` and not batch-only. This is
  the repo's recurring starvation defect and the conclusion is right even though two rows of its
  table are not.

---

## Wrong-edit findings — following the spec here produces incorrect code

### 1. The three hook points no longer exist

The spec routes at `cublas.cc:1594`, `rocblas.cc:138`, `netlib_lapack.cc:404`. All three are
dead; `cublas.cc:1594` is now a line inside an instantiation macro block. **WP1 left exactly one
public `trsm`**, the facade at `src/dispatch/entry_points/level3.cc:156-171`; the three backends
now own `trsm_vendor` only (`cublas.cc:1092`, `rocblas.cc:131`, `netlib_lapack.cc:427`).

One hook, in the facade, **before** the vendor-available test — anything after `level3.cc:165`
is unreachable in the vendor-free build WP3 exists for. This also fixes netlib's missing
`trsm_validate_params` for every backend in one edit, since the facade validates nothing today.

### 2. `parse_cublasdx_variant_request` was deleted; `TrsmVariant` is the vocabulary WP0 removed

The spec's §6.4/§8 propose `enum class TrsmVariant {Vendor,Native,Auto}` plus
`trsm_variant_request()` via `parse_cublasdx_variant_request`. That function no longer exists —
`src/backends/route_common.hh:35-41` is its tombstone, recording that all four callers now go
through `dispatch::parse_route_env`. Verified: no definition anywhere in `src/` or `include/`.

Worse, the env variable is wrong. `legacy_variable_for` (`route_env.hh:109-121`) has **no
`Op::trsm` case**, so `BATCHLAS_TRSM_VARIANT` is read by nothing. The variable is
**`BATCHLAS_TRSM_ROUTE`**. The spec instructs the implementer to pin and test the native route
with a variable no code reads (spec:553, :555, :622, :707).

### 3. A single `trsm_use_native()` bool cannot express the state the vendor-free build needs

The spec's §10 predicate mixes env read, structural correctness and speed thresholds into one
boolean. This is the trap `route_gemm.hh:5-30` is written to prevent: *"supports() ==
correctness only… preferred() == the measured window… the env read lives in the alias table."*

The consequence is concrete, not stylistic. `route_resolve.hh:60-63` implements the vendor-off
fallback by re-walking the order testing **only** `is_native(*r) && Table::supports(*r, s)`. With
the thresholds in the only predicate, every real-type cell and everything below the starvation
cut has *no route at all* in a vendor-free build, and `level3.cc:165-167` throws — defeating the
work package's own purpose. **Every number in spec:649-662 belongs in `preferred()`; none of it
belongs in `supports()`.**

### 4. The SLM size formula writes out of bounds

§3.4 specifies the transpose staging tile with **stride `NB_STAGE + 1`** (to keep SLM column
reads conflict-free), but §4.1's size formula allocates `NB_STAGE * WG`. Re-derived by hand: at
`WG=128, NB_STAGE=16` the last index is `15 + 127*17 = 2174` into a 2048-element allocation —
**127 elements past the end**.

Fix: `+ (side == Left ? (NB_STAGE + 1) * WG : 0)`. Every §4.2 `Side::Left` total gains
`WG*sizeof(T)`; worst case (cdouble, N=16, Left) goes 35 200 → 37 248 B, still under the 45 056
budget, so no feasibility row flips. Precedent for putting the padded stride in the size:
`group_blas_subgroup_common.hh:56,58`.

### 5. `TriangularTransform` is in the wrong namespace

Spec §6.1 cites `batchlas::device::detail::TriangularTransform`. Verified by hand: the struct is
at `group_blas_common.hh:102`, inside `namespace batchlas::device` (opened at `:19`), and
`namespace detail` does not open until `:179`. Correct name is
**`batchlas::device::TriangularTransform`**. The *Tag* form `detail::TriangularTransformTag`
(`:646`) genuinely is in `detail`. The spec's own body usage at spec:70 is unqualified and
already correct — only the §6.1 citation is wrong. Compile error if transcribed.

### 6. ✱ The documented test command runs zero tests and exits 0

Spec:573 gives `ctest -L blas -L ortho`. Repeated `-L` is an **AND**, and no test carries two
component labels (`tests/CMakeLists.txt:171-182` returns on the first matching component). Run
by hand: **`Total Tests: 0`, exit code 0** — a silent false green, under a spec section whose
entire correctness argument rests on those targets running.

**✱ The verification pass proposed `ctest -L "blas\|ortho"`, and that is also wrong** — run by
hand it likewise returns 0 tests. The working form is a bare pipe:

Measured, all four run by hand:

- `ctest -L blas -L ortho` (two `-L` flags) → **0 tests**, exit 0. This is what the spec says.
- `ctest -L blas` → 15 tests. `ctest -L ortho` → 5 tests.
- One `-L`, alternation written with a **backslash before the pipe** → **0 tests**. This was the
  verification pass's proposed correction and it is also broken.
- One `-L`, alternation written with a **bare pipe**, no backslash → **20 tests** (= 15 + 5).

So: use one `-L` with a bare pipe between the two labels, or use the `-R` form, which the spec
gets right. Quote the argument so the shell does not treat the pipe as a command separator.

---

## The measurement gate — the spec's first gate is not executable as written

§1 makes `n_cta(T)` a hard gate: "must be *confirmed* with `-Xcuda-ptxas -v` before any other
code is written". **It cannot be done per-TU.** Device code is AOT-compiled to an sm_89 cubin at
the *shared-library device link* (`cmake/BatchLASDetectSYCL.cmake:528,544-552`), so the flag on a
single-TU compile is reported "argument unused". This is not drift — it is identical at
`aa827f5`, i.e. a pre-existing authoring error.

**On `stack frame == 0` (spec:703)** this document originally said the spec was wrong, because
220 of 376 entry functions in this library carry a non-zero stack frame *with*
`0 bytes spill stores, 0 bytes spill loads`. That generalisation does not hold for the TRSM
kernel and was itself corrected by running the gate — see the ✱ section below. What remains true
regardless: since the flag silently does nothing on a per-TU compile, grepping such a log for
"spill" finds nothing and reads as "no spill". That is a phantom measurement whichever gate you
use, and it is the reason the recipe below exists.

**Working recipe:** `scripts/register_probe.sh`, which replays
`build/src/CMakeFiles/batchlas_sycl.dir/link.txt` verbatim with a second
`-Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v` pair appended and `-o` redirected.
No reconfigure needed.

### ✱ The gate, corrected again by running it (WP3 step 3)

This document originally said: gate on spill bytes, **not** on stack frame. Measured against the
actual TRSM kernels, that is wrong, and the spec's `stack frame == 0` was right — for a reason
neither document had:

| type | N | registers | stack frame | spill |
|---|---|---|---|---|
| float | 8 / 16 / 32 | 42 / 76 / 114 | 0 | 0 |
| float | **64** | 119 | **256 B** | 0 |
| double | 8 / 16 / 32 | 59 / 100 / 153 | 0 | 0 |
| double | **64** | 145 | **512 B** | 0 |

**Nothing spills**, including `double N=64` — so the spec's "256 B/thread cliff" is falsified,
as this document said. But 256 B is 64 floats and 512 B is 64 doubles: that is `x[]` itself,
placed in local memory rather than promoted to registers. ptxas calls that a **stack frame**,
not a spill, because the array was never in registers to be spilled out of — and register
residency is V1's entire thesis. A spill-only check passes N=64 while the design is void.

The distinction is kernel-specific and that is why both documents got it wrong from one side
each: in the GEMM kernels this document generalised from, 220 of 376 entry functions carry a
benign non-zero frame; in **this** kernel the only thing that can be on the stack is the
accumulator array.

**The gate is: `stack frame == 0` AND `0 bytes spill stores/loads` AND
`registers × WG <= 65536`.**

Measured capacity: **`n_cta(float) = 32`, `n_cta(double) = 32`.** The spec predicted float 64;
its instruction "if x[64] spills, reduce `n_cta(float)` to 32" reached the right number by a
mechanism that does not occur.

Each kernel appears twice (`…_with_offset` and not) and the two can differ by a couple of
registers; take the max, and grep by mangled name.

**The `256 B/thread` cliff the spec derives `n_cta` from does not exist.** `gemm_kernels.cc:725-735`
records the opposite, measured: at an 8×8 tile double compiles to 208 registers and
`complex<float>` to 247, *both spill-free*. So `n_cta(double)=32` and `n_cta(cdouble)=16` are
hypotheses. Put **N=64 double** — the 128-accumulator configuration measured spill-free — in the
step-3 falsification set before accepting them.

---

## Two budget claims that are already false

- **spec:713** ("if the `batchlas_sycl_obj` link grows past ~30 s, cut the bucket ladders")
  fires unconditionally with *zero* TRSM code: the link measures **43.9 s**. It also names the
  wrong target — `batchlas_sycl_obj` is an OBJECT library with no link step
  (`src/CMakeLists.txt:35`); the link unit is the shared lib (`:180-182`). Make the budget a
  **delta against 43.9 s**, measured immediately before the step that adds kernels.
- **spec:477** ("`src/sycl/` is a small, isolated device-link unit"). Still isolated, no longer
  small — 376 entry functions in the one TU after WP2.

---

## `OpShape::compute_units` is dead, and `preferred()` needs it

`route.hh:240` declares it; verified by hand, it has **zero writers and zero readers** in
`include/`, `src/` and `tests/`. The starvation guard §3.3 wants (`batch*q` against the SM count)
cannot be expressed until something populates it — and it must be populated by the *shape
builder*, not the table, because `route_resolve.hh:19-21` requires the table to stay pure ("no
getenv, no SYCL query"). Until then it reads 0, so `preferred()` must return false rather than
divide by it.

---

## What WP3 can and cannot claim

**Unblocked vendor-free:** `trsm` itself — `level3.cc` used to throw for every call. With WP2's
GEMM that makes gemm+trsm the first vendor-free level-3 pair.

**✱ But not `trsm_tests` as a suite, which this document originally claimed.** Measured after
steps 4-6: 16 `TrsmOperationsTest` cases now pass vendor-free where every one previously threw
— and they are the two **CUDA** parameterisations. The NETLIB ones still fail, because the CTA
kernel is a GPU kernel and `supports()` correctly reports `is_gpu == false` as unsupported. The
vendor-free failing *set* is therefore byte-identical to the WP2 baseline: the suite stays red
while most of its GPU content went green. A CPU trsm is separate work.

**Still red, and trsm cannot help.** Each of these is gated on a *different* missing op:

| suite | actually blocked by |
|---|---|
| `ortho_tests` | `potrf` (`ortho.cc:200,288`), `geqrf`/`orgqr` (`:377`), `syev` (`:339`) |
| `cond_tests` | `syev` (`cond.cc:46,52`), `getrf`/`getri` via `inv` |
| `inverse_tests` | `getrf`/`getri` |

So the spec's headline end-to-end validation target, `ortho_tests`, **is not available in a
vendor-free build** — it validates trsm on a vendor-present box only. The honest claim is that
WP3 removes `trsm` from the vendor-dependency list; it makes no *extension* vendor-free.

**Performance: unclaimed by default.** Step 6 routes nothing; later steps flip cells only where
measured.

---

## Open questions, each with what settles it

1. **`>=` vs `>` in the WG ladder** (spec:183 vs the table at :209-210) — a spec
   self-contradiction, not a fact about the tree. Decide, then edit one of the two.
2. **Should the trsm shape builder populate `compute_units`, or should `Queue` cache it for
   every op?** `src/util/queue-impl.cc` already queries `max_compute_units`.
3. ~~**The starvation constant `8` (spec:654) and "complex flips native" (spec:659)** are both
   hypotheses.~~ ✱ **SETTLED BY STEP 9, AND BOTH ARE WRONG.** The grid was run; see
   `experiments/wp3_s9/`. The starvation guard is not merely unimplementable, it is
   *refuted*: at `batch=8, q=32` the product is 256 against its own threshold of 32,768,
   and native wins those cells 2.2–2.4×. And "complex flips native" understates the
   result — `double` wins **32 of 32** saturated cells (1.39–9.62×) and `float` wins every
   `Side::Right` cell (1.54–4.59×). The stated kill criterion ("if native real exceeds
   1.10× vendor, real stays vendor-first and only complex flips") did not fire.
4. **`n_cta(double)`/`n_cta(cdouble)`** — settled by the register gate above.
5. **DPC++'s SLM carveout** — `local_accessor` lowering to dynamic shared memory with a quantised
   Ada carveout could cap CTAs regardless of the register arithmetic. Same `ncu` run as §4.4.
6. **Does `record_level3_route` accept a trsm-shaped call?** (`level3_coverage.hh`) — trsm's `k`
   is the triangular order, not a GEMM `k`. One read before step 6, or the route-diff instrument
   reports nothing for trsm while looking healthy.
7. **Should `BATCHLAS_TRSM_VARIANT` exist as a legacy alias at all?** It never shipped; not adding
   it is defensible. A decision, not a measurement.

---

## ✱ What step 9 measured, and the three claims it disproved

The grid exists (`experiments/wp3_s9/`, with its own README), `preferred()` is live,
and this is the first WP3 change that moves traffic. Three corrections, each to
something this document or the spec asserted:

**The §10 grid cannot be run as written.** Nine of its 54 cells exceed this box's
24 GB, the largest asking 70.9 GB once the harness's pristine copy of `B` is counted.
The implemented grid caps at 6 GB and prints every dropped cell. My own first draft
of the cap table omitted the pristine copy and understated every row by ~2×, which
would have put four dropped cells back into the table on paper while the code kept
dropping them — the comment now says to read the figures off `trsm_grid_bytes()`
rather than re-derive them.

**"WP3 cannot claim a speed result" is now false, but the claim it can make is
narrower than the win suggests.** `preferred()` moved **zero library decisions**.
Every trsm call the test suite issues runs at batch ≤ 5, below the measured batch
floor of 8, so the route diff `wp3-cx → wp3-s9` shows only two changed rows and
both are `route_vocabulary_tests` recording its own `resolve_trsm_route` calls. The
speed claim rests entirely on the `ortho` A/B — 80/80 cells at or above parity,
1.15–2.72×, with the route unset so that `preferred()` is what selects. A flip
validated only by the suite would have been validated by nothing.

**The `Side::Left` staging tile of §3.4 is no longer optional-and-unmeasured.** It
is the one losing region in the whole grid: `float`, `Side::Left`, order ≥ 32,
0.57–0.87×, with a sharp cliff between order 16 and 32 and flat across q and batch.
`double` at the same shapes wins 1.39–6.37× — same kernel, same access pattern,
opposite verdict, because cuBLAS's double triangular path is weak enough that the
over-fetch does not decide the race. The predicate encodes the cliff
(`float && Side::Left → order <= 16`) rather than papering over it, so building the
tile has a known prize and a ready-made A/B.

### Revised remaining order

| step | what | status |
|---|---|---|
| 9 | the §10 grid | **done** — `experiments/wp3_s9/` |
| 10 | widen `preferred()` for complex | **done, and wider than planned** — folded into step 9; complex, double, and float-Right all flipped together because the grid ranked them in one pass |
| 11 | per-cell real flips | **done** — same commit; the per-cell structure is the `float`/`Side::Left` split |
| 12 | the `Side::Left` SLM staging tile (§3.4) | **done** — `experiments/wp3_s12/`; float/Left window moved from `order <= 16` to `order <= 128` |
| 13 | `MatrixView::operator()(Slice,Slice)` passing the parent pointer array (`matrix.hh:1140`) | open, reported, deliberately untouched — needs its own verification pass |

---

## ✱ What step 12 built, and the fourth spec claim it corrected

The §3.4 staging tile exists. `float`, `Side::Left`, orders 32–128 went from
0.70–0.79× to 1.19–3.20×, and `preferred()`'s clause widened accordingly. Order
256 stays the vendor's at 0.76–0.93×.

**§3.4 named the right factor at the wrong level.** It predicts "8× over-fetch
on both the read and the write-allocate", which reads as DRAM traffic. Measured
with ncu: the factor is right (31.4 load sectors per request against a coalesced
floor of 4, i.e. 7.85×) but DRAM is at **0.75–0.85× of the analytic floor**, i.e.
*below* it. The bytes a lane skips at step `s` are the bytes it wants at steps
`s+1..s+7` and they are still in cache. The defect is entirely LSU/L1
transaction count. The tile fixes it either way — 31.39 → 5.13 on the load,
32.00 → 4.00 on the store, 0.517 → 0.145 ms — but had this been read as a
bandwidth problem the obvious responses (vectorised loads, wider tiles) would
have been aimed at the wrong resource.

**Staging is gated to the real types, and that is a measurement, not a
simplification.** Applied to complex it costs those kernels their register
residency: `complex<float>` N=32 Left gains a 464-byte stack frame and
`complex<double>` N=32 a 232-byte one, both with zero spill, because the nested
round loop stops fully unrolling for the wide bodies. They also cannot benefit —
over-fetch is `32/sizeof(T)` lanes per sector, so a 16-byte scalar is capped at
2×, and float loses precisely because 32/4 = 8. Measured after gating: complex
is 1.00–1.01×, i.e. untouched, and all 24 kernels are back to zero frame.

**The end-to-end check had a hole this step also closed.** `ortho_benchmark`
hardcoded `Transpose::NoTrans`, and `ortho.cc:205,289` select the trsm side from
exactly that flag — so step 9's caller-level validation covered `Side::Right`
only, and the entire `Side::Left` half of the table had never been exercised
through a real caller. `arg4` now selects it: with the route unset, 80/80 cells
at or above parity, and at order 256 the default correctly tracks the vendor.

### Revised remaining order

| step | what | status |
|---|---|---|
| 9–11 | the grid, and the `preferred()` flips | done |
| 12 | the `Side::Left` SLM staging tile | **done** |
| 13 | where between order 128 and 256 the float/Left window actually ends | open — the grid jumps, so the boundary is placed at the largest order measured to win |
| 14 | `MatrixView::operator()(Slice,Slice)` passing the parent pointer array (`matrix.hh:1140`) | open, reported, deliberately untouched |
