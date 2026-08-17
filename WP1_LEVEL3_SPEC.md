# WP1 — Free the level-3 tile routes from cuBLAS

Output of a 12-agent design pass: five agents mapped the tree, three designed the work from
independent angles, three judged those designs adversarially through one lens each
(correctness/linkage, performance, verifiability). This supersedes §WP1 of
`VENDOR_INDEPENDENCE_PLAN.md`, which it contradicts in five places.

Every claim below is cited by file and line and was re-checked against the source before
being written down. Where the mapping contradicted the plan, the plan is wrong — the
corrections are listed first because three of them change what WP1 *is*.

## What the plan got wrong

**1. `batchlas::gemm` is not "the public, already-routing entry point".** The plan's central
instruction is to retarget the expansions' terminal GEMM at it. But the facade is

```cpp
// src/dispatch/entry_points/level3.cc:60-64
if constexpr (!dispatch::level3_vendor_available<Back>) {
    dispatch::throw_no_vendor_route<T>(dispatch::Op::gemm, Back, dispatch::kLevel3Library<Back>);
} else {
    return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
}
```

The three-way route (cuBLASDx → native SYCL → raw cuBLAS) lives in `backend::gemm_vendor` at
`cublas.cc:153-178` — *inside the cuBLAS-gated TU WP1 exists to escape*, and
`level3_vendor_available<CUDA>` is `bool(BATCHLAS_HAS_CUBLAS)`
(`vendor_available.hh:34-38`). `level3.cc:24-26` admits this in its own header comment.

Consequence: retargeting at the public `gemm` is the right *destination* but, on its own,
converts a vendor-free `NoRouteError` naming `symm` into one naming `gemm`. The facade's
`gemm` must gain a native arm first, or WP1 delivers nothing vendor-free.

**2. `symm` has no tile kernel.** `symm_custom_dispatch.cc` includes no `*_tiles.hh` and calls
no tile kernel (`grep -c '_tiles(' == 0`). Its only portable kernel is
`detail::expand_mirrored<float,false>` at `:100` (`triangular_expand.hh:189-258`); everything
after the expansion is `gemm_cublasdx` at `:112` and `:123`. So for `symm`, WP1 is "free an
*expansion* kernel and re-point its GEMM", not "free a tile kernel". Three ops carry tile
kernels, not four.

**3. The four tile routes are unreachable outside `cublas.cc`.** `symm_use_cuda_custom`,
`syrk_use_cuda_custom`, `syr2k_use_cuda_custom` and `trmm_use_cuda_custom` have exactly four
call sites in the tree — `cublas.cc:269`, `:763`, `:882`, `:1005` — and the facade never calls
them. Compiling the TUs in every configuration makes them *linked* everywhere and *called*
nowhere. `route_compiled.hh:36-39` claims flipping `level3_tile_kernels_compiled` is "the only
edit needed here"; it is not.

**4. The measured crossover constants are 4 and 256, not 2 and 128.** The plan quotes the
measured *loss region* ("batch <= 2 && n <= 128"); the guard in code is its complement over a
wider region — `batch >= kExpandMinBatch(4) || max_dim >= kExpandMinDim(256)`
(`triangular_expand.hh:44-45,59`). WP1 must preserve 4 and 256.

Two further crossovers sit outside WP1's stated scope entirely: `herk`'s
`batch >= 4 && n <= 768` (`cublas.cc:403-409`) and `her2k`'s `batch >= 2 || n >= 128`
(`expansion_budget.hh:112-116`). `hemm`'s expansion call is at `cublas.cc:319`, inside the
vendor TU, and is not moved by WP1.

**5. The level-3 ops have no `RouteTable` and never call `resolve_route`.** Only `gemm`,
`gesvd`, `ormqr` and `syev` have specialisations. WP0 gave `symm`/`syrk`/`syr2k`/`trmm` the
Route *vocabulary* (`parse_route_env`, `is_plain_vendor`) but not the *resolver*; their
thresholds are hand-rolled `if`-chains, expressed as neither `supports()` nor `preferred()`.

This one also invalidates an instrument: `scripts/route_diff.sh` records nothing for the four
ops WP1 changes. Repairing that is step 0.

## Corrections to WP0's own output

`VENDOR_FREE_BASELINE.md` says "`gemm`'s native register-tiled kernel exists and is
vendor-free, so `gemm_tests` fails only on the shapes outside `gemm_custom_problem_supported`".
Per correction 1 that cannot be true — vendor-free `gemm` throws on *every* call, before
reaching any route. The S7 static table's `gemm native=1` is not wrong on its own terms
(`coverage.cc:139-143` defines `native` as **linked**, and the kernel is linked in
`batchlas_sycl`) but it is misleading for the purpose the table exists to serve: the burn-down
question is *reachable*, not *linked*. Both documents are corrected as part of step 0.

## The design

Three designs were produced and judged. `retarget-only` won on every lens
(9 / 7 / 8 against `split-tu`'s 3 / 4 / 6).

`split-tu` — split each TU into portable and CUDA halves — was killed by a **confirmed silent
route change**. It transcribes syrk's gate thresholds into `RouteTable::preferred`, but the
live code uses those thresholds *gate-only*: `syrk_cuda_custom`'s Auto arm takes
`syrk_triangular_tiles` unconditionally once the gram test fails (`syrk_custom_dispatch.cc:215-218`),
with no second preference check. Since `triangular_tiles_per_side(256) == 2`
(`triangular_tiles.hh:118-123`), a transcribed `>= 3` rule rejects the tile route for
`129 <= n <= 383` at every batch, sending n=256 to a path that writes **both triangles** —
a shape `tests/syrk_tests.cc:427-428` names explicitly. Two judges found this independently.

The winning thesis: **the four TUs have two distinct terminal couplings, and only one can be
pointed at a public entry point.**

- *Downward* (`gemm_cublasdx` at symm:112,123, syrk:148, syr2k:107,111 — trmm has none) can
  become `batchlas::gemm<Backend::CUDA,float>`. On this box that is provably route-identical:
  gemm's unset default is `{Vendor,Auto}`, `resolve_route` returns it verbatim while a vendor
  exists (`route_resolve.hh:78-79`), so `gemm_use_sycl_custom` is false and the call lands on
  the same `gemm_vendor_impl<CUDA,float>` that `gemm_cublasdx` already falls through to with
  MathDx absent.
- *Sideways* (`*_vendor_cuda_raw`, ten sites) **cannot** become the public `symm`/`syrk`/
  `syr2k`/`trmm`. Every one is reached *after* a gate that already returned true, so the
  public op re-enters the same gate and recurses without bound — reachable today with
  `BATCHLAS_SYMM_ROUTE=custom` on a CPU queue. It needs a seam that forwards to the vendor
  when compiled and throws `NoRouteError` when not. All three judges called this the single
  most valuable finding in the pass.

## Steps

Each leaves the tree building and carries a check that fails if the step is done wrong.

| # | Step | Route risk |
|---|---|---|
| S0 | Instrument the four level-3 decisions for coverage; correct the two docs | none (pure addition) |
| S1 | A portable vendor-fallback seam, replacing all ten `*_vendor_cuda_raw` calls | none |
| S2 | **Retarget the GEMM terminal at the public `gemm`** | the only perf-visible edit |
| S3 | Fence the cuBLASDx fused tails and CUDA includes behind `#if BATCHLAS_HAS_CUBLAS` | none (dead on this box) |
| S4 | Relax the CMake gate — move names to `BACKEND_COMMON_SOURCES`, move no file | none |
| S5 | Give the facade's `gemm` a native arm, so the retargeted terminal is not a throw | vendor-free only |
| S6 | Move the four route gates out of `cublas.cc` into the facade, so kernels are reachable | none (pure relocation) |
| S7 | Do **not** flip `level3_tile_kernels_compiled`; record why | none |

**S0 first, because it is what makes the rest checkable.** The four ops are invisible to
`route_diff.sh` today (correction 5). Adding `coverage::record_if_enabled` at each dispatcher's
decision point — recording the route actually taken, changing no decision — makes every
subsequent step's acceptance a CSV `diff` over all four scalar types, rather than an eyeball
over a kernel trace. The kernel trace cannot do this job: its `Record` holds a `sycl::event`,
so it cannot see a vendor-to-vendor route change at all.

**S4's non-obvious detail.** The gate cannot be relaxed by deleting the
`if(BATCHLAS_HAS_CUBLAS)`. `BACKEND_CUDA_SOURCES` feeds `batchlas_backends_cuda_obj`, which
`src/CMakeLists.txt:17-19` does not *create* when no CUDA math library is present — so
deleting the `if()` leaves the four TUs uncompiled in exactly the configuration WP1 exists
for. The four names have to move to a different list.

**S7's reason for refusing the flip.** A bare `true` claims kernels that do not exist. The
flag's five consumers (`coverage.cc:152`, `ormqr_blocked.cc:115`, `ortho.cc:179,182`,
`sytrd_blocked.cc:819`) admit types the moved TUs do not serve, because the non-float routing
stayed in `cublas.cc`: `ortho.cc`'s `gram_via_syrk` admits double, whose gram-tile route is
`cublas.cc:766-779`; `ormqr_blocked.cc` admits double trmm, routed at `cublas.cc:1008-1015`;
and `syr2k_triangular_tiles` has exactly **one** call site in the whole tree, at float
(`syr2k_custom_dispatch.cc:173`), so there is no syr2k tile route for double or complex at
all. Flipping the flag would re-introduce precisely the defect S8 removed.

## Known deltas, accepted or deferred

- **Heterogeneous `symm` (S2).** `symm_problem_supported` does not reject a heterogeneous
  batch, unlike its syrk and syr2k counterparts. Today its expanded GEMM reaches the
  strided-batched call on max dims; after S2 it reaches `gemm_heterogeneous_vendor_impl`.
  That is probably a correctness *improvement*, but it is unmeasured and untested — flagged,
  not silently shipped.
- **MathDx-present boxes.** Untestable here (`BATCHLAS_HAS_CUBLASDX 0`,
  `mathdx_DIR-NOTFOUND`). S2 changes their inner-GEMM selection. Stated, not measured — and
  not claimed as verified.
- **`symm` has no `expansion_fits()` ceiling** where hemm/herk/her2k all have one
  (`cublas.cc:318`, `:538`). A real gap, but adding it *is* a route change and belongs in its
  own commit with its own measurement, not bundled into WP1.
- **Two live routing defects, out of scope.** `BATCHLAS_SYRK_ROUTE=native` falls through
  `syrk_custom_dispatch.cc:210/215/220` into raw cuBLAS at `:227`; `BATCHLAS_SYR2K_ROUTE=native`
  throws a cuBLASDx message at `syr2k:181` with no `forced` guard. Both are pre-existing, both
  are "a native request lands on the vendor", and both should be fixed with a test rather than
  in passing.
- **`GesvdShape`-style `RouteTable`s for the level-3 ops** are deliberately *not* built here.
  Adding the table as a pure, unwired addition alongside an equivalence test — modelled on
  `tests/route_gemm_equivalence_tests.cc` — is cheap and safe; *wiring* it is the change that
  moved n=256 onto the wrong kernel, and it needs its own measurement.

## Outcome — all eight steps landed

| step | delivered | evidence |
|---|---|---|
| S0 | the four routes become measurable | 4-suite run: 96 rows for one op → 312 across five |
| S1 | portable vendor seam, 10 call sites | 3016 decisions identical; capture CSVs byte-identical |
| S2 | terminal GEMM → public entry point | routes identical; timings within noise at saturating batch |
| S3 | fused tails leave the dispatchers | `nm -C`: **no** CUDA symbol in any of the four `.o` |
| S4 | TUs leave the CUDA object library | vendor-free links them with **no** CUDA object library present |
| S5 | facade `gemm` gains a native arm | `gemm_tests` 48/184 → 167/184; suite 20/53 → **24/53** |
| S6 | gates move to the facade | tile kernels **reached** vendor-free: 41 native rows, previously 0 |
| S7 | tile predicate gains a scalar parameter | vendor-present unchanged by construction; failing set byte-identical |

Vendor-present reported **3016 distinct routing decisions at every one of the eight steps**.

Three things worth carrying forward:

- **A false win was nearly reported.** `syr2k` at n=1024 looked 10.9% faster after S2.
  Repeating S1 at that shape gave a 5.65–6.40 ms spread; the "win" was noise. The
  flattering direction needs the same scepticism as the alarming one.
- **A pre-existing blocker was found, not caused.** `symm_benchmark`, `syrk_benchmark` and
  `syr2k_benchmark` all abort before printing anything — a SYCL scheduler assertion
  (`adjustNDRangePerKernel: NDR.LocalSize[0] == 0`) that fires on the host backend at tiny
  shapes and, attributed by revert-and-rebuild, fires identically without any WP1 change.
  The level-3 benchmarks are currently unusable; S2 needed a standalone harness.
- **The instrument had to be repaired before it could judge anything.** Three defects in the
  S7 coverage tool surfaced only by *using* it: the gate-declined half was unrecorded,
  `uplo`/`side`/`diag` were not in the key, and `emit()` opened with `"w"` so each of 53 test
  binaries truncated the last. Each looked healthy while reporting almost nothing.

## Acceptance

- Vendor-present: `ctest` unchanged at its documented baseline, and `scripts/route_diff.sh`
  reports **identical** decisions across all four scalar types for
  `symm`/`syrk`/`syr2k`/`trmm`/`gemm` before and after each step.
- Vendor-free: the four TUs compile and link for the first time, and the tile kernels are
  demonstrably *reached* (not merely linked) — proven by coverage rows, not by symbol presence.
  A symbol being present is not evidence it is the one that runs; that mistake has already
  been made once in this project and cost a measurement.
- Any route change is named in advance and measured at saturating batch, never at batch=1.
