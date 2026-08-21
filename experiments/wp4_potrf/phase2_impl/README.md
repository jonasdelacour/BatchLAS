# WP4 Phase 2 — correctness evidence for `potrf_blocked.cc`

Everything here is the *implementation* phase's evidence. The *measure* phase
(open questions 5 and 6, the `nb`/`W` sweeps) is in `../phase2_ab/`.

Build: `bash build.sh` (vendor-present, `build/`) and `bash build_nv.sh`
(vendor-free, `build-novendor/`). Both link the harness against the already-built
`.so`s, so it sees exactly the library `ctest` does.

Run: `bash final.sh` — every check on the shipped build. `bash nv.sh` — the same
driver in the vendor-free build with **no route pinned at all**. `bash sizing.sh`
— the ortho-shaped workspace query. `bash pin.sh` — the pin, asked rather than
assumed.

GPU: every script pins `CUDA_VISIBLE_DEVICES=1`. Nothing here is timed, so the
`gpu_guard.sh` discipline (`../phase2_ab/gpu_guard.sh`) does not apply; a
contended card changes no residual.

## What the harness checks, and why each check exists

The parent is allocated at `ld = n + 7`, `stride = ld*n + 13`, i.e. a
leading dimension that is not the row count and a batch stride that is not
`ld*cols`. Every sub-view the driver builds inherits both, and the
`[FIX-B-trap]` failure mode is invisible at `ld == rows`.

| check | the defect it exists for |
|---|---|
| host multiply-back residual on item **0 AND item batch-1** | item 0 sits at offset 0 and cannot move when the batch stride is wrong — the measure phase's own residual was blind to `PHASE2_BREAK=stride` until it was extended past item 0 |
| **the upper triangle, bit for bit** (the whole window is pre-poisoned and the HPD input is written into the lower triangle only) | a lower-triangle residual cannot see the trailing update writing the opposite triangle. `PHASE2_BREAK=nofold` stayed GREEN on all four types in the measure phase and is 11% *cheaper* |
| `info` — planted failures at a known global column | the leaf reports an index LOCAL to its sub-view and writes it UNCONDITIONALLY; the driver must translate and merge |
| **finiteness of a failed item's `A`, with a NaN pivot** | a merely negative pivot divides to a finite number and is not discriminating — see the `noquench` row below |
| `max |imag(diag L)| == 0` exactly | the forced-real diagonal, on a blocked-sized complex case (spec open question 9) |
| `vendorcmp` against `backend::potrf_vendor` | independent agreement with cuSOLVER |

## The pin, asked rather than assumed

An unrecognised `BATCHLAS_POTRF_ROUTE` silently means vendor. The harness's
views are built by the 6-arg `MatrixView` constructor and therefore carry no
pointer array, which cuSOLVER's batched potrf requires (`cublas.cc:1220` →
`matrix.cc:2369`). So the vendor leg *aborts* on them, and that is the pin proof:

```
BATCHLAS_POTRF_ROUTE=blocked           -> PASS
BATCHLAS_POTRF_ROUTE=native:blocked    -> PASS
BATCHLAS_POTRF_ROUTE=cta               -> what(): data_ptrs target is null   (n=256 > 155: supports() rejects CTA, falls back to the vendor)
BATCHLAS_POTRF_ROUTE=vendor            -> what(): data_ptrs target is null
BATCHLAS_POTRF_ROUTE=typo_not_a_route  -> what(): data_ptrs target is null   (the recorded "unrecognised means vendor" trap, demonstrated)
(no env at all)                        -> what(): data_ptrs target is null   (preferred() is all-false: Auto still takes cuSOLVER)
```

The `direct` mode calls `sycl_potrf::potrf_blocked_dispatch` and cannot be served
by a vendor at all.

## Deliberate breaks — WHAT WAS BROKEN AND WHETHER IT TURNED RED

`breaks.sh` / `breaks2.sh` are kept for the record but **do nothing against the
shipped driver**: they drive a `POTRF2_BREAK` switch that was patched into
`src/extensions/potrf_blocked.cc` for one build and then removed. Re-running them
requires re-applying that patch. Results as measured:

| break | what it did | outcome |
|---|---|---|
| `nofold` | diagonal `WxW` product written straight into `A` instead of scratch + fold | **residual stayed GREEN and bit-identical** (4.352e-07 both). Caught only by the upper-triangle check: 15872 words changed (float), 11520 (cdouble). This is the repository's blind-guard shape, reproduced. |
| `stride` | every sub-view built with the child's default stride (`matrix.cc:1839`) | RED everywhere, all four types |
| `conj` | `Transpose::Trans` in the trailing update for every type | RED for cdouble (1.97e-02), correctly a **no-op for float** — `Trans` *is* what real types ship |
| `nozero` | the `info` zero pre-pass removed | RED: `info` comes back as the caller's `-12345` and every item is quenched, residual 9.96e-01 |
| `nomerge` | `info[b] = leaf_info[b]` unconditionally (last-panel-wins) | residual stayed GREEN; the `info` check went RED reporting `(0,0,0,0)` — a successful later panel had erased a real failure. Only an `info` test sees this. |
| `noquench` | failed items not quenched | **GREEN against a negative planted pivot** (a negative pivot divides to a finite number). Went RED only after the harness was extended to plant a **NaN** pivot: `nonfinite=19695`. The honest reading is that the quench earns its keep against NaN/zero pivots and nothing else. |

## Results on the shipped build

`final.sh`, all PASS: facade + `BATCHLAS_POTRF_ROUTE=blocked` at
n ∈ {256, 512, 1000} (1000 exercises the short final block for every type),
`direct` at n=256 batch=128, `vendorcmp` at n=512 batch=32, and `info` at n=300.

Residuals: float 4.4e-07…5.5e-07, double 5.4e-16…1.0e-15, cfloat 2.7e-07…4.1e-07,
cdouble 5.3e-16…7.7e-16. `max |L_vendor - L_blocked| / scale` is at the same
level as either factor's own residual for all four types.

`info` at n=300, nb 128/96/96/64: planted failures at global columns 100 and 200
report **101** and **201** — the 201 case is in the *second* block, so it
exercises the local→global translation; an item with failures planted at both
columns reports **101** (first wins across panels); a NaN pivot reports 101 and
the item stays finite; healthy items report 0 and factor correctly.

Open question 9, `max |imag(diag(A22))|` after one panel by hand (input diagonal
exactly real): cfloat 9.9e-09 against a real part of 6.4e+02 (relative 1.6e-11,
about 1e-4 ulp); cdouble 3.6e-17 against 6.4e+02 (relative 5.7e-20). So the
residue is **non-zero but at rounding level, and it is absorbed, not
accumulated**: the leaf's load transform re-reads the diagonal as
`T(real(A(c,c)), 0)` before any `sqrt`, and `max |imag(diag L)|` is **exactly
0.0** after every full factorisation measured, up to n=1000 (8–16 panels).

## Vendor-free build

`nv.sh`, no env pinned: all four types PASS at n=256 (above every ceiling, so
`Algorithm::Blocked`) and at n=64 (below, so the CTA leaf). Before Phase 2 the
n=256 case threw `NoRouteError`.

`sizing.sh` replicates `ortho.cc:72-78` — the public query over a
**measuring-mode**, unbacked view — and confirms the new max()-over-native-tiers
query stays pure with respect to memory contents. Vendor-present it returns 512 B
for every shape, i.e. unchanged (the route is `{Vendor, Auto}` and `native_need`
stays 0). Vendor-free it returns 2560 B at k=5 and 68–134 KB at n=256; the split
is the `W x W x batch` product buffer, which the layout does not draw at all when
the whole matrix is one block.
