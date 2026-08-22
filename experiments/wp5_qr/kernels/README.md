# WP5 — native `geqrf` + `orgqr`: what was built, and how it was verified

Everything here is about the KERNELS. The baseline that sized the problem is in
`../baseline/`; this directory holds the correctness harness, the break sweeps
and their raw output.

Run from the worktree root. GPU pinned with `CUDA_VISIBLE_DEVICES=1` throughout.

---

## 1. What ships

| file | what it is |
|---|---|
| `src/extensions/geqrf_cta_device.hh` | **all** the geqrf device code: a LAPACK-faithful `larfg`, two tile accessors, and ONE `geqr2_panel_device` body |
| `src/extensions/geqrf_cta.cc` | capacities, the fit predicate, the two panel launchers, the one decision site between them, `geqrf_cta_dispatch` |
| `src/extensions/geqrf_blocked.cc` | the right-looking blocked driver: panel → `pack_v` → `larft` → three GEMMs through the injected seam |
| `src/extensions/larft_wy.hh` | `larft` + `pack_v`, factored OUT of `ormqr_blocked.cc` rather than copied a sixth time |
| `src/extensions/orgqr_blocked.cc` | `orgqr` = `ormqr` on an identity, through an injected routed apply |

Route-neutral: `preferred()` is still false for both ops, so a vendor-present
build keeps taking cuSOLVER for every shape. What changed is the **vendor-free**
build, where `route_resolve.hh:60-63` now finds a supported native route instead
of throwing `NoRouteError`.

---

## 2. The harness

`qrcheck.cpp`, built twice (`build_v.sh` → `qrcheck_v` against `build/`,
`build_nv.sh` → `qrcheck_nv` against `build-novendor/`), so "vendor-free" is the
BUILD and not an environment variable inside a build that still links cuSOLVER.

It calls the **direct entry points** (`geqrf_cta_dispatch`,
`geqrf_blocked_dispatch`, `orgqr_blocked_dispatch`), never the facade, because
`route_resolve.hh:101` falls through to `automatic()` when a forced route is
unsupported — so a pinned-route test that is wrong about one gate runs cuSOLVER
and passes green. It ALSO prints what the facade would resolve for the same
shape (`route=`), so the pin is checked rather than trusted.

Columns:

| column | what it detects |
|---|---|
| `route=` | the facade's resolution for this shape and this `BATCHLAS_GEQRF_ROUTE` |
| `leaf=` | which panel leaf the leading panel takes: 1 = local-memory resident, 2 = global |
| `qr=` | ‖QRx − Ax‖∞ / ‖Ax‖∞ with Q applied from the PACKED reflectors |
| `orth=` | ‖QᴴQx − x‖∞ / ‖x‖∞ on the EXPLICIT Q that native `orgqr` produced |
| `qrQ=` | ‖QRx − Ax‖∞ with that explicit Q — catches a Q that is orthonormal but is not this A's Q |
| `dF=`, `dtau=` | ELEMENTWISE max relative difference against the VENDOR's own `geqrf` output |
| `dimag=` | max |Im(R_ii)| / max|R_ii| — R's diagonal must be REAL under the LAPACK convention |

`dF`/`dtau`/`dimag` are the **drop-in** columns and they exist because of a
decision recorded at the top of `geqrf_cta_device.hh`: the native `larfg`
follows LAPACK's REAL-beta convention, not `internal::larfg`'s phase-preserving
one. Break K3 below is what proves those three columns are not decoration.

---

## 3. Control — every cell green, both builds

`run_v.txt` (vendor build) and `run_nv.txt` (vendor-free build). 10 shapes ×
4 types × up to 2 tiers.

Worst value over every row of `run_v.txt`:

| type | qr | orth | qrQ | dF | dtau | dimag |
|---|---|---|---|---|---|---|
| float   | 7.2e-07 | 1.5e-06 | 8.4e-07 | 3.2e-06 | 9.2e-06 | 0 |
| double  | 4.8e-15 | 3.2e-15 | 2.3e-15 | 8.2e-15 | 3.1e-14 | 0 |
| cfloat  | 9.0e-07 | 1.5e-06 | 1.5e-06 | 2.0e-06 | 1.3e-05 | 0 |
| cdouble | 3.9e-15 | 3.2e-15 | 2.5e-15 | 3.8e-15 | 3.1e-14 | 0 |

`dF`/`dtau` at the level of the type's own rounding is the headline: the native
factor is **elementwise the same factorisation cuSOLVER/cuBLAS produces**, tau
included, for all four scalar types. `dimag = 0` exactly, everywhere.

In `build-novendor/` the same rows are green and `route=` reads `native:cta` or
`native:blocked` — the `no route for geqrf<T> ... built without cuBLAS` that four
suites die on is gone.

Shapes, and why each one is in the list:

| shape | why |
|---|---|
| 64×64 | fits the CTA tier for every type; `n % nb == 0` for both widths |
| 65×65 | `n % 32 == 1` and `n % 16 == 1` — the residue `ec1a178` was GREEN at |
| 66×66 | `n % 32 == 2` and `n % 16 == 2` — the residue it FAILED at |
| 100×64 | m > n, so the last panel's reflectors outrun its own width |
| 129×33 | tall, odd, `n % 32 == 1` |
| 200×200 | blocked, several panels, `n % nb == 8` for both widths |
| 256×256 | blocked, `n` an exact multiple of BOTH block widths |
| 300×200 | blocked, m > n, short final panel AND middle panels |
| 512×128 | blocked and tall — the leading panel is too big to be resident for cdouble, so BOTH leaves get exercised in one run |
| 128×128 b=64 | a batch large enough that a per-item stride error cannot hide |

---

## 4. The break sweep

Two kinds. **Reference breaks** (`BREAK=n`, `run_breaks.sh`) damage the
checker's own reference and prove the probes can discriminate. **Kernel breaks**
(`run_kernel_breaks.sh`, patches in the scratch dir) delete or invert the exact
thing a check is supposed to guard, rebuild the CTA device-code cluster (~2 min
each), rerun, and revert. Nothing is left in the tree.

### 4a. Reference breaks — `breaks_ref.txt`

| break | 66×66 (square, real) | 300×200 (m > n) | verdict |
|---|---|---|---|
| 1 — drop the LAST reflector | float/double **GREEN**, complex RED 4.1e-02 | **RED for every type**, 1.8e-01 – 1.9e-01 | discriminates, but only off a square real matrix |
| 2 — reversed reflector order | RED, 1.7 – 2.2 | RED, 1.6 – 1.9 | discriminates |
| 3 — drop Q's last column | RED, orth 0.84 – 0.95 | RED, orth 0.56 – 0.72 | discriminates |
| 4 — conjugate tau | float/double **GREEN**, complex RED 6.0e-01 | float/double **GREEN**, complex RED 2.7e-01 | correct NULL result for real |
| 5 — drop a MIDDLE reflector | RED, 0.51 – 0.82 | RED, 0.53 – 0.66 | discriminates |

**BREAK=1 reproduces the recorded vacuity, on this implementation.** On a square
REAL matrix, deleting the last reflector leaves the residual BIT-IDENTICAL,
because `larfg` returns tau = 0 for a 1×1 real trailing reflector. It turns red
for complex at the same shape, and red for EVERY type at 300×200. That is the
whole argument for having `m > n` shapes in the list, and it is measured here
rather than inherited.

**BREAK=4 is a null result for float and double, and that is correct** —
conjugation is the identity on a real scalar.

### 4b. Kernel breaks — `breaks_kernel.txt`

| break | what it deletes | outcome |
|---|---|---|
| **K1** | `conj(tau)` in the panel apply (zgeqr2 applies Hᴴ) | float/double **GREEN** (correct null — conj is the identity on a real scalar); complex **RED**: qr 6.0e-02 – 4.5e-01, qrQ 8.9e-02 – 4.5e-01, dF 1.6 – 1.8, dtau 0.30 – 1.00 |
| **K2** | `Tᴴ` → `T` in the WY trailing update | `cta` **GREEN** (it has no WY update — correct), `blocked` **RED for every type**: qr 0.49 – 1.29, dF 1.4 – 1.9 |
| **K3** | LAPACK's REAL beta → `internal::larfg`'s phase-preserving complex beta | qr, orth, qrQ **ALL GREEN** for every type. `dimag` **RED** 0.94 – 0.97 and `dF` 1.4 – 1.8, `dtau` 0.19 – 0.71, complex only. |
| **K4** | barrier B2 (between every work-item's read of A(j,j) and work-item 0's write of beta) | **NOTHING TURNED RED.** |
| **K5** | tau's batch stride `k` → the panel's own `ib` | **RED everywhere** — `nan` in qr/orth/qrQ/dF and `dtau` 6.2e+03 – 1.5e+04, at every blocked shape with more than one panel |
| **K6** | W1/W2 batch stride → the current panel's `nb*n2` | **NOTHING TURNED RED.** |
| **K7** | the sub-view over the CALLER's matrix loses its explicit stride and takes the constructor's `ld*cols` default | **RED, catastrophically**: qr 1.3e+25 – 1.9e+91, orth up to 1.2e+184, dF 1.7 – 1.8, at every blocked shape with batch > 1 |

Five of these deserve more than a table row.

**K3 is the most important result in this file.** Switching the complex beta to
the phase-preserving convention produces an equally VALID factorisation — every
residual column stays green to 1e-15 / 1e-6 — and it is NOT the factorisation the
vendor produces. Only `dimag`, `dF` and `dtau` see it. This is the concrete
demonstration of the fifth recorded blind guard in this repository
(`tests/potrf_tests.cc:895-908`, "a residual bound is satisfied by either
implementation"): **a residual test cannot guard a convention, and geqrf's
convention is its contract with ormqr, orgqr, ormbr, sy2sb and band_reduction.**
Any future test for this family that checks only ‖QR − A‖ and ‖QᴴQ − I‖ is
blind to the one property that makes the native kernel a drop-in.

**K4 turned nothing red, and that is reported rather than hidden.** Barrier B2 is
required by the SYCL memory model — `sycl::reduce_over_group` converges control
flow but is not specified to order memory, and every work-item reads A(j,j) as
`alpha` before work-item 0 overwrites it with beta. On this compiler and this
device the two intervening work-group reductions happen to carry a barrier, so
deleting B2 is benign HERE. The barrier stays; the honest statement is that the
harness cannot prove it is needed, and no test in this family should be written
as though it could.

**K6 turned nothing red, and the reason makes it a BAD break rather than a
missing guard.** W1 and W2 are private scratch: the three GEMMs are the only
things that read or write them, they all use the same view, and any batch stride
they AGREE on is arithmetically fine. So there is nothing there for a checker to
catch, and the "explicit stride" discipline the break was aimed at does not
actually bite on scratch. It bites on a sub-view of the CALLER's matrix — which
is what K7 tests instead, and K7 is red by 25 to 91 orders of magnitude. The
lesson is transferable: a stride break is only meaningful on a view whose stride
is fixed by someone else. (W1/W2 still carry an explicit stride, and it still has
to match what `geqrf_blocked_layout` reserved — `n2` SHRINKS as `j0` advances, so
a per-panel stride fits while the allocation was made for the widest one.)

**K5 exposed a defect in the CHECKER, which is why the sweep was run twice.**
The first K5 run printed `qr=4.788e-07` — green — with tau poisoned to −12345 for
most batch items. The probes overflowed to NaN and `std::max(0.0, NaN)` returns
`0.0`, so a NaN residual read as a PERFECT one. `qrcheck.cpp` now uses a
NaN-propagating `nanmax` in all four probes, K5 turns fully red, and the whole
control and both break sweeps were re-run against the fixed checker. **The same
defect is present in `experiments/wp5_qr/baseline/wp5qr.cpp`, which these probes
were adapted from, and in every harness derived from it.**


---

## 5. Performance — a sanity check, not a routing gate

`preferred()` is FALSE for both ops, so nothing here decides anything. This
exists so that a catastrophic number — the batch-only-parallelism defect this
repository keeps rediscovering — cannot go unnoticed.

`run_perf.sh` reuses `../baseline/wp5qr.cpp` UNCHANGED: the same program linked
once against `build/` and once against `build-novendor/`, so "vendor-free" is the
BUILD (`experiments/wp4_potrf/phase2_ab/realpotrf.cpp`'s pattern). It times the
PUBLIC `geqrf`, warms the JIT and clocks inside the harness, and checks the
residual in the same process. Cells are the baseline table's, so the ms column is
directly comparable. Relative sd is under 1% in every cell; raw in `perf.csv`.

vendor `build/` ms → vendor-free `build-novendor/` ms (speedup, >1 = native ahead):

| n, batch | float | double | cfloat | cdouble |
|---|---|---|---|---|
| 64, 8192   | 6.17 → 2.93 (**2.1x**) | 15.76 → 29.62 (0.53x) | 13.14 → 4.96 (**2.6x**) | 31.46 → 58.91 (0.53x) |
| 128, 4096  | 30.96 → 15.38 (**2.0x**) | 60.82 → 37.14 (**1.6x**) | 59.30 → 19.47 (**3.0x**) | 115.98 → 224.0 (0.52x) |
| 256, 2048  | 121.6 → 21.6 (**5.6x**) | 228.8 → 72.2 (**3.2x**) | 227.3 → 44.4 (**5.1x**) | 435.3 → 518.0 (0.84x) |
| 512, 512   | 371.1 → 30.5 (**12.2x**) | 686.1 → 101.3 (**6.8x**) | 561.3 → 63.2 (**8.9x**) | 1111.8 → 789.5 (**1.4x**) |
| 1024, 128  | 2111.8 → 51.3 (**41.1x**) | 4285.2 → 170.0 (**25.2x**) | 3420.5 → 108.9 (**31.4x**) | 5994.2 → 1392.0 (**4.3x**) |

Read the vendor column with the baseline's caveat: `cuBLAS geqrfBatched` is
latency-bound and nowhere near saturated at n >= 512, so its wall time is nearly
independent of batch there. The ms column IS a valid absolute target at each of
these cells; the implied GFLOP/s is not a statement about its ceiling.

**The four losing cells are all FP64 at small n, and the cause is the hardware.**
An RTX 4090 runs FP64 at 1:64 of FP32. The CTA tier at n = 64 is compute-bound in
the reflector reductions, so double comes out ~10x behind float on the same
shape; the vendor does not show that because at these sizes it is launch-bound
rather than compute-bound. Nothing about the `nd_range` is at fault — it is one
work-group per matrix with 32*CT work-items and real intra-matrix width, not the
one-work-item-per-matrix shape of `gebrd.cc:45`. cdouble additionally halves
occupancy again (a 64x64 cdouble tile is 64 KB, one resident block per SM). When
`preferred()` is eventually written, these are the cells it must exclude.

**A WORKSPACE COST WORTH KNOWING BEFORE `preferred()` FLIPS.** The facade takes
`max` over EVERY supported native tier, and `supports()` deliberately puts no
lower extent bound on the Blocked arm (so a forced `blocked` at a small shape
cannot fall through to the vendor). So a caller at n = 64, batch = 8192 pays the
BLOCKED layout even though the route it takes is CTA, whose workspace is 0:
168 MB for float and 671 MB for cdouble, against the vendor's 98 KB. Sizing W1/W2
on the widest trailing block (`n - nb`) rather than on `n` took ~28% off that and
is what ships; the remainder is the `max` policy, which is deliberate
(`WP4_POTRF_SPEC_CORRECTIONS.md`: it turns a query/call disagreement into an
over-allocation instead of an under-allocation). For comparison, the vendor
`orgqr` it replaces asks for 1164 MB / 4644 MB at the same cell.

---

## 6. Reproducing

```bash
bash experiments/wp5_qr/kernels/build_v.sh      # -> qrcheck_v   (build/)
bash experiments/wp5_qr/kernels/build_nv.sh     # -> qrcheck_nv  (build-novendor/)

CUDA_VISIBLE_DEVICES=1 ./experiments/wp5_qr/kernels/qrcheck_v          # control
CUDA_VISIBLE_DEVICES=1 ./experiments/wp5_qr/kernels/qrcheck_nv         # vendor-free control

bash experiments/wp5_qr/kernels/run_breaks.sh                          # reference breaks
bash experiments/wp5_qr/kernels/run_kernel_breaks.sh K1 K2 K3 K4 K5 K6 # kernel breaks
```

`run_kernel_breaks.sh` drives `/home/jonaslacour/.claude/jobs/20812aa0/tmp/kbreak.py`,
which is a scratch file on purpose: the breaks must not be committable.

`DUMPTAU=1` adds per-probe diagnostics (poison count in tau, per-batch-item
numerator and denominator) — the instrument that found the NaN defect.

## 7. Diff hygiene

`.gitignore` here covers the two compiled harnesses and the usual profiler
blobs. Only `.cpp`, `.sh`, `.md` and the text outputs are tracked.
