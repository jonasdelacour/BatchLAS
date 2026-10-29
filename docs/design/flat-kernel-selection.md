# Flat kernel selection {#design_flat_selection}

> **Status:** current · checked 2026-10-29

Each op that reads `BATCHLAS_<OP>_ROUTE` (29 ops, one `src/ops/<op>/choice.hh` each) chooses its
kernel in one place: `select::choose()` in `src/select/`, called from the op's own file. A choice is
a value from a `std::variant` of kernel families; `can_run` decides correctness, per-device tables
rank the candidates for speed. Every op's families, keys, grid and shipped tables: the generated
@ref selection_tables page. Tables, ledgers and tiers: @ref design_tuning_data. The tuner engine:
@ref design_tiered_tuning.

## 2. Selection rules

| # | Rule |
|---|---|
| R1 | Three hops, one op file: public function → `choose()` → `std::visit` → launch, all in `src/ops/<op>/<op>.cc`. Only `choice.hh` (the vocabulary) and the shared `src/select/` live elsewhere. |
| R2 | A choice is a value: a `std::variant` of family structs with int fields. One spelling (`lpanel:panel=8`) is used in tables, pins, traces, coverage rows and test names. |
| R3 | `can_run` is correctness only. If it is false, the launch would throw or return a wrong answer. It never encodes "slower", and it agrees exactly with what the launch accepts (tested). |
| R4 | Speed lives only in tables. The last-resort order is by generality, not speed. |
| R5 | Sizing and running call the same `choose()`. `<op>_buffer_size` returns the chosen family's need. |
| R6 | A bad pin throws `batchlas::invalid_argument`: an unknown spelling, a non-candidate, or a choice whose `can_run` is false for the shape. The class words `native` and `vendor` are the exception (§5.2). |
| R7 | Every candidate can be pinned, and a test runs each one on shapes that straddle its limits. |
| R8 | A device with no table borrows another device's table, warns once per op, and tags every trace line with the table it used. |

## 4. Op file layout

### 4.1 Files

`src/ops/<op>/choice.hh` holds the family structs, `candidates<T>()`, `key_names`, the tuner grid
and the `select::OpSpec` (op, vendor library group, last-resort order). `src/ops/<op>/<op>.cc`
holds `key_of`, `can_run`, `launch`, `workspace` and the public entry points.

### 4.3 The op file `<op>.cc`

- The entry point validates its arguments, then calls `select::run`, which picks, opens the trace
  and coverage scope, and launches inside it. `<op>_buffer_size` calls `select::pick`.
- Field-less families derive from `select::NoFields<"name">`; `select::all_of<Choice>()` lists them
  in declaration order, which is also the tie order.
- The vendor arm calls the backend only under `if constexpr (select::has_library<B>(spec.vendor))`;
  otherwise `select::no_vendor` throws `NoRouteError` and records the `miss` row.
- Nested ops decide for themselves: a blocked driver calls the public `gemm`, `trsm`, `potrf`,
  `getrf` or `ormqr`. No parent override forces a child choice.
- A heterogeneous batch has no route: every vendor family refuses it too (known-defects #12). gemm
  splits it into homogeneous items before `select::run`; posv and gesv throw before `choose()`.

## 5. The shared helper `src/select/select.hh`

Everything in `src/select/` (`select.hh`, `select.cc`, `vendor.hh`, `coverage.hh`) is generic over
an op's choice type: device facts, spelling and pins, tables and lookup, trace and coverage.

### 5.1 Device facts

`select::Device`, built by `select::describe` and memoized per (device, backend, `has_vendor`),
carries the table key (`sm_<cc>`, `rocm`, `intel`, `apple_<chip>`, `cpu`), family, arch number,
`is_gpu`, `has_sg32`, `has_vendor`, the local-memory budget, `max_wg`, `max_groups_yz` (65535 on
CUDA) and `scratch_budget`; the field comments in `select.hh` are the reference. Every AMD GPU is
`rocm` (no `gcnArchName` query yet).

### 5.2 Spelling and pins

- **Spelling:** `family[:field=value]...` (`tiny`, `lpanel:panel=8`, `reg:m=128:n=128:k=8:u=1`,
  `vendor`). The positional form `lpanel:8` is accepted; input is case-folded and trimmed.
- **Pin values:** `auto`, `native` (the best runnable non-vendor entry of the nearest row),
  `vendor`, or a spelling. `native` and `vendor` fall back to Auto with a one-time warning when
  nothing in their class can run; any other bad value throws. Retired spellings (`native:cta`,
  `vendor:auto`, `batchlas_cta`, gemm kernel names) throw, and the old `_VARIANT`/`_PROVIDER`
  variables are not read.
- **Tests** pin with `select::ScopedPin` (thread-local, nesting, wins over the environment).
  `ScopedPin(op, word, StrictPin{})`, used by the tuner, makes a class word that cannot serve the
  shape throw instead of falling back.
- The pin is read from the settings snapshot: a raw `setenv` mid-process is not seen until
  `detail::reload_settings()` ([route pins are raw strings](environment.md#environment-route-pins-are-raw-strings)).

### 5.4 Tables and provenance {#flat-selection-tables-and-provenance}

Each table is `tuned/<op>.<device>.txt`, generated from its ledger and embedded at build time, split
into one table per dtype and parsed once per (op, dtype). `BATCHLAS_TUNED_DIR=<dir>` replaces a
built-in file of the same name (trace tag `[override sm_120]`). Format, ledger, tiers and kernel
hashes: @ref design_tuning_data.

```
# op=potrf device=sm_120 batchlas=<sha> kernels=<hash> family_kernels=<family>:<hash>,... date=<date>
# keys: dtype:exact uplo:exact n:log:3 batch:log
dtype=float uplo=L n=64 batch=8192 | lpanel:panel=8 0.2258 | lpanel:panel=16 0.3238 | blocked 0.5222 | vendor 0.4373 | cta 0.5300 # deep
```

- A row lists candidates in the tuner's rank order (survivors by median with 3% ties in candidate
  order, then race-eliminated arms, then carried-over ones), so printed times need not ascend. Times
  are ms per call for the whole batch; the comment is the row's ledger tier.
- An untimed row (`transcribed`, `prior`) prints `-` for each time.
- A kernel-hash mismatch is a warning, never an error ([kernel hashes](tuning-data.md#tuning-data-kernel-hash)).
- Every entry must be a member of `candidates<T>()`, checked on first use and by `tuned_tables_tests`.

Row origins: measured (`deep`, `coarse`, `calibrate`, `preview`, `custom`), `converted` (old
forced-route sweeps; potrf sm_89), `prior` (none shipped) and `transcribed` (the old router's
preference order, untimed). Every op ships sm_89 and sm_120 tables for every dtype it instantiates,
so neither device borrows; per-table provenance is in @ref tuned_tables_readme.

### 5.5 How a call chooses: lookup, borrow order, last resort

`select::choose()` tries, in order:

1. A pin (`ScopedPin` or `BATCHLAS_<OP>_ROUTE`). A bad pin throws.
2. For each table in borrow order, the nearest row; the first entry whose `can_run` is true wins.
3. The first runnable candidate in the op's last-resort order.
4. Otherwise `batchlas::unsupported`; `select::pick` turns that into `NoRouteError` plus a coverage
   `miss` row. A defective built-in table throws `internal_error`, a defective `BATCHLAS_TUNED_DIR`
   table `invalid_argument`.

**Nearest row.** Exact keys must match, dropping exact keys from the right until some row matches.
Among those rows the smallest \f$\sum_i w_i\,|\log_2(\mathrm{row}_i/\mathrm{key}_i)|\f$ over the
`:log` keys wins (`<name>:log:<w>`, default weight 1); ties within 1e-9 go to the smaller log keys.

**Borrowing** (`tables_in_borrow_order`, `src/select/select.cc`): the device's own table; then the
same family, nearest arch at or below, then above; then families `sm`, `gfx`, `intel`, `apple`, any
other, `cpu`. A CPU uses only its own table. An own table whose row has nothing runnable falls
through without the warning (trace tag `fallthrough from`). The first borrowed table warns once per op:

```
batchlas: potrf has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)
```

**Last resort.** `OpSpec::rules.last_resort` lists families in generality order, the rest follow in
list order; default `blocked`, then `vendor` (`kBlockedThenVendor`).

**Trace and coverage.** `BATCHLAS_SELECT_TRACE=1` prints one line per call (key, choice, time,
runner-up or `tied with`, table tag; nested calls indented). Each call writes a coverage `reached`
row (`chosen_algo`, `chosen_origin`) that `scripts/route_diff.sh` diffs. A derived leg (gemm's
aligned or predicated launch) shows only in `BATCHLAS_KERNEL_TRACE`.

## 6. The tuner `tools/tune/batchlas_tune`

The normal workflow is tiered (`--tier preview|coarse|deep|calibrate`): @ref design_tiered_tuning
and @ref tune_tool_readme. The custom two-pass protocol is selected by any of `--reps`, `--warm`,
`--passes`, `--remeasure`, `--refine-ratio`, `--no-refine`, `--no-jit`, `--ld-pad`, `--raw`.

### 6.1 What it times

Every entry of `candidates<T>()`, pinned with a strict `ScopedPin` through the public op, on inputs
built by `tools/tune/<op>_spec.cc`. Each arm is checked with `batchlas::verify`
(@ref design_verification); a failing one is recorded `bad` and dropped from the cell. A candidate
whose `can_run` is false at a cell is recorded as skipped.

### 6.2 Grid and edge refinement

The grid is `choice.hh`'s `grid_*`. Inputs are capped at 4 GiB (`--cap-gib`); tiered runs skip any
dimension above 2048 (`--max-dim`). The custom protocol bisects n at the geometric midpoint where
neighbours disagree until `hi/lo < 1.1` (batch is not refined).

### 6.3 Noise protocol

Custom protocol: a throwaway JIT pass; per cell, candidates interleaved with 1.5 s warm-up each and
16 rotated reps, median taken; the grid run twice, the second in reverse order, mean of the two
medians, re-measured once if they differ by more than 10%.

Both protocols:

- **Tie rule:** within 3% of the best is a tie, and ties go to `candidates<T>()` order. Native
  candidates precede `vendor`, so near-ties go native (costing at most 3%). Example
  (`tune_tests`): blocked 14.10 ms vs vendor 13.74 ms tie, blocked wins.
- **Guard:** one measuring process per GPU, behind a flock; it refuses a GPU holding any foreign
  compute context. `--devices` is required; the launcher is SYCL-free and holds no CUDA context.

### 6.4 Output

Cell records append to `benchmarks/results/tuning/ledger/<op>.<device>.jsonl` (Git LFS);
`scripts/sweep_to_table.py` writes `tuned/<op>.<device>.txt`. `--gate` compares Auto against an old
choice where they differ: FAIL when the new time exceeds `--gate-limit` (1.05) × the old in both
passes; exit 1 on FAIL, 3 when a row errors or no cell was gated.

### 6.5 Staleness check

`sweep_to_table.stale_kernel_warnings`, used by `sweep_to_table.py --check` and
`.github/ci/check_tuned_tables.py`. A wholly untimed table is never stale; a stale table is a
warning. There is no configure-time check.

## 8. Tests

Each op has an `<op>_candidates_tests` suite (20 files, `tests/*_candidates_tests.cc`); the writing
rules are in @ref dev_agent_guide §8.

### 8.1 Every candidate, pinned

Every `candidates<T>()` entry for every dtype, pinned with `ScopedPin`, on shapes straddling its
`can_run` limits both ways, checked with `batchlas::verify`, with non-natural `ld`/stride, complex
data, accepted poison and a >= 1024-item bit-identical batch for local-memory tiers.

### 8.2 `can_run` equals launch

Where `can_run` is true the direct dispatch does not throw; where false it refuses (keeps R3 honest).
`spmm`, `syr2k` and `hermitian` suites have no `CanRunEqualsLaunch` test.

### 8.3 Sizing

Under each pin, a workspace of exactly `<op>_buffer_size(...)` bytes inside a poisoned arena; nothing
is written past the end.

### 8.4 Table validity

`tests/tuned_tables_tests.cc`, no GPU: file names and keys match `choice.hh`, every entry is a
candidate of its dtype, every op ships every dtype on every shipped device, each table holds its
op's grid, and each row's first runnable candidate passes `can_run` at the row's own key.

### 8.5 Pins and borrowing

`tests/select_tests.cc`, no GPU: unknown and `can_run`-false pins throw, every removed alias throws,
and with synthetic tables the §5.5 borrow order, the once-per-op warning, the silent CPU last resort
and the warning-free fallthrough hold.

## 10. Validation and gates

The 2026-10 migration gates passed on the tables of that time: failing names matched base in vendor
and vendor-free trees, off-grid agreement with the old router was 100% for the phase-5 and level-3
ops (gemm not run), and potrf on sm_120 had 0 FAIL over 33 timed changed cells (worst 1.014×).

**Acceptance gate for a changed op:**

1. The op's `*_candidates_tests`, `tuned_tables_tests` and `select_tests` pass in the vendor and
   vendor-free trees; failing names compared with `tests/known-failures.txt`.
2. `scripts/sweep_to_table.py --check` passes.
3. At saturation (batch ≥ 128) the new choice is not more than 1.05× slower than the old at off-grid
   points, reproduced with the order reversed (`batchlas_tune --gate`).
4. A reviewer finds the kernel for a call by reading `<op>.cc` and its table.

## 11. Open items

- **Retune.** 31 of the 59 shipped tables are wholly transcribed (most sm_89 tables, `spmm.cpu.txt`,
  and on sm_120 hemm, herk, her2k, potri, potrs, sygst, sygvd, sysv, sytrf, sytrs); the sm_89 potrf
  rows are converted. They encode the old router's choices, untimed.
- **sm_89 live gate** for potrf and the transcribed sm_89 tables.
- **potrf.** The blocked leaf is hard-wired to `potrf_cta_dispatch`; `nb` and `W` come from
  `PotrfBlockedConst` (`BATCHLAS_POTRF_NB`, `BATCHLAS_POTRF_W`), not fields.
- **Grid-z ceiling.** gemm and the level-3 natives chunk a batch past 65535
  (`src/ops/level3/batched.hh`, gemm's `launch`). Other ops that put the batch in grid z were not
  audited; ormqr's report (known-defects #16) did not reproduce on sm_120.
- **`select::level3_tile_substitution_enabled`** is conservative (a native GPU backend, and float or
  CUDA with cuBLAS; `src/select/vendor.hh`). Widening it moves the vendor-free routes of ortho and
  ormqr, so it is a separate change.

## 12. Per-op families

Families, keys, grids and ranked-first ranges per table are on @ref selection_tables; the predicate
in `<op>.cc` is the reference for `can_run`. This section keeps what neither says.

### potrf

`lpanel` and `blocked` are Lower only. Upper calls use the Lower rows (the exact key is dropped)
unless the table has Upper rows; sm_120 has them. The converted sm_89 table has no current `lpanel`
timings, so it never picks `lpanel`.

### posv {#phase-3-posv}

`tiny` is the fused factor-and-solve kernel; `cta` is the public `potrf` plus one fused solve
(`ops::potrs::solve`), `blocked` the public `potrf` plus two public trsm. The children choose for
themselves, so the tuner's `cta` and `blocked` times include them. No vendor family: a `vendor` pin
warns and runs Auto. Empty problems and heterogeneous batches throw `internal_error` before
`choose()`. The names `cta` and `blocked` are misnomers kept so benchviz and the scripts keep working.

### trsm {#phase-3-trsm}

`cta` and `sg_left` (Left only) are register-resident solvers for order ≤ 32; `blocked` solves
32-wide diagonal blocks with `cta` and updates with the public gemm. `uplo` and `diag` are not keys:
an sm_120 A/B over 56 cells changed no first choice. `A.batch_size() != B.batch_size()` throws
`invalid_argument` in `trsm_validate_params`. Batch offsets in the complex vendor substitute are 64-bit.

### gemm {#phase-3-gemm}

Families, forms and windows: [GEMM: what ships](../perf/gemm.md#gemm-what-ships). `layout=packed`
means A, B and C contiguous with 16-byte-aligned bases. A heterogeneous batch is split into
homogeneous items before selection. Natives need `is_gpu || !has_vendor` (a CPU with a host BLAS
keeps the vendor) and a precision `native_serves_precision` accepts. The aligned or predicated leg,
the transpose instantiation, the `small` bucket and TR/TC/stages are derived in the launcher, never a
field or key. A pin on a form the configuration is not instantiated for is `can_run`-false and throws.
On sm_89 (transcribed) nearest-row lookup differs from the old router on some off-grid shapes;
accepted until the sm_89 retune.

### posv, trsm and gemm grids {#phase-3-grids}

The grids in `choice.hh` are spelled again by the transcriber (`tuned/README.md`). trsm keys only
side × trans (N/T), because the full side × uplo × trans × diag lattice was infeasible to time. gemm
declares its grid per (ta, tb, layout): squares for every form and both layouts; panels and skinny
shapes for the issued forms only, `packed` panels from m, n ≥ 128; large-k Gram and skinny shapes
(not transcribed for sm_89). Real types use all four (ta, tb) pairs, complex types the issued ones.

### gemv {#phase-5-gemv}

`cta` (bodies 3 and 5) needs a GPU, sub-group 32 and `trans != N`; `direct` (bodies 1, 2, 4) has no
GPU gate. The body and segment width are derived (`BATCHLAS_GEMV_SEGT` steers the width). `out` and
`red` are the lengths of y and x and swap with `trans`.

### geqrf {#phase-5-geqrf}

`can_run` lives in `src/ops/geqrf/can_run.hh` so a host test can call it with a synthetic device.
`form` is sq, tall or wide; `aspect` is max/min by integer division. Callers that size once for a
bounding panel use `geqrf_buffer_size_bound`, the maximum over every family the device can run.

### gesv {#phase-5-gesv}

`tiny` is the fused LU factor and solve; `blocked` is the public `getrf` then the public `getrs`
(workspace `getrf_buffer_size + getrs_buffer_size`). No vendor family.

### getrf {#phase-5-getrf}

Natives need a square A; the CUDA vendor does too (cuBLAS `getrfBatched` is square-only). The panel
leaf, laswp mode, blocking factor and tiny bucket are derived in the driver.

### getri {#phase-5-getri}

`blocked` writes P into C and makes two public trsm calls; last resort is `vendor`, then `blocked`.

### getrs {#phase-5-getrs}

Natives need a conforming pair (A square, `B.rows == n`, equal batch); a non-conforming pair goes to
the vendor. `transA` is not a key.

### orgqr {#phase-5-orgqr}

`blocked` is the identity fill plus the public `ormqr` and needs `n <= m`; the vendor is a per-item
loop and runs n > m and CPU queues. No batch key. Sizing goes through `choose()`, which fixed
[the orgqr_buffer_size latent defect](../perf/qr.md#the-orgqr_buffer_size-latent-defect).

### ormqr {#phase-5-ormqr}

The WY width is a positive `block_size_hint` clamped to [1, k], else
`tuning::ormqr_block_size_for_n`. Complex `Transpose::Trans` throws before `choose()`. `blocked` has
no batch term (known-defects #16 did not reproduce on sm_120).

### spmm {#phase-5-spmm}

`direct` (CSR gather for NoTrans A, scatter otherwise) has no GPU gate: the NETLIB queue relies on
it. The vendor refuses complex `transB == ConjTrans` with nrhs 1 on CUDA (cuSPARSE leaves C unwritten)
and any non-N transpose on NETLIB. No nnz key (nnz is in device memory); a CPU never borrows, so
`spmm.cpu.txt` ships. cuSPARSE is not bit-reproducible, so tests identify a pinned vendor by its trace.

### syev {#phase-5-syev}

`cta`, `cta_fused` and `jacobi` are the n ≤ 32 sub-group solvers (`cta` and `cta_fused` not on
ROCm); `blocked` and `two_stage` take Upper by mirroring into Lower, so `uplo` is not a key. A
non-square A throws `invalid_argument`. ROCm has no syev table and borrows the sm tables (untested).

### gesvd

The jobs are canonicalised once (Thin to All where they coincide). cuSOLVER refuses
`max(m, n) > 32`, non-packed batches and thin factors, so Auto reaches the vendor for those only where
no native family runs. Known defects #14, #15 and #18.

### 12.9 Level-3 (symm, syrk, syr2k, trmm, hemm, herk, her2k) {#level-3-four-symm-syrk-syr2k-trmm}

`NoFields` families. The vendor family calls `backend::<op>_vendor` under
`has_library<B>(spec.vendor)`, not the public entry, so there is no recursion. Shared pieces are in
`src/ops/level3/batched.hh`:

- Every native launch chunks the batch into sub-views of at most `Device::max_groups_yz` items, so
  `can_run` has no batch term.
- The scratch families (`expand`, `fold`) also chunk to the scratch budget (`expansion_chunk`,
  `BATCHLAS_EXPAND_MAX_BYTES`) and refuse only a shape whose single item does not fit.
- rocBLAS wraps no hemm, herk or her2k, so their vendor family refuses ROCm.
- symm, syrk and syr2k take complex `T` as the complex-symmetric (unconjugated) update; complex<double>
  `triangular` is not built (its accumulators fill the register file).
- `src/ops/her2k/route.hh` holds her2k's `can_run` so `sytrd_blocked` asks the same selection
  (`picks_batched_route`).

## 13. Design decisions

| Topic | Chosen | Rejected |
|---|---|---|
| Source of speed | Measured or transcribed tables per (op, device), dtype a key | Hand-written device rules; first-call autotuning; a fitted cost model |
| Choice type | `std::variant` of family structs with int fields | One type per config; a flat struct; an enum per kernel; virtual kernel objects |
| Nested ops | Each op decides through the public function | A parent that plans everything |
| Unmeasured shapes | Nearest row by weighted log distance | Generated regions; power-of-two buckets |
| Storage | One text format, embedded at build time, `BATCHLAS_TUNED_DIR` overrides | Compiled C++ tables; runtime files only |
| Untuned device | Borrow the same family's nearest arch, then other families | Per-op fact rules; a conservative vendor default |
| Ties | Within 3% is a tie, ties go to candidate-list order | Keep the previous winner unless 5% better |
| Staleness | Hash in the header, warning only | Rejecting stale tables |
| Workspace | Sized through the same `choose()` | An explicit plan object; the max over candidates |
| Candidate list | A `constexpr` list in `choice.hh`, also the compiled set | Per-family ranges; a separate tuner config |
| CPU backend | Tuned like any device (`*.cpu.txt`) | Borrowing from a GPU |
