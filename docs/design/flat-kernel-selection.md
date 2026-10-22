# Flat kernel selection {#design_flat_selection}

> **Status:** current · checked 2026-10-21

Each op that reads `BATCHLAS_<OP>_ROUTE` (19 ops) chooses its kernel in one place: `select::choose()`
in `src/select/`, called from the op's own file in `src/ops/<op>/`. A choice is a value from a
`std::variant` of kernel families. `can_run` decides correctness; per-device tables rank the
candidates for speed. Tables, ledgers and tiers: @ref design_tuning_data. The tuner engine:
@ref design_tiered_tuning. The phase 3 plan (posv, trsm, gemm): @ref design_flat_selection_phase3_plan.

## 2. Selection rules

| # | Rule |
|---|---|
| R1 | Three hops, one op file: public function → `choose()` → `std::visit` → launch, all in `src/ops/<op>/<op>.cc`. Only `choice.hh` (the vocabulary) and the shared `src/select/` live elsewhere. |
| R2 | A choice is a value: a `std::variant` of family structs with int fields. One spelling (`lpanel:panel=8`) is used in tables, pins, traces, coverage rows and test names. |
| R3 | `can_run` is correctness only. If it is false, the launch would throw or return a wrong answer. It never encodes "slower", and it agrees exactly with what the launch accepts (tested). |
| R4 | Speed lives only in tables. The last-resort order is by generality, not speed. |
| R5 | Sizing and running call the same `choose()`. `<op>_buffer_size` returns the chosen family's need. |
| R6 | A bad pin throws `batchlas::invalid_argument` (a `std::invalid_argument`): an unknown spelling, a non-candidate, or a choice whose `can_run` is false for the shape. The class words `native` and `vendor` are the exception (§5.2). |
| R7 | Every candidate can be pinned, and a test runs each one on shapes that straddle its limits. |
| R8 | A device with no table borrows another device's table, warns once per op, and tags every trace line with the table it used. |

## 4. Op file layout

### 4.1 Files

| File | Holds |
|---|---|
| `src/ops/<op>/choice.hh` | family structs, `candidates<T>()`, `key_names`, the tuner grid, the `select::OpSpec` (op, vendor library group `select::Lib`, last-resort order) |
| `src/ops/<op>/<op>.cc` | `key_of`, `can_run`, `launch`, `workspace`, the public entry points |

### 4.3 The op file `<op>.cc`

- The public entry point validates its arguments, then calls `select::run`, which picks, opens the
  trace and coverage scope, and launches inside it. `<op>_buffer_size` calls `select::pick` and
  returns the chosen family's need.
- Field-less families derive from `select::NoFields<"name">`. For an op with no fields,
  `select::all_of<Choice>()` lists the candidates in declaration order, which is also the tie order.
- The vendor arm calls the backend only under `if constexpr (select::has_library<B>(spec.vendor))`;
  otherwise `select::no_vendor` throws `NoRouteError` and records the `miss` row.
- Nested ops decide for themselves: a blocked driver calls the public `gemm`, `trsm`, `potrf`,
  `getrf` or `ormqr`. No parent override forces a child choice; one is built only if a measurement
  needs it.
- Common native `can_run` terms: a GPU (`is_gpu`), sub-group 32 (`has_sg32`) where the kernel needs
  it, a homogeneous batch, extents and batch of at least 1, and `batch <= max_groups_yz` for kernels
  that put the batch in grid z.
- A heterogeneous batch has no route: every vendor family refuses it too, because the vendor loops
  run every item at the top-level extents (known-defects #12). gemm splits it into homogeneous items
  before `select::run`; posv and gesv throw before `choose()`.

## 5. The shared helper `src/select/select.hh`

Everything in `src/select/` is generic over an op's choice type: device facts, spelling and pins,
tables and lookup, trace and coverage (`select.hh`, `select.cc`, `vendor.hh`, `coverage.hh`).

### 5.1 Device facts

`select::Device` (`src/select/select.hh`) is built by `select::describe` and memoized per (device,
backend, `has_vendor`):

| Field | Meaning |
|---|---|
| `key` | table key: `sm_<cc>` (CUDA compute capability), `rocm`, `intel`, `apple_<chip>` (Metal, from the device name: `apple_m5_max`), `cpu` |
| `family` | `sm`, `gfx` (`rocm` included), `intel`, `apple`, `cpu` |
| `arch_number` | 120, 89, 90 (gfx90a), 5 (apple_m5_max), 0 for cpu |
| `is_gpu`, `has_sg32` | device type; sub-group size 32 supported |
| `has_vendor` | the op's library group (`OpSpec::vendor`) is compiled in for this backend |
| `slm_budget`, `slm_per_item`, `slm_budget_for(wg)` | local memory less a 4 KiB reserve; Metal adds 8 B per work-item |
| `max_wg` | `MAX_WORK_GROUP_SIZE` |
| `max_groups_yz` | work-groups in SYCL dim 0/1 of a 2-D/3-D launch: 65535 (CUDA grid z/y), \f$2^{32}-1\f$ on Metal |
| `scratch_budget` | one call's scratch lease: min(`MAX_MEM_ALLOC_SIZE`, `GLOBAL_MEM_SIZE` / 4) |

ROCm has no `gcnArchName` query yet, so every AMD GPU is `rocm` and borrows like an unknown `gfx`
(`device_from_key`, `src/select/select.cc`).

### 5.2 Spelling and pins

- **Spelling:** `family[:field=value]...`, for example `tiny`, `lpanel:panel=8`,
  `reg:m=128:n=128:k=8:u=1`, `vendor`. The positional form `lpanel:8` is accepted as input. Input
  is case-folded and trimmed.
- **Pin values:** `auto`, `native` (the best runnable non-vendor entry of the nearest row), `vendor`,
  or a spelling. `native` and `vendor` fall back to Auto with a one-time warning when nothing in
  their class can run. Any other value that does not parse, is not a compiled candidate, or that
  `can_run` refuses, throws.
- **Removed spellings throw:** `native:cta`, `vendor:auto`, `batchlas_cta`, and gemm kernel names
  such as `tiled16` or `128x128x8`. `BATCHLAS_<OP>_VARIANT`, `BATCHLAS_<OP>_PROVIDER` and
  `BATCHLAS_GEMM_SYCL_KERNEL` are not read.
- **Tests** pin with `select::ScopedPin`, a thread-local, nesting slot that wins over the
  environment. `ScopedPin(op, word, StrictPin{})` (the tuner's arms) makes a class word that cannot
  serve the shape throw instead of falling back.
- **Snapshot:** the pin is read from the settings snapshot. A raw `setenv` mid-process is not seen
  until `detail::reload_settings()` runs (`ScopedEnvVar` does this). See
  [route pins are raw strings](environment.md#environment-route-pins-are-raw-strings).

### 5.4 Tables and provenance {#flat-selection-tables-and-provenance}

Each table file is `tuned/<op>.<device>.txt`, generated from its ledger and embedded at build time;
the loader splits it into one table per dtype, parsed once per (op, dtype). `BATCHLAS_TUNED_DIR=<dir>`
replaces a built-in file with a file of the same name (the trace then shows `[override sm_120]`).
Format, ledger, tiers, kernel hashes and the check: @ref design_tuning_data.

```
# op=potrf device=sm_120 batchlas=<sha> kernels=<hash> family_kernels=<family>:<hash>,... date=<date>
# source=ledger:benchmarks/results/tuning/ledger/potrf.sm_120.jsonl tiers=deep:1374,...
# keys: dtype:exact uplo:exact n:log:3 batch:log
dtype=float uplo=L n=64 batch=8192 | lpanel:panel=8 0.2258 | lpanel:panel=16 0.3238 | blocked 0.5222 | vendor 0.4373 | cta 0.5300 # deep
dtype=float uplo=L n=512 batch=2048 | vendor 12.68 | blocked 13.76 | lpanel:panel=8 14.62 # deep
```

- A row lists the candidates in the tuner's rank order: survivors by median, ties within 3% of the
  best in candidate-list order; then arms eliminated in the race, by median; then candidates carried
  over from an earlier record (`record_from_arms`, `tools/tune/schedule.cc`). Printed times
  therefore need not ascend (row 1 above). Times are milliseconds per call for the whole batch; the
  comment is the row's ledger tier.
- An untimed row (`transcribed`, `prior`) has the form
  `dtype=float uplo=L n=8 nrhs=1 batch=128 | tiny - | cta - # transcribed`; the header then needs a
  `source=ledger:` (or, in a hand-made single table, `source=transcribed:<hex sha>`).
- `kernels=` and `family_kernels=` are kernel hash v2 over the sources each spec lists between its
  `kernel-sources-begin` and `kernel-sources-end` markers
  ([kernel hashes](tuning-data.md#tuning-data-kernel-hash)). A mismatch is a warning, never an
  error; the table stays in use.
- Every table entry must be a member of `candidates<T>()`. The loader checks this on first use, and
  `tuned_tables_tests` checks every shipped table without a GPU.

Row origins (ledger tiers, best first `deep`, `coarse`, `calibrate`, `preview`, `custom`,
`converted`, `prior`, `transcribed`; @ref design_tuning_data):

| Origin | Meaning | Shipped where (headers of `tuned/*.txt`, 2026-10-21) |
|---|---|---|
| measured (`deep`, `coarse`, `calibrate`, `preview`, `custom`) | `batchlas_tune` timings | every sm_120 table (deep; gemm float and cfloat re-measured 2026-10-21) |
| converted | timings from earlier forced-route sweeps, on old kernels | potrf on sm_89 (241 rows) |
| prior | a calibrate run's untimed copy of a borrowed table's nearest row | none shipped |
| transcribed | the old router's preference order at each grid cell, untimed | every other sm_89 table, `spmm.cpu.txt`, sm_120 gemm double and cdouble, and sm_120 rows the deep lattice measured nothing near |

Every op ships a table for every dtype it instantiates on sm_89 and sm_120 (39 files; inventory in
@ref tuned_tables_readme), so neither device borrows. Retuning sm_89 is open (§11).

### 5.5 How a call chooses: lookup, borrow order, last resort

`select::choose()` tries, in order:

1. A pin, from `select::ScopedPin` or `BATCHLAS_<OP>_ROUTE`. A bad pin throws.
2. For each table in borrow order, the nearest row; the first entry whose `can_run` is true wins.
3. The first runnable candidate in the op's last-resort order.
4. Otherwise it throws `batchlas::unsupported` (a `std::runtime_error`); `select::pick` turns that
   into `NoRouteError` plus a coverage `miss` row when the op's vendor library is absent and no
   candidate can run. A defective built-in table throws `batchlas::internal_error`; a defective
   `BATCHLAS_TUNED_DIR` table throws `batchlas::invalid_argument`.

**Nearest row.** Exact keys (`name:exact`) must match, dropping exact keys from the right until some
row matches. Among those rows the smallest weighted distance
\f$\sum_i w_i\,|\log_2(\mathrm{row}_i/\mathrm{key}_i)|\f$ over the `:log` keys wins; the weight comes
from `<name>:log:<w>` (default 1). Ties within 1e-9 go to the row with the smaller log keys, in
`# keys:` order. gemm memoizes the scan per key.

**Borrowing (§5.5).** For a device key `d` (`tables_in_borrow_order`, `src/select/select.cc`):

1. the table for `d`;
2. other tables of the same family, nearest arch number at or below `d`, then nearest above;
3. other families in the order `sm`, `gfx`, `intel`, `apple`, any other, `cpu`.

A CPU device uses only its own table; without one it goes to the last resort silently. A device
with its own table whose row has nothing runnable falls through to the next table without the borrow
warning (trace tag `fallthrough from`). The first borrowed table prints one warning per op per process:

```
batchlas: potrf has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)
```

**Last resort.** `OpSpec::rules.last_resort` lists families in generality order; the remaining
candidates follow in list order. Default `blocked`, then `vendor` (`kBlockedThenVendor`).

**Trace and coverage.**

- `BATCHLAS_SELECT_TRACE=1` prints one line per call on stderr: key fields, the chosen spelling, its
  time, the runner-up (or `tied with` within 3%), and the table tag. Nested calls are indented.
- Each call writes a coverage `reached` row. `chosen_algo` holds the spelling and `chosen_origin`
  holds `native` or `vendor`. `scripts/route_diff.sh` diffs these rows.
- `BATCHLAS_KERNEL_TRACE` is separate. Use it to see a derived leg (gemm's aligned or predicated
  launch), which the coverage row does not record.

```
potrf float n=64 batch=8192 -> lpanel:panel=8  0.226 ms, next lpanel:panel=16 0.324 ms  [sm_120]
potrf float n=512 batch=2048 -> vendor  12.68 ms, next blocked 13.76 ms  [sm_120]
```

## 6. The tuner `tools/tune/batchlas_tune`

The normal workflow is tiered (`--tier preview|coarse|deep|calibrate`, the lean per-cell protocol):
@ref design_tiered_tuning and @ref tune_tool_readme. §6.1-§6.3 below are the spec and the custom
(two-pass) protocol, selected by any of `--reps`, `--warm`, `--passes`, `--remeasure`,
`--refine-ratio`, `--no-refine`, `--no-jit`, `--ld-pad`, `--raw`.

```
batchlas_tune <op> --tier deep --dtype float,cfloat --devices <gpu> --out tuned/
batchlas_tune <op> --dtype float --devices <gpu> --raw /tmp/raw          # custom protocol
```

### 6.1 What it times

Every entry of `candidates<T>()`, pinned with a strict `ScopedPin` and run through the public op.

- **Inputs:** built per op by `tools/tune/<op>_spec.cc` (SPD matrices for potrf).
- **Verification:** `batchlas::verify` on each arm (@ref design_verification). A failing candidate
  is recorded `bad` and dropped from that cell.
- **Skips:** a candidate whose `can_run` is false at a cell is recorded as skipped, not timed.

### 6.2 Grid and edge refinement

- A coarse grid from `choice.hh` (`grid_*`). For potrf: 34 values of n from 1 to 1280, batch 128,
  512, 2048, 8192, 32768, per `uplo`. Inputs are capped at 4 GiB (`--cap-gib`); tiered runs also
  skip any matrix dimension above 2048 (`--max-dim`).
- Custom protocol: where neighbouring n points have different winners, n is bisected at the
  geometric midpoint until `hi/lo < 1.1`. Batch is not refined. Refined points become ordinary rows.
  Tiered refinement: @ref design_tiered_tuning.

### 6.3 Noise protocol

Custom protocol:

1. A throwaway JIT pass over every cell.
2. Per cell, the candidates are interleaved with a 1.5 s warm-up each, then 16 timed reps with the
   order rotated each rep. The median is taken.
3. The grid is run twice, the second time in reverse candidate order. A cell's time is the mean of
   its two pass medians. If the two medians differ by more than 10%, the cell is measured once more.

Both protocols:

4. **Tie rule:** every candidate within 3% of the best is tied, and ties go to the `candidates<T>()`
   order. Native candidates precede `vendor`, so near-ties go to the native kernel; this costs at most
   3%. Example (`tune_tests`): blocked 14.10 ms vs vendor 13.74 ms tie, blocked wins. Moving
   `Vendor{}` to the front reverses it.
5. **Guard:** one measuring process per GPU, behind a flock. It refuses to run while a foreign
   compute process is on the GPU or utilisation is above 5% (`--allow-idle-foreign` tolerates idle
   contexts). `--devices` is required. The launcher is SYCL-free and holds no CUDA context.

### 6.4 Output

Cell records are appended to the ledger `benchmarks/results/tuning/ledger/<op>.<device>.jsonl`
(Git LFS), from which `scripts/sweep_to_table.py` writes `tuned/<op>.<device>.txt` in the §5.4
format (@ref design_tuning_data). `--gate` compares Auto against an old choice on the cells where
they differ: FAIL when the new time is above `--gate-limit` (1.05) × the old in both passes. Exit
code 1 on FAIL, 3 when a row is an error or no cell was gated.

### 6.5 Staleness check

One function, `sweep_to_table.stale_kernel_warnings`, used by `scripts/sweep_to_table.py --check`
and `.github/ci/check_tuned_tables.py`: see [kernel hashes](tuning-data.md#tuning-data-kernel-hash).
A table whose rows are all converted, prior or transcribed is never stale. A stale table is a
warning, never an error. There is no configure-time check.

## 8. Tests

Each op has an `<op>_candidates_tests` suite (19, `tests/*_candidates_tests.cc`) built from the
parts below. The rules for writing them are in @ref dev_agent_guide.

### 8.1 Every candidate, pinned

Loop over `candidates<T>()` for every dtype, pin each with `ScopedPin`, run it on shapes that
straddle its `can_run` limits in both directions, and check the result, info and the untouched
triangle with `batchlas::verify`. Required inputs: non-natural `ld` and stride, complex data with a
nonzero imaginary part, poison the kernel will accept, and a saturating batch (at least 1024
identical items, bit-identical results) for shared-local-memory tiers.

### 8.2 `can_run` equals launch

For every candidate and straddling shape: where `can_run` is true the direct dispatch call does
not throw, and where it is false the direct call refuses. This keeps R3 honest. 17 of the 19 suites
have a `CanRunEqualsLaunch` test; `spmm_candidates_tests` and `syr2k_candidates_tests` do not.

### 8.3 Sizing

Under each pin, run with a workspace of exactly `<op>_buffer_size(...)` bytes inside a poisoned
arena and check that nothing is written past the end.

### 8.4 Table validity

`tests/tuned_tables_tests.cc`, no GPU. Every embedded file is `<op>.<device>.txt` keyed
`dtype:exact` plus the op's `choice.hh` keys; every per-dtype table parses and names only members of
that dtype's `candidates<T>()`; every op ships every dtype on every shipped device; each table holds
exactly its op's grid. Timed rows of a non-ledger table must ascend apart from ties (every shipped
table is a ledger table, whose order is the race's). On the matching device, each row's first
runnable candidate also passes `can_run` at that row's own key.

### 8.5 Pins and borrowing

No GPU (`tests/select_tests.cc`). An unknown pin throws, and so does a `can_run`-false pin. Every
old alias throws (each op's should-throw test lists the removed spellings). With synthetic tables,
the borrow order of §5.5 holds (Apple included), the warning prints once per op, a CPU without its
own table takes the last resort silently, and an own table with nothing runnable falls through
without the warning.

## 10. Validation and gates

Results of the migration gates (2026-10, on the tables of that time; not repeated on the current
tables):

| Check | Result |
|---|---|
| Failing test names, vendor and vendor-free trees | Matched the base for each migrated op. Each deliberate break of an op file turned a narrow, named set red. |
| Off-grid data gate (random points, old router vs nearest row plus `can_run`) | 100% agreement for the phase-5 ops and the level-3 four, on sm_89 or sm_120. Dropping threshold rows fails the gate. Not run for gemm. |
| Coverage of Auto calls, old vs new build | Same `reached` row as the old build, except the deliberate changes listed under each op. |
| Live timing, potrf on sm_120 | 120 off-grid cells: 54 chose the same kernel. 33 changed and were timed in two reversed passes: 0 FAIL, worst 1.014× (cdouble n=44, batch 8192), the others 0.60 to 0.994×. The remaining 33 were not timed: 32 exceed 12 GiB, and \f$n^2\cdot\mathrm{batch} \ge 2^{31}\f$ aborts in `factor_bench` on any route. |
| Live timing, potrf on sm_89 | Open. The sm_89 table is converted and has no current `lpanel` timings. |

**Acceptance gate for a changed op:**

1. Correctness: the op's `*_candidates_tests`, `tuned_tables_tests` and `select_tests` pass in the
   vendor and vendor-free trees. Failing names are compared with `tests/known-failures.txt`.
2. `scripts/sweep_to_table.py --check` passes.
3. Live, at saturation (batch ≥ 128): the new choice is not more than 1.05× slower than the old
   routing at off-grid points between grid values, interleaved in one process; a loss counts only if
   it reproduces with the order reversed (`batchlas_tune --gate`).
4. Readable: a reviewer finds the kernel for a given call by reading `<op>.cc` and its table.

## 11. Open items

- **Retune sm_89.** Every sm_89 table is transcribed (potrf: converted), as are sm_120 gemm double
  and cdouble (FP64 is not tuned on 1:64 parts), `spmm.cpu.txt`, and sm_120 rows the deep lattice
  measured nothing near. Transcribed rows encode the old router's choices, untimed.
- **sm_89 live gate** for potrf and for every op with a transcribed sm_89 table.
- **hemm, herk and her2k** select by hand in `src/backends/cublas.cc` (vendor builds only). They
  expand or fold into the public gemm when `expansion_preferred`, `herk_gemm_preferred`
  (batch ≥ 4 and n ≤ 768, `cublas.cc:265-271`) or `her2k_gemm_preferred` (batch ≥ 2 or n ≥ 128,
  `src/expansion_budget.hh:74-78`) hold and the expansion fits, and otherwise use the per-item vendor
  loop. They honour `BATCHLAS_EXPAND_ROUTE=expand|loop`; herk's gram opt-in is
  `BATCHLAS_SYRK_ROUTE=gram`. They have no `BATCHLAS_<OP>_ROUTE` and throw `NoRouteError` in a
  vendor-free build (including Metal, `tests/known-failures-acpp-metal.txt`). Planned families: hemm
  `{expand, vendor}`, herk `{gram, fold, vendor}`, her2k `{fold, vendor}`.
- **potrf.** The blocked leaf is hard-wired to `potrf_cta_dispatch`
  (`src/ops/potrf/kernels/potrf_blocked.cc:306`). `Blocked` has no fields: `nb` and `W` come from
  `PotrfBlockedConst`, overridable by `BATCHLAS_POTRF_NB` and `BATCHLAS_POTRF_W`. `lpanel:panel=16`
  is timed only in the sm_120 table (165 of the 189 float `uplo=L` rows, 21 ranking it first).
  Upper above the `cta` ceiling has no vendor-free route (known-defects #19).
- **Grid-z ceiling.** gemm and the level-3 four check `batch <= max_groups_yz`. Other ops that put
  the batch in dimension 0 of a 3-D range were not audited. ormqr's blocked driver was reported to
  hit it (known-defects #16), but that did not reproduce on sm_120.
- **`select::level3_tile_substitution_enabled`** is conservative (a native GPU backend, and float or
  CUDA with cuBLAS; `src/select/vendor.hh:104-105`). Widening it moves the vendor-free routes of
  ortho and ormqr, so that is a separate change.

Closed since 2026-10-06: gemm `direct` rebases each item in 64 bits
(`src/ops/gemm/kernels/gemm_kernels.cc:309-312`); the stale `options.hh` line numbers in the potrf
header comment are gone.

Known defects that affect routing are in [known defects](known-defects.md): #12 (heterogeneous
batches on the vendor, closed by refusing them in every family's `can_run`) and #13 (the cuBLASLt
segfault on complex double gemm, fixed with 16-byte aligned scalars).

## 12. Per-op families

Candidates and keys come from the op's `choice.hh`; `can_run` from `<op>.cc`. "Transcribed" patterns
and row counts describe the sm_89 table (per dtype); sm_120 is deep-measured unless a line says
otherwise. The generated @ref selection_tables page lists every op's families and shipped tables.

### potrf (Lower and Upper)

- **Families:** `tiny`, `cta`, `lpanel:panel=8`, `blocked`, `vendor`. Float also has
  `lpanel:panel=16`. Float has 6 candidates, the other dtypes 5.
- **Keys:** `uplo:exact n:log:3 batch:log` (work \f$\sim n^3\f$).
- **`can_run`** (`src/ops/potrf/potrf.cc:34-57`): native needs a GPU, sub-group 32, a homogeneous
  batch, `n >= 1` and `batch >= 1`.
  - `tiny`: `n <= potrf_tiny_max_n<T>()` and `max_wg >= kPotrfTinyWgSize` (64).
  - `cta`: `n <= potrf_cta_max_n_for_slm<T>(slm_budget)` (min blocks 4).
  - `lpanel`: Lower only, `n <= potrf_lpanel_max_n_for_slm<T>(budget, max_wg, kMinBlocksPerSm, panel)`
    (0 for a panel width `T` does not instantiate).
  - `blocked`: Lower only, `potrf_blocked_available<T>()` and `cta_max_n_for_slm >= 1`.
  - `vendor`: `has_vendor` and a homogeneous batch.
- **Last resort:** `blocked`, then `vendor`, then the rest.
- **Workspace:** `blocked` needs `potrf_blocked_buffer_size` alone; its gemm and trsm take none.
- **Upper:** Upper calls use the Lower rows (the exact key is dropped) unless the table has Upper
  rows. sm_120 has Upper rows (162 float, 627 in all).
- **Data:** sm_89 is converted from the forced-route archive, which has no current-era `lpanel`
  timings, so the sm_89 table never picks `lpanel`.

### posv (Cholesky solve)

- **Families:** `tiny`, `cta`, `blocked`, all field-less. No vendor family: a `vendor` pin warns and
  runs Auto.
- **Keys:** `uplo:exact n:log:3 nrhs:log batch:log`.
- **`can_run`:** `tiny` needs `n <= posv_tiny_max_n<T>()`, `nrhs <= kPosvTinyMaxRhs` and
  `max_wg >= kPosvTinyWgSize`; `cta` needs `nrhs <= kGetrsFusedMaxRhs` and
  `n·nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`; `blocked` a homogeneous batch. Empty problems
  and heterogeneous batches throw `internal_error` before `choose()`.
- **Last resort:** `blocked`.
- **Data:** sm_89 transcribed: `tiny | cta | blocked` inside the old tiny window and `cta | blocked`
  elsewhere.

> **Note:** the `max_wg >= 64` term cannot fail on any device with `max_wg` of 64 or more, so no
> test can turn it red on current hardware.

### trsm {#phase-33-trsm}

- **Families:** `cta`, `sg_left`, `blocked`, `vendor`, all field-less.
- **Keys:** `side:exact trans:exact order:log:2 q:log batch:log`, with `q = side==Left ? B.cols : B.rows`
  and ConjTrans folded to T. `uplo` and `diag` are not keys.
- **`can_run`** (`src/ops/trsm/trsm.cc:45-61`): common terms are a GPU, homogeneous operands, and
  order, q and batch of at least 1.
  - `cta`: `order <= trsm_cta_max_n<T>()` (32) and `max_wg >= 32`.
  - `sg_left`: Left only, sub-group 32, `order <= kTrsmSgLeftMaxN` (32) and `max_wg >= kTrsmSgLeftWgSize` (128).
  - `blocked`: `trsm_blocked_available<T>()`, `trsm_cta_max_n >= 1` and `max_wg >= 32`.
  - `vendor`: `has_vendor` and homogeneous operands.
- **Last resort:** `blocked`, then `vendor`.
- **Errors:** `A.batch_size() != B.batch_size()` throws `batchlas::invalid_argument` in `trsm_validate_params`.
- **Data:** sm_89 transcribed: `cta | blocked | vendor` for order ≤ 32 (7,680 of 17,280 rows) and
  `blocked | vendor` above (9,600 rows); no transcribed row names `sg_left`.
- **`uplo` and `diag` (sm_120 A/B, 56 cells):** four combinations per cell, for float and cdouble.
  The first entry is the same for all four in every cell; lower-ranked entries swap in two cells.
  diag=U is clearly faster for cdouble, but no choice changes, so neither is a key. Add `diag:exact`
  if a timed diag=U table is ever wanted.
- **Indexing:** batch offsets in the complex vendor substitute are 64-bit. A 32-bit offset wrapped at
  \f$2^{31}\f$ elements and faulted.

### gemm

The full family, form and window reference is [the gemm page](../perf/gemm.md) ("GEMM: what ships").

- **Families:** `direct`, `tiled`, `small` (every scalar), `reg:m=..:n=..:k=..:u=..` (float only, 13
  `reg_configs`), `wide:m=..:n=..:k=..` (9 `wide_configs`; `wide:m=128:n=128:k=8` complex<float>
  only), `splitk:m=..:n=..:k=..:s=..` (six wide host tiles × s ∈ {2, 4, 8}, every scalar), `vendor`.
  Candidates: float 43, double 30, complex<float> 31, complex<double> 30.
- **Keys:** `ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. ConjTrans folds to T for
  real scalars. `layout=packed` when A, B and C are contiguous with 16-byte-aligned bases, otherwise
  `strided`.
- **Last resort:** `direct`, then `vendor`.
- **Common term** (`src/ops/gemm/gemm.cc:97-102`): native needs `is_gpu || !has_vendor` (a CPU with a
  host BLAS keeps the vendor), `precision == Default`, m, n, k, batch ≥ 1 and homogeneous operands;
  every 3-D launch (all but `small`) also needs `batch <= max_groups_yz`.
- **Forms:** `can_run` admits exactly the instantiated (config, ta, tb) forms (`reg_form`,
  `wide_form`); a real Trans uses the C-folded instance, a complex form must be listed exactly.
- **Derived in the launcher, never a field, key or term:** the aligned or predicated leg, the
  transpose instantiation, the `small` bucket, and TR, TC and the stages.
- **Data:** sm_89: 38,826 transcribed rows. Float: `small` first on the 30 NN squares up to 48,
  otherwise `vendor` then the old native kernel. Double: native everywhere (`tiled` 4,395 rows,
  `direct` 102, `wide` 64·64·16 15). Complex: `vendor` first everywhere. sm_120: 22,203 deep rows
  (float, cfloat, 2026-10-21) and 14,314 transcribed (double, cdouble, and lattice gaps).
- **Known difference from the old router (sm_89):** between grid points, nearest-row lookup does not
  reproduce the old predicates. On random off-grid shapes the first runnable native entry differs
  from the old kernel for 6.7% of double shapes and 31.7% of float shapes. Fast-path divisibility
  is not a key, so packed shapes that are not multiples of the tile take the predicated leg. Both
  are accepted until the sm_89 retune.

### gemv {#phase-5-gemv}

- **Families:** `cta` (bodies 3 and 5), `direct` (bodies 1, 2 and 4), `vendor`. The body split and the
  segment width are derived; `BATCHLAS_GEMV_SEGT` steers the width.
- **`can_run`:** A homogeneous, x and y with A's batch, `x.size() == red` and `y.size() == out`.
  `cta` also needs a GPU, sub-group 32 and `trans != N` (`ops::gemv::device_allows`). `direct` has
  no GPU gate. `vendor` needs `has_vendor` and a homogeneous A.
- **Last resort:** `vendor`, then `direct`.
- **Keys:** `trans:exact out:log red:log batch:log`. `out` and `red` are the lengths of y and x, and
  they swap with `trans`. Grid: out 255|256, red 63|64 and 352|353, batch 319|320. 1,408 rows.
- **sm_89 ranking:** `cta | vendor | direct` for complex double with T inside the old window (48
  cells), `vendor | cta | direct` for the other T cells, and `vendor | direct` under N.
- No gemv validator exists, so a non-conforming call goes to the vendor (known-defects #1).

### geqrf {#phase-5-geqrf}

- **Families:** `tiny` (m == n and `n <= geqrf_tiny_max_n_for_slm`), `cta` (`geqrf_cta_fits`),
  `blocked` (`geqrf_blocked_available` and `geqrf_cta_max_elems_for_slm >= 1`), `vendor`. Native
  needs a GPU, sub-group 32, a homogeneous batch and m ≥ n ≥ 1: no native family takes a wide shape.
- **`can_run`:** in `src/ops/geqrf/can_run.hh`, so a host test can call it with a synthetic device.
  `tiny` has no `max_wg` check.
- **Keys:** `form:exact n:log:3 aspect:log`. `form` is sq, tall or wide, and `aspect` is max/min by
  integer division. 636 rows. The grid straddles the tiny windows, the order floors (64, 76, 48, 256),
  the tall clause and the CTA/blocked crossover (float 96, double 48).
- **Workspace:** `geqrf_buffer_size` is the chosen family's need. Callers that size once for a
  bounding panel use `geqrf_buffer_size_bound`, the maximum over every family this device can run.

### gesv {#phase-5-gesv}

- **Families:** `tiny` (fused LU factor and solve) and `blocked` (public `getrf`, then public `getrs`).
  No vendor family.
- **`tiny` `can_run`:** backend not NETLIB, a GPU, sub-group 32, homogeneous A and B,
  `n <= gesv_tiny_max_n<T>()` (32; cdouble 16), `nrhs <= 4` and `max_wg >= 64`.
- **Keys:** `n:log:3 nrhs:log`. sm_89: float is `tiny`, then `blocked`, for n ≤ 32; cfloat is `tiny`
  for n ≤ 16; everything else is `blocked`.
- **Workspace:** `getrf_buffer_size + getrs_buffer_size`.

### getrf {#phase-5-getrf}

- **Families:** `tiny` (n ≤ `getrf_tiny_max_n<T>()`, 32 or 16 for cdouble, and `max_wg >= 64`),
  `cta` (n ≤ `getrf_cta_max_n_for_slm<T>(budget)`), `blocked` (the public gemm and trsm are injected,
  and `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1`), `vendor`. Native families need backend not
  NETLIB, a GPU, sub-group 32 and a square A. The vendor on CUDA also needs a square A (cuBLAS
  `getrfBatched` is square-only).
- **Keys:** `n:log:3 batch:log`. The grid straddles the tiny windows (float 5..32, cfloat 5..7 and
  9..24), the cdouble ceiling of 16, and the blocked floors (float 256, cfloat 512, or 256 at batch ≥ 256).
- The panel leaf, the laswp mode, the blocking factor and the tiny bucket are derived in the driver.

### getri {#phase-5-getri}

- **Families:** `blocked` (P is written into C, then two public trsm calls) and `vendor` (cuBLAS
  `getriBatched`, rocSOLVER, LAPACKE). Last resort: `vendor`, then `blocked`.
- **`can_run`:** `blocked` needs a GPU, sub-group 32, backend not NETLIB, a square A, `n` and `batch`
  at least 1, and a homogeneous batch. `vendor` is `factorization_vendor_available<B>` and a
  homogeneous batch.
- **Keys:** `n:log:3 batch:log`. `batch` is kept for a retune, though every sm_89 batch row has the
  same ranking. sm_89 grid: 127|128 (float) and 255|256 (cfloat); double and cdouble are vendor at
  every n.
- Sizing reads the metadata only.

### getrs {#phase-5-getrs}

- **Families:** `cta` (`max_wg >= 32`, `nrhs <= 8`, and `n·nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`),
  `blocked` (two public trsm calls), `vendor`.
- **Native `can_run`:** backend not NETLIB, a GPU, sub-group 32, a conforming pair (A square,
  `B.rows == n`, equal batch), homogeneous operands, and n, nrhs and batch at least 1. A
  non-conforming pair goes to the vendor.
- **Keys:** `n:log:2 nrhs:log batch:log`. `transA` is not a key. Grid: n 31|32, nrhs 2|3, 4|5, 63|64,
  127|128 and 8, batch 127|128. 2,160 rows.

### orgqr {#phase-5-orgqr}

- **Families:** `blocked` (the identity fill plus the public `ormqr`) and `vendor` (a per-item loop).
  Last resort: `vendor`, then `blocked`. The vendor runs n > m and CPU queues; neither family takes a
  heterogeneous batch.
- **`can_run(blocked)`:** GPU, the driver compiled, homogeneous, m, n and batch ≥ 1, and `n <= m`.
- **Keys:** `m:log n:log:2`, with no batch key. The only old threshold is 512, native iff
  `rows <= 512 && cols <= 512`. The grid puts 512|513 on both axes, restricted to n ≤ m, for 153 rows.
- `orgqr_buffer_size` had a latent defect, fixed by sizing through `choose()`. See
  [the orgqr_buffer_size latent defect](../perf/qr.md#the-orgqr_buffer_size-latent-defect).

### ormqr {#phase-5-ormqr}

- **Families:** `blocked` (larft and level-3 WY updates) and `vendor`. The WY width comes from a
  positive `block_size_hint` clamped to [1, k], or else `tuning::ormqr_block_size_for_n`.
- **`can_run`:** both refuse a heterogeneous batch; `blocked` needs a GPU, `vendor` needs `has_vendor`.
- **Keys:** `side:exact trans:exact m:log k:log q:log batch:log`, where `m` is the order of Q,
  `k = min(rows, cols)`, and `q` is the extent of C that Q does not act on. 540 rows.
- Complex `Transpose::Trans` throws `invalid_argument` from `ormqr` and `ormqr_buffer_size` before
  `choose()`. The vendor spells a real ConjTrans as Trans.
- `blocked` has no batch term: the 65535 grid-z report (known-defects #16) did not reproduce on sm_120.
  Vendor-free past 65535 its sub-gemm or sub-trmm may throw `NoRouteError` itself.

### spmm {#phase-5-spmm}

- **Families:** `direct` (CSR, `spmm_native_csr`: gather for NoTrans A, scatter otherwise) and
  `vendor` (cuSPARSE, rocSPARSE, netlib). Last resort: `vendor`, then `direct`.
- **`can_run(direct)`:** CSR, the body for `transA` compiled, `one_spmm()`, homogeneous B and C, and
  batch ≥ 1. No GPU gate: the NETLIB native queue relies on `direct`.
- **`can_run(vendor)`:** `has_vendor`, homogeneous B and C, and two exclusions. On CUDA, complex with
  `transB == ConjTrans` and nrhs 1 is excluded, because cuSPARSE leaves C unwritten. On NETLIB, any
  transpose is excluded, because netlib throws. The former cdouble N/N one-column exclusion is
  deleted (known-defects #13, #17).
- **Keys:** `transA:exact transB:exact m:log nrhs:log batch:log`. No nnz key: the per-item nnz is in
  device memory. 500 rows per table. A `cpu` table is transcribed, because a CPU never borrows.
- cuSPARSE's default algorithm is not bit-reproducible, so the tests identify a pinned vendor by its
  trace line.

### syev {#phase-5-syev}

- **Families:** `cta` (`syev_cta`), `cta_fused`, `jacobi` (`syev_jacobi_cta`), `blocked`, `two_stage`
  and `vendor`, in tie order.
- **`can_run`** (`src/ops/syev/syev.cc:51-66`): native needs backend not NETLIB, a GPU, a homogeneous
  batch, a square A and n ≥ 1. `cta`, `cta_fused` and `jacobi` also need n ≤ 32 and sub-group 32;
  `cta` and `cta_fused` are not ROCm. `blocked` and `two_stage` need batch ≥ 1. A non-square A throws
  `invalid_argument`.
- **Keys:** `jobz:exact n:log:3 batch:log`. `uplo` is not a key, because the large-n drivers mirror
  Upper into Lower. Grid: n 8|9, 24|25, 32|33, 256|257, 320|321, 448|449, 512|513, 1024|1025. 370 rows.
- **sm_89 transcribed pattern, jobz=V:**
  - float: `jacobi` for n ≤ 8, `cta_fused` for 9 to 32, `blocked` to 448, `two_stage` to 1024, then `vendor`;
  - double: `jacobi`, `blocked` to 448, then `vendor`;
  - cfloat: `cta_fused` for n ≤ 8, `cta` to 32, `blocked` to 512, then `vendor`;
  - cdouble: `cta` to 24, `vendor` for 25 to 32, `blocked` to 256, then `vendor`.
  - jobz=N goes to `two_stage` above 320 in every dtype.
- ROCm has no syev table and borrows the sm tables (untested).

### gesvd

- **Families:** `jacobi` (`gesvdj_cta`), `cta`, `blocked`, `vendor`. The jobs are canonicalised once,
  Thin to All where they coincide.
- **`can_run`** (`src/ops/gesvd/gesvd.cc:63-81`): native needs a GPU, a homogeneous batch and
  `m, n, batch >= 1`.
  - `jacobi`: sub-group 32, not Hermitian, and `max(m, n) <= gesvd_jacobi_max_dim<T>(vectors, slm_budget)`
    (64, halved until its local memory fits).
  - `cta`: sub-group 32, `max(m, n) <= 32`, not thin, and `herm ? m == n : real T`.
  - `blocked`: `herm ? m == n && Lower : real T`.
  - `vendor`: `has_vendor` and a homogeneous batch. cuSOLVER refuses `max(m, n) > 32`, non-packed
    batches and thin factors, so Auto reaches the vendor for those shapes only where no native family
    runs.
- **Keys:** `herm:exact vec:exact m:log:1.5 n:log:1.5`. Grid 32|33 and 64|65. 2,025 rows.
- Known defects #14, #15 and #18.

### 12.9 Level-3 four (symm, syrk, syr2k, trmm) {#level-3-four-symm-syrk-syr2k-trmm}

The four level-3 ops use the same structure with `NoFields` families. The vendor family calls
`backend::<op>_vendor` under `has_library<B>(spec.vendor)`, not the public entry, so there is no
recursion. Float-only kernels are instantiated only in `if constexpr` arms.

| op | candidates | keys (rows per dtype, sm_89) | Auto (sm_89 transcription) | last resort |
|---|---|---|---|---|
| syrk | float `gram, triangular, vendor`; double `gram, vendor` | `form:exact trans:exact n:log:2 k:log batch:log` (18,450) | n ≤ 128 `gram`; above, `triangular` when squareish or (n ≥ 257, k ≥ 8, batch·T(T+1)/2 ≥ 160, T = ceil(n/128)), else `vendor`; ConjTrans `vendor` first; double above 128 `vendor` | `triangular, gram, vendor` |
| syr2k | float `triangular, vendor`; double `vendor` | `n:log:2 k:log batch:log` (455) | float batch ≥ 2 `triangular`; batch 1 `vendor`; double `vendor` | `triangular, vendor` |
| symm | `expand, vendor` (float, double) | `form:exact m:log n:log batch:log` (5,292) | float `expand` iff sq and (batch ≥ 4 or max(m, n) ≥ 256); double `vendor` | `expand, vendor` |
| trmm | `triangular, expand, vendor` (all four dtypes) | `side:exact order:log:2 q:log batch:log` (360) | Left `triangular`; Right `expand` | `expand, triangular, vendor` |

**Shared terms:**

- Native families run on a native GPU backend (`select::native_gpu_backend<B>`: CUDA and SYCL; symm's
  `expand` also on ROCm), on a GPU, with homogeneous operands, extents and batch ≥ 1, and
  `batch <= max_groups_yz` (65535 on CUDA: the batch is grid z, and 65536 throws at launch).
  The NETLIB host stays on the vendor.
- The vendor family needs `has_vendor` and refuses a heterogeneous operand on every backend.
- syrk, syr2k, symm and trmm return a no-op event for an empty problem.
- `form` (symm and syrk) is sq when `2·min ≥ max`; otherwise tall when `a > 2b`, else wide, with `a`
  and `b` as in `form_of` (for symm, C's rows and columns).

**Per op:**

- **syrk.** `gram` is `syrk_gram_tiles`, float and double, for n ≤ 128 (`syrk_gram_supported`); above
  128 a pin throws, because the kernel would answer wrongly. It also needs `max_wg >= gram_threads(n)`
  and `slm_budget >= gram_slm_bytes<T>(n)`. `triangular` is float only, and needs
  `max_wg >= kTriangularWg` (256) and `T(T+1)/2 <= max_groups_yz` tiles. ConjTrans is passed as
  Trans, and the `trans` key keeps real ConjTrans on the vendor.
- **syr2k.** `triangular` is float only, with the same `max_wg` and tile terms. The launch folds a
  real ConjTrans to Trans, and the key has no `trans`, so it follows the Trans rows.
- **symm.** `expand` is the mirrored expansion into a queue workspace lease, then the public gemm. It
  needs `max_wg >= kExpandWg` (256) and `expansion_fits(q, k, batch, bytes, max_groups_yz)`.
  `BATCHLAS_EXPAND_MAX_BYTES` lowers the limit. symm does not read `BATCHLAS_EXPAND_ROUTE`. On ROCm the
  `vendor` arm is `symmetrize_gemm` (symmetrize plus the public gemm), because rocBLAS has no batched
  symm wrapper.
- **trmm.** `triangular` is Side::Left only, with `max_wg >= 256` and `trmm_tile_groups <= max_groups_yz`
  (row × column tiles in grid y). `expand` is the expansion plus the public gemm at beta 0, and it
  needs `max_wg >= 256` and `expansion_fits`. `BATCHLAS_TRMM_ROUTE=vendor` means the `cublas?trmm`
  loop; the expansion is `expand`.

## 13. Design decisions

| Topic | Chosen | Rejected |
|---|---|---|
| Source of speed | Measured or transcribed tables per (op, device), dtype a key | Hand-written device rules; first-call autotuning; a fitted cost model |
| Choice type | `std::variant` of family structs with int fields | One type per compiled config; a flat struct with family and fields; an enum per kernel; virtual kernel objects |
| Nested ops | Each op decides through the public function. A parent override only where a measurement needs one. | A parent that plans everything |
| Unmeasured shapes | Nearest row by weighted log distance over the op's keys | Generated regions; power-of-two buckets |
| Storage | One text format, embedded at build time, parsed by the same parser as pins. `BATCHLAS_TUNED_DIR` overrides. | Compiled C++ tables; runtime files only |
| Untuned device | Borrow the same family's nearest arch, then the other families | Per-op fact rules; a conservative vendor default |
| Row content | Every candidate in rank order, with times | A single winner |
| Ties | Within 3% of the best is a tie, and ties go to candidate-list order | Keep the previous winner unless 5% better; best of N |
| Staleness | Hash in the header, warning only | Rejecting stale tables; a version stamp only |
| Workspace | Sized through the same `choose()` | An explicit plan object; the maximum over candidates |
| Pins | Env var, plus RAII `ScopedPin` for tests | Path-scoped pins; env only |
| Candidate list | A `constexpr` list in `choice.hh`, which is also the compiled set | Per-family ranges; a separate tuner config |
| Golden routing snapshot test | Not built | A snapshot of every route compared in CI |
| Tuner grid | A coarse log grid, plus refinement where neighbours disagree | A fixed grid only; recording real workloads |
| CPU backend | Tuned like any device (`*.cpu.txt`); a CPU with a host BLAS keeps gemm on the vendor | Borrowing from a GPU |
| Pilot op | potrf | gemm first; a smaller op first |
