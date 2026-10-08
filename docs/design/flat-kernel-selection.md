# Flat kernel selection {#design_flat_selection}

> **Status:** current · checked 2026-10-06

Each op that reads `BATCHLAS_<OP>_ROUTE` (19 ops) chooses its kernel in one place: `select::choose()`
in `src/select/`, called from the op's own file in `src/ops/<op>/`. A choice is a value from a
`std::variant` of kernel families. `can_run` decides correctness, and per-device tables rank the
candidates for speed. The old `RouteTable` layer is gone. The phase history is in
@ref design_flat_selection_phase3_plan.

## 2. Selection rules

| # | Rule |
|---|---|
| R1 | Three hops, one op file: public function → `choose()` → `std::visit` → launch, all in `src/ops/<op>/<op>.cc`. Only `choice.hh` (the vocabulary) and the shared `select.hh` live elsewhere. |
| R2 | A choice is a value: a `std::variant` of family structs with int fields. One spelling (`lpanel:panel=8`) is used in tables, pins, traces, coverage rows and test names. |
| R3 | `can_run` is correctness only. If it is false, the launch would throw or return a wrong answer. It never encodes "slower", and it agrees exactly with what the launch accepts (tested). |
| R4 | Speed lives only in tables. The last-resort order is by generality, not speed. |
| R5 | Sizing and running call the same `choose()`. `<op>_buffer_size` returns the chosen family's need. |
| R6 | A bad pin throws `std::invalid_argument`: an unknown spelling, or a choice whose `can_run` is false for the shape. The class words `native` and `vendor` are the exception (see §5.2). |
| R7 | Every candidate can be pinned, and a test runs each one on shapes that straddle its limits. |
| R8 | A device with no table borrows another device's table, warns once per op, and tags every trace line with the table it used. |

## 4. Op file layout

### 4.1 Files

Each op has two files:

- `src/ops/<op>/choice.hh` holds the vocabulary: the family structs, `candidates<T>()`, `key_names`,
  the tuner grid, and the `select::OpSpec` (the op, its vendor library group, and its last-resort
  rules).
- `src/ops/<op>/<op>.cc` holds `key_of`, `can_run`, `launch`, `workspace`, and the public entry
  points.

### 4.3 The op file `<op>.cc`

The public entry point validates its arguments, then calls `select::run`, which opens the trace and
coverage scope around the launch. `<op>_buffer_size` calls `select::pick` and returns the chosen
family's need. Families with no fields use `NoFields<"name">`. For ops with no fields,
`select::all_of<Choice>()` lists the candidates in declaration order, which is also the tie order.

The vendor family calls the backend only under `if constexpr (select::has_library<B>(spec.vendor))`.
Otherwise `select::no_vendor` throws `NoRouteError` and records the `miss` row.

Nested ops decide for themselves: a blocked driver calls the public `gemm`, `trsm`, `potrf`, `getrf`
or `ormqr`, which choose their own kernels. A parent override that forces a child choice is not built.
It is needed only if a measurement shows that a parent needs a different child.

Common `can_run` terms for the native families are a GPU (`is_gpu`), sub-group 32 (`has_sg32`)
where the kernel needs it, a homogeneous batch, and extents and batch of at least 1. A heterogeneous
batch goes to the vendor, or has no route, because the vendor loops run every item at the top-level
extents.

## 5. The shared helper `src/select/select.hh`

Everything in `src/select/` is generic over an op's choice type: device facts, spelling and pins, tables and lookup, trace and coverage.

### 5.1 Device facts

`select::Device` is memoized per queue device and holds the facts that rules and table lookup use:

```cpp
struct Device {
  std::string key;     // "sm_120", "sm_89", "gfx90a", "cpu", "intel_<id>"
  std::string family;  // "sm", "gfx", "intel", "cpu"
  int arch_number;     // 120, 89, 90 (gfx90a), 0 for cpu
  bool is_gpu, has_sg32, has_vendor;  // has_vendor: the op's library group is compiled in
  int64_t slm_budget;
  int max_wg;
};
```

The CUDA key comes from the compute capability, the ROCm key from `gcnArchName`, and host queues
use `cpu`.

### 5.2 Spelling and pins

- **Spelling:** `family[:field=value]...`, for example `tiny`, `lpanel:panel=8`,
  `reg:m=128:n=128:k=8:u=1`, `vendor`. The positional form `lpanel:8` is accepted as input. Input
  is case-folded and trimmed.
- **Pin values:** `auto`, `native` (the best runnable non-vendor candidate), `vendor`, or a spelling.
  `native` and `vendor` fall back to Auto with a warning when nothing in their class can run.
  Any other value that does not parse, or that `can_run` refuses, throws.
- **Removed spellings throw:** `native:cta`, `vendor:auto`, `batchlas_cta`, and gemm kernel names
  such as `tiled16` or `128x128x8`. `BATCHLAS_<OP>_VARIANT`, `BATCHLAS_<OP>_PROVIDER` and
  `BATCHLAS_GEMM_SYCL_KERNEL` are not read.
- **Tests** pin with `select::ScopedPin`, a thread-local slot that wins over the environment.
- **Snapshot:** the pin is read from the settings snapshot, which is captured once. A raw `setenv`
  mid-process is not seen until `detail::reload_settings()` runs (`ScopedEnvVar` does this). See
  [route pins are raw strings](environment.md#environment-route-pins-are-raw-strings).

### 5.4 Tables and provenance

Each table is one file, `tuned/<op>.<dtype>.<device>.txt`, embedded at build time and parsed once
per op. `BATCHLAS_TUNED_DIR=<dir>` replaces a built-in table with a file of the same name (the trace
then shows `[override sm_120]`).

```
# op=potrf dtype=float device=sm_120 batchlas=<sha> kernels=<hash> source=<origin>
# keys: uplo:exact n:log:3 batch:log
uplo=L n=64  batch=8192  | lpanel:panel=8 0.302 | vendor 0.490 | cta 0.530 | blocked 0.536
uplo=L n=512 batch=2048  | blocked 14.10 | vendor 13.74 | lpanel:panel=8 14.84   # tie: within 3%, list order
```

- Each row lists every measured candidate in rank order. Times are milliseconds per call for the
  whole batch.
- A row is either all timed or all untimed. An untimed row has the form `uplo=L n=8 nrhs=1 batch=128 | tiny - | cta -`, and
  the header then needs `source=transcribed:<hex sha>`.
- `kernels=<hash>` is the first 8 hex digits of a SHA-256 over the op's kernel sources, listed in
  `tools/tune/<op>_spec.cc` between the `kernel-sources-begin` and `kernel-sources-end` markers. A
  mismatch or `unknown` is a warning, in CMake and in `.github/ci/check_tuned_tables.py`. The table
  stays in use.
- A candidate in a table must be a member of `candidates<T>()`. The loader checks this on first use,
  and `tuned_tables_tests` checks every shipped table without a GPU.

Every shipped table has one of three origins:

| Origin | Meaning | Examples |
|---|---|---|
| measured | `batchlas_tune` timings (`source=tuner:<jsonl>`) | trsm float and double on sm_120 |
| converted | timings from earlier sweeps, written by `scripts/sweep_to_table.py` | potrf on sm_120 and sm_89; posv on sm_120 |
| transcribed | the old router's preference order, evaluated at each grid cell; untimed | all other sm_89 and sm_120 tables; the cpu spmm table |

The inventory per op, dtype and device is in @ref tuned_tables_readme. Every op ships a table for
every dtype it instantiates on sm_89 and sm_120, so no device borrows in the normal case. Retuning the
transcribed tables with the tuner is open (see §11).

### 5.5 How a call chooses: lookup, borrow order, last resort

`select::choose()` tries these in order:

1. A pin, from `select::ScopedPin` or `BATCHLAS_<OP>_ROUTE`. A bad pin throws.
2. For each table in borrow order, the nearest row (below). The first candidate of that row whose
   `can_run` is true wins.
3. Otherwise, the first runnable candidate in the op's last-resort order.
4. Otherwise it throws. In a vendor-free build the throw is `NoRouteError`, and the coverage row is a `miss`.

**Nearest row.** Exact keys (`name:exact`) must match, dropping the rightmost exact key when no row
matches all of them. Among the rows kept, the smallest weighted distance
Σ w·|log2(row / key)| over the `:log` keys wins. The weight comes from `<name>:log:<w>`
(default 1). Ties within 1e-9 go to the row with the smaller log keys, compared in `# keys:` order.

**Borrowing (§5.5).** For a device key `d`:
1. the table for `d`;
2. the other tables of the same family (`sm`, `gfx`, `intel`), nearest arch number at or below
   `d`, then nearest above;
3. every other family, in the order `sm`, `gfx`, `intel`, with `cpu` last.

A CPU never borrows a GPU table. The first borrowed table prints one warning per op per process:

```
batchlas: potrf has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)
```

**Last resort.** The candidate list reordered by generality. The default is `blocked`, then
`vendor`, then the rest in list order. Ops with another order list it in their `choice.hh`.

**Trace and coverage.**
- `BATCHLAS_SELECT_TRACE=1` prints one line per call on stderr: the chosen spelling, its time, the
  runner-up, and the table tag. Nested calls are indented.
- Each call writes a coverage `reached` row. `chosen_algo` holds the spelling and `chosen_origin`
  holds `native` or `vendor`. `scripts/route_diff.sh` diffs these rows.
- `BATCHLAS_KERNEL_TRACE` is separate. Use it to see a derived leg (for example gemm's aligned or
  predicated launch), which the coverage row does not record.

```
potrf float n=64  batch=8192 -> lpanel:panel=8   0.302 ms, next vendor 0.490 ms          [sm_120]
potrf float n=512 batch=2048 -> blocked          14.10 ms, tied with vendor 13.74 ms (3%)  [sm_120]
```

## 6. The tuner `tools/tune/batchlas_tune`

```
batchlas_tune <op> --dtype float,double,cfloat,cdouble --devices <gpu> --out tuned/ \
              --raw benchmarks/results/tuning/
```

### 6.1 What it times

Every entry of `candidates<T>()`, pinned with `ScopedPin` and run through the public op.
- **Inputs:** built per op by `tools/tune/<op>_spec.cc` (SPD matrices for potrf).
- **Verification:** each run is checked against a host reference (a residual for potrf, a
  componentwise backward error for trsm, a componentwise error for gemm). A failing candidate is
  dropped from that cell.
- **Skips:** a candidate whose `can_run` is false at a cell is recorded as skipped, not timed.

### 6.2 Grid and edge refinement

- A coarse `grid_n × grid_batch` per `uplo`. For potrf: 34 values of n from 1 to 1280, batch 128,
  512, 2048, 8192, 32768. Inputs are capped at 4 GiB.
- Where neighbouring n points have different winners, n is bisected with a geometric midpoint until
  the bracket is under 10% (`hi/lo < 1.1`). Batch is not refined. Refined points become ordinary rows.

### 6.3 Noise protocol

1. A throwaway JIT pass over every cell.
2. Per cell, the candidates are interleaved with a 1.5 s warm-up each, then 16 timed reps with the
   order rotated each rep. The median is taken.
3. The grid is run twice, the second time in reverse candidate order. A cell's time is the mean of
   its two pass medians. If the two medians differ by more than 10%, the cell is measured once more.
4. **Tie rule:** every candidate within 3% of the best is tied, and ties go to the `candidates<T>()`
   order. Native candidates precede `vendor`, so near-ties go to the native kernel (potrf float
   n=512, batch 2048, sm_120: blocked 14.10 ms vs vendor 13.74 ms, blocked wins; costs at most 3%).
   Moving `Vendor{}` to the front reverses it.
5. **Guard:** one measuring process per GPU, behind a flock. It refuses to run if a foreign compute
   process is on the GPU or utilisation is above 5%. `--devices` is required. The driver is a
   SYCL-free launcher, so it holds no CUDA context.

### 6.4 Output

`tuned/<op>.<dtype>.<device>.txt` in the §5.4 format, written by `scripts/sweep_to_table.py --tuner`,
with `kernels=<hash>`. The raw per-rep JSONL goes to `benchmarks/results/tuning/` (Git LFS).
`--gate` compares Auto against an old choice on the cells where they differ. FAIL means the new
time is above 1.05× the old in both passes. Exit code 1 on FAIL, 3 when a row is an error or no cell
was gated.

### 6.5 Staleness check

A CMake configure step and `.github/ci/check_tuned_tables.py` recompute each op's kernel hash. A
header hash that differs, or says `unknown`, prints a warning (`tuned/potrf.float.sm_120.txt is
stale (kernels unknown -> 3f9a1c2e); run tools/tune`). It is never an error.

## 8. Tests

Each migrated op has an `<op>_candidates_tests` suite built from the five parts below. potrf's is
`tests/potrf_candidates_tests.cc`. The rules for writing them are in @ref dev_agent_guide.

### 8.1 Every candidate, pinned

Loop over `candidates<T>()` for every dtype, pin each with `ScopedPin`, run it on shapes that
straddle its `can_run` limits in both directions, and check the result, info and the untouched
triangle. Required inputs: non-natural `ld` and stride, complex data with a nonzero imaginary part,
poison the kernel will accept, and a saturating batch (at least 1024 identical items, bit-identical
results) for shared-local-memory tiers.

### 8.2 `can_run` equals launch

For every candidate and straddling shape: where `can_run` is true the direct dispatch call does
not throw, and where it is false the direct call refuses. This keeps R3 honest.

### 8.3 Sizing

Under each pin, run with a workspace of exactly `<op>_buffer_size(...)` bytes inside a poisoned
arena and check that nothing is written past the end.

### 8.4 Table validity

`tests/tuned_tables_tests.cc`, no GPU. Every embedded table parses, names only members of
`candidates<T>()`, and has strictly ascending ranked times apart from ties. On the matching device,
each row's first runnable candidate also passes `can_run` at that row's own key.

### 8.5 Pins and borrowing

No GPU (`tests/select_tests.cc`). An unknown pin throws, and so does a `can_run`-false pin. Every
old alias must throw (each op's should-throw test lists the removed spellings). With synthetic
tables, the borrow order of §5.5 holds and the warning prints once.

## 10. Validation and gates

| Check | Result |
|---|---|
| Failing test names, vendor and vendor-free trees | Match the base for each migrated op. Each deliberate break of an op file turned a narrow, named set red. |
| Off-grid data gate (random points, old router vs nearest row plus `can_run`) | 100% agreement for the phase-5 ops and the level-3 four, on sm_89 or sm_120. Dropping threshold rows fails the gate. Not run for gemm. |
| Coverage of Auto calls, old vs new build | Same `reached` row as the old build, except the deliberate changes listed under each op. |
| Live timing, potrf on sm_120 | 120 off-grid cells: 54 chose the same kernel. 33 changed and were timed in two reversed passes: 0 FAIL, worst 1.014× (cdouble n=44, batch 8192), the others 0.60 to 0.994×. The remaining 33 were not timed: 32 exceed 12 GiB, and n²·batch ≥ 2³¹ aborts in `factor_bench` on any route. |
| Live timing, potrf on sm_89 | Open. The sm_89 tables are sparse and have no current `lpanel` timings. |

**Acceptance gate for a migrated op:**
1. Correctness: the op's `*_candidates_tests`, `tuned_tables_tests` and `select_tests` pass in the
   vendor and vendor-free trees. Failing names are compared with `tests/known-failures.txt`.
2. `scripts/sweep_to_table.py --check` passes.
3. Live, at saturation (batch ≥ 128): the new choice is not more than 1.05× slower than the old routing at
   off-grid points between grid n values. The run is interleaved in one process, and a loss counts
   only if it reproduces with the order reversed.
4. Readable: a reviewer finds the kernel for a given call by reading `<op>.cc` and its table.

**Tests required for each candidate:**
- Pinned, on shapes that straddle its limits in both directions. Non-natural `ld` and stride,
  complex data with a nonzero imaginary part, poison in the untouched triangle, and a saturating batch
  of 1024 for shared-local-memory tiers.
- `CanRunEqualsLaunch`: where `can_run` is true the direct call does not throw, and where it is false the direct call refuses.
- Sizing: the workspace is exactly `<op>_buffer_size` bytes inside a poisoned arena.
- Table validity: every shipped table parses, names only members of `candidates<T>()`, has strictly
  ascending times apart from ties, and on its own device has a first runnable entry at each row's key.
- Pins and borrowing: an unknown or `can_run`-false pin throws, the borrow order holds on synthetic tables, and the warning prints once.

## 11. Open items

- **Retune.** The transcribed tables (sm_89 everywhere; sm_120 gemm, cfloat and cdouble trsm, and the
  spmm cpu table) and the converted potrf and posv tables are to be replaced with tuner output and
  re-stamped with hashes. Until then, transcribed rows encode the old router's choices, untimed.
- **sm_89 live gate** for potrf and for the other ops with transcribed sm_89 tables.
- **hemm, herk and her2k** still select by hand in `src/backends/cublas.cc` (vendor builds only). They
  expand or fold into the public gemm when `expansion_preferred`, `herk_gemm_preferred`
  (batch ≥ 4 and n ≤ 768) or `her2k_gemm_preferred` (batch ≥ 2 or n ≥ 128) hold, and otherwise use the per-item vendor loop.
  They honour `BATCHLAS_EXPAND_ROUTE=expand|loop`, and herk's gram opt-in is `BATCHLAS_SYRK_ROUTE=gram`.
  They have no `BATCHLAS_<OP>_ROUTE` and throw `NoRouteError` in a vendor-free build. The planned families are hemm
  `{expand, vendor}`, herk `{gram, fold, vendor}` and her2k `{fold, vendor}`.
- **potrf.** The blocked leaf is hard-wired to `potrf_cta_dispatch` (`potrf_blocked.cc:329`). `Blocked`
  has no fields: `nb` and `W` come from `PotrfBlockedConst`, overridable by `BATCHLAS_POTRF_NB` and `BATCHLAS_POTRF_W`.
  `lpanel:panel=16` has never been timed, so no table contains it.
- **gemm `direct`** indexes the batch with `int` offsets (`b * stride`), which can overflow at a large
  batch times stride. It is not a `can_run` term. The fix is 64-bit offsets.
- **Grid-z ceiling.** Other ops that put the batch in dimension 0 of a 3-D range abort above 65535
  work-groups. The phase-5 kernels were not audited for it, and ormqr's blocked driver hits it (known-defects #16).
- **`select::level3_tile_route_available`** is conservative (float, or any type with cuBLAS). Widening
  it moves the vendor-free routes of ortho and ormqr, so that is a separate change.
- **Stale comment** at `potrf.hh:51-53` (wrong `options.hh` line numbers).

Known defects that affect routing are in [known defects](known-defects.md) (for example #12, heterogeneous batches
on the vendor, and #13, the cuBLASLt segfault on complex double gemm with a unit dimension, which
`gemm_vendor_impl` works around with the typed `cublasZgemmStridedBatched`).

## 12. Per-op families

Candidate counts and keys below come from the op's `choice.hh`. Tables are on sm_89 and sm_120
unless a line says otherwise.

### potrf (Lower and Upper)

- **Families:** `tiny`, `cta`, `lpanel:panel=8`, `blocked`, `vendor`. Float also has
  `lpanel:panel=16`, which no table times yet. Float has 6 candidates, the other dtypes 5.
- **Keys:** `uplo:exact n:log:3 batch:log`. `n` weighs 3 because the work grows as n³.
- **`can_run`:** native needs a GPU, sub-group 32, a homogeneous batch, `n >= 1` and `batch >= 1`.
  - `tiny`: `n <= potrf_tiny_max_n<T>()` and `max_wg >= 64`.
  - `cta`: `n <= potrf_cta_max_n_for_slm<T>(slm_budget)` (min blocks 4).
  - `lpanel`: Lower only, the panel width instantiated for `T`, and `n <= potrf_lpanel_max_n_for_slm`.
  - `blocked`: Lower only, and `cta_max_n_for_slm >= 1`.
  - `vendor`: `has_vendor`.
- **Last resort:** `blocked`, then `vendor`, then the rest.
- **Workspace:** `blocked` needs `potrf_blocked_buffer_size` alone. Its gemm and trsm take no
  workspace.
- **Upper:** Upper calls use the Lower rows (the exact key is dropped). sm_120 has measured Upper
  rows for float only, so the other dtypes rank only the Upper-capable entries of a Lower row.
- **Data:** sm_120 is converted from the routing sweeps. sm_89 is converted from the archive, which
  has no current-era `lpanel` timings, so the sm_89 tables never pick `lpanel`.

### posv (Cholesky solve)

- **Families:** `tiny`, `cta`, `blocked`. All are field-less. There is no vendor family, so a
  `vendor` pin warns and runs Auto.
- **Keys:** `uplo:exact n:log:3 nrhs:log batch:log`.
- **`can_run`:** `tiny` also needs `max_wg >= 64`. Empty problems and heterogeneous batches throw
  `internal_error` before `choose()`.
- **Last resort:** `blocked`.
- **Data:** sm_89 is transcribed (untimed): `tiny | cta | blocked` inside the old tiny window and
  `cta | blocked` elsewhere. sm_120 is converted from the posv sweep, with duplicate re-runs
  resolved to the later measurement.

> **Note:** the `max_wg >= 64` term cannot fail on any device with `max_wg` of 64 or more, so no
> test can turn it red on current hardware.

### trsm {#phase-33-trsm}

- **Families:** `cta`, `sg_left`, `blocked`, `vendor`. All are field-less.
- **Keys:** `side:exact trans:exact order:log:2 q:log batch:log`, with `q = side==Left ? B.cols : B.rows`
  and ConjTrans folded to T. `uplo` and `diag` are not keys.
- **`can_run`:** the common terms are a GPU, a homogeneous operand, and order, q and batch of at least 1.
  - `cta`: `order <= trsm_cta_max_n<T>()` (32, a build constant) and `max_wg >= 32`.
  - `sg_left`: Left only, sub-group 32, `order <= 32` and `max_wg >= 128`.
  - `blocked`: `trsm_blocked_available<T>()`, `trsm_cta_max_n >= 1` and `max_wg >= 32`.
  - `vendor`: `has_vendor`.
- **Last resort:** `blocked`, then `vendor`.
- **Errors:** `A.batch_size() != B.batch_size()` throws `invalid_argument`.
- **Data:** float and double on sm_120 are measured. The other sm_120 dtypes and all sm_89 tables
  are transcribed. The sm_89 transcription is `cta | blocked | vendor` for order at most 32 (7,680 of
  17,280 rows) and `blocked | vendor` above that (9,600 rows). No transcribed row names `sg_left`.
- **`uplo` and `diag` (sm_120 A/B, 56 cells):** four combinations per cell, for float and cdouble.
  The first entry is the same for all four combinations in every cell. Lower-ranked entries swap in
  two cells. diag=U is clearly faster for cdouble, but a choice does not change, so neither is a key.
  Adding `diag:exact` is the decision to take if a timed diag=U table is ever wanted.
- **Indexing:** batch offsets in the complex vendor substitute are 64-bit. A 32-bit offset wrapped at
  2³¹ elements and faulted.

### gemm

- **Families:** `direct`, `tiled`, `small` (real only), `reg:m=..:n=..:k=..:u=..` (float only, 10
  configurations), `wide:m=..:n=..:k=..` (5 configurations, every scalar), `vendor`.
  Candidates: float 19, double 9, complex 8.
- **Keys:** `ta:exact tb:exact layout:exact m:log n:log k:log batch:log`. ConjTrans folds to T for
  real scalars. `layout=packed` when A, B and C are contiguous with 16-byte-aligned bases, otherwise
  `strided`.
- **Last resort:** `direct`, then `vendor`.
- **Forms:** `can_run` lists the instantiated (configuration, ta, tb) forms. A real Trans uses a wide
  ConjTrans instantiation, and a complex Trans has none.

  | Choice | Forms |
  |---|---|
  | `direct`, `tiled`, `small` | any (`small` is real and max(m, n, k) ≤ 64) |
  | `reg` 32·32·8, 64·64·8, 128·64·32 (u=4 and u=2), 128·128·8 | NN |
  | `reg` 64·64·16, 128·32·16, 128·32·32 | NN, NT, TN, TT |
  | `reg` 128·64·16 | NT, TN, TT |
  | `reg` 32·128·16 | NN, TN, TT |
  | `wide` 64·64·16 | NN, CN, NC |
  | `wide` 128·32·16 | NC |
  | `wide` 32·128·16 | CN |
  | `wide` 32·32·16, 16·16·16 | NN |

- **Terms:** `direct`, `tiled`, `reg` and `wide` need `batch <= 65535`, because the batch is the
  CUDA grid z. `small` needs sub-group 32, except the float NN tiled leg for 33 to 56. Native
  families run when `is_gpu || !has_vendor`.
- **Derived in the launcher, never a field, key or term:** the aligned or predicated leg, the
  transpose instantiation, the `small` bucket, and TR, TC and the stages.
- **Data:** 38,826 transcribed rows, untimed. Float: `small` first on the 30 NN squares up to 48,
  otherwise `vendor` then the old native kernel. Double: native everywhere (`tiled` 4,395 rows,
  `direct` 102, `wide` 64·64·16 15). Complex: `vendor` first everywhere.
- **Known difference from the old router:** between grid points, nearest-row lookup does not
  reproduce the old predicates. On random off-grid shapes the first runnable native entry differs
  from the old kernel for 6.7% of double shapes and 31.7% of float shapes. Fast-path divisibility
  is not a key, so packed shapes that are not multiples of the tile take the predicated leg. Both
  are accepted until the retune.

### gemv {#phase-5-gemv}

- **Families:** `cta` (bodies 3 and 5), `direct` (bodies 1, 2 and 4), `vendor`. The body split and the
  segment width are derived. `BATCHLAS_GEMV_SEGT` steers the width.
- **`can_run`:** A is homogeneous, x and y have A's batch, and `x.size() == red` and `y.size() == out`.
  `cta` also needs GPU, sub-group 32 and `trans != N` (`ops::gemv::device_allows`). `direct` has
  no GPU gate. `vendor` needs `has_vendor`.
- **Keys:** `trans:exact out:log red:log batch:log`. `out` and `red` are the lengths of y and x, and
  they swap with `trans`. Grid: out 255|256, red 63|64 and 352|353, batch 319|320. 1,408 rows.
- **Ranking:** `cta | vendor | direct` for complex double with T inside the old window (48 cells),
  `vendor | cta | direct` for the other T cells, and `vendor | direct` under N.
- No gemv validator exists, so a non-conforming call goes to the vendor (known-defects #1).

### geqrf {#phase-5-geqrf}

- **Families:** `tiny` (m == n and `n <= geqrf_tiny_max_n_for_slm`), `cta` (m ≥ n and
  `geqrf_cta_fits`), `blocked` (m ≥ n and a CTA panel leaf exists, `geqrf_cta_max_elems_for_slm >= 1`),
  `vendor`. No native family takes a wide shape.
- **`can_run`:** in `src/ops/geqrf/can_run.hh`, so a host test can call it with a synthetic device.
  `tiny` has no `max_wg` check.
- **Keys:** `form:exact n:log:3 aspect:log`. `form` is sq, tall or wide, and `aspect` is max/min by
  integer division. 636 rows. The grid straddles the tiny windows, the order floors (64, 76, 48, 256),
  the tall clause and the CTA/blocked crossover (float 96, double 48).
- **Workspace:** `geqrf_buffer_size` is the chosen family's need. Callers that size once for a
  bounding panel use `geqrf_buffer_size_bound`, the maximum over every family this device can run.

### gesv {#phase-5-gesv}

- **Families:** `tiny` (fused LU factor and solve) and `blocked` (public `getrf`, then public `getrs`).
  There is no vendor family.
- **`tiny` `can_run`:** B is not NETLIB, a GPU, sub-group 32, homogeneous A and B, `n <= gesv_tiny_max_n<T>()`
  (32; cdouble 16), `nrhs <= 4` and `max_wg >= 64`.
- **Keys:** `n:log:3 nrhs:log`. Float is `tiny`, then `blocked`, for n ≤ 32. cfloat is `tiny` for
  n ≤ 16. Everything else is `blocked`.
- **Workspace:** `getrf_buffer_size + getrs_buffer_size`.

### getrf {#phase-5-getrf}

- **Families:** `tiny` (n ≤ `getrf_tiny_max_n<T>()`, 32 or 16 for cdouble, and `max_wg >= 64`),
  `cta` (n ≤ `getrf_cta_max_n_for_slm<T>(budget)`), `blocked` (the public gemm and trsm are injected,
  and `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1`), `vendor`. Native families need B not NETLIB and
  a square A.
- **Keys:** `n:log:3 batch:log`. The grid straddles the tiny windows (float 5..32, cfloat 5..7 and
  9..24), the cdouble ceiling of 16, and the blocked floors (float 256, cfloat 512, or 256 at batch ≥ 256).
- The panel leaf, the laswp mode, the blocking factor and the tiny bucket are derived in the driver.

### getri {#phase-5-getri}

- **Families:** `blocked` (P is written into C, then two public trsm calls) and `vendor` (cuBLAS
  `getriBatched`, rocSOLVER, LAPACKE).
- **`can_run`:** `blocked` needs GPU, sub-group 32, B not NETLIB, a square A, `n` and `batch` at
  least 1, and a homogeneous batch. `vendor` is `factorization_vendor_available<B>`.
- **Keys:** `n:log:3 batch:log`. `batch` is kept for a retune, though every batch row has the same ranking.
  Grid: 127|128 (float) and 255|256 (cfloat). Double and cdouble are vendor at every n.
- Sizing reads the metadata only.

### getrs {#phase-5-getrs}

- **Families:** `cta` (`max_wg >= 32`, `nrhs <= 8`, and `n·nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`),
  `blocked` (two public trsm calls), `vendor`.
- **Native `can_run`:** B not NETLIB, a GPU, sub-group 32, a conforming pair (A square, `B.rows == n`,
  equal batch), no heterogeneous operand, and n, nrhs and batch at least 1. A non-conforming pair goes to the vendor.
- **Keys:** `n:log:2 nrhs:log batch:log`. `transA` is not a key. Grid: n 31|32, nrhs 2|3, 4|5, 63|64,
  127|128 and 8, batch 127|128. 2,160 rows.

### orgqr {#phase-5-orgqr}

- **Families:** `blocked` (the identity fill plus the public `ormqr`) and `vendor` (a per-item loop).
  Last resort: `vendor`, then `blocked`. The vendor runs every shape, including n > m, CPU queues and
  heterogeneous batches.
- **`can_run(blocked)`:** GPU, the driver compiled, homogeneous, m, n and batch ≥ 1, and `n <= m`.
- **Keys:** `m:log n:log:2`, with no batch key. The only old threshold is 512, native iff
  `rows <= 512 && cols <= 512`. The grid puts 512|513 on both axes, restricted to n ≤ m, for 153 rows.
- `orgqr_buffer_size` has a latent defect, fixed by sizing through `choose()`. See
  [the orgqr_buffer_size latent defect](../perf/qr.md#the-orgqr_buffer_size-latent-defect).

### ormqr {#phase-5-ormqr}

- **Families:** `blocked` (larft and level-3 WY updates) and `vendor`. The WY width comes from a
  positive `block_size_hint` clamped to [1, k], or else `tuning::ormqr_block_size_for_n`.
- **`can_run`:** `blocked` needs a GPU, and `vendor` needs `has_vendor`.
- **Keys:** `side:exact trans:exact m:log k:log q:log batch:log`, where `m` is the order of Q, `k = min(rows, cols)`,
  and `q` is the extent of C that Q does not act on. 540 rows.
- Complex `Transpose::Trans` throws `invalid_argument` from `ormqr` and `ormqr_buffer_size` before
  `choose()`. The vendor spells a real ConjTrans as Trans.
- `blocked` hits the 65535 grid-z ceiling (known-defects #16).

### spmm {#phase-5-spmm}

- **Families:** `direct` (CSR, `spmm_native_csr`) and `vendor` (cuSPARSE, rocSPARSE, netlib). Last
  resort: `vendor`, then `direct`.
- **`can_run(direct)`:** CSR, the body compiled, `one_spmm()`, no heterogeneous B or C, and batch ≥ 1.
  There is no GPU gate, because the NETLIB native queue relies on `direct`.
- **`can_run(vendor)`:** `has_vendor`, and three exclusions the old `supports()` lacked. On CUDA,
  complex with `transB == ConjTrans` and nrhs 1 is excluded, because cuSPARSE leaves C unwritten.
  On CUDA, cdouble N/N with one column is excluded, because it segfaults the host inside cuSPARSE.
  On NETLIB, any transpose is excluded, because netlib throws.
- **Keys:** `transA:exact transB:exact m:log nrhs:log batch:log`. There is no nnz key, since the
  per-item nnz is in device memory. 500 rows per table. A `cpu` table is transcribed, because a CPU never borrows.
- cuSPARSE's default algorithm is not bit-reproducible, so the tests identify a pinned vendor by its trace line.
- Known defects #13 and #17.

### syev {#phase-5-syev}

- **Families:** `cta` (`syev_cta`), `cta_fused`, `jacobi` (`syev_jacobi_cta`), `blocked`, `two_stage`
  and `vendor`, in tie order.
- **`can_run`:** native needs a non-NETLIB backend, a GPU, a square A and n ≥ 1. `cta`, `cta_fused`
  and `jacobi` also need n ≤ 32 and sub-group 32. `blocked` and `two_stage` need batch ≥ 1. A
  non-square A throws `invalid_argument`.
- **Keys:** `jobz:exact n:log:3 batch:log`. `uplo` is not a key, because the large-n drivers mirror
  Upper into Lower. Grid: n 8|9, 24|25, 32|33, 256|257, 320|321, 448|449, 512|513, 1024|1025. 370 rows.
- **Transcribed pattern, jobz=V:**
  - float: `jacobi` for n ≤ 8, `cta_fused` for 9 to 32, `blocked` to 448, `two_stage` to 1024, then `vendor`;
  - double: `jacobi`, `blocked` to 448, then `vendor`;
  - cfloat: `cta_fused` for n ≤ 8, `cta` to 32, `blocked` to 512, then `vendor`;
  - cdouble: `cta` to 24, `vendor` for 25 to 32, `blocked` to 256, then `vendor`.
  - jobz=N goes to `two_stage` above 320 in every dtype.
- ROCm has no syev table and borrows the sm tables (untested).

### gesvd

- **Families:** `jacobi` (`gesvdj_cta`), `cta`, `blocked`, `vendor`. The jobs are canonicalised once,
  Thin to All where they coincide.
- **`can_run`:** native needs a GPU and `m, n, batch >= 1`.
  - `jacobi`: sub-group 32, not Hermitian, and `max(m, n) <= 64` (32 for cdouble with vectors).
  - `cta`: sub-group 32, `max(m, n) <= 32`, not thin, and `herm ? m == n : real T`.
  - `blocked`: `herm ? m == n && Lower : real T`.
  - `vendor`: `has_vendor` alone. cuSOLVER refuses `max(m, n) > 32`, non-packed batches and thin
    factors, so Auto reaches the vendor for those shapes only where no native family runs.
- **Keys:** `herm:exact vec:exact m:log:1.5 n:log:1.5`. Grid 32|33 and 64|65. 2,025 rows.
- Known defects #14 and #15.

### 12.9 Level-3 four (symm, syrk, syr2k, trmm) {#level-3-four-symm-syrk-syr2k-trmm}

The four level-3 ops use the same structure as the op files above, with `NoFields` families. The
vendor family calls `backend::<op>_vendor` under `has_library<B>(spec.vendor)`, not the public entry,
so there is no recursion. Float-only kernels are instantiated only in `if constexpr` arms.

| op | candidates | keys (rows) | Auto (sm_89 and sm_120 transcription) | last resort |
|---|---|---|---|---|
| syrk | float `gram, triangular, vendor`; double `gram, vendor` | `form:exact trans:exact n:log:2 k:log batch:log` (18,450) | n ≤ 128 `gram`; above, `triangular` when squareish or (n ≥ 257, k ≥ 8, batch·T(T+1)/2 ≥ 160, T = ceil(n/128)), else `vendor`; ConjTrans `vendor` first; double above 128 `vendor` | `triangular, gram, vendor` |
| syr2k | float `triangular, vendor`; double `vendor` | `n:log:2 k:log batch:log` (455) | float batch ≥ 2 `triangular`; batch 1 `vendor`; double `vendor` | `triangular, vendor` |
| symm | `expand, vendor` (float, double) | `form:exact m:log n:log batch:log` (5,292) | float `expand` iff sq and (batch ≥ 4 or max(m, n) ≥ 256); double `vendor` | `expand, vendor` |
| trmm | `triangular, expand, vendor` (all four dtypes) | `side:exact order:log:2 q:log batch:log` (360) | Left `triangular`; Right `expand` | `expand, triangular, vendor` |

**Shared terms:**
- Native families run only on CUDA (ROCm and the host stay on the vendor), on a GPU, with homogeneous
  operands, extents and batch ≥ 1, and `batch <= 65535`. The batch is the grid z dimension, and 65536
  throws at launch.
- The vendor family needs `has_vendor` and refuses a heterogeneous operand on every backend.
- syrk, syr2k, symm and trmm return a no-op event for an empty problem.
- `form` (symm and syrk) is sq when `2·min ≥ max`. Otherwise it is tall when `a > 2b` and wide when it is not, with `a` and `b` as in `form_of`
  (for symm, C's rows and columns).

**Per op:**
- **syrk.** `gram` is `syrk_gram_tiles`, for n ≤ 128 and float and double. Above 128 a pin throws,
  because the kernel would answer wrongly. It also needs `max_wg >= gram_threads(n)`. `triangular`
  is float only, and needs `max_wg >= 256` and `T(T+1)/2 <= 65535` tiles. ConjTrans is passed as
  Trans, and the `trans` key keeps real ConjTrans on the vendor.
- **syr2k.** `triangular` is float only, with the same `max_wg` and tile terms, and `transA != ConjTrans`.
  Real ConjTrans stays on the vendor.
- **symm.** `expand` is the mirrored expansion into a queue workspace, then the public gemm. It needs
  `max_wg >= 256` and `expansion_fits(q, k, batch, bytes)`. `BATCHLAS_EXPAND_MAX_BYTES` lowers the
  limit. symm does not read `BATCHLAS_EXPAND_ROUTE`.
- **trmm.** `triangular` is Side::Left only, with `max_wg >= 256` and `ceil(m/tile_m) * ceil(q/128) <= 65535`
  tiles in grid y. `expand` is the expansion plus the public gemm at beta 0, and it needs
  `max_wg >= 256` and `expansion_fits`. `BATCHLAS_TRMM_ROUTE=vendor` means the `cublas?trmm` loop,
  which is what `expand` used to mean.

## 13. Design decisions

| Topic | Chosen | Rejected |
|---|---|---|
| Source of speed | Measured or transcribed tables per (op, dtype, device) | Hand-written device rules; first-call autotuning; a fitted cost model |
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
| Tuner grid | A coarse log grid, plus bisection where neighbours disagree | A fixed grid only; recording real workloads |
| CPU backend | Tuned like any device (`*.cpu.txt`) | Borrowing from a GPU |
| Pilot op | potrf | gemm first; a smaller op first |
