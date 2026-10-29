# Tuning data layout {#design_tuning_data}

> **Status:** current · checked 2026-10-29

The only normative page for tables, ledgers, tiers, staleness and generation: two files per
(op, device), one source of truth. The engine that writes the ledgers is @ref design_tiered_tuning.

| File | Role | Written by | Stored |
| --- | --- | --- | --- |
| `benchmarks/results/tuning/ledger/<op>.<device>.jsonl` | every measurement and every untimed opinion: the only truth | `batchlas_tune` (append) | Git LFS |
| `tuned/<op>.<device>.txt` | the table the library loads | `scripts/sweep_to_table.py`, from the ledger only | plain text, embedded at build time |

The key and candidate vocabulary lives only in `src/ops/<op>/choice.hh`. The tuner copies it into
each run line of the ledger; nothing else spells it.

**Shipped set (2026-10-29):** 59 tables. sm_120: 19 tables of `deep` rows (plus `transcribed`
rows where no deep row lies near, beyond the 2048 dimension cap for example) and 10 untimed ones
(hemm, her2k, herk, potri, potrs, sygst, sygvd, sysv, sytrf, sytrs). sm_89: 29 tables, 9 with
measured rows (geqrf, hemm, potrf, potri, potrs, sygst, symm, sysv, sytrs), the rest untimed
(`transcribed`). cpu: `spmm.cpu.txt`, `transcribed`.

## The ledger file {#tuning-data-ledger-file}

One JSONL file per (op, device). Every line carries `"dtype"`, except `rehash`. Line kinds:

- `run`: one per (run id, dtype): host, device, device name, `batchlas` (commit), argv, date,
  tier, `keys` (the `# keys:` text from `choice.hh`), `candidates` (`|`-joined spellings in
  candidate order), `protocol` (the per-cell protocol: `lean-cold`, operands out of L2 before every
  timed run; `lean` = the same race L2-warm; `full` = the fixed warm-up engine
  before it; absent in ledgers written before the field), `guard` (the guard backend: `nvml`,
  `nvidia-smi` or `off`; absent before the field), `prior` (calibrate runs only: the borrowed table as
  `<device>@<header date>`; table generation ignores it), `grid` (the log axes' full lattice the run planned
  on, `n=1,2,4 batch=8,16`; `tune_replay --ledger` reads it; absent before the field). A later `run` line with the same (run id, dtype) replaces the earlier one.
- `cell`: one measurement of one cell (`run_id`, `key`, `tier`, `round`, `date`, `ranked`, `cands`, per-candidate
  `h/s/r/m/lo/hi/n`, and `v<i>: 0` for a candidate timed but never verified in that cell: outside the
  lean race's tie set, already verified in its size decade; absent = verified, as in every older
  ledger; table generation keeps it in the row). Diagnostics that table generation ignores: wall
  seconds by phase (`ph_setup`, `ph_warm`, `ph_timed`, `ph_verify`, `ph_ld`, `ph_guard`, `ph_ipc`,
  `ph_spawn`, `ph_start`, plus `ph_mode` and `ph_arms`), summed over every try and child, and NVML
  clock metadata (`clk_n`, `clk_capped` with reason 0x4, `clk_mhz`, `throttled: 1` when more than 25%
  were capped, `throttle_reraces`).
- `audit`: one fresh-process audit (the carve-out check): `key`, `verdict`, `fresh_ms`, `warm_ms`,
  `audit_s` (its wall seconds). ld audit verdicts are cell statuses, not audit lines.
- `rehash`: `{"kind":"rehash","family":F,"from":<stored hash>,"to":<hash now>,"at":<commit>,"proof":...,"why":...}`
  for the op and device, no dtype ("Kernel hashes" below).

Readers take the lines in file order. Ties in the best-record rule ("later line") are therefore
judged by file order.

Tiers, best first: `deep`, `coarse`, `calibrate`, `preview`, `custom`, `converted`, `prior`,
`transcribed`.

- `calibrate`: a measured tier, probes and refinement of `--tier calibrate`; bracketed like preview
  (2 lattice steps).
- `converted`: medians from the old forced-route sweeps (sm_89 potrf). Timed, but measured on
  old kernels.
- `prior`: a calibrate run's copy of the borrowed table's nearest row at a grid cell it did not
  measure, untimed (`-`), under the calibrate run's id.
- `transcribed`: the old router's order, untimed (`-`).
- `converted`, `prior` and `transcribed` carry no kernel hash and are exempt from the freshness rule.
  They compete only among themselves for a cell, so one still stands where the measured record's
  ranking is empty. A measured row within 2 lattice steps on every log key (equal exact keys)
  removes a converted or transcribed row from the table, kept or not; a prior row does too. A prior
  row gives way to a measured record with its own key, or to a coarse or deep row within 2 lattice
  steps, so the copies fill only a calibrate run's gaps (@ref design_tiered_tuning, "Engine: calibrate tier from a borrowed table").

Writers append under `flock(LOCK_EX)`, one `write()` per line, and repair a torn last line under
the same lock. Several dtype jobs and processes share one file. A reader ignores a truncated last
line with a warning, so a cell is either fully recorded or absent.

**Which record counts.** Per cell, the best current record: a measured record of a `lean-cold` run
first (another protocol's ranks below it whatever its tier, and the tuner re-measures its cell), then
by tier (order above), then the later line. A `cell` record is:

- **current** when every candidate's family hash matches the source tree. A candidate stored as
  `skipped` is not compared (a change to what can run goes through the hashed `<op>.cc`); one
  skipped with reason `dominated:<key>` was never tried, so its family is stale whatever its hash;
- **partly stale** when only some hashes changed, or a family appeared that the record never raced:
  the next run re-races just those candidates against the stored winner and runner-up, at the
  stored tier, and appends a merged record;
- **stale** when the stored winner's hash changed or its family was removed: the cell is re-measured
  at the running tier.

A record in which every candidate is `error` counts as no record. `best_records` (`ledger.cc` and
`sweep_to_table.py`) applies this one rule, so planning, refinement, `--status` and table
generation agree.

**Dirty runs.** A `run` whose `batchlas` ends in `-dirty` is left out with its cells, with a
warning, by every reader (C++ `read_ledger`, `sweep_to_table.py`). `batchlas_tune` refuses a
tree with uncommitted changes to tracked files (`git status --porcelain` of `--repo`) unless
`--allow-dirty` is given; such a run is stamped `<sha>-dirty` and never reaches `tuned/`.

## Kernel hashes {#tuning-data-kernel-hash}

Each candidate of a cell stores its family's hash. A family hashes its spec's `kernel-sources`
files (common, its own, `kernel-deps`), hash v2:

- sha256 over the lines `"<sha256(normalized file)>  <path>\n"` in list order, first 8 hex.
- normalized: `/* ... */` blocks removed (an unterminated one runs to the end), then `//` to the
  end of the line, then every whitespace byte. Naive: string literals are not special.
- One rule in two implementations: `kernel_hash` in `tools/tune/tune_core.cc` and
  `scripts/sweep_to_table.py` (`.github/ci/check_tuned_tables.py` imports the latter). The shared
  vector `tests/data/kernel_hash_v2/` is checked by `tune_tests`, `sweep_to_table.py --self-test`
  and `check_tuned_tables.py --self-test`.
- Table staleness is one function, `sweep_to_table.stale_kernel_warnings`: the header's
  `kernels=`/`family_kernels=` against the sources now. A table whose `tiers=` counts no
  deep, coarse, calibrate, preview or custom row (all converted/prior/transcribed) is never stale,
  whatever hashes its header names; a missing or unparsable `tiers=` is checked.
- Shared device headers (`src/sycl/kernel_attrs.hh`, `src/sycl/sscp_target.hh`,
  `src/kernels/common/sg_partition/` and `include/batchlas/util/group-collectives.hh`) sit in a spec's common section, once,
  wherever its sources include them; `tune_tests` (`SharedKernelHeadersAreCommonInEverySpec`)
  follows the includes and checks it.

A comment or whitespace edit therefore leaves every record current. Any other edit to a hashed
file stales the records of every family that hashes it.

A `rehash` line is the only way a stored hash is reinterpreted: readers (C++ and Python) treat a
candidate hash `from` of family `F` as `to`. Its `proof` says why that is sound:

- **`normalized`**: the v1 (raw-file) migration; `F`'s files at the producing commit equal today's
  modulo comments and whitespace.
- **`sample:<run id>`** (a real kernel edit): `sweep_to_table.py --rehash-sample RUN_ID --op OP
  --device DEV [--family F...]` compares every cell of a run measured with today's kernels against the
  best record under the old hashes. A cell agrees when the winners match or the old winner is within
  the 3% tie band; a disagreeing cell is settled by a deep record with today's kernels. All settled:
  one line per stale family is appended; otherwise nothing is written and the unsettled cells are
  printed for `batchlas_tune <op> --tier deep --remeasure-keys FILE --only-named`.
- **`moved:<base commit>`** (a file move): `sweep_to_table.py --rehash-moved BASE [--op OP]
  [--device DEV]` proves each HEAD path is its base path under a git rename with the same basename,
  equal after normalization once includes are resolved to repo paths. A content edit, a changed list
  length, an include that now resolves elsewhere, or an include form it does not model
  (`__has_include`, `#include_next`, macro includes) gives no proof. It refuses uncommitted changes
  to the hashed files and is idempotent.

A sample must be stratified: every dtype and exact-key combination of the old records and the range
of every log key, at least 64 cells. `sweep_to_table.py --emit-sample FILE --op OP --device DEV
[--cells N]` writes one; `batchlas_tune <op> --tier custom --dtype ... --remeasure-keys FILE`
measures exactly those cells (a cell above `--cap-gib` is still skipped, which can leave an exact-key
combination missing).

`-dirty` runs are never rehashed. Records that no proof covers stay stale, and a stale family
must be re-measured.

## The table file {#tuning-data-table-file}

```
# op=potrf device=sm_120 batchlas=209d9fb8 kernels=0d38569e family_kernels=tiny:0bf1ed4c,... date=2026-10-08
# source=ledger:benchmarks/results/tuning/ledger/potrf.sm_120.jsonl tiers=deep:1345,coarse:0,...
# keys: dtype:exact uplo:exact n:log:3 batch:log
dtype=float uplo=L n=1 batch=128 | tiny 0.007526 | lpanel:panel=8 0.007722 | ... # deep
```

- `dtype` is the first key, exact, in the order float, double, cfloat, cdouble. The loader
  (`select::parse_tables`) splits the file into one table per dtype without the `dtype` key.
  Selection (`select::choose`, nearest row, borrowing, the last resort) is unchanged.
- The header's `batchlas` and `date` come from the newest run (date, then run id) over all
  dtypes. `kernels` is the op's hash over its whole kernel-sources block; `family_kernels` holds
  the hashes the table was derived with.
- A file not named `<op>.<device>.txt` is rejected by the embed step and skipped, with a warning,
  in `BATCHLAS_TUNED_DIR`, which reads the same names. An override file replaces the built-in
  table dtype by dtype: a dtype with rows in it uses them, a dtype it lacks keeps the built-in rows.

## Generation and the check {#tuning-data-generation-and-check}

`sweep_to_table.py` has one input, the ledger:

```
scripts/sweep_to_table.py                      # regenerate every tuned/<op>.<device>.txt from its ledger
scripts/sweep_to_table.py --op potrf --device sm_120 --out DIR
scripts/sweep_to_table.py --check              # CI: re-derive every table byte for byte
scripts/sweep_to_table.py --self-test
scripts/sweep_to_table.py --rehash-sample RUN_ID --op gemm --device sm_120   # "Kernel hashes"
scripts/sweep_to_table.py --rehash-moved BASE                                # "Kernel hashes"
```

- **Rows:** each cell takes its best current record; a lower-tier row is emitted only where no
  higher-tier row lies within the lower tier's lattice spacing under `select::choose`'s distance,
  and every row ends in its tier comment (`# deep`). Output is byte-reproducible.
- **Header:** `tiers=` counts every tier in the order above, zeros included.
- **Keys and candidates** come from each dtype's newest run line, never from Python. A cell key
  field that is not a table key (trsm's `uplo`, `diag`) is a hidden grid axis: it is projected
  away, and a dtype whose cells vary it is refused (an A/B run is raw data, not a table).
- **Freshness** in `--check` is judged with the hashes in the table header. The check therefore
  proves that the table equals the function of the ledger, whatever the sources say today.
- **Stale kernels** (header hashes against `tools/tune/<op>_spec.cc`'s kernel-sources block) are
  a warning, never a CI failure or a configure-time check: a stale table stays in use until the op is
  retuned. `sweep_to_table.py --check` and `.github/ci/check_tuned_tables.py` print the same warnings
  from `sweep_to_table.stale_kernel_warnings` ("Kernel hashes" above).
- **CI** (`.github/workflows/ci.yml`, "Tuned tables equal their ledgers") fetches only the
  ledger LFS objects, cached by their pointers' oids, and runs `--check` and `--self-test`.

The library side is tested by `tests/tuned_tables_tests.cc` (suite `TunedTables`): the embedded set
equals `tuned/` byte for byte, keys and spellings match `choice.hh`, timed rows obey the 3% tie
order, every op ships every dtype on sm_89 and sm_120, and the posv, trsm, gemm, gemv, geqrf and
orgqr tables cover exactly the `choice.hh` grid.

## License

SPDX-License-Identifier: MIT
