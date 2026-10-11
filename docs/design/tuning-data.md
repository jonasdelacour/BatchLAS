# Tuning data layout {#design_tuning_data}

> **Status:** implemented in the tuning PR (2026-10-11), decided by the maintainer on 2026-10-09
> ("Full: 2 files per op/device"). Replaces the per-dtype tables, the per-run ledger directories,
> the transcriber CSVs and the converted routing sweeps. Engine: @ref design_tiered_tuning.

Two files per (op, device), one source of truth, one vocabulary.

| File | Role | Written by | Stored |
| --- | --- | --- | --- |
| `benchmarks/results/tuning/ledger/<op>.<device>.jsonl` | every measurement and every untimed opinion: the only truth | `batchlas_tune` (append) | Git LFS |
| `tuned/<op>.<device>.txt` | the table the library loads | `scripts/sweep_to_table.py`, from the ledger only | plain text, embedded at build time |

The key and candidate vocabulary lives only in `src/ops/<op>/choice.hh`. The tuner copies it into
each run line of the ledger; nothing else spells it.

## The ledger file {#tuning-data-ledger-file}

One JSONL file per (op, device). Every line carries `"dtype"`, except `rehash`. Line kinds:

- `run`: one per (run id, dtype): host, device, device name, `batchlas` (commit), argv, date,
  tier, `keys` (the `# keys:` text from `choice.hh`), `candidates` (`|`-joined spellings in
  candidate order). A later `run` line with the same (run id, dtype) replaces the earlier one.
- `cell`: one measurement of one cell (`key`, `tier`, `round`, `ranked`, per-candidate
  `h/s/r/m/lo/hi/n`).
- `audit`: the ld audit verdicts.
- `rehash`: `{"kind":"rehash","family":F,"from":<stored hash>,"to":<hash now>,"at":<commit>,"proof":...,"why":...}`
  for the op and device, no dtype ("Kernel hashes" below).

Readers take the lines in file order. Ties in the best-record rule ("later line") are therefore
judged by file order.

Tiers, best first: `deep`, `coarse`, `preview`, `custom`, `converted`, `transcribed`.

- `converted`: medians from the old forced-route sweeps (sm_89 potrf). Timed, but measured on
  old kernels.
- `transcribed`: the old router's order, untimed (`-`).
- Both carry no kernel hash and are exempt from the freshness rule. They compete only among
  themselves for a cell, so one still stands where the measured record's ranking is empty. A
  measured row within 2 lattice steps on every log key (equal exact keys) removes them from the
  table, kept or not.

Writers append under `flock(LOCK_EX)`, one `write()` per line, and repair a torn last line under
the same lock. Several dtype jobs and processes share one file.

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
- One rule in three implementations: `kernel_hash` in `tools/tune/tune_core.cc`,
  `scripts/sweep_to_table.py` and `.github/ci/check_tuned_tables.py`. The shared vector
  `tests/data/kernel_hash_v2/` is checked by `tune_tests`, `sweep_to_table.py --self-test` and
  `check_tuned_tables.py --self-test`.

A comment or whitespace edit therefore leaves every record current. Any other edit to a hashed
file stales the records of every family that hashes it.

A `rehash` line is the only way a stored hash is reinterpreted: readers (C++ and Python) treat a
candidate hash `from` of family `F` as `to`. Its `proof` says why that is sound:

- **`normalized`** (the migration): written per (family, stored hash) only after checking from
  git, at every commit that produced the hash (the run line's `batchlas`), that (1) the v1 hash
  (the same manifest over raw files) of `F`'s sources at that commit equals the stored hash, and
  (2) `F`'s source list at that commit equals today's and each file is equal to today's modulo
  comments and whitespace.
- **`sample:<run id>`** (a real kernel edit): a run measured with today's kernels agrees with the
  old records. `scripts/sweep_to_table.py --rehash-sample RUN_ID --op OP --device DEV [--family F...]`
  compares every cell of that run with the best record usable under the old hashes. A cell agrees
  when the winners match, or when the old winner is within the 3% tie band of the sample's winner
  in the sample's own times. A disagreeing cell is settled once a deep record measured with
  today's kernels holds it (it then outranks the old record by date). If every cell agrees or is
  settled, one line per stale family is appended (`"why": "<n> cells agree, <m> re-measured at
  deep"`, `"at"`: HEAD); otherwise nothing is written and the unsettled cells are printed as
  `--remeasure-keys` lines, to be re-measured at deep.

A sample is stratified: it covers every combination of dtype and exact keys the old records hold,
and the range of every log key. `--rehash-sample` refuses a sample of fewer than 64 cells or one
that misses an exact-key combination. The tuner records a sample as tier `custom`:
`batchlas_tune <op> --tier custom --dtype ... --remeasure-keys FILE` measures exactly the named
cells, every candidate, with no lattice and no refinement.

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
  in `BATCHLAS_TUNED_DIR`, which reads the same names.

## Generation and the check {#tuning-data-generation-and-check}

`sweep_to_table.py` has one input, the ledger:

```
scripts/sweep_to_table.py                      # regenerate every tuned/<op>.<device>.txt from its ledger
scripts/sweep_to_table.py --op potrf --device sm_120 --out DIR
scripts/sweep_to_table.py --check              # CI: re-derive every table byte for byte
scripts/sweep_to_table.py --self-test
scripts/sweep_to_table.py --rehash-sample RUN_ID --op gemm --device sm_120   # "Kernel hashes"
```

- **Keys and candidates** come from each dtype's newest run line, never from Python. A cell key
  field that is not a table key (trsm's `uplo`, `diag`) is a hidden grid axis: it is projected
  away, and a dtype whose cells vary it is refused (an A/B run is raw data, not a table).
- **Freshness** in `--check` is judged with the hashes in the table header. The check therefore
  proves that the table equals the function of the ledger, whatever the sources say today.
- **Stale kernels** (header hashes against `tools/tune/<op>_spec.cc`'s kernel-sources block) are
  a warning. A kernel edit does not fail CI; it makes the table stale until the op is retuned.
  `check_tuned_tables.py` reports the same per op; there is no configure-time check.
- **CI** (`.github/workflows/ci.yml`, "Tuned tables equal their ledgers") fetches only the
  ledger LFS objects, cached by their pointers' oids, and runs `--check` and `--self-test`.

The library side is tested in C++. `tuned_tables_tests` checks that every embedded table's
`# keys:` line equals `dtype:exact` followed by the op's `choice.hh` keys, and that every spelling
is one of that dtype's candidates.

## Migration (one-off) {#tuning-data-migration}

`scripts/migrate_tuning_data.py`, committed with the data and deleted in the next commits of the
same PR (the transcribers went the same way), wrote the new files from the old ones:

- **Deep run (sm_120):** the 183 run files merged into the 19 per-op ledgers: every line, in
  file-name-then-line order per old directory, with `dtype` added. 146,544 lines.
- **Untimed rows (sm_120):** rows tagged `# transcribed` in the shipped tables became
  `transcribed` cells under run `transcribed-<sha>`, the sha of the transcribed table they came
  from (`aad906cb^`).
- **sm_89 and cpu transcribed tables:** became `transcribed` cells, with run lines carrying the
  table's keys, the sm_120 run's candidates and `batchlas` set to the transcribed sha.
- **sm_89 potrf converted rows** became `converted` cells (the rows' `# noisy` marks, 1 row,
  were dropped).
- **Rehash:** 59 verified `rehash` lines (`proof: normalized`), every family of 18 ops. gemv's were refused: its
  records predate real edits to `src/ops/gemv/gemv.cc` and `src/sycl/device_scalar.hh`, and its
  full re-run (`20261009T042647-threadripper02-422873`) was `-dirty`. The gemv sm_120 table
  holds only transcribed rows until gemv is re-measured.
- **Check:** each new table, regenerated with today's hashes and restricted to one dtype, has
  exactly the rows of the old per-dtype table (keys, entries, times, tier tags), gemv sm_120
  excepted.

Deleted afterwards:

- the per-dtype tables and the per-run ledger directories;
- `tuned/transcribed/*.csv` and the potrf sweeps under `benchmarks/results/routing/`
  (`sm120_potrf_sweep{,_edges}.jsonl` and their campaign files, `sm89_potrf_archive.jsonl`);
- the converter, transcriber and raw-tuner inputs of `sweep_to_table.py`, with its Python
  `OpSpec`, and `cmake/BatchLASTunedStaleness.cmake`.

Kept, because a page or a test cites them (no table is generated from them): the old raw trsm
tuner output `benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl` (`docs/perf/trsm.md`),
`routing/sm120_posv_sweep.jsonl` (`docs/perf/potrf.md`, `docs/perf/blackwell.md`) and
`routing/sm120_potrf_phase2_gate.csv` (`tune_tests`).

The deleted files remain in git history at the merge base of this PR.

## License

SPDX-License-Identifier: MIT
