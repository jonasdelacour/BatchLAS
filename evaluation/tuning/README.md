# BatchLAS tuning harness (bottom-up) {#tuning_harness}

> **Status:** current. Reference for the scripts that regenerate `include/batchlas/tuning_params.hh`.

The harness grid-searches the constants in `include/batchlas/tuning_params.hh` by running the
benchmark executables in `build/benchmarks/`. What the constants are, their size buckets, the
`BATCHLAS_TUNE_*` overrides, knobs that shadow each other and the tune-through-the-consumer rule:
@ref perf_tuning. Kernel selection tables are a different tool, @ref tune_tool_readme.

## Workflow

```sh
scripts/b stedc_benchmark sytrd_blocked_benchmark ormqr_blocked_benchmark \
          gesvd_blocked_benchmark sb2st_hh_benchmark syev_benchmark     # -DBATCHLAS_BUILD_BENCHMARKS=ON
python3 evaluation/tuning/tune.py --backend CUDA --type float --validate --skip-missing
python3 evaluation/tuning/tune.py --backend CUDA --type float --skip-missing \
    --space evaluation/tuning/spaces/default.json --out build/tuning/profile.json
python3 evaluation/tuning/sensitivity.py build/tuning/profile.json
python3 evaluation/tuning/generate_tuning_header.py \
    --profile build/tuning/profile.json --out /tmp/tuning_params.hh
diff include/batchlas/tuning_params.hh /tmp/tuning_params.hh
```

Then A/B each candidate at its consumer with the `BATCHLAS_TUNE_*` override and port the winning
`inline constexpr` values into the committed header by hand ([regenerating the
header](../../docs/perf/tuning.md#tuning-regenerating-the-header)). Never overwrite the header: shipped
constants outside the grid would be downgraded. There is no CMake target that writes it. A full
default-space run takes about 12 minutes on CUDA/float (RTX 4090); to shorten it, cut large-n cases
from the space, not iteration counts.

## `tune.py`

Runs every bench of a space file over its grid and writes a JSON profile.

| flag | default | meaning |
|---|---|---|
| `--backend` | required | backend passed to the benchmarks (`CUDA`, `ROCM`, `NETLIB`, `MKL`) |
| `--type` | required | scalar type passed to the benchmarks |
| `--space` | `evaluation/tuning/spaces/default.json` | space file |
| `--build-dir` | `<repo>/build` | where `benchmarks/<exe>` is found |
| `--out` | `<build-dir>/tuning/profile.json` | profile path |
| `--warmup`, `--min-time`, `--min-iters`, `--max-iters` | 2, 25.0, 1, 20 | passed to each benchmark |
| `--topk` | 10 | candidates kept per bench in `top` |
| `--validate` | off | one cheap invocation per bench (executable exists, metric parses), then exit |
| `--skip-missing` | off | skip benches whose executable is missing |
| `--skip-failed` | off | skip benches that fail to run. Avoid: a broken bench becomes a silent omission in a profile that looks complete |
| `--verbose` | off | print each invocation |

Profile: `meta` (`generated_at`, `hostname`, `platform`, `backend`, `dtype`, `build_dir`) and
`results[]`, one per bench: `bench`, `exe`, `metric`, `direction`, `arg_spec`, `pre_tune` (winners),
`best` (the set best across cases; only combos legal for every case), `top`, and `per_case_best`
(the per-case winners the bucketed header uses).

## Space files

| file | purpose |
|---|---|
| `spaces/default.json` | benches the generator reads: `stedc`, `sytrd_blocked`, `ormqr_blocked`, `gesvd`, `sb2st`, `syev`, `sy2sb`. A bench outside this list is measured and then dropped |
| `spaces/unwired.json` | parameters with no constant yet (`steqr_cta`, `syev_two_stage_kd`, `sb2st_block`); a profile for a human to read |

Each entry of `spaces[]`:

| key | meaning |
|---|---|
| `bench` | name the generator looks up |
| `exe` | executable (default `<bench>_benchmark`) |
| `metric`, `direction` | CSV column to read; `min` (default) or `max` |
| `arg_spec` | positional benchmark arguments, in order |
| `cases` | problem sizes, each with `fixed` args and a case-local `tune` grid; grids are searched as a union, each case sweeping only its own |
| `pre_tune` | optional phases `{"params": {...}, "cases": [...]}` run first; winners become fixed params of every case |
| `env` | optional `{param: ENV_VAR}`: passed as environment variables, not positional args (must not be in `arg_spec`); part of the measurement cache key |
| `row` | substring selecting the benchmark row. Required when an executable registers several benchmarks (`sb2st_hh_benchmark`: `BM_SB2ST_HH_BACK`); the harness refuses to guess |

A grid must contain the value the header ships, or a retune can "win" against a grid that never held
the incumbent ([how the constants were measured](../../docs/perf/tuning.md#tuning-how-the-constants-were-measured)).

## `sensitivity.py`

```sh
python3 evaluation/tuning/sensitivity.py build/tuning/profile.json [--noise 1.0]
```

For each (bench, case, parameter) it holds the other parameters fixed and reports how far the metric
moves across that parameter's values; below `--noise` percent (default 1.0) the parameter counts as
inert and is a candidate to drop from the space. Pass the profile, not a sweep log: env-only
parameters never appear in the benchmark's CSV columns.

## `generate_tuning_header.py`

```sh
python3 evaluation/tuning/generate_tuning_header.py --profile PROFILE.json --out HEADER.hh \
    [--fallback-<knob>-<bucket> N ...]
```

Writes a full `tuning_params.hh` from the profile's `per_case_best`, one constant per bucket
(`tiny`, `small`, `medium`, `large`, `xlarge`). Buckets no case reaches take the `--fallback-*`
value, else the bench's `best`. Fallback flags exist for `ormqr-block-size`, `gebrd-block-size`,
`sytrd-block-size`, `sytrd-fuse-panel-update`, `stedc-{recursion-threshold, merge-variant,
threads-per-root, wg-multiplier}` (each per bucket) and `latrd-lower-panel-wg-hint` (one value).

Coupling between benches:

- When a `syev` entry exists, `SYTRD_BLOCK_SIZE_*` comes from syev's coupled `nb`, so syev's `nb`
  grid must cover sytrd_blocked's. syev tunes `nb`, the LATRD wg hint and the fused panel update as
  one tuple against end-to-end time.
- stedc's constants feed syev's and gesvd's tridiagonal solves: tune stedc first.
- STEDC cases start at n = 64; the default space pre-tunes `recursion_threshold` at n = 64 only.

## Promoting a parameter out of `unwired.json`

1. Add `<NAME>_{TINY..XLARGE}` constants and a `*_for_n` accessor with a `BATCHLAS_TUNE_*` override
   to `tuning_params.hh`, and mirror the accessor in the generator's template.
2. Teach `generate_tuning_header.py` to derive the constants from the bench.
3. Make the production call site read the accessor instead of its hardcoded default; without this
   step the constant is never used.
