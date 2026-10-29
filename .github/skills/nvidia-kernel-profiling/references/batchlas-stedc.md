# Profiling STEDC

`build/benchmarks/stedc_benchmark` (`benchmarks/stedc_benchmark.cc`) takes seven positional args; a
missing one reads as 0:

| # | Arg | Values |
| --- | --- | --- |
| 1-2 | `n`, `batch` | |
| 3 | `recursion_threshold` | 32 (the CTA cap, agent-guide §11); `<= 0` takes the tables |
| 4 | `merge_variant` | `-1` Auto (tables), `0` Baseline, `1` Fused, `2` FusedCta (`include/batchlas/blas/extensions.hh`) |
| 5-6 | `threads_per_root`, `wg_multiplier` | `<= 0` takes the tuning tables; FusedCta with threads_per_root > 32 runs `StedcFusedWgMerge` |
| 7 | `algorithm` | `0` Levels, `1` Recursive |

Merge kernels are `StedcFusedCtaMerge` and `StedcFusedWgMerge` (`src/ops/syev/stedc/stedc_merge_cta.cc`),
leaves go through `steqr_dispatch` (at leaf width <= 32, `SteqrCTAKernel` in `src/ops/syev/steqr/steqr_cta.cc`). Pin the variant and threads per root
whose kernel you filter on:

```bash
mkdir -p output/profiling
ncu --set basic --kernel-name-base demangled --kernel-name regex:StedcFusedCtaMerge \
  --launch-count 1 --target-processes all --force-overwrite \
  --export output/profiling/stedc_ncu_merge \
  build/benchmarks/stedc_benchmark --backend=CUDA --type=float --warmup=5 \
  --min_iters=1 --max_iters=1 --min_time=0 256 256 32 2 32 0 0
```

Run the same args under `nsys` first: leaves and merges both carry weight, and setup kernels appear
in the summary. Measurements and rationale live in `docs/perf/stedc.md`.
