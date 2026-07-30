# Code-volume and duplication audit

## Scope

This is a read-only, repository-wide scan of the C++, CMake, Python, plotting, evaluation, benchmark, and test code. The goal is to identify opportunities to reduce verbosity and repeated implementations without merging algorithmically distinct performance paths.

No production code was changed as part of this audit.

## Executive summary

The most valuable opportunities are:

1. Remove duplicate benchmark targets in [benchmarks/CMakeLists.txt](benchmarks/CMakeLists.txt#L1-L68).
2. Consolidate shared command-line and numerical helpers used by the accuracy benchmarks.
3. Share benchmark execution and CSV parsing between [evaluation/perf_eval.py](evaluation/perf_eval.py) and [evaluation/tuning/tune.py](evaluation/tuning/tune.py).
4. Factor the repeated tuning-header command arguments in [cmake/BatchLASGeneratedHeaders.cmake](cmake/BatchLASGeneratedHeaders.cmake#L98-L260).
5. Make the tuning-header generator data-driven rather than repeating per-parameter bucket logic.
6. Share generic Cartesian-product and filtering primitives between [include/util/minibench.hh](include/util/minibench.hh) and [include/util/miniacc.hh](include/util/miniacc.hh).
7. Consolidate identical CUDA/ROCm dispatch and repeated environment-variable parsing where backend semantics permit it.

The recommendations below are ordered roughly by likely maintenance benefit and confidence.

## Priority 1: high-value shared utilities

### Current-tree correction: benchmark target list is already unique

The earlier scan suggested duplicate `gesvd_cta_acc` and `gesvd_blocked_acc` entries. A fresh inspection before refactoring found that [benchmarks/CMakeLists.txt](benchmarks/CMakeLists.txt#L1-L68) currently contains each target exactly once. No target-list change was made; removing entries would have been incorrect.

### 2. Accuracy benchmark option parsing is repeated three times

**Impact:** High  
**Confidence:** High

The following files independently implement very similar `Options`, string helpers, option-value extraction, argument parsing, help text scaffolding, sampling controls, backend/type selection, seed handling, and output configuration:

- [orthogonality_accuracy.cc](benchmarks/orthogonality_accuracy.cc#L26-L104)
- [eigensolver_accuracy.cc](benchmarks/eigensolver_accuracy.cc#L19-L155)
- [steqr_accuracy.cc](benchmarks/steqr_accuracy.cc#L22-L130)

The implementations should not be forced into one identical options structure because defaults and benchmark-specific switches differ. Instead, introduce reusable parsing primitives and a common options component, for example:

- `starts_with()`
- `to_lower()`
- `get_option_value()`
- common backend/type/output parsing
- common condition-number and sampling options
- common help-text fragments

Each benchmark can retain its own options structure and add benchmark-specific fields.

**Important behavior to preserve:** different defaults for `scheme` and `cta_shift`, additional SYEVX/block-size options in the eigensolver benchmark, and implementation-specific choices in the STEQR benchmark.

### 3. Tridiagonal extraction and orthogonality residuals are duplicated

**Impact:** High  
**Confidence:** High

`extract_tridiagonal()` is effectively duplicated in [orthogonality_accuracy.cc](benchmarks/orthogonality_accuracy.cc#L108-L129) and [eigensolver_accuracy.cc](benchmarks/eigensolver_accuracy.cc#L176-L198). `orthogonality_residuals()` is also duplicated in [orthogonality_accuracy.cc](benchmarks/orthogonality_accuracy.cc#L133-L151) and [eigensolver_accuracy.cc](benchmarks/eigensolver_accuracy.cc#L160-L180).

The implementations use the same views, strides, one-dimensional SYCL kernel, diagonal/subdiagonal copies, and wait behavior.

**Recommendation:** Add a benchmark-only helper such as `benchmarks/accuracy_utils.hh` containing the common extraction and residual routines. Keep eigensolver-specific eigenvalue residual logic local, and do not move benchmark-only helpers into the public library API.

### 4. Python benchmark execution and CSV parsing are duplicated

**Impact:** High  
**Confidence:** High

[evaluation/perf_eval.py](evaluation/perf_eval.py#L1-L193) and [evaluation/tuning/tune.py](evaluation/tuning/tune.py#L1-L120) each contain versions of repository/build discovery, executable path construction, subprocess execution, temporary CSV handling, minibench CSV parsing, integer argument extraction, average-time parsing, and standard-deviation parsing.

**Recommendation:** Add a small shared module, for example `evaluation/common/benchmark_runner.py` or `evaluation/common/minibench.py`, with functions such as:

- `repo_root()`
- `default_build_dir()`
- `run_minibench()`
- `parse_minibench_csv()`
- `extract_integer_args()`

Keep caller-specific result models and tuning/scoring behavior separate.

### 5. Tuning-header command arguments are repeated in CMake

**Impact:** High  
**Confidence:** High

[cmake/BatchLASGeneratedHeaders.cmake](cmake/BatchLASGeneratedHeaders.cmake#L98-L260) contains two long, nearly identical invocations of `generate_tuning_header.py`, each with a large list of fallback arguments.

Adding or renaming a tuning parameter currently requires editing both command blocks.

**Recommendation:** Centralize the shared fallback arguments in a CMake list or helper function. Keep profile and output paths as caller-specific arguments.

### 6. Tuning-header generation repeats bucket definitions and dispatch logic

**Impact:** Medium–High  
**Confidence:** High

[evaluation/tuning/generate_tuning_header.py](evaluation/tuning/generate_tuning_header.py#L120-L330) repeats five-bucket definitions, fallback arguments, generated constants, and lookup functions for several tuning parameters. STEDC fallback resolution repeats the same pattern again in [generate_tuning_header.py](evaluation/tuning/generate_tuning_header.py#L380-L425).

**Recommendation:** Define buckets and parameters in data tables, then generate constants, fallback handling, and lookup functions in loops. Introduce a helper for bucket fallback resolution. Preserve existing generated names and threshold boundaries for compatibility.

### 7. Generic Cartesian-product logic is duplicated in minibench and miniacc

**Impact:** Medium–High  
**Confidence:** High

Both [include/util/minibench.hh](include/util/minibench.hh#L66-L105) and [include/util/miniacc.hh](include/util/miniacc.hh#L70-L110) contain list/range parsing, and both independently build Cartesian products in [minibench.hh](include/util/minibench.hh#L742-L754) and [miniacc.hh](include/util/miniacc.hh#L677-L692).

**Recommendation:** Add a type-generic internal `cartesian_product()` helper and share parsing primitives where semantics match. Preserve the distinct integer and floating-point option behavior.

### 8. CUDA and ROCm `call_backend()` implementations are structurally identical

**Impact:** Medium–High  
**Confidence:** High

The CUDA and ROCm versions of `call_backend()` in [src/linalg-impl.hh](src/linalg-impl.hh#L628-L655) both select the callable by scalar type, convert arguments, prepend the handle, and call `check_status()`.

**Recommendation:** Combine the implementation under a shared backend guard or move the generic template to a common internal section. First verify that status types, conversion helpers, and overloads are compatible in every supported backend configuration.

### 9. Environment-variable boolean parsing is repeated

**Impact:** Medium  
**Confidence:** High

Similar truth-value parsing appears in [src/extensions/sytrd_blocked.cc](src/extensions/sytrd_blocked.cc#L43-L55), [src/extensions/band_reduction.cc](src/extensions/band_reduction.cc#L56-L70), and [src/util/kernel-trace.hh](src/util/kernel-trace.hh#L40-L60). Lowercasing and string comparison are also repeated in [src/backends/gemm_variant.hh](src/backends/gemm_variant.hh#L45-L80), [src/sycl/gemm_kernels.cc](src/sycl/gemm_kernels.cc#L15-L35), and [include/blas/dispatch/env.hh](include/blas/dispatch/env.hh#L1-L45).

**Recommendation:** Provide one internal environment utility for retrieving, lowercasing, and interpreting values such as true/false. Preserve existing accepted spellings, defaults, and intentional caching behavior.

## Priority 2: worthwhile consolidation

### 10. Backend/type filtering is duplicated between minibench and miniacc

**Impact:** Medium  
**Confidence:** Medium–High

The filtering and command-line concepts in [include/util/minibench.hh](include/util/minibench.hh#L600-L730) and [include/util/miniacc.hh](include/util/miniacc.hh#L640-L670) overlap, including backend extraction and real versus complex type matching.

**Recommendation:** Share only the metadata matching primitives. Do not merge the benchmark runners, result types, or registration systems.

### 11. Explicit backend/type instantiation tails repeat across extensions

**Impact:** Medium  
**Confidence:** Medium

Several extension files repeat backend guards and real/complex type instantiation macros, including [src/extensions/steqr.cc](src/extensions/steqr.cc#L65-L90), [src/extensions/stedc.cc](src/extensions/stedc.cc#L526-L547), [src/extensions/syev_cta.cc](src/extensions/syev_cta.cc#L706-L721), [src/extensions/syev_blocked.cc](src/extensions/syev_blocked.cc#L526-L541), [src/extensions/stedc_flat.cc](src/extensions/stedc_flat.cc#L922-L940), and [src/extensions/sytrd_blocked.cc](src/extensions/sytrd_blocked.cc#L916-L931).

**Recommendation:** Add standard backend-expansion macros in [src/util/template-instantiations.hh](src/util/template-instantiations.hh), but retain local macros when supported backend/type sets differ. Keeping some local repetition can make support coverage easier to inspect.

### 12. Device BLAS CMake helpers repeat target setup

**Impact:** Medium  
**Confidence:** High

The device BLAS helper functions in [benchmarks/CMakeLists.txt](benchmarks/CMakeLists.txt#L70-L138) repeat `add_executable()`, compile definitions, and `batchlas_configure_binary_target()`.

**Recommendation:** Implement one generic helper accepting a target, source, and compile-definition list. Keep the named category wrappers if they improve discoverability.

### 13. Plot-comparison scripts duplicate CLI and execution scaffolding

**Impact:** Medium  
**Confidence:** High

[plotting/htev_compare_plot.py](plotting/htev_compare_plot.py#L400-L545) and [plotting/syev_compare_plot.py](plotting/syev_compare_plot.py#L480-L580) both implement similar `--run` handling, binary selection, CSV output, backend/type filters, warmup, size validation, benchmark execution, CSV loading, and output selection.

[plotting/bench_common.py](plotting/bench_common.py#L1-L195) already provides a suitable foundation.

**Recommendation:** Extract parser and execution helpers such as `add_common_benchmark_args()`, `validate_n_batches()`, and `run_comparison_pair()`. Pass defaults explicitly so HTEV/SYEV-specific behavior remains intact.

### 14. BANDR1 playground scripts share generation setup

**Impact:** Low–Medium  
**Confidence:** Medium–High

[playground/bandr1_evolution.py](playground/bandr1_evolution.py#L20-L105) and [playground/plot_bandr1_evolution.py](playground/plot_bandr1_evolution.py#L360-L430) duplicate driver resolution and generation options.

**Recommendation:** Share only path normalization and command construction. Keep the two CLIs separate because one is a wrapper and the other is a plotting workflow.

### 15. LAPACK wrapper dispatch repeats float/double selection

**Impact:** Low–Medium  
**Confidence:** High

The wrappers `call_lapack_steqr()`, `call_lapack_sterf()`, and `call_lapack_stedc()` in [benchmarks/steqr_accuracy.cc](benchmarks/steqr_accuracy.cc#L130-L170) repeat the same float/double dispatch pattern.

**Recommendation:** Use a small type-traits mapping or generic helper if more wrappers are added. For the current small number of functions, readability may outweigh a more abstract implementation.

### 16. `call_backend_nh()` repeats scalar-type dispatch from `call_backend()`

**Impact:** Medium  
**Confidence:** High

`call_backend()` and `call_backend_nh()` in [src/linalg-impl.hh](src/linalg-impl.hh#L628-L671) both dispatch across four scalar types. The non-handle version mainly differs in tuple construction and return behavior.

**Recommendation:** Factor only the scalar-type callable selection if doing so preserves useful compiler diagnostics. Avoid an abstraction that obscures backend errors.

### 17. Repeated local `to_lower()` implementations

**Impact:** Low–Medium  
**Confidence:** High

Lowercasing is independently implemented in [benchmarks/eigensolver_accuracy.cc](benchmarks/eigensolver_accuracy.cc#L46-L51), [benchmarks/orthogonality_accuracy.cc](benchmarks/orthogonality_accuracy.cc#L46-L51), [benchmarks/steqr_accuracy.cc](benchmarks/steqr_accuracy.cc#L45-L51), [src/backends/gemm_variant.hh](src/backends/gemm_variant.hh#L55-L80), [src/sycl/gemm_kernels.cc](src/sycl/gemm_kernels.cc#L15-L35), and [include/blas/dispatch/env.hh](include/blas/dispatch/env.hh#L1-L45).

**Recommendation:** Use one small internal ASCII-only `to_lower_ascii()` helper, while avoiding accidental public-header dependencies and locale-sensitive behavior.

### 18. Repeated backend-specialized test registration

**Impact:** Low–Medium  
**Confidence:** Medium–High

[tests/minibench_cli_tests.cc](tests/minibench_cli_tests.cc#L12-L91) repeats registration and command-line test bodies across backend-specific preprocessor branches.

**Recommendation:** Keep the guards, but reduce repeated body code with a registration macro or helper and a compile-time backend table.

## Intentional duplication to preserve

### Algorithm variants

The separate CTA, blocked, work-group, vendor, legacy, and two-stage implementations in files such as [src/extensions/steqr_cta.cc](src/extensions/steqr_cta.cc), [src/extensions/steqr_wg.cc](src/extensions/steqr_wg.cc), [src/extensions/steqr_legacy.cc](src/extensions/steqr_legacy.cc), [src/extensions/syev_cta.cc](src/extensions/syev_cta.cc), [src/extensions/syev_blocked.cc](src/extensions/syev_blocked.cc), and [src/extensions/syev_two_stage.cc](src/extensions/syev_two_stage.cc) should not be merged merely because they look similar.

They may intentionally differ in synchronization, workspace layout, tunability, benchmark comparability, or performance behavior. Consolidate only demonstrably identical scalar math, validation, workspace accounting, or dispatch plumbing.

### Explicit instantiation lists

Local backend/type instantiation lists can serve as useful documentation of supported configurations. A fully centralized registry could hide intentional exclusions or differences in scalar and workspace support. Prefer small shared macros over a wholesale registry unless build-matrix testing demonstrates that the registry remains clear.

## Suggested implementation order

1. Remove duplicate `gesvd_*` entries from [benchmarks/CMakeLists.txt](benchmarks/CMakeLists.txt#L1-L54).
2. Extract shared accuracy numerical helpers.
3. Introduce shared accuracy option and string-parsing primitives.
4. Share Python benchmark execution and CSV parsing.
5. Centralize CMake tuning-header fallback arguments.
6. Make tuning-header generation data-driven.
7. Share minibench/miniacc generic parsing and Cartesian-product utilities.
8. Consolidate CUDA/ROCm dispatch after compiling all relevant backend configurations.
9. Centralize environment parsing.
10. Apply lower-impact CMake, plotting, test, and instantiation cleanups.

## Validation guidance

Each refactoring should be validated incrementally:

- configure and build the affected CMake targets;
- run the relevant benchmark or accuracy executable with representative real and complex types where supported;
- run the minibench/miniacc CLI tests;
- compare generated tuning headers before and after the generator change;
- compare benchmark CSV schemas and plotting outputs;
- compile at least one CUDA/ROCm and one host-only configuration when changing backend dispatch.
