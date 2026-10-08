# Architecture and design {#architecture_index}

Design records: decisions and their rationale. Measured evidence is in @ref perf_evidence.

## Kernel selection

| Page | What you find there |
| --- | --- |
| @subpage design_flat_selection "Flat kernel selection" | Families, per-device tuned tables, `select::choose`, pins, rules R1-R8. API: @ref api_selection; inventory: @ref selection_tables. |
| @subpage design_tiered_tuning "Tiered tuning" | The tuning engine behind the tables: tiers, racing, ledgers, workers. |
| @subpage md_docs_2extending "Extending BatchLAS" | Adding an op or entry point: `src/ops/<op>/`, tables, tuner spec, tests. |
| @subpage md_docs_2design_2vendor-independence "Vendor independence" | The vendor seam, the vendor-free build, the coverage instrument; the replaced RouteTable layer as history. |
| @subpage md_docs_2design_2vendor-free-status "Vendor-free status" | Current state of the vendor-free build. |

## Core model and API

| Page | What you find there |
| --- | --- |
| @subpage design_matrix_model "Matrix model" | Column-major storage, `ld`/stride/batch, owning matrices vs views, the device-side view. |
| @subpage design_api_conventions "API conventions" | Option structs vs positional overloads, owning-argument forwarding, argument checks, per-item info. |
| @subpage design_workspace "Workspace" | The per-Queue arena, leases, `BumpAllocator` sizing. |
| @subpage design_error_model "Error model" | Exception hierarchy, convergence reporting. |
| @subpage design_environment "Environment" | Every `BATCHLAS_*` variable the library reads. |
| @subpage design_symbol_visibility "Symbol visibility" | What `BATCHLAS_API` must cover. |
| @subpage design_runtime_internals "Runtime internals" | Queue implementation, settings snapshot, embedded tables, template instantiation. |
| @subpage design_device_group_blas "Device-side group BLAS" | In-kernel work-group and sub-group BLAS templates. |

## Operations

| Page | What you find there |
| --- | --- |
| @subpage design_syevx "syevx: algorithm survey" | Selected-eigenpair solvers considered, verdicts. |
| @subpage design_syevx_range "syevx: range selection" | The range API and its hard cases. |
| @subpage design_gesvd "gesvd: design" | One-sided Jacobi, the bidiagonal pipeline, kernel design rules. |

## Build

| Page | What you find there |
| --- | --- |
| @subpage design_cpu_target_detection "CPU target detection" | How the build decides whether CPU SYCL kernels exist, and which tests depend on it. |
| @subpage design_build_performance "Build performance" | Header-structure decisions that keep consumer TUs cheap (e.g. why the umbrella header excludes device code). |

## Defects and history

| Page | What you find there |
| --- | --- |
| @subpage md_docs_2design_2known-defects "Known defects" | Located-but-unfixed defects. |
| @subpage design_flat_selection_phase3_plan "Flat selection, phase 3 plan (history)" | Execution plan for posv, trsm and gemm. |
| @subpage md_docs_2design_2small-n-factorization-plan "Small-n factorization campaign (history)" | Campaign plan and comment-density burn-down; kept for its evidence and cited anchors. |
