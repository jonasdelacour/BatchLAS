# Architecture and design {#architecture_index}

Design records: what was decided, why, and what was rejected. Where a decision
rests on a measurement, the page links the evidence in @ref perf_evidence.

## Kernel selection

| Page | What it covers |
| --- | --- |
| @subpage design_flat_selection "Flat kernel selection" | How every op picks the kernel that runs: families, per-device tuned tables, `select::choose`, pins, rules R1-R8. Developer API: @ref selection; inventory: @ref selection_tables. |
| @subpage md_docs_2extending "Extending BatchLAS" | Adding an op or an entry point: `src/ops/<op>/`, tables, tuner spec, tests. |
| @subpage md_docs_2design_2vendor-independence "Vendor independence" | The vendor seam, the vendor-free build and the coverage instrument; the RouteTable layer it replaced, as history. |
| @subpage md_docs_2design_2vendor-free-status "Vendor-free status" | Where the vendor-free build stands. |

## Core model and API

| Page | What it covers |
| --- | --- |
| @subpage design_matrix_model "Matrix model" | Column-major storage, `ld`/stride/batch semantics, owning matrices vs views, the device-side view. |
| @subpage design_api_conventions "API conventions" | Option structs vs positional overloads, owning-argument forwarding, argument checks, per-item info. |
| @subpage design_workspace "Workspace" | The per-Queue arena, leases and `BumpAllocator` sizing. |
| @subpage design_error_model "Error model" | The exception hierarchy and convergence reporting. |
| @subpage design_environment "Environment" | Every `BATCHLAS_*` variable the library reads. |
| @subpage design_symbol_visibility "Symbol visibility" | What `BATCHLAS_API` must cover and why. |
| @subpage design_runtime_internals "Runtime internals" | Queue implementation, settings snapshot, embedded tables, template instantiation. |
| @subpage design_device_group_blas "Device-side group BLAS" | The in-kernel work-group and sub-group BLAS templates. |

## Operations

| Page | What it covers |
| --- | --- |
| @subpage design_syevx "syevx: algorithm survey" | Selected-eigenpair solvers considered, verdicts, what not to build. |
| @subpage design_syevx_range "syevx: range selection" | The range API and its hard cases. |
| @subpage design_gesvd "gesvd: design" | One-sided Jacobi, the bidiagonal pipeline and the kernel design rules. |

## Build

| Page | What it covers |
| --- | --- |
| @subpage design_cpu_target_detection "CPU target detection" | How the build decides whether CPU SYCL kernels exist, and which tests depend on it. |

## Defects and history

| Page | What it covers |
| --- | --- |
| @subpage md_docs_2design_2known-defects "Known defects" | Located-but-unfixed defects. |
| @subpage design_flat_selection_phase3_plan "Flat selection, phase 3 plan (history)" | The execution plan for posv, trsm and gemm. |
| @subpage md_docs_2design_2small-n-factorization-plan "Small-n factorization campaign (history)" | The campaign plan and its comment-density burn-down; kept for its evidence and its cited anchors. |
