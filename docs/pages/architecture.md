# Architecture and design {#architecture_index}

Design records: what was decided, why, and what was rejected. Where a decision
rests on a measurement, the page links the evidence in @ref perf_evidence.

## Dispatch and vendor independence

| Page | What it covers |
| --- | --- |
| @subpage md_docs_2design_2vendor-independence "Vendor independence" | The Route vocabulary, the entry-point facade, the vendor gate, the resolver and the coverage instrument. |
| @subpage md_docs_2design_2vendor-free-status "Vendor-free status" | Where the vendor-free build stands. |
| @subpage md_docs_2extending "Extending BatchLAS" | Adding an entry point: headers, forwarding overloads, `*_buffer_size`, routing. |

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
| @subpage md_docs_2design_2small-n-factorization-plan "Small-n factorization campaign (history)" | The campaign plan and its comment-density burn-down; kept for its evidence and its cited anchors. |
