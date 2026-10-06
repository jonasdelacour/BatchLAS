# Inbox from shard H3-conventions

## -> docs/design/symbol-visibility.md: symbol visibility: BinaryOp is a template-argument enum too

`batchlas::linalg::BinaryOp` carries `BATCHLAS_API` for the same reason `Backend` and
`MatrixFormat` do (the note at the top of `include/batchlas/blas/enums.hh`, which points at
`#symbol-visibility-enums-used-as-template-arguments` on this page). It is the third enum in the
public surface used as a template *argument*, and an instantiation takes the minimum of the
template's visibility and its arguments'. Without the annotation, all four
`elementwise_into<float, BinaryOp::*>` specialisations stayed hidden while their `BATCHLAS_API`
declaration looked correct. Found by set-diffing the library's exported symbols against what
consumers actually link, not by reading: nothing about the declaration looks wrong.

Code sites now pointing here:

- `include/batchlas/blas/linalg-ops.hh` (above `enum class BATCHLAS_API BinaryOp`):
  `evidence: docs/design/symbol-visibility.md#symbol-visibility-binaryop-is-a-template-argument-enum-too`

## -> docs/design/build-performance.md: build performance: the umbrella header excludes device code

`<batchlas/blas/linalg.hh>` (what `<batchlas.hh>` includes) deliberately does NOT pull in
`<batchlas/blas/device.hh>` (the device-side group-BLAS kernel templates) or, through
`<batchlas/blas/functions.hh>`, `<sycl/sycl.hpp>`. Together they cost about 4.1 s per consumer
TU, and 71 of the 114 test/benchmark TUs used neither (count as recorded in the header comment
before 2026-09-30; not re-measured). A consumer that needs `batchlas::device::*` includes
`<batchlas/blas/device.hh>` itself. Both edges have to stay cut: the headers form a cycle, so
restoring either one re-pulls the whole umbrella and the saving vanishes. (AGENTS.md section 7
lists "cutting one edge of a header cycle" among the measured dead ends; this is the same cycle.)

Code sites now pointing here:

- `include/batchlas/blas/linalg.hh` (above `#include <batchlas/blas/functions.hh>`):
  `evidence: docs/design/build-performance.md#build-performance-the-umbrella-header-excludes-device-code`

## -> docs/design/known-defects.md: linalg::qr returns a wrong QR after an earlier call in the process

`linalg::qr` is deliberately absent from `include/batchlas/blas/linalg-ops.hh`. The composition
`geqrf` + `triangular_mask_into` + `orgqr` returned \f$QR \ne A\f$ once an earlier `linalg::qr`
test had run in the same process, and passed when run alone. Cause unknown; the original
header note read "repro: tests/linalg_layer_tests.cc, 4x". AGENTS.md section 9 describes it as
"an unexplained cross-Queue wrong-answer defect". Open debt: locate the cross-Queue state (arena
reuse, a static, or a stale event) before the wrapper can be added.

Code sites now pointing here:

- `include/batchlas/blas/linalg-ops.hh` (end of namespace `batchlas::linalg`):
  `evidence: docs/design/known-defects.md#linalgqr-returns-a-wrong-qr-after-an-earlier-call-in-the-process`

## -> docs/pages/architecture.md: link the new API conventions page

Add a row under "Dispatch and vendor independence" (or a new "API" table):
`| @subpage design_api_conventions "API conventions" | Two spellings per entry point, option structs, the dispatch and owning-argument overloads and their traps, the USM pointer check, shape checks, info spans, the linalg layer. |`
The page is `docs/design/api-conventions.md` and is otherwise an orphan top-level page.

## -> .github/ci/comment_density_waivers.txt: four waivers now free

`include/batchlas/blas/extra.hh` (line 88), `include/batchlas/blas/linalg-ops.hh` (108),
`include/batchlas/blas/linalg.hh` (109) and `include/batchlas/blas/queue-dispatch.hh` (110) are
under 18% after this pass; `check_comment_density.py` lists them as free to delete.
