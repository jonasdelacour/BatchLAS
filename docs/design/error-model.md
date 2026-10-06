# Error model: the exception hierarchy {#design_error_model}

> **Covers:** why `include/batchlas/error.hh` exists, which failures it covers and which it
> deliberately does not, the rules that keep the tag base `batchlas::exception` catchable, and
> what each class promises a caller about retrying.
> **Status:** current. The hierarchy and per-item convergence status landed together in
> `8cc43456` (2026-09-09). The user-facing summary is
> [What gets thrown](../cpp-api.md#what-gets-thrown); the API reference is the `errors` group.

Every failure BatchLAS diagnoses itself is thrown as a `batchlas::` type that derives from
both the `std::` exception the site threw before and an empty tag, `batchlas::exception`. A
caller can therefore catch by *reason* (`workspace_error`, `unsupported`, ...), by *origin*
(`batchlas::exception`), or as before (`std::exception`). The export requirement on these
classes is recorded in @ref design_symbol_visibility.

## error model: why a hierarchy exists

Before `error.hh` the library threw 572 raw `std::` exceptions and nothing else. A consumer could
not write `catch (const batchlas::error&)`, and, more importantly, could not tell a shape mismatch
from an unsupported route from a workspace shortfall from a device failure. It could therefore
not recover from one class while letting another propagate. In a batched solver that is the
difference between "retry this batch smaller" and "abort the run".

Each leaf keeps its old `std::` base, so every existing handler still matches and pybind11's
default translator still maps the same C++ exception to the same Python type
(`std::invalid_argument` to `ValueError`, `std::out_of_range` to `IndexError`,
`std::runtime_error` to `RuntimeError`). Deriving the runtime arm from `std::runtime_error` is
load-bearing: several in-tree call sites and tests catch `std::runtime_error` around calls whose
throws now land in that arm.

## error model: failures that stay outside the hierarchy

Two failure classes are deliberately not wrapped, so `catch (const batchlas::exception&)` catches
every failure BatchLAS diagnoses, not every failure a BatchLAS call can produce:

- `std::bad_alloc`, thrown from three sites where a `sycl::malloc_*` returned null
  (`src/util/sycl-util-impl.cc`, `src/queue.hh`, `src/extensions/sytrd_sy2sb.cc`). It is the
  standard type for allocation failure, carries no message worth preserving, and is what pybind11
  maps to `MemoryError`. Wrapping it would lose `MemoryError` and gain nothing.
- `sycl::exception`, raised by the SYCL runtime itself. It is not ours to reclassify.

A consumer that must not let anything escape still needs a `catch (const std::exception&)`
behind the BatchLAS handler.

## error model: the three tag-base rules

`batchlas::exception` is an empty tag whose only job is to make
`catch (const batchlas::exception& e)` match every class in the header and nothing else. Three
rules govern it, and each one fails silently (no compiler diagnostic at any point) when broken.
All three were verified by compiling a model of the hierarchy, and `tests/error_model_tests.cc`
asserts them.

1. **It must not derive from `std::exception`.** Every leaf already carries one `std::exception`
   subobject through its `std::` base. A second one, brought in by the tag, makes
   `catch (const std::exception&)` face two ambiguous base subobjects: the handler stops matching
   and the exception runs to `std::terminate`. An ambiguous base is only diagnosed at a cast, and
   a catch clause is not one.
2. **Every leaf must inherit it virtually.** Under non-virtual inheritance a future class
   deriving from two arms (say `bad_shape : public invalid_argument, public unsupported`) would
   get two tag subobjects, and `catch (const batchlas::exception&)` would stop matching that
   class, again silently and again terminating. `detail::exception_bridge<StdBase>` is the only
   place the inheritance is spelled, so no future class can get it wrong. The cost is one vptr,
   paid on the error path only.
3. **It must not declare `what()`.** `std::exception::what()` lives in a different base
   subobject and does not override a pure virtual declared on the tag, so adding one makes every
   leaf abstract ("cannot declare variable to be of abstract type"). This is the one rule that
   fails loudly. `message()` is the differently named accessor that reaches the text from a tag
   handler.

The tag's copy constructor and assignment are public rather than protected purely so that no
access check can ever stand between a leaf's implicitly defined copy constructor and this
virtual base. The pure virtual `message()` already makes the tag abstract.

A fourth rule, that every class in the header carries `BATCHLAS_API`, is rule 1's failure mode
reached by a different road; it is recorded in
[symbol visibility: exception typeinfo must be exported](symbol-visibility.md#symbol-visibility-exception-typeinfo-must-be-exported).

## error model: what each class tells a caller

| class | meaning | retry? |
| --- | --- | --- |
| `invalid_argument` | the call violates the API contract | no; fix the call |
| `out_of_range` | an index is outside its container (`MatrixView::at`, `batch_item`) | no; separate only so Python indexing keeps raising `IndexError` |
| `error` | base of the runtime arm; nothing throws a bare `error` today | n/a |
| `unsupported` | no route, kernel or backend in this build on this device serves the request | not as asked; a different route, backend, type or shape may work |
| `device_error` | the device or its vendor runtime failed | sometimes; the only class where a retry can be right. A repeating status code is a real fault |
| `workspace_error` | the scratch handed in is too small, or the arena ran out | yes: re-query `*_buffer_size()`, or halve the batch |
| `convergence_error` | an iteration did not converge or a factorisation broke down (LAPACK `info > 0`) | with different parameters, not the same ones |
| `internal_error` | BatchLAS is internally inconsistent | never; report it |
| `api_misuse` | a well-formed call in the wrong state, order or thread | reorder the calls or confine the object to one thread |

`error` has no direct thrower: every site adjudicated during the migration fitted one of the five
children. It is kept as a catchable base and as the home for a future failure that fits none of
them. `internal_error` covers the sites that used to throw `std::logic_error` (a resolver that
picked a native arm no linked kernel serves, a capability query and the facade that reads it
disagreeing, an uninjected internal seam, a branch documented unreachable). Each of those is an
invariant of the library, not a statement about the arguments. `api_misuse` differs from
`invalid_argument` because no argument is wrong: only when and from where the call was made.

## error model: batch-wide convergence errors

A `convergence_error` is batch-wide: it says some item in the batch failed, not which. The header
originally noted that per-item status spans were "work package A-1" and that several tiers
(every `stedc` merge arm, `steqr_wg`, `syev_jacobi_cta`, `gesvdj_cta`) did not even raise the
exception but returned a wrong answer silently.

**Partly stale.** Per-item `info` spans landed in the same commit as the hierarchy (`8cc43456`):
`potrf`, `getrf`, `getri` report factorisation status and `syev`, `syevx`, `gesvd`, `steqr`,
`stedc` report convergence status. See
[Convergence status](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc) for the
semantics and the remaining gaps (a leaf `steqr` under `stedc` is not reported through `stedc`'s
`info`; `stein` has no convergence test). Whether each of the four tiers named above now writes
its status has not been re-audited for this page.
