# Error model {#design_error_model}

> **Status:** current.

Failures BatchLAS diagnoses are thrown as `batchlas::` types. Each derives from its former `std::`
base and from the empty tag `batchlas::exception`. Callers catch by reason, by origin, or as
`std::exception`. User summary: [What gets thrown](../cpp-api.md#what-gets-thrown). Export
requirements: @ref design_symbol_visibility.

## error model: why a hierarchy exists

- Each leaf keeps its `std::` base, so existing handlers match and pybind11 maps types as before
  (`invalid_argument` to `ValueError`, `out_of_range` to `IndexError`, `runtime_error` to
  `RuntimeError`).
- The runtime arm must derive from `std::runtime_error`. In-tree code catches that type around calls
  that now throw into this arm.

## error model: failures that stay outside the hierarchy

Not wrapped, so `catch (const batchlas::exception&)` does not see them:

- `std::bad_alloc` from null `sycl::malloc_*` returns (`src/util/sycl-util-impl.cc`, `src/queue.hh`,
  `src/extensions/sytrd_sy2sb.cc`). Wrapping it would lose pybind11's `MemoryError`.
- `sycl::exception`, from the SYCL runtime.

## error model: kernel selection throws outside the hierarchy

Flat kernel selection (`src/select/`, see `docs/design/flat-kernel-selection.md`) throws plain `std::`
types. None of these match `batchlas::exception`:

| Site | Thrown | When |
| --- | --- | --- |
| `select::detail::resolve_pin` (`src/select/select.hh`) | `std::invalid_argument` | `BATCHLAS_<OP>_ROUTE` (or `ScopedPin`) does not parse, is not a compiled candidate, or cannot run the shape. `native` and `vendor` warn and fall back to Auto. |
| `select::detail::walk` | `std::runtime_error` | No candidate in any table row or the last-resort list can run the call. |
| `select::detail::validate`, `parse_table` | `std::runtime_error` | A tuned table (built in or from `BATCHLAS_TUNED_DIR`) names an unknown spelling. |
| `select::throw_no_vendor_route` (`src/select/vendor.hh`) | `batchlas::NoRouteError` (`include/batchlas/no_route.hh`) | No vendor library and no native kernel for the request, typically `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`. |

Ops that parse pin words outside `src/select` throw `std::invalid_argument` for an unknown word and
`batchlas::unsupported` for a known pin this build cannot serve. By meaning, the `select` throws
should be `invalid_argument` and `unsupported`. Changing them alters what `std::` catchers see, so
that change needs its own commit.

## error model: the three tag-base rules

`batchlas::exception` is an empty tag. It makes `catch (const batchlas::exception&)` match every class
in the header and nothing else. All three rules fail silently when broken. `tests/error_model_tests.cc`
asserts them.

1. **Do not derive the tag from `std::exception`.** Each leaf already has one `std::exception`
   subobject. A second one makes `catch (const std::exception&)` ambiguous, and the exception calls
   `std::terminate`.
2. **Inherit the tag virtually in every leaf.** Only `detail::exception_bridge<StdBase>` spells the
   inheritance. A class with two arms would otherwise get two tag subobjects, and the tag catch would
   stop matching it. The cost is one vptr on the error path.
3. **Do not declare `what()` on the tag.** It does not override a pure virtual, so every leaf becomes
   abstract and the build fails. Use `message()` to read the text from a tag handler.

The tag's copy operations are public, so no access check sits between a leaf's copy constructor and
the virtual base. The pure virtual `message()` keeps the tag abstract.

Rule four (every class carries `BATCHLAS_API`) is in
[symbol visibility: exception typeinfo must be exported](symbol-visibility.md#symbol-visibility-exception-typeinfo-must-be-exported).

## error model: what each class tells a caller

| Class | Meaning | Retry? |
| --- | --- | --- |
| `invalid_argument` | The call violates the API contract. | No; fix the call. |
| `out_of_range` | An index is outside its container (`MatrixView::at`, `batch_item`). | No. Kept separate so Python raises `IndexError`. |
| `error` | Base of the runtime arm. Nothing throws it directly. | n/a |
| `unsupported` | No route, kernel or backend in this build on this device serves the request. | Not as asked; another route, backend, type or shape may work. |
| `device_error` | The device or its vendor runtime failed. | Sometimes. A repeating status code is a real fault. |
| `workspace_error` | Scratch is too small, or the arena ran out. | Yes: re-query `*_buffer_size()`, or halve the batch. |
| `convergence_error` | An iteration did not converge, or a factorisation broke down (LAPACK `info > 0`). | With different parameters. |
| `internal_error` | BatchLAS is internally inconsistent. | Never; report it. |
| `api_misuse` | A valid call in the wrong state, order or thread. | Reorder the calls, or confine the object to one thread. |

- `error` is the catchable base and the home for a failure that fits no child.
- `internal_error` covers library invariants (formerly `std::logic_error`). Selection throws are
  [listed above](#error-model-kernel-selection-throws-outside-the-hierarchy).
- `gesv` and `posv` (`src/ops/gesv/gesv.cc`, `src/ops/posv/posv.cc`) throw `internal_error` for an
  empty problem or a heterogeneous batch before selection runs, so a pin cannot change it. By
  meaning these are `invalid_argument` or `unsupported`.
- `api_misuse` is for a valid argument used at the wrong time or from the wrong thread.

## error model: batch-wide convergence errors

A `convergence_error` says some item failed, not which. Per-item status is reported by `potrf`, `getrf`
and `getri` (factorisation) and by `syev`, `syevx`, `gesvd`, `steqr` and `stedc` (convergence). See
[Convergence status](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc).

- Gap: a leaf `steqr` under `stedc` is not reported through `stedc`'s `info`.
- Gap: `stein` has no convergence test.
- Not re-audited: whether the `stedc` merge arms, `steqr_wg`, `syev_jacobi_cta` and `gesvdj_cta`
  now write status. They once returned wrong answers without raising.
