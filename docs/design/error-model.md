# Error model {#design_error_model}

> **Status:** current.

Failures BatchLAS diagnoses are thrown as `batchlas::` types. Each derives from its former `std::`
base and from the empty tag `batchlas::exception`. Callers catch by reason, by origin, or as
`std::exception`. User summary: [What gets thrown](../cpp-api.md#what-gets-thrown). Export
requirements: @ref design_symbol_visibility.

## error model: why a hierarchy exists

- Each leaf keeps its `std::` base, so existing handlers match. The Python classes keep the
  builtin base each arm mapped to before (`ValueError`, `IndexError`, `RuntimeError`); see
  [Python exception classes](#error-model-python-exception-classes).
- The runtime arm must derive from `std::runtime_error`. In-tree code catches that type around calls
  that now throw into this arm.

## error model: failures that stay outside the hierarchy

Not wrapped, so `catch (const batchlas::exception&)` does not see them:

- `std::bad_alloc` from null `sycl::malloc_*` returns (`src/util/sycl-util-impl.cc`, `src/queue.hh`,
  `src/ops/syev/sytrd/sytrd_sy2sb.cc`). Wrapping it would lose pybind11's `MemoryError`.
- `sycl::exception`, from the SYCL runtime.

## error model: kernel selection throws {#error-model-kernel-selection-throws-outside-the-hierarchy}

Flat kernel selection (`src/select/`, see `docs/design/flat-kernel-selection.md`) throws only
`batchlas::` types.

| Site | Thrown | When |
| --- | --- | --- |
| `select::detail::resolve_pin` (`src/select/select.hh`) | `invalid_argument` | `BATCHLAS_<OP>_ROUTE` (or `ScopedPin`) does not parse, is not a compiled candidate, or cannot run the shape. `native` and `vendor` warn and fall back to Auto (a `StrictPin` throws). |
| `select::detail::walk` | `unsupported` | No candidate in any table row or the last-resort list can run the call. |
| `select::detail::validate`, `parse_table`, `parse_tables`, `Table::nearest` | `internal_error` | A built-in tuned table does not parse, names an unknown spelling or dtype, or keys on a field the op does not supply. |
| the same, for a `BATCHLAS_TUNED_DIR` table | `invalid_argument` | The same defects in a user table; the message starts `BATCHLAS_TUNED_DIR: `. |
| `select::throw_no_vendor_route` (`src/select/vendor.hh`) | `NoRouteError` (`include/batchlas/no_route.hh`), an `unsupported` | No vendor library and no native kernel for the request, typically `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`. |

- `select::pick` turns only an `unsupported` from the walk into `NoRouteError`; a table defect
  propagates unchanged.
- Ops that parse pin words outside `src/select` throw `invalid_argument` for an unknown word and
  `unsupported` for a known pin this build cannot serve.

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
| `error` | Base of the runtime arm. The library never throws it directly; `tools/tune` and `benchmarks/` throw it for an OS I/O failure (open, lock, write, a failed child command). | n/a |
| `unsupported` | No route, kernel or backend in this build on this device serves the request. | Not as asked; another route, backend, type or shape may work. |
| `device_error` | The device or its vendor runtime failed. | Sometimes. A repeating status code is a real fault. |
| `workspace_error` | Scratch is too small, or the arena ran out. | Yes: re-query `*_buffer_size()`, or halve the batch. |
| `convergence_error` | An iteration did not converge, or a factorisation broke down (LAPACK `info > 0`). | With different parameters. |
| `internal_error` | BatchLAS is internally inconsistent. | Never; report it. |
| `api_misuse` | A valid call in the wrong state, order or thread. | Reorder the calls, or confine the object to one thread. |

- `error` is the catchable base and the home for a failure that fits no child.
- `internal_error` covers library invariants (formerly `std::logic_error`). Selection throws are
  [listed above](#error-model-kernel-selection-throws-outside-the-hierarchy).
- `gesv` and `posv` (`src/ops/gesv/gesv.cc`, `src/ops/posv/posv.cc`) throw `invalid_argument` for
  an empty problem or a heterogeneous batch before selection runs, so a pin cannot change it.

## error model: Python exception classes {#error-model-python-exception-classes}

`python/batchlas/bindings/runtime.cc` registers one class per C++ class and a translator; all are
exported from `batchlas`.

| C++ | Python | Builtin base |
| --- | --- | --- |
| `exception` | `BatchLASException` | `Exception` |
| `invalid_argument`, and any other `std::invalid_argument` | `InvalidArgumentError` | `ValueError` |
| `out_of_range`, and any other `std::out_of_range` | `OutOfRangeError` | `IndexError` |
| `error` | `BatchLASError` | `RuntimeError` |
| `unsupported`, `device_error`, `workspace_error`, `convergence_error`, `internal_error`, `api_misuse` | `UnsupportedError`, `DeviceError`, `WorkspaceError`, `ConvergenceError`, `InternalError`, `ApiMisuseError` (each a `BatchLASError`) | `RuntimeError` |
| `NoRouteError` | `NoRouteError` (an `UnsupportedError`), with `op`, `backend` and `dtype` attributes | `RuntimeError` |

- The `std::` arms also map a plain `std::invalid_argument` or `std::out_of_range` from outside
  BatchLAS.
- A `std::runtime_error` outside the hierarchy stays a plain `RuntimeError`.

## error model: batch-wide convergence errors

A `convergence_error` says some item failed, not which. Per-item status is reported by `potrf`, `getrf`
and `getri` (factorisation) and by `syev`, `syevx`, `gesvd`, `steqr` and `stedc` (convergence). See
[Convergence status](../cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc).

- Gap: a leaf `steqr` under `stedc` is not reported through `stedc`'s `info`.
- Gap: `stein` has no convergence test.
