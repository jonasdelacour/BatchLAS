# Inbox from shard H7-dispatch-device

Material moved out of `include/batchlas/blas/dispatch/*.hh` and `include/batchlas/util/*.hh`
for pages H7 does not own. Headings marked "cited" are already pointed at by code; please create
them verbatim.

## -> docs/perf/dispatch.md: Dispatch: the ormqr chooser that forced past supports

(Cited. Suggested place: under "Correctness findings", next to the 108x buffer-size bullet, which
it complements.)

`RouteTable<Op::ormqr>` replaced the smallest of the three Provider-based choosers, and it is where
the cost of conflating "forced" with "supported" was easiest to see. `choose_ormqr_provider` opened
with

    Provider chosen = normalize_ormqr_vendor_like(policy.forced);
    if (chosen != Provider::Auto) return chosen;

so a forced provider was returned **without ever being checked against `ormqr_supports_blocked`**.
Two defects followed, both now fixed by construction rather than by remembering a check:

1. **Forcing could run an unsupported kernel.** `ormqr_supports_blocked` is false for complex with
   `Transpose::Trans` and on any non-GPU queue, but `ormqr_dispatch`'s tail was
   `if (chosen == Vendor) vendor else blocked`, so `BATCHLAS_ORMQR_PROVIDER=blocked` ran the blocked
   path on exactly the inputs the predicate exists to exclude.
2. **The buffer size and the call could disagree.** For a forced value that is neither Vendor nor
   Blocked (`cta`, `two_stage`, `jacobi`, all of which parse), `ormqr_dispatch` fell into its `else`
   arm and reset to Vendor, while `ormqr_buffer_size_dispatch`'s tail
   (`if (chosen == Vendor) vendor_size; return blocked_size`) returned the blocked size; the caller
   then hit "ormqr: insufficient workspace for chosen provider" from the call it had just sized for.
   (The 2560 vs 276480 byte instance is already recorded under Correctness findings.)

Both share the Provider enum's root: "the user asked for this" and "this can serve the shape" were
the same value. Splitting `supports()` from the forced request makes the first impossible; resolving
once through a pure table makes the second impossible.

Two table details that go with it: the order `{Native, Blocked}, {Vendor, Auto}` is what the shared
`std::array<Provider, 6>` came to for ormqr (`BatchLAS_CTA`, `_TwoStage`, `_Jacobi` were listed but
matched no branch, so they were inert padding). And ormqr has **no measured window**: no shape ever
sent a supported blocked call to the vendor, so `preferred()` equals "native and supported". It is
kept as a separate function because a future crossover goes there; in `supports()` it would become a
correctness claim.

Code sites: `include/batchlas/blas/dispatch/route_ormqr.hh` (file header).

## -> docs/design/vendor-independence.md: The vendor gate: why the tile route predicate is per backend and scalar

(Cited. Suggested place: a subsection of "The vendor gate", after the `level3_tile_route_available`
paragraph.)

Four sites in `src/extensions/` used to ask "is the tile kernel linked" as `B == Backend::CUDA`, one
with the comment "Not a statement about CUDA the vendor -- it is where the kernel is wired." The
comment was right and the expression wrong: the tile kernels are portable SYCL (verified by compiling
`triangular_expand.hh` and the `*_tiles.hh` family standalone at `-fsycl-targets=spir64_x86_64`);
they live in `{symm,syrk,syr2k,trmm}_custom_dispatch.cc`, which `src/backends/CMakeLists.txt`
compiled only with cuBLAS because their dispatch terminated in `*_vendor_cuda_raw`. With
`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` the backend is still `Backend::CUDA`, the tile TUs are not
compiled, and every such site claimed a kernel that was not linked.

WP1 S7 recorded that the header's own prediction was wrong. It said that once WP1 freed the four TUs
the predicate "becomes true for every backend -- and that is the only edit needed". A bare `true`
would have been wrong in two directions:

* **Too wide in type.** The float routes became reachable everywhere once S6 put their gate in the
  facade; the double and complex ones did not: syrk's non-float gram branch and trmm's non-float tile
  branch stayed in `cublas.cc`, and syr2k has no non-float tile route at all
  (`syr2k_triangular_tiles` has one call site, in the float-only dispatcher). A bare `true` tells
  `ortho.cc`'s `gram_via_syrk` (which admits double) that a kernel exists, and vendor-free that call
  throws.
* **Too wide in backend.** The facade gate is guarded on `Backend::CUDA`, so nothing is wired for
  ROCM or NETLIB; claiming otherwise re-introduces the defect WP0 S8 removed (the Backend enum
  standing in for a question it does not answer).

Hence the second template parameter: the answer varies per (backend, scalar). Vendor-present
behaviour is unchanged by construction (with `BATCHLAS_HAS_CUBLAS` true it is true for every T on
CUDA); only the vendor-free float case moved, which was the WP1 gain.

Code sites: `include/batchlas/blas/dispatch/route_compiled.hh` (`level3_tile_route_available`).

## -> docs/design/vendor-independence.md: The vendor gate: history of the per-library predicates

(Cited. Suggested place: a subsection of "The vendor gate".)

Until WP0 S5 "is a vendor implementation compiled in" could not be asked at all: the public entry
point was defined inside the vendor TU, so "no vendor" and "no entry point" were the same condition
and the answer was trivially yes. The spec proposed seven stub TUs under `src/dispatch/absent/*.cc`
defining a throwing `backend::<op>_vendor` per absent library; it was declined because it restates all
26 vendor signatures, and S5b's bugs were precisely signature divergence between restated copies
(already summarised in the section; keep this as the provenance).

`factorization_vendor_available` was first keyed on cuBLAS alone. On NVIDIA the group spans two
libraries (getrf and getri are `cublas<t>getrfBatched` / `getriBatched`; geqrf, orgqr, ormqr and
getrs's batch <= 1 arm are cuSOLVER: `cusolverDnXgeqrf`, `cusolverDnSormqr`, ...), so the predicate
claimed a geqrf vendor route whenever cuBLAS was on and told a vendor-free user to "re-enable cuBLAS"
to get geqrf back, which does not restore it. It now requires both. A finer per-op split would have
to move every call site and is open debt; please add it to "What is still open, architecturally".

Code sites: `include/batchlas/blas/dispatch/vendor_available.hh` (`factorization_vendor_available`).

## -> docs/design/vendor-independence.md: Coverage instrument: why the compile-time gate was rejected

(Not cited; suggested as extra paragraphs in "Vendor independence: the coverage instrument", which
already has the weak-symbol summary.)

Why two tables: `docs/design/vendor-free-status.md` records which test SUITES fail, which is the
right unit for spotting regressions and the wrong unit for planning: "ortho_tests fails" does not say
whether the gap is potrf, trsm, geqrf, orgqr or gemv, and one missing kernel fails a dozen suites.
The static table names the ops with no native kernel; the dynamic one says which shapes real callers
hit, so WP3-WP8 could cover those first.

Why the compile-time gate was wrong twice over. The stated reason ("the counters are cheap per call,
but gemm is called in inner loops") did not survive reading the code: `resolve_route` runs once per
op invocation, not per element, and already walks the order calling `supports()`/`preferred()`, so a
predicted branch on a global bool is far cheaper than the work beside it. The failure that bit: with
`-DBATCHLAS_ENABLE_COVERAGE=ON`, `libbatchlas_backends.so` referenced `coverage::record`, yet
`gemm_tests` produced a file with a correct header and zero `reached` rows, because `gemm_tests`
carried its own `resolve_route_uninstrumented<Op::gemm, float>` weak copy that ELF bound ahead of the
library's. A compile-time switch on an inline function in a header is sound only if every TU in the
process agrees, which a library cannot enforce on its consumers.

Code sites: none (the header keeps the invariant and cites
`#vendor-independence-the-coverage-instrument`).

## -> docs/design/vendor-independence.md: stale statements to update (no new heading)

* "The resolver", rule 5: a forced vendor is honoured only when `vendor_available` **and**
  `supports()` hold (`route_resolve.hh`, the `is_vendor(forced)` branch).
* "The resolver", trap paragraph: "every decision goes to the vendor with no message" is stale on two
  counts: the unset default is now `{Auto, Auto}` for every op, and `warn_unparsed_route_env`
  (`route_env.hh`) prints a one-time stderr warning for an unparsed value. A typo now resolves to Auto
  *with* a warning.
* "The three routing predicates": "The comment at `route_resolve.hh:40-45` still says 'every op but
  geqrf' — stale" no longer applies; that comment is gone.
* "Vendor independence: the coverage instrument" and "What is still open" item 4: the
  `-DBATCHLAS_ENABLE_COVERAGE` claim is gone from `route_resolve.hh`
  (it remains in `tests/route_vocabulary_tests.cc`).
* Every `route_resolve.hh:NN`, `route.hh:NN`, `coverage.hh:NN`, `no_route.hh:NN`,
  `vendor_available.hh:NN` and `route_compiled.hh:NN` line citation on the page shifted in this pass
  (doc comments were added). Suggest citing by symbol (`resolve_route_uninstrumented`,
  `native_tier_preferred_or_default`, `level3_tile_route_available`, ...) instead of by line.
* `NoRouteError`'s old header comment claimed the message names "the backend ... and the shape"; it
  names neither (the page already says the backend is discarded). The header now says so.

## -> docs/design/known-defects.md: kernel-heuristics: the power-of-two rounding assumes a 32-bit long

(Not cited; found while documenting, not measured.)

`compute_optimal_wg_size` (`include/batchlas/util/kernel-heuristics.hh`) rounds REDUCTION and SCAN
work-group sizes with `size_t(1) << (31 - __builtin_clzl(base_wg_size))`. `__builtin_clzl` counts
leading zeros of a 64-bit `unsigned long` on LP64, so for base 256 the shift count is 31 - 55 = -24:
undefined behaviour (x86 masks it to 40, giving 2^40, which the later `problem_size` and
`MAX_WORK_GROUP_SIZE` clamps then cut down). The result is therefore min(problem_size, device max)
rather than a power of two. Callers: `src/matrix.cc:390` (REDUCTION, `rows`) and `:497` (SCAN,
`rows + 1`). Worth checking whether those kernels assume a power-of-two work-group. Fix: `63 -
__builtin_clzl` or `std::bit_floor`.

Also in that header, recorded in its doc comments: `compute_optimal_wg_size`'s `batch_size` and
`memory_per_problem` are unused (the latter guards an empty `if`), and
`compute_batched_nd_range_sizes`'s `footprint_per_problem` estimate is overwritten on every path and
`max_wg_size_for_kernel` is unused.

## -> docs/pages/architecture.md: link the new design page

Add a row: `| @subpage design_device_group_blas "Device-side group BLAS" | The in-kernel
batchlas::device templates: calling convention, sub-group vs work-group executors, the local-memory
workspace protocol, register and SLM caveats. |`

## -> docs/design/known-defects.md (or a fixed-defects list): miniacc --help used to exit

(Optional; history removed from `include/batchlas/util/miniacc.hh`.) `miniacc::ParseCommandLine`'s
`--help` arm used to call `std::exit(0)`. `miniacc.hh` is installed (cmake/BatchLASPackaging.cmake
installs `include/batchlas` wholesale; only `minibench*.hh` and `bench_structured.hh` are excluded),
and a library header must never terminate its host: `exit()` runs no destructors for live automatic
objects in the caller's frames, and a consumer that parses an argv containing `--help`, or embeds the
harness, cannot stop it. It now sets `CliOptions::help_requested` and `MiniAccMain` returns 0.
