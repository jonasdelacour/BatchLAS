# Readiness remediation plan

Execution plan for the findings in the BatchLAS readiness audit (2026-09-08,
tree `ab6319d`). Written for hand-off to subagents: every work package (WP)
states its inputs, exact files, steps, acceptance criteria as runnable
commands, and what it must NOT do. Read the whole of §0 before starting any WP.

## 0. Ground rules for every WP

**Where the code is.** Two trees matter:

| ref | what it is |
|---|---|
| `origin/main` (`ab6319d`, fetched 2026-09-08) | what ships. Has NO routing machinery (`Route`, `RouteTable`, `resolve_route`), no native trsm/gemv/spmm/potrf/lu/qr kernels, no `BATCHLAS_ENABLE_VENDOR_BLAS`. Still carries the legacy `Provider`/`DispatchPolicy` headers (`include/batchlas/blas/dispatch/{provider,env,context}.hh`). Last merges into it: #94 (docs), #106 (doc citations), #108 (code reduction, 2026-09-08). |
| `origin/stack/pr11-wp8-spmm` (`7dbdfc1`, 2026-09-06) | **the landing candidate.** Tip of the stacked PRs #95–#105 plus #107's review fixes. Per the GitHub API every one of #95–#107 is `merged=True`, but each merged into the *previous stack branch* (#95 → `stack/pr00-docs`, #96 → `stack/pr01-wp0-route`, … #107 → `stack/pr11-wp8-spmm`); none has `base=main`. 40 commits ahead of `main`, 10 behind (= #108). Has 19 `route_*` files, `src/dispatch/`, `scripts/rocm_syntax_check.sh`, `scripts/route_diff.sh`; deletes `provider.hh`/`env.hh`/`context.hh`. |
| `consolidate-vendor-independence` (`e3f679c`, 2026-09-05) | **stale — do not land from it.** A Sep-5 snapshot that predates #107 (two build breaks, one wrong answer) and #108. 42 files differ from the stack tip. Kept only as a reference. |

**The docs on `main` describe the stack, not `main`.** 28 of the 53 source
paths cited by `docs/perf/*.md`, `docs/design/known-defects.md` and
`docs/design/vendor-free-status.md` do not exist on `main`. Do not "fix" a
doc by deleting a citation that is absent on `main`; it is present on the
stack tip. WP0 lands the stack first; everything after WP0 is done on top of it.

**Do not trust a local `main`.** The audit that produced this plan first
compared against a session-start `main` snapshot and mis-stated the PR
state. Always `git fetch origin` and compare against `origin/main`; for PR
state use the GitHub API (`https://api.github.com/repos/jonasdelacour/BatchLAS/pulls/<n>`
→ `merged`, `base.ref`), not `merge-base --is-ancestor` on a local ref.

**Branch and PR discipline.** One branch per WP, named `readiness/wpN-<slug>`,
based on `main` after WP0 has merged. Small WPs may be one PR; WP3, WP4, WP5,
WP8 are explicitly phased and each phase is its own PR. Never push to `main`.
Commit messages follow the existing `type(scope): what` style
(`git log --oneline -30`).

**Build and test scoping.** The full suite is 15–20 min and `main` is not
green. Follow `tests/README.md`: scope with `ctest -L <label>` /
`-R '^name$'` / `-LE slow`; run the full suite only as the final gate and
**diff failing test names** against the baseline, never compare counts.
Known-failing on `main` (last full run, 2026-08-12):

```
lanczos_tests:   LanczosTestBase.LanczosTest, LanczosTestBase.ToeplitzEigenpairs
sytrd_blocked_tests: SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal
steqr_tests:     SteqrTest/0.StressExtremeMagnitudesN32, SteqrTest/1.StressExtremeMagnitudesN32
```

Presets: `dev-tests` is the correctness build (CPU + GPU targets, all test
instantiations). `dev-gpu-tests` is 30% faster and instantiates half the
typed suites — never use it for a final gate. Set
`CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)`. A one-line edit costs a full device
relink of its `.so` (~3 min); batch your edits.

**Running binaries directly** needs `OPENBLAS_CORETYPE=SKYLAKEX` (ctest sets
it; a bare `./build/tests/foo` does not).

**Toolchain on this box.** `/opt/dpcpp-cuda/bin/clang++` (intel/llvm, `--cuda`),
CUDA 13.2, RTX 4090 ×2 (`sm_89`), g++ 11.4, CMake 3.28, Docker with the
`nvidia` runtime, 20 cores, 125 GB. No `gh` CLI, no `scikit-build-core`,
no Actions runner installed.

**Blind-guard rule.** A new test is not done until you have applied the
defect it names, rebuilt, and watched it go red. Say in the PR which break
you applied. (`docs/design/known-defects.md`, "The checklist this produces".)

**Comment density.** Repo norm is 12–18%; match the surrounding file.

---

## Dependency graph

```
WP0 land vendor-independence ──┬──> WP6 realise measured wins
                               ├──> WP9 hygiene (docs, Provider drift, rocm script)
                               │
WP1 namespace ────────┐        │
WP2 CI ───────────────┼──> (all later WPs are verified by WP2's runner)
WP3 ABI/export ───────┤
WP4 errors + info ────┤
WP5 settings ─────────┘
WP7 compiler decouple + v0.2.0 + wheel   (needs WP1, WP3, WP9)
WP8 half / bf16 / mixed precision        (needs WP0, WP4)
```

WP0, WP1, WP2 can start in parallel on day one. WP0 must merge before WP6/WP8/WP9.
WP1 and WP3 both touch `include/batchlas/util/*.hh` — do WP1 first, rebase WP3.

---

## WP0 — Land the vendor-independence stack on `main`

**Why first.** `docs/` on `main` cites 28 source paths that only exist on
the stack tip. Every routing/perf finding in the audit (P-2, A-4, half of
P-3) is about code that has not shipped. Nothing in `docs/perf/` is true of
`main` until this lands. The stack was reviewed and merged PR-by-PR into
itself; the missing step is the final merge of its tip into `main`, which
#108 has since made non-trivial.

**Input.** `origin/stack/pr11-wp8-spmm` @ `7dbdfc1` (contains #95–#107);
`origin/main` @ `ab6319d` (contains #108).

**Measured 2026-09-08:** `git merge origin/main` into the stack tip conflicts
in exactly these 14 files (probe merge, aborted):

```
include/batchlas/blas/dispatch/context.hh     deleted by the stack; #108 edited it → DELETE
include/batchlas/blas/extensions.hh           stack: routed entry points; #108: generated forwarder
include/batchlas/blas/extra.hh                same shape as extensions.hh
include/batchlas/blas/functions/syev.hh       stack: syev's RouteTable lives here; #108: forwarder macros
src/backends/cublas.cc                        #108 collapsed the 4 Level-3 vendor-wrapper mechanisms and
src/backends/cusolver.cc                        the 8 instantiation epilogues into op-name tables/one macro
src/backends/cusparse.cc                        (de69e33, 0889f9b, 1c3aef8); the stack moved the public
src/backends/mkl.cc                             entry points OUT of these TUs into src/dispatch/entry_points/
src/backends/netlib_lapack.cc                   and gated them on vendor_available<>. Both are wanted.
src/backends/rocblas.cc
src/backends/rocsolver.cc
src/backends/rocsparse.cc
src/sycl/gemm_kernels.cc                      #108: 33 forwarders → one NTTP launcher table (a511f84);
                                              stack WP2: new kernel variants (predicated 128x128, 64x64_k16_wide)
tests/options_api_tests.cc                    both sides added tests → keep both
```

**Resolution rule, per file.** The stack owns *what is called* (routing,
entry-point location, `vendor_available` gates, `supports`/`preferred`
tables). #108 owns *how the vendor TUs are spelled* (one instantiation macro,
op-name tables, the single `ScopedEnvVar`, the generated owning-argument
forwarder). Re-apply #108's mechanism on top of the stack's structure; never
the reverse. Concretely for the eight backend TUs: start from the stack's
version (entry points already removed), then port #108's macro collapse to
what remains. For `gemm_kernels.cc`: keep #108's NTTP launcher table and add
the stack's new variants as rows. For `syev.hh`/`extensions.hh`/`extra.hh`:
the stack's declarations + #108's `BATCHLAS_ACCEPT_OWNING` form. Memory
notes `pr93-stacked-split` and `stale-main-ref-trap` describe the earlier
round of exactly this conflict.

**Steps.**
1. `git fetch origin && git checkout -b readiness/wp0-land-stack origin/stack/pr11-wp8-spmm && git merge origin/main`. Resolve the 14 files per the rule above. Commit the merge with a message that lists, per file, which side's mechanism won and why.
2. Configure `dev-tests`, build, run `ctest -LE slow`. Then the full suite once. Record the failing-name set; it must be a subset of the `main` baseline above plus whatever PR #107's description names as intentionally changed. `tests/route_vocabulary_tests.cc` and the 24 forced-kernel gemm tests #108 rerouted (`575007d`) must both pass — they are the two sides' own regression guards for this merge.
3. Re-run the vendor-free configuration once: `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF -DBATCHLAS_ENABLE_CUDA=ON`, build, `ctest -LE slow`. Expected ≈ 35/57 suites (host rows fail by design — `docs/design/vendor-free-status.md` §"Why the ctest pass count is the wrong instrument"). Confirm the GPU rows of `trsm_tests`, `spmm_tests`, `gemv_tests` pass.
4. Run `.github/ci/run_local_checks.sh` and, after `cmake --install build --prefix /tmp/inst`, `run_local_checks.sh /tmp/inst`.
5. Open ONE PR against `main` titled "Vendor independence 12/11: land the stack on main" (link #95–#107 in the body; they are already merged into the stack and need no action). Body: the per-file merge decisions from step 1, the failing-name diff from step 2, the vendor-free summary from step 3. After it merges, delete the twelve `stack/*` branches and `consolidate-vendor-independence`.

**Acceptance.** Merge into `main`; `git ls-files | grep -c route_` ≥ 19;
`include/batchlas/blas/dispatch/provider.hh` gone; every path cited in
`docs/` resolves:
```
grep -rhoE '`(src|include)/[A-Za-z0-9_/.-]+\.(hh|cc|cu)' docs/ | tr -d '`' | sort -u | while read p; do [ -e "$p" ] || echo MISSING $p; done   # prints nothing
```

**Do not** re-measure or re-tune anything in this WP. Landing, not improving.

---

## WP1 — Move the global-namespace public types into `namespace batchlas`

**Finding D-1.** Installed headers define these at global scope:

| header | global names |
|---|---|
| `include/batchlas/util/sycl-device-queue.hh` | `Device`, `DeviceProperty`, `DeviceType`, `Event`, `EventImpl`, `Policy`, `Queue`, `QueueImpl`, `Vendor` |
| `include/batchlas/util/sycl-span.hh` | `Span`, `is_std_array` |
| `include/batchlas/util/mempool.hh` | `BumpAllocator` |
| `include/batchlas/util/sycl-vector.hh` | `UnifiedVector` |
| `include/batchlas/util/reference-wrapper.hh` | `ReferenceWrapper` |
| `include/batchlas/util/workspace.hh` | forward-declares `Queue`, `Span`; `friend struct ::Queue;` at :163 |

In-tree usage: `Queue` in 253 files, `Event` 136, `UnifiedVector` 116,
`Span` 106, `Device` 104, `BumpAllocator` 50. 122 files already have
`using namespace batchlas;`. Only `src/util/sycl-util-impl.cc` is entirely
outside `namespace batchlas`. Explicit global qualifications to fix:
`workspace.hh:163`, `minibench_structured.hh:38` (`::Queue&`),
`bench_structured.hh:47,278` (`::Span<T>`), `src/util/queue-impl.cc`
(out-of-line `Queue::`/`Event::` definitions — wrap in the namespace).

**Design.**
- Wrap each header's declarations in `namespace batchlas { ... }`.
- At the END of each affected header add a compatibility block:
  ```cpp
  #ifndef BATCHLAS_NO_GLOBAL_NAMES
  using batchlas::Queue; using batchlas::Device; /* ... every name moved from this header */
  #endif
  ```
  A using-declaration that names the same entity as a `using namespace batchlas;` is not ambiguous, so in-tree code keeps compiling either way.
- Define `BATCHLAS_NO_GLOBAL_NAMES` PRIVATE on every in-tree target (`batchlas_build_options` in `cmake/BatchLASTargetHelpers.cmake` or wherever `batchlas_apply_object_options` lives, plus tests, benchmarks, python) so the tree itself never relies on the shim. Consumers get the shim by default in v0.2; document removal in v0.3 (`CHANGELOG.md`, WP9).
- Python bindings (`python/batchlas/bindings/support.hh` uses bare `Queue`) are in-tree: fix them, do not rely on the shim.

**Steps.**
1. Move the types; fix the seven explicit `::` sites; fix `sycl-util-impl.cc`.
2. Build `dev-tests` with `BATCHLAS_NO_GLOBAL_NAMES` defined in-tree; fix every error (expect mostly files that use the names outside any namespace and without `using namespace batchlas` — add the `using` or qualify).
3. Add `.github/ci/check_no_global_names.py`: parse every `include/**/*.hh`, strip comments, and fail on any `struct|class|enum|using|template … struct` declaration at brace-depth 0 that is not inside `namespace batchlas` and not inside the `#ifndef BATCHLAS_NO_GLOBAL_NAMES` block. Wire it into `ci.yml` (`public-headers` job) and `run_local_checks.sh`. Prove it: temporarily add `struct Foo {};` at global scope in a header, run the script, see it fail, revert.
4. Update `README.md` "Using the C++ API" and `docs/cpp-api.md` (§"The short version") — the example already writes `using namespace batchlas;`, which is now sufficient. Add one paragraph to `docs/cpp-api.md` naming `BATCHLAS_NO_GLOBAL_NAMES`.
5. Build `examples/consumer` against an install (`cmake --install build --prefix /tmp/inst`, then the recipe in `examples/consumer/CMakeLists.txt`) both with and without `-DBATCHLAS_NO_GLOBAL_NAMES=1` in the consumer's compile definitions; the consumer's `main.cc` must compile both ways (edit it to use `batchlas::Queue` explicitly).

**Acceptance.** `python3 .github/ci/check_no_global_names.py` passes; full
`ctest` failing-name set unchanged vs baseline; `consumer_package_tests`
passes; `nm -C build/src/libbatchlas_core.so | grep -c ' Queue::'` is 0 and
`grep -c ' batchlas::Queue::'` > 0.

---

## WP2 — CI that compiles, links, runs tests and installs

**Finding D-4.** `ci.yml` runs three static Python checks and nothing else.

**Two jobs, one runner.**

### 2a. Self-hosted GPU job (primary gate)
- Install the GitHub Actions runner on this box as a systemd service (`~/actions-runner`, labels `self-hosted,linux,x64,cuda,rtx4090`). Runner runs on the host, not in Docker: the compiler is a self-built tree at `/opt/dpcpp-cuda` that would otherwise need bind-mounting. Document the install in `docs/ci.md` including the `sycl-ls` check and `OPENBLAS_CORETYPE`. (This step is the one part a subagent cannot finish alone — it needs the repo's runner registration token from Settings → Actions → Runners. Prepare everything, and stop with the exact command the user must run.)
- Workflow job `gpu-build-test` in `ci.yml`: `runs-on: [self-hosted, cuda]`, `concurrency: { group: gpu-runner, cancel-in-progress: false }` (two harnesses on the device corrupt each other's timings and can deadlock — `docs/perf/README.md` measurement rules), timeout 120 min.
  1. `cmake --preset dev-tests -DBATCHLAS_CCACHE_SHARE_ACROSS_TREES=ON` (ccache on the host makes per-PR builds incremental; without it a clean build is ~1 h).
  2. `cmake --build --preset dev-tests`.
  3. `ctest --test-dir build/presets/dev-tests -LE slow --output-on-failure --output-junit ctest.xml` — **do not fail on non-zero exit**; instead run `.github/ci/compare_failures.py ctest.xml tests/known-failures.txt`, which fails the job only when a test NOT in `tests/known-failures.txt` failed, and *warns* when a listed test passed (so the list can be pruned). Commit `tests/known-failures.txt` seeded with the five names in §0.
  4. `cmake --install build/presets/dev-tests --prefix "$RUNNER_TEMP/inst"` then `.github/ci/run_local_checks.sh "$RUNNER_TEMP/inst"` — the exported-package check the current CI header says it "cannot run".
  5. Upload `ctest.xml` and `LastTest.log` as artifacts.
- Nightly (`schedule:`) variant runs the full `ctest` (with `slow`) and the vendor-free configure/build/`-LE slow` from WP0 step 3.

### 2b. Hosted CPU-only compile job (secondary)
- `ubuntu-22.04`, install `intel-oneapi-compiler-dpcpp-cpp` and oneDPL via the apt recipe in `AGENTS.md` §2a (cache the apt download with `actions/cache`).
- `cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DBATCHLAS_ENABLE_CUDA=OFF -DBATCHLAS_BUILD_TESTS=OFF -DBATCHLAS_CPU_TARGET=none` then `cmake --build build --target batchlas_components`. `CPU_TARGET=none` compiles all host C++ and links every `.so` without any device pass — it catches every compile and host-link error in minutes; kernel bodies are still parsed. Measure the wall time on the first run; if under 40 min, add a nightly variant with `BATCHLAS_CPU_TARGET=spir64_x86_64` and `BATCHLAS_BUILD_TESTS=ON` running `ctest -LE slow` (expect ~48/56 green per `tests/README.md`; use the same known-failures mechanism with a second list `tests/known-failures-cpu.txt`).
- Then `cmake --install` + `run_local_checks.sh --package`, and build `examples/consumer` against that prefix with `-DBATCHLAS_CONSUMER_USE_FSYCL=OFF`.

**Steps.** Write `compare_failures.py` first (pure Python, unit-tested with a fabricated junit file); then `ci.yml`; then the runner docs. Rewrite the `ci.yml` header comment to describe what is now covered and what still is not (no AMD, no Intel GPU, one CUDA version).

**Acceptance.** A PR that deliberately breaks a header (e.g. removes a `;`) turns 2b red; a PR that deliberately breaks a kernel result (e.g. flips a sign in `src/sycl/gemm/register_128x128.hh`) turns 2a red with the new failing name in the log; a no-op PR is green. Include all three runs' links in the WP's PR.

---

## WP3 — Versioned, bounded ABI: SOVERSION, one exported target, hidden visibility

**Finding D-2.** 14 `.so` (250 MB), no `VERSION`/`SOVERSION`, default
visibility (4,895 dynamic symbols in `libbatchlas_extensions_cta.so`), 16
targets exported into `BatchLAS::` that `BatchLASConfig.cmake.in` itself says
are not independently usable. The components have 14 symbol cycles and are
linked with `-Wl,--no-as-needed` "until the .so merge lands"
(`src/CMakeLists.txt:280-292`). The device link is per-`.so` and
single-threaded; merging everything into one `.so` for *development* would
make every incremental edit relink the world (measured 2026-08-05: 198 s for
one file).

**Phase 3a — SOVERSION (trivial, do first).**
In `batchlas_configure_component()` (`src/CMakeLists.txt:76`):
`set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})`.
Change `write_basic_package_version_file(... COMPATIBILITY SameMajorVersion)` in
`cmake/BatchLASPackaging.cmake:115` to `ExactVersion` while the major is 0
(SemVer: 0.x promises nothing across minors). Acceptance: `readelf -d
build/src/libbatchlas_core.so | grep SONAME` shows `libbatchlas_core.so.0`;
install tree has the `.so`, `.so.0`, `.so.0.1.0` triple.

**Phase 3b — One public target.**
Add `option(BATCHLAS_MONOLITHIC_LIBRARY "Link all components into one libbatchlas.so" OFF)`.
When ON: `add_library(batchlas SHARED $<TARGET_OBJECTS:batchlas_core_obj> … all in BATCHLAS_OBJECT_LIBS>)` instead of `INTERFACE`; the component `SHARED` targets are not created; `batchlas_install_package()` installs and exports only `batchlas` (+ `batchlas_sycl_options` if its INSTALL_INTERFACE is non-empty — check `cmake/BatchLASTargetHelpers.cmake`; if it is empty on the install interface, drop it from the export). Delete the `--no-as-needed` workaround in that mode. When OFF (dev default): keep today's split, but move the component targets into a second export set `BatchLASComponentTargets` with `NAMESPACE BatchLAS::_component_::`, so `BatchLAS::batchlas` stays the only sensible name and the config file's `check_required_components` still fails for anyone naming a component.
Set `BATCHLAS_MONOLITHIC_LIBRARY=ON` in the `cuda` preset and in a new `release` preset (`Release`, tests off, monolithic, `BATCHLAS_NVIDIA_ARCH` from the machine). **Measure** the monolithic device link time (`time cmake --build … --target batchlas`) and record it in the PR; if it exceeds 15 min the release preset stays the only place it is on.
Acceptance: `BatchLASTargets.cmake` in a monolithic install lists exactly one library target; `examples/consumer` and `python/` build and run against it; `consumer_package_tests` passes in both modes (parametrise the ctest over the option).

**Phase 3c — Hidden visibility (monolithic mode only).**
`target_compile_options(... -fvisibility=hidden -fvisibility-inlines-hidden)` on every object library when monolithic. `include(GenerateExportHeader)`; `generate_export_header(batchlas EXPORT_FILE_NAME include/batchlas/export.hh EXPORT_MACRO_NAME BATCHLAS_API)`; install it next to `backend_config.h`. Annotate:
- classes with out-of-line members: `Queue`, `Device`, `Event`, `Matrix`, `MatrixView`, `VectorView`, `UnifiedVector`, `BumpAllocator` (`class BATCHLAS_API Queue`).
- every explicit instantiation declared in the public headers (`extern template class BATCHLAS_API Matrix<float, MatrixFormat::Dense>;` etc. — the definitions are in `src/matrix.cc:2413-2440`).
- the 21 op families: the instantiation macros in `src/util/template-instantiations.hh` emit the definitions; the declarations in `include/batchlas/blas/functions/*.hh` and `extensions.hh`/`extra.hh`/`linalg*.hh` get `BATCHLAS_API`. Do it macro-by-macro, rebuilding `examples/consumer` and `python/` after each family: an undefined reference at consumer link is the to-do list.
- SYCL kernels need no host export; `SYCL_EXTERNAL` device functions are unaffected.
Acceptance: `nm -D --defined-only libbatchlas.so | wc -l` reported before/after (expect an order of magnitude down); consumer + python link; full `ctest` failing-name set unchanged. Do not attempt hidden visibility in split mode — cross-`.so` references would each need the right macro and the mode is being retired.

---

## WP4 — Error model and per-item convergence status

### Phase 4a — `batchlas::` exception hierarchy
New header `include/batchlas/error.hh` (installed; included from `sycl-device-queue.hh` so every public header sees it):
```cpp
namespace batchlas {
struct exception { virtual ~exception() = default; };          // tag base only — MUST NOT derive from std::exception
class invalid_argument : public std::invalid_argument, public exception { using std::invalid_argument::invalid_argument; };
class error            : public std::runtime_error,    public exception { using std::runtime_error::runtime_error; };
class unsupported      : public error { using error::error; };  // no backend/route/kernel serves this shape/type/device
class device_error     : public error { using error::error; };  // SYCL, CUDA, cuBLAS, cuSOLVER, cuSPARSE, ROCm, MKL failures
class workspace_error  : public error { using error::error; };  // BumpAllocator overflow, buffer_size mismatch
class convergence_error: public error { using error::error; };  // only from explicit check modes (see 4b)
}
```
The tag base deliberately does not derive from `std::exception`, so `catch (const std::exception&)` (14 sites in tests) stays unambiguous and `catch (const batchlas::exception&)` catches everything BatchLAS throws.

Mechanical rewrite of the 472 throw sites (`grep -rn "throw std::" src include --include=*.cc --include=*.hh --include=*.cu`; top files: `src/matrix.cc` 50, `gesvd_blocked.cc` 22, `iluk.cc` 19, `src/sort.hh` 18, `band_reduction.cc` 18):
- `throw std::invalid_argument(` → `throw batchlas::invalid_argument(` (251 sites, no judgement needed).
- `throw std::runtime_error(` (221): classify by message with these regexes, in order, first match wins: `/not supported|unsupported|no (backend|route|provider|kernel)|does not support|cannot (route|dispatch|serve)/i` → `unsupported`; `/sycl|cuda|cublas|cusolver|cusparse|rocblas|rocsolver|rocsparse|hip|mkl|device/i` → `device_error`; `/workspace|BumpAllocator|buffer_size|pool|allocat/i` → `workspace_error`; `/converge/i` → `convergence_error`; else `error`. Write the classifier as a script, apply it, then read every `unsupported`/`device_error` diff hunk by hand — the regexes are a first pass.
- `std::out_of_range` (5) → `batchlas::invalid_argument`; `std::bad_alloc` (3) → leave.
- `include/batchlas/util/miniacc.hh:675` `std::exit(0)`: replace with a return path; a library header must not terminate the process.
- `struct [[nodiscard]] Event` in `sycl-device-queue.hh:185`. This makes every `Event`-returning call warn on discard. Build with `-Wall` locally (the tree sets no warning flags — do not add `-Werror`) and fix in-tree discards by `(void)` where the discard is intentional (fire-and-forget on an in-order queue) — say so in a comment the first time per file.
- Tests: `tests/error_model_tests.cc` — for one representative site per class, assert the thrown type via `EXPECT_THROW(…, batchlas::unsupported)` AND that `catch (const std::exception&)` still catches it AND `catch (const batchlas::exception&)` does. Break it: change one throw back to `std::runtime_error` and watch the typed expectation fail.
- Docs: rewrite `docs/cpp-api.md` §"What gets thrown" (line ~926) around the hierarchy.

### Phase 4b — `info` on the iterative routines
Today only `potrf` and `getrf` take `Span<int32_t> info` (`potrf.hh:33-62` documents the convention and the old-arity forwarder that keeps 4-arg callers compiling). Extend the same convention, same forwarder pattern, to: `syev`, `syevx`, `gesvd`, `steqr`, `stedc`. Semantics: one `int32` per batch item; `0` converged; `>0` LAPACK-like (`steqr`/`stedc`/`syev`: number of off-diagonal elements that did not reach zero, or `1` if that count is not tracked; `gesvd`: `1` for a `bdsqr`/Jacobi item that hit its sweep cap); empty span = not requested (no cost).

Existing plumbing to reuse — do not invent new flags:
- `src/extensions/bdsqr.cc:99-334` already computes `fail_flags[b]`, then only *checks* them under an env var. Surface them.
- `src/extensions/steqr_cta.cc:219-232` has a per-item `status[]` and an opt-in `BATCHLAS_STEQR_CTA_CHECK` that `ctx.wait()`s and throws. Surface `status` into `info`; keep the check mode but make it throw `convergence_error` (WP5 will fold the env var into settings).
- `src/extensions/stedc_secular.cc:622` `assert(converged && …)` and `:682` `iter >= 100`: turn the assert into a per-item flag set (an `assert` in device code is a no-op in release — this is a silent wrong answer today).
- `src/extensions/gesvdj_cta.cc:494` and `src/extensions/syev_jacobi_cta.cc:323`: the sweep loops have no flag. Add `converged = off_norm <= tol` after the loop; write `info[b] = !converged`.
- `syev_two_stage.cc`, `syev_blocked.cc`, `syev_cta*.cc`: forward the tridiagonal solver's `info`. Vendor arms (`cusolver.cc` `syevjBatched`/`syevd`, `netlib_lapack.cc`): cuSOLVER already returns an info array — copy it into the caller's span instead of dropping it; netlib returns scalar `info` per call — write it per item.
- `linalg::eigh`/`linalg::svd` value-returning wrappers (`include/batchlas/blas/linalg*.hh`): add an `info` member to their result structs.
- Python: expose as an optional `return_info=False` kwarg on `syev`/`gesvd`/`syevx` returning a NumPy `int32` array.

Tests, one per routine in the existing suite file: (i) an ordinary batch → all `info == 0`, asserted on a span pre-poisoned with `-1`; (ii) a forced non-convergence — set the routine's `max_sweeps`/`max_iter` param to 1 on a batch that needs more (the `StressExtremeMagnitudes` inputs in `tests/steqr_tests.cc` are a ready source) → at least one `info != 0`, and the eigenvalues for `info == 0` items still pass the residual check. Break (ii) by hard-coding `info[b] = 0` in one kernel and watch it go red.

---

## WP5 — Environment variables resolved once into a settings object

**Finding A-3.** 85 distinct `BATCHLAS_*` names read at 72 `getenv` sites,
including `BATCHLAS_SKIP_POINTER_CHECKS` and `BATCHLAS_LATRD_GRID_FORCE_UNSAFE`.
Regenerate the list: `grep -rhoE '"BATCHLAS_[A-Z0-9_]+"' src include | tr -d '"' | sort -u`.
15 test files mutate the environment through `ScopedEnvVar`
(`include/batchlas/util/env.hh:108`), so "read once" must be re-readable.

**Design.**
- `include/batchlas/settings.hh`: `struct Settings` with typed fields grouped as `routing` (per-op `Route` or legacy spelling, as strings parsed by the existing `parse_route_env`), `geometry` (the `*_WG`, `*_TILE*`, `*_GROUPS`, `*_KD`, `*_NB`, `*_MIN_N` ints), `debug` (`DEBUG*`, `DUMP_*`, `*_TRACE*`, `*_PROFILING`, `*_CHECK`), `unsafe` (`SKIP_POINTER_CHECKS`, `LATRD_GRID_FORCE_UNSAFE`, `CTA_DEBUG_SYNC`). Every field has the default the call site uses today.
- `const Settings& batchlas::settings()` — reads the whole environment once under `std::call_once`, thread-safe. `void batchlas::configure(const Settings&)` — programmatic override; allowed until the first `Queue` is constructed, throws `batchlas::error` afterwards. `void batchlas::detail::reload_settings()` — re-reads the environment; called from `ScopedEnvVar`'s constructor and destructor so the 15 test files need no change.
- `option(BATCHLAS_ALLOW_UNSAFE_ENV "Honour BATCHLAS_SKIP_POINTER_CHECKS / *_FORCE_UNSAFE / CTA_DEBUG_SYNC from the environment" OFF)`. When OFF, those three fields keep their defaults regardless of the environment and a set variable produces one `stderr` warning at first use. ON in the `dev*` and `benchmarks` presets, OFF in `cuda`/`release`.
- Replace every `getenv`/`env_*_or` call in `src/` with a `settings().x.y` read. `parse_provider_env`/`parse_route_env` keep their parsing but take the string from `Settings`, not from `getenv`. `include/batchlas/util/env.hh` becomes an implementation detail (`src/util/`), except `ScopedEnvVar` which tests use — keep it public.
- Document every field in `docs/cpp-api.md` (new section "Configuration"), generated from the struct by a small script so it cannot drift (`scripts/gen_settings_doc.py`; CI check compares its output with the committed section).

**Acceptance.** `grep -rn "getenv" src include | grep -v "src/util/settings.cc\|ScopedEnvVar"` is empty; `tests/route_vocabulary_tests.cc` and every other `ScopedEnvVar` user still passes; new `tests/settings_tests.cc` proves (a) `configure()` beats the environment, (b) `configure()` after a `Queue` exists throws, (c) with `BATCHLAS_ALLOW_UNSAFE_ENV=OFF` setting `BATCHLAS_SKIP_POINTER_CHECKS=1` does not skip the check (poison a host pointer, expect `invalid_argument`). Full `ctest` failing-name set unchanged.

---

## WP6 — Realise the measured native wins (after WP0)

**Finding P-2.** On the branch, `geqrf`/`orgqr` default to cuSOLVER with
3.24×/7.85× native geomeans unrealised (`docs/perf/qr.md` "Open debts" 1);
`potrf`'s `preferred()` is all-false (`docs/perf/potrf.md` debt 1); Level-3
has no `RouteTable` (`docs/perf/level3.md` debt 3).

**This is measurement work.** The rules are in `docs/perf/README.md`
("Measurement rules") and must be followed literally: at saturation, one
harness on the box, JIT warmed, interleaved A/B medians, relative-sd gate,
bracketing non-winner at every window edge. Verify a routing change by
**diffing the chosen route** (`scripts/route_diff.sh`) over the shape grid,
not by timing alone. Always at large batch — batch=1 numbers are not wanted.

**Order and per-op recipe** (one PR each):
1. `geqrf` float, then cfloat: harness `benchmarks/geqrf_benchmark.cc`. Sweep the grid in `docs/perf/qr.md` §"cta-vs-blocked-crossover" plus a bracketing row past each proposed edge. Flip the cells in `preferred()` of `include/batchlas/blas/dispatch/route_geqrf.hh` (path per WP0 tree). Run `ctest -L blas -R '^(geqrf|orgqr|ormqr)'` and the full suite once. Record the grid as a table in the header next to the predicate (the `syev.hh` convention), and update `docs/perf/qr.md` "Open debts".
2. `orgqr` (same harness family, `benchmarks/orgqr_benchmark.cc`).
3. `potrf` float: `docs/perf/potrf.md` names the three-part gate that "has not been run" — run it. Note debt 5: `Uplo::Upper` is refused by `supports()`; that stays.
4. Level-3 `RouteTable`s: add the tables *unwired* with an equivalence test against the existing `if`-chains (`docs/perf/level3.md` debt 3 says this is cheap); wiring is a separate measured PR per op.

**Acceptance per op.** PR carries: the sweep CSV (committed under `benchmarks/results/`), the before/after `route_diff` output, the failing-name diff of the op's suites. A flip without a bracketing non-winner is rejected.

---

## WP7 — Decouple the compiler pin, tag v0.2.0, build a wheel

### 7a — Consumer may use GCC
`README.md` and `cmake/BatchLASConfig.cmake.in:22-70` say the whole consuming
project must use the same clang because `requires`-clauses are mangled into
symbol names (Itanium ABI, clang ≥ 16) and GCC does not mangle them. Only
symbols *defined in the `.so`* and *referenced by the consumer* matter:
member templates that are `inline` in the header (`matrix.hh:133-244`, the
`DenseMatrixFormat`/`CsrMatrixFormat`-constrained accessors) are instantiated
by the consumer itself and are harmless. Find the real offenders:
```
nm -C build/src/libbatchlas_core.so | grep -i 'requires\|Q[0-9]' | head     # clang encodes the clause as a Q… component
```
For each constrained member that is declared in a header and defined in
`src/matrix.cc` (or any other `.cc`), replace the `requires` with an
`std::enable_if_t` default template argument — mangled identically by both
compilers. Do the same for any constrained free function that is explicitly
instantiated (`grep -rn "requires" include/batchlas/blas/linalg-ops.hh:392,420`,
`extensions.hh:2330,2339`, and the `BATCHLAS_ACCEPT_OWNING`/`BATCHLAS_DISPATCH_ON_QUEUE`
macro overloads in `queue-dispatch.hh:258,316,331` — those are header-inline
and should be fine; verify rather than assume).

Acceptance test, added to CTest as `consumer_gcc_tests` (label `packaging;slow`):
install with DPC++, then configure `examples/consumer` with
`-DCMAKE_CXX_COMPILER=g++ -DBATCHLAS_CONSUMER_USE_FSYCL=OFF`, build, run
`hello_batched_gemm` with `LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib`. Then
downgrade the config-file compiler-mismatch message from `WARNING` to `STATUS`
when the mismatch is clang→GCC and the `requires` audit passed, and rewrite
README "Four things that will bite you" item 1.

### 7b — Portable device code
Today `BATCHLAS_NVIDIA_ARCH` is one arch (default `sm_50`, tested `sm_89`).
(i) Make it a list: `-DBATCHLAS_NVIDIA_ARCH="sm_80;sm_89;sm_90"` →
`-fsycl-targets=nvidia_gpu_sm_80,nvidia_gpu_sm_89,nvidia_gpu_sm_90` in
`cmake/BatchLASDetectSYCL.cmake:437-470`. (ii) Determine whether the DPC++
CUDA fat binary carries PTX that the driver can JIT for a newer arch:
`clang-offload-extract`/`cuobjdump --list-ptx` on a built `.so`, and a run on
device 1 with `SYCL_DEVICE_ALLOWLIST` after building for `sm_80` only. Record
the answer in `docs/cpp-api.md` — if PTX is embedded, a single `sm_80` build
covers every newer NVIDIA GPU at a JIT cost and that becomes the release
setting; if not, the release builds the list.
(iii) oneDPL: `find_package(oneDPL CONFIG QUIET)` first; if not found,
`FetchContent` from `https://github.com/uxlfoundation/oneDPL` (header-only)
pinned to a tag; keep `ONEDPL_ROOT` as an override. Drop the hardcoded
`/opt/intel/oneapi/dpl/latest/include` from the search list's first position
(keep it as a hint).

### 7c — Release furniture and v0.2.0
`CHANGELOG.md` (Keep-a-Changelog, first entry lists WP0–WP7), bump
`project(BatchLAS VERSION 0.2.0)`, tag `v0.2.0` (annotated), a
`release.yml` workflow that on a `v*` tag builds the monolithic `release`
preset on the self-hosted runner and attaches `libbatchlas.so.*` + headers
as a tarball. Stability statement in `README.md`: what is covered by SemVer
from 0.2 (the 21 op headers, `linalg`, `Queue`/`Matrix`/`Span`), what is
not (`internal/`, `util/minibench*`, `tuning_params.hh`).

### 7d — Wheel
`pyproject.toml` at the repo root with `scikit-build-core` (`pip install scikit-build-core` on the box first): `cmake.args = ["--preset", "release", "-DBATCHLAS_BUILD_PYTHON=ON"]`, wheel contents = `python/batchlas` + the `.so`s (`install(TARGETS …)` from WP3 with `INSTALL_RPATH=$ORIGIN`), `libsycl.so` and the DPC++ runtime libs copied in from `/opt/dpcpp-cuda/lib` (`auditwheel repair` cannot see a custom toolchain — do the copy in CMake, and document that the wheel is `linux_x86_64`, not `manylinux`). `python/CMakeLists.txt` currently copies files into `build/python`; add an `install()` rule that scikit-build-core picks up. Acceptance: `pip wheel . -w dist/` on the box, then in a fresh venv on the same box `pip install dist/*.whl && python -c "import batchlas; print(batchlas.available_backends())"` — with **no** `PYTHONPATH` and no `LD_LIBRARY_PATH`. PyPI publication is out of scope.

---

## WP8 — Half, bfloat16, and mixed-precision solve (after WP0, WP4)

**Finding P-1.** Scalar set is `{float, double, complex<float>, complex<double>}`
(`src/util/template-instantiations.hh:69-72`); no `sycl::half`, no
`bfloat16`, no TF32, no refinement. Python rejects anything else
(`python/batchlas/_api.py:59-62`).

**Phase 8a — `half` gemm through cuBLAS (smallest complete increment).**
`src/backends/cublas.cc:70-90` already calls `cublasGemmEx` /
`cublasGemmStridedBatchedEx` with a per-`T` data-type mapping. Add
`sycl::half` → `CUDA_R_16F` inputs, `CUBLAS_COMPUTE_32F` (fp32 accumulate),
`alpha`/`beta` as `float`. Add a `BATCHLAS_FOR_EACH_HALF_TYPE` macro and
instantiate ONLY `gemm` and `gemm_buffer_size` for it (plus `MatrixView<sycl::half, Dense>`,
`Matrix<sycl::half, Dense>` in `src/matrix.cc`, and `UnifiedVector<sycl::half>`).
`GemmOptions<sycl::half>` carries `float alpha, beta`. `supports()` for the
native route returns false for half until 8b; `preferred()` untouched.
Python: accept `float16` in `_normalize_dtype` for `gemm` only.
Tests: `tests/gemm_half_tests.cc` — compare against a `double` reference with
relative tolerance `2e-3` at `n ∈ {16, 64, 256}`, batch 1024, both layouts,
NN/NT/TN; and a `[[maybe_unused]]` accumulation-order check that a sum of
4096 ones is exactly 4096 (fp32 accumulate) and not 2048 (fp16 accumulate).
Break: switch compute type to `CUBLAS_COMPUTE_16F` and watch the second test fail.

**Phase 8b — native half gemm.** `src/sycl/gemm/register_128x128.hh:93` is
`template <typename T, bool AlignedFastPath>` with `T accum[…]` at `:174`.
Split into `TIn` and `TAcc`: loads/`Packet4` in `TIn`, `accum` and the
`alpha/beta` epilogue in `TAcc`, store converting back. Keep `T == TAcc` for
the four existing types so their code is byte-identical (check with the
`ptx-codegen-comparison` skill in `.github/skills/`). Register-residency rule
from the tree: thread tile must shrink as the scalar widens — for half it may
grow; try 8×8 first, measure, then 8×16. Route: `supports()` true for half on
GPU; `preferred()` measured against 8a's cuBLAS path per the WP6 rules.

**Phase 8c — bfloat16.** Repeat 8a/8b with `sycl::ext::oneapi::bfloat16` →
`CUDA_R_16BF`. Guard with a configure-time check that the DPC++ in use
provides the extension (`__has_include(<sycl/ext/oneapi/bfloat16.hpp>)`).

**Phase 8d — mixed-precision `gesv`.** New entry point
`gesv_refine<B, double>(ctx, A, B, X, info, opts)` (LAPACK `dsgesv` shape):
copy `A` to a `float` workspace, `getrf<float>` + `getrs<float>`, then iterate
`r = b − A·x` in `double` (`gemm<double>`), solve the correction in `float`,
`x += dx`, until `‖r‖ ≤ n·ε_double·‖A‖‖x‖` or `max_iter` (default 30) — write
`info[b] = iterations` or `−1` if not converged (LAPACK convention). Test:
well-conditioned random batch (`random_cond` in `src/extra/random_cond.cc`
with κ = 1e3) reaches `double` accuracy in ≤ 5 iterations; κ = 1e9 sets
`info = −1` and the result still equals the pure-`float` solve.

---

## WP9 — Hygiene (after WP0)

All small; one PR.
1. **Root working documents.** `git mv` the 19 non-README/AGENTS `*.md` at the repo root into `docs/archive/` with an `index.md` that gives each one line and the tag/PR it fed. Do not delete — they are cited by memory and by `docs/perf/`. Update any relative link that breaks (`grep -rn "\](\.\./\|\](SYEV\|\](GESVD" docs README.md`).
2. **Provider doc drift (A-4).** After WP0 the three `Provider`-carrying headers are gone (WP0's merge deletes `context.hh`; `provider.hh`/`env.hh` are already absent on the stack tip); verify with `grep -rn Provider include/` (expect only comments in `dispatch.md`'s history section). If any survives, migrate it to `Route` per `docs/perf/dispatch.md`.
3. **`scripts/rocm_syntax_check.sh`** arrives with WP0. Run it once (`/opt/rocm*` headers are on this box per memory) and make it a `-LE slow` ctest (`rocm_syntax_check`, label `packaging`) so it cannot vanish again unnoticed.
4. **Project furniture.** `CONTRIBUTING.md` (build presets, test scoping rule, the blind-guard checklist, comment-density norm, PR expectations incl. failing-name diff), `SECURITY.md` (contact + "no untrusted input hardening claimed"), `.github/CODEOWNERS` (`* @jonasdelacour`), `.github/ISSUE_TEMPLATE/bug.yml` (asks for `sycl-ls`, compiler realpath, CUDA version, the configure line, and the failing test name).
5. **README "Tested platforms"** — add the CI rows from WP2 so the table describes what is enforced.
6. **API reference.** `BATCHLAS_BUILD_DOCS` builds nothing; wire Doxygen (`docs/Doxyfile.in`, `INPUT = include/batchlas`, `EXTRACT_ALL = NO`, `WARN_IF_UNDOCUMENTED = YES`) and a `docs` target; run once and commit the count of undocumented public entities as the baseline in `CONTRIBUTING.md`. Publishing (GitHub Pages) is optional.

---

## Out of scope, deliberately

- AdaptiveCpp support, AMD/Intel GPU CI: no hardware, no second toolchain here. Record as "not supported" in README rather than "untested".
- Fixing the five baseline test failures is not a WP here; they are tracked by `tests/known-failures.txt` (WP2) and each needs its own diagnosis (see memory notes on `lanczos` two-column multiply, `steqr` extreme magnitudes).
- CUDA-graph capture, multi-GPU, heterogeneous (pointer-array) batches: real gaps, but each is a design effort, not a fix.

## Hand-off checklist per WP

- [ ] branch `readiness/wpN-<slug>` from `main` (post-WP0 for WP6/8/9)
- [ ] every acceptance command in this file run and its output in the PR body
- [ ] failing-name diff vs `tests/known-failures.txt`, not a count
- [ ] the "break" that proved each new test, named in the PR
- [ ] `docs/` updated where the WP changed a documented contract
