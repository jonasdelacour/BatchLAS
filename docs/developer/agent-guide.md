# Agent environment guide {#dev_agent_guide}

> **Status:** current · RTX 4090 box (jrhnbr) unless a line says otherwise · digest: @ref dev_agent_core; read by section (`scripts/agent/doc_section.py agent-guide §N`).

Working rules for agents and contributors: toolchain, build, testing, routing, measurement, GPU
kernel facts, repository conventions and agent workflow. Requirements, presets and every CMake
option are in @ref guide_build; this page holds the rules and traps.

**TL;DR.** Exercised: self-built CUDA DPC++ (`/opt/dpcpp-cuda`), CUDA 13.2, RTX 4090 (sm_89),
Ubuntu 26.04. NVIDIA needs a CUDA-capable DPC++ (§2b): stock oneAPI `icpx` builds a CPU-only
library without complaint. `sycl-ls` must list `[cuda:gpu]`; `-- Using SYCL targets:
spir64_x86_64` at configure means CPU-only. `-DBATCHLAS_ENABLE_CUDA=ON` aborts without a CUDA
device; the default `AUTO` does not. Apple GPU (AdaptiveCpp Metal): @ref guide_build, traps below.

## 1. Prerequisite Packages

| Component | apt | dnf | pacman |
| --- | --- | --- | --- |
| BLAS/LAPACK | libblas-dev liblapack-dev | blas-devel lapack-devel | blas lapack |
| CBLAS & LAPACKE headers | liblapacke-dev | lapack-devel | lapacke |
| Build tools | build-essential cmake git | `@development-tools` cmake git | base-devel cmake git |
| Git LFS (`benchmarks/results/`) | git-lfs | git-lfs | git-lfs |

Ubuntu's `libopenblas-dev` omits `lapacke.h`: install `liblapacke-dev` too. No package: §4.

## 2. Installing a SYCL Compiler

### 2a. Intel oneAPI DPC++/C++ (icpx): Intel GPUs and CPU-only builds

Wrong for NVIDIA (no CUDA adapter). From Intel's APT repo (`https://apt.repos.intel.com/oneapi`)
install only `intel-oneapi-compiler-dpcpp-cpp`; `source /opt/intel/oneapi/setvars.sh` per shell.

### 2b. CUDA-capable DPC++: NVIDIA GPUs

The Codeplay **oneAPI for NVIDIA GPUs** plugin on top of §2a, or `intel/llvm` built with `--cuda`
(`/opt/dpcpp-cuda` on the 4090 box; every number in the repository), plus a CUDA Toolkit (13.2).
Configure with `-DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++`; a consumer of an installed
BatchLAS uses the same compiler for its whole project (`docs/cpp-api.md`).

### 2c. AdaptiveCpp: the second SYCL implementation

AdaptiveCpp 25.10 (@ref design_sycl_implementations): presets `acpp` / `acpp-tests` (trees
`build-acpp*`). CI's `acpp-compile` job compiles every target (`generic`) and runs nothing. An acpp
tree is gated against `tests/known-failures-acpp.txt` (`compare_failures.py --sycl-impl ACPP`).

## 3. Verifying the Toolchain

`sycl-ls` lists your GPU (e.g. `[cuda:gpu]`); `cmake --version` >= 3.17 (3.21 for presets);
`pkg-config --exists lapacke cblas` succeeds.

## 4. Building netlib LAPACKE/CBLAS from Source (if distro packages are unavailable)

```bash
git clone https://github.com/Reference-LAPACK/lapack.git && cd lapack
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLAPACKE=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j8 && sudo cmake --install build && sudo ldconfig
```

## 5. CMake Configuration Hints

LAPACKE and CBLAS are optional (`find_netlib_libs`, `cmake/BatchLASDependencies.cmake`; without
them NETLIB is off). Elsewhere on disk: `-DLAPACKE_LIBRARY=<path> -DCBLAS_LIBRARY=<path>`.

## 6. Quick Smoke Test

```bash
scripts/seed_build_tree.sh ~/BatchLAS/build build   # new worktree, BEFORE editing (§7); else cmake -B build .
scripts/b stedc_tests                               # library + one test binary
ctest --test-dir build -R '^stedc_tests(_heavy)?$' --output-on-failure
scripts/affected_tests.py --print-build-cmd --print-ctest-cmd --build-dir build   # what a diff can break
ctest --test-dir build -LE 'slow|heavy'             # broadest iteration scope (quick tier)
scripts/gate.sh                                     # before pushing (4090 box: one tree per GPU)
scripts/gate.sh --seed-from build-dev-tests --seed-allow-dirty   # first run
sh .github/ci/run_local_checks.sh [PREFIX]         # touched cmake/, include/ or install rules
```

- `scripts/b`: named targets only (none = exit 64), `-j8` cap (`BATCHLAS_ALLOW_WIDE=1` lifts it),
  memory admission (§7), exit 70 when a requested binary is missing, log in `<build>/logs/`.
- `scripts/gate.sh` owns two persistent trees: `build-gate` (full targets, full ctest, ledger diff)
  and `build-gate-vf` (vendor-free, `-LE slow`), both relinking every executable
  (`BATCHLAS_LINK_DEPENDS_NO_SHARED=OFF`). It holds the build lock through ctest and sets
  `ONEAPI_DEVICE_SELECTOR` (§8). After both builds it runs the two ctests at once, vf on the
  first visible GPU and full on the second, each serial with `DPCPP_CPU_NUM_CUS`=cores/2;
  `CUDA_VISIBLE_DEVICES=0` or `--serial` runs them one after the other. Read the peak RSS line
  (`<tree>/gate/rss.tsv`, `docs/ci.md`). Seed a missing gate tree only from a `dev-tests` tree
  (plain `build/` differs in `BATCHLAS_ALLOW_UNSAFE_ENV`). Never `--parallel` (§8) here: it was
  validated only on threadripper02, where `scripts/agent/dispatch.sh` now forbids it.
- `run_local_checks.sh PREFIX` adds the `--package` checks CI cannot run (needs
  `cmake --install build --prefix PREFIX`); `ctest -R '^consumer_package_tests$'` builds
  `examples/consumer/` against an install.

## Known Pitfalls

- **Silent CPU-only build on an NVIDIA box**: no CUDA adapter; `spir64_x86_64` at configure (TL;DR).
- **oneAPI 2026 icpx without the Codeplay plugin** ships no CUDA UR adapter: put
  `/opt/dpcpp-cuda/lib` first on `LD_LIBRARY_PATH`, ahead of `setvars.sh`, for configure, build
  and every run.
- **icpx defaults to `-fp-model=fast`**, which stops LLVM inlining libspirv builtins (`sycl::fma`,
  barriers, id queries become CALLs; cfloat gemm 6-15x slower, tests green). The build passes
  `-ffp-model=precise`; `ctest -R '^device_calls_tests$'` (`scripts/check_device_calls.py`) fails
  on any `__spirv_*` call. Never add a later `-ffp-model=fast`/`-ffast-math`.
- **Missing `lapacke.h`**: §1. **Several BLAS providers**: `update-alternatives --config libblas.so.3`.
  **Device not found**: driver and SYCL adapter must match the compiler version.
- **`libsycl.so.9: cannot open shared object file`**: built binaries carry the runtime as
  `RUNPATH`; a bare `sycl-ls` does not. Export `LD_LIBRARY_PATH=<dpcpp-prefix>/lib`.
- **Wrong double results from NETLIB**: some OpenBLAS builds pick a broken dgemm (0.3.20 on
  Cooperlake). Configure runs a health check (`cmake/BatchLASBlasHealthCheck.cmake`) and ctest
  sets `OPENBLAS_CORETYPE`; a bare `./build/tests/foo` does not. Retry with
  `OPENBLAS_CORETYPE=SKYLAKEX` before blaming BatchLAS.
- **`-fopenmp` binds to the NVIDIA HPC SDK's libgomp** when it is on `LIBRARY_PATH`, then fails at
  load. `scripts/b` unsets `LIBRARY_PATH`; by hand use `env -u LIBRARY_PATH`.
- **`nvidia-smi`: "Driver/library version mismatch"**: unattended-upgrades replaced the driver under
  the running kernel; only a reboot or module reload fixes it.
- **A CUDA binary "hangs at startup" at 100% CPU+GPU**: usually a stuck kernel, not JIT. Run
  `SYCL_UR_TRACE=2 ./bin > trace.log 2>&1`, map `urKernelCreate` handles to names, read the last
  `urEnqueueKernelLaunch`. `compute-sanitizer` cannot attach; for out-of-bounds rebuild with
  `-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG"` (device asserts in `VectorView::at`).

**Metal (macOS, AdaptiveCpp).** Setup, presets and the toolchain script: @ref guide_build; design:
`docs/design/sycl-implementations.md#sycl-impl-metal`.

- **`ACPP_ADAPTIVITY_LEVEL=0` breaks every float kernel** (it JITs a TU's double kernels too; a GPU
  `Queue` throws `unsupported`). Leave it unset. A double call on the Apple GPU throws by design.
- **Clear `~/.acpp/apps` after reinstalling AdaptiveCpp** (the JIT cache outlives it), or give a
  run its own cache with `ACPP_APPDB_DIR=<dir>`.
- **The `CPATH` leak**: `make install` resets the acpp shebang to `/usr/bin/env python3`, whose
  Command Line Tools shim exports `CPATH=/usr/local/include` (a `/usr/local` GoogleTest then
  shadows the fetched one). Re-run the build script after any reinstall; use
  `env -u CPATH -u LIBRARY_PATH`. **`DYLD_LIBRARY_PATH`**, not `LD_`; SIP strips it from `/usr/bin`.
- **Debug**: `ACPP_DEBUG_LEVEL=3`, `__ACPP_PRINT_METAL_CODE=1` (MSL), `__ACPP_PRINT_IR_BEFORE_EMIT=1`;
  `BATCHLAS_METAL_FLUSH_EACH_OP=1` attributes a GPU fault to one op.
- **A data-dependent loop around barriers never ends on the OpenMP device** unless its condition
  is a group collective (`tridiagonal_solver` broadcasts it under acpp). A host test hanging in
  `__acpp_sscp_kernel` is this.
- **Float subnormals flush to zero** although the device reports denorm support: probe and skip.

## 7. Build Performance

Findings and before/after tables: @ref design_build_performance.

- **Device-link-bound.** The unit of device linking is the `.so` (`ls build/src/*.so`): one changed
  object re-runs `sycl-post-link`, ptxas and CPU AOT for every object in its library, and `-j`
  cannot shorten it. One-line edits rebuild in 12-22 s (small `.so`) to 200-275 s (cta, posv, gemm
  kernels); a fresh tree ~5 min with a warm ccache (4090 box, 2026-10-24). Critical path: the
  `libbatchlas_sycl` relink (194 s) and headers ~280 of 318 objects include (`matrix.hh` & co).
- **CPU AOT** was 942 of 1,186 CPU-seconds of the old `libbatchlas_sycl` link: kernels whose every
  native `can_run` arm needs `d.is_gpu` go in a `NO_CPU_TARGETS` library; a TU a CPU queue can run
  (gemm, gemv, spmm, NETLIB) keeps its CPU image, so gemm's reg/wide rows are shard TUs
  (`BATCHLAS_GEMM_SHARDS`; a `static_assert` catches an unsharded row).
- **Fast-loop options** default on (table in @ref guide_build): `BATCHLAS_SYCL_GLOBAL_OFFSET=OFF`
  (no `_with_offset` twins, ~half of every NVPTX link), `BATCHLAS_PRIVATE_TMPDIR=ON` (leaked device
  temporaries filled a tmpfs `/tmp`; `rm -rf <build>/.tmp` is safe while no build runs),
  `BATCHLAS_LINK_DEPENDS_NO_SHARED=ON` (no executable relinks; links are `-z now`, so a removed
  symbol fails at load), `CMAKE_BUILD_RPATH_USE_ORIGIN=ON` (a seeded tree loads its own `.so`).
- **New trees**: `scripts/seed_build_tree.sh SRC DST` (9 s) before editing, from a tree with the
  same preset (`--allow-dirty` for a dirty seed). `scripts/prewarm_cache.sh --print-cron` keeps
  ccache warm for origin/main. Reconfigure an existing tree rather than configuring a fresh one.
- **Cap at `-j8`**: heavy SYCL TUs exceed 14 GB. RAM is the only limit on concurrent builds:
  `scripts/b` admits a build while `MemAvailable` >= `BATCHLAS_ADMIT_MEM_GB` (36), staggers starts
  by `BATCHLAS_ADMIT_RAMP_S` (120 s), and below `BATCHLAS_PAUSE_MEM_GB` (10) SIGSTOPs all but the
  oldest until `BATCHLAS_RESUME_MEM_GB` (24). `BATCHLAS_BUILD_LOCK_MODE=exclusive` (the default
  where there is no `/proc/meminfo`, i.e. macOS) is one build per box; `scripts/gate.sh` always
  takes the exclusive lock.
- **ccache** (`BATCHLAS_USE_CCACHE`, on when `ccache` is on `PATH` or in `~/.local/bin`): the
  generated `<build>/batchlas-ccache` wrapper lets a second tree hit the first tree's cache
  (187/196 hits). Misses are TUs with an absolute path in a `-D`. Links are never cached.
- **GPU-only iteration**: `-DBATCHLAS_CPU_TARGET=none` (`dev-gpu*` presets; 248 s cold vs 284-334 s)
  lacks the NETLIB/CPU instantiations, so it is never the gate. Vendor-free builds only at the gate.
- `BATCHLAS_SYCL_LINK_JOBS` (`auto` = cores, 4..32): a link costs the sum of its batch maxima; a
  one-module `.so` gains nothing. Profile per module by replaying the driver's `-###` stages.
- `cmake --build --target A B C` can exit 0 and skip links: `scripts/b` checks (exit 70); by hand
  `test -x build/tests/<name>`.
- Time a rebuild by appending a line, not `touch` (a ccache hit times only the link).
  `-fsyntax-only` with `-fsycl-targets` is rejected: use real `-c` compiles.
- After touching `src/backends/roc*.cc` or a signature they spell out, run
  `scripts/rocm_syntax_check.sh` (needs `/opt/rocm/include/roc*/`; skips everything on the 4090 box).
- Tiny-kernel TUs build with `-mllvm -pragma-unroll-threshold=262144`; without it LLVM silently
  drops `#pragma unroll` and register arrays spill to the stack.
- Ruled-out build ideas (Ninja, lld, PCH, unity, `-fno-sycl-rdc`): the table at the end of §11.

## 8. Testing Policy

- **Do not run the full ctest by default.** Scope: one case `--gtest_filter` -> one binary
  `ctest -R '^X(_heavy)?$'` (anchor the regex) -> one component `-L util|blas|ortho|tridiag|eig|sparse`
  -> `-LE 'slow|heavy'` -> `-LE slow` -> full only before pushing or after touching shared code
  (`Queue`, `Matrix`/`MatrixView`, mempool, `sg_partition`, `include/batchlas/util`).
  `scripts/affected_tests.py` maps a diff to binaries (callers included; over-selects by design).
- **Case tiers.** A binary in `BATCHLAS_HEAVY_TESTS` is two entries: `X` (quick cases) and
  `X_heavy` (label `heavy`: saturating-batch race guards, table sweeps). Run `X_heavy` before
  claiming `X` green. `-LE slow` includes heavy entries; `-LE 'slow|heavy'` does not.
- **Full-target tree on the 4090 box**: `ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'` and
  `CUDA_VISIBLE_DEVICES=0`, else CPU backends land on `native_cpu` ("No kernel named").
- **`scripts/ctest_gpus.sh`** (ctest's arguments) runs one test per GPU slot from
  `<build>/ctest_resources.json` (`BATCHLAS_TEST_GPUS`, `BATCHLAS_TEST_GPU_SLOTS` default 2; only
  with a `[cuda:gpu]` at configure). threadripper02, 4x RTX PRO 6000, `-LE slow`: 469 s serial,
  126 s at 1 slot/GPU, 78 s at 2. Those GPUs are now hidden from agents (GPU load crashed the box)
  and the 2x4090 box was OOM-killed beside a build or a second ctest (`docs/ci.md`), so each gate
  ctest stays serial; the two trees overlap only after both builds, under the lock (§6).
- A process that sees several GPUs is slow (threadripper02: 132 s with 4 visible, 18 s with 1;
  cause unknown): run a binary by hand with `CUDA_VISIBLE_DEVICES=<n>`.
- `BATCHLAS_TEST_BACKEND=CUDA` / `BATCHLAS_TEST_FLOAT_TYPE=float` turn the other instantiations
  into GTEST_SKIPs: same case list, coverage silently gone.
- **main is not green.** Compare failing test **names**, never counts, against
  `tests/known-failures.txt` plus `tests/known-failures-host-<host>.txt` (`compare_failures.py`
  marks those `[host]`). Attribute the rest with `python3 scripts/agent/attribute_failures.py
  --tree build-gate <ctest names>` (PRE-EXISTING / NEW / FLAKY / FIXED per case), never by hand.
- Known flakes: `syev_cta_tests` (concurrency wrong answer, so `RUN_SERIAL`); `sytrd_blocked_tests`
  on untouched main. A CUDA-off tree fakes failures ("Not Run", CTA kernels throw without
  sub-group 32).

**Writing tests: guards that cannot fail.** A green suite has passed a wrong answer more than a
dozen times. Every new or rewritten test needs:

1. A deliberate break of the code under test, once per axis, giving a *narrow, named* red set.
2. The break runs where the code is reachable: pin the route (`BATCHLAS_<OP>_ROUTE=<spelling>`,
   `select::ScopedPin`) or use the vendor-free build, or a table may send the shape to the vendor.
3. Non-natural values for every accessor read (`ld`, batch `stride`, `inc`, rows!=cols), complex
   data with nonzero imaginary part, ConjTrans, beta!=0: derived values pass the natural case.
4. Poison the code will accept (an in-range index with a large finite value, not NaN), surviving
   upstream code (cuSOLVER Upper `potrf` overwrites the lower triangle).
5. For any shared-local-memory kernel, a saturating batch (>=1024, one matrix repeated, each result
   bit-identical to item 0). Small tests run at wg=32, which hides cross-sub-group SLM races.
6. Thresholds straddled in both directions.
7. No self-referential reference (one through the same tables or routes agrees by construction).
8. Graded input (condition ~1e±6) for eigen/tridiagonal accuracy; GOE matrices hide relative bugs.
9. For a capacity guard, a *launch* at the advertised ceiling, not an arithmetic re-derivation.
10. Verify a revert of an untracked file with `md5sum` against a pristine copy (`git diff` is silent).
11. Verify with `batchlas::verify` (`EXPECT_VERIFY(T, Check::kind, n, value)`,
    `tests/test_utils.hh`; @ref design_verification); no local residual or tolerance helpers.

**Writing fast tests.** Kernels under test were 4-7% of wall in the slowest gate cases (2026-10
audit); the harness took the rest. Keep every guard above and spend nothing beside it:

1. Observe in-process. A threadsafe death-test child replays the body up to its call and pays
   0.5-1.4 s of startup, so reading state through children is quadratic. If one is unavoidable:
   one per TEST, as its first statement.
2. Generate a deterministic input once per test (cache by type, n, batch, seed); copy and
   re-poison it per axis value. A graded fill at n >= 1024 costs 12-46 s.
3. No host loops over `malloc_shared` (`UnifiedVector`/`Matrix`) a kernel has touched: use
   `test_utils::usm_bulk` (`q.memcpy`) or a device-side check. Build input images once, memcpy in.
4. Allocate outside loops: one problem or workspace per test at the largest size, poisoned per
   run. No per-candidate rebuild of an identical problem; no `UnifiedVector` copy-assign in a hot
   loop (it builds a `sycl::queue`).
5. Verify in batch: one `EXPECT_VERIFY` over all items; per-item loops only to name failures.
6. Saturating batches (guard 5) generate a few representatives (`period` 1-7) and copy them; no
   RNG draw per item.
7. Drop an axis or a backend only after reading the predicate or kernel that ignores it (NETLIB
   ignores `cta_*`); four "independent" axes in the audit were not.
8. Budget: a quick case over ~2 s or a binary over ~15 s goes to `heavy`/`slow` with a reason in
   `tests/CMakeLists.txt`. Time from per-case gtest XML or the gate junit, never
   `CTestCostData.txt` (a serial ctest skews it). No assertion-free cases in ctest.

## 9. Architecture: Routing and Dispatch

Design: @ref design_flat_selection (per-op families in its "Per-op families" section); tables,
ledgers and staleness: @ref design_tuning_data.

- `src/ops/<op>/choice.hh` lists an op's choices (a `std::variant` of families with int knobs, e.g.
  `lpanel:panel=8`); `src/ops/<op>/<op>.cc` holds `key_of`, `can_run()`, launch and workspace;
  `select::run` / `select::pick` and the op's `select::OpSpec` do the rest.
- **`can_run()` is correctness only**: false means the launch would throw or give a wrong answer.
  A speed threshold there makes the shape unservable vendor-free.
- **Speed lives only in tables.** `select::choose` takes the first runnable entry of the nearest row
  of `tuned/<op>.<device>.txt` (exact keys by longest prefix; `n`/`batch` by weighted log
  distance). A device with no table borrows the nearest and warns once; the CPU never borrows;
  nothing runnable falls to the last resort. `BATCHLAS_TUNED_DIR=<dir>` overrides tables without
  a rebuild; `BATCHLAS_SELECT_TRACE=1` prints every choice and the runner-up.
- **Pins**: `BATCHLAS_<OP>_ROUTE` takes `auto`, `native`, `vendor` or a spelling (`lpanel:panel=8`,
  `lpanel:8`). **A bad pin throws** (unparsable, not compiled, cannot run the shape, old aliases
  such as `tiled16` or `cublasdx`); only `native`/`vendor` with nothing runnable fall back to Auto,
  with a warning. A raw `setenv` is unseen until `detail::reload_settings()` (`ScopedEnvVar`).
- Level-3 vendor families refuse a heterogeneous batch on every backend, so such a call has no
  route; rocBLAS has no hemm/herk/her2k.
- **Vendor-free**: only `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` is a valid vendor-free measurement.
  With nothing runnable an op throws `batchlas::NoRouteError` and records a coverage `miss` row.

**Verifying routes**

- **Linked != reached.** Set `BATCHLAS_COVERAGE_OUT` and read the `reached` rows
  (`chosen_algo` is the full spelling); `scripts/route_diff.sh` diffs them, `coverage_merge.sh`
  merges shards. Verify a routing change by diffing the chosen route, not by timing. A derived leg
  (gemm aligned vs predicated) needs `BATCHLAS_KERNEL_TRACE=1`, never while timing (+~60%).
- Sub-ops count: a "native" syev, geqrf or getrf can call cuBLAS gemm through the public `gemm`.

**Routing bug patterns**

- Highest-yield: **a finished candidate no table row ranks first.** Check every candidate is timed
  and that `can_run` ever admits it (hard `return false`, terms naming another family's types).
- **Read the predicate, not the comment.** `docs/perf/<op>.md` quotes each `can_run` term with file
  and line and bracketing evidence. Source comments cite `evidence: docs/<page>.md#anchor`;
  `check_evidence_anchors.py` fails on a renamed heading and requires cited slugs to be unique
  across `docs/`.
- A kernel's fast-*leg* predicate used as a routing gate sends calls to a slower kernel (1.7x and
  3-4x, twice). A leg is derived in the launcher, never a family field, key or `can_run` term.
- Guarding on a device family (`BATCHLAS_HAS_CUDA_BACKEND`) where the code needs a *library*: only
  a vendor-free build exposes it.

**API traps**

- The variadic queue-dispatch overload (`BATCHLAS_DISPATCH_ON_QUEUE`) beats option-struct overloads
  for non-const lvalues and prvalues, so checks only in `options.hh` are unreachable on the
  ordinary spelling. Test through the ordinary spelling.
- Write `PotrfOptions{}`, never a bare `{}` (it selects the positional `Uplo{}` = Upper).
- `BumpAllocator` checks the aligned size but advances by the raw size: size workspaces with
  `BumpAllocator::measuring()` + `required_bytes()`, branch `*_buffer_size` and run identically,
  and never test `ws.data() != nullptr` (sizing mode hands out unbacked pointers).
- The workspace arena belongs to the `Queue`: never a per-call `Queue`. Leak long-lived SYCL
  objects held in statics (destruction after the runtime hangs).
- Pivots are packed 1-based int32 everywhere; `cusolverDnXgetrs` wants int64.
- `linalg::qr` is absent on purpose; located-but-unfixed defects: `docs/design/known-defects.md`.

## 10. Measurement Rules (benchmarks lie silently here)

- **Compare and tune only at saturation, batch >= 128** (doubling the batch buys <10% per item);
  batch 1 does not matter. Still profile across batch: a work-group count equal to the batch is a
  bug.
- **Time cold operands**: the tuner and minibench reset, scrub L2 (`bench::scrub_l2`), wait, then
  time one call per sample (`protocol=lean-cold`). L2-warm `lean` data read up to 1.9x (gemm) or 3x
  (gemv) fast below the L2 edge
  (`docs/perf/batch-sensitivity.md#batch-sensitivity-the-cold-protocol`).
- **Do not tune or optimise FP64 on a 1:64 GPU** (sm_89, sm_120): keep it correct, skip it in
  retunes (`--dtype float,cfloat`), defer its performance to 1:2 hardware or the CPU backend.
- **One measuring process on the box at a time**, whatever the device pin (5.5x errors with stable
  rel_sd, nothing in `nvidia-smi`). Measure on a headless GPU; use `benchmarks/gpu_guard.sh` and a
  `flock` against your own duplicates.
- **Warm up**: idle clocks are ~210 MHz. `--warmup=3 --min_iters=8`, discard the first pass over
  every cell. CUDA images are AOT (no JIT, ~0.3 s module load); generic `spir64`, acpp `generic`
  and Metal images JIT on first use.
- Alternate A/B order between reps; per-rep CV ~3%, so an effect under 5% needs 14-16 reps.
- Google Benchmark `--name` is a substring match: filter on the CSV name column.
- Admit rows on cross-pass reproduction, not rel_sd alone; bracket every window edge with a
  measured non-winner.
- "vs vendor": check which route ran, what the vendor is (cuBLAS complex trsm and batched orgqr are
  per-item loops or our substitutes; `syevjBatched` beats `XsyevBatched` at n<=32); re-measure it.
- Rebuild every bench binary after a table or `can_run` change (harnesses resolve the printed route
  in their own TU). Check an argument is used: time scales with it.
- Sanity anchors (RTX 4090): FP32 SGEMM ~47-54 TFLOP/s (~80 means TF32), FP64 ~1.44 TFLOP/s, DRAM
  ~950 GB/s achievable; cuBLAS gemv is at the DRAM roof. A cost non-monotonic in n is a routing bug.
- nsys sees cuBLAS kernels, `BATCHLAS_KERNEL_TRACE` does not. Also: `BATCHLAS_KERNEL_TRACE_PATH`,
  `BATCHLAS_GESVD_PROFILE=1`, `BATCHLAS_TUNE_*` (A/B a constant without a rebuild), `BATCHLAS_BENCH_*`.
- Confirm a GEMM at beta=1 with a strided `ld` (panel updates carry the parent's `ld`). Pin one
  kernel with `BATCHLAS_GEMM_ROUTE=<spelling>` (`reg:m=128:n=128:k=8:u=1`, `wide:m=64:n=64:k=16`,
  `splitk:m=16:n=16:k=16:s=4`, `small`, `tiled`, `direct`); compare kernels with
  `tools/tune/batchlas_tune` (`gemm_spec.cc`).
- benchviz (`python3 benchmarks/benchviz`) measures *its checkout's* `build/`: run `benchviz info`
  first. Its batchlas arm pins the native walk. Figures follow `benchmarks/benchviz/style.py`.
- Raw grids go in `benchmarks/results/` (Git LFS: `git lfs install` on every machine, or CI's
  `check_lfs_pointers.py` fails); older evidence at tag `perf-evidence/vendor-independence`.
- `include/batchlas/tuning_params.hh` is checked in, tuned on float only (`evaluation/tuning/`):
  sweep another type before blaming it; check a knob for a second consumer or a bypass.

## 11. GPU Kernel Design Facts (sm_89; mostly general)

- **Register gate**: `ceil(warps/4)*32*regs <= 16384` (four sub-partitions), not
  `regs*wg <= 65536`. SYCL accepts the launch and the driver aborts. Standalone `-Xcuda-ptxas -v`
  counts differ from the in-library kernel: prefer a measured launch ceiling.
- Design against occupancy: `complex<double>` x 512 threads exceeds the register file; shrink the
  thread tile as the scalar widens. Return accumulators by value, never through a `T&`
  out-parameter (spills the array, 43%).
- `std::complex` `*` emits an isnan branch plus `__mulsc3`. Write out the real arithmetic in the
  hot loop only (1.2-1.3x); everywhere costs occupancy. No `-ffast-math`.
- The 48 KB SLM launch hole is sticky per CUfunction: put the discriminating launch first. SLM
  staged by a lane-strided loop and read by another index needs a barrier (hidden at wg=32).
- Launch bounds (`intel::min_work_groups_per_cu`) apply only on a **functor's** `operator()`
  (1.1-1.9x on tiny kernels; too tight is 2.3x slower, and it moves routing windows).
- Coalescing: one work-item per *output* element (NoTrans gemv coalesced, Trans reduces). laswp is
  an index gather in SLM (7.9x fewer sectors).
- GEMM: 128x128x8 tiles, 8x8 thread tile, aligned shared strides, B staged [k][n] give cuBLAS
  parity for float NN; the epilogue varies m fastest. Internal demand is panel updates (large m,n;
  k 8-136; ~60% transposed), so a `min_dim` gate never fires.
- `chunked_partition` masks cost ~4-5 instructions per collective; judge `sg_partition` changes by
  real-kernel A/B (steqr/syev_cta, n=5..16, batch 16384), never microbenchmarks.
- CTA kernels are capped at n <= 32 (the sub-group width). `STEDC_RECURSION_THRESHOLD = 32` is
  that invariant, not a tuning constant (a 33+ leaf falls to `steqr_wg`, 14x cliff).
- Byte-counting traffic models miss serial recurrences: check the critical path first.
- Open: the latrd panel symv is L1-over-fetch-bound (12-16x); a single-read symv is the open 2.7x.

**Ruled out: do not retry without a new idea**

| Alternative | Result | Verdict |
| --- | --- | --- |
| Ninja | Makefile null build is already 0.07 s | no gain |
| lld, PCH, unity builds | driver cannot emit a PCH; unity builds collide kernel names | dead |
| `-fno-sycl-rdc`, explicit `-fsycl-device-code-split` | measured, no gain (@ref design_build_performance) | dead |
| Cutting one edge of a header cycle | the umbrella header cuts both edges | dead |
| Plain `CMAKE_CXX_COMPILER_LAUNCHER=ccache` | 3/196 cross-tree hits (wrapper: 187/196) | use the wrapper |
| CTA eigensolver above n=32 | 85-211x slower | dead |
| Block Jacobi as a speed path | 2-11x slower | accuracy-only opt-in |
| Cooperative TRSM diagonal solve; faster native gemv; maskless steqr at P<32 | no gain | dead |
| Packed sub-group getrf with runtime n in SLM | no gain | dead (register-resident not refuted) |
| Givens chase | Householder chase 5.6x cheaper | dead |
| GEMM kernel fix for strided `ld`; double-buffering or packing B | no gain; the fix was routing | dead |
| Grid-stride / persistent loops | equal under-filled, 15% slower saturated | only for the 65535 cap |
| Per-item k-split lanes, potrf `lpanel` small batch | 1.14-1.18x, loses from B>=752 | dead (`docs/perf/batch-sensitivity.md#batch-sensitivity-measured-dead-ends`) |

## 12. Repository Conventions and Git

- **Docs are a Doxygen site** (`sh scripts/build_docs.sh [out]`; CI treats warnings as errors).
  `docs/developer/documentation.md` says where things go. Cited heading slugs are an interface.
- **No Markdown outside `docs/`** except directory `README.md`s (not at the root), `.claude/` and
  `.github/` (`check_markdown_locations.py`). No root plan files.
- **Source layout** (`docs/extending.md`): `src/ops/<op>/` holds `<op>.cc`, `choice.hh`, shared
  headers and `kernels/`; sub-ops nest under the parent; shared device code in
  `src/kernels/common/`. `src/ops/CMakeLists.txt` groups by device-link cluster, not folder.
- **Includes are root-based** (`#include <src/ops/potrf/potrf_native.hh>`), never quoted.
- **Throw `batchlas::` types** (`include/batchlas/error.hh`), never `std::` (bar `std::bad_alloc`);
  messages open with the lowercase op name (`gesv: ...`); warnings via `batchlas::detail::warn`.
- **Comment density <=18% per file** (`.github/ci/check_comment_density.py`; waivers in
  `comment_density_waivers.txt`). A line with a trailing comment counts as code. Comments are for
  invariants, deliberate oddities, traps and `evidence:` pointers. `///` contract docs in
  `include/batchlas/` are exempt; rationale in `///` is not.
- Stacked PRs: `merged` can mean into the previous stack branch; read `base.ref`.
  `/code-review ultra` rejects diffs over 500 files or 8,000 lines: split by work-package commits.
- `nm -C` truncates concept-constrained names: diff mangled names (`scripts/facade_symbol_check.sh`).
- **Worktree Bash guard** refuses: heredocs mentioning git, `for` loops with computed arguments,
  awk, `<(...)`, `cd X; git` and `git -C <other checkout>`, git mixed with other tools, `grep hash`.
  Run one plain command per call, use `scripts/agent/` (README there) and the Edit tool; a script
  in `$CLAUDE_JOB_DIR/tmp` only as a last resort. Wait with `scripts/agent/wait_for.sh` in the
  background, never a sleep loop or `pgrep -f`.

## 13. Primary-Machine Facts (the RTX 4090 box; re-check on any other machine)

Hosts, cores, memory, GPUs and per-host rules: `scripts/agent/dispatch.sh` (`hosts`, `rules_of`).

- 4090 box: no ROCm headers; MAGMA 2.10 Release/sm_89 at `~/magma/build-rel89`; system OpenBLAS
  0.3.32 (passes the dgemm check). Device 0 drives the display (harmless under light graphics
  load); device 1 is often the other user's. Check `nvidia-smi` before timing; coordinate reboots.
- `perf` and `gdb -p` do not work there (use the UR trace); unattended-upgrades is not blacklisted
  for `nvidia-*`, so driver mismatches recur (Known Pitfalls).
- threadripper02 is CPU/build only; its GPUs are hidden from agents.

**New machine**, before trusting a result: `sycl-ls` shows the GPU; configure prints the right
target; `nvidia-smi` reports the expected architecture; record its FP32/FP64/DRAM ceilings as
anchors. Every routing window and constant was measured on sm_89 (some tables on sm_120); on any
other GPU (Apple, or 1:2 FP64 where double GEMM verdicts invert) treat the tables as hypotheses,
re-measure at saturation, and record the machine with new data in `benchmarks/results/`.

## 14. Agent Workflow Rules

Tools: `scripts/agent/README.md`. Skills: `/sync-main`, `/fanout`, `/tune`, `/prepush`.

- **Start from current main**: `git fetch origin`, branch from `origin/main`; read newer files with
  `git show origin/main:<path>`.
- **Check the machine before GPU work**: `scripts/agent/gpu_health.sh`. On FAIL tell the user at
  once; do not carry on "compile-verified".
- **Never wait blind**: `run_in_background`; `scripts/agent/wait_for.sh --log F --stall-min M`;
  check in with the user every ~2-3 h of autonomous work.
- **Keep the PR the size of the ask.** Ask first before: new helper headers or classes (or custom
  benchmark code where a class exists), porting main's code into the PR's style, loosening a
  tolerance, ledger edits, committing campaign data (`benchmarks/results/` beyond tuning
  ledgers). `scripts/agent/pr_scope.py` lists these; paste its REVIEW lines before every push.
- **Generated files are regenerated**: `tuned/*.txt` with `scripts/sweep_to_table.py`, TuneHash pins
  with `scripts/agent/repin_tune_hashes.py`.
- **CI runs nothing you could not have run**: `scripts/agent/prepush.sh` (strict docs build, tuned
  tables, `run_local_checks.sh`); after pushing, `scripts/agent/ci_failures.sh <pr>`.
- **Ask before building alternatives**: sketch each design and ask; embed tooling in benchviz.
  Subagents (`/fanout`): pre-flight the base and own the shared files.
- **Digest, don't dump**: `conflicts.py`, `wf_dump.py`, `doc_section.py`; long evidence to files.

## License

SPDX-License-Identifier: MIT
