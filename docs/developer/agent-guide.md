# Agent environment guide {#dev_agent_guide}

> **Status:** current · RTX 4090 box and threadripper02 unless a line says otherwise · moved here from the root `AGENTS.md` on 2026-10-07; `.claude/CLAUDE.md` imports it.

Terse working rules for agents and contributors: toolchain, build performance, testing policy,
routing architecture, measurement rules, GPU kernel facts and repository conventions.

**TL;DR.** Build dependencies: CMake >=3.17 (>=3.21 for `cmake --preset`), a C++20 compiler with
SYCL 2020 support **and a backend for your GPU vendor**; optionally netlib LAPACKE + CBLAS (the
NETLIB host backend and the LAPACKE test oracles). Runtime needs the same plus a GPU the SYCL
runtime exposes. The exercised configuration is
a self-built CUDA-enabled DPC++ (`/opt/dpcpp-cuda`) against CUDA 13.2 and an RTX 4090 (sm_89) on
Ubuntu 26.04; see "Tested platforms" in `docs/guide/build.md`.

**Apple GPU (macOS):** a separate AdaptiveCpp Metal build (presets `acpp-metal` /
`acpp-metal-tests`, toolchain from `scripts/macos/build_adaptivecpp_metal.sh`, M5 Max, macOS 27).
GPU = `Backend::SYCL`, float and complex<float> only; double runs on a CPU `Queue` via Homebrew
OpenBLAS; vendor-free. Recipe: "macOS: Apple GPU through AdaptiveCpp Metal" in
`docs/guide/build.md`; design: `docs/design/sycl-implementations.md#sycl-impl-metal`; traps below.

> **Warning:** NVIDIA targets need a CUDA-capable DPC++. The stock
> `intel-oneapi-compiler-dpcpp-cpp` package has **no CUDA adapter**: on an NVIDIA machine it
> configures cleanly and builds a CPU-only library. Use Intel oneAPI plus the Codeplay **oneAPI for
> NVIDIA GPUs** plugin, or a self-built `intel/llvm` configured with `--cuda`.
>
> - `sycl-ls` must list a `[cuda:gpu]` entry.
> - `-- Using SYCL targets: spir64_x86_64` in the configure output means a CPU-only build, whatever
>   GPUs are present. A CUDA configure names the architecture, e.g. `nvidia_gpu_sm_89`.
> - `-DBATCHLAS_ENABLE_CUDA=ON` means "require it": without a `[cuda:gpu]` the configure aborts with
>   a `FATAL_ERROR` naming the missing entry (see "CMake options" in `docs/guide/build.md`).
>   The default `AUTO` silently builds CPU-only.

## 1. Prerequisite Packages

| Component | Debian/Ubuntu (apt) | Fedora/RHEL-like (dnf) | Arch Linux (pacman) | Source build (fallback) |
| --- | --- | --- | --- | --- |
| BLAS/LAPACK (Fortran APIs) | libblas-dev liblapack-dev | blas-devel lapack-devel | blas lapack | see §4 |
| C interface (CBLAS & LAPACKE headers) | liblapacke-dev | lapack-devel | lapacke | see §4 |
| Build tools | build-essential cmake git | `@development-tools` cmake git | base-devel cmake git | — |
| Git LFS (raw benchmark results; docs/perf/README.md) | git-lfs | git-lfs | git-lfs | — |

`libopenblas-dev` alone is not enough: Ubuntu's OpenBLAS package omits `lapacke.h`, so install
`liblapacke-dev` too (a packaging decision, not a BatchLAS bug).

## 2. Installing a SYCL Compiler

### 2a. Intel oneAPI DPC++/C++ (icpx): Intel GPUs and CPU-only builds

The easy path, and the WRONG path for NVIDIA (no CUDA adapter; see the TL;DR and §2b).

```bash
# 1. Add Intel's APT repo and key (root)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
  sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

# 2. Install the minimal SYCL compiler package
sudo apt install intel-oneapi-compiler-dpcpp-cpp      # SYCL 2025.x

# (optional) Classic compilers for C/C++ & Fortran
sudo apt install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic \
                 intel-oneapi-compiler-fortran
```

Per shell session:

```bash
source /opt/intel/oneapi/setvars.sh   # sets PATH, LD_LIBRARY_PATH, MKLROOT, etc.
```

The single compiler package is enough for CPU or Intel GPU; the full intel-basekit is not needed.

### 2b. CUDA-capable DPC++: NVIDIA GPUs

Both options give a clang-family `clang++`/`icpx` that emits NVPTX:

- **Codeplay oneAPI for NVIDIA GPUs**: install oneAPI as in §2a, then the matching Codeplay plugin.
  It adds the CUDA UR adapter, after which `sycl-ls` reports `[cuda:gpu]`.
- **Self-built `intel/llvm`**: clone https://github.com/intel/llvm, configure with `--cuda` (plus
  `--cmake-opt=-DCMAKE_INSTALL_PREFIX=<prefix>`). This machine uses it at `/opt/dpcpp-cuda`; every
  number in the repository was measured with it.

Also install a CUDA Toolkit (13.2 here) and pass the compiler explicitly:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++
```

A consumer of an installed BatchLAS must configure the *whole* consuming project with this same
compiler ("Building against BatchLAS" in `docs/cpp-api.md`).

### 2c. AdaptiveCpp: the second SYCL implementation

BatchLAS also builds with the AdaptiveCpp 25.10 release (`BATCHLAS_SYCL_IMPL=ACPP`; design
@ref design_sycl_implementations). Presets `acpp` and `acpp-tests` set
`CMAKE_CXX_COMPILER=/opt/adaptivecpp-25.10.0-cuda13.2/bin/acpp` (trees `build-acpp`,
`build-acpp-tests`). The hosted CI job `acpp-compile` compiles every target with
`--acpp-targets=generic` and CUDA off; it runs nothing. An acpp test tree is gated against its own
ledger, `tests/known-failures-acpp.txt` (`compare_failures.py --sycl-impl ACPP`, or
`run_local_checks.sh` on an acpp tree).

## 3. Verifying the Toolchain

```bash
icpx --version          # or: /opt/dpcpp-cuda/bin/clang++ --version
cmake --version         # ≥3.17 (≥3.21 for cmake --preset)
sycl-ls                 # must list your GPU, e.g. a [cuda:gpu] entry
pkg-config --exists lapacke cblas && echo "LAPACKE & CBLAS found"
```

## 4. Building netlib LAPACKE/CBLAS from Source (if distro packages are unavailable)

```bash
git clone https://github.com/Reference-LAPACK/lapack.git
cd lapack
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLAPACKE=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j $(nproc)
sudo cmake --install build   # installs liblapacke.so, libcblas.so, headers
```

Add the install prefix (e.g. `/usr/local/lib`) to `LD_LIBRARY_PATH`, or run `sudo ldconfig`.

## 5. CMake Configuration Hints

There is no `find_package` for SYCL: the compiler is the SYCL implementation. LAPACKE and CBLAS
are found with `find_library` (`find_netlib_libs` in `cmake/BatchLASDependencies.cmake`): `lapacke`
in the Debian multiarch directories first, then the default search path; CBLAS (`cblas`, else
`blas`) from LAPACKE's own directory first. Both are optional: without them configure warns and
disables the NETLIB host backend. A library outside those paths is named explicitly:

```bash
cmake -S . -B build -DLAPACKE_LIBRARY=/opt/lib/liblapacke.so -DCBLAS_LIBRARY=/opt/lib/libcblas.so
```

## 6. Quick Smoke Test

```bash
# A new worktree: seed its tree from a built one BEFORE editing (§7), else configure.
scripts/seed_build_tree.sh ~/BatchLAS/build build    # 9 s; first build is incremental
cmake -B build .                                     # no seed: full build 5 min with a warm ccache
# GPU-only iteration tree (nvptx64 only; never the gate, §7):
#   cmake -B build-gpu . -DBATCHLAS_CPU_TARGET=none

# Iterate through scripts/b: named targets only (no target = exit 64), -j8 cap, a
# box-wide build lock, a free-memory wait, exit 70 when a requested binary is
# missing after a "successful" build. One target = the library plus one test binary.
scripts/b stedc_tests
ctest --test-dir build -R '^stedc_tests(_heavy)?$' --output-on-failure

# Which binaries can my diff break? (vs merge-base with origin/main; shared code
# falls back to the quick tier; --heavy adds the _heavy entries)
scripts/affected_tests.py --print-build-cmd --print-ctest-cmd --build-dir build

# Broadest iteration scope: the quick tier (no slow suites, no heavy case tiers).
ctest --test-dir build -LE 'slow|heavy'

# Before pushing, build and run everything with scripts/gate.sh, never on an
# iteration tree such as build/. Its two persistent trees: build-gate (full targets,
# full ctest, ledger diff) and build-gate-vf (-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF,
# -LE slow), both with BATCHLAS_LINK_DEPENDS_NO_SHARED=OFF, so every executable is
# relinked against the .so it tests (that option is ON, and executables are not
# relinked, in every other tree). It holds the build lock through ctest. One test per
# GPU slot (--parallel, §8) only where ctest_gpus.sh has been validated
# (threadripper02); on the 2x4090 box, and wherever configure printed "GPU test
# resources: none", keep it serial (the default). First run on the 4090 box: 67 min
# (full: build 14 + ctest 21; vendor-free: build 13 + ctest 19); later runs rebuild
# incrementally. Seed a missing gate tree only from a dev-tests-configured tree:
# from plain build/ (BATCHLAS_ALLOW_UNSAFE_ENV differs, so backend_config.h does)
# 221 of 243 objects recompile. The gate prints the option drift. Its ctest gets
# ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu' when unset and sycl-ls lists native_cpu (§8);
# pin the GPU yourself. --seed-allow-dirty: the checkout has any untracked or modified
# file (seed_build_tree.sh refuses a dirty seed otherwise, and the full tree fails).
CUDA_VISIBLE_DEVICES=0 scripts/gate.sh     # 2x4090 box (OOM history, docs/ci.md; GPU 1 is shared)
scripts/gate.sh --parallel                 # threadripper02
CUDA_VISIBLE_DEVICES=0 scripts/gate.sh --seed-from build-dev-tests --seed-allow-dirty   # first run

# Also before pushing, if you touched cmake/, include/ or the install rules:
# the packaging gate. These are the same checks CI runs (.github/workflows/ci.yml).
sh .github/ci/run_local_checks.sh

# The --package mode is the one CI cannot run, because it needs a real install
# from a real SYCL build. Give run_local_checks.sh a prefix and it adds it:
cmake --install build --prefix /tmp/batchlas-prefix
sh .github/ci/run_local_checks.sh /tmp/batchlas-prefix

# The end-to-end version of the same thing — install, then configure, build and
# run examples/consumer/ as a standalone outside project:
ctest --test-dir build -R '^consumer_package_tests$' --output-on-failure
```

## Known Pitfalls

- **Silent CPU-only build on an NVIDIA box**: the compiler has no CUDA adapter;
  `-- Using SYCL targets: spir64_x86_64` is the tell (TL;DR, §2b).
- **icpx (oneAPI 2026) defaults to `-fp-model=fast`**, which stops LLVM inlining libspirv builtins:
  `sycl::fma`, barriers and id queries become out-of-line CALLs in ~98% of kernels (cfloat gemm
  6-15x, potrf cfloat 6x slower) with every test green. The build passes `-ffp-model=precise` for
  IntelLLVM; `ctest -R '^device_calls_tests$'` (`scripts/check_device_calls.py`) scans the built PTX
  and fails on any `__spirv_*` call. Never add a later `-ffp-model=fast`/`-ffast-math` (the last flag
  wins). benchviz warns about such a build.
- **oneAPI 2026 icpx on an NVIDIA box without the Codeplay plugin**: oneAPI's libsycl ships no CUDA
  UR adapter. Put `/opt/dpcpp-cuda/lib` FIRST on `LD_LIBRARY_PATH`, ahead of `setvars.sh`, for
  configure, build and every run. Otherwise `sycl-ls` shows no `[cuda:gpu]` and configure silently
  picks `spir64_x86_64`.
- **Missing `lapacke.h`**: install `liblapacke-dev` even with `libopenblas-dev`.
- **Multiple BLAS providers**: `sudo update-alternatives --config libblas.so.3`.
- **Device not found at runtime**: the GPU driver and its SYCL adapter must match the compiler
  version (Level-Zero runtime for Intel, CUDA adapter and driver for NVIDIA).
- **`libsycl.so.9: cannot open shared object file`**: the DPC++ runtime is not on the loader path.
  Export `LD_LIBRARY_PATH=<dpcpp-prefix>/lib`, or add it to `/etc/ld.so.conf.d` and run `ldconfig`.
- **Wrong double results from the host (NETLIB) backend**: some OpenBLAS builds pick a broken dgemm
  kernel (0.3.20 on Cooperlake Xeons; errors ~66 at n=128). Configure runs a dgemm health check
  (`cmake/BatchLASBlasHealthCheck.cmake`) and ctest sets `OPENBLAS_CORETYPE`; **a bare
  `./build/tests/foo` does not**. For a double-only CPU failure, re-run with
  `OPENBLAS_CORETYPE=SKYLAKEX` before blaming BatchLAS.
- **`-fopenmp` binds to the NVIDIA HPC SDK's libgomp** when it is on `LIBRARY_PATH` (symbol version
  literally `VERSION`), then fails at load time. Link with `env -u LIBRARY_PATH`;
  `readelf -d <lib> | grep gomp` should show `libgomp.so.1`.
- **`nvidia-smi` says "Driver/library version mismatch"**: unattended-upgrades replaced the driver
  under the running kernel. Compare `cat /proc/driver/nvidia/version` with
  `modinfo nvidia | grep ^version`; only a reboot (or module reload) fixes it. Not a BatchLAS problem.
- **A CUDA binary that "hangs at startup" at 100% CPU+GPU** is usually a stuck kernel, not JIT. Run
  `SYCL_UR_TRACE=2 ./bin > trace.log 2>&1` (no pipe), map `urKernelCreate` handles to names, read the
  last `urEnqueueKernelLaunch`. `compute-sanitizer` cannot attach (SYCL calls `cuInit` first); for GPU
  out-of-bounds, rebuild with `-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG"` to re-enable the
  device asserts in `VectorView::at` / `KernelMatrixView`.

**Metal (macOS, AdaptiveCpp; M5 Max).** Design and limits: `docs/design/sycl-implementations.md#sycl-impl-metal`.

- **`ACPP_ADAPTIVITY_LEVEL=0` breaks every float kernel**: level 0 JITs a TU's kernels together,
  double ones included, and Metal cannot compile double. A GPU `Queue` throws `unsupported` at
  level 0 (`backend_dispatch_tests_adaptivity0`). Leave it unset (1) or higher.
- **A double call on the Apple GPU throws `unsupported`** by design (no fp64). Use
  `Queue(Device("cpu"))`. `set_backend(NETLIB)` on the GPU throws too.
- **Monolithic only on macOS**: split component dylibs have cyclic undefined symbols and ld64
  rejects them; configure fails without `-DBATCHLAS_MONOLITHIC_LIBRARY=ON`. Test trees also need
  `BATCHLAS_HIDDEN_VISIBILITY=OFF` (the presets set both).
- **Homebrew OpenBLAS has no `liblapacke`**: LAPACKE and CBLAS are inside `libopenblas.dylib`.
  `BATCHLAS_OPENBLAS_ROOT` (probed at `/opt/homebrew/opt/openblas`) finds it; frameworks are off for
  that search so vecLib's `cblas.h` never pairs with it.
- **Keg-only paths**: `llvm@21`, `lld@21`, `libomp` and `openblas` are not on default search paths.
  Use the `/opt/homebrew/opt/<pkg>` paths, never `Cellar/<version>`.
- **`brew upgrade` can break acpp**: `~/opt/adaptivecpp/etc/AdaptiveCpp/acpp-core.json` holds the
  LLVM and libomp paths; a `Cellar/` path there dangles after an upgrade (the build script warns).
  Rebuild with the script, or `brew pin llvm@21 libomp`.
- **Clear `~/.acpp/apps` after reinstalling AdaptiveCpp**: the JIT cache outlives the toolchain, so a
  patched translator otherwise runs only on kernels it has not cached. `ACPP_APPDB_DIR=<dir>` gives a
  run its own cache.
- **The patches are local** (`scripts/macos/adaptivecpp-patches/`, branch `batchlas-metal`). The
  script reuses an existing branch and stops on a patch-count mismatch: `ACPP_REAPPLY=1` re-applies.
- **`make install` resets the acpp shebang** to `#!/usr/bin/env python3`. When `/usr/bin` precedes
  Homebrew on `PATH` that is the Command Line Tools shim, which exports `CPATH=/usr/local/include`
  and `LIBRARY_PATH=/usr/local/lib` into every compile: a `/usr/local` GoogleTest then shadows the
  fetched one and tests fail to link. The script pins `acpp`/`syclcc`/`syclcc-clang` to Homebrew's
  `python3`; re-run it (or its `sed`) after any reinstall. `env -u CPATH -u LIBRARY_PATH` is the
  belt-and-braces for configure, build and ctest.
- **`DYLD_LIBRARY_PATH`, not `LD_LIBRARY_PATH`**, selects dylibs (e.g. a test runtime prefix:
  `DYLD_LIBRARY_PATH=<acpp-prefix>/lib`). SIP strips `DYLD_*` from `/usr/bin` interpreters and shell
  scripts run through them.
- **`ACPP_VISIBILITY_MASK=omp;metal`** (set by the presets): double and `batchlas::verify` need the
  OpenMP device, and configure fails without it.
- **Debug**: `ACPP_DEBUG_LEVEL=3` logs every Metal dispatch ("Dispatching kernel");
  `__ACPP_PRINT_METAL_CODE=1` prints the generated MSL (with IR comments);
  `__ACPP_PRINT_IR_BEFORE_EMIT=1` prints the LLVM IR the emitter receives (look for `<4 x`,
  `double`, `llvm.nvvm`). A GPU fault under coarse events covers many ops:
  `BATCHLAS_METAL_FLUSH_EACH_OP=1` commits one command buffer per op to attribute it.
- **A data-dependent loop around barriers never ends on the OpenMP device** (the host NETLIB
  instantiations run their native kernels there), even at batch 1, unless the loop condition is a
  group collective: `tridiagonal_solver` broadcasts it (`sycl::group_broadcast`) for NETLIB under
  acpp only. A host test that hangs at 100% CPU in `__acpp_sscp_kernel` is this.
- **Float subnormals flush to zero** on the Apple GPU, though the device reports denorm support.
  A test that needs one must probe the device and skip.
- **Routing on Apple is borrowed**: no `tuned/*.apple_m5_max.txt`, so every op borrows the nearest
  table (sm_120 here) and warns once. Every window and constant is an sm_89/sm_120 hypothesis there.
- **Run ctest serially** on the Mac (`ctest --preset acpp-metal-tests` sets `-j1`); its failure
  ledger is `tests/known-failures-acpp-metal.txt`.

## 7. Build Performance

The build is **SYCL device-link-bound**, not compile-bound. The unit of device linking is the
shared library: changing one object re-runs `sycl-post-link` + ptxas + CPU-target AOT for every
object in that `.so`, and `-j` cannot shorten it. Full build of a fresh tree: 4090 box (-j8,
library + 90 test binaries) 312-314 s with a warm ccache; 1260-1370 s for c960d7b1/cc9cee8c with a
partial ccache (118/77 hits), before the sycl and backends links were cut (no cold build of the
current tree measured); threadripper02 (-j32) 284-334 s cold. Findings and before/after tables:
@ref design_build_performance.

**The loop on the 4090 box** (2026-10-24; one-line append, then rebuild the library):

| Edit | Rebuild | Edit | Rebuild |
| --- | --- | --- | --- |
| `stedc.cc` (a small .so) | 22 s | a cta TU (`steqr_cta.cc`) | 112 s (was 618) |
| `gemm.cc` (backends) | 12 s | `posv_tiny.cc` (one-module .so) | 201 s (was 637) |
| `trmm.cc`, `trsm_native.cc` | 66 s (was 520), 62 s | `gemm_kernels.cc` | 213 s (was 1024) |
| `tuning_params.hh` | 11 compiles (was 128) | a header every gemm shard includes | 275 s |

**Remaining critical path:** the `libbatchlas_sycl` relink (194 s, largest module 59 s, so more
link jobs help only on bigger boxes), the posv/gesv single-module links (~169 s), headers that ~280
of 318 objects include (`matrix.hh`, `settings`, `error`, `enums`, `workspace`), and in tests
`gemm_candidates_tests_heavy` (351 s, the longest single entry).

**Component libraries.** 16 `.so` (`ls build/src/*.so`); one changed object relinks only its own.
GPU-only (`NO_CPU_TARGETS`, no CPU AOT image): `extensions_cta`, `extensions_posv`,
`extensions_gesv` (one tiny-kernel TU each, the cta link's two big modules), `extensions_level3`
(trsm kernels plus symm, hemm, herk, her2k, syrk, syr2k, trmm: every native arm needs `d.is_gpu`).
A TU that a CPU queue can run (gemm, gemv, spmm, NETLIB paths) must keep its CPU image.

**Fast-loop options** (all default to the fast setting; `docs/guide/build.md` "CMake options"):

- `BATCHLAS_SYCL_GLOBAL_OFFSET=OFF` passes `-mllvm -enable-global-offset=false`: no `_with_offset`
  twin per kernel (half of all kernel entries), about half of every NVPTX link. A kernel using a
  SYCL 1.2.1 global offset would lose it; nothing in the tree does.
- `BATCHLAS_PRIVATE_TMPDIR=ON`: every compile and link runs with `TMPDIR` under `<build>/.tmp`.
  DPC++ device temporaries in a tmpfs `/tmp` are RAM; leaked ones reached 38-43 GiB (~12 GiB/day)
  on a zero-swap box. `rm -rf <build>/.tmp` is safe while no build of that tree runs.
- `BATCHLAS_LINK_DEPENDS_NO_SHARED=ON`: a `.so` change no longer relinks ~150 executables (full
  build after a `stedc.cc` edit 71 -> 22 s). Executables and the component `.so` link `-z now`, so a
  symbol removed from a `.so` fails when any binary loads it, not when the call runs.
  `scripts/gate.sh` turns it OFF.
- `CMAKE_BUILD_RPATH_USE_ORIGIN=ON`: build-tree RUNPATHs are `$ORIGIN`-relative, so a copied
  (seeded) tree loads its own `.so`, never the seed's.

**New trees.**

- `scripts/seed_build_tree.sh SRC_BUILD DST_BUILD` copies a built tree, rewrites its paths and
  touches only files that differ from the seed's commit: 9 s, then the first one-line-edit
  `scripts/b stedc_tests` is 20 s. Seed before editing, from a tree configured like the target
  (same preset); a dirty seed checkout needs `--allow-dirty`. Targets with an absolute path in a
  `-D` recompile.
- `scripts/prewarm_cache.sh` keeps ccache warm for origin/main: builds it in
  `~/.cache/batchlas-prewarm/main` at -j4, takes the build lock and skips (exit 0) when another
  build holds it, memory is short or main is unchanged; that `build/` is also a clean seed. Cron it
  from the main checkout (`--print-cron` prints the line):
  `*/30 * * * * ~/BatchLAS/scripts/prewarm_cache.sh >> /tmp/batchlas-prewarm.log 2>&1`.
- **A/B tuning constants with the `BATCHLAS_TUNE_*` env overrides, never by editing
  `include/batchlas/tuning_params.hh` per trial** (§10). Edit the header once, to port the winner.

**Build rules.**

- **Cap build parallelism at `-j8`** unless you have measured headroom. Heavy SYCL TUs exceed 14 GB
  each, and wide builds (`-j20`, `-j28`, or two `-j` builds side by side) have been OOM-killed
  several times. Run one build at a time; add a second only after checking `free -g`.
- **ccache is on by default** (`BATCHLAS_USE_CCACHE`, `cmake/BatchLASCcache.cmake`) when `ccache`
  is on `PATH` or in `~/.local/bin`; configure prints `ccache enabled (...)`. No ccache: drop the
  static binary from https://github.com/ccache/ccache/releases into `~/.local/bin` (no root).
  - The generated `<build>/batchlas-ccache` wrapper sets depend mode, `CCACHE_BASEDIR` (deepest
    common parent of source and build dir, `BATCHLAS_CCACHE_BASEDIR`), `CCACHE_NOHASHDIR` and
    sloppiness, so a **second tree hits the first tree's cache**: 187/196 hits, 203 s vs 284 s cold
    (another pair: 196/196, 176 s vs 329 s). Misses are TUs with an absolute path in a `-D`.
  - Plain `CMAKE_CXX_COMPILER_LAUNCHER=ccache` without those settings got 3/196.
  - Links are never cached: even an all-hits fresh tree pays every device and test link.
  - Preset trees are `build-<preset>`, depth 1 like `build/`, so they share entries with it; the
    old depth-3 `build/presets/<preset>` hashed different relative paths and never did.
- **Never configure a fresh tree per stage** when an existing tree can be reused: reconfigure and
  rebuild it.
- **Iterate GPU-only**: `-DBATCHLAS_CPU_TARGET=none` (or the `dev-gpu` / `dev-gpu-tests` presets)
  builds nvptx64 only: 248 s cold against full-target colds of 284-334 s on the same box (13-25%;
  not a back-to-back pair), and every relink is cheaper. The NETLIB/CPU instantiations need the CPU
  target, so a GPU-only tree is **not a gate**: the final pre-push run uses full targets
  (`dev-tests`/`cuda` or a default configure), and the vendor-free tree
  (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) is built **only at that final gate**.
- Build only the target you work on (`scripts/b stedc_tests`). Every test target is in the default
  target, so a plain build links them all.
- **`BATCHLAS_SYCL_LINK_JOBS` matters for big links.** llvm-foreach runs batches of N modules and
  waits for each batch's slowest, so a link costs the sum of its batch maxima. Default `auto` = the
  logical cores, 4..32 (20 on the 4090 box). One module cannot be split: a one-module .so gains
  nothing, and that module is the library's floor.
- **The CPU AOT pass (`opencl-aot`) was most of a full-target link** (942 of 1,186 CPU-seconds in
  `libbatchlas_sycl`). A kernel whose every native `can_run` arm needs `d.is_gpu` belongs in a
  `NO_CPU_TARGETS` library. gemm keeps its CPU image, so its reg/wide rows are generated shard TUs
  (`BATCHLAS_GEMM_SHARDS`, `src/ops/CMakeLists.txt`; a `static_assert` catches an unsharded row):
  one gemm TU was two ~400 s modules.
- To profile a device link per module, replay the driver's `-###` commands one stage at a time
  and run each llvm-foreach item yourself (`sycl-post-link -split=auto` writes a `.table` naming each
  module's `.sym`); per-module CPU AOT time is what to balance shards on.
- `cmake --build --target A B C` with many targets can exit 0 and still skip some links.
  `scripts/b` checks each requested output (exit 70); by hand, `test -x build/tests/<name>`.
- Measured and ruled out (do not retry): Ninja (the Makefile null build is already 0.07 s), lld, PCH
  (the driver cannot emit one), unity builds (kernel-name collisions), `-fno-sycl-rdc`, explicit
  `-fsycl-device-code-split`, cutting one edge of a header cycle (the umbrella header cuts both
  edges; `docs/design/build-performance.md`).
- `-fsyntax-only` together with `-fsycl-targets` is rejected by this driver. Use real `-c` compiles
  as the oracle.
- To time a real rebuild, append a line rather than `touch`: with ccache a touched file is a cache
  hit and the timing measures only the link.
- ROCm TUs can be syntax-checked without an AMD GPU: `scripts/rocm_syntax_check.sh` (headers in
  `/opt/rocm/include/roc*/`; gone from the 4090 box since the 26.04 upgrade, so it skips every TU
  there). Run it after touching `src/backends/roc*.cc` or any signature or
  instantiation the ROCm TUs also spell out.
- Tiny-kernel TUs build with `-mllvm -pragma-unroll-threshold=262144`. Without it LLVM silently
  declines `#pragma unroll`, register arrays go to the stack, and tests stay green.

## 8. Testing Policy

- **Do not run the full ctest by default.** Scope to what you changed: one case
  `./build/tests/X --gtest_filter=...` -> one binary `ctest -R '^X(_heavy)?$'` (`-R` is a substring
  regex, so anchor it) -> one component `-L util|blas|ortho|tridiag|eig|sparse` -> the quick tier
  `-LE 'slow|heavy'` -> `-LE slow` -> full only before pushing or after touching shared code
  (`Queue`, `Matrix`/`MatrixView`, the mempool, `sg_partition`, `include/batchlas/util`).
  `scripts/affected_tests.py` maps a diff to the binaries (callers included: a gemm change runs
  potrf, geqrf, syev...) and prints the `scripts/b` and ctest lines; shared code falls back to the
  quick tier. It over-selects by design (grep-based caller graph); the gate still runs everything.
- **Case tiers.** A binary in `BATCHLAS_HEAVY_TESTS` (`tests/CMakeLists.txt`) is two ctest entries:
  `X` runs its quick cases and `X_heavy` (label `heavy`) its saturating-batch race guards and
  AutoReads table sweeps. `ctest -R '^X$'` is the quick half only; before claiming a binary green,
  run `X_heavy` too. `-LE slow` includes every heavy entry; the quick tier `-LE 'slow|heavy'` does
  not. Serial on the 4090 box: quick tier 416 s (104 entries), full suite 1287 s (138);
  `gemm_candidates_tests` 36 s + `_heavy` 353 s (was 1063 s in one entry).
- **Full-target tree on the 4090 box:** run ctest with `ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'`
  and `CUDA_VISIBLE_DEVICES=0` (GPU 1 is usually the other user's). Without the selector the CPU
  backends land on `native_cpu` and fail with "No kernel named".
- **On threadripper02, run multi-test scopes through `scripts/ctest_gpus.sh`** (same arguments as
  ctest, e.g. `scripts/ctest_gpus.sh -LE slow`; `--test-dir <build>` first, default `build/`).
  - Configure writes `<build>/ctest_resources.json` from `nvidia-smi --list-gpus`
    (`BATCHLAS_TEST_GPUS=<n>`, `=0` disables; `BATCHLAS_TEST_GPU_SLOTS`, default 2). Every GPU test
    has `RESOURCE_GROUPS gpus:1`, and `tests/ctest_gpu_env.sh` sets `CUDA_VISIBLE_DEVICES` to its
    slot's GPU. A pre-set `CUDA_VISIBLE_DEVICES=1` trims the run to that one GPU.
  - Only when configure's `sycl-ls` lists a `[cuda:gpu]`; otherwise no spec is written and you run
    plain `ctest`, which ignores all of this.
  - `syev_cta_tests` and `consumer_package_tests` are `RUN_SERIAL` (flake; three parallel consumer
    builds).
  - Measured `-LE slow` (88 tests, threadripper02, 4x RTX PRO 6000): serial with
    `CUDA_VISIBLE_DEVICES=1` 469 s, `ctest_gpus.sh` 1 slot/GPU 126 s, **2 slots/GPU 78 s** (critical
    path is `gemm_candidates_tests` alone; 115 s for the 96 tests after #147); failing names
    identical in all modes.
  - **Validated only on threadripper02** (64 cores, ample RAM). The 2x4090 box (20 cores, 125 GB,
    zero swap) was OOM-killed twice by a build or second ctest beside ctest
    (`docs/ci.md#oom-kill-during-ctest`); keep its full gate serial until `ctest_gpus.sh` is
    measured there.
- Observed on threadripper02 only, cause unknown: a process that sees every GPU is slow.
  `trsm_candidates_tests` took 132 s with 4 GPUs visible and 18 s with one (one case: 0.37 s with 1,
  1.2 s with 2, 5.1 s with 4). `ONEAPI_DEVICE_SELECTOR=cuda:N` did not help, so the cost is below
  SYCL device selection (CUDA or UR adapter/platform init); not checked on the 2x4090 box. Running a
  binary by hand there: set `CUDA_VISIBLE_DEVICES=<n>`.
- `BATCHLAS_TEST_BACKEND=CUDA` skips the NETLIB/CPU instantiations and `BATCHLAS_TEST_FLOAT_TYPE=float`
  skips the other types. Both are GTEST_SKIPs (case list unchanged), but the coverage is silently gone.
- **main is not green.** Accepted failures are in `tests/known-failures.txt`; CI diffs against that
  ledger (`docs/ci.md`). Compare failing test **names** against a baseline, never pass/fail counts.
  To attribute a failure: save your diff, `git checkout <base> -- <files>`, rebuild only that test
  target, re-run, compare names.
- `syev_cta_tests` has a concurrency-dependent wrong-answer flake (a few percent of runs under
  `ctest -j2` on the 4090 box), so it is `RUN_SERIAL` (`BATCHLAS_RUN_SERIAL_TESTS` in
  `tests/CMakeLists.txt`). 0 of 15 concurrent runs on threadripper02 does not rule out a few-percent
  rate. `sytrd_blocked_tests` also flakes on untouched main. Check the baseline before blaming your
  change.
- A CUDA-off (CPU-only) tree shows many fake failures: unbuilt tests appear as "Not Run", and there is
  no sub-group size 32, so CTA kernels throw.
- Before pushing, if you touched `cmake/`, `include/` or the install rules, run
  `sh .github/ci/run_local_checks.sh`. It includes the per-file comment-density gate (§12).

**Writing tests: guards that cannot fail.** More than a dozen recorded cases of a green suite
passing a wrong answer. Every new or rewritten test needs:

1. A deliberate break of the code under test, once per axis, giving a *narrow, named* red set. A
   break that turns everything red is too coarse to prove anything.
2. The break must run where the code is reachable. If the table sends the shape to the vendor, the
   native kernel is never exercised. Pin it (`BATCHLAS_<OP>_ROUTE=<spelling>` or a
   `select::ScopedPin`) or use the vendor-free build.
3. Non-natural values for every accessor the op reads (`ld`, batch `stride`, `inc`, rows!=cols), plus
   complex data with a nonzero imaginary part, ConjTrans, and beta!=0. The bug to catch is a kernel
   that *derives* a value instead of reading it; derived values are correct in the natural case.
4. Poison the code under test will accept: an in-range index with a large finite value, not NaN or an
   out-of-range index that the kernel's own guard discards. It must also survive the upstream code
   (cuSOLVER Upper `potrf` overwrites the lower triangle).
5. A saturating-batch case for any shared-local-memory kernel: batch >=1024, every item the same
   matrix, each result bit-identical to item 0. Small batches cannot race, and work-group-size
   ladders put every small test on wg=32, the one width where cross-sub-group SLM races are hidden.
6. Thresholds straddled in both directions. A routing test covering one shape proves nothing about
   the router.
7. No self-referential reference paths. A reference that goes through the same tuning tables or
   routes as the code under test agrees with it by construction.
8. Graded input (e.g. condition numbers ~1e±6) for eigen/tridiagonal accuracy. GOE-style random
   matrices cannot see relative-accuracy bugs.
9. For a capacity guard, a *launch* at the advertised ceiling, not an arithmetic re-derivation.
10. To verify a revert of an untracked file, use `md5sum` against a pristine copy (`git diff` prints
    nothing).
11. Verify results with `batchlas::verify`; in a test the macro is
    `EXPECT_VERIFY(T, Check::kind, n, value)` (`tests/test_utils.hh`; @ref design_verification). No
    local residual, reference or tolerance helpers.

## 9. Architecture: Routing and Dispatch

**Flat selection** (every op that reads `BATCHLAS_<OP>_ROUTE`; `docs/design/flat-kernel-selection.md`).
There is no RouteTable and no `include/batchlas/blas/dispatch/`.

- `src/ops/<op>/choice.hh` lists an op's choices (a `std::variant` of families with int knobs, e.g.
  `lpanel:panel=8`). `src/ops/<op>/<op>.cc` holds `key_of`, `can_run()`, launch and workspace.
  `select::run` / `select::pick` (device, choose, the vendor-free `NoRouteError`, trace, coverage)
  and the op's `select::OpSpec` (Op, vendor `Lib`, last resort) do the rest.
- `can_run()` is correctness only: false means the launch would throw or give a WRONG ANSWER. Never
  put a speed threshold there, or the shape becomes unservable vendor-free.
- Speed lives only in tables. `select::choose` (`src/select/select.hh`) takes the first runnable
  entry of the nearest row of `tuned/<op>.<device>.txt`: exact keys match by longest prefix (dropped
  from the right until a row matches); `n`/`batch` are log distance, weighted per key (`n:log:3`: potrf work ~ n^3 x batch). A device with no table
  borrows the nearest one and warns once; the CPU never borrows; nothing runnable falls to the op's
  last resort.
- Tables, ledgers, tiers, staleness and generation: @ref design_tuning_data (the only normative page).
- `BATCHLAS_TUNED_DIR=<dir>` overrides tables without a rebuild; `BATCHLAS_SELECT_TRACE=1` prints
  every choice and the runner-up.

Per-op choices:

| Op | Choices |
| --- | --- |
| trsm | `cta`, `sg_left`, `blocked`, `vendor` (a `cta` or `sg_left` pin above order 32 throws) |
| posv | `tiny`/`cta`/`blocked`, no vendor family; its children select for themselves |
| gemm | `direct`, `tiled`, `small`, `reg:m=..:n=..:k=..:u=..` (float), `wide:m=..:n=..:k=..`, `vendor`; the aligned/predicated leg is derived inside the launcher, never a field (`docs/perf/gemm.md#choices-flat-selection-p34`) |
| syrk | `gram` (n <= 128), `triangular` (float and double), `vendor`; keys `form` = sq\|tall\|wide and `trans` |
| syr2k | `triangular` (float and double; a real ConjTrans runs as Trans), `vendor` |
| symm | `expand` (mirrored expansion + public gemm, float and double, one item's scratch must fit), `vendor`; key `form` = sq\|tall\|wide |
| hemm | `expand` (conjugating mirror + public gemm), `vendor` |
| herk | `gram` (n <= 128), `triangular` (cfloat), `fold` (public gemm of the n x n product + fold), `vendor`; key `trans` |
| her2k | `triangular` (cfloat), `fold`, `vendor` |
| trmm | `triangular` (Side::Left), `expand` (expansion + public gemm, either side), `vendor` (the `cublas?trmm` loop only; the old `vendor` pin meant expand+gemm, that is now `expand`) |

Level-3 (`docs/design/flat-kernel-selection.md` §12.9): all `NoFields`, native families on CUDA and
SYCL GPUs (gemm-based `expand`/`fold` on ROCm too), homogeneous batch; every native launch chunks the
batch to grid z (65535) and the scratch families also to the scratch budget, so no `can_run` has a
batch term. rocBLAS wraps no hemm/herk/her2k: their vendor family refuses ROCm. Every level-3 vendor family refuses a
heterogeneous batch on every backend (the loops run each item at the top-level extents), so a
heterogeneous call has no route. `cublasdx` and the old level-3 pin-word list are gone: every
`cublasdx` pin throws as an unknown family.

**Env override** `BATCHLAS_<OP>_ROUTE` (`settings().routing.route("<op>")`, 22 ops) takes `auto`,
`native`, `vendor` or a spelling (`lpanel:panel=8`, `lpanel:8`), case-folded and trimmed.

- **A bad pin THROWS** (unparsable, not a compiled candidate, or cannot run the shape). Only the class
  words `native` and `vendor` with no runnable choice of their class fall back to Auto, with a
  warning (so `vendor` in a vendor-free build means Auto).
- No aliases: the old `native:cta`, `vendor:auto`, `batchlas_cta` and gemm kernel names (`tiled16`,
  `128x128x8`) throw; `BATCHLAS_<OP>_VARIANT` / `_PROVIDER` / `BATCHLAS_GEMM_SYCL_KERNEL` are not read.
- Coverage `chosen_algo` is the spelling.
- The value comes from the settings snapshot: a raw `setenv` mid-process is not seen until
  `detail::reload_settings()` (`ScopedEnvVar` does it;
  `docs/design/environment.md#environment-route-pins-are-raw-strings`).

**hemm, herk and her2k** select from tables like every other op (their sm_89/sm_120 rows transcribe
the deleted cuBLAS-backend rules). `BATCHLAS_EXPAND_ROUTE` and herk's `BATCHLAS_SYRK_ROUTE=gram` opt-in
are gone: pin `BATCHLAS_HEMM_ROUTE`, `BATCHLAS_HERK_ROUTE`, `BATCHLAS_HER2K_ROUTE`.

**cuBLASDx/MathDx is deleted** (code, MathDx probe, `BATCHLAS_ENABLE_CUBLASDX`,
`BATCHLAS_HAS_CUBLASDX`, `BATCHLAS_GEMM_CUBLASDX_KERNEL`); it never ran, MathDx was absent on both
boxes.

**Entry points and vendor-free builds**

- Public entry points live in `src/ops/<op>/<op>.cc`. Vendor-availability constants and `throw_no_vendor_route` are in
  `src/select/vendor.hh`; `batchlas::NoRouteError` is in `<batchlas/no_route.hh>`.
- `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` builds a vendor-free library; **that is the only valid
  vendor-free measurement**. With nothing runnable an op throws `NoRouteError` and records a coverage
  `miss` row.

**Verifying routes**

- **Linked != reached.** A symbol in `nm`, a `linked` coverage row or a green test count does not show
  a kernel runs. Set `BATCHLAS_COVERAGE_OUT` and read the `reached` rows. `scripts/route_diff.sh`
  captures and diffs them; `coverage_merge.sh` merges per-pid shards. Verify a routing change by
  *diffing the chosen route* across the input space, not by timing it.
- For a migrated op (gemm included) `chosen_algo` is the full spelling (`reg:m=128:n=128:k=8:u=1`),
  so route_diff sees gemm tile changes. It does not see a derived leg (gemm's aligned vs predicated)
  or a non-migrated op's internal kernel pick: use `BATCHLAS_KERNEL_TRACE=1` (never while timing, it
  inflates ~60%).
- `tools/tune --gate` against a pre-P3.4 parent reads its gemm coverage as `native:register_tiled`,
  which now means "best native": gate gemm from a kernel-trace old-choice CSV.
- Sub-ops count: a "native" syev, geqrf or getrf can still call cuBLAS gemm underneath. Every library
  gemm call (blocked drivers' default seams, hemm's `expand` and herk's/her2k's `fold`, symm's and
  trmm's `expand` families) goes through the public `gemm` and its table (`gemm_custom` and the
  `gemm_use_sycl_custom` re-route are deleted).

**Routing bug patterns**

- Highest-yield perf pattern: **a finished candidate that no table row ranks first.** Before writing a
  kernel, list the op's candidates and check each can ever be chosen: is it timed in the tables, does
  `can_run` ever admit it? Look for hard `return false` and for terms naming a different family's
  type list.
- **Read the predicate, not the comment.** Comments describing routing windows drift.
  `docs/perf/<op>.md` quotes each `can_run` term and window with its file and line number and gives bracketing
  evidence. Several hundred source comments cite `evidence: docs/<page>.md#anchor`, so renaming a
  heading breaks them; `check_evidence_anchors.py` catches this and requires every cited slug to be
  unique across `docs/` (Doxygen numbers repeated headings site-wide). Give a cited heading
  distinctive text. The generated Evidence index lists what cites each section.
- A kernel's fast-*leg* predicate (one the dispatcher re-evaluates to choose aligned vs predicated)
  used as a *routing* gate sends the call to a different, slower kernel. Found twice in
  `select_kernel_variant` (1.7x and 3-4x; deleted in P3.4). Under flat selection a leg is derived in
  the launcher and must never become a family field, a key or a `can_run` term (layout
  `packed|strided` is a table key, not a gate).
- Guarding on a device family (`BATCHLAS_HAS_CUDA_BACKEND`) where the code needs a *library* is a
  recurring bug; only a vendor-free build exposes it.

**API traps**

- The variadic queue-dispatch overload (`BATCHLAS_DISPATCH_ON_QUEUE`) must be `requires`-constrained.
  Even constrained, it beats option-struct overloads for non-const lvalue and prvalue arguments, so
  per-call checks that live only in `options.hh` are unreachable on the ordinary spelling. Test checks
  through the ordinary spelling.
- Write `PotrfOptions{}`, never a bare `{}`: `{}` selects the positional `Uplo{}` = Upper.
- Workspace sizing: `BumpAllocator` checks the alignment-rounded size but advances by the raw size, so
  an exactly simulated size is too small. Size with `BumpAllocator::measuring()` + `required_bytes()`.
  A `*_buffer_size` and its run path must branch identically. Sizing mode hands out unbacked pointers:
  never test `ws.data() != nullptr` to mean "the caller passed a workspace"; use separate overloads.
- The workspace arena belongs to the `Queue`, so a per-call `Queue` defeats pooling. Leak long-lived
  SYCL objects held in statics: static destruction after the runtime is gone hangs.
- `linalg::` wrappers that own a local copy of `A` wait before returning (`~Matrix` frees USM without
  waiting); multi-launch wrappers order themselves on an out-of-order Queue.
  `linalg::qr` is deliberately absent because of an unexplained cross-Queue wrong-answer defect
  (`docs/design/known-defects.md#linalgqr-returns-a-wrong-qr-after-an-earlier-call-in-the-process`).
- Pivots are packed 1-based int32 everywhere. `cusolverDnXgetrs` wants int64.
- Known located-but-unfixed defects: `docs/design/known-defects.md`.

## 10. Measurement Rules (benchmarks lie silently here)

- **Compare algorithms only at saturation** (doubling the batch buys <10% per item); below it you
  measure launch, allocation and clock-ramp overhead. **Benchmark and tune at batch >=128; batch=1
  does not matter to this project.** Still *profile* across batch sizes: batch-only-parallel kernels
  (one work-group per matrix) starve at small batch. If a kernel's work-group count equals the batch
  size, that is the bug.
- **Do not tune or optimize double / complex<double> on a 1:64-FP64 GPU** (RTX 4090 sm_89, RTX PRO
  6000 sm_120, every consumer and workstation part): FP64 is compute-bound there almost everywhere,
  so kernel and routing work buys little and its windows do not carry over. Keep FP64 *correct*
  (tests, no spills), skip it in retunes (`--dtype float,cfloat`), and defer its performance work
  to high-FP64 hardware: a data-centre GPU (1:2) or the CPU backend.
- **Run one measuring process on the box at a time**, whatever the device pin. Two GPUs on one NUMA
  node and one UVM driver produced 5.5x errors with stable rel_sd while `nvidia-smi` showed no
  foreign process. A display attached to a GPU also slows L2-resident vendor cells: measure on a
  headless GPU. Use `benchmarks/gpu_guard.sh`, plus a `flock` against your own duplicate jobs.
- Warm up: idle clocks are ~210 MHz. Use `--warmup=3 --min_iters=8` and discard a first pass over
  every cell. CUDA device images are AOT SASS: no JIT, ~0.3 s module load per process. The CPU image
  of a full-target tree (`spir64_x86_64`) is AOT too (`opencl-aot` at the device link). JIT applies
  to a generic `spir64` target, AdaptiveCpp `generic` and Metal images, whose first process pays it.
- Alternate the A/B order between reps and discard the first pass. Per-rep CV is ~3%, so an effect
  under 5% needs 14-16 reps; 6 reps is a screen, not a verdict.
- The Google Benchmark `--name` filter is a *substring* match. Filter on the CSV name column, which is
  quoted and contains a comma (`awk -F,` offsets shift).
- Admit rows on cross-pass reproduction, not rel_sd alone: a noise filter that deletes a reproducible
  loss fabricates a window. Bracket every window edge with a measured non-winner.
- Before quoting "vs vendor": check which route ran, what "the vendor" is (cuBLAS complex trsm and
  batched orgqr are per-item loops or our own substitute kernels; cuSOLVER `syevjBatched` beats the
  `XsyevBatched` we call at n<=32), and re-measure the vendor, because versions move.
- Rebuild every bench binary after a table or `can_run` change: harnesses resolve the *printed* route
  in their own TU, so relinking only the `.so` leaves the route column lying.
- Check that an argument is used, not just echoed: time should scale with it. Harnesses written before
  the code (design studies) do not benchmark the shipped code.
- Sanity anchors (RTX 4090): real FP32 SGEMM ~47-54 TFLOP/s (~80 means TF32), FP64 ceiling ~1.44
  TFLOP/s (1/64 rate), DRAM ~1008 GB/s peak, ~950 achievable. cuBLAS gemv is already at the DRAM
  roof: do not try to beat it.
- A cost table that is non-monotonic in n (a larger n running faster) is a routing bug.
- nsys sees cuBLAS kernels; `BATCHLAS_KERNEL_TRACE` does not (it hid 35% of a cfloat syev). Also
  useful: `BATCHLAS_KERNEL_TRACE_PATH` (Chrome trace), `BATCHLAS_GESVD_PROFILE=1`, `BATCHLAS_TUNE_*`
  overrides (A/B any tuning constant without a rebuild), `BATCHLAS_BENCH_LD_PAD[_A/_B/_C]` and
  `BATCHLAS_BENCH_BETA` in the gemm benchmarks.
- Always confirm a GEMM at beta=1 and with a strided `ld`: panel updates always carry their parent's
  `ld`. Pin one gemm kernel with `BATCHLAS_GEMM_ROUTE=<spelling>` (`reg:m=128:n=128:k=8:u=1`,
  `wide:m=64:n=64:k=16`, `small`, `tiled`, `direct`; old kernel names such as `128x128x8` and
  `tiled16` throw); it throws if that tile has no instantiation for the call's transpose form.
  `BATCHLAS_GEMM_SYCL_KERNEL` and `BATCHLAS_GEMM_EXPERIMENTAL` are not read. Compare gemm kernels with
  `tools/tune/batchlas_tune` (`gemm_spec.cc`: beta=1, complex alpha, strided cells padded by
  `--ld-pad`, every arm verified).
- benchviz (`python3 benchmarks/benchviz`) measures whatever is in *its checkout's* `build/`, not HEAD:
  run `benchviz info` first. Its batchlas arm pins the native walk, not Auto. Figures follow the
  user's style (`benchmarks/benchviz/style.py`: CM usetex, full box, dotted lines with large markers,
  2σ band, viridis maps), not generic defaults.
- Raw result grids go in `benchmarks/results/` (Git LFS; run `git lfs install` on every machine, or
  the commit stores raw files and CI's `check_lfs_pointers.py` fails). Older evidence is at tag
  `perf-evidence/vendor-independence` (`git show <tag>:experiments/...`). `docs/perf` is the distilled
  layer.
- Retuning takes ~12 min (`evaluation/tuning/`). The checked-in `include/batchlas/tuning_params.hh` is
  the only copy (nothing is generated); port constants into it by hand. **Tuning was float-only**:
  constants measured on float are applied to double and complex, so when another type looks bad, sweep
  that type first. Check whether a knob has a second consumer (aliasing) or is bypassed on the path
  (shadowing) before trusting "no change".

## 11. GPU Kernel Design Facts (sm_89; mostly general)

- Register gate: registers are split across four sub-partitions of 16,384. The bound is
  `ceil(warps/4)*32*regs <= 16384`, **not** `regs*wg <= 65536`. SYCL accepts the launch and the driver
  rejects it as an uncaught abort. Standalone `-Xcuda-ptxas -v` register counts differ from the
  in-library kernel: prefer a measured launch ceiling.
- Design against occupancy, not spills: 8x8 double tiles do not spill, but `complex<double>` x 512
  threads exceeds the register file. Shrink the thread tile as the scalar widens. Never pass an
  accumulator element by `T&` out-parameter (spills the whole array, 43%); return by value.
- `std::complex` `*` emits an isnan branch plus `__mulsc3` (Annex G, no `-ffast-math`). Write out the
  real arithmetic *in the hot loop only* (1.2-1.3x); converting every site cost occupancy and was
  slower. Do not add `-ffast-math`.
- The 48 KB SLM launch hole: an attribute sticky per CUfunction means test launch *order* can hide it.
  Put the discriminating launch first.
- Launch bounds (`intel::min_work_groups_per_cu`) apply only on a **functor's** `operator()`, not a
  lambda. Worth 1.1-1.9x on tiny kernels, but a cap that is too tight is 2.3x slower, and a change
  moves routing windows.
- SLM staged by a lane-strided loop and read by a different index needs a barrier. NVIDIA hides the
  race at wg=32.
- Column-major coalescing: one work-item per *output* element. With that mapping NoTrans gemv is
  coalesced and Trans needs the reduction. Row interchanges (laswp) should be an index gather in SLM,
  not a per-column walk (7.9x fewer sectors).
- GEMM: 128x128x8 tiles with an 8x8 thread tile (64 accumulators), aligned (not +1) shared strides and
  B staged [k][n] give cuBLAS parity for float NN. The epilogue must vary the m index fastest.
  Internal GEMM demand is *panel updates* (large m,n; k = blocking factor 8-136; ~60% transposed), so
  a `min_dim` gate never fires on it. The native GEMM is ~2x off on strided `ld` from exposed B-load
  latency; double-buffering and packing B were both measured as no gain.
- DPC++ `chunked_partition` masks cost ~4-5 instructions per collective. Full-warp or maskless
  rewrites lost in real kernels even where microbenchmarks won. Judge partition-primitive changes
  (`src/kernels/common/sg_partition/`) by real-kernel A/B (steqr/syev_cta benchmarks, n=5..16, batch
  16384), never by microbenchmark.
- CTA kernels are capped at n <= 32 (the sub-group width). `STEDC_RECURSION_THRESHOLD = 32` is that
  invariant, not a tuning constant: a leaf of 33+ falls from `steqr_cta` to `steqr_wg` (14x cliff).
- Traffic models that count bytes miss serial recurrences and lost parallelism. Check the critical
  path and the work-item count first.

**Measured dead ends: do not re-propose without a new idea**

| Idea | Result |
| --- | --- |
| Lifting the CTA eigensolver above n=32 | 85-211x slower |
| Block Jacobi as a speed path | 2-11x slower (accuracy-only opt-in) |
| Cooperative TRSM diagonal solve | dead end |
| Packed sub-group getrf with runtime n in SLM | dead end (register-resident designs are *not* refuted) |
| Faster native gemv | dead end |
| Maskless steqr at P<32 | dead end |
| Givens chase | Householder chase is 5.6x cheaper |
| GEMM kernel fix for strided `ld` | the fix was routing |

The latrd panel symv is L1-over-fetch-bound (12-16x), not DRAM-bound; a MAGMA-style single-read symv
is the open 2.7x.

## 12. Repository Conventions and Git

- **Docs are a Doxygen 1.18 + doxygen-awesome site** and the project's database of results and
  decisions: `sh scripts/build_docs.sh [out]` (or `-DBATCHLAS_BUILD_DOCS=ON`, target
  `batchlas_docs`). The governing page is `docs/developer/documentation.md`; read it before adding a
  page. CI's `docs` job builds with warnings as errors and deploys `main` to GitHub Pages.
- **Where things go:** measurements, A/B grids, rejected alternatives -> `docs/perf/<op>.md`; design
  decisions -> `docs/design/`; math -> `docs/algorithms/`; user guides -> `docs/guide/`; process ->
  `docs/developer/`; raw data -> `benchmarks/results/` (LFS). A new page opens `# Title {#area_topic}`
  plus a one-line status. Math is `\f$…\f$` / `\f[…\f]`, not `$…$`.
- **No Markdown outside `docs/`** except a directory `README.md` (never at the root),
  `.claude/CLAUDE.md` (it only imports this page) and `.github/`, which holds the project README
  (`check_markdown_locations.py`). No root Markdown at all, so no root plan files: put conclusions in
  the area page.
- **Source layout** (table: `docs/extending.md`): `src/ops/<op>/` holds `<op>.cc`, `choice.hh` and
  headers shared by the op's kernels; `kernels/` holds its kernel TUs and single-consumer headers.
  Internal sub-ops nest under the parent (`syev/{stedc,steqr,sytrd,tridiag}/`,
  `gesvd/{gebrd,bdsqr,bdsdc,ormbr}/`). Shared device code: `src/kernels/common/`. `src/sycl/` is
  only DPC++/AdaptiveCpp portability; `src/backends/` only vendor wrappers; `src/util/` also holds
  the `batchlas_extra` sources. CMake lists (`src/ops/CMakeLists.txt`) group by device-link
  cluster, not folder, and keep the old `batchlas_extensions_*` names. In-tree includes are
  root-based, `#include <src/ops/potrf/potrf_native.hh>`, `<tests/test_utils.hh>`; never quoted or
  relative (`check_no_unprefixed_includes.py` fails on `#include "..."`).
- **Comment density <=18% per file** (`.github/ci/check_comment_density.py`). A code line with a
  trailing comment counts as code. The repo-wide average hides per-file violations: check per file.
  - A comment is for invariants, "looks wrong but is deliberate" notes and traps, plus an
    `evidence: docs/<page>.md#anchor` pointer.
  - Doxygen doc comments (`///`, `//!`, `/** */`) in `include/batchlas/` are exempt, but only for the
    contract (brief, params, returns, `@pre`, `@throws`, `@ingroup` from `docs/pages/api_groups.dox`);
    rationale in `///` is still rationale.
  - Waivers: `.github/ci/comment_density_waivers.txt`.
- A `main` ref from session start is a snapshot. `git fetch origin` before claiming something is absent
  from a branch.
- For stacked PRs, `merged` can mean merged into the previous stack branch, not into main. Read
  `base.ref`.
- `/code-review ultra` rejects diffs over 500 files or 8,000 lines. Split along work-package commit
  boundaries: a by-op split does not compile, because the dispatch facade includes every op's native
  header.
- `nm -DC ... | awk '{print $3}'` truncates demangled names at the first space, and `nm -C` cannot
  demangle concept-constrained templates. Diff full mangled names (`scripts/facade_symbol_check.sh`).
- In worktree-isolated agent sessions, the Bash guard refuses heredocs, `for` loops with computed
  arguments and awk programs. Write scripts to `$CLAUDE_JOB_DIR/tmp` and run them. Wait on a PID with
  `kill -0`, not `pgrep -f` (it matches itself).

## 13. Primary-Machine Facts (the RTX 4090 box; re-check on any other machine)

- 2x RTX 4090 (128 SMs, 72 MB L2, 24 GB, FP64 1/64), Xeon w5-2445, 20 cores, 123 GB, zero swap,
  Ubuntu 26.04 (upgraded 2026-10-20/21: GCC 15, gtest 1.17; a tree or ccache from before it fails
  test links with an undefined `MakeAndRegisterTestInfo`, so rebuild it), DPC++ at
  `/opt/dpcpp-cuda`, CUDA 13.2, no ROCm headers, MAGMA 2.10 Release/sm_89 at
  `~/magma/build-rel89`. Device 0 drives the display: measure on device 1.
- The box is shared with another user who runs GPU sweeps. Check `nvidia-smi` before timing anything
  and coordinate before a reboot. `perf` and `gdb -p` do not work (perf_event_paranoid, ptrace): use
  the UR trace.
- The system OpenBLAS is 0.3.32 since the upgrade and passes the configure-time dgemm check with no
  coretype; 0.3.20 picked the broken Cooperlake dgemm (Known Pitfalls).
- Unattended-upgrades is not blacklisted for `nvidia-*`, so driver mismatches recur.

On a **new machine**, before trusting any result: `sycl-ls` shows your GPU; configure prints the right
SYCL target; `nvidia-smi` (or equivalent) reports the architecture you think it does; record that
architecture's FP32/FP64/DRAM ceilings as sanity anchors. **Every routing window and tuning constant
in the tree was measured on sm_89** (some tables on sm_120). On a different GPU, an Apple GPU included, especially one with 1:2 FP64 where double
GEMM verdicts invert, treat the tuned tables (hemm/herk/her2k's transcribe hand-written windows) as
hypotheses. Re-measure at saturation before quoting any ratio, and record the machine alongside new
data in `benchmarks/results/`.

## License

SPDX-License-Identifier: MIT
