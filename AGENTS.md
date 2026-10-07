BatchLAS Agent Environment Guide

TL;DR  Build-time dependencies: CMake ≥3.17 (≥3.21 if you want the cmake --preset workflow), a C++20 compiler with SYCL 2020 support **that has a backend for your GPU vendor**, netlib LAPACK/LAPACKE + CBLAS, and oneDPL headers. Runtime dependencies are the same plus a GPU your SYCL runtime actually exposes. The configuration that is actually exercised here is a CUDA-enabled DPC++ (self-built intel/llvm, installed at /opt/dpcpp-cuda) against CUDA 13.2 and an NVIDIA RTX 4090 (sm_89) on Ubuntu 22.04 — see the "Tested platforms" table in README.md.

⚠️  **NVIDIA targets need a CUDA-capable DPC++.** The stock `intel-oneapi-compiler-dpcpp-cpp` package in §2 has **no CUDA adapter**. On an NVIDIA machine it configures cleanly, with no warning and no error, and builds a CPU-only library. Use either

* Intel oneAPI plus the Codeplay **oneAPI for NVIDIA GPUs** plugin, or
* a self-built `intel/llvm` configured with `--cuda` (this is what `/opt/dpcpp-cuda` is).

Check before you build: `sycl-ls` must list a `[cuda:gpu]` entry. And read the configure output — **`-- Using SYCL targets: spir64_x86_64` means you are about to build, and benchmark, a CPU-only build**, whatever GPUs are in the box. A CUDA-enabled configure names the architecture instead, e.g. `nvidia_gpu_sm_89`. Forcing `-DBATCHLAS_ENABLE_CUDA=ON` under a compiler with no CUDA adapter still does not give you CUDA, but it no longer does so silently: `ON` now means "require it", and the configure aborts with a `FATAL_ERROR` naming the missing `[cuda:gpu]` entry (see "Common CMake options" in README.md). It used to configure for `nvidia_gpu_sm_50`, compile and link with exit 0, and fail at run time with `No kernel named ... was found`. The default `AUTO` does not go down that path at all — with no `[cuda:gpu]` it simply builds the CPU-only library described above.

⸻

1. Prerequisite Packages

Component	Debian/Ubuntu (apt)	Fedora/RHEL-like (dnf)	Arch Linux (pacman)	Source build (fallback)
BLAS/LAPACK (Fortran APIs)	libblas-dev liblapack-dev	blas-devel lapack-devel	blas lapack	see §4
C interface (CBLAS & LAPACKE headers)	liblapacke-dev	lapack-devel	lapacke	see §4
Build tools	build-essential cmake git	@development-tools cmake git	base-devel cmake git	—
Git LFS (raw benchmark results; docs/perf/README.md)	git-lfs	git-lfs	git-lfs	—

Why not just libopenblas-dev? Ubuntu’s OpenBLAS package omits lapacke.h; you still need liblapacke-dev for the C interface, or build LAPACKE yourself. This is a packaging decision, not a BatchLAS bug.

⸻

2. Installing a SYCL Compiler

2a. Intel® oneAPI DPC++/C++ (icpx) — Intel GPUs and CPU-only builds

This is the easy path, and it is the WRONG path if you are targeting NVIDIA: the package below ships no CUDA adapter. See the warning in the TL;DR, and §2b.

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

After installation, configure the environment for each shell session:

source /opt/intel/oneapi/setvars.sh   # sets PATH, LD_LIBRARY_PATH, MKLROOT, etc.

You do not need the entire intel-basekit; the single compiler package is enough to build BatchLAS for CPU or an Intel GPU.

2b. CUDA-capable DPC++ — NVIDIA GPUs

Two options, both giving a clang-family `clang++`/`icpx` that emits NVPTX:

* **Codeplay oneAPI for NVIDIA GPUs**: install Intel oneAPI as in §2a, then the Codeplay plugin matching your oneAPI version. It adds the CUDA UR adapter, after which `sycl-ls` reports `[cuda:gpu]`.
* **Self-built `intel/llvm`**: clone https://github.com/intel/llvm and configure with `--cuda` (plus `--cmake-opt=-DCMAKE_INSTALL_PREFIX=<prefix>`). This is what this machine uses; it lives at `/opt/dpcpp-cuda` and is the compiler every number in the repository was measured with.

Either way you also need a CUDA Toolkit (13.2 here) and you point CMake at the compiler explicitly:

cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++

Note for consumers of an installed BatchLAS: the *whole* consuming project has to be configured with this same compiler. See "Consuming BatchLAS from CMake" in README.md for why.

2c. oneDPL

Several sources include `<oneapi/dpl/...>` unconditionally, so oneDPL headers are a hard dependency even for a CUDA build. A self-built `intel/llvm` (`/opt/dpcpp-cuda`) does not bundle them. The build looks under `/opt/intel/oneapi/dpl/latest/include`, which is what `sudo apt install intel-oneapi-dpl` gives you; oneDPL is header-only, so a clone of https://github.com/oneapi-src/oneDPL works too as long as its `include/` ends up at that path.

⸻

3. Verifying the Toolchain

icpx --version          # or: /opt/dpcpp-cuda/bin/clang++ --version
cmake --version         # ≥3.17 (≥3.21 for cmake --preset)
sycl-ls                 # must list your GPU, e.g. a [cuda:gpu] entry
pkg-config --exists lapacke cblas && echo "LAPACKE & CBLAS found"


⸻

4. Building netlib LAPACKE/CBLAS from Source (if distro packages are unavailable)

git clone https://github.com/Reference-LAPACK/lapack.git
cd lapack
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLAPACKE=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j $(nproc)
sudo cmake --install build   # installs liblapacke.so, libcblas.so, headers

Add the install prefix (e.g. /usr/local/lib) to LD_LIBRARY_PATH or run sudo ldconfig so that the linker can locate the libraries.

⸻

5. CMake Configuration Hints

BatchLAS searches for LAPACKE via find_package(LAPACK REQUIRED COMPONENTS CBLAS LAPACKE) and for SYCL via find_package(SYCL REQUIRED) (provided by Intel’s compiler). If your LAPACKE install lives outside standard prefixes, set:

export CMAKE_PREFIX_PATH="/opt/netlib:$CMAKE_PREFIX_PATH"


⸻

6. Quick Smoke Test

cmake -B build .
# GPU-only iteration tree (nvptx64 only, ~25% faster; never the gate, §7):
#   cmake -B build-gpu . -DBATCHLAS_CPU_TARGET=none

# Iterating on one algorithm: this builds the library and only this one test
# binary. Do not build the default target while iterating — it also builds the
# other 48 test executables, which you are not about to run.
cmake --build build --target stedc_tests -j"$(nproc)"
ctest --test-dir build -R '^stedc_tests$' --output-on-failure

# Before pushing, build and run everything, one test per GPU slot (§8). The
# final gate is a full-target tree (not BATCHLAS_CPU_TARGET=none), plus the
# vendor-free tree (-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF) built only now.
cmake --build build -j"$(nproc)"
scripts/ctest_gpus.sh --test-dir build

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


⸻

Known Pitfalls
	•	Silent CPU-only build on an NVIDIA box: the compiler has no CUDA adapter. `-- Using SYCL targets: spir64_x86_64` in the configure output is the tell. See the TL;DR and §2b.
	•	icpx (oneAPI 2026) defaults to `-fp-model=fast`, which stops LLVM inlining libspirv builtins: `sycl::fma`, barriers and id queries become out-of-line CALLs in ~98% of kernels (cfloat gemm 6-15x, potrf cfloat 6x slower), and every test stays green. The build passes `-ffp-model=precise` for IntelLLVM; `ctest -R '^device_calls_tests$'` (scripts/check_device_calls.py) scans the built PTX and fails on any `__spirv_*` call. Never add a later `-ffp-model=fast`/`-ffast-math`: the last flag wins. benchviz warns about such a build.
	•	oneAPI 2026 icpx on an NVIDIA box without the Codeplay plugin: oneAPI's own libsycl ships no CUDA UR adapter, so put a CUDA-enabled DPC++ runtime (`/opt/dpcpp-cuda/lib`) FIRST on `LD_LIBRARY_PATH`, ahead of `setvars.sh`'s entries, for configure, build and every run. Otherwise `sycl-ls` shows no `[cuda:gpu]` and configure silently picks `spir64_x86_64`.
	•	Missing lapacke.h: install liblapacke-dev even if you already have libopenblas-dev.
	•	Multiple BLAS providers: choose the backend with sudo update-alternatives --config libblas.so.3.
	•	device not found at runtime: ensure your GPU driver and its SYCL adapter match the compiler version — the Level-Zero runtime for Intel, the CUDA adapter and driver for NVIDIA.
	•	libsycl.so.9: cannot open shared object file when running anything: the DPC++ runtime is not on the loader path. Export LD_LIBRARY_PATH=<dpcpp-prefix>/lib, or add it to /etc/ld.so.conf.d and run ldconfig.
	•	Wrong double-precision results from the host (NETLIB) backend: some OpenBLAS builds pick a broken dgemm kernel (0.3.20 on Cooperlake Xeons; errors ~66 at n=128). Configure runs a dgemm health check (`cmake/BatchLASBlasHealthCheck.cmake`) and ctest sets `OPENBLAS_CORETYPE` for you; **a bare `./build/tests/foo` does not**. For a double-only CPU failure, re-run with `OPENBLAS_CORETYPE=SKYLAKEX` before blaming BatchLAS.
	•	Linking anything with `-fopenmp` binds to the NVIDIA HPC SDK's libgomp when it is on `LIBRARY_PATH` (symbol version literally `VERSION`), then fails at load time. Link with `env -u LIBRARY_PATH`; `readelf -d <lib> | grep gomp` should show `libgomp.so.1`.
	•	`nvidia-smi` says "Driver/library version mismatch": unattended-upgrades replaced the driver under the running kernel. Compare `cat /proc/driver/nvidia/version` with `modinfo nvidia | grep ^version`; only a reboot (or module reload) fixes it. Not a BatchLAS problem.
	•	A CUDA binary that "hangs at startup" at 100% CPU+GPU is usually a stuck kernel, not JIT. Find it with `SYCL_UR_TRACE=2 ./bin > trace.log 2>&1` (no pipe), map `urKernelCreate` handles to names, and read the last `urEnqueueKernelLaunch`. `compute-sanitizer` cannot attach (SYCL calls cuInit first); for GPU out-of-bounds, rebuild with `-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG"` to re-enable the device asserts in `VectorView::at` / `KernelMatrixView`.

⸻

7. Build Performance

The build is **SYCL device-link-bound**, not compile-bound. The unit of device linking is the shared library: changing one object re-runs `sycl-post-link` + ptxas + CPU-target AOT for every object in that `.so`, single-threaded, so `-j` cannot shorten it. Measured on threadripper02 (64 cores, 4x RTX PRO 6000, default target = library + 82 test binaries, -j32): fresh tree, no cache ~285–335 s.
	•	**ccache is on by default** (`BATCHLAS_USE_CCACHE`, cmake/BatchLASCcache.cmake) when `ccache` is on `PATH` or in `~/.local/bin`; configure prints `ccache enabled (...)`. No ccache? Drop the static binary from https://github.com/ccache/ccache/releases into `~/.local/bin`, no root needed. The generated `<build>/batchlas-ccache` wrapper sets depend mode, `CCACHE_BASEDIR` (= deepest common parent of source and build dir, `BATCHLAS_CCACHE_BASEDIR`), `CCACHE_NOHASHDIR` and sloppiness, so a **second tree hits the first tree's cache**: 187/196 hits, 203 s vs 284 s cold. The misses are TUs with an absolute path in a `-D`. Plain `CMAKE_CXX_COMPILER_LAUNCHER=ccache` without those settings got 3/196. Links are never cached, so ~170–200 s is the floor for a fresh tree.
	•	**Never configure a fresh tree per stage** when an existing tree can be reused: reconfigure and rebuild it. Even all-cache-hits, a fresh tree pays every device link and test link again.
	•	**Iterate GPU-only**: `-DBATCHLAS_CPU_TARGET=none` (or the `dev-gpu` / `dev-gpu-tests` presets) builds nvptx64 only: 248 s vs ~330 s fresh (-25%), and every relink is cheaper. The NETLIB/CPU instantiations need the CPU target, so a GPU-only tree is **not a gate**: the final pre-push run uses the full targets (`dev-tests`/`cuda` or a default configure), and a vendor-free tree (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) is built **only at that final gate**, never per edit.
	•	Build only the target you are working on (`--target stedc_tests`). Most test targets are EXCLUDE_FROM_ALL.
	•	An incremental relink of `libbatchlas_backends` after a one-line `select.cc` edit is ~17 s, whatever `BATCHLAS_SYCL_LINK_JOBS` is (4: 17 s, 16: 18 s). Do not tune it.
	•	`cmake --build --target A B C` with many targets can exit 0 and still skip some links. Check `test -x build/tests/<name>`, not the exit code.
	•	Measured and ruled out (do not retry): Ninja (the Makefile null build is already 0.07 s), lld, PCH (the driver cannot emit one), unity builds (kernel-name collisions), `-fno-sycl-rdc`, explicit `-fsycl-device-code-split`, cutting one edge of a header cycle.
	•	`-fsyntax-only` together with `-fsycl-targets` is rejected by this driver. Use real `-c` compiles as the oracle.
	•	To time a real rebuild, append a line rather than `touch`: with ccache on, a touched file is a cache hit and the timing measures only the link.
	•	The ROCm TUs can be syntax-checked without an AMD GPU: `scripts/rocm_syntax_check.sh` (headers live in `/opt/rocm/include/roc*/`, not directly under `include/`). Run it after touching `src/backends/roc*.cc` or any signature or instantiation that the ROCm TUs also spell out.
	•	Tiny-kernel TUs build with `-mllvm -pragma-unroll-threshold=262144`. Without it, LLVM silently declines `#pragma unroll`, register arrays go to the stack, and tests stay green.

⸻

8. Testing Policy

	•	**Do not run the full ctest by default.** Scope the run to what you changed:
	  one case `./build/tests/X --gtest_filter=...` → one binary `ctest -R '^X$'` (`-R` is a substring regex, so anchor it) → one component `-L util|blas|ortho|tridiag|eig|sparse` → `-LE slow` → full only before pushing or after touching shared code (`Queue`, `Matrix`/`MatrixView`, the mempool, `sg_compat`/`sg_partition`, `include/util`).
	•	**Any multi-test run goes through `scripts/ctest_gpus.sh`** (same arguments as ctest, e.g. `scripts/ctest_gpus.sh -LE slow`; `--test-dir <build>` first, default `build/`). Configure writes `<build>/ctest_resources.json` from `nvidia-smi --list-gpus` (`BATCHLAS_TEST_GPUS=<n>`, `=0` disables; `BATCHLAS_TEST_GPU_SLOTS`, default 2); every GPU test has `RESOURCE_GROUPS gpus:1`, and `tests/ctest_gpu_env.sh` sets `CUDA_VISIBLE_DEVICES` to its slot's GPU. Plain `ctest` ignores all of this. Measured `-LE slow` (88 tests, threadripper02, 4x RTX PRO 6000): plain serial 1737 s (another job shared the GPUs for part of it), serial with `CUDA_VISIBLE_DEVICES=1` 469 s, `ctest_gpus.sh` 1 slot/GPU 126 s, **2 slots/GPU 78 s** (the critical path is `gemm_candidates_tests` alone; 115 s for the 96 tests after #147); failing names identical in all modes.
	•	A process that sees every GPU is slow: `trsm_candidates_tests` takes 132 s with 4 GPUs visible and 18 s with one (one case: 0.37 s with 1 visible, 1.2 s with 2, 5.1 s with 4). It is CUDA-level: `ONEAPI_DEVICE_SELECTOR=cuda:N` does not help. Running a binary by hand on a multi-GPU box? Set `CUDA_VISIBLE_DEVICES=<n>`.
	•	`BATCHLAS_TEST_BACKEND=CUDA` skips the NETLIB/CPU instantiations, and `BATCHLAS_TEST_FLOAT_TYPE=float` skips the other types. Both are GTEST_SKIPs, so the case list is unchanged, but that coverage is silently gone.
	•	**main is not green.** The accepted failures are listed in `tests/known-failures.txt`, and CI diffs against that ledger (docs/ci.md). Compare failing test **names** against a baseline, never pass/fail counts. To attribute a failure: save your diff, `git checkout <base> -- <files>`, rebuild only that test target, re-run, and compare names.
	•	`syev_cta_tests` has a concurrency-dependent wrong-answer flake (fails a few percent of runs under `ctest -j2` on the 4090 box; 0 of 15 concurrent `ctest_gpus.sh` runs on threadripper02, so it is not `RUN_SERIAL`; if it flakes there, give it `RUN_SERIAL`). `sytrd_blocked_tests` also flakes on untouched main. Check the baseline before blaming your change.
	•	A CUDA-off (CPU-only) tree shows many fake failures: tests that were never built appear as "Not Run", and there is no sub-group size 32, so CTA kernels throw.
	•	Before pushing, if you touched cmake/, include/ or the install rules, run `sh .github/ci/run_local_checks.sh`. It includes the per-file comment-density gate (§12).

**Writing tests: guards that cannot fail.** This repo has more than a dozen recorded cases of a green suite passing a wrong answer. Every new or rewritten test needs:
	1.	A deliberate break of the code under test, once per axis. Require a *narrow, named* red set. A break that turns everything red is too coarse to prove anything.
	2.	The break must run where the code is reachable. If the table sends the shape to the vendor, the native kernel is never exercised. Pin it (`BATCHLAS_<OP>_ROUTE=<spelling>` or a `select::ScopedPin`) or use the vendor-free build.
	3.	Non-natural values for every accessor the op reads (`ld`, batch `stride`, `inc`, rows≠cols) plus complex data with a nonzero imaginary part, ConjTrans, and beta≠0. The bug to catch is a kernel that *derives* a value instead of reading it, and derived values are correct in the natural case.
	4.	Poison that the code under test will accept. Use an in-range index with a large finite value, not NaN or an out-of-range index that the kernel's own guard discards. The poison must also survive the upstream code: cuSOLVER Upper `potrf` overwrites the lower triangle.
	5.	A saturating-batch case for any shared-local-memory kernel: batch ≥1024, every item the same matrix, each result bit-identical to item 0. Small batches cannot race. Work-group-size ladders also mean every small test lands on wg=32, the one width where cross-sub-group SLM races are hidden.
	6.	Thresholds straddled in both directions. A routing test that covers one shape proves nothing about the router.
	7.	No self-referential reference paths. A reference that goes through the same tuning tables or routes as the code under test agrees with it by construction.
	8.	Graded input (e.g. condition numbers ~1e±6) for eigen/tridiagonal accuracy. GOE-style random matrices cannot see relative-accuracy bugs.
	9.	For a capacity guard, a *launch* at the advertised ceiling, not an arithmetic re-derivation of it.
	10.	To verify a revert of an untracked file, use `md5sum` against a pristine copy (`git diff` prints nothing).

⸻

9. Architecture: Routing and Dispatch

	•	**Flat selection (every op that reads `BATCHLAS_<OP>_ROUTE`; docs/design/flat-kernel-selection.md).** There is no RouteTable and no `include/batchlas/blas/dispatch/` (deleted in phase 5). `src/ops/<op>/choice.hh` lists an op's choices (a `std::variant` of families with int knobs, e.g. `lpanel:panel=8`) and `src/ops/<op>/<op>.cc` holds `key_of`, `can_run()`, launch and workspace; `select::run` / `select::pick` (device, choose, the vendor-free `NoRouteError`, trace and coverage) and the op's `select::OpSpec` (Op, vendor `Lib`, last resort) do the rest. `can_run()` is correctness only: false means the launch would throw or give a WRONG ANSWER; never put a speed threshold there, or the shape becomes unservable vendor-free. Speed lives only in tables: `select::choose` (`src/select/select.hh`) takes the first runnable entry of the nearest row of `tuned/<op>.<dtype>.<device>.txt`: exact keys must match, `n`/`batch` are log distance, weighted per key (`n:log:3`: potrf work ~ n^3 x batch). A device with no table borrows the nearest one and warns once; the CPU never borrows; nothing runnable falls to the op's last resort. Tables are generated, not hand-edited: `scripts/sweep_to_table.py` (`--check` is run by hand as part of the per-op acceptance gate; CI does not run it). A table headed `source=transcribed:<sha>` (entries `<spelling> -`) is an old router's order evaluated per grid cell by a transcriber (deleted after use; provenance in `tuned/README.md`), **untimed**: it reproduces the deleted windows, it is not a measurement. `BATCHLAS_TUNED_DIR=<dir>` overrides them without a rebuild; `BATCHLAS_SELECT_TRACE=1` prints every choice and the runner-up. Per-op choices: trsm `cta`, `sg_left`, `blocked`, `vendor` (a `cta` or `sg_left` pin above order 32 throws); posv `tiny`/`cta`/`blocked`, no vendor family, its children select for themselves; gemm `direct`, `tiled`, `small`, `reg:m=..:n=..:k=..:u=..` (float), `wide:m=..:n=..:k=..`, `vendor`, with the aligned/predicated leg derived inside the launcher, never a field (docs/perf/gemm.md#choices-flat-selection-p34); the level-3 four (docs/design/flat-kernel-selection.md §12 "Level-3 four"), all `NoFields`, native families CUDA GPU only, homogeneous batch, batch <= 65535: syrk `gram` (n <= 128; float and double), `triangular` (float), `vendor`; syr2k `triangular` (float; real ConjTrans stays on the vendor), `vendor`; symm `expand` (mirrored expansion + public gemm, float and double, needs `expansion_fits`), `vendor`; trmm `triangular` (Side::Left), `expand` (expansion + public gemm, either side), `vendor` (the `cublas?trmm` loop only: the old `vendor` pin meant expand+gemm, that is now `expand`). Every level-3 vendor family also refuses a heterogeneous batch on every backend (the loops run each item at the top-level extents), so a heterogeneous call has no route. symm and syrk key on `form` = sq|tall|wide, syrk also on `trans`. `cublasdx` and the old level-3 pin-word list are gone: every `cublasdx` pin throws as an unknown family.
	•	Env override: `BATCHLAS_<OP>_ROUTE` (`settings().routing.route("<op>")`, 19 ops) takes `auto`, `native`, `vendor` or a spelling (`lpanel:panel=8`, `lpanel:8`), case-folded and trimmed. **A bad pin THROWS** (unparsable, not a compiled candidate, or cannot run the shape); only the class words `native` and `vendor` with no runnable choice of their class fall back to Auto, with a warning (so `vendor` in a vendor-free build means Auto). There are no aliases: the old `native:cta`, `vendor:auto`, `batchlas_cta`, gemm kernel names (`tiled16`, `128x128x8`) all throw, and `BATCHLAS_<OP>_VARIANT` / `_PROVIDER` / `BATCHLAS_GEMM_SYCL_KERNEL` are not read. Coverage `chosen_algo` is the spelling.
	•	hemm, herk and her2k are the ops still choosing by hand-written rule (`cublas.cc`, vendor builds only: expand/fold into the public gemm vs the per-item vendor loop, `BATCHLAS_EXPAND_ROUTE=expand|loop` pins it; herk's gram opt-in is `BATCHLAS_SYRK_ROUTE=gram`). They have no `BATCHLAS_<OP>_ROUTE` and throw `NoRouteError` vendor-free. Their migration is a listed follow-up (docs/design/flat-kernel-selection.md §11).
	•	cuBLASDx/MathDx is deleted (the code, the MathDx probe, `BATCHLAS_ENABLE_CUBLASDX`, `BATCHLAS_HAS_CUBLASDX`, `BATCHLAS_GEMM_CUBLASDX_KERNEL`). It never ran: MathDx was absent on both boxes.
	•	Public entry points live in `src/ops/<op>/<op>.cc` (`src/ops/level3/level3.cc` for hemm/herk/her2k). The vendor-availability constants and `throw_no_vendor_route` are in `src/select/vendor.hh`; `batchlas::NoRouteError` is in `<batchlas/no_route.hh>`. `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` builds a vendor-free library, and **that is the only valid vendor-free measurement**. In a vendor-free build an op with nothing runnable throws `NoRouteError` and records a coverage `miss` row.
	•	**Linked ≠ reached.** A symbol in `nm`, a `linked` coverage row, or a green test count does not show that a kernel runs. Set `BATCHLAS_COVERAGE_OUT` and read the `reached` rows. `scripts/route_diff.sh` captures and diffs them, and `coverage_merge.sh` merges the per-pid shards. Verify a routing change by *diffing the chosen route* across the input space, not by timing it. For a migrated op (gemm included) `chosen_algo` is the full spelling (`reg:m=128:n=128:k=8:u=1`), so route_diff sees gemm tile changes; it does not see a derived leg (gemm's aligned vs predicated) or a non-migrated op's internal kernel pick. Use `BATCHLAS_KERNEL_TRACE=1` for those (never while timing, it inflates ~60%). `tools/tune --gate` against a pre-P3.4 parent reads its gemm coverage as `native:register_tiled`, which now means "best native": gate gemm from a kernel-trace old-choice CSV.
	•	Sub-ops count too: a "native" syev, geqrf or getrf can still call cuBLAS gemm underneath. Since P3.4 every library gemm call, including the blocked drivers' default seams, the hemm/herk/her2k expansions in `cublas.cc` and symm's and trmm's `expand` families, goes through the public `gemm` and its table (`gemm_custom` and the `gemm_use_sycl_custom` re-route are deleted).
	•	Highest-yield perf pattern: **a finished candidate that no table row ranks first.** Before writing a kernel, list the op's candidates and check whether each one can ever be chosen: is it timed in the tables, and does `can_run` ever admit it? Look for hard `return false` and for terms that name a different family's type list.
	•	**Read the predicate, not the comment.** Comments describing routing windows drift. docs/perf/<op>.md quotes each `can_run` term and window with `file:line` and gives the bracketing evidence. About 300 source comments cite `docs/perf/*.md#anchor`, so renaming a heading breaks them; `check_evidence_anchors.py` catches this.
	•	Kernel-selection bug to look for: a kernel's fast-*leg* predicate (one the dispatcher re-evaluates to choose aligned vs predicated) also used as a *routing* gate, so failing it sends the call to a different, slower kernel. Found twice in `select_kernel_variant` (1.7x and 3–4x; deleted in P3.4). Under flat selection a leg is derived in the launcher and must never become a family field, a key or a `can_run` term (layout `packed|strided` is a table key, not a gate).
	•	Guarding on a device family (`BATCHLAS_HAS_CUDA_BACKEND`) where the code needs a *library* is a recurring bug, and only a vendor-free build exposes it.

**API traps**
	•	The variadic queue-dispatch overload (`BATCHLAS_DISPATCH_ON_QUEUE`) must be `requires`-constrained. Even constrained, it beats option-struct overloads for non-const lvalue and prvalue arguments, so per-call checks that live only in `options.hh` are unreachable on the ordinary spelling. Test checks through the ordinary spelling.
	•	Write `PotrfOptions{}`, never a bare `{}`: `{}` selects the positional `Uplo{}` = Upper.
	•	Workspace sizing: `BumpAllocator` checks the alignment-rounded size but advances by the raw size. An exactly simulated size is therefore too small. Size with `BumpAllocator::measuring()` + `required_bytes()`. A `*_buffer_size` and its run path must branch identically. Sizing mode hands out unbacked pointers, so never test `ws.data() != nullptr` to mean "the caller passed a workspace"; use separate overloads.
	•	The workspace arena belongs to the `Queue`, so a per-call `Queue` defeats pooling. Leak long-lived SYCL objects that are held in statics, because static destruction after the runtime is gone hangs.
	•	`linalg::` value-returning wrappers free their scratch while kernels are still enqueued. `linalg::qr` is deliberately absent because of an unexplained cross-Queue wrong-answer defect.
	•	Pivots are packed 1-based int32 everywhere. `cusolverDnXgetrs` wants int64.
	•	Known located-but-unfixed defects are listed in `docs/design/known-defects.md`.

⸻

10. Measurement Rules (benchmarks lie silently here)

	•	**Compare algorithms only at saturation.** Below saturation you are measuring launch, allocation and clock-ramp overhead. Saturated means that doubling the batch buys <10% per item. **Benchmark and tune at large batch (≥128); batch=1 does not matter to this project.** Still *profile* across batch sizes, because batch-only-parallel kernels (one work-group per matrix) starve at small batch and hide there. If a kernel's work-group count equals the batch size, that is the bug.
	•	**Run one measuring process on the box at a time**, whatever the device pin. Two GPUs on one NUMA node and one UVM driver produced 5.5x errors with stable rel_sd while `nvidia-smi` showed no foreign process on either card. A display attached to a GPU also slows L2-resident vendor cells, so measure on a headless GPU. Use `benchmarks/gpu_guard.sh`, plus a `flock` against your own duplicate jobs.
	•	Warm up: idle clocks are ~210 MHz. Use `--warmup=3 --min_iters=8`. The first *process* to run a kernel pays JIT (the cache in `~/.cache/batchlas_sycl` is shared), so do a throwaway pass over every cell first.
	•	Alternate the A/B order between reps and discard the first pass. Per-rep CV is ~3%, so an effect under 5% needs 14–16 reps. Treat 6 reps as a screen, not a verdict.
	•	The Google Benchmark `--name` filter is a *substring* match. Filter on the CSV name column. That column is quoted and contains a comma, so `awk -F,` column offsets shift.
	•	Admit rows on cross-pass reproduction, not on rel_sd alone. A noise filter that deletes a reproducible loss fabricates a window. Bracket every window edge with a measured non-winner.
	•	Before quoting "vs vendor": check which route actually ran, check what "the vendor" is (cuBLAS complex trsm and batched orgqr are per-item loops or our own substitute kernels; cuSOLVER `syevjBatched` beats the `XsyevBatched` we call at n≤32), and re-measure the vendor, because vendor versions move.
	•	Rebuild every bench binary after a table or `can_run` change. Harnesses resolve the *printed* route in their own TU, so relinking only the `.so` leaves the route column lying.
	•	Check that an argument is used, not just echoed: time should scale with it. Harnesses written before the code (design studies) are not benchmarks of the shipped code.
	•	Sanity anchors (RTX 4090): real FP32 SGEMM ≈ 47–54 TFLOP/s (≈80 means TF32), FP64 ceiling ≈ 1.44 TFLOP/s (1/64 rate), DRAM ≈ 1008 GB/s peak, ~950 achievable. cuBLAS gemv is already at the DRAM roof, so do not try to beat it.
	•	A cost table that is non-monotonic in n (a larger n running faster) is a routing bug.
	•	nsys sees cuBLAS kernels; `BATCHLAS_KERNEL_TRACE` does not (it hid 35% of a cfloat syev). Also useful: `BATCHLAS_KERNEL_TRACE_PATH` (Chrome trace), `BATCHLAS_GESVD_PROFILE=1`, `BATCHLAS_TUNE_*` overrides (A/B any tuning constant without a rebuild), `BATCHLAS_BENCH_LD_PAD[_A/_B/_C]` and `BATCHLAS_BENCH_BETA` in the gemm benchmarks. Always confirm a GEMM at beta=1, and with a strided `ld`: panel updates always carry their parent's `ld`. Pin one gemm kernel with `BATCHLAS_GEMM_ROUTE=<spelling>` (`reg:m=128:n=128:k=8:u=1`, `wide:m=64:n=64:k=16`, `small`, `tiled`, `direct`; the old kernel names such as `128x128x8` and `tiled16` throw): it throws if that tile has no instantiation for the call's transpose form. `BATCHLAS_GEMM_SYCL_KERNEL` and `BATCHLAS_GEMM_EXPERIMENTAL` are not read. Compare gemm kernels with `tools/tune/batchlas_tune` (`gemm_spec.cc`: beta=1, complex alpha, strided cells padded by `--ld-pad`, every arm verified).
	•	benchviz (`python3 benchmarks/benchviz`) measures whatever is in *its checkout's* `build/`, not HEAD. Run `benchviz info` first. Its batchlas arm pins the native walk, not Auto. Figures must follow the user's style (`benchmarks/benchviz/style.py`, from `plotting/stylesheet.py`: CM usetex, full box, dotted lines with large markers, 2σ band, viridis maps), not generic defaults.
	•	Raw result grids go in `benchmarks/results/` (Git LFS; run `git lfs install` on every machine, or the commit stores raw files and CI's `check_lfs_pointers.py` fails). Older evidence is at tag `perf-evidence/vendor-independence` (`git show <tag>:experiments/...`). docs/perf is the distilled layer.
	•	Retuning takes ~12 min (`evaluation/tuning/`). The CMake `batchlas_tuning_header` target is a no-op, because the checked-in `include/batchlas/tuning_params.hh` wins; port constants by hand. **Tuning was float-only.** Constants measured on float are applied to double and complex, so when another type looks bad, sweep that type first. Check whether a knob has a second consumer (aliasing) or is bypassed on the path (shadowing) before trusting "no change".

⸻

11. GPU Kernel Design Facts (sm_89; mostly general)

	•	Register gate: registers are split across four sub-partitions of 16,384. The bound is `ceil(warps/4)*32*regs <= 16384`, **not** `regs*wg <= 65536`. The launch is accepted by SYCL and rejected by the driver as an uncaught abort. Register counts from a standalone `-Xcuda-ptxas -v` differ from the in-library kernel, so prefer a measured launch ceiling.
	•	Design against occupancy, not spills: 8x8 double tiles do not spill, but `complex<double>` × 512 threads exceeds the register file. Shrink the thread tile as the scalar widens. Never pass an accumulator element by `T&` out-parameter, because that spills the whole array (43%); return by value.
	•	`std::complex` `*` emits an isnan branch plus `__mulsc3` (Annex G, since no `-ffast-math`). Write out the real arithmetic *in the hot loop only* (1.2–1.3x). Converting every site cost occupancy and was slower. Do not add `-ffast-math`.
	•	The 48 KB SLM launch hole: an attribute that is sticky per CUfunction means test launch *order* can hide it. Put the discriminating launch first.
	•	Launch bounds (`intel::min_work_groups_per_cu`) only apply on a **functor's** `operator()`, not on a lambda. They are worth 1.1–1.9x on tiny kernels, but a cap that is too tight is 2.3x slower, and a change moves routing windows.
	•	SLM staged by a lane-strided loop and read by a different index needs a barrier. NVIDIA hides the race at wg=32.
	•	Column-major coalescing: assign one work-item per *output* element. With that mapping NoTrans gemv is coalesced and Trans needs the reduction. Row interchanges (laswp) should be an index gather in SLM, not a per-column walk (7.9x fewer sectors).
	•	GEMM: 128x128x8 tiles with an 8x8 thread tile (64 accumulators), aligned (not +1) shared strides, and B staged [k][n] give cuBLAS parity for float NN. The epilogue must vary the m index fastest. Internal GEMM demand is *panel updates* (large m,n; k = blocking factor 8–136; ~60% transposed), so a `min_dim` gate never fires on it. The native GEMM is ~2x off on strided `ld` because of exposed B-load latency; double-buffering and packing B were both measured as no gain.
	•	DPC++ `chunked_partition` masks cost ~4–5 instructions per collective. Full-warp or maskless rewrites lost in real kernels even where microbenchmarks won. Judge partition-primitive changes (`src/extensions/sg_partition/`) by real-kernel A/B (steqr/syev_cta benchmarks, n=5..16, batch 16384), never by microbenchmark.
	•	CTA kernels are capped at n ≤ 32 (the sub-group width). `STEDC_RECURSION_THRESHOLD = 32` is that invariant, not a tuning constant: a leaf of 33+ falls from `steqr_cta` to `steqr_wg` (14x cliff).
	•	Traffic models that count bytes miss serial recurrences and lost parallelism. Before trusting one, check the critical path and the work-item count.

**Measured dead ends — do not re-propose without a new idea:** lifting the CTA eigensolver above n=32 (85–211x slower), block Jacobi as a speed path (2–11x slower; accuracy-only opt-in), a cooperative TRSM diagonal solve, a packed sub-group getrf with runtime n in SLM (register-resident designs are *not* refuted), a faster native gemv, maskless steqr at P<32, the Givens chase (the Householder chase is 5.6x cheaper), a GEMM kernel fix for strided `ld` (the fix was routing). The latrd panel symv is L1-over-fetch-bound (12–16x), not DRAM-bound; a MAGMA-style single-read symv is the open 2.7x.

⸻

12. Repository Conventions and Git

	•	**Comment density ≤18% per file** is enforced by `.github/ci/check_comment_density.py`. A code line with a trailing comment counts as code. The repo-wide average hides per-file violations, so check per file. Lab-notebook material (grids, geomeans, rejected hypotheses) goes in `docs/perf/`, with an `evidence: docs/perf/<op>.md#anchor` pointer in the code. A comment is for invariants, "looks wrong but is deliberate" notes, and traps. Waivers live in `.github/ci/comment_density_waivers.txt`.
	•	A `main` ref from session start is a snapshot. `git fetch origin` before claiming something is absent from a branch.
	•	For stacked PRs, `merged` can mean merged into the previous stack branch, not into main. Read `base.ref`.
	•	`/code-review ultra` rejects diffs over 500 files or 8,000 lines. Split along work-package commit boundaries: a by-op split does not compile, because the dispatch facade includes every op's native header.
	•	`nm -DC ... | awk '{print $3}'` truncates demangled names at the first space, and `nm -C` cannot demangle concept-constrained templates. Diff full mangled names (`scripts/facade_symbol_check.sh` does this).
	•	In worktree-isolated agent sessions, the Bash guard refuses heredocs, `for` loops with computed arguments, and awk programs. Write scripts to `$CLAUDE_JOB_DIR/tmp` and run them. Wait on a PID with `kill -0`, not `pgrep -f` (it matches itself).

⸻

13. Primary-Machine Facts (the RTX 4090 box; re-check on any other machine)

	•	2x RTX 4090 (128 SMs, 72 MB L2, 24 GB, FP64 1/64), Xeon w5-2445, Ubuntu 22.04, DPC++ at `/opt/dpcpp-cuda`, CUDA 13.2, ROCm 6.2.4 headers (no AMD GPU), MAGMA 2.10 Release/sm_89 at `~/magma/build-rel89`. Device 0 drives the display, so measure on device 1.
	•	The box is shared with another user who runs GPU sweeps. Check `nvidia-smi` before timing anything, and coordinate before a reboot. `perf` and `gdb -p` do not work here (perf_event_paranoid, ptrace), so use the UR trace.
	•	The system OpenBLAS 0.3.20 picks the broken Cooperlake dgemm (see Known Pitfalls).
	•	Unattended-upgrades is not blacklisted for `nvidia-*`, so driver mismatches recur.

On a **new machine**, before trusting any result: `sycl-ls` shows your GPU; configure prints the right SYCL target; `nvidia-smi` (or equivalent) reports the architecture you think it does; record that architecture's FP32/FP64/DRAM ceilings as your sanity anchors. **Every routing window and tuning constant in the tree was measured on sm_89.** On a different GPU, especially one with 1:2 FP64 where double GEMM verdicts invert, treat the tuned tables and hemm/herk/her2k's hand-written windows as hypotheses. Re-measure at saturation before quoting any ratio, and record the machine alongside new data in `benchmarks/results/`.

⸻

License

SPDX-License-Identifier: MIT