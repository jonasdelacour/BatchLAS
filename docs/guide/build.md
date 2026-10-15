# Building, testing and benchmarking {#guide_build}

> **Status:** current · moved from the project README on 2026-10-09.

Requirements, tested platforms, CMake presets and options, the test and benchmark builds, the
Python package and the install traps. Calling the library and linking an installed package are in
[the C++ API](../cpp-api.md) ("Building and installing BatchLAS", "Building against
BatchLAS").

## Requirements

- CMake 3.17+ (`FindCUDAToolkit` is 3.17, ctest `RESOURCE_GROUPS` 3.16; 3.21+ for `cmake --preset`,
  `CMakePresets.json` is schema version 3).
- A C++20 compiler with SYCL support: a DPC++/Clang-family compiler, clang 16 or newer.
- Optional: CUDA Toolkit (NVIDIA), ROCm (AMD), LAPACKE and CBLAS (netlib host backend), Python 3 +
  pybind11 + NumPy + SciPy (Python bindings).

> **Warning:** the SYCL compiler must have a backend for your GPU vendor. Stock Intel oneAPI
> `icpx` has no CUDA adapter: on an NVIDIA machine it configures cleanly, prints
> `-- Using SYCL targets: spir64_x86_64`, and builds a **CPU-only** library without warning. Use the
> Codeplay *oneAPI for NVIDIA GPUs* plugin on top of oneAPI, or a self-built `intel/llvm`
> configured with `--cuda`. `sycl-ls` must list a `[cuda:gpu]` entry.

Package suggestions and oneAPI notes: @ref dev_agent_guide.

## Tested platforms

Only the Primary configuration is exercised regularly; everything else is untested rather than
known-good. `.github/workflows/ci.yml` runs static checks on GitHub-hosted runners (no SYCL
compiler there, so nothing is configured or compiled) and a self-hosted GPU job on the Primary
machine: configure with CUDA required, build all libraries and tests, `ctest`, install, check the
export. A nightly job on that runner adds slow suites, the consumer packaging test and a
vendor-free build.

| | Compiler | CUDA | GPU / arch | OS | Status |
| --- | --- | --- | --- | --- | --- |
| Primary | `intel/llvm` DPC++, clang 22.0.0git, built with `--cuda` (installed at `/opt/dpcpp-cuda`) | 13.2 | NVIDIA RTX 4090, `sm_89` | Ubuntu 22.04 | Library, tests and benchmarks built and run here daily |
| Second GPU box | same toolchain (threadripper02) | 13.2 | NVIDIA RTX PRO 6000, `sm_120` | Linux | Built and tested; `sm_120` tuning tables measured here |
| CI — static checks | none (no toolchain) | — | none | `ubuntu-latest` | List files, source-level export and public headers only; nothing configured, compiled or run |
| CI — GPU gate | the Primary row's toolchain, on a self-hosted runner on that machine | 13.2 | NVIDIA RTX 4090, `sm_89` | Ubuntu 22.04 | Build + `ctest -LE slow` + install on every push and non-fork PR; full `ctest`, packaging and a vendor-free build nightly |
| CPU only | Intel oneAPI `icpx` 2025.x | — | none (`spir64_x86_64` / `native_cpu`) | Ubuntu 22.04 | Configures and builds; **no NVIDIA support** (see the warning above). Not built by CI |
| AMD / ROCm | — | — | — | — | Code paths exist; not built or run |
| Apple GPU (macOS) | AdaptiveCpp `develop` at `9ec394ef` plus local patches 0001-0013 ([below](#macos-apple-gpu-through-adaptivecpp-metal)), Homebrew `llvm@21` 21.1.8 / `lld@21`, Command Line Tools only (no Xcode) | — | Apple M5 Max (`applegpu_g17s`), Metal | macOS 27.0.1 | Library and full test suite built and run (`acpp-metal-tests`); float and complex<float> on the GPU, double on the host through Homebrew OpenBLAS 0.3.34. Not built by CI |
| Windows | — | — | — | — | Untested |

Other NVIDIA architectures should work: the build detects the local GPU, and
`-DBATCHLAS_NVIDIA_ARCH=sm_XX` overrides it.

`main` has known test failures, so the GPU job diffs each run against the ledger
`tests/known-failures.txt` instead of using `ctest`'s exit code. See [the CI page](../ci.md) for
coverage, self-hosted runner setup and retiring a ledger entry.

## Presets

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8   # the presets set no job count; wider builds have OOM-killed
cmake --preset dev
cmake --build --preset dev
```

| preset | builds |
| --- | --- |
| `dev` | default `RelWithDebInfo` library |
| `dev-tests` | library plus the full test suite |
| `fast-dev` | library plus the smoke-test subset |
| `benchmarks` | benchmarks and the table tuner (`tools/tune`) |
| `cuda` | `dev-tests` with `BATCHLAS_ENABLE_CUDA=AUTO` (not `ON`: without a `[cuda:gpu]` it still configures, CPU-only) and `BATCHLAS_ALLOW_UNSAFE_ENV=OFF`; the pre-push gate |
| `dev-gpu` / `dev-gpu-tests` | drop the `native_cpu` SYCL target (`BATCHLAS_CPU_TARGET=none`) |
| `release` | `Release`, no tests or benchmarks, one shared library |
| `acpp` / `acpp-tests` | `dev` / `dev-tests` built with the AdaptiveCpp 25.10 driver (`--acpp-targets=generic`) into `build-acpp` / `build-acpp-tests` ([SYCL implementations](../design/sycl-implementations.md)) |
| `acpp-metal` / `acpp-metal-tests` | macOS only: Apple GPU through AdaptiveCpp Metal into `build-acpp-metal` / `build-acpp-metal-tests` ([macOS](#macos-apple-gpu-through-adaptivecpp-metal)); the test preset runs `ctest` serially |

`BATCHLAS_ALLOW_UNSAFE_ENV` is `ON` in every preset that inherits `dev` (`dev`, `dev-tests`,
`fast-dev`, `benchmarks`, `dev-gpu*`, `acpp*`), which exist to measure and debug, and `OFF` in `cuda`
and `release`, which run the arm a release build runs.

`dev-gpu` is 13-25% off a cold build of the default target on one 4-GPU box, but it is **GPU
coverage only**: about half of every typed test suite is not instantiated and `ctest` still reports
green. Use `dev-tests` or `cuda` for the pre-push gate, plus a vendor-free tree
(`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) at that gate only.

## CMake options

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBATCHLAS_BUILD_TESTS=ON \
  -DBATCHLAS_BUILD_BENCHMARKS=OFF \
  -DBATCHLAS_BUILD_PYTHON=OFF
cmake --build build -j8
```

| option | meaning |
| --- | --- |
| `BATCHLAS_BUILD_TESTS` | build unit tests |
| `BATCHLAS_BUILD_BENCHMARKS` | build benchmark executables and `batchlas_tune` |
| `BATCHLAS_BUILD_PYTHON` | build the Python package |
| `BATCHLAS_ENABLE_CUDA` | `AUTO` (default; enable cuBLAS/cuSOLVER when the SYCL runtime exposes a CUDA device), `ON` (require it; configure fails without an NVIDIA target, see "Device and toolkit detection") or `OFF`. Boolean spellings are accepted (`1`/`TRUE`/`YES` mean `ON`, `0`/`FALSE`/`NO` mean `OFF`); an existing cache entry keeps its value |
| `BATCHLAS_ENABLE_VENDOR_BLAS` | the one switch over every vendor library (cuBLAS/cuSOLVER/cuSPARSE, rocBLAS/rocSOLVER/rocSPARSE, netlib LAPACKE/CBLAS), re-read on every configure; `OFF` builds the vendor-free library ([vendor-free status](../design/vendor-free-status.md)). The three CUDA libraries are all-or-nothing: a toolkit missing one fails configure |
| `BATCHLAS_MONOLITHIC_LIBRARY` | one `libbatchlas.so` instead of the per-component libraries (default `OFF`; the `release` preset turns it on) |
| `BATCHLAS_ALLOW_UNSAFE_ENV` | honour the `BATCHLAS_*` overrides that disable a safety check (`batchlas::Settings::unsafe`; default `OFF`, see "Presets") |
| `BATCHLAS_DISABLE_CUDA_FTZ` | no flush-to-zero in CUDA device code (default `ON`) |
| `BATCHLAS_KEEP_CUDA_INTERMEDIATES` | keep CUDA/SYCL device intermediates for PTX/SASS analysis (default `OFF`) |
| `BATCHLAS_SANITIZER_FRIENDLY_DEBUG` | unwind-friendly debug flags for sanitizer runs (default `OFF`) |
| `BATCHLAS_OPENBLAS_CORETYPE` | `OPENBLAS_CORETYPE` that ctest sets to avoid a broken OpenBLAS kernel; empty means none (set by the health check) |
| `BATCHLAS_BUILD_DOCS` | add the `batchlas_docs` target (`scripts/build_docs.sh`) |
| `BATCHLAS_BUILD_DEVICE_BLAS_BENCHMARKS` | build the `device_blas_*` benchmarks (default `OFF`) |
| `BATCHLAS_STRIP_RELWITHDEBINFO_G` | drop the toolchain's `-g` from `CMAKE_CXX_FLAGS_RELWITHDEBINFO` (default `ON`; BatchLAS adds `-gline-tables-only`; full DWARF in device images has failed CUDA JIT program builds) |
| `BATCHLAS_ENABLE_ROCM` | enable ROCm backend support even if no AMD GPU is detected |
| `BATCHLAS_CPU_TARGET` | SYCL CPU target: `auto`, `native_cpu`, `spir64_x86_64`, `none` |
| `BATCHLAS_TEST_TARGET_SET` | `all` or `smoke` |
| `BATCHLAS_AMD_ARCH` | override ROCm target architecture |
| `BATCHLAS_NVIDIA_ARCH` | empty (default): the detected NVIDIA GPUs; `sm_XX`: build that architecture instead (cross builds, no GPU needed). The old default `sm_50` is read as empty |
| `BATCHLAS_USE_CCACHE` | cache compilations with ccache when on `PATH` or in `~/.local/bin` (default `ON`; your own `CMAKE_CXX_COMPILER_LAUNCHER` wins) |
| `BATCHLAS_CCACHE_SHARE_ACROSS_TREES` | let sibling checkouts and worktrees share cache entries (default `ON`; turn off for source-level debugging) |
| `BATCHLAS_CCACHE_MAXSIZE` | ccache size limit (default `20G`) |
| `BATCHLAS_CCACHE_BASEDIR` | ccache base directory (default: deepest common parent of source and build directories, or `$HOME` if that is `/`) |
| `BATCHLAS_TEST_GPUS` | GPUs written to `<build>/ctest_resources.json` for `scripts/ctest_gpus.sh`: `auto` (`nvidia-smi --list-gpus`), a count, or `0` to disable. Needs a `[cuda:gpu]` in `sycl-ls`, else a count is a configure error |
| `BATCHLAS_TEST_GPU_SLOTS` | concurrent GPU tests per device under `scripts/ctest_gpus.sh` |
| `BATCHLAS_SYCL_LINK_JOBS` | parallelism of the DPC++ SYCL device link (default `4`, `1` disables) |
| `BATCHLAS_SYCL_IMPL` | `AUTO` (default; `ACPP` when `CMAKE_CXX_COMPILER` is the `acpp` driver), `DPCPP` or `ACPP`. Only the matching compiler is accepted; `ACPP` without an `acpp` compiler takes `acpp` from `PATH`, else `/opt/adaptivecpp/bin`, also replacing a cached compiler (`ccmake`); CMake then resets the cache and the value reads `AUTO` |
| `BATCHLAS_ACPP_TARGETS` | `--acpp-targets` of an AdaptiveCpp build (default `generic`) |
| `BATCHLAS_ENABLE_METAL` | AdaptiveCpp only: `AUTO` (default; dispatch an Apple GPU as `Backend::SYCL` when `acpp-info` lists a Metal device), `ON` (require it; configure fails without one) or `OFF` (needs the GPU hidden with `ACPP_VISIBILITY_MASK=omp`; configure fails while `acpp-info` lists it) |
| `BATCHLAS_DEVICE_FP64` | `AUTO` (default), `ON` or `OFF`: instantiate double and complex<double> for `Backend::SYCL`. `AUTO` resolves to `OFF` when Metal is enabled, and `ON` with Metal is a configure error. CUDA, ROCm and the host always keep all four types |
| `BATCHLAS_OPENBLAS_ROOT` | OpenBLAS prefix whose `libopenblas` provides LAPACKE and CBLAS. On macOS it is probed at `/opt/homebrew/opt/openblas` and `/usr/local/opt/openblas`; set it, the headers go on the library's include path. Empty on Linux by default, which keeps the system LAPACKE |
| `BATCHLAS_MONOLITHIC_LIBRARY` | one shared library instead of the component libraries (the `release` shape). Required on macOS: ld64 rejects the components' cyclic undefined symbols |
| `BATCHLAS_HIDDEN_VISIBILITY` | `-fvisibility=hidden` on the monolithic library (default `ON`; only with `BATCHLAS_MONOLITHIC_LIBRARY`). `OFF` lets the in-tree tests link against it, since they reach about 1,000 unexported internal symbols |
| `BATCHLAS_ACPP_SLM_OPTIN` | AdaptiveCpp + CUDA: a launch interposer that opts kernels into more than 48 KiB of local memory (default `OFF`; [SYCL implementations](../design/sycl-implementations.md#sycl-impl-slm-budget)) |

### Device and toolkit detection {#guide_build_detection}

Configure needs only the compiler (`-DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++`, `icpx`, or
`-DBATCHLAS_SYCL_IMPL=ACPP`); no `setvars.sh`, `LD_LIBRARY_PATH`, `CUDA_PATH` or `nvcc` on `PATH`.

- DPC++ devices come from the `sycl-ls` next to the compiler, run with the compiler's `lib/` on the
  loader path. If it lists no `[cuda:gpu]` but the toolchain ships `libur_adapter_cuda.so`,
  `nvidia-smi` supplies the GPUs. AdaptiveCpp uses `acpp-info`, then `nvidia-smi`.
- The CUDA toolkit is what `FindCUDAToolkit` finds (`CUDAToolkit_ROOT`, `$CUDA_PATH`, `nvcc` on
  `PATH`, `/usr/local/cuda`). Failing that, AdaptiveCpp's own `default-cuda-path`
  (`etc/AdaptiveCpp/acpp-cuda.json`), then the newest NVIDIA HPC SDK toolkit
  (`/opt/nvidia/hpc_sdk/*/*/cuda/<version>`), cached as `CUDAToolkit_ROOT`.
- Built and installed binaries carry an RPATH to the DPC++ runtime (`<compiler>/../lib`), the
  vendor libraries they link and `$ORIGIN`, so they run without `LD_LIBRARY_PATH`.
- A self-built `intel/llvm --cuda` has no CPU adapter: outside a oneAPI environment there is no CPU
  device, and the tree is GPU-only (`No SYCL CPU device detected`).

## macOS: Apple GPU through AdaptiveCpp Metal

The Apple GPU runs through AdaptiveCpp's Metal backend. Kernels are compiled once to generic SSCP IR
and JIT-translated to Metal Shading Language at their first launch. Design, limits and patches:
[AdaptiveCpp Metal](../design/sycl-implementations.md#sycl-impl-metal).

| | On an Apple GPU `Queue` | On a CPU `Queue` |
| --- | --- | --- |
| Backend | `Backend::SYCL`: native kernels only, no vendor GPU library (no MPS) | `Backend::NETLIB`: Homebrew OpenBLAS (LAPACKE, CBLAS) plus native kernels on the AdaptiveCpp OpenMP device |
| Types | float and complex<float>. A double or complex<double> call throws `batchlas::unsupported` before any launch | all four |
| Not available | hemm, herk, her2k (`NoRouteError`: they have no native family yet); any shape whose every family needs more local memory than Apple's 32 KiB, or than a `can_run` occupancy slice of it (cfloat Upper potrf/posv at n=24 to about 50) | — |

`batchlas::verify` and the queue-less `Matrix` helpers run their double arithmetic on the CPU device
([verification](../design/verification.md#verification-no-fp64-device)).

**Toolchain.** Needs the Command Line Tools and Homebrew; Xcode is not needed. The script installs
`cmake llvm@21 lld@21 libomp boost openblas python` with brew, fetches metal-cpp, checks out AdaptiveCpp
`develop` at `9ec394ef` on the local branch `batchlas-metal`, applies
`scripts/macos/adaptivecpp-patches/*.patch` with `git am`, then builds and installs:

```bash
scripts/macos/build_adaptivecpp_metal.sh   # -> ~/opt/adaptivecpp
~/opt/adaptivecpp/bin/acpp-info -l         # must list the OpenMP host device and the Apple GPU (Metal)
rm -rf ~/.acpp/apps                        # after every reinstall: the JIT cache outlives the toolchain
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `ACPP_SRC` | `~/opt/src/AdaptiveCpp` | source checkout |
| `ACPP_PREFIX` | `~/opt/adaptivecpp` | install prefix; the presets read `$HOME/opt/adaptivecpp/bin/acpp` |
| `METAL_CPP_DIR` | `~/opt/src/metal-cpp` | metal-cpp headers (downloaded when missing) |
| `ACPP_BASE` / `ACPP_BRANCH` | `9ec394ef` / `batchlas-metal` | upstream commit and the local patch branch |
| `ACPP_REAPPLY=1` | off | recreate the branch from the patch files. An existing branch is otherwise reused, and the script stops if its commit count differs from the number of patches |
| `SKIP_BREW=1` | off | skip `brew install` |

The script configures AdaptiveCpp as its upstream macOS CI does (`ACPP_COMPILER_FEATURE_PROFILE=full`,
Metal on, CUDA/ROCm/OpenCL/Level Zero off), with Homebrew `opt/` paths so a patch-level `brew
upgrade` does not leave stale `Cellar/` paths in `etc/AdaptiveCpp/acpp-core.json`; it warns if any
remain. It then pins the `acpp`, `syclcc` and `syclcc-clang` shebangs to Homebrew's `python3`. The
patches are local only and not sent upstream; what each fixes, with its evidence, is in
`scripts/macos/adaptivecpp-patches/README.md`.

**Build and test.**

```bash
cmake --preset acpp-metal-tests
cmake --build build-acpp-metal-tests --target gemm_tests -j8   # an explicit cap: every SYCL TU takes ~300 MB under acpp
ctest --preset acpp-metal-tests -R '^gemm_tests$'
```

The presets set `ACPP_VISIBILITY_MASK=omp;metal` (configure and ctest), `BATCHLAS_ENABLE_METAL=ON`,
`BATCHLAS_DEVICE_FP64=OFF`, `BATCHLAS_ENABLE_VENDOR_BLAS=ON` (on macOS the only vendor library is
netlib: LAPACKE and CBLAS from `/opt/homebrew/opt/openblas` for the CPU `Queue`; the Apple GPU has
no vendor family either way), `BATCHLAS_MONOLITHIC_LIBRARY=ON` with `BATCHLAS_HIDDEN_VISIBILITY=OFF`,
and `BATCHLAS_TEST_GPUS=0`. `acpp-metal-tests` also sets `CMAKE_DISABLE_FIND_PACKAGE_GTest=ON`, so
GoogleTest is fetched and built by `acpp` against Homebrew LLVM's libc++, never a system GoogleTest
built by another compiler. Configure prints
`Native SYCL GPU backend (Metal): TRUE; device fp64: OFF`.

A full serial `ctest` takes about an hour; the first launch of each kernel pays the MSL translation
and the Metal compile. The failure ledger of this tree is `tests/known-failures-acpp-metal.txt`,
which `.github/ci/compare_failures.py --build-dir build-acpp-metal-tests` picks from the cached
`BATCHLAS_GPU_FAMILY_RESOLVED=METAL`. The macOS traps (adaptivity level, the JIT cache, `DYLD_` paths,
the `CPATH` leak) are in @ref dev_agent_guide, "Known Pitfalls".

> **Warning:** every tuning table and constant was measured on NVIDIA GPUs. An Apple GPU has no
> table of its own: it is keyed `apple_m5_max`, borrows the nearest table and warns once per op.
> Its routes are hypotheses until the device is tuned.

## Tests

```bash
cmake --preset dev-tests
cmake --build --preset dev-tests
ctest --test-dir build/presets/dev-tests --output-on-failure
```

`fast-dev` builds only the smoke subset (`util_span_tests`, `util_vector_tests`, `matrix_tests`,
`mempool_tests`, `backend_dispatch_tests`, `options_api_tests`, `settings_tests`,
`linalg_layer_tests`; `BATCHLAS_SMOKE_TEST_TARGETS` in `tests/CMakeLists.txt`). None covers
a specific algorithm. To iterate on one algorithm, build the library plus that one test:

```bash
cmake --build build/presets/dev-tests --target stedc_tests -j"$(nproc)"
ctest --test-dir build/presets/dev-tests -R '^stedc_tests$' --output-on-failure
```

With a `[cuda:gpu]` in `sycl-ls`, `scripts/ctest_gpus.sh` runs one test per GPU slot. Labels,
the smoke set and the GPU slots: [tests/README.md](../../tests/README.md).

## Benchmarks

Benchmarks (BLAS kernels, QR/SVD, eigensolvers, band reduction, sparse) live in `benchmarks/`:

```bash
cmake --preset benchmarks
cmake --build --preset benchmarks
```

With `BATCHLAS_BUILD_BENCHMARKS=ON` the benchmark translation units join the default `all` target,
so leave the option `OFF` in trees you iterate on. To build only them, or one:

```bash
cmake --build build --target batchlas_benchmarks -j"$(nproc)"   # all of them
cmake --build build --target gemm_benchmark -j"$(nproc)"        # just one
```

Measurement rules (saturation, warm-up, one process per box): @ref dev_agent_guide.

## Python bindings

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBATCHLAS_BUILD_PYTHON=ON \
  -DBATCHLAS_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"
PYTHONPATH="$PWD/build/python" python3 -c "import batchlas; print(batchlas.available_backends())"
```

The importable package lands in `build/python` and links against the in-tree
`BatchLAS::batchlas`. It has NumPy dense-array support, SciPy sparse wrappers,
`available_backends()`, `available_devices()`, `compiled_features()` and elementwise arithmetic
(`add`, `subtract`, `multiply`, `divide`, `axpby`, `scale`). Twelve self-checking notebooks in
`python/examples/` are committed with output:

```bash
cd python/examples
PYTHONPATH=../../build/python jupyter lab            # open them
PYTHONPATH=../../build/python python3 run_all.py     # execute and check all twelve
```

## Install traps

The consumer CMake recipe, the same-compiler rule, `-fsycl` flags and `LD_LIBRARY_PATH` are in
[the C++ API](../cpp-api.md); @ref consumer_example is a runnable outside project. Three
more traps:

- **The install is AOT-pinned to the GPU architecture it was built for** (detected at configure
  time). On another GPU generation it fails at run time with `No kernel named ... was found`.
  Rebuild with `-DBATCHLAS_NVIDIA_ARCH=sm_XX`.
- **`cmake --install` never removes files.** A leftover `blas/`, `util/` or `internal/` directory in
  `<prefix>/include` from an earlier install shadows the current headers. Under a shared prefix
  such as `/usr/local` those names may belong to another package: check before removing.
- **`SOVERSION` 0, no release tag.** Every 0.x library is `.so.0` and the package version check is
  `ExactVersion`, so an ABI change inside 0.x is not visible to the loader. Pin to a commit. The
  default build ships several `.so` files;
  the `release` preset (`BATCHLAS_MONOLITHIC_LIBRARY=ON`) builds one.
  `find_package(BatchLAS CONFIG REQUIRED COMPONENTS ...)` is not supported: link
  `BatchLAS::batchlas`.

## Repository layout

| directory | contents |
| --- | --- |
| `include/` | public C++ headers |
| `src/` | library; `src/ops/<op>/` holds each op's entry point, choices and launchers, `src/select/` the table lookup |
| `tuned/` | per-device selection tables (@ref tuned_tables_readme) |
| `tools/tune/` | `batchlas_tune`, the table generator (@ref tune_tool_readme) |
| `tests/` | GoogleTest unit tests and the smoke subset |
| `benchmarks/` | performance and accuracy benchmarks; raw results in `benchmarks/results/` (Git LFS) |
| `python/` | pybind11 bindings, Python facade, tests, `examples/` |
| `scripts/` | campaign, result-processing and CI helpers |
| `playground/` | Python reference implementations cited by the band-reduction and sb2st sources |
| `examples/` | a minimal external CMake consumer |
| `evaluation/` | tuning-constant and perf-regression harnesses; not built, not installed |
| `docs/` | this site; `docs/Doxyfile` and `docs/theme/` build it |
