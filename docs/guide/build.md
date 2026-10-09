# Building, testing and benchmarking {#guide_build}

> **Status:** current · moved from the project README on 2026-10-09.

Requirements, tested platforms, CMake presets and options, the test and benchmark builds, the
Python package and the install traps. Calling the library and linking an installed package are in
[the C++ API](../cpp-api.md) ("Building and installing BatchLAS", "Building against
BatchLAS").

## Requirements

- CMake 3.14+ (3.21+ for `cmake --preset`; `CMakePresets.json` is schema version 3).
- A C++20 compiler with SYCL support: a DPC++/Clang-family compiler, clang 16 or newer.
- oneDPL headers, a hard dependency (several sources include `<oneapi/dpl/...>`). The build looks
  under `/opt/intel/oneapi/dpl/latest/include`; set `ONEDPL_ROOT` otherwise.
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
| macOS / Windows | — | — | — | — | Untested |

Other NVIDIA architectures should work: the build detects the local GPU, and
`-DBATCHLAS_NVIDIA_ARCH=sm_XX` overrides it.

`main` has known test failures, so the GPU job diffs each run against the ledger
`tests/known-failures.txt` instead of using `ctest`'s exit code. See [the CI page](../ci.md) for
coverage, self-hosted runner setup and retiring a ledger entry.

## Presets

```bash
export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"   # the presets set no job count
cmake --preset dev
cmake --build --preset dev
```

| preset | builds |
| --- | --- |
| `dev` | default `RelWithDebInfo` library |
| `dev-tests` | library plus the full test suite |
| `fast-dev` | library plus the smoke-test subset |
| `benchmarks` | benchmarks and the tuner (`BATCHLAS_ENABLE_TUNING=ON`) |
| `cuda` | CUDA-enabled build with tests; the pre-push gate |
| `dev-gpu` / `dev-gpu-tests` | drop the `native_cpu` SYCL target (`BATCHLAS_CPU_TARGET=none`) |
| `release` | `Release`, no tests or benchmarks, one shared library |

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
cmake --build build -j"$(nproc)"
```

| option | meaning |
| --- | --- |
| `BATCHLAS_BUILD_TESTS` | build unit tests |
| `BATCHLAS_BUILD_BENCHMARKS` | build benchmark executables and `batchlas_tune` |
| `BATCHLAS_BUILD_PYTHON` | build the Python package |
| `BATCHLAS_ENABLE_CUDA` | `AUTO` (default; enable cuBLAS/cuSOLVER when the SYCL runtime exposes a CUDA device), `ON` (require it; configure fails without a `[cuda:gpu]` in `sycl-ls`) or `OFF`. A cache entry from the old boolean option is migrated on the first re-configure (`OFF` becomes `AUTO`, `ON` stays `ON`); pass `-DBATCHLAS_ENABLE_CUDA=OFF` to force off |
| `BATCHLAS_ENABLE_VENDOR_BLAS` | master switch over cuBLAS/cuSOLVER/cuSPARSE/rocBLAS; `OFF` builds the vendor-free library ([vendor-free status](../design/vendor-free-status.md)) |
| `BATCHLAS_CUDA_DEVICE_LINE_INFO` | pass `--generate-line-info` to the NVPTX backend in Debug/RelWithDebInfo, for `ncu`/Nsight (default `OFF`; has failed CUDA JIT program builds) |
| `BATCHLAS_STRIP_RELWITHDEBINFO_G` | drop the toolchain's `-g` from `CMAKE_CXX_FLAGS_RELWITHDEBINFO` (default `ON`; BatchLAS adds `-gline-tables-only`; full DWARF in device images has failed CUDA JIT program builds) |
| `ONEDPL_ROOT` | root of a oneDPL installation not on the default search path; configure fails without oneDPL |
| `BATCHLAS_ENABLE_ROCM` | enable ROCm backend support even if no AMD GPU is detected |
| `BATCHLAS_ENABLE_NETLIB` | enable the host netlib backend |
| `BATCHLAS_ENABLE_TUNING` | enable tuning targets; for benchmark builds |
| `BATCHLAS_CPU_TARGET` | SYCL CPU target: `auto`, `native_cpu`, `spir64_x86_64`, `none` |
| `BATCHLAS_TEST_TARGET_SET` | `all` or `smoke` |
| `BATCHLAS_AMD_ARCH` | override ROCm target architecture |
| `BATCHLAS_NVIDIA_ARCH` | override CUDA target architecture |
| `BATCHLAS_USE_CCACHE` | cache compilations with ccache when on `PATH` or in `~/.local/bin` (default `ON`; your own `CMAKE_CXX_COMPILER_LAUNCHER` wins) |
| `BATCHLAS_CCACHE_SHARE_ACROSS_TREES` | let sibling checkouts and worktrees share cache entries (default `ON`; turn off for source-level debugging) |
| `BATCHLAS_CCACHE_BASEDIR` | ccache base directory (default: deepest common parent of source and build directories, or `$HOME` if that is `/`) |
| `BATCHLAS_TEST_GPUS` | GPUs written to `<build>/ctest_resources.json` for `scripts/ctest_gpus.sh`: `auto` (`nvidia-smi --list-gpus`), a count, or `0` to disable. Needs a `[cuda:gpu]` in `sycl-ls`, else a count is a configure error |
| `BATCHLAS_TEST_GPU_SLOTS` | concurrent GPU tests per device under `scripts/ctest_gpus.sh` |
| `BATCHLAS_SYCL_LINK_JOBS` | parallelism of the SYCL device link (default `4`, `1` disables) |

## Tests

```bash
cmake --preset dev-tests
cmake --build --preset dev-tests
ctest --test-dir build/presets/dev-tests --output-on-failure
```

`fast-dev` builds only the smoke subset (`util_span_tests`, `util_vector_tests`, `matrix_tests`,
`mempool_tests`, `backend_dispatch_tests`, `options_api_tests`, `linalg_layer_tests`). None covers
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
- **No `SOVERSION`, no release tag.** Pin to a commit. The default build ships several `.so` files;
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
