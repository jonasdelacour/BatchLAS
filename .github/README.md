
<div align="center">
  <img src="https://raw.githubusercontent.com/jonasdelacour/BatchLAS/main/BatchLAS_logo_transparent.png" alt="BatchLAS Logo" width="200">
</div>

# BatchLAS

BatchLAS is a SYCL-first batched linear algebra library with optional vendor backends for CUDA,
ROCm and netlib BLAS/LAPACK. The repository holds the C++ library, an optional pybind11 Python
package, a unit-test suite, benchmarks and tuning scripts.

- SYCL is mandatory. The project builds as C++20 and defaults to `RelWithDebInfo`.
- The installed CMake package exports `BatchLAS::batchlas`. Link the umbrella target; the
  component libraries are not independently linkable.
- The CMake presets in `CMakePresets.json` are the recommended entry point.

## Documentation

The documentation is a Doxygen site (Doxygen 1.18 or newer) built from `include/` and `docs/`:

```bash
sh scripts/build_docs.sh                  # writes build/docs/html/index.html
# or, from a configured tree:
cmake -S . -B build -DBATCHLAS_BUILD_DOCS=ON
cmake --build build --target batchlas_docs
```

The script takes an output directory as its argument; set `DOXYGEN=/path/to/doxygen` if Doxygen
is not on `PATH`. Every page is also plain Markdown. Code comments cite pages as
`evidence: docs/<page>.md#<anchor>`.

| directory | contents |
| --- | --- |
| `docs/guide/` | how to use a feature |
| `docs/design/` | design decisions and architecture, including [known defects](../docs/design/known-defects.md) |
| `docs/algorithms/` | mathematics and derivations behind the kernels |
| `docs/perf/` | routing windows, measured grids, rejected alternatives ([index](../docs/perf/README.md)) |
| `docs/developer/` | process and tooling, including [documentation conventions](../docs/developer/documentation.md) |

C++ calling conventions: [docs/cpp-api.md](../docs/cpp-api.md). Running tests:
[tests/README.md](../tests/README.md).

## Using the C++ API

The backend comes from the `Queue`, options are structs with defaults, and workspaces are leased
from a per-queue arena:

```cpp
#include <batchlas.hh>                 // the umbrella header
using namespace batchlas;

Queue ctx(Device::default_device());   // backend resolved from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, A.view(), {.uplo = Uplo::Upper});
ctx.wait();                            // results are not readable before this
```

The `batchlas::linalg` layer has value-returning and elementwise operations:

```cpp
auto X = linalg::solve(ctx, A.view(), B.view());   // A X = B
auto e = linalg::eigh(ctx, A.view());              // e.values, e.vectors
auto P = linalg::multiply(ctx, A.view(), B.view());  // Hadamard, not matmul
ctx.wait();                                        // required before reading X, e, P
```

- **Matrices are column-major**, and every pointer given to a `MatrixView` must be
  device-accessible (USM). A `std::vector` compiles and then aborts on a GPU backend.
- **Entry points enqueue and return.** Nothing is readable until `ctx.wait()` or a wait on the
  returned `Event`.
- **A `Queue` is single-threaded.** Use one `Queue` per thread; sharing one corrupts its
  workspace arena.

Details: [docs/cpp-api.md](../docs/cpp-api.md). Adding entry points to the library:
[docs/extending.md](../docs/extending.md). A buildable external consumer:
[examples/consumer/](../examples/consumer/README.md).

## Performance

`Auto` routes per routine, per `n` and per batch size between BatchLAS kernels and the vendor
library (cuSOLVER / cuBLAS where available). Where the vendor wins for a shape, `Auto` takes the
vendor path.

Committed measurements (RTX 4090, float, large batch, CUDA backend):

| routine | shape | BatchLAS | vendor | ratio |
| --- | --- | --- | --- | --- |
| `syev`, eigenvectors | `n = 320, batch = 819` | 67.8 µs/matrix | 203.0 (cuSOLVER) | 3.0x |
| `syev`, eigenvectors | `n = 448, batch = 585` | 195.3 µs/matrix | 400.6 (cuSOLVER) | 2.1x |
| `gesvd` (one-sided Jacobi) | `n = 8, batch = 16384` | 0.0064 µs/matrix | 0.339 (`cusolverDnXgesvdjBatched`) | 53x |
| `gesvd` (one-sided Jacobi) | `n = 32, batch = 16384` | 0.468 µs/matrix | 0.768 (`cusolverDnXgesvdjBatched`) | 1.6x |

- `syev` was measured 2026-08-07 (median of 5, harness-default block size, one process on the
  device). The vendor wins at large `n`: an earlier sweep has it 1.65x ahead at `n = 2048`, on a
  row flagged as not saturated. Grids: [syev evidence](../docs/perf/syev.md#syev-the-blocked-over-cusolver-headline-measurement).
- `gesvd` raw data: `benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`. Grids:
  [gesvd evidence](../docs/perf/gesvd.md#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix).
- Ratios measured on an unsaturated device are mostly overhead and do not transfer. Re-run on
  your own machine: see [Benchmarks and Tuning](#benchmarks-and-tuning).

## Implemented Surface Area

- **Dense BLAS:** `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`; complex Hermitian
  forms `hemm`, `herk`, `her2k`.
- **Factorizations and solvers:** `potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`, `ormqr`,
  `syev`, `gesvd`.
- **Sparse and spectral:** `spmm`, `syevx` (partial symmetric eigensolves), `lanczos`, `steqr`,
  `stedc` and related tridiagonal helpers, `ritz_values`, `iluk`.
- **Utilities:** `ortho` (several algorithms), matrix generators, norms, condition numbers,
  transpose.
- **Python:** with `BATCHLAS_BUILD_PYTHON=ON` the build produces a `batchlas` package with NumPy
  dense-array support, SciPy sparse wrappers, `available_backends()`, `available_devices()`,
  `compiled_features()` and elementwise arithmetic (`add`, `subtract`, `multiply`, `divide`,
  `axpby`, `scale`). Twelve self-checking notebooks in `python/examples/` are committed with
  output; see `python/examples/README.md`.

```bash
cd python/examples
PYTHONPATH=../../build/python jupyter lab            # open them
PYTHONPATH=../../build/python python3 run_all.py     # execute and check all twelve
```

## Repository Layout

- `include/`: public C++ headers
- `src/`: library implementation and backend/component targets
- `tests/`: GoogleTest unit tests and smoke-test subset
- `benchmarks/`: performance and accuracy benchmarks
- `python/`: pybind11 bindings, Python facade, tests, `examples/`
- `scripts/`: benchmark campaign and result-processing helpers
- `playground/`: Python reference implementations cited by the band-reduction and sb2st sources
- `docs/`: documentation pages; `docs/Doxyfile` and `docs/theme/` build the site
- `examples/`: a minimal external CMake consumer
- `evaluation/`: tuning and perf-regression harnesses; not built, not installed

## Requirements

- CMake 3.17+ (3.21+ for `cmake --preset`; `CMakePresets.json` is schema version 3)
- A C++20 compiler with SYCL support (a DPC++/Clang-family compiler; the CMake logic targets
  IntelLLVM/Clang-style SYCL compilers)
- oneDPL headers, a hard dependency (several sources include `<oneapi/dpl/...>`). The build looks
  under `/opt/intel/oneapi/dpl/latest/include`; set `ONEDPL_ROOT` otherwise.
- Optional: CUDA Toolkit (NVIDIA), ROCm (AMD), LAPACKE and CBLAS (netlib host backend), Python 3 +
  pybind11 + NumPy + SciPy (Python bindings)

> **Warning:** the SYCL compiler must have a backend for your GPU vendor. Stock Intel oneAPI
> `icpx` has no CUDA adapter: on an NVIDIA machine it configures cleanly, prints
> `-- Using SYCL targets: spir64_x86_64`, and builds a **CPU-only** library without warning. Use the
> Codeplay *oneAPI for NVIDIA GPUs* plugin on top of oneAPI, or a self-built `intel/llvm`
> configured with `--cuda`. `sycl-ls` must list a `[cuda:gpu]` entry.

Package suggestions and oneAPI notes: [agent environment guide](../docs/developer/agent-guide.md).

### Tested platforms

Only the Primary configuration is exercised regularly; everything else is untested rather than
known-good. `.github/workflows/ci.yml` runs static checks on GitHub-hosted runners (no SYCL
compiler there, so nothing is configured or compiled) and a self-hosted GPU job on the Primary
machine: configure with CUDA required, build all libraries and tests, `ctest`, install, check the
export. A nightly job on that runner adds slow suites, the consumer packaging test and a
vendor-free build.

| | Compiler | CUDA | GPU / arch | OS | Status |
| --- | --- | --- | --- | --- | --- |
| Primary | `intel/llvm` DPC++, clang 22.0.0git, built with `--cuda` (installed at `/opt/dpcpp-cuda`) | 13.2 | NVIDIA RTX 4090, `sm_89` | Ubuntu 22.04 | Library, tests and benchmarks built and run here daily |
| CI — static checks | none (no toolchain) | — | none | `ubuntu-latest` | List files, source-level export and public headers only; nothing configured, compiled or run |
| CI — GPU gate | the Primary row's toolchain, on a self-hosted runner on that machine | 13.2 | NVIDIA RTX 4090, `sm_89` | Ubuntu 22.04 | Build + `ctest -LE slow` + install on every push and non-fork PR; full `ctest`, packaging and a vendor-free build nightly |
| CPU only | Intel oneAPI `icpx` 2025.x | — | none (`spir64_x86_64` / `native_cpu`) | Ubuntu 22.04 | Configures and builds; **no NVIDIA support** (see the warning above). Not built by CI |
| AMD / ROCm | — | — | — | — | Code paths exist; not built or run |
| macOS / Windows | — | — | — | — | Untested |

Other NVIDIA architectures should work: the build detects the local GPU, and
`-DBATCHLAS_NVIDIA_ARCH=sm_XX` overrides it. Only `sm_89` has been run.

`main` has known test failures, so the GPU job diffs each run against the ledger
`tests/known-failures.txt` instead of using `ctest`'s exit code. See
**[docs/ci.md](../docs/ci.md)** for coverage, self-hosted runner setup and retiring a ledger entry.

## Build

### Recommended Preset Workflow

```bash
export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"   # the presets set no job count
cmake --preset dev
cmake --build --preset dev
```

Presets:

- `dev`: default `RelWithDebInfo` library build
- `dev-tests`: library plus the full test suite
- `fast-dev`: library plus the smoke-test subset
- `benchmarks`: benchmark build with tuning support
- `cuda`: CUDA-enabled build when the environment supports it
- `dev-gpu` / `dev-gpu-tests`: drop the `native_cpu` SYCL target (`BATCHLAS_CPU_TARGET=none`);
  13-25% off a cold build of the default target on one 4-GPU box. **GPU coverage only**: about
  half of every typed test suite is not instantiated and `ctest` still reports green. Use
  `dev-tests` or `cuda` for the pre-push gate, plus a vendor-free tree
  (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) at that gate only. With a `[cuda:gpu]` in `sycl-ls`,
  `scripts/ctest_gpus.sh` runs one test per GPU slot (`tests/README.md`).
- `acpp` / `acpp-tests`: `dev` / `dev-tests` built with the AdaptiveCpp 25.10 driver
  (`--acpp-targets=generic`) into `build-acpp` / `build-acpp-tests`. Work in progress: the library
  does not compile under acpp yet (`docs/design/sycl-implementations.md`).

### Manual Configuration

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBATCHLAS_BUILD_TESTS=ON \
  -DBATCHLAS_BUILD_BENCHMARKS=OFF \
  -DBATCHLAS_BUILD_PYTHON=OFF

cmake --build build -j"$(nproc)"
```

Common CMake options:

| option | meaning |
| --- | --- |
| `BATCHLAS_BUILD_TESTS` | build unit tests |
| `BATCHLAS_BUILD_BENCHMARKS` | build benchmark executables |
| `BATCHLAS_BUILD_PYTHON` | build the Python package |
| `BATCHLAS_ENABLE_CUDA` | `AUTO` (default; enable cuBLAS/cuSOLVER when the SYCL runtime exposes a CUDA device), `ON` (require it; configure fails without a `[cuda:gpu]` in `sycl-ls`) or `OFF`. A cache entry from the old boolean option is migrated on the first re-configure (`OFF` becomes `AUTO`, `ON` stays `ON`); pass `-DBATCHLAS_ENABLE_CUDA=OFF` to force off |
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
| `BATCHLAS_SYCL_LINK_JOBS` | parallelism of the DPC++ SYCL device link (default `4`, `1` disables) |
| `BATCHLAS_SYCL_IMPL` | `AUTO` (default; `ACPP` when `CMAKE_CXX_COMPILER` is the `acpp` driver), `DPCPP` or `ACPP`. Only the matching compiler is accepted |
| `BATCHLAS_ACPP_TARGETS` | `--acpp-targets` of an AdaptiveCpp build (default `generic`) |

## Test

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

Before pushing, build and run everything:

```bash
cmake --build build/presets/dev-tests -j"$(nproc)"
ctest --test-dir build/presets/dev-tests
```

## Benchmarks and Tuning

Benchmarks (BLAS kernels, QR/SVD, eigensolvers, band reduction, sparse) live in `benchmarks/`:

```bash
cmake --preset benchmarks
cmake --build --preset benchmarks
```

With `BATCHLAS_BUILD_BENCHMARKS=ON` the 61 benchmark translation units join the default `all`
target, so leave the option `OFF` in trees you iterate on. To build only them, or one:

```bash
cmake --build build --target batchlas_benchmarks -j"$(nproc)"   # all of them
cmake --build build --target gemm_benchmark -j"$(nproc)"        # just one
```

`scripts/` has campaign helpers and result processing. Tuning is enabled by
`BATCHLAS_ENABLE_TUNING`.

## Python Bindings

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBATCHLAS_BUILD_PYTHON=ON \
  -DBATCHLAS_BUILD_TESTS=ON

cmake --build build -j"$(nproc)"
PYTHONPATH="$PWD/build/python" python3 -c "import batchlas; print(batchlas.available_backends())"
```

The importable package lands in `build/python`. The pybind11 extension links against the
installed or in-tree `BatchLAS::batchlas`.

## Consuming BatchLAS from CMake

A buildable example lives in [examples/consumer/](../examples/consumer/README.md).

### The short version

```bash
# 1. install BatchLAS out of an existing build tree
cmake --install build --prefix "$HOME/inst"

# 2. configure YOUR project with the SAME SYCL compiler BatchLAS was built with
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_PREFIX_PATH="$HOME/inst"

# 3. build and run; the DPC++ runtime has to be findable at load time
cmake --build build -j"$(nproc)"
LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:$LD_LIBRARY_PATH ./build/my_app
```

```cmake
cmake_minimum_required(VERSION 3.17)
project(my_app CXX)

find_package(BatchLAS CONFIG REQUIRED)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE BatchLAS::batchlas)
```

Substitute your DPC++ prefix for `/opt/dpcpp-cuda`. The compiler an install was built with is
`CMAKE_CXX_COMPILER` in the `CMakeCache.txt` of its build tree.

### Four things that will bite you if you skip them

**1. The whole consuming project must use the same SYCL compiler.** Clang encodes C++20
`requires` clauses into mangled names; GCC and Clang < 16 do not. `Matrix`'s constrained
constructors therefore get different symbol names, and the link fails without mentioning BatchLAS
or constraints:

```
undefined reference to `batchlas::Matrix<float, (batchlas::MatrixFormat)0>::Matrix<...>(int, int, int, int, int)'
```

The clang symbol carries an extra `Q...` component for the requires-clause. There is no
consumer-side workaround; set `-DCMAKE_CXX_COMPILER=` to the same compiler.

**2. `-fsycl` is not added by the package.** The public headers keep `<sycl/sycl.hpp>` out, so a
TU that only calls the documented API compiles without `-fsycl`. A TU that includes
`<batchlas/blas/device.hh>` or `<sycl/sycl.hpp>`, or writes its own kernels, needs the flags,
with the same `-fsycl-targets` value the library was built with:

```cmake
target_compile_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
target_link_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
```

**3. The install is AOT-pinned to the GPU architecture it was built for** (`sm_89` on the
reference machine, detected at configure time). On a different GPU generation it fails at run time
with `No kernel named ... was found`. Rebuild for the target, overriding detection if needed; the
build also records the CUDA toolkit it found:

```bash
cmake -S . -B build -DBATCHLAS_NVIDIA_ARCH=sm_80
```

**4. `LD_LIBRARY_PATH` must cover the DPC++ runtime.** BatchLAS's libraries carry a `RUNPATH` to
the install prefix; the SYCL runtime does not follow. If DPC++ is outside the ldconfig path, the
binary dies with

```
error while loading shared libraries: libsycl.so.9: cannot open shared object file
```

Export `LD_LIBRARY_PATH=<dpcpp-prefix>/lib`. For containers and CI, add a file to
`/etc/ld.so.conf.d/` and run `ldconfig`: the SYCL runtime `dlopen`s its UR adapters by bare soname,
which no RPATH on your binary covers.

### Where the headers land

Everything installs under `<prefix>/include/batchlas/` plus the umbrella
`<prefix>/include/batchlas.hh`; nothing else in your include root:

```cpp
#include <batchlas.hh>                        // umbrella
#include <batchlas/blas/linalg.hh>            // or reach in directly
#include <batchlas/util/sycl-device-queue.hh>
```

`cmake --install` never removes files. A leftover `blas/`, `util/` or `internal/` directory in
`<prefix>/include` from an earlier install shadows the current headers. Check it is ours before
removing it, since under a shared prefix such as `/usr/local` those names may belong to another
package:

```bash
ls <prefix>/include/blas
```

### What the package does and does not give you

- It exports `BatchLAS::batchlas` and the generated configuration headers the public interface
  needs.
- `find_package(BatchLAS CONFIG REQUIRED COMPONENTS ...)` is not supported; link the umbrella
  target.
- The library ships as several `.so` files without an `SOVERSION`, and there is no released tag.
  Pin to a commit.

## Development Notes

- `batchlas` is an interface facade over split component libraries.
- Each op's entry point, choices and launchers live in `src/ops/<op>/`; table lookup, vendor
  availability, trace and coverage are in `src/select/`; per-device tables are in `tuned/`. Design:
  [docs/design/flat-kernel-selection.md](../docs/design/flat-kernel-selection.md).
- Notes and measurements live in `docs/`, not in Markdown elsewhere: `docs/perf/`, `docs/design/`,
  `docs/algorithms/`, `docs/developer/` and [docs/extending.md](../docs/extending.md). Build,
  testing and measurement rules: [agent environment guide](../docs/developer/agent-guide.md).

## License

BatchLAS is licensed under the MIT License. See `LICENSE`.
