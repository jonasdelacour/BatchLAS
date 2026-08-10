
<div align="center">
  <img src="BatchLAS_logo_transparent.png" alt="BatchLAS Logo" width="200">
</div>

# BatchLAS

BatchLAS is a SYCL-first batched linear algebra library with optional vendor backends for CUDA, ROCm, netlib BLAS/LAPACK, and oneMKL. The repository currently contains the C++ library, an optional pybind11-based Python package, a broad unit-test suite, benchmark executables, tuning scripts, and research notebooks used to validate newer eigensolver and factorization work.

## Current Status

- SYCL is mandatory for building the library.
- The project builds as `C++20` and defaults to `RelWithDebInfo`.
- The installed CMake package exports `BatchLAS::batchlas` plus the component
  libraries it is built from. Link the umbrella target; the components are not
  independently linkable.
- The repository includes active work on dense factorizations, spectral routines, orthogonalization, sparse eigensolvers, and performance benchmarking.
- Recommended development entry points are the CMake presets in `CMakePresets.json`.

## Using the C++ API

The backend comes from the `Queue`, options are structs with defaults, and
workspaces are leased from a per-queue arena, so a call is usually one line:

```cpp
#include <batchlas.hh>                 // the umbrella header
using namespace batchlas;

Queue ctx(Device::default_device());   // backend resolved from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, A.view(), {.uplo = Uplo::Upper});
ctx.wait();                            // results are not readable before this
```

There is also a `batchlas::linalg` convenience layer with value-returning and
elementwise operations:

```cpp
auto X = linalg::solve(ctx, A.view(), B.view());   // A X = B
auto e = linalg::eigh(ctx, A.view());              // e.values, e.vectors
auto P = linalg::multiply(ctx, A.view(), B.view());  // Hadamard, not matmul
ctx.wait();                                        // required before reading X, e, P
```

Three things that are easy to get wrong and are covered in full in
[docs/cpp-api.md](docs/cpp-api.md):

- **Matrices are column-major**, and every pointer you hand to a `MatrixView`
  must be device-accessible (USM). A `std::vector` compiles and then aborts the
  process on a GPU backend.
- **Entry points enqueue and return.** Nothing is readable until `ctx.wait()`
  (or a wait on the returned `Event`).
- **A `Queue` is single-threaded.** Use one `Queue` per thread; sharing one
  across threads corrupts its workspace arena.

See **[docs/cpp-api.md](docs/cpp-api.md)** for the data-layout and memory
contract, the full conventions, the workspace-lifetime caveat on out-of-order
queues, and a migration guide from the older `gemm<Backend::CUDA, float>(...)`
spelling — which still compiles. A complete, buildable external consumer lives
in [`examples/consumer/`](examples/consumer/).

## Performance

The thing to know before the numbers: **`Auto` routes per routine, per `n`, per
batch size**, between BatchLAS's own kernels and the vendor library (cuSOLVER /
cuBLAS where available). Adopting BatchLAS therefore should not make you slower
than the vendor loop you have today — where the vendor wins for a shape, that is
the path `Auto` takes.

Two measurements that are committed in this repository, with their conditions,
rather than a headline number:

- **`syev`, eigenvectors, float, RTX 4090, CUDA backend** (grid in
  `include/blas/functions/syev.hh`, measured 2026-08-07, µs per matrix, median
  of 5, harness-default block size, one process on the device): at
  `n = 320, batch = 819` BatchLAS's blocked solver runs at 67.8 µs
  vs cuSOLVER's 203.0 µs (**3.0x**); at `n = 448, batch = 585` it is 195.3 vs
  400.6 (**2.1x**). The vendor wins at large `n` — an earlier sweep in the same
  header has it 1.65x ahead at `n = 2048`, on a row the header itself flags as
  not saturated — and `Auto` routes there accordingly. The header carries the
  full grids, including the corrections that superseded earlier ones.
- **`gesvd` vs `cusolverDnXgesvdjBatched`, float, RTX 4090**
  (`benchmarks/results/gesvd_vs_gesvdj_rtx4090.csv`): at `n = 8, batch = 16384`
  BatchLAS's one-sided Jacobi SVD is 0.0064 µs/matrix vs 0.339 µs/matrix; at
  `n = 32, batch = 16384` it is 0.468 vs 0.768.

Both of those are single-machine numbers from the "Tested platforms" table below,
at large batch. Ratios measured on an *unsaturated* device are mostly overhead
and do not transfer; if the numbers matter to your decision, run them yourself —
see [Benchmarks and Tuning](#benchmarks-and-tuning) for how to build the suite,
and `benchmarks/results/` for the committed raw output.

## Implemented Surface Area

The public C++ headers under `include/` currently expose these main groups of functionality.

### Dense BLAS and Factorization

- `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`
- `hemm`, `herk`, `her2k` (complex Hermitian forms)
- `potrf`, `getrf`, `getrs`, `getri`
- `geqrf`, `orgqr`, `ormqr`
- `syev`, `gesvd`

### Sparse and Spectral Extensions

- `spmm`
- `syevx` for partial symmetric eigensolves
- `lanczos`
- `steqr`, `stedc`, and related tridiagonal helpers
- `ritz_values`
- `iluk` preconditioning support

### Orthogonalization and Utilities

- `ortho` with multiple orthogonalization algorithms
- matrix generators and structured constructors
- norms, condition numbers, transpose, and related helpers

### Python Package

When `BATCHLAS_BUILD_PYTHON=ON`, the repository builds a `batchlas` Python package with NumPy dense-array support and SciPy sparse wrappers for the supported public APIs. The Python facade also exposes convenience helpers such as `available_backends()`, `available_devices()`, and `compiled_features()`, plus elementwise arithmetic (`add`, `subtract`, `multiply`, `divide`, `axpby`, `scale`) over batched dense arrays.

Twelve self-checking example notebooks covering the whole Python surface live in `python/examples/`. They are committed with output from a reference run, so they render on GitHub without executing anything:

```bash
cd python/examples
PYTHONPATH=../../build/python jupyter lab            # open them
PYTHONPATH=../../build/python python3 run_all.py     # execute and check all twelve
```

See `python/examples/README.md` for the index, the array/batching conventions, and current known issues.

## Repository Layout

- `include/`: public C++ headers
- `src/`: library implementation and backend/component targets
- `tests/`: GoogleTest-based unit tests and smoke-test subset
- `benchmarks/`: performance and accuracy benchmark executables
- `python/`: pybind11 bindings, Python facade, Python tests, and `examples/`
- `scripts/`: benchmark campaign helpers and result-processing scripts
- `playground/`: notebooks and exploratory scripts for algorithm work
- `docs/`: the C++ API reference (`docs/cpp-api.md`) and design documentation
- `examples/`: a minimal external CMake consumer
- `experiments/`, `plotting/`, `evaluation/`: research scaffolding; not part of
  the build and not installed

## Requirements

Minimum build requirements:

- CMake 3.17+ (3.17 is what `find_package(CUDAToolkit)` needs; 3.21+ if you want
  the `cmake --preset` workflow, since `CMakePresets.json` is schema version 3)
- A C++20 compiler with SYCL support — in practice a DPC++/Clang-family compiler
- A SYCL runtime/toolchain discoverable by CMake
- oneDPL headers. Several sources include `<oneapi/dpl/...>` unconditionally, so
  this is a hard dependency, not an option. The build looks for it under
  `/opt/intel/oneapi/dpl/latest/include` today.

**Your SYCL compiler must have a backend for your GPU vendor.** This is the
single most common way to get a working build that quietly does the wrong thing:
stock Intel oneAPI `icpx` has no CUDA adapter, so on an NVIDIA machine it
configures cleanly, prints `-- Using SYCL targets: spir64_x86_64`, and builds a
**CPU-only** library with no warning. For NVIDIA you need a CUDA-capable DPC++:
the Codeplay *oneAPI for NVIDIA GPUs* plugin on top of oneAPI, or a self-built
`intel/llvm` configured with `--cuda`. Verify before building — `sycl-ls` must
list a `[cuda:gpu]` entry.

Common optional dependencies:

- CUDA Toolkit for NVIDIA backends
- ROCm for AMD backends
- LAPACKE and CBLAS for the netlib host backend
- oneMKL for the optional MKL backend
- Python 3, pybind11, NumPy, and SciPy for Python bindings

Notes:

- SYCL support is not optional in the current build system.
- The CMake logic is primarily written around IntelLLVM/Clang-style SYCL compilers.
- The default build type is `RelWithDebInfo`, not `Debug`.

For a Linux-oriented environment setup with package suggestions and oneAPI notes, see `AGENTS.md`.

### Tested platforms

Only one configuration is exercised regularly. CI (`.github/workflows/ci.yml`)
checks list files, the exported package and the public headers only — a
GitHub-hosted runner has no SYCL compiler, so nothing is configured, compiled,
tested or run there. Everything outside the Primary row is untested rather than
known-good.

| | Compiler | CUDA | GPU / arch | OS | Status |
| --- | --- | --- | --- | --- | --- |
| Primary | `intel/llvm` DPC++, clang 22.0.0git, built with `--cuda` (installed at `/opt/dpcpp-cuda`) | 13.2 | NVIDIA RTX 4090, `sm_89` | Ubuntu 22.04 | Library, tests and benchmarks built and run here daily |
| CPU only | Intel oneAPI `icpx` 2025.x | — | none (`spir64_x86_64` / `native_cpu`) | Ubuntu 22.04 | Configures and builds; **no NVIDIA support** — see the warning above |
| AMD / ROCm | — | — | — | — | Code paths exist; not built or run by anyone here |
| oneMKL backend | — | — | Intel GPU | — | Code paths exist; not built or run by anyone here |
| macOS / Windows | — | — | — | — | Untested; no attempt made |

Other NVIDIA architectures should work — the build detects the local GPU and can
be pointed elsewhere with `-DBATCHLAS_NVIDIA_ARCH=sm_XX` — but nothing but
`sm_89` has been run.

## Build

### Recommended Preset Workflow

Configure and build using the checked-in presets:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"   # the presets set no job count
cmake --preset dev
cmake --build --preset dev
```

The build presets deliberately do not hardcode a job count (it used to be 20,
which oversubscribes a small machine and undersubscribes a large one). Set
`CMAKE_BUILD_PARALLEL_LEVEL` once as above, or pass `--parallel N` per build.

Useful presets currently provided:

- `dev`: default `RelWithDebInfo` library build
- `dev-tests`: library build with the full test suite enabled
- `fast-dev`: library build plus the smoke-test subset
- `benchmarks`: benchmark build with tuning support enabled
- `cuda`: optional CUDA-enabled build when the environment supports it
- `dev-gpu` / `dev-gpu-tests`: fast iteration; drops the `native_cpu` SYCL
  target (`BATCHLAS_CPU_TARGET=none`), which takes about 30% off every compile
  and every device link. **GPU coverage only**: roughly half of every typed test
  suite is not instantiated, and `ctest` still reports green. Use `dev-tests` or
  `cuda` for the pre-push gate.

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

- `BATCHLAS_BUILD_TESTS`: build unit tests
- `BATCHLAS_BUILD_BENCHMARKS`: build benchmark executables
- `BATCHLAS_BUILD_PYTHON`: build the Python package
- `BATCHLAS_ENABLE_CUDA`: `AUTO` (default; enable the cuBLAS/cuSOLVER backend
  when the SYCL runtime exposes a CUDA device), `ON` (require it — configure
  fails if `sycl-ls` shows no `[cuda:gpu]`) or `OFF`. Plain booleans still work.
  A build directory configured before this became a tri-state carries the old
  `BATCHLAS_ENABLE_CUDA:BOOL` entry; the first re-configure migrates it (`OFF`,
  which used to be overridden silently, becomes `AUTO`; `ON` stays `ON`) and
  says so. Pass `-DBATCHLAS_ENABLE_CUDA=OFF` if you really want it off
- `BATCHLAS_CUDA_DEVICE_LINE_INFO`: pass `--generate-line-info` to the NVPTX
  backend in Debug/RelWithDebInfo builds, for `ncu`/Nsight (default `OFF`; it has
  been observed to fail CUDA JIT program builds)
- `BATCHLAS_STRIP_RELWITHDEBINFO_G`: drop the toolchain's `-g` from
  `CMAKE_CXX_FLAGS_RELWITHDEBINFO` (default `ON`; BatchLAS adds
  `-gline-tables-only` itself, and full DWARF in the device images has been
  observed to fail CUDA JIT program builds)
- `ONEDPL_ROOT`: root of a oneDPL installation, when it is not on the default
  search path — see Requirements. The build fails at configure time without it
- `BATCHLAS_ENABLE_ROCM`: enable ROCm backend support even if no AMD GPU is auto-detected
- `BATCHLAS_ENABLE_NETLIB`: enable the host netlib backend
- `BATCHLAS_ENABLE_MKL`: enable the oneMKL backend
- `BATCHLAS_ENABLE_TUNING`: enable tuning targets; intended for benchmark builds
- `BATCHLAS_CPU_TARGET`: override SYCL CPU target selection (`auto`, `native_cpu`, `spir64_x86_64`, `none`)
- `BATCHLAS_TEST_TARGET_SET`: choose `all` or `smoke`
- `BATCHLAS_AMD_ARCH`: override ROCm target architecture
- `BATCHLAS_NVIDIA_ARCH`: override CUDA target architecture
- `BATCHLAS_USE_CCACHE`: cache compilations with ccache when available (default `ON`)
- `BATCHLAS_CCACHE_SHARE_ACROSS_TREES`: let sibling checkouts and worktrees share
  cache entries (default `ON`; turn off for source-level debugging)
- `BATCHLAS_SYCL_LINK_JOBS`: parallelism for the SYCL device link (default `4`, `1` disables)

## Test

Build tests and run them with either the preset or a manual build:

```bash
cmake --preset dev-tests
cmake --build --preset dev-tests
ctest --test-dir build/presets/dev-tests --output-on-failure
```

The `fast-dev` preset builds only the smoke subset:

- `util_span_tests`
- `util_vector_tests`
- `matrix_tests`
- `mempool_tests`
- `backend_dispatch_tests`
- `options_api_tests`
- `linalg_layer_tests`

None of those covers a specific algorithm, so `fast-dev` is not the preset to
iterate in. To work on one algorithm, use a full tree and build only the binary
you care about — the library plus that one test, and nothing else:

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

The repository contains a large benchmark suite under `benchmarks/`, including BLAS kernels, QR/SVD paths, eigensolvers, band reduction, and sparse workflows. A typical benchmark build looks like this:

```bash
cmake --preset benchmarks
cmake --build --preset benchmarks
```

Benchmark executables are **not** part of the default `all` target — they are 61
heavy translation units that no test run needs. The preset above still builds
them all, because it names the `batchlas_benchmarks` aggregate explicitly. In a
hand-configured tree, ask for them by name or via the aggregate:

```bash
cmake --build build --target batchlas_benchmarks -j"$(nproc)"   # all of them
cmake --build build --target gemm_benchmark -j"$(nproc)"        # just one
```

The `scripts/` directory contains campaign helpers and archived CSV outputs from prior runs. Tuning support is wired through `BATCHLAS_ENABLE_TUNING` and the optional `BATCHLAS_TUNING_PROFILE` cache entry.

## Python Bindings

Enable the Python package like this:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBATCHLAS_BUILD_PYTHON=ON \
  -DBATCHLAS_BUILD_TESTS=ON

cmake --build build -j"$(nproc)"
```

The build places the importable package under `build/python`, so a build-tree import looks like:

```bash
PYTHONPATH="$PWD/build/python" python3 -c "import batchlas; print(batchlas.available_backends())"
```

The extension module is built with pybind11 and linked against the installed or in-tree `BatchLAS::batchlas` target.

## Consuming BatchLAS from CMake

A complete, buildable example of everything in this section lives in
[`examples/consumer/`](examples/consumer/) — start from that rather than from the
snippets below.

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

and in your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.17)
project(my_app CXX)

find_package(BatchLAS CONFIG REQUIRED)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE BatchLAS::batchlas)
```

Substitute your own DPC++ prefix for `/opt/dpcpp-cuda` throughout. If you do not
know which compiler a given install was built with, look for
`CMAKE_CXX_COMPILER` in the `CMakeCache.txt` of the build tree it came from.

### Four things that will bite you if you skip them

**1. The whole consuming project must use the same SYCL compiler.** Not "a C++20
compiler", and not just the targets that touch BatchLAS. Clang encodes C++20
`requires` clauses into mangled names and GCC (and Clang < 16) does not, so
`Matrix`'s constrained constructors get *different symbol names* under the two
compilers. The failure is at link time and does not mention BatchLAS or
constraints:

```
undefined reference to `batchlas::Matrix<float, (batchlas::MatrixFormat)0>::Matrix<...>(int, int, int, int, int)'
```

`nm` will show you a symbol that looks like the one you want — the clang mangling
carries an extra `Q...` component for the requires-clause. There is no consumer-side
workaround; pass `-DCMAKE_CXX_COMPILER=` pointing at the same compiler.

**2. `-fsycl` is your business, not the package's.** The exported target does not
force SYCL flags onto your translation units. That is deliberate: the public
BatchLAS headers keep `<sycl/sycl.hpp>` out on purpose, so a TU that only calls
the documented API compiles without `-fsycl` and without paying for a device
compilation pass. But if a TU of yours includes `<blas/device.hh>`, includes
`<sycl/sycl.hpp>`, or writes its own kernels, it needs the flags and you add them
yourself:

```cmake
target_compile_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
target_link_options(my_app PRIVATE -fsycl -fsycl-targets=nvidia_gpu_sm_89)
```

Use the same `-fsycl-targets` value the library was built with; mixing them is not
a configuration anyone has tested.

**3. The install is AOT-pinned to the GPU architecture it was built for.** Device
code is compiled ahead of time for the arch detected at configure time (`sm_89`
on the reference machine), so an install tree is not portable to a different GPU
generation. Copying it to another card gives a *runtime* error —
`No kernel named ... was found` — not a build error. Rebuild for the target
machine, overriding detection if needed:

```bash
cmake -S . -B build -DBATCHLAS_NVIDIA_ARCH=sm_80
```

The same applies to CUDA: the build records the CUDA toolkit it found.

**4. `LD_LIBRARY_PATH` must cover the DPC++ runtime.** BatchLAS's own libraries
carry a `RUNPATH` to the install prefix, but the SYCL runtime does not follow —
if DPC++ lives outside the ldconfig search path, your binary dies with

```
error while loading shared libraries: libsycl.so.9: cannot open shared object file
```

Export `LD_LIBRARY_PATH=<dpcpp-prefix>/lib` for interactive use. For containers
and CI the more robust fix is to drop a file in `/etc/ld.so.conf.d/` and run
`ldconfig`, because the SYCL runtime also `dlopen`s its UR adapters by bare
soname and those are not covered by any RPATH you could set on your binary.

### What the package does and does not give you

- It exports `BatchLAS::batchlas` and the generated configuration headers the
  public interface needs.
- `find_package(BatchLAS CONFIG REQUIRED COMPONENTS ...)` is not supported;
  the components are not independently linkable. Link the umbrella target.
- The library is shipped as several `.so` files without an `SOVERSION`, and
  there is no released tag yet. Pin to a commit, not to a version.

## Development Notes

- The top-level `batchlas` target is an interface facade over split component libraries.
- The repository includes implementation notes for ongoing work in the root markdown files and under `docs/`.
- `playground/` contains exploratory notebooks and scripts used during algorithm development.

## License

BatchLAS is licensed under the MIT License. See `LICENSE` for the full text.