# Building, testing and benchmarking {#guide_build}

> **Status:** current · checked against `CMakePresets.json` and `cmake/` on 2026-10-29.

Requirements, presets, the CMake options a user sets, macOS, tests, benchmarks, Python and install
traps. Linking an installed package is in [the C++ API](../cpp-api.md); build-speed internals are in
@ref dev_agent_guide §7.

## Requirements

- CMake 3.17+ (3.21+ for `cmake --preset`).
- A C++20 SYCL compiler: DPC++ (`intel/llvm` or oneAPI `icpx`) or AdaptiveCpp 25.10 (`acpp`).
  Configure stops on a compiler that is not Clang-family.
- Optional: CUDA Toolkit, ROCm, LAPACKE + CBLAS (netlib host backend), Python 3 + pybind11 + NumPy
  (+ SciPy for sparse). Packages and oneAPI notes: @ref dev_agent_guide §1-§5.

> **Warning:** the SYCL compiler needs a backend for your GPU. Stock oneAPI `icpx` has no CUDA
> adapter: on an NVIDIA machine it configures cleanly, prints `-- Using SYCL targets: spir64_x86_64`
> and builds a **CPU-only** library. Use the Codeplay *oneAPI for NVIDIA GPUs* plugin or an
> `intel/llvm` built with `--cuda`; `sycl-ls` must list a `[cuda:gpu]`.

## Tested platforms

Only the first row is exercised daily and gated by CI ([CI](../ci.md)); the rest are built and
tested by hand, or untested.

| | Compiler | GPU / arch | Status |
| --- | --- | --- | --- |
| Primary | `intel/llvm` DPC++ (clang 22) built `--cuda`, CUDA 13.2, Ubuntu 26.04 | RTX 4090, `sm_89` | Library, tests, benchmarks; CI GPU gate |
| Second GPU box | same toolchain | RTX PRO 6000, `sm_120` | Built and tested; `sm_120` tables measured here |
| AdaptiveCpp generic | AdaptiveCpp 25.10.0, LLVM 20, SSCP, CUDA 13.2 | RTX PRO 6000, `sm_120` | Built and tested; ledger `tests/known-failures-acpp.txt` |
| AdaptiveCpp multipass CUDA | AdaptiveCpp `develop` (25.10.0+), `BATCHLAS_ACPP_TARGETS=cuda:sm_89`, CUDA 12.8 | RTX 4090, `sm_89` | Built and tested; ledger `tests/known-failures-acpp-cuda.txt` ([multipass CUDA](../design/sycl-implementations.md#sycl-impl-cuda-aot)) |
| Apple GPU | AdaptiveCpp + local patches, Homebrew `llvm@21`, macOS 27 | M5 Max, Metal | float and complex<float> on the GPU ([below](#macos-apple-gpu-through-adaptivecpp-metal)) |
| CPU only | oneAPI `icpx` 2025.x | none | Builds; no NVIDIA support |
| AMD / ROCm, Windows | — | — | Untested |

Other NVIDIA architectures should work (the build targets the detected GPU). `main` has known test
failures: compare against `tests/known-failures.txt`, not `ctest`'s exit code.

## Presets

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8   # the presets set no job count; wider builds have OOM-killed
cmake --preset dev
cmake --build --preset dev
```

Each preset builds into `build-<preset>` beside `build/`.

| preset | builds |
| --- | --- |
| `dev` | `RelWithDebInfo` component libraries |
| `dev-tests` | library plus the full test suite |
| `fast-dev` | library plus the smoke tests |
| `benchmarks` | benchmarks (`batchlas_benchmarks`); `batchlas_tune` with `--target batchlas_tune` |
| `cuda` | `dev-tests` with `BATCHLAS_ALLOW_UNSAFE_ENV=OFF` (`BATCHLAS_ENABLE_CUDA=AUTO`: still CPU-only without a `[cuda:gpu]`) |
| `dev-gpu`, `dev-gpu-tests` | `BATCHLAS_CPU_TARGET=none`: GPU coverage only, half of every typed suite is not instantiated yet `ctest` is green |
| `release` | `Release`, one shared library, no tests |
| `acpp`, `acpp-tests` | built with AdaptiveCpp (`--acpp-targets=generic`) ([SYCL implementations](../design/sycl-implementations.md)) |
| `acpp-metal`, `acpp-metal-tests` | macOS Apple GPU ([below](#macos-apple-gpu-through-adaptivecpp-metal)) |

`BATCHLAS_ALLOW_UNSAFE_ENV` is `ON` in every preset except `cuda` and `release`. Test presets run
`ctest` serially. The pre-push gate is `scripts/gate.sh` (a `dev-tests` tree plus a vendor-free
tree).

## CMake options

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBATCHLAS_BUILD_TESTS=ON
cmake --build build -j8
```

| option | meaning |
| --- | --- |
| `BATCHLAS_BUILD_TESTS`, `BATCHLAS_TEST_TARGET_SET` | unit tests (default `ON` top-level, `OFF` as a subproject); `all` (default) or `smoke` |
| `BATCHLAS_BUILD_BENCHMARKS`, `BATCHLAS_BUILD_DEVICE_BLAS_BENCHMARKS` | benchmarks plus `batchlas_tune` with a GPU backend; the `device_blas_*` benchmarks (both default `OFF`) |
| `BATCHLAS_BUILD_PYTHON`, `BATCHLAS_PYTHON_INSTALL_DIR` | Python package (default `OFF`); install dir relative to the prefix (default `lib/pythonX.Y/site-packages`) |
| `BATCHLAS_BUILD_DOCS` | the `batchlas_docs` target |
| `BATCHLAS_SYCL_IMPL` | `AUTO` (default; `ACPP` when the compiler is `acpp`), `DPCPP` or `ACPP` (finds `acpp` on `PATH`, else `/opt/adaptivecpp/bin`) |
| `BATCHLAS_ENABLE_CUDA` | `AUTO` (default; when the SYCL runtime exposes a CUDA device), `ON` (configure fails without one) or `OFF` |
| `BATCHLAS_ENABLE_ROCM` | ROCm even without a detected AMD GPU (default `OFF`) |
| `BATCHLAS_ENABLE_METAL` | AdaptiveCpp only: `AUTO`, `ON` or `OFF` (`OFF` needs `ACPP_VISIBILITY_MASK=omp`) |
| `BATCHLAS_NVIDIA_ARCH`, `BATCHLAS_AMD_ARCH` | empty (default): the detected GPU; `sm_XX` / `gfxNNN` to cross-build |
| `BATCHLAS_CPU_TARGET` | `auto` (default), `native_cpu`, `spir64_x86_64` or `none` |
| `BATCHLAS_ENABLE_VENDOR_BLAS` | `OFF` builds the vendor-free library (no cuBLAS/cuSOLVER/cuSPARSE, roc*, LAPACKE/CBLAS; [status](../design/vendor-independence.md#vendor-free-status)). The three CUDA libraries are all-or-nothing |
| `BATCHLAS_DEVICE_FP64` | `AUTO` (default; `OFF` on Metal), `ON`, `OFF`: double types for `Backend::SYCL` |
| `BATCHLAS_MONOLITHIC_LIBRARY` | one `libbatchlas.so` (default `OFF`; `ON` and required on macOS; `release` sets it) |
| `BATCHLAS_HIDDEN_VISIBILITY` | `-fvisibility=hidden` on the monolithic library (default `ON`, `OFF` when tests are built) |
| `BATCHLAS_ALLOW_UNSAFE_ENV` | honour the `BATCHLAS_*` overrides that disable a safety check (default `OFF`) |
| `BATCHLAS_FP_CONTRACT` | `on` (default) or `off` ([below](#guide_build_fp_contract)) |
| `BATCHLAS_DISABLE_CUDA_FTZ` | `--ftz=false` in the CUDA JIT options (default `ON`) |
| `BATCHLAS_ACPP_TARGETS`, `BATCHLAS_ACPP_CUDA_PATH` | `--acpp-targets` (default `generic`); the CUDA 12.x toolkit for multipass CUDA (CUDA 13 is a configure error) |
| `BATCHLAS_ACPP_SLM_OPTIN` | AdaptiveCpp + CUDA: let kernels use over 48 KiB local memory (default `OFF`; [details](../design/sycl-implementations.md#sycl-impl-slm-budget)) |
| `BATCHLAS_OPENBLAS_ROOT` | OpenBLAS prefix providing LAPACKE and CBLAS (probed under Homebrew on macOS; empty on Linux) |
| `BATCHLAS_OPENBLAS_CORETYPE` | `OPENBLAS_CORETYPE` for ctest; set by the configure health check |
| `BATCHLAS_TEST_GPUS`, `BATCHLAS_TEST_GPU_SLOTS` | GPUs (`auto`, a count, `0`) and tests per GPU (default `2`) for `scripts/ctest_gpus.sh` |
| `BATCHLAS_KEEP_CUDA_INTERMEDIATES`, `BATCHLAS_SANITIZER_FRIENDLY_DEBUG` | PTX/SASS intermediates; unwind-friendly debug flags (default `OFF`) |

Build-speed options, all defaulting to the fast setting (agent guide §7): `BATCHLAS_USE_CCACHE`,
`BATCHLAS_CCACHE_SHARE_ACROSS_TREES`, `BATCHLAS_CCACHE_MAXSIZE` (`20G`), `BATCHLAS_CCACHE_BASEDIR`,
`BATCHLAS_SYCL_LINK_JOBS` (`auto`), `BATCHLAS_SYCL_GLOBAL_OFFSET` (`OFF`), `BATCHLAS_PRIVATE_TMPDIR`,
`BATCHLAS_LINK_DEPENDS_NO_SHARED`, `BATCHLAS_STRIP_RELWITHDEBINFO_G` (all `ON`) and
`CMAKE_BUILD_RPATH_USE_ORIGIN` (BatchLAS defaults it `ON`).

### Floating-point contraction {#guide_build_fp_contract}

The library is built with `-ffp-contract=on`: `a*b + c` in **one expression** may become one FMA; a
product stored and added in a later statement is never fused. DPC++ already defaults to `on`
(`icpx` also gets `-ffp-model=precise`); AdaptiveCpp defaults to `fast`, so BatchLAS passes
`-ffp-contract=<BATCHLAS_FP_CONTRACT>` to every acpp TU. Checked on one test kernel: DPC++ device IR
has one `llvm.fmuladd` under `on`, none under `off`; acpp multipass sm_120 PTX has 2 `fma.rn.f64`
with no flag, 1 under `on`, 0 under `off`.

`off` (for bitwise agreement with code built that way) is neither measured nor tested by CI. It
covers BatchLAS's own TUs only: templates instantiated from installed headers take the consumer's
flags, and vendor libraries contract as they were built.

### Device and toolkit detection {#guide_build_detection}

Configure needs only the compiler (`-DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++`, `icpx`, or
`-DBATCHLAS_SYCL_IMPL=ACPP`): no `setvars.sh`, `LD_LIBRARY_PATH` or `nvcc` on `PATH`.

- Devices come from the compiler's own `sycl-ls` (AdaptiveCpp: `acpp-info`), then `nvidia-smi`
  (DPC++: only when the toolchain ships `libur_adapter_cuda.so`).
- The CUDA toolkit is what `FindCUDAToolkit` finds, then AdaptiveCpp's `default-cuda-path`, then the
  newest NVIDIA HPC SDK toolkit; the result is cached as `CUDAToolkit_ROOT`.
- Binaries carry an RPATH to the DPC++ runtime, the vendor libraries and `$ORIGIN`.
- A self-built `intel/llvm --cuda` has no CPU adapter: outside oneAPI the tree is GPU-only
  (`No SYCL CPU device detected`).

## macOS: Apple GPU through AdaptiveCpp Metal

Kernels compile to generic SSCP IR and are JIT-translated to Metal at first launch (design and
limits: [AdaptiveCpp Metal](../design/sycl-implementations.md#sycl-impl-metal)).

- An Apple GPU `Queue` is `Backend::SYCL`: native kernels only, float and complex<float>; a double
  call throws `batchlas::unsupported`. hemm, herk and her2k have no route, nor does a shape whose
  every family needs more than Apple's 32 KiB of local memory.
- A CPU `Queue` is `Backend::NETLIB` over Homebrew OpenBLAS: all four types.

Needs the Command Line Tools and Homebrew, not Xcode. The script brews `cmake llvm@21 lld@21 libomp
boost openblas python`, checks out AdaptiveCpp `develop` at `9ec394ef`, applies
`scripts/macos/adaptivecpp-patches/*.patch` (each explained in that directory's README) and
installs to `~/opt/adaptivecpp` (`ACPP_PREFIX`; also `ACPP_SRC`, `ACPP_REAPPLY=1`, `SKIP_BREW=1`,
see the script header):

```bash
scripts/macos/build_adaptivecpp_metal.sh
~/opt/adaptivecpp/bin/acpp-info -l     # must list the OpenMP host device and the Apple GPU
rm -rf ~/.acpp/apps                    # after every reinstall: the JIT cache outlives the toolchain
cmake --preset acpp-metal-tests
cmake --build build-acpp-metal-tests --target gemm_tests -j8
ctest --preset acpp-metal-tests -R '^gemm_tests$'
```

Configure prints `Native SYCL GPU backend (Metal): TRUE; device fp64: OFF`. A full serial `ctest`
takes about an hour (first-launch JIT); its failure ledger is
`tests/known-failures-acpp-metal.txt`. macOS traps (adaptivity level, JIT cache, `DYLD_` paths,
`CPATH`): agent guide, "Known Pitfalls".

> **Warning:** every tuning table was measured on NVIDIA GPUs. An Apple GPU borrows the nearest
> table and warns once per op; its routes are untuned.

## Tests

```bash
cmake --preset dev-tests && cmake --build --preset dev-tests
ctest --test-dir build-dev-tests -LE 'slow|heavy' --output-on-failure    # the quick tier
scripts/b -B build-dev-tests stedc_tests                                  # one test binary
ctest --test-dir build-dev-tests -R '^stedc_tests(_heavy)?$'
```

Binaries in `BATCHLAS_HEAVY_TESTS` register `X` and `X_heavy` (label `heavy`); run both before
calling `X` green. `scripts/affected_tests.py --print-build-cmd --print-ctest-cmd` maps a diff to
tests; `scripts/ctest_gpus.sh` runs one test per GPU slot. Labels and the smoke set:
[tests/README.md](../../tests/README.md); the developer loop: agent guide §8.

## Benchmarks

`cmake --preset benchmarks && cmake --build --preset benchmarks`, or in any tree
`cmake --build build --target gemm_benchmark -j8`. With `BATCHLAS_BUILD_BENCHMARKS=ON` the
benchmarks join `all`, so leave it `OFF` in trees you iterate on. Measurement rules: agent guide §10.

## Python bindings

```bash
cmake -S . -B build -DBATCHLAS_BUILD_PYTHON=ON && cmake --build build -j8
PYTHONPATH="$PWD/build/python" python3 -c "import batchlas; print(batchlas.available_backends())"
PYTHONPATH="$PWD/build/python" python3 -m pytest python/tests
```

The package covers NumPy arrays, SciPy sparse wrappers, the `batchlas.linalg` submodule, the C++
exception hierarchy and route pins (`set_route`, `route`, `route_pin`). `cmake --install` puts it in
`<prefix>/<BATCHLAS_PYTHON_INSTALL_DIR>/batchlas`. Twelve self-checking notebooks are in
`python/examples/` (`python3 run_all.py` runs them all).

## Install traps

Consumer CMake recipe, the same-compiler rule and `-fsycl` flags: [the C++ API](../cpp-api.md);
@ref consumer_example is a runnable outside project.

- **The install is AOT-pinned to the GPU it was built for.** On another generation it fails with
  `No kernel named ... was found`; rebuild with `-DBATCHLAS_NVIDIA_ARCH=sm_XX`.
- **`cmake --install` never removes files.** A leftover `blas/`, `util/` or `internal/` in
  `<prefix>/include` shadows current headers; under a shared prefix those names may belong to
  another package.
- **`SOVERSION` 0, no release tag.** Every 0.x library is `.so.0` and the version check is
  `ExactVersion`, so pin to a commit. Link `BatchLAS::batchlas`;
  `find_package(BatchLAS CONFIG REQUIRED COMPONENTS ...)` is not supported.

## Repository layout

| directory | contents |
| --- | --- |
| `include/` | public C++ headers |
| `src/` | library; `src/ops/<op>/` per op ([layout](../extending.md)), `src/select/` the table lookup |
| `tuned/`, `tools/tune/` | per-device selection tables (@ref tuned_tables_readme) and their generator `batchlas_tune` (@ref tune_tool_readme) |
| `tests/`, `benchmarks/`, `python/` | GoogleTest suites; benchmarks (raw results in `benchmarks/results/`, Git LFS); Python bindings |
| `examples/`, `scripts/`, `docs/` | a minimal external consumer; campaign and CI helpers; this site |
| `playground/`, `evaluation/` | Python references cited by the band-reduction sources; tuning harnesses (not built) |
