# Consuming BatchLAS from an outside CMake project {#consumer_example}

A standalone project that calls `find_package(BatchLAS CONFIG REQUIRED)`, runs a batched
`gemm` and checks the result against a hand-computed reference. It is never
`add_subdirectory()`'d: everything it needs must come from the install prefix.

```
consumer/
  CMakeLists.txt                     # the whole recipe
  main.cc                            # batched gemm + reference, layout documented
  decoy_include/blas/enums.hh              # include-collision probe, see below
  decoy_include/util/workspace.hh          #   "
  decoy_include/internal/ormqr_blocked.hh  #   "
```

`../consumer_test.sh` runs this from CTest: it installs to a temporary prefix, then configures,
builds, runs and checks. To run it by hand:

```bash
examples/consumer_test.sh --build-dir build --compiler /opt/dpcpp-cuda/bin/clang++
```

## Three requirements

**1. `-DCMAKE_CXX_COMPILER` must be the SYCL compiler that built BatchLAS.** Set it for the whole
project and for every project that links BatchLAS:

```bash
cmake -S examples/consumer -B build-consumer \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_PREFIX_PATH="$HOME/inst"
```

- A consumer that submits its own kernels needs `-fsycl`. This example adds it
  (`BATCHLAS_CONSUMER_USE_FSYCL`, on by default). The exported package does not force SYCL flags
  onto consumer translation units.
- BatchLAS's public templates carry C++20 `requires` clauses. Clang mangles the constraint into the
  symbol name; GCC 11 and Clang 15 and older do not. Headers compile under `g++`, then the link fails
  with `undefined reference to batchlas::Matrix<float, ...>::Matrix<float, ...>(int, int, int, int, int)`,
  for a symbol that `nm` shows as present. There is no partial adoption: the consuming project
  must use this compiler.

`find_package(BatchLAS CONFIG REQUIRED)` succeeds under the wrong compiler, so the failure appears
later, in code that does not mention BatchLAS.

**2. Call `ctx.wait()` before reading results.** Every call is asynchronous and returns an `Event`.
Reading the output early returns the buffer's old contents (usually the zeros it was initialised
with). That is a silently wrong answer, not a crash. `main.cc` marks the line this depends on.

Pointers passed to a `MatrixView` must be device-accessible (USM). A `std::vector<float>` compiles,
and the entry points that take their backend from the `Queue` throw `std::invalid_argument` naming
the argument. A CPU backend reads host memory without complaint, so run the check on the device you
ship on. The owning `Matrix` allocates USM shared memory, so host-side fills are fine.

**3. `LD_LIBRARY_PATH` must cover the DPC++ runtime.**

```bash
LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:"$HOME/inst/lib" ./build-consumer/hello_batched_gemm
```

The installed libraries need `libsycl.so.9` and carry no RUNPATH that finds it, so add the
compiler's runtime directory (`<dir of the compiler>/../lib`). CMake gives the executable an RPATH
for the BatchLAS libraries it links. Adding `<prefix>/lib` is harmless if the package is relocated.

## Layout contract

Dense matrices are column-major and batched by stride. Element `(i, j)` of batch item `b` is at

```cpp
view.data_ptr()[b * view.stride() + j * view.ld() + i]
```

`ld()` is at least `rows`, and `stride()` at least `ld() * cols`. `Matrix(rows, cols, batch)` packs
`ld() == rows` and `stride() == rows * cols`. A matrix built over someone else's buffer can have
larger values; `main.cc` reads both layouts through one `at()` helper.

## The decoy headers

`decoy_include/` holds three files: `blas/enums.hh`, `util/workspace.hh` and
`internal/ormqr_blocked.hh`. Each is a bare `#error` with its own sentinel, and none is meant to
compile. They stand in for consumer headers that use the top-level names a library could claim.

The target's own include directories are searched before those propagated by imported targets
(`-isystem`, searched last). An unprefixed `#include <util/workspace.hh>` in a BatchLAS header would
therefore pick up the consumer's file, and the probe fails.

`consumer_test.sh` builds the example again with `-DBATCHLAS_CONSUMER_DECOY=ON` and fails on anything
but a clean build. A `BATCHLAS_DECOY_*_SHADOWED` sentinel is a regression. The script also checks
that nothing is installed at `<prefix>/include/{blas,util,internal}`.
