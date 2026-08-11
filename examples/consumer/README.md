# Consuming BatchLAS from an outside CMake project

A standalone project — its own `cmake_minimum_required`, its own `project()`,
`find_package(BatchLAS CONFIG REQUIRED)` and nothing else — that runs a batched
`gemm` and checks the result against a hand-computed reference. It is not part
of the BatchLAS build; it is deliberately never `add_subdirectory()`'d, because
the point is that everything it needs must come out of the *install prefix*.

```
consumer/
  CMakeLists.txt                     # the whole recipe, ~15 lines of it real
  main.cc                            # batched gemm + reference, layout documented
  decoy_include/blas/enums.hh              # include-collision probe, see below
  decoy_include/util/workspace.hh          #   "
  decoy_include/internal/ormqr_blocked.hh  #   "
```

`../consumer_test.sh` drives this from CTest: install to a temporary prefix,
configure, build, run, check. Run it by hand the same way:

```bash
examples/consumer_test.sh --build-dir build --compiler /opt/dpcpp-cuda/bin/clang++
```

## Three things that are load-bearing

**1. `-DCMAKE_CXX_COMPILER` must be the SYCL compiler that built BatchLAS.**

```bash
cmake -S examples/consumer -B build-consumer \
      -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++ \
      -DCMAKE_PREFIX_PATH="$HOME/inst"
```

Not just for this target — for the whole project, and for every project that
links it. Two independent reasons:

- BatchLAS's headers are compiled by a SYCL compiler in its own build, and a
  consumer that submits kernels of its own needs `-fsycl`. This example adds it
  itself (`BATCHLAS_CONSUMER_USE_FSYCL`, on by default); the exported package
  does not force SYCL flags onto consumer translation units.
- BatchLAS's public templates carry C++20 `requires` clauses, and Clang mangles
  the constraint into the symbol name while GCC 11 (and Clang ≤ 15) does not.
  Headers compile cleanly under `g++` and then the link fails with
  `undefined reference to batchlas::Matrix<float, ...>::Matrix<float, ...>(int, int, int, int, int)`
  for a symbol `nm` will happily show you is present. There is no partial
  adoption: the consuming project moves to this compiler or it does not link.

`find_package(BatchLAS CONFIG REQUIRED)` succeeds under the wrong compiler, so
the failure lands later, in a place that does not mention BatchLAS.

**2. `ctx.wait()` before reading results.**

Every call is asynchronous and returns an `Event`. Reading the output matrix
without synchronising gives you whatever was in the buffer beforehand — usually
the zeros you initialised it with, i.e. a silently wrong answer rather than a
crash. `main.cc` marks the one line this depends on.

The related trap, also in `main.cc`: pointers handed to a `MatrixView` must be
device-accessible (USM). A `std::vector<float>` compiles, returns correct
results on a CPU backend, and aborts the process with
`CUDA_ERROR_ILLEGAL_ADDRESS` on a GPU one. The owning `Matrix` allocates USM
shared memory, so filling it from the host is fine.

**3. `LD_LIBRARY_PATH` must cover the DPC++ runtime.**

```bash
LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:"$HOME/inst/lib" ./build-consumer/hello_batched_gemm
```

The installed libraries record `DT_NEEDED libsycl.so.9` and carry no RUNPATH
that finds it, so the loader needs to be told where the compiler's runtime
lives (`<dir of the compiler>/../lib`). CMake gives the executable an RPATH
covering the BatchLAS libraries it linked, but adding `<prefix>/lib` is free
insurance if the binary is moved or the package is relocated.

## Layout contract

Dense matrices are **column-major** and batched by a stride. Element `(i, j)` of
batch item `b` is at

```cpp
view.data_ptr()[b * view.stride() + j * view.ld() + i]
```

`ld()` (leading dimension, `>= rows`) and `stride()` (`>= ld * cols`) are chosen
by the library when it allocates, and are not required to equal `rows` and
`rows * cols`. `main.cc` routes every access through one `at()` helper for
exactly that reason.

## The decoy headers

`decoy_include/` holds three files — `blas/enums.hh`, `util/workspace.hh` and
`internal/ormqr_blocked.hh` — none of which is a BatchLAS header and none of
which is meant to compile. Each is a bare `#error` with its own sentinel. They
stand in for an ordinary consumer that happens to own headers by those names,
which are exactly the three top-level directories BatchLAS used to install into.

The mechanism is CMake's include ordering: a target's own include directories
come *before* anything an imported target propagates (imported targets propagate
theirs as `-isystem`, searched last). So if any installed BatchLAS header still
spelled a cross-include `#include <util/workspace.hh>`, it would get the
consumer's file, and the error message would name files the consumer has never
seen. Every one of the three names is on the real include chain reached from
`<batchlas/blas/linalg.hh>`, so a regression in any of them fires.

Since the public headers moved to `include/batchlas/` and are spelled
`<batchlas/...>`, the decoy build must now succeed. `consumer_test.sh` builds
the example a second time with `-DBATCHLAS_CONSUMER_DECOY=ON` and **fails** on
anything but a clean build: a `BATCHLAS_DECOY_*_SHADOWED` sentinel in the log is
reported as a regression, and any other failure means the probe stopped probing.

That is the second half of the guarantee. The first half — that nothing is
installed at `<prefix>/include/{blas,util,internal}` at all, plus the matching
positive check that everything *is* at `<prefix>/include/batchlas/` — is
asserted separately against the install tree. Both are needed: the include root
was already clean back when every internal spelling was still unprefixed.
