# Running the BatchLAS tests {#testing}

**Do not run the full suite on every edit.** It takes 15–20 minutes, and the
time is extremely lopsided — a handful of binaries hold nearly all of it, so a
scoped run gives you the same signal in seconds.

Pick the narrowest scope that covers what you changed:

| scope | command |
|---|---|
| one test case | `./build/tests/stedc_tests --gtest_filter='StedcTest/0.BatchedMatrices'` |
| one binary | `ctest -R '^stedc_tests$'` |
| one component | `ctest -L tridiag` |
| everything quick | `ctest -LE slow` (65 of 70 tests, about 95 s) |
| everything | `ctest` (70 tests) |

The counts are the ones measured on the Primary machine under the `dev-tests`
preset and recorded in [the CI page](../docs/ci.md#running-all-of-it-locally);
they move as binaries are added, so treat them as a sanity check, not a gate.

`ctest -R` takes a *substring* regex — `-R syev` matches every `syev*` binary
(`syev_tests`, `syevx_tests`, `syev_cta_tests`, ...). Anchor it with `^...$`
when you mean one.

## Labels

Component labels, one per binary (see `CMakeLists.txt`):

`util`, `blas`, `ortho`, `tridiag`, `eig`, `sparse`

Run `ctest -L <component>` for the subsystem you touched. **If you changed
shared low-level code** — `Queue`, `Matrix`/`MatrixView`, the memory pool,
`sg_compat`/`sg_partition`, anything under `include/batchlas/util` — a component label is not enough;
run the full suite.

The `slow` label marks the binaries that dominate wall-clock
(`BATCHLAS_SLOW_TESTS` in `tests/CMakeLists.txt`: `stedc_tests`, `steqr_tests`,
`sytrd_sb2st_tests`, `gesvd_tests`, plus `consumer_package_tests`).
`ctest -LE slow` is the best default for a broad-but-quick check, but it is not
a pre-push gate: it never runs those five. Keep the list honest: if a test grows
past ~15 s, label it `slow` rather than letting it bloat the default run, and do
not leave a test there that no longer dominates — everything listed is invisible
to the iteration run.

## The sub-group partition layer

`sg_partition_tests` covers `src/extensions/sg_partition/` (`SubGroupPartition<P, Masked>`
and every collective) on whatever backend the device pass selects. It is
header-only and links no BatchLAS library, so it builds on its own:
`cmake --build build --target sg_partition_tests`. Run it once per device,
since each has its own backend:

```bash
ONEAPI_DEVICE_SELECTOR=cuda:0     ./build/tests/sg_partition_tests  # NVPTX backend, SG = 32 only
ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build/tests/sg_partition_tests  # SPIR-V backend, SG = 8/16/32/64
```

The OpenCL CPU run needs a spir64 image (`-DBATCHLAS_CPU_TARGET=spir64_x86_64`;
the default `native_cpu` has sub-group size 1, so every case skips there). Sizes
a device lacks are skipped, not failed. A backend that breaks a collective under
divergence tends to hang rather than fail, hence the 300 s ctest timeout.

When a partition collective sits under a branch in a test, every lane of the
chunk must still reach it: call it unconditionally and branch on the result.

## The shared fixture: `test_utils::BatchLASTest`

Typed suites derive from `test_utils::BatchLASTest<Config>` in
`tests/test_utils.hh`, where `Config` carries `ScalarType` and `BackendVal`. The
type lists come from `backend_types<Config>` (every compiled backend × `float`,
`double`, `std::complex<float>`, `std::complex<double>`),
`backend_types_filtered<Config, IncludeComplex>` and `backend_types_complex<Config>`.

`SetUp()` does, in order:

1. skip if `BATCHLAS_TEST_BACKEND` filters the backend out;
2. skip if `BATCHLAS_TEST_FLOAT_TYPE` filters the scalar type out;
3. build `this->ctx`, a `std::shared_ptr<Queue>` **pinned to the config's
   backend** — a GPU queue for CUDA/ROCm/MKL, `Device("cpu")` for NETLIB — so a
   test body can call `syev(*this->ctx, ...)` without naming the backend and
   still exercise the queue-dispatch path callers use. A missing GPU, a
   `sycl::exception` with `errc::runtime` / `errc::feature_not_supported`, or a
   non-SYCL construction failure is a `GTEST_SKIP`; any other SYCL error is
   rethrown and fails the case.

`TearDown()` waits on the queue. A suite should not declare its own `ctx` or
`SetUp()`; add per-suite state in the derived fixture.

**The NETLIB instantiations exist only when CPU device code is compiled**
(`BATCHLAS_HAS_CPU_TARGET`). Under `dev-gpu-tests`, or any build without a CPU
SYCL target, they vanish from the type lists and the typed-test *indices shift*
(`SteqrTest/0` becomes float/CUDA). See
[CPU SYCL target detection](../docs/design/build-cpu-target-detection.md).

## Cutting runtime further

The two runtime filters that `BatchLASTest` applies work on any binary:

```bash
BATCHLAS_TEST_BACKEND=CUDA     ./build/tests/steqr_tests   # skip NETLIB/CPU
BATCHLAS_TEST_FLOAT_TYPE=float ./build/tests/steqr_tests   # skip double/complex<double>
```

| variable | accepted values (case-insensitive) | runs |
|---|---|---|
| `BATCHLAS_TEST_BACKEND` | `CUDA`, `ROCM`, `MKL`, `NETLIB` | only that backend's instantiations |
| `BATCHLAS_TEST_FLOAT_TYPE` | `float` | `float` and `std::complex<float>` |
| | `double` | `double` and `std::complex<double>` |
| | `complex` | both complex types |

Both can be combined, and both work through `ctest` as well
(`BATCHLAS_TEST_BACKEND=CUDA ctest -L eig`).

`BATCHLAS_TEST_BACKEND=CUDA` is the single biggest no-code-change lever: the
NETLIB instantiations run the host O(n^3) reference solves, and on `steqr_tests`
they were 91% of the runtime. Timings recorded when the filters were introduced
(CUDA only vs all backends): `trmm_tests` ~1.5 s vs ~58 s, `stedc_tests` ~7.6 s
vs ~34 s, `trsm_tests` ~1.2 s vs ~5.4 s, `gemv_tests` ~0.4 s vs ~1.8 s. Those
are historical and the suites have changed since; the ratio, not the seconds,
is the point.

Traps:

- **Both filters are `GTEST_SKIP()`s.** The case list and the pass count look
  the same; the filtered coverage is silently gone. They cut compute, not
  process startup.
- **An unrecognised value skips everything.** `BATCHLAS_TEST_BACKEND=cuda0`
  prints a warning and skips every case; an unrecognised
  `BATCHLAS_TEST_FLOAT_TYPE` (e.g. `single`) skips every case *without* a
  warning. A run where everything skipped is not a pass.

## Writing tests that stay fast

Two rules cover most of it:

1. **Never combine large `n` with large `batch`.** `n` drives the algorithmic
   depth you actually want to test (D&C merge levels, panel count, bulge-chase
   sweeps). `batch` only multiplies that work. Cover them separately — large
   `n` at small batch, large batch at small `n`. Their product is where cost
   explodes for no added coverage. (A shared-local-memory kernel still needs one
   saturating-batch case at small `n`; see `AGENTS.md` §8.)

2. **Watch the reference solve.** A test that builds `Matrix::Zeros(n, n, batch)`
   and runs `syev` / `ritz_values` / `netlib_ref_eigs_dense` over it pays
   O(n^3)·batch, on the *host* for the NETLIB instantiations. That reference,
   not the kernel under test, is usually what makes a test slow.

Also: if every test body in a file starts with
`using float_type = typename base_type<T>::type;` and computes only in
`float_type`, the complex instantiations from `backend_types<Config>` are
bit-identical re-runs of the real ones. Use
`backend_types_filtered<Config, false>` instead and halve the file for free.

## Note on the baseline

The suite is **not green on `main`**. The accepted failures are listed, by test
*name* and pinned type parameter, in `tests/known-failures.txt` (at the time of
writing: two `lanczos_tests` cases and `steqr_tests`'
`StressExtremeMagnitudesN32` for two types), and CI diffs every run against that
ledger — see [the CI page](../docs/ci.md#the-known-failures-workflow).
`syev_cta_tests` is flaky under `ctest -j2`/`-j4` and `sytrd_blocked_tests` also
flakes on untouched `main`. Double-precision *CPU-only* failures are usually the
known-bad OpenBLAS Cooperlake `dgemm` kernel on this machine, not a BatchLAS
bug — CMake detects this and sets `OPENBLAS_CORETYPE` for tests run through
ctest (a bare `./build/tests/foo` does not). Always diff subtest *names* against
a baseline rather than trusting a pass/fail count.
