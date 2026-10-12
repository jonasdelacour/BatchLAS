# Running the BatchLAS tests {#testing}

> **Status:** current · counts measured under the `dev-tests` preset on the primary machine

Run the narrowest scope that covers your change. The full suite takes 15-20 minutes and a few
binaries hold nearly all of it.

| scope | command |
|---|---|
| one test case | `./build/tests/stedc_tests --gtest_filter='StedcTest/0.BatchedMatrices'` |
| one binary | `ctest -R '^stedc_tests$'` |
| one component | `ctest -L tridiag` |
| everything quick | `ctest -LE slow` (65 of 70 tests, about 95 s) |
| everything | `ctest` (70 tests) |

Counts move as binaries are added (see [the CI page](../docs/ci.md#running-all-of-it-locally)).
`ctest -R` is a substring regex: `-R syev` matches `syev_tests`, `syevx_tests`, `syev_cta_tests`.
Anchor with `^...$` for one binary.

## Running on every GPU at once

For anything wider than one binary, use `scripts/ctest_gpus.sh` with ctest's arguments:

```bash
scripts/ctest_gpus.sh -LE slow                    # build/ by default
scripts/ctest_gpus.sh --test-dir build-vf -L eig  # another tree
```

- Configure writes `<build>/ctest_resources.json`: one entry per `nvidia-smi --list-gpus` device,
  `BATCHLAS_TEST_GPU_SLOTS` slots each (default 2).
- Every GPU test has `RESOURCE_GROUPS gpus:1`; `tests/ctest_gpu_env.sh` sets
  `CUDA_VISIBLE_DEVICES` to its slot's GPU.
- A pre-set `CUDA_VISIBLE_DEVICES` list is indexed instead, and the script trims the spec and
  `-j` to its length (`CUDA_VISIBLE_DEVICES=1 scripts/ctest_gpus.sh ...` runs on GPU 1 only).
- `-DBATCHLAS_TEST_GPUS=<n>` overrides the count; `=0` disables.
- Only a tree whose configure-time `sycl-ls` lists `[cuda:gpu]` gets a spec; Level Zero and HIP
  trees run plain serial `ctest`.
- `syev_cta_tests` and `consumer_package_tests` are `RUN_SERIAL`. Plain `ctest` without
  `--resource-spec-file` ignores the resource groups.
- Use the serial run for a `GTEST_OUTPUT=xml:<dir>/` gate (docs/ci.md): a route-pinned rerun
  writes the same file name as its twin, and in parallel they can overlap.

Measured `-LE slow` on threadripper02 (4x RTX PRO 6000), the only box validated:

| mode | time |
|---|---|
| serial, one GPU visible | 469 s |
| 1 slot per GPU | 126 s |
| 2 slots per GPU | 78 s |

Failing names are identical in all modes. On the 2x RTX 4090 box (125 GB, zero swap, two OOM
kills under concurrency; `docs/ci.md`) keep the full gate serial. On threadripper02 a process
that sees all four GPUs runs several times slower than one that sees one (cause unknown); set
`CUDA_VISIBLE_DEVICES` when running a binary by hand.

## Labels

Component labels, one per binary (see `CMakeLists.txt`): `util`, `blas`, `ortho`, `tridiag`,
`eig`, `sparse`. Run `ctest -L <component>` for the subsystem you touched. After changing shared
low-level code (`Queue`, `Matrix`/`MatrixView`, the memory pool, `sg_compat`/`sg_partition`,
anything under `include/batchlas/util`) run the full suite.

The `slow` label (`BATCHLAS_SLOW_TESTS` in `tests/CMakeLists.txt`) marks `stedc_tests`,
`steqr_tests`, `sytrd_sb2st_tests`, `gesvd_tests` and `consumer_package_tests`. `ctest -LE slow`
is a good broad check but not a pre-push gate. Label a test `slow` when it grows past about 15 s,
and remove the label when it no longer dominates.

## The sub-group partition layer

`sg_partition_tests` covers `src/extensions/sg_partition/` (`SubGroupPartition<P, Masked>` and
every collective). It is header-only, links no BatchLAS library, and builds alone:
`cmake --build build --target sg_partition_tests`. Run it once per device:

```bash
ONEAPI_DEVICE_SELECTOR=cuda:0     ./build/tests/sg_partition_tests  # NVPTX backend, SG = 32 only
ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build/tests/sg_partition_tests  # SPIR-V backend, SG = 8/16/32/64
```

- The OpenCL CPU run needs `-DBATCHLAS_CPU_TARGET=spir64_x86_64`; the default `native_cpu` has
  sub-group size 1 and every case skips.
- Sizes a device lacks are skipped, not failed.
- A collective broken under divergence tends to hang, hence the 300 s ctest timeout.
- A collective under a branch in a test must still be reached by every lane of the chunk: call it
  unconditionally and branch on the result.

## The shared fixture: `test_utils::BatchLASTest`

Typed suites derive from `test_utils::BatchLASTest<Config>` (`tests/test_utils.hh`); `Config`
carries `ScalarType` and `BackendVal`. Type lists: `backend_types<Config>` (every compiled backend
x `float`, `double`, `std::complex<float>`, `std::complex<double>`),
`backend_types_filtered<Config, IncludeComplex>`, `backend_types_complex<Config>`.

`SetUp()`, in order:

1. skips if `BATCHLAS_TEST_BACKEND` filters the backend out;
2. skips if `BATCHLAS_TEST_FLOAT_TYPE` filters the scalar type out;
3. builds `this->ctx`, a `std::shared_ptr<Queue>` pinned to the config's backend (GPU queue for
   CUDA/ROCm, `Device("cpu")` for NETLIB). A missing GPU, a `sycl::exception` with
   `errc::runtime` / `errc::feature_not_supported`, or a non-SYCL construction failure is a
   `GTEST_SKIP`; any other SYCL error is rethrown and fails the case.

`TearDown()` waits on the queue. A suite must not declare its own `ctx` or `SetUp()`; put
per-suite state in the derived fixture.

> **Note:** NETLIB instantiations exist only when CPU device code is compiled
> (`BATCHLAS_HAS_CPU_TARGET`). Under `dev-gpu-tests` they vanish from the type lists and typed-test
> indices shift (`SteqrTest/0` becomes float/CUDA). See
> [CPU SYCL target detection](../docs/design/build-cpu-target-detection.md).

## Cutting runtime further

Two runtime filters from `BatchLASTest` work on any binary, directly or through `ctest`:

```bash
BATCHLAS_TEST_BACKEND=CUDA     ./build/tests/steqr_tests   # skip NETLIB/CPU
BATCHLAS_TEST_FLOAT_TYPE=float ./build/tests/steqr_tests   # skip double/complex<double>
```

| variable | accepted values (case-insensitive) | runs |
|---|---|---|
| `BATCHLAS_TEST_BACKEND` | `CUDA`, `ROCM`, `NETLIB` | only that backend's instantiations |
| `BATCHLAS_TEST_FLOAT_TYPE` | `float` | `float` and `std::complex<float>` |
| | `double` | `double` and `std::complex<double>` |
| | `complex` | both complex types |

The two combine. `BATCHLAS_TEST_BACKEND=CUDA` is the largest lever: NETLIB instantiations run host
O(n^3) reference solves (91% of `steqr_tests` runtime). Original CUDA-only vs all-backend timings
(suites have changed since; the ratio is the point): `trmm_tests` ~1.5 s vs ~58 s, `stedc_tests`
~7.6 s vs ~34 s, `trsm_tests` ~1.2 s vs ~5.4 s, `gemv_tests` ~0.4 s vs ~1.8 s.

> **Warning:** both filters are `GTEST_SKIP()`s. Case list and pass count look unchanged while the
> filtered coverage is gone, and startup is not cut. An unrecognised value skips every case:
> `BATCHLAS_TEST_BACKEND=cuda0` warns, `BATCHLAS_TEST_FLOAT_TYPE=single` skips silently. A run
> where everything skipped is not a pass.

## Writing tests that stay fast

1. **Do not combine large `n` with large `batch`.** `n` drives algorithmic depth (D&C merge
   levels, panel count, bulge-chase sweeps); `batch` only multiplies work. Cover large `n` at
   small batch and large batch at small `n`. A shared-local-memory kernel still needs one
   saturating-batch case at small `n` (`docs/developer/agent-guide.md` §8).
2. **Watch the reference solve.** `Matrix::Zeros(n, n, batch)` plus `syev` / `ritz_values` /
   `batchlas::verify::eigenvalues` cost O(n^3)·batch on the host for NETLIB instantiations; the
   reference, not the kernel, usually makes a test slow.
3. If every test body starts with `using float_type = typename base_type<T>::type;` and computes
   only in `float_type`, the complex instantiations are bit-identical re-runs. Use
   `backend_types_filtered<Config, false>`.

## Note on the baseline

The suite is not green on `main`. `tests/known-failures.txt` lists accepted failures by test name
and pinned type parameter (two `lanczos_tests` cases and `steqr_tests`'
`StressExtremeMagnitudesN32` for two types). CI diffs every run against that ledger; see
[the CI page](../docs/ci.md#the-known-failures-workflow).

- `syev_cta_tests` is flaky under `ctest -j2`/`-j4`; `sytrd_blocked_tests` also flakes on
  untouched `main`.
- Double-precision CPU-only failures are usually the broken OpenBLAS Cooperlake `dgemm` kernel
  on this machine. CMake detects it and sets `OPENBLAS_CORETYPE` for tests run through ctest; a
  bare `./build/tests/foo` does not.
- Compare subtest names against a baseline, not pass/fail counts.
