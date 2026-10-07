# Running the BatchLAS tests

**Do not run the full suite on every edit.** It takes 15–20 minutes, and the
time is extremely lopsided — a handful of binaries hold nearly all of it, so a
scoped run gives you the same signal in seconds.

Pick the narrowest scope that covers what you changed:

| scope | command |
|---|---|
| one test case | `./build/tests/stedc_tests --gtest_filter='StedcTest/0.BatchedMatrices'` |
| one binary | `ctest -R '^stedc_tests$'` |
| one component | `ctest -L tridiag` |
| everything quick | `ctest -LE slow` (38 of 45) |
| everything | `ctest` |

`ctest -R` takes a *substring* regex — `-R syev` matches seven binaries. Anchor
it with `^...$` when you mean one.

## Running on every GPU at once

For anything wider than one binary, use `scripts/ctest_gpus.sh` with the same
arguments you would give ctest:

```bash
scripts/ctest_gpus.sh -LE slow                    # build/ by default
scripts/ctest_gpus.sh --test-dir build-vf -L eig  # another tree
```

Configure writes `<build>/ctest_resources.json` (one entry per
`nvidia-smi --list-gpus` device, `BATCHLAS_TEST_GPU_SLOTS` slots each, default
2). Every GPU test carries `RESOURCE_GROUPS gpus:1`, and `tests/ctest_gpu_env.sh`
restricts it to its slot's GPU through `CUDA_VISIBLE_DEVICES`. A pre-set
`CUDA_VISIBLE_DEVICES` list is indexed instead, and `ctest_gpus.sh` trims the
spec and `-j` to its length (`CUDA_VISIBLE_DEVICES=1 scripts/ctest_gpus.sh ...`
runs on GPU 1 only). `-DBATCHLAS_TEST_GPUS=<n>` overrides the count, `=0` turns
it off. Only a tree whose configure-time `sycl-ls` lists `[cuda:gpu]` gets a
spec: the isolation is `CUDA_VISIBLE_DEVICES`, so a Level Zero or HIP tree runs
plain serial `ctest`. `syev_cta_tests` and
`consumer_package_tests` are `RUN_SERIAL`. Plain `ctest` without
`--resource-spec-file` ignores the resource groups. Keep the
serial run for a `GTEST_OUTPUT=xml:<dir>/` gate (docs/ci.md): a route-pinned
rerun writes the same file name as its twin, and in parallel they can overlap.

On threadripper02 (4x RTX PRO 6000), `-LE slow` took 469 s serial with one GPU
visible, 126 s at 1 slot per GPU and 78 s at 2, with the same failing names in
every mode. This is the only box it was validated on. On the 2x RTX 4090 box
(125 GB, zero swap, two OOM kills under concurrency, `docs/ci.md`) keep the full
gate serial until it has been measured there. Also seen on threadripper02 only,
cause unknown: a process that sees all four GPUs runs several times slower than
one that sees one, so set `CUDA_VISIBLE_DEVICES` when running a binary by hand.

## Labels

Component labels, one per binary (see `CMakeLists.txt`):

`util`, `blas`, `ortho`, `tridiag`, `eig`, `sparse`

Run `ctest -L <component>` for the subsystem you touched. **If you changed
shared low-level code** — `Queue`, `Matrix`/`MatrixView`, the memory pool,
`sg_compat`/`sg_partition`, anything under `include/batchlas/util` — a component label is not enough;
run the full suite.

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

The `slow` label marks the binaries that dominate wall-clock. `ctest -LE slow`
is the best default for a broad-but-quick check. Keep the list in
`CMakeLists.txt` honest: if a test grows past ~15 s, label it `slow` rather
than letting it bloat the default run.

## Cutting runtime further

Two runtime env filters (implemented in `test_utils.hh`) work on any binary:

```bash
BATCHLAS_TEST_BACKEND=CUDA   ./build/tests/steqr_tests   # skip NETLIB/CPU
BATCHLAS_TEST_FLOAT_TYPE=float ./build/tests/steqr_tests # skip double/complex
```

`BATCHLAS_TEST_BACKEND=CUDA` is the single biggest no-code-change lever: the
NETLIB instantiations run the host O(n^3) reference solves, and on `steqr_tests`
they were 91% of the runtime. Both filters `GTEST_SKIP()` at runtime, so they
cut compute, not process startup.

## Writing tests that stay fast

Two rules cover most of it:

1. **Never combine large `n` with large `batch`.** `n` drives the algorithmic
   depth you actually want to test (D&C merge levels, panel count, bulge-chase
   sweeps). `batch` only multiplies that work. Cover them separately — large
   `n` at small batch, large batch at small `n`. Their product is where cost
   explodes for no added coverage.

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

The suite is **not green on `main`** (`lanczos`, `stedc`, `steqr` have failures;
`syev_cta` is flaky under `-j4`). Double-precision *CPU-only* failures are
usually the known-bad OpenBLAS Cooperlake `dgemm` kernel on this machine, not a
BatchLAS bug — CMake detects this and sets `OPENBLAS_CORETYPE` for tests run
through ctest. Always diff subtest *names* against a baseline rather than
trusting a pass/fail count.
