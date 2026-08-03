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

## Labels

Component labels, one per binary (see `CMakeLists.txt`):

`util`, `blas`, `ortho`, `tridiag`, `eig`, `sparse`

Run `ctest -L <component>` for the subsystem you touched. **If you changed
shared low-level code** — `Queue`, `Matrix`/`MatrixView`, the memory pool,
`sg_compat`, anything under `include/util` — a component label is not enough;
run the full suite.

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
