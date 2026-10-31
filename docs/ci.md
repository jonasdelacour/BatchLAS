# CI: what it proves, and how to stand the runner up {#dev_ci}

> **Status:** current · self-hosted runner: RTX 4090 (sm_89), CUDA 13.2, Ubuntu 22.04

Static checks and the AdaptiveCpp compile jobs run on GitHub-hosted runners. The GPU jobs run on one
self-hosted workstation, only when the repository variable `GPU_RUNNER` is `true`. CI gates one
configuration (CUDA DPC++ on `sm_89`), no other row of
[Tested platforms](guide/build.md#tested-platforms). `.github/workflows/ci.yml` is the authority.

## The jobs

Triggers: push to `main`, pull requests, `workflow_dispatch` (input `run_nightly`), cron `17 3 * * *`.

| Job | Runs on | When | Does |
| --- | --- | --- | --- |
| `cmake-lint` | `ubuntu-latest` | always | `check_cmake_syntax.py`; `cmake --list-presets=all` (no configure) |
| `exported-package` | `ubuntu-latest` | always | `check_exported_package.py --source`: no host paths in the export |
| `public-headers` | `ubuntu-latest` | always | `check_public_headers.py` (no `src/` includes), `check_no_unprefixed_includes.py`, `check_no_global_names.py` |
| `comment-density` | `ubuntu-latest` | always | `check_comment_density.py` (18% per file), `check_evidence_anchors.py`, `compare_failures.py --self-test`, `check_markdown_locations.py`, each after its `--self-test` |
| `lfs-pointers` | `ubuntu-latest` | always | `check_lfs_pointers.py`: no raw content under an LFS path |
| `tuned-tables` | `ubuntu-latest` | always | `check_tuned_tables.py --self-test`, then the stale-table report (warning only) |
| `tuned-tables-ledgers` | `ubuntu-latest` | always | Fetches ledger LFS objects, `sweep_to_table.py --self-test` and `--check` ([generation and the check](design/tuning-data.md#tuning-data-generation-and-check)) |
| `docs` | `ubuntu-latest` | always | `gen_db_pages.py --self-test`, pinned Doxygen 1.18.0, `BATCHLAS_DOCS_STRICT=1 sh scripts/build_docs.sh build/docs` (fails on any warning or dead cited anchor); uploads `docs-html` |
| `docs-deploy` | `ubuntu-latest` | push to `main` | Publishes the Pages artifact; fails until an admin sets Settings → Pages → Source to "GitHub Actions" |
| `acpp-compile` | `ubuntu-24.04` | always | AdaptiveCpp 25.10 generic target, CUDA off; compiles every target. 90 min |
| `acpp-cuda-consumer` | `ubuntu-24.04` | always | Vendor-free BatchLAS for `cuda:sm_89` via acpp, installed, consumed as C++23. Nothing runs. 120 min |
| `gpu-build-test` | `[self-hosted, linux, x64, cuda]` | push, non-fork PR | Configure with `-DBATCHLAS_ENABLE_CUDA=ON`, assert `nvidia_gpu_sm_89`, `cmake --build --preset dev-tests`, `ctest -LE slow`, `compare_failures.py --min-tests 95`, `cmake --install` + `run_local_checks.sh <prefix>`. 180 min |
| `gpu-nightly-full` | same | cron, `run_nightly` | Full `ctest`, `compare_failures.py --min-tests 100 --require consumer_package_tests`; then a vendor-free build + `ctest -LE slow` with `continue-on-error` (reports, no ledger). 360 min |

Both GPU jobs share concurrency group `batchlas-gpu-runner` (`cancel-in-progress: false`) and refuse
fork PRs (the runner executes PR content as the box's user). Doxygen version bumps:
[Building the site](developer/documentation.md#building-the-site).

**Not covered:** AMD/ROCm; CPU-only DPC++/`icpx` builds (no host-only mode exists; reasoning at the
foot of `ci.yml`); macOS, Windows; Python bindings and benchmarks under DPC++; performance; other
NVIDIA architectures and CUDA versions; multi-GPU; flakiness; anything a skipped test would cover.

### The acpp-compile job

See [Two SYCL implementations](design/sycl-implementations.md). `--acpp-targets=generic` needs no GPU
or CUDA toolkit and has no device link; it catches DPC++-only spellings.

| Step | What |
| --- | --- |
| Toolchain | Ubuntu 24.04 `clang-20 llvm-20-dev libclang-20-dev libomp-20-dev`, netlib `liblapacke-dev liblapack-dev libblas-dev`, `ccache` |
| AdaptiveCpp | `.github/ci/build_adaptivecpp.sh`: tag `v25.10.0` (must resolve to `9f842c70`), OpenMP + generic only; cached on tag, commit, `llvm-20-dev` version, script hash |
| BatchLAS | `.github/ci/acpp_compile.sh`: preset `acpp-tests`, `Release`, `-Werror=unknown-attributes`, `BATCHLAS_ENABLE_CUDA=OFF`, benchmarks on; asserts `Using AdaptiveCpp targets: generic`, `Using AdaptiveCpp for SYCL:`, `Found LAPACKE:`; builds `all` |
| Completeness | `.github/ci/check_built_tests.py`: every registered ctest executable exists |

Blind spot: with CUDA off, `src/backends/{cublas,cusolver,cusparse}.cc`, `tools/tune`,
`tune_race_gpu_tests` and every `#if BATCHLAS_HAS_CUDA_BACKEND` block are not compiled.

```bash
.github/ci/build_adaptivecpp.sh "$HOME/acpp-ci"          # or reuse an existing 25.10 install
JOBS=32 .github/ci/acpp_compile.sh "$HOME/acpp-ci" build/acpp-ci
```

### The acpp-cuda-consumer job {#ci_acpp_cuda_consumer_job}

`acpp -std=c++23 --acpp-targets=cuda:sm_89` as an outside project, against CUDA 12.8 from NVIDIA's apt
repo (acpp 25.10's clang rejects CUDA 13; see
[multipass CUDA](design/sycl-implementations.md#sycl-impl-cuda-aot)). AdaptiveCpp is built with
`ACPP_CUDA_PATH`; BatchLAS with `BATCHLAS_ACPP_TARGETS=cuda:sm_89`, `BATCHLAS_ENABLE_VENDOR_BLAS=OFF`,
target `batchlas_components`. Asserts: `backend_config.h` has `BATCHLAS_HAS_CUDA_BACKEND 1` and 0 for
`CUBLAS`, `CUSOLVER`, `CUSPARSE`, `LAPACKE`, `CBLAS`; `readelf -d` of every installed `libbatchlas*.so`
`NEEDED`s no LAPACKE, CBLAS, BLAS or CUDA math library; the `examples/consumer` verbose build shows
`-std=c++23` and `--acpp-targets=cuda:sm_89`.

## Standing up the self-hosted runner

> **Warning:** no runner has yet been observed working end to end. Expect to iterate on the first run.

Get a one-hour registration token, the version and the URL from Settings → Actions → Runners → New
self-hosted runner.

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v<VERSION>/actions-runner-linux-x64-<VERSION>.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/jonasdelacour/BatchLAS --token <REGISTRATION_TOKEN> \
  --name batchlas-4090 --labels self-hosted,linux,x64,cuda --work _work --unattended --replace
sudo ./svc.sh install jonaslacour     # as jonaslacour, not root: HOME must be /home/jonaslacour for ccache
sudo ./svc.sh start && sudo ./svc.sh status
```

Use the system service, not `./run.sh` (inherits the login shell, so it hides the environment trap)
and not a `systemd --user` unit (`Linger=no`).

## The environment trap

The service gets `/etc/environment` only: no `/opt/dpcpp-cuda`, no CUDA, no `LD_LIBRARY_PATH`. Under
`BATCHLAS_ENABLE_CUDA=AUTO` that configures and passes `ctest` as a CPU-only build, so `ci.yml` records
the toolchain (`clang++ --version`, `sycl-ls`, `nvidia-smi -L`), pins `-DBATCHLAS_ENABLE_CUDA=ON` and
greps the configure log for `nvidia_gpu_sm_89`. Keep all three.

| Variable | Value |
| --- | --- |
| `PATH` | prepend `/opt/dpcpp-cuda/bin`, `/usr/local/cuda-13.2/bin` |
| `LD_LIBRARY_PATH` | `/opt/dpcpp-cuda/lib` (mandatory), `/usr/local/cuda-13.2/lib64`, `/opt/intel/oneapi-tbb-2022.3.0/lib/intel64/gcc4.8`, `/usr/lib/x86_64-linux-gnu` |
| `CUDA_PATH` | `/usr/local/cuda-13.2` |
| `HOME` | `/home/jonaslacour` (from the service user) |

Must not be set: `LIBRARY_PATH` (drags the HPC SDK `libgomp` into links), `BATCHLAS_TEST_BACKEND` /
`BATCHLAS_TEST_FLOAT_TYPE` (a typo skips every test green), `OPENBLAS_CORETYPE` (set per test).

The workflow `env:` block of both GPU jobs is authoritative; `~/actions-runner/.env` repeats `PATH`
and `LD_LIBRARY_PATH` for pre-`env:` steps and is read at service start. Verify with `sycl-ls`: it
must list a `[cuda:gpu]` entry.

## The known-failures workflow

`main` is not green. The gate is not `ctest`'s exit code: `.github/ci/compare_failures.py` judges the
gtest XML and ctest JUnit against the ledger `tests/known-failures.txt` (expected failures, one per
line, each with a `#` reason; a debt ledger, not an allowlist).

```bash
mkdir -p build-dev-tests/gtest-xml
GTEST_OUTPUT=xml:build-dev-tests/gtest-xml/ \
    ctest --preset dev-tests --output-junit ctest.xml      # trailing slash: one XML per executable
python3 .github/ci/compare_failures.py --junit ctest.xml \
    --gtest-dir build-dev-tests/gtest-xml \
    --require consumer_package_tests --expected-tests-from build-dev-tests
```

| Flag | Meaning |
| --- | --- |
| `--known` | Ledger; default by tree: `known-failures.txt` (DPC++), `-acpp.txt` (`--sycl-impl ACPP`), `-acpp-metal.txt` (`--gpu-family METAL`), `-acpp-cuda.txt` (`--acpp-targets cuda:sm_XX`); `--build-dir <tree>` reads these from the cache |
| `--host-ledger` | Per-host supplement `tests/known-failures-host-<host>.txt`; DPC++ trees only |
| `--expected-tests-from <tree>` | JUnit count must equal `ctest -N` of the tree (catches a truncated run); add `--label-exclude slow` after `-LE slow` |
| `--min-tests N` | Fixed floor from a known-good full configure of `main`, never the tree under test. CI: 95 (`-LE slow`), 100 (full) |
| `--require <test>` | Anything but a real pass is an error; full runs only (`consumer_package_tests` is `slow`) |

Exit 1 on `NEW FAILURES` (failed, not listed) and on `NEWLY PASSING` (listed, passed: delete the line
in the same commit as the fix). Also reported: `SKIPPED` (no case executed), `LISTED BUT NOT IN THIS
REPORT` (did not run), `[COARSE: …]` (matched at ctest level only).

Entries are `<ctest-test>::<Suite>.<Case>`; pin typed tests with `;type_param=<spelling>` (the script
prints the observed spelling; a mismatched pin is an error). Typed-test indices shift with the
configuration (`SteqrTest/0` differs under `BATCHLAS_CPU_TARGET=none`), and `lanczos_tests` compiles to
zero tests without a GPU backend, so its entries land in `LISTED BUT NOT IN THIS REPORT`.

## Running all of it locally

```bash
.github/ci/run_local_checks.sh                 # exactly what the hosted static jobs run
.github/ci/run_local_checks.sh /tmp/inst       # ...plus the package installed with cmake --install
BATCHLAS_BUILD_DIR=build .github/ci/run_local_checks.sh   # picks the tree's ledger
```

It builds the docs when Doxygen 1.18+ is on `PATH` (`DOXYGEN=` to override; `BATCHLAS_DOCS_STRICT=1`
to match the `docs` job).

Full gate as the nightly runs it (the preset sets `execution.jobs=1`; do not add `-j`):

```bash
cmake --preset dev-tests -DBATCHLAS_ENABLE_CUDA=ON
cmake --build --preset dev-tests
ctest --preset dev-tests
```

`scripts/gate.sh` does this on persistent trees:
`build-gate` (full ctest, `compare_failures.py`; report `build-gate/gate/compare.txt`) and
`build-gate-vf` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`, `-LE slow`, no ledger), summary in
`build-gate/gate-summary.txt`. It sets `ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'` when unset and
`sycl-ls` lists `native_cpu`. Serial on the 4090 box (2026-10): `-LE 'slow|heavy'` 416 s, full 1287 s.

Both trees build first. With two GPUs visible, the two ctests then run at once, each serial: vf on
the first GPU, full on the second, each with `DPCPP_CPU_NUM_CUS` at cores/2 (the Intel OpenCL CPU
runtime's thread count; on threadripper02 the default 64-thread pool burned 96 s of sys time for
7 s of wall). The full tree used to wait for the vf tree's 3939 s. Each tree's ctest gets its own
`TMPDIR` (`<tree>/gate/tmp`), so the death-test scratch paths cannot collide. Without
`CUDA_VISIBLE_DEVICES`, GPUs above `BATCHLAS_GPU_BUSY_UTIL` % (30) or `BATCHLAS_GPU_BUSY_MEM_MB`
(4096) are skipped, as in `scripts/ctest_gpus.sh`; with one idle GPU left, both trees run on it,
one after the other. So does a busy box: a 1-minute load average above `BATCHLAS_GATE_MAX_LOAD`
(default cores/4) or another process above 200% CPU (the summary names it). `--serial`, a single
visible GPU (`CUDA_VISIBLE_DEVICES=0`), `--parallel` or
`--only` run one tree at a time. `--no-build` tests the trees as built (the summary header says
`no-build`), and `BATCHLAS_GATE_DIR` / `BATCHLAS_GATE_VF_DIR` / `BATCHLAS_BUILD_LOCK` point
the gate at other trees and another lock.

Each ctest writes `<tree>/gate/rss.tsv` (`scripts/gate_rss.py`; Linux): peak summed RSS per ctest
entry, death-test children included, the box's lowest `MemAvailable`, and a cross-check against
`/usr/bin/time`'s exact largest process. The summary quotes the top three entries. After each run,
the gate rewrites `Testing/Temporary/CTestCostData.txt` from the junit times. A serial ctest never
reads that file and writes time/runs, so the costs had sunk to half the real times, and any
ranking built on them was wrong.

Not a pre-push gate: `-LE slow` (no slow suites, no `consumer_package_tests`), `-LE 'slow|heavy'` (also
no saturating-batch race guards), `BATCHLAS_TEST_TARGET_SET=smoke`. Scoping table: `tests/README.md`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Runner offline | Unit not enabled after reboot; registration removed; failed self-update | `sudo ~/actions-runner/svc.sh status`, `journalctl -u 'actions.runner.*' -n 100`; restart or re-run `config.sh --replace` |
| Workflow stuck for hours | No online runner, or the `batchlas-gpu-runner` group is held | Check the Actions tab; times out at 6 h |
| ccache never warms | Bare `ccache` launcher hashes absolute paths | Use `<builddir>/batchlas-ccache` (`cmake/BatchLASCcache.cmake`). Links are never cached; the nightly vendor-free tree is cold by construction |
| Green, but packaging untested | `consumer_package_tests` skips with code 77 | `--require consumer_package_tests` (nightly), or run the full `ctest` |
| Green, zero assertions | `GTEST_SKIP()` in `BatchLASTest::SetUp` (bad `BATCHLAS_TEST_BACKEND`/`_FLOAT_TYPE`) | Read `SKIPPED` in the `compare_failures.py` report, not the exit code |

### OOM kill during ctest

`ctest` dies with no test failure or "Child aborted"; `dmesg -T | grep -i 'killed process'` names the
victim. The box has 125 GB and no swap, and both recorded kills had a build or a second `ctest`
running. Stop everything else and keep each `ctest` serial. The gate's two-tree phase is the
exception, and it is safe for these reasons: it starts only after both builds have finished, it
holds the build lock, so no build can start beside it, and it records per-entry peak RSS and the
box's lowest `MemAvailable` (`<tree>/gate/rss.tsv`). If that minimum nears zero, run
`scripts/gate.sh --serial`.
