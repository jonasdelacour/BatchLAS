# CI: what it proves, and how to stand the runner up

> **Status:** current · self-hosted runner: RTX 4090 (sm_89), CUDA 13.2, Ubuntu 22.04

Static checks run on GitHub-hosted runners. The GPU jobs run on one self-hosted workstation, and
they are the only jobs that configure, compile, install or execute anything. CI gates one
configuration (CUDA DPC++ on `sm_89`) and no other row of the README "Tested platforms" table.
`.github/workflows/ci.yml` is the authority on what runs.

## What CI covers and what it does not

| Covered | Where | Notes |
| --- | --- | --- |
| CMake lists parse; `CMakePresets.json` validates | hosted | `cmake --list-presets`; no configure |
| Exported package has no host paths (source mode) | hosted | Sees only literal paths in the list files |
| Public headers do not include `src/`; no unprefixed `<blas/…>`, `<util/…>`, `<internal/…>`; no type in the consumer's global namespace | hosted | Text scans; no build sees the last one |
| No C++ source over the 18% comment-density ceiling | hosted | `check_comment_density.py`. Doxygen doc comments in `include/batchlas/` are exempt; see [Writing API documentation](developer/documentation.md#writing-api-documentation) |
| Every `evidence:` pointer resolves; cited slugs are unique across `docs/` | hosted | `check_evidence_anchors.py` |
| No Markdown outside `docs/`; no raw content under an LFS path | hosted | `check_markdown_locations.py`, `check_lfs_pointers.py` |
| Docs site builds with zero Doxygen warnings; published on push to `main` | hosted | `docs` and `docs-deploy`; see [The docs jobs](#the-docs-jobs) |
| Configure with real CUDA DPC++ | GPU, per PR | `-DBATCHLAS_ENABLE_CUDA=ON`, then grep for `Using SYCL targets: … nvidia_gpu_sm_89` |
| Compile every component library and test binary | GPU, per PR | `cmake --build --preset dev-tests`: 95 library TUs, 61 test executables |
| Run the suite on a GPU | GPU, per PR | `ctest --preset dev-tests -LE slow`: 65 of 70 tests. The only place wrong numbers, launch failures and device aborts are caught |
| Package in package mode | GPU, per PR | `cmake --install` to a temp prefix, then `run_local_checks.sh <prefix>` |
| Four slow suites; packaging end to end | GPU, nightly | Full 70-test `ctest`, with `consumer_package_tests` under `--require` |
| Vendor-free build configures, compiles, links | GPU, nightly | Report-only (`continue-on-error`); not green, no ledger |

Not covered: AMD/ROCm; CPU-only and `icpx` builds (see
[Why there is no hosted compile job](#why-there-is-no-hosted-compile-job)); macOS and Windows;
Python bindings and benchmarks (both off in `dev-tests`, so a benchmark that stops compiling
passes); performance (nothing is timed); other NVIDIA architectures, CUDA versions and DPC++
builds; multi-GPU behaviour; flakiness (one run, one verdict); and anything a skipped test would
have covered ([A green run that tested nothing](#a-green-run-that-tested-nothing)).

The gate is a single self-hosted workstation. If it is off, wedged or busy with a human's build,
every pull request queues behind it.

## The jobs

| Job | Runs on | Trigger | Does |
| --- | --- | --- | --- |
| Static checks (`cmake-lint`, `exported-package`, `public-headers`, `comment-density`, `lfs-pointers`) | `ubuntu-latest` | every trigger | Scripts under `.github/ci/`; all run locally with `.github/ci/run_local_checks.sh` |
| `gpu-build-test` | `[self-hosted, linux, x64, cuda]` | push to `main`, non-fork PRs | Configure, assert the CUDA target, build, `ctest -LE slow`, `compare_failures.py`, install and check the export. Timeout 180 min |
| `gpu-nightly-full` | same | 03:17 UTC, `workflow_dispatch` with `run_nightly` | Full 70-test `ctest`, then a vendor-free configure, build and ctest that reports without gating. Timeout 360 min |
| `docs` | `ubuntu-latest` | every trigger | Builds the site (below) |
| `docs-deploy` | `ubuntu-latest` | push to `main`, after `docs` | Publishes the Pages artifact |

### The docs jobs

- **`docs`** runs `docs/tools/gen_db_pages.py --self-test`, downloads Doxygen 1.18.0 (sha256 pinned
  in `ci.yml`), installs Graphviz, then runs `BATCHLAS_DOCS_STRICT=1 sh scripts/build_docs.sh
  build/docs`. Strict mode fails on any Doxygen warning. `docs/tools/check_doc_anchors.py` fails on a
  cited anchor that is not a live section id. Uploads `docs-html` (14 days). The checkout fetches
  full history (`fetch-depth: 0`) because the Results database dates each file by its adding commit.
- **`docs-deploy`** uses `actions/deploy-pages`. `pages: write` and `id-token: write` go to this job
  alone, and it runs no repository code.

> **Note:** a repository admin must enable GitHub Pages once (Settings → Pages → Source: "GitHub
> Actions"). Until then `docs-deploy` fails and `docs` stays green. To bump Doxygen, see
> [Building the site](developer/documentation.md#building-the-site).

### Required properties of the GPU jobs

- **No fork pull requests.** The runner executes PR content as this user, with this `$HOME`, ccache
  and both GPUs. The `if:` on both jobs restricts them to branches in this repository.
  `pull_request_target` is not a fix: same execution, more secrets.
- **One harness on the box at a time.** Both jobs share the `batchlas-gpu-runner` group with
  `cancel-in-progress: false`, so a half-finished build is queued behind, not discarded. The group
  cannot arbitrate a human's own build or `ctest` on the same machine; see
  [OOM kill during ctest](#oom-kill-during-ctest).

## Standing up the self-hosted runner

> **Warning:** no runner has been installed and observed working end to end (`~/actions-runner`
> does not exist; `gh` is absent). The prerequisites are measured; the assembly is not. Expect to
> iterate once on the first run.

### Registration token

`config.sh` needs a registration token (not a personal access token; not stored on this machine).
Get it from **GitHub → the repository → Settings → Actions → Runners → New self-hosted runner**. It
is valid for one hour. The page prints the release version and download URL to use in the commands below.

### Install

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner

# Version and URL: copy from Settings -> Actions -> Runners -> New self-hosted runner.
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v<VERSION>/actions-runner-linux-x64-<VERSION>.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz

./config.sh \
  --url https://github.com/jonasdelacour/BatchLAS \
  --token <REGISTRATION_TOKEN_FROM_THE_SETTINGS_PAGE> \
  --name batchlas-4090 \
  --labels self-hosted,linux,x64,cuda \
  --work _work \
  --unattended --replace
```

`config.sh` adds `self-hosted`, `Linux` and `X64` itself; `cuda` is the distinguishing label. Label
matching in `runs-on:` is case-insensitive.

Install the service rather than using `./run.sh`. An interactive `run.sh` inherits the login-shell
environment that makes the project build, so it works while the service does not.

```bash
sudo ./svc.sh install jonaslacour     # run AS jonaslacour, not as root
sudo ./svc.sh start
sudo ./svc.sh status
```

`loginctl show-user jonaslacour` reports `Linger=no`, so a `systemd --user` unit would not survive
logout; the service must be a system service. A service running as `root`, or as a user whose
`HOME` is not `/home/jonaslacour`, loses all ccache sharing
([ccache is cold, or silently useless](#ccache-is-cold-or-silently-useless)).

## The environment trap

A systemd service inherits none of `~/.bashrc`. It gets `/etc/environment`: the stock `PATH`, no
`/opt/dpcpp-cuda`, no CUDA, no `LD_LIBRARY_PATH`. No `/etc/profile.d` script or `ld.so.conf.d` entry
supplies them, and the installed binaries carry neither `RPATH` nor `RUNPATH`.

- With `LD_LIBRARY_PATH` unset, `sycl-ls` exits 127 (`libsycl.so.9: cannot open shared object
  file`). The compiler driver survives the same environment.
- CMake calls `sycl-ls` three times. Every failure path in `cmake/BatchLASDetectSYCL.cmake` is a
  `message(STATUS)` or `message(WARNING)` followed by `return()`, never `FATAL_ERROR`.

Under the default `BATCHLAS_ENABLE_CUDA=AUTO`, a naively installed runner detects no GPU, prints
`No specific GPU architectures detected, using default JIT compilation`, and then configures,
compiles, installs and passes `ctest` on a CPU-only library that never emitted NVPTX. This is the
pitfall in the TL;DR of [the agent guide](developer/agent-guide.md).

### What must be set

| Variable | Value | Why |
| --- | --- | --- |
| `PATH` | prepend `/opt/dpcpp-cuda/bin`, `/usr/local/cuda-13.2/bin` | `cmake/BatchLASCompilerBootstrap.cmake` finds `sycl-ls` first, then the compiler beside it |
| `LD_LIBRARY_PATH` | `/opt/dpcpp-cuda/lib` (mandatory), `/usr/local/cuda-13.2/lib64`, `/opt/intel/oneapi-tbb-2022.3.0/lib/intel64/gcc4.8`, `/usr/lib/x86_64-linux-gnu` | Nothing SYCL runs without the first entry |
| `CUDA_PATH` | `/usr/local/cuda-13.2` | Fallback for `--cuda-path` when `find_package(CUDAToolkit)` misses |
| `HOME` | `/home/jonaslacour` | ccache is looked up in `$HOME/.local/bin`. Set by the service user, not the workflow |

Must not be set:

- **`LIBRARY_PATH`.** `~/.bashrc` exports `LIBRARY_PATH=$LD_LIBRARY_PATH` after adding the HPC SDK's
  `compilers/lib`, which drags that SDK's `libgomp` into every link. The workflow lists the four
  directories explicitly and leaves `LIBRARY_PATH` unset.
- **`BATCHLAS_TEST_BACKEND` / `BATCHLAS_TEST_FLOAT_TYPE`.** A typo produces a green, vacuous run;
  see [A green run that tested nothing](#a-green-run-that-tested-nothing).
- **`OPENBLAS_CORETYPE`.** `tests/CMakeLists.txt` sets it per test, and the configure-time dgemm
  health check runs with `--unset=OPENBLAS_CORETYPE`; a global value defeats that probe. Source
  `build/presets/dev-tests/batchlas-env.sh` only for a CI step that launches a binary outside `ctest`.

### Where to put it

- **The workflow `env:` block** (on both GPU jobs) is the version-controlled, authoritative copy.
- **`~/actions-runner/.env`** holds the same `PATH` and `LD_LIBRARY_PATH` for steps that run before a
  job's `env:` applies (`actions/checkout`, artifact upload). It is read at service start, so restart
  the service after editing.

### Verify

```bash
sycl-ls        # must list an entry beginning [cuda:gpu]
```

A failure here is silent, so `ci.yml` asserts it three ways (keep all three):

1. A "Record the toolchain" step runs `clang++ --version`, `cmake --version`, `sycl-ls` and
   `nvidia-smi -L` before building.
2. Configure uses `-DBATCHLAS_ENABLE_CUDA=ON`, not the preset's inherited `AUTO`. `ON` aborts with
   `FATAL_ERROR` naming the missing `[cuda:gpu]` entry.
3. The configure log is grepped for `Using SYCL targets: … nvidia_gpu_sm_89`. `spir64_x86_64` means a
   CPU-only build.

## The known-failures workflow

`main` is not green. Four gtest cases fail stably on a clean tree:

| ctest test | gtest case |
| --- | --- |
| `lanczos_tests` | `LanczosTestBase.LanczosTest` |
| `lanczos_tests` | `LanczosTestBase.ToeplitzEigenpairs` |
| `steqr_tests` | `SteqrTest/0.StressExtremeMagnitudesN32` (float, `Backend::NETLIB`) |
| `steqr_tests` | `SteqrTest/1.StressExtremeMagnitudesN32` (double, `Backend::NETLIB`) |

The gate is not `ctest`'s exit code. **`tests/known-failures.txt`** is the ledger: expected
failures, one per line, each with a comment saying why. It is a debt ledger, not an allowlist.
**`.github/ci/compare_failures.py`** judges the run's machine-readable reports against it.

### Running it by hand

The script needs both report layers: per-case gtest XML (the only view inside a binary) and ctest's
JUnit (the only place `consumer_package_tests`, a bash script with no gtest XML, and never-built
executables appear).

```bash
mkdir -p build/presets/dev-tests/gtest-xml
GTEST_OUTPUT=xml:build/presets/dev-tests/gtest-xml/ \
    ctest --preset dev-tests --output-junit ctest.xml
python3 .github/ci/compare_failures.py \
    --junit ctest.xml \
    --gtest-dir build/presets/dev-tests/gtest-xml \
    --require consumer_package_tests \
    --expected-tests 70
```

- The trailing slash on `GTEST_OUTPUT` is required: it selects directory mode, one XML per executable.
- `--known` defaults to `tests/known-failures.txt`.
- Use `--require` and `--expected-tests` on a full run only. After `ctest -LE slow`,
  `consumer_package_tests` does not run and the count is 65, so the per-PR job passes neither.

### Both directions fail the gate

- **Unexpected failure** (failed, not in the ledger): exit 1, under `NEW FAILURES`.
- **Newly passing listed case**: exit 1, under `NEWLY PASSING`. Fix it by deleting the ledger line in
  the same commit as the fix. A warning in a green run would be invisible, since the GPU job uploads
  its report only on failure.

Other report sections: `SKIPPED` (binaries in which no case executed, a vacuous run);
`LISTED BUT NOT IN THIS REPORT` (ledger entries the run never reached); `[COARSE: …]` (an entry
matched only at ctest-test level, so the specific case is unverified).

### Adding and removing entries

Fix the bug if you can. Otherwise, add a line keyed `<ctest-test>::<Suite>.<Case>`, for example
`lanczos_tests::LanczosTestBase.LanczosTest`, with the reason in a `#` comment. For a typed test, pin
the type: `steqr_tests::SteqrTest/0.StressExtremeMagnitudesN32;type_param=<spelling>`. Run the suite
once; the script prints the observed spelling. A typed entry without the pin warns, and a mismatched
pin is a hard error.

On `NEWLY PASSING`, confirm it is a real fix (`LISTED BUT NOT IN THIS REPORT` means the case did not
run), delete the line in the same commit, and if the case is gone (renamed, deleted, not
instantiated), say so in the commit message.

### Baseline caveats

- **Typed-test indices depend on the build configuration.** `SteqrTest/0` and `/1` are float and
  double `NETLIB` under `dev-tests`. Under `dev-gpu-tests` (`BATCHLAS_CPU_TARGET=none`) the host block
  vanishes and `SteqrTest/0` becomes a different test. The `;type_param=` pin guards against this.
- **`lanczos_tests` compiles to zero tests without a GPU backend.** Its file sits in one
  `#if BATCHLAS_HAS_GPU_BACKEND` with no `#else`. The target links and reports Passed with `tests="0"`,
  and both ledger entries land in `LISTED BUT NOT IN THIS REPORT`.

## Running all of it locally

Static checks take seconds and need no toolchain:

```bash
.github/ci/run_local_checks.sh                 # exactly what the hosted jobs run
.github/ci/run_local_checks.sh /tmp/inst       # ...plus the real installed package
```

The wrapper builds the docs into `build/docs` when Doxygen 1.18+ is on `PATH` (not strict; set
`BATCHLAS_DOCS_STRICT=1` to match the `docs` job). Run `cmake --install build/presets/dev-tests --prefix /tmp/inst`
first and pass the prefix to check the package before pushing anything that touches the install rules.

Full gate, as the nightly runs it (the preset sets `execution.jobs=1`; do not add `-j`):

```bash
cmake --preset dev-tests -DBATCHLAS_ENABLE_CUDA=ON
cmake --build --preset dev-tests            # tens of minutes
ctest --preset dev-tests
```

Full `ctest` is 70 tests. `ctest -LE slow` is 65 tests in about 95 s. Do not run the full suite on
every edit; `tests/README.md` has the scoping table. `scripts/ctest_gpus.sh` was validated only on
threadripper02; CI and local full gates stay serial.

Two scopes are wrong for a pre-push gate: `ctest -LE slow` excludes the four `slow` suites and
`consumer_package_tests`; `BATCHLAS_TEST_TARGET_SET=smoke` (the `fast-dev` preset) runs 9 binaries, with
none of the four baseline failures. It is an edit-loop tool only.

## Why there is no hosted compile job

`BATCHLAS_CPU_TARGET=none` does not disable the device pass; it only skips appending
`native_cpu` / `spir64_x86_64` to the SYCL target list. `-fsycl` is unconditional, so on a GPU-less
runner the driver falls back to its default `spir64` device pass (verified with `clang++ -fsycl -###`
and the `__CLANG_OFFLOAD_BUNDLE` sections in the object). There is no host-only mode.

A hosted job would pay several GB of oneAPI apt and a device pass over 95 TUs, to build a CPU-only
library that cannot catch a CUDA-backend compile error. It becomes worthwhile with a second
self-hosted machine or a prebuilt toolchain container with a warm ccache. Reasoning: the comment at
the foot of `ci.yml`.

## Troubleshooting

### The runner shows offline

```bash
sudo ~/actions-runner/svc.sh status
sudo journalctl -u 'actions.runner.*' -n 100 --no-pager
```

- **Symptom:** runner offline.
- **Cause, most common first:** a reboot with the unit not enabled; the registration removed from
  Settings → Actions → Runners; a self-update that could not reach GitHub.
- **Fix:** start the unit, or re-run `config.sh` with a fresh token and `--replace`.

- **Symptom:** a workflow stuck for hours. A job for `[self-hosted, linux, x64, cuda]` with no online
  runner waits, then times out after six hours. The same happens when another job holds the
  `batchlas-gpu-runner` group.
- **Fix:** check the Actions tab before assuming the service is dead.

### ccache is cold, or silently useless

- **Symptom:** the first build after a fresh clone or toolchain change takes tens of minutes (95
  library and 61 test TUs). The 180-minute timeout is expected; do not kill it. The cache never warms.
- **Cause:** without `CCACHE_BASEDIR` and `CCACHE_NOHASHDIR=1`, ccache hashes absolute paths, and each
  runner ref lives at a different path. A bare `CMAKE_CXX_COMPILER_LAUNCHER=ccache` got 3/196 hits, with
  no error. Settings in `~/.config/ccache` are not read by the job either.
- **Fix:** use the generated `<builddir>/batchlas-ccache` wrapper (`cmake/BatchLASCcache.cmake`). It
  exports `CCACHE_DEPEND=1`, `CCACHE_MAXSIZE=20G`, `CCACHE_SLOPPINESS`, `CCACHE_BASEDIR` and
  `CCACHE_NOHASHDIR=1`. Run ccache outside it and it sees the 5.0 G default and can evict. With the
  wrapper a second tree hit 187/196; the misses are TUs with an absolute path in a `-D`.
- Links are never cached. About 213 s of a 214 s all-hits rebuild is device linking, and `-j` does
  not shorten it. The nightly vendor-free build shares no entries (different vendor macros), so it is
  cold by construction.

### OOM kill during ctest

- **Symptom:** `ctest` dies mid-run with no test failure, or a test reports "Child aborted".
  `dmesg -T | grep -i 'killed process'` names the victim.
- **Cause:** concurrency. The machine has 125 GB and zero swap, so memory pressure is a kill, not a
  slowdown. Both recorded kills had a build or a second `ctest` running.
- **Fix:** stop everything else on the box (interactive builds, a second `ctest`, benchmarks). Keep
  `ctest` serial; do not add `-j`. In `GTEST_OUTPUT` directory mode the XML is named after the
  executable, so a route-pinned rerun and its unpinned twin collide. Alone, every suite fits
  (`steqr_tests`, the largest, takes 12 s).

### A green run that skipped consumer_package_tests

- **Symptom:** green run, but packaging was not tested.
- **Cause:** ctest exits 0 for a skipped test. `consumer_package_tests` has `SKIP_RETURN_CODE 77`, and
  `examples/consumer_test.sh` has eight exit-77 paths. The dangerous one is "no usable device", reached
  after a successful install, configure, build and launch.
- **Fix:** the nightly passes `--require consumer_package_tests`, so anything but a real pass is an
  error. Do not add `--require` to the per-PR job; it never reaches the test. To check packaging on a
  branch, run the nightly by hand (`workflow_dispatch` with `run_nightly`) or the full `ctest` locally.

In ctest JUnit, only `status="run"` with no child is executed-and-passed. Skips are `status="notrun"`
with a `<skipped>` child, which includes never-built binaries. The root summary attributes are unreliable.

### A green run that tested nothing

- **Symptom:** all tests green, zero assertions executed.
- **Cause:** `BatchLASTest::SetUp` in `test_utils.hh` calls `GTEST_SKIP()` when `should_run_backend` or
  `should_run_float_type` rejects an instantiation. A skip does not fail, so a binary with every case
  skipped exits 0. An unrecognised `BATCHLAS_TEST_BACKEND` skips everything, so one typo in a CI
  variable gives 70/70 green.
- **Defences:** CI sets neither `BATCHLAS_TEST_BACKEND` nor `BATCHLAS_TEST_FLOAT_TYPE`, and their
  absence from `env:` is a gate. `compare_failures.py` names such binaries under `SKIPPED`, read that
  section rather than the exit code. `--expected-tests 70` catches missing executables. The
  assertions in [The environment trap](#the-environment-trap) catch a CPU-only build.
