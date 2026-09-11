# CI: what it proves, and how to stand the runner up

This project has two kinds of CI job, with very different reach. The **static
checks** run on GitHub-hosted runners, need no toolchain, and finish in seconds;
they catch a narrow class of packaging and header defects. The **GPU jobs** run
on a self-hosted runner — one workstation — and are the only thing that
configures, compiles, installs or executes anything.

Read the coverage table before you trust a green tick. The honest summary is
that CI is now a real gate for *one* configuration (CUDA DPC++ on `sm_89`,
Ubuntu 22.04) and no gate at all for every other row of README's "Tested
platforms" table.

`.github/workflows/ci.yml` is the authority on what runs; its header comments
carry the reasoning per step. This document is the operator's half: how to make
the runner exist, and what to do when it misbehaves.

## What CI covers and what it does not

| Covered | Where | Notes |
| --- | --- | --- |
| CMake list files parse; `CMakePresets.json` validates | hosted | Static parse plus `cmake --list-presets`. No configure. |
| Exported package carries no host paths — *source* mode | hosted | Only sees paths written *literally* in the list files. |
| Public headers do not reach into `src/` | hosted | |
| No unprefixed `<blas/…>`, `<util/…>` or `<internal/…>` include | hosted | |
| No type declared in the consumer's global namespace | hosted | No build can see this defect; a text scan is the only cheap oracle. |
| Configure with a real CUDA DPC++ | GPU, per PR | `-DBATCHLAS_ENABLE_CUDA=ON`, plus an explicit grep for `Using SYCL targets: … nvidia_gpu_sm_89`. |
| Compile every component library and every test binary | GPU, per PR | `cmake --build --preset dev-tests` builds the default `all` target: the component shared objects from 95 source TUs, plus 61 test executables. |
| Run the test suite on a GPU | GPU, per PR | `ctest --preset dev-tests -LE slow` — 65 of 70 tests. Wrong numbers, kernel launch failures and device-side aborts are caught here and nowhere else. |
| Exported package carries no host paths — *package* mode | GPU, per PR | `cmake --install` to a temp prefix, then `run_local_checks.sh <prefix>`. This is the half a hosted runner cannot reach: generating the export needs a build. |
| The four slow suites, and packaging end to end | GPU, **nightly only** | The full 70-test `ctest`, including `consumer_package_tests` under `--require`. |
| Vendor-free build still configures, compiles and links | GPU, **nightly only** | Report-only (`continue-on-error`). It is not green and is not expected to be; it has no ledger of its own. |

| **Not covered** | Why |
| --- | --- |
| AMD / ROCm | No AMD hardware and no ROCm toolchain on the runner. Those code paths are compiled by nobody. |
| oneMKL backend, Intel GPU | Same: not built, not run. |
| CPU-only / `icpx` builds | The runner has a CUDA DPC++ only. A break that shows up solely without a GPU backend passes CI. There is deliberately **no hosted compile job**; see [Why there is no hosted compile job](#why-there-is-no-hosted-compile-job). |
| macOS, Windows | No attempt made. |
| Python bindings | `dev-tests` sets `BATCHLAS_BUILD_PYTHON=OFF`. `python/` is neither built nor tested by CI. |
| Benchmarks | Benchmark targets are `EXCLUDE_FROM_ALL`, so the default build never compiles them. A benchmark that no longer compiles passes CI. |
| Performance regressions | Nothing is timed. A change that is correct and 10x slower is green. |
| Other NVIDIA architectures, other CUDA versions, other DPC++ builds | Only `sm_89` / CUDA 13.2 / this `/opt/dpcpp-cuda` is built and run. |
| Multi-GPU behaviour | The box has two 4090s; nothing in the suite exercises more than one deliberately. |
| Flakiness | One run in, one verdict out. Nothing is repeated. |
| Anything a skipped test would have covered | See [A green run that tested nothing](#a-green-run-that-tested-nothing) — the failure mode to worry about most. |

**One machine.** The primary gate is a single self-hosted workstation. If it is
off, wedged, or busy with the human's own build, every pull request queues
behind it. There is no second runner and no cloud fallback.

## The jobs

**Static checks** (`ubuntu-latest`): `cmake-lint`, `exported-package`,
`public-headers`. Their reach is unchanged from before the GPU job existed.
Every one is a script under `.github/ci/`, so all of it runs locally with no
toolchain:

```bash
.github/ci/run_local_checks.sh
```

**`gpu-build-test`** (`runs-on: [self-hosted, linux, x64, cuda]`), on every push
to `main` and every non-fork pull request. Configure → assert the CUDA target →
build → `ctest -LE slow` → `compare_failures.py` → install and check the
generated export. Timeout 180 minutes, which is deliberately loose: a cold
ccache is tens of minutes on its own.

**`gpu-nightly-full`**, on the 03:17 UTC schedule and on `workflow_dispatch`
with `run_nightly`. The full 70-test `ctest` (the four `slow` suites plus
`consumer_package_tests`), then a vendor-free configure/build/ctest that reports
rather than gates. Timeout 360 minutes.

Two properties of the GPU jobs are not optional, and both are already in the
workflow:

* **No fork pull requests.** A self-hosted runner on a workstation executes
  whatever the pull request contains, as this user, with this user's `$HOME`,
  this user's ccache and both GPUs. The `if:` condition on both jobs restricts
  them to branches in this repository. `pull_request_target` is not the fix —
  it is the same execution with more secrets in scope.
* **One harness on the box at a time.** Both jobs share the
  `batchlas-gpu-runner` concurrency group, with `cancel-in-progress: false` so a
  half-finished build is queued behind rather than thrown away. What that
  **cannot** arbitrate is the human's own interactive build or `ctest` on the
  same workstation; see [OOM kill during ctest](#oom-kill-during-ctest).

## Standing up the self-hosted runner

Everything below is assembled from prerequisites measured on this machine.
**No runner has actually been installed and observed working end to end** —
`~/actions-runner` does not exist and the `gh` CLI is not present — so treat
this as "verified prerequisites, unverified assembly", and expect to iterate
once on the first run.

### The one step nobody but the repository owner can do

`config.sh` needs a **registration token**. It is not a personal access token,
and it is not in any file on this machine. Get it from

> **GitHub → the repository → Settings → Actions → Runners → New self-hosted
> runner**

which prints a token valid for one hour, together with the exact release version
and download URL for the current runner package. Use the version that page
prints rather than the placeholder below; the runner self-updates afterwards,
but the initial download must match what GitHub is serving.

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

`config.sh` assigns `self-hosted`, `Linux` and `X64` on its own; passing them
again is harmless, and `cuda` is the label that actually distinguishes this
runner. Label matching in `runs-on:` is case-insensitive, so the workflow's
lowercase `linux, x64` matches.

**Do not stop at `./run.sh`.** An interactive `run.sh` started from a login
shell inherits the environment that makes this project build — which is exactly
why it will appear to work while the service does not. Install the service:

```bash
sudo ./svc.sh install jonaslacour     # run AS jonaslacour, not as root -- see below
sudo ./svc.sh start
sudo ./svc.sh status
```

The user argument matters twice over. `loginctl show-user jonaslacour` reports
`Linger=no`, so a `systemd --user` unit would not survive logout — it has to be
a system service. And a system service running as `root`, or as any user whose
`HOME` is not `/home/jonaslacour`, silently loses all ccache sharing (see
[ccache is cold, or silently useless](#ccache-is-cold-or-silently-useless)).

## The environment trap

**Read this before the first run.** It is the single most likely way to get a
green CI that tested nothing.

A systemd service inherits **none** of `~/.bashrc`. It gets `/etc/environment`,
which on this machine is the stock `PATH="/usr/local/sbin:…:/snap/bin"` — no
`/opt/dpcpp-cuda`, no CUDA, and no `LD_LIBRARY_PATH` at all. There is no
`/etc/profile.d` script and no `ld.so.conf.d` entry that would supply them:
`/opt/dpcpp-cuda/lib` is on the loader path *only* because `~/.bashrc` puts it
there, and the installed binaries carry neither `RPATH` nor `RUNPATH`.

### What that produces

Measured on this box, not inferred:

* With `LD_LIBRARY_PATH` unset, **`sycl-ls` itself exits 127** —
  `libsycl.so.9: cannot open shared object file`.
* The compiler *driver* survives that same bare environment and compiles fine.
  Only `sycl-ls` dies.
* CMake calls `sycl-ls` three times (compiler bootstrap, GPU-architecture
  detection, CPU-target detection). Every failure path in
  `cmake/BatchLASDetectSYCL.cmake` is a `message(STATUS)` or `message(WARNING)`
  followed by `return()` — never a `FATAL_ERROR`.

So under the default `BATCHLAS_ENABLE_CUDA=AUTO`, a naively installed runner
detects no GPU, prints `No specific GPU architectures detected, using default
JIT compilation`, and then **configures, compiles, links, installs and passes
`ctest` on a CPU-only library that never emitted a line of NVPTX**. This is the
pitfall AGENTS.md names in its TL;DR as the single most common way to get a
working build that quietly does the wrong thing, and a runner service is the
most likely place in this project to hit it.

### What must be set

| Variable | Value | Why |
| --- | --- | --- |
| `PATH` | prepend `/opt/dpcpp-cuda/bin` and `/usr/local/cuda-13.2/bin` | `cmake/BatchLASCompilerBootstrap.cmake` finds the compiler by first finding `sycl-ls` and then looking beside it. The compiler being installed is not enough — `sycl-ls` must resolve. |
| `LD_LIBRARY_PATH` | `/opt/dpcpp-cuda/lib` (**mandatory**), `/usr/local/cuda-13.2/lib64`, `/opt/intel/oneapi-tbb-2022.3.0/lib/intel64/gcc4.8`, `/usr/lib/x86_64-linux-gnu` | Nothing SYCL runs without the first entry. |
| `CUDA_PATH` | `/usr/local/cuda-13.2` | `BatchLASDetectSYCL.cmake` reads it as the fallback for `--cuda-path` when `find_package(CUDAToolkit)` misses. |
| `HOME` | `/home/jonaslacour` | ccache's `BATCHLAS_CCACHE_BASEDIR` defaults to it; a different `HOME` costs the whole cache with no error. Set by the service user, not by the workflow. |

And two things that must **not** be set:

* **`LIBRARY_PATH`.** Do not mirror `~/.bashrc` here. That file does
  `export LIBRARY_PATH=$LD_LIBRARY_PATH` after having already put the HPC SDK's
  `compilers/lib` on it, which drags that SDK's `libgomp` into every link on
  this box. The workflow lists the four directories above explicitly and leaves
  `LIBRARY_PATH` unset on purpose.
* **`BATCHLAS_TEST_BACKEND` / `BATCHLAS_TEST_FLOAT_TYPE`.** See
  [A green run that tested nothing](#a-green-run-that-tested-nothing). A typo in
  either one produces a fully green, fully vacuous run. Do not add them to
  "focus" a CI run.

`OPENBLAS_CORETYPE` deliberately does **not** appear in that table.
`tests/CMakeLists.txt` sets it as a per-test `ENVIRONMENT` property on all three
registration paths, and the configure-time dgemm health check runs its own probe
with `--unset=OPENBLAS_CORETYPE`, so that a stray inherited value cannot mask a
broken default. Setting it globally on the runner would defeat that probe. It
resolves to `SKYLAKEX` on this machine and is written to
`build/presets/dev-tests/batchlas-env.sh`; **source that file only if you add a
CI step that launches a test or benchmark binary directly, outside `ctest`.**

### Where to put it

Two places, and use both:

* **The workflow's `env:` block.** Already present on both GPU jobs in
  `ci.yml`, with the reasoning inline. This is the version-controlled,
  reviewable copy and the one to treat as authoritative.
* **The runner's `~/actions-runner/.env` file.** Put the same `PATH` and
  `LD_LIBRARY_PATH` there. The runner reads it when the service starts, so it
  covers what the workflow does not — `actions/checkout`, artifact upload, and
  anything a future job runs before its own `env:` applies. Restart the service
  after editing it; it is read at start, not per job.

### Verify it — in the workflow, not just here

```bash
sycl-ls        # must list an entry beginning [cuda:gpu]
```

Its failure mode is silence, which is why the assertion belongs in the job. All
three of these are already in `ci.yml`, and none should be removed:

1. A "Record the toolchain" step runs `clang++ --version`, `cmake --version`,
   `sycl-ls` and `nvidia-smi -L` before anything is built, so a failure report
   carries the environment and a broken `LD_LIBRARY_PATH` surfaces as an error
   rather than as a silent CPU build.
2. Configure with `-DBATCHLAS_ENABLE_CUDA=ON` rather than the preset's inherited
   `AUTO`. `ON` means "require it": the configure aborts with a `FATAL_ERROR`
   naming the missing `[cuda:gpu]` entry instead of degrading quietly.
3. Grep the configure log for `Using SYCL targets: … nvidia_gpu_sm_89`.
   AGENTS.md names that exact line as the tell; `Using SYCL targets:
   spir64_x86_64` means you are about to test a CPU-only build, whatever GPUs
   are in the box.

## The known-failures workflow

`main` is **not green**. Four gtest cases fail on a clean tree, stably, in two
binaries:

| ctest test | gtest case |
| --- | --- |
| `lanczos_tests` | `LanczosTestBase.LanczosTest` |
| `lanczos_tests` | `LanczosTestBase.ToeplitzEigenpairs` |
| `steqr_tests` | `SteqrTest/0.StressExtremeMagnitudesN32` (float, `Backend::NETLIB`) |
| `steqr_tests` | `SteqrTest/1.StressExtremeMagnitudesN32` (double, `Backend::NETLIB`) |

A job that failed on any non-zero `ctest` exit could therefore never go green,
and a job that never goes green is switched off within a week. So the gate is
not `ctest`'s exit code:

* **`tests/known-failures.txt`** is the ledger: the cases expected to fail, one
  per line, each with a comment saying why. It is a **debt ledger, not an
  allowlist** — every line is a bug someone still owes.
* **`.github/ci/compare_failures.py`** reads the run's machine-readable reports
  and judges them against that ledger.

### Running it by hand

The script needs both report layers, because neither alone is sufficient — the
per-case gtest XML is the only thing that can see inside a binary, and ctest's
JUnit is the only place `consumer_package_tests` (a bash script that writes no
gtest XML) and a never-built executable appear at all:

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

The **trailing slash** on `GTEST_OUTPUT` is required — it selects directory
mode, in which every gtest binary drops its own XML named after the executable.
`--known` defaults to `tests/known-failures.txt`. Use `--require` and
`--expected-tests` only on a full run: after `ctest -LE slow`,
`consumer_package_tests` legitimately did not run and the count is 65, so the
per-PR job passes neither flag.

### Both directions, and why the second one matters more

* **An unexpected failure** — a case that failed and is not in the ledger —
  **fails the gate** (exit 1), under `NEW FAILURES`.
* **A newly-passing listed case** — one that is in the ledger but passed — is
  reported under `NEWLY PASSING` and **also fails the gate** (exit 1), so that
  deleting the line is unavoidable rather than optional. A warning inside a
  green tick would be invisible: the GPU job uploads its report only on
  failure, so on a green run the warning would exist nowhere but in scrollback.
  Fix is one line — delete the entry. This direction is not decoration: the
  previous recorded baseline included
  `SytrdBlockedLatrdGridCudaTest.GridMatchesLegacyTridiagonal`, which now
  passes. A one-directional ledger would have hidden that fix indefinitely and
  left the entry sitting there ready to mask the regression that re-breaks it.

Three other sections in the report are worth reading rather than skimming:
`SKIPPED` names any binary in which *no* case executed (the shape of a vacuous
run), `LISTED BUT NOT IN THIS REPORT` names ledger entries the run never
reached, and a ledger entry matched only at ctest-test level is tagged
`[COARSE: …]` to say the specific case is unverified.

### Adding an entry

Don't, if you can fix it instead. If you must:

* The key is `<ctest-test>::<Suite>.<Case>`, for example
  `lanczos_tests::LanczosTestBase.LanczosTest`. `#` starts a comment.
* For a **typed** test, pin the type as well:
  `steqr_tests::SteqrTest/0.StressExtremeMagnitudesN32;type_param=<spelling>`.
  Do not invent the spelling — run the suite once, and the script prints the
  observed one for you to paste in. It warns on any typed entry that lacks the
  pin, and it treats a *mismatched* pin as a hard error rather than honouring
  the index.
* Write the comment. "Why is this here" is the whole value of the file; an entry
  with no reason is indistinguishable, a year later, from a mistake.

### Removing an entry — the part that actually matters

When `compare_failures.py` prints `NEWLY PASSING`:

1. Confirm it is a real fix, not an accident of configuration. A case reported
   as `LISTED BUT NOT IN THIS REPORT` has **not** passed — it did not run.
2. Delete the line from `tests/known-failures.txt` **in the same commit as the
   fix**, so the ledger shrinks with the debt rather than lagging it.
3. If the case is gone rather than fixed (renamed, deleted, no longer
   instantiated), say so in the commit message. A vanished test is not a passing
   test.

### Two caveats on the baseline itself

* **The typed-test index is build-configuration dependent.** `SteqrTest/0` and
  `/1` are float and double `NETLIB` **under the `dev-tests` preset**, because
  that configuration includes the host-backend block of the type list. Under
  `dev-gpu-tests` (`BATCHLAS_CPU_TARGET=none`) that block vanishes and
  `SteqrTest/0` silently becomes a *different test*. This is what the
  `;type_param=` pin defends against, and it is why the GPU job is pinned to the
  `dev-tests` preset.
* **`lanczos_tests` compiles to zero tests without a GPU backend.** The whole
  file sits inside one `#if BATCHLAS_HAS_GPU_BACKEND` with no `#else`; the
  target still links (gtest supplies `main`) and reports Passed with
  `tests="0"`. So "lanczos_tests passed" on a CPU-only build is vacuous, both of
  its ledger entries land in `LISTED BUT NOT IN THIS REPORT`, and that is
  another reason the environment trap above is dangerous rather than merely
  annoying.

## Running all of it locally

**Static checks** — no toolchain, seconds:

```bash
.github/ci/run_local_checks.sh                 # exactly what the hosted jobs run
.github/ci/run_local_checks.sh /tmp/inst       # ...plus the real installed package
```

The optional argument is an install prefix (`cmake --install
build/presets/dev-tests --prefix /tmp/inst`). It enables the `--package` half of
the export check, which reads the `BatchLASConfig` / `BatchLASTargets` cmake
actually generated rather than the literals in the list files. Run it before
pushing anything that touches the install rules. The GPU job runs this same
command, deliberately: CI running the command a human runs is what keeps that
command from rotting.

**The full gate**, as the nightly job runs it:

```bash
cmake --preset dev-tests -DBATCHLAS_ENABLE_CUDA=ON
cmake --build --preset dev-tests            # tens of minutes; see below
ctest --preset dev-tests
```

The test preset already sets `execution.jobs=1`, so `ctest --preset dev-tests`
is serial by construction and needs no `-j`.

**Scoping a local run.** Do not run the full suite on every edit —
`tests/README.md` has the scoping table (`ctest -R '^name$'` for one binary,
`ctest -L <component>` for a subsystem, `ctest -LE slow` for a broad-but-quick
pass). Current measured counts on this tree: full `ctest` is **70 tests**;
`ctest -LE slow` is **65 tests in about 95 s**. (`tests/README.md`'s table
quotes an older 38-of-45; the counts here are the measured current ones.)

Two scoping choices are wrong for a pre-push gate, however convenient:

* **`ctest -LE slow` excludes `consumer_package_tests`**, which is
  `slow`-labelled, along with `stedc_tests`, `steqr_tests`, `sytrd_sb2st_tests`
  and `gesvd_tests`. This is exactly why the nightly job exists.
* **`BATCHLAS_TEST_TARGET_SET=smoke`** (the `fast-dev` preset) replaces the test
  list with 9 binaries, registers no `consumer_package_tests`, registers none of
  the route-pinned reruns, and contains none of the four baseline failures. It
  is an edit-loop tool, not a gate.

**The OOM caveat: run the GPU suite with nothing else on the box.** This machine
has 125 GB and **zero swap**, which is why memory pressure here is a kill rather
than a slowdown. A plain full `ctest` was killed by the OOM killer twice — both
times with a build or a second `ctest` running concurrently. Run alone, every
suite fits comfortably; `steqr_tests`, the largest, finishes in 12 s. The
constraint is *concurrency*, not any one test.

## Why there is no hosted compile job

It was specified, investigated and rejected on measurement; the reasoning is
written out at the foot of `ci.yml`, and the short version is that its premise
is false. `BATCHLAS_CPU_TARGET=none` does not disable the device pass — it only
declines to append `native_cpu` / `spir64_x86_64` to the SYCL target list.
`-fsycl` is unconditional, so on a GPU-less runner the list ends up empty and
the driver falls back to its default `spir64` device pass (verified with
`clang++ -fsycl -###`, and by the `__CLANG_OFFLOAD_BUNDLE` sections in the
resulting object). There is no host-only mode in this build system.

So a hosted job would pay several GB of oneAPI apt, a real device pass over 95
TUs, and ~15 single-threaded per-`.so` device links, on the runner with the
fewest cores — to produce the CPU-only build AGENTS.md warns against, which can
never catch a CUDA-backend compile error. What would change the answer: a second
self-hosted machine, or a prebuilt container image with the toolchain and a warm
ccache baked in.

## Troubleshooting

### The runner shows offline

Check the service, not the workflow:

```bash
sudo ~/actions-runner/svc.sh status
sudo journalctl -u 'actions.runner.*' -n 100 --no-pager
```

Common causes, in the order they actually occur: the machine rebooted and the
unit is not enabled; the registration was removed from the Settings → Actions →
Runners page (re-run `config.sh` with a fresh token and `--replace`); the runner
tried to self-update and could not reach GitHub.

A job queued against `runs-on: [self-hosted, linux, x64, cuda]` with no matching
online runner does not fail — it *waits*, and then times out after six hours. A
workflow that appears stuck for hours is this, not a hung test. The same shape
appears when the runner is online but another job holds the
`batchlas-gpu-runner` concurrency group; check the Actions tab for a run in
progress before assuming the service is dead.

### ccache is cold, or silently useless

The first build on a fresh clone, or after a toolchain change, compiles all 95
library TUs and all 61 test TUs from scratch. That is tens of minutes, and it is
why the job's timeout is 180 minutes. Do not kill it.

What is *not* expected is a cache that never warms up:

* **`HOME` must be `/home/jonaslacour`.** `BATCHLAS_CCACHE_SHARE_ACROSS_TREES`
  is ON and sets `CCACHE_BASEDIR=$HOME` plus `CCACHE_NOHASHDIR=1`, which is
  precisely what lets a build in one checkout hit a cache another populated —
  and a GitHub Actions runner checks every ref out into a *different* absolute
  path. With the wrong `HOME`, ccache hashes absolute paths, the hit rate goes
  to roughly zero, and there is no error and no log line. The cache just looks
  useless.
* **The settings live in a generated wrapper, not in `~/.config/ccache`.**
  `cmake/BatchLASCcache.cmake` generates `<builddir>/batchlas-ccache`, a shell
  wrapper that exports `CCACHE_DEPEND=1`, `CCACHE_MAXSIZE=20G`,
  `CCACHE_BASEDIR` and `CCACHE_NOHASHDIR=1` before exec'ing ccache. Any ccache
  invocation *not* through that wrapper sees the on-disk 5.0 G default and can
  evict. `ccache -s` currently reports 5.71 GB against that 20 G ceiling.
* The nightly's vendor-free build shares no entries with the default one — the
  vendor macros differ — so it is a second cold compile every night by
  construction, not a cache fault.

And a floor no cache can lower: **links are not cached.**
`CMAKE_CXX_COMPILER_LAUNCHER` does not apply to link steps, and the SYCL device
link is single-threaded and runs once per shared object — around 213 s of a
214 s all-cache-hits rebuild. `-j` does not shorten it. Expect minutes even on a
perfect cache hit, and do not read that as a broken cache.

### OOM kill during ctest

Symptom: `ctest` dies mid-run with no test failure, or a test is reported "Child
aborted"; `dmesg -T | grep -i 'killed process'` names the victim.

The cause is essentially always concurrency, because swap is zero. Fixes, in
order:

1. Make sure nothing else is running on the box — no interactive build, no
   second `ctest`, no benchmark campaign. This is the whole cause of both
   recorded kills.
2. Keep `ctest` serial. The `dev-tests` test preset already sets
   `execution.jobs=1`; do not add `-j` to the CI invocation. It also keeps the
   route-pinned reruns' gtest XML filenames deterministic, which matters
   separately — in `GTEST_OUTPUT` directory mode the file is named after the
   *executable*, so a route-pinned rerun and its unpinned twin collide and are
   told apart only by execution order.
3. The `batchlas-gpu-runner` concurrency group already stops two CI runs
   overlapping. Note what it does *not* buy you: it cannot stop a human working
   on the same machine. The runner and the workstation are the same 20-core,
   125 GB box, and no workflow file can arbitrate that.

### A green run that skipped consumer_package_tests

**ctest exits 0 for a skipped test.** `consumer_package_tests` carries
`SKIP_RETURN_CODE 77`, and `examples/consumer_test.sh` has eight distinct
exit-77 paths: cmake missing, no `--build-dir`, no configured build tree, no
compiler given, the compiler not executable, BatchLAS not built — and, the
dangerous one, **the built example reporting no usable device**. That last path
is reached *after* a successful install, configure, build and launch. On a
runner whose GPU is busy or whose environment is wrong, it converts a real
device failure into a green tick and a false "packaging tested" claim.

That is correct behaviour on a laptop and wrong in CI, which is why the nightly
job passes `--require consumer_package_tests`: anything other than a real pass
is then an error, and the report names which prerequisite was missing (the
script prints `[consumer] SKIP: <reason>`, which reaches the JUnit
`<system-out>`).

Note the asymmetry, and do not "fix" it: the **per-PR job does not pass
`--require`**, because it runs `-LE slow` and so never reaches that test.
Requiring it there would fail every pull request. If you want packaging checked
against a branch, run the nightly by hand — `workflow_dispatch` with
`run_nightly` — or run the full `ctest` locally.

The underlying discriminator, if you are reading a report by hand: in ctest's
JUnit, `status="run"` with no child element is the *only* thing that means the
test executed and passed. A skip is `status="notrun"` with a
`<skipped message="SKIP_RETURN_CODE=77"/>` child. And a test binary that was
**never built** is also `status="notrun"` with a `<skipped>` child, differing
only in the message text (`Unable to find executable`) — while ctest folds it
into the file's `skipped=` count and its own exit code calls it a failure. The
summary attributes on the root element are not trustworthy;
`compare_failures.py` classifies every case individually and treats an
unrecognised skip message as a hard failure for exactly this reason.

### A green run that tested nothing

The worst outcome, because it looks like the best one.

`test_utils.hh`'s `BatchLASTest::SetUp` calls `GTEST_SKIP()` when
`should_run_backend` or `should_run_float_type` rejects an instantiation.
`GTEST_SKIP` does not fail, so a binary in which *every* case skipped exits 0
and ctest records it as passed — indistinguishable, at the ctest level, from
full coverage. And `should_run_backend` treats an **unrecognised**
`BATCHLAS_TEST_BACKEND` value as "skip everything": it warns to stderr and
returns false for every backend.

So a single typo in a CI environment variable yields 70/70 green with zero
assertions executed. This is the same mechanism that once made a suite look
flaky when it was in fact being skipped entirely.

What defends against it:

* **CI sets neither `BATCHLAS_TEST_BACKEND` nor `BATCHLAS_TEST_FLOAT_TYPE`.**
  Their absence from the workflow's `env:` block is a gate, not an oversight.
  They are edit-loop levers.
* Only the gtest-level XML can see this; ctest cannot. `compare_failures.py`'s
  `SKIPPED` section names every binary in which no case executed — but
  **reporting is all it does**, so read that section rather than trusting the
  exit code.
* `--expected-tests 70` on a full run catches the adjacent failure: thirty
  missing executables are thirty extra "skips" as far as the JUnit is concerned.
* The `sycl-ls` and `Using SYCL targets:` assertions described in
  [The environment trap](#the-environment-trap). A CPU-only build is the most
  likely way to arrive here.
