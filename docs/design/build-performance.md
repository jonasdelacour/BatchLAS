# Build performance {#design_build_performance}

> **Status:** current · device-link sections measured 2026-10-24 on the 2x RTX 4090 box · the
> build rules that follow from them (and the measured dead ends) are in section 7 of
> `docs/developer/agent-guide.md`.

Two kinds of cost. The public headers keep consumer and test translation units cheap to compile,
but cannot shorten the device link; the library split and link options below are what shorten it.

## build performance: the umbrella header excludes device code

`<batchlas/blas/linalg.hh>`, which `<batchlas.hh>` includes, does not include:

- `<batchlas/blas/device.hh>`, the device-side group BLAS templates (@ref design_device_group_blas), or
- `<sycl/sycl.hpp>`, which `<batchlas/blas/functions.hh>` would otherwise bring in.

Together they cost about 4.1 s per consumer TU. 71 of the 114 test and benchmark TUs use neither.
These figures predate 2026-09-30 and have not been re-measured.

- A consumer that needs `batchlas::device::*` includes `<batchlas/blas/device.hh>` and builds with `-fsycl`.
- A consumer that passes a `sycl::queue` or `sycl::event` across the boundary includes
  `<batchlas/sycl_interop.hh>`, the one public header that includes `<sycl/sycl.hpp>` directly.

**Both edges must stay cut.** The headers form a cycle. Restoring either include brings back the whole
umbrella, so cutting one edge gains nothing.

Code sites: `include/batchlas/blas/linalg.hh` (above `functions.hh`) and `include/batchlas/sycl_interop.hh`
(above its `<sycl/sycl.hpp>` include). Consumers must configure with the same SYCL compiler; see
`docs/cpp-api.md`, "Building against BatchLAS".

## build performance: where device-link time went in 2026-10

Measured on the 2x RTX 4090 box (20 cores, 123 GB, zero swap, DPC++ at `/opt/dpcpp-cuda`, full
targets `spir64_x86_64,nvidia_gpu_sm_89`, -j8) against `c960d7b1`. Per-module times come from
replaying a link's driver `-###` commands one stage at a time and running each llvm-foreach item
alone. The build-time rules that follow from this are in section 7 of @ref dev_agent_guide.

### build performance: global-offset kernel twins

DPC++'s GlobalOffset pass gives every kernel a `<name>_with_offset` twin that nothing launches:
2,619 of the 5,305 kernel entry points across the 13 libraries of the time. Passing
`-mllvm -enable-global-offset=false` to the device link (`BATCHLAS_SYCL_GLOBAL_OFFSET=OFF`, the
default; probed at configure) removes them:

| Module or library | With twins | Without |
| --- | --- | --- |
| posv_tiny module | 319 s | 164 s |
| gesv_tiny module | 131 s | 70 s |
| syev_cta_fused module | 20 s | 12 s |
| a one-object .so | 20.8 s | 11.1 s |
| cta cubin entries | 1,262 (631 twins) | 631 (cta 517 + posv 57 + gesv 57) |

The cost: a kernel launched with a SYCL 1.2.1 global offset (`parallel_for(range, id, k)`,
`get_offset()`) would no longer see it. Nothing in the tree does. `scripts/register_probe.sh`
reads either spelling.

### build performance: one device module is the link floor

`sycl-post-link -split=auto` cuts a library's device code into modules, and llvm-foreach runs
them in batches of `BATCHLAS_SYCL_LINK_JOBS`, waiting for the slowest of each batch (checked with
`sleep` jobs: 5,1,1,1,1,1 s at `--jobs=2` take 7 s). A link costs the sum of its batch maxima,
and one module is never split, so the largest module is the floor of every relink of its library.

- The old cta link: 39 modules at `--jobs=4`, 741 s under load and at least 345 s idle. posv_tiny
  alone was 319 s (57 kernels, 193 MB of PTX; the TU compiles in 32 s), gesv_tiny 131 s, the other
  37 modules 260 s together.
- `gemm_kernels.cc` was two modules of 518 and 357 kernels, whose CPU AOT took 408 s and 390 s.
- The old default of 4 jobs also spread the backends link over 4 batches (171 + 153 + 97 s of
  syrk/herk/syr2k CPU AOT). `auto` (the logical cores, 4..32) is 20 here.

### build performance: CPU AOT dominated the sycl and backends links

The `opencl-aot` step that builds the `spir64_x86_64` image was most of each full-target link, not
NVPTX: 942 of 1,186 CPU-seconds in `libbatchlas_sycl`, 590 of 713 in `libbatchlas_backends`.
Every native family of trsm, symm, hemm, herk, her2k, syrk, syr2k and trmm requires `d.is_gpu` in
its `can_run`, so their CPU images could never run. gemm's cannot be dropped: a vendor-free CPU
queue runs it.

### build performance: the posv, gesv and level-3 library splits

Three `NO_CPU_TARGETS` libraries, each a device-code cluster of its own (no `SYCL_EXTERNAL` crosses
them, so a bad split would fail loudly in ptxas):

- `batchlas_extensions_posv` and `batchlas_extensions_gesv`: `posv_tiny.cc` and `gesv_tiny.cc`,
  out of `batchlas_extensions_cta`, for latrd's reason. An edit to any other cta TU no longer
  re-runs their codegen, and the two link in parallel with cta.
- `batchlas_extensions_level3`: the seven level-3 ops' `<op>.cc` (out of backends) and the two trsm
  kernel TUs (out of sycl). GPU-only, so no CPU AOT.
- gemm's reg and wide config rows became 23 generated shard TUs (`BATCHLAS_GEMM_SHARDS` and
  `gemm_shard.cc.in` in `src/ops/CMakeLists.txt`, launchers in `gemm_launch.hh`, rows in
  `gemm_rows.hh`). A `static_assert` in `gemm_kernels.cc` fails on a row with no shard. The main
  gemm TU compiles in 20 s (was 256 s), and the largest sycl module is now 59 s.

After the splits the libraries are: sycl 38 spir + 35 nvptx modules (194 s), backends 1 module
(1.5 s replay), level3 12 nvptx modules (about 47 s). One opencl-aot job peaks at 1.2 GB (the
largest process of a one-file rebuild, was 3.5 GB); the largest process of a full build is 1.6 GB,
and relinking all 16 libraries raised box memory by 11 GB at most.

### build performance: executables relinked on every library change

Every test and benchmark executable depended on all component `.so` files by path, so any library
change relinked about 150 of them: 2.5 min of a warm build (0.63 s for `stedc_tests` up to 9.1 s for
`trmm_candidates_tests`). `BATCHLAS_LINK_DEPENDS_NO_SHARED=ON` sets `CMAKE_LINK_DEPENDS_NO_SHARED`
and links executables with `-Wl,-z,now`: a stale executable missing a removed symbol then fails at
load time. Lazily bound, it passed whenever the missing call never ran (checked with a deliberate
deletion). `scripts/gate.sh` turns the option off.

### build performance: leaked device temporaries in tmpfs /tmp

The DPC++ driver writes `<tu>-sm_89-<hash>.{bc,s,o,cubin}` to `$TMPDIR`. A link that is killed or
fails leaves them, and `/tmp` here is tmpfs: 9,932 files, 38-43 GiB of RAM on a zero-swap box,
growing about 12 GiB/day. A full build added 777 files (3.4 GB) and a cta relink 117 (1.11 GB).
`BATCHLAS_PRIVATE_TMPDIR=ON` runs every compile and link through `<build>/batchlas-tmpdir`, which
sets `TMPDIR` to a fresh directory under `<build>/.tmp` and removes it on exit; afterwards 0 files
reached `/tmp` across a full build, relinks and failed builds. TMPDIR is not in ccache's hash.

### build performance: tuning_params.hh fan-out

`tuning_params.hh` reached 193 objects across every library through four default arguments in
public headers (`extensions.hh`, `internal/ormqr_blocked.hh`), so porting one tuned constant cost a
world rebuild (26.9 and 27.1 min measured). Those defaults are now `0`, resolved to the same MEDIUM
constant inside the `.cc`; an edit to the header recompiles 11 objects (`make -n`: 128 before). A/B
a constant with its `BATCHLAS_TUNE_*` override instead of editing the header per trial.

### build performance: before and after the posv and gesv split

`c960d7b1` against `cc9cee8c` (global offset off, posv/gesv split, no exe relinks, private TMPDIR),
same configure, -j8, one build at a time:

| Measurement | Before | After |
| --- | --- | --- |
| `steqr_cta.cc` edit, cta target | 617.8 s (607.9 s link) | 111.9 s (102.0 s link) |
| `posv_tiny.cc` edit | 637.0 s (cta target) | 200.7 s (posv target, 169.1 s link) |
| full build after reverting a cta/posv edit | 654 s, 90 exe links | 172 s, 0 exe links |
| `stedc.cc` edit, full default build | 71.4 s, 90 exe links | 22.1 s, 0 exe links |
| `tuning_params.hh` edit (`make -n`) | 128 compiles | 11 compiles |
| `/tmp` growth per full build | +777 files, +3.4 GB | 0 |
| `gemm_candidates_tests` | 1063 s, one entry | 35.8 s quick + 352.6 s `_heavy`, same 359 cases |

The cold full build was not faster (1260.8 s before, 1370.3 s after, with fewer ccache hits after
header changes): the sycl (772 s) and backends (680 s) links were still its critical path, which is
what the second round cut.

### build performance: before and after the level-3 split and gemm shards

`0ed2c1b3` against `57fa142d`; one-line append, then `--target batchlas_components -j8`:

| Edit | Before | After |
| --- | --- | --- |
| `src/ops/gemm/kernels/gemm_kernels.cc` | 1024 s | 213 s |
| `register_wide_transposed.hh` (every gemm shard includes it) | about 1024 s | 275 s |
| `src/ops/trmm/trmm.cc` | 520 s | 66 s |
| `trsm_native.cc` | about 763 s (sycl relink) | 62 s |
| `gemm.cc` (backends) | about 505 s (backends relink) | 12 s |
| sycl.so relink alone | 763 s | 194 s |
| fresh tree, full default build, warm ccache (two runs) | 900 s / 843 s | 312 s / 314 s |

"About" rows were not measured separately; they are the relink they trigger. 33 affected tests
and `run_local_checks.sh` passed after the change.

### build performance: what is still slow

- The sycl relink (194 s): many modules of under 60 s, so more link jobs help only on a bigger
  machine.
- The posv and gesv links (about 169 s each): one module each, the floor.
- Header edits: about 280 of 318 objects include `matrix.hh`, `settings`, `error`, `enums` or
  `workspace`.
- A new tree without a warm cache: every compile plus every link. 1260-1370 s was measured on
  c960d7b1/cc9cee8c with a partial ccache (118/77 hits), before the sycl and backends links were
  cut; no cold build of the current tree was measured. Seeding a new tree
  (`scripts/seed_build_tree.sh`, 9 s, then a 20 s first build) or a cache warmed by
  `scripts/prewarm_cache.sh` avoids it.
- Tests: `gemm_candidates_tests_heavy` (351 s) is the longest single ctest entry. The first
  `scripts/gate.sh` run takes 67 min: full tree build 14 min and serial ctest 21 min, vendor-free
  build 13 min and `-LE slow` 19 min.
