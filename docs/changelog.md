# Changelog {#changelog}

> **Status:** current · every user-visible change, newest first

All notable changes to BatchLAS are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). From 1.0.0 the major version is
the library's `SOVERSION`, and `find_package(BatchLAS 1.x)` accepts any 1.y install
(`SameMajorVersion`). What the promise covers is in
[versioning and compatibility](cpp-api.md#versioning-and-compatibility); how a release is cut is in
[releasing](developer/releasing.md). Numbers like `#258` are pull requests on
[github.com/jonasdelacour/BatchLAS](https://github.com/jonasdelacour/BatchLAS/pulls?q=is%3Apr);
short hexadecimal ids are commits of the 1.0 release branch.

## [Unreleased] {#changelog-unreleased}

### Added {#changelog-unreleased-added}

- `device::launch<Routine, Tier, T>::run(args...)` calls the routine `Routine` names in its tier,
  with the routine tag's modes, so one `launch` type describes the whole in-kernel call: with
  `using Potrf = device::launch<device::routine::potrf<Uplo::Lower>, Tier, T>`,
  `Potrf::run(ex, A, ws, s.live)` is `device::potrf<Uplo::Lower, Tier>(ex, A, ws, s.live)`. Every
  routine tag has a `device::routine_call`: the LAPACK ones in `<batchlas/device/lapack_run.hh>`
  (included by `<batchlas/device/lapack.hh>`), the sparse ones in
  `<batchlas/device/sparse_run.hh>`. The device examples use `run`, with the routine tag named
  `<Op>Routine` and its launch `<Op>`.

## [1.0.0] - 2026-11-22 {#changelog-1-0-0}

First release. Everything below is new relative to "no release"; the groups say what a user of a
pre-release commit (version 0.1.0) will notice.

### Added {#changelog-1-0-0-added}

**Batched operations** (one call factors, solves or decomposes every matrix of a batch):

- Level-3 and level-2 BLAS: `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`, and the
  Hermitian `hemm`, `herk`, `her2k`, with native kernels at cuBLAS parity on `sm_89` and `sm_120`
  (#60, #62, #188, #190, #195).
- Factorizations and solvers: `potrf`, `potrs`, `potri`, `posv`; `getrf`, `getrs`, `getri`,
  `gesv`, `inv` (#5, #115, #202); `geqrf`, `orgqr`, `ormqr` (#10, #21); Bunch-Kaufman
  `sytrf`, `sytrs`, `sysv` (#206); non-square `getrf` and `geqrf`, and `Upper` `potrf` (#195).
- Symmetric and Hermitian eigenvalues: `syev` (one-sided Jacobi, fused CTA, blocked and two-stage
  drivers, #39, #40, #45), `syevx` with LAPACK-style index (`il`/`iu`) and value (`vl`/`vu`)
  ranges through direct, Chebyshev-filtered and LOBPCG solvers (#44, #49, #54), `lanczos`,
  `ritz_values` (#24, #32), the generalized problem `sygst`, `sygvd`/`hegvd`, `sygvd_factored`
  (#205), the matrix function `syfn` (#203) and eigenvector alignment `syev_align` (#204).
- Tridiagonal and band reduction: `sytrd`, `steqr`, `stedc` (level-synchronous divide and
  conquer, #43, #55), `stebz`, `stein`, `tridiagonal_solver`.
- SVD: `gesvd` (one-sided Jacobi and blocked bidiagonalization, thin vectors, complex), `gebrd`,
  `bdsqr`, `bdsdc`, `ormbr` (#66, #67, #195).
- Sparse: CSR `spmm` and `iluk` (ILU(k)) as a LOBPCG preconditioner (#46, #105).
- Utilities: `ortho`, `norm`, `cond`, `transpose` (#6), `lascl`, matrix and CSR generators.
- Scalar types `float`, `double`, `std::complex<float>` and `std::complex<double>` (#31); Apple
  GPUs run `float` and `std::complex<float>` only.
- Masked batches: every op honours a `MatrixView` mask and skips the masked-out matrices (#210,
  #211, #212).

**Backends and platforms:**

- `CUDA` backend: native SYCL kernels routed per shape against cuBLAS, cuSOLVER and cuSPARSE.
- `NETLIB` backend: host LAPACKE/CBLAS plus native kernels on a CPU device (#7, #26).
- Vendor-free builds (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`): native batched `gemm`, `trsm`, `potrf`,
  QR, LU, `gemv` and `spmm` on the GPU (#94, #109), native factorizations at n = 4..512 (#115) and
  double `syev` for n = 449..1600 (#201).
- Host-native routes in vendor-free builds: a host queue with no LAPACKE/CBLAS runs BatchLAS's own
  kernels, the ones the GPU runs, through the `BATCHLAS_HAS_HOST_NATIVE` gate (a5827d36). Adopted
  by `syrk`, `herk`, `syr2k`, `her2k`, `symm`, `hemm`, `trmm`, `trsm` (93a04e38); `getrf`,
  `getri`, `getrs`, `gesv` (4ffe6fb8); `potrf`, `potrs`, `posv`, `sytrf`, `sytrs`, `sysv`, with
  `potri` through its children (f189aa8e); `geqrf`, `orgqr`, `ormqr` (4b9afda2); `gesvd` blocked,
  `sygst` blocked, `sygvd` (masked batches too) and `syfn` (f3110eca); `syev` blocked, `syevx`
  Direct, `steqr` and `stedc` divide and conquer (5d813bbd, 021e11db). `gemm`, `gemv` and `spmm`
  already ran natively on a vendor-free host. Which families stay GPU-only there, and why, is in
  [the host-native gate](design/vendor-independence.md#the-host-native-gate). These host routes
  honour a batch mask where a host BLAS refuses one: `gemm`, the level-3 ops, `potrf`, `posv`,
  `potrs`, `sygvd_factored`, blocked `ormqr`, `ortho`, `ritz_values`, `syfn`, `syfn_from_eigen`,
  `syev_align` and `lanczos` (tested in 8684a076).

- Two SYCL implementations from one tree: DPC++ (`intel/llvm` with CUDA) and AdaptiveCpp 25.10
  (#162, #178, #180); AdaptiveCpp on Linux is **experimental** and carries its own failure ledgers.
- Apple GPUs through AdaptiveCpp Metal (`Backend::SYCL`, `float` and `complex<float>` on the GPU,
  `double` on the host) with tuned `apple_m5_max` tables (#182, #216, #217, #218).

**Kernel selection and tuning:**

- Flat, per-device kernel selection: each op reads `tuned/<op>.<device>.txt`; tables ship for
  RTX 4090 (`sm_89`), RTX PRO 6000 (`sm_120`) and Apple M5 Max (#144, #146, #147, #164).
- `BATCHLAS_<OP>_ROUTE=<choice>` pins one op's kernel and `BATCHLAS_SELECT_TRACE=1` prints every
  choice (#109, #144, #224).
- `batchlas_tune`, a tiered tuning engine to measure tables for new hardware (#152, #186, #191).
- `batchlas::verify`, one verification library shared by tests, benchmarks and the tuner (#163).

**APIs:**

- A C++20 API (C++23 consumers supported) built on `Queue`, `Matrix`/`MatrixView`, option structs
  with designated initializers and returned `Event`s; workspaces sized by `*_buffer_size` and
  leased per queue (#50, #92, #183).
- A `batchlas::` exception hierarchy (`invalid_argument`, `unsupported`, `convergence_error`, ...)
  with per-item convergence status (#113, #194).
- `batchlas::linalg`: `matmul`, `solve`, `eigh`, `svd` and elementwise operations (#185).
- `batchlas::device`: BLAS and LAPACK callable from inside your own SYCL kernels (#250), usable
  header-only from a plain checkout with `-I <checkout>/include` (#259; f5b78b5c, fffe59a0).
- Four new ctest-checked device API examples in `examples/device/`: `geqrf_orgqr.cc` (QR with Q
  formed), `sytrf_sytrs.cc` (an LDL^T solve), `gesvd_any_n.cc` (the SVD at any n) and
  `spmm_csr.cc` (a CSR sparse-dense product), each in `float` and, where the device has fp64,
  `double` (ctest `device_example_geqrf_orgqr`, `_sytrf_sytrs`, `_gesvd_any_n`, `_spmm_csr`).
- `<batchlas/version.hh>` with `BATCHLAS_VERSION_MAJOR`, `_MINOR`, `_PATCH` and
  `BATCHLAS_VERSION_STRING`, included by `<batchlas.hh>` and usable header-only; the Python module
  exports the same string as `batchlas.__version__` (48ddcc27).
- `queue_from_sycl` interop with an existing `sycl::queue` (#199).
- Optional Python bindings (pybind11) with NumPy/SciPy interop, `batchlas.linalg` and a
  notebook examples suite (#41, #185).

**Build, install and documentation:**

- `make`, `make install`, `make check` over CMake presets; an installed CMake package
  (`find_package(BatchLAS CONFIG)`, target `BatchLAS::batchlas`) and a `batchlas.pc` pkg-config
  file (#70, #114, #258).
- Versioned shared libraries (`SOVERSION`), one exported target and a bounded export surface
  (#114); `@loader_path` install rpaths on macOS (#218).
- A source-only GitHub Release per `v*` tag, its notes taken from this file
  (`.github/workflows/release.yml`, 21fd1d35).
- A documentation site with the API reference, user guides, algorithm pages and the measurements
  behind every routing decision (#149, #153, #233, #234); the versioning and compatibility promise
  classifies every installed header as covered or not (94a82d64, 31bcc10c).
- CI: static checks, AdaptiveCpp compile and consumer jobs, and a GPU build-and-test gate (#111,
  #170). The consumer test checks that `find_package` accepts 1.0 and refuses 0.99, 1.1 and 2.0
  (955fc341, 2b905dcf, 54847b98).

**Docs (the device API reference):**

- A user guide to the in-kernel device API, [the device API](guide/device-api.md), with the
  compiled examples; [device-la.md](design/device-la.md) is now the design record, and the
  device measurements moved to [perf/device-la.md](perf/device-la.md).
- The API reference "In-kernel BLAS and LAPACK" is split into child groups (launch, executors,
  operands, arithmetic, BLAS levels 1 to 3, seven LAPACK families, sparse, and low-level);
  `batchlas::device::detail` is hidden, and the device inline macros no longer appear in
  signatures.
- The semantic-versioning promise for the device API covers the members of those child groups
  except the low-level group
  ([versioning and compatibility](cpp-api.md#versioning-and-compatibility)).
- Every device header has a Doxygen file brief and lists its members in its group: the core,
  launch and operand headers (`device.hh`, `core`, `arith`, `exec`, `grid`, `operand`, `interop`,
  `view`, `scalar`, `reduce`, `sg_partition`, `tier`, `launch`, `workspace`); BLAS levels 1 to 3,
  the LAPACK auxiliaries and sparse, with each routine's contract; the solvers (`potrf`, `potrs`,
  `posv`, `potri`, `trtri`, `lauum`, `getrf`, `getrs`, `getri`, `gesv`, `inv`, `sytrf`, `sytrs`,
  `sysv`, `sygst`) with their tiers, workspace and every `info` value; QR and SVD (`geqrf`,
  `larft`, `larfb`, `orgqr`, `ormqr`, `ormbr`, `ortho`, `gebrd`, `gesvd`, `bdsqr`, `bdsdc`); and
  the eigensolvers (`syev`, `sytrd`, `sy2sb`, `sb2st`, `steqr`, `stedc`, `stebz`, `stein`,
  `ritz_values`), each with its 32-wide sub-group requirement.
- Panel kernels, tuned-gemm internals and kernel-specific helpers are listed in the low-level
  group, which the compatibility promise does not cover.
- The device signature index moves under API reference, In-kernel BLAS and LAPACK. It is grouped
  by device API group and header, wrapped to 80 columns, and gives each signature its brief and a
  link to the member; low-level helpers are listed apart. Its page is now
  `group__design__device__la__api.html` (old bookmarks to `design_device_la_api.html` break).
- Docs theme: code examples no longer break lines inside linked identifiers, table code no longer
  breaks mid-identifier, scrollable code blocks and tables show a visible scrollbar, and two pages
  no longer show a stray backslash in code.

- Vendor-free hosts also run `syevx`'s DirectSubset, LOBPCG and Filtered families and `ormbr` natively,
  each honouring a batch mask (fd7f8fc7, 88092f2c, e485696e). Tests that used to skip there run: `gemm`
  with k = 0, blocked `sytrs` parts, the level-3 and `syfn` mask cases, the `sytrf` device API and
  blocked `gesvd` parity (c7d34ace, a6ed12c3, 953f010f).

### Changed {#changelog-1-0-0-changed}

- The library version is 1.0.0 and `SOVERSION` is 1 (`libbatchlas*.so.1`, `libbatchlas*.1.dylib`);
  the CMake package version check moved from `ExactVersion` to `SameMajorVersion`. Configure stops
  when `include/batchlas/version.hh` disagrees with `project(BatchLAS VERSION ...)` (48ddcc27).
- Public types live in `namespace batchlas`; nothing is declared in the consumer's global
  namespace (#110).
- `BATCHLAS_*` environment variables are read once into a `Settings` object (#112).
- `Matrix` is a `MatrixView`; the owning-argument forwarders are gone (#183).
- Headers install under `<prefix>/include/batchlas/`, included as `<batchlas/...>` (#70, #184).
- The Apple GPU build needs AdaptiveCpp with every local patch in
  `scripts/macos/adaptivecpp-patches/`, now 0001 to 0015. Patch 0014 infers Metal pointer address
  spaces over PHI and select cycles, and 0015 binds null pointer kernel arguments as null; without
  0014 the `complex<float>` paths of `potrf`, `posv`,
  `potrs`, `potri`, `sygvd` and `ortho` fail to compile their shaders (388da162, cb7244fe,
  3bc8fa93, b2284df3). See
  [macOS: Apple GPU through AdaptiveCpp Metal](guide/build.md#macos-apple-gpu-through-adaptivecpp-metal).
- `gesvd` in CTA mode solves the bidiagonal directly (`bdsdc`/`bdsqr`) like blocked, instead of
  through the normal equations; `BATCHLAS_GESVD_BIDIAG=normal` still selects the old path
  (61d8703e).
- The CTA entry points (`sytrd_cta`, `gebrd_cta`, `ormqx_cta`, `syev_jacobi_cta`,
  `syev_cta_fused`, `gesvdj_cta`) throw `batchlas::unsupported` on a CPU queue instead of failing
  in the SYCL runtime, and every caller that picks a CTA kernel by sub-group size also asks for a
  GPU (4a73d559).
- `potri` gains a `blocked:nb=32` candidate for complex types, last in candidate order, so a device
  with 32 KiB of local memory has a kernel above the CTA ceiling; `can_run` and the launch read the
  same ceiling (7e5d7baa).
- Blocked `sytrs` splits the batch into near-equal parts, so its `trsm` and `gemm` children select
  at one batch size and every item of a batch gets the same bits (9693cc0a).
- On a vendor-free host, `gesvd` refuses a masked batch (its `gebrd` and `ormbr` children cannot
  skip an item), and `syev` runs `blocked` only (f3110eca, e9e516da).

- `gesvd` with `BATCHLAS_GESVD_BIDIAG=normal`: a thin tall U is forced to the direct bidiagonal
  solve in CTA mode too, as in blocked mode, and complex types never take the normal equations
  (d166fba4).
- `her2k` past the device's grid runs as `fold` slices on every CUDA build, vendor-free included
  (3fe1b945).

### Fixed {#changelog-1-0-0-fixed}

- `gesvd_cta` left singular vectors lost orthogonality as eps times kappa squared
  ([defect 18](design/known-defects.md#defect-18-gesvd_cta-left-vectors-lose-orthogonality-as-eps-times-kappa-squared),
  61d8703e).
- Apple GPU (Metal) wrong answers:
  - `sygst` blocked, and through it `sygvd`: a partly live sub-group passed `sygs2_group`'s
    barriers twice, because AdaptiveCpp's Metal code repeats a barrier on both sides of a per-item
    branch (b733fbab). The diagonal group now fits the device's local memory, so `complex<float>`
    `blocked:nb=32` runs on Apple (55903879).
  - `syev_cta` returned a wrong largest eigenvalue on graded input: Metal flushes subnormals, and
    `larfg_lanes`' unscaled sum of squares lost every |x| below 2^-63; the norm is now scaled
    by a power of two (53fcac77).
  - Jacobi's float rotation cosine on AdaptiveCpp takes the FMA Newton step the NVPTX path uses
    (53fcac77).
  - `potri` admitted orders its CTA kernel could not launch (7e5d7baa).
  - `steqr_wg` returned a non-orthonormal `Z` for masked `float` batches: a per-pass fill of the
    rotation slots lost writes in AdaptiveCpp's batched Metal command buffer (the null-binding bug
    patch 0015 fixes, below). The sweep kernel now writes every rotation slot itself and the fill is
    gone (2dea7e5a).
  - `latrd_lower_run` (the blocked `sytrd` panel), the `latrd` grid kernel and `sb2st_hh_chase`
    scale a reflector's norm by a power of two before the sum of squares, as `larfg_lanes` does, so
    tiny entries keep their norm where subnormals flush (da319bdf, a66f431d, 02b51e38, 28a0b78a).
- DPC++ OpenCL CPU (vendor-free host) wrong answers and crashes:
  - `portable::group_broadcast` reached only lanes 0..15 of a 32-wide sub-group, which broke the
    blocked device `syev` from n = 18; it is now exact for every trivially copyable type (34eb2e99,
    1066d690).
  - `portable::joint_exclusive_scan` carried a wrong value between chunks, so device `stedc`
    gathered out of bounds (ed2c6148).
  - `ormqx_cta` did not pin its sub-group size to 32 and returned item-dependent garbage on a
    narrower sub-group (265915cb).
  - `sygs2_group`'s unrolled loads were miscompiled for `complex<double>` (85e71a45), and the
    `gesv` tiny tier's CPU image solved float n = 31, 32 wrongly (b9f9a8f0, 4fe69005).
  - `sytrd_sb2st`'s work-group kernel let later sub-groups read the next step's counters; they are
    now double-buffered (ef5757dc).
  - `syev` blocked and `syevx` Direct under-sized their workspace when the device asks for
    128-byte alignment (09d24192).
- `steqr` scaled a block of subnormal entries by infinity; `stedc`'s graded float leaves now
  converge (84782e3e).
- `linalg` wrappers freed a queue's buffers while a copy was still running when a call inside them
  threw; they now drain the queue first (81b11727).
- `steqr`: inverted QL/QR direction choice (#122) and a pathological runtime in the legacy path
  (#42).
- `stedc`: a FusedCta deadlock (#55) and values-only mode (#220).
- Work-group broadcasts fenced on AdaptiveCpp multipass CUDA/HIP, fixing wrong `syev` and
  `gesvd` results there (#219).
- Packed CTA kernels no longer return before a barrier (#209).
- ILU(k) factorization (#46); raw-view vendor calls, `lanczos`, AdaptiveCpp queue ordering and
  `trsm` with `alpha = 0` (#220).
- Backend parity and kernel-numerics fixes from the readiness audit (#168, #169, #171, #173, #175,
  #176).
- CPU-only builds (#126) and builds for NVIDIA architectures DPC++ has no alias for (#129).

- Apple GPU (Metal): AdaptiveCpp's Metal runtime left a null pointer kernel argument's buffer slot
  unset in flat argument mode, so a kernel read a stale pointer bound by an earlier encoder of the
  same command buffer. That was the lost `steqr_wg` fill above; patch 0015 binds null as null
  (cc59373d, ee120f5a).
- `gebrd` (`gebd2` and the group panel) and the legacy `sytrd` small path scale their reflector
  norms by a power of two, so entries below 2^-63 keep their reflector; the unused unscaled `larfg`
  branch is gone (bbad010e, b7907a00).
- The spectral norm (blocked `syev`) of a rank-deficient float matrix turned NaN: a column whose
  entries had all decayed to subnormals got an infinite reflector scale. `larfg_scalars` now
  returns the identity when 1 / (alpha - beta) overflows; every finite case is bit-identical
  (19794733).
- `syevx` Direct, Filtered and DirectSubset under-reported their workspace by up to one alignment
  quantum on an unaligned workspace base and threw `workspace_error`; they now size it with a
  `measuring()` replay (7dd5cbb4, 88092f2c).
- `gesvd` in CTA mode under `BATCHLAS_GESVD_BIDIAG=normal` could write past a thin tall U; only the
  CTA entry's thin-job refusal kept it unreachable (d166fba4).
- `syevx` DirectSubset needs a device with 32-wide sub-groups (`sytrd_sy2sb`'s WY update). Auto and
  `BATCHLAS_SYEVX_ALGORITHM` run Direct elsewhere (AdaptiveCpp's OpenMP host), and an explicit
  `SyevxAlgorithm::DirectSubset`, `syevx_direct_subset` or its size query throws
  `batchlas::unsupported` before any launch, instead of failing mid-call in `larft` (757348c6).

### Known limitations {#changelog-1-0-0-known-limitations}

- Some norms still use an unscaled sum of squares: `ortho`'s normalisation, `lanczos`' beta,
  `ritz_values`, `syevx` LOBPCG, `house2` in the tridiagonal solver and `lange` (Frobenius). On a
  device that flushes subnormals (Apple GPUs) they lose accuracy when every entry of a vector is
  below about 2^-63; no test input reaches that, and they are scheduled for 1.x
  ([known defects](design/known-defects.md)).
- AdaptiveCpp on Linux (generic and multipass CUDA) is experimental in 1.0.0: it is built and tested,
  but its known failures (`tests/known-failures-acpp*.txt`) are not release-blocking. ROCm is
  unsupported (code paths only).

[Unreleased]: https://github.com/jonasdelacour/BatchLAS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jonasdelacour/BatchLAS/releases/tag/v1.0.0
