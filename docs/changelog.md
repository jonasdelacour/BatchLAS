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

Nothing yet.

## [1.0.0] - 2026-11-XX {#changelog-1-0-0}

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

<!-- wave-3b placeholder (Added): host routes and other additions from rel4/skip-burn and the
     other wave-3b packages' integration_needs; filled at integration, then this comment goes. -->

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
  `scripts/macos/adaptivecpp-patches/`, now 0001 to 0014. Patch 0014 infers Metal pointer address
  spaces over PHI and select cycles; without it the `complex<float>` paths of `potrf`, `posv`,
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

<!-- wave-3b placeholder (Changed): behaviour changes from the wave-3b packages' integration_needs;
     filled at integration, then this comment goes. -->

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
  - `steqr_wg` returned a non-orthonormal `Z` for masked `float` batches: a large per-pass fill
    of the rotation slots lost writes in AdaptiveCpp's batched Metal command buffer (mechanism not
    isolated). The sweep kernel now writes every rotation slot itself and the fill is gone
    (2dea7e5a).
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

<!-- wave-3b placeholder (Fixed): fixes from rel4/skip-burn and the other wave-3b packages'
     integration_needs; filled at integration, then this comment goes. -->

[Unreleased]: https://github.com/jonasdelacour/BatchLAS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jonasdelacour/BatchLAS/releases/tag/v1.0.0
