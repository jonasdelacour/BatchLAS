# In-kernel BLAS and LAPACK: the device API {#guide_device_api}

> **Status:** current · checked 2026-11-22 against `include/batchlas/device/` for 1.0.0

`batchlas::device` is BLAS levels 1 to 3, the LAPACK routines BatchLAS ships and sparse products,
callable from inside your own SYCL kernel on one matrix per executor (a work-item, a sub-group or
part of one, or a work-group). This guide covers the launch frame, the conventions every routine
follows, executors, tiers, workspace, info values, the scalar types, what the device must provide,
and per family what exists, with the compiled examples. The members are in @ref api_device; every
signature, by group and header, with its brief, is on @ref design_device_la_api. The design
rationale is @ref design_device_la.

## What the device API is {#device-guide-what-it-is}

- **Your kernel, our routines.** You write the `parallel_for`; a device routine is a function you
  call inside it. BatchLAS's own host routes launch the same bodies, so `device::potrf` computes
  what the host `potrf` computes.
- **One matrix per executor.** An *executor* is the set of work-items that cooperates on one
  matrix. Every work-item of it calls the routine with the same arguments.
- **Everything that shapes code is a type.** The executor, the operand layout, the algorithm, the
  LAPACK modes (`Uplo`, `Side`, `Transpose`, ...) and the *tier* (where the matrix lives:
  registers, local memory or global memory) are template parameters, so a kernel compiles only the
  path it takes.
- **The caller owns the barriers.** A BLAS-level routine adds no barrier unless its signature
  names one; LAPACK-level routines are collective and make their results visible on return.
- **Routines return, never throw, on the device.** LAPACK-level routines return LAPACK's `info`.
  Host helpers (`with_tier`) throw `batchlas::` exceptions.

## Building a kernel {#device-guide-building-a-kernel}

The device API is header-only. Point the compiler at `include/` of a checkout or an install and
pass your usual SYCL flags; nothing is configured or linked but the SYCL runtime. The commands, the
`<batchlas/config.hh>` override macros and the AdaptiveCpp complex-arithmetic include are in
[The device API without building BatchLAS](../cpp-api.md#the-device-api-without-building-batchlas).
The interop overloads (`device::mat(KernelMatrixView)`, `device::item(view, b)`) take host views,
and building those views (`Matrix`, `Queue`) needs the built library.

### Which header to include {#device-guide-which-header-to-include}

- `<batchlas/device.hh>`: the core. Launch, tiers and workspace; executors; operands, views and
  interop; scalars and arithmetic; level 1, level 2, the register-tier level 3 and the LAPACK
  auxiliaries.
- `<batchlas/device/lapack.hh>`: the core and every LAPACK routine. One routine alone:
  `<batchlas/device/lapack/<routine>.hh>`, e.g. `<batchlas/device/lapack/potrf.hh>`.
- `<batchlas/device/level3/group.hh>`: the in-group level 3 (gemm and friends, one matrix per
  work-group, any m, n, k). `device.hh` does not include it; `lapack.hh` does, through the LAPACK
  headers built on it.
- `<batchlas/device/sparse.hh>`: CSR operands, `spmm` and ILU(k). Neither umbrella includes it.
- `<batchlas/device/detail/...>` and `batchlas::device::detail` are private.

### Macros a kernel uses {#device-guide-macros}

| Macro | Where | Use |
| --- | --- | --- |
| `BATCHLAS_REQD_SG_SIZE(n)` | after the kernel lambda's parameter list | the sub-group size the tier is compiled for, `L::sg_size`. DPC++: `[[sycl::reqd_sub_group_size(n)]]`; AdaptiveCpp: empty, kernels run at the native sub-group size |
| `BATCHLAS_DEVICE_INLINE`, `BATCHLAS_DEVICE_HELPER` | on your own device functions, optional | always-inline and plain `inline`; every public routine carries one of them |
| `BATCHLAS_DEVICE_BODY` | on a lambda passed to a loop routine (`for_each_owned` and friends) | always-inline |
| `BATCHLAS_LAUNCH_BOUNDS(max, min)` | on a kernel, optional | DPC++ NVPTX device code: `intel::max_work_group_size` and `intel::min_work_groups_per_cu`; empty elsewhere |

Call routines from the kernel lambda or from a named function, never first through a functor's
`operator()` or a local lambda: DPC++'s kernel front end mishandles a kernel body entered that way.

## Quick start: the launch frame {#device-guide-quick-start}

Every routine with a tier is launched this way; only the call inside the kernel changes. This is
the kernel of `examples/device/potrf_any_n.cc`, which ctest builds and runs (the whole program is
under [Worked examples](#device-guide-worked-examples)):

\snippet{trimleft} device/potrf_any_n.cc device-guide-frame

On the host, `with_tier` picks one tier for this `n` and device and calls the lambda with it. Each
tier is its own kernel. `launch<R, Tier, T>` (`launch.hh`) then gives that tier's launch:

| Member of `launch<R, Tier, T>` | Meaning |
| --- | --- |
| `executor`, `exec(it)` | `tier_info<R, Tier, T>::executor`, built as `executor{it}` |
| `wg_size`, `sg_size` | the work-group and sub-group size the tier is compiled for |
| `per_group`, `team(it)` | matrices per work-group (`32 / N` for a packed tier, else 1); this item's executor in its work-group |
| `nd_range(batch)` | `ceil(batch / per_group)` work-groups of `wg_size` items |
| `slot(it, batch)` | `MatrixSlot{b, live}`: this executor's matrix index and whether it is a real matrix |
| `buffer_size(extents...)` | one matrix's `WorkspaceSize`; the same value as `device::buffer_size<R, Tier, T>(extents...)` |
| `workspace(it, slm, gws, sz)` | this matrix's `Workspace`: its `local_bytes` slice of `slm` and `global_slice(gws, sz, b)` |
| `local_workspace<L>(h, sz)` (host) | the work-group's `LocalWorkspace`: `per_group` slices, 16-byte aligned, one slab when the tier needs none |

The routine tag `R` names the routine and its modes: `device::routine::<op><modes...>`, e.g.
`routine::potrf<Uplo::Lower>`, `routine::ormqr<Side::Left, Transpose::ConjTrans>`,
`routine::syev<JobType::EigenVectors, Uplo::Lower>`. The in-group level 3 has no routine tag and
builds its frame by hand ([Level 3](#device-guide-family-blas3)).

## Conventions {#device-guide-conventions}

The rules every public routine follows. If you know the netlib routine, these give you the call;
each header states its routine's contract.

| Rule | What you write |
| --- | --- |
| C1 Naming | The netlib name, lower case, one name for every scalar type: `potrf`, `gebd2`, `larfg`, never `_real` / `_complex`. Overloads differ only in operand kind (a `Thread` register tile or a group view), never in argument order or return type. |
| C2 Template parameters | `<modes in netlib order..., Tier or Cfg or Alg, Ops, Scope, deduced...>`; modes are enums: `ormqr<Side::Left, Transpose::ConjTrans, Tier>`, `sygst<Itype::AinvB, Uplo::Lower, Tier>`, `ormbr<Vect::Q, Side::Left, Transpose::NoTrans>`, `syev<JobType::EigenVectors, Uplo::Lower, Tier>`. |
| C3 Runtime arguments | `(ex, operands in netlib order, free sizes (kd, k), ws, opts, live)`; sizes, `ld` and LAPACK work arrays are dropped. BLAS puts alpha before its product and beta before the output at every level: `axpy(ex, alpha, x, y)`, `gemm<TA, TB, Cfg>(ex, alpha, A, B, beta, C, ws)`; out-of-place trmm is `trmm_out`. LAPACK: `getrs(ex, A, ipiv, B, ws, live)`, `ormqr(ex, A, tau, C, ws)`, `sb2st(ex, ab, d, e, tau, kd, ws)`. |
| C4 Operands are views | `Mat`, `Vec`, `ItemMat`, `Frag`, `LaneVec`; `ipiv` and `tau` are `Vec`s. Build them from pointers with `device::mat(p, m, n, ld)` and `device::vec(p, n[, inc])` (`operand.hh`), or from host views with `device::mat(view)`, `device::vec(view)` and `device::item(view, b)` (`interop.hh`). |
| C5 Executor | The tier's `tier_info<R, Tier, T>::executor`, built as `Ex ex{it}` (`launch<>::exec`); a partition of a sub-group is `device::make_partition<P>(sg)`. Indices and sizes are `int`. |
| C6 Tier names residency | `Reg` registers, `Cta` local memory, `Blocked` global memory plus a local panel or diagonal block. |
| C7 Workspace | `const Workspace& ws` in the C3 position (`= {}` where a routine may need none); `ws.global` is this matrix's slice. Size it with `launch<R, Tier, T>::buffer_size(extents...)` or `device::buffer_size<R, Tier, T>(extents...)`; there is no per-routine sizer. |
| C8 Return | A LAPACK-level routine returns `int info`, the same on every item. BLAS returns `void` or the reduced value. Options are `<Op>Options` structs with defaults (`SyevOptions<R>`, `StedcOptions<R>`, `BdsqrOptions<R>`, `SteqrOptions`, ...), `R` the real type, passed after `ws`. |
| C9 Dead slots | A packed tier (`per_group > 1`) takes a trailing `bool live = true`. A dead slot passes `s.live == false` and still calls: it reads and writes nothing, reaches every collective and returns 0. |
| C10 Modes, tags, scopes | Modes are enums (`Side`, `Uplo`, `Transpose`, `Diag`, `Direct`, `Refl`, `DiagFill`, `Vect`, `Itype`, `Axis`, `Storage`, `PackOrder`, and the host's `JobType`, `SvdVectors`, `NormType`, `OrthoAlgorithm`); one `Scope{Own, Sub, Group}`. Policy tags live only in `alg::` (algorithm, distribution, numerical convention), `arith::` (scalar arithmetic) and `epi::` (epilogue); reduction operators in `reduce_op::`. |
| C11 Style | `class` template parameters, the executor argument `ex`, lower-case runtime names. |

Where a header does not yet follow a rule, the design record lists it under
[open debts](../design/device-la.md#device-la-open-debts).

## Executors {#device-guide-executors}

| Executor | Items | Used by |
| --- | --- | --- |
| `WorkGroup<WG, SG>` | the whole work-group, `WG` a compile-time size | Cta and Blocked tiers, the in-group level 3 |
| `Teams<WG, P, Order, SG>` | `WG / P` teams of `P` items | team-per-row level 2 (gemv, symv) |
| `SubGroup<SG>` | one sub-group | panel factorizations, shuffle reductions |
| `Partition<P, Masked, Impl>` | `P` lanes of a sub-group, `32 / P` matrices per sub-group | Reg tiers (lane r holds row r) |
| `SgpPartition<P>` (`sg_partition.hh`) | a `Partition` over the backend's sub-group partition, `make_partition<P>(sg)` | the Cta tiers of syev, steqr, sytrd, gebrd; sb2st |
| `Thread` | one item | serial steps (lagtf / lagts, trsv) |
| `RtGroup<Scope, SG>` | a work-group or sub-group whose width is a run-time value | routines launched at several widths (sytrf, getrs) |

- Every item of the executor calls a routine with the same arguments. An item that returns early
  deadlocks the routine's next barrier or shuffle; a dead slot passes `live = false` instead.
- `ex.leader()` is the item that writes a uniform result such as `info`.
- Only `WorkGroup` and `RtGroup<Scope::Group>` have a work-group barrier. `sync(ex)` is the barrier
  of the executor's scope: a sub-group barrier on `Teams`, `SubGroup` and `Partition`, nothing on
  `Thread`.

## Tiers and with_tier {#device-guide-tiers}

| Tier | Where the matrix lives | Bound |
| --- | --- | --- |
| `tier::Reg<N, Exact>` | registers of a `Partition<N>` or `SubGroup` | `n <= N <= 32` (complex<double>: 16) |
| `tier::Cta<N, Alg>` | local memory of one `WorkGroup` or `SgpPartition<N>` | `n <= N`, and the plan fits the device's local memory |
| `tier::Blocked<NB, Alg>` | global memory; an NB-wide panel or diagonal block in local memory, the rest in your global workspace | any n |
| `tier::upto<NMax>` | the first tier of the routine's list admitting `NMax` (compile time) | |

Per routine, tier and scalar, `tier_info<Routine, Tier, T>` states `executor`, `wg_size`,
`sg_size`, `per_group`, `admits(n)` and `buffer_size(extents...)`. `tiers_t<Routine, T>` lists the
routine's tiers in the order `with_tier` tries them: by bound, smallest first, except a fallback
that follows the tier it replaces (sygst's `Blocked<16>` after `Blocked<32>`). `ortho` and `ritz_values` have a `tier_info` but no tier list:
name their tier yourself.

`with_tier<Routine, T, List = tiers_t<Routine, T>>(device_or_limits, n, f, name = Routine::name)`
runs on the host. It calls `f(Tier{})` for the first tier of `List` that admits `n`, fits the
device's local memory (`per_group` slices, less the launch scratch below) and maximum work-group
size, runs at a sub-group size the device has and, for a double `T`, finds fp64.

- Each listed tier is a separate instantiation of the kernel `f` launches, and exactly one runs.
  Pass your own `List` to compile fewer kernels, e.g. for potrf
  `tier::list<tier::Cta<64>, tier::Blocked<32>>`.
- When no tier admits `n`, it throws `batchlas::invalid_argument` with the message
  `"<name>: no device tier admits n=<n>"`.
- `device_or_limits` is a `sycl::device` or a `DeviceLimits` (local memory, sub-group sizes,
  maximum work-group size, fp64, per-item launch scratch), which you can fill by hand to plan for
  another device.

## Workspace {#device-guide-workspace}

| Type | Meaning |
| --- | --- |
| `WorkspaceSize` | `local_bytes` and `global_bytes` of ONE matrix, and their `align` (16) |
| `Workspace` | the matrix's slice: a decorated local pointer `local` and a global pointer `global` |
| `Plan` | a constexpr bump allocator over the two spaces; the free functions `carve_local` / `carve_global` turn its slots into pointers |
| `global_slice(base, size, b)` | matrix b's global slice of one buffer of `size.global_bytes * batch` |

- Size with `device::buffer_size<routine::<op><modes...>, Tier, T>(extents...)`. It is the plan
  the routine carves its pointers with, so the size and the use cannot drift apart.
- Local bytes go into the 16-byte aligned `sycl::local_accessor` that `local_workspace<L>(h, sz)`
  makes for the work-group. Global bytes are one device allocation of `global_bytes * batch`.
- Reg tiers need none. A Blocked tier's global slice holds what its level 3 cannot keep in local
  memory (ormqr's V, T and the larfb work arrays; stedc's two n x n blocks).
- The in-group level 3 has no routine tag: its stage is the first `l3::smem_bytes<Cfg>()` bytes of
  `ws.local`, and it needs no global part.

## Info values and errors {#device-guide-info-and-errors}

- A LAPACK-level routine returns LAPACK's `info`, the same value on every item of the executor:
  - `0`: success.
  - `k > 0`: the routine's own failure, with LAPACK's partial-result contract (see the table
    under [Scalar types](#device-guide-scalar-types)).
  - `< 0`: an invalid argument, where the routine checks its arguments. The numbering is per
    family, not one rule; each routine's reference lists its values, and its group says which
    rule it follows:
    - Cholesky (@ref api_device_lapack_chol), `syev`, `sytrd`, `steqr` and `sygst` count from
      the executor, `ex` being argument 1: -2 is `A` (not square, or `n` above the tier's bound)
      and the next numbers are the next operands and the workspace.
    - QR, `gebrd`, `ormbr`, `gesvd`, `sy2sb` and `sb2st` count from the first matrix, `A` being
      argument 1 (`geqrf`: -2 a short `tau`, -3 a null workspace).
    - `sytrf`, `sytrs` and `sysv` return LAPACK's own argument numbers (-1 the executor width,
      -2 `N`, -4 or -5 `LDA`, -7 `B`, -8 `LDB`).
    - `ortho` returns -2 when `k` exceeds the `tier::Cta<N>` bound.
    - `getrf`, `getrs`, `getri`, `gesv`, `inv`, `stedc`, `stebz`, `stein`, `bdsqr`, `bdsdc` and
      `ritz_values` check no arguments and return no negative value (`getrs` always returns 0).
- A routine never throws in device code. `info` is not written anywhere for you: store it from
  `ex.leader()`, as the quick start does.
- Host-side errors throw `batchlas::` types whose message opens with the routine's name (`with_tier`
  above); see [What gets thrown](../cpp-api.md#what-gets-thrown).

## Scalar types {#device-guide-scalar-types}

Scalars are `float`, `double`, `std::complex<float>`, `std::complex<double>` and any plain
`{re, im}` struct such as `Cx<R>`, all through `Arith<T>`. `std::complex` arithmetic goes through
`Arith`, never the Annex G `operator*` (no `__mulsc3` in device code). Pivots are 1-based `int32`.

| Routine | Scalar types | `info > 0` |
| --- | --- | --- |
| potrf | all four; `Reg<32>` not complex<double>, whose Blocked tier is `Blocked<16>` | the leading minor of order k is not positive definite |
| potrs, posv, potri, trtri, lauum | all four (posv complex<double>: `Blocked<16>`) | posv as potrf |
| getrf, getrs, gesv, getri, inv | all four; getrf `Reg<32>` not complex<double> | getrf, gesv, getri, inv: `U(k, k)` is exactly zero |
| sytrf, sytrs, sysv | all four | `D(k, k)` is exactly zero |
| geqrf, orgqr, ormqr, larft, larfb, ortho | all four (ormqr, larfb: complex `Trans` does not compile, use `ConjTrans`) | |
| syev, sytrd, sygst, sy2sb, sb2st | all four | syev: the tridiagonal solver did not converge |
| steqr | real `d`, `e`; `z` real or complex on `Blocked<1>` | Cta: 1 when a block did not converge; Blocked: the number of `e(j) != 0` left |
| stedc, stebz, stein | real only | stedc: a secular root did not converge |
| gebrd | all four on `Blocked`; `Cta<N>` real only | |
| gesvd | all four | bdsqr did not converge |
| bdsqr, bdsdc | real only | did not converge |
| spmm, iluk_factorize, iluk_apply | all four | |

## Device requirements {#device-guide-device-requirements}

- **32-wide sub-groups.** Every tier is compiled for `sg_size = 32`, and `with_tier` skips a tier
  whose sub-group size the device does not list. The routines without a tier (`larft`, `larfb`,
  the in-group level 3) and the Blocked tiers that reach `larft` or `sy2sb` assume 32-wide
  sub-groups and do not check it at run time; see the sub-group-32 entry of
  [known defects](../design/known-defects.md#unverified-candidates).
- **Local memory.** A Cta tier holds the matrix in local memory and a Blocked tier a panel.
  `with_tier` compares the plan with the device's local memory, less the 8 bytes per item that
  AdaptiveCpp's Metal backend appends to every launch (`local_mem_bytes(d, wg)`;
  [the launch budget](../design/sycl-implementations.md#sycl-impl-slm-budget)).
- **fp64.** A `double` or `std::complex<double>` tier needs a device with fp64; `with_tier` skips it
  otherwise.
- **Apple GPUs (AdaptiveCpp Metal).** No fp64, 32 KB of local memory, and kernels run at the native
  sub-group size (32). The toolchain must carry BatchLAS's AdaptiveCpp patches: without them the
  complex<float> shaders of potrf, posv, potrs, potri and ortho do not compile, and a null pointer
  kernel argument is misread
  ([macOS: Apple GPU through AdaptiveCpp Metal](build.md#macos-apple-gpu-through-adaptivecpp-metal),
  [toolchain traps](../design/known-defects.md#known-defects-toolchain-traps)). A header-only
  build for Metal defines `BATCHLAS_DEVICE_NO_VECTOR_TYPES=1`.
- **CPU devices.** Local memory and sub-group sizes differ per runtime, and `with_tier` reads them.
  The DPC++ OpenCL CPU runtime has its own traps, listed under
  [toolchain traps](../design/known-defects.md#known-defects-toolchain-traps).

## Routines by family {#device-guide-routines-by-family}

Each family links its reference group, where every member has its contract. Tier lists are
`tiers_t<routine::<op>, T>`, in the order `with_tier` tries them.

### Launch, tiers and workspace {#device-guide-family-launch}

@ref api_device_launch. `launch`, `with_tier`, `tier::Reg` / `Cta` / `Blocked` / `upto` / `list`,
`tier_info`, `tiers_t`, `resolve_tier_t`, `DeviceLimits`, `local_mem_bytes` (`tier.hh`);
`local_workspace<L>`, `LocalWorkspace`, `buffer_size<R, Tier, T>`, `MatrixSlot` / `matrix_slot`
(`launch.hh`); `WorkspaceSize`, `Workspace`, `Plan`, `carve_local`, `carve_global`,
`sub_workspace`, `global_slice` (`workspace.hh`).

### Executors {#device-guide-family-exec}

@ref api_device_exec. `WorkGroup`, `Teams`, `SubGroup`, `Partition`, `Thread`, `RtGroup` (members
`bcast`, `uniform`, `shuffle`, `shift`), `sync`, `sync<Scope>`, `Scratch`, `Slot<N>`, `Grid1D`
(`exec.hh`); `SgpPartition`, `make_partition<P>(sg)` (`sg_partition.hh`); `Grid2D`,
`batched2d_shape`, the host launcher `batched2d` (`grid.hh`).

### Operands, views and interop {#device-guide-family-operands}

@ref api_device_operands. Extents, `Scope`, layout and scalar tags, `MaybeZero`, `static_for`
(`core.hh`); `Vec`, `Mat`, `vec`, `mat`, `Frag`, `Tile`, `Published`, `load` / `store` / `fill`,
`LaneVec`, `mat_strided`, `mat_local` (`operand.hh`); `row` / `col` / `window`, `op_view<TR>`,
`conj_view`, `herm_view`, `sym_view`, `PackedTri`, `slice` (`view.hh`); `vec(VectorView)`,
`vec_unit`, `mat(KernelMatrixView)`, `item(view, b)`, `item(view, b, m, n)`, `ItemMat`, `ItemVec`
(`interop.hh`).

### Scalars and arithmetic {#device-guide-family-arith}

@ref api_device_arith. `Arith<T>`, `is_complex_v`, `real_t`, the `arith::` policies and
`arith_t<Ops, T>`, `abs1`, `abs2`, `conj`, `conj_if`, `store_epi` with the `epi::` epilogues
(`arith.hh`); `lartg`, `Givens`, `lae2`, `laev2`, `las2`, `larfg_scalars`, `lamch`, `SafeRecip`
(`scalar.hh`). The scalar routines take no executor and also run on the host.

### BLAS level 1 and reductions {#device-guide-family-blas1}

@ref api_device_blas1. `axpy`, `scal`, `copy`, `swap`, `fill`, `dot`, `dotc`, `nrm2<Alg>`,
`lassq`, `asum`, `iamax`, `allreduce`, `for_each_owned` and friends (`level1.hh`); `shuffle_xor`,
`shuffle_down`, `butterfly`, `ArgMax`, `argmax_allreduce`, `alg::NanAsInf` / `NanSkip` /
`NanPropagate` (`reduce.hh`). `nrm2` has no default algorithm: name `alg::Blue` for overflow
safety or `alg::Unscaled` for speed.

### BLAS level 2 {#device-guide-family-blas2}

@ref api_device_blas2. `gemv`, `symv` / `hemv`, `ger` / `geru` / `gerc`, `her2`, `trmv`, `trsv`,
`dgmm`, `gbmv` (`level2.hh`), in BLAS argument order `(ex, alpha, A, x, beta, y)`.

### BLAS level 3 {#device-guide-family-blas3}

@ref api_device_blas3. The register tier `gemm`, `gemm_sub` and `trsm` on a `Tile` (`level3.hh`),
and the in-group `gemm`, `symm` / `hemm`, `syrk` / `herk`, `syr2k` / `her2k`, `trmm_out` and
`trsm` on one matrix per work-group (`level3/group.hh`). The in-group forms take an `l3::cfg`
preset instead of a tier; it fixes the executor (`WorkGroup<Cfg::threads>`) and the local stage
(`l3::smem_bytes<Cfg>()` bytes at the start of `ws.local`). The frame of
`examples/device/gemm_group.cc`:

\snippet{trimleft} device/gemm_group.cc device-guide-gemm

### Cholesky {#device-guide-family-chol}

@ref api_device_lapack_chol.

| Routine | Tiers |
| --- | --- |
| `potrf<UL, Tier>(ex, A, ws, live)` | `Reg<8>`, `Reg<16>`, `Reg<32>` (not complex<double>), `Cta<64>`, `Cta<128>`, `Blocked<32>` (complex<double>: `Blocked<16>`) |
| `potrs<UL, Tier>(ex, A, B, ws)`, `posv<UL, Tier>(ex, A, B, ws)` | `Blocked<32>` (posv complex<double>: `Blocked<16>`) |
| `potri<UL, Tier>`, `trtri<UL, DG, Tier>`, `lauum<UL, Tier>`, each `(ex, A, ws)` (`potri.hh`) | `Blocked<32>` |

### LU {#device-guide-family-lu}

@ref api_device_lapack_lu. `ipiv` is a `Vec<int>`, 1-based.

| Routine | Tiers |
| --- | --- |
| `getrf<Tier>(ex, A, ipiv, ws, live)` | `Reg<8>`, `Reg<16>`, `Reg<32>` (not complex<double>), `Cta<64>`, `Cta<96>`, `Cta<128>`, `Blocked<32>` |
| `getrs<TR, Tier>(ex, A, ipiv, B, ws, live)`, `gesv<Tier>(ex, A, ipiv, B, ws, live)`, `getri<Tier>(ex, A, ipiv, C, ws, live)`, `inv<Tier>(ex, A, Ainv, ws, live)` | `Blocked<32>` |

The getrs tiers share getrf's executor and local plan, so both run in one kernel
(`examples/device/getrs_any_n.cc`):

\snippet{trimleft} device/getrs_any_n.cc device-guide-getrs

### Symmetric indefinite (LDL^T) {#device-guide-family-ldl}

@ref api_device_lapack_ldl.

| Routine | Tiers |
| --- | --- |
| `sytrf<UL, Tier>(ex, A, ipiv, ws)` | `Cta<32>`, `Blocked<1>`; another `Blocked<NB>` when named |
| `sytrs<UL, Tier>(ex, A, ipiv, B, ws)` | `Cta<32>`, `Cta<64>`, `Cta<128>`, `Blocked<1>`; another `Blocked<NB>` when named |
| `sysv<UL, Tier>(ex, A, ipiv, B, ws)` | `Cta<32>` (one fused local tile), `Blocked<1>`; another `Blocked<NB>` when named |

### QR and orthogonal transforms {#device-guide-family-qr}

@ref api_device_lapack_qr.

| Routine | Tiers |
| --- | --- |
| `geqrf<Tier>(ex, A, tau, ws)`, `orgqr<Tier>(ex, A, tau, ws)`, `ormqr<SD, TR, Tier>(ex, A, tau, C, ws)` | `Blocked<larfb_default_nb<T>()>`: `Blocked<16>` for `double`, `Blocked<32>` for `float` and both complex types |
| `larft(ex, V, tau, T)`, `larfb<SD, TR, Direct, Cfg>(ex, V, T, C, ws)` | no tier; `larfb` takes an `l3::cfg` preset |
| `gebrd<Tier>(ex, A, d, e, tauq, taup, ws)` (m >= n) | `Blocked<16>` for every T; real T first `Cta<4>` .. `Cta<32>`, called as `gebrd<Tier>(ex, it, A, d, e, tauq, taup, ws, live)`; another `Blocked<NB>`, 1 <= NB <= 64, when named |
| `ormbr<Vect, SD, TR>(ex, A, tau, C, ws)` | `Blocked<1>` |
| `ortho<TR, Tier, Cfg, Alg>(ex, A, ws)` (CGS2, Cholesky QR) | no list: `Cta<N>`, or for CGS2 also `Blocked<NB>`, with N or NB named by you |

### Symmetric eigensolvers {#device-guide-family-eig}

@ref api_device_lapack_eig. `R` is the real type of `T`.

| Routine | Tiers |
| --- | --- |
| `syev<J, UL, Tier>(ex, A, W, ws, SyevOptions<R>{}, live)` (ascending `W`; `A` holds the eigenvectors) | `Cta<4>`, `Cta<8>`, `Cta<16>`, `Cta<32>`, `Blocked<32>` |
| `sytrd<UL, Tier>(ex, A, d, e, tau, ws, live)` | `Cta<4>` .. `Cta<32>`; `Blocked<32>` for `Uplo::Lower` only |
| `steqr<J, Tier>(ex, d, e, z, ws, SteqrOptions{}, live)` (eigenvalues unsorted in `d`) | `Cta<4>` .. `Cta<32>`, `Blocked<1>` |
| `stedc<J, Tier>(ex, d, e, w[, z], ws, StedcOptions<R>{})` | `Blocked<1>` |
| `stebz<Tier>(ex, d, e, w, m, ws, StebzOptions<R>{})` (`m` an `int&` out), `stein<Tier>(ex, d, e, w, kb, z, ws, SteinOptions<R>{})` | `Blocked<1>` |
| `sygst<Itype, UL, Tier>(ex, A, B, ws)` | `Cta<8>`, `Cta<16>`, `Cta<32>`, `Blocked<32>`, then `Blocked<16>` where the `Blocked<32>` diagonal block does not fit local memory |
| `sy2sb<UL, Tier>(ex, A, AB, tau, kd, ws)` (dense to band, `Uplo::Lower` only) | `Blocked<32>`; `Blocked<64>` when named |
| `sb2st<UL, Tier>(ex, ab, d, e, tau, kd, ws)` (band to tridiagonal) | `Cta<4096>`, `Blocked<32>` |
| `ritz_values<Tier, Cfg>(ex, A, V, w, ws)`; tier-free `ritz_values(ex, A, V, w)` | no list: `Blocked<NB>`, with NB named by you |

There is no device `syevx` or Jacobi solver: `syevx.hh` (the host syevx route's selection step)
and `jacobi.hh` (the host Jacobi kernels' rotations) hold only `device::detail` code. The merge kernels of `stedc` and `stein` (`laed2`, `laed3`,
`laed4`, `lagtf`, `lagts`) are in @ref api_device_lowlevel.

### Singular value decomposition {#device-guide-family-svd}

@ref api_device_lapack_svd.

| Routine | Tiers |
| --- | --- |
| `gesvd<JU, JVT, Tier>(ex, A, S, U, Vh, ws)` (any m x n) | `Blocked<16>` |
| `bdsqr<Tier>(ex, d, e, s[, U, Vh], ws, BdsqrOptions<R>{})`, `bdsdc<Tier>(ex, d, e, s, U, Vh, ws, BdsdcOptions{})` (real) | `Blocked<1>`, the default `Tier` |

The panel leaves of the QR and SVD routines, `geqr2`, `geqr2_reg`, `gebd2` (the unblocked
bidiagonal reduction) and `labrd` (the panel of the blocked one), are low-level helpers in
@ref api_device_lowlevel, not covered by the compatibility promise; call `geqrf` and `gebrd`.

### LAPACK auxiliaries {#device-guide-family-aux}

@ref api_device_lapack_aux. `lacpy`, `laset`, `mirror`, `laswp<Direct>`, `lapmr`, `rot`, `lasr`,
`largv` / `lartv` / `lar2v`, `larfg`, `larfg_lanes` (one element per item), `larf`,
`lanst<NormType>`, `lange<NormType>` (a matrix norm), `laqge`, `lascl`, `lacgv`, `trttp` /
`tpttr`, `geam` (`lapack_aux.hh`).

### Sparse {#device-guide-family-sparse}

@ref api_device_sparse (`<batchlas/device/sparse.hh>`).

| Routine | Tiers |
| --- | --- |
| `spmm<TA, TB, Tier>(ex, alpha, A, B, beta, C, ws)`, `A` a CSR item from `csr_item(view, b)` | `Blocked<64 / sizeof(T)>`: `Blocked<16>` for `float`, `Blocked<8>` for `double` and `complex<float>`, `Blocked<4>` for `complex<double>` |
| `iluk_factorize<Tier>(ex, A, plan, M, opts, ws)` (options before `ws`), `iluk_apply<Tier>(ex, M, rhs, y, ws)` | `Blocked<1>` |

- `spmm` with `NoTrans` `A` gives a result bit-identical to the host route. With `Trans` or
  `ConjTrans` `A` it scales `C` by `beta` and then scatters into it with relaxed atomics, in an
  unspecified order, so the last bits of `C` may vary from run to run.
- `IlukPlan`, the symbolic plan `iluk_factorize` takes, has no installed builder: the library's
  host symbolic phase builds it for the ILU(k) route, and that code is not an installed header.
  Outside the library you can use `iluk_factorize` only by filling the plan's index arrays
  yourself, to the layout `IlukPlan` documents. `iluk_apply` also runs on a factor the host
  ILU(k) made (`iluk_batch` of its view).

### Low-level helpers {#device-guide-family-lowlevel}

@ref api_device_lowlevel lists the helpers that live in `batchlas::device` because the public
routines use them. They are documented so the public signatures can be read, and they are **not**
covered by the compatibility promise
([Versioning and compatibility](../cpp-api.md#versioning-and-compatibility)).

## Worked examples {#device-guide-worked-examples}

Each is a complete program in `examples/device/` that ctest builds and runs
(`device_example_<name>`); it exits 0 only when its residuals are small and every `info` it
reads is 0.
`device_header_only_tests` also builds each one from `include/` alone.

### Cholesky for any n: potrf_any_n.cc {#device-guide-example-potrf}

`with_tier` picks `Reg`, `Cta` or `Blocked` from `n`; batch 253 leaves dead slots in the packed
`Reg<8>` launch.

\include device/potrf_any_n.cc

### LU solve for any n: getrs_any_n.cc {#device-guide-example-getrs}

`getrf` then `getrs` in one kernel, with off-diagonal pivots.

\include device/getrs_any_n.cc

### Symmetric eigenproblem for any n: syev_any_n.cc {#device-guide-example-syev}

`Cta<N>` runs one matrix per `SgpPartition<N>` (32 / N per work-group), `Blocked<32>` one per
`WorkGroup<256>`.

\include device/syev_any_n.cc

### In-group gemm: gemm_group.cc {#device-guide-example-gemm}

No tier: the `l3::cfg` preset fixes the executor and the local stage.

\include device/gemm_group.cc

### QR, then Q: geqrf_orgqr.cc {#device-guide-example-geqrf}

\include device/geqrf_orgqr.cc

### LDL^T factor and solve: sytrf_sytrs.cc {#device-guide-example-sytrf}

\include device/sytrf_sytrs.cc

### Singular values for any n: gesvd_any_n.cc {#device-guide-example-gesvd}

\include device/gesvd_any_n.cc

### CSR times dense: spmm_csr.cc {#device-guide-example-spmm}

\include device/spmm_csr.cc

## Not provided as device routines {#device-guide-not-provided}

- `gesvdj` (one-sided Jacobi SVD) at any tier; use `gesvd`.
- The `syevd` and `syevx` drivers: compose `sytrd`, `stedc`, `stebz` and `stein` in one kernel.
- The two-stage `syev` driver: `sy2sb` and `sb2st` are routines, the driver is host-only.
- Cta tiers of `potrs`, `posv` and `potri`.
- Lanczos (operator-based, host-only).

Why each is missing: [migration status](../design/device-la.md#device-la-migration-status).

## Known limitations {#device-guide-known-limitations}

- The tier-free routines do not check for 32-wide sub-groups ([requirements](#device-guide-device-requirements)).
- Some headers deviate from the conventions (the `gebrd` Cta tier takes the `nd_item`;
  `iluk_factorize` takes its options before `ws`; `stebz` returns `m` through an `int&`): the
  [open debts](../design/device-la.md#device-la-open-debts) list them all.
- Unscaled sums of squares in `ortho` and `ritz_values` lose accuracy when every entry is below
  about \f$2^{-63}\f$ on a device that flushes subnormals
  ([known defects](../design/known-defects.md#unverified-candidates)).

## See also {#device-guide-see-also}

- @ref api_device &ndash; the reference groups and the compatibility rule.
- @ref guide_cpp_api &ndash; the host API, header-only use, versioning.
- @ref design_device_la &ndash; why the API looks this way; measurements in @ref perf_device_la.
- @ref design_workspace, @ref design_error_model, @ref design_known_defects.
