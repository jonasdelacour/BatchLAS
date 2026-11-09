# Device LA: BLAS and LAPACK inside your kernel {#design_device_la_v2}

> **Status:** current · checked 2026-11-06

`batchlas::device` (`inline namespace v2`) is the public in-kernel linear-algebra layer: BLAS levels 1-3
and the LAPACK routines BatchLAS ships, callable from a user's own SYCL kernel on one matrix per
executor (a work-group, a sub-group or a partition of one). The host routes launch the same bodies, so
a device call computes what the host op computes. Executor, layout, algorithm and tier are
compile-time types and the caller owns every barrier, so a call compiles to the SASS of the
hand-written code it replaced.

Every LAPACK-level op except Lanczos has a tier for its sizes inside one work-group: register and
local-memory tiers for small n, and a blocked tier over the in-group Level 3 and a caller-provided
global workspace above them. The ops whose blocked tier is still open are listed under the migration
status below.

## Device LA v2: quick start

```cpp
#include <batchlas/device/lapack/potrf.hh>   // core headers come with it
namespace dv = batchlas::device;
using Routine = dv::routine::potrf<batchlas::Uplo::Lower>;

// host: pick ONE tier for this n and device; each tier is its own kernel
dv::with_tier<Routine, float>(q.get_device(), n, [&]<class Tier>(Tier) {
    constexpr int wg = dv::tier_info<Routine, Tier, float>::wg_size;
    const dv::WorkspaceSize ws = dv::potrf_buffer_size<batchlas::Uplo::Lower, Tier, float>(n);
    // ... a local_accessor of ws.local_bytes, then in the kernel (Cta / Blocked tiers):
    //   const dv::WorkGroup<wg, 32> ex{it};
    //   int info = dv::potrf<batchlas::Uplo::Lower, Tier>(ex, dv::mat(a + b * stride, n, n, ld), w);
});
```

The complete program, with the register tier's `Partition<N>` launch and a residual check, is
`examples/device/potrf_any_n.cc` (ctest `device_example_potrf_any_n`). Kernels are compiled with
`-fsycl` against the installed headers; nothing links but the SYCL runtime.

## Device LA v2: why the v1 layer cost performance

Measured on an idle RTX 4090 (sm_89, batch 32768, n 32/64, work-group 64/256; lab `benchmarks/sass_lab`).
Four causes, each forced or hidden by the v1 API (`group_blas`, deleted in v2.2):

| Cause | Evidence (v1 / hand) |
| --- | --- |
| Generic L2/L3 algorithms: one work-group reduction and barrier per output element | gemv N 1.87-2.83x; trmv up to 5.0x (f64); symv up to 4.9x (f64); gemm 32^3 from SLM 7.3x f32, 130x f64 |
| Executor size read at run time (`get_local_range()`) | dot f32 75 vs 56 instructions; symv f32 23 -> 40 REG; latrd f32 736 -> 1032 instructions |
| Run-time path selection: every path is compiled, REG is the maximum over paths | gemv N 38-44 REG vs 19-28; gemm with workspace 128 REG, 9 BAR |
| Op-level barriers: an op promises no thread-to-data map, so callers add barriers | lab latrd 15 BAR vs 7; in-tree latrd panel 30 vs 19 BAR |

Ruled out, by SASS and time: view types (equal SASS at compile-time extents), out-of-line calls (0
library CALLs), address-space loss (0 generic LD/ST), spills (LMEM 0) and `reqd_work_group_size`.

## Device LA v2: the rules

- **Executor is a type.** `WorkGroup<WG,SG>` (the only one with `barrier()` / `publish()`),
  `Teams<WG,P,Order,SG>` (no barrier member: a barrier in a team region does not compile),
  `SubGroup<SG>`, `Partition<P,Masked,Impl>`, `Thread`, and `RtGroup<Scope,SG>` for a run-time width.
- **Layout is a type** (`Cyclic`, `RowFrag`, `ColPerLane`, `SgPerCol`, `Uniform`). Ops on the same
  executor and layout give an item the same elements, so they compose without a barrier.
- **Algorithm is a type.** A defaulted `Alg` parameter is chosen from (op, executor, trans); there is no
  run-time predicate on workspace, extents, alignment or sub-group size. v1's algorithm survives as the
  opt-in `alg::GroupReducePerOutput`. `nrm2<Alg>` has **no default**: `Blue` costs 2.9-4.5x the
  instructions and a median 2.3x the time of `Unscaled`, against a 10% bar, so every call names one.
- **Tier is a type** (below): a kernel instantiates one tier, so it never carries a second path's registers.
- **Barriers belong to the caller.** An operand that needs cross-lane visibility is `Published<V>`, made
  only by `ex.publish(v)` (a barrier) or `assume_published(v)`. LAPACK-level ops are the exception: they
  are collective, read inputs visible on entry and end with a barrier, so results are visible on return.
- **Scalars are tags:** `One`, `Zero`, `NegOne`, `Scalar<T>`, `MaybeZero`. `beta = Zero` never reads the
  output. Level 1/2 argument order is BLAS's (`axpy(ex, alpha, x, y)`); Level 3 and LAPACK ops take the
  host op's name and run-time argument order with the queue replaced by the executor, and the host's
  run-time modes (uplo, side, trans, itype, job) become template parameters.
- **Arithmetic policies** `arith::{Explicit, ExplicitInline, Native, Fma}` via `arith_t<Ops,T>`. Strict
  (1:1) ports use `Native` for bit identity; algorithm ports use `Explicit`.
- **Inline policy: take the attribute of the replaced code.** `BATCHLAS_DLA_INLINE` (always_inline) for
  leaf arithmetic, accessors and expression text; `BATCHLAS_DLA_HELPER` (plain inline) for ops that
  replace a plain-inline helper (an always_inline call after a loop is spliced ahead of the loop body,
  and LLVM unrolls a loop differently once a call is inlined); `BATCHLAS_DLA_BODY` on lambdas passed to
  loop ops, which is also the sanctioned kernel-body invoker (`scripts/check_kernel_entry_calls.py`).
- **Call ops from the kernel lambda or a named function**, never first through a functor `operator()`
  or a local lambda: DPC++'s SemaSYCL mis-handles a kernel body entered that way
  (`kernel_entry_calls_tests` enforces it for in-tree kernels).
- **Reductions** expand packs; never `#pragma unroll` around a collective (stedc +40%, LDS 2-3x).
- **Floating-point shape is explicit where two targets must agree.** `-ffp-contract=on` leaves each
  `a*b+c` to the backend; spir64_x86 contracts masked and full SIMD paths differently. Where CPU and GPU
  results must match bit for bit, spell the `sycl::fma` the NVPTX build already forms
  (`steqr_v2::tridiag_rot`).
- **Placement:** one implementation of everything. A BLAS, auxiliary, elementwise or epilogue helper is
  core (`include/batchlas/device/*.hh`) even with one user; only an op's own LAPACK routine is
  family-specific and lives in `include/batchlas/device/lapack/<routine>.hh`. Nothing under `src/`
  defines into `batchlas::device`; `device_placement_tests` (`scripts/check_device_placement.py` and
  its shrinking allowlist) enforces it.

## Device LA v2: executors

| Executor | Items | Use |
| --- | --- | --- |
| `WorkGroup<WG, SG>` | the whole work-group, `WG` a compile-time size | Cta and Blocked tiers, in-group Level 3; the only executor with `barrier()` and `publish()` |
| `Teams<WG, P, Order, SG>` | `WG / P` teams of `P` items | team-per-row Level 2 (gemv, symv); no barrier member |
| `SubGroup<SG>` | one sub-group | panel leaves (potf2, getf2, sytrf), shuffle reductions |
| `Partition<P, Masked, Impl>` | `P` lanes of a sub-group, `32 / P` matrices per sub-group | Reg tiers (lane r holds row r) |
| `SgpPartition<P>` (`sg_partition.hh`) | a partition over the backend's sub-group partition | sb2st chases |
| `Thread` | one item | serial steps (lagtf / lagts, trsv) |
| `RtGroup<Scope, SG>` | a work-group or sub-group of run-time width | bodies the host launches at several widths (sytrf, getrs) |

Every item of the executor calls an op with the same arguments; an item that returns early deadlocks
the op's next barrier or shuffle. `ex.leader()` is the item that writes a uniform result (info).
`interop.hh` turns BatchLAS views into operands: `vec(VectorView)`, `mat(KernelMatrixView)` and
`item(view, b)` (matrix `b` of a batched view).

## Device LA v2: tiers and with_tier

| Tier | Where the matrix lives | Bound |
| --- | --- | --- |
| `tier::Reg<N, Exact>` | registers of a `Partition<N>` or `SubGroup` | `n <= N <= 32` (complex<double>: 16) |
| `tier::Cta<N, Alg>` | local memory of one `WorkGroup` | `n <= N`, and the plan fits the device's local memory |
| `tier::Blocked<NB, Alg>` | global memory; an NB-wide panel or diagonal block in local memory, the rest in the caller's global workspace | any n |
| `tier::upto<NMax>` | the first tier of the routine's list admitting `NMax` (compile time) | |

Each op declares a routine tag `device::routine::<op><modes...>` (`potrf<UL>`, `ormqr<SD, TR>`,
`sygst<Itype, UL>`, `stedc<J>`, `sb2st<UL, KDMax>`, `spmm<TA, TB>`, ...). Per routine, tier and
scalar, `tier_info<Routine, Tier, T>` states `wg_size`, `sg_size`, `admits(n)` and
`buffer_size(n)`; `tiers_t<Routine, T>` lists the routine's tiers smallest first.

`with_tier<Routine, T>(device_or_limits, n, f)` runs on the host. It calls `f(Tier{})` for the first
tier that admits n, fits the device's local memory and maximum work-group size, runs at a sub-group
size the device has and, for a double `T`, finds fp64. Each listed tier is a separate instantiation
of the kernel `f` launches and exactly one runs, so no kernel pays for a tier it does not take (a
run-time `if (n <= 32)` inside one kernel compiles both paths, the v1 failure). When nothing admits,
it throws `batchlas::invalid_argument("<op>: no device tier admits n=<n>")`. `DeviceLimits` can be
built by hand to plan for another device (`device_core_tier_tests` uses a 32 KB, no-fp64 mock).

Device limits that filter tiers: AdaptiveCpp's Metal backend has no fp64 and 32 KB of local memory,
and ignores `BATCHLAS_REQD_SG_SIZE` (kernels run at the native sub-group size, 32 there); a CPU
device's local memory and sub-group sizes differ per runtime. `with_tier` reads them all.

## Device LA v2: workspace

| Type | Meaning |
| --- | --- |
| `WorkspaceSize` | `local_bytes` and `global_bytes` of ONE matrix, and their `align` (16) |
| `Workspace` | the matrix's slice: a decorated local pointer and a global pointer |
| `Plan` | a constexpr bump allocator over the two spaces (`carve_local`, `carve_global`) |
| `global_slice(base, size, b)` | matrix b's global slice of one buffer of `size.global_bytes * batch` |

The host sizes a launch with `<op>_buffer_size<modes..., Tier, T>(n...)`, which equals the plan the op
carves its pointers with, so size and use cannot drift apart (`device_api_*` tests assert it per tier).
Local bytes go into a 16-byte aligned `sycl::local_accessor` of the work-group; global bytes are one
device allocation of `global_bytes * batch`. Reg tiers need none. A Blocked tier's global slice holds
what its Level 3 cannot keep in local memory (ormqr's V, T and larfb's W1/W2; stedc's two n x n blocks).

## Device LA v2: info, errors and types

- LAPACK-level ops return LAPACK's `info`, the same value on every item: 0 success; `k > 0` the
  routine's own failure (potrf: leading minor k not positive definite; sytrf: D(k,k) exactly zero;
  stedc: a secular root did not converge), with LAPACK's partial-result contract; `-i` a bad argument
  i counted after the executor (-2 typically A not square or n above the tier's bound, -3 / -4 a
  missing or misaligned workspace). An op never throws in device code.
- Host-side errors throw `batchlas::` types whose message opens with the op's name (`with_tier`).
- Scalars: `float`, `double`, `std::complex<float>`, `std::complex<double>` and any POD `{re, im}`
  (`Cx<R>`) through `Arith<T>`; `std::complex` arithmetic goes through `Arith`, never Annex-G
  `operator*` (no `__mulsc3` in device code). Pivots are 1-based `int32`.

## Device LA v2: API map

Namespace `batchlas::device` (`inline namespace v2`: the mangled names of `batchlas::device::v2`, so a
kernel's symbols and SASS do not depend on the spelling). Include the umbrella `<batchlas/device.hh>`
(core, no LAPACK) or single headers. Doxygen group: @ref api_device.

| Header (`include/batchlas/device/`) | Provides |
| --- | --- |
| `../device.hh` | umbrella (every header below except `sg_partition.hh`, `lapack/`, `level3/`, `sparse.hh` and `detail/`) |
| `core.hh` | inline macros, extents, `Scope`, layout tags (`OnePerItem`), `unroll::*`, `Range`, `alg::*`, scalar tags, `MaybeZero`, `None` / `NoOp`, `static_for`, `static_extent_v`, `op::FMax` / `FMin`, `Scratch`, `Part`, `scope::*`, `loop::*`, `alpha_side::*`, `tri_diag::*` |
| `arith.hh` | `Arith<T>` (real, `std::complex`, any POD `{re, im}` such as `Cx<R>`), `arith::*` policies, `arith_t`, `arith::FmaSplit` (`ops_arith_t`), `abs1`, `abs2`, `abs_inf`, `conj`, `conj_if`, `force_real`, `lift`, `by_parts`, `scale_real`, `div_real`, `sub_scale_real`, `select`, `tag_mul`, `tag_mac`, `store<Shape, Ops>` (`epi::Sum` default, `MacChain`, `MulMac`, `Linear`, `LinearNegZero`) |
| `exec.hh` | `WorkGroup`, `Teams`, `SubGroup`, `Partition`, `Thread`, `RtGroup` (members `bcast`, `uniform`, `shuffle`, `shift`), `sync`, `sync<scope::*>`, index-only `Slot<N>` / `Slot<>`, `ElemAt<Exact>` / `ElemIn`, `Grid1D<Exact>`, `Fit<Ex>`, `bcast_policy::Shuffle` / `Slm<Mpw, Acc>` |
| `grid.hh` | `Grid2D`, `batched2d_shape`, `batched2d` host launcher |
| `operand.hh` | `Vec`, `Mat`, `mat`, `mat_wide`, `Aligned`, `Frag`, `Tile`, `Published`, `load` / `store` / `fill`, `load_row<Part, Herm>` / `store_row<Part>` (column window), `unit_head<refl::*, unit_fill::*>`, `LaneVec` / `lane_vec`, `frag_bcast`, `neighbor_exchange`, `StridedMat` / `mat_strided`, `mat_local` / `tile_local` |
| `interop.hh` | `vec(VectorView)`, `mat(KernelMatrixView)`, `item(view, b)` |
| `level1.hh` | `for_each_owned[_2d]`, `for_each_team_owned<loop::*>`, `for_each_owned_tiles<TR, TC>`, `axpy`, `scal`, `copy`, `swap`, `fill`, `map_many`, `interleave`, `reverse`, `upper_bound`, `rscl`, `dot[c]`, `dot_partial`, `dot_slab<N>`, `dotN<K>`, `dot_many`, `nrm2<Alg>`, `lassq`, `asum`, `iamax`, `argmax` (also over `Scratch`, a `Range` and a skip index), `find_first` (also `none`, `LaneVec`), `prod_scaled` / `ScaledProd` / `sqrt_scaled` |
| `level2.hh` | `gemv<TR, Alg, Ops, Shape, AlphaSide>` (`alg::TeamPerRow`, `alg::ColDot<W>`), `symv` / `hemv`, `ger[u,c]<Alg, Ops>` (`alg::Flat`), `ger_row`, `her2<Part, Alg, Ops>`, `trsv` / `trmv` (Thread), `trsv<UL, TR, tri_diag::*, Alg>` (`alg::LanePerRow`), `dgmm<Side>`, `gbmv<KL, KU>` |
| `level3.hh` | register tier: `gemm<TA, TB, Shape, Ops>`, `gemm_sub<Ops>`, `trsm<SD, UL, TR, DG, Ops>` on `Tile` operands |
| `level3/` | the in-group Level 3 (`group.hh`: `gemm`, `symm` / `hemm`, `syrk` / `herk`, `syr2k` / `her2k`, `trmm`, `trsm` on one matrix per work-group, any m, n, k) and the tuned gemm components it is built from: `cfg.hh` (`cfg::RegTiled`, `Smem`, `smem_elems`), `stage.hh`, `mma.hh`, `pipeline.hh`, `schedule.hh`, `epilogue.hh` (`LinearEpilogue`), `packet.hh` (`Vec4`, `Packet`), `copy_async.hh` (cp.async, target-guarded) |
| `lapack_aux.hh` | `lacpy<Part, OP, Alg, edge::*, diag_fill::*>`, `laset` (block offset, `Frag`, `Tile`), `mirror`, `laswp`, `laswp_index`, `laswp_gather`, `lapmr`, `IpivSink`, `rot` (`rot_conv::Reflector2`), `lasr`, `largv` / `lartv` / `lar2v`, `larfg` (also one element per item), `larf<Side, Ops, larf_form::*, loop::*>` (also on a `Frag` column), `lanst<NormType>`, `lange<NormType>`, `laqge`, `lascl<Part>`, `lacgv`, `trttp` / `tpttr` |
| `scalar.hh` | `Givens`, `lartg<W>` (`lartg_alg::Exact` / `Fast`), `lae2`, `laev2`, `las2`, `wilkinson_shift`, `eig2x2_nearest`, `larfg_scalars`, `lamch<C>`, `qnan`, `phase`, `sign_nonzero_real`, `SafeRecip` |
| `reduce.hh` | `shuffle_xor`, `shuffle_down`, `butterfly<Lo, Hi, Order>`, `shift_reduce<Count>`, `ArgMax`, `argmax_merge<Tie, NaN>`, `ArgMaxMerge`, `allreduce(ex, Scratch, v, op)`, `AtomicMinSlot`, `tie::*`, `nan::*` |
| `view.hh` | `row` / `col` / `window`, `flip<axis::*>`, `conj_view`, `conj_if`, `op_view<TR>`, `herm_view<UL>`, `herm_view<UL, storage::*, loads::*>`, `sym_view<UL>`, `PackedTri<UL, order::*>`, `packed`, `Packet<T, Bytes>`, `fn_vec`, `mapped`, `slice`, `gathered`, `shifted`, `refs` |
| `sparse.hh` | CSR `spmm<TA, TB>` (Blocked), `iluk_factorize` / `iluk_apply` (Blocked<1>) |
| `tier.hh` | `tier::Reg` / `Cta` / `Blocked` / `upto` / `list`, `tier_info`, `tiers_t`, `resolve_tier_t`, `DeviceLimits`, `with_tier` (host) |
| `workspace.hh` | `WorkspaceSize`, `Workspace`, `Plan` (constexpr bump carve), `carve_local`, `carve_global`, `global_slice` |
| `sg_partition.hh` | `SgpPartition`, `wrap`: `Partition` over `detail/sg_partition/` (`SubGroupPartition`) |
| `lapack/` | one header per LAPACK routine, the table below |
| `detail/` | `sg_partition/` (per-backend sub-group partitions), `kernel_attrs.hh` (`BATCHLAS_REQD_SG_SIZE`, `BATCHLAS_LAUNCH_BOUNDS`, `BATCHLAS_UNROLL_FULL`), `sscp_target.hh` (`sycl_impl::on_ptx()`) |

## Device LA v2: ops and tiers per family

Tier lists are `tiers_t<routine::<op>, T>`, smallest first. Each header's first comment states the
op's contract, executor per tier, info values and workspace.

| Family | Op (`lapack/<header>`) | Tiers | Tests |
| --- | --- | --- | --- |
| Cholesky | `potrf<UL, Tier>(ex, A, ws)` | `Reg<8>`, `Reg<16>`, `Reg<32>` (not complex<double>), `Cta<64>`, `Cta<128>`, `Blocked<32 / 16>` | `device_api_potrf`, `device_api_chol` |
| | `potrs`, `posv`, `potri` / `trtri` / `lauum` (`potri.hh`) | `Blocked<32 / 16>` | `device_api_chol` |
| LU | `getrf<Tier>(ex, A, ipiv, ws)` | `Reg<8>`, `Reg<16>`, `Reg<32>`, `Cta<64>`, `Cta<96>`, `Cta<128>`, `Blocked<32>` | `device_api_getrf`, `device_api_lu` |
| | `getrs`, `gesv`, `getri`, `inv` | `Blocked<32>` | `device_api_lu` |
| LDL^T | `sytrf<UL, Tier>(ex, A, ipiv, ws)`, `sytrs`, `sysv` | `Cta<32>` (sytrs also `Cta<64>`, `Cta<128>`), `Blocked<1>`; `Blocked<NB>` by name | `device_api_sytrf` |
| QR | `geqrf<Tier>(ex, A, tau, ws)`, `orgqr`, `ormqr<SD, TR, Tier>`; `larft`, `larfb` | `Blocked<NB, Alg>` (NB from `larfb_default_nb<T>`) | `device_api_qr` |
| Generalized eigen | `sygst<Itype, UL, Tier>(ex, A, B, ws)` | `Cta<8>`, `Cta<16>`, `Cta<32>`, `Blocked<32>`, `Blocked<16>` | `device_api_sygst` |
| Band / tridiagonal | `sy2sb<UL, Tier>(ex, A, AB, tau, kd, ws)` (dense to band) | `Blocked<32>`; `Blocked<64>` by name | `device_api_sy2sb` |
| | `sb2st<UL, Tier>(pex, ab, d, e, tau, kd, ws)` | `Cta<4096>`, `Blocked<32>` | `device_api_sb2st` |
| | `stebz`, `stein`, `stedc<J, Tier>` (real T) | `Blocked<1>` (one body for every n) | `device_api_stebz-stein`, `device_api_stedc` |
| | `laed2`, `laed3`, `laed4`, `lagtf`, `lagts`, `syevx` selection, `jacobi` | building blocks of the above and of the host routes | `device_blas_v2_*` |
| SVD | `gebrd<Tier>(ex, it, A, d, e, tauq, taup, ws)` (real T); `gesvd` steps | `Cta<4>` .. `Cta<32>` | `device_api_svd` |
| | `gebrd<Tier>(ex, A, d, e, tauq, taup, ws)` (every T, m >= n), `ormbr<Vect, SD, TR>`, `bdsqr`, `bdsdc` (real B), `gesvd<JU, JVT, Tier>(ex, A, S, U, Vh, ws)` (any m x n) | `Blocked<16>` (gebrd, gesvd), `Blocked<1>` (ormbr, bdsqr, bdsdc) | `device_api_svd` |
| Level 3 | `level3/group.hh` ops, `level3.hh` register tier | one matrix per work-group, `Cfg` = `l3::cfg::RegTiled` preset | `device_api_l3`, `device_core_level3` |
| Level 2, sparse | `gemv`; `spmm<TA, TB>`, `iluk_*` (`sparse.hh`) | Blocked | `device_api_gemv`, `device_api_sparse` |
| Utilities | `ortho<TR, Algo, Tier>` (CGS2, Cholesky QR), `ritz_values`, `norm` | `Cta<N>`, `Blocked<NB>` | `device_api_util` |

Lanczos is not a device op (operator-based, decision F-D5).

## Device LA v2: worked examples

One per family; `ex`, `it` and `w` are the executor, the `nd_item` and the `Workspace` of the quick
start, `b` the matrix index. The host picks `Tier` with `with_tier` or names one.

```cpp
// Cholesky, any n: examples/device/potrf_any_n.cc (Reg / Cta / Blocked through with_tier)
int info = dv::potrf<Uplo::Lower, Tier>(ex, dv::item(A, b), w);

// LDL^T solve, n <= 32 in one fused local tile (sysv `cta`'s bodies), else Blocked<1>
int info = dv::sysv<Uplo::Lower, dv::tier::Cta<32>>(ex, dv::item(A, b), dv::item(B, b), ipiv + b * n, w);

// QR then Q^H C, one matrix per WorkGroup<256, 32>; V, T, W1/W2 in this matrix's global slice
using QR = dv::tier::Blocked<32>;
const dv::Workspace w{local, dv::global_slice(gbuf, dv::ormqr_buffer_size<Side::Left, Transpose::ConjTrans, QR, T>(m, n), b)};
dv::geqrf<QR>(ex, dv::item(A, b), dv::vec(tau + b * k, k), w);
dv::ormqr<Side::Left, Transpose::ConjTrans, QR>(ex, dv::item(A, b), dv::item(C, b), dv::vec(tau + b * k, k), w);

// generalized eigenproblem: B = L L^T (potrf), then C = L^-1 A L^-T in place of A
dv::potrf<Uplo::Lower, dv::tier::Blocked<32>>(ex, dv::item(B, b), w);
dv::sygst<1, Uplo::Lower, dv::tier::Blocked<32>>(ex, dv::item(A, b), dv::item(B, b), w);

// tridiagonal eigenvectors by divide and conquer (real T, ascending w, Z(:, j) for w(j))
dv::stedc<JobType::EigenVectors, dv::tier::Blocked<1>>(ex, d, e, lambda, w, params, Z);

// bidiagonal reduction of a small real matrix in local memory (the host GebrdCTAKernel's body)
dv::gebrd<dv::tier::Cta<32>>(ex, it, dv::item(A, b), d, e, tauq, taup, w);

// in-group gemm, any m, n, k: the host reg route's tile body, bit-identical for float
using Cfg = dv::l3::cfg::RegTiled<float, 32, 32, 8, 2, 2>;            // 256 items
sycl::local_accessor<float> smem(dv::l3::smem_elems<Cfg>(), h);        // in the command group
dv::gemm<Transpose::NoTrans, Transpose::Trans, Cfg>(ex, dv::item(A, b), dv::item(B, b), dv::item(C, b),
                                                     dv::One{}, dv::Zero{}, dv::l3::smem_of<Cfg>(smem));
```

## Device LA v2: the SASS-parity gate

`scripts/sass_parity.sh --base <build> --cand <build> --lib <target> [--mode strict|algo]` pairs kernels
across two builds and compares per kernel REG, STACK, LMEM, static SMEM, instruction count and an 18-class
histogram from `cuobjdump`. Exit 0 pass, 1 fail, 2 "A/B" (a run-time A/B decides), 77 no NVPTX image.
`--identical` hashes each kernel's SASS text instead: every pair byte-identical (the proof a tuned
table's hash may be re-pinned without re-measuring, decision 18).

| Rule | strict (1:1 port) | algo (algorithm change) |
| --- | --- | --- |
| BAR SHFL LDS STS LDG STG | equal | BAR <= base |
| REG | <= base + 2 | <= base + max(2, 5%) |
| Total instructions | +-3% | +-10% |
| FMA | >= base | >= 0.9 x base |
| Blocks/SM | any change: A/B | decrease fails, increase A/B |
| Run time (median cand/base) | <= 1.02 | A/B required |

Both modes fail new LDL/STL/STACK/LMEM, more SMEM, a new PTX call into `batchlas::device` or
`__mulsc3`-style helpers, and more generic LD/ST. The total is one-sided: a shrink beyond the band passes
when every hard class is <= base, REG <= base + 2 and blocks/SM is unchanged. The launch WG comes from
PTX `.reqntid` / `.maxntid`, else `scripts/sass_parity_wg.txt`, else it is unknown and the occupancy
rule only warns. FP64-only deltas on the 1:64 RTX 4090 are judged by the float analog (neutral or
better) plus FP64 time <= 1.02x. `sass_parity_tests` runs the rule self-test, then checks each committed
snapshot in `tests/sass_snapshots/` (`PotrfTinyKernel`, `PosvTinyKernel`) against the build with the
algo thresholds; refresh a snapshot with `--snapshot-write` from the final build when a reviewed change
moves it (the snapshots are clang 22 specific).

## Device LA v2: migration status

v1 `group_blas` (`include/batchlas/blas/device/`, about 6600 lines), its tests and benchmarks are
deleted; every in-tree kernel is on `batchlas::device`. Lab parity: 84/84 v2 lab kernels have SASS
identical to the hand code. Timing (RTX 4090, v2/base median, < 1 is faster; selected): trsm SgLeft
0.840 f / 0.931 cf, gesvdj 0.737, larft 0.581, sb2st chase cf 0.486, sygst 0.850 cf / 0.901 f, syevx
0.924, tridiag 0.919, sytrf / sytrs Cta 0.985-1.0025.

| Accepted exception (user decision) | Measured |
| --- | --- |
| latrd lower panel: the v2 body is the only body (the SLM floor cap and tuned occupancy land with the syev-sym unit) | float n = 256, batch 1024: 1.19x (no occupancy setting recovers it; decision 20); float / cfloat n >= 512, batch >= 512: 0.82-0.97x |
| `GetrsFusedNKernel`: one v2 body for every T (the complex<float> hand body is deleted) | complex<float> 0.96-1.01x at nrhs <= 4, 1.02-1.05x at nrhs 6-8 |
| `PotrfCta<float, 8, 4, SubGroup>` | +3% instructions, 1.015x (decision 20) |
| syfn `ScaleKernel` | +7.6% instructions, A/B within 1.02x (decision 20) |
| `Sytd2CTAKernel<float,16>`, `SpmmGather<float,16,0>` | 1.021, 1.022 (decision 15) |
| `BdsdcExtract<float>`: REG 36 -> 40, blocks/SM unchanged | A/B 1.001 (float), worst median 1.0028 |
| bit-identity: gesvd Hermitian mirror Im(diag) = 0, real and complex axpby FMA contraction, Frobenius-norm reduction order, the algorithm ports (ormqr Right fused msub, larft v2, complex syevx / krylov / gebrd, gesvdj, tridiag solver, stedc `nrm2<Blue>`) | decisions 12 and 15 |

| Not yet a device op for every n | Status |
| --- | --- |
| gesvdj as a device op (any tier) | not started: the host gesvdj is a CTA kernel (n <= 64, SLM-resident); a `Blocked` one-sided Jacobi over global columns has no host route to back or check it, and `gesvd` `Blocked` already covers every n |
| syev / syevd / syevx drivers, sytrd, latrd as device ops, the latrd SLM floor cap | the syev-sym unit: latrd float n = 256 and complex<float> n = 1024 at batch 128 are 1.03x, above the 1.02x rule |
| the two-stage driver | sy2sb and sb2st are device ops; the driver stays host-launched |
| Cta tiers of potrs, posv, potri | the host `tiny` / `fused` / `cta` bodies stay host-launched |
| gemm register tiles, gemv_native, trsm diagonal solve, verification, sort | tuned host kernels on the device components, no separate device op |

## Device LA v2: open debts

- clang 24 drops `[[intel::max_work_group_size]]` (`BATCHLAS_LAUNCH_BOUNDS`, `include/batchlas/device/detail/kernel_attrs.hh`):
  port it to the `max_work_group_size` kernel property before moving off clang 22.
- `PotriCtaKernel<cfloat>` WG clamped to `kernel_max_wg_size` (352, not 384); the timing is unmeasured.
- The tuned tables' kernel hashes now cover the device headers (WP-R); the stale families are re-measured
  per device (`tuned/README.md`, `docs/design/tuning-data.md`).
