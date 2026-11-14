# Device LA: BLAS and LAPACK inside your kernel {#design_device_la}

> **Status:** current · checked 2026-11-14

`batchlas::device` is the public in-kernel linear-algebra layer: BLAS levels 1-3
and the LAPACK routines BatchLAS ships, callable from a user's own SYCL kernel on one matrix per
executor (a work-group, a sub-group or a partition of one). The host routes launch the same bodies, so
a device call computes what the host op computes. Executor, layout, algorithm and tier are
compile-time types and the caller owns every barrier, so a call compiles to the SASS of the
hand-written code it replaced.

Every LAPACK-level op except Lanczos has a tier for its sizes inside one work-group: register and
local-memory tiers for small n, and a blocked tier over the in-group Level 3 and a caller-provided
global workspace above them. The ops whose blocked tier is still open are listed under the migration
status below.

## Device LA: quick start

```cpp
#include <batchlas/device/lapack.hh>   // every LAPACK-level op; batchlas/device.hh is the core
using namespace batchlas;
using R = device::routine::potrf<Uplo::Lower>;

// host: pick ONE tier for this n and device; each tier is its own kernel
device::with_tier<R, float>(q.get_device(), n, [&]<class Tier>(Tier) {
    using L = device::launch<R, Tier, float>;
    const auto sz = L::buffer_size(n);  // per matrix; the caller allocates gws = sz.global_bytes * batch
    q.submit([&](sycl::handler& h) {
        auto slm = device::local_workspace<L>(h, sz);  // per_group 16-byte aligned slices
        h.parallel_for(L::nd_range(batch), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(L::sg_size) {
            const auto ex = L::exec(it);         // tier_info::executor{it}
            const auto s = L::slot(it, batch);   // {b, live}: a packed tier's tail slots are dead
            const int info = device::potrf<Uplo::Lower, Tier>(ex, device::mat(a + s.b * stride, n, n, ld),
                                                              L::workspace(it, slm, gws, sz), s.live);
            if (s.live && ex.leader()) info_out[s.b] = info;
        });
    });
});
```

`launch<R, Tier, T>` (`launch.hh`) reads everything from `tier_info<R, Tier, T>`: `executor`,
`wg_size`, `sg_size`, `per_group` (matrices per work-group) and `buffer_size(extents...)`, which
`device::buffer_size<R, Tier, T>(extents...)` also returns. Its in-kernel index is
`matrix_slot(wg, per_group, team, batch)`, the one the library's packed launchers use. `with_tier`
counts `per_group` slices and, on acpp Metal, the 8 B per item the runtime appends to every launch
(`local_mem_bytes(d, wg)`). The complete program, with a residual check and a batch that leaves dead
slots, is `examples/device/potrf_any_n.cc` (ctest `device_example_potrf_any_n`). Kernels are
compiled with `-fsycl` against the installed headers; nothing links but the SYCL runtime.

## Device LA: why the v1 layer cost performance

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

## Device LA: the rules

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
  output. Argument order is BLAS's at every level: alpha before its product, beta before the output
  (`axpy(ex, alpha, x, y)`, `gemm<TA, TB, Cfg>(ex, alpha, A, B, beta, C, ws)`; out-of-place trmm is
  `trmm_out`). LAPACK ops take the netlib name with the queue replaced by the executor, and the host's
  run-time modes (uplo, side, trans, itype, job) become template parameters.
- **Arithmetic policies** `arith::{Explicit, ExplicitInline, Native, Fma}` via `arith_t<Ops,T>`. Strict
  (1:1) ports use `Native` for bit identity; algorithm ports use `Explicit`.
- **Inline policy: take the attribute of the replaced code.** `BATCHLAS_DEVICE_INLINE` (always_inline) for
  leaf arithmetic, accessors and expression text; `BATCHLAS_DEVICE_HELPER` (plain inline) for ops that
  replace a plain-inline helper (an always_inline call after a loop is spliced ahead of the loop body,
  and LLVM unrolls a loop differently once a call is inlined); `BATCHLAS_DEVICE_BODY` on lambdas passed to
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

## Device LA: executors

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

## Device LA: tiers and with_tier

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

## Device LA: workspace

| Type | Meaning |
| --- | --- |
| `WorkspaceSize` | `local_bytes` and `global_bytes` of ONE matrix, and their `align` (16) |
| `Workspace` | the matrix's slice: a decorated local pointer and a global pointer |
| `Plan` | a constexpr bump allocator over the two spaces (`carve_local`, `carve_global`) |
| `global_slice(base, size, b)` | matrix b's global slice of one buffer of `size.global_bytes * batch` |

The host sizes a launch with `device::buffer_size<routine::<op><modes...>, Tier, T>(extents...)` (the
same value as `launch<R, Tier, T>::buffer_size`; there is no per-op sizer), which equals the plan the op
carves its pointers with, so size and use cannot drift apart (`device_lapack_*` tests assert it per tier).
Local bytes go into a 16-byte aligned `sycl::local_accessor` of the work-group; global bytes are one
device allocation of `global_bytes * batch`. Reg tiers need none. A Blocked tier's global slice holds
what its Level 3 cannot keep in local memory (ormqr's V, T and larfb's W1/W2; stedc's two n x n blocks).

## Device LA: info, errors and types

- LAPACK-level ops return LAPACK's `info`, the same value on every item: 0 success; `k > 0` the
  routine's own failure (potrf: leading minor k not positive definite; sytrf: D(k,k) exactly zero;
  stedc: a secular root did not converge), with LAPACK's partial-result contract; `-i` a bad argument
  i counted after the executor (-2 typically A not square or n above the tier's bound, -3 / -4 a
  missing or misaligned workspace). An op never throws in device code.
- Host-side errors throw `batchlas::` types whose message opens with the op's name (`with_tier`).
- Scalars: `float`, `double`, `std::complex<float>`, `std::complex<double>` and any POD `{re, im}`
  (`Cx<R>`) through `Arith<T>`; `std::complex` arithmetic goes through `Arith`, never Annex-G
  `operator*` (no `__mulsc3` in device code). Pivots are 1-based `int32`.

## Device LA: API map

Namespace `batchlas::device` (a plain namespace; private helpers in `batchlas::device::detail`, tier tags
and `tier::is_reg_v` / `is_cta_v` / `is_blocked_v` in `batchlas::device::tier`).
Include the umbrella `<batchlas/device.hh>` (core, no LAPACK) or single headers. Doxygen group: @ref api_device.

| Header (`include/batchlas/device/`) | Provides |
| --- | --- |
| `../device.hh` | umbrella (every header below except `lapack/`, `level3/`, `sparse.hh` and `detail/`) |
| `core.hh` | inline macros, extents, `Scope`, layout tags (`OnePerItem`), `alg::Unroll*`, `Range`, `alg::*`, scalar tags, `MaybeZero`, `None` / `NoOp`, `static_for`, `static_extent_v`, `reduce_op::FMax` / `FMin`, `Scratch`, `Part`, `DiagFill`, `alg::Lockstep` / `Diverge`, `alg::Div` / `Recip` / `RealDiv` |
| `arith.hh` | `Arith<T>` (real, `std::complex`, any POD `{re, im}` such as `Cx<R>`; the complex kinds share `ComplexArith<T, Parts>`), `is_complex_v`, `dev_scalar_t`, `re`, `arith::*` policies, `arith_t`, `arith::FmaSplit` (`ops_arith_t`), `abs1`, `abs2`, `abs_inf`, `conj`, `conj_if`, `force_real`, `lift`, `by_parts`, `scale_real`, `div_real`, `sub_scale_real`, `select`, `tag_mul`, `tag_mac`, `store_epi<Shape, Ops>` (`epi::Sum` default, `MacChain`, `MulMac`, `Linear`, `LinearNegZero`) |
| `exec.hh` | `WorkGroup`, `Teams`, `SubGroup`, `Partition`, `Thread`, `RtGroup` (each `Ex{it}`; members `bcast`, `uniform`, `shuffle`, `shift`), `sync`, `sync<Scope>`, index-only `Slot<N>` / `Slot<>`, `ElemAt<Exact>` / `ElemIn`, `Grid1D<Exact>`, `Fit<Ex>`, `alg::BcastShuffle` / `BcastSlm<Mpw, Acc>` |
| `grid.hh` | `Grid2D`, `batched2d_shape`, `batched2d` host launcher |
| `operand.hh` | `Vec`, `Mat`, `mat`, `mat_wide`, `Aligned`, `Frag`, `Tile`, `Published`, `load` / `store` / `fill`, `load_row<Part, TR>` / `store_row<Part>` (column window), `unit_head<Refl, DiagFill>` (`tail(k)`), `LaneVec` / `lane_vec`, `frag_bcast`, `neighbor_exchange`, `StridedMat` / `mat_strided`, `mat_local` / `tile_local`, `for_each(Thread, Tile, i0, j0, f)` |
| `interop.hh` | `vec(VectorView)`, `mat(KernelMatrixView)`, `item(view, b)` (`item(v, b, n)`; `ItemMat` `col` / `row` / `sub`) |
| `level1.hh` | `for_each_owned[_2d]` (`alg::RowOwner`), `for_each_owned_2d_batched<U>`, `map2d`, `for_each_team_owned<alg::Lockstep | Diverge>`, `for_each_owned_tiles<TR, TC>`, `axpy`, `scal`, `copy`, `swap`, `fill`, `map_many`, `interleave`, `reverse`, `upper_bound`, `rscl`, `dot[c]`, `dot_partial<TR>`, `dot_slab<N>`, `dotN<K, TR>`, `dot_many`, `nrm2<Alg>`, `lassq`, `asum`, `iamax<NaN, Key>` (a `Range`, a tie, `ArgMaxSlots`, a skip index), `find_first` (also `none`, `LaneVec`), `prod_scaled` / `ScaledProd` / `sqrt_scaled` |
| `level2.hh` | `gemv<TR, Alg, Ops, Shape, Side>` (`alg::TeamPerRow`, `alg::ColDot<W>`), `symv` / `hemv`, `ger[u,c]<Alg, Ops>` (`alg::Flat`), `ger_row<TR>`, `her2<UL, Alg, Ops>`, `trmv<UL, TR, DG, Ops, Alg>` (Thread; `alg::Serial`, `alg::SplitDot<W>`), one `trsv<UL, TR, DG, Div, Alg, Ops, Upd>` (Div `alg::Div | Recip | RealDiv | DiagApply`, a diagonal operand or step object (`TpDiag`, `guarded_recip`); Alg `alg::Serial`, `alg::SerialRows<R>`, `alg::SplitDot<W>`, `alg::LanePerRow` (lane register, Cyclic `Frag`, RowFrag block), `alg::LanePerRowRolled`; Upd `alg::Fused | Unfused | UnfusedRight | DotThenSub | Axpy`), `dgmm<Side>`, `gbmv<KL, KU>`, `ger<Ops>` on register `Frag`s into a `Tile` |
| `level3.hh` | register tier: `gemm<TA, TB, Shape, Ops>`, `gemm_sub<K, Ops>`, `trsm<SD, UL, TR, DG, Ops>` into a `Tile` (`A`, `B` of `gemm_sub` and `A` of `trsm` may be any `a(i, j)` operand) |
| `level3/` | the in-group Level 3 (`group.hh`: `gemm`, `symm` / `hemm`, `syrk` / `herk` (`l3::rank_k`), `syr2k` / `her2k` (`l3::rank_2k`), `trmm`, `trmm_out`, `trsm` on one matrix per work-group, alpha before its product and beta before the output, any m, n, k) and the tuned gemm components it is built from: `cfg.hh` (`cfg::RegTiled`, `smem_elems`, `smem_bytes`), `stage.hh`, `mma.hh`, `pipeline.hh`, `schedule.hh`, `epilogue.hh` (`LinearEpilogue`), `packet.hh` (`Vec4`, `Packet`), `copy_async.hh` (cp.async, target-guarded) |
| `lapack_aux.hh` | `lacpy<Part, OP, Alg, alg::Edge*, DiagFill>` (`alg::RowMirror`: the Hermitian completion, one row per item; `live`), `laset` (block offset, `Frag`, Cyclic `Frag`, `Tile`), `mirror`, `laswp<Direct>`, `laswp_index`, `laswp_gather`, `lapmr<Direct, Alg, Base>`, `IpivSink`, `rot<Conv, Ops>` (`alg::RotReflector2`), `lasr<Pivot, Direct, Conv>`, `largv` / `lartv<Rot, Ops>` / `lar2v<Rot, Ops>`, `larfg` (also one element per item), `larf<Side, Ops, alg::ScaleW | ScaleV, alg::Lockstep | Diverge>` (also on a `Frag` column), `lanst<NormType>`, `lange<NormType>`, `laqge`, `lascl<Part>`, `lacgv`, `trttp` / `tpttr` (also a column range with run-time uplo and direction), `trimask` / `tri_expand` (`Tri`, `UploC` / `DiagC`), `geam`, the SLM tile moves `tile_load[_trans]` / `tile_store[_pair, _diag]`, trsm's `tp_pack_lower` / `tp_diag_recip` / `lacpy_tile_in` / `lacpy_tile_out` / `frag_copy_out` |
| `scalar.hh` | `Givens`, `lartg<W>` (`alg::LartgExact` / `LartgFast` / `Promote`), `lae2`, `laev2`, `las2`, `lapack_shift`, `eig2x2_nearest<alg::TieLowest | TieHighest>`, `detail::block_scale<alg::UnscaleRecip | UnscaleRatio>`, `larfg_scalars`, `lamch<C>`, `qnan`, `phase`, `sign_nonzero_real`, `SafeRecip` |
| `reduce.hh` | `shuffle_xor`, `shuffle_down`, `butterfly<Lo, Hi, Order>`, `shift_reduce<Count>` (a lane stride), `ArgMax`, `argmax_merge<Tie, NaN>`, `ArgMaxMerge`, `ArgMaxSlots`, `argmax_allreduce`, `allreduce(ex, v, op)` / `allreduce(ex, scr, v, op, trailing_barrier)` (a `T` or a `std::array`), `AtomicMinSlot`, `alg::Tie*`, `alg::Nan*` |
| `view.hh` | `row` / `col` / `window`, `flip<Axis>`, `conj_view`, `conj_if`, `op_view<TR>`, `herm_view<UL>`, `herm_view<UL, Storage, alg::OneLoad | TwoLoads>`, `herm_view<Load, Ops>(a, uplo)` (`alg::ThreeLoads`), `sym_view<UL>` (one `HermView`), `PackedTri<UL, PackOrder>`, `packed`, `TpDiag`, trsm's `CanonTri` / `CanonRhs` / `FlipRows` / `ColsFrom` / `MapCols`, `Packet<T, Bytes>`, `fn_vec`, `mapped`, `slice`, `gathered`, `shifted`, `refs` |
| `sparse.hh` | CSR `spmm<TA, TB, Tier>(ex, alpha, A, B, beta, C, ws)` (Blocked), `iluk_factorize<Tier>(ex, A, plan, M, opts, ws)` / `iluk_apply<Tier>(ex, M, rhs, out, ws)` (Blocked<1>) |
| `tier.hh` | `tier::Reg` / `Cta` / `Blocked` / `upto` / `list`, `tier_info` (`executor`, `per_group`), `tiers_t`, `resolve_tier_t`, `DeviceLimits`, `local_mem_bytes`, `with_tier` (host) |
| `launch.hh` | `launch<R, Tier, T>` (`buffer_size`, `nd_range`, `exec`, `slot`, `workspace`, `sg_size`), `local_workspace<L>`, `buffer_size<R, Tier, T>`, `MatrixSlot` / `matrix_slot` |
| `workspace.hh` | `WorkspaceSize`, `Workspace`, `Plan` (constexpr bump carve), `carve_local`, `carve_global`, `global_slice` |
| `sg_partition.hh` | `SgpPartition`, `make_partition<P>(sg)`, `wrap`: `Partition` over `detail/sg_partition/` (`SubGroupPartition`) |
| `lapack/` | one header per LAPACK routine, the table below; `lapack.hh` includes them all |
| `detail/` | `sg_partition/` (per-backend sub-group partitions), `kernel_attrs.hh` (`BATCHLAS_REQD_SG_SIZE`, `BATCHLAS_LAUNCH_BOUNDS`, `BATCHLAS_UNROLL_FULL`), `sscp_target.hh` (`sycl_impl::on_ptx()`) |

## Device LA: ops and tiers per family

Tier lists are `tiers_t<routine::<op>, T>`, smallest first. Each header's first comment states the
op's contract, executor per tier, info values and workspace.

| Family | Op (`lapack/<header>`) | Tiers | Tests |
| --- | --- | --- | --- |
| Cholesky | `potrf<UL, Tier>(ex, A, ws, live)` | `Reg<8>`, `Reg<16>`, `Reg<32>` (not complex<double>), `Cta<64>`, `Cta<128>`, `Blocked<32 / 16>` | `device_lapack_potrf`, `device_lapack_chol` |
| | `potrs`, `posv`, `potri` / `trtri` / `lauum` (`potri.hh`) | `Blocked<32 / 16>` | `device_lapack_chol` |
| LU | `getrf<Tier>(ex, A, ipiv, ws, live)` (`ipiv` a `Vec<int>`) | `Reg<8>`, `Reg<16>`, `Reg<32>`, `Cta<64>`, `Cta<96>`, `Cta<128>`, `Blocked<32>` | `device_lapack_getrf`, `device_lapack_lu` |
| | `getrs<TR, Tier>(ex, A, ipiv, B, ws, live)`, `gesv(ex, A, ipiv, B, ws, live)`, `getri(ex, A, ipiv, C, ws, live)`, `inv` | `Blocked<32>` | `device_lapack_lu` |
| LDL^T | `sytrf<UL, Tier>(ex, A, ipiv, ws)`, `sytrs` / `sysv(ex, A, ipiv, B, ws)` | `Cta<32>` (sytrs also `Cta<64>`, `Cta<128>`), `Blocked<1>`; `Blocked<NB>` by name | `device_lapack_sytrf` |
| QR | `geqrf<Tier>(ex, A, tau, ws)`, `orgqr`, `ormqr<SD, TR, Tier>(ex, A, tau, C, ws)`; `larft(ex, V, tau, T)`, `larfb` | `Blocked<NB, Alg>` (NB from `larfb_default_nb<T>`) | `device_lapack_qr` |
| Generalized eigen | `sygst<Itype, UL, Tier>(ex, A, B, ws)` | `Cta<8>`, `Cta<16>`, `Cta<32>`, `Blocked<32>`, `Blocked<16>` | `device_lapack_sygst` |
| Symmetric eigen | `syev<J, UL, Tier>(ex, A, W, ws, SyevOptions<R>{}, live)`; `sytrd<UL, Tier>(ex, A, d, e, tau, ws, live)`; `steqr<J, Tier>(ex, d, e, z, ws, SteqrOptions{}, live)` (real d, e) | `Cta<4>` .. `Cta<32>`, `Blocked<32>` (sytrd: Lower only; steqr: `Blocked<1>`) | `device_lapack_syev`, `device_lapack_sytrd`, `device_lapack_steqr` |
| Band / tridiagonal | `sy2sb<UL, Tier>(ex, A, AB, tau, kd, ws)` (dense to band) | `Blocked<32>`; `Blocked<64>` by name | `device_lapack_sy2sb` |
| | `sb2st<UL, Tier>(ex, ab, d, e, tau, kd, ws)` | `Cta<4096>`, `Blocked<32>` | `device_lapack_sb2st` |
| | `stebz(ex, d, e, w, m, ws, StebzOptions<R>{})`, `stein(ex, d, e, w, kb, z, ws, SteinOptions<R>{})`, `stedc<J, Tier>(ex, d, e, w, z, ws, StedcOptions<R>{})` (real T) | `Blocked<1>` (one body for every n) | `device_lapack_stebz-stein`, `device_lapack_stedc` |
| | `laed2`, `laed3`, `laed4` (returns info, the root through an out operand), `lagtf`, `lagts`, `syevx`, `jacobi` | building blocks of the above and of the host routes | `device_lapack_*` |
| SVD | `gebrd<Tier>(ex, it, A, d, e, tauq, taup, ws)` (real T); `gesvd` steps | `Cta<4>` .. `Cta<32>` | `device_lapack_svd` |
| | `gebrd<Tier>(ex, A, d, e, tauq, taup, ws)` (every T, m >= n), `ormbr<Vect, SD, TR>`, `bdsqr(ex, d, e, s, U, Vh, ws, BdsqrOptions<R>{})`, `bdsdc(ex, d, e, s, U, Vh, ws)` (real B), `gesvd<JU, JVT, Tier>(ex, A, S, U, Vh, ws)` (any m x n) | `Blocked<16>` (gebrd, gesvd), `Blocked<1>` (ormbr, bdsqr, bdsdc) | `device_lapack_svd` |
| Level 3 | `level3/group.hh` ops, `level3.hh` register tier | one matrix per work-group, `Cfg` = `l3::cfg::RegTiled` preset | `device_lapack_l3`, `device_core_level3` |
| Level 2, sparse | `gemv`; `spmm<TA, TB, Tier>`, `iluk_factorize` / `iluk_apply` (`sparse.hh`) | Blocked | `device_lapack_gemv`, `device_lapack_sparse` |
| Utilities | `ortho<TR, Tier, Cfg, Alg>` (CGS2, Cholesky QR), `ritz_values`; a matrix norm is `lange<NormType>` (`lapack_aux.hh`) | `Cta<N>`, `Blocked<NB>` | `device_lapack_util` |

Lanczos is not a device op (operator-based, decision F-D5).

## Device LA: worked examples

One per family; `ex`, `it` and `w` are the executor, the `nd_item` and the `Workspace` of the quick
start, `b` the matrix index. The host picks `Tier` with `with_tier` or names one.

```cpp
// Cholesky, any n: examples/device/potrf_any_n.cc (Reg / Cta / Blocked through with_tier)
int info = dv::potrf<Uplo::Lower, Tier>(ex, dv::item(A, b), w);

// LDL^T solve, n <= 32 in one fused local tile (sysv `cta`'s bodies), else Blocked<1>
int info = dv::sysv<Uplo::Lower, dv::tier::Cta<32>>(ex, dv::item(A, b), dv::vec(ipiv + b * n, n), dv::item(B, b), w);

// LU solve (ipiv a Vec<int>, 1-based): A = P L U, then op(A) X = B in place of B
dv::getrf<Tier>(ex, dv::item(A, b), dv::vec(ipiv + b * n, n), w);
dv::getrs<Transpose::NoTrans, Tier>(ex, dv::item(A, b), dv::vec(ipiv + b * n, n), dv::item(B, b), w);

// QR then Q^H C, one matrix per WorkGroup<256, 32>; V, T, W1/W2 in this matrix's global slice
using QR = dv::tier::Blocked<32>;
const dv::Workspace w{local, dv::global_slice(gbuf, dv::buffer_size<dv::routine::ormqr<Side::Left, Transpose::ConjTrans>, QR, T>(m, n), b)};
dv::geqrf<QR>(ex, dv::item(A, b), dv::vec(tau + b * k, k), w);
dv::ormqr<Side::Left, Transpose::ConjTrans, QR>(ex, dv::item(A, b), dv::vec(tau + b * k, k), dv::item(C, b), w);

// generalized eigenproblem: B = L L^T (potrf), then C = L^-1 A L^-T in place of A
dv::potrf<Uplo::Lower, dv::tier::Blocked<32>>(ex, dv::item(B, b), w);
dv::sygst<dv::Itype::AinvB, Uplo::Lower, dv::tier::Blocked<32>>(ex, dv::item(A, b), dv::item(B, b), w);

// tridiagonal eigenvectors by divide and conquer (real T, ascending w, Z(:, j) for w(j))
dv::stedc<JobType::EigenVectors, dv::tier::Blocked<1>>(ex, d, e, lambda, Z, w, dv::StedcOptions<float>{});

// dense symmetric eigenproblem (ascending W; A holds the eigenvectors); options after the workspace
dv::syev<JobType::EigenVectors, Uplo::Lower, Tier>(ex, dv::item(A, b), dv::vec(W + b * n, n), w, dv::SyevOptions<float>{});

// bidiagonal reduction of a small real matrix in local memory (the host GebrdCTAKernel's body);
// w is this partition's slice (launch<routine::gebrd, Tier, T>::workspace)
dv::gebrd<dv::tier::Cta<32>>(ex, it, dv::item(A, b), d, e, tauq, taup, w);

// in-group gemm, any m, n, k: the host reg route's tile body, bit-identical for float; the
// stage is the first l3::smem_bytes<Cfg>() of w.local (no global part)
using Cfg = dv::l3::cfg::RegTiled<float, 32, 32, 8, 2, 2>;            // 256 items
dv::gemm<Transpose::NoTrans, Transpose::Trans, Cfg>(ex, dv::One{}, dv::item(A, b), dv::item(B, b), dv::Zero{},
                                                     dv::item(C, b), w);

// CSR times dense, BLAS argument order: C = alpha A B + beta C (no workspace)
dv::spmm<Transpose::NoTrans, Transpose::NoTrans, Tier>(ex, alpha, dv::csr_item(A, b), dv::item(B, b), beta, dv::item(C, b));
```

## Device LA: the SASS-parity gate

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

## Device LA: migration status

v1 `group_blas` (`include/batchlas/blas/device/`, about 6600 lines), its tests and benchmarks are
deleted; every in-tree kernel is on `batchlas::device`. Lab parity: 84/84 v2 lab kernels have SASS
identical to the hand code. Timing (RTX 4090, v2/base median, < 1 is faster; selected): trsm SgLeft
0.840 f / 0.931 cf, gesvdj 0.737, larft 0.581, sb2st chase cf 0.486, sygst 0.850 cf / 0.901 f, syevx
0.924, tridiag 0.919, sytrf / sytrs Cta 0.985-1.0025.

| Accepted exception (user decision) | Measured |
| --- | --- |
| latrd lower panel: the v2 body is the only body (SLM floor cap and tuned occupancy) | waived, `BM_SYTRD_BLOCKED` v2/base median: float n = 256, batch 1024: 1.217x (9 reps; no occupancy setting recovers it; decision 20); float n = 256, batch 128: 1.030x and complex<float> n = 1024, batch 128: 1.030x (15 reps; decision 22); the other shapes 0.83-1.015x, float / cfloat n >= 512, batch 1024: 0.83-0.89x |
| `GetrsFusedNKernel`: one v2 body for every T (the complex<float> hand body is deleted) | complex<float> 0.96-1.01x at nrhs <= 4, 1.02-1.05x at nrhs 6-8 |
| `PotrfCta<float, 8, 4, SubGroup>` | +3% instructions, 1.015x (decision 20) |
| syfn `ScaleKernel` | +7.6% instructions, A/B within 1.02x (decision 20) |
| `Sytd2CTAKernel<float,16>`, `SpmmGather<float,16,0>` | 1.021, 1.022 (decision 15) |
| `BdsdcExtract<float>`: REG 36 -> 40, blocks/SM unchanged | A/B 1.001 (float), worst median 1.0028 |
| bit-identity: gesvd Hermitian mirror Im(diag) = 0, real and complex axpby FMA contraction, Frobenius-norm reduction order, the algorithm ports (ormqr Right fused msub, larft v2, complex syevx / krylov / gebrd, gesvdj, tridiag solver, stedc `nrm2<Blue>`) | decisions 12 and 15 |

| Not yet a device op for every n | Status |
| --- | --- |
| gesvdj as a device op (any tier) | not started: the host gesvdj is a CTA kernel (n <= 64, SLM-resident); a `Blocked` one-sided Jacobi over global columns has no host route to back or check it, and `gesvd` `Blocked` already covers every n |
| syevd / syevx drivers | `sytrd`, `stedc`, `stebz` / `stein` compose in one kernel; no driver op |
| the two-stage driver | sy2sb and sb2st are device ops; the driver stays host-launched |
| Cta tiers of potrs, posv, potri | the host `tiny` / `fused` / `cta` bodies stay host-launched |
| gemm register tiles, gemv_native, trsm diagonal solve, verification, sort | tuned host kernels on the device components, no separate device op |

## Device LA: open debts

- clang 24 drops `[[intel::max_work_group_size]]` (`BATCHLAS_LAUNCH_BOUNDS`, `include/batchlas/device/detail/kernel_attrs.hh`):
  port it to the `max_work_group_size` kernel property before moving off clang 22.
- `PotriCtaKernel<cfloat>` WG clamped to `kernel_max_wg_size` (352, not 384); the timing is unmeasured.
- The tuned tables' kernel hashes now cover the device headers (WP-R); the stale families are re-measured
  per device (`tuned/README.md`, `docs/design/tuning-data.md`).
