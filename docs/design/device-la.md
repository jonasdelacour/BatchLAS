# Device API: design record {#design_device_la}

> **Status:** current · checked 2026-11-22 against `include/batchlas/device/` for 1.0.0

Why `batchlas::device` looks the way it does: the design rules, the SASS-parity gate that holds a
migrated kernel to the hand-written code it replaced, where the migration stands, and where the
headers do not yet follow the conventions. Executor, layout, algorithm and tier are compile-time
types and the caller owns every barrier, so a call compiles to the SASS of the hand-written code;
the host routes launch the same bodies, so a device call computes what the host op computes. How
to use the API: @ref guide_device_api. The measurements behind these decisions:
@ref perf_device_la.

## The rules {#device-la-the-rules}

The design rules behind the [conventions](../guide/device-api.md#device-guide-conventions); each
keeps a call at the SASS of the hand-written code.

- **Executor is a type.** `WorkGroup<WG,SG>` (with `RtGroup<Scope::Group, SG>`, the only executors
  with a work-group barrier), `Teams<WG,P,Order,SG>` (no barrier member: a barrier in a team region
  does not compile), `SubGroup<SG>`, `Partition<P,Masked,Impl>`, `Thread`, and `RtGroup<Scope,SG>`
  for a run-time width.
- **Layout is a type** (`Cyclic`, `RowFrag`, `ColPerLane`, `SgPerCol`, `Uniform`). Ops on the same
  executor and layout give an item the same elements, so they compose without a barrier.
- **Algorithm is a type.** A defaulted `Alg` parameter is chosen from (op, executor, trans); there is no
  run-time predicate on workspace, extents, alignment or sub-group size. v1's algorithm survives as the
  opt-in `alg::GroupReducePerOutput`. `nrm2<Alg>` has **no default**: `Blue` costs several times the
  instructions and time of `Unscaled` ([measured](../perf/device-la.md#device-la-why-the-v1-layer-cost-performance)),
  so every call names one.
- **Tier is a type:** a kernel instantiates one tier, so it never carries a second path's registers.
  A run-time `if (n <= 32)` inside one kernel compiles both paths and the kernel carries the larger
  register count, which is how the v1 layer lost performance. `with_tier` therefore picks the tier
  on the host, and each tier is its own kernel.
- **Barriers belong to the caller.** An operand that needs cross-lane visibility is `Published<V>`, made
  only by `ex.publish(v)` (a barrier) or `assume_published(v)`. LAPACK-level ops are the exception: they
  are collective, read inputs visible on entry and end with a barrier, so results are visible on return.
- **Scalars are tags:** `One`, `Zero`, `NegOne`, `Scalar<T>`, `MaybeZero`. `beta = Zero` never reads the
  output. The host's run-time modes (uplo, side, trans, itype, job) are template parameters (C2).
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
  (`device::detail::tridiag_rot` in `lapack/steqr.hh`).
- **Placement:** one implementation of everything. A BLAS, auxiliary, elementwise or epilogue helper is
  core (`include/batchlas/device/*.hh`) even with one user; only an op's own LAPACK routine is
  family-specific and lives in `include/batchlas/device/lapack/<routine>.hh`. Nothing under `src/`
  defines into `batchlas::device`; `device_placement_tests` (`scripts/check_device_placement.py` and
  its shrinking allowlist) enforces it.
- **Public surface:** the compatibility promise covers the members of the `api_device` child
  groups except @ref api_device_lowlevel. Helpers that public routines need but users should not
  build on stay in `batchlas::device` (moving them is a code change that re-hashes the tuned
  tables) and are tagged with the low-level group.

## The SASS-parity gate {#device-la-the-sass-parity-gate}

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

## Migration status {#device-la-migration-status}

v1 `group_blas` (`include/batchlas/blas/device/`, about 6600 lines), its tests and benchmarks are
deleted; every in-tree kernel is on `batchlas::device`. The lab parity count, the timings against
the replaced code and the accepted exceptions are on @ref perf_device_la.

| Not yet a device op for every n | Status |
| --- | --- |
| gesvdj as a device op (any tier) | not started: the host gesvdj is a CTA kernel (n <= 64, SLM-resident); a `Blocked` one-sided Jacobi over global columns has no host route to back or check it, and `gesvd` `Blocked` already covers every n |
| syevd / syevx drivers | `sytrd`, `stedc`, `stebz` / `stein` compose in one kernel; no driver op |
| the two-stage driver | sy2sb and sb2st are device ops; the driver stays host-launched |
| Cta tiers of potrs, posv, potri | the host `tiny` / `fused` / `cta` bodies stay host-launched |
| gemm register tiles, gemv_native, trsm diagonal solve, verification, sort | tuned host kernels on the device components, no separate device op |
| Lanczos | operator-based; stays a host op |

## Open debts {#device-la-open-debts}

- clang 24 drops `[[intel::max_work_group_size]]` (`BATCHLAS_LAUNCH_BOUNDS`, `include/batchlas/device/detail/kernel_attrs.hh`):
  port it to the `max_work_group_size` kernel property before moving off clang 22.
- `PotriCtaKernel<cfloat>` WG clamped to `kernel_max_wg_size` (352, not 384); the timing is unmeasured.
- The tuned tables' kernel hashes cover the device headers, so a code change in a device header makes
  the families that hash it stale until they are re-measured per device (`tuned/README.md`,
  @ref design_tuning_data).
- The routines without a tier (`larft`, `larfb`, the in-group Level 3) and the Blocked tiers that
  reach `larft` or `sy2sb` need 32-wide sub-groups and check nothing at run time
  ([known defects](known-defects.md#unverified-candidates)).
- `ortho` and `ritz_values` have a `tier_info` but no `tiers<>` list, so `tiers_t`, `with_tier` and
  `tier::upto` do not compile for them; the caller names the tier.
- Where the headers do not yet follow the conventions (each change is a codegen gate, so it waits for one):
  C5: the `gebrd` Cta tier takes the `nd_item` beside `ex`. C4: `geqr2_reg` (`Real*`, `D*` scratch),
  `larfb_from_panel` (`T*` tau and buffers), `getf2_panel_device` (`int*` pivots and info), `laed2` /
  `laed3` (accessor callables, `std::int64_t` / `std::int32_t` sizes). C7: the in-group Level 3 has no
  routine tag, so `buffer_size<routine::gemm<TA, TB>, Cfg, T>()` does not exist; size its stage with
  `l3::smem_bytes<Cfg>()` and build the `Workspace` by hand (`gemm_group.cc`). C8: `BdsdcOptions` and
  `SteqrOptions` are not templated on the real type; `stebz` returns m through an `int&`;
  `iluk_factorize(ex, A, plan, M, opts, ws)` takes its options before `ws`. C11: plain
  `inline` on `geqr2` (by-value operands), `getf2_panel_device`, `sytf2` / `lasyf` / `lasyf_undo`,
  `sytrs_col` / `syconv` / `syconv_gather` and the `l3` packet, copy-async and `operand_value` helpers.
