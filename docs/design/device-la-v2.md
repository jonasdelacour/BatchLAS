# Device LA v2: the in-kernel BLAS/LAPACK layer {#design_device_la_v2}

> **Status:** current · checked 2026-11-04

`batchlas::v2` (`#include <batchlas/blas/device/v2/dla.hh>`) is the in-kernel linear-algebra layer the
LAPACK-like kernels are written in. Executor, layout and algorithm are compile-time types and the caller
owns every barrier, so a v2 call compiles to the SASS of the hand-written code it replaces. The older
runtime-shaped API is @ref design_device_group_blas.

## Device LA v2: why the v1 layer cost performance

Measured on an idle RTX 4090 (sm_89, batch 32768, n 32/64, work-group 64/256; lab `benchmarks/sass_lab`).
Four causes, each forced or hidden by the v1 API:

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
- **Barriers belong to the caller.** An operand that needs cross-lane visibility is `Published<V>`, made
  only by `ex.publish(v)` (a barrier) or `assume_published(v)`.
- **Scalars are tags:** `One`, `Zero`, `NegOne`, `Scalar<T>`. `beta = Zero` never reads the output.
  Argument order is BLAS's (`axpy(ex, alpha, x, y)`).
- **Arithmetic policies** `arith::{Explicit, ExplicitInline, Native, Fma}` via `arith_t<Ops,T>`. Strict
  (1:1) ports use `Native` for bit identity; algorithm ports use `Explicit`.
- **Inline policy: take the attribute of the replaced code.** `BATCHLAS_DLA_INLINE` (always_inline) for
  leaf arithmetic, accessors and expression text; `BATCHLAS_DLA_HELPER` (plain inline) for ops that
  replace a plain-inline helper (an always_inline call after a loop is spliced ahead of the loop body,
  and LLVM unrolls a loop differently once a call is inlined); `BATCHLAS_DLA_BODY` on lambdas passed to
  loop ops, which is also the sanctioned kernel-body invoker (`scripts/check_kernel_entry_calls.py`).
- **Reductions** expand packs; never `#pragma unroll` around a collective (stedc +40%, LDS 2-3x).
- **Floating-point shape is explicit where two targets must agree.** `-ffp-contract=on` leaves each
  `a*b+c` to the backend; spir64_x86 contracts masked and full SIMD paths differently. Where CPU and GPU
  results must match bit for bit, spell the `sycl::fma` the NVPTX build already forms
  (`steqr_v2::tridiag_rot`).
- **Placement:** an op goes in `include/` when two families use it or it is a named BLAS/LAPACK
  primitive; otherwise in `src/ops/<op>/<op>_v2.hh` or `src/kernels/common/dla_v2_<name>.hh`.

## Device LA v2: API map

| Header (`include/batchlas/blas/device/v2/`) | Provides |
| --- | --- |
| `dla.hh` | umbrella |
| `core.hh` | inline macros, extents, `Scope`, layout tags, `unroll::*`, `Range`, `alg::*`, `arith::*`, scalar tags, `Scratch` |
| `exec.hh` | `WorkGroup`, `Teams`, `SubGroup`, `Partition`, `Thread`, `RtGroup`, `sync` |
| `grid.hh` | `Grid2D`, `batched2d_shape`, `batched2d` host launcher |
| `operand.hh` | `Vec`, `Mat`, `Aligned`, `Frag`, `Tile`, `Published`, `load` / `store` / `fill` |
| `interop.hh` | `vec(VectorView)`, `mat(KernelMatrixView)`, `item(view, b)` |
| `level1.hh` | `for_each_owned[_2d]`, `axpy`, `scal`, `copy`, `swap`, `dot[c]`, `dotN<K>`, `dot_many`, `nrm2<Alg>`, `lassq`, `asum`, `iamax`, `argmax`, `find_first` |
| `level2.hh` | `gemv`, `symv` / `hemv`, `ger[u,c]`, `trsv` / `trmv` (Thread) |
| `lapack.hh` | `lacpy`, `laset`, `mirror`, `laswp`, `rot`, `lasr`, `larfg`, `larf` |
| `scalar.hh` | `Givens`, `lartg`, `lae2`, `laev2`, `las2`, `wilkinson_shift`, `larfg_scalars` |

Shared private ops in `src/kernels/common/`: `dla_v2_partition.hh` (`SgpPartition`), `dla_v2_scalar.hh`
(`Arith<Cx>`), `dla_v2_potf2.hh` (`potf2_lower`), `dla_v2_jacobi.hh` (`sym_schur2`), `dla_v2_larft.hh`,
`dla_v2_sparse.hh`. Tests: `tests/device_blas_v2_tests.cc` and one `device_blas_v2_<unit>_tests.cc` per
migrated family.

## Device LA v2: the SASS-parity gate

`scripts/sass_parity.sh --base <build> --cand <build> --lib <target> [--mode strict|algo]` pairs kernels
across two builds and compares per kernel REG, STACK, LMEM, static SMEM, instruction count and an 18-class
histogram from `cuobjdump`. Exit 0 pass, 1 fail, 2 "A/B" (a run-time A/B decides), 77 no NVPTX image.

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
rule only warns. `sass_parity_tests` runs the rule self-test, then checks each committed snapshot in
`tests/sass_snapshots/` (`PotrfTinyKernel`, `PosvTinyKernel`) against the build with the algo thresholds;
refresh a snapshot with `--snapshot-write` from the final build when a reviewed change moves it.

## Device LA v2: migration status

Lab parity: 84/84 v2 lab kernels have SASS identical to the hand code (REG +0, total +0.0%). 23 family
units are on v2. Timing (RTX 4090, v2/base median, < 1 is faster; selected): trsm SgLeft 0.840 f /
0.931 cf, gesvdj 0.737, larft 0.581, sb2st chase cf 0.486, sygst 0.850 cf / 0.901 f, syevx 0.924,
tridiag 0.919. Accepted by user decision: `Sytd2CTAKernel<float,16>` 1.021 (a 0.06 ms kernel, noisy) and
`SpmmGather<float,16,0>` 1.022; FP64-only deltas whose f32 analog is neutral (the FP64 proxy rule).

| Stays legacy | Why |
| --- | --- |
| `GetrsFusedNKernel<complex<float>>` | v2 body 1.062x slower (2293 vs 1302 instructions); `fused_launch_notrans` routes complex<float> to the pre-v2 body |
| latrd lower panel | strict port passes SASS but runs 1.022 f / 1.160 cf: legacy speed depends on its register-capped occupancy. Pending a decision (P3) |
| triangular tile kernels (l3-triangular-tiles) | on hold; the tuned source of truth |
| gemm register tiles, gemv_native, trsm diagonal solve, verification, iluk, sort, stedc laed4 | not migrated: tuned code or no in-kernel BLAS shape |

## Device LA v2: open debts

- latrd P3 occupancy decision (SLM floor, tuned occupancy or a gate rule).
- clang 24 drops `[[intel::max_work_group_size]]` (`BATCHLAS_LAUNCH_BOUNDS`, `src/sycl/kernel_attrs.hh`):
  port it to the `max_work_group_size` kernel property before moving off clang 22.
- `PotriCtaKernel<cfloat>` WG clamped to `kernel_max_wg_size` (352, not 384); the timing is unmeasured.
