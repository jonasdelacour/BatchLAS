# Device API migration measurements {#perf_device_la}

> **Status:** current · RTX 4090 (sm_89), CUDA backend · measured during the migration onto
> `batchlas::device`, before 1.0.0

The measurements behind the device API's design (@ref design_device_la): why the first in-kernel
layer was replaced, how the migrated kernels time against the hand-written code they replaced, and
the exceptions accepted on the way. The user guide is @ref guide_device_api.

## Why the v1 layer cost performance {#device-la-why-the-v1-layer-cost-performance}

Measured on an idle RTX 4090 (sm_89, batch 32768, n 32/64, work-group 64/256; lab `benchmarks/sass_lab`).
Four causes, each forced or hidden by the v1 API (`group_blas`, since deleted):

| Cause | Evidence (v1 / hand) |
| --- | --- |
| Generic L2/L3 algorithms: one work-group reduction and barrier per output element | gemv N 1.87-2.83x; trmv up to 5.0x (f64); symv up to 4.9x (f64); gemm 32^3 from SLM 7.3x f32, 130x f64 |
| Executor size read at run time (`get_local_range()`) | dot f32 75 vs 56 instructions; symv f32 23 -> 40 REG; latrd f32 736 -> 1032 instructions |
| Run-time path selection: every path is compiled, REG is the maximum over paths | gemv N 38-44 REG vs 19-28; gemm with workspace 128 REG, 9 BAR |
| Op-level barriers: an op promises no thread-to-data map, so callers add barriers | lab latrd 15 BAR vs 7; in-tree latrd panel 30 vs 19 BAR |

Ruled out, by SASS and time: view types (equal SASS at compile-time extents), out-of-line calls (0
library CALLs), address-space loss (0 generic LD/ST), spills (LMEM 0) and `reqd_work_group_size`.

The scaled `nrm2<alg::Blue>` costs 2.9-4.5x the instructions and a median 2.3x the time of
`nrm2<alg::Unscaled>`, against a 10% bar, which is why `nrm2` has no default algorithm
([the rules](../design/device-la.md#device-la-the-rules)).

## Migration timings {#device-la-migration-timings}

Lab parity: 84/84 device LA lab kernels have SASS identical to the hand code. Timing (RTX 4090,
device LA / legacy median, < 1 is faster; selected): trsm SgLeft 0.840 f / 0.931 cf, gesvdj 0.737,
larft 0.581, sb2st chase cf 0.486, sygst 0.850 cf / 0.901 f, syevx 0.924, tridiag 0.919, sytrf /
sytrs Cta 0.985-1.0025.

## Accepted exceptions {#device-la-accepted-exceptions}

Migrated kernels that miss a threshold of [the SASS-parity
gate](../design/device-la.md#device-la-the-sass-parity-gate), or give up bit identity with the code
they replaced, and ship by user decision.

| Accepted exception (user decision) | Measured |
| --- | --- |
| latrd lower panel: the device LA body is the only body (SLM floor cap and tuned occupancy) | waived, `BM_SYTRD_BLOCKED` device LA / legacy median: float n = 256, batch 1024: 1.217x (9 reps; no occupancy setting recovers it; decision 20); float n = 256, batch 128: 1.030x and complex<float> n = 1024, batch 128: 1.030x (15 reps; decision 22); the other shapes 0.83-1.015x, float / cfloat n >= 512, batch 1024: 0.83-0.89x |
| `GetrsFusedNKernel`: one device LA body for every T (the complex<float> hand body is deleted) | complex<float> 0.96-1.01x at nrhs <= 4, 1.02-1.05x at nrhs 6-8 |
| `PotrfCta<float, 8, 4, SubGroup>` | +3% instructions, 1.015x (decision 20) |
| syfn `ScaleKernel` | +7.6% instructions, A/B within 1.02x (decision 20) |
| `Sytd2CTAKernel<float,16>`, `SpmmGather<float,16,0>` | 1.021, 1.022 (decision 15) |
| `BdsdcExtract<float>`: REG 36 -> 40, blocks/SM unchanged | A/B 1.001 (float), worst median 1.0028 |
| bit-identity: gesvd Hermitian mirror Im(diag) = 0, real and complex axpby FMA contraction, Frobenius-norm reduction order, the algorithm ports (ormqr Right fused msub, the device LA larft, complex syevx / krylov / gebrd, gesvdj, tridiag solver, stedc `nrm2<Blue>`) | decisions 12 and 15 |
