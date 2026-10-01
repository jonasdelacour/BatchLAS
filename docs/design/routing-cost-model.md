# Per-architecture routing from a fitted cost model

Status: proposal (2026-10-01). Supersedes the `is_sm120_family()` windows held back from
PR #133. Nothing here is implemented yet.

## Why

PR #133 retuned BatchLAS for sm_120 by adding `if (is_sm120_family(cc)) {…} else {sm_89}`
forks. That works for two GPUs and does not scale to a third:

- **Architecture is a boolean.** Every unmeasured device (sm_80, sm_90, sm_100, AMD,
  Intel) silently gets the 4090 windows, presented as "the default".
- **One kind of fact lives in at least five places:**
  - `route_*.hh` predicates;
  - free functions with a silent `cuda_cc = 0` default (`syev_saturated_algorithm_for_n`);
  - the hand-written overlay `syev_arch_tuning.hh`, which exists because the generated
    `tuning_params.hh` has no arch dimension;
  - per-arch structs inside `.cc` files (`PotrfBlockedConstSm120`, `GetrsFusedRegs120`);
  - about 20 selectors inside launchers that never pass through `RouteTable`, so
    `route_diff` and coverage cannot see them (gemm `select_kernel_variant`,
    `trsm_left_use_sg`, `geqrf_cta_leg`, latrd grid min-n, nb pickers).
- **Windows are benchmark grids transcribed by hand into boolean expressions.** There are
  about 113 literal thresholds (30 distinct values) in the route headers alone, plus about
  15 in the gemm selector. Some encode single-cell quirks, for example the gesv cfloat Tiny
  ceiling of 3, trsm cfloat `n == 16` only, and the getrs cfloat "n 48..64, nrhs 16..24,
  batch 4096..8192" edge. They also encode the cuBLAS/cuSOLVER version, which is not part
  of any key.
- **Every op threads the arch key by hand.** 6 of 11 `*_op_shape` builders fill
  `OpShape::cuda_cc`. A missed site gives no error; it just routes as an sm_89.

## Decisions taken (2026-10-01)

1. Split #133. Fixes that help on every GPU land first; the sm_120 windows wait for this
   design.
2. Per-architecture knowledge is a **generated table**, one per architecture, produced from
   benchviz campaigns and committed as a header.
3. A device without a profile uses the **nearest measured architecture**. The substitution
   is visible in coverage and in `benchviz info`.
4. Routes are chosen by a **cost model**, not by window predicates.

## Design

### 1. The predicates split into capacity and cost

| | Meaning | Source | Changes per arch? |
|---|---|---|---|
| `supports()` | correctness: the route gives a right answer and fits | capacity facts: SLM budget, register rows, sub-group sizes, build facts | values do (register rows, launch ceilings), but measured as **launch ceilings**, never fitted |
| `preferred()` / `native_tier_preferred()` | speed | **deleted**, replaced by `argmin cost` | yes, through the profile |

Register rows such as `GetrsFusedRegs120` are capacity facts. They go in the profile's
`capacity` section and are produced by `scripts/register_probe.sh`, never by a fit.

### 2. Route choice

```
resolve(op, shape):
    forced via BATCHLAS_<OP>_ROUTE        -> as today (supports() still gates it)
    candidates = { r in order : supports(r, shape) }
    pick       = argmin_r  cost(profile, op, r, T, shape)
    native wins over vendor only if cost_native < (1 - margin) * cost_vendor
```

- The margin, about 5% and a per-profile constant, stops routes flapping near ties. It also
  replaces the hand-written "conservative" clauses.
- The vendor-free build uses the same argmin without the vendor, so `native_tier_preferred`
  disappears too.
- Kernel variants become sub-routes priced the same way. These are gemm `KernelVariant`,
  trsm V1 vs the sub-group Left kernel, the geqrf resident vs skinny leg, and the nb/W
  pickers. Coverage records them, and `route_diff` sees them.

### 3. Shape of the cost model

Analytic skeleton plus a few fitted constants per (profile, op, route, scalar):

```
groups  = launch geometry of the route at this shape    (from the launcher, not re-derived)
waves   = ceil(groups / (resident_groups_per_cu * compute_units))
t       = t_launch * n_launches
        + waves * max( work_flops / (groups_per_wave * F_route),
                       bytes      / (groups_per_wave * B_route) )
        + t_serial(n)                                     (serial recurrences, e.g. per-column)
```

- Fitted parameters: `t_launch`, `F_route`, `B_route` and one serial coefficient. That is
  about 4 per route per scalar, against dozens of literals per op today.
- The launcher exports launch geometry and per-group resident limits (one source of truth;
  AGENTS.md §8.7 forbids re-deriving them).
- Why not a pure surrogate fitted per cell, such as log-linear or a tree: it reproduces the
  grid it was fitted on, quirks included, and extrapolates badly to new n, batch or device.
  The analytic skeleton carries SM count, clock and occupancy, so nearest-arch fallbacks
  degrade gracefully.
- Cliffs the model cannot express are capacity facts and belong in `supports()`. Examples
  are the 48 KB SLM hole, register spills and the CTA n ≤ 32 limit.

### 4. Profiles

```
evaluation/routing/
  fit.py                  benchviz results.jsonl -> profile JSON + validation report
  profiles/sm_89.json     fitted, plus capacity section, plus provenance
  profiles/sm_120.json
include/batchlas/arch/
  profiles.hh             generated: constexpr tables, one per profile
  arch_key.hh             ArchKey{vendor, cc}, nearest-profile map
```

- **Provenance** for each profile: the campaign name, git sha, device name, driver,
  cuBLAS/cuSOLVER versions, and the compiler plus fp-model (the #133 fp-model lesson).
- **benchviz must record compute capability.** Today it stores only the nvidia-smi name.
- **Nearest-profile map**: explicit and short.
  - sm_80, sm_86, sm_89 and sm_90 map to `sm_89`.
  - sm_100 to sm_121 map to `sm_120`.
  - Anything else maps to `sm_89`, flagged `nearest=true`.
- `BATCHLAS_ROUTING_PROFILE=<name>` forces a profile, which gives an A/B test without a
  rebuild.
- `tuning_params.hh` gains the same arch dimension, generated from the same profiles.
  `syev_arch_tuning.hh` is deleted.

### 5. Validation is regret, not test colour

`fit.py` emits a report per profile:

- **Regret** for each measured cell: `time(route picked by the model) / time(best measured
  route)`. The report gives the geomean, p95 and max, plus the cells with regret over 1.1.
- **Comparison with the hand-written windows** on the same cells, so each op's migration
  shows whether it gained or lost.
- **Held-out cells**, fitted on one half of the grid and scored on the other, because a model
  scored on its own training cells proves nothing (AGENTS.md §8.7).

The CI gate checks that committed profiles and `profiles.hh` agree, and that held-out regret
stays under a threshold. Routing tests stop pinning literal edges. They pin regret, and they
straddle capacity limits in `supports()`.

## Migration

| Phase | Content | Behaviour change |
|---|---|---|
| 0 | PR #133 split: fixes that help every GPU (fp-model, device-call guard, `get_devices`, trsm ladder, getrs sub-group spread, LPanel loads, spmm pair-load) | yes, every GPU, kernel-level only |
| 1 | `ArchKey`, the profile loader, and `cuda_cc` filled centrally (inside `resolve_route`, not per op); coverage records profile and nearest; benchviz records cc | none (prove with `route_diff`) |
| 2 | Pilot op: cost model plus `fit.py` for **one** op on both profiles; compare regret against the hand windows | that op only |
| 3 | Roll out op by op; gemm `KernelVariant` and the nb pickers become sub-routes; delete `is_sm120_family` and every `preferred()` | per op |
| 4 | Land #133's sm_120 kernels (getrs Tiny, geqrf skinny, trsm sub-group Left, gemm small tiles) as routes the model prices | sm_120 gains |

## Decisions taken (2026-10-01, after #134 merged)

- **Pilot op: potrf.** All four native tiers (Tiny, CTA, LPanel, Blocked) and the vendor
  route exist on main.
- **sm_89 profile: fitted from archived evidence** (`perf-evidence/vendor-independence` and
  `benchmarks/results/`). The fit report has to state per dtype and route where that evidence
  is too thin.
- **fp64 is measured now** on sm_120, in the same forced-route sweep as float and cfloat.
- **GPU budget:** all four GPUs on threadripper02.
  - The sweep uses GPUs 1–3, and GPU 0 is kept for functional tests, so no test process runs
    next to a measured cell.
  - A follow-up pass can use GPU 0 once nothing else needs it.
- **Data:** existing campaigns time only the route the native walk picked, plus the vendor.
  A fit needs every supported route timed on the same cell, so benchviz gains a forced-route
  sweep mode with arms `route:<origin>:<algo>`.
