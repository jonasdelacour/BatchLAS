# Phase 3 execution plan: flat selection for posv, trsm and gemm

Status: plan, 2026-10-04. The maintainer's answers to §5's questions are in `flat-kernel-selection.md` §13, and they override this file where the two differ (Q1 yes; tuner pulled forward; blackwell kernels first; gemm Q8 yes; Q5, Q3 and Q7 not approved).

This is a read-only plan. Nothing was edited or built. Line numbers are worktree `flat-select` @ 886537e8. Where a map disagreed with the code, I checked the code: `select.cc:225-360`, `select.hh:290-340`, `route_trsm.hh:67-80`, `level3.cc:73-85`, `cublas.cc:111,133`, `gemm_variant.hh:310-319`, `small_batched.hh:23-26` and `gemm_kernels.cc:842-851`.

## 0. Corrections to the brief and the design doc

1. **§11's posv bullet is stale and should be struck in the posv PR.**
   - `flat-kernel-selection.md:643` says posv calls the potrf tier drivers directly.
   - In fact posv calls the public `potrf<Back,T>` at `factorization.cc:735,740` (main `:871,876`) and the public `trsm<Back,T>` at `:751-760`.
   - Its only direct driver calls are its own kernels: `posv_tiny_dispatch` (`:730`) and `potrs_fused_dispatch` (`:736`).
2. **`Table::nearest` drops every exact key at once** (`select.cc:335-344`): `filter` is all-or-nothing. That is harmless for potrf, which has one exact key. It is wrong for trsm (side, trans) and gemm (ta, tb, layout), where a missing combination could land on a row for the other side. This needs a helper change before trsm.
3. **`select::Device` has only `has_vendor_solver`** (`select.hh:45`), which defaults from `solver_vendor_available` (`:56-59`). trsm and gemm need the level-3 vendor flag (`level3_vendor_available<B>`).
4. **Table times must be numbers** (`select.cc:295-299`). Any "untimed" row therefore needs a format extension (§2).
5. **gemm's native-vs-vendor decision sits inside the vendor translation units.**
   - The vendor build goes straight to `gemm_vendor` (`level3.cc:85`).
   - `cublas.cc:133` then re-routes to `gemm_custom` through `gemm_use_sycl_custom` (`gemm_variant.hh:310-319`).
   - symm, syrk, syr2k and trmm call `gemm_vendor` directly (15 sites in `cublas.cc`).
   - R1 forces this decision out of `cublas.cc`/`rocblas.cc`. As a side effect those level-3 fallbacks stop reaching native gemm (risk K5).
6. **The doc's gemm example `tiled:tile=64:k=8` (§9) matches nothing compiled.** `Tiled16` is the only shared-memory tile. The register families are `RegTile{M,N,K,…}` (`register_launchers.hh:22-42`).

## 1. Order and PR boundaries

The stack, every PR based on the one before it:

```
worktree-flat-select (#138, phases 1-2)
 └─ P3.0  select: per-key exact drop, has_vendor_blas, untimed-row format, generic converter/transcriber
     └─ P3.1  posv
         └─ P3.2  tools/tune core (+ --gate mode)                  ← harness only, no op change
             └─ P3.3  trsm
                 └─ P3.4  gemm
```

- P3.0 and P3.2 are infrastructure, so P3.1, P3.3 and P3.4 each carry only one op's migration (the §3 rule "old code deleted in the migrating PR").
- posv does not need P3.2: `factor_bench --arms` already gives verified, interleaved timing for posv.
- If the maintainer chooses to land the blackwell kernels first (Q4), a **P3.2b** "port blackwell kernels, no routing" goes between P3.2 and P3.3.

### P3.0: changes to `src/select`

| Change | Where | Test |
|---|---|---|
| **Per-key exact drop.** Keep the rows that match the longest prefix of exact keys, in `# keys:` order, and drop exact keys from the right. potrf has one exact key, so its behaviour does not change. | `select.cc:335-344` | `select_tests`: a synthetic table with `side:exact trans:exact` and a missing (R,T) combination must stay on side=R |
| **`Device::has_vendor_blas`.** Fill it from `level3_vendor_available<B>`, add it to the `describe` memo key, and give `device_of` an overload. | `select.hh:38-59`, `select.cc:216` | `select_tests` |
| **Untimed entries.** `<spelling> -` parses as "rank given, no time". The header tag is `source=transcribed:<sha>`. The trace prints `transcribed` instead of `x ms`. `tuned_tables_tests:86-106` skips the time checks for `-` rows but still validates the spellings. | `select.cc:295-302`, `format_detail :420` | `select_tests`, `tuned_tables_tests` |
| **Generic converter.** Generalise `scripts/sweep_to_table.py`, which is potrf-only (op filters at `:105,:171`, file names at `:66,:72`), into per-op specs: key builder, arm→spelling map, CSV reader for `factor_bench`, JSONL reader for the tuner. Add a `--transcribe` mode (§2). | `scripts/` | `--check` on the potrf tables must be unchanged |

## 1.1 posv (P3.1)

Files: new `src/ops/posv/{choice.hh,posv.cc}`.

**Families.** All three are fieldless. Every knob is derived from the shape: Tiny's N/NC/NR buckets (`posv_tiny.cc:323-427`) and Cta's nb/wg (`getrs_fused.cc:1001-1005`).

```cpp
struct Tiny    : select::NoFields<"tiny"> {};     // posv_tiny_dispatch (fused factor + solve)
struct Cta     : select::NoFields<"cta"> {};      // public potrf + potrs_fused_dispatch
struct Blocked : select::NoFields<"blocked"> {};  // public potrf + 2x public trsm
candidates<T>() = {Tiny, Cta, Blocked}            // same for all 4 dtypes; no vendor family
aliases     {native:tiny→tiny, native:cta→cta, native:blocked→blocked}
last_resort {"blocked"}
key_names   {"uplo:exact", "n:log:3", "nrhs:log", "batch:log"}   // flops n³/3 + 2n²·nrhs
grid_n      = potrf grid_n ∩ [1,1024];  grid_nrhs {1,2,4,8,16,64};  grid_batch {128,512,2048,8192,32768}
```

- Keep the old names. Renaming to `fused`/`composed` would churn `factor_bench solve_outer_pin` (`:533-536`), `run_solve_grid.sh` and benchviz (`test_benchviz.py:42`).
- The class word `native` behaves as Auto. `vendor` warns and falls back to Auto (`select.hh:351,360`), which matches today (`route_resolve.hh:67-68`).

**`can_run`.** The common term is `native = d.is_gpu && d.has_sg32 && !A.het && !B.het && n>=1 && nrhs>=1 && batch>=1` (old `route_posv.hh:41-44`).

| Family | Clause | Driver check it mirrors |
|---|---|---|
| Tiny | `native && n <= posv_tiny_max_n<T>()` (16 for cdouble, else 32) `&& nrhs <= kPosvTinyMaxRhs` (4) `&& d.max_wg >= kPosvTinyWgSize` | `posv_tiny.cc:463-504`. The last term is **new**: today `supports()` (`route_posv.hh:47-55`) lacks the `:487-490` check. Add a sycl-free `kPosvTinyWgSize` in `solve_native.hh`, plus a `static_assert` against `kTinyWg` (`posv_tiny.cc:46`), following the `kPotrfTinyWgSize` precedent. |
| Cta | `native && nrhs <= kGetrsFusedMaxRhs` (8) `&& n*nrhs <= getrs_fused_max_rhs_elems<T>(d.slm_budget)` | `getrs_fused.cc:976-998`. `d.slm_budget` equals `LOCAL_MEM-4096`, the same budget `posv_route.hh:61-62` uses. |
| Blocked | `true` once `posv_validate_params` has passed | It has no checks of its own. A failing child surfaces as the child's own error (§3 "each op decides for itself"; option (a) in the posv map). **As built: `!A.het && !B.het`**, because vendor potrf/trsm do not fail on a heterogeneous batch, they solve at the full order (flat-kernel-selection.md §12, Phase 3.1). |

- Empty shapes (n=0 or nrhs=0) need a decision. Today the call reaches `solve_throw_unroutable`. Keep that, as an explicit early throw before `choose()`. Do not add a silent no-op.

**`launch` and `workspace`** (R1, R5). Move `factorization.cc:718-781` verbatim, including the kAdj logic at `:748-749`.

| Family | workspace |
|---|---|
| Tiny | `posv_tiny_buffer_size` |
| Cta | `potrf_buffer_size<B,T>` |
| Blocked | `potrf_buffer_size<B,T>` |

- `key_of = {uplo, n=A.rows, nrhs=B.cols, batch}`.
- The `TraceScope` shape is `square_shape(n,batch)` with `s.n=nrhs` and `s.uplo`. That matches `posv_route.hh:39-47`.
- `native_facts` gives `existed=1`.

**Sub-op calls:** public `potrf` (Cta, Blocked) and public `trsm` ×2 (Blocked). No parent override.

**Deleted:**
- `include/batchlas/blas/dispatch/route_posv.hh`, an installed header. The precedent is `route_potrf.hh` in phase 2.
- `src/backends/posv_route.hh` and its include at `factorization.cc:43`.
- posv's part of `SOLVE_ONE` (`:808-809`).
- Keep `Op::posv` in the enum until phase 5 (ABI).
- Stale comments to fix: `linalg-ops.hh:302`, `getrs_fused.cc:966`, `getrs_native.hh:80`, `factorization.cc:644-650`, `posv_tests.cc:260,437`.

**Tests**

| Old test (`tests/posv_tests.cc`, 14 TESTs) | Fate |
|---|---|
| include of `posv_route.hh` (`:20`) | delete |
| P6 `TinyRefusesShapesAboveItsCeilings` (`:415-431`) | fold into `CanRunEqualsLaunch` |
| P7 `AutoTakesTheMeasuredWindow` (`:441-480`) | delete |
| P7b `FusedSolveArmSolvesOnBothTriangles` (`:482-529`) | port to `ScopedPin<PosvChoice>(Cta{})` and drop the readback |
| P8 (`:535-560`) | port to `ScopedPin(Blocked{})` |
| P1-P5, P9 | keep |

New `tests/posv_candidates_tests.cc` (label `blas`), following §8:
- every candidate pinned, straddling n=cap/cap+1 (both caps: 16 for cdouble, 32 otherwise), nrhs 4/5 for Tiny, nrhs 8/9 for Cta, and a Cta **launch** at exactly `n*nrhs == max_rhs_elems`;
- both uplo values; non-natural ld and stride on A and B; complex data with an imaginary part;
- a batch of 1024 identical items with bit-identical results (Tiny uses SLM for float/cfloat at N≥16; Cta uses SLM);
- the other triangle poisoned, with potrf pinned native;
- the exact workspace inside a poisoned arena;
- `CanRunEqualsLaunch` against the two direct drivers;
- pin tests: unknown and can_run-false pins throw, aliases parse, `vendor` warns;
- `AutoReadsEveryKeyField`: hand-read rows straddling nrhs (cfloat n=28, nrhs 2 vs 4) and batch, on both devices.

## 1.2 trsm (P3.3)

Files: new `src/ops/trsm/{choice.hh,trsm.cc}`. The public entry moves out of `level3.cc:124-176`.

**Families.** Phase 3 keeps every knob derived, following the potrf precedent (§11 "Blocked has no fields").

```cpp
struct Cta     : select::NoFields<"cta"> {};      // trsm_native_v1_dispatch; bucket N∈{8,16,32} + wg ladder derived
struct Blocked : select::NoFields<"blocked"> {};  // trsm_native_blocked + public gemm; outer = env/side default
struct Vendor  : select::NoFields<"vendor"> {};   // trsm_vendor
candidates<T>() = {Cta, Blocked, Vendor}          // all 4 dtypes
aliases     {native:cta→cta, native:blocked→blocked}
last_resort {"blocked","vendor"}                  // potrf precedent; CPU: blocked can't run → vendor (= today)
key_names   {"side:exact","trans:exact","order:log:2","q:log","batch:log"}   // work ∝ order²·q·batch
```

- **sg-left family (from P3.2b).** P3.2b ports `trsm_native_sg_left_dispatch` (Side::Left, order ≤ 32) with no route, env pin or bench arm. Before the sm_120 and sm_89 trsm sweeps, P3.3 must add it to `candidates<T>()` as its own family with a pin spelling (R7), so the tuner can rank it (design doc §13).
- **trans key:** {N, T}. C folds to T because `do_conj` is the only difference (`trsm_native.cc:42,196`).
- **uplo and diag are not keys.** That is valid only after a one-time A/B shows they make ≤3% difference for cta, blocked and vendor on both GPUs, as part of the P3.3 sweep. If vendor is not invariant, add `uplo:exact` and accept 2× the rows.
- **Fields deferred to phase 4:** `cta:wg` {32,64,128,256} would delete ladder rules T4/T5 (`trsm_native.hh:40,43`). `blocked:outer` {32,64,128,256} would delete `BATCHLAS_TRSM_OUTER_NB` and T6 (`trsm_native.cc:366-383`). A static alias cannot express today's side-dependent outer default.

**`can_run`.** `order=A.rows`, `q = side==Left ? B.cols : B.rows`, `bs=A.batch`. The common term is `native = d.is_gpu && !A.het && !B.het && order>=1 && q>=1 && bs>=1`.

| Family | Clause | Driver check it mirrors |
|---|---|---|
| Cta | `native && order <= trsm_cta_max_n<T>()` (32) | `trsm_native.cc:345-353` throws above 32 |
| Blocked | `native && trsm_blocked_available<T>() && trsm_cta_max_n<T>() >= 1` | `route_trsm.hh:54-55`, `trsm_native.cc:550-553` |
| Vendor | `d.has_vendor_blas` | — |

- **Batch mismatch moves into `trsm_validate_params`** (`trsm.hh:39-96`) and throws `invalid_argument`. Today `trsm_op_shape` returns nullopt and the call silently goes to vendor (`trsm_route.hh:37`). This is a behaviour change (Q6).
- No `has_sg32` term: V1 and V2 use no sub-group ops, and the old `supports()` never had one.

**`launch`:** Cta → `trsm_native_v1_dispatch`. Blocked → `trsm_native_blocked(..., public gemm<B,T>)`, moved from `level3.cc:150-165`. Vendor → `trsm_vendor`. There is no workspace (R5: trsm has no `_buffer_size`). `TraceScope` sets m, n, k, side, uplo, transA and diag by hand. That also fixes the trsm coverage rows, which today carry `Backend::AUTO`.

**Deleted:**
- `route_trsm.hh` (96 lines) and `src/backends/trsm_route.hh` (82 lines).
- Old rules T1 (`route_trsm.hh:70` `batch<8`) and T2/T2b (`:72-78`).
- The unreachable default seams `trsm_native.cc:396-406` (→ `gemm_custom`) and `potrf_blocked.cc:244-253` (→ `trsm_native_blocked`). Every caller already injects the public op; making the seam parameter mandatory removes 2 routing bypasses.

**Tests**

| Old test (`tests/route_vocabulary_tests.cc`) | Fate |
|---|---|
| `trsm_shape` helper `:320-345`; tests `:346`, `:365`, `:386`, `:397`, `:415` | delete |
| `:422` | port to `trsm_candidates_tests`: vendor-free last resort is cta at order 32, blocked at 4096 |
| `:437` | port the same way: cta_max=0 means not runnable |
| `:445` | port the same way: CPU, heterogeneous, k=0, k=65 |
| `:474` | port the same way: q follows side |
| `:2266` (getri) | stays; fix its stale citations |

- `trsm_tests.cc`: keep all 38. The 8 facade tests (`:189-217`, batch ≤5) start exercising native kernels (risk K3).
- New `trsm_candidates_tests.cc`:
  - every candidate pinned at order 32/33 and across bucket edges 8/9 and 16/17;
  - both sides; all 7 internally issued (side, uplo, trans, diag) combinations;
  - ConjTrans with complex data and an imaginary alpha; non-natural ld and stride on A and B;
  - sub-views that carry the parent ld (the potrf panel pattern, `potrf_blocked.cc:300-310`);
  - batch ≥1024 identical items with bit-identical results;
  - `CanRunEqualsLaunch`; pin tests.
- The ctest re-run with `BATCHLAS_TRSM_ROUTE=native` (`tests/CMakeLists.txt:383`) stays.

## 1.3 gemm (P3.4)

Files:
- new `src/ops/gemm/{choice.hh,gemm.cc}`;
- the vendor TUs keep only `gemm_vendor_impl` (`cublas.cc:68-95`; rocblas likewise);
- `select_kernel_variant` (`gemm_kernels.cc:470-620`) and its launch switch (`:634-852`) become direct per-family launchers called from `gemm.cc`'s `std::visit`.

**Families.** Fields are only the knobs a selector chooses.

Derived inside the launcher, not fields:
- the transpose instantiation;
- the aligned vs predicated leg (`register_launchers.hh:61-76`, `gemm_kernels.cc:769,783`);
- the SmallBatched bucket (`small_batched.hh:28`);
- VA/VB = 4;
- TR, TC and Stages, from a constexpr per-config table in `choice.hh`.

| Family | Fields | float candidates | double | complex | Old enumerators folded (`gemm_kernels.hh:10-65`) |
|---|---|---|---|---|---|
| `direct` | — | 1 | 1 | 1 | 1 |
| `tiled` | — (tile 16 only) | 1 | 1 | 1 | 2 |
| `small` | — | 1 | 1 | — | 43 |
| `reg` | `m n k u` | 10: 32·32·8, 64·64·8, 64·64·16, 128·32·16, 128·32·32, 128·64·16, 32·128·16, 128·64·32 u4, 128·64·32 u2, 128·128·8 | — | — | 3-18, 21-23, 30, 31, 34, 40-42 |
| `wide` | `m n k` | 3: 64·64·16, 128·32·16, 32·128·16 | 3 | 3 | 35-39 |
| `vendor` | — | 1 | 1 | 1 | — |
| **Total** | | **17** | **7** | **6** | |

- **Dropped (10):**
  - #19, the dead alias (no spelling, `:224-225`);
  - #20, #24-26: S1U1, S2U2, TT8x4, TT4x8, which are pin-only and never chosen by Auto;
  - the 5 experimental variants #27-29, #32-33. Split-K also allocates per call (`split_k.hh:133`), which breaks R5.
  - `S2U1Aligned` and `S2U1Generic` (#22, #23) are legs, not choices. They become direct launcher unit tests.
  - cuBLASDx is left out: MathDx is absent, `available()` returns false (`gemm_cublasdx.cu:442-448`), and it cannot be tested under R7. Its TUs keep compiling.
- **Spellings:** `reg:m=128:n=32:k=32:u=1`, `wide:m=128:n=32:k=16`. `select::parse` has no field defaults (`select.hh:141-185`), so 4 fields is the practical maximum.
- **Aliases:** every legacy `BATCHLAS_GEMM_SYCL_KERNEL` name (1-4 per variant, `gemm_kernels.cc:183-292`) maps to its spelling. `register_tiled`, `sycl` and `custom` map to the class word `native`. The legacy `_VARIANT=native` (which means Vendor:Direct, `route_env.hh:109-113`) maps to `vendor` on `_VARIANT` only, until phase 5.
- **last_resort `{"vendor","direct"}`.** This deviates from potrf's native-first order, and the justification is generality (R4). Vendor is the only family that serves `precision != Default` (`route_gemm.hh:29`) and every CPU/NETLIB call. With direct first, an untuned CPU would run direct instead of netlib (Q3).
- **Keys:** `ta:exact tb:exact layout:exact m:log n:log k:log batch:log`.
  - For a real scalar, `key_of` folds C to T. Today potrf's panel produces real ConjTrans through trsm (`trsm_native.cc:456-457`) and misses the transposed tiles (`gemm_kernels.cc:505-511`).
  - `layout = packed` means A, B and C are contiguous with 16 B bases; anything else is `strided`. This is the data form of the leg-as-gate decisions at `:529`, `:533` and `:576`, which are deleted.
  - beta, alpha and the heterogeneous flag are not keys. Heterogeneous batches are handled before `choose()`, as at `level3.cc:61` today.

**`can_run`** (R3). This must reject what the launch rejects, or would silently fall back to `Tiled16`/`Direct` for. Today 18 NN-only register variants compute the **wrong answer** on transposed calls, because `launch_reg` takes no transpose (`register_tiled_common.hh:136-137`).

| Family | Clause |
|---|---|
| all native | `precision==Default && m,n,k>0 && !het` |
| direct, tiled | true |
| small | `!complex<T> && max(m,n,k) <= kSmallMaxDim` (64) `&& d.max_wg >= kSmallWg` (128) |
| reg | `is_same<T,float> && instantiated(cfg, ta', tb')` with ta', tb' after the C→T fold. 128·128·8 is NN only. 128·64·16 has no NN form. 32·128·16 has no NT form. The NN-only configs are NN only. |
| wide | `wide_trans_matches<T>(cfg, ta, tb)` (`register_wide_transposed.hh:88`), plus a measured launch ceiling for double/complex (AGENTS §8.9; the 128×128 comment at `gemm_kernels.cc:771-777` records a register abort) |
| vendor | `d.has_vendor_blas` |

**New reachability.** TN, NT and TT instantiations that are pin-only today become selectable through the table. They need R7 tests.

**Deleted or rewritten:**
- `route_gemm.hh` (83 lines);
- `gemm_variant.hh` routing (`:139-164`, `:252-319`);
- `gemm_use_sycl_custom` at `cublas.cc:133` and `rocblas.cc:50`;
- the `BATCHLAS_GEMM_SYCL_KERNEL` and `_EXPERIMENTAL` readers (`gemm_kernels.cc:29-54`, `:294-361`). An unknown name currently means Direct, which violates R6.
- Optionally `persistent.hh` (253 lines) and `split_k.hh` (169 lines).
- The 4 `gemm_custom` default seams (`potrf_blocked.cc:241`, `geqrf_blocked.cc:170`, `getrf_blocked.cc:198`, `trsm_native.cc:403`) become mandatory injections.

**Tests:**
- Delete `route_gemm_equivalence_tests.cc` (427 lines, 14 TESTs) and `route_vocabulary_tests.cc:131,144,203`.
- Rewrite the selector asserts in `gemm_tests.cc:220-344` and `:2333-2398`.
- Port the 15 `_SYCL_KERNEL` sites (`:127, 1402, 1510-1553, 1856-1986, 2032-2122`) to `ScopedPin<GemmChoice>`.
- Update `settings_tests.cc:243-256`.
- New `gemm_candidates_tests.cc`:
  - every candidate pinned on tile-multiple and non-multiple shapes, both layouts, every instantiated trans pair;
  - **a can_run-false pin on a transposed call must throw**: this is the guard for the 18 wrong-answer variants;
  - beta≠0, strided ld (parent-ld sub-views), complex data with an imaginary part;
  - batch ≥1024 identical items for the SLM tiles; `CanRunEqualsLaunch`.
- Port these benchmarks and scripts to `BATCHLAS_GEMM_ROUTE=<spelling>`: `gemm_128x{32x32,64x32}_family_benchmark.cc`, `gemm_heterogeneous_benchmark.cc`, `scripts/run_gemm_{steady_campaign,family_sweep,steady_profile}.sh`, `register_probe.sh`.

## 2. Table source per op and device

**What exists:** no posv, trsm or gemm data can be converted under §7.
- posv: sm_89 only, no `cta` arm, pins never confirmed (`resolved_route=skipped`), n ≤ 32.
- trsm: native-vs-vendor only, pre-barrier kernel era.
- gemm: one variant per shape (the Auto choice) against vendor only, never a per-variant grid.

### Options

| # | Option | Format extension? | Pros | Cons |
|---|---|---|---|---|
| A | Measure on sm_120 with existing harnesses | no | posv works today (`factor_bench --arms=tiny,cta,blocked`, verified, interleaved) | no trsm/gemm harness meets §6.1/§6.3; `factor_bench` lacks per-rep rotation (`:804-806`) |
| B | Pull the phase-4 tuner core forward (P3.2): in-process `ScopedPin`, rotation, host verify, two reversed passes, JSONL, tie rule, `--gate` mode | no | one harness for the sweeps, the §10.3 gate and phase 4; it is the only route for trsm and gemm | ~2-3 days of code before trsm |
| C | Borrow sm_120 on sm_89 (R8, loud) | no | zero work | inverts today's sm_89-tuned behaviour (5.5 picks the nearest above); the sm_89 gate probably fails; the weakest choice for gemm, where float verdicts are arch-sensitive |
| D | **Transcribe old routing**: evaluate today's resolver at every grid cell and write the old preference order as an untimed ranked list. For example posv inside the tiny window gives `tiny - \| cta - \| blocked -`. | **yes** (`-` times, `source=transcribed:<sha>`; P3.0) | sm_89 behaviour unchanged on the grid until a retune; generated on this box, because main's predicates read no arch (no `is_sm120_family` on main) and capacity terms stay in `can_run` at run time | departs from the §3 "ranked list with times" decision; off-grid, nearest-point lookup can differ from the old threshold (the gate times exactly those cells); trsm's old T1 (`batch<8`) is lost below grid batch 128 |
| E | Measure on sm_89 | no | real data | 1 GPU, shared box, display on device 0; posv ~15 h, trsm ~22 h, gemm ~60-80 h at full protocol |

**How to transcribe (D).** Use a host-only generator linked against the parent build's headers, run *before* the old code is deleted:
- posv and trsm: header-only `route_*.hh` predicates;
- gemm: `select_kernel_variant` exposed once, plus a variant→spelling map.

There are two cross-checks. On a small sample, a coverage readback (`route_diff.sh`) on the parent binary must agree. For gemm the check is an untimed `BATCHLAS_KERNEL_TRACE=1` readback, because coverage records only `native:register_tiled`.

### Recommendation

| Op | (a) sm_120 (this box) | (b) sm_89 |
|---|---|---|
| posv | **A**: a new driver around `factor_bench`, modelled on `routing/sm120_potrf_phase2_gate.py`, with per-arm coverage readback and two reversed passes, over the §1.1 grid. Lower on all dtypes; Upper on float at n 8-256 × batch 8192, as potrf did. | **D** (transcribed `tiny_window`, `route_posv.hh:97-104`). Optional: E on a reduced grid (n ≤ 64 × nrhs {1,2,4} × batch {8192, 32768}, ~3 h) when the 4090 is free. |
| trsm | **B**, then the sweep with the tuner | **D** |
| gemm | **B**, then the sweep, with a screening pass (Q5) | **D** |
| cpu (all) | No table. CPU never borrows (§12), so it goes to the last resort: posv `blocked`, trsm `vendor`, gemm `vendor`. That equals today. | — |

**Coupling.** posv's `cta` and `blocked` times include potrf's and trsm's choices. Re-sweep posv on sm_120 after P3.3 merges, or fold the children's table hashes into posv's §6.5 hash in phase 4.

## 3. Key space for trsm and gemm

### trsm

The per-dtype grid:
- order {1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024} (18)
- q {1,2,4,8,16,32,64,128,256,512,1024,4096} (12)
- batch {128, 512, 2048, 8192, 32768} (5)

That is 1,080 points. A 4 GiB cap leaves about 740-870 runnable points per dtype.

| Exact keys | Combinations | Rows per device (4 dtypes) | Approximate sweep, 4 GPUs |
|---|---|---|---|
| side × uplo × trans(N/T/C) × diag | 24 | 77k | ≥ 24 h (infeasible) |
| internally issued only | 7 | 22.6k | ~10 h |
| **side × trans(N/T)** (recommended) | 4 | 12.9k | **~5.5 h** (measure map: 22 h on 1 GPU / 5.5 h on 4) |

How to keep it tractable:
1. Fold C→T, and treat uplo and diag as non-keys after the invariance A/B.
2. Shrink batch to {128, 2048, 32768} for q > 256. Above that, cost is linear in batch.
3. Never time Cta above order 32: it is pruned by `can_run`, so cells with order > 32 have 2 candidates.
4. Put the phase-4 fields (wg, outer) on a sparse sub-grid only.

### gemm

A full lattice of m×n×k×batch×ta×tb×layout is 10⁴-10⁵ cells with up to 17 candidates, which is infeasible. Use a declared, demand-driven grid per (ta, tb, layout) instead:

| Grid part | Points |
|---|---|
| Squares | m=n=k ∈ {8,16,24,32,48,64,96,128,192,256,384,512,768,1024} |
| Panels (AGENTS §11: k = 8-136, about 60% transposed) | m,n ∈ {32,64,128,256,512,1024,2048} × k ∈ {8,16,32,64,96,128} for the forms actually issued: NN, NT/NC (potrf `:357,:371`, trsm Right), TN/CN (geqrf `:297-305`, trsm Left transposed) |
| Skinny | m×32×k and 32×n×k for the trsm Right and potrf W=32 updates |
| Batch | {128, 2048, 32768} |

Rules:
- ta/tb: real types use 4 combinations (C folded); complex types use only the issued ones (NN, NC, CN, CT plus NT/TN), not all 9.
- layout: `strided` everywhere; `packed` on the squares and the panels with m,n ≥ 128. Below that, can_run and the legs make layout irrelevant.
- Note on the per-key drop rule: a `packed` call at small m sees `packed` rows, because they exist. So the packed grid must include small squares too. At 14 points that is cheap.
- Prune per cell with `can_run`: about 6.7 float candidates per cell on average, 3.3 double, 4.0 complex.
- Estimate: ~2.7k cells per float device gives 59.5 h on 1 GPU and 15 h on 4 at the full protocol (measure map). A screening pass (1 pass, 0.5 s warm-up, keep candidates within 1.5× of the best) followed by the full §6.3 protocol on the survivors cuts that to **≈6-8 h on 4 GPUs**. This departs from §6.3 (Q5).

## 4. Gate plan (§10) per op

**Common method.** "Old Auto" is pinnable in every op, because the launch bodies move verbatim (check this with `git diff -M` on the moved blocks).

1. Run the **parent build** (the stack parent; see Q2) with `BATCHLAS_COVERAGE_OUT` (gemm: `BATCHLAS_KERNEL_TRACE=1`, untimed) and record the old choice at each gate cell.
2. Map it to a spelling. For gemm, use the alias table from P3.4.
3. In the **branch binary**, time arm A = `pin:<old spelling>` against arm B = `auto`, interleaved in one process, two passes with the order reversed.
4. FAIL if B/A > 1.05 and the loss reproduces.
5. Skip cells where both pick the same choice: they are identical by construction. List them in the evidence CSV.

Both arms run the same children, so only the op's own selection differs. Run one measuring process per box (`gpu_guard.sh` + `flock`); on the 4090 use device 1.

| Op | 1. Correctness: ctest (vendor and vendor-free builds; diff names against `tests/known-failures.txt`) | 2. Pure data | 3. Live off-grid cells, saturation | Harness |
|---|---|---|---|---|
| posv | `^posv_tests$\|^posv_candidates_tests$\|^tuned_tables_tests$\|^select_tests$\|^potrf_candidates_tests$` | converter `--check` | n {10,13,18,22,26,30,36,44,72,104,144,208} × nrhs {1,2,3,5,8} × batch {512, 8192, 32768} × 4 dtypes, Lower. Watch cfloat n=26-30 at nrhs 1-3, and nrhs=5 (today CTA). | `factor_bench --arms=<old>,native` (`native` = the Auto walk) |
| trsm | plus `trsm_tests`, `trsm_tests` re-run with `ROUTE=native`, `trsm_candidates_tests`, and the callers `potrf_tests`, `getrs`/`getri`/`gesv`, `ortho` | `--check` | order 36→44, 64→72, 96→104, 128→144 (plus ≤32: 10, 20, 28) × q {24, 48, 96, 768} × batch {512, 8192, 32768} × (side, trans) × dtype | tuner `--gate` (P3.2) |
| gemm | plus `gemm_tests`, `gemm_candidates_tests`, `device_calls_tests`, and every gemm consumer label (`ctest -L blas`, `ctest -L ortho`, `ctest -L eig`; this is shared code, so run the full ctest before pushing) | `--check` | off-grid squares {40, 80, 160, 320, 640}; panels off-grid in k {12, 24, 48, 112} and m {96, 384}; both layouts; the issued trans forms; batch {512, 8192, 32768}. Always beta=1 with strided ld (AGENTS §10). | tuner `--gate` |

**sm_89 with transcribed tables.**
- On-grid cells are identical by construction.
- Run a pure-data diff: old predicate at each off-grid point against the nearest transcribed row. Time **only** the cells that differ on the 4090. That should be tens of cells, about 1-2 h per op.
- Run it while the box is free; the user's potrf sm_89 gate is using it now.

**What reviewers must see in the trsm gate output:**
- batch < 8 now native instead of vendor;
- potrf's panel (R/L/C/N, order 64-128, float, batch < 128) moving away from vendor;
- debt 17 (composed posv 5.2× behind vendor at n=17) showing up as `vendor` ranked first for small order and small q, or not.

## 5. Risks and maintainer questions

**Risks**

| # | Risk | Mitigation |
|---|---|---|
| K1 | Without a table, posv falls to the last resort `blocked`, 2-20× slower than tiny at n ≤ 32 | Tables are a merge blocker in P3.1, and `AutoReadsEveryKeyField` catches a missing table |
| K2 | `worktree-blackwell-tuning` (64 commits, 51 src/include files) edits `select_kernel_variant` and the trsm router, both of which phase 3 deletes | Q4 |
| K3 | trsm tests at batch ≤5 move from vendor to native (old T1 is lost) and may surface latent native failures | run the caller suites in the P3.3 gate; compare names with the baseline |
| K4 | Complex trsm "vendor" on CUDA is BatchLAS's own substitution kernel (`cublas.cc:860-963`), with a plausible `int` overflow at `:880-881` (cfloat order 512 × batch 8192 = 2³¹) | host verify in the tuner catches it. Cap the cells or fix it as a separate one-line `int64_t` PR first. **PLAUSIBLE, not run.** |
| K5 | symm, syrk, syr2k and trmm call `gemm_vendor` directly (`cublas.cc`, 15 sites) and lose the native re-route at `:133` | make those calls go through the public `gemm<B,T>` in P3.4, or accept pure vendor for them (Q7) |
| K6 | The C→T fold for real gemm and trsm changes vendor-free routing for potrf/trsm children: probably faster, but a routing change | it is called out in the gate's routing diff |
| K7 | The sm_120 multi-GPU sweep breaks AGENTS §10's one-process-per-box rule | the potrf precedent used 3 GPUs; re-measure 5% of cells single-GPU and compare |
| K8 | `factor_bench` / `run_solve_grid` pins outside `can_run` now throw (cdouble tiny at n=17-32 in the default `ORDERS`, `run_solve_grid.sh:54`) | the posv driver skips cells where can_run is false (rows marked `bad=1 pin refused`) |
| K9 | Phase 4: `potrf_nb_env` latches the env in a static (`potrf_blocked.cc:56`) | out of scope; record it in §11 |

**Questions for the maintainer** (each with a recommended answer)

1. **Q1: Is it acceptable for sm_89 trsm/gemm/posv tables to be transcribed old routing (untimed `-` entries, `source=transcribed`) until a phase-4 retune on the 4090?** This departs from §3 "ranked list with times". *Recommended: yes.* It keeps sm_89 behaviour unchanged, needs no 4090 time beyond the off-grid diff gate, and every alternative either inverts sm_89 tuning (borrowing) or costs 60-100 h on a shared box.
2. **Q2: Should each PR gate against its stack parent rather than `main`?** *Recommended: yes, per op.* This isolates one op's selection change; for posv against main, potrf would also differ. Run one final combined gate against `main` before the stack merges.
3. **Q3: May gemm's `last_resort` put `vendor` before `direct`?** This deviates from potrf's native-first order. *Recommended: yes.* Vendor is the only family that serves `precision != Default` and is the CPU/NETLIB path, so it is the more general one (R4). trsm and posv keep native-first.
4. **Q4: Should the blackwell kernels (`trsm_sg_left.cc`, 2 extra wide gemm configs) land before or after the trsm/gemm PRs?** *Recommended: before, as P3.2b, kernels only, with every `is_sm120_family`/`cuda_cc` predicate dropped.* One sm_120 sweep then ranks them, instead of a second sweep and a hand port into deleted code. If that is not wanted, migrate main's set and add them later as candidates plus a re-sweep.
5. **Q5: Is a screening pass acceptable for gemm (1 pass, short warm-up, keep candidates within 1.5× of the best), followed by the full §6.3 protocol on the survivors?** *Recommended: yes, for gemm only.* It cuts the sweep from about 15 h to 6-8 h on 4 GPUs. Every shipped time still comes from the full protocol.
6. **Q6: Should a trsm batch mismatch (A.batch ≠ B.batch) throw `invalid_argument` in `trsm_validate_params`?** Today it silently goes to vendor. *Recommended: yes.*
7. **Q7: In P3.4, should symm, syrk, syr2k and trmm switch from `gemm_vendor` to the public `gemm`?** *Recommended: yes.* Otherwise they silently lose native gemm on the shapes where `:133` re-routes today, and that loss is invisible to the gemm gate.
8. **Q8: Delete the 5 experimental gemm variants and the 4 pin-only register variants (and optionally `persistent.hh`, `split_k.hh`)?** *Recommended: delete them.* They were never chosen by Auto, and split-K breaks R5. Promote one later as a field if a measurement justifies it.
9. **Q9: Should trsm's uplo and diag stay out of the key, given a ≤3% invariance A/B?** *Recommended: yes.* If the vendor fails the A/B, add `uplo:exact` (2× the rows).
10. **Q10: Keep the posv names `cta` and `blocked` (misnomers) rather than renaming to `fused` and `composed` with aliases?** *Recommended: keep them.* No churn in benchviz or the scripts.

## 6. Effort and wall time

Compute time is on threadripper02 with 3-4 GPUs. 4090 time is device 1, solo.

| PR | Code and tests | Sweep | Gate | Elapsed |
|---|---|---|---|---|
| P3.0 select, converter, transcriber | 1.5 d | — | select/tuned tests | 2 d |
| P3.1 posv | 1-1.5 d | sm_120 ~5 h on 3 GPUs (1654 cells, ~2.3 arms); sm_89 transcribe: minutes | sm_120 ~2 h; sm_89 diff cells ~1 h | 3-4 d |
| P3.2 tuner core and `--gate` | 2-3 d | — | self-test: reproduce one potrf row within 3% | 3 d |
| (P3.2b blackwell kernel port, if Q4 says before) | 1-2 d | — | kernel unit tests | 2 d |
| P3.3 trsm | 2 d | sm_120 ~5.5 h on 4 GPUs, plus the uplo/diag A/B ~1 h | sm_120 ~3 h; sm_89 ~1-2 h | 4-5 d |
| P3.4 gemm | 5-7 d: vendor-TU split, 43→17 fold, 2436-line `gemm_tests` rewrite, 17 caller ports | sm_120 6-8 h on 4 GPUs with screening (15 h without) | sm_120 ~4 h; full ctest 20 min; sm_89 ~2 h | 8-10 d |
| **Total** | **~13-17 engineer-days** | **~17-29 GPU-box hours** | | **~3.5-4.5 weeks** elapsed, serialized by stack review |

The critical path is P3.2 (tuner) → P3.4 (gemm code and its 2436-line test rewrite). posv (P3.1) can start now, alongside the running sm_89 potrf gate; its sm_120 sweep needs only the existing `factor_bench`.
