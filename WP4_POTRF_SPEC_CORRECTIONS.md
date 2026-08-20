# WP4 POTRF — corrections to the spec, before any code is written

`WP4_POTRF_SPEC.md` was authored at `13ee56f` ("docs: WP3 (trsm) and WP4 (potrf) kernel
specifications"). **`13ee56f` predates WP0's completion and the whole of WP1, WP2 and WP3** —
64 commits have landed since. This document records what a read-only verification pass against
`d6dd78d` found. Read it alongside the spec; where they disagree, this wins.

Method: six independent readers, one per claim-cluster (dispatch, primitives, contract, memory,
tests, numerics), each finding then adversarially refuted by a separate reader instructed to
default to "does not hold". Two findings were overturned outright and are marked ✱ below rather
than deleted; nine more were downgraded in severity. Everything else survived. Nothing was
modified, nothing was built, no benchmark was run.

The single most consequential item is not a dispatch-vocabulary item. It is **W1**: the spec's
entire SLM budget, and therefore every per-type `n` ceiling in §4.2 and every routing threshold
that follows from them, is computed against a number that is 2.2× too small — refuted by a kernel
already shipping in this tree.

---

## What survives, and is not re-litigated here

**The algebra is sound.** Re-derived by hand, none of it touched by WP0–WP3:

- **§2.3's stale-pivot fix.** Lane `k`'s `d[k]` has been decremented once per earlier `p`, so the
  shuffle at spec:56 broadcasts the *fully updated* Schur diagonal; the publish predicate
  `lane < ib && lane >= k` keeps lanes `ib..31` out of the A21 panel; the "no second barrier"
  argument holds because iteration `k+1` writes column `k+1` while iteration `k`'s update reads
  column `k`. The primitive is in production: `sycl::select_from_group` at
  `include/batchlas/blas/device/detail/group_blas_gemm.hh:77`, `group_blas_trmm.hh:118,132`,
  `group_blas_symm.hh:81,92`.
- **§2.4's forward substitution.** `X L11ᴴ = a` gives `X_c·conj(L(c,c)) = a_c − Σ_{p<c} X_p·conj(L(c,p))`,
  and `L(c,c)` is real so the divisor is `diag[c]` (spec:94-97). Dividing rather than
  reciprocal-multiplying matches reference `?trsm`. `mul_conj_b(a,b)` at spec:138 is exactly
  `a·conj(b)`.
- **§2.5's tile-index inverse.** `Σ_{c<ct}(Rt−c) = ct·Rt − ct(ct−1)/2`, `off[Rt] = Rt(Rt+1)/2`,
  and `rt = ct + (t − off[ct])` ranges over `[ct, Rt)`. `Rt ≤ 27` holds across all four ladders
  (spec:172-177) against the ceilings; worst case float `n=105, TS=4, NB=8` → `Rt=25`, 5 search
  steps. Replacing the float `sqrt` inverse with the prefix table is a strict improvement — *but
  see W9, the table has no home*.
- **§2.5's `device::herk` differential oracle.** All four sub-claims verified: `dispatch_rankk`
  guards every fast path with `if constexpr (detail::NdItemLike<Exec>)`
  (`group_blas_rankk.hh:437`) and falls through to `generic::rankk` at `:462`; `generic::rankk`
  (`:49-83`) does two `matrix_entry` loads per FMA at `:68-69` and contains **no barrier**;
  `rankk_workspace_elements` returns 0 at `:390` for a Group launch, because
  `make_group_launch_info` stamps `kind = Group` (`group_blas_common.hh:40-42`) and `NdItemLike`
  (`:181-187`) requires `get_sub_group()`, which a bare `sycl::group` lacks. The call spelling at
  spec:140 matches the convenience overload at `group_blas_rankk.hh:565-573` exactly.
- **§7's error analysis.** `κ(L11) = κ(A11)^{1/2}` from `σ(A11) = σ(L11)²`, and Cauchy interlacing
  on a leading principal submatrix of an SPD matrix gives `κ(A11) ≤ κ(A)`. The rejection of
  diagonal-block inversion stands, and §2 proposes none: the only reciprocal is the scalar
  `r = 1/dkk` applied to a column (spec:62-65), exactly LAPACK `?potf2`'s `sscal(1/ajj, …)`, and
  (P2) divides at spec:97. This is the same call the trsm spec made
  (`WP3_TRSM_SPEC_CORRECTIONS.md:17-19`).

**The syrk/herk rejection (§2.6) survives, and the reason is now stronger than the spec's own.**
Every behavioural claim holds: `kGramMaxTile = 128` and the `C.rows() > kGramMaxTile` gate at
`src/backends/syrk_gram_tiles.hh:65,319`; the non-float fall-through to a per-batch host loop at
`src/backends/cublas.cc:724-726`; `herk_gemm_preferred` = `batch >= 4 && n <= 768` at
`cublas.cc:382-388`; the arena lease `ctx.workspace(product_bytes)` at `cublas.cc:520`; the
~9 µs/launch price at `src/backends/syrk_custom_dispatch.cc:189`. **New since the spec:** WP1 S6
left the syrk custom gate in the facade guarded `Back == CUDA && is_same_v<T,float>`
(`src/dispatch/entry_points/level3.cc:341-343`), and herk forwards straight to the vendor with a
throw when none exists (`:302-307`). So in a **vendor-free build — the build WP4 exists for —
double `syrk` and every `herk` have no route at all and throw.** That is decisive where launch
latency was only expensive. Restate the reason; keep the decision.

**The routed-`gemm` instruction (§2.6, spec:158-160) is the one place the spec is ahead of where
WP3 started.** It calls the public `gemm<B>` with `GemmOptions`, not `sycl_gemm::gemm_custom` —
which is exactly what WP3 step 16 had to retrofit into trsm (`level3.cc:186-231`, the injected
`TrsmTrailingGemm` at `src/sycl/trsm_native.hh:105-111`, 18.8 ms → 11.19 ms). Carry it forward
unchanged. Its **justification** is stale — see S1 — but its instruction is right.

**[FIX-B-trap] (spec:160) is still true and still untouched.** `MatrixView::operator()(Slice,Slice)`
at `include/batchlas/blas/matrix.hh:1129` carries the comment "Do not propagate the parent
pointer-array into a slice" at `:1136-1139` and the very next line, `:1140`, passes
`data_ptrs_.data()`. Build every sub-view explicitly from `A.data_ptr() + off` with parent `ld`
**and `stride` and `batch`** — `trsm_native.cc:590-599` records that the constructor defaults
stride to `ld*cols` when 0 is passed, "and every batch item after the first reads the wrong
matrix". `ortho.cc:77` (spec says `:75`) builds its Gram `C` with a pointer array.

**`fold_symmetric_product_into_triangle` behaves as §2.6 assumes.** Signature
`(Queue&, C, product, T beta, Uplo)` at `src/extensions/symmetric_product_fold.hh:29-34`; early
return on `total_elements == 0` at `:37-40`; `ignore_c = beta == T(0)` at `:49` applied at `:68`,
so `beta == 0` never reads `C`. And `accumulate_hermitian` (`cublas.cc:407-433`) genuinely has no
zero-extent guard, so §2.6's `m2 == 0` skip is required.

**The nd_range is clean — checked against both of this repo's recurring defects.** Concurrency is
`ceil_div(batch,G)·G·L ≥ batch·L` (spec:196-197) and `batch·tiles·TR` (spec:230-231), never
`batch` alone; and neither `L` nor `G` is a function of `batch`, so there is no `SMs/batch`-style
cap to make a path structurally dead at large batch. `OpShape`'s only machine-size fields are
`max_sub_group` and `compute_units` (`route.hh:239-240`) and §3 reads neither. §3.5's two worked
figures re-derive exactly.

**§3.3's barriers are legal in both scopes.** No barrier sits in divergent control flow; `ib` is
uniform (spec:53) and `akk` is a shuffled value, so both `break`s are uniform; the per-`k`
sub-group barrier inside `if (sg_id == 0)` is entered by that whole sub-group, which is
well-formed. The early-return precedent is real: `src/extensions/sytrd_cta.cc:101-111`. **Sound
only at `G == 1` under WorkGroup scope** — which is what W10 breaks.

**The CTA tile term of the SLM formula is correct.** Checked for the WP3-style padded-stride
overrun and it is *not* present: the largest index any phase produces is `(n−1)·LDA + (n−1)`,
which is `< LDA·n` for both parities of `n`. Contrast `WP3_TRSM_SPEC_CORRECTIONS.md`'s finding 4.
The `LDA = n | 1` justification also holds — the same odd-LD bijection argument is in the tree at
`src/extensions/gesvdj_cta.cc:176-180`.

**§4.3's packed-triangle figures re-derive exactly** (8320 / 4224 / 4224 / 2176 B), including the
[FIX-B1-secondary] correction of the candidate's 8448 for `complex<float>`, and "registers, not
SLM, is the limit there" holds under the spec's own occupancy formula at `TR = 256`.
`regs × WG ≤ 65536` is respected with wide margin everywhere (max WG 256 at 64–96 regs = 16 384 –
24 576 of 65 536).

**§5.1's signature claim holds byte for byte.** `potrf_buffer_size` at
`include/batchlas/blas/functions/potrf.hh:44-47` and `potrf` at `:60-65` are character-identical
to the spec, and the facade defines and instantiates exactly these
(`src/dispatch/entry_points/factorization.cc:159-183`, `:209-217`). The option-struct spellings
(`options.hh:539-568`) and the `EmptyBracesAreAmbiguous` `= delete` guard (`:570-589`, deletes at
`:591-595`) are untouched, and a distinctly named `potrf_dispatch` introduces no new overload
candidate (the variadic dispatch macro cannot deduce a braced-init-list, `options.hh:113-121`).

**§5.5's `max(chosen, vendor)` is right, and load-bearing for a second reason the spec does not
give** — see W8. **The BumpAllocator contract it depends on is intact**: `allocate()`'s capacity
check compares the alignment-rounded size against bytes left from the *unaligned* cursor while
the cursor advances by the raw extent, so the sizing pool takes `max(need_for_check, need_for_data)`
(`include/batchlas/util/mempool.hh:83-105`), and `required_bytes()` rounds to the coarsest
alignment quantum (`:44-58`). `workspace_bytes(Fn&&)` at `:185-190` is exact, and the one-layout-two-
passes precedent is at `src/extensions/sytrd_blocked.cc:656-687`. `max(a,b)` is safe **only**
because both terms come from `required_bytes()`/`allocation_size`; do not "optimise" the layout
functions into a hand-summed arithmetic expression.

**§5.4's two solve shapes are ordinary interior points of WP3's canonicalisation.** Re-derived
against `src/sycl/trsm_native.cc:75-93`: (Right, Lower, Trans/ConjTrans, NonUnit) →
`do_trans=1, op_is_lower=0, unit=0, fwd=1`; (Left, Upper, …) → `op_is_lower=1, fwd=1`. Neither is
a corner, and `RouteTable<Op::trsm>::supports()` (`route_trsm.hh:134-186`) has no Side/Uplo/Trans
restriction at all.

**Nine of §6's citation rows are usable verbatim**, re-verified at or within one line:
`select_from_group` (5 sites), `BumpAllocator::measuring()/required_bytes()/workspace_bytes`,
`fold_symmetric_product_into_triangle`, `tuning_env_override` (`tuning_params.hh:33-41`), the
sub-group-32 enumeration (`sytrd_cta.cc:319-333`), `util::get_raw_ptr`
(`sycl-local-accessor-helpers.hh:23`), `group_reduce_sum_select_from_group`
(`sytrd_cta_device.hh:79`), and the `matrix.hh` slice trap. The "deliberately not used" list also
re-verifies: `device::fill` exists (`group_blas_fill.hh:8,21`, zero barriers in the whole
35-line header) and every level-3 fast path really is float-only
(`group_blas_subgroup_common.hh:433-452`, `:454-476`).

**Two test commands work as written.** `ctest -R "options_api_tests|linalg_layer_tests|ortho_tests"`
selects exactly 3 tests (run by hand: #8, #11, #20) — spec:465's Phase-0 gate is executable.
And spec:590's benchmark claim holds: `batchlas_register_benchmark` sets `EXCLUDE_FROM_ALL ON`
(`benchmarks/CMakeLists.txt:77`), no benchmark is registered as a ctest test, and adding
`potrf_benchmark` to `BENCHMARK_TARGETS` (`:1-69`) is the whole edit. §9.1's label table is also
correct for all four targets it labels (`tests/CMakeLists.txt:136-137,141,143`), and all four
`options_api_tests` line ranges at spec:505 are still exact.

---

## Wrong-edit findings — following the spec here produces incorrect code

Ordered by severity.

### W1. The SLM budget is 2.2× too small, and every per-type capacity in the spec follows from it

**Spec:** spec:264 — `slm_budget = runtime_local_mem_size − 4096  // = 45056 on this box`, and
spec:267 — "The 99 KB sm_89 opt-in carveout is not exposed through SYCL `local_accessor`, so
48 KB stays the hard per-work-group ceiling and 45056 is what we spend." spec:5 presents
49152/45056 as re-verified facts about the device.

**Reality:** `DeviceProperty::LOCAL_MEM_SIZE` maps to `sycl::info::device::local_mem_size`
(`src/util/queue-impl.cc:323`), and the tree records that query returning **101,376 B** on this
box, not 49,152. `src/extensions/gesvdj_cta.cc:1011-1016`, read verbatim: *"Per-problem LDS at
C=64 with the V tile resident is 37,952 B for float, 71,744 B for double and complex<float>, and
138,816 B for complex<double>; **this device reports 101,376 B**, so complex<double> with vectors
does not launch at all and the others fall to 2 or 1 work-groups per SM."* All three byte figures
re-derive exactly from `gesvdj_cta.cc:194-218` (LD=65, kTileElems=4160, kPairTabBytes=4032).
Corroborated at `include/batchlas/blas/dispatch/route_gesvd.hh:63-64` and
`tests/gesvdj_cta_tests.cc:88`.

**The double/C=64 route is shipping and reachable** (`gesvdj_cta.cc:1022-1028` returns 64 for
every non-cdouble type; `route_gesvd.hh:88` gates only on `max_dim > gesvd_jacobi_max_dim<T>`),
so a **71,744-byte `local_accessor` allocation launches in production today**
(`gesvdj_cta.cc:245-257`). The 48 KB ceiling is refuted by a kernel already in the tree.

The 49152 in `build/include/batchlas/device_limits.hh:23` is not a detected property at all:
`cmake/BatchLASDetectSYCL.cmake:44-45` hardcodes it for any `^nvidia_gpu_sm_[0-9]+$`, and
`collect_sycl_device_limit_info` (`cmake:243-356`) parses Type/Name/sub_group_sizes/Architecture
from `sycl-ls --verbose` and never `local_mem_size`.

**Consequence.** spec:264 and spec:472 both tell the implementer to size from the *runtime*
query; spec:273's table asserts the result is 45056. `runtime − 4096 = 97,280`. The two cannot
both be transcribed. Re-derived with §4.1's own formula at 97,280, the fit ceilings are
**float 155 / double 109 / complex<float> 109 / complex<double> 77**, not 105/74/74/52 — ~1.47×
in `n`, ~2.2× in matrix area. (Boundaries checked: float 155 → 96,228 fits, 156 → 98,096 does
not; double 109 → 95,240 fits, 110 → 97,872 does not; cdouble 77 → 94,992 fits, 78 → 98,592 does
not.) Ship the small numbers in `potrf_cta_max_n<T>()` and `supports()` returns false for a band
the kernel can hold — and per `route_resolve.hh:60-63`, which re-walks the order testing only
`is_native(*r) && supports(*r,s)`, **float `n` in 106..155 has no route at all in a vendor-free
build**. spec:593's grid ("plus 52/74 exactly at the per-type ceilings") measures the wrong
boundary.

The §4.2 table is *internally* correct against 45056 — all four ceilings and all six spot-checks
re-derive — so this is not an arithmetic error. It is the input.

**Note the honest uncertainty.** 101,376 is what `gesvdj_cta` records and 71,744 is what
production allocates; nobody in this pass ran a potrf-shaped kernel at 97,280. See Open question 1.

### W2. The batch thresholds are inside `supports()`, which makes both native routes unreachable — including by force

**Spec:** spec:559 `return n <= potrf_cta_max_n<T>(slm_budget) && batch >= kPotrfCtaMinBatch;`
and spec:567 `return batch >= kPotrfBlockedMinBatch;`, both under the heading "Hard gate
(correctness/fit)"; spec:574 then sets `kPotrfCtaMinBatch = kPotrfBlockedMinBatch = INT_MAX` at
merge, "i.e. both native providers are reachable only by force".

**Reality:** the tree requires `supports() == correctness only, never a speed cutoff` /
`preferred() == the measured window`, and says why in as many words at
`include/batchlas/blas/dispatch/route_gemm.hh:20-28`: *"Move the window into `supports` and a
1024³ float GEMM at batch 256 suddenly has no supported route at all."* `route_trsm.hh:14-28`
restates it: *"A speed number in supports() does not make trsm slower on a vendor-free box; it
makes trsm THROW."*

Two independent failures follow, and the second is the one that guts §9:

1. **The vendor-free fallback dies.** `route_resolve.hh:56-66`: the preference walk at `:57-58`,
   then `if (!vendor_available)` at `:60` re-walking on `is_native(*r) && Table::supports(*r, s)`
   at `:62`, then `return Route{Origin::Vendor, Algorithm::Auto}` at `:65`. The facade then
   throws at `factorization.cc:165-167`. With supports() false at every shape, potrf keeps
   throwing `NoRouteError` in a `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build *after* the native
   kernel is written and linked.
2. **"Reachable only by force" is false.** `route_resolve.hh:8-10` states that *"a forced route
   bypasses `preferred` … but never bypasses `supports`"*, implemented at `:101`
   (`if (Table::supports(forced, s)) return forced;`) falling through to `automatic()` at `:111`.
   The bare-origin branch at `:87-98` gates on supports() in both walks too. So **every §9 test
   that obeys spec:514 and pins the route silently runs cuSOLVER and passes green over an
   untested native kernel** — the exact outcome spec:514 says it exists to prevent.

**Fix:** the merge-state knob is `preferred()`, not a default and emphatically not `supports()`.
That is literally how trsm shipped: `route_trsm.hh:53-55` records "preferred() was all-false
until WP3 step 9 measured the grid". Move **every number** in §10.2's table, both batch
thresholds, the "n > 1024 blocked loses" prior and §10.3's 0.90× gate into `preferred()`.

**Related, same file, same class:** spec:567's `if (n <= potrf_cta_max_n<T>(...)) return false;`
in the *blocked* arm is a fit judgement between two native routes, not a correctness claim. The
tree expresses that with the order ladder — `route_trsm.hh:116-120`, *"The order is a capability
ladder, not a preference"* — and trsm's Blocked arm carries no lower bound at all (`:172-177`).
With it in supports(), a forced `blocked` at small `n` does not measure the blocked kernel; per
`route_resolve.hh:101-111` it falls back to `automatic()`, which at merge (preferred all-false)
returns `{Vendor, Auto}`. §10.3 asks for all three routes pinned at overlapping `n` — the real
overlap is the ortho grid's `k = 16/32` for all types and `k = 64` for double/`complex<float>`
(the readers' first draft cited `k = 128` as a straddle; that is wrong, 128 exceeds every CTA
ceiling).

### W3. The dispatch vocabulary the spec speaks was deleted; §1, §6, §8 and §9.1 do not compile

**Spec:** spec:11 (`Provider::Auto` = Vendor, `DispatchPolicy::forced`), spec:381
(`DispatchPolicy::require_in_order`), spec:415 (§6's reuse row), spec:462 (§8 step 0.2,
`choose_potrf_provider`), spec:474 (`policy.forced == BatchLAS_CTA`), spec:486
(`Provider::BatchLAS_Blocked`), spec:514, spec:569 (`DeviceCaps` / `query_caps`), spec:574
(`default_order_cta_blocked_vendor_netlib`, `std::array<Provider,6>`).

**Reality:** `ls include/batchlas/blas/dispatch/` returns coverage.hh, no_route.hh, op.hh,
route_compiled.hh, route_env.hh, route_gemm.hh, route_gesvd.hh, route.hh, route_ormqr.hh,
route_resolve.hh, route_trsm.hh, vendor_available.hh — **no provider.hh, context.hh or env.hh**
(deleted in `52acc65`, "wp0 S4f: move syev onto Route, and delete the Provider mechanism").
`route.hh:32-35` is the tombstone: *"This is the only routing vocabulary in the tree. `Provider`,
DispatchPolicy and the three dispatch/{provider,env,context}.hh headers are gone."* Grep over
`include/`, `src/` and `tests/` finds `Provider::`, `DispatchPolicy`, `policy_from_env`,
`query_caps`, `DeviceCaps` and `require_in_order` only in comments and unrelated gtest suite
*names* (`tests/gemm_tests.cc`). `default_order_cta_blocked_vendor_netlib` has **zero** hits
anywhere. `choose_potrf_provider` has zero hits; the three choosers that existed survive only as
tombstone prose (`route_ormqr.hh:7`, `ormqr.hh:196-197`, `route_gesvd.hh:11`).

Piece by piece, what replaces each:

| spec spelling | what to write instead |
|---|---|
| `Provider`, `DispatchPolicy` | `Route{Origin, Algorithm}` (`route.hh:49`, `:67`, `:95`) |
| `choose_potrf_provider`, `policy_from_env` | `RouteTable<Op::potrf, T>` + `dispatch::resolve_route` |
| `default_order_cta_blocked_vendor_netlib` (`env.hh:57-64`) | a file-scope `inline constexpr Route kPotrfOrder[]` of natural length plus `order_begin()/order_end()` computed with `sizeof` — see `route_trsm.hh:121-125,327-330`. `route_gemm.hh:43-46` notes this "removes the truncation hazard of the four hand-counted `std::array<Provider,6>` sites", so the hazard §10.2 is trying to avoid is already structurally impossible |
| `const DeviceCaps& caps`, `caps.is_gpu` | `OpShape` (`route.hh:222-244`), whose device facts are **fields**: `is_gpu` `:238`, `max_sub_group` `:239`, `compute_units` `:240`, `heterogeneous_batch` `:236`. `route.hh:217-220` records why `query_caps` was removed (three SYCL round-trips + a heap allocation per op invocation) |
| `slm_budget` / `has_sg32` as call parameters | fields on a derived shape struct, filled by a builder in `src/`. `TrsmShape` carries `cta_max_n` (`route_trsm.hh:97`) and `blocked_available` (`:110`) for exactly this reason, with `trsm_cta_max_n<T>()` *declared* in the table header at `:84-85` and *defined* in the kernel TU |

**There is no `RouteTable<Op::potrf, T>` in the tree** — grep for `struct RouteTable<` returns
exactly five specialisations (`route_gemm.hh:54`, `route_ormqr.hh:51`, `route_gesvd.hh:75`,
`route_trsm.hh:128`, `functions/syev.hh:817`). But `Op::potrf` already exists (`route.hh:137`)
with its name at `:195`, so the env stem and the coverage rows come free the moment a table is
written. The missing artefacts are `include/batchlas/blas/dispatch/route_potrf.hh` and a
`src/backends/potrf_route.hh` builder.

**Also gone:** spec:381's `DispatchPolicy::require_in_order`. The rule is right and the tree has
a well-established spelling used by at least ten drivers:
`if (!ctx.in_order()) throw std::runtime_error("<name>: requires an in-order Queue");` —
`sytrd_blocked.cc:1008-1010`, `gebrd_blocked.cc:123`, `syev_two_stage.cc:69`,
`ormqr_blocked.cc:629`, and others. `Queue::in_order()` is a plain accessor
(`include/batchlas/util/sycl-device-queue.hh:315`), constructor default `true` (`:239`).

**Consequence:** every line of §1, §6, §8 and §9.1 naming these is a compile error. Note also
that spec:11's framing — a default provider expressing "no user-visible route change on merge" —
has no referent: `legacy_unset_default` returns `{Origin::Auto, Algorithm::Auto}` for **every**
op since WP2 E6 (`route_env.hh:123-148`), and `Origin::Auto` runs the preference walk
(`route_resolve.hh:67-69`).

### W4. `BATCHLAS_POTRF_PROVIDER` is read by nothing; the working name is `BATCHLAS_POTRF_ROUTE`, and nothing reads *that* yet either

**Spec:** spec:11, spec:362, spec:474, spec:514, spec:590 and §2.5's `BATCHLAS_POTRF_UPDATE=herk`
(spec:140).

**Reality:** `grep -rn BATCHLAS_POTRF` over `.cc/.hh/.py/.sh/.txt/.cmake` in the worktree returns
**zero hits** outside the spec itself. `legacy_variable_for` (`route_env.hh:109-121`) has cases
for gemm/symm/syrk/syr2k/trmm/syev/gesvd/ormqr and `default: return {}` at `:119` — **no
`Op::potrf` case**, so `parse_route_env`'s legacy arm never fires for potrf. The canonical name is
built at `route_env.hh:217` as `"BATCHLAS_" + op_env_stem(op) + "_ROUTE"` =
**`BATCHLAS_POTRF_ROUTE`**. Accepted values: `vendor`/`native` (bare origin, `:48-53`),
`cta`/`blocked` (bare algorithm, `:55-71`, which implies `Origin::Native` at `:92-97`), or
`native:cta`.

**It only works once a `potrf_route` builder calls `parse_route_env(Op::potrf)`.** Today the
facade calls `backend::potrf_vendor` unconditionally (`factorization.cc:159-171`) with no
`resolve_route` anywhere on the path.

**`BATCHLAS_POTRF_UPDATE=herk` cannot be a Route at all** — there is no `Algorithm` value that
expresses "trailing update = herk" (`route.hh:159-176`). If it is kept, make it an explicit local
`getenv` in the kernel TU, documented as being *outside* the route vocabulary; do not launder it
in as if it were one.

**Consequence:** identical to `WP3_TRSM_SPEC_CORRECTIONS.md` finding 2. Every §9 test and every
§10.3 benchmark row that thinks it pinned `cta` or `blocked` silently runs the vendor. The whole
grid is vendor-vs-vendor and Phase 1 ships untested by construction. The `DispatchPolicy` form at
spec:514 does not compile; the env form compiles, passes, and tests nothing.

**The two live ways to pin** are: `BATCHLAS_POTRF_ROUTE=cta|blocked|vendor` once the builder
reads it; or, at the pure layer, calling `RouteTable<Op::potrf,T>::supports/preferred` and
`resolve_route` directly with an explicit `Route{Origin::Native, Algorithm::CTA}` — which is how
`tests/route_vocabulary_tests.cc` pins trsm, including a `ClearRouteEnv` RAII guard at `:52-60`
that clears both spellings so an inherited variable cannot poison the case (`kCta` constructed at
`:392`, asserted at `:407,:409,:453,:463`).

### W5. There is exactly one hook point, it is the facade, and it is above the vendor test — `src/linalg-impl.hh` has no potrf in it

**Spec:** spec:463 (§8 step 0.3) — "`src/linalg-impl.hh` / instantiation sites | route the public
entry points through `potrf_dispatch`".

**Reality:** `grep -n potrf src/linalg-impl.hh` returns exactly **one** line, `:732`, inside a
comment about LAPACKE return values. No entry point, no forwarder, no instantiation. WP0 S5c
moved both public entry points into the facade TU, where they are defined *and* instantiated:
`potrf` at `src/dispatch/entry_points/factorization.cc:159-171`, `potrf_buffer_size` at
`:173-183`, the `POTRF_ALL` macro at `:209-217`, expanded for CUDA at `:226`, ROCM at `:232`,
NETLIB at `:237` (keyed on the *device family*, `:219-224`).

**The structural point:** in **both** functions the vendor-available test is the first statement —
`:165` and `:177` are `if constexpr (!dispatch::solver_vendor_available<B>)` whose false arm
throws. **Anything placed after it is unreachable in the vendor-free build WP4 exists for.** The
native hook goes at the top of each body.

The pattern to copy end-to-end is trsm in `level3.cc`, in this order: validate (`:174`, with
`:167-173` explaining why), resolve the route passing vendor availability (`:179-181`), dispatch
native (`:186-230`), vendor `if constexpr` **last** (`:232-237`).

**Two details that differ from the obvious guess:**

- potrf's availability predicate is **`solver_vendor_available<B>`** with `kSolverLibrary<B>`
  (cuSOLVER on NVIDIA), not `factorization_vendor_available` (cuBLAS). The two differ on CUDA —
  `vendor_available.hh:41-45` vs `:47-52`, `kSolverLibrary` at `:68-70`.
- **Pass `/*vendor_available=*/dispatch::solver_vendor_available<B>` into `resolve_route`**, as
  trsm does at `level3.cc:181`. `syev` does **not** (`syev.hh:948` takes the default `true`,
  `route_resolve.hh:36`), and an op that omits it never reaches the `:60-64` vendor-free fallback
  at all.

### W6. Three correctness gates are missing from `supports()`, and nothing on the potrf path validates arguments

§10.1's predicate list is `is_gpu`, `has_sg32`, `n < 1`, the SLM ceiling, and blocked's
`uplo == Upper`. **Those five are right** and each has a live precedent — GPU-only
(`route_trsm.hh:138-142`, explicitly labelled "Not a speed judgement"), degenerate extents
(`:153-161`), the capacity cap ("a hard capacity, not a tuning knob: above it there is no kernel
object to launch", `:163-170`), and the sub-group enumeration the spec correctly insists on,
which is verbatim what `sytrd_cta.cc:319-333` does today. Keep them. Three more are missing:

**(a) `blocked_available` — is the blocked driver compiled into this build?** trsm treats this as
correctness and says why at `route_trsm.hh:99-110`: *"Reporting Blocked as supported while it does
not exist makes `resolve_route` hand a vendor-free caller a route the facade cannot service … the
table must describe the build, not the design."* The gate is `return s.blocked_available &&
s.cta_max_n >= 1;` (`:172-177`). §8 puts `potrf_blocked.cc` in Phase 2 while §10.1's blocked arm
would report support from Phase 1. *(Contingency noted: with the merge INT_MAX in place the arm
is false everywhere anyway, so this surfaces exactly once W2 is fixed — which is the fix this
document recommends.)* `route_compiled.hh:1-24` states the general LINKED-vs-REACHABLE question,
including that `B == Backend::CUDA` "is wrong TODAY in the vendor-free build".

**(b) `has_sg32` must gate the blocked arm too**, since §8 step 2.1 makes the blocked leaf the
Phase-1 `[[sycl::reqd_sub_group_size(32)]]` device function. This one is an inference, not a
transcription: `route_trsm.hh`'s supports() contains no sub-group check at all, and syev gates
`max_sub_group >= 32` only on its CTA arm (`syev.hh:837`) — the weaker test the spec explicitly
rejects. See Open question 3.

**(c) `heterogeneous_batch`.** §5.2 (spec:313) asserts "uniform `n`, `ld`, `stride`" as if
contracted. It is not. `MatrixView<T,Dense>::is_heterogeneous()` (`matrix.hh:1034`) returns
`!active_rows_.empty() || !active_cols_.empty()`, such views are publicly constructible
(`Matrix::set_active_dims` `:771`, `MatrixView::with_active_dims` `:1339`), `OpShape` carries the
field (`route.hh:236`), and trsm makes it a **correctness** gate with a named reason
(`route_trsm.hh:144-151`): *"One launch covers the whole batch with a single (order, q, ld,
stride) tuple, so per-item extents would be read at the wrong addresses."* A native potrf has
identically no batch walker; it reads at `data_ptr() + b*stride` with the capacity extents and
silently factorises the wrong shapes.

  Two readers disagreed on today's exposure and both readings are worth carrying:
  the cuSOLVER path is equally blind (`cusolver.cc:61,64-65`), so this is a *hardening* gap, not
  a regression — but netlib's **batched** path is not blind: `netlib_lapack.cc:1029` calls
  `A_view[i].rows()`, and `MatrixView::operator[]` → `batch_item` (`src/matrix.cc:2311-2313,
  2027-2029`) constructs the item from the *active* per-item extents. That strengthens the
  prescription rather than weakening it.

  **And the gate needs a writer.** `src/backends/trsm_route.hh:40-56` never sets
  `s.heterogeneous_batch` — the only writer in the tree is `gemm_variant.hh:209` (predicate at
  `:27-32`). An implementer who copies trsm's builder verbatim gets the same dead gate. The
  predicate and the builder line must land together.

**(d) Nothing validates.** There is no `potrf_validate_params` anywhere — grep for
`validate_params` returns only `functions/trsm.hh:39`, `level3.cc:174`, `cublas.cc:1104`,
`rocblas.cc:148`. `require_square`/`require_info_span` are attached only to the *option-struct*
overloads (`options.hh:548-549,557-558,565-566`), and the workspace-taking `<Backend B>` overload
at `:539-543` — **the spelling `ortho.cc:200` uses** — has neither. So the positional entry point
can be handed a non-square view directly, and §10.1 has no `m == n` gate (syev's table opens with
one, `syev.hh:828`); §10.1's signature takes only `int n`, so non-squareness is not even
representable in it. Add the `m == n` gate, and follow trsm's precedent by hoisting validation
into the facade *before* the shape builder — the builder reads `A.rows()`/`A.batch_size()` and
would otherwise index a non-conforming view.

### W7. The route table must stay pure — the env read goes in the `src/` builder, not in `supports()`

**Spec:** spec:474 (§8 step 1.4) — "`potrf_supports_cta` returns true only under
`policy.forced == BatchLAS_CTA` or `BATCHLAS_POTRF_PROVIDER=cta`".

**Reality:** two violations. (a) `route_resolve.hh:19-20`: *"Everything here reads only its
arguments — no getenv, no SYCL query — which is what makes an op and its `*_buffer_size` query
reach the same route by construction."* Repeated at `route_gemm.hh:22-23,30-32`. (b) An env-gated
`supports()` also breaks forcing: `resolve_route` never bypasses supports() (`:8-10`), so making
support conditional on the force makes the two mutually recursive in meaning.

The live counter-example is `src/backends/trsm_route.hh`: the shape builder at `:30-57`, the env
read at `:61-77` (`parse_route_env(dispatch::Op::trsm)` at `:73`, `legacy_unset_default` at
`:75`), with the file header at `:5-8` saying it lives in `src/` for exactly this reason.

*A reader's stated rationale here needs correcting:* "a table that calls getenv desynchronises
`potrf` from `potrf_buffer_size` mid-process" is not the differentiator — a builder-level read has
the identical hazard, since both entry points call `parse_route_env` once each per call. The real
reasons are the documented purity contract and **header linkability**: `route_trsm.hh:74-82`
spells out that a table which calls into the kernel TU cannot land before the kernel does. That
is why `trsm_cta_max_n<T>()` is *declared* in the header and *defined* in the kernel TU.

### W8. The buffer-size query: `choose_potrf_provider` does not exist, and the vendor path re-enters the public query

**Spec:** spec:358 — "`potrf_buffer_size` calls the **same** `choose_potrf_provider` the call uses
and returns `max(chosen_provider_size, vendor_size)`."

**Two separate problems.**

**(a) The symbol does not exist** (W3). The *principle* survives and is enforced structurally:
`factorization.cc:8-10` — *"Each op moves TOGETHER WITH ITS BUFFER-SIZE QUERY. Splitting them
would let the two resolve differently, which is the defect class S4d found in ormqr (buffer size
2560 bytes, call demanded 276480)."* The live precedent is syev: one builder
`detail::syev_route<B,T>` (`syev.hh:940-949`), two call sites (`:965-967` and `:1088-1090`). If
the hook lands only in `potrf` and not in `potrf_buffer_size`, the query returns the vendor size
while the call runs a native kernel — an under-allocation, not a slowdown.

**(b) The cuSOLVER vendor implementation calls the PUBLIC query.** `src/backends/cusolver.cc:56`,
unconditional, inside `backend::potrf_vendor`:

```
auto Lwork = potrf_buffer_size<B>(ctx, descrA, uplo) - BumpAllocator::allocation_size<int>(ctx, 1);
```

`namespace batchlas::backend` declares only `potrf_vendor_buffer_size`, so unqualified lookup
escapes to `batchlas::potrf_buffer_size` — **the facade** (`potrf.hh:44`,
`factorization.cc:173-183`). `Lwork` is then consumed at `:57-58` (`pool.allocate<std::byte>`)
and `:61` (passed to `cusolverDnXpotrf` as its workspace size) on the `batch_size() == 1` branch.

Today facade == vendor, so the loop is invisible. The moment `max(chosen, vendor)` lands in the
public query, a batch-1 cuSOLVER call is handed `native_blocked_scratch_bytes − 16` — tens of MB —
as its cuSOLVER workspace size. It does not throw (both terms are alignment multiples, so the pool
fits exactly), which is precisely what makes it dangerous: the vendor path's workspace slicing
becomes a function of the native chooser. **Repoint `cusolver.cc:56` at
`backend::potrf_vendor_buffer_size<B,T>` as step 0, before touching the public query.**

This is also the strongest reason to keep the `max()`: `options.hh:546-552`'s arena overload
performs two independent resolutions (query at `:550`, call at `:551`), and the vendor path's own
allocation is downstream of the query's answer.

### W9. `off[]` has no term in the SLM formula, no specified writer, and no barrier

**Spec:** spec:112 ("`off[]` is an SLM prefix table (`Rt <= 27` entries)"), spec:137 ("the table
is ≤ 108 bytes"), indexed at spec:114-115, arriving in the signature at spec:426 as
`const int* off`. spec:262: `slm_per_matrix = LDA*n*sizeof(T) + NB*sizeof(real_t) + 64
// tile + diag[] + fail/pad`.

**Three defects, all in the same object:**

1. **No size term.** The formula's own comment accounts for all three of its terms. 108 bytes does
   not fit a 64-byte "fail/pad" slot that spec:262 already spends on `slm_fail`. Re-derived worst
   cases from spec:186-187 and the spec:177 constants: float `n=105, nb=16, TS=4` → `Rt_0 = 23`
   → 92 B; complex<double> `n=52, nb=8, TS=2` → `Rt_0 = 22` → 88 B. An implementer who carves
   `LDA*n` elements, then `NB` reals, then a 64-byte tail, and writes `off[0..22]` into it, writes
   **28 B past the end** at `G=1` and into matrix `g+1`'s tile at `G>1` — silently wrong results
   only when packing is on. (Whether it is an actual OOB depends on the implementer packing it
   into the tail rather than declaring a second `local_accessor`; what is *unconditional* is that
   the only sizing formula in the spec omits an object the spec itself sizes, and that formula
   drives both the budget check and `potrf_cta_max_n<T>()`.)
2. **No writer.** `off[]` depends on `Rt = ceil_div(m2, TS)`, and `m2 = n − j − ib` **shrinks at
   every panel**. It must be rewritten in SLM at the start of every panel. The spec says nothing
   about who writes it or when.
3. **No barrier.** §3.3's barrier list at spec:217 is stated exhaustively — (a) after the load,
   (b) after (P1), (c) after (P2), (d) after (P3) — and has no slot for publishing `off[]`.

**Consequence:** computed once from the first panel's `Rt`, **every panel after the first decodes
onto the wrong `(rt, ct)`** and silently updates the wrong blocks of A22; recomputed without a
barrier, it races the binary search at spec:114. The residual test would catch it only for `n`
large enough to have more than one panel.

**Fix:** add `+ 4*ceil_div(n − nb, TS)` to spec:262, or hoist `off[]` to one copy per work-group
(legal, since `n` and `nb` are work-group-uniform) — see Open question 7. Re-checked: no §4.2
feasibility row flips even at the 45056 budget (float 105 → 44,320 B; cdouble 52 → 44,312 B).
Precedent for putting the auxiliary object in the size:
`group_blas_subgroup_common.hh:56,58`. Note also that at cdouble `n=53` the true overshoot is
**16 bytes**, not the 176 spec:278's arithmetic implies (it uses `NB=16` where spec:177 gives
`complex<double>` `NB=8`) — so there is far less slack there than the spec suggests.

### W10. §3.4's leaf scope contradicts §3.2's own L ladder, for float

**Spec:** spec:225 — "**Leaf** — the CTA kernel launched on the `ib × ib` diagonal sub-view, at
`Scope::SubGroup` with `G` matrices per work-group."

**Reality:** §3.2's own arithmetic says otherwise for float. With `NB_o = 64` (spec:238) the leaf
order is 64; `nb = 16` → `m2_0 = 48` → `Rt_0 = 12` → `Ntiles_0 = 78`, which lands in the
`64 < Ntiles_0 ≤ 256` bucket at spec:189-191 → **L = 64** → spec:193-195 forces **G = 1** →
spec:215 makes the scope **WorkGroup**. spec:204's own worked table agrees: "`n=64 → 78 → L=64`".
And spec:215's WorkGroup row fixes `matrix id = wg_id` and justifies "out-of-range item cannot
occur" with `num_wg == batch`, both true only at `G == 1`.

**Two narrowings, both real:** it is **float-only** (double and `complex<float>` at `NB_o=32`
give `Ntiles_0 = 10` → L=32 → SubGroup; `complex<double>` at `NB_o=16` likewise), and it is
contingent on `resolve_potrf_nb` picking the spec:177 default `nb=16` — at `nb=32` the float leaf
gives `Ntiles_0 = 36` → L=32 and spec:225 would be right.

**Consequence:** obey spec:225's `Scope::SubGroup` for a 64-order float leaf while spec:189-191
computes L=64, and barriers (b)/(c)/(d) disagree about whether they are sub-group or work-group
barriers — a sub-group barrier across a 64-work-item matrix is exactly the race §3.3 exists to
fix (spec:210). One of spec:225 and spec:238 has to change before either is implemented.

*(A related hole: spec:184's `nb = resolve_potrf_nb<T>(n, hint)` has no type for `hint` anywhere
in the document, and the `Provider`/`DispatchPolicy` vocabulary it would have drawn on is gone
(W3). `nb` feeds `m2_0` → `Rt_0`/`Ntiles_0` → `L` → `G` → `wg_size` → `Scope`, so the whole
nd_range derivation is downstream of an unspelled parameter.)*

### W11. `ctest -L blas -L ortho` runs zero tests and exits 0

**Spec:** spec:512 — "Per the repo's selective-testing policy: scope with `ctest -L blas -L ortho`
during development."

**Measured by hand in this worktree's `build/`:** `ctest -L blas -L ortho --show-only` prints
`Total Tests: 0` and exits **0**. Repeated `-L` is an AND, and no test carries two component
labels: `batchlas_test_component` (`tests/CMakeLists.txt:174-183`) `return()`s on the first
matching component, and `:251-257` sets `LABELS` to exactly that one component plus optionally
`slow`.

Measured alternatives, all by hand: `ctest -L blas` → 15; `ctest -L ortho` → 5;
`ctest -L "blas|ortho"` (one `-L`, **bare pipe**, quoted) → **20**;
`ctest -L 'blas\|ortho'` (backslash-pipe) → **0**.

This is verbatim the defect `WP3_TRSM_SPEC_CORRECTIONS.md:94-111` recorded, reproduced in this
spec — under a section whose entire argument is "these six targets cover potrf". **Always pass
`--no-tests=error`:** measured, `ctest -L blas -L ortho --no-tests=error` exits 8 with "No tests
were found!!!" instead of the silent exit 0.

**The exact commands for potrf work:**

- iterate: `ctest --test-dir build -R '^(potrf_tests|options_api_tests|linalg_layer_tests|ortho_tests)$' --no-tests=error --output-on-failure` — measured 3 tests today, 4 once `potrf_tests` exists.
- broad pre-push sweep: `ctest --test-dir build -L "util|blas|ortho" -LE slow --no-tests=error` — measured 33 tests.

*(A related scope inaccuracy, downgraded to stale: even the corrected `-L "blas|ortho"` misses four
of the six targets §9.1 names — `options_api_tests` and `linalg_layer_tests` are `util`
(`tests/CMakeLists.txt:136-137`), `syevx_tests`/`lanczos_tests` are `eig` (`:149,:151`). The
readers first called this "no potrf assertion runs at all"; that overstates it, because
`ortho_tests` sweeps `OrthoAlgorithm::Chol2`/`ShiftChol3` through potrf with an orthogonality
residual (`tests/ortho_tests.cc:250-264`, `:88-98`). What the recommended scope really drops is
the **direct** potrf assertions.)*

### W12. `src/extensions/` is eight disjoint device-link object libraries, and §8 names no CMake work

**Spec:** §8 steps 1.1, 1.2 (`src/extensions/potrf_cta_device.hh`, `potrf_cta.cc`) and 2.1
(`potrf_blocked.cc`), each listed as a file to create with no CMake edit named — even though
steps 1.5 and 1.6 do name `tests/CMakeLists.txt` and `benchmarks/CMakeLists.txt`.

**Reality:** `src/extensions/CMakeLists.txt` declares **eight** disjoint source lists at `:1, 13,
29, 41, 60, 75, 79, 87` and `target_sources`'s them into eight OBJECT libraries at `:93-100`.
There is no glob. The file's own comment at `:53-57` states the rule: *"The grouping is NOT by
topic, it is by device-code cluster: SYCL device functions are resolved across translation units
within a library, so a source must sit with the sources whose device symbols it calls. Splitting
through a cluster is a hard link error (`ptxas fatal: Unresolved extern function ...`)."*

**Consequence** (corrected from the readers' first draft, which called this silent — the file
itself says at `:57` "never a silent miscompile — so a bad regrouping fails the build
immediately"): both failure modes are **loud** build failures. A `.cc` in the directory but in no
list is never compiled, so the facade's references to its explicit instantiations are undefined
symbols; and `potrf_cta.cc` in `EXTENSIONS_CTA_SOURCES` with `potrf_blocked.cc` in
`EXTENSIONS_FACTORIZATION_SOURCES`, while the blocked driver calls the CTA leaf's device function,
is `ptxas fatal: Unresolved extern function`. **§8 must name the target list, and both potrf
sources must share one cluster.** Note `batchlas_extensions_cta_obj` is also the one library
configured `NO_CPU_TARGETS` (`src/CMakeLists.txt:69`), which changes what a NETLIB instantiation
can mean there.

### W13. `alpha` is the **first** member of `TrsmOptions`, not the fourth

**Spec:** spec:329 — "`alpha` in `TrsmOptions` sits in position 4 to match `trmm`,
`trsm.hh:87-95`."

**Reality:** three errors in one sentence. (a) `TrsmOptions` is
`{ T alpha; Side side; Uplo uplo; Transpose trans; Diag diag; }` at `options.hh:257-264` — alpha
is member **1**; position 4 is `trans`. (b) The "alpha sits in position 4" comment is at
`functions/trsm.hh:95-99` (the readers first cited `:99-103`; grep confirms the single occurrence
at `:95`), and it is correct **about the positional entry point** — `trsm<B,T>(ctx, A, B, alpha,
side, uplo, transA, diag)` at `:100-108`. The error is confined to attributing that position to
the option struct. (c) `backend::trsm_vendor` takes alpha **last** (`:153-161`), with `= delete`
tombstones for the old positional order at `:128-132` and `:134-138`.

**Consequence:** designated initialisers must follow declaration order, so
`{.side = …, .alpha = …}` is a compile error. Read `src/extensions/ortho.cc:202-205` for the
working spelling.

---

## The measurement gate — the spec's first gate is not executable, and is missing two of its three conditions

§8's Phase-1 gate (spec:478) reads: "`-Rpass-analysis=kernel-resource-usage` shows **no spill** of
`x[]`/`acc[]`; SASS shows `sqrt` + `div`, not `rsqrt.approx`." §2.4 (spec:104) says the same.

**(a) The flag exists nowhere in this tree except this spec.** Grep over `cmake/`, `scripts/` and
every other `.md`: `-Rpass-analysis` occurs at exactly two places, `WP4_POTRF_SPEC.md:104` and
`:478`. Device code here is AOT-compiled to an sm_89 cubin at the **shared-library device link**,
not per TU: `build/src/CMakeFiles/batchlas_sycl.dir/link.txt` carries `-fsycl-targets=nvidia_gpu_sm_89,native_cpu`
and `-Xsycl-target-backend=nvptx64-nvidia-cuda`, and `cmake/BatchLASDetectSYCL.cmake:528` calls it
"link-time device compilation" (the `-Xcuda-ptxas -v` block at `:544-552` is gated on
`BATCHLAS_KEEP_CUDA_INTERMEDIATES`). `scripts/register_probe.sh:4-8` states the consequence:
*"`-Xcuda-ptxas -v` on a compile is reported \"argument unused\" and produces nothing."* **A
per-TU log with no "spill" line reads as "no spill".** That is a phantom measurement.

**Working recipe** (added by WP3 at `9e62e2e`, after `13ee56f`):
`scripts/register_probe.sh <out.log> [grep-pattern]` — it `cd`s to `build/src`, replays
`CMakeFiles/batchlas_sycl.dir/link.txt` verbatim with a second
`-Xsycl-target-backend=nvptx64-nvidia-cuda -Xcuda-ptxas -v` pair appended and `-o` redirected
(`:35-46`), then reports entry-function count, kernels with non-zero spill, and per-kernel
`Used N registers` by mangled name. Each kernel appears twice (`<name>` and `<name>_with_offset`);
take the max. Make the link-time budget a **delta** against the recorded baseline of **43.4 s /
376 entry functions** (`:31-32`), not an absolute.

**(b) A spill-only gate passes the exact configuration §3.4 proposes.**
`WP3_TRSM_SPEC_CORRECTIONS.md:160`, after the gate was actually run:
**"The gate is: `stack frame == 0` AND `0 bytes spill stores/loads` AND `registers × WG <= 65536`."**
Its measured table at `:142-147` — for a kernel with the same shape, a per-work-item `T x[N]`:

| type | N | registers | stack frame | spill |
|---|---|---|---|---|
| float | 8 / 16 / 32 | 42 / 76 / 114 | 0 | 0 |
| float | **64** | 119 | **256 B** | 0 |
| double | 8 / 16 / 32 | 59 / 100 / 153 | 0 | 0 |
| double | **64** | 145 | **512 B** | 0 |

256 B is 64 floats: that is `x[]` itself, in local memory rather than registers. ptxas calls it a
*frame*, not a spill, because the array was never in registers to be spilled out of — and
register residency is the entire thesis. `experiments/wp3_s12/README.md:63-67` and
`experiments/wp3_s13/README.md:84-86` record the same, and step 13 **reverted** it.

**§3.4's float `NB_o = 64` is `x[64]` floats = 256 B — byte-for-byte the measured non-resident
case.** WP3 predicted float 64 and shipped 32 (`WP3_TRSM_SPEC_CORRECTIONS.md:162`). Put float
`NB_o = 64` and the CTA kernel's float `NB = 32` in the falsification set before accepting either.

**(c) The tree contradicts itself about the gate, and the readers split on it.**
`scripts/register_probe.sh:14-22` still states the **two**-condition gate and says explicitly
*"Stack frame is the WRONG gate"* (its evidence: 220 of 376 entry functions carry a benign
non-zero frame with zero spills). `WP3_TRSM_SPEC_CORRECTIONS.md:136-160` is the later ✱ correction
that overturned exactly that sentence. The distinction is kernel-specific: in the GEMM kernels a
frame is benign; in an accumulator-array kernel the only thing that can be on the stack **is** the
accumulator. Use the three-condition gate for potrf; treat `register_probe.sh`'s header as stale.

**(d) The `regs × WG` third condition cannot be what fails here.** Max WG is 256 (spec:196,
spec:227) at 64–96 regs = 16 384–24 576 of 65 536; `src/sycl/gemm_kernels.cc:720-735` records 208
registers (double 8×8) and 247 (`complex<float>`), both spill-free, as observed values. The
exposure is entirely conditions one and two.

**(e) §7 item 3's contingency has nothing to fire on today.** Grep over `CMakeLists.txt`,
`cmake/`, `CMakePresets.json` and every `*.cmake` for `fast-math|ffast|ffp-contract|ffp-model|Ofast`
returns **zero hits**. Do not let that become a reason to skip the PTX/SASS inspection — DPC++
contracts by default without any flag. (And `1/sqrt(FLT_TRUE_MIN)` is 2.67e22, not spec:443's
1.5e22; the conclusion `< FLT_MAX` is untouched.)

---

## Stale findings — out of date but harmless

**Dispatch and plumbing**

- **§8 step 0.1 is already done.** All three backends own `*_vendor` forms only:
  `cusolver.cc:27,48` (instantiations `:500,:508`), `rocsolver.cc:19,38` (`:440,:442`),
  `netlib_lapack.cc:1003,1543` (macros `:1584-1585`), with `sig::potrf_vendor`/
  `sig::potrf_vendor_buffer_size` already declared at `potrf.hh:26-41`. Delete the step.
- **§6's reuse row 2 is fine, row 1 is dead.** `tuning_env_override` is at `tuning_params.hh:33-41`
  with its query/call desynchronisation HAZARD documented immediately above at `:27-32` — the same
  hazard §5.5 flags for the route variable. Row 1's cited `ormqr.hh:161-176` is now
  `ormqr_vendor_buffer_size_or_throw` (`:158-166`) and two namespace opens; ormqr's routing is
  `RouteTable<Op::ormqr, T>` at `route_ormqr.hh:51`, built by `ormqr_route` at `ormqr.hh:202-210`.
- **§8 step 3.1's destination is wrong.** The syev routing grid, with its measured tables,
  MEASUREMENT CORRECTION note and SCOPE caveats, is in `include/batchlas/blas/functions/syev.hh`
  (~`:176-412`), **not** `tuning_params.hh` — grep for `route|ROUTE|provider` in that header
  returns zero lines. The current home for a measured window is `RouteTable::preferred()`
  (`route_trsm.hh:188-325`, each clause citing its cells). `tuning_params.hh` is right only for
  `nb`/`L`/`G` (§8 1.3); putting a routing threshold there mixes a route decision into a
  workspace-sizing input.
- **§8 step 1.2's `× {CUDA, ROCM, NETLIB}` instantiation is unnecessary.**
  `BATCHLAS_FOR_EACH_SCALAR_TYPE_1` exists (`src/util/template-instantiations.hh:50`) and the
  ×backend pattern is genuine for older extensions (`trmm.cc:117`, `syev_cta_fused.cc:618`,
  `gesvdj_cta.cc:1218`), but WP3's native kernel — the closest analogue — instantiates **per
  scalar type only**, no `Backend` parameter (`trsm_native.cc:820-838, 902-905, 912-925, 927-930`).
  A 3× multiplication of a device-compiled family with no vendor dependency, in a build
  `src/sycl/CMakeLists.txt:1-3` documents as device-link-bound.
- **spec:501's citation is dead.** `tests/CMakeLists.txt:243` is a comment; there are no literal
  `add_test` NAMEs in the file. Names come from `TEST_TARGETS` (`:18-75`), turned into tests by
  one generated `add_test(NAME ${test_name} COMMAND ${test_name})` at `:249`.
- **potrf.hh line drift.** spec:306's `:28-49` lands mostly on the *vendor* aliases (`:26-41`,
  annotated "a vendor parameter list can differ from the public one"); the public declarations are
  `:44-47` and `:60-65`, the 4-arity forwarder `:72-78`, the `Matrix<T>` overloads `:80-102`.
  spec:314's `potrf.hh:40-43` is `Uplo);` / `} // namespace sig` / blanks — the sentence it means
  is at `:56-59`, and its mechanism at `src/linalg-impl.hh:757-761`. In `options.hh` the option
  overloads are `:539-568` (exact) and the guard spans `:570-595` (also defensible: `:570` opens
  `namespace detail`, `:588` is the enum, `:592`/`:595` the deletes).
- **§5.2's "exceptions: none, ever" (spec:373) is incomplete.** The *deducing* option overloads
  throw `std::invalid_argument` for a non-square `A` (`options.hh:129-131`) and a short non-empty
  `info` span (`:170-172`), and the facade throws `NoRouteError` vendor-free
  (`factorization.cc:165-167`); `tests/options_api_tests.cc:584,732-733` are live `EXPECT_THROW`s.
  §5.2's own row, scoped to non-PD input, is literally true.
- **§5.6's info table omits negative values.** netlib writes LAPACKE's return status straight
  through (`netlib_lapack.cc:1023,1030`), which is negative for an illegal argument. §5.9's
  zero/non-zero contract absorbs it (`tests/options_api_tests.cc:464-466`, quoted correctly);
  §5.6's table does not mention it.

**Primitives and citations**

- **`info_target` moved and changed a parameter type.** Now `src/linalg-impl.hh:767-771`, last
  parameter `size_t count`, not `int count` (spec:411 says `:714-718`). Two facts §5.6 omits:
  the fallback is `>= count`, so a **short non-empty** span behaves exactly like an empty one
  here, silently and by design (`:763-766`); and netlib has **no** pool fallback at all
  (`netlib_lapack.cc:1012-1016`, `want_info = info.size() >= batch_size()`), where cuSOLVER
  (`:59,:63`) and rocSOLVER (`:49`) do.
- **`sytrd_blocked.cc:242-285` calls `her2k`, not `herk`.** The function is
  `update_vw_lower_small` (`:239`) and its primitive is `device::her2k` (`:254-256, :278-280`) —
  a different Tag, a different dispatcher (`dispatch_rank2k`, `group_blas_rankk.hh:393`), a
  different generic body. **`device::herk` has zero callers in `src/`**; its only in-tree
  exercises are `tests/device_blas_tests.cc:1569-1573,1786-1790`. The oracle still builds (see
  survives), but an implementer told "already in production" would be its first production caller.
- **`rankk.hh` does not exist.** spec:405 and spec:445 cite `rankk.hh:71-79`; the file is
  `include/batchlas/blas/device/detail/group_blas_rankk.hh` and `:73-79` is the right code (the
  hermitian `value = T(value.real(), 0)` inside `generic::rankk`). Likewise spec:410's
  `trmm.hh:118,132` / `symm.hh:81,92` are the `group_blas_*` device headers — files named
  `functions/trmm.hh` and `functions/symm.hh` **do** exist and contain no `select_from_group`, so
  the shorthand sends a grep to the wrong file rather than to nothing.
- **§6's namespace hazard** (pre-empting `WP3_TRSM_SPEC_CORRECTIONS.md` finding 5): §6 gives no
  namespaces and its rows straddle two. `make_group_launch_info`, `herk`,
  `herk_workspace_elements` are `batchlas::device` (opened `group_blas_common.hh:19`);
  `triangular_storage_contains`, `rankk_rhs_transform`, `NdItemLike` are
  `batchlas::device::detail` (`:179`–`:1047`). Transcribing one qualification onto the other does
  not compile.
- **`KernelMatrixView`'s SLM precedent drifted.** The constructor is exact at `matrix.hh:243-247`
  and takes a sixth defaulted `int batch_size = 1`, so the 5-argument spelling compiles. But
  `sytrd_blocked.cc:540-548` is a `dotc`/`axpy`/barrier; the actual SLM-view constructions are
  `:535-536` and `:551-554`. Heed `matrix.hh:234-242`: a stride left at 0 with `batch_size > 1`
  addresses wrongly.
- **§2.6's syrk/herk citations have all drifted.** `cublas.cc:745-750` (syrk host loop) →
  `:722-726`; `:403-409` (herk n≤768) → `:382-388`; `:538-545` (arena lease) → `:516-521`;
  `:137` (strided-batched) → `:118-128`. And the float OR at `syrk_custom_dispatch.cc:109-197`
  has **two clauses the spec omits** — `batch * triangular_tile_count(n) >= 160` (`:117`) and
  `n >= 16` plus `tiled_work >= 8` (`:136,:143`) — plus a leading branch returning true for any
  non-Auto forced route (`:176-178`). Both omissions narrow the custom window further, so §2.6's
  conclusion holds *a fortiori*. `BATCHLAS_SYRK_VARIANT` does still resolve
  (`legacy_variable_for(Op::syrk)`, `route_env.hh:113`) — unlike its potrf sibling.
- **`src/sycl/device_scalar.hh` already supplies §6's "new device code" item 1.** WP3 lifted the
  Annex-G-free complex arithmetic into a shared header that did not exist at `13ee56f`: `Cx<R>`,
  `dev_is_complex_v` (`:60`), `dev_conj` (`:108-116`), `dev_mul` (`:118-126`, "Written out for the
  same reason `fma_acc` is: keep Annex-G out of it"), `dev_recip`/`dev_div` (`:160-189`),
  `dev_isfinite` (`:140-150`), `fma_acc` (`:81-96`), with the header recording verified PTX
  containing "zero `__mulsc3`, zero `__muldc3`, zero `call.uni`" (`:3-23`). `mul_conj_b(a,b)` is
  `dev_mul(a, dev_conj(b))`. `real_part` has no shared spelling — it exists privately at least
  eight times (`ritz_values.cc:67`, `syev_jacobi_cta.cc:85`, `syev_cta_fused.cc:80`,
  `ortho.cc:191`, `sytrd_sb2st.cc:97`, `lanczos.cc:46`, `band_reduction.cc:41`,
  `sytrd_sb2st_cta.cc:98`); `div_by_real` has none. **Caveat if adopted:** `device_scalar.hh` is a
  pointer-boundary re-typing (the launcher `reinterpret_cast`s `std::complex<R>*` to `Cx<R>*`), so
  it commits the kernel body to `Cx<R>` rather than `T` — a design decision, not a drop-in.
- **WP3's trsm has an exported surface and a large private one.** Reusable:
  `trsm_cta_max_n<T>()` (`trsm_native.hh:58-59`, **32 for all four types**),
  `trsm_blocked_available<T>()` (`:65-66`, true for all four),
  `trsm_native_v1_dispatch<T>` (`:72-80`), `trsm_native_blocked<T>` (`:116-125`), and the
  `TrsmTrailingGemm<T>` injection seam (`:105-111`). **Not** reusable: every device helper and
  structural lambda, because `namespace {` opens at `trsm_native.cc:64` and closes at `:228`
  (`canonicalise` `:83`, `smallest_bucket_ge` `:110`, `trsm_max_bucket` `:141`, `tri_idx` `:148`,
  `trsm_stage_rows` `:186`, `trsm_stage_left` `:213`, `finite_recip` `:219`, `class TrsmCtaKernel`
  `:226`), and the blocked driver's `sub()`/`stored_off()`/`apply_update()` structure is local to
  `trsm_native_blocked` (`:~690-804`). Two consequences: **(i)** potrf's panel solve
  (`L21 = A21 L11⁻ᴴ`) is a `Side::Right, Uplo::Lower, ConjTrans` trsm, which the routed `trsm`
  now serves natively for all four types with `preferred()` measured at **167 of 168 cells
  winning** (`route_trsm.hh:212`) — measure it before writing `PotrfPanelSolveKernel`, and note
  trsm already has the `Side::Left`/`Upper` half §3.4 defers to Phase 3. **(ii)**
  `trsm_cta_max_n<T>() == 32` for *all four types*, measured by register probe, so any potrf CTA
  capacity above ~32 that §4 predicts must be re-probed, not assumed. *(Do not cite
  `trsm_native.cc:906-911`, which still says "V2 does not exist yet, for any type" — it is a stale
  leftover contradicted by the instantiations immediately below it.)*
- **`TrsmTrailingGemm` is the precedent for the injection seam §2.6 needs and §8 does not name** —
  how a kernel TU stays free of the dispatch layer while the facade injects the routed gemm
  (`level3.cc:186-231`, `trsm_native.hh:105-111`).

**§2.6's GEMM justification (S1)**

spec:158's "every launch is `cublasGemmStridedBatchedEx` (`cublas.cc:137`) and therefore
genuinely batched for all four scalar types" is **false since WP2**, though the *instruction* it
justifies is right (see survives). Three facts:

1. `legacy_unset_default` returns `{Auto, Auto}` for every op (`route_env.hh:145-148`);
   `kGemmOrder` puts `{Native, RegisterTiled}` first (`route_gemm.hh:48-51`); `resolve_route`
   takes the first supported **and** preferred (`route_resolve.hh:57-58`); and
   `RouteTable<Op::gemm, double>::preferred` needs only `is_gpu` (`:91`), `!heterogeneous`
   (`:99`), `batch >= 64` (`:122`), then `return s.k >= 2` (`:206`) — **no transpose test**. The
   vendor path consults it too: `cublas.cc:155` calls `gemm_use_sycl_custom`
   (`gemm_variant.hh:239-248`) and dispatches to `sycl_gemm::gemm_custom` at `:156`.
   `cublasGemmStridedBatchedEx` is at `cublas.cc:118`, and `:106-107` takes a plain `cublasGemmEx`
   at `batch_size() <= 1`.
2. Inside `gemm_custom`, **any transposed operand short-circuits at
   `src/sycl/gemm_kernels.cc:460-472`** to `max_dim <= 32 ? Direct : Tiled16`, before the
   wide-scalar ladder — so the 64×64 wide tile at `:583-587` is structurally unreachable for
   potrf's `transB = ConjTrans` trailing update.
3. Every operand is a **sub-view carrying the parent `ld`**, and the native fast paths are gated
   on `is_contiguous_dense_matrix` (`ld == rows && stride == ld*cols`,
   `src/sycl/gemm/register_tiled_common.hh:74-77`, a hard conjunct at `:92-94`) or `ld % 4` /
   `ld % VecLen` alignment (`register_128x128.hh:84-90`, `register_64x64_k16_wide.hh:201-207`).
   WP3 measured the effect on the identical operand shapes: **0.86–0.98× at `ld == rows` versus
   0.43–0.62× at the real `ld`**, cuBLAS barely moving (`trsm_native.hh:87-97`,
   `level3.cc:198-216`). `gemm_kernels.cc:564-582`'s own demand table says of this population —
   large m,n / small k / transposed — "for that population this gate cannot fire at any problem
   size". Note `OpShape` carries **no leading dimension** (`route.hh:227-241`), so the router
   cannot distinguish the case.

**Consequence:** restate spec:158 as "the routed gemm, whatever it resolves to", and record the
strided-`ld` collapse (WP3's open step 17) as a known cost the blocked driver inherits. Do **not**
fix it by calling `sycl_gemm::gemm_custom` directly — that is WP3 step 16's defect in reverse.
§10.3's grid does benchmark potrf itself over `(n, batch, type, uplo)` (spec:588-601), which
issues the real sub-views by construction, so it *can* see this; a square-matrix GEMM benchmark
structurally cannot.

**Numerics and arithmetic slips**

- **§2.6's "12.5 % redundant arithmetic" is 25 %** on its own waste-over-useful basis (the same
  basis on which a naive gemm-into-scratch is priced at 100 %). Per panel the discard is `W²/2`
  cells over `ceil(m2/W)` panels = `m2·W/2` against a useful `m2²/2`, i.e. `W/m2 = 128/512 = 25 %`.
  (Over total issued work it is `W/(m2+W) = 20 %`; neither reading gives 12.5 %.) The spec
  contradicts itself: spec:609's mitigation (a) says "keeps the waste at `W/m2`". `W = 128` was
  chosen against the wrong number and the cost of a larger `W` grows twice as fast as stated. The
  adjacent scratch figure `128²·512·8 = 67,108,864 B` is right.
- **The diagonal-block fold takes no alpha.** `fold_symmetric_product_into_triangle` computes
  `C = product + beta·C` (`symmetric_product_fold.hh:29-34, :68`). spec:155 states `alpha = −1,
  beta = 1` for the sub-diagonal rectangle; spec:156 states **neither** for the diagonal block, so
  the subtraction has to come from an unstated GEMM `alpha = T(-1), beta = T(0)`. *(Readers split
  on the failure: one said copying step 1 gives `A22 += L21 L21ᴴ`; the refuter is right that
  copying `(alpha=-1, beta=1)` gives the correct sign plus a garbage read of uninitialised
  scratch, and only `alpha=+1` produces the `+=`. Either way, say it explicitly.)* In-tree
  precedent points the right way: the herk path does `gemm_vendor(..., alpha, T(0), ...)` into
  `product` then folds with `beta` (`cublas.cc:530`).
- **§7 item 5's "forced real at three points" covers only the CTA kernel.** Phase 2 forces it at
  none: the sub-diagonal rectangle never touches the diagonal, and the fold has no real-part
  projection (`symmetric_product_fold.hh:68`), unlike `accumulate_hermitian`, which documents "the
  diagonal is real on exit" (`cublas.cc:404-406`) and is a function the blocked driver does not
  call. **Downgraded from wrong-edit** because every *global* diagonal reaches a `sqrt` only
  through the leaf, and §2.1's load transform (spec:23-25) loads the diagonal as
  `T(real(A(c,c)), 0)`, discarding the residue before use; `L21` entries are strictly
  sub-diagonal and never square-rooted. (The residue is exactly zero absent FMA contraction, since
  `a·conj(a)` cancels term-by-term; contraction is what breaks it.) What remains: §7's enumeration
  is incomplete for Phase 2 and **T7 should cover a blocked-sized complex case**.
- **§2.3's scale branch is unguarded.** spec:64-65's `else if (lane > k) d[k] = d[k] * r;` has no
  `lane < ib`, unlike the publish (`:66`) and the update (`:71`), so lanes `ib..31` read and write
  uninitialised registers every `k`. Never published, never read, so numerically harmless — but it
  is UB in the exact spot [FIX-A1.4] (spec:77) argues unguarded lanes are the bug being fixed.
- **§3.2's worked table has one wrong row.** spec:204's `n=48 → 21` should be **36**
  (`m2_0 = 32`, `Rt_0 = 8`, `8·9/2 = 36`); 21 would need `Rt_0 = 6`, i.e. `n ∈ 37..40`. The other
  three re-derive exactly, and `L = 32` either way — but this is the only worked check of the
  [FIX-A2.1] correction.
- **Three §4.2 occupancy cells are off by one**, each by dropping the `NB·sizeof(real_t) + 64`
  their own formula includes: at 25600, `double`/`complex<float>` are **55**, not 56; at 12800,
  `float` is **55**, not 56. The 45056 fit-ceiling row and the 17066 row are correct in all cells.
- **spec:286's `~4 register-limited` for `n=64` is 12**, per spec:284's own 80 regs/thread:
  `floor(65536/(80·64)) = 12`, so SLM binds at 6 blocks = **384 threads/SM**, not 256. The other
  two rows re-derive correctly. Understating this by 1.5× is what invites a redesign.
- **spec:177's `complex<double>` `acc` regs is 16, not 8** (`TS²·sizeof(T)/4 = 2·2·16/4`);
  spec:179's own peak formula already gives 32 for that row.
- **spec:179 says a 32-register budget, spec:238 says "the same 64-register budget as §3.1".**
  §3.1 never states 64. At 32 the `x[NB]` caps are exactly spec:174-177's ladder; at 64 they are
  exactly spec:238's `NB_o` ladder. This contradiction is how float `NB_o = 64` entered the design
  unexamined — see the measurement-gate section.
- **spec:248's `n ≳ 56` for the (P1) idle window is float/double/`complex<float>` only.** With
  `complex<double>`'s `TS=2, NB=8` the threshold is `n > 28`, so WorkGroup scope and its idle
  sub-group cover **29..52 of that type's entire 1..52 resident range**. §10's profiling plan,
  aimed at `n ≳ 56`, samples the wrong region for cdouble.
- **spec:267's "every existing device-BLAS sizing decision uses it" is false.** The cmake half is
  right (`BatchLASDetectSYCL.cmake:57-68`, subtraction at `:61`, plus an unmentioned 16384 floor
  at `:62-64` that is inert here) and `group_blas_subgroup_common.hh:60-61` is exact — but all 11
  uses of the budget are inside that one header. Three other runtime SLM decisions use the **raw**
  `LOCAL_MEM_SIZE` with no reserve: `sytrd_blocked.cc:726-733`, `tridiag_solver.cc:160,255`,
  `steqr_legacy.cc:379`; and `gesvdj_cta.cc:221-222` subtracts only its own `kPairTabBytes`. There
  is no tree-wide 4096 convention.
- **spec:206's "unlike every comparable kernel here" is false.** The three named kernels are exact
  (`syev_cta_fused.cc:185`, `gesvdj_cta.cc:297`, `sytrd_sb2st_cta.cc:403`), but
  `sytrd_cta.cc:95-97` launches with **no** attribute and relies on the host-side enumeration at
  `:319-333` — the very kernel spec:214 cites as its early-return precedent. The recommendation
  (carry the attribute **and** enumerate on the host) is right; only the quantifier is wrong.
- **spec:5's `device_limits.hh:23,28-29`:** the 45056 literal wraps to `:30`. Cite
  `subgroup_workspace_budget_bytes()` (`:34-36`) instead — the file is generated and moves on any
  reconfigure. *(One reader filed this as stale; the refuter is right that it is a formatting
  artefact and the values are where the spec says. The substantive point about that header not
  describing this device belongs entirely to W1.)*

**Tests**

- **`cond_tests` no longer touches potrf at all.** Grep returns only comment lines
  `tests/cond_tests.cc:370,374,375` (the block runs `:367-387`). The generator's default is
  `OrthoAlgorithm::CGS2` — `include/batchlas/blas/extra.hh:141`, with `:130-133` saying *"CGS2 …
  uses no potrf at all, so it cannot be caught by the unchecked info code."* **One of the six
  targets §9.1 offers as existing potrf coverage is empty.** *(Readers disagreed: one cited
  `cond_tests.cc:381-382`'s "The default is now Householder"; that comment is itself stale
  relative to `extra.hh:141`. `random_cond.cc:237-246` forces Householder only for
  `Backend::NETLIB`.)* Recreating the coverage needs an explicit `OrthoAlgorithm::Chol2`.
- **§5.8's justification is wrong even though its conclusion holds.**
  `CondTest.RandomMatrixGeneratorIsAlwaysFinite` (`cond_tests.cc:388-426`) **does** assert
  NaN-ness: `std::isfinite` at `:413`, `EXPECT_EQ(non_finite_items, 0)` at `:422`. It is
  unaffected because potrf is off its path (above), not because it "asserts orthogonality". Do not
  lean on §5.8 when scoping T9. Also spec:447's quoted seed sentence is at `cond_tests.cc:384`,
  **outside** its cited `:365-378`; the block is `:367-388`.
- **§5.7 rule 1's coverage claim is false.** "The existing `options_api_tests.cc:507-514` passes a
  real span and would not see it" — it would. `:509` constructs
  `UnifiedVector<int32_t> info(batch, kNeverWritten)` with `kNeverWritten = -12345` (`:498`), so
  with no zeroing pre-pass the guard `if (info[b] != 0) return;` trips on the caller's own
  sentinel and both `:512` and `:513` fail loudly. T8 is still worth writing; it is not covering a
  hole. *(Downgraded from wrong-edit: the natural implementation zeroes the local span
  `info_target` returned, which is correct in both branches. Still, say it explicitly — zero
  **`info_target(...)`'s result**, never `info_out`, because a short non-empty span silently
  becomes pool scratch (`linalg-impl.hh:768-770`) and the positional overload `ortho.cc:200` uses
  performs no length check.)*
- **§9.3's T3 constraint is unsatisfiable at two of its own `n`.** spec:537 asks for `G > 1`
  while spec:536's list contains 63 and 65; by spec:189-195 with float `TS=4, nb=16`, `n=63` →
  `Ntiles_0 = 78` → L=64 → **G=1**, and `n=65` → 91 → L=64 → G=1. Forcing `nb=32` does not rescue
  it (`slm_per_matrix = 16068`, `24576/16068 = 1`). *(Downgraded: spec:536 and spec:537 serve
  different purposes — the `n` list is ragged-panel predicate coverage, exercised at any `G` — and
  at `G=1` there is no neighbouring matrix, so the write lands in that matrix's own 16-element
  guard and is caught. Coverage shifts detector rather than vanishing.)* The `n` that do reach
  `G>1` are 9, 15, 17, 31.
- **§9.2's T1 grid violates a written repo rule.** `tests/README.md:54-58`: *"Never combine large
  `n` with large `batch` … Their product is where cost explodes for no added coverage."* T1
  (spec:518) pairs `n=512` with `batch=128` while spec:516 demands label `blas`, not `slow`
  (`tests/CMakeLists.txt:130-131` sets the ~15 s rule, `:155-171` the slow list with the
  `iluk_tests` note at `:157-159`). *(Downgraded: the cost depends on how the residual is formed,
  which spec:518 does not say. The in-tree pattern is a **device** product plus an `O(n²·batch)`
  host comparison — `tests/ortho_tests.cc:85-98` — under which those cells are milliseconds. Split
  it per README:54-58 anyway: large `n` at batch ∈ {1,3}, batch=128 only at `n ≤ 65`.)*
- **§9.1's "runs at `n = 8, batch = 2`" describes only the two info tests.** Other potrf call
  sites in `options_api_tests.cc` run at `n=24/batch=3` (`:246`), `n=20/batch=2` (`:287`),
  `n=32/batch=2` (`:345`), `n=24/batch=2` (`:393`), `n=8/batch=4` (`:579-580`). So `n` reaches 32,
  inside the CTA envelope. The caveat's **conclusion** holds through *batch*, which never exceeds
  4 anywhere in the file, so no measurable `kPotrfCtaMinBatch` selects native there.
- **§9.1's trsm row is half false.** "`Side::Right` and `Diag::Unit` are never tested" — WP3 added
  the full canonical cross product at `tests/trsm_tests.cc:409-429` (both sides × uplo × transpose
  × both diags, per scalar type) plus `:431-462`. The second half survives: the original
  `TrsmOperationsTest` still issues one unbatched call per item (`:169-174`). The spec marks the
  row "relevant to WP3, not here".
- **§9.3's `-UNDEBUG` recipe is a flag without an incantation.** `matrix.hh:135-149` is exact (the
  `operator()` asserts, compiled out by NDEBUG), but NDEBUG arrives from **two** places:
  `CMAKE_CXX_FLAGS_RELWITHDEBINFO = "-O2 -g -DNDEBUG"` (`build/CMakeCache.txt:242`) **and**
  `cmake/BatchLASOptions.cmake:239-243`'s `target_compile_definitions`. The working form is a
  fresh build dir with `-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -UNDEBUG"`; it works only because
  CMake's rule is `$DEFINES $INCLUDES $FLAGS`, so the `-U` in FLAGS lands after both `-D`
  (rationale at `STEDC_MERGE_OPTIMIZATION.md:119-126`). The obvious `target_compile_definitions`
  route cannot undefine it at all.
- **Independent pre-existing defect, reported not because §9.3 needs it but because it is real:**
  `matrix.hh:387, :401, :413` each read `assert("Invalid slice dimensions …")` with no
  `&& false` — a non-null pointer, always true, dead even under `-UNDEBUG`; contrast
  `:1757`'s correct `assert("…" && false)`. *(The readers' consequence for §9.3 is retracted:
  those three guards fire only on a **degenerate** extent (`r_len <= 0`), and the blocked driver
  does not slice a `KernelMatrixView` anyway — spec:160 mandates explicit construction.)*
- **Adding `potrf_tests` needs two edits, not one.** Add to `TEST_TARGETS`
  (`tests/CMakeLists.txt:18-75`) **and** to `BATCHLAS_TEST_LABELS_blas` (`:138-141`); do **not**
  add to `BATCHLAS_SLOW_TESTS` (`:160-167`). `batchlas_add_test_target` (`:185-266`) then supplies
  `add_executable` from `${test_name}.cc` (`:186`), the configure branch (`:218-224`), `add_test`
  (`:249`), the label (`:251-257`) and the `OPENBLAS_CORETYPE` environment (`:259-265`).
  **Hazard:** target without label makes `batchlas_test_component` return `unlabelled` (`:182`),
  and the binary becomes invisible to every `-L` run while still passing in the full run — the
  same silent-false-green family as W11. §9.2's heading already says label `blas`, so this is a
  note, not a correction.
- **Selective testing, as the repo practises it** (`tests/README.md:1-34`): never run the full
  suite on an edit (15–20 min); narrowest scope first — one case via `--gtest_filter`, one binary
  via `ctest -R '^name$'` (README:17-18 warns `-R` is a substring regex, **anchor it**), one
  component via a **single** `-L`, broad-but-quick via `ctest -LE slow`; full `ctest` pre-push
  only. README:14's "38 of 45" is now **53 of 58** measured; bare `ctest` → 58. Runtime levers at
  README:38-48: `BATCHLAS_TEST_BACKEND=CUDA`, `BATCHLAS_TEST_FLOAT_TYPE=float`, both `GTEST_SKIP`
  at runtime (`tests/test_utils.hh:30,:272`).

---

## ✱ Findings the refuter overturned

Two verification-pass findings were wrong and are recorded rather than deleted, because the
sibling document is better for marking its own.

### ✱ "T3 states no oracle and cannot fail on the defect it was designed for"

A reader argued §9.3's flagship test specifies no assertion on the computed factor, so it is a
build-configuration ritual. **The oracle is specified, one section away:** `WP4_POTRF_SPEC.md:441`
states normatively, for the whole test plan, *"Every test asserts a residual norm; none compares
entries."* That is exactly the per-item `‖A − LLᴴ‖_F` the finding proposed as the remedy. The
supporting facts stand — `matrix.hh:135-149` are the only live element-access asserts, and an
in-tile cross-lane write at `Sd(lane,k)` evades both them and an end-of-allocation canary — but
spec:540 already concedes the assert half in as many words, and a residual catches it. **Salvage:**
§9.3 would read better if it restated the residual and the T4 cross-item comparison (spec:521)
locally instead of leaving them at spec:441/:521.

### ✱ "§7's accuracy gate has no enforcement point in §8's Phase 3"

A reader noted §8's Phase-3 table (spec:492-497) lists only "run the grid / flip Auto / Upper /
coverage table" and never the §7 harnesses. **The enforcement point exists at spec:600**, in §10.3:
*"Gate to flip a cell to Auto: `t_native ≤ 0.90 · t_vendor` … **and** the §7 accuracy harnesses
show no regression against `ACC_ORTHO_HOUSEHOLDER`, **and** `ortho_benchmark` shows the win end to
end."* Step 3.2 flips "exactly the measured cells", i.e. cells that cleared that gate, of which
the §7 harness is an explicit conjunct. The tooling half of the finding is true and harmless:
`ACC_ORTHO_CHOL2/CHOLESKY/SHIFTCHOL3/HOUSEHOLDER` are defined at
`benchmarks/orthogonality_miniacc.cc:111-128` and registered at `:147-150`;
`--impl/--samples/--log10-cond-max` parse at `benchmarks/orthogonality_accuracy.cc:60,79,83` with
`ortho_all` as a group at `:125-131`; both targets are `EXCLUDE_FROM_ALL`
(`benchmarks/CMakeLists.txt:4,51,77`).

**Also downgraded, and worth knowing:** the §8 step 1.5 test-label claim was reported as a
wrong-edit and is not one — §9.2's heading already names the label. The real test-plan defect in
that territory is spec:512 (W11). And the "a chooser that calls `A.data_ptrs(ctx)` would fault on
the sizing path" hazard is **real but not a spec defect**: the mechanism checks out
(`ortho.cc:51-54,78,423-425`; `mempool.hh:146,179-184`; `matrix.cc:2365-2383` launches and waits),
but §5.5 never instructs anyone to do it and §5.2 (spec:313) affirmatively states `data_ptrs` is
never consulted. Keep it as a hazard note next to the query, not as a correction.

---

## Open questions, each with what settles it

1. **What is the real per-work-group SLM ceiling for a `local_accessor`?** Everything in §4
   depends on it (W1). What settles it: run the shipping `gesvdj_cta` double/C=64 path (it already
   allocates 71,744 B) under `ncu` and read the static shared-memory figure; then a one-off kernel
   allocating 97,280 B on this box. If 97,280 launches, `potrf_cta_max_n<T>()` is
   155/109/109/77 and §10.3's `n` list must be re-cut. If it does not, find the actual cap — it is
   **not** 49,152, because 71,744 already runs.
2. **Is the larger CTA capacity worth having, given occupancy?** At 97,280 B the CTA kernel runs
   at **1 block/SM** (`floor(102400/97280)`). `gesvdj_cta.cc:1011-1016` records exactly this
   trade-off and calls occupancy, not the hard cap, the binding constraint. What settles it: the
   §10.3 grid extended to the larger `n`, at saturation, against cuSOLVER — the answer may be that
   the *fit* ceiling rises while the *preferred* window does not.
3. **How is `has_sg32` expressed in the shape?** There is no route-table precedent for the
   enumerated form; syev uses the weaker `s.max_sub_group >= 32` (`syev.hh:837`), and the
   enumeration exists only host-side (`sytrd_cta.cc:319-333`). What settles it: a decision — add a
   `bool has_sg32` field to `PotrfShape` fed from `sub_group_sizes` by the builder (recommended,
   and what §10.1 correctly asks for), or accept syev's test and document the weakening.
4. **Which register gate is authoritative?** `scripts/register_probe.sh:14-22` says two conditions
   and "stack frame is the WRONG gate"; `WP3_TRSM_SPEC_CORRECTIONS.md:160` says three and
   overturns that sentence. What settles it: nothing new — run the three-condition gate on the
   potrf CTA kernel, and edit the script header, which is stale.
5. **Does the panel solve need a new kernel at all?** WP3's routed `trsm(Side::Right, Uplo::Lower,
   ConjTrans, NonUnit)` already serves the `m2 × ib` shape for all four types, 167/168 cells
   winning (`route_trsm.hh:212`). What settles it: A/B the routed trsm against
   `PotrfPanelSolveKernel` on the real panel shapes *at the parent `ld`*, before writing the
   kernel. Note trsm's float/`Side::Right` clause is `s.batch >= 128 || order <= 32`
   (`route_trsm.hh:304`), not the blanket `batch >= 8`.
6. **What does the double trailing update actually route to, and what does it cost at the real
   `ld`?** (S1.) What settles it: run the trailing update through `route_diff.sh` with
   `BATCHLAS_GEMM_ROUTE` unset, then measure the same six shapes at `ld == rows` and at the parent
   `ld`. WP3's numbers for the identical shapes are 0.86–0.98× vs 0.43–0.62×.
7. **Is `off[]` one copy per work-group or one per matrix?** (W9.) Per-work-group is legal (`n`
   and `nb` are work-group-uniform) and removes `G−1` copies from `slm_per_matrix`; per-matrix
   keeps the packing arithmetic uniform. A decision, then one SLM-formula edit and one barrier.
8. **Who populates `heterogeneous_batch`?** The field exists (`route.hh:236`) with exactly one
   writer in the tree (`gemm_variant.hh:209`); trsm's own builder never sets it, so trsm's gate is
   decorative today. What settles it: a decision — set it in `potrf_route`'s builder (and, since
   trsm has the same hole, probably in `trsm_route.hh:40-56` at the same time).
9. **Does Phase 2 need a real-part projection in the fold for complex?** What settles it: a
   blocked-sized complex T7 that reads `imag(diag(A22))` after one panel. If §2.1's load absorbs
   it as argued, say so in §7 and extend T7; if FMA contraction makes it grow across panels, add
   the fourth forcing point.
10. **What is `W` worth, recomputed?** The diagonal-block decomposition wastes `W/m2 = 25 %`, not
    12.5 %, so the cost of a larger `W` grows twice as fast as §2.6 assumes. What settles it:
    re-run the §2.6 trade-off at 25 % before fixing `W = 128`.

---

## Revised implementation order

`[J]`/`[M]` sizing from §8 is retained where the step survives. Steps marked **new** do not exist
in §8.

| step | what | status |
|---|---|---|
| 0.0 | Repoint `src/backends/cusolver.cc:56` at `backend::potrf_vendor_buffer_size<B,T>` | **new — do first.** W8(b): the vendor path currently sizes its own cuSOLVER workspace from the public, soon-to-be-routed query |
| 0.1 | rename backends to `backend::potrf_vendor*` | **already done by WP0 S5** (`cusolver.cc:27,48`; `rocsolver.cc:19,38`; `netlib_lapack.cc:1003,1543`) — delete from the plan |
| 0.2 | Measure the real SLM ceiling and re-derive `potrf_cta_max_n<T>()` | **new — blocks §4 and §10.** W1, Open questions 1–2 |
| 0.3 | `scripts/register_probe.sh` baseline (43.4 s / 376 entry functions), before any kernel | **new.** Every later register claim is a delta against this |
| 0.4 | new `include/batchlas/blas/dispatch/route_potrf.hh`: `PotrfShape : OpShape` (`+ cta_max_n`, `blocked_available`, `has_sg32`), `kPotrfOrder[] = {{Native,CTA},{Native,Blocked},{Vendor,Auto}}`, `order_begin/order_end`, `RouteTable<Op::potrf,T>` — **pure**, `supports()` = correctness only (incl. `m==n`, `heterogeneous_batch`, `blocked_available`, `has_sg32`), `preferred()` all-false | replaces §8 0.2. W2, W3, W6, W7 |
| 0.5 | new `src/backends/potrf_route.hh`: shape builder (fills `is_gpu`, `cta_max_n`, `blocked_available`, `has_sg32`, `heterogeneous_batch`) + `parse_route_env(Op::potrf)` + `legacy_unset_default` | replaces §8 0.3 — **not** `src/linalg-impl.hh`, which has no potrf. W5, W7 |
| 0.6 | Hook **both** `potrf` (`factorization.cc:159-171`) and `potrf_buffer_size` (`:173-183`) **above** the `if constexpr` at `:165`/`:177`, passing `/*vendor_available=*/dispatch::solver_vendor_available<B>`; hoist validation into the facade | replaces §8 0.3's second half. W5, W6(d), W8(a) |
| 0.7 | Phase-0 gate: `ctest -R "options_api_tests|linalg_layer_tests|ortho_tests"` green, zero behaviour change | **works as written** (spec:465, measured: 3 tests) |
| 1.1 | `src/extensions/potrf_cta_device.hh` — consider `#include "src/sycl/device_scalar.hh"` for `dev_mul`/`dev_conj` instead of a sixth private copy | §8 1.1, plus S(device_scalar) |
| 1.2 | `src/extensions/potrf_cta.cc` + **name the `src/extensions/CMakeLists.txt` source list**; both potrf sources must share one device-code cluster | §8 1.2 + **W12** |
| 1.3 | `off[]`: decide per-matrix vs per-work-group, add its term to `slm_per_matrix`, give it a writer and a barrier | **new — W9.** Blocks any SLM sizing |
| 1.4 | Resolve the §3.4/§3.2 scope contradiction (float `NB_o=64` → L=64 → G=1 → WorkGroup) | **new — W10.** Blocks the leaf launch |
| 1.5 | ~~`potrf_supports_cta` reads the env~~ | **deleted — W7.** The table is pure; the env read is step 0.5 |
| 1.6 | `tests/potrf_tests.cc` + `TEST_TARGETS` **and** `BATCHLAS_TEST_LABELS_blas`; pin with `BATCHLAS_POTRF_ROUTE=cta` or with an explicit `Route` at the pure layer per `tests/route_vocabulary_tests.cc:52-60,392` | §8 1.5, corrected by W4, W11 |
| 1.7 | Phase-1 gate: T1–T9 green **with the route actually pinned**; `scripts/register_probe.sh` shows `stack frame == 0` AND `0 spill` AND `regs × WG ≤ 65536`; float `NB_o=64` and float `NB=32` in the falsification set; PTX shows `sqrt`+`div` | replaces spec:478. W4, measurement-gate section |
| 2.0 | A/B the routed `trsm(Right, Lower, ConjTrans, NonUnit)` against a new `PotrfPanelSolveKernel`, at the parent `ld` | **new — Open question 5.** May delete a whole kernel from §8 2.1 |
| 2.1 | `potrf_blocked.cc` — trailing update through the **routed** `gemm` via an injected seam modelled on `TrsmTrailingGemm` (`trsm_native.hh:105-111`); diagonal-block GEMM `alpha = T(-1), beta = T(0)`; `m2 == 0` skips everything; every sub-view built explicitly with parent `ld` **and stride and batch** | §8 2.1, corrected by S1 and the fold-alpha note |
| 2.2 | flip `blocked_available` true in the builder; extend `supports()`'s Blocked arm | replaces §8 2.2's Provider vocabulary |
| 2.3 | extend T1–T9 to blocked sizes, pinned with `BATCHLAS_POTRF_ROUTE=blocked`; add a **blocked-sized complex** T7 | §8 2.3, corrected by W4 and Open question 9 |
| 3.1 | Run the §10.3 grid — at the real leading dimension, at saturation, `n` list re-cut against step 0.2's ceilings | §8 3.1, corrected by W1 and S1 |
| 3.2 | Record the measured window in `RouteTable<Op::potrf,T>::preferred()` with per-clause cell citations — **not** in `tuning_params.hh` | replaces §8 3.1's second half and 3.2. W2 |
| 3.3 | Flip cells only where spec:600's three-part gate fires (0.90×, §7 accuracy harnesses, `ortho_benchmark` end to end) | §8 3.2, unchanged — the gate is at spec:600, see ✱ |
| 3.4 | `Uplo::Upper` for blocked | §8 3.3, unchanged |
| 3.5 | coverage table / `route_diff.sh` | §8 3.4, unchanged |
| — | `MatrixView::operator()(Slice,Slice)` propagating the parent pointer array (`matrix.hh:1140`) | open, reported, deliberately untouched — WP3's open step 19, and [FIX-B-trap] is the workaround |
| — | the native GEMM's strided-`ld` collapse | open — WP3's step 17; potrf's blocked driver inherits it |
| — | `matrix.hh:387,401,413`'s always-true `assert("…")` | open, pre-existing, unrelated to potrf's test plan |
