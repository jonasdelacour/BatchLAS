# WP3 step 12 — the `Side::Left` staging tile

Spec §3.4 specified an SLM transpose staging tile and it was deliberately not
built in step 7, because at that point the coalescing cost was predicted, not
measured. Step 9 measured it: `float`, `Side::Left`, order ≥ 32 was the only
losing region in the entire routing grid. This is the fix.

## Diagnose first — and the diagnosis corrected the spec

`profile.sh` (ncu) on three cells whose ranking was already known:

| cell | load sec/req | store sec/req | DRAM vs floor | time |
|---|---|---|---|---|
| float Left n=32 | **31.39** (7.85×) | **32.00** (8.00×) | 0.85× | 0.517 ms |
| float Left n=8 | 31.20 | 32.00 | 0.51× | 0.023 ms |
| float Right n=32 | 5.13 (1.28×) | 4.00 (1.00×) | 0.75× | 0.141 ms |

A coalesced 32-lane float load moves 128 B = 4 sectors in one request, so 31.4
is 7.85× — almost exactly the 8× §3.4 predicted.

**But not at the level §3.4 said.** It calls this "8× over-fetch on both the
read and the write-allocate", which reads as DRAM traffic. DRAM measures
**0.75–0.85× of the analytic floor** `2·q·n·sizeof(T)·batch` — *below* it. The
bytes a lane skips at step `s` are the bytes it wants at steps `s+1..s+7`, and
they survive in cache until then. The defect is purely LSU/L1 transaction
count. The fix is the same either way, but "short of DRAM bandwidth" would have
been the wrong thing to optimise — and that exact misreading has already cost
this repo one panel kernel.

## After

| float Left n=32 | before | after |
|---|---|---|
| load sectors/request | 31.39 | **5.13** |
| store sectors/request | 32.00 | **4.00** |
| kernel time | 0.517 ms | **0.145 ms** |

0.145 ms against `Side::Right`'s 0.141 ms — the side asymmetry is gone.

## Routing effect (`left-{vendor,native}.csv`, worst cell per order)

| float `Side::Left` | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| before | 1.61× | 1.34× | **0.70×** | **0.79×** | **0.71×** | **0.57×** |
| after | 1.60× | 1.73× | **1.79×** | **1.49×** | **1.19×** | 0.76× |

`double` improved everywhere too (1.39–6.37× → 1.51–8.92×). So `preferred()`'s
float/Left clause moved from `order <= 16` to `order <= 128`. **Order 256 stays
the vendor's** at 0.76–0.93×; the grid jumps 128 → 256 and does not say where in
between it turns, so the boundary sits at the largest order measured to win.

## Only the real types stage — a measured exclusion

Applying the tile to all four types cost the complex kernels their register
residency (`scripts/register_probe.sh`, `Side::Left`, registers / stack frame):

| type | N=8 | N=16 | N=32 |
|---|---|---|---|
| float | 27 / 0 B | 36 / 0 B | 53 / 0 B |
| double | 40 / 0 B | 60 / 0 B | 90 / 0 B |
| `complex<float>` | 40 / 0 B | 56 / 0 B | 72 / **464 B** |
| `complex<double>` | 70 / **16 B** | 104 / **16 B** | 170 / **232 B** |

Spill stores and loads are zero in every row — which is exactly why the *frame*
and not the spill counter is the gate: a frame with no spill is the accumulator
array sitting in local memory. The nested round loop stops fully unrolling for
the wide bodies.

They also do not need it. Over-fetch is `32/sizeof(T)` lanes per sector, so a
16-byte scalar is capped at 2×. **float is the only type that loses precisely
because 32/4 = 8.** With the gate, all 24 kernels are back to zero frame, zero
spill, and complex measures 1.00–1.01× either way — i.e. untouched.

## End-to-end

`ortho_benchmark` hardcoded `Transpose::NoTrans`, and `ortho.cc:205,289` pick
the side from exactly that flag — so the whole `Side::Left` half of trsm's table
had **never** been measured at caller level. `arg4` now selects it.

* forced native, `Side::Left`: float orders 16–128 win **1.29–2.38×**; order 256
  loses 0.78–0.94×, matching the kernel grid.
* **route unset, so `preferred()` chooses: 80/80 at or above parity**, worst
  0.99×, best 2.38× — and at order 256 the default tracks *vendor* (4.08 vs
  vendor 4.07, native 4.35), i.e. the predicate correctly declines native there.

## Two measurement failures worth keeping

**A contaminated profile nearly added a bogus gate.** The first post-tile
profile showed n=8 at 0.028 ms against 0.023 ms before, and I was ready to gate
staging on order to protect it. On the clean grid n=8 is **1.01–1.04×** — there
was nothing to protect. The gate would have been an artefact encoded as a
constant.

**`gpu_guard.sh` cannot see the operator.** A sweep came back with 22 of 180
native cells at 10–103% relative standard deviation while the guard reported the
run clean. Cause: **two copies of my own sweep script**, both queued politely
behind a co-tenant's job, both starting when it freed. The guard samples
*foreign* processes at the start and end; this was neither foreign nor at an
endpoint. `left_sweep.sh` now takes a `flock` and **deletes** any leg with cells
above 10% sd. A clean leg has 0 of 180.

(Separately, a genuine foreign tenant did appear mid-run once, and there the
guard worked exactly as designed — it printed "DISCARD these numbers and re-run",
and they were discarded.)

## Verified

`trsm_tests` 81/81, and mutation-tested: breaking the tile stride fails 16
tests, dropping the direction flag from the staged address map fails 12.
`ctest -L 'blas|ortho'` 20/20. Full vendor-present 52/53 (documented
`lanczos_tests` baseline). Vendor-free `trsm_tests` 49 passing with the failing
set byte-identical to baseline. Route diff `wp3-s9 → wp3-s12`: one new row, and
it is `route_vocabulary_tests` recording its own `resolve_trsm_route` call —
zero library decisions moved, since every suite trsm call is below the batch
floor.
