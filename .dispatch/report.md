# GPU validation, device-LA v2 wave 1 (JRHNBR, RTX 4090 GPU 1)

Base: `origin/dla-v2-base` a3ac2ed8 (worktree `val-base`). Each candidate is `origin/inv/<family>` in its own
seeded worktree `val-<family>`. Everything ran with `CUDA_VISIBLE_DEVICES=1` and
`ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'`.

## Method

- **Tests:** `ctest -R '^(<family targets>)(_heavy|_native|_slow)?$'`. The targets come from
  `affected_tests.py --paths` over the family's `src/ops` and `src/kernels` files; larft uses the list
  from its gate. The same union of 83 tests ran on base: 81/83 pass. The two failures are `gesvd_tests`
  (`GesvdTest/{2,3}.CtaProviderCoversAllJobCombinations`, in `known-failures.txt`) and
  `syev_blocked_tests_heavy` (`EigenvaluesOnlyLowerMatchesNetlib` x4, in
  `known-failures-host-jrhnbr.txt`).
- **Bit-identity:** one harness binary is built without an rpath, and `LD_LIBRARY_PATH` swaps the base
  and candidate `.so` files. The harness is in `.dispatch/evidence/bitid/` and covers every op in the
  wave at batch 1024 with `ld = rows+3`, a stride padded by 7 and filled with NaN poison, and complex
  data with Im(diag) ≠ 0. It records 3464 FNV hashes per run. Two base runs were identical, and the
  negative controls (a deliberate route change and a bidiagonal-solver swap) were detected.
- **Timing:** `.dispatch/evidence/ab.py`.
  - Each comparison runs the same `val-base` benchmark binary against the base and candidate `.so`.
  - Settings: `--warmup=3 --min_iters=8`, base and candidate alternate order, the first round is
    discarded, and each figure is the median of 14 reps.
  - Each timing block holds `flock /tmp/batchlas-measure.lock`, and GPU 0 utilization is logged for
    every block.
  - The orchestrator withdrew the requirement that GPU 0 be idle at 15:28. Potrf and bdsdc were timed
    earlier, with both GPUs idle.
  - The first gebrd pass overlapped the `dispatch-tdm-cuda` benchmark runs on GPU 1, which do not use
    the lock. That pass was discarded and re-run. Before each block, the script now waits until no other
    measuring run (a `gpu_guard.sh 1` process) is on GPU 1. No block after that was contended.

## Per family

| family | ctest (cand) | failures vs base | bit-identity (3464 hashes) | fix pushed |
|---|---|---|---|---|
| potri | 3/3 | none | identical | none |
| potrf-cta-lpanel | 23/23 | none | identical, including complex CTA (algo) and blocked fold n=256 | none |
| geqrf | 42/44, then 43/44 counting the fix | new: `device_blas_v2_geqrf_tests` (fixed); `syev_blocked_heavy` NETLIB x4 also fails on base | identical | **inv/geqrf 37ccf05f** |
| syev-drivers-fused | 27/28 | `syev_blocked_heavy` x4, also fails on base | identical | none |
| gesvd-ormbr | 9/10 | `gesvd_tests` x2, also fails on base | identical | none |
| bdsqr-bdsdc | 8/10, then 9/10 counting the fix | new: `device_blas_v2_bdsqr-bdsdc_tests` SegFault (fixed); `gesvd_tests` x2 also fails on base | identical | **inv/bdsqr-bdsdc dd429378** |
| tridiag-stebz-stein | 37/38 | `syev_blocked_heavy` x4, also fails on base | identical, including the algo ports (tridiagonal_solver, stebz, stein, syevx subset) | none |
| larft-orgqr-ormqr | 20/21 | `gesvd_tests` x2, also fails on base | 3288 identical, 176 differ (see below) | none |
| gebrd | 9/10 | `gesvd_tests` x2, also fails on base | 3040 identical, 424 differ (see below) | none |

- **larft differences:** every real `ormqx_cta` output is identical (L/R, QR/QL, N/T), and so is blocked
  ormqr with `IMPL` unset. The differences are all algo-mode outputs:
  - complex `ormqx_cta` (80 hashes);
  - blocked ormqr with `BATCHLAS_ORMQR_IMPL=device`, the v2 larft, for all types (48 hashes);
  - downstream, complex `syev` cta A (16 hashes) and complex `gesvd` U/Vh (32 hashes).
- **gebrd differences:** every real output is identical (gebrd blocked, cta and unblocked; real
  gesvd). Complex gebrd blocked and unblocked differ for cf and cd (200 hashes), and so do complex gesvd
  S, U, Vh and A (224 hashes). The tests pass.

### Fixes (test-only, no kernel change, so no `sass_parity.sh` re-run)

- **dd429378, inv/bdsqr-bdsdc:** `SelectSortCarriesColumns` sized each batch item as
  `n + rows*ld`, but C spans `(n-1)*ld + rows`. The fill loop wrote 7 elements past the end of the host
  vector, which corrupted the heap. Each item is now `n + n*ld`. 22/22 cases pass, and the
  bdsdc/bdsqr ctests pass.
- **37ccf05f, inv/geqrf:** two fixes. All 24 cases pass and the geqrf ctests pass (8/8).
  - `ButterflyOrdersAreBitExact` scaled its input by the global index mod 7, which broke the 4-item
    period, so the test could never pass. It now scales by the lane index mod 7, and a deliberate break
    is caught for all 4 types.
  - `Geqr2ComposedFromOps<complex<double>>` launched 1024 work-items at 66 registers with no
    work-group bound. I added a test-local `TEST_WG_BOUND` macro (`max_work_group_size`, plus
    `reqd_work_group_size` only when `BATCHLAS_NVPTX_REQD_WG_OK` is set). **This is a new helper the
    brief did not ask for; the reviewer should approve or replace it.**

## Timing (cand/base median ratio; <1 means the candidate is faster)

The full table with min-max ranges is in `.dispatch/evidence/timing.log`.

| family | case | ratio | GPU 0 util |
|---|---|---|---|
| potrf cta | float n=16 / n=48, batch 32768 | **1.084** / 1.005 | idle (old gate) |
| potrf cta | cfloat n=16 / n=48 | 0.895 / **1.060** | idle |
| potrf lpanel:panel=8 | float n=16/32/64 | 1.018 / 0.970 / 1.002 | idle |
| potrf blocked | cfloat n=256 b512 / n=512 b128 | 1.023 / 1.087 (ranges overlap) | idle |
| bdsdc (gesvd_blocked) | f, cf, n=16/32/64, b1024 | 0.999–1.001 | idle |
| gebrd blocked panel | float n=32/64/128, b256 | **1.044 / 1.041** / 0.999 | median 38% |
| gebrd blocked panel | cfloat n=32/64/128 | 0.781 / 0.753 / 0.739 | 38% |
| gebrd unblocked | float n=32/64 | 0.906 / 0.868 | 86% |
| gebrd unblocked | cfloat n=32/64 | 0.391 / 0.382 | 86% |
| gebrd_cta (gesvd_cta) | float n=16/32, b16384 | 0.996 / 1.004 | 0% |
| ormqr_cta Right | float n=13–32 | 0.72–0.84 | 0% |
| ormqr_cta Right | cfloat n=13–32 | 0.45–0.68 | 0% |
| ormqr_cta Left | float n=13–32 | 0.98–1.005 | 0% |
| ormqr_cta Left | cfloat n=13–32 | 0.52–0.86 | 0% |
| ormqr_blocked, v2 larft (`IMPL=device`) vs base `IMPL=device` | f n=64/128/256 | 0.39–0.51 | 0% |
| same | cf n=64/128/256 | 0.23–0.46 | 0% |
| tridiagonal_solver (stehr) | float n=32/64/128, b1024 | 1.040 / 1.003 / 0.976 | 0% |
| syevx DirectSubset (stebz, stebz+stein) | float n=64/128 | 0.983–0.993 | 0% |

Not timed: potri, geqrf, syev-drivers-fused and gesvd-ormbr. They are strict ports with no
occupancy-step register change, and geqrf's register panel is FP64.

## Merge blockers and open items

1. **potrf-cta-lpanel:** two CTA slowdowns at saturation with tight ranges. float n=16 goes from
   0.1221 to 0.1323 ms (+8.4%), and cfloat n=48 from 2.467 to 2.615 ms (+6.0%). The outputs are
   bit-identical, so this is a code-generation or occupancy effect. Check registers and SASS before
   merging.
2. **gebrd:** the blocked panel is 4% slower for float at n=32 and n=64. Complex outputs change, and
   the gate already advised against merging gebrd as strict. It needs a reviewer decision on algo
   mode; complex speeds up 1.3–2.6x.
3. **larft-orgqr-ormqr:** the algo-mode outputs differ (complex ormqr_cta and the v2 larft) and
   propagate into complex syev and gesvd. All tests pass and the speedups are large (up to 4.4x). It
   needs a reviewer decision to accept these as algo changes.
4. **geqrf:** the `TEST_WG_BOUND` helper is a scope addition that needs approval.
5. **Harness gaps:** Masked instantiations, syev two_stage/jacobi, gesvd cta/jacobi, direct
   bdsqr/bdsdc/ormbr calls, complex syevx, and potrf cta at n=96 (that route throws) were not
   hash-checked.
6. **Contention:** GPU 1 is shared with the `dispatch-tdm-cuda` session, whose benchmarks do not take
   the lock. Results from overlapping periods were discarded (see Method).

The push SHA for this report is the commit on `dispatch/gpuval-wave1` that adds this file.
