# Inbox from shard A1-syev

Material found in the retired SYEV_* root notes that belongs on pages A1 does not own.

## For docs/perf/syevx.md (from SYEV_PERF_IDEATION.md, written at e2ff635)

- **Filtered collapses at large batch; profile before tuning (ideation #7, unmeasured).**
  Filtered wins at n=1024, k/n ~ 1%, batch 1 (62 ms vs 114 ms Direct) and loses at batch 64
  (23.5 ms vs 3.5 ms per matrix, 6.7x). A path that wins when the GPU is starved and loses when
  it is saturated suggests the filter (two GEMMs per Chebyshev step, which should saturate well)
  is not the cost, and the ortho / Rayleigh-Ritz tail is. One profile before any further tuning;
  if the tail is the cost, the same fix helps Filtered and LOBPCG at once (both spend their
  non-GEMM time in ortho plus a small projected syev).
- **Two sweeps SYEVX_PLAN.md s13 listed as "now possible" and still undone** as of the retune
  plan: s7.11 the projected-syev provider sweep, and the s7.5 soft-locking A/B under Chol2
  (BATCHLAS_SYEVX_SOFT_LOCK=1). syevx.md:627 already lists the second; check the first.
- The projected Rayleigh-Ritz solve / BATCHLAS_SYEV_CTA_MAX_N / ILUKTests collision is recorded
  in `docs/perf/syev.md#syev-the-lobpcg-projected-solve-knob`; syevx.md already links it
  (syevx.md:622 and :635), so this item is done.

## For whoever owns src/extensions/sytrd_blocked.cc (~line 813)

The comment block marked `OPEN:` says the her2k crossover at the panel's narrow shapes "awaits
an A/B of her2k against the GEMM pair at n2 in {224,480}, k in {16,24,32}". That A/B was run on
2026-08-08: her2k beats the pair 1.32-1.33x at n2=480, batch 512, cfloat, k=16/24/32, and the
host-loop fallback is 1.23x slower than the pair. The OPEN note can become
`evidence: docs/perf/syev.md#syev-her2k-trailing-update-for-complex-float-wp3`.

## For whoever owns src/extensions/syev_blocked.cc (line 259)

The comment names `syev.hh syev_saturated_provider_for_n_values`; the function is now
`syev_saturated_algorithm_for_n_values`.

## For whoever owns src/extensions/two_stage_common.hh (choose_two_stage_kd, lines ~41-103)

The comment above `choose_two_stage_kd` carries three measurement grids (the original kd
table, the 2026-08-04 re-measure, the nb-hint on/off A/B) and the grid-latrd n=2048 note. All
of it is now in `docs/perf/syev.md#syev-the-two-stage-band-width-kd` (reviewer added the
original table and the hint-off row). The comment can shrink to the invariant ("kd = 32; do not
restate two-stage as a plain n >= 1024 rule") plus
`evidence: docs/perf/syev.md#syev-the-two-stage-band-width-kd`.
