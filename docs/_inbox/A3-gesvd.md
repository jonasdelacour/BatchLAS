# Inbox from shard A3-gesvd

- **docs/perf/README.md** (index table): add a row for `gesvd` -> [gesvd.md](gesvd.md) (`@ref perf_gesvd`).
  Native by default: yes — `native:jacobi` for real max(m,n) <= 32 and all complex general input,
  `native:blocked` above, `native:cta` for Hermitian input.
- **README.md** (A5): the gesvd headline numbers (0.0064 vs 0.339 us/matrix at n=8; 0.468 vs 0.768 at n=32)
  are now recorded at `docs/perf/gesvd.md#gesvd-readme-headline-jacobi-vs-gesvdjbatched-per-matrix`.
- **docs/design/known-defects.md** candidate (observed while verifying, not measured): in
  `src/extensions/gesvdj_cta.cc`, the global rescale's nmax/nmin reductions read
  `Nrm_local[base_n + lane]` for `lane < CC` only, so on the C=64 rung columns 32..63 do not
  influence beta. Correctness unaffected (beta is a power of two); overflow/underflow headroom for
  graded 33..64 input is narrower than the design claims. Recorded in
  `docs/design/gesvd.md#gesvdj_cta-global-power-of-two-scaling`.
- **Comment burn-down candidates** (lab-notebook grids now duplicated in docs/perf/gesvd.md):
  `src/extensions/gesvd_blocked.cc` above `gesvd_use_blocked_gebrd` (unblocked-vs-blocked gebrd grids;
  and the m=1024 tall grid from fea82ed) and above the bidiagonal-solver
  selector (bdsdc/normal/bdsqr accuracy and cost tables); the m=1024 tall grid is now also in
  docs/perf/gesvd.md. Point them at
  `docs/perf/gesvd.md#gesvd-the-unblocked-gebrd-cliff` and
  `docs/perf/gesvd.md#gesvd-tier-3-bdsdc-as-the-bidiagonal-solver`.
- **docs/design/vendor-free-status.md** row for gesvd could link `@ref perf_gesvd` for the wide-band rule
  evidence (`docs/perf/gesvd.md#gesvd-the-wide-band-33-to-64`).
