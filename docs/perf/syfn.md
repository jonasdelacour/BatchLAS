# syfn: matrix functions of Hermitian matrices {#perf_syfn}

> **Status:** current · composition only, nothing here is timed (RTX 4090 sm_89, CUDA 13.2, 2026-10-25)

`syfn` forms \f$ F = V f(\Lambda) V^H \f$ per batch item (fullerenes plan P2.2). gfn2-xTB needs it twice:
the Loewdin orthogonaliser \f$ S^{-1/2} \f$ once per geometry, and the density matrix
\f$ P = C f(\varepsilon) C^H \f$ with Fermi occupations \f$ f \f$ in every SCC iteration. The plan text is
lost; the contract below is the one this track chose.

## syfn composition {#syfn-composition}

Two entries (`include/batchlas/blas/functions/syfn.hh`):

| Entry | Does | Writes |
|---|---|---|
| `syfn(ctx, A, F, fn, uplo, ws, info)` | `syev` on a copy of the `uplo` triangle of `A`, then `syfn_from_eigen` | `F`, in full; `A` untouched; `F` may be `A` |
| `syfn_from_eigen(ctx, V, W, F, fn, ws, info)` | \f$ F = V \operatorname{diag}(f(W)) V^H \f$ from given pairs, `V` n x m with m <= n | `F`, in full; its storage span must not overlap `V`'s (checked) |

- **The function** is a value, `SyfnFunction<Real>{kind, weights}`: `inv_sqrt()`, `sqrt()`, `inverse()`,
  `exp()`, or `with_weights(span)` with `m * batch` real weights packed like the eigenvalues. There is
  no device function pointer: the kind is an enum switched inside one element-wise kernel, and any
  other function (Fermi occupations included) is computed by the caller as weights. With
  `syfn(A)` weight j pairs with the j-th smallest eigenvalue. The weights must be device-accessible;
  both entries check them (the deducing spelling's own pointer check cannot see inside the struct).
- **Two product forms.** For `InvSqrt`, `Sqrt` and `Exp`, \f$ f \ge 0 \f$, so the columns are scaled
  by \f$ \sqrt{f(w_j)} \f$ and one public `herk` (real: `syrk`) forms \f$ Y Y^H \f$: half the flops of a gemm.
  `Inverse` and `Weights` may be negative, so the columns are scaled by \f$ f(w_j) \f$ and one public
  `gemm` forms \f$ Y V^H \f$. Weights are not inspected on the host to pick the cheaper form: a
  Fermi-occupation call always takes the gemm. Both ops select their own routes.
- **Output.** `F` is written in full and is exactly Hermitian: herk writes one triangle and the two
  halves of a gemm round differently, so `MatrixView::hermitize(Lower)` mirrors the lower triangle
  (conjugated, real diagonal) last. `uplo` names only the input triangle.
- **Out of domain.** \f$ \lambda \le 0 \f$ for `InvSqrt`, \f$ \lambda < 0 \f$ for `Sqrt`,
  \f$ \lambda = 0 \f$ for `Inverse` get weight 0: `F` is \f$ f \f$ on the rest of the spectrum (for an
  exactly singular `Inverse`, the pseudo-inverse). `info[b]` counts them, LAPACK-style positive.
  In `syfn`, `info[b] > n` means `syev` did not converge for that item (`info - n` is its status,
  `F` unspecified), the `sygv` convention. No threshold is applied: a rounding-level positive
  eigenvalue of a numerically singular overlap is inverted as is; screen it with the weights entry.
- **No Op and no route.** There is one algorithm and no vendor counterpart, so `syfn` is not in
  `RoutingSettings::ops` and has no `BATCHLAS_SYFN_ROUTE` or table. Vendor-free it runs wherever
  `syev`, `herk`/`syrk` and `gemm` have native routes (a GPU queue).
- **Workspace** via `BumpAllocator::measuring()`: `syfn_from_eigen` takes \f$ n m + m \f$ scalars per
  item, `syfn` \f$ 2 n^2 + 2 n \f$ (the copy of `A`, the eigenvalues and the core with m = n) plus
  `syev`'s own workspace. Sizing it asks `syev` for its size, so `syfn_buffer_size` itself throws
  `NoRouteError` where `syev` has no route. All launches run on
  `detail::on_in_order_queue`, so an out-of-order `Queue` is safe.

## syfn tests and deliberate breaks {#syfn-tests-and-breaks}

`tests/syfn_tests.cc` builds each item as \f$ Q \operatorname{diag}(d) Q^H \f$ with \f$ Q \f$ two
Householder reflectors (complex vectors for complex types), so every reference is a closed form in
double: Loewdin \f$ \|F A F - I\| / \kappa \f$ on a graded spectrum (\f$ \kappa = 10^6 \f$ double,
\f$ 10^3 \f$ float, whose 7 digits a \f$ 10^6 \f$ condition would exhaust), \f$ e^A \f$ in place against
\f$ Q e^D Q^H \f$, \f$ F F = A \f$ and \f$ F A = I \f$ on an indefinite \f$ A \f$, the weights and
herk forms against a host sum on a non-orthonormal padded `V`, `syfn(A)` with signed weights on an
unsorted spectrum against the ascending pairing (and the deducing spelling bit-identical to the
explicit one), per-item `info` with the other items correct, 1024 identical items bit-identical, an
out-of-order queue bit-identical, and argument checks (partial `F`/`V` overlap, host weights).

Deliberate breaks of `src/ops/syfn/syfn.cc`, each reverted (md5 checked), on the full-target
tree (configs 0-3 NETLIB, 4-7 CUDA; float, double, cfloat, cdouble). Green before: 80/80 (72 before
the review fixes of 2026-10-26; the last four rows are from that round).

| Break | Red set |
|---|---|
| `InvSqrt` scale \f$ w^{-1/2} \f$ instead of \f$ w^{-1/4} \f$ | `InvSqrtOfGradedSpdIsLoewdin`, `FromEigenMatchesHostReference`, `OutOfDomainEigenvaluesSetInfo`, `SaturatingBatchIsBitIdentical`; all 8 configs (32) |
| column scale reads `V` at ld = n | `FromEigenMatchesHostReference` only (8): the only padded `V` |
| gemm form uses `Trans`, not `ConjTrans` | `SqrtAndInverseOfIndefinite`, `FromEigenMatchesHostReference`; complex only (8) |
| `hermitize(Upper)` | every test with a herk-form product (6 tests x 8 = 48); `NoEigenpairs`, `OutOfOrder`, `Rejects` green |
| `info` a flag, not a count | `OutOfDomainEigenvaluesSetInfo` only (8) |
| `Sqrt` domain `w <= 0` | `OutOfDomainEigenvaluesSetInfo` only (8): the exact-zero straddle |
| `m = 0` leaves `F` unwritten | `NoEigenpairsGivesZero` only (8) |
| `syfn` without `on_in_order_queue` | `OutOfOrderQueueMatchesInOrder` only: 7 of 8 configs in one run, 19 of 24 over three repeats (every config at least once, NETLIB every time). Still a race; with batch 8 and no pending writer it caught 2 of 8 |
| weights paired in reverse order | `WeightsPairWithAscendingEigenvalues`, `FromEigenMatchesHostReference`; all 8 configs (16) |
| overlap check back to a start-pointer compare | `RejectsBadArguments` only, on its "V is later columns of F" assertion; all 8 configs |
| no weights pointer check | `RejectsBadArguments` only, on both "host weights" assertions (explicit and deducing spelling); CUDA only (4): a CPU queue reaches host memory |

Vendor-free tree (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`): the 36 CUDA cases pass on native routes; the
NETLIB cases skip, and `SyfnVendorFree.HostWithoutLapackeThrowsNoRoute` checks the `NoRouteError`.
There, the `Trans` break turns the same two tests red on CUDA cfloat/cdouble (4) and `hermitize(Upper)`
the same six on CUDA (24).

Not covered by a break: `info > n` after a `syev` convergence failure (no input makes `syev` fail),
and the drain before a sub-op's exception propagates (removing it left everything green: whether the
freed workspace is still in use is a race). Vendor-free, only `syfn_from_eigen` reaches the drain:
`syfn` and `syfn_buffer_size` throw while sizing `syev`'s workspace, before any kernel.

## syfn timing to do {#syfn-timing-to-do}

Untimed. On a quiet box, at saturation, the split to measure is `syev` against the product
(`herk` vs `gemm` form) for gfn2's shapes (n = 480, batch 1819; n = 1200, batch 76).
