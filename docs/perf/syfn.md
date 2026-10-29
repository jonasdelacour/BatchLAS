# syfn: matrix functions of Hermitian matrices {#perf_syfn}

> **Status:** current · composition only, untimed

`syfn` forms \f$ F = V f(\Lambda) V^H \f$ per batch item. gfn2-xTB needs it for
the Loewdin orthogonaliser \f$ S^{-1/2} \f$ and the density matrix \f$ P = C f(\varepsilon) C^H \f$
with Fermi occupations \f$ f \f$.

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
Householder reflectors held in double, so every reference is a closed form; Loewdin and the
out-of-order-queue inputs come from `fill_graded_hermitian` (\f$ \kappa = 10^6 \f$ double, \f$ 10^3 \f$
float). All storage is padded and poisoned (`tests/test_support/`).

Deliberate breaks of `src/ops/syfn/syfn.cc` (2026-10-26/27, configs NETLIB + CUDA, all four types; green
before 80/80), each with a narrow red set:

| Break | Red set |
|---|---|
| `InvSqrt` scale \f$ w^{-1/2} \f$ | Loewdin, `FromEigenMatchesHostReference`, `OutOfDomainEigenvaluesSetInfo`, `SaturatingBatchIsBitIdentical` |
| column scale reads `V` at ld = n | `FromEigenMatchesHostReference` only (the only padded `V`) |
| gemm form uses `Trans`, not `ConjTrans` | `SqrtAndInverseOfIndefinite`, `FromEigenMatchesHostReference`, `WeightsPairWithAscendingEigenvalues`; complex only |
| `hermitize(Upper)` | every herk-form test (6) |
| `info` a flag, not a count; `Sqrt` domain `w <= 0` | `OutOfDomainEigenvaluesSetInfo` only |
| `m = 0` leaves `F` unwritten | `NoEigenpairsGivesZero` only |
| no `on_in_order_queue` | `OutOfOrderQueueMatchesInOrder` only, as a race (19 of 24 config-runs) |
| weights paired in reverse | `WeightsPairWithAscendingEigenvalues`, `FromEigenMatchesHostReference` |
| overlap check / weights pointer check removed | `RejectsBadArguments` only (pointer check: CUDA only) |

Vendor-free (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) the CUDA cases pass on native routes and
`SyfnVendorFree.HostWithoutLapackeThrowsNoRoute` checks the `NoRouteError`.

Not covered by a break: `info > n` after a `syev` convergence failure (no input makes `syev` fail), and
the drain before a sub-op's exception propagates (removing it stays green: a race).

## syfn timing to do {#syfn-timing-to-do}

Untimed. To measure at saturation: `syev` against the product (`herk` vs `gemm` form) for gfn2's shapes
(n = 480, batch 1819; n = 1200, batch 76).
