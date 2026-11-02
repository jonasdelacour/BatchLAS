# steqr: implicit QR/QL on symmetric tridiagonals {#algo_steqr}

> **Status:** current · checked 2026-11-01.

`steqr` computes \f$T = Q \Lambda Q^T\f$ for a batch of symmetric tridiagonal \f$T\f$ by shifted
implicit QR/QL sweeps of Givens rotations, deflating negligible off-diagonals, as LAPACK `dsteqr`
does [2]. A sub-group kernel (`steqr_cta`) serves small \f$n\f$, a work-group kernel (`steqr_wg`) the
rest. Measurements: [perf/steqr](../perf/steqr.md).

## steqr: problem and notation

- \f$T\f$ has diagonal \f$d_1..d_n\f$ and off-diagonal \f$e_1..e_{n-1}\f$, the only stored data.
- \f$u = \varepsilon_{\mathrm{mach}}/2\f$; \f$\mathrm{safmin}\f$, \f$\mathrm{safmax}\f$ bound the safe range
  ([`src/math-helpers.hh:L226-L258`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L226-L258)).
- \f$[\ell, h]\f$ is an unreduced block; \f$m\f$ the first \f$i \ge \ell\f$ with \f$e_i = 0\f$ (else \f$h\f$);
  `max_sweeps` caps sweeps per block.
- \f$(c, s, r) = \operatorname{lartg}(f, g)\f$ defines the rotation
  \f$G = \left[\begin{smallmatrix} c & s \\ -s & c\end{smallmatrix}\right]\f$:

\f[
G\begin{bmatrix} f \\ g \end{bmatrix} = \begin{bmatrix} r \\ 0 \end{bmatrix},\qquad
r = \operatorname{sign}(f)\sqrt{f^2+g^2},\; c = f/r,\; s = g/r,
\f]

with Anderson's safe scaling outside \f$(\sqrt{\mathrm{safmin}}, \sqrt{\mathrm{safmax}/2})\f$ [3, 4]
([`src/math-helpers.hh:L373-L396`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L373-L396)). The FP32 CTA chase computes \f$c, s\f$ by multiplying with a
Newton-refined `rsqrt` instead of dividing by \f$r\f$ ([`src/ops/syev/steqr/steqr_cta_device.hh:L49-L57`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L49-L57)).

## steqr: the implicit shifted step

One QR step with shift \f$\mu\f$ is \f$T - \mu I = \hat Q R\f$, \f$T^+ = R\hat Q + \mu I = \hat Q^T T \hat Q\f$.
By the implicit-Q theorem, while \f$T^+\f$ is unreduced (true on a deflated block) \f$\hat Q\f$ is fixed
up to column signs by its first column, so the step on a block of size \f$k\f$ is a chain of
\f$k-1\f$ rotations [1, §7.4, §8.3]:

1. Rotation 1, from \f$(d_1 - \mu,\, e_1)\f$, creates a bulge \f$z\f$ at \f$(1, 3)\f$.
2. Rotation \f$i\f$, from \f$(e_{i-1}, z)\f$, annihilates \f$z\f$ and pushes it one row down.

([`src/ops/syev/steqr/steqr_wg.cc:L169-L181`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L169-L181); CTA `EXP`: [`src/ops/syev/steqr/steqr_cta_device.hh:L392-L395`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L392-L395).)
With \f$\sigma = -s\f$ ([`src/ops/syev/steqr/steqr_internal.hh:L16-L20`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_internal.hh#L16-L20)), the similarity on rows and
columns \f$i, i+1\f$ touches five band entries:

\f[
\begin{aligned}
d_i^+ &= c^2 d_i - 2c\sigma e_i + \sigma^2 d_{i+1}, &
d_{i+1}^+ &= \sigma^2 d_i + 2c\sigma e_i + c^2 d_{i+1},\\
e_i^+ &= c\sigma(d_i - d_{i+1}) + (c^2 - \sigma^2) e_i, &
e_{i-1}^+ &= c\, e_{i-1} - \sigma z = r,\\
e_{i+1}^+ &= c\, e_{i+1}, &
z^+ &= -\sigma e_{i+1}
\end{aligned}
\f]

([`src/ops/syev/steqr/steqr_internal.hh:L65-L82`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_internal.hh#L65-L82), [`src/ops/syev/steqr/steqr_cta_device.hh:L395-L408`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L395-L408)).

## steqr: choosing QR versus QL

A sweep converges the end its shift is taken from: QR converges \f$d_h\f$, QL (the same step on
\f$JTJ\f$, \f$J\f$ the reversal of \f$[\ell, h]\f$) converges \f$d_\ell\f$. As in `dsteqr` [2], the solver
converges the smaller end first: QL if \f$|d_\ell| \le |d_h|\f$, else QR. The inverted rule loses
relative accuracy on graded input ([evidence](../perf/steqr.md#cta-steqr-chase-micro-structure-decisions)).

- `steqr_wg`: QR on physical indices, QL on virtual indices \f$v \mapsto k-1-v\f$; the orientation is
  stored for the eigenvector replay ([`src/ops/syev/steqr/steqr_wg.cc:L140-L141`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L140-L141)).
- `steqr_cta`: only a QL loop exists; QR mirrors the block (\f$T \leftarrow JTJ\f$, \f$Q \leftarrow QJ\f$),
  runs QL, and mirrors back, also on failure
  ([`src/ops/syev/steqr/steqr_cta_device.hh:L620-L623`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L620-L623), [`:L240-L244`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L240-L244)).

## steqr: the Wilkinson shift

\f$\mu\f$ is the eigenvalue of the end block \f$\left[\begin{smallmatrix} \alpha & \eta \\ \eta & \delta\end{smallmatrix}\right]\f$
closest to the corner \f$\alpha\f$ being converged. Two evaluations, equal in exact arithmetic:

- `Wilkinson`: both roots by the `dlae2` formulas, then the closer one
  ([`src/math-helpers.hh:L481-L531`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L481-L531)). `steqr_wg` always uses it ([`src/ops/syev/steqr/steqr_wg.cc:L165-L167`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L165-L167)).
- `Lapack` (CTA default): \f$\gamma = (\delta-\alpha)/(2\eta)\f$,
  \f$\mu = \alpha - \eta/\bigl(\gamma + \operatorname{sign}(\gamma)\sqrt{\gamma^2+1}\bigr)\f$, the
  cancellation-free form of `dsteqr` [2] ([`src/ops/syev/steqr/steqr_cta_device.hh:L340-L342`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L340-L342)).

Wilkinson-shifted tridiagonal QR/QL converges globally [5, 7], cubically in practice [1, §8.3].

## steqr: the relative deflation criterion

\f$e_i\f$ is set to zero when

\f[
e_i^2 \le u^2\,|d_i|\,|d_{i+1}| + \mathrm{safmin},
\f]

the `EPS2` test of `dsteqr` [2] ([`src/ops/syev/steqr/steqr_cta_device.hh:L220-L223`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L220-L223),
[`src/ops/syev/steqr/steqr_wg.cc:L184-L190`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L184-L190)); nothing scales with \f$\|T\|\f$, `safmin` only guards
underflow. (`dsteqr` also splits at block entry on \f$|e_i| \le u\sqrt{|d_i d_{i+1}|}\f$; BatchLAS does
not.) The dropped entry is at most \f$u\max(|d_i|,|d_{i+1}|) + \sqrt{\mathrm{safmin}}\f$, so by Weyl's
inequality no eigenvalue moves more: a backward-stable perturbation [1, §8.1].

## steqr: safe scaling and the 2x2 solve

- **Scaling.** With \f$a = \max(|d_i|, |e_i|)\f$ over the block, the block is scaled by
  \f$\mathrm{ssfmax}/a\f$ if \f$a > \mathrm{ssfmax} = \sqrt{\mathrm{safmax}}/3\f$, by \f$\mathrm{ssfmin}/a\f$ if
  \f$0 < a < \mathrm{ssfmin} = \sqrt{\mathrm{safmin}}/\varepsilon_{\mathrm{mach}}^2\f$
  ([`src/math-helpers.hh:L238-L246`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L238-L246); `dsteqr` divides by \f$u^2\f$ [2], 4x larger), and back afterwards.
  CTA scales once per block
  ([`src/ops/syev/steqr/steqr_cta_device.hh:L590-L614`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L590-L614), [`:L671-L678`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L671-L678)); `steqr_wg` around every sweep
  ([`src/ops/syev/steqr/steqr_wg.cc:L143-L156`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L143-L156), [`:L192-L200`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L192-L200)).
- **2x2 blocks** are solved in closed form by `laev2` [6]: \f$\lambda_1\f$, \f$\lambda_2\f$ and the unit
  eigenvector \f$(c_1, s_1)\f$ of \f$\lambda_1\f$, applied to \f$Q\f$ as one rotation
  ([`src/math-helpers.hh:L559-L651`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L559-L651); [`src/ops/syev/steqr/steqr_cta_device.hh:L276-L293`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L276-L293),
  [`src/ops/syev/steqr/steqr_wg.cc:L121-L136`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L121-L136)).

## steqr: the CTA update schemes EXP and PG

`SteqrParams::cta_update_scheme` (default `EXP`) picks one of two equal-in-exact-arithmetic forms of
the QL step on \f$[\ell, m]\f$.

- **`EXP`** chases the bulge upward with the band update above, in virtual indices \f$v \mapsto m - v\f$,
  seeded with \f$(x, z) = (d_m - \mu, e_{m-1})\f$
  ([`src/ops/syev/steqr/steqr_cta_device.hh:L368-L369`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L368-L369), [`:L378-L442`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L378-L442)).
- **`PG`** is the scalar recurrence of the `dsteqr` QL loop [2]. With \f$g = d_m - \mu\f$,
  \f$c = s = 1\f$, \f$p = 0\f$, for \f$i = m-1, \dots, \ell\f$:

\f[
\begin{aligned}
&f = s e_i,\quad \beta = c e_i,\quad (c, s, \rho) = \operatorname{lartg}(g, f),\quad
e_{i+1} \leftarrow \rho \;\;(i \ne m-1),\\
&g \leftarrow d_{i+1} - p,\quad \rho \leftarrow (d_i - g)s + 2c\beta,\quad p = s\rho,\quad
d_{i+1} \leftarrow g + p,\quad g \leftarrow c\rho - \beta,
\end{aligned}
\f]

  then \f$d_\ell \leftarrow d_\ell - p\f$, \f$e_\ell \leftarrow g\f$
  ([`src/ops/syev/steqr/steqr_cta_device.hh:L468-L541`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L468-L541)). The lockstep solver always uses `EXP`.

## steqr: the CTA kernel

`steqr_cta` maps one tridiagonal to a chunk of \f$P \in \{4, 8, 16, 32\}\f$ lanes, the smallest
\f$P \ge n\f$ ([`src/ops/syev/steqr/steqr_cta.cc:L235-L243`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta.cc#L235-L243)); lane \f$i\f$ holds \f$d_i, e_i\f$ in
registers and row \f$i\f$ of \f$Q\f$ in shared memory, so each rotation updates \f$Q_{:,\{i,i+1\}}\f$
lane-locally ([`src/ops/syev/steqr/steqr_cta_device.hh:L114-L143`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L114-L143)).

```text
budget <- n * max_sweeps
for each block [bb, be] (left-to-right walk; e_{bb-1} <- 0; deflate the tail):
  scale; if |d_be| < |d_bb|: mirror (QR as QL)
  l <- bb
  while l < be:
    up to max_sweeps times:
      deflate [l, be]; m <- first i >= l with e_i = 0, else be
      m == l     : l += 1, next l
      m == l + 1 : exact 2x2 on (l, l+1); l += 2, next l
      budget == 0: fail
      one implicit QL step on [l, m]; budget -= 1
    no progress: fail
  mirror back; unscale; stop on failure
```

([`src/ops/syev/steqr/steqr_cta_device.hh:L560-L683`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L560-L683)). A failed item raises `info` to 1
([`src/ops/syev/steqr/steqr_cta.cc:L129-L136`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta.cc#L129-L136)). `steqr_cta_solve` runs these steps in one of three
loop shapes, lockstep, flat or nested ([`src/ops/syev/steqr/steqr_cta_device.hh:L991-L1005`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta_device.hh#L991-L1005)); the fused
`syev` calls it directly ([`src/ops/syev/kernels/syev_cta_fused.cc:L249`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_cta_fused.cc#L249), [`:L342`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_cta_fused.cc#L342), [`:L367`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/kernels/syev_cta_fused.cc#L367)), `stedc`
leaves via `steqr_dispatch` ([`src/ops/syev/stedc/stedc.cc:L556`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L556)).

## steqr: the work-group kernel

`steqr_wg` repeats \f$n-1\f$ passes ([`src/ops/syev/steqr/steqr_wg.cc:L320-L334`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L320-L334)); each pass:

1. lists every unreduced block of the batch as `(item, start, end)` by a prefix sum ([`:L56-L100`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L56-L100));
2. runs up to `max_sweeps` QR or QL sweeps per block, in parallel over blocks, stopping at the first
   deflation; size-2 blocks use `laev2` ([`:L121-L203`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L121-L203));
3. replays the stored rotations on \f$Q\f$, one work-group per block ([`:L224-L235`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L224-L235)).

`info` counts the off-diagonals still nonzero after the last pass ([`:L341-L353`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L341-L353)). `steqr_legacy`, the
only solver of the NETLIB `stedc` arm ([`src/ops/syev/stedc/stedc.cc:L1030-L1035`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L1030-L1035)), runs the same
algorithm in its own copy ([`src/ops/syev/steqr/steqr_legacy.cc:L44-L276`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_legacy.cc#L44-L276)) but stops once every
\f$e_i = 0\f$ and replays only the sweeps performed ([`:L388-L409`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_legacy.cc#L388-L409), [`:L254-L258`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_legacy.cc#L254-L258)).

## steqr: routing, eigenvectors and output

- **Route.** `steqr_cta_can_run` (`src/ops/syev/steqr/steqr_internal.hh`, shared by `steqr.cc`,
  `steqr_cta.cc` and `syev.cc`) takes `steqr_cta` when \f$0 < n \le 32\f$ on a GPU with sub-group
  size 32; the routers also exclude ROCm; otherwise `steqr_wg`. There is no
  `choice.hh` or tuned table.
- **Eigenvectors.** Each rotation \f$G_i\f$ (the \f$G\f$ above embedded at \f$(i, i+1)\f$) does
  \f$Q \leftarrow Q G_i^T\f$, so \f$Q^T T Q = \Lambda\f$. \f$Q\f$ starts as \f$I\f$, or as the caller's matrix
  with `back_transform` ([`src/ops/syev/steqr/steqr_cta.cc:L170-L174`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta.cc#L170-L174), [`src/ops/syev/steqr/steqr_wg.cc:L305-L307`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L305-L307)).
- **Output.** \f$\lambda_i = d_i\f$; with `sort` the pairs are sorted in `sort_order`, ascending by
  default ([`src/ops/syev/steqr/steqr_cta.cc:L272-L275`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_cta.cc#L272-L275), [`src/ops/syev/steqr/steqr_wg.cc:L364-L367`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/steqr/steqr_wg.cc#L364-L367)).

## steqr: references

1. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013, §7.4, §8.1, §8.3. ISBN 978-1-4214-0794-4.
   https://nhigham.com/2013/05/31/fourth-edition-of-matrix-computations/
2. LAPACK, `DSTEQR` reference source. https://www.netlib.org/lapack/double/dsteqr.f
3. LAPACK, `DLARTG` reference source (`dlartg.f90`).
   https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlartg.f90
4. E. Anderson, "Algorithm 978: Safe Scaling in the Level 1 BLAS", *ACM Trans. Math. Softw.* 44(1),
   2017. https://doi.org/10.1145/3061665
5. J. H. Wilkinson, "Global convergence of tridiagonal QR algorithm with origin shifts", *Linear
   Algebra Appl.* 1(3):409-420, 1968. https://doi.org/10.1016/0024-3795(68)90017-7
6. LAPACK, `DLAEV2` reference source. https://www.netlib.org/lapack/double/dlaev2.f
7. W. Hoffmann, B. N. Parlett, "A new proof of global convergence for the tridiagonal QL algorithm",
   *SIAM J. Numer. Anal.* 15(5):929-937, 1978. https://doi.org/10.1137/0715060
