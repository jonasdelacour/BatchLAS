# stedc: divide and conquer for symmetric tridiagonals {#algo_stedc}

> **Status:** current · checked 2026-11-01.

`stedc` computes \f$T = V \Lambda V^T\f$ for a batch of symmetric tridiagonals by Cuppen's method
[1]: tear \f$T\f$ into two halves plus a rank-one term, solve the halves, merge through a secular
equation [2]. Deflation and the Gu-Eisenstat recomputation of the update vector keep the
eigenvectors orthogonal [3]. Leaves use @ref algo_steqr; tuning and timings are in @ref perf_stedc.

## stedc: tearing T into a rank-one update

Let \f$T\f$ have diagonal \f$a_i\f$ and off-diagonal \f$b_i\f$. With \f$m = \lfloor n/2 \rfloor\f$,
\f$\beta = b_m\f$ and \f$u = \mathbf{e}_m + \operatorname{sgn}(\beta)\,\mathbf{e}_{m+1}\f$,

\f[
T = \operatorname{diag}(\hat T_1, \hat T_2) + |\beta|\, u u^T, \qquad
\hat T_1 = T_{1:m,1:m} - |\beta|\,\mathbf{e}_m \mathbf{e}_m^T, \quad
\hat T_2 = T_{m+1:n,m+1:n} - |\beta|\,\mathbf{e}_1 \mathbf{e}_1^T
\f]

([`src/ops/syev/stedc/stedc.cc:560-584`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L560-L584)). With \f$\hat T_i = Q_i \Lambda_i Q_i^T\f$ and
\f$Q = \operatorname{diag}(Q_1, Q_2)\f$,

\f[
Q^T T Q = D + \sigma z z^T, \qquad
D = \operatorname{diag}(\Lambda_1, \Lambda_2) = \operatorname{diag}(d_1, \dots, d_n), \quad
z = \tfrac{1}{\sqrt 2}\begin{bmatrix} Q_1^T \mathbf{e}_m \\ \operatorname{sgn}(\beta)\, Q_2^T \mathbf{e}_1 \end{bmatrix}, \quad
\sigma = 2|\beta|,
\f]

and \f$\|z\|_2 = 1\f$ ([`src/ops/syev/stedc/stedc.cc:180-187`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L180-L187)). The poles \f$d_i\f$ are sorted
ascending, permuting \f$z\f$ and a column map alike ([`src/ops/syev/stedc/stedc.cc:192-194`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L192-L194)).

## stedc: two deflation tests

Both use \f$\theta = 8\varepsilon \max(\max_i |d_i|, \max_i |z_i|)\f$
([`src/ops/syev/stedc/stedc.cc:229-239`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L229-L239)), the tolerance of `dlaed2` [5].

- **Close poles.** For \f$j = 1, \dots, n-1\f$ in order, if \f$|d_{j+1} - d_j| \le \theta\f$, the
  Givens pair from `lartg`\f$(z_{j+1}, z_j)\f$ sets \f$z_j \leftarrow 0\f$,
  \f$z_{j+1} \leftarrow (z_j^2 + z_{j+1}^2)^{1/2}\f$ and rotates columns
  \f$(Q_{:,j}, Q_{:,j+1}) \leftarrow (c Q_{:,j} - s Q_{:,j+1},\; s Q_{:,j} + c Q_{:,j+1})\f$
  ([`src/ops/syev/stedc/stedc.cc:241-264`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L241-L264)). \f$d\f$ is not rotated (an \f$O(\theta)\f$
  perturbation); `dlaed2` rotates it [5].
- **Small weight.** Pole \f$i\f$ is kept iff \f$|\beta|\,|z_i| > \theta\f$
  ([`src/ops/syev/stedc/stedc.cc:283-289`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L283-L289)), i.e. \f$\sigma|z_i| > 2\theta\f$: twice `dlaed2`'s
  threshold [5].

A deflated \f$(d_i, Q_{:,i})\f$ is an eigenpair to \f$O(\theta)\f$. The \f$k\f$ kept poles go first,
in order ([`src/ops/syev/stedc/stedc.cc:294-317`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L294-L317)). \f$\theta\f$ mixes \f$d\f$ (units of \f$T\f$) with
unitless \f$z\f$, so it presumes \f$\|T\|_{\max} \approx 1\f$; `dstedc` scales \f$T\f$ first [5],
BatchLAS does not, and over-deflates when \f$\|T\| \ll 1\f$.

## stedc: the secular equation and interlacing

On the kept poles \f$d_1 < \dots < d_k\f$, the eigenvalues of \f$D + \sigma z z^T\f$ solve [2]

\f[
g(\lambda) = \frac{1}{\sigma} + \sum_{i=1}^{k} \frac{z_i^2}{d_i - \lambda} = 0, \qquad
d_1 < \lambda_1 < d_2 < \dots < d_k < \lambda_k \le d_k + \sigma,
\f]

with eigenvectors \f$q_j \propto (D - \lambda_j I)^{-1} z\f$. With \f$\lambda_j = o + \tau_j\f$ about a
pole \f$o\f$, \f$\delta_{ij} = d_i - \lambda_j\f$ is formed as \f$(d_i - o) - \tau_j\f$, so
\f$\delta_{oj} = -\tau_j\f$ exactly
([`src/ops/syev/stedc/stedc_merge_cta.cc:184-201`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L184-L201), [`623-629`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L623-L629)).

## stedc: the rational root iteration

The GPU solver is rocSOLVER's `seq_solve`/`seq_solve_ext` [6]: `dlaed4` [5] without the three-pole
`dlaed6` step. Fix \f$j < k\f$, write \f$\delta_i = \delta_{ij}\f$, \f$\Delta = d_{j+1} - d_j\f$,
\f$\mu = (d_j + d_{j+1})/2\f$, and \f$\tilde\psi\f$ for \f$g\f$ without poles \f$j, j+1\f$.

1. **Origin.** If \f$g(\mu) = \tilde\psi(\mu) + 2(z_{j+1}^2 - z_j^2)/\Delta > 0\f$, then
   \f$o = d_j\f$, \f$\tau \in (0, \Delta/2)\f$; else \f$o = d_{j+1}\f$, \f$\tau \in (-\Delta/2, 0)\f$
   ([`src/ops/syev/stedc/stedc_merge_cta.cc:310-318`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L310-L318)).
2. **Initial guess.** Freeze \f$c = \tilde\psi(\mu)\f$; for \f$o = d_j\f$ solve
   \f$c\tau^2 - a\tau + b = 0\f$, \f$a = c\Delta + z_j^2 + z_{j+1}^2\f$, \f$b = z_j^2 \Delta\f$:
   \f$\tau_0 = 2b/(a + r)\f$ if \f$a > 0\f$, else \f$(a - r)/(2c)\f$, with \f$r = |a^2 - 4bc|^{1/2}\f$
   ([`src/ops/syev/stedc/stedc_merge_cta.cc:322-330`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L322-L330)).
3. **Step.** \f$\eta\f$ solves \f$C\eta^2 - A\eta + B = 0\f$ with
   \f$A = (\delta_j + \delta_{j+1}) g - \delta_j \delta_{j+1} g'\f$,
   \f$B = \delta_j \delta_{j+1} g\f$ and \f$C\f$ from Li's fixed-weight or middle-way scheme [4]
   ([`src/ops/syev/stedc/stedc_merge_cta.cc:235-273`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L235-L273)):

   \f[
   C_{\text{fw}} = g - \delta_{j+1} g' - (d_j - d_{j+1}) \frac{z_j^2}{\delta_j^2} \;\; (o = d_j), \qquad
   C_{\text{mw}} = g - \delta_j \psi' - \delta_{j+1} \phi',
   \f]

   \f$\psi' = \sum_{i \le j} z_i^2/\delta_i^2\f$, \f$\phi' = \sum_{i > j} z_i^2/\delta_i^2\f$. The
   first step is fixed-weight ([`src/ops/syev/stedc/stedc_merge_cta.cc:374`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L374)); it continues iff
   \f$\mp g > |g_{\text{prev}}|/10\f$ for \f$o = d_j\f$ / \f$d_{j+1}\f$ ([`385-386`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L385-L386)); thereafter
   the scheme toggles when \f$g\f$ keeps its sign and \f$|g|\f$ drops less than 10x ([`415-418`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L415-L418)).
4. **Safeguards.** Newton \f$\eta = -g/g'\f$ if \f$g\eta \ge 0\f$; if \f$\tau + \eta\f$ leaves the
   bracket, step halfway to its end ([`src/ops/syev/stedc/stedc_merge_cta.cc:275-277`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L275-L277), [`158-160`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L158-L160)).
5. **Stop** at \f$|g| \le \varepsilon\,\mathrm{err}\f$ ([`src/ops/syev/stedc/stedc_merge_cta.cc:365`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L365),
   [`393`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L393)), \f$\mathrm{err}\f$ the running bound `ERRETM` of `dlaed4` ([`354-356`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L354-L356)), or at the step
   cap, raising `info`: `max_sec_iter` (`FusedCta`), 50 (`Baseline`, `Fused`), 100 (`Legacy`).

The last root takes \f$o = d_k\f$, \f$\tau \in (0, \sigma/2]\f$ or \f$[\sigma/2, \sigma]\f$ by the
sign of \f$g(d_k + \sigma/2)\f$, and the single scheme \f$C = g - \delta_{k-1}\psi' - \delta_k\phi'\f$
([`src/ops/syev/stedc/stedc_merge_cta.cc:448-480`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L448-L480), [`530-539`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L530-L539)). For \f$k = 1\f$,
\f$\lambda = d_1 + \sigma z_1^2\f$ ([`439-442`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L439-L442)). `Baseline` and `Fused` use a copy of this solver,
[`src/ops/syev/stedc/stedc_secular_roots.hh:127-611`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_secular_roots.hh#L127-L611).

> **Warning:** the `Legacy` solver never reads `rho`; it solves the \f$\sigma = 1\f$ equation
> ([`src/ops/syev/stedc/stedc_secular.cc:72-88`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_secular.cc#L72-L88)).

## stedc: recomputing z for orthogonal eigenvectors

Provided \f$d_i < \hat\lambda_i < d_{i+1}\f$, the computed \f$\hat\lambda_j\f$ are the exact
eigenvalues of \f$D + \sigma \tilde z \tilde z^T\f$, where by Löwner's identity [3]

\f[
\tilde z_i^2 = \frac{\prod_{j=1}^{k} (\hat\lambda_j - d_i)}{\sigma \prod_{j \ne i} (d_j - d_i)}, \qquad
\operatorname{sgn} \tilde z_i = \operatorname{sgn} z_i .
\f]

The kernels form \f$|\tilde z_i| \propto |\delta_{ii} \prod_{j \ne i} \delta_{ij}/(d_i - d_j)|^{1/2}\f$,
paired as in `dlaed3` [5], dropping \f$\sigma^{-1/2}\f$, which normalization absorbs
([`src/ops/syev/stedc/stedc.cc:398-421`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L398-L421); fused kernels:
[`src/ops/syev/stedc/stedc_merge_kernels.hh:63-125`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_kernels.hh#L63-L125), skipped if `enable_rescale` is off). Then

\f[
\tilde q_j = \frac{(D - \hat\lambda_j I)^{-1} \tilde z}{\|(D - \hat\lambda_j I)^{-1} \tilde z\|_2},
\qquad (\tilde q_j)_i \propto \tilde z_i / \delta_{ij}
\f]

([`src/ops/syev/stedc/stedc.cc:438-450`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L438-L450); [`src/ops/syev/stedc/stedc_merge_cta.cc:719-753`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L719-L753), scaled
norm [`636-713`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L636-L713)). As exact eigenvectors of a nearby matrix, the \f$\tilde q_j\f$ are orthogonal to
working accuracy however close the roots; with deflation the method is backward stable [3] (for
\f$\|T\|_{\max} \approx 1\f$).

## stedc: back-transform by GEMM

With \f$W = [\tilde q_1, \dots, \tilde q_k]\f$, \f$G\f$ the deflation rotations, \f$\Pi\f$ the sort
and deflation permutations and \f$\Pi_\lambda\f$ the final eigenvalue sort
([`src/ops/syev/stedc/stedc.cc:467-468`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L467-L468)),

\f[
V = (Q G \Pi)\,(W \oplus I_{n-k})\, \Pi_\lambda .
\f]

As \f$A(W \oplus I) = [A_{:,1:k} W \mid A_{:,k+1:n}]\f$, for \f$n \ge 512\f$ and
\f$k_{\max} = \max_{\text{batch}} k \le 0.75n\f$ the GEMM narrows to \f$k_{\max}\f$ columns
([`src/ops/syev/stedc/stedc.cc:492-515`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L492-L515)). Level-driver merges below the root write through
even/odd views with two GEMMs and never narrow ([`src/ops/syev/stedc/stedc.cc:476-490`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L476-L490)).

## stedc: recursive and level drivers

- **Recursive** ([`src/ops/syev/stedc/stedc.cc:551-603`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L551-L603)): `steqr` (sorted) if \f$n \le n_0\f$, else
  tear, recurse on both halves, merge; one launch chain per node.
- **Level-synchronous**: `plan_stedc_levels` picks \f$L\f$ and leaf \f$\ell = \lceil n/2^L \rceil\f$
  with \f$\max(2, n_0/2) \le \ell \le n_0\f$, least padding of \f$N = \ell\, 2^L\f$, ties toward
  \f$\ell = n_0\f$ ([`src/ops/syev/stedc/stedc_levels_plan.hh:29-46`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_levels_plan.hh#L29-L46)). Pad rows \f$i \in [n, N)\f$
  (0-based) get \f$b_i = 0\f$, \f$a_i = 2\gamma + 1 + (i - n)\f$ (\f$\gamma\f$ the Gershgorin
  bound): they sort last, \f$V_{1:n,1:n}\f$ is the answer, and they enlarge \f$\theta\f$ in the merges
  holding them ([`src/ops/syev/stedc/stedc.cc:779-797`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L779-L797)). All tearings are applied in one pass
  ([`src/ops/syev/stedc/stedc.cc:805-818`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L805-L818)), one `steqr` call solves all \f$2^L\f$ leaves ([`924`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L924)), and
  level \f$l\f$ merges its \f$2^l\f$ nodes per item in one launch ([`941-971`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L941-L971)).

## stedc: routes and merge kernels

`stedc_dispatch` routes on `StedcParams` ([`src/ops/syev/stedc/stedc.cc:1030-1068`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L1030-L1068)), resolved per
\f$n\f$ by `resolve_stedc_tuning` ([`33-61`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L33-L61)); there is no `choice.hh`.

| Condition | Path |
| --- | --- |
| `Backend::NETLIB` | `steqr_legacy` |
| `jobz = NoEigenVectors` | full solve into a workspace (merges need child vectors) |
| `algorithm != Recursive`, \f$L > 0\f$, padded or `eigvects` packed ([`1052-1061`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L1052-L1061)) | `stedc_levels_impl` |
| otherwise | `stedc_impl` |

On GPU, \f$n_0\f$, `merge_variant` (default `FusedCta`) and lanes per root \f$p\f$ come from
[`tuning_params.hh:107-128`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/include/batchlas/tuning_params.hh#L107-L128). Elsewhere the solver is `Legacy`, the merge `Baseline`, and \f$n_0 = 1\f$ unless
`recursion_threshold` is set, so \f$L = 0\f$ and recursion runs to \f$1 \times 1\f$ leaves
([`src/ops/syev/stedc/stedc.cc:49-58`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L49-L58)).

| Merge | Code | Per node |
| --- | --- | --- |
| `Baseline` | [`src/ops/syev/stedc/stedc.cc:345-452`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc.cc#L345-L452) | thread per root |
| `Fused`; `FusedCta` without sub-group 32 | [`src/ops/syev/stedc/stedc_merge_kernels.cc:21-108`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_kernels.cc#L21-L108) | thread per root |
| `FusedCta`, \f$p \le 32\f$ | [`src/ops/syev/stedc/stedc_merge_cta.cc:782-861`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L782-L861) | \f$p\f$ lanes per root, \f$d, z\f$ in local memory |
| `FusedCta`, \f$p > 32\f$ | [`src/ops/syev/stedc/stedc_merge_cta.cc:865-952`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_merge_cta.cc#L865-L952) | work-group, roots serial |
| `Legacy` solver, any variant | [`src/ops/syev/stedc/stedc_secular.cc:37-186`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/syev/stedc/stedc_secular.cc#L37-L186) | work-group, roots serial, log-domain product |

## stedc: references

1. J. J. M. Cuppen, "A divide and conquer method for the symmetric tridiagonal eigenproblem",
   *Numer. Math.* 36:177-195, 1981. https://doi.org/10.1007/BF01396757
2. J. R. Bunch, C. P. Nielsen, D. C. Sorensen, "Rank-one modification of the symmetric
   eigenproblem", *Numer. Math.* 31:31-48, 1978. https://doi.org/10.1007/BF01396012
3. M. Gu, S. C. Eisenstat, "A divide-and-conquer algorithm for the symmetric tridiagonal
   eigenproblem", *SIAM J. Matrix Anal. Appl.* 16(1):172-191, 1995.
   https://doi.org/10.1137/S0895479892241287
4. R.-C. Li, "Solving secular equations stably and efficiently", LAPACK Working Note 89, 1993.
   https://www.netlib.org/lapack/lawnspdf/lawn89.pdf
5. LAPACK reference sources `dstedc.f`, `dlaed2.f`, `dlaed3.f`, `dlaed4.f`.
   https://github.com/Reference-LAPACK/lapack/tree/master/SRC
6. rocSOLVER, `seq_solve` and `seq_solve_ext`.
   https://github.com/ROCm/rocSOLVER/blob/develop_deprecated/library/src/include/lapack_device_functions.hpp
