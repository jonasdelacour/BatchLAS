# potrf: Cholesky factorization, solve and inverse {#algo_potrf}

> **Status:** current · checked 2026-11-01.

`potrf` factors a batch of Hermitian positive definite matrices as \f$A = LL^H\f$ (or \f$U^H U\f$)
without pivoting; `potrs` solves \f$AX = B\f$ by two triangular solves, `posv` chains the two, and
`potri` forms \f$A^{-1}\f$ by triangular inversion and a triangular product, as LAPACK does [4-7].
Timings: @ref perf_potrf, @ref perf_potrs, @ref perf_potri.

## potrf: the identity and the unblocked recurrence

Let \f$A \in \mathbb{C}^{n\times n}\f$, \f$A = A^H\f$, \f$x^H A x > 0\f$ for all \f$x \ne 0\f$. Then

\f[
\begin{pmatrix} \alpha & a^H \\ a & A_2 \end{pmatrix}
= \begin{pmatrix} \lambda & 0 \\ \ell & L_2 \end{pmatrix}
  \begin{pmatrix} \lambda & \ell^H \\ 0 & L_2^H \end{pmatrix},
\qquad \lambda = \sqrt{\alpha},\quad \ell = a/\lambda,\quad L_2 L_2^H = A_2 - \ell\ell^H .
\f]

The Schur complement \f$A_2 - \ell\ell^H\f$ is again positive definite; recursing on it is the
*right-looking* form [1, §4.2]. `dpotf2` computes the same entries column by column (left-looking) [4].

- **Pivot.** \f$\delta_j\f$, the (1,1) entry of the \f$j\f$-th Schur complement; \f$\ell_{jj} = \sqrt{\delta_j}\f$.
- **Failure.** Step \f$j\f$ stops unless \f$\delta_j > 0\f$, tested as `!(d > 0)` so NaN fails too
  (`dpotf2`: `AJJ.LE.ZERO.OR.DISNAN(AJJ)` [4]). `info` \f$= j\f$ (1-based) of the first failure: the
  leading minor of order \f$j\f$ is not positive definite.
- **Upper.** \f$U = L^H\f$; every route but `lpanel` also runs the transposed schedule.
- **Stability.** \f$\hat L\hat L^H = A + \Delta A\f$, \f$|\Delta A| \le \gamma_{n+1}|\hat L||\hat L^H|\f$,
  \f$\gamma_k = ku/(1-ku)\f$ [2, Thm 10.3], sharpened to \f$(n+1)u\f$ in [3]. With
  \f$\||L||L^H|\|_2 \le n\|A\|_2\f$ [2, §10.1] the error is normwise small, so no pivoting is needed.

## potrf: the blocked identity

For a block size \f$b\f$, \f$A_{11} \in \mathbb{C}^{b\times b}\f$:

\f[
L_{11}L_{11}^H = A_{11}, \qquad L_{21} = A_{21}L_{11}^{-H}, \qquad
A_{22} \leftarrow A_{22} - L_{21}L_{21}^H .
\f]

Right-looking applies the three steps to the leading block column and recurses on \f$A_{22}\f$.
Left-looking first subtracts \f$\sum_{k<j} L_{:,k}L_{j,k}^H\f$ from block column \f$j\f$, then
factors it; reference `dpotrf` does this (`dsyrk`, `dpotrf2`, then `dgemm`, `dtrsm`) [4].

```text
for j = 0, b, 2b, ...                    right-looking
    A11 := chol(A11)                      leaf; info += j on failure
    A21 := A21 * L11^{-H}                 trsm (Right, Lower, ConjTrans)
    A22 := A22 - A21 * A21^H              lower triangle only
```

## potrf: routes and predicates

*native* = GPU, sub-group 32, homogeneous batch ([`src/ops/potrf/potrf.cc:L42`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/potrf.cc#L42)); `max_wg` = largest
work-group. `can_run` ([`src/ops/potrf/potrf.cc:L36-L62`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/potrf.cc#L36-L62)) checks correctness only. The route is the
first entry of the nearest `tuned/potrf.<device>.txt` row that `can_run` admits; failing that,
`blocked`, `vendor`, then candidate order ([`src/select/select.hh:L367-L397`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/select/select.hh#L367-L397),
[`src/select/select.hh:L537`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/select/select.hh#L537)).

| Route | Schedule | Storage | Predicate |
|---|---|---|---|
| `tiny` | unblocked, right-looking | registers, lane \f$r\f$ owns row \f$r\f$ | native, \f$n \le 32\f$ (16 for `complex<double>`), `max_wg` \f$\ge 64\f$ |
| `cta` | right-looking, \f$b = 8\f$ | whole matrix in SLM | native, \f$n \le\f$ `potrf_cta_max_n_for_slm` |
| `lpanel:8`, `lpanel:16` | left-looking | one \f$n \times b\f$ panel in SLM | native, Lower, \f$n \le\f$ `potrf_lpanel_max_n_for_slm`; `:16` float only ([`src/ops/potrf/choice.hh:L36-L39`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/choice.hh#L36-L39)) |
| `blocked` | right-looking | global; leaf `cta`, public `trsm`/`gemm` | native, compiled for \f$T\f$, `potrf_cta_max_n_for_slm` \f$\ge 1\f$ |
| `vendor` | library | library | vendor present, homogeneous, no batch mask |

`tiny`, `cta` and `lpanel` give each batch item its own work-group or sub-group partition [8].

## potrf: the tiny and CTA kernels

**`tiny`.** For column \f$j\f$, lane \f$j\f$ broadcasts \f$\delta_j = a_{jj}\f$
([`src/ops/potrf/kernels/potrf_tiny.cc:L139-L140`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_tiny.cc#L139-L140)); then \f$\ell_{jj} = \sqrt{\delta_j}\f$,
\f$\ell_{rj} = a_{rj}\cdot(1/\ell_{jj})\f$ ([`src/ops/potrf/kernels/potrf_tiny.cc:L153-L165`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_tiny.cc#L153-L165)), and
\f$a_{rk} \mathrel{-}= \ell_{rj}\overline{\ell_{kj}}\f$ for \f$j < k \le r\f$
([`src/ops/potrf/kernels/potrf_tiny.cc:L168-L180`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_tiny.cc#L168-L180)). Upper loads the conjugate transpose
([`src/ops/potrf/kernels/potrf_tiny.cc:L122-L123`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_tiny.cc#L122-L123)).

**`cta`.** The blocked identity with \f$b = 8\f$, each work-item owning a \f$\mathrm{TS}\times\mathrm{TS}\f$
register tile, \f$\mathrm{TS} \in \{2,4\}\f$ ([`src/ops/potrf/kernels/potrf_cta.cc:L43-L46`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_cta.cc#L43-L46)). One load
into SLM, Upper as \f$\tilde a_{ic} = \overline{a_{ci}}\f$ with the diagonal made real
([`include/batchlas/device/lapack/potrf.hh:L427-L435`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L427-L435), [`include/batchlas/device/lapack/potrf.hh:L453-L466`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L453-L466)), one store
([`include/batchlas/device/lapack/potrf.hh:L729-L732`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L729-L732)). Per panel, three barrier-separated phases:

1. \f$L_{11}\f$ on one sub-group by the `tiny` recurrence ([`include/batchlas/device/lapack/potrf.hh:L292-L333`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L292-L333)).
2. \f$L_{21} = A_{21}L_{11}^{-H}\f$ row by row,
   \f$x_c = (a_{ic} - \sum_{p<c} x_p\overline{\ell_{cp}})/\ell_{cc}\f$
   ([`include/batchlas/device/lapack/potrf.hh:L471-L494`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L471-L494)).
3. \f$A_{22} \mathrel{-}= L_{21}L_{21}^H\f$, one tile per lower-triangle tile
   ([`include/batchlas/device/lapack/potrf.hh:L525-L571`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L525-L571)).

## potrf: the left-looking panel kernel

`lpanel` keeps only the current \f$n \times b\f$ panel in SLM, so it reaches orders `cta` cannot
hold; factored columns are re-read from global memory. For panel \f$J = j{:}j{+}b\f$, one work-item
per row \f$r\f$:

\f[
G_{ri} = \sum_{k<j} \ell_{rk}\,\overline{\ell_{j+i,k}}, \qquad P_{ri} = a_{r,j+i} - G_{ri}
\f]

([`include/batchlas/device/lapack/potrf.hh:L907-L927`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L907-L927); \f$G\f$ at
[`include/batchlas/device/lapack/potrf.hh:L912-L922`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L912-L922)). \f$P\f$ is factored
in SLM by the unblocked recurrence ([`include/batchlas/device/lapack/potrf.hh:L928-L930`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L928-L930), [`include/batchlas/device/lapack/potrf.hh:L351-L393`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L351-L393)) and
stored before the next panel reads it ([`include/batchlas/device/lapack/potrf.hh:L931`](https://github.com/jonasdelacour/BatchLAS/blob/2256c2c08680101badb7e1e27d221d5fddf6bceb/include/batchlas/device/lapack/potrf.hh#L931)).

## potrf: the blocked driver

Right-looking over global memory, \f$b = \mathrm{nb}\f$ ([`src/ops/potrf/kernels/potrf_blocked.cc:L338-L421`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L338-L421)):

- **Leaf.** `cta` factors \f$A_{11}\f$ ([`src/ops/potrf/kernels/potrf_blocked.cc:L348-L349`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L348-L349)); a fixup
  sets `info` \f$= j + \mathrm{info}_{\text{leaf}}\f$ and resets a failed item's panel to the
  identity so later solves never divide by a bad pivot ([`src/ops/potrf/kernels/potrf_blocked.cc:L196-L249`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L196-L249)).
- **Panel.** Public `trsm`: \f$A_{21} := A_{21}L_{11}^{-H}\f$, Upper \f$U_{12} := U_{11}^{-H}A_{12}\f$
  ([`src/ops/potrf/kernels/potrf_blocked.cc:L363-L366`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L363-L366)).
- **Trailing update**, in column strips of width \f$W\f$. `gemm` forms \f$-L_cL_c^H\f$ (Upper
  \f$-U_c^H U_c\f$) in scratch and only its stored triangle is added to the diagonal block
  ([`src/ops/potrf/kernels/potrf_blocked.cc:L382-L396`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L382-L396)); the rest of the strip takes
  \f$C \mathrel{-}= L_rL_c^H\f$ (Upper \f$C \mathrel{-}= U_c^H U_r\f$)
  ([`src/ops/potrf/kernels/potrf_blocked.cc:L398-L414`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L398-L414)).
- **Block sizes.** `PotrfBlockedConst` ([`src/ops/potrf/kernels/potrf_blocked.cc:L53-L56`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L53-L56)), overridable
  by `BATCHLAS_POTRF_NB` / `BATCHLAS_POTRF_W`; \f$\mathrm{nb}\f$ is clamped to the device's CTA
  ceiling and to \f$n\f$, then rounded to whole `trsm` CTA blocks
  ([`src/ops/potrf/kernels/potrf_blocked.cc:L77-L104`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrf/kernels/potrf_blocked.cc#L77-L104)).

## potrs: the two triangular solves

\f$LY = B\f$, then \f$L^H X = Y\f$ (Upper: \f$U^H Y = B\f$, \f$UX = Y\f$), as `dpotrs` [5]:

\f[
y_i = \Big(b_i - \sum_{k<i}\ell_{ik}y_k\Big)\Big/\ell_{ii}, \qquad
x_i = \Big(y_i - \sum_{k>i}\overline{\ell_{ki}}\,x_k\Big)\Big/\ell_{ii}.
\f]

Each substitution gives \f$(T + \Delta T)\hat x = b\f$, \f$|\Delta T| \le \gamma_n|T|\f$ [2, Thm 8.5]
(\f$\gamma_{n+1}\f$ when scaling by \f$1/\ell_{ii}\f$, [`src/ops/trsm/kernels/trsm_native.cc:L199`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/trsm/kernels/trsm_native.cc#L199));
end to end \f$(A + \Delta A)\hat x = b\f$, \f$|\Delta A| \le \gamma_{3n+1}|\hat L||\hat L^H|\f$ [2, Thm 10.4].

- **`trsm`.** Two public `trsm` calls ([`src/ops/potrs/kernels/potrs_trsm.cc:L27-L32`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrs/kernels/potrs_trsm.cc#L27-L32)); always admitted.
- **`fused`.** getrs's fused kernel ([`src/ops/potrs/kernels/potrs_trsm.cc:L21`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrs/kernels/potrs_trsm.cc#L21)): one work-group per
  item holds \f$B\f$ in SLM and runs both solves. `fused_fits`: GPU, sub-group 32,
  \f$\mathrm{nrhs} \le 8\f$, \f$n\cdot\mathrm{nrhs}\f$ within the SLM cap
  ([`src/ops/potrs/potrs.hh:L19-L23`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrs/potrs.hh#L19-L23), applied at [`src/ops/potrs/potrs.cc:L39`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potrs/potrs.cc#L39)). Diagonal blocks
  are solved by sub-group shuffles ([`src/ops/getrs/kernels/getrs_fused.cc:L716-L736`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/getrs/kernels/getrs_fused.cc#L716-L736)); off-diagonal
  coupling is an axpy \f$y_i \mathrel{-}= \sum_{k\in J}\ell_{ik}y_k\f$
  ([`src/ops/getrs/kernels/getrs_fused.cc:L739-L756`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/getrs/kernels/getrs_fused.cc#L739-L756)) or a dot
  \f$y_j \mathrel{-}= \sum_{i\ \mathrm{solved}}\overline{\ell_{ij}}\,y_i\f$
  ([`src/ops/getrs/kernels/getrs_fused.cc:L694-L713`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/getrs/kernels/getrs_fused.cc#L694-L713)), whichever walks the stored triangle
  contiguously. The backward solve mirrors this ([`src/ops/getrs/kernels/getrs_fused.cc:L760-L828`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/getrs/kernels/getrs_fused.cc#L760-L828)).

## posv: factor then solve

`posv` = `potrf` + `potrs` per item (predicates [`src/ops/posv/posv.cc:L43-L57`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/posv.cc#L43-L57)).

| Route | Composition | Predicate |
|---|---|---|
| `tiny` | `tiny` factorization, both solves with \f$L\f$ in registers | \f$n \le 32\f$ (16 for `complex<double>`), \f$\mathrm{nrhs} \le 4\f$, `max_wg` \f$\ge 64\f$ |
| `cta` | public `potrf`, fused `potrs` ([`src/ops/posv/posv.cc:L68-L73`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/posv.cc#L68-L73)) | native, `fused_fits` |
| `blocked` | public `potrf`, `trsm` pair ([`src/ops/posv/posv.cc:L74-L78`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/posv.cc#L74-L78)) | homogeneous batch |

In `tiny`, lane \f$r\f$ holds row \f$r\f$ of \f$L\f$. Forward, \f$\ell_{ri}\f$ is local and lane
\f$i\f$ broadcasts \f$\ell_{ii}\f$, \f$b_i\f$ ([`src/ops/posv/kernels/posv_tiny.cc:L219-L244`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/kernels/posv_tiny.cc#L219-L244));
backward, lane \f$i\f$ broadcasts the column \f$\overline{\ell_{ir}}\f$
([`src/ops/posv/kernels/posv_tiny.cc:L250-L304`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/kernels/posv_tiny.cc#L250-L304)). An exactly zero stored \f$\ell_{ii}\f$ yields
\f$y_i = 0\f$ (\f$x_i = 0\f$) instead of \f$\pm\infty\f$ ([`src/ops/posv/kernels/posv_tiny.cc:L222-L239`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/kernels/posv_tiny.cc#L222-L239),
[`src/ops/posv/kernels/posv_tiny.cc:L285-L298`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/posv/kernels/posv_tiny.cc#L285-L298)); other failed pivots are divided through. After
`info` \f$> 0\f$, \f$X\f$ is undefined, as in LAPACK.

## potri: the inverse from trtri and lauum

With \f$X := L^{-1}\f$, \f$Y := A^{-1} = X^H X\f$: `trtri`, then `lauum`, as `dpotri` [6]. An exactly
zero \f$\ell_{ii}\f$ sets `info` \f$= i\f$ and leaves the item unchanged
([`src/ops/potri/kernels/potri_cta.cc:L180-L187`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L180-L187)). `cta` needs \f$n \le\f$ the CTA ceiling,
`blocked:nb` needs \f$\min(\mathrm{nb}, n) \le\f$ it ([`src/ops/potri/potri.cc:L50-L51`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/potri.cc#L50-L51)). Error
bounds for triangular-inversion variants: [9], [2, Ch. 14].

**`cta`** (whole matrix in SLM). \f$L = MD\f$, \f$D = \mathrm{diag}(L)\f$, \f$M\f$ unit lower with
\f$m_{ij} = \ell_{ij}/\ell_{jj}\f$ ([`src/ops/potri/kernels/potri_cta.cc:L173`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L173); \f$D^{-1}\f$ at
[`src/ops/potri/kernels/potri_cta.cc:L170`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L170)), so

\f[
X = D^{-1}Z, \qquad MZ = I .
\f]

Forward substitution in tile rows of \f$\mathrm{TS}\f$: row \f$K\f$ solves within itself
([`src/ops/potri/kernels/potri_cta.cc:L215-L231`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L215-L231)), then each row \f$I > K\f$ takes
\f$Z_I \mathrel{-}= M_{IK}Z_K\f$ ([`src/ops/potri/kernels/potri_cta.cc:L233-L252`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L233-L252)). Then
\f$X = D^{-1}Z\f$ ([`src/ops/potri/kernels/potri_cta.cc:L256-L269`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L256-L269)) and
\f$y_{ij} = \sum_{k\ge\max(i,j)}\overline{x_{ki}}\,x_{kj}\f$ ([`src/ops/potri/kernels/potri_cta.cc:L284-L300`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_cta.cc#L284-L300)).

**`blocked`** (\f$\mathrm{nb}\times\mathrm{nb}\f$ blocks). Triangular inversion uses

\f[
\begin{pmatrix} L_{11} & 0 \\ L_{21} & L_{22}\end{pmatrix}^{-1}
= \begin{pmatrix} X_{11} & 0 \\ -X_{22}L_{21}X_{11} & X_{22}\end{pmatrix}.
\f]

One CTA launch inverts every diagonal block ([`src/ops/potri/kernels/potri_blocked.cc:L108`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_blocked.cc#L108)); then,
last block column first, two `trmm` form \f$T = X_{22}L_{21}\f$ and \f$X_{21} = -TX_{11}\f$
([`src/ops/potri/kernels/potri_blocked.cc:L110-L117`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_blocked.cc#L110-L117); Upper
[`src/ops/potri/kernels/potri_blocked.cc:L120-L127`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_blocked.cc#L120-L127)). `dtrtri` instead calls
`dtrsm` with the uninverted \f$L_{11}\f$ [6]. `lauum` takes block rows \f$i\f$ in ascending order, as
blocked `dlauum` [7]:

\f[
Y_{i,<i} = X_{ii}^H X_{i,<i} + X_{>i,i}^H X_{>i,<i}, \qquad
Y_{ii} = X_{ii}^H X_{ii} + X_{>i,i}^H X_{>i,i},
\f]

the first a `trmm` plus `gemm` ([`src/ops/potri/kernels/potri_blocked.cc:L142-L150`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_blocked.cc#L142-L150)), the second the
CTA `lauum` plus `herk`/`syrk` ([`src/ops/potri/kernels/potri_blocked.cc:L154-L160`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/potri/kernels/potri_blocked.cc#L154-L160)). Row \f$i\f$ reads
only block rows \f$\ge i\f$ of \f$X\f$, not yet overwritten.

## potrf: references

1. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins, 2013, §4.2. ISBN 978-1-4214-0794-4.
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
   https://doi.org/10.1137/1.9780898718027
3. S. M. Rump, C.-P. Jeannerod, "Improved backward error bounds for LU and Cholesky factorizations",
   *SIAM J. Matrix Anal. Appl.* 35(2):684-698, 2014. https://doi.org/10.1137/130927231
4. LAPACK `dpotf2.f`, `dpotrf.f`: https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dpotf2.f,
   https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dpotrf.f
5. LAPACK `dpotrs.f`: https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dpotrs.f
6. LAPACK `dpotri.f`, `dtrtri.f`: https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dpotri.f,
   https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dtrtri.f
7. LAPACK `dlauum.f`: https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dlauum.f
8. T. Dong, A. Haidar, S. Tomov, J. Dongarra, "A fast batched Cholesky factorization on a GPU",
   *ICPP 2014*, IEEE, pp. 432-440. https://doi.org/10.1109/ICPP.2014.52
9. J. J. Du Croz, N. J. Higham, "Stability of methods for matrix inversion", *IMA J. Numer. Anal.*
   12(1):1-19, 1992. https://doi.org/10.1093/imanum/12.1.1
