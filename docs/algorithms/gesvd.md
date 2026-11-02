# gesvd: singular value decomposition of batched small matrices {#algo_gesvd}

> **Status:** current · checked 2026-11-01.

`gesvd` factors each \f$m \times n\f$ item as \f$A = U \Sigma V^H\f$: Householder reduction to
\f$A = QBP^H\f$ with \f$B\f$ real bidiagonal, then an SVD of \f$B\f$ (Golub-Kahan tridiagonal,
bidiagonal QR or \f$B^TB\f$); or one-sided Jacobi on \f$A\f$. Hermitian input goes through `syev`.
Design: @ref design_gesvd. Measurements: @ref perf_gesvd.

## gesvd: problem and routes

\f$k = \min(m,n)\f$, \f$\sigma_1 \ge \dots \ge \sigma_k \ge 0\f$, \f$\varepsilon\f$ = machine
epsilon. For \f$m < n\f$ native routes factor \f$A^H\f$ and swap \f$U, V\f$
([`src/ops/gesvd/kernels/gesvd_blocked.cc:L833-L921`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L833-L921), [`src/ops/gesvd/kernels/gesvdj_cta.cc:L306`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L306));
below, \f$m \ge n\f$. Native routes need a GPU, a homogeneous batch, \f$m, n \ge 1\f$
([`src/ops/gesvd/gesvd.cc:L80`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc#L80)); kernel choice: [`gesvd_blocked.cc:L985-L995`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L985-L995).

| Route | Method | `can_run` ([`gesvd.cc`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc)) |
|---|---|---|
| `jacobi` | one-sided Jacobi | [`:L87-L90`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc#L87-L90): sub-group 32, not Hermitian, \f$\max(m,n) \le\f$ largest \f$C \in \{64, \dots, 4\}\f$ fitting local memory ([`gesvd_native.hh:L35-L38`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd_native.hh#L35-L38)) |
| `cta` | `gebrd_cta` (square) or unblocked `gebrd`, then \f$B^TB\f$; Hermitian: `syev_cta` | [`:L91-L94`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc#L91-L94): sub-group 32, \f$\max(m,n) \le 32\f$, not thin, real or Hermitian square |
| `blocked` | blocked `gebrd` if \f$k \ge\f$ `gesvd_blocked_gebrd_min` (default 1, [`gesvd_blocked.cc:L46-L50`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L46-L50)); `BATCHLAS_GESVD_BIDIAG` = `bdsdc` (default) \| `bdsqr` \| `normal` ([`gesvd_native.hh:L45-L55`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd_native.hh#L45-L55)); `ormbr`. Hermitian: `syev_blocked` | [`:L95-L99`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc#L95-L99): Hermitian \f$\Rightarrow\f$ square, mask honoured |
| `vendor` | solver library (no \f$A^H\f$ swap) | [`:L100`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gesvd.cc#L100) |

## gesvd: Householder bidiagonalisation

Alternating left and right reflectors give [1, 2]
\f[
H_n^H \cdots H_1^H\, A\, G_1 \cdots G_{n-1} = \begin{bmatrix} B \\ 0 \end{bmatrix}, \qquad
B = \operatorname{bidiag}(d_1, \dots, d_n;\ e_1, \dots, e_{n-1}),
\f]
so \f$A = QBP^H\f$, \f$Q = H_1 \cdots H_n\f$, \f$P = G_1 \cdots G_{n-1}\f$.
\f$H_i = I - \tau^Q_i v_i v_i^H\f$ zeroes \f$A(i{+}1{:}m, i)\f$, \f$G_i = I - \tau^P_i u_i u_i^H\f$
zeroes \f$A(i, i{+}2{:}n)\f$; the vectors overwrite what they zero. The reduction is normwise
backward stable [2].

**Reflector.** For \f$(\alpha, x)\f$: \f$\beta = -\operatorname{sign}(\operatorname{Re}\alpha)\sqrt{|\alpha|^2 + \|x\|_2^2}\f$,
\f$\tau = (\beta - \alpha)/\beta\f$, \f$x \leftarrow x/(\alpha - \beta)\f$
([`src/math-helpers.hh:L429-L434`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/math-helpers.hh#L429-L434)). As in `zlarfg` [6], \f$\beta\f$ is real for complex
\f$\alpha\f$ ([`src/ops/gesvd/gebrd/larfg_real_beta.hh:L24-L29`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/larfg_real_beta.hh#L24-L29)), so \f$B\f$ is real and the real
solvers below serve complex input.

- **Unblocked**, one work-item per matrix: \f$A \leftarrow H_i^H A\f$
  ([`src/ops/gesvd/gebrd/gebrd.cc:L46-L73`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd.cc#L46-L73)), then \f$A \leftarrow A G_i\f$ ([`:L85-L113`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd.cc#L85-L113)); complex
  input conjugates row \f$i\f$ around \f$G_i\f$, as `zgebd2` does ([`:L178`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd.cc#L178), [`:L191`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd.cc#L191)).
- **Blocked**, `?labrd` [6], one work-group per matrix. With \f$L = [v_1 \cdots v_b]\f$, \f$R\f$
  the stored rows \f$u_i^H\f$, and subscript 2 for trailing rows/columns, a panel of \f$b\f$ steps
  accumulates \f$X\f$ (\f$m \times b\f$), \f$Y\f$ (\f$n \times b\f$) and ends with two `gemm`s
  ([`src/ops/gesvd/gebrd/gebrd_blocked.cc:L377-L378`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd_blocked.cc#L377-L378)):
  \f[
  A_{22} \leftarrow A_{22} - L_2 Y_2^H - X_2 R_2 .
  \f]
  Step \f$i\f$ appends (real case) \f$y_i = \tau^Q_i(A^T v_i - Y L^T v_i - R^T X^T v_i)\f$
  ([`:L238-L265`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd_blocked.cc#L238-L265)) and \f$x_i = \tau^P_i(A u_i - L Y^T u_i - X R u_i)\f$ ([`:L319-L352`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/gebrd/gebrd_blocked.cc#L319-L352)).

## gesvd: the Golub-Kahan tridiagonal (bdsdc)

`bdsdc` here is `stedc` on \f$T_{GK}\f$ [5], not LAPACK's divide-and-conquer `dbdsdc`. Under the
perfect shuffle \f$(v_1, u_1, \dots, v_n, u_n)\f$,
\f$\left[\begin{smallmatrix} 0 & B^T \\ B & 0 \end{smallmatrix}\right]\f$ becomes [1, 5]
\f[
T_{GK} = \operatorname{tridiag}(f, 0, f),\quad f = (d_1, e_1, \dots, e_{n-1}, d_n),\qquad
T_{GK} z_\pm = \pm\sigma z_\pm,\quad z_\pm = \tfrac{1}{\sqrt 2}(v_1, \pm u_1, \dots, v_n, \pm u_n),
\f]
since its rows read \f$Bv = \sigma u\f$, \f$B^T u = \sigma v\f$. No product is formed, so the
error in \f$\sigma\f$ is \f$O(\varepsilon\|B\|)\f$, not \f$O(\varepsilon\|B\|^2/\sigma)\f$ as via
\f$B^TB\f$.

1. **Sign fold** (`stedc` needs \f$f \ge 0\f$): \f$\eta_1 = 1\f$,
   \f$\eta_{i+1} = \eta_i \operatorname{sign} f_i\f$, \f$S = \operatorname{diag}(\eta)\f$, so
   \f$ST_{GK}S\f$ has off-diagonal \f$|f|\f$ ([`src/ops/gesvd/bdsdc/bdsdc.cc:L121-L127`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L121-L127)).
2. **Solve** \f$ST_{GK}S = Z\Lambda Z^T\f$, \f$\Lambda\f$ ascending (@ref algo_stedc; [`:L415-L423`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L415-L423)).
3. **Extract** \f$\sigma_j = \max(\lambda_{2n+1-j}, 0)\f$ from eigenvector \f$Sz\f$: its
   \f$v\f$-positions give \f$v_j\f$, its \f$u\f$-positions \f$u_j\f$ ([`:L226`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L226), [`:L229`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L229)). Each half
   is renormalised; a half of norm \f$\le 1/16\f$ is zeroed ([`:L199-L231`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L199-L231)).
4. **Repair** columns with \f$\sigma_j \le 500\,\varepsilon\,\sigma_{\max}\f$ ([`:L266-L285`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L266-L285)): seed
   with the axis least covered by the final columns ([`:L299-L320`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L299-L320)), two passes of modified
   Gram-Schmidt ([`:L323-L341`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L323-L341)), normalise ([`:L343-L357`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsdc/bdsdc.cc#L343-L357)).

## gesvd: implicit QR on the bidiagonal (bdsqr)

One work-item per matrix runs Golub-Kahan-Reinsch steps after `dbdsqr` [1, 4] on the lowest
unreduced block \f$B(l{:}r, l{:}r)\f$. Let \f$M = \max_j(|d_j|, |e_j|)\f$.

- **Tolerance:** \f$\mathrm{tol} = \max(10, \min(100, \varepsilon^{-1/8}))\,\varepsilon\f$; at most
  \f$\max(32, 6n^2)\f$ iterations ([`src/ops/gesvd/bdsqr/bdsqr.cc:L104-L108`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L104-L108)).
- **Deflation:** \f$e_i \leftarrow 0\f$ if
  \f$|e_i| \le \max(\varepsilon(|d_i| + |d_{i+1}|),\ \mathrm{tol}\,M)\f$ ([`:L141-L158`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L141-L158)). The
  second term is `dbdsqr`'s absolute criterion [4]: accuracy is normwise, not relative as in [3].
- **Zero diagonal:** if \f$|d_z| \le \max(\mathrm{tol}\,M, \mathrm{realmin})\f$ (smallest normal
  number), set \f$d_z = 0\f$ and chase \f$e_z\f$ out with Givens rotations [2, §8.6.2]: left
  rotations rightwards for \f$z < r\f$, right rotations leftwards for \f$z = r\f$ ([`:L180-L228`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L180-L228)).
  `dbdsqr` leaves this to its zero-shift sweep [3].
- **Shift:** \f$\mu\f$ = smaller singular value of the trailing \f$2 \times 2\f$ block (`las2`),
  0 if \f$(\mu/d_l)^2 < \varepsilon\f$. The sweep starts from column 1 of
  \f$(B^TB - \mu^2 I)/d_l\f$ ([`:L233-L240`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L233-L240)):
  \f$(\phi, \gamma) = \big((|d_l| - \mu)(\operatorname{sign} d_l + \mu/d_l),\ e_l\big) = \big((d_l^2 - \mu^2)/d_l,\ e_l\big)\f$.
- **Chase:** for \f$i = l, \dots, r-1\f$, a right rotation `lartg`\f$(\phi, \gamma)\f$ into
  \f$V^T\f$, then a left one into \f$U\f$ ([`:L242-L280`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L242-L280)), top to bottom even for \f$\mu = 0\f$.
- **Finish:** \f$d_i < 0\f$ flips row \f$i\f$ of \f$V^T\f$; sort descending ([`:L288-L317`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/bdsqr/bdsqr.cc#L288-L317)).

## gesvd: the normal-equations path (cta)

Used by `cta`, and by `blocked` under `normal` except for complex input or a thin \f$U\f$ with
\f$m > n\f$, which solve \f$B\f$ directly ([`src/ops/gesvd/kernels/gesvd_blocked.cc:L69-L71`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L69-L71),
[`:L928`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L928)). From \f$B^TB = V_B\Sigma^2 V_B^T\f$:
\f[
(B^TB)_{ii} = d_i^2 + e_{i-1}^2,\quad (B^TB)_{i,i+1} = d_i e_i,\qquad
\sigma_j = \sqrt{\max(\lambda_{n+1-j}, 0)},\quad u_j = Bv_j/\sigma_j \ \text{if } \sigma_j > \varepsilon\sigma_1
\f]
([`:L290-L300`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L290-L300), [`:L362-L364`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L362-L364), [`:L643-L654`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L643-L654)).

- **Solver:** `cta`: `steqr_cta` (@ref algo_steqr); `blocked`: `stedc` with the `bdsdc` sign fold
  ([`:L216-L224`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L216-L224), [`:L256-L262`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L256-L262)).
- **Missing left vectors:** if \f$U\f$ is wanted and \f$m > n\f$ or some
  \f$\sigma_j \le \varepsilon\sigma_1\f$ ([`:L1092-L1094`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L1092-L1094), [`:L679-L680`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L679-L680)), \f$BB^T\f$ of order
  \f$m\f$ is solved too, \f$(BB^T)_{ii} = d_i^2 + e_i^2\f$, \f$(BB^T)_{i,i+1} = e_i d_{i+1}\f$
  ([`:L326-L340`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L326-L340)); its eigenvectors are copied, not orthogonalised, into the undetermined columns
  ([`:L716-L722`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L716-L722)).
- **Accuracy:** any \f$\sigma_j \lesssim \sqrt\varepsilon\,\sigma_1\f$ has no correct digits.

## gesvd: back-transformation with ormbr

With \f$B = U_B \Sigma V_B^T\f$ written into the leading blocks of `U = Vh = I`:
\f[
U = Q \begin{bmatrix} U_B & 0 \\ 0 & I_{m-n} \end{bmatrix}, \qquad V^H = V_B^T P^H
\f]
([`src/ops/gesvd/kernels/gesvd_blocked.cc:L1129-L1151`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L1129-L1151)).

- **\f$Q\f$** ([`:L728-L755`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L728-L755); [`src/ops/gesvd/ormbr/ormbr.cc:L39-L48`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L39-L48), [`:L405-L421`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L405-L421)): `ormqx_cta`
  for square operands with \f$n \le 32\f$ on a native GPU backend, else `ormqr_blocked`.
- **\f$P^H\f$** (`ormbr('P', Right, ConjTrans)`): one-warp `ormbr_apply_p_cta` under the same
  condition ([`ormbr.cc:L50-L66`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L50-L66), [`:L429-L431`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L429-L431)). Otherwise \f$b\f$ conjugated row reflectors form
  the columns of \f$W\f$ ([`:L186-L194`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L186-L194)), \f$G_i \cdots G_{i+b-1} = I - W T_W W^H\f$ [7]
  ([`:L287-L292`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L287-L292)), and \f$C \leftarrow C - (CW)\,T_W^H W^H\f$ ([`:L321-L327`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L321-L327)), blocks in reverse
  order ([`:L336-L338`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/ormbr/ormbr.cc#L336-L338)).

## gesvd: the Hermitian shortcut

If \f$A = A^H = W\Lambda W^H\f$, then
\f$A = \sum_j |\lambda_j|\, w_j (\operatorname{sign}\lambda_j\, w_j)^H\f$. Sorting \f$|\lambda|\f$
descending by permutation \f$\rho\f$: \f$\sigma_j = |\lambda_{\rho(j)}|\f$,
\f$u_j = w_{\rho(j)}\f$, \f$v_j = \operatorname{sign}(\lambda_{\rho(j)})\, w_{\rho(j)}\f$
([`src/ops/gesvd/kernels/gesvd_blocked.cc:L393-L412`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L393-L412), [`:L436`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L436), [`:L471-L482`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L471-L482)), with \f$W\f$ from
`syev_cta` or `syev_blocked` ([`:L1211-L1218`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvd_blocked.cc#L1211-L1218); @ref algo_syev).

## gesvd: one-sided Jacobi

Rotate column pairs until \f$AJ\f$ (\f$J\f$ the product of all rotations) has orthogonal columns
[10]; then \f$\sigma_c = \|(AJ)_{:,c}\|\f$, \f$U_{:,c} = (AJ)_{:,c}/\sigma_c\f$, \f$V = J\f$. With
\f$a_{pq} = A_{:,p}^H A_{:,q}\f$ ([`src/ops/gesvd/kernels/gesvdj_cta.cc:L431-L461`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L431-L461)), the rotation
on \f$p < q\f$ diagonalises the \f$2 \times 2\f$ Gram block [2] ([`:L489-L505`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L489-L505), [`:L523-L526`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L523-L526),
[`:L553-L576`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L553-L576)):
\f[
\zeta = \frac{a_{qq} - a_{pp}}{2|a_{pq}|},\ \
t = \frac{\operatorname{sign}\zeta}{|\zeta| + \sqrt{1+\zeta^2}},\ \
c = \frac{1}{\sqrt{1+t^2}},\ \ s = tc,\ \ \delta = \frac{\bar a_{pq}}{|a_{pq}|},
\f]
\f[
[A_{:,p}\ A_{:,q}] \leftarrow [A_{:,p}\ \ \delta A_{:,q}] \begin{bmatrix} c & s \\ -s & c \end{bmatrix}\ \ (\text{same for } V),
\qquad a_{pp} \mathrel{-}= t|a_{pq}|,\ \ a_{qq} \mathrel{+}= t|a_{pq}|.
\f]
Real data use \f$\delta = 1\f$ and signed \f$a_{pq}\f$ for \f$|a_{pq}|\f$ ([`:L492-L495`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L492-L495)).

**Pivot rule.** Rotate only if \f$|a_{pq}| > \max(\mathrm{realmin},\ c_{\mathrm{tol}}\, n\varepsilon\sqrt{a_{pp}a_{qq}})\f$,
\f$c_{\mathrm{tol}}\f$ = `tol_multiplier` ([`:L231`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L231), [`:L485-L487`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L485-L487)). This relative test bounds
\f$|\delta\sigma_j|/\sigma_j\f$ by \f$O(\varepsilon)\max_k \kappa(A_k D_k^{-1})\f$ over the iterates,
\f$D_k\f$ their column norms; in practice this is near \f$\kappa(AD^{-1})\f$ [8, 9]. A sweep is
\f$n'-1\f$ rounds of \f$n'/2\f$ disjoint round-robin pairs, \f$n'\f$ = \f$n\f$ rounded up to even
([`:L103-L117`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L103-L117), [`:L224`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L224), [`:L252-L253`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L252-L253)).

```text
A <- 2^-l A, l = round((log2 max a_cc + log2 min a_cc>0)/4), c < min(n,32)   (1)
up to 30 sweeps: recompute a_cc = ||A_c||^2; rotate every pivot pair;
  stop after two consecutive sweeps with no rotation                         (2)
sigma_c = ||A_c|| 2^l ranked descending; U_c = A_c/||A_c||; V^H = J^H        (3)
```

Code (`gesvdj_cta.cc`): (1) [`358-388`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L358-L388); (2) [`583-587`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L583-L587); (3) [`616-664`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L616-L664), [`776-783`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L776-L783).

Columns of \f$U\f$ with \f$\sigma_c \le \varepsilon\sigma_{\max}\f$ ([`:L640`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L640); times
`zero_sigma_multiplier`, default 1) and columns \f$n{+}1..m\f$ of a full \f$U\f$ are filled from
coordinate axes by two passes of modified Gram-Schmidt ([`:L707-L724`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L707-L724)), accepting an axis whose
squared residual norm \f$\lVert r\rVert_2^2\f$ exceeds \f$1/(2m)\f$ ([`:L685`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L685), [`:L732`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/gesvd/kernels/gesvdj_cta.cc#L732)).

## gesvd: references

1. G. Golub, W. Kahan, "Calculating the singular values and pseudo-inverse of a matrix", *J. SIAM
   Ser. B Numer. Anal.* 2(2):205-224, 1965. https://doi.org/10.1137/0702016
2. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins, 2013.
3. J. Demmel, W. Kahan, "Accurate singular values of bidiagonal matrices", *SIAM J. Sci. Stat.
   Comput.* 11(5):873-912, 1990. https://doi.org/10.1137/0911052
4. LAPACK `dbdsqr`. https://netlib.org/lapack/explore-html/db/dcc/dbdsqr_8f_source.html
5. O. Marques, J. Demmel, P. B. Vasconcelos, "Bidiagonal SVD computation via an associated
   tridiagonal eigenproblem", LAWN 295, 2018; LAPACK `dbdsvdx`.
   https://netlib.org/lapack/explore-html/d2/dd4/dbdsvdx_8f_source.html
6. LAPACK `dlabrd`, `zlarfg`. https://netlib.org/lapack/explore-html/dc/d89/dlabrd_8f_source.html,
   https://www.netlib.org/lapack/explore-html/dc/d23/zlarfg_8f_source.html
7. R. Schreiber, C. Van Loan, "A storage-efficient WY representation for products of Householder
   transformations", *SIAM J. Sci. Stat. Comput.* 10(1):53-57, 1989.
   https://doi.org/10.1137/0910005
8. J. Demmel, K. Veselić, "Jacobi's method is more accurate than QR", *SIAM J. Matrix Anal. Appl.*
   13(4):1204-1245, 1992. https://doi.org/10.1137/0613074
9. Z. Drmač, K. Veselić, "New fast and accurate Jacobi SVD algorithm. I", *SIAM J. Matrix Anal.
   Appl.* 29(4):1322-1342, 2008. https://doi.org/10.1137/050639193
10. M. R. Hestenes, "Inversion of matrices by biorthogonalization and related results", *J. SIAM*
    6(1):51-90, 1958. https://doi.org/10.1137/0106005
