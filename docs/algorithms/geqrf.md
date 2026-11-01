# geqrf: Householder QR, and forming and applying Q {#algo_geqrf}

> **Status:** current · checked 2026-11-01.

`geqrf` factors every matrix of a batch as \f$A = QR\f$, storing \f$Q\f$ as a product of Householder
reflectors in place of \f$A\f$; `ormqr` applies \f$Q\f$ or \f$Q^H\f$ to another matrix and `orgqr` forms
its leading columns. Small matrices run the unblocked algorithm in registers or local memory, larger
ones a blocked right-looking update of three GEMMs in compact WY form.
Measurements: @ref perf_qr.

## geqrf: problem and storage

For \f$A \in \mathbb{F}^{m \times n}\f$, \f$\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}\f$, \f$k = \min(m, n)\f$:

\f[
A = QR, \qquad Q = H_1 H_2 \cdots H_k, \qquad H_j = I - \tau_j v_j v_j^H,
\f]

with \f$Q\f$ unitary, \f$R\f$ upper trapezoidal, \f$v_j(1{:}j{-}1) = 0\f$ and \f$v_j(j) = 1\f$ [3].
Math indices are 1-based, code 0-based.

- On exit \f$R\f$ is the upper triangle of \f$A\f$, \f$v_j(j{+}1{:}m)\f$ overwrites \f$A(j{+}1{:}m, j)\f$, and
  \f$\tau_j\f$ is `tau[b*k + j]` for batch item \f$b\f$; `ormqr`, `orgqr`, `ormbr`, `sy2sb` read it.
- For complex data \f$\tau_j\f$ is complex, so \f$H_j\f$ is unitary but not Hermitian:
  \f$H_j^H = I - \bar\tau_j v_j v_j^H \ne H_j\f$.

## geqrf: the Householder reflector

`larfg` maps a column \f$(\alpha; x)\f$, \f$\alpha\f$ on the diagonal, to a real multiple of
\f$e_1\f$ [3, `zlarfg`] (`geqrf_larfg_scalars`, `src/ops/geqrf/geqrf_cta_device.hh:93-136`):

\f[
H^H \begin{bmatrix} \alpha \\ x \end{bmatrix} = \begin{bmatrix} \beta \\ 0 \end{bmatrix}, \qquad
\beta = -\operatorname{sign}(\operatorname{Re}\alpha)\,\lVert (\alpha; x) \rVert_2 \in \mathbb{R}, \qquad
\tau = \frac{\beta - \alpha}{\beta}, \qquad
v = \begin{bmatrix} 1 \\ x / (\alpha - \beta) \end{bmatrix}.
\f]

- The sign avoids cancellation in \f$\alpha - \beta\f$; real \f$\beta\f$ gives
  \f$1 \le \operatorname{Re}\tau \le 2\f$, \f$\lvert\tau - 1\rvert \le 1\f$ [3, `zlarfg`], [4].
  \f$\operatorname{Re}\alpha = \pm 0\f$ gives \f$\beta < 0\f$ (`:118`; LAPACK's `SIGN` agrees for \f$+0\f$ only).
- \f$x = 0\f$, \f$\alpha\f$ real: \f$\tau = 0\f$, \f$H = I\f$ (`:106`). In place of LAPACK's underflow
  loop, \f$\beta\f$ and \f$\tau\f$ come from \f$(\alpha; x)/s\f$, \f$s\f$ the column's largest
  \f$\lvert\operatorname{Re}\rvert\f$ or \f$\lvert\operatorname{Im}\rvert\f$ (`:110-126`); \f$x\f$ is
  divided by \f$\alpha - \beta\f$ when its reciprocal is not finite (`:128-133`).

## geqrf: the unblocked factorization

`geqr2` eliminates one column at a time, applying \f$H_j^H\f$ to the columns right of it [3, `zgeqr2`]:

\f[
(\tau_j, \beta_j, v_j) = \operatorname{larfg}\big(A(j{:}m, j)\big), \qquad
A(j{:}m, c) \leftarrow A(j{:}m, c) - \bar\tau_j\, v_j \big(v_j^H A(j{:}m, c)\big), \quad c > j.
\f]

```text
for j = 1 .. k:
  s, sigma <- column absmax, scaled tail sum of squares  (row reductions)
  (tau, beta, f) <- larfg(A(j,j), s, sigma)              f = 1/(alpha - beta)
  A(j,j) <- beta;  A(j+1:m, j) *= f;  tau_j <- tau
  for c = j+1 .. n:                                      (a sub-group per column)
    w <- v^H A(j:m, c);  A(j:m, c) -= conj(tau) * w * v
```

`src/ops/geqrf/geqrf_cta_device.hh:226-310` (reductions `:232-253`, apply `:291-309`). Cost
\f$2n^2(m - n/3)\f$ real flops for \f$m \ge n\f$, about four times that for complex [1, §5.2].

## geqrf: the compact WY form

A product of \f$r\f$ reflectors is \f$H_1 \cdots H_r = I - V T V^H\f$, with
\f$V = [v_1, \dots, v_r]\f$ unit lower trapezoidal and \f$T\f$ \f$r \times r\f$ upper triangular [6],
compacting the WY form of [5]. Appending one reflector:

\f[
(I - V T V^H)(I - \tau v v^H) = I -
\begin{bmatrix} V & v \end{bmatrix}
\begin{bmatrix} T & -\tau\, T V^H v \\ 0 & \tau \end{bmatrix}
\begin{bmatrix} V & v \end{bmatrix}^H .
\f]

`larft` (forward, columnwise) evaluates it column by column, as LAPACK's level-2 `zlarft` does [3]:

\f[
T_{1:j-1,\,j} = -\tau_j\, V_{:,1:j-1}^H v_j, \qquad
T_{1:j-1,\,j} \leftarrow T_{1:j-1,1:j-1}\, T_{1:j-1,\,j}, \qquad T_{jj} = \tau_j .
\f]

- The two steps are `src/kernels/common/larft_wy.hh:92-104` and `:107-115`; one work-group per
  matrix. \f$\tau_j = 0\f$ sets \f$T_{jj} = 0\f$ and leaves column \f$j\f$ at its zero fill (`:73-79`, `:84-90`).
- `pack_v_panel_batched` writes \f$V\f$ from the factored \f$A\f$: unit diagonal, zeros above (`:275-281`).

## geqrf: the blocked right-looking update

At panel offset \f$j_0\f$ with width \f$i_b = \min(n_b, k - j_0)\f$, let
\f$P = A(j_0{:}m, j_0{:}j_0{+}i_b{-}1)\f$ and \f$C = A(j_0{:}m, j_0{+}i_b{:}n)\f$. The panel's reflectors
give \f$Q_P^H = (I - V T V^H)^H = I - V T^H V^H\f$, so [3, `zgeqrf`]:

\f[
P \to (V, \tau) \text{ by geqr2}, \qquad T = \operatorname{larft}(V, \tau), \qquad
W_1 = V^H C, \quad W_2 = T^H W_1, \quad C \leftarrow C - V W_2 .
\f]

In `src/ops/geqrf/kernels/geqrf_blocked.cc`: panel loop `:262-267`, `geqr2` `:280-283`, \f$V\f$
`:299-300`, \f$T\f$ `:308-312`, the three products (routed `gemm`) `:324-333`.

- \f$n_b = 16\f$ for `double`, 32 otherwise, clamped to \f$k\f$ (`geqrf_blocked.cc:47-60`).
- \f$m < n\f$: \f$\lceil k/n_b \rceil\f$ panels (\f$k = m\f$ reflectors) whose updates reach column \f$n\f$.

## geqrf: stability

With hats for computed quantities and \f$u\f$ the unit roundoff, if \f$mn\,u\f$ is small then
\f$A + \Delta A = Q\hat R\f$, \f$Q\f$ the exactly unitary product of reflectors from the computed
\f$\hat v_j\f$, \f$\lVert\Delta a_j\rVert_2 \le \tilde\gamma_{mn} \lVert a_j\rVert_2\f$ per column
[2, Thm. 19.4]; `orgqr`'s \f$\hat Q\f$ is unitary to \f$O(mn\,u)\f$ [2, Ch. 19], [7]. The blocked (WY)
form has a bound of the same form by its own analysis [2, §19.5].

## geqrf: kernels and routing

`tuned/geqrf.<device>.txt` ranks the routes; `can_run` (`src/ops/geqrf/can_run.hh:28-44`) admits the
correct ones. No route takes a heterogeneous batch; native routes need a GPU and sub-group size 32.

| Route | `can_run` adds | Kernel and mapping |
| --- | --- | --- |
| `tiny` | \f$m = n \le\f$ `geqrf_tiny_max_n_for_slm` (\f$\le 32\f$) | one matrix per partition of \f$N \in \{8, 16, 32\}\f$ lanes; lane \f$i\f$ holds row \f$i\f$ in registers |
| `cta` | `geqrf_cta_fits`: \f$mn\f$ scalars fit the occupancy-scaled local memory | `geqr2_panel_device` on the matrix in local memory; a work-group per matrix, or a sub-group when \f$m, n \le 32\f$ and two or more fit a work-group (`geqrf_cta.cc:83-97`) |
| `blocked` | `geqrf_blocked_available`, CTA tier present | panel leaf, `larft`, three routed GEMMs |
| `vendor` | a vendor library; masked batches only on NETLIB | `backend::geqrf_vendor` |

- **`tiny`** reduces with XOR butterflies (`src/ops/geqrf/kernels/geqrf_tiny_device.hh:23-39`) and
  applies in column chunks: \f$y_k = \bar\tau \sum_r \bar v_r a_{rk}\f$ summed in row order, then
  \f$a_{rk} \mathrel{-}= v_r y_k\f$ (`src/ops/geqrf/kernels/geqrf_tiny.cc:205-251`).
- **`blocked` panel leaf** (`src/ops/geqrf/kernels/geqrf_cta.cc:426-436`): *Register* for `double`
  panels of width \f$\le 16\f$, height \f$\le 128\f$ (`:388-401`), one row per work-item
  (`src/ops/geqrf/geqrf_panel_reg_device.hh:154-214`); else *Resident* if the panel fits local
  memory (`geqrf_leaf_fits`); else *Global*, the same device function on global memory.

## ormqr: applying Q

`ormqr` overwrites \f$C\f$ with \f$QC\f$, \f$Q^H C\f$, \f$CQ\f$ or \f$CQ^H\f$ [3, `zunmqr`]. Grouping the
reflectors into blocks of width \f$n_b\f$, \f$Q = Q_1 Q_2 \cdots Q_p\f$ with
\f$Q_i = I - V_i T_i V_i^H\f$, and \f$\operatorname{op}(T) = T\f$ (NoTrans) or \f$T^H\f$ (ConjTrans):

\f[
\text{Left: } W_1 = V^H C,\; W_2 = \operatorname{op}(T) W_1,\; C \leftarrow C - V W_2; \qquad
\text{Right: } W_1 = C V,\; W_2 = W_1 \operatorname{op}(T),\; C \leftarrow C - W_2 V^H .
\f]

- Block \f$i\f$ holds reflectors \f$i_0{:}i_0{+}i_b{-}1\f$ and touches rows (Left) or columns (Right)
  \f$i_0{:}n_q\f$ of \f$C\f$, \f$n_q\f$ the order of \f$Q\f$. `src/ops/ormqr/kernels/ormqr_blocked.cc`:
  \f$V\f$, \f$T\f$ `:187-197`; Left `:200-224`; Right `:225-245`.
- Order: \f$QC = Q_1(\cdots(Q_p C))\f$ runs the last block first; (Left, ConjTrans) and
  (Right, NoTrans) run forward (`ormqr_blocked.cc:249-258`), as in `zunmqr`.
- Left: \f$W_2\f$ is a `trmm` for real types with \f$i_b \le 64\f$ where the tile kernel is enabled, or
  as `BATCHLAS_ORMQR_WY` pins it (`src/ops/ormqr/wy_spelling.hh:33-43`); otherwise, and always for
  Right (`ormqr_blocked.cc:238`), a `gemm`.
- In `src/ops/ormqr/ormqr.cc`: \f$n_b\f$ is the caller's hint clamped to \f$[1, k]\f$, else the tuning
  table (`:77-82`); routes `blocked` (GPU) and `vendor` (`:53-63`); complex `Trans` throws before
  routing, use `ConjTrans` (`:67-72`).
- `ormqx_cta` is not a route: `ormbr`, `gesvd` and `syev_cta` call it for square \f$n \le 32\f$ QR or
  QL factors, one rank-1 update per reflector (`src/ops/ormqr/kernels/ormqr_cta.cc:98-154`; limit `:559-560`).

## orgqr: forming Q

`orgqr` overwrites \f$A\f$ (\f$m \ge n\f$, \f$n\f$ reflectors) with the first \f$n\f$ columns of \f$Q\f$
[3, `zungqr`]:

\f[
Q_{:,1:n} = H_1 H_2 \cdots H_n \begin{bmatrix} I_n \\ 0 \end{bmatrix}.
\f]

- `blocked`: \f$C \leftarrow I_{m \times n}\f$, then the routed `ormqr` (Left, NoTrans), then
  \f$A \leftarrow C\f$ (`src/ops/orgqr/kernels/orgqr_blocked.cc:229-240`, `:248`, `:261-273`);
  \f$n_b\f$ as `geqrf` (`:40-53`).
- Each block updates all \f$n\f$ columns of \f$C\f$; `zungqr` skips the zero block of \f$[I_n; 0]\f$.
- `can_run` (`src/ops/orgqr/orgqr.cc:41-54`): `blocked` needs a GPU and \f$n \le m\f$; `vendor` a
  vendor library, masked batches only on NETLIB; neither takes a heterogeneous batch.

## geqrf: references

1. G. H. Golub, C. F. Van Loan. *Matrix Computations*, 4th ed., §5.2. Johns Hopkins University
   Press, 2013. https://www.press.jhu.edu/books/title/10678/matrix-computations
2. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Ch. 19. SIAM, 2002.
   https://doi.org/10.1137/1.9780898718027
3. LAPACK reference implementation: `zlarfg.f`, `zgeqr2.f`, `zgeqrf.f`, `zunmqr.f`, `zungqr.f`
   (https://github.com/Reference-LAPACK/lapack/tree/master/SRC); level-2 `zlarft.f` from v3.11.0
   (https://github.com/Reference-LAPACK/lapack/blob/v3.11.0/SRC/zlarft.f).
4. R. B. Lehoucq. *The Computation of Elementary Unitary Matrices*. LAPACK Working Note 72,
   UT-CS-94-233, 1995. https://www.netlib.org/lapack/lawnspdf/lawn72.pdf
5. C. Bischof, C. Van Loan. The WY representation for products of Householder matrices. *SIAM J.
   Sci. Stat. Comput.* 8(1):s2-s13, 1987. https://doi.org/10.1137/0908009
6. R. Schreiber, C. Van Loan. A storage-efficient WY representation for products of Householder
   transformations. *SIAM J. Sci. Stat. Comput.* 10(1):53-57, 1989. https://doi.org/10.1137/0910005
7. N. J. Higham. What is a QR factorization? 2020. https://nhigham.com/2020/11/10/what-is-a-qr-factorization/
