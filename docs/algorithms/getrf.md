# getrf: LU with partial pivoting {#algo_getrf}

> **Status:** current · checked 2026-11-01.

`getrf` factors every matrix of a batch as \f$PA = LU\f$ by Gaussian elimination with partial
pivoting. `getrs`, `gesv`, `getri` and `inv` reuse the factors to solve \f$AX = B\f$ or form
\f$A^{-1}\f$. All three native kernels run the same elimination step, held in registers, in
local memory, or blocked right-looking around a panel. Measurements are in @ref perf_lu.

## getrf: the factorization and the pivot list

For \f$A \in \mathbb{F}^{m \times n}\f$ and \f$r = \min(m, n)\f$, `getrf` overwrites \f$A\f$ with
\f$L\f$ (\f$m \times r\f$, unit lower trapezoidal, unit diagonal not stored) and \f$U\f$
(\f$r \times n\f$, upper trapezoidal), where

\f[
PA = LU, \qquad P = T_r \cdots T_2 T_1,
\f]

and \f$T_k\f$ swaps rows \f$k\f$ and \f$p_k \ge k\f$ [1, Ch. 3], [4]. \f$T_k\f$ also denotes the
transposition \f$(k\ p_k)\f$ acting on indices.

- `ipiv` is the interchange list \f$(p_1, \dots, p_r)\f$, not a permutation: 1-based and global
  (`src/ops/getrf/getrf_cta_device.hh:144`), int32 packed into the first half of the int64 span
  with stride \f$r\f$ (`src/ops/getrf/kernels/getrf_blocked.cc:280`,
  `src/ops/getrf/kernels/getrf_blocked.cc:289`).
- `info` is the smallest \f$k\f$ with \f$u_{kk} = 0\f$ exactly (no tolerance), or 0; the
  factorization still completes (`src/ops/getrf/getrf_cta_device.hh:161-165`).
- The pivot metric is \f$\operatorname{cabs1}(z) = |\operatorname{Re} z| + |\operatorname{Im} z|\f$
  as in `izamax` [6], not the modulus (`src/ops/getrf/getrf_cta_device.hh:34-41`); ties go to
  the smallest row. Unlike `izamax`, a NaN is never chosen
  (`src/ops/getrf/kernels/getrf_tiny.cc:140`) and cabs1 overflow is not rescaled.

## getrf: the elimination step

Step \f$k = 1, \dots, r\f$ on the partly reduced \f$A\f$ is `dgetf2` [4]:

\f[
\begin{aligned}
p_k &= \arg\max_{i \ge k} \operatorname{cabs1}(a_{ik}), &
a_{k,:} &\leftrightarrow a_{p_k,:}, \\
\ell_{ik} &= a_{ik} / a_{kk} \quad (i > k), &
a_{ij} &\leftarrow a_{ij} - \ell_{ik} a_{kj} \quad (i, j > k).
\end{aligned}
\f]

`getf2_panel_device` implements it:

| Step | What runs | Lines |
| --- | --- | --- |
| \f$p_k\f$ | strided scan, XOR butterfly per sub-group, scan over sub-group slots | `src/ops/getrf/getrf_cta_device.hh:103-139` |
| swap | all \f$n\f$ columns, finished \f$L\f$ columns included | `src/ops/getrf/getrf_cta_device.hh:149-157` |
| \f$a_{kk} = 0\f$ | the column below is zero too, so the update is a no-op | `src/ops/getrf/getrf_cta_device.hh:161-165` |
| \f$\ell_{ik}\f$ | times \f$1/a_{kk}\f$, or a divide when \f$1/a_{kk}\f$ is non-finite or zero (the role of `dgetf2`'s `sfmin` test [4]) | `src/ops/getrf/getrf_cta_device.hh:166-180` |
| update | rank-1, flattened over the \f$(m-k) \times (n-k)\f$ block | `src/ops/getrf/getrf_cta_device.hh:185-196` |

**Stability.** The computed factors satisfy \f$\hat L \hat U = PA + \Delta A\f$ with
\f$|\Delta A| \le \gamma_r |\hat L| |\hat U|\f$ (real arithmetic; a slightly larger constant for
complex) [2, Thm 9.3]. Pivoting bounds \f$|\ell_{ik}| \le 1\f$ for real data and
\f$\le \sqrt{2}\f$ under cabs1, so the growth factor
\f$\rho = \max|u_{ij}| / \max|a_{ij}|\f$ is at most \f$2^{r-1}\f$, or \f$(1+\sqrt{2})^{r-1}\f$
under cabs1 [2, Ch. 9]. The real bound is attained [7], [3], but growth stays small in
practice [3].

## getrf: lazy row relabelling in registers

The `tiny` kernel and the register panel leaf keep row \f$i\f$ in the registers of work-item
\f$i\f$ and never move it. Each work-item instead carries a label \f$\rho\f$, the row of
\f$PA\f$ that its data currently occupies. The swap of step \f$k\f$ rewrites only labels:

\f[
\rho \leftarrow
\begin{cases}
k & \rho = p_k, \\
p_k & \rho = k, \\
\rho & \text{otherwise}.
\end{cases}
\f]

Work-items with \f$\rho > k\f$ scale and update; the one with \f$\rho = k\f$ broadcasts the
pivot row. At the end each row is stored at row \f$\rho\f$. The label carries the whole row,
\f$L\f$ columns included, so the full-width swap is free.

| Phase | `tiny` | register leaf |
| --- | --- | --- |
| argmax, relabel | `src/ops/getrf/kernels/getrf_tiny.cc:138-149` | `src/ops/getrf/kernels/getrf_panel_reg_device.hh:67-96` |
| pivot row, scale, update | local memory (32-bit types) or shuffles (fp64): `src/ops/getrf/kernels/getrf_tiny.cc:151-211` | local memory: `src/ops/getrf/kernels/getrf_panel_reg_device.hh:99-127` |
| store at \f$\rho\f$ | `src/ops/getrf/kernels/getrf_tiny.cc:214-227` | `src/ops/getrf/kernels/getrf_panel_reg_device.hh:132-138` |

## getrf: the blocked right-looking form

The block width is \f$n_b = \min(32, r)\f$ (`src/ops/getrf/kernels/getrf_blocked.cc:40-56`).
At column \f$j_0\f$, with \f$b = \min(n_b, r - j_0)\f$, partition the unreduced trailing part
as \f$\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}\f$ with
\f$A_{11} \in \mathbb{F}^{b \times b}\f$, and let \f$L_{\text{prev}}\f$ be the finished
\f$L\f$ columns to its left. One step is `dgetrf`'s loop [5]:

\f[
\begin{aligned}
\text{(P)}\quad & P_1 \begin{bmatrix} A_{11} \\ A_{21} \end{bmatrix}
               = \begin{bmatrix} L_{11} \\ L_{21} \end{bmatrix} U_{11}, \\
\text{(S)}\quad & \begin{bmatrix} A_{12} \\ A_{22} \end{bmatrix} \leftarrow
               P_1 \begin{bmatrix} A_{12} \\ A_{22} \end{bmatrix}, \qquad
               L_{\text{prev}} \leftarrow P_1 L_{\text{prev}}, \\
\text{(T)}\quad & U_{12} = L_{11}^{-1} A_{12}, \\
\text{(G)}\quad & A_{22} \leftarrow A_{22} - L_{21} U_{12},
\end{aligned}
\f]

and the next step repeats on \f$A_{22}\f$. In exact arithmetic it chooses the same pivots as the
unblocked step. Reference `dgetrf` factors (P) with the recursive `dgetrf2`; here (P) is
unblocked. Finished columns are never read again, so their interchanges commute with later
steps and can be postponed: the columns of the block ending at \f$c\f$ receive
\f$T_r \cdots T_{c+1}\f$ after the loop.

| Phase | Implementation | Lines |
| --- | --- | --- |
| (P) | register leaf if selected and `getrf_panel_reg_fits`, else `getf2_panel_device` | `src/ops/getrf/kernels/getrf_blocked.cc:322-328`, `src/ops/getrf/kernels/getrf_cta.cc:363-384` |
| (S) right | local-memory gather or `laswp` walk (same permutation) | `src/ops/getrf/kernels/getrf_blocked.cc:345-355` |
| (T) | routed public `trsm`: left, lower, unit | `src/ops/getrf/kernels/getrf_blocked.cc:360-363` |
| (G) | routed public `gemm`, \f$\alpha = -1\f$, \f$\beta = 1\f$ | `src/ops/getrf/kernels/getrf_blocked.cc:369-373` |
| (S) left | deferred by default (gather, else walk; mode `src/ops/getrf/kernels/getrf_blocked.cc:261-263`), else in-loop `src/ops/getrf/kernels/getrf_blocked.cc:335-340` | `src/ops/getrf/kernels/getrf_blocked.cc:381-402` |

## getrf: kernels and routing

The routes are `tiny`, `cta`, `blocked` and `vendor` (`src/ops/getrf/choice.hh`). `can_run`
(`src/ops/getrf/getrf.cc:46-73`) admits the native routes only on a GPU with sub-group size 32,
for a homogeneous batch, and never on NETLIB. Among the admitted routes,
`tuned/getrf.<device>.txt` decides; `BATCHLAS_GETRF_ROUTE` pins one.

| Route | One matrix per | Storage | `can_run` beyond the native term |
| --- | --- | --- | --- |
| `tiny` | sub-group partition (4-32 lanes), row \f$i\f$ in lane \f$i\f$ | registers | square, \f$n \le 32\f$ (complex double 16), `max_wg >= 64` |
| `cta` | work-group, or one sub-group of a packed group when \f$m, n \le 32\f$ | local memory, odd `ld` | `getrf_cta_max_n_for_slm` / `getrf_cta_fits_rect`, occupancy-scaled |
| `blocked` | work-group per panel | global | type available, panel argmax slots fit; any \f$m \times n\f$ |
| `vendor` | cuBLAS, rocSOLVER, LAPACKE | | square on CUDA; masked batch only on NETLIB |

## getrs: solving with the factors

From \f$PA = LU\f$:

\f[
AX = B:\ X = U^{-1} L^{-1} (P B), \qquad
A^{T} X = B:\ X = P^{T} L^{-T} U^{-T} B,
\f]

and \f$A^{H}\f$ is analogous. Forming \f$PB\f$ applies \f$T_1\f$ first (a forward walk);
forming \f$P^T B\f$ applies \f$T_r\f$ first (a backward walk). This is `dgetrs` with `dlaswp`
increments \f$\pm 1\f$ [8].

- `blocked`: a local-memory gather or the `laswp` walk (same permutation), plus two routed
  `trsm` calls (`src/ops/getrs/kernels/getrs_native.cc:301-332`); the walk is
  `src/kernels/common/lu_laswp.hh:58-79`.
- `cta` (fused): one work-group per matrix keeps \f$B\f$ and one \f$b_s \times b_s\f$ diagonal
  block in local memory (\f$b_s = 16\f$, or 32 for \f$n \ge 1024\f$;
  `src/ops/getrs/kernels/getrs_fused.cc:39-46`). Lane \f$i\f$ owns row \f$i\f$ of the block and
  substitutes by sub-group broadcast; the rows below update in parallel:

\f[
y_J \leftarrow L_{JJ}^{-1} y_J, \qquad y_I \leftarrow y_I - L_{IJ} y_J \quad (I > J),
\f]

  then the same backward with \f$U\f$. The code: \f$PB\f$
  `src/ops/getrs/kernels/getrs_fused.cc:269-275`, \f$L\f$ `src/ops/getrs/kernels/getrs_fused.cc:279-336`,
  \f$U\f$ `src/ops/getrs/kernels/getrs_fused.cc:339-396`, and \f$P^T\f$ in the transposed kernel
  `src/ops/getrs/kernels/getrs_fused.cc:592-598`.
- `can_run` (`src/ops/getrs/getrs.cc:42-60`): `cta` needs \f$n_{rhs} \le 8\f$ and
  \f$n \cdot n_{rhs}\f$ within the local-memory cap.

## gesv: factor and solve

- `blocked`: the public `getrf`, then the public `getrs` (`src/ops/gesv/gesv.cc:64-70`).
- `tiny` (\f$n \le 32\f$, complex double 16; \f$n_{rhs} \le 4\f$): eliminates on the augmented
  \f$[A \mid B]\f$. The rows of \f$B\f$ carry the same labels \f$\rho\f$, so the elimination ends
  at \f$[U \mid Y]\f$ with \f$Y = L^{-1} P B\f$, with no `laswp`
  (`src/ops/gesv/kernels/gesv_tiny.cc:220-261`). Back substitution is column-oriented:

\f[
x_i = y_i / u_{ii}, \qquad y_k \leftarrow y_k - u_{ki} x_i \ \ (k < i), \qquad i = n, \dots, 1
\f]

  (`src/ops/gesv/kernels/gesv_tiny.cc:264-299`). A zero \f$u_{ii}\f$ gives \f$x_i = 0\f$ and
  `info` reports it (LAPACK leaves \f$X\f$ undefined). \f$X\f$ is stored unpermuted, at row
  \f$\rho\f$.
- There is no vendor route. `can_run` is at `src/ops/gesv/gesv.cc:44-55`.

## getri: the inverse from the factors

\f$A^{-1} = U^{-1} L^{-1} P\f$. The `blocked` route sets \f$C := P\f$, then
\f$C \leftarrow L^{-1} C\f$ and \f$C \leftarrow U^{-1} C\f$, both by routed `trsm`
(`src/ops/getri/kernels/getri_blocked.cc:230-245`). Row \f$i\f$ of \f$P\f$ holds its 1 in
column \f$\pi(i) = T_1(T_2(\cdots T_r(i)))\f$: \f$i\f$ walked backwards through the list
(`src/ops/getri/kernels/getri_blocked.cc:102-113`). `info` is the first exact
\f$u_{ii} = 0\f$ (`src/ops/getri/kernels/getri_blocked.cc:115-125`).

- Unlike `dgetri` [9] (`dtrtri` on \f$U\f$, solve \f$XL = U^{-1}\f$, swap columns), no
  triangular inverse and no workspace; the cost is two \f$n\f$-column `trsm` calls. \f$C\f$ must
  not alias \f$A\f$
  (`src/ops/getri/kernels/getri_blocked.cc:197-202`).
- The routes are `blocked` and `vendor` (cuBLAS `getriBatched`, rocSOLVER, LAPACKE);
  `can_run` is at `src/ops/getri/getri.cc:48-60`.

## inv: composition

`inv` copies \f$A\f$ into its workspace, factors the copy with `getrf`, and writes
\f$A^{-1}\f$ with `getri` (`src/ops/inv/inv.cc:55-59`). Each child routes independently.

## getrf: references

1. G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press,
   2013, ISBN 978-1-4214-0794-4. https://www.press.jhu.edu/books/title/10678/matrix-computations
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
   https://doi.org/10.1137/1.9780898718027
3. M. E. Myers, R. A. van de Geijn, *Advanced Linear Algebra: Foundations to Frontiers*, §6.4.5,
   "Is LU with partial pivoting stable?".
   https://www.cs.utexas.edu/~flame/laff/alaff/chapter06-LUpiv-is-it-stable.html
4. LAPACK reference `dgetf2.f`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetf2.f
5. LAPACK reference `dgetrf.f`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetrf.f
6. Reference BLAS `izamax.f90`. https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/izamax.f90
7. N. J. Higham, D. J. Higham, "Large growth factors in Gaussian elimination with pivoting",
   *SIAM J. Matrix Anal. Appl.* 10(2):155-164, 1989. https://doi.org/10.1137/0610012
8. LAPACK reference `dgetrs.f`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetrs.f
9. LAPACK reference `dgetri.f`. https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetri.f
