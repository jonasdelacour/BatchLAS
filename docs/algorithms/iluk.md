# iluk: incomplete LU with level of fill {#algo_iluk}

> **Status:** current · checked 2026-11-01.

`iluk_factorize` computes \f$A_b \approx L_b U_b\f$ for a batch of sparse matrices that share one CSR
pattern: row-wise Gaussian elimination restricted to a level-of-fill pattern, followed by ILUT-style
row thresholding and a pivot floor. `iluk_apply` solves \f$L_b U_b x = r\f$ by level-scheduled
substitution. Measurements: @ref perf_iluk.

## iluk: problem and notation

- \f$A \in \mathbb{F}^{n \times n}\f$, \f$\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}\f$, in CSR. Every
  active batch item must have the same pattern \f$\mathcal{S}(A)\f$ (device check
  [`src/ops/iluk/iluk.cc:798-815`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L798-L815), called from `iluk_factorize` at [`1220`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1220)). Columns are assumed
  sorted; this is not checked. \f$a_{i\cdot}\f$ is row \f$i\f$, \f$\mathrm{nnz}(a_{i\cdot})\f$ its count.
- One CSR factor holds \f$L\f$ (entries \f$l_{ij}\f$) strictly below the diagonal, with unit diagonal
  implied, and \f$U\f$ (entries \f$u_{ij}\f$) on and above it. \f$A = LU - R\f$, where \f$R\f$ is everything
  discarded.
- No pivoting: rows and columns stay in the order of \f$A\f$.

| Symbol | `ILUKParams` field | Default |
| --- | --- | --- |
| \f$k\f$ | `levels_of_fill` | 0 |
| \f$\sigma\f$ | `diagonal_shift` | \f$10^{-8}\f$ |
| \f$\tau\f$ | `drop_tolerance` | \f$10^{-4}\f$ |
| \f$p\f$ | `fill_factor` (ratio to \f$\mathrm{nnz}(a_{i\cdot})\f$) | 10 |
| \f$\delta\f$ | `diag_pivot_threshold` | 0.1 |
| MILU | `modified_ilu` | true |

## iluk: the symbolic level-of-fill pattern

Initial levels are \f$\mathrm{lev}_{ij} = 0\f$ if \f$(i,j) \in \mathcal{S}(A)\f$ or \f$i = j\f$, and
\f$\infty\f$ otherwise ([`src/ops/iluk/iluk.cc:216-223`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L216-L223)). Row \f$i\f$ visits, in ascending order, every
column \f$j < i\f$ already in its level map, including fill created earlier in the same row. Each entry
\f$(j,c)\f$, \f$c > j\f$, of the finished row \f$j\f$ proposes

\f[
\mathrm{lev}_{ic} \leftarrow \min\bigl(\mathrm{lev}_{ic},\ \max(\mathrm{lev}_{ij}, \mathrm{lev}_{jc}) + 1\bigr).
\f]

A proposal above \f$k\f$ is never inserted ([`src/ops/iluk/iluk.cc:225-239`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L225-L239)). The pattern is
\f$\mathcal{P}_k = \{(i,j) : \mathrm{lev}_{ij} \le k\}\f$, with row slice
\f$\mathcal{P}_k(i) = \{c : (i,c) \in \mathcal{P}_k\}\f$.

- This is the **max** rule [1, eq. (10.18)]. Saad's Def. 10.5 uses the **sum** rule
  \f$\min(\mathrm{lev}_{ic}, \mathrm{lev}_{ij} + \mathrm{lev}_{jc} + 1)\f$ [1, eq. (10.17)]. Since
  \f$\max(a,b) \le a + b\f$ for \f$a, b \ge 0\f$, induction gives
  \f$\mathrm{lev}^{\max} \le \mathrm{lev}^{\mathrm{sum}}\f$. So \f$\mathcal{P}_k\f$ equals Saad's
  ILU(\f$k\f$) pattern for \f$k \le 1\f$ and contains it for \f$k \ge 2\f$.
- \f$\mathcal{P}_0 = \mathcal{S}(A) \cup \{(i,i)\}\f$, which is ILU(0).
- The symbolic phase runs once, on the shared pattern. The numeric phase only removes entries, so
  \f$|\mathcal{P}_k|\f$ bounds the final nnz (`iluk_buffer_size`, [`src/ops/iluk/iluk.cc:1294-1297`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1294-L1297)).

## iluk: the numeric IKJ elimination

This is row-wise (IKJ) Gaussian elimination [1, Alg. 10.3] on \f$\mathcal{P}_k\f$. A working row
\f$w\f$ over \f$\mathcal{P}_k(i)\f$ starts at zero and receives \f$a_{i\cdot}\f$
([`src/ops/iluk/iluk.cc:426-437`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L426-L437)). Then, for each \f$j \in \mathcal{P}_k(i)\f$ with \f$j < i\f$, in
ascending order:

\f[
w_j \leftarrow \frac{w_j}{u_{jj}}, \qquad
w_c \leftarrow w_c - w_j\, u_{jc} \quad \text{for } c > j,\ c \in \mathcal{P}_k(j) \cap \mathcal{P}_k(i)
\f]

([`src/ops/iluk/iluk.cc:463-471`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L463-L471)). Updates with \f$c \notin \mathcal{P}_k(i)\f$ are never formed; they
contribute to \f$R\f$. Row \f$j\f$ is final when it is used: it has already been thresholded and
stabilized.

```text
for i = 1..n
    w := a(i,:) on P_k(i)
    for j in P_k(i), j < i, ascending
        w(j) := w(j) / u(j,j)
        w(c) := w(c) - w(j) * u(j,c)    for c > j in P_k(j) and P_k(i)
    threshold w; MILU adds dropped values to w(i)
    w(i) := stabilize(w(i))
    l(i,1:i-1) := w(1:i-1);  u(i,i:n) := w(i:n)
```

## iluk: thresholding and fill control

After row \f$i\f$ is eliminated, let \f$s_i = \max_{c \in \mathcal{P}_k(i)} |w_c|\f$, counting the
diagonal and the multipliers. The diagonal is always kept.

1. **Drop.** An off-diagonal \f$w_c\f$ with \f$|w_c| \le \tau \max(s_i, 1)\f$ is set to zero
   ([`src/ops/iluk/iluk.cc:166-182`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L166-L182)).
2. **Quota.** At most \f$q_i = \max\bigl(0, \lceil p \cdot \mathrm{nnz}(a_{i\cdot}) \rceil - 1\bigr)\f$
   off-diagonals survive. The largest \f$|w_c|\f$ are kept, and ties go to the smaller column
   ([`src/ops/iluk/iluk.cc:184-202`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L184-L202)).
3. **MILU.** Each dropped value is added to \f$w_i\f$ ([`src/ops/iluk/iluk.cc:174-176`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L174-L176), [`196-198`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L196-L198)).

| | ILUT\f$(m,\tau)\f$ [1, §10.4.1, Alg. 10.6] | `iluk` |
| --- | --- | --- |
| Drop scale | \f$\tau \lVert a_{i\cdot} \rVert\f$ (original row) | \f$\tau \max(s_i, 1)\f$ (eliminated row) |
| When | after each division and after the row | after the row only |
| Count cap | \f$m\f$ largest in \f$L\f$ and \f$m\f$ in \f$U\f$ | \f$q_i\f$ over \f$L\f$ and \f$U\f$ together |
| Pattern | none beyond the cap | \f$\mathcal{P}_k\f$ first |

Classical MILU [3] subtracts the row sum of \f$R\f$ from \f$u_{ii}\f$ [1, §10.3.5]. Here, two
differences break that:

- Only the values dropped in steps 1-2 are compensated. Updates never formed because they fall
  outside \f$\mathcal{P}_k\f$ are not.
- Dropped \f$L\f$ multipliers are added too. Dropping \f$l_{ic}\f$ changes row \f$i\f$ of \f$LU\f$ by
  \f$-l_{ic} u_{c\cdot}\f$, whose row sum is \f$-l_{ic}\, u_{c\cdot} e\f$, not \f$-l_{ic}\f$. So in
  general \f$LUe \ne Ae\f$.

## iluk: pivot stabilization

Let \f$s'_i\f$ be the row maximum after thresholding and
\f$\theta_i = \max\bigl(|\sigma|,\ \delta \max(s'_i, 1)\bigr)\f$. The first matching case wins:

\f[
u_{ii} =
\begin{cases}
w_i & |w_i| \ge \theta_i,\ w_i \ne 0, \\
w_i + \sigma & \sigma \ne 0,\ |w_i + \sigma| \ge \theta_i,\ w_i + \sigma \ne 0, \\
\sigma & \sigma \ne 0,\ |\sigma| = \theta_i, \\
\theta_i & \text{otherwise, if } \theta_i > 0.
\end{cases}
\f]

([`src/ops/iluk/iluk.cc:49-80`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L49-L80), applied at [`477-479`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L477-L479)). If no case matches, which needs
\f$\sigma = \delta = 0\f$ and \f$w_i = 0\f$, the call throws `convergence_error`.

Why the floor: plain ILU with a static pattern cannot break down on an M-matrix [2; 1, Thm. 10.2].
Value-based dropping and MILU compensation, both on by default, carry no such guarantee for general
\f$A\f$. The floor gives \f$1/|u_{ii}| \le 1/\theta_i\f$, which bounds the growth this pivot causes in
the later multipliers \f$l_{ji}\f$ and in the \f$U\f$ solve.

## iluk: batching and the host/device split

There is no flat route: no `choice.hh`, no tuned table, no `BATCHLAS_ILUK_ROUTE`.
`iluk_prefer_device` takes the device path when the batch has at least 32 items, and
`BATCHLAS_ILUK_DEVICE=0|1` pins the choice ([`src/ops/iluk/iluk.cc:304-312`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L304-L312)). Both paths apply the
same rules, including the shared quota tie-break.

- **Host.** `compute_iluk` runs the algorithm above, one item at a time.
- **Device.** `build_iluk_symbolic` turns \f$\mathcal{P}_k\f$ into per-row step lists
  ([`src/ops/iluk/iluk.cc:712-740`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L712-L740)). Each step holds a target slot, a pivot-diagonal slot and the
  (source, destination) update pairs. The device does only the arithmetic. Updates whose source was
  dropped stay in the list as zero no-ops.
- **Shared pattern.** An entry survives if any active item kept it, and an item that dropped it
  stores zero there (host [`src/ops/iluk/iluk.cc:510-549`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L510-L549), device [`1023-1034`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1023-L1034)).

Row \f$i\f$ of the elimination depends on the rows \f$j < i\f$ in its \f$L\f$ pattern, which gives the
depth

\f[
d_L(i) = \max_{j < i,\ (i,j) \in \mathcal{P}_k} \bigl(d_L(j) + 1\bigr), \qquad d_L(i) = 0 \text{ if no such } j
\f]

([`src/ops/iluk/iluk.cc:273-285`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L273-L285), called on \f$\mathcal{P}_k\f$ at [`742`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L742)). Rows of equal depth are
independent [1, §11.6.3]. There is one `ILUKEliminateKernel` launch per depth, with one work-item per
(row, item). Each launch applies the step list, the thresholding and the pivot floor
([`src/ops/iluk/iluk.cc:943-1005`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L943-L1005)).

## iluk: the preconditioner apply

For each (item, right-hand-side column), one work-group of 128 work-items solves \f$LUx = r\f$
([`src/ops/iluk/iluk.cc:1452-1454`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1452-L1454)):

\f[
y_i = r_i - \sum_{j < i} l_{ij}\, y_j, \qquad
x_i = \frac{1}{\tilde u_{ii}} \Bigl( y_i - \sum_{j > i} u_{ij}\, x_j \Bigr)
\f]

(forward [`src/ops/iluk/iluk.cc:1467-1483`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1467-L1483), backward [`1485-1504`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1485-L1504)). Both solves are level-scheduled on
the final pattern (host [`553-554`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L553-L554), device [`1069-1070`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1069-L1070)): the forward solve uses \f$d_L\f$, and the
backward solve uses \f$d_U(i) = \max_{j > i,\ (i,j)\ \text{stored}} (d_U(j) + 1)\f$, scanned in
decreasing row order ([`274-278`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L274-L278)). A group barrier separates levels, so the serial chain is the number
of levels, not \f$n\f$.

\f$\tilde u_{ii}\f$ re-applies the shift with strict inequalities ([`src/ops/iluk/iluk.cc:83-104`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L83-L104),
used at [`1495`](https://github.com/jonasdelacour/BatchLAS/blob/f37b12ee415cb887872c59b85cbd01a2d06586a3/src/ops/iluk/iluk.cc#L1495)). Because \f$|u_{ii}| \ge \theta_i \ge |\sigma|\f$, this is the identity unless
\f$|u_{ii}| = |\sigma|\f$. Then \f$u_{ii} + \sigma\f$ is applied, so a case-3 pivot \f$\sigma\f$
becomes \f$2\sigma\f$.

## iluk: references

1. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
   https://doi.org/10.1137/1.9780898718003; author copy
   https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf. Alg. 10.3 (IKJ ILU); §10.3.3, Def. 10.5
   and eqs. (10.17)-(10.18) (level of fill); §10.3.5 (MILU); §10.4.1, Alg. 10.6 (ILUT); Thm. 10.2;
   §11.6.3 (level scheduling).
2. J. A. Meijerink, H. A. van der Vorst, "An iterative solution method for linear systems of which
   the coefficient matrix is a symmetric M-matrix", *Math. Comp.* 31(137):148-162, 1977.
   https://doi.org/10.1090/S0025-5718-1977-0438681-4
3. I. Gustafsson, "A class of first order factorization methods", *BIT* 18(2):142-156, 1978.
   https://doi.org/10.1007/BF01931691
