# STEQR Mathematics and Pseudoalgorithm

This note describes the mathematics implemented by BatchLAS `steqr`, `steqr_wg`, and `steqr_cta` for real symmetric tridiagonal eigendecomposition.

The solver takes a batch of tridiagonal matrices

$$
T = \operatorname{tridiag}(e_1, \dots, e_{n-1}; d_1, \dots, d_n)
$$

with diagonal entries $d_i$ and off-diagonal entries $e_i$, and computes eigenvalues

$$
T = Q \Lambda Q^T,
$$

where $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ and, if requested, $Q$ is the orthogonal eigenvector matrix.

The implementation is an implicit shifted QR/QL method with deflation, specialized for batched execution. The workgroup path and CTA path use the same mathematical core but differ in how they represent the active subproblems, how they book-keep splits, and how they accumulate the local orthogonal transformations.

## 1. Problem Representation

For each matrix in the batch, the solver stores:

$$
T =
\begin{bmatrix}
d_1 & e_1 &        &        & 0 \\
e_1 & d_2 & e_2    &        &   \\
    & e_2 & d_3    & \ddots &   \\
    &     & \ddots & \ddots & e_{n-1} \\
0   &     &        & e_{n-1} & d_n
\end{bmatrix}.
$$

The algorithm never forms dense similarity transforms on $T$. Instead, it uses Givens rotations to chase a bulge through the tridiagonal structure while preserving symmetry.

## 2. Deflation Criterion

If an off-diagonal entry is sufficiently small, the matrix splits into independent subproblems. BatchLAS uses the LAPACK-style relative deflation test

$$
|e_i|^2 \le \varepsilon^2 |d_i| |d_{i+1}| + \mathrm{safmin},
$$

where $\varepsilon$ is machine precision and $\mathrm{safmin}$ is the safe minimum.

If the inequality holds, the implementation sets

$$
e_i \leftarrow 0,
$$

and the tridiagonal matrix splits as

$$
T =
\begin{bmatrix}
T_{11} & 0 \\
0      & T_{22}
\end{bmatrix}.
$$

Important implementation detail:

- The public parameter `zero_threshold` currently does not control the actual convergence test.
- Both `steqr_wg` and `steqr_cta` use the relative criterion above.

## 3. Safe Scaling

Before a sweep on an active block, the implementation computes a block norm

$$
\|T\|_\infty^{(\mathrm{block})} \approx \max\left( |d_i|, |e_i| \right)
$$

over the active block and rescales if necessary to avoid overflow or underflow.

If the norm is too large, the block is scaled by

$$
\alpha = \frac{\mathrm{ssfmax}}{\|T\|},
$$

and if too small but nonzero, by

$$
\alpha = \frac{\mathrm{ssfmin}}{\|T\|}.
$$

After convergence of the active block, the entries are rescaled back by $\alpha^{-1}$.

## 4. Shift Selection

Each implicit step uses a shift $\mu$ derived from a $2 \times 2$ trailing or leading principal block, depending on whether the iteration is QR or QL.

### 4.1 Wilkinson Shift

Given

$$
B = \begin{bmatrix} a & b \\ b & c \end{bmatrix},
$$

its eigenvalues are

$$
\lambda_{\pm} = \frac{a+c}{2} \pm \sqrt{\left(\frac{a-c}{2}\right)^2 + b^2}.
$$

The Wilkinson shift is the eigenvalue closer to the endpoint being converged:

$$
\mu =
\begin{cases}
\lambda_+ & \text{if } |\lambda_+ - c| < |\lambda_- - c|, \\
\lambda_- & \text{otherwise.}
\end{cases}
$$

In the QL case, the code forms the same expression from the leading $2 \times 2$ block, choosing the eigenvalue closest to the top endpoint instead.

### 4.2 LAPACK-Style Stable Shift

The CTA path also exposes a stable implicit-shift formula. For a local $2 \times 2$ block,

$$
g = \frac{d_2 - d_1}{2 e_1}, \qquad
r = \sqrt{g^2 + 1},
$$

and the shift is computed as

$$
\mu = d_1 - \frac{e_1}{g + \operatorname{copysign}(r, g)}.
$$

This avoids subtractive cancellation in the classical quadratic formula.

### 4.3 Path Differences

- `steqr_wg` always uses a Wilkinson shift.
- `steqr_cta` allows `Lapack` or `Wilkinson` through `SteqrShiftStrategy`.

## 5. Givens Rotations

The basic transformation is a Givens rotation

$$
G(c,s) =
\begin{bmatrix}
c & -s \\
s &  c
\end{bmatrix},
\qquad c^2 + s^2 = 1,
$$

chosen so that

$$
G(c,s)^T
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
r \\
0
\end{bmatrix}.
$$

BatchLAS obtains $(c,s,r)$ from `lartg(x,y)`.

In an implicit QR or QL step, the algorithm applies a sequence of such rotations as orthogonal similarities

$$
T \leftarrow G_k^T T G_k,
$$

which preserves symmetry and introduces then chases a single bulge.

## 6. Local Similarity Update on the Tridiagonal Band

The most important local operation is a similarity transformation with an adjacent Givens rotation acting on rows and columns $i$ and $i+1$.

Let

$$
\widehat G_i = \operatorname{diag}(I_{i-1}, G(c,s), I_{n-i-1}),
$$

where

$$
G(c,s) =
\begin{bmatrix}
c & -s \\
s &  c
\end{bmatrix}.
$$

Because $T$ is tridiagonal, the similarity

$$
T^+ = \widehat G_i^T T \widehat G_i
$$

changes only the local band around indices $i-1$, $i$, $i+1$, $i+2$.

### 6.1 Exact Local Window Being Updated

Before the update, the relevant local window is

$$
T_{\mathrm{loc}} =
\begin{bmatrix}
\alpha & \beta & 0 & 0 \\
\beta  & d_i   & e_i & 0 \\
0       & e_i   & d_{i+1} & \gamma \\
0       & 0     & \gamma & \delta
\end{bmatrix},
$$

where

$$
\alpha = d_{i-1}, \qquad \beta = e_{i-1}, \qquad \gamma = e_{i+1}, \qquad \delta = d_{i+2},
$$

whenever these entries exist.

After applying the similarity on the middle $2 \times 2$ block, the updated local structure is

$$
T_{\mathrm{loc}}^+ =
\begin{bmatrix}
\alpha & c\beta - s b_{\mathrm{prev}} & -s\beta & 0 \\
c\beta - s b_{\mathrm{prev}} & d_i^+ & e_i^+ & -s\gamma \\
-s\beta & e_i^+ & d_{i+1}^+ & c\gamma \\
0 & -s\gamma & c\gamma & \delta
\end{bmatrix},
$$

with the understanding that $b_{\mathrm{prev}} = 0$ for the first rotation in a sweep, and where the entry $-s\gamma$ is precisely the new bulge that must be chased to the next position.

The active data touched by one step is therefore

$$
(e_{i-1}, d_i, e_i, d_{i+1}, e_{i+1}),
$$

and nothing outside this local band changes.

### 6.2 Scalar Formulas Used by the Explicit Update

The workgroup path and the CTA `EXP` scheme use the explicit formulas

$$
d_i^+ = c(c d_i - e_i s) - s(e_i c - s d_{i+1}),
$$

$$
d_{i+1}^+ = c(c d_{i+1} + e_i s) + s(e_i c + s d_i),
$$

$$
e_i^+ = c(c e_i + s d_i) - s(c d_{i+1} + s e_i),
$$

$$
e_{i-1}^+ = c e_{i-1} - s b_{\mathrm{prev}},
$$

$$
e_{i+1}^+ = c e_{i+1},
$$

and the outgoing bulge is

$$
b_{\mathrm{new}} = -s e_{i+1}.
$$

In words: one step annihilates the current off-band bulge, updates one adjacent $2 \times 2$ pivot block, and creates the next bulge.

### 6.3 First Step Versus Interior Step

At the first step of a shifted QR or QL iteration, the rotation is generated from the shifted active corner

$$
\begin{bmatrix}
d_{\mathrm{edge}} - \mu \\
e_{\mathrm{edge}}
\end{bmatrix},
$$

so the bulge is created by the shift.

At every later step, the rotation is generated from

$$
\begin{bmatrix}
e_{i-1}^{(\mathrm{current})} \\
b_{\mathrm{prev}}
\end{bmatrix},
$$

so the bulge is not recreated from scratch; it is merely moved by one position.

## 7. Choosing QR Versus QL

For an active block with endpoints $d_{\ell}$ and $d_r$, BatchLAS chooses the direction that converges from the smaller-magnitude endpoint.

The rule is

$$
\text{use QR if } |d_{\ell}| \le |d_r|,
$$

$$
\text{use QL otherwise.}
$$

Interpretation:

- QR chases the bulge from top to bottom and converges the bottom eigenvalue first.
- QL chases the bulge from bottom to top and converges the top eigenvalue first.

This is the same policy used in LAPACK-style tridiagonal QR solvers.

## 8. Exact $2 \times 2$ Solve

Whenever a deflated block has size $2$, the solver stops iterating and diagonalizes it analytically.

For

$$
T_2 = \begin{bmatrix} a & b \\ b & c \end{bmatrix},
$$

the eigenvalues are $\lambda_\pm$ as above. If eigenvectors are requested, the implementation also computes a rotation

$$
Q_2 = \begin{bmatrix} c_2 & -s_2 \\ s_2 & c_2 \end{bmatrix}
$$

such that

$$
Q_2^T T_2 Q_2 = \operatorname{diag}(\lambda_1, \lambda_2).
$$

That rotation is accumulated into the global eigenvector matrix.

## 9. Split Structure and Book-Keeping

The solver does not iterate on the whole tridiagonal matrix at once after deflation. It iterates on maximal active blocks separated by zero off-diagonals.

### 9.1 Mathematical Definition of a Split

Define the zero set

$$
\mathcal Z = \{ i \in \{1,\dots,n-1\} : e_i = 0 \}.
$$

Then the active blocks are the maximal index intervals

$$
I_k = [s_k, t_k] \subseteq \{1,\dots,n\}
$$

such that

$$
e_j \neq 0 \quad \text{for all } j = s_k, \dots, t_k-1,
$$

and the interval is maximal with respect to this property. Equivalently,

$$
e_{s_k-1} = 0 \text{ or } s_k = 1,
\qquad
e_{t_k} = 0 \text{ or } t_k = n.
$$

Hence the exact decomposition is

$$
T = \operatorname{diag}(T_{I_1}, T_{I_2}, \dots, T_{I_p}).
$$

Each block $T_{I_k}$ may then be processed independently.

### 9.2 Workgroup Path Split Book-Keeping

The workgroup path performs the following book-keeping on every outer pass.

For each batch item, it scans the off-diagonal array from left to right and records every maximal run

$$
e_s, e_{s+1}, \dots, e_{t-1}
$$

of nonzero entries. That run corresponds to the diagonal index block

$$
[s, t].
$$

In code terms, the stored pair is `(start_ix, end_ix)` with `end_ix` exclusive in slice notation, so the local subproblem is

$$
d[s:t], \qquad e[s:t-1].
$$

These per-matrix block descriptors are then compacted into a global list of triples

$$
(\text{batch index}, \text{start index}, \text{end index}),
$$

using an inclusive scan over the number of blocks found in each matrix. After compaction, all active blocks across the batch are processed in parallel.

This is why the workgroup path naturally has a two-stage iteration pattern:

1. discover and compact active blocks,
2. apply one Francis sweep to each discovered block.

After the sweep, new zeros may have appeared in the local $e$ arrays, so the next outer pass rebuilds the block list from scratch.

### 9.3 CTA Path Split Book-Keeping

The CTA path does not build a global block list. Instead, each subgroup partition processes one matrix in registers and walks through the block decomposition locally.

It maintains a running left boundary index

$$
b_{\mathrm{next}}.
$$

At each outer iteration,

$$
b_{\mathrm{begin}} \leftarrow b_{\mathrm{next}},
$$

and it explicitly enforces the left boundary by setting

$$
e_{b_{\mathrm{begin}}-1} \leftarrow 0
$$

whenever that index exists.

It then deflates the remaining tail and defines the right boundary as the first zero to the right:

$$
b_{\mathrm{end}} = \min\{ i \ge b_{\mathrm{begin}} : e_i = 0 \},
$$

with the convention $b_{\mathrm{end}}=n$ if no such zero exists. The next block is then

$$
b_{\mathrm{next}} \leftarrow b_{\mathrm{end}} + 1.
$$

So the CTA path is not building a global queue of subproblems. It is traversing the current block partition of one matrix from left to right.

### 9.4 How Inner Splits Are Used During Convergence

Once a block $[\ell:r]$ has been selected, the iteration does not necessarily work on the whole block to completion in one shot. Instead, after every implicit step it re-tests for deflation and locates the first or last newly created zero, depending on QL or QR.

For QL, with the current active window $[\ell:m]$, it finds

$$
p = \min\{ i \in [\ell, m-1] : e_i = 0 \},
$$

and then:

- if $p = \ell$, the eigenvalue at the left edge has converged,
- if $p = \ell+1$, a $2 \times 2$ block remains and is solved exactly,
- otherwise the next implicit step is applied only to $[\ell:p]$.

For QR, with the current active window $[m:r]$, it finds

$$
p = \max\{ i \in [m+1, r] : e_{i-1} = 0 \},
$$

and then:

- if $p = r$, the eigenvalue at the right edge has converged,
- if $p = r-1$, a $2 \times 2$ block remains and is solved exactly,
- otherwise the next implicit step is applied only to $[p:r]$.

This is the precise sense in which the active interval shrinks during the iteration.

## 10. Paper-Style Pseudoalgorithms

The presentation below is closer to the style used in numerical linear algebra papers: a small number of top-level algorithms, each with clearly separated state, active blocks, and inner updates.

### Algorithm 1. BatchLAS STEQR

```text
Input:
  diagonal d = (d_1, ..., d_n), off-diagonal e = (e_1, ..., e_{n-1})
  jobz, parameters params
Output:
  eigenvalues lambda and, optionally, eigenvectors Q

if jobz = EigenVectors and back_transform = false then
  Q <- I_n
end if

copy (d, e) into working storage

if CTA specialization is available and n <= subgroup capacity then
  (d, e, Q) <- Algorithm 4 applied independently to each batch item
else
  repeat n-1 times
    construct the list of maximal active blocks using Algorithm 2
    for each active block in parallel do
      apply one shifted Francis sweep using Algorithm 3
    end for
    if jobz = EigenVectors then
      apply the stored rotations to Q
    end if
  end repeat
end if

lambda <- d
if sorting is requested then
  sort (lambda, Q) jointly
end if
```

### Algorithm 2. Active-Block Construction From Deflation

```text
Input:
  off-diagonal e = (e_1, ..., e_{n-1})
Output:
  list of maximal active blocks I_1, ..., I_p

p <- 0
i <- 1
while i <= n-1 do
  if e_i = 0 then
    i <- i + 1
  else
    s <- i
    while i <= n-1 and e_i != 0 do
      i <- i + 1
    end while
    t <- i
    p <- p + 1
    I_p <- [s, t]
  end if
end while
```

The interval $I_p = [s,t]$ represents the tridiagonal subproblem on diagonal entries $d_s, \dots, d_t$ and off-diagonal entries $e_s, \dots, e_{t-1}$.

### Algorithm 3. One Shifted Sweep on an Active Block

```text
Input:
  active block [s:t], local tridiagonal data (d, e)
Output:
  updated local tridiagonal data and optional stored Givens rotations

if t = s then
  return
end if

if t = s + 1 then
  diagonalize the 2x2 block exactly
  return
end if

scale the active block into a safe range if necessary

if |d_s| <= |d_t| then
  use QR ordering
else
  use QL ordering by reversing the local index map
end if

form shift mu from the trailing 2x2 block in the chosen ordering
compute the first Givens rotation from (d_edge - mu, e_edge)
apply the local similarity update of Section 6

for each remaining local index do
  compute the next rotation from (current coupling, incoming bulge)
  apply the same local similarity update
end for

apply the relative deflation test to every local off-diagonal
undo scaling
```

### Algorithm 4. CTA STEQR on One Active Matrix

```text
Input:
  one tridiagonal matrix (d, e)
Output:
  converged diagonal d and optional eigenvector matrix Q

load d and e into registers
cache Q in shared memory if jobz = EigenVectors
budget <- n * max_sweeps
next_block_begin <- 1

while next_block_begin <= n do
  block_begin <- next_block_begin
  if block_begin > 1 then
    e_(block_begin-1) <- 0
  end if

  deflate the tail [block_begin:n]
  block_end <- first index i >= block_begin with e_i = 0, or n if none exists
  next_block_begin <- block_end + 1

  if block_end <= block_begin then
    continue
  end if

  scale the block [block_begin:block_end] if necessary

  if |d_block_end| < |d_block_begin| then
    run QL convergence from the left edge:
      repeatedly locate the first split p,
      solve 1x1 or 2x2 edge cases when reached,
      otherwise apply one implicit QL step on [ell:p]
  else
    run QR convergence from the right edge:
      repeatedly locate the last split p,
      solve 1x1 or 2x2 edge cases when reached,
      otherwise apply one implicit QR step on [p:r]
  end if

  rescale the block back
  decrease budget after every implicit step
  if budget is exhausted then
    mark non-convergence and stop
  end if
end while
```

### 10.1 QR Sweep in Virtual Coordinates

For QR, the workgroup path uses the physical ordering

$$
(d_1, \dots, d_n), \qquad (e_1, \dots, e_{n-1}).
$$

The first rotation annihilates the first subdiagonal of the shifted matrix:

$$
G_1^T
\begin{bmatrix}
d_1 - \mu \\
e_1
\end{bmatrix}
=
\begin{bmatrix}
r \\
0
\end{bmatrix}.
$$

Each subsequent rotation annihilates the bulge created by the previous one.

### 10.2 QL Sweep in Virtual Coordinates

For QL, the same formulas are reused after reversing the local indexing. Equivalently, the implementation defines virtual coordinates

$$
\tilde d_k = d_{n-k+1}, \qquad \tilde e_k = e_{n-k},
$$

per active block and applies the exact same QR-style bulge chase to

$$
(\tilde d_1, \dots, \tilde d_{n_{\mathrm{local}}}),
\qquad
(\tilde e_1, \dots, \tilde e_{n_{\mathrm{local}}-1}).
$$

This is mathematically a QL iteration in the original coordinates.

## 11. CTA Update Schemes

The CTA implementation exposes two mathematically equivalent but operationally different update schemes.

### 11.1 `EXP`: Explicit Similarity Update

This scheme mirrors the workgroup formulas directly. Each step constructs a Givens rotation from either

$$
(d_1 - \mu, e_1)
$$

for the first step, or from

$$
(e_{j-1}^{(\mathrm{updated})}, b_{j-1})
$$

for later steps, and then explicitly updates the local tridiagonal entries using the formulas in Section 6.

This makes the CTA path mathematically closest to `steqr_wg`.

### 11.2 `PG`: Parlett-Gray Style Recurrence

This scheme uses the classical implicit recurrence instead of rewriting each local $2 \times 2$ similarity explicitly.

In QR form, the recurrence updates scalar state variables $(g, p, c, s)$ across the sweep:

$$
f = s e_i,
$$

$$
(c_i, s_i, r_i) = \operatorname{lartg}(g, f),
$$

$$
g_2 = d_i - p,
$$

$$
r_2 = (d_{i+1} - g_2)s_i + 2 c_i c e_i,
$$

$$
p \leftarrow s_i r_2,
$$

$$
d_i \leftarrow g_2 + p,
$$

$$
g \leftarrow c_i r_2 - c e_i.
$$

The QL recurrence is the reversed analogue. This is closer to the compact scalar formulas used in LAPACK's implicit tridiagonal QR/QL kernels.

## 12. Exact Inner Recurrences for the CTA Path

For completeness, the CTA path supports two ways of writing the same implicit step.

### 12.1 `EXP` as an Explicit Local Similarity

This is the formulation in Section 6. The state is the local tridiagonal band plus the current bulge. Every lane conceptually owns one diagonal entry and one off-diagonal entry, and one implicit step updates the local $4 \times 4$ window displayed above.

### 12.2 `PG` as a Scalar Recurrence

The `PG` formulation avoids explicitly writing every local similarity in matrix form and instead propagates a few scalar recurrence variables. In QR form, one step is

$$
f_i = s_{i-1} e_i,
$$

$$
(c_i, s_i, r_i) = \operatorname{lartg}(g_{i-1}, f_i),
$$

$$
g_i^{\star} = d_i - p_{i-1},
$$

$$
r_i^{\star} = (d_{i+1} - g_i^{\star}) s_i + 2 c_i c_{i-1} e_i,
$$

$$
p_i = s_i r_i^{\star},
$$

$$
d_i \leftarrow g_i^{\star} + p_i,
$$

$$
g_i = c_i r_i^{\star} - c_{i-1} e_i.
$$

The QL version is the mirrored recurrence obtained by reversing the sweep direction. Mathematically, this is still the same implicit similarity transformation; it is only a different state representation.

## 13. Eigenvector Accumulation

If eigenvectors are requested, the solver accumulates the orthogonal similarities into $Q$.

If `back_transform = false`, the implementation starts from

$$
Q \leftarrow I.
$$

Otherwise it treats the provided matrix as an incoming basis and right-applies the tridiagonal eigenvector transforms.

For each local Givens rotation acting on adjacent columns $k$ and $k+1$,

$$
Q_{:,\{k,k+1\}} \leftarrow Q_{:,\{k,k+1\}} G(c,s).
$$

In the QL case the physical column order is reversed relative to the virtual sweep order, but the mathematical operation is still a right multiplication by the corresponding adjacent rotation.

The workgroup path stores the rotations generated during the sweep and applies them afterward. The CTA path usually applies them immediately to a shared-memory cache of $Q$.

## 14. Sorting

After convergence, the implementation copies the final diagonal to the output eigenvalue array:

$$
\lambda_i \leftarrow d_i.
$$

If `sort = true`, it then sorts the eigenpairs, by default in ascending order:

$$
\lambda_1 \le \lambda_2 \le \dots \le \lambda_n.
$$

When eigenvectors are present, the same permutation is applied to the columns of $Q$.

## 15. What Is Specific to BatchLAS

The mathematics is standard implicit tridiagonal QR/QL iteration, but the implementation has a few BatchLAS-specific choices:

- It is batched from the ground up.
- It dynamically chooses between a general workgroup path and a small-$n$ CTA path.
- The CTA path exposes two update schemes: `PG` and `EXP`.
- The user-facing `zero_threshold` parameter is currently not the actual deflation criterion.
- Eigenpair sorting is integrated as part of the solver pipeline.

In short, the core algorithm is:

$$
T_0 \xrightarrow{\text{deflation + implicit shifted QR/QL steps}} T_k = \Lambda,
$$

with orthogonal accumulation

$$
Q = G_1 G_2 \cdots G_p,
\qquad
Q^T T_0 Q = \Lambda.
$$

That is the mathematics implemented by BatchLAS STEQR.