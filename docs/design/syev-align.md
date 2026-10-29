# syev_align: eigenvector continuity {#design_syev_align}

> **Status:** current · checked 2026-10-29. Correctness only: nothing has been timed.

`syev_align` rotates a batch of new eigenbases onto a previous one, cluster by cluster, so a
degenerate or near-degenerate level keeps the basis it had (the Ih C60 Hückel levels are 1-, 3-,
4-, 5- and 9-fold).

## syev_align: the algorithm

Per item, with \f$ m \f$ = `V.cols()`: one `gemm` forms \f$ M = V_{old}^H V \f$ (it selects its
own route; `syev_align` has no table and no vendor counterpart); one work-item per cluster runs a
one-sided Jacobi SVD \f$ M_{cc} = U \Sigma W^H \f$ and stores \f$ R^H = U W^H \f$, completing
columns with \f$ \sigma < p\,\varepsilon\,\sigma_{max} \f$ by Gram-Schmidt so \f$ R \f$ stays
unitary; then \f$ V_c := V_c R \f$ from a workspace copy of `V`.

The launches chain through the workspace, so they run on an in-order queue
(`detail::on_in_order_queue`), ordered after and before the caller's out-of-order `Queue`
(`AlignsOnAnOutOfOrderQueue`). `syev_align_validate_params` throws `invalid_argument` before any
allocation when `V.rows() * V.cols()` exceeds `INT_MAX`, since the workspace views take `int`
extents (`RejectsSizesPastTheIntRange`, 46341 x 46341 against 46341 x 46340).

## syev_align: the cluster rule and the absolute floor

Columns \f$ j-1 \f$ and \f$ j \f$ share a cluster when

\f[ \lambda_j - \lambda_{j-1} \le \mathrm{tol} \cdot \max(|\lambda_0|, |\lambda_{m-1}|), \f]

chained into maximal runs: one absolute threshold per item. **There is no absolute floor**: a
floor makes the result depend on the units of \f$ A \f$ (a whole spectrum scaled below it becomes
one cluster), while the relative rule is scale invariant. `NearZeroClusterUsesTheItemScale` puts a
pair at \f$ 2 \cdot 10^{-14} \f$ with gaps \f$ 0.9 \f$ and \f$ 1.1 \f$ times the threshold, and the
same spectra scaled by \f$ 10^{-14} \f$; any floor at or above \f$ 1.1 \cdot 10^{-27} \f$ fails it.

The rule fails only when the values passed in do not reach \f$ \|A\| \f$, such as a `syevx` subset of
near-zero eigenvalues: the threshold can fall below their \f$ \varepsilon \|A\| \f$ rounding and a
degenerate level splits. The caller knows \f$ \|A\| \f$; pass
\f$ \mathrm{tol} \cdot \|A\| / \max(|\lambda_0|, |\lambda_{m-1}|) \f$.

## syev_align: open debts

- Not timed. One work-item per cluster is batch-and-cluster parallel only (a 9-fold level at batch
  128 is 128 serial SVDs of size 9). Time it against `syev` on C60 at batch \f$ \ge 128 \f$ before
  calling it cheap.
- In a vendor-free build the host (NETLIB) instantiations skip the seven cases whose inputs come
  from `syev`, which has no host route there.
