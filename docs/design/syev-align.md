# syev_align: eigenvector continuity {#design_syev_align}

> **Status:** current · checked 2026-10-25 (fullerenes plan P2.5). Correctness only: nothing has been timed.

`syev_align` rotates a batch of new eigenbases onto a previous one, cluster by cluster, so a
degenerate or near-degenerate level keeps the basis it had (the Ih C60 Hückel levels are 1-, 3-,
4-, 5- and 9-fold). The cluster tolerance is relative to the item's spectrum and has no absolute
floor; the section below gives the cases that settled this.

## syev_align: the algorithm

For each item, with \f$ m \f$ = `V.cols()`:

1. One `gemm` forms \f$ M = V_{old}^H V \f$ (\f$ m \times m \f$, in the workspace). It selects
   its own route; `syev_align` has no kernel selection, no table and no vendor counterpart.
2. One work-item per cluster runs a one-sided Jacobi SVD of the cluster's diagonal block
   \f$ M_{cc} = U \Sigma W^H \f$ and stores \f$ R^H = U W^H \f$ over the block. Columns of
   \f$ U \f$ whose singular value is below \f$ p\,\varepsilon\,\sigma_{max} \f$ are completed by
   Gram-Schmidt on unit vectors, so \f$ R \f$ stays unitary when \f$ M_{cc} \f$ is rank deficient.
3. `V` is copied into the workspace and one element-wise kernel writes \f$ V_c := V_c R \f$.

The four launches chain through the workspace, so `syev_align` runs them on an in-order queue
(`detail::on_in_order_queue`): on an out-of-order `Queue` it makes an in-order one that waits on
the caller's pending work, and the caller's queue is ordered after it. The test
`AlignsOnAnOutOfOrderQueue` repeats the call on a `Queue(ctx, false)` and covers all three
orderings: `V` is produced on that queue and not waited, and the result is read through a copy
submitted to it after the call.

`syev_align_validate_params` throws `invalid_argument` before any allocation or launch when
`V.rows() * V.cols()` exceeds `INT_MAX`: the workspace views of \f$ V \f$ and \f$ M \f$ take
`int` extents and strides, and that product bounds both (\f$ m \le n \f$). Every device index is
64-bit. `RejectsSizesPastTheIntRange` straddles it at 46341 x 46341 (throws) and
46341 x 46340 (sized).

## syev_align: the cluster rule and the absolute floor

Columns \f$ j-1 \f$ and \f$ j \f$ share a cluster when

\f[ \lambda_j - \lambda_{j-1} \le \mathrm{tol} \cdot \max(|\lambda_0|, |\lambda_{m-1}|), \f]

chained, so a cluster is a maximal run of such neighbours. The threshold is one absolute number
per item: a cluster near zero is judged by the same threshold as every other cluster of the item,
not by its own magnitude.

**Decision: no absolute floor.** The case that raised the question is a cluster of values near
\f$ 10^{-14} \f$ in an item with \f$ \max|W| = 1 \f$. The rule clusters it correctly: the
threshold is \f$ \mathrm{tol} \f$ itself, and rounding moves those values by about
\f$ \varepsilon \|A\| \f$, far below any useful \f$ \mathrm{tol} \f$. `NearZeroClusterUsesTheItemScale`
puts a pair at \f$ 2 \cdot 10^{-14} \f$ with gaps \f$ 0.9 \f$ and \f$ 1.1 \f$ times the
threshold (\f$ \mathrm{tol} = 10^{-13} \f$, \f$ \max|\lambda| = 1 \f$): one cluster and two
clusters. The same two spectra scaled by \f$ 10^{-14} \f$ give the same partition. Any absolute
floor at or above \f$ 1.1 \cdot 10^{-27} \f$ merges the scaled split pair, so the test fails under a floor.

An absolute floor would make the result depend on the units of \f$ A \f$: at a floor of
\f$ \delta \f$, a whole spectrum scaled below \f$ \delta \f$ becomes one cluster. The relative
rule is scale invariant.

The rule goes wrong only when the values passed in do not reach the scale of \f$ \|A\| \f$:
for example the lowest few eigenpairs of a matrix of norm 1 when all of them are near zero (a
`syevx` subset). Then the threshold is \f$ \mathrm{tol} \cdot \max|\lambda_{subset}| \f$, and it
can fall below the \f$ \varepsilon \|A\| \f$ rounding of the values, so a numerically degenerate
level splits. The caller knows \f$ \|A\| \f$ and `syev_align` does not. Pass
\f$ \mathrm{tol} \cdot \|A\| / \max(|\lambda_0|, |\lambda_{m-1}|) \f$ in that case.

## syev_align: open debts

- Nothing has been timed. At saturation the gemm and the per-cluster SVD are the candidates, and
  one work-item per cluster is batch-and-cluster parallel only: a 9-fold level at batch 128 is
  128 serial SVDs of size 9. Time `syev_align` against `syev` on C60 (n = 60) at batch
  \f$ \ge 128 \f$ before calling it cheap.
- In a vendor-free build the host (NETLIB) instantiations skip the seven cases whose inputs come
  from `syev`, which has no host route there (`syev_tests` fails the same way on main).
  `OrthogonalSubspacesKeepV` and the two argument tests still run on the host; every case runs
  on the GPU.
