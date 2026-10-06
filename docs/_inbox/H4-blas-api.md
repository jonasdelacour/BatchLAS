# Inbox from shard H4-blas-api

Source files: `include/batchlas/blas/functions/{gemm,gemv,hemm,her2k,herk,iluk,spmm,symm,syr2k,syrk,trmm,trsm}.hh`.
Nothing in these headers was measurement material; what moved out is history. Everything else became
Doxygen API documentation (groups `blas2`, `blas3`, `sparse`, and `dispatch` for the `backend::*_vendor`
declarations).

## -> docs/cpp-api.md: trsm: alpha moved next to the matrices

`trsm`'s `alpha` sits in position 4, immediately after the matrices, to match `trmm`
(`functions/trmm.hh`). It used to come last, so the two triangular routines disagreed on where the
scalar went and only one of them could be written from memory. Two `= delete` overloads with the old
order (`Side, Uplo, Transpose, Diag, T` after the matrices) turn the old spelling into a diagnostic
rather than leaving it to be rediscovered.

Why tombstones rather than nothing: `Side`/`Uplo`/`Transpose`/`Diag` are all `enum class`, so nothing
implicitly converts to or from `T` and a stale call could never have silently compiled into a wrong
answer; but without the deleted overloads the error would be "no matching function", which does not
say what changed. Both spellings need one: deleting only the `MatrixView` overload would leave a
`Matrix`-argument call binding to the new order with `alpha` where `side` belongs.

The vendor signature `backend::trsm_vendor` still takes `alpha` last, which is why the
`sig::*_vendor` aliases are spelled out per op rather than aliased to the public signature.

Code sites that now point here:
- `include/batchlas/blas/functions/trsm.hh` (comment above the primary `trsm` declaration):
  `evidence: docs/cpp-api.md#trsm-alpha-moved-next-to-the-matrices`

## -> docs/design/vendor-independence.md: optional addition under the existing "The entry-point facade"

No new heading is needed; the code cites the existing `#the-entry-point-facade`. The per-op headers
carried one detail the section does not: where the public definitions lived before WP0 S5 moved them
to `src/dispatch/entry_points/level3.cc` — `cublas.cc:1568`, `rocblas.cc:99`, `netlib_lapack.cc:288`
(line numbers as of that change, now stale). The same declaration-only `*_vendor` pattern had been
used all along by `syev_vendor` (`functions/syev.hh`) and `ormqr_vendor` (`functions/ormqr.hh`). If
the coordinator wants to keep that, append a sentence to the section; otherwise drop it.

Code sites citing the existing anchor (added by this pass, one per `backend::*_vendor` declaration):
`gemm.hh`, `gemv.hh`, `hemm.hh`, `her2k.hh`, `herk.hh`, `spmm.hh`, `symm.hh`, `syr2k.hh`, `syrk.hh`,
`trmm.hh`, `trsm.hh` (all under `include/batchlas/blas/functions/`).

## Dropped as already recorded elsewhere (no action)

- `gemm.hh`: the note that the C++ `gemm_heterogeneous` alias was removed and that the Python binding's
  `gemm_heterogeneous` is not redundant. Both are already in `docs/cpp-api.md` ("`gemm` handles a
  heterogeneous batch natively"). The contract part (m == 0 / n == 0 items skipped, k == 0 item is
  `C := beta*C`, from `docs/perf/gemm.md` "Correctness findings") is now in the `gemm` API doc.
- `herk.hh`, `her2k.hh`, `hemm.hh`: the real-alpha / complex-alpha / Hermitian-diagonal rationale is kept,
  shortened, in the API docs (it is contract).
