# Inbox from shard H7-device

## -> docs/design/known-defects.md: device group BLAS: the 3-D tile-group race in gemm, symm and trmm

(Not cited from code yet; found while documenting, read from the dispatch code, not reproduced.)
Full write-up, with what would settle it, is on
`docs/design/device-group-blas.md#device-group-blas-the-3-d-launch-generic-fallback`; a one-entry
pointer here is enough.

With a `sycl::nd_item<3>` executor, dimensions 1 and 2 of the group id index output tiles, but only
the tiled kernels read them. `gemm`, `syrk` and `syr2k` guard their generic fallback to tile-group
(0, 0), yet `gemm`'s non-register sub-group path (`detail::subgroup::gemm`, float, register path
ineligible) runs before that guard and covers the whole output per work-group, and `symm` / `trmm`
have no guard at all. Every tile-group then writes all of `C`: benign at `beta == 0`, wrong at
`beta != 0` or for an in-place (aliased) `trmm`. Sites:
`include/batchlas/blas/device/detail/group_blas_gemm.hh` (dispatch, before the 3-D guard),
`group_blas_symm.hh` and `group_blas_trmm.hh` (dispatch functions). Fix: hoist the 3-D guard above
the non-register sub-group path and add it to symm/trmm; test with a >= 2 x 2 tile-group launch,
`beta != 0` and no workspace.

## -> .github/ci/comment_density_waivers.txt: three dead waivers

The checker reports these waivers as now free to delete (files under 18%):
`include/batchlas/util/kernel-heuristics.hh` (line 107),
`include/batchlas/internal/ormqr_blocked.hh` (line 101),
`include/batchlas/internal/sytrd_blocked.hh` (line 102).
(A6 already listed `include/batchlas/util/sycl-local-accessor-helpers.hh`.)

## -> docs/pages/architecture.md: link the device group BLAS page

Still open from `docs/_inbox/H7-dispatch-device.md` ("link the new design page"): the page
`design_device_group_blas` is not a `@subpage` of any section page, so it renders as an orphan.
