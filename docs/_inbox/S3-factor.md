# Inbox from shard S3-factor

## -> docs/perf/syev.md: syev: the Upper-to-Lower mirror for Lower-only providers

Moved from the header comment of `src/extensions/uplo_mirror.hh`.

`syev_blocked` and `syev_two_stage`, and everything under them (`sytrd_blocked`,
`sytrd_sy2sb`, `sytrd_sb2st`), implement `Uplo::Lower` only; `sytrd_blocked` used to throw
outright on `Upper`. So every `Uplo::Upper` call fell back to the vendor however much faster
the native providers were at that shape: a routing loss caused by a missing O(n^2) step in
front of an O(n^3) solve. `mirror_upper_to_lower` is that step.

For a Hermitian matrix both triangles carry the same operator, `A[j][i] == conj(A[i][j])`.
Writing the upper triangle into the lower one gives a matrix whose LOWER triangle describes
exactly the input operator, so the existing Lower path produces identical eigenvalues and
eigenvectors. The cost is O(n^2 * batch) against the solve's O(n^3 * batch), below noise at
every size where routing matters.

In place is safe: `syev` documents `A` as overwritten
(`include/batchlas/blas/functions/syev.hh`), and the Lower path destroys `A` during the
reduction regardless. The diagonal is left alone; for complex input its imaginary part is not
forced to zero, matching what the Lower path already assumes of a Hermitian input.

The header is declaration-only, with explicit instantiations in `uplo_mirror.cc`, because a
SYCL kernel name class must have exactly one definition in the program: defining the kernel
inline in the header and calling it from both `syev_blocked.cc` and `syev_two_stage.cc`
produced "definition with same mangled name" ODR errors.

Code sites that now point here:
- `src/extensions/uplo_mirror.hh:3`
  (`evidence: docs/perf/syev.md#syev-the-upper-to-lower-mirror-for-lower-only-providers`)

## Waiver note (for the coordinator)

`src/extensions/larft_wy.hh` is now 14.83%: its waiver line (waivers.txt:151) can be deleted.
`src/extensions/uplo_mirror.hh` (53%), `src/backends/getrs_route.hh` (35%),
`src/backends/orgqr_route.hh` (28%) and `src/util/resident_capacity.hh` (27%) were cut down in
this pass but remain above 18% (short files whose remaining comments are invariants), so their waiver lines are
still needed. Waiver reasons that say "untouched by this campaign" / "predates the small-n comment
budget" for those files are now stale; suggested reason: "burned down in the Doxygen pass; the
remaining lines are traps; narrative in docs/perf/{qr,lu}.md".
