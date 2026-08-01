"""Symmetric and Hermitian eigensolvers: the syev family.

BatchLAS ships several full-spectrum symmetric eigensolvers. They compute the
same thing; they differ in how the work is mapped onto the device:

  syev             general-purpose driver (vendor path where available)
  syev_cta         one work-group per matrix: sytrd_cta -> steqr_cta -> ormqx_cta
  syev_jacobi_cta  one work-group per matrix, two-sided Jacobi (high relative accuracy)
  syev_blocked     blocked reduction + divide-and-conquer, for medium/large n
  syev_two_stage   dense -> band -> tridiagonal reduction, for large n

Use syev_variant_support() to ask the device which of these it can actually run.

Run with:  python 06_symmetric_eigensolvers.py
"""

from __future__ import annotations

import numpy as np

import batchlas as bl

from _common import (
    batched_symmetric,
    eigenvalue_error,
    header,
    preferred_device,
    report,
    residual,
    section,
)


def main() -> None:
    header("6. Symmetric eigensolvers")
    device = preferred_device()
    batch = 8

    section("Which variants does this device support?")
    small = batched_symmetric(batch, 16, seed=1)
    for key, value in bl.syev_variant_support(small, uplo="lower", device=device).items():
        report(key, value)

    section("syev: eigenvalues and eigenvectors")
    # The input is read from the lower triangle by default.
    reference = np.linalg.eigvalsh(small)
    values, vectors = bl.syev(small, device=device)
    report("eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
    report("|A V - V diag(w)|", residual(small, values, vectors), tol=1e-10)

    section("Eigenvalues only")
    values = bl.syev(small, compute_vectors=False, device=device)
    report("shape", values.shape)
    report("error", eigenvalue_error(values, reference), tol=1e-10)

    section("The small-matrix variants (n <= 32): one work-group per problem")
    for name in ("syev_cta", "syev_jacobi_cta"):
        values, vectors = getattr(bl, name)(small, device=device)
        report(f"{name:16s} eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)
        report(f"{name:16s} residual", residual(small, values, vectors), tol=1e-10)

    section("The medium/large variants")
    medium = batched_symmetric(4, 128, seed=2)
    medium_reference = np.linalg.eigvalsh(medium)
    for name in ("syev", "syev_blocked", "syev_two_stage"):
        try:
            values, vectors = getattr(bl, name)(medium, device=device)
            report(f"{name:16s} eigenvalue error", eigenvalue_error(values, medium_reference), tol=1e-8)
            report(f"{name:16s} residual", residual(medium, values, vectors), tol=1e-8)
        except (RuntimeError, NotImplementedError) as exc:
            report(f"{name:16s}", f"unavailable ({type(exc).__name__}: {exc})")

    section("Tuning a variant through its options object")
    # syev_cta drives a CTA-resident QR iteration; SteqrOptions controls it.
    options = bl.SteqrOptions(max_sweeps=200, cta_shift_strategy="wilkinson", sort_order="ascending")
    values, _ = bl.syev_cta(small, options=options, device=device)
    report("eigenvalue error", eigenvalue_error(values, reference), tol=1e-10)

    # syev_blocked drives divide-and-conquer; StedcOptions controls that.
    options = bl.StedcOptions(recursion_threshold=32, max_sec_iter=80)
    try:
        values, _ = bl.syev_blocked(medium, options=options, device=device)
        report("blocked with custom stedc options", eigenvalue_error(values, medium_reference), tol=1e-8)
    except (RuntimeError, NotImplementedError) as exc:
        report("blocked with custom stedc options", f"unavailable ({type(exc).__name__})")

    section("Hermitian (complex) input")
    rng = np.random.default_rng(3)
    z = rng.standard_normal((batch, 16, 16)) + 1j * rng.standard_normal((batch, 16, 16))
    z = (z + z.conj().transpose(0, 2, 1)) / 2.0
    z_reference = np.linalg.eigvalsh(z)
    values, vectors = bl.syev(z, device=device)
    report("eigenvalue error", eigenvalue_error(values, z_reference), tol=1e-10)
    report("|A V - V diag(w)|", residual(z, values.astype(z.dtype), vectors), tol=1e-10)

    section("uplo: which triangle holds your data")
    # Both triangles of a full symmetric matrix give the same spectrum, whichever
    # side you nominate.
    for side in ("lower", "upper"):
        values = bl.syev(small, compute_vectors=False, uplo=side, device=device)
        report(f"uplo={side}", eigenvalue_error(values, reference), tol=1e-10)

    # Supplying only the lower triangle also works, which confirms the strict
    # upper half really is ignored for uplo="lower".
    values = bl.syev(np.tril(small), compute_vectors=False, uplo="lower", device=device)
    report("lower triangle only", eigenvalue_error(values, reference), tol=1e-10)
    # The mirror image (upper triangle only, uplo="upper") is NOT reliable on the
    # CUDA path today -- see the note in README.md. Pass the full matrix instead.


if __name__ == "__main__":
    main()
