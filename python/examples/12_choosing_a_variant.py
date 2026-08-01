"""Choosing a variant: batching, device placement, and measured throughput.

The whole point of a batched library is that many small problems are solved in
one launch. This example measures that, and shows how to pick between the syev
variants for a given (n, batch) instead of guessing.

Timings here include the host round trip (NumPy in, NumPy out), which is what
you actually pay from Python.

Run with:  python 12_choosing_a_variant.py
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
    section,
    timed,
)


def main() -> None:
    header("12. Choosing a variant")
    device = preferred_device()
    report("device", device or "library default")

    section("One launch beats a Python loop")
    # Same arithmetic either way; the difference is launch overhead and occupancy.
    batch, n = 512, 16
    matrices = batched_symmetric(batch, n, seed=1)

    _, batched_time = timed(lambda: bl.syev_cta(matrices, compute_vectors=False, device=device))

    def one_at_a_time():
        return [bl.syev_cta(matrices[i], compute_vectors=False, device=device) for i in range(batch)]

    _, looped_time = timed(one_at_a_time, repeats=1)

    report(f"batched  ({batch} x {n}x{n})", f"{batched_time * 1e3:.2f} ms")
    report(f"looped   ({batch} calls)", f"{looped_time * 1e3:.2f} ms")
    report("speed-up", f"{looped_time / batched_time:.1f}x")

    section("How throughput scales with batch size")
    for batch_size in (1, 16, 128, 1024):
        matrices = batched_symmetric(batch_size, 16, seed=2)
        _, elapsed = timed(lambda: bl.syev_cta(matrices, compute_vectors=False, device=device))
        report(
            f"batch={batch_size:<5d}",
            f"{elapsed * 1e3:7.2f} ms total, {elapsed / batch_size * 1e6:7.1f} us per matrix",
        )

    section("Which syev variant wins at which size?")
    # syev_cta and syev_jacobi_cta only accept n <= 32; the blocked and two-stage
    # paths are built for larger n. Measure rather than assume.
    for n, batch_size in ((16, 512), (32, 256), (128, 32), (512, 4)):
        matrices = batched_symmetric(batch_size, n, seed=3)
        reference = np.linalg.eigvalsh(matrices)
        support = bl.syev_variant_support(matrices, device=device)
        candidates = ["syev"]
        if n <= 32 and support["cta"]:
            candidates += ["syev_cta", "syev_jacobi_cta"]
        if support["blocked"]:
            candidates.append("syev_blocked")
        if support["two_stage"]:
            candidates.append("syev_two_stage")

        report(f"n={n}, batch={batch_size}", "")
        for name in candidates:
            try:
                values, elapsed = timed(
                    lambda fn=name: getattr(bl, fn)(matrices, compute_vectors=False, device=device)
                )
                error = eigenvalue_error(values, reference)
                report(f"  {name:16s}", f"{elapsed * 1e3:8.2f} ms   max eigenvalue error {error:.2e}")
            except (RuntimeError, NotImplementedError) as exc:
                report(f"  {name:16s}", f"unavailable ({type(exc).__name__})")

    section("CPU versus GPU for the same call")
    # Small batches often do not justify a device transfer; large ones clearly do.
    # Note syev_cta and friends need a sub-group width of 32 and so are GPU-only;
    # the general syev driver runs on both.
    for batch_size in (1, 256):
        matrices = batched_symmetric(batch_size, 16, seed=4)
        line = []
        for target in ("cpu", "gpu"):
            try:
                _, elapsed = timed(lambda t=target: bl.syev(matrices, compute_vectors=False, device=t))
                line.append(f"{target}: {elapsed * 1e3:8.2f} ms")
            except (RuntimeError, NotImplementedError) as exc:
                line.append(f"{target}: unavailable ({type(exc).__name__})")
        report(f"batch={batch_size:<5d}", "   ".join(line))

    section("Skipping eigenvectors is a real saving")
    matrices = batched_symmetric(256, 32, seed=5)
    _, with_vectors = timed(lambda: bl.syev_cta(matrices, compute_vectors=True, device=device))
    _, values_only = timed(lambda: bl.syev_cta(matrices, compute_vectors=False, device=device))
    report("with eigenvectors", f"{with_vectors * 1e3:.2f} ms")
    report("values only", f"{values_only * 1e3:.2f} ms")

    section("float32 versus float64")
    matrices = batched_symmetric(512, 16, seed=6)
    for dtype in (np.float32, np.float64):
        typed = matrices.astype(dtype)
        _, elapsed = timed(lambda m=typed: bl.syev_cta(m, compute_vectors=False, device=device))
        report(f"{np.dtype(dtype).name}", f"{elapsed * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
