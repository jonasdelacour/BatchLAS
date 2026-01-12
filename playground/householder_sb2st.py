"""householder_sb2st.py

Portable NumPy reference for Householder SB2ST-style (stage-2) reduction:
symmetric/Hermitian *band* → (real) tridiagonal (d,e).

This is a correctness- and readability-first implementation of a *bulge-chasing*
schedule using short Householder reflectors and dense window updates.

Input storage (LAPACK lower band)
--------------------------------
AB has shape (kd+1, n) and stores the lower band:

    AB[r, j] = A[j+r, j]    for r = 0..kd

Algorithm (LAPACK SB2ST schedule, stage-2)
----------------------------------------
This module implements the same high-level bulge-chasing schedule as LAPACK’s
`*_SB2ST` stage-2 routines (e.g. `SSYTRD_SB2ST`) for the *lower-band* case.

The key idea is to control fill-in structurally (via the chase ordering) rather
than by truncation/zeroing outside an assumed working band.

Internally we keep an *expanded* lower band of width at least `2*kd` (matching
LAPACK’s A/AW stacked layout) and apply short Householder reflectors to small
principal blocks and adjacent rectangular panels, as dictated by the SB2ST
schedule.

Outputs
-------
Returns a real tridiagonal via (d,e) and a tridiagonal band view AB_tri (2,n)
with AB_tri[0,j]=T[j,j] and AB_tri[1,j]=T[j+1,j].

Optional “HOUS2-like” store
---------------------------
If return_hous2=True, we return a reflector log storing (start index, v, tau)
for each step. This mirrors the idea of LAPACK’s stage-2 Householder storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Iterator
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Band helpers (LAPACK lower symmetric/Hermitian band storage)
# AB[r, j] = A[j+r, j] for r = 0..kd.
# -----------------------------------------------------------------------------


def band_get(AB: np.ndarray, kd: int, i: int, j: int) -> np.generic:
    """Get A[i,j] from lower-band AB (Hermitian assumed).

    If (i,j) is outside the stored lower band, returns 0.
    For i < j, returns conj(A[j,i]) to satisfy Hermitian symmetry.
    """
    n = AB.shape[1]
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError("i/j out of range")

    if i >= j:
        r = i - j
        if r > kd:
            return AB.dtype.type(0)
        return AB[r, j]

    val = band_get(AB, kd, j, i)
    return np.conjugate(val)


def band_set(AB: np.ndarray, kd: int, i: int, j: int, val: np.generic) -> None:
    """Set A[i,j] in lower-band AB (Hermitian assumed).

    Writes to the stored lower triangle (i >= j). For i < j, writes the
    conjugate value to (j,i). Entries outside the stored bandwidth are ignored.
    For complex inputs, diagonal entries are forced real.
    """
    n = AB.shape[1]
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError("i/j out of range")

    if i < j:
        band_set(AB, kd, j, i, np.conjugate(val))
        return

    r = i - j
    if r > kd:
        return

    if i == j and np.iscomplexobj(AB):
        val = np.array(val).real.astype(AB.real.dtype) + 0j
    AB[r, j] = val


def extract_dense_window(
    AB: np.ndarray, kd: int, r0: int, r1: int, c0: int, c1: int
) -> np.ndarray:
    """Extract dense window A[r0:r1, c0:c1] using band_get."""
    W = np.empty((r1 - r0, c1 - c0), dtype=AB.dtype)
    for ii, i in enumerate(range(r0, r1)):
        for jj, j in enumerate(range(c0, c1)):
            W[ii, jj] = band_get(AB, kd, i, j)
    return W


def scatter_dense_window(
    AB: np.ndarray,
    kd: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    W: np.ndarray,
) -> None:
    """Scatter a dense window back into AB, enforcing Hermitian symmetry."""
    for ii, i in enumerate(range(r0, r1)):
        for jj, j in enumerate(range(c0, c1)):
            if i >= j and (i - j) <= kd:
                band_set(AB, kd, i, j, W[ii, jj])


def lower_band_to_dense(AB: np.ndarray, kd: int) -> np.ndarray:
    """Reconstruct full dense Hermitian/symmetric matrix from lower band."""
    n = AB.shape[1]
    A = np.zeros((n, n), dtype=AB.dtype)
    for j in range(n):
        rmax = min(kd, n - 1 - j)
        for r in range(rmax + 1):
            i = j + r
            A[i, j] = AB[r, j]
            if i != j:
                A[j, i] = np.conjugate(AB[r, j])
    if np.iscomplexobj(A):
        np.fill_diagonal(A, A.diagonal().real)
    return A


def dense_to_lower_band(A: np.ndarray, kd: int) -> np.ndarray:
    """Pack dense Hermitian/symmetric A into lower band AB."""
    n = A.shape[0]
    AB = np.zeros((kd + 1, n), dtype=A.dtype)
    for j in range(n):
        rmax = min(kd, n - 1 - j)
        for r in range(rmax + 1):
            AB[r, j] = A[j + r, j]
    return AB


# -----------------------------------------------------------------------------
# Householder primitive (LARFG-style)
# -----------------------------------------------------------------------------


def _safe_norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def householder(x: np.ndarray) -> Tuple[np.ndarray, np.generic, np.generic]:
    """Return (v, tau, beta) with (I - tau v v^H) x = [beta,0,...]^T and v[0]=1."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.shape[0]
    if n == 0:
        return x.copy(), x.dtype.type(0), x.dtype.type(0)
    if n == 1:
        v = x.copy()
        v[0] = x.dtype.type(1)
        return v, x.dtype.type(0), x[0]

    alpha = x[0]
    x_tail = x[1:]
    xnorm = _safe_norm2(x_tail)

    if xnorm == 0 and (np.isrealobj(alpha) or (np.iscomplexobj(alpha) and alpha.imag == 0)):
        v = np.zeros_like(x)
        v[0] = x.dtype.type(1)
        return v, x.dtype.type(0), alpha

    if np.iscomplexobj(x):
        alpha_abs = float(np.abs(alpha))
        xfull_norm = np.hypot(alpha_abs, xnorm)
        alpha_phase = x.dtype.type(1) if alpha_abs == 0 else alpha / x.dtype.type(alpha_abs)
        beta = -alpha_phase * x.dtype.type(xfull_norm)
        tau = (beta - alpha) / beta
        v = x.copy()
        v[0] = alpha - beta
        v /= v[0]
        v[0] = x.dtype.type(1)
        return v, tau, beta

    xfull_norm = float(np.hypot(float(alpha), xnorm))
    beta = -np.copysign(xfull_norm, float(alpha))
    tau = (beta - alpha) / beta
    v = x.copy()
    v[0] = alpha - beta
    v /= v[0]
    v[0] = x.dtype.type(1)
    return v, tau, x.dtype.type(beta)


def apply_householder_two_sided_dense(W: np.ndarray, u: np.ndarray, tau: np.generic) -> None:
    """In-place W <- (I - tau u u^H) W (I - tau u u^H)^H."""
    if tau == 0:
        return
    uH = u.conjugate()
    w = uH @ W
    W -= tau * np.outer(u, w)
    w2 = W @ u
    W -= np.conjugate(tau) * np.outer(w2, uH)


def apply_householder_left_dense(C: np.ndarray, v: np.ndarray, tau: np.generic) -> None:
    """In-place C <- (I - tau v v^H) C."""
    if tau == 0:
        return
    vH = v.conjugate()
    w = vH @ C
    C -= np.outer(tau * v, w)


def apply_householder_right_dense(C: np.ndarray, v: np.ndarray, tau: np.generic) -> None:
    """In-place C <- C (I - tau v v^H)."""
    if tau == 0:
        return
    vH = v.conjugate()
    w = C @ v
    C -= np.outer(w, tau * vH)


# -----------------------------------------------------------------------------
# SB2ST stage-2 building blocks
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Reflector:
    v: np.ndarray
    tau: np.generic


class KernelType(IntEnum):
    """LAPACK's SB2ST kernel types (UPLO='L')."""

    START = 1
    UPDATE_AND_GENERATE = 2
    APPLY_STORED = 3


@dataclass(frozen=True)
class KernelCall:
    """One scheduled SB2ST kernel application.

    Indices are 0-based and inclusive for the block range.
    sweep_id is kept 1-based to preserve LAPACK parity semantics.
    """

    kernel_type: KernelType
    sweep_id: int
    block_start: int
    block_end: int


def _read_col_segment(ABw: np.ndarray, kd_work: int, col: int, row0: int, row1_inclusive: int) -> np.ndarray:
    """Read A[row0:row1_inclusive+1, col] into a dense vector."""
    return np.array(
        [band_get(ABw, kd_work, i, col) for i in range(row0, row1_inclusive + 1)],
        dtype=ABw.dtype,
    )


def _write_col_segment_as_beta_and_zeros(
    ABw: np.ndarray, kd_work: int, col: int, row0: int, row1_inclusive: int, beta: np.generic
) -> None:
    """Overwrite A[row0:row1_inclusive, col] with [beta, 0, ..., 0]^T inside the band."""
    band_set(ABw, kd_work, row0, col, beta)
    for i in range(row0 + 1, row1_inclusive + 1):
        band_set(ABw, kd_work, i, col, ABw.dtype.type(0))


def _apply_two_sided_to_principal_block(
    ABw: np.ndarray,
    kd_work: int,
    block_start: int,
    block_end: int,
    reflector: Reflector,
) -> None:
    """Apply (I - tau v v^H) A (I - tau v v^H)^H to A[block_start:block_end]^2."""
    if reflector.tau == 0:
        return
    W = extract_dense_window(ABw, kd_work, block_start, block_end + 1, block_start, block_end + 1)
    apply_householder_two_sided_dense(W, reflector.v, reflector.tau)
    scatter_dense_window(ABw, kd_work, block_start, block_end + 1, block_start, block_end + 1, W)


def _apply_reflector_right_to_panel(
    ABw: np.ndarray,
    kd_work: int,
    row0: int,
    row1_inclusive: int,
    col0: int,
    col1_inclusive: int,
    reflector: Reflector,
) -> None:
    """Apply panel <- panel (I - tau v v^H) to A[row0:row1, col0:col1]."""
    if reflector.tau == 0 or row0 > row1_inclusive or col0 > col1_inclusive:
        return
    B = extract_dense_window(ABw, kd_work, row0, row1_inclusive + 1, col0, col1_inclusive + 1)
    apply_householder_right_dense(B, reflector.v, reflector.tau)
    scatter_dense_window(ABw, kd_work, row0, row1_inclusive + 1, col0, col1_inclusive + 1, B)


def _apply_reflector_left_to_panel(
    ABw: np.ndarray,
    kd_work: int,
    row0: int,
    row1_inclusive: int,
    col0: int,
    col1_inclusive: int,
    reflector: Reflector,
) -> None:
    """Apply panel <- (I - tau v v^H) panel to A[row0:row1, col0:col1]."""
    if reflector.tau == 0 or row0 > row1_inclusive or col0 > col1_inclusive:
        return
    C = extract_dense_window(ABw, kd_work, row0, row1_inclusive + 1, col0, col1_inclusive + 1)
    apply_householder_left_dense(C, reflector.v, reflector.tau)
    scatter_dense_window(ABw, kd_work, row0, row1_inclusive + 1, col0, col1_inclusive + 1, C)


# -----------------------------------------------------------------------------
# Stage-2 store (HOUS2-like)
# -----------------------------------------------------------------------------


@dataclass
class Hous2:
    starts: List[int]
    vs: List[np.ndarray]
    taus: List[np.ndarray]


def _expand_band(AB: np.ndarray, kd: int, kd_work: int) -> np.ndarray:
    """Embed AB (kd) into a larger lower band storage (kd_work)."""
    if kd_work < kd:
        raise ValueError("kd_work must be >= kd")
    n = AB.shape[1]
    ABw = np.zeros((kd_work + 1, n), dtype=AB.dtype)
    ABw[: kd + 1, :] = AB
    return ABw


def _extract_tridiagonal_real(AB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = AB.shape[1]
    dtype = AB.dtype
    AB_tri = np.zeros((2, n), dtype=dtype)
    AB_tri[0, :] = AB[0, :]
    if n > 1:
        AB_tri[1, : n - 1] = AB[1, : n - 1]

    d = AB_tri[0, :].real.astype(np.float64, copy=False)
    if n <= 1:
        return AB_tri, d, np.zeros((0,), dtype=np.float64)

    e = np.abs(AB_tri[1, : n - 1]).astype(np.float64, copy=False)
    AB_tri[1, : n - 1] = e.astype(dtype)

    if np.iscomplexobj(AB_tri):
        AB_tri[0, :] = AB_tri[0, :].real.astype(dtype)
    return AB_tri, d, e


# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------


def sb2st_householder(
    AB_in: np.ndarray,
    kd: int,
    *,
    pad: Optional[int] = None,
    block_size: int = 1,
    return_hous2: bool = False,
    check_fill: bool = True,
    fill_tol: Optional[float] = None,
    max_sweeps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Hous2]]:
    """Reduce a Hermitian/symmetric band matrix to a real tridiagonal (d,e)."""
    AB = np.asarray(AB_in)
    if AB.ndim != 2:
        raise ValueError("AB_in must be 2D")
    if kd < 0:
        raise ValueError("kd must be non-negative")
    if AB.shape[0] != kd + 1:
        raise ValueError("AB_in must have shape (kd+1, n)")

    n = AB.shape[1]
    if n == 0:
        AB_tri = np.zeros((2, 0), dtype=AB.dtype)
        d = np.zeros((0,), dtype=np.float64)
        e = np.zeros((0,), dtype=np.float64)
        return AB_tri, d, e, Hous2([], [], []) if return_hous2 else None

    if pad is None:
        pad = kd
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be non-negative")
    kd_work = min(n - 1, kd + pad)

    if fill_tol is None:
        # Conservative default: scaled to dtype.
        eps = np.finfo(np.float64).eps
        fill_tol = 1000.0 * eps

    if max_sweeps is None:
        # In practice, a handful of sweeps is enough for small kd.
        max_sweeps = max(1, kd + 2)

    ABw = _expand_band(AB, kd, kd_work)
    hous2 = Hous2([], [], []) if return_hous2 else None

    # LAPACK SB2ST stage-2 requires an expanded band that can hold the bulge
    # (conceptually 2*kd for the lower case). We keep the more flexible pad
    # API, but enforce the minimal requirement here.
    if kd_work < min(n - 1, 2 * kd):
        raise ValueError(
            f"pad too small for SB2ST stage-2: need kd_work>=2*kd (got kd_work={kd_work}, kd={kd}). "
            "Increase pad (recommended: pad=kd)."
        )

    _sb2st_lapack_schedule_inplace(
        ABw,
        kd=kd,
        kd_work=kd_work,
        hous2=hous2,
        check_fill=check_fill,
        fill_tol=float(fill_tol),
    )

    # Ensure we actually reached tridiagonal form inside the stored band.
    max_off = _max_off_tridiagonal_in_band(ABw)
    if max_off > float(fill_tol) * 10.0:
        raise ValueError(
            f"reduction did not reach tridiagonal: max |ABw[r>=2]|={max_off:.3e}. "
            "Increase max_sweeps and/or pad."
        )

    # Extract tridiagonal from the final band state.
    AB_tri = np.zeros((2, n), dtype=AB.dtype)
    AB_tri[0, :] = ABw[0, :]
    if n > 1:
        AB_tri[1, : n - 1] = ABw[1, : n - 1]
    AB_tri, d, e = _extract_tridiagonal_real(AB_tri)
    return AB_tri, d, e, hous2


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _run_kernel_lower(
    ABw: np.ndarray,
    *,
    kd: int,
    kd_work: int,
    call: KernelCall,
    reflector_cache: list[Dict[int, Reflector]],
    hous2: Optional[Hous2],
) -> None:
    """Run one SB2ST kernel for the lower-band case.

    This is a readability-first, NumPy-only translation of the three LAPACK
    kernel types. The schedule determines which kernel to apply and on what
    block range.
    """

    n = ABw.shape[1]
    parity = (call.sweep_id - 1) & 1
    block_start = call.block_start
    block_end = call.block_end

    if not (0 <= block_start <= block_end < n):
        return

    if call.kernel_type == KernelType.START:
        # Create a reflector from column (block_start-1), rows block_start..block_end.
        source_col = block_start - 1
        if source_col < 0:
            return

        x = _read_col_segment(ABw, kd_work, source_col, block_start, block_end)
        v, tau, beta = householder(x)
        reflector = Reflector(v=v, tau=tau)
        reflector_cache[parity][block_start] = reflector

        _write_col_segment_as_beta_and_zeros(ABw, kd_work, source_col, block_start, block_end, beta)
        _apply_two_sided_to_principal_block(ABw, kd_work, block_start, block_end, reflector)

        if hous2 is not None and tau != 0:
            hous2.starts.append(block_start)
            hous2.vs.append(v.copy())
            hous2.taus.append(np.array(tau, dtype=ABw.dtype))
        return

    if call.kernel_type == KernelType.APPLY_STORED:
        reflector = reflector_cache[parity].get(block_start)
        if reflector is None or reflector.tau == 0:
            return
        _apply_two_sided_to_principal_block(ABw, kd_work, block_start, block_end, reflector)
        return

    if call.kernel_type != KernelType.UPDATE_AND_GENERATE:
        return

    current = reflector_cache[parity].get(block_start)
    if current is None or current.tau == 0:
        return

    # Rows below the block where the bulge lives (limited by kd).
    panel_row0 = block_end + 1
    panel_row1 = min(block_end + kd, n - 1)
    if panel_row0 > panel_row1:
        return

    # Apply the current reflector on the right to A[panel_row0:panel_row1, block_start:block_end].
    _apply_reflector_right_to_panel(
        ABw,
        kd_work,
        row0=panel_row0,
        row1_inclusive=panel_row1,
        col0=block_start,
        col1_inclusive=block_end,
        reflector=current,
    )

    # Generate the next reflector from column block_start, rows panel_row0..panel_row1.
    x2 = _read_col_segment(ABw, kd_work, block_start, panel_row0, panel_row1)
    v2, tau2, beta2 = householder(x2)
    next_reflector = Reflector(v=v2, tau=tau2)
    reflector_cache[parity][panel_row0] = next_reflector

    _write_col_segment_as_beta_and_zeros(ABw, kd_work, block_start, panel_row0, panel_row1, beta2)

    if hous2 is not None and tau2 != 0:
        hous2.starts.append(panel_row0)
        hous2.vs.append(v2.copy())
        hous2.taus.append(np.array(tau2, dtype=ABw.dtype))

    # Apply the new reflector on the left to A[panel_row0:panel_row1, block_start+1:block_end].
    _apply_reflector_left_to_panel(
        ABw,
        kd_work,
        row0=panel_row0,
        row1_inclusive=panel_row1,
        col0=block_start + 1,
        col1_inclusive=block_end,
        reflector=next_reflector,
    )


def _iter_lapack_style_schedule_calls(n: int, kd: int) -> Iterator[KernelCall]:
    """Yield SB2ST kernel calls following LAPACK's stage-2 schedule (UPLO='L').

    This keeps the same arithmetic as LAPACK, but exposes the result as a
    stream of small, self-describing operations.
    """

    if n <= 2 or kd <= 1:
        return

    # LAPACK uses THGRSIZ=N, GRSIZ=1, SHIFT=3.
    thread_grid_size = n
    group_size = 1
    shift = 3
    steps_per_column = _ceil_div(shift, group_size)
    num_thread_grids = _ceil_div(n - 1, thread_grid_size)

    for grid_id in range(1, num_thread_grids + 1):
        start_sweep_1b = (grid_id - 1) * thread_grid_size + 1
        last_sweep_1b = min(start_sweep_1b + thread_grid_size - 1, n - 1)

        sweep_anchor_1b = start_sweep_1b
        global_step_1b = sweep_anchor_1b
        while global_step_1b <= (n - 1):
            end_sweep_1b = min(global_step_1b, last_sweep_1b)
            if sweep_anchor_1b > end_sweep_1b:
                break

            for step_in_column_1b in range(1, steps_per_column + 1):
                sweep_id_1b = sweep_anchor_1b
                while sweep_id_1b <= end_sweep_1b:
                    # LAPACK's task id / type logic.
                    task_id = (global_step_1b - sweep_id_1b) * (steps_per_column * group_size) + (
                        step_in_column_1b - 1
                    ) * group_size + 1

                    if task_id == 1:
                        kernel_type = KernelType.START
                    else:
                        kernel_type = KernelType((task_id % 2) + 2)  # 2 or 3

                    if kernel_type == KernelType.UPDATE_AND_GENERATE:
                        bulge_colpt_1b = (task_id // 2) * kd + sweep_id_1b
                        block_start_1b = bulge_colpt_1b - kd + 1
                        block_end_1b = min(bulge_colpt_1b, n)
                        last_index_1b = bulge_colpt_1b
                    else:
                        bulge_colpt_1b = ((task_id + 1) // 2) * kd + sweep_id_1b
                        block_start_1b = bulge_colpt_1b - kd + 1
                        block_end_1b = min(bulge_colpt_1b, n)
                        if (block_start_1b >= block_end_1b - 1) and (block_end_1b == n):
                            last_index_1b = n
                        else:
                            last_index_1b = 0

                    # Defensive bounds near the edges.
                    if block_start_1b >= 2 and block_end_1b >= block_start_1b:
                        yield KernelCall(
                            kernel_type=kernel_type,
                            sweep_id=sweep_id_1b,
                            block_start=block_start_1b - 1,
                            block_end=block_end_1b - 1,
                        )

                    if last_index_1b >= (n - 1):
                        sweep_anchor_1b += 1

                    sweep_id_1b += 1

            global_step_1b += 1


def _sb2st_lapack_schedule_inplace(
    ABw: np.ndarray,
    *,
    kd: int,
    kd_work: int,
    hous2: Optional[Hous2],
    check_fill: bool,
    fill_tol: float,
) -> None:
    """Run the LAPACK-like SB2ST bulge-chasing schedule in-place (UPLO='L')."""

    n = ABw.shape[1]
    if n <= 2 or kd <= 1:
        return

    # Minimal structural requirement for the stage-2 bulge workspace.
    if kd_work < min(n - 1, 2 * kd):
        raise ValueError("kd_work must be >= 2*kd for the SB2ST schedule")

    reflector_cache: list[Dict[int, Reflector]] = [dict(), dict()]

    for call in _iter_lapack_style_schedule_calls(n, kd):
        _run_kernel_lower(
            ABw,
            kd=kd,
            kd_work=kd_work,
            call=call,
            reflector_cache=reflector_cache,
            hous2=hous2,
        )

        if check_fill:
            # During the chase, nonzeros beyond the first subdiagonal are expected.
            # We keep this hook to help catch obvious regressions (e.g. NaNs).
            if not np.isfinite(_max_off_tridiagonal_in_band(ABw)):
                raise ValueError("non-finite values encountered during SB2ST chase")


def _max_off_tridiagonal_in_band(ABw: np.ndarray) -> float:
    """Max magnitude of entries outside the first subdiagonal in lower-band storage."""
    if ABw.shape[0] <= 2:
        return 0.0
    tail = ABw[2:, :]
    return float(np.max(np.abs(tail))) if tail.size else 0.0
