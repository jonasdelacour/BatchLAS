"""householder_sb2st_legacy.py

Legacy / exploratory bulge-chasing implementations for band → tridiagonal.

These routines were kept around during development for debugging and comparison,
but they are *not* the primary SB2ST reference anymore.

The main implementation lives in `householder_sb2st.py` and follows a LAPACK-like
SB2ST stage-2 schedule (lower-band).

Nothing in the main path imports this module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from householder_sb2st import (
    Hous2,
    apply_householder_two_sided_dense,
    band_get,
    extract_dense_window,
    householder,
    scatter_dense_window,
)


def _window_bounds_fullband(n: int, kd_work: int, active_row0: int, active_row1: int) -> Tuple[int, int]:
    """Window [w0,w1) that contains all nonzeros coupled to active indices."""
    w0 = max(0, active_row0 - kd_work)
    w1 = min(n, active_row1 + kd_work + 1)
    return w0, w1


def _max_outside_band(W: np.ndarray, w0: int, kd_work: int) -> float:
    """Max |W[i,j]| where global |(w0+i)-(w0+j)| > kd_work."""
    m = W.shape[0]
    maxv = 0.0
    for ii in range(m):
        gi = w0 + ii
        for jj in range(m):
            gj = w0 + jj
            if abs(gi - gj) > kd_work:
                maxv = max(maxv, float(np.abs(W[ii, jj])))
    return maxv


def _enforce_hermitian_inplace(W: np.ndarray) -> None:
    W[:, :] = 0.5 * (W + W.conjugate().T)
    if np.iscomplexobj(W):
        np.fill_diagonal(W, W.diagonal().real)


def sb2st_apply_one_step_legacy(
    ABw: np.ndarray,
    *,
    step_col: int,
    kd: int,
    kd_work: int,
    hous2: Optional[Hous2],
    check_fill: bool,
    fill_tol: float,
) -> None:
    """Apply one short Householder step (legacy bulge chase) in-place to ABw."""
    n = ABw.shape[1]
    m = min(kd, n - 1 - step_col)
    if m <= 1:
        return

    active_row0 = step_col + 1
    active_row1 = step_col + m

    x = np.array([band_get(ABw, kd_work, active_row0 + t, step_col) for t in range(m)], dtype=ABw.dtype)
    v, tau, _beta = householder(x)
    if tau == 0:
        return

    w0, w1 = _window_bounds_fullband(n, kd_work, active_row0, active_row1)
    W = extract_dense_window(ABw, kd_work, w0, w1, w0, w1)

    u = np.zeros((w1 - w0,), dtype=ABw.dtype)
    u[active_row0 - w0 : active_row1 - w0 + 1] = v
    apply_householder_two_sided_dense(W, u, tau)
    _enforce_hermitian_inplace(W)

    if check_fill:
        max_fill = _max_outside_band(W, w0, kd_work)
        if max_fill > fill_tol:
            raise ValueError(
                f"bulge fill exceeded kd_work={kd_work}: max |outside-band|={max_fill:.3e}. "
                f"Increase pad (current pad={kd_work - kd})."
            )

    scatter_dense_window(ABw, kd_work, w0, w1, w0, w1, W)

    if np.iscomplexobj(ABw):
        ABw[0, step_col] = ABw[0, step_col].real.astype(ABw.dtype)

    if hous2 is not None:
        hous2.starts.append(active_row0)
        hous2.vs.append(v.copy())
        hous2.taus.append(np.array(tau, dtype=ABw.dtype))


def sb2st_unblocked_bulge_chase_legacy(
    ABw: np.ndarray,
    *,
    kd: int,
    kd_work: int,
    hous2: Optional[Hous2],
    check_fill: bool,
    fill_tol: float,
    max_sweeps: int,
) -> None:
    """Legacy unblocked chase: repeated one-step updates."""
    n = ABw.shape[1]
    for _sweep in range(max_sweeps):
        for j in range(0, max(0, n - 2)):
            sb2st_apply_one_step_legacy(
                ABw,
                step_col=j,
                kd=kd,
                kd_work=kd_work,
                hous2=hous2,
                check_fill=check_fill,
                fill_tol=fill_tol,
            )
        if ABw.shape[0] <= 2 or float(np.max(np.abs(ABw[2:, :]))) <= fill_tol * 10.0:
            break


def sb2st_blocked_bulge_chase_legacy(
    ABw: np.ndarray,
    *,
    kd: int,
    kd_work: int,
    block_size: int,
    hous2: Optional[Hous2],
    check_fill: bool,
    fill_tol: float,
    max_sweeps: int,
) -> None:
    """Legacy blocked chase: process a chunk in one extracted window."""
    n = ABw.shape[1]
    for _sweep in range(max_sweeps):
        j = 0
        while j < max(0, n - 2):
            b = min(block_size, (n - 2) - j)

            active_row0 = j + 1
            active_row1 = (j + b - 1) + min(kd, n - 1 - (j + b - 1))
            w0, w1 = _window_bounds_fullband(n, kd_work, active_row0, active_row1)
            W = extract_dense_window(ABw, kd_work, w0, w1, w0, w1)

            for jj in range(b):
                step_col = j + jj
                m = min(kd, n - 1 - step_col)
                if m <= 1:
                    continue

                a0 = step_col + 1
                a1 = step_col + m

                col = step_col - w0
                x = W[(a0 - w0) : (a1 - w0 + 1), col].copy()
                v, tau, _beta = householder(x)
                if tau == 0:
                    continue

                u = np.zeros((w1 - w0,), dtype=ABw.dtype)
                u[a0 - w0 : a1 - w0 + 1] = v
                apply_householder_two_sided_dense(W, u, tau)
                _enforce_hermitian_inplace(W)

                if hous2 is not None:
                    hous2.starts.append(a0)
                    hous2.vs.append(v.copy())
                    hous2.taus.append(np.array(tau, dtype=ABw.dtype))

            if check_fill:
                max_fill = _max_outside_band(W, w0, kd_work)
                if max_fill > fill_tol:
                    raise ValueError(
                        f"bulge fill exceeded kd_work={kd_work}: max |outside-band|={max_fill:.3e}. "
                        f"Increase pad (current pad={kd_work - kd})."
                    )

            scatter_dense_window(ABw, kd_work, w0, w1, w0, w1, W)
            j += b

        if ABw.shape[0] <= 2 or float(np.max(np.abs(ABw[2:, :]))) <= fill_tol * 10.0:
            break
