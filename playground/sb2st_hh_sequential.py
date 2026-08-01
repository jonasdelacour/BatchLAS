"""Sequential (un-pipelined) Householder SB2ST schedule + reflector store.

`householder_sb2st.py` reproduces LAPACK's pipelined schedule (THGRSIZ=N,
GRSIZ=1, SHIFT=3), which exists to give *multi-core CPU* parallelism: sweep s+1
can start once sweep s is 3 tasks ahead. On a GPU we get parallelism from the
batch dimension and from lanes inside each small window instead, and one
work-group processes its problem sequentially anyway. So the pipelining buys
nothing and costs a lot of index arithmetic to port.

This module implements the plain sequential schedule -- for each sweep, do the
TYPE1 elimination then chase the bulge to the bottom -- and is validated against
the same criteria as the pipelined one (see validate_hous2_q.py):

    Q = H_1 H_2 ... H_m  in generation order,  Q^H A Q tridiagonal,
    diag = d, |subdiag| = e.

All windows are at most kd x kd, which is what makes this portable to a
work-group-resident kernel.

Run: python3 playground/sb2st_hh_sequential.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from householder_sb2st import (  # noqa: E402
    apply_householder_left_dense,
    apply_householder_right_dense,
    apply_householder_two_sided_dense,
    band_get,
    band_set,
    dense_to_lower_band,
    householder,
)
from test_householder_sb2st import make_random_hermitian_banded  # noqa: E402


@dataclass
class ReflectorStore:
    """One reflector per entry: acts on rows [start, start + len(v))."""

    starts: List[int] = field(default_factory=list)
    vs: List[np.ndarray] = field(default_factory=list)
    taus: List[np.generic] = field(default_factory=list)

    def add(self, start: int, v: np.ndarray, tau: np.generic) -> None:
        self.starts.append(start)
        self.vs.append(v.copy())
        self.taus.append(tau)


def _get_window(ABw, kdw, r0, r1, c0, c1):
    """Dense copy of A[r0:r1+1, c0:c1+1] from band storage."""
    W = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=ABw.dtype)
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            W[i - r0, j - c0] = band_get(ABw, kdw, i, j)
    return W


def _put_window(ABw, kdw, r0, r1, c0, c1, W) -> None:
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            band_set(ABw, kdw, i, j, W[i - r0, j - c0])


def sb2st_hh_sequential(
    AB: np.ndarray, kd: int
) -> Tuple[np.ndarray, np.ndarray, ReflectorStore]:
    """Band -> tridiagonal by sequential Householder bulge chasing.

    Returns (d, e_signed, store). e_signed is the raw subdiagonal (no abs), so
    the caller can apply its own phase convention.
    """
    n = AB.shape[1]
    kdw = min(2 * kd, max(n - 1, 0))
    ABw = np.zeros((kdw + 1, n), dtype=AB.dtype)
    ABw[: kd + 1, :] = AB[: kd + 1, :]

    store = ReflectorStore()

    if kd > 1:
        for st in range(0, n - 2):
            r0 = st + 1
            r1 = min(st + kd, n - 1)
            if r1 <= r0:
                continue

            # --- TYPE 1: annihilate column st below the subdiagonal.
            x = np.array([band_get(ABw, kdw, i, st) for i in range(r0, r1 + 1)],
                         dtype=ABw.dtype)
            v, tau, beta = householder(x)
            store.add(r0, v, tau)
            band_set(ABw, kdw, r0, st, beta)
            for i in range(r0 + 1, r1 + 1):
                band_set(ABw, kdw, i, st, ABw.dtype.type(0))

            W = _get_window(ABw, kdw, r0, r1, r0, r1)
            apply_householder_two_sided_dense(W, v, tau)
            _put_window(ABw, kdw, r0, r1, r0, r1, W)

            # --- chase the bulge to the bottom.
            while True:
                p0 = r1 + 1
                p1 = min(r1 + kd, n - 1)
                if p0 > p1:
                    break

                # Right-apply the current reflector -> creates the bulge.
                Bw = _get_window(ABw, kdw, p0, p1, r0, r1)
                apply_householder_right_dense(Bw, v, tau)
                _put_window(ABw, kdw, p0, p1, r0, r1, Bw)

                # Annihilate the bulge's first column.
                x2 = np.array([band_get(ABw, kdw, i, r0) for i in range(p0, p1 + 1)],
                              dtype=ABw.dtype)
                v2, tau2, beta2 = householder(x2)
                store.add(p0, v2, tau2)
                band_set(ABw, kdw, p0, r0, beta2)
                for i in range(p0 + 1, p1 + 1):
                    band_set(ABw, kdw, i, r0, ABw.dtype.type(0))

                if r1 >= r0 + 1:
                    Cw = _get_window(ABw, kdw, p0, p1, r0 + 1, r1)
                    apply_householder_left_dense(Cw, v2, tau2)
                    _put_window(ABw, kdw, p0, p1, r0 + 1, r1, Cw)

                Ww = _get_window(ABw, kdw, p0, p1, p0, p1)
                apply_householder_two_sided_dense(Ww, v2, tau2)
                _put_window(ABw, kdw, p0, p1, p0, p1, Ww)

                v, tau = v2, tau2
                r0, r1 = p0, p1

    d = np.array([np.real(band_get(ABw, kdw, i, i)) for i in range(n)], dtype=np.float64)
    e_signed = np.array([band_get(ABw, kdw, i + 1, i) for i in range(n - 1)],
                        dtype=ABw.dtype)
    return d, e_signed, store


def build_q(n: int, store: ReflectorStore, dtype) -> np.ndarray:
    Q = np.eye(n, dtype=dtype)
    for start, v, tau in zip(store.starts, store.vs, store.taus):
        if tau == 0:
            continue
        m = v.shape[0]
        block = Q[:, start : start + m]
        w = block @ v
        block -= np.outer(w, tau * v.conjugate())
    return Q


def main() -> int:
    failures = 0
    total = 0
    for dtype in (np.float64, np.complex128):
        for n in (16, 24, 32, 48, 64):
            for kd in (2, 3, 4, 8, 16):
                if kd >= n:
                    continue
                total += 1
                A = make_random_hermitian_banded(n, kd, dtype, seed=1234 + n + 10 * kd)
                AB = dense_to_lower_band(A, kd)
                d, e_signed, store = sb2st_hh_sequential(AB, kd)
                Q = build_q(n, store, A.dtype)
                B = Q.conjugate().T @ A @ Q

                scale = max(np.linalg.norm(A), 1.0)
                offtri = np.linalg.norm(
                    [B[i, j] for i in range(n) for j in range(n) if abs(i - j) > 1]
                ) / scale
                d_err = np.linalg.norm(np.real(np.diag(B)) - d) / scale
                sub = np.array([B[i + 1, i] for i in range(n - 1)])
                e_err = np.linalg.norm(sub - e_signed) / scale
                orth = np.linalg.norm(
                    Q.conjugate().T @ Q - np.eye(n, dtype=A.dtype)
                )

                # Spectrum check against the dense reference.
                T = np.zeros((n, n), dtype=np.float64)
                np.fill_diagonal(T, d)
                ae = np.abs(e_signed).astype(np.float64)
                T[np.arange(1, n), np.arange(0, n - 1)] = ae
                T[np.arange(0, n - 1), np.arange(1, n)] = ae
                spec = np.max(np.abs(np.linalg.eigvalsh(A) - np.linalg.eigvalsh(T))) / scale

                ok = max(offtri, d_err, e_err, orth, spec) < 1e-11
                if not ok:
                    failures += 1
                print(
                    f"{'ok' if ok else 'FAIL':4s} n={n:3d} kd={kd:2d} "
                    f"{np.dtype(dtype).name:11s} nrefl={len(store.starts):5d} "
                    f"offtri={offtri:.2e} d={d_err:.2e} e={e_err:.2e} "
                    f"orth={orth:.2e} spec={spec:.2e}"
                )

    print()
    if failures:
        print(f"{failures} of {total} cases FAILED")
        return 1
    print(f"all {total} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
