"""Validate that the HOUS2 reflector store from `sb2st_householder` actually
reconstructs the stage-2 similarity transform.

`test_householder_sb2st.py` only checks that the resulting (d,e) has the right
spectrum; it never checks the stored reflectors. But the whole point of storing
them is the eigenvector back-transform Z_A = Q2 @ Z_T, so the store has to
satisfy

    Q^H A Q = T        with   Q = H_1 H_2 ... H_m

where H_k = I - tau_k v_k v_k^H embedded at rows [start_k, start_k + len(v_k)),
and the H_k appear in *generation* order. This script checks exactly that, plus
orthogonality of Q, over the same case grid the unit test uses.

Run: python3 playground/validate_hous2_q.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from householder_sb2st import dense_to_lower_band, sb2st_householder  # noqa: E402
from test_householder_sb2st import make_random_hermitian_banded, tridiag_from_de  # noqa: E402


def build_q_from_hous2(n: int, hous2, dtype) -> np.ndarray:
    """Q = H_1 H_2 ... H_m, each H_k embedded at rows [start_k, start_k+len(v_k))."""
    Q = np.eye(n, dtype=dtype)
    for start, v, tau in zip(hous2.starts, hous2.vs, hous2.taus):
        if tau == 0:
            continue
        m = v.shape[0]
        # Q <- Q * H_k  (right-multiply keeps the product in generation order).
        block = Q[:, start : start + m]
        w = block @ v
        block -= np.outer(w, tau * v.conjugate())
    return Q


def run_case(n: int, kd: int, dtype, block_size: int) -> tuple[float, float, int]:
    A = make_random_hermitian_banded(n, kd, dtype, seed=1234 + n + 10 * kd)
    AB = dense_to_lower_band(A, kd)

    _, d, e, hous2 = sb2st_householder(
        AB,
        kd,
        pad=2 * kd,
        block_size=block_size,
        return_hous2=True,
        check_fill=True,
        max_sweeps=12,
    )

    Q = build_q_from_hous2(n, hous2, A.dtype)
    B = Q.conjugate().T @ A @ Q

    scale = max(np.linalg.norm(A), 1.0)

    # Is B tridiagonal at all? This isolates "is Q the right transform" from the
    # separate question of the sign/phase convention on the subdiagonal, since
    # _extract_tridiagonal_real takes e = |subdiag| (a diagonal similarity, not
    # part of Q).
    offtri = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                offtri[i, j] = True
    offtri_err = np.linalg.norm(B[offtri]) / scale

    # Compare magnitudes, which are phase-invariant.
    d_err = np.linalg.norm(np.real(np.diag(B)) - d) / scale
    sub = np.array([B[i + 1, i] for i in range(n - 1)])
    e_err = np.linalg.norm(np.abs(sub) - e) / scale

    orth_err = np.linalg.norm(Q.conjugate().T @ Q - np.eye(n, dtype=A.dtype))
    return offtri_err, d_err, e_err, orth_err, len(hous2.starts)


def main() -> int:
    cases = []
    for dtype in (np.float64, np.complex128):
        for bs in (1, 4):
            if dtype is np.complex128 and bs != 1:
                continue
            for n in (16, 24, 32):
                for kd in (2, 4, 8):
                    if kd >= n:
                        continue
                    cases.append((n, kd, dtype, bs))

    failures = []
    for n, kd, dtype, bs in cases:
        offtri, d_err, e_err, orth, nrefl = run_case(n, kd, dtype, bs)
        ok = offtri < 1e-10 and d_err < 1e-10 and e_err < 1e-10 and orth < 1e-10
        if not ok:
            failures.append((n, kd, np.dtype(dtype).name, bs))
        print(
            f"{'ok' if ok else 'FAIL':4s} n={n:3d} kd={kd:2d} {np.dtype(dtype).name:11s} bs={bs} "
            f"nrefl={nrefl:4d} offtri={offtri:.3e} d_err={d_err:.3e} "
            f"e_err={e_err:.3e} orth={orth:.3e}"
        )

    if failures:
        print(f"\n{len(failures)} of {len(cases)} cases FAILED")
        return 1
    print(f"\nall {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
