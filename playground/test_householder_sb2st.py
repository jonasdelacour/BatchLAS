import unittest

import os
import sys

import numpy as np

# Ensure `householder_sb2st.py` in this folder is importable when tests are run
# from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

from householder_sb2st import dense_to_lower_band, lower_band_to_dense, sb2st_householder


def make_random_hermitian_banded(n: int, kd: int, dtype: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype) == np.float64:
        A = rng.standard_normal((n, n), dtype=np.float64)
        A = 0.5 * (A + A.T)
    elif np.dtype(dtype) == np.complex128:
        Ar = rng.standard_normal((n, n), dtype=np.float64)
        Ai = rng.standard_normal((n, n), dtype=np.float64)
        A = (Ar + 1j * Ai).astype(np.complex128)
        A = 0.5 * (A + A.conjugate().T)
        np.fill_diagonal(A, A.diagonal().real)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    # Enforce bandedness |i-j| <= kd.
    for i in range(n):
        for j in range(n):
            if abs(i - j) > kd:
                A[i, j] = 0
    # Re-enforce Hermitian after zeroing.
    A = 0.5 * (A + A.conjugate().T)
    if np.iscomplexobj(A):
        np.fill_diagonal(A, A.diagonal().real)
    return A.astype(dtype, copy=False)


def tridiag_from_de(d: np.ndarray, e: np.ndarray) -> np.ndarray:
    n = d.shape[0]
    T = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(T, d)
    if n > 1:
        T[np.arange(1, n), np.arange(0, n - 1)] = e
        T[np.arange(0, n - 1), np.arange(1, n)] = e
    return T


class TestHouseholderSB2ST(unittest.TestCase):
    def _run_case(self, n: int, kd: int, dtype: np.dtype, block_size: int) -> None:
        A = make_random_hermitian_banded(n, kd, dtype, seed=1234 + n + 10 * kd)
        AB = dense_to_lower_band(A, kd)

        AB_tri, d, e, hous2 = sb2st_householder(
            AB,
            kd,
            pad=2 * kd,  # expanded band to hold transient bulge fill
            block_size=block_size,
            return_hous2=True,
            check_fill=True,
            max_sweeps=12,
        )

        # Basic shape checks
        self.assertEqual(AB_tri.shape, (2, n))
        self.assertEqual(d.shape, (n,))
        self.assertEqual(e.shape, (max(0, n - 1),))

        # d,e must be real float64
        self.assertEqual(d.dtype, np.float64)
        self.assertEqual(e.dtype, np.float64)

        # Reconstruct T (real tridiagonal) and verify it is symmetric tridiagonal.
        T = tridiag_from_de(d, e)
        self.assertLess(np.linalg.norm(T - T.T), 1e-12)
        mask = np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= 1:
                    mask[i, j] = False
        self.assertLess(np.linalg.norm(T[mask]), 1e-10)

        # Eigenvalue check vs dense reference.
        # A is Hermitian/symmetric; eigvalsh is appropriate.
        wA = np.linalg.eigvalsh(A)
        wT = np.linalg.eigvalsh(T)
        # Relative tolerance scaling: allow a bit more for larger n.
        # Windowed bulge chasing is a similarity transform when pad is sufficient;
        # allow modest tolerance due to repeated window symmetrization.
        rtol = 5e-8
        atol = 5e-8
        np.testing.assert_allclose(wA, wT, rtol=rtol, atol=atol)

        # HOUS2 store sanity: number of reflectors is n-2 (some may be skipped if m<=1)
        self.assertEqual(len(hous2.starts), len(hous2.vs))
        self.assertEqual(len(hous2.starts), len(hous2.taus))

    def test_float64_unblocked(self):
        for n in (16, 24, 32):
            for kd in (2, 4, 8):
                if kd >= n:
                    continue
                self._run_case(n, kd, np.float64, block_size=1)

    def test_complex128_unblocked(self):
        for n in (16, 24, 32):
            for kd in (2, 4, 8):
                if kd >= n:
                    continue
                self._run_case(n, kd, np.complex128, block_size=1)

    def test_float64_blocked_delayed_scatter(self):
        for n in (16, 24, 32):
            for kd in (2, 4, 8):
                if kd >= n:
                    continue
                self._run_case(n, kd, np.float64, block_size=4)


if __name__ == "__main__":
    unittest.main()
