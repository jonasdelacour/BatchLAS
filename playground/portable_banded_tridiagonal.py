"""portable_banded_tridiagonal

Portable, blocked Householder reduction of a symmetric/Hermitian banded matrix
to tridiagonal form.

Design goals
------------
1) Portability / translatability: simple, explicit loop structure.
2) Banded-awareness: never forms the full dense matrix during reduction;
     operates on a band representation using only small local dense scratch.
3) Correctness: each step is a true similarity transform; self-test checks
     tridiagonality and eigenvalue preservation.

Algorithm
---------
This implements the paper’s Algorithm 1 ("bandr1"): a blocked QR factorization
of a tall-skinny band block, followed by structured Pre/Sym/Post updates that
chase the bulge while keeping fill bounded within a working band.

Householder application strategy
-------------------------------
Within each QR(B) step, we use compact WY (V,T) to apply the Householder
reflectors to Pre/Sym/Post in one shot ("delayed application").
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------------
# Band storage (lower), helpers (Hermitian/symmetric)
# -----------------------------------------------------------------------------
# AB[r, j] = A[j+r, j] for r=0..kd.


def dense_to_lower_band(A: np.ndarray, kd: int) -> np.ndarray:
    A = np.asarray(A)
    n = A.shape[0]
    AB = np.zeros((kd + 1, n), dtype=A.dtype)
    for j in range(n):
        rmax = min(kd, n - 1 - j)
        for r in range(rmax + 1):
            AB[r, j] = A[j + r, j]
    return AB


def lower_band_to_dense(AB: np.ndarray, kd: int) -> np.ndarray:
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


def band_get(AB: np.ndarray, kd: int, i: int, j: int) -> np.generic:
    n = AB.shape[1]
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError("i/j out of range")
    if i < j:
        return np.conjugate(band_get(AB, kd, j, i))
    r = i - j
    if r > kd:
        return AB.dtype.type(0)
    return AB[r, j]


def band_set(AB: np.ndarray, kd: int, i: int, j: int, val: np.generic) -> None:
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


def extract_dense_window(AB: np.ndarray, kd: int, w0: int, w1: int) -> np.ndarray:
    W = np.empty((w1 - w0, w1 - w0), dtype=AB.dtype)
    for ii, i in enumerate(range(w0, w1)):
        for jj, j in enumerate(range(w0, w1)):
            W[ii, jj] = band_get(AB, kd, i, j)
    return W


def scatter_dense_window(AB: np.ndarray, kd: int, w0: int, W: np.ndarray) -> None:
    m = W.shape[0]
    for ii in range(m):
        i = w0 + ii
        for jj in range(m):
            j = w0 + jj
            if i >= j and (i - j) <= kd:
                band_set(AB, kd, i, j, W[ii, jj])


def extract_dense_block(
    AB: np.ndarray, kd: int, r0: int, r1: int, c0: int, c1: int
) -> np.ndarray:
    """Extract dense block A[r0:r1, c0:c1] using band_get."""
    W = np.empty((r1 - r0, c1 - c0), dtype=AB.dtype)
    for ii, i in enumerate(range(r0, r1)):
        for jj, j in enumerate(range(c0, c1)):
            W[ii, jj] = band_get(AB, kd, i, j)
    return W


def scatter_dense_block(
    AB: np.ndarray, kd: int, r0: int, r1: int, c0: int, c1: int, W: np.ndarray
) -> None:
    """Scatter dense block back to the stored lower band."""
    for ii, i in enumerate(range(r0, r1)):
        for jj, j in enumerate(range(c0, c1)):
            if i >= j and (i - j) <= kd:
                band_set(AB, kd, i, j, W[ii, jj])


# -----------------------------------------------------------------------------
# Householder primitives (real+complex)
# -----------------------------------------------------------------------------


def householder_vec(x: np.ndarray) -> Tuple[np.ndarray, np.generic, np.generic]:
    """Return (v, tau, beta) with (I - tau v v^H) x = [beta, 0, ...]^T and v[0]=1.

    This is a LARFG-style Householder constructor that works for real and
    complex vectors.
    """
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
    xnorm = float(np.linalg.norm(x_tail))

    if xnorm == 0.0 and (np.isrealobj(alpha) or (np.iscomplexobj(alpha) and alpha.imag == 0)):
        v = np.zeros_like(x)
        v[0] = x.dtype.type(1)
        return v, x.dtype.type(0), alpha

    if np.iscomplexobj(x):
        alpha_abs = float(np.abs(alpha))
        full_norm = float(np.hypot(alpha_abs, xnorm))
        alpha_phase = x.dtype.type(1) if alpha_abs == 0 else alpha / x.dtype.type(alpha_abs)
        beta = -alpha_phase * x.dtype.type(full_norm)
        tau = (beta - alpha) / beta
        v = x.copy()
        v0 = alpha - beta
        v /= v0
        v[0] = x.dtype.type(1)
        return v, tau, beta

    # Real
    alpha_r = float(alpha)
    beta_r = -np.copysign(np.hypot(alpha_r, xnorm), alpha_r)
    beta = x.dtype.type(beta_r)
    tau = (beta - alpha) / beta
    v = x.copy()
    v0 = alpha - beta
    v /= v0
    v[0] = x.dtype.type(1)
    return v, tau, beta


def _apply_householder_left(panel: np.ndarray, v: np.ndarray, tau: np.generic) -> None:
    """panel <- (I - tau v v^H) panel (acts on rows)."""
    if tau == 0:
        return
    vH = v.conjugate()
    w = vH @ panel
    panel -= np.outer(tau * v, w)


def _apply_householder_right(panel: np.ndarray, v: np.ndarray, tau: np.generic) -> None:
    """panel <- panel (I - tau v v^H)^H = panel (I - conj(tau) v v^H) (acts on cols)."""
    if tau == 0:
        return
    vH = v.conjugate()
    w = panel @ v
    panel -= np.outer(w, np.conjugate(tau) * vH)


def _enforce_hermitian_on_indices(W: np.ndarray, indices: range) -> None:
    """Enforce Hermitian symmetry only for rows/cols in indices (local)."""
    m = W.shape[0]
    for i in indices:
        # diagonal
        if np.iscomplexobj(W):
            W[i, i] = W[i, i].real + 0j
        # row/col
        for j in range(m):
            W[j, i] = np.conjugate(W[i, j])


def max_outside_band(W: np.ndarray, w0: int, kd_work: int) -> float:
    m = W.shape[0]
    maxv = 0.0
    for ii in range(m):
        gi = w0 + ii
        for jj in range(m):
            gj = w0 + jj
            if abs(gi - gj) > kd_work:
                maxv = max(maxv, float(abs(W[ii, jj])))
    return maxv


def window_bounds(n: int, kd_work: int, active_row0: int, active_row1: int) -> Tuple[int, int]:
    """Window [w0,w1) containing all indices coupled (within kd_work) to active rows."""
    w0 = max(0, active_row0 - kd_work)
    w1 = min(n, active_row1 + kd_work + 1)
    return w0, w1


@dataclass
class Reflectors:
    starts: List[int]
    vs: List[np.ndarray]
    taus: List[np.ndarray]


@dataclass
class QRBlock:
    """Householder QR factors for a tall-skinny block.

    We store the reflectors in a translation-friendly format: each step k
    produces a vector v (with v[0]=1) and a scalar tau, acting on rows k..m-1.
    """

    vs: List[np.ndarray]
    taus: List[np.ndarray]


def _qr_compact_wy(qr: QRBlock, m: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """Build compact WY representation for the QR reflectors.

    Given Householder vectors v_k (with v_k[0]=1) acting on rows k..m-1,
    returns (V, T) such that:

        Q = H0 H1 ... H_{r-1} = I - V T V^H

    where r = len(qr.vs), V is (m x r) and T is (r x r) upper-triangular.

    Then:
      - apply Q^H on the left:  C <- Q^H C = (I - V T^H V^H) C
      - apply Q   on the right: C <- C Q   = C (I - V T V^H)

    This is the standard “delayed application” / blocked Householder form.
    """
    r = len(qr.vs)
    if r == 0:
        return np.zeros((m, 0), dtype=dtype), np.zeros((0, 0), dtype=dtype)

    V = np.zeros((m, r), dtype=dtype)
    for k, v in enumerate(qr.vs):
        V[k:, k] = v

    T = np.zeros((r, r), dtype=dtype)
    for i, tau_arr in enumerate(qr.taus):
        tau = tau_arr.item()
        if tau == 0:
            continue
        T[i, i] = tau
        if i == 0:
            continue

        # LARFT (forward, columnwise) style build of T.
        vi = V[i:, i]
        w = -tau * (V[i:, :i].conjugate().T @ vi)
        T[:i, i] = T[:i, :i] @ w

    return V, T


def _compact_wy_from_reflector_list(
    n: int,
    starts: List[int],
    vs: List[np.ndarray],
    taus: List[np.ndarray],
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build compact WY (V,T) for an arbitrary list of embedded Householders.

    Each reflector is H = I - tau v v^H, where v is stored with v[0]=1 on the
    active subvector and is embedded at rows start:start+len(v).

    Returns (V,T) such that product(H_0 ... H_{r-1}) = I - V T V^H.
    """
    r = len(vs)
    if r == 0:
        return np.zeros((n, 0), dtype=dtype), np.zeros((0, 0), dtype=dtype)

    V = np.zeros((n, r), dtype=dtype)
    for k, (start, v) in enumerate(zip(starts, vs)):
        if v.ndim != 1:
            raise ValueError("v must be 1D")
        end = start + int(v.shape[0])
        if not (0 <= start < n) or not (0 < end <= n):
            raise ValueError("reflector embedding out of bounds")
        V[start:end, k] = v

    T = np.zeros((r, r), dtype=dtype)
    for i, tau_arr in enumerate(taus):
        tau = tau_arr.item()
        if tau == 0:
            continue
        T[i, i] = tau
        if i == 0:
            continue
        vi = V[:, i]
        w = -tau * (V[:, :i].conjugate().T @ vi)
        T[:i, i] = T[:i, :i] @ w

    return V, T


def _apply_compact_wy_left_qh(C: np.ndarray, V: np.ndarray, T: np.ndarray) -> None:
    """C <- Q^H C, with Q = I - V T V^H."""
    if V.size == 0:
        return
    W = V.conjugate().T @ C
    W = T.conjugate().T @ W
    C -= V @ W


def _apply_compact_wy_left_q(C: np.ndarray, V: np.ndarray, T: np.ndarray) -> None:
    """C <- Q C, with Q = I - V T V^H."""
    if V.size == 0:
        return
    W = V.conjugate().T @ C
    W = T @ W
    C -= V @ W


def _apply_compact_wy_right_q(C: np.ndarray, V: np.ndarray, T: np.ndarray) -> None:
    """C <- C Q, with Q = I - V T V^H."""
    if V.size == 0:
        return
    W = C @ V
    W = W @ T
    C -= W @ V.conjugate().T


def _qr_householder_factor_inplace(B: np.ndarray) -> QRBlock:
    """In-place Householder QR on a dense block B.

    After return, B has R in its upper trapezoid and zeros below, and the
    returned QRBlock contains the reflectors (v,tau) needed to apply Q^T (left)
    and Q (right) without explicitly forming Q.
    """
    m, n = B.shape
    r = min(m, n)
    vs: List[np.ndarray] = []
    taus: List[np.ndarray] = []
    for k in range(r):
        x = B[k:, k].copy()
        v, tau, beta = householder_vec(x)
        vs.append(v.astype(B.dtype, copy=False))
        taus.append(np.array(tau, dtype=B.dtype))
        if tau != 0:
            panel = B[k:, k:]
            _apply_householder_left(panel, v, tau)
            B[k:, k:] = panel
        B[k, k] = beta
        if k + 1 < m:
            B[k + 1 :, k] = 0
    return QRBlock(vs=vs, taus=taus)


def _apply_qt_left_inplace(C: np.ndarray, qr: QRBlock) -> None:
    """Apply Q^T (or Q^H) from QR to C on the left: C <- Q^T C."""
    for k, (v, tau_arr) in enumerate(zip(qr.vs, qr.taus)):
        tau = tau_arr.item()
        if tau == 0:
            continue
        panel = C[k:, :]
        _apply_householder_left(panel, v, tau)
        C[k:, :] = panel


def _apply_q_right_inplace(C: np.ndarray, qr: QRBlock) -> None:
    """Apply Q from QR to C on the right: C <- C Q.

    In our QR construction we apply reflectors H0, H1, ... on the left, so
    Q^T = H_{r-1}...H_0 and Q = (Q^T)^T = H_0 ... H_{r-1}.
    Therefore C <- C Q applies reflectors in *forward* order.
    """
    for k, (v, tau_arr) in enumerate(zip(qr.vs, qr.taus)):
        tau = tau_arr.item()
        if tau == 0:
            continue
        panel = C[:, k:]
        _apply_householder_right(panel, v, tau)
        C[:, k:] = panel


def bandr1_inplace(
    ABw: np.ndarray,
    *,
    b: int,
    d: int,
    nb: int,
    kd_work: int,
    fill_tol: float,
    record: Optional[Reflectors] = None,
) -> int:
    """Algorithm 1 (bandr1) with compact-WY (delayed) Householder application.

    This variant stores all Householder reflectors generated by each QR(B)
    factorization and applies them to Pre/Sym/Post in *one shot* using a compact
    WY (V,T) representation. This is the typical “delayed application” strategy
    used to turn a sequence of Householder updates into BLAS-3 friendly blocks.

    Note: due to bulge-chasing dependencies, we still must apply the resulting
    Q (to Pre/Sym/Post) before advancing to the next chase step.
    """
    n = ABw.shape[1]
    if not (1 <= d < b):
        raise ValueError("need 1 <= d < b")
    if not (1 <= nb <= b - d):
        raise ValueError("need 1 <= nb <= b-d")

    b_tilde = b - d

    for j1 in range(0, max(0, n - b_tilde - 1), nb):
        j2 = min(j1 + nb - 1, n - 1)
        i1 = j1 + b_tilde
        i2 = min(j1 + b + nb - 1, n - 1)

        while i1 < n:
            if i1 > i2:
                break

            B = extract_dense_block(ABw, kd_work, i1, i2 + 1, j1, j2 + 1)
            qr = _qr_householder_factor_inplace(B)
            V, T = _qr_compact_wy(qr, m=B.shape[0], dtype=B.dtype)
            scatter_dense_block(ABw, kd_work, i1, i2 + 1, j1, j2 + 1, B)

            if record is not None:
                for k, (v, tau_arr) in enumerate(zip(qr.vs, qr.taus)):
                    tau = tau_arr.item()
                    if tau == 0:
                        continue
                    record.starts.append(i1 + k)
                    record.vs.append(v.copy())
                    record.taus.append(np.array(tau, dtype=ABw.dtype))

            # Pre: A(i1:i2, j2+1:i1-1) <- Q^H * that
            c0 = j2 + 1
            c1 = i1 - 1
            if c0 <= c1:
                Pre = extract_dense_block(ABw, kd_work, i1, i2 + 1, c0, c1 + 1)
                _apply_compact_wy_left_qh(Pre, V, T)
                scatter_dense_block(ABw, kd_work, i1, i2 + 1, c0, c1 + 1, Pre)

            # Sym: A(i1:i2, i1:i2) <- Q^H * A * Q
            Sym = extract_dense_block(ABw, kd_work, i1, i2 + 1, i1, i2 + 1)
            _apply_compact_wy_left_qh(Sym, V, T)
            _apply_compact_wy_right_q(Sym, V, T)
            if np.iscomplexobj(Sym):
                np.fill_diagonal(Sym, Sym.diagonal().real)
            scatter_dense_block(ABw, kd_work, i1, i2 + 1, i1, i2 + 1, Sym)

            # Post: A(i2+1:min(i2+b,n), i1:i2) <- that * Q
            r0 = i2 + 1
            r1 = min(i2 + b, n - 1)
            if r0 <= r1:
                Post = extract_dense_block(ABw, kd_work, r0, r1 + 1, i1, i2 + 1)
                _apply_compact_wy_right_q(Post, V, T)
                scatter_dense_block(ABw, kd_work, r0, r1 + 1, i1, i2 + 1, Post)

            new_j1 = i1
            new_j2 = min(new_j1 + nb - 1, n - 1)
            new_i1 = i1 + b
            new_i2 = min(i2 + b, n - 1)
            j1, j2, i1, i2 = new_j1, new_j2, new_i1, new_i2

    if b_tilde + 1 <= kd_work:
        ABw[b_tilde + 1 :, :] = 0

    if b_tilde >= 2 and ABw[b_tilde + 1 :, :].size:
        max_out = float(np.max(np.abs(ABw[b_tilde + 1 :, :])))
        if max_out > fill_tol:
            raise ValueError(
                f"band reduction did not stay within b_tilde={b_tilde}: max spill={max_out:.3e}"
            )

    return b_tilde


# Back-compat alias (older name used during development).
bandr1_inplace_delayed = bandr1_inplace


def banded_to_tridiagonal_blocked(
    AB_in: np.ndarray,
    kd: int,
    *,
    pad: Optional[int] = None,
    block_size: int = 8,
    max_sweeps: Optional[int] = None,
    fill_tol: Optional[float] = None,
    return_reflectors: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Reflectors]]:
    """Reduce Hermitian/symmetric lower-band AB (kd+1,n) to real tridiagonal (d,e).

    This repeatedly applies Algorithm 1 ("bandr1") until semibandwidth reaches 1.
    Within each QR(B) step we use compact WY (V,T) to apply the Householder
    reflectors to Pre/Sym/Post in one shot.
    """
    AB = np.asarray(AB_in)
    if AB.ndim != 2:
        raise ValueError("AB_in must be 2D")
    if kd < 0:
        raise ValueError("kd must be >= 0")
    if AB.shape[0] != kd + 1:
        raise ValueError("AB_in must have shape (kd+1, n)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n = AB.shape[1]
    if n <= 1:
        d = AB[0, :].real.astype(np.float64, copy=False)
        e = np.zeros((0,), dtype=np.float64)
        return d, e, Reflectors([], [], []) if return_reflectors else None

    if pad is None:
        pad = kd
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be >= 0")

    kd_work = min(n - 1, 2 * kd + pad)
    if kd_work < min(n - 1, 2 * kd):
        raise ValueError(
            f"pad too small: need kd_work>=2*kd (kd={kd}, kd_work={kd_work})."
        )

    if max_sweeps is None:
        max_sweeps = max(0, kd - 1)
    if fill_tol is None:
        eps = np.finfo(np.float64).eps
        fill_tol = 5000.0 * eps

    ABw = np.zeros((kd_work + 1, n), dtype=AB.dtype)
    ABw[: kd + 1, :] = AB

    refl = Reflectors([], [], []) if return_reflectors else None

    b = kd
    sweeps = 0
    while b > 1:
        if sweeps >= int(max_sweeps):
            break

        nb_target = int(block_size)
        b_tilde_target = max(1, min(nb_target, b - 1))
        d_red = b - b_tilde_target
        nb = min(nb_target, b - d_red)

        b = bandr1_inplace(
            ABw,
            b=b,
            d=d_red,
            nb=nb,
            kd_work=kd_work,
            fill_tol=float(fill_tol),
            record=refl,
        )
        sweeps += 1

    if kd_work >= 2:
        max_off = float(np.max(np.abs(ABw[2:, :]))) if ABw[2:, :].size else 0.0
        if max_off > float(fill_tol) * 10.0:
            raise ValueError(
                f"did not reach tridiagonal: max |off-tridiag|={max_off:.3e}. "
                "Increase kd_work (pad) or reduce block_size."
            )

    d_out = ABw[0, :].real.astype(np.float64, copy=True)
    if n <= 1:
        e_out = np.zeros((0,), dtype=np.float64)
    else:
        e_out = np.abs(ABw[1, : n - 1]).astype(np.float64, copy=True)
    return d_out, e_out, refl


# Back-compat alias.
banded_to_tridiagonal_blocked_delayed = banded_to_tridiagonal_blocked


def form_q_from_reflectors(
    n: int,
    refl: Reflectors,
    dtype: np.dtype,
    *,
    block_size: int = 0,
) -> np.ndarray:
    """Build dense Q such that A_tridiag = Q^H A Q for the recorded reflectors.

    If block_size > 0, forms Q in a blocked fashion using compact WY batches.
    """
    if block_size is None:
        block_size = 0
    block_size = int(block_size)

    if block_size <= 0:
        Q_left = np.eye(n, dtype=dtype)
        for start, v, tau_arr in zip(refl.starts, refl.vs, refl.taus):
            tau = tau_arr.item()
            if tau == 0:
                continue
            rows = Q_left[start : start + v.shape[0], :]
            _apply_householder_left(rows, v, tau)
            Q_left[start : start + v.shape[0], :] = rows
        return Q_left.conjugate().T

    Q_left = np.eye(n, dtype=dtype)
    m = len(refl.vs)
    for p in range(0, m, block_size):
        q = min(p + block_size, m)

        # Sequential application over [p,q) left-multiplies by H_{q-1} ... H_p.
        # Build a compact WY representation of that product.
        starts = refl.starts[p:q][::-1]
        vs = refl.vs[p:q][::-1]
        taus = refl.taus[p:q][::-1]
        V, T = _compact_wy_from_reflector_list(n, starts, vs, taus, dtype=dtype)
        _apply_compact_wy_left_q(Q_left, V, T)

    return Q_left.conjugate().T

def is_tridiagonal(M: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a Hermitian/symmetric matrix is tridiagonal within a tolerance."""
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(M[i, j]) > tol:
                return False
    return True

def test_tridiagonalization():
    """Self-check: structure + eigenvalues for real and complex Hermitian."""
    rng = np.random.default_rng(0)

    def make_random_hermitian_banded(n: int, kd: int, dtype: np.dtype) -> np.ndarray:
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
            raise ValueError(dtype)

        for i in range(n):
            for j in range(n):
                if abs(i - j) > kd:
                    A[i, j] = 0
        A = 0.5 * (A + A.conjugate().T)
        if np.iscomplexobj(A):
            np.fill_diagonal(A, A.diagonal().real)
        return A

    for dtype in (np.float64, np.complex128):
        for n in (24, 32):
            for kd in (2, 4, 8):
                if kd >= n:
                    continue

                A = make_random_hermitian_banded(n, kd, dtype)
                AB = dense_to_lower_band(A, kd)

                d, e, refl = banded_to_tridiagonal_blocked(
                    AB,
                    kd,
                    pad=kd,
                    block_size=4,
                    return_reflectors=True,
                    fill_tol=1e-12,
                )

                T = np.diag(d) + np.diag(e, k=-1) + np.diag(e, k=1)
                assert is_tridiagonal(T, tol=1e-10)
                assert np.linalg.norm(T - T.T) < 1e-12

                wA = np.linalg.eigvalsh(A)
                wT = np.linalg.eigvalsh(T)
                np.testing.assert_allclose(wA, wT, rtol=5e-8, atol=5e-8)

                # Stronger check: build explicit Q (blocked) and verify similarity.
                Q = form_q_from_reflectors(n, refl, dtype=A.dtype, block_size=16)
                A2 = Q.conjugate().T @ A @ Q
                assert is_tridiagonal(A2, tol=1e-8)

                # Fix phases/signs so off-diagonals are nonnegative real.
                t = np.diag(A2, k=1)
                phases = np.ones((n,), dtype=A2.dtype)
                for k in range(n - 1):
                    tk = t[k]
                    at = float(np.abs(tk))
                    if at == 0.0:
                        phases[k + 1] = phases[k]
                    else:
                        phases[k + 1] = phases[k] * (np.conjugate(tk) / at)
                D = np.diag(phases)
                A3 = D.conjugate().T @ A2 @ D
                assert is_tridiagonal(A3, tol=1e-8)
                if np.iscomplexobj(A3):
                    np.fill_diagonal(A3, A3.diagonal().real)
                np.testing.assert_allclose(np.diag(A3).real, d, rtol=1e-7, atol=1e-7)
                if n > 1:
                    np.testing.assert_allclose(np.diag(A3, k=1).real, e, rtol=1e-7, atol=1e-7)

    print("All tests passed.")

if __name__ == "__main__":
    test_tridiagonalization()

