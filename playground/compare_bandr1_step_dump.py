#!/usr/bin/env python3
"""compare_bandr1_step_dump

Compares a single BANDR1 chase-step dump produced by the C++ implementation
against the portable Python reference implementation.

Usage:
  python3 playground/compare_bandr1_step_dump.py --dump-dir output/bandr1_dumps

It auto-discovers "ABw_band_before_...csv" files and, for each (step,batch),
replays the same step on the Python side, then reports max-abs differences for
available dumped blocks (B_after_keepR, Sym_after, Mid_after, Right_after,
Post_after, Pre_after) and ABw_band_after.

Environment on C++ side to produce dumps:
  BATCHLAS_DUMP_BANDR1_STEP=1
  BATCHLAS_DUMP_BANDR1_DIR=output/bandr1_dumps   (optional)
  BATCHLAS_DUMP_BANDR1_BATCH=0                  (optional)

Notes:
- Default QR path is the portable reference implementation in
    portable_banded_tridiagonal.py.
- You can alternatively use NumPy/LAPACK QR ("--qr numpy") to better match the
    C++ backend's sign/phase conventions for R.
- We do NOT compare the post-GEQRF panel (it contains Householder vectors in the
    LAPACK format), only the R-only panel dump (B_after_keepR).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Make sure we can import the reference module when running from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "playground"))

import portable_banded_tridiagonal as ref  # noqa: E402


_STEP_RE = re.compile(
    r"ABw_band_before_i1(?P<i1>-?\d+)_i2(?P<i2>-?\d+)_j1(?P<j1>-?\d+)_j2(?P<j2>-?\d+)_b(?P<b>-?\d+)\.csv$"
)


def load_csv_matrix(path: str) -> np.ndarray:
    rows = None
    cols = None
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# rows,"):
                    parts = line.split(",")
                    # "# rows,<r>,cols,<c>"
                    rows = int(parts[1])
                    cols = int(parts[3])
                continue
            i_s, j_s, re_s, im_s = line.split(",")
            data.append((int(i_s), int(j_s), float(re_s), float(im_s)))

    if rows is None or cols is None:
        raise ValueError(f"missing header rows/cols in {path}")

    # Decide dtype: complex if any imag is non-zero.
    any_im = any(abs(im) != 0.0 for (_, _, _, im) in data)
    dtype = np.complex128 if any_im else np.float64
    out = np.zeros((rows, cols), dtype=dtype)
    for i, j, re_v, im_v in data:
        out[i, j] = re_v + (1j * im_v if any_im else 0.0)
    return out


@dataclass
class StepKey:
    i1: int
    i2: int
    j1: int
    j2: int
    batch: int


def discover_steps(dump_dir: str) -> list[StepKey]:
    keys: list[StepKey] = []
    for path in glob.glob(os.path.join(dump_dir, "ABw_band_before_*.csv")):
        m = _STEP_RE.search(os.path.basename(path))
        if not m:
            continue
        keys.append(
            StepKey(
                i1=int(m.group("i1")),
                i2=int(m.group("i2")),
                j1=int(m.group("j1")),
                j2=int(m.group("j2")),
                batch=int(m.group("b")),
            )
        )
    keys.sort(key=lambda k: (k.i1, k.j1, k.batch))
    return keys


def maybe_load(dump_dir: str, tag: str, key: StepKey) -> Optional[np.ndarray]:
    name = f"{tag}_i1{key.i1}_i2{key.i2}_j1{key.j1}_j2{key.j2}_b{key.batch}.csv"
    path = os.path.join(dump_dir, name)
    if not os.path.exists(path):
        return None
    return load_csv_matrix(path)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def band_storage_to_dense_hermitian(ab: np.ndarray) -> np.ndarray:
    """Reconstruct dense Hermitian matrix from lower-band storage.

    Storage format matches BatchLAS' ABw: shape (kd+1, n), where entry ab[r, j]
    stores A[j+r, j] (lower triangle), for 0 <= r <= kd.
    """
    if ab.ndim != 2:
        raise ValueError(f"expected 2D band storage, got shape={ab.shape}")
    kd = ab.shape[0] - 1
    n = ab.shape[1]
    if kd < 0 or n < 0:
        raise ValueError(f"invalid band storage shape={ab.shape}")

    dense = np.zeros((n, n), dtype=ab.dtype)
    for j in range(n):
        max_r = min(kd, (n - 1) - j)
        for r in range(max_r + 1):
            i = j + r
            v = ab[r, j]
            if i == j:
                # Diagonal of Hermitian matrix is real.
                dense[i, j] = np.real(v)
            else:
                dense[i, j] = v
                dense[j, i] = np.conjugate(v)
    return dense


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument(
        "--qr",
        choices=["reference", "numpy"],
        default="reference",
        help="QR implementation to use for replay: 'reference' (portable) or 'numpy' (LAPACK-backed)",
    )
    ap.add_argument(
        "--eigs",
        action="store_true",
        help="Also compute eigvalsh() of the reconstructed dense Hermitian matrices from ABw_band_before/after",
    )
    args = ap.parse_args()

    steps = discover_steps(args.dump_dir)
    if not steps:
        print(f"No ABw_band_before dumps found in {args.dump_dir}")
        return 2

    for key in steps:
        ABw_before = maybe_load(args.dump_dir, "ABw_band_before", key)
        if ABw_before is None:
            continue

        kd_work = ABw_before.shape[0] - 1
        n = ABw_before.shape[1]

        # Load optional block dumps (if present).
        B_keepR = maybe_load(args.dump_dir, "B_after_keepR", key)
        Sym_after = maybe_load(args.dump_dir, "Sym_after", key)
        Mid_after = maybe_load(args.dump_dir, "Mid_after", key)
        Right_after = maybe_load(args.dump_dir, "Right_after", key)
        Post_after = maybe_load(args.dump_dir, "Post_after", key)
        Pre_after = maybe_load(args.dump_dir, "Pre_after", key)
        ABw_after = maybe_load(args.dump_dir, "ABw_band_after", key)

        # Replay the step using the reference implementation.
        ABw = np.array(ABw_before, copy=True)

        # The reference uses a band API; operate directly on ABw.
        i1, i2, j1, j2 = key.i1, key.i2, key.j1, key.j2
        m = i2 - i1 + 1
        r = j2 - j1 + 1

        if m <= 0 or r <= 0:
            continue

        B = ref.extract_dense_block(ABw, kd_work, i1, i2 + 1, j1, j2 + 1)
        m_rows, n_cols = B.shape

        if args.qr == "reference":
            qr = ref._qr_householder_factor_inplace(B)
            V, T = ref._qr_compact_wy(qr, m=B.shape[0], dtype=B.dtype)

            # After QR, reference scatters B (R-only)
            ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, j1, j2 + 1, B)

            def apply_left_qh(C: np.ndarray) -> None:
                ref._apply_compact_wy_left_qh(C, V, T)

            def apply_right_q(C: np.ndarray) -> None:
                ref._apply_compact_wy_right_q(C, V, T)
        else:
            # Use NumPy/LAPACK QR to better match the backend QR convention.
            # Q is m×m, R is m×n (with zeros in the last m-n rows).
            Q, R = np.linalg.qr(B, mode="complete")
            B[:, :] = R
            ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, j1, j2 + 1, B)

            def apply_left_qh(C: np.ndarray) -> None:
                # C <- Q^H C
                C[:, :] = Q.conjugate().T @ C

            def apply_right_q(C: np.ndarray) -> None:
                # C <- C Q
                C[:, :] = C @ Q

        # Left-of-panel block: columns strictly left of j1 that are still within
        # the work band. This is required for later chase steps where j1 advances.
        left_c0 = max(0, i1 - kd_work)
        left_c1 = min(j1, i1)
        if left_c0 < left_c1:
            Left = ref.extract_dense_block(ABw, kd_work, i1, i2 + 1, left_c0, left_c1)
            apply_left_qh(Left)
            ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, left_c0, left_c1, Left)

        # Mid block (between B and Sym), restricted to what the band storage can represent.
        # The C++ implementation dumps only the in-band portion, so infer the exact
        # column window from the dump shape when available.
        if Mid_after is not None:
            mid_cols = Mid_after.shape[1]
            mid_c1 = i1
            mid_c0 = mid_c1 - mid_cols
            if mid_cols > 0:
                Mid = ref.extract_dense_block(ABw, kd_work, i1, i2 + 1, mid_c0, mid_c1)
                apply_left_qh(Mid)
                ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, mid_c0, mid_c1, Mid)
            else:
                Mid = None
        else:
            Mid = None

        # Sym block
        Sym = ref.extract_dense_block(ABw, kd_work, i1, i2 + 1, i1, i2 + 1)
        apply_left_qh(Sym)
        apply_right_q(Sym)
        if np.iscomplexobj(Sym):
            ref._enforce_hermitian_on_indices(Sym, range(Sym.shape[0]))
        ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, i1, i2 + 1, Sym)

        # Right block (i1:i2, i2+1:...)
        if Right_after is not None:
            right_cols = Right_after.shape[1]
            right_c0 = i2 + 1
            right_c1 = right_c0 + right_cols
            Right = ref.extract_dense_block(ABw, kd_work, i1, i2 + 1, right_c0, right_c1)
            apply_left_qh(Right)
            ref.scatter_dense_block(ABw, kd_work, i1, i2 + 1, right_c0, right_c1, Right)
        else:
            Right = None

        # Post block (below Sym)
        if Post_after is not None:
            post_rows = Post_after.shape[0]
            post_r0 = i2 + 1
            post_r1 = post_r0 + post_rows
            Post = ref.extract_dense_block(ABw, kd_work, post_r0, post_r1, i1, i2 + 1)
            apply_right_q(Post)
            ref.scatter_dense_block(ABw, kd_work, post_r0, post_r1, i1, i2 + 1, Post)
        else:
            Post = None

        # Pre block (above Sym, within kd_work)
        if Pre_after is not None:
            pre_rows = Pre_after.shape[0]
            pre_r1 = i1
            pre_r0 = pre_r1 - pre_rows
            Pre = ref.extract_dense_block(ABw, kd_work, pre_r0, pre_r1, i1, i2 + 1)
            apply_right_q(Pre)
            ref.scatter_dense_block(ABw, kd_work, pre_r0, pre_r1, i1, i2 + 1, Pre)
        else:
            Pre = None

        print(f"step i1={i1} i2={i2} j1={j1} j2={j2} batch={key.batch} kd_work={kd_work} n={n}")

        if args.eigs:
            A0 = band_storage_to_dense_hermitian(ABw_before)
            A1 = band_storage_to_dense_hermitian(ABw)
            w0 = np.linalg.eigvalsh(A0)
            w1 = np.linalg.eigvalsh(A1)
            abs_diff = float(np.max(np.abs(w1 - w0)))
            denom = np.maximum(1.0, np.max(np.abs(w0)))
            rel_diff = float(abs_diff / denom)
            print(f"  eig(before vs replay_after): max|Δ|={abs_diff:.3e}  rel={rel_diff:.3e}")

            if ABw_after is not None:
                A1_cpp = band_storage_to_dense_hermitian(ABw_after)
                w1_cpp = np.linalg.eigvalsh(A1_cpp)
                abs_diff_cpp = float(np.max(np.abs(w1_cpp - w0)))
                rel_diff_cpp = float(abs_diff_cpp / denom)
                print(f"  eig(before vs cpp_after):    max|Δ|={abs_diff_cpp:.3e}  rel={rel_diff_cpp:.3e}")

        # Compare blocks that we have dumps for.
        if B_keepR is not None:
            # B in Python is R-only now.
            print(f"  B_after_keepR: max|diff|={max_abs_diff(B_keepR, B):.3e}")
        if Sym_after is not None:
            print(f"  Sym_after:     max|diff|={max_abs_diff(Sym_after, Sym):.3e}")
        if Mid_after is not None and Mid is not None:
            print(f"  Mid_after:     max|diff|={max_abs_diff(Mid_after, Mid):.3e}")
        if Right_after is not None and Right is not None:
            print(f"  Right_after:   max|diff|={max_abs_diff(Right_after, Right):.3e}")
        if Post_after is not None and Post is not None:
            print(f"  Post_after:    max|diff|={max_abs_diff(Post_after, Post):.3e}")
        if Pre_after is not None and Pre is not None:
            print(f"  Pre_after:     max|diff|={max_abs_diff(Pre_after, Pre):.3e}")
        if ABw_after is not None:
            print(f"  ABw_band_after:max|diff|={max_abs_diff(ABw_after, ABw):.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
