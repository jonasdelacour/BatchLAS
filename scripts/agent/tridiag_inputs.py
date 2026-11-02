#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write accuracy::share_inputs chunk files for the steqr/stedc hash grid (tridiag_hashes.sh).

  tridiag_inputs.py DIR N SAMPLES BATCH tridiag|dense float|double [SEED]

DIR/<first sample>.bin holds one chunk of BATCH items (column-major n x n, native dtype), read
back by `steqr_accuracy --inputs DIR` / `eigensolver_accuracy --inputs DIR` in place of the
generated matrix. The generator's NETLIB arm returns a diagonal T (src/util/random_cond.cc), so
without these files every NETLIB cell is a no-op solve and a hash grid cannot see a broken
kernel. Item k depends only on (SEED, k), so batch 1 and batch N see identical items.
Item classes (k mod 8), after the WP2 harness: uniform, graded up (QL), graded down (QR),
splits, 1e+30, 1e-30, Wilkinson W+, clustered. dense: unit, 1e+6, 1e-6, graded.
"""
import os
import sys

import numpy as np


def tridiag(rng, n, k):
    cls = k % 8
    sign = lambda m: rng.choice((-1.0, 1.0), m)
    if cls == 0:
        d, e = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n - 1)
    elif cls in (1, 2):
        t = np.arange(n) / max(1, n - 1)
        d = 10.0 ** ((-6 + 12 * t) if cls == 1 else (6 - 12 * t)) * sign(n)
        e = np.sqrt(np.abs(d[:-1] * d[1:])) * rng.uniform(0.1, 1, n - 1)
    elif cls == 3:
        d = rng.uniform(-1, 1, n) * np.where(np.arange(n) % 5 == 0, 100.0, 1.0)
        u = rng.random(n - 1)
        e = np.where(u < 0.2, 0.0, np.where(u < 0.3, 1e-9, rng.uniform(-1, 1, n - 1)))
    elif cls in (4, 5):
        s = 1e30 if cls == 4 else 1e-30
        d, e = rng.uniform(-1, 1, n) * s, rng.uniform(-1, 1, n - 1) * s
    elif cls == 6:
        d, e = np.abs(np.arange(n) - (n - 1) / 2.0), np.ones(n - 1)
    else:
        d, e = 1.0 + 1e-6 * rng.uniform(-1, 1, n), 1e-3 * rng.uniform(-1, 1, n - 1)
    return np.diag(d) + np.diag(e, 1) + np.diag(e, -1)


def dense(rng, n, k):
    a = rng.uniform(-1, 1, (n, n)) * (1.0, 1e6, 1e-6, 1.0)[k % 4]
    if k % 4 == 3:
        i = np.arange(n)
        a *= 10.0 ** (-3.0 * (i[:, None] + i[None, :]) / n)
    return np.tril(a) + np.tril(a, -1).T


def main():
    out, n, samples, batch, kind, dtype = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5], sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261101
    gen = {"tridiag": tridiag, "dense": dense}[kind]
    np_t = {"float": np.float32, "double": np.float64}[dtype]
    os.makedirs(out, exist_ok=True)
    for first in range(0, samples, batch):
        items = [gen(np.random.default_rng([seed, k]), n, k) for k in range(first, min(first + batch, samples))]
        np.stack(items).astype(np_t).transpose(0, 2, 1).tofile(os.path.join(out, f"{first}.bin"))


if __name__ == "__main__":
    main()
