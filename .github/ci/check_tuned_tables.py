#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warn about stale tuned selection tables (docs/design/tuning-data.md#tuning-data-kernel-hash).

    python3 .github/ci/check_tuned_tables.py              # warnings only, always exits 0
    python3 .github/ci/check_tuned_tables.py --self-test  # the shared hash vector; exits 1 on failure

One staleness rule, scripts/sweep_to_table.py stale_kernel_warnings: a table's header hashes
(kernels=, family_kernels=) against hash v2 of tools/tune/<op>_spec.cc's kernel-sources and
kernel-deps now. A table whose tiers= counts only converted/prior/transcribed rows carries no hash
and is never stale. A stale table stays in use: staleness is never an error. Reads only tuned/ and the
kernel sources (no ledger, so no LFS). sweep_to_table.py --check prints the same warnings.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import sweep_to_table  # noqa: E402


def check(repo):
    tuned = os.path.join(repo, "tuned")
    stale = 0
    for name in sorted(f for f in os.listdir(tuned) if f.endswith(".txt")):
        warnings = sweep_to_table.stale_kernel_warnings(os.path.join(tuned, name), repo)
        stale += any(w.startswith("warning:") for w in warnings)
        for w in warnings:
            print(w)
    print(f"check_tuned_tables: {stale} stale table(s) (warning only)")
    return 0


def self_test(repo):
    problem = sweep_to_table.hash_vector_problem(repo)
    if problem:
        print("self-test FAILED:", problem)
    print(f"check_tuned_tables --self-test: {'FAILED' if problem else 'OK'}")
    return 1 if problem else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    return self_test(args.repo) if args.self_test else check(args.repo)


if __name__ == "__main__":
    sys.exit(main())
