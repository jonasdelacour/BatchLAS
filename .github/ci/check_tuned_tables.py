#!/usr/bin/env python3
"""Warn about stale tuned selection tables (docs/design/tuning-data.md, "The table file").

    python3 .github/ci/check_tuned_tables.py              # warnings only, always exits 0
    python3 .github/ci/check_tuned_tables.py --self-test  # the hash and parser; exits 1 on failure

Each tuned/<op>.<device>.txt header carries kernels=<hash> (the op's whole kernel-sources block)
and family_kernels=<family>:<hash>,... The op-level hash is recomputed from tools/tune/<op>_spec.cc:
the quoted paths between the lines holding "kernel-sources-begin" and "kernel-sources-end",
kernel hash v2 = sha256 over the lines "<sha256(normalized file)>  <path>\\n" in list order, first 8
hex digits, where normalized = /* */ blocks removed (an unterminated one runs to the end), then
// to end of line, then every whitespace byte. batchlas_tune and scripts/sweep_to_table.py compute
the same value (shared vector: tests/data/kernel_hash_v2). A table whose hash differs gets a
warning and stays in use: staleness is never an error. scripts/sweep_to_table.py --check names
the stale families too. An op without a spec file is reported once and skipped.
"""

import argparse
import hashlib
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_kernel_list(text):
    out, inside = [], False
    for line in text.splitlines():
        if "kernel-sources-begin" in line:
            inside = True
            continue
        if "kernel-sources-end" in line:
            break
        if inside:
            out += re.findall(r'"([^"]+)"', line)
    return out


def normalize(data):
    data = re.sub(rb"/\*.*?(?:\*/|\Z)", b"", data, flags=re.S)
    data = re.sub(rb"//[^\n]*", b"", data)
    return re.sub(rb"[ \t\n\r\f\v]+", b"", data)


def kernel_hash(repo, paths):
    """The 8-hex v2 hash, or None plus the first missing path."""
    manifest = ""
    for p in paths:
        try:
            with open(os.path.join(repo, p), "rb") as f:
                manifest += hashlib.sha256(normalize(f.read())).hexdigest() + "  " + p + "\n"
        except OSError:
            return None, p
    return hashlib.sha256(manifest.encode()).hexdigest()[:8], None


def header_kernels(text):
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        m = re.search(r"\bkernels=(\S+)", line)
        if m:
            return m.group(1)
    return "unknown"


def check(repo):
    tuned = os.path.join(repo, "tuned")
    hashes, stale = {}, 0
    for name in sorted(f for f in os.listdir(tuned) if f.endswith(".txt")):
        op = name.split(".")[0]
        if op not in hashes:
            spec = os.path.join(repo, "tools", "tune", f"{op}_spec.cc")
            if not os.path.exists(spec):
                print(f"note: no tools/tune/{op}_spec.cc, {op} tables not checked")
                hashes[op] = None
                continue
            with open(spec) as f:
                h, missing = kernel_hash(repo, parse_kernel_list(f.read()))
            hashes[op] = h if h else f"missing:{missing}"
        want = hashes[op]
        if want is None:
            continue
        with open(os.path.join(tuned, name)) as f:
            have = header_kernels(f.read())
        if have != want:
            stale += 1
            print(f"warning: tuned/{name} is stale (kernels {have} -> {want}); retune the op "
                  "(scripts/sweep_to_table.py --check names the families)")
    print(f"check_tuned_tables: {stale} stale table(s) (warning only)")
    return 0


def self_test(repo):
    bad = []
    if hashlib.sha256(b"abc").hexdigest()[:8] != "ba7816bf":
        bad.append("sha256('abc')")
    spec = 'x "not/this"\n// kernel-sources-begin\n  "a.cc", "b.hh",\n};\n// kernel-sources-end\n"c.cc"\n'
    if parse_kernel_list(spec) != ["a.cc", "b.hh"]:
        bad.append(f"parse_kernel_list -> {parse_kernel_list(spec)}")
    with tempfile.TemporaryDirectory() as d:
        for name, body in (("a.cc", b"one // c\n"), ("b.hh", b"t w\to")):
            with open(os.path.join(d, name), "wb") as f:
                f.write(body)
        manifest = (hashlib.sha256(b"one").hexdigest() + "  a.cc\n" +
                    hashlib.sha256(b"two").hexdigest() + "  b.hh\n")
        want = hashlib.sha256(manifest.encode()).hexdigest()[:8]
        if kernel_hash(d, ["a.cc", "b.hh"]) != (want, None):
            bad.append("kernel_hash manifest (normalized files)")
        if kernel_hash(d, ["a.cc", "zz.cc"]) != (None, "zz.cc"):
            bad.append("kernel_hash missing file")
    with open(os.path.join(repo, "tests", "data", "kernel_hash_v2", "expected.txt")) as f:
        vec, path = f.read().split()
    if kernel_hash(repo, [path]) != (vec, None):
        bad.append(f"shared vector {path}: {kernel_hash(repo, [path])} != {vec}")
    for text, want in (("# op=x kernels=1a2b3c4d date=y\n", "1a2b3c4d"), ("# op=x\n# keys: n:log\n", "unknown"),
                       ("# op=x\nuplo=L | a 1  # kernels=zz\n", "unknown")):
        if header_kernels(text) != want:
            bad.append(f"header_kernels({text!r})")
    for b in bad:
        print("self-test FAILED:", b)
    print(f"check_tuned_tables --self-test: {'FAILED' if bad else 'OK'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    return self_test(args.repo) if args.self_test else check(args.repo)


if __name__ == "__main__":
    sys.exit(main())
