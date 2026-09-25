#!/usr/bin/env python3
"""Check that every file .gitattributes routes through Git LFS is committed as an LFS pointer.

Raw benchmark results live under benchmarks/results/ and are stored in LFS, so a PR that adds a
grid costs a three-line pointer per file in its diff instead of the grid itself -- before this,
one PR carried 14,608 lines of CSV and log, 30% of its diff.

The failure this guards is silent. On a machine WITHOUT git-lfs installed, git has no `lfs`
filter driver, ignores the attribute, and commits the file's raw content with no warning at all.
Nothing downstream notices either: the file reads correctly, the data is intact, and the only
symptom is the diff bloat the attribute exists to prevent -- plus an LFS-tracked path that now
holds a non-pointer, which later confuses every clone that does have LFS.

So this reads each tracked file's BLOB (not the working tree, which LFS smudges back to real
content) and requires it to be a pointer. Which paths are LFS-tracked is asked of git itself via
`git check-attr`, never re-derived from the .gitattributes text, so the policy has one spelling.

Usage:
    python3 .github/ci/check_lfs_pointers.py [--rev REV]   # default: the index
    python3 .github/ci/check_lfs_pointers.py --self-test
`--rev` checks a commit's tree against the CURRENT policy, which is how the gate was armed: the
commit before the LFS conversion must fail. Exit code 0 = clean, 1 = findings.
"""

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The LFS pointer spec: the version line first, an oid and a size, and under 1024 bytes in
# total. The size bound matters -- a real CSV that happens to start with the version line
# is not a pointer. https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
VERSION = b"version https://git-lfs.github.com/spec/v1"
OID = re.compile(rb"^oid sha256:[0-9a-f]{64}$", re.M)
SIZE = re.compile(rb"^size [0-9]+$", re.M)
MAX_POINTER_BYTES = 1024


def is_pointer(blob):
    if len(blob) >= MAX_POINTER_BYTES:
        return False
    lines = blob.split(b"\n")
    return lines[0] == VERSION and bool(OID.search(blob)) and bool(SIZE.search(blob))


def git(*args, stdin=None):
    return subprocess.run(["git", "-C", REPO, *args], input=stdin, capture_output=True,
                          check=True).stdout


def tracked_files(rev):
    out = git("ls-files", "-z") if rev is None else git("ls-tree", "-r", "-z", "--name-only", rev)
    return [p for p in out.decode().split("\0") if p]


def lfs_paths(paths):
    if not paths:
        return []
    out = git("check-attr", "-z", "filter", "--stdin", stdin=("\0".join(paths) + "\0").encode())
    fields = out.decode().split("\0")
    # -z output is repeated (path, attribute, value) triples.
    return [fields[i] for i in range(0, len(fields) - 2, 3) if fields[i + 2] == "lfs"]


def read_blobs(rev, paths):
    spec = ":" if rev is None else rev + ":"
    out = git("cat-file", "--batch", stdin="".join(spec + p + "\n" for p in paths).encode())
    blobs, pos = {}, 0
    for p in paths:
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].split()
        if header[-1] == b"missing":
            raise SystemExit("check_lfs_pointers: git could not read %s%s" % (spec, p))
        size = int(header[2])
        blobs[p] = out[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1   # the blob is followed by a newline
    return blobs


def check(rev):
    paths = lfs_paths(tracked_files(rev))
    blobs = read_blobs(rev, paths)
    raw = [p for p in paths if not is_pointer(blobs[p])]
    where = "the index" if rev is None else rev
    for p in raw:
        print("%s: error: LFS-tracked but committed as raw content (%d bytes) in %s"
              % (p, len(blobs[p]), where))
    if raw:
        print("\nThis happens when git-lfs is not installed where the file was committed. Fix:\n"
              "    git lfs install\n"
              "    git add --renormalize %s\n"
              "and commit. The data is not lost -- only how it is stored changes."
              % " ".join(sorted({os.path.dirname(p) for p in raw})))
    print("check_lfs_pointers: %d LFS-tracked file(s) in %s, %d raw" % (len(paths), where, len(raw)))
    return 1 if raw else 0


def self_test():
    oid = b"oid sha256:" + b"a" * 64
    cases = [
        ("a valid pointer", VERSION + b"\n" + oid + b"\nsize 14295\n", True),
        ("a raw CSV", b"op,type,m,n\ngeqrf,float,4,4\n", False),
        ("an empty file", b"", False),
        ("a pointer with no size line", VERSION + b"\n" + oid + b"\n", False),
        ("a pointer with a short oid", VERSION + b"\noid sha256:abc\nsize 1\n", False),
        ("the version line NOT first", oid + b"\n" + VERSION + b"\nsize 1\n", False),
        # A real file may begin with the version line; the size bound is what refuses it.
        ("1 KiB of data behind a pointer header",
         VERSION + b"\n" + oid + b"\nsize 1\n" + b"x" * MAX_POINTER_BYTES, False),
    ]
    bad = 0
    for label, blob, want in cases:
        got = is_pointer(blob)
        print("  %s  %-40s want %-5s got %s" % ("ok  " if got == want else "FAIL", label, want, got))
        bad += got != want
    print("check_lfs_pointers self-test: %d case(s), %d failure(s)" % (len(cases), bad))
    return 1 if bad else 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    rev = argv[argv.index("--rev") + 1] if "--rev" in argv else None
    return check(rev)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
