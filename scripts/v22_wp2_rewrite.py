#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""v2.2 WP2 (design C.1): move device LA v2 to the public tree and spell it batchlas::device.

    python3 scripts/v22_wp2_rewrite.py [--dry-run]

One mechanical pass, run once on a clean tree (it refuses a dirty one):
  1. git mv include/batchlas/blas/device/v2/*.hh -> include/batchlas/device/ (dla.hh becomes the
     umbrella include/batchlas/device.hh, lapack.hh becomes lapack_aux.hh), and the toolchain
     headers the core needs -> include/batchlas/device/detail/: src/kernels/common/sg_partition/,
     src/sycl/kernel_attrs.hh, src/sycl/sscp_target.hh (a public header never includes src/).
  2. Paths: `#include <old>` -> `#include <new>`; repo paths (docs, spec lists, scripts) likewise.
  3. Namespaces, outside string literals: every opener of batchlas::device::v2[::sub] becomes
     `namespace batchlas::device::inline v2[::sub]` (a plain reopen after an inline declaration
     warns); every alias `namespace X = ...device::v2;` becomes `namespace X = ...device;` (the
     name v2 is kept: `dla` is a family namespace, batchlas::potrf_native::dla, and `dev` a common
     variable name), so `v2::` qualifiers keep compiling unchanged; any other `device::v2`
     spelling becomes `device`.
The inline namespace mangles exactly like batchlas::device::v2, so no symbol and no kernel name
changes (gate: scripts/sass_parity.py --identical on every library).
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_V2 = "include/batchlas/blas/device/v2"
NEW_DEV = "include/batchlas/device"
SELF = "scripts/v22_wp2_rewrite.py"

# Files whose text is data about other code (fixtures, demangled names, regenerated lists).
SKIP_FILES = {
    SELF,
    "scripts/check_device_placement.py",
    "scripts/check_kernel_entry_calls.py",
    "scripts/device_placement_allowlist.txt",
    "scripts/sass_parity_wg.txt",
    "scripts/sass_parity_smem.txt",
}
SKIP_PREFIXES = ("tests/sass_snapshots/", "benchmarks/results/", "tests/data/", ".github/ci/fixtures/")
TEXT_EXT = (".hh", ".h", ".hpp", ".cc", ".cpp", ".cxx", ".py", ".md", ".txt", ".cmake", ".sh", ".in", ".json", ".yml")
CXX_EXT = (".hh", ".h", ".hpp", ".cc", ".cpp", ".cxx", ".in")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def moves():
    """(old repo path, new repo path) for every moved file."""
    out = []
    for f in sorted(git("ls-files", OLD_V2).split()):
        base = os.path.basename(f)
        new = {"dla.hh": "include/batchlas/device.hh", "lapack.hh": f"{NEW_DEV}/lapack_aux.hh"}.get(base, f"{NEW_DEV}/{base}")
        out.append((f, new))
    for f in sorted(git("ls-files", "src/kernels/common/sg_partition").split()):
        out.append((f, f"{NEW_DEV}/detail/sg_partition/{os.path.basename(f)}"))
    for base in ("kernel_attrs.hh", "sscp_target.hh"):
        out.append((f"src/sycl/{base}", f"{NEW_DEV}/detail/{base}"))
    return out


def inc(p):
    """Include spelling of a repo path (include/ is on the include path, src/ is spelled src/)."""
    return p[len("include/"):] if p.startswith("include/") else p


def path_rules(mv):
    rules = []
    # Longest first, so a file rule wins over its directory rule.
    for old, new in sorted(mv, key=lambda m: -len(m[0])):
        if inc(old) != old:   # include/... : the include spelling has no prefix
            rules.append((re.compile(r"(?<![\w/.])" + re.escape(old)), new))
            rules.append((re.compile(r"(?<![\w/.])" + re.escape(inc(old))), inc(new)))
        else:                 # src/... : `<src/x>` and `"src/x"` in an #include are include spellings
            rules.append((re.compile(r"(?<=[<])" + re.escape(old)), inc(new)))
            rules.append((re.compile(r"(?<=#include \")" + re.escape(old)), inc(new)))
            rules.append((re.compile(r"(?<![\w/.<])" + re.escape(old)), new))
    for old_dir, new_dir in (("src/kernels/common/sg_partition", f"{NEW_DEV}/detail/sg_partition"),
                             (OLD_V2, NEW_DEV), ("include/batchlas/blas/device", NEW_DEV),
                             (inc(OLD_V2), inc(NEW_DEV))):
        rules.append((re.compile(r"(?<![\w/.<])" + re.escape(old_dir) + r"(?![\w.])"), new_dir))
        rules.append((re.compile(r"(?<=<)" + re.escape(old_dir) + r"(?![\w.])"), inc(new_dir)))
    return rules


STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
OPENER = re.compile(r"^(\s*namespace\s+(?:::)?batchlas::device::)v2\b(?!\s*=)")
CLOSE_COMMENT = re.compile(r"(//\s*namespace\s+(?:::)?batchlas::device)::v2\b")
ALIAS_V2 = re.compile(r"\bnamespace\s+v2\s*=\s*((?:::)?(?:batchlas::)?device)::v2\s*;")
ALIAS_ANY = re.compile(r"(\bnamespace\s+\w+\s*=\s*(?:::)?(?:batchlas::)?device)::v2\b")
DEVICE_V2 = re.compile(r"\bdevice::v2(::|\b)")


def outside_strings(line, fn):
    out, pos = [], 0
    for m in STRING.finditer(line):
        out.append(fn(line[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(line[pos:]))
    return "".join(out)


def rewrite_cxx(text):
    lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):   # preprocessor lines: paths only
            lines.append(line)
            continue

        def fix(s):
            s = OPENER.sub(r"\1inline v2", s)
            s = CLOSE_COMMENT.sub(r"\1", s)
            s = ALIAS_V2.sub(r"namespace v2 = \1;", s)
            s = ALIAS_ANY.sub(r"\1", s)
            s = DEVICE_V2.sub(lambda m: "device" + (m.group(1) if m.group(1) == "::" else ""), s)
            return s

        lines.append(outside_strings(line, fix))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the moves and changed files only")
    a = ap.parse_args()
    if git("status", "--porcelain", "--untracked-files=no").strip() and not a.dry_run:
        sys.exit("v22_wp2_rewrite: the tree is dirty; commit or stash first")
    mv = moves()
    rules = path_rules(mv)
    for old, new in mv:
        print(f"move {old} -> {new}")
        if not a.dry_run:
            os.makedirs(os.path.join(REPO, os.path.dirname(new)), exist_ok=True)
            git("mv", old, new)
    changed, leftover = [], []
    for f in git("ls-files").split("\n"):
        if not f or f in SKIP_FILES or f.startswith(SKIP_PREFIXES) or not f.endswith(TEXT_EXT):
            continue
        path = os.path.join(REPO, f)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="surrogateescape") as fh:
            text = fh.read()
        new = text
        for rx, rep in rules:
            new = rx.sub(rep, new)
        if f.endswith(".md"):
            new = DEVICE_V2.sub(lambda m: "device" + (m.group(1) if m.group(1) == "::" else ""), new)
        if f.endswith(CXX_EXT):
            new = rewrite_cxx(new)
        if new != text:
            changed.append(f)
            if not a.dry_run:
                with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
                    fh.write(new)
    print(f"v22_wp2_rewrite: {len(mv)} moves, {len(changed)} files rewritten")
    for f in leftover:
        print(f"  check by hand: v2:: left in {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
