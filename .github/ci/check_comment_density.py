#!/usr/bin/env python3
"""Enforce the comment-density ceiling on BatchLAS C++ sources.

WHY THIS EXISTS
---------------
Measured numbers, ratio tables, A/B results, saturation ladders and bracketing
cells belong in docs/perf/<op>.md, not in the source. Code comments carry
CORRECTNESS RATIONALE ONLY -- why a gate exists, why a spelling is load-bearing,
what breaks if you change it -- and point at the evidence with

    evidence: docs/perf/<page>.md#<anchor>

That rule has regressed twice under hand review alone, so it is now mechanical.
The repo's house band is 12-18% comment lines over non-blank lines. Only the
CEILING is enforced: a terse file is not a defect, so there is no floor. A file
that genuinely earns more gets one line in
.github/ci/comment_density_waivers.txt with a reason.

THE COUNTING RULE
-----------------
denominator  every line with at least one non-whitespace character.
numerator    every non-blank line whose non-whitespace content is ENTIRELY
             comment.

The second half is the load-bearing half:

    // explain                     <- comment line
    /* explain */                  <- comment line
     * continuation of a block     <- comment line
    int x = 0;  // explain         <- CODE line, NOT a comment line
    /* explain */ int x = 0;       <- CODE line
    foo();      /* opens a block   <- CODE line; the lines after it are
       still inside the block */      comment lines up to and including `*/`,
                                      unless code follows that `*/`.

Counting a trailing comment as a comment line would let a file reach 100%
without a single line of prose, which is the opposite of what is being
measured. A whitespace-only line is blank even inside a block comment: it is in
neither the numerator nor the denominator.

Comment markers inside string literals, character literals and raw strings do
not open a comment. `1'000'000` is not a character literal.

USAGE
-----
    python3 .github/ci/check_comment_density.py                 # src include tests benchmarks
    python3 .github/ci/check_comment_density.py src/extensions  # a subtree
    python3 .github/ci/check_comment_density.py --ceiling 25
    python3 .github/ci/check_comment_density.py --fix-suggest   # what to move, with line ranges
    python3 .github/ci/check_comment_density.py --all           # list every file, not just offenders
    python3 .github/ci/check_comment_density.py --self-test     # arm the counter in both directions

Exit code 0 = every file at or under the ceiling, 1 = offenders (or a broken
waiver line, or a failed self-test).
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DIRS = ("src", "include", "tests", "benchmarks")
DEFAULT_CEILING = 18.0
WAIVER_LIST_LIMIT = 12
BAND_FLOOR = 12.0          # reported for context only; never enforced
WAIVERS = os.path.join(".github", "ci", "comment_density_waivers.txt")

SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp",
            ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh")

# Directory fragments that are never ours to police. .gitignore already keeps
# build trees out -- we only ever look at files git knows about -- so this is
# only for vendored sources that are checked in.
SKIP_PARTS = ("third_party", "thirdparty", "extern", "external", "_deps",
              "__pycache__")

# `R"delim(` with any of the C++ encoding prefixes, not preceded by an
# identifier character, so that `FOOR"x"` is not read as a raw string.
RAW_OPEN = re.compile(r'(?:u8|u|U|L)?R"([^()\\ \t]{0,16})\(')
IDENT = re.compile(r'[A-Za-z0-9_]')


# --------------------------------------------------------------------------
# the counter
# --------------------------------------------------------------------------

def classify(text):
    """Return (comment_lines, non_blank_lines, comment_flags).

    comment_flags[i] is True when line i (0-based) is a comment-only line.
    """
    lines = text.splitlines()
    flags = [False] * len(lines)
    comment_lines = 0
    non_blank = 0

    state = "normal"       # normal | block | raw | line_cont | str_cont
    raw_delim = ""
    str_quote = ""

    for idx, line in enumerate(lines):
        has_code = False
        has_comment = False
        i = 0
        n = len(line)

        while i < n:
            c = line[i]

            if state == "block":
                if line.startswith("*/", i):
                    has_comment = True
                    state = "normal"
                    i += 2
                else:
                    if not c.isspace():
                        has_comment = True
                    i += 1
                continue

            if state == "raw":
                term = ')' + raw_delim + '"'
                pos = line.find(term, i)
                if pos < 0:
                    if line[i:].strip():
                        has_code = True
                    i = n
                else:
                    has_code = True
                    state = "normal"
                    i = pos + len(term)
                continue

            if state == "str_cont":
                # A string literal continued across a backslash-newline.
                has_code = True
                j = i
                closed = False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == str_quote:
                        closed = True
                        j += 1
                        break
                    j += 1
                if closed:
                    state = "normal"
                    i = j
                else:
                    i = n
                continue

            if state == "line_cont":
                # A // comment continued by a trailing backslash.
                if line.strip():
                    has_comment = True
                i = n
                continue

            # --- state == "normal" -------------------------------------
            if line.startswith("//", i):
                has_comment = True
                if line.rstrip().endswith("\\"):
                    state = "line_cont"
                i = n
                continue

            if line.startswith("/*", i):
                has_comment = True
                state = "block"
                i += 2
                continue

            if c in ('"', "'"):
                if c == "'" and i > 0 and IDENT.match(line[i - 1]):
                    has_code = True          # digit separator: 1'000'000
                    i += 1
                    continue
                has_code = True
                str_quote = c
                j = i + 1
                closed = False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == str_quote:
                        closed = True
                        j += 1
                        break
                    j += 1
                if not closed and line.rstrip().endswith("\\"):
                    state = "str_cont"
                    i = n
                else:
                    i = j
                continue

            if c in "RuUL":
                m = RAW_OPEN.match(line, i)
                if m and not (i > 0 and IDENT.match(line[i - 1])):
                    has_code = True
                    raw_delim = m.group(1)
                    state = "raw"
                    i = m.end()
                    continue

            if not c.isspace():
                has_code = True
            i += 1

        if state == "line_cont" and not line.rstrip().endswith("\\"):
            state = "normal"

        if not line.strip():
            continue                      # blank: in neither count
        non_blank += 1
        if has_comment and not has_code:
            comment_lines += 1
            flags[idx] = True

    return comment_lines, non_blank, flags


def density(comment_lines, non_blank):
    return 100.0 * comment_lines / non_blank if non_blank else 0.0


# --------------------------------------------------------------------------
# file discovery
# --------------------------------------------------------------------------

def list_files(root, dirs, tracked_only):
    """Files git knows about under `dirs`. Honours .gitignore.

    --cached picks up tracked files; --others --exclude-standard picks up new
    files that are not ignored, because a brand-new file is exactly the one
    most likely to arrive over budget.
    """
    args = ["git", "-C", root, "ls-files", "-z", "--cached"]
    if not tracked_only:
        args += ["--others", "--exclude-standard"]
    args += ["--"] + list(dirs)
    try:
        out = subprocess.run(args, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print("check_comment_density: cannot list files: %s" % exc, file=sys.stderr)
        return None
    names = [p.decode("utf-8", "replace") for p in out.split(b"\0") if p]
    keep = []
    for rel in sorted(set(names)):
        if not rel.endswith(SUFFIXES):
            continue
        if any(part in SKIP_PARTS for part in rel.split("/")):
            continue
        if os.path.isfile(os.path.join(root, rel)):
            keep.append(rel)
    return keep


# --------------------------------------------------------------------------
# waivers
# --------------------------------------------------------------------------

def load_waivers(root):
    """Parse the allowlist. Returns (entries, errors).

    One `path # reason` per line. The reason is mandatory: a waiver without one
    is indistinguishable from an oversight, which is how an allowlist turns
    into the place files go to be forgotten.
    """
    path = os.path.join(root, WAIVERS)
    entries = []
    errors = []
    if not os.path.isfile(path):
        return entries, errors
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" not in line:
                errors.append("%s:%d: error: waiver `%s` has no `# reason`"
                              % (WAIVERS, lineno, line))
                continue
            pat, reason = line.split("#", 1)
            pat, reason = pat.strip(), reason.strip()
            if not pat:
                errors.append("%s:%d: error: waiver has no path" % (WAIVERS, lineno))
            elif not reason:
                errors.append("%s:%d: error: waiver `%s` has an empty reason"
                              % (WAIVERS, lineno, pat))
            else:
                entries.append((pat, reason, lineno))
    return entries, errors


def match_waiver(rel, entries):
    for pat, reason, lineno in entries:
        if rel == pat or fnmatch.fnmatch(rel, pat):
            return (pat, reason, lineno)
    return None


# --------------------------------------------------------------------------
# --fix-suggest
# --------------------------------------------------------------------------

def comment_blocks(text, flags, top=3):
    """Longest runs of comment-only lines: [(start, end, length, first_text)]."""
    lines = text.splitlines()
    blocks = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            blocks.append((start + 1, i, i - start))
            start = None
    if start is not None:
        blocks.append((start + 1, len(flags), len(flags) - start))
    blocks.sort(key=lambda b: (-b[2], b[0]))
    return [(s, e, ln, lines[s - 1].strip()) for s, e, ln in blocks[:top]]


# --------------------------------------------------------------------------
# self-test: arm the counter in both directions
# --------------------------------------------------------------------------

SELF_TEST_CASES = [
    ("all_comment", "// one\n// two\n// three\n// four\n",
     100.0, "an all-comment file must read ~100%"),
    ("all_code", "int a = 1;\nint b = 2;\nint c = 3;\n",
     0.0, "an all-code file must read 0%"),
    ("trailing",
     "int a = 1;  // why\nint b = 2;  // why\nint c = 3;  // why\nint d = 4;  /* why */\n",
     0.0, "trailing comments on every code line must NOT read 100%"),
    ("in_band",
     # 2 comment lines / 14 non-blank = 14.29%, inside the 12-18% band
     "// a header line\n#include <x>\n\nint f() {\n    int a = 1;\n    int b = 2;\n"
     "    int c = 3;\n    // why this gate exists\n    if (a) return b;\n"
     "    return c;\n}\nint g() { return 1; }\nint h() { return 2; }\n"
     "int i() { return 3; }\nint j() { return 4; }\n",
     14.29, "a file inside the band must pass"),
    ("block", "/* one\n * two\n * three */\nint a = 1;\n",
     75.0, "a multi-line block comment counts every one of its lines"),
    ("block_then_code", "/* explain */ int a = 1;\nint b = 2;\n",
     0.0, "a block comment followed by code on the same line is a code line"),
    ("code_then_block",
     "int a = 1;  /* opens\n   still inside\n   closes */ int b = 2;\nint c = 3;\n",
     25.0, "only the fully-enclosed middle line of a trailing block counts"),
    ("string_marker",
     'const char* s = "// not a comment";\nconst char* t = "/* nor this */";\n',
     0.0, "comment markers inside string literals do not open a comment"),
    ("raw_string",
     'auto s = R"(// not a comment\n/* nor this */)";\nint a = 1;\n',
     0.0, "comment markers inside a raw string do not open a comment"),
    ("digit_sep", "int a = 1'000'000;  // why\nint b = 2'000'000;\n",
     0.0, "a digit separator is not a character literal"),
    ("blank_in_block", "/* one\n\n   two */\nint a = 1;\nint b = 2;\nint c = 3;\n",
     40.0, "a whitespace-only line inside a block comment is blank, not a comment"),
    ("empty", "", 0.0, "an empty file is 0%, not a division by zero"),
]

CEILING_CASES = [
    (18.0, False, "exactly at the ceiling passes"),
    (18.1, True, "a hair over the ceiling fails"),
    (4.0, False, "well under the ceiling passes"),
    (100.0, True, "an all-comment file fails"),
]


def self_test():
    print("== check_comment_density self-test (expected vs observed)")
    bad = 0
    for name, text, expect, why in SELF_TEST_CASES:
        got_c, got_n, _ = classify(text)
        got = density(got_c, got_n)
        ok = abs(got - expect) < 0.05
        bad += 0 if ok else 1
        print("  %-4s %-16s expected %6.2f%%  observed %6.2f%%  (%d/%d)  %s"
              % ("ok" if ok else "FAIL", name, expect, got, got_c, got_n, why))
    for d, expect_fail, why in CEILING_CASES:
        got_fail = d > DEFAULT_CEILING + 1e-9
        ok = got_fail == expect_fail
        bad += 0 if ok else 1
        print("  %-4s %-16s %6.2f%% vs ceiling %.0f%%: expected %-4s observed %-4s  %s"
              % ("ok" if ok else "FAIL", "gate", d, DEFAULT_CEILING,
                 "fail" if expect_fail else "pass",
                 "fail" if got_fail else "pass", why))
    print("check_comment_density self-test: %d case(s), %d failure(s)"
          % (len(SELF_TEST_CASES) + len(CEILING_CASES), bad))
    return 1 if bad else 0


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fail when a C++ source is over the comment-density ceiling.")
    ap.add_argument("paths", nargs="*",
                    help="directories or files to scan (default: %s)" % " ".join(DEFAULT_DIRS))
    ap.add_argument("--root", default=REPO, help="repository root")
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING,
                    help="maximum comment%% (default %.0f)" % DEFAULT_CEILING)
    ap.add_argument("--show-waivers", action="store_true",
                    help="list every waived file with its reason, however many "
                         "there are (the default collapses a long list to a count)")
    ap.add_argument("--min-lines", type=int, default=0, metavar="N",
                    help="ignore files with fewer than N non-blank lines (default 0: none)")
    ap.add_argument("--fix-suggest", action="store_true",
                    help="for each offender, print its longest comment blocks with line ranges")
    ap.add_argument("--all", action="store_true", help="list every file, not just offenders")
    ap.add_argument("--tracked-only", action="store_true",
                    help="skip new files git does not have staged yet")
    ap.add_argument("--self-test", action="store_true",
                    help="run the counter against synthetic fixtures and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    root = os.path.abspath(args.root)
    targets = args.paths if args.paths else list(DEFAULT_DIRS)

    files = list_files(root, targets, args.tracked_only)
    if files is None:
        return 1

    waivers, errors = load_waivers(root)

    rows = []
    for rel in files:
        try:
            with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            errors.append("%s: error: cannot read (%s)" % (rel, exc))
            continue
        c, nb, flags = classify(text)
        rows.append((rel, c, nb, density(c, nb), text, flags))

    ceiling = args.ceiling
    offenders = []
    waived = []
    for rel, c, nb, d, text, flags in rows:
        if nb < args.min_lines or d <= ceiling + 1e-9:
            continue
        hit = match_waiver(rel, waivers)
        if hit:
            waived.append((rel, c, nb, d, hit))
        else:
            offenders.append((rel, c, nb, d, text, flags))

    offenders.sort(key=lambda r: (-r[3], r[0]))
    waived.sort(key=lambda r: -r[3])

    # A waiver that matches nothing is a typo or a stale path, and a stale
    # allowlist entry silently widens the gate for whatever moves into that
    # name next.
    #
    # Resolve it against the WHOLE repo, not against `targets`: a subtree scan
    # (`... check_comment_density.py src/extensions`) legitimately sees none of
    # the waived files, and reporting every waiver as stale there would make
    # every narrow run exit 1 for no reason.
    if waivers:
        universe = files if list(targets) == list(DEFAULT_DIRS) else \
            (list_files(root, DEFAULT_DIRS, args.tracked_only) or [])
        for pat, reason, lineno in waivers:
            if not any(rel == pat or fnmatch.fnmatch(rel, pat) for rel in universe):
                errors.append("%s:%d: error: waiver `%s` matches no file in %s"
                              % (WAIVERS, lineno, pat, " ".join(DEFAULT_DIRS)))

    if args.all:
        print("== every scanned file, densest first")
        for rel, c, nb, d, _t, _f in sorted(rows, key=lambda r: (-r[3], r[0])):
            mark = " " if d <= ceiling else ("W" if match_waiver(rel, waivers) else "!")
            print("  %s %6.2f%%  %4d/%-5d %s" % (mark, d, c, nb, rel))
        print()

    if waived:
        # A waiver has to keep justifying itself out loud -- but 138 of them printing two
        # lines each is 276 lines on every local run, and a wrapper nobody can read the
        # output of is a wrapper nobody runs. Above WAIVER_LIST_LIMIT the block collapses
        # to a count and the worst few; --show-waivers always prints all of them.
        show_all = args.show_waivers or len(waived) <= WAIVER_LIST_LIMIT
        shown = waived if show_all else waived[:WAIVER_LIST_LIMIT]
        print("== waived (allowlisted, still over the %.0f%% ceiling): %d file(s)"
              % (ceiling, len(waived)))
        for rel, c, nb, d, (pat, reason, lineno) in shown:
            print("  %6.2f%%  %4d/%-5d %s" % (d, c, nb, rel))
            print("            %s:%d -- %s" % (WAIVERS, lineno, reason))
        if not show_all:
            print("  ... and %d more, worst first; --show-waivers lists every one with "
                  "its reason." % (len(waived) - len(shown)))
        print()

    if offenders:
        print("== over the comment-density ceiling (%.0f%%), worst first" % ceiling)
        print("   house band is %.0f-%.0f%%; only the ceiling is enforced, a terse "
              "file is not a defect." % (BAND_FLOOR, ceiling))
        print()
        for rel, c, nb, d, text, flags in offenders:
            over = max(int(c - (ceiling / 100.0) * nb) + 1, 1)
            print("%s:1: error: %.2f%% comment lines (%d/%d non-blank), ceiling %.0f%% "
                  "-- move ~%d line(s) of evidence to docs/perf/<op>.md and leave an "
                  "`evidence: docs/perf/<page>.md#<anchor>` pointer"
                  % (rel, d, c, nb, ceiling, over))
            if args.fix_suggest:
                for s, e, ln, first in comment_blocks(text, flags):
                    print("        longest block: lines %d-%d (%d lines): %s"
                          % (s, e, ln, first[:76]))
        print()

    for err in errors:
        print(err)

    total_nb = sum(r[2] for r in rows)
    total_c = sum(r[1] for r in rows)
    print("check_comment_density: %d file(s), repo density %.2f%% (%d/%d), "
          "%d over ceiling, %d waived"
          % (len(rows), density(total_c, total_nb), total_c, total_nb,
             len(offenders), len(waived)))
    if offenders and not args.fix_suggest:
        print("  re-run with --fix-suggest to see which blocks to move.")
    return 1 if (offenders or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
