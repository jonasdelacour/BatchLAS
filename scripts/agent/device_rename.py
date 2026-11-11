#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Device LA rename map (PR #250 cleanup plan, section 1): drop every v2-era spelling.

    python3 scripts/agent/device_rename.py                      # dry run, every step: per-rule counts
    python3 scripts/agent/device_rename.py --steps R1,R2 -v     # dry run of two steps, per-file counts
    python3 scripts/agent/device_rename.py --steps R1,R2 --apply

Token-level and idempotent: a second --apply changes nothing. C++ rules see only code (comments,
string and character literals are masked), except where a rule names comments. Steps:
  R1  `namespace batchlas::device::inline v2[::sub] {` -> `namespace batchlas::device[::sub] {`, and the
      closing comments `// namespace batchlas::device::inline v2...` likewise (every C++ file), and
      the placement allowlist's `batchlas::device::v2::` names.
  R2  include/: `v2::` self-qualifications -> '' (code only).
  R3  src/ tests/ examples/ benchmarks/ tools/: delete `namespace (v2|dv|dv2|dla|gc) = batchlas::device[::l3];`
      and rewrite its uses to `device::` / `device::l3::` (`batchlas::device::` in a file without
      `using namespace batchlas;` or a `namespace batchlas` scope, or with `using namespace sycl;`).
      Aliases onto other namespaces are R5's.
  R4  BATCHLAS_DLA_{INLINE,HELPER,BODY} -> BATCHLAS_DEVICE_* (every text file, scripts and fixtures too).
  R5  `<routine>_detail` -> `detail` (openers, closers, qualifiers, aliases onto them), colliding
      names first get the routine prefix (COLLIDE); batchlas::sgp / batchlas::sycl_impl ->
      batchlas::device::detail; backend `<b>_detail` -> `detail::<b>`.
  R6  `[<r>_]detail::is_{reg,cta,blocked}_v` -> `tier::is_*_v`; the per-op definitions are deleted.
  R7  file moves (MOVES), test target renames (TEST_RENAMES), the doc slug and heading renames.
  R8  tag and enum renames (TAG_RENAMES): plan section 2, C10. The table is the R-d step's to complete.
Files that are data about other code (snapshots, results, hash data) are never touched.
Exit codes: 0 done (dry run or apply), 1 a rule reported a site it cannot rewrite, 2 usage.
Delete together with scripts/v22_wp2_rewrite.py once #244 has rebased (plan decision 19).
"""
import argparse
import collections
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELF = "scripts/agent/device_rename.py"
SKIP_PREFIXES = ("tests/sass_snapshots/", "benchmarks/results/", "tests/data/", "tuned/")
SKIP_FILES = {SELF, "scripts/v22_wp2_rewrite.py"}
CXX_EXT = (".hh", ".h", ".hpp", ".cc", ".cpp", ".cxx", ".in", ".inc")
TEXT_EXT = CXX_EXT + (".py", ".md", ".dox", ".txt", ".cmake", ".sh", ".json", ".yml", ".yaml")
ALIAS_SCOPES = ("src/", "tests/", "examples/", "benchmarks/", "tools/")

# ---------------------------------------------------------------------------------------------
# Masking: C++ text split into code and non-code (comment / literal) pieces.

LEX = re.compile(r'//[^\n]*|/\*.*?\*/|R"(?P<d>[^(\s]{0,16})\(.*?\)(?P=d)"|"(?:\\.|[^"\\\n])*"'
                 r"|(?<![\w'])'(?:\\.|[^'\\\n])+'", re.S)


def pieces(text):
    """[(kind, str)] with kind 'code', 'comment' or 'literal'; joining the strs gives text back."""
    out, pos = [], 0
    for m in LEX.finditer(text):
        if m.start() > pos:
            out.append(("code", text[pos:m.start()]))
        out.append(("comment" if m.group().startswith("/") else "literal", m.group()))
        pos = m.end()
    if pos < len(text):
        out.append(("code", text[pos:]))
    return out


def sub_kinds(text, rx, rep, kinds=("code",)):
    """rx.sub(rep) on the pieces of the given kinds; returns (text, count)."""
    total, out = 0, []
    for kind, s in pieces(text):
        if kind in kinds:
            s, n = rx.subn(rep, s)
            total += n
        out.append(s)
    return "".join(out), total


def code_only(text):
    return "".join(s if k == "code" else re.sub(r"[^\n]", " ", s) for k, s in pieces(text))


# ---------------------------------------------------------------------------------------------
# Rules. Each takes (rel, text) and returns (new text, count, [problems]).

R1_OPEN = re.compile(r"(\bnamespace\s+(?:::)?batchlas::device)::inline\s+v2\b(?=\s*(?:::\s*\w+\s*)*\{)")
R1_CLOSE = re.compile(r"(//\s*namespace\s+(?:::)?batchlas::device)::(?:inline\s+)?v2\b")


def r1(rel, text):
    if rel == ALLOWLIST:   # its names are the namespaces the guard reports
        text, n = re.subn(r"\bbatchlas::device::v2::", "batchlas::device::", text)
        return text, n, []
    if not rel.endswith(CXX_EXT):
        return text, 0, []
    text, a = sub_kinds(text, R1_OPEN, r"\1")
    text, b = sub_kinds(text, R1_CLOSE, r"\1", kinds=("comment",))
    left = [f"{rel}: `inline v2` left" for _ in re.finditer(r"\binline\s+(?:namespace\s+)?v2\b", code_only(text))]
    return text, a + b, left


R2_QUAL = re.compile(r"(?<![\w:])v2::")


def r2(rel, text):
    if not (rel.startswith("include/") and rel.endswith(CXX_EXT)):
        return text, 0, []
    text, n = sub_kinds(text, R2_QUAL, "")
    bad = [f"{rel}: `namespace v2` left" for _ in re.finditer(r"\bnamespace\s+v2\b", code_only(text))]
    return text, n, bad


R3_ALIAS = re.compile(r"^[ \t]*namespace\s+(v2|dv|dv2|dla|gc)\s*=\s*(?:::)?(?:batchlas::)?device(::l3)?\s*;[^\n]*\n",
                      re.M)


def r3(rel, text):
    if not (rel.startswith(ALIAS_SCOPES) and rel.endswith(CXX_EXT)):
        return text, 0, []
    code = code_only(text)
    targets = {}
    for m in R3_ALIAS.finditer(code):
        tgt = "device::l3" if m.group(2) else "device"
        if targets.setdefault(m.group(1), tgt) != tgt:
            return text, 0, [f"{rel}: alias {m.group(1)} names two targets"]
    if not targets:
        return text, 0, []
    sycl = re.search(r"\busing\s+namespace\s+(?:::)?sycl\s*;", code)
    inside = re.search(r"\busing\s+namespace\s+(?:::)?batchlas\s*;|\bnamespace\s+batchlas\b(?!\s*=)", code)
    prefix = "" if inside and not sycl else "batchlas::"
    n_lines = 0
    spans = [m.span() for m in R3_ALIAS.finditer(code)]
    for a, b in reversed(spans):
        text = text[:a] + text[b:]
        n_lines += 1
    total = n_lines
    for alias, tgt in targets.items():
        text, n = sub_kinds(text, re.compile(rf"(?<![\w:]){alias}::"), f"{prefix}{tgt}::")
        total += n
    return text, total, []


R4_MACRO = re.compile(r"\bBATCHLAS_DLA_(INLINE|HELPER|BODY)\b")


def r4(rel, text):
    if not rel.endswith(TEXT_EXT):
        return text, 0, []
    text, n = R4_MACRO.subn(r"BATCHLAS_DEVICE_\1", text)
    return text, n, []


# R5 tables. COLLIDE: names defined in more than one <routine>_detail (l1_collide.txt). A type
# (CamelCase) gets a CamelCase routine prefix, any other name a snake_case one, so CtaPlan and
# cta_plan of one routine stay distinct. The R-b step reviews this table before applying.
COLLIDE = ("Layout", "layout", "cta_plan", "CtaPlan", "BlockedPlan", "blocked_plan", "Params", "BlockedCfg",
           "cfg_of")
BACKENDS = ("nvptx", "amdgcn", "spirv", "sscp")
R5_KEEP = ("detail",)   # `detail::` itself is the target


def collide_name(routine, name):
    if name[0].isupper():
        return "".join(p.capitalize() for p in routine.split("_")) + name
    return f"{routine}_{name}"


R5_OPEN = re.compile(r"\bnamespace\s+((?:(?:::)?batchlas::)?(?:device::)?)(\w+)_detail\b")
R5_QUAL = re.compile(r"(?<![\w])(\w+)_detail::")
R5_SGP = re.compile(r"(?<![\w:])(?:::)?batchlas::(?:sgp|sycl_impl)\b")


def r5(rel, text):
    if not rel.endswith(CXX_EXT):
        return text, 0, []
    total, problems = 0, []
    opened = sorted({m.group(2) for m in R5_OPEN.finditer(code_only(text))} - set(R5_KEEP) - set(BACKENDS))
    local = [r for r in opened if r not in BACKENDS]
    for name in COLLIDE:
        owners = [r for r in local if re.search(rf"\b{name}\b", code_only(text))]
        if len(owners) > 1:
            problems.append(f"{rel}: {name} is used under {', '.join(owners)}; rename by hand")
            continue
        if owners:
            text, n = sub_kinds(text, re.compile(rf"(?<![\w:]){name}\b"), collide_name(owners[0], name))
            total += n
    for name in COLLIDE:
        text, n = sub_kinds(text, re.compile(rf"\b(\w+)_detail::{name}\b"),
                            lambda m, nm=name: f"{m.group(1)}_detail::{collide_name(m.group(1), nm)}")
        total += n

    def opener(m):
        r = m.group(2)
        if r in R5_KEEP:
            return m.group(0)
        return f"namespace {m.group(1)}detail::{r}" if r in BACKENDS else f"namespace {m.group(1)}detail"
    text, n = sub_kinds(text, R5_OPEN, opener)
    total += n
    text, n = sub_kinds(text, R5_OPEN, opener, kinds=("comment",))
    total += n

    def qual(m):
        r = m.group(1)
        if r in R5_KEEP:
            return m.group(0)
        return f"detail::{r}::" if r in BACKENDS else "detail::"
    text, n = sub_kinds(text, R5_QUAL, qual)
    total += n
    text, n = sub_kinds(text, R5_SGP, "batchlas::device::detail")
    total += n
    return text, total, problems


R6_QUAL = re.compile(r"(?<![\w:])(?:(?:\w+_)?detail::)?is_(reg|cta|blocked)_v\b")
R6_DEF = re.compile(r"^[ \t]*(?:template\s*<[^;{]*?>\s*)?(?:inline\s+)?constexpr\s+bool\s+is_(?:reg|cta|blocked)_v\b"
                    r"[^;]*;[ \t]*(?://[^\n]*)?\n", re.M)


def r6(rel, text):
    if not rel.endswith(CXX_EXT) or rel == "include/batchlas/device/tier.hh":
        return text, 0, []
    code = code_only(text)
    spans = [m.span() for m in R6_DEF.finditer(code)]
    for a, b in reversed(spans):
        text = text[:a] + text[b:]
    text, n = sub_kinds(text, R6_QUAL, r"tier::is_\1_v")
    return text, n + len(spans), []


# R7 tables (plan section 1 "Headers and files", "Tests", "Docs", "sass_lab").
MOVES = (
    ("docs/design/device-la-v2.md", "docs/design/device-la.md"),
    ("benchmarks/sass_lab/ops_v2a.cc", "benchmarks/sass_lab/ops_device_a.cc"),
    ("benchmarks/sass_lab/ops_v2b.cc", "benchmarks/sass_lab/ops_device_b.cc"),
    ("benchmarks/sass_lab/v2lab.hh", "benchmarks/sass_lab/device_lab.hh"),
    ("benchmarks/sass_lab/v2_parity.py", "benchmarks/sass_lab/device_parity.py"),
    ("benchmarks/sass_lab/v2_timing.py", "benchmarks/sass_lab/device_timing.py"),
)
TEST_RENAMES = (
    (r"\bdevice_blas_v2_gemm-components_tests\b", "device_core_level3_components_tests"),
    (r"\bdevice_blas_v2_elementwise-glue_tests\b", "device_core_lapack_aux_tests"),
    (r"\bdevice_blas_v2_(gemv-native|trsm|syev-cta-trio|krylov-ortho-ritz)_tests\b", r"\1_kernels_tests"),
    (r"\bdevice_blas_v2_tests\b", "device_core_ops_tests"),
    (r"\bdevice_api_(\w[\w-]*)_tests\b", r"device_lapack_\1_tests"),
)
TEXT_RENAMES = (
    (r"\bdesign_device_la_v2\b", "design_device_la"),
    (r"\bDevice LA v2: ", "Device LA: "),
)


def r7(rel, text):
    if not rel.endswith(TEXT_EXT):
        return text, 0, []
    total = 0
    for old, new in MOVES:
        for a, b in ((old, new), (os.path.basename(old), os.path.basename(new))):
            text, n = re.subn(rf"(?<![\w.-]){re.escape(a)}(?![\w-])", b, text)
            total += n
    for rx, rep in TEST_RENAMES + TEXT_RENAMES:
        if rel == "docs/perf/sytrd.md" and rx == TEXT_RENAMES[1][0]:
            continue   # the evidence heading records a historical measurement (plan section 1)
        text, n = re.subn(rx, rep, text)
        total += n
    return text, total, []


# R8: plan section 2 C10 (decision D2). Code-only, `<old>` -> `<new>`; the R-d step completes it
# and runs the collision check (a new name already defined in its target namespace) first.
TAG_RENAMES = (
    (r"\bconv::Phase\b", "alg::Phase"),
    (r"\brot_conv::Transposed\b", "alg::RotTransposed"),
    (r"\bbcast_policy::Slm\b", "alg::BcastSlm"),
    (r"\btie::Lowest\b", "alg::TieLowest"),
    (r"\btie::Highest\b", "alg::TieHighest"),
)


def r8(rel, text):
    if not rel.endswith(CXX_EXT):
        return text, 0, []
    total = 0
    for rx, rep in TAG_RENAMES:
        text, n = sub_kinds(text, re.compile(rx), rep)
        total += n
    return text, total, []


RULES = collections.OrderedDict((("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4), ("R5", r5), ("R6", r6),
                                 ("R7", r7), ("R8", r8)))
# Rules that also rewrite the guard scripts and fixtures (data about other code otherwise).
TOUCH_TOOLS = {"R4", "R7"}
ALLOWLIST = "scripts/device_placement_allowlist.txt"
TOOL_FILES = {"scripts/check_device_placement.py", "scripts/check_kernel_entry_calls.py", ALLOWLIST,
              "scripts/sass_parity_wg.txt", "scripts/sass_parity_smem.txt"}


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def files():
    for f in git("ls-files").split("\n"):
        if f and f not in SKIP_FILES and not f.startswith(SKIP_PREFIXES) and f.endswith(TEXT_EXT):
            if os.path.isfile(os.path.join(REPO, f)):
                yield f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(RULES), help="comma list of R1..R8 (default: all)")
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--paths", nargs="*", help="limit to these repo paths or prefixes")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-file counts")
    a = ap.parse_args()
    steps = [s.strip().upper() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in RULES]
    if bad:
        print(f"device_rename: unknown step(s) {', '.join(bad)}", file=sys.stderr)
        return 2
    steps = [s for s in RULES if s in steps]   # always in map order
    if a.apply and "R7" in steps:
        for old, new in MOVES:
            if os.path.exists(os.path.join(REPO, old)):
                git("mv", old, new)
                print(f"move {old} -> {new}")
    counts, per_file, problems, changed = collections.Counter(), collections.defaultdict(dict), [], []
    for rel in files():
        if a.paths and not any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in a.paths):
            continue
        path = os.path.join(REPO, rel)
        with open(path, encoding="utf-8", errors="surrogateescape") as fh:
            text = fh.read()
        new = text
        for s in steps:
            if rel in TOOL_FILES and s not in TOUCH_TOOLS and not (rel == ALLOWLIST and s == "R1"):
                continue
            new, n, probs = RULES[s](rel, new)
            problems += probs
            if n:
                counts[s] += n
                per_file[rel][s] = n
        if new != text:
            changed.append(rel)
            if a.apply:
                with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
                    fh.write(new)
    if a.verbose:
        for rel in changed:
            print(f"  {rel}: " + ", ".join(f"{s} {n}" for s, n in per_file[rel].items()))
    for s in steps:
        print(f"{s}: {counts[s]} rewrites in {sum(1 for r in per_file if s in per_file[r])} files")
    for p in problems:
        print(f"  check by hand: {p}")
    print(f"device_rename: {'applied' if a.apply else 'dry run'}, {len(changed)} files "
          f"{'rewritten' if a.apply else 'would change'}, {len(problems)} by-hand sites")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
