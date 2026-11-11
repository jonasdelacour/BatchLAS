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
  R3  include/ src/ tests/ examples/ benchmarks/ tools/: delete every `namespace A = T;` whose T is in the
      device LA layer (batchlas::device[::sub], a sibling <routine>_detail, a src device namespace such
      as syev_v2, tiny_native, sycl_device, sycl_trsm::dla) and rewrite `A::` inside the alias's scope to
      the qualified T (`device::...`; `batchlas::device::...` in a file without `using namespace
      batchlas;` or a `namespace batchlas` scope, or with `using namespace sycl;`; relative in include/).
      Host aliases (ops::<op>, select, test_support, std::filesystem, GEMM wide_v2) stay.
  R4  BATCHLAS_DLA_{INLINE,HELPER,BODY} -> BATCHLAS_DEVICE_* (every text file, scripts and fixtures too).
  R5  the 40 `<routine>_detail` (ROUTINES) -> `detail` (openers, closers, qualifiers, mentions);
      first the LOCAL_RENAMES (colliding names get the routine prefix, tier bodies become
      <routine>_<tier>_run) and the PROMOTED LAPACK names lose the qualifier. The sg_partition
      backends' <b>_detail, batchlas::sgp and batchlas::sycl_impl are not merged yet.
  R6  `[<r>_]detail::is_{reg,cta,blocked}_v`, `ormqr_is_blocked_v` -> `tier::is_*_v`; the per-op
      definitions are deleted (tier.hh holds the one copy; sygst's bounded ones were renamed by R5).
  R7  file moves (MOVES), test target renames (TEST_RENAMES), the doc slug and heading renames.
  R8  tag and enum renames (TAG_RENAMES, MODE_COMPARE): plan section 2, C10 (WP R-d); the enum
      definitions and the parameters that change kind (class -> enum) were edited by hand.
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


R3_ALIAS = re.compile(r"^[ \t]*namespace\s+(\w+)\s*=\s*(?:::)?([\w:]+?)\s*;[^\n]*\n", re.M)
# Alias targets in the device LA layer: batchlas::device[::sub], a sibling <routine>_detail (include/),
# and the src namespaces that hold device code (plan section 1 "Banned"). Host aliases stay.
R3_DEVICE = re.compile(r"(?:batchlas::)?device(?:::\w+)*|\w+_detail")
R3_SRC = re.compile(r"(?:batchlas::)?(?:syev_v2|stedc_v2|krylov_v2|tiny_native|geqrf_native|getrf_native|"
                    r"potrf_native|sycl_device|sycl_potri|sycl_trsm::dla|sycl_gemv::dla)")


def block_end(code, pos):
    """End of the block scope that encloses pos; len(code) at namespace or file scope (a namespace
    alias in an unnamed or reopened namespace stays visible after its closing brace)."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        c = code[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                break
            depth -= 1
    else:
        return len(code)
    if re.search(r"\bnamespace(?:\s+[\w:]+(?:\s+inline\s+\w+)?)?\s*$", code[max(0, i - 200):i]):
        return len(code)
    depth = 0
    for j in range(pos, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            if depth == 0:
                return j
            depth -= 1
    return len(code)


def r3_spelling(rel, target, prefix):
    """The qualified spelling R3 writes for an alias target (without the trailing ::)."""
    target = re.sub(r"^batchlas::", "", target)
    if rel.startswith("include/"):   # already inside batchlas::device
        return re.sub(r"^device(::)?", "", target) or "device"
    return prefix + target


def r3_decls(code, raw=None):
    """[(start, end, scope end, alias, target)] of the device-layer aliases declared in code."""
    decls, raw = [], dict(raw or {})
    for m in R3_ALIAS.finditer(code):
        alias, target = m.group(1), re.sub(r"^batchlas::", "", m.group(2))
        head, _, rest = target.partition("::")
        if head in raw and rest:    # an alias of an alias (`tr = dv::tier`)
            target = f"{raw[head]}::{rest}"
        elif not (R3_DEVICE.fullmatch(target) or R3_SRC.fullmatch(target)):
            continue
        raw[alias] = target
        decls.append((m.start(), m.end(), block_end(code, m.start()), alias, target))
    return decls


_R3_TREE = {}


def r3_inherited(rel):
    """Aliases rel sees at namespace scope from the repo headers it includes (transitively), read
    from disk once, before any file is rewritten."""
    if not _R3_TREE:
        exports, incs = {}, {}
        for f in files():
            if not f.endswith(CXX_EXT):
                continue
            code = code_only(open(os.path.join(REPO, f), errors="surrogateescape").read())
            exports[f] = {d[3]: d[4] for d in r3_decls(code) if d[2] == len(code)}
            incs[f] = re.findall(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', code, re.M)
        _R3_TREE.update(exports=exports, incs=incs)
    exports, incs = _R3_TREE["exports"], _R3_TREE["incs"]

    def resolve(src, inc):
        for base in (os.path.dirname(src), "include", "", "tests", "src"):
            cand = os.path.normpath(os.path.join(base, inc))
            if cand in exports:
                return cand
        return None
    seen, todo, out = {rel}, [rel], {}
    while todo:
        cur = todo.pop()
        for inc in incs.get(cur, ()):
            h = resolve(cur, inc)
            if h and h not in seen:
                seen.add(h)
                todo.append(h)
                for a, t in exports[h].items():
                    out.setdefault(a, t)
    return out


def r3(rel, text):
    if not (rel.startswith(ALIAS_SCOPES + ("include/",)) and rel.endswith(CXX_EXT)):
        return text, 0, []
    code = code_only(text)
    sycl = re.search(r"\busing\s+namespace\s+(?:::)?sycl\s*;", code)
    inside = re.search(r"\busing\s+namespace\s+(?:::)?batchlas\s*;|\bnamespace\s+batchlas\b(?!\s*=)", code)
    prefix = "" if inside and not sycl else "batchlas::"
    inherited = r3_inherited(rel)
    # An included header's alias is a declaration at offset 0 that deletes nothing.
    decls = [(0, 0, len(code), a, t) for a, t in sorted(inherited.items())] + r3_decls(code, inherited)
    decls = [(a, b, e, al, r3_spelling(rel, t, prefix)) for a, b, e, al, t in decls]
    if not decls:
        return text, 0, []
    # Edits in original coordinates: delete each declaration line, and rewrite each use with the
    # innermost (latest) declaration of that name whose scope covers it.
    edits = {a: (b, "") for a, b, _, _, _ in decls if b > a}
    for alias in {d[3] for d in decls}:
        for u in re.finditer(rf"(?<![\w:]){alias}::", code):
            cover = [d for d in decls if d[3] == alias and d[1] <= u.start() < d[2]]
            if cover and not any(d[0] <= u.start() < d[1] for d in decls):
                edits[u.start()] = (u.end(), f"{cover[-1][4]}::")
    for a in sorted(edits, reverse=True):
        b, rep = edits[a]
        text = text[:a] + rep + text[b:]
    total, problems = len(edits), []
    left_code = code_only(text)
    for alias in sorted({d[3] for d in decls}):
        if re.search(rf"(?<![\w:]){alias}::", left_code) and not re.search(rf"\bnamespace\s+{alias}\b", left_code):
            problems.append(f"{rel}: `{alias}::` used outside the scope of its removed alias")
    return text, total, problems


R4_MACRO = re.compile(r"\bBATCHLAS_DLA_(INLINE|HELPER|BODY)\b")


def r4(rel, text):
    if not rel.endswith(TEXT_EXT):
        return text, 0, []
    total, out = 0, []
    for line in text.splitlines(keepends=True):
        if "BREAK L6" not in line:   # the placement guard's fixture of the old spelling keeps it
            line, n = R4_MACRO.subn(r"BATCHLAS_DEVICE_\1", line)
            total += n
        out.append(line)
    return "".join(out), total, []


# R5 tables. ROUTINES: the per-routine detail namespaces of include/batchlas/device that merge into
# batchlas::device::detail (the sg_partition backends' <b>_detail and src's host *_detail stay).
ROUTINES = ("bdsdc", "bdsqr", "gebd2", "gebrd", "geqrf", "gesv", "gesvd", "getf2", "getrf", "getri", "getrs",
            "iluk", "inv", "jacobi", "laed2", "laed3", "laed4", "larfb", "larft", "orgqr", "ormqr", "ortho", "posv",
            "potrf", "potri", "potrs", "ritz_values", "sb2st", "spmm", "stebz", "stedc", "stein", "steqr", "sy2sb",
            "syev", "sygst", "sysv", "sytrd", "sytrf", "sytrs")
R5_SKIP = ("src/backends/",)   # cusolver.cc's host gesvd_detail
# (routine, old, new), applied before the merge: inside the routine's own detail blocks (unqualified)
# and wherever `<routine>_detail::old` is spelled. The names defined in two routines' detail
# (l1_collide.txt: a type gets a CamelCase routine prefix, anything else a snake_case one; Params
# becomes <routine>_params, since batchlas:: has the host StebzParams, StedcParams, SteinParams), stein's
# Scratch (it would hide device::Scratch from core detail), sygst's bounded tier traits (R6 hoists
# the plain ones), and the tier bodies src calls, named <routine>_<tier>_run.
LOCAL_RENAMES = tuple((r, "Layout", f"{p}Layout") for r, p in (
    ("bdsdc", "Bdsdc"), ("bdsqr", "Bdsqr"), ("ortho", "Ortho"), ("sb2st", "Sb2st"), ("stebz", "Stebz"),
    ("stedc", "Stedc"), ("stein", "Stein"), ("sy2sb", "Sy2sb"))) + tuple(
    (r, "layout", f"{r}_layout") for r in ("ortho", "stebz", "stein")) + tuple(
    (r, "cta_plan", f"{r}_cta_plan") for r in ("gebrd", "getrf", "steqr", "syev", "sygst", "sytrd")) + tuple(
    (r, "BlockedPlan", f"{p}BlockedPlan") for r, p in (
        ("gebrd", "Gebrd"), ("gesvd", "Gesvd"), ("potrf", "Potrf"), ("ritz_values", "RitzValues"), ("sygst", "Sygst"))) + tuple(
    (r, "CtaPlan", f"{p}CtaPlan") for r, p in (("gebrd", "Gebrd"), ("getrf", "Getrf"), ("potrf", "Potrf"), ("sygst", "Sygst"))) + tuple(
    (r, "Params", f"{r}_params") for r in ("bdsqr", "stebz", "stedc", "stein")) + tuple(
    (r, "blocked_plan", f"{r}_blocked_plan") for r in ("syev", "sygst", "sytrd")) + tuple(
    (r, "BlockedCfg", f"{p}BlockedCfg") for r, p in (("gebrd", "Gebrd"), ("potrf", "Potrf"))) + tuple(
    (r, "cfg_of", f"{r}_cfg_of") for r in ("larfb", "sy2sb")) + (
    ("stein", "Scratch", "SteinScratch"),
    ("sygst", "is_cta_v", "sygst_is_cta_v"), ("sygst", "is_blocked_v", "sygst_is_blocked_v"),
    ("potrf", "cta_body", "potrf_cta_run"), ("potrf", "potrf_reg", "potrf_reg_run"),
    ("getrf", "cta_run", "getrf_cta_run"), ("getrf", "reg_run", "getrf_reg_run"),
    ("sytrf", "sytrf_cta_factor", "sytrf_cta_run"), ("sytrs", "sytrs_cta_solve", "sytrs_cta_run"))
# (routine, name): moved out of <routine>_detail into batchlas::device by hand (LAPACK names).
PROMOTED = (("syev", "org2l"), ("syev", "orm2l"))


def detail_spans(code):
    """[(routine, start, end)] of every `namespace [batchlas::device::]<routine>_detail {` block."""
    out, stack = [], []
    for m in re.finditer(r"\bnamespace\s+((?:::)?[\w:]+)\s*\{|\{|\}", code):
        if m.group(0) == "{":
            stack.append(None)
        elif m.group(0) == "}":
            if stack:
                top = stack.pop()
                if top:
                    out.append((top[0], top[1], m.start()))
        else:
            leaf = m.group(1).split("::")[-1]
            r = leaf[:-len("_detail")] if leaf.endswith("_detail") else None
            stack.append((r, m.end()) if r in ROUTINES else None)
    return out


def r5(rel, text):
    if not rel.endswith(CXX_EXT) or rel.startswith(R5_SKIP):
        return text, 0, []
    total, problems = 0, []
    alt = "|".join(sorted(ROUTINES, key=len, reverse=True))
    # 1. Local renames inside the routine's own detail blocks, then its qualified spellings.
    code = code_only(text)
    edits = []
    for r, s, e in detail_spans(code):
        for rr, old, new in LOCAL_RENAMES:
            if rr == r:
                edits += [(m.start(), m.end(), new) for m in re.finditer(rf"(?<![\w.:>]){old}\b", code[:e]) if m.start() >= s]
    for a, b, new in sorted(edits, reverse=True):
        text = text[:a] + new + text[b:]
    total += len(edits)
    for r, old, new in LOCAL_RENAMES:
        text, n = sub_kinds(text, re.compile(rf"(?<![\w]){r}_detail::{old}\b"), f"{r}_detail::{new}", ("code", "comment"))
        total += n
    # 2. Promoted LAPACK names lose the qualifier.
    for r, name in PROMOTED:
        text, n = sub_kinds(text, re.compile(rf"(?<![\w]){r}_detail::{name}\b"), name, ("code", "comment"))
        total += n
    # 3. Openers, closing comments, qualifiers and bare mentions -> detail.
    text, n = sub_kinds(text, re.compile(rf"\b(namespace\s+(?:(?:::)?batchlas::device::)?)(?:{alt})_detail\b"),
                        r"\1detail", ("code", "comment"))
    total += n
    text, n = sub_kinds(text, re.compile(rf"(?<![\w])(?:{alt})_detail\b"), "detail", ("code", "comment"))
    total += n
    for m in re.finditer(r"\busing\s+namespace\s+(?:\w+::)*detail\s*;", code_only(text)):
        if rel.startswith("include/batchlas/device/"):
            problems.append(f"{rel}: `using namespace detail;` pulls every merged name in; qualify instead")
    return text, total, problems


R6_QUAL = re.compile(r"(?<![\w:])((?:(?:::)?batchlas::)?(?:device::)?)(?:(?:\w+_)?detail::)?(?:ormqr_)?is_(reg|cta|blocked)_v\b")
R6_DEF = re.compile(r"^[ \t]*(?:template\s*<[^;{]*?>\s*)?(?:inline\s+)?constexpr\s+bool\s+(?:ormqr_)?is_(?:reg|cta|blocked)_v\b"
                    r"[^;]*;[ \t]*(?://[^\n]*)?\n", re.M)


def r6(rel, text):
    if not rel.endswith(CXX_EXT) or rel == "include/batchlas/device/tier.hh" or not rel.startswith(("include/", "src/", "tests/", "examples/", "benchmarks/", "tools/")):
        return text, 0, []
    code = code_only(text)
    spans = [m.span() for m in R6_DEF.finditer(code)]
    for a, b in reversed(spans):
        text = text[:a] + text[b:]
    text, n = sub_kinds(text, R6_QUAL, r"\1tier::is_\2_v")
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
# Binaries and sources. D10 folds seven device_blas_v2_<r> binaries (and elementwise-glue) into the
# device_lapack_<r> (device_core_lapack_aux) binary as an extra source tests/<...>_suite.cc
# (tests/CMakeLists.txt BATCHLAS_TEST_SUITE_SOURCES); a source name maps to the suite file, the
# binary name to its owner. The git mv and the CMake fold of the R-b commit are not replayed here.
FOLDED = (("getrf", "getrf"), ("potrf-cta-lpanel", "potrf"), ("sb2st-sy2sb", "sb2st"), ("stedc", "stedc"),
          ("sygst-sygvd-syfn", "sygst"), ("sytrf-sytrs", "sytrf"), ("tridiag-stebz-stein", "stebz-stein"))
TEST_RENAMES = tuple(
    (rf"\bdevice_blas_v2_{x}_tests\.cc\b", f"device_lapack_{x}_suite.cc") for x, _ in FOLDED) + tuple(
    (rf"\bdevice_blas_v2_{x}_tests\b", f"device_lapack_{r}_tests") for x, r in FOLDED) + (
    (r"\bdevice_blas_v2_elementwise-glue_tests\.cc\b", "device_core_elementwise-glue_suite.cc"),
    (r"\bdevice_blas_v2_elementwise-glue_tests\b", "device_core_lapack_aux_tests"),
    (r"\bdevice_blas_v2_gemm-components_tests\b", "device_core_level3_components_tests"),
    (r"\bdevice_blas_v2_(gemv-native|trsm|syev-cta-trio|krylov-ortho-ritz)_tests\b", r"\1_kernels_tests"),
    (r"\bdevice_blas_v2_tests\b", "device_core_ops_tests"),
    (r"\bdevice_blas_v2_([\w-]+?)_tests\b", r"device_lapack_\1_tests"),
    (r"\bdevice_api_(\w[\w-]*)_tests\b", r"device_lapack_\1_tests"),
    (r"\bdevice_api_(\w+)", r"device_lapack_\1"),
)
TEXT_RENAMES = (
    (r"\bdesign_device_la_v2\b", "design_device_la"),
    (r"\bDevice LA v2: ", "Device LA: "),
    (r"(?<![\w-])device-la-v2-(?=\w)", "device-la-"),   # heading slugs of the renamed page
)
# sass_lab identifiers and variant labels ("v2" was the device LA variant).
LAB_RENAMES = (
    (r"\bkV2(\w*)", r"kDevice\1"), (r"\breg_v2([ab])(_wg)?\b", r"reg_device_\1\2"),
    (r"\bregister_v2([ab])\b", r"register_device_\1"), (r"\bops_v2([ab])\b", r"ops_device_\1"),
    (r"\bops_v2\*", "ops_device_*"), (r"\bdevice LA v2\b", "device LA"), (r"\bv2(_scaled|_blue|_scratch)?\b", r"device\1"),
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
    if rel.startswith("benchmarks/sass_lab/"):
        for rx, rep in LAB_RENAMES:
            text, n = re.subn(rx, rep, text)
            total += n
    return text, total, []


# R8: plan section 2 C10 (decision D2). Mode tags become enums (Side, Direct, Refl, PackOrder, Axis,
# Storage, DiagFill, Scope); policy tags move into alg with self-describing names (none of the new
# alg names was defined before: the collision check of the R-d step); reduction operators move to
# reduce_op. Applied to code, comments and literals (each pattern is a qualified spelling), and to
# Markdown. tri_diag stays a policy (alg::UnitDiag, Div, Recip, RealDiv): its Diag-enum form needs
# the two trsv overload sets merged first (D7). The enum definitions and the template parameters
# that change kind (class -> enum) are edited by hand; MODE_COMPARE rewrites their is_same_v tests.
TAG_RENAMES = (
    (r"(?<![\w:])(?:(?:::)?batchlas::)?device::alpha_side::(Left|Right)\b", r"batchlas::Side::\1"),
    (r"\balpha_side::(Left|Right)\b", r"Side::\1"),
    (r"\bdir::(Forward|Backward)\b", r"Direct::\1"),
    (r"\brefl::(QR|QL)\b", r"Refl::\1"),
    (r"\border::(ByCols|ByRows)\b", r"PackOrder::\1"),
    (r"\baxis::(Rows|Cols|Both)\b", r"Axis::\1"),
    (r"\bstorage::(Full|Band)\b", r"Storage::\1"),
    (r"\b(?:diag|unit)_fill::(Keep|Real|Unit|Zero)\b", r"DiagFill::\1"),
    (r"\btri_diag::Unit\b", "alg::UnitDiag"),
    (r"\btri_diag::Real\b", "alg::RealDiv"),
    (r"\btri_diag::(Div|Recip)\b", r"alg::\1"),
    (r"\bscope::Part\b", "Scope::Own"),
    (r"\bscope::(Sub|Group)\b", r"Scope::\1"),
    (r"\bScope::Auto\b", "Scope::Own"),
    (r"\bconv::Lapack\b", "alg::LarfgLapack"),
    (r"\bconv::(Phase|RealBeta)\b", r"alg::\1"),
    (r"\brot_conv::(Lapack|Transposed|Reflector2)\b", r"alg::Rot\1"),
    (r"\blartg_alg::(Exact|Fast)\b", r"alg::Lartg\1"),
    (r"\blarf_form::(ScaleW|ScaleV)\b", r"alg::\1"),
    (r"\bbcast_policy::(Shuffle|Slm)\b", r"alg::Bcast\1"),
    (r"\bloads::One\b", "alg::OneLoad"),
    (r"\bloads::Two\b", "alg::TwoLoads"),
    (r"\bunroll::(Default|None|Count|Full)\b", r"alg::Unroll\1"),
    (r"\bloop::(Lockstep|Diverge)\b", r"alg::\1"),
    (r"\bedge::(None|Guard|ZeroPad)\b", r"alg::Edge\1"),
    (r"\bkey::(Abs1|Abs|Real)\b", r"alg::Key\1"),
    (r"\btie::(Lowest|Highest)\b", r"alg::Tie\1"),
    (r"\bnan::(First|Skip|Propagate)\b", r"alg::Nan\1"),
    (r"\bbfly::(Down|Asc)\b", r"alg::Bfly\1"),
    (r"\bbfly::inline_body\b", "detail::butterfly_inline"),
    (r"\bws::Space\b", "WsSpace"),
    (r"\bws::Slot\b", "WsSlot"),
    (r"\bop::(ReduceN|Reduce|FMax|FMin)\b", r"reduce_op::\1"),
)
MODE_ENUM = r"((?:(?:::)?batchlas::)?(?:device::)?(?:Side|Direct|Refl|PackOrder|Axis|Storage|DiagFill)::\w+)"
MODE_COMPARE = (
    (r"!\s*std::is_same_v<\s*(\w+)\s*,\s*" + MODE_ENUM + r"\s*>", r"\1 != \2"),
    (r"std::is_same_v<\s*(\w+)\s*,\s*" + MODE_ENUM + r"\s*>", r"\1 == \2"),
)


def r8(rel, text):
    if not rel.endswith(CXX_EXT + (".md",)):
        return text, 0, []
    total = 0
    kinds = ("code", "comment", "literal")
    for rx, rep in TAG_RENAMES:
        if rel.endswith(".md"):
            text, n = re.subn(rx, rep, text)
        else:
            text, n = sub_kinds(text, re.compile(rx), rep, kinds)
        total += n
    if rel.endswith(CXX_EXT):
        for rx, rep in MODE_COMPARE:
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
