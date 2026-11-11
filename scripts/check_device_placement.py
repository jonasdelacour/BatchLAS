#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Placement guard for the public device LA layer (v22 design A.0).

    scripts/check_device_placement.py [--root DIR] [--allowlist FILE] [--write-allowlist] [--self-test]
                                       [--layout]

Rules, each a (rule, file, name) finding:
  src-device     src/ defines a name in namespace batchlas::device (any sub-namespace): src/ops/<op>/
                 keeps host code, kernel functors and launchers; device code is public.
  lapack-header  include/batchlas/device/lapack/<r>.hh defines a name that is not one of its routines
                 (ROUTINES[<r>], default {<r>}), a `<routine>_...` name, or in its `detail` (merged: the file owns it).
  lapack-core    a public device header outside lapack/ defines a LAPACK routine (LAPACK_NAMES, or a
                 `<routine>_...` name): it belongs in lapack/<routine>.hh.
  duplicate      one unqualified namespace-scope function name defined in two device files with
                 overlapping arity ranges: public device headers, src/ definitions in batchlas::device,
                 and every namespace-scope function of a src/ family header (*_v2.hh, dla_v2_*.hh).
Device files: include/batchlas/device.hh, include/batchlas/device/**, include/batchlas/blas/device/**
(v1 and v2 until they move), and src/ files with a definition in batchlas::device.

The allowlist (default scripts/device_placement_allowlist.txt) holds today's findings, one
`<rule> <file> <name>` per line; later work packages shrink it to empty. A finding not on it fails,
and so does an entry that no longer matches (delete the line, or --write-allowlist).
The parser is a scan over comment- and string-stripped text: it tracks namespace and brace scopes and
names each namespace-scope declaration (function, class, enum, alias, variable). Declarations without
a body (`void f(int);`, `struct S;`), using-declarations and class template specializations
(`struct tier_info<routine::potrf<UL>, ...> {...}`: a routine's facts for a core template) are not
definitions.
Layout rules (#250 cleanup plan section 1; layout_findings lists them) are enforced the same way
(since WP R-d): a layout finding not on the allowlist fails, and so does a stale layout entry.
--layout lists every layout finding, allowlisted or not.
Exit codes: 0 clean, 1 findings or stale entries (or a failed self-test).
"""
import argparse
import collections
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ALLOWLIST = os.path.join(REPO, "scripts", "device_placement_allowlist.txt")
EXTS = (".cc", ".hh", ".h", ".hpp", ".cpp")
PUBLIC_DEVICE = ("include/batchlas/device/", "include/batchlas/blas/device/")
PUBLIC_DEVICE_FILES = ("include/batchlas/device.hh", "include/batchlas/blas/device.hh")
LAPACK_DIR = "include/batchlas/device/lapack/"
FAMILY_HEADER = re.compile(r"^src/.*(?:_v2|/dla_v2_\w+)\.hh$")

# Routines a lapack/<header>.hh carries besides its own name (v22 design C.1).
ROUTINES = {"potri": ("potri", "trtri", "lauum"), "jacobi": ("jacobi", "syevj", "gesvj"),
            "sytrd": ("sytrd", "latrd", "sytd2"), "gebrd": ("gebrd", "labrd", "gebd2"),
            "potrf": ("potrf", "potf2"), "getrf": ("getrf", "getf2"), "geqrf": ("geqrf", "geqr2"),
            "sytrf": ("sytrf", "sytf2", "lasyf"), "sytrs": ("sytrs", "syconv"),
            "orgtr": ("orgtr", "org2l"), "ormtr": ("ormtr", "orm2l"), "stebz": ("stebz", "laebz"),
            "syev": ("syev", "org2l", "orm2l"),  # the Cta tier's column-owner xORG2L / xORM2L
            "stedc": ("stedc", "laed4"), "sygst": ("sygst", "sygs2")}
# The LAPACK routines of v22 design C.1 and A.3 (core auxiliaries such as lartg, larfg, lascl are not).
LAPACK_NAMES = frozenset({
    "potrf", "potf2", "potrs", "posv", "potri", "trtri", "lauum", "getrf", "getf2", "getrs", "gesv", "getri",
    "geqrf", "geqr2", "orgqr", "ormqr", "sytrf", "sytf2", "lasyf", "sytrs", "syconv", "sysv", "sygst", "sygs2",
    "sygvd", "sytrd", "latrd", "sytd2", "orgtr", "org2l", "ormtr", "orm2l", "steqr", "stedc", "laed4", "stebz",
    "laebz", "stein", "sy2sb", "sb2st", "syev", "syevx", "syevj", "gesvj", "gebrd", "labrd", "gebd2", "ormbr",
    "bdsqr", "bdsdc", "gesvd", "gesvdj"})

STRIP = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|^[ \t]*#(?:[^\n]*\\\n)*[^\n]*",
                   re.S | re.M)
NS_OPEN = re.compile(r"(?:inline\s+)?namespace\b\s*((?:inline\s+)?[A-Za-z_]\w*(?:\s*::\s*(?:inline\s+)?[A-Za-z_]\w*)*)?"
                     r"\s*(?:\[\[[^\]]*\]\]\s*)?\{")
KEYWORDS = frozenset({"decltype", "alignas", "noexcept", "requires", "sizeof", "alignof", "static_assert",
                      "__attribute__", "__declspec", "if", "return", "operator", "explicit"})
IDENT = re.compile(r"[A-Za-z_]\w*")
Decl = collections.namedtuple("Decl", "ns name kind line arity")


def strip(text):
    return STRIP.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def drop_template_heads(head):
    """`template <...>` prefixes (balanced angles) and [[attributes]] removed from a declaration head."""
    head = re.sub(r"\[\[.*?\]\]", " ", head)
    while True:
        m = re.match(r"\s*template\s*<", head)
        if not m:
            return head.strip()
        depth, j = 1, m.end()
        while j < len(head) and depth:
            depth += {"<": 1, ">": -1}.get(head[j], 0)
            j += 1
        head = head[j:]


def top_level(head, ch):
    """Index of the first `ch` outside (), [] and {} in head, or -1."""
    depth = 0
    for i, c in enumerate(head):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ch and depth == 0:
            return i
    return -1


def arity(params):
    """(min, max) argument counts of a parameter list text; max None = variadic."""
    params = params.strip()
    if not params or params == "void":
        return (0, 0)
    parts, depth, cur = [], 0, ""
    for c in params:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    variadic = any("..." in p for p in parts)
    required = sum(1 for p in parts if top_level(p, "=") < 0 and "..." not in p)
    return (required, None if variadic else len(parts))


def function_name(head):
    """(name, (min, max)) of a function head `... name(params) ...`, or None."""
    for m in re.finditer(r"\(", head):
        before = head[:m.start()].rstrip()
        angles = re.sub(r"->|<<|>>|operator\s*\S+", "", before)
        if before.count("(") != before.count(")") or angles.count("<") != angles.count(">"):
            continue
        op = re.search(r"\boperator\s*(\(\s*\)|[^\w\s(]+|\w+)\s*$", before)
        if op:
            name = "operator" + op.group(1).replace(" ", "")
        else:
            idm = re.search(r"([A-Za-z_]\w*)\s*$", before)
            if not idm or idm.group(1) in KEYWORDS or idm.group(1).isupper():
                continue
            name = idm.group(1)
            if before[:idm.start()].rstrip().endswith("::"):
                return None  # an out-of-line member definition, not a namespace-scope name
        depth, j = 1, m.end()
        while j < len(head) and depth:
            depth += {"(": 1, ")": -1}.get(head[j], 0)
            j += 1
        return name, arity(head[m.end():j - 1])
    return None


def classify(head, has_body):
    """(kind, name, arity) for a namespace-scope declaration head, or None if it defines nothing."""
    head = drop_template_heads(re.sub(r"\s+", " ", head))
    if not head or re.match(r"(using\s+namespace|static_assert|extern\b|friend\b|typedef\b)", head):
        return None
    m = re.match(r"using\s+([A-Za-z_]\w*)\s*=", head)
    if m:
        return ("alias", m.group(1), None)
    if head.startswith("using "):
        return None
    m = re.match(r"(?:enum\s+(?:class\s+|struct\s+)?|(?:struct|class|union)\s+)(?:alignas\s*\([^)]*\)\s*|[A-Z_]+\s+)*"
                 r"([A-Za-z_]\w*)", head)
    if m:
        if head[m.end():].lstrip().startswith("<"):
            return None  # a specialization (tier_info<routine::potrf<UL>, ...>) names no new entity
        return ("type", m.group(1), None) if has_body else None
    eq = top_level(head, "=")
    if eq >= 0 and eq < (top_level(head, "(") if top_level(head, "(") >= 0 else len(head)) + 1:
        idm = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", head[:eq])
        return ("variable", idm.group(1), None) if idm else None
    fn = function_name(head)
    if fn:
        return ("function", fn[0], fn[1]) if has_body else None
    if has_body:
        idm = re.search(r"([A-Za-z_]\w*)\s*$", head)
        return ("variable", idm.group(1), None) if idm and idm.group(1) not in KEYWORDS else None
    return None


def declarations(text):
    """Every namespace-scope declaration of stripped text as Decl(ns, name, kind, line, arity)."""
    out, stack, head_start, paren = [], [("ns", "")], 0, 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        top = stack[-1]
        if top[0] == "ns" and paren == 0 and c == "n" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = NS_OPEN.match(text, i)
            if m:
                name = re.sub(r"\binline\s+|\s+", "", m.group(1) or "(anonymous)")
                stack.append(("ns", f"{top[1]}::{name}".lstrip(":")))
                i = m.end()
                head_start = i
                continue
        if c in "()":
            paren += 1 if c == "(" else -1
        elif c == "{":
            if top[0] == "ns" and paren == 0:
                head = text[head_start:i]
                if re.fullmatch(r"\s*extern\s*", head):
                    stack.append(("ns", top[1]))
                    head_start = i + 1
                    i += 1
                    continue
                d = classify(head, True)
                if d:
                    out.append(Decl(top[1], d[1], d[0], text.count("\n", 0, i) + 1, d[2]))
                stack.append(("body", top[1]))
            else:
                stack.append(("other", top[1]))
        elif c == "}":
            closed = stack.pop() if len(stack) > 1 else stack[-1]
            if stack[-1][0] == "ns" and closed[0] in ("ns", "body"):
                head_start = i + 1
                paren = 0
        elif c == ";" and top[0] == "ns" and paren == 0:
            d = classify(text[head_start:i], False)
            if d:
                out.append(Decl(top[1], d[1], d[0], text.count("\n", 0, i) + 1, d[2]))
            head_start = i + 1
        i += 1
    return out


def in_device(ns):
    return ns == "batchlas::device" or ns.startswith("batchlas::device::")


def is_public_device(rel):
    return rel in PUBLIC_DEVICE_FILES or rel.startswith(PUBLIC_DEVICE)


def routine_of(name, routines):
    return any(name == r or name.startswith(r + "_") for r in routines)


def overlap(a, b):
    lo = max(a[0], b[0])
    hi = min(x for x in (a[1], b[1]) if x is not None) if (a[1], b[1]) != (None, None) else None
    return hi is None or lo <= hi


def sources(root):
    files = {}
    for top in ("src", "include"):
        for d, _, names in os.walk(os.path.join(root, top)):
            for name in names:
                if name.endswith(EXTS):
                    p = os.path.join(d, name)
                    files[os.path.relpath(p, root).replace(os.sep, "/")] = p
    return dict(sorted(files.items()))


def findings(root):
    """{(rule, file, name): first line} over the tree at root."""
    found, funcs = {}, collections.defaultdict(list)

    def add(rule, rel, name, line):
        found.setdefault((rule, rel, name), line)

    for rel, path in sources(root).items():
        every = declarations(strip(open(path, errors="replace").read()))
        decls = [d for d in every if in_device(d.ns)]
        public = is_public_device(rel)
        if FAMILY_HEADER.search(rel):
            for d in every:
                if d.kind == "function" and not in_device(d.ns) and not d.name.startswith("operator"):
                    funcs[d.name].append((rel, d))
        if not decls or not (public or rel.startswith("src/")):
            continue
        for d in decls:
            qual = f"{d.ns}::{d.name}"
            if rel.startswith("src/"):
                add("src-device", rel, qual, d.line)
            if rel.startswith(LAPACK_DIR):
                stem = os.path.splitext(rel[len(LAPACK_DIR):])[0].replace("/", "_")
                rs = ROUTINES.get(stem, (stem,))
                det = d.ns == "batchlas::device::detail" or any(seg.startswith(tuple(r + "_" for r in rs)) for seg in d.ns.split("::"))
                if not (routine_of(d.name, rs) or det):
                    add("lapack-header", rel, qual, d.line)
            elif public and d.kind in ("function", "type") and routine_of(d.name, LAPACK_NAMES):
                add("lapack-core", rel, qual, d.line)
            if d.kind == "function" and not d.name.startswith("operator"):
                funcs[d.name].append((rel, d))
    for name, defs in funcs.items():
        for rel, d in defs:
            for rel2, d2 in defs:
                if rel2 != rel and overlap(d.arity, d2.arity):
                    add("duplicate", rel, f"{name}/{rel2}", d.line)
    return found


DEV = "batchlas::device"
LAYOUT_NS = frozenset({DEV} | {f"{DEV}::{s}" for s in (
    "detail", "detail::nvptx", "detail::amdgcn", "detail::spirv", "detail::sscp", "tier", "routine", "l3", "l3::cfg",
    "alg", "arith", "epi", "reduce_op")})
SRC_DEVICE_NS = ("potrf_native", "getrf_native", "geqrf_native", "tiny_native", "sycl_device", "syev_v2", "stedc_v2",
                 "krylov_v2", "ops::trsm::dla", "ops::gemv::dla")
V2_FILE = re.compile(r"(?:^|[_\-/])v2(?:[_\-.]|$)|^v2")
V2_KEEP = re.compile(r"wide_v2|kernel_hash_v2|nvml")
INLINE_NS = re.compile(r"\binline\s+namespace\b|\bnamespace\s+[\w:]*::inline\s")
DEVICE_ALIAS = re.compile(r"\bnamespace\s+(\w+)\s*=\s*((?:::)?(?:batchlas::)?device\b(?:::\w+)*)\s*;")
TIER_TRAIT = re.compile(r"^is_(?:reg|cta|blocked)_v$")


def layout_findings(root):
    """{(rule, file, name): first line} for the plan section-1 layout (enforced since WP R-d).

    layout-ns         a definition in a batchlas::device namespace outside LAYOUT_NS, or any inline
                      namespace, in a device file
    layout-alias      src/ or examples/: `namespace X = batchlas::device...`
    layout-src-ns     src/: a definition in a src device namespace (SRC_DEVICE_NS)
    layout-detail-use src/ or examples/: a device detail name other than `detail::<routine>_run`
    layout-detail-dup one name defined in two device detail namespaces (the merge would collide)
    layout-tier-trait is_{reg,cta,blocked}_v defined outside batchlas::device::tier
    layout-v2         a v2 file name or a BATCHLAS_DLA_* macro in the device layer
    """
    found, dets, detail_ns = {}, collections.defaultdict(set), set()

    def add(rule, rel, name, line):
        found.setdefault((rule, rel, name), line)

    texts = {}
    for rel, path in sources(root).items():
        text = strip(open(path, errors="replace").read())
        texts[rel] = text
        decls = declarations(text)
        dev = [d for d in decls if in_device(d.ns)]
        device_file = is_public_device(rel) or bool(dev)
        for d in dev:
            leaf = d.ns.split("::")[-1]
            if leaf.endswith("_detail") or leaf == "detail":
                detail_ns.add(leaf)
                if not re.search(r"::(?:nvptx|amdgcn|spirv|sscp)(?:_detail)?\b", d.ns):
                    dets[d.name].add(d.ns)
            if device_file and d.ns not in LAYOUT_NS:
                add("layout-ns", rel, d.ns, d.line)
            if TIER_TRAIT.match(d.name) and d.ns != f"{DEV}::tier":
                add("layout-tier-trait", rel, f"{d.ns}::{d.name}", d.line)
        if device_file:
            for m in INLINE_NS.finditer(text):
                add("layout-ns", rel, "inline namespace", text.count("\n", 0, m.start()) + 1)
        if rel.startswith("src/"):
            for d in decls:
                rest = d.ns[len("batchlas::"):] if d.ns.startswith("batchlas::") else ""
                if any(rest == s or rest.startswith(s + "::") for s in SRC_DEVICE_NS):
                    add("layout-src-ns", rel, f"{d.ns}::{d.name}", d.line)
        if (is_public_device(rel) or rel.startswith("src/")) and V2_FILE.search(os.path.basename(rel)) \
                and not V2_KEEP.search(rel):
            add("layout-v2", rel, "file name", 1)
    for top in ("src", "include", "examples"):
        for d, _, names in os.walk(os.path.join(root, top)):
            for name in names:
                if not name.endswith(EXTS):
                    continue
                rel = os.path.relpath(os.path.join(d, name), root).replace(os.sep, "/")
                text = texts.get(rel) or strip(open(os.path.join(d, name), errors="replace").read())
                raw = open(os.path.join(d, name), errors="replace").read()
                m = re.search(r"\bBATCHLAS_DLA_(?:INLINE|HELPER|BODY)\b", raw)
                if m:
                    add("layout-v2", rel, "BATCHLAS_DLA_* macro", raw.count("\n", 0, m.start()) + 1)
                if not rel.startswith(("src/", "examples/")):
                    continue
                for m in DEVICE_ALIAS.finditer(text):
                    add("layout-alias", rel, f"{m.group(1)} = {m.group(2)}", text.count("\n", 0, m.start()) + 1)
                if detail_ns:
                    alts = sorted(n for n in detail_ns if n != "detail") + ["device::detail"]
                    rx = re.compile(r"(?<![\w])(?:(?:::)?batchlas::)?(?:device::)?(" + "|".join(alts) + r")::(\w+)")
                    for m in rx.finditer(text):
                        if not m.group(2).endswith("_run"):
                            add("layout-detail-use", rel, f"{m.group(1)}::{m.group(2)}",
                                text.count("\n", 0, m.start()) + 1)
    for name, spaces in dets.items():
        if len(spaces) > 1:
            add("layout-detail-dup", "-", f"{name}: {' '.join(sorted(spaces))}", 0)
    return found


LAYOUT_TREE = {
    "include/batchlas/device/core.hh": """
        namespace batchlas::device::inline v2 { struct Ok {}; }                    // BREAK L1 (inline)
        namespace batchlas::device::scope { struct Part {}; }                     // BREAK L2 (mode namespace)
        namespace batchlas::device::tier { template <class T> constexpr bool is_cta_v = true; }
        namespace batchlas::device::potrf_detail { template <class T> constexpr bool is_cta_v = true;  // BREAK L3, L4
            struct Layout {}; }
        namespace batchlas::device::stein_detail { struct Layout {}; }            // BREAK L5 (dup)
        #define X BATCHLAS_DLA_INLINE                                              // BREAK L6
        """,
    "src/ops/foo/foo.cc": """
        namespace v2 = batchlas::device;                                         // BREAK L7
        namespace batchlas::tiny_native { inline int f(int k) { return k; } }    // BREAK L8
        namespace batchlas { int g() { return device::potrf_detail::Layout{}, device::detail::potrf_run<int>(); } }
        """,
}
LAYOUT_WANT = {
    ("layout-ns", "include/batchlas/device/core.hh", "inline namespace"),
    ("layout-ns", "include/batchlas/device/core.hh", "batchlas::device::v2"),
    ("layout-ns", "include/batchlas/device/core.hh", "batchlas::device::scope"),
    ("layout-ns", "include/batchlas/device/core.hh", "batchlas::device::potrf_detail"),
    ("layout-ns", "include/batchlas/device/core.hh", "batchlas::device::stein_detail"),
    ("layout-tier-trait", "include/batchlas/device/core.hh", "batchlas::device::potrf_detail::is_cta_v"),
    ("layout-detail-dup", "-", "Layout: batchlas::device::potrf_detail batchlas::device::stein_detail"),
    ("layout-v2", "include/batchlas/device/core.hh", "BATCHLAS_DLA_* macro"),
    ("layout-alias", "src/ops/foo/foo.cc", "v2 = batchlas::device"),
    ("layout-src-ns", "src/ops/foo/foo.cc", "batchlas::tiny_native::f"),
    ("layout-detail-use", "src/ops/foo/foo.cc", "potrf_detail::Layout"),
}


def layout_self_test():
    with tempfile.TemporaryDirectory() as root:
        for rel, text in LAYOUT_TREE.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(text)
        got = set(layout_findings(root))
    for f in sorted(got - LAYOUT_WANT):
        print(f"  unexpected: {f}")
    for f in sorted(LAYOUT_WANT - got):
        print(f"  missed: {f}")
    ok = got == LAYOUT_WANT
    print(f"check_device_placement: layout self-test {'passed' if ok else 'FAILED'} ({len(got)} findings, 7 rules)")
    return 0 if ok else 1


def read_allowlist(path):
    if not os.path.isfile(path):
        return set()
    out = set()
    for ln in open(path):
        ln = ln.split("#", 1)[0].strip()
        if ln:
            rule, rel, name = ln.split(None, 2)
            out.add((rule, rel, name))
    return out


def write_allowlist(path, found):
    with open(path, "w") as fh:
        fh.write("# device_placement_tests allowlist: today's A.0 and layout violations (scripts/check_device_placement.py).\n"
                 "# One `<rule> <file> <name>` per line; work packages delete lines, never add them.\n"
                 "# Regenerate: python3 scripts/check_device_placement.py --write-allowlist\n")
        for rule, rel, name in sorted(found):
            fh.write(f"{rule} {rel} {name}\n")


SELF_TREE = {
    # rule src-device: a definition in batchlas::device under src/ (BREAK 1), host code passes
    "src/ops/foo/foo_v2.hh": """
        namespace batchlas::device::v2::lu {
        template <class T> BATCHLAS_DEVICE_INLINE T pivot(T a, int n = 0) { return a; }   // BREAK 1
        void declared_only(int);
        }
        namespace batchlas { namespace device { inline namespace v2 { struct Packed { int x; }; } } }  // BREAK 1b
        namespace batchlas::device::v2 { namespace lu { using ok = int; } }   // BREAK 1c (alias)
        """,
    "src/ops/foo/foo.cc": """
        namespace v2 = batchlas::device::v2;
        namespace batchlas { struct FooKernel { void operator()(int) const { v2::lu::pivot(1); } }; }
        """,
    # rule lapack-header: potrf.hh may define potrf, potf2_*, potrf_detail::*; not gemv (BREAK 2)
    "include/batchlas/device/lapack/potrf.hh": """
        namespace batchlas::device { inline namespace v2 {
        template <class Ex, class M> void potrf(const Ex& ex, const M& a) {}
        template <class Ex, class M> void potf2_lower(const Ex& ex, const M& a) {}
        namespace potrf_detail { inline int step(int k) { return k; } }
        constexpr int potrf_buffer_size(int n) { return n; }
        template <class T> struct tier_info<routine::potrf, Cta<64>, T> { static constexpr int wg_size = 128; };
        template <class Ex, class M> void gemv(const Ex& ex, const M& a) {}     // BREAK 2
        } }
        """,
    # rule lapack-core: a routine in a core header (BREAK 3); an auxiliary (larfg) passes
    "include/batchlas/device/core.hh": """
        namespace batchlas::device::inline v2 {
        template <class Ex, class V> void larfg(const Ex& ex, V v) {}
        template <class Ex, class V> void getf2(const Ex& ex, V v) {}           // BREAK 3
        struct S { void getrf() {} };
        template <class T, int N> struct Frag { T v[N]; };
        }
        """,
    # rule duplicate: sum_to_lane0 twice with overlapping arity (BREAK 4); conj_if 1 vs 2 args passes
    "include/batchlas/device/reduce.hh": """
        namespace batchlas::device::v2 {
        template <int SG, class G, class T> T sum_to_lane0(const G& g, T v) { return v; }
        template <bool C, class T> T conj_if(T a) { return a; }
        template <class T, class... U> Refs<T, 1 + int(sizeof...(U))> refs(T& a, U&... b) { return {}; }
        }
        """,
    "include/batchlas/device/level2.hh": """
        namespace batchlas::device::v2 { namespace glue {
        template <class G, class T> T sum_to_lane0(const G& g, T v, int s = 1) { return v; }  // BREAK 4
        template <class V> auto conj_if(V v, bool c) { return v; }
        template <class T> Refs<T, int(2)> refs(T& a, T& b) { return {}; }      // BREAK 5 (vs variadic)
        auto operator*(int a, int b) -> decltype(a) { return a; }
        } }
        """,
}
SELF_WANT = {
    ("src-device", "src/ops/foo/foo_v2.hh", "batchlas::device::v2::lu::pivot"),
    ("src-device", "src/ops/foo/foo_v2.hh", "batchlas::device::v2::Packed"),
    ("src-device", "src/ops/foo/foo_v2.hh", "batchlas::device::v2::lu::ok"),
    ("lapack-header", "include/batchlas/device/lapack/potrf.hh", "batchlas::device::v2::gemv"),
    ("lapack-core", "include/batchlas/device/core.hh", "batchlas::device::v2::getf2"),
    ("duplicate", "include/batchlas/device/reduce.hh", "sum_to_lane0/include/batchlas/device/level2.hh"),
    ("duplicate", "include/batchlas/device/level2.hh", "sum_to_lane0/include/batchlas/device/reduce.hh"),
    ("duplicate", "include/batchlas/device/reduce.hh", "refs/include/batchlas/device/level2.hh"),
    ("duplicate", "include/batchlas/device/level2.hh", "refs/include/batchlas/device/reduce.hh"),
}


def self_test():
    with tempfile.TemporaryDirectory() as root:
        for rel, text in SELF_TREE.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(text)
        got = set(findings(root))
        allow = os.path.join(root, "allow.txt")
        write_allowlist(allow, got | {("src-device", "src/gone.hh", "batchlas::device::x")})
        stale = read_allowlist(allow) - got
    ok = got == SELF_WANT and stale == {("src-device", "src/gone.hh", "batchlas::device::x")}
    if not ok:
        for f in sorted(got - SELF_WANT):
            print(f"  unexpected: {f}")
        for f in sorted(SELF_WANT - got):
            print(f"  missed: {f}")
    print(f"check_device_placement: self-test {'passed' if ok else 'FAILED'} ({len(got)} findings, 4 rules)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--allowlist", default=DEFAULT_ALLOWLIST)
    ap.add_argument("--write-allowlist", action="store_true", help="replace the allowlist with today's findings")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--layout", action="store_true", help="list every layout finding (default: counts only)")
    a = ap.parse_args()
    if self_test() != 0 or layout_self_test() != 0:
        return 1
    if a.self_test:
        return 0
    found = findings(a.root)
    layout = layout_findings(a.root)
    found.update(layout)
    if a.write_allowlist:
        write_allowlist(a.allowlist, found)
        print(f"check_device_placement: wrote {len(found)} entries to {a.allowlist}")
        return 0
    allowed = read_allowlist(a.allowlist)
    new = sorted(set(found) - allowed)
    stale = sorted(allowed - set(found))
    for rule, rel, name in new:
        print(f"{rel}:{found[(rule, rel, name)]}: {rule}: {name}")
    for rule, rel, name in stale:
        print(f"{os.path.relpath(a.allowlist, a.root)}: stale entry (fixed? delete the line): {rule} {rel} {name}")
    count = collections.Counter(k[0] for k in found if k not in layout)
    print(f"check_device_placement: {len(found) - len(layout)} findings "
          f"({', '.join(f'{k} {v}' for k, v in sorted(count.items()))}), {len(new)} not allowlisted, {len(stale)} stale")
    if a.layout:
        for rule, rel, name in sorted(layout):
            print(f"{rel}:{layout[(rule, rel, name)]}: {rule}: {name}")
    lcount = collections.Counter(r for r, _, _ in layout)
    print(f"check_device_placement: layout findings {len(layout)} "
          f"({', '.join(f'{k} {v}' for k, v in sorted(lcount.items()))}, all enforced against the allowlist)"
          f"{'' if a.layout else '; --layout lists them'}")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
