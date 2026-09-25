#!/usr/bin/env python3
"""Check that no installed header declares a type in the consumer's global namespace.

Sibling guarantee to check_no_unprefixed_includes.py. That one keeps `batchlas`
the only name we claim in a consumer's *include root*; this one keeps `batchlas`
the only name we claim in a consumer's *global namespace*.

The defect it exists for: six installed headers declared 14 types -- Device,
DeviceProperty, DeviceType, Event, EventImpl, Policy, Queue, QueueImpl, Vendor,
Span, is_std_array, BumpAllocator, UnifiedVector, ReferenceWrapper -- at global
scope instead of inside `namespace batchlas`. `::Span`, `::Device`, `::Event`,
`::Queue`, `::Vendor` and `::Policy` are among the most collision-prone names in
GPU/HPC C++: any consumer that owns one of them gets an ambiguity, a
redeclaration conflict, or (worst) an ODR violation that links and then
misbehaves.

Nothing else catches a regression here. The in-tree build cannot: every
translation unit in this repo either says `using namespace batchlas;` or is
itself inside namespace batchlas, so a name at global scope and the same name in
batchlas are indistinguishable from in here. consumer_package_tests does not
catch it either -- examples/consumer/main.cc is a friendly consumer that owns no
colliding name. The failure is only ever visible from a consumer's source, which
is exactly the population no CI job in this repo compiles.

WHAT IS FLAGGED
    A struct / class / union / enum / enum class / using / typedef declaration
    at brace depth 0, i.e. in the global namespace. `template <...>` is
    transparent, so `template <typename T> struct Span;` is flagged on its
    `struct`.

THE ONE SANCTIONED EXCEPTION
    A compatibility shim at the end of a moved header:

        #ifndef BATCHLAS_NO_GLOBAL_NAMES
        using batchlas::Queue;
        #endif

    A using-declaration naming the same entity as a `using namespace batchlas;`
    is not ambiguous, so in-tree code compiles either way while a consumer gets
    an opt-out. Declarations inside such a block are allowed -- but ONLY
    `using batchlas::Something;`. Anything else (a struct definition, a
    `using namespace batchlas;`, an alias to a non-batchlas entity) is reported,
    because a shim that can hold arbitrary declarations is not an exception, it
    is a hole.

WHAT IS DELIBERATELY NOT FLAGGED
    * Free functions and function templates at global scope. D-1 is about type
      names; a function collides only when it is also called ambiguously, and
      include/batchlas/util/bench_structured.hh legitimately keeps
      bench_event_elapsed_ms() at global scope.
    * Declarations inside a non-batchlas namespace -- `namespace bench` and
      `namespace minibench` in the two structured-benchmark headers. Those sit
      at brace depth 1, they claim a namespace name rather than a type name, and
      both headers are EXCLUDEd from install (cmake/BatchLASPackaging.cmake), so
      a consumer never sees them.
    * Macros. Preprocessor lines are blanked before scanning (their bodies carry
      unbalanced braces -- see MINI_BENCHMARK_MAIN in util/minibench.hh), so a
      type declared by a macro body is invisible here.

Usage:
    python3 .github/ci/check_no_global_names.py [include-dir]
Exit code 0 = clean, 1 = findings.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INCLUDE_ROOT = os.path.join(REPO, "include")

HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp")

# The macro whose #ifndef opens the sanctioned shim block.
SHIM_MACRO = "BATCHLAS_NO_GLOBAL_NAMES"

# Declaration introducers reported at brace depth 0. `enum class` is covered by
# `enum`: the scan reports on the first keyword and then stops treating the rest
# of the statement as a fresh declaration.
DECL_KEYWORDS = frozenset(("struct", "class", "union", "enum", "using", "typedef"))

# Specifiers that may precede a declaration without ending it, so the token
# after them still starts one (`extern template struct Span<int>;`).
TRANSPARENT = frozenset(("extern", "inline", "static", "constexpr", "consteval",
                         "constinit", "thread_local"))

# Identifiers, `::` as a single token (so a shim's `batchlas::Queue` reassembles),
# the structural punctuation the scanner cares about, then anything else.
TOKEN = re.compile(r"[A-Za-z_]\w*|::|[{}();<>=,]|\S")

DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*([A-Za-z_]\w*)?[ \t]*(.*)$")


def _blank(text):
    """Same length as `text`, but only newlines survive (keeps line numbers)."""
    return "".join("\n" if ch == "\n" else " " for ch in text)


def strip_comments_and_literals(text):
    """Blank //, /* */, "...", '...' and R"delim(...)delim".

    Character-by-character rather than by regex because the cases that matter
    are exactly the ones a regex gets wrong: a `//` inside a string, a quote
    inside a comment, and a raw string holding either.

    cmakeparse.clean_source does the same job for CMake and is NOT reusable
    here: it treats `#` as a comment introducer, which would blank every
    `#include` and every `#ifndef` in the file -- including the shim's.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            out.append(_blank(text[i:end]))
            i = end
        elif ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(_blank(text[i:end]))
            i = end
        elif ch in "\"'":
            # A raw string is the only literal whose end is not `\`-escapable.
            if ch == '"' and i > 0 and text[i - 1] == "R":
                m = re.compile(r'"([^(\s\\]*)\(').match(text, i)
                if m:
                    closer = ")" + m.group(1) + '"'
                    end = text.find(closer, m.end())
                    end = n if end < 0 else end + len(closer)
                    out.append(_blank(text[i:end]))
                    i = end
                    continue
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
            out.append(_blank(text[i:end]))
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def strip_directives(text):
    """Blank every preprocessor line; return (code, shim_line_ranges).

    Blanking is not optional: macro bodies carry unbalanced braces and would
    desynchronise the depth counter for the rest of the file.

    The returned ranges are the 1-based inclusive line spans over which
    `#ifndef BATCHLAS_NO_GLOBAL_NAMES` is in effect. Nesting is tracked so an
    unrelated #ifdef inside a shim does not close it, and an #else flips the
    branch back out of the shim (the else-branch is the *defined* one).
    """
    lines = text.split("\n")
    code = []
    spans = []
    cond_stack = []            # one entry per open #if/#ifdef/#ifndef
    shim_open_at = None        # 1-based line where the current shim opened
    continued = False          # previous directive line ended with a backslash
    for idx, line in enumerate(lines, 1):
        directive = continued or line.lstrip().startswith("#")
        if not directive:
            code.append(line)
            continued = False
            continue
        code.append(_blank(line))
        if not continued:
            m = DIRECTIVE.match(line)
            name = (m.group(1) or "") if m else ""
            rest = (m.group(2) or "").strip() if m else ""
            if name in ("if", "ifdef", "ifndef"):
                is_shim = bool(
                    (name == "ifndef" and rest.split()[:1] == [SHIM_MACRO])
                    or (name == "if"
                        and re.fullmatch(r"!\s*defined\s*\(?\s*%s\s*\)?" % SHIM_MACRO,
                                         rest)))
                cond_stack.append(is_shim)
                if is_shim and shim_open_at is None:
                    shim_open_at = idx
            elif name in ("else", "elif") and cond_stack:
                if cond_stack[-1] and shim_open_at is not None:
                    spans.append((shim_open_at, idx))
                    shim_open_at = None
                cond_stack[-1] = False
            elif name == "endif" and cond_stack:
                if cond_stack.pop() and shim_open_at is not None:
                    spans.append((shim_open_at, idx))
                    shim_open_at = None
        continued = line.rstrip().endswith("\\")
    if shim_open_at is not None:            # unterminated #ifndef: still a shim
        spans.append((shim_open_at, len(lines)))
    return "\n".join(code), spans


def _skip_angles(tokens, k):
    """Index just past the balanced <...> starting at or after tokens[k]."""
    while k < len(tokens) and tokens[k][0] != "<":
        if tokens[k][0] in (";", "{", "}"):
            return k
        k += 1
    depth = 0
    while k < len(tokens):
        tok = tokens[k][0]
        if tok == "<":
            depth += 1
        elif tok == ">":
            depth -= 1
            if depth == 0:
                return k + 1
        elif tok in (";", "{", "}"):
            return k
        k += 1
    return k


def scan(text):
    """Yield (line, keyword, name, in_shim) for each declaration at brace depth 0."""
    code, shim_spans = strip_directives(strip_comments_and_literals(text))
    tokens = [(m.group(0), code.count("\n", 0, m.start()) + 1)
              for m in TOKEN.finditer(code)]

    findings = []
    depth = 0
    stmt_start = True
    i = 0
    while i < len(tokens):
        tok, line = tokens[i]
        if tok == "{":
            depth += 1
            stmt_start = True
        elif tok == "}":
            depth = max(0, depth - 1)
            stmt_start = True
        elif tok == ";":
            stmt_start = True
        elif not stmt_start:
            pass
        elif tok == "namespace":
            # `namespace a::b {`, `namespace {` and the alias `namespace a = b;`
            # all end this statement; the brace, if any, is counted next pass.
            stmt_start = False
        elif tok == "template":
            # Transparent: the declaration it introduces is what gets reported.
            i = _skip_angles(tokens, i + 1)
            continue
        elif tok in TRANSPARENT:
            pass                                  # stays stmt_start
        elif tok in DECL_KEYWORDS and depth == 0:
            j = i + 1
            if tok == "enum" and j < len(tokens) and tokens[j][0] in ("class", "struct"):
                j += 1                            # `enum class Policy`
            name = ""
            while j < len(tokens) and (re.match(r"[A-Za-z_]\w*$", tokens[j][0])
                                       or tokens[j][0] == "::"):
                # Space only between two adjacent identifiers, so a qualified
                # name stays `batchlas::Queue` while a directive reads
                # `namespace batchlas` rather than `namespacebatchlas`.
                if name and name[-1].isalnum() and tokens[j][0][0].isalpha():
                    name += " "
                name += tokens[j][0]
                j += 1
                if tok != "using":                # only a using-decl is qualified
                    break
            findings.append((line, tok, name or "<unnamed>",
                             any(lo <= line <= hi for lo, hi in shim_spans)))
            stmt_start = False
        else:
            stmt_start = False
        i += 1
    return findings


def main(argv):
    root = os.path.abspath(argv[1]) if len(argv) > 1 else INCLUDE_ROOT
    if not os.path.isdir(root):
        print("check_no_global_names: no such directory: %s" % root)
        return 1

    problems = []
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            if not name.endswith(HEADER_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, REPO)
            if rel.startswith(".."):               # invoked on a tree outside the repo
                rel = os.path.relpath(path, root)
            checked += 1
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for line, keyword, decl, in_shim in scan(text):
                if in_shim:
                    # The shim is an exception for compatibility ALIASES only.
                    if keyword == "using" and decl.startswith("batchlas::"):
                        continue
                    problems.append((rel, line,
                                     "`%s %s` inside the %s shim -- that block may hold "
                                     "only `using batchlas::Name;` declarations"
                                     % (keyword, decl, SHIM_MACRO)))
                else:
                    problems.append((rel, line,
                                     "`%s %s` is declared in the consumer's global "
                                     "namespace -- move it inside `namespace batchlas` "
                                     "(a deliberate compatibility alias belongs in an "
                                     "`#ifndef %s` block)" % (keyword, decl, SHIM_MACRO)))

    for rel, line, message in sorted(problems):
        print("%s:%d: error: %s" % (rel, line, message))
    print("check_no_global_names: %d header(s) checked, %d problem(s)"
          % (checked, len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
