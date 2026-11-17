#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The device API signature page, generated from include/batchlas/device/ on every docs build.

Lists, per public header, every function declared at namespace scope in batchlas::device or a
public sub-namespace (l3, tier, cfg, ...; never detail or an anonymous namespace) with its
signature exactly as written, whitespace collapsed, and the public types the header declares.
A lexical scan, not a compiler: comments, strings and preprocessor lines are blanked, braces are
matched, and only namespace-scope declarations are read (members, bodies and lambdas are not).

Usage (gen_db_pages.py calls page(); this entry point prints it):
    python3 docs/tools/device_api_page.py [--root <repo>]
"""

import argparse
import os
import re
import sys

LABEL = "design_device_la_api"
HEADER_DIR = os.path.join("include", "batchlas", "device")
SUBDIRS = ("", "lapack", "level3")
PRIVATE_NS = ("detail", "")
TYPE_KW = re.compile(r"^(?:struct|class|union|enum(?:\s+class|\s+struct)?)\s+(?:alignas\((?:[^()]|\([^()]*\))*\)\s*)?([A-Za-z_]\w*)")
CONCEPT = re.compile(r"^concept\s+([A-Za-z_]\w*)\s*=")
NAMESPACE = re.compile(r"^(?:inline\s+)?namespace\s*([\w:]*)\s*$")
NON_FUNC = re.compile(r"^(?:using|typedef|static_assert|friend|extern\s+template|template\s*class|return)\b")


def blank(text):
    """Comments, string and char literals and preprocessor lines become spaces (newlines kept)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c, nxt = text[i], text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(c + re.sub(r"[{}();<>\\]", " ", text[i + 1:min(j, n)]) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    lines = "".join(out).split("\n")
    cont = False
    for k, line in enumerate(lines):
        if cont or line.lstrip().startswith("#"):
            cont = line.rstrip().endswith("\\")
            lines[k] = ""
    return "\n".join(lines)


def strip_template_headers(s):
    """s without its leading template <...> headers (angles matched outside parentheses)."""
    while True:
        m = re.match(r"\s*template\s*<", s)
        if not m:
            return s.strip()
        depth, paren, i = 1, 0, m.end()
        while i < len(s) and depth:
            ch = s[i]
            paren += ch == "("
            paren -= ch == ")"
            if not paren:
                depth += ch == "<"
                depth -= ch == ">"
            i += 1
        s = s[i:]
        s = re.sub(r"^\s*requires\b.*?(?=\b(?:BATCHLAS_|constexpr|inline|static|struct|class|union|enum|concept|[A-Za-z_][\w:<>]*\s+[A-Za-z_]\w*\s*\())",
                   "", s, count=1, flags=re.S)


def func_name(decl):
    """The declared name of a function declaration (text up to its parameter list), else None."""
    depth, i = 0, 0
    while i < len(decl):
        ch = decl[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth <= 0:
            head = decl[:i].rstrip()
            if "=" in re.sub(r"operator\s*\S+$", "", head):
                return None
            m = re.search(r"(operator\s*(?:\(\)|[^\s(]+)|[A-Za-z_]\w*)$", head)
            if not m or m.group(1) in ("if", "for", "while", "switch", "sizeof", "decltype", "alignas", "noexcept"):
                return None
            if not re.search(r"[\w>&*]\s+" + re.escape(m.group(1)) + r"$", head) and not head.startswith(("auto", "constexpr")):
                return None
            return m.group(1)
        i += 1
    return None


def scan(text):
    """[(namespace path, kind, name, signature)] for every namespace-scope declaration."""
    src = blank(text)
    stack, found, buf, paren = [], [], [], 0
    for ch in src:
        at_ns = all(kind == "ns" for kind, _ in stack)
        if not at_ns:
            if ch == "{":
                stack.append(("skip", None))
            elif ch == "}":
                stack.pop()
                if all(kind == "ns" for kind, _ in stack):
                    buf = []
            continue
        if ch in "()":
            paren += 1 if ch == "(" else -1
            buf.append(ch)
            continue
        if paren > 0 or ch not in "{};":
            buf.append(ch)
            continue
        stmt = " ".join("".join(buf).split())
        buf = []
        ns = [part for kind, parts in stack for part in parts]
        if ch == "}":
            if stack:
                stack.pop()
            continue
        m = NAMESPACE.match(stmt)
        if ch == "{" and m:
            stack.append(("ns", m.group(1).split("::") if m.group(1) else [""]))
            continue
        body = strip_template_headers(stmt)
        tm, cm = TYPE_KW.match(body), CONCEPT.match(body)
        if tm:
            if ch == "{" and not re.match(r"\s*<", body[tm.end():]):
                found.append((ns, "type", tm.group(1), stmt))
        elif cm:
            found.append((ns, "type", cm.group(1), stmt))
        elif body and not NON_FUNC.match(body):
            name = func_name(body)
            if name:
                found.append((ns, "func", name, stmt))
        if ch == "{":
            stack.append(("skip", None))
    return found


def public_ns(ns):
    return ns[:2] == ["batchlas", "device"] and not any(part in PRIVATE_NS for part in ns[2:])


def headers(root):
    for sub in SUBDIRS:
        d = os.path.join(root, HEADER_DIR, sub)
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else ():
            if name.endswith(".hh"):
                yield os.path.join(sub, name) if sub else name, os.path.join(d, name)


def summary(text):
    """The header's first comment sentence (after SPDX and #pragma)."""
    lines = []
    for line in text.splitlines()[:12]:
        s = line.strip()
        if s.startswith("//") and "SPDX" not in s:
            lines.append(s.lstrip("/ ").strip())
        elif lines and not s.startswith("//"):
            break
    first = " ".join(lines)
    m = re.match(r"(.+?[.:])(\s|$)", first)
    return (m.group(1) if m else first).rstrip(":") if first else ""


def page(root):
    out = ["# Device LA: API signatures {#%s}" % LABEL, "",
           "> **Status:** generated · `docs/tools/device_api_page.py` on every docs build; do not edit.", "",
           "Every public function of `batchlas::device` with its signature as the header declares it, by header "
           "(`include/batchlas/device/`). Conventions, tiers and the launch frame: @ref design_device_la. "
           "`detail` and anonymous namespaces, members and specialisations are left out; a sub-namespace "
           "(`l3`, `tier`, ...) is named in a comment above its first function.", ""]
    total = 0
    for rel, path in headers(root):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        items = [x for x in scan(text) if public_ns(x[0])]
        funcs = [x for x in items if x[1] == "func"]
        types = sorted({"::".join(x[0][2:] + [x[2]]) for x in items if x[1] == "type"})
        if not funcs and not types:
            continue
        out += ["## %s {#device-api-%s}" % (rel, re.sub(r"[^a-z0-9]+", "-", rel[:-3].lower())), ""]
        if summary(text):
            out += [summary(text), ""]
        if types:
            out += ["Types: " + ", ".join("`%s`" % t for t in types) + ".", ""]
        if funcs:
            out.append("```cpp")
            last_ns, seen = ["batchlas", "device"], set()
            for ns, _, _, sig in funcs:
                if sig in seen:
                    continue
                seen.add(sig)
                if ns != last_ns:
                    out.append("// namespace %s" % "::".join(ns))
                last_ns = ns
                cut = len(sig) - len(strip_template_headers(sig))
                out += [s for s in (sig[:cut].strip(), sig[cut:].strip() + ";") if s]
                total += 1
            out += ["```", ""]
    out.insert(6, "%d signatures." % total)
    out.insert(7, "")
    return "\n".join(out) + "\n"


def self_test():
    fixture = """// SPDX-License-Identifier: MIT
#pragma once
// Fixture: one public op, one detail helper.
#define X(a) { a }
namespace batchlas::device {
namespace detail { template <class T> int hidden(T) { return 0; } }
template <class T> struct Opts { int k = 1; };
template <Uplo UL, class Tier, class Ex, class MA>
BATCHLAS_DEVICE_INLINE int potrf(const Ex& ex, const MA& a, const Workspace& ws = {}, bool live = true) {
    auto f = [&](int i) { return "}"; };
    return 0;
}
template <class T> inline constexpr bool flag_v = sizeof(T) > 4;
namespace l3 { template <class C> concept Config = true; BATCHLAS_DEVICE_INLINE void rank_k(int k); }
}
"""
    found = [(".".join(ns), kind, name) for ns, kind, name, _ in scan(fixture) if public_ns(ns)]
    want = [("batchlas.device", "type", "Opts"), ("batchlas.device", "func", "potrf"),
            ("batchlas.device.l3", "type", "Config"), ("batchlas.device.l3", "func", "rank_k")]
    sig = [s for ns, k, n, s in scan(fixture) if n == "potrf"]
    ok = found == want and sig and sig[0].endswith("const Workspace& ws = {}, bool live = true)")
    return ok, "device api scan %s, got %s" % ("ok" if ok else "FAILED", found)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        ok, msg = self_test()
        print(msg)
        return 0 if ok else 1
    sys.stdout.write(page(args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
