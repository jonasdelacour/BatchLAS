#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The device API signature index, generated from include/batchlas/device/ on every docs build.

Lists every function declared at namespace scope in batchlas::device or a public sub-namespace
(l3, tier, cfg, ...; never detail or an anonymous namespace), and the public types, aliases and
forward declarations, grouped by the api_device child groups (docs/pages/api_groups.dox). Each
signature is wrapped to WIDTH columns and followed by its brief: the @brief of its doc comment,
else the first sentence of the plain comment above it. api_device_lowlevel members come last, in
a section marked not covered. The output is a Doxygen group, child of api_device, so the site
navigation lists it under the API reference; its name is the page's old label.

A lexical scan, not a compiler: comments, strings and preprocessor lines are blanked, braces are
matched, and only namespace-scope declarations are read (members, bodies and lambdas are not).
A header's group is its own @addtogroup / @ingroup tag, else the group in api_groups.dox that
names it, else FALLBACK.

Usage (gen_db_pages.py calls page(); this entry point prints it):
    python3 docs/tools/device_api_page.py [--root <repo>] [--self-test]
"""

import argparse
import os
import re
import sys
import textwrap

LABEL = "design_device_la_api"
TITLE = "All signatures"
PARENT = "api_device"
LOWLEVEL = "api_device_lowlevel"
GROUPS_DOX = os.path.join("docs", "pages", "api_groups.dox")
HEADER_DIR = os.path.join("include", "batchlas", "device")
SUBDIRS = ("", "lapack", "level3")
UMBRELLAS = ("device", "lapack")
PRIVATE_NS = ("detail", "")
WIDTH = 80
FALLBACK = {
    "level3/cfg.hh": "api_device_blas3",
    "lapack/laed3.hh": LOWLEVEL, "lapack/laed4.hh": LOWLEVEL,
    "lapack/lagtf.hh": LOWLEVEL, "lapack/lagts.hh": LOWLEVEL,
    "level3/copy_async.hh": LOWLEVEL, "level3/epilogue.hh": LOWLEVEL, "level3/mma.hh": LOWLEVEL,
    "level3/packet.hh": LOWLEVEL, "level3/pipeline.hh": LOWLEVEL, "level3/schedule.hh": LOWLEVEL,
    "level3/stage.hh": LOWLEVEL,
}
TYPE_KW = re.compile(r"^(?:struct|class|union|enum(?:\s+class|\s+struct)?)\s+(?:alignas\((?:[^()]|\([^()]*\))*\)\s*)?([A-Za-z_]\w*)")
CONCEPT = re.compile(r"^concept\s+([A-Za-z_]\w*)\s*=")
ALIAS = re.compile(r"^using\s+([A-Za-z_]\w*)\s*=")
NAMESPACE = re.compile(r"^(?:inline\s+)?namespace\s*([\w:]*)\s*$")
NON_FUNC = re.compile(r"^(?:using|typedef|static_assert|friend|extern\s+template|template\s*class|return)\b")
MACRO = re.compile(r"\bBATCHLAS_[A-Z0-9_]+\b(?:\s*\((?:[^()]|\([^()]*\))*\))?\s*")
VENDOR_ATTR = re.compile(r"\[\[(?:(?:gnu|clang|intel|sycl)::[^\]]*|maybe_unused)\]\]\s*")
TEMPLATE_ID = r"[\w:]+<[^<>]*(?:<[^<>]*>[^<>]*)*>(?:\([^()]*\))?"
CMD = r"[@\\]"
ABBREV = re.compile(r"\b(?:e\.g|i\.e|cf|vs|etc|resp|approx)\.$")


def blank(text):
    """Comments, string and char literals and preprocessor lines become spaces; offsets kept."""
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
            out.append(c + re.sub(r"[{}();<>\\\n]", " ", text[i + 1:min(j, n)]) + (c if j < n else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    lines = "".join(out).split("\n")
    cont = False
    for k, line in enumerate(lines):
        if cont or line.lstrip().startswith("#"):
            cont = line.rstrip().endswith("\\")
            lines[k] = " " * len(line)
    return "\n".join(lines)


def template_cut(s):
    """Length of s's leading template <...> headers and requires clause (angles outside parens)."""
    pos = 0
    while True:
        m = re.match(r"\s*template\s*<", s[pos:])
        if not m:
            break
        depth, paren, i = 1, 0, pos + m.end()
        while i < len(s) and depth:
            ch = s[i]
            paren += ch == "("
            paren -= ch == ")"
            if not paren:
                depth += ch == "<"
                depth -= ch == ">"
            i += 1
        pos = i
        m = re.match(r"\s*requires\b.*?(?=\b(?:BATCHLAS_|constexpr|inline|static|struct|class|union|enum|concept|using|[A-Za-z_][\w:<>]*\s+[A-Za-z_]\w*\s*\())",
                     s[pos:], flags=re.S)
        if m:
            pos += m.end()
    return pos


def strip_template_headers(s):
    return s[template_cut(s):].strip()


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
    """[(namespace path, kind, name, signature, offset)] for every namespace-scope declaration."""
    src = blank(text)
    stack, found, buf, paren, start = [], [], [], 0, None
    for pos, ch in enumerate(src):
        at_ns = all(kind == "ns" for kind, _ in stack)
        if not at_ns:
            if ch == "{":
                stack.append(("skip", None))
            elif ch == "}":
                stack.pop()
                if all(kind == "ns" for kind, _ in stack):
                    buf, start = [], None
            continue
        if start is None and not ch.isspace():
            start = pos
        if ch in "()":
            paren += 1 if ch == "(" else -1
            buf.append(ch)
            continue
        if paren > 0 or ch not in "{};":
            buf.append(ch)
            continue
        stmt = " ".join("".join(buf).split())
        at, buf, start = start, [], None
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
        tm, cm, am = TYPE_KW.match(body), CONCEPT.match(body), ALIAS.match(body)
        if tm:
            rest = body[tm.end():]
            if (ch == "{" and not re.match(r"\s*<", rest)) or (ch == ";" and not rest.strip()):
                found.append((ns, "type", tm.group(1), stmt, at))
        elif cm:
            found.append((ns, "type", cm.group(1), stmt, at))
        elif am and ch == ";":
            found.append((ns, "type", am.group(1), stmt, at))
        elif body and not NON_FUNC.match(body):
            name = func_name(body)
            if name:
                found.append((ns, "func", name, stmt, at))
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


# ------------------------------------------------------------------ comments -

def comment_before(text, at):
    """(is_doc, lines) of the comment block that ends on the line above offset at."""
    line_start = text.rfind("\n", 0, at) + 1
    if text[line_start:at].strip():
        return False, []
    lines = text[:line_start].split("\n")[:-1]
    got, doc = [], None
    k = len(lines) - 1
    if k >= 0 and lines[k].rstrip().endswith("*/") and not lines[k].lstrip().startswith("//"):
        j = k
        while j >= 0 and "/*" not in lines[j]:
            j -= 1
        if j < 0:
            return False, []
        block = "\n".join(lines[j:k + 1])
        block = block[block.find("/*"):]
        is_doc = block.startswith(("/**", "/*!"))
        body = re.sub(r"^/\*[*!]?|\*/$", "", block.strip())
        return is_doc, [re.sub(r"^\s*\*?\s?", "", ln) for ln in body.split("\n")]
    while k >= 0:
        s = lines[k].strip()
        if not s.startswith("//") or s.startswith("///<"):
            break
        this_doc = s.startswith(("///", "//!")) and not s.startswith("////")
        if doc is not None and this_doc != doc:
            break
        doc = this_doc
        got.append(re.sub(r"^//[/!]?\s?", "", s))
        k -= 1
    return bool(doc), got[::-1]


def head_comment(text):
    """(is_doc, lines) of the header's first comment block (after SPDX, #pragma and blanks)."""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and (not lines[k].strip() or "SPDX" in lines[k] or lines[k].startswith("#pragma")):
        k += 1
    if k >= len(lines) or not lines[k].lstrip().startswith(("//", "/*")):
        return False, []
    j = k
    if lines[k].lstrip().startswith("/*"):
        while j < len(lines) and "*/" not in lines[j]:
            j += 1
    else:
        while j + 1 < len(lines) and lines[j + 1].lstrip().startswith("//"):
            j += 1
    offset = sum(len(ln) + 1 for ln in lines[:j + 1])
    is_doc, got = comment_before(text + "\n", offset)
    return is_doc, [ln for ln in got if "SPDX-License-Identifier" not in ln]


def sentence(text):
    """The first sentence of text: up to a '.' that ends a word and starts a new one."""
    depth, tick = 0, False
    for i, ch in enumerate(text):
        if ch == "`":
            tick = not tick
        elif not tick and ch in "([":
            depth += 1
        elif not tick and ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "." and not tick and not depth and text[i - 1:i] != "." and (i + 1 == len(text) or text[i + 1] == " "):
            if not ABBREV.search(text[:i + 1]):
                return text[:i + 1]
    return text.rstrip(" :;,") + ("." if text and not text.rstrip().endswith((".", "!", "?")) else "")


def paragraph(lines):
    """The first paragraph of plain comment lines: stops at a blank line or an indented list."""
    out = []
    for ln in lines:
        if not ln.strip() or (out and re.match(r"\s{2,}\S", ln)):
            break
        out.append(ln.strip())
    return " ".join(out)


def doc_brief(lines):
    """The brief of Doxygen comment lines: the @brief paragraph, else the first sentence."""
    segments, cur = [], []
    for ln in lines:
        if re.match(r"\s*" + CMD + r"[{}]\s*$", ln):
            segments.append(cur)
            cur = []
        else:
            cur.append(ln)
    texty = [seg for seg in segments + [cur]
             if any(s.strip() and not re.match(r"\s*" + CMD + r"(?:addtogroup|defgroup|ingroup|name|file)\b", s)
                    for s in seg)]
    lines = texty[-1] if texty else []
    for k, ln in enumerate(lines):
        m = re.search(CMD + r"(?:brief|short)\s+(.*)", ln)
        if m:
            out = [m.group(1).strip()]
            for nxt in lines[k + 1:]:
                if not nxt.strip() or re.match(r"\s*" + CMD + r"\w", nxt):
                    break
                out.append(nxt.strip())
            return " ".join(out).strip()
    keep = []
    for ln in lines:
        s = re.sub(CMD + r"file(?:\s+[\w./-]+\.\w+)?", "", ln)
        if re.match(r"\s*" + CMD + r"(?!f[$\[]|ref\b|p\b|c\b|a\b|e\b|b\b)\w+", s):
            if keep:
                break
            continue
        if not s.strip():
            if keep:
                break
            continue
        keep.append(s.strip())
    return sentence(" ".join(keep)) if keep else ""


def plain_brief(lines):
    return sentence(paragraph(lines)) if lines else ""


def tags(lines, command):
    return re.findall(CMD + command + r"\s+(\w+)", "\n".join(lines))


def group_ranges(text):
    """[(begin, end, group)] of the @addtogroup / @defgroup @{ ... @} ranges in doc comments."""
    ranges, stack, pending = [], [], None
    doc = re.compile(r"///[^\n]*|//![^\n]*|/\*[*!].*?\*/", re.S)
    token = re.compile(CMD + r"(addtogroup|defgroup|weakgroup|name)\b\s*(\w*)|" + CMD + r"([{}])")
    for m in doc.finditer(text):
        if m.group(0).startswith("///<"):
            continue
        for t in token.finditer(m.group(0)):
            if t.group(1):
                pending = t.group(2) if t.group(1) != "name" else None
            elif t.group(3) == "{":
                stack.append((m.start() + t.start(), pending))
                pending = None
            elif stack:
                begin, grp = stack.pop()
                if grp:
                    ranges.append((begin, m.start() + t.start(), grp))
    ranges += [(begin, len(text), grp) for begin, grp in stack if grp]
    return ranges


def to_text(brief, doc):
    """A brief made safe for a group page: links dropped (scope differs), plain text escaped."""
    if doc:
        if re.search(CMD + r"copy(?:brief|doc|details)\b", brief):
            return ""
        brief = re.sub(CMD + r"ref\s+([\w:]+)(?:\s+\"([^\"]*)\")?", lambda m: m.group(2) or "`%s`" % m.group(1), brief)
        brief = re.sub(CMD + r"(?:p|c)\s+([\w:]+(?:\.\w+)*)", r"`\1`", brief)
        brief = re.sub(CMD + r"(?:a|e)\s+(\w+)", r"*\1*", brief)
        brief = re.sub(CMD + r"(?:link|endlink)\b\s*", "", brief)
        brief = re.sub(r"(^|\s)#(\w+)", r"\1`\2`", brief)
    else:
        if brief.count("`") % 2:
            brief = brief.replace("`", "'")
        parts = brief.split("`")
        for k in range(0, len(parts), 2):
            parts[k] = re.sub(TEMPLATE_ID + r"|\S*[<*\\@#%&|]\S*", lambda m: code_token(m.group(0)), parts[k])
        brief = "`".join(parts)
    return brief.replace("*/", "* /").strip()


def code_token(tok):
    """A plain-comment word with markup characters, as a code span (its trailing punctuation outside)."""
    lead = re.match(r"[(\"']*", tok).group(0)
    core, trail = tok[len(lead):], ""
    while core and (core[-1] in "\"'.,;:" or (core[-1] == ")" and core.count(")") > core.count("("))):
        core, trail = core[:-1], core[-1] + trail
    return "%s`%s`%s" % (lead, core, trail) if core else tok


# ------------------------------------------------------------------- groups -

def child_groups(root):
    """[(name, title, header stems)] of the api_device child groups, in api_groups.dox order."""
    path = os.path.join(root, GROUPS_DOX)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = []
    for block in re.findall(r"/\*\*(.*?)\*/", text, re.S):
        m = re.search(CMD + r"defgroup\s+(\w+)\s+(.*)", block)
        if not m or not re.search(CMD + r"ingroup\s+" + PARENT + r"\b", block):
            continue
        stems = [os.path.basename(h)[:-3] for h in re.findall(r"([\w/]+\.hh)", block)]
        if ":" in m.group(2):
            stems += [w.strip() for w in m.group(2).split(":", 1)[1].split(",")]
        out.append((m.group(1), m.group(2).strip(), [s for s in stems if s not in UMBRELLAS]))
    return out


def header_group(rel, text, groups):
    """The group a header's declarations default to, or None."""
    _, head = head_comment(text)
    tagged = tags(head, "addtogroup") + tags(head, "ingroup")
    if tagged:
        return tagged[0]
    stem = os.path.basename(rel)[:-3]
    for name, _, stems in groups:
        if stem in stems:
            return name
    return FALLBACK.get(rel.replace(os.sep, "/"))


# ------------------------------------------------------------------- layout -

def pairs_of(s, angles):
    """[(open, close)] of s's top-level bracket pairs, or None when they do not balance."""
    out, stack = [], []
    for i, ch in enumerate(s):
        if ch == "<":
            if angles and s[i + 1:i + 2] not in ("<", "=") and s[i - 1:i] != "<":
                stack.append(("<", i))
            continue
        if ch == ">":
            if not angles or s[i - 1:i] == "-" or s[i + 1:i + 2] == "=" or not stack or stack[-1][0] != "<":
                continue
        elif ch in "([{":
            stack.append((ch, i))
            continue
        elif ch in ")]}":
            if not stack or stack[-1][0] != "([{"[")]}".index(ch)]:
                return None
        else:
            continue
        _, i0 = stack.pop()
        if not stack:
            out.append((i0, i))
    return None if stack else out


def groups_of(s):
    """Top-level bracket pairs; angle brackets count only where they balance."""
    got = pairs_of(s, True)
    return got if got is not None else (pairs_of(s, False) or [])


def split_args(s):
    """s split at its top-level commas."""
    pairs, parts, last = groups_of(s), [], 0
    for i, ch in enumerate(s):
        if ch == "," and not any(a < i < b for a, b in pairs):
            parts.append(s[last:i].strip())
            last = i + 1
    return parts + [s[last:].strip()]


def wrap(s, indent="", width=WIDTH):
    """s as lines no wider than width where it can be: the widest bracket list one item per line."""
    if len(indent) + len(s) <= width:
        return [indent + s]
    pairs = [(a, b) for a, b in groups_of(s) if b - a > 1]
    if not pairs:
        return [indent + s]
    a, b = max(pairs, key=lambda p: p[1] - p[0])
    head, args, tail = s[:a + 1], split_args(s[a + 1:b]), s[b:]
    col = len(indent) + len(head)
    out = []
    if len(args) == 1:
        return [indent + head] + wrap(args[0] + tail, indent + "    ", width)
    if col <= 40:
        pad = " " * col
        for k, arg in enumerate(args):
            lines = wrap(arg + ("," if k < len(args) - 1 else tail), pad, width)
            if k == 0:
                lines[0] = indent + head + lines[0][col:]
            out += lines
    else:
        out.append(indent + head)
        for k, arg in enumerate(args):
            out += wrap(arg + ("," if k < len(args) - 1 else ""), indent + "    ", width)
        out.append(indent + tail.lstrip())
    return out


def clean(sig):
    return " ".join(VENDOR_ATTR.sub("", MACRO.sub("", sig)).split())


def format_sig(sig):
    """The signature's lines: each template header, the requires clause and the declaration."""
    sig = clean(sig)
    cut = template_cut(sig)
    tmpl, decl, lines = sig[:cut].strip(), sig[cut:].strip(), []
    while tmpl:
        m = re.match(r"template\s*<", tmpl)
        if not m:
            lines += wrap(tmpl, "    ")
            break
        depth, paren, i = 1, 0, m.end()
        while i < len(tmpl) and depth:
            paren += tmpl[i] == "("
            paren -= tmpl[i] == ")"
            if not paren:
                depth += tmpl[i] == "<"
                depth -= tmpl[i] == ">"
            i += 1
        lines += wrap(tmpl[:i])
        tmpl = tmpl[i:].strip()
    return [ln.replace("*/", "* /") for ln in lines + wrap(decl + ";")]


# --------------------------------------------------------------------- page -

def entries(root, groups):
    """{group: [(rel, summary, [entry])]} in header order; entry = (kind, name, ns, lines, brief)."""
    by_group = {}
    for rel, path in headers(root):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        items = [x for x in scan(text) if public_ns(x[0])]
        if not items:
            continue
        is_doc, head = head_comment(text)
        summary = to_text(doc_brief(head) if is_doc else plain_brief(head), is_doc)
        default, ranges = header_group(rel, text, groups), group_ranges(text)
        per, seen = {}, set()
        for ns, kind, name, sig, at in items:
            key = (kind, name) if kind == "type" else clean(sig)
            if key in seen:
                continue
            seen.add(key)
            doc, lines = comment_before(text, at)
            brief = to_text(doc_brief(lines) if doc else plain_brief(lines), doc)
            own = tags(lines, "ingroup") if doc else []
            inside = [g for b, e, g in ranges if b <= at < e]
            grp = own[0] if own else inside[-1] if inside else default
            per.setdefault(grp, []).append((kind, name, ns, format_sig(sig) if kind == "func" else sig, brief))
        for grp, items_ in per.items():
            by_group.setdefault(grp, []).append((rel, summary, items_))
    return by_group


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def render_header(rel, summary, items, ids, suffix):
    base = "device-api-" + slug(rel[:-3])
    hid = base if base not in ids else base + "-" + suffix
    ids.add(hid)
    out = ["@subsection %s %s" % (hid, rel), ""]
    if summary:
        out += [summary, ""]
    types = [x for x in items if x[0] == "type"]
    funcs = [x for x in items if x[0] == "func"]
    if types:
        out.append("Types:")
        for _, name, ns, _, brief in types:
            out.append("- %s%s" % ("::".join(ns + [name]), ": " + brief if brief else ""))
        out.append("")
    k = 0
    while k < len(funcs):
        run = [funcs[k]]
        while k + len(run) < len(funcs) and funcs[k + len(run)][1:3] == run[0][1:3] \
                and not funcs[k + len(run)][4]:
            run.append(funcs[k + len(run)])
        k += len(run)
        _, name, ns, _, brief = run[0]
        out.append("```cpp")
        for j, item in enumerate(run):
            out += ([""] if j else []) + item[3]
        out += ["```", ""]
        link = "" if name.startswith("operator") else "::".join(ns + [name]) + "()"
        text = brief + (" &rarr; " + link if brief and link else link)
        if text:
            out += [text, ""]
    return out, len(funcs)


def page(root):
    groups = child_groups(root)
    known = {name: title for name, title, _ in groups}
    by_group = entries(root, groups)
    order = [g for g, _, _ in groups if g != LOWLEVEL and g in by_group]
    order += sorted(g for g in by_group if g not in known and g not in (LOWLEVEL, None))
    order += [g for g in (None, LOWLEVEL) if g in by_group]
    body, ids, total, low = [], set(), 0, 0
    for grp in order:
        short = (grp or "unplaced").replace(PARENT + "_", "")
        if grp == LOWLEVEL:
            body += ["@section device-api-group-%s Not covered: low-level helpers" % slug(short), "",
                     "These live in `batchlas::device` because public routines use them. They are **not** "
                     "covered by the compatibility promise and may change or disappear in any release; "
                     "they are listed so the signatures above can be read. Group page: %s." %
                     ("@ref " + grp if grp in known else "`%s`" % grp), ""]
        elif grp is None:
            body += ["@section device-api-group-unplaced Headers in no group", "",
                     "These headers name no api_device child group and none names them; until one does, "
                     "they are not covered.", ""]
        else:
            body += ["@section device-api-group-%s %s" % (slug(short), known.get(grp, grp)), "",
                     "Group page: %s." % ("@ref " + grp if grp in known else "`%s`" % grp), ""]
        for rel, summary, items in by_group[grp]:
            lines, n = render_header(rel, summary, items, ids, slug(short))
            body += lines
            total += n
            low += n if grp == LOWLEVEL else 0
    head = ["/**", "@defgroup %s %s" % (LABEL, TITLE), "@ingroup %s" % PARENT,
            "@brief Every public signature of `batchlas::device`, by group and header, each with its brief.", "",
            "> **Status:** generated · `docs/tools/device_api_page.py` on every docs build; do not edit.", "",
            "Generated from `include/batchlas/device/`. Each signature is wrapped to %d columns and "
            "followed by its brief, from its doc comment or, where it has none, from the comment above it; "
            "the qualified name after the brief links to the member's documentation when Doxygen has it. "
            "Signatures are grouped by the child groups of @ref %s, then by header. `detail` and anonymous "
            "namespaces, class members, specialisations and variable templates are left out, and so are "
            "the export and inlining macros. Conventions, tiers and the launch frame: @ref guide_device_api."
            % (WIDTH, PARENT), "",
            "%d signatures; %d of them are low-level helpers, listed last and not covered by the "
            "compatibility promise." % (total, low), ""]
    return "\n".join(reflow(head + body) + ["*/"]) + "\n"


def reflow(lines, width=100):
    """Prose lines filled to width (code spans kept whole); fences, commands and headings as is."""
    out, fence = [], False
    for ln in lines:
        if ln.startswith("```"):
            fence = not fence
        if fence or ln.startswith(("```", "@", "/**")) or len(ln) <= width:
            out.append(ln)
            continue
        glued = re.sub(r"`[^`]*`", lambda m: m.group(0).replace(" ", "\0"), ln)
        sub = "  " if ln.startswith("- ") else ""
        out += [x.replace("\0", " ") for x in textwrap.wrap(glued, width, subsequent_indent=sub,
                                                            break_long_words=False, break_on_hyphens=False)]
    return out


# ---------------------------------------------------------------- self-test -

FIXTURE = """// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Fixture ops: one public op, e.g. potrf. A detail helper.
/// @addtogroup api_device_lapack_chol
/// @{
#define X(a) { a }
namespace batchlas::device {
namespace detail { template <class T> int hidden(T) { return 0; } }
/// Options of the fixture (a struct).
template <class T> struct Opts { int k = 1; };
template <class T> struct tier_info;
template <class T> using opts_t = Opts<T>;
/// @brief Cholesky of one matrix: the factor
/// overwrites a.
/// @param ex the executor
template <Uplo UL, class Tier, class Ex, class MA>
BATCHLAS_DEVICE_INLINE int potrf(const Ex& ex, const MA& a, const Workspace& ws = {}, bool live = true, int a_very_long_parameter_name = 0) {
    auto f = [&](int i) { return "}"; };
    return 0;
}
template <class T> inline constexpr bool flag_v = sizeof(T) > 4;
/// @}
// Plain: the panel <leaf> of potrf. Kept for the host.
/// @ingroup api_device_lowlevel
BATCHLAS_DEVICE_HELPER void potf2(int n);
namespace l3 { template <class C> concept Config = true; BATCHLAS_DEVICE_INLINE void rank_k(int k); }
}
"""


def self_test():
    found = [(".".join(ns), kind, name) for ns, kind, name, _, _ in scan(FIXTURE) if public_ns(ns)]
    want = [("batchlas.device", "type", "Opts"), ("batchlas.device", "type", "tier_info"),
            ("batchlas.device", "type", "opts_t"), ("batchlas.device", "func", "potrf"),
            ("batchlas.device", "func", "potf2"),
            ("batchlas.device.l3", "type", "Config"), ("batchlas.device.l3", "func", "rank_k")]
    sigs = {n: (s, at) for _, _, n, s, at in scan(FIXTURE)}
    potrf_sig, potrf_at = sigs["potrf"]
    doc, lines = comment_before(FIXTURE, potrf_at)
    lines_out = format_sig(potrf_sig)
    is_doc, head = head_comment(FIXTURE)
    checks = {
        "scan": found == want,
        "signature": potrf_sig.endswith("bool live = true, int a_very_long_parameter_name = 0)"),
        "brief": doc and doc_brief(lines) == "Cholesky of one matrix: the factor overwrites a.",
        "file brief": is_doc and doc_brief(head) == "Fixture ops: one public op, e.g. potrf. A detail helper.",
        "plain head": plain_brief(["Device LA executors. The size of", "every executor."]) == "Device LA executors.",
        "colon head": plain_brief(["sb2st: band -> tridiagonal, on storage:", "  a(x)   the op"]) == "sb2st: band -> tridiagonal, on storage.",
        "ingroup": tags(comment_before(FIXTURE, sigs["potf2"][1])[1], "ingroup") == ["api_device_lowlevel"],
        "range": [g for b, e, g in group_ranges(FIXTURE) if b <= potrf_at < e] == ["api_device_lapack_chol"]
                 and not [g for b, e, g in group_ranges(FIXTURE) if b <= sigs["rank_k"][1] < e],
        "macro": not any("BATCHLAS_" in ln for ln in lines_out),
        "wrap": len(lines_out) > 3 and max(len(ln) for ln in lines_out) <= WIDTH
                and lines_out[1].startswith("int potrf(const Ex& ex,"),
        "escape": to_text("the panel <leaf> of a_b* (x@y).", False) == "the panel `<leaf>` of `a_b*` (`x@y`).",
    }
    bad = [k for k, ok in checks.items() if not ok]
    return not bad, "device api scan %s%s" % ("ok" if not bad else "FAILED: ", ", ".join(bad))


def check_page(root):
    """Problems of the page generated from root's headers: wide code lines, leaked commands, no group."""
    text, bad, fence = page(root), [], False
    for ln in text.split("\n"):
        if ln.startswith("```"):
            fence = not fence
        elif fence and len(ln) > WIDTH + 10:
            bad.append("code line of %d columns: %s" % (len(ln), ln[:60]))
        elif not fence and not ln.startswith("@") and re.search(CMD + r"(?:file|brief|addtogroup|ingroup|param|tparam|return|retval)\b", ln):
            bad.append("doc command in the text: %s" % ln[:60])
    if "@section device-api-group-unplaced " in text:
        bad.append("a header in no api_device child group")
    groups = child_groups(root)
    if groups:
        known = {name for name, _, _ in groups}
        bad += ["group %s is not defined in %s" % (g, GROUPS_DOX) for g in entries(root, groups) if g and g not in known]
    if not re.search(r"^\d+ signatures;", text, re.M) or text.count("```cpp") < 100:
        bad.append("fewer than 100 signature blocks")
    return bad


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        ok, msg = self_test()
        problems = check_page(args.root)
        print(msg + "".join("\n  " + p for p in problems))
        return 0 if ok and not problems else 1
    sys.stdout.write(page(args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
