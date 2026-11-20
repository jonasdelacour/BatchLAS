#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Print one version's section of docs/changelog.md as GitHub release notes.

A release is cut by pushing a tag vX.Y.Z (.github/workflows/release.yml). Before
anything is published this checks that the tag, project(BatchLAS VERSION ...) in
CMakeLists.txt, BATCHLAS_VERSION_STRING in include/batchlas/version.hh and a
dated changelog heading agree.

Keep a Changelog headings are read as `## [1.0.0] - 2026-11-30`, with or without
the brackets, with -, en or em dash, an optional trailing Doxygen id `{#...}`
and an optional `[YANKED]` (refused). `## [Unreleased]` is never a release. The
section runs to the next level-2 heading outside a code fence. The heading is
not printed (the release title carries it); `{#...}` ids are stripped from the
subheadings; link reference definitions the section uses are appended; with
--link-base, relative links are made absolute against <base>/docs/.

USAGE
    python3 .github/ci/release_notes.py v1.0.0                 # notes on stdout
    python3 .github/ci/release_notes.py v1.0.0 --check         # agreement only
    python3 .github/ci/release_notes.py v1.0.0 --link-base https://github.com/o/r/blob/v1.0.0
    python3 .github/ci/release_notes.py --self-test            # fixture under .github/ci/fixtures
Exit 0 = ok, 1 = a disagreement or a missing, undated, yanked or empty section.
"""
import argparse
import datetime
import os
import posixpath
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(HERE, "fixtures", "release_notes")

HEADING = re.compile(
    r"^##[ \t]+\[?v?(?P<ver>[^\]\s]+)\]?"
    r"(?:[ \t]+[-" "\u2013\u2014" r"][ \t]+(?P<date>\S+))?"
    r"(?P<rest>.*?)[ \t]*$")
DOX_ID = re.compile(r"[ \t]*\{#[A-Za-z0-9_-]+\}[ \t]*$")
LINK_DEF = re.compile(r"^ {0,3}\[(?P<label>[^\]]+)\]:[ \t]*(?P<url>\S+)")
FENCE = re.compile(r"^ {0,3}(```|~~~)")
INLINE_LINK = re.compile(r"(\]\()(?P<target>[^)\s]+)")
CMAKE_VERSION = re.compile(r"^project\(BatchLAS[^)]*VERSION\s+([0-9][0-9.]*)", re.M)
HEADER_VERSION = re.compile(r'^#define BATCHLAS_VERSION_STRING "([^"]*)"', re.M)


def parse(text):
    """Return ({version: (date, rest, body_lines)}, {label_lower: (label, url)})."""
    sections, defs, current, fenced = {}, {}, None, False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            d = LINK_DEF.match(line)
            if d:
                defs[d.group("label").lower()] = (d.group("label"), d.group("url"))
                continue
            m = HEADING.match(line) if line.startswith("## ") else None
            if m:
                current = m.group("ver")
                sections[current] = (m.group("date"), m.group("rest"), [])
                continue
            if line.startswith("#"):
                line = DOX_ID.sub("", line)
        if current is not None:
            sections[current][2].append(line)
    return sections, defs


def absolute(target, link_base):
    """A link relative to docs/changelog.md, made absolute; anything else unchanged."""
    if not link_base or re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target, re.I):
        return target
    return link_base.rstrip("/") + "/" + posixpath.normpath("docs/" + target)


def notes(text, version, link_base=None):
    """Return (errors, body) for `## [version]` of a changelog text."""
    sections, defs = parse(text)
    if version not in sections:
        return [f"release_notes: docs/changelog.md has no '## [{version}]' section"], ""
    date, rest, lines = sections[version]
    errors = []
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
            raise ValueError
        datetime.date.fromisoformat(date)
    except ValueError:
        errors.append(f"release_notes: '## [{version}]' is not dated YYYY-MM-DD (got {date!r})")
    if "yanked" in rest.lower():
        errors.append(f"release_notes: '## [{version}]' is marked [YANKED]")
    body = "\n".join(lines).strip()
    if not body:
        errors.append(f"release_notes: '## [{version}]' is empty")
    if re.search(r"(^|\s)[@\\]ref\s", body):
        errors.append(f"release_notes: '## [{version}]' uses @ref, which GitHub cannot render; "
                      "use a Markdown link")
    used = {m.group(1).lower() for m in re.finditer(r"\[([^\]]+)\](?!\()", body)}
    tail = [f"[{defs[k][0]}]: {absolute(defs[k][1], link_base)}" for k in sorted(used) if k in defs]
    body = INLINE_LINK.sub(lambda m: m.group(1) + absolute(m.group("target"), link_base), body)
    return errors, body + "\n" + ("\n" + "\n".join(tail) + "\n" if tail else "")


def first_match(path, pattern):
    try:
        with open(path, encoding="utf-8") as f:
            m = pattern.search(f.read())
    except OSError:
        return None
    return m.group(1) if m else None


def run(tag, root, check_only, link_base=None, out=sys.stdout):
    version = tag[1:] if tag.startswith("v") else tag
    errors = []
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append(f"release_notes: tag {tag!r} is not vX.Y.Z")
    for what, path, pattern in (
            ("CMakeLists.txt project()", "CMakeLists.txt", CMAKE_VERSION),
            ("include/batchlas/version.hh", os.path.join("include", "batchlas", "version.hh"),
             HEADER_VERSION)):
        found = first_match(os.path.join(root, path), pattern)
        if found != version:
            errors.append(f"release_notes: {what} says {found}, the tag says {version}")
    try:
        with open(os.path.join(root, "docs", "changelog.md"), encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        errors.append(f"release_notes: cannot read docs/changelog.md ({e.strerror})")
        text = ""
    sec_errors, body = notes(text, version, link_base) if text else ([], "")
    errors += sec_errors
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if not check_only:
        out.write(body)
    return 0


def self_test():
    with open(os.path.join(FIXTURES, "changelog.md"), encoding="utf-8") as f:
        log = f.read()
    with open(os.path.join(FIXTURES, "expected-1.0.0.md"), encoding="utf-8") as f:
        expected = f.read()
    base = "https://github.com/o/r/blob/v1.0.0"
    cases = []

    def case(name, ok):
        cases.append((name, bool(ok)))

    errors, body = notes(log, "1.0.0", base)
    case("1.0.0 extracts exactly the expected notes", not errors and body == expected)
    if body != expected:
        sys.stderr.write("--- got ---\n" + body + "--- expected ---\n" + expected)
    case("Unreleased is not a release", notes(log, "Unreleased")[0])
    case("unbracketed, en-dash heading is read", notes(log, "0.9.0") == ([], "- Unbracketed.\n"))
    case("a heading inside a code fence ends nothing", "0.0.1" not in parse(log)[0])
    case("[YANKED] is refused", any("YANKED" in e for e in notes(log, "0.8.0")[0]))
    case("an undated section is refused", notes(log, "0.7.0")[0])
    case("an impossible date is refused", notes(log, "0.6.0")[0])
    case("an empty section is refused", any("empty" in e for e in notes(log, "0.5.0")[0]))
    case("@ref is refused", any("@ref" in e for e in notes(log, "0.4.0")[0]))
    case("a missing section is refused", notes(log, "2.0.0")[0])

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        os.makedirs(os.path.join(d, "include", "batchlas"))
        with open(os.path.join(d, "docs", "changelog.md"), "w", encoding="utf-8") as f:
            f.write(log)

        def tree(cmake, header):
            with open(os.path.join(d, "CMakeLists.txt"), "w", encoding="utf-8") as f:
                f.write(f"cmake_minimum_required(VERSION 3.17)\n"
                        f"project(BatchLAS VERSION {cmake} LANGUAGES CXX)\n")
            with open(os.path.join(d, "include", "batchlas", "version.hh"), "w",
                      encoding="utf-8") as f:
                f.write(f'#define BATCHLAS_VERSION_STRING "{header}"\n')

        class Sink:
            text = ""

            def write(self, s):
                Sink.text += s

        quiet = open(os.devnull, "w")
        saved, sys.stderr = sys.stderr, quiet
        try:
            tree("1.0.0", "1.0.0")
            case("run: agreeing tree passes", run("v1.0.0", d, False, base, Sink()) == 0
                 and Sink.text == expected)
            case("run: tag without a section fails", run("v1.0.1", d, True) == 1)
            case("run: pre-release tag fails", run("v1.0.0-rc1", d, True) == 1)
            tree("1.0.1", "1.0.0")
            case("run: CMake disagreeing fails", run("v1.0.0", d, True) == 1)
            tree("1.0.0", "1.0.1")
            case("run: version.hh disagreeing fails", run("v1.0.0", d, True) == 1)
        finally:
            sys.stderr = saved
            quiet.close()

    for name, ok in cases:
        if not ok:
            print(f"release_notes self-test: FAILED: {name}", file=sys.stderr)
    good = all(ok for _, ok in cases)
    print(f"release_notes self-test: {'ok' if good else 'FAILED'} ({len(cases)} cases)")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--check", action="store_true", help="check agreement, print nothing")
    ap.add_argument("--link-base", help="make relative links absolute against <base>/docs/")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=REPO)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.tag:
        ap.error("a tag is required")
    return run(a.tag, a.root, a.check, a.link_base)


if __name__ == "__main__":
    sys.exit(main())
