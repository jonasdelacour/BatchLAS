#!/usr/bin/env python3
"""Syntax-check every CMake list file in the repository.

Catches the class of mistake that only shows up when somebody configures the
project: unbalanced parentheses, an unterminated string, an `if()` without its
`endif()`, an `else()` that belongs to no `if()`. None of that needs a compiler,
a GPU, or a build tree, so it can run on any machine.

What it does NOT do: evaluate anything. A command that exists but is misused
(wrong keyword, missing target) still gets through here — only a real configure
finds that, and a real configure needs the SYCL compiler.

Usage:
    python3 .github/ci/check_cmake_syntax.py [paths...]
Exit code 0 = clean, 1 = findings.
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmakeparse import clean_source  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_DIRS = {".git", ".claude", "build", "_deps", "node_modules", "__pycache__"}

# Blocks that must be closed, and the command that closes them.
OPENERS = {
    "if": "endif",
    "foreach": "endforeach",
    "while": "endwhile",
    "function": "endfunction",
    "macro": "endmacro",
    "block": "endblock",
}
CLOSERS = {v: k for k, v in OPENERS.items()}
# Commands that may only appear inside a particular open block.
MIDDLES = {"else": "if", "elseif": "if"}


def list_files(roots):
    files = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name == "CMakeLists.txt" or name.endswith(".cmake") or name.endswith(".cmake.in"):
                    files.append(os.path.join(dirpath, name))
    return sorted(set(files))


def check_file(path):
    """Return a list of (line, message)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    cleaned, findings = clean_source(text)
    findings = list(findings)

    depth = 0
    open_at = []          # line numbers of unclosed '('
    stack = []            # (opener, line)
    i = 0
    n = len(cleaned)
    line = 1
    word = []
    word_line = 1
    while i < n:
        ch = cleaned[i]
        if ch == "\n":
            line += 1
            i += 1
            if not word:
                word_line = line
            continue
        if ch.isalnum() or ch == "_":
            if not word:
                word_line = line
            word.append(ch)
            i += 1
            continue
        if ch == "(":
            if depth == 0:
                name = "".join(word).lower()
                if name in OPENERS:
                    stack.append((name, word_line))
                elif name in CLOSERS:
                    want = CLOSERS[name]
                    if not stack:
                        findings.append((word_line, "%s() with no matching %s()" % (name, want)))
                    elif stack[-1][0] != want:
                        findings.append((word_line, "%s() closes %s() opened at line %d"
                                         % (name, stack[-1][0], stack[-1][1])))
                        stack.pop()
                    else:
                        stack.pop()
                elif name in MIDDLES:
                    if not stack or stack[-1][0] != MIDDLES[name]:
                        findings.append((word_line, "%s() outside any %s() block" % (name, MIDDLES[name])))
                elif not name:
                    findings.append((line, "'(' where a command name was expected"))
            depth += 1
            open_at.append(line)
            word = []
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                findings.append((line, "unmatched ')'"))
                depth = 0
            elif open_at:
                open_at.pop()
            word = []
            i += 1
            continue
        if word:
            word = []
        i += 1

    for opened in open_at:
        findings.append((opened, "'(' is never closed"))
    for name, opened in stack:
        findings.append((opened, "%s() is never closed by %s()" % (name, OPENERS[name])))
    return sorted(findings)


def main(argv):
    roots = argv[1:] or [REPO]
    files = list_files(roots)
    if not files:
        print("check_cmake_syntax: no CMake files found under %s" % ", ".join(roots))
        return 1
    total = 0
    for path in files:
        for line, message in check_file(path):
            total += 1
            print("%s:%d: error: %s" % (os.path.relpath(path, REPO), line, message))
    print("check_cmake_syntax: %d file(s) checked, %d problem(s)" % (len(files), total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
