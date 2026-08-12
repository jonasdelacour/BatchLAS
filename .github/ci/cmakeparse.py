"""Minimal CMake list-file lexer shared by the CI checks in this directory.

It is deliberately not a full CMake implementation: it blanks comments and
argument literals so that parentheses and command names can be located without
executing anything. `cmake -P` would parse a list file properly, but it also
*runs* it (BatchLASCcache.cmake writes files, BatchLASDependencies.cmake fetches
things), which is not something a lint job should do.
"""

import re

_BRACKET_OPEN = re.compile(r"(=*)\[")
_NAME_AT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t\r\n]*\(")


def _blank(text):
    """Same length as `text`, but only newlines survive (keeps line numbers)."""
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _bracket_end(text, start, equals):
    """Index just past the closing `]==]` of a bracket comment/argument."""
    closer = "]" + "=" * equals + "]"
    idx = text.find(closer, start)
    if idx < 0:
        return -1
    return idx + len(closer)


def clean_source(text):
    """Blank out comments, quoted arguments and bracket arguments.

    Returns (cleaned_text, errors), where errors is a list of
    (line, message) for anything that never terminates.
    """
    errors = []
    out = []
    i = 0
    n = len(text)
    line = 1
    while i < n:
        ch = text[i]
        if ch == "\n":
            out.append("\n")
            i += 1
            line += 1
            continue
        if ch == "#":
            m = _BRACKET_OPEN.match(text, i + 1)
            if m:
                end = _bracket_end(text, m.end(), len(m.group(1)))
                if end < 0:
                    errors.append((line, "unterminated bracket comment"))
                    end = n
                out.append(_blank(text[i:end]))
                line += text.count("\n", i, end)
                i = end
                continue
            end = text.find("\n", i)
            if end < 0:
                end = n
            out.append(_blank(text[i:end]))
            i = end
            continue
        if ch == '"':
            j = i + 1
            terminated = False
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    terminated = True
                    break
                j += 1
            end = j + 1 if terminated else n
            if not terminated:
                errors.append((line, "unterminated quoted argument"))
            out.append(_blank(text[i:end]))
            line += text.count("\n", i, end)
            i = end
            continue
        if ch == "[":
            m = _BRACKET_OPEN.match(text, i + 1)
            if m:
                end = _bracket_end(text, m.end(), len(m.group(1)))
                if end < 0:
                    errors.append((line, "unterminated bracket argument"))
                    end = n
                out.append(_blank(text[i:end]))
                line += text.count("\n", i, end)
                i = end
                continue
        out.append(ch)
        i += 1
    return "".join(out), errors


def iter_commands(text, cleaned=None):
    """Yield (name, args_text, line, arg_start) for each top-level invocation.

    `args_text` is sliced out of the ORIGINAL text, so literals (flags, paths)
    are visible to the caller; structure comes from the cleaned text.
    """
    if cleaned is None:
        cleaned, _ = clean_source(text)
    i = 0
    n = len(cleaned)
    while i < n:
        m = _NAME_AT.search(cleaned, i)
        if not m:
            return
        depth = 1
        j = m.end()
        while j < n and depth:
            if cleaned[j] == "(":
                depth += 1
            elif cleaned[j] == ")":
                depth -= 1
            j += 1
        if depth:
            return  # unbalanced; the syntax check reports it
        line = cleaned.count("\n", 0, m.start()) + 1
        yield m.group(1), text[m.end():j - 1], line, m.end()
        i = j
