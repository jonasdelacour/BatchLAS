#!/usr/bin/env python3
"""Claude Code PreToolUse hook (matcher Bash): refuse builds that OOM the box or link everything.

Blocks (exit 2, reason on stderr):
  * `cmake --build` / `make` with -j/--parallel/--jobs above the cap (8), with no number
    (unbounded), or with a computed value such as -j$(nproc); also CMAKE_BUILD_PARALLEL_LEVEL or
    MAKEFLAGS above the cap on the same command line;
  * an untargeted `cmake --build <dir>` (no --target/-t, or only `all`; a --preset whose build
    preset lists targets counts as targeted) unless that simple command carries
    BATCHLAS_FULL_BUILD=1, or an earlier `export BATCHLAS_FULL_BUILD=1` on the line set it.
Heredoc bodies are text, not commands, unless the heredoc feeds a shell (bash <<EOF).
Everything else passes (exit 0), and so does anything this script fails to parse.
The cap is BATCHLAS_HOOK_MAX_JOBS in the hook's own environment (e.g. settings.json "env" on a
box with measured headroom), never something the command line can set.
evidence: docs/developer/agent-guide.md section 7 (Build Performance)
"""

import json
import os
import re
import shlex
import sys

WRAPPERS = {"env", "nice", "ionice", "time", "timeout", "nohup", "stdbuf", "taskset", "chrt",
            "sudo", "command", "exec", "builtin", "unbuffer"}
WRAPPER_ARG_OPTS = {"nice": {"-n"}, "ionice": {"-c", "-n", "-p"}, "env": {"-u", "-C", "-S"},
                    "timeout": {"-s", "-k", "--signal", "--kill-after"}, "taskset": {"-c"},
                    "sudo": {"-u", "-g"}}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
PUNCT = ";&|()\n"
KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "while", "until", "do", "time"}

ALTERNATIVE = ("Use scripts/b instead: `scripts/b [-B build] <target>...` (caps -j at 8, takes the box-wide "
               "build lock, checks memory, logs, and verifies the target was linked). The full pre-push "
               "build is scripts/gate.sh. See docs/developer/agent-guide.md section 7.")


def max_jobs():
    try:
        return int(os.environ.get("BATCHLAS_HOOK_MAX_JOBS", "8"))
    except ValueError:
        return 8


HEREDOC = re.compile(r"(?<!<)<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
SHELLS = re.compile(r"(^|[\s;&|(/])(ba|da|z)?sh([\s;&|)]|$)")


def split_heredocs(command):
    """(command with heredoc bodies removed, [bodies fed to a shell])."""
    lines, out, shell_bodies, i = command.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in HEREDOC.finditer(line):
            strip_tabs, word, body = m.group(1) == "-", m.group(3), []
            while i < len(lines):
                end = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if end == word:
                    break
                body.append(lines[i - 1])
            if SHELLS.search(line):
                shell_bodies.append("\n".join(body))
    return "\n".join(out), shell_bodies


def tokenize(command):
    lex = shlex.shlex(command.replace("\\\n", " "), posix=True, punctuation_chars=PUNCT)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
    return list(lex)


def simple_commands(tokens):
    cur = []
    for tok in tokens:
        if tok and all(c in PUNCT for c in tok):
            if cur:
                yield cur
            cur = []
        else:
            cur.append(tok)
    if cur:
        yield cur


def strip_prefix(words):
    """Drop leading VAR=value words and wrapper commands (nice -n 19, env -u X, timeout 60, ...)."""
    assigns, i = [], 0
    while i < len(words):
        w = words[i]
        if w in KEYWORDS:
            i += 1
            continue
        if ASSIGN.match(w):
            assigns.append(w)
            i += 1
            continue
        base = os.path.basename(w)
        if base not in WRAPPERS:
            break
        i += 1
        takes = WRAPPER_ARG_OPTS.get(base, set())
        while i < len(words) and words[i].startswith("-"):
            opt = words[i]
            i += 1
            if opt in takes and i < len(words):
                i += 1
        if base == "timeout" and i < len(words):
            i += 1
        if base == "taskset" and i < len(words) and re.match(r"^(0x)?[0-9a-fA-F,\-]+$", words[i]):
            i += 1
        if base == "chrt" and i < len(words) and words[i].isdigit():
            i += 1
    return assigns, words[i:]


def jobs_value(text):
    """'' (no number = unbounded), an int, or None for a computed value."""
    if text == "":
        return ""
    return int(text) if text.isdigit() else None


def check_jobs(value, where, cap):
    if value == "":
        return "%s has no job count, which means unbounded parallelism" % where
    if value is None:
        return "%s uses a computed job count (e.g. $(nproc)); write a number <= %d" % (where, cap)
    if value > cap:
        return "%s asks for %d jobs; the cap is %d (heavy SYCL TUs exceed 14 GB each, zero swap)" % (where, value, cap)
    return None


def scan_make_args(args, cap):
    i = 0
    while i < len(args):
        a = args[i]
        m = re.match(r"^-([A-Za-z]*)j(.*)$", a) if not a.startswith("--") else None
        if a in ("-j", "--jobs"):
            nxt = args[i + 1] if i + 1 < len(args) else ""
            if nxt.isdigit():
                i += 1
                err = check_jobs(int(nxt), "make %s %s" % (a, nxt), cap)
            else:
                err = check_jobs("", "make %s" % a, cap)
            if err:
                return err
        elif a.startswith("--jobs="):
            err = check_jobs(jobs_value(a.split("=", 1)[1]), "make " + a, cap)
            if err:
                return err
        elif m and set(m.group(1)) <= set("kisBnqrRstwe"):
            err = check_jobs(jobs_value(m.group(2)), "make " + a, cap)
            if err:
                return err
        i += 1
    return None


def preset_has_targets(name):
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        with open(os.path.join(root, "CMakePresets.json"), encoding="utf-8") as fh:
            presets = json.load(fh).get("buildPresets", [])
    except (OSError, ValueError):
        return False
    return any(p.get("name") == name and p.get("targets") for p in presets)


def scan_cmake_build(args, cap, full_ok):
    """ARGS are the words after `cmake`; returns a reason or None."""
    targets = []
    preset = None
    native = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            native = args[i + 1:]
            break
        if a in ("-j", "--parallel"):
            nxt = args[i + 1] if i + 1 < len(args) else ""
            if nxt.isdigit():
                i += 1
                err = check_jobs(int(nxt), "cmake --build %s %s" % (a, nxt), cap)
            elif nxt and not nxt.startswith("-"):
                err = check_jobs(None, "cmake --build %s %s" % (a, nxt), cap)
            else:
                err = check_jobs("", "cmake --build %s" % a, cap)
            if err:
                return err
        elif a.startswith("--parallel="):
            err = check_jobs(jobs_value(a.split("=", 1)[1]), "cmake --build " + a, cap)
            if err:
                return err
        elif a.startswith("-j") and len(a) > 2:
            err = check_jobs(jobs_value(a[2:]), "cmake --build " + a, cap)
            if err:
                return err
        elif a.startswith("--target="):
            targets.append(a.split("=", 1)[1])
        elif a in ("--target", "-t"):
            while i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                targets.append(args[i])
        elif a == "--preset" and i + 1 < len(args):
            preset = args[i + 1]
        elif a.startswith("--preset="):
            preset = a.split("=", 1)[1]
        i += 1
    err = scan_make_args(native, cap)
    if err:
        return err
    targeted = bool(targets) and "all" not in targets
    if preset and preset_has_targets(preset):
        targeted = True
    if not targeted and not full_ok:
        return ("untargeted `cmake --build` builds the library AND every test and benchmark "
                "executable (~150 links, and so does --target all); name the targets you will "
                "run (--target X), or set "
                "BATCHLAS_FULL_BUILD=1 if you really mean everything")
    return None


def env_parallel_reason(assigns, cap):
    for a in assigns:
        name, _, value = a.partition("=")
        if name == "CMAKE_BUILD_PARALLEL_LEVEL":
            err = check_jobs(jobs_value(value), a, cap)
            if err:
                return err
        if name == "MAKEFLAGS":
            err = scan_make_args(value.split(), cap)
            if err:
                return err
    return None


def reason_for(command, cap, depth=0):
    command, shell_bodies = split_heredocs(command)
    if depth < 3:
        for body in shell_bodies:
            err = reason_for(body, cap, depth + 1)
            if err:
                return err
    try:
        tokens = tokenize(command)
    except ValueError:
        return None
    all_assigns = [t for t in tokens if ASSIGN.match(t)]
    exported_full = False
    for words in simple_commands(tokens):
        assigns, rest = strip_prefix(words)
        if not rest:
            continue
        base = os.path.basename(rest[0])
        if base == "export":
            exported_full |= "BATCHLAS_FULL_BUILD=1" in rest[1:]
            continue
        is_build = False
        if base == "cmake" and "--build" in rest[1:]:
            is_build = True
            full_ok = exported_full or "BATCHLAS_FULL_BUILD=1" in assigns
            err = scan_cmake_build(rest[1:], cap, full_ok)
        elif base in ("make", "gmake"):
            is_build = True
            err = scan_make_args(rest[1:], cap)
        else:
            err = None
            if depth < 3 and base in ("bash", "sh", "zsh", "dash") and "-c" in rest:
                idx = rest.index("-c")
                if idx + 1 < len(rest):
                    err = reason_for(rest[idx + 1], cap, depth + 1)
        if not err and is_build:
            err = env_parallel_reason(assigns + all_assigns, cap)
        if err:
            return err
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        if payload.get("tool_name", "Bash") != "Bash":
            return 0
        command = (payload.get("tool_input") or {}).get("command") or ""
        reason = reason_for(command, max_jobs())
    except Exception:  # never block on a parse failure
        return 0
    if reason:
        sys.stderr.write("BLOCKED by scripts/agent_build_guard.py: %s.\n%s\n" % (reason, ALTERNATIVE))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
