#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self-test for the host handling of scripts/agent/dispatch.sh, offline.

    python3 scripts/tests/test_dispatch_hosts.py

Fake `ssh` and `bash` on PATH record their arguments and run nothing, so no session starts and
no box is contacted; dispatch.sh itself runs under the real bash. The cases pin the host order
(mac, jrhnbr, tr02), the jump-host names, the unknown-host message and every exit code.
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "dispatch.sh")

UNKNOWN = "dispatch: unknown host '%s' (jrhnbr, tr02, mac)\n"
SSH_EXIT = 255
# Records argv as fields separated by \x1f, one invocation per \x1e; arguments may hold newlines.
FAKE = """#!/bin/sh
{ printf '%%s\\037' "%s"; for a in "$@"; do printf '%%s\\037' "$a"; done; printf '\\036'; } >> "$DISPATCH_FAKE_LOG"
exit %d
"""
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def ssh(host):
    return ("ssh", host)


LOCAL = ("bash", "-lc")

# (args, exit code, stdout or None for "not checked", stderr, recorded calls as (tool, first arg))
# A bogus host does not stop on(): ssh_of's die only leaves its $(...), so ssh runs with host "".
CASES = [
    (["start", "bogus", "n", "/nonexistent-brief"], 2, "", UNKNOWN % "bogus", []),
    (["start", "Mac", "n", "/nonexistent-brief"], 2, "", UNKNOWN % "Mac", []),
    (["start", "", "n", "/nonexistent-brief"], 2, "", UNKNOWN % "", []),
    (["start", "mac", "n", "/nonexistent-brief"], 2, "", "dispatch: no brief file /nonexistent-brief\n", []),
    (["start", "jrhnbr", "n", "/nonexistent-brief"], 2, "", "dispatch: no brief file /nonexistent-brief\n", []),
    (["start", "tr02", "n", "/nonexistent-brief"], 2, "", "dispatch: no brief file /nonexistent-brief\n", []),
    (["report", "bogus", "n"], SSH_EXIT, "", UNKNOWN % "bogus", [ssh("")]),
    (["logs", "bogus", "n"], 0, "", UNKNOWN % "bogus", [ssh("")]),
    (["stop", "bogus", "n"], SSH_EXIT, "", UNKNOWN % "bogus", [ssh("")]),
    (["rm", "bogus", "n"], SSH_EXIT, "", UNKNOWN % "bogus", [ssh("")]),
    (["attach", "bogus", "n"], SSH_EXIT, "", 2 * (UNKNOWN % "bogus"), [("ssh", "-t")]),
    (["list", "bogus"], 0, "== bogus\n(unreachable)\n", "", [ssh("")]),
    (["list"], 0, "== mac\n== jrhnbr\n(unreachable)\n== tr02\n(unreachable)\n", "",
     [LOCAL, ssh("JRHNBR-jump"), ssh("threadripper02-jump")]),
    (["list", "tr02 mac"], 0, "== tr02\n(unreachable)\n== mac\n", "", [ssh("threadripper02-jump"), LOCAL]),
    (["hosts"], 0, "== mac  (unreachable)\n== jrhnbr  (unreachable)\n== tr02  (unreachable)\n", "",
     [LOCAL, ssh("JRHNBR-jump"), ssh("threadripper02-jump")]),
    ([], 2, None, "", []),
    (["nonsense"], 2, None, "", []),
]

USAGE_FIRST = "# Run an unattended Claude session on another box (or this one) in its own worktree, from the Mac."
USAGE_LAST = "# dispatch/NAME, writes .dispatch/report.md and then stays idle until `stop`."
USAGE_LINES = 20


def make_fakes(d):
    for tool, code in (("ssh", SSH_EXIT), ("bash", 0)):
        p = os.path.join(d, tool)
        with open(p, "w") as f:
            f.write(FAKE % (tool, code))
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)


def recorded(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        calls = [c.split("\x1f")[:-1] for c in f.read().split("\x1e")[:-1]]
    out = []
    for c in calls:
        if c[0] == "ssh" and c[1:5] == SSH_OPTS:
            out.append(("ssh", c[5]))
        else:
            out.append((c[0], c[1] if len(c) > 1 else None))
    return out


def check_usage(out):
    lines = out.split("\n")
    if lines[-1] != "" or len(lines) - 1 != USAGE_LINES:
        return ["usage is %d lines, want %d" % (len(lines) - 1, USAGE_LINES)]
    bad = []
    if lines[0] != USAGE_FIRST:
        bad.append("usage first line %r" % lines[0])
    if lines[USAGE_LINES - 1] != USAGE_LAST:
        bad.append("usage last line %r" % lines[USAGE_LINES - 1])
    return bad


def main():
    real_bash = shutil.which("bash")
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        fakes = os.path.join(d, "bin")
        os.mkdir(fakes)
        make_fakes(fakes)
        for args, code, want_out, want_err, want_calls in CASES:
            log = os.path.join(d, "calls")
            if os.path.exists(log):
                os.remove(log)
            env = dict(os.environ, PATH=fakes + os.pathsep + os.environ.get("PATH", ""),
                       HOME=d, DISPATCH_FAKE_LOG=log)
            r = subprocess.run([real_bash, SCRIPT] + args, capture_output=True, text=True, env=env,
                               stdin=subprocess.DEVNULL, timeout=60)
            calls = recorded(log)
            bad = []
            if r.returncode != code:
                bad.append("exit %d, want %d" % (r.returncode, code))
            if want_out is None:
                bad += check_usage(r.stdout)
            elif r.stdout != want_out:
                bad.append("stdout %r, want %r" % (r.stdout, want_out))
            if r.stderr != want_err:
                bad.append("stderr %r, want %r" % (r.stderr, want_err))
            if calls != want_calls:
                bad.append("calls %r, want %r" % (calls, want_calls))
            print("  %-4s %r" % ("FAIL" if bad else "ok", args))
            for b in bad:
                print("       " + b)
            fails += bool(bad)
    print("dispatch_hosts: %d/%d cases pass" % (len(CASES) - fails, len(CASES)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
