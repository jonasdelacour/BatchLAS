#!/usr/bin/env python3
"""Self-test for scripts/agent_build_guard.py: runs the hook as Claude Code does (JSON on stdin).

    python3 scripts/tests/test_agent_build_guard.py
"""

import json
import os
import subprocess
import sys

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_build_guard.py")
ROOT = os.path.dirname(os.path.dirname(HOOK))

BLOCK, PASS = 2, 0
CASES = [
    # -j caps
    ("cmake --build build --target stedc_tests -j8", PASS),
    ("cmake --build build --target stedc_tests -j 8", PASS),
    ("cmake --build build --target stedc_tests --parallel 4", PASS),
    ("cmake --build build --target stedc_tests -j20", BLOCK),
    ("cmake --build build --target stedc_tests -j 20", BLOCK),
    ("cmake --build build --target stedc_tests --parallel=16", BLOCK),
    ("cmake --build build -j --target stedc_tests", BLOCK),
    ("cmake --build build --target stedc_tests -j", BLOCK),
    ("cmake --build build --target x -j$(nproc)", BLOCK),
    ("cmake --build build --target x -j \"$(nproc)\"", BLOCK),
    ("cmake --build build --target x -- -j32", BLOCK),
    ("cmake --build build --target x -- -j4", PASS),
    ("CMAKE_BUILD_PARALLEL_LEVEL=20 cmake --build build --target x", BLOCK),
    ("export CMAKE_BUILD_PARALLEL_LEVEL=32; cmake --build build --target x", BLOCK),
    ("CMAKE_BUILD_PARALLEL_LEVEL=8 cmake --build build --target x", PASS),
    ("make -j20", BLOCK),
    ("make -j", BLOCK),
    ("make -j 4 -C build stedc_tests", PASS),
    ("make -kj32 all", BLOCK),
    ("make --jobs=12", BLOCK),
    ("make -C build -j$(nproc) stedc_tests", BLOCK),
    ("MAKEFLAGS=-j24 make -C build x", BLOCK),
    ("make docs", PASS),
    ("cd build && make -j16 stedc_tests", BLOCK),
    ("cd build\nmake -j16 stedc_tests", BLOCK),
    ("nice -n 19 cmake --build build --target x -j 12", BLOCK),
    ("timeout 600 env -u LIBRARY_PATH cmake --build build --target x -j64", BLOCK),
    ("bash -c 'cmake --build build --target x -j32'", BLOCK),
    ("if make -j20; then echo ok; fi", BLOCK),
    ("cmake --build build \\\n  --target x -j 40", BLOCK),
    # untargeted builds
    ("cmake --build build", BLOCK),
    ("cmake --build build -j8", BLOCK),
    ("cmake --build build-gate -j 8 2>&1 | tail -n 20", BLOCK),
    ("BATCHLAS_FULL_BUILD=1 cmake --build build -j8", PASS),
    ("cmake --build build -t stedc_tests", PASS),
    ("cmake --build build --target=stedc_tests", PASS),
    ("cmake --build --preset dev", PASS),           # build preset lists targets
    ("cmake --build --preset dev-tests", BLOCK),    # builds everything
    ("scripts/gate.sh --only full", PASS),
    ("sh scripts/gate.sh; cmake --build build-gate", BLOCK),
    ("cat scripts/gate.sh; cmake --build build -j8", BLOCK),
    ("cmake --build build --target all -j8", BLOCK),
    ("cmake --build build --target -j8", BLOCK),
    ("cmake --build build --target stedc_tests all -j8", BLOCK),
    ("cmake --build build --target stedc_tests posv_tests -j8", PASS),
    ("export BATCHLAS_FULL_BUILD=1; cmake --build build -j8", PASS),
    ("BATCHLAS_FULL_BUILD=1 true; cmake --build build -j8", BLOCK),
    ("echo BATCHLAS_FULL_BUILD=1; cmake --build build", BLOCK),
    # heredoc bodies are text, unless a shell reads them
    ("cat > /tmp/x.sh <<'EOF2'\ncmake --build build\nEOF2", PASS),
    ("cat > /tmp/x.sh <<EOF\nmake -j\ncmake --build build -j $(nproc)\nEOF\nls", PASS),
    ("cat <<-EOF > notes\n\tmake -j20\n\tEOF\nmake -j20", BLOCK),
    ("bash <<'EOF'\ncmake --build build\nEOF", BLOCK),
    ("cat <<EOF | sh\nmake -j32\nEOF", BLOCK),
    # not builds
    ("scripts/b stedc_tests", PASS),
    ("BATCHLAS_FULL_BUILD=1 scripts/b -B build-gate", PASS),
    ("cmake -S . -B build -DBATCHLAS_CPU_TARGET=none", PASS),
    ("ctest --test-dir build -R '^stedc_tests$' -j4", PASS),
    ("echo 'make -j' > notes.txt", PASS),
    ("grep -rn 'cmake --build' docs | head", PASS),
    ("git log --oneline -5", PASS),
    ("ls", PASS),
    # never block on garbage
    ("cmake --build build -j20 'unterminated", PASS),
    ("", PASS),
]
RAW = [
    ("not json at all", PASS),
    (json.dumps({"tool_name": "Read", "tool_input": {"file_path": "/x"}}), PASS),
    (json.dumps({"tool_name": "Bash", "tool_input": None}), PASS),
]


def run(stdin, env=None):
    proc = subprocess.run([sys.executable, HOOK], input=stdin, capture_output=True, text=True,
                          env=env, cwd=ROOT)
    return proc.returncode, proc.stderr


def main():
    bad = 0
    env = dict(os.environ, CLAUDE_PROJECT_DIR=ROOT)
    env.pop("BATCHLAS_HOOK_MAX_JOBS", None)
    cases = [(json.dumps({"tool_name": "Bash", "tool_input": {"command": c}}), want, c) for c, want in CASES]
    cases += [(raw, want, raw[:60]) for raw, want in RAW]
    for stdin, want, label in cases:
        rc, err = run(stdin, env)
        ok = rc == want and (rc != 2 or "scripts/b" in err)
        bad += not ok
        print("  %-4s %s  %r" % ("ok" if ok else "FAIL", "block" if want else "pass ", label))
        if not ok:
            print("       exit %d, stderr: %s" % (rc, err.strip()))
    wide = dict(env, BATCHLAS_HOOK_MAX_JOBS="32")
    rc, _ = run(json.dumps({"tool_name": "Bash", "tool_input": {"command": "make -j32"}}), wide)
    ok = rc == 0
    bad += not ok
    print("  %-4s pass   BATCHLAS_HOOK_MAX_JOBS=32 in the hook environment admits make -j32" % ("ok" if ok else "FAIL"))
    print("agent_build_guard self-test: %d failure(s) of %d" % (bad, len(cases) + 1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
