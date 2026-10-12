#!/usr/bin/env python3
"""Fail if a registered ctest executable was not built.

    .github/ci/check_built_tests.py <build-dir>

For a compile-only job (acpp-compile): nothing runs the tests, so a test target that dropped
out of `all` would never be compiled and nothing would say so. Reads `ctest --show-only=json-v1`;
the first command word inside the build tree that is not a directory is the test's executable
(tests/ctest_gpu_env.sh may wrap it), and it must exist and be executable. Tests whose command
names nothing in the build tree (cmake -P, consumer_test.sh) are not build outputs.
"""
import json
import os
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    build = os.path.realpath(sys.argv[1])
    out = subprocess.run(["ctest", "--test-dir", build, "--show-only=json-v1"],
                         capture_output=True, text=True, check=True).stdout
    tests = json.loads(out).get("tests", [])
    built, missing = [], []
    for t in tests:
        inside = [os.path.realpath(w) for w in t.get("command") or []]
        inside = [w for w in inside if w.startswith(build + os.sep) and not os.path.isdir(w)]
        if not inside:
            continue
        exe = inside[0]
        ok = os.path.isfile(exe) and os.access(exe, os.X_OK)
        (built if ok else missing).append((t["name"], exe))
    print(f"check_built_tests: {len(tests)} ctest tests, {len(built)} built executables, "
          f"{len(missing)} missing")
    for name, exe in missing:
        print(f"  MISSING {name}: {exe}")
    if not built:
        print("check_built_tests: no test executable in the build tree (tests not enabled?)")
        return 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
