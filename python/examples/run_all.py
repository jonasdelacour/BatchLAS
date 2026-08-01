"""Run every example in order and summarise the result.

The examples are notebooks written in the "percent" cell format, which means
they are also ordinary Python scripts. This runner executes the `.py` files
directly, so validating them needs no Jupyter kernel and takes seconds.

To rebuild the rendered `.ipynb` files instead, use `build_notebooks.py`.

Usage:
    python run_all.py            # run all examples
    python run_all.py 05 06      # only the examples whose name starts with 05/06

Exits non-zero if any example raises or prints a failed check.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    scripts = sorted(p for p in here.glob("[0-9][0-9]_*.py"))
    if len(sys.argv) > 1:
        wanted = tuple(sys.argv[1:])
        scripts = [p for p in scripts if p.name.startswith(wanted)]
    if not scripts:
        print("no matching examples")
        return 1

    failures: list[str] = []
    for script in scripts:
        print(f"\n{'=' * 72}\nRunning {script.name}\n{'=' * 72}")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(here),
            capture_output=True,
            text=True,
        )
        sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            failures.append(f"{script.name} (exit {completed.returncode})")
        elif "[FAIL]" in completed.stdout:
            failed_checks = sum(1 for line in completed.stdout.splitlines() if "[FAIL]" in line)
            failures.append(f"{script.name} ({failed_checks} failed checks)")

    print(f"\n{'=' * 72}")
    print(f"{len(scripts) - len(failures)}/{len(scripts)} examples passed")
    for failure in failures:
        print(f"  FAILED: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
