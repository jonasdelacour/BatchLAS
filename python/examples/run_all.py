"""Execute every example notebook and summarise the result.

Each notebook verifies its own results against NumPy/SciPy and prints `[ok  ]`
or `[FAIL]` for every check, so executing them cleanly is a smoke test of the
Python bindings.

Usage:
    python run_all.py                  # execute all notebooks, report failures
    python run_all.py 05 06            # only those whose name starts with 05/06
    python run_all.py --save           # also write the executed output back in

`--save` refreshes the outputs committed in the notebooks; use it after editing
one so the rendered version matches what the code now produces.

Exits non-zero if any notebook raises or reports a failed check.

Requires `nbformat`, `nbclient` and `ipykernel`:

    pip install nbformat nbclient ipykernel
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="*", help="only run notebooks whose name starts with these")
    parser.add_argument("--save", action="store_true", help="write executed outputs back into the notebooks")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    args = parser.parse_args()

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        print(f"missing dependency ({exc.name}): pip install nbformat nbclient ipykernel", file=sys.stderr)
        return 1

    here = pathlib.Path(__file__).resolve().parent
    notebooks = sorted(here.glob("[0-9][0-9]_*.ipynb"))
    if args.prefixes:
        notebooks = [p for p in notebooks if p.name.startswith(tuple(args.prefixes))]
    if not notebooks:
        print("no matching notebooks")
        return 1

    failures: list[str] = []
    for path in notebooks:
        print(f"\n{'=' * 72}\nRunning {path.name}\n{'=' * 72}", flush=True)
        notebook = nbformat.read(path, as_version=4)

        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(here)}},
            allow_errors=True,
        )
        client.execute()

        errors = []
        failed_checks = 0
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(f"{output.get('ename', 'error')}: {output.get('evalue', '')}")
                elif output.get("output_type") == "stream":
                    text = output.get("text", "")
                    sys.stdout.write(text)
                    failed_checks += text.count("[FAIL]")

        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)

        if errors:
            failures.append(f"{path.name} ({errors[0].split(':')[0]})")
        elif failed_checks:
            failures.append(f"{path.name} ({failed_checks} failed checks)")

        if args.save:
            nbformat.write(notebook, path)
            print(f"  (saved executed output to {path.name})")

    print(f"\n{'=' * 72}")
    print(f"{len(notebooks) - len(failures)}/{len(notebooks)} notebooks passed")
    for failure in failures:
        print(f"  FAILED: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
