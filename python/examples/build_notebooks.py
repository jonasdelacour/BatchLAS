"""Turn the percent-format example scripts into executed Jupyter notebooks.

The `.py` files are the source of truth: they are plain Python, they run as
scripts, and they diff cleanly. They are also written in the widely supported
"percent" cell format, so Jupyter, JupyterLab, VS Code and PyCharm already open
them as notebooks.

This script additionally materialises real `.ipynb` files so the examples render
with formatted prose and captured output on GitHub and in any plain notebook
viewer.

Usage:
    python build_notebooks.py                 # rebuild and execute every notebook
    python build_notebooks.py 05 06           # only these
    python build_notebooks.py --no-execute    # convert without running anything

Requires `nbformat`; execution additionally requires `nbclient` and `ipykernel`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def _strip_comment_prefix(lines: list[str]) -> str:
    """Turn the `# `-prefixed body of a markdown cell back into markdown."""
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def parse_percent_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into a list of ``(kind, source)`` cells."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    body: list[str] = []

    def flush() -> None:
        source = "\n".join(body).strip("\n")
        if source:
            cells.append((kind, source))

    for line in text.splitlines():
        if line.startswith(CELL_MARKER):
            flush()
            kind = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
            body = []
            continue
        body.append(line)
    flush()

    return [
        (cell_kind, _strip_comment_prefix(source.splitlines()) if cell_kind == "markdown" else source)
        for cell_kind, source in cells
    ]


def build_notebook(cells: list[tuple[str, str]]):
    import nbformat

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="*", help="only build examples whose name starts with these")
    parser.add_argument("--no-execute", action="store_true", help="convert without running the notebooks")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell execution timeout in seconds")
    args = parser.parse_args()

    try:
        import nbformat
    except ImportError:
        print("nbformat is required: pip install nbformat", file=sys.stderr)
        return 1

    here = pathlib.Path(__file__).resolve().parent
    scripts = sorted(p for p in here.glob("[0-9][0-9]_*.py"))
    if args.prefixes:
        scripts = [p for p in scripts if p.name.startswith(tuple(args.prefixes))]
    if not scripts:
        print("no matching examples", file=sys.stderr)
        return 1

    failures: list[str] = []
    for script in scripts:
        notebook = build_notebook(parse_percent_cells(script.read_text()))

        if not args.no_execute:
            try:
                from nbclient import NotebookClient
            except ImportError:
                print("nbclient is required to execute: pip install nbclient", file=sys.stderr)
                return 1

            print(f"executing {script.name} ...", flush=True)
            client = NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(here)}},
                allow_errors=True,
            )
            client.execute()

            errored = [
                output
                for cell in notebook.cells
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]
            failed_checks = sum(
                output.get("text", "").count("[FAIL]")
                for cell in notebook.cells
                for output in cell.get("outputs", [])
                if output.get("output_type") == "stream"
            )
            if errored:
                failures.append(f"{script.name} ({errored[0].get('ename', 'error')})")
            elif failed_checks:
                failures.append(f"{script.name} ({failed_checks} failed checks)")

        target = script.with_suffix(".ipynb")
        nbformat.write(notebook, target)
        print(f"wrote {target.name}")

    for failure in failures:
        print(f"  FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
