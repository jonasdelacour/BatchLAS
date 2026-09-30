#!/usr/bin/env python3
"""Generate the database pages of the documentation site.

Two pages are derived from the tree on every docs build, so they cannot drift:

  results_database.md   one row per raw result file in benchmarks/results/:
                        campaign, op, dtype, machine, when it was committed,
                        whether this checkout holds the data or only the LFS
                        pointer, and which documentation pages cite it.
  evidence_index.md     the reverse of every `evidence: docs/<page>.md#<a>`
                        pointer: for each cited section, the code that
                        depends on it.

Usage:
    python3 docs/tools/gen_db_pages.py --out build/docs/generated
    python3 docs/tools/gen_db_pages.py --self-test
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, ".github", "ci"))
import check_evidence_anchors as evidence  # noqa: E402

RESULTS = os.path.join("benchmarks", "results")
DTYPES = ("cdouble", "cfloat", "double", "float")
OPS = ("geqrf", "getrf", "getrs", "orgqr", "potrf", "posv", "gesv", "gemm",
       "syevx", "syev", "gesvd", "gesvdj", "getri", "trsm", "ormqr")
MACHINES = {"rtx4090": "RTX 4090 (sm_89)"}
DEFAULT_MACHINE = "RTX 4090 (sm_89), primary box — not recorded in file"
LFS_HEADER = "version https://git-lfs.github.com/spec/v1"


def git(*args, cwd=REPO):
    try:
        out = subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True, check=True)
        return out.stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def added_commits(root):
    """Path -> (date, short hash) of the commit that first added it."""
    log = git("log", "--diff-filter=A", "--format=@%cs %h", "--name-only", "--", RESULTS, cwd=root)
    added = {}
    stamp = None
    for line in log.splitlines():
        if line.startswith("@"):
            stamp = tuple(line[1:].split(" ", 1))
        elif line.strip() and stamp:
            added[line.strip()] = stamp  # log is newest first; keep the oldest
    return added


def classify(name):
    stem = os.path.splitext(name)[0]
    tokens = stem.split("_")
    campaign = tokens[0] if re.fullmatch(r"p\d+", tokens[0]) else "_".join(tokens[:2]) if tokens[0] == "factor" else tokens[0]
    op = next((t for t in tokens if t in OPS), "")
    dtype = next((t.rstrip("0123456789") for t in tokens if t.rstrip("0123456789") in DTYPES), "")
    machine = next((MACHINES[t] for t in tokens if t in MACHINES), DEFAULT_MACHINE)
    return campaign, op, dtype, machine


def describe_data(path):
    """(status, summary) for a result file as it exists in this checkout."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(1 << 20)
    except OSError:
        return "missing", ""
    if head.startswith(LFS_HEADER):
        size = re.search(r"^size (\d+)", head, re.M)
        return "LFS pointer", ("%s bytes in LFS" % size.group(1)) if size else ""
    if path.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(head)))
        if rows:
            return "present", "%d rows; columns: %s" % (len(rows) - 1, ", ".join(rows[0][:8]) + (" …" if len(rows[0]) > 8 else ""))
    return "present", "%d lines" % head.count("\n")


def doc_citations(root):
    """Result file name -> sorted docs pages mentioning it."""
    cites = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "docs")):
        dirnames[:] = sorted(d for d in dirnames if d not in evidence.SKIP_DIRS)
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            text = open(path, encoding="utf-8", errors="replace").read()
            for hit in set(re.findall(r"(?:benchmarks/results/)?([\w.+-]+\.(?:csv|jsonl|log|txt|sh))", text)):
                cites.setdefault(hit, set()).add(os.path.relpath(path, root))
    return {k: sorted(v) for k, v in cites.items()}


def results_page(root):
    folder = os.path.join(root, RESULTS)
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    added = added_commits(root)
    cites = doc_citations(root)
    rows, pointers = [], 0
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        campaign, op, dtype, machine = classify(name)
        status, summary = describe_data(path)
        pointers += status == "LFS pointer"
        date, commit = added.get(os.path.join(RESULTS, name), ("", ""))
        cited = ", ".join("`%s`" % c for c in cites.get(name, [])) or "—"
        rows.append("| `%s` | %s | %s | %s | %s | %s %s | %s | %s |" % (
            name, campaign, op or "—", dtype or "—", machine, date, ("`%s`" % commit) if commit else "",
            status + (" (" + summary + ")" if summary else ""), cited))
    out = [
        "# Results database {#results_database}",
        "",
        "Generated by `docs/tools/gen_db_pages.py` on every docs build. Do not edit.",
        "",
        "Every raw measurement grid lives in `benchmarks/results/` (Git LFS). This page",
        "is the catalogue: the campaign it belongs to, what it measured, where it was",
        "measured, when it was committed, and which evidence pages distil it. The",
        "distilled conclusions are in @ref perf_evidence; the rules for adding a grid",
        "are in @ref documentation_conventions.",
        "",
        "%d files, %d of them only an LFS pointer in the checkout this site was built" % (len(rows), pointers),
        "from (run `git lfs pull` to materialise them).",
        "",
        "| File | Campaign | Op | Type | Machine | Added | Data | Cited by |",
        "|------|----------|----|------|---------|-------|------|----------|",
    ] + rows
    return "\n".join(out) + "\n"


def evidence_page(root):
    refs = evidence.collect_references(root)
    by_page = {}
    for target, sites in refs.items():
        page, _, anchor = target.partition("#")
        by_page.setdefault(page, {}).setdefault(anchor, []).extend(sites)
    out = [
        "# Evidence index {#evidence_index}",
        "",
        "Generated by `docs/tools/gen_db_pages.py` on every docs build. Do not edit.",
        "",
        "Code comments do not carry measurements; they carry a pointer",
        "`evidence: docs/<page>.md#<section>` to the section that holds the",
        "measurement. This is the reverse map: for each cited section, every place in",
        "the code, tests and build that depends on it. Before changing or deleting a",
        "section, check what cites it here. `.github/ci/check_evidence_anchors.py`",
        "keeps every pointer resolvable.",
        "",
        "%d pointers into %d sections of %d pages." % (
            sum(len(s) for s in refs.values()), sum(len(a) for a in by_page.values()), len(by_page)),
        "",
    ]
    for page in sorted(by_page):
        out += ["## %s" % page, "", "| Section | Cited by |", "|---------|----------|"]
        for anchor in sorted(by_page[page]):
            sites = sorted(set(by_page[page][anchor]))
            label = ("`#%s`" % anchor) if anchor else "(whole page)"
            out.append("| %s | %s |" % (label, "<br>".join("`%s`" % s for s in sites)))
        out.append("")
    return "\n".join(out) + "\n"


def write(out_dir, root):
    os.makedirs(out_dir, exist_ok=True)
    pages = {"results_database.md": results_page(root), "evidence_index.md": evidence_page(root)}
    for name, text in pages.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    return pages


def self_test():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, RESULTS))
        os.makedirs(os.path.join(root, "docs", "perf"))
        os.makedirs(os.path.join(root, "src"))
        with open(os.path.join(root, RESULTS, "p2_gesv_cfloat.csv"), "w") as fh:
            fh.write("op,type,n\ngesv,cfloat,4\n")
        with open(os.path.join(root, RESULTS, "syevx_sparse_gap_rtx4090.csv"), "w") as fh:
            fh.write(LFS_HEADER + "\noid sha256:0\nsize 42\n")
        with open(os.path.join(root, "docs", "perf", "x.md"), "w") as fh:
            fh.write("# X\n\n## The window\n\nSee benchmarks/results/p2_gesv_cfloat.csv.\n")
        with open(os.path.join(root, "src", "a.cc"), "w") as fh:
            # Spelled in two halves so the evidence checker does not read the fixture.
            fh.write("// evid" + "ence: docs/perf/x.md#the-window\n")
        pages = write(os.path.join(root, "out"), root)
        res, ev = pages["results_database.md"], pages["evidence_index.md"]
        checks = [
            ("gesv row", "| `p2_gesv_cfloat.csv` | p2 | gesv | cfloat |" in res),
            ("present data", "present (1 rows" in res),
            ("citation", "`docs/perf/x.md`" in res),
            ("lfs pointer", "LFS pointer (42 bytes in LFS)" in res),
            ("machine from name", "| RTX 4090 (sm_89) |" in res),
            ("evidence reverse map", "| `#the-window` | `src/a.cc:1` |" in ev),
        ]
        bad = [name for name, ok in checks if not ok]
        if bad:
            print("gen_db_pages --self-test: FAILED: %s" % ", ".join(bad))
            return 1
    print("gen_db_pages --self-test: ok, %d checks" % len(checks))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="directory to write the generated pages into")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out is required")
    pages = write(args.out, REPO)
    print("gen_db_pages: wrote %s to %s" % (", ".join(sorted(pages)), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
