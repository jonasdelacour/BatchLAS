#!/usr/bin/env python3
"""Generate the database pages of the documentation site.

Three pages are derived from the tree on every docs build, so they cannot drift:

  results_database.md   one row per raw result file in benchmarks/results/:
                        campaign, op, dtype, machine, when it was committed,
                        whether this checkout holds the data or only the LFS
                        pointer, and which documentation pages cite it.
  evidence_index.md     the reverse of every `evidence: docs/<page>.md#<a>`
                        pointer: for each cited section, the code that
                        depends on it.
  selection_tables.md   every op under src/ops/: its kernel families (from
                        choice.hh), table keys and tuner grid, and for each
                        tuned/<op>.<dtype>.<device>.txt table its provenance
                        and which family ranks first over which n range.

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
    names = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in filenames:
            if not f.startswith("."):
                names.append(os.path.relpath(os.path.join(dirpath, f), folder))
    names.sort()
    added = added_commits(root)
    cites = doc_citations(root)
    rows, pointers = [], 0
    for name in names:
        path = os.path.join(folder, name)
        campaign, op, dtype, machine = classify(os.path.basename(name))
        if os.sep in name:
            campaign = name.split(os.sep)[0] + "/" + campaign
        status, summary = describe_data(path)
        pointers += status == "LFS pointer"
        date, commit = added.get(os.path.join(RESULTS, name), ("", ""))
        cited = ", ".join("`%s`" % c for c in cites.get(os.path.basename(name), [])) or "—"
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


OPS_DIR = os.path.join("src", "ops")
TUNED_DIR = "tuned"
FAMILY = re.compile(r'^struct\s+(\w+)\s*:\s*select::NoFields<"([^"]+)">\s*\{\s*\};\s*(?://\s*(.*))?$')
STRUCT = re.compile(r'^struct\s+(\w+)\s*\{(?:\s*//\s*(.*))?$')
NAMED = re.compile(r'static\s+constexpr\s+std::string_view\s+name\s*=\s*"([^"]+)"')
FIELDS = re.compile(r'static\s+constexpr\s+std::array<std::string_view,\s*\d+>\s+fields\{([^}]*)\}')
SPEC = re.compile(r'select::OpSpec\s+spec\{([^;]*)\};\s*(?://\s*(.*))?')
KEYS = re.compile(r'key_names\{([^}]*)\}', re.S)
GRID = re.compile(r'(grid_\w+)\{([^}]*)\}', re.S)
# Which evidence page records each op's measurements.
PERF_PAGE = {"posv": "potrf", "getrf": "lu", "getrs": "lu", "getri": "lu", "gesv": "lu",
             "geqrf": "qr", "orgqr": "qr", "ormqr": "qr", "symm": "level3", "syrk": "level3",
             "syr2k": "level3", "trmm": "level3", "hemm": "level3", "herk": "level3", "her2k": "level3"}


def parse_choice(path):
    """Families (spelling, struct, fields, note), spec, keys and grids of one choice.hh."""
    text = open(path, encoding="utf-8", errors="replace").read()
    fams, current = [], None
    for line in text.splitlines():
        line = line.strip()
        m = FAMILY.match(line)
        if m:
            fams.append({"name": m.group(2), "struct": m.group(1), "fields": [], "note": m.group(3) or ""})
            continue
        m = STRUCT.match(line)
        if m:
            current = {"name": "", "struct": m.group(1), "fields": [], "note": m.group(2) or ""}
            continue
        if current is not None:
            n = NAMED.search(line)
            if n:
                current["name"] = n.group(1)
            f = FIELDS.search(line)
            if f:
                current["fields"] = re.findall(r'"([^"]+)"', f.group(1))
            if line.startswith("};"):
                if current["name"]:
                    fams.append(current)
                current = None
    spec = SPEC.search(text)
    keys = KEYS.search(text)
    grids = [(g, " ".join(v.split())) for g, v in GRID.findall(text)]
    return {"families": fams,
            "spec": spec.group(1).strip() if spec else "",
            "spec_note": (spec.group(2) or "").strip() if spec else "",
            "keys": re.findall(r'"([^"]+)"', keys.group(1)) if keys else [],
            "grids": grids}


def parse_table(path):
    """Header fields, row count, timed flag and first-ranked family per row of one tuned table."""
    header, rows = {}, []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for tok in re.findall(r"(\w+)=(\S+)", line):
                header.setdefault(tok[0], tok[1])
            continue
        key, _, ranked = line.partition("|")
        entries = [e.strip() for e in ranked.split("|") if e.strip()]
        if not entries:
            continue
        first = entries[0].split()[0]
        timed = len(entries[0].split()) > 1 and entries[0].split()[1] != "-"
        n = re.search(r"(?:^|\s)n=(\d+)", key)
        rows.append((first, int(n.group(1)) if n else None, timed))
    return header, rows


def provenance(header, rows):
    src = header.get("source", "")
    if src.startswith("tuner:"):
        return "measured"
    if src.startswith("transcribed:") or (rows and not any(t for _, _, t in rows)):
        return "transcribed"
    return "converted" if rows else "empty"


def winners(rows):
    """'family xN (n a..b)' for each first-ranked family, most frequent first."""
    count, lo, hi = {}, {}, {}
    for fam, n, _ in rows:
        count[fam] = count.get(fam, 0) + 1
        if n is not None:
            lo[fam] = min(lo.get(fam, n), n)
            hi[fam] = max(hi.get(fam, n), n)
    parts = []
    for fam in sorted(count, key=lambda f: -count[f]):
        span = (" (n %d..%d)" % (lo[fam], hi[fam])) if fam in lo else ""
        parts.append("`%s` %d%s" % (fam, count[fam], span))
    return ", ".join(parts)


def perf_ref(root, op):
    page = PERF_PAGE.get(op, op)
    path = os.path.join(root, "docs", "perf", page + ".md")
    if not os.path.isfile(path):
        return ""
    first = open(path, encoding="utf-8", errors="replace").readline()
    m = re.search(r"\{#([A-Za-z0-9_]+)\}", first)
    return "@ref " + (m.group(1) if m else "md_docs_2perf_2" + page)


def selection_page(root):
    ops_dir = os.path.join(root, OPS_DIR)
    ops = sorted(d for d in os.listdir(ops_dir)
                 if os.path.isfile(os.path.join(ops_dir, d, "choice.hh"))) if os.path.isdir(ops_dir) else []
    tables = {}
    tuned = os.path.join(root, TUNED_DIR)
    for name in sorted(os.listdir(tuned)) if os.path.isdir(tuned) else []:
        parts = name.split(".")
        if len(parts) == 4 and parts[3] == "txt":
            tables.setdefault(parts[0], []).append((parts[1], parts[2], os.path.join(tuned, name)))
    out = [
        "# Kernel selection tables {#selection_tables}",
        "",
        "Generated by `docs/tools/gen_db_pages.py` on every docs build. Do not edit.",
        "",
        "Every op routes through flat kernel selection: `src/ops/<op>/choice.hh` declares the",
        "kernel *families* the op can run (and the knobs each one takes), the table keys a call",
        "is matched on, and the tuner's grid; `tuned/<op>.<dtype>.<device>.txt` ranks the families",
        "per measured shape and `select::choose` takes the first runnable entry of the nearest",
        "row. The design is @ref md_docs_2design_2flat-kernel-selection; the measurements behind",
        "each op's families are on its evidence page.",
        "",
        "**Provenance** of a table: *measured* (timed by `tools/tune`), *converted* (from timed",
        "forced-route sweeps in `benchmarks/results/routing/`) or *transcribed* (the pre-selection",
        "router's preference order, untimed). **Ranked first** counts the rows whose first-ranked",
        "family is each family, with the n range those rows cover where the table has an `n` key.",
        "",
        "%d ops, %d tables." % (len(ops), sum(len(v) for v in tables.values())),
        "",
    ]
    for op in ops:
        info = parse_choice(os.path.join(ops_dir, op, "choice.hh"))
        ref = perf_ref(root, op)
        out += ["## %s {#selection_%s}" % (op, op), ""]
        out.append("Source: `src/ops/%s/choice.hh`%s." % (op, (" · evidence: " + ref) if ref else ""))
        out.append("")
        if info["families"]:
            out += ["| Family | Knobs | Kernel |", "|--------|-------|--------|"]
            for f in info["families"]:
                out.append("| `%s` | %s | %s |" % (f["name"], ", ".join("`%s`" % x for x in f["fields"]) or "—",
                                                  f["note"].replace("|", "\\|") or "—"))
            out.append("")
        if info["keys"]:
            out.append("Table keys: %s." % ", ".join("`%s`" % k for k in info["keys"]))
        if info["spec"]:
            out.append("Spec: `OpSpec{%s}`%s" % (info["spec"], (" — " + info["spec_note"]) if info["spec_note"] else ""))
        for g, v in info["grids"]:
            out.append("Tuner `%s`: %s." % (g, v))
        out.append("")
        rows = tables.get(op, [])
        if rows:
            out += ["| dtype | device | provenance | date | rows | ranked first |",
                    "|-------|--------|------------|------|------|--------------|"]
            for dtype, dev, path in sorted(rows, key=lambda r: (r[1], r[0])):
                header, trows = parse_table(path)
                out.append("| %s | %s | %s | %s | %d | %s |" % (
                    dtype, dev, provenance(header, trows), header.get("date", "—"), len(trows), winners(trows) or "—"))
            out.append("")
        else:
            out += ["No tuned tables for this op.", ""]
    return "\n".join(out) + "\n"


def write(out_dir, root):
    os.makedirs(out_dir, exist_ok=True)
    pages = {"results_database.md": results_page(root), "evidence_index.md": evidence_page(root),
             "selection_tables.md": selection_page(root)}
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
        os.makedirs(os.path.join(root, RESULTS, "routing"))
        with open(os.path.join(root, RESULTS, "routing", "sm89_potrf_archive.jsonl"), "w") as fh:
            fh.write("{}\n")
        os.makedirs(os.path.join(root, OPS_DIR, "potrf"))
        with open(os.path.join(root, OPS_DIR, "potrf", "choice.hh"), "w") as fh:
            fh.write('struct Tiny : select::NoFields<"tiny"> {};  // registers\n'
                     'struct Lpanel {\n    static constexpr std::string_view name = "lpanel";\n'
                     '    static constexpr std::array<std::string_view, 1> fields{"panel"};\n};\n'
                     'inline constexpr select::OpSpec spec{Op::potrf, select::Lib::solver};\n'
                     'inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};\n'
                     'inline constexpr std::array<int, 3> grid_n{1, 2, 4};\n')
        os.makedirs(os.path.join(root, TUNED_DIR))
        with open(os.path.join(root, TUNED_DIR, "potrf.float.sm_89.txt"), "w") as fh:
            fh.write("# op=potrf dtype=float device=sm_89 date=2026-10-04\n# source=benchmarks/x.jsonl\n"
                     "n=1 batch=8 | tiny 0.01 | lpanel:8 0.2\nn=4 batch=8 | tiny 0.01 | lpanel:8 0.2\n"
                     "n=64 batch=8 | lpanel:8 0.1 | tiny 0.3\n")
        pages = write(os.path.join(root, "out"), root)
        res, ev = pages["results_database.md"], pages["evidence_index.md"]
        sel = pages["selection_tables.md"]
        checks = [
            ("gesv row", "| `p2_gesv_cfloat.csv` | p2 | gesv | cfloat |" in res),
            ("present data", "present (1 rows" in res),
            ("citation", "`docs/perf/x.md`" in res),
            ("lfs pointer", "LFS pointer (42 bytes in LFS)" in res),
            ("machine from name", "| RTX 4090 (sm_89) |" in res),
            ("evidence reverse map", "| `#the-window` | `src/a.cc:1` |" in ev),
            ("results subdirectory", "`routing/sm89_potrf_archive.jsonl`" in res),
            ("selection family", "| `tiny` | — | registers |" in sel),
            ("selection knobs", "| `lpanel` | `panel` |" in sel),
            ("selection keys", "Table keys: `n:log:3`, `batch:log`." in sel),
            ("selection table row", "| float | sm_89 | converted | 2026-10-04 | 3 | `tiny` 2 (n 1..4), `lpanel:8` 1 (n 64..64) |" in sel),
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
