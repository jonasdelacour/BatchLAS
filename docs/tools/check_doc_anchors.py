#!/usr/bin/env python3
"""Check that every cited evidence anchor is a live link in the generated site.

.github/ci/check_evidence_anchors.py proves that `evidence: docs/<page>.md#<a>`
names a heading under GitHub's slug rules. This proves the other half: that
Doxygen gave the same heading the same id. Doxygen's GITHUB id style matches
GitHub's slug for a heading whose slug is unique across the whole site, but it
numbers repeats site-wide (`what-ships-5`), where GitHub numbers per page. The
evidence checker therefore also requires cited slugs to be site-unique; this
script is the end-to-end proof, run against Doxygen's XML output.

Usage:
    python3 docs/tools/check_doc_anchors.py --xml build/docs/xml
Exit code 0 = every cited anchor resolves, 1 = findings, 2 = bad invocation.
"""

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, ".github", "ci"))
import check_evidence_anchors as evidence  # noqa: E402

COMPOUND = re.compile(r'<compoundname>([^<]+)</compoundname>')
LOCATION = re.compile(r'<location file="([^"]+)"')
SECTION = re.compile(r'<sect\d id="([^"]+)"')


def page_ids(xml_dir):
    """Map repo-relative .md path -> set of section ids Doxygen generated."""
    pages = {}
    for name in os.listdir(xml_dir):
        if not name.endswith(".xml") or name in ("index.xml", "combine.xslt"):
            continue
        with open(os.path.join(xml_dir, name), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if 'kind="page"' not in text[:2000]:
            continue
        loc = LOCATION.search(text)
        if not loc:
            continue
        path = loc.group(1)
        rel = os.path.relpath(path, REPO) if os.path.isabs(path) else path
        ids = set()
        for full in SECTION.findall(text):
            ids.add(full.split("_1", 1)[1] if "_1" in full else full)
        pages.setdefault(rel, set()).update(ids)
    return pages


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xml", required=True, help="Doxygen XML output directory")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.xml):
        print(f"check_doc_anchors: no XML directory at {args.xml}", file=sys.stderr)
        return 2
    pages = page_ids(args.xml)
    findings = []
    checked = 0
    for ref, sites in sorted(evidence.collect_references(REPO).items()):
        path, _, anchor = ref.partition("#")
        if not anchor or not path.endswith(".md"):
            continue
        checked += 1
        if path not in pages:
            findings.append(f"{ref}: page is not part of the site (cited at {sites[0]})")
        elif anchor not in pages[path]:
            near = sorted(i for i in pages[path] if i.startswith(anchor))[:3]
            findings.append(f"{ref}: Doxygen id differs {near} (cited at {sites[0]})")
    for f in findings:
        print(f)
    print(f"check_doc_anchors: {checked} cited anchors, {len(findings)} not live in the site")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
