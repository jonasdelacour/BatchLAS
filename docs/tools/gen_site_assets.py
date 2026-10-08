#!/usr/bin/env python3
"""Generate the site's navigation tree and full-text search index from Doxygen's output.

The theme (docs/theme/bl-site.js) replaces Doxygen's tree view and symbol search with a
Material-style left navigation and a section-level full-text search. Both need data that
only exists after Doxygen has run, so build_docs.sh calls this on the finished site:

  bl-nav.js     window.BL_NAV: the page tree (\\subpage order) plus the API group tree.
  bl-search.js  window.BL_SEARCH: one record per section of every page, group and class.
  *.html        asset URLs get ?v=<content hash> so a redeploy is never masked by a
                browser's cached copy of the previous stylesheet.

Usage:
    python3 docs/tools/gen_site_assets.py --site <out>     (<out>/html and <out>/xml)
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys

PAGE_KINDS = ("page", "group")
API_KINDS = ("class", "struct", "union", "namespace", "concept")
VERSIONED = ("batchlas.css", "bl-site.js", "bl-nav.js")
# Pages whose Doxygen title is a file name rather than a heading.
RENAMES = {"README": "Overview and install"}

SECTION_HEAD = re.compile(
    r'<a class="anchor" id="([^"]+)"></a>\s*(?:</p>\s*)?'
    r'<h([1-4])[^>]*>(?:\s*<a class="anchor" id="[^"]+"></a>)?(.*?)</h\2>', re.S)
INLINE_HEAD = re.compile(r'<h([1-4]) class="doxsection"><a class="anchor" id="([^"]+)"></a>(.*?)</h\1>', re.S)
GROUP_HEAD = re.compile(r'<h2 class="groupheader">(?:<a id="([^"]+)" name="[^"]+"></a>)?(.*?)</h2>', re.S)
MEMBER_HEAD = re.compile(r'<a id="([^"]+)" name="[^"]+"></a>\s*<h2 class="memtitle">(.*?)</h2>', re.S)
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
SKIP_BLOCKS = re.compile(r'<div class="ttc"[^>]*>.*?</div></div>|<div class="dyn(?:header|content)".*?</div>'
                         r'|<script.*?</script>', re.S)


def text_of(fragment):
    return WS.sub(" ", html.unescape(TAG.sub(" ", fragment))).strip()


def read_xml(xml_dir, refid):
    path = os.path.join(xml_dir, refid + ".xml")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def compounds(xml_dir):
    with open(os.path.join(xml_dir, "index.xml"), encoding="utf-8") as fh:
        index = fh.read()
    return re.findall(r'<compound refid="([^"]+)" kind="([^"]+)"><name>([^<]*)</name>', index)


def title_of(xml):
    m = re.search(r"<title>(.*?)</title>", xml, re.S)
    return text_of(m.group(1)) if m else ""


def html_name(refid):
    return "index.html" if refid == "indexpage" else refid + ".html"


def build_tree(xml_dir, refid, child_tag, seen):
    if refid in seen:
        return None
    seen.add(refid)
    xml = read_xml(xml_dir, refid)
    title = title_of(xml) or refid
    node = {"t": RENAMES.get(title, title), "u": html_name(refid)}
    kids = [build_tree(xml_dir, c, child_tag, seen)
            for c in re.findall(r"<%s refid=\"([^\"]+)\"" % child_tag, xml)]
    kids = [k for k in kids if k]
    if kids:
        node["c"] = kids
    return node


def build_nav(xml_dir):
    seen = set()
    root = build_tree(xml_dir, "indexpage", "innerpage", seen)
    nav = [{"t": "Home", "u": "index.html"}] + root.get("c", [])
    api = build_tree(xml_dir, "group__api", "innergroup", set())
    if api:
        api["t"] = "API reference"
        api.setdefault("c", []).extend([
            {"t": "All classes", "u": "annotated.html"},
            {"t": "All files", "u": "files.html"},
        ])
        nav.insert(2, api)
    return nav


def page_title(doc):
    m = re.search(r'<div class="title">(.*?)(?:<div class="ingroups">|</div>)', doc, re.S)
    return text_of(m.group(1)) if m else ""


def contents_of(doc):
    start = doc.find('<div class="contents">')
    end = doc.rfind("<!-- start footer part -->")
    if start < 0:
        return ""
    return SKIP_BLOCKS.sub(" ", doc[start:end if end > start else len(doc)])


def sections(doc, name):
    """Split one page into (anchor, heading, text) records in document order."""
    body = contents_of(doc)
    cuts = []
    for rx, idg, tg in ((INLINE_HEAD, 2, 3), (GROUP_HEAD, 1, 2), (MEMBER_HEAD, 1, 2)):
        for m in rx.finditer(body):
            cuts.append((m.start(), m.end(), m.group(idg) or "", text_of(m.group(tg)).lstrip("\u25c6 ")))
    cuts.sort()
    out = []
    prev_end, prev_id, prev_title = 0, "", ""
    for start, end, anchor, title in cuts:
        out.append((prev_id, prev_title, text_of(body[prev_end:start])))
        prev_end, prev_id, prev_title = end, anchor, title
    out.append((prev_id, prev_title, text_of(body[prev_end:])))
    return [(a, t, x) for a, t, x in out if x or t]


def build_search(html_dir, xml_dir):
    records = []
    for refid, kind, _ in compounds(xml_dir):
        if kind not in PAGE_KINDS + API_KINDS:
            continue
        name = html_name(refid)
        path = os.path.join(html_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            doc = fh.read()
        ptitle = page_title(doc) or refid
        for anchor, title, text in sections(doc, name):
            records.append({
                "p": ptitle,
                "t": title,
                "u": name + ("#" + anchor if anchor else ""),
                "x": text[:4000],
                "k": 1 if kind in API_KINDS else 0,
            })
    return records


def write_js(path, var, data):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.%s=" % var)
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")


def version_assets(html_dir):
    tags = {}
    for name in VERSIONED:
        path = os.path.join(html_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                tags[name] = hashlib.sha1(fh.read()).hexdigest()[:10]
    rx = re.compile(r'((?:href|src)=")(%s)(")' % "|".join(re.escape(n) for n in tags))
    for name in os.listdir(html_dir):
        if not name.endswith(".html"):
            continue
        path = os.path.join(html_dir, name)
        with open(path, encoding="utf-8", errors="replace") as fh:
            doc = fh.read()
        new = rx.sub(lambda m: "%s%s?v=%s%s" % (m.group(1), m.group(2), tags[m.group(2)], m.group(3)), doc)
        if new != doc:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True, help="Doxygen output dir holding html/ and xml/")
    args = ap.parse_args()
    html_dir = os.path.join(args.site, "html")
    xml_dir = os.path.join(args.site, "xml")
    search = build_search(html_dir, xml_dir)
    search_js = os.path.join(html_dir, "bl-search.js")
    write_js(search_js, "BL_SEARCH", search)
    with open(search_js, "rb") as fh:
        search_v = hashlib.sha1(fh.read()).hexdigest()[:10]
    nav = build_nav(xml_dir)
    nav_js = os.path.join(html_dir, "bl-nav.js")
    write_js(nav_js, "BL_NAV", nav)
    with open(nav_js, "a", encoding="utf-8") as fh:
        fh.write('window.BL_SEARCH_V="%s";\n' % search_v)
    tags = version_assets(html_dir)
    size = os.path.getsize(os.path.join(html_dir, "bl-search.js"))
    print("gen_site_assets: %d nav roots, %d search sections (%.1f MB), versioned %s"
          % (len(nav), len(search), size / 1e6, ", ".join(sorted(tags))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
