#!/usr/bin/env python3
"""Generate src/routing/generated/<op>_rules.inc from routing/rules/*/<op>[.hand].rules.

    python3 evaluation/routing/gen_rules_header.py [--op potrf|getrs|all] [--check]

One rules grammar for every op; only SCHEMA differs (the categorical key and the numeric
features, in Features order):

    R0001 <dtype> <key2> <f0> [lo,hi] <f1> [lo,hi] ... -> tierA > tierB ; src=<source>
    R0002 <dtype> <key2> ...                           -> live ; src=<source>

`live` marks a contested box: the policy changes inside it, so the engine prices at run time.

Layout (no C++ edit to add an arch -- the engine looks sets up by the arch string at run time):
    routing/rules/<arch>/<op>.rules       model rules for <arch> (its gate passed)
    routing/rules/<arch>/<op>.hand.rules  today's hand windows, sampled with <arch>'s facts
    routing/rules/hand/<op>.rules         facts-independent hand rules (no compiled_for)
Overrides (routing/overrides/<arch>/<op>.rules, ids O0001, evidence=<pointer> required) are
placed ahead of that arch's model rules of the same key, so first-match lets them win.

Every set is proven to cover the whole feature domain (a hole is an error, not a fallback),
and a model set must name the current sha of the profile it was compiled from. --check
regenerates in memory and fails if a committed .inc differs (a ctest).
"""

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RULES = os.path.join(REPO, "routing", "rules")
OVERRIDES = os.path.join(REPO, "routing", "overrides")
DTYPES = ("float", "double", "cfloat", "cdouble")
OVERRIDE_ID_BASE = 90000          # O0001 prints as R90001
SRC = {"hand": "Hand", "model": "Model", "measured": "Measured", "override": "Override"}
SCHEMA = {
    "potrf": {"key2": ("L", "U"), "features": ("n", "batch")},
    "getrs": {"key2": ("N", "T", "C"), "features": ("n", "nrhs", "batch")},
}
HEAD = re.compile(r"^([RO])(\d+) (\w+) (\w+) (.+?) -> (.+?) ; src=(\w+)( evidence=\S+)?$")
BOX = re.compile(r"(\w+) \[(\d+),(\w+)\]")
INF = 1 << 62
FACTS = ("cus", "local_mem", "max_wg", "sg")


def parse(path, names, sch):
    rules, meta = [], {}
    for raw in open(path):
        line = raw.rstrip("\n")
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta.setdefault(k, v)
            continue
        if not line.strip():
            continue
        m = HEAD.match(line)
        if not m:
            sys.exit(f"gen_rules_header: {path}: cannot parse: {line}")
        kind, rid, d, k2, boxes, rank, src, ev = m.groups()
        if kind == "O" and not ev:
            sys.exit(f"gen_rules_header: {path}: an override needs evidence=<docs pointer>: {line}")
        box = {f: (int(lo), INF if hi == "inf" else int(hi)) for f, lo, hi in BOX.findall(boxes)}
        if set(box) != set(sch["features"]):
            sys.exit(f"gen_rules_header: {path}: features {sorted(box)} != schema "
                     f"{list(sch['features'])}: {line}")
        tiers = [t.strip() for t in rank.split(">")]
        if tiers == ["live"]:
            tiers = []
        elif "live" in tiers:
            sys.exit(f"gen_rules_header: {path}: `live` must stand alone: {line}")
        for t in tiers:
            if t not in names:
                names.append(t)
        rules.append({"id": int(rid) + (OVERRIDE_ID_BASE if kind == "O" else 0),
                      "key": DTYPES.index(d) * len(sch["key2"]) + sch["key2"].index(k2),
                      "order": 0 if kind == "O" else 1,
                      "lo": [box[f][0] for f in sch["features"]],
                      "hi": [box[f][1] for f in sch["features"]], "rank": tiers, "src": src})
    rules.sort(key=lambda r: (r["key"], r["order"], r["id"]))
    return rules, meta


def hole(rules, nf):
    """A point of [1, inf)^nf no box contains, or None. Between consecutive box edges the set
    of boxes containing a coordinate is constant, so checking every edge is a proof."""
    def rec(rs, d):
        pts = sorted({1} | {r["lo"][d] for r in rs} | {r["hi"][d] + 1 for r in rs
                                                       if r["hi"][d] < INF})
        for p in pts:
            sub = [r for r in rs if r["lo"][d] <= p <= r["hi"][d]]
            if not sub:
                return [p]
            if d + 1 < nf:
                bad = rec(sub, d + 1)
                if bad is not None:
                    return [p] + bad
        return None
    return rec(rules, 0)


def lim(v):
    return "kInf" if v >= INF else str(int(v))


def emit_set(o, op, tag, rules, meta, names, nkeys):
    o.append(f"inline constexpr Rule k_{tag}_rules[] = {{")
    for r in rules:
        rank = ", ".join(f"Candidate{{{names.index(t)}, {{}}}}" for t in r["rank"])
        lo = list(map(str, r["lo"])) + ["0"] * (4 - len(r["lo"]))
        hi = [lim(x) for x in r["hi"]] + ["0"] * (4 - len(r["hi"]))
        o.append(f"    {{{r['id']}u, {r['key']}, {{{', '.join(lo)}}}, {{{', '.join(hi)}}}, "
                 f"{{{rank}}}, {len(r['rank'])}, Source::{SRC[r['src']]}}},")
    o.append("};")
    off = [0] * (nkeys + 1)
    for r in rules:
        off[r["key"] + 1] += 1
    for k in range(nkeys):
        off[k + 1] += off[k]
    o.append(f"inline constexpr std::uint32_t k_{tag}_off[] = {{{', '.join(map(str, off))}}};")
    prov = meta.get("provenance", "").replace('"', "'")
    cf = [meta.get(k, "0") for k in FACTS]
    kind = SRC[meta.get("source", "hand")]
    o.append(f"inline constexpr RuleSet k_{tag}{{\"{op}\", \"{meta.get('arch', tag)}\", \"{prov}\",")
    o.append(f"    k_{tag}_rules, k_{tag}_rules + {len(rules)}, k_{tag}_off, {nkeys}, "
             f"{{{', '.join(cf)}}}, Source::{kind}}};")


def check_provenance(path, meta):
    """A model set must have been compiled from the profile that ships today."""
    prov = meta.get("provenance", "")
    m = re.match(r"profiles/(\w+)\.json@([0-9a-f]+);", prov)
    if not m:
        sys.exit(f"gen_rules_header: {path}: a model set needs provenance=profiles/<arch>.json@<sha>")
    prof = os.path.join(HERE, "profiles", m.group(1) + ".json")
    sha = hashlib.sha256(open(prof, "rb").read()).hexdigest()
    if not sha.startswith(m.group(2)):
        sys.exit(f"gen_rules_header: {path} was compiled from {m.group(1)}.json@{m.group(2)} but "
                 f"the profile is now @{sha[:12]}: re-run compile_rules.py")


def generate(op):
    sch = SCHEMA[op]
    nkeys = len(DTYPES) * len(sch["key2"])
    cand = json.load(open(os.path.join(REPO, "routing", "candidacy.json")))[op]
    names = list(cand["tiers"])
    pin_only = cand.get("pinnable_only", [])
    sets = {}
    for d in sorted(os.listdir(RULES)):
        base = re.sub(r"\W", "_", d)
        for fname, tag in ((op + ".rules", base), (op + ".hand.rules", base + "_hand")):
            p = os.path.join(RULES, d, fname)
            if not os.path.exists(p):
                continue
            rules, meta = parse(p, names, sch)
            if meta.get("source") == "model":
                check_provenance(p, meta)
            ov = os.path.join(OVERRIDES, d, op + ".rules")
            if fname == op + ".rules" and os.path.exists(ov):
                extra, _ = parse(ov, names, sch)
                rules = sorted(extra + rules, key=lambda r: (r["key"], r["order"], r["id"]))
            for k in range(nkeys):
                bad = hole([r for r in rules if r["key"] == k], len(sch["features"]))
                if bad is not None:
                    sys.exit(f"gen_rules_header: {os.path.relpath(p, REPO)} key {k} has a hole at "
                             f"{dict(zip(sch['features'], bad))}")
            sets[tag] = (rules, meta)
    if not any(m.get("source") == "hand" for _, m in sets.values()):
        sys.exit(f"gen_rules_header: no hand rules for {op}")
    o = [f"// GENERATED by evaluation/routing/gen_rules_header.py from routing/rules/*/{op}*.rules",
         "// -- do not edit; regenerate. tests/CMakeLists.txt checks it (--check).",
         "#pragma once",
         "",
         f"namespace batchlas::routing::{op}_rules {{",
         "",
         f"// key = dtype * {len(sch['key2'])} + index of {list(sch['key2'])}; "
         f"features = {list(sch['features'])}",
         f"inline constexpr std::array<std::string_view, {len(names)}> kNames{{"
         + ", ".join(f'"{n}"' for n in names) + "};",
         "// Registered tiers that no rule ranks ON PURPOSE (pinnable, never auto-chosen).",
         f"inline constexpr std::array<std::string_view, {len(pin_only)}> kPinnableOnly{{"
         + ", ".join(f'"{n}"' for n in pin_only) + "};",
         ""]
    for tag, (rules, meta) in sets.items():
        emit_set(o, op, tag, rules, meta, names, nkeys)
        o.append("")
    o.append("// Every set; routing::pick_rules chooses by arch string and device facts at run time.")
    o.append(f"inline constexpr const RuleSet* kSets[] = {{"
             + ", ".join(f"&k_{t}" for t in sets) + "};")
    o.append("")
    o.append(f"}}  // namespace batchlas::routing::{op}_rules")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="all")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ops = [op for op in SCHEMA
           if any(os.path.exists(os.path.join(RULES, d, f)) for d in os.listdir(RULES)
                  for f in (op + ".rules", op + ".hand.rules"))] if a.op == "all" else [a.op]
    stale = []
    for op in ops:
        out = os.path.join(REPO, "src", "routing", "generated", op + "_rules.inc")
        text = generate(op)
        if a.check:
            cur = open(out).read() if os.path.exists(out) else ""
            if cur != text:
                stale.append(os.path.relpath(out, REPO))
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(out, REPO)}")
    if stale:
        sys.exit("gen_rules_header: stale, regenerate: " + " ".join(stale))
    if a.check:
        print("generated rules headers are current; every set covers its domain: " + " ".join(ops))


if __name__ == "__main__":
    main()
