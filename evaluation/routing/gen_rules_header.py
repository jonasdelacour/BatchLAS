#!/usr/bin/env python3
"""Generate src/routing/generated/<op>_rules.inc from routing/rules/*/<op>.rules.

    python3 evaluation/routing/gen_rules_header.py [--op potrf|getrs|all] [--check]

One rules grammar for every op; only SCHEMA differs (the categorical key and the numeric
features, in Features order):

    R0001 <dtype> <key2> <f0> [lo,hi] <f1> [lo,hi] ... -> tierA > tierB ; src=<source>

Overrides (routing/overrides/<arch>/<op>.rules, ids O0001, evidence=<pointer> required) are
placed ahead of the generated rules of the same key, so first-match lets them win.
--check regenerates in memory and fails if a committed .inc differs (a ctest). The arch index
maps each routing profile to its model RuleSet when the profile's gate passed and a rules
file exists, and to the hand RuleSet otherwise.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RULES = os.path.join(REPO, "routing", "rules")
OVERRIDES = os.path.join(REPO, "routing", "overrides")
DTYPES = ("float", "double", "cfloat", "cdouble")
PROFILES = ("sm_89", "sm_120")   # arch::RoutingProfile, Unset excluded
OVERRIDE_ID_BASE = 90000          # O0001 prints as R90001
SRC = {"hand": "Hand", "model": "Model", "measured": "Measured", "override": "Override"}
SCHEMA = {
    "potrf": {"key2": ("L", "U"), "features": ("n", "batch")},
    "getrs": {"key2": ("N", "T", "C"), "features": ("n", "nrhs", "batch")},
}
HEAD = re.compile(r"^([RO])(\d+) (\w+) (\w+) (.+?) -> (.+?) ; src=(\w+)( evidence=\S+)?$")
BOX = re.compile(r"(\w+) \[(\d+),(\w+)\]")


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
        box = {f: (int(lo), hi) for f, lo, hi in BOX.findall(boxes)}
        if set(box) != set(sch["features"]):
            sys.exit(f"gen_rules_header: {path}: features {sorted(box)} != schema "
                     f"{list(sch['features'])}: {line}")
        tiers = [t.strip() for t in rank.split(">")]
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


def lim(v):
    return "kInf" if v == "inf" else str(int(v))


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
    cf = [meta.get(k, "0") for k in ("cus", "local_mem", "max_wg", "sg")]
    o.append(f"inline constexpr RuleSet k_{tag}{{\"{op}\", \"{meta.get('arch', tag)}\", \"{prov}\",")
    o.append(f"    k_{tag}_rules, k_{tag}_rules + {len(rules)}, k_{tag}_off, {nkeys}, "
             f"{{{', '.join(cf)}}}}};")


def generate(op):
    sch = SCHEMA[op]
    nkeys = len(DTYPES) * len(sch["key2"])
    cand = json.load(open(os.path.join(REPO, "routing", "candidacy.json")))[op]
    names = list(cand["tiers"])
    pin_only = cand.get("pinnable_only", [])
    sets = {}
    for arch in sorted(os.listdir(RULES)):
        p = os.path.join(RULES, arch, op + ".rules")
        if not os.path.exists(p):
            continue
        sets[arch] = parse(p, names, sch)
        ov = os.path.join(OVERRIDES, arch, op + ".rules")
        if os.path.exists(ov):
            extra, _ = parse(ov, names, sch)
            sets[arch] = (sorted(extra + sets[arch][0],
                                 key=lambda r: (r["key"], r["order"], r["id"])), sets[arch][1])
    if "hand" not in sets:
        sys.exit(f"gen_rules_header: routing/rules/hand/{op}.rules is required")
    o = [f"// GENERATED by evaluation/routing/gen_rules_header.py from routing/rules/*/{op}.rules",
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
    for arch, (rules, meta) in sets.items():
        emit_set(o, op, arch, rules, meta, names, nkeys)
        o.append("")
    o.append("// The ONE arch index: a profile whose gate passed and that has a model rules file")
    o.append("// gets it; every other profile, and Unset, gets the hand transcription.")
    o.append("inline const RuleSet& rules_for_profile(arch::RoutingProfile p) {")
    o.append("    switch (p) {")
    for prof in PROFILES:
        pj = json.load(open(os.path.join(HERE, "profiles", prof + ".json")))
        gate = pj["gate"]["verdict"] if pj.get("op") == op else "n/a (profile is " + pj.get("op", "?") + ")"
        use = prof if (gate == "PASS" and prof in sets) else "hand"
        o.append(f"        case arch::RoutingProfile::{prof}: return k_{use};   // gate {gate}")
    o.append("        default: return k_hand;")
    o.append("    }")
    o.append("}")
    o.append("")
    o.append(f"}}  // namespace batchlas::routing::{op}_rules")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="all")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ops = [op for op in SCHEMA if os.path.exists(os.path.join(RULES, "hand", op + ".rules"))] \
        if a.op == "all" else [a.op]
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
        print("generated rules headers are current: " + " ".join(ops))


if __name__ == "__main__":
    main()
