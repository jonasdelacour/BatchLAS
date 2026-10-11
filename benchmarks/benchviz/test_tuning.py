"""python3 -m unittest discover -s benchmarks/benchviz -p 'test_tuning.py'"""
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tuning  # noqa: E402
from tuning import stt  # noqa: E402

CANDS = ["tiny", "cta", "lpanel:panel=8", "blocked", "vendor"]  # potrf float, choice.hh order


def hashes():
    block = stt.parse_kernel_block(stt.read_spec_source(stt.OP_BY_NAME["potrf"]))
    return stt.family_hashes(stt.REPO, block, list(dict.fromkeys(stt.spelling_family(c) for c in CANDS)))


def cell(key, tier, ranked, med, run="r1", date="2026-10-01", status=None, bad_hash=(), rnd=0):
    """A potrf ledger record: med = {candidate: median}; a candidate absent from med was skipped."""
    fh, rec = hashes(), {"kind": "cell", "run_id": run, "tier": tier, "key": key, "round": rnd, "date": date,
                         "ranked": "|".join(ranked), "cands": "|".join(CANDS)}
    for i, c in enumerate(CANDS):
        fam = stt.spelling_family(c)
        st = (status or {}).get(c) or ("ok" if c in med else "skipped")
        rec.update({f"h{i}": "00000000" if fam in bad_hash else fh[fam], f"s{i}": st, f"r{i}": "",
                    f"m{i}": med.get(c), f"lo{i}": med.get(c), f"hi{i}": med.get(c), f"n{i}": 6})
    return rec


def write_ledger(root, recs, audits=()):
    d = Path(root) / "potrf.float.sm_120"
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"kind": "run", "run_id": "r1", "date": "2026-10-01", "tier": "deep", "batchlas": "x",
              "keys": stt.OP_BY_NAME["potrf"].keys, "candidates": "|".join(CANDS)}] + list(recs) + \
        [{"kind": "audit", "run_id": "r1", "key": k, "verdict": v} for k, v in audits]
    (d / "r1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


class Status(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.tuned = Path(self.tmp.name) / "ledger", Path(self.tmp.name) / "tuned"
        self.tuned.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_higher_tier_wins_and_stale_and_audits_count(self):
        write_ledger(self.root, [
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}, date="2026-09-01"),
            # newer, but a lower tier: never replaces the deep record
            cell("uplo=L,n=8,batch=128", "preview", ["cta", "tiny"], {"tiny": 1.0, "cta": 0.5}, run="r2", date="2026-10-05"),
            # the winner's kernels changed: no current record for this key
            cell("uplo=L,n=16,batch=128", "coarse", ["cta", "tiny"], {"tiny": 1.0, "cta": 0.5}, bad_hash={"cta"}),
            # a loser's kernels changed: partly stale, still the best record
            cell("uplo=U,n=8,batch=128", "coarse", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}, bad_hash={"cta"}),
        ], audits=[("uplo=L,n=8,batch=128", "ok"), ("uplo=U,n=8,batch=128", "mismatch:winner tiny/cta")])
        s = tuning.Ledgers(self.root, self.tuned).status()
        r = next(x for x in s["rows"] if x["op"] == "potrf" and x["device"] == "sm_120")
        self.assertEqual((r["cells"], r["best"], r["stale"], r["partial"]), (3, 2, 1, 1))
        self.assertEqual((r["tiers"]["deep"], r["tiers"]["coarse"], r["tiers"]["preview"]), (1, 1, 0))
        self.assertEqual(r["audits"], {"ok": 1, "mismatch": 1, "inconclusive": 0})
        self.assertIsNone(r["table"])

    def test_diff_predicts_the_speedup_of_the_ledger_pick(self):
        write_ledger(self.root, [
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta", "vendor"], {"tiny": 1.0, "cta": 3.0, "vendor": 4.0}),
            # the table's first entry (blocked) refused this cell: the table runs its second, vendor
            cell("uplo=L,n=64,batch=128", "deep", ["cta", "vendor"], {"cta": 1.0, "vendor": 2.0}),
            cell("uplo=U,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 1.5}),
        ])
        (self.tuned / "potrf.float.sm_120.txt").write_text(
            "# op=potrf dtype=float device=sm_120 source=ledger:x date=2026-09-01\n# keys: uplo:exact n:log:3 batch:log\n"
            "uplo=L n=8 batch=128 | cta 3 | tiny 1\n"
            "uplo=L n=64 batch=128 | blocked 1 | vendor 2 | cta 1\n"
            "uplo=U n=8 batch=128 | tiny 1 | cta 1.5\n")
        v = tuning.Ledgers(self.root, self.tuned).op_view("potrf", "float", "sm_120")
        by = {tuple(c[0]): c for c in v["cells"]}
        n8 = by[("L", 8, 128)]
        self.assertEqual((v["cands"][n8[7]], v["cands"][n8[9]], n8[8]), ("cta", "tiny", 3.0))
        n64 = by[("L", 64, 128)]
        self.assertEqual((v["cands"][n64[7]], v["cands"][n64[9]], n64[8]), ("vendor", "cta", 2.0))
        u8 = by[("U", 8, 128)]
        self.assertEqual(u8[7], u8[9])
        self.assertAlmostEqual(n8[4], 2.0)  # runner-up margin: cta 3.0 over tiny 1.0, minus 1

    def test_cell_lists_every_record_best_first(self):
        write_ledger(self.root, [
            cell("uplo=L,n=8,batch=128", "preview", ["cta"], {"cta": 2.0}, run="r0", date="2026-08-01"),
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}),
        ])
        recs = tuning.Ledgers(self.root, self.tuned).cell("potrf", "float", "sm_120", ["L", 8, 128])["records"]
        self.assertEqual([(r["tier"], r["best"]) for r in recs], [("deep", True), ("preview", False)])
        self.assertEqual(recs[0]["cands"][0]["reps"], 6)


class NearestMatchesTheReference(unittest.TestCase):
    def test_random_queries_including_missing_exact_keys(self):
        keyspec = stt.parse_keys("side:exact trans:exact order:log:2 q:log batch:log")
        rng = random.Random(7)
        rows = [((rng.choice("LR"), rng.choice("NT"), 2 ** rng.randint(1, 9), 2 ** rng.randint(0, 8), 2 ** rng.randint(7, 15)),
                 [(f"c{i}", None)]) for i in range(300)]
        rows = [r for r in rows if r[0][0] == "L" or r[0][1] == "N"]  # (R, T) absent: drops to the side-only pool
        qs = [(rng.choice("LR"), rng.choice("NT"), rng.randint(1, 700), rng.randint(1, 300), rng.randint(64, 40000))
              for _ in range(500)]
        got = tuning.Nearest(rows, keyspec).lookup(qs)
        self.assertEqual(got, [stt.nearest(rows, q, keyspec)[1] for q in qs])


class Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = tuning.runs_dir(Path(self.tmp.name)) / "r"
        self.d.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def events(self, evs, pid=None):
        (self.d / "run.json").write_text(json.dumps({"pid": pid, "argv": ["x"], "request": {"devices": [0]}, "started": 1.0}))
        (self.d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))

    def test_fold(self):
        k = {"op": "potrf", "dtype": "float", "uplo": "L", "n": 8, "batch": 128}
        self.events([{"ev": "plan_job", "op": "potrf", "dtype": "float", "cells": 10, "measure": 6, "refine_cells": 3,
                      "est_s": 5, "est_refine_s": 2, "mix": "6 measure, 4 skip:current", "t": 1},
                     {"ev": "plan", "cells": 6, "refine_cells": 3, "est_total_s": 7, "t": 1},
                     {"ev": "cell_start", **k, "gpu": 0, "t": 2}, {"ev": "eliminated", **k, "cand": "blocked", "round": 2, "t": 3},
                     {"ev": "cell_done", **k, "ranked": "tiny|cta", "tier": "deep", "round": 1, "t": 4},
                     {"ev": "audit", **k, "verdict": "mismatch:winner tiny/cta", "t": 5}])
        (self.d / "exit").write_text("0\n")
        s = tuning.run_state(Path(self.tmp.name), "r")
        j = s["jobs"][0]
        self.assertEqual((s["state"], s["done"], s["total"]), ("finished", 1, 9))
        self.assertEqual((j["done"], j["refined"], j["eliminated"], j["audit_bad"], j["measure"]), (1, 1, 1, 1, 6))
        self.assertEqual(s["gpus"], [])
        cellfeed = [f for f in s["feed"] if f["kind"] == "cell"][0]
        self.assertEqual((cellfeed["win"], cellfeed["eliminated"], cellfeed["key"]), ("tiny", 1, "uplo=L n=8 batch=128"))
        self.assertEqual(s["feed"][0]["kind"], "audit")

    def test_states(self):
        self.events([], pid=None)
        self.assertEqual(tuning.RunState(self.d).snapshot()["state"], "interrupted")
        (self.d / "stop").write_text("1")
        self.assertEqual(tuning.RunState(self.d).snapshot()["state"], "stopped")
        (self.d / "exit").write_text("3")
        self.assertEqual(tuning.RunState(self.d).snapshot()["state"], "stopped")
        (self.d / "stop").unlink()
        self.assertEqual(tuning.RunState(self.d).snapshot()["state"], "failed")

    def test_a_dead_child_is_not_alive(self):
        p = subprocess.Popen(["true"])
        tuning._children[p.pid] = p
        time.sleep(0.5)  # exited, not reaped: a zombie
        self.assertFalse(tuning._alive(p.pid))  # the zombie case: os.kill(pid, 0) alone says alive

    def test_bad_requests(self):
        b = Path("/bin/true")
        for req in ({"ops": [], "dtypes": ["float"], "tier": "coarse", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "ultra", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "coarse", "devices": []}):
            with self.assertRaises(ValueError):
                tuning.run_argv(b, req, Path("/l"))
        argv = tuning.run_argv(b, {"ops": ["potrf", "nope"], "dtypes": ["float"], "tier": "deep", "devices": [2, 0],
                                   "max_dim": 2048}, Path("/l"))
        self.assertEqual(argv[1:8], ["potrf", "--tier", "deep", "--dtype", "float", "--devices", "0,2"])
        self.assertIn("--max-dim", argv)


if __name__ == "__main__":
    unittest.main()
