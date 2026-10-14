"""python3 -m unittest discover -s benchmarks/benchviz -p 'test_tuning.py'"""
import json
import os
import random
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tuning  # noqa: E402
from tuning import stt  # noqa: E402

CANDS = ["tiny", "cta", "lpanel:panel=8", "blocked", "vendor"]
KEYS = "uplo:exact n:log:3 batch:log"


def hashes():
    return tuning.current_hashes("potrf", list(dict.fromkeys(stt.spelling_family(c) for c in CANDS)))


def run(rid="r1", dtype="float", date="2026-10-01", batchlas="x"):
    return {"kind": "run", "dtype": dtype, "run_id": rid, "date": date, "tier": "deep", "batchlas": batchlas,
            "keys": KEYS, "candidates": "|".join(CANDS)}


def cell(key, tier, ranked, med, run="r1", date="2026-10-01", bad_hash=(), rnd=0, dtype="float", old_hash="00000000"):
    """A potrf ledger record: med = {candidate: median}; a candidate absent from med was skipped."""
    fh, rec = hashes(), {"kind": "cell", "dtype": dtype, "run_id": run, "tier": tier, "key": key, "round": rnd,
                         "date": date, "ranked": "|".join(ranked), "cands": "|".join(CANDS)}
    for i, c in enumerate(CANDS):
        fam = stt.spelling_family(c)
        rec.update({f"h{i}": old_hash if fam in bad_hash else fh[fam], f"s{i}": "ok" if c in med else "skipped",
                    f"r{i}": "", f"m{i}": med.get(c), f"lo{i}": med.get(c), f"hi{i}": med.get(c), f"n{i}": 6})
    return rec


def write_ledger(root, lines, audits=(), name="potrf.sm_120.jsonl"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lines = list(lines) + [{"kind": "audit", "dtype": d, "run_id": "r1", "key": k, "verdict": v} for d, k, v in audits]
    (root / name).write_text("".join(json.dumps(x) + "\n" for x in lines))


def write_table(tuned, rows, fam=None, batchlas="x", date="2026-10-01"):
    fam = fam or hashes()
    head = (f"# op=potrf device=sm_120 batchlas={batchlas} kernels=k family_kernels="
            + ",".join(f"{f}:{h}" for f, h in fam.items()) + f" date={date}\n"
            f"# source=ledger:x tiers=deep:{len(rows)}\n# keys: dtype:exact " + KEYS + "\n")
    (Path(tuned) / "potrf.sm_120.txt").write_text(head + "".join(r + "\n" for r in rows))


class Status(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.tuned = Path(self.tmp.name) / "ledger", Path(self.tmp.name) / "tuned"
        self.tuned.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, s, dtype="float"):
        return next(x for x in s["rows"] if x["op"] == "potrf" and x["device"] == "sm_120" and x["dtype"] == dtype)

    def test_higher_tier_wins_stale_audits_and_the_dtype_filter(self):
        write_ledger(self.root, [
            run(), run(dtype="double"),
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}, date="2026-09-01"),
            # newer, but a lower tier: never replaces the deep record
            cell("uplo=L,n=8,batch=128", "preview", ["cta", "tiny"], {"tiny": 1.0, "cta": 0.5}, run="r2", date="2026-10-05"),
            # the winner's kernels changed: no current record for this key
            cell("uplo=L,n=16,batch=128", "coarse", ["cta", "tiny"], {"tiny": 1.0, "cta": 0.5}, bad_hash={"cta"}),
            # a loser's kernels changed: partly stale, still the best record
            cell("uplo=U,n=8,batch=128", "coarse", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}, bad_hash={"cta"}),
            cell("uplo=L,n=8,batch=128", "transcribed", ["vendor"], {}, run="t", dtype="double"),
        ], audits=[("float", "uplo=L,n=8,batch=128", "ok"), ("float", "uplo=U,n=8,batch=128", "mismatch:winner tiny/cta"),
                   ("double", "uplo=L,n=8,batch=128", "mismatch:x")])
        s = tuning.Ledgers(self.root, self.tuned).status()
        r = self.row(s)
        self.assertEqual((r["cells"], r["best"], r["stale"], r["partial"]), (3, 2, 1, 1))
        self.assertEqual((r["tiers"]["deep"], r["tiers"]["coarse"], r["tiers"]["preview"]), (1, 1, 0))
        self.assertEqual(r["audits"], {"ok": 1, "mismatch": 1, "inconclusive": 0})
        self.assertIsNone(r["table"])
        d = self.row(s, "double")
        self.assertEqual((d["cells"], d["best"], d["tiers"]["transcribed"], d["rows"]["transcribed"]), (1, 1, 1, 1))
        self.assertEqual(d["audits"]["mismatch"], 1)
        fams = {f["name"]: f for f in s["files"]["potrf.sm_120"]["families"]}
        self.assertEqual(fams["cta"]["stale"], {"float": 2})  # n=16 (winner) and U n=8 (loser)
        self.assertIsNone(fams["cta"]["table"])

    def test_rehash_lines_and_dirty_runs_follow_read_ledger(self):
        old = cell("uplo=L,n=8,batch=128", "deep", ["cta", "tiny"], {"tiny": 1.0, "cta": 0.5}, bad_hash={"cta"},
                   old_hash="v1cta000")
        dirty = [run("d", batchlas="abc-dirty"), cell("uplo=U,n=8,batch=128", "deep", ["tiny"], {"tiny": 1.0}, run="d")]
        write_ledger(self.root, [run(), old] + dirty)
        r = self.row(tuning.Ledgers(self.root, self.tuned).status())
        self.assertEqual((r["best"], r["stale"], r["runs"]), (0, 1, 1))
        rehash = {"kind": "rehash", "family": "cta", "from": "v1cta000", "to": hashes()["cta"], "at": "a", "proof": "normalized"}
        write_ledger(self.root, [run(), old, rehash] + dirty)
        s = tuning.Ledgers(self.root, self.tuned).status()
        self.assertEqual((self.row(s)["best"], self.row(s)["stale"]), (1, 0))
        self.assertTrue(any("abc-dirty" in w for w in s["files"]["potrf.sm_120"]["warnings"]))

    def test_table_staleness_behind_and_sync(self):
        write_ledger(self.root, [run(), cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0})])
        stale_fam = dict(hashes(), blocked="deadbeef")
        write_table(self.tuned, ["dtype=float uplo=L n=8 batch=128 | tiny 1.000 | cta 2.000 # deep"], stale_fam)
        L = tuning.Ledgers(self.root, self.tuned)
        s = L.status()
        f = s["files"]["potrf.sm_120"]
        blocked = next(x for x in f["families"] if x["name"] == "blocked")
        self.assertEqual((blocked["table"], blocked["now"] != "deadbeef"), ("deadbeef", True))
        self.assertTrue(f["table_warnings"] and "blocked:deadbeef->" in f["table_warnings"][0])
        r = self.row(s)
        self.assertEqual((r["table"]["rows"], r["table"]["tiers"]["deep"], r["table"]["behind"]), (1, 1, False))
        self.assertEqual(L.op_view("potrf", "float", "sm_120")["table"]["differ"], 0)
        # a newer run: the header no longer names it, and its cell is a row the table lacks
        write_ledger(self.root, [run(), run("r2", date="2026-10-02"),
                                 cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}),
                                 cell("uplo=U,n=8,batch=128", "deep", ["tiny"], {"tiny": 1.0}, run="r2")])
        self.assertTrue(self.row(L.status())["table"]["behind"])
        self.assertEqual(L.op_view("potrf", "float", "sm_120")["table"]["differ"], 1)

    def test_diff_predicts_the_speedup_of_the_ledger_pick(self):
        write_ledger(self.root, [
            run(),
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta", "vendor"], {"tiny": 1.0, "cta": 3.0, "vendor": 4.0}),
            # the table's first entry (blocked) refused this cell: the table runs its second, vendor
            cell("uplo=L,n=64,batch=128", "deep", ["cta", "vendor"], {"cta": 1.0, "vendor": 2.0}),
            cell("uplo=U,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 1.5}),
        ])
        write_table(self.tuned, ["dtype=float uplo=L n=8 batch=128 | cta 3 | tiny 1 # deep",
                                 "dtype=float uplo=L n=64 batch=128 | blocked 1 | vendor 2 | cta 1 # deep",
                                 "dtype=float uplo=U n=8 batch=128 | tiny 1 | cta 1.5 # deep",
                                 "dtype=double uplo=U n=8 batch=128 | vendor 1 # deep"])
        v = tuning.Ledgers(self.root, self.tuned).op_view("potrf", "float", "sm_120")
        self.assertEqual((v["names"], v["table"]["rows"]), (["uplo", "n", "batch"], 3))
        by = {tuple(c[0]): c for c in v["cells"]}
        n8 = by[("L", 8, 128)]
        self.assertEqual((v["cands"][n8[7]], v["cands"][n8[9]], n8[8]), ("cta", "tiny", 3.0))
        n64 = by[("L", 64, 128)]
        self.assertEqual((v["cands"][n64[7]], v["cands"][n64[9]], n64[8]), ("vendor", "cta", 2.0))
        u8 = by[("U", 8, 128)]
        self.assertEqual(u8[7], u8[9])
        self.assertAlmostEqual(n8[4], 2.0)  # runner-up margin: cta 3.0 over tiny 1.0, minus 1
        # the map fills unmeasured keys from these: every entry in order, -1 for a spelling the ledger lacks
        rows = {tuple(k): [v["cands"][i] if i >= 0 else None for i in e] for k, e in v["rows"]}
        self.assertEqual(rows[("L", 8, 128)], ["tiny", "cta", "vendor"])
        trows = {tuple(k): [v["cands"][i] if i >= 0 else None for i in e] for k, e in v["table_rows"]}
        self.assertEqual(trows[("L", 64, 128)], ["blocked", "vendor", "cta"])

    def test_cell_lists_every_record_best_first(self):
        write_ledger(self.root, [
            run(),
            cell("uplo=L,n=8,batch=128", "preview", ["cta"], {"cta": 2.0}, run="r0", date="2026-08-01"),
            cell("uplo=L,n=8,batch=128", "deep", ["tiny", "cta"], {"tiny": 1.0, "cta": 2.0}),
            cell("uplo=L,n=8,batch=128", "deep", ["tiny"], {"tiny": 1.0}, dtype="double"),
        ])
        recs = tuning.Ledgers(self.root, self.tuned).cell("potrf", "float", "sm_120", ["L", 8, 128])["records"]
        self.assertEqual([(r["tier"], r["best"]) for r in recs], [("deep", True), ("preview", False)])
        self.assertEqual(recs[0]["cands"][0]["reps"], 6)

    def test_an_lfs_pointer_is_an_error_not_a_crash(self):
        self.root.mkdir()
        (self.root / "potrf.sm_120.jsonl").write_text(stt.LFS_POINTER + "\noid sha256:00\nsize 1\n")
        s = tuning.Ledgers(self.root, self.tuned).status()
        self.assertIn("git lfs pull", s["files"]["potrf.sm_120"]["error"])

    def test_the_checked_in_data_loads(self):
        """Every shipped table equals its ledger (what `sweep_to_table.py --check` proves)."""
        L = tuning.Ledgers()
        v = L.op_view("potrf", "float", "sm_120")
        if not v["cells"] and L.file("potrf", "sm_120") and L.file("potrf", "sm_120").error:
            self.skipTest("ledger not fetched (git lfs pull)")
        self.assertEqual(v["table"]["differ"], 0)
        self.assertIn("potrf", tuning.spec_ops())


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


FAKE_TUNER = """#!/bin/sh
# A stand-in batchlas_tune: --help lists the current flags; --plan writes plan events to --progress-fd.
if [ "$1" = "--help" ]; then
  echo "usage: batchlas_tune <op> --tier preview|coarse|deep --plan --progress-fd N --ledger DIR"
  echo "  --device-key sm_NN --remeasure-keys FILE --allow-dirty"
  exit 0
fi
echo "$@" > "$(dirname "$0")/argv"
fd=""; prev=""
for a in "$@"; do [ "$prev" = "--progress-fd" ] && fd="$a"; prev="$a"; done
[ -n "$fd" ] || exit 2
echo '{"ev":"plan_job","op":"potrf","dtype":"float","cells":4,"measure":3,"mix":"3 measure","est_s":1,"refine_cells":0,"est_refine_s":0}' >&"$fd"
echo '{"ev":"plan","cells":3,"est_s":1,"refine_cells":0,"est_refine_s":0,"est_total_s":1}' >&"$fd"
"""


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

    def fake_build(self, script):
        b = Path(self.tmp.name) / "build"
        (b / "tools" / "tune").mkdir(parents=True)
        p = b / "tools" / "tune" / "batchlas_tune"
        p.write_text(script)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return b, p

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
        b, keys = Path("/bin/true"), Path(self.tmp.name) / "keys.txt"
        for req in ({"ops": [], "dtypes": ["float"], "tier": "coarse", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "ultra", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "transcribed", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "coarse", "devices": []},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "custom", "devices": [0]},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "coarse", "devices": [0],
                     "remeasure_keys": "trsm float side=L"},
                    {"ops": ["potrf"], "dtypes": ["float"], "tier": "coarse", "devices": [0], "remeasure_keys": "potrf float"}):
            with self.assertRaises(ValueError):
                tuning.run_argv(b, req, Path("/l"), keys)
        argv = tuning.run_argv(b, {"ops": ["potrf", "nope"], "dtypes": ["float"], "tier": "deep", "devices": [2, 0],
                                   "max_dim": 2048, "allow_dirty": True}, Path("/l"), keys)
        self.assertEqual(argv[1:6], ["potrf", "--tier", "deep", "--dtype", "float"])
        self.assertEqual(argv[argv.index("--devices") + 1], "0,2")
        self.assertIn("--max-dim", argv)
        self.assertIn("--allow-dirty", argv)
        self.assertNotIn("--remeasure-keys", argv)
        argv = tuning.run_argv(b, {"ops": ["potrf"], "dtypes": ["float"], "tier": "custom", "devices": [],
                                   "device_key": "sm_120", "remeasure_keys": "potrf float uplo=L,n=8,batch=128\n"},
                               Path("/l"), keys, plan=True)
        self.assertEqual(argv[argv.index("--device-key") + 1], "sm_120")
        self.assertNotIn("--devices", argv)
        self.assertEqual(argv[argv.index("--remeasure-keys") + 1], str(keys))
        self.assertEqual(keys.read_text(), "potrf float uplo=L,n=8,batch=128\n")

    def test_find_tuner_says_why(self):
        self.assertIn("no build tree", tuning.find_tuner([])[1])
        self.assertIn("no build has", tuning.find_tuner([Path(self.tmp.name)])[1])
        b, p = self.fake_build("#!/bin/sh\necho 'usage: batchlas_tune <op> --devices N --raw DIR'\n")
        binary, why = tuning.find_tuner([b])
        self.assertIsNone(binary)
        self.assertIn("predates the current tuner", why)
        p.write_text(FAKE_TUNER)
        self.assertEqual(tuning.find_tuner([b]), (p, None))

    def test_plan_runs_the_tuner_with_plan_flags(self):
        _, p = self.fake_build(FAKE_TUNER)
        req = {"ops": ["potrf"], "dtypes": ["float"], "tier": "custom", "devices": [], "device_key": "sm_120",
               "remeasure_keys": "potrf float uplo=L,n=8,batch=128"}
        got = tuning.plan(p, req, Path("/l"))
        self.assertEqual((got["total"]["cells"], got["jobs"][0]["measure"]), (3, 3))
        argv = (p.parent / "argv").read_text().split()
        self.assertEqual(argv[:3], ["potrf", "--tier", "custom"])
        for flag in ("--plan", "--device-key", "--remeasure-keys", "--progress-fd", "--ledger", "--repo"):
            self.assertIn(flag, argv)

    def test_launch_writes_the_keys_into_the_run(self):
        _, p = self.fake_build(FAKE_TUNER)
        req = {"ops": ["potrf"], "dtypes": ["float"], "tier": "custom", "devices": [0],
               "remeasure_keys": "potrf float uplo=L,n=8,batch=128"}
        name = tuning.launch(Path(self.tmp.name), p, req, Path("/l"), "k")
        d = tuning.runs_dir(Path(self.tmp.name)) / name
        for _ in range(50):
            if (d / "exit").exists():
                break
            time.sleep(0.1)
        self.assertEqual((d / "remeasure-keys.txt").read_text(), "potrf float uplo=L,n=8,batch=128\n")
        self.assertEqual(tuning.run_state(Path(self.tmp.name), name)["state"], "finished")
        with self.assertRaises(ValueError):
            tuning.launch(Path(self.tmp.name), p, dict(req, remeasure_keys=""), Path("/l"), "bad")
        self.assertFalse((tuning.runs_dir(Path(self.tmp.name)) / "bad").exists())


if __name__ == "__main__":
    unittest.main()
