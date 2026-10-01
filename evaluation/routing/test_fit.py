#!/usr/bin/env python3
"""Recovery tests for fit.py:  python3 evaluation/routing/test_fit.py [-v]

The pipeline cases need build/tests/potrf_plan_dump (or $POTRF_PLAN_DUMP) and skip without it.
Synthetic times are generated from KNOWN constants through the C++ plan terms; the test is
that the fitter gets the constants and the routes back from held-out orders.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit  # noqa: E402

PLAN_DUMP = os.environ.get("POTRF_PLAN_DUMP", os.path.join(fit.REPO, "build/tests/potrf_plan_dump"))

# Per-route truth (t_launch, s_per_flop, s_per_byte, t_step). Different shapes on purpose so
# the crossovers between routes move with n and batch.
TRUTH = {
    "native:tiny": (3e-6, 2e-9, 2e-9, 4e-8),
    "native:cta": (4e-6, 4.5e-10, 1.5e-9, 6e-8),
    "native:lpanel": (5e-6, 2.6e-9, 1e-9, 3e-8, 2e-9),
    "native:blocked": (6e-6, 1e-13, 5e-12, 2e-8),   # batch-wide work: GPU-wide rates
    "vendor": (4e-5, 2e-13, 6e-12, 2e-7, 0.0, 1e-9),
}


class CombinePin(unittest.TestCase):
    def test_matches_launch_plan_combine(self):
        # potrf_plan_tests LaunchPlanCost.CombineMatchesTheFitterPin holds the same numbers.
        got = fit.combine((2, 1e6, 3e5, 40, 5e5), (5e-6, 2e-12, 1e-11, 3e-8, 1e-11))
        self.assertAlmostEqual(got, 1e-5 + 5e-6 + 1.2e-6, delta=1e-18)
        # Additive (vendor) form: flop and byte add; item outside the max.
        got = fit.combine((1, 2, 3, 4, 0, 5, 1), (1, 1, 1, 1, 0, 1))
        self.assertEqual(got, 15)


class FitterRecovery(unittest.TestCase):
    def test_constants_come_back_from_noisy_synthetic_terms(self):
        rng = random.Random(7)
        truth = (5e-6, 2e-11, 1e-9, 5e-8)
        data = []
        for _ in range(240):
            terms = (rng.choice((1, 2, 5, 17)), 10 ** rng.uniform(3, 9), 10 ** rng.uniform(2, 7),
                     10 ** rng.uniform(0, 4))
            data.append((terms, fit.combine(terms, truth) * math.exp(rng.gauss(0, 0.01))))
        const, stats = fit.fit_constants(data)
        for j, name in enumerate(fit.PARAMS[:len(truth)]):
            self.assertLess(abs(const[j] / truth[j] - 1), 0.05, f"{name}: {const[j]} vs {truth[j]}")
        self.assertLess(stats["rms_log"], 0.02)

    def test_an_invisible_term_is_zero_not_noise(self):
        rng = random.Random(3)
        truth = (5e-6, 2e-11, 0.0, 5e-8)   # bytes never bind
        data = []
        for _ in range(120):
            terms = (1, 10 ** rng.uniform(4, 9), 10 ** rng.uniform(0, 2), 10 ** rng.uniform(0, 3))
            data.append((terms, fit.combine(terms, truth)))
        const, stats = fit.fit_constants(data)
        self.assertIn("s_per_byte", stats["inactive"])
        self.assertEqual(const[2], 0.0)


class FallbackNeverBorrowsAnUnseenTerm(unittest.TestCase):
    CFG = {"facts": {"fp64_rate": 1 / 64}, "borrows": []}

    @staticmethod
    def entry(c, inactive):
        return {"constants": list(c), "fit": {"inactive": list(inactive)}, "fallback": None}

    def test_inactive_term_comes_from_the_next_source(self):
        # other_uplo never saw the flop term; the precision partner did.
        m = {("native:cta", "double", "U"): self.entry((1e-6, 0.0, 2e-9, 3e-8), ["s_per_flop"]),
             ("native:cta", "float", "L"): self.entry((9e-6, 5e-11, 7e-9, 8e-8), [])}
        e = fit.fallback(m, "native:cta", "double", "L", self.CFG)
        self.assertEqual(e["constants"], [1e-6, 5e-11 * 64, 2e-9, 3e-8, 0.0, 0.0])
        self.assertEqual(e["fallback"]["terms"]["s_per_flop"]["rule"], "precision_scaled")
        self.assertEqual(e["fallback"]["terms"]["t_launch"]["rule"], "other_uplo")

    def test_a_term_no_source_saw_makes_the_key_unfittable(self):
        m = {("native:cta", "cfloat", "L"): self.entry((6e-5, 0.0, 0.0, 1e-6),
                                                       ["s_per_flop", "s_per_byte"])}
        e = fit.fallback(m, "native:cta", "cdouble", "L", self.CFG)
        self.assertEqual(e["unfittable"], ["s_per_flop", "s_per_byte"])
        model = fit.complete(m, self.CFG, unfit := {})
        self.assertNotIn(("native:cta", "cdouble", "L"), model)
        self.assertIn(("native:cta", "cdouble", "L"), unfit)

    def test_own_rows_below_the_rule_give_a_thin_fit_not_unfittable(self):
        # Same unusable source, but the key has 6 rows of its own at ONE batch size.
        m = {("native:cta", "cfloat", "L"): self.entry((6e-5, 0.0, 0.0, 1e-6),
                                                       ["s_per_flop", "s_per_byte"])}
        truth = (2e-6, 3e-11, 1e-9, 5e-8)
        rows = []
        for n in (4, 8, 12, 16, 24, 32):
            terms = (1, 1e3 * n ** 3, 40.0 * n * n, float(n))
            rows.append((terms, fit.combine(terms, truth), n, 32768, ["x"]))
        key = ("native:cta", "cdouble", "L")
        model = fit.complete(m, self.CFG, unfit := {}, {key: rows})
        self.assertNotIn(key, unfit)
        e = model[key]
        self.assertTrue(e["thin"])
        self.assertEqual(e["fallback"]["terms"]["s_per_flop"]["rule"], "thin")
        self.assertEqual(e["fallback"]["terms"]["t_launch"]["rule"], "precision_scaled")
        self.assertEqual(e["support"]["batch_min"], 32768)
        # Fewer than 3 own rows is still unfittable.
        model = fit.complete(m, self.CFG, unfit := {}, {key: rows[:2]})
        self.assertIn(key, unfit)


@unittest.skipUnless(os.path.exists(PLAN_DUMP), f"needs {PLAN_DUMP}")
class PipelineRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fit_test_")
        ns = list(range(2, 41, 3)) + [48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224, 256,
                                      320, 384, 448, 512]
        shapes = [(d, u, n, b) for d in ("float", "double") for u in ("L", "U") for n in ns
                  for b in (256, 4096, 32768)]
        feats = fit.plan_features(PLAN_DUMP, fit.load_facts("sm_120"), shapes)
        rng = random.Random(11)
        cls.results = os.path.join(cls.tmp, "results.jsonl")
        with open(cls.results, "w") as f:
            for s in shapes:
                for route, plan in feats[s]["routes"].items():
                    if not (plan["supported"] and plan["fits"]):
                        continue
                    if s[0] == "double" and route == "native:lpanel":
                        continue   # no rows: the precision fallback must price it
                    t = fit.combine(plan["terms"], TRUTH[route]) * math.exp(rng.gauss(0, 0.01))
                    f.write(json.dumps({"op": "potrf", "dtype": s[0], "uplo": s[1], "m": s[2],
                                        "n": s[2], "batch": s[3], "route": route,
                                        "arm": "route:" + route, "time_ms": t * 1e3,
                                        "ok": True, "rel_sd": 0.01}) + "\n")
        cls.out = os.path.join(cls.tmp, "out")
        cls.stats = fit.main(["--results", cls.results, "--profile", "sm_120", "--plan-dump",
                              PLAN_DUMP, "--out", cls.out, "--compare-windows", "--grid"])
        with open(os.path.join(cls.out, "profile.json")) as f:
            cls.profile = json.load(f)

    def test_heldout_regret_is_near_one(self):
        self.assertGreater(self.stats["cells"], 100)
        self.assertLess(self.stats["geomean"], 1.01)
        self.assertLess(self.stats["max"], 1.20)

    def test_constants_recovered_where_active(self):
        for route in ("native:tiny", "native:cta", "native:blocked", "vendor"):
            e = self.profile["routes"][route]["float"]["L"]
            for j, name in enumerate(fit.PARAMS):
                if name in e["fit"]["inactive"]:
                    continue
                got = e["constants"][name]
                # Blocked's batch-wide work and its leaf chain are nearly collinear: 0.5.
                tol = 0.5 if route == "native:blocked" else 0.25
                self.assertLess(abs(got / TRUTH[route][j] - 1), tol, f"{route} {name}")

    def test_missing_dtype_takes_the_documented_fallback(self):
        e = self.profile["routes"]["native:lpanel"]["double"]["L"]
        self.assertIsNone(e["support"])
        self.assertEqual(e["fallback"]["rule"], "precision_scaled")
        self.assertEqual(e["fallback"]["from"], "native:lpanel/float/L")
        src = self.profile["routes"]["native:lpanel"]["float"]["L"]["constants"]
        self.assertEqual(e["fallback"]["flop_factor"], 64.0)
        self.assertAlmostEqual(e["constants"]["s_per_flop"], 64.0 * src["s_per_flop"], delta=1e-30)
        for name in ("t_launch", "s_per_byte", "t_step"):
            self.assertEqual(e["constants"][name], src[name], name)

    def test_support_region_is_the_measured_box(self):
        s = self.profile["routes"]["native:tiny"]["float"]["L"]["support"]
        self.assertEqual((s["n_min"], s["n_max"], s["batch_min"], s["batch_max"]),
                         (2, 32, 256, 32768))

    def test_report_lists_extrapolation_and_route_diff(self):
        with open(os.path.join(self.out, "report.md")) as f:
            rep = f.read()
        self.assertIn("## Extrapolated regions", rep)
        self.assertIn("fallback precision_scaled <- native:lpanel/float/L", rep)
        self.assertIn("### Route diff", rep)


    def test_gate_passes_against_its_own_truth(self):
        # Synthetic truth is the model's own form, so it must beat or tie today's windows.
        g = self.stats["gate"]
        self.assertGreater(g["paired_cells"], 50)
        self.assertLessEqual(g["model"]["max"], g["today"]["max"])
        self.assertEqual(g["verdict"], "PASS")

    def test_cross_profile_borrow_scales_by_device_facts(self):
        # A second, synthetic "sm_89" profile with NO LPanel rows borrows sm_120's LPanel.
        src = os.path.join(self.tmp, "b.jsonl")
        with open(self.results) as f, open(src, "w") as g:
            for line in f:
                if '"native:lpanel"' not in line:
                    g.write(line)
        out = os.path.join(self.tmp, "out_b")
        fit.main(["--results", src, "--profile", "sm_89", "--plan-dump", PLAN_DUMP, "--out", out,
                  "--borrow", "sm_120=" + os.path.join(self.out, "profile.json")])
        with open(os.path.join(out, "profile.json")) as f:
            prof = json.load(f)
        e = prof["routes"]["native:lpanel"]["float"]["L"]
        self.assertEqual(e["fallback"]["rule"], "other_profile")
        self.assertEqual(e["fallback"]["from"], "sm_120/native:lpanel/float/L")
        a, b = fit.load_facts("sm_120"), fit.load_facts("sm_89")
        ff = (a["fp32_tflops"] / a["cus"]) / (b["fp32_tflops"] / b["cus"])
        bf = (a["dram_gbps"] / a["cus"]) / (b["dram_gbps"] / b["cus"])
        s = self.profile["routes"]["native:lpanel"]["float"]["L"]["constants"]
        self.assertAlmostEqual(e["constants"]["s_per_flop"], s["s_per_flop"] * ff, delta=1e-30)
        self.assertAlmostEqual(e["constants"]["s_per_byte"], s["s_per_byte"] * bf, delta=1e-25)
        self.assertEqual(e["constants"]["t_launch"], s["t_launch"])
        self.assertEqual(e["constants"]["t_step"], s["t_step"])
        # Only MEASURED entries are borrowed: sm_120's double LPanel is itself a fallback,
        # so double LPanel has no source anywhere and is never a candidate.
        self.assertNotIn("double", prof["routes"]["native:lpanel"])

    def test_min_data_rule_borrows_a_single_batch_key(self):
        src = os.path.join(self.tmp, "c.jsonl")
        with open(self.results) as f, open(src, "w") as g:
            for line in f:
                j = json.loads(line)
                if (j["route"], j["dtype"], j["uplo"]) == ("native:cta", "float", "L") \
                        and j["batch"] != 4096:
                    continue
                g.write(line)
        out = os.path.join(self.tmp, "out_c")
        fit.main(["--results", src, "--profile", "sm_120", "--plan-dump", PLAN_DUMP, "--out", out])
        with open(os.path.join(out, "profile.json")) as f:
            e = json.load(f)["routes"]["native:cta"]["float"]["L"]
        self.assertEqual(e["fallback"]["rule"], "other_uplo")


@unittest.skipUnless(os.path.exists(PLAN_DUMP), f"needs {PLAN_DUMP}")
class FactsFileMatchesThisDevice(unittest.TestCase):
    def test_queried_facts(self):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/dpcpp-cuda/lib:" + env.get("LD_LIBRARY_PATH", "")
        try:
            out = subprocess.run([PLAN_DUMP, "--query-device"], capture_output=True, text=True,
                                 env=env, timeout=120, check=True).stdout
            q = json.loads(out)
        except Exception as e:
            self.skipTest(f"no GPU query: {e}")
        prof = {89: "sm_89", 120: "sm_120"}.get(q["cuda_cc"])
        if not prof:
            self.skipTest(f"no profile for cc {q['cuda_cc']}")
        f = fit.load_facts(prof)
        self.assertEqual((f["local_mem"], f["max_wg"], f["cus"]),
                         (q["local_mem"], q["max_wg"], q["cus"]))


if __name__ == "__main__":
    unittest.main()
