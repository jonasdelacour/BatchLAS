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
    "native:lpanel": (5e-6, 2e-10, 2e-9, 3e-8),
    "native:blocked": (6e-6, 6e-11, 1e-9, 2e-8),
    "vendor": (4e-5, 2e-13, 6e-12, 2e-7),
}


class CombinePin(unittest.TestCase):
    def test_matches_launch_plan_combine(self):
        # potrf_plan_tests LaunchPlanCost.CombineMatchesTheFitterPin holds the same numbers.
        got = fit.combine((2, 1e6, 3e5, 40), (5e-6, 2e-12, 1e-11, 3e-8))
        self.assertAlmostEqual(got, 1e-5 + 3e-6 + 1.2e-6, delta=1e-18)


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
        for j, name in enumerate(fit.PARAMS):
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


@unittest.skipUnless(os.path.exists(PLAN_DUMP), f"needs {PLAN_DUMP}")
class PipelineRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fit_test_")
        ns = list(range(2, 41, 3)) + [48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224, 256,
                                      320, 384, 448, 512]
        shapes = [(d, u, n, b) for d in ("float", "double") for u in ("L", "U") for n in ns
                  for b in (256, 4096, 32768)]
        feats = fit.plan_features(PLAN_DUMP, fit.PRESETS["sm_120"], shapes)
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
        self.assertLess(self.stats["max"], 1.10)

    def test_constants_recovered_where_active(self):
        for route in ("native:tiny", "native:cta", "native:blocked", "vendor"):
            e = self.profile["routes"][route]["float"]["L"]
            for j, name in enumerate(fit.PARAMS):
                if name in e["fit"]["inactive"]:
                    continue
                got = e["constants"][name]
                self.assertLess(abs(got / TRUTH[route][j] - 1), 0.25, f"{route} {name}")

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


if __name__ == "__main__":
    unittest.main()
