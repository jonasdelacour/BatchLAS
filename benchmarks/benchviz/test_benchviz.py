"""python3 -m unittest discover -s benchmarks/benchviz"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ops import OPS, PRESETS, Grid, batch_ladder, memory_cap, parse_list, plan_cells  # noqa: E402
from plots import paired, saturated  # noqa: E402
from runner import classify, parse_coverage  # noqa: E402

HDR = "kind,op,scalar,backend,shape_class,m,n,k,batch,chosen_origin,chosen_algo,calls\n"


def rec(route, ok=True, subroutes=()):
    return {"route": route, "ok": ok, "reason": "ok", "subroutes": list(subroutes)}


class Classify(unittest.TestCase):
    def test_pins_that_landed_pass(self):
        self.assertTrue(classify(rec("native:cta"), OPS["potrf"], "batchlas")["ok"])
        self.assertTrue(classify(rec("vendor:auto"), OPS["potrf"], "vendor")["ok"])

    def test_native_pin_that_fell_to_vendor_is_dropped(self):
        r = classify(rec("vendor:auto"), OPS["potrf"], "batchlas")
        self.assertFalse(r["ok"])
        self.assertIn("vendor:auto", r["reason"])

    def test_composed_op_is_judged_by_its_sub_ops(self):
        # Both gesv arms take the outer `blocked` route; only the sub-ops differ.
        v = classify(rec("native:blocked", subroutes=["getrf=vendor:auto", "getrs=vendor:auto"]),
                     OPS["gesv"], "vendor")
        self.assertTrue(v["ok"])
        self.assertEqual(v["route"], "getrf=vendor:auto+getrs=vendor:auto")
        n = classify(rec("native:blocked", subroutes=["getrf=vendor:auto", "getrs=native:cta"]),
                     OPS["gesv"], "batchlas")
        self.assertFalse(n["ok"])

    def test_composed_op_with_no_sub_op_rows_is_not_trusted(self):
        self.assertFalse(classify(rec("native:blocked"), OPS["posv"], "vendor")["ok"])


class Coverage(unittest.TestCase):
    def test_top_route_sub_routes_and_vendor_freedom(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cov.123"), "w") as fh:
                fh.write(HDR)
                fh.write("reached,syev,float,CUDA,1,64,64,64,1024,native,blocked,6\n")
                fh.write("reached,gemm,float,AUTO,1,64,64,64,1024,vendor,auto,25\n")
                fh.write("linked,gemm,float,CUDA,1,0,0,0,0,native,cta,0\n")  # linked is not reached
            c = parse_coverage(os.path.join(d, "cov"), "syev")
        self.assertEqual(c["route"], "native:blocked")
        self.assertEqual(c["subroutes"], ["gemm=vendor:auto"])
        self.assertFalse(c["vendor_free"])

    def test_no_rows_is_unknown_not_native(self):
        with tempfile.TemporaryDirectory() as d:
            c = parse_coverage(os.path.join(d, "cov"), "potrf")
        self.assertEqual(c["route"], "unknown")
        self.assertFalse(classify({**c, "ok": True, "reason": "ok"}, OPS["potrf"], "batchlas")["ok"])


class Ladder(unittest.TestCase):
    def test_ladder_is_descending_and_memory_capped(self):
        b = batch_ladder(OPS["syev"], "cdouble", 1024, PRESETS["full"], mem_gib=3.0)
        self.assertEqual(b, sorted(b, reverse=True))
        self.assertLessEqual(b[0] * OPS["syev"].footprint * 1024 * 1024 * 16, 3 * (1 << 30))

    def test_ops_only_get_their_types(self):
        cells = plan_cells(["gesvd"], ["cfloat", "float"], PRESETS["smoke"], 3.0)
        self.assertTrue(cells)
        self.assertEqual({c.dtype for c in cells}, {"float"})


class GridSpec(unittest.TestCase):
    def test_list_syntax(self):
        self.assertEqual(parse_list("4:32"), [4, 8, 16, 32])
        self.assertEqual(parse_list("8:32:8, 5"), [5, 8, 16, 24, 32])
        with self.assertRaises(ValueError):
            parse_list("16:x")

    def test_modes(self):
        op = OPS["gemm"]
        sat = Grid.from_dict({**PRESETS["quick"].to_dict(), "batch_mode": "saturated"})
        self.assertEqual(len(batch_ladder(op, "float", 64, sat)), 1)
        lst = Grid.from_dict({**PRESETS["quick"].to_dict(), "batch_mode": "list", "batches": "100,1000"})
        self.assertEqual(batch_ladder(op, "float", 64, lst), [1000, 100])

    def test_orders_are_clipped_to_what_the_op_supports(self):
        g = Grid.from_dict({**PRESETS["quick"].to_dict(), "orders": "16:128"})
        self.assertEqual({c.n for c in plan_cells(["gesvd"], ["float"], g)}, {16, 32})

    def test_old_campaign_config_still_loads(self):
        g = Grid.from_config({"preset": "smoke", "mem_gib": 2.0, "orders": [8, 16]})
        self.assertEqual((g.batch_step, g.mem_gib, g.orders), (16, 2.0, [8, 16]))


class Pairing(unittest.TestCase):
    def row(self, arm, batch, t, ok=True, n=32, stamp=0.0):
        return dict(op="potrf", dtype="float", m=n, n=n, nrhs=0, batch=batch, arm=arm, ok=ok,
                    time_ms=t, rel_sd=0.01, route="x", vendor_free=True, t=stamp)

    def test_speedup_is_vendor_over_batchlas_and_failures_drop_the_cell(self):
        w = paired([self.row("batchlas", 1024, 1.0), self.row("vendor", 1024, 3.0),
                    self.row("batchlas", 4096, 2.0, ok=False), self.row("vendor", 4096, 5.0)])
        self.assertEqual(len(w), 1)
        self.assertAlmostEqual(float(w.speedup.iloc[0]), 3.0)

    def test_saturated_picks_largest_measured_batch(self):
        w = paired([self.row("batchlas", 1024, 1.0), self.row("vendor", 1024, 3.0),
                    self.row("batchlas", 4096, 2.0), self.row("vendor", 4096, 5.0)])
        self.assertEqual(int(saturated(w).batch.iloc[0]), 4096)

    def test_a_remeasured_cell_uses_the_newest_row(self):
        w = paired([self.row("batchlas", 1024, 9.0, stamp=1), self.row("batchlas", 1024, 1.0, stamp=2),
                    self.row("vendor", 1024, 2.0, stamp=1)])
        self.assertAlmostEqual(float(w.speedup.iloc[0]), 2.0)


class Rectangular(unittest.TestCase):
    def square_grid(self):
        return Grid.from_dict({**PRESETS["quick"].to_dict(), "rect": False})

    def test_square_cells_keep_their_keys_args_and_batches(self):
        # A resumed campaign must not re-measure: these are the pre-plane values.
        cells = plan_cells(["gemm", "trsm", "spmm", "ormqr"], ["float"], self.square_grid())
        for c in cells:
            self.assertEqual((c.m, c.nrhs), (c.n, 0))
        c = next(c for c in cells if c.op == "gemm" and c.n == 64)
        self.assertEqual(OPS["gemm"].cell_args(c), [64, 64, 64, c.batch])
        self.assertEqual(memory_cap(OPS["gemm"], "float", 1024, self.square_grid()),
                         int(3 * (1 << 30) // (3.0 * 1024 * 1024 * 4)))
        c = next(c for c in cells if c.op == "spmm")
        self.assertEqual(OPS["spmm"].cell_args(c)[2], 16)

    def test_old_campaign_grids_do_not_grow_a_plane(self):
        g = Grid.from_config({"grid": {k: v for k, v in PRESETS["quick"].to_dict().items() if k != "rect"}})
        self.assertFalse(g.rect)
        self.assertTrue(PRESETS["quick"].rect)

    def test_plane_cells_are_valid_shapes_at_one_batch(self):
        cells = plan_cells(["geqrf", "gemm", "trsm", "getrs"], ["float"], PRESETS["quick"])
        qr = [c for c in cells if c.op == "geqrf"]
        self.assertTrue(all(c.n <= c.m for c in qr))
        self.assertTrue(any(c.n < c.m for c in qr))
        rect = [c for c in cells if not OPS[c.op].is_square(c.m, c.n, c.nrhs)]
        self.assertEqual(len({(c.op, c.m, c.n, c.nrhs) for c in rect}), len(rect))
        g = next(c for c in cells if c.op == "gemm" and c.n == 256 and c.nrhs == 16)
        self.assertEqual(OPS["gemm"].cell_args(g), [256, 256, 16, g.batch])
        t = next(c for c in cells if c.op == "trsm" and c.n == 32 and c.nrhs == 512)
        self.assertEqual(OPS["trsm"].cell_args(t), [32, 512, t.batch])
        # The square point of the plane is the square sweep's cell, never a second one.
        self.assertFalse([c for c in cells if c.op == "gemm" and c.nrhs == c.n])
        self.assertTrue({c.nrhs for c in cells if c.op == "getrs"} >= {1, 4, 256})

    def test_flops_use_the_real_third_dimension(self):
        self.assertEqual(float(OPS["gemm"].flops(64, 64, 0, "float")), 2.0 * 64 ** 3)
        self.assertEqual(float(OPS["gemm"].flops(64, 64, 8, "float")), 2.0 * 64 * 64 * 8)
        self.assertEqual(float(OPS["gemv"].flops(128, 32, 0, "float")), 2.0 * 128 * 32)
        self.assertEqual(float(OPS["trsm"].flops(32, 32, 512, "cfloat")), 4.0 * 32 * 32 * 512)


class PlaneFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import style
        style.apply(usetex=False)

    def rows(self):
        out = []
        for (m, n, sp) in ((64, 64, 2.0), (128, 64, 1.5), (128, 128, 0.8), (256, 32, 3.0)):
            for arm, t in (("batchlas", 1.0), ("vendor", sp)):
                out.append(dict(op="geqrf", dtype="float", m=m, n=n, nrhs=0, batch=1024, arm=arm, ok=True,
                                time_ms=t, rel_sd=0.01, route="x", vendor_free=True, t=0.0))
        return out

    def test_n_figures_see_square_cells_only(self):
        w = paired(self.rows())
        self.assertEqual(sorted(saturated(w).n), [64, 128])
        self.assertEqual(sorted(saturated(w).speedup), [0.8, 2.0])

    def test_maps_render_and_place_cells_by_shape(self):
        import matplotlib.pyplot as plt
        from plots import fig_speedup_2d, fig_throughput_2d, plane_points
        w = paired(self.rows())
        p = plane_points(w, "geqrf").set_index(["x", "y"])
        self.assertAlmostEqual(float(p.loc[(32, 256)].speedup), 3.0)
        for fn in (fig_speedup_2d, fig_throughput_2d):
            fig = fn(w, "geqrf", {})
            self.assertIsNotNone(fig)
            plt.close(fig)
        sq = paired([r for r in self.rows() if r["m"] == r["n"]])
        self.assertIsNone(fig_speedup_2d(sq, "geqrf", {}))   # the diagonal alone is not a map
        self.assertIsNone(fig_speedup_2d(w, "potrf", {}))


def row(op, n, arm, time_ms, batch=1024, t=0.0, dtype="float", ok=True):
    return dict(op=op, dtype=dtype, m=n, n=n, nrhs=0, batch=batch, arm=arm, ok=ok, time_ms=time_ms,
                rel_sd=0.01, route="native:x", vendor_free=True, t=t, reason="ok")


class SubOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import style
        style.apply(usetex=False)

    def test_single_arm_and_the_arguments_syev_passes(self):
        for name in ("stedc", "steqr", "sytrd", "sytrd_cta", "sy2sb", "sb2st"):
            self.assertTrue(OPS[name].single, name)
            self.assertEqual([a.key for a in OPS[name].arms], ["batchlas"])
        from ops import Cell
        # Auto everything for stedc; 50 sweeps and interleaved vectors for steqr.
        self.assertEqual(OPS["stedc"].cell_args(Cell("stedc", "float", 320, 320, 0, 512)), [320, 512, 0, -1, 0, 0, -1])
        self.assertEqual(OPS["steqr"].cell_args(Cell("steqr", "float", 64, 64, 0, 512)), [64, 512, 50, 1, 0])
        # syev_blocked's panel: the complex override at 256 < n <= 512 only.
        self.assertEqual(OPS["sytrd"].cell_args(Cell("sytrd", "cfloat", 384, 384, 0, 64))[2], 32)
        self.assertEqual(OPS["sytrd"].cell_args(Cell("sytrd", "float", 384, 384, 0, 64))[2], 8)
        self.assertEqual(OPS["sytrd"].cell_args(Cell("sytrd", "double", 1024, 1024, 0, 64))[2], 48)
        self.assertEqual(OPS["sy2sb"].cell_args(Cell("sy2sb", "float", 64, 64, 0, 8)), [64, 8, 32, 0])
        self.assertEqual(max(c.n for c in plan_cells(["sytrd_cta"], ["float"], PRESETS["quick"])), 32)

    def test_single_arm_cells_pair_alone_and_draw_throughput_only(self):
        import matplotlib.pyplot as plt
        from plots import fig_heatmap, fig_speedup_n, fig_summary, fig_throughput_n
        rows = [row("stedc", n, "batchlas", n / 64, batch=b) for n in (64, 128) for b in (256, 1024)]
        rows += [row("potrf", 64, "batchlas", 1.0), row("potrf", 64, "vendor", 2.0),
                 row("potrf", 128, "batchlas", 1.0)]   # an unpaired two-arm cell still drops
        w = paired(rows)
        self.assertEqual(len(w[w.op == "stedc"]), 4)
        self.assertTrue(w[w.op == "stedc"].speedup.isna().all())
        self.assertEqual(list(w[w.op == "potrf"].n), [64])
        self.assertIsNone(fig_speedup_n(w, "stedc", {}))
        for fn in (fig_throughput_n, fig_heatmap):
            fig = fn(w, "stedc", {})
            self.assertIsNotNone(fig)
            plt.close(fig)
        fig = fig_summary(w, "float", {})
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()], ["potrf"])
        plt.close(fig)


class Compare(unittest.TestCase):
    def setUp(self):
        from store import Campaign
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for name, sha, t_mine, t_vendor in (("old", "aaaaaaa", 2.0, 1.0), ("new", "bbbbbbb", 1.0, 1.1)):
            c = Campaign.create(self.root, name, {
                "ops": ["potrf", "stedc"], "types": ["float"], "backend": "cuda", "grid": PRESETS["quick"].to_dict(),
                "provenance": {"device": "RTX", "builds": [{"dir": f"/b/{name}", "built": "x", "built_from": sha}]}})
            for n in (64, 128):
                c.append(row("potrf", n, "batchlas", t_mine))
                c.append(row("potrf", n, "vendor", t_vendor))
                c.append(row("stedc", n, "batchlas", 2 * t_mine))
            if name == "new":
                c.append(row("stedc", 256, "batchlas", 1.0))   # measured in one build only

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_the_candidate_against_the_baseline(self):
        import compare
        cmp = compare.create(self.root, "cmp", "old", "new")
        w = paired(cmp.rows())
        # The baseline's BatchLAS arm sits in the reference slot: 2x faster in the new build.
        self.assertEqual(sorted(set(w.speedup.round(6))), [2.0])
        self.assertEqual(sorted(w[w.op == "stedc"].n), [64, 128])   # the stedc speedup a campaign cannot give
        ctl = compare.control(cmp)
        self.assertEqual((ctl["matched"], ctl["only_new"], ctl["only_base"]), (4, 1, 0))
        self.assertAlmostEqual(ctl["vendor"]["geomean"], 1.0 / 1.1)
        p = cmp.config["provenance"]
        self.assertEqual((p["ref_label"], p["new_label"]), ("Build aaaaaaa", "Build bbbbbbb"))
        self.assertIn("4 cells paired", compare.report(cmp))

    def test_logs_on_different_grids_pair_by_time_per_matrix(self):
        import compare
        from store import Campaign
        other = os.path.join(self.tmp.name, "elsewhere")   # another checkout's benchviz_runs/
        c = Campaign.create(other, "ladder", {"ops": ["potrf"], "types": ["float"], "backend": "cuda",
                                              "provenance": {"builds": [{"dir": "/b/c", "built": "y", "built_from": "ccccccc"}]}})
        for b, t in ((256, 0.5), (4096, 4.0)):             # never at batch 1024, the other log's
            c.append(row("potrf", 64, "batchlas", t, batch=b))
        c.append(row("potrf", 128, "batchlas", 1.0))      # batch 1024: an identical cell
        cmp = compare.create(self.root, "cmp", "old", os.path.join(c.dir, "results.jsonl"))
        w = paired(cmp.rows()).set_index("n")
        # old: 2.0 ms for 1024 matrices; ladder: 4.0 ms for 4096 -> 2x per matrix, at the larger batch.
        self.assertAlmostEqual(float(w.loc[64].speedup), 2.0)
        self.assertEqual(int(w.loc[64].batch), 4096)
        self.assertAlmostEqual(float(w.loc[128].speedup), 2.0)
        ctl = compare.control(cmp)
        self.assertEqual((ctl["matched"], ctl["rescaled"]), (2, 1))
        exact = compare.create(self.root, "cmp-exact", "old", c.dir, exact=True)
        self.assertEqual(list(paired(exact.rows()).n), [128])
        self.assertIn("different batch grids", " ".join(cmp.config["provenance"]["warnings"]))

    def test_any_arm_against_any_arm(self):
        import compare
        vv = compare.create(self.root, "vv", "old", "new", base_arm="vendor", new_arm="vendor")
        w = paired(vv.rows())
        self.assertEqual(sorted(set(w.op)), ["potrf"])             # stedc has no vendor arm
        self.assertAlmostEqual(float(w.speedup.iloc[0]), 1.0 / 1.1)
        self.assertIsNone(compare.control(vv)["vendor"]["geomean"])   # the control is what is compared
        p = vv.config["provenance"]
        self.assertEqual((p["ref_label"], p["new_label"]), ("Build aaaaaaa vendor", "Build bbbbbbb vendor"))
        mine = compare.create(self.root, "mine", "new", "new", base_arm="vendor")
        self.assertAlmostEqual(float(paired(mine.rows()).speedup.iloc[0]), 1.1)   # BatchLAS vs vendor, one log
        with self.assertRaises(ValueError):
            compare.create(self.root, "self", "new", "new")

    def test_follows_its_sources_and_cannot_be_run(self):
        import compare
        from store import Campaign
        cmp = compare.create(self.root, "cmp", "old", "new")
        k = cmp.data_key()
        Campaign(self.root, "old").append(row("stedc", 256, "batchlas", 3.0))
        self.assertNotEqual(cmp.data_key(), k)
        self.assertEqual(compare.control(cmp)["matched"], 5)
        with self.assertRaises(ValueError):
            Campaign.create(self.root, "cmp", {"ops": ["potrf"], "types": ["float"], "backend": "cuda"})
        with self.assertRaises(ValueError):
            compare.create(self.root, "cmp2", "cmp", "new")
        with self.assertRaises(ValueError):
            compare.create(self.root, "old", "old", "new")   # never overwrites a measured campaign

    def test_figures_name_the_builds(self):
        import compare
        import matplotlib.pyplot as plt
        import style
        from plots import fig_speedup_n
        style.apply(usetex=False)
        cmp = compare.create(self.root, "cmp", "old", "new")
        fig = fig_speedup_n(paired(cmp.rows()), "stedc", cmp.config["provenance"])
        self.assertIn("Build aaaaaaa", fig.axes[0].get_ylabel())
        plt.close(fig)


class RunEndpoint(unittest.TestCase):
    """The page's Run button: a child that dies at startup must come back as an error, not "idle"."""

    def post_run(self, child: str):
        import json
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer
        from unittest import mock
        import server
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "child.sh")
            with open(exe, "w") as f:
                f.write("#!/bin/sh\n" + child + "\n")
            os.chmod(exe, 0o755)
            server.Handler.root = server.Path(tmp) / "runs"
            server.Handler.build_dirs, server.Handler.read_only, server.Handler.token = [], False, ""
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            body = json.dumps({"ops": ["potrf"], "types": ["float"], "preset": "quick", "campaign": "t"}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/api/run", data=body, method="POST")
            try:
                with mock.patch.object(sys, "executable", exe), mock.patch("runner.detect_gpus", return_value=[]):
                    try:
                        with urllib.request.urlopen(req) as r:
                            return json.loads(r.read())
                    except urllib.error.HTTPError as e:
                        return json.loads(e.read())
            finally:
                httpd.shutdown()
                httpd.server_close()
                p = server._procs.pop("t", None)
                if p and p.poll() is None:
                    p.kill()
                    p.wait()

    def test_a_child_that_crashes_at_startup_is_reported(self):
        j = self.post_run("echo \"ModuleNotFoundError: No module named 'pandas'\"; exit 1")
        self.assertNotIn("ok", j)
        self.assertIn("exited immediately (code 1)", j["error"])
        self.assertIn("pandas", j["error"])

    def test_a_child_that_keeps_running_is_ok(self):
        self.assertEqual(self.post_run("sleep 30"), {"ok": True, "campaign": "t"})


class Rerender(unittest.TestCase):
    """The page's Re-render button: progress, refusal of a second click, and the failure reason."""

    def rerender(self, child: str, name: str):
        import time
        from unittest import mock
        import server
        from store import Campaign
        exe = os.path.join(self.tmp.name, f"{name}.sh")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\n" + child + "\n")
        os.chmod(exe, 0o755)
        camp = Campaign.create(self.tmp.name, name, {"ops": ["potrf"], "types": ["float"], "backend": "cuda"})
        with mock.patch.object(sys, "executable", exe):
            first = server._rerender(camp)
            second = server._rerender(camp)
        during = server.render_state(camp)
        server._rendering[name][0].wait(10)
        time.sleep(0.05)
        return first, second, during, server.snapshot(Path(self.tmp.name), name)["render"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_and_refusal_while_active(self):
        first, second, during, after = self.rerender("sleep 1", "good")
        self.assertEqual(first, ({"ok": True}, 200))
        self.assertEqual(second[1], 409)
        self.assertTrue(during["active"])
        self.assertEqual((after["active"], after["ok"], after.get("error")), (False, True, None))

    def test_failure_names_the_error(self):
        *_, after = self.rerender("sleep 1; echo 'RuntimeError: latex was not able to process'; exit 3", "bad")
        self.assertFalse(after["active"] or after["ok"])
        self.assertIn("code 3", after["error"])
        self.assertIn("latex was not able", after["error"])


if __name__ == "__main__":
    unittest.main()
