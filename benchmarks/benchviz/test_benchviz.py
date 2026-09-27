"""python3 -m unittest discover -s benchmarks/benchviz"""
import os
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
