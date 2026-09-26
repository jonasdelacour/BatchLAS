"""The live dashboard: standard library only (http.server + server-sent events).

The server never measures anything itself. A run started from the page is a
child `benchviz run` process writing to the same campaign directory a CLI run
would, and the page watches that directory -- so a campaign started from a
terminal (or over ssh, or before the server was up) is watched the same way.

Bind address defaults to 127.0.0.1: the page can start processes on the GPU
box. From a laptop: ssh -L 8765:localhost:8765 <box>, then open localhost:8765.
"""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ops import OPS, PRESETS, TYPE_LABEL, TYPES, Grid, plan_cells
from store import Campaign, list_campaigns

HERE = Path(__file__).resolve().parent
_procs: dict = {}  # campaign -> Popen started by this server
_lock = threading.Lock()


_cache: dict = {}


def _analysis(camp: Campaign) -> dict:
    """Pairing and per-op statistics, recomputed only when results.jsonl changes."""
    try:
        st = camp.results.stat()
        key = (str(camp.dir), st.st_size, st.st_mtime)
    except FileNotFoundError:
        key = (str(camp.dir), 0, 0)
    if _cache.get("key") == key:
        return _cache["val"]
    import numpy as np
    from plots import paired, saturated

    rows = camp.rows()
    latest: dict = {}
    for r in rows:  # a re-measured cell counts once, by its newest row
        latest[(r["op"], r["dtype"], r["m"], r["n"], r["nrhs"], r["batch"], r["arm"])] = r
    w = paired(rows)
    sat = saturated(w) if not w.empty else w
    cfg = camp.config
    try:
        planned = plan_cells(cfg["ops"], cfg["types"], Grid.from_config(cfg))
    except Exception:
        planned = []
    per_op: dict = {}
    for c in planned:
        per_op.setdefault(c.op, {"planned": 0})["planned"] += 2
    for r in latest.values():
        o = per_op.setdefault(r["op"], {"planned": 0})
        o["ok" if r.get("ok") else "failed"] = o.get("ok" if r.get("ok") else "failed", 0) + 1
        o["last_t"] = max(o.get("last_t", 0), r.get("t", 0))
    for op, o in per_op.items():
        o.setdefault("ok", 0)
        o.setdefault("failed", 0)
        o["planned"] = max(o["planned"], o["ok"] + o["failed"])
        d = sat[sat.op == op] if not sat.empty else sat
        o["prec"] = {}
        for t in TYPES:
            dt = d[d.dtype == t] if not d.empty else d
            if dt.empty:
                continue
            sp = dt.speedup.astype(float).to_numpy()
            o["prec"][t] = {"geomean": float(np.exp(np.log(sp).mean())), "min": float(sp.min()),
                            "max": float(sp.max()), "wins": int((sp > 1).sum()), "n": int(sp.size)}
        if not d.empty:
            o["geomean"] = float(np.exp(np.log(d.speedup.astype(float)).mean()))
    failures = [{k: r.get(k) for k in ("op", "dtype", "n", "batch", "arm", "reason", "t")}
                for r in latest.values() if not r.get("ok")]
    failures.sort(key=lambda r: r.get("t") or 0)
    speed = {}
    if not w.empty:
        for r in w.itertuples():
            speed[(r.op, r.dtype, r.n, r.batch)] = float(r.speedup)
    activity = []
    for r in rows[-14:][::-1]:
        activity.append({k: r.get(k) for k in ("op", "dtype", "n", "batch", "arm", "time_ms", "ok", "reason", "route", "t")}
                        | {"speedup": speed.get((r["op"], r["dtype"], r["n"], r["batch"]))})
    walls = [r["wall_s"] for r in rows[-40:] if r.get("wall_s")]
    rate = 60.0 * len(walls) / sum(walls) if walls and sum(walls) > 0 else None
    val = {"per_op": per_op, "failures": failures[-300:], "activity": activity, "rate_per_min": rate,
           "mean_wall": sum(walls) / len(walls) if walls else None, "rows": len(rows),
           "planned": sum(o["planned"] for o in per_op.values()), "paired": w}
    _cache.update(key=key, val=val)
    return val


def snapshot(root: Path, name: str) -> dict:
    camp = Campaign(root, name)
    if not (camp.dir / "campaign.json").exists():
        return {"campaign": name, "missing": True}
    cfg = camp.config
    st = camp.status()
    an = _analysis(camp)
    figs = []
    if camp.figures.exists():
        for p in sorted(camp.figures.rglob("*.png")):
            if ".tmp." in p.name:
                continue
            rel = p.relative_to(camp.figures)
            figs.append({"path": str(rel), "group": rel.parts[0], "name": p.stem, "mtime": p.stat().st_mtime})
    log = camp.dir / "run.log"
    tail = []
    if log.exists():
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 8000))
            tail = fh.read().decode(errors="replace").splitlines()[-60:]
    if st.get("state") == "running" and st.get("pid"):
        try:
            os.kill(st["pid"], 0)
        except OSError:
            st["state"] = "interrupted"
    eta = None
    if st.get("state") == "running" and an["mean_wall"] and st.get("total"):
        eta = (st["total"] - st.get("done", 0)) * an["mean_wall"]
    return {
        "campaign": name, "config": {**{k: cfg.get(k) for k in ("ops", "types", "preset", "backend", "gpu")},
                                     "grid": Grid.from_config(cfg).to_dict()},
        "provenance": cfg.get("provenance", {}), "status": st, "eta_s": eta,
        "rate_per_min": an["rate_per_min"], "per_op": an["per_op"], "failures": an["failures"],
        "activity": an["activity"], "figures": figs, "log": tail, "rows": an["rows"], "planned": an["planned"],
    }


def op_table(root: Path, name: str, op: str) -> dict:
    camp = Campaign(root, name)
    if not (camp.dir / "campaign.json").exists() or op not in OPS:
        return {"rows": []}
    w = _analysis(camp)["paired"]
    if w.empty or not (w.op == op).any():
        return {"rows": []}
    d = w[w.op == op].sort_values(["dtype", "n", "batch"])
    cols = ["dtype", "n", "batch", "time_ms_batchlas", "time_ms_vendor", "speedup", "route_batchlas", "route_vendor"]
    out = d[[c for c in cols if c in d]].copy()
    out["dtype"] = out["dtype"].map({t: i for i, t in enumerate(TYPES)}).fillna(9)
    out = out.sort_values(["dtype", "n", "batch"])
    out["dtype"] = out["dtype"].map(dict(enumerate(TYPES)))
    return {"rows": json.loads(out.to_json(orient="records"))}


def request_grid(b: dict) -> Grid:
    base = PRESETS.get(b.get("preset") or "quick", PRESETS["quick"]).to_dict()
    return Grid.from_dict({**base, **(b.get("grid") or {})})


def plan_estimate(b: dict, mean_wall: float) -> dict:
    ops = [o for o in b.get("ops", []) if o in OPS]
    types = [t for t in b.get("types", []) if t in TYPES]
    try:
        grid = request_grid(b)
    except (ValueError, TypeError) as e:
        return {"cells": 0, "arm_cells": 0, "seconds": 0, "error": str(e)}
    cells = plan_cells(ops, types, grid) if ops and types else []
    # The n x batch occupancy the preview draws: one mark per (n, batch) in the union.
    marks = sorted({(c.n, c.batch) for c in cells})
    return {"cells": len(cells), "arm_cells": 2 * len(cells), "seconds": 2 * len(cells) * mean_wall,
            "orders": sorted({c.n for c in cells}), "batches": sorted({c.batch for c in cells}),
            "marks": marks, "grid": grid.to_dict()}


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")
    build_dirs: list = []
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the terminal for the run log
        pass

    # ------------------------------------------------------------ helpers
    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/meta":
            return self._json({
                "ops": [{"name": k, "title": v.title, "types": list(v.types), "notes": v.notes, "group": v.group, "vendor": list(v.vendor),
                         "orders": list(v.orders), "min_order": v.min_order, "max_order": v.max_order}
                        for k, v in OPS.items()],
                "types": [{"name": t, "label": TYPE_LABEL[t]} for t in TYPES],
                "presets": {k: v.to_dict() for k, v in PRESETS.items()},
                "campaigns": list_campaigns(self.root),
            })
        if u.path == "/api/state":
            return self._json(snapshot(self.root, q.get("c", [""])[0]))
        if u.path == "/api/op":
            return self._json(op_table(self.root, q.get("c", [""])[0], q.get("op", [""])[0]))
        if u.path == "/api/events":
            return self._events(q.get("c", [""])[0])
        if u.path.startswith("/fig/"):
            # /fig/<campaign>/<path under figures/>
            parts = unquote(u.path[len("/fig/"):]).split("/", 1)
            if len(parts) != 2:
                return self._send(404, b"")
            base = (self.root / parts[0] / "figures").resolve()
            p = (base / parts[1]).resolve()
            if base not in p.parents or not p.is_file():
                return self._send(404, b"")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            extra = {"Content-Disposition": f'inline; filename="{parts[0]}_{p.parent.name}_{p.name}"'}
            return self._send(200, p.read_bytes(), ctype, extra)
        self._send(404, b"not found", "text/plain")

    def _events(self, name: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = None
        beat = 0.0
        try:
            while True:
                snap = snapshot(self.root, name) if name else {}
                snap["campaigns"] = list_campaigns(self.root)
                blob = json.dumps(snap)
                if blob != last:
                    self.wfile.write(f"data: {blob}\n\n".encode())
                    self.wfile.flush()
                    last = blob
                elif time.time() - beat > 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    beat = time.time()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            return

    # ------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
        except Exception:
            return self._json({"error": "bad json"}, 400)
        if u.path == "/api/run":
            return self._run(b)
        if u.path == "/api/plan":
            return self._json(plan_estimate(b, _cache.get("val", {}).get("mean_wall") or 4.0))
        if u.path == "/api/stop":
            name = b.get("campaign", "")
            if name not in list_campaigns(self.root):
                return self._json({"error": "no such campaign"}, 404)
            Campaign(self.root, name).request_stop()
            return self._json({"ok": True, "note": "stops after the current cell"})
        if u.path == "/api/replot":
            name = b.get("campaign", "")
            if name not in list_campaigns(self.root):
                return self._json({"error": "no such campaign"}, 404)
            cmd = [sys.executable, str(HERE), "plot", name, "--root", str(self.root)]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return self._json({"ok": True})
        self._json({"error": "unknown endpoint"}, 404)

    def _run(self, b: dict):
        ops = [o for o in b.get("ops", []) if o in OPS]
        types = [t for t in b.get("types", []) if t in TYPES]
        try:
            grid = request_grid(b)
        except (ValueError, TypeError) as e:
            return self._json({"error": f"grid: {e}"}, 400)
        if not ops or not types:
            return self._json({"error": "choose at least one op and one precision"}, 400)
        name = "".join(ch for ch in str(b.get("campaign") or "") if ch.isalnum() or ch in "-_.") \
            or time.strftime("cuda-%Y%m%d-%H%M%S")
        with _lock:
            p = _procs.get(name)
            if p and p.poll() is None:
                return self._json({"error": f"{name} is already running"}, 409)
            cmd = [sys.executable, str(HERE), "run", "--root", str(self.root), "--campaign", name,
                   "--ops", ",".join(ops), "--types", ",".join(types),
                   "--grid-json", json.dumps(grid.to_dict()), "--gpu", str(int(b.get("gpu", 1)))]
            for d in self.build_dirs:
                cmd += ["--build-dir", str(d)]
            (self.root / name).mkdir(parents=True, exist_ok=True)
            log = open(self.root / name / "run.log", "a")
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            _procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONUNBUFFERED": "1", "BENCHVIZ_STDOUT_IS_LOG": "1"},
                                            start_new_session=True)
        return self._json({"ok": True, "campaign": name})


def serve(root: Path, host: str, port: int, build_dirs: list):
    root.mkdir(parents=True, exist_ok=True)
    Handler.root = root
    Handler.build_dirs = build_dirs
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"benchviz dashboard: http://{host}:{port}/   (campaigns in {root})", flush=True)
    if host in ("127.0.0.1", "localhost"):
        print(f"  from another machine: ssh -L {port}:localhost:{port} {os.uname().nodename}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Runs started here are left running on purpose: they are resumable
        # campaigns, not children of a browser tab. Stop them from the page.
        httpd.server_close()
