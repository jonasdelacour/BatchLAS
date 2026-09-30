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
from store import Campaign, build_info, describe_build, list_campaigns

HERE = Path(__file__).resolve().parent
_procs: dict = {}  # campaign -> Popen started by this server
_bg: list = []     # replot children
_lock = threading.Lock()


def _reaper():
    """Collect exited children; unpolled, each finished run lingers as a zombie."""
    while True:
        with _lock:
            for p in [*_procs.values(), *_bg]:
                p.poll()
            _bg[:] = [p for p in _bg if p.returncode is None]
        time.sleep(5.0)


_cache: dict = {}


def _shape(r) -> str:
    """"n 64" for a square cell, "m 256 · n 32" on a rectangular op's plane."""
    return OPS[r["op"]].shape_text(r["m"], r["n"], r["nrhs"]) if r.get("op") in OPS else f"n {r.get('n')}"


def _analysis(camp: Campaign) -> dict:
    """Pairing and per-op statistics, recomputed only when results.jsonl changes."""
    key = camp.data_key()
    if _cache.get("key") == key:
        return _cache["val"]
    import numpy as np
    from plots import paired, saturated, throughput, throughput_label

    rows = camp.rows()
    latest: dict = {}
    for r in rows:  # a re-measured cell counts once, by its newest row
        latest[(r["op"], r["dtype"], r["m"], r["n"], r["nrhs"], r["batch"], r["arm"])] = r
    w = paired(rows)
    sat = saturated(w) if not w.empty else w
    cfg = camp.config
    try:  # a comparison plans nothing: its cells are whatever both sources measured
        planned = [] if cfg.get("kind") == "compare" else plan_cells(cfg["ops"], cfg["types"], Grid.from_config(cfg))
    except Exception:
        planned = []
    per_op: dict = {}
    for c in planned:
        per_op.setdefault(c.op, {"planned": 0})["planned"] += len(OPS[c.op].arms)
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
            sp = dt.speedup.astype(float).dropna().to_numpy()
            if sp.size:
                o["prec"][t] = {"geomean": float(np.exp(np.log(sp).mean())), "min": float(sp.min()),
                                "max": float(sp.max()), "wins": int((sp > 1).sum()), "n": int(sp.size)}
            elif op in OPS:  # no reference arm: the best saturated throughput, and where
                y = throughput(op, t, dt.m, dt.n, dt.nrhs, dt.batch, dt.time_ms_batchlas.astype(float))
                i = int(np.nanargmax(y))
                o["prec"][t] = {"peak": float(y[i]), "at_n": int(dt.n.iloc[i]), "n": int(len(y)),
                                "unit": throughput_label(op).split("[")[1].rstrip("]")}
        sp = d.speedup.astype(float).dropna() if not d.empty else []
        if len(sp):
            o["geomean"] = float(np.exp(np.log(sp).mean()))
    failures = [{k: r.get(k) for k in ("op", "dtype", "n", "batch", "arm", "reason", "t")} | {"shape": _shape(r)}
                for r in latest.values() if not r.get("ok")]
    failures.sort(key=lambda r: r.get("t") or 0)
    speed = {}
    if not w.empty:
        for r in w.itertuples():
            if r.speedup == r.speedup:  # NaN for a single-arm op, and not valid JSON
                speed[(r.op, r.dtype, r.m, r.n, r.nrhs, r.batch)] = float(r.speedup)
    activity = []
    for r in rows[-14:][::-1]:
        activity.append({k: r.get(k) for k in ("op", "dtype", "n", "batch", "arm", "time_ms", "ok", "reason", "route", "t")}
                        | {"shape": _shape(r),
                           "speedup": speed.get((r["op"], r["dtype"], r["m"], r["n"], r["nrhs"], r["batch"]))})
    walls = [r["wall_s"] for r in rows[-40:] if r.get("wall_s")]
    rate = 60.0 * len(walls) / sum(walls) if walls and sum(walls) > 0 else None
    val = {"per_op": per_op, "failures": failures[-300:], "activity": activity, "rate_per_min": rate,
           "mean_wall": sum(walls) / len(walls) if walls else None, "rows": len(rows),
           "planned": sum(o["planned"] for o in per_op.values()), "paired": w}
    if cfg.get("kind") == "compare":
        import compare
        val["control"] = compare.control(camp)
    _cache.update(key=key, val=val)
    return val


_replotting: dict = {}  # comparison -> (data key, Popen)


def _replot_compare(camp: Campaign) -> None:
    """A comparison has no runner to replot it: re-render when a source gains rows."""
    key = camp.data_key()
    with _lock:
        was = _replotting.get(camp.name)
        if was and (was[0] == key or was[1].poll() is None):
            return
        cmd = [sys.executable, str(HERE), "plot", camp.name, "--root", str(camp.root)]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        _replotting[camp.name] = (key, p)
        _bg.append(p)


_rendering: dict = {}  # campaign -> (Popen, start time), re-renders asked for from the page


def _rerender(camp: Campaign) -> tuple[dict, int]:
    with _lock:
        was = _rendering.get(camp.name)
        if was and was[0].poll() is None:
            return {"error": f"{camp.name} is already re-rendering"}, 409
        log = open(camp.dir / "plot.log", "w")
        cmd = [sys.executable, str(HERE), "plot", camp.name, "--root", str(camp.root)]
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        log.close()
        _rendering[camp.name] = (p, time.time())
        _bg.append(p)
    return {"ok": True}, 200


def render_state(camp: Campaign) -> dict:
    with _lock:
        was = _rendering.get(camp.name)
    if not was:
        return {"active": False}
    p, t0 = was
    code = p.poll()
    out = {"active": code is None, "started": t0, "ok": code == 0}
    if code:
        lines = (camp.dir / "plot.log").read_text(errors="replace").strip().splitlines()
        out["error"] = f"re-render failed (code {code}): " + next(
            (ln for ln in reversed(lines) if "Error" in ln), lines[-1] if lines else "")[:400]
    return out


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
    extra = {}
    if cfg.get("kind") == "compare":
        extra = {"kind": "compare", "base": cfg.get("base"), "new": cfg.get("new"), "control": an.get("control")}
        _replot_compare(camp)
    return {**extra,
        "campaign": name, "config": {**{k: cfg.get(k) for k in ("ops", "types", "preset", "backend", "gpu", "gpus")},
                                     "grid": Grid.from_config(cfg).to_dict()},
        "provenance": cfg.get("provenance", {}), "status": st, "eta_s": eta,
        "rate_per_min": an["rate_per_min"], "per_op": an["per_op"], "failures": an["failures"],
        "activity": an["activity"], "figures": figs, "log": tail, "rows": an["rows"], "planned": an["planned"],
        "render": render_state(camp),
    }


def op_table(root: Path, name: str, op: str) -> dict:
    camp = Campaign(root, name)
    if not (camp.dir / "campaign.json").exists() or op not in OPS:
        return {"rows": []}
    w = _analysis(camp)["paired"]
    if w.empty or not (w.op == op).any():
        return {"rows": []}
    d = w[w.op == op]
    cols = ["dtype", "m", "n", "nrhs", "batch", "time_ms_batchlas", "time_ms_vendor", "speedup", "route_batchlas", "route_vendor"]
    out = d[[c for c in cols if c in d]].copy()
    out["shape"] = [OPS[op].shape_text(int(m), int(n), int(k)) for m, n, k in zip(out.m, out.n, out.nrhs)]
    out["dtype"] = out["dtype"].map({t: i for i, t in enumerate(TYPES)}).fillna(9)
    out = out.sort_values(["dtype", "n", "m", "nrhs", "batch"])
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
    # The n x batch occupancy the preview draws: one mark per (n, batch) of the square
    # cells. The rectangular ones are counted, not drawn: they have no single n.
    sq = [c for c in cells if OPS[c.op].is_square(c.m, c.n, c.nrhs)]
    marks = sorted({(c.n, c.batch) for c in sq})
    arm_cells = sum(len(OPS[c.op].arms) for c in cells)
    return {"cells": len(cells), "arm_cells": arm_cells, "seconds": arm_cells * mean_wall,
            "orders": sorted({c.n for c in sq}), "batches": sorted({c.batch for c in sq}),
            "marks": marks, "rect_cells": len(cells) - len(sq), "grid": grid.to_dict()}


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")
    read_only: bool = False
    token: str = ""
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
    def _authorized(self) -> bool:
        """With a token set, every request needs it: once as ?t=... in the link,
        then from the cookie that first response sets."""
        if not self.token:
            return True
        import hmac
        q = parse_qs(urlparse(self.path).query).get("t", [""])[0]
        cookie = self.headers.get("Cookie", "")
        have = [c.split("=", 1)[1] for c in cookie.split("; ") if c.startswith("bvt=")]
        return any(hmac.compare_digest(x, self.token) for x in [q, *have] if x)

    def do_GET(self):
        if not self._authorized():
            return self._send(403, b"This dashboard needs the full link, including its ?t= token.", "text/plain")
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            extra = {"Set-Cookie": f"bvt={self.token}; Path=/; HttpOnly; SameSite=Lax; Secure"} if self.token else None
            return self._send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8", extra)
        if u.path == "/api/meta":
            return self._json({
                "read_only": self.read_only,
                "repo": str(HERE.parents[1]),
                "builds": [build_info(d) for d in self.build_dirs],
                "gpus": __import__("runner").detect_gpus(),
                "ops": [{"name": k, "title": v.title, "types": list(v.types), "notes": v.notes, "group": v.group, "vendor": list(v.vendor),
                         "orders": list(v.orders), "min_order": v.min_order, "max_order": v.max_order,
                         "plane": {"x": v.plane.x, "y": v.plane.y} if v.plane else None, "arms": len(v.arms)}
                        for k, v in OPS.items()],
                "types": [{"name": t, "label": TYPE_LABEL[t]} for t in TYPES],
                "presets": {k: v.to_dict() for k, v in PRESETS.items()},
                "campaigns": list_campaigns(self.root),
            })
        if u.path == "/api/logs":
            from store import discover_logs
            return self._json([x for x in discover_logs() if x["kind"] != "compare"])
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
        # Chunked framing: an unframed stream on a keep-alive connection cannot be
        # delimited, and proxies (cloudflared, VS Code forwarding) stall on it.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(data: bytes):
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        last, beat = None, time.time()
        try:
            while True:
                snap = snapshot(self.root, name) if name else {}
                snap["campaigns"] = list_campaigns(self.root)
                blob = json.dumps(snap)
                if blob != last:
                    chunk(f"data: {blob}\n\n".encode())
                    last, beat = blob, time.time()
                elif time.time() - beat > 15:
                    chunk(b": keepalive\n\n")
                    beat = time.time()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return

    # ------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        if not self._authorized():
            return self._json({"error": "forbidden"}, 403)
        # A comparison only reads logs and writes its small config: no GPU, so read-only allows it.
        if self.read_only and u.path not in ("/api/plan", "/api/compare"):
            return self._json({"error": "This copy of the dashboard is read-only; start runs from the GPU box."}, 403)
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
            camp = Campaign(self.root, name)
            camp.request_stop()
            st = camp.status()
            try:  # a runner that already died cannot honour the request itself
                os.kill(int(st.get("pid") or 0), 0) if st.get("pid") else None
                alive = bool(st.get("pid"))
            except (OSError, ValueError):
                alive = False
            if not alive and st.get("state") == "running":
                camp.set_status("stopped")
            return self._json({"ok": True})
        if u.path == "/api/compare":
            return self._compare(b)
        if u.path == "/api/replot":
            name = b.get("campaign", "")
            if name not in list_campaigns(self.root):
                return self._json({"error": "no such campaign"}, 404)
            return self._json(*_rerender(Campaign(self.root, name)))
        self._json({"error": "unknown endpoint"}, 404)

    def _compare(self, b: dict):
        import compare
        ab, an = b.get("base_arm") or "batchlas", b.get("new_arm") or "batchlas"
        try:
            base, new = (compare.resolve(self.root, str(b.get(k, ""))) for k in ("base", "new"))
            arms = "" if ab == an == "batchlas" else f"-{ab}-{an}"
            name = "".join(ch for ch in str(b.get("name") or f"cmp-{base.name}-vs-{new.name}{arms}")
                           if ch.isalnum() or ch in "-_.")
            camp = compare.create(self.root, name, str(base), str(new), ab, an, bool(b.get("exact")))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        _replot_compare(camp)
        return self._json({"ok": True, "campaign": name})

    def _run(self, b: dict):
        ops = [o for o in b.get("ops", []) if o in OPS]
        types = [t for t in b.get("types", []) if t in TYPES]
        try:
            grid = request_grid(b)
        except (ValueError, TypeError) as e:
            return self._json({"error": f"grid: {e}"}, 400)
        if not ops or not types:
            return self._json({"error": "choose at least one op and one precision"}, 400)
        from runner import detect_gpus
        gpus = sorted({int(g) for g in (b.get("gpus") or [b.get("gpu", 1)])})
        known = {g["index"] for g in detect_gpus()}
        if known and not set(gpus) <= known:
            return self._json({"error": f"GPU {gpus} not on this box; it has GPU {sorted(known)}"}, 400)
        name = "".join(ch for ch in str(b.get("campaign") or "") if ch.isalnum() or ch in "-_.") \
            or time.strftime("cuda-%Y%m%d-%H%M%S")
        with _lock:
            p = _procs.get(name)
            if p and p.poll() is None:
                return self._json({"error": f"{name} is already running"}, 409)
            cmd = [sys.executable, str(HERE), "run", "--root", str(self.root), "--campaign", name,
                   "--ops", ",".join(ops), "--types", ",".join(types),
                   "--grid-json", json.dumps(grid.to_dict()), "--gpu", ",".join(map(str, gpus))]
            for d in self.build_dirs:
                cmd += ["--build-dir", str(d)]
            (self.root / name).mkdir(parents=True, exist_ok=True)
            log = open(self.root / name / "run.log", "a")
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            p = _procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, "PYTHONUNBUFFERED": "1", "BENCHVIZ_STDOUT_IS_LOG": "1"},
                                                start_new_session=True)
            log.close()
        # A child that dies at startup (e.g. a missing pandas) never writes a
        # status, so without this the page just keeps showing "idle".
        try:
            code = p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return self._json({"ok": True, "campaign": name})
        tail = (self.root / name / "run.log").read_text(errors="replace").strip().splitlines()[-3:]
        return self._json({"error": f"run exited immediately (code {code}): " + " | ".join(tail),
                           "campaign": name}, 500)


def serve(root: Path, host: str, port: int, build_dirs: list, read_only: bool = False, token: str = ""):
    root.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_reaper, daemon=True).start()
    Handler.root = root
    Handler.build_dirs = build_dirs
    Handler.read_only = read_only
    Handler.token = token
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    # Port forwarders (VS Code Remote, ssh -L localhost:...) may dial ::1 rather
    # than 127.0.0.1; listen on the IPv6 loopback too so both reach the page.
    if host in ("127.0.0.1", "localhost"):
        import socket

        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        try:
            v6 = V6(("::1", port), Handler)
            v6.daemon_threads = True
            threading.Thread(target=v6.serve_forever, daemon=True).start()
        except OSError:
            pass
    print(f"benchviz dashboard: http://{host}:{port}/   (campaigns in {root})", flush=True)
    for d in build_dirs or []:
        print("  runs use " + describe_build(build_info(d)), flush=True)
    if not build_dirs:
        print("  WARNING: no build directory found; runs will fail. See README.md step 1.", flush=True)
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
