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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ops import OPS, PRESETS, TYPE_LABEL, TYPES
from store import Campaign, list_campaigns

HERE = Path(__file__).resolve().parent
_procs: dict = {}  # campaign -> Popen started by this server
_lock = threading.Lock()


def snapshot(root: Path, name: str) -> dict:
    camp = Campaign(root, name)
    if not (camp.dir / "campaign.json").exists():
        return {"campaign": name, "missing": True}
    cfg = camp.config
    rows = camp.rows()
    st = camp.status()
    per_op: dict = {}
    failures = []
    walls = []
    for r in rows:
        o = per_op.setdefault(r["op"], {"ok": 0, "failed": 0})
        if r.get("ok"):
            o["ok"] += 1
        else:
            o["failed"] += 1
            failures.append({k: r.get(k) for k in ("op", "dtype", "n", "batch", "arm", "reason")})
        if r.get("wall_s"):
            walls.append(r["wall_s"])
    figs = []
    if camp.figures.exists():
        for p in sorted(camp.figures.rglob("*.png")):
            if ".tmp." in p.name:
                continue
            rel = p.relative_to(camp.figures)
            figs.append({"path": str(rel), "group": rel.parts[0], "name": p.stem,
                         "mtime": p.stat().st_mtime})
    log = camp.dir / "run.log"
    tail = []
    if log.exists():
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 6000))
            tail = fh.read().decode(errors="replace").splitlines()[-40:]
    running = st.get("state") == "running"
    pid = st.get("pid")
    if running and pid:
        try:
            os.kill(pid, 0)
        except OSError:
            running = False
            st["state"] = "interrupted"
    eta = None
    if running and walls and st.get("total"):
        recent = walls[-40:]
        eta = (st["total"] - st.get("done", 0)) * (sum(recent) / len(recent))
    return {
        "campaign": name, "config": {k: cfg.get(k) for k in ("ops", "types", "preset", "backend", "gpu")},
        "provenance": cfg.get("provenance", {}), "status": st, "eta_s": eta,
        "per_op": per_op, "failures": failures[-200:], "figures": figs, "log": tail,
        "rows": len(rows),
    }


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
                "ops": [{"name": k, "title": v.title, "types": list(v.types), "notes": v.notes}
                        for k, v in OPS.items()],
                "types": [{"name": t, "label": TYPE_LABEL[t]} for t in TYPES],
                "presets": list(PRESETS),
                "campaigns": list_campaigns(self.root),
            })
        if u.path == "/api/state":
            return self._json(snapshot(self.root, q.get("c", [""])[0]))
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
        preset = b.get("preset", "quick")
        if not ops or not types or preset not in PRESETS:
            return self._json({"error": "choose at least one op and one precision"}, 400)
        name = "".join(ch for ch in str(b.get("campaign") or "") if ch.isalnum() or ch in "-_.") \
            or time.strftime("cuda-%Y%m%d-%H%M%S")
        with _lock:
            p = _procs.get(name)
            if p and p.poll() is None:
                return self._json({"error": f"{name} is already running"}, 409)
            cmd = [sys.executable, str(HERE), "run", "--root", str(self.root), "--campaign", name,
                   "--ops", ",".join(ops), "--types", ",".join(types), "--preset", preset,
                   "--gpu", str(int(b.get("gpu", 1)))]
            if b.get("orders"):
                cmd += ["--orders", ",".join(str(int(x)) for x in str(b["orders"]).split(",") if x.strip())]
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
