"""A campaign on disk: config, append-only results, status, figures.

    <root>/<name>/campaign.json   the request, plus provenance (git, device)
    <root>/<name>/results.jsonl   one line per (cell, arm), appended as it lands
    <root>/<name>/status.json     progress, rewritten atomically
    <root>/<name>/figures/        <op>/*.pdf|png|svg
    <root>/<name>/STOP            presence asks the runner to stop

Append-only JSONL is what makes a campaign resumable and watchable at once:
the runner never rewrites a row, the dashboard and the plotter only read.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "benchviz_runs"


def _sh(*argv) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


# What the binaries are compiled from; benchviz's own Python is not.
_SOURCE_PATHS = ("src", "include", "benchmarks", "cmake", "CMakeLists.txt", ":(exclude)benchmarks/benchviz")


def build_info(build_dir: Path) -> Dict[str, object]:
    """Which source a build directory's binaries were compiled from.

    The checkout benchviz runs from says nothing about this: a worktree's
    build/ can be days behind its HEAD. Nothing records the SHA at build time,
    so it is estimated as the last commit before the newest library was
    linked, and the build is stale when source changed after that.
    """
    b = Path(build_dir).resolve()
    info: Dict[str, object] = {"dir": str(b)}
    libs = [p for p in (b / "src").glob("libbatchlas*.so*") if p.is_file()]
    if not libs:
        info["error"] = "no libbatchlas*.so under src/"
        return info
    built = max(p.stat().st_mtime for p in libs)
    info["built"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(built))
    cache = b / "CMakeCache.txt"
    src = None
    if cache.exists():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:"):
                src = Path(line.split("=", 1)[1])
            elif line.startswith("CMAKE_CXX_COMPILER:"):
                info["compiler"] = line.split("=", 1)[1]
    if src is None or not (src / ".git").exists():
        return info
    git = ("git", "-C", str(src))
    info["source"] = str(src)
    info["branch"] = _sh(*git, "rev-parse", "--abbrev-ref", "HEAD")
    info["head"] = _sh(*git, "rev-parse", "--short", "HEAD")
    sha = _sh(*git, "rev-list", "-1", f"--before={int(built)}", "HEAD")
    info["built_from"] = sha[:7]
    changed = _sh(*git, "diff", "--name-only", sha, "HEAD", "--", *_SOURCE_PATHS).split() if sha else []
    edited = [f for f in _sh(*git, "diff", "--name-only", "HEAD", "--", *_SOURCE_PATHS).split()
              if (src / f).exists() and (src / f).stat().st_mtime > built]
    info["changed_since"] = len(set(changed) | set(edited))
    info["stale"] = bool(changed or edited)
    # A worktree's HEAD can itself be behind: count what main has that this build lacks.
    if sha:
        n = _sh(*git, "rev-list", "--count", "--no-merges", f"{sha}..main", "--", *_SOURCE_PATHS)
        info["behind_main"] = int(n) if n.isdigit() else None
    return info


def harness_targets() -> List[str]:
    from ops import OPS
    return sorted({op.binary for op in OPS.values()})


def describe_build(bi: dict) -> str:
    if bi.get("error"):
        return f"{bi['dir']}: {bi['error']}"
    s = f"{bi['dir']}  built {bi.get('built')}"
    if bi.get("built_from"):
        s += f" from ~{bi['built_from']} ({bi.get('branch')}, HEAD {bi.get('head')})"
    if bi.get("behind_main"):
        s += f"\n  NOTE: {bi['behind_main']} source commit(s) on main are not in this build"
    if bi.get("stale"):
        s += f"\n  WARNING: STALE BUILD -- {bi['changed_since']} source file(s) changed since it was built;" \
             f" rebuild with:\n    cmake --build {bi['dir']} -j\"$(nproc)\" --target {' '.join(harness_targets())}"
    return s


def provenance(backend: str, gpu: int) -> Dict[str, str]:
    p = {
        "git_sha": _sh("git", "-C", str(REPO), "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_sh("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")),
        "host": os.uname().nodename,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if backend == "cuda":
        q = _sh("nvidia-smi", f"--id={gpu}", "--query-gpu=name,driver_version", "--format=csv,noheader")
        if q:
            name, _, drv = q.partition(",")
            p.update(device=name.strip(), driver=drv.strip())
        p["vendor_label"] = "cuSOLVER"
    else:
        p["device"] = _sh("bash", "-c", "rocminfo 2>/dev/null | grep -m1 'Marketing Name' | cut -d: -f2").strip()
        p["vendor_label"] = "rocSOLVER"
    return p


class Campaign:
    def __init__(self, root: Path, name: str):
        self.root = Path(root)
        self.name = name
        self.dir = self.root / name
        self.results = self.dir / "results.jsonl"
        self.figures = self.dir / "figures"

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: Path, name: str, config: dict) -> "Campaign":
        c = cls(root, name)
        c.dir.mkdir(parents=True, exist_ok=True)
        c.figures.mkdir(exist_ok=True)
        cfg_path = c.dir / "campaign.json"
        # What THIS invocation runs; the unions below are what the plots show.
        config["request"] = {"ops": list(config["ops"]), "types": list(config["types"])}
        if cfg_path.exists():
            # Resuming or extending: ops and types accumulate, so the plots keep
            # showing what earlier invocations measured; everything else is the
            # latest request's.
            old = json.loads(cfg_path.read_text())
            if old.get("backend") != config.get("backend"):
                raise ValueError(f"campaign {name} is a {old.get('backend')} campaign")
            for k in ("ops", "types"):
                config[k] = list(dict.fromkeys([*old.get(k, []), *config[k]]))
            new = config.get("provenance") or {}
            config["provenance"] = old.get("provenance", new)
            # A resume may run a different build; keep every one it used.
            builds = config["provenance"].setdefault("builds", [])
            for bi in new.get("builds", []):
                if all((x.get("dir"), x.get("built")) != (bi.get("dir"), bi.get("built")) for x in builds):
                    builds.append(bi)
        cfg_path.write_text(json.dumps(config, indent=2))
        (c.dir / "STOP").unlink(missing_ok=True)
        return c

    @property
    def config(self) -> dict:
        return json.loads((self.dir / "campaign.json").read_text())

    # ------------------------------------------------------------ results
    def append(self, rec: dict) -> None:
        with open(self.results, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    def rows(self) -> List[dict]:
        if not self.results.exists():
            return []
        out = []
        with open(self.results) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line being written right now
        return out

    def done_keys(self) -> set:
        # Failed cells are retried on resume, except deterministic failures.
        keep = ("batchlas arm resolved", "vendor arm resolved", "binary ")
        return {f"{r['op']}|{r['dtype']}|{r['m']}|{r['n']}|{r['nrhs']}|{r['batch']}|{r['arm']}"
                for r in self.rows() if r.get("ok") or str(r.get("reason", "")).startswith(keep)}

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        p = self.dir / "status.json"
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"state": "idle"}

    def set_status(self, state: str, **kw) -> None:
        s = self.status()
        s.update(kw, state=state, updated=time.time())
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(s))
        os.replace(tmp, self.dir / "status.json")

    def request_stop(self) -> None:
        (self.dir / "STOP").touch()

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()


def list_campaigns(root: Path) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    cs = [d for d in root.iterdir() if (d / "campaign.json").exists()]
    return [d.name for d in sorted(cs, key=lambda d: d.stat().st_mtime, reverse=True)]
