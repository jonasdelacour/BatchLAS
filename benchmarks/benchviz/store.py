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
            elif line.startswith("CMAKE_BUILD_TYPE:"):
                info["build_type"] = line.split("=", 1)[1] or "(none)"
    info.update(_compile_flags(b))
    info["warnings"] = build_warnings(info, _has_nvidia_gpu())
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


def _compile_flags(b: Path) -> Dict[str, object]:
    """Compiler id, fp model and SYCL targets, read from what the build compiled with."""
    out: Dict[str, object] = {}
    for f in sorted((b / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")):
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith('set(CMAKE_CXX_COMPILER_ID "'):
                out["compiler_id"] = line.split('"')[1]
    flags = b / "src" / "CMakeFiles" / "batchlas_sycl_obj.dir" / "flags.make"
    if not flags.exists():
        flags = next(iter(sorted((b / "src" / "CMakeFiles").glob("*/flags.make"))), None)
    if flags and flags.exists():
        text = " ".join(l for l in flags.read_text(errors="replace").splitlines() if l.startswith("CXX_FLAGS"))
        models = [t.split("=", 1)[1] for t in text.split() if t.startswith("-ffp-model=")]
        out["fp_model"] = models[-1] if models else "(compiler default)"  # the last one wins
        targets = [t.split("=", 1)[1] for t in text.split() if t.startswith("-fsycl-targets=")]
        out["sycl_targets"] = targets[-1] if targets else ""
    return out


def _has_nvidia_gpu() -> bool:
    return bool(_sh("nvidia-smi", "-L"))


def build_warnings(info: dict, nvidia_box: bool) -> List[str]:
    """Build configurations whose numbers are wrong for a known reason.

    evidence: docs/perf/blackwell.md#device-call-guard
    """
    w = []
    if info.get("compiler_id") == "IntelLLVM" and "fp_model" in info and info["fp_model"] != "precise":
        w.append(f"icpx build without -ffp-model=precise (fp model: {info['fp_model']}): device code "
                 "calls sycl::fma and barriers out of line, native kernels run 2-15x slow")
    if nvidia_box and "sycl_targets" in info and "nvptx" not in str(info["sycl_targets"]):
        w.append(f"NVIDIA GPU present but SYCL targets are '{info['sycl_targets'] or 'none'}': "
                 "a CPU-only build (put /opt/dpcpp-cuda/lib first on LD_LIBRARY_PATH and reconfigure)")
    return w


def harness_targets() -> List[str]:
    from ops import OPS
    return sorted({op.binary for op in OPS.values()})


def describe_build(bi: dict) -> str:
    if bi.get("error"):
        return f"{bi['dir']}: {bi['error']}"
    s = f"{bi['dir']}  built {bi.get('built')}"
    if bi.get("built_from"):
        s += f" from ~{bi['built_from']} ({bi.get('branch')}, HEAD {bi.get('head')})"
    if bi.get("build_type") or bi.get("compiler_id"):
        s += f"\n  {bi.get('compiler_id', '?')} {bi.get('build_type', '?')}, fp model {bi.get('fp_model', '?')}," \
             f" targets {bi.get('sycl_targets', '?')}"
    for w in bi.get("warnings") or []:
        s += f"\n  WARNING: {w}"
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
            if old.get("kind") == "compare":
                raise ValueError(f"{name} is a log comparison, not a campaign that can be run")
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

    def is_compare(self) -> bool:
        try:
            return self.config.get("kind") == "compare"
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def data_key(self) -> tuple:
        """Changes whenever rows() would return something new."""
        if self.is_compare():
            import compare
            return (str(self.dir), compare.data_key(self))
        try:
            st = self.results.stat()
            return (str(self.dir), st.st_size, st.st_mtime)
        except FileNotFoundError:
            return (str(self.dir), 0, 0)

    def rows(self) -> List[dict]:
        if self.is_compare():  # derived from its two source campaigns (compare.py)
            import compare
            return compare.rows(self)
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


def log_roots() -> List[Path]:
    """Every benchviz_runs/ on this box: this checkout's, the main checkout's, and
    each worktree's. Campaigns land in whichever checkout ran them, so a log worth
    comparing is often in another one."""
    common = _sh("git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir")
    main = Path(common).parent if common else REPO
    cands = [REPO / "benchviz_runs", main / "benchviz_runs",
             *sorted((main / ".claude" / "worktrees").glob("*/benchviz_runs"))]
    out = []
    for c in cands:
        if c.is_dir() and c.resolve() not in out:
            out.append(c.resolve())
    return out


def discover_logs(roots: Optional[List[Path]] = None) -> List[dict]:
    """Every campaign under the log roots, newest first, with what a picker shows."""
    out = []
    for root in roots or log_roots():
        for d in root.iterdir():
            p = d / "campaign.json"
            if not p.is_file():
                continue
            try:
                cfg = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            builds = [b for b in (cfg.get("provenance") or {}).get("builds") or [] if b.get("built_from")]
            res = d / "results.jsonl"
            out.append({
                "path": str(d), "name": d.name, "checkout": root.parent.name, "kind": cfg.get("kind", "campaign"),
                "build": builds[0]["built_from"] if builds else None, "ops": cfg.get("ops", []),
                "types": cfg.get("types", []), "grid": (cfg.get("grid") or {}).get("name") or cfg.get("preset"),
                "started": (cfg.get("provenance") or {}).get("started"),
                "rows": sum(1 for _ in open(res)) if res.is_file() else 0,
                "mtime": (res if res.is_file() else p).stat().st_mtime,
            })
    return sorted(out, key=lambda x: x["mtime"], reverse=True)


def list_campaigns(root: Path) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    cs = [d for d in root.iterdir() if (d / "campaign.json").exists()]
    return [d.name for d in sorted(cs, key=lambda d: d.stat().st_mtime, reverse=True)]
