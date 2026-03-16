#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


TRACE_RE = re.compile(r'BATCHLAS_KERNEL_TRACE_SCOPE\("([^"]+)"\)')
TOKEN_STOPWORDS = {
    "acc",
    "accuracy",
    "benchmark",
    "bench",
    "blocked",
    "cta",
    "custom",
    "family",
    "flat",
    "legacy",
    "steady",
    "tests",
}


@dataclass(order=True)
class BenchmarkCandidate:
    sort_key: tuple[int, str] = field(init=False, repr=False)
    score: int
    target: str
    source: Path
    reasons: List[str] = field(default_factory=list)
    hits: List[str] = field(default_factory=list)
    binary: Path | None = None

    def __post_init__(self) -> None:
        self.sort_key = (-self.score, self.target)


@dataclass(order=True)
class TraceCandidate:
    sort_key: tuple[int, str] = field(init=False, repr=False)
    score: int
    label: str
    source: Path

    def __post_init__(self) -> None:
        self.sort_key = (-self.score, self.label)


def repo_root_from(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "benchmarks" / "CMakeLists.txt").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate BatchLAS repository root")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def query_terms(raw_queries: Iterable[str]) -> tuple[List[str], List[str]]:
    normalized: List[str] = []
    tokens: List[str] = []
    for raw in raw_queries:
        path = Path(raw)
        stem = path.stem if path.suffix else path.name
        for part in (raw, stem):
            value = normalize(part)
            if value and value not in normalized:
                normalized.append(value)
        for part in re.split(r"[^A-Za-z0-9]+", stem):
            value = normalize(part)
            if len(value) >= 3 and value not in TOKEN_STOPWORDS and value not in tokens:
                tokens.append(value)
    return normalized, tokens


def first_matching_lines(text: str, needles: Iterable[str], *, limit: int = 3) -> List[str]:
    results: List[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(needle and needle in lowered for needle in needles):
            snippet = line.strip()
            if snippet and snippet not in results:
                results.append(snippet)
        if len(results) >= limit:
            break
    return results


def benchmark_candidates(repo_root: Path, queries: List[str], tokens: List[str]) -> List[BenchmarkCandidate]:
    build_dir = repo_root / "build" / "benchmarks"
    candidates: List[BenchmarkCandidate] = []
    for source in sorted((repo_root / "benchmarks").glob("*.cc")):
        text = source.read_text(encoding="utf-8")
        lowered = text.lower()
        target = source.stem
        target_norm = normalize(target)
        score = 0
        reasons: List[str] = []

        for query in queries:
            if not query:
                continue
            if query == target_norm:
                score += 120
                reasons.append(f"target matches {query}")
            elif query in target_norm:
                score += 60
                reasons.append(f"target contains {query}")
            if query in lowered:
                score += 45
                reasons.append(f"source mentions {query}")

        for token in tokens:
            if token in target_norm:
                score += 12
                reasons.append(f"target token {token}")
            if token in lowered:
                score += 6
                reasons.append(f"source token {token}")

        if score == 0:
            continue

        hits = first_matching_lines(text, [*queries, *tokens])
        binary = build_dir / target if (build_dir / target).exists() else None
        reasons = list(dict.fromkeys(reasons))[:4]
        candidates.append(BenchmarkCandidate(score=score, target=target, source=source, reasons=reasons, hits=hits, binary=binary))

    return sorted(candidates)[:8]


def trace_candidates(repo_root: Path, queries: List[str], tokens: List[str]) -> List[TraceCandidate]:
    results: List[TraceCandidate] = []
    search_roots = [repo_root / "src", repo_root / "tests"]
    for root in search_roots:
        for source in root.rglob("*.[ch]c*"):
            text = source.read_text(encoding="utf-8", errors="ignore")
            lowered_path = normalize(str(source.relative_to(repo_root)))
            for label in TRACE_RE.findall(text):
                label_norm = normalize(label)
                score = 0
                for query in queries:
                    if query and query in label_norm:
                        score += 80
                    if query and query in lowered_path:
                        score += 30
                for token in tokens:
                    if token in label_norm:
                        score += 15
                    if token in lowered_path:
                        score += 8
                if score:
                    results.append(TraceCandidate(score=score, label=label, source=source))
    dedup: List[TraceCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(results):
        key = (item.label, str(item.source))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= 10:
            break
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description="Find likely BatchLAS benchmark and trace targets for a function, file, or symbol.")
    parser.add_argument("query", nargs="+", help="Source path, function name, benchmark family, or trace label fragment")
    args = parser.parse_args()

    repo_root = repo_root_from(Path(__file__))
    queries, tokens = query_terms(args.query)

    benchmarks = benchmark_candidates(repo_root, queries, tokens)
    traces = trace_candidates(repo_root, queries, tokens)

    print(f"Repository: {repo_root}")
    print(f"Query: {' '.join(args.query)}")
    print()

    if not benchmarks:
        print("Benchmark candidates: none")
    else:
        print("Benchmark candidates:")
        for index, candidate in enumerate(benchmarks, start=1):
            print(f"{index}. {candidate.target} (score {candidate.score})")
            print(f"   source: {candidate.source.relative_to(repo_root)}")
            if candidate.binary:
                print(f"   binary: {candidate.binary}")
            print(f"   why: {', '.join(candidate.reasons)}")
            for hit in candidate.hits:
                print(f"   hit: {hit}")
    print()

    if not traces:
        print("Trace scopes: none")
    else:
        print("Trace scopes:")
        for trace in traces:
            print(f"- {trace.label} [{trace.source.relative_to(repo_root)}]")

    print()
    print("Related helpers:")
    print("- evaluation/perf_eval.py supports trace-aware regression cases for stedc, steqr, sytrd_cta, ormqr_cta, and syev_cta.")
    print("- scripts/run_gemm_steady_profile.sh is the repo's existing combined nsys/ncu example.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
