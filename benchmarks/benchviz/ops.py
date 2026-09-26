"""The op registry: what to run per LAPACK op, over which grid, and its flop
count. Why only these ten ops, and why the BatchLAS arm is pinned `native`
rather than `auto`: README.md, "What is compared"."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

TYPES = ("float", "double", "cfloat", "cdouble")
TYPE_BYTES = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
# LAPACK's one-letter precision prefix; what a paper labels its curves with.
TYPE_PREFIX = {"float": "S", "double": "D", "cfloat": "C", "cdouble": "Z"}
TYPE_LABEL = {
    "float": "single", "double": "double",
    "cfloat": "single complex", "cdouble": "double complex",
}


def _cplx(t: str) -> bool:
    return t.startswith("c")


# Leading-order LAWN 41 counts; complex is 4x real, as in LAPACK's timing suite.
def _flops(real: float, t: str) -> float:
    return real * (4.0 if _cplx(t) else 1.0)


def flops_potrf(m, n, nrhs, t): return _flops(n ** 3 / 3.0, t)
def flops_getrf(m, n, nrhs, t): return _flops(2.0 * n ** 3 / 3.0, t)
def flops_getrs(m, n, nrhs, t): return _flops(2.0 * n * n * nrhs, t)
def flops_geqrf(m, n, nrhs, t): return _flops(2.0 * m * n * n - 2.0 * n ** 3 / 3.0, t)
def flops_orgqr(m, n, nrhs, t): return _flops(2.0 * m * n * n - 2.0 * n ** 3 / 3.0, t)
def flops_gesv(m, n, nrhs, t): return _flops(2.0 * n ** 3 / 3.0 + 2.0 * n * n * nrhs, t)
def flops_posv(m, n, nrhs, t): return _flops(n ** 3 / 3.0 + 2.0 * n * n * nrhs, t)
# ormqr applies Q (from an m x n geqrf, k = n reflectors) to an m x m C from the left.
def flops_ormqr(m, n, nrhs, t): return _flops(4.0 * m * m * n - 2.0 * m * n * n, t)


@dataclass(frozen=True)
class Arm:
    key: str                  # "batchlas" | "vendor"
    env: Tuple[Tuple[str, str], ...] = ()
    fb_arm: Optional[str] = None    # factor_bench --arms value
    bench_name: Optional[str] = None  # minibench --name filter (must be unambiguous)
    # A direct call that bypasses dispatch records no coverage row; the route is
    # then known by construction and stated here.
    fixed_route: Optional[str] = None


@dataclass
class OpSpec:
    name: str
    title: str                # what a figure calls it
    harness: str              # "factor_bench" | "minibench"
    binary: str
    types: Sequence[str]
    orders: Sequence[int]
    arms: Tuple[Arm, Arm]
    flops: Optional[Callable] = None
    # Device bytes per matrix / (n^2 sizeof T); caps the batch ladder. Conservative.
    footprint: float = 4.0
    nrhs: int = 0
    # minibench positional args after (n, batch) -- or the full arg builder.
    args: Optional[Callable[[int, int], List[int]]] = None
    max_batch: int = 32768
    notes: str = ""
    # Untimed same-process setup ops (getrs needs an LU): their routes pollute
    # coverage, so vendor-freedom is undetermined. README.md, "Routes".
    setup_ops: Tuple[str, ...] = ()
    # Composed ops take the outer `blocked` route in BOTH arms; the arm is decided
    # by these sub-ops' routes (factor_bench.cc composed_pins).
    composed_of: Tuple[str, ...] = ()

    def cell_args(self, n: int, batch: int) -> List[int]:
        return self.args(n, batch) if self.args else [n, batch]


def _fb(name, title, flops, nrhs=0, footprint=4.0, orders=None, notes="", setup_ops=(), composed_of=()):
    return OpSpec(
        name=name, title=title, harness="factor_bench", binary="factor_bench",
        types=TYPES,
        orders=orders or (4, 8, 16, 32, 64, 128, 256, 512),
        arms=(Arm("batchlas", fb_arm="native"), Arm("vendor", fb_arm="vendor")),
        flops=flops, nrhs=nrhs, footprint=footprint, notes=notes, setup_ops=setup_ops,
        composed_of=composed_of,
    )


OPS: Dict[str, OpSpec] = {}
for _s in (
    _fb("potrf", "Cholesky factorization", flops_potrf),
    _fb("getrf", "LU factorization", flops_getrf),
    _fb("getrs", "LU solve", flops_getrs, nrhs=1, footprint=5.0, setup_ops=("getrf",)),
    _fb("geqrf", "QR factorization", flops_geqrf),
    _fb("orgqr", "QR: form Q", flops_orgqr, footprint=5.0, setup_ops=("geqrf",)),
    _fb("gesv", "General linear solve", flops_gesv, nrhs=1, footprint=5.0,
        notes="composed: getrf + getrs, both sub-ops pinned per arm", composed_of=("getrf", "getrs")),
    _fb("posv", "SPD linear solve", flops_posv, nrhs=1, footprint=5.0,
        notes="composed: potrf + trsm, both sub-ops pinned per arm", composed_of=("potrf", "trsm")),
    OpSpec(
        name="ormqr", title="QR: apply Q", harness="minibench", binary="ormqr_benchmark",
        types=("float", "double"),
        orders=(8, 16, 32, 64, 128, 256, 512),
        arms=(Arm("batchlas", env=(("BATCHLAS_ORMQR_ROUTE", "native"),), bench_name="BM_ORMQR<"),
              Arm("vendor", env=(("BATCHLAS_ORMQR_ROUTE", "vendor"),), bench_name="BM_ORMQR<")),
        flops=flops_ormqr, footprint=5.0, setup_ops=("geqrf",),
        args=lambda n, b: [n, n, b],
    ),
    OpSpec(
        name="syev", title="Symmetric eigensolver", harness="minibench", binary="syev_benchmark",
        types=TYPES,
        orders=(8, 16, 32, 64, 128, 256, 512, 1024),
        arms=(Arm("batchlas", env=(("BATCHLAS_SYEV_ROUTE", "native"),), bench_name="BM_SYEV<"),
              Arm("vendor", env=(("BATCHLAS_SYEV_ROUTE", "vendor"),), bench_name="BM_SYEV<")),
        footprint=6.0,
        # n, batch, nb=0 (tuned), fuse=2 (tuned), jobz=1 (vectors), uplo=0 (lower)
        args=lambda n, b: [n, b, 0, 2, 1, 0],
        notes="eigenvalues and eigenvectors; throughput in matrices/s (no canonical flop count)",
    ),
    OpSpec(
        name="gesvd", title="Singular value decomposition", harness="minibench",
        binary="gesvd_vendor_benchmark",
        types=("float", "double"),
        orders=(4, 8, 16, 24, 32),
        arms=(Arm("batchlas", bench_name="BM_GESVD_BATCHLAS_CTA<", fixed_route="native:cta"),
              Arm("vendor", bench_name="BM_GESVD_CUSOLVER_JACOBI<", fixed_route="vendor:gesvdj_batched")),
        footprint=6.0,
        args=lambda n, b: [n, b, 1, 1],
        notes="cuSOLVER gesvdjBatched is capped at n = 32; vendor time includes the V -> V^H transpose",
    ),
):
    OPS[_s.name] = _s


# ----------------------------------------------------------------- the grid
@dataclass(frozen=True)
class Preset:
    name: str
    min_batch: int
    batch_step: int      # multiplicative step of the batch ladder
    order_stride: int    # take every k-th order of the op's ladder
    reps: int


PRESETS = {
    # Enough for all three figures, with a coarse heatmap: ~5 batches per order.
    "quick": Preset("quick", min_batch=64, batch_step=4, order_stride=1, reps=5),
    # Publication grid: every power of two in batch.
    "full": Preset("full", min_batch=32, batch_step=2, order_stride=1, reps=9),
    # Smoke test of the pipeline end to end.
    "smoke": Preset("smoke", min_batch=256, batch_step=16, order_stride=3, reps=3),
}


def batch_ladder(op: OpSpec, dtype: str, n: int, preset: Preset, mem_gib: float) -> List[int]:
    """Powers of `batch_step` up to the memory cap, largest first so the
    saturated point of each n (the speedup-vs-n curve) lands before the heatmap."""
    per_matrix = op.footprint * n * n * TYPE_BYTES[dtype]
    if op.nrhs:
        per_matrix += 3.0 * n * op.nrhs * TYPE_BYTES[dtype]
    cap = int(mem_gib * (1 << 30) // max(per_matrix, 1.0))
    cap = min(cap, op.max_batch)
    out, b = [], preset.min_batch
    while b <= cap:
        out.append(b)
        b *= preset.batch_step
    if not out:
        out = [max(1, cap)]
    return sorted(out, reverse=True)


@dataclass(frozen=True)
class Cell:
    op: str
    dtype: str
    m: int
    n: int
    nrhs: int
    batch: int

    def key(self, arm: str) -> str:
        return f"{self.op}|{self.dtype}|{self.m}|{self.n}|{self.nrhs}|{self.batch}|{arm}"


def plan_cells(ops: Sequence[str], types: Sequence[str], preset: Preset,
               mem_gib: float, orders: Optional[Sequence[int]] = None) -> List[Cell]:
    cells: List[Cell] = []
    for name in ops:
        op = OPS[name]
        for t in types:
            if t not in op.types:
                continue
            ladder = list(orders) if orders else list(op.orders)[:: preset.order_stride]
            for n in ladder:
                for b in batch_ladder(op, t, n, preset, mem_gib):
                    cells.append(Cell(name, t, n, n, op.nrhs, b))
    return cells
