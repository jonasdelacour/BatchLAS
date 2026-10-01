"""The op registry: what to run per LAPACK op, over which grid, and its flop
count. Why only these ten ops, and why the BatchLAS arm is pinned `native`
rather than `auto`: README.md, "What is compared"."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES = ("float", "double", "cfloat", "cdouble")
TYPE_BYTES = {"float": 4, "double": 8, "cfloat": 8, "cdouble": 16}
# LAPACK's one-letter precision prefix; what a paper labels its curves with.
TYPE_PREFIX = {"float": "S", "double": "D", "cfloat": "C", "cdouble": "Z"}
TYPE_LABEL = {
    "float": "single", "double": "double",
    "cfloat": "single complex", "cdouble": "double complex",
}


def vendor_name(op: str, backend: str = "cuda") -> str:
    v = OPS[op].vendor if op in OPS else ("vendor", "vendor")
    return v[1] if backend == "rocm" else v[0]


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
def _k(k, dflt):
    """A BLAS cell's third field: 0 on a square cell (the key it has always had),
    so the third dimension is then the op's square default."""
    return np.where(np.asarray(k) > 0, k, dflt)


def flops_gemm(m, n, k, t): return _flops(2.0 * m * n * _k(k, n), t)
def flops_gemv(m, n, k, t): return _flops(2.0 * m * n, t)
def flops_trsm(m, n, k, t): return _flops(1.0 * n * n * _k(k, n), t)
def flops_trmm(m, n, k, t): return _flops(1.0 * n ** 3, t)
def flops_syrk(m, n, k, t): return _flops(1.0 * n * (n + 1) * _k(k, n), t)
def flops_syr2k(m, n, k, t): return _flops(2.0 * n * n * _k(k, n), t)


SPMM_NNZ_ROW, SPMM_NRHS = 16, 16
def flops_spmm(m, n, k, t): return _flops(2.0 * n * SPMM_NNZ_ROW * _k(k, SPMM_NRHS), t)


# ormqr applies Q (from an m x n geqrf, k = n reflectors) to an m x m C from the left.
def flops_ormqr(m, n, nrhs, t): return _flops(4.0 * m * m * n - 2.0 * m * n * n, t)


# Dense to tridiagonal, and dense to band (its leading order is the same).
def flops_sytrd(m, n, nrhs, t): return _flops(4.0 * n ** 3 / 3.0, t)


@dataclass(frozen=True)
class Arm:
    key: str                  # "batchlas" | "vendor"
    env: Tuple[Tuple[str, str], ...] = ()
    fb_arm: Optional[str] = None    # factor_bench --arms value
    bench_name: Optional[str] = None  # minibench --name filter (must be unambiguous)
    # A direct call that bypasses dispatch records no coverage row; the route is
    # then known by construction and stated here.
    fixed_route: Optional[str] = None


@dataclass(frozen=True)
class Plane:
    """The 2-D shape sweep of a rectangular op: one cell per (x, y), each at its
    saturated batch, drawn as the speedup_2d and throughput_2d maps. `cell` maps
    (x, y) to the Cell's (m, n, third) and `coords` maps back. The point that is
    the op's square cell keeps the square cell's key, so both sweeps share it."""
    x: str                    # short names, for the dashboard: "n", "k", "q"
    y: str
    x_label: str              # axis labels, units in brackets
    y_label: str
    cell: Callable[[int, int], Tuple[int, int, int]]
    coords: Callable[[int, int, int], Tuple[int, int]]
    x_values: Optional[Sequence[int]] = None      # None: the op's orders
    valid: Callable[[int, int], bool] = lambda x, y: True
    # Operand area at (m, n, third) in units of n^2, times the op's footprint. It is
    # n^2 on the square cell, so both sweeps size an item alike there.
    area: Callable[[int, int, int], float] = lambda m, n, k: float(m) * n


def _tall(x_label="Columns $n$ [1]", y_label="Rows $m$ [1]", valid=lambda x, y: True,
          area=lambda m, n, k: float(m) * n) -> Plane:
    """x = n columns, y = m rows."""
    return Plane(x="n", y="m", x_label=x_label, y_label=y_label, valid=valid, area=area,
                 cell=lambda x, y: (y, x, 0), coords=lambda m, n, k: (n, m))


def _third(x, x_label, y_label, square_k=0, x_values=None, area=lambda m, n, k: float(n) * n) -> Plane:
    """y = the order n (m = n), x = the cell's third field. The square cell stores
    `square_k` there: 0 for the BLAS ops, meaning third = n; 1 for the solves (nrhs)."""
    return Plane(x=x, y="n", x_label=x_label, y_label=y_label, x_values=x_values, area=area,
                 cell=lambda xv, yv: (yv, yv, 0 if square_k == 0 and xv == yv else xv),
                 coords=lambda m, n, k: (k if k else n, n))


@dataclass
class OpSpec:
    name: str
    title: str                # what a figure calls it
    harness: str              # "factor_bench" | "minibench"
    binary: str
    types: Sequence[str]
    orders: Sequence[int]
    arms: Tuple[Arm, ...]           # (batchlas,) for an op no vendor library has
    flops: Optional[Callable] = None
    # Device bytes per matrix / (n^2 sizeof T); caps the batch ladder. Conservative.
    footprint: float = 4.0
    nrhs: int = 0
    # minibench positional args of a cell, from its (m, n, third, batch).
    args: Optional[Callable[[int, int, int, int], List[int]]] = None
    # The same, for a harness whose arguments also depend on the precision.
    typed_args: Optional[Callable[[int, int, int, int, str], List[int]]] = None
    max_batch: int = 32768
    notes: str = ""
    # Untimed same-process setup ops (getrs needs an LU): their routes pollute
    # coverage, so vendor-freedom is undetermined. README.md, "Routes".
    setup_ops: Tuple[str, ...] = ()
    min_order: int = 1
    max_order: int = 4096
    group: str = "LAPACK"
    n_label: str = "Matrix Order $n$ [1]"
    vendor: Tuple[str, str] = ("cuSOLVER", "rocSOLVER")   # the library the vendor arm reaches: (CUDA, ROCm)
    # Device bytes of one item at (n, third), when n^2 * footprint is the wrong model (sparse).
    bytes_per_item: Optional[Callable[[int, int, str], float]] = None
    # Composed ops take the outer `blocked` route in BOTH arms; the arm is decided
    # by these sub-ops' routes (factor_bench.cc composed_pins).
    composed_of: Tuple[str, ...] = ()
    # Rectangular ops also get a 2-D shape sweep; None for the square-only ones.
    plane: Optional[Plane] = None
    # The dispatch Op this spec measures (coverage rows, BATCHLAS_<OP>_ROUTE); default name.
    dispatch_op: str = ""
    fb_flags: Tuple[str, ...] = ()   # extra factor_bench flags, e.g. --uplo=upper
    opt_in: bool = False             # left out of `--ops all`

    def __post_init__(self):
        self.dispatch_op = self.dispatch_op or self.name

    @property
    def single(self) -> bool:
        """No vendor arm: a campaign plots its throughput, and a speedup only
        appears in a log comparison (compare.py)."""
        return len(self.arms) == 1

    def cell_args(self, cell: "Cell") -> List[int]:
        if self.typed_args:
            return self.typed_args(cell.m, cell.n, cell.nrhs, cell.batch, cell.dtype)
        return self.args(cell.m, cell.n, cell.nrhs, cell.batch) if self.args else [cell.n, cell.batch]

    def is_square(self, m: int, n: int, k: int) -> bool:
        """Is (m, n, third) the op's square cell, the one the n-figures plot?"""
        return (m, k) == self.shape(n)

    def shape_text(self, m: int, n: int, k: int) -> str:
        if self.plane is None or self.is_square(m, n, k):
            return f"n {n}"
        x, y = self.plane.coords(m, n, k)
        return f"{self.plane.y} {y} · {self.plane.x} {x}"

    def shape(self, n: int) -> Tuple[int, int]:
        """(m, third dim) of the cell at order n; the third is nrhs or k."""
        return n, self.nrhs


def _fb(name, title, flops, nrhs=0, footprint=4.0, orders=None, notes="", setup_ops=(), composed_of=(), plane=None):
    return OpSpec(
        name=name, title=title, harness="factor_bench", binary="factor_bench",
        types=TYPES,
        orders=orders or (4, 8, 16, 32, 64, 128, 256, 512),
        arms=(Arm("batchlas", fb_arm="native"), Arm("vendor", fb_arm="vendor")),
        flops=flops, nrhs=nrhs, footprint=footprint, notes=notes, setup_ops=setup_ops, max_order=2048,
        composed_of=composed_of, plane=plane,
    )


def _blas(name, title, binary, bench, flops, native="native", types=TYPES, orders=(), footprint=3.0,
          args=None, notes="", bytes_per_item=None, n_label="Matrix Order $n$ [1]", plane=None):
    var = f"BATCHLAS_{name.upper()}_ROUTE"
    return OpSpec(
        name=name, title=title, harness="minibench", binary=binary, types=types, orders=orders,
        arms=(Arm("batchlas", env=((var, native),), bench_name=bench), Arm("vendor", env=((var, "vendor"),), bench_name=bench)),
        flops=flops, footprint=footprint, args=args, notes=notes, group="BLAS", bytes_per_item=bytes_per_item,
        n_label=n_label, plane=plane,
        max_batch=65536, max_order=max(orders),
        vendor=("cuSPARSE", "rocSPARSE") if name == "spmm" else ("cuBLAS", "rocBLAS"),
    )


def _stage(name, title, binary, bench, route, orders, types=TYPES, footprint=4.0, flops=None,
           args=None, typed_args=None, max_order=None, notes=""):
    return OpSpec(
        name=name, title=title, harness="minibench", binary=binary, types=types, orders=orders,
        # The route is the one the harness calls directly; coverage still records
        # the sub-ops under it, which is what decides vendor-freedom.
        arms=(Arm("batchlas", bench_name=bench, fixed_route=route),),
        flops=flops, footprint=footprint, args=args, typed_args=typed_args, notes=notes,
        group="Eigensolver stages", max_order=max_order or max(orders), vendor=("none", "none"),
    )


def sytrd_nb(n: int, dtype: str) -> int:
    """syev_blocked's panel width: tuning_params.hh's SYTRD_BLOCK_SIZE_* buckets,
    with the complex override syev_blocked.cc applies at 256 < n <= 512."""
    if dtype.startswith("c") and 256 < n <= 512:
        return 32
    return 8 if n <= 128 else 16 if n <= 256 else 8 if n <= 512 else 48


def two_stage_kd(n: int) -> int:
    """choose_two_stage_kd (two_stage_common.hh): kd = 32, below n."""
    return max(1, min(32, n - 1))


# factor_bench takes n <= m, and potrf/getrf only square: README.md, "Rectangular ops".
_QR_PLANE = _tall(valid=lambda x, y: x <= y)
_SOLVE_PLANE = _third("nrhs", "Right-Hand Sides [1]", "Matrix Order $n$ [1]", square_k=1,
                      x_values=(1, 4, 16, 64, 256))

OPS: Dict[str, OpSpec] = {}
for _s in (
    _fb("potrf", "Cholesky factorization", flops_potrf),
    _fb("getrf", "LU factorization", flops_getrf),
    _fb("getrs", "LU solve", flops_getrs, nrhs=1, footprint=5.0, setup_ops=("getrf",), plane=_SOLVE_PLANE),
    _fb("geqrf", "QR factorization", flops_geqrf, plane=_QR_PLANE),
    _fb("orgqr", "QR: form Q", flops_orgqr, footprint=5.0, setup_ops=("geqrf",), plane=_QR_PLANE),
    _fb("gesv", "General linear solve", flops_gesv, nrhs=1, footprint=5.0, plane=_SOLVE_PLANE,
        notes="native arm is the shipped route: fused tiny in its window, else getrf + getrs", composed_of=("getrf", "getrs")),
    _fb("posv", "SPD linear solve", flops_posv, nrhs=1, footprint=5.0, plane=_SOLVE_PLANE,
        notes="native arm is the shipped route: fused tiny in its window, else potrf + trsm", composed_of=("potrf", "trsm")),
    OpSpec(
        name="ormqr", title="QR: apply Q", harness="minibench", binary="ormqr_benchmark",
        types=("float", "double"),
        orders=(8, 16, 32, 64, 128, 256, 512),
        arms=(Arm("batchlas", env=(("BATCHLAS_ORMQR_ROUTE", "native"),), bench_name="BM_ORMQR<"),
              Arm("vendor", env=(("BATCHLAS_ORMQR_ROUTE", "vendor"),), bench_name="BM_ORMQR<")),
        flops=flops_ormqr, footprint=5.0, setup_ops=("geqrf",),
        args=lambda m, n, k, b: [m, n, b],
        # Q is m x m and outweighs A (m x n).
        plane=_tall(x_label="Reflectors $n$ [1]", y_label="Order of $Q$, $m$ [1]", valid=lambda x, y: x <= y,
                    area=lambda m, n, k: float(m) * m),
    ),
    OpSpec(
        name="syev", title="Symmetric eigensolver", harness="minibench", binary="syev_benchmark",
        types=TYPES,
        orders=(8, 16, 32, 64, 128, 256, 512, 1024),
        arms=(Arm("batchlas", env=(("BATCHLAS_SYEV_ROUTE", "native"),), bench_name="BM_SYEV<"),
              Arm("vendor", env=(("BATCHLAS_SYEV_ROUTE", "vendor"),), bench_name="BM_SYEV<")),
        footprint=6.0,
        # n, batch, nb=0 (tuned), fuse=2 (tuned), jobz=1 (vectors), uplo=0 (lower)
        args=lambda m, n, k, b: [n, b, 0, 2, 1, 0],
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
        args=lambda m, n, k, b: [n, b, 1, 1],
        notes="cuSOLVER gesvdjBatched is capped at n = 32; vendor time includes the V -> V^H transpose",
        max_order=32,
    ),
    # ---------------------------------------------------------------- syev's stages
    # No vendor library ships these batched, so one arm; README.md, "Sub-operations".
    # The arguments are the defaults syev passes, spelled explicitly so that any
    # older build's harness runs the same cell.
    _stage("stedc", "Tridiagonal divide and conquer", "stedc_benchmark", "BM_STEDC<", "native:stedc",
           orders=(32, 64, 128, 256, 320, 512, 640, 1024), footprint=14.0, types=("float", "double"),
           # Measured peak, RTX 4090 float: 13 n^2 per matrix at n = 256, 6.6 at 1024.
           # rec_threshold 0 and Auto merge / algorithm: the tuning tables, as syev gets them.
           args=lambda m, n, k, b: [n, b, 0, -1, 0, 0, -1],
           notes="eigenvalues and eigenvectors of a random tridiagonal; tuned threshold, Auto merge and driver"),
    _stage("steqr", "Tridiagonal QR iteration", "steqr_benchmark", "BM_STEQR<", "native:steqr",
           orders=(8, 16, 32, 64, 128), footprint=64.0, types=("float", "double"),
           # Measured peak: ~60 n^2 per matrix at every n; the workspace, not the operands.
           # max_sweeps 50 and interleaved working vectors, as SteqrParams defaults;
           # the harness's legacy schema pins the Wilkinson shift.
           args=lambda m, n, k, b: [n, b, 50, 1, 0],
           notes="eigenvalues and eigenvectors; n <= 32 is steqr_cta, above it steqr_wg; Wilkinson shift"),
    _stage("sytrd", "Tridiagonal reduction, blocked", "sytrd_blocked_benchmark", "BM_SYTRD_BLOCKED<",
           "native:blocked", orders=(32, 64, 128, 256, 512, 1024), flops=flops_sytrd,
           typed_args=lambda m, n, k, b, t: [n, b, sytrd_nb(n, t), 0],
           notes="syev_blocked's stage 1, at its panel width nb"),
    _stage("sytrd_cta", "Tridiagonal reduction, one CTA", "sytrd_cta_benchmark", "BM_SYTRD_CTA<", "native:cta",
           orders=(4, 8, 16, 24, 32), flops=flops_sytrd, types=("float", "double"), max_order=32,
           args=lambda m, n, k, b: [n, b, 0, 0], notes="the n <= 32 reduction syev_cta runs"),
    _stage("sy2sb", "Two-stage: dense to band", "sytrd_sy2sb_benchmark", "BM_SYTRD_SY2SB<", "native:sy2sb",
           orders=(64, 128, 256, 512, 1024), flops=flops_sytrd,
           args=lambda m, n, k, b: [n, b, two_stage_kd(n), 0],
           notes="syev_two_stage's stage 1 at its band width kd = 32"),
    _stage("sb2st", "Two-stage: band to tridiagonal", "sytrd_sb2st_benchmark", "BM_SYTRD_SB2ST<", "native:sb2st",
           orders=(64, 128, 256, 512, 1024),
           args=lambda m, n, k, b: [n, b, two_stage_kd(n), 0],
           notes="syev_two_stage's bulge chase from kd = 32; throughput in matrices/s"),
    # ---------------------------------------------------------------- BLAS
    # The square cells are m = n = k and store k as 0. Arms are the canonical
    # BATCHLAS_<OP>_ROUTE only: legacy spellings mean different things per op
    # (route_env.hh:106-134).
    _blas("gemm", "General matrix multiply", "gemm_benchmark", "BM_GEMM<", flops_gemm,
          orders=(8, 16, 32, 64, 128, 256, 512, 1024), footprint=3.0, args=lambda m, n, k, b: [m, n, k or n, b],
          plane=_third("k", "Inner Dimension $k$ [1]", "Output Order $m = n$ [1]",
                       area=lambda m, n, k: (m * (k or n) + (k or n) * n + m * n) / 3.0)),
    _blas("gemv", "Matrix-vector multiply", "gemv_benchmark", "BM_GEMV<", flops_gemv,
          orders=(16, 32, 64, 128, 256, 512, 1024, 2048), footprint=1.2, args=lambda m, n, k, b: [m, n, b],
          plane=_tall()),
    _blas("trsm", "Triangular solve", "trsm_benchmark", "BM_TRSM<", flops_trsm,
          orders=(8, 16, 32, 64, 128, 256, 512), footprint=3.0, args=lambda m, n, k, b: [n, k or n, b],
          plane=_third("q", "Right-Hand Sides $q$ [1]", "Triangle Order $n$ [1]",
                       area=lambda m, n, k: (n * n + 2.0 * n * (k or n)) / 3.0)),
    # trmm_benchmark's operands only agree at m = n = k, so trmm stays square.
    _blas("trmm", "Triangular multiply", "trmm_benchmark", "BM_TRMM<", flops_trmm, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=3.0,
          args=lambda m, n, k, b: [n, n, n, b], notes="native triangular-tile kernel is CUDA float only"),
    _blas("syrk", "Symmetric rank-k update", "syrk_benchmark", "BM_SYRK<", flops_syrk, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=3.0,
          args=lambda m, n, k, b: [n, k or n, k or n, b],
          plane=_third("k", "Rank $k$ [1]", "Matrix Order $n$ [1]",
                       area=lambda m, n, k: (n * n + 2.0 * n * (k or n)) / 3.0),
          notes="native triangular-tile kernel is float only (double records no route); plain native is a wrong-answer route"),
    _blas("syr2k", "Symmetric rank-2k update", "syr2k_benchmark", "BM_SYR2K<", flops_syr2k, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=4.0,
          args=lambda m, n, k, b: [n, k or n, k or n, b],
          plane=_third("k", "Rank $k$ [1]", "Matrix Order $n$ [1]",
                       area=lambda m, n, k: (n * n + 3.0 * n * (k or n)) / 4.0),
          notes="native triangular-tile kernel is float only"),
    _blas("spmm", "Sparse x dense multiply", "spmm_benchmark", "BM_SPMM_Grid<", flops_spmm,
          orders=(256, 512, 1024, 2048, 4096, 8192, 16384),
          args=lambda m, n, k, b: [n, SPMM_NNZ_ROW, k or SPMM_NRHS, b, 0, 0, 1],
          n_label="Matrix Rows $n$ [1]",
          bytes_per_item=lambda n, k, t: 3.0 * (n * SPMM_NNZ_ROW * (TYPE_BYTES[t] + 4)
                                                + 2 * n * (k or SPMM_NRHS) * TYPE_BYTES[t]),
          plane=Plane(x="nrhs", y="n", x_label="Right-Hand Sides [1]", y_label="Matrix Rows $n$ [1]",
                      x_values=(4, 8, 16, 32, 64, 128),
                      cell=lambda x, y: (y, y, 0 if x == SPMM_NRHS else x),
                      coords=lambda m, n, k: (k or SPMM_NRHS, n)),
          notes=f"CSR, random pattern, {SPMM_NNZ_ROW} nonzeros per row, {SPMM_NRHS} right-hand sides; n is the row count"),
):
    OPS[_s.name] = _s

# potrf Upper as its own op: Cell has no uplo field, and LPanel/Blocked are Lower-only
# in supports(), so an Upper sweep is mostly a record of which routes refuse it.
OPS["potrf_upper"] = OpSpec(**{**OPS["potrf"].__dict__, "name": "potrf_upper",
                               "title": "Cholesky factorization (Upper)",
                               "dispatch_op": "potrf", "fb_flags": ("--uplo=upper",), "opt_in": True})


# ----------------------------------------------------------------- route sweep
# Arm `route:<origin>:<algorithm>` pins BATCHLAS_<OP>_ROUTE to that route, so a cost
# model can be fitted to every route on the same cell (docs/design/routing-cost-model.md).
ROUTE_ARM = "route:"
_DISPATCH = REPO_ROOT / "include" / "batchlas" / "blas" / "dispatch"
# Fallback when the headers are unreadable; source of truth is k<Op>Order in
# include/batchlas/blas/dispatch/route_<op>.hh (native entries only, in order).
ROUTE_FALLBACK: Dict[str, Tuple[str, ...]] = {
    "potrf": ("native:tiny", "native:cta", "native:lpanel", "native:blocked"),
    "getrf": ("native:tiny", "native:cta", "native:blocked"),
    "getrs": ("native:cta", "native:blocked"),
    "geqrf": ("native:tiny", "native:cta", "native:blocked"),
    "orgqr": ("native:blocked",),
    "gesv": ("native:tiny", "native:blocked"),
    "posv": ("native:tiny", "native:cta", "native:blocked"),
}


def _algorithm_names() -> Dict[str, str]:
    """Algorithm enumerator -> its to_string() spelling, read from route.hh."""
    text = (_DISPATCH / "route.hh").read_text()
    return dict(re.findall(r'case Algorithm::(\w+):\s*return "(\w+)";', text))


def native_routes(op: str) -> Tuple[str, ...]:
    """The op's native routes in the library's k<Op>Order, as `native:<algorithm>`.
    The vendor route is the separate `vendor` arm."""
    op = OPS[op].dispatch_op if op in OPS else op
    try:
        names = _algorithm_names()
        text = (_DISPATCH / f"route_{op}.hh").read_text()
        body = re.search(r"inline constexpr Route k\w+Order\[\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
        out = tuple(f"native:{names[a]}" for o, a in re.findall(r"\{Origin::(\w+),\s*Algorithm::(\w+)\}", body)
                    if o == "Native")
        if out:
            return out
    except (OSError, AttributeError, KeyError):
        pass
    return ROUTE_FALLBACK.get(op, ())


def route_arm(op: str, route: str) -> Arm:
    """The arm pinning `route` (e.g. "native:lpanel"). factor_bench takes the arm's
    name as its pin; the minibench harnesses read the env var."""
    spec = OPS[op]
    var = f"BATCHLAS_{spec.dispatch_op.upper()}_ROUTE"
    bench = next((a.bench_name for a in spec.arms if a.bench_name), None)
    return Arm(ROUTE_ARM + route, env=((var, route),), fb_arm=route, bench_name=bench)


def op_arms(op: str, sweep: str = "ab") -> Tuple[Arm, ...]:
    """`ab`: the batchlas/vendor pair. `routes`: every native route plus vendor."""
    spec = OPS[op]
    if sweep == "ab":
        return tuple(spec.arms)
    if sweep != "routes":
        raise ValueError(f"sweep {sweep!r}")
    vendor = tuple(a for a in spec.arms if a.key == "vendor")
    return tuple(route_arm(op, r) for r in native_routes(op)) + vendor


def find_arm(op: str, key: str) -> Arm:
    for a in OPS[op].arms:
        if a.key == key:
            return a
    if key.startswith(ROUTE_ARM):
        return route_arm(op, key[len(ROUTE_ARM):])
    raise KeyError(f"{op} has no arm {key}")


# ----------------------------------------------------------------- the grid
def parse_list(spec) -> List[int]:
    """"4,8,16:512,24" -> ints. `a:b` doubles from a to b; `a:b:s` steps by s."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return sorted({int(x) for x in spec})
    try:
        return _parse_list(str(spec))
    except ValueError:
        raise ValueError(f"cannot read {spec!r}: use a list (16,32), a doubling range (4:512) "
                         f"or a stepped range (8:128:8)") from None


def _parse_list(spec: str) -> List[int]:
    out = set()
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) == 1:
            out.add(int(parts[0]))
            continue
        lo, hi = int(parts[0]), int(parts[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"bad range {tok!r}")
        if len(parts) == 3:
            out.update(range(lo, hi + 1, max(1, int(parts[2]))))
        else:
            v = lo
            while v <= hi:
                out.add(v)
                v *= 2
    return sorted(out)


@dataclass
class Grid:
    """What a campaign measures. `orders=None` means each op's own ladder."""
    orders: Optional[List[int]] = None
    order_stride: int = 1
    batch_mode: str = "ladder"          # ladder | list | saturated
    batch_min: int = 64
    batch_max: int = 32768
    batch_step: int = 4
    batches: Optional[List[int]] = None
    reps: int = 5
    mem_gib: float = 3.0
    # Also sweep each rectangular op's Plane (at the saturated batch). False here so a
    # campaign saved before the field existed keeps its plan; every preset turns it on.
    rect: bool = False
    name: str = "custom"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        g = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if g.orders is not None:
            g.orders = parse_list(g.orders)
        if g.batches is not None:
            g.batches = parse_list(g.batches)
        if g.batch_mode not in ("ladder", "list", "saturated"):
            raise ValueError(f"batch_mode {g.batch_mode!r}")
        if g.batch_step < 2 and g.batch_mode == "ladder":
            raise ValueError("batch_step must be >= 2")
        return g

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        if "grid" in cfg:
            return cls.from_dict(cfg["grid"])
        # Campaigns from before grids existed; "imported" ones never had a preset.
        name = cfg.get("preset", "quick")
        g = PRESETS.get(name, PRESETS["quick"])
        return cls.from_dict({**g.to_dict(), "mem_gib": cfg.get("mem_gib") or g.mem_gib,
                              "orders": cfg.get("orders"), "name": name, "rect": False})


PRESETS = {
    "smoke": Grid(order_stride=3, batch_min=256, batch_step=16, reps=3, rect=True, name="smoke"),
    "saturation": Grid(batch_mode="saturated", reps=7, rect=True, name="saturation"),
    "quick": Grid(batch_min=64, batch_step=4, reps=5, rect=True, name="quick"),
    "full": Grid(batch_min=32, batch_step=2, reps=9, rect=True, name="full"),
}


def memory_cap(op: "OpSpec", dtype: str, n: int, grid: Grid, m: Optional[int] = None, k: Optional[int] = None) -> int:
    """Largest batch that fits the budget; (m, k) default to the op's square cell."""
    if m is None:
        m, k = op.shape(n)
    if op.bytes_per_item:
        per_matrix = op.bytes_per_item(n, k, dtype)
    else:
        area = op.plane.area(m, n, k) if op.plane else float(m) * n
        per_matrix = op.footprint * area * TYPE_BYTES[dtype]
        if op.nrhs:
            per_matrix += 3.0 * n * k * TYPE_BYTES[dtype]
    return max(1, min(int(grid.mem_gib * (1 << 30) // max(per_matrix, 1.0)), op.max_batch, grid.batch_max))


def batch_ladder(op: "OpSpec", dtype: str, n: int, grid: Grid, mem_gib: Optional[float] = None) -> List[int]:
    """Batches for one (op, n), largest first so each n's saturated point (the
    speedup-vs-n curve) lands before the rest of the heatmap."""
    if mem_gib is not None:
        grid = Grid.from_dict({**grid.to_dict(), "mem_gib": mem_gib})
    cap = memory_cap(op, dtype, n, grid)
    if grid.batch_mode == "saturated":
        out = [1 << (cap.bit_length() - 1)]
    elif grid.batch_mode == "list":
        out = [b for b in (grid.batches or []) if b <= cap] or [1 << (cap.bit_length() - 1)]
    else:
        out, b = [], grid.batch_min
        while b <= cap:
            out.append(b)
            b *= grid.batch_step
        out = out or [cap]
    return sorted(set(out), reverse=True)


def op_orders(op: "OpSpec", grid: Grid) -> List[int]:
    base = list(grid.orders) if grid.orders else list(op.orders)[:: max(1, grid.order_stride)]
    return [n for n in base if op.min_order <= n <= op.max_order]


@dataclass(frozen=True)
class Cell:
    op: str
    dtype: str
    m: int
    n: int
    nrhs: int
    batch: int

    def key(self, arm: str, pass_: int = 0) -> str:
        return row_key(self.__dict__, arm, pass_)


def row_key(r: dict, arm: str, pass_: int = 0) -> str:
    """A (cell, arm) result's identity; a repeat pass (route sweeps) is its own row."""
    k = f"{r['op']}|{r['dtype']}|{r['m']}|{r['n']}|{r['nrhs']}|{r['batch']}|{arm}"
    return f"{k}|p{pass_}" if pass_ else k


def plan_cells(ops: Sequence[str], types: Sequence[str], grid: Grid, mem_gib: Optional[float] = None,
               orders: Optional[Sequence[int]] = None) -> List[Cell]:
    if mem_gib is not None or orders:
        grid = Grid.from_dict({**grid.to_dict(), **({"mem_gib": mem_gib} if mem_gib is not None else {}),
                               **({"orders": list(orders)} if orders else {})})
    cells: List[Cell] = []
    for name in ops:
        op = OPS[name]
        for t in types:
            if t not in op.types:
                continue
            for n in op_orders(op, grid):
                for b in batch_ladder(op, t, n, grid):
                    m, k = op.shape(n)
                    cells.append(Cell(name, t, m, n, k, b))
            if grid.rect and op.plane:
                cells += plane_cells(op, t, grid, have=set(cells))
    return cells


def plane_axes(op: "OpSpec", grid: Grid) -> Tuple[List[int], List[int]]:
    ys = op_orders(op, grid)
    xs = list(op.plane.x_values)[:: max(1, grid.order_stride)] if op.plane.x_values else ys
    return xs, ys


def plane_cells(op: "OpSpec", dtype: str, grid: Grid, have=frozenset()) -> List[Cell]:
    """One cell per valid (x, y), at the largest power-of-two batch that fits: a
    map over shapes can show one batch per shape, and only the saturated one is a
    fair ratio. A square point the square sweep already measures is not repeated."""
    xs, ys = plane_axes(op, grid)
    swept = {(h.m, h.n, h.nrhs) for h in have if h.op == op.name and h.dtype == dtype}
    out = []
    for y in ys:
        for x in xs:
            if not op.plane.valid(x, y):
                continue
            m, n, k = op.plane.cell(x, y)
            if max(m, n) > op.max_order or (op.is_square(m, n, k) and (m, n, k) in swept):
                continue
            cap = memory_cap(op, dtype, n, grid, m, k)
            out.append(Cell(op.name, dtype, m, n, k, 1 << (cap.bit_length() - 1)))
    return out
