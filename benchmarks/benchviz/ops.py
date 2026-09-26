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
# BLAS cells are square (m = n = k, as the benchmark args are built), so the
# counts are written in n alone; the cell's third field is not k for them.
def flops_gemm(m, n, k, t): return _flops(2.0 * n ** 3, t)
def flops_gemv(m, n, k, t): return _flops(2.0 * n ** 2, t)
def flops_trsm(m, n, k, t): return _flops(1.0 * n ** 3, t)
def flops_trmm(m, n, k, t): return _flops(1.0 * n ** 3, t)
def flops_syrk(m, n, k, t): return _flops(1.0 * n * (n + 1) * n, t)
def flops_syr2k(m, n, k, t): return _flops(2.0 * n ** 3, t)


SPMM_NNZ_ROW, SPMM_NRHS = 16, 16
def flops_spmm(m, n, k, t): return _flops(2.0 * n * SPMM_NNZ_ROW * SPMM_NRHS, t)


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
    min_order: int = 1
    max_order: int = 4096
    group: str = "LAPACK"
    n_label: str = "Matrix Order $n$ [1]"
    vendor: Tuple[str, str] = ("cuSOLVER", "rocSOLVER")   # the library the vendor arm reaches: (CUDA, ROCm)
    # Device bytes of one matrix at order n, when n^2 * footprint is the wrong model (sparse).
    bytes_per_item: Optional[Callable[[int, str], float]] = None
    # Composed ops take the outer `blocked` route in BOTH arms; the arm is decided
    # by these sub-ops' routes (factor_bench.cc composed_pins).
    composed_of: Tuple[str, ...] = ()

    def cell_args(self, n: int, batch: int) -> List[int]:
        return self.args(n, batch) if self.args else [n, batch]

    def shape(self, n: int) -> Tuple[int, int]:
        """(m, third dim) of the cell at order n; the third is nrhs or k."""
        return n, self.nrhs


def _fb(name, title, flops, nrhs=0, footprint=4.0, orders=None, notes="", setup_ops=(), composed_of=()):
    return OpSpec(
        name=name, title=title, harness="factor_bench", binary="factor_bench",
        types=TYPES,
        orders=orders or (4, 8, 16, 32, 64, 128, 256, 512),
        arms=(Arm("batchlas", fb_arm="native"), Arm("vendor", fb_arm="vendor")),
        flops=flops, nrhs=nrhs, footprint=footprint, notes=notes, setup_ops=setup_ops, max_order=1024,
        composed_of=composed_of,
    )


def _blas(name, title, binary, bench, flops, native="native", types=TYPES, orders=(), footprint=3.0,
          args=None, notes="", bytes_per_item=None, n_label="Matrix Order $n$ [1]"):
    var = f"BATCHLAS_{name.upper()}_ROUTE"
    return OpSpec(
        name=name, title=title, harness="minibench", binary=binary, types=types, orders=orders,
        arms=(Arm("batchlas", env=((var, native),), bench_name=bench), Arm("vendor", env=((var, "vendor"),), bench_name=bench)),
        flops=flops, footprint=footprint, args=args, notes=notes, group="BLAS", bytes_per_item=bytes_per_item,
        n_label=n_label,
        max_batch=65536, max_order=max(orders),
        vendor=("cuSPARSE", "rocSPARSE") if name == "spmm" else ("cuBLAS", "rocBLAS"),
    )


OPS: Dict[str, OpSpec] = {}
for _s in (
    _fb("potrf", "Cholesky factorization", flops_potrf),
    _fb("getrf", "LU factorization", flops_getrf),
    _fb("getrs", "LU solve", flops_getrs, nrhs=1, footprint=5.0, setup_ops=("getrf",)),
    _fb("geqrf", "QR factorization", flops_geqrf),
    _fb("orgqr", "QR: form Q", flops_orgqr, footprint=5.0, setup_ops=("geqrf",)),
    _fb("gesv", "General linear solve", flops_gesv, nrhs=1, footprint=5.0,
        notes="native arm is the shipped route: fused tiny in its window, else getrf + getrs", composed_of=("getrf", "getrs")),
    _fb("posv", "SPD linear solve", flops_posv, nrhs=1, footprint=5.0,
        notes="native arm is the shipped route: fused tiny in its window, else potrf + trsm", composed_of=("potrf", "trsm")),
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
        max_order=32,
    ),
    # ---------------------------------------------------------------- BLAS
    # All square (m = n = k). Arms are the canonical BATCHLAS_<OP>_ROUTE only:
    # legacy spellings mean different things per op (route_env.hh:106-134).
    _blas("gemm", "General matrix multiply", "gemm_benchmark", "BM_GEMM<", flops_gemm,
          orders=(8, 16, 32, 64, 128, 256, 512, 1024), footprint=3.0, args=lambda n, b: [n, n, n, b]),
    _blas("gemv", "Matrix-vector multiply", "gemv_benchmark", "BM_GEMV<", flops_gemv,
          orders=(16, 32, 64, 128, 256, 512, 1024, 2048), footprint=1.2, args=lambda n, b: [n, n, b]),
    _blas("trsm", "Triangular solve", "trsm_benchmark", "BM_TRSM<", flops_trsm,
          orders=(8, 16, 32, 64, 128, 256, 512), footprint=3.0, args=lambda n, b: [n, n, b]),
    _blas("trmm", "Triangular multiply", "trmm_benchmark", "BM_TRMM<", flops_trmm, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=3.0,
          args=lambda n, b: [n, n, n, b], notes="native triangular-tile kernel is CUDA float only"),
    _blas("syrk", "Symmetric rank-k update", "syrk_benchmark", "BM_SYRK<", flops_syrk, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=3.0,
          args=lambda n, b: [n, n, n, b], notes="native triangular-tile kernel is float only (double records no route); plain native is a wrong-answer route"),
    _blas("syr2k", "Symmetric rank-2k update", "syr2k_benchmark", "BM_SYR2K<", flops_syr2k, native="triangular",
          types=("float",), orders=(16, 32, 64, 128, 256, 512, 1024), footprint=4.0,
          args=lambda n, b: [n, n, n, b], notes="native triangular-tile kernel is float only"),
    _blas("spmm", "Sparse x dense multiply", "spmm_benchmark", "BM_SPMM_Grid<", flops_spmm,
          orders=(256, 512, 1024, 2048, 4096, 8192, 16384),
          args=lambda n, b: [n, SPMM_NNZ_ROW, SPMM_NRHS, b, 0, 0, 1],
          n_label="Matrix Rows $n$ [1]",
          bytes_per_item=lambda n, t: 3.0 * (n * SPMM_NNZ_ROW * (TYPE_BYTES[t] + 4) + 2 * n * SPMM_NRHS * TYPE_BYTES[t]),
          notes=f"CSR, random pattern, {SPMM_NNZ_ROW} nonzeros per row, {SPMM_NRHS} right-hand sides; n is the row count"),
):
    OPS[_s.name] = _s


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
                              "orders": cfg.get("orders"), "name": name})


PRESETS = {
    "smoke": Grid(order_stride=3, batch_min=256, batch_step=16, reps=3, name="smoke"),
    "saturation": Grid(batch_mode="saturated", reps=7, name="saturation"),
    "quick": Grid(batch_min=64, batch_step=4, reps=5, name="quick"),
    "full": Grid(batch_min=32, batch_step=2, reps=9, name="full"),
}


def memory_cap(op: "OpSpec", dtype: str, n: int, grid: Grid) -> int:
    per_matrix = op.footprint * n * n * TYPE_BYTES[dtype]
    if op.bytes_per_item:
        per_matrix = op.bytes_per_item(n, dtype)
    elif op.nrhs:
        per_matrix += 3.0 * n * op.nrhs * TYPE_BYTES[dtype]
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

    def key(self, arm: str) -> str:
        return f"{self.op}|{self.dtype}|{self.m}|{self.n}|{self.nrhs}|{self.batch}|{arm}"


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
    return cells
