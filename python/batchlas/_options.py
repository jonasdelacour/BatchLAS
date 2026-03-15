from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ILUKOptions:
    levels_of_fill: int = 0
    diagonal_shift: object = 1e-8
    drop_tolerance: float = 1e-4
    fill_factor: float = 10.0
    diag_pivot_threshold: float = 0.1
    modified_ilu: bool = True
    validate_batch_sparsity: bool = True


@dataclass(slots=True)
class SyevxOptions:
    algorithm: str = "chol2"
    ortho_iterations: int = 2
    iterations: int = 100
    extra_directions: int = 0
    find_largest: bool = True
    absolute_tolerance: object = None
    relative_tolerance: object = None
    store_every: int = 1
    store_current_residual: bool = False
    store_convergence_rate: bool = True
    store_ritz_values: bool = False


@dataclass(slots=True)
class LanczosOptions:
    ortho_algorithm: str = "cgs2"
    ortho_iterations: int = 2
    reorthogonalization_iterations: int = 2
    sort_enabled: bool = True
    sort_order: str = "ascending"


@dataclass(slots=True)
class SteqrOptions:
    block_size: int = 32
    max_sweeps: int = 10
    zero_threshold: object = None
    back_transform: bool = False
    block_rotations: bool = False
    sort: bool = True
    transpose_working_vectors: bool = True
    sort_order: str = "ascending"
    cta_wg_size_multiplier: int = 1
    cta_shift_strategy: str = "lapack"
    cta_update_scheme: str = "exp"


@dataclass(slots=True)
class SytrdBandReductionOptions:
    d_seq: list[int] = field(default_factory=lambda: [0])
    block_size_seq: list[int] = field(default_factory=lambda: [32])
    max_sweeps: int = -1
    max_steps: int = 1
    kd_work: int = 0


@dataclass(slots=True)
class StedcOptions:
    recursion_threshold: int = 0
    secular_solver: str = "rocm"
    leaf_steqr_params: SteqrOptions = field(default_factory=SteqrOptions)
    merge_variant: str = "auto"
    merge_threads: int = 128
    max_sec_iter: int = 50
    enable_rescale: bool = True
    secular_threads_per_root: int = 0
    secular_cta_wg_size_multiplier: int = 0


def options_to_dict(options):
    if options is None:
        return {}
    if isinstance(options, dict):
        raw = dict(options)
    else:
        raw = asdict(options)
    return {key: value for key, value in raw.items() if value is not None}
