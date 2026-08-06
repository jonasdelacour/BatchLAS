# Batched TRMM and SYRK kernels for the shapes we actually call them with

PR #61 declined to use `trmm` or `syrk` anywhere in `src/extensions`, and was
right to on the evidence it had: neither op reached a batched kernel at the
shapes its callers produce. This is the other half of that result -- writing the
kernels that were missing, and re-measuring.

Hardware: RTX 4090 / sm_89, CUDA 13.2, RelWithDebInfo, one GPU, no other compute
processes resident. All figures float, `--warmup=5 --min_iters=20`, run-to-run
stddev under 1% except where noted. Harness is `benchmarks/gemm_vs_level3_benchmark.cc`.
Every shape is paired with a batch large enough to saturate; batch = 1 is not
measured and is not a design target.

## What was missing

**SYRK.** `syrk_triangular_tiles` (PR #60) is a 128x128 tile-masked kernel and
its router requires at least three tiles a side, i.e. `n >= 257`. Below that
there is nothing for a triangular *grid* to skip, so the router correctly
refused it and every Gram matrix fell to `syrk_vendor_impl` -- a host loop
issuing one `cublasSsyrk` per batch member. At batch 2048 that is 33 ms against
a 0.33 ms GEMM.

**TRMM.** Nothing in the tree exploited triangularity. `src/extensions/trmm.cc`
recurses only until the block is 256 wide and then calls the GEMM it is meant to
replace, so for every `ib` in {16,32,64,128,256} the structure was never used.
On CUDA the op does not even reach that recursion: `trmm_vendor_impl` expands
the triangle into a `k x k x batch` scratch buffer and hands it to a full GEMM.

## SYRK: `syrk_gram_tiles.hh`

Sizes the tile to `n` rather than fixing it at 128, so one tile is the whole of
C. Both operands of `A^T A` are then the same columns of A, so there is one
shared tile instead of two and A crosses the bus exactly once -- which is what
matters, because arithmetic intensity here is `n/4` flop per byte, 8 at n = 32
against the 4090's ridge near 40.

The triangle is taken at *thread-tile* granularity: only the 136 of 256 thread
tiles that meet the requested half are carried, so 160 threads instead of 256.
Masking only the epilogue leaves the block doing exactly a GEMM's arithmetic,
which is why the first cut could match a GEMM at n = 128 and never beat one.

`gemm` and `syrk before` are one paired run on `main`; `syrk now` is this branch.

| m (reduction) | n | batch | gemm | syrk before | syrk now | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | **0.0780** | 4.29x | 431x |
| 512 | 32 | 2048 | 0.711 | 61.075 | **0.1711** | 4.16x | 357x |
| 512 | 64 | 1024 | 0.345 | 31.911 | **0.1875** | 1.84x | 170x |
| 1024 | 64 | 1024 | 0.668 | 60.397 | **0.3326** | 2.01x | 182x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | 0.99x | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | **0.3820** | 1.06x | 77x |

At n = 32 the kernel reads 71 MB in 78 us -- 933 GB/s, so it is at the memory
roofline and there is nothing further to get. At n = 128 it is at parity with
GEMM, and that is close to the ceiling too: reading A once costs 298 us against
GEMM's measured 409, so the whole prize there was 1.37x, not 2x.

n >= 256 is untouched and still on `syrk_triangular_tiles`: 0.89x at n = 256,
1.28x at 512, 1.57x at 1024. The n = 256 cell is a pre-existing loss.

## TRMM: `trmm_triangular_tiles.hh`

The structure worth exploiting is the k-loop bound, not a mask. An output tile
rooted at row `m0` only ever touches `op(A)_{i,p}` for `p >= i`, so it can start
its reduction at `m0` -- the arithmetic is not done and then discarded. `uplo`
and `trans` collapse into one flag, `lower_eff`, because transposing an upper
triangle gives a lower one.

How much that saves is worth stating precisely, because it is **not** the
textbook 2x: with `R` row tiles the reduction shrinks to `(R+1)/2R` of the
square -- 1.0x at R = 1, 1.33x at R = 2, only 1.78x by R = 8.

`before` is `BATCHLAS_TRMM_VARIANT=vendor` on this branch (the expansion plus
GEMM), measured in the same session.

| m | nC | batch | gemm | trmm before | trmm now | vs before | vs gemm |
|---|---|---|---|---|---|---|---|
| 32 | 256 | 2048 | 0.187 | 0.186 | **0.167** | 1.11x | 1.12x |
| 64 | 256 | 2048 | 0.352 | 0.399 | **0.333** | 1.20x | 1.06x |
| 64 | 512 | 1024 | 0.333 | 0.351 | **0.320** | 1.10x | 1.04x |
| 128 | 512 | 1024 | 0.674 | 0.784 | 0.699 | 1.12x | 0.97x |
| 128 | 1024 | 512 | 0.637 | 0.687 | 0.686 | 1.00x | 0.93x |
| 256 | 1024 | 256 | 0.758 | 0.852 | 0.915 | 0.93x | 0.83x |
| 256 | 256 | 512 | 0.467 | 0.690 | **0.536** | 1.29x | 0.87x |
| 512 | 512 | 128 | 0.762 | 0.961 | **0.713** | 1.35x | 1.07x |
| 512 | 1024 | 64 | 0.754 | 0.838 | **0.690** | 1.22x | 1.09x |
| 1024 | 1024 | 32 | 1.514 | 1.662 | **1.146** | 1.45x | 1.32x |

Faster than the route it replaces at 9 of 10 shapes. Against the *GEMM
spelling* it wins at `m <= 64` and `m >= 512` and loses in between, and the
reason is structural rather than a tuning miss:

- `m <= 64` is one row tile, so there is no k-range to skip and the arithmetic
  is a GEMM's. It wins anyway on fusion -- no `k x k x batch` expansion written
  and read back.
- `m >= 512` is four or more row tiles, so the k-range removes 1.6x-1.8x of the
  arithmetic, enough to cover this kernel running at ~70% of cuBLAS's per-flop
  rate.
- Between, one or two row tiles buys at most 1.33x, and that same 70% eats it.

### A routing mistake worth recording

The first version of the router gated the tile kernel to `m <= 64 || m >= 512`,
picked straight off the "vs gemm" column above. That is the wrong comparison:
the router chooses between the tile kernel and the *vendor*, not between trmm
and gemm. The gate sent `m = 128..256` back to the expansion and cost up to
1.29x on exactly the shapes it was meant to protect. There is now no threshold.

## Two measurement corrections to PR #61

1. **The TW benchmark charged trmm a zeroing pass it does not need.** The CUDA
   and ROCm paths overwrite C (`beta = 0`); only the MKL fallback in
   `src/extensions` accumulates. At ib = 32, nC = 256, batch 2048 that fill is
   ~70 us of a 189 us measurement. Removed.
2. **The TW shapes cannot show a trmm win and the roofline says so in advance.**
   With m in the tens, B and C dominate the traffic and the GEMM is already
   bandwidth bound near 800 GB/s, so halving its arithmetic buys nothing it can
   spend. `BM_Square_*` was added for shapes on the other side of the ridge.

## A bug this found, and the test that now covers it

The 128-wide SYRK tile initially split each thread's 8 rows into two 4-wide
bands 64 apart -- what the square 128x128 kernels do to spread shared-memory
banks. That is incompatible with taking the triangle at thread-tile
granularity, which decides a whole thread tile is inside the requested half
from its tile indices alone. Thread (0,1) then owns element (64,4), which is in
the lower triangle while its tile is not, so nothing wrote it.

It failed quietly and only at `n > 64`. `syrk_tests` pinned exactly one shape,
n = 96, which happened to catch it -- but that shape reaches only one of the
kernel's three tile widths. `SyrkTest.NarrowShapesMatchGemmReference` now sweeps
n in {24, 32, 48, 64, 96, 128} x trans x uplo with k = 200 (deliberately not a
multiple of the k chunk), so each tile width and each partial tile is exercised.

## What is not done

- **Neither op is wired into `ortho` yet, and that is deliberate.** `syrk` is
  now 1.8x-4.3x faster than the GEMM spelling at ortho's Gram shapes in
  **CUDA float**. In double and complex there is still no batched kernel --
  those fall to the per-batch host loop -- so switching the call sites today
  would trade a 2x float win for a ~100x double loss. The batched double/complex
  route is the prerequisite.
- `trmm`'s tile kernel is `Side::Left` only. `ormqr`/`ormbr` use both sides;
  the right-side branch still takes the expansion.
- The tile kernel runs at ~70% of cuBLAS's per-flop rate. Closing that would
  turn the `m = 128..256` band from a loss into a ~1.2x win.
- SYRK at n = 256 (0.89x, `syrk_triangular_tiles`) is a pre-existing loss and
  was not touched.
