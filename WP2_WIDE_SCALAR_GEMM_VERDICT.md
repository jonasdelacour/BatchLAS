# Final engineering verdict — wide-scalar GEMM tiles for BatchLAS

**Bottom line: land exactly one kernel — the 64×64×16 macro tile with a 4×4 thread tile (16 accumulators) — for `double`, `complex<float>` and `complex<double>`. Do not land anything for `float`. The value is a 7.5× / 3.6× win over BatchLAS's own `Tiled16` fallback for complex, not a win over NVIDIA.**

---

## 1. The numbers

RTX 4090 / sm_89, GPU 0, warm clocks (2565–2790 MHz), serialized. TFLOP/s = 2·mnk (real) / 8·mnk (complex). "Tiled16" is a faithful standalone replica of the kernel BatchLAS actually runs today for every non-float type (`src/sycl/gemm_kernels.cc:510-514`). Bold = per-cell best candidate.

| dtype | shape | β | cuBLAS | **Tiled16 (in-tree today)** | **64x64-k16-t4x4** | 128x128-t8x4 | 128x64-t8x4 | complex-split (best cfg) | winner vs cuBLAS | winner vs Tiled16 |
|---|---|---|---|---|---|---|---|---|---|---|
| double | 256³ b512 | 0 | 1.215 | 1.314 | 1.333 | 1.354 | 1.287 | **1.37** | 1.10–1.13× | 1.01–1.04× |
| double | 256³ b512 | 1 | 1.235 | 1.292 | 1.327 | 1.350 | 1.305 | **1.38** | 1.07–1.12× | 1.03–1.07× |
| double | 512³ b128 | 0 | 1.232 | 1.333 | **1.413** | 1.383 | 1.308 | 1.38 | 1.15× | 1.06× |
| double | 512³ b128 | 1 | 1.246 | 1.324 | 1.402 | 1.368 | 1.301 | **1.42** | 1.13× | 1.06× |
| double | 1024³ b32 | 0 | 1.247 | 1.310 | 1.415 | 1.386 | 1.320 | **1.42** | 1.14× | 1.08× |
| double | 1024³ b32 | 1 | 1.245 | 1.316 | **1.415** | 1.383 | 1.316 | 1.39 | 1.14× | 1.08× |
| cdouble | 256³ b512 | 0 | 1.242 | 0.389 | 1.387 | **1.391** | 1.374 | 1.37 | 1.12× | 3.57× |
| cdouble | 256³ b512 | 1 | 1.238 | 0.388 | 1.384 | **1.389** | 1.372 | 1.38 | 1.12× | 3.57× |
| cdouble | 512³ b128 | 0 | 1.244 | 0.387 | 1.391 | **1.396** | 1.382 | 1.38 | 1.12× | 3.59× |
| cdouble | 512³ b128 | 1 | 1.242 | 0.386 | 1.391 | **1.394** | 1.380 | 1.42 | 1.12× | 3.60× |
| cdouble | 1024³ b32 | 0 | 1.246 | 0.391 | 1.396 | **1.398** | 1.389 | 1.38 | 1.12× | 3.57× |
| cdouble | 1024³ b32 | 1 | 1.245 | 0.392 | 1.396 | **1.398** | 1.382 | 1.40 | 1.12× | 3.56× |
| cfloat | 256³ b512 | 0 | 48.91 | 6.72 | **49.50** | 48.37 | 43.20 | 48.00 | 1.01× | 7.37× |
| cfloat | 256³ b512 | 1 | 44.96 | 6.82 | **48.57** | 42.92 | 41.67 | 45.05 | 1.08× | 7.12× |
| cfloat | 512³ b128 | 0 | 47.37 | 6.66 | 47.85 | **51.19** | 45.73 | 49.36 | 1.08× | 7.69× |
| cfloat | 512³ b128 | 1 | 49.06 | 6.61 | **49.77** | 48.37 | 45.22 | 49.65 | 1.01× | 7.53× |
| cfloat | 1024³ b32 | 0 | 49.23 | 6.93 | 48.04 | **52.08** | 46.36 | 50.65 | 1.06× | 7.51× |
| cfloat | 1024³ b32 | 1 | 46.95 | 6.91 | 48.52 | **50.46** | 46.17 | 47.22 | 1.07× | 7.30× |
| float | 512³ b128 | 1 | 44.38 | 3.64 | 36.69 | 26.79 | 37.86 | 41.33 *(see note)* | **0.93×** | — |
| float | 1024³ b32 | 1 | 49.08 | 3.65 | 37.02 | 29.34 | 38.36 | 41.70 | **0.85×** | — |

Registers / spills (`-Xcuda-ptxas -v`, sm_89), all four candidates compile clean with the exact required command line:

| candidate | float | double | cfloat | cdouble | spills |
|---|---|---|---|---|---|
| 64x64-k16-t4x4 | 55/56 | 72/76 | 72/80 | 132/134 | 0 bytes, all 16 entries |
| 128x128-k8-t8x4 | 63/66 | 126/128 | 122/122 | 234/246 (tile forced down to 128×64) | 0 bytes, all 16 entries |
| 128x64-k8-t8x4 | 63/66 | 126/130 | 128/168 | 230/250 | 0 bytes, all 16 entries |
| complex-split | 117 | 72–210 | 72–247 | 126–255 | 1 of 48 config-pairs spills (cdouble 8×8, ~3.4 KB) |

**Data-quality marks — nothing here is filled with an estimate:**

- `64x64-k16-t4x4` bench rows print `maxrelerr=-1` (benchmarked with `--skip-check`); its correctness is in separate `--check-only` runs in `check.log`, at round-off, both paths, both betas. **Not a gap, but a different file.**
- cuBLAS **CGEMM** has ~±5 % run-to-run spread (5 repeats: 44.76–45.69, mean 45.35 at 512³ b128 β=1). Every `cfloat` "vs cuBLAS" ratio above is therefore ±5 %; the 5-run means give the winner 48.41 vs 45.35 = **1.068×**, which exceeds the combined spread. cuBLAS ZGEMM/DGEMM spread is 0.5–0.7 %.
- `complex-split` `cdouble 128x128x8/8x4` is **UNLAUNCHABLE** (208 regs × 512 threads = 106,496 > 65,536): it throws at launch. **No number exists and none is invented.** Its `cdouble 8×8` config spills 3.4 KB and still runs, 3.5 % slower than the best cdouble tile.
- The `float` row for `complex-split` was measured **twice with a 15 % discrepancy** (41.33 in `bench_complex_split.log`, 35.87 in `bench_gapfill_b.log`, same shape/β/binary). **Treat the whole float column as indicative only.** It does not change the verdict — every candidate loses to cuBLAS SGEMM for float on both readings.
- The correctness checks run on small fixed shapes (128×128×32 b3 aligned; 70×53×37 / 100×70×13 / 129×257×9 ragged), **not on the timed shapes.**
- Three of four harnesses use a default **out-of-order** queue with all timed iterations enqueued before one `wait()` — a formal RMW race at β≠0. It did not distort these numbers (2048+ workgroups vs 128 SMs; the one in-order harness and the serialized cuBLAS baseline agree within 2 %), but it is unsound at small batch and must be fixed before reuse.

---

## 2. Verdict per scalar type

### `complex<float>` → **land `64x64-k16-t4x4`.** 
It is at or above cuBLAS CGEMM in five of six cells (1.01–1.08×) and 2 % below in the sixth (1024³ β=0), and it is **7.0–7.7× the incumbent `Tiled16`** in every cell. `128x128-t8x4` is 4–8 % faster at the two largest shapes but **13 % slower at 256³ b512 β=1** (42.92 vs 48.57) — and 256³-at-large-batch is exactly the regime BatchLAS cares about. `complex-split` matches it but costs 247 registers and 1 block/SM to do so. Decided on the measurement, not the ratio argument: the tile-vs-occupancy scan (32:1 @ 8 warps, 21.3:1 @ 16 warps, 16:1 @ 24 warps) lands **within 5 % across the board**, so the FFMA:shared-load ratio is *not* the discriminator for complex — pick the cheap kernel.

### `complex<double>` → **land the same `64x64-k16-t4x4`.**
All four candidates are 1.11–1.12× cuBLAS ZGEMM and within 2 % of each other in all 24 cells, because FP64 issue rate dominates everything. `128x128-t8x4` wins by 0.1–0.4 % — noise. That does not buy a second kernel, a second tile shape, or the per-dtype macro-tile switch-down that `128x128-t8x4` needs to stay launchable for cdouble. The 64×64 kernel uses **one shape for all four scalars** and is the only candidate with no unlaunchable and no spilling configuration. Win over the incumbent: **3.56–3.60×**.

### `double` → **land it (it comes free with the above), but do not sell it as the win.**
1.07–1.15× over cuBLAS DGEMM, but only **1.01–1.08× over the `Tiled16` kernel BatchLAS already ships** — and just 1.4 % at 256³ b512 β=0. The reason is structural and worth writing into the header: FP64 on a 4090 is 1/64 of FP32, so the ceiling at the observed clocks is ~1.43–1.44 TFLOP/s. The candidate reaches 1.415 (**99 % of ceiling**), cuBLAS 1.25 (87 %), and the naive one-accumulator-per-thread `Tiled16` already reaches 1.33 (**92 %**). There was never 3× on the table for double and no tile design can find it. 4:1, 8:1 and 16:1 FFMA:shared ratios all land within 4 % for double — the shared pipe is over-provisioned by an order of magnitude for FP64 on a consumer part. **This conclusion is 4090-specific and will invert on a 1:2-FP64 datacenter part**, where `Tiled16` would not be near the ceiling and the ratio *would* bind.

### `float` → **none. Keep the in-tree `Tiled128x128RegisterK8`.**
Every candidate is 0.60–0.93× of cuBLAS SGEMM. Halving the thread tile to fit wide scalars costs float exactly what the 64-accumulator tile bought it. Nothing here should touch the float ladder.

**Two "established facts" in the brief are refuted by measurement and should be corrected in the ground truth:** "128 registers of accumulator alone… cannot fit, and it spills" is **false** — `double` at an 8×8 tile (128 accumulator registers) compiles to 208 total with **zero** spill, and `complex<float>` at 8×8 to 247 with **zero** spill. Only `complex<double>` at 8×8 (256 accumulator registers) genuinely spills, and even that costs 3.5 % here. The real wall is not spilling but *launchability*: registers × threads ≤ 65,536. The parity-target framing held exactly as stated (cuBLAS strict-FP32 SGEMM measures 44–52, nowhere near 80). No candidate has the scattered-epilogue defect: the worst β=0→β=1 degradation is 5.5 %, and at the one cell where candidates drop 16 % (float, 256³ b512) **cuBLAS drops 23 %** — that is C traffic, not addressing.

---

## 3. Integration plan

Source to port: `/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wide_scalar_gemm/tile-64x64-k16-t4x4.cpp`.

### 3.1 New header `src/sycl/gemm/register_64x64_k16_wide.hh`

Port `launch_wsg` verbatim into the repo's launcher shape. Five things in that file are load-bearing and must survive the port — each was found by reading PTX, and dropping any one of them reverts a measured property:

1. **The access granule is 16 BYTES, not 4 elements.** Do **not** reuse `Packet4<T>` from `register_128x128.hh`: it is `alignas(4*sizeof(T))`, i.e. 32 B for `double` and 64 B for `complex<double>` — load forms that do not exist in SASS. Use `Vec16<D>` with `N = 16/sizeof(D)` (4/2/2/1). This is what keeps an 8-lane LDS phase on exactly the 32 banks for every scalar width.
2. **`__attribute__((may_alias))` on the punning types**, or -O3 reorders shared stores against fragment loads across the barrier.
3. **The whole-granule staging copy must be a native LLVM vector type** (`typedef double Raw16Base __attribute__((ext_vector_type(2)))`), not a struct copy — SROA splits a `struct{T v[N];}` copy into element loads/stores and the 16-byte form is lost (measured: 4×`ld.global.b64` where 2×`ld.global.v2.b64` was intended).
4. **`std::complex` must never reach device code.** Re-type to POD `Cx<R>` at the pointer boundary and write the MAC as four `sycl::fma`s. Verified: zero `__mulsc3`/`__muldc3`, zero `call.uni`, in every instantiation.
5. **Shared strides exactly `TileM`/`TileN`, m fastest-varying in the epilogue.** Verified in PTX: zero scalar shared loads, all 768 are `v2.b64`/`v4.b32`; epilogue C traffic is vector `ld/st.global`.

Plus the fast-path predicate, mirroring `can_use_128x128_fast_path` but with the 16-byte granule:

```cpp
// Does this problem satisfy everything the unpredicated path assumes?
// Note the granule: 16 bytes, not 4 elements. VecLen is 4/2/2/1 for
// float/double/complex<float>/complex<double>, so the byte width of every
// vector access is pinned at 128 bits for every scalar.
template <typename T>
inline bool can_use_64x64_k16_wide_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatrixView<T, MatrixFormat::Dense>& C) {
    constexpr int TileM = 64, TileN = 64, TileK = 16;
    constexpr int VecLen = 16 / static_cast<int>(sizeof(T));
    const auto m = A.rows();
    const auto k = A.cols();
    const auto n = B.cols();
    if ((m % TileM) != 0 || (n % TileN) != 0 || (k % TileK) != 0) {
        return false;
    }
    auto aligned = [](const T* p, auto ld, auto stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0 &&
            (ld % VecLen) == 0 && (stride % VecLen) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(B.data_ptr(), B.ld(), B.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

template <typename T, bool AlignedFastPath = false>
Event launch_register_64x64_k16_wide(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     const MatrixView<T, MatrixFormat::Dense>& C,
                                     T alpha,
                                     T beta,
                                     const char* (*kernel_trace_name)(KernelVariant));
```

`alpha`/`beta` are `T` at this boundary and must be reinterpreted to `Cx<R>` before entering the kernel body, exactly as the operand pointers are.

### 3.2 Enum, names, force-by-name — `src/sycl/gemm_kernels.hh` and `.cc`

```cpp
// gemm_kernels.hh, after Tiled128x128RegisterK8:
    // The wide-scalar tile: 4x4 thread tile, 16 accumulators, and a 16-byte
    // (not 4-element) access granule, so double / complex<float> /
    // complex<double> all get vectorized conflict-free fragment loads. The
    // only register-tiled variant that serves a non-float scalar.
    Tiled64x64RegisterK16Wide,
```
```cpp
// gemm_kernels.cc, kernel_trace_name():
    case KernelVariant::Tiled64x64RegisterK16Wide:
        return "gemm_sycl_register_64x64_k16_wide";

// gemm_kernels.cc, kernel_variant_matches_name():
    case KernelVariant::Tiled64x64RegisterK16Wide:
        return name == "register64x64k16wide" || name == "reg64x64k16wide" ||
            name == "64x64x16wide";
```
and add `KernelVariant::Tiled64x64RegisterK16Wide` to the variant list enumerated in `forced_kernel_variant()` (the `for (auto variant : {...})` around `src/sycl/gemm_kernels.cc:314`). Do **not** add it to `is_experimental_kernel_variant` — it is measured, not experimental.

### 3.3 Routing predicate — `select_kernel_variant`, `src/sycl/gemm_kernels.cc`

Insert immediately before the existing `if constexpr (std::is_same_v<T, double>)` clause at the end of the function (by this point `transA == transB == NoTrans`, because the transposed case returned above):

```cpp
    // Wide scalars (double, complex<float>, complex<double>) have no register
    // kernel at all today: they fall off the float ladder above straight to
    // Tiled16 -- one accumulator per thread, std::complex operator* (and its
    // isnan branch plus __mulsc3 call) in the inner loop, and a scattered
    // epilogue. Measured on RTX 4090 / sm_89 at 256^3 b512, 512^3 b128 and
    // 1024^3 b32, at beta 0 and beta 1, against a standalone replica of
    // Tiled16 and against cuBLAS:
    //
    //   complex<float>  : 7.0-7.7x Tiled16, 0.98-1.08x cuBLAS CGEMM
    //   complex<double> : 3.56-3.60x Tiled16, 1.12x cuBLAS ZGEMM
    //   double          : 1.01-1.08x Tiled16, 1.07-1.15x cuBLAS DGEMM
    //
    // double is small on purpose: FP64 on a 4090 is 1/64 of FP32, the ceiling
    // is ~1.44 TFLOP/s, and Tiled16 already reaches 92% of it. Do not read the
    // double row as a win for the tile design; it is not, on this part.
    // See experiments/wide_scalar_gemm/measure/.
    //
    // Two gates, both deliberate and both conservative:
    //   * The unpredicated path only, exactly like the 128x128 float kernel
    //     above. The predicated path is correct (round-off on 70x53x37) but
    //     has never been timed against Tiled16.
    //   * min_dim >= 256, the smallest dimension in the measured grid. Smaller
    //     shapes are very likely wins too -- Tiled16 is 7x slower -- but they
    //     are unmeasured, and this file does not route on likelihood.
    if constexpr (!std::is_same_v<T, float>) {
        if (min_dim >= 256 && can_use_64x64_k16_wide_fast_path<T>(A, B, C)) {
            return KernelVariant::Tiled64x64RegisterK16Wide;
        }
    }
```

and the dispatch case in `gemm_custom`, alongside the `Tiled128x128RegisterK8` case:

```cpp
    case KernelVariant::Tiled64x64RegisterK16Wide:
        // NN only: the kernel reads A as m x k and B as k x n directly, so it
        // cannot serve a transposed operand. Unlike the 128x128 float kernel
        // every scalar is supported -- a 4x4 thread tile is 16 accumulators,
        // i.e. 32 registers for double and complex<float> and 64 for
        // complex<double>, measured at 72-134 total with zero spill bytes on
        // sm_89. The selector never picks it outside these bounds, but it can
        // be forced by name, so fall back rather than compute the wrong thing.
        if (transA == Transpose::NoTrans && transB == Transpose::NoTrans) {
            if (can_use_64x64_k16_wide_fast_path<T>(A, B, C)) {
                return launch_register_64x64_k16_wide<T, true>(
                    ctx, A, B, C, alpha, beta, kernel_trace_name);
            }
            return launch_register_64x64_k16_wide<T, false>(
                ctx, A, B, C, alpha, beta, kernel_trace_name);
        }
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
```

### 3.4 Which existing gate it sits behind

**`gemm_use_sycl_custom()` in `src/backends/gemm_variant.hh`** — the `BATCHLAS_GEMM_VARIANT` gate, which **defaults to `Vendor`**. With the change above and nothing else, default behaviour is byte-for-byte unchanged; the kernel is reachable only under `BATCHLAS_GEMM_VARIANT=sycl` (plus, for the double case only, under `auto`, which already lets square `double` problems with `batch >= 64` and `max_dim <= 512` through). **Land it this way first.** That is a self-contained, zero-risk PR whose entire user-visible effect is that the forced/opt-in SYCL path stops being 7× slower than the vendor for complex.

Realizing the win under `auto` is a **second, separately reviewable change**, and only after the follow-up in §5. Today `gemm_use_sycl_custom` refuses every complex problem outright:

```cpp
    // src/backends/gemm_variant.hh -- replace the blanket complex refusal.
    // It was correct while the only complex route was Tiled16, which measures
    // 3.6-7.7x slower than cuBLAS. The 64x64x16 wide-scalar register kernel
    // measures 0.98-1.12x of cuBLAS on the aligned NN square bucket, so let
    // exactly that bucket through and keep refusing everything else. The
    // enclosing function has already required m == n == k and batch >= 64.
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
            return false;
        }
        return max_dim >= 256 && max_dim <= 1024 &&
               (m % 64) == 0 && (n % 64) == 0 && (k % 16) == 0;
    }
```

### 3.5 Tests

Clone `BatchedGemmForcedSyclRegister128x128K8KernelAligned` and `…Ragged` in `tests/gemm_tests.cc:1964-2033` as `…Register64x64K16WideAligned` / `…Ragged`, forcing `BATCHLAS_GEMM_SYCL_KERNEL=64x64x16wide`, **and delete the float-only `GTEST_SKIP`** — this is the one variant where all four scalar types must run. Keep the aligned shape a multiple of 64 in m,n and 16 in k (e.g. 256), keep the ragged one ragged in all three (200×130×70 works: it exercises the predicated path and the partial final k-step), and keep `alpha=2, beta=-1` on the ragged case — a β=0 test structurally cannot see an epilogue defect.

---

## 4. What this does *not* buy the vendor-independence effort

Worth stating plainly even though the answer to "is anything worth landing" is yes. The measured margin over cuBLAS is 1.0–1.12× for complex and 1.07–1.15× for double, on **aligned, square, NN, large-batch** shapes only. Outside that envelope — ragged shapes, any transpose, small n — BatchLAS still falls to `Tiled16`, which is 3.6–7.7× off the vendor for complex. So: **complex GEMM is no longer vendor-dependent for performance in the aligned NN square bucket, and remains vendor-dependent everywhere else.** The correct-but-slower self-sufficiency fallback stays what it is today (`Tiled16`), and that is fine — but its complex path should get the two cheap fixes from this work regardless of tiles, because they are pure profit and independent of shape: **write the complex multiply out explicitly** (kills `__mulsc3`/`__muldc3` and the Annex-G isnan branch) and **make m the fastest-varying index in its epilogue**.

---

## 5. The single most valuable follow-up experiment

**Instrument BatchLAS's real complex and double GEMM call sites, then measure the new kernel's *predicated* and *transposed* coverage against `Tiled16` on that actual distribution.**

Concretely: run the complex-typed test and benchmark suites with `BATCHLAS_KERNEL_TRACE` on, histogram the `(m, n, k, batch, transA, transB, 16-byte-alignment)` tuples that reach `gemm`, then time the aligned-NN gate's hit rate and A/B the predicated path against `Tiled16` on the top shapes.

Why this and not a faster tile: every number in §1 is from a shape that satisfies `m%64 == n%64 == k%16 == 0`, NN, batch ≥ 32. The routing predicate in §3.3 fires only there. **A 7.5× win behind a gate that never fires is worth exactly zero**, and nothing measured so far tells us how often it fires inside BatchLAS — where GEMM is mostly called from panel updates and two-stage reductions with ragged trailing blocks and TN/NT operands. That one experiment decides three things at once: whether to widen the gate below 256, whether to invest in a transposed variant (currently the kernel is NN-only and falls back), and whether the predicated path is safe to route. Every other candidate follow-up (retuning the tile, chasing the last 4 % against cuBLAS CGEMM, re-testing on a 1:2-FP64 part) is worth less until the coverage question is answered.