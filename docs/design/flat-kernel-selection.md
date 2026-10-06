# Flat kernel selection

Status: **phases 1-5 implemented; tables: measured/converted/transcribed per `tuned/README.md`;
retune pending.** Phases 1-2 and P3.0-P3.4 (select infrastructure, posv, the tuner, blackwell
kernels, trsm, gemm) and phase 5 (gemv, geqrf, gesv, gesvd, getrf, getri, getrs, orgqr, ormqr, spmm
and syev, plus the rip of the old dispatch layer, the legacy env vocabulary and its aliases) are
built on `flat-select-mega` (2026-10-05, one PR); no `RouteTable` is left. Every op ships tables for
every dtype on sm_89 and sm_120. Open: the sm_89 live gate (§12 "Gate results") and phase 4, the
retune that replaces the transcribed tables with measured ones. As built: §12; decisions: §13.

Plan agreed 2026-10-02; deviations from the sketch are in §12. Written against `main` at `a1063892`. It is meant
to be executed from `main` in a fresh session, phase by phase. Nothing here depends on PRs #133,
#135 or #136, or on any branch other than `main`. The only exception is the potrf route-sweep
data, which this doc's PR copies onto `main` (`benchmarks/results/routing/`).

How to use this doc:
- Every decision below is settled. The reasoning is recorded so it is not re-argued.
- If something on `main` contradicts a fact quoted here, `main` wins. Fix the doc in the same PR.
- Phases are sequential. Each one ends with its own acceptance check (§10).

---

## 1. Why

Today a reader cannot tell which kernel a call will run without opening about ten files. Here is
the potrf path on `main`, one hop per line:

```
potrf(ctx, A, opts)                         include/batchlas/blas/options.hh:437
 → with_backend / BATCHLAS_DISPATCH_ON_QUEUE include/batchlas/blas/queue-dispatch.hh:32,269
 → potrf<B,T>(...)                          src/dispatch/entry_points/factorization.cc:657
 → backend::potrf_route                     src/backends/potrf_route.hh:66
   → potrf_op_shape (device facts)          src/backends/potrf_route.hh:20
   → dispatch::parse_route_env              include/batchlas/blas/dispatch/route_env.hh:163
   → resolve_potrf_route                    include/batchlas/blas/dispatch/route_potrf.hh:161
   → resolve_route (+ coverage)             include/batchlas/blas/dispatch/route_resolve.hh:105
   → resolve_route_uninstrumented           include/batchlas/blas/dispatch/route_resolve.hh:28
     → RouteTable::supports / preferred / native_tier_preferred / tiny_window / best_native_tier
                                            include/batchlas/blas/dispatch/route_potrf.hh:38-152
 → if-chain over Algorithm                  src/dispatch/entry_points/factorization.cc:673-714
 → potrf_lpanel_dispatch → launch params → parallel_for
                                            src/extensions/potrf_lpanel.cc:246,84,164
```

Workspace sizing (`potrf_buffer_size`, `factorization.cc:717`) does not follow this path. It
sizes for the maximum over every supported tier, so it can never agree exactly with what runs.
Blocked potrf then routes its trsm and gemm calls through two more RouteTables, and gemm adds a
third selection mechanism inside (`select_kernel_variant`, 43 `KernelVariant`s). None of those
inner choices shows up anywhere a reader would look.

Across the library the machinery is:
- 15 RouteTables;
- about 6,500 lines of core routing (dispatch headers, resolver, env parsing, coverage, per-op
  shape adapters, entry points);
- about 1,100 lines of level-3 selection that bypasses RouteTables;
- 1,500 lines of gemm variant selection.

Every speed threshold in it is a constant measured on one GPU (sm_89). None is keyed on the
device: `OpShape::cuda_cc` exists but is never assigned.

We looked at twelve current, performance-tuned GPU projects: llama.cpp, MLX, vLLM, FlashAttention,
PyTorch SDPA and TunableOp, oneDNN, DeepGEMM, Marlin, Triton, tinygrad and Modular. Every one of
them decides in **one place per operation**. The ones that tune per device keep the numbers as
**data per device** (vLLM's MoE configs, TunableOp, Triton's cache), with a default when there is
no data. The good ones print the choice.

## 2. Rules

These are the acceptance criteria for every op that migrates. A pull request that breaks one is
not finished.

| # | Rule |
|---|---|
| R1 | **Three hops, one file.** Public function → `choose()` → `std::visit` → launch, all in `src/ops/<op>/<op>.cc`. A reader needs no other file to know what runs. The only exceptions are the op's vocabulary header (`choice.hh`: the variant, the candidate list, the grid) and the shared `select.hh`. |
| R2 | **A choice is a value.** It is a `std::variant` of small family structs. One spelling (`lpanel:panel=8`) is used everywhere: in tables, pins, trace output, coverage rows and test names. |
| R3 | **`can_run` is correctness only.** If `can_run` is false, the launch would throw or give a wrong answer. A false `can_run` never means "slower". It must agree exactly with what the launch accepts (a test checks this, §8). |
| R4 | **Speed lives only in tables.** There are no hand-written speed thresholds in code. The last-resort list (§5.5) is ordered by generality, not speed. |
| R5 | **Sizing and running call the same `choose()`.** `<op>_buffer_size` returns what the chosen family needs and nothing more. A nested op adds its children's sizes by calling their public sizing functions. |
| R6 | **A bad pin is an error.** An unknown spelling, or a choice whose `can_run` is false for the shape, throws `std::invalid_argument` with the reason. Today an unrecognised value silently means Auto; that changes. |
| R7 | **Every candidate is reachable and tested.** Each entry in the candidate list can be pinned, and a test runs each one on shapes that straddle its limits. |
| R8 | **Unmeasured devices borrow, loudly.** A device with no table borrows another device's table, warns once per op, and tags every trace line with the table it used. |

## 3. Decisions

These were decided with the maintainer, round by round, on 2026-10-02. "Rejected" lists the
options that were considered and turned down.

| Topic | Decision | Rejected |
|---|---|---|
| Where per-device choices come from | Measured tables per (op, dtype, device), produced by a tuner | hand-written device-fact rules; first-call autotuning; fitted cost model |
| What may live outside the op file | the op file, its `choice.hh`, and one shared helper header (`select.hh`: pins, trace, tables) | fully self-contained files; a central policy file per device |
| Choice type | `std::variant` of family structs whose fields are ints | one type per compiled config; a flat struct with family + fields; an enum per kernel; virtual kernel objects |
| Nested ops | Each op decides for itself through the public function. A parent override is added only when a measurement shows a parent needs a different child (§5.7). | parent plans everything; hybrid from day one |
| Unmeasured shapes | The nearest measured point, by log distance over the op's keys | generated regions; power-of-two buckets |
| Storage | One text format, embedded at build time and parsed once with the same parser as pins. `BATCHLAS_TUNED_DIR` can override the tables. | compiled C++ tables; runtime files only |
| Untuned device | Borrow from the same vendor, nearest arch number at or below this one, else the nearest above. If nothing in that row can run, try the other tables in order. | per-op fact rules; a conservative vendor default |
| Row content | A ranked list of every measured candidate, with times | a single winner |
| CPU backend | Tuned like any device (`*.cpu.txt`) | borrowing |
| Borrow notice | Warn once per op, plus a trace tag | trace only; requiring an opt-in |
| Candidate list | A `constexpr` list in the op's `choice.hh`, which is also the compiled set | per-family ranges; a separate tuner config |
| Tuner grid | A declared coarse log grid, plus bisection where neighbouring points have different winners | a fixed grid only; recording real workloads |
| Tuner noise | Interleaved reps and medians. Within 3% of the best counts as a tie; ties go to candidate-list order. | keeping the previous winner unless 5% better; best-of-N |
| Staleness | Each table header carries a hash of the op's kernel sources. A mismatch is a CI and CMake warning; the table stays in use. | rejecting stale tables; a version stamp only |
| Pins | The env var, plus an RAII `ScopedPin` for tests | path-scoped pins; env only |
| Workspace | Sizing goes through the same `choose()` | an explicit plan object; the maximum over candidates |
| Tests | Every candidate pinned; every shipped table row validated | (not chosen: a golden routing snapshot; a launch-matches-choice check, see §11) |
| Pilot | potrf | gemm first; a small op first |
| First tables | Converted from the existing potrf route sweeps | fresh tuner runs (these come in phase 4) |
| Acceptance gate | Not more than 5% slower than today's routing at saturation, reproduced | the same choices as today; tests passing only |
| Old code | Each op's old routing is deleted in the PR that migrates it. The two systems never run side by side for one op. | — |

## 4. The design, end to end for potrf

### 4.1 Files

```
src/select/select.hh             shared: Device, pins, ScopedPin, trace, Table, parsing (§5)
src/select/select.cc
src/ops/potrf/choice.hh          vocabulary: families, PotrfChoice, candidates, grid, keys
src/ops/potrf/potrf.cc           choose(), can_run(), launch, potrf(), potrf_buffer_size()
tuned/potrf.float.sm_120.txt     data, one file per (op, dtype, device); plain git, NOT LFS
tuned/potrf.float.sm_89.txt
...
tools/tune/batchlas_tune.cc      the tuner (§6)
tools/tune/potrf_spec.cc         how to build and check a potrf problem for the tuner
scripts/sweep_to_table.py        converts the existing route sweeps into tables (§7)
```

`tuned/` sits at the repo root, outside `benchmarks/results/`, on purpose: tables must stay
readable text in diffs. Everything under `benchmarks/results/` is Git LFS.

The kernel sources (`src/extensions/potrf_{tiny,cta,lpanel,blocked}.cc` and their `_device.hh`)
are kept. They lose their routing role but keep their launch-parameter functions and their own
argument checks.

### 4.2 The vocabulary: `src/ops/potrf/choice.hh`

```cpp
#pragma once
#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::potrf {

// A family is one kernel driver. Its int fields are the knobs the selector chooses. Anything the
// driver derives from shape and device (Tiny's N bucket, CTA's scope and packing, Blocked's nb
// clamp) stays derived inside the launch code and is printed in the trace, never chosen.
struct Tiny {
  static constexpr std::string_view name = "tiny";
  static constexpr std::array<std::string_view, 0> fields{};
  std::array<int, 0> values() const { return {}; }
  static Tiny from(std::array<int, 0>) { return {}; }
  bool operator==(const Tiny&) const = default;
};
struct Cta     { /* same shape as Tiny, name = "cta" */ };
struct Lpanel {
  int panel = 8;  // NB template argument; 16 is instantiated for float only
  static constexpr std::string_view name = "lpanel";
  static constexpr std::array<std::string_view, 1> fields{"panel"};
  std::array<int, 1> values() const { return {panel}; }
  static Lpanel from(std::array<int, 1> v) { return {v[0]}; }
  bool operator==(const Lpanel&) const = default;
};
struct Blocked { /* name = "blocked", no fields in phase 2; see §11 */ };
struct Vendor  { /* name = "vendor" */ };

using PotrfChoice = std::variant<Tiny, Cta, Lpanel, Blocked, Vendor>;

// Every compiled choice, once. The tuner times exactly these, a pin must name one of them, the
// table test checks rows against them, and the launch visit must handle them. The order is the
// tie-break order (§6.3): simpler first.
template <class T>
constexpr auto candidates() {
  if constexpr (std::is_same_v<T, float>)
    return std::array<PotrfChoice, 6>{Tiny{}, Cta{}, Lpanel{8}, Lpanel{16}, Blocked{}, Vendor{}};
  else
    return std::array<PotrfChoice, 5>{Tiny{}, Cta{}, Lpanel{8}, Blocked{}, Vendor{}};
}

// Table keys: an exact-match key first, then log-distance keys (§5.4); n weighs 3 (§12).
struct Key { int uplo; int64_t n; int64_t batch; };
inline constexpr std::array<std::string_view, 3> key_names{"uplo:exact", "n:log:3", "batch:log"};

// The tuner's coarse grid (§6.2). This is today's sweep grid, so converted tables and tuned
// tables line up.
inline constexpr std::array<int, 34> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48,
    56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024, 1280};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::potrf
```

The zero-field families are verbose by design, so nothing is hidden. A helper such as
`struct Cta : NoFields<"cta"> {}` is allowed only if it stays under ten lines in `select.hh`.

### 4.3 The op file: `src/ops/potrf/potrf.cc`

This is the whole selection path. Kernel bodies are elsewhere; every decision is here.

```cpp
#include "ops/potrf/choice.hh"
#include "select/select.hh"
#include "extensions/potrf_native.hh"   // potrf_{tiny,cta,lpanel,blocked}_dispatch + their limits

namespace batchlas::ops::potrf {

template <class T>
Key key_of(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
  return {static_cast<int>(uplo), A.rows(), A.batch_size()};
}

// Correctness only (R3). Each line is a limit the driver itself enforces; the
// can_run-equals-launch test (§8) keeps them identical.
template <class T>
bool can_run(const PotrfChoice& c, const select::Device& d, const MatrixView<T, MatrixFormat::Dense>& A,
             Uplo uplo) {
  const int64_t n = A.rows();
  const bool native_ok = d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && n >= 1 && A.batch_size() >= 1;
  return std::visit(overloaded{
      [&](Tiny)          { return native_ok && n <= tiny_max_n<T>() && d.max_wg >= 64; },
      [&](Cta)           { return native_ok && n <= cta_max_n_for_slm<T>(d.slm_budget); },
      [&](const Lpanel& l) {
        return native_ok && uplo == Uplo::Lower && lpanel_nb_built<T>(l.panel) &&
               n <= lpanel_max_n_for_slm<T>(d.slm_budget, d.max_wg);
      },
      [&](Blocked)       { return native_ok && uplo == Uplo::Lower && cta_max_n_for_slm<T>(d.slm_budget) >= 1; },
      [&](Vendor)        { return d.has_vendor; },   // spec.vendor's library is compiled in
  }, c);
}

template <Backend B, class T>
void launch(Queue& q, const PotrfChoice& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
            Span<std::byte> ws, Span<int32_t> info) {
  std::visit(overloaded{
      [&](Tiny)            { sycl_potrf::potrf_tiny_dispatch<T>(q, A, uplo, ws, info); },
      [&](Cta)             { sycl_potrf::potrf_cta_dispatch<T>(q, A, uplo, ws, info); },
      [&](const Lpanel& l) { sycl_potrf::potrf_lpanel_dispatch<T>(q, A, uplo, ws, info, l.panel); },
      [&](Blocked) {
        sycl_potrf::potrf_blocked_dispatch<T>(q, A, uplo, ws, info,
            /*trailing_gemm=*/[&](auto&&... a) { gemm<B, T>(q, a...); },   // public gemm: decides for itself
            /*panel_solve=*/  [&](auto&&... a) { trsm<B, T>(q, a...); });  // public trsm: decides for itself
      },
      [&](Vendor) -> Event {
        if constexpr (select::has_library<B>(spec.vendor)) return backend::potrf_vendor<B, T>(q, A, uplo, ws, info);
        else select::no_vendor<B, T>(spec);   // NoRouteError + a coverage `miss` row
      },
  }, c);
}

template <class T>
size_t workspace(const PotrfChoice& c, Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);
// One visit, same five cases. Blocked adds gemm_buffer_size/trsm_buffer_size of its sub-shapes.

}  // namespace batchlas::ops::potrf

// ---- public entry points (signatures unchanged from include/batchlas/blas/functions/potrf.hh) ----
template <Backend B, typename T>
Event potrf(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<std::byte> ws,
            Span<int32_t> info) {
  namespace o = ops::potrf;
  potrf_validate_params(A, uplo);
  // device_of, choose (NoRouteError when vendor-free and nothing runs), the trace line and coverage
  // row (scalar and backend filled in; uplo is part of the key, never inferred), then launch inside it.
  return select::run<B, T>(o::spec, q, o::key_of(A, uplo), o::candidates<T>(),
      [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, A, uplo); },
      {.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo}, {},
      [&](const auto& c) { return o::launch<B, T>(q, c, A, uplo, ws, info); });
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
  potrf_validate_params(A, uplo);
  const auto c = select::pick<B, T>(ops::potrf::spec, q, ops::potrf::key_of(A, uplo), ops::potrf::candidates<T>(),
                                    [&](const auto& k, const auto& d) { return ops::potrf::can_run<T>(k, d, A, uplo); });
  return ops::potrf::workspace<T>(c, q, A, uplo);   // R5
}
```

The limit functions in this sketch are shorthand for the ones that exist today: `potrf_tiny_max_n<T>()`,
`potrf_cta_max_n_for_slm<T>(budget)` (min_blocks 4), `potrf_lpanel_max_n_for_slm<T>(budget, max_wg)` and
`potrf_blocked_available<T>()`. Each `can_run` line must equal the corresponding `supports()` clause in
`route_potrf.hh:38-72` **plus** that driver's own extra gates. Tiny's `MAX_WORK_GROUP_SIZE >= 64`
(`potrf_tiny.cc:261-266`) is one such gate; it is missing from today's `supports()`.

What disappears for potrf:
- `route_potrf.hh` (167 lines), with its windows, `tiny_window`, `best_native_tier` and
  `cta_last_order`;
- `src/backends/potrf_route.hh` (83 lines);
- the potrf branches of `factorization.cc:657-777`.

The public API is unchanged, and so are the `PotrfOptions` overloads in `options.hh`, which still
forward to the two functions above.

### 4.4 What a reader sees

```
$ BATCHLAS_SELECT_TRACE=1 ./app
potrf float n=64  batch=8192 -> lpanel:panel=8   0.302 ms, next vendor 0.490 ms          [sm_120]
potrf float n=512 batch=2048 -> blocked          14.10 ms, tied with vendor 13.74 ms (3%)  [sm_120]
  trsm  float ...            -> (old routing: Native:CTA)
  gemm  float ...            -> (old routing: Vendor)
potrf float n=64  batch=8192 -> lpanel:panel=8   [sm_89 table, borrowed for sm_86]
```

Until trsm and gemm migrate, their lines come from a small shim in the old resolver that prints
the resolved `Route` in the same indented format. That shim is about 10 lines and is deleted with
the old resolver.

## 5. `src/select/select.hh`: the shared helper

This is the only shared code. Target size is about 300 lines plus the parser, and every function
in it is generic over the choice variant.

### 5.1 Device facts

```cpp
struct Device {
  std::string key;          // table key: "sm_120", "sm_89", "gfx90a", "cpu", "intel_<id>"
  std::string family;       // "sm", "gfx", "intel", "cpu": borrowing stays inside a family first
  int arch_number;          // 120, 89, 90 (gfx90a), 0 for cpu
  bool is_gpu, has_sg32;
  bool has_vendor;          // the asking op's library group (OpSpec::vendor) is compiled in
  int64_t slm_budget;       // resident::device_slm_budget(LOCAL_MEM_SIZE), as today
  int max_wg;
};
template <Backend B> const Device& device_of(const Queue& q, Lib vendor = Lib::none);   // memoized
```

Facts are filled in from the same queries `potrf_op_shape` makes today (`potrf_route.hh:20-62`).
- **CUDA:** `key` comes from `Device::cuda_compute_capability()` (`queue-impl.cc:381`).
- **ROCm:** `key` comes from `gcnArchName`.
- **Host queues:** `key` is `"cpu"`.

### 5.2 Spelling and parsing (one spelling everywhere, R2)

- **Output, always:** `family[:field=value]...`. Examples: `tiny`, `lpanel:panel=8`, `vendor`.
- **Input:** the output form, or positional shorthand (`lpanel:8`).
- One function parses every source: `parse<Choice>(std::string_view) -> std::expected<Choice, std::string>`.
  Its callers are pins, table rows and `BATCHLAS_TUNED_DIR` files. If `std::expected` is not
  available in the toolchain, use `std::optional` plus an out-parameter for the error message.

### 5.3 Pins

```cpp
// BATCHLAS_<OP>_ROUTE is read on every call (factor_bench's ArmEnv sets it per arm with setenv).
// Values: auto | native | vendor | <spelling>.  "native" means: the best runnable non-Vendor
// candidate in the ranked row.  Anything else that does not parse throws (R6).
template <class Choice> class ScopedPin;   // tests: sets a thread-local slot that wins over the env
```

There are two compatibility notes:
- The env var name stays `BATCHLAS_POTRF_ROUTE`, so `factor_bench`, `run_factor_grid.sh` and
  benchviz keep working.
- The old spellings `native:tiny`, `native:cta`, `native:lpanel` and `native:blocked` are accepted
  as aliases for `tiny`, `cta`, `lpanel:panel=8` and `blocked` until phase 5 (removed there). The bare word
  `native` keeps its meaning.

A pin whose `can_run` is false throws. Today it silently falls back to Auto.

### 5.4 Tables and lookup

Text format (one file per op × dtype × device):

```
# op=potrf dtype=float device=sm_120 batchlas=a1063892 kernels=unknown date=2026-10-02
# source=benchmarks/results/routing/sm120_potrf_sweep.jsonl (converted, passes 1+2)
# keys: uplo:exact n:log:3 batch:log
uplo=L n=64  batch=8192  | lpanel:panel=8 0.302 | vendor 0.490 | cta 0.530 | blocked 0.536
uplo=L n=128 batch=8192  | lpanel:panel=8 1.358 | vendor 1.764 | blocked 2.512
uplo=L n=512 batch=2048  | blocked 14.10 | vendor 13.74 | lpanel:panel=8 14.84   # tie: within 3%, list order
uplo=U n=64  batch=8192  | cta 0.543 | vendor 0.654
```

Times are milliseconds per call for the whole batch, which matches `time_ms` in the sweep data.
The `| choice time` pairs are in rank order.

A transcribed table (§13) writes `-` for every time: `uplo=L n=8 nrhs=1 batch=128 | tiny - | cta -`.
Each row is all-timed or all-untimed, and the loader rejects a mixed row at its line. An untimed
row needs a `source=transcribed:<sha>` header, which is rejected at its line unless `<sha>` is
non-empty hex (`scripts/sweep_to_table.py --transcribe` resolves `--sha` with `git rev-parse`). A
table may hold both row kinds; `sweep_to_table.py --check` applies the same three rules. Untimed rows are walked in rank order like timed ones, are exempt from the
§6.3 tie rule, and trace as `transcribed` where a timed row prints its times.

Embedding:
- CMake turns each `tuned/*.txt` into a `constexpr std::string_view` in one generated `.cc`.
  C++23 `#embed` is not available in DPC++ yet, so this is a short CMake `file(READ)` script.
- Tables are parsed once, on the first call of each op.
- `BATCHLAS_TUNED_DIR=<dir>`: any file there with the same name replaces the built-in one, and
  the trace tag then says `[override sm_120]`.

`select::choose` is the only selection algorithm in the library:

```cpp
template <class Choice, class Key, size_t N, class CanRun>
Choice choose(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
              const std::array<Choice, N>& candidates, CanRun can_run) {
  if (auto pin = read_pin<Choice>(op))                       // ScopedPin, else BATCHLAS_<OP>_ROUTE
    return resolve_pin(*pin, candidates, can_run, op);       // throws on unknown or !can_run (R6)
  for (const Table<Choice>* t : tables_in_borrow_order(op, dtype, d)) {   // own table first (§5.5)
    if (const Row<Choice>* row = t->nearest(key))            // longest matching exact prefix; log keys: min distance
      for (const auto& [c, ms] : row->ranked)
        if (can_run(c)) return note(t, d, c);                // note(): warn-once if borrowed, trace tag
  }
  for (const Choice& c : last_resort(candidates))            // generality order (§5.5)
    if (can_run(c)) return c;
  throw std::runtime_error(std::string(op) + ": no runnable kernel on " + d.key);
}
```

- **Nearest:** the `:exact` keys form a prefix in `# keys:` order. The rows that match all of them
  are kept; if none does, the last exact key is dropped and the shorter prefix is tried, down to
  no exact key at all. With `side:exact trans:exact`, a missing (R,T) combination keeps the
  `side=R` rows rather than jumping to the nearest row of any side. Among the kept rows, the row
  with the smallest `Σ w·|log2(row_key / key)|` over the `:log` keys wins, where `<name>:log:<w>`
  sets the weight `w` (1 when omitted; see §12). On a tie (within 1e-9), the row with the smaller
  log keys wins, compared in `# keys:` order (`n` first). With potrf's single exact key this is
  the all-or-nothing drop: Upper calls use a Lower-only table, with `can_run` removing the
  Lower-only choices.

### 5.5 Borrow order and last resort

`tables_in_borrow_order(op, dtype, d)`:
1. the table whose device equals `d.key`;
2. the other tables in the same `family`, nearest `arch_number` at or below `d.arch_number`
   first, then the nearest above;
3. every other family's tables, in a fixed order: `sm`, `gfx`, `intel`, then `cpu` last.

The first table taken from step 2 or 3 prints one warning per op, per process:

```
batchlas: potrf has no float table for sm_86; borrowing sm_89 (run tools/tune to tune this device)
```

`last_resort(candidates)` is the candidate list reordered by generality: the family that runs on
the most shapes first. For potrf that is `blocked`, then `vendor`, then the rest in list order.
It is reached only when no row in any table has a runnable candidate.

### 5.6 Trace and coverage

`TraceScope` does three things:
- when `BATCHLAS_SELECT_TRACE=1`, it prints one line per call to stderr, as in §4.4;
- it keeps a thread-local depth counter so that children are indented;
- it records the coverage row.

The coverage row keeps the existing format (`src/dispatch/coverage.cc:102-140`): a `reached`
row, with the choice spelling in the `chosen_algo` column and `native`/`vendor` in
`chosen_origin`. This matters because `factor_bench`, `scripts/route_diff.sh`,
`run_factor_grid.sh` and benchviz verify pins by reading these rows ("a pin is never taken on
trust"). `BATCHLAS_KERNEL_TRACE` is unrelated and stays as it is.

### 5.7 Parent override (not built in phase 2)

Build this only when a measurement shows a parent needs a different child than the child would
pick for itself:
- an internal overload `gemm(q, args, std::optional<GemmChoice> forced)`;
- an optional child field on the parent's family (e.g. `Blocked::update`);
- the trace line says `(forced by potrf)`.

## 6. The tuner: `tools/tune/batchlas_tune`

The tuner produces every number the library runs on, so it follows the measurement rules in
AGENTS.md §10 without exception.

```
batchlas_tune potrf --dtype float,double,cfloat,cdouble --devices 1 --out tuned/ \
              --raw benchmarks/results/tuning/
```

### 6.1 What it times

For each dtype it times every entry of `candidates<T>()`, pinned with `ScopedPin`, through the
public `potrf()`.
- **Inputs:** each cell gets SPD inputs built by `tools/tune/potrf_spec.cc`.
- **Verification:** each run is host-verified on items 0 and batch-1, the same residual check
  `factor_bench` uses.
- **Skips:** a candidate whose `can_run` is false at a cell is recorded as skipped, not timed.

### 6.2 Grid and edge refinement

1. Measure the coarse grid `grid_n × grid_batch` for each `uplo` the op supports. Cap the
   matrices at 4 GiB, as the existing sweeps do.
2. For each batch, wherever neighbouring `n` points have different winners, bisect `n` between
   them, measuring each midpoint, until the bracket is under 10% (`hi/lo < 1.1`).
3. Do not refine `batch`.
4. Refined points become ordinary rows.

### 6.3 Noise protocol

1. Run a throwaway JIT pass over every cell first.
2. At each cell, interleave the candidates and warm up for 1.5 s per candidate.
3. Take 16 timed reps, rotating the candidate order every rep. Report the median.
4. Make two passes over the whole grid, the second in reverse candidate order. A cell's time is
   the mean of its two pass medians.
5. If the two medians differ by more than 10%, re-measure that cell once.
6. **Tie rule:** every candidate within 3% of the best ranks as tied, and tied candidates are
   ordered by their position in `candidates<T>()`. The same data therefore always gives the same
   table.
   **Consequence:** `candidates<T>()` lists the native tiers before `vendor`, so near-ties go to
   the native kernel. For example, float n=512 at batch 2048 on sm_120 has blocked at 14.10 ms and
   vendor at 13.74 ms: a tie, so blocked is chosen. That is deliberate (vendor independence), and
   it costs at most 3% at such a cell. Moving `Vendor{}` to the front of the list reverses it.
7. Run one measuring process on the box at a time. Use `benchmarks/gpu_guard.sh` plus a `flock`,
   and use a headless device (device 1 on the 4090 box).

### 6.4 Output

- `tuned/<op>.<dtype>.<device>.txt` in the §5.4 format, with `kernels=<hash>`. The hash is a
  SHA-256, truncated to 8 hex digits, of the op's kernel sources, listed in the op's
  `tools/tune/<op>_spec.cc`.
- The raw per-rep timings go to `benchmarks/results/tuning/<op>.<dtype>.<device>.jsonl` (LFS) as
  provenance.

### 6.5 Staleness check

A CMake configure step and a CI script (`.github/ci/check_tuned_tables.py`) recompute each op's
kernel hash. A table whose header hash differs, or says `unknown`, gives:

```
warning: tuned/potrf.float.sm_120.txt is stale (kernels unknown -> 3f9a1c2e); run tools/tune
```

It is a warning, never an error.

## 7. Seed tables from the existing sweeps: `scripts/sweep_to_table.py`

The first tables are converted from the data this PR brings to `main`. See
`benchmarks/results/routing/README.md` for the full provenance.

| File | Device | Content |
|---|---|---|
| `sm120_potrf_sweep.jsonl` | RTX PRO 6000 Blackwell (sm_120), build `a1063892` | Lower; 4 dtypes; n 2–1024 × batch 128–32768 (541 cells × 5 arms × 2 passes); plus Upper float, n 8–256, batch 8192 |
| `sm120_potrf_sweep_edges.jsonl` | same | n = 1, 1280, 1536 edges |
| `sm89_potrf_archive.jsonl` | RTX 4090 (sm_89) | an archive across several kernel eras; field `kernel_current` |

Conversion rules:
1. `op == "potrf"` rows are `uplo=L`; `op == "potrf_upper"` rows are `uplo=U`. Keep rows with `ok == true` whose reached `route` equals the pinned arm. Rows with
   `ok == false` record `supports()`, not timings.
   posv (phase 3) reads its reached route from the sweep's `reached` field and its `uplo` as
   `Lower`/`Upper`. Its driver's `ok` also fails `rel_sd > 0.10`, which hits launch-bound n=1
   cells; a row whose `reason` is exactly `relsd` (`info_nonzero == 0`, finite `residual`, reached
   = pinned) is a measurement, so it is kept and its row marked `# noisy` as in rule 3. Fallback,
   unsupported, residual, info and error rows still drop. `--self-test` (also run by `--check`)
   checks these rules on rows copied from the live sweeps, `tests/data/sweep_to_table_rows.jsonl`.
2. For sm_89, keep only `kernel_current == true`. Cells left with fewer than two current arms are
   dropped, and the script prints how many were dropped.
3. A cell's time per arm is the mean of the pass medians (sm_120 has passes 1 and 2). If the
   passes differ by more than 10%, the row gets a trailing `# noisy` comment.
4. Map arms to spellings: `route:native:tiny` → `tiny`, `route:native:cta` → `cta`,
   `route:native:lpanel` → `lpanel:panel=8` (the sweep ran the default NB), `route:native:blocked`
   → `blocked`, `vendor` → `vendor`.
5. Apply the §6.3 tie rule, then write the §5.4 format with `kernels=unknown`. Every converted
   table therefore starts out flagged stale, as intended.
6. `cpu` tables:
   - On `main` every native potrf tier requires `is_gpu && has_sg32`, so on a host queue only
     `vendor` (netlib) can run.
   - The `cpu` table is therefore a one-candidate table. It is generated by the tuner in phase 4,
     and the converter does not produce it.
   - Until then, the CPU falls to the last resort, which picks `vendor`, the same as today.

## 8. Tests

**Added for potrf in phase 2:**

1. **Every candidate, pinned** (`tests/potrf_candidates_tests.cc`). This test:
   - loops over `candidates<T>()` for all four dtypes;
   - pins each candidate with `ScopedPin`;
   - runs it on shapes that straddle its `can_run` limits in both directions;
   - checks residual, info and the untouched triangle.

   It follows the AGENTS.md §8 rules: non-natural `ld` and stride, complex data with nonzero
   imaginary parts, a saturating batch (≥ 1024 identical items, bit-identical results) for the SLM
   tiers, and poison the kernel will accept. The existing tier suites in `potrf_tests.cc`,
   `potrf_tiny_cases.inc` and `potrf_lpanel_cases.inc` already do most of this through
   `BATCHLAS_POTRF_ROUTE`. Port them to `ScopedPin` rather than writing new ones.
2. **`can_run` equals launch.** For every candidate and every shape in the straddling sets:
   - when `can_run` is true, the direct `potrf_*_dispatch` call does not throw;
   - when it is false, the direct call refuses.

   This keeps R3 honest, and it replaces today's `DirectEntryPointRefusesWhatSupportsRefuses`
   tests.
3. **Sizing.** For every candidate under a pin, run with a workspace of exactly
   `potrf_buffer_size(...)` bytes inside a poisoned arena, and check that nothing is written past
   the end. This replaces `BufferSizeCoversEverySupportedNativeTier`, which pins today's
   over-cover.
4. **Table validity** (`tests/tuned_tables_tests.cc`, no GPU needed). Every embedded table:
   - parses;
   - names only entries of `candidates<T>()`;
   - has strictly ascending ranked times, apart from ties.

   On the matching device, each row's first runnable candidate must also pass `can_run` at that
   row's own key.
5. **Pins and borrowing** (no GPU needed). This test checks that:
   - an unknown pin throws, and so does a `can_run`-false pin;
   - ~~the legacy aliases parse~~ (phase 5 removed every alias; each op's should-throw test now
     lists the old spellings);
   - with synthetic tables, the borrow order of §5.5 holds, and the warning is printed once.

**Deleted in the same PR**, because they assert the old predicates:
- the `RoutePotrf.*` suite in `tests/route_vocabulary_tests.cc` (from `:492`);
- `potrf_tests.cc:1419-1541`;
- `potrf_tiny_cases.inc:531-610`;
- `potrf_lpanel_cases.inc:460-562`;
- the `potrf_route` readback in `posv_tests.cc:468-492`.

The `route-native` ctest re-run of `potrf_tests` (`tests/CMakeLists.txt:372-375`) stays, because
`native` keeps its meaning.

## 9. Phases

| Phase | Work | Ends with |
|---|---|---|
| 0 | This doc, plus the potrf sweep data copied to `main` | PR merged |
| 1 | `src/select/` (Device, parsing, pins, `ScopedPin`, Table, `choose`, borrow order, `TraceScope` + coverage row), the CMake embed step, the table-validity and pin/borrow tests on synthetic tables, `scripts/sweep_to_table.py` | `tests/select_tests` green; converter produces `tuned/potrf.*.{sm_120,sm_89}.txt` from the routing data |
| 2 | potrf migrated as in §4; old potrf routing and its tests deleted (§8); `docs/perf/potrf.md` and the AGENTS.md routing section updated | §10 gate passes on sm_120 and sm_89 |
| 3 | posv (it calls potrf), then trsm and gemm (Blocked's children), each as its own PR with the same gate. gemm folds `select_kernel_variant`'s 43 variants into families with fields (e.g. `tiled:tile=64:k=8`) | each op's gate |
| 4 | Build `tools/tune`; retune potrf on sm_120, sm_89 and cpu, replacing the converted tables (hashes stamped) | tables no longer stale; gate re-run |
| 5 | The remaining ops, one PR each; then delete `include/batchlas/blas/dispatch/`, the `src/backends/*_route.hh` adapters, `route_vocabulary_tests.cc`, the legacy env aliases and the route vocabulary docs | no `RouteTable` left |

Phases 3–5 are planned but not committed to. The maintainer decides after phase 2. (Done: phase 3
as P3.0-P3.4; phase 4's tuner as P3.2; phase 5 as one mega PR, §12. Phase 4's retunes are open.)

## 10. Acceptance gate (for every migrated op)

1. **Correctness:**
   - `ctest -R '^potrf_tests$|^potrf_candidates_tests$|^tuned_tables_tests$|^select_tests$|^posv_tests$'`
     is green;
   - a vendor-free build (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) runs the same tests;
   - failures are compared by name against `tests/known-failures.txt`.
2. **No routing change by accident.** On the sweep grid, the new choice must be the table's
   first runnable entry. This is a pure-data check by the converter's `--check` mode.
3. **Not slower than today, live, at saturation (batch ≥ 128)**, on each tuned GPU:
   1. Build two binaries, `main` (old routing) and the branch.
   2. On the off-grid points between grid `n` values (36→44, 64→72, 96→104, 128→144, 192→208,
      256→272, 320→352, 448→480, 640→704, 1024→1152), for each dtype at batch 512, 8192 and 32768,
      time today's Auto choice and the new choice interleaved in one `factor_bench` process with
      `--arms`.
   3. **FAIL** if the new time is more than 1.05× the old time, and the loss reproduces in a
      second pass with the arm order reversed.
   4. Off-grid points are where nearest-point lookup guesses. On-grid points equal the measured
      winner by construction, so they prove nothing.
4. **Readable.** A reviewer who has not seen the code finds the kernel for `potrf float n=100
   batch=8192` on sm_120 by reading `potrf.cc` and the table file. No other file is needed.

The sm_89 gate needs the RTX 4090 box. The sm_120 gate needs the Blackwell box (threadripper02).

## 11. Known gaps and follow-ups

- **Blocked's inner leaf factorization stays hard-wired** to `potrf_cta_dispatch`
  (`potrf_blocked.cc:329`), and so does its fold kernel. These are steps inside one kernel
  family, not selection. Making the leaf call the public `potrf` would allow recursion and
  re-tuning, so it is a separate change, made only after measuring.
- **`Blocked` has no fields in phase 2.** Its `nb` and `W` still come from
  `PotrfBlockedConst`, overridable by `BATCHLAS_POTRF_NB`/`BATCHLAS_POTRF_W`. Making them fields
  (`blocked:nb=128:w=32`) and deleting those env vars belongs in phase 4, when the tuner can sweep
  them.
- **`Lpanel{16}` (float) has never been timed.** It was unreachable (`nb_hint` was always 0), so
  converted tables do not contain it. It becomes eligible only after the phase-4 retune, and its
  pinned test (§8.1) is new coverage.
- **Declined tests and their cost.** No golden routing snapshot and no launch-matches-choice
  check were chosen. Without the second, a `choose()` that says one thing while the launch does
  another is caught only by reading the code. The R1 structure makes that a 3-hop read, but it is
  not automated.
- **The CPU backend runs no native potrf tier on `main`.** Tuning it is still worth doing for
  ops whose native kernels do run on host queues. It does not add native potrf coverage to CPU
  CI.
- ~~**`posv` calls the potrf tier drivers directly**~~ (struck, P3.1): it never did. posv calls the
  public `potrf` and `trsm`; its only direct driver calls are its own kernels. It is migrated in
  `src/ops/posv/` (§13).
- **gemm's `direct` kernel indexes the batch with `int` offsets** (`b * stride`), which can
  overflow at large batch x stride, as trsm's complex substitute did (K4). It predates P3.4 and is
  not a `can_run` term; fix it with 64-bit offsets rather than a capacity gate.
- **gemm's sm_120 table** is open after P3.4 (§12 Phase 3.4).
- **Other ops may share gemm's grid-z ceiling**: any kernel that puts the batch in SYCL dim 0 of a
  3-D range aborts past 65535 work-groups there. gemm's `can_run` now carries it (§12 Phase 3.4);
  the migrated ops' kernels were not audited for it. ormqr blocked hits it (known-defects #16).
- **Stale comments on `main`** to fix when touching these files:
  - ~~`potrf_native.hh:4`~~, ~~`coverage.cc:195`~~, ~~`factorization.cc:727-730`~~ (rewritten or
    deleted by phase 5);
  - `potrf.hh:51-53` (wrong `options.hh` line numbers).

## 12. As built (phases 1-5)

Where the code differs from the sketches above, the code wins. These are the differences.

**Phase 1, `src/select/`:**
- **The class words `native` and `vendor` fall back to Auto with a warning** (once per op and
  word) when nothing in their class can run the shape. They do not throw. These are the two
  exceptions to R6. Bare `native` keeps today's meaning, which the `route-native` re-run of
  `potrf_tests` relies on. Bare `vendor` joined it after the final review: in a vendor-free build
  a `vendor` pin threw, so `factor_bench`'s default `vendor,native` arms aborted the process,
  where the old router had fallen through to automatic. Concrete spellings and aliases still throw
  `std::invalid_argument` (an alias that maps to `vendor` is concrete). Phase 5 deleted the aliases.
- **A CPU device never borrows a GPU table.** Without its own `*.cpu.txt` it goes straight to the
  last resort. A GPU device can still borrow across families (§5.5 step 3).
- **Tables store spellings, validated on first use.** A table is parsed into spellings without
  knowing the op's `Choice` type; each entry is checked against `candidates<T>()` the first time
  `choose()` reads that table. `tuned_tables_tests` does the same check for every shipped table
  without a GPU.
- **`Key` is a list of name/value pairs** (`select::Key = std::vector<KeyField>`), not a
  per-op struct. Field names must match the table's `# keys:` line; a missing one throws.
- **`TraceScope` takes an `OpShape`** (the coverage key), built with `square_shape<B, T>(n, batch)`
  plus the fields the op sets itself (`uplo`).
- **`choose()` takes a `Rules{aliases, last_resort}`** from the op. The aliases and the
  last-resort order are op data, not helper code. Phase 5: `Rules{last_resort}`, no aliases.
- Zero-field families use `NoFields<"name">`, as §4.2 allows.

**Phase 2, potrf:**
- No `struct Key` in `choice.hh`; `key_of` in `potrf.cc` builds a `select::Key`. `key_names`
  stays in `choice.hh`.
- `aliases`, `last_resort` and `rules` live in `choice.hh`, so tests and `factor_bench` use the
  library's list (`aliases` deleted in phase 5, with the bare-`lpanel` alias below).
- **One alias beyond §5.3: bare `lpanel` → `lpanel:panel=8`.** In the old vocabulary a bare
  algorithm word meant native; `tiny`, `cta` and `blocked` already parse as themselves.
  potrf never had a `_VARIANT`/`_PROVIDER` variable, so no other legacy spelling exists.
- Tiny's work-group gate uses a new `kPotrfTinyWgSize` in `potrf_native.hh`, because the real
  `kTinyWgSize` lives in a header that pulls in `<sycl/sycl.hpp>`. A `static_assert` in
  `potrf_tiny.cc` keeps them equal.
- `Lpanel`'s `can_run` calls `potrf_lpanel_max_n_for_slm<T>(budget, max_wg, kMinBlocksPerSm,
  panel)`, which returns 0 for a panel width the type does not instantiate. That one call covers
  the width check (no separate `lpanel_nb_built`) and the SLM/work-group limits for that width.
- CTA's `can_run` is exactly the old `supports()` clause. The driver's `wg_size <= max_wg` check
  is left out: it can only fail when `max_wg < 32`, which the sub-group-32 gate already excludes.
- **Blocked's workspace is `potrf_blocked_buffer_size` alone.** gemm and trsm take no workspace,
  so the "adds gemm/trsm buffer sizes" line in §4.3 does not apply.
- The `posv_tests` readback (§8) is replaced, not deleted: `FusedSolveArmSolvesOnBothTriangles`
  must know whether a native potrf ran, so a helper pins each non-vendor candidate with
  `ScopedPin` and asks `potrf_buffer_size` whether it accepts the pin.
- Three test edits outside §8's deletion list: `JustPastTheCeilingHasNoCtaRoute` now checks a
  `cta` pin is accepted at the ceiling and throws one past it; the "NOT the guard" `potrf_route`
  lines in the CTA, Blocked, Tiny and LPanel facade tests are gone (the bit-exact guards stay);
  `RouteVocabulary.AlgorithmEnumeratorValuesAreAbi` is kept and the Tiny round trip moved to
  `RouteVocabulary.TinyVocabularyRoundTrip`.
- **Tests (§8):** `tests/potrf_candidates_tests.cc` (label `blas`) holds §8.1-8.3 and §8.5:
  pinned candidates straddling their ceilings, pinned-equals-direct-driver bit for bit, a
  saturating batch of 1024, `CanRunEqualsLaunch`, the exact workspace in a poisoned arena, and the
  pin tests (unknown/uncompiled/can_run-false pins throw, aliases, bare `native`, `ScopedPin` over
  the environment). The §8.4 device half is `ShippedRowsRunOnTheirOwnDevice` (every entry of every
  row of the own-device table is accepted as a pin at that row's key) and
  `AutoReadsEveryKeyField` (hand-read Auto rows straddling `batch` per dtype and `uplo` for
  float, on sm_120 and sm_89). The tier suites pin with `ScopedPin`; the potrf direct-refusal tests
  are gone. Deliberate breaks B1-B9 (tests report) each gave a narrow red set.
- **Vendor-free misses:** when no candidate can run and the vendor is compiled out, `choose()`
  funnels into `throw_no_vendor_route`, so the `NoRouteError` and the coverage `miss` row survive.
- **Not done in phase 2:** the `potrf.hh:51-53` stale line numbers (§11) are not fixed. The sm_89
  half of the §10 gate is not run (see "Gate results").

**After the final review:**
- **Weighted `:log` keys.** A key spec may give a `:log` key a weight, `<name>:log:<w>` (a positive
  integer or decimal, default 1), and `nearest()` minimises `Σ w·|log2(row/key)|`; the tie rule is
  unchanged. potrf declares `uplo:exact n:log:3 batch:log`: its work grows as `n^3` and linearly in
  `batch`, so the distance approximates a log-cost distance. The trigger was the sparse sm_89
  tables: with equal weights float L n=24 batch=512 mapped to the n=80 batch=2048 row (`vendor`,
  where the native tiers are about 2.2x faster), double n=3 batch=128 to the n=256 row, and
  vendor-free sm_89 ran `blocked` on 3x3 matrices. With `n:log:3` the first two land on their
  own-n rows (`cta` and `tiny`). `scripts/sweep_to_table.py` writes the weighted `# keys:` line and
  its `nearest()` reads the weights from it; only the `# keys:` lines of `tuned/*.txt` changed.
  Over a 46-n x 5-batch grid of `uplo=L` points, 279 of 1840 table lookups changed their first
  entry (80 on sm_120, all at n >= 256 with batch >= 8192, which used to fall back to an n <= 320
  row, and now take `vendor` or `blocked` from an own-n row; 199 on sm_89: 84 at n <= 32, 82 of
  them `vendor` -> a native tier, 60 large-n `vendor` -> `blocked`, 12 `vendor` -> `cta`, and 43
  mid-n batch=32768 cells that used to borrow an n=32 row and now take a nearer-n row: 41 `vendor`,
  2 `cta`).
  Before/after lists: `--check --check-points` over that grid.
- **Coverage native flags.** potrf's `reached` row carries `native_route_existed` = any non-vendor
  candidate compiled and `native_route_supported` = any of them passes `can_run` for this shape
  (`select::native_facts`, passed through `TraceScope`), instead of the constants 1 and -1.
- **Vendor-free.** `potrf_candidates_tests` keys its expectations on `solver_vendor_available<B>`
  (Auto skips `vendor`; a `vendor` pin falls back). `posv_tests`'
  `FusedSolveArmSolvesOnBothTriangles` failed vendor-free on main too (Upper above both
  Upper-capable potrf tiers has no route without cuSOLVER); the test now expects `NoRouteError`
  there. `factor_bench` reports a refused pin as that arm's `bad=1` row
  (`pin refused: ...`) and keeps running the other arms.

**Phase 3.0 (select infrastructure):**
- **Exact keys drop one at a time, from the right.** §5.4 described an all-or-nothing exact-key
  filter. `Table::nearest` now keeps the rows matching the longest prefix of the `:exact` keys in
  `# keys:` order, so with `side:exact trans:exact` a missing (R,T) keeps the `side=R` rows; the
  weighted `Σ w·|log2(row/key)|` then decides among them. The exact keys need not be contiguous
  (`trans:exact n:log side:exact` works; `SelectNearest.ExactKeysAreDroppedFromTheRightOneAtATime`).
  For potrf's single exact key the behaviour is unchanged. `scripts/sweep_to_table.py`'s
  `nearest()` mirrors it.
- **Untimed rows** (`<spelling> -`) and `source=transcribed:<hex sha>` (§5.4, §13).
- **`Device::has_vendor_blas`** beside `has_vendor_solver`, both part of `describe()`'s memo key.

**Phase 3.1, posv** (`src/ops/posv/{choice.hh,posv.cc}`, plan §1.1):
- Families `Tiny`, `Cta`, `Blocked`, all `NoFields`, the same for every dtype; aliases
  `native:{tiny,cta,blocked}`; `last_resort {"blocked"}`; `# keys: uplo:exact n:log:3 nrhs:log batch:log`.
  No vendor family: bare `native` is Auto, `vendor` warns and falls back to Auto. The old names are
  kept, so `factor_bench`, `run_solve_grid.sh` and benchviz needed no vocabulary change.
- **New `can_run` term: Tiny requires `d.max_wg >= kPosvTinyWgSize` (64).** The old `supports()`
  lacked the driver's own work-group check (`posv_tiny.cc`), so on a device with `max_wg < 64` Auto
  could pick a driver that then throws. `kPosvTinyWgSize` lives in the sycl-free `solve_native.hh`, with a
  `static_assert(kTinyWg == kPosvTinyWgSize)` in `posv_tiny.cc` (the `kPotrfTinyWgSize` precedent).
  Cta's clause and the common `native` term are the old ones unchanged.
- **sm_89 tables are transcribed, untimed** (§13). `tools/transcribe/posv_transcribe.cc` (deleted in phase 5, `tuned/README.md`) builds
  host-only with g++ against a tree that still has `route_posv.hh` (`7e71a6e0`). It specialises
  `RouteTable` for a private `Op` value that forwards to the real `RouteTable<Op::posv, T>` but
  reports already-ranked routes as unsupported, then calls the real
  `resolve_route_uninstrumented(Auto, s, vendor_available = false)` repeatedly per cell, so the
  order comes from the deleted predicates, not a hand list. Capacities are unlimited, the device is a
  sub-group-32 GPU; the list stops after `blocked` (always runnable). Output:
  `tuned/transcribed/posv.sm_89.csv` (7920 rows, plain git, not LFS) and
  `tuned/posv.{float,double,cfloat,cdouble}.sm_89.txt` (1980 rows each, `source=transcribed:7e71a6e0`).
  Rows are `tiny | cta | blocked` inside the old `tiny_window` (float/double 720, cfloat 680,
  cdouble 480 rows) and `cta | blocked` elsewhere. Rebuilding the transcriber reproduces the CSV
  md5; `scripts/sweep_to_table.py --check` passes.
- **sm_120 posv tables: converted in the phase-5 fold** (below, "Phase 5 fold"), from
  `benchmarks/results/routing/sm120_posv_sweep.jsonl`, which was still running when P3.1 was
  written. Until then sm_120 borrowed the sm_89 transcription with the R8 warning.
- Cross-check: on 22 sm_120 cells (four dtypes, both uplo, the window edges cfloat n = 24/28 at
  nrhs 2 vs 4, cdouble n = 16 vs 20, float n = 32 vs 36, nrhs 8/16/64, float n = 1024 and cdouble
  n = 256) the coverage `reached` rows of the new build equal the parent build's. Off-grid cells
  use nearest-row lookup and can differ from the old thresholds at window edges (cfloat n = 25, 26
  at nrhs <= 2); those are what the sm_89 gate times.
- Deleted: `include/batchlas/blas/dispatch/route_posv.hh` (installed header),
  `src/backends/posv_route.hh`, and posv's half of `SOLVE_ONE` in `factorization.cc`.
  `Op::posv` stays in the enum until phase 5.
- **Implementer deviations from plan §1.1:**
  - **Embed generator (P3.0 infrastructure).** With ~0.5 MB of tables, `constexpr EmbeddedTable
    kTables[]` exceeded clang's constexpr step limit (building a `string_view` from a literal runs
    strlen at compile time). `cmake/BatchLASEmbedTables.cmake` now emits one `constexpr char kTextN[]`
    per table, with the length taken from `sizeof`.
  - **Empty shapes** (n, nrhs or batch 0) and **heterogeneous batches** (active dims on A or B) still
    throw `batchlas::internal_error`, from an explicit `throw_if_unservable` before `choose()`; the type
    is unchanged, the message is new. `can_run(Blocked)` is `homogeneous`, not the plan's `true`: the
    plan's premise that a failing child surfaces its own error does not hold here, because vendor
    `potrf` and `trsm` accept a heterogeneous batch and solve at the full storage order (known-defects
    [#12](known-defects.md#12-vendor-potrf-and-trsm-accept-a-heterogeneous-batch)).
    Under the first draft (`true`) such a call ran silently instead of throwing.
  - **Trace line.** `TraceScope` and `detail::trace_open` take an optional `const Key& fields`; empty
    prints `n= batch=` as before, posv passes `{n, nrhs, batch}`.
  - **Coverage** records posv's `reached` row at launch only, not in `posv_buffer_size` (as potrf),
    so a solve's `calls` count drops by one.
  - **Tests.** `tests/posv_candidates_tests.cc` (ctest label `blas`) holds the §8 suite: pinned
    straddles of every limit, Cta launched at exactly `getrs_fused_max_rhs_elems` and refused one
    past it, each pin bit-for-bit against the direct kernels, a saturating batch of 1024, the exact
    workspace in a poisoned arena, the pin rules, `AutoReadsEveryKeyField` on a synthetic table, the
    coverage native flags, and `HeterogeneousBatchIsRefusedUnderEveryPin` (A-only and B-only
    heterogeneous, Auto and every pin, `posv` and `posv_buffer_size`). P6
    `TinyRefusesShapesAboveItsCeilings` is deleted from `posv_tests.cc`; `CanRunEqualsLaunch`
    absorbs it. P7 is deleted; P7b pins posv with `ScopedPin(Cta{})` and potrf with
    `ScopedPin<PotrfChoice>("potrf", "native")`, without the readback; P8 pins `ScopedPin(Blocked{})`.
    Known gaps: the Tiny `max_wg` gate cannot go red on hardware whose `max_wg >= 64` (the predicate
    sees only the real device, as potrf's `kPotrfTinyWgSize` gate), and no test runs posv on NETLIB
    (the fixture is GPU-only). `tuned_tables_tests` gains `PosvTablesDeclareChoiceKeyNames` and
    `PosvSm89TablesHoldExactlyTheChoiceGrid`; the second exists because the transcriber spells the
    `choice.hh` grid by hand (it cannot include `select.hh`).
  - `factor_bench`'s posv pins go through select via a `select_pin_parsed<Choice>` shared with potrf;
    a refused pin is that arm's `bad=1` row.

**Phase 3.2, the tuner core** (`tools/tune/`, usage and raw schema in `tools/tune/README.md`):
- `batchlas_tune` builds with the benchmarks. Specs for potrf and posv (`<op>_spec.cc` behind
  `spec.hh`'s `OpSpec`); trsm and gemm specs land with their PRs. The host-only logic (tie rule,
  rotation, bisection and the refinement round, attempt selection and the re-measure rule,
  JSONL, SHA-256, the coverage `reached` parser, the `--old-csv` parser, the gate exit code,
  the `--devices` fence, the guard's process scan) is `tune_core.cc`, tested by
  `tests/tune_tests.cc` (label `util`, no GPU), which also checks the CMake staleness hash
  against the driver's and C++ `rank()` against the converter's on 400 random cells.
  Deliberate breaks of each of these turned exactly one or two named tests red.
- **The driver holds no CUDA context.** Linking libbatchlas enumerates devices during static
  init, which retains a primary context (about 550 MiB) on every visible GPU. `batchlas_tune` is
  therefore a SYCL-free launcher (`launcher.cc`) that starts `batchlas_tune_impl` with
  `CUDA_VISIBLE_DEVICES=""` for driver modes; children get their GPU set explicitly. The guard
  dies if the driver's pid ever appears on a GPU.
- **Tables go through the converter.** The tuner writes raw JSONL and runs
  `scripts/sweep_to_table.py --tuner`, which formats rows with the same `table_text` as converted
  tables. Header: `kernels=<hash>`, `source=tuner:<jsonl>`. `--check` re-derives a tuned table from
  its JSONL when present, and a tuned table supersedes the converted source of its (op, dtype,
  device).
- **§6.4 hash definition:** `sha256sum <files> | sha256sum`, first 8 hex digits, over the paths
  between the `kernel-sources-begin`/`-end` markers of `tools/tune/<op>_spec.cc`. The driver, the
  CI checker (`.github/ci/check_tuned_tables.py`, also in `run_local_checks.sh` and a CI job) and the
  CMake configure step (`cmake/BatchLASTunedStaleness.cmake`, one `WARNING` listing every stale
  table) each compute it. potrf: its tier kernels and `choice.hh`; posv: its own kernels and
  `choice.hh` only (children's coupling not followed yet, plan §2).
- **Departures from §6:** the flag is `--devices` (a list), not `--device`, and it is required (no
  default can land on a display GPU); when the caller exported `CUDA_VISIBLE_DEVICES`, every
  `--devices` entry must lie inside it. All listed GPUs must report the same device key. Multi-GPU
  runs one child per GPU behind a per-GPU flock, which departs from AGENTS.md §10 as the seed
  sweeps did. The guard is built in rather than `gpu_guard.sh`, with the same checks: no foreign
  compute process and utilization ≤ 5% before each child, no foreign process after it (else the
  child's numbers are discarded and it is retried), unparseable process entries counted as
  foreign; other busy GPUs on the box draw a warning. A failed child is retried once; if it fails
  again, each arm is run alone, the arms that crash alone are recorded as errors and the rest are
  timed together. The
  re-measure (§6.3 step 5) triggers when any candidate's two pass medians differ by more than 10%,
  repeats both passes, and the repeat replaces the first attempt unless no child of the repeat
  timed anything, in which case the first attempt stands. Refinement midpoints are
  geometric, round(sqrt(lo·hi)). The cap counts one copy of the op's inputs (potrf n²·batch,
  posv (n²+n·nrhs)·batch, times the scalar size). A candidate whose verification fails is dropped
  from that cell (`bad` in the raw file), not written as a flagged row. A refinement midpoint
  that exists but has no winner is not re-measured; its bracket is reported as `stalled`. The
  legacy CMake custom
  target `batchlas_tune` (`evaluation/tuning`, tuning_params constants) is renamed
  `batchlas_tune_constants` to free the name.
- **`--gate`** takes old choices from a CSV (legacy aliases normalised) or probes a parent
  `batchlas_tune` (`--parent-bin`, P3.2 or later) through its coverage `reached` row; the branch's
  Auto is probed the same way, and only differing cells are timed (`auto` vs the old pin, two
  passes, reversed). FAIL needs new/old > 1.05 in both passes. The gate cannot pass silently:
  exit 1 on any FAIL, else exit 3 when any row is `ERROR` or `BAD_ROW` or no cell was gated;
  cells over the cap get the verdict `cap`; an `--old-csv` row of the wrong width or of a dtype
  outside an explicit `--dtype` is an error, never a skipped cell.
- **Self-test (GPU 0, threadripper02, against `tuned/potrf.float.sm_120.txt`, full protocol):**
  potrf float L n=64 b8192: lpanel:panel=8 0.2851 ms (table 0.3023, -5.7%), vendor 0.4800 (-2.0%),
  cta 0.5561 (+4.9%), blocked 0.5587 (+4.3%), same first entry. n=128 b8192: lpanel 1.392 (+2.5%),
  vendor 1.808 (+2.5%), blocked 2.568 (+2.2%), same first entry (its pass 2 hit the re-measure).
  n=512 b2048: vendor 13.52 (-1.6%), blocked 14.09 (-0.1%), lpanel 14.73 (-0.7%); the first entry
  flips from `blocked` to `vendor` because blocked/vendor is 1.042 here against 1.026 in the sweep,
  either side of the 3% tie edge. posv float L nrhs=2 b8192 against the sm_120 posv sweep (single
  pass): n=24 tiny 0.04768 (+1.5%), cta 0.1022 (+4.3%), blocked 0.1691 (+5.8%); n=32 tiny 0.06917
  (-1.4%), cta 0.1223 (-0.7%), blocked 0.1779 (-0.2%). **These numbers are not protocol evidence:**
  they were taken with `--no-guard` while a three-GPU posv measurement sweep ran on the same box
  (AGENTS.md §10), and with the pre-launcher driver holding its own context on GPU 0. The -5.7% at
  n=64 and the n=512 flip need a re-run on an otherwise idle box before anything is read into
  them.

**Phase 3.3, trsm** (`src/ops/trsm/{choice.hh,trsm.cc}`, plan §1.2):
- Families `Cta`, `SgLeft` (spelling `sg_left`, P3.2b's `trsm_native_sg_left_dispatch`, its own
  family as the P3.2b note asks), `Blocked`, `Vendor`, all `NoFields`, the same for every dtype;
  aliases `native:{cta,blocked}`; `last_resort {"blocked","vendor"}`;
  `# keys: side:exact trans:exact order:log:2 q:log batch:log` with `q = side==Left ? B.cols : B.rows`
  and ConjTrans folded to `T`. uplo and diag are not keys, pending the invariance A/B (below).
  `cta:wg` and `blocked:outer` stay derived (phase 4). trsm has no `_buffer_size`, so R5 is vacuous.
- `can_run`: the common term is `is_gpu && !A.het && !B.het && order, q, batch >= 1`. Cta adds
  `order <= trsm_cta_max_n<T>()`; SgLeft adds `side == Left && has_sg32 && order <= 32 &&
  max_wg >= kTrsmSgLeftWgSize` (128; its `reqd_sub_group_size(32)` and work-group, with
  `static_assert`s in `trsm_sg_left.cc` against the sycl-free constants in `trsm_native.hh`);
  Blocked adds `trsm_blocked_available<T>() && trsm_cta_max_n<T>() >= 1`; Vendor is
  `d.has_vendor_blas`. **New term: Cta and Blocked require `max_wg >= 32`**, because V1's ladder
  (`trsm_v1_ladder_wg`) launches at least 32 lanes whatever `max_wg` says; it cannot go red on real
  hardware. Heterogeneous batches still go to the vendor (known-defects #12, unchanged).
- **`A.batch_size() != B.batch_size()` throws `invalid_argument` in `trsm_validate_params`** (Q6).
  It used to reach the vendor silently because the old shape builder returned nullopt.
- The public `trsm` moved out of `src/dispatch/entry_points/level3.cc`. Its coverage row now
  carries the real backend (the old builder never set it, so every trsm row read `AUTO`), keeps
  `m = B.rows, n = B.cols, k = order`, and records uplo, side, transA and diag; the trace prints
  `side trans order q batch`. Deleted: `include/batchlas/blas/dispatch/route_trsm.hh` (installed),
  `src/backends/trsm_route.hh`, and old rules T1 (`batch < 8` -> vendor) and T2 (float Side::Right
  at batch < 128 above order 32 -> vendor). `Op::trsm` stays in the enum.
- **Seams made mandatory.** `trsm_native_blocked`'s trailing gemm and `potrf_blocked_dispatch`'s
  panel solve no longer default to `gemm_custom` / `trsm_native_blocked`; an empty one throws
  `invalid_argument`. Every library caller already injected the public op; three direct-driver
  tests in `potrf_tests.cc` and the V2 helper in `trsm_tests.cc` now pass the public trsm / gemm.
  `potrf_blocked_dispatch`'s trailing gemm keeps its `gemm_custom` fallback until P3.4 (a default
  argument cannot precede the now-mandatory one, so both defaults are gone and callers pass `{}`).
  P3.4 made that empty fallback the public gemm (see Phase 3.4); the phase-5 rip made it
  mandatory too.
- **sm_89 tables are transcribed, untimed.** `tools/transcribe/trsm_transcribe.cc` (deleted in phase 5, `tuned/README.md`; host g++
  against 8b9adeb3) re-runs the real `resolve_route_uninstrumented` over the old table, first with
  the vendor present, then (after the vendor is ranked) vendor-free for the remaining natives. The
  CTA capacity is applied, not unlimited: `trsm_cta_max_n<T>()` is a build constant (32 for every
  type and device), and an unlimited one would rank `cta` first in every row. On the grid
  (batch >= 128) the old preference was native everywhere, so rows are `cta | blocked | vendor` at
  order <= 32 (7680 of 17280) and `blocked | vendor` above (9600). Below the grid T1 and T2 are lost
  (checked with the same transcriber on batch {4, 64}: `vendor|cta|blocked`, `vendor|blocked`). No
  transcribed row names `sg_left`. Output `tuned/transcribed/trsm.sm_89.csv`,
  `tuned/trsm.<dtype>.sm_89.txt` (4320 rows each); `--check` passes. sm_120 borrows them (R8) until
  the tuner sweep.
- Cross-check against the parent build (886537e8, old routing) on GPU 0 of threadripper02, one
  process per cell, coverage `reached` rows: 25 cells (4 dtypes, both sides, N/T/C, order 1-1024
  across the 32/48 edge, q 1-4096, batch 128-32768, on- and off-grid) agree. A 26th (cdouble R T
  order 384 q 1 b128) crashes on both builds (known-defects #13; the new trace names `blocked`, the
  old predicate gives `blocked`). Three below-grid cells differ as expected: float L N 16/4/b4,
  float R N 64/8/b64 and cdouble L N 8/8/b1 went to the vendor and now run `cta`, `blocked`, `cta`.
- **K4, the complex "vendor" trsm** (BatchLAS's own kernel in `cublas.cc`): the `int` batch offset
  `b * strideA` wrapped at 2^31 elements and faulted (`CUDA_ERROR_ILLEGAL_ADDRESS`). The plan's
  shape, cfloat order 512 x batch 8192, is exactly 2^31 elements and just fits (last offset
  2^31 - 1); batch 8193 faulted on the parent. Fixed with 64-bit offsets; guarded by
  `TrsmVendor.ComplexSubstituteIndexesPast2To31Elements` (order 512, batch 8193, about 17 GB, skips
  below 32 GiB of device memory; the pre-fix source faults it).
- **cuBLASLt crash workaround** (known-defects #13): with T1 gone, trsm at batch <= 5 runs
  `blocked`, whose single-rhs trailing updates are complex<double> gemms with a unit dimension, which
  segfault inside cuBLASLt on threadripper02 (the parent crashes the same way at batch >= 8).
  `gemm_vendor_impl` now uses the typed `cublasZgemmStridedBatched` there. Without it
  `posv_tests`, `posv_candidates_tests` and `getrf_tests` crashed (getrf also crashes on the parent).
- Tuner spec `tools/tune/trsm_spec.cc`: diagonally dominant triangular A with the other triangle
  poisoned (1e6), random B, alpha 1.5-0.5i; verification is the componentwise backward error
  max_i |op(A) X - alpha B|_i / (|op(A)| |X| + |alpha| |B|)_i on items 0 and batch-1 over at most
  32 sampled right-hand sides; `trans=T` times ConjTrans for a complex scalar. The off-diagonal
  coupling is 0.5/(1+|r-c|)^2, independent of order. The first draft (coupling 0.5/order/(1+|r-c|),
  normwise residual) let a `blocked` whose trailing gemm did nothing pass Tol<float> from order 256
  up (review finding, modelled at 3e-5..2e-6). Armed: with the trailing gemm's alpha forced to 0,
  every `blocked` cell of a float/cfloat/double probe (L/R x N/T x order 64/256/1024, q 32, b128)
  is `bad` at 0.14-0.82 while the vendor stays `ok`; unbroken, the worst is 6.6e-7 (float) and
  3.3e-15 (double). `bytes()` counts A, B0 and X (it counted B once), so `--cap-gib` means what
  it says; `posv_spec` likewise now counts A0, A, B0 and X.
  **The uplo/diag A/B is a flag:** `uplo` and `diag` are hidden grid axes fixed at L and N, so
  `--grid uplo=L:U --grid diag=N:U --no-refine` times all four; the converter refuses to make a
  table from such a run. A pin is probed by one untimed run (no sizing call). Smoke (GPU 0, 2 passes
  x 4 reps, other GPUs busy, not evidence): float/cfloat order 16/64 x q 4/64 x b8192 ranked all
  four candidates with no bad verification; `sg_left` led every Left order-16 q=4 cell (2.2x over
  cta for float). The A/B smoke (double, order 24/96, q 8, b2048) showed vendor Side::Right order 24
  0.45 -> 0.32 ms from diag N to U, so the vendor may fail the <= 3% invariance; the real A/B is
  part of the sm_120 sweep.
- `factor_bench`'s posv `composed` arm pins trsm `cta` only up to `sycl_trsm::trsm_cta_max_n<T>()`
  (asked at run time, like potrf's `potrf_tiny_max_n<T>()`) and `native` above (a `cta` pin past
  its ceiling now throws). `trsm_benchmark` announces the pin without the old parser.
- Tests: `route_vocabulary_tests` loses `trsm_shape` and its 9 `RouteTrsm.*` tests; the four the
  plan names are ported to `trsm_candidates_tests`: `:422` -> `VendorFreeLastResortIsBlocked`;
  `:437`/`:445` -> `CanRunFalsePinsThrow`, `BlockedServesPastTheCtaCeiling`,
  `HeterogeneousBatchHasNoNativeRoute` and `TrsmCandidatesCpu.CpuQueueRunsNoNativeFamily` (`:437`
  not exactly: `trsm_cta_max_n` is a build constant, so "cta_max = 0" cannot be simulated);
  `:474` -> `TraceKeyQFollowsSide`. `RouteGetri`'s citations of
  `route_trsm.hh` now cite trsm's `can_run`. `tuned_tables_tests` gains
  `TrsmTablesDeclareChoiceKeyNames` and `TrsmSm89TablesHoldExactlyTheChoiceGrid` (since the
  phase-5 fold `TrsmTranscribedTablesHoldExactlyTheChoiceGrid`, which also covers sm_120 complex).
- `trsm_candidates_tests` (typed over 4 dtypes, GPU 0, vendor build): every candidate straddling
  its limits on both sides, all 24 side/uplo/trans/diag combinations, non-natural ld/stride with
  large finite poison, parent-ld sub-views, pinned-equals-direct bit for bit, a batch-1024
  saturating case, pins/errors, tables, coverage columns, and ortho's Right-side shapes
  (`OrthoCallerShapes`: order 2-3, q 12, T and C, both uplo). Deliberate breaks, each restored and
  md5-verified, red sets per dtype /4-/7: cta cap widened -> 6 tests; sg_left allowed on Right ->
  `PinnedCandidatesStraddleTheirLimits`, `CanRunEqualsLaunch`, `CanRunFalsePinsThrow`; C->T fold
  dropped -> `AutoReadsEveryKeyField` ("trans (C folds to T)" row only); cta on Left running the
  sg_left driver -> `PinnedRunIsTheDirectKernelBitForBit`; q always `B.cols` ->
  `AutoReadsEveryKeyField` (Right q row), `TraceKeyQFollowsSide`, `CanRunFalsePinsThrow`; batch
  check dropped -> `BatchMismatchThrows`; heterogeneity term dropped from the native `can_run` ->
  `HeterogeneousBatchHasNoNativeRoute` only (cta, sg_left, blocked "accepted"); `key_of` order =
  `B.rows()` -> `AutoReadsEveryKeyField` ("order (Right: A.rows)" row only),
  `TraceKeyQFollowsSide`, `AutoReadsTheSm89TranscribedTable` (the last skips once sm_120 has its
  own table; the first two do not). The trace key is `key_of()` itself, so the trace always shows
  the key the table lookup used.
- **sm_120 (phase-5 fold):** float and double are tuner tables
  (`benchmarks/results/tuning/trsm.{float,double}.sm_120.jsonl`); the maintainer stopped the sweep
  during cfloat, so cfloat and cdouble are the sm_89 transcription written for sm_120. Until the
  fold, sm_120 borrowed sm_89 with the R8 warning. The plan said that if the A/B showed a family
  moving more than 3% with uplo or diag, those would become keys before conversion; the A/B
  below fails that literal test, mostly on diag, and the keys were left unchanged for the
  maintainer to decide.
- **uplo/diag invariance A/B (sm_120 only, run before the sweep).** `batchlas_tune trsm --grid
  uplo=L:U --grid diag=N:U --no-refine`, float and cdouble, side L/R x trans N/T x order 8, 32,
  128, 512 x q 4, 64 x batch 2048, full protocol: 32 float and 24 cdouble cells x 4 (uplo, diag)
  combinations (float L T 128/64 lost its U,N combination). Per candidate, the spread is max/min - 1
  over the 4 combinations:

  | dtype | candidate | spread <= 3% | median | max | uplo only <= 3% (max) | diag only <= 3% (max) | pass-to-pass noise (median / p90) |
  |---|---|---|---|---|---|---|---|
  | float | cta | 3/16 | 5.6% | 16.3% | 7/16 (13.1%) | 5/16 (16.3%) | 0.7% / 6.6% |
  | float | sg_left | 1/8 | 4.6% | 5.9% | 4/8 (5.3%) | 4/8 (5.8%) | 0.6% / 3.2% |
  | float | blocked | 6/31 | 5.8% | 19.0% | 9/31 (16.7%) | 8/31 (18.9%) | 0.5% / 4.8% |
  | float | vendor | 0/31 | 10.6% | 20.2% | 5/31 (15.4%) | 2/31 (20.2%) | 0.8% / 4.6% |
  | cdouble | cta | 0/16 | 13.2% | 26.0% | 14/16 (5.0%) | 0/16 (23.4%) | 0.2% / 1.2% |
  | cdouble | sg_left | 0/8 | 58.9% | 82.1% | 7/8 (4.0%) | 0/8 (82.1%) | 0.2% / 0.9% |
  | cdouble | blocked | 4/24 | 7.4% | 26.8% | 22/24 (5.2%) | 4/24 (22.6%) | 0.2% / 1.2% |
  | cdouble | vendor | 1/24 | 26.9% | 93.6% | 24/24 (2.3%) | 1/24 (91.5%) | 0.1% / 0.4% |

  So the plan's per-candidate test fails. diag=U is clearly faster for cdouble (no complex
  division: vendor 0.21 -> 0.11 ms at order 8 q 64, sg_left 1.47 -> 0.81 ms at order 32 q 64).
  uplo moves cdouble by at most 5.2% (vendor 2.3%). The float spreads in both axes, 4-20%, sit
  near float's pass-to-pass noise (p90 3-7%). **The ranking is invariant**: in all 56 cells
  every combination has the same first entry, and the full ranked list differs in 2 cells,
  where entries 2 and 3 swap (float L T 32/4, cdouble L N 32/4). The table answers "which
  candidate", not "how fast", so a key on uplo or diag would add rows without changing a
  choice on this grid. Keys unchanged (`side trans order q batch`), as decided in the fold;
  adding `diag:exact` is the maintainer's call if a timed diag=U table is ever wanted. Not run
  on sm_89. Raw and the script that prints this table: `benchmarks/results/tuning/trsm_uplo_diag_ab/`.
- **Implementer deviations from the plan / orchestrator brief:** (1) the transcriber applies
  `trsm_cta_max_n` (32) instead of unlimited capacities (see above); (2) the new `max_wg >= 32`
  term on Cta/Blocked and `max_wg >= 128` + sub-group 32 on SgLeft; (3) Vendor `can_run` is still
  `has_vendor_blas` alone (known-defects #12 left for its own phase); (4) both `potrf_blocked`
  default arguments removed, not only the panel-solve one; (5) the out-of-scope `cublas.cc`
  typed-ZGEMM workaround (#13); (6) the vendor-free failure baseline came from the main snapshot's
  `build-vf` (same old trsm/getrf/ortho routing), since the parent tree has none; (7)
  `trsm_candidates_tests` and the four ported cases were written by a separate pass (see Tests
  above); (8) `trsm_native.cc` still includes the now-unused `gemm_kernels.hh` (left to avoid another
  device-link rebuild).
- **Gate status (threadripper02, GPU 0):** the vendor build's 15 affected suites pass except
  `ortho_tests`, which segfaults at the same case on the parent (#13, gemv path). The crash
  masks more than that case: excluding `OrthoMatrixTest/7.OrthogonalizeMatrix`, the next cdouble
  case (`OrthoAgainstMTest/7.OrthogonalizeMatrixAgainstM`) segfaults too, in both builds, so no
  cdouble CUDA ortho case completes. Its trsm calls (Right, T, order 2-3, q 12, batch 2) moved
  from the vendor to `cta`; `trsm_candidates_tests`' `OrthoCallerShapes` covers that shape
  directly. The vendor-free
  build fails the same 42 names as the old-routing baseline, none added or removed (NETLIB-only
  cases plus `LuTest/{4,6}.TinyRoutesInsideItsMeasuredWindowAndNowhereElse`).
  `facade_symbol_check trsm` is OK; `run_local_checks.sh` fails only `check_cmake_syntax` on the
  untracked `build-vf/` in the worktree. The ROCm syntax check was not run (no ROCm headers on
  this box). Plan risks: K3 materialised (the batch <= 5 suites moving to native exposed #13,
  worked around for gemm); K4 is real and fixed (above); K6 (C folded to T) moves no cell on the
  grid, since the sm_89 rows are side/order-determined and identical for N, T and C, and the
  26-cell cross-check covered all three; it starts to matter only once a timed table differs.

**Phase 3.4, gemm** (`src/ops/gemm/{choice.hh,gemm.cc}`, plan §1.3; per-kernel detail in
`docs/perf/gemm.md#choices-flat-selection-p34`):
- Families `direct`, `tiled`, `small` (real only), `reg:m=..:n=..:k=..:u=..` (float only, 10
  configs), `wide:m=..:n=..:k=..` (5 configs, every scalar), `vendor`. Candidates: float 19,
  double 9, cfloat/cdouble 8. `# keys: ta:exact tb:exact layout:exact m:log n:log k:log batch:log`,
  ConjTrans folded to `T` for real scalars; `layout=packed` when A, B, C are contiguous with
  16-byte-aligned bases, else `strided`. `last_resort {"direct","vendor"}` (§13). No workspace, so
  R5 is vacuous. The old `RouteTable<Op::gemm>`, `select_kernel_variant` and the cuBLAS/rocBLAS
  `gemm_use_sycl_custom` re-route were three deciders; they are now one table row.
- **Fold table** (forms after the C->T fold; the old variant list is in gemm.md):

  | choice | forms | old `KernelVariant`s |
  |---|---|---|
  | `direct` / `tiled` / `small` | any | `Direct` / `Tiled16` / `SmallBatched` |
  | `reg` 32·32·8, 64·64·8, 128·64·32 u=4 and u=2, 128·128·8 | NN | the NN-only register tiles |
  | `reg` 64·64·16, 128·32·16, 128·32·32 | NN NT TN TT | `{,TN,NT,TT}`; 128·32·32 also `S2U1{,Aligned,Generic}` (legs) |
  | `reg` 128·64·16 / 32·128·16 | NT TN TT / NN TN TT | `Tiled128x64RegisterK16*` / `Tiled32x128RegisterK16*` |
  | `wide` 64·64·16 / 128·32·16 / 32·128·16 | NN CN NC / NC / CN | the `...Wide{,CN,NC}` tiles |
  | `wide` 32·32·16, 16·16·16 | NN | the two P3.2b tiles (decision 5) |

  A real `Trans` is served by a wide `ConjTrans` instantiation; a complex `Trans` is not.
- **Deleted** (decision 1): the four pin-only `Tiled128x32RegisterK32{S1U1,S2U2,S2U2TT8x4,S2U2TT4x8}`
  and the five experimental `Tiled128x32RegisterK32{Persistent,SplitK4,S1U4}`,
  `Tiled128x64RegisterK32LargeTT4x8{,U2}`, with `src/sycl/gemm/{persistent,split_k}.hh`, the
  `BATCHLAS_GEMM_EXPERIMENTAL` gate, `include/batchlas/blas/dispatch/route_gemm.hh` (installed),
  the routing half of `src/backends/gemm_variant.hh`, `gemm_custom`, `wide_transposed_tile_for`,
  the cuBLASDx env path in `gemm_vendor` (cuBLASDx is unreachable from gemm; its TUs still build for
  the level-3 ops) and `tests/route_gemm_equivalence_tests.cc`. Their names have no alias and throw.
- **Correctness.** `can_run` lists exactly the instantiated (config, ta, tb) forms, so the 18
  NN-only register variants that silently computed NN on a transposed call (`launch_reg` took no
  transpose) can no longer be chosen, and a pin onto such a form throws. The launchers
  (`gemm_direct`, `gemm_tiled`, `gemm_small`, `gemm_reg`, `gemm_wide` in `gemm_kernels.cc`) throw
  too instead of falling back to `Tiled16`. The aligned vs predicated leg, the transpose
  instantiation, the `small` bucket and TR/TC/stages are derived in the launcher, never a field, a
  key or a `can_run` term (`layout` is a key, not a gate), so the leg-predicate-as-routing-gate
  defect (`docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix`) cannot recur.
- **Callers (decision 4, risk K5).** The six `gemm_vendor` calls in `cublas.cc` (hemm x2, herk,
  her2k, trmm x2) call the public `gemm<B,T>`; float symm/syrk/syr2k's custom dispatch already did.
  The potrf, geqrf and getrf blocked drivers' trailing-gemm seam is the public gemm, not
  `gemm_custom` (P3.4 made an empty seam fall back to it through `with_backend`; the phase-5 rip
  made the seam mandatory in all three). `trsm_native.cc` drops
  its unused `gemm_kernels.hh` include (P3.3 deviation 8 closed). A heterogeneous batch is split
  into homogeneous items before `choose()` in every build; each item chooses for itself.
- **select changes.** `Rules` gains `class_aliases` (`register_tiled`, `sycl`, `custom` ->
  `native`; `vendor:auto` -> `vendor`) and `legacy_aliases`, applied before the class words, so an
  alias to `vendor` stays concrete (§12 phase 1; both arrays and the `_VARIANT` read are deleted in
  phase 5). `pin_text` reads `BATCHLAS_GEMM_VARIANT` only when
  `_ROUTE` is unset, with its own words (`native`, `cuda-native`, `direct-cuda`, `cublasdx`, `dx` ->
  `vendor`; `sycl`, `custom` -> `native`). `Table::nearest` is memoized per table (the gemm tables
  hold thousands of rows); host cost measured +2-4 µs per tiny call against old routing, noisy.
  `TraceScope` records m, n, k, transA, transB, backend and precision; the trace prints the key.
- **sm_89 tables are transcribed, untimed** (decision 6). `tools/transcribe/gemm_transcribe.cc` (deleted in phase 5, `tuned/README.md`)
  links against a built 424a45bc tree and calls the real old `preferred()` and the exported
  `select_kernel_variant<T>`. `packed` cells use contiguous aligned views, `strided` ones
  ld = rows + 1 with an odd batch stride (fails every aligned-leg predicate). A row is the old
  kernel, its old forced-name fallback (`reg`/`wide` -> `tiled`, `small` -> `direct`), the other
  of `tiled`/`direct`, then `small` for a real max(m, n, k) <= 64 (the only 1-D launch, so the
  one native that survives batch > 65535), with `vendor` first where the old route was the vendor
  and last otherwise. **Edge rows** (review fix; transcription only, not in the tuner grid): the
  old `preferred()` had edges below the grid (batch < 64 and double k < 2 went to the vendor; float
  was native only on NN squares <= 48), which nearest lookup carried the native rows across. Real
  types also get batch {1, 63, 64}, double gets k {1, 2} at every (form, layout, m, n), and float NN
  gets the squares 1, 2, 4, 40, 49, 56 plus one-axis-off neighbours (1..64) of the squares 8..48.
  Batch and double-k edges are now exact; float near-squares between a square and its off-axis
  neighbours (e.g. 40x40x41) still land on whichever row is nearer, an accepted deviation for the
  gate. Pinned by `GemmTranscribedTable.Sm89BracketsTheOldVendorEdges` (red with the pre-fix
  tables). Output `tuned/transcribed/gemm.sm_89.csv` (38,826 rows), `tuned/gemm.float.sm_89.txt`
  (11,226 rows), `tuned/gemm.double.sm_89.txt` (12,672), `tuned/gemm.{cfloat,cdouble}.sm_89.txt`
  (7,464 each, unchanged); `--check` passes (run
  through a wrapper: the live `sm120_posv_sweep.jsonl` carried a duplicate row). The rows: float
  `small` first on the 30 NN squares up to 48, else `vendor` then the old native kernel; double
  native everywhere (`tiled` 4,395 rows, `direct` 102, `wide:m=64:n=64:k=16` 15); complex `vendor`
  first everywhere. Cross-check against the parent (GPU 0, kernel trace + coverage, Auto and
  `ROUTE=native`, 5 cells off-grid): 34 of 34 cells pick the same kernel, outputs correct; the
  pre-fix tables disagreed on every below-grid cell (batch 8/16, double k = 1, float 52^3 and
  40x40x48), which the edge rows fix. Five off-grid cells did not sample the rectangle edges; see
  the two phase-5 review bullets below.
- **Off-grid disagreement with the old router (phase-5 review; recorded, not fixed).** The
  transcribed rows are exact on the grid (a Python replica of 7f9e65ca's `select_kernel_variant`
  reproduces 12,672/12,672 double and 11,226/11,226 float rows), but nearest-row lookup does not
  reproduce the old predicates between grid points. On random off-grid shapes (every dim >= 8,
  half of them panel shapes with k in 8..128) the first runnable native entry differs from the old
  kernel on 6.7% of double and 31.7% of float shapes. Cause: the double `max_dim <= 24` (NN) and
  `<= 32` (transposed) Direct/Tiled16 edges and the float `min(m,n)` 32|64|128 and k 128 edges are
  bracketed only on squares, so a rectangle snaps to a square row. Kernel traces confirm it on the
  factorizations' panel updates: geqrf double 80x80 b256 (NN 32x16x16, TN 16x64x16: Tiled16 ->
  direct); vendor-free getrf/gesv double n=100 and getrs n=200 (4x4x32: Tiled16 -> direct);
  vendor-free geqrf float 600x90 b64 (NN 600x58x32: reg 32x32 -> reg 128x128; NN 568x26x32:
  Tiled16 -> reg 32x32; TN 32x58x32: Tiled16 -> small); vendor-free orgqr float 600x300 (TN
  32x300x600: Tiled16 -> reg 128x32x32, which main gated on m >= 128; NN 344x12x32: Tiled16 -> reg
  32x32). Impact: double gemm in every build (it was native everywhere), every real dtype in
  vendor-free builds; float and complex stay vendor-first in vendor builds. The §13 sm_89 off-grid
  gate was never run for gemm, so "Auto equals the old routing" does NOT hold off-grid for gemm.
  Accepted as a pending-retune deviation (no measurement pass in phase 5); a re-transcription would
  need rectangle points straddling those edges (`git show 0bd26dfe:tools/transcribe/gemm_transcribe.cc`).
- **Fast-path divisibility is not a key (phase-5 review; recorded).** Main sent double/complex NN
  to `Tiled64x64RegisterK16Wide` only when `min_dim >= 256 && can_use_64x64_k16_wide_fast_path`
  (m, n multiples of 64, k of 16, aligned), and the float reg 128x128 and aligned 128x64/128x32 NN
  tiles only on their own divisibility predicates; the predicated wide leg was never chosen. The
  transcriber evaluated `packed` grid points, all multiples of 64, and the key has no divisibility
  term (by design: the leg is derived in the launcher, R3 keeps it out of `can_run`). So a packed
  non-multiple shape near those rows now takes the predicated leg: double 304^3 b64 runs
  `wide:m=64:n=64:k=16` where main ran Tiled16 (also syev double n=300's 304^3 update), as do
  668x3x904 and 541x680x1657 (rows at 768^3); vendor-free float shows the same on the reg tiles.
  Untimed against Tiled16 (`docs/perf/gemm.md`); accepted pending the phase-4 retune.
- **Review fixes to `can_run` (R3).** `small` requires `d.has_sg32` except on the float NN tiled
  leg (33..56), whose kernel has no `reqd_sub_group_size` (`small_fits` in `choice.hh`, probed on
  synthetic devices). `direct`, `tiled`, `reg` and `wide` put the batch in SYCL dim 0 = CUDA grid z,
  so they require batch <= `kMaxGridBatch` (65535); before, Auto chose them past it and the launch
  threw ("Number of work-groups exceed limit"), in the parent too. `GridCeilingIsACanRunTerm` pins
  every such choice at 65535 (runs) and 65536 (refused, C untouched); Auto past it runs `small`
  (real) or the vendor (complex; vendor-free complex has no 1-D native and throws `NoRouteError`).
  An empty batch returns before `choose()` under any pin (the old native range was a no-op; the
  `batch >= 1` term alone left vendor-free no route). The unpredicated leg of reg 128·64·32 u=4/u=2,
  reg 128·128·8 and wide 64·64·16 NN now has its own kernel-trace name (`..._aligned`, as 128·32·32
  had), so a fast-leg predicate stuck at false goes red in `PinnedChoiceLaunchesItsOwnKernel`.
- **sm_120 is transcribed (phase-5 fold).** No gemm sweep ran; `tuned/gemm.*.sm_120.txt` are the
  sm_89 CSV written for sm_120 (`--device sm_120`), so Auto there runs what the 4090 router chose
  without borrowing (the blackwell.md windows are hypotheses for the sweep).
  The sweep uses `tools/tune/gemm_spec.cc` (beta = 1, alpha with an imaginary part, strided cells
  padded by max(1, `--ld-pad`), componentwise error against a double/cdouble host reference on items
  0 and batch-1 over 64 sampled columns, the plan §3 demand-driven grid per dtype, refinement on `k`,
  full §6.3 protocol, no screening pass: decision 2). `OpSpec::grid` became virtual and takes the
  dtype; `--grid` filters the declared grid instead of replacing axes. Armed: NN run on the
  128·32·16 NT form gave `bad` (0.52) for that arm only on a square cell and an illegal address on
  a panel cell, and exactly one red gtest; restored and md5-verified.
- **Gate status (GPU 0).** Vendor build `ctest -LE slow`: apart from `gemm_tests` and
  `gemm_tests_native` the failing names match the parent log (syevx, lanczos, gemv and ortho
  segfaults from #13, cond, syev_blocked). Every gemm consumer suite (trsm, potrf, posv, geqrf,
  getrf, orgqr, symm, hemm, herk, her2k, syrk, syr2k, trmm, the candidates suites, `_native`
  reruns, tuned_tables, select, route_vocabulary, settings, tune, device_calls) passes. The 142
  `gemm_tests` failures were pinned-kernel tests that now throw instead of falling back (120 GPU
  pins on NETLIB instantiations, 22 on CUDA: deleted names, NN-only small tiles on transposed
  calls, `reg` for non-float, `small` for complex, complex `Trans` on wide), none a wrong answer;
  `gemm_tests` was then ported to `ScopedPin` and refusal assertions and passes, and
  `gemm_candidates_tests` covers every candidate. The vendor-free build's non-gemm
  failing names equal a parent vendor-free build (`p34base/build-vf`). `facade_symbol_check gemm`
  OK; `--gate` against the parent reads old gemm coverage as `native:register_tiled`, which now
  means "best native", so gate gemm from a kernel-trace old-choice CSV.
- **Implementer deviations from plan §1.3:** (1) last resort `{direct, vendor}` (§13);
  (2) native families run when `d.is_gpu || !d.has_vendor_blas`: without the second term a
  vendor-free host queue lost gemm and 32 parent-green names went red; (3) `wide` has 5 configs,
  not 3 (the P3.2b tiles); (4) the `small` work-group term is exact (`small_wg`: 144 or 196 lanes for
  float NN 33-56), not 128, and cannot go red on real hardware; (5) a set `BATCHLAS_GEMM_SYCL_KERNEL`
  throws (its names are `BATCHLAS_GEMM_ROUTE` aliases) instead of being ignored; (6) a heterogeneous
  batch chooses per item in vendor builds too (it used to loop on the vendor); (7) the gemm
  benchmarks and `scripts/run_gemm_*.sh` were ported to `BATCHLAS_GEMM_ROUTE`, and the cuBLASDx row
  of the heterogeneous benchmark became `native`. Not in `can_run`, unchanged from before: the
  direct kernel's `int` batch offsets can overflow at large batch x stride (§11).

**Behaviour changes visible to callers:**
- A bad pin throws. `factor_bench`'s posv `composed` arm therefore pins potrf to `tiny` only up to
  the type's tiny ceiling (16 for cdouble, 32 otherwise) and to `native` above it.
- The vendor coverage readback is `vendor:vendor`, not `vendor:auto`. benchviz matches
  `startswith("vendor")` and the converter accepts both.

**Data findings from the converter** (for the gate and phase 4):
- The sm_89 archive has no current-era `lpanel` timings, so the sm_89 tables never pick `lpanel`,
  although `docs/perf/potrf.md#the-measured-lpanel-window` measured it winning there.
- The sm_120 sweeps have `uplo=U` rows for float only. The other sm_120 dtypes serve Upper from
  the nearest Lower row (the exact-key drop of §5.4), so they rank only the Upper-capable entries
  of a Lower measurement.
- **Off-grid lookups near the 4 GiB sweep cap landed far away in `n`** while `n` and `batch`
  weighed equally: float n=704 batch=8192 mapped to the n=320 row, since the large-n rows at that
  batch were never measured. With `n:log:3` it maps to the n=640 batch=2048 row (`blocked`). The
  grid is still unfilled there (phase 4).

**Gate results (2026-10-04, threadripper02, sm_120):**
1. Correctness: `select_tests`, `tuned_tables_tests`, `potrf_candidates_tests`, `potrf_tests`,
   `potrf_tests_native`, `posv_tests`, `route_vocabulary_tests` pass in the vendor build and in a
   `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` build. In the vendor-free build, `posv_tests`
   `FusedSolveArmSolvesOnBothTriangles` failed identically on `main` (no native potrf for Upper
   past the CTA ceiling); the test now expects `NoRouteError` there. `ctest -LE slow` over all
   built tests: the branch and `main` have the same 11 failing names on this box (all pre-existing),
   plus three new tests that pass.
2. `scripts/sweep_to_table.py --check`: OK.
3. Live, at saturation: 120 off-grid cells (§10.3 n points × batch 512/8192/32768 × 4 dtypes,
   Lower). 54 chose the same kernel as `main`; 33 changed and were timed interleaved in two passes
   with reversed arm order: 0 FAIL, worst 1.014x (cdouble n=44 b8192), the rest 0.60-0.994x. 33
   cells were not timed: 32 have matrices over 12 GiB, and n²·batch ≥ 2³¹ aborts in `factor_bench`
   on `main` and on the branch alike (float n=272 b32768), independent of the route. Evidence:
   `benchmarks/results/routing/sm120_potrf_phase2_gate.csv` and its README section.
4. Readable: `potrf float n=100 batch=8192` on sm_120 → `potrf.cc` `choose()` → nearest row
   `uplo=L n=96 batch=8192` in `tuned/potrf.float.sm_120.txt` → `lpanel:panel=8`.
5. Open: the sm_89 live gate. The sm_89 tables are sparse (32-88 rows) and contain no current-era
   lpanel timings, so on sm_89 the new choices are least certain; run it before merging, or accept
   it as a phase-4 retune item.

### Phase 5, the eleven transcribed ops

These eleven ops share one recipe, so it is stated once:

- **Layout (R1).** `src/ops/<op>/{choice.hh,<op>.cc}` holds the vocabulary and the whole path:
  public entry -> `choose()` -> one `std::visit` launch. The public entry points moved there out of
  `src/dispatch/entry_points/` (and, for gesvd, ormqr and syev, out of inline header templates;
  their installed headers keep only the `BATCHLAS_API` declarations and the `sig::` aliases).
- **Families.** One per kernel driver, all `NoFields`, the same list for every dtype. The old router
  chose no knob, so every derived parameter (blocking factor, panel leaf, tiny bucket, CTA packing,
  WY width, work-group size) stays derived in its driver.
- **can_run (R3)** is the driver's own refusal, clause for clause. The native terms common to the
  dense factorizations are a GPU, sub-group 32, a homogeneous batch, extents >= 1, and, where the
  native kernel packs 1-based int32 pivots into the int64 span, `B != NETLIB` (LAPACKE writes real
  int64). The vendor term reads `select::Device::has_vendor_solver`, filled from
  `factorization_vendor_available<B>` (`src/select/vendor.hh`: cuBLAS and cuSOLVER on CUDA), which
  is the same constant that compiles the Vendor arm. A vendor-free build where nothing can run
  throws `NoRouteError` through `throw_no_vendor_route`, so coverage keeps its `miss` row.
- **Workspace (R5).** `<op>_buffer_size` runs the same `choose()` and returns the chosen family's
  need, not the old `max(every supported tier, vendor)`. A caller that sizes once and then pins a
  different family re-queries.
- **Pins (R6).** A pin `can_run` refuses, or an unknown spelling, throws `invalid_argument`. Under
  the old router both silently meant Auto, and a forced unsupported route fell to the vendor. On a
  NETLIB queue native pins are now evaluated (and throw) where the old dispatch skipped resolution.
- **Tables.** The old predicates read no architecture, so one transcription is written for sm_89
  and sm_120 (spmm also cpu) with identical rows (`tuned_tables_tests` asserts it), all
  `source=transcribed:424a45bc` (provenance and regeneration: `tuned/README.md`). The grid puts an
  integer point on both sides of every threshold an old predicate read and is a full product, so
  the weighted nearest row snaps per axis and reproduces every step exactly. Device capacities
  (SLM ceilings) are unlimited in the transcription and re-applied by `can_run`; build constants
  (tiny ceilings) are applied. Each row lists the old vendor-present Auto choice, then the order
  the old router took once each higher entry was excluded, then the old vendor-free walk.
- **Coverage.** `chosen_algo` is the choice spelling (§5.6): a vendor row reads `vendor,vendor`
  where the old one read `vendor,auto`; the backend column is the real one.
- **Gates**, run per op on GPU 0 of the sm_89 or sm_120 box against the same targets built from
  `424a45bc`: (a) failing gtest names identical to the base in the vendor and vendor-free trees,
  plus deliberate breaks of `<op>.cc` each turning a narrow named red set; (b) an off-grid data gate
  replaying thousands of random points through the old `RouteTable` + `resolve_route` (the
  transcriber's point mode) and through the converter's `nearest` plus a `can_run` model, at 100%
  required (a negative control, removing threshold rows, proved each gate can fail); (c) the
  coverage `reached` row of an Auto call, one process per cell, on the old and new binaries. The
  per-break red sets and per-cell lists are in the original notes,
  `git show 94cefb3a:docs/design/flat-select-p5/<op>.md`.

What follows is only what differs per op.

### Phase 5, gemv

Families `cta` (`gemv_native_cta`, bodies 3/5), `direct` (`gemv_native_direct`, bodies 1/2/4) and
`vendor` (`gemv_vendor`, `d.has_vendor_blas`). The body split and segment width stay derived
(`BATCHLAS_GEMV_SEGT` still steers W). `can_run` adds the agreement checks the deleted
`gemv_op_shape` returned `nullopt` on (A homogeneous, x and y batch equal to A's, `x.size() == red`,
`y.size() == out`); cta also needs `ops::gemv::device_allows` (`choice.hh`, host-callable: GPU,
sub-group 32, `trans != N`); direct has no GPU gate (native_cpu). There is still no gemv validator
(known-defects #1), so a non-conforming call goes to the vendor as before. Keys `trans:exact out:log
red:log batch:log`; out/red are y's and x's lengths (they swap with trans), ConjTrans folds to T.
Grid: out 255|256, red 63|64 and 352|353, batch 319|320, 1408 rows. Rankings: `cta|vendor|direct`
on the 48 complex<double> T cells inside the old window, `vendor|cta|direct` on every other T cell,
`vendor|direct` under N. Gate (sm_120): (b) 100.00% in all 32 (device, dtype, scenario) cells, and
removing the red=63 rows drops cdouble to 99.20%; (c) 21/26 vendor cells identical, the other 5
crash inside cuBLAS Zgemv in both binaries (known-defects #13), 26/26 vendor-free.

### Phase 5, geqrf

Families `tiny` (`m == n`, `n <= geqrf_tiny_max_n_for_slm`), `cta` (`m >= n`, `geqrf_cta_fits`,
which adds the padded-launch-hole check the old area test lacked; the two agree for every
`(m, n) <= 512` at the 97,280 B and 45,056 B budgets), `blocked` (`m >= n`, the driver compiled, and
a CTA panel leaf present: `geqrf_cta_max_elems_for_slm >= 1`) and `vendor`. No native family takes
a wide shape. `can_run` is in `src/ops/geqrf/can_run.hh` so the host test `GeqrfCanRunDevice` can
call it with a synthetic `select::Device`; on these boxes it is the only test that can catch a
dropped `is_gpu`, `has_sg32` or CTA-leaf clause. Keys `form:exact n:log:3 aspect:log` (`form` is
sq/tall/wide, `aspect = max/min` by integer division; work ~ n^3 aspect). The old tall-panel clause
`m >= 128 && n >= 32 && m >= A*n` equals `n >= 32 && aspect >= A` for `A >= 4`. Grid points straddle
the tiny windows, the order floors (64/76/48/256), the tall clause and the CTA/blocked crossover
(float 96, double 48); 636 rows. Deviations: `geqrf_buffer_size` is the chosen family's only, so the
three callers that size once at a bounding panel and factor smaller sub-views (`band_reduction.cc`
twice, `sytrd_sy2sb.cc`) call the internal `geqrf_buffer_size_bound` (`src/ops/geqrf/geqrf.hh`, the
maximum over every family this device can run; guard `BoundCoversEverySubViewChoice`); the
trailing-gemm seam is the public gemm (mandatory since the phase-5 rip); the tiny driver has no `max_wg` check, so
`can_run(tiny)` has none either. Gate: (b) 100.000% in all 16 cells at the real capacities, a 48 KiB
budget and small synthetic capacities; removing four threshold points drops it to ~99% and exits 1;
(c) 24/24 vendor, 23/23 vendor-free (the wide cell throws `NoRouteError` in both).

### Phase 5, gesv

Families `tiny` (`gesv_tiny_dispatch`, fused LU factor + solve) and `blocked` (public `getrf` then
public `getrs`, each child choosing its own kernel); no vendor arm on any backend, so a `vendor` pin
warns once and runs Auto, as posv. `can_run(tiny)`: `B != NETLIB`, GPU, sub-group 32, homogeneous A
and B, `n <= gesv_tiny_max_n<T>()` (32, cdouble 16), `nrhs <= kGesvTinyMaxRhs` (4) and
`max_wg >= kGesvTinyWgSize` (64, new, `static_assert`-ed against the kernel; the old `supports()`
lacked the driver's work-group check). Heterogeneous and empty problems throw `internal_error`
before `choose()`. Blocked's workspace is `getrf_buffer_size + getrs_buffer_size` (the launch cuts the
span at getrf's size). Keys `n:log:3 nrhs:log`; float is `tiny|blocked` for n <= 32, cfloat for
n <= 16, `blocked` elsewhere. Gate: (b) 2500/2500 in every dtype on both devices; (c) 21/21 through
`factor_bench gesv --arms=native`.

### Phase 5, gesvd

Families `jacobi` (`gesvdj_cta`), `cta` (`gesvd_cta`), `blocked` (`gesvd_blocked`) and `vendor`. The
jobs are canonicalised once (Thin -> All where they coincide) and `can_run`, `key_of`, workspace and
launch see the same jobs. Native term `is_gpu && m, n, batch >= 1`; jacobi adds `has_sg32 && !herm
&& max(m,n) <= gesvd_jacobi_max_dim<T>(vectors)` (64; cdouble with vectors 32); cta adds
`has_sg32 && max(m,n) <= kGesvdCtaMaxDim (32) && !thin && (herm ? m == n : real T)`; blocked adds
`herm ? m == n && Lower : real T`. Two clauses are deliberately not R3-exact and
`CanRunEqualsLaunch` documents both: blocked Hermitian Upper is refused although the driver mirrors
it (opening it needs its own correctness evidence, see known-defects #14), and vendor is
`has_vendor_solver` alone although cuSOLVER `gesvdjBatched` refuses `max(m,n) > 32`, non-packed
batches and thin factors at launch (Auto reaches it only where no native family runs, the old
outcome). The ceilings live in the new sycl-free `src/extensions/gesvd_native.hh`. Keys `herm:exact
vec:exact m:log:1.5 n:log:1.5` (`herm` N/L/U, `vec` none/all/thin), grid 32|33 and 64|65, 2025 rows.
The reached row is recorded at launch only, not in `gesvd_buffer_size`. Gate: (b) 100.00% in every
dtype, vendor and vendor-free, both devices (4000 points); removing the 33 and 65 points drops float
to 96.8%; (c) 25/27 agree, the other two (cfloat 70x70, cdouble 40x40 general) take the vendor in
both and get the identical cuSOLVER refusal. Found: known-defects #14 and #15.

### Phase 5, getrf

Families `tiny` (n <= `getrf_tiny_max_n<T>()`, 32 or cdouble 16, and `max_wg >= kGetrfTinyWgSize`
= 64, new), `cta` (n <= `getrf_cta_max_n_for_slm<T>(budget)`), `blocked` (`getrf_blocked_dispatch`
with the public gemm and trsm injected; needs `getrf_cta_max_n_for_slm<T>(budget, 1) >= 1`, the
panel leaf's own check rather than the old occupancy-scaled one; both are true on every sm_89 and
sm_120 dtype) and `vendor`; every native family needs `B != NETLIB` and a square A. The leaf
(`BATCHLAS_GETRF_LEAF`), laswp mode, blocking factor and tiny bucket stay derived. The trailing
gemm seam is mandatory (it defaulted to `sycl_gemm::gemm_custom`). Keys `n:log:3 batch:log`; the grid
straddles the tiny windows (float 5..32, cfloat 5..7 and 9..24), the cdouble ceiling 16, the double
vendor-free CTA order 32 and the blocked floors (float 256, cfloat 512, or 256 at batch >= 256).
`factor_bench`'s gesv `composed` arm pins getrf `tiny` only where it fits. Gate: (b) 100.00% in
every dtype on both devices, vendor-present and vendor-free at three CTA capacities; removing four
threshold rows drops cfloat to 95.70%; (c) 22/22.

### Phase 5, getri

Families `blocked` (`getri_blocked_dispatch`: P written into C, then two public trsm calls) and
`vendor` (`cublas<t>getriBatched`, rocSOLVER, LAPACKE). `can_run(blocked)`: GPU, sub-group 32,
`B != NETLIB`, the driver compiled, square, n and batch >= 1, homogeneous; the choice is a function
of A alone because `getri_buffer_size` has no C. `can_run(vendor)` is
`factorization_vendor_available<B>` itself (a first version also required rocBLAS, which
`getri_vendor<ROCM>` does not need). Sizing reads metadata only (`SizingReadsMetadataOnly` passes a
null data pointer). Keys `n:log:3 batch:log`: `batch` is a key the old router never read, kept so a
retune can split on it; every batch row holds the same ranking. Grid 127|128 (float) and 255|256
(cfloat); double and cdouble are vendor at every n. Gate: (b) 4000/4000 in every (device, dtype,
vendor) cell; dropping 127 and 255 drops it to ~98%; (c) 22/22.

### Phase 5, getrs

Families `cta` (`getrs_fused_dispatch`: `max_wg >= 32`, `nrhs <= kGetrsFusedMaxRhs` (8),
`n*nrhs <= getrs_fused_max_rhs_elems<T>(slm_budget)`), `blocked` (`getrs_blocked_dispatch` with two
public trsm) and `vendor`. The native gate is `B != NETLIB`, GPU, sub-group 32, a conforming pair (A
square, `B.rows == n`, equal batch), no heterogeneous A or B, and n, nrhs, batch >= 1: exactly the old
`supports()` plus `getrs_op_shape`'s `nullopt` cases, so a non-conforming pair still goes to the
vendor. Keys `n:log:2 nrhs:log batch:log` (`transA` is not a key; no old predicate read it). Grid n
31|32, nrhs 2|3, 4|5, 63|64, 127|128 and 8, batch 127|128; 2160 rows (`vendor cta blocked` 13720
cells, `cta vendor blocked` 1760, `blocked vendor cta` 1800). Gate: (b) 100.000% in all 32
configurations (3300 points each); (c) 22/22 (7 cta, 3 blocked, 12 vendor; cdouble left out because
factor_bench's vendor getrf hits known-defects #13).

### Phase 5, orgqr

Families `blocked` (`orgqr_blocked_dispatch`: the identity fill plus the public `ormqr`, so a native
orgqr is also governed by `BATCHLAS_ORMQR_ROUTE`) and `vendor` (the per-item library loop). Last
resort `vendor, blocked`, by generality: the vendor runs every shape, including `n > m`, CPU queues
and heterogeneous batches. `can_run(blocked)`: GPU, the driver compiled, homogeneous, m, n,
batch >= 1, `n <= m`; the old complex-Trans exclusion is gone because the apply is fixed at
`(Left, NoTrans)`. Keys `m:log n:log:2`; grid 1..8192 on both axes with 512|513 (the only old
threshold, native iff `rows <= 512 && cols <= 512`), restricted to `n <= m`, 153 rows. R5 keeps the
fix for [the orgqr_buffer_size latent defect](../perf/qr.md#the-orgqr_buffer_size-latent-defect).
Gate: (b) 100.00% in all 32 cells for a full and an edge band; deleting the 513 rows drops float to
98.36% / 91.92%; (c) 20/20 (11 blocked, 9 vendor).

### Phase 5, ormqr

Families `blocked` (`ormqr_blocked`: larft plus level-3 WY updates; the WY width stays derived from a
positive `block_size_hint` clamped to [1, k], else `tuning::ormqr_block_size_for_n`) and `vendor`.
`can_run`: blocked `is_gpu`, vendor `has_vendor_solver`; the drivers check their own dimensions.
Keys `side:exact trans:exact m:log k:log q:log batch:log` (`m` the order of Q, `k = min(rows, cols)`,
`q` the extent of C that Q does not act on); the old predicates read only trans and `is_gpu`, so the
log keys move no transcribed decision and the gate is 100% by construction; 540 rows. Behaviour
changes: complex `Transpose::Trans` throws `invalid_argument` from `ormqr` and `ormqr_buffer_size`
before `choose()` (the old router sent it to cuSOLVER, which failed with status 3; no library caller
passes it); the vendor arm spells a real ConjTrans as Trans; the vendor-free build passes the real
vendor availability to `choose` (the old resolver assumed one, docs/perf/qr.md debt 12, resolved);
the out-of-order-queue sequencing of the old `ormqr_dispatch` is kept. Gate: (c) 18/22 agree, the 4
complex-T cells differ as intended. Known gap: known-defects #16.

### Phase 5, spmm

Families `direct` (`spmm_native_csr`) and `vendor` (cuSPARSE, rocSPARSE, netlib); last resort
`vendor, direct`. Derived inside Direct: the body (gather for N, scale + atomic scatter otherwise),
the gather's column block and the complex pair load. `can_run(direct)`: CSR, the body compiled,
`one_spmm()` (the old shape builder's checks), no heterogeneous B or C, batch >= 1, no GPU gate (the
NETLIB native_cpu queue relies on it); the driver has no checks of its own. `can_run(vendor)` is
`d.has_vendor_blas`, which for spmm carries `sparse_vendor_available<B>` (`select::Device` has no
sparse flag), plus three terms the old `supports()` lacked: on CUDA, complex with
`transB == ConjTrans` and nrhs 1 (cuSPARSE returns an unchecked error and leaves C unwritten; the old
Auto routed some of these, which now run `direct`), and cdouble N/N with one column (a host segfault
inside cuSPARSE, reachable only through a pin); on NETLIB, any transpose (netlib's host loop throws
`unsupported`), so NETLIB Auto for transposes moves from a throw onto `direct`. Keys `transA:exact
transB:exact m:log nrhs:log batch:log` (ConjTrans folds to T; no nnz or density key, because the
per-item nnz is in device memory and `spmm_buffer_size` runs the same `choose()`); rows are constant
across the size axes, 500 per table. A third device, `cpu`, is transcribed, because a CPU never
borrows a GPU table and the old choice depended on transA. cuSPARSE's default algorithm is not
bit-reproducible, so the candidate tests identify a pinned vendor by its trace line. Gate: (b)
100.000% on all 12 tables over 10 000 points; (c) 80/80. The CUDA vendor terms and the inherited
misalignment gap are known-defects #13 and #17.

### Phase 5, syev

Families `cta` (`syev_cta`), `cta_fused` (`syev_cta_fused`), `jacobi` (`syev_jacobi_cta`), `blocked`,
`two_stage` and `vendor`, in tie order. The old router had one CTA route and picked its driver
inside (`syev_choose_small_kernel<T>`, from type and `n <= 8`); that was a choice, so the three
drivers are families now, and `cta` names `syev_cta` only. `can_run`: native needs a non-NETLIB
backend, a GPU, a square A and n >= 1; the small three add n <= 32 and `has_sg32`; blocked and
two_stage add batch >= 1; no heterogeneity term (no driver checks it). Keys `jobz:exact n:log:3
batch:log`; uplo is not a key (both large-n drivers mirror Upper into Lower), batch is kept for a
timed table. Grid n 8|9, 24|25, 32|33, 256|257, 320|321, 448|449, 512|513, 1024|1025; 370 rows.
Transcribed pattern (jobz=V): float `jacobi` n <= 8, `cta_fused` 9-32, `blocked` to 448, `two_stage`
to 1024, then `vendor`; double `jacobi`, `blocked` to 448, `vendor`; cfloat `cta_fused` n <= 8, `cta`
to 32, `blocked` to 512, `vendor`; cdouble `cta` to 24, `vendor` 25-32, `blocked` to 256, `vendor`.
jobz=N goes to `two_stage` above 320 in every dtype. Behaviour changes: a vendor-free build no longer
throws where the old Auto preferred the vendor (the old `syev_route` assumed one), so 11 vendor-free
failing names disappeared; non-square A throws `invalid_argument` before `choose()`; ROCm has no
syev table and borrows the sm tables with the R8 warning (untested); `syev_supports_{cta,blocked,
two_stage}`, the Python binding's introspection, are out-of-line wrappers over `can_run`. The knobs
`BATCHLAS_SYEV_SMALL_KERNEL` and `BATCHLAS_SYEV_CTA_MAX_N` and their `Settings` fields were retired
in the rip. Gate: (b) 100.00% on all 8 (device, dtype) pairs for the first runnable choice with and
without the vendor and for the whole ranking (24224 lookups); dropping n = 33 and 449 drops it to
98.08-99.44%; (c) 25/25. Found: known-defects #14 (`OtherTriangleIsNeverRead` carries the skip list).

### Phase 5 fold, tuned tables

(2026-10-05; maintainer: one mega PR, no new tuning or measurement.)
every table that already existed went into `tuned/`, and every op now ships all four dtypes on
sm_89 and sm_120 (spmm also cpu), so neither device borrows. Inventory, per op x dtype x device
(measured / converted / transcribed): `tuned/README.md`; `EveryOpShipsATableForEveryDtypeOnEveryShippedDevice`
in `tuned_tables_tests` holds it.
- **posv sm_120, converted** from `benchmarks/results/routing/sm120_posv_sweep.jsonl` (now committed
  through LFS, with its driver scripts; provenance and resume in the routing README). 3329 cells,
  4 tables (891/827/827/784 rows, 150 `# noisy`), `batchlas=886537e8`. The sweep's resume re-ran
  224 complete pass-2 cells, so 672 (cell, arm, pass) appear twice; the converter's new
  `Source.dedupe_latest` (set for this source only; elsewhere a duplicate stays fatal) keeps the
  later kept row, and a dropped row (a refused re-run) never displaces a measurement: 422
  replacements, the duplicate pairs agreeing to a median ratio of 0.9995. Against the transcription
  sm_120 used to borrow, the first runnable entry changes in 86-239 rows per dtype, mostly `cta` ->
  `blocked` (float 86 of 891 rows, cdouble 239 of 784).
- **trsm sm_120 float/double, tuner tables**: raw in `benchmarks/results/tuning/`, tables
  re-derived byte for byte from it (the only header change is the raw path). The first entry changes
  in 1975 of 4452 float rows (`blocked` -> `vendor` 1007, `cta` -> `sg_left` 876) and 898 of 4098
  double rows: the first time Auto picks `sg_left`. `kernels=c923160f` is reported stale against
  `a33fbfee`, a false alarm (an unused include was removed from `trsm_native.cc`). cfloat/cdouble
  stay transcribed (sweep stopped during cfloat; its partial raw is not used).
- **Transcribed sm_120 tables written from the sm_89 CSVs**: gemm (all four dtypes) and trsm
  cfloat/cdouble. New `--transcribe ... --device sm_120` writes a one-device CSV's rows for another
  device, with `transcriber_device=sm_89` in the header for `--check`; it never overwrites a tuner
  table. Valid because neither transcriber reads a device fact (their device argument only labels
  rows); `tuned_tables_tests` checks the sm_120 rows equal the sm_89 rows.
- Tests: the trsm and gemm `AutoReadsTheSm89TranscribedTable` cases now run on any device whose
  table is the transcription (sm_120 gemm, sm_120 complex trsm) instead of requiring device sm_89;
  three posv cases that hard-coded the sm_89 Auto pick (`LegacyAliasesSelectTheirChoice`,
  `ScopedPinBeatsTheEnvironment`, `CoverageRowCarriesNativeFlags`) now compare against the Auto
  pick the device's table makes. The tie-rule check in `tuned_tables_tests` exempts an entry within
  print rounding of the 3% edge (the trsm and posv tables had 17 rows like `blocked 19.30 | cta
  19.89`, tied by the printed digits but not by the converter's unrounded times).
- `python3 scripts/sweep_to_table.py --check` passes on all 124 files; potrf's tables are byte-identical.

### Phase 5 rip, the old layer removed

(2026-10-05; maintainer: rip out all legacy.)
- `include/batchlas/blas/dispatch/` is gone, with `route_vocabulary_tests` and every `RouteTable`. New homes:
  `batchlas::Op`, `ScalarKind` and `NoRouteError` in the installed, SYCL-free `<batchlas/no_route.hh>` (`Op::iluk`
  dropped, renumbered: an ABI break, pre-1.0); the coverage instrument in `src/select/coverage.{hh,cc}` (`batchlas::coverage`,
  CSV columns unchanged; `OpShape` is `coverage::Shape` without the never-read device fields); the vendor-availability
  constants, `level3_tile_route_available` and `throw_no_vendor_route` in `src/select/vendor.hh` (`batchlas::select`);
  the syev/ormqr `*_vendor_or_throw` shims in `src/ops/{syev,ormqr}/vendor.hh`; `is_sm120_family` next to
  `Device::cuda_compute_capability`; `op_external` inlined at its 19 call sites. `src/dispatch/` is gone too: the
  level-3 entry points are `src/ops/level3/level3.cc`.
- `Settings::routing` is `route(std::string_view op)` over the 19 ops that read `BATCHLAS_<OP>_ROUTE` (throws for any
  other name); `legacy[]`, `legacy_route()`, `canonical[]` and the inert hemm/herk/her2k/iluk slots are gone, as are
  `selection.gemm_sycl_kernel`, `selection.syev_small_kernel` and `geometry.syev_cta_max_n`.
- No aliases: `select::Rules` keeps only `last_resort`; every op's `aliases` array and gemm's `class_aliases` /
  `legacy_aliases` are deleted, and each op's should-throw test lists the removed spellings. `BATCHLAS_<OP>_VARIANT`,
  `BATCHLAS_<OP>_PROVIDER` and `BATCHLAS_GEMM_SYCL_KERNEL` are not read (tests assert that setting them changes nothing).
- The level-3 four parse their own `BATCHLAS_<OP>_ROUTE` words (`src/backends/route_common.hh`, `level3_pin`), throw on
  an unknown word, and record coverage with `record_choice` (`vendor:vendor`, `native:triangular`, ...). The
  deliberately wrong `DiagFullGemm` measurement route is deleted; `native` now takes the tile kernel (it used to fall
  to `DiagFullGemm` for syrk and to an unrequested cuBLASDx throw for syr2k), and a `cublasdx` pin that cannot run
  throws instead of falling back. symm's `cublasdx`-pinned tests pin `expand`, the route they always measured.
- **Transcribers deleted.** `tools/transcribe/` (14 C++ transcribers and their 8 off-grid gate
  scripts) compiled only against a tree that still had the old router. The tables keep their
  provenance in the header: `source=transcribed:424a45bc` (100 tables), `7e71a6e0` (posv sm_89, 4),
  `8b9adeb3` (trsm sm_89, 6). The sources are `git show 0bd26dfe:tools/transcribe/<file>`;
  `tuned/transcribed/*.csv` stay, so `--check` still re-derives every table (`tuned/README.md`).
- **Dead code deleted** (each proven unreferenced: one definition, no caller, no `nm` U reference):
  `backend::gemm_cublasdx()`, `gemm_vendor_cuda_raw()` and their helpers; the cuSolverDx wrapper
  (its kernels were gated on `BATCHLAS_ENABLE_CUSOLVERDX_WRAPPER`, defined nowhere, so it always
  fell back to cuSOLVER) with its two benchmarks; six internal capacity/debug probes
  (`geqrf_cta_max_m`, `geqrf_cta_max_elems`, `geqrf_cta_debug_launch`, `geqrf_tiny_max_n`,
  `getrf_cta_max_n`, `orgqr_blocked_debug_block_size`); unused internal templates; the unbuilt
  `benchmarks/gemm_custom.cc`; and, once `gemm_cublasdx()` was gone, `cublasdx_gemm::launch_float`,
  `variant_supported`, `GemmLaunchDescriptor` and the cuBLASDx GEMM kernel templates.
  `gemm_cublasdx.cu` keeps only `cublasdx_gemm::available()`, which the level-3 fused gate reads;
  it no longer includes `<cublasdx.hpp>`, so it compiles the same with or without MathDx. With
  the cuSolverDx wrapper gone, its build plumbing is gone too: the `BATCHLAS_ENABLE_CUSOLVERDX`
  option, the `cusolverdx.hpp` probe and `mathdx::cusolverdx` link, and the installed
  `BATCHLAS_HAS_CUSOLVERDX` macro in `backend_config.h` (nothing read it). Also deleted: the
  unbuilt `benchmarks/device_blas_level1_benchmark.cc` and the never-called CMake function
  `batchlas_add_device_blas_variant_target`.
- **One seam contract.** The potrf, geqrf and getrf blocked drivers all REQUIRE their injected
  child ops (trailing gemm; potrf's and getrf's panel trsm): an empty `std::function` throws
  `invalid_argument`. The `with_backend` fallbacks in potrf and geqrf were reachable only from
  direct-driver tests (every `src/ops` caller injects the public op) and are deleted; the tests
  pass the public gemm (`gemm_seam()`) and assert the throw (`GeqrfTest.DirectEntryPoints...`,
  `PotrfBlockedTest.BlockedEmptySeamsThrow`).
- **`NoRouteError` stays reachable from the umbrella.** `<batchlas/blas/functions.hh>` (so
  `<batchlas.hh>`) includes `<batchlas/no_route.hh>`, as main's gesvd/ormqr/syev headers did for
  the old path; spelling migration `batchlas::dispatch::NoRouteError` -> `batchlas::NoRouteError`
  (`docs/cpp-api.md`). `examples/consumer` static-asserts it with only the umbrella included.

### After phase 5, one run() per op (2026-10-06)

The fifteen op files had grown the same plumbing around their four real functions (`key_of`,
`can_run`, `launch`, `workspace`): a private `overloaded`, a `choose` wrapper with a try/catch that
turned "nothing runnable" into `NoRouteError`, a `native_facts` wrapper, a device helper, the
`square_shape` + `TraceScope` sequence and a 20-line instantiation block. That now lives once in
`select.hh`:

- **`select::OpSpec`** in each `choice.hh`: the `Op`, the `select::Lib` its Vendor family calls
  (`level3`, `factorization`, `solver`, `sparse`, or `none` for gesv/posv) and its `Rules`, which
  default to `{"blocked", "vendor"}`. gemm, gemv, spmm, getri and orgqr keep their own order.
- **`select::run`** opens the trace/coverage scope around `launch(choice)`; **`select::pick`** is
  the choice alone, for `*_buffer_size`. Both raise `NoRouteError` for a vendor-free miss, so the
  per-op catch is gone. `select::no_vendor<B, T>(spec)` is the Vendor arm's `else`.
- **`Device::has_vendor`** replaces `has_vendor_solver` and `has_vendor_blas`. The two flags had
  carried four library groups (getrf/getrs/geqrf/orgqr/ormqr passed the factorization group in the
  solver slot, spmm the sparse group in the BLAS slot). The phase notes above use the old names.
- **`select::all_of<Choice>()`** is `candidates<T>()` for every field-less op: declaration order is
  the tie-break order (unchanged for all thirteen). Only gemm and potrf list knobs by hand.
- `coverage::Shape` is built with designated initializers; `square_shape` is gone.
  `select::on_in_order_queue` replaces three copies of the in-order wrapper (ormqr, syev, gesvd).
- One behaviour moved: ormqr's insufficient-workspace check now runs inside the trace scope, as
  syev's and gesvd's already did, so that throw leaves a coverage `reached` row.

Routing is unchanged: `scripts/route_diff.sh` captured identical `reached` rows before and after.

## 13. Phase 3 decisions (maintainer, 2026-10-04)

The full plan, with file:line maps, is in `flat-kernel-selection-phase3-plan.md`. These decisions were made after it:
- **Stack.** P3.0 select infrastructure → P3.1 posv → P3.2 tuner core (`tools/tune`, plus a `--gate` mode, pulled forward from phase 4) → P3.2b blackwell kernels → P3.3 trsm → P3.4 gemm. Each PR is based on the one before it.
- **sm_89 tables are transcribed old routing.** For posv, trsm and gemm, today's router is evaluated at every grid cell. Its preference order becomes an untimed ranked row (`tiny - | cta - | blocked -`, header `source=transcribed:<sha>`). This departs from §3 "ranked list with times" until a phase-4 retune on the 4090. On-grid cells are unchanged by construction. The sm_89 gate times only the off-grid cells where the nearest transcribed row disagrees with the old predicate.
- **Blackwell kernels before trsm/gemm.** The kernels from `worktree-blackwell-tuning` (`trsm_sg_left.cc`, 2 wide gemm configs) are ported first, as kernels only, with every `is_sm120_family`/`cuda_cc` predicate dropped. The sweeps then rank them as ordinary candidates.
- **gemm:** delete the 5 experimental variants and the 4 pin-only register variants. Rejected: a screening pass (the full §6.3 protocol is used); vendor before direct in last resort; routing symm/syrk/syr2k/trmm through the public gemm. These are to be revisited when P3.4 starts.
- **§11 posv bullet is wrong.** posv calls the public `potrf`/`trsm`. Its only direct driver calls are its own kernels (`posv_tiny_dispatch`, `potrs_fused_dispatch`). Nothing needs migrating there.
- **Shared box: idle foreign contexts are allowed** (`--allow-idle-foreign`, and benchmarks' posv sweep resume used an equivalent guard); a busy GPU or a new foreign process still refuses.
- **gemm, decided 2026-10-04 (before P3.4):** symm, syrk, syr2k and trmm call the public `gemm<B,T>` instead of `gemm_vendor`, so they get the table-driven choice once the `cublas.cc`/`rocblas.cc` reroute is deleted. gemm's `last_resort` is native first: `direct`, then `vendor` (`can_run` keeps `precision != Default` and the CPU on vendor).
