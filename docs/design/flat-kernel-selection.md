# Flat kernel selection

Status: **phases 1-2 implemented on branch worktree-flat-select (2026-10-04). §10 gate: passed on
sm_120; the sm_89 live gate (needs the RTX 4090 box) is still open. See §12 "Gate results".**
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
      [&](Vendor)        { return d.has_vendor_solver; },
  }, c);
}

template <Backend B, class T>
PotrfChoice choose(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
  const auto d = select::device_of<B>(q);
  auto ok = [&](const PotrfChoice& c) { return can_run<T>(c, d, A, uplo); };
  return select::choose<PotrfChoice>("potrf", dtype_name<T>(), d, key_of(A, uplo), candidates<T>(), ok);
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
      [&](Vendor)          { backend::potrf_vendor<B, T>(q, A, uplo, ws, info); },
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
  potrf_validate_params(A, uplo);
  const auto c = ops::potrf::choose<B, T>(q, A, uplo);
  auto shape = select::square_shape<B, T>(A.rows(), A.batch_size());   // coverage key: scalar, backend
  shape.uplo = uplo;                                                     // ... and uplo, never inferred
  select::TraceScope trace("potrf", c, shape);   // prints, records coverage, indents children
  ops::potrf::launch<B, T>(q, c, A, uplo, ws, info);
  return q.get_event();
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
  potrf_validate_params(A, uplo);
  return ops::potrf::workspace<T>(ops::potrf::choose<B, T>(q, A, uplo), q, A, uplo);   // R5
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
  bool is_gpu, has_sg32, has_vendor_solver;
  int64_t slm_budget;       // resident::device_slm_budget(LOCAL_MEM_SIZE), as today
  int max_wg;
};
template <Backend B> Device device_of(Queue& q);   // computed once per Queue device and cached
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
  as aliases for `tiny`, `cta`, `lpanel:panel=8` and `blocked` until phase 5. The bare word
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
    if (const Row<Choice>* row = t->nearest(key))            // exact keys must match; log keys: min distance
      for (const auto& [c, ms] : row->ranked)
        if (can_run(c)) return note(t, d, c);                // note(): warn-once if borrowed, trace tag
  }
  for (const Choice& c : last_resort(candidates))            // generality order (§5.5)
    if (can_run(c)) return c;
  throw std::runtime_error(std::string(op) + ": no runnable kernel on " + d.key);
}
```

- **Nearest:** rows must match every `:exact` key. Among those, the row with the smallest
  `Σ w·|log2(row_key / key)|` over the `:log` keys wins, where `<name>:log:<w>` sets the weight `w`
  (1 when omitted; see §12). On a tie, the smaller `n` wins, then the
  smaller `batch`. If no row matches the exact keys, the exact-key filter is dropped for that
  table. This is how Upper calls use a Lower-only table, with `can_run` removing the Lower-only
  choices.

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
batchlas_tune potrf --dtype float,double,cfloat,cdouble --device 1 --out tuned/ \
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
   - the legacy aliases parse;
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

Phases 3–5 are planned but not committed to. The maintainer decides after phase 2.

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
- **`posv` calls the potrf tier drivers directly** (`factorization.cc:871,876`). It keeps working
  through phase 2 and migrates first in phase 3.
- **Stale comments on `main`** to fix when touching these files:
  - `potrf_native.hh:4` ("preferred() is false for every tier");
  - `coverage.cc:195` (LPanel missing);
  - `potrf.hh:51-53` (wrong `options.hh` line numbers);
  - `factorization.cc:727-730` ("both tiers").

## 12. As built (phases 1-2)

Where the code differs from the sketches above, the code wins. These are the differences.

**Phase 1, `src/select/`:**
- **The class words `native` and `vendor` fall back to Auto with a warning** (once per op and
  word) when nothing in their class can run the shape. They do not throw. These are the two
  exceptions to R6. Bare `native` keeps today's meaning, which the `route-native` re-run of
  `potrf_tests` relies on. Bare `vendor` joined it after the final review: in a vendor-free build
  a `vendor` pin threw, so `factor_bench`'s default `vendor,native` arms aborted the process,
  where the old router had fallen through to automatic. Concrete spellings and aliases still throw
  `std::invalid_argument` (an alias that maps to `vendor` is concrete).
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
  last-resort order are op data, not helper code.
- Zero-field families use `NoFields<"name">`, as §4.2 allows.

**Phase 2, potrf:**
- No `struct Key` in `choice.hh`; `key_of` in `potrf.cc` builds a `select::Key`. `key_names`
  stays in `choice.hh`.
- `aliases`, `last_resort` and `rules` live in `choice.hh`, so tests and `factor_bench` use the
  library's list.
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

## 13. Phase 3 decisions (maintainer, 2026-10-04)

The full plan, with file:line maps, is in `flat-kernel-selection-phase3-plan.md`. These decisions were made after it:
- **Stack.** P3.0 select infrastructure → P3.1 posv → P3.2 tuner core (`tools/tune`, plus a `--gate` mode, pulled forward from phase 4) → P3.2b blackwell kernels → P3.3 trsm → P3.4 gemm. Each PR is based on the one before it.
- **sm_89 tables are transcribed old routing.** For posv, trsm and gemm, today's router is evaluated at every grid cell. Its preference order becomes an untimed ranked row (`tiny - | cta - | blocked -`, header `source=transcribed:<sha>`). This departs from §3 "ranked list with times" until a phase-4 retune on the 4090. On-grid cells are unchanged by construction. The sm_89 gate times only the off-grid cells where the nearest transcribed row disagrees with the old predicate.
- **Blackwell kernels before trsm/gemm.** The kernels from `worktree-blackwell-tuning` (`trsm_sg_left.cc`, 2 wide gemm configs) are ported first, as kernels only, with every `is_sm120_family`/`cuda_cc` predicate dropped. The sweeps then rank them as ordinary candidates.
- **gemm:** delete the 5 experimental variants and the 4 pin-only register variants. Rejected: a screening pass (the full §6.3 protocol is used); vendor before direct in last resort; routing symm/syrk/syr2k/trmm through the public gemm. These are to be revisited when P3.4 starts.
- **§11 posv bullet is wrong.** posv calls the public `potrf`/`trsm`. Its only direct driver calls are its own kernels (`posv_tiny_dispatch`, `potrs_fused_dispatch`). Nothing needs migrating there.

