# Agent core rules {#dev_agent_core}

> **Status:** current · the always-loaded digest of @ref dev_agent_guide · introduced 2026-10-25 to cut per-session context.

`.claude/CLAUDE.md` imports this page into every agent session and subagent; the full guide
(`docs/developer/agent-guide.md`, 61 KB) is normative and is read by section on demand. Each
rule below names its section there (§N). Read a section, never the whole guide:

```bash
python3 scripts/agent/doc_section.py agent-guide §10          # one section of the guide
python3 scripts/agent/doc_section.py perf_gemm --depth 2      # outline of any docs page
python3 scripts/agent/doc_section.py --find 'splitk'          # which sections mention a term
```

## Core: what loads where

| Need | Read |
| --- | --- |
| Toolchain install, oneAPI/AdaptiveCpp, LAPACKE from source | §1-§5, `docs/guide/build.md` |
| Known Pitfalls (CUDA-less icpx, `-fp-model=fast`, libgomp, driver mismatch, stuck kernels, Metal/macOS) | "Known Pitfalls" |
| Build performance tables, fast-loop CMake options, new trees, ruled-out build ideas | §7 |
| Routing internals, per-op choices, pins, vendor-free, API traps | §9 |
| Measurement protocol, warm-up, A/B reps, sanity anchors, benchviz | §10 |
| GPU kernel facts (register gate, SLM races, complex multiply) and measured dead ends | §11 |
| The 4090 box, threadripper02, new-machine checklist | §13 |
| A docs/perf page (often 70-200 KB) | `doc_section.py <page> --depth 2`, then the section |

## Core: build and iterate

- The exercised toolchain is a self-built CUDA DPC++ at `/opt/dpcpp-cuda` (CUDA 13.2, sm_89).
  `-- Using SYCL targets: spir64_x86_64` at configure means a silent CPU-only build (TL;DR, §2b).
- New worktree: `scripts/seed_build_tree.sh ~/BatchLAS/build build` before editing (9 s) rather
  than a fresh configure (§6, §7).
- Build through `scripts/b <target>` only: named targets, `-j8` cap, box-wide lock, exit 70 when
  a requested binary is missing. Its full output goes to `build/logs/`; read the log's tail or
  grep it, never the whole log. One build at a time; heavy TUs exceed 14 GB (§7).
- The build is device-link-bound: the `.so` is the link unit. Time a rebuild by appending a line,
  not `touch` (ccache hit). Ninja, lld, PCH, unity builds and `-fno-sycl-rdc` are measured dead (§7).
- `cmake --build --target A B C` can exit 0 and skip links: check `test -x build/tests/<name>` (§7).
- A/B tuning constants with `BATCHLAS_TUNE_*` env overrides; edit `tuning_params.hh` once, for the
  winner (§7, §10).
- A GPU-only tree (`-DBATCHLAS_CPU_TARGET=none`) is never the gate (§7).

## Core: testing

- Never run the full ctest by default. Scope: one case `--gtest_filter` -> `ctest -R '^X(_heavy)?$'`
  -> `-L <component>` -> `-LE 'slow|heavy'` -> `-LE slow`. `scripts/affected_tests.py
  --print-build-cmd --print-ctest-cmd` maps a diff to binaries. Before claiming `X` green, run
  `X_heavy` too (§8).
- Before pushing: `scripts/gate.sh` (`CUDA_VISIBLE_DEVICES=0` on the 4090 box, `--parallel` only on
  threadripper02), plus `scripts/agent/prepush.sh` (the `/prepush` skill) (§6, §14).
- Full-target ctest on the 4090 box needs `ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'`, else CPU
  backends fail with "No kernel named" (§8).
- main is not green: compare failing test *names* against `tests/known-failures.txt` (+ the
  `known-failures-host-<host>.txt` ledger), never counts. Attribute with
  `scripts/agent/attribute_failures.py`, not a hand-built main tree (§8).
- `BATCHLAS_TEST_BACKEND` / `BATCHLAS_TEST_FLOAT_TYPE` skip coverage silently (§8).
- New or rewritten tests follow the 11 "guards that cannot fail" rules in §8: a deliberate break
  per axis with a narrow red set, reachable (pin the route), non-natural `ld`/stride/inc and
  complex data, poison the code accepts, a batch >= 1024 bit-identical case for SLM kernels,
  thresholds straddled both ways, no self-referential reference, graded eigen inputs, launch at
  the advertised ceiling, `EXPECT_VERIFY` only.

## Core: routing and correctness traps

- Selection is flat: `src/ops/<op>/choice.hh` + `can_run()` (correctness only, never speed) +
  `tuned/<op>.<device>.txt` (speed only). `BATCHLAS_<OP>_ROUTE=<spelling>` pins; a bad pin throws;
  `BATCHLAS_SELECT_TRACE=1` prints choices (§9).
- Linked is not reached: verify a route from `BATCHLAS_COVERAGE_OUT` `reached` rows or
  `scripts/route_diff.sh`, not symbols or green counts (§9).
- Read the predicate, not the comment; comments describing windows drift (§9).
- Vendor-free claims need `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` (§9).
- Write `PotrfOptions{}`, never `{}`; size workspaces with `BumpAllocator::measuring()`; never use
  a per-call `Queue`; pivots are 1-based int32 (§9).

## Core: measuring

- Compare and tune only at saturation, batch >= 128; batch 1 does not matter here (§10).
- One measuring process on the box at a time; check `scripts/agent/gpu_health.sh` and
  `nvidia-smi` first (the box is shared) (§10, §13, §14).
- Do not tune or optimise FP64 on a 1:64 GPU (sm_89, sm_120); keep it correct (§10).
- Warm up (`--warmup=3 --min_iters=8`), discard the first pass, alternate A/B order; under 5%
  needs 14-16 reps. Rebuild bench binaries after a table or `can_run` change (§10).

## Core: repository conventions

- Docs are a Doxygen site; no Markdown outside `docs/` except directory READMEs, `.claude/` and
  `.github/`. Cited heading slugs are an interface (`check_evidence_anchors.py`) (§12).
- Throw `batchlas::` types; messages open with the lowercase op name (§12).
- Comment density <= 18% per file, checked per file, not by average (§12).
- Root-based includes `<src/...>`, never quoted (§12).
- Worktree sessions: the Bash guard refuses heredocs mentioning git, computed `for` loops, awk,
  `<(...)`, `cd X; git` and git mixed with other tools. One plain command per call; Edit tool for
  edits; scripts in `$CLAUDE_JOB_DIR/tmp` as a last resort; wait with
  `scripts/agent/wait_for.sh` in the background (§12).

## Core: agent workflow

- Start from current `origin/main` (`git fetch origin`); a session-start checkout can be days
  behind (§14).
- `scripts/agent/gpu_health.sh` before GPU work; on FAIL tell the user, do not continue
  "compile-verified" (§14).
- Never wait blind: `run_in_background`, `wait_for.sh --stall-min`, check in every ~2-3 h (§14).
- Keep the PR the size of the ask; ask first before new helpers, tolerance changes, ledger edits
  or committed campaign data. Paste `pr_scope.py` REVIEW lines before every push (§14).
- Regenerate generated files (`scripts/sweep_to_table.py`, `repin_tune_hashes.py`) (§14).
- Ask before building alternative designs; embed tooling in benchviz (§14).

## Core: subagent and workflow tiers

Every subagent and every workflow `agent()` call runs on a tier chosen for its task. The tiers are
the project agents in `.claude/agents/`; `scripts/agent_tier_guard.py` (PreToolUse hook) blocks
an Agent call or a Workflow script that picks none.

| Tier | Model, effort | For |
| --- | --- | --- |
| `scout` | haiku, low | read-only locating: files, symbols, call sites, log lines, counts |
| `mechanic` | sonnet, medium | a precisely specified edit, regeneration, named build + test |
| `engineer` | opus, high | implementation and debugging that needs judgement (kernels, routing, tests, perf) |
| `verifier` | opus, high | refuting one claim, diff or measurement |
| `architect` | opus, xhigh | design and hardest root causes; rarely |

- Agent tool: `subagent_type: "<tier>"`. Anything else (`general-purpose`, `Explore`, `Plan`)
  must pass both `model` and `effort`; this page is the instruction that authorises setting
  `effort`. `fork`, `claude-code-guide` and `statusline-setup` keep their own definitions.
- Workflow: `agent(prompt, {agentType: '<tier>', ...})`, or both `model` and `effort`. Give
  the cheap stages (scan, list, extract, format) `scout` or `mechanic`, and spend `verifier`
  or `architect` only on judge and verify stages.
- Pick the cheapest tier that can do the task, and escalate one tier on a failed or shallow
  result rather than starting high. A `// tier-guard: off <reason>` line disables the
  workflow check for that script.

## Core: keeping context small

- Read docs by section (`doc_section.py`) and source by range (`Read` offset/limit, `grep -n`
  first). The worst offenders are `docs/perf/gemm.md` (200 KB), `tests/gemm_candidates_tests.cc`
  (180 KB) and the `docs/perf/{lu,qr,potrf}.md` pages (67-86 KB).
- Check plots from their data (CSV, JSON, `benchviz` text output). Screenshot only to judge layout,
  at <= 1000 px wide; a full-page PNG costs as much as a large source file.
- Cap command output: `| tail -n 40`, `grep -c`, `head`; `ps -eo pid,etime,comm | grep ...`, not
  `ps aux`; `journalctl ... | grep -E ... | tail`, never a full journal.
- Hand log trawls, multi-file archaeology and diff reviews to a subagent and keep its conclusion;
  digest workflow output with `scripts/agent/wf_dump.py`, merges with `scripts/agent/conflicts.py`.
- Persist long evidence to a file in `$CLAUDE_JOB_DIR/tmp` and quote its path, not its text.
