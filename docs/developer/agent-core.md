# Agent core rules {#dev_agent_core}

> **Status:** current · the always-loaded digest of @ref dev_agent_guide, imported by `.claude/CLAUDE.md` into every session and subagent.

The full guide (`docs/developer/agent-guide.md`) is normative; each rule names its section (§N).
Read one section, never the whole guide or a whole docs page (several are 40-60 KB):

```bash
python3 scripts/agent/doc_section.py agent-guide §10          # one section of the guide
python3 scripts/agent/doc_section.py perf_gemm --depth 2      # outline of any docs page
python3 scripts/agent/doc_section.py --find 'splitk'          # which sections mention a term
```

Map: §1-§5 toolchain (CMake options: `docs/guide/build.md`); "Known Pitfalls" toolchain and Metal
traps; §7 build; §8 testing; §9 routing, API traps; §10 measuring; §11 GPU kernel facts, every
ruled-out idea; §12 conventions; §13 machines (`scripts/agent/dispatch.sh hosts`); §14 workflow.

## Core: build and iterate

- Toolchain: self-built CUDA DPC++ at `/opt/dpcpp-cuda` (CUDA 13.2, sm_89). `-- Using SYCL
  targets: spir64_x86_64` at configure means a silent CPU-only build (TL;DR, §2b).
- New worktree: `scripts/seed_build_tree.sh ~/BatchLAS/build build` before editing (§6, §7).
- Build only via `scripts/b <target>` (`-j8` cap, memory admission, exit 70 when a requested binary
  is missing); read the tail of its log in `build/logs/`. A plain `cmake --build --target A B C`
  can exit 0 and skip links (§7).
- The `.so` is the device-link unit; time a rebuild by appending a line, not `touch` (§7).
- A/B tuning constants with `BATCHLAS_TUNE_*`; edit `tuning_params.hh` once, for the winner (§10).
- A GPU-only tree (`-DBATCHLAS_CPU_TARGET=none`) is never the gate (§7).

## Core: testing and measuring

- Never run the full ctest by default: `--gtest_filter` -> `ctest -R '^X(_heavy|_slow)?$'` -> `-L
  <component>` -> `-LE 'slow|heavy'` -> `-LE slow`; `scripts/affected_tests.py --print-build-cmd
  --print-ctest-cmd` maps a diff to binaries. Run `X_heavy`/`X_slow` before calling `X` green (§8).
- Before pushing: `scripts/gate.sh` (full and vf, one idle GPU each) and `scripts/agent/prepush.sh`.
  Full-target ctest on the 4090 box needs `ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'` (§6, §8).
- Fast tests (§8 "Writing fast tests"): observe in-process, fill once, no host loops over USM.
- main is not green: compare failing *names* with `tests/known-failures*.txt`; attribute with
  `scripts/agent/attribute_failures.py`. `BATCHLAS_TEST_BACKEND`/`_FLOAT_TYPE` skip silently (§8).
- New tests follow the 11 "guards that cannot fail" (§8): a narrow deliberate break per axis, a
  pinned route, non-natural `ld`/stride/inc and complex data, accepted poison, batch >= 1024
  bit-identical for SLM kernels, thresholds straddled, no self-referential reference, graded eigen
  inputs, a launch at the ceiling, `EXPECT_VERIFY` only.
- Measure only at saturation, batch >= 128, one measuring process per box (`gpu_health.sh`,
  `nvidia-smi` first); no FP64 tuning on 1:64 GPUs. `--warmup=3 --min_iters=8`, discard the first
  pass, alternate A/B order; under 5% needs 14-16 reps; rebuild bench binaries after a table or
  `can_run` change (§10).

## Core: routing and correctness traps

- Selection is flat: `src/ops/<op>/choice.hh` + `can_run()` (correctness only) +
  `tuned/<op>.<device>.txt` (speed only). `BATCHLAS_<OP>_ROUTE=<spelling>` pins, a bad pin throws;
  `BATCHLAS_SELECT_TRACE=1` prints choices (§9).
- Linked is not reached: verify a route from `BATCHLAS_COVERAGE_OUT` `reached` rows or
  `scripts/route_diff.sh`. Read the predicate, not the comment. Vendor-free claims need
  `-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF` (§9).
- `PotrfOptions{}`, never `{}`; size workspaces with `BumpAllocator::measuring()`; no per-call
  `Queue`; pivots are 1-based int32 (§9).

## Core: conventions and workflow

- No Markdown outside `docs/` except directory READMEs, `.claude/` and `.github/`; cited heading
  slugs are an interface (`check_evidence_anchors.py`). Throw `batchlas::` types, messages open with
  the lowercase op name; root-based `<src/...>` includes; comment density <= 18% per file (§12).
- Worktree Bash guard: no heredocs mentioning git, computed `for` loops, awk, `<(...)`,
  `cd X; git`, git mixed with other tools. One plain command per call, the Edit tool for edits,
  `scripts/agent/wait_for.sh` in the background to wait (§12).
- Start from `origin/main` (`git fetch origin`); `gpu_health.sh` before GPU work, and on FAIL tell
  the user; never wait blind, check in every ~2-3 h (§14).
- Keep the PR the size of the ask: ask before new helpers, tolerance changes, ledger edits,
  committed campaign data or alternative designs; paste `pr_scope.py` REVIEW lines before every
  push. Regenerate generated files (`sweep_to_table.py`, `repin_tune_hashes.py`) (§14).

## Core: subagent and workflow tiers

Every subagent and workflow `agent()` call runs on a tier (`.claude/agents/`); the PreToolUse hook
`scripts/agent_tier_guard.py` blocks an Agent call or Workflow script that picks none.

| Tier | Model, effort | For |
| --- | --- | --- |
| `scout` | haiku, low | read-only locating: files, symbols, call sites, log lines, counts |
| `mechanic` | sonnet, medium | a precisely specified edit, regeneration, named build + test |
| `engineer` | opus, high | implementation and debugging that needs judgement |
| `verifier` | opus, high | refuting one claim, diff or measurement |
| `architect` | opus, xhigh | design and hardest root causes; rarely |

- Agent tool: `subagent_type: "<tier>"`. Any other type must pass both `model` and `effort`; this
  page authorises setting `effort`. `fork`, `claude-code-guide`, `statusline-setup` keep their own.
- Workflow: `agent(prompt, {agentType: '<tier>', ...})`, or both `model` and `effort`. Cheap stages
  get `scout`/`mechanic`; `verifier`/`architect` only judge and verify.
- Pick the cheapest tier that can do the task; escalate one tier on a failed or shallow result.
  `// tier-guard: off <reason>` disables the workflow check for that script.

## Core: keeping context small

- Source by range (`grep -n`, then `Read` offset/limit; `tests/gemm_candidates_tests.cc` is
  ~180 KB). Check plots from their data (screenshots <= 1000
  px). Cap output (`| tail -n 40`, `grep -c`, `ps -eo pid,etime,comm`); never a full journal or log.
- Hand log trawls and diff reviews to a subagent; digest with `scripts/agent/wf_dump.py` and
  `conflicts.py`. Persist long evidence to `$CLAUDE_JOB_DIR/tmp` and quote the path.
