# Developer guide {#developer_index}

| Page | What it covers |
| --- | --- |
| @subpage documentation_conventions "Documentation conventions" | How this site is built, where each kind of material goes, the anchor contract, API-doc style. |
| @subpage md_docs_2ci "Continuous integration" | The CI jobs, what each one guards, and the local equivalent. |
| @subpage testing "Running the tests" | Test labels, scoping a run, the known-failures ledger, the test fixture. |
| @subpage benchviz "benchviz" | BatchLAS vs vendor benchmarking and figures. |
| @subpage perf_regression "Performance regression evaluation" | The perf baseline harness. |
| @subpage tuned_tables_readme "Tuned selection tables" | The `tuned/` tables: format, provenance, how each one was produced. |
| @subpage tune_tool_readme "batchlas_tune" | The tuner that measures the selection tables. |
| @subpage tuning_harness "Tuning harness" | Regenerating the `tuning_params.hh` constants. |
| @subpage dev_agent_guide "Agent environment guide" | The terse working rules: toolchain, build, testing, measurement, kernel facts. |

The agent-facing environment guide, @ref dev_agent_guide (imported by `.claude/CLAUDE.md`), holds
the terse rules this site expands on (toolchain, testing policy, measurement
rules, kernel design facts).
