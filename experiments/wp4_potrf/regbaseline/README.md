# WP4 step 0.3 — register/spill baseline, taken before any potrf kernel exists

Provenance of the files in this directory, and how to reproduce and diff them.
Numbers here are inputs to a later comparison, not conclusions.

Tree: worktree `vendor-independence-plan`, commit `33c3c38`, working tree clean,
`cmake --build build -j 32` reported everything up to date (0.12 s) before probing.
Device: RTX 4090, sm_89, 128 SMs (`occupancy_steps_sm89.txt` line 1).

## Files

| file | what |
|---|---|
| `regprobe_baseline_33c3c38.log` | raw `ptxas -v` output, `libbatchlas_sycl.so` device link (`scripts/register_probe.sh`) |
| `regprobe_baseline_extensions_cta_33c3c38.log` | raw `ptxas -v` output, `libbatchlas_extensions_cta.so` device link (`regprobe_any.sh`) |
| `batchlas_sycl.tsv` | normalized, sorted per-kernel table from the first log |
| `batchlas_extensions_cta.tsv` | same for the second |
| `summarize_ptxas.awk` | the normalizer (see the two parsing traps in its header) |
| `regprobe_any.sh` | `scripts/register_probe.sh` generalized to any of the 14 device-link targets |
| `occupancy_steps.cu`, `occupancy_steps_sm89.txt` | blocks/SM vs registers/thread and vs shared memory, from CUDA's own occupancy model on this device |

`scripts/register_probe.sh` links only `sycl/gemm_kernels.cc.o` and
`sycl/trsm_native.cc.o` (`build/src/CMakeFiles/batchlas_sycl.dir/link.txt`).
Per WP4_POTRF_SPEC_CORRECTIONS.md W12, `potrf_cta.cc` lands in `src/extensions/`,
i.e. in a **different** shared library — hence the second log and `regprobe_any.sh`.

## Baseline

| unit | entry fns | max regs | entries with non-zero stack frame | own-block spill | callee-block spill | link |
|---|---|---|---|---|---|---|
| `batchlas_sycl` | 424 | 226 | 220 | 0 | 0 | 49.65 s / 50.26 s (two runs) |
| `batchlas_extensions_cta` | 732 | 255 | 180 | 0 | 16 | 107.33 s |

## Reproduce and diff

```bash
cd /home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
B=experiments/wp4_potrf/regbaseline

# batchlas_sycl (gemm + trsm)
bash scripts/register_probe.sh /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_sycl.log
awk -f $B/summarize_ptxas.awk /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_sycl.log \
  | sort > /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_sycl.tsv
diff -u $B/batchlas_sycl.tsv /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_sycl.tsv

# whichever extensions library potrf_cta.cc ends up in
bash $B/regprobe_any.sh batchlas_extensions_cta \
  /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_cta.log
awk -f $B/summarize_ptxas.awk /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_cta.log \
  | sort > /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_cta.tsv
diff -u $B/batchlas_extensions_cta.tsv /home/jonaslacour/.claude/jobs/20812aa0/tmp/now_cta.tsv
```

Diff the `.tsv`, never the `.log`: two links of the *identical* tree produce a
1760-line diff of the raw logs (`Compile time = …` alone) and a **0-line** diff of
the normalized table. Both were run; the byte-identical `.tsv` is measured, not assumed.
