#!/usr/bin/env bash
# D1's STEP 0, settled by counter rather than by model: is the interchange
# WALK's DRAM unit the 32 B sector or the 128 B L1 line? The two answers put the
# right-hand gather's crossover 3.5x apart (float 288 vs 1024), and that constant
# is the one number that can silently invert a window.
#
#   l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum  -- sectors the L1 asked
#   dram__bytes.sum                                      -- what left DRAM
#
# TWO HAZARDS, both paid for once here:
#   * --kernel-name matches the MANGLED name unless --kernel-name-base demangled
#     is given; without it ncu profiles the run and emits NO rows, which looks
#     exactly like a kernel that does no memory traffic (wp3_s12/profile.sh).
#   * ncu REFUSES the vendor-free binaries on this box ("Cuda is initialized
#     before the tool"), so the profiled binary is getrfab_v -- the SAME source
#     linked against build/ -- with BATCHLAS_GETRF_ROUTE pinned to native:blocked.
#     The pin is what makes the vendor-present build reach the driver at all.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export WARM_S=0.0
export BATCHLAS_GETRF_ROUTE=native:blocked
NCU=/usr/local/cuda-13.2/bin/ncu
T="${T:-float}"; N="${N:-512}"; B="${B:-8}"; LC="${LC:-6}"
M="dram__bytes.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,gpu__time_duration.sum"

for arm in inloop defer_gather; do
  BATCHLAS_GETRF_LASWP=$arm "$NCU" --kernel-name-base demangled \
      --kernel-name "regex:LuLaswp" --launch-count "$LC" \
      --metrics "$M" --csv --page raw \
      "$D/getrfab_v" "$T" "$N" "$B" 1 "$arm" "$arm" 2>/dev/null \
    | python3 "$D/ncu_summarise.py" "$arm" "$T" "$N" "$B"
done
