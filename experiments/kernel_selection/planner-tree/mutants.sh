#!/bin/sh
# Deliberate breaks of the PlanTree planner, one per axis (AGENTS.md s8). Each mutant
# recompiles ONLY tests/potrf_planner_tests.cc with -DBATCHLAS_PLAN_MUTANT=<k> (or the
# missing-Exec mock) against the already-built library, then runs the equivalence, order and
# plan==launch tests. The planner headers' ABI tag (plan.hh) gives the mutant TU its own
# symbols, so the LIBRARY column (mismatches=) must stay 0 while tu-mismatches= goes red.
#   1 window edge 256 -> 255      2 cost-model margin ignored
#   3 Tiers order LPanel<->Blocked 4 LPanel legality drops the Upper gate
#   5 CtaWg moved before Cta (shared Route: must change NOTHING)
#   6 CtaWg made selectable (the candidacy net removed: TierOrder test must go red)
#   7 LPanel Exec ignores the plan's NB   8 Cta Exec launches a re-derived scope/geometry
#   drop-exec: Exec<B, CtaWg> removed -> must FAIL TO COMPILE
# usage: mutants.sh <repo> <outdir> [mutant ...]   (GPU picked by CUDA_VISIBLE_DEVICES)
set -u
R="$1"; O="$2"; shift 2
B="$R/build"
mkdir -p "$O"
CXX=/opt/dpcpp-cuda/bin/clang++
CP=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2
FLAGS="-isystem /opt/include -O1 -DNDEBUG -std=c++20 -DGTEST_HAS_PTHREAD=1 -fsycl -Wno-c++20-extensions \
 -Wno-option-ignored -fsycl-unnamed-lambda --cuda-path=$CP -Xsycl-target-backend=nvptx64-nvidia-cuda \
 --cuda-gpu-arch=sm_120 -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -I$R/include -I$B/include \
 "
S="$B/src"
LIBS="$S/libbatchlas_core.so.0.1.0 $S/libbatchlas_backends.so.0.1.0 $S/libbatchlas_extensions_eigen.so.0.1.0 \
 $S/libbatchlas_extensions_factorization.so.0.1.0 $S/libbatchlas_extensions_symmetric.so.0.1.0 \
 $S/libbatchlas_extensions_tridiag.so.0.1.0 $S/libbatchlas_extensions_sytrd.so.0.1.0 \
 $S/libbatchlas_extensions_latrd.so.0.1.0 $S/libbatchlas_extensions_stedc.so.0.1.0 \
 $S/libbatchlas_extensions_cta.so.0.1.0 $S/libbatchlas_util.so.0.1.0 $S/libbatchlas_extra.so.0.1.0 \
 $S/libbatchlas_sycl.so.0.1.0 $S/libbatchlas_backends_cuda.so.0.1.0"
export LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:${LD_LIBRARY_PATH:-}
for m in "$@"; do
  if [ "$m" = drop-exec ]; then D="-DBATCHLAS_PLAN_MOCK_DROP_CTA_WG_EXEC"; else D="-DBATCHLAS_PLAN_MUTANT=$m"; fi
  $CXX $FLAGS "-DBATCHLAS_ROUTING_RESULTS_DIR=\"$R/benchmarks/results/routing\"" $D -c "$R/tests/potrf_planner_tests.cc" -o "$O/m_$m.o" > "$O/m_$m.compile.log" 2>&1
  rc=$?
  echo "mutant $m: compile exit $rc"
  if [ $rc -ne 0 ]; then grep -m3 "error" "$O/m_$m.compile.log"; continue; fi
  $CXX -O1 -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 --cuda-path=$CP \
    -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_120 "$O/m_$m.o" -o "$O/m_$m" \
    -Wl,-rpath,$S /usr/lib/x86_64-linux-gnu/libgtest.a /usr/lib/x86_64-linux-gnu/libgtest_main.a \
    /usr/lib/x86_64-linux-gnu/libgtest.a $LIBS > "$O/m_$m.link.log" 2>&1 || { echo "link failed"; continue; }
  "$O/m_$m" --gtest_filter='PlannerTest.Equivalence*:PlannerTest.TierOrder*:PlannerTest.PlanEqualsLaunch' > "$O/m_$m.run.log" 2>&1
  echo "mutant $m: run exit $?"
  grep "^\[equiv\]" "$O/m_$m.run.log" | grep -v "mismatches=0 tu-mismatches=0 " | cut -c1-120
  grep "^\[order\]\|FAILED  \]" "$O/m_$m.run.log" | sort -u | cut -c1-120
done
