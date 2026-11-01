#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Output-hash grid for the steqr/stedc rewrite (tridiag-modernize DESIGN section 7 and WP2-WP4):
# steqr_accuracy --hash-only --vectors over every cell, each run REPS times, and every
# batch-position cell at batch POS and batch 1. A bit-identical branch reproduces base's CSV.
#
#   scripts/agent/tridiag_hashes.sh <build-dir> <out.csv> cpu|gpu
#
# cpu: NETLIB backend on the host queue (steqr is steqr_wg, NETLIB stedc is steqr_legacy; no
#      CTA kernel and no D&C merge runs there). gpu: CUDA backend, plus the NETLIB cells.
# Environment: REPS (2), SAMPLES (cpu 64, gpu 512), STEDC_SAMPLES (cpu 16, gpu 128),
#   POS (cpu 64, gpu 4099; the batch-position cells' sample count), COND (3; one log10-cond bin),
#   SEED (1234), TYPES ("float double"), SECTIONS ("netlib steqr_cta steqr_wg stedc"),
#   INPUTS (<out dir>/tridiag-inputs: tridiag_inputs.py writes hard T there once, shared by trees).
# Output: <out.csv> holds rep,cell,batch,impl,n,dtype,items,w_hash,z_hash (one line per cell and
# rep); per-sample CSVs go to <out.csv>.d/. The verdicts (run-to-run, batch-position) are
# printed and written to <out.csv>.summary. Exit 1 on a failed run, an empty grid or a
# nondeterministic cell; 2 on usage.
set -euo pipefail

[ "$#" -eq 3 ] || { sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
build=$1 out=$2 mode=$3
case $mode in cpu|gpu) ;; *) echo "tridiag_hashes: mode must be cpu or gpu" >&2; exit 2 ;; esac
bench="$build/benchmarks/steqr_accuracy"
[ -x "$bench" ] || { echo "tridiag_hashes: no $bench (scripts/b -B $build steqr_accuracy)" >&2; exit 2; }

# Non-interactive shells skip ~/.bashrc, so the UR adapters are not found and every cell aborts
# with "No gpu device available" (as gpu_health.sh does, put the toolchain's lib first).
if [ -d /opt/dpcpp-cuda/lib ]; then export LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi

if [ "$mode" = cpu ]; then dflt_s=64 dflt_sd=16 dflt_pos=64; else dflt_s=512 dflt_sd=128 dflt_pos=4099; fi
reps=${REPS:-2} samples=${SAMPLES:-$dflt_s} stedc_samples=${STEDC_SAMPLES:-$dflt_sd} pos=${POS:-$dflt_pos}
cond=${COND:-3} seed=${SEED:-1234} types=${TYPES:-float double}
sections=" ${SECTIONS:-netlib steqr_cta steqr_wg stedc} "
here=$(cd "$(dirname "$0")" && pwd)
inputs=${INPUTS:-$(dirname "$out")/tridiag-inputs}
want() { [[ $sections == *" $1 "* ]]; }
dir="$out.d"
mkdir -p "$dir"
echo "rep,cell,batch,impl,n,dtype,items,w_hash,z_hash" > "$out"
failed=0

# run REP CELL SAMPLES BATCH [env VAR=VALUE ...] -- [steqr_accuracy args ...]
run() {
    local rep=$1 cell=$2 s=$3 b=$4
    shift 4
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local tag n="" t="" prev="" a
    tag=$(printf '%s' "r${rep}_${cell}_b${b}_${envs[*]+${envs[*]}}_$*" | tr -c 'A-Za-z0-9_.=-' '_')
    for a in "$@"; do
        case $prev in --n) n=$a ;; --type) t=$a ;; esac
        prev=$a
    done
    # Written before the run: --inputs falls back to writing the generated matrix when a chunk is missing.
    local in="$inputs/tridiag_${t}_n${n}_s${s}_b${b}"
    [ -f "$in/.done" ] || { python3 -I "$here/tridiag_inputs.py" "$in" "$n" "$s" "$b" tridiag "$t" && touch "$in/.done"; }
    env "${envs[@]+"${envs[@]}"}" "$bench" --hash-only --vectors --samples "$s" --batch "$b" \
        --log10-cond-min "$cond" --log10-cond-max "$cond" --seed "$seed" --output "$dir/$tag.csv" \
        --inputs "$in" "$@" \
        > "$dir/$tag.out" 2>&1 || {
        echo "tridiag_hashes: FAILED $tag (see $dir/$tag.out)" >&2; failed=$((failed + 1))
        if grep -q 'No [a-z]* device available' "$dir/$tag.out"; then
            echo "tridiag_hashes: no SYCL device; check ONEAPI_DEVICE_SELECTOR and LD_LIBRARY_PATH" >&2; exit 1
        fi
        return 0; }
    { grep '^hash_total,' "$dir/$tag.out" || true; } | sed "s/^hash_total,/$rep,$cell,$b,/" >> "$out"
}

# cell REP NAME SAMPLES POSITION(0|1) [env ...] -- [args ...]: at batch SAMPLES, and at batch 1 too
# when POSITION (the batch-position check).
cell() {
    local rep=$1 name=$2 s=$3 position=$4
    shift 4
    run "$rep" "$name" "$s" "$s" "$@"
    if [ "$position" = 1 ]; then run "$rep" "$name" "$s" 1 "$@"; fi
}

for rep in $(seq 1 "$reps"); do
    for t in $types; do
        # steqr_wg on the host queue: every steqr n (the host never takes steqr_cta).
        want netlib && for n in 3 4 5 7 8 12 13 16 24 32 33 40 48 64; do
            p=0; case $n in 4|13|32|33|48) p=1 ;; esac
            cell "$rep" "netlib_steqr_wg" "$( [ "$p" = 1 ] && echo "$pos" || echo "$samples" )" "$p" -- \
                --backend NETLIB --type "$t" --n "$n" --impl steqr
        done
        # NETLIB stedc is steqr_legacy (WP3's fold).
        want netlib && for n in 32 64 100 320; do
            p=0; case $n in 64|320) p=1 ;; esac
            cell "$rep" "netlib_stedc_legacy" "$( [ "$p" = 1 ] && echo "$pos" || echo "$stedc_samples" )" "$p" -- \
                --backend NETLIB --type "$t" --n "$n" --impl stedc
        done
        [ "$mode" = gpu ] || continue
        # WP2: steqr_cta, every scheme x shift.
        want steqr_cta && for n in 3 4 5 7 8 12 13 16 24 32; do
            for scheme in exp pg; do
                for shift in lapack wilkinson; do
                    p=0; [ "$scheme/$shift" = exp/lapack ] && p=1
                    cell "$rep" "steqr_cta_${scheme}_${shift}" "$( [ "$p" = 1 ] && echo "$pos" || echo "$samples" )" "$p" -- \
                        --backend CUDA --type "$t" --n "$n" --impl steqr_cta --scheme "$scheme" --cta-shift "$shift"
                done
            done
        done
        # WP3: steqr_wg on the GPU.
        want steqr_wg && for n in 33 40 48 64; do
            cell "$rep" "steqr_wg" "$pos" 1 -- --backend CUDA --type "$t" --n "$n" --impl steqr
        done
        # WP4/WP5: stedc drivers x merge variants x threads per root (FusedCta only reads tpr).
        want stedc && for n in 64 100 320 512 640; do
            for algo in levels recursive; do
                cell "$rep" "stedc_${algo}_baseline" "$stedc_samples" 0 -- \
                    --backend CUDA --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" --stedc-variant baseline
                cell "$rep" "stedc_${algo}_fused" "$stedc_samples" 0 -- \
                    --backend CUDA --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" --stedc-variant fused
                for tpr in 4 8 16 32; do
                    p=0; [ "$algo/$tpr" = levels/8 ] && case $n in 64|320) p=1 ;; esac
                    cell "$rep" "stedc_${algo}_fusedcta_tpr${tpr}" "$( [ "$p" = 1 ] && echo "$pos" || echo "$stedc_samples" )" "$p" -- \
                        --backend CUDA --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" \
                        --stedc-variant fusedcta --stedc-tpr "$tpr"
                done
            done
            # The env knob reaches the same field as --stedc-tpr: equal hashes prove it is read.
            cell "$rep" "stedc_levels_fusedcta_envtpr16" "$stedc_samples" 0 BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT=16 -- \
                --backend CUDA --type "$t" --n "$n" --impl stedc --stedc-algorithm levels --stedc-variant fusedcta
        done
    done
done

python3 -I - "$out" "$failed" <<'PY' | tee "$out.summary"
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
key = lambda r: (r["cell"], r["impl"], r["n"], r["dtype"])
by_rep = defaultdict(dict)
for r in rows:
    by_rep[(key(r), r["batch"])][r["rep"]] = (r["items"], r["w_hash"], r["z_hash"])
rr_bad = [k for k, v in by_rep.items() if len(set(v.values())) > 1]
failed = int(sys.argv[2])
print(f"failed runs: {failed}")
print(f"run_to_run: {len(by_rep) - len(rr_bad)}/{len(by_rep)} cells equal across reps")
for k in rr_bad: print("  NONDETERMINISTIC", *k)
pos_bad, pos_n = [], 0
for (k, b), v in by_rep.items():
    if b != "1": continue
    for (k2, b2), v2 in by_rep.items():
        if k2 == k and b2 != "1":
            pos_n += 1
            if v["1"][1:] != v2["1"][1:]: pos_bad.append((k, b2))
print(f"batch_position: {pos_n - len(pos_bad)}/{pos_n} cells equal at batch 1 and batch N")
for k, b in pos_bad: print("  POSITION-DEPENDENT", *k, "batch", b)
sys.exit(1 if rr_bad or pos_bad or failed or not by_rep else 0)
PY
