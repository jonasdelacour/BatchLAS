#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Output-hash grid for the steqr/stedc rewrite (tridiag-modernize DESIGN section 7 and WP2-WP4):
# steqr_accuracy --hash-only --vectors over every cell, each run REPS times, and every
# batch-position cell at batch POS and batch 1. A bit-identical branch reproduces base's CSV.
#
#   scripts/agent/tridiag_hashes.sh <build-dir> <out.csv> cpu|gpu [CUDA|ROCM|SYCL]
#
# cpu: NETLIB backend on the host queue (steqr is steqr_wg, NETLIB stedc is steqr_legacy; no
#      CTA kernel and no D&C merge runs there). gpu: the GPU backend (default CUDA; SYCL is
#      AdaptiveCpp Metal, TYPES=float without device fp64), plus the NETLIB cells.
# Environment: REPS (2), SAMPLES (cpu 64, gpu 512), STEDC_SAMPLES (cpu 16, gpu 128),
#   POS (cpu 64, gpu 4099; the batch-position cells' sample count), COND (3; one log10-cond bin),
#   SEED (20261101; tridiag_inputs.py's seed, another value gets its own _seed<SEED> input dirs),
#   TYPES ("float double"), SECTIONS ("netlib steqr_cta steqr_wg stedc"),
#   INPUTS (<out dir>/tridiag-inputs: tridiag_inputs.py writes hard T there once, shared by trees).
#   Input directories are read-only once written, so a missing chunk fails the run instead of
#   letting share_inputs write the blind generator's matrix. *_wide cells (double only) use
#   tridiag_wide inputs, whose 6e+153/1e-130 classes switch double block scaling on.
# Output: <out.csv> holds rep,cell,batch,impl,n,dtype,items,w_hash,z_hash (one line per cell and
# rep); per-sample CSVs go to <out.csv>.d/. The verdicts (self-check, run-to-run, batch-position, env knob)
# are printed and written to <out.csv>.summary. A selfcheck_* cell reruns with e[0] of item 0
# scaled by 1 + 2^-10 and must change both hashes. Exit 1 on a failed run (a run with no hash_total
# line counts), an empty grid, a blind self-check, a nondeterministic cell or an ignored env knob;
# 2 on usage.
set -euo pipefail

[ "$#" -eq 3 ] || [ "$#" -eq 4 ] || { sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
build=$1 out=$2 mode=$3 gpu_be=${4:-CUDA}
case $mode in cpu|gpu) ;; *) echo "tridiag_hashes: mode must be cpu or gpu" >&2; exit 2 ;; esac
case $gpu_be in CUDA|ROCM|SYCL) ;; *) echo "tridiag_hashes: backend must be CUDA, ROCM or SYCL" >&2; exit 2 ;; esac
bench="$build/benchmarks/steqr_accuracy"
[ -x "$bench" ] || { echo "tridiag_hashes: no $bench (scripts/b -B $build steqr_accuracy)" >&2; exit 2; }

# Non-interactive shells skip ~/.bashrc, so the UR adapters are not found and every cell aborts
# with "No gpu device available" (as gpu_health.sh does, put the toolchain's lib first).
if [ -d /opt/dpcpp-cuda/lib ]; then export LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi

if [ "$mode" = cpu ]; then dflt_s=64 dflt_sd=16 dflt_pos=64; else dflt_s=512 dflt_sd=128 dflt_pos=4099; fi
reps=${REPS:-2} samples=${SAMPLES:-$dflt_s} stedc_samples=${STEDC_SAMPLES:-$dflt_sd} pos=${POS:-$dflt_pos}
cond=${COND:-3} seed=${SEED:-20261101} types=${TYPES:-float double}
in_sfx=; [ "$seed" = 20261101 ] || in_sfx=_seed$seed
sections=" ${SECTIONS:-netlib steqr_cta steqr_wg stedc} "
here=$(cd "$(dirname "$0")" && pwd)
inputs=${INPUTS:-$(dirname "$out")/tridiag-inputs}
want() { [[ $sections == *" $1 "* ]]; }
dir="$out.d"
mkdir -p "$dir"
echo "rep,cell,batch,impl,n,dtype,items,w_hash,z_hash" > "$out"
failed=0 perturb=0 kind=tridiag

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
    local in="$inputs/${kind}_${t}_n${n}_s${s}_b${b}${in_sfx}"
    [ -f "$in/.done" ] || { python3 -I "$here/tridiag_inputs.py" "$in" "$n" "$s" "$b" "$kind" "$t" "$seed" && touch "$in/.done"; }
    chmod a-w "$in"
    if [ "$perturb" = 1 ]; then
        local src=$in
        in="${src}_perturbed"
        [ -f "$in/.done" ] || {
            mkdir -p "$in" && cp "$src"/*.bin "$in"/ && chmod u+w "$in"/0.bin
            python3 -I -c 'import sys, numpy as np
a = np.memmap(sys.argv[1], dtype=np.float32 if sys.argv[2] == "float" else np.float64, mode="r+")
n = int(sys.argv[3]); a[1] *= 1 + 2.0 ** -10; a[n] *= 1 + 2.0 ** -10; a.flush()' "$in/0.bin" "$t" "$n"
            touch "$in/.done"; }
        chmod a-w "$in"
    fi
    env "${envs[@]+"${envs[@]}"}" "$bench" --hash-only --vectors --samples "$s" --batch "$b" \
        --log10-cond-min "$cond" --log10-cond-max "$cond" --seed "$seed" --output "$dir/$tag.csv" \
        --inputs "$in" "$@" \
        > "$dir/$tag.out" 2>&1 || {
        echo "tridiag_hashes: FAILED $tag (see $dir/$tag.out)" >&2; failed=$((failed + 1))
        if grep -q 'No [a-z]* device available' "$dir/$tag.out"; then
            echo "tridiag_hashes: no SYCL device; check ONEAPI_DEVICE_SELECTOR and LD_LIBRARY_PATH" >&2; exit 1
        fi
        return 0; }
    # Exit 0 and no line (a misspelled --impl runs no arm) must not drop the cell from every verdict.
    grep -q '^hash_total,' "$dir/$tag.out" || {
        echo "tridiag_hashes: NO HASH $tag (see $dir/$tag.out)" >&2; failed=$((failed + 1)); return 0; }
    grep '^hash_total,' "$dir/$tag.out" | sed "s/^hash_total,/$rep,$cell,$b,/" >> "$out"
}

# cell REP NAME SAMPLES POSITION(0|1) [env ...] -- [args ...]: at batch SAMPLES, and at batch 1 too
# when POSITION (the batch-position check).
cell() {
    local rep=$1 name=$2 s=$3 position=$4
    shift 4
    run "$rep" "$name" "$s" "$s" "$@"
    if [ "$position" = 1 ]; then run "$rep" "$name" "$s" 1 "$@"; fi
}

# wide CELL-ARGS: cell on tridiag_wide inputs (double: classes 4/5 are 6e+153/1e-130 and scale).
wide() { kind=tridiag_wide; cell "$@"; kind=tridiag; }

# selfcheck REP NAME [env ...] -- [args ...]: 64 items as given, then with item 0's e[0] perturbed.
# A solver that never reads the off-diagonal (a diagonal input, say) leaves the hashes unchanged.
selfcheck() {
    local rep=$1 name=$2
    shift 2
    run "$rep" "selfcheck_$name" 64 64 "$@"
    perturb=1
    run "$rep" "selfcheck_${name}_perturbed" 64 64 "$@"
    perturb=0
}

for rep in $(seq 1 "$reps"); do
    for t in $types; do
        if want netlib; then
            selfcheck "$rep" netlib_steqr -- --backend NETLIB --type "$t" --n 13 --impl steqr
            selfcheck "$rep" netlib_stedc -- --backend NETLIB --type "$t" --n 64 --impl stedc
        fi
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
        if [ "$t" = double ] && want netlib; then
            for n in 13 48; do wide "$rep" netlib_steqr_wg_wide "$samples" 0 -- --backend NETLIB --type double --n "$n" --impl steqr; done
            wide "$rep" netlib_stedc_legacy_wide "$stedc_samples" 0 -- --backend NETLIB --type double --n 64 --impl stedc
        fi
        [ "$mode" = gpu ] || continue
        want steqr_cta && selfcheck "$rep" steqr_cta -- --backend "$gpu_be" --type "$t" --n 13 --impl steqr_cta --scheme exp --cta-shift lapack
        want steqr_wg && selfcheck "$rep" steqr_wg -- --backend "$gpu_be" --type "$t" --n 48 --impl steqr
        want stedc && selfcheck "$rep" stedc -- --backend "$gpu_be" --type "$t" --n 64 --impl stedc --stedc-algorithm levels --stedc-variant fusedcta
        # WP2: steqr_cta, every scheme x shift.
        want steqr_cta && for n in 3 4 5 7 8 12 13 16 24 32; do
            for scheme in exp pg; do
                for shift in lapack wilkinson; do
                    p=0; [ "$scheme/$shift" = exp/lapack ] && p=1
                    cell "$rep" "steqr_cta_${scheme}_${shift}" "$( [ "$p" = 1 ] && echo "$pos" || echo "$samples" )" "$p" -- \
                        --backend "$gpu_be" --type "$t" --n "$n" --impl steqr_cta --scheme "$scheme" --cta-shift "$shift"
                done
            done
        done
        # WP3: steqr_wg on the GPU.
        want steqr_wg && for n in 33 40 48 64; do
            cell "$rep" "steqr_wg" "$pos" 1 -- --backend "$gpu_be" --type "$t" --n "$n" --impl steqr
        done
        # WP4/WP5: stedc drivers x merge variants x threads per root (FusedCta only reads tpr).
        want stedc && for n in 64 100 320 512 640; do
            for algo in levels recursive; do
                cell "$rep" "stedc_${algo}_baseline" "$stedc_samples" 0 -- \
                    --backend "$gpu_be" --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" --stedc-variant baseline
                cell "$rep" "stedc_${algo}_fused" "$stedc_samples" 0 -- \
                    --backend "$gpu_be" --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" --stedc-variant fused
                for tpr in 4 8 16 32; do
                    p=0; [ "$algo/$tpr" = levels/8 ] && case $n in 64|320) p=1 ;; esac
                    cell "$rep" "stedc_${algo}_fusedcta_tpr${tpr}" "$( [ "$p" = 1 ] && echo "$pos" || echo "$stedc_samples" )" "$p" -- \
                        --backend "$gpu_be" --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" \
                        --stedc-variant fusedcta --stedc-tpr "$tpr"
                done
                # tpr > 32 takes stedc_merge_fused_wg (stedc_merge_cta.cc), which no tpr <= 32 cell reaches.
                p=0; [ "$algo" = levels ] && case $n in 64|320) p=1 ;; esac
                cell "$rep" "stedc_${algo}_fusedwg_tpr64" "$( [ "$p" = 1 ] && echo "$pos" || echo "$stedc_samples" )" "$p" -- \
                    --backend "$gpu_be" --type "$t" --n "$n" --impl stedc --stedc-algorithm "$algo" \
                    --stedc-variant fusedcta --stedc-tpr 64
            done
            # The env knob reaches the same field as --stedc-tpr: the summary requires the tpr16 hashes
            # and not the default tpr8 ones.
            cell "$rep" "stedc_levels_fusedcta_envtpr16" "$stedc_samples" 0 BATCHLAS_TUNE_STEDC_THREADS_PER_ROOT=16 -- \
                --backend "$gpu_be" --type "$t" --n "$n" --impl stedc --stedc-algorithm levels --stedc-variant fusedcta
        done
        if [ "$t" = double ]; then
            want steqr_cta && for n in 13 32; do
                wide "$rep" steqr_cta_exp_lapack_wide "$samples" 0 -- \
                    --backend "$gpu_be" --type double --n "$n" --impl steqr_cta --scheme exp --cta-shift lapack
            done
            want steqr_wg && wide "$rep" steqr_wg_wide "$samples" 0 -- --backend "$gpu_be" --type double --n 48 --impl steqr
            want stedc && for n in 64 320; do
                wide "$rep" stedc_levels_fusedcta_wide "$stedc_samples" 0 -- \
                    --backend "$gpu_be" --type double --n "$n" --impl stedc --stedc-algorithm levels --stedc-variant fusedcta
            done
        fi
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
blind, pairs = [], 0
for (k, b), v in by_rep.items():
    p = by_rep.get(((k[0] + "_perturbed",) + k[1:], b)) if k[0].startswith("selfcheck_") else None
    for rep in (v if p else ()):
        pairs += 1
        if rep in p and (v[rep][1] == p[rep][1] or v[rep][2] == p[rep][2]): blind.append((k, rep))
print(f"self_check: {pairs - len(blind)}/{pairs} perturbed-input cells change w_hash and z_hash")
for k, rep in blind: print("  BLIND", *k, "rep", rep)
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
hz = lambda c, k, b: {r: h[1:] for r, h in by_rep.get(((c,) + k[1:], b), {}).items()}
# envtpr16 must equal tpr16, and differ from the default tpr8 where that cell ran at the same batch
# (at n = 64 and 320 tpr8 is a batch-position cell, so its inputs differ).
env_bad, env_n, env_t8 = [], 0, 0
for (k, b), v in by_rep.items():
    if k[0] != "stedc_levels_fusedcta_envtpr16": continue
    env_n += 1
    e, t16, t8 = hz(k[0], k, b), hz("stedc_levels_fusedcta_tpr16", k, b), hz("stedc_levels_fusedcta_tpr8", k, b)
    env_t8 += bool(t8)
    if not t16 or e != t16 or any(e[r] == t8.get(r) for r in e): env_bad.append((k, b))
print(f"env_knob: {env_n - len(env_bad)}/{env_n} envtpr16 cells equal tpr16 ({env_t8} also checked against tpr8)")
for k, b in env_bad: print("  ENV-KNOB-IGNORED", *k, "batch", b)
sys.exit(1 if rr_bad or pos_bad or blind or env_bad or failed or not by_rep or pairs == 0 else 0)
PY
