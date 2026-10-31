#!/usr/bin/env bash
# Pre-push gate on persistent trees: configure once, rebuild incrementally, full ctest, ledger diff.
#
#   scripts/gate.sh [--only full|vf] [--serial] [--no-build] [--seed-from TREE [--seed-allow-dirty]]
#                   [--parallel]
#
#   build-gate     full targets, preset dev-tests (the ledger tests/known-failures.txt was
#                  recorded under it); full ctest, slow suites included; gated by
#                  .github/ci/compare_failures.py
#   build-gate-vf  the same plus -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF; ctest -LE slow (the heavy
#                  case tiers included: their vendor-free arms run nowhere else). It has no
#                  ledger (see the nightly CI job), so it gates on build success and reports
#                  failing names, plus the diff against this tree's previous gate run.
#
# Both trees build first, one after the other. Then, with two or more GPUs visible
# (CUDA_VISIBLE_DEVICES, else the idle ones nvidia-smi lists, by the BATCHLAS_GPU_BUSY_* test of
# scripts/ctest_gpus.sh), the two ctests run at once: vf on the first GPU, full on the second,
# each with DPCPP_CPU_NUM_CUS=cores/2 for its opencl:cpu half and its own TMPDIR. With one GPU,
# a busy box (1-minute load above BATCHLAS_GATE_MAX_LOAD, default cores/4, or another process
# above 200% CPU), --serial, --parallel or --only they run one after the other (full on the second GPU, vf on the
# first, when two are visible; both on the one idle GPU). INT/TERM kill both ctests.
#
#   --serial       never run the two ctests at once
#   --no-build     skip configure and build: test the trees as they are
#   --seed-from T  seed a tree that does not exist yet from built tree T (scripts/seed_build_tree.sh),
#                  only when T's BATCHLAS_ENABLE_VENDOR_BLAS matches that tree's
#   --seed-allow-dirty  pass --allow-dirty to the seeding (T's checkout has uncommitted files)
#   --parallel     scripts/ctest_gpus.sh instead of serial ctest (validated on threadripper02 only)
#
# Environment: BATCHLAS_GATE_CUDA (default ON), BATCHLAS_GATE_PRESET (default dev-tests),
# BATCHLAS_GATE_DIR / BATCHLAS_GATE_VF_DIR (default build-gate / build-gate-vf),
# BATCHLAS_GATE_CXX (compiler for a new tree; default /opt/dpcpp-cuda/bin/clang++ when present),
# BATCHLAS_GATE_CPU_CUS (DPCPP_CPU_NUM_CUS per tree when the ctests overlap; default cores/2).
# ctest runs with ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu' when it is unset and sycl-ls lists
# both a native_cpu and an opencl:cpu device: else CPU queues land on native_cpu, which has no
# image in a spir64_x86_64 build ("No kernel named").
# Every configure forces the gate's shape (full CPU+GPU targets, every test, split libraries,
# no offset twins), so a GPU-only or smoke seed cannot turn the gate into a smaller run.
# The box-wide build lock (/tmp/batchlas-build.lock) is held for the whole gate, ctest included:
# a build beside a full ctest is how the 2x4090 box was OOM-killed (docs/ci.md). Each ctest
# records per-entry peak RSS in <tree>/gate/rss.tsv (scripts/gate_rss.py, Linux only).
# After each ctest, CTestCostData.txt is rewritten from its junit times (serial ctest skews it).
# Exit 0 only when every selected tree built and the full tree's ledger diff is clean.
set -eu
set -o pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
only=""
seed_from=""
seed_dirty=()
parallel=0
serial=0
build=1
while [ "$#" -gt 0 ]; do
    case $1 in
        --only) only=$2; shift 2 ;;
        --seed-from) seed_from=$(cd "$2" && pwd); shift 2 ;;
        --seed-allow-dirty) seed_dirty=(--allow-dirty); shift ;;
        --parallel) parallel=1; shift ;;
        --serial) serial=1; shift ;;
        --no-build) build=0; shift ;;
        -h|--help) sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate: unknown argument $1" >&2; exit 64 ;;
    esac
done
case $only in ''|full|vf) ;; *) echo "gate: --only takes full or vf" >&2; exit 64 ;; esac

cd "$root"
full_dir=${BATCHLAS_GATE_DIR:-build-gate}
vf_dir=${BATCHLAS_GATE_VF_DIR:-build-gate-vf}
preset=${BATCHLAS_GATE_PRESET:-dev-tests}
cuda=${BATCHLAS_GATE_CUDA:-ON}
cxx=${BATCHLAS_GATE_CXX:-}
[ -n "$cxx" ] || [ ! -x /opt/dpcpp-cuda/bin/clang++ ] || cxx=/opt/dpcpp-cuda/bin/clang++
shape=(-DBATCHLAS_CPU_TARGET=auto -DBATCHLAS_TEST_TARGET_SET=all -DBATCHLAS_SYCL_GLOBAL_OFFSET=OFF)
[ "$(uname -s)" = Darwin ] || shape+=(-DBATCHLAS_MONOLITHIC_LIBRARY=OFF)
summary_dir=$full_dir; [ "$only" != vf ] || summary_dir=$vf_dir
case $summary_dir in /*) summary="$summary_dir/gate-summary.txt" ;; *) summary="$root/$summary_dir/gate-summary.txt" ;; esac
t_start=$(date +%s)

lock=${BATCHLAS_BUILD_LOCK:-/tmp/batchlas-build.lock}
[ -e "$lock" ] || { : > "$lock" && chmod 666 "$lock"; } 2>/dev/null || true
exec 9<"$lock"
if ! flock -n 9; then
    echo "gate: waiting for $lock, held by: $(cat "$lock.info" 2>/dev/null || echo unknown)" >&2
    flock 9
fi
{ printf 'pid %s user %s since %s running scripts/gate.sh in %s\n' "$$" "$(id -un)" "$(date '+%F %T')" "$root" > "$lock.info"; } 2>/dev/null || true
export BATCHLAS_BUILD_LOCK_HELD=1

# Async children ignore SIGINT and inherit fd 9 (the lock), so Ctrl-C alone would orphan a
# ctest that keeps the lock and the GPU. Each ctest gets its own process group (setsid, else
# perl's setpgrp); the traps kill those groups (else the pid and its children) and the subshells.
children=(); subshells=()
stop_children() {
    local p
    for p in ${subshells[@]+"${subshells[@]}"}; do kill -TERM "$p" 2>/dev/null || true; done
    for p in ${children[@]+"${children[@]}"}; do
        kill -TERM -- "-$p" 2>/dev/null || { pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; } || true
    done
}
trap 'stop_children; exit 130' INT
trap 'stop_children; exit 143' TERM
setsid=()
if command -v setsid > /dev/null; then setsid=(setsid)
elif command -v perl > /dev/null; then setsid=(perl -e 'setpgrp(0, 0); exec @ARGV or die "gate: exec $ARGV[0]: $!\n"'); fi

lines=()
overall=0
note() { lines+=("$*"); echo "gate: $*"; }
elapsed() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }
notes_file=/dev/null
tnote() { printf '%s\n' "$*" >> "$notes_file"; echo "gate: $*"; }

failing_names() {
    python3 -c '
import sys, xml.etree.ElementTree as ET
skip = ("SKIP_RETURN_CODE", "SKIP_REGULAR_EXPRESSION_MATCHED")
for case in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    status = case.get("status", "")
    sk = case.find("skipped")
    msg = sk.get("message", "") if sk is not None else ""
    if status == "fail" or (status == "notrun" and not msg.startswith(skip)):
        print(case.get("name"))
' "$1" | sort -u
}

# A serial ctest (-j 1) never reads CTestCostData.txt, so it writes time/runs: the gate trees'
# costs read 1/2, then 1/3, ... of the real times (ctest 4.2.3 and 4.3.2). After each run, the
# file is rewritten from this run's junit as one run per entry.
reset_cost_data() {
    local junit=$2 cost="$1/Testing/Temporary/CTestCostData.txt"
    [ -f "$cost" ] || return 0
    if [ -f "$junit" ] && python3 -c '
import sys, xml.etree.ElementTree as ET
for case in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    if case.get("status") in ("run", "fail") and case.get("time"):
        print(case.get("name"), 1, case.get("time"))
print("---")
' "$junit" > "$cost.tmp"; then
        mv "$cost.tmp" "$cost"
    else
        rm -f "$cost" "$cost.tmp"
    fi
}

build_tree() {
    local kind=$1 dir=$2 vendor want have rc t0
    local -a extra=()
    if [ "$kind" = vf ]; then vendor=OFF; extra=(-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF); else vendor=ON; fi

    if [ ! -f "$dir/CMakeCache.txt" ] && [ -n "$seed_from" ]; then
        have=$(sed -n 's/^BATCHLAS_ENABLE_VENDOR_BLAS:[A-Z]*=//p' "$seed_from/CMakeCache.txt" | head -n 1)
        want=$vendor
        if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            note "$kind: not seeding: $dir exists without a CMakeCache.txt"
        elif [ "${have:-ON}" = "$want" ]; then
            "$here/seed_build_tree.sh" "${seed_dirty[@]}" "$seed_from" "$dir" || { note "$kind: seeding from $seed_from failed"; return 1; }
            cp "$dir/CMakeCache.txt" "$dir/gate-seed-cache.txt"
            cp "$dir/include/batchlas/backend_config.h" "$dir/gate-seed-config.h" 2>/dev/null || true
        else
            note "$kind: not seeding from $seed_from (BATCHLAS_ENABLE_VENDOR_BLAS=${have:-ON}, need $want)"
        fi
    fi

    [ -f "$dir/CMakeCache.txt" ] || [ -z "$cxx" ] || extra+=("-DCMAKE_CXX_COMPILER=$cxx")
    mkdir -p "$dir/gate"
    if ! cmake --preset "$preset" -B "$dir" -DBATCHLAS_ENABLE_CUDA="$cuda" "${shape[@]}" \
            -DBATCHLAS_LINK_DEPENDS_NO_SHARED=OFF "${extra[@]}" > "$dir/gate/configure.log" 2>&1; then
        note "$kind: CONFIGURE FAILED ($dir/gate/configure.log)"; return 1
    fi
    note "$kind: configured $dir ($(grep -o 'Using SYCL targets:.*' "$dir/gate/configure.log" | head -n 1))"
    if [ -f "$dir/gate-seed-cache.txt" ]; then
        # A seed configured differently (a plain `cmake -B build` has ALLOW_UNSAFE_ENV=OFF,
        # dev-tests ON) regenerates backend_config.h, which ~90% of objects include.
        note "$kind: seed options changed by the gate configure: $(python3 -c '
import re, sys
def cache(p):
    return dict(m.groups() for m in (re.match(r"((?:BATCHLAS|CMAKE_BUILD_TYPE)\w*):[A-Z]+=(.*)", l) for l in open(p)) if m)
a, b = cache(sys.argv[1]), cache(sys.argv[2])
print(" ".join("%s=%s->%s" % (k, a.get(k, "-"), b.get(k, "-")) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)) or "none")
' "$dir/gate-seed-cache.txt" "$dir/CMakeCache.txt")"
        cmp -s "$dir/gate-seed-config.h" "$dir/include/batchlas/backend_config.h" ||
            note "$kind: backend_config.h differs from the seed's: expect a near-full recompile (seed from a tree configured like the gate)"
        rm -f "$dir/gate-seed-cache.txt" "$dir/gate-seed-config.h"
    fi

    t0=$(date +%s); rc=0
    BATCHLAS_FULL_BUILD=1 "$here/b" -B "$dir" > "$dir/gate/build.out" 2>&1 || rc=$?
    note "$kind: build exit $rc in $(elapsed $(( $(date +%s) - t0 ))) ($(grep -o '[^ ]*/logs/[^ )]*' "$dir/gate/build.out" | tail -n 1))"
    [ "$rc" -eq 0 ]
}

# ctest, peak RSS, failing names and the ledger diff of one tree. Notes go to
# <dir>/gate/test-notes.txt, so two trees can run at once in background subshells.
test_tree() {
    local kind=$1 dir=$2 gpu=$3 cus=$4 rc t0 cpid spid=""
    notes_file="$dir/gate/test-notes.txt"
    : > "$notes_file"
    local abs=$dir
    case $dir in /*) ;; *) abs="$root/$dir" ;; esac
    local junit="$abs/gate/ctest-junit.xml" gx="$abs/gate/gtest-xml"
    local -a filter=()
    [ "$kind" = full ] || filter=(-LE slow)
    # Per-tree TMPDIR: death tests reuse fixed temp_directory_path() names across processes.
    local -a envp=(GTEST_OUTPUT="xml:$gx/" TMPDIR="$abs/gate/tmp")
    [ -z "$selector" ] || envp+=(ONEAPI_DEVICE_SELECTOR="$selector")
    [ -z "$gpu" ] || envp+=(CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" CUDA_VISIBLE_DEVICES="$gpu")
    [ -z "$cus" ] || envp+=(DPCPP_CPU_NUM_CUS="$cus")
    local -a runner=(ctest --test-dir "$dir" -j 1)
    [ "$parallel" = 0 ] || runner=("$here/ctest_gpus.sh" --test-dir "$dir")
    local -a timer=()
    rm -f "$dir/gate/ctest-maxrss.txt" "$dir/gate/rss.tsv"
    [ -z "$gnu_time" ] || timer=("$gnu_time" -f 'maxrss_kb %M' -o "$dir/gate/ctest-maxrss.txt")
    rm -rf "$gx" "$junit" "$abs/gate/tmp"; mkdir -p "$gx" "$abs/gate/tmp"
    t0=$(date +%s); rc=0
    ${setsid[@]+"${setsid[@]}"} env "${envp[@]}" ${timer[@]+"${timer[@]}"} "${runner[@]}" \
        --output-junit "$junit" ${filter[@]+"${filter[@]}"} > "$dir/gate/ctest.out" 2>&1 &
    cpid=$!
    children+=("$cpid")
    if [ -d /proc/self ]; then
        python3 "$here/gate_rss.py" --pid "$cpid" --log "$dir/gate/ctest.out" --out "$dir/gate/rss.tsv" &
        spid=$!
        children+=("$spid")
    fi
    wait "$cpid" || rc=$?
    [ -z "$spid" ] || wait "$spid" || true
    children=()
    reset_cost_data "$dir" "$junit"
    tnote "$kind: ctest ${filter[*]:-(full)}${gpu:+ on CUDA_VISIBLE_DEVICES=$gpu}${cus:+ DPCPP_CPU_NUM_CUS=$cus} exit $rc in $(elapsed $(( $(date +%s) - t0 ))): $(grep -E 'tests passed|tests failed' "$dir/gate/ctest.out" | tail -n 1)"
    if [ -f "$dir/gate/rss.tsv" ]; then
        tnote "$kind: $(python3 "$here/gate_rss.py" --summary "$dir/gate/rss.tsv" --exact "$dir/gate/ctest-maxrss.txt") ($dir/gate/rss.tsv)"
    fi
    [ -f "$junit" ] || { tnote "$kind: no junit report written"; return 1; }

    [ ! -f "$dir/gate/failing.txt" ] || mv "$dir/gate/failing.txt" "$dir/gate/failing.prev.txt"
    failing_names "$junit" > "$dir/gate/failing.txt"
    tnote "$kind: $(grep -c . "$dir/gate/failing.txt" || true) failing ctest test(s): $(tr '\n' ' ' < "$dir/gate/failing.txt")"

    local -a cmp=(python3 .github/ci/compare_failures.py --junit "$junit" --gtest-dir "$gx"
                  --build-dir "$dir" --expected-tests-from "$dir")
    if [ "$kind" = full ]; then
        cmp+=(--min-tests 100 --require consumer_package_tests)
    else
        cmp+=(--label-exclude slow)
    fi
    rc=0
    "${cmp[@]}" > "$dir/gate/compare.txt" 2>&1 || rc=$?
    if [ "$kind" = full ]; then
        tnote "full: ledger diff $( [ "$rc" -eq 0 ] && echo CLEAN || echo FAILED ) ($dir/gate/compare.txt)"
        [ "$rc" -eq 0 ] || tail -n 20 "$dir/gate/compare.txt" | sed 's/^/    /' >> "$notes_file"
        return "$rc"
    fi
    tnote "vf: ledger diff exit $rc, informational only (no vendor-free ledger exists)"
    if [ -f "$dir/gate/failing.prev.txt" ]; then
        local new fixed
        new=$(comm -13 "$dir/gate/failing.prev.txt" "$dir/gate/failing.txt" | tr '\n' ' ')
        fixed=$(comm -23 "$dir/gate/failing.prev.txt" "$dir/gate/failing.txt" | tr '\n' ' ')
        tnote "vf: vs previous gate run: new failures [${new}] now passing [${fixed}]"
    fi
    return 0
}

collect_notes() {
    [ ! -f "$1/gate/test-notes.txt" ] || while IFS= read -r l; do lines+=("$l"); done < "$1/gate/test-notes.txt"
}

# Why the box is too busy to overlap the two ctests, else nothing: a 1-minute load average above
# $1, or a process (Linux: a 1 s /proc sample) above 200% CPU. Called between the builds and the
# ctests, when the gate itself runs nothing.
box_busy() {
    python3 - "$1" <<'EOF'
import os, sys, time
max_load = float(sys.argv[1])
load = os.getloadavg()[0]
if load > max_load:
    print(f"1-minute load {load:.2f} > {max_load:g}")
    sys.exit()
if not os.path.isdir("/proc/self"):
    sys.exit()
def ticks():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as f:
                head, rest = f.read().rsplit(")", 1)
            v = rest.split()
            out[pid] = (head.split("(", 1)[1], int(v[11]) + int(v[12]))
        except (OSError, IndexError, ValueError):
            pass
    return out
a = ticks(); time.sleep(1); b = ticks()
hz = os.sysconf("SC_CLK_TCK")
for pid, (comm, t) in b.items():
    pct = 100.0 * (t - a.get(pid, (comm, t))[1]) / hz
    if pct > 200:
        print(f"pid {pid} ({comm}) at {pct:.0f}% CPU")
        break
EOF
}

export LC_ALL=C
selector=""
if [ -z "${ONEAPI_DEVICE_SELECTOR+set}" ]; then
    sycl_ls=$(command -v sycl-ls || true)
    [ -n "$sycl_ls" ] || [ -z "$cxx" ] || [ ! -x "$(dirname "$cxx")/sycl-ls" ] || sycl_ls=$(dirname "$cxx")/sycl-ls
    devices=$([ -z "$sycl_ls" ] || "$sycl_ls" 2>/dev/null || true)
    if grep -q '^\[native_cpu:' <<<"$devices" && grep -q '^\[opencl:cpu\]' <<<"$devices"; then
        selector='cuda:*;opencl:cpu'
        note "ctest runs with ONEAPI_DEVICE_SELECTOR='$selector' (sycl-ls lists native_cpu and opencl:cpu)"
    fi
fi
gnu_time=""
if /usr/bin/time --version 2>&1 | grep GNU > /dev/null; then gnu_time=/usr/bin/time; fi

gpus=(); busy=""; gpu_src="in CUDA_VISIBLE_DEVICES"
if [ "$cuda" = ON ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
        IFS=, read -r -a gpus <<<"$CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi > /dev/null; then
        # scripts/ctest_gpus.sh's busy test and knobs: never pin a tree to a GPU in use.
        gpu_src="idle"
        while IFS=, read -r idx util mem; do
            case $idx in ''|*[!0-9]*) continue ;; esac
            case "$util$mem" in ''|*[!0-9]*) util=0; mem=0 ;; esac
            if [ "${BATCHLAS_GPU_BUSY_SKIP:-1}" != 0 ] && { [ "$util" -gt "${BATCHLAS_GPU_BUSY_UTIL:-30}" ] ||
                    [ "$mem" -gt "${BATCHLAS_GPU_BUSY_MEM_MB:-4096}" ]; }; then
                busy="$busy GPU $idx (${util}%, ${mem} MiB)"
            else
                gpus+=("$idx")
            fi
        done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [ -z "$busy" ] || note "ctest: skipping busy$busy (BATCHLAS_GPU_BUSY_SKIP=0 keeps them)"
    fi
fi
gpu_vf=""; gpu_full=""
if [ "$parallel" = 0 ]; then
    if [ "${#gpus[@]}" -ge 2 ]; then gpu_vf=${gpus[0]}; gpu_full=${gpus[1]}
    elif [ "${#gpus[@]}" -eq 1 ] && [ -n "$busy" ]; then gpu_vf=${gpus[0]}; gpu_full=${gpus[0]}; fi
fi
[ "$gpu_src" != idle ] || [ "${#gpus[@]}" -gt 0 ] || [ -z "$busy" ] || note "ctest: every GPU is busy; running unpinned"

kinds=()
for kind in full vf; do
    [ -z "$only" ] || [ "$only" = "$kind" ] || continue
    dir=$full_dir; [ "$kind" = full ] || dir=$vf_dir
    if [ "$build" = 0 ]; then
        [ -f "$dir/CMakeCache.txt" ] || { note "$kind: --no-build but $dir has no CMakeCache.txt"; overall=1; continue; }
        mkdir -p "$dir/gate"
    elif ! build_tree "$kind" "$dir"; then
        overall=1; continue
    fi
    kinds+=("$kind")
done

cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
loaded=""
if [ "${#kinds[@]}" -eq 2 ] && [ "$serial" = 0 ] && [ -n "$gpu_full" ] && [ "$gpu_full" != "$gpu_vf" ]; then
    loaded=$(box_busy "${BATCHLAS_GATE_MAX_LOAD:-$(( cores / 4 > 0 ? cores / 4 : 1 ))}" || true)
fi
if [ "${#kinds[@]}" -eq 2 ] && [ "$serial" = 0 ] && [ -n "$gpu_full" ] && [ "$gpu_full" != "$gpu_vf" ] && [ -z "$loaded" ]; then
    # Builds are done and the lock is held, so nothing else on the box allocates beside the
    # two ctests (the recorded OOM kills both had a build running); rss.tsv shows the margin.
    cus=${BATCHLAS_GATE_CPU_CUS:-$(( cores / 2 > 0 ? cores / 2 : 1 ))}
    note "ctest: full on GPU $gpu_full and vf on GPU $gpu_vf at once, DPCPP_CPU_NUM_CUS=$cus each"
    t0=$(date +%s)
    # A subshell drops the parent's traps; the main trap TERMs it, its own kills its ctest group.
    { subshells=(); trap 'stop_children; exit 143' TERM; test_tree full "$full_dir" "$gpu_full" "$cus" || exit "$?"; } &
    pid_full=$!
    subshells+=("$pid_full")
    { subshells=(); trap 'stop_children; exit 143' TERM; test_tree vf "$vf_dir" "$gpu_vf" "$cus" || exit "$?"; } &
    pid_vf=$!
    subshells+=("$pid_vf")
    wait "$pid_full" || overall=1
    wait "$pid_vf" || overall=1
    subshells=()
    collect_notes "$full_dir"
    collect_notes "$vf_dir"
    note "ctest phase (both trees at once) took $(elapsed $(( $(date +%s) - t0 )))"
else
    why="${#gpus[@]} GPU(s) $gpu_src"
    [ -z "$loaded" ] || why="box busy: $loaded"
    [ "$serial" = 0 ] || why=--serial
    [ "$parallel" = 0 ] || why=--parallel
    [ "${#kinds[@]}" -eq 2 ] || why="one tree to test"
    [ "${#kinds[@]}" -eq 0 ] ||
        note "ctest: one tree at a time${gpu_full:+, full on GPU $gpu_full and vf on GPU $gpu_vf} ($why)"
    for kind in "${kinds[@]}"; do
        if [ "$kind" = full ]; then
            test_tree full "$full_dir" "$gpu_full" "" || overall=1
            collect_notes "$full_dir"
        else
            test_tree vf "$vf_dir" "$gpu_vf" "" || overall=1
            collect_notes "$vf_dir"
        fi
    done
fi

mkdir -p "$(dirname "$summary")"
{
    echo "BatchLAS gate summary  $(date '+%F %T')  host $(hostname)"
    echo "HEAD $(git -C "$root" rev-parse --short=12 HEAD)$(git -C "$root" diff --quiet HEAD -- 2>/dev/null || echo ' (+ uncommitted changes)')  preset $preset  CUDA=$cuda  ${only:+only=$only}$( [ "$build" = 1 ] || echo '  no-build (trees not rebuilt: binaries may predate HEAD)')"
    printf '%s\n' "${lines[@]}"
    echo "total $(elapsed $(( $(date +%s) - t_start )))  verdict: $( [ "$overall" -eq 0 ] && echo PASS || echo FAIL )"
} > "$summary.tmp"
mv "$summary.tmp" "$summary"
echo
cat "$summary"
exit "$overall"
