#!/usr/bin/env bash
# Pre-push gate on persistent trees: configure once, rebuild incrementally, full ctest, ledger diff.
#
#   scripts/gate.sh [--only full|vf] [--seed-from TREE [--seed-allow-dirty]] [--parallel]
#
#   build-gate     full targets, preset dev-tests (the ledger tests/known-failures.txt was
#                  recorded under it); full ctest, slow suites included; gated by
#                  .github/ci/compare_failures.py
#   build-gate-vf  the same plus -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF; ctest -LE slow (the heavy
#                  case tiers included: their vendor-free arms run nowhere else). It has no
#                  ledger (see the nightly CI job), so it gates on build success and reports
#                  failing names, plus the diff against this tree's previous gate run.
#
#   --seed-from T  seed a tree that does not exist yet from built tree T (scripts/seed_build_tree.sh),
#                  only when T's BATCHLAS_ENABLE_VENDOR_BLAS matches that tree's
#   --seed-allow-dirty  pass --allow-dirty to the seeding (T's checkout has uncommitted files)
#   --parallel     scripts/ctest_gpus.sh instead of serial ctest (validated on threadripper02 only)
#
# Environment: BATCHLAS_GATE_CUDA (default ON), BATCHLAS_GATE_PRESET (default dev-tests),
# BATCHLAS_GATE_DIR / BATCHLAS_GATE_VF_DIR (default build-gate / build-gate-vf),
# BATCHLAS_GATE_CXX (compiler for a new tree; default /opt/dpcpp-cuda/bin/clang++ when present).
# Every configure forces the gate's shape (full CPU+GPU targets, every test, split libraries,
# no offset twins), so a GPU-only or smoke seed cannot turn the gate into a smaller run.
# The box-wide build lock (/tmp/batchlas-build.lock) is held for the whole gate, ctest included:
# a build beside a full ctest is how the 2x4090 box was OOM-killed (docs/ci.md).
# Exit 0 only when every selected tree built and the full tree's ledger diff is clean.
set -eu
set -o pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
only=""
seed_from=""
seed_dirty=()
parallel=0
while [ "$#" -gt 0 ]; do
    case $1 in
        --only) only=$2; shift 2 ;;
        --seed-from) seed_from=$(cd "$2" && pwd); shift 2 ;;
        --seed-allow-dirty) seed_dirty=(--allow-dirty); shift ;;
        --parallel) parallel=1; shift ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
summary="$root/$summary_dir/gate-summary.txt"
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

lines=()
overall=0
note() { lines+=("$*"); echo "gate: $*"; }
elapsed() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

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

run_tree() {
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
    t0=$(date +%s)
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
    [ "$rc" -eq 0 ] || return 1

    local junit="$root/$dir/gate/ctest-junit.xml" gx="$root/$dir/gate/gtest-xml"
    local -a filter=()
    [ "$kind" = full ] || filter=(-LE slow)
    rm -rf "$gx" "$junit"; mkdir -p "$gx"
    t0=$(date +%s); rc=0
    if [ "$parallel" = 1 ]; then
        GTEST_OUTPUT="xml:$gx/" "$here/ctest_gpus.sh" --test-dir "$dir" --output-junit "$junit" "${filter[@]}" \
            > "$dir/gate/ctest.out" 2>&1 || rc=$?
    else
        GTEST_OUTPUT="xml:$gx/" ctest --test-dir "$dir" -j 1 --output-junit "$junit" "${filter[@]}" \
            > "$dir/gate/ctest.out" 2>&1 || rc=$?
    fi
    note "$kind: ctest ${filter[*]:-(full)} exit $rc in $(elapsed $(( $(date +%s) - t0 ))): $(grep -E 'tests passed|tests failed' "$dir/gate/ctest.out" | tail -n 1)"
    [ -f "$junit" ] || { note "$kind: no junit report written"; return 1; }

    [ ! -f "$dir/gate/failing.txt" ] || mv "$dir/gate/failing.txt" "$dir/gate/failing.prev.txt"
    failing_names "$junit" > "$dir/gate/failing.txt"
    note "$kind: $(grep -c . "$dir/gate/failing.txt" || true) failing ctest test(s): $(tr '\n' ' ' < "$dir/gate/failing.txt")"

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
        note "full: ledger diff $( [ "$rc" -eq 0 ] && echo CLEAN || echo FAILED ) ($dir/gate/compare.txt)"
        [ "$rc" -eq 0 ] || while IFS= read -r l; do lines+=("    $l"); done < <(tail -n 20 "$dir/gate/compare.txt")
        return "$rc"
    fi
    note "vf: ledger diff exit $rc, informational only (no vendor-free ledger exists)"
    if [ -f "$dir/gate/failing.prev.txt" ]; then
        local new fixed
        new=$(comm -13 "$dir/gate/failing.prev.txt" "$dir/gate/failing.txt" | tr '\n' ' ')
        fixed=$(comm -23 "$dir/gate/failing.prev.txt" "$dir/gate/failing.txt" | tr '\n' ' ')
        note "vf: vs previous gate run: new failures [${new}] now passing [${fixed}]"
    fi
    return 0
}

export LC_ALL=C
for kind in full vf; do
    [ -z "$only" ] || [ "$only" = "$kind" ] || continue
    dir=$full_dir; [ "$kind" = full ] || dir=$vf_dir
    run_tree "$kind" "$dir" || overall=1
done

mkdir -p "$(dirname "$summary")"
{
    echo "BatchLAS gate summary  $(date '+%F %T')  host $(hostname)"
    echo "HEAD $(git -C "$root" rev-parse --short=12 HEAD)$(git -C "$root" diff --quiet HEAD -- 2>/dev/null || echo ' (+ uncommitted changes)')  preset $preset  CUDA=$cuda  ${only:+only=$only}"
    printf '%s\n' "${lines[@]}"
    echo "total $(elapsed $(( $(date +%s) - t_start )))  verdict: $( [ "$overall" -eq 0 ] && echo PASS || echo FAIL )"
} > "$summary.tmp"
mv "$summary.tmp" "$summary"
echo
cat "$summary"
exit "$overall"
