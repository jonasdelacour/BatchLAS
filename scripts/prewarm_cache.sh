#!/usr/bin/env bash
# Keep ccache warm for current origin/main: build it in a dedicated worktree, at the same
# depth-1 build dirs agents use, so a new worktree's first build is mostly cache hits.
# It is also a clean, current seed for scripts/seed_build_tree.sh.
#
#   scripts/prewarm_cache.sh [--force] [--print-cron]
#
# Environment:
#   BATCHLAS_PREWARM_DIR    the dedicated worktree (default ${XDG_CACHE_HOME:-~/.cache}/batchlas-prewarm/main)
#   BATCHLAS_PREWARM_TREES  space-separated NAME[=PRESET] build trees (default "build": the plain
#                           `cmake -B build` agent config; e.g. "build build-gate=dev-tests")
#   BATCHLAS_PREWARM_JOBS   parallelism (default 4)
#   BATCHLAS_MIN_MEM_GB     skip below this MemAvailable (default 32)
#   BATCHLAS_PREWARM_CXX    compiler (default /opt/dpcpp-cuda/bin/clang++ when present)
#
# Skips (exit 0) when origin/main is unchanged since the last successful prewarm, when another
# build holds the box-wide lock or a compiler/linker is running, or when memory is short.
set -eu
set -o pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
force=0
for arg in "$@"; do
    case $arg in
        --force) force=1 ;;
        --print-cron)
            echo "# crontab -e (not installed by this script); run it from the main checkout, not a worktree:"
            echo "*/30 * * * * $here/prewarm_cache.sh >> /tmp/batchlas-prewarm.log 2>&1"
            exit 0 ;;
        *) echo "usage: $0 [--force] [--print-cron]" >&2; exit 64 ;;
    esac
done

dir=${BATCHLAS_PREWARM_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/batchlas-prewarm/main}
trees=${BATCHLAS_PREWARM_TREES:-build}
jobs=${BATCHLAS_PREWARM_JOBS:-4}
min_gb=${BATCHLAS_MIN_MEM_GB:-32}
mkdir -p "$(dirname "$dir")"
log() { echo "prewarm $(date '+%F %T'): $*"; }

exec 8>"$(dirname "$dir")/prewarm.lock"
flock -n 8 || { log "another prewarm is running; skipping"; exit 0; }

# cron's environment is bare; mirror the agent setup (agent guide section 2b) without
# LIBRARY_PATH, which drags the HPC SDK's libgomp into -fopenmp links.
unset LIBRARY_PATH
if [ -d /opt/dpcpp-cuda ]; then
    PATH=/opt/dpcpp-cuda/bin:$PATH
    LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
for cuda in /usr/local/cuda /usr/local/cuda-13.2; do
    [ -d "$cuda/bin" ] && PATH=$PATH:$cuda/bin && break
done
PATH=$PATH:$HOME/.local/bin
export PATH LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cxx=${BATCHLAS_PREWARM_CXX:-}
[ -n "$cxx" ] || [ ! -x /opt/dpcpp-cuda/bin/clang++ ] || cxx=/opt/dpcpp-cuda/bin/clang++

git -C "$repo" fetch --quiet origin main
rev=$(git -C "$repo" rev-parse origin/main)
stamp="$dir.rev"
if [ "$force" != 1 ] && [ "$(cat "$stamp" 2>/dev/null || true)" = "$rev" ]; then
    log "origin/main ${rev:0:12} already prewarmed; nothing to do"
    exit 0
fi

busy=$(pgrep -l '^(make|gmake|ninja|clang\+\+|clang-[0-9]+|icpx|acpp|ptxas|sycl-post-link|llvm-link|clang-offload-b|clang-linker-wr)$' || true)
if [ -n "$busy" ]; then
    log "another build is running; skipping: $(printf '%s' "$busy" | head -n 3 | tr '\n' ' ')"
    exit 0
fi
avail=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if [ "$avail" -lt "$min_gb" ]; then
    log "MemAvailable ${avail} GB < ${min_gb} GB; skipping"
    exit 0
fi

if [ ! -e "$dir/.git" ]; then
    log "creating worktree $dir"
    git -C "$repo" worktree add --detach "$dir" "$rev"
else
    git -C "$dir" checkout --quiet --detach --force "$rev"
    git -C "$dir" clean -fdq
fi
cd "$dir"

status=0
for spec in $trees; do
    name=${spec%%=*}
    preset=""
    [ "$name" = "$spec" ] || preset=${spec#*=}
    case $name in */*) log "tree $name is not depth 1 (ccache hashes relative -I paths); skipping"; status=1; continue ;; esac
    args=(-S . -B "$name")
    [ -z "$preset" ] || args=(--preset "$preset" -B "$name")
    [ -z "$cxx" ] || args+=("-DCMAKE_CXX_COMPILER=$cxx")
    log "configure $name ${preset:+(preset $preset)} at ${rev:0:12}"
    if ! nice -n 19 cmake "${args[@]}" > "$dir/$name.configure.log" 2>&1; then
        log "configure of $name failed: $dir/$name.configure.log"; status=1; continue
    fi
    if ! grep -q 'Using SYCL targets:.*nvidia_gpu' "$dir/$name.configure.log" && [ "${BATCHLAS_PREWARM_ALLOW_CPU:-0}" != 1 ]; then
        log "$name configured without a CUDA target (see $dir/$name.configure.log); not warming a cache nobody uses"
        status=1; continue
    fi
    log "build $name (-j$jobs, nice 19)"
    rc=0
    BATCHLAS_FULL_BUILD=1 BATCHLAS_BUILD_JOBS=$jobs BATCHLAS_BUILD_LOCK_WAIT=0 BATCHLAS_MIN_MEM_GB=$min_gb \
        BATCHLAS_MEM_WAIT_MAX=0 nice -n 19 "$here/b" -B "$name" > /dev/null 2>"$dir/$name.build.err" || rc=$?
    if [ "$rc" -eq 75 ] || [ "$rc" -eq 69 ]; then
        log "build of $name deferred (lock busy or memory short); will retry next run"; exit 0
    elif [ "$rc" -ne 0 ]; then
        log "build of $name failed (exit $rc): $(tail -n 1 "$dir/$name.build.err")"; status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "$rev" > "$stamp"
    log "prewarmed origin/main ${rev:0:12}"
    command -v ccache > /dev/null && ccache --show-stats 2>/dev/null | grep -iE 'hits|cache size' | head -n 4 || true
fi
exit "$status"
