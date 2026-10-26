#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run a headless Claude session on another box (or this one) in its own worktree, from the Mac.
#
#   scripts/agent/dispatch.sh hosts                          load, memory, GPUs, dispatches per box
#   scripts/agent/dispatch.sh start HOST NAME BRIEF [--gpu N] [--base REF]
#   scripts/agent/dispatch.sh list [HOST]                    dispatched sessions and their state
#   scripts/agent/dispatch.sh logs HOST NAME [N]             last N assistant messages (default 8)
#   scripts/agent/dispatch.sh report HOST NAME               the session's .dispatch/report.md
#   scripts/agent/dispatch.sh stop HOST NAME
#   scripts/agent/dispatch.sh rm HOST NAME                   remove the worktree and local branch
#
# HOST: jrhnbr (2x RTX 4090, shared), tr02 (threadripper02: CPU and build work only, its GPUs
# are hidden and --gpu is refused, since GPU load has crashed it), mac (Apple M5 Max, Metal).
# start: worktree ~/BatchLAS/.claude/worktrees/dispatch-NAME on branch dispatch/NAME from
# origin/main, a seeded build tree, then `claude -p` under nohup (pid in .dispatch/pid) with the host rules
# prepended to BRIEF. The session pushes dispatch/NAME and writes .dispatch/report.md.
set -u
die() { echo "dispatch: $*" >&2; exit 2; }

ssh_of() {
    case "$1" in
        jrhnbr) echo JRHNBR-jump ;;
        tr02) echo threadripper02-jump ;;
        mac) echo local ;;
        *) die "unknown host '$1' (jrhnbr, tr02, mac)" ;;
    esac
}

count_running() {
    local repo; repo=$(repo_of "$1")
    echo "n=0; for p in $repo/.claude/worktrees/dispatch-*/.dispatch/pid; do [ -f \$p ] && kill -0 \$(cat \$p) 2>/dev/null && n=\$((n+1)); done; echo \$n"
}

max_jobs() { case "$1" in tr02) echo 2 ;; jrhnbr) echo 2 ;; mac) echo 3 ;; esac; }

on() {
    local h; h=$(ssh_of "$1"); shift
    if [ "$h" = local ]; then bash -lc "$*"
    else ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "bash -lc $(printf '%q' "$*")"; fi
}

repo_of() { [ "$1" = mac ] && echo "$HOME/Projects/BatchLAS" || echo '$HOME/BatchLAS'; }

rules_of() {
    case "$1" in
        tr02) cat <<'EOF'
- Host threadripper02: 64 cores, 503 GB, 4x RTX PRO 6000 (sm_120). GPU work here has crashed the
  box several times, so the GPUs are hidden from you (CUDA_VISIBLE_DEVICES is empty,
  ONEAPI_DEVICE_SELECTOR=opencl:cpu). Do builds, CPU tests and analysis only. Do not unset these
  variables, run nvidia-smi workloads, multi-GPU sweeps or `scripts/gate.sh --parallel`.
EOF
            ;;
        jrhnbr) cat <<'EOF'
- Host JRHNBR: 20 cores, 123 GB, 2x RTX 4090 (sm_89), shared with another user. Run
  scripts/agent/gpu_health.sh before GPU work; measure only on the GPU you were given
  (CUDA_VISIBLE_DEVICES) and only one measuring process at a time.
EOF
            ;;
        mac) cat <<'EOF'
- Host Mac Studio: Apple M5 Max, 18 cores, 36 GB, AdaptiveCpp (CPU OpenMP + Metal backends), no
  GPU FP64; also the user's desktop. Build through scripts/b only (-j18, box-wide lock, memory
  floor), never an unbounded -j (it has kernel-panicked this machine). Owns: acpp/Metal builds and
  tests, CPU-backend correctness incl. FP64, code and test writing, Python/benchviz, docs. CUDA
  results and sm_89 timing come from jrhnbr, not here.
EOF
            ;;
    esac
}

cmd_start() {
    [ "$#" -ge 3 ] || die "start HOST NAME BRIEF [--gpu N] [--base REF]"
    local host=$1 name=$2 brief=$3 gpu="" base=origin/main
    shift 3
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --gpu) gpu=$2; shift 2 ;;
            --base) base=$2; shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    case "$host" in jrhnbr | tr02 | mac) ;; *) die "unknown host '$host' (jrhnbr, tr02, mac)" ;; esac
    [ -f "$brief" ] || die "no brief file $brief"
    [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "name must match [A-Za-z0-9._-]+"
    [ "$host" = tr02 ] && [ -n "$gpu" ] && die "tr02 is CPU/build only; no --gpu"
    [ "$host" = mac ] && [ -n "$gpu" ] && die "mac has no CUDA GPU index"

    local repo wt running env seeded
    repo=$(repo_of "$host")
    wt="$repo/.claude/worktrees/dispatch-$name"
    running=$(on "$host" "$(count_running "$host")" || true)
    [ "${running:-0}" -lt "$(max_jobs "$host")" ] ||
        die "$host already runs $running dispatches (max $(max_jobs "$host")); pick another host"

    case "$host" in
        tr02) env="export CUDA_VISIBLE_DEVICES= ONEAPI_DEVICE_SELECTOR=opencl:cpu" ;;
        jrhnbr) env="export ONEAPI_DEVICE_SELECTOR='cuda:*;opencl:cpu'${gpu:+ CUDA_VISIBLE_DEVICES=$gpu}" ;;
        mac) env="export BATCHLAS_BUILD_JOBS=18 BATCHLAS_ALLOW_WIDE=1 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0='!gh auth git-credential'" ;;
    esac

    on "$host" "set -e; cd $repo; git fetch -q origin
        GIT_LFS_SKIP_SMUDGE=1 git worktree add -q -b dispatch/$name $wt $base ||
            { git branch -D dispatch/$name 2>/dev/null; exit 1; }
        ex=\$(git rev-parse --git-common-dir)/info/exclude
        grep -qx '.dispatch/' \$ex 2>/dev/null || echo '.dispatch/' >> \$ex
        mkdir -p $wt/.dispatch
        seed=\${BATCHLAS_SEED_BUILD:-$repo/build}
        if [ -d \$seed ]; then cd $wt && scripts/seed_build_tree.sh --allow-dirty \$seed build > .dispatch/seed.log 2>&1 || echo 'seed failed: see .dispatch/seed.log'; fi" ||
        die "worktree setup failed on $host"

    {
        echo "You are a dispatched BatchLAS session started from the Mac Studio. Work autonomously;"
        echo "nobody answers questions or permission prompts, so make reasonable assumptions and note them."
        rules_of "$host"
        seeded="build tree seeded in ./build (.dispatch/seed.log)"
        echo "- Worktree $wt, branch dispatch/$name from $base, $seeded. LFS files are"
        echo "  pointers; run \`git lfs pull\` if the task needs them."
        echo "- Commit on your branch and \`git push -u origin dispatch/$name\`. Never push main; open"
        echo "  a PR only if the brief asks."
        echo "- Finish by writing .dispatch/report.md: what you did, results with numbers, the pushed"
        echo "  commit SHA, and anything unresolved."
        echo
        echo "Task:"
        cat "$brief"
    } | on "$host" "cat > $wt/.dispatch/brief.md"

    on "$host" "cat > $wt/.dispatch/run.sh <<'EOF'
cd \"\$(dirname \"\$0\")/..\"
$env
claude=\$(command -v claude || echo \$HOME/.local/bin/claude)
\"\$claude\" -p \"\$(cat .dispatch/brief.md)\" -n dispatch-$name --permission-mode auto \\
    --permission-prompts none --output-format stream-json --verbose \\
    > .dispatch/log.jsonl 2> .dispatch/stderr.log
echo \$? > .dispatch/exit
EOF
        nohup bash $wt/.dispatch/run.sh > /dev/null 2>&1 < /dev/null & echo \$! > $wt/.dispatch/pid" ||
        die "launch failed on $host"
    echo "started $name on $host: $wt (branch dispatch/$name)"
}

cmd_list() {
    local hosts=${1:-"mac jrhnbr tr02"} h repo
    for h in $hosts; do
        repo=$(repo_of "$h")
        echo "== $h"
        on "$h" "for d in $repo/.claude/worktrees/dispatch-*/; do
            [ -d \$d/.dispatch ] || continue
            n=\$(basename \$d); n=\${n#dispatch-}
            if [ -f \$d/.dispatch/pid ] && kill -0 \$(cat \$d/.dispatch/pid) 2>/dev/null; then s=running
            elif [ -f \$d/.dispatch/exit ]; then s=\"exit \$(cat \$d/.dispatch/exit)\"
            else s=dead; fi
            [ -f \$d/.dispatch/report.md ] && s=\"\$s, report\"
            printf '%-32s %-18s %s\n' \$n \"\$s\" \"\$(git -C \$d log -1 --format='%h %cr %s' | cut -c1-60)\"
        done; true" 2>/dev/null || echo "(unreachable)"
    done
}

cmd_logs() {
    [ "$#" -ge 2 ] || die "logs HOST NAME [N]"
    local repo; repo=$(repo_of "$1")
    on "$1" "tail -n 400 $repo/.claude/worktrees/dispatch-$2/.dispatch/log.jsonl" |
        python3 -c '
import json, sys
n = int(sys.argv[1]); out = []
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("type") == "assistant":
        for c in m.get("message", {}).get("content", []):
            if c.get("type") == "text" and c["text"].strip(): out.append(c["text"].strip())
    elif m.get("type") == "result":
        out.append("[result %s] %s" % (m.get("subtype"), str(m.get("result", ""))[:2000]))
for t in out[-n:]: print(t[:1500]); print("---")
' "${3:-8}"
}

cmd_hosts() {
    local h
    for h in mac jrhnbr tr02; do
        printf '== %s  ' "$h"
        on "$h" "printf 'load %s  ' \"\$(uptime | sed 's/.*average[s]*: //')\"
            if command -v free >/dev/null; then free -g | sed -n 2p | tr -s ' ' | cut -d' ' -f2,7 | sed 's/^/mem total,avail GB /'
            else echo \"mem \$((\$(sysctl -n hw.memsize) >> 30)) GB\"; fi
            echo \"dispatches running: \$($(count_running "$h"))\"
            command -v flock >/dev/null && [ -e /tmp/batchlas-build.lock ] && ! flock -n /tmp/batchlas-build.lock true &&
                echo \"build lock held: \$(cat /tmp/batchlas-build.lock.info 2>/dev/null)\"
            command -v nvidia-smi >/dev/null && timeout 20 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu /'
            true" 2>&1 | grep -v 'stdio forwarding\|Connection closed\|channel 0' || echo "(unreachable)"
    done
}

sub=${1:-}; [ "$#" -gt 0 ] && shift
case "$sub" in
    hosts) cmd_hosts ;;
    start) cmd_start "$@" ;;
    list) cmd_list "$@" ;;
    logs) cmd_logs "$@" ;;
    report) [ "$#" -ge 2 ] || die "report HOST NAME"
        on "$1" "cat $(repo_of "$1")/.claude/worktrees/dispatch-$2/.dispatch/report.md" ;;
    stop) [ "$#" -ge 2 ] || die "stop HOST NAME"
        on "$1" "pkill -P \$(cat $(repo_of "$1")/.claude/worktrees/dispatch-$2/.dispatch/pid)" ;;
    rm) [ "$#" -ge 2 ] || die "rm HOST NAME"
        d="$(repo_of "$1")/.claude/worktrees/dispatch-$2"
        on "$1" "[ -f $d/.dispatch/pid ] && kill -0 \$(cat $d/.dispatch/pid) 2>/dev/null && { echo 'still running; stop it first'; exit 1; }
            cd $(repo_of "$1") && git worktree remove --force $d && git branch -D dispatch/$2" ;;
    *) sed -n '3,17p' "$0"; exit 2 ;;
esac
