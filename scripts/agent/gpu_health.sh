#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Is this box fit for GPU work right now? Run before any build-and-measure or test campaign,
# and stop (tell the user) on a FAIL instead of carrying on "compile-verified".
#
#   scripts/agent/gpu_health.sh [--quiet]
#
# Checks: nvidia-smi answers (a hang or "Driver/library version mismatch" means unattended
# upgrades replaced the driver under the running kernel: only a reboot or module reload fixes
# it); the loaded module matches the installed one; sycl-ls lists a [cuda:gpu]; recent Xid
# errors; foreign compute processes per GPU (the box is shared).
# Exit 0 healthy (warnings allowed), 1 unusable.
set -u
quiet=0; [ "${1:-}" = --quiet ] && quiet=1
fail=0
say() { [ "$quiet" = 1 ] && [ "$1" = ok ] && return 0; printf '%-5s %s\n' "$1" "$2"; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
    say FAIL "nvidia-smi not installed"; exit 1
fi
smi=$(timeout 20 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1)
rc=$?
if [ "$rc" = 124 ]; then
    say FAIL "nvidia-smi hung for 20 s: the driver is wedged (reboot needed; tell the user)"; fail=1
elif [ "$rc" != 0 ]; then
    say FAIL "nvidia-smi: $(echo "$smi" | head -n 2 | tr '\n' ' ')"; fail=1
else
    while IFS= read -r l; do say ok "gpu $l"; done <<<"$smi"
fi

loaded=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -n 1)
installed=$(modinfo nvidia 2>/dev/null | sed -n 's/^version: *//p' | head -n 1)
if [ -n "$loaded" ] && [ -n "$installed" ] && [ "$loaded" != "$installed" ]; then
    say FAIL "driver mismatch: loaded module $loaded, installed $installed (unattended-upgrades; reboot)"; fail=1
fi

sycl_ls=$(command -v sycl-ls || echo /opt/dpcpp-cuda/bin/sycl-ls)
if [ -x "$sycl_ls" ]; then
    devs=$(LD_LIBRARY_PATH=/opt/dpcpp-cuda/lib:${LD_LIBRARY_PATH:-} timeout 60 "$sycl_ls" 2>&1)
    n=$(grep -c '^\[cuda:gpu\]' <<<"$devs")
    if [ "$n" -gt 0 ]; then say ok "sycl-ls: $n [cuda:gpu] device(s)"
    else say FAIL "sycl-ls lists no [cuda:gpu] (cuInit failing?): $(head -n 3 <<<"$devs" | tr '\n' ' ')"; fail=1; fi
else
    say WARN "no sycl-ls found"
fi

xid=$(journalctl -k --since "-2h" --no-pager 2>/dev/null | grep -c 'NVRM: Xid' || true)
[ "${xid:-0}" -gt 0 ] && say WARN "$xid NVRM Xid line(s) in the kernel log in the last 2 h"

if [ "$rc" = 0 ]; then
    apps=$(timeout 20 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null)
    while IFS=, read -r _uuid p mem; do
        [ -n "${p// }" ] || continue
        owner=$(ps -o user= -p "${p// }" 2>/dev/null || echo '?')
        cmd=$(ps -o comm= -p "${p// }" 2>/dev/null || echo '?')
        mib=$(tr -dc 0-9 <<<"$mem")
        if [ "$owner" = "$(id -un)" ]; then say ok "own GPU process pid${p} ($cmd,${mem})"
        elif [ "${mib:-0}" -lt 1024 ]; then say ok "light foreign process pid${p} ($owner, $cmd,${mem}): harmless for timing"
        else say WARN "foreign GPU process pid${p} ($owner, $cmd,${mem}): do not time anything on that GPU"; fi
    done <<<"$apps"
fi

for lk in /tmp/batchlas-build.lock /tmp/batchlas_tune_gpu*.lock; do
    [ -e "$lk" ] || continue
    if ! flock -n "$lk" true 2>/dev/null; then
        say WARN "$lk is held: $(cat "$lk.info" 2>/dev/null || echo 'holder unknown; see scripts/agent/tune_procs.sh')"
    fi
done

[ "$fail" = 0 ] && say ok "verdict: healthy" || say FAIL "verdict: GPU work is blocked; tell the user now"
exit "$fail"
