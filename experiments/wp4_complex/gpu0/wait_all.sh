#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."
prev=""
for i in $(seq 1 600); do
    a=$(ls experiments/wp4_complex/gpu0/raw/*.csv 2>/dev/null | wc -l)
    b=$(ls experiments/wp4_complex/gpu0/raw2/*.csv 2>/dev/null | wc -l)
    c=$(ls experiments/wp4_complex/gpu0/raw3/*.csv 2>/dev/null | wc -l)
    k=$(wc -l < experiments/wp4_complex/gpu0/kernels.txt 2>/dev/null || echo 0)
    cur="$a/$b/$c/$k"
    if [ "$cur" != "$prev" ]; then
        echo "sweep1=$a/360 sweep2=$b/192 sweep3=$c/112 kernels=$k/120"
        prev="$cur"
    fi
    if ! pgrep -f "wp4_complex/gpu0/chain2.sh" > /dev/null; then
        echo "ALL_DONE sweep1=$a sweep2=$b sweep3=$c kernels=$k"
        break
    fi
    sleep 120
done
