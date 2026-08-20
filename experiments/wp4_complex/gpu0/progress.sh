#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."
prev=-1
for i in $(seq 1 400); do
    n=$(ls experiments/wp4_complex/gpu0/raw/*.csv 2>/dev/null | wc -l)
    if [ "$n" != "$prev" ]; then
        echo "csv=$n foreign=[$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')]"
        prev=$n
    fi
    if [ "$n" -ge 360 ]; then echo SWEEP_DONE; break; fi
    sleep 60
done
