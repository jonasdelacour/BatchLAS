#!/usr/bin/env bash
# Drop any raw CSV that is not exactly header + 2 type rows -- i.e. a cell whose
# benchmark process was interrupted. Keeps the resume logic from adopting a
# half-written file.
set -uo pipefail
cd "$(dirname "$0")"
for d in raw raw2 raw3; do
    [ -d "$d" ] || continue
    for f in "$d"/*.csv; do
        [ -e "$f" ] || continue
        n=$(wc -l < "$f")
        if [ "$n" -ne 3 ]; then
            echo "removing $f ($n lines)"
            rm -f "$f"
        fi
    done
    echo "$d: $(ls "$d"/*.csv 2>/dev/null | wc -l) intact"
done
