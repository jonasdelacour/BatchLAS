#!/usr/bin/env bash
#
# Merge the per-process coverage shards into one table.
#
# dispatch/coverage.cc writes $BATCHLAS_COVERAGE_OUT.<pid>, one file per
# process, because a ctest run is 53 separate binaries and a single shared file
# meant each one truncated the last (see the comment on emit()). This collapses
# the shards.
#
#   scripts/coverage_merge.sh <base-path>
#
# Writes <base-path> and removes the shards. Keeps one header, sums `calls` for
# rows that appear in more than one binary, and de-duplicates the `linked` rows
# (every process emits the same static table, so without this they appear 53
# times).

set -uo pipefail

base="${1:?usage: $0 <base-path>}"
shopt -s nullglob
shards=("$base".[0-9]*)

if [[ ${#shards[@]} -eq 0 ]]; then
    printf 'coverage_merge: no shards matching %s.<pid> -- did anything run with BATCHLAS_COVERAGE_OUT set?\n' "$base" >&2
    exit 1
fi

python3 - "$base" "${shards[@]}" <<'PY'
import sys, os, csv

base, shards = sys.argv[1], sys.argv[2:]
header = None
reached, misses, linked = {}, {}, {}

for path in shards:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        continue
    if header is None:
        header = rows[0]
    for r in rows[1:]:
        if not r:
            continue
        kind = r[0]
        # `calls` is column 11. Everything else identifies the row, so the key
        # is the row with that one field blanked -- two binaries hitting the
        # same op/shape/route must sum, not appear twice.
        key = tuple(r[:11] + r[12:])
        table = {"reached": reached, "miss": misses, "linked": linked}.get(kind)
        if table is None:
            continue
        if key in table:
            try:
                table[key][11] = str(int(table[key][11] or 0) + int(r[11] or 0))
            except ValueError:
                pass
        else:
            table[key] = list(r)

with open(base, "w", newline="") as fh:
    w = csv.writer(fh)
    if header:
        w.writerow(header)
    for table in (reached, misses, linked):
        for row in table.values():
            w.writerow(row)

print(f"merged {len(shards)} shards -> {base} "
      f"({len(reached)} reached, {len(misses)} miss, {len(linked)} linked)")
PY
status=$?

if [[ $status -eq 0 ]]; then
    rm -f "${shards[@]}"
fi
exit $status
