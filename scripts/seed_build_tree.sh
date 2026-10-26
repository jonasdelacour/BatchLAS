#!/usr/bin/env bash
# Seed a new worktree's build tree from an already-built tree, so its first build is incremental.
#
#   scripts/seed_build_tree.sh [options] SRC_BUILD DST_BUILD
#     --dst-src DIR    destination source tree (default: git toplevel containing DST_BUILD's parent)
#     --allow-dirty    accept a seed checkout with uncommitted changes (its dirty paths are rebuilt)
#     --build T...     after seeding, build these targets through scripts/b and check RUNPATHs
#                      (must be the LAST option: every later word is a target)
#
#   scripts/seed_build_tree.sh ~/BatchLAS/build build       # from inside a fresh worktree
#
# Steps: cp -a --reflink=auto (cp -ac, an APFS clone, on macOS); rewrite the seed's absolute source/build paths in every text
# file (CMakeCache.txt, Makefiles, flags.make, link.txt, depend files, CTestTestfile.cmake,
# the batchlas-ccache wrapper); patch seed RUNPATHs to $ORIGIN form (scripts/seed_build_tree_fix.py);
# give unchanged sources the seed checkout's mtimes and touch every file that differs from
# the seed's commit (git diff, uncommitted and untracked files); `cmake -S -B` to regenerate.
# Make then rebuilds exactly the changed TUs and their dependents.
# Seed BEFORE editing in the new worktree; an edit made while this runs may be missed (the
# changed set is recomputed once at the end to narrow that window).
set -eu
set -o pipefail
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd)
fix="$here/seed_build_tree_fix.py"
allow_dirty=0
dst_src=""
build_targets=()
pos=()
while [ "$#" -gt 0 ]; do
    case $1 in
        --allow-dirty) allow_dirty=1; shift ;;
        --dst-src) dst_src=$2; shift 2 ;;
        --build) shift; build_targets=("$@"); break ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "seed: unknown option $1" >&2; exit 64 ;;
        *) pos+=("$1"); shift ;;
    esac
done
[ "${#pos[@]}" -eq 2 ] || { echo "usage: $0 [--allow-dirty] [--dst-src DIR] SRC_BUILD DST_BUILD [--build T...]" >&2; exit 64; }
t0=$(date +%s)

die() { echo "seed: $*" >&2; exit 1; }
cache_get() { sed -n "s/^$2:[A-Z]*=//p" "$1/CMakeCache.txt" | head -n 1; }

[ -f "${pos[0]}/CMakeCache.txt" ] || die "${pos[0]} has no CMakeCache.txt"
src_build=$(cd "${pos[0]}" && pwd -P)
old_build=$(cache_get "$src_build" CMAKE_CACHEFILE_DIR)
old_src=$(cache_get "$src_build" CMAKE_HOME_DIRECTORY)
[ -n "$old_build" ] && [ -n "$old_src" ] || die "cannot read CMAKE_CACHEFILE_DIR / CMAKE_HOME_DIRECTORY"
[ -d "$old_src/.git" ] || [ -f "$old_src/.git" ] || die "seed source $old_src is not a git checkout"

dst_arg=${pos[1]}
[ ! -e "$dst_arg" ] || [ -z "$(ls -A "$dst_arg" 2>/dev/null)" ] || die "$dst_arg exists and is not empty"
dst_parent=$(cd "$(dirname "$dst_arg")" && pwd -P)
dst_build="$dst_parent/$(basename "$dst_arg")"
if [ -z "$dst_src" ]; then
    dst_src=$(git -C "$dst_parent" rev-parse --show-toplevel) || die "pass --dst-src"
fi
dst_src=$(cd "$dst_src" && pwd -P)
[ -f "$dst_src/CMakeLists.txt" ] || die "$dst_src has no CMakeLists.txt"

# Depend files and flags.make hold build-relative paths (-I../..): the build dirs must sit
# at the same depth below their sources (the names may differ: build vs build-gate).
depth() { python3 -c 'import os,sys; r=os.path.relpath(sys.argv[1], sys.argv[2]); print(-1 if r.startswith("..") else r.count("/") + 1)' "$1" "$2"; }
d_old=$(depth "$old_build" "$old_src")
d_new=$(depth "$dst_build" "$dst_src")
[ "$d_old" = "$d_new" ] && [ "$d_old" != -1 ] || die "layout differs: seed build is at depth $d_old below its source, destination at $d_new (-1: outside)"

seed_rev=$(git -C "$old_src" rev-parse HEAD)
seed_dirty=$(git -C "$old_src" status --porcelain=v1 --untracked-files=all | cut -c4- | sed 's/.* -> //')
if [ -n "$seed_dirty" ] && [ "$allow_dirty" != 1 ]; then
    printf '%s\n' "$seed_dirty" | head -n 10 >&2
    die "seed checkout $old_src is dirty; commit, or pass --allow-dirty (dirty paths are rebuilt)"
fi
git -C "$dst_src" cat-file -e "$seed_rev^{commit}" 2>/dev/null || die "seed commit $seed_rev is unknown to $dst_src (git fetch?)"

work=$(mktemp -d "${TMPDIR:-/tmp}/batchlas-seed.XXXXXX")
trap 'rm -rf "$work"' EXIT

changed_sets() {
    {
        git -C "$dst_src" diff --name-only --no-renames "$seed_rev"
        git -C "$dst_src" ls-files --others --exclude-standard
        [ -z "$seed_dirty" ] || printf '%s\n' "$seed_dirty"
    } | sort -u > "$work/changed"
    git -C "$dst_src" ls-files | sort -u | comm -23 - "$work/changed" > "$work/unchanged"
}

echo "seed: copying $src_build -> $dst_build"
if cp --reflink=auto --version >/dev/null 2>&1; then clone() { cp -a --reflink=auto "$@"; }
else clone() { cp -ac "$@" 2>/dev/null || cp -a "$@"; }; fi
mkdir -p "$dst_build"
# Nested trees (build/presets/* from before the depth-1 preset layout) are not copied.
for entry in "$src_build"/* "$src_build"/.[!.]*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    name=${entry##*/}
    case $name in logs|Testing) continue ;; _deps|CMakeFiles) clone "$entry" "$dst_build/"; continue ;; esac
    if [ -d "$entry" ] && [ ! -e "$old_src/$name" ] && [ -n "$(find "$entry" -maxdepth 2 -name CMakeCache.txt -print -quit)" ]; then
        echo "seed: skipping nested build tree $entry"
        continue
    fi
    clone "$entry" "$dst_build/"
done

pairs=("$old_build=$dst_build" "$old_src=$dst_src")
[ "$src_build" = "$old_build" ] || pairs+=("$src_build=$dst_build")
python3 "$fix" rewrite "$dst_build" "${pairs[@]}"
python3 "$fix" runpath "$dst_build" "$old_build" "$dst_build" "$old_src" "$dst_src"

changed_sets
python3 "$fix" mtimes "$old_src" "$dst_src" "$work/changed" "$work/unchanged"

echo "seed: regenerating with cmake -S $dst_src -B $dst_build"
python3 "$fix" keep-outputs "$dst_build" "$work"
cmake -S "$dst_src" -B "$dst_build" > "$work/configure.log" 2>&1 || { tail -n 40 "$work/configure.log" >&2; die "cmake regenerate failed"; }
python3 "$fix" restore-outputs "$dst_build" "$work"
grep -E 'Using SYCL targets|ccache enabled' "$work/configure.log" || true

cp "$work/changed" "$work/changed.first"
changed_sets
late=$(comm -13 "$work/changed.first" "$work/changed")
if [ -n "$late" ]; then
    printf '%s\n' "$late" > "$work/late"
    python3 "$fix" mtimes "$old_src" "$dst_src" "$work/late" /dev/null
fi

nchanged=$(grep -c . "$work/changed" || true)
cat <<EOF
seed: summary
  seed tree       $src_build (source $old_src @ ${seed_rev:0:12}${seed_dirty:+, dirty})
  destination     $dst_build (source $dst_src @ $(git -C "$dst_src" rev-parse --short=12 HEAD))
  size            $(du -sh "$dst_build" | cut -f1)
  changed files   $nchanged (touched: make rebuilds them and their dependents)
$(head -n 15 "$work/changed" | sed 's/^/                  /')
  elapsed         $(( $(date +%s) - t0 )) s
  next            scripts/b -B $dst_build <target>   (preview: cmake --build $dst_build --target <t> -- -n | grep -c 'Building CXX')
EOF

if [ "${#build_targets[@]}" -gt 0 ]; then
    "$here/b" -B "$dst_build" "${build_targets[@]}"
    python3 "$fix" runpath "$dst_build" "$old_build" "$dst_build" "$old_src" "$dst_src" --check
    sample=$(find "$dst_build/tests" -maxdepth 1 -type f -name '*_tests' -perm -u+x -print -quit 2>/dev/null || true)
    if [ -n "$sample" ]; then
        if command -v readelf >/dev/null; then rp=$(readelf -d "$sample" | grep -E 'RUNPATH|RPATH' || true)
        else rp=$(otool -l "$sample" | grep -A2 LC_RPATH | grep -o 'path [^ ]*' | tr '\n' ' ' || true); fi
        echo "seed: sample $sample: ${rp:-no RUNPATH}"
    fi
fi
