#!/usr/bin/env bash
# run_bitid.sh <libtree_build_dir> <out.txt>
# Runs every bit-identity config with the harness loading <libtree_build_dir>/src/*.so.
# Writes hashes to <out.txt>, select traces/stderr to <out.txt>.err, the resolved .so paths to
# <out.txt>.libs. Two trees are bit-identical on the covered configs iff `diff` of the two outs is empty.
# Optional: ONLY=<regex> restricts to tags matching it; EXTRA_ENV="K=V ..." is added to every process.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$1" && pwd)"
OUT="$2"
BIN="$HERE/bitid"
[[ -x "$BIN" ]] || { echo "build the harness first: $HERE/build.sh" >&2; exit 2; }
[[ -e "$TREE/src/libbatchlas_core.so.0" ]] || { echo "no libs in $TREE/src" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export ONEAPI_DEVICE_SELECTOR='cuda:*'
export LD_LIBRARY_PATH="$TREE/src:/opt/dpcpp-cuda/lib"
export BATCHLAS_SELECT_TRACE=1
# Clear every knob this script sets, so the caller's environment cannot leak into a config.
for v in $(env | grep -o '^BATCHLAS_[A-Z0-9_]*' | grep -v '^BATCHLAS_SELECT_TRACE$'); do unset "$v"; done

: >"$OUT"; : >"$OUT.err"
{
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    ldd "$BIN" | grep -E 'batchlas|libsycl|libur_'
    echo "--- LD_DEBUG=libs (batchlas lookups)"
    LD_DEBUG=libs "$BIN" none 2>&1 | grep -E 'calling init: .*batchlas' | sort -u
} >"$OUT.libs" 2>&1
if grep 'libbatchlas' "$OUT.libs" | grep -v "$TREE/src" | grep -q '=>'; then
    echo "ERROR: some libbatchlas resolved outside $TREE/src" >&2; exit 3
fi

# Stale-library guard: a .so older than the tree's HEAD commit was not rebuilt for this branch.
HEAD_T=$(git -C "$TREE/.." log -1 --format=%ct HEAD 2>/dev/null || echo 0)
for so in "$TREE"/src/libbatchlas_*.so.0.1.0; do
    if [[ $(stat -c %Y "$so") -lt $HEAD_T ]]; then
        echo "WARN stale: $(basename "$so") older than HEAD $(git -C "$TREE/.." log -1 --format=%h HEAD)" | tee -a "$OUT.libs" >&2
    fi
done

run() {  # run <tag> [ENV=V ...] -- <group> [key=val ...]
    local tag="$1"; shift
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift
    if [[ -n "${ONLY:-}" ]] && ! [[ "$tag" =~ $ONLY ]]; then return; fi
    echo "### $tag ${envs[*]} :: $*" >>"$OUT.err"
    # shellcheck disable=SC2086
    env "${envs[@]}" ${EXTRA_ENV:-} "$BIN" "$@" "tag=$tag" >>"$OUT" 2>>"$OUT.err"
    local rc=$?
    [[ $rc -eq 0 ]] || echo "PROCESS_FAIL $tag $* rc=$rc" >>"$OUT"
}

B=batch=1024
# 1. potrf
run potrf:cta            BATCHLAS_POTRF_ROUTE=cta              -- potrf n=16,48,96 $B
run potrf:lpanel         BATCHLAS_POTRF_ROUTE=lpanel:panel=8   -- potrf n=16,32,64 $B
run potrf:lpanel16       BATCHLAS_POTRF_ROUTE=lpanel:panel=16  -- potrf n=16,32,64 types=f $B
run potrf:blocked        BATCHLAS_POTRF_ROUTE=blocked          -- potrf n=256 $B
# 2. potri
run potri:cta            BATCHLAS_POTRI_ROUTE=cta              -- potri n=8,16,32,48 $B
run potri:blocked64      BATCHLAS_POTRI_ROUTE=blocked:nb=64    -- potri n=64,130 $B
run potri:blocked128     BATCHLAS_POTRI_ROUTE=blocked:nb=128   -- potri n=160 types=f $B
# 3. geqrf (+ orgqr on its output), forced panel leaves, orgqr on vendor-geqrf input
run geqrf:tiny           BATCHLAS_GEQRF_ROUTE=tiny             -- geqrf shapes=8x8,16x16,32x32 $B
run geqrf:cta            BATCHLAS_GEQRF_ROUTE=cta              -- geqrf shapes=16x16,32x32,48x48,64x32,64x64 $B
run geqrf:blocked        BATCHLAS_GEQRF_ROUTE=blocked          -- geqrf shapes=64x64,128x96,256x64,96x128 $B
run geqrf:blocked+LEAFreg BATCHLAS_GEQRF_ROUTE=blocked BATCHLAS_GEQRF_LEAF=reg -- geqrf shapes=64x64,128x96,256x64 $B
run geqrf_leaf           BATCHLAS_GEMM_ROUTE=                  -- geqrf_leaf shapes=96x32,128x64,512x64 $B
run orgqr:blocked        BATCHLAS_GEQRF_ROUTE=vendor BATCHLAS_ORGQR_ROUTE=blocked -- orgqr shapes=64x64,96x48,128x128 $B
# 4. ormqr: CTA kernel (ormqx_cta), blocked with and without the device impl (latched per process)
run ormqx_cta            BATCHLAS_ORMQR_ROUTE=                 -- ormqx_cta n=5,13,16,29,32 $B
run ormqr:blocked        BATCHLAS_ORMQR_ROUTE=blocked          -- ormqr shapes=96x64,48x48,130x33 $B
run ormqr:blocked+IMPLdevice BATCHLAS_ORMQR_ROUTE=blocked BATCHLAS_ORMQR_IMPL=device -- ormqr shapes=96x64,48x48,130x33 $B
# 5. syev
run syev:cta             BATCHLAS_SYEV_ROUTE=cta               -- syev n=5,8,16,32 $B
run syev:cta_fused       BATCHLAS_SYEV_ROUTE=cta_fused         -- syev n=5,8,16,32 $B
run syev:blocked         BATCHLAS_SYEV_ROUTE=blocked           -- syev n=64,150 $B
run tridiag_solver       BATCHLAS_SYEV_ROUTE=                  -- tridiag n=16,32,64,150 $B
# 6. stebz + stein, syevx direct subset (stebz + stein path)
run stebz_stein          BATCHLAS_SYEV_ROUTE=                  -- stebz_stein n=16,64,150 $B
run syevx_subset         BATCHLAS_SYEV_ROUTE=                  -- syevx n=32,96 types=f,d $B
# 7. gesvd (blocked, both bidiagonal solvers), Hermitian overload, gebrd variants
GS=shapes=16x16,32x32,64x64,29x16,45x32,77x64,16x29,32x45,64x77
run gesvd:blocked+bdsdc  BATCHLAS_GESVD_ROUTE=blocked BATCHLAS_GESVD_BIDIAG=bdsdc -- gesvd $GS $B
run gesvd:blocked+bdsqr  BATCHLAS_GESVD_ROUTE=blocked BATCHLAS_GESVD_BIDIAG=bdsqr -- gesvd $GS $B
run gesvd:herm+blocked   BATCHLAS_GESVD_ROUTE=blocked          -- gesvd herm=1 shapes=16x16,32x32,64x64 jobs=A,N $B
run gebrd                BATCHLAS_GESVD_ROUTE=                 -- gebrd shapes=16x16,32x32,29x16,16x29,64x64,77x64,64x77 $B
echo "done: $(grep -c . "$OUT") lines, $(grep -c ' THROW ' "$OUT") THROW, $(grep -c PROCESS_FAIL "$OUT") PROCESS_FAIL -> $OUT"
