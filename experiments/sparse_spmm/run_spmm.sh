#!/usr/bin/env bash
# WP8 spmm bake-off runner -- ONE route per process, ONE process on the box.
#
# HYGIENE THIS SCRIPT ENFORCES (see README.md for why each one exists):
#   * CUDA_VISIBLE_DEVICES=1 always. Device 0 drives Xorg/gnome-shell (~1.1 GB
#     resident) and has been measured depressing a vendor arm by up to 1.8x on
#     L2-resident cells.
#   * It refuses to start if any OTHER process holds memory on the target GPU,
#     because nvidia-smi --query-compute-apps is PER DEVICE and a sweep on the
#     other card has been measured inflating a cell 5.5x while both cards
#     reported zero foreign compute processes.
#   * One route per process, pinned through BATCHLAS_SPMM_ROUTE, and the route
#     actually resolved is written to a coverage CSV beside every timing CSV --
#     an unrecognised route value parses to nothing and falls back to Auto
#     SILENTLY, so the environment variable is never the evidence.
#   * --min_time is large enough that minibench escalates its inner batch count
#     past the 1 ms floor and the reported stddev is a real run-to-run spread.
#
# usage: run_spmm.sh <route> <tag> <family> <type> <outdir> ARGS...
#   route   vendor | native:direct         (pinned into BATCHLAS_SPMM_ROUTE)
#   tag     free-form; names the CSV
#   family  a --name substring, e.g. BM_SPMM_Grid
#   type    float | double | cfloat | cdouble
#   ARGS    the minibench positional grid; comma lists allowed on every axis
#           m nnzrow nrhs batch transB beta pattern transA
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$D/../.." && pwd)"
BIN="${BIN:-$ROOT/build/benchmarks/spmm_benchmark}"

ROUTE="${1:?route}"; shift
TAG="${1:?tag}"; shift
FAMILY="${1:?family}"; shift
TYPE="${1:?type}"; shift
OUTDIR="${1:?outdir}"; shift

GPU="${GPU:-1}"
mkdir -p "$OUTDIR"

# --- the exclusivity guard -------------------------------------------------
# Any foreign process holding memory on the target card invalidates the run.
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 200 ]; then
  echo "REFUSING: GPU $GPU already holds ${used} MiB -- another harness is live." >&2
  exit 3
fi
mine=$(pgrep -c -f 'spmm_benchmark' || true)
if [ "${mine:-0}" -gt 0 ]; then
  echo "REFUSING: an spmm_benchmark process is already running." >&2
  exit 3
fi

SAFE="${ROUTE//:/_}"
CSV="$OUTDIR/${TAG}_${TYPE}_${SAFE}.csv"
COV="$OUTDIR/${TAG}_${TYPE}_${SAFE}.routes.csv"
LOG="$OUTDIR/${TAG}_${TYPE}_${SAFE}.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export BATCHLAS_SPMM_ROUTE="$ROUTE"
export BATCHLAS_COVERAGE_OUT="$COV"

# Record the clock the run started at, so a cold-clock row can be recognised.
nvidia-smi -i "$GPU" --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$LOG"

# WARM-UP IS A WALL-CLOCK BUDGET IN THE BENCHMARK ITSELF, NOT A CALL COUNT.
# spmm_benchmark.cc's SetPrepare warms each row for BATCHLAS_SPMM_WARM_MS
# (default 400 ms) before minibench sees it. Measured justification, cell L,
# vendor, device 1 (probe/warmup_probe.sh, probe/order_probe.sh):
#   minibench warmup=2   -> first row 0.16544 ms, rel_sd 0.0495
#   minibench warmup=250 -> first row 0.16179 ms, rel_sd 0.0016
#   second row of ANY process                    0.1616-0.1620, rel_sd 0.0015
# i.e. the 210 MHz idle clock costs 2.3% and lands on the first row only, and a
# CALL-counted warm-up cannot price it uniformly (250 calls is 40 ms on a cheap
# cell and 13.5 s on the 54 ms m=4096 cdouble cell). minibench's own warm-up is
# therefore left near its default.
export BATCHLAS_SPMM_WARM_MS="${WARM_MS:-400}"
"$BIN" --name="$FAMILY" --type="$TYPE" \
       --warmup="${WARMUP:-2}" \
       --warmup_internal="${WARMUP_INTERNAL:-2}" \
       --min_time="${MIN_TIME:-400}" \
       --min_iters="${MIN_ITERS:-20}" \
       --csv="$CSV" "$@" >> "$LOG" 2>&1

nvidia-smi -i "$GPU" --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader >> "$LOG"
echo "wrote $CSV"
# coverage.cc:135 writes ONE FILE PER PID, as $BATCHLAS_COVERAGE_OUT.<pid>.
cat "$COV".* > "$COV" 2>/dev/null || true
rm -f "$COV".[0-9]*
echo "route rows: $(grep -c ',spmm,' "$COV" 2>/dev/null || true)"
grep ',spmm,' "$COV" 2>/dev/null | awk -F, '{print "  route: "$10":"$11" transA="$19" transB="$20" calls="$12}' | sort -u
