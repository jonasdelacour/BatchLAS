#!/bin/sh
set -eu

# nvprof-like terminal summary for ROCm using rocprofv3.
# Usage:
#   scripts/rocprof_nvprof_like.sh ./benchmarks/gemm_benchmark 128 128 128 32 --backend=ROCM --type=float

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <app> [args...]" >&2
  exit 2
fi

if ! command -v rocprofv3 >/dev/null 2>&1; then
  echo "rocprofv3 not found in PATH. Install ROCm profiling tools (rocm-profiler-sdk)." >&2
  exit 127
fi

# Resolve the target app and improve common UX errors:
# - if user passes a source file (e.g. benchmarks/foo.cc), try build/benchmarks/foo
# - if target is not executable, fail with a clear message
app="$1"
shift

if [ ! -x "$app" ]; then
  case "$app" in
    *.cc)
      script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
      app_base=$(basename -- "$app")
      app_name=${app_base%.cc}
      candidate="$script_dir/../build/benchmarks/$app_name"
      if [ -x "$candidate" ]; then
        echo "[rocprof] Using built benchmark: $candidate" >&2
        app="$candidate"
      fi
      ;;
  esac
fi

if [ ! -x "$app" ]; then
  echo "Error: target is not executable: $app" >&2
  echo "Hint: pass a built binary, e.g. ./build/benchmarks/gemm_benchmark" >&2
  exit 126
fi

# Avoid HIP adapter error spam during normal in-flight polling.
export UR_LOG_HIP="${UR_LOG_HIP:-level:quiet}"

# Keep selector explicit if user did not set it.
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-hip:*}"

# Print a terminal summary similar to nsys nvprof top-kernel breakdown.
exec rocprofv3 --kernel-trace --stats --summary -- "$app" "$@"
