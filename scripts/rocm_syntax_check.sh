#!/bin/bash
#
# Syntax-check the three ROCm vendor TUs, which this machine never compiles.
#
# WHY THIS EXISTS
#
# WP0_DISPATCH_SPEC.md ranks as its number-one risk that S5 (moving the public
# op definitions out of the vendor TUs) edits rocblas.cc / rocsolver.cc /
# rocsparse.cc, and that "neither compiles here". Each op's move has to be
# atomic across every vendor TU or the build gets duplicate or undefined
# symbols -- so an unverifiable edit to three of them is the whole hazard.
#
# It turns out to be verifiable. The edits are declaration- and
# instantiation-level, which is exactly what -fsyntax-only checks, and
# /opt/rocm-6.2.4 on this box carries all three vendor headers (under
# include/roc*/roc*.h -- note the subdirectory, which is why a naive
# `ls /opt/rocm/include/rocblas.h` says they are absent).
#
# THE ONE EXPECTED ERROR
#
# This DPC++ is built for CUDA only, so `sycl::get_native<ext_oneapi_hip>` at
# linalg-impl.hh:1030 has no matching overload. That is a property of the
# toolchain, not of the code, and it is the ONLY error each TU produces. So the
# gate is: exactly that error and nothing else. Any other diagnostic is a real
# defect in the ROCm sources.
#
# Turning the CUDA macros OFF is not incidental -- it is most of what makes this
# a real check, since it exercises the per-library #if structure S2 introduced
# in a configuration nothing else here builds.
#
# Usage:  scripts/rocm_syntax_check.sh          (from the repo root)
# Exit 0 iff all three TUs produce only the expected error.

set -u

CXX="${BATCHLAS_ROCM_SYNTAX_CXX:-/opt/dpcpp-cuda/bin/clang++}"
ROCM="${ROCM_PATH:-/opt/rocm}"
BUILD_INCLUDE="${BATCHLAS_BUILD_INCLUDE:-build/include}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

EXPECTED_ERROR="no matching function for call to 'get_native'"

if [ ! -f "$BUILD_INCLUDE/batchlas/backend_config.h" ]; then
    echo "error: $BUILD_INCLUDE/batchlas/backend_config.h not found."
    echo "       Configure a build first, or set BATCHLAS_BUILD_INCLUDE."
    exit 2
fi

cat > "$WORK/force_rocm.h" <<'EOF'
#include <batchlas/backend_config.h>
/* A ROCm build has no CUDA. */
#undef BATCHLAS_HAS_CUDA_BACKEND
#define BATCHLAS_HAS_CUDA_BACKEND 0
#undef BATCHLAS_HAS_CUBLAS
#define BATCHLAS_HAS_CUBLAS 0
#undef BATCHLAS_HAS_CUSOLVER
#define BATCHLAS_HAS_CUSOLVER 0
#undef BATCHLAS_HAS_CUSPARSE
#define BATCHLAS_HAS_CUSPARSE 0
#undef BATCHLAS_HAS_CUBLASDX
#define BATCHLAS_HAS_CUBLASDX 0
#undef BATCHLAS_HAS_CUSOLVERDX
#define BATCHLAS_HAS_CUSOLVERDX 0
/* ...and does have ROCm. */
#undef BATCHLAS_HAS_ROCM_BACKEND
#define BATCHLAS_HAS_ROCM_BACKEND 1
#undef BATCHLAS_HAS_ROCBLAS
#define BATCHLAS_HAS_ROCBLAS 1
#undef BATCHLAS_HAS_ROCSOLVER
#define BATCHLAS_HAS_ROCSOLVER 1
#undef BATCHLAS_HAS_ROCSPARSE
#define BATCHLAS_HAS_ROCSPARSE 1
#undef BATCHLAS_HAS_ANY_VENDOR_BLAS
#define BATCHLAS_HAS_ANY_VENDOR_BLAS 1
EOF

rc=0

check() {
    local tu="$1" hdr="$2"
    local log="$WORK/$(basename "$tu").log"

    if [ ! -f "$hdr" ]; then
        # Never pass silently: a check that covers nothing is worse than none.
        echo "SKIP  $tu  (missing $hdr)"
        rc=1
        return
    fi

    "$CXX" -fsyntax-only -std=c++20 -fsycl \
        -Iinclude -I"$BUILD_INCLUDE" -Isrc -I"$ROCM/include" \
        -include "$WORK/force_rocm.h" \
        -Wno-unused-command-line-argument \
        "$tu" > "$log" 2>&1

    local total unexpected
    total=$(grep -c "error:" "$log")
    unexpected=$(grep "error:" "$log" | grep -vc "$EXPECTED_ERROR")

    if [ "$unexpected" -eq 0 ]; then
        echo "PASS  $tu  ($total expected error(s), 0 unexpected)"
    else
        echo "FAIL  $tu  ($unexpected unexpected error(s) of $total)"
        grep "error:" "$log" | grep -v "$EXPECTED_ERROR" | head -20
        rc=1
    fi
}

check src/backends/rocblas.cc   "$ROCM/include/rocblas/rocblas.h"
check src/backends/rocsolver.cc "$ROCM/include/rocsolver/rocsolver.h"
check src/backends/rocsparse.cc "$ROCM/include/rocsparse/rocsparse.h"

exit $rc
