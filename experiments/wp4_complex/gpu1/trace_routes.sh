#!/usr/bin/env bash
# Which SYCL kernel does the vendor-free (native) route actually run, per shape
# class in the measured complex demand? Confirmed, not assumed -- BATCHLAS_GEMM_
# VARIANT does not force a kernel and an unrecognised value silently means
# vendor, so the only trustworthy answer is the trace.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
TR="$D/trace"
mkdir -p "$TR"
run() {
    local tag="$1"; shift
    rm -f "$TR/$tag.json"
    BATCHLAS_KERNEL_TRACE=1 BATCHLAS_KERNEL_TRACE_PATH="$TR/$tag.json" \
    BATCHLAS_GEMM_ROUTE="${ROUTE:-native}" \
        "$D/cx_gemm_bench" "$@" 1 >/dev/null 2>&1
    printf '%-46s %s\n' "$tag [$*]" "$(python3 -c "
import json,collections,sys
try:
    d=json.load(open('$TR/$tag.json'))
except Exception as e:
    print('NO TRACE', e); raise SystemExit
ev=d['traceEvents'] if isinstance(d,dict) else d
print(collections.Counter(e.get('name') for e in ev).most_common(3))
")"
}
run sq512_nn      cfloat  512 512 512 32  N N
run sq512_cn      cfloat  512 512 512 32  C N
run sq256_nn      cfloat  256 256 256 128 N N
run panel_nc      cfloat  512 512 48  128 N C
run panel_nn      cfloat  512 512 48  128 N N
run herk129_cn    cfloat  128 128 48  512 C N
run skinny_nn     cfloat  16  64  16  8192 N N
run sq512_nn_cd   cdouble 512 512 512 16  N N
run sq512_cn_cd   cdouble 512 512 512 16  C N
run panel_nc_cd   cdouble 512 512 48  128 N C
