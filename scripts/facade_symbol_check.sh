#!/bin/bash
#
# Did the public entry points actually leave the vendor component?
#
# WP0 S5 moves each public op's DEFINITION out of the vendor TU and into
# src/dispatch/entry_points/. That is not something a diff can confirm -- a
# forwarder left behind, or an instantiation pointing at the wrong template,
# still compiles and links. So this asks the object files instead:
#
#   batchlas::<op><Backend::CUDA, float>            must be ABSENT from the
#                                                   cuBLAS component and
#                                                   PRESENT in the facade one
#   batchlas::backend::<op>_vendor<CUDA, float>     must still be in the
#                                                   cuBLAS component
#
# MANGLED NAMES, NOT DEMANGLED. `nm -C` cannot demangle a concept-constrained
# template -- symm/syrk/syr2k take RealScalar and hemm/herk/her2k take
# ComplexScalar -- and silently leaves those symbols mangled instead. Matching
# the demangled spelling therefore reports the constrained ops as missing when
# they are present, which is exactly the false alarm this note exists to
# prevent. Everything below matches the Itanium mangling directly.
#
# Usage:  scripts/facade_symbol_check.sh [op ...]

set -u
cd "$(dirname "$0")/.." || exit 1

CUDA_SO=build/src/libbatchlas_backends_cuda.so
FACADE_SO=build/src/libbatchlas_backends.so

for f in "$CUDA_SO" "$FACADE_SO"; do
    [ -f "$f" ] || { echo "error: $f not found; build first"; exit 2; }
done

OPS=${*:-"gemm gemv trsm trmm symm syrk syr2k hemm herk her2k"}

# _ZN8batchlas<len><op>ILNS_7BackendE<n>E...   -- <n> is 1 for CUDA.
mangled_public() { printf '_ZN8batchlas%d%sILNS_7BackendE1E' "${#1}" "$1"; }
# ...and the vendor one sits in namespace backend.
mangled_vendor() {
    local v="${1}_vendor"
    printf '_ZN8batchlas7backend%d%sILNS_7BackendE1E' "${#v}" "$v"
}

rc=0
printf '%-8s %-14s %-14s %-14s %s\n' op cuBLAS facade vendor-impl verdict
for op in $OPS; do
    pub=$(mangled_public "$op")
    ven=$(mangled_vendor "$op")
    in_cuda=$(nm --defined-only "$CUDA_SO"   | grep -c "$pub")
    in_facade=$(nm --defined-only "$FACADE_SO" | grep -c "$pub")
    vendor_impl=$(nm --defined-only "$CUDA_SO" | grep -c "$ven")

    if [ "$in_cuda" -eq 0 ] && [ "$in_facade" -ge 1 ] && [ "$vendor_impl" -ge 1 ]; then
        verdict=OK
    else
        verdict="FAIL"
        rc=1
    fi
    printf '%-8s %-14s %-14s %-14s %s\n' "$op" "$in_cuda" "$in_facade" "$vendor_impl" "$verdict"
done

if [ "$rc" -ne 0 ]; then
    echo
    echo "Expected: 0 in the cuBLAS component, >=1 in the facade, >=1 vendor impl."
fi
exit $rc
