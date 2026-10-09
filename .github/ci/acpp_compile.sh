#!/bin/sh
# GPU-less AdaptiveCpp compile gate: configure BatchLAS with acpp, then build every target.
#
#   .github/ci/acpp_compile.sh <acpp-prefix> <build-dir> [extra cmake args...]
#
# --acpp-targets=generic compiles device code to target-independent IR, so the library, the
# tests, the benchmarks and tools/tune all compile without a GPU or a CUDA toolkit. A
# DPC++-only spelling (sycl::ext::oneapi::*, queue::get_backend(), backend::ext_oneapi_cuda,
# ext_oneapi_submit_barrier, kernel_bundle queries...) outside the impl seams is a compile
# error here and nowhere else in CI. CUDA is off, so code under BATCHLAS_HAS_CUDA_BACKEND
# (the vendor TUs) is not compiled; docs/ci.md lists the gap.
#
# JOBS sets the parallelism (default: nproc). Nothing is run, only compiled and linked.
set -eu

prefix=${1:?usage: acpp_compile.sh <acpp-prefix> <build-dir> [cmake args...]}
build=${2:?usage: acpp_compile.sh <acpp-prefix> <build-dir> [cmake args...]}
shift 2
jobs=${JOBS:-$(nproc)}
src=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$build"
build=$(CDPATH= cd -- "$build" && pwd)
log="$build/acpp-configure.log"

# The acpp-tests preset with the compiler moved to this install. Release: a compile gate needs
# no debug info, and -g doubles the SSCP objects. acpp only warns on a raw DPC++ attribute
# ([[intel::max_work_group_size]], [[sycl::reqd_sub_group_size]]) outside kernel_attrs.hh.
cd "$src"
cmake --preset acpp-tests -B "$build" \
    -DCMAKE_CXX_COMPILER="$prefix/bin/acpp" \
    -DCMAKE_CXX_FLAGS=-Werror=unknown-attributes \
    -DCMAKE_BUILD_TYPE=Release \
    -DBATCHLAS_SYCL_IMPL=ACPP \
    -DBATCHLAS_ENABLE_CUDA=OFF \
    -DBATCHLAS_BUILD_BENCHMARKS=ON \
    "$@" > "$log" 2>&1 || { cat "$log"; exit 1; }
cat "$log"

# A configure that silently lost a component compiles less and stays green. Each tell below is
# a line the corresponding module prints.
fail=0
need() {
    grep -q "$1" "$log" || { echo "::error::acpp configure: no '$1' ($2)"; fail=1; }
}
need 'Using AdaptiveCpp targets: generic' "not the acpp generic flow"
need 'Using AdaptiveCpp for SYCL:' "the DPC++ module ran instead"
need 'Found LAPACKE:' "no host backend, so the NETLIB TUs would not compile"
grep -q '^BATCHLAS_SYCL_IMPL_RESOLVED:INTERNAL=ACPP$' "$build/CMakeCache.txt" \
    || { echo "::error::BATCHLAS_SYCL_IMPL_RESOLVED is not ACPP"; fail=1; }
[ "$fail" -eq 0 ] || exit 1

cmake --build "$build" -j "$jobs"

# `all` must have produced every registered test executable: one that fell out of `all` was not
# compiled, and nothing runs the tests here to notice.
python3 -I "$src/.github/ci/check_built_tests.py" "$build"
