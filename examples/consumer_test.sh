#!/usr/bin/env bash
#
# Install BatchLAS to a throwaway prefix and consume it the way an outside
# project would: a standalone CMake project, find_package(BatchLAS CONFIG
# REQUIRED), build, run, check the numbers.
#
# Nothing here looks at the BatchLAS source or build tree except to install
# from it. Everything the example needs has to come out of the install prefix,
# which is the only way a packaging regression gets caught before a user hits
# it. Run it by hand exactly as CTest does:
#
#   examples/consumer_test.sh --build-dir build --compiler /opt/dpcpp-cuda/bin/clang++
#
# Exit codes: 0 pass, 1 fail, 77 skip (CTest SKIP_RETURN_CODE).

set -uo pipefail

SKIP=77

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_dir=""
example_dir="${script_dir}/consumer"
compiler="${BATCHLAS_CONSUMER_CXX_COMPILER:-${CXX:-}}"
cmake_bin="${CMAKE_COMMAND:-cmake}"
work_dir=""
keep=0

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}"
    echo
    echo "options: --build-dir DIR --example-dir DIR --compiler PATH --cmake PATH --work-dir DIR --keep"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)   build_dir="$2"; shift 2 ;;
        --example-dir) example_dir="$2"; shift 2 ;;
        --compiler)    compiler="$2"; shift 2 ;;
        --cmake)       cmake_bin="$2"; shift 2 ;;
        --work-dir)    work_dir="$2"; shift 2 ;;
        --keep)        keep=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "consumer_test: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

say()  { printf '[consumer] %s\n' "$*"; }
# On failure the logs are the whole story, so stop the cleanup trap from taking
# them with it.
fail() { keep=1; printf '[consumer] FAIL: %s\n' "$*" >&2; exit 1; }
skip() { printf '[consumer] SKIP: %s\n' "$*"; exit "${SKIP}"; }

# ---------------------------------------------------------------------------
# Phase 0: prerequisites. Missing tools or an unbuilt tree mean this test has
# nothing to say, so skip; only an actual packaging defect may fail it.
# ---------------------------------------------------------------------------
command -v "${cmake_bin}" >/dev/null 2>&1 || skip "cmake not found (${cmake_bin})"
[[ -n "${build_dir}" ]] || skip "no --build-dir given"
[[ -f "${build_dir}/cmake_install.cmake" ]] || skip "no configured BatchLAS build tree at ${build_dir}"
[[ -f "${example_dir}/CMakeLists.txt" ]] || fail "no consumer example at ${example_dir}"

if [[ -z "${compiler}" ]]; then
    skip "no SYCL compiler given (--compiler / BATCHLAS_CONSUMER_CXX_COMPILER / CXX)"
fi
[[ -x "${compiler}" ]] || skip "compiler ${compiler} is not executable"

# The install step copies build products; if nothing was built, that is a
# missing prerequisite rather than a packaging bug.
if ! compgen -G "${build_dir}/src/libbatchlas*.so" >/dev/null && \
   ! compgen -G "${build_dir}/src/libbatchlas*.a" >/dev/null; then
    skip "BatchLAS is not built in ${build_dir} (run the build first)"
fi

if [[ -z "${work_dir}" ]]; then
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/batchlas-consumer.XXXXXX")" || fail "mktemp failed"
    created_work_dir=1
else
    mkdir -p "${work_dir}" || fail "cannot create ${work_dir}"
    created_work_dir=0
fi
cleanup() {
    if [[ ${keep} -eq 0 && ${created_work_dir} -eq 1 ]]; then
        rm -rf "${work_dir}"
    else
        say "artifacts kept in ${work_dir}"
    fi
}
trap cleanup EXIT

prefix="${work_dir}/prefix"
log_dir="${work_dir}/logs"
mkdir -p "${prefix}" "${log_dir}"

# The DPC++ runtime lives next to the compiler and is not on any RUNPATH: the
# installed libraries record DT_NEEDED libsycl.so.9 and nothing that finds it.
compiler_bin_dir="$(cd "$(dirname "${compiler}")" && pwd)"
sycl_lib_dir="$(cd "${compiler_bin_dir}/.." && pwd)/lib"

say "build tree : ${build_dir}"
say "compiler   : ${compiler}"
say "prefix     : ${prefix}"

# ---------------------------------------------------------------------------
# Phase 1: install to the throwaway prefix.
# ---------------------------------------------------------------------------
if ! "${cmake_bin}" --install "${build_dir}" --prefix "${prefix}" >"${log_dir}/install.log" 2>&1; then
    tail -n 40 "${log_dir}/install.log" >&2
    fail "cmake --install failed (full log: ${log_dir}/install.log)"
fi
say "installed ($(find "${prefix}" -type f | wc -l) files)"

# ---------------------------------------------------------------------------
# Phase 2: the package must be findable, and must not squat the consumer's
# include root. Generic top-level directories in <prefix>/include collide with
# every other package under a shared prefix such as /usr/local, and shadow the
# consumer's own headers of the same name. Everything belongs under
# <prefix>/include/batchlas.
# ---------------------------------------------------------------------------
[[ -f "${prefix}/lib/cmake/BatchLAS/BatchLASConfig.cmake" ]] \
    || fail "no BatchLASConfig.cmake under ${prefix}/lib/cmake/BatchLAS"

squatters=()
for d in blas util internal; do
    [[ -e "${prefix}/include/${d}" ]] && squatters+=("include/${d}")
done
if [[ ${#squatters[@]} -gt 0 ]]; then
    fail "install squats the consumer's include root: ${squatters[*]} -- install BatchLAS's headers under \${CMAKE_INSTALL_INCLUDEDIR}/batchlas instead"
fi
say "include root is clean (no top-level blas/, util/, internal/)"

# The matching positive assertion. A clean include root is also what a
# completely broken install(DIRECTORY) destination produces, so check that the
# headers actually landed where <batchlas/...> will find them -- one from each
# moved directory, both generated headers, and the umbrella.
for header in batchlas/blas/linalg.hh \
              batchlas/blas/enums.hh \
              batchlas/util/workspace.hh \
              batchlas/internal/ormqr_blocked.hh \
              batchlas/backend_config.h \
              batchlas/device_limits.hh \
              batchlas.hh; do
    [[ -f "${prefix}/include/${header}" ]] \
        || fail "missing installed header: include/${header} -- check install(DIRECTORY)/install(FILES) destinations in cmake/BatchLASPackaging.cmake"
done
# The old export root needed <prefix>/include/batchlas as the include dir, so
# the generated headers were installed to include/batchlas/batchlas/ to be
# spelled <batchlas/...>. With the include ROOT exported that is now one level
# too deep, and nothing in-tree would notice.
[[ -e "${prefix}/include/batchlas/batchlas" ]] \
    && fail "headers are double-nested at include/batchlas/batchlas/ -- the generated-header install(FILES) destination was not updated"
say "public headers are installed under include/batchlas/ and reachable as <batchlas/...>"

# ---------------------------------------------------------------------------
# Phase 3: the actual consumer. Configure, build, run, check the numbers.
# ---------------------------------------------------------------------------
consumer_build="${work_dir}/build-consumer"
configure_consumer() {
    # $1 = build dir, rest = extra cmake args
    local dir="$1"; shift
    "${cmake_bin}" -S "${example_dir}" -B "${dir}" \
        -DCMAKE_CXX_COMPILER="${compiler}" \
        -DCMAKE_PREFIX_PATH="${prefix}" \
        "$@" ${BATCHLAS_CONSUMER_EXTRA_CMAKE_ARGS:-}
}

if ! configure_consumer "${consumer_build}" >"${log_dir}/configure.log" 2>&1; then
    tail -n 40 "${log_dir}/configure.log" >&2
    fail "consumer configure failed (full log: ${log_dir}/configure.log)"
fi
if ! "${cmake_bin}" --build "${consumer_build}" --parallel >"${log_dir}/build.log" 2>&1; then
    tail -n 40 "${log_dir}/build.log" >&2
    # If the failure is in the SYCL driver rather than in BatchLAS's headers,
    # -DBATCHLAS_CONSUMER_USE_FSYCL=OFF isolates it: the example submits no
    # kernels of its own and compiles without the flag.
    fail "consumer build failed (full log: ${log_dir}/build.log)"
fi
say "consumer configured and built"

export LD_LIBRARY_PATH="${sycl_lib_dir}:${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
run_log="${log_dir}/run.log"
"${consumer_build}/hello_batched_gemm" >"${run_log}" 2>&1
run_rc=$?
sed 's/^/[consumer]   /' "${run_log}"

if [[ ${run_rc} -eq ${SKIP} ]]; then
    skip "example reported no usable device"
fi
[[ ${run_rc} -eq 0 ]] || fail "consumer example exited ${run_rc}"
grep -q 'PASS' "${run_log}" || fail "consumer example did not report PASS"
say "consumer ran and the numbers are right"

# ---------------------------------------------------------------------------
# Phase 4: the include-collision probe. Build the same example again with
# consumer-owned blas/enums.hh, util/workspace.hh and internal/ormqr_blocked.hh
# ahead of BatchLAS's headers -- see consumer/decoy_include/util/workspace.hh.
#
# Phase 2 proves the install does not squat the include ROOT. This proves the
# other half: that no installed BatchLAS header reaches for an unprefixed
# <blas/...>, <util/...> or <internal/...> spelling, which would find the
# consumer's file instead. Both halves are needed; the include root was already
# clean while every internal spelling was still unprefixed.
#
# This is a hard assertion now. Anything but a clean build is a failure.
# ---------------------------------------------------------------------------
decoy_build="${work_dir}/build-decoy"
decoy_log="${log_dir}/decoy.log"
decoy_rc=0
if configure_consumer "${decoy_build}" -DBATCHLAS_CONSUMER_DECOY=ON >"${decoy_log}" 2>&1; then
    "${cmake_bin}" --build "${decoy_build}" --parallel >>"${decoy_log}" 2>&1
    decoy_rc=$?
else
    decoy_rc=1
fi

if [[ ${decoy_rc} -eq 0 ]]; then
    say "include-collision probe: CLEAN -- consumer headers named blas/enums.hh, util/workspace.hh and"
    say "                         internal/ormqr_blocked.hh shadow nothing; BatchLAS is reachable only as <batchlas/...>"
elif grep -q 'BATCHLAS_DECOY_[A-Z_]*_SHADOWED' "${decoy_log}"; then
    grep -o 'BATCHLAS_DECOY_[A-Z_]*_SHADOWED' "${decoy_log}" | sort -u | sed 's/^/[consumer]   /'
    tail -n 40 "${decoy_log}" >&2
    fail "REGRESSION: an installed BatchLAS header still spells an unprefixed <blas/...>, <util/...> or <internal/...> include (full log: ${decoy_log})"
else
    tail -n 40 "${decoy_log}" >&2
    fail "include-collision probe failed for an unexpected reason (full log: ${decoy_log})"
fi

say "PASS"
exit 0
