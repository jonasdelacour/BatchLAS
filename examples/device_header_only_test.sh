#!/usr/bin/env bash
#
# The device API from a plain checkout: `-I <source>/include` and the SYCL flags, nothing
# configured, installed or linked. Three phases, each one a way the claim can break:
#   1. one TU including every public device header compiles and links header-only, and its
#      static_assert sees BATCHLAS_HEADER_ONLY (batchlas/config.hh took the fallback);
#   2. the same TU with the build's generated include dir added must FAIL that static_assert,
#      so phase 1 cannot pass on a generated config it found somewhere (skipped without one);
#   3. every examples/device/*.cc builds the same way, runs, and exits 0.
#
#   examples/device_header_only_test.sh --compiler PATH [--build-include DIR] [--work-dir DIR] \
#       [--keep] -- <SYCL flags...>
#
# Exit codes: 0 pass, 1 fail, 77 skip (CTest SKIP_RETURN_CODE).

set -uo pipefail

SKIP=77
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
compiler=""
build_include=""
work_dir=""
keep=0
flags=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compiler)      compiler="$2"; shift 2 ;;
        --build-include) build_include="$2"; shift 2 ;;
        --work-dir)      work_dir="$2"; shift 2 ;;
        --keep)          keep=1; shift ;;
        --)              shift; flags=("$@"); break ;;
        -h|--help)       sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "device_header_only_test: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

say()  { printf '[header-only] %s\n' "$*"; }
fail() { keep=1; printf '[header-only] FAIL: %s\n' "$*" >&2; exit 1; }
skip() { printf '[header-only] SKIP: %s\n' "$*"; exit "${SKIP}"; }

[[ -n "${compiler}" ]] || skip "no SYCL compiler given (--compiler)"
[[ -x "${compiler}" ]] || skip "compiler ${compiler} is not executable"

if [[ -z "${work_dir}" ]]; then
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/batchlas-header-only.XXXXXX")" || fail "mktemp failed"
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

cxx=("${compiler}" -std=c++20 -O2 -I "${source_dir}/include" "${flags[@]}")
say "compiler : ${cxx[*]}"

# Phase 1: every public device header, header-only.
all_tu="${work_dir}/all_device_headers.cc"
{
    (cd "${source_dir}/include" && find batchlas/device batchlas/device.hh -name '*.hh' \
        -not -path '*/detail/*' | LC_ALL=C sort | sed 's|.*|#include <&>|')
    echo 'static_assert(BATCHLAS_HEADER_ONLY == 1, "batchlas/config.hh used a generated config");'
    echo 'int main() { return 0; }'
} >"${all_tu}"
n_headers="$(grep -c '^#include' "${all_tu}")"
[[ ${n_headers} -gt 20 ]] || fail "found only ${n_headers} device headers under ${source_dir}/include"
"${cxx[@]}" "${all_tu}" -o "${work_dir}/all_device_headers" >"${work_dir}/all.log" 2>&1 || {
    tail -n 40 "${work_dir}/all.log" >&2
    fail "the ${n_headers} public device headers do not compile and link header-only (log: ${work_dir}/all.log)"
}
say "phase 1: ${n_headers} public device headers compile and link with -I include alone"

# Phase 2: the probe must notice a generated config.
if [[ -n "${build_include}" && -f "${build_include}/batchlas/backend_config.h" ]]; then
    if "${cxx[@]}" -I "${build_include}" -fsyntax-only "${all_tu}" >"${work_dir}/leak.log" 2>&1; then
        fail "the BATCHLAS_HEADER_ONLY probe passed with ${build_include} on the path; phase 1 proves nothing"
    fi
    grep -q 'used a generated config' "${work_dir}/leak.log" \
        || { tail -n 20 "${work_dir}/leak.log" >&2; fail "with ${build_include} the TU failed for another reason"; }
    say "phase 2: with the generated config on the path the probe fails, as it must"
else
    say "phase 2: skipped (no --build-include with a generated backend_config.h)"
fi

# Phase 3: the worked examples, built header-only, run and checked.
ran=0
for src in "${source_dir}"/examples/device/*.cc; do
    name="$(basename "${src}" .cc)"
    exe="${work_dir}/${name}"
    "${cxx[@]}" "${src}" -o "${exe}" >"${work_dir}/${name}.build.log" 2>&1 || {
        tail -n 40 "${work_dir}/${name}.build.log" >&2
        fail "${name} does not build header-only (log: ${work_dir}/${name}.build.log)"
    }
    "${exe}" >"${work_dir}/${name}.run.log" 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        tail -n 20 "${work_dir}/${name}.run.log" >&2
        fail "${name} built header-only exited ${rc} (log: ${work_dir}/${name}.run.log)"
    fi
    ran=$((ran + 1))
    say "phase 3: ${name} built header-only and its numbers check out"
done
[[ ${ran} -gt 0 ]] || fail "no examples/device/*.cc found"

say "PASS"
exit 0
