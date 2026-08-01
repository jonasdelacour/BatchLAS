# Detects host BLAS kernels that compute wrong results.
#
# Some OpenBLAS builds ship CPU-dispatch kernels that are simply broken on the
# machine that auto-detection picks them for. The known case is OpenBLAS 0.3.20
# (Ubuntu 22.04 / Debian) selecting its "Cooperlake" dgemm kernel on Sapphire
# Rapids parts such as the Xeon w5-2445: dgemm then returns results that are off
# by O(1) - O(100) at many sizes, while sgemm is fine. Everything layered on top
# (eigensolvers, Ritz values, residual checks) silently inherits the garbage.
#
# OpenBLAS reads OPENBLAS_CORETYPE in its library constructor, which runs before
# main(), so a program cannot repair this for itself at runtime. The only usable
# lever is the environment of the process. This module therefore:
#
#   1. compiles and runs a small dgemm self-check against the detected BLAS,
#   2. if it fails, searches OPENBLAS_CORETYPE candidates for one that is correct,
#   3. publishes the result in BATCHLAS_OPENBLAS_CORETYPE so the build can put it
#      in the environment of anything it launches (see tests/CMakeLists.txt), and
#   4. tells the user the exact export line needed for their own binaries.

option(BATCHLAS_CHECK_BLAS_HEALTH "Verify at configure time that the host BLAS computes dgemm correctly" ON)

set(BATCHLAS_OPENBLAS_CORETYPE "" CACHE STRING
    "OPENBLAS_CORETYPE value needed to avoid a broken OpenBLAS kernel. Empty means none needed. Set explicitly to override detection.")

# OK | WORKAROUND | BROKEN | UNKNOWN (not checked / probe unusable)
set(BATCHLAS_BLAS_HEALTH_STATUS "UNKNOWN" CACHE INTERNAL "Result of the host BLAS dgemm health check")

# Tried in order; earlier entries keep more performance than later ones.
set(_batchlas_coretype_candidates SKYLAKEX HASWELL SANDYBRIDGE NEHALEM PRESCOTT)

function(_batchlas_write_dgemm_check out_file)
    file(WRITE "${out_file}" [==[
/* Returns 0 when dgemm agrees with a naive reference at every size, 1 otherwise. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void cblas_dgemm(int order, int transa, int transb, int m, int n, int k,
                 double alpha, const double* a, int lda, const double* b, int ldb,
                 double beta, double* c, int ldc);

int main(void) {
    const int sizes[] = {64, 128, 200, 256, 512};
    const int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int bad = 0;

    for (int s = 0; s < n_sizes; ++s) {
        const int n = sizes[s];
        double* a = (double*)malloc(sizeof(double) * (size_t)n * n);
        double* b = (double*)malloc(sizeof(double) * (size_t)n * n);
        double* c = (double*)calloc((size_t)n * n, sizeof(double));
        if (!a || !b || !c) { free(a); free(b); free(c); return 2; }

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                a[row + (size_t)col * n] = sin(0.5 * (row + 1) * (col + 1));
                b[row + (size_t)col * n] = cos(0.25 * (row + 1) * (col + 2));
            }
        }

        /* 102 = CblasColMajor, 111 = CblasNoTrans */
        cblas_dgemm(102, 111, 111, n, n, n, 1.0, a, n, b, n, 0.0, c, n);

        double worst = 0.0;
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                double want = 0.0;
                for (int i = 0; i < n; ++i) {
                    want += a[row + (size_t)i * n] * b[i + (size_t)col * n];
                }
                const double diff = fabs(want - c[row + (size_t)col * n]);
                if (diff > worst) worst = diff;
            }
        }
        /* Rounding differences are ~1e-13 here; a broken kernel is off by O(1)+. */
        if (worst > 1e-6) {
            fprintf(stderr, "dgemm wrong at n=%d (max abs error %g)\n", n, worst);
            bad = 1;
        }
        free(a); free(b); free(c);
    }
    return bad;
}
]==])
endfunction()

# Runs the compiled checker, optionally forcing OPENBLAS_CORETYPE.
# Sets ${result_var} to TRUE when dgemm is correct.
function(_batchlas_run_dgemm_check exe coretype result_var)
    set(_env_args "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        # Ignore any OPENBLAS_CORETYPE the developer happens to have exported,
        # otherwise a broken default would be masked during detection.
        list(APPEND _env_args --unset=OPENBLAS_CORETYPE)
    endif()
    if(NOT coretype STREQUAL "")
        list(APPEND _env_args "OPENBLAS_CORETYPE=${coretype}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_env_args} "${exe}"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET
        ERROR_QUIET
        TIMEOUT 120)

    if(_rc EQUAL 0)
        set(${result_var} TRUE PARENT_SCOPE)
    else()
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(batchlas_check_blas_health blas_libraries)
    if(NOT BATCHLAS_CHECK_BLAS_HEALTH)
        return()
    endif()
    if(NOT blas_libraries)
        return()
    endif()
    if(BATCHLAS_BLAS_HEALTH_CHECKED)
        # Already resolved in an earlier configure. Re-state an active workaround
        # so it does not silently disappear from view.
        if(NOT BATCHLAS_OPENBLAS_CORETYPE STREQUAL "")
            message(STATUS "BLAS health check: host BLAS needs OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE} (cached); ctest sets it automatically")
        endif()
        return()
    endif()

    set(_scratch "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/batchlas_blas_check")
    set(_src "${_scratch}/dgemm_check.c")
    set(_exe "${_scratch}/dgemm_check")
    file(MAKE_DIRECTORY "${_scratch}")
    _batchlas_write_dgemm_check("${_src}")

    # Build as C with plain optimisation flags: the project's CXX flags carry SYCL
    # options that are irrelevant (and costly) for this probe.
    enable_language(C)
    try_compile(_built
        "${_scratch}/build" "${_src}"
        COMPILE_DEFINITIONS ""
        LINK_LIBRARIES ${blas_libraries} m
        CMAKE_FLAGS "-DCMAKE_C_FLAGS=-O2"
        COPY_FILE "${_exe}"
        OUTPUT_VARIABLE _build_log)

    if(NOT _built)
        message(STATUS "BLAS health check: could not build the dgemm probe; skipping")
        set(BATCHLAS_BLAS_HEALTH_STATUS "UNKNOWN" CACHE INTERNAL "" FORCE)
        set(BATCHLAS_BLAS_HEALTH_CHECKED TRUE CACHE INTERNAL "")
        return()
    endif()

    # An explicit user setting wins; we only verify it.
    if(NOT BATCHLAS_OPENBLAS_CORETYPE STREQUAL "")
        _batchlas_run_dgemm_check("${_exe}" "${BATCHLAS_OPENBLAS_CORETYPE}" _ok)
        if(_ok)
            message(STATUS "BLAS health check: dgemm correct with user-specified OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
            set(BATCHLAS_BLAS_HEALTH_STATUS "WORKAROUND" CACHE INTERNAL "" FORCE)
        else()
            message(WARNING "BLAS health check: dgemm is STILL WRONG with the requested OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}")
            set(BATCHLAS_BLAS_HEALTH_STATUS "BROKEN" CACHE INTERNAL "" FORCE)
        endif()
        set(BATCHLAS_BLAS_HEALTH_CHECKED TRUE CACHE INTERNAL "")
        return()
    endif()

    _batchlas_run_dgemm_check("${_exe}" "" _ok)
    if(_ok)
        message(STATUS "BLAS health check: host dgemm is correct")
        set(BATCHLAS_BLAS_HEALTH_STATUS "OK" CACHE INTERNAL "" FORCE)
        set(BATCHLAS_BLAS_HEALTH_CHECKED TRUE CACHE INTERNAL "")
        return()
    endif()

    # Broken. Look for a kernel that works.
    set(_working "")
    foreach(_cand IN LISTS _batchlas_coretype_candidates)
        _batchlas_run_dgemm_check("${_exe}" "${_cand}" _cand_ok)
        if(_cand_ok)
            set(_working "${_cand}")
            break()
        endif()
    endforeach()

    if(_working STREQUAL "")
        message(WARNING
            "BLAS health check: the host BLAS computes dgemm INCORRECTLY and no "
            "OPENBLAS_CORETYPE value was found that fixes it. Double-precision "
            "results from the NETLIB/host backend cannot be trusted. Upgrade or "
            "replace the BLAS library.")
        set(BATCHLAS_BLAS_HEALTH_STATUS "BROKEN" CACHE INTERNAL "" FORCE)
        set(BATCHLAS_BLAS_HEALTH_CHECKED TRUE CACHE INTERNAL "")
        return()
    endif()

    set(BATCHLAS_OPENBLAS_CORETYPE "${_working}" CACHE STRING
        "OPENBLAS_CORETYPE value needed to avoid a broken OpenBLAS kernel." FORCE)
    set(BATCHLAS_BLAS_HEALTH_STATUS "WORKAROUND" CACHE INTERNAL "" FORCE)
    set(BATCHLAS_BLAS_HEALTH_CHECKED TRUE CACHE INTERNAL "")

    message(WARNING
        "The host BLAS computes dgemm INCORRECTLY with its auto-detected kernel "
        "(a known OpenBLAS 0.3.20 'Cooperlake' defect on recent Intel parts). "
        "OPENBLAS_CORETYPE=${_working} produces correct results and will be set "
        "for all tests run through ctest.\n"
        "Anything you launch yourself needs it too:\n"
        "    export OPENBLAS_CORETYPE=${_working}\n"
        "or source the generated ${CMAKE_BINARY_DIR}/batchlas-env.sh. "
        "The real fix is a newer OpenBLAS; re-run CMake afterwards and this "
        "workaround will disappear (delete BATCHLAS_OPENBLAS_CORETYPE from the "
        "cache to force a fresh probe).")
endfunction()

# Writes a small shell snippet users can source so their own runs inherit the
# workaround. Always generated so the path is stable; it is a no-op when healthy.
function(batchlas_write_env_script)
    if(NOT BATCHLAS_OPENBLAS_CORETYPE STREQUAL "")
        file(WRITE "${CMAKE_BINARY_DIR}/batchlas-env.sh"
             "# Works around a broken host BLAS kernel detected at configure time.\n"
             "export OPENBLAS_CORETYPE=${BATCHLAS_OPENBLAS_CORETYPE}\n")
    elseif(BATCHLAS_BLAS_HEALTH_STATUS STREQUAL "OK")
        file(WRITE "${CMAKE_BINARY_DIR}/batchlas-env.sh"
             "# Host BLAS dgemm was verified correct at configure time; nothing to set.\n")
    elseif(BATCHLAS_BLAS_HEALTH_STATUS STREQUAL "BROKEN")
        file(WRITE "${CMAKE_BINARY_DIR}/batchlas-env.sh"
             "# WARNING: host BLAS dgemm is wrong and no OPENBLAS_CORETYPE fixed it.\n"
             "# Double-precision host-backend results cannot be trusted.\n")
    else()
        file(WRITE "${CMAKE_BINARY_DIR}/batchlas-env.sh"
             "# Host BLAS was not health-checked; nothing to set.\n")
    endif()
endfunction()
