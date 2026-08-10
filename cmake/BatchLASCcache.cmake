# Compilation caching.
#
# ccache does cache -fsycl fat objects — including the nvptx64 and native_cpu
# device images — and it reproduces them byte for byte. A cached TU costs 0.02 s
# instead of ~15 s. Two settings decide whether that is a win or a loss, and
# neither of them is ccache's default, which is why this file generates a
# launcher wrapper rather than just pointing CMAKE_CXX_COMPILER_LAUNCHER at
# ccache:
#
#   depend_mode  In its default preprocessor mode ccache runs an extra full
#                preprocess to compute the hash. Under -fsycl the driver
#                preprocesses once per compilation phase, so a cache MISS costs
#                +52 % (14.9 s -> 22.7 s on stedc.cc). That makes a cold build —
#                a fresh clone, a new worktree, a rebase — materially slower than
#                no ccache at all. Depend mode hashes the compiler's own -MD
#                depfile instead, so a miss is free. CMake always emits -MD, for
#                both the Makefile and Ninja generators.
#
#   base_dir +   Without these, ccache hashes absolute paths, so a build in one
#   hash_dir     worktree never hits the cache populated by another. That is the
#                whole explanation for a cache that shows 0 hits after hundreds
#                of compiles. With them, a brand-new worktree builds from cache.
#
# What this does NOT speed up: the dev loop. Editing a .cc is a miss by
# definition (15.2 s either way), and editing a header misses every dependent.
# The payoff is switching branches, reverting an experiment, `make clean` in the
# same tree, and configuring a new worktree at an already-built commit. Do not
# count it towards any edit-build-test number. Note also that the launcher only
# wraps compiles -- CMAKE_CXX_COMPILER_LAUNCHER does not apply to link steps, and
# links are ~213 s of a 214 s all-cache-hits rebuild, so that is the floor.
#
# Two caveats worth knowing before you debug:
#
#   Stale hits are possible, though rare. If you add a header that *shadows* an
#   existing one earlier on the -I path, ccache's manifest-based header tracking
#   returns the old object. This is not a depend-mode bug -- default mode gets it
#   equally wrong. It surfaces as a wrong test result, so if you have just added
#   a header and something inexplicable fails, `ccache -C` before believing it.
#
#   hash_dir=false means a restored object carries the DW_AT_comp_dir of
#   whichever tree first compiled it. The code is identical (a hit requires
#   identical source), but if that tree has since been deleted the debugger will
#   fail to find sources. Set BATCHLAS_CCACHE_SHARE_ACROSS_TREES=OFF if you are
#   doing source-level debugging and would rather have paths than cache hits.

option(BATCHLAS_USE_CCACHE "Cache C++ compilations with ccache when it is available" ON)
option(BATCHLAS_CCACHE_SHARE_ACROSS_TREES
    "Let separate checkouts/worktrees share cache entries (see DW_AT_comp_dir caveat)" ON)
set(BATCHLAS_CCACHE_BASEDIR "$ENV{HOME}" CACHE PATH
    "Paths below this are hashed relative, so sibling checkouts share cache entries")
set(BATCHLAS_CCACHE_MAXSIZE "20G" CACHE STRING "ccache size limit")

if(NOT BATCHLAS_USE_CCACHE)
    return()
endif()

# Only when BatchLAS is the top-level project. Everything below writes global
# state - a generated script in the binary dir and a CACHE ... FORCE write to
# CMAKE_CXX_COMPILER_LAUNCHER - and under add_subdirectory()/FetchContent that
# would silently reroute the *consuming* project's compiler through our wrapper.
# This file runs before BatchLASOptions.cmake, so BATCHLAS_IS_TOP_LEVEL does not
# exist yet; compare the directories directly.
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "BatchLAS is a subproject: leaving CMAKE_CXX_COMPILER_LAUNCHER to the parent project")
    return()
endif()

if(CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER already set - leaving it alone: ${CMAKE_CXX_COMPILER_LAUNCHER}")
    return()
endif()

find_program(BATCHLAS_CCACHE_PROGRAM ccache)
if(NOT BATCHLAS_CCACHE_PROGRAM)
    message(STATUS "ccache not found - compilations will not be cached")
    return()
endif()

# The settings above have to reach ccache as environment variables, because a
# developer's ~/.config/ccache/ccache.conf is not ours to edit and a build that
# silently falls back to preprocessor mode is slower than no cache at all. A
# generated wrapper is the only way to guarantee they are set for every compile.
set(_batchlas_ccache_launcher "${CMAKE_BINARY_DIR}/batchlas-ccache")
set(_batchlas_ccache_env
    "export CCACHE_DEPEND=1\nexport CCACHE_MAXSIZE=${BATCHLAS_CCACHE_MAXSIZE}\n")
if(BATCHLAS_CCACHE_SHARE_ACROSS_TREES)
    string(APPEND _batchlas_ccache_env
        "export CCACHE_BASEDIR=${BATCHLAS_CCACHE_BASEDIR}\nexport CCACHE_NOHASHDIR=1\n")
endif()

# Written via a staging directory and file(COPY ... FILE_PERMISSIONS) rather than
# file(CHMOD), which needs CMake 3.19 while this project declares 3.14.
file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/batchlas-ccache-stage/batchlas-ccache"
    "#!/bin/sh\n"
    "# Generated by cmake/BatchLASCcache.cmake - do not edit.\n"
    "${_batchlas_ccache_env}"
    "exec \"${BATCHLAS_CCACHE_PROGRAM}\" \"$@\"\n"
)
file(COPY "${CMAKE_BINARY_DIR}/CMakeFiles/batchlas-ccache-stage/batchlas-ccache"
    DESTINATION "${CMAKE_BINARY_DIR}"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(CMAKE_CXX_COMPILER_LAUNCHER "${_batchlas_ccache_launcher}" CACHE STRING "" FORCE)

if(BATCHLAS_CCACHE_SHARE_ACROSS_TREES)
    message(STATUS "ccache enabled (depend mode, shared below ${BATCHLAS_CCACHE_BASEDIR}): ${BATCHLAS_CCACHE_PROGRAM}")
else()
    message(STATUS "ccache enabled (depend mode, this tree only): ${BATCHLAS_CCACHE_PROGRAM}")
endif()
