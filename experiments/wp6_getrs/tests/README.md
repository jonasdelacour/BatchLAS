# The fused-getrs blind-guard tooling

These are the scripts behind the FUSED-GETRS BREAK RECORD at the bottom of
`tests/getrf_tests.cc`. Each one corrupts ONE guarded property at its source,
the caller rebuilds the `.so`, and the whole `getrf_tests` binary is re-run.

    break.py   <name>   src/extensions/getrs_fused.cc          -- the kernel and its geometry
    break2.py  <name>   factorization.cc + route_getrs.hh      -- the facade arm and the tier tie-break
    break3.py  <name>   route_getrs.hh + getrs_fused.cc        -- supports()' ceilings and their re-check
    <n>.py     restore                                          -- put every file that script touches back

    runbreak.sh / runbreak2.sh / runbreak3.sh <name>
        patch -> `cmake --build build -j 32 --target getrf_tests` -> run -> print the failing set
    dryrun.sh
        apply every break.py patch in turn WITHOUT building, purely to prove each
        anchor still matches. Run this first after any edit to getrs_fused.cc: a
        break whose anchor has drifted silently becomes a no-op, and a no-op break
        reports "nothing turned red" for the wrong reason.

Every patch asserts its anchor matched an exact number of times, so a drifted
anchor is a hard failure rather than a silent no-op.

TWO THINGS TO KNOW BEFORE RE-RUNNING THEM.

  * THE PATHS ARE ABSOLUTE, and the backup copies live under the scratch
    directory of the session that wrote them
    (`/home/jonaslacour/.claude/jobs/20812aa0/tmp/`). That directory will not
    exist later. Point `TMP` at a fresh scratch directory and the scripts will
    re-take their own pristine copies on the first patch -- FROM WHATEVER IS ON
    DISK, so make sure the tree is clean before the first one runs.

  * `dispatch_gates` TAKES THE PROCESS DOWN (SIGABRT). Run it FILTERED, once per
    scalar type, or the three types after float never execute and the run reports
    nothing about them. This is the same rule `tests/getrf_tests.cc`'s own break
    record states for `short_final` and `piv_stride_nb`.

Nothing here is built by CMake and nothing in the library references it.
