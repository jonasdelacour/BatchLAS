#!/usr/bin/env python3
"""Deliberate breaks for the WP6 LU kernels.

This repository keeps shipping tests that cannot fail by construction -- SIX
recorded instances, one written IN THE SAME CHANGE as the fix it guards. So every
property the verification harness claims to check is corrupted here, the .so is
REBUILT, the harness is re-run, and the outcome recorded. A break that turns
nothing red is the most valuable thing this file can produce.

usage:  break.py apply <name>       # patch the source
        break.py revert <name>      # put it back
        break.py list
"""
import subprocess
import sys
import os

W = "/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan"

BREAKS = {
    # (S-left) The panel's interchanges applied to the columns ALREADY
    # factorised, [0, j0). LAPACK's DLASWP(J-1, A, LDA, J, J+JB-1). Dropping it
    # leaves L's finished columns behind while every later pivot permutes the
    # rows, so P A = L U no longer holds -- but ONLY once there is more than one
    # panel, which is why the n=31 row must stay green and n>=33 must go red.
    "laswp_left": (
        "src/extensions/getrf_blocked.cc",
        "        if (j0 > 0) {\n            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(",
        "        if (false) {\n            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(",
    ),
    # getrs's transposed permutation is P^T, i.e. the SAME list walked BACKWARDS.
    # Walking it forwards computes P instead. Must leave NoTrans untouched.
    # NOTE ON THE ANCHOR, and it is a recorded near-miss: the first version of
    # this entry anchored on the indented call line alone. The NoTrans call and
    # the transposed tail call differ ONLY in indentation, and an 8-space anchor
    # is a SUBSTRING of the 12-space line, so `revert` matched the wrong one and
    # left BOTH walks inverted. It was caught only because the next break's run
    # showed getrs failing for float and double -- types that break could not
    # touch. Anchor on the closing brace, which only the tail call has.
    "getrs_reverse": (
        "src/extensions/getrs_native.cc",
        "/*k1=*/n, /*forward=*/false);\n}",
        "/*k1=*/n, /*forward=*/true);\n}",
    ),
    # The pivot METRIC. LAPACK's I?AMAX uses cabs1 = |Re| + |Im|; the true
    # modulus is a DIFFERENT but equally valid selection rule, so the residual
    # cannot see it and only the elementwise pivot-sequence oracle can. Real
    # types are unaffected by construction, so float/double must stay green.
    "pivot_metric": (
        "src/extensions/getrf_cta_device.hh",
        "        return sycl::fabs(a.re) + sycl::fabs(a.im);",
        "        return sycl::sqrt(a.re * a.re + a.im * a.im);",
    ),
    # THE SHORT FINAL PANEL. Stopping the loop at the last FULL panel drops the
    # trailing ib < nb columns entirely. n = 64, 96, 128 are exact multiples of
    # nb = 32 and must stay green; n = 31, 33, 100 must go red.
    "short_final": (
        "src/extensions/getrf_blocked.cc",
        "    for (int j0 = 0; j0 < n; j0 += nb) {",
        "    for (int j0 = 0; j0 + nb <= n; j0 += nb) {",
    ),
    # getri writes F, whose ones sit at (i, perm[i]). Writing them at
    # (perm[i], i) is F^T = F^-1 -- a permutation matrix either way, so nothing
    # about the SHAPE of C changes and only the residual can see it.
    "getri_perm_t": (
        "src/extensions/getri_blocked.cc",
        "                    Cb[static_cast<std::ptrdiff_t>(r) * ldc + i] =",
        "                    Cb[static_cast<std::ptrdiff_t>(i) * ldc + r] =",
    ),
    # TIMING-ONLY breaks. These produce WRONG ANSWERS by construction (the row
    # interchange IS the op) and exist solely to PRICE it, the way the baseline's
    # BREAK=laswp accidentally did. Never read a residual from a run using them.
    # Each anchor is unique: the getrs pair differ in the `forward` flag and the
    # getrf pair in the ncols expression, which is what the exactly-once check
    # below enforces.
    "getrs_nolaswp": (
        "src/extensions/getrs_native.cc",
        "/*k0=*/0, /*k1=*/n, /*forward=*/true);",
        "/*k0=*/0, /*k1=*/0, /*forward=*/true);",
    ),
    "getrf_nolaswp_left": (
        "src/extensions/getrf_blocked.cc",
        "ctx, a_ptr, ld, stride, /*ncols=*/j0, batch,\n                piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);",
        "ctx, a_ptr, ld, stride, /*ncols=*/j0, batch,\n                piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j0, /*forward=*/true);",
    ),
    "getrf_nolaswp_right": (
        "src/extensions/getrf_blocked.cc",
        "/*ncols=*/n2, batch,\n            piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j2, /*forward=*/true);",
        "/*ncols=*/n2, batch,\n            piv_ptr, /*piv_stride=*/n, /*k0=*/j0, /*k1=*/j0, /*forward=*/true);",
    ),
    # THE TIER WINDOW. Widening double's ceiling to infinity removes the one
    # place native_tier_preferred disagrees with kGetrfOrder, so the vendor-free
    # walk falls back to the order array's CTA-first ladder. It must turn
    # RouteGetrf.NativeTierPreferredIsDeclaredAndPinsTheMeasuredTierChoice red --
    # a window that no test can see is a window nothing guards.
    "tier_window": (
        "include/batchlas/blas/dispatch/route_getrf.hh",
        "                // 0.98 at n=64, 0.85 at n=76, 0.77 at n=96 -- blocked ahead\n                // everywhere the two arms are actually different code.\n                return 32;",
        "                // 0.98 at n=64, 0.85 at n=76, 0.77 at n=96 -- blocked ahead\n                // everywhere the two arms are actually different code.\n                return 1 << 30;",
    ),
    # The row exchange inside the panel leaf covers ALL n columns of the tile,
    # including the ones LEFT of k that already hold finished L. Restricting it
    # to c >= k is the classic silently-wrong LU.
    "leaf_swap_right": (
        "src/extensions/getrf_cta_device.hh",
        "            for (int c = tid; c < n; c += wg) {",
        "            for (int c = k + tid; c < n; c += wg) {",
    ),
}


def patch(name, forward):
    path, old, new = BREAKS[name]
    p = os.path.join(W, path)
    s = open(p).read()
    a, b = (old, new) if forward else (new, old)
    # EXACTLY ONCE, both directions. An anchor that matches twice -- or that is a
    # SUBSTRING of a second, differently indented occurrence -- silently patches
    # the wrong line, which is how the getrs_reverse revert above left BOTH
    # permutation walks inverted. A count check turns that into a hard stop.
    n_here = s.count(a)
    if n_here != 1:
        print(f"FATAL: anchor for {name} matches {n_here} times in {path} "
              f"({'apply' if forward else 'revert'}); it must match exactly once")
        sys.exit(2)
    open(p, "w").write(s.replace(a, b, 1))
    print(f"{'applied' if forward else 'reverted'} {name} in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for k in BREAKS:
            print(k)
        sys.exit(0)
    patch(sys.argv[2], sys.argv[1] == "apply")
