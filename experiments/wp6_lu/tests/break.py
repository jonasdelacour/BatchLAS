#!/usr/bin/env python3
"""Corrupt one guarded property of the WP6 LU kernels, or revert it.

    ./break.py <name>          apply
    ./break.py <name> --revert restore

EVERY ANCHOR MUST MATCH EXACTLY ONCE, in both directions. That rule is not
defensive: WP6's kernel-side break tooling patched the WRONG line because an
8-space anchor is a substring of the 12-space line, and left BOTH permutation
walks inverted in the tree. It was caught only because the NEXT break's run
showed failures for types that break could not touch. A break that silently
misses, or silently hits twice, produces a break RESULT that is a lie.

Each entry names the property it corrupts and which test is expected to see it.
"""
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

BREAKS = {
    # ---------------------------------------------------------------- B1
    # THE PIVOT BASE. LAPACK's ipiv is 1-BASED; write it 0-based.
    # Expected: the elementwise pivot oracle and the ||PA - LU|| reconstruction.
    "piv_base_zero": ("src/extensions/getrf_cta_device.hh",
        "            piv_item[k] = piv_base + p + 1;      // GLOBAL, 1-BASED, LAPACK ipiv",
        "            piv_item[k] = piv_base + p;          // BREAK B1: 0-based"),

    # ---------------------------------------------------------------- B2
    # THE SWAP DIRECTION in the transposed getrs: F^{-1} is the SAME list walked
    # BACKWARDS. Walk it forwards.
    # Expected: GetrsSolvesAllThreeTransposeModes, Trans and ConjTrans only.
    "getrs_forward": ("src/extensions/getrs_native.cc",
        "        piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/false);",
        "        piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/true);  // BREAK B2"),

    # ---------------------------------------------------------------- B3
    # THE INFO OFFSET, made BLOCK-LOCAL instead of global.
    # Expected: SingularColumnGivesGlobalOneBasedInfoFirstFailureWins.
    "info_block_local": ("src/extensions/getrf_cta_device.hh",
        "                info_local = static_cast<int32_t>(piv_base + k + 1);",
        "                info_local = static_cast<int32_t>(k + 1);   // BREAK B3: block-local"),

    # ---------------------------------------------------------------- B4
    # THE SHORT FINAL PANEL, dropped: stop the panel loop at the last FULL panel.
    # Expected: every order that is not a multiple of nb.
    "short_final": ("src/extensions/getrf_blocked.cc",
        "    for (int j0 = 0; j0 < n; j0 += nb) {",
        "    for (int j0 = 0; j0 + nb <= n; j0 += nb) {   // BREAK B4: drop the short final panel"),

    # ---------------------------------------------------------------- B5
    # A SUB-VIEW BUILT WITH rows INSTEAD OF THE PARENT ld.
    # Expected: every blocked shape whose trailing update is non-empty.
    "subview_ld": ("src/extensions/getrf_blocked.cc",
        "            nr, nc, ld, stride, batch, ptrs.data());",
        "            nr, nc, nr, stride, batch, ptrs.data());   // BREAK B5: rows, not parent ld"),

    # ---------------------------------------------------------------- B6
    # THE PERMUTATION ON THE WRONG SIDE FOR transA: apply it to the INPUT (as
    # NoTrans does) rather than to the OUTPUT.
    # Expected: GetrsSolvesAllThreeTransposeModes, Trans and ConjTrans only.
    "getrs_perm_first": ("src/extensions/getrs_native.cc",
        """    // Trans and ConjTrans. The transpose flag is passed THROUGH to both solves --
    // ConjTrans on a real type is Trans, which the trsm layer already handles.
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper, transA, Diag::NonUnit);""",
        """    // BREAK B6: the permutation moved to the INPUT, the NoTrans side.
    (void)lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
        ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
        piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/false);
    if (!ctx.in_order()) ctx.wait();
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper, transA, Diag::NonUnit);"""),

    # B6 needs its trailing output-side permutation removed too, or the two
    # cancel. Applied together as one break; this is its second hunk.
    "getrs_perm_first2": ("src/extensions/getrs_native.cc",
        """    return lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
        ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
        piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, /*forward=*/false);
}""",
        """    return ctx.get_event();   // BREAK B6b: output-side permutation removed
}"""),

    # ---------------------------------------------------------------- B7
    # THE 48 KB PAD, removed.
    # Expected: the pad-arithmetic half of ResidentLeafLaunchHoleAt48KiB, and --
    # on a device that actually refuses the band -- the launch half too.
    "hole_pad": ("src/extensions/getrf_cta.cc",
        "    return (bytes > kGetrfHoleLo && bytes <= kGetrfHoleHi) ? kGetrfHolePadTo : bytes;",
        "    return bytes;   // BREAK B7: the 48 KB pad removed"),

    # ---------------------------------------------------------------- B8
    # THE PIVOT METRIC: cabs1 -> the modulus, which is what cuBLAS does.
    # Expected: PivotSelectionUsesCabs1AndNotTheModulus, and -- because ORACLE 3
    # is metric-aware -- the ordinary complex sweeps as well.
    "pivot_metric": ("src/extensions/getrf_cta_device.hh",
        "        return sycl::fabs(a.re) + sycl::fabs(a.im);",
        "        return sycl::sqrt(a.re * a.re + a.im * a.im);   // BREAK B8: modulus, not cabs1"),

    # ---------------------------------------------------------------- B9
    # THE LEFT LASWP, dropped: the panel's interchanges are not applied to the
    # columns [0, j0) that already hold finished L.
    # Expected: every blocked order with more than one panel.
    "laswp_left": ("src/extensions/getrf_blocked.cc",
        "        if (j0 > 0) {\n            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>(",
        "        if (false) {   // BREAK B9: the left LASWP dropped\n            (void)lu_native::lu_laswp_launch<GetrfBlockedLaswpTag, T>("),

    # ---------------------------------------------------------------- B10
    # getri's BACKWARD permutation trace, run forwards.
    # Expected: GetriInvertsAndLeavesTheFactorUntouched.
    "getri_forward": ("src/extensions/getri_blocked.cc",
        "                    for (int k = n - 1; k >= 0; --k) {",
        "                    for (int k = 0; k < n; ++k) {   // BREAK B10: forward trace"),

    # ---------------------------------------------------------------- B11
    # THE LEAF'S ROW EXCHANGE, restricted to columns >= k, which is the classic
    # silently-wrong LU: L's finished columns do not travel with the exchange.
    # Expected: every shape whose leaf sees more than one column of finished L.
    "leaf_swap_right": ("src/extensions/getrf_cta_device.hh",
        "            for (int c = tid; c < n; c += wg) {",
        "            for (int c = k + tid; c < n; c += wg) {   // BREAK B11: columns >= k only"),

    # ---------------------------------------------------------------- B12
    # AN EPSILON FLOOR in the singularity predicate -- the thing the ground brief
    # suggested and WP6 declined, because `info` is a public contract shared with
    # LAPACK and cuBLAS and a floor diverges from both invisibly.
    # Expected: NearlySingularIsNotFlagged.
    "info_epsilon_floor": ("src/extensions/getrf_cta_device.hh",
        "        const D d = A.at(k, k);\n        if (batchlas::sycl_device::dev_is_zero(d)) {",
        "        const D d = A.at(k, k);\n        if (lu_cabs1<D>(d) < R(1e-20)) {   // BREAK B12: an epsilon floor"),

    # ---------------------------------------------------------------- B13
    # THE PIVOT STRIDE in the blocked driver: the vendor's layout is the ORDER of
    # the whole matrix per item; use the panel width instead.
    # Expected: every batch > 1 blocked shape with more than one panel.
    "piv_stride_nb": ("src/extensions/getrf_blocked.cc",
        "                                       ld, stride, mp, ib, batch,\n                                       piv_ptr, n, j0, info.data(), nullptr);",
        "                                       ld, stride, mp, ib, batch,\n                                       piv_ptr, ib, j0, info.data(), nullptr);   // BREAK B13"),

    # ---------------------------------------------------------------- B14
    # getri writes F TRANSPOSED into C -- the permutation on the wrong side.
    # Expected: GetriInvertsAndLeavesTheFactorUntouched.
    "getri_perm_t": ("src/extensions/getri_blocked.cc",
        "                    Cb[static_cast<std::ptrdiff_t>(r) * ldc + i] =",
        "                    Cb[static_cast<std::ptrdiff_t>(i) * ldc + r] =   // BREAK B14"),
}

# Breaks that must be applied together.
GROUPS = {"getrs_perm_first": ["getrs_perm_first", "getrs_perm_first2"]}


def patch(name, revert):
    names = GROUPS.get(name, [name])
    for nm in names:
        rel, old, new = BREAKS[nm]
        path = os.path.join(ROOT, rel)
        src = open(path).read()
        a, b = (new, old) if revert else (old, new)
        cnt = src.count(a)
        if cnt != 1:
            raise SystemExit(
                "ANCHOR MATCHED %d TIMES (must be exactly 1) for '%s' in %s\n---\n%s\n---"
                % (cnt, nm, rel, a))
        open(path, "w").write(src.replace(a, b, 1))
        print("%s %s in %s" % ("reverted" if revert else "applied", nm, rel))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in BREAKS and sys.argv[1] not in GROUPS:
        print("breaks: " + " ".join(sorted(set(list(BREAKS) + list(GROUPS)))))
        raise SystemExit(2)
    patch(sys.argv[1], "--revert" in sys.argv)
