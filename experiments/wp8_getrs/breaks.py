#!/usr/bin/env python3
# GATE-D. Deliberate breaks against the SHIPPED getrs sources, applied to the
# tree, rebuilt, run, shown RED, and reverted.
#
# THE ANCHOR RULE, learned the hard way in this campaign: every anchor must match
# EXACTLY ONCE in both directions. An 8-space anchor is a substring of the
# 12-space line, and once left both permutation walks inverted in the tree. This
# script asserts the count in BOTH directions and refuses to write otherwise.
#
# usage: breaks.py list
#        breaks.py apply <name>
#        breaks.py revert <name>
import pathlib
import sys

W = pathlib.Path("/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan")
SRC = W / "src/extensions/getrs_native.cc"
HDR = W / "src/extensions/getrs_native.hh"

BREAKS = {
    # B1. The reversed index walk is dropped: the transposed getrs applies P
    # instead of P^{-1}. This is the half of the transposed case that NO NoTrans
    # test can see, and before WP6 no test in the suite issued a Trans getrs at
    # all. Must go RED on the Trans/ConjTrans rows of both getrs tests.
    "gather_forward_only": (SRC, """                    if (forward) {
                        for (int k = 0; k < n; ++k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    } else {""", """                    if (true) {
                        for (int k = 0; k < n; ++k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    } else {"""),

    # B2. The gather writes the tile back UNPERMUTED. The permutation is silently
    # dropped for the gather arm only, so the walk arm still passes -- which is
    # exactly what makes the bit-for-bit assertion the discriminating one, and
    # what a residual-only test on a diagonally dominant matrix would miss (see
    # getrf_native.hh note 5: on the dominant matrix alone, dropping the
    # interchange leaves the residual BIT-IDENTICAL -- this suite's matrix is
    # dominant AND ROW-PERMUTED for that reason).
    "gather_identity_map": (SRC,
        "                            tile[static_cast<std::size_t>(col) * ldt + idxs[row]];",
        "                            tile[static_cast<std::size_t>(col) * ldt + row];"),

    # B3. The default boundary is inverted: the gather runs BELOW
    # kGetrsPermGatherMinNrhs and the walk above it. The WP7 error -- a predicate
    # written on the wrong side of its own axis -- transcribed into a test. Must
    # go RED on the decision-surface test, and ONLY there: nothing else in the
    # suite can see which spelling ran.
    "boundary_inverted": (SRC,
        "    return nrhs >= kGetrsPermGatherMinNrhs;",
        "    return nrhs <= kGetrsPermGatherMinNrhs;"),

    # B4. The capacity refusal is disabled, so the gather claims to fit at any
    # order and would launch a kernel whose local memory the device cannot
    # allocate. The refusal is SILENT by design (RouteTable<Op::getrs,T> has no
    # field for a laswp capacity), so no other test in the suite can see it.
    # (Its FIRST spelling replaced only the final `return`, and the suite stayed
    # green: the early `if (slm_budget <= int_bytes) return false` still fired at
    # the order the test probes, so the refusal was never actually disabled. A
    # break that does not reach the code it names is as vacuous as a test that
    # does not. Both guards are now removed together.)
    "capacity_always_fits": (SRC,
        """    if (slm_budget <= int_bytes) return false;
    const std::size_t col_bytes =
        static_cast<std::size_t>(n | 1) * sizeof(typename sycl_device::DevMap<T>::type);
    return (slm_budget - int_bytes) >= col_bytes;""",
        """    (void)slm_budget; (void)int_bytes;   /* BREAK: the refusal is disabled */
    return true;"""),

    # B5. The knob stops being read, so both A/B arms run the default. This is
    # the ELEVENTH blind guard in its own right: without the spelling read-back
    # the bit-identity test would compare one arm with itself and pass green, and
    # the A/B harness would report a flat 1.00x.
    "knob_ignored": (SRC,
        '    const char* const s = std::getenv("BATCHLAS_GETRS_LASWP");',
        '    const char* const s = nullptr; (void)std::getenv("BATCHLAS_GETRS_LASWP");'),
}


def apply(name, forward=True):
    path, a, b = BREAKS[name]
    src, dst = (a, b) if forward else (b, a)
    text = path.read_text()
    n_src, n_dst = text.count(src), text.count(dst)
    assert n_src == 1, f"{name}: source anchor matched {n_src} times, want exactly 1"
    assert n_dst == 0, f"{name}: target text already present {n_dst} times"
    path.write_text(text.replace(src, dst))
    print(f"{'APPLIED' if forward else 'REVERTED'} {name} in {path.name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for k in BREAKS:
            print(k)
    elif cmd == "apply":
        apply(sys.argv[2], True)
    elif cmd == "revert":
        apply(sys.argv[2], False)
    else:
        sys.exit(f"unknown command {cmd}")
