#!/usr/bin/env python3
"""GATE-D breaks for the four preferred() clauses this pass ships.

RULE, learned the hard way in this campaign and restated in D4 part A2: every
anchor must match EXACTLY ONCE in both directions. An anchor that matches twice
edits the wrong copy; an anchor that matches zero times reverts nothing and
leaves the tree broken. Both are checked here before anything is written.

A BREAK AGAINST preferred() IS NOT VACUOUS IN EITHER BUILD, and that is worth
saying because the campaign's usual GATE-D worry is the opposite. With a window
LANDED the clause is what a vendor-PRESENT build consults on every Origin::Auto
call, so route_vocabulary_tests (pure layer, both builds) and getrf_tests
(real device, vendor present) both reach it. The vendor-free build reaches it
too, through automatic()'s first walk.

usage: breaks.py list
       breaks.py apply <name>
       breaks.py revert <name>
"""
import sys, pathlib

W = pathlib.Path("/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan")

BREAKS = {
    # ---- getrs: the per-type nrhs boundary -----------------------------------
    "getrs_boundary": (
        "include/batchlas/blas/dispatch/route_getrs.hh",
        # THE ANCHOR NEEDS ITS INDENTATION. Without the leading newline and the
        # twelve spaces this matched TWICE -- the header comment quotes the
        # clause verbatim, and the first spelling of this break edited the
        # COMMENT. The exactly-once check is what caught it.
        "\n            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;",
        "\n            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 65;",
        "moves float's boundary off the measured rung by one"),
    # ---- getrs: the measured batch floor ------------------------------------
    "getrs_nofloor": (
        "include/batchlas/blas/dispatch/route_getrs.hh",
        "            if (s.batch < 128) return false;\n",
        "",
        "deletes the batch floor -- the clause then admits batch 1, where the "
        "composition measures 0.055x-0.33x"),
    # ---- getri: the per-type order boundary ---------------------------------
    "getri_boundary": (
        "include/batchlas/blas/dispatch/route_getri.hh",
        "if constexpr (std::is_same_v<T, float>)               return s.order() >= 128;",
        "if constexpr (std::is_same_v<T, float>)               return s.order() >= 129;",
        "moves float's order boundary off the measured rung by one"),
    # ---- getri: the per-TYPE split, which is the half most likely to be lost -
    "getri_typeleak": (
        "include/batchlas/blas/dispatch/route_getri.hh",
        "if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 256;",
        "if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 128;",
        "gives cfloat float's boundary -- the exact error the old header made, "
        "and cfloat n=128 is a 0.71x LOSS"),
    # ---- getrf: the per-type order boundary ---------------------------------
    "getrf_boundary": (
        "include/batchlas/blas/dispatch/route_getrf.hh",
        "if constexpr (std::is_same_v<T, float>)               return s.order() >= 256;",
        "if constexpr (std::is_same_v<T, float>)               return s.order() >= 255;",
        "moves float's order boundary off the measured rung by one"),
    # ---- getrf: double must earn nothing ------------------------------------
    "getrf_doubleleak": (
        "include/batchlas/blas/dispatch/route_getrf.hh",
        "        return false;   // double and cdouble earn nothing at any order",
        "        return s.order() >= 512;   // BREAK",
        "admits double, whose best cell anywhere is 1.067 and which loses at "
        "0.743 at n=512 batch 1024"),
    # ---- gemv: the AXIS. This is the defect WP7 caught twice. ---------------
    "gemv_axisswap": (
        "include/batchlas/blas/dispatch/route_gemv.hh",
        "            return red >= 64 && red <= 352 && out >= 256 && s.batch >= 320;",
        "            return out >= 64 && out <= 352 && red >= 256 && s.batch >= 320;",
        "spells the band on out_len instead of red_len -- the window inverts"),
    # ---- gemv: the batch term, which the WP7 clause family could not express -
    "gemv_nobatch": (
        "include/batchlas/blas/dispatch/route_gemv.hh",
        "            return red >= 64 && red <= 352 && out >= 256 && s.batch >= 320;",
        "            return red >= 64 && red <= 352 && out >= 256;",
        "drops the batch floor -- admits 0.9562 at out 512, red 128, batch 128"),
    # ---- gemv: the type gate ------------------------------------------------
    "gemv_alltypes": (
        "include/batchlas/blas/dispatch/route_gemv.hh",
        "        if constexpr (std::is_same_v<T, std::complex<double>>) {",
        "        if constexpr (true) {",
        "admits float, double and cfloat -- refuted at 0.9340, 0.9722, 0.6644"),
}

def check(path, needle, n=1):
    txt = (W / path).read_text()
    c = txt.count(needle)
    if c != n:
        raise SystemExit(f"ANCHOR MATCHED {c} TIMES (want {n}) in {path}:\n{needle!r}")
    return txt

def main():
    cmd = sys.argv[1]
    if cmd == "list":
        for k, v in BREAKS.items(): print(f"{k:18s} {v[0].split('/')[-1]:20s} {v[3]}")
        return
    name = sys.argv[2]
    path, old, new, why = BREAKS[name]
    if cmd == "apply":
        txt = check(path, old)
        if new: check(path, new, 0)
        (W / path).write_text(txt.replace(old, new, 1))
        print(f"APPLIED {name}: {why}")
    elif cmd == "revert":
        txt = (W / path).read_text()
        if new:
            if txt.count(new) != 1: raise SystemExit(f"revert anchor matched {txt.count(new)} times")
            (W / path).write_text(txt.replace(new, old, 1))
        else:
            # a deletion break: re-insert before the line that follows it
            after = "            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;"
            if txt.count(after) != 1: raise SystemExit("revert anchor not unique")
            (W / path).write_text(txt.replace(after, old + after, 1))
        print(f"REVERTED {name}")

main()
