#!/usr/bin/env python3
"""Scan the PTX AdaptiveCpp JIT-compiled for out-of-line calls, missing kernels and stack arrays.

    scripts/check_acpp_jit.py --appdb DIR                  # scan an existing ACPP_APPDB_DIR
    scripts/check_acpp_jit.py --build-dir B --run 'VAR=v tests/getrf_tests --gtest_filter=...' ...
    scripts/check_acpp_jit.py --appdb DIR --list           # per-family table, no verdict
    scripts/check_acpp_jit.py --self-test

The acpp counterpart of check_device_calls.py (DPC++). An acpp build compiles kernels to
generic IR and the runtime JIT-compiles each kernel at its first launch, so there is no PTX in
the libraries to scan. The runtime keeps the PTX it hands the CUDA driver in
<appdb>/apps/<app>-<hash>/jit-cache/<kernel>.<config>.jit: plain PTX, one .entry per file, one
file per specialization (adaptivity specializes invariant kernel arguments, so one kernel can
have hundreds). Only kernels that ran are there; --run runs commands against a fresh appdb
first (<build>/acpp_jit_check/appdb unless --appdb is given; their output goes to run.log
beside it). A leading VAR=value in a --run command sets that variable for it.

Three checks, every one a FAIL:
  calls     SSCP inlines every callee: any PTX `call` or `.extern .func` is a finding. This
            is where an unresolved libcall (__mulsc3, design page R1) or a builtin that stayed
            out of line (__spirv_*, __acpp_sscp_*) shows up. The PTX is cached even when
            ptxas then rejects it, so a kernel that failed to load is still scanned.
  required  with --run (or --require), each REQUIRED family must have been JIT-compiled at
            least once, so a renamed kernel or a test that stopped reaching it turns it red.
  stack     a RESIDENT family (BATCHLAS_UNROLL_FULL register tiles, tiny and CTA kernels) must
            have no __local_depot: a stack array means LLVM did not unroll or promote it.
            Known cases are in STACK_KNOWN with a reason and a byte ceiling; a known family
            that grows past its ceiling fails too.
Design: docs/design/sycl-implementations.md#sycl-impl-jit-ptx-check.

Exit codes: 0 clean, 1 findings, 77 nothing to check (no PTX in the appdb: no CUDA device,
or not an acpp build).
"""
import argparse
import collections
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

# label -> regex over the demangled kernel name. acpp names a kernel by its lambda or functor
# type, so these name the launcher (or the functor), not DPC++'s kernel-name tag.
RESIDENT = {
    "gemm 128x128": r"sycl_gemm::Gemm128x128Body<",
    "gemm register tiled": r"sycl_gemm::launch_register_tiled<",
    "gemm 64x64 k16 wide": r"sycl_gemm::launch_register_64x64_k16_wide<",
    "gemm wide transposed": r"sycl_gemm::launch_wide_transposed<",
    "gemm small batched": r"sycl_gemm_small::launch_small_batched<",
    "getrf tiny": r"sycl_getrf::\(anonymous namespace\)::GetrfTinyBody<",
    "gesv tiny": r"sycl_gesv::\(anonymous namespace\)::GesvTinyBody<",
    "posv tiny": r"sycl_posv::\(anonymous namespace\)::PosvTinyBody<",
    "potrf tiny": r"sycl_potrf::\(anonymous namespace\)::potrf_tiny_launch<",
    "geqrf tiny": r"sycl_geqrf::\(anonymous namespace\)::geqrf_tiny_launch<",
    "getrf panel reg": r"sycl_getrf::\(anonymous namespace\)::getrf_panel_reg_launch<",
    "getrs fused": r"sycl_getrs::\(anonymous namespace\)::(fused_launch_(no)?trans|potrs_fused_launch)<",
    "potrf cta": r"sycl_potrf::\(anonymous namespace\)::potrf_cta_launch<",
    "potrf lpanel": r"sycl_potrf::\(anonymous namespace\)::potrf_lpanel_launch<",
    "steqr cta": r"batchlas::steqr_cta_impl<",
    "syev cta fused": r"batchlas::syev_cta_fused_impl<",
    "gesvdj cta": r"batchlas::\(anonymous namespace\)::gesvdj_cta_impl<",
    "ormqr cta": r"batchlas::ormqx_cta_impl<",
    "trsm sg_left": r"sycl_trsm::trsm_native_sg_left<",
    "trsm native": r"sycl_trsm::trsm_native_v1<",
    "level-3 tiles": r"backend::detail::launch_(syrk_gram|syrk_triangular|syr2k_triangular|trmm_triangular)_tiles<",
}

# Families the --run set must reach (a subset of RESIDENT plus the CTA eigensolver path).
REQUIRED = ("gemm 128x128", "gemm register tiled", "getrf tiny", "gesv tiny", "potrf lpanel",
            "potrf cta", "trsm sg_left", "steqr cta", "syev cta fused")

# (regex over the demangled name, largest __local_depot in bytes, reason). Measured on
# threadripper02, acpp 25.10 generic, sm_120, ACPP_ADAPTIVITY_LEVEL=1. The DPC++ column is the
# same kernel's depot in the DPC++ build (cuobjdump -ptx of build-dpcpp).
STACK_KNOWN = (
    (r"getrf_panel_reg_launch<std::complex<double>, 32>", 512,
     "acpp only (DPC++ GetrfPanelRegKernel 0 B): rA[32] of std::complex<double> stays a stack "
     "array although its loads are unrolled (constant offsets); about 600 ld + 650 st.local "
     "per kernel. Cause not investigated"),
    (r"launch_register_tiled<float, 128, 64, 32, 8, 4, 4, 4, [24], 2,", 128,
     "both builds, same two tiles (reg:m=128:n=64:k=32 u=4 and u=2; DPC++ "
     "GemmRegisterTiledKernel 128 B)"),
    (r"trsm_native_sg_left<", 512,
     "both builds, same kernels and the same bytes (DPC++ TrsmSgLeftKernel 128-512 B)"),
    (r"syev_cta_fused_impl<std::complex<", 512,
     "both builds, sizes differ by order (cdouble n=32: 512 B vs DPC++ SyevCtaFusedKernel "
     "272 B; n=4: 64 vs 136 B); DPC++ real 16-72 B where acpp has none"),
    (r"gesvdj_cta_impl<", 256,
     "both builds; acpp smaller (DPC++ GesvdjCTAKernel up to 736 B)"),
    (r"(fused_launch_(no)?trans|potrs_fused_launch)<std::complex<double>, ", 16,
     "acpp only (DPC++ 0 B): the 16 B temporary of a std::complex<double> row swap in local "
     "memory (pivot interchange) stays on the stack, generic-addressed, in some specializations"),
    (r"launch_wide_transposed<std::complex<double>, ", 16,
     "acpp only (DPC++ GemmWideTransposedKernel 0 B): the 16 B std::complex<double> staged from "
     "global to local memory in the tile load (zeroed, conditionally loaded) stays on the stack"),
)

BANNED_CALLEE = re.compile(r"__(mul|div)[sdx]c3|__spirv_|__clc_|__acpp_sscp_|__nv_")

ENTRY_RE = re.compile(r"^\s*(?:\.visible\s+|\.weak\s+)?\.entry\s+([\w$.]+)", re.M)
EXTERN_FUNC_RE = re.compile(r"^\s*\.extern\s+\.func\s+(?:\([^)]*\)\s*)?([\w$.]+)", re.M)
CALL_RE = re.compile(r"^\s*(?:\{\s*)?(?:@!?%\w+\s+)?call(?:\.uni)?\s+(?:\([^)]*\)\s*,\s*)?([\w$.]+)", re.M)
DEPOT_RE = re.compile(r"\.local\s+\.align\s+\d+\s+\.b8\s+__local_depot\d+\[(\d+)\]")
LOCAL_ACCESS_RE = re.compile(r"\b(?:ld|st)(?:\.\w+)*\.local\b|\b(?:ld|st)(?:\.\w+)*\s+[^;]*\[%SPL?\b")
COMMENT_RE = re.compile(r"//[^\n]*")


class Kernel:
    __slots__ = ("mangled", "name", "files", "calls", "externs", "depot", "local_ops")

    def __init__(self, mangled):
        self.mangled = mangled
        self.name = mangled
        self.files = 0
        self.calls = collections.Counter()
        self.externs = set()
        self.depot = 0
        self.local_ops = 0


def scan_ptx(text, kernels):
    """Fold one PTX module into kernels{mangled: Kernel}. Returns the entry names it had."""
    text = COMMENT_RE.sub("", text)
    names = ENTRY_RE.findall(text)
    if not names:
        return []
    externs = set(EXTERN_FUNC_RE.findall(text))
    calls = CALL_RE.findall(text)
    depot = sum(int(d) for d in DEPOT_RE.findall(text))
    local_ops = len(LOCAL_ACCESS_RE.findall(text))
    for n in names:
        k = kernels.setdefault(n, Kernel(n))
        k.files += 1
        k.calls.update(calls)
        k.externs |= externs
        k.depot = max(k.depot, depot)
        k.local_ops = max(k.local_ops, local_ops)
    return names


def scan_appdb(appdb):
    kernels = {}
    ptx_files = 0
    for path in glob.glob(os.path.join(appdb, "apps", "*", "jit-cache", "*.jit")):
        with open(path, "rb") as f:
            head = f.read(256)
            if b".target sm_" not in head and b"NVPTX" not in head:
                continue
            body = head + f.read()
        if scan_ptx(body.decode("utf-8", "replace"), kernels):
            ptx_files += 1
    demangle(kernels)
    return kernels, ptx_files


def demangle(kernels):
    names = sorted(kernels)
    tool = shutil.which("c++filt")
    if not tool or not names:
        return
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout
    for mangled, dem in zip(names, out.split("\n")):
        kernels[mangled].name = dem


def families(kernels):
    fam = {label: [] for label in RESIDENT}
    for k in kernels.values():
        for label, pat in RESIDENT.items():
            if re.search(pat, k.name):
                fam[label].append(k)
    return fam


def short(name, width=150):
    s = re.sub(r"^void __acpp_sscp_kernel<hipsycl::glue::__sscp_dispatch::\w+<", "", name)
    return s if len(s) <= width else s[:width] + "..."


def check(kernels, ptx_files, require, verbose):
    status = 0
    fam = families(kernels)
    called = [k for k in kernels.values() if k.calls or k.externs]
    print(f"check_acpp_jit: {ptx_files} PTX files, {len(kernels)} kernels, "
          f"{len(called)} with calls, {sum(1 for k in kernels.values() if k.depot)} with a stack depot")

    if called:
        status = 1
        print("calls (SSCP inlines everything; each of these is out of line):")
        for k in sorted(called, key=lambda k: -sum(k.calls.values()))[: None if verbose else 15]:
            callees = sorted(set(k.calls) | k.externs)
            banned = [c for c in callees if BANNED_CALLEE.search(c)]
            print(f"  {sum(k.calls.values()):5d} {short(k.name)} -> {', '.join((banned or callees)[:4])}")

    print("families (kernels, specializations, largest stack depot, local ld/st):")
    for label, ks in fam.items():
        depot = max((k.depot for k in ks), default=0)
        ops = max((k.local_ops for k in ks), default=0)
        need = require and label in REQUIRED
        verdict = "MISSING" if need and not ks else ("absent" if not ks else "ok")
        bad = []
        for k in ks:
            if not k.depot:
                continue
            known = next((kn for kn in STACK_KNOWN if re.search(kn[0], k.name)), None)
            if known is None or k.depot > known[1]:
                bad.append((k, known))
        if bad:
            verdict = "STACK"
        elif ks and depot:
            verdict = "known"
        print(f"  {label:22s} kernels={len(ks):4d} files={sum(k.files for k in ks):5d} "
              f"depot={depot:5d}B local_ops={ops:5d} {verdict}")
        if verdict in ("MISSING", "STACK"):
            status = 1
        for k, known in bad[: None if verbose else 5]:
            why = "not in STACK_KNOWN" if known is None else f"over the known {known[1]} B"
            print(f"      {k.depot:5d}B {k.local_ops:5d} ops  {short(k.name)}  ({why})")
    if verbose:
        print("known stack cases:")
        for pat, ceiling, reason in STACK_KNOWN:
            print(f"  <= {ceiling:4d}B {pat}\n      {reason}")
    return status


def run_commands(build_dir, appdb, commands):
    if os.path.isdir(appdb):
        shutil.rmtree(appdb)
    os.makedirs(appdb)
    env = dict(os.environ, ACPP_APPDB_DIR=appdb)
    env.setdefault("ACPP_ADAPTIVITY_LEVEL", "1")
    status = 0
    log_path = os.path.join(os.path.dirname(appdb), "run.log")
    with open(log_path, "w") as log:
        for cmd in commands:
            argv = shlex.split(cmd)
            run_env = dict(env)
            while argv and re.match(r"^[A-Z_][A-Z0-9_]*=", argv[0]):
                key, _, value = argv.pop(0).partition("=")
                run_env[key] = value
            if not os.path.isabs(argv[0]):
                argv[0] = os.path.join(build_dir, argv[0])
            print("check_acpp_jit: run", cmd, flush=True)
            log.write(f"==== {cmd}\n")
            log.flush()
            rc = subprocess.run(argv, env=run_env, cwd=build_dir, stdout=log,
                                stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"check_acpp_jit: exit {rc}: {cmd} (output in {log_path})")
                status = 1
    return status


SELF_TEST_PTX = """
//
// Generated by LLVM NVPTX Back-End
//
.version 8.7
.target sm_120
.address_size 64
.extern .func (.param .align 8 .b8 func_retval0[16]) __muldc3(.param .b64 a);
.visible .entry _Z3Badv()
{
	.local .align 8 .b8 	__local_depot0[16];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	mov.u64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	st.u64 	[%SP+8], %rd1;
	ld.local.u64 	%rd2, [%SPL];
	{ // callseq 0, 0
	call.uni (retval0), __muldc3, (param0);
	} // callseq 0
	// call.uni __not_a_call in a comment
	ret;
}
"""


def self_test():
    ks = {}
    scan_ptx(SELF_TEST_PTX, ks)
    scan_ptx(SELF_TEST_PTX.replace("_Z3Badv", "_Z4Goodv").replace(
        "\t.local .align 8 .b8 \t__local_depot0[16];\n", "").replace(
        "__muldc3", "x").replace(".extern .func (.param .align 8 .b8 func_retval0[16]) x(.param .b64 a);\n", ""), ks)
    bad, good = ks.get("_Z3Badv"), ks.get("_Z4Goodv")
    ok = (bad is not None and good is not None
          and dict(bad.calls) == {"__muldc3": 1} and bad.externs == {"__muldc3"}
          and bad.depot == 16 and bad.local_ops == 2
          and good.depot == 0 and good.externs == set() and dict(good.calls) == {"x": 1}
          and BANNED_CALLEE.search("__muldc3") and not BANNED_CALLEE.search("vprintf"))
    known_ok = all(re.compile(p) for p, _, _ in STACK_KNOWN) and set(REQUIRED) <= set(RESIDENT)
    ok = ok and known_ok
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "apps", "t-1", "jit-cache"))
        with open(os.path.join(d, "apps", "t-1", "jit-cache", "1.2.jit"), "w") as f:
            f.write(SELF_TEST_PTX)
        with open(os.path.join(d, "apps", "t-1", "jit-cache", "3.4.jit"), "wb") as f:
            f.write(b"\x7fELF host object")
        found, files = scan_appdb(d)
        ok = ok and files == 1 and list(found) == ["_Z3Badv"]
    print("check_acpp_jit self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--appdb", help="ACPP_APPDB_DIR to scan (with --run: wiped and refilled)")
    ap.add_argument("--build-dir", help="where --run commands resolve and run")
    ap.add_argument("--run", action="append", default=[], help="command to run with the appdb set")
    ap.add_argument("--list", action="store_true", help="print the family table, never fail")
    ap.add_argument("--require", action="store_true", help="enforce REQUIRED without --run")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    # Self-test first, always: a scan that finds nothing must be told apart from one that cannot parse.
    st = self_test()
    if st != 0 or a.self_test:
        return st
    status = 0
    appdb = a.appdb
    if a.run:
        if not a.build_dir:
            ap.error("--run needs --build-dir")
        appdb = appdb or os.path.join(a.build_dir, "acpp_jit_check", "appdb")
        status = run_commands(os.path.abspath(a.build_dir), os.path.abspath(appdb), a.run)
    if not appdb:
        ap.error("pass --appdb or --run")
    kernels, files = scan_appdb(appdb)
    if not files:
        print(f"check_acpp_jit: no PTX under {appdb} (no CUDA device, or not an acpp build)")
        return 77 if status == 0 else status
    st = check(kernels, files, require=bool(a.run) or a.require, verbose=a.verbose)
    return 0 if a.list else (status or st)


if __name__ == "__main__":
    sys.exit(main())
