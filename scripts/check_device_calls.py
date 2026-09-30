#!/usr/bin/env python3
"""Fail if NVPTX device code calls SPIR-V builtins out of line.

    scripts/check_device_calls.py build/                  # scan build/src/libbatchlas_*.so
    scripts/check_device_calls.py --lib a.so --lib b.so   # scan named libraries
    scripts/check_device_calls.py --self-test

icpx defaults to -fp-model=fast. Device code is then tagged with a flush-to-zero
denormal mode that the linked libspirv bitcode does not share, LLVM refuses to
inline across the mismatch, and sycl::fma, barriers and work-item id queries all
become `call`s: 98% of kernels, 6-15x on cfloat gemm. The build now passes
-ffp-model=precise; this check keeps it that way, whichever compiler or flag
spelling a future configure ends up with.
evidence: docs/perf/blackwell.md#device-call-guard

It reads PTX, not SASS. SASS `CALL.REL` also covers the IEEE division and sqrt
slow paths that ptxas emits as local subroutines in a correct build, while a PTX
`call` names its callee.

Exit codes: 0 clean, 1 offending calls or a representative kernel missing,
77 nothing to check (no NVPTX image in any library: a CPU-only build).
"""
import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

FATBIN_MAGIC = b"\x50\xed\x55\xba"

# Callee names that must never survive as an out-of-line call.
BANNED_CALLEE = re.compile(r"__spirv_|__clc_|__mux_|_Z\d+__spirv")

# Mangled-name fragments of kernels that must be present AND call-free. A kernel
# that is renamed or moved to another library turns the check red instead of
# silently shrinking what it looks at.
REPRESENTATIVE = {
    "gemm wide cfloat": r"GemmRegister64x64K16WideKernelISt7complexIfELb1E",
    "potrf lpanel": r"PotrfLpanelKernel",
    "trsm cta": r"TrsmCtaKernel",
}

ENTRY_RE = re.compile(r"^\s*(?:\.visible\s+|\.weak\s+)?\.entry\s+([\w$.]+)", re.M)
FUNC_RE = re.compile(r"^\s*(?:\.visible\s+|\.weak\s+|\.extern\s+)?\.func\s+(?:\([^)]*\)\s*)?([\w$.]+)", re.M)
CALL_RE = re.compile(r"^\s*(?:@!?%\w+\s+)?call(?:\.uni)?\s+(?:\([^)]*\)\s*,\s*)?([\w$.]+)", re.M)


def find_cuobjdump(explicit):
    if explicit:
        return explicit
    for env in ("CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"):
        root = os.environ.get(env)
        if root and os.path.isfile(os.path.join(root, "bin", "cuobjdump")):
            return os.path.join(root, "bin", "cuobjdump")
    return shutil.which("cuobjdump")


def fatbins(path):
    """Yield every CUDA fatbin embedded in a shared library, by header walk."""
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while True:
        i = data.find(FATBIN_MAGIC, off)
        if i < 0:
            return
        version, header, size = struct.unpack_from("<HHQ", data, i + 4)
        if version != 1 or header != 16 or i + header + size > len(data):
            off = i + 4
            continue
        yield data[i:i + header + size]
        off = i + header + size


def split_functions(ptx):
    """Map each .entry/.func name to its body text."""
    marks = [(m.start(), m.group(1), "entry") for m in ENTRY_RE.finditer(ptx)]
    marks += [(m.start(), m.group(1), "func") for m in FUNC_RE.finditer(ptx)]
    marks.sort()
    out = []
    for n, (start, name, kind) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(ptx)
        out.append((kind, name, ptx[start:end]))
    return out


def scan_ptx(ptx):
    """[(entry name, [callee, ...])] for every .entry in one PTX module."""
    result = []
    for kind, name, body in split_functions(ptx):
        if kind == "entry":
            result.append((name, CALL_RE.findall(body)))
    return result


def ptx_of(blob, cuobjdump):
    with tempfile.NamedTemporaryFile(suffix=".fatbin", delete=False) as t:
        t.write(blob)
    try:
        run = subprocess.run([cuobjdump, "-ptx", t.name], capture_output=True, text=True)
    finally:
        os.unlink(t.name)
    if run.returncode != 0:
        raise RuntimeError(f"cuobjdump -ptx failed: {run.stderr.strip()}")
    return run.stdout


def libraries(build_dir):
    found = {}
    for p in sorted(glob.glob(os.path.join(build_dir, "src", "libbatchlas_*.so*"))):
        real = os.path.realpath(p)
        if os.path.isfile(real):
            found.setdefault(real, p)
    return sorted(found)


def check(libs, cuobjdump, verbose):
    entries = []
    images = 0
    for lib in libs:
        for blob in fatbins(lib):
            images += 1
            ptx = ptx_of(blob, cuobjdump)
            for name, callees in scan_ptx(ptx):
                entries.append((os.path.basename(lib), name, callees))
    if images == 0:
        print("check_device_calls: no NVPTX image in", ", ".join(libs) or "(no libraries)")
        return 77
    if not entries:
        print(f"check_device_calls: {images} images but no PTX .entry: cannot check (cubin-only build?)")
        return 1

    status = 0
    bad = [(lib, n, [c for c in cs if BANNED_CALLEE.search(c)]) for lib, n, cs in entries]
    bad = [b for b in bad if b[2]]
    other = sum(1 for _, _, cs in entries if cs and not any(BANNED_CALLEE.search(c) for c in cs))
    print(f"check_device_calls: {len(libs)} libraries, {images} images, {len(entries)} kernels, "
          f"{len(bad)} with out-of-line builtin calls, {other} with other calls")
    for label, pat in REPRESENTATIVE.items():
        hits = [e for e in entries if re.search(pat, e[1])]
        calls = sum(len([c for c in e[2] if BANNED_CALLEE.search(c)]) for e in hits)
        verdict = "MISSING" if not hits else ("FAIL" if calls else "ok")
        print(f"  {label:18s} kernels={len(hits):4d} builtin_calls={calls:6d} {verdict}")
        if verdict != "ok":
            status = 1
    if bad:
        status = 1
        bad.sort(key=lambda b: -len(b[2]))
        print("offending kernels (most calls first):")
        for lib, name, callees in bad[: None if verbose else 15]:
            top = sorted(set(callees), key=callees.count, reverse=True)[:3]
            print(f"  {len(callees):6d} {lib} {name[:110]} -> {', '.join(top)}")
        print("Likely cause: an icpx build without -ffp-model=precise "
              "(see AGENTS.md Known Pitfalls).")
    return status


SELF_TEST_PTX = """
.func (.param .b32 func_retval0) _Z15__spirv_ocl_fmafff(.param .b32 a, .param .b32 b, .param .b32 c)
{
	fma.rn.ftz.f32 	%f4, %f1, %f2, %f3;
	ret;
}
.visible .entry _ZTS3Bad(.param .u64 p)
{
	{ // callseq 0, 0
	call.uni (retval0), _Z15__spirv_ocl_fmafff, (param0, param1, param2);
	} // callseq 0
	@%p1 call.uni _Z22__spirv_ControlBarrierjjj, (param0, param1, param2);
	ret;
}
.visible .entry _ZTS4Good(.param .u64 p)
{
	fma.rn.f32 	%f4, %f1, %f2, %f3;
	// call.uni _Z15__spirv_ocl_fmafff in a comment is not a call
	ret;
}
"""


def self_test():
    got = dict(scan_ptx(SELF_TEST_PTX))
    ok = (set(got) == {"_ZTS3Bad", "_ZTS4Good"}
          and got["_ZTS3Bad"] == ["_Z15__spirv_ocl_fmafff", "_Z22__spirv_ControlBarrierjjj"]
          and got["_ZTS4Good"] == []
          and all(BANNED_CALLEE.search(c) for c in got["_ZTS3Bad"]))
    blob = FATBIN_MAGIC + struct.pack("<HHQ", 1, 16, 4) + b"abcd"
    ok = ok and [len(b) for b in fatbins_from_bytes(b"xx" + blob + b"yy" + blob)] == [20, 20]
    print("check_device_calls self-test:", "ok" if ok else f"FAILED {got}")
    return 0 if ok else 1


def fatbins_from_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(data)
    try:
        return list(fatbins(t.name))
    finally:
        os.unlink(t.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir", nargs="?")
    ap.add_argument("--lib", action="append", default=[])
    ap.add_argument("--cuobjdump")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    # Always self-test first: a parser that finds no calls must be told apart from
    # a build that makes none.
    st = self_test()
    if st != 0 or a.self_test:
        return st
    libs = list(a.lib)
    if a.build_dir:
        libs += libraries(a.build_dir)
    if not libs:
        print("check_device_calls: no libraries to scan (pass a build dir or --lib)")
        return 77
    tool = find_cuobjdump(a.cuobjdump)
    if not tool:
        print("check_device_calls: cuobjdump not found (set CUDA_PATH or pass --cuobjdump)")
        return 77
    return check(libs, tool, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
