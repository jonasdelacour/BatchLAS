#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Source scan for the DPC++ SemaSYCL kernel-body quirk in device LA users.

    scripts/check_kernel_entry_calls.py [--root DIR] [--self-test]

SemaSYCL (SemaSYCL.cpp, isSYCLKernelBodyFunction and its FIXME) takes the kernel functor's
operator() as the kernel body and force-inlines it into the entry, unless that operator()
itself calls an operator() directly: then the FIRST such callee becomes the "body" and is
force-inlined instead, and the real body stays a separate function until device link. A v2
port that removes or adds such a call changes the kernel's inlining, so SASS parity breaks
for a reason that is not the port (potri, trsm, l3tt, gemm; wf3_synthesis.txt 3.5).

Rule for device LA users: the kernel lambda (or a kernel functor's operator()(sycl::nd_item...))
calls named methods or free functions, never a local lambda, an immediately invoked lambda,
or a device view's operator() (`A(i, j)` on `auto A = device::mat(...)`) directly. Calls inside a
nested lambda body are not direct and pass. BATCHLAS_DEVICE_BODY (always_inline) is the sanctioned
invoker: a call of a local lambda declared with it, an immediately invoked lambda carrying it,
and every call in a kernel lambda / operator() that itself carries it pass (the attribute
inlines the callee whatever SemaSYCL picks as the body).

What is scanned: src/ and include/ sources that include a v2 header DIRECTLY: batchlas/blas/
device/v2/, the public tree batchlas/device/ or batchlas/device.hh, src/kernels/common/dla_v2_*,
or a family's private <src/...*_v2.hh> (batchlas/device/detail/ alone does not count: it holds the
toolchain shims every kernel includes, kernel_attrs.hh and sg_partition/); and every file under include/batchlas/device/ (v22
design A.0: the lapack/<routine>.hh bodies are public, so they are scanned like users). A file
that only reaches v2 through another header (math-helpers.hh -> v2/scalar.hh) is not a v2 user.
Tests are not scanned: their codegen is not gated.
What is not caught (needs semantics a scan does not have): a callable passed in as a
parameter, a captured functor member, views not declared from a device:: factory or type.
Exit codes: 0 clean, 1 violations (or a failed self-test).
"""
import argparse
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE_INCLUDE = re.compile(r"#\s*include\s*[<\"](?:batchlas/blas/device/v2/|batchlas/device/(?!detail/)|batchlas/device\.hh[>\"]|"
                        r"src/kernels/common/dla_v2_|src/[^>\"]*_v2\.hh[>\"])")
BODY_ATTR = "BATCHLAS_DEVICE_BODY"
PUBLIC_TREE = os.path.join("include", "batchlas", "device") + os.sep
STRIP = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)
LAUNCH = re.compile(r"\b(?:parallel_for|single_task)\b(?:\s*<)?")
FUNCTOR = re.compile(r"\boperator\(\)\s*\(\s*(?:const\s+)?sycl::(?:nd_item|item|id)\b")
LAMBDA_DECL = re.compile(r"\b(?:const\s+)?auto\s*&{0,2}\s*(\w+)\s*=\s*\[")
VIEW_DECL = re.compile(r"\b(?:const\s+)?(?:auto\s*&{0,2}\s*(\w+)\s*=\s*(?:(?:::)?batchlas::)?device::\w+"
                       r"|(?:(?:::)?batchlas::)?device::(?:Vec|Mat|Frag|Tile)\b[^;{=()]*?\s(\w+)\s*[{(=])")
OPEN = {"(": ")", "[": "]", "{": "}"}


def strip(text):
    return STRIP.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def match(text, i):
    """Index just past the bracket that closes text[i]."""
    want = [OPEN[text[i]]]
    j = i + 1
    while j < len(text) and want:
        ch = text[j]
        if ch in OPEN:
            want.append(OPEN[ch])
        elif want and ch == want[-1]:
            want.pop()
        j += 1
    return j


def is_lambda_intro(text, i):
    k = i - 1
    while k >= 0 and text[k].isspace():
        k -= 1
    return k < 0 or text[k] in "=(,{;:?&|!" or text[k - 5:k + 1] == "return"


def marked(text, i, body_start):
    """True when the lambda introducer at text[i] carries BATCHLAS_DEVICE_BODY before its body."""
    return BODY_ATTR in text[i:body_start]


def lambda_body(text, i):
    """(body start, body end) of the lambda whose capture list opens at text[i], or None."""
    j = match(text, i)
    k = j
    while k < len(text) and text[k] != "{":
        if text[k] == ";":
            return None
        k = match(text, k) if text[k] in "([" else k + 1
    return (k, match(text, k)) if k < len(text) else None


def kernel_bodies(text):
    """[(offset, body)] for every kernel lambda and kernel functor operator()."""
    out = []
    for m in LAUNCH.finditer(text):
        i = m.end()
        depth = 1 if text[i - 1] == "<" else 0
        while depth and i < len(text):
            depth += {"<": 1, ">": -1}.get(text[i], 0)
            i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "(":
            continue
        end = match(text, i)
        args = text[i:end]
        depth_ok, depth = [], 0
        for p, ch in enumerate(args):
            if ch == "[" and depth == 1 and is_lambda_intro(args, p):
                depth_ok.append(p)
            depth += (ch in "([{") - (ch in ")]}")
        if depth_ok:
            span = lambda_body(text, i + depth_ok[-1])
            if span and not marked(text, i + depth_ok[-1], span[0]):
                out.append((span[0], text[span[0]:span[1]]))
    for m in FUNCTOR.finditer(text):
        close = match(text, m.start() + m.group().index("(", 10))
        k = text.find("{", close)
        if k >= 0 and ";" not in text[close:k] and BODY_ATTR not in text[close:k]:
            out.append((k, text[k:match(text, k)]))
    return out


def direct_text(body):
    """The body with nested lambda bodies blanked; immediately invoked lambdas are reported."""
    out, iife, i = list(body), [], 1
    while i < len(body):
        if body[i] == "[" and is_lambda_intro(body, i):
            span = lambda_body(body, i)
            if span:
                s, e = span
                rest = body[e:].lstrip()
                if rest.startswith("(") and not marked(body, i, s):
                    iife.append(i)
                for p in range(s + 1, e - 1):
                    if out[p] != "\n":
                        out[p] = " "
                i = e
                continue
        i += 1
    return "".join(out), iife


def violations(text):
    """[(offset, message)] for one stripped source text."""
    marked_decl, plain_decl = set(), set()
    for m in LAMBDA_DECL.finditer(text):
        span = lambda_body(text, m.end() - 1)
        (marked_decl if span and marked(text, m.end() - 1, span[0]) else plain_decl).add(m.group(1))
    lambdas = marked_decl | plain_decl
    views = {a or b for a, b in VIEW_DECL.findall(text)} - lambdas
    found = []
    for off, body in kernel_bodies(text):
        flat, iife = direct_text(body)
        for p in iife:
            found.append((off + p, "immediately invoked lambda in the kernel body"))
        for m in re.finditer(r"(?<![\w.>:])(\w+)\s*\(", flat):
            name = m.group(1)
            if name in plain_decl:
                found.append((off + m.start(), f"direct call of local lambda '{name}'"))
            elif name in views:
                found.append((off + m.start(), f"direct call of device view operator() '{name}(...)'"))
    return found


def device_users(root):
    """Files that include a device LA header directly (no transitive closure)."""
    files = {}
    for top in ("src", "include"):
        for d, _, names in os.walk(os.path.join(root, top)):
            for n in names:
                if n.endswith((".cc", ".hh", ".h", ".hpp", ".cpp")):
                    p = os.path.join(d, n)
                    files[os.path.relpath(p, root)] = open(p, errors="replace").read()
    users = {f for f, t in files.items() if DEVICE_INCLUDE.search(t) or f.startswith(PUBLIC_TREE)}
    return {f: files[f] for f in sorted(users) if "/blas/device/v2/" not in f}


SELF_TEST = """
#include <batchlas/blas/device/v2/dla.hh>
void good(sycl::handler& h) {
    auto helper = [&](int x) { return x; };
    impl::parallel_for<K1>(h, r, [=](sycl::nd_item<1> it) {
        const auto A = device::mat(p, n, n, ld);
        device::scal(ex, A.col(0), helper);        // passed, not called
        for_each_owned(ex, n, [&](int i) { A(i, i) = 0; });   // nested: not direct
        body(it);                              // a free function
    });
}
void bad(sycl::handler& h) {
    auto step = [&](int i) { return i; };
    h.parallel_for<K2>(r, [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
        const auto W = device::vec(p, n);
        step(0);                               // BREAK 1
        W(3) = 1;                              // BREAK 2
        [&] { W(0) = 0; }();                   // BREAK 3
    });
}
struct Body {
    void operator()(sycl::nd_item<1> it) const {
        auto tmp = [&] { return 1; };
        tmp();                                 // BREAK 4
    }
};
void sanctioned(sycl::handler& h) {
    auto stage = [&](int t) BATCHLAS_DEVICE_BODY { return t; };
    h.parallel_for<K3>(r, [=](sycl::nd_item<1> it) {
        stage(0);                              // marked local lambda: the sanctioned invoker
        [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DEVICE_BODY { stage(J...); }(seq);
    });
    h.parallel_for<K4>(r, [=](sycl::nd_item<1> it) BATCHLAS_DEVICE_BODY BATCHLAS_REQD_SG_SIZE(32) {
        const auto V = device::vec(p, n);
        V(0) = 1;                              // marked kernel lambda: every direct call passes
        stage(1);
    });
}
struct MarkedBody {
    void operator()(sycl::nd_item<1> it) const BATCHLAS_DEVICE_BODY {
        const auto V = device::vec(p, n);
        V(1) = 0;
    }
};
"""

# Scope: only files that include a v2 header directly (or a private *_v2.hh) are users.
SCOPE_TEST = {
    "src/x/helpers.hh": "#include <batchlas/blas/device/v2/scalar.hh>\n",
    "src/x/via_helpers.cc": "#include <src/x/helpers.hh>\n",
    "src/x/fam_v2.hh": "#include <batchlas/blas/device/v2/dla.hh>\n",
    "src/x/fam.cc": "#include <src/x/fam_v2.hh>\n",
    "src/x/quoted.cc": "#include \"src/kernels/common/dla_v2_scalar.hh\"\n",
    "src/x/legacy.cc": "#include <src/x/legacy.hh>\n",
    "src/x/public_user.cc": "#include <batchlas/device/lapack/potrf.hh>\n",
    "src/x/umbrella_user.cc": "#include <batchlas/device.hh>\n",
    "src/x/not_device.cc": "#include <batchlas/device_info.hh>\n",
    "src/x/attrs_only.cc": "#include <batchlas/device/detail/kernel_attrs.hh>\n",
    "include/batchlas/device/lapack/potrf.hh": "#include <batchlas/device/core.hh>\n",
    "include/batchlas/device/detail/no_include.hh": "int x;\n",
    "include/batchlas/blas/device/v2/core.hh": "#include <batchlas/blas/device/v2/scalar.hh>\n",
}


def self_test():
    got = sorted(m for _, m in violations(strip(SELF_TEST)))
    want = sorted(["direct call of local lambda 'step'", "direct call of device view operator() 'W(...)'",
                   "immediately invoked lambda in the kernel body", "direct call of local lambda 'tmp'"])
    with tempfile.TemporaryDirectory() as root:
        for rel, text in SCOPE_TEST.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as fh:
                fh.write(text)
        scope = sorted(device_users(root))
    want_scope = ["include/batchlas/device/detail/no_include.hh", "include/batchlas/device/lapack/potrf.hh",
                  "src/x/fam.cc", "src/x/fam_v2.hh", "src/x/helpers.hh", "src/x/public_user.cc", "src/x/quoted.cc",
                  "src/x/umbrella_user.cc"]
    ok = got == want and scope == want_scope
    print(f"check_kernel_entry_calls: self-test {'passed' if ok else 'FAILED'}: {got}; scope {scope}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if self_test() != 0:
        return 1
    if a.self_test:
        return 0
    users = device_users(a.root)
    bad = 0
    for f, text in users.items():
        st = strip(text)
        for off, msg in violations(st):
            print(f"{f}:{st.count(chr(10), 0, off) + 1}: {msg}")
            bad += 1
    print(f"check_kernel_entry_calls: {len(users)} device LA user files, {bad} violations")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
