#!/usr/bin/env python3
"""Check that the CMake package BatchLAS exports stays consumable off this box.

Two modes, because CI has no SYCL compiler and therefore cannot generate the
export at all:

  --source   (default) Lint the list files that produce the export. Flags a
             literal SYCL flag or a literal host path handed to an exported
             batchlas target's INTERFACE/PUBLIC usage requirements without a
             $<BUILD_INTERFACE:...> guard. Cheap, runs anywhere, no build.

  --package DIR
             The real gate. Point it at an install prefix, an installed
             lib/cmake/BatchLAS, or a build tree, and it reads the generated
             BatchLASConfig.cmake / BatchLASTargets*.cmake. Anything that
             survives into those files is what a consumer actually gets, so a
             hardcoded /usr/local/cuda-13.2, an absolute
             /usr/lib/x86_64-linux-gnu/liblapacke.so, or a -fsycl on an
             INTERFACE property is a failure. Run this after every install:

                 cmake --install build --prefix /tmp/inst
                 python3 .github/ci/check_exported_package.py --package /tmp/inst

Both modes exit 0 = clean, 1 = findings.
"""

import argparse
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmakeparse import iter_commands  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Toolchain flags that must never reach a consumer's compile line: they are
# specific to the SYCL compiler, the CUDA install and the GPU arch of whatever
# machine built the package.
BAD_FLAGS = (
    "-fsycl",
    "--cuda-path",
    "-Xclang",
    "-Xcuda-ptxas",
    "-Xsycl-target-backend",
    "-fno-sycl",
    "--offload-arch",
)

# Absolute host paths. An exported package addresses everything it installs
# through ${_IMPORT_PREFIX}; anything rooted in one of these is a snapshot of
# the build machine's filesystem.
BAD_PATHS = re.compile(
    r"(?<![A-Za-z0-9_./-])/(?:usr/local/cuda|usr/lib/x86_64-linux-gnu|usr/lib64|usr/lib/|usr/local/lib|opt/|home/)"
)

# Variables that hold absolute host paths (or imported targets that carry them)
# by construction: the LAPACKE/CBLAS .so files find_library() resolved, the
# CUDA::/roc:: imported targets whose INTERFACE_INCLUDE_DIRECTORIES point at this
# box's toolkit. They are legitimate PRIVATE link libraries and nothing else - a
# PUBLIC or INTERFACE one puts the build machine on every consumer's link line
# and drags a find_dependency() into the generated config. Named explicitly
# rather than pattern-matched because most ${...} on a usage requirement is fine
# (${BATCHLAS_COMPONENT_TARGETS}, for one) and a lint that cries wolf gets
# switched off.
MACHINE_PINNED_VARS = (
    "BATCHLAS_NETLIB_LINK_LIBRARIES",
    "BATCHLAS_CUDA_LINK_LIBRARIES",
    "BATCHLAS_ROCM_LINK_LIBRARIES",
    "BATCHLAS_MKL_LINK_LIBRARIES",
)

# Usage-requirement commands whose arguments end up in the export.
USAGE_COMMANDS = {
    "target_compile_options",
    "target_link_options",
    "target_compile_definitions",
    "target_include_directories",
    "target_link_libraries",
}

SOURCE_SKIP_DIRS = {".git", ".claude", "build", "_deps", "__pycache__"}
PACKAGE_FILES = re.compile(r"^BatchLAS(Config|Targets)[A-Za-z0-9_-]*\.cmake$")
MESSAGE_CALL = re.compile(r"(^|[^A-Za-z0-9_])message\s*\(", re.IGNORECASE)
INFORMATIONAL_SET = re.compile(
    r"set\s*\(\s*(BatchLAS_CXX_COMPILER|BatchLAS_SYCL_TARGETS)\b", re.IGNORECASE)


def source_files():
    files = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in SOURCE_SKIP_DIRS]
        for name in filenames:
            if name == "CMakeLists.txt" or name.endswith(".cmake") or name.endswith(".cmake.in"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def check_sources():
    findings = []
    for path in source_files():
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        rel = os.path.relpath(path, REPO)
        for name, args, line, _ in iter_commands(text):
            if name.lower() not in USAGE_COMMANDS:
                continue
            words = args.split()
            if not words:
                continue
            target = words[0]
            if not target.startswith("batchlas"):
                continue
            # Per argument, not per command. A single call can carry a PRIVATE
            # section and a PUBLIC one, and it can carry one guarded entry next
            # to an unguarded one; judging the whole command by "does the text
            # contain BUILD_INTERFACE / PUBLIC anywhere" lets both through.
            # Starts PUBLIC: target_link_libraries' legacy keyword-less form is
            # all-PUBLIC, and every other command here requires the keyword as
            # its first argument, so nothing is judged on this default.
            scope = "PUBLIC"
            reported_flag = False
            reported_path = False
            for word in words[1:]:
                if word in ("PRIVATE", "PUBLIC", "INTERFACE"):
                    scope = word
                    continue
                if scope not in ("PUBLIC", "INTERFACE"):
                    continue
                if "BUILD_INTERFACE" in word:
                    continue  # already guarded; the installed export drops it
                if not reported_flag:
                    for flag in BAD_FLAGS:
                        if flag in word:
                            findings.append((rel, line,
                                             "%s() puts %s on %s's exported interface without $<BUILD_INTERFACE:>"
                                             % (name, flag, target)))
                            reported_flag = True
                            break
                if not reported_path:
                    m = BAD_PATHS.search(word)
                    if m:
                        findings.append((rel, line,
                                         "%s() puts the absolute host path %s... on %s's exported interface"
                                         % (name, word[m.start():m.start() + 40], target)))
                        reported_path = True
                for var in MACHINE_PINNED_VARS:
                    if "${%s}" % var in word:
                        findings.append((rel, line,
                                         "%s() puts ${%s} on %s's exported interface; it expands to absolute "
                                         "host paths / imported toolkit targets and must be PRIVATE"
                                         % (name, var, target)))
                        break
    return findings


def package_files(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "_deps"}]
        for name in filenames:
            if PACKAGE_FILES.match(name):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def check_package(root):
    files = package_files(root)
    if not files:
        print("check_exported_package: no BatchLASConfig/BatchLASTargets found under %s" % root)
        print("  (install first: cmake --install <build> --prefix <root>)")
        return None
    findings = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            # Depth of an open message() call. Its body is diagnostic prose, not
            # an exported property: BatchLASConfig.cmake's compiler-mismatch
            # warning has to quote both the compiler it was built with and the
            # -fsycl the consumer may need, and flagging that text would make
            # the only honest way to warn about machine pinning look like
            # machine pinning.
            message_depth = 0
            for lineno, raw in enumerate(fh, 1):
                line = raw.rstrip("\n")
                if line.lstrip().startswith("#"):
                    continue
                in_message = message_depth > 0
                if MESSAGE_CALL.search(line):
                    in_message = True
                if in_message:
                    message_depth += line.count("(") - line.count(")")
                    if message_depth < 0:
                        message_depth = 0
                    continue
                # Informational variables, not usage requirements. Nothing
                # consumes them as a path or a flag; BatchLASConfig.cmake only
                # compares and prints them. Everything else is still checked.
                if INFORMATIONAL_SET.match(line.lstrip()):
                    continue
                # $<BUILD_INTERFACE:...> is dropped by CMake when it generates the
                # install export, so anything still carrying it is a build-tree
                # export and is fine.
                if "BUILD_INTERFACE" in line:
                    continue
                for flag in BAD_FLAGS:
                    if flag in line:
                        findings.append((path, lineno,
                                         "exported property carries the build machine's toolchain flag %s" % flag))
                        break
                m = BAD_PATHS.search(line)
                if m:
                    findings.append((path, lineno,
                                     "exported property carries the absolute host path %s"
                                     % line[m.start():m.start() + 60].split(";")[0].rstrip('"')))
    print("check_exported_package: %d generated package file(s) checked" % len(files))
    return findings


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", metavar="DIR",
                        help="install prefix / build tree holding a generated BatchLASTargets.cmake")
    parser.add_argument("--source", action="store_true",
                        help="lint the list files that produce the export (default)")
    args = parser.parse_args(argv[1:])

    findings = []
    ran = False
    if args.package:
        result = check_package(args.package)
        if result is None:
            return 1
        findings += result
        ran = True
    if args.source or not args.package:
        findings += check_sources()
        ran = True
    if not ran:
        return 1

    for path, line, message in findings:
        print("%s:%d: error: %s" % (path, line, message))
    if findings:
        print("check_exported_package: %d problem(s) - this package is pinned to the machine that built it"
              % len(findings))
        return 1
    print("check_exported_package: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
