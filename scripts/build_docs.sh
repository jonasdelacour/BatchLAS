#!/bin/sh
# Build the BatchLAS documentation site (Doxygen + doxygen-awesome-css).
#
#   sh scripts/build_docs.sh [out-dir]        default out-dir: build/docs
#
# Environment:
#   DOXYGEN             doxygen binary (default: doxygen on PATH); 1.18+ required
#   BATCHLAS_DOCS_STRICT=1   treat every Doxygen warning as an error
#
# This is the only build path: the CMake target batchlas_docs and the CI docs
# job both call it, so the site cannot depend on a configured SYCL tree.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
out=${1:-$root/build/docs}
case $out in /*) ;; *) out=$(pwd)/$out ;; esac
doxygen=${DOXYGEN:-doxygen}

if ! command -v "$doxygen" >/dev/null 2>&1; then
    echo "build_docs.sh: '$doxygen' not found. Install Doxygen 1.18+ or set DOXYGEN=/path/to/doxygen." >&2
    exit 2
fi
ver=$("$doxygen" --version | sed 's/[^0-9.].*//')
major=$(echo "$ver" | cut -d. -f1)
minor=$(echo "$ver" | cut -d. -f2)
if [ "$major" -lt 1 ] || { [ "$major" -eq 1 ] && [ "$minor" -lt 18 ]; }; then
    echo "build_docs.sh: Doxygen $ver is too old; 1.18+ is required (MARKDOWN_ID_STYLE=GITHUB and doxygen-awesome 2.5)." >&2
    exit 2
fi

mkdir -p "$out/generated"
cd "$root"
python3 docs/tools/gen_db_pages.py --out "$out/generated"

BATCHLAS_VERSION=$(sed -n 's/^project(BatchLAS[^)]*VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt | head -n 1)
BATCHLAS_DOCS_OUT=$out
BATCHLAS_DOCS_GENERATED=$out/generated
if [ "${BATCHLAS_DOCS_STRICT:-0}" = 1 ]; then
    BATCHLAS_DOCS_WARN_AS_ERROR=FAIL_ON_WARNINGS
else
    BATCHLAS_DOCS_WARN_AS_ERROR=NO
fi
if command -v dot >/dev/null 2>&1; then BATCHLAS_DOCS_HAVE_DOT=YES; else BATCHLAS_DOCS_HAVE_DOT=NO; fi
export BATCHLAS_VERSION BATCHLAS_DOCS_OUT BATCHLAS_DOCS_GENERATED BATCHLAS_DOCS_WARN_AS_ERROR BATCHLAS_DOCS_HAVE_DOT

"$doxygen" docs/Doxyfile
python3 docs/tools/check_doc_anchors.py --xml "$out/xml"

n=$(wc -l < "$out/doxygen-warnings.log" | tr -d ' ')
echo "build_docs.sh: site at $out/html/index.html ($n warning line(s) in $out/doxygen-warnings.log)"
