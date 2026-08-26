#!/usr/bin/env bash
# WP8/I3 -- THE DEVICE-LINK DELTA.
#
# This campaign is device-link-bound: libbatchlas_sycl.so is the SYCL link unit
# and adding entry functions is the way a kernel change costs build time.
# Body 4's twenty instantiations were measured FREE (58.03 s absent vs 57.59 s
# present), which is the protocol this copies: same TU, same target, alternating
# arms, idle box.
#
# ARM A: body 5 present -- twelve kernels, 4 types x W in {2,4,8}.
# ARM B: kGemvSegTransEmit forced false, so the launcher's `if constexpr` drops
#        the whole switch and NO GemvSegTKernel is emitted at all.
#
# Alternated A,B,A,B,... so a thermal or page-cache drift has to hit both.
set -uo pipefail
W=/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan
SRC="$W/src/sycl/gemv_native.cc"
SAVE=/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/gemv_native.cc.linksave
REPS="${REPS:-3}"
TARGET="${TARGET:-batchlas_sycl}"
BUILD="${BUILD:-$W/build}"

cp "$SRC" "$SAVE"
stub () {
  python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = """inline constexpr bool kGemvSegTransEmit =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;"""
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, "inline constexpr bool kGemvSegTransEmit = false;"))
PY
}

one () {   # $1 = label
  touch "$SRC"
  local t0 t1
  t0=$(date +%s.%N)
  cmake --build "$BUILD" --target "$TARGET" -j > /dev/null 2>&1
  t1=$(date +%s.%N)
  echo "$1 $(echo "$t1 - $t0" | bc)"
}

echo "# device-link time, target $TARGET in $BUILD, $REPS reps per arm, alternated"
for r in $(seq 1 "$REPS"); do
  cp "$SAVE" "$SRC"; one "present"
  stub;               one "stubbed"
done
cp "$SAVE" "$SRC"
cmake --build "$BUILD" --target "$TARGET" -j > /dev/null 2>&1
echo "# restored and rebuilt"
