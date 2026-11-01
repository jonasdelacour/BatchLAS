#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SASS-parity gate for in-kernel linear-algebra migrations; all logic is in sass_parity.py.
#   scripts/sass_parity.sh --base <build> --cand <build> --lib <target|.so> [--pair 'B=C'] [--mode strict|algo] [--build]
# Exit 0 pass, 1 fail, 77 no NVPTX image or no cuobjdump.
exec python3 "$(dirname "$0")/sass_parity.py" "$@"
