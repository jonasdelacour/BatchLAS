#!/bin/bash
P=/home/jonaslacour/.claude/jobs/20812aa0/tmp/break.py
for b in piv_base unit_u last_row conj trans_perm_forward perm_wrong_side swap_solves reg_cap cap_inversion cap_band rhs_ld hole_pad; do
  if python3 "$P" "$b" >/dev/null 2>&1; then echo "OK   $b"; else echo "FAIL $b"; python3 "$P" "$b" 2>&1 | tail -3; fi
done
python3 "$P" restore
