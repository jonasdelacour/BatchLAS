# Normalize a `ptxas -v` device-link log into a stable, diffable per-kernel table.
#
# Usage: awk -f summarize_ptxas.awk <log> | sort
# Columns (TAB separated):
#   1 entry function (mangled)
#   2 regs            "Used N registers"
#   3 own_stack       stack frame of the entry's OWN properties block
#   4 own_spill_st    spill stores of the entry's OWN block
#   5 own_spill_ld    spill loads  of the entry's OWN block
#   6 callee_spill_st max spill stores over NON-INLINED callee blocks of this kernel
#   7 callee_spill_ld max spill loads  over the same
#   8 cum_stack       "N bytes cumulative stack size"
#   9 barriers
#
# TWO PARSING TRAPS THIS ENCODES, both found by cross-checking against a raw
# grep of the same log (experiments/wp4_potrf/regbaseline):
#
# (1) The bytes line is
#     "<S> bytes stack frame, <A> bytes spill stores, <B> bytes spill loads"
#     -> fields 1, 5, 9.  Fields 4 and 8 are the words "frame,"/"stores,", which
#     awk coerces to 0 -- i.e. reading the wrong index reports "never spills".
#
# (2) One entry function can be followed by SEVERAL "Function properties for"
#     blocks: its own, then one per non-inlined callee. In this tree
#     GesvdjCTAKernel<complex<double>,32,64,false> reports 0 spill in its own
#     block while its callee gesvdj_cta_impl<...> spills 4612/4596 bytes
#     (log lines 3949-3956). Keeping only the last block scores that kernel
#     spill-free. Columns 6-7 exist so the gate cannot be passed that way.
#
# "Compile time" is dropped (nondeterministic), so two runs over an unchanged
# tree diff byte-identically.
/Compiling entry function/ {
  name = $0; sub(/.*Compiling entry function '/, "", name); sub(/' for .*/, "", name);
  cur = name; seen[cur] = 1; prop = ""; next
}
/Function properties for/ {
  prop = $0; sub(/.*Function properties for /, "", prop); next
}
/bytes stack frame, .* bytes spill stores, .* bytes spill loads/ {
  if (cur == "") next;
  split($0, f, " ");
  s = f[1] + 0; a = f[5] + 0; b = f[9] + 0;
  if (prop == cur) { own_s[cur] = s; own_a[cur] = a; own_b[cur] = b }
  else { if (a > cal_a[cur]) cal_a[cur] = a; if (b > cal_b[cur]) cal_b[cur] = b }
  next
}
/Used [0-9]+ registers/ {
  if (cur == "") next;
  match($0, /Used [0-9]+ registers/); regs[cur] = substr($0, RSTART + 5, RLENGTH - 15) + 0;
  if (match($0, /used [0-9]+ barriers/)) bar[cur] = substr($0, RSTART + 5, RLENGTH - 14) + 0;
  if (match($0, /[0-9]+ bytes cumulative stack size/)) cum[cur] = substr($0, RSTART, RLENGTH - 28) + 0;
  next
}
END {
  for (k in seen)
    printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
           k, regs[k], own_s[k], own_a[k], own_b[k], cal_a[k], cal_b[k], cum[k], bar[k];
}
