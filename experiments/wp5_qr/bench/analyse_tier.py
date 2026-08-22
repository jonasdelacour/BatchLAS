#!/usr/bin/env python3
"""tier.csv: CTA against Blocked, vendor-free build, per (type, n).

THE ONE THING THIS SCRIPT EXISTS TO DO is refuse to tabulate a pin that did not
take. route_resolve.hh:101 falls through to automatic() when a forced route is
unsupported, so `BATCHLAS_GEQRF_ROUTE=cta` at an order whose m*n exceeds
cta_max_elems silently runs the BLOCKED driver -- and in a vendor-present build
would silently run cuSOLVER. Every row whose route column does not match its pin
is printed as PIN-DID-NOT-TAKE and excluded.

Ratio convention: blocked_ms / cta_ms, so >1 means CTA IS AHEAD.
"""
import sys, csv, collections

def main(path):
    d = collections.defaultdict(dict)
    for r in csv.reader(open(path)):
        if not r or r[0] == 'bin' or len(r) < 18:
            continue
        d[(r[2], int(r[4]), int(r[5]))][r[17].strip()] = r
    print('type,n,batch,cta_ms,blocked_ms,blocked/cta,winner,note')
    for k in sorted(d, key=lambda x: (x[0], x[1])):
        c, b = d[k].get('cta'), d[k].get('blocked')
        if not c or not b:
            print(f'  MISSING_ARM {k}')
            continue
        if c[14] != 'native:cta':
            print(f'  PIN-DID-NOT-TAKE {k[0]} n={k[1]}: cta pin resolved to '
                  f'{c[14]} (m*n={k[1]*k[1]} > cta_max_elems={c[15]}); excluded')
            continue
        if b[14] != 'native:blocked':
            print(f'  PIN-DID-NOT-TAKE {k[0]} n={k[1]}: blocked pin resolved to '
                  f'{b[14]}; excluded')
            continue
        cm, bm = float(c[6]), float(b[6])
        rel = bm / cm
        # At n <= nb the blocked driver is ONE panel with no trailing update and
        # its leaf is the resident one, i.e. literally the same code as the CTA
        # tier -- a NULL CELL, not a tie. The condition is n <= nb, taken from
        # geqrf_nb_for_type (geqrf_blocked.cc:165-179): 32 for float/cfloat/
        # cdouble, 16 for double. Deriving it from "the ratio is close to 1"
        # instead would mislabel double n=48, which really does run three panels
        # and lands at 0.98x by coincidence.
        nb = 16 if k[0] == 'double' else 32
        note = 'NULL CELL: n <= nb, blocked IS the resident leaf' if k[1] <= nb else ''
        win = 'cta' if rel > 1.0 else 'blocked'
        print(f'{k[0]},{k[1]},{k[2]},{cm:.4f},{bm:.4f},{rel:.3f},{win},{note}')

if __name__ == '__main__':
    main(sys.argv[1])
