#!/usr/bin/env python3
"""The getrs batch floor, walked down to 1.

The first clean pass shows nrhs=128 winning by 5.6x-6.1x at batch 32 AND 64, so
a floor at 128 -- the campaign's saturation policy -- would give away measured
wins. A floor has to be bracketed from BELOW by a measured non-winner, and the
only evidence there is from a contaminated sweep (0.055x-0.33x at batch 1-2).
So: measure batch 1..64 cleanly at two orders and put the floor where the ladder
actually turns.
"""
for t in ('float', 'double'):
    for n in (64, 512):
        for b in (1, 2, 4, 8, 16, 32, 64):
            print(f"getrs:{t}:{n}:128:{b}")
