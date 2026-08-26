#!/usr/bin/env python3
"""The SHIPPED clause's admitted set, sampled on every axis it names, plus the
rungs just outside every boundary -- for a confirmation pass on DEVICE 1.

Device 0 drives the display. WP8-I3 measured that as depressing the VENDOR arm
on L2-resident cells by up to 1.8x while leaving the native arm untouched. Every
cell here is at least 128 MB, and this pass's own cross-device control found the
two cards agreeing to 2.1% (vendor) and 0.1% (native) at that footprint -- but a
routing decision should not rest on that agreement alone, so the shipped set is
re-measured on the clean card.
"""
print("# admitted set")
for tr in ('T', 'C'):
    for out in (256, 512, 1024, 2048):
        for red in (64, 128, 256, 352):
            for b in (320, 512, 1024):
                if out * red * b * 16 / 2**30 <= 11.0:
                    print(f"cdouble:{out}:{red}:{b}:{tr}")
print("# the rungs just OUTSIDE every boundary, which must stay non-winners")
for tr in ('T', 'C'):
    for out, red, b in ((512, 128, 256), (256, 128, 192), (192, 128, 512),
                        (512, 384, 512), (512, 32, 512), (512, 40, 512)):
        print(f"cdouble:{out}:{red}:{b}:{tr}")
