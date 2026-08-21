#!/usr/bin/env python3
"""Attribute a blocked-potrf run's GPU time to the driver's stages.

`nsys stats --report cuda_gpu_kern_sum` totals per KERNEL NAME over the whole
process, which here contains BOTH arms -- cuSOLVER and the blocked driver -- plus
the restore memcpys.  The two arms are separable by name and the split is spelled
out below rather than inferred, so a kernel nobody classified lands in `other`
and is visible instead of being silently attributed to a stage.

cuSOLVER's kernels are the C-style ones: potrf_syrk_nc_kernel,
potrf_syrk_T16_nc_kernel, potrf_reset_info, potrfBatch_trsm_lower,
potrf_cta_lower_batch (note that last one is cuSOLVER's, NOT BatchLAS's
PotrfCtaKernel -- the names are one underscore apart and confusing them puts
120 ms of vendor time inside the native leaf).

Calls-per-arm is recovered from a kernel that runs exactly once per call:
potrf_reset_info for the vendor arm, PotrfBlockedFixupKernel for the blocked arm
(the fixup runs once per PANEL, so it is divided by n/nb).  That is what turns
process totals into per-call milliseconds comparable with main.csv.
"""
import csv, glob, os, re, collections

D = os.path.dirname(os.path.abspath(__file__))

VENDOR_KERNELS = re.compile(
    r'potrf_syrk|potrf_reset_info|potrfBatch_trsm_lower|potrf_cta_lower_batch', re.I)

BLOCKED = [
    ('leaf',    re.compile(r'PotrfCtaKernel')),
    ('fixup',   re.compile(r'PotrfBlockedFixupKernel')),
    ('fold',    re.compile(r'fold_symmetric_product_into_triangle')),
    ('trsm',    re.compile(r'TrsmCtaKernel|trsm_vendor|batch_trsm|sycl_trsm')),
    ('gemm',    re.compile(r'Gemm\w*Kernel|sgemm|dgemm|z\d*gemm|c\d*gemm|cutlass|ampere_|turing_|sm\d+_xmma')),
    ('ptrarray',re.compile(r'init_data_ptr_array|offsetPointerArray')),
]

# CAVEAT ON THE `gemm` BUCKET, and it matters at the 20% level: the native
# panel trsm (trsm_native_blocked) has its OWN injected trailing gemm (WP3), so
# in the `nn` configuration part of `gemm` belongs to the panel solve and not to
# potrf's trailing update.  The two are separable by LAUNCH COUNT, not by name.
# float n=1024 nb=128 W=32 issues 217 trailing gemms per potrf call
# (112 W-wide diagonal blocks + 105 below-diagonal rectangles) and the profile
# shows 672/8 = 84 GemmRegisterTiled + 1064/8 = 133 GemmTiled16 = 217 exactly,
# leaving the third entry (168/8 = 21) as the trsm's.  Same arithmetic for
# cdouble: 945 of 960.  The README quotes the SEPARATED figures; this script
# prints the raw bucket.

# n/nb, i.e. fixup launches per blocked call.
PANELS = {('float', 1024): 1024 // 128, ('cdouble', 1024): 1024 // 64}

for f in sorted(glob.glob(os.path.join(D, 'nsys', '*_cuda_gpu_kern_sum.csv'))):
    tag = os.path.basename(f).replace('_cuda_gpu_kern_sum.csv', '')
    t, n, b, cfg = tag.split('_')
    n, b = int(n), int(b)
    vend = collections.Counter(); blk = collections.Counter()
    vcalls = bcalls = 0
    other = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            tt = float(r.get('Total Time (ns)') or 0) / 1e6
            cnt = int(float(r.get('Instances') or 0))
            nm = r.get('Name') or ''
            if VENDOR_KERNELS.search(nm):
                vend[nm.split('<')[0].split('(')[0]] += tt
                if 'potrf_reset_info' in nm:
                    vcalls = cnt
                continue
            hit = None
            for k, rx in BLOCKED:
                if rx.search(nm):
                    hit = k; break
            if hit is None:
                other.append((tt, cnt, nm)); continue
            blk[hit] += tt
            if hit == 'fixup':
                bcalls = cnt // PANELS[(t, n)]
    vtot = sum(vend.values()); btot = sum(blk.values())
    print(f"=== {t} n={n} batch={b} cfg={cfg}   "
          f"vendor {vcalls} calls, blocked {bcalls} calls")
    if vcalls:
        print(f"    cuSOLVER          {vtot / vcalls:9.2f} ms/call")
    if bcalls:
        print(f"    blocked total     {btot / bcalls:9.2f} ms/call")
        for k, v in blk.most_common():
            print(f"      {k:12}    {v / bcalls:9.2f} ms/call  {100 * v / btot:5.1f}%")
    for tt, cnt, nm in other:
        print(f"    UNCLASSIFIED {tt:8.2f} ms {cnt:6} x {nm[:70]}")
    print()
