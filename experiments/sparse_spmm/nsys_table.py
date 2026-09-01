#!/usr/bin/env python3
"""Fold the nsys kernel summaries into one table: per spmm CALL, how much is GPU
kernel and how much is host chain.

The spmm kernel of each arm is identified by name -- cuSPARSE's own kernels for
the vendor arm, the SpmmGatherKernel / SpmmScaleKernel / SpmmScatterKernel
mangled names for the native arm -- and the matrix-GENERATION kernels
(fill_random_sparse_hermitian, and the Random/Zeros fills) are excluded, since
they run once during setup and are outside every timed region.

Host time per call = wall time per call (from the run's own CSV) minus the summed
kernel time per call.
"""
import csv, glob, os, re, sys

D = os.path.dirname(os.path.abspath(__file__)) + '/nsys'

SETUP = ('fill_random_sparse_hermitian', 'fill_random', 'fill_zero', 'Fill',
         'memset', 'Memset', 'fill_identity')
NATIVE = ('SpmmGatherKernel', 'SpmmScaleKernel', 'SpmmScatterKernel')


def is_setup(name):
    return any(s in name for s in SETUP)


def main():
    rows = []
    for csvf in sorted(glob.glob(D + '/*cuda_gpu_kern_sum.csv')):
        stem = csvf.split('/')[-1]
        m = re.match(r'nsys_(\w+?)_(vendor|native_direct)_(.+?)_cuda_gpu_kern_sum\.csv', stem)
        if not m:
            continue
        typ, route, tag = m.groups()
        args = tag.split('_')
        runcsv = csvf.replace('_cuda_gpu_kern_sum.csv', '.csv')
        try:
            r = list(csv.DictReader(open(runcsv)))[-1]
        except (FileNotFoundError, IndexError):
            continue
        wall_ms = float(r['avg_ms'])
        tot_ns, insts, parts = 0.0, 0, []
        for k in csv.DictReader(open(csvf)):
            name = k['Name']
            if is_setup(name):
                continue
            n = int(k['Instances'])
            t = float(k['Total Time (ns)'])
            tot_ns += t
            insts = max(insts, n)
            short = ('gather' if 'SpmmGatherKernel' in name else
                     'scale' if 'SpmmScaleKernel' in name else
                     'scatter' if 'SpmmScatterKernel' in name else
                     name.split('(')[0][:44])
            parts.append(f"{short} x{n} {float(k['Avg (ns)'])/1e6:.4f}ms")
        # calls = instances of the busiest spmm kernel (the gather/scale/scatter
        # pair both fire once per call, so max() is the call count either way)
        calls = insts if insts else 1
        kern_ms = tot_ns / 1e6 / calls
        rows.append((typ, route, ' '.join(args), wall_ms, kern_ms,
                     wall_ms - kern_ms, calls, '; '.join(parts)))

    print(f"{'type':7s} {'route':14s} {'m nnz nrhs b tB be pat tA':28s} "
          f"{'wall/call':>10s} {'kernel/call':>12s} {'host/call':>10s} {'host%':>7s}  kernels")
    for t, route, a, w, k, h, c, p in sorted(rows, key=lambda x: (x[2], x[1])):
        print(f"{t:7s} {route:14s} {a:28s} {w:10.5f} {k:12.5f} {h:10.5f} "
              f"{100*h/w:6.1f}%  {p}")


if __name__ == '__main__':
    main()
