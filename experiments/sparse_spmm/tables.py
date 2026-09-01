#!/usr/bin/env python3
"""The small named tables the README quotes, one function per table."""
import csv, collections, sys

D = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/sparse_spmm'


def rows(pass_):
    return list(csv.DictReader(open(f'{D}/{pass_}/joined.csv')))


def cfedge():
    ns = [8, 9, 12, 16, 17, 20, 25, 32, 50]
    for p in ('cfedge1', 'cfedge2'):
        g = collections.defaultdict(dict)
        for r in rows(p):
            g[(r['typ'], int(r['pattern']))][int(r['nrhs'])] = float(r['ratio'])
        print(f'{p} -- transA=NoTrans, m=2048, nnz/row=16, batch=512, transB=Trans '
              f'(pat 0 = banded, 1 = scattered)')
        print('%-9s %-4s' % ('type', 'pat') + ''.join('%8d' % n for n in ns))
        for k in sorted(g):
            print('%-9s %-4d' % k + ''.join('%8.3f' % g[k].get(n, float("nan")) for n in ns))
        print()


def satext():
    for p in ('sat1', 'sat2'):
        g = collections.defaultdict(dict)
        for r in rows(p):
            g[(r['typ'], int(r['nrhs']), int(r['pattern']))][int(r['batch'])] = (
                1000 * float(r['t_vendor']) / int(r['batch']),
                1000 * float(r['t_native']) / int(r['batch']),
                float(r['ratio']))
        bs = [1024, 2048, 4096, 8192]
        print(f'{p} -- lanczos shape m=1024 nnz/row=3, transA=NoTrans: '
              f'per-item us (vendor / native) and ratio')
        print('%-9s %-5s %-4s' % ('type', 'nrhs', 'pat')
              + ''.join('%24s' % f'b={b}' for b in bs))
        for k in sorted(g):
            line = '%-9s %-5d %-4d' % k
            for b in bs:
                if b in g[k]:
                    v, n, r = g[k][b]
                    line += '%24s' % f'{v:.3f}/{n:.3f} r={r:.3f}'
                else:
                    line += '%24s' % '-'
            print(line)
        print()


def scatter_ladder():
    for p in ('scl1', 'scl2'):
        try:
            rs = rows(p)
        except FileNotFoundError:
            continue
        bs = [128, 256, 512, 1024]
        g = collections.defaultdict(dict)
        for r in rs:
            g[(r['typ'], int(r['m']), int(r['nnzrow']), int(r['nrhs']))][int(r['batch'])] = (
                1000 * float(r['t_vendor']) / int(r['batch']),
                1000 * float(r['t_native']) / int(r['batch']),
                float(r['ratio']))
        print(f'{p} -- transA=Trans, scattered pattern: per-item us (vendor/native), ratio')
        print('%-9s %-6s %-4s %-5s' % ('type', 'm', 'nnz', 'nrhs')
              + ''.join('%24s' % f'b={b}' for b in bs))
        for k in sorted(g):
            line = '%-9s %-6d %-4d %-5d' % k
            for b in bs:
                if b in g[k]:
                    v, n, r = g[k][b]
                    line += '%24s' % f'{v:.3f}/{n:.3f} r={r:.3f}'
                else:
                    line += '%24s' % '-'
            print(line)
        print()


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'cfedge'): cfedge()
    if which in ('all', 'satext'): satext()
    if which in ('all', 'scl'): scatter_ladder()
