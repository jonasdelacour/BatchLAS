import collections, os, sys

store = '.route-diff'
def load(label):
    d = {}
    for line in open(os.path.join(store, label + '.routes')):
        f = line.rstrip('\n').split(',')
        # kind,op,scalar,backend,shape_class,origin,algo,c13,c14,uplo,side,diag,transA,transB
        key = (f[1], f[2], f[3], f[4], f[7], f[8], f[9], f[10], f[11], f[12], f[13])
        d.setdefault(key, set()).add((f[5], f[6]))
    return d

a = load(sys.argv[1])
b = load(sys.argv[2])

moved, added, removed = [], [], []
for k in sorted(set(a) | set(b)):
    ra, rb = a.get(k), b.get(k)
    if ra is None:
        added.append((k, rb))
    elif rb is None:
        removed.append((k, ra))
    elif ra != rb:
        moved.append((k, ra, rb))

def isauto(k): return k[2] == 'AUTO'

print('=== MOVED DECISIONS (same key, different route) ===')
byop = collections.Counter()
for k, ra, rb in moved:
    byop[(k[0], k[2] == 'AUTO')] += 1
for (op, auto), n in sorted(byop.items()):
    print(f'  op={op:8s} backend={"AUTO(pure-layer fabrication)" if auto else "REAL(library decision)"}  {n} keys')
print()
print('  library (backend != AUTO) moved decisions, in full:')
for k, ra, rb in moved:
    if isauto(k):
        continue
    op, scalar, backend, shape_class = k[0], k[1], k[2], k[3]
    ta, tb = k[9], k[10]
    print(f'    {op} {scalar:15s} {backend:7s} shape_class={shape_class:5s} '
          f'transA={ta} transB={tb}  {sorted(ra)} -> {sorted(rb)}')

print()
print('=== KEYS ONLY IN', sys.argv[2], '(new shapes) ===')
c = collections.Counter((k[0], k[2]) for k, _ in added)
for (op, be), n in sorted(c.items()):
    print(f'  op={op:8s} backend={be:7s} {n} keys')
print('=== KEYS ONLY IN', sys.argv[1], '(disappeared shapes) ===')
c = collections.Counter((k[0], k[2]) for k, _ in removed)
for (op, be), n in sorted(c.items()):
    print(f'  op={op:8s} backend={be:7s} {n} keys')
