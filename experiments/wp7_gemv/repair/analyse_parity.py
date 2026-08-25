import csv, sys, statistics
from collections import defaultdict

D = '/home/jonaslacour/BatchLAS/.claude/worktrees/vendor-independence-plan/experiments/wp7_gemv/'

def load(path):
    d = defaultdict(dict)
    fo = 0
    for r in csv.DictReader(open(path)):
        if r['route'] == 'FAILED' or not r['GBs']:
            print('FAILED ROW', r['arm'], r['type'], r['m'], r['n'], r['transA']); continue
        if r.get('relerr') and float(r['relerr']) != 0.0:
            print('NONZERO RELERR', r)
        fo = max(fo, int(r.get('foreign') or 0))
        k = (r['type'], int(r['out_len']), int(r['red_len']), int(r['batch']), r['transA'])
        d[k][r['arm']] = (float(r['GBs']), r['route'])
    return d, fo

def ratios(d):
    out = {}
    for k, v in d.items():
        arm = 'native:direct' if k[4] == 'N' else 'native:cta'
        if 'vendor' not in v or arm not in v: continue
        # route column must agree with the pin
        assert v[arm][1] == arm, f'route column lies: {k} {v[arm]}'
        assert v['vendor'][1] == 'vendor:auto', f'route column lies: {k} {v["vendor"]}'
        out[k] = (v[arm][0] / v['vendor'][0], v['vendor'][0], v[arm][0])
    return out

files = sys.argv[1:]
sets = []
for f in files:
    d, fo = load(D + f)
    print(f'{f}: {len(d)} cells, max foreign processes seen = {fo}')
    sets.append(ratios(d))

keys = sorted(set(sets[0]) & set(sets[1])) if len(sets) > 1 else sorted(sets[0])
print(f'\ncells common to all passes: {len(keys)}')

worst = []
for k in keys:
    rs = [s[k][0] for s in sets]
    worst.append((min(rs), k, rs, sets[0][k][1], sets[0][k][2]))
worst.sort()

below50 = [w for w in worst if w[0] < 0.50]
below85 = [w for w in worst if w[0] < 0.85]
print(f'\nCELLS BELOW 0.50x (worst of passes): {len(below50)}')
for w in below50:
    print(f'   {w[0]:.3f}  type={w[1][0]:8} out={w[1][1]:5} red={w[1][2]:5} batch={w[1][3]:5} transA={w[1][4]}  passes={[round(x,3) for x in w[2]]}  vendor={w[3]:.1f} native={w[4]:.1f}')
print(f'\nCELLS BELOW 0.85x (worst of passes): {len(below85)}')
for w in below85:
    print(f'   {w[0]:.3f}  type={w[1][0]:8} out={w[1][1]:5} red={w[1][2]:5} batch={w[1][3]:5} transA={w[1][4]}  passes={[round(x,3) for x in w[2]]}  vendor={w[3]:.1f} native={w[4]:.1f}')

allr = [min(s[k][0] for s in sets) for k in keys]
print(f'\nsummary over {len(keys)} cells (worst of passes): min {min(allr):.3f}  median {statistics.median(allr):.3f}  max {max(allr):.3f}')
print(f'  >= 0.85x : {sum(1 for r in allr if r>=0.85)}/{len(allr)}')
print(f'  in [0.95,1.05] : {sum(1 for r in allr if 0.95<=r<=1.05)}/{len(allr)}')
print(f'  >= 1.15x : {sum(1 for r in allr if r>=1.15)}/{len(allr)}')

if len(sets) > 1:
    spread = [max(s[k][0] for s in sets)/min(s[k][0] for s in sets) for k in keys]
    print(f'  cross-pass ratio spread: median {statistics.median(spread):.4f}  worst {max(spread):.4f}')

# the NoTrans short-output family specifically
fam = [w for w in worst if w[1][4]=='N' and w[1][1] < 32]
if fam:
    fr = [w[0] for w in fam]
    print(f'\nTHE FIXED FAMILY (transA=N, out_len < 32): {len(fam)} cells')
    print(f'  min {min(fr):.3f}  median {statistics.median(fr):.3f}  max {max(fr):.3f}  below 0.50x: {sum(1 for r in fr if r<0.50)}')

if len(sets) > 1:
    print('\ncells with cross-pass spread > 1.10:')
    for k in keys:
        lo = min(s[k][0] for s in sets); hi = max(s[k][0] for s in sets)
        if hi/lo > 1.10:
            print(f'   spread {hi/lo:.3f}  type={k[0]:8} out={k[1]:5} red={k[2]:5} batch={k[3]:5} transA={k[4]}  passes={[round(s[k][0],3) for s in sets]}')
