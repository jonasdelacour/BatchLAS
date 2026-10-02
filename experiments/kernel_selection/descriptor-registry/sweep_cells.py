import json,sys,collections
import os
root=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../benchmarks/results/routing/')
keys=collections.Counter(); cells=set()
for f in ['sm120_potrf_sweep.jsonl','sm120_potrf_sweep_edges.jsonl']:
    for line in open(root+f):
        r=json.loads(line); keys.update(r.keys())
        up='upper' if r['op']=='potrf_upper' else 'lower'
        cells.add((r['dtype'],str(up).lower(),int(r['n']),int(r['batch'])))
print(keys.most_common(), file=sys.stderr)
print(len(cells), file=sys.stderr)
print(collections.Counter(c[1] for c in cells), file=sys.stderr)
with open(sys.argv[1],'w') as o:
    for c in sorted(cells): o.write('%s %s %d %d\n'%c)
