import re,collections,sys
cur=None; c=collections.OrderedDict(); other=collections.Counter()
for l in open(sys.argv[1]):
    if l.startswith('###'): cur=l.split()[1]; c.setdefault(cur,collections.Counter()); continue
    m=re.match(r'(\w+) (\w+) .*-> (\S+)\s*(\[\w+\])?',l)
    if m: c[cur][f"{m.group(1)}->{m.group(3)}{m.group(4) or ''}"]+=1
    elif l.strip(): other[re.sub(r'[0-9]+','#',l.strip())[:110]]+=1
for k,v in c.items(): print(k, dict(v))
print('--- other stderr kinds'); [print(n,s) for s,n in other.most_common(15)]
