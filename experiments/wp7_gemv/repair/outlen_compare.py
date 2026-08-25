import sys
D='/home/jonaslacour/.claude/jobs/20812aa0/tmp/wp7/repair/'
def load(f):
    d={}
    for line in open(D+f):
        p=line.strip().split(',')
        if len(p)<10: continue
        grp,ty,m,n,b,tr,route=p[1],p[2],p[3],p[4],p[5],p[6],p[7]
        d[(grp,ty,int(m),int(n),int(b),'vendor' if 'vendor' in route else 'native')]=float(p[11])
    return d
b4=load('sw_body4.csv'); b1=load('sw_body1.csv')
keys=sorted({k[:5] for k in b1 if k[5]=='native'})
print(f"{'group':9} {'ty':8} {'m':>3} {'batch':>6} {'vendor':>8} {'body1':>8} {'body4':>8} {'r1':>6} {'r4':>6}  pick")
for k in keys:
    v=b1[k+('vendor',)]; n1=b1[k+('native',)]; n4=b4[k+('native',)]
    print(f"{k[0]:9} {k[1]:8} {k[2]:>3} {k[4]:>6} {v:8.1f} {n1:8.1f} {n4:8.1f} {n1/v:6.3f} {n4/v:6.3f}  {'body4' if n4>n1 else 'BODY1'}")
