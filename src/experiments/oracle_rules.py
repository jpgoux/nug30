import sys,math
from oracle_load import load
t=open('/home/claude/nug30/data/qaplib/nug20.dat').read().split(); n=int(t[0]); v=list(map(int,t[1:]))
A=[v[i*n:(i+1)*n] for i in range(n)]; B=[v[n*n+i*n:n*n+(i+1)*n] for i in range(n)]
ecc=[sum(r) for r in A]; flow=[sum(r) for r in B]
print('flow by facility', sorted(range(n),key=lambda i:-flow[i])[:6], 'ecc by cell', sorted(range(n),key=lambda i:-ecc[i])[:6])
thr=2568
def feats(x):
    m=x['m']; z=x['z']; U=x['U']; out=[]
    for k in range(2):
        for line in range(m):
            ch=[U[line][o] if k==0 else U[o][line] for o in range(m)]
            g=[thr-z-u for u in ch]
            cnt=sum(1 for q in g if q>=0)
            out.append(dict(k=k,line=line,cnt=cnt,sumU=sum(ch),
                soft={kap:sum(math.exp(kap*q) for q in g if q>=0) for kap in (0.02,0.05,0.1,0.2)},
                stat=ecc[x['F'][line]] if k==0 else flow[x['L'][line]]))
    return out
for fn in sys.argv[1:]:
    X=[x for x in load(fn) if x['d']>0]
    res={}
    for x in X:
        f=feats(x); S=x['S']; m=x['m']
        def pick(key): return S[min(range(2*m),key=key)]
        rules={
         'default':x['d'],'best':x['b'],
         'maxflow col':S[m+max(range(m),key=lambda j:flow[x['L'][j]])],
         'maxecc row':S[max(range(m),key=lambda i:ecc[x['F'][i]])],
         'cnt,sumU (rule2 nosym)':pick(lambda i:(f[i]['cnt'],-f[i]['sumU'])),
        }
        for kap in (0.02,0.05,0.1,0.2): rules['soft%.2f'%kap]=pick(lambda i:f[i]['soft'][kap])
        for kap in (0.02,0.05,0.1): rules['soft%.2f cols'%kap]=S[m+min(range(m),key=lambda j:f[m+j]['soft'][kap])]
        for kap in (0.02,0.05,0.1): rules['soft%.2f rows'%kap]=S[min(range(m),key=lambda j:f[j]['soft'][kap])]
        for r,vv in rules.items(): res[r]=res.get(r,0)+vv
    print(fn,len(X)); 
    for r,vv in sorted(res.items(),key=lambda a:a[1]): print('  %-26s %8d  %.2f'%(r,vv,vv/res['default']))
