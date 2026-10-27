import numpy as np, collections, sys
from agg2 import load
inst=sys.argv[1]; thr=float(sys.argv[2])
t=open(f'/home/claude/nug30/data/qaplib/{inst}.dat').read().split(); n=int(t[0]); v=list(map(int,t[1:]))
A=np.array(v[:n*n]).reshape(n,n); B=np.array(v[n*n:2*n*n]).reshape(n,n)
for fn in sys.argv[3:]:
    X=[x for x in load(fn) if x['d']>0]; tot=collections.Counter()
    for x in X:
        m=x['m']; U=np.array(x['U']); z=x['z']; S=x['S']; F=x['F']; L=x['L']
        ok=(z+U)<=thr
        live=list(ok.sum(1))+list(ok.sum(0))
        mn=min(live); c=[i for i in range(2*m) if live[i]==mn]
        def pick(key): return S[max(c,key=key)]
        rowecc=lambda i: sum(A[F[i]][F[k]] for k in range(m)) if i<m else -1e18
        colecc=lambda i: sum(B[L[i-m]][L[k]] for k in range(m)) if i>=m else -1e18
        def lines(i): return U[i] if i<m else U[:,i-m]
        def okl(i): return ok[i] if i<m else ok[:,i-m]
        f={'first':lambda i:-i,'rowecc':rowecc,'colmaxflow':colecc,'colminflow':lambda i:-colecc(i) if i>=m else -1e18,
           'sumU':lambda i:lines(i).sum(),'sumliveU':lambda i:lines(i)[okl(i)].sum(),'minliveU':lambda i:lines(i)[okl(i)].min() if okl(i).any() else 1e9,
           'maxliveU':lambda i:lines(i)[okl(i)].max() if okl(i).any() else 1e9,'meanU2':lambda i:(lines(i)**2).sum(),
           'colfirst':lambda i:(i>=m, -i),'rowfirst':lambda i:(i<m,-i)}
        for k,g in f.items(): tot[k]+=pick(g)
        tot['oracle_tie']+=min(S[i] for i in c); tot['def']+=x['d']; tot['best']+=x['b']
    print(fn,len(X),' '.join(f"{k}:{tot['def']/v:.2f}" for k,v in tot.items()))
