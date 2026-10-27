import numpy as np, sys
from agg2 import load
def rd(inst):
    t=open(f'/home/claude/nug30/data/qaplib/{inst}.dat').read().split(); n=int(t[0]); v=list(map(int,t[1:]))
    return np.array(v[:n*n]).reshape(n,n).astype(float), np.array(v[n*n:2*n*n]).reshape(n,n).astype(float)
def feats(inst,fn,thr):
    A,B=rd(inst); out=[]
    for x in load(fn):
        if x['d']<=0: continue
        m=x['m']; U=np.array(x['U']); z=x['z']; S=np.array(x['S'],float); F=x['F']; L=x['L']
        ok=(z+U)<=thr; live=np.r_[ok.sum(1),ok.sum(0)]; mn=live.min(); c=np.where(live==mn)[0]
        if len(c)<2: continue
        As=A[np.ix_(F,F)].sum(1); Bs=B[np.ix_(L,L)].sum(1); Ag=A[F].sum(1); Bg=B[L].sum(1)
        rows=[]
        for i in c:
            r=i<m; line=U[i] if r else U[:,i-m]; okl=ok[i] if r else ok[:,i-m]
            lu=line[okl]; gap=thr-z
            e_dyn=(As[i]-As.mean())/As.std() if r else (Bs[i-m]-Bs.mean())/Bs.std()
            e_st=(Ag[i]-Ag.mean())/Ag.std() if r else (Bg[i-m]-Bg.mean())/Bg.std()
            rows.append([r, e_dyn, e_st, line.sum()/gap/m, (lu.mean()/gap if len(lu) else 1), (lu.min()/gap if len(lu) else 1), (lu.max()/gap if len(lu) else 1), np.sort(line)[1]/gap if m>1 else 0])
        out.append((np.array(rows),S[c],x['d']))
    return out
sets={'n20L3':feats('nug20','o20_L3c.txt',2569),'n20L4':feats('nug20','o20_L4.txt',2569),'n22L3':feats('nug22','o22_L3.txt',3595),'n22L4':feats('nug22','o22_L4.txt',3595),'n20L7':feats('nug20','o20_L7.txt',2569)}
names=['row','ecc_dyn','ecc_st','sumU','meanliveU','minliveU','maxliveU','2ndU']
def score(data,w):
    tot=0;D=0
    for X,S,d in data: tot+=S[np.argmax(X@w)]; D+=d
    return D/tot
def oracle(data): return sum(d for _,_,d in data)/sum(S.min() for _,S,_ in data)
rng=np.random.default_rng(1)
for tr in sets:
    best=None
    data=sets[tr]
    for it in range(3000):
        w=rng.normal(size=len(names)) if it else np.eye(len(names))[1]
        sc=score(data,w)
        if best is None or sc>best[0]: best=(sc,w)
    w=best[1]
    print(f'train {tr}: fit {best[0]:.2f} (oracle-tie {oracle(data):.2f}) | '+' '.join(f'{k}:{score(v,w):.2f}' for k,v in sets.items() if k!=tr), '| w',' '.join(f'{n}={x:+.2f}' for n,x in zip(names,w/np.abs(w).max())))
print('rowecc_dyn only:',' '.join(f'{k}:{score(v,np.array([5,1,0,0,0,0,0,0.])):.2f}' for k,v in sets.items()))
