# Borne "congelee": duals du LP RLT1 racine, reevalues a chaque noeud par un seul LAP.
# usage: frozen.py inst.dat fixes.txt
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.path.insert(0,'/tmp/claude-0/qap/sdp')
from rlt1 import load, node

def build(m,Ar,Br,Cr):
    pairs=[(a,c) for a in range(m) for c in range(a+1,m)]
    bd=[(b,d) for b in range(m) for d in range(m) if b!=d]
    bdi={p:i for i,p in enumerate(bd)}
    pidx={p:i for i,p in enumerate(pairs)}
    nx=m*m; nbd=len(bd); ny=len(pairs)*nbd
    c=np.zeros(nx+ny)
    for a in range(m):
        for b in range(m): c[a*m+b]=Ar[a,a]*Br[b,b]+Cr[a,b]
    for pi,(a,cc) in enumerate(pairs):
        base=nx+pi*nbd
        for k,(b,d) in enumerate(bd):
            c[base+k]=Ar[a,cc]*Br[b,d]+Ar[cc,a]*Br[d,b]
    rows=[];cols=[];vals=[];rhs=[]; r=0
    for a in range(m):
        for b in range(m): rows.append(r);cols.append(a*m+b);vals.append(1.0)
        rhs.append(1.0); r+=1
    for b in range(m):
        for a in range(m): rows.append(r);cols.append(a*m+b);vals.append(1.0)
        rhs.append(1.0); r+=1
    for pi,(a,cc) in enumerate(pairs):
        base=nx+pi*nbd
        for d in range(m):
            for b in range(m):
                if b!=d: rows.append(r);cols.append(base+bdi[(b,d)]);vals.append(1.0)
            rows.append(r);cols.append(cc*m+d);vals.append(-1.0); rhs.append(0.0); r+=1
        for b in range(m):
            for d in range(m):
                if d!=b: rows.append(r);cols.append(base+bdi[(b,d)]);vals.append(1.0)
            rows.append(r);cols.append(a*m+b);vals.append(-1.0); rhs.append(0.0); r+=1
    for cc in range(m):
        for d in range(m):
            for b in range(m):
                if b==d: continue
                for a in range(m):
                    if a==cc: continue
                    if a<cc: rows.append(r);cols.append(nx+pidx[(a,cc)]*nbd+bdi[(b,d)]);vals.append(1.0)
                    else:    rows.append(r);cols.append(nx+pidx[(cc,a)]*nbd+bdi[(d,b)]);vals.append(1.0)
                rows.append(r);cols.append(cc*m+d);vals.append(-1.0); rhs.append(0.0); r+=1
    return c, sp.coo_matrix((vals,(rows,cols)),shape=(r,nx+ny)).tocsr(), np.array(rhs), pairs, bd, bdi, pidx, nx, nbd

from scipy.optimize import linear_sum_assignment
inst=sys.argv[1]
n,A,B=load(inst)
m,Ar,Br,Cr,K=node(n,A,B,{})
assert K==0
c,Aeq,rhs,pairs,bd,bdi,pidx,nx,nbd=build(n,Ar,Br,Cr)
t0=time.time()
res=linprog(c,A_eq=Aeq,b_eq=rhs,bounds=(0,None),method='highs')
root=res.fun; rc=np.maximum(c-Aeq.T@res.eqlin.marginals,0)  # couts reduits >= 0 (projetes)
blam=float(res.eqlin.marginals@rhs)
import numpy as _np; _np.save('/tmp/claude-0/qap/runs/split/rc14.npy',rc)
print(f'# racine LP={root:.2f} b.lambda={blam:.2f} t={time.time()-t0:.0f}s status={res.status}',flush=True)
rcx=rc[:nx].reshape(n,n)
def rcy(a,b,cc,d):
    # cout reduit de la paire a->b, cc->d (a != cc, b != d)
    if a<cc: return rc[nx+pidx[(a,cc)]*nbd+bdi[(b,d)]]
    return rc[nx+pidx[(cc,a)]*nbd+bdi[(d,b)]]
for line in open(sys.argv[2]):
    line=line.strip()
    if not line: continue
    fx=dict(tuple(map(int,x.split(':'))) for x in line.split())
    F=[i for i in range(n) if i not in fx]; L=[j for j in range(n) if j not in set(fx.values())]
    mm=len(F)
    s=blam+sum(rcx[i,fx[i]] for i in fx)
    ks=list(fx)
    s+=sum(rcy(ks[p],fx[ks[p]],ks[q],fx[ks[q]]) for p in range(len(ks)) for q in range(p+1,len(ks)))
    R=np.zeros((mm,mm))
    for ai,a in enumerate(F):
        for bi,b in enumerate(L):
            R[ai,bi]=rcx[a,b]+sum(rcy(a,b,k,fx[k]) for k in fx)
    ff=0.0
    for p1 in range(mm):
        for q1 in range(p1+1,mm):
            a1,c1=F[p1],F[q1]; best=1e18
            for b1 in L:
                for d1 in L:
                    if b1!=d1:
                        v=rcy(a1,b1,c1,d1)
                        if v<best: best=v
            ff+=best
    ri,ci=linear_sum_assignment(R)
    print(f'{line} | frozen {s+R[ri,ci].sum():.2f} ff {ff:.2f}',flush=True)
