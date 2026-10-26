# RLT1 (Adams-Johnson niveau 1, avec symetrie) en LP exact via HiGHS.
# Exploration Python seulement: donne le PLAFOND de la montee duale de Hahn-Grant a un noeud.
# usage: rlt1.py inst.dat "i:j i:j ..."   -> imprime K+LP
import sys, time
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog

def load(inst):
    t=open(inst).read().split(); n=int(t[0])
    v=np.array(list(map(float,t[1:1+2*n*n])))
    return n, v[:n*n].reshape(n,n), v[n*n:].reshape(n,n)

def node(n,A,B,fx):
    F=[i for i in range(n) if i not in fx]; vals=set(fx.values())
    L=[j for j in range(n) if j not in vals]; m=len(F)
    K=sum(A[i,q]*B[fx[i],fx[q]] for i in fx for q in fx)
    Ar=A[np.ix_(F,F)].copy(); Br=B[np.ix_(L,L)].copy()
    Cr=np.zeros((m,m))
    for a,i in enumerate(F):
        for b,j in enumerate(L):
            Cr[a,b]=2*sum(A[i,k]*B[j,fx[k]] for k in fx)
    return m,Ar,Br,Cr,K

def rlt1(m,Ar,Br,Cr,K):
    pairs=[(a,c) for a in range(m) for c in range(a+1,m)]
    bd=[(b,d) for b in range(m) for d in range(m) if b!=d]
    bdi={p:i for i,p in enumerate(bd)}
    nx=m*m; nbd=len(bd); ny=len(pairs)*nbd
    c=np.zeros(nx+ny)
    for a in range(m):
        for b in range(m): c[a*m+b]=Ar[a,a]*Br[b,b]+Cr[a,b]
    for pi,(a,cc) in enumerate(pairs):
        base=nx+pi*nbd
        for k,(b,d) in enumerate(bd):
            c[base+k]=Ar[a,cc]*Br[b,d]+Ar[cc,a]*Br[d,b]
    rows=[];cols=[];vals=[];rhs=[]
    r=0
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
    # famille 3 (somme sur l'usine partenaire) : pour chaque (c,d) et b != d,
    # somme_{a<c} y[(a,c)][b][d] + somme_{a>c} y[(c,a)][d][b] = x[c][d]
    pidx={pq:i for i,pq in enumerate(pairs)}
    for cc in range(m):
        for d in range(m):
            for b in range(m):
                if b==d: continue
                for a in range(m):
                    if a==cc: continue
                    if a<cc: rows.append(r);cols.append(nx+pidx[(a,cc)]*nbd+bdi[(b,d)]);vals.append(1.0)
                    else:    rows.append(r);cols.append(nx+pidx[(cc,a)]*nbd+bdi[(d,b)]);vals.append(1.0)
                rows.append(r);cols.append(cc*m+d);vals.append(-1.0); rhs.append(0.0); r+=1
    Aeq=sp.coo_matrix((vals,(rows,cols)),shape=(r,nx+ny)).tocsr()
    t0=time.time()
    res=linprog(c,A_eq=Aeq,b_eq=np.array(rhs),bounds=(0,None),method='highs')
    return K+res.fun, time.time()-t0, res.status

if __name__=='__main__':
    n,A,B=load(sys.argv[1])
    fx=dict(tuple(map(int,x.split(':'))) for x in sys.argv[2].split()) if len(sys.argv)>2 and sys.argv[2] else {}
    m,Ar,Br,Cr,K=node(n,A,B,fx)
    v,t,st=rlt1(m,Ar,Br,Cr,K)
    print(f'm={m} rlt1={v:.2f} t={t:.1f}s status={st}')
