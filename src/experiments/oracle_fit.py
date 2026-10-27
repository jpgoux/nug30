import sys,math,random
import numpy as np
from oracle_load import load
from oracle_rules import feats, ecc, flow
def rows(X):
    R=[]
    for x in X:
        f=feats(x); m=x['m']; S=x['S']
        ls=[math.log(s) for s in S]; mu=sum(ls)/len(ls)
        E=[ecc[c] for c in x['F']]; Fl=[flow[c] for c in x['L']]
        me,se=np.mean(E),np.std(E)+1e-9; mf,sf=np.mean(Fl),np.std(Fl)+1e-9
        for i,q in enumerate(f):
            st=(ecc[x['F'][q['line']]]-me)/se if q['k']==0 else (flow[x['L'][q['line']]]-mf)/sf
            R.append((x, i, ls[i]-mu, [1.0 if q['k']==0 else 0.0, math.log(q['soft'][0.05]+1), q['cnt']/m, st if q['k']==0 else 0, st if q['k']==1 else 0]))
    return R
def evaluate(X,w):
    tot=0
    for x in X:
        f=feats(x); m=x['m']
        E=[ecc[c] for c in x['F']]; Fl=[flow[c] for c in x['L']]
        me,se=np.mean(E),np.std(E)+1e-9; mf,sf=np.mean(Fl),np.std(Fl)+1e-9
        sc=[]
        for q in f:
            st=(ecc[x['F'][q['line']]]-me)/se if q['k']==0 else (flow[x['L'][q['line']]]-mf)/sf
            v=[1.0 if q['k']==0 else 0.0, math.log(q['soft'][0.05]+1), q['cnt']/m, st if q['k']==0 else 0, st if q['k']==1 else 0]
            sc.append(float(np.dot(w,v)))
        tot+=x['S'][int(np.argmin(sc))]
    return tot
for fn in sys.argv[1:]:
    X=[x for x in load(fn) if x['d']>0]; random.seed(1); random.shuffle(X)
    h=len(X)//2; tr,te=X[:h],X[h:]
    R=rows(tr); M=np.array([r[3] for r in R]); y=np.array([r[2] for r in R])
    w,*_=np.linalg.lstsq(M,y,rcond=None)
    d_te=sum(x['d'] for x in te); b_te=sum(x['b'] for x in te)
    print(fn,'w=',np.round(w,3),'test: default',d_te,'learned',evaluate(te,w),'best',b_te, 'ratio %.2f'%(evaluate(te,w)/d_te))
