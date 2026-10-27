import sys,collections,math
def load(fn):
    out=[]
    for l in open(fn):
        if not l.startswith('N ') or ' | ' not in l: continue
        head,tail=l.split(' | ')
        pre,rest=head.split(' F ',1)
        t=pre.split(); m=int(t[3]); z=float(t[5]); dk,dl=map(int,t[7].split(':'))
        Fs,rest=rest.split(' L ',1); Ls,rest=rest.split(' U ',1); Us,Ss=rest.split(' S ',1)
        F=list(map(int,Fs.split(','))); L=list(map(int,Ls.split(',')))
        U=list(map(float,Us.split(','))); U=[U[i*m:(i+1)*m] for i in range(m)]
        S=[int(x.split('/')[0]) for x in Ss.split()]
        d=int(tail.split()[1]); b=int(tail.split()[3])
        out.append(dict(m=m,z=z,dk=dk,dl=dl,F=F,L=L,U=U,S=S,d=d,b=b))
    return out
if __name__=='__main__':
    for fn in sys.argv[1:]:
        X=load(fn); X=[x for x in X if x['d']>0]
        D=sum(x['d'] for x in X); B=sum(x['b'] for x in X)
        print(fn,len(X),'nodes; default',D,'best',B,'ratio %.2f'%(D/B))
        # fixed rules: always row of grid cell c (smallest free index), always col, etc.
        r0=sum(x['S'][0] for x in X); c0=sum(x['S'][x['m']] for x in X)
        print('  first free row (cell)',r0,'first free col (facility)',c0)
        # best by kind
        rb=sum(min(x['S'][:x['m']]) for x in X); cb=sum(min(x['S'][x['m']:]) for x in X)
        print('  best row only',rb,'best col only',cb)
        # which actual cell is best when row is best
        cc=collections.Counter(); fc=collections.Counter()
        for x in X:
            i=x['S'].index(x['b'])
            if i<x['m']: cc[x['F'][i]]+=1
            else: fc[x['L'][i-x['m']]]+=1
        print('  best rows (cell id):',cc.most_common(6)); print('  best cols (facility id):',fc.most_common(6))
