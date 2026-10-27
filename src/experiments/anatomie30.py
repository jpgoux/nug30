"""Anatomy of a nug instance: flow matrix structure, exact automorphisms, near-twin facilities."""
import sys, itertools, numpy as np
fn = sys.argv[1]
t = open(fn).read().split(); n = int(t[0]); v = np.array(list(map(int, t[1:1 + 2 * n * n])))
A = v[:n * n].reshape(n, n); B = v[n * n:].reshape(n, n)   # A: distances (grid), B: flows
print("n", n, "A max", A.max(), "B max", B.max(), "B zeros off-diag", int((B == 0).sum() - n), "of", n * (n - 1))
deg = B.sum(1); print("flow row sums (sorted):", sorted(deg.tolist()))
print("flow values histogram:", {int(k): int(c) for k, c in zip(*np.unique(B[np.triu_indices(n, 1)], return_counts=True))})
# exact automorphisms of a symmetric matrix by backtracking with signature refinement
def automorphisms(M, limit=100000):
    n = len(M); sig = [tuple(sorted(M[i])) for i in range(n)]
    cnt = [0]; auts = []
    def rec(p, used):
        i = len(p)
        if i == n: auts.append(tuple(p)); return
        for j in range(n):
            if used[j] or sig[j] != sig[i] or M[j][j] != M[i][i]: continue
            if any(M[j][p[k]] != M[i][k] for k in range(i)): continue
            cnt[0] += 1
            if cnt[0] > limit: return
            p.append(j); used[j] = 1; rec(p, used); p.pop(); used[j] = 0
    rec([], [0] * n); return auts
autA = automorphisms(A.tolist()); autB = automorphisms(B.tolist())
print("automorphisms: distances", len(autA), "flows", len(autB))
# near twins: Hamming distance between flow rows (ignoring the two mutual entries)
pairs = []
for i in range(n):
    for k in range(i + 1, n):
        d = sum(1 for q in range(n) if q != i and q != k and B[i][q] != B[k][q])
        l1 = sum(abs(int(B[i][q]) - int(B[k][q])) for q in range(n) if q != i and q != k)
        pairs.append((d, l1, i, k))
pairs.sort(); print("closest facility pairs (entries differing, L1 difference, i, k):", pairs[:8])
# residual grid symmetry along the eccentric-cell branching: cells ordered by row sum of A
rs = A.sum(1); order = np.argsort(-rs, kind='stable'); print("cells by eccentricity (A row sum):", [(int(c), int(rs[c])) for c in order[:8]])
def stab_size(fixed):
    return sum(1 for g in autA if all(g[c] == c for c in fixed))
for d in range(0, 7): print("cells fixed:", [int(c) for c in order[:d]], "stabilizer of the grid group:", stab_size(order[:d]))
