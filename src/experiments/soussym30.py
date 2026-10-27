"""Exact sub-symmetries of nug subproblems: automorphisms of the distance matrix on random free-cell sets and of the
flow matrix on random free-facility sets; costs modulo 4 and 8 over random permutations."""
import sys, random, numpy as np
from anatomie30 import automorphisms
fn = sys.argv[1]; random.seed(1)
t = open(fn).read().split(); n = int(t[0]); v = np.array(list(map(int, t[1:1 + 2 * n * n])))
A = v[:n * n].reshape(n, n); B = v[n * n:].reshape(n, n)
for m in (26, 24, 22, 20, 18, 16, 14, 12):
    ca = cb = 0; trials = 300; maxa = maxb = 1
    for _ in range(trials):
        S = random.sample(range(n), m); T = random.sample(range(n), m)
        a = len(automorphisms(A[np.ix_(S, S)].tolist())); b = len(automorphisms(B[np.ix_(T, T)].tolist()))
        ca += a > 1; cb += b > 1; maxa = max(maxa, a); maxb = max(maxb, b)
    print(f"m={m}: free-cell sets with a nontrivial isometry {ca}/{trials} (largest group {maxa}); free-facility sets with a nontrivial flow automorphism {cb}/{trials} (largest {maxb})")
mods = {4: set(), 8: set(), 3: set()}
for _ in range(20000):
    p = list(range(n)); random.shuffle(p); c = int(sum(A[i][k] * B[p[i]][p[k]] for i in range(n) for k in range(n)))
    for q in mods: mods[q].add(c % q)
print("cost residues over 20000 random permutations:", {q: sorted(s) for q, s in mods.items()})
