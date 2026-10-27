"""Statistics of Frank-Wolfe trajectories written by qap_bb -fwlog.
Line: depth m nfw nfw2 it exit, then per iteration (target - best, f(X) - target).
exit: 1 fathomed, 2 stopped by the nfw2 rule, 3 iteration limit, 4 other."""
import sys, collections
calls = collections.Counter(); iters = collections.Counter()
after_f = collections.Counter()     # iterations spent after f(X) < target first holds (fathom impossible)
fath_at = collections.Counter()     # fathomed calls: iteration of the fathom
for line in open(sys.argv[1]):
    t = line.split()
    d, m, nfw, nfw2, it, ex = map(int, t[:6])
    v = list(map(float, t[6:]))
    gz, gf = v[0::2], v[1::2]
    kind = 'main' if nfw2 < nfw else 'look'
    calls[kind, ex] += 1; iters[kind, ex] += it
    if ex != 1:
        k0 = next((k for k, x in enumerate(gf) if x < 0), None)
        if k0 is not None: after_f[kind, ex] += it - k0 - 1
    else:
        fath_at[kind, min(it, 40)] += 1
tot = sum(iters.values())
print("kind exit calls iterations share after_f<target")
for k in sorted(calls): print(k, calls[k], iters[k], "%.3f" % (iters[k] / tot), after_f[k])
