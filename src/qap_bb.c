/*
 * qap_bb.c : branch and bound for the quadratic assignment problem with the
 * QPB bound of Anstreicher and Brixius (Math. Programming 89, 2001), computed
 * by Frank-Wolfe, in the spirit of Anstreicher, Brixius, Goux, Linderoth
 * (Math. Programming 91, 2002). Reimplementation from the published
 * descriptions, written on 2026-10-24.
 *
 * Problem: min_p sum_{i,k} A[i][k] B[p(i)][p(k)], A and B symmetric.
 * "rows" are A-indices, "columns" are B-indices, X[i][j] = 1 iff p(i) = j.
 *
 * Modes:
 *   qap_bb solve    file.dat [options]     serial branch and bound
 *   qap_bb estimate file.dat [options]     Knuth estimator of tree size/time
 *   qap_bb par      file.dat [options]     shared-pool parallel B&B (threads)
 * Options: -ub v (initial upper bound), -nfw k (max FW iterations per node),
 *          -dives d, -q q (importance sampling power), -threads t, -seed s,
 *          -sb d (strong branching down to depth d), -nosym, -tmax s, -v
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

#define NMAX 32
typedef double mat[NMAX][NMAX];

static int N;
static double A[NMAX][NMAX], B[NMAX][NMAX];
static double C0[NMAX][NMAX];   /* global linear term (after diagonal shifts) */
static const char *opt_shift = NULL;

/* ---------- options ---------- */
static int opt_nfw = 150, opt_sb = 0, opt_sym = 1, opt_verbose = 0;
static int opt_nbest = 30, opt_nfw3 = 30;
static double opt_q = 0.0;
static int opt_intprune = 1;
static int opt_fb = 0;
static int opt_umax = 0;
static int opt_glbpre = -1; static long g_glbcalls = 0, g_glbfath = 0;
static int opt_fixorder = 0;
static int opt_topd = -1; static double opt_topx = 1;   /* -topd d -topx x: x times more FW iterations down to depth d */
static double opt_r2p = 0;   /* -r2p p: rule 2 weighs each surviving child by its relative gap to the power p (0: count, as in 2001) */
static int opt_ext = 0, opt_extn = 1; static double opt_extd = 0; static __thread int tl_noext = 0;
static int g_timeig = 0; static double g_teig_tot = 0; static long g_emiss = 0, g_ehit = 0;
static void print_teig(void) { if (g_timeig) fprintf(stderr, "eig: %.2f s in %ld decompositions, %ld cache hits\n", g_teig_tot, g_emiss, g_ehit); }   /* -umax: child bounds from the best iteration for each child, not only the best iteration overall */
static int opt_dom = 0;   /* -dom: prune a node whose partial assignment is improved by swapping two fixed facilities, whatever the completion */
static long g_dom_pruned = 0;
static long g_fwstat[2][2]; static int g_gapstat = 0; static long g_gaph[2][7];
/* -fwlog file k: one line per k-th Frank-Wolfe call: depth, m, nfw, nfw2, iterations, exit (1 fathomed,
 * 2 rule nfw2, 3 limit, 4 other), then per iteration target - best and f(X) - target */
/* -fstop km kl: stop Frank-Wolfe once f(X) < target (no fathom possible) after km iterations at a node,
 * kl in a look-ahead child, instead of waiting for nfw2; 0 keeps nfw2 */
static int opt_fstopm = 0, opt_fstopl = 0;
/* -prof: cycle counters by phase (rdtsc), bank 0 = nodes of the search, bank 1 = look-ahead children */
#include <x86intrin.h>
static double now(void);
static int g_prof = 0; static unsigned long long g_pc[2][16]; static long g_pn[2][16]; static double g_prof_t0; static unsigned long long g_prof_c0;
static const char *g_pcname[16] = {"setup K Ar Br Cr", "eig (misses)", "setup LAP+build S,T", "LAP it=0", "LAP it>0", "applyMperm", "fused update", "update_st", "branch() total", "eval fathomed", "eval branching", "inherit_x", "FW other", "", "", ""};
#define PROF_T(bank, k, c0) do { if (g_prof) { g_pc[bank][k] += __rdtsc() - (c0); g_pn[bank][k]++; } } while (0)
static void print_prof(void) {
    double hz = (__rdtsc() - g_prof_c0) / (now() - g_prof_t0);
    fprintf(stderr, "prof: tsc %.3f GHz\n", hz / 1e9);
    for (int b = 0; b < 2; b++) for (int k = 0; k < 16; k++) if (g_pn[b][k])
        fprintf(stderr, "prof %s %-20s %10.3f s %12ld calls %8.2f us/call\n", b ? "look" : "main", g_pcname[k], g_pc[b][k] / hz, g_pn[b][k], g_pc[b][k] / hz / g_pn[b][k] * 1e6);
}
static FILE *g_fwlog = NULL; static long g_fwlog_k = 1, g_fwlog_n = 0; static __thread int tl_depth = 0;
static void print_glb(void) { if (opt_glbpre >= 0) fprintf(stderr, "glb: %ld calls, %ld fathomed\n", g_glbcalls, g_glbfath); }
static void print_gap(void) { if (!g_gapstat) return; for (int h = 0; h < 2; h++) { fprintf(stderr, "branching nodes with f(X) %s thr, gap thr - z in <1, <2, <5, <10, <20, <50, more:", h ? ">=" : "<"); for (int b = 0; b < 7; b++) fprintf(stderr, " %ld", g_gaph[h][b]); fputc('\n', stderr); } }   /* serial search: [fathomed][count, FW iterations] of the nodes evaluated */
static int opt_dombr = 0;   /* -dombr: also drop dominated children when branching (and count them out in rule 2) */   /* -fb: pairs eliminated by z + U at a node stay forbidden in its subtree */
#define FBIG 1e12   /* integer data: prune if bound > ub - 1 (else bound >= ub) */

/* branching strategy by depth (Brixius-Anstreicher 2001, Tables 1-2) */
typedef struct { int maxdepth, rule, nfw1, nfw2, nfw3, nbest; } rule_t;
static rule_t strat[8]; static int nstrat = 0;
/* Scale of the Frank-Wolfe iteration limits NFW1, NFW2, NFW3 of the 2001 tables.
 * Those limits were set for machines about 25 times slower; with the S, T update
 * a factor 0.7 is faster on nug20 and nug22 (more nodes, cheaper nodes). */
static double opt_nfwx = 0.7;
static void scale_strategy(void) {
    if (opt_nfwx == 1) return;
    for (int k = 0; k < nstrat; k++) {
        strat[k].nfw1 = (int)(strat[k].nfw1 * opt_nfwx + 0.5); strat[k].nfw2 = (int)(strat[k].nfw2 * opt_nfwx + 0.5);
        strat[k].nfw3 = (int)(strat[k].nfw3 * opt_nfwx + 0.5);
    }
}
static int opt_c4 = -1, opt_c3 = -1, opt_nb4 = -1, opt_nb3 = -1;   /* overrides of the strategy table */
static void set_strategy(char s) {
    nstrat = 0;
    int c4 = -1, c3 = -1;
    if (s == 'B') { c4 = 1; c3 = 2; } else if (s == 'C') { c4 = 1; c3 = 3; } else if (s == 'D') { c4 = 2; c3 = 4; }
    else if (s == 'E') { c4 = 3; c3 = 5; }
    if (s == 'A') { strat[nstrat++] = (rule_t){NMAX, 2, 150, 100, 0, 0}; scale_strategy(); return; }
    if (opt_c4 >= 0) c4 = opt_c4;
    if (opt_c3 >= 0) c3 = opt_c3;
    strat[nstrat++] = (rule_t){c4, 4, 150, 100, 50, opt_nb4 > 0 ? opt_nb4 : 20};
    strat[nstrat++] = (rule_t){c3, 3, 100, 100, 25, opt_nb3 > 0 ? opt_nb3 : 10};
    strat[nstrat++] = (rule_t){NMAX, 2, 75, 50, 0, 0};
    scale_strategy();
}
static const rule_t *rule_at(int depth) {
    for (int k = 0; k < nstrat; k++) if (depth <= strat[k].maxdepth) return &strat[k];
    return &strat[nstrat - 1];
}
static int opt_par = 1; static long g_gran = 1;   /* every permutation cost is a multiple of g_gran (-nopar: assume 1) */
/* largest achievable cost strictly below ub is g*ceil(ub/g) - g; fathom if bound > it */
static inline double prune_thr(double ub) { return !opt_intprune ? ub - 1e-6 : g_gran > 1 && isfinite(ub) ? g_gran * ceil(ub / g_gran - 1e-9) - g_gran + 1e-6 : ub - 1 + 1e-6; }

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ---------- LAPACK ---------- */
extern void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
                   const int *lda, double *w, double *work, const int *lwork, int *info);

/* ---------- RNG ---------- */
typedef struct { unsigned long long s; } rng_t;
static inline unsigned long long rng_next(rng_t *r) {
    r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17; return r->s;
}
static inline double rng_unif(rng_t *r) { return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }

/* ---------- reading ---------- */
static void read_qaplib(const char *fn) {
    FILE *f = fopen(fn, "r");
    if (!f) { perror(fn); exit(1); }
    if (fscanf(f, "%d", &N) != 1 || N > NMAX) { fprintf(stderr, "bad n\n"); exit(1); }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (fscanf(f, "%lf", &A[i][j]) != 1) exit(2);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (fscanf(f, "%lf", &B[i][j]) != 1) exit(2);
    fclose(f);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
        if (A[i][j] != A[j][i] || B[i][j] != B[j][i]) { fprintf(stderr, "matrices must be symmetric\n"); exit(1); }
}

static long lgcd(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a; }

static double qap_cost(const int *p) {
    double c = 0;
    for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) c += A[i][k] * B[p[i]][p[k]];
    return c;
}

/* ---------- LAP: shortest augmenting path (Hungarian, potentials) ----------
 * min sum c[i][p(i)], returns optimum, row duals u, column duals v with
 * c[i][j] - u[i] - v[j] >= 0 and sum u + sum v = optimum. */
static double lap(int n, const double c[NMAX][NMAX], int *rowsol, double *uo, double *vo) {
    double u[NMAX + 1], v[NMAX + 1], minv[NMAX + 1];
    int p[NMAX + 1], way[NMAX + 1];
    char used[NMAX + 1];
    for (int j = 0; j <= n; j++) { u[j] = 0; v[j] = 0; p[j] = 0; way[j] = 0; }
    for (int i = 1; i <= n; i++) {
        p[0] = i; int j0 = 0;
        for (int j = 0; j <= n; j++) { minv[j] = INFINITY; used[j] = 0; }
        do {
            used[j0] = 1; int i0 = p[j0], j1 = 0; double delta = INFINITY;
            for (int j = 1; j <= n; j++) if (!used[j]) {
                double cur = c[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    double tot = 0;
    for (int j = 1; j <= n; j++) { rowsol[p[j] - 1] = j - 1; }
    for (int i = 0; i < n; i++) { uo[i] = u[i + 1]; tot += u[i + 1]; }
    for (int j = 0; j < n; j++) { vo[j] = v[j + 1]; tot += v[j + 1]; }
    return tot;
}


/* ---------- LAP: Jonker-Volgenant (1987), dense, floating point ----------
 * Column reduction, reduction transfer, two augmenting row reduction passes,
 * then shortest augmenting paths. With warm != 0 the column prices v given
 * on input are reused (previous Frank-Wolfe iteration) and the column
 * reduction phase is replaced by a greedy assignment to row minima.
 * On output u[i] = min_j (c[i][j] - v[j]), so (u, v) is always dual feasible
 * and sum u + sum v is a valid lower bound, equal to the optimum. */
static int opt_lap = 4;   /* 0 Hungarian, 1 JV cold, 2 JV warm, 3 JV warm with JV start, 4 shortest paths (default), 5 = 4 with Hungarian start */

/* h[j] = c[j] - v[j] for j < dim, h[j] = +inf for dim <= j < NMAX; returns
 * min_j h[j].  Exact: same subtraction and comparisons as the scalar loop. */
#ifdef __AVX512F__
#include <immintrin.h>
static inline double row_h_min(const double *c, const double *v, int dim, double *h) {
    __m512d mn = _mm512_set1_pd(INFINITY), inf = mn;
    for (int b = 0; b < dim; b += 8) {
        int r = dim - b; __mmask8 k = r >= 8 ? 0xFF : (__mmask8)((1u << r) - 1);
        __m512d x = _mm512_mask_sub_pd(inf, k, _mm512_loadu_pd(c + b), _mm512_loadu_pd(v + b));
        _mm512_storeu_pd(h + b, x); mn = _mm512_min_pd(x, mn);
    }
    return _mm512_reduce_min_pd(mn);
}
/* first j with h[j] == x (x is known to be present) */
static inline int first_eq(const double *h, double x) {
    __m512d X = _mm512_set1_pd(x);
    for (int b = 0; b < NMAX; b += 8) {
        __mmask8 k = _mm512_cmp_pd_mask(_mm512_loadu_pd(h + b), X, _CMP_EQ_OQ);
        if (k) return b + __builtin_ctz(k);
    }
    return 0;
}
static inline double min_except(const double *h, int jx, int dim) {
    double save = ((double *)h)[jx]; ((double *)h)[jx] = INFINITY;
    __m512d mn = _mm512_loadu_pd(h);
    for (int b = 8; b < dim; b += 8) mn = _mm512_min_pd(_mm512_loadu_pd(h + b), mn);
    ((double *)h)[jx] = save;
    return _mm512_reduce_min_pd(mn);
}
#else
static inline double row_h_min(const double *c, const double *v, int dim, double *h) {
    double mn = INFINITY;
    for (int j = 0; j < NMAX; j++) { h[j] = j < dim ? c[j] - v[j] : INFINITY; if (h[j] < mn) mn = h[j]; }
    return mn;
}
static inline int first_eq(const double *h, double x) { int j = 0; while (h[j] != x) j++; return j; }
static inline double min_except(const double *h, int jx, int dim) {
    double mn = INFINITY; for (int j = 0; j < dim; j++) if (j != jx && h[j] < mn) mn = h[j]; return mn;
}
#endif

static int opt_simdlap = 0, opt_spkeep = 0, opt_spu = 1, opt_spfuse = 0;   /* spfuse: argmin inside the relaxation pass, measured slower (nug17 6.3 s -> 9.2 s) */   /* SIMD rewrite measured slower on nug16a (9.0 s vs 8.6 s) */
static double lap_jv(int dim, const double c[NMAX][NMAX], int *rowsol, double *u, double *v, int warm) {
    int colsol[NMAX], freer[NMAX], collist[NMAX], matches[NMAX], pred[NMAX];
    double d[NMAX];
    int numfree = 0;
    for (int j = 0; j < dim; j++) colsol[j] = -1;
    if (!warm) {
        for (int i = 0; i < dim; i++) matches[i] = 0;
        for (int j = dim - 1; j >= 0; j--) {
            double mn = c[0][j]; int imin = 0;
            for (int i = 1; i < dim; i++) if (c[i][j] < mn) { mn = c[i][j]; imin = i; }
            v[j] = mn;
            if (++matches[imin] == 1) { rowsol[imin] = j; colsol[j] = imin; }
            else colsol[j] = -1;
        }
        for (int i = 0; i < dim; i++) {
            if (matches[i] == 0) freer[numfree++] = i;
            else if (matches[i] == 1) {
                int j1 = rowsol[i]; double mn = INFINITY;
                for (int j = 0; j < dim; j++) if (j != j1 && c[i][j] - v[j] < mn) mn = c[i][j] - v[j];
                v[j1] -= mn;
            } else {
                /* several columns chose row i: keep the one in rowsol, free the others */
            }
        }
        /* make colsol consistent: columns whose row was assigned another column are free */
        for (int j = 0; j < dim; j++) if (colsol[j] >= 0 && rowsol[colsol[j]] != j) colsol[j] = -1;
    } else {
        for (int i = 0; i < dim; i++) {
            double mn = INFINITY; int jm = 0;
            if (opt_simdlap) { double h[NMAX]; mn = row_h_min(c[i], v, dim, h); jm = first_eq(h, mn); }
            else for (int j = 0; j < dim; j++) { double h = c[i][j] - v[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
            if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[numfree++] = i;
        }
    }
    /* augmenting row reduction */
    for (int loop = 0; loop < 2; loop++) {
        int k = 0, prv = numfree; numfree = 0;
        int guard = 0;
        while (k < prv) {
            int i = freer[k++];
            double umin, usub; int j1 = 0, j2 = 0;
            if (opt_simdlap) {   /* j1 = first argmin, j2 = first argmin over j != j1: same as the scalar scan */
                double h[NMAX]; umin = row_h_min(c[i], v, dim, h); j1 = first_eq(h, umin);
                usub = min_except(h, j1, dim); h[j1] = NAN; j2 = first_eq(h, usub);
            } else {
            umin = c[i][0] - v[0]; usub = INFINITY;
            const double *ci = c[i];
            for (int j = 1; j < dim; j++) {   /* branch-free form of the classic scan, same result */
                double h = ci[j] - v[j];
                int lt1 = h < umin, lt2 = h < usub;
                double nsub = lt2 ? h : usub; int nj2 = lt2 ? j : j2;
                usub = lt1 ? umin : nsub; j2 = lt1 ? j1 : nj2;
                umin = lt1 ? h : umin; j1 = lt1 ? j : j1;
            }
            }
            int i0 = colsol[j1];
            if (umin < usub - 1e-12) v[j1] -= (usub - umin);
            else if (i0 > -1) { j1 = j2; i0 = colsol[j2]; }
            rowsol[i] = j1; colsol[j1] = i;
            if (i0 > -1) {
                if (umin < usub - 1e-12 && ++guard < 10 * dim * dim) freer[--k] = i0;
                else freer[numfree++] = i0;
            }
        }
    }
    /* augmentation */
    for (int f = 0; f < numfree; f++) {
        int freerow = freer[f], low = 0, up = 0, last = 0, endofpath = -1, found = 0;
        double mn = 0;
        for (int j = 0; j < dim; j++) { d[j] = c[freerow][j] - v[j]; pred[j] = freerow; collist[j] = j; }
        do {
            if (up == low) {
                last = low - 1;
                mn = d[collist[up++]];
                for (int k = up; k < dim; k++) {
                    int j = collist[k]; double h = d[j];
                    if (h <= mn) { if (h < mn) { up = low; mn = h; } collist[k] = collist[up]; collist[up++] = j; }
                }
                for (int k = low; k < up; k++) if (colsol[collist[k]] < 0) { endofpath = collist[k]; found = 1; break; }
            }
            if (!found) {
                int j1 = collist[low++], i = colsol[j1];
                double h = c[i][j1] - v[j1] - mn;
                for (int k = up; k < dim; k++) {
                    int j = collist[k]; double v2 = c[i][j] - v[j] - h;
                    if (v2 < d[j]) {
                        pred[j] = i;
                        if (v2 == mn) {
                            if (colsol[j] < 0) { endofpath = j; found = 1; break; }
                            collist[k] = collist[up]; collist[up++] = j;
                        }
                        d[j] = v2;
                    }
                }
            }
        } while (!found);
        for (int k = 0; k <= last; k++) { int j1 = collist[k]; v[j1] += d[j1] - mn; }
        int i;
        do { i = pred[endofpath]; colsol[endofpath] = i; int j1 = endofpath; endofpath = rowsol[i]; rowsol[i] = j1; } while (i != freerow);
    }
    double tot = 0;
    for (int i = 0; i < dim; i++) {
        double mn = INFINITY;
        if (opt_simdlap) { double h[NMAX]; mn = row_h_min(c[i], v, dim, h); }
        else for (int j = 0; j < dim; j++) { double h = c[i][j] - v[j]; if (h < mn) mn = h; }
        u[i] = mn; tot += mn;
    }
    for (int j = 0; j < dim; j++) tot += v[j];
    return tot;
}


/* ---------- LAP: shortest augmenting paths only, written for SIMD ----------
 * Warm start from column prices v: each row takes its cheapest column if free.
 * Every remaining free row is then assigned by one Dijkstra search on reduced
 * costs (Jonker-Volgenant augmentation step); the inner loops run over all
 * columns with masks instead of branches, so the compiler vectorizes them.
 * Same optimum as lap_jv; ties may be broken differently. */
static double lap_sp(int dim, const double c[NMAX][NMAX], int *rowsol, double *u, double *v, int keep) {
    int colsol[NMAX], freer[NMAX], pred[NMAX], scan[NMAX], nfree = 0;
    double d[NMAX] __attribute__((aligned(64)));
    double done[NMAX] __attribute__((aligned(64)));   /* 0 = open, +inf = closed */
    for (int j = 0; j < dim; j++) colsol[j] = -1;
    if (keep) {   /* rowsol holds the previous assignment: keep each row whose column is still its cheapest */
        for (int i = 0; i < dim; i++) {
            double mn = INFINITY; const double *ci = c[i];
            for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; mn = h < mn ? h : mn; }
            int j0 = rowsol[i];
            if (ci[j0] - v[j0] <= mn) colsol[j0] = i; else freer[nfree++] = i;
        }
    } else for (int i = 0; i < dim; i++) {
        double mn = INFINITY; int jm = 0; const double *ci = c[i];
        for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
        if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[nfree++] = i;
    }
    for (int f = 0; f < nfree; f++) {
        int fr = freer[f], ns = 0, jend = -1;
        const double *cf = c[fr];
        double mn;
        if (opt_spfuse) {   /* the search for the closest open column is done in the relaxation pass */
            double nmn = INFINITY; int njs = 0;
            for (int j = 0; j < dim; j++) { double x = cf[j] - v[j]; d[j] = x; pred[j] = fr; done[j] = 0;
                int lt = x < nmn; nmn = lt ? x : nmn; njs = lt ? j : njs; }
            for (;;) {
                mn = nmn; int js = njs;
                done[js] = INFINITY; scan[ns++] = js;
                if (colsol[js] < 0) { jend = js; break; }
                int i = colsol[js]; const double *ci = c[i];
                double h = ci[js] - v[js] - mn;
                nmn = INFINITY; njs = 0;
                for (int j = 0; j < dim; j++) {   /* relax the edges of row i, then keep the closest open column */
                    double v2 = ci[j] - v[j] - h, dj = d[j], dn = done[j];
                    int up = (v2 < dj) & (dn == 0);
                    dj = up ? v2 : dj; d[j] = dj; pred[j] = up ? i : pred[j];
                    double x = dj + dn; int lt = x < nmn; nmn = lt ? x : nmn; njs = lt ? j : njs;
                }
            }
        } else {
        for (int j = 0; j < dim; j++) { d[j] = cf[j] - v[j]; pred[j] = fr; done[j] = 0; }
        for (;;) {
            mn = INFINITY; int js = 0;   /* closest open column */
            for (int j = 0; j < dim; j++) { double x = d[j] + done[j]; int lt = x < mn; mn = lt ? x : mn; js = lt ? j : js; }
            done[js] = INFINITY; scan[ns++] = js;
            if (colsol[js] < 0) { jend = js; break; }
            int i = colsol[js]; const double *ci = c[i];
            double h = ci[js] - v[js] - mn;
            for (int j = 0; j < dim; j++) {   /* relax the edges of row i */
                double v2 = ci[j] - v[j] - h;
                int up = (v2 < d[j]) & (done[j] == 0);
                d[j] = up ? v2 : d[j]; pred[j] = up ? i : pred[j];
            }
        }
        }
        for (int t = 0; t < ns; t++) { int j = scan[t]; v[j] += d[j] - mn; }
        int i, j = jend;
        do { i = pred[j]; colsol[j] = i; int k = rowsol[i]; rowsol[i] = j; j = k; } while (i != fr);
    }
    double tot = 0;
    if (opt_spu) for (int i = 0; i < dim; i++) {   /* complementary slackness: zero reduced cost on the assignment */
        int j = rowsol[i]; u[i] = c[i][j] - v[j]; tot += u[i];
    } else for (int i = 0; i < dim; i++) {
        double mn = INFINITY; const double *ci = c[i];
        for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; mn = h < mn ? h : mn; }
        u[i] = mn; tot += mn;
    }
    for (int j = 0; j < dim; j++) tot += v[j];
    return tot;
}

/* lap_sp with the column loops padded to a multiple of 4: columns dim..dim4-1 get
 * cost +inf and dual 0, so they are never chosen, and every loop has a trip count
 * the compiler can split into whole vectors with no scalar remainder. */
static double lap_spw(int dim, double c[NMAX][NMAX], int *rowsol, double *u, double *v) {
    int colsol[NMAX], freer[NMAX], pred[NMAX] __attribute__((aligned(64))), scan[NMAX], nfree = 0;
    double d[NMAX] __attribute__((aligned(64)));
    double done[NMAX] __attribute__((aligned(64)));
    int dim4 = (dim + 3) & ~3;
    if (dim4 & 3) __builtin_unreachable();
    for (int i = 0; i < dim; i++) for (int j = dim; j < dim4; j++) c[i][j] = INFINITY;
    for (int j = dim; j < dim4; j++) v[j] = 0;
    for (int j = 0; j < dim; j++) colsol[j] = -1;
    for (int i = 0; i < dim; i++) {
        double mn = INFINITY; int jm = 0; const double *ci = c[i];
        for (int j = 0; j < dim4; j++) { double h = ci[j] - v[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
        if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[nfree++] = i;
    }
    for (int f = 0; f < nfree; f++) {
        int fr = freer[f], ns = 0, jend = -1;
        const double *cf = c[fr];
        for (int j = 0; j < dim4; j++) { d[j] = cf[j] - v[j]; pred[j] = fr; done[j] = 0; }
        double mn;
        for (;;) {
            mn = INFINITY; int js = 0;
            for (int j = 0; j < dim4; j++) { double x = d[j] + done[j]; int lt = x < mn; mn = lt ? x : mn; js = lt ? j : js; }
            done[js] = INFINITY; scan[ns++] = js;
            if (colsol[js] < 0) { jend = js; break; }
            int i = colsol[js]; const double *ci = c[i];
            double h = ci[js] - v[js] - mn;
            for (int j = 0; j < dim4; j++) {
                double v2 = ci[j] - v[j] - h;
                int up = (v2 < d[j]) & (done[j] == 0);
                d[j] = up ? v2 : d[j]; pred[j] = up ? i : pred[j];
            }
        }
        for (int t = 0; t < ns; t++) { int j = scan[t]; v[j] += d[j] - mn; }
        int i, j = jend;
        do { i = pred[j]; colsol[j] = i; int k = rowsol[i]; rowsol[i] = j; j = k; } while (i != fr);
    }
    double tot = 0;
    for (int i = 0; i < dim; i++) { int j = rowsol[i]; u[i] = c[i][j] - v[j]; tot += u[i]; }
    for (int j = 0; j < dim; j++) tot += v[j];
    return tot;
}

/* Approximate LAP for the Frank-Wolfe direction: each row takes its cheapest
 * reduced column if still free, the remaining rows take the cheapest free column.
 * u_i = min_j (c_ij - v_j) keeps (u, v) dual feasible, so sum u + sum v is a valid
 * lower bound on the LAP value; *pc returns the cost of the permutation found. */
static int opt_alap = 0;   /* 0 exact LAP at every iteration; k: exact one iteration in k */
static double lap_greedy(int dim, const double c[NMAX][NMAX], int *rowsol, double *u, const double *v, double *pc) {
    int colsol[NMAX], freer[NMAX], nfree = 0; double tot = 0, cost = 0;
    for (int j = 0; j < dim; j++) colsol[j] = -1;
    for (int i = 0; i < dim; i++) {
        double mn = INFINITY; int jm = 0; const double *ci = c[i];
        for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
        u[i] = mn; tot += mn;
        if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[nfree++] = i;
    }
    for (int f = 0; f < nfree; f++) {
        int i = freer[f], jb = -1; double mn = INFINITY; const double *ci = c[i];
        for (int j = 0; j < dim; j++) if (colsol[j] < 0) { double h = ci[j] - v[j]; if (h < mn) { mn = h; jb = j; } }
        colsol[jb] = i; rowsol[i] = jb;
    }
    for (int j = 0; j < dim; j++) tot += v[j];
    for (int i = 0; i < dim; i++) cost += c[i][rowsol[i]];
    *pc = cost;
    return tot;
}

/* lap_sp in single precision: twice as many values per vector instruction.
 * Only the assignment and the column duals v come from the float search; the
 * row duals are then recomputed in double as u_i = min_j (c_ij - v_j), so
 * (u, v) is dual feasible in double and sum u + sum v stays a valid lower bound. */
static double lap_spf(int dim, const double c[NMAX][NMAX], int *rowsol, double *u, double *v) {
    float cf[NMAX][NMAX] __attribute__((aligned(64)));
    float vf[NMAX] __attribute__((aligned(64)));
    int colsol[NMAX], freer[NMAX], pred[NMAX], scan[NMAX], nfree = 0;
    float d[NMAX] __attribute__((aligned(64)));
    float done[NMAX] __attribute__((aligned(64)));
    for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) cf[i][j] = (float)c[i][j];
    for (int j = 0; j < dim; j++) { vf[j] = (float)v[j]; colsol[j] = -1; }
    for (int i = 0; i < dim; i++) {
        float mn = INFINITY; int jm = 0; const float *ci = cf[i];
        for (int j = 0; j < dim; j++) { float h = ci[j] - vf[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
        if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[nfree++] = i;
    }
    for (int f = 0; f < nfree; f++) {
        int fr = freer[f], ns = 0, jend = -1;
        const float *c0 = cf[fr];
        for (int j = 0; j < dim; j++) { d[j] = c0[j] - vf[j]; pred[j] = fr; done[j] = 0; }
        float mn;
        for (;;) {
            mn = INFINITY; int js = 0;
            for (int j = 0; j < dim; j++) { float x = d[j] + done[j]; int lt = x < mn; mn = lt ? x : mn; js = lt ? j : js; }
            done[js] = INFINITY; scan[ns++] = js;
            if (colsol[js] < 0) { jend = js; break; }
            int i = colsol[js]; const float *ci = cf[i];
            float h = ci[js] - vf[js] - mn;
            for (int j = 0; j < dim; j++) {
                float v2 = ci[j] - vf[j] - h;
                int up = (v2 < d[j]) & (done[j] == 0);
                d[j] = up ? v2 : d[j]; pred[j] = up ? i : pred[j];
            }
        }
        for (int t = 0; t < ns; t++) { int j = scan[t]; vf[j] += d[j] - mn; }
        int i, j = jend;
        do { i = pred[j]; colsol[j] = i; int k = rowsol[i]; rowsol[i] = j; j = k; } while (i != fr);
    }
    double tot = 0;
    for (int j = 0; j < dim; j++) { v[j] = vf[j]; tot += v[j]; }
    for (int i = 0; i < dim; i++) {
        double mn = INFINITY; const double *ci = c[i];
        for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; mn = h < mn ? h : mn; }
        u[i] = mn; tot += mn;
    }
    return tot;
}

/* ---------- V matrices: orthonormal basis of e-perp, per size ---------- */
static double Vb[NMAX + 1][NMAX][NMAX]; /* Vb[m] is m x (m-1) */
static void init_V(void) {
    for (int m = 2; m <= NMAX; m++) {
        /* Householder H = I - 2 w w^T / w^T w, w = e/sqrt(m) - e_m ; H e/sqrt(m) = e_m */
        double w[NMAX], ww = 0, s = 1.0 / sqrt((double)m);
        for (int i = 0; i < m; i++) w[i] = s; w[m - 1] -= 1.0;
        for (int i = 0; i < m; i++) ww += w[i] * w[i];
        for (int i = 0; i < m; i++) for (int j = 0; j < m - 1; j++)
            Vb[m][i][j] = (i == j ? 1.0 : 0.0) - 2.0 * w[i] * w[j] / ww;
    }
}

/* ---------- workspace for one bound computation ---------- */
/* Spectral data of V^T M V depends only on the set of free rows (A side) or
 * free columns (B side): siblings and look-ahead candidates share it. */
#define ECACHE 512
typedef struct { unsigned mask; int m; double lam[NMAX]; double VW[NMAX][NMAX]; } eig_t;
typedef struct {
    eig_t ea[ECACHE], eb[ECACHE];
    long hits, miss;
    double Ar[NMAX][NMAX], Br[NMAX][NMAX], Cr[NMAX][NMAX];
    double Sp[NMAX][NMAX], Tp[NMAX][NMAX];
    double X[NMAX][NMAX], MX[NMAX][NMAX], G[NMAX][NMAX], MP[NMAX][NMAX], U[NMAX][NMAX];
    double APB[NMAX][NMAX];
    int pprev[NMAX], apb_ok, apb_age;
    unsigned maskF, maskL;              /* free facilities / locations of the node being evaluated */   /* incremental Ar P Br: permutation behind APB */
    double trST;
    const eig_t *eA, *eB;               /* spectral data of the node, for the S, T update */
    double sd[NMAX], td[NMAX];          /* current dual vectors: Sp = VW_A diag(sd) VW_A^T, Tp likewise */
    long nlap, nfw, nupd;
    int eperm[NMAX], eperm_best[NMAX]; unsigned eused;   /* -enum */
    int x0_ok; double X0[NMAX][NMAX];   /* Frank-Wolfe starting point inherited from the parent */
    struct pfw_s *pa;                   /* active set of the pairwise Frank-Wolfe */
    int nfb; unsigned char fbl[NMAX][NMAX];   /* forbidden pairs of the node, local indices */
    int Up[NMAX], up_ok;                /* assignment of the LAP behind U (row a -> column Up[a]) */
    double Wmax[NMAX][NMAX];            /* -umax: max over the FW iterations of z_k + U_k */
    double teig;                        /* -timeig: seconds in the spectral decompositions */
    double flast;                       /* f(X) + K at the end of the last Frank-Wolfe run */
    int inh_ok, inh_live; double Si[NMAX][NMAX], Ti[NMAX][NMAX], trSTi;   /* -inh: S, T inherited from the parent (principal submatrices) */
} work_t;

/* Pairwise Frank-Wolfe (Lacoste-Julien and Jaggi): X is kept as a convex
 * combination of atoms, the starting point (atom marked perm[.][0] = -1) and the
 * permutations found by the LAP. Each step moves weight from the atom with the
 * largest <G, A> (away atom) to the LAP permutation. M(A) is stored per atom. */
#define AMAX 320
typedef struct pfw_s { int n; int perm[AMAX][NMAX]; double lam[AMAX], cr[AMAX]; double A0[NMAX][NMAX]; double MA[AMAX][NMAX][NMAX]; } pfw_t;
static int opt_pfw = 0;

/* Starting point of a child (a -> b) from the parent's Frank-Wolfe point X (m x m):
 * drop row a and column b, then add the rank-one term x_col x_row / s, with
 * x_col = X[.][b], x_row = X[a][.], s = 1 - X[a][b]. Row p then sums to
 * (1 - X[p][b]) + X[p][b] (1 - X[a][b]) / s = 1, and likewise for columns: the
 * point is doubly stochastic, so the Frank-Wolfe bounds stay valid. */
static int opt_lasize = 0, opt_lanb = 5, opt_lanfw = 18;   /* rule 3 below the strategy's levels while m >= opt_lasize */
static int opt_lacap = 1;   /* look-ahead: child bounds capped at the threshold, hopeless candidates abandoned */
static int opt_warm = 2;   /* 1: look-ahead children, 2: children in the serial search, start from the parent's point; 4: S, T fitted there */
/* -inh k: a child (a -> b) starts Frank-Wolfe with S' = S[-a,-a], T' = T[-b,-b] taken from the parent, no spectral
 * decomposition and no LAP of the eigenvalues: the parent's function restricted to the child's face is the child's
 * function with these S', T', convex on the child's affine hull, exact on its permutations. A child that reaches the
 * S, T update (iteration opt_upd) switches to its own S, T. Bit 1: children of the search, bit 2: look-ahead children. */
static int opt_inh = 0; static long g_inh_used = 0, g_inh_switch = 0;
static void print_inh(void) { if (opt_inh) fprintf(stderr, "inh: %ld nodes started on inherited S, T; %ld switched to their own\n", g_inh_used, g_inh_switch); }
static void inherit_st(work_t *dst, const double S[NMAX][NMAX], const double T[NMAX][NMAX], double trST, int m, int a, int b) {
    int r = 0;
    for (int p = 0; p < m; p++) { if (p == a) continue; int c = 0;
        for (int q = 0; q < m; q++) { if (q == a) continue; dst->Si[r][c++] = S[p][q]; } r++; }
    r = 0;
    for (int p = 0; p < m; p++) { if (p == b) continue; int c = 0;
        for (int q = 0; q < m; q++) { if (q == b) continue; dst->Ti[r][c++] = T[p][q]; } r++; }
    dst->trSTi = trST - S[a][a] - T[b][b]; dst->inh_ok = 1;
}
static void inherit_x(work_t *dst, const double X[NMAX][NMAX], int m, int a, int b) {
    double sden = 1 - X[a][b];
    dst->x0_ok = 0;
    if (sden < 1e-6) return;
    int r = 0;
    for (int p = 0; p < m; p++) { if (p == a) continue;
        int c = 0; double xp = X[p][b] / sden;
        for (int q = 0; q < m; q++) { if (q == b) continue; dst->X0[r][c++] = X[p][q] + xp * X[a][q]; }
        r++; }
    dst->x0_ok = 1;
}

/* c = a b for m <= W: rows of NMAX doubles, the j loop has a constant trip
 * count W so it compiles to full-width FMAs; columns j >= m of c hold junk
 * that is never read (column j of c only depends on column j of b). */
#define MM_W(W) static void matmul_w##W(int m, const double a[NMAX][NMAX], const double b[NMAX][NMAX], double c[NMAX][NMAX]) { \
    for (int i = 0; i < m; i++) { double acc[W] = {0}; \
        for (int k = 0; k < m; k++) { double aik = a[i][k]; \
            for (int j = 0; j < W; j++) acc[j] += aik * b[k][j]; } \
        for (int j = 0; j < W; j++) c[i][j] = acc[j]; } }
MM_W(8) MM_W(16) MM_W(24) MM_W(32)
static int opt_fastmm = 0;
static void matmul_fast(int m, const double a[NMAX][NMAX], const double b[NMAX][NMAX], double c[NMAX][NMAX]) {
    if (m <= 8) matmul_w8(m, a, b, c); else if (m <= 16) matmul_w16(m, a, b, c);
    else if (m <= 24) matmul_w24(m, a, b, c); else matmul_w32(m, a, b, c);
}
static void matmul(int m, const double a[NMAX][NMAX], const double b[NMAX][NMAX], double c[NMAX][NMAX]) {
    if (opt_fastmm) { matmul_fast(m, a, b, c); return; }
    for (int i = 0; i < m; i++) {
        double row[NMAX] = {0};
        for (int k = 0; k < m; k++) { double aik = a[i][k]; if (aik == 0) continue;
            for (int j = 0; j < m; j++) row[j] += aik * b[k][j]; }
        for (int j = 0; j < m; j++) c[i][j] = row[j];
    }
}

/* eigen-decomposition of V^T M V, returns eigvals ascending and eigvectors W (columns) */
static void proj_eig(int m, const double M[NMAX][NMAX], double *lam, double W[NMAX][NMAX]) {
    int k = m - 1;
    double T1[NMAX][NMAX], H[NMAX * NMAX];
    /* T1 = M V  (m x k) */
    for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
        double s = 0; for (int l = 0; l < m; l++) s += M[i][l] * Vb[m][l][j]; T1[i][j] = s; }
    /* H = V^T T1, column major for LAPACK (symmetric anyway) */
    for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) {
        double s = 0; for (int l = 0; l < m; l++) s += Vb[m][l][i] * T1[l][j]; H[i + j * k] = s; }
    int lwork = 3 * NMAX * NMAX, info; double wk[3 * NMAX * NMAX];
    dsyev_("V", "U", &k, H, &k, lam, wk, &lwork, &info);
    if (info) { fprintf(stderr, "dsyev failed\n"); exit(3); }
    for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) W[i][j] = H[i + j * k];
}

/* S' = V W diag(s) W^T V^T */
static void build_proj(int m, const double W[NMAX][NMAX], const double *s, double out[NMAX][NMAX]) {
    int k = m - 1; double VW[NMAX][NMAX];
    for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
        double t = 0; for (int l = 0; l < k; l++) t += Vb[m][i][l] * W[l][j]; VW[i][j] = t; }
    for (int i = 0; i < m; i++) for (int j = 0; j <= i; j++) {
        double t = 0; for (int l = 0; l < k; l++) t += VW[i][l] * s[l] * VW[j][l];
        out[i][j] = out[j][i] = t; }
}

static int opt_ecache = 1, opt_memo = 1;

static const eig_t *get_eig(work_t *w, eig_t *tab, unsigned mask, int m, const double M[NMAX][NMAX]) {
    unsigned h = (mask * 2654435761u) >> 23;   /* 9 bits */
    eig_t *e = &tab[h & (ECACHE - 1)];
    if (opt_ecache && e->m == m && e->mask == mask) { w->hits++; g_ehit++; return e; }
    w->miss++;
    double W[NMAX][NMAX]; int k = m - 1;
    unsigned long long pce = g_prof ? __rdtsc() : 0;
    double t0 = g_timeig ? now() : 0;
    proj_eig(m, M, e->lam, W);
    if (g_timeig) { w->teig += now() - t0; g_teig_tot += now() - t0; g_emiss++; }
    for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
        double t = 0; for (int l = 0; l < k; l++) t += Vb[m][i][l] * W[l][j]; e->VW[i][j] = t; }
    e->mask = mask; e->m = m;
    PROF_T(tl_noext, 1, pce);
    return e;
}
static void build_from_vw(int m, const double VW[NMAX][NMAX], const double *s, double out[NMAX][NMAX]) {
    int k = m - 1;
    for (int i = 0; i < m; i++) {
        double r[NMAX]; for (int l = 0; l < k; l++) r[l] = VW[i][l] * s[l];
        for (int j = 0; j <= i; j++) { double t = 0; for (int l = 0; l < k; l++) t += r[l] * VW[j][l]; out[i][j] = out[j][i] = t; }
    }
}
/* LAP on the costs lambda_i mu_j with lambda ascending and mu ascending: these costs
 * are Monge once the columns are reversed, so the assignment i -> k-1-i is optimal
 * (rearrangement inequality) and the north-west staircase gives dual s, t with
 * s_i + t_j <= lambda_i mu_j, tight on the assignment. Returns NAN if the duals fail
 * the feasibility check (rounding), the caller then solves the LAP. */
static int opt_monge = 0;   /* closed form, measured worse: other optimal duals, nug17 57 085 -> 64 095 nodes */
static double monge_lap(int k, const double *la, const double *mu, double *s, double *t) {
    int i = 0, j = 0; double ra = 1, rb = 1;
    s[0] = 0; t[k - 1] = la[0] * mu[k - 1];
    while (i < k - 1 || j < k - 1) {
        int down = (j == k - 1) || (i < k - 1 && ra < rb);
        if (down) { rb -= ra; i++; ra = 1; s[i] = la[i] * mu[k - 1 - j] - t[k - 1 - j]; }
        else      { ra -= rb; j++; rb = 1; t[k - 1 - j] = la[i] * mu[k - 1 - j] - s[i]; }
    }
    double scale = 1, viol = 0, tot = 0;
    for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) {
        double c = la[a] * mu[b]; if (fabs(c) > scale) scale = fabs(c);
        double x = s[a] + t[b] - c; if (x > viol) viol = x; }
    if (viol > 1e-9 * scale) return NAN;
    for (int l = 0; l < k; l++) tot += s[l] + t[l];
    return tot;
}
/* setup of the QPB quadratic: computes Sp, Tp, trST for current Ar, Br */
static double setup_qpb(work_t *w, int m) {
    int k = m - 1;
    double L[NMAX][NMAX], s[NMAX], t[NMAX];
    int sol[NMAX];
    const eig_t *ea = get_eig(w, w->ea, w->maskF, m, w->Ar);
    const eig_t *eb = get_eig(w, w->eb, w->maskL, m, w->Br);
    unsigned long long pcs = g_prof ? __rdtsc() : 0;
    for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) L[i][j] = ea->lam[i] * eb->lam[j];
    double pb = opt_monge ? monge_lap(k, ea->lam, eb->lam, s, t) : NAN;
    if (isnan(pb)) pb = lap(k, L, sol, s, t);   /* <lambda, mu>_-  and its dual s, t */
    build_from_vw(m, ea->VW, s, w->Sp);
    build_from_vw(m, eb->VW, t, w->Tp);
    w->trST = pb;
    w->eA = ea; w->eB = eb;
    memcpy(w->sd, s, sizeof(double) * k); memcpy(w->td, t, sizeof(double) * k);
    PROF_T(tl_noext, 2, pcs);
    return pb;
}

/* Update of S and T at the current point X (UPDATE parameter of Brixius and
 * Anstreicher). For any (s, t) with s_i + t_j <= lambda_i mu_j, Q is convex on
 * the affine hull and f(P) = QAP cost on every permutation P, so the bound stays
 * valid. At fixed X, f = <X, Ar X Br> + <Cr, X> + sum_l s_l alpha_l + sum_l t_l beta_l
 * with alpha_l = 1 - |X^T VW_A[:,l]|^2, beta_l = 1 - |X VW_B[:,l]|^2, which have
 * equal sums. The best (s, t) is then the dual of a transportation problem with
 * costs lambda_i mu_j and margins alpha, beta. These costs are Monge once lambda
 * is ascending and mu descending, so the north-west corner rule is optimal and
 * its staircase basis gives the duals. Returns 1 if S, T changed. */
static int opt_upd = 30, opt_updstop = 0;
static int update_st(work_t *w, int m) {
    int k = m - 1; const eig_t *ea = w->eA, *eb = w->eB;
    double al[NMAX], be[NMAX], s[NMAX], t[NMAX];
    for (int l = 0; l < k; l++) { al[l] = 1; be[l] = 1; }
    for (int j = 0; j < m; j++) {       /* (X^T VW_A)[j][l] */
        double r[NMAX] = {0};
        for (int i = 0; i < m; i++) { double x = w->X[i][j]; for (int l = 0; l < k; l++) r[l] += x * ea->VW[i][l]; }
        for (int l = 0; l < k; l++) al[l] -= r[l] * r[l];
    }
    for (int i = 0; i < m; i++) {       /* (X VW_B)[i][l] */
        double r[NMAX] = {0};
        for (int j = 0; j < m; j++) { double x = w->X[i][j]; for (int l = 0; l < k; l++) r[l] += x * eb->VW[j][l]; }
        for (int l = 0; l < k; l++) be[l] -= r[l] * r[l];
    }
    /* north-west corner: rows lambda ascending (index i), columns mu descending (index k-1-j) */
    const double *la = ea->lam, *mu = eb->lam;
    int i = 0, j = 0; double ra = al[0], rb = be[k - 1];
    s[0] = 0; t[k - 1] = la[0] * mu[k - 1];
    while (i < k - 1 || j < k - 1) {
        int down = (j == k - 1) || (i < k - 1 && ra < rb);
        if (down) { rb -= ra; i++; ra = al[i]; s[i] = la[i] * mu[k - 1 - j] - t[k - 1 - j]; }
        else      { ra -= rb; j++; rb = be[k - 1 - j]; t[k - 1 - j] = la[i] * mu[k - 1 - j] - s[i]; }
    }
    double scale = 1, viol = 0, gain = 0;
    for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) {
        double c = la[a] * mu[b]; if (fabs(c) > scale) scale = fabs(c);
        double x = s[a] + t[b] - c; if (x > viol) viol = x; }
    if (viol > 1e-9 * scale) return 0;   /* not dual feasible: keep S, T */
    for (int l = 0; l < k; l++) gain += al[l] * (s[l] - w->sd[l]) + be[l] * (t[l] - w->td[l]);
    if (gain <= 1e-9 * scale) return 0;
    memcpy(w->sd, s, sizeof(double) * k); memcpy(w->td, t, sizeof(double) * k);
    build_from_vw(m, ea->VW, s, w->Sp);
    build_from_vw(m, eb->VW, t, w->Tp);
    double tr = 0; for (int l = 0; l < k; l++) tr += s[l] + t[l];
    w->trST = tr; w->nupd++;
    return 1;
}

/* M(X) = Ar X Br - Sp X - X Tp, for general X */
static void applyM(work_t *w, int m, const double X[NMAX][NMAX], double out[NMAX][NMAX]) {
    double T1[NMAX][NMAX];
    matmul(m, w->Ar, X, T1);
    matmul(m, T1, w->Br, out);
    matmul(m, w->Sp, X, T1);
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) out[i][j] -= T1[i][j];
    matmul(m, X, w->Tp, T1);
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) out[i][j] -= T1[i][j];
}

/* M(P) for a permutation p (row i -> column p[i]); also APB = Ar P Br */
static int opt_incr = 1, opt_fuse = 1;   /* incremental update of Ar P Br between Frank-Wolfe iterations */
static void applyMperm(work_t *w, int m, const int *p, double out[NMAX][NMAX], double *xmp) {
    /* (Ar P Br)[i][j] = sum_k Ar[i][k] Br[p[k]][j].  Successive LAP solutions
     * usually differ in a few positions k; then APB changes by the rank-one
     * terms Ar[:,k] (Br[p[k]] - Br[pprev[k]]), O(m^2) each instead of O(m^3). */
    int D[NMAX], nd = 0;
    const int mr = (m + 3) & ~3;   /* row width rounded for SIMD; Br is zero-padded up to mr */
    if (opt_incr && w->apb_ok && w->apb_age < 64) {
        for (int k = 0; k < m; k++) if (p[k] != w->pprev[k]) D[nd++] = k;
    } else nd = m + 1;
    if (nd < m) {   /* nd rank-one updates of m x mr cost less than one m x m x mr product */
        for (int t = 0; t < nd; t++) {
            int k = D[t]; const double *bn = w->Br[p[k]], *bo = w->Br[w->pprev[k]];
            double db[NMAX]; for (int j = 0; j < mr; j++) db[j] = bn[j] - bo[j];
            for (int i = 0; i < m; i++) { double a = w->Ar[i][k]; double *row = w->APB[i];
                for (int j = 0; j < mr; j++) row[j] += a * db[j]; }
        }
        w->apb_age++;
    } else {
        for (int i = 0; i < m; i++) {
            double acc[NMAX]; for (int j = 0; j < mr; j++) acc[j] = 0;
            for (int k = 0; k < m; k++) { double a = w->Ar[i][k]; const double *b = w->Br[p[k]];
                for (int j = 0; j < mr; j++) acc[j] += a * b[j]; }
            for (int j = 0; j < mr; j++) w->APB[i][j] = acc[j];
        }
        w->apb_ok = 1; w->apb_age = 0;
    }
    memcpy(w->pprev, p, sizeof(int) * m);
    int pinv[NMAX]; for (int k = 0; k < m; k++) pinv[p[k]] = k;
    if (xmp) {   /* also <X, M(P)>, for the line search */
        double acc = 0;
        for (int i = 0; i < m; i++) { const double *tp = w->Tp[p[i]], *sp = w->Sp[i], *ab = w->APB[i], *x = w->X[i]; double *o = out[i];
            for (int j = 0; j < m; j++) { double t = ab[j] - sp[pinv[j]] - tp[j]; o[j] = t; acc += x[j] * t; } }
        *xmp = acc;
    } else
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++)
        /* (Sp P)[i][j] = Sp[i][pinv[j]] ; (P Tp)[i][j] = Tp[p[i]][j] */
        out[i][j] = w->APB[i][j] - w->Sp[i][pinv[j]] - w->Tp[p[i]][j];
}

static inline double dotm(int m, const double a[NMAX][NMAX], const double b[NMAX][NMAX]) {
    double s = 0; for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) s += a[i][j] * b[i][j]; return s;
}

/* Frank-Wolfe on QPB. K = constant of the node. Returns best lower bound z
 * (constant included); w->U holds the reduced costs with
 * cost(X) >= z + <U, X> for every permutation X of the subproblem.
 * Stops when z > target (node fathomed) or after nfw iterations or stall.
 * If a permutation found has cost < *ub_found, it is reported in bestp. */
static double qpb_fw(work_t *w, int m, double K, double target, int nfw, int nfw2,
                     double *ub_found, int *bestp) {
    if (w->inh_ok) {
        memcpy(w->Sp, w->Si, sizeof(double) * NMAX * NMAX); memcpy(w->Tp, w->Ti, sizeof(double) * NMAX * NMAX);
        w->trST = w->trSTi; w->eA = w->eB = NULL; w->inh_ok = 0; w->inh_live = 1; g_inh_used++;
    } else { setup_qpb(w, m); w->inh_live = 0; }
    w->apb_ok = 0;
    double inv = 1.0 / m;
    if (w->x0_ok) {   /* inherited starting point */
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->X[i][j] = w->X0[i][j];
        if ((opt_warm & 4) && !w->inh_live) update_st(w, m);   /* S, T fitted to the inherited point before the first step */
        applyM(w, m, w->X, w->MX);
        w->x0_ok = 0;
    } else {
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->X[i][j] = inv;
    {   /* X0 = ee^T/m: Ar X0 Br = (Ar e)(e^T Br)/m, and Sp e = 0, e^T Tp = 0 */
        double ra[NMAX], cb[NMAX];
        for (int i = 0; i < m; i++) { double t = 0; for (int k = 0; k < m; k++) t += w->Ar[i][k]; ra[i] = t; }
        for (int j = 0; j < m; j++) { double t = 0; for (int k = 0; k < m; k++) t += w->Br[k][j]; cb[j] = t; }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->MX[i][j] = ra[i] * cb[j] * inv;
    }
    }
    double s2 = dotm(m, w->Cr, w->X);   /* <Cr, X>, kept up to date in O(m) when opt_fuse */
    double f = dotm(m, w->X, w->MX) + s2 + w->trST;
    pfw_t *PA = NULL;
    if (opt_pfw) {
        if (!w->pa) w->pa = calloc(1, sizeof(pfw_t));
        PA = w->pa; PA->n = 1; PA->perm[0][0] = -1; PA->lam[0] = 1; PA->cr[0] = s2;
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { PA->A0[i][j] = w->X[i][j]; PA->MA[0][i][j] = w->MX[i][j]; }
    }
    if (opt_fuse) for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->G[i][j] = 2 * w->MX[i][j] + w->Cr[i][j];
    double best = -INFINITY, u[NMAX], v[NMAX];
    int p[NMAX];
    double hist_best_at[8]; int it;
    int nfwc = nfw;
    int nfwstop = nfw2, fsk = tl_noext ? opt_fstopl : opt_fstopm; if (fsk > 0 && fsk < nfwstop) nfwstop = fsk;
    int lg = g_fwlog && (g_fwlog_n++ % g_fwlog_k) == 0, lgx = 4; double lgz[512], lgf[512];
    for (it = 0; it < nfwc; it++) {
        /* -ext k d: at the last allowed iteration, if f(X) is still above the target (the node may
         * still be fathomed) and the bound is within d of it, allow k more iterations, up to
         * opt_extn times; any iteration count gives a valid bound */
        if (opt_ext && !tl_noext && it == nfwc - 1 && nfwc < nfw + opt_ext * opt_extn && f + K >= target && target - best < opt_extd) nfwc += opt_ext;
        if (!opt_fuse) for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->G[i][j] = 2 * w->MX[i][j] + w->Cr[i][j];
        unsigned long long pcl = g_prof ? __rdtsc() : 0;
        double lapv, gp = NAN;   /* gp: <G, P> when the LAP is approximate */
        /* -fb: forbidden pairs get cost FBIG in the LAP only. For every permutation P of the
         * subtree, f(P) >= f(X) + <G, P - X> = f(X) - <G, X> + <G', P> >= f(X) - <G, X> + LAP(G'),
         * since G' = G off the forbidden pairs and P avoids them; X itself may be anywhere. */
        double gsave[NMAX * NMAX]; int ns = 0;
        if (w->nfb) for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) if (w->fbl[i][j]) { gsave[ns++] = w->G[i][j]; w->G[i][j] = FBIG; }
        if (opt_alap > 1 && it > 0 && it % opt_alap != 0 && opt_fuse) {
            double v0[NMAX]; memcpy(v0, v, sizeof(double) * m);
            lapv = lap_greedy(m, w->G, p, u, v, &gp);
            if (gp - (2 * (f - w->trST - s2) + s2) > -1e-9 * (1 + fabs(gp))) { gp = NAN; memcpy(v, v0, sizeof(double) * m); lapv = lap_sp(m, w->G, p, u, v, 0); }   /* not a descent direction: exact LAP */
        }
        else if (opt_lap == 0 || (opt_lap == 2 && it == 0)) lapv = lap(m, w->G, p, u, v);
        else if (opt_lap == 3 && it == 0) lapv = lap_jv(m, w->G, p, u, v, 0);   /* JV with its own column reduction */
        else if (opt_lap == 5 && it == 0) lapv = lap(m, w->G, p, u, v);
        else if (opt_lap == 4 || opt_lap == 5 || opt_lap == 6 || opt_lap == 7) {
            if (it == 0) for (int j = 0; j < m; j++) { double mn = INFINITY; for (int i = 0; i < m; i++) mn = w->G[i][j] < mn ? w->G[i][j] : mn; v[j] = mn; }
            lapv = opt_lap == 6 ? lap_spf(m, w->G, p, u, v) : opt_lap == 7 ? lap_spw(m, w->G, p, u, v) : lap_sp(m, w->G, p, u, v, it > 0 && opt_spkeep);
        }
        else {
            if (opt_lap == 1) for (int j = 0; j < m; j++) { double mn = INFINITY; for (int i = 0; i < m; i++) if (w->G[i][j] < mn) mn = w->G[i][j]; v[j] = mn; }
            lapv = lap_jv(m, w->G, p, u, v, 1);
        }
        w->nlap++;
        PROF_T(tl_noext, it == 0 ? 3 : 4, pcl);
        double gx0 = NAN;
        if (ns) {   /* <G, X> needs the true G: restore after keeping the reduced costs */
            int k = 0; double gxs = 0;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) if (w->fbl[i][j]) { double g = gsave[k++]; gxs += (g - FBIG) * w->X[i][j]; }
            gx0 = gxs;   /* correction of <G', X> towards <G, X> when <G, X> is taken from G' */
        }
        /* <G, X> = 2 <MX, X> + <Cr, X>, and f = <MX, X> + <Cr, X> + trST */
        double gx = opt_fuse ? 2 * (f - w->trST - s2) + s2 : dotm(m, w->G, w->X);
        if (ns && !opt_fuse) gx += gx0;
        double z = f - gx + lapv + K;
        if (opt_umax && isnan(gp)) {   /* -umax: z_k + U_k[a][b] bounds every child (a, b) at each iteration k: keep the max */
            if (best == -INFINITY) for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->Wmax[i][j] = -INFINITY;
            for (int i = 0; i < m; i++) { const double *g = w->G[i]; double *wm = w->Wmax[i], zu = z - u[i];
                for (int j = 0; j < m; j++) { double t = zu + g[j] - v[j]; if (t > wm[j]) wm[j] = t; } }
        }
        if (z > best && isnan(gp)) {   /* U from exact LAP duals only */
            best = z;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->U[i][j] = w->G[i][j] - u[i] - v[j];
            memcpy(w->Up, p, sizeof(int) * m); w->up_ok = 1;
            /* shift: we store z without K? no: z includes K */
        }
        if (ns) { int k = 0; for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) if (w->fbl[i][j]) w->G[i][j] = gsave[k++]; }

        if (lg && it < 512) { lgz[it] = target - best; lgf[it] = f + K - target; }
        if (best > target) { it++; lgx = 1; break; }
        double xmp = 0;
        unsigned long long pca = g_prof ? __rdtsc() : 0;
        applyMperm(w, m, p, w->MP, opt_fuse ? &xmp : NULL);
        PROF_T(tl_noext, 5, pca);
        /* cost of the permutation p (upper bound) */
        double cp = K; for (int i = 0; i < m; i++) cp += w->APB[i][p[i]] * 0; /* placeholder */
        {
            double c = K;
            for (int i = 0; i < m; i++) { c += w->APB[i][p[i]] + w->Cr[i][p[i]]; }
            /* note: <P, Ar P Br> = sum_i (Ar P Br)[i][p[i]] */
            cp = c;
        }
        if (cp < *ub_found) { *ub_found = cp; if (bestp) memcpy(bestp, p, sizeof(int) * m); }
        /* line search on D = P - X */
        if (PA) {   /* pairwise step */
            int ai = 0, fi = -1; double ga = -INFINITY;
            for (int t = 0; t < PA->n; t++) {
                double g = 0; const int *pt = PA->perm[t];
                if (pt[0] < 0) g = dotm(m, w->G, PA->A0); else for (int i = 0; i < m; i++) g += w->G[i][pt[i]];
                if (g > ga) { ga = g; ai = t; }
                if (pt[0] >= 0 && fi < 0 && !memcmp(pt, p, sizeof(int) * m)) fi = t;
            }
            double gd = lapv - ga;
            if (fi == ai || gd > -1e-9) { it++; break; }
            const int *pa = PA->perm[ai]; double (*Ma)[NMAX] = PA->MA[ai];
            double q = 0;
            for (int i = 0; i < m; i++) q += w->MP[i][p[i]] - Ma[i][p[i]];
            if (pa[0] < 0) { for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) q -= PA->A0[i][j] * (w->MP[i][j] - Ma[i][j]); }
            else for (int i = 0; i < m; i++) q -= w->MP[i][pa[i]] - Ma[i][pa[i]];
            double amax = PA->lam[ai];
            double alpha = (q > 1e-12) ? -gd / (2 * q) : amax;
            if (alpha > amax) alpha = amax;
            if (pa[0] < 0) { for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->X[i][j] -= alpha * PA->A0[i][j]; }
            else for (int i = 0; i < m; i++) w->X[i][pa[i]] -= alpha;
            double crp = 0;
            for (int i = 0; i < m; i++) { w->X[i][p[i]] += alpha; crp += w->Cr[i][p[i]];
                double *mx = w->MX[i], *g = w->G[i]; const double *mp = w->MP[i], *ma = Ma[i], *cr = w->Cr[i];
                for (int j = 0; j < m; j++) { double t = mx[j] + alpha * (mp[j] - ma[j]); mx[j] = t; g[j] = 2 * t + cr[j]; } }
            s2 += alpha * (crp - PA->cr[ai]);
            f += alpha * gd + alpha * alpha * q;
            if (fi < 0) {
                if (PA->n == AMAX) { it++; break; }
                fi = PA->n++; memcpy(PA->perm[fi], p, sizeof(int) * m); PA->lam[fi] = 0; PA->cr[fi] = crp;
                for (int i = 0; i < m; i++) memcpy(PA->MA[fi][i], w->MP[i], sizeof(double) * m);
            }
            PA->lam[fi] += alpha; PA->lam[ai] -= alpha;
            if (PA->lam[ai] <= 1e-12) {   /* drop step: remove the away atom */
                int last = --PA->n;
                if (ai != last) { memcpy(PA->perm[ai], PA->perm[last], sizeof(int) * NMAX); PA->lam[ai] = PA->lam[last]; PA->cr[ai] = PA->cr[last];
                    for (int i = 0; i < m; i++) memcpy(PA->MA[ai][i], PA->MA[last][i], sizeof(double) * m); }
            }
            if (it + 1 >= nfw2 && f + K < target) { it++; break; }
            continue;
        }
        double gd = (isnan(gp) ? lapv : gp) - gx;   /* <G, P - X> = <G,P> - <G,X>, <G,P> = lap value when exact */
        double q = 0;
        if (opt_fuse) {   /* q = <P - X, MP - MX> = sum_i (MP - MX)[i][p_i] - <X, MP> + <X, MX> */
            for (int i = 0; i < m; i++) q += w->MP[i][p[i]] - w->MX[i][p[i]];
            q += (f - w->trST - s2) - xmp;
        } else
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
            double d = ((p[i] == j) ? 1.0 : 0.0) - w->X[i][j];
            double md = w->MP[i][j] - w->MX[i][j];
            q += d * md;
        }
        double alpha = (q > 1e-12) ? -gd / (2 * q) : 1.0;
        if (alpha > 1) alpha = 1; if (alpha <= 0) { it++; break; }
        unsigned long long pcf = g_prof ? __rdtsc() : 0;
        if (opt_fuse) {   /* X, MX and the next gradient G in one pass */
            double b = 1 - alpha, cpx = 0;
            for (int i = 0; i < m; i++) { double *x = w->X[i], *mx = w->MX[i], *g = w->G[i]; const double *mp = w->MP[i], *cr = w->Cr[i];
                for (int j = 0; j < m; j++) { x[j] *= b; double t = mx[j] + alpha * (mp[j] - mx[j]); mx[j] = t; g[j] = 2 * t + cr[j]; }
                x[p[i]] += alpha; cpx += cr[p[i]]; }
            s2 = b * s2 + alpha * cpx;
        } else
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
            double d = ((p[i] == j) ? 1.0 : 0.0) - w->X[i][j];
            w->X[i][j] += alpha * d;
            w->MX[i][j] += alpha * (w->MP[i][j] - w->MX[i][j]);
        }
        f += alpha * gd + alpha * alpha * q;
        PROF_T(tl_noext, 6, pcf);
        if (opt_upd && (it + 1) % opt_upd == 0 && it + 1 < nfw) { unsigned long long pcu = g_prof ? __rdtsc() : 0; int ch;
          if (w->inh_live) { setup_qpb(w, m); w->inh_live = 0; update_st(w, m); ch = 1; g_inh_switch++; } else ch = update_st(w, m);
          if (ch) {
            applyM(w, m, w->X, w->MX);
            s2 = dotm(m, w->Cr, w->X);
            f = dotm(m, w->X, w->MX) + s2 + w->trST;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->G[i][j] = 2 * w->MX[i][j] + w->Cr[i][j];
          }
          PROF_T(tl_noext, 7, pcu);
        }
        /* termination rule of Brixius-Anstreicher: stop at NFW2 if the QP
         * value f(X_k) is already below the incumbent (cannot fathom) */
        if (it + 1 >= nfwstop && f + K < target) {
            /* before giving up, one S, T update may lift f above the target */
            if (opt_updstop && !w->inh_live && update_st(w, m)) {
                applyM(w, m, w->X, w->MX);
                s2 = dotm(m, w->Cr, w->X);
                f = dotm(m, w->X, w->MX) + s2 + w->trST;
                for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->G[i][j] = 2 * w->MX[i][j] + w->Cr[i][j];
                if (f + K >= target) continue;
            }
            it++; lgx = 2; break;
        }
    }
    if (lg) {
        if (lgx == 4 && it >= nfwc) lgx = 3;
        char buf[512 * 24 + 64]; int o = sprintf(buf, "%d %d %d %d %d %d", tl_depth, m, nfw, nfw2, it, lgx);
        for (int k = 0; k < it && k < 512; k++) o += sprintf(buf + o, " %.1f %.1f", lgz[k], lgf[k]);
        buf[o++] = '\n'; fwrite(buf, 1, o, g_fwlog);
    }
    (void)hist_best_at;
    static int trace = -1; if (trace < 0) trace = getenv("QPB_TRACE") != NULL;
    w->nfw += it; w->flast = f + K; if (trace) fprintf(stderr, "f(X)=%.3f z=%.3f\n", f + K, best);
    return best;
}


/* ---------- RLT1 dual ascent (Hahn and Grant; steps 1a to 3 of Hahn, Hightower, Johnson,
 * Guignard and Roucairol, level-1 case) ----------
 * Cost of a completion = R + sum_ab L[a][b] x_ab + sum_{a!=c, b!=d} Q[a][b][c][d] y_abcd, with
 * y_abcd = y_cdab. Every operation below keeps the cost of every permutation unchanged and every
 * L, Q entry nonnegative (reduced costs of exact LAP duals), so R is a valid lower bound at any time. */
typedef struct { int m; double *Q; double L[NMAX][NMAX]; double R; double vs[NMAX * NMAX][NMAX]; int warm; } rlt_t;
#define QX(m, a, b, c, d) ((((size_t)(a) * (m) + (b)) * (m) + (c)) * (m) + (d))
static int opt_rhalf = 0;   /* 1: a submatrix collects only half of its twin costs */
static void rlt_init(rlt_t *r, int m, double Ar[NMAX][NMAX], double Br[NMAX][NMAX], double Cr[NMAX][NMAX], double K) {
    r->m = m; r->R = K; r->warm = 0;
    if (!r->Q) r->Q = malloc(sizeof(double) * NMAX * NMAX * NMAX * NMAX);
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        r->L[a][b] = Cr[a][b] + Ar[a][a] * Br[b][b];
        for (int c = 0; c < m; c++) for (int d = 0; d < m; d++)
            r->Q[QX(m, a, b, c, d)] = (c != a && d != b) ? Ar[a][c] * Br[b][d] : 0;
    }
}
/* LAP with u_i = min_j (c_ij - v_j): returns sum u + sum v, leaves c - u - v >= 0 in red (exactly, in floating point) */
static double lap_reduce(int n, double c[NMAX][NMAX], double *v, int warm) {
    int rs[NMAX]; double u[NMAX];
    if (!warm) for (int j = 0; j < n; j++) { double mn = INFINITY; for (int i = 0; i < n; i++) mn = c[i][j] < mn ? c[i][j] : mn; v[j] = mn; }
    lap_sp(n, c, rs, u, v, 0);
    double tot = 0;
    for (int i = 0; i < n; i++) { double mn = INFINITY; for (int j = 0; j < n; j++) { double h = c[i][j] - v[j]; c[i][j] = h; mn = h < mn ? h : mn; }
        for (int j = 0; j < n; j++) c[i][j] -= mn; tot += mn; }
    for (int j = 0; j < n; j++) tot += v[j];
    return tot;
}
/* one pass: spread L over the rows of each submatrix, one LAP per submatrix (after collecting
 * its twin costs), its value back to L, then the LAP on L raises R */
static double rlt_pass(rlt_t *r) {
    int m = r->m, n1 = m - 1; double c[NMAX][NMAX];
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {   /* step 1a */
        double t = r->L[a][b] / n1; if (t == 0) continue;
        for (int cc = 0; cc < m; cc++) if (cc != a) { double *q = r->Q + QX(m, a, b, cc, 0); for (int d = 0; d < m; d++) if (d != b) q[d] += t; }
        r->L[a][b] = 0;
    }
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {   /* step 2 */
        int i = 0;
        for (int cc = 0; cc < m; cc++) if (cc != a) { int j = 0;
            for (int d = 0; d < m; d++) if (d != b) {
                double *tw = r->Q + QX(m, cc, d, a, b), *q = r->Q + QX(m, a, b, cc, d);
                double x = opt_rhalf ? 0.5 * *tw : *tw;
                *q += x; *tw -= x; c[i][j++] = *q; }
            i++; }
        double val = lap_reduce(n1, c, r->vs[a * m + b], r->warm);
        i = 0;
        for (int cc = 0; cc < m; cc++) if (cc != a) { int j = 0; double *q = r->Q + QX(m, a, b, cc, 0);
            for (int d = 0; d < m; d++) if (d != b) q[d] = c[i][j++];
            i++; }
        r->L[a][b] += val;
    }
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) c[a][b] = r->L[a][b];   /* step 3 */
    double vm[NMAX]; double val = lap_reduce(m, c, vm, 0);
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) r->L[a][b] = c[a][b];
    r->R += val; r->warm = 1;
    return r->R;
}


/* Lagrangian evaluation on top of a DP state: for ANY antisymmetric shift lam (lam_abcd added to
 * Q_abcd, subtracted from its twin Q_cdab) the value R + LAP_master(L_ab + LAP_ab(Q + lam)) is a
 * valid lower bound (exact LAPs, no sign condition). Subgradient: x_ab y^ab_cd - x_cd y^cd_ab. */
static double rlt_lag(rlt_t *r, const double *lam, int *xm, int ys[NMAX * NMAX][NMAX]) {
    int m = r->m, n1 = m - 1; double c[NMAX][NMAX], Lm[NMAX][NMAX];
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        int i = 0;
        for (int cc = 0; cc < m; cc++) if (cc != a) { int j = 0;
            for (int d = 0; d < m; d++) if (d != b) { size_t k = QX(m, a, b, cc, d), kt = QX(m, cc, d, a, b);
                c[i][j++] = r->Q[k] + lam[k] - lam[kt]; }
            i++; }
        int rs[NMAX]; double u[NMAX], *v = r->vs[a * m + b];
        double val = lap_sp(n1, c, rs, u, v, 0);
        { double tot = 0; for (int ii = 0; ii < n1; ii++) { double mn = INFINITY; for (int j = 0; j < n1; j++) { double h = c[ii][j] - v[j]; mn = h < mn ? h : mn; } tot += mn; }
          for (int j = 0; j < n1; j++) tot += v[j]; val = tot; }
        Lm[a][b] = r->L[a][b] + val;
        i = 0; for (int cc = 0; cc < m; cc++) if (cc != a) { int col = rs[i]; ys[a * m + b][cc] = col < b ? col : col + 1; i++; }
        ys[a * m + b][a] = -1;
    }
    double u[NMAX], v[NMAX];
    for (int j = 0; j < m; j++) { double mn = INFINITY; for (int i = 0; i < m; i++) mn = Lm[i][j] < mn ? Lm[i][j] : mn; v[j] = mn; }
    lap_sp(m, Lm, xm, u, v, 0);
    double tot = 0; for (int i = 0; i < m; i++) { double mn = INFINITY; for (int j = 0; j < m; j++) { double h = Lm[i][j] - v[j]; mn = h < mn ? h : mn; } tot += mn; }
    for (int j = 0; j < m; j++) tot += v[j];
    return r->R + tot;
}


/* ---------- symmetry: automorphism groups of A and B ---------- */
#define GMAX 4096
typedef struct { int n; int perm[GMAX][NMAX]; } group_t;
static group_t GA, GB;

static int aut_rec(const double M[NMAX][NMAX], int *sig, char *usedc, int d, group_t *g, long *budget) {
    if (--(*budget) < 0) return -1;
    if (d == N) { if (g->n >= GMAX) return -1; memcpy(g->perm[g->n++], sig, sizeof(int) * N); return 0; }
    for (int c = 0; c < N; c++) if (!usedc[c]) {
        if (M[d][d] != M[c][c]) continue;
        int ok = 1;
        for (int k = 0; k < d && ok; k++) if (M[d][k] != M[c][sig[k]]) ok = 0;
        if (!ok) continue;
        sig[d] = c; usedc[c] = 1;
        if (aut_rec(M, sig, usedc, d + 1, g, budget) < 0) return -1;
        usedc[c] = 0;
    }
    return 0;
}
static void compute_aut(const double M[NMAX][NMAX], group_t *g) {
    int sig[NMAX]; char used[NMAX] = {0}; long budget = 20000000;
    g->n = 0;
    if (aut_rec(M, sig, used, 0, g, &budget) < 0) { g->n = 1; for (int i = 0; i < N; i++) g->perm[0][i] = i; }
}

/* ---------- node / search state ---------- */
typedef struct {
    int rc[NMAX];   /* row -> column, -1 if free */
    int cr[NMAX];   /* column -> row, -1 if free */
    int depth;
    unsigned fb[NMAX];   /* forbidden pairs (-fb): bit j of fb[i] set if row i cannot take column j in this subtree */
} node_t;

typedef struct {
    long nodes, fathom_at_bound, lap, fw;
    long nodes_depth[NMAX + 1];
    double t_bound;
} stats_t;

static _Atomic double g_ub;           /* shared incumbent value */
static int g_best[NMAX];
static pthread_mutex_t g_ub_lock = PTHREAD_MUTEX_INITIALIZER;
static double g_t0;
static FILE *g_log = NULL;            /* event journal (JSONL) */
static const char *g_prog = NULL;     /* progress file, rewritten every 10 s */
static double g_prog_last = 0;
static _Atomic long g_nodes_all = 0;
static long g_depth_all[NMAX + 1];
/* serial search position, for live monitoring: at depth d the search is in child
 * g_pos[d] (1-based) of g_npos[d] kept children; g_nextlb[d] is the bound of the next
 * unexplored sibling (children are sorted by bound, so the smallest one left) */
static int g_pos[NMAX + 1], g_npos[NMAX + 1], g_curdepth = -1;
static double g_nextlb[NMAX + 1];
static void write_progress(int final) {
    if (!g_prog) return;
    double t = now();
    if (!final && t - g_prog_last < 10) return;
    g_prog_last = t;
    char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.tmp", g_prog);
    FILE *f = fopen(tmp, "w"); if (!f) return;
    fprintf(f, "{\"elapsed_s\":%.1f,\"nodes\":%ld,\"ub\":%.0f,\"final\":%d,\"nodes_depth\":[", t - g_t0, (long)g_nodes_all, (double)g_ub, final);
    int last = 0; for (int d = 0; d <= NMAX; d++) if (g_depth_all[d]) last = d;
    for (int d = 0; d <= last; d++) fprintf(f, "%ld%s", g_depth_all[d], d < last ? "," : "");
    fprintf(f, "]");
    if (g_curdepth >= 0) {
        double lb = INFINITY;
        fprintf(f, ",\"path\":[");
        for (int d = 0; d < g_curdepth; d++) { fprintf(f, "%s[%d,%d]", d ? "," : "", g_pos[d], g_npos[d]); if (g_nextlb[d] < lb) lb = g_nextlb[d]; }
        fprintf(f, "],\"lb_open\":%.2f", isinf(lb) ? -1.0 : lb);
    }
    fprintf(f, "}\n"); fclose(f); rename(tmp, g_prog);
}

static void report_solution(const int *p, double c, int who) {
    pthread_mutex_lock(&g_ub_lock);
    if (c < g_ub) {
        g_ub = c; memcpy(g_best, p, sizeof(int) * N);
        if (opt_verbose) fprintf(stderr, "[%.2fs] worker %d new incumbent %.0f\n", now() - g_t0, who, c);
        if (g_log) { fprintf(g_log, "{\"t\":%.3f,\"event\":\"incumbent\",\"worker\":%d,\"value\":%.0f}\n", now() - g_t0, who, c); fflush(g_log); }
    }
    pthread_mutex_unlock(&g_ub_lock);
}

/* evaluate a node: builds reduced problem, runs QPB.
 * outputs: m, F (free rows), L (free cols), z, U in w->U */
typedef struct { int m; int F[NMAX], L[NMAX]; double z; double K; } eval_t;

/* -dom: exchange dominance. For fixed facilities i -> j and k -> l, swapping them changes
 * the cost of any completion p by
 *   D = (A_ii - A_kk)(B_ll - B_jj) + C0_il + C0_kj - C0_ij - C0_kl
 *     + 2 sum_{t != i,k} (A_it - A_kt)(B_{l p(t)} - B_{j p(t)}).
 * Over the free facilities the sum is sum_a alpha_a beta_{pi(a)}, a rank-one assignment:
 * its maximum pairs alpha and beta sorted the same way (rearrangement inequality). If even
 * the maximum is < 0, every permutation of the subtree has a strictly better one (the swap),
 * so the subtree holds no optimal permutation; the optimal ones are never removed, which keeps
 * the search exact together with bound pruning and symmetry. */
static int cmp_dbl(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static inline void isort(double *v, int n) {   /* insertion sort, n <= NMAX: faster than qsort here */
    for (int x = 1; x < n; x++) { double t = v[x]; int y = x; while (y > 0 && v[y - 1] > t) { v[y] = v[y - 1]; y--; } v[y] = t; }
}
static int dominated(const node_t *nd) {
    int fx[NMAX], nf = 0, fr[NMAX], nr = 0, fc[NMAX], nc = 0;
    for (int i = 0; i < N; i++) { if (nd->rc[i] >= 0) fx[nf++] = i; else fr[nr++] = i; }
    for (int j = 0; j < N; j++) if (nd->cr[j] < 0) fc[nc++] = j;
    if (nf < 2) return 0;
    double al[NMAX], be[NMAX];
    for (int x = 0; x < nf; x++) for (int y = x + 1; y < nf; y++) {
        int i = fx[x], k = fx[y], j = nd->rc[i], l = nd->rc[k];
        double D = (A[i][i] - A[k][k]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[k][j] - C0[i][j] - C0[k][l];
        for (int z = 0; z < nf; z++) { int t = fx[z]; if (t == i || t == k) continue; D += 2 * (A[i][t] - A[k][t]) * (B[l][nd->rc[t]] - B[j][nd->rc[t]]); }
        for (int a = 0; a < nr; a++) al[a] = A[i][fr[a]] - A[k][fr[a]];
        for (int b = 0; b < nc; b++) be[b] = B[l][fc[b]] - B[j][fc[b]];
        isort(al, nr); isort(be, nc);
        double mx = 0; for (int a = 0; a < nr; a++) mx += al[a] * be[a];
        if (D + 2 * mx < -1e-9) return 1;
    }
    if (opt_dom == 3 && nf >= 3) {
        /* -dom3: 3-cycles of fixed facilities i -> l, k -> n, q -> j (j, l, n their locations),
         * both orientations. The free part is (A_i - A_q) b1^T + (A_k - A_q) b2^T with
         * b1 = B_l - B_j, b2 = B_n - B_l (rank two); its maximum over completions is at most
         * the sum of the two rank-one maxima, so the test stays valid (sufficient). */
        for (int x = 0; x < nf; x++) for (int y = x + 1; y < nf; y++) for (int z = y + 1; z < nf; z++) for (int o = 0; o < 2; o++) {
            int S[3] = {fx[x], o ? fx[z] : fx[y], o ? fx[y] : fx[z]}, P[3], Q[3];
            for (int e = 0; e < 3; e++) P[e] = nd->rc[S[e]];
            for (int e = 0; e < 3; e++) Q[e] = P[(e + 1) % 3];
            double D = 0;
            for (int e = 0; e < 3; e++) {
                for (int f = 0; f < 3; f++) D += A[S[e]][S[f]] * (B[Q[e]][Q[f]] - B[P[e]][P[f]]);
                D += C0[S[e]][Q[e]] - C0[S[e]][P[e]];
                for (int g = 0; g < nf; g++) { int t = fx[g]; if (t == S[0] || t == S[1] || t == S[2]) continue;
                    D += 2 * A[S[e]][t] * (B[Q[e]][nd->rc[t]] - B[P[e]][nd->rc[t]]); }
            }
            double mx = 0;
            for (int h = 0; h < 2; h++) {   /* h = 0: (A_S0 - A_S2) (B_Q0 - B_P0); h = 1: (A_S1 - A_S2) (B_Q1 - B_P1) */
                for (int a = 0; a < nr; a++) al[a] = A[S[h]][fr[a]] - A[S[2]][fr[a]];
                for (int b = 0; b < nc; b++) be[b] = B[Q[h]][fc[b]] - B[P[h]][fc[b]];
                isort(al, nr); isort(be, nc);
                for (int a = 0; a < nr; a++) mx += al[a] * be[a];
            }
            if (D + 2 * mx < -1e-9) return 1;
        }
    }
    if (opt_dom < 2 || opt_dom == 3 || nr < 2) return 0;
    /* -dom 2: fixed facility i -> j against a free location l. Whatever free facility t
     * takes l, moving i to l and t to j changes the cost by the same formula with k = t;
     * the free part runs over the free facilities other than t and the free locations
     * other than l. If the maximum is < 0 for every t, every completion is improved. */
    for (int x = 0; x < nf; x++) {
        int i = fx[x], j = nd->rc[i];
        for (int c = 0; c < nc; c++) {
            int l = fc[c], all = 1;
            int nb = 0; for (int b = 0; b < nc; b++) if (b != c) be[nb++] = B[l][fc[b]] - B[j][fc[b]];
            qsort(be, nb, sizeof(double), cmp_dbl);
            for (int r = 0; r < nr && all; r++) {
                int t = fr[r];
                double D = (A[i][i] - A[t][t]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[t][j] - C0[i][j] - C0[t][l];
                for (int z = 0; z < nf; z++) { int u = fx[z]; if (u == i) continue; D += 2 * (A[i][u] - A[t][u]) * (B[l][nd->rc[u]] - B[j][nd->rc[u]]); }
                int na = 0; for (int a = 0; a < nr; a++) if (a != r) al[na++] = A[i][fr[a]] - A[t][fr[a]];
                qsort(al, na, sizeof(double), cmp_dbl);
                double mx = 0; for (int a = 0; a < na; a++) mx += al[a] * be[a];
                if (D + 2 * mx >= -1e-9) all = 0;
            }
            if (all) return 1;
        }
    }
    return 0;
}

/* same test restricted to the pairs that contain the new assignment i -> j of a child of nd */
static int dominated_child(const node_t *nd, int i, int j) {
    int fx[NMAX], nf = 0, fr[NMAX], nr = 0, fc[NMAX], nc = 0;
    for (int t = 0; t < N; t++) { if (nd->rc[t] >= 0) fx[nf++] = t; else if (t != i) fr[nr++] = t; }
    for (int c = 0; c < N; c++) if (nd->cr[c] < 0 && c != j) fc[nc++] = c;
    double al[NMAX], be[NMAX];
    for (int y = 0; y < nf; y++) {
        int k = fx[y], l = nd->rc[k];
        double D = (A[i][i] - A[k][k]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[k][j] - C0[i][j] - C0[k][l];
        for (int z = 0; z < nf; z++) { int t = fx[z]; if (t == k) continue; D += 2 * (A[i][t] - A[k][t]) * (B[l][nd->rc[t]] - B[j][nd->rc[t]]); }
        for (int a = 0; a < nr; a++) al[a] = A[i][fr[a]] - A[k][fr[a]];
        for (int b = 0; b < nc; b++) be[b] = B[l][fc[b]] - B[j][fc[b]];
        isort(al, nr); isort(be, nc);
        double mx = 0; for (int a = 0; a < nr; a++) mx += al[a] * be[a];
        if (D + 2 * mx < -1e-9) return 1;
    }
    return 0;
}

/* the test for every child (F[a] -> L[b]) of a node at once: the sorted vectors depend on
 * (child row, fixed facility) and (child column, fixed facility) only, so they are sorted
 * m * nf times each, and each child costs O(nf * m) */
static void dominated_children(const node_t *nd, const eval_t *ev, unsigned char out[NMAX][NMAX]) {
    int m = ev->m, fx[NMAX], nf = 0;
    for (int t = 0; t < N; t++) if (nd->rc[t] >= 0) fx[nf++] = t;
    memset(out, 0, sizeof(unsigned char) * NMAX * NMAX);
    if (nf == 0 || m < 2) return;
    static __thread double SA[NMAX][NMAX][NMAX], SB[NMAX][NMAX][NMAX];   /* [fixed y][a or b][sorted, m - 1] */
    for (int y = 0; y < nf; y++) {
        int k = fx[y], l = nd->rc[k];
        for (int a = 0; a < m; a++) { int i = ev->F[a], n = 0; double *v = SA[y][a];
            for (int r = 0; r < m; r++) if (r != a) v[n++] = A[i][ev->F[r]] - A[k][ev->F[r]];
            isort(v, n); }
        for (int b = 0; b < m; b++) { int j = ev->L[b], n = 0; double *v = SB[y][b];
            for (int c = 0; c < m; c++) if (c != b) v[n++] = B[l][ev->L[c]] - B[j][ev->L[c]];
            isort(v, n); }
    }
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        int i = ev->F[a], j = ev->L[b];
        for (int y = 0; y < nf; y++) {
            int k = fx[y], l = nd->rc[k];
            double D = (A[i][i] - A[k][k]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[k][j] - C0[i][j] - C0[k][l];
            for (int z = 0; z < nf; z++) { int t = fx[z]; if (t == k) continue; D += 2 * (A[i][t] - A[k][t]) * (B[l][nd->rc[t]] - B[j][nd->rc[t]]); }
            const double *u = SA[y][a], *v = SB[y][b]; double mx = 0;
            for (int x = 0; x < m - 1; x++) mx += u[x] * v[x];
            if (D + 2 * mx < -1e-9) { out[a][b] = 1; break; }
        }
    }
}

/* -domlb: bound and exchange together. For every completion P of the node, cost(P) >= z + <U, P>
 * (QPB at the node) and the swap of fixed i, k changes the cost by D + <W, P>, W rank one.
 * If for some lambda in [0, 1]
 *   min_P  lambda (z + <U, P> - thr) - (1 - lambda)(D + <W, P>)  > 0,
 * every P either costs more than thr or is strictly improved by the swap: the node goes. */
static int opt_domlb = 0; static long g_domlb_pruned = 0;
static int domlb(const node_t *nd, const eval_t *ev, const work_t *w) {
    int m = ev->m, fx[NMAX], nf = 0;
    for (int t = 0; t < N; t++) if (nd->rc[t] >= 0) fx[nf++] = t;
    double thr = prune_thr(g_ub), c[NMAX][NMAX]; int rs[NMAX]; double u[NMAX + 1], v[NMAX + 1];
    static const double lam[] = {0.3, 0.5, 0.7};
    for (int x = 0; x < nf; x++) for (int y = x + 1; y < nf; y++) {
        int i = fx[x], k = fx[y], j = nd->rc[i], l = nd->rc[k];
        double D = (A[i][i] - A[k][k]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[k][j] - C0[i][j] - C0[k][l];
        for (int z = 0; z < nf; z++) { int t = fx[z]; if (t == i || t == k) continue; D += 2 * (A[i][t] - A[k][t]) * (B[l][nd->rc[t]] - B[j][nd->rc[t]]); }
        double al[NMAX], be[NMAX];
        for (int a = 0; a < m; a++) al[a] = 2 * (A[i][ev->F[a]] - A[k][ev->F[a]]);
        for (int b = 0; b < m; b++) be[b] = B[l][ev->L[b]] - B[j][ev->L[b]];
        for (int q = 0; q < 3; q++) {
            double L = lam[q];
            for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) c[a][b] = L * w->U[a][b] - (1 - L) * al[a] * be[b];
            double val = lap(m, c, rs, u, v) + L * (ev->z - thr) - (1 - L) * D;
            if (val > 1e-6) return 1;
        }
    }
    return 0;
}

/* -domcov k: covering by several swaps. Pieces q: each swap of two fixed facilities gives
 * g_q(P) = -(D_q + <W_q, P>) (> 0 when the swap improves P), and the bound gives
 * g_0(P) = z + <U, P> - thr (> 0 when P is above the threshold). If for weights mu >= 0
 * summing to 1, min_P sum_q mu_q g_q(P) > 0, every P has some q with g_q(P) > 0: either it
 * costs more than thr, or a swap improves it strictly; the node holds no optimal permutation
 * below the threshold. The weights follow k steps of exponentiated subgradient ascent,
 * one LAP per step. */
static int opt_domcov = 0; static long g_domcov_pruned = 0;
static int domcov(const node_t *nd, const eval_t *ev, const work_t *w) {
    int m = ev->m, fx[NMAX], nf = 0;
    for (int t = 0; t < N; t++) if (nd->rc[t] >= 0) fx[nf++] = t;
    if (nf < 2) return 0;
    enum { QMAX = NMAX * (NMAX - 1) / 2 + 1 };
    static __thread double cq[QMAX], al[QMAX][NMAX], be[QMAX][NMAX], mu[QMAX];
    double thr = prune_thr(g_ub); int nq = 1;
    cq[0] = ev->z - thr;
    for (int x = 0; x < nf; x++) for (int y = x + 1; y < nf; y++) {
        int i = fx[x], k = fx[y], j = nd->rc[i], l = nd->rc[k];
        double D = (A[i][i] - A[k][k]) * (B[l][l] - B[j][j]) + C0[i][l] + C0[k][j] - C0[i][j] - C0[k][l];
        for (int z = 0; z < nf; z++) { int t = fx[z]; if (t == i || t == k) continue; D += 2 * (A[i][t] - A[k][t]) * (B[l][nd->rc[t]] - B[j][nd->rc[t]]); }
        cq[nq] = -D;
        for (int a = 0; a < m; a++) al[nq][a] = -2 * (A[i][ev->F[a]] - A[k][ev->F[a]]);
        for (int b = 0; b < m; b++) be[nq][b] = B[l][ev->L[b]] - B[j][ev->L[b]];
        nq++;
    }
    for (int q = 0; q < nq; q++) mu[q] = 1.0 / nq;
    double c[NMAX][NMAX], u[NMAX + 1], v[NMAX + 1]; int rs[NMAX];
    double scale = fabs(cq[0]) + 1;
    for (int it = 0; it < opt_domcov; it++) {
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
            double t = mu[0] * w->U[a][b];
            for (int q = 1; q < nq; q++) t += mu[q] * al[q][a] * be[q][b];
            c[a][b] = t;
        }
        double g = lap(m, c, rs, u, v);
        for (int q = 0; q < nq; q++) g += mu[q] * cq[q];
        if (g > 1e-6) return 1;
        /* piece values at the minimizer, then mu_q *= exp(eta v_q) */
        double vq[QMAX], vmax = 0;
        for (int q = 0; q < nq; q++) {
            double t = cq[q];
            if (q == 0) for (int a = 0; a < m; a++) t += w->U[a][rs[a]];
            else for (int a = 0; a < m; a++) t += al[q][a] * be[q][rs[a]];
            vq[q] = t; if (fabs(t) > vmax) vmax = fabs(t);
        }
        if (it == 0) scale = vmax + 1e-9;
        double tot = 0;
        for (int q = 0; q < nq; q++) { mu[q] *= exp(2.0 * vq[q] / scale); tot += mu[q]; }
        for (int q = 0; q < nq; q++) mu[q] /= tot;
    }
    return 0;
}

static int opt_enum = 0; static long g_enum_nodes = 0;
/* depth-first over the completions, facility a to location b; the partial cost is a lower bound
 * when every entry of A, B and C0 is nonnegative (checked at start, -enum refused otherwise) */
static void enum_rec(work_t *w, int m, int a, double cost, double cut, double *best, const int *L) {
    if (a == m) { if (cost < *best) { *best = cost; memcpy(w->eperm_best, w->eperm, sizeof(int) * m); } return; }
    double lim = *best < cut ? *best : cut;
    for (int b = 0; b < m; b++) if (!(w->eused >> b & 1)) {
        double c = cost + w->Cr[a][b] + w->Ar[a][a] * w->Br[b][b];
        for (int c2 = 0; c2 < a; c2++) c += w->Ar[a][c2] * w->Br[b][w->eperm[c2]] + w->Ar[c2][a] * w->Br[w->eperm[c2]][b];
        if (c >= lim) continue;
        w->eperm[a] = b; w->eused |= 1u << b; enum_rec(w, m, a + 1, c, cut, best, L); w->eused &= ~(1u << b);
        lim = *best < cut ? *best : cut;
    }
    (void)L;
}


/* ---------- -domtab: dominance between nodes with the same free facilities and free locations ----------
 * Two such nodes N1, N2 have the same quadratic part on every completion pi; their costs differ by
 * K2 - K1 + sum_a (Cr2 - Cr1)[a][pi(a)]. If K2 - K1 + LAP(Cr2 - Cr1) >= 0, every completion of N2 costs
 * at least the same completion of N1. In serial depth-first search N1's subtree is finished before
 * N2 is evaluated (same masks, so neither descends from the other), hence N2 can be pruned.
 * Cr is integer for integer data and is stored exactly as int32. */
static int opt_domtab = 0, opt_dtmlo = 4, opt_dtmhi = 30, opt_dtmax = 16; static long g_domtab_pruned = 0, g_domtab_ins = 0, g_domtab_lap = 0;
typedef struct { unsigned long long key; double K, z; int next; long off, uoff; } dtent_t;
static float *g_dtu = NULL; static long g_dtuu = 0; static long g_domtab_tr = 0;
static int *g_dth = NULL; static dtent_t *g_dte = NULL; static int g_dtn = 0, g_dtcap = 0; static int *g_dta = NULL; static long g_dtau = 0, g_dtacap = 0;
static void domtab_init(void) {
    if (g_dth) return;
    g_dth = malloc(sizeof(int) << 24); memset(g_dth, 0xff, sizeof(int) << 24); g_dtcap = 8 << 20; g_dte = malloc(sizeof(dtent_t) * g_dtcap);
    g_dtacap = 300L << 20; g_dta = malloc(sizeof(int) * g_dtacap); g_dtu = malloc(sizeof(float) * g_dtacap);
}
static inline unsigned dt_hash(unsigned long long key) { return (unsigned)((key * 0x9E3779B97F4A7C15ull) >> 40); }
/* returns 1 if the node can be pruned: dominated by a twin, or its bound transferred from a twin exceeds thr.
 * Transfer: cost(N1, pi) >= z1 + <U1, P_pi> for every completion (QPB reduced costs, stored rounded down),
 * so cost(N2, pi) >= z1 + K2 - K1 + LAP(U1 + Cr2 - Cr1). */
static int domtab_check(work_t *w, int m, double K, double thr) {
    domtab_init();
    unsigned long long key = ((unsigned long long)w->maskF << 32) | w->maskL; unsigned h = dt_hash(key);
    int seen = 0;
    for (int e = g_dth[h]; e >= 0 && seen < opt_dtmax; e = g_dte[e].next) {
        if (g_dte[e].key != key) continue; seen++;
        const int *c1 = g_dta + g_dte[e].off; const float *u1 = g_dtu + g_dte[e].uoff; double D[NMAX][NMAX], dk = K - g_dte[e].K, rmin = 0;
        for (int a = 0; a < m; a++) { double mn = INFINITY; for (int b = 0; b < m; b++) { double x = w->Cr[a][b] - c1[a * m + b]; D[a][b] = x; mn = x < mn ? x : mn; } rmin += mn; }
        if (dk + rmin >= -1e-9) { g_domtab_pruned++; return 1; }
        int rs[NMAX]; double u[NMAX], v[NMAX]; g_domtab_lap++;
        if (dk + lap(m, D, rs, u, v) >= -1e-9) { g_domtab_pruned++; return 1; }
        if (opt_domtab > 1 && isfinite(g_dte[e].z) && g_dte[e].z + dk > thr - 200) {
            for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) D[a][b] += u1[a * m + b];
            if (g_dte[e].z + dk + lap(m, D, rs, u, v) > thr) { g_domtab_tr++; return 1; }
        }
    }
    return 0;
}
static void domtab_store(work_t *w, int m, double K, double z) {
    domtab_init();
    if (!(g_dtn < g_dtcap && g_dtau + m * m <= g_dtacap)) return;
    unsigned long long key = ((unsigned long long)w->maskF << 32) | w->maskL; unsigned h = dt_hash(key);
    dtent_t *t = &g_dte[g_dtn]; t->key = key; t->K = K; t->z = z; t->next = g_dth[h];
    long off = g_dtau; t->off = off; t->uoff = off;
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        g_dta[off + a * m + b] = (int)lrint(w->Cr[a][b]);
        float f = (float)w->U[a][b]; if ((double)f > w->U[a][b]) f = nextafterf(f, -INFINITY); g_dtu[off + a * m + b] = f; }
    g_dtau += m * m; g_dth[h] = g_dtn++; g_domtab_ins++;
}

static void eval_node1(const node_t *nd, work_t *w, eval_t *ev, int nfw, int nfw2, int who) {
    unsigned long long pc0 = g_prof ? __rdtsc() : 0;
    int inh = w->inh_ok; w->inh_ok = 0;
    int m = 0, mc = 0;
    w->maskF = w->maskL = 0;
    for (int i = 0; i < N; i++) if (nd->rc[i] < 0) { ev->F[m++] = i; w->maskF |= 1u << i; }
    for (int j = 0; j < N; j++) if (nd->cr[j] < 0) { ev->L[mc++] = j; w->maskL |= 1u << j; }
    ev->m = m;
    if (opt_dom && m > 1 && dominated(nd)) {
        ev->z = INFINITY; w->x0_ok = 0; __atomic_fetch_add(&g_dom_pruned, 1, __ATOMIC_RELAXED);
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) w->U[a][b] = 0;
        return;
    }
    w->nfb = 0;
    if (opt_fb) for (int a = 0; a < m; a++) { unsigned f = nd->fb[ev->F[a]];
        for (int b = 0; b < m; b++) { int x = (f >> ev->L[b]) & 1; w->fbl[a][b] = x; w->nfb += x; } }
    double K = 0;
    for (int i = 0; i < N; i++) if (nd->rc[i] >= 0) for (int k = 0; k < N; k++) if (nd->rc[k] >= 0)
        K += A[i][k] * B[nd->rc[i]][nd->rc[k]];
    for (int i = 0; i < N; i++) if (nd->rc[i] >= 0) K += C0[i][nd->rc[i]];
    ev->K = K;
    for (int a = 0; a < m; a++) {
        int i = ev->F[a];
        for (int b = 0; b < m; b++) { w->Ar[a][b] = A[i][ev->F[b]]; w->Br[a][b] = B[ev->L[a]][ev->L[b]]; }
    }
    for (int a = 0; a < m; a++) for (int b = m; b < ((m + 3) & ~3); b++) w->Br[a][b] = 0;
    {   /* Cr = 2 Afix Bfix^T + C0: Afix[a][t] = A[F[a]][k_t], Bfix[b][t] = B[L[b]][rc[k_t]] over the fixed k_t (same summation order as a loop over k) */
        int kf[NMAX], nf = 0; for (int k = 0; k < N; k++) if (nd->rc[k] >= 0) kf[nf++] = k;
        double Af[NMAX][NMAX], Bf[NMAX][NMAX];
        for (int a = 0; a < m; a++) { const double *Ai = A[ev->F[a]], *Bj = B[ev->L[a]];
            for (int t = 0; t < nf; t++) { Af[a][t] = Ai[kf[t]]; Bf[a][t] = Bj[nd->rc[kf[t]]]; } }
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
            double s = 0; const double *x = Af[a], *y = Bf[b];
            for (int t = 0; t < nf; t++) s += x[t] * y[t];
            w->Cr[a][b] = 2 * s + C0[ev->F[a]][ev->L[b]];
        }
    }
    PROF_T(tl_noext, 0, pc0);
    if (opt_domtab && m >= opt_dtmlo && m <= opt_dtmhi && !tl_noext && domtab_check(w, m, K, prune_thr(g_ub))) {
        ev->z = INFINITY; w->x0_ok = 0;
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) w->U[a][b] = 0;
        return;
    }
    double target = prune_thr(g_ub);   /* integer data: fathom if bound > ub - 1 */
    if (m <= 1) {
        double c = K + (m == 1 ? w->Ar[0][0] * w->Br[0][0] + w->Cr[0][0] : 0);
        ev->z = c; if (m == 1) w->U[0][0] = 0;
        if (c < g_ub) { int p[NMAX]; for (int i = 0; i < N; i++) p[i] = nd->rc[i]; if (m == 1) p[ev->F[0]] = ev->L[0]; report_solution(p, c, who); }
        return;
    }
    if (m <= opt_enum) {   /* -enum k: nodes with at most k free facilities are solved by enumeration, the node is closed */
        double best = INFINITY; w->eused = 0; enum_rec(w, m, 0, K, target < g_ub ? g_ub : target, &best, ev->L);
        if (best < g_ub) { int p[NMAX]; for (int i = 0; i < N; i++) p[i] = nd->rc[i]; for (int a2 = 0; a2 < m; a2++) p[ev->F[a2]] = ev->L[w->eperm_best[a2]]; report_solution(p, best, who); }
        ev->z = INFINITY; w->x0_ok = 0; __atomic_fetch_add(&g_enum_nodes, 1, __ATOMIC_RELAXED);
        for (int a2 = 0; a2 < m; a2++) for (int b = 0; b < m; b++) w->U[a2][b] = 0;
        return;
    }
    tl_depth = nd->depth;
    if (opt_topd >= 0 && nd->depth <= opt_topd && !tl_noext) { nfw = (int)(nfw * opt_topx); nfw2 = (int)(nfw2 * opt_topx); }
    if (opt_glbpre >= 0 && nd->depth >= opt_glbpre && m > 2) {
        /* -glbpre d: Gilmore-Lawler bound first, from depth d on. For a completion p of the node,
         * sum_{a,c free} Ar_ac Br_{p(a)p(c)} >= sum_a [Ar_aa Br_{p(a)p(a)} + minprod(Ar_a., Br_{p(a).})],
         * minprod pairing the off-diagonal entries in opposite orders, so
         * cost >= K + LAP(Cr + L). Cheap (m^3 plus one LAP); if it fathoms, Frank-Wolfe is skipped. */
        double ra[NMAX][NMAX], rb[NMAX][NMAX], Lc[NMAX][NMAX], uu[NMAX + 1], vv[NMAX + 1]; int rs[NMAX];
        for (int a = 0; a < m; a++) { int n2 = 0; for (int c = 0; c < m; c++) if (c != a) ra[a][n2++] = w->Ar[a][c]; isort(ra[a], m - 1);
            n2 = 0; for (int c = 0; c < m; c++) if (c != a) rb[a][n2++] = w->Br[a][c]; isort(rb[a], m - 1); }
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) { double t = w->Ar[a][a] * w->Br[b][b] + w->Cr[a][b];
            for (int k = 0; k < m - 1; k++) t += ra[a][k] * rb[b][m - 2 - k]; Lc[a][b] = t; }
        double g = K + lap(m, Lc, rs, uu, vv); g_glbcalls++;
        if (g > target) { ev->z = g; w->x0_ok = 0; g_glbfath++; for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) w->U[a][b] = 0; return; }
    }
    double ubf = INFINITY; int bp[NMAX];
    w->up_ok = 0; w->inh_ok = inh;
    ev->z = qpb_fw(w, m, K, target, nfw, nfw2, &ubf, bp);
    if (opt_umax && ev->z > -INFINITY) for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) { double t = w->Wmax[a][b] - ev->z; if (t > w->U[a][b]) w->U[a][b] = t; }
    if (w->nfb) {   /* forbidden pairs: reduced cost FBIG, shown to the branching rules as just eliminated */
        double cap = fmax(0, target - ev->z) + 1;
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) if (w->fbl[a][b] && w->U[a][b] > cap) w->U[a][b] = cap;
    }
    if (ubf < g_ub) {
        int p[NMAX]; for (int i = 0; i < N; i++) p[i] = nd->rc[i];
        for (int a = 0; a < m; a++) p[ev->F[a]] = ev->L[bp[a]];
        report_solution(p, ubf, who);
    }
}

/* -dual file: a second QPB on the equivalent data A + diag(d), B + diag(g), C0 shifted
 * (same cost for every permutation), tried only when the first one does not fathom; used
 * only as a fathoming test, the branching data stay those of the first bound. One thread. */
static const char *opt_dual = NULL; static int opt_dualmin = 0;
static double A2[NMAX][NMAX], B2[NMAX][NMAX], C02[NMAX][NMAX];
static long g_dualcalls = 0, g_dualfath = 0;
static void print_dual(void) { if (opt_dual) fprintf(stderr, "dual: %ld calls, %ld fathomed\n", g_dualcalls, g_dualfath); }
static void swap_data(void) {
    static double t[NMAX][NMAX];
    memcpy(t, A, sizeof t); memcpy(A, A2, sizeof t); memcpy(A2, t, sizeof t);
    memcpy(t, B, sizeof t); memcpy(B, B2, sizeof t); memcpy(B2, t, sizeof t);
    memcpy(t, C0, sizeof t); memcpy(C0, C02, sizeof t); memcpy(C02, t, sizeof t);
}
static void eval_node(const node_t *nd, work_t *w, eval_t *ev, int nfw, int nfw2, int who) {
    eval_node1(nd, w, ev, nfw, nfw2, who);
    if (!opt_dual || ev->m <= 2 || nd->depth < opt_dualmin || !(ev->z <= prune_thr(g_ub))) return;
    static work_t *ws = NULL; if (!ws) ws = calloc(1, sizeof(work_t));
    ws->x0_ok = 0; eval_t e2;
    swap_data(); eval_node1(nd, ws, &e2, nfw, nfw2, who); swap_data();
    g_dualcalls++;
    if (e2.z > prune_thr(g_ub)) { g_dualfath++; ev->z = e2.z; w->x0_ok = 0;
        for (int a = 0; a < ev->m; a++) for (int b = 0; b < ev->m; b++) w->U[a][b] = 0; }
}

/* -fb: every pair eliminated at this node by z + U stays forbidden below it (for any
 * permutation P of the subtree, cost(P) >= z + <U, P>, U >= 0) */
static void add_fb(node_t *nd, const eval_t *ev, const work_t *w) {
    double thr = prune_thr(g_ub);
    for (int a = 0; a < ev->m; a++) for (int b = 0; b < ev->m; b++)
        if (ev->z + w->U[a][b] > thr) nd->fb[ev->F[a]] |= 1u << ev->L[b];
}

/* symmetry subgroup at a node: elements fixing every assigned row (GA) or column (GB) */
static int stab(const group_t *g, const int *assigned, int *idx) {
    int c = 0;
    for (int e = 0; e < g->n; e++) {
        int ok = 1;
        for (int i = 0; i < N && ok; i++) if (assigned[i] >= 0 && g->perm[e][i] != i) ok = 0;
        if (ok) idx[c++] = e;
    }
    return c;
}

/* child list */
typedef struct { int i, j; double lb; } child_t;

static int cmp_child(const void *a, const void *b) {
    double x = ((const child_t *)a)->lb, y = ((const child_t *)b)->lb;
    return (x > y) - (x < y);
}

/* choose branching and produce children (sorted by bound estimate).
 * Returns number of children. */
/* -sens: sharper elimination from the same LAP. U >= 0 is the reduced cost matrix of an optimal
 * assignment p* (U[i][p*(i)] = 0), and cost(P) >= z + <U, P>. Over the permutations with
 * P_ab = 1 the minimum of <U, P> is U_ab plus the shortest alternating path that sends the row
 * r = p*^-1(b), which lost its column, to the column p*(a) that a left free: row i -> column j
 * costs U_ij, column j -> its row under p* costs 0. One Dijkstra towards each column p*(a)
 * gives all b at once: O(m^3) per node. U_ab is replaced by this minimum, which is >= U_ab and
 * keeps cost(P) >= z + U'_ab for every P with P_ab = 1 (the elimination and branching rules
 * use U only through such single-pair bounds and sums of them). */
static int opt_brecc_lo = 0, opt_brecc_hi = -1, opt_brtie = 0, opt_brdyn = 0, opt_laecc = 0, opt_brall = 0, opt_latie_lo = 0, opt_latie_hi = -1, opt_latiek = 8; static double opt_lak = 0; static double g_rowsumA[NMAX];
static int opt_sens = 0;
static void sharpen_U(work_t *w, int m) {
    if (!w->up_ok) return;
    int inv[NMAX]; for (int a = 0; a < m; a++) inv[w->Up[a]] = a;
    double S[NMAX][NMAX];
    for (int a = 0; a < m; a++) {
        int c = w->Up[a];
        double D[NMAX], d[NMAX]; char done[NMAX];   /* D: column labels, d: row labels */
        for (int j = 0; j < m; j++) { D[j] = INFINITY; done[j] = 0; }
        for (int i = 0; i < m; i++) d[i] = INFINITY;
        D[c] = 0;
        for (int it = 0; it < m; it++) {
            int jm = -1; double mn = INFINITY;
            for (int j = 0; j < m; j++) if (!done[j] && D[j] < mn) { mn = D[j]; jm = j; }
            if (jm < 0) break;
            done[jm] = 1;
            for (int i = 0; i < m; i++) {
                double t = w->U[i][jm] + mn;
                if (t < d[i]) { d[i] = t; int j2 = w->Up[i]; if (!done[j2] && t < D[j2]) D[j2] = t; }
            }
        }
        for (int b = 0; b < m; b++) S[a][b] = w->U[a][b] + (b == c ? 0 : d[inv[b]]);
    }
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) w->U[a][b] = S[a][b];
    w->up_ok = 0;
}

static int branch(const node_t *nd, work_t *w, eval_t *ev, child_t *ch, int who) {
    if (opt_sens) sharpen_U(w, ev->m);
    int m = ev->m; double z = ev->z, thr = prune_thr(g_ub);
    static __thread int idxA[GMAX], idxB[GMAX];
    int nA = opt_sym ? stab(&GA, nd->rc, idxA) : 1;   /* acts on rows: reduces column branching */
    int nB = opt_sym ? stab(&GB, nd->cr, idxB) : 1;   /* acts on columns: reduces row branching */
    /* effective U with orbit max (if an element of an orbit is eliminated, the whole orbit is) */
    /* row branching: fix row a, children over columns b (orbits of GB stabilizer on columns) */
    int best_kind = -1, best_line = -1, best_cnt = 1 << 30; double best_sum = -1; double best_sc = INFINITY;
    static __thread unsigned char domc[NMAX][NMAX];
    if (opt_dombr) dominated_children(nd, ev, domc); else memset(domc, 0, sizeof domc);
    int allow_row = 1, allow_col = 1;
    if (nA > 1 || nB > 1) { if (nB >= nA) allow_col = 0; else allow_row = 0; }
    int colpos[NMAX], rowpos[NMAX];
    for (int b = 0; b < m; b++) colpos[ev->L[b]] = b;
    for (int a = 0; a < m; a++) rowpos[ev->F[a]] = a;
    /* orbit representative flags */
    char repCol[NMAX], repRow[NMAX]; double Ueff[NMAX][NMAX];
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) Ueff[a][b] = w->U[a][b];
    for (int b = 0; b < m; b++) repCol[b] = 1;
    for (int a = 0; a < m; a++) repRow[a] = 1;
    if (allow_row && nB > 1) {
        for (int b = 0; b < m; b++) {
            int j = ev->L[b];
            for (int e = 0; e < nB; e++) { int j2 = GB.perm[idxB[e]][j]; int b2 = colpos[j2];
                if (j2 < j) repCol[b] = 0;
                for (int a = 0; a < m; a++) if (w->U[a][b2] > Ueff[a][b]) Ueff[a][b] = w->U[a][b2]; }
        }
    }
    if (allow_col && nA > 1) {
        for (int a = 0; a < m; a++) {
            int i = ev->F[a];
            for (int e = 0; e < nA; e++) { int i2 = GA.perm[idxA[e]][i]; int a2 = rowpos[i2];
                if (i2 < i) repRow[a] = 0;
                for (int b = 0; b < m; b++) if (w->U[a2][b] > Ueff[a][b]) Ueff[a][b] = w->U[a2][b]; }
        }
    }
    int cntR[NMAX], cntC[NMAX];   /* live children per line, for -latie */
    if (allow_row) for (int a = 0; a < m; a++) {
        int cnt = 0; double s = 0;
        double sc = 0;
        for (int b = 0; b < m; b++) if (repCol[b]) { if (z + Ueff[a][b] <= thr && !domc[a][b]) { cnt++; if (opt_r2p > 0) sc += pow((thr - z - Ueff[a][b]) / (thr - z + 1e-9), opt_r2p); } s += Ueff[a][b]; }
        if (opt_r2p > 0) { if (sc < best_sc || (sc == best_sc && s > best_sum)) { best_sc = sc; best_cnt = cnt; best_sum = s; best_kind = 0; best_line = a; } }
        else if (cnt < best_cnt || (cnt == best_cnt && s > best_sum)) { best_cnt = cnt; best_sum = s; best_kind = 0; best_line = a; }
        cntR[a] = cnt;
    }
    if (allow_col) for (int b = 0; b < m; b++) {
        int cnt = 0; double s = 0;
        double sc = 0;
        for (int a = 0; a < m; a++) if (repRow[a]) { if (z + Ueff[a][b] <= thr && !domc[a][b]) { cnt++; if (opt_r2p > 0) sc += pow((thr - z - Ueff[a][b]) / (thr - z + 1e-9), opt_r2p); } s += Ueff[a][b]; }
        if (opt_r2p > 0) { if (sc < best_sc || (sc == best_sc && s > best_sum)) { best_sc = sc; best_cnt = cnt; best_sum = s; best_kind = 1; best_line = b; } }
        else if (cnt < best_cnt || (cnt == best_cnt && s > best_sum)) { best_cnt = cnt; best_sum = s; best_kind = 1; best_line = b; }
        cntC[b] = cnt;
    }
    double ecc[NMAX];   /* eccentricity of each free row: A row sum, over all rows or (-brdyn) over the free rows only */
    for (int a = 0; a < m; a++) { double t = 0; if (opt_brdyn) for (int c = 0; c < m; c++) t += A[ev->F[a]][ev->F[c]]; else t = g_rowsumA[ev->F[a]]; ecc[a] = t; }
    if (opt_brtie && allow_row && opt_r2p <= 0) {   /* -brtie: among the lines with the fewest live children, prefer the row of A with the largest row sum */
        double be = -INFINITY; int ba = -1;
        for (int a = 0; a < m; a++) { int cnt = 0;
            for (int b = 0; b < m; b++) if (repCol[b] && z + Ueff[a][b] <= thr && !domc[a][b]) cnt++;
            if (cnt == best_cnt && ecc[a] > be) { be = ecc[a]; ba = a; } }
        if (ba >= 0) { best_kind = 0; best_line = ba; }
    }
    if (opt_fixorder) { best_kind = 0; best_line = 0; }   /* -fixorder: lowest free row, for bound comparisons */
    int brforced = 0;   /* -brall: the eccentric row replaces the look-ahead too */
    if (opt_brecc_hi >= 0 && nd->depth >= opt_brecc_lo && nd->depth <= opt_brecc_hi && allow_row && (rule_at(nd->depth)->rule == 2 || opt_brall)) {
        /* -brecc lo hi: branch on the free row whose A row sum is largest (on a grid: the most eccentric cell) */
        double bs = -INFINITY; for (int a = 0; a < m; a++) { double e = ecc[a]; if (e > bs) { bs = e; best_kind = 0; best_line = a; } }
        brforced = opt_brall;
    }
    /* Rules 3 and 4 (Brixius-Anstreicher 2001): prospective bounds on the
     * NBEST rows and NBEST columns with the largest sums of U (Rule 1 order). */
    const rule_t *R = rule_at(nd->depth);
    rule_t Rsz;
    int latie = 0;   /* -latie lo hi: look-ahead restricted to the rule-2 ties */
    if (R->rule == 2 && opt_latie_hi >= 0 && nd->depth >= opt_latie_lo && nd->depth <= opt_latie_hi && opt_r2p <= 0 && !brforced) {
        int nt = 0; if (allow_row) for (int a = 0; a < m; a++) nt += cntR[a] == best_cnt; if (allow_col) for (int b = 0; b < m; b++) nt += cntC[b] == best_cnt;
        if (nt > 1) { latie = 1; Rsz = *R; Rsz.rule = 3; Rsz.nbest = opt_latiek; Rsz.nfw3 = opt_lanfw; R = &Rsz; }
    }
    if (R->rule == 2 && opt_lasize && m >= opt_lasize) {   /* look-ahead decided by the size of the subproblem */
        Rsz = *R; Rsz.rule = 3; Rsz.nbest = opt_lanb; Rsz.nfw3 = opt_lanfw; R = &Rsz;
    }
    double zpro[NMAX]; int have_pro = 0;
    if ((R->rule == 3 || R->rule == 4) && m > 2 && !opt_fixorder && !brforced) {
        typedef struct { int kind, line; double s; } cand_t;
        cand_t rows[NMAX], cols[NMAX], cands[2 * NMAX]; int nr = 0, ncl = 0, nc = 0;
        if (allow_row) for (int a = 0; a < m; a++) { double s = 0; for (int b = 0; b < m; b++) if (repCol[b]) s += Ueff[a][b]; rows[nr++] = (cand_t){0, a, s}; }
        if (allow_col) for (int b = 0; b < m; b++) { double s = 0; for (int a = 0; a < m; a++) if (repRow[a]) s += Ueff[a][b]; cols[ncl++] = (cand_t){1, b, s}; }
        for (int x = 0; x < nr; x++) for (int y = x + 1; y < nr; y++) if (rows[y].s > rows[x].s) { cand_t t = rows[x]; rows[x] = rows[y]; rows[y] = t; }
        for (int x = 0; x < ncl; x++) for (int y = x + 1; y < ncl; y++) if (cols[y].s > cols[x].s) { cand_t t = cols[x]; cols[x] = cols[y]; cols[y] = t; }
        if (latie) {   /* keep only the tied lines, largest sum of U first */
            int k = 0; for (int x = 0; x < nr; x++) if (cntR[rows[x].line] == best_cnt) rows[k++] = rows[x]; nr = k;
            k = 0; for (int x = 0; x < ncl; x++) if (cntC[cols[x].line] == best_cnt) cols[k++] = cols[x]; ncl = k;
        }
        for (int x = 0; x < nr && x < R->nbest; x++) cands[nc++] = rows[x];
        if (opt_laecc && allow_row) {   /* -laecc: the most eccentric free row is always a candidate */
            int be = 0; for (int a = 1; a < m; a++) if (ecc[a] > ecc[be]) be = a;
            int in = 0; for (int x = 0; x < nc; x++) if (cands[x].kind == 0 && cands[x].line == be) in = 1;
            if (!in) cands[nc++] = (cand_t){0, be, 0};
        }
        for (int x = 0; x < ncl && x < R->nbest; x++) cands[nc++] = cols[x];
        /* look-ahead workspace, kept per thread: its eigen-decomposition cache (keyed by
         * the free sets) stays valid for the whole run, and it is allocated once */
        static __thread work_t *w2keep = NULL;
        if (!w2keep) w2keep = calloc(1, sizeof(work_t));
        work_t *w2 = w2keep; w2->nlap = w2->nfw = 0;
        double bestscore = -INFINITY; int bk = best_kind, bl = best_line;
        double zline[NMAX];
        /* child (a,b) appears in the candidate list of row a and of column b:
         * evaluate it once (the evaluation is deterministic) */
        static __thread double memo_z[NMAX][NMAX], memo_v[NMAX][NMAX];
        static __thread unsigned char memo_ok[NMAX][NMAX];
        memset(memo_ok, 0, sizeof memo_ok);
        int capped = opt_lacap && R->rule == 3 && !(opt_lak > 0);
        for (int c = 0; c < nc; c++) {
            double score = 0;
            int ord[NMAX], nrem = 0;
            for (int o = 0; o < m; o++) {
                int a = cands[c].kind == 0 ? cands[c].line : o, b = cands[c].kind == 0 ? o : cands[c].line;
                zline[o] = INFINITY;
                if (cands[c].kind == 0 ? !repCol[b] : !repRow[a]) continue;
                ord[nrem++] = o;
            }
            if (capped) for (int x = 1; x < nrem; x++) {   /* most promising children first: smallest U */
                int o = ord[x], y = x; double key = cands[c].kind == 0 ? Ueff[cands[c].line][o] : Ueff[o][cands[c].line];
                while (y > 0) { int o2 = ord[y - 1]; double k2 = cands[c].kind == 0 ? Ueff[cands[c].line][o2] : Ueff[o2][cands[c].line];
                    if (k2 <= key) break; ord[y] = o2; y--; }
                ord[y] = o;
            }
            int aborted = 0;
            for (int x = 0; x < nrem; x++) {
                int o = ord[x];
                int a = cands[c].kind == 0 ? cands[c].line : o, b = cands[c].kind == 0 ? o : cands[c].line;
                /* capped score: a child above the threshold counts thr, whatever its bound;
                 * the remaining children add at most thr each, so a candidate that cannot
                 * beat the best one is abandoned */
                if (capped && score + (nrem - x) * thr <= bestscore) { aborted = 1; break; }
                double zij, vij = 0;
                if (z + Ueff[a][b] > thr) zij = z + Ueff[a][b];      /* eliminated child: bound from U */
                else if (opt_memo && memo_ok[a][b]) { zij = memo_z[a][b]; vij = memo_v[a][b]; }
                else {
                    node_t chn = *nd; chn.rc[ev->F[a]] = ev->L[b]; chn.cr[ev->L[b]] = ev->F[a]; chn.depth++;
                    if (opt_warm & 1) inherit_x(w2, w->X, m, a, b);
                    if (opt_inh & 2) inherit_st(w2, w->Sp, w->Tp, w->trST, m, a, b);
                    eval_t e2; tl_noext = 1; eval_node(&chn, w2, &e2, R->nfw3, R->nfw3, who); tl_noext = 0;
                    w2->x0_ok = 0;
                    zij = fmax(e2.z, z + Ueff[a][b]);
                    if (R->rule == 4 && e2.m > 1) {
                        for (int x = 0; x < e2.m; x++) { double rs = 0, cs = 0;
                            for (int y = 0; y < e2.m; y++) { rs += w2->U[x][y]; cs += w2->U[y][x]; }
                            if (rs > vij) vij = rs; if (cs > vij) vij = cs; }
                    }
                    memo_z[a][b] = zij; memo_v[a][b] = vij; memo_ok[a][b] = 1;
                }
                zline[o] = zij;
                if (opt_lak > 0 && R->rule == 3) { if (zij <= thr) score -= exp(fmin(opt_lak * (thr - zij), 600)); }   /* -lak k: minimise the sum of exp(k gap) over the live children */
                else score += (R->rule == 4) ? (m - 1) * fmin(zij, 1e15) + vij : fmin(zij, capped ? thr : 1e15);
            }
            if (aborted) continue;
            if (score > bestscore) { bestscore = score; bk = cands[c].kind; bl = cands[c].line; memcpy(zpro, zline, sizeof(double) * m); have_pro = 1; }
        }
        w->nlap += w2->nlap; w->nfw += w2->nfw;
        best_kind = bk; best_line = bl;
    }
    int nch = 0;
    for (int o = 0; o < m; o++) {
        int a = best_kind == 0 ? best_line : o, b = best_kind == 0 ? o : best_line;
        if (best_kind == 0 ? !repCol[b] : !repRow[a]) continue;
        double lb = z + Ueff[a][b];
        if (have_pro && zpro[o] > lb) lb = zpro[o];
        if (lb > thr) continue;
        if (domc[a][b]) { __atomic_fetch_add(&g_dom_pruned, 1, __ATOMIC_RELAXED); continue; }
        ch[nch++] = (child_t){ev->F[a], ev->L[b], lb};
    }
    qsort(ch, nch, sizeof(child_t), cmp_child);
    return nch;
}

/* ---------- serial DFS ---------- */
/* ---------- checkpoint of the serial search ----------
 * The open part of the tree is the node about to be evaluated, the siblings not yet
 * explored at every level of the current path, and the resume items not yet started.
 * They are written in the order the search would visit them, so a resumed run
 * explores the same tree. File: header line, then one node per line:
 * "k i1 j1 ... ik jk" (assignments row -> column). Rewritten every opt_ckpt_s seconds. */
static const char *opt_ckpt = NULL, *opt_resume = NULL;
static double opt_ckpt_s = 600, g_ckpt_last = 0;
static const child_t *g_chs[NMAX + 1]; static int g_nch[NMAX + 1], g_cidx[NMAX + 1];
static node_t g_base; static int g_base_depth = 0;
static node_t *g_rlist = NULL; static int g_rn = 0, g_rnext = 0;
static long g_nodes_before = 0; static double g_elapsed_before = 0;
static void ckpt_node(FILE *f, const node_t *base, int bd, int d, const child_t *last) {
    int k = 0, I[NMAX], J[NMAX];
    for (int i = 0; i < N; i++) if (base->rc[i] >= 0) { I[k] = i; J[k++] = base->rc[i]; }
    for (int e = bd; e < d; e++) { I[k] = g_chs[e][g_cidx[e]].i; J[k++] = g_chs[e][g_cidx[e]].j; }
    if (last) { I[k] = last->i; J[k++] = last->j; }
    fprintf(f, "%d", k); for (int t = 0; t < k; t++) fprintf(f, " %d %d", I[t], J[t]); fputc('\n', f);
}
static void write_ckpt(const node_t *nd) {
    char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.tmp", opt_ckpt);
    FILE *f = fopen(tmp, "w"); if (!f) return;
    fprintf(f, "nugckpt 1 n %d ub %.0f nodes %ld elapsed %.1f\n", N, (double)g_ub, g_nodes_before + (long)g_nodes_all, g_elapsed_before + now() - g_t0);
    ckpt_node(f, nd, 0, 0, NULL);                       /* the node about to be evaluated */
    for (int d = nd->depth - 1; d >= g_base_depth; d--)  /* unexplored siblings, deepest level first */
        for (int c = g_cidx[d] + 1; c < g_nch[d]; c++) ckpt_node(f, &g_base, g_base_depth, d, &g_chs[d][c]);
    for (int r = g_rnext; r < g_rn; r++) ckpt_node(f, &g_rlist[r], 0, 0, NULL);
    fclose(f); rename(tmp, opt_ckpt);
    g_ckpt_last = now();
}
static void read_resume(const char *fn) {
    FILE *f = fopen(fn, "r"); if (!f) { perror(fn); exit(1); }
    int n; double ub; long nodes; double el;
    if (fscanf(f, "nugckpt 1 n %d ub %lf nodes %ld elapsed %lf", &n, &ub, &nodes, &el) != 4 || n != N) { fprintf(stderr, "bad checkpoint\n"); exit(1); }
    if (ub < g_ub) g_ub = ub;
    g_nodes_before = nodes; g_elapsed_before = el;
    int cap = 1024; g_rlist = malloc(sizeof(node_t) * cap);
    int k;
    while (fscanf(f, "%d", &k) == 1) {
        if (g_rn == cap) { cap *= 2; g_rlist = realloc(g_rlist, sizeof(node_t) * cap); }
        node_t *x = &g_rlist[g_rn++];
        for (int i = 0; i < N; i++) x->rc[i] = x->cr[i] = -1, x->fb[i] = 0;
        for (int t = 0; t < k; t++) { int i, j; if (fscanf(f, "%d %d", &i, &j) != 2) exit(5); x->rc[i] = j; x->cr[j] = i; }
        x->depth = k;
    }
    fclose(f);
    fprintf(stderr, "resume: %d open nodes, %ld nodes already done, ub %.0f\n", g_rn, nodes, ub);
}


/* bake an antisymmetric shift lam into the state: Q += lam - lam^T, then exact slave and master
 * reductions; afterwards every entry is >= 0 again and R = the Lagrangian value at lam */
static void rlt_bake(rlt_t *r, const double *lam) {
    int m = r->m, n1 = m - 1; double c[NMAX][NMAX];
    size_t M4 = (size_t)m * m * m * m; double *T = malloc(sizeof(double) * M4);
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) for (int cc = 0; cc < m; cc++) for (int d = 0; d < m; d++) {
        size_t k = QX(m, a, b, cc, d); T[k] = (cc != a && d != b) ? r->Q[k] + lam[k] - lam[QX(m, cc, d, a, b)] : 0; }
    memcpy(r->Q, T, sizeof(double) * M4); free(T);
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        int i = 0;
        for (int cc = 0; cc < m; cc++) if (cc != a) { int j = 0; for (int d = 0; d < m; d++) if (d != b) c[i][j++] = r->Q[QX(m, a, b, cc, d)]; i++; }
        double val = lap_reduce(n1, c, r->vs[a * m + b], 0);
        i = 0;
        for (int cc = 0; cc < m; cc++) if (cc != a) { int j = 0; double *q = r->Q + QX(m, a, b, cc, 0); for (int d = 0; d < m; d++) if (d != b) q[d] = c[i][j++]; i++; }
        r->L[a][b] += val;
    }
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) c[a][b] = r->L[a][b];
    double vm[NMAX]; double val = lap_reduce(m, c, vm, 0);
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) r->L[a][b] = c[a][b];
    r->R += val;
}
static int opt_sg = 0; static double opt_sgth = 1.4, opt_defl = 0; static long g_sg_it = 0;
/* subgradient on the twin splits from the current state, Polyak step towards thr + 1; bakes the best point.
 * opt_defl > 0: deflected direction d = g + beta d_prev (Camerini-Fratta-Maffioli, beta from the obtuse rule) */
static double rlt_sg(rlt_t *r, int iters, double thr) {
    int m = r->m; size_t M4 = (size_t)m * m * m * m;
    double *lam = calloc(M4, sizeof(double)), *bl = calloc(M4, sizeof(double));
    double *g = opt_defl > 0 ? calloc(M4, sizeof(double)) : NULL, *dp = opt_defl > 0 ? calloc(M4, sizeof(double)) : NULL;
    static __thread int ys[NMAX * NMAX][NMAX]; int xm[NMAX];
    double best = r->R; int have = 0, hasd = 0;
    for (int it = 0; it < iters; it++) {
        double z = rlt_lag(r, lam, xm, ys); g_sg_it++;
        if (z > best + 1e-9) { best = z; memcpy(bl, lam, sizeof(double) * M4); have = 1; }
        if (best > thr) break;
        int cnt = 0;
        for (int a = 0; a < m; a++) { int b = xm[a]; for (int cc = 0; cc < m; cc++) if (cc != a) { int d = ys[a * m + b][cc];
            if (!(xm[cc] == d && ys[cc * m + d][a] == b)) cnt++; } }
        if (!cnt) break;
        if (!g) {
            double step = opt_sgth * (thr + 1 - z) / (2.0 * cnt);
            for (int a = 0; a < m; a++) { int b = xm[a]; for (int cc = 0; cc < m; cc++) if (cc != a) { int d = ys[a * m + b][cc];
                if (!(xm[cc] == d && ys[cc * m + d][a] == b)) lam[QX(m, a, b, cc, d)] += step; } }
            continue;
        }
        memset(g, 0, sizeof(double) * M4);
        for (int a = 0; a < m; a++) { int b = xm[a]; for (int cc = 0; cc < m; cc++) if (cc != a) { int d = ys[a * m + b][cc];
            if (!(xm[cc] == d && ys[cc * m + d][a] == b)) { g[QX(m, a, b, cc, d)] += 1; g[QX(m, cc, d, a, b)] -= 1; } } }
        double beta = 0;
        if (hasd) { double gd = 0, dd = 0; for (size_t k = 0; k < M4; k++) { gd += g[k] * dp[k]; dd += dp[k] * dp[k]; }
            if (gd < 0 && dd > 0) beta = -opt_defl * gd / dd; }
        double nn = 0; for (size_t k = 0; k < M4; k++) { dp[k] = g[k] + beta * dp[k]; nn += dp[k] * dp[k]; }
        hasd = 1;
        double step = opt_sgth * (thr + 1 - z) / nn;
        for (size_t k = 0; k < M4; k++) lam[k] += 0.5 * step * dp[k];
    }
    if (have) rlt_bake(r, bl);
    free(lam); free(bl); free(g); free(dp);
    return r->R;
}

static int opt_hyb = 0, opt_hybm = 6, opt_hybpass = 20; static double opt_hybd = 40, g_hyb_t = 0; static long g_hyb_calls = 0, g_hyb_fath = 0;
/* -hybi: the RLT1 state of a node that the ascent did not prune is handed down to its children
 * (exact child of a dual state, see rbb_dfs), so a child starts from its parent's duals instead of cold.
 * Serial search only. Counters: inherited states, children pruned by the inherited bound alone,
 * inherited children pruned after work, passes and subgradient iterations spent on inherited states. */
static int opt_hybi = 0, opt_hybisg = 50, opt_hybipass = 3, opt_hybidep = -1; static double g_hi_tinh = 0; static double opt_hybieps = 0.5;
static long g_hi_inh = 0, g_hi_free = 0, g_hi_work = 0, g_hi_pass = 0, g_hi_sg = 0, g_hi_cold = 0, g_hi_coldf = 0; static double g_hi_t = 0, g_hi_gain = 0;
static rlt_t *g_hs[NMAX + 2]; static int g_hs_ok[NMAX + 2], g_hsF[NMAX + 2][NMAX], g_hsL[NMAX + 2][NMAX];
static void rlt_child(const rlt_t *r, rlt_t *ch, int ab, int b) {
    int m = r->m, n1 = m - 1, ri[NMAX], ci[NMAX], k = 0;
    for (int a = 0; a < m; a++) if (a != ab) ri[k++] = a;
    k = 0; for (int d = 0; d < m; d++) if (d != b) ci[k++] = d;
    ch->m = n1; ch->R = r->R + r->L[ab][b]; ch->warm = 0;
    for (int i = 0; i < n1; i++) for (int j = 0; j < n1; j++) { int c = ri[i], d = ci[j];
        ch->L[i][j] = r->L[c][d] + r->Q[QX(m, ab, b, c, d)] + r->Q[QX(m, c, d, ab, b)];
        for (int i2 = 0; i2 < n1; i2++) for (int j2 = 0; j2 < n1; j2++) ch->Q[QX(n1, i, j, i2, j2)] = r->Q[QX(m, c, d, ri[i2], ci[j2])]; }
}
/* ---------- B&B on the RLT1 dual state alone (experiment -rltbb) ----------
 * The child a->b of a DP state (R, L, Q >= 0) is exact and inherits everything:
 * R' = R + L_ab, L'_cd = L_cd + Q_abcd + Q_cdab, Q' = Q restricted. All entries stay >= 0,
 * so R' >= R: the bound only grows going down, and R + L_ab already bounds the child. */
static long g_rbb_nodes = 0, g_rbb_pass = 0; static int opt_rbb_pass = 3; static double opt_rbb_eps = 0.5;
static int g_rbb_perm[NMAX];
static void rbb_dfs(rlt_t *r, const int *F, const int *Lc, int depth) {
    int m = r->m; g_rbb_nodes++;
    double thr = prune_thr(g_ub);
    if (m == 1) { double c = r->R + r->L[0][0];
        if (c < g_ub) { int p[NMAX]; memcpy(p, g_rbb_perm, sizeof p); p[F[0]] = Lc[0]; report_solution(p, c, 0); }
        return; }
    for (int k = 0; k < opt_rbb_pass; k++) { double z0 = r->R; rlt_pass(r); g_rbb_pass++; if (r->R > thr || r->R - z0 < opt_rbb_eps) break; }
    if (opt_sg && r->R <= thr && m > 3) rlt_sg(r, opt_sg, thr);
    if (r->R > thr) return;
    /* branch on the facility whose row of reduced master costs has the largest sum (most children cut) */
    int ab = 0; double bs = -1;
    for (int a = 0; a < m; a++) { double t = 0; for (int b = 0; b < m; b++) t += r->L[a][b] > thr - r->R ? 1e6 : r->L[a][b]; if (t > bs) { bs = t; ab = a; } }
    int ord[NMAX], no = 0; for (int b = 0; b < m; b++) ord[no++] = b;
    for (int i = 1; i < no; i++) { int x = ord[i], j = i; while (j > 0 && r->L[ab][ord[j - 1]] > r->L[ab][x]) { ord[j] = ord[j - 1]; j--; } ord[j] = x; }
    rlt_t *ch = calloc(1, sizeof(rlt_t)); ch->Q = malloc(sizeof(double) * (size_t)m * m * m * m);
    for (int t = 0; t < no; t++) {
        int b = ord[t]; if (r->R + r->L[ab][b] > prune_thr(g_ub)) continue;
        int n1 = m - 1, F2[NMAX], L2[NMAX], ri[NMAX], ci[NMAX], k = 0;
        for (int a = 0; a < m; a++) if (a != ab) { F2[k] = F[a]; ri[k++] = a; }
        k = 0; for (int d = 0; d < m; d++) if (d != b) { L2[k] = Lc[d]; ci[k++] = d; }
        ch->m = n1; ch->R = r->R + r->L[ab][b]; ch->warm = 0;
        for (int i = 0; i < n1; i++) for (int j = 0; j < n1; j++) { int c = ri[i], d = ci[j];
            ch->L[i][j] = r->L[c][d] + r->Q[QX(m, ab, b, c, d)] + r->Q[QX(m, c, d, ab, b)];
            for (int i2 = 0; i2 < n1; i2++) for (int j2 = 0; j2 < n1; j2++) ch->Q[QX(n1, i, j, i2, j2)] = r->Q[QX(m, c, d, ri[i2], ci[j2])]; }
        g_rbb_perm[F[ab]] = Lc[b];
        rbb_dfs(ch, F2, L2, depth + 1);
        g_rbb_perm[F[ab]] = -1;
    }
    free(ch->Q); free(ch);
}


static void dfs(node_t *nd, work_t *w, stats_t *st, int who) {
    eval_t ev;
    if (opt_ckpt && now() - g_ckpt_last > opt_ckpt_s) write_ckpt(nd);
    double t = now(); long fw0 = w->nfw;
    unsigned long long pcn = g_prof ? __rdtsc() : 0;
    { const rule_t *R = rule_at(nd->depth); eval_node(nd, w, &ev, R->nfw1, R->nfw2, who); }
    PROF_T(0, (ev.m <= 1 || ev.z > prune_thr(g_ub)) ? 9 : 10, pcn);
    st->t_bound += now() - t;
    if (opt_domtab && ev.m >= opt_dtmlo && ev.m <= opt_dtmhi && isfinite(ev.z)) domtab_store(w, ev.m, ev.K, ev.z);
    if (ev.m > 1) { int f = ev.z > prune_thr(g_ub); g_fwstat[f][0]++; g_fwstat[f][1] += w->nfw - fw0;
        if (!f && g_gapstat) { double thr = prune_thr(g_ub), g = thr - ev.z; int hi = w->flast >= thr; int b = g < 1 ? 0 : g < 2 ? 1 : g < 5 ? 2 : g < 10 ? 3 : g < 20 ? 4 : g < 50 ? 5 : 6;
            g_gaph[hi][b]++; } }
    st->nodes++; st->nodes_depth[nd->depth]++;
    g_nodes_all++; g_depth_all[nd->depth]++; if ((g_nodes_all & 255) == 0) write_progress(0);
    if (ev.m <= 1 || ev.z > prune_thr(g_ub)) { st->fathom_at_bound++; return; }
    if (opt_hybi) {   /* -hybi: inherited or cold RLT1 state; prune if its bound passes the threshold */
        int dd = nd->depth; double thr = prune_thr(g_ub), th0 = now(); g_hs_ok[dd] = 0;
        if (!g_hs[dd]) { g_hs[dd] = calloc(1, sizeof(rlt_t)); g_hs[dd]->Q = malloc(sizeof(double) * NMAX * NMAX * NMAX * NMAX); }
        rlt_t *hr = g_hs[dd]; int have = 0;
        if (dd > 0 && g_hs_ok[dd - 1] && ev.m >= 2) {   /* child of a node with a state: find the fixed pair */
            int fi = -1, fj = -1, mp = ev.m + 1;
            for (int a = 0; a < mp; a++) if (nd->rc[g_hsF[dd - 1][a]] >= 0) { fi = a; break; }
            for (int b = 0; b < mp; b++) if (nd->cr[g_hsL[dd - 1][b]] >= 0) { fj = b; break; }
            if (fi >= 0 && fj >= 0) { rlt_child(g_hs[dd - 1], hr, fi, fj); have = 1; g_hi_inh++;
                if (hr->R > thr) { g_hi_free++; g_hi_t += now() - th0; st->fathom_at_bound++; return; }
                double r0 = hr->R;
                for (int k = 0; k < opt_hybipass && hr->R <= thr; k++) { double z0 = hr->R; rlt_pass(hr); g_hi_pass++; if (hr->R - z0 < opt_hybieps) break; }
                if (hr->R <= thr && opt_hybisg) { long i0 = g_sg_it; rlt_sg(hr, opt_hybisg, thr); g_hi_sg += g_sg_it - i0; }
                g_hi_gain += hr->R - r0;
                g_hi_tinh += now() - th0;
                if (hr->R > thr) { g_hi_work++; g_hi_t += now() - th0; st->fathom_at_bound++; return; } } }
        if (!have && ev.m >= opt_hybm && (opt_hybidep >= 0 ? dd == opt_hybidep : ev.z > thr - opt_hybd)) {   /* cold start: in the band, as -hyb, or at depth -hybidep */
            rlt_init(hr, ev.m, w->Ar, w->Br, w->Cr, ev.K); have = 1; g_hi_cold++;
            for (int k = 0; k < opt_hybpass && hr->R <= thr; k++) rlt_pass(hr);
            if (hr->R <= thr && opt_sg) rlt_sg(hr, opt_sg, thr);
            if (hr->R > thr) { g_hi_coldf++; g_hi_t += now() - th0; st->fathom_at_bound++; return; } }
        if (have && ev.m > 2) { g_hs_ok[dd] = 1; memcpy(g_hsF[dd], ev.F, sizeof(int) * ev.m); memcpy(g_hsL[dd], ev.L, sizeof(int) * ev.m); }
        g_hi_t += now() - th0;
    }
    if (opt_hyb && ev.m >= opt_hybm && ev.z > prune_thr(g_ub) - opt_hybd) {   /* -hyb: RLT1 ascent on the nodes QPB leaves near the threshold */
        static __thread rlt_t *hr = NULL; if (!hr) hr = calloc(1, sizeof(rlt_t));
        double th0 = now(); g_hyb_calls++;
        rlt_init(hr, ev.m, w->Ar, w->Br, w->Cr, ev.K);
        double thr = prune_thr(g_ub);
        for (int k = 0; k < opt_hybpass && hr->R <= thr; k++) rlt_pass(hr);
        if (hr->R <= thr && opt_sg) rlt_sg(hr, opt_sg, thr);
        g_hyb_t += now() - th0;
        if (hr->R > thr) { g_hyb_fath++; st->fathom_at_bound++; return; }
    }
    if (opt_domlb && domlb(nd, &ev, w)) { g_domlb_pruned++; return; }
    if (opt_domcov && domcov(nd, &ev, w)) { g_domcov_pruned++; return; }
    unsigned fbsave[NMAX];
    if (opt_fb) { memcpy(fbsave, nd->fb, sizeof fbsave); add_fb(nd, &ev, w); }
    child_t ch[NMAX];
    unsigned long long pcb = g_prof ? __rdtsc() : 0;
    int nch = branch(nd, w, &ev, ch, who);
    PROF_T(0, 8, pcb);
    int d0 = nd->depth;
    double (*Xp)[NMAX] = NULL, (*Sp)[NMAX] = NULL, (*Tp)[NMAX] = NULL; double trp = w->trST; int posF[NMAX], posL[NMAX];
    if ((opt_warm & 2) || (opt_inh & 1)) for (int a = 0; a < ev.m; a++) { posF[ev.F[a]] = a; posL[ev.L[a]] = a; }
    if (opt_warm & 2) {   /* children start from the parent's final point, restricted (inherit_x) */
        Xp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Xp, w->X, sizeof(double) * NMAX * NMAX);
    }
    if (opt_inh & 1) {   /* children start from the parent's S, T, restricted (inherit_st) */
        Sp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Sp, w->Sp, sizeof(double) * NMAX * NMAX);
        Tp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Tp, w->Tp, sizeof(double) * NMAX * NMAX);
    }
    for (int c = 0; c < nch; c++) {
        if (ch[c].lb > prune_thr(g_ub)) continue;
        if (Xp) { unsigned long long pci = g_prof ? __rdtsc() : 0; inherit_x(w, (const double (*)[NMAX])Xp, ev.m, posF[ch[c].i], posL[ch[c].j]); PROF_T(0, 11, pci); }
        if (Sp) inherit_st(w, (const double (*)[NMAX])Sp, (const double (*)[NMAX])Tp, trp, ev.m, posF[ch[c].i], posL[ch[c].j]);
        g_chs[d0] = ch; g_nch[d0] = nch; g_cidx[d0] = c;
        g_pos[d0] = c + 1; g_npos[d0] = nch; g_nextlb[d0] = c + 1 < nch ? ch[c + 1].lb : INFINITY; g_curdepth = d0 + 1;
        nd->rc[ch[c].i] = ch[c].j; nd->cr[ch[c].j] = ch[c].i; nd->depth++;
        dfs(nd, w, st, who);
        nd->depth--; nd->rc[ch[c].i] = -1; nd->cr[ch[c].j] = -1;
    }
    free(Xp); free(Sp); free(Tp);
    if (opt_fb) memcpy(nd->fb, fbsave, sizeof fbsave);
    g_curdepth = d0;
}

/* ---------- Knuth estimator ---------- */
/* -top L: the levels above L are enumerated exactly (every node evaluated, as in the
 * search), and the dives start from each surviving node at depth L. The estimate is
 * exact count above L plus, for each frontier node, the mean of its own dives: the
 * randomness is confined below L, where single subtrees are smaller. */
static int opt_top = 0, opt_coldest = 0; static const char *g_tlog = NULL;
/* depth profile of the estimate: nodes, time and nodes fathomed after evaluation, by depth */
static double g_pN[NMAX + 2], g_pT[NMAX + 2], g_pF[NMAX + 2], g_wsc = 1;
typedef struct { node_t *v; int n, cap; double cnt, t; } frontier_t;
static void top_enum(node_t *nd, work_t *w, frontier_t *F) {
    eval_t ev; double t = now();
    { const rule_t *R = rule_at(nd->depth); eval_node(nd, w, &ev, R->nfw1, R->nfw2, 0); }
    if (ev.m <= 1 || ev.z > prune_thr(g_ub) || (opt_domlb && domlb(nd, &ev, w)) || (opt_domcov && domcov(nd, &ev, w))) { F->cnt++; F->t += now() - t;
        g_pN[nd->depth]++; g_pT[nd->depth] += now() - t; g_pF[nd->depth]++; return; }
    unsigned fbsave[NMAX]; memcpy(fbsave, nd->fb, sizeof fbsave);
    if (opt_fb) add_fb(nd, &ev, w);
    child_t ch[NMAX]; int nch = branch(nd, w, &ev, ch, 0);
    F->cnt++; F->t += now() - t; g_pN[nd->depth]++; g_pT[nd->depth] += now() - t;
    double (*Xp)[NMAX] = NULL, (*Sp)[NMAX] = NULL, (*Tp)[NMAX] = NULL; double trp = w->trST; int posF[NMAX], posL[NMAX];
    if (!opt_coldest) {
        for (int a = 0; a < ev.m; a++) { posF[ev.F[a]] = a; posL[ev.L[a]] = a; }
        if (opt_warm & 2) { Xp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Xp, w->X, sizeof(double) * NMAX * NMAX); }
        if (opt_inh & 1) { Sp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Sp, w->Sp, sizeof(double) * NMAX * NMAX);
            Tp = malloc(sizeof(double) * NMAX * NMAX); memcpy(Tp, w->Tp, sizeof(double) * NMAX * NMAX); }
    }
    for (int c = 0; c < nch; c++) {
        if (ch[c].lb > prune_thr(g_ub)) continue;
        if (Xp) inherit_x(w, (const double (*)[NMAX])Xp, ev.m, posF[ch[c].i], posL[ch[c].j]);
        if (Sp) inherit_st(w, (const double (*)[NMAX])Sp, (const double (*)[NMAX])Tp, trp, ev.m, posF[ch[c].i], posL[ch[c].j]);
        nd->rc[ch[c].i] = ch[c].j; nd->cr[ch[c].j] = ch[c].i; nd->depth++;
        if (nd->depth >= opt_top) {
            if (F->n == F->cap) { F->cap = F->cap ? 2 * F->cap : 256; F->v = realloc(F->v, sizeof(node_t) * F->cap); }
            F->v[F->n++] = *nd;
        } else top_enum(nd, w, F);
        nd->depth--; nd->rc[ch[c].i] = -1; nd->cr[ch[c].j] = -1;
    }
    w->x0_ok = 0; w->inh_ok = 0;   /* a frontier node stored for the dives is evaluated later from its own start */
    free(Xp); free(Sp); free(Tp);
    memcpy(nd->fb, fbsave, sizeof fbsave);
}
/* one dive from node start; returns the weighted node count and time of the dive */
static void dive_from(const node_t *start, work_t *w, rng_t *r, double *estN, double *estT, long *totnodes) {
    node_t nd = *start; double weight = 1, zroot = NAN; *estN = 0; *estT = 0;
    for (;;) {
        eval_t ev; double t = now();
        { const rule_t *R = rule_at(nd.depth); eval_node(&nd, w, &ev, R->nfw1, R->nfw2, 0); }
        child_t ch[NMAX]; int nch = 0;
        int leaf = (ev.m <= 1 || ev.z > prune_thr(g_ub));
        if (!leaf && opt_domlb && domlb(&nd, &ev, w)) leaf = 1;
        if (!leaf && opt_domcov && domcov(&nd, &ev, w)) leaf = 1;
        if (!leaf) { if (opt_fb) add_fb(&nd, &ev, w); nch = branch(&nd, w, &ev, ch, 0); }
        double dt = now() - t; (*totnodes)++;
        if (isnan(zroot)) zroot = ev.z;
        *estN += weight; *estT += weight * dt;
        g_pN[nd.depth] += g_wsc * weight; g_pT[nd.depth] += g_wsc * weight * dt;
        if (leaf) g_pF[nd.depth] += g_wsc * weight;
        if (leaf || nch == 0) break;
        double pr[NMAX], tot = 0;
        for (int c = 0; c < nch; c++) {
            double gap = (g_ub - ch[c].lb) / (g_ub - zroot + 1e-9); if (gap < 1e-6) gap = 1e-6;
            pr[c] = opt_q > 0 ? pow(gap, opt_q) : 1.0; tot += pr[c];
        }
        double x = rng_unif(r) * tot; int c = 0;
        while (c < nch - 1 && x > pr[c]) { x -= pr[c]; c++; }
        weight *= tot / pr[c];
        if (!opt_coldest) {   /* the chosen child starts as in the search: parent's point and, with -inh, parent's S, T */
            int pf = 0, pl = 0; for (int a = 0; a < ev.m; a++) { if (ev.F[a] == ch[c].i) pf = a; if (ev.L[a] == ch[c].j) pl = a; }
            if (opt_warm & 2) inherit_x(w, w->X, ev.m, pf, pl);
            if (opt_inh & 1) inherit_st(w, w->Sp, w->Tp, w->trST, ev.m, pf, pl);
        }
        nd.rc[ch[c].i] = ch[c].j; nd.cr[ch[c].j] = ch[c].i; nd.depth++;
    }
}
static void estimate_top(int dives, unsigned long long seed) {
    work_t *w = calloc(1, sizeof(work_t));
    rng_t r = {seed * 2654435761ULL + 1};
    double t0 = now(); long totnodes = 0;
    node_t root; for (int i = 0; i < N; i++) root.rc[i] = root.cr[i] = -1, root.fb[i] = 0; root.depth = 0;
    frontier_t F = {0};
    top_enum(&root, w, &F);
    double t1 = now() - t0;
    int k = F.n ? (dives + F.n - 1) / F.n : 0;        /* dives per frontier node */
    double sumN = F.cnt, sumT = F.t, var = 0;
    /* restart-safe: each frontier node has its own random stream, and its result is appended to
     * -tlog; on a rerun the nodes already in the log are read back instead of recomputed */
    double *lgN = calloc(F.n, sizeof(double)), *lgT = calloc(F.n, sizeof(double)), *lgV = calloc(F.n, sizeof(double)); char *have = calloc(F.n, 1);
    FILE *tl = NULL;
    if (g_tlog) {
        FILE *g = fopen(g_tlog, "r"); int fi; double a, b, c;
        if (g) { while (fscanf(g, "%d %lf %lf %lf", &fi, &a, &b, &c) == 4) if (fi >= 0 && fi < F.n) { lgN[fi] = a; lgT[fi] = b; lgV[fi] = c; have[fi] = 1; } fclose(g); }
        tl = fopen(g_tlog, "a");
    }
    for (int f = 0; f < F.n; f++) {
        double s = 0, s2 = 0, st = 0;
        if (have[f]) { sumN += lgN[f]; sumT += lgT[f]; var += lgV[f]; continue; }
        rng_t rf = {(seed * 2654435761ULL + 1) ^ ((unsigned long long)(f + 1) * 0x9E3779B97F4A7C15ULL)}; (void)r;
        g_wsc = 1.0 / k;
        for (int d = 0; d < k; d++) { double eN, eT; dive_from(&F.v[f], w, &rf, &eN, &eT, &totnodes); s += eN; s2 += eN * eN; st += eT; }
        double m = s / k; sumN += m; sumT += st / k;
        double vf = k > 1 ? fmax(0, s2 / k - m * m) / (k - 1) : 0; var += vf;
        if (tl) { fprintf(tl, "%d %.6e %.6e %.6e\n", f, m, st / k, vf); fflush(tl); }
        if (g_prog && (now() - g_prog_last >= 10 || f + 1 == F.n)) {
            g_prog_last = now();
            char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.tmp", g_prog);
            FILE *pf = fopen(tmp, "w");
            if (pf) { fprintf(pf, "{\"elapsed_s\":%.1f,\"top\":%d,\"frontier\":%d,\"frontier_done\":%d,\"top_nodes\":%.0f,\"nodes\":%ld,\"final\":%d}\n",
                              now() - t0, opt_top, F.n, f + 1, F.cnt, totnodes, f + 1 == F.n); fclose(pf); rename(tmp, g_prog); }
        }
    }
    printf("{\"mode\":\"estimate_top\",\"n\":%d,\"top\":%d,\"top_nodes\":%.0f,\"top_s\":%.1f,\"frontier\":%d,\"dives_per_node\":%d,\"q\":%.2f,\"ub\":%.0f,"
           "\"est_nodes\":%.4e,\"est_nodes_stderr\":%.3e,\"est_time_s\":%.4e,\"nodes_evaluated\":%ld,\"wall_s\":%.1f}\n",
           N, opt_top, F.cnt, t1, F.n, k, opt_q, (double)g_ub, sumN, sqrt(var), sumT, totnodes + (long)F.cnt, now() - t0);
    {   int last = 0; for (int d = 0; d <= N; d++) if (g_pN[d] > 0) last = d;
        printf("{\"profile\":[");
        for (int d = 0; d <= last; d++) printf("%s[%d,%.4e,%.4e,%.4e]", d ? "," : "", d, g_pN[d], g_pT[d], g_pF[d]);
        printf("],\"profile_partial\":%d}\n", (int)(g_tlog != NULL)); }
    if (tl) fclose(tl); free(lgN); free(lgT); free(lgV); free(have);
    free(F.v); free(w);
}
static void estimate(int dives, unsigned long long seed) {
    work_t *w = calloc(1, sizeof(work_t));
    rng_t r = {seed * 2654435761ULL + 1};
    double sumN = 0, sumT = 0, sumN2 = 0; double t0 = now();
    long totnodes = 0;
    for (int d = 0; d < dives; d++) {
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        double weight = 1, estN = 0, estT = 0;
        double zroot = 0;
        for (;;) {
            eval_t ev; double t = now();
            { const rule_t *R = rule_at(nd.depth); eval_node(&nd, w, &ev, R->nfw1, R->nfw2, 0); }
            child_t ch[NMAX]; int nch = 0;
            int leaf = (ev.m <= 1 || ev.z > prune_thr(g_ub));
            if (!leaf) nch = branch(&nd, w, &ev, ch, 0);
            double dt = now() - t; totnodes++;
            if (nd.depth == 0) zroot = ev.z;
            estN += weight; estT += weight * dt;
            if (leaf || nch == 0) break;
            /* choose child: uniform, or proportional to (relative gap)^q */
            double pr[NMAX], tot = 0;
            for (int c = 0; c < nch; c++) {
                double gap = (g_ub - ch[c].lb) / (g_ub - zroot + 1e-9); if (gap < 1e-6) gap = 1e-6;
                pr[c] = opt_q > 0 ? pow(gap, opt_q) : 1.0; tot += pr[c];
            }
            double x = rng_unif(&r) * tot; int c = 0;
            while (c < nch - 1 && x > pr[c]) { x -= pr[c]; c++; }
            weight *= tot / pr[c];
            nd.rc[ch[c].i] = ch[c].j; nd.cr[ch[c].j] = ch[c].i; nd.depth++;
        }
        sumN += estN; sumN2 += estN * estN; sumT += estT;
        if (g_prog && (now() - g_prog_last >= 10 || d + 1 == dives)) {
            g_prog_last = now();
            char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.tmp", g_prog);
            FILE *f = fopen(tmp, "w");
            if (f) {
                fprintf(f, "{\"elapsed_s\":%.1f,\"dives_done\":%d,\"dives_total\":%d,\"nodes\":%ld,\"est_nodes\":%.4e,\"est_time_s\":%.4e,\"final\":%d}\n",
                        now() - t0, d + 1, dives, totnodes, sumN / (d + 1), sumT / (d + 1), d + 1 == dives);
                fclose(f); rename(tmp, g_prog);
            }
        }
        if (opt_verbose && (d + 1) % (dives / 10 > 0 ? dives / 10 : 1) == 0)
            fprintf(stderr, "dive %d: mean nodes %.3e, mean time %.3e s\n", d + 1, sumN / (d + 1), sumT / (d + 1));
    }
    double mN = sumN / dives, sd = sqrt(fmax(0, sumN2 / dives - mN * mN) / dives);
    printf("{\"mode\":\"estimate\",\"n\":%d,\"dives\":%d,\"q\":%.2f,\"nfw\":%d,\"ub\":%.0f,"
           "\"est_nodes\":%.4e,\"est_nodes_stderr\":%.3e,\"est_time_s\":%.4e,\"sec_per_node\":%.3e,\"nodes_evaluated\":%ld,\"wall_s\":%.1f}\n",
           N, dives, opt_q, opt_nfw, (double)g_ub, mN, sd, sumT / dives, (sumT / dives) / mN, totnodes, now() - t0);
    free(w);
}

/* ---------- parallel: shared pool of open nodes, workers with time slices ----------
 * Pool holds nodes; a worker takes the deepest node, explores it depth-first
 * with an explicit stack for up to tmax seconds; if time is up and the pool is
 * short of work, it returns its shallowest open nodes to the pool
 * ("finish-up" style: give back big pieces, keep the small ones). The incumbent
 * is shared: any improvement is seen by all workers at their next node. */
typedef struct { node_t nd; double lb; } pitem_t;
static pitem_t *pool; static int pool_n = 0, pool_cap = 0, pool_busy = 0;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_cond = PTHREAD_COND_INITIALIZER;
static double opt_tmax = 5.0;
static int opt_threads = 4;
static long tasks_done = 0, nodes_returned = 0;

static void pool_push(const node_t *nd, double lb) {
    if (pool_n == pool_cap) { pool_cap = pool_cap ? 2 * pool_cap : 1024; pool = realloc(pool, pool_cap * sizeof(pitem_t)); }
    pool[pool_n].nd = *nd; pool[pool_n].lb = lb; pool_n++;
}

/* parent's final Frank-Wolfe point, shared by the children of one branching on a
 * worker's stack (children start from it, see inherit_x); freed by the last child.
 * Nodes handed back to the pool drop it and start from the centre. */
typedef struct { int ref, m, posF[NMAX], posL[NMAX]; double X[NMAX][NMAX]; } xpar_t;
typedef struct { node_t nd; double lb; xpar_t *xp; int ci, cj; } sitem_t;
static inline void xpar_drop(xpar_t **xp) { if (*xp && --(*xp)->ref == 0) free(*xp); *xp = NULL; }
typedef struct { int id; stats_t st; work_t *w; sitem_t *stack; int sp; pthread_mutex_t mtx; } wk_t;
static wk_t *g_wk = NULL;

/* checkpoint of the parallel search: pool + every worker's stack. Workers change their
 * stack under their own mutex (or under pool_lock when taking a task), so holding all
 * worker mutexes, then pool_lock, gives a consistent snapshot. Order is not preserved. */
static void write_ckpt_par(void) {
    for (int k = 0; k < opt_threads; k++) pthread_mutex_lock(&g_wk[k].mtx);
    pthread_mutex_lock(&pool_lock);
    char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.tmp", opt_ckpt);
    FILE *f = fopen(tmp, "w");
    if (f) {
        fprintf(f, "nugckpt 1 n %d ub %.0f nodes %ld elapsed %.1f\n", N, (double)g_ub, g_nodes_before + (long)g_nodes_all, g_elapsed_before + now() - g_t0);
        for (int k = 0; k < opt_threads; k++) for (int t = 0; t < g_wk[k].sp; t++) ckpt_node(f, &g_wk[k].stack[t].nd, 0, 0, NULL);
        for (int t = 0; t < pool_n; t++) ckpt_node(f, &pool[t].nd, 0, 0, NULL);
        fclose(f); rename(tmp, opt_ckpt);
    }
    g_ckpt_last = now();
    pthread_mutex_unlock(&pool_lock);
    for (int k = 0; k < opt_threads; k++) pthread_mutex_unlock(&g_wk[k].mtx);
}

static void *worker(void *arg) {
    wk_t *me = arg;
    sitem_t *stack = me->stack;
    for (;;) {
        if (opt_ckpt && me->id == 0 && now() - g_ckpt_last > opt_ckpt_s) write_ckpt_par();
        pthread_mutex_lock(&pool_lock);
        while (pool_n == 0 && pool_busy > 0) pthread_cond_wait(&pool_cond, &pool_lock);
        if (pool_n == 0 && pool_busy == 0) { pthread_cond_broadcast(&pool_cond); pthread_mutex_unlock(&pool_lock); break; }
        /* take deepest node (ties: best bound) */
        int bi = 0;
        for (int k = 1; k < pool_n; k++)
            if (pool[k].nd.depth > pool[bi].nd.depth || (pool[k].nd.depth == pool[bi].nd.depth && pool[k].lb < pool[bi].lb)) bi = k;
        sitem_t it = {pool[bi].nd, pool[bi].lb, NULL, 0, 0}; pool[bi] = pool[--pool_n];
        pool_busy++;
        int sp = 0; stack[sp++] = it; me->sp = sp;   /* under pool_lock: never lost from a snapshot */
        pthread_mutex_unlock(&pool_lock);

        double tstart = now();
        long nodes0 = me->st.nodes;
        while (sp > 0) {
            if (opt_ckpt && me->id == 0 && now() - g_ckpt_last > opt_ckpt_s) write_ckpt_par();
            pthread_mutex_lock(&me->mtx);
            if (now() - tstart > opt_tmax) {
                pthread_mutex_lock(&pool_lock);
                int need = pool_n < 4 * opt_threads;
                if (need) {
                    /* give back the shallowest half of the stack (bottom entries) */
                    int give = sp > 1 ? sp / 2 : 0;
                    for (int k = 0; k < give; k++) { xpar_drop(&stack[k].xp); pool_push(&stack[k].nd, stack[k].lb); nodes_returned++; }
                    memmove(stack, stack + give, sizeof(sitem_t) * (sp - give)); sp -= give;
                    pthread_cond_broadcast(&pool_cond);
                }
                pthread_mutex_unlock(&pool_lock);
                tstart = now();
            }
            sitem_t cur = stack[--sp];
            if (cur.lb > prune_thr(g_ub)) { xpar_drop(&cur.xp); me->sp = sp; pthread_mutex_unlock(&me->mtx); continue; }
            if (cur.xp) { inherit_x(me->w, (const double (*)[NMAX])cur.xp->X, cur.xp->m, cur.xp->posF[cur.ci], cur.xp->posL[cur.cj]); xpar_drop(&cur.xp); }
            eval_t ev; double t = now();
            { const rule_t *R = rule_at(cur.nd.depth); eval_node(&cur.nd, me->w, &ev, R->nfw1, R->nfw2, me->id); }
            me->st.t_bound += now() - t;
            me->st.nodes++; me->st.nodes_depth[cur.nd.depth]++;
            g_nodes_all++; __atomic_fetch_add(&g_depth_all[cur.nd.depth], 1, __ATOMIC_RELAXED);
            if (me->id == 0 && (me->st.nodes & 255) == 0) write_progress(0);
            if (ev.m <= 1 || ev.z > prune_thr(g_ub)) { me->sp = sp; pthread_mutex_unlock(&me->mtx); continue; }
            if ((opt_domlb && domlb(&cur.nd, &ev, me->w)) || (opt_domcov && domcov(&cur.nd, &ev, me->w))) { me->sp = sp; pthread_mutex_unlock(&me->mtx); continue; }
            if (opt_fb) add_fb(&cur.nd, &ev, me->w);
            child_t ch[NMAX]; int nch = branch(&cur.nd, me->w, &ev, ch, me->id);
            xpar_t *xp = NULL;
            if ((opt_warm & 2) && nch > 0) {
                xp = malloc(sizeof(xpar_t)); xp->ref = nch; xp->m = ev.m;
                memcpy(xp->X, me->w->X, sizeof(xp->X));
                for (int a = 0; a < ev.m; a++) { xp->posF[ev.F[a]] = a; xp->posL[ev.L[a]] = a; }
            }
            for (int c = nch - 1; c >= 0; c--) { /* push worst first so best is explored first */
                sitem_t s; s.nd = cur.nd; s.nd.rc[ch[c].i] = ch[c].j; s.nd.cr[ch[c].j] = ch[c].i; s.nd.depth++; s.lb = ch[c].lb;
                s.xp = xp; s.ci = ch[c].i; s.cj = ch[c].j;
                stack[sp++] = s;
            }
            me->sp = sp; pthread_mutex_unlock(&me->mtx);
        }
        pthread_mutex_lock(&pool_lock);
        pool_busy--; tasks_done++;
        if (g_log) { fprintf(g_log, "{\"t\":%.3f,\"event\":\"task_done\",\"worker\":%d,\"root_depth\":%d,\"nodes\":%ld,\"pool\":%d}\n", now() - g_t0, me->id, it.nd.depth, me->st.nodes - nodes0, pool_n); fflush(g_log); }
        pthread_cond_broadcast(&pool_cond);
        pthread_mutex_unlock(&pool_lock);
    }
    return NULL;
}

/* ---------- simple robust tabu search for an initial incumbent ---------- */
static double tabu(int iters, unsigned long long seed, int *best) {
    rng_t r = {seed * 7919 + 17};
    int p[NMAX]; for (int i = 0; i < N; i++) p[i] = i;
    for (int i = N - 1; i > 0; i--) { int j = rng_next(&r) % (i + 1); int t = p[i]; p[i] = p[j]; p[j] = t; }
    double cur = qap_cost(p), bestv = cur; memcpy(best, p, sizeof(int) * N);
    static double delta[NMAX][NMAX]; static long tl[NMAX][NMAX];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) tl[i][j] = -(N * i + j);
    #define DELTA(r_, s_) ({ int rr = (r_), ss = (s_); double d = (A[rr][rr]-A[ss][ss])*(B[p[ss]][p[ss]]-B[p[rr]][p[rr]]) + (A[rr][ss]-A[ss][rr])*(B[p[ss]][p[rr]]-B[p[rr]][p[ss]]); \
        for (int k = 0; k < N; k++) if (k != rr && k != ss) d += (A[k][rr]-A[k][ss])*(B[p[k]][p[ss]]-B[p[k]][p[rr]]) + (A[rr][k]-A[ss][k])*(B[p[ss]][p[k]]-B[p[rr]][p[k]]); d; })
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) delta[i][j] = DELTA(i, j);
    int tmin = (int)(0.9 * N), tmax = (int)(1.1 * N), aspir = N * N * 2;
    for (long it = 1; it <= iters; it++) {
        int ri = -1, si = -1; double bd = INFINITY; int already = 0;
        for (int i = 0; i < N - 1; i++) for (int j = i + 1; j < N; j++) {
            int autorized = (tl[i][p[j]] < it) || (tl[j][p[i]] < it);
            int aspired = (tl[i][p[j]] < it - aspir) || (tl[j][p[i]] < it - aspir) || (cur + delta[i][j] < bestv);
            if ((aspired && !already) || (aspired && already && delta[i][j] < bd) || (!aspired && !already && delta[i][j] < bd && autorized)) {
                ri = i; si = j; bd = delta[i][j]; if (aspired) already = 1;
            }
        }
        if (ri < 0) continue;
        int t = p[ri]; p[ri] = p[si]; p[si] = t; cur += delta[ri][si];
        int tt = tmin + rng_next(&r) % (tmax - tmin + 1);
        tl[ri][p[si]] = it + tt; tl[si][p[ri]] = it + tt;
        if (cur < bestv) { bestv = cur; memcpy(best, p, sizeof(int) * N); }
        for (int i = 0; i < N - 1; i++) for (int j = i + 1; j < N; j++) {
            if (i != ri && i != si && j != ri && j != si)
                delta[i][j] += (A[ri][i]-A[ri][j]+A[si][j]-A[si][i]) * (B[p[si]][p[i]]-B[p[si]][p[j]]+B[p[ri]][p[j]]-B[p[ri]][p[i]])
                             + (A[i][ri]-A[j][ri]+A[j][si]-A[i][si]) * (B[p[i]][p[si]]-B[p[j]][p[si]]+B[p[j]][p[ri]]-B[p[i]][p[ri]]);
            else delta[i][j] = DELTA(i, j);
        }
    }
    return bestv;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s solve|estimate|par|root|tabu file.dat [options]\n", argv[0]); return 1; }
    const char *mode = argv[1];
    read_qaplib(argv[2]);
    init_V();
    double ub0 = NAN; int dives = 1000, tabu_iters = 20000; unsigned long long seed = 1;
    const char *logfn = NULL; char strat_name = 'C';
    for (int a = 3; a < argc; a++) {
        if (!strcmp(argv[a], "-ub")) ub0 = atof(argv[++a]);
        else if (!strcmp(argv[a], "-enum")) opt_enum = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-hyb")) { opt_hyb = 1; opt_hybd = atof(argv[++a]); opt_sg = atoi(argv[++a]); opt_rhalf = 1; }
        else if (!strcmp(argv[a], "-hybm")) opt_hybm = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-defl")) opt_defl = atof(argv[++a]);
        else if (!strcmp(argv[a], "-domtab")) opt_domtab = 1;
        else if (!strcmp(argv[a], "-domtab2")) opt_domtab = 2;
        else if (!strcmp(argv[a], "-dtm")) { opt_dtmlo = atoi(argv[++a]); opt_dtmhi = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-hybpass")) opt_hybpass = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-hybi")) { opt_hybi = 1; opt_hybd = atof(argv[++a]); opt_sg = atoi(argv[++a]); opt_hybisg = atoi(argv[++a]); opt_rhalf = 1; }
        else if (!strcmp(argv[a], "-hybipass")) opt_hybipass = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-hybidep")) opt_hybidep = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-sgth")) opt_sgth = atof(argv[++a]);
        else if (!strcmp(argv[a], "-nfw")) opt_nfw = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-dives")) dives = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-q")) opt_q = atof(argv[++a]);
        else if (!strcmp(argv[a], "-noincr")) opt_incr = 0;
        else if (!strcmp(argv[a], "-noecache")) opt_ecache = 0;
        else if (!strcmp(argv[a], "-nomemo")) opt_memo = 0;
        else if (!strcmp(argv[a], "-nofastmm")) opt_fastmm = 0;
        else if (!strcmp(argv[a], "-nosimdlap")) opt_simdlap = 0;
        else if (!strcmp(argv[a], "-nospkeep")) opt_spkeep = 0;
        else if (!strcmp(argv[a], "-nospu")) opt_spu = 0;
        else if (!strcmp(argv[a], "-nofuse")) opt_fuse = 0;
        else if (!strcmp(argv[a], "-upd")) opt_upd = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-updstop")) opt_updstop = 1;
        else if (!strcmp(argv[a], "-nfwx")) opt_nfwx = atof(argv[++a]);
        else if (!strcmp(argv[a], "-warm")) opt_warm = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-alap")) opt_alap = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-monge")) opt_monge = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-spfuse")) opt_spfuse = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-lacap")) opt_lacap = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-c4")) opt_c4 = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-top")) opt_top = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-tlog")) g_tlog = argv[++a];
        else if (!strcmp(argv[a], "-c3")) opt_c3 = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nb4")) opt_nb4 = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nb3")) opt_nb3 = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-lasize")) opt_lasize = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-lanb")) opt_lanb = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-lanfw")) opt_lanfw = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-pfw")) opt_pfw = 1;
        else if (!strcmp(argv[a], "-threads")) opt_threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-seed")) seed = atoll(argv[++a]);
        else if (!strcmp(argv[a], "-sb")) opt_sb = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nbest")) opt_nbest = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nfw3")) opt_nfw3 = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-nosym")) opt_sym = 0;
        else if (!strcmp(argv[a], "-tmax")) opt_tmax = atof(argv[++a]);
        else if (!strcmp(argv[a], "-tabu")) tabu_iters = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-log")) logfn = argv[++a];
        else if (!strcmp(argv[a], "-shift")) opt_shift = argv[++a];
        else if (!strcmp(argv[a], "-fb")) opt_fb = 1;
        else if (!strcmp(argv[a], "-dom")) opt_dom = 1;
        else if (!strcmp(argv[a], "-dom2")) opt_dom = 2;
        else if (!strcmp(argv[a], "-sens")) opt_sens = 1;
        else if (!strcmp(argv[a], "-umax")) opt_umax = 1;
        else if (!strcmp(argv[a], "-dual")) { opt_dual = argv[++a]; atexit(print_dual); }
        else if (!strcmp(argv[a], "-dualmin")) opt_dualmin = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-glbpre")) { opt_glbpre = atoi(argv[++a]); atexit(print_glb); }
        else if (!strcmp(argv[a], "-fixorder")) opt_fixorder = 1;
        else if (!strcmp(argv[a], "-topd")) opt_topd = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-topx")) opt_topx = atof(argv[++a]);
        else if (!strcmp(argv[a], "-r2p")) opt_r2p = atof(argv[++a]);
        else if (!strcmp(argv[a], "-ext")) { opt_ext = atoi(argv[++a]); opt_extd = atof(argv[++a]); }
        else if (!strcmp(argv[a], "-extn")) opt_extn = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-fstop")) { opt_fstopm = atoi(argv[++a]); opt_fstopl = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-inh")) { opt_inh = atoi(argv[++a]); atexit(print_inh); }
        else if (!strcmp(argv[a], "-coldest")) opt_coldest = 1;
        else if (!strcmp(argv[a], "-prof")) { g_prof = 1; g_prof_t0 = now(); g_prof_c0 = __rdtsc(); atexit(print_prof); }
        else if (!strcmp(argv[a], "-fwlog")) { g_fwlog = fopen(argv[++a], "w"); g_fwlog_k = atol(argv[++a]); }
        else if (!strcmp(argv[a], "-gapstat")) { g_gapstat = 1; atexit(print_gap); }
        else if (!strcmp(argv[a], "-timeig")) { g_timeig = 1; atexit(print_teig); }
        else if (!strcmp(argv[a], "-dom3")) opt_dom = 3;
        else if (!strcmp(argv[a], "-domlb")) opt_dom = opt_domlb = 1;
        else if (!strcmp(argv[a], "-domcov")) { opt_dom = 1; opt_domcov = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-dombr")) opt_dom = opt_dombr = 1;
        else if (!strcmp(argv[a], "-dombr0")) opt_dombr = 1;
        else if (!strcmp(argv[a], "-lap")) opt_lap = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-strat")) strat_name = argv[++a][0];
        else if (!strcmp(argv[a], "-nointprune")) opt_intprune = 0;
        else if (!strcmp(argv[a], "-nopar")) opt_par = 0;
        else if (!strcmp(argv[a], "-brtie")) opt_brtie = 1;
        else if (!strcmp(argv[a], "-brdyn")) opt_brdyn = 1;
        else if (!strcmp(argv[a], "-brall")) opt_brall = 1;
        else if (!strcmp(argv[a], "-latie")) { opt_latie_lo = atoi(argv[++a]); opt_latie_hi = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-latiek")) opt_latiek = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-laecc")) opt_laecc = 1;
        else if (!strcmp(argv[a], "-lak")) opt_lak = atof(argv[++a]);
        else if (!strcmp(argv[a], "-brecc")) { opt_brecc_lo = atoi(argv[++a]); opt_brecc_hi = atoi(argv[++a]); }
        else if (!strcmp(argv[a], "-prog")) g_prog = argv[++a];
        else if (!strcmp(argv[a], "-ckpt")) opt_ckpt = argv[++a];
        else if (!strcmp(argv[a], "-ckpt_s")) opt_ckpt_s = atof(argv[++a]);
        else if (!strcmp(argv[a], "-resume")) opt_resume = argv[++a];
        else if (!strcmp(argv[a], "-v")) opt_verbose = 1;
    }
    if (logfn) g_log = fopen(logfn, "w");
    set_strategy(strat_name);
    g_t0 = now();
    compute_aut(A, &GA); compute_aut(B, &GB);
    for (int i = 0; i < N; i++) { double t = 0; for (int k = 0; k < N; k++) t += A[i][k]; g_rowsumA[i] = t; }
    if (opt_par) {   /* A, B symmetric with zero diagonal: cost = 2 sum_{i<k} A_ik B_p(i)p(k), a multiple of 2 gcd(A) gcd(B) */
        long ga = 0, gb = 0; int ok = 1;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            if (A[i][j] != floor(A[i][j]) || B[i][j] != floor(B[i][j]) || (i == j && (A[i][i] != 0 || B[i][i] != 0))) ok = 0;
            ga = lgcd(ga, labs((long)A[i][j])); gb = lgcd(gb, labs((long)B[i][j])); }
        g_gran = ok && ga && gb ? 2 * ga * gb : 1;
        if (opt_verbose) fprintf(stderr, "cost granularity %ld\n", g_gran);
    }
    if (opt_verbose) fprintf(stderr, "n=%d |Aut(A)|=%d |Aut(B)|=%d\n", N, GA.n, GB.n);
    int p[NMAX];
    double th0 = now();
    double hv = tabu(tabu_iters, seed, p);
    double t_heur = now() - th0;
    if (opt_enum) for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
        if (A[i][j] < 0 || B[i][j] < 0 || C0[i][j] < 0) { fprintf(stderr, "-enum needs nonnegative data\n"); exit(2); }
    /* with -shift the increment Cr[a][b] + Ar[a][a] Br[b][b] of enum_rec is the unshifted one, so it stays nonnegative */
    if (opt_shift) {
        /* equivalent reformulation: A + diag(d), B + diag(g), linear term C0 */
        FILE *f = fopen(opt_shift, "r"); double d[NMAX], g[NMAX];
        for (int i = 0; i < N; i++) if (fscanf(f, "%lf", &d[i]) != 1) exit(4);
        for (int i = 0; i < N; i++) if (fscanf(f, "%lf", &g[i]) != 1) exit(4);
        fclose(f);
        double dA[NMAX], dB[NMAX]; for (int i = 0; i < N; i++) { dA[i] = A[i][i]; dB[i] = B[i][i]; }
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C0[i][j] = -(d[i] * dB[j] + dA[i] * g[j] + d[i] * g[j]);
        for (int i = 0; i < N; i++) { A[i][i] += d[i]; B[i][i] += g[i]; }
    }
    if (opt_dual) {
        FILE *f = fopen(opt_dual, "r"); double d[NMAX], g[NMAX];
        for (int i = 0; i < N; i++) if (fscanf(f, "%lf", &d[i]) != 1) exit(4);
        for (int i = 0; i < N; i++) if (fscanf(f, "%lf", &g[i]) != 1) exit(4);
        fclose(f);
        memcpy(A2, A, sizeof A2); memcpy(B2, B, sizeof B2);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C02[i][j] = C0[i][j] - (d[i] * B[j][j] + A[i][i] * g[j] + d[i] * g[j]);
        for (int i = 0; i < N; i++) { A2[i][i] += d[i]; B2[i][i] += g[i]; }
    }
    g_ub = INFINITY; report_solution(p, hv, -1);
    if (!isnan(ub0) && ub0 < g_ub) g_ub = ub0;     /* external bound (e.g. known optimum + 1) */
    if (!strcmp(mode, "tabu")) { printf("{\"tabu\":%.0f,\"t\":%.2f}\n", hv, t_heur); return 0; }
    if (getenv("GLBMIX")) {   /* experiment: bound lambda * QPB(A, B, C0 + (1 - lambda)/lambda L), L the Gilmore-Lawler linear minorant */
        double lam = atof(getenv("GLBMIX"));
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double a[NMAX], b[NMAX]; int na = 0, nb = 0;
            for (int k = 0; k < N; k++) if (k != i) a[na++] = A[i][k];
            for (int l = 0; l < N; l++) if (l != j) b[nb++] = B[j][l];
            isort(a, na); isort(b, nb);
            double mp = 0; for (int t = 0; t < na; t++) mp += a[t] * b[na - 1 - t];
            double Lij = A[i][i] * B[j][j] + mp;
            C0[i][j] = lam * C0[i][j] + (1 - lam) * Lij;
        }
        for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) A[i][k] *= lam;
        if (lam == 0) { /* GLB itself: LAP of L */ }
    }
    if (!strcmp(mode, "rltbb")) {   /* experiment: B&B on the inherited RLT1 dual state */
        opt_rhalf = getenv("HALF") ? atoi(getenv("HALF")) : 1;
        if (getenv("PASS")) opt_rbb_pass = atoi(getenv("PASS")); if (getenv("SGIT")) opt_sg = atoi(getenv("SGIT")); if (getenv("THETA")) opt_sgth = atof(getenv("THETA")); if (getenv("EPS")) opt_rbb_eps = atof(getenv("EPS"));
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        work_t *w = calloc(1, sizeof(work_t)); eval_t ev; double save = g_ub; g_ub = INFINITY;
        eval_node(&nd, w, &ev, opt_nfw, opt_nfw, 0); g_ub = save;
        static rlt_t rr; rlt_init(&rr, ev.m, w->Ar, w->Br, w->Cr, ev.K);
        int RP = getenv("ROOTPASS") ? atoi(getenv("ROOTPASS")) : 50; for (int k = 0; k < RP; k++) rlt_pass(&rr);
        double root = rr.R, t0 = now(); for (int i = 0; i < N; i++) g_rbb_perm[i] = -1;
        rbb_dfs(&rr, ev.F, ev.L, 0);
        printf("{\"mode\":\"rltbb\",\"n\":%d,\"root\":%.2f,\"opt\":%.0f,\"nodes\":%ld,\"passes\":%ld,\"sg_it\":%ld,\"wall_s\":%.2f}\n", N, root, (double)g_ub, g_rbb_nodes, g_rbb_pass, g_sg_it, now() - t0);
        return 0;
    }
    if (!strcmp(mode, "rlt1")) {   /* RLT1 dual ascent at one node (FIX="i:j ..."), ITERS passes */
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        if (getenv("FIX")) { const char *q = getenv("FIX"); int i, j, k;
            while (sscanf(q, "%d:%d%n", &i, &j, &k) == 2) { nd.rc[i] = j; nd.cr[j] = i; nd.depth++; q += k; } }
        int IT = getenv("ITERS") ? atoi(getenv("ITERS")) : 20; opt_rhalf = getenv("HALF") ? atoi(getenv("HALF")) : 0;
        work_t *w = calloc(1, sizeof(work_t)); eval_t ev; double save = g_ub; g_ub = INFINITY;
        double tq = now(); eval_node(&nd, w, &ev, opt_nfw, opt_nfw, 0); tq = now() - tq; g_ub = save;
        static rlt_t rr; double t0 = now();
        rlt_init(&rr, ev.m, w->Ar, w->Br, w->Cr, ev.K);
        printf("{\"m\":%d,\"qpb\":%.2f,\"t_qpb\":%.5f,\"rlt\":[", ev.m, ev.z, tq);
        for (int it = 1; it <= IT; it++) { double z = rlt_pass(&rr); printf("%s[%d,%.2f,%.4f]", it > 1 ? "," : "", it, z, now() - t0); }
        printf("]");
        if (getenv("CYC")) {   /* cycles of SG k iterations + bake + one DP pass */
            int k = atoi(getenv("CYC")), nc = getenv("NCYC") ? atoi(getenv("NCYC")) : 50; double thr = getenv("THR") ? atof(getenv("THR")) : g_ub;
            if (getenv("THETA")) opt_sgth = atof(getenv("THETA"));
            double t1 = now(); printf(",\"cyc\":[");
            for (int c = 1; c <= nc; c++) { rlt_sg(&rr, k, thr); rlt_pass(&rr); printf("%s[%d,%.2f,%.4f]", c > 1 ? "," : "", c, rr.R, now() - t1); if (rr.R > thr) break; }
            printf("]");
        }
        if (getenv("SGF")) {   /* rlt_sg from the same state at several budgets (DEFL = deflection gamma) */
            double thr = getenv("THR") ? atof(getenv("THR")) : g_ub; if (getenv("THETA")) opt_sgth = atof(getenv("THETA"));
            if (getenv("DEFL")) opt_defl = atof(getenv("DEFL"));
            size_t M4 = (size_t)ev.m * ev.m * ev.m * ev.m; static rlt_t cp; double *Qs = malloc(sizeof(double) * M4);
            printf(",\"sgf\":[");
            int bud[] = {30, 100, 300, 1000, 3000}; for (int k = 0; k < 5; k++) {
                cp = rr; cp.Q = Qs; memcpy(Qs, rr.Q, sizeof(double) * M4); long i0 = g_sg_it; double t1 = now();
                rlt_sg(&cp, bud[k], thr); printf("%s[%d,%.2f,%ld,%.4f]", k ? "," : "", bud[k], cp.R, g_sg_it - i0, now() - t1); }
            printf("]");
        }
        int SG = getenv("SG") ? atoi(getenv("SG")) : 0;
        if (SG) {   /* subgradient on the twin splits, Polyak step towards THR */
            int m = ev.m; double thr = getenv("THR") ? atof(getenv("THR")) : g_ub, th = getenv("THETA") ? atof(getenv("THETA")) : 1.0;
            double *lam = calloc((size_t)m * m * m * m, sizeof(double)); static int ys[NMAX * NMAX][NMAX]; int xm[NMAX], xinv[NMAX];
            double best = rr.R, t1 = now(); printf(",\"sg\":[");
            for (int it = 1; it <= SG; it++) {
                double z = rlt_lag(&rr, lam, xm, ys); if (z > best) best = z;
                if (it == 1 || it % 10 == 0 || it == SG) printf("%s[%d,%.2f,%.2f,%.4f]", it > 1 ? "," : "", it, z, best, now() - t1);
                if (best > thr) break;
                /* g_abcd = x_ab y^ab_cd - x_cd y^cd_ab, kept antisymmetric through lam_abcd - lam_cdab */
                for (int a = 0; a < m; a++) xinv[xm[a]] = a;
                double g2 = 0; int cnt = 0;
                for (int a = 0; a < m; a++) { int b = xm[a]; for (int cc = 0; cc < m; cc++) if (cc != a) { int d = ys[a * m + b][cc];
                    /* +1 on (ab,cd) unless the twin slave (cd) chose (ab) with x_cd = 1 too */
                    int twin = (xm[cc] == d && ys[cc * m + d][a] == b); if (!twin) cnt++; } }
                g2 = 2.0 * cnt; if (g2 == 0) { printf("]"); break; }
                double step = th * (thr - z) / g2; if (step < 0) step = -step;
                for (int a = 0; a < m; a++) { int b = xm[a]; for (int cc = 0; cc < m; cc++) if (cc != a) { int d = ys[a * m + b][cc];
                    int twin = (xm[cc] == d && ys[cc * m + d][a] == b); if (!twin) lam[QX(m, a, b, cc, d)] += step; } }
                if (it == SG) printf("]");
            }
            printf(",\"best\":%.2f", best);
        }
        printf("}\n");
        return 0;
    }
    if (!strcmp(mode, "dbound")) {   /* QPB + D at one node (FIX="i:j ..."): sup over (s, t, D) by Polyak supergradient, exact convexity repair */
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        if (getenv("FIX")) { const char *q = getenv("FIX"); int i, j, k;
            while (sscanf(q, "%d:%d%n", &i, &j, &k) == 2) { nd.rc[i] = j; nd.cr[j] = i; nd.depth++; q += k; } }
        double thr = getenv("THR") ? atof(getenv("THR")) : g_ub; int OUT = getenv("OUT") ? atoi(getenv("OUT")) : 40, INN = getenv("INN") ? atoi(getenv("INN")) : 30;
        double theta = getenv("THETA") ? atof(getenv("THETA")) : 1.0;
        work_t *w = calloc(1, sizeof(work_t)); eval_t ev; double save = g_ub; g_ub = INFINITY;
        double tq = now(); eval_node(&nd, w, &ev, opt_nfw, opt_nfw, 0); tq = now() - tq; g_ub = save;
        int m = ev.m, n1 = m - 1, NN = n1 * n1;
        static double Ar[NMAX][NMAX], Br[NMAX][NMAX], Cr[NMAX][NMAX], V[NMAX][NMAX], Ah[NMAX][NMAX], Bh[NMAX][NMAX], WA[NMAX][NMAX], WB[NMAX][NMAX], UA[NMAX][NMAX], UB[NMAX][NMAX];
        double K = ev.K, lam[NMAX], mu[NMAX];
        for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) { Ar[a][b] = w->Ar[a][b]; Br[a][b] = w->Br[a][b]; Cr[a][b] = w->Cr[a][b]; }
        { double hw[NMAX], nn = 0; for (int i = 0; i < m; i++) { hw[i] = 1 / sqrt(m) - (i == m - 1); nn += hw[i] * hw[i]; }
          for (int i = 0; i < m; i++) for (int k = 0; k < n1; k++) V[i][k] = (i == k) - 2 * hw[i] * hw[k] / nn; }
        for (int k = 0; k < n1; k++) for (int l = 0; l < n1; l++) { double x = 0, y = 0;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { x += V[i][k] * Ar[i][j] * V[j][l]; y += V[i][k] * Br[i][j] * V[j][l]; }
            Ah[k][l] = x; Bh[k][l] = y; }
        { double *H = malloc(sizeof(double) * n1 * n1), wk[64 * NMAX]; int lw = 64 * NMAX, info;
          for (int pass = 0; pass < 2; pass++) { double (*M)[NMAX] = pass ? Bh : Ah, (*U)[NMAX] = pass ? UB : UA; double *ev2 = pass ? mu : lam;
            for (int k = 0; k < n1; k++) for (int l = 0; l < n1; l++) H[k + l * n1] = M[k][l];
            dsyev_("V", "U", &n1, H, &n1, ev2, wk, &lw, &info);
            for (int k = 0; k < n1; k++) for (int l = 0; l < n1; l++) U[k][l] = H[k + l * n1]; }
          free(H); }
        for (int i = 0; i < m; i++) for (int k = 0; k < n1; k++) { double x = 0, y = 0; for (int r = 0; r < n1; r++) { x += V[i][r] * UA[r][k]; y += V[i][r] * UB[r][k]; } WA[i][k] = x; WB[i][k] = y; }
        double sv[NMAX], tv[NMAX];
        { static double c[NMAX][NMAX]; int rs[NMAX]; double u[NMAX + 1], v[NMAX + 1];
          for (int k = 0; k < n1; k++) for (int l = 0; l < n1; l++) c[k][l] = lam[k] * mu[l];
          lap(n1, c, rs, u, v); for (int k = 0; k < n1; k++) { sv[k] = u[k]; tv[k] = v[k]; } }
        static double D[NMAX][NMAX], S[NMAX][NMAX], T[NMAX][NMAX], X[NMAX][NMAX], G[NMAX][NMAX], Y[NMAX][NMAX], Sh[NMAX][NMAX], Th[NMAX][NMAX];
        memset(D, 0, sizeof D);
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) X[i][j] = 1.0 / m;
        double *HH = malloc(sizeof(double) * NN * NN), *AV = malloc(sizeof(double) * NN * NMAX), *ew = malloc(sizeof(double) * NN), *wk2 = NULL;
        for (int k = 0; k < n1; k++) for (int k2 = 0; k2 < n1; k2++) for (int i = 0; i < m; i++) AV[(k * n1 + k2) * NMAX + i] = V[i][k] * V[i][k2];
        double best = -INFINITY, t0 = now(), teig = 0; int lwq = -1, info; double wq; dsyev_("N", "U", &NN, HH, &NN, ew, &wq, &lwq, &info); lwq = (int)wq; wk2 = malloc(sizeof(double) * lwq);
        #define MM(R, P, Q) for (int i_ = 0; i_ < m; i_++) for (int j_ = 0; j_ < m; j_++) { double z_ = 0; for (int k_ = 0; k_ < m; k_++) z_ += P[i_][k_] * Q[k_][j_]; R[i_][j_] = z_; }
        for (int it = 0; it <= OUT; it++) {
            for (int k = 0; k < n1; k++) for (int l = 0; l < n1; l++) { double x = 0, y = 0; for (int r = 0; r < n1; r++) { x += UA[k][r] * sv[r] * UA[l][r]; y += UB[k][r] * tv[r] * UB[l][r]; } Sh[k][l] = x; Th[k][l] = y; }
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double x = 0, y = 0; for (int r = 0; r < n1; r++) { x += WA[i][r] * sv[r] * WA[j][r]; y += WB[i][r] * tv[r] * WB[j][r]; } S[i][j] = x; T[i][j] = y; }
            /* exact convexity check on the affine hull, repair by a uniform shift of s */
            double te = now();
            for (int k = 0; k < n1; k++) for (int k2 = 0; k2 < n1; k2++) { const double *a = AV + (k * n1 + k2) * NMAX; double Da[NMAX];
                for (int j = 0; j < m; j++) { double z = 0; for (int i = 0; i < m; i++) z += a[i] * D[i][j]; Da[j] = z; }
                for (int l = 0; l < n1; l++) for (int l2 = 0; l2 < n1; l2++) { const double *b = AV + (l * n1 + l2) * NMAX; double z = 0; for (int j = 0; j < m; j++) z += Da[j] * b[j];
                    HH[(k * n1 + l) + (size_t)(k2 * n1 + l2) * NN] = Ah[k][k2] * Bh[l][l2] - Sh[k][k2] * (l == l2) - (k == k2) * Th[l][l2] - z; } }
            dsyev_("N", "U", &NN, HH, &NN, ew, wk2, &lwq, &info); teig += now() - te;
            double lmin = ew[0], marg = 1e-7 * (1 + fabs(ew[NN - 1]));
            if (lmin < marg) { double c = marg - lmin; for (int k = 0; k < n1; k++) sv[k] -= c;
                for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double x = 0; for (int r = 0; r < n1; r++) x += WA[i][r] * WA[j][r]; S[i][j] -= c * x; } }
            double cst = K; for (int k = 0; k < n1; k++) cst += sv[k] + tv[k];
            /* Frank-Wolfe on q over the doubly stochastic matrices */
            double lb = -INFINITY, qx = 0;
            for (int f = 0; f < INN; f++) {
                static double AX[NMAX][NMAX], AXB[NMAX][NMAX], SX[NMAX][NMAX], XT[NMAX][NMAX];
                MM(AX, Ar, X); MM(AXB, AX, Br); MM(SX, S, X); MM(XT, X, T);
                double qv = cst, gx = 0;
                for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double hx = AXB[i][j] - SX[i][j] - XT[i][j] - D[i][j] * X[i][j];
                    qv += X[i][j] * (hx + Cr[i][j] + D[i][j]); G[i][j] = 2 * hx + Cr[i][j] + D[i][j]; gx += G[i][j] * X[i][j]; }
                int rs[NMAX]; double u[NMAX + 1], v[NMAX + 1]; double lp = lap(m, G, rs, u, v);
                double l1 = qv + lp - gx; if (l1 > lb) lb = l1; qx = qv;
                for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) Y[i][j] = (rs[i] == j) - X[i][j];
                MM(AX, Ar, Y); MM(AXB, AX, Br); MM(SX, S, Y); MM(XT, Y, T);
                double gy = 0, cy = 0; for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { gy += G[i][j] * Y[i][j]; cy += Y[i][j] * (AXB[i][j] - SX[i][j] - XT[i][j] - D[i][j] * Y[i][j]); }
                double gam = cy > 1e-12 ? -gy / (2 * cy) : 1; if (gam > 1) gam = 1; if (gam < 0) gam = 0;
                for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) X[i][j] += gam * Y[i][j];
            }
            if (lb > best) best = lb;
            if (getenv("VERB")) fprintf(stderr, "it %d lb %.2f q %.2f lmin %.3g best %.2f t %.3f\n", it, lb, qx, lmin, best, now() - t0);
            if (best > thr || it == OUT) break;
            /* supergradient of phi at X, Polyak step towards thr */
            static double gD[NMAX][NMAX]; double gs[NMAX], gt[NMAX], g2 = 0;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { gD[i][j] = X[i][j] - X[i][j] * X[i][j]; g2 += gD[i][j] * gD[i][j]; }
            for (int k = 0; k < n1; k++) { double a = 0, b = 0; for (int j = 0; j < m; j++) { double x = 0, y = 0; for (int i = 0; i < m; i++) { x += X[i][j] * WA[i][k]; y += X[j][i] * WB[i][k]; } a += x * x; b += y * y; }
                gs[k] = 1 - a; gt[k] = 1 - b; g2 += gs[k] * gs[k] + gt[k] * gt[k]; }
            double eta = theta * (thr + 1 - lb) / (g2 + 1e-12);
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) D[i][j] += eta * gD[i][j];
            for (int k = 0; k < n1; k++) { sv[k] += eta * gs[k]; tv[k] += eta * gt[k]; }
        }
        printf("{\"m\":%d,\"qpb\":%.2f,\"qpbd\":%.2f,\"t_qpb\":%.5f,\"t\":%.4f,\"t_eig\":%.4f}\n", m, ev.z, best, tq, now() - t0, teig);
        return 0;
    }
    if (!strcmp(mode, "root")) {
        work_t *w = calloc(1, sizeof(work_t)); node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        if (getenv("FIX")) { const char *q = getenv("FIX"); int i, j, k;   /* FIX="i:j i:j ...": evaluate that node */
            while (sscanf(q, "%d:%d%n", &i, &j, &k) == 2) { nd.rc[i] = j; nd.cr[j] = i; nd.depth++; q += k; } }
        eval_t ev; double t = now();
        double save = g_ub; g_ub = INFINITY;
        eval_node(&nd, w, &ev, opt_nfw, opt_nfw, 0); g_ub = save;
        printf("{\"root_qpb\":%.2f,\"pb\":%.2f,\"fw_iters\":%ld,\"t\":%.4f,\"ub\":%.0f}\n", ev.z, w->trST, w->nfw, now() - t, (double)g_ub);
        return 0;
    }
    if (!strcmp(mode, "frontier")) {   /* exact QPB subtree size below each surviving node at depth -top L */
        work_t *w = calloc(1, sizeof(work_t));
        node_t root; for (int i = 0; i < N; i++) root.rc[i] = root.cr[i] = -1, root.fb[i] = 0; root.depth = 0;
        frontier_t F = {0}; top_enum(&root, w, &F);
        printf("# top %d: %.0f nodes above, %d frontier nodes, ub %.0f\n", opt_top, F.cnt, F.n, (double)g_ub);
        for (int f = 0; f < F.n; f++) {
            long n0 = (long)g_nodes_all; double t = now(); stats_t st; memset(&st, 0, sizeof st);
            node_t nd = F.v[f]; g_base = nd; g_base_depth = nd.depth; dfs(&nd, w, &st, 0);
            printf("F %d %ld %.3f", f, (long)g_nodes_all - n0, now() - t);
            for (int i = 0; i < N; i++) if (F.v[f].rc[i] >= 0) printf(" %d:%d", i, F.v[f].rc[i]);
            printf("\n"); fflush(stdout);
        }
        return 0;
    }
    if (!strcmp(mode, "oracle")) {   /* branching oracle: at sampled frontier nodes (depth -top L), the exact subtree size under each branching line */
        work_t *w = calloc(1, sizeof(work_t)), *w2 = calloc(1, sizeof(work_t));
        node_t root; for (int i = 0; i < N; i++) root.rc[i] = root.cr[i] = -1, root.fb[i] = 0; root.depth = 0;
        frontier_t F = {0}; top_enum(&root, w, &F);
        int step = getenv("STEP") ? atoi(getenv("STEP")) : 1, off = getenv("OFF") ? atoi(getenv("OFF")) : 0;
        printf("# top %d: %d frontier nodes, ub %.0f, every %d-th from %d\n", opt_top, F.n, (double)g_ub, step, off);
        static double Xs[NMAX][NMAX];
        for (int f = off; f < F.n; f += step) {
            node_t nd = F.v[f]; eval_t ev; const rule_t *R = rule_at(nd.depth);
            w->x0_ok = 0; eval_node(&nd, w, &ev, R->nfw1, R->nfw2, 0);
            if (ev.m <= 1 || ev.z > prune_thr(g_ub)) continue;
            int m = ev.m; double thr = prune_thr(g_ub), U[NMAX][NMAX]; memcpy(Xs, w->X, sizeof Xs);
            for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) U[a][b] = w->U[a][b];
            child_t ch[NMAX]; int nch = branch(&nd, w, &ev, ch, 0);
            int dk = -1, dl = -1;   /* the default line */
            if (nch >= 2) { if (ch[0].i == ch[1].i) dk = 0; else dk = 1; }
            else if (nch == 1) dk = 0;
            if (dk == 0 && nch) for (int a = 0; a < m; a++) if (ev.F[a] == ch[0].i) dl = a;
            if (dk == 1) for (int b = 0; b < m; b++) if (ev.L[b] == ch[0].j) dl = b;
            printf("N %d m %d z %.2f default %d:%d F", f, m, ev.z, dk, dl);
            for (int a = 0; a < m; a++) printf("%s%d", a ? "," : " ", ev.F[a]);
            printf(" L"); for (int b = 0; b < m; b++) printf("%s%d", b ? "," : " ", ev.L[b]);
            printf(" U"); for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) printf("%s%.1f", a + b ? "," : " ", U[a][b]);
            printf(" S");
            long best = -1, dsz = -1;
            for (int kind = 0; kind < 2; kind++) for (int line = 0; line < m; line++) {
                long n0 = (long)g_nodes_all; double t = now();
                for (int o = 0; o < m; o++) {
                    int a = kind == 0 ? line : o, b = kind == 0 ? o : line;
                    if (ev.z + U[a][b] > thr) continue;
                    node_t c = nd; c.rc[ev.F[a]] = ev.L[b]; c.cr[ev.L[b]] = ev.F[a]; c.depth++;
                    inherit_x(w2, (const double (*)[NMAX])Xs, m, a, b);
                    stats_t st; memset(&st, 0, sizeof st); dfs(&c, w2, &st, 0);
                }
                long sz = 1 + (long)g_nodes_all - n0;
                printf(" %ld/%.3f", sz, now() - t); fflush(stdout);
                if (best < 0 || sz < best) best = sz;
                if (kind == dk && line == dl) dsz = sz;
            }
            printf(" | default %ld best %ld\n", dsz, best); fflush(stdout);
        }
        return 0;
    }
    if (!strcmp(mode, "estimate")) { if (opt_top > 0) estimate_top(dives, seed); else estimate(dives, seed); return 0; }
    double t0 = now();
    stats_t tot = {0}; long lap_tot = 0, fw_tot = 0;
    if (!strcmp(mode, "solve")) {
        work_t *w = calloc(1, sizeof(work_t));
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        g_ckpt_last = now(); g_base = nd; g_base_depth = 0;
        if (opt_resume) {
            read_resume(opt_resume);
            for (g_rnext = 0; g_rnext < g_rn; ) {
                g_base = g_rlist[g_rnext++]; g_base_depth = g_base.depth;
                nd = g_base; dfs(&nd, w, &tot, 0);
            }
        } else dfs(&nd, w, &tot, 0);
        if (opt_ckpt) { FILE *f = fopen(opt_ckpt, "w"); if (f) { fprintf(f, "nugckpt 1 n %d ub %.0f nodes %ld elapsed %.1f\n", N, (double)g_ub, g_nodes_before + (long)g_nodes_all, g_elapsed_before + now() - g_t0); fclose(f); } }
        lap_tot = w->nlap; fw_tot = w->nfw;
    } else if (!strcmp(mode, "par")) {
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1, nd.fb[i] = 0; nd.depth = 0;
        if (opt_resume) { read_resume(opt_resume); for (int r = 0; r < g_rn; r++) pool_push(&g_rlist[r], -INFINITY); }
        else pool_push(&nd, -INFINITY);
        pthread_t th[64]; static wk_t wk[64]; g_wk = wk; g_ckpt_last = now();
        for (int k = 0; k < opt_threads; k++) { memset(&wk[k], 0, sizeof(wk_t)); wk[k].id = k; wk[k].w = calloc(1, sizeof(work_t));
            wk[k].stack = malloc(sizeof(sitem_t) * NMAX * NMAX * 2); pthread_mutex_init(&wk[k].mtx, NULL); }
        for (int k = 0; k < opt_threads; k++) pthread_create(&th[k], NULL, worker, &wk[k]);
        for (int k = 0; k < opt_threads; k++) {
            pthread_join(th[k], NULL);
            tot.nodes += wk[k].st.nodes; tot.fathom_at_bound += wk[k].st.fathom_at_bound; tot.t_bound += wk[k].st.t_bound;
            for (int d = 0; d <= N; d++) tot.nodes_depth[d] += wk[k].st.nodes_depth[d];
            lap_tot += wk[k].w->nlap; fw_tot += wk[k].w->nfw;
        }
        if (opt_ckpt) { FILE *f = fopen(opt_ckpt, "w"); if (f) { fprintf(f, "nugckpt 1 n %d ub %.0f nodes %ld elapsed %.1f\n", N, (double)g_ub, g_nodes_before + (long)g_nodes_all, g_elapsed_before + now() - g_t0); fclose(f); } }
    }
    double el = now() - t0;
    write_progress(1);
    printf("{\"mode\":\"%s\",\"strategy\":\"%c\",\"n\":%d,\"opt\":%.0f,\"heuristic\":%.0f,\"t_heur\":%.2f,\"nodes\":%ld,\"laps\":%ld,\"fw_iters\":%ld,"
           "\"wall_s\":%.2f,\"cpu_bound_s\":%.2f,\"threads\":%d,\"tasks\":%ld,\"symA\":%d,\"symB\":%d,\"dom\":%ld,\"enum\":%ld,\"domlb\":%ld,\"fw_branch\":[%ld,%ld],\"fw_fathom\":[%ld,%ld],\"perm\":[",
           mode, strat_name, N, (double)g_ub, hv, t_heur, tot.nodes, lap_tot, fw_tot, el, tot.t_bound,
           strcmp(mode, "par") ? 1 : opt_threads, tasks_done, GA.n, GB.n, g_dom_pruned, g_enum_nodes, g_domlb_pruned + g_domcov_pruned, g_fwstat[0][0], g_fwstat[0][1], g_fwstat[1][0], g_fwstat[1][1]);
    for (int i = 0; i < N; i++) printf("%d%s", g_best[i] + 1, i < N - 1 ? "," : "");
    printf("],\"nodes_depth\":[");
    int last = 0; for (int d = 0; d <= N; d++) if (tot.nodes_depth[d]) last = d;
    for (int d = 0; d <= last; d++) printf("%ld%s", tot.nodes_depth[d], d < last ? "," : "");
    printf("]}\n");
    if (opt_domtab) fprintf(stderr, "{\"domtab_pruned\":%ld,\"domtab_tr\":%ld,\"domtab_ins\":%ld,\"domtab_lap\":%ld,\"ints\":%ld}\n", g_domtab_pruned, g_domtab_tr, g_domtab_ins, g_domtab_lap, g_dtau);
    if (opt_hyb) fprintf(stderr, "{\"hyb_calls\":%ld,\"hyb_fath\":%ld,\"hyb_t\":%.2f}\n", g_hyb_calls, g_hyb_fath, g_hyb_t);
    if (opt_hybi) fprintf(stderr, "{\"cold\":%ld,\"cold_fath\":%ld,\"inherited\":%ld,\"inh_fath_free\":%ld,\"inh_fath_work\":%ld,\"passes\":%ld,\"sg_it\":%ld,\"mean_gain\":%.2f,\"rlt_t\":%.2f,\"inh_t\":%.2f}\n", g_hi_cold, g_hi_coldf, g_hi_inh, g_hi_free, g_hi_work, g_hi_pass, g_hi_sg, g_hi_inh ? g_hi_gain / (g_hi_inh - g_hi_free) : 0, g_hi_t, g_hi_tinh);
    if (g_log) fclose(g_log);
    return 0;
}
