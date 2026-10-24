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
static int opt_intprune = 1;   /* integer data: prune if bound > ub - 1 (else bound >= ub) */

/* branching strategy by depth (Brixius-Anstreicher 2001, Tables 1-2) */
typedef struct { int maxdepth, rule, nfw1, nfw2, nfw3, nbest; } rule_t;
static rule_t strat[8]; static int nstrat = 0;
static void set_strategy(char s) {
    nstrat = 0;
    int c4 = -1, c3 = -1;
    if (s == 'B') { c4 = 1; c3 = 2; } else if (s == 'C') { c4 = 1; c3 = 3; } else if (s == 'D') { c4 = 2; c3 = 4; }
    else if (s == 'E') { c4 = 3; c3 = 5; }
    if (s == 'A') { strat[nstrat++] = (rule_t){NMAX, 2, 150, 100, 0, 0}; return; }
    strat[nstrat++] = (rule_t){c4, 4, 150, 100, 50, 20};
    strat[nstrat++] = (rule_t){c3, 3, 100, 100, 25, 10};
    strat[nstrat++] = (rule_t){NMAX, 2, 75, 50, 0, 0};
}
static const rule_t *rule_at(int depth) {
    for (int k = 0; k < nstrat; k++) if (depth <= strat[k].maxdepth) return &strat[k];
    return &strat[nstrat - 1];
}
static inline double prune_thr(double ub) { return opt_intprune ? ub - 1 + 1e-6 : ub - 1e-6; }

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
static int opt_lap = 2;   /* 0 = Hungarian, 1 = JV cold, 2 = JV warm start */

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

static int opt_simdlap = 0;   /* SIMD rewrite measured slower on nug16a (9.0 s vs 8.6 s) */
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
    long nlap, nfw;
} work_t;

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
    if (opt_ecache && e->m == m && e->mask == mask) { w->hits++; return e; }
    w->miss++;
    double W[NMAX][NMAX]; int k = m - 1;
    proj_eig(m, M, e->lam, W);
    for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
        double t = 0; for (int l = 0; l < k; l++) t += Vb[m][i][l] * W[l][j]; e->VW[i][j] = t; }
    e->mask = mask; e->m = m;
    return e;
}
static void build_from_vw(int m, const double VW[NMAX][NMAX], const double *s, double out[NMAX][NMAX]) {
    int k = m - 1;
    for (int i = 0; i < m; i++) {
        double r[NMAX]; for (int l = 0; l < k; l++) r[l] = VW[i][l] * s[l];
        for (int j = 0; j <= i; j++) { double t = 0; for (int l = 0; l < k; l++) t += r[l] * VW[j][l]; out[i][j] = out[j][i] = t; }
    }
}
/* setup of the QPB quadratic: computes Sp, Tp, trST for current Ar, Br */
static double setup_qpb(work_t *w, int m) {
    int k = m - 1;
    double L[NMAX][NMAX], s[NMAX], t[NMAX];
    int sol[NMAX];
    const eig_t *ea = get_eig(w, w->ea, w->maskF, m, w->Ar);
    const eig_t *eb = get_eig(w, w->eb, w->maskL, m, w->Br);
    for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) L[i][j] = ea->lam[i] * eb->lam[j];
    double pb = lap(k, L, sol, s, t);   /* <lambda, mu>_-  and its dual s, t */
    build_from_vw(m, ea->VW, s, w->Sp);
    build_from_vw(m, eb->VW, t, w->Tp);
    w->trST = pb;
    return pb;
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
static int opt_incr = 1;   /* incremental update of Ar P Br between Frank-Wolfe iterations */
static void applyMperm(work_t *w, int m, const int *p, double out[NMAX][NMAX]) {
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
    setup_qpb(w, m);
    w->apb_ok = 0;
    double inv = 1.0 / m;
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->X[i][j] = inv;
    {   /* X0 = ee^T/m: Ar X0 Br = (Ar e)(e^T Br)/m, and Sp e = 0, e^T Tp = 0 */
        double ra[NMAX], cb[NMAX];
        for (int i = 0; i < m; i++) { double t = 0; for (int k = 0; k < m; k++) t += w->Ar[i][k]; ra[i] = t; }
        for (int j = 0; j < m; j++) { double t = 0; for (int k = 0; k < m; k++) t += w->Br[k][j]; cb[j] = t; }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->MX[i][j] = ra[i] * cb[j] * inv;
    }
    double f = dotm(m, w->X, w->MX) + dotm(m, w->Cr, w->X) + w->trST;
    double best = -INFINITY, u[NMAX], v[NMAX];
    int p[NMAX];
    double hist_best_at[8]; int it;
    for (it = 0; it < nfw; it++) {
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->G[i][j] = 2 * w->MX[i][j] + w->Cr[i][j];
        double lapv;
        if (opt_lap == 0 || (opt_lap == 2 && it == 0)) lapv = lap(m, w->G, p, u, v);
        else {
            if (opt_lap == 1) for (int j = 0; j < m; j++) { double mn = INFINITY; for (int i = 0; i < m; i++) if (w->G[i][j] < mn) mn = w->G[i][j]; v[j] = mn; }
            lapv = lap_jv(m, w->G, p, u, v, 1);
        }
        w->nlap++;
        double gx = dotm(m, w->G, w->X);
        double z = f - gx + lapv + K;
        if (z > best) {
            best = z;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) w->U[i][j] = w->G[i][j] - u[i] - v[j];
            /* shift: we store z without K? no: z includes K */
        }
        if (best > target) { it++; break; }
        applyMperm(w, m, p, w->MP);
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
        double gd = lapv - gx;            /* <G, P - X> = <G,P> - <G,X>, <G,P> = lap value */
        double q = 0;
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
            double d = ((p[i] == j) ? 1.0 : 0.0) - w->X[i][j];
            double md = w->MP[i][j] - w->MX[i][j];
            q += d * md;
        }
        double alpha = (q > 1e-12) ? -gd / (2 * q) : 1.0;
        if (alpha > 1) alpha = 1; if (alpha <= 0) { it++; break; }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
            double d = ((p[i] == j) ? 1.0 : 0.0) - w->X[i][j];
            w->X[i][j] += alpha * d;
            w->MX[i][j] += alpha * (w->MP[i][j] - w->MX[i][j]);
        }
        f += alpha * gd + alpha * alpha * q;
        /* termination rule of Brixius-Anstreicher: stop at NFW2 if the QP
         * value f(X_k) is already below the incumbent (cannot fathom) */
        if (it + 1 >= nfw2 && f + K < target) { it++; break; }
    }
    (void)hist_best_at;
    static int trace = -1; if (trace < 0) trace = getenv("QPB_TRACE") != NULL;
    w->nfw += it; if (trace) fprintf(stderr, "f(X)=%.3f z=%.3f\n", f + K, best);
    return best;
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

static void eval_node(const node_t *nd, work_t *w, eval_t *ev, int nfw, int nfw2, int who) {
    int m = 0, mc = 0;
    w->maskF = w->maskL = 0;
    for (int i = 0; i < N; i++) if (nd->rc[i] < 0) { ev->F[m++] = i; w->maskF |= 1u << i; }
    for (int j = 0; j < N; j++) if (nd->cr[j] < 0) { ev->L[mc++] = j; w->maskL |= 1u << j; }
    ev->m = m;
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
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        int i = ev->F[a], j = ev->L[b]; double s = 0;
        for (int k = 0; k < N; k++) if (nd->rc[k] >= 0) s += A[i][k] * B[j][nd->rc[k]];
        w->Cr[a][b] = 2 * s + C0[i][j];
    }
    double target = prune_thr(g_ub);   /* integer data: fathom if bound > ub - 1 */
    if (m <= 1) {
        double c = K + (m == 1 ? w->Ar[0][0] * w->Br[0][0] + w->Cr[0][0] : 0);
        ev->z = c; if (m == 1) w->U[0][0] = 0;
        if (c < g_ub) { int p[NMAX]; for (int i = 0; i < N; i++) p[i] = nd->rc[i]; if (m == 1) p[ev->F[0]] = ev->L[0]; report_solution(p, c, who); }
        return;
    }
    double ubf = INFINITY; int bp[NMAX];
    ev->z = qpb_fw(w, m, K, target, nfw, nfw2, &ubf, bp);
    if (ubf < g_ub) {
        int p[NMAX]; for (int i = 0; i < N; i++) p[i] = nd->rc[i];
        for (int a = 0; a < m; a++) p[ev->F[a]] = ev->L[bp[a]];
        report_solution(p, ubf, who);
    }
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
static int branch(const node_t *nd, work_t *w, eval_t *ev, child_t *ch, int who) {
    int m = ev->m; double z = ev->z, thr = prune_thr(g_ub);
    static __thread int idxA[GMAX], idxB[GMAX];
    int nA = opt_sym ? stab(&GA, nd->rc, idxA) : 1;   /* acts on rows: reduces column branching */
    int nB = opt_sym ? stab(&GB, nd->cr, idxB) : 1;   /* acts on columns: reduces row branching */
    /* effective U with orbit max (if an element of an orbit is eliminated, the whole orbit is) */
    /* row branching: fix row a, children over columns b (orbits of GB stabilizer on columns) */
    int best_kind = -1, best_line = -1, best_cnt = 1 << 30; double best_sum = -1;
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
    if (allow_row) for (int a = 0; a < m; a++) {
        int cnt = 0; double s = 0;
        for (int b = 0; b < m; b++) if (repCol[b]) { if (z + Ueff[a][b] <= thr) cnt++; s += Ueff[a][b]; }
        if (cnt < best_cnt || (cnt == best_cnt && s > best_sum)) { best_cnt = cnt; best_sum = s; best_kind = 0; best_line = a; }
    }
    if (allow_col) for (int b = 0; b < m; b++) {
        int cnt = 0; double s = 0;
        for (int a = 0; a < m; a++) if (repRow[a]) { if (z + Ueff[a][b] <= thr) cnt++; s += Ueff[a][b]; }
        if (cnt < best_cnt || (cnt == best_cnt && s > best_sum)) { best_cnt = cnt; best_sum = s; best_kind = 1; best_line = b; }
    }
    /* Rules 3 and 4 (Brixius-Anstreicher 2001): prospective bounds on the
     * NBEST rows and NBEST columns with the largest sums of U (Rule 1 order). */
    const rule_t *R = rule_at(nd->depth);
    double zpro[NMAX]; int have_pro = 0;
    if ((R->rule == 3 || R->rule == 4) && m > 2) {
        typedef struct { int kind, line; double s; } cand_t;
        cand_t rows[NMAX], cols[NMAX], cands[2 * NMAX]; int nr = 0, ncl = 0, nc = 0;
        if (allow_row) for (int a = 0; a < m; a++) { double s = 0; for (int b = 0; b < m; b++) if (repCol[b]) s += Ueff[a][b]; rows[nr++] = (cand_t){0, a, s}; }
        if (allow_col) for (int b = 0; b < m; b++) { double s = 0; for (int a = 0; a < m; a++) if (repRow[a]) s += Ueff[a][b]; cols[ncl++] = (cand_t){1, b, s}; }
        for (int x = 0; x < nr; x++) for (int y = x + 1; y < nr; y++) if (rows[y].s > rows[x].s) { cand_t t = rows[x]; rows[x] = rows[y]; rows[y] = t; }
        for (int x = 0; x < ncl; x++) for (int y = x + 1; y < ncl; y++) if (cols[y].s > cols[x].s) { cand_t t = cols[x]; cols[x] = cols[y]; cols[y] = t; }
        for (int x = 0; x < nr && x < R->nbest; x++) cands[nc++] = rows[x];
        for (int x = 0; x < ncl && x < R->nbest; x++) cands[nc++] = cols[x];
        work_t *w2 = malloc(sizeof(work_t)); w2->nlap = w2->nfw = 0;
        double bestscore = -INFINITY; int bk = best_kind, bl = best_line;
        double zline[NMAX];
        /* child (a,b) appears in the candidate list of row a and of column b:
         * evaluate it once (the evaluation is deterministic) */
        static __thread double memo_z[NMAX][NMAX], memo_v[NMAX][NMAX];
        static __thread unsigned char memo_ok[NMAX][NMAX];
        memset(memo_ok, 0, sizeof memo_ok);
        for (int c = 0; c < nc; c++) {
            double score = 0;
            for (int o = 0; o < m; o++) {
                int a = cands[c].kind == 0 ? cands[c].line : o, b = cands[c].kind == 0 ? o : cands[c].line;
                zline[o] = INFINITY;
                if (cands[c].kind == 0 ? !repCol[b] : !repRow[a]) continue;
                double zij, vij = 0;
                if (z + Ueff[a][b] > thr) zij = z + Ueff[a][b];      /* eliminated child: bound from U */
                else if (opt_memo && memo_ok[a][b]) { zij = memo_z[a][b]; vij = memo_v[a][b]; }
                else {
                    node_t chn = *nd; chn.rc[ev->F[a]] = ev->L[b]; chn.cr[ev->L[b]] = ev->F[a]; chn.depth++;
                    eval_t e2; eval_node(&chn, w2, &e2, R->nfw3, R->nfw3, who);
                    zij = fmax(e2.z, z + Ueff[a][b]);
                    if (R->rule == 4 && e2.m > 1) {
                        for (int x = 0; x < e2.m; x++) { double rs = 0, cs = 0;
                            for (int y = 0; y < e2.m; y++) { rs += w2->U[x][y]; cs += w2->U[y][x]; }
                            if (rs > vij) vij = rs; if (cs > vij) vij = cs; }
                    }
                    memo_z[a][b] = zij; memo_v[a][b] = vij; memo_ok[a][b] = 1;
                }
                zline[o] = zij;
                score += (R->rule == 4) ? (m - 1) * fmin(zij, 1e15) + vij : fmin(zij, 1e15);
            }
            if (score > bestscore) { bestscore = score; bk = cands[c].kind; bl = cands[c].line; memcpy(zpro, zline, sizeof(double) * m); have_pro = 1; }
        }
        w->nlap += w2->nlap; w->nfw += w2->nfw; free(w2);
        best_kind = bk; best_line = bl;
    }
    int nch = 0;
    for (int o = 0; o < m; o++) {
        int a = best_kind == 0 ? best_line : o, b = best_kind == 0 ? o : best_line;
        if (best_kind == 0 ? !repCol[b] : !repRow[a]) continue;
        double lb = z + Ueff[a][b];
        if (have_pro && zpro[o] > lb) lb = zpro[o];
        if (lb > thr) continue;
        ch[nch++] = (child_t){ev->F[a], ev->L[b], lb};
    }
    qsort(ch, nch, sizeof(child_t), cmp_child);
    return nch;
}

/* ---------- serial DFS ---------- */
static void dfs(node_t *nd, work_t *w, stats_t *st, int who) {
    eval_t ev;
    double t = now();
    { const rule_t *R = rule_at(nd->depth); eval_node(nd, w, &ev, R->nfw1, R->nfw2, who); }
    st->t_bound += now() - t;
    st->nodes++; st->nodes_depth[nd->depth]++;
    g_nodes_all++; g_depth_all[nd->depth]++; if ((g_nodes_all & 255) == 0) write_progress(0);
    if (ev.m <= 1 || ev.z > prune_thr(g_ub)) { st->fathom_at_bound++; return; }
    child_t ch[NMAX];
    int nch = branch(nd, w, &ev, ch, who);
    int d0 = nd->depth;
    for (int c = 0; c < nch; c++) {
        if (ch[c].lb > prune_thr(g_ub)) continue;
        g_pos[d0] = c + 1; g_npos[d0] = nch; g_nextlb[d0] = c + 1 < nch ? ch[c + 1].lb : INFINITY; g_curdepth = d0 + 1;
        nd->rc[ch[c].i] = ch[c].j; nd->cr[ch[c].j] = ch[c].i; nd->depth++;
        dfs(nd, w, st, who);
        nd->depth--; nd->rc[ch[c].i] = -1; nd->cr[ch[c].j] = -1;
    }
    g_curdepth = d0;
}

/* ---------- Knuth estimator ---------- */
static void estimate(int dives, unsigned long long seed) {
    work_t *w = calloc(1, sizeof(work_t));
    rng_t r = {seed * 2654435761ULL + 1};
    double sumN = 0, sumT = 0, sumN2 = 0; double t0 = now();
    long totnodes = 0;
    for (int d = 0; d < dives; d++) {
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
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

typedef struct { int id; stats_t st; work_t *w; } wk_t;

static void *worker(void *arg) {
    wk_t *me = arg;
    typedef struct { node_t nd; double lb; } sitem_t;
    sitem_t *stack = malloc(sizeof(sitem_t) * NMAX * NMAX * 2);
    for (;;) {
        pthread_mutex_lock(&pool_lock);
        while (pool_n == 0 && pool_busy > 0) pthread_cond_wait(&pool_cond, &pool_lock);
        if (pool_n == 0 && pool_busy == 0) { pthread_cond_broadcast(&pool_cond); pthread_mutex_unlock(&pool_lock); break; }
        /* take deepest node (ties: best bound) */
        int bi = 0;
        for (int k = 1; k < pool_n; k++)
            if (pool[k].nd.depth > pool[bi].nd.depth || (pool[k].nd.depth == pool[bi].nd.depth && pool[k].lb < pool[bi].lb)) bi = k;
        sitem_t it = {pool[bi].nd, pool[bi].lb}; pool[bi] = pool[--pool_n];
        pool_busy++;
        pthread_mutex_unlock(&pool_lock);

        double tstart = now(); int sp = 0; stack[sp++] = it;
        long nodes0 = me->st.nodes;
        while (sp > 0) {
            if (now() - tstart > opt_tmax) {
                pthread_mutex_lock(&pool_lock);
                int need = pool_n < 4 * opt_threads;
                if (need) {
                    /* give back the shallowest half of the stack (bottom entries) */
                    int give = sp > 1 ? sp / 2 : 0;
                    for (int k = 0; k < give; k++) { pool_push(&stack[k].nd, stack[k].lb); nodes_returned++; }
                    memmove(stack, stack + give, sizeof(sitem_t) * (sp - give)); sp -= give;
                    pthread_cond_broadcast(&pool_cond);
                }
                pthread_mutex_unlock(&pool_lock);
                tstart = now();
            }
            sitem_t cur = stack[--sp];
            if (cur.lb > prune_thr(g_ub)) continue;
            eval_t ev; double t = now();
            { const rule_t *R = rule_at(cur.nd.depth); eval_node(&cur.nd, me->w, &ev, R->nfw1, R->nfw2, me->id); }
            me->st.t_bound += now() - t;
            me->st.nodes++; me->st.nodes_depth[cur.nd.depth]++;
            g_nodes_all++; __atomic_fetch_add(&g_depth_all[cur.nd.depth], 1, __ATOMIC_RELAXED);
            if (me->id == 0 && (me->st.nodes & 255) == 0) write_progress(0);
            if (ev.m <= 1 || ev.z > prune_thr(g_ub)) continue;
            child_t ch[NMAX]; int nch = branch(&cur.nd, me->w, &ev, ch, me->id);
            for (int c = nch - 1; c >= 0; c--) { /* push worst first so best is explored first */
                sitem_t s; s.nd = cur.nd; s.nd.rc[ch[c].i] = ch[c].j; s.nd.cr[ch[c].j] = ch[c].i; s.nd.depth++; s.lb = ch[c].lb;
                stack[sp++] = s;
            }
        }
        pthread_mutex_lock(&pool_lock);
        pool_busy--; tasks_done++;
        if (g_log) { fprintf(g_log, "{\"t\":%.3f,\"event\":\"task_done\",\"worker\":%d,\"root_depth\":%d,\"nodes\":%ld,\"pool\":%d}\n", now() - g_t0, me->id, it.nd.depth, me->st.nodes - nodes0, pool_n); fflush(g_log); }
        pthread_cond_broadcast(&pool_cond);
        pthread_mutex_unlock(&pool_lock);
    }
    free(stack);
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
        else if (!strcmp(argv[a], "-nfw")) opt_nfw = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-dives")) dives = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-q")) opt_q = atof(argv[++a]);
        else if (!strcmp(argv[a], "-noincr")) opt_incr = 0;
        else if (!strcmp(argv[a], "-noecache")) opt_ecache = 0;
        else if (!strcmp(argv[a], "-nomemo")) opt_memo = 0;
        else if (!strcmp(argv[a], "-nofastmm")) opt_fastmm = 0;
        else if (!strcmp(argv[a], "-nosimdlap")) opt_simdlap = 0;
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
        else if (!strcmp(argv[a], "-lap")) opt_lap = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-strat")) strat_name = argv[++a][0];
        else if (!strcmp(argv[a], "-nointprune")) opt_intprune = 0;
        else if (!strcmp(argv[a], "-prog")) g_prog = argv[++a];
        else if (!strcmp(argv[a], "-v")) opt_verbose = 1;
    }
    if (logfn) g_log = fopen(logfn, "w");
    set_strategy(strat_name);
    g_t0 = now();
    compute_aut(A, &GA); compute_aut(B, &GB);
    if (opt_verbose) fprintf(stderr, "n=%d |Aut(A)|=%d |Aut(B)|=%d\n", N, GA.n, GB.n);
    int p[NMAX];
    double th0 = now();
    double hv = tabu(tabu_iters, seed, p);
    double t_heur = now() - th0;
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
    g_ub = INFINITY; report_solution(p, hv, -1);
    if (!isnan(ub0) && ub0 < g_ub) g_ub = ub0;     /* external bound (e.g. known optimum + 1) */
    if (!strcmp(mode, "tabu")) { printf("{\"tabu\":%.0f,\"t\":%.2f}\n", hv, t_heur); return 0; }
    if (!strcmp(mode, "root")) {
        work_t *w = calloc(1, sizeof(work_t)); node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
        eval_t ev; double t = now();
        double save = g_ub; g_ub = INFINITY;
        eval_node(&nd, w, &ev, opt_nfw, opt_nfw, 0); g_ub = save;
        printf("{\"root_qpb\":%.2f,\"pb\":%.2f,\"fw_iters\":%ld,\"t\":%.4f,\"ub\":%.0f}\n", ev.z, w->trST, w->nfw, now() - t, (double)g_ub);
        return 0;
    }
    if (!strcmp(mode, "estimate")) { estimate(dives, seed); return 0; }
    double t0 = now();
    stats_t tot = {0}; long lap_tot = 0, fw_tot = 0;
    if (!strcmp(mode, "solve")) {
        work_t *w = calloc(1, sizeof(work_t));
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
        dfs(&nd, w, &tot, 0);
        lap_tot = w->nlap; fw_tot = w->nfw;
    } else if (!strcmp(mode, "par")) {
        node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
        pool_push(&nd, -INFINITY);
        pthread_t th[64]; wk_t wk[64];
        for (int k = 0; k < opt_threads; k++) { memset(&wk[k], 0, sizeof(wk_t)); wk[k].id = k; wk[k].w = calloc(1, sizeof(work_t)); pthread_create(&th[k], NULL, worker, &wk[k]); }
        for (int k = 0; k < opt_threads; k++) {
            pthread_join(th[k], NULL);
            tot.nodes += wk[k].st.nodes; tot.fathom_at_bound += wk[k].st.fathom_at_bound; tot.t_bound += wk[k].st.t_bound;
            for (int d = 0; d <= N; d++) tot.nodes_depth[d] += wk[k].st.nodes_depth[d];
            lap_tot += wk[k].w->nlap; fw_tot += wk[k].w->nfw;
        }
    }
    double el = now() - t0;
    write_progress(1);
    printf("{\"mode\":\"%s\",\"strategy\":\"%c\",\"n\":%d,\"opt\":%.0f,\"heuristic\":%.0f,\"t_heur\":%.2f,\"nodes\":%ld,\"laps\":%ld,\"fw_iters\":%ld,"
           "\"wall_s\":%.2f,\"cpu_bound_s\":%.2f,\"threads\":%d,\"tasks\":%ld,\"symA\":%d,\"symB\":%d,\"perm\":[",
           mode, strat_name, N, (double)g_ub, hv, t_heur, tot.nodes, lap_tot, fw_tot, el, tot.t_bound,
           strcmp(mode, "par") ? 1 : opt_threads, tasks_done, GA.n, GB.n);
    for (int i = 0; i < N; i++) printf("%d%s", g_best[i] + 1, i < N - 1 ? "," : "");
    printf("],\"nodes_depth\":[");
    int last = 0; for (int d = 0; d <= N; d++) if (tot.nodes_depth[d]) last = d;
    for (int d = 0; d <= last; d++) printf("%ld%s", tot.nodes_depth[d], d < last ? "," : "");
    printf("]}\n");
    if (g_log) fclose(g_log);
    return 0;
}
