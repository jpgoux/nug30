/* Certified DNN lower bound for the QAP (Zhao, Karisch, Rendl, Wolkowicz 1998),
 * solved by the ADMM of Oliveira, Wolkowicz and Xu (2018), certified at any
 * iteration from the multiplier as in Brixius et al. (technical report, October 2026):
 *
 *   for any Z and lambda >= lambda_max(V' Z V):
 *     cost(perm) >= s + GL(L + Z) - (m + 1) lambda
 *
 * where GL is the two-level Gilmore-Lawler bound on the lifted matrix.
 * Written from the report's description; the code is ours.
 *
 * Subproblem: m facilities, m locations,
 *   cost(pi) = s + sum_{i,k} a_ik b_{pi(i) pi(k)} + sum_i c_{i pi(i)},  A, B symmetric.
 * Lifted index of x_ij (facility i at location j): p = 1 + i m + j, N = m^2 + 1.
 * Face basis V = [ (1, e/m)/sqrt2 | 0 ; Vm (x) Vm ], K = (m-1)^2 + 1, with Vm an
 * orthonormal basis of e-perp; products with V cost O(m^5) through two "sandwiches".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "dnn.h"
#include <time.h>
static double tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
double g_dnn_prof[8];

void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, int *, const int *, int *);
void ssyevd_(const char *, const char *, const int *, float *, const int *, float *, float *, const int *, int *, const int *, int *);
void spotrf_(const char *, const int *, float *, const int *, int *);
void dpotrf_(const char *, const int *, double *, const int *, int *);

/* ---------- dense LAP (shortest augmenting paths, Hungarian with potentials) ---------- */
/* returns the optimal value of min sum c[i][perm(i)], c row-major n x n (n <= DNN_MMAX) */
static double lap_dense(int n, const double *c, int ld) {
    double u[DNN_MMAX + 1], v[DNN_MMAX + 1], minv[DNN_MMAX + 1];
    int p[DNN_MMAX + 1], way[DNN_MMAX + 1]; char used[DNN_MMAX + 1];
    for (int j = 0; j <= n; j++) { u[j] = v[j] = 0; p[j] = 0; way[j] = 0; }
    for (int i = 1; i <= n; i++) {
        p[0] = i; int j0 = 0;
        for (int j = 0; j <= n; j++) { minv[j] = INFINITY; used[j] = 0; }
        do {
            used[j0] = 1; int i0 = p[j0], j1 = 0; double delta = INFINITY;
            const double *ci = c + (long)(i0 - 1) * ld;
            for (int j = 1; j <= n; j++) if (!used[j]) {
                double cur = ci[j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    double val = 0;
    for (int j = 1; j <= n; j++) val += c[(long)(p[j] - 1) * ld + (j - 1)];
    return val;
}

/* ---------- workspace ---------- */
typedef struct {
    int m, N, K, q;          /* q = m - 1 */
    double *Vm;              /* m x q, orthonormal basis of e-perp (row-major) */
    float *Vmf, *VmTf, *f1, *f2, *f3, *f4; int fl;   /* fl: face products in single precision (ADMM steps only) */
    double *VmT;             /* q x m */
    double *t1, *t2;         /* scratch, N*N each */
    double *W, *evec, *eval, *work; int *iwork; int lwork, liwork;
    float *Wf, *evalf, *workf; int lworkf, liworkf;
} ws_t;

static ws_t *ws_new(int m) {
    ws_t *w = calloc(1, sizeof(ws_t));
    int q = m - 1, N = m * m + 1, K = q * q + 1;
    w->m = m; w->q = q; w->N = N; w->K = K;
    w->Vm = malloc(sizeof(double) * m * q); w->VmT = malloc(sizeof(double) * m * q);
    /* Householder: H = I - 2 h h'/(h'h), h = e/sqrt(m) - e_{m-1}; H e_{m-1} = e/sqrt(m);
     * the other m-1 columns of H are an orthonormal basis of e-perp */
    double h[DNN_MMAX], hh = 0;
    for (int i = 0; i < m; i++) h[i] = 1.0 / sqrt(m) - (i == m - 1);
    for (int i = 0; i < m; i++) hh += h[i] * h[i];
    for (int i = 0; i < m; i++) for (int a = 0; a < q; a++) {
        double x = (i == a) - 2 * h[i] * h[a] / hh;
        w->Vm[i * q + a] = x; w->VmT[a * m + i] = x;
    }
    w->t1 = malloc(sizeof(double) * N * N); w->t2 = malloc(sizeof(double) * N * N);
    w->Vmf = malloc(sizeof(float) * m * q); w->VmTf = malloc(sizeof(float) * m * q);
    for (int t = 0; t < m * q; t++) { w->Vmf[t] = (float)w->Vm[t]; w->VmTf[t] = (float)w->VmT[t]; }
    w->f1 = malloc(sizeof(float) * N * N); w->f2 = malloc(sizeof(float) * N * N); w->f3 = malloc(sizeof(float) * N * N); w->f4 = malloc(sizeof(float) * (long)N * N);
    w->W = malloc(sizeof(double) * K * K); w->evec = malloc(sizeof(double) * K * K); w->eval = malloc(sizeof(double) * K);
    int lw = -1, liw = -1, info, iq; double wq;
    dsyevd_("V", "L", &K, w->evec, &K, w->eval, &wq, &lw, &iq, &liw, &info);
    w->lwork = (int)wq + 1; w->liwork = iq + 1;
    w->work = malloc(sizeof(double) * w->lwork); w->iwork = malloc(sizeof(int) * w->liwork);
    w->Wf = malloc(sizeof(float) * K * K); w->evalf = malloc(sizeof(float) * K);
    float wqf; lw = -1; liw = -1;
    ssyevd_("V", "L", &K, w->Wf, &K, w->evalf, &wqf, &lw, &iq, &liw, &info);
    w->lworkf = (int)wqf + 1; w->liworkf = iq + 1;
    w->workf = malloc(sizeof(float) * w->lworkf);
    if (w->liworkf > w->liwork) { free(w->iwork); w->liwork = w->liworkf; w->iwork = malloc(sizeof(int) * w->liwork); }
    return w;
}
static void ws_free(ws_t *w) {
    free(w->Vm); free(w->VmT); free(w->Vmf); free(w->VmTf); free(w->f1); free(w->f2); free(w->f3); free(w->f4); free(w->t1); free(w->t2); free(w->W); free(w->evec); free(w->eval);
    free(w->work); free(w->iwork); free(w->Wf); free(w->evalf); free(w->workf); free(w);
}

/* out[r] = P' in[r] P for nr row-blocks, in[r] p x p, P p x q, out[r] q x q */
static void sandwich(int nr, int p, int q, const double *P, const double *in, double *out, double *tmp) {
    /* tmp[(r,k)][b] = sum_l in[(r,k)][l] P[l][b] */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr * p, q, p, 1.0, in, p, P, q, 0.0, tmp, q);
    for (int r = 0; r < nr; r++)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, q, q, p, 1.0, P, q, tmp + (long)r * p * q, q, 0.0, out + (long)r * q * q, q);
}
static void sandwichf(int nr, int p, int q, const float *P, const float *in, float *out, float *tmp) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr * p, q, p, 1.0f, in, p, P, q, 0.0f, tmp, q);
    for (int r = 0; r < nr; r++)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, q, q, p, 1.0f, P, q, tmp + (long)r * p * q, q, 0.0f, out + (long)r * q * q, q);
}
static void transpose(int r, int c, const double *in, double *out) {
    for (int i0 = 0; i0 < r; i0 += 32) for (int j0 = 0; j0 < c; j0 += 32)
        for (int i = i0; i < i0 + 32 && i < r; i++) for (int j = j0; j < j0 + 32 && j < c; j++) out[(long)j * r + i] = in[(long)i * c + j];
}

/* R = V' M V, M N x N symmetric, R K x K */
static void face_in(ws_t *w, const double *M, double *R) {
    int m = w->m, q = w->q, N = w->N, K = w->K, m2 = m * m, q2 = q * q;
    const double s2 = sqrt(0.5), um = 1.0 / m;
    /* Mxx is M[1.., 1..]: copy to contiguous m2 x m2 */
    double *X = w->t1, *Y = w->t2;
    for (int i = 0; i < m2; i++) memcpy(X + (long)i * m2, M + (long)(i + 1) * N + 1, sizeof(double) * m2);
    /* Mxx W: rows i (m2), each row an m x m matrix -> q x q */
    double *A1 = malloc(sizeof(double) * m2 * q2), *tmp = w->fl ? NULL : malloc(sizeof(double) * (long)m2 * m * q);
    if (w->fl) {
        float *Xf = w->f1, *Af = w->f2, *Tf = w->f3, *Yf = w->f4;
        for (long t = 0; t < (long)m2 * m2; t++) Xf[t] = (float)X[t];
        sandwichf(m2, m, q, w->Vmf, Xf, Af, Tf);
        for (long t = 0; t < (long)m2 * q2; t++) A1[t] = Af[t];
        for (int i0 = 0; i0 < m2; i0 += 32) for (int j0 = 0; j0 < q2; j0 += 32)
            for (int i = i0; i < i0 + 32 && i < m2; i++) for (int j = j0; j < j0 + 32 && j < q2; j++) Yf[(long)j * m2 + i] = Af[(long)i * q2 + j];
        sandwichf(q2, m, q, w->Vmf, Yf, Xf, Tf);
        for (long t = 0; t < (long)q2 * q2; t++) X[t] = Xf[t];
    } else {
    sandwich(m2, m, q, w->Vm, X, A1, tmp);                       /* A1 = Mxx W : m2 x q2 */
    transpose(m2, q2, A1, Y);                                    /* Y = (Mxx W)' : q2 x m2 */
    sandwich(q2, m, q, w->Vm, Y, X, tmp);                        /* X = W' Mxx W : q2 x q2 */
    }
    /* u' Mxx u, u' Mxx W (= colsum of A1 / m), M0x W, M0x u */
    double uMu = 0, m0u = 0;
    double *uMW = calloc(q2, sizeof(double)), *m0W = calloc(q2, sizeof(double));
    for (int i = 0; i < m2; i++) for (int b = 0; b < q2; b++) uMW[b] += A1[(long)i * q2 + b];
    for (int b = 0; b < q2; b++) uMW[b] *= um;
    { double rs = 0; for (int i = 0; i < m2; i++) for (int j = 0; j < m2; j++) rs += M[(long)(i + 1) * N + 1 + j]; uMu = rs * um * um; }
    for (int i = 0; i < m2; i++) m0u += M[1 + i] * um;
    /* M0x W: M0x as an m x m matrix, Vm' M0x Vm */
    { double t[DNN_MMAX * DNN_MMAX]; sandwich(1, m, q, w->Vm, M + 1, m0W, t); }
    R[0] = 0.5 * (M[0] + 2 * m0u + uMu);
    for (int b = 0; b < q2; b++) { double x = s2 * (m0W[b] + uMW[b]); R[1 + b] = x; R[(long)(1 + b) * K] = x; }
    for (int a = 0; a < q2; a++) memcpy(R + (long)(1 + a) * K + 1, X + (long)a * q2, sizeof(double) * q2);
    free(A1); free(tmp); free(uMW); free(m0W);
}

/* Y = V R V', R K x K symmetric, Y N x N */
static void face_out(ws_t *w, const double *R, double *Yo) {
    int m = w->m, q = w->q, N = w->N, K = w->K, m2 = m * m, q2 = q * q;
    const double s2 = sqrt(0.5), um = 1.0 / m;
    double *X = w->t1, *Y = w->t2;
    for (int a = 0; a < q2; a++) memcpy(X + (long)a * q2, R + (long)(1 + a) * K + 1, sizeof(double) * q2);
    double *B1 = w->fl ? NULL : malloc(sizeof(double) * q2 * m2), *tmp = w->fl ? NULL : malloc(sizeof(double) * (long)m2 * m * q);
    if (w->fl) {
        float *Xf = w->f1, *Bf = w->f2, *Tf = w->f3, *Yf = w->f4;
        for (long t = 0; t < (long)q2 * q2; t++) Xf[t] = (float)X[t];
        sandwichf(q2, q, m, w->VmTf, Xf, Bf, Tf);
        for (int i0 = 0; i0 < q2; i0 += 32) for (int j0 = 0; j0 < m2; j0 += 32)
            for (int i = i0; i < i0 + 32 && i < q2; i++) for (int j = j0; j < j0 + 32 && j < m2; j++) Yf[(long)j * q2 + i] = Bf[(long)i * m2 + j];
        sandwichf(m2, q, m, w->VmTf, Yf, Xf, Tf);
        for (long t = 0; t < (long)m2 * m2; t++) X[t] = Xf[t];
    } else {
    sandwich(q2, q, m, w->VmT, X, B1, tmp);                      /* B1 = Rr W' : q2 x m2 */
    transpose(q2, m2, B1, Y);                                    /* Y = W Rr : m2 x q2 */
    sandwich(m2, q, m, w->VmT, Y, X, tmp);                       /* X = W Rr W' : m2 x m2 */
    }
    double r00 = R[0], Wr0[DNN_MMAX * DNN_MMAX];
    { double t[DNN_MMAX * DNN_MMAX]; sandwich(1, q, m, w->VmT, R + 1, Wr0, t); }   /* W r0 as m x m */
    Yo[0] = 0.5 * r00;
    for (int i = 0; i < m2; i++) { double x = 0.5 * r00 * um + s2 * Wr0[i]; Yo[1 + i] = x; Yo[(long)(1 + i) * N] = x; }
    for (int i = 0; i < m2; i++) {
        double *yr = Yo + (long)(i + 1) * N + 1; const double *xr = X + (long)i * m2;
        double ci = 0.5 * r00 * um * um + s2 * um * Wr0[i];
        for (int j = 0; j < m2; j++) yr[j] = xr[j] + ci + s2 * um * Wr0[j];
    }
    free(B1); free(tmp);
}

/* ---------- certificate ---------- */
/* two-level Gilmore-Lawler bound of min <M, y y'> over lifted permutations */
static double gl2(int m, const double *M) {
    int N = m * m + 1;
    double c[DNN_MMAX * DNN_MMAX], blk[DNN_MMAX * DNN_MMAX];
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
        int p = 1 + i * m + j, r = 0;
        for (int k = 0; k < m; k++) { if (k == i) continue;
            int cc = 0; const double *row = M + (long)p * N + 1 + k * m;
            for (int l = 0; l < m; l++) { if (l == j) continue; blk[r * (m - 1) + cc++] = row[l]; }
            r++; }
        double lij = m > 1 ? lap_dense(m - 1, blk, m - 1) : 0;
        c[i * m + j] = 2 * M[p] + M[(long)p * N + p] + lij;
    }
    return M[0] + lap_dense(m, c, m);
}

/* upper bound on lambda_max(S), S n x n symmetric, by Rump's criterion:
 * if floating-point Cholesky of G = d I - S succeeds, lambda_min(G) >= -c with
 * c = gam/(1-gam) tr(G) + 4 (3n + max G_ii) eta, gam = gamma_{n+1}; then lambda_max(S) <= d + c.
 * Returns INFINITY when no d tried works. */
static double rump_lmax(int n, const double *S, double dguess, double *scratch) {
    const double u = DBL_EPSILON / 2, eta = DBL_TRUE_MIN;
    double gam = (n + 1) * u / (1 - (n + 1) * u), al = gam / (1 - gam);
    double d = dguess > 0 ? dguess : 1e-12;
    for (int tries = 0; tries < 40; tries++, d *= 4) {
        double tr = 0, mx = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) scratch[(long)i * n + j] = -S[(long)i * n + j];
        for (int i = 0; i < n; i++) { scratch[(long)i * n + i] += d; tr += scratch[(long)i * n + i]; if (scratch[(long)i * n + i] > mx) mx = scratch[(long)i * n + i]; }
        int info; dpotrf_("L", &n, scratch, &n, &info);
        if (info == 0) {
            double c = al * tr + 4 * (3.0 * n + mx) * eta;
            double lm = d + c; return lm * (1 + 4 * u);
        }
    }
    return INFINITY;
}

/* ---------- the bound ---------- */
double dnn_bound(const dnn_prob_t *P, const dnn_opt_t *o, dnn_res_t *res) {
    int m = P->m;
    memset(res, 0, sizeof *res); res->bound = -INFINITY;
    if (m < 2 || m > DNN_MMAX) return -INFINITY;
    ws_t *w = ws_new(m);
    int N = w->N, K = w->K;
    long NN = (long)N * N, KK = (long)K * K;
    const double u = DBL_EPSILON / 2;
    /* lifted cost, scaled to max |L| = 1 */
    double *L = calloc(NN, sizeof(double));
    double sumabs = 0, mx = 0;
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) for (int k = 0; k < m; k++) for (int l = 0; l < m; l++) {
        double x = 0.5 * (P->A[i * m + k] * P->B[j * m + l] + P->A[k * m + i] * P->B[l * m + j]);
        L[(long)(1 + i * m + j) * N + 1 + k * m + l] = x; sumabs += fabs(x); if (fabs(x) > mx) mx = fabs(x);
    }
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double x = 0.5 * P->C[i * m + j]; L[1 + i * m + j] = x; L[(long)(1 + i * m + j) * N] = x; sumabs += 2 * fabs(x); if (fabs(x) > mx) mx = fabs(x); }
    double sc = mx > 0 ? mx : 1;
    for (long t = 0; t < NN; t++) L[t] /= sc;
    sumabs /= sc;
    double cut = (o->cutoff - P->s) / sc;     /* fathom when GL - margin exceeds cut */
    double beta = o->beta > 0 ? o->beta : 0.1 * m, gstep = o->gamma > 0 ? o->gamma : 1.618;
    /* state */
    double *Y = calloc(NN, sizeof(double)), *Z = calloc(NN, sizeof(double)), *VRV = calloc(NN, sizeof(double));
    double *Mz = calloc(NN, sizeof(double)), *R = calloc(KK, sizeof(double)), *S = calloc(KK, sizeof(double));
    double *scr = malloc(sizeof(double) * KK);
    if (o->Z0) for (long t = 0; t < NN; t++) Z[t] = o->Z0[t] / sc;   /* warm start, given in unscaled units */
    /* start: y = (1, x = 1/m), Y = y y' with gangster entries zero */
    for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) {
        double ya = a ? 1.0 / m : 1, yb = b ? 1.0 / m : 1; Y[(long)a * N + b] = ya * yb; }
    /* gangster mask: (ij),(il) l != j and (ij),(kj) k != i */
    #define GANG(p, r) ((p) > 0 && (r) > 0 && (p) != (r) && ((((p) - 1) / m == ((r) - 1) / m) || (((p) - 1) % m == ((r) - 1) % m)))
    for (int a = 1; a < N; a++) for (int b = 1; b < N; b++) if (GANG(a, b)) Y[(long)a * N + b] = 0;
    int it, every = o->cert_every > 0 ? o->cert_every : 10, maxit = o->maxit > 0 ? o->maxit : 1000;
    double best = -INFINITY, hist[64]; for (int k = 0; k < 64; k++) hist[k] = -INFINITY;
    int sub_p = 0, sub_redo = 0, sub_nf = getenv("DNN_NF") ? atoi(getenv("DNN_NF")) : 4; double sub_sig = 0, sub_a = 0;
    float *sH0 = o->sub ? malloc(sizeof(float) * KK) : NULL, *sQ = o->sub ? malloc(sizeof(float) * KK) : NULL, *sG = o->sub ? malloc(sizeof(float) * KK) : NULL, *sH = o->sub ? malloc(sizeof(float) * KK) : NULL, *sth = o->sub ? malloc(sizeof(float) * K) : NULL;
    for (it = 1; it <= maxit; it++) {
        /* W = V'(Y + Z/beta)V */
        for (long t = 0; t < NN; t++) Mz[t] = Y[t] + Z[t] / beta;
        double tp0 = tnow(); w->fl = o->single; face_in(w, Mz, w->W); double tp1 = tnow(); g_dnn_prof[0] += tp1 - tp0;
        /* eigendecomposition of W */
        int info = 0, nneg = 0;
        int full = !o->sub || it == 1 || sub_p == 0 || (it - 1) % o->sub == 0 || sub_redo;
        if (o->single || !full) for (long t = 0; t < KK; t++) w->Wf[t] = (float)w->W[t];
        if (full) {
            if (o->single) {
                ssyevd_("V", "L", &K, w->Wf, &K, w->evalf, w->workf, &w->lworkf, w->iwork, &w->liwork, &info);
                for (int k = 0; k < K; k++) w->eval[k] = w->evalf[k];
                for (long t = 0; t < KK; t++) w->evec[t] = w->Wf[t];
                if (o->sub) for (long t = 0; t < KK; t++) w->Wf[t] = (float)w->W[t];
            } else {
                memcpy(w->evec, w->W, sizeof(double) * KK);
                dsyevd_("V", "L", &K, w->evec, &K, w->eval, w->work, &w->lwork, w->iwork, &w->liwork, &info);
            }
            if (info) break;
            while (nneg < K && w->eval[nneg] < 0) nneg++;
            for (int k = 0; k < nneg; k++) { double f = sqrt(-w->eval[k]); for (int t = 0; t < K; t++) scr[(long)k * K + t] = w->evec[(long)k * K + t] * f; }
            if (o->sub) {   /* subspace for the next iterations: the nneg + pad lowest eigenvectors */
                sub_p = nneg + 16 + nneg / 4; if (sub_p > K) sub_p = K;
                for (int t = 0; t < K; t++) for (int k = 0; k < sub_p; k++) sQ[(long)t * sub_p + k] = (float)w->evec[(long)k * K + t];
                sub_sig = 1.05 * fabs(w->eval[K - 1]) + 1e-3; sub_a = w->eval[sub_p - 1]; if (sub_a < 0) sub_a = 0; sub_redo = 0;
            }
        } else {
            /* one filtered subspace step and Rayleigh-Ritz: X = (sig I - W) Q, Q = orth(X), H = Q' W Q */
            int pp = sub_p;
            /* Chebyshev filter of degree sub_nf damping [sub_a, sub_sig] (Zhou and Saad 2007) */
            {
                double e = 0.5 * (sub_sig - sub_a), c = 0.5 * (sub_sig + sub_a); if (e <= 0) e = 1e-6;
                float *X0 = sH0, *Y1 = sG;
                memcpy(X0, sQ, sizeof(float) * K * pp);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, pp, K, (float)(1.0 / e), w->Wf, K, sQ, pp, 0.0f, Y1, pp);
                for (long t = 0; t < (long)K * pp; t++) Y1[t] -= (float)(c / e) * X0[t];
                for (int f = 2; f <= sub_nf; f++) {
                    /* Y2 = 2 (W Y1 - c Y1)/e - X0, stored into X0 */
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, pp, K, (float)(2.0 / e), w->Wf, K, Y1, pp, -1.0f, X0, pp);
                    for (long t = 0; t < (long)K * pp; t++) X0[t] -= (float)(2 * c / e) * Y1[t];
                    float *tt = X0; X0 = Y1; Y1 = tt;
                    double nr = 0; for (long t = 0; t < (long)K * pp; t++) nr = fmax(nr, fabs(Y1[t]));
                    if (nr > 1e15) { float sc2 = (float)(1.0 / nr); for (long t = 0; t < (long)K * pp; t++) { Y1[t] *= sc2; X0[t] *= sc2; } }
                }
                if (Y1 != sG) memcpy(sG, Y1, sizeof(float) * K * pp);
            }
            /* Cholesky QR, twice */
            int ok = 1;
            for (int pass = 0; pass < 2 && ok; pass++) {
                cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, pp, K, 1.0f, sG, pp, 0.0f, sH, pp);
                int inf; spotrf_("L", &pp, sH, &pp, &inf);   /* row-major upper = column-major lower */
                if (inf) { ok = 0; break; }
                cblas_strsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, K, pp, 1.0f, sH, pp, sG, pp);
            }
            if (!ok) { sub_redo = 1; it--; continue; }
            memcpy(sQ, sG, sizeof(float) * K * pp);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, pp, K, 1.0f, w->Wf, K, sQ, pp, 0.0f, sG, pp);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pp, pp, K, 1.0f, sQ, pp, sG, pp, 0.0f, sH, pp);
            ssyevd_("V", "L", &pp, sH, &pp, sth, w->workf, &w->lworkf, w->iwork, &w->liwork, &info);
            if (info) { sub_redo = 1; it--; continue; }
            /* Ritz vectors: sG = sQ * U, U rows = eigenvectors of H */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, K, pp, pp, 1.0f, sQ, pp, sH, pp, 0.0f, sG, pp);
            memcpy(sQ, sG, sizeof(float) * K * pp);
            while (nneg < pp && sth[nneg] < 0) nneg++;
            sub_a = sth[pp - 1] > 0 ? sth[pp - 1] : 0;
            for (int k = 0; k < nneg; k++) { double f = sqrt(-(double)sth[k]); for (int t = 0; t < K; t++) scr[(long)k * K + t] = sQ[(long)t * pp + k] * f; }
            if (nneg > pp - 4) sub_redo = 1;   /* the negative space outgrew the subspace */
        }
        double tp2 = tnow(); g_dnn_prof[1] += tp2 - tp1;
        /* R = W - negative part = W + scr' scr (any R keeps the certificate valid: S is recomputed and checked) */
        memcpy(R, w->W, sizeof(double) * KK);
        if (nneg > 0) cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, nneg, 1.0, scr, K, scr, K, 1.0, R, K);
        double tp3 = tnow(); g_dnn_prof[2] += tp3 - tp2;
        face_out(w, R, VRV);
        double tp4 = tnow(); g_dnn_prof[3] += tp4 - tp3;
        int cert = (it % every == 0) || it == maxit;
        double tp5 = tnow();
        if (cert) {
            /* Z' = Z + beta (Y - V R V'), M = L + Z' */
            for (long t = 0; t < NN; t++) Mz[t] = L[t] + Z[t] + beta * (Y[t] - VRV[t]);
            /* symmetrize exactly */
            for (int a = 0; a < N; a++) for (int b = a + 1; b < N; b++) { double x = 0.5 * (Mz[(long)a * N + b] + Mz[(long)b * N + a]); Mz[(long)a * N + b] = Mz[(long)b * N + a] = x; }
            /* S = V' Z' V computed from Z' = M - L */
            double *Zp = w->t1 == NULL ? NULL : NULL; (void)Zp;
            double *Zc = malloc(sizeof(double) * NN); double zf = 0, m1 = 0;
            for (long t = 0; t < NN; t++) { Zc[t] = Mz[t] - L[t]; zf += Zc[t] * Zc[t]; m1 += fabs(Mz[t]); }
            zf = sqrt(zf);
            w->fl = 0; face_in(w, Zc, S);
            free(Zc);
            double lmx = 0; for (int k = 0; k < K; k++) { /* crude guess from the diagonal */ if (S[(long)k * K + k] > lmx) lmx = S[(long)k * K + k]; }
            double lam = rump_lmax(K, S, fmax(lmx, 1e-14 * (zf + 1)), scr);
            /* the face basis is applied in floating point: allowance 64 m u ||Z'||_F (report, 2.3 (i)) */
            lam += 64.0 * m * u * zf;
            double gl = gl2(m, Mz);
            double margin = (m + 1) * fmax(lam, 0) + 8.0 * (m + 2) * u * m1 + (m + 2) * u * sumabs;
            double bnd = gl - (m + 1) * lam - margin + (m + 1) * fmax(lam, 0);   /* gl - (m+1) lam - rounding allowances */
            bnd = gl - (m + 1) * lam - (8.0 * (m + 2) * u * m1 + (m + 2) * u * sumabs);
            (void)margin;
            if (isfinite(bnd) && bnd > best) best = bnd;
            if (o->verbose) fprintf(stderr, "dnn it %4d  cert %.6f (unscaled %.3f)  lam %.2e  nneg %d\n", it, bnd, P->s + bnd * sc, lam, nneg);
            if (best > cut) { it++; break; }
            /* give-up rule: stop when even a generous linear extrapolation of the last 50 iterations
             * (o->giveup times their gain) stays below the cutoff; this only saves time */
            if (o->giveup > 0) {
                int slot = (it / every) % 64; hist[slot] = best;
                int back = 50 / every; if (back < 1) back = 1;
                if (it >= (o->giveup_min > 0 ? o->giveup_min : 100) && isfinite(hist[(slot - back + 64) % 64])) {
                    double g = best - hist[(slot - back + 64) % 64];
                    if (best + o->giveup * fmax(g, 0) < cut) { it++; res->gaveup = 1; break; }
                }
            }
        }
        double tp6 = tnow(); g_dnn_prof[4] += tp6 - tp5;
        /* Y = clip(V R V' - (L + Z)/beta), Y00 = 1, gangster 0, 0 <= Y <= 1 */
        for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) {
            long t = (long)a * N + b; double x = VRV[t] - (L[t] + Z[t]) / beta;
            if (a == 0 && b == 0) x = 1; else if (GANG(a, b)) x = 0; else x = x < 0 ? 0 : (x > 1 ? 1 : x);
            Y[t] = x;
        }
        /* Z = Z + gamma beta (Y - V R V') */
        for (long t = 0; t < NN; t++) Z[t] += gstep * beta * (Y[t] - VRV[t]);
        g_dnn_prof[5] += tnow() - tp6;
    }
    res->iters = it - 1;
    res->bound = best > -INFINITY ? P->s + best * sc : -INFINITY;
    /* rounding of the final rescale: subtract a relative allowance */
    if (isfinite(res->bound)) res->bound -= 4 * u * (fabs(P->s) + fabs(best * sc));
    res->fathomed = res->bound > o->cutoff;
    if (o->Zout) for (long t = 0; t < NN; t++) o->Zout[t] = Z[t] * sc;
    free(sH0); free(sQ); free(sG); free(sH); free(sth);
    free(L); free(Y); free(Z); free(VRV); free(Mz); free(R); free(S); free(scr);
    ws_free(w);
    return res->bound;
}
