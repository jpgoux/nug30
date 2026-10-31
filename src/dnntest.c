/* tests of the certified DNN bound:
 *   dnntest rand m count seed   random instances, bound against enumeration
 *   dnntest root file.dat [maxit] [single]   root bound of a QAPLIB instance
 *   dnntest node file.dat i:j,... cutoff [maxit] [sub] [geo] [refine]   cold bound of a node (0-based fixings), as qap_bb -dnn calls it */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "dnn.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static double brute(int m, const double *A, const double *B, const double *C) {
    int p[DNN_MMAX], c[DNN_MMAX] = {0}; for (int i = 0; i < m; i++) p[i] = i;
    double best = INFINITY;
    for (;;) {
        double v = 0;
        for (int i = 0; i < m; i++) { v += C[i * m + p[i]]; for (int k = 0; k < m; k++) v += A[i * m + k] * B[p[i] * m + p[k]]; }
        if (v < best) best = v;
        int i = 1; while (i < m && c[i] >= i) c[i++] = 0;
        if (i >= m) break;
        int j = (i % 2) ? c[i] : 0; int t = p[i]; p[i] = p[j]; p[j] = t; c[i]++;
    }
    return best;
}


/* reduced problem of a node: fixings loc[k] (or -1), free facilities I and locations J in increasing order */
static int build_node(int n, const double *A, const double *B, const int *loc, double **pa, double **pb, double **pc, double *ps, int *I, int *J) {
    int used[64] = {0}, m = 0, mj = 0;
    for (int k = 0; k < n; k++) if (loc[k] >= 0) used[loc[k]] = 1;
    for (int i = 0; i < n; i++) if (loc[i] < 0) I[m++] = i;
    for (int j = 0; j < n; j++) if (!used[j]) J[mj++] = j;
    double *a = malloc(sizeof(double) * m * m), *b = malloc(sizeof(double) * m * m), *c = calloc(m * m, sizeof(double)), s0 = 0;
    for (int x = 0; x < m; x++) for (int y = 0; y < m; y++) { a[x * m + y] = A[I[x] * n + I[y]]; b[x * m + y] = B[J[x] * n + J[y]]; }
    for (int x = 0; x < m; x++) for (int y = 0; y < m; y++) for (int k = 0; k < n; k++) if (loc[k] >= 0)
        c[x * m + y] += A[I[x] * n + k] * B[J[y] * n + loc[k]] + A[k * n + I[x]] * B[loc[k] * n + J[y]];
    for (int k = 0; k < n; k++) if (loc[k] >= 0) for (int l = 0; l < n; l++) if (loc[l] >= 0) s0 += A[k * n + l] * B[loc[k] * n + loc[l]];
    *pa = a; *pb = b; *pc = c; *ps = s0; return m;
}

int main(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[1], "rand")) {
        int m = atoi(argv[2]), cnt = atoi(argv[3]); srand(atoi(argv[4]));
        int bad = 0; double gsum = 0;
        for (int t = 0; t < cnt; t++) {
            double A[DNN_MMAX * DNN_MMAX], B[DNN_MMAX * DNN_MMAX], C[DNN_MMAX * DNN_MMAX];
            int kind = t % 3;   /* 0 symmetric zero diagonal, 1 with diagonal, 2 non-symmetric with C */
            for (int i = 0; i < m; i++) for (int j = 0; j <= i; j++) {
                double a = rand() % 10, b = rand() % 10;
                if (i == j && kind == 0) a = b = 0;
                A[i * m + j] = A[j * m + i] = a; B[i * m + j] = B[j * m + i] = b;
                if (kind == 2) { A[j * m + i] = rand() % 10; }
            }
            for (int i = 0; i < m * m; i++) C[i] = kind ? rand() % 30 - 5 : 0;
            double opt = brute(m, A, B, C);
            dnn_prob_t P = {m, A, B, C, 0};
            dnn_opt_t o = {0}; o.cutoff = INFINITY; o.maxit = 200 + rand() % 300; o.cert_every = 10; o.single = t % 2;
            dnn_res_t r; dnn_bound(&P, &o, &r);
            if (r.bound > opt + 1e-9) { bad++; printf("VIOLATION t=%d kind=%d opt=%.6f bound=%.9f\n", t, kind, opt, r.bound); }
            gsum += (opt - r.bound) / fmax(1, fabs(opt));
        }
        printf("rand m=%d: %d instances, %d violations, mean relative gap %.4f\n", m, cnt, bad, gsum / cnt);
        return bad != 0;
    }
    if (argc >= 7 && !strcmp(argv[1], "sib")) {   /* dnntest sib file.dat parent_fixings row cutoff j1,j2,...: children (row -> j) warm from the parent, then warm from the last fathomed sibling (its Z permuted) */
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n);
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        int loc[64]; for (int i = 0; i < n; i++) loc[i] = -1;
        for (char *p = argv[3]; *p; ) { int i, j, k; if (sscanf(p, "%d:%d%n", &i, &j, &k) != 2) break; loc[i] = j; p += k; if (*p == ',') p++; }
        int row = atoi(argv[4]); double cut = atof(argv[5]);
        int js[64], nj = 0; for (char *p = argv[6]; *p; ) { int j, k; if (sscanf(p, "%d%n", &j, &k) != 1) break; js[nj++] = j; p += k; if (*p == ',') p++; }
        int I[64], J[64]; double *a, *b, *c, s0; int m = build_node(n, A, B, loc, &a, &b, &c, &s0, I, J);
        int N = m * m + 1; double *Zp = malloc(sizeof(double) * N * N);
        dnn_prob_t P = {m, a, b, c, s0}; dnn_opt_t o = {0}; o.cutoff = cut; o.maxit = 4000; o.single = 1; o.sub = getenv("DNN_SUB") ? atoi(getenv("DNN_SUB")) : 20; o.geo = 1; o.giveup = 5; o.giveup_min = 300; o.lap_auto = 1; o.Zout = Zp;
        dnn_res_t r; double t = now(); dnn_bound(&P, &o, &r);
        printf("parent %s m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[3], m, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        int pa = -1; for (int x = 0; x < m; x++) if (I[x] == row) pa = x;
        int m2 = m - 1, N2 = m2 * m2 + 1;
        double *Zsib = malloc(sizeof(double) * N2 * N2), *Z0 = malloc(sizeof(double) * N2 * N2), *Zout = malloc(sizeof(double) * N2 * N2); int have_sib = 0, jsib = -1;
        long it_par = 0, it_sib = 0; double t_par = 0, t_sib = 0;
        for (int k = 0; k < nj; k++) {
            int jc = js[k]; int pb = -1; for (int y = 0; y < m; y++) if (J[y] == jc) pb = y;
            if (pa < 0 || pb < 0) { printf("child %d:%d not free\n", row, jc); continue; }
            loc[row] = jc; int I2[64], J2[64]; double *a2, *b2, *c2, s2; build_node(n, A, B, loc, &a2, &b2, &c2, &s2, I2, J2); loc[row] = -1;
            /* warm from the parent */
            int map[DNN_MMAX * DNN_MMAX + 1], q = 1; map[0] = 0;
            for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) if (i != pa && j != pb) map[q++] = 1 + i * m + j;
            for (int x = 0; x < N2; x++) for (int y = 0; y < N2; y++) Z0[(long)x * N2 + y] = Zp[(long)map[x] * N + map[y]];
            dnn_prob_t P2 = {m2, a2, b2, c2, s2}; dnn_opt_t o2 = o; o2.Zout = Zout; o2.Z0 = Z0;
            t = now(); dnn_bound(&P2, &o2, &r); double dt = now() - t; it_par += r.iters; t_par += dt;
            printf("child %d:%d warm-parent  m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", row, jc, m2, r.bound, r.fathomed, r.gaveup, r.iters, dt);
            int fath_par = r.fathomed;
            if (have_sib && !getenv("DNN_NOSIB")) {
                /* sibling (row -> jsib) had free locations J minus jsib; here J minus jc: location jc of the sibling plays the role of jsib */
                int posS[64], posC[64]; for (int y = 0; y < 64; y++) posS[y] = posC[y] = -1;
                { int y = 0; for (int x = 0; x < m; x++) if (J[x] != jsib) posS[J[x]] = y++; }
                { int y = 0; for (int x = 0; x < m; x++) if (J[x] != jc) posC[J[x]] = y++; }
                int mp[DNN_MMAX * DNN_MMAX + 1]; mp[0] = 0;
                for (int i = 0; i < m2; i++) for (int y = 0; y < m2; y++) { int lc = J2[y]; int ls = lc == jsib ? jc : lc; mp[1 + i * m2 + y] = 1 + i * m2 + posS[ls]; }
                for (int x = 0; x < N2; x++) for (int y = 0; y < N2; y++) Z0[(long)x * N2 + y] = Zsib[(long)mp[x] * N2 + mp[y]];
                dnn_opt_t o3 = o; o3.Zout = NULL; o3.Z0 = Z0; o3.verbose = getenv("DNN_V") != NULL;
                t = now(); dnn_bound(&P2, &o3, &r); dt = now() - t; it_sib += r.iters; t_sib += dt;
                printf("child %d:%d warm-sibling %d m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", row, jc, jsib, m2, r.bound, r.fathomed, r.gaveup, r.iters, dt);
            } else { it_sib += r.iters; t_sib += dt; }
            if (fath_par) { memcpy(Zsib, Zout, sizeof(double) * N2 * N2); have_sib = 1; jsib = jc; }
            free(a2); free(b2); free(c2);
        }
        printf("total warm-parent: %ld iters %.1f s; warm-sibling chain: %ld iters %.1f s\n", it_par, t_par, it_sib, t_sib);
        return 0;
    }
    if (argc >= 6 && !strcmp(argv[1], "anc")) {   /* dnntest anc file.dat ancestor_fixings node_fixings cutoff [maxit_anc]: the node cold, then warm from the ancestor's multiplier (any number of levels) */
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n);
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        int la[64], lb[64]; for (int i = 0; i < n; i++) la[i] = lb[i] = -1;
        for (char *p = argv[3]; *p; ) { int i, j, k; if (sscanf(p, "%d:%d%n", &i, &j, &k) != 2) break; la[i] = j; p += k; if (*p == ',') p++; }
        for (char *p = argv[4]; *p; ) { int i, j, k; if (sscanf(p, "%d:%d%n", &i, &j, &k) != 2) break; lb[i] = j; p += k; if (*p == ',') p++; }
        double cut = atof(argv[5]); int maxa = argc > 6 ? atoi(argv[6]) : 4000; int nogu = argc > 7 && atoi(argv[7]);
        int Ia[64], Ja[64], Ib[64], Jb[64]; double *aa, *ba, *ca, sa, *ab, *bb, *cb, sb;
        int ma = build_node(n, A, B, la, &aa, &ba, &ca, &sa, Ia, Ja), mb = build_node(n, A, B, lb, &ab, &bb, &cb, &sb, Ib, Jb);
        int Na = ma * ma + 1, Nb = mb * mb + 1; double *Za = malloc(sizeof(double) * Na * Na), *Z0 = malloc(sizeof(double) * Nb * Nb);
        dnn_prob_t Pa = {ma, aa, ba, ca, sa}; dnn_opt_t o = {0}; o.cutoff = cut; o.maxit = maxa; o.single = 1; o.sub = 20; o.geo = 1; o.giveup = 5; o.giveup_min = 300; o.lap_auto = 1; o.Zout = Za; if (nogu) o.giveup = 0;
        dnn_res_t r; double t = now(); dnn_bound(&Pa, &o, &r);
        printf("ancestor %s m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[3], ma, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        dnn_prob_t Pb = {mb, ab, bb, cb, sb}; dnn_opt_t o2 = o; o2.Zout = NULL; o2.maxit = 4000;
        t = now(); dnn_bound(&Pb, &o2, &r);
        printf("node %s cold m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[4], mb, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        int posF[64], posL[64]; for (int x = 0; x < ma; x++) { posF[Ia[x]] = x; posL[Ja[x]] = x; }
        int map[DNN_MMAX * DNN_MMAX + 1]; map[0] = 0;
        for (int i = 0; i < mb; i++) for (int j = 0; j < mb; j++) map[1 + i * mb + j] = 1 + posF[Ib[i]] * ma + posL[Jb[j]];
        for (int x = 0; x < Nb; x++) for (int y = 0; y < Nb; y++) Z0[(long)x * Nb + y] = Za[(long)map[x] * Na + map[y]];
        o2.Z0 = Z0; t = now(); dnn_bound(&Pb, &o2, &r);
        printf("node %s warm m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[4], mb, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        return 0;
    }
    if (argc >= 6 && !strcmp(argv[1], "kid")) {   /* dnntest kid file.dat parent_fixings i:j cutoff: parent (gu 300, cap 4000), then the child cold, warm, warm folded */
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n);
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        int loc[64]; for (int i = 0; i < n; i++) loc[i] = -1;
        for (char *p = argv[3]; *p; ) { int i, j, k; if (sscanf(p, "%d:%d%n", &i, &j, &k) != 2) break; loc[i] = j; p += k; if (*p == ',') p++; }
        int ci, cj; sscanf(argv[4], "%d:%d", &ci, &cj); double cut = atof(argv[5]);
        int I[64], J[64]; double *a, *b, *c, s0; int m = build_node(n, A, B, loc, &a, &b, &c, &s0, I, J);
        int N = m * m + 1; double *Zp = malloc(sizeof(double) * N * N);
        dnn_prob_t P = {m, a, b, c, s0}; dnn_opt_t o = {0}; o.cutoff = cut; o.maxit = 4000; o.single = 1; o.sub = 20; o.geo = 1; o.giveup = 5; o.giveup_min = 300; o.lap_auto = 1; o.Zout = Zp; o.verbose = getenv("DNN_V") != NULL;
        dnn_res_t r; double t = now(); dnn_bound(&P, &o, &r);
        printf("parent %s m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[3], m, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        int pa = -1, pb = -1; for (int x = 0; x < m; x++) { if (I[x] == ci) pa = x; if (J[x] == cj) pb = x; }
        loc[ci] = cj; int I2[64], J2[64]; double *a2, *b2, *c2, s2; int m2 = build_node(n, A, B, loc, &a2, &b2, &c2, &s2, I2, J2);
        int N2 = m2 * m2 + 1, map[DNN_MMAX * DNN_MMAX + 1], q = 1; map[0] = 0;
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) if (i != pa && j != pb) map[q++] = 1 + i * m + j;
        double *Z0 = malloc(sizeof(double) * N2 * N2);
        for (int mode = 0; mode < 3; mode++) {
            for (int x = 0; x < N2; x++) for (int y = 0; y < N2; y++) Z0[(long)x * N2 + y] = Zp[(long)map[x] * N + map[y]];
            if (mode == 2) { int rr = 1 + pa * m + pb; for (int x = 1; x < N2; x++) { Z0[x] += Zp[(long)rr * N + map[x]]; Z0[(long)x * N2] += Zp[(long)rr * N + map[x]]; } Z0[0] += 2 * Zp[rr] + Zp[(long)rr * N + rr]; }
            dnn_prob_t P2 = {m2, a2, b2, c2, s2}; dnn_opt_t o2 = o; o2.Zout = NULL; o2.Z0 = mode ? Z0 : NULL;
            t = now(); dnn_bound(&P2, &o2, &r);
            printf("child %s %s m=%d: bound %.4f fathomed %d gaveup %d iters %d %.2f s\n", argv[4], mode == 0 ? "cold" : mode == 1 ? "warm" : "warm-fold", m2, r.bound, r.fathomed, r.gaveup, r.iters, now() - t);
        }
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "node")) {
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n);
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        int loc[DNN_MMAX * 2]; for (int i = 0; i < n; i++) loc[i] = -1;
        int usedl[DNN_MMAX * 2] = {0};
        for (char *p = argv[3]; *p; ) { int i, j, k; if (sscanf(p, "%d:%d%n", &i, &j, &k) != 2) break; loc[i] = j; usedl[j] = 1; p += k; if (*p == ',') p++; }
        int I[DNN_MMAX * 2], J[DNN_MMAX * 2], m = 0, mj = 0;
        for (int i = 0; i < n; i++) if (loc[i] < 0) I[m++] = i;
        for (int j = 0; j < n; j++) if (!usedl[j]) J[mj++] = j;
        double *a = malloc(sizeof(double) * m * m), *b = malloc(sizeof(double) * m * m), *c = calloc(m * m, sizeof(double)), s0 = 0;
        for (int x = 0; x < m; x++) for (int y = 0; y < m; y++) { a[x * m + y] = A[I[x] * n + I[y]]; b[x * m + y] = B[J[x] * n + J[y]]; }
        for (int x = 0; x < m; x++) for (int y = 0; y < m; y++) for (int k = 0; k < n; k++) if (loc[k] >= 0)
            c[x * m + y] += A[I[x] * n + k] * B[J[y] * n + loc[k]] + A[k * n + I[x]] * B[loc[k] * n + J[y]];
        for (int k = 0; k < n; k++) if (loc[k] >= 0) for (int l = 0; l < n; l++) if (loc[l] >= 0) s0 += A[k * n + l] * B[loc[k] * n + loc[l]];
        dnn_prob_t P = {m, a, b, c, s0};
        dnn_opt_t o = {0}; o.cutoff = atof(argv[4]); o.maxit = argc > 5 ? atoi(argv[5]) : 4000; o.single = 1; o.sub = argc > 6 ? atoi(argv[6]) : 20;
        if (argc > 7 && atoi(argv[7])) { o.geo = 1; o.giveup = 5; o.giveup_min = 300; }
        o.refine = argc > 8 ? atoi(argv[8]) : 0; o.lap_auto = argc > 9 ? atoi(argv[9]) : 0; o.verbose = getenv("DNN_V") != NULL;
        dnn_res_t r; double t = now(); dnn_bound(&P, &o, &r);
        extern double g_dnn_prof[12], g_dnn_sub[6]; fprintf(stderr, "sub: cheb %.2f cholqr %.2f WQ+H %.2f eigH %.2f ritz %.2f s\n", g_dnn_sub[0], g_dnn_sub[1], g_dnn_sub[2], g_dnn_sub[3], g_dnn_sub[4]);
        fprintf(stderr, "prof: face_in %.2f eig %.2f negpart %.2f face_out %.2f cert %.2f YZ %.2f s, %.0f full eigendecompositions, mean subspace %.0f, mean nneg %.0f, chol fails %.0f, outgrown %.0f\n", g_dnn_prof[0], g_dnn_prof[1], g_dnn_prof[2], g_dnn_prof[3], g_dnn_prof[4], g_dnn_prof[5], g_dnn_prof[6], g_dnn_prof[7] / g_dnn_prof[9], g_dnn_prof[8] / g_dnn_prof[9], g_dnn_prof[10], g_dnn_prof[11]);
        printf("node %s m=%d: bound %.4f cutoff %.1f fathomed %d gaveup %d iters %d %.2f s (%.1f ms/it) refinements %ld/%ld/%ld\n", argv[3], m, r.bound, o.cutoff, r.fathomed, r.gaveup, r.iters, now() - t, 1000 * (now() - t) / (r.iters > 0 ? r.iters : 1), g_dnn_refine[0], g_dnn_refine[1], g_dnn_refine[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "root")) {
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n), *C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        dnn_prob_t P = {n, A, B, C, 0};
        dnn_opt_t o = {0}; o.cutoff = argc > 5 ? atof(argv[5]) : INFINITY; o.maxit = argc > 3 ? atoi(argv[3]) : 300; o.single = argc > 4 ? atoi(argv[4]) : 0; o.verbose = 1; o.sub = argc > 6 ? atoi(argv[6]) : 0;
        dnn_res_t r; double t = now(); dnn_bound(&P, &o, &r);
        extern double g_dnn_prof[12]; fprintf(stderr, "prof: face_in %.2f eig %.2f negpart %.2f face_out %.2f cert %.2f YZ %.2f s\n", g_dnn_prof[0], g_dnn_prof[1], g_dnn_prof[2], g_dnn_prof[3], g_dnn_prof[4], g_dnn_prof[5]);
        printf("root %s n=%d: certified %.4f after %d iterations, %.2f s (%.1f ms/it)\n", argv[2], n, r.bound, r.iters, now() - t, 1000 * (now() - t) / r.iters);
        return 0;
    }
    fprintf(stderr, "usage\n"); return 2;
}
