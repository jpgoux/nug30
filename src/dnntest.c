/* tests of the certified DNN bound:
 *   dnntest rand m count seed   random instances, bound against enumeration
 *   dnntest root file.dat [maxit] [single]   root bound of a QAPLIB instance */
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
    if (argc >= 3 && !strcmp(argv[1], "root")) {
        FILE *f = fopen(argv[2], "r"); int n; if (!f || fscanf(f, "%d", &n) != 1) return 1;
        double *A = malloc(sizeof(double) * n * n), *B = malloc(sizeof(double) * n * n), *C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &A[i]) != 1) return 1;
        for (int i = 0; i < n * n; i++) if (fscanf(f, "%lf", &B[i]) != 1) return 1;
        fclose(f);
        dnn_prob_t P = {n, A, B, C, 0};
        dnn_opt_t o = {0}; o.cutoff = argc > 5 ? atof(argv[5]) : INFINITY; o.maxit = argc > 3 ? atoi(argv[3]) : 300; o.single = argc > 4 ? atoi(argv[4]) : 0; o.verbose = 1; o.sub = argc > 6 ? atoi(argv[6]) : 0;
        dnn_res_t r; double t = now(); dnn_bound(&P, &o, &r);
        extern double g_dnn_prof[8]; fprintf(stderr, "prof: face_in %.2f eig %.2f negpart %.2f face_out %.2f cert %.2f YZ %.2f s\n", g_dnn_prof[0], g_dnn_prof[1], g_dnn_prof[2], g_dnn_prof[3], g_dnn_prof[4], g_dnn_prof[5]);
        printf("root %s n=%d: certified %.4f after %d iterations, %.2f s (%.1f ms/it)\n", argv[2], n, r.bound, r.iters, now() - t, 1000 * (now() - t) / r.iters);
        return 0;
    }
    fprintf(stderr, "usage\n"); return 2;
}
