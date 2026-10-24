/* Microbenchmark: 4 shortest-path LAPs one after the other (lap_sp) against
 * 4 LAPs in lockstep, one per AVX2 lane (lap_sp4). Same answers required. */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NMAX 32
typedef __m256d v4;

static double lap_sp(int dim, const double c[NMAX][NMAX], int *rowsol, double *u, double *v) {
    int colsol[NMAX], freer[NMAX], pred[NMAX], scan[NMAX], nfree = 0;
    double d[NMAX] __attribute__((aligned(64)));
    double done[NMAX] __attribute__((aligned(64)));
    for (int j = 0; j < dim; j++) colsol[j] = -1;
    for (int i = 0; i < dim; i++) {
        double mn = INFINITY; int jm = 0; const double *ci = c[i];
        for (int j = 0; j < dim; j++) { double h = ci[j] - v[j]; int lt = h < mn; mn = lt ? h : mn; jm = lt ? j : jm; }
        if (colsol[jm] < 0) { colsol[jm] = i; rowsol[i] = jm; } else freer[nfree++] = i;
    }
    for (int f = 0; f < nfree; f++) {
        int fr = freer[f], ns = 0, jend = -1;
        const double *cf = c[fr];
        for (int j = 0; j < dim; j++) { d[j] = cf[j] - v[j]; pred[j] = fr; done[j] = 0; }
        double mn;
        for (;;) {
            mn = INFINITY; int js = 0;
            for (int j = 0; j < dim; j++) { double x = d[j] + done[j]; int lt = x < mn; mn = lt ? x : mn; js = lt ? j : js; }
            done[js] = INFINITY; scan[ns++] = js;
            if (colsol[js] < 0) { jend = js; break; }
            int i = colsol[js]; const double *ci = c[i];
            double h = ci[js] - v[js] - mn;
            for (int j = 0; j < dim; j++) {
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

/* 4 problems of the same size, cost of lane k at C[i][j][k], duals V[j][k] */
static void lap_sp4(int dim, const double C[NMAX][NMAX][4], int rowsol[4][NMAX], double V[NMAX][4], double tot[4]) {
    int colsol[4][NMAX], freer[4][NMAX], nfree[4] = {0}, pred[NMAX][4] __attribute__((aligned(32)));
    double D[NMAX][4] __attribute__((aligned(32))), DONE[NMAX][4] __attribute__((aligned(32)));
    int scan[4][NMAX];
    for (int k = 0; k < 4; k++) for (int j = 0; j < dim; j++) colsol[k][j] = -1;
    for (int i = 0; i < dim; i++) {   /* greedy start, all lanes at once (same row i) */
        v4 mn = _mm256_set1_pd(INFINITY); __m256i jm = _mm256_setzero_si256();
        for (int j = 0; j < dim; j++) {
            v4 h = _mm256_sub_pd(_mm256_load_pd(C[i][j]), _mm256_load_pd(V[j]));
            v4 lt = _mm256_cmp_pd(h, mn, _CMP_LT_OQ);
            mn = _mm256_blendv_pd(mn, h, lt);
            jm = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(jm), _mm256_castsi256_pd(_mm256_set1_epi64x(j)), lt));
        }
        long long jj[4]; _mm256_storeu_si256((__m256i *)jj, jm);
        for (int k = 0; k < 4; k++) { int j = (int)jj[k];
            if (colsol[k][j] < 0) { colsol[k][j] = i; rowsol[k][i] = j; } else freer[k][nfree[k]++] = i; }
    }
    int maxf = 0; for (int k = 0; k < 4; k++) if (nfree[k] > maxf) maxf = nfree[k];
    const v4 inf = _mm256_set1_pd(INFINITY), zero = _mm256_setzero_pd();
    for (int f = 0; f < maxf; f++) {
        int act[4], fr[4], ns[4] = {0}, jend[4];
        for (int k = 0; k < 4; k++) { act[k] = f < nfree[k]; fr[k] = act[k] ? freer[k][f] : 0; }
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < 4; k++) { D[j][k] = act[k] ? C[fr[k]][j][k] - V[j][k] : INFINITY; pred[j][k] = fr[k]; DONE[j][k] = act[k] ? 0 : INFINITY; }
        }
        double mnk[4];
        int live = 0; for (int k = 0; k < 4; k++) live += act[k];
        while (live) {
            v4 mn = inf; __m256i js = _mm256_setzero_si256();
            for (int j = 0; j < dim; j++) {
                v4 x = _mm256_add_pd(_mm256_load_pd(D[j]), _mm256_load_pd(DONE[j]));
                v4 lt = _mm256_cmp_pd(x, mn, _CMP_LT_OQ);
                mn = _mm256_blendv_pd(mn, x, lt);
                js = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(js), _mm256_castsi256_pd(_mm256_set1_epi64x(j)), lt));
            }
            double mnv[4]; long long jsv[4]; _mm256_storeu_pd(mnv, mn); _mm256_storeu_si256((__m256i *)jsv, js);
            long long gi[4]; double hk[4];
            for (int k = 0; k < 4; k++) {
                gi[k] = 0; hk[k] = INFINITY;   /* finished lane: relaxation is a no-op */
                if (!act[k]) continue;
                int j = (int)jsv[k];
                DONE[j][k] = INFINITY; scan[k][ns[k]++] = j;
                if (colsol[k][j] < 0) { jend[k] = j; mnk[k] = mnv[k]; act[k] = 0; live--; continue; }
                int i = colsol[k][j];
                hk[k] = C[i][j][k] - V[j][k] - mnv[k];
                gi[k] = (long long)i * NMAX * 4 + k;
            }
            if (!live) break;
            __m256i idx = _mm256_loadu_si256((__m256i *)gi);
            v4 h = _mm256_loadu_pd(hk);
            __m256i iv = _mm256_set_epi64x(gi[3] / (NMAX * 4), gi[2] / (NMAX * 4), gi[1] / (NMAX * 4), gi[0] / (NMAX * 4));
            const double *base = &C[0][0][0];
            for (int j = 0; j < dim; j++) {
                v4 cij = _mm256_i64gather_pd(base + j * 4, idx, 8);
                v4 v2 = _mm256_sub_pd(_mm256_sub_pd(cij, _mm256_load_pd(V[j])), h);
                v4 dj = _mm256_load_pd(D[j]);
                v4 up = _mm256_and_pd(_mm256_cmp_pd(v2, dj, _CMP_LT_OQ), _mm256_cmp_pd(_mm256_load_pd(DONE[j]), zero, _CMP_EQ_OQ));
                _mm256_store_pd(D[j], _mm256_blendv_pd(dj, v2, up));
                __m256i pj = _mm256_cvtepi32_epi64(_mm_load_si128((__m128i *)pred[j]));
                pj = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(pj), _mm256_castsi256_pd(iv), up));
                _mm_store_si128((__m128i *)pred[j], _mm256_cvtepi64_epi32(pj));
            }
        }
        for (int k = 0; k < 4; k++) {
            if (f >= nfree[k]) continue;
            for (int t = 0; t < ns[k]; t++) { int j = scan[k][t]; V[j][k] += D[j][k] - mnk[k]; }
            int i, j = jend[k];
            do { i = pred[j][k]; colsol[k][j] = i; int q = rowsol[k][i]; rowsol[k][i] = j; j = q; } while (i != fr[k]);
        }
    }
    for (int k = 0; k < 4; k++) { double t = 0;
        for (int i = 0; i < dim; i++) { int j = rowsol[k][i]; t += C[i][j][k] - V[j][k]; }
        for (int j = 0; j < dim; j++) t += V[j][k];
        tot[k] = t; }
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    int dim = argc > 1 ? atoi(argv[1]) : 14, reps = argc > 2 ? atoi(argv[2]) : 20000;
    static double c[4][NMAX][NMAX] __attribute__((aligned(64))), C[NMAX][NMAX][4] __attribute__((aligned(64)));
    srand(1);
    int nsets = 64; double (*sets)[4][NMAX][NMAX] = aligned_alloc(64, sizeof(double) * nsets * 4 * NMAX * NMAX);
    for (int s = 0; s < nsets; s++) for (int k = 0; k < 4; k++) for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++)
        sets[s][k][i][j] = (rand() % 100000) / 7.0 + ((i * 7 + j * 3) % 11) * 50;
    double t0 = now(), chk1 = 0, chk2 = 0;
    for (int r = 0; r < reps; r++) {
        int s = r % nsets;
        for (int k = 0; k < 4; k++) {
            double v[NMAX], u[NMAX]; int rs[NMAX];
            for (int j = 0; j < dim; j++) { double mn = INFINITY; for (int i = 0; i < dim; i++) mn = fmin(mn, sets[s][k][i][j]); v[j] = mn; }
            chk1 += lap_sp(dim, sets[s][k], rs, u, v);
        }
    }
    double t1 = now();
    for (int r = 0; r < reps; r++) {
        int s = r % nsets;
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) for (int k = 0; k < 4; k++) C[i][j][k] = sets[s][k][i][j];
        double V[NMAX][4] __attribute__((aligned(32))), tot[4]; int rs[4][NMAX];
        for (int j = 0; j < dim; j++) for (int k = 0; k < 4; k++) { double mn = INFINITY; for (int i = 0; i < dim; i++) mn = fmin(mn, C[i][j][k]); V[j][k] = mn; }
        lap_sp4(dim, (const double (*)[NMAX][4])C, rs, V, tot);
        chk2 += tot[0] + tot[1] + tot[2] + tot[3];
    }
    double t2 = now();
    (void)c;
    printf("m=%d  sequential %.3f us/LAP  lockstep %.3f us/LAP (incl. transpose)  ratio %.2f  check %.6g %.6g\n",
           dim, (t1 - t0) / reps / 4 * 1e6, (t2 - t1) / reps / 4 * 1e6, (t1 - t0) / (t2 - t1), chk1, chk2);
    return 0;
}
