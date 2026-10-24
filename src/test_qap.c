/*
 * test_qap.c : unit tests of qap_bb.c against brute force.
 * The solver is included as is (its main renamed), so the tests exercise the
 * exact code used by the campaign.
 *
 *   1. LAP: Hungarian, Jonker-Volgenant cold and warm, against enumeration
 *      of all permutations; dual feasibility and zero duality gap.
 *   2. QPB bound: never above the best completion, at the root and at random
 *      partial assignments, with every option combination.
 *   3. Full branch and bound: optimum equal to brute force, with and without
 *      symmetries, with and without each optimization, strategies C and D.
 */
#define main qap_main
#include "qap_bb.c"
#undef main

static int n_fail = 0, n_check = 0;
#define CHECK(cond, ...) do { n_check++; if (!(cond)) { n_fail++; fprintf(stderr, "FAIL " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static rng_t R = { 0x9E3779B97F4A7C15ull };
static int rnd(int k) { R.s ^= R.s << 13; R.s ^= R.s >> 7; R.s ^= R.s << 17; return (int)(R.s % (unsigned long long)k); }

/* next permutation in lexicographic order; returns 0 after the last */
static int next_perm(int *p, int n) {
    int i = n - 2; while (i >= 0 && p[i] > p[i + 1]) i--;
    if (i < 0) return 0;
    int j = n - 1; while (p[j] < p[i]) j--;
    int t = p[i]; p[i] = p[j]; p[j] = t;
    for (int a = i + 1, b = n - 1; a < b; a++, b--) { t = p[a]; p[a] = p[b]; p[b] = t; }
    return 1;
}

/* ---------- 1. LAP ---------- */
static void test_lap(void) {
    static double c[NMAX][NMAX];
    for (int trial = 0; trial < 3000; trial++) {
        int n = 2 + rnd(7), range = (trial % 3 == 0) ? 4 : 1000;   /* small range: many ties */
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] = rnd(range) - (trial % 2 ? range / 2 : 0);
        int p[NMAX]; for (int i = 0; i < n; i++) p[i] = i;
        double best = INFINITY;
        do { double s = 0; for (int i = 0; i < n; i++) s += c[i][p[i]]; if (s < best) best = s; } while (next_perm(p, n));
        for (int alg = 0; alg < 3; alg++) {
            int rs[NMAX]; double u[NMAX] = {0}, v[NMAX] = {0}, val;
            if (alg == 0) val = lap(n, c, rs, u, v);
            else if (alg == 1) {   /* JV cold: v initialised by column minima, as in qpb_fw */
                for (int j = 0; j < n; j++) { double mn = INFINITY; for (int i = 0; i < n; i++) if (c[i][j] < mn) mn = c[i][j]; v[j] = mn; }
                val = lap_jv(n, c, rs, u, v, 0);
            } else {               /* JV warm: duals of a perturbed matrix */
                static double c2[NMAX][NMAX]; int rs2[NMAX]; double u2[NMAX], v2[NMAX];
                for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c2[i][j] = c[i][j] + rnd(range / 4 + 1);
                lap(n, c2, rs2, u2, v2);
                for (int j = 0; j < n; j++) v[j] = v2[j];
                val = lap_jv(n, c, rs, u, v, 1);
            }
            double s = 0; int used[NMAX] = {0}, okperm = 1;
            for (int i = 0; i < n; i++) { if (rs[i] < 0 || rs[i] >= n || used[rs[i]]++) okperm = 0; else s += c[i][rs[i]]; }
            CHECK(okperm, "lap alg %d n %d: not a permutation", alg, n);
            CHECK(fabs(val - best) < 1e-6 && fabs(s - best) < 1e-6, "lap alg %d n %d: value %g, assignment %g, optimum %g", alg, n, val, s, best);
            double su = 0, mins = INFINITY;
            for (int i = 0; i < n; i++) su += u[i] + v[i];
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double r = c[i][j] - u[i] - v[j]; if (r < mins) mins = r; }
            CHECK(mins > -1e-6, "lap alg %d n %d: reduced cost %g < 0", alg, n, mins);
            CHECK(fabs(su - best) < 1e-6, "lap alg %d n %d: sum of duals %g != %g", alg, n, su, best);
        }
    }
}

/* ---------- random instances ---------- */
/* grid: A = Manhattan distances on an r x c grid (symmetries, like nug);
 * otherwise A random symmetric. B random symmetric flows with zeros. */
static void make_instance(int n, int grid_r) {
    N = n;
    for (int i = 0; i < n; i++) for (int k = 0; k <= i; k++) {
        double a;
        if (grid_r) { int c = n / grid_r; a = abs(i / c - k / c) + abs(i % c - k % c); }
        else a = (i == k) ? 0 : rnd(10);
        A[i][k] = A[k][i] = a;
        double b = (i == k) ? 0 : (rnd(3) ? rnd(10) : 0);
        B[i][k] = B[k][i] = b;
    }
    memset(C0, 0, sizeof C0);
}

static double brute(int *bestp) {
    int p[NMAX]; for (int i = 0; i < N; i++) p[i] = i;
    double best = INFINITY;
    do { double s = qap_cost(p); if (s < best) { best = s; if (bestp) memcpy(bestp, p, sizeof(int) * N); } } while (next_perm(p, N));
    return best;
}

/* best completion of a partial assignment */
static double brute_completion(const node_t *nd) {
    int F[NMAX], L[NMAX], m = 0, mc = 0, p[NMAX], q[NMAX];
    for (int i = 0; i < N; i++) if (nd->rc[i] < 0) F[m++] = i;
    for (int j = 0; j < N; j++) if (nd->cr[j] < 0) L[mc++] = j;
    for (int a = 0; a < m; a++) q[a] = a;
    double best = INFINITY;
    do { for (int i = 0; i < N; i++) p[i] = nd->rc[i]; for (int a = 0; a < m; a++) p[F[a]] = L[q[a]];
         double s = qap_cost(p); if (s < best) best = s; } while (next_perm(q, m));
    return best;
}

static void set_opts(int combo) {
    opt_incr = combo & 1; opt_ecache = (combo >> 1) & 1; opt_memo = (combo >> 2) & 1;
    opt_fastmm = (combo >> 3) & 1; opt_lap = (combo >> 4) & 1 ? 0 : 2;
}

/* ---------- 2. QPB bound validity ---------- */
static void test_bound(void) {
    for (int trial = 0; trial < 120; trial++) {
        int n = 5 + rnd(4);
        make_instance(n, (trial % 2 && n % 2 == 0) ? 2 : 0);
        for (int combo = 0; combo < 32; combo += 3) {
            set_opts(combo);
            work_t *w = calloc(1, sizeof(work_t));
            for (int k = 0; k < 4; k++) {
                node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
                int fix = k == 0 ? 0 : rnd(N - 2);
                for (int f = 0; f < fix; f++) { int i, j; do i = rnd(N); while (nd.rc[i] >= 0); do j = rnd(N); while (nd.cr[j] >= 0);
                    nd.rc[i] = j; nd.cr[j] = i; nd.depth++; }
                g_ub = INFINITY;
                eval_t ev; eval_node(&nd, w, &ev, 150, 150, 0);
                double opt = brute_completion(&nd);
                CHECK(ev.z <= opt + 1e-6, "bound n %d combo %d fixed %d: QPB %.6f > best completion %.0f", N, combo, fix, ev.z, opt);
                CHECK(g_ub >= opt - 1e-6, "bound n %d: incumbent %.0f below optimum %.0f", N, (double)g_ub, opt);
            }
            free(w);
        }
    }
}

/* ---------- 3. full branch and bound ---------- */
static double solve_bb(int sym, char strat, double ub, long *nodes) {
    opt_sym = sym; set_strategy(strat);
    compute_aut(A, &GA); compute_aut(B, &GB);
    g_ub = ub; memset(g_best, -1, sizeof g_best);
    work_t *w = calloc(1, sizeof(work_t)); stats_t st = {0};
    node_t nd; for (int i = 0; i < N; i++) nd.rc[i] = nd.cr[i] = -1; nd.depth = 0;
    dfs(&nd, w, &st, 0); free(w);
    if (nodes) *nodes = st.nodes;
    return g_ub;
}

static void test_bb(void) {
    for (int trial = 0; trial < 60; trial++) {
        int n = 6 + rnd(3), grid = (n % 2 == 0 && trial % 2) ? 2 : 0;
        make_instance(n, grid);
        double opt = brute(NULL);
        for (int combo = 0; combo < 32; combo += 5) for (int sym = 0; sym < 2; sym++) for (int s = 0; s < 2; s++) {
            set_opts(combo);
            char strat = s ? 'D' : 'C';
            double v = solve_bb(sym, strat, INFINITY, NULL);
            CHECK(fabs(v - opt) < 1e-6, "bb n %d grid %d sym %d strat %c combo %d: found %.0f, optimum %.0f", n, grid, sym, strat, combo, v, opt);
            CHECK(g_best[0] >= 0 && fabs(qap_cost(g_best) - opt) < 1e-6, "bb n %d: returned permutation costs %.0f, optimum %.0f", n, g_best[0] >= 0 ? qap_cost(g_best) : -1, opt);
            /* paper's conditions: upper bound fixed at optimum + 1, the proof must find the optimum */
            v = solve_bb(sym, strat, opt + 1, NULL);
            CHECK(fabs(v - opt) < 1e-6, "bb ub=opt+1 n %d sym %d strat %c combo %d: found %.0f, optimum %.0f", n, sym, strat, combo, v, opt);
        }
    }
}

int main(void) {
    init_V();
    double t = now();
    test_lap();   printf("LAP              : %d checks, %d failures\n", n_check, n_fail); int c0 = n_check, f0 = n_fail;
    test_bound(); printf("QPB bound        : %d checks, %d failures\n", n_check - c0, n_fail - f0); c0 = n_check; f0 = n_fail;
    test_bb();    printf("Branch and bound : %d checks, %d failures\n", n_check - c0, n_fail - f0);
    printf("%s : %d checks, %d failures, %.1f s\n", n_fail ? "FAILED" : "OK", n_check, n_fail, now() - t);
    return n_fail != 0;
}
