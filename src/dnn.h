#ifndef DNN_H
#define DNN_H
#define DNN_MMAX 32
typedef struct { int m; const double *A, *B, *C; double s; } dnn_prob_t;   /* A, B, C: m x m row-major */
typedef struct {
    double cutoff;        /* stop at the first certified value above it */
    int maxit, cert_every, single, verbose;
    int sub;              /* > 0: subspace iteration on the negative eigenspace, full eigendecomposition every sub iterations */
    double giveup;        /* > 0: give up when best + giveup * (gain of the last 50 iterations) < cutoff */
    int giveup_min;       /* give-up allowed from this iteration on (0: 100) */
    int geo;
    int refine;
    int lap_auto;         /* 1: inner LAPs of the certificate by reduction bounds, exact only near the cutoff */           /* 1: when the semidefinite penalty costs the fathom, project Z' in double and certify again */              /* 1: geometric-tail give-up of Brixius (giveup = 5 there) instead of the linear one */
    double beta, gamma;   /* 0: defaults beta = 0.1 m, gamma = 1.618 */
    const double *Z0;     /* optional warm start, N x N (unscaled units), N = m^2 + 1 */
    double *Zout;         /* optional: final multiplier, unscaled */
    double *kid;          /* optional, m x m: when not fathomed, a bound for each child i -> j (else -inf) */
} dnn_opt_t;
typedef struct { double bound; int iters, fathomed, gaveup; } dnn_res_t;
extern long g_dnn_refine[3];
double dnn_bound(const dnn_prob_t *P, const dnn_opt_t *o, dnn_res_t *res);
#endif
