# NUG30 Laboratory

*Français ci-dessous. English version further down.*

## Français

Ce dépôt refait en 2026, sur une seule machine, le calcul de l'article

> K. Anstreicher, N. Brixius, J.-P. Goux, J. Linderoth, *Solving large quadratic assignment problems on computational grids*, Mathematical Programming 91 (2002) 563-588. Prix SIAG/OPT de la SIAM, 2002.

L'article avait résolu NUG30 (problème d'affectation quadratique de taille 30, QAPLIB) en 2000 par séparation et évaluation, avec la borne QPB calculée par Frank-Wolfe, sur une grille Condor d'environ 650 machines en moyenne pendant 7 jours.

Le site du laboratoire (bilingue, suivi des calculs) est dans `docs/index.html` et peut être servi par GitHub Pages.

### Contenu

| Dossier | Contenu |
|---|---|
| `src/qap_bb.c` | Solveur C : borne QPB (Anstreicher-Brixius), Frank-Wolfe avec recherche linéaire exacte, LAP hongrois puis Jonker-Volgenant à chaud, règles de branchement 2, 3 et 4 et stratégies A à E de Brixius-Anstreicher (OMS 2001), symétries des matrices de distance (groupes d'automorphismes), estimateur de Knuth, mode parallèle à pool partagé (threads). |
| `src/reference/qap_bb_campagne.c` | Code tel qu'utilisé pour la campagne nug12 à nug24 (algorithme de 2000), gardé comme référence. |
| `src/test_qap.c`, `tests/` | Tests contre la force brute (LAP, validité de la borne, B&B complet avec et sans symétries) et banc de non-régression (optimum et empreinte de l'arbre). |
| `scripts/campagne.py` | Lanceur de la campagne : file de calculs, état écrit toutes les 5 s pour le site. |
| `data/qaplib/` | Instances nug de QAPLIB et solutions connues, copiées depuis [mhahsler/qap](https://github.com/mhahsler/qap). |
| `results/` | Résultats JSON de chaque calcul, estimations de Knuth comprises (commande, nœuds, temps, permutation optimale). |
| `docs/` | Site et instantané de l'état de la campagne. |

### Compiler et lancer

```sh
cd src && make            # gcc, LAPACK, BLAS (Debian/Ubuntu : liblapack-dev, libblas-dev)
# sans les paquets -dev : make LIBS="-l:liblapack.so.3 -l:libblas.so.3 -lm -lpthread"
./qap_bb solve ../data/qaplib/nug20.dat -strat C -ub 2571 -nointprune
./qap_bb estimate ../data/qaplib/nug25.dat -strat D -dives 400 -q 1.5
./qap_bb par ../data/qaplib/nug17.dat -strat C -threads 4
make test                 # tests et banc de non-régression, environ 1 min 30 s
```

`-ub opt+1 -nointprune` reproduit les conditions de l'article : on prouve l'optimalité avec une borne supérieure fixée à l'optimum plus un.

### Premiers résultats (1 cœur Xeon 2,1 GHz)

| Instance | Nœuds | Temps | Article (HP C3000) |
|---|---|---|---|
| nug18 (C) | 206 272 | 158 s | 251 470 nœuds, 69,2 min |
| nug20 (C) | 973 503 | 306 s | 1 040 308 nœuds, 145,8 min |
| nug21 (C) | 1 088 248 | 383 s | 1 698 093 nœuds, 212,3 min |

Les résultats complets et à jour sont sur le site.

### Références

- K. M. Anstreicher, N. W. Brixius, *A new bound for the quadratic assignment problem based on convex quadratic programming*, Mathematical Programming 89 (2001) 341-357.
- N. W. Brixius, K. M. Anstreicher, *Solving quadratic assignment problems using convex quadratic programming relaxations*, Optimization Methods and Software 16 (2001) 49-68.
- R. Jonker, A. Volgenant, *A shortest augmenting path algorithm for dense and sparse linear assignment problems*, Computing 38 (1987) 325-340.
- R. E. Burkard, S. E. Karisch, F. Rendl, *QAPLIB, a quadratic assignment problem library*, J. Global Optimization 10 (1997) 391-403.

Les articles protégés ne sont pas inclus dans ce dépôt.

## English

This repository redoes in 2026, on a single machine, the computation of

> K. Anstreicher, N. Brixius, J.-P. Goux, J. Linderoth, *Solving large quadratic assignment problems on computational grids*, Mathematical Programming 91 (2002) 563-588. SIAM SIAG/OPT Prize, 2002.

That paper solved NUG30 (QAPLIB quadratic assignment instance of size 30) in 2000 by branch and bound, using the QPB bound computed by Frank-Wolfe, on a Condor grid of about 650 machines on average over 7 days.

The lab site (bilingual, live tracking) is `docs/index.html` and can be served by GitHub Pages.

### Contents

| Folder | Contents |
|---|---|
| `src/qap_bb.c` | C solver: QPB bound (Anstreicher-Brixius), Frank-Wolfe with exact line search, Hungarian LAP then warm-started Jonker-Volgenant, branching rules 2, 3 and 4 and strategies A to E of Brixius-Anstreicher (OMS 2001), symmetries of the distance matrices (automorphism groups), Knuth estimator, shared-pool parallel mode (threads). |
| `src/reference/qap_bb_campagne.c` | Code as used for the nug12 to nug24 campaign (2000 algorithm), kept as reference. |
| `src/test_qap.c`, `tests/` | Tests against brute force (LAP, bound validity, full B&B with and without symmetries) and regression bench (optimum and tree fingerprint). |
| `scripts/campagne.py` | Campaign runner: job queue, state written every 5 s for the site. |
| `data/qaplib/` | QAPLIB nug instances and known solutions, copied from [mhahsler/qap](https://github.com/mhahsler/qap). |
| `results/` | JSON result of every run (command, nodes, time, optimal permutation). |
| `docs/` | Site and snapshot of the campaign state. |

### Build and run

See the commands above. `-ub opt+1 -nointprune` reproduces the paper's conditions: optimality is proved with an upper bound fixed at the optimum plus one.

### First results (1 core, Xeon 2.1 GHz)

See the table above (nodes, seconds, and the paper's figures on an HP C3000). Complete, up-to-date results are on the site.

### References

See the list above. Copyrighted papers are not included in this repository.
