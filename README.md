# nug30 Laboratory

*Français d'abord. [English version below](#english).*

Site du laboratoire : <https://jpgoux.github.io/nug30/> (fichier `docs/index.html`).

## Français

### Le projet

En juin 2000, l'article

> K. Anstreicher, N. Brixius, J.-P. Goux, J. Linderoth, *Solving large quadratic assignment problems on computational grids*, Mathematical Programming 91 (2002) 563-588. Prix SIAG/OPT de la SIAM, 2002. Preprint : [Optimization Online, 2000](https://optimization-online.org/2000/10/233/).

a résolu nug30, problème d'affectation quadratique de taille 30 de la QAPLIB, par séparation et évaluation, en une semaine, sur une grille Condor de 653 machines en moyenne (1 009 au pic).

Le défi de ce dépôt : résoudre nug30 en 2026 **sur une seule machine de 4 cœurs, en une journée**. Il faut pour cela un facteur 25 à 30 par rapport au code de 2000 reproduit. Le travail suit deux voies mesurées séparément :

- **implémentation** : même borne, même règle de branchement, calcul plus rapide (objectif ×3) ;
- **algorithme** : borne ou branchement différents, donc un arbre plus petit (le reste du facteur).

Tout est en C, sans dépendance autre que LAPACK et BLAS.

### Le problème

Placer n usines sur n emplacements (une permutation p) pour minimiser
`somme_{i,k} A[i][k] · B[p(i)][p(k)]`, où A donne les flux entre usines et B les distances entre emplacements (instances de Nugent : distances de Manhattan sur une grille, par exemple 5 × 6 pour nug30).

### Architecture du solveur (`src/qap_bb.c`)

Un seul fichier d'environ 1 250 lignes, organisé en couches, de la plus basse à la plus haute.

**1. Affectation linéaire (LAP).** Chaque itération de Frank-Wolfe résout un LAP dense de taille m (nombre d'usines encore libres). C'est la couche la plus sollicitée, donc la plus optimisée.
- `lap` : méthode hongroise, sert de référence et pour le petit LAP de la borne.
- `lap_jv` : Jonker-Volgenant à chaud, avec réduction de lignes écrite sans branchement (option `-lap 2`).
- `lap_sp` (défaut) : un plus court chemin augmentant (Dijkstra sur tableau, en O(m²)) par ligne restée libre après une affectation gloutonne. Les boucles sont sans branchement et vectorisées par le compilateur. Les duales u se lisent sur l'affectation (écart complémentaire), sans passage supplémentaire sur la matrice.

**2. Borne QPB (`setup_qpb`, `qpb_fw`).** La borne d'Anstreicher et Brixius (2001) est une relaxation quadratique convexe.
- `setup_qpb` construit les matrices de projection Sp et Tp à partir des valeurs propres de A et B restreintes aux usines et emplacements libres. Les décompositions spectrales sont mises en cache par sous-ensemble, car les frères partagent les mêmes lignes libres (`get_eig`).
- `qpb_fw` lance Frank-Wolfe depuis X = eeᵀ/m avec recherche linéaire exacte. Chaque itération enchaîne un LAP sur le gradient G, une borne inférieure z, puis le produit A·P·B mis à jour par rangs un quand la permutation change peu (`applyMperm`). Les produits scalaires ⟨G, X⟩ et ⟨C, X⟩ sont tenus à jour en O(m), et le gradient suivant est construit pendant la mise à jour de X.
- On s'arrête dès que z dépasse la meilleure solution connue moins un (nœud élagué), ou après le nombre d'itérations fixé par la stratégie.
- La matrice des coûts réduits U = G − u − v donne, pour chaque affectation (i → j), une borne de l'enfant correspondant.

**3. Évaluation d'un nœud (`eval_node`).** Un nœud fixe une partie de la permutation. On calcule la constante K des paires fixées, les matrices réduites Ar, Br et le terme linéaire Cr, puis on appelle `qpb_fw`. Chaque permutation rencontrée par Frank-Wolfe est aussi une solution réalisable : elle peut améliorer la borne supérieure.

**4. Branchement (`branch`).** Règles 2, 3 et 4 de Brixius et Anstreicher (OMS 2001), choisies selon la profondeur par les stratégies A à E (`set_strategy`).
- Règle 2 : on branche sur la ligne ou la colonne dont la somme des coûts réduits U est la plus grande.
- Règles 3 et 4 (anticipation) : on évalue réellement les meilleurs enfants candidats par quelques itérations de Frank-Wolfe, et on garde la ligne ou la colonne qui en élimine le plus. Un enfant (i → j) figure à la fois dans la liste de la ligne i et dans celle de la colonne j ; il n'est calculé qu'une fois.
- Les symétries des matrices de distance (groupe d'automorphismes calculé par `compute_aut`, restreint au stabilisateur des affectations déjà faites) suppriment les enfants équivalents.
- Les enfants sont triés par borne croissante.

**5. Parcours de l'arbre.**
- `dfs` : profondeur d'abord, en série.
- Mode `par` : pool partagé de nœuds ouverts et threads POSIX. Chaque fil prend le nœud le plus profond, l'explore pendant une tranche de temps, puis rend ses nœuds les moins profonds si le pool manque de travail. La meilleure solution est partagée et vue par tous à leur nœud suivant.
- `estimate` : estimateur de Knuth (plongées aléatoires), pour prévoir la taille de l'arbre.

**6. Suivi et reprise.**
- `-prog f` : fichier de progression réécrit toutes les 10 s (nœuds par profondeur, chemin courant dans l'arbre, plus petite borne ouverte). Le site le lit pour l'arbre en direct.
- `-ckpt f -ckpt_s s` : sauvegarde des nœuds ouverts toutes les s secondes ; `-resume f` reprend. En série, l'arbre parcouru est identique à celui d'un calcul sans interruption.
- `-log f` : journal des événements (nouvelles solutions).

### Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `src/qap_bb.c` | Le solveur décrit ci-dessus. |
| `src/reference/qap_bb_campagne.c` | Le code de 2000 reproduit, tel qu'utilisé pour la campagne nug12 à nug24, gardé comme référence des mesures. |
| `src/test_qap.c` | Tests contre la force brute : LAP (toutes les variantes, à froid et à chaud), validité de la borne QPB à des nœuds aléatoires pour toutes les combinaisons d'options, séparation et évaluation complète avec et sans symétries. |
| `tests/` | Banc de non-régression (`expected.txt` : optimum, nombre de nœuds et d'itérations de Frank-Wolfe de nug12 à nug17) et test de reprise après checkpoint. |
| `scripts/` | `campagne.py` : file de calculs et état écrit toutes les 5 s ; `make_snapshot.py` : instantané embarqué dans le site. |
| `data/qaplib/` | Instances nug de la QAPLIB et solutions connues, copiées depuis [mhahsler/qap](https://github.com/mhahsler/qap). |
| `results/` | Résultat JSON de chaque calcul (commande, nœuds, temps, permutation optimale, estimations de Knuth). |
| `docs/` | Site bilingue (GitHub Pages) et état de la campagne. |

### Compiler, tester, lancer

```sh
cd src && make            # gcc, LAPACK, BLAS (Debian/Ubuntu : liblapack-dev, libblas-dev)
# sans les paquets -dev : make LIBS="-l:liblapack.so.3 -l:libblas.so.3 -lm -lpthread"
make test                 # force brute, non-régression, reprise après checkpoint (environ 2 min)
./qap_bb solve ../data/qaplib/nug20.dat -strat C -ub 2571 -nointprune
./qap_bb par ../data/qaplib/nug22.dat -strat C -threads 4 -ckpt ckpt.txt -ckpt_s 600
./qap_bb estimate ../data/qaplib/nug30.dat -strat D -dives 400 -q 1.5
```

`-ub opt+1 -nointprune` reproduit les conditions de l'article : on prouve l'optimalité avec une borne supérieure fixée à l'optimum plus un.

### Comment on mesure un gain

Chaque modification passe d'abord `make test`. Le gain est mesuré sur cinq instances (nug14, nug15, nug16a, nug16b, nug17), résolues entièrement dans les conditions de l'article, sur un cœur, en gardant le meilleur de deux exécutions. On publie la somme des temps, le gain par instance et leur moyenne géométrique, et l'écart de nombre de nœuds quand l'arbre change. Le tableau à jour est dans l'onglet « De nug25 à nug30 » du site.

### Références

- K. M. Anstreicher, N. W. Brixius, *A new bound for the quadratic assignment problem based on convex quadratic programming*, Mathematical Programming 89 (2001) 341-357.
- N. W. Brixius, K. M. Anstreicher, *Solving quadratic assignment problems using convex quadratic programming relaxations*, Optimization Methods and Software 16 (2001) 49-68.
- R. Jonker, A. Volgenant, *A shortest augmenting path algorithm for dense and sparse linear assignment problems*, Computing 38 (1987) 325-340.
- R. E. Burkard, S. E. Karisch, F. Rendl, *QAPLIB, a quadratic assignment problem library*, J. Global Optimization 10 (1997) 391-403.

Les articles protégés par le droit d'auteur ne sont pas inclus dans ce dépôt.

## English

### The project

In June 2000, the paper

> K. Anstreicher, N. Brixius, J.-P. Goux, J. Linderoth, *Solving large quadratic assignment problems on computational grids*, Mathematical Programming 91 (2002) 563-588. SIAM SIAG/OPT Prize, 2002. Preprint: [Optimization Online, 2000](https://optimization-online.org/2000/10/233/).

solved nug30, the size-30 quadratic assignment instance of QAPLIB, by branch and bound, in one week, on a Condor grid of 653 machines on average (1,009 at peak).

The challenge of this repository: solve nug30 in 2026 **on a single 4-core machine, in one day**. This needs a factor of 25 to 30 over the reproduced 2000 code. The work follows two tracks, measured separately:

- **implementation**: same bound, same branching rule, faster computation (goal ×3);
- **algorithm**: different bound or branching, hence a smaller tree (the rest of the factor).

Everything is in C, with no dependency other than LAPACK and BLAS.

### The problem

Place n facilities on n locations (a permutation p) to minimize
`sum_{i,k} A[i][k] · B[p(i)][p(k)]`, where A holds the flows between facilities and B the distances between locations (Nugent instances: Manhattan distances on a grid, for example 5 × 6 for nug30).

### Solver architecture (`src/qap_bb.c`)

A single file of about 1,250 lines, organized in layers, from lowest to highest.

**1. Linear assignment (LAP).** Every Frank-Wolfe iteration solves a dense LAP of size m (number of facilities still free). This is the most used layer, hence the most optimized.
- `lap`: Hungarian method, used as reference and for the small LAP of the bound.
- `lap_jv`: warm-started Jonker-Volgenant, with a branch-free row reduction (option `-lap 2`).
- `lap_sp` (default): one shortest augmenting path (array-based Dijkstra, O(m²)) per row left free after a greedy assignment. Loops are branch-free and vectorized by the compiler. The duals u are read from the assignment (complementary slackness), with no extra pass over the matrix.

**2. QPB bound (`setup_qpb`, `qpb_fw`).** The Anstreicher-Brixius bound (2001) is a convex quadratic relaxation.
- `setup_qpb` builds the projection matrices Sp and Tp from the eigenvalues of A and B restricted to the free facilities and locations. Eigen-decompositions are cached per subset, since siblings share the same free rows (`get_eig`).
- `qpb_fw` runs Frank-Wolfe from X = eeᵀ/m with exact line search. Each iteration chains a LAP on the gradient G, a lower bound z, then the product A·P·B, updated by rank-one terms when the permutation changes little (`applyMperm`). The inner products ⟨G, X⟩ and ⟨C, X⟩ are kept up to date in O(m), and the next gradient is built during the update of X.
- It stops as soon as z exceeds the incumbent minus one (node fathomed), or after the number of iterations set by the strategy.
- The reduced-cost matrix U = G − u − v gives, for each assignment (i → j), a bound on the corresponding child.

**3. Node evaluation (`eval_node`).** A node fixes part of the permutation. We compute the constant K of the fixed pairs, the reduced matrices Ar, Br and the linear term Cr, then call `qpb_fw`. Every permutation met by Frank-Wolfe is also a feasible solution and may improve the incumbent.

**4. Branching (`branch`).** Rules 2, 3 and 4 of Brixius and Anstreicher (OMS 2001), chosen by depth through strategies A to E (`set_strategy`).
- Rule 2: branch on the row or column with the largest sum of reduced costs U.
- Rules 3 and 4 (look-ahead): the best candidate children are actually evaluated by a few Frank-Wolfe iterations, and the row or column that fathoms the most is kept. A child (i → j) belongs both to the list of row i and to that of column j; it is computed only once.
- Symmetries of the distance matrices (automorphism group computed by `compute_aut`, restricted to the stabilizer of the assignments already made) remove equivalent children.
- Children are sorted by increasing bound.

**5. Tree search.**
- `dfs`: depth first, serial.
- `par` mode: shared pool of open nodes and POSIX threads. Each worker takes the deepest node, explores it for a time slice, then gives back its shallowest nodes if the pool runs short of work. The incumbent is shared and seen by all workers at their next node.
- `estimate`: Knuth estimator (random dives), to predict the tree size.

**6. Monitoring and restart.**
- `-prog f`: progress file rewritten every 10 s (nodes per depth, current path in the tree, smallest open bound). The site reads it for the live tree.
- `-ckpt f -ckpt_s s`: open nodes saved every s seconds; `-resume f` restarts. In serial mode the tree explored is identical to that of an uninterrupted run.
- `-log f`: event journal (new incumbents).

### Repository layout

| Folder | Contents |
|---|---|
| `src/qap_bb.c` | The solver described above. |
| `src/reference/qap_bb_campagne.c` | The reproduced 2000 code, as used for the nug12 to nug24 campaign, kept as the measurement reference. |
| `src/test_qap.c` | Tests against brute force: LAP (all variants, cold and warm), QPB bound validity at random nodes for every option combination, full branch and bound with and without symmetries. |
| `tests/` | Regression bench (`expected.txt`: optimum, node count and Frank-Wolfe iterations from nug12 to nug17) and checkpoint restart test. |
| `scripts/` | `campagne.py`: job queue and state written every 5 s; `make_snapshot.py`: snapshot embedded in the site. |
| `data/qaplib/` | QAPLIB nug instances and known solutions, copied from [mhahsler/qap](https://github.com/mhahsler/qap). |
| `results/` | JSON result of every run (command, nodes, time, optimal permutation, Knuth estimates). |
| `docs/` | Bilingual site (GitHub Pages) and campaign state. |

### Build, test, run

```sh
cd src && make            # gcc, LAPACK, BLAS (Debian/Ubuntu: liblapack-dev, libblas-dev)
# without the -dev packages: make LIBS="-l:liblapack.so.3 -l:libblas.so.3 -lm -lpthread"
make test                 # brute force, regression, checkpoint restart (about 2 min)
./qap_bb solve ../data/qaplib/nug20.dat -strat C -ub 2571 -nointprune
./qap_bb par ../data/qaplib/nug22.dat -strat C -threads 4 -ckpt ckpt.txt -ckpt_s 600
./qap_bb estimate ../data/qaplib/nug30.dat -strat D -dives 400 -q 1.5
```

`-ub opt+1 -nointprune` reproduces the paper's conditions: optimality is proved with an upper bound fixed at the optimum plus one.

### How a gain is measured

Every change first passes `make test`. The gain is measured on five instances (nug14, nug15, nug16a, nug16b, nug17), fully solved under the paper's conditions, on one core, keeping the best of two runs. We publish the sum of times, the per-instance gain and their geometric mean, and the node-count difference when the tree changes. The current table is in the "From nug25 to nug30" tab of the site.

### References

See the list in the French section. Copyrighted papers are not included in this repository.
