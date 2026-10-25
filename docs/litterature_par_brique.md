# Bilan de la littérature, brique par brique (nuit du 24 au 25 octobre 2026)

But de ce bilan : trouver, sans changer de borne, de quoi résoudre nug30 en une journée sur 4 cœurs avec QPB. Chaque source porte son mode de lecture, car la plupart des éditeurs et plusieurs dépôts sont bloqués depuis la machine de calcul :

- **lu en entier** : texte complet lu ;
- **lu (résumé de page)** : texte complet passé par un outil de lecture qui en rend un résumé ; les chiffres cités viennent de ce résumé et sont à revérifier sur le PDF ;
- **résumé seul** : seul l'abstract de l'éditeur a été lu.

Ce que nous avons mesuré nous-mêmes est signalé par « mesuré ». Les briques vont des couches basses (celles qui prennent le temps de calcul) vers les couches hautes (celles qui fixent la taille de l'arbre).

## Où va le temps aujourd'hui

Profil sur nug20 (gprof) : LAP 56 %, produit A·P·B incrémental 16 %, évaluation du nœud 7 %, corps de Frank-Wolfe 6 %, produits de matrices du point de départ hérité 5 %, LAP de mise en place 4 %, valeurs propres 4 %. Sur nug30 (mesuré avec `-timeig`), une décomposition spectrale coûte 17 µs et pèse environ 10 % du temps. Un nœud élagué coûte en moyenne 6,4 itérations de Frank-Wolfe, un nœud branché environ 40.

Conséquence : même si le LAP et les valeurs propres devenaient gratuits, le gain plafonnerait vers × 2,5. Les gains de vitesse par nœud que l'on peut encore espérer sont de l'ordre de × 1,1 à × 1,3. Le facteur manquant (× 15 environ) doit venir de la taille de l'arbre.

## Brique 1 : le LAP dense de petite taille (m de 2 à 29)

| Source | Ce qu'elle apporte | Ce que nous en avons fait | Lecture |
|---|---|---|---|
| Jonker et Volgenant, Computing 38 (1987) | LAPJV : réduction de colonnes, transfert de réduction, deux passes de réduction de lignes augmentante, puis plus courts chemins (Dijkstra avec ensemble SCAN). L'initialisation prend 60 à 70 % du temps ; plus rapide que ses concurrents à n = 50 et 200. Rien sur le démarrage à chaud. | Notre LAP en dérive. Mesuré : le départ JV complet est plus lent chez nous que le hongrois démarré à chaud (11 lignes libres au lieu de 5). | lu (résumé de page) |
| Dell'Amico et Toth, survey du LAP dense (Discrete Applied Math. 2000) | Comparaisons à partir de n = 200 seulement ; LAPm, APC et CTCS sont bons ; l'initialisation JV est recommandée. Rien sur les tailles 10 à 30 qui nous concernent. | Aucune piste nouvelle pour nos tailles. | lu (résumé de page) |
| Enchères (Bertsekas), utilisées sur GPU par Gonçalves et al. | Bonnes en parallèle massif ; sur un cœur et m ≤ 30, les plus courts chemins restent meilleurs dans les comparaisons de Dell'Amico et Toth. | Non essayé. | méthode connue, non relue cette nuit |
| Notre propre mesure | Démarrage à chaud par les prix des colonnes de l'itération précédente : LAP deux fois plus rapide. AVX-512, simple précision, quatre LAP par registre : plus lents ou perdants pour m ≥ 18. | Gardé : démarrage à chaud. Rejeté : le reste. | mesuré |

Ce qui reste à tenter dans cette brique : réoptimiser le LAP quand seule une ligne du gradient change beaucoup (la littérature sur les « LAP séquentiels » que nous avons cherchée cette nuit ne donne rien de précis pour m ≤ 30 ; à creuser avec un accès bibliothèque).

## Brique 2 : les valeurs propres des sous-matrices

| Source | Ce qu'elle apporte | Ce que nous en avons fait | Lecture |
|---|---|---|---|
| Golub, SIAM Review 15 (1973) ; Bunch, Nielsen et Sorensen, Numer. Math. 31 (1978) ; Gu et Eisenstat, SIMAX 15 (1994) | Après une modification de rang un d'une matrice symétrique diagonalisée, les nouvelles valeurs propres sont les racines d'une équation séculaire, en O(n²) au lieu de O(n³) ; vecteurs propres en forme close, calcul stable chez Gu et Eisenstat. LAPACK la fournit (`dlaed4`, nom vu dans sa documentation, non lue). | Idée non écrite : quand on fixe une usine, la matrice projetée perd une ligne et une colonne, ce qui s'écrit comme deux modifications de rang un. On pourrait passer de 17 µs à quelques µs. Gain plafonné à environ 10 % du temps. | résumé seul (page encyclopédique qui cite ces trois articles) |
| Notre propre mesure | Les décompositions ne dépendent que des usines et emplacements libres : on les réutilise entre frères. | Gardé (A2 du journal). | mesuré |

## Brique 3 : Frank-Wolfe pour QPB

| Source | Ce qu'elle apporte | Ce que nous en avons fait | Lecture |
|---|---|---|---|
| Brixius et Anstreicher, OMS 16 (2001), sections 2 à 4 | Le schéma que nous reproduisons : Frank-Wolfe, mise à jour de S et T, règles 1 à 4 de branchement, stratégies C et D. | Reproduit, puis accéléré (journal, chantiers A et B). | lu en entier |
| Braun, Pokutta et Zink, « Lazy Frank-Wolfe » (arXiv 1610.05120) | Garder en mémoire les sommets déjà vus et n'appeler l'oracle (ici le LAP) que si aucun sommet en cache ne progresse assez. | Piste ouverte : sur nos nœuds élagués (6,4 itérations), le cache serait petit ; gain incertain. | lu (résumé de page) |
| Lacoste-Julien et Jaggi, Frank-Wolfe par paires | Convergence linéaire sur un polytope. | Mesuré : racine de nug30 meilleure (5 318 contre 5 295 après 50 itérations), nug22 −10 % de nœuds, mais temps × 1,03 seulement. | méthode connue, non relue cette nuit |
| Hendrych et al., Boscia (arXiv 2208.11010) | Frank-Wolfe dans un B&B : démarrage depuis l'ensemble actif du parent, arrêt précoce sur l'écart de Frank-Wolfe, resserrement des bornes par l'écart dual. | Démarrage depuis le parent déjà fait (B3, × 1,2 à 1,3). Arrêt précoce déjà présent par les seuils de progression. Le resserrement des bornes correspond à nos paires interdites héritées (C1), sans gain mesuré. | lu (résumé de page) |
| Buchheim et al., B&B à Frank-Wolfe (arXiv 1507.05914) | Démarrage à chaud ; on élague dès que la borne inférieure de Frank-Wolfe dépasse le seuil, sans attendre la convergence. | Déjà fait chez nous (test à chaque itération). | lu (résumé de page) |

## Brique 4 : la borne elle-même (famille QPB et voisines)

| Source | Valeur à la racine de nug30 | Coût | Lecture |
|---|---|---|---|
| QPB (Anstreicher et Brixius 2001) | 5 365 dans les tableaux de Mittelmann et Peng ; 5 341 à 5 363 chez nous selon S, T et le nombre d'itérations | 0,3 ms par nœud en moyenne dans notre arbre | mesuré |
| Xia, OMS 23 (2008) : relaxation en cône du second ordre (SOCP) | au moins QPB | O(n²) variables | résumé seul |
| Xia et Gharibi, J. Comb. Optim. 30 (2015) | deux bornes : la première, équivalente à la SOCP de 2008, toujours meilleure que QPB mais chère ; la seconde « strictement meilleure que QPB dans la plupart des cas » et aussi facile à calculer | non dit dans l'abstract | résumé seul (texte complet inaccessible) |
| Ding et Wolkowicz, MSDR3 (rapport CORR 2006-22) | 5 446 | 12 206 s avec SeDuMi | lu (résumé de page) |
| Mittelmann et Peng (optimization-online 2008, publié en 2010) : découpage de la distance de Manhattan, modèle (31) | 5 478 | 2 unités de temps du tableau (non précisées dans le résumé) | lu (résumé de page) |
| Notre essai SDP « lifté » (Python, exploration seulement) | nug12 : 529 (QPB 482) ; nug15 : 1 070 (QPB 992) | secondes à minutes | mesuré |
| de Klerk et Sotirov, SDP et symétrie de groupe | meilleures bornes connues sur plusieurs instances de QAPLIB | lourd | résumé seul |
| Fujii et al., DNN lagrangienne et crochet de Newton (arXiv 2101.09629) | très proche de l'optimum ; nug30 résolu avec 1 728 cœurs et énumération à m ≤ 7 | lourd | lu (résumé de page) |

L'idée de Mittelmann et Peng mérite d'être notée ici car elle touche nos instances de près. Pour une grille k × l, la distance de Manhattan s'écrit B = Br + Bc (lignes et colonnes), et ((k−1)/2) E − Br et ((l−1)/2) E − Bc sont semi-définies positives. Comme X E Xᵀ = E pour une permutation, le coût devient une constante moins Tr(A Yr) − Tr(A Yc) avec Yr = X B̄r Xᵀ et Yc = X B̄c Xᵀ semi-définies, de diagonale et de sommes de lignes connues, et couplées par yr_ij + yc_ij ≤ (k + l − 4)/2 pour i ≠ j (deux emplacements distincts sont à distance au moins 1). C'est ce couplage qui fait le gain : nous avons essayé cette nuit la somme de deux QPB, une par axe, sans couplage, et obtenu 5 034 sur nug30, moins bien que QPB (mesuré).

Le point le plus prometteur pour rester dans la famille QPB est la seconde borne de Xia et Gharibi 2015, annoncée aussi facile que QPB. Il faut son texte complet.

## Brique 5 : branchement, dominance, symétrie

| Source | Ce qu'elle apporte | Ce que nous en avons fait | Lecture |
|---|---|---|---|
| Brixius et Anstreicher 2001 ; Anstreicher, Brixius, Goux et Linderoth 2002 | Règles 1 à 4, anticipation par borne des enfants, symétries de la grille | Reproduit | lu en entier (OMS 2001) |
| Gonçalves et al., RLT2 sur GPU | Branchement fort sur la ligne ou colonne de plus grande borne ; seuil de progression relative K de 0,01 % à 100 %, non publié par instance ; un sous-arbre par fil | Notre anticipation fait déjà un branchement fort | lu (résumé de page) |
| Dominance par échange | Nous n'avons trouvé dans la littérature lue aucune règle de dominance pour le QAP général dans un B&B à borne convexe. La règle écrite cette nuit (`-dom`) compare une affectation partielle à celle obtenue par échange de deux usines fixées, avec un maximum exact en O(m log m) par l'inégalité de réarrangement. | Mesuré : nug22 × 1,97 ; nug24 −6,8 % de nœuds ; nug30, aucun effet sur l'estimation | recherche |

## Brique 6 : estimation de la taille de l'arbre

| Source | Ce qu'elle apporte | Lecture |
|---|---|---|
| Knuth, Math. Comp. (1975) | Estimateur par plongées aléatoires, utilisé dans l'article de 2002 | méthode connue, non relue cette nuit |
| Hendel, Anderson, Le Bodic et Pfetsch (2022) | Estimation en cours de résolution par fréquence des feuilles et forêt aléatoire, à un facteur 3 près | résumé seul |

Nos estimations de nug30 (profil par profondeur, 6 120 plongées) donnent 1,6 à 2,5 × 10¹⁰ nœuds selon la graine et le seuil, cohérent avec les 1,19 × 10¹⁰ de 2002. L'écart-type d'une seule estimation est grand : deux graines pour une même configuration diffèrent couramment de 30 %.

## Brique 7 : parallélisme et points de comparaison

| Calcul | Moyens | Lecture |
|---|---|---|
| Anstreicher, Brixius, Goux et Linderoth 2002 | 11 892 893 499 nœuds, 6,94 années-CPU de HP C3000, environ 7 jours de grille | notre point de départ |
| Gonçalves et al. (RLT2, GPU) | nug30 : 103 380 nœuds, 88 867 s | lu (résumé de page) |
| Fujii et al. 2021 | nug30 avec 1 728 cœurs | lu (résumé de page) |

## Ce que ce bilan change pour la suite

1. Les couches basses ne donnent plus que × 1,1 à × 1,3 (LAP séquentiel, valeurs propres par équation séculaire). À écrire quand même, car elles se multiplient avec le reste.
2. La taille de l'arbre est le levier. Dans la famille QPB, deux candidates : la seconde borne de Xia et Gharibi 2015 (texte à obtenir) et une version bon marché du couplage de Mittelmann et Peng.
3. Les règles d'élagage combinatoires (dominance) aident sur nug22 mais pas sur nug30, où 95 % des nœuds évalués à la profondeur 10 sont élagués par la borne elle-même.
