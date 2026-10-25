# Journal des optimisations (nug30 Laboratory)

Ce journal garde, pour chaque modification du code de 2000 reproduit, le principe mathématique, la raison pour laquelle la borne reste valide, et la mesure. Il servira de base à un article. Les essais sans gain y figurent aussi.

**Mesure.** Banc de cinq instances (nug14, nug15, nug16a, nug16b, nug17), stratégie C, borne supérieure fixée à l'optimum plus un (`-ub opt+1 -nointprune`), un cœur Xeon 2,1 GHz, meilleur de deux exécutions sauf mention contraire. Gain = somme des temps avant / après ; on donne aussi les gains par instance et leur moyenne géométrique. Les modifications de l'algorithme sont aussi mesurées sur nug20 et nug22. Chaque modification passe `make test` : comparaison à la force brute (LAP, validité de la borne, séparation et évaluation complète), non-régression, reprise après checkpoint.

**Notations.** Au nœud courant, m usines libres, matrices réduites Ar, Br (m × m), terme linéaire Cr, constante K. V est une base orthonormée de l'orthogonal de e. La borne QPB (Anstreicher et Brixius, 2001) minimise sur les matrices doublement stochastiques X

f(X) = ⟨X, Ar X Br⟩ − ⟨X, S X⟩ − ⟨X, X T⟩ + ⟨Cr, X⟩ + tr S + tr T,

avec S = V Ŝ Vᵀ, T = V T̂ Vᵀ. Frank-Wolfe (FW) donne à chaque itération k la borne z_k = f(X_k) − ⟨G_k, X_k⟩ + LAP(G_k) + K, où G_k = 2 M(X_k) + Cr et M(X) = Ar X Br − S X − X T.

## Chantier A : implémentation

### A1. Options du compilateur (× 1,07, même arbre)
`-funroll-loops -fno-math-errno -fno-trapping-math`. Aucune opération n'est réordonnée de façon non associative, donc les résultats sont identiques au bit près.

### A2. Cache des décompositions spectrales, A·P·B incrémental (× 1,27, arbre à ±2 %)
- Ŝ et T̂ dépendent seulement des valeurs et vecteurs propres de Vᵀ Ar V et Vᵀ Br V, donc de l'ensemble des lignes libres (resp. colonnes libres). Les frères d'un nœud et les candidats de l'anticipation partagent ces ensembles : les décompositions sont mises en cache par masque de bits.
- Ar P Br pour une permutation P : quand la permutation de l'itération k diffère de la précédente en d positions k₁, …, k_d, on a (Ar P Br) = (Ar P' Br) + Σ_t Ar[:, k_t] (Br[p(k_t)] − Br[p'(k_t)]), soit d mises à jour de rang un en O(m²) chacune au lieu d'un produit en O(m³).
- Point de départ X₀ = eeᵀ/m : Ar X₀ Br = (Ar e)(eᵀ Br)/m, et S e = 0, eᵀ T = 0, donc M(X₀) se calcule en O(m²).
- L'ordre des sommes change, donc les arrondis : nug17 passe de 58 918 à 58 818 nœuds.

### A3. Un seul calcul par enfant dans l'anticipation (× 1,56, même arbre)
Avec les règles 3 et 4, l'enfant (i → j) figure dans la liste de la ligne i et dans celle de la colonne j. Le code de 2000 calculait deux fois sa borne prospective. Le calcul est déterministe (même nœud, mêmes données, même point de départ). S'il a lieu une seconde fois après une baisse de la borne supérieure, il pourrait seulement s'arrêter plus tôt ; la valeur gardée reste une borne valide, car chaque itération de FW en fournit une. Même nombre de nœuds avec et sans, sur les six instances du banc.

### A4. Réduction de lignes du LAP sans branchement (× 1,19, même arbre)
Profil en temps réel (rdtsc) : le LAP de Jonker-Volgenant prend 63 % du temps, dont 47 % dans la réduction de lignes (plus petit et deuxième plus petit coût réduit d'une ligne). Réécrite avec des sélections sans branchement, elle donne le même résultat au bit près.

### A5. LAP par plus courts chemins, espace de travail persistant (× 1,35, arbre à +0,2 %)
- La réduction de lignes faisait en moyenne 57 réaffectations par appel pour libérer environ 5 lignes. Remplacement : affectation gloutonne de chaque ligne à sa colonne de plus petit coût réduit, puis un plus court chemin augmentant (Dijkstra sur tableau, O(m²)) par ligne libre, avec mise à jour des duales v_j ← v_j + d_j − d_min sur les colonnes atteintes. Boucles sans branchement, vectorisées par le compilateur.
- Pourquoi l'arbre change : la valeur optimale du LAP est unique, mais pas les duales optimales (u, v). Or U = G − u − v sert au choix de la ligne ou colonne de branchement et à la borne des enfants. Des duales différentes, toutes optimales, donnent d'autres décisions. Nœuds : nug14 +6,2 %, nug15 −0,3 %, nug16a −0,3 %, nug16b −1,5 %, nug17 +0,25 %.
- L'anticipation réallouait 8,6 Mo (toutes les matrices de FW de taille 32 × 32) à chaque branchement ; l'espace est désormais gardé par fil.

### A6. Frank-Wolfe en moins de passages (× 1,17, même arbre)
Chaque itération parcourait huit fois des matrices m × m. Identités utilisées :
- f = ⟨M X, X⟩ + ⟨Cr, X⟩ + tr S + tr T, donc ⟨G, X⟩ = 2⟨MX, X⟩ + ⟨Cr, X⟩ = 2(f − tr S − tr T − s₂) + s₂ avec s₂ = ⟨Cr, X⟩ ;
- s₂ se met à jour en O(m) : s₂ ← (1 − α) s₂ + α Σ_i Cr[i][p(i)] ;
- le pas de ligne q = ⟨P − X, M P − M X⟩ = Σ_i (MP − MX)[i][p(i)] − ⟨X, MP⟩ + ⟨X, MX⟩, où ⟨X, MP⟩ est accumulé pendant le calcul de MP ;
- G suivant = 2 MX + Cr est écrit pendant la mise à jour de X et MX ;
- duales du LAP : u_i = c_{i,σ(i)} − v_{σ(i)} (écarts complémentaires) au lieu d'un minimum par ligne.
Seuls les arrondis changent (quelques itérations de FW sur nug12 et nug17), nombre de nœuds identique.

### A7. Terme linéaire Cr en produit dense (nug22 × 1,04, même arbre)
Cr[a][b] = 2 Σ_k A[F_a][k] B[L_b][π(k)] + C0, la somme portant sur les usines k déjà fixées. Le code parcourait les N usines avec un test par k. On rassemble d'abord les colonnes fixées dans deux tableaux m × n_f (Af[a][t] = A[F_a][k_t], Bf[b][t] = B[L_b][π(k_t)]), puis Cr = 2 Af Bfᵀ + C0 en boucles contiguës. Les termes sont sommés dans le même ordre, donc le résultat est identique et l'arbre aussi. Gain plus grand attendu sur nug30 (N et le nombre d'usines fixées augmentent).

## Chantier B : algorithme

### B1. Mise à jour de S et T, et itérations de FW × 0,7 (banc × 1,19 ; nug20 × 1,11 ; nug22 × 1,20)
- Brixius et Anstreicher (OMS 2001) mettent à jour S et T toutes les 30 itérations (paramètre UPDATE) ; notre reproduction ne le faisait pas.
- Validité pour tout couple (s, t) avec s_i + t_j ≤ λ_i μ_j (λ, μ valeurs propres de Vᵀ Ar V et Vᵀ Br V, Ŝ = W_A diag(s) W_Aᵀ, T̂ = W_B diag(t) W_Bᵀ) : la forme quadratique est convexe sur l'enveloppe affine des matrices doublement stochastiques, et pour une permutation P, ⟨P, SP⟩ = tr S et ⟨P, PT⟩ = tr T, donc f(P) est exactement le coût de P. Le minimum de f sur le polytope minore donc l'optimum.
- Choix de (s, t) au point X courant : f = ⟨X, Ar X Br⟩ + ⟨Cr, X⟩ + Σ_l s_l α_l + Σ_l t_l β_l avec α_l = 1 − ‖Xᵀ V w_l‖², β_l = 1 − ‖X V w'_l‖². On a Σ α_l = Σ β_l = m − ‖X‖². Maximiser sur l'ensemble réalisable est le dual d'un problème de transport de coûts λ_i μ_j et de marges α, β. Avec λ croissant et μ décroissant, ces coûts vérifient la propriété de Monge ((λ_k − λ_i)(μ_l − μ_j) ≤ 0 pour i < k, j < l) : la règle du coin nord-ouest est optimale, et sa base en escalier donne les duales. Coût O(m³) pour les produits Xᵀ V W, appelé toutes les 30 itérations ; on vérifie la réalisabilité et on n'accepte que si f augmente.
- Racine de nug30, 150 itérations : z passe de 5 325,7 à 5 349,3 ; à convergence, 5 341 sans mise à jour contre 5 363,9 avec, proche de la valeur de 2001 (écart à l'optimum 12,4 %, soit 5 365).
- Les limites NFW1, NFW2, NFW3 de 2001 ont été fixées pour des machines environ 25 fois plus lentes. Multipliées par 0,7 : plus de nœuds (+6 à +9 % sur le banc), mais moins chers.

### B2. Anticipation : abandon des candidates sans espoir (banc × 1,65 ; nug20 × 1,15 ; nug22 × 1,09)
- Mesure préalable (chronomètre autour de la boucle d'anticipation) : l'anticipation prend 85 % du temps sur nug16a, 78 % sur nug17, mais 30 % sur nug20 et 20 % sur nug22. La stratégie C ne l'applique qu'aux niveaux 0 à 3 comptés depuis la racine ; les grands arbres ont l'essentiel de leurs nœuds plus bas.
- Règle 3 de 2001 : pour chaque ligne ou colonne candidate, on calcule la borne z_ab de chacun de ses enfants (quelques itérations de FW) et on garde la candidate de plus grande somme Σ z_ab.
- Modification : un enfant dont la borne dépasse le seuil d'élagage θ est éliminé quelle que soit sa valeur exacte ; on remplace donc z_ab par min(z_ab, θ) dans la somme. Chaque terme restant est alors au plus θ : si la somme partielle plus θ fois le nombre d'enfants restants ne dépasse pas la meilleure somme connue, la candidate ne peut plus gagner et on l'abandonne. Les enfants sont parcourus par U croissant (les plus prometteurs d'abord) pour que la somme partielle reste basse tôt.
- Validité : seule la règle de choix de la ligne de branchement change ; toute règle de choix donne un arbre correct, les bornes des enfants restent celles de FW.
- Effet : la troncature favorise les candidates qui éliminent le plus d'enfants plutôt que celles dont les bornes dépassent de beaucoup le seuil, ce qui réduit aussi le nombre de nœuds (nug14 −36 %, nug17 −4 %, nug20 −0,9 %). Gains par instance du banc : 1,53 / 1,69 / 1,42 / 1,52 / 1,77.
- Leçon : le banc de petites instances surestime les gains de l'anticipation. Chaque gain doit désormais être mesuré aussi sur nug20 et nug22.

### B3. Les enfants partent du point final du père (banc × 1,08 ; nug20 × 1,21 ; nug22 × 1,29)
- Avant : chaque nœud lançait Frank-Wolfe depuis X₀ = eeᵀ/m.
- Maintenant, dans la recherche en profondeur série, le père garde son point final X (m × m). L'enfant qui fixe la ligne a sur la colonne b part de X privé de la ligne a et de la colonne b, plus le terme de rang un X[:, b] X[a, :] / (1 − X[a][b]). Sommes de lignes : Σ_q≠b X[p][q] + X[p][b] (1 − X[a][b]) / (1 − X[a][b]) = 1 ; idem pour les colonnes. Le point est donc doublement stochastique, et tout point de départ donne une borne valide.
- Effet sur nug22 : itérations de FW 34,3 M → 23,9 M (−30 %), nœuds 2,04 M → 1,91 M (−6 %). Le père et l'enfant ont des problèmes proches, et l'enfant près du seuil de coupe l'atteint plus vite.
- Différence avec l'essai sans gain du même nom (tableau ci-dessous) : celui-là ne portait que sur les enfants évalués par l'anticipation, avec 18 à 35 itérations seulement.
- Variante : réajuster S et T au point de départ (`-warm 6`) : nug20 × 1,23, nug22 × 1,22 ; non retenue.
- Reprise après checkpoint : les nœuds relus dans le fichier partent du centre (le point du père n'est pas sauvé), donc l'arbre repris diffère un peu (nug17 : 57 108 nœuds contre 57 085). Le test vérifie la reprise exacte avec `-warm 0` et l'écart de moins de 1 % avec le défaut.
- Recherche parallèle (4 fils) : chaque branchement range le point du père dans un bloc partagé par ses enfants sur la pile du fil, libéré par le dernier enfant ; un nœud rendu au réservoir commun repart du centre. nug22 sur 4 cœurs : 42,2 s → 34,8 s (× 1,21), 1,91 M nœuds. Efficacité face au série : 94 % sans ce départ, 88 % avec (mesures série faites à 4 calculs simultanés, donc un peu pessimistes).

### Mesure directe face au code de 2000 (24 octobre, 19 h 30 UTC)
Même machine, même charge, stratégie C, borne supérieure optimum + 1 : nug20 285,8 s → 75,3 s (× 3,80) ; nug22 691,5 s → 155,0 s (× 4,46).

## Gain de chaque modification selon la taille (24 octobre, 20 h UTC)

Méthode : on part du code actuel et on désactive les modifications une à une, dans l'ordre inverse de leur introduction, sur chaque instance (stratégie C, un cœur, borne opt + 1, quatre calculs à la fois sur la machine pour toutes les mesures). La ligne « reste » est le rapport entre le code de 2000 et le code actuel une fois tout ce qui peut l'être désactivé ; elle regroupe la réduction de lignes sans branchement et l'espace de travail persistant de l'anticipation.

| Modification | nug14 | nug15 | nug16a | nug16b | nug17 | nug20 | nug22 |
|---|---|---|---|---|---|---|---|
| A1 Options du compilateur | 1,06 | 1,06 | 1,06 | 1,14 | 1,05 | 1,05 | 1,03 |
| A2 Cache des valeurs propres, A·P·B incrémental | 1,45 | 1,34 | 1,33 | 1,31 | 1,34 | 1,25 | 1,20 |
| A3 Un seul calcul par enfant | 1,57 | 1,58 | 1,51 | 1,45 | 1,40 | 1,16 | 1,07 |
| A4 + A5 (reste) | 1,18 | 1,22 | 1,23 | 1,15 | 1,25 | 1,22 | 1,18 |
| A5 LAP par plus courts chemins | 1,32 | 1,22 | 1,43 | 1,46 | 1,36 | 1,41 | 1,75 |
| A6 Frank-Wolfe fusionné | 1,26 | 1,30 | 1,12 | 0,99 | 1,16 | 1,16 | 1,16 |
| B1 Mise à jour S, T et itérations × 0,7 | 1,09 | 1,09 | 1,20 | 1,26 | 1,20 | 1,06 | 1,20 |
| B2 Abandon des candidates | 1,39 | 1,44 | 1,96 | 1,49 | 1,78 | 1,16 | 1,13 |
| B3 Enfants au point du père (mesuré à part) | 1,00 | 1,07 | 1,08 | 1,17 | 1,09 | 1,21 | 1,29 |
| **Total face au code de 2000 (sans B3)** | **7,22** | **6,85** | **9,91** | **6,84** | **8,33** | **3,71** | **4,31** |
| **Total avec B3** | **7,2** | **7,3** | **10,7** | **8,0** | **9,1** | **4,5** | **5,6** |
| Temps 2000 → 2026 (s) | 7,15 → 0,99 | 4,25 → 0,62 | 18,8 → 1,90 | 5,40 → 0,79 | 59,1 → 7,09 | 285 → 76,9 | 691 → 160 |

Lecture (tendance : croît pour A5 et B3, stable pour A1, A4, A6, B1, décroît pour A2, A3, B2) :
- Les modifications qui accélèrent chaque nœud (A1, A4, A5, A6, B1, B3) gardent leur gain quand la taille croît ; A5 augmente même (× 1,3 à × 1,75), car la part du LAP croît avec m.
- Les modifications qui portent sur l'anticipation (A2 en partie, A3, B2) perdent de leur effet : leur gain suit la part de l'anticipation dans le temps total (80 % sur nug17, 20 % sur nug22 en stratégie C).
- Stratégie D sur nug20 : code de 2000 717,7 s, code actuel 89,5 s (× 8,0), dont B2 × 1,98 ; la stratégie C reste plus rapide en temps absolu (76,9 s).
- Nombre de nœuds, code actuel / code de 2000 : nug14 0,75 ; nug15 0,92 ; nug16a 0,96 ; nug16b 0,99 ; nug17 1,02 ; nug20 1,08 ; nug22 0,94.

## Chantier C : coupes sans changer de borne (nuit du 24 au 25 octobre)

Objectif fixé par Jean-Pierre : battre le temps de l'arbre RLT2 de Gonçalves et al. (nug30, 103 380 nœuds, 88 867 s avec un GPU) en gardant QPB. Point de départ : l'estimation de l'arbre de nug30 en stratégie D vaut 1,6 à 2,5 × 10¹⁰ nœuds, soit 5 à 6 × 10⁶ s sur un cœur (environ 15 jours sur 4 cœurs) ; il manque un facteur 15 environ.

### Profil de l'arbre de nug30 par profondeur (graine 19, `-top 3`, 6 120 plongées)

Estimation 1,62 × 10¹⁰ nœuds, 5,18 × 10⁶ s. Par niveau (nœuds estimés, temps, part des nœuds élagués après évaluation) :

| Niveau | m | Nœuds | Temps (s) | Élagués après FW |
|---|---|---|---|---|
| 4 | 26 | 1,6 × 10⁵ | 12 500 | 4 % |
| 5 | 25 | 3,2 × 10⁶ | 2 100 | 18 % |
| 6 | 24 | 5,3 × 10⁷ | 26 000 | 46 % |
| 7 | 23 | 4,6 × 10⁸ | 196 000 | 65 % |
| 8 | 22 | 2,5 × 10⁹ | 760 000 | 81 % |
| 9 | 21 | 4,7 × 10⁹ | 1 280 000 | 78 % |
| 10 | 20 | 7,3 × 10⁹ | 2 700 000 | 95 % |
| 11 | 19 | 1,2 × 10⁹ | 200 000 | 100 % |

Lecture : la moitié du temps va aux nœuds du niveau 10, presque tous élagués dès leur évaluation. Sur nug20 (stratégie C, avec la coupe C2 ci-dessous), un nœud élagué coûte 6,4 itérations de FW en moyenne et un nœud qui se sépare 40 ; les nœuds élagués sont nombreux mais bon marché.

### C1. Paires interdites héritées (sans gain)
- Principe : au nœud, pour toute permutation P du sous-arbre, coût(P) ≥ z + ⟨U, P⟩ avec U ≥ 0. Si z + U_ab dépasse le seuil, aucune permutation du sous-arbre ne contient (a, b) : le couple reste interdit chez tous les descendants. Dans le LAP de FW, le coût des couples interdits passe à 10¹² ; la borne reste valide car f(P) ≥ f(X) − ⟨G, X⟩ + ⟨G′, P⟩ ≥ f(X) − ⟨G, X⟩ + LAP(G′) pour toute permutation P qui évite ces couples, X pouvant être n'importe où.
- Mesure : optimums retrouvés de nug12 à nug18, nœuds à ±0,5 %. Les couples interdits avaient déjà un grand coût réduit, le LAP les évitait de lui-même.

### C2. Dominance par échange (`-dom`) : nug22 × 1,97
- Principe : soient deux installations placées i → j et k → l. Échanger leurs emplacements change le coût de toute complétion p de
  Δ = (A_ii − A_kk)(B_ll − B_jj) + C0_il + C0_kj − C0_ij − C0_kl + 2 Σ_{t placée, t ≠ i, k} (A_it − A_kt)(B_{l p(t)} − B_{j p(t)}) + 2 Σ_{t libre} α_t β_{p(t)},
  avec α_t = A_it − A_kt et β_c = B_lc − B_jc. La dernière somme est une affectation de coût de rang un : son maximum sur les complétions s'obtient en rangeant α et β dans le même ordre (inégalité de réarrangement), en O(m log m). Si même ce maximum laisse Δ < 0, toute permutation du sous-arbre est strictement améliorée par l'échange.
- Validité : une permutation optimale n'est jamais strictement améliorée, donc aucun sous-arbre coupé ne contient de permutation optimale. Avec la coupe par la borne et la réduction par symétrie : l'orbite d'une permutation optimale est faite de permutations optimales, aucune n'est coupée par dominance, et la symétrie en garde au moins une.
- Coût : d(d − 1)/2 paires par nœud au niveau d, deux tris de m nombres chacune (tri par insertion, plus rapide que qsort ici).
- Mesures (stratégie C, ub = opt + 1, un cœur, machine chargée à 4 ou 5 calculs ; depuis le redémarrage de 23 h 26 UTC la machine est un Xeon à 2,8 GHz avec AVX-512, les temps ne se comparent donc pas à ceux des sections précédentes ; les nombres de nœuds sont les mêmes, seuls les nombres d'itérations de FW bougent un peu avec le processeur) :

| Instance | Nœuds sans | Nœuds avec | Temps sans | Temps avec |
|---|---|---|---|---|
| nug16a | 10 746 | 7 526 | 2,27 s | 1,94 s |
| nug17 | 57 085 | 40 520 | 7,4 s | 6,1 s |
| nug18 | 202 854 | 164 176 | 21,5 s | 19,1 s |
| nug20 | 992 813 | 982 633 | 79,3 s | 76,6 s |
| nug22 | 1 908 513 | 928 329 | 160,4 s | 81,3 s |

  Le gain dépend de l'instance : fort sur nug22 (grille 2 × 11), presque nul sur nug20 (grille 4 × 5).
- Je n'ai pas trouvé cette règle dans les sources lues cette nuit ; la recherche n'a pas été exhaustive.

### C2 bis à C2 quinquies : variantes de la dominance (sans gain)
| Variante | Idée | Résultat |
|---|---|---|
| Installation placée contre emplacement libre (`-dom2`) | i → j placée, emplacement libre l : quel que soit le libre t qui prend l, i va en l et t en j ; même formule de rang un, à vérifier pour tout t | nœuds identiques à `-dom` sur nug12 à nug17, temps + 50 % |
| Borne et échange mêlés (`-domlb`) | pour λ ∈ {0,3 ; 0,5 ; 0,7} : si min_P λ(z + ⟨U, P⟩ − seuil) − (1 − λ)(D + ⟨W, P⟩) > 0, toute P dépasse le seuil ou est améliorée ; un LAP par paire et par λ | 64 nœuds de plus coupés sur nug17, plus lent |
| Couverture par plusieurs échanges (`-domcov k`) | combinaison convexe de tous les échanges et de la borne, poids par montée de sous-gradient exponentiée, un LAP par pas | k = 20 : nug17 40 257 nœuds au lieu de 40 520, temps × 1,5 |
| Cycles de trois installations placées (`-dom3`) | partie libre de rang deux, majorée par la somme de deux maximums de rang un | nug17 −2 % de nœuds, nug18 −0,3 %, plus lent |
| Filtrage des enfants dominés au moment de séparer (`-dombr`) | les enfants dominés ne comptent plus dans la règle 2 | nug17 35 881 nœuds, nug18 149 529, mais temps égal ou supérieur |

## Profil du code actuel (nug20, 24 octobre 20 h 30 UTC)

gprof, LAP et A·P·B non incorporés : LAP par plus courts chemins 56 %, A·P·B incrémental 16 %, évaluation du nœud 7 %, corps de Frank-Wolfe 6 %, produits de matrices (point de départ hérité) 5 %, LAP de mise en place 4 %, valeurs propres 4 %. Par appel du LAP : m = 14,0 en moyenne, 5,4 lignes libres après l'affectation gloutonne, 5,4 colonnes balayées par ligne libre.

## Essais sans gain

| Essai | Idée | Résultat |
|---|---|---|
| Même boucle LAP en AVX-512 | registres de 8 doubles | plus lent (9,0 s contre 8,6 s sur nug16a) |
| Compilation guidée par profil, clang | | plus lent |
| LAP en simple précision | 16 valeurs par registre ; duales recalculées en double, u_i = min_j (c_ij − v_j), pour garder (u, v) réalisable | 8 % plus lent : conversion et passage final coûtent plus que le gain sur m ≤ 17 |
| Départ JV au lieu du hongrois, garder l'affectation précédente dans le LAP | | plus lent (11 lignes libres au lieu de 5) |
| Point de départ hérité du père pour les enfants de l'anticipation | X enfant = X père privé de la ligne a et de la colonne b, plus le terme de rang un X[:, b] X[a, :] / (1 − X[a][b]), qui rend la matrice exactement doublement stochastique | borne de chaque enfant +2 à +5 après 50 itérations (encore 10 sous la valeur convergée) ; arbre inchangé à 1 % près |
| Frank-Wolfe par paires (Lacoste-Julien et Jaggi) | X tenu comme combinaison convexe d'atomes ; on déplace du poids de l'atome de plus grand ⟨G, A⟩ vers la permutation du LAP | racine de nug30 après 50 itérations : 5 318 au lieu de 5 295 ; sur l'arbre, nug22 −10 % de nœuds mais temps × 1,03 seulement, et moins bon que FW classique quand on réduit les itérations |
| Anticipation décidée par la taille du sous-problème | règle 3 sous le niveau 3 tant que m ≥ k (k = 10 à 14, 1 à 5 candidates, 5 à 18 itérations) | nœuds divisés par 2 à 4 (nug17 60 174 → 14 755), mais temps plus long : nug17 6,8 s → 7,8 s au mieux, nug20 75 s → 100 s |
| LAP groupés dans un cœur | 4 LAP de même taille en parallèle, un par case d'un registre AVX2 (4 doubles) ; recherche du minimum sans réduction horizontale, mais la relaxation lit une ligne différente par case (lecture dispersée, gather) | micro-banc sur coûts aléatoires, temps par LAP : m = 10 × 1,25 ; m = 14 × 1,09 ; m = 18 × 0,94 ; m = 24 × 0,63 ; m = 29 × 0,74. Perdant aux tailles des nœuds de nug30 (m ≥ 18), non intégré |
| Recherche du minimum dans la passe de relaxation du LAP | une seule passe sur les colonnes au lieu de deux par étape de Dijkstra, même résultat au bit près | nug17 6,3 s → 9,2 s : la boucle fusionnée n'est plus vectorisée |
| LAP de mise en place de S, T en forme close | les coûts λ_i μ_j sont de Monge : affectation i → k − 1 − i et duales en escalier | autres duales optimales, donc autres S, T : nug17 57 085 → 64 095 nœuds, plus lent |
| Boucles du LAP alignées sur 4 colonnes | colonnes de remplissage à coût infini, sans reste scalaire | nug20 61,8 s → 61,2 s, dans le bruit |
| LAP approché une itération sur k | affectation gloutonne ; duales u_i = min_j (c_ij − v_j), réalisables, donc borne valide ; LAP exact si la direction n'est pas une descente ; U et la meilleure borne pris seulement sur les LAP exacts | k = 2 : nug17 6,09 s → 6,10 s, nug20 × 1,03 ; k = 3 : nug17 × 1,0. Les directions approchées demandent plus d'itérations |
| Itérations de FW × 0,5 et × 0,4 au lieu de × 0,7, avec le départ au point du père | | nug20 62,1 s → 56,9 s → 53,5 s, mais nug22 120,6 s → 123,8 s (+18 % de nœuds à × 0,5) ; × 0,7 gardé |
| A·P·B par symétrie | les matrices de nug sont symétriques | sans objet : Ar P Br n'est pas symétrique (P ne l'est pas) |
| Plafond de convergence | FW avec 2 et 4 fois plus d'itérations | nug20 : −10 % et −13 % de nœuds, temps + 44 % et + 120 % |
| Profondeur de l'anticipation, estimée sur nug30 (24 oct., 22 h UTC) | règle 4 jusqu'au niveau a, règle 3 jusqu'au niveau b ; (a, b) = (1, 4), (2, 3), (2, 4) = D, (2, 5), (3, 4), (3, 5) = E ; Knuth 300 plongées, graines 7 et 11 | moyenne géométrique des nœuds : 1,9 × 10¹⁰, 2,0 × 10¹⁰, 5,8 × 10⁹, 5,4 × 10⁹, 1,0 × 10¹⁰, 4,8 × 10⁹. Deux graines diffèrent parfois d'un facteur 10 : l'estimateur ne tranche pas à ×1,5 près ; moins d'anticipation est nettement pire |
| Profondeur de l'anticipation, mesurée sur nug24 entier (24 oct., 23 h UTC) | mêmes réglages, 1 cœur chacun, 4 calculs en même temps, code actuel | D (2, 4) : 22 784 046 nœuds, 2 025 s ; (3, 4) : 21 909 011, 2 603 s ; (2, 5) : 16 381 520, 2 942 s ; E : 15 977 220, 3 070 s. Un niveau de plus retire jusqu'à 30 % des nœuds mais chaque nœud coûte 1,5 fois plus : D gardé. Face au code de 2000 en D (3 h 05), × 5,5 sur nug24 |

## Vitesse d'un cœur de 2026 face à la HP C3000

Même code (celui de 2000), même instance, un cœur. Temps de l'article : OMS 2001, HP C3000.

| Instance | ms par nœud (2000) | ms par nœud (2026) | Rapport |
|---|---|---|---|
| nug18 (C) | 16,5 | 0,77 | × 21,6 |
| nug20 (C) | 8,4 | 0,31 | × 26,7 |
| nug21 (C) | 7,5 | 0,35 | × 21,3 |
| nug22 (C) | 6,6 | 0,34 | × 19,4 |
| nug24 (D) | 11,0 | 0,48 | × 23,0 (23,2 M nœuds en 3 h 05, optimum 3 488) |

nug30 a demandé 6,94 années de HP C3000, soit 94 à 127 jours sur un cœur de 2026, 24 à 32 jours sur 4 cœurs. Objectif retenu : × 30 par le code et l'algorithme.

### Suite de la nuit (01 h UTC, 25 octobre) : essais sans gain, en bref
- `-sens` (élimination par chemin alternant), `-umax`, `-ext`, `-r2p`, `-topd`, `GLBMIX`, `-dom2/-dom3/-domlb/-domcov` : au plus −5 % de nœuds, jamais de gain de temps net.
- `-glbpre 0` (Gilmore-Lawler avant Frank-Wolfe) : nug18, 195 nœuds élagués sur 388 438 appels ; inutile.
- Découpage de la distance de Manhattan par axe, sans couplage : nug30 5 034 contre 5 363 ; moins bon.
- SDP et reformulation convexe (QCR) à la racine seule (exploration en Python, hors du solveur) : nug12 529 contre 482 à la racine, mais arbre plus gros (34 442 nœuds contre 6 087).
- Décalages diagonaux de la racine (code actuel) : racine nug30 5 499 contre 5 349, mais nug18 392 210 nœuds contre 202 854.
- nug24 stratégie D avec `-dom` : 21 242 561 nœuds, 2 663 s (référence sans dom sur la nouvelle machine en cours).
- Estimations nug30 (graine 19) : 1,62 × 10¹⁰ nœuds sans dom, 1,74 × 10¹⁰ avec ; seuil de preuve ub = opt (graine 23) : 2,25 × 10¹⁰. Différences dans le bruit de l'estimateur.
- Bilan de la littérature par brique : `docs/litterature_par_brique.md`.
- Seconde borne QPB sur données décalées, utilisée seulement pour élaguer (`-dual`, décalages de la racine) : nug18, 245 nœuds élagués sur 182 246 essais ; inutile.
- Décalages optimisés nœud par nœud (pour PB, par Powell, puis QPB à 150 itérations ; mesure hors solveur, mode `root` avec `FIX`) : nug20, 60 nœuds de la frontière de profondeur 3, médiane 9 % de l'écart au seuil comblé, 3 nœuds élagués de plus (poids 10⁻⁵ de l'arbre) ; à la profondeur 6, médiane −3 % (les décalages bons pour PB dégradent QPB). Piste abandonnée.
