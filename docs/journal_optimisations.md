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
| **Total face au code de 2000** | **7,22** | **6,85** | **9,91** | **6,84** | **8,33** | **3,71** | **4,31** |
| Temps 2000 → 2026 (s) | 7,15 → 0,99 | 4,25 → 0,62 | 18,8 → 1,90 | 5,40 → 0,79 | 59,1 → 7,09 | 285 → 76,9 | 691 → 160 |

Lecture :
- Les modifications qui accélèrent chaque nœud (A1, A4, A5, A6, B1) gardent leur gain quand la taille croît ; A5 augmente même (× 1,3 à × 1,75), car la part du LAP croît avec m.
- Les modifications qui portent sur l'anticipation (A2 en partie, A3, B2) perdent de leur effet : leur gain suit la part de l'anticipation dans le temps total (80 % sur nug17, 20 % sur nug22 en stratégie C).
- Stratégie D sur nug20 : code de 2000 717,7 s, code actuel 89,5 s (× 8,0), dont B2 × 1,98 ; la stratégie C reste plus rapide en temps absolu (76,9 s).
- Nombre de nœuds, code actuel / code de 2000 : nug14 0,75 ; nug15 0,92 ; nug16a 0,96 ; nug16b 0,99 ; nug17 1,02 ; nug20 1,08 ; nug22 0,94.

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
| Plafond de convergence | FW avec 2 et 4 fois plus d'itérations | nug20 : −10 % et −13 % de nœuds, temps + 44 % et + 120 % |

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
