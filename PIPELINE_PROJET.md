# Pipeline — Segmentation de fruits par K-means dans les espaces RGB, HSV et Lab

## 1. Objectif et périmètre

**Sujet :** classification automatique de fruits en supermarché à partir de leur couleur dominante.

**Travail principal :** segmenter des images de pommes, bananes et poivrons, puis comparer K-means dans trois espaces couleur : **RGB, HSV et Lab**, avec **K ∈ {3, 4, 5, 6}**.

Le projet doit répondre à trois questions :

1. Quel espace couleur sépare le mieux les régions visibles des fruits et le fond ?
2. Comment le choix de K influence-t-il la segmentation ?
3. Quelles sont les limites de la couleur lorsque plusieurs fruits ont une apparence proche ?

**Périmètre étendu :** une reconnaissance supervisée de pommes, bananes et poivrons est implémentée en complément de l'étude K-means. Elle utilise un inventaire et des photos de diagnostic distincts, décrits dans [data/recognition/README.md](data/recognition/README.md). Les piments sont inclus dans la catégorie poivron. Aucun modèle préentraîné ni nouvel outil externe n'est nécessaire.

> K-means produit des groupes de pixels de couleurs proches. Un numéro de cluster n'est pas une classe de fruit. Reconnaître automatiquement « pomme », « banane » ou « poivron » nécessite une étape supplémentaire d'association ou de classification. Une pomme verte et un poivron vert peuvent appartenir au même groupe de couleur.

### Livrables attendus

- Une comparaison visuelle des segmentations dans les trois espaces couleur.
- Un tableau d'évaluation des clusters et des segmentations.
- Une explication des différences observées et des limites.
- Du code reproductible, un rapport et une courte présentation des résultats.

## 2. Vue d'ensemble

```text
Collecter et décrire les images
              ↓
Séparer les images de mise au point et d'évaluation
              ↓
Préparer les images et annoter un sous-ensemble
              ↓
Construire les représentations RGB, HSV et Lab
              ↓
Appliquer K-means pour K = 3, 4, 5 et 6
              ↓
Reconstruire et visualiser les cartes de clusters
              ↓
Évaluer la cohérence des clusters et la segmentation
              ↓
Comparer les espaces, choisir K et analyser les échecs
              ↓
Rédiger le rapport et préparer la démonstration
```

## 3. Environnement et organisation

### Outils proposés

| Outil | Utilisation |
| --- | --- |
| Python | Implémentation du pipeline |
| NumPy | Manipulation des pixels et des tableaux |
| scikit-image | Lecture, redimensionnement et conversions couleur |
| scikit-learn | K-means et mesures de qualité des clusters |
| Matplotlib | Comparaisons visuelles et graphiques |
| pandas | Tableaux et export CSV |
| Jupyter | Exploration et présentation des expériences |

Utiliser un environnement virtuel et enregistrer les versions des dépendances réellement installées. OpenCV reste facultatif ; s'il est utilisé, convertir explicitement ses images BGR en RGB.

### Arborescence cible

```text
mini-projet/
├── PIPELINE_PROJET.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Images originales, conservées intactes
│   ├── processed/           # Images préparées
│   ├── masks/               # Annotations de référence
│   └── metadata.csv         # Source, conditions, catégorie, partition
├── src/
│   ├── preprocessing.py
│   ├── color_spaces.py
│   ├── segmentation.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── run_experiments.py
├── results/
│   ├── figures/
│   ├── labels/              # Cartes de clusters
│   ├── metrics.csv
│   └── config.json
└── report/
    ├── rapport.md
    └── presentation.pdf
```

Cette arborescence est une cible de travail ; ce document ne suppose pas que ces fichiers existent déjà.

## 4. Étape 1 — Constituer le jeu d'images

### Actions

- [ ] Réunir environ **30 à 60 images** pour un premier jeu de données gérable.
- [ ] Inclure des pommes, des bananes, des poivrons et des scènes avec plusieurs produits.
- [ ] Varier les couleurs : rouge, vert, jaune et couleurs proches du fond.
- [ ] Prévoir des images simples sur fond uni et des scènes de marché plus complexes.
- [ ] Inclure des variations d'éclairage, des ombres et quelques reflets.
- [ ] Conserver les originaux et documenter les sources ainsi que les droits d'utilisation.
- [ ] Éviter les doublons et les images presque identiques.

Créer un fichier `metadata.csv` contenant au minimum :

```text
image_id,filename,source,fruit_types,background,lighting,difficulty,split
```

Réserver environ **un tiers des images à l'évaluation finale**. Utiliser les autres pour mettre au point le prétraitement, les paramètres et les règles d'interprétation. Garder les photos d'une même scène dans la même partition.

### Sources de données possibles

| Source | Utilité pour ce projet | Limite à prendre en compte |
| --- | --- | --- |
| [Fruits-360, dépôt des auteurs](https://github.com/Horea94/Fruit-Images-Dataset) | Démarrer avec des pommes, bananes et poivrons sur fond simple | Le fond des images individuelles a déjà été remplacé par du blanc ; ces images ne suffisent pas à démontrer une segmentation en situation réelle |
| Photos personnelles de marché ou de supermarché | Couvrir les fonds complexes, ombres, reflets et produits mélangés | Les masques doivent être annotés pour une évaluation quantitative |
| [MinneApple, article et données associés](https://arxiv.org/abs/1909.06441) | Ajouter un sous-ensemble de pommes en verger avec annotations humaines | Domaine différent du supermarché et limité aux pommes ; analyser ces résultats séparément |

Fruits-360 fournit des classes d'images, qui ne constituent pas des masques de segmentation de référence. Les vues successives d'un même fruit étant très proches, éviter de les répartir aléatoirement entre mise au point et évaluation. Pour MinneApple, choisir des images dont les annotations sont accessibles, via le [dépôt des auteurs](https://github.com/nicolaihaeni/MinneApple).

**Sortie attendue :** un ensemble d'images diversifié et un inventaire traçable.

**Critère de passage :** les images couvrent des cas faciles et difficiles, et la partition finale est fixée avant le choix de la meilleure configuration.

## 5. Étape 2 — Préparer les images et les références

### Prétraitement commun

- [ ] Lire chaque image en RGB et gérer correctement son orientation.
- [ ] Traiter explicitement les images en niveaux de gris ou avec transparence.
- [ ] Utiliser des images sRGB ; convertir un autre profil colorimétrique vers sRGB ou documenter l'hypothèse en l'absence de profil.
- [ ] Redimensionner en conservant les proportions, par exemple avec un côté maximal de **512 pixels** et sans agrandir les petites images.
- [ ] Après le redimensionnement PIL, convertir les valeurs RGB en nombres flottants dans `[0, 1]`, puis vérifier leur plage.
- [ ] Conserver exactement la même image préparée pour les trois espaces.
- [ ] Vérifier visuellement un échantillon avant de lancer les expériences.

Pour une image `uint8`, utiliser `img_as_float32` ou une division explicite par 255. Une simple conversion `astype(float)` conserve les valeurs de 0 à 255. À l'inverse, ne pas diviser de nouveau une image déjà dans `[0, 1]` ; `img_as_float32` ne remet pas automatiquement à l'échelle une entrée déjà flottante. Voir la [documentation des conversions de types](https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.img_as_float32).

Commencer **sans filtrage**. Un léger filtre pourra faire l'objet d'une expérience séparée si le bruit pose problème. Ne pas retirer manuellement le fond avant l'expérience principale : la séparation du fond fait partie du problème étudié.

### Annotations de référence

Annoter, par exemple, **10 à 15 images**, en couvrant les deux partitions et plusieurs niveaux de difficulté :

- Un masque binaire : pixels des fruits/produits d'intérêt contre fond.
- Facultativement, des masques par type de produit pour une extension de reconnaissance.

Pour un masque binaire, adopter une convention unique : `0 = fond`, `1 = produit d'intérêt`. Définir comment traiter les feuilles, les tiges, les étiquettes et les zones cachées. Redimensionner les masques avec une interpolation au plus proche voisin.

Prévoir au moins **5 images annotées dans la partition finale** pour ce premier prototype, sans les utiliser pour régler les paramètres. Avec un si petit effectif, présenter les scores comme exploratoires et montrer les résultats individuels ; augmenter les annotations si le temps le permet.

**Sortie attendue :** images préparées, masques alignés et conventions d'annotation documentées.

## 6. Étape 3 — Construire les représentations couleur

Chaque image de taille `H × W × 3` devient une matrice de caractéristiques comportant **une ligne par pixel**. Les coordonnées spatiales ne sont pas ajoutées dans l'expérience principale afin d'étudier l'effet de la couleur seule.

### RGB

Utiliser les canaux `R`, `G`, `B` dans `[0, 1]`.

**Hypothèse à vérifier :** représentation simple, mais sensible aux variations d'intensité et à l'éclairage.

### HSV

Convertir l'image RGB en HSV. Avec scikit-image, `H`, `S` et `V` sont dans `[0, 1]`.

La teinte est circulaire : `H ≈ 0` et `H ≈ 1` représentent des teintes proches. Le modèle HSV est décrit dans l'article de [Smith, *Color Gamut Transform Pairs*, 1978](https://www.icst.pku.edu.cn/mipl/docs/Color78.pdf). Pour éviter la discontinuité de teinte avec une distance euclidienne, le **choix d'encodage de ce projet** est :

```text
θ = 2πH
x1 = S × cos(θ)
x2 = S × sin(θ)
x3 = V
```

Cette représentation conserve trois caractéristiques : `x1` et `x2` dans `[-1, 1]`, `x3` dans `[0, 1]`. K-means accepte les valeurs négatives. La saturation réduit l'influence d'une teinte peu significative sur les pixels gris. Nommer cette variante **HSV circulaire** dans les résultats et expliquer le choix ; ce n'est pas une distance perceptuelle universelle ni une méthode prescrite par l'article.

Ne pas ramener uniquement `x1` et `x2` dans `[0, 1]` sans le signaler : les diviser par deux divise par quatre leur contribution aux distances au carré, relativement à `V`. Ajouter éventuellement le HSV direct `[H, S, V]` comme expérience secondaire pour montrer l'effet de la circularité ; cela ajoute **4 configurations par image** aux 12 principales.

**Hypothèse à vérifier :** une organisation par teinte et saturation peut aider à distinguer les couleurs des produits, tout en restant sensible aux ombres via `V`.

### Lab

Utiliser `rgb2lab` sur le sRGB flottant, avec `illuminant="D65"` et `observer="2"` explicitement enregistrés dans la configuration. Ces conventions sont précisées dans la [documentation de la conversion sRGB → Lab](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab).

Dans l'expérience principale, appliquer **le même facteur aux trois canaux** :

```text
x1 = L* / 100
x2 = a* / 100
x3 = b* / 100
```

Ce choix conserve les distances Lab à un facteur global près : `d = ΔE76 / 100`, puisque [ΔE76 est la distance euclidienne en Lab](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.deltaE_cie76). Il conserve donc l'objectif de K-means Lab, à un facteur multiplicatif près. Les composantes chromatiques peuvent rester négatives ; il n'est pas nécessaire d'obtenir trois canaux dans `[0, 1]`.

Une normalisation `L*/100`, `(a*+128)/255`, `(b*+128)/255` est une **variante pondérée**, à réserver à une expérience complémentaire : pour un même écart numérique, elle multiplie le poids relatif de la luminosité dans la distance au carré par environ `6,5`. Elle ne conserve pas la distance ΔE76. Ne pas confondre les valeurs Lab de scikit-image avec l'encodage Lab entier d'OpenCV.

**Hypothèse à vérifier :** la séparation entre luminosité et composantes chromatiques peut améliorer certaines segmentations. Le résultat dépend des images et de la pondération retenue.

### Règles de comparaison

- Fixer les transformations et les pondérations avant l'évaluation finale.
- Ne pas recalculer un min/max indépendamment pour chaque image.
- Ne pas imposer la même plage à tous les canaux : une pondération est un choix de métrique, pas une garantie de comparaison équitable. Ne pas ajouter de standardisation canal par canal dans l'expérience principale.
- Garder la même résolution, les mêmes pixels et les mêmes paramètres K-means.
- Présenter la comparaison comme celle de **trois représentations explicitement définies** : la normalisation et l'encodage HSV influencent aussi le résultat.
- Réserver toute suppression de luminosité, telle que Lab avec seulement `a*` et `b*`, à une expérience complémentaire.

**Sortie attendue :** trois matrices de caractéristiques par image et une configuration décrivant les conversions.

## 7. Étape 4 — Appliquer K-means

### Paramètres de départ

| Paramètre | Valeur proposée |
| --- | --- |
| Nombre de clusters | `3`, `4`, `5`, `6` |
| Initialisation | `k-means++` |
| Nombre d'initialisations | `n_init = 10` |
| Limite d'itérations | `max_iter = 300` |
| Tolérance de convergence | `tol = 1e-4` |
| Algorithme | `algorithm = "lloyd"` |
| Graine principale | `random_state = 42` |

Fixer explicitement `n_init = 10` : la valeur automatique dépend de l'initialisation et ne garantit pas dix essais. Enregistrer `n_iter_` et les éventuels avertissements. Voir la [documentation de KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

### Boucle expérimentale

```text
Pour chaque image :
    Préparer l'image une seule fois
    Pour chaque espace parmi RGB, HSV circulaire et Lab :
        Construire les caractéristiques des pixels
        Pour K parmi 3, 4, 5 et 6 :
            Ajuster K-means sur les pixels de cette image
            Replacer les étiquettes dans une carte H × W
            Enregistrer les labels, les mesures et les figures
```

Cela produit **12 configurations par image**. Ici, K-means est ajusté séparément sur chaque image : il ne s'agit pas d'un classificateur global entraîné sur l'ensemble du jeu de données.

Commencer avec tous les pixels à la résolution prévue. Si le temps de calcul devient trop long, ajuster K-means sur un sous-échantillon déterministe, identique pour les trois espaces, puis prédire les labels de tous les pixels. Documenter ce changement.

Vérifier que l'ensemble d'ajustement contient au moins K vecteurs de couleur distincts. Si moins de K clusters sont effectivement obtenus, enregistrer `n_clusters_effective` et signaler le cas ; ne pas le présenter comme une segmentation normale à K clusters.

**Point de contrôle :** K représente des groupes de couleur ; il ne correspond pas nécessairement au nombre de fruits ni au nombre de types de fruits présents.

## 8. Étape 5 — Visualiser et interpréter les clusters

Pour chaque configuration, produire :

- L'image originale préparée.
- Une carte de clusters en fausses couleurs, avec une légende.
- Une reconstruction où chaque cluster prend la **couleur RGB moyenne de ses pixels d'origine**.
- La proportion de pixels de chaque cluster et sa couleur moyenne.
- Si une règle de sélection est définie, le masque des produits d'intérêt.

La reconstruction par moyenne RGB permet de comparer les trois espaces sans devoir inverser leur encodage.

Les numéros des clusters sont arbitraires et peuvent changer entre deux exécutions. Ne pas supposer que le cluster `0` représente toujours le fond ou que le cluster `1` représente toujours une banane.

### Passage des clusters au masque des produits

Pour calculer une IoU ou un Dice, il faut transformer la carte de clusters en masque binaire. Définir une règle à partir des images de mise au point, par exemple à partir de la couleur moyenne, de la saturation et de la présence aux bords. Geler cette règle avant l'évaluation finale et documenter ses hypothèses, notamment pour les fonds complexes.

**Première règle automatique à implémenter sur fond simple :** définir une bande périphérique d'épaisseur `max(1, arrondi(0,02 × min(H, W)))` pixels, puis considérer comme fond le cluster ayant le plus de pixels dans cette bande. Les autres clusters forment le masque prédit. Fixer une règle de départage déterministe. Cette heuristique suppose que le fond domine les bords ; elle peut échouer si le fruit touche le bord ou si le fond occupe plusieurs clusters. La même règle s'applique aux trois espaces.

Si une règle plus élaborée utilise des seuils de couleur, les calculer à partir des pixels RGB d'origine des clusters, avec les mêmes conventions pour tous les espaces. Un éventuel nettoyage morphologique doit être évalué séparément, avec conservation des scores avant nettoyage.

Une union de clusters choisie manuellement reste possible pour une démonstration, mais doit être signalée comme **sélection manuelle**. Si les masques de référence servent à choisir la meilleure union de clusters, le score obtenu est une **borne optimiste**, pas une mesure de segmentation automatique.

**Sortie attendue :** planches comparatives lisibles et règle explicite d'interprétation des clusters.

## 9. Étape 6 — Évaluer les résultats

### Mesures proposées

| Mesure | Ce qu'elle évalue | Précaution |
| --- | --- | --- |
| Silhouette | Compacité et séparation des groupes dans leur représentation | Calculer sur un échantillon fixe ; les espaces ont des distances différentes |
| Inertie par pixel | Somme des distances quadratiques aux centres, divisée par le nombre de pixels évalués | Comparer les K d'une même représentation ; la somme brute dépend aussi de la taille de l'image |
| MSE RGB de reconstruction | Fidélité des couleurs reconstruites, évaluée dans le même RGB pour les trois espaces | Plus faible = meilleure fidélité ; ne mesure pas la reconnaissance des fruits |
| Temps de calcul | Coût de la segmentation | Même machine, même résolution, même protocole |
| IoU | Recouvrement entre masque prédit et masque annoté | Nécessite une règle explicite de sélection des clusters |
| Dice | Accord entre masque prédit et masque annoté | Utiliser les mêmes images et conventions pour tous les essais |
| ARI, en complément | Accord entre partitions, sans dépendre des numéros des clusters | Prévoir des annotations de régions adaptées ; distinguer les scores binaires et multiclasses |

```text
IoU  = |prédiction ∩ référence| / |prédiction ∪ référence|
Dice = 2 × |prédiction ∩ référence| / (|prédiction| + |référence|)
MSE_RGB = moyenne sur les pixels et canaux de (RGB_original - RGB_reconstruit)²
```

Pour les scores binaires, évaluer explicitement la classe **produit = 1**, avec `jaccard_score(..., average="binary", pos_label=1)` et `f1_score(..., average="binary", pos_label=1)` après aplatissement des masques. Le F1 binaire correspond ici au Dice. Ne pas mélanger ces scores avec une moyenne incluant le fond. Voir les documentations de [Jaccard/IoU](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) et du [F1/Dice](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Fixer les conventions : référence non vide et prédiction vide → `IoU = Dice = 0` ; référence vide et prédiction non vide → `0` ; les deux vides → valeur manquante dans la moyenne sur les fruits et décompte séparé. Ne pas remplacer une mesure invalide par zéro.

Pour la silhouette, utiliser par exemple jusqu'à **2 000 pixels par image**, avec les mêmes indices pour chaque configuration. Elle exige `2 ≤ nombre de labels présents ≤ nombre de pixels échantillonnés − 1` ; une silhouette calculée sans tous les clusters reste définie, mais doit être signalée. Voir la [documentation de silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

En cas d'ajustement sur un sous-échantillon, `inertia_` concerne ce sous-échantillon. Pour comparer les images complètes, recalculer la somme des distances au carré de **tous** les pixels à leurs centres prédits, puis la diviser par `H × W`. Avec des résolutions différentes, agréger les inerties par pixel plutôt que les sommes brutes.

La MSE utilise toujours la reconstruction par moyenne RGB décrite à l'étape 5 et le RGB dans `[0, 1]`. Elle apporte un critère commun sans annotation, mais favorise naturellement l'objectif RGB : l'utiliser comme diagnostic de quantification, pas comme arbitre unique de la segmentation.

Pour l'ARI, comparer des partitions clairement définies. L'ARI entre un masque binaire et K clusters peut être calculé, mais pénalise aussi la fragmentation d'un même fruit en plusieurs couleurs. Pour mesurer la stabilité entre graines, l'ARI entre deux cartes de clusters à K fixé évite tout appariement des numéros. Voir la [documentation de l'ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).

> Ne pas choisir le meilleur espace uniquement grâce à son inertie ou à sa silhouette. Ces scores décrivent la géométrie propre à chaque représentation. La qualité visuelle et les mesures sur des références communes sont nécessaires pour juger la segmentation.

### Tableau détaillé à exporter

```text
image_id,split,color_space,k,seed,n_pixels,n_fit_pixels,n_clusters_effective,n_iter,inertia_fit,inertia_per_pixel,mse_rgb,silhouette,conversion_time_s,kmeans_time_s,iou,dice,selection_method,notes
```

Laisser les mesures de segmentation vides lorsqu'aucun masque n'est disponible. Conserver les résultats par image avant toute agrégation.

### Tableau de synthèse attendu

| Espace | K | Silhouette moyenne | IoU moyenne | Dice moyen | Temps K-means moyen |
| --- | --- | --- | --- | --- | --- |
| RGB | 3 | À calculer | À calculer | À calculer | À calculer |
| RGB | 4 | À calculer | À calculer | À calculer | À calculer |
| … | … | … | … | … | … |
| HSV circulaire | 3 à 6, une ligne par K | À calculer | À calculer | À calculer | À calculer |
| Lab | 3 à 6, une ligne par K | À calculer | À calculer | À calculer | À calculer |

Calculer les moyennes **par image**, avec l'écart-type et le nombre d'images évaluées. Comparer les configurations sur les mêmes images et examiner séparément les fonds simples et les scènes complexes. Pour chaque mesure, indiquer le nombre de valeurs valides et les exclusions ; ne pas comparer silencieusement des moyennes calculées sur des images différentes. Ajouter l'inertie par pixel et la MSE RGB au tableau si elles sont utilisées dans l'analyse.

## 10. Étape 7 — Choisir une configuration et analyser les différences

### Procédure

1. Comparer les 12 configurations sur les images de mise au point.
2. Sur quelques images représentatives de mise au point, répéter avec les graines `0`, `1`, `2`, `3`, `4` pour observer la sensibilité à l'initialisation, en conservant `n_init = 10`. Comparer les cartes avec l'ARI et les scores de segmentation disponibles.
3. Choisir un couple espace/K selon la qualité de segmentation, le coût et la stabilité.
4. Fixer les paramètres et la règle de sélection des clusters.
5. Évaluer les configurations prévues sur les images réservées, sans réajuster les choix à leurs résultats.
6. Présenter les cas réussis et les cas d'échec, même si un espace semble meilleur en moyenne.

### Questions à traiter dans le rapport

- Un petit K fusionne-t-il le fruit avec le fond ?
- Un grand K découpe-t-il un seul fruit en zones d'ombre, de reflet et de couleur ?
- Les fruits de couleurs proches restent-ils confondus ?
- Les arrière-plans colorés perturbent-ils la sélection des produits ?
- Les différences d'éclairage modifient-elles les clusters ?
- Le meilleur résultat moyen reste-t-il satisfaisant sur les scènes difficiles ?
- La sélection des clusters nécessite-t-elle encore une intervention manuelle ?

Ne pas annoncer à l'avance que HSV ou Lab sera meilleur. Les conclusions doivent s'appuyer sur les résultats obtenus.

## 11. Étape 8 — Reconnaissance des fruits, en extension

Cette étape complète la segmentation si l'on veut rapprocher le prototype du contexte des caisses automatiques.

L'extension est disponible dans `src/region_recognition.py`, `src/recognition.py`, `src/train_recognition.py` et `src/recognize.py`. Elle combine des propositions K-means Lab, la séparation de certaines régions reliées, un SVM local et un rejet du fond appris sur les données du projet. Son inventaire distingue les 41 anciennes images d'apprentissage et les deux photos externes de diagnostic. Les partitions de la comparaison K-means sont conservées. La validation regroupe les régions par scène. Les photos de diagnostic participent au choix de la variante, sans ajuster ses poids : leur bilan n'est pas indépendant. Les limites et confusions résiduelles figurent dans le rapport.

1. Extraire les régions candidates depuis les masques.
2. Décrire chaque région par sa couleur dominante et, si nécessaire, par sa forme ou sa texture.
3. Définir une association aux types de produits à partir des seules images de mise au point.
4. Évaluer les prédictions sur les images réservées avec des annotations par produit.
5. Présenter une matrice de confusion et les confusions entre fruits de couleurs proches.

La simple association « jaune → banane » doit être présentée comme une heuristique limitée. Si plusieurs fruits se touchent, la séparation des objets constitue un problème supplémentaire.

## 12. Rapport et démonstration

### Structure du rapport

1. Contexte, problématique et objectifs.
2. Données, sources, partitions et annotations.
3. Principe de K-means et représentations couleur utilisées.
4. Prétraitement, normalisation et protocole expérimental.
5. Comparaisons visuelles pour les trois espaces et les quatre valeurs de K.
6. Tableau d'évaluation et graphiques des scores en fonction de K.
7. Analyse des différences, stabilité et cas d'échec.
8. Conclusion fondée sur les mesures et pistes d'amélioration.

### Démonstration minimale

- Choisir une image.
- Afficher l'original et les segmentations RGB, HSV circulaire et Lab pour le même K.
- Faire varier K de 3 à 6.
- Afficher les proportions des clusters et les mesures disponibles.
- Montrer un cas facile et un cas difficile.

Une présentation dans un notebook suffit ; une interface web n'est pas nécessaire au livrable principal.

## 13. Ordre de réalisation conseillé

| Phase | Travail | Résultat vérifiable |
| --- | --- | --- |
| 1 | Environnement, collecte et inventaire | Images lisibles et métadonnées complètes |
| 2 | Prétraitement et premières annotations | Images et masques alignés |
| 3 | K-means RGB sur une image, puis les trois espaces | Premières cartes de clusters correctes |
| 4 | Boucle sur tous les K et toutes les images | Résultats et configurations enregistrés |
| 5 | Évaluation et comparaison | CSV détaillé, synthèse et figures |
| 6 | Analyse, rapport et démonstration | Livrables reproductibles et conclusions argumentées |

## 14. Checklist de fin

- [ ] Les trois représentations RGB, HSV circulaire et Lab sont implémentées et documentées.
- [ ] Les valeurs de K de 3 à 6 sont toutes comparées.
- [ ] Les mêmes images et paramètres communs sont utilisés pour chaque espace.
- [ ] Les conversions couleur et les plages numériques sont vérifiées.
- [ ] La représentation HSV circulaire et la pondération Lab sont décrites ; les variantes complémentaires restent séparées.
- [ ] Les graines, versions des dépendances et paramètres sont enregistrés.
- [ ] Les cartes de clusters et les visualisations sont exportées.
- [ ] Le tableau d'évaluation contient les résultats par image et leur synthèse.
- [ ] Les inerties sont rapportées par pixel et les métriques invalides sont signalées.
- [ ] Les IoU/Dice indiquent clairement comment les clusters deviennent un masque.
- [ ] Les sélections manuelles ou optimistes sont identifiées.
- [ ] Les conclusions distinguent segmentation par couleur et reconnaissance du type de fruit.
- [ ] Le rapport explique les différences observées, les limites et les cas d'échec.
- [ ] Le README donne la procédure permettant de reproduire les expériences.

**Première action concrète :** collecter un petit lot de 6 à 10 images variées et faire fonctionner la chaîne complète sur une image, avant de lancer l'ensemble des expériences.

## 15. Références et lectures utiles

Sources consultées lors de la révision du **3 novembre 2026**. Les tailles de jeu de données, la résolution, les graines et les seuils proposés dans ce pipeline sont des choix pratiques pour le mini-projet ; ils ne sont pas imposés par ces sources.

| Référence | Apport au projet |
| --- | --- |
| [Smith, A. R. (1978), *Color Gamut Transform Pairs*, SIGGRAPH, p. 12–19](https://www.icst.pku.edu.cn/mipl/docs/Color78.pdf) | Fondements du modèle HSV ; l'encodage trigonométrique utilisé ici reste un choix explicite du protocole |
| [Hambali et al. (2016), *Intelligent segmentation of fruit images using an integrated thresholding and adaptive K-means method (TSNKM)*, Jurnal Teknologi, 78(6-5), p. 13–20](https://repo.uum.edu.my/id/eprint/20635/) | Exemple de segmentation de fruits combinant seuillage et K-means adaptatif ; intérêt de traiter les variations d'illumination |
| [Häni, Roy et Isler (2019/2020), *MinneApple: A Benchmark Dataset for Apple Detection and Segmentation*](https://arxiv.org/abs/1909.06441) | Images de verger annotées, utiles pour étudier les cas difficiles ; domaine à distinguer du supermarché |
| [Fruits-360, dépôt officiel des auteurs](https://github.com/Horea94/Fruit-Images-Dataset) | Données simples pour démarrer ; documentation de la capture et du fond blanc prétraité |
| [scikit-learn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | Paramètres, initialisations multiples et définition de l'inertie |
| [scikit-learn — Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) et [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) | Évaluation interne, conditions de validité et comparaison de partitions |
| [scikit-image — Conversions couleur et ΔE76](https://scikit-image.org/docs/stable/api/skimage.color.html) | Conventions sRGB/Lab/HSV et distances couleur |

**Utilisation dans le rapport :** résumer la méthode et les conditions de chaque article cité. L'article TSNKM emploie une méthode différente du K-means standard prévu ici ; ses conclusions ou scores ne peuvent pas être transférés directement à ce projet.
