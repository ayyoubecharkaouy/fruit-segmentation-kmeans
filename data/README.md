# Jeu d'images — préparation

## État actuel

**51 images sont inventoriées : 35 de mise au point et 16 d'évaluation.** Le jeu comprend les 48 images Fruits-360 obtenues après rééquilibrage et 3 ajouts Kaggle. Les effectifs actifs sont enregistrés dans `preparation_status.json` et les images correspondantes dans `metadata.csv`. La [fiche d'organisation](ORGANISATION.md) décrit les noms corrigés et les ajouts.

Une première inspection du jeu initial montre 30 images individuelles sur fond blanc, une photo d'étal de pommes (`img031`), neuf photos de pommiers (`img032` à `img040`) et cinq photos de bananes sur fond sombre (`img041` à `img045`). Pour corriger cette surreprésentation des pommes, une [mise à jour vers 48 images](REBALANCEMENT.md) est disponible : 5 photos par produit et 3 scènes mixtes, en plus des 30 images individuelles. Elle s'applique avec `python scripts/rebalance_data.py`, depuis la racine du projet. `rebalance_status.json` indique si elle a effectivement réussi.

L'inspection d'une planche ne remplace pas la revue détaillée des originaux ni les annotations de pixels ; `review_status` reste à vérifier.

## Télécharger les données

Depuis la racine du projet :

```bash
python -m pip install -r requirements-data.txt
python scripts/prepare_data.py
```

Si Pillow est déjà installé, seule la deuxième commande est nécessaire. Aucun compte Kaggle ou jeton GitHub n'est requis. Le script télécharge uniquement les fichiers sélectionnés, sans cloner l'intégralité des jeux de données.

Commandes sans téléchargement :

```bash
python scripts/prepare_data.py --plan
python scripts/prepare_data.py --check
```

La première affiche les effectifs prévus ; la deuxième vérifie les images déjà disponibles. Un code de sortie non nul indique une sélection non résolue, des fichiers manquants, invalides ou des doublons exacts. Une interruption peut être suivie d'une relance : les images présentes sont vérifiées et réutilisées.

## Sélection initiale : 45 images, avant rééquilibrage

| Contenu | Mise au point | Évaluation | Total |
| --- | ---: | ---: | ---: |
| Pommes individuelles | 6 | 4 | 10 |
| Bananes individuelles | 8 | 2 | 10 |
| Poivrons individuels | 6 | 4 | 10 |
| Photos de pommes du dépôt multi | 10 | 0 | 10 |
| Photos de bananes du dépôt multi | 0 | 5 | 5 |
| **Total** | **30** | **15** | **45** |

Les sources sont les dépôts officiels [Fruits-360 100×100](https://github.com/fruits-360/fruits-360-100x100) et [Fruits-360 multi](https://github.com/fruits-360/fruits-360-multi). Les images individuelles ont un fond blanc prétraité. Les photos multi sont des photographies fournies par l'auteur, sans garantie qu'elles correspondent à des étals de supermarché.

Cette sélection constitue un **jeu de démarrage**, avec plusieurs variétés et couleurs, mais pas un échantillon représentatif du supermarché. Les photos personnelles proposées dans le pipeline ne sont pas remplacées par des images artificielles. Compléter ensuite avec des scènes où pommes, bananes et poivrons sont mélangés, ainsi que des fonds complexes.

## Fichiers créés par le téléchargement

```text
data/
├── raw/
│   ├── fruits360_simple/       # 30 images originales
│   ├── fruits360_multi/        # 18 photographies originales
│   ├── ajouts_manuels/
│   │   ├── simple/            # 2 piments issus de Kaggle
│   │   └── multi/             # 1 photo de bananes issue de Kaggle
│   └── photos_personnelles/    # Réservé aux compléments futurs
├── licenses/                  # Licences des versions téléchargées
├── selection_plan.json        # Groupes, effectifs, graine et partitions prévus
├── download_manifest.json     # Noms exacts, révisions Git et empreintes sources
├── metadata.csv               # Inventaire des images présentes et valides
├── preparation_status.json    # Effectifs, fichiers manquants et erreurs
├── apercu.jpg                 # Planche des images réellement téléchargées
└── README.md
```

La première exécution résout et enregistre les révisions Git des deux dépôts. Les exécutions suivantes réutilisent ce manifeste. Pour chaque fichier, le script vérifie l'empreinte Git du contenu, l'ouverture de l'image et son mode RGB, puis enregistre dimensions et SHA-256. Les originaux ne sont ni redimensionnés ni recolorés.

Le script utilise les références et les arbres Git non récursifs pour limiter la taille des réponses de l'API. Il lit les réponses par blocs, vérifie leur longueur annoncée lorsqu'elle est fournie et réessaie jusqu'à trois fois si un transfert est interrompu ou si le JSON reçu est invalide. En cas d'échec persistant, le message précise l'URL concernée. Une erreur de résolution ne nécessite pas de supprimer `preparation_status.json` : il suffit de relancer le programme après correction du problème.

Les images individuelles sont sélectionnées de manière déterministe dans les dossiers indiqués, avec la graine 42. Les photos multi sont définies par leurs noms ou par un filtre explicite dans le plan actif. Si les sources ont changé ou si un fichier manque, le script signale l'erreur ; aucune image de substitution n'est inventée.

Les ajouts Kaggle sont déclarés dans `manual_additions.json` et inclus comme `local_items` dans les plans actifs. Ils sont contrôlés par leur SHA-256 local ; ils ne reçoivent pas d'empreinte Git ou de licence Fruits-360. Si l'un de ces fichiers manque, le programme demande sa restauration plutôt que de traiter l'URL du jeu de données comme celle d'une image.

## Vérification visuelle avant les expériences

Après téléchargement :

1. Ouvrir `apercu.jpg`, puis examiner les originaux si nécessaire.
2. Confirmer les produits visibles ; `fruit_types` vient initialement des dossiers et noms sources.
3. Renseigner `background`, `lighting` et `difficulty` lorsqu'ils valent `a_verifier`. Le niveau `facile_attendue` des images individuelles est une hypothèse à confirmer.
4. Repérer les images trop semblables et vérifier que les mêmes scènes ou objets ne se retrouvent pas dans les deux partitions.
5. Enregistrer les corrections durables dans `download_manifest.json`, puis relancer `--check` pour régénérer l'inventaire et l'aperçu.

`scene_id` désigne ici un **groupe conservateur de prises de vue supposées liées**, pas un identifiant de scène certifié par l'auteur. Les images d'une même variété sont regroupées dans une seule partition. Les bananes Lady Finger des deux dépôts partagent aussi le même groupe. Les familles des photos multi restent dans une seule partition. Cela réduit les fuites évidentes, mais une revue visuelle reste nécessaire.

Les variétés retenues pour l'évaluation peuvent différer de celles de mise au point : ce choix teste aussi le transfert entre variétés et couleurs. Le jeu reste petit et les scènes complexes ne sont pas équilibrées entre tous les produits ; rapporter les résultats par image et par source.

Les masques de segmentation ne sont pas fournis par cette préparation. Les créer à l'étape suivante sur un sous-ensemble de 10 à 15 images, dont au moins 5 d'évaluation. Ne pas considérer les catégories d'images comme des annotations de pixels.

## Attribution

Sources : **Mihai Oltean, Fruits-360 dataset, 2017–**, dépôts officiels indiqués ci-dessus. Les dépôts consultés déclarent la licence **CC BY-SA 4.0** ; le téléchargement conserve leurs fichiers `LICENSE` à la révision choisie. Conserver l'attribution, le lien source et la licence avec les données utilisées dans le rapport ou redistribuées.

Les aperçus sont des adaptations de présentation ; les fichiers de `raw/` conservent les octets originaux des sources.
