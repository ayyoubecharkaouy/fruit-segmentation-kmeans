# Segmentation de fruits — K-means et espaces couleur

Mini-projet de comparaison des segmentations en RGB, HSV circulaire et Lab, pour K de 3 à 6.

- [Pipeline complet du projet](PIPELINE_PROJET.md)
- [Préparation du jeu d'images](data/README.md)
- [Plan de sélection actif](data/selection_plan.json)
- [État réel des téléchargements](data/preparation_status.json)
- [Organisation et noms des images](data/ORGANISATION.md)

## Images organisées

Le jeu actif contient **51 images**, réparties en **35 images de mise au point et 16 d'évaluation**. Les trois ajouts Kaggle sont rangés dans `data/raw/ajouts_manuels/`, avec des identifiants uniques et leurs références dans l'inventaire. Leurs licences restent à confirmer. Les 48 images Fruits-360 conservent leurs identifiants et leurs sources.

## Préparer les images

Depuis la racine du projet, dans un terminal avec accès à Internet :

```bash
python -m pip install -r requirements-data.txt
python scripts/prepare_data.py
```

Le jeu initial contient 30 images individuelles et 15 photos du dépôt Fruits-360 multi. L'inventaire actif est disponible dans `data/metadata.csv`, les effectifs dans `data/preparation_status.json` et la planche d'aperçu dans `data/apercu.jpg`.

## Équilibrer les photos et ajouter les mélanges

Le [plan équilibré](data/selection_plan_balanced.json) prévoit 5 photos de pommes, 5 de bananes, 5 de poivrons et 3 scènes mixtes, en complément des 30 images individuelles. Pour appliquer cette mise à jour depuis un terminal connecté :

```bash
python scripts/rebalance_data.py
```

Le script conserve les images réutilisées, télécharge les 8 images supplémentaires et archive les 5 photos de pommes retirées. Le jeu actif passe à **48 images** seulement après vérification complète. Le bilan de cette opération est enregistré dans `data/rebalance_status.json`. Voir la [procédure de rééquilibrage](data/REBALANCEMENT.md).

Les sources, paramètres, partitions et limites de cette sélection sont décrits dans [data/README.md](data/README.md). Les masques de référence et les expériences K-means seront préparés aux étapes suivantes.
