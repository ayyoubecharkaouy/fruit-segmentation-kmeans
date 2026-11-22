# Segmentation et reconnaissance de fruits

Le projet compare K-means en **RGB, HSV circulaire et Lab**, puis reconnaît des **pommes, bananes et poivrons** à partir de couleur, forme et texture. Les piments sont associés à `poivron`.

Les classifieurs sont entraînés sur les données locales du projet. Aucun outil supplémentaire, modèle préentraîné ou service distant n'est nécessaire.

- [Rapport et résultats](RAPPORT_PROJET.md)
- [Pipeline de l'étude K-means](PIPELINE_PROJET.md)
- [Données de reconnaissance](data/recognition/README.md)
- [Méthode de reconnaissance révisée](results/recognition/region_correction/README.md)

## Reconnaître les produits d'une image

Depuis la racine du projet, avec l'environnement virtuel existant activé :

```bash
python src/recognize.py data/test/test_img001.png
python src/recognize.py data/test/test_img002.png
```

Pour une autre photo :

```bash
python src/recognize.py chemin/photo.jpg --output-dir results/recognition/predictions
```

Le programme utilise la photo et `models/fruit_classifier.joblib`. Il ne consulte ni ses catégories de référence ni les annotations ni le nom du fichier pour prédire.

Sur fond uniforme, il conserve la localisation par composantes connexes. En scène complexe, des groupes K-means Lab et leurs teintes voisines proposent des régions ; un watershed sépare certaines composantes reliées. Un descripteur décrit la forme, les couleurs et la texture intérieure à taille normalisée. Le SVM reconnaît les trois catégories. Une forêt apprise sur des exemples de fond rejette des propositions dans les scènes complexes. Les boîtes redondantes sont supprimées.

Trois fichiers sont enregistrés : l'image avec boîtes et catégories, un masque des régions retenues et le JSON contenant coordonnées, scores, propositions rejetées et empreinte du modèle. Les coordonnées concernent la photo orientée à sa résolution originale. Les scores n'ont pas été validés comme probabilités de réussite sur de nouvelles scènes.

**Limites :** la reconnaissance reste exploratoire. Des pommes et poivrons peuvent être confondus ; des produits inconnus ou du décor peuvent recevoir une catégorie cible. Les objets partiellement cachés et ceux de même couleur qui se touchent restent difficiles. La présence des trois catégories ne garantit pas la justesse de chaque boîte.

## Reproduire l'apprentissage et les essais

```bash
python scripts/prepare_recognition_dataset.py
python src/train_recognition.py
python scripts/demo_recognition.py
python src/evaluate_recognition.py
```

L'inventaire `data/recognition/dataset.csv` contient **41 images d'apprentissage et 2 photos externes de diagnostic**. Parmi les images d'apprentissage, 36 fournissent 100 régions positives. Sept images possèdent des zones de fond annotées, dont cinq nouvelles sources négatives. Les coins des images simples complètent les exemples de fond. Le fichier `data/metadata.csv` et les résultats de segmentation restent inchangés.

Le prototype utilise le SVM avec `C=1`. Cette variante a été retenue après les diagnostics sur les deux photos externes. Leurs pixels ne servent pas à entraîner les classifieurs, mais ces photos ont participé aux décisions de développement : elles ne constituent donc pas un test indépendant. `--classifier auto` permet de retenir uniquement le meilleur F1 de validation entre scènes, qui peut fournir une autre variante.

Les nouveaux tableaux figurent dans `results/recognition/region_correction/`, les deux images annotées dans `results/recognition/user_tests/`. Le modèle et les sorties du prototype initial sont conservés dans `results/recognition/before_region_correction/`.

## Étude comparative K-means

Le jeu initial contient **51 images** : 35 pour la mise au point et 16 pour l'évaluation. Quinze images disposent de masques manuels. Les trois représentations et K = 3 à 6 donnent **612 configurations**.

```bash
python src/preprocessing.py
python src/create_masks.py
python src/segmentation.py
python src/evaluation.py
python src/visualization.py
```

`evaluation.py` et `visualization.py` réutilisent les cartes existantes sans relancer K-means. Les sorties restent dans `results/labels`, `results/tables` et `results/figures`. Les scores IoU/Dice évaluent le masque fruit/fond ; ils ne mesurent pas la reconnaissance des catégories.

Les six figures de synthèse du rapport sont conservées. Les 1 224 figures de détail et les CSV par image ont été supprimés lors du nettoyage : les 612 configurations restent disponibles dans `global_results.csv` et leurs cartes `.npy`. Ces exports de détail peuvent être régénérés par `segmentation.py`.

Les cartes `.npy`, l'environnement Python, les caches, les sauvegardes `data/history/` et les prédictions libres sont exclus de Git. Après un clonage, exécuter `python src/segmentation.py` pour générer les cartes avant de relancer `evaluation.py` ou `visualization.py`. Les tableaux de synthèse, les figures du rapport, les données du projet et les modèles locaux sont conservés dans le dépôt.

## Environnement et vérifications

Les bibliothèques sont celles du projet : Pillow, NumPy, pandas, scikit-image, scikit-learn, Matplotlib, SciPy et joblib. Labelme sert à l'annotation initiale. Les versions du modèle sont enregistrées dans son JSON ; la borne scikit-learn de `requirements.txt` conserve l'API probabiliste du SVM utilisée par ce prototype.

```bash
python -m unittest discover -s tests -v
```

Les tests couvrent l'exclusion des photos de diagnostic de l'apprentissage, la séparation des scènes, le rejet du fond, les propositions multiples, l'alignement des formes inclinées, l'extraction de texture et les coordonnées des boîtes.

## Sources et organisation des données

Les 48 images Fruits-360 et trois ajouts Kaggle du jeu initial sont documentés dans l'inventaire. Les licences non renseignées et changements de domaine sont signalés dans le rapport.

- [Préparation initiale](data/README.md)
- [Organisation des images](data/ORGANISATION.md)
- [Inventaire de segmentation](data/metadata.csv)
- [Inventaire de reconnaissance](data/recognition/dataset.csv)

Les scripts historiques de téléchargement ne sont pas nécessaires pour utiliser les données et modèles locaux existants.
