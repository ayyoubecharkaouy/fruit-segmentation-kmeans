# Données de reconnaissance

## Deux protocoles distincts

L'étude K-means utilise les 51 images de `data/metadata.csv`, avec sa partition mise au point / évaluation. Ce fichier et ses résultats sont conservés.

La reconnaissance utilise `dataset.csv` :

- `recognition_split = train` : 41 anciennes images, dont 36 fournissent les 100 régions positives ; cinq nouvelles sources de fond complètent l'apprentissage.
- `recognition_split = test` : les deux images de diagnostic de `data/test`, exclues de l'ajustement des poids mais impliquées dans le choix de la variante.
- `split` : partition historique de segmentation, conservée pour la traçabilité.

Certaines anciennes images d'évaluation de segmentation servent donc à l'apprentissage de la **nouvelle expérience de reconnaissance**. Leur ancien statut ne constitue plus un test de reconnaissance. Les images externes sont séparées par chemin, identifiant, scène et empreinte de fichier. Les quasi-doublons visuels ne sont pas détectés automatiquement.

## Catégories des régions

Les images simples possèdent une catégorie dans les métadonnées. Leur plus grande région fournit un exemple. Les polygones des scènes à une seule catégorie héritent de cette catégorie. Les piments sont associés explicitement à `poivron`.

Le fichier `instance_classes.json` décrit séparément les catégories des 32 polygones de `img051`, attribuées par inspection visuelle. Ces polygones délimitent des pommes ; les bananes visibles sous la caisse ne sont pas délimitées par ces polygones. Une empreinte SHA-256 lie cette attribution à la version du JSON. Les annotations binaires initiales ne sont pas modifiées.

| Catégorie | Images sources | Régions |
| --- | --- | --- |
| banane | 11 | 17 |
| poivron | 13 | 40 |
| pomme | 12 | 43 |

Les régions d'une même image reçoivent ensemble le poids d'une image. Les classes sont ensuite équilibrées. La validation exclut à chaque tour toutes les régions d'un même `scene_id`, afin de conserver ensemble les vues et les polygones d'une scène.

## Références externes

Les catégories présentes dans les deux nouvelles images ont été indiquées lors de leur ajout. `data/test/annotations.json` contient leurs empreintes et, pour `test_img001.png`, neuf boîtes approximatives établies par inspection visuelle : trois pommes, trois bananes et trois poivrons. Ces boîtes servent exclusivement au bilan de test.

L'image complexe ne possède pas de référence exhaustive par objet. Son évaluation de présence des catégories est donc complétée par l'analyse visuelle des erreurs. Les deux images ont servi au diagnostic du développement ; elles ne démontrent pas une généralisation indépendante sur des scènes réelles quelconques. Leur origine photographique ou synthétique et leurs conditions de réutilisation ne sont pas documentées.

Les fichiers de référence sont lus par `evaluate_recognition.py`, jamais par `predict_photo()`.

## Reproduction

```bash
python scripts/prepare_recognition_dataset.py
python src/train_recognition.py
python scripts/demo_recognition.py
python src/evaluate_recognition.py
```

L'exécution de la reconnaissance seule utilise le modèle sauvegardé et une image, sans accès réseau ni annotation.

## Révision de la reconnaissance par régions

Le protocole actif ajoute cinq anciennes images comme sources de fond : `img033`, `img035`, `img047`, `img049` et `img050`. L'inventaire contient donc 41 images d'apprentissage ; les 100 régions positives proviennent toujours de 36 images. Les anciens effectifs de 36 images décrits plus haut correspondent au premier prototype.

`background_regions.json` délimite des rectangles de feuillage, sol, paroi et fond sombre dans sept images. Les coordonnées ont été établies par inspection visuelle des seules images d'apprentissage. Chaque attribution est liée à l'image par SHA-256. Les autres pixels des scènes ne sont pas déclarés automatiquement négatifs, car des fruits visibles peuvent ne pas être annotés. Les coins des images simples sont utilisés après exclusion du masque du fruit et de son voisinage.

La version révisée corrige le signe de l'alignement et ajoute 118 caractéristiques de contour et de texture normalisée aux 615 caractéristiques historiques. La classification utilise les régions originales. Les variations d'éclairage, saturation, netteté et contour servent au rejet fruit/fond. Toutes les variantes restent avec leur scène pendant la validation.

Les deux photos externes ont été utilisées pour diagnostiquer les erreurs et retenir la variante SVM `C=1`. Elles sont exclues de l'ajustement des poids, mais ont participé au choix de la méthode. Leur bilan est une démonstration de développement. Le meilleur F1 de validation entre scènes n'est pas nécessairement celui de la variante retenue pour ces diagnostics.

Les détails et tableaux actifs sont dans `results/recognition/region_correction/`. Le premier prototype et son inventaire sont archivés dans `results/recognition/before_region_correction/`.
