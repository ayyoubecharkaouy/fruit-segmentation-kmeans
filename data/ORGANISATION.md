# Organisation des images et contrôle des noms

## État du jeu organisé

Le jeu contient **51 images actives** : 35 de mise au point et 16 d'évaluation. Les 48 images déjà inventoriées conservent leurs identifiants et leurs octets. Les trois ajouts ont été enregistrés et rangés selon leur provenance et leur type de scène.

```text
raw/
├── fruits360_simple/         # 30 images individuelles de Fruits-360
├── fruits360_multi/          # 18 photos de Fruits-360
├── ajouts_manuels/
│   ├── simple/               # 2 piments sur fond uni clair
│   └── multi/                # 1 photo de bananes en extérieur
└── photos_personnelles/      # Réservé aux photos personnelles futures
```

Les ajouts ne sont pas rangés parmi les sources Fruits-360 : leur provenance Kaggle est conservée séparément.

## Corrections des noms

| Chemin avant organisation, sous `data/` | Chemin actuel, sous `data/` | Contenu vérifié visuellement |
| --- | --- | --- |
| `raw/fruits360_multi/img054.png` | `raw/ajouts_manuels/multi/img054.png` | Bananes en extérieur |
| `raw/fruits360_simple/img031.png` | `raw/ajouts_manuels/simple/img055.png` | Piment vert allongé sur fond clair |
| `raw/fruits360_simple/img032.jpg` | `raw/ajouts_manuels/simple/img056.jpg` | Piment rouge allongé sur fond clair |

Les identifiants `img031` et `img032` étaient déjà utilisés par des photos de pommes du dossier multi. Ils ne sont plus réutilisés pour les deux piments. Chaque image active possède désormais un identifiant global unique, au format `imgNNN`, avec l'extension correspondant à son format. Les fichiers PNG restent des PNG ; les JPEG ne sont pas réencodés.

Les numéros 036 à 040 correspondent à des images archivées lors du rééquilibrage. Les trous dans la numérotation sont conservés pour garder la traçabilité. Le nombre d'images se lit dans l'inventaire, pas à partir du numéro maximal.

## Sources des ajouts

| Images | Jeu déclaré par l'utilisateur | Référence |
| --- | --- | --- |
| `img054` | Agricultural crops image classification | [Kaggle — Agricultural crops image classification](https://www.kaggle.com/datasets/mdwaqarazam/agricultural-crops-image-classification) |
| `img055`, `img056` | Chili Plant Disease Detection | [Kaggle — Chili Plant Disease Detection](https://www.kaggle.com/datasets/shuvokumarbasak4004/chili-plant-disease-detection) |

Ces provenances sont déclarées par l'utilisateur. Le nom exact des fichiers à l'intérieur des jeux Kaggle et leurs licences ne sont pas encore vérifiés. L'inventaire conserve donc `license = a_verifier` et un statut de provenance explicite, sans réutiliser la licence CC BY-SA de Fruits-360. `original_filename` conserve le nom local au moment de l'import ; `source_file` reste vide lorsque le nom dans le jeu source n'a pas été communiqué.

## Partitions et catégories

Les partitions des 48 images existantes sont conservées. Les deux piments, issus du même jeu avec des conditions visuelles proches, sont regroupés dans la mise au point ; la nouvelle photo de bananes est affectée à l'évaluation. Cette affectation constitue une convention de groupe conservatrice, sans prétendre connaître la séance de prise de vue d'origine.

Conformément au périmètre choisi, les formes larges et allongées sont conservées. `fruit_types` garde les descriptions `poivron` et `piment` ; pour une analyse commune, les regrouper sous **Peppers (poivrons et piments)**. K-means utilise les caractéristiques couleur et n'attribue pas directement ces catégories aux pixels.

## Vérifier et reprendre l'organisation

```bash
python scripts/prepare_data.py --check
```

Ce contrôle ne nécessite pas Internet. Il vérifie les fichiers référencés, signale les images non inventoriées et régénère le CSV ainsi que l'aperçu. Les ajouts sont contrôlés à partir de leurs empreintes SHA-256 enregistrées.

```bash
python scripts/organize_data.py
```

Cette deuxième commande rejoue les déclarations de `manual_additions.json` et peut être relancée sans créer de nouveaux identifiants. Les références permanentes doivent être modifiées dans ce fichier pour les ajouts, puis réappliquées. `metadata.csv` est un export régénéré.

Les anciens inventaires, plans, aperçus et correspondances des noms sont sauvegardés dans `history/<date>_organisation/`. Le rapport `organisation_status.json` indique les déplacements réellement effectués.
