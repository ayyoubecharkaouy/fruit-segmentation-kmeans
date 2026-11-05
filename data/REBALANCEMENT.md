# Rééquilibrer le jeu d'images

## Appliquer la mise à jour

Depuis la racine du projet, dans le même environnement virtuel :

```bash
python scripts/rebalance_data.py
```

Le script nécessite Internet. Les images déjà téléchargées et conservées sont réutilisées. La commande peut être relancée après une interruption. Le bilan réel est enregistré dans `rebalance_status.json` ; le plan seul ne signifie pas que les nouvelles images sont présentes.

Pour afficher la cible sans téléchargement :

```bash
python scripts/rebalance_data.py --plan
```

## Cible : 48 images

| Groupe | Nombre | Mise au point | Évaluation |
| --- | ---: | ---: | ---: |
| Pommes individuelles | 10 | 6 | 4 |
| Bananes individuelles | 10 | 8 | 2 |
| Poivrons individuels | 10 | 6 | 4 |
| Photos de pommes | 5 | 5 | 0 |
| Photos de bananes | 5 | 0 | 5 |
| Photos de poivrons | 5 | 5 | 0 |
| Scènes mixtes | 3 | 3 | 0 |
| **Total** | **48** | **33** | **15** |

Les 18 photos du dépôt multi se trouveront toutes dans `raw/fruits360_multi/` après activation. Les trois groupes de photos consacrés à un produit sont équilibrés à cinq images chacun ; les scènes mixtes constituent un groupe complémentaire.

Les fichiers mixtes, repérés dans le [répertoire officiel](https://github.com/fruits-360/fruits-360-multi/tree/main/test-multiple_fruits), sont :

- `apple_banana.jpg` : pommes et bananes attendues.
- `apple_banana_lemon_1.jpg` : pommes, bananes et citron attendus.
- `apple_banana_pomegranate_pepper.jpg` : pommes, bananes, grenade et poivron attendus.

La dernière image doit donc couvrir les trois produits cibles dans la même scène. Les catégories attendues viennent des noms sources : confirmer visuellement le contenu après téléchargement. Il ne s'agit pas de trois vues garanties du trio complet. Les autres produits visibles, comme le citron et la grenade, sont conservés dans `fruit_types` pour préparer des annotations cohérentes.

Les cinq photos de poivrons sont sélectionnées dans le même dépôt avec un filtre sur les noms commençant par `pepper` ou `peppers`. Leurs noms et empreintes exacts seront consignés dans `rebalance_manifest.json`. Si cinq fichiers ne sont pas disponibles, le script s'arrête avec le nom du groupe concerné au lieu de substituer des photos d'un autre produit.

## Conservation des données

1. La sélection utilise les mêmes révisions Git que le jeu initial pour réutiliser les 30 images individuelles et 10 photos conservées.
2. Les 5 photos de poivrons et les 3 scènes mixtes sont téléchargées et vérifiées dans `staging/rebalance/`.
3. Tant qu'une image manque ou échoue aux contrôles, les 45 images du jeu actif et son inventaire restent inchangés.
4. Lorsque les 48 images sont valides et sans doublon exact, les cinq photos de pommes excédentaires sont déplacées dans `history/<date>/raw/fruits360_multi/`.
5. Le manifeste, le plan actif, l'inventaire et la planche d'aperçu sont mis à jour. Leurs versions précédentes sont conservées dans le même dossier d'historique.

Les identifiants des images conservées restent stables ; les nouvelles images reçoivent de nouveaux identifiants. Ne pas déduire le nombre d'images de l'identifiant maximal.

## Limites à documenter

Les photos d'une même famille restent dans une seule partition, notamment les bananes Lady Finger. Les scènes mixtes sont groupées dans la mise au point pour limiter le risque de répartir des prises de vue liées entre les deux partitions. Les photos complexes ne sont donc pas équilibrées par produit dans l'évaluation finale ; rapporter les résultats par groupe et compléter ultérieurement avec de nouvelles scènes indépendantes.

Le rééquilibrage ne crée pas de masques de référence et ne produit aucune image synthétique. Les fichiers originaux conservent leur attribution Fruits-360 et les licences déjà téléchargées.
