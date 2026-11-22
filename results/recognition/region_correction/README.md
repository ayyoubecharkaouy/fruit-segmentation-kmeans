# Reconnaissance par régions et rejet du fond

Cette expérience utilise les bibliothèques existantes, un SVM local et une forêt de rejet apprise sur les données du projet. Aucun modèle préentraîné ou service distant n'est utilisé.

## Localisation

La dispersion Lab du contour détermine si l'hypothèse d'un fond uniforme est compatible avec l'image. Dans ce cas, les composantes de la localisation initiale sont conservées et le rejet appris du fond n'est pas appliqué.

Sur un contour hétérogène, K-means regroupe les couleurs Lab avec K = 6 et un poids de 0,25 pour L*. Les centres sont ajustés sur au plus 20 000 pixels tirés avec la graine 42, puis tous les pixels sont affectés. Les composantes suffisamment saturées et les unions de teintes voisines fournissent des propositions. Fermeture et remplissage des trous limitent les coupures dues aux ombres et reflets. Un watershed de la transformée de distance sépare des composantes irrégulières présentant plusieurs noyaux. Les petites régions et celles occupant plus de 30 % de l'image sont exclues dans ce mode.

Les centres K-means sont recalculés à l'inférence, sans annotation ni catégorie de référence. Les expériences comparatives de segmentation conservent leur protocole et leurs résultats initiaux.

## Descripteur et classification

Le signe de la rotation d'alignement a été corrigé. La compatibilité des anciens descripteurs est conservée pour les modèles historiques. Le descripteur révisé contient 733 valeurs : les 615 historiques et 118 supplémentaires de texture et contour. Les ajouts comprennent LBP à deux échelles, variations locales Lab, gradients intérieurs, spectre du contour et histogramme conjoint teinte/saturation. La texture est mesurée après mise à taille comparable du produit et exclusion de sa bordure lorsque sa surface le permet.

Le SVM utilise 189 caractéristiques : géométrie, couleurs, texture native et nouveaux descripteurs locaux. Les variables sont standardisées dans chaque apprentissage. Le noyau est RBF ; le prototype actif utilise C = 1. Les régions originales sont utilisées pour cette classification.

La validation exclut toutes les régions d'un même `scene_id`. La sélection compare la forêt initiale et des SVM C = 0,3, 1 et 3, avec la même règle d'affectation par score maximal qu'à l'inférence. C = 0,3 obtient le meilleur F1 entre scènes, mais confond les pommes de la photo externe à fond simple. C = 1 a été retenu pour le prototype après ces diagnostics. Les deux photos sont donc des exemples de développement, exclus des poids d'apprentissage mais impliqués dans le choix de la variante. Elles ne constituent pas un test indépendant.

## Rejet et doublons

Une forêt distingue fruit/fond à partir de 58 caractéristiques couleur/LBP. Ses exemples négatifs proviennent de rectangles inspectés dans sept anciennes images, ainsi que des coins des images simples après exclusion du fruit et de son voisinage. Cinq images constituent de nouvelles sources négatives : le protocole compte 41 images d'apprentissage, dont 36 fournissent les 100 régions positives.

Les variations d'éclairage, saturation, netteté et contour servent à cet apprentissage du rejet. Toutes les variantes restent dans leur scène pendant la validation. Le seuil est choisi en maximisant la moyenne du rappel des fruits et de la spécificité du fond sur les prédictions tenues à l'écart. Ce choix peut rejeter des fruits. Il n'est appliqué qu'aux propositions de scènes complexes.

Les propositions dont le meilleur score de catégorie est inférieur à 0,45 sont écartées. Une suppression des boîtes redondantes compare toutes les catégories et les inclusions. Les scores du SVM utilisent son estimation probabiliste interne ; ils ne représentent pas une fiabilité validée sur des scènes inconnues.

## Fichiers et limites

`model_selection.csv`, `training_samples.csv`, `cross_validation_predictions.csv` décrivent la classification. `background_training_samples.csv`, `background_cross_validation.csv` et `background_validation.csv` décrivent le rejet. `evaluation_summary.json`, les tableaux de détection et la matrice de confusion concernent les deux photos externes. Les images annotées actives sont dans `../user_tests/` ; les anciennes sorties sont dans `../before_region_correction/`.

Les boîtes de l'image simple sont approximatives. L'image complexe ne possède pas de référence exhaustive par objet. La présence correcte des catégories ne valide ni chaque boîte ni le nombre d'objets. Le rejet appris sur du feuillage, des parois et des fonds simples ne couvre pas tous les produits inconnus ; orange, oignon ou tomate peuvent rester confondus avec une catégorie cible.
