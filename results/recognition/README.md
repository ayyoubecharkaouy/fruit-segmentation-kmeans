# Résultats de reconnaissance

La version active utilise `models/fruit_classifier.joblib`.

- `region_correction/` : tableaux, évaluation et comparaison de la méthode révisée.
- `user_tests/` : images, masques et JSON actifs des deux photos de diagnostic.
- `before_region_correction/` : modèle, inventaire et sorties du premier prototype avant révision.

Les résultats du prototype historique sont centralisés dans `before_region_correction/`, sans copies en doublon à la racine. Le bilan actif est `region_correction/evaluation_summary.json`.
