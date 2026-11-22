import pandas as pd
import numpy as np
from PIL import Image
import os
# analyser des resultats
# import le fichier .csv
results = pd.read_csv('results/tables/global_results.csv')
# charger aussi note metadata
metadata = pd.read_csv('data/metadata.csv')

# ajoute colonne split a results
results = results.merge(
    metadata[['image_id', 'split']],
    on='image_id',
    how='left',
    validate='many_to_one'
)

# recupirer seulement les lignes ou split == mise_au_point et garder les autres images pour l'evolution
tuning_results = results[
    results['split'] == 'mise_au_point'
].copy()

tuning_groups = tuning_results.groupby(['color_space', 'k'])

# generer une resumer on utilisant les fonctions 'mean', 'std' et 'count'
tuning_summary = tuning_groups[
    ['inertia_per_pixel', 'mse_rgb', 'silhouette', 'execution_time']
].agg(['mean', 'std', 'count'])

# transfore les noms des colonnes de tuning_summary en reaunissant le nom de la mesure et celui du calcul
# pour obtinir des noms simples
tuning_summary.columns = [
    f'{measure}_{stat}'
    for measure, stat in tuning_summary.columns
]
tuning_summary = tuning_summary.reset_index()
# enregistrer sous forme fichier csv
tuning_summary.to_csv("results/tables/tuning_summary.csv", index=False)

# charger une exemple de fichier qui contient les label (groupe) de chaque pixel
label_map = np.load("results/labels/img001_RGB_k3_seed42_labels.npy")

def create_predicted_mask(label_map):
    height, width = label_map.shape
    # counteur size de 2% du plus petit cote avec un min de 1
    border_width = max(1, round(0.02 * min(width, height)))

    # Création de la bande du contour pour identifier le groupe du fond.
    border_mask = np.zeros_like(label_map, dtype=bool) # des 0 en False donne meme size de label_map mais avec des valeur "False"
    # ajouter le border dans mask
    border_mask[:border_width,:] = True # bord supérieur
    border_mask[:,:border_width] = True # bord gauche
    border_mask[-border_width:, :] = True  # bord inférieur
    border_mask[:, -border_width:] = True  # bord droit

    # reuprer les goupes de pixel de border (groupes de contour)
    border_labels = label_map[border_mask == True] 

    # le nombre de pixels pour chaque groupe (contour)
    border_counts = np.bincount(border_labels)

    # Le groupe le plus fréquent sur le contour représente le fond.
    background_cluster = np.argmax(border_counts)

    predicted_mask = label_map != background_cluster  # False : fond ; True : produit.

    return predicted_mask, background_cluster

predicted_mask, background_cluster = create_predicted_mask(label_map)

image = Image.open("data/masks/img001_mask.png") # charger image 001 qui on a deja defaire fond avec 0 et fruit 255
reference_mask = np.array(image) # image en array (numpy)
reference_mask = reference_mask > 0 # transformer array en boolean array (fond: False, fruit: True )

def compute_overlap_scores(reference_mask, predicted_mask):
    # Intersection et union des masques de référence et de prédiction.
    intersection =  (reference_mask & predicted_mask).sum()  # Nombre de pixels de produit communs.
    union =  (reference_mask | predicted_mask).sum()
    iou = np.nan
    if union > 0:
        iou = intersection / union # SCORE: 1 bon 0 mauvise 


    dice = np.nan
    if (intersection + union) != 0:
        dice = (2 * intersection) / (intersection + union) # 2eme SCORE: 1 pour un accord parfait et 0 sans zone commune

    return iou, dice

iou, dice = compute_overlap_scores(reference_mask, predicted_mask)

# Lecture des identifiants des images annotées manuellement.
with open('data/annotation_selection.txt', 'r') as f:
    annotated_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]

# SELECTION SEULEMENT LES RESULTATS DE TUNING_RESULTS QUI SONT EXISTE DANS annotated_ids
annotated_tuning_results = tuning_results[tuning_results['image_id'].isin(annotated_ids)].copy()

overlap_results = []

for row in annotated_tuning_results.itertuples(index=False):
    label_map = np.load(f"results/labels/{row.image_id}_{row.color_space}_k{row.k}_seed{row.seed}_labels.npy")

    # predition de mask
    predicted_mask, background_cluster = create_predicted_mask(label_map)

    image = Image.open(f"data/masks/{row.image_id}_mask.png") # charger image 001 qui on a deja defaire fond avec 0 et fruit 255
    reference_mask = np.array(image) # image en array (numpy)
    reference_mask = reference_mask > 0 # transformer array en boolean array (fond: False, fruit: True )

    # scores
    iou, dice = compute_overlap_scores(reference_mask, predicted_mask)

    overlap_results.append({
        'image_id': row.image_id,
        'color_space': row.color_space,
        'k': row.k,
        'seed': row.seed,
        'split': row.split,
        'background_cluster': background_cluster,
        'iou': iou,
        'dice': dice
    })


# transformer la boucle en tableau pandas
overlap_df = pd.DataFrame(overlap_results)

# eneregister
overlap_df.to_csv("results/tables/tuning_overlap_scores.csv", index=False)
    
overlap_groups = overlap_df.groupby(['color_space', 'k'])

# generer une resumer on utilisant les fonctions 'mean', 'std' et 'count'
overlap_summary = overlap_groups[
    ['iou', 'dice']
].agg(['mean', 'std', 'count'])

# transfore les noms des colonnes de overlap_summary en reaunissant le nom de la mesure et celui du calcul
# pour obtinir des noms simples
overlap_summary.columns = [
    f'{measure}_{stat}'
    for measure, stat in overlap_summary.columns
]
overlap_summary = overlap_summary.reset_index()
# enregistrer sous forme fichier csv
overlap_summary.to_csv("results/tables/tuning_overlap_summary.csv", index=False)

# classer les configurations par IoU moyenne décroissante
ranked_overlap = overlap_summary.sort_values('iou_mean', ascending=False)
print(ranked_overlap.to_string(index=False))

# examiner les résultats individuels de Lab avec K = 3, en commençant par les échecs
candidate_results = overlap_df[
    (overlap_df['color_space'] == 'Lab') & (overlap_df['k'] == 3)
].sort_values('iou', ascending=True)
print("\nRésultats individuels : Lab, K = 3 (IoU croissante)")
print(candidate_results[['image_id', 'iou', 'dice']].to_string(index=False))

# configuration fixée avant l'évaluation finale : Lab, K = 3, graine 42
# la règle conserve un seul groupe de fond, majoritaire sur le contour
final_results = results[
    (results['split'] == 'evaluation')
    & results['image_id'].isin(annotated_ids)
    & (results['color_space'] == 'Lab')
    & (results['k'] == 3)
    & (results['seed'] == 42)
].copy()

# calculer les scores finaux avec les réglages déjà fixés
final_overlap_results = []
for row in final_results.itertuples(index=False):
    label_map = np.load(
        f"results/labels/{row.image_id}_{row.color_space}_k{row.k}_seed{row.seed}_labels.npy",
        allow_pickle=False
    )
    predicted_mask, background_cluster = create_predicted_mask(label_map)
    with Image.open(f"data/masks/{row.image_id}_mask.png") as image:
        reference_mask = np.array(image) > 0

    if reference_mask.shape != predicted_mask.shape:
        raise ValueError(f"Dimensions des masques différentes pour {row.image_id}")

    iou, dice = compute_overlap_scores(reference_mask, predicted_mask)
    final_overlap_results.append({
        'image_id': row.image_id,
        'color_space': row.color_space,
        'k': row.k,
        'seed': row.seed,
        'split': row.split,
        'background_cluster': background_cluster,
        'selection_method': 'border_majority_2pct',
        'iou': iou,
        'dice': dice
    })

final_overlap_df = pd.DataFrame(final_overlap_results)
final_overlap_df.to_csv("results/tables/final_overlap_scores.csv", index=False)
print("\nÉvaluation finale : Lab, K = 3, graine 42")
print(final_overlap_df[['image_id', 'iou', 'dice']].to_string(index=False))

# résumer les scores finaux, une image comptant pour une observation
final_groups = final_overlap_df.groupby(['color_space', 'k', 'seed', 'selection_method'])
final_summary = final_groups[['iou', 'dice']].agg(['mean', 'std', 'count'])
final_summary.columns = [
    f'{measure}_{stat}'
    for measure, stat in final_summary.columns
]
final_summary = final_summary.reset_index()
final_summary.to_csv("results/tables/final_overlap_summary.csv", index=False)
print("\nRésumé de l'évaluation finale")
print(final_summary.to_string(index=False))






    
