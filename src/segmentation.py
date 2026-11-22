from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import time
from sklearn.metrics import silhouette_score
from PIL import Image
from color_spaces import prepare_color_features
import matplotlib.pyplot as plt
import os


# fonction pour segmenter les pixels en utilisant KMeans
def segment_kmeans(features, k):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, algorithm='lloyd', random_state=42)
    labels = model.fit_predict(features)
    return model, labels

# fonction pour reconstruire l'image RGB à partir des labels et des moyennes de clusters
def reconstruct_rgb(rgb, labels, k):
    # utiliser les pixels RGB et les groupes fournis a la fonction
    rgb_pixels = rgb.reshape(-1, 3)
    cluster_rgb_means = np.zeros((k, 3), dtype=np.float32)
    for cluster_id in range(k):
        cluster_pixels = rgb_pixels[labels == cluster_id]
        if len(cluster_pixels) > 0:
            cluster_rgb_means[cluster_id] = np.mean(cluster_pixels, axis=0)

    reconstructed_pixels = cluster_rgb_means[labels]
    reconstructed_rgb = reconstructed_pixels.reshape(rgb.shape)

    return reconstructed_rgb, cluster_rgb_means


def run_experiments(image_id) :
    image = Image.open(f"data/processed/{image_id}.png")
    rgb = np.array(image).astype(np.float32) / 255.0
    features = prepare_color_features(rgb)
    feature_sets = {
        'RGB': features[0],
        'HSV_circulaire': features[1],
        'Lab': features[2],
    }
    # creation de dossier labels
    os.makedirs('results/labels', exist_ok=True)
    # creation d'un dossier pour stocker ces resultats sous fichiers CSV
    os.makedirs('results/tables', exist_ok=True)
    # creation d'une dossier pour stocker les images de resultats
    os.makedirs('results/figures', exist_ok=True)

    # ====== choisir au max 2000 index de pixel de l'image aleatoire (pour faire une test)
    experiment_results = [] # liste utilise pour creer un tableux comparative
    rng = np.random.default_rng(42)
    n_pixels = features[0].shape[0]
    sample_indices = rng.choice(
        n_pixels,
        size=min(2000, n_pixels),
        replace=False
    )

    for feature in feature_sets.items():
        space_name, space_features = feature[0], feature[1]
        print(f"Segmentation using {space_name} features : {space_features.shape}")
        for k in (3, 4, 5, 6):
            start_time = time.perf_counter()
            model, labels = segment_kmeans(space_features, k=k)
            end_time = time.perf_counter()
            # calculer le temps d'execution de la segmentation
            execution_time = end_time - start_time

            sample_features = space_features[sample_indices] # selectionner les pixels
            sample_labels = labels[sample_indices] # selectionner les groupes de ces pixels
            n_sample_clusters = len(np.unique(sample_labels)) # le nombre de goupes
            silhouette = np.nan
            if 2 <= n_sample_clusters < len(sample_labels):
                silhouette = silhouette_score(sample_features, sample_labels)


            reconstructed_rgb, cluster_rgb_means = reconstruct_rgb(rgb, labels, k)

            # calculer la moyenne des distances au carré entre les pixels RGB originaux et les pixels reconstruits (MEAN SQUARED ERROR)
            mse_rgb = np.mean((rgb - reconstructed_rgb) ** 2)

            # calcul la moyenne des distances au carre entre les pixels et leurs centres
            inertia_per_pixel = model.inertia_ /  space_features.shape[0]
            # ajouter dans experiment_results
            experiment_results.append({
                'image_id': image_id,
                'color_space': space_name, # type de l'espace de couleur utilise pour la segmentation
                'k': k, # nombre de groupes (clusters)
                'seed': 42, # graine pour la reproduction des resultats
                'inertia_per_pixel': inertia_per_pixel, # moyenne des distances au carre entre les pixels et leurs centres
                'mse_rgb': mse_rgb, # moyenne des distances au carre entre les pixels RGB originaux et les pixels reconstruits
                'execution_time': execution_time, # temps d'execution de la segmentation
                'n_iter': model.n_iter_, # nombre d'iterations pour atteindre la convergence
                'silhouette': silhouette # sore de silhouette
            })
        
            label_map = labels.reshape(rgb.shape[0], rgb.shape[1])
            # enregistrement du label_map qui continet les labels de chaque pixel dans un fichier .npy ()
            np.save(f'results/labels/{image_id}_{space_name}_k{k}_seed42_labels.npy', label_map)
        
            plt.figure()
            plt.imshow(label_map, cmap='tab10', interpolation='nearest')
            plt.title(f"{image_id} - {space_name} - KMeans Segmentation (k={k})")
            plt.colorbar(ticks=range(k), boundaries=[i - 0.5 for i in range(k + 1)])
            
            plt.savefig(f'results/figures/{image_id}_{space_name}_k{k}_seed42_clusters.png', dpi=150, bbox_inches='tight')
            plt.close()
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(rgb,  interpolation='nearest')
            axes[0].set_title("Original Image")
            axes[1].imshow(reconstructed_rgb,  interpolation='nearest')
            axes[1].set_title(f"Reconstructed {space_name} (k={k})")
            fig.savefig(f'results/figures/{image_id}_{space_name}_k{k}_seed42_reconstruction.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    plt.show()

    # transformer experiment_results en un tableau pandas
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(f"results/tables/{image_id}_metrics.csv", index=False)

    return results_df

if __name__ == '__main__':
    metadata = pd.read_csv('data/metadata.csv')
    image_ids = metadata['image_id']
    results_list = []

    for image_id in image_ids:
        print(f"================ IMAGE : {image_id} ")
        results = run_experiments(image_id)
        results_list.append(results)
        print("================")

    # transformer les resultats de toutes les images en tableau pandas
    results_list_pd = pd.concat(results_list, ignore_index=True)
    # EN CSV
    results_list_pd.to_csv(f"results/tables/global_results.csv", index=False)
