# test de pretretement d'une seul image

# importations des bibliotheques necessaires
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

def preprocess_image(image_path, output_path):
    # verifier si le chemin de l'image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Le chemin de l'image '{image_path}' n'existe pas.")
    
    # chargemenr de l'image 
    image = Image.open(image_path)
    # image name
    image_name = os.path.basename(image_path)

    # corriger son orientation (orienter l'image dans le bon sens selon les métadonnées EXIF)
    image = ImageOps.exif_transpose(image)

    # convertir l'image en RGB, pour avoir 3 canaux de couleur (rouge, vert, bleu) pour chaque pixel de l'image
    image = image.convert('RGB')

    # en reduisant la taille de l'image a 512x512 pixels au maximum (pour ne pas avoir des images trop grandes qui peuvent ralentir le traitement)
    if image.size[0] > 512 or image.size[1] > 512:
        max_size = (512, 512)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # convertir l'image en tableau numpy
    image_array = np.array(image)

    # normaliser les valeurs des pixels de l'image pour qu'elles soient comprises entre 0 et 1
    image_norm = image_array / 255.0

    # affichage de dim de tableau normalise
    print("Dimensions du tableau normalisé :", image_norm.shape) 
    # affichage de max et min de tableau normalise
    print("Valeur maximale du tableau normalisé :", np.max(image_norm))
    print("Valeur minimale du tableau normalisé :", np.min(image_norm))

    # enregistrer l'image au format PNG dans le dossier data/processed
    # creer le dossier data/processed s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)
    image.save(os.path.join(output_path, image_name.replace(image_name.split('.')[-1], 'png')), format='PNG')

    return image_norm

# exemmple 
# image_norm = preprocess_image('data/raw/fruits360_multi/img046.jpg', 'data/processed')
# print("Image prétraitée et enregistrée avec succès !", "Shape:", image_norm.shape, "Dtype:", image_norm.dtype)

# fire le fichier data/metadata.csv qui contient metadata de notre dataset
metadata_file = pd.read_csv('data/metadata.csv')

# application de la fonction preprocess_image sur chaque image du dataset
for index, row in metadata_file.iterrows():
    image_path = f"data/{row['filename']}"
    output_path = 'data/processed'
    print('=================================================================')
    try:
        preprocess_image(image_path, output_path)
        print(f"Image '{image_path}' prétraitée et enregistrée avec succès !")
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image '{image_path}': {e}")
        break

    print('=================================================================')