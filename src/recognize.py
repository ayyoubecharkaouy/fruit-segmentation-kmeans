"""Localiser et nommer les produits d'une photo, sans annotation à l'inférence."""

import argparse
from pathlib import Path

from recognition import DEFAULT_MODEL, ROOT, load_classifier, predict_photo, save_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='Photo à analyser')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/recognition/predictions')
    parser.add_argument('--min-score', type=float, default=None, help='Seuil des votes du classifieur, entre 0 et 1')
    args = parser.parse_args()
    try:
        bundle = load_classifier(args.model)
        result, original, mask = predict_photo(args.image, bundle, args.min_score)
        image_path, json_path = save_prediction(result, original, mask, args.output_dir)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(1, f'Erreur : {error}\n')
    print('Catégories reconnues : ' + (', '.join(result['classes_presentes']) or 'aucune catégorie retenue'))
    for region in result['detections']:
        print(f"Région {region['region_id']} : {region['label']} ; score = {region['score']:.3f} ; boîte = {region['bbox']}")
    print(f'Image annotée : {image_path}')
    print(f'Résultats JSON : {json_path}')


if __name__ == '__main__':
    main()
