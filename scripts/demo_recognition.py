"""Exécuter la reconnaissance sur les photos réservées de data/test."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recognition import DEFAULT_MODEL, load_classifier, predict_photo, save_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/recognition/user_tests')
    args = parser.parse_args()
    bundle = load_classifier(args.model)
    photos = sorted(path for path in (ROOT / 'data/test').glob('*')
                    if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
    if not photos:
        raise FileNotFoundError('Aucune photo dans data/test.')
    for path in photos:
        result, original, mask = predict_photo(path, bundle)
        image_path, json_path = save_prediction(result, original, mask, args.output_dir)
        print(f'{path.name} : {len(result["detections"])} régions ; catégories = {", ".join(result["classes_presentes"])}')
        print(f'Image annotée : {image_path}')
        print(f'Résultats : {json_path}')


if __name__ == '__main__':
    main()
