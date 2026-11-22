"""Partition propre à la reconnaissance, indépendante du protocole K-means.

Les anciennes images étiquetées fournissent l'apprentissage. Les nouvelles
photos de data/test sont réservées au test externe et ne sont pas apprises.
L'inventaire de segmentation data/metadata.csv n'est jamais modifié.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'pomme', 'banane', 'poivron', 'piment'}


def prepare(root=ROOT):
    root = Path(root)
    metadata = pd.read_csv(root / 'data/metadata.csv')
    explicit_path = root / 'data/recognition/instance_classes.json'
    explicit = json.loads(explicit_path.read_text()) if explicit_path.exists() else {}
    background_path = root / 'data/recognition/background_regions.json'
    backgrounds = json.loads(background_path.read_text()) if background_path.exists() else {}
    records = []
    for row in metadata.itertuples(index=False):
        simple = row.source in ('fruits360_simple', 'kaggle_chili') and row.fruit_types in CLASSES
        annotated = (root / 'data/masks' / f'{row.image_id}.json').exists()
        valid_annotation = annotated and (row.fruit_types in CLASSES or row.image_id in explicit)
        if not simple and not valid_annotation and row.image_id not in backgrounds:
            continue
        filename = Path('data/processed') / f'{row.image_id}.png'
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if row.image_id in backgrounds and digest != backgrounds[row.image_id]['image_sha256']:
            raise ValueError(f'Image modifiée depuis l’annotation du fond : {row.image_id}')
        records.append({
            'image_id': row.image_id, 'filename': filename.as_posix(),
            'fruit_types': row.fruit_types, 'source': row.source, 'scene_id': row.scene_id,
            'split': row.split, 'recognition_split': 'train',
            'sha256': digest,
        })
    test_directory = root / 'data/test'
    paths = sorted(path for path in test_directory.glob('*') if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
    if not paths:
        raise ValueError('Aucune photo externe dans data/test.')
    annotations_path = test_directory / 'annotations.json'
    references = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
    for path in paths:
        # Les catégories de référence sont des annotations de test, jamais
        # une information utilisée pour prédire les catégories d'une photo.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = references.get(path.name)
        if entry and entry['image_sha256'] != digest:
            raise ValueError(f'La photo a changé depuis son annotation : {path.name}')
        expected = '|'.join(sorted(entry['expected_classes'])) if entry else ''
        records.append({
            'image_id': f'external_{path.stem}', 'filename': path.relative_to(root).as_posix(),
            'fruit_types': expected, 'source': 'external_test', 'scene_id': f'external_{path.stem}',
            'split': '', 'recognition_split': 'test',
            'sha256': digest,
        })
    dataset = pd.DataFrame(records)
    if dataset['image_id'].duplicated().any():
        raise ValueError('Identifiants de reconnaissance dupliqués.')
    train_hashes = set(dataset.loc[dataset.recognition_split == 'train', 'sha256'])
    test_hashes = set(dataset.loc[dataset.recognition_split == 'test', 'sha256'])
    if train_hashes & test_hashes:
        raise ValueError('Une image externe est identique à une image d’apprentissage.')
    output = root / 'data/recognition/dataset.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    return dataset


if __name__ == '__main__':
    dataset = prepare()
    print(dataset.groupby('recognition_split').size().to_string())
    print('Inventaire de reconnaissance : data/recognition/dataset.csv')
    print('Inventaire de segmentation conservé : data/metadata.csv')
