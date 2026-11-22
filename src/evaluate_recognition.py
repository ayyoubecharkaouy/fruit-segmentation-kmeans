"""Évaluer la reconnaissance sur la partition réservée, sans réentraînement."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from recognition import (
    CLASSES, DEFAULT_MODEL, ROOT, _polygon_mask, canonical_class,
    load_classifier, predict_photo,
)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(predictions, reference_boxes, reference_class, threshold=0.5):
    """Apparier une fois chaque boîte, avec catégorie correcte et IoU suffisante."""
    if not predictions or not reference_boxes:
        return 0, len(predictions), len(reference_boxes)
    ious = np.array([[box_iou(item['bbox'], box) for box in reference_boxes] for item in predictions])
    reference_classes = [reference_class] * len(reference_boxes) if isinstance(reference_class, str) else reference_class
    if len(reference_classes) != len(reference_boxes):
        raise ValueError('Chaque boîte de référence doit avoir une catégorie.')
    correct_class = np.array([[item['label'] == category for category in reference_classes] for item in predictions])
    eligible = (ious >= threshold) & correct_class
    reward = eligible * (min(ious.shape) + 1 + ious)
    rows, cols = linear_sum_assignment(-reward)
    true_positive = int(eligible[rows, cols].sum())
    return true_positive, len(predictions) - true_positive, len(reference_boxes) - true_positive


def object_classification_pairs(predictions, boxes, classes, threshold=0.5):
    """Apparier géométriquement avant de comparer les catégories, sans les favoriser."""
    matched_predictions, matched_references, pairs = set(), set(), []
    if predictions and boxes:
        ious = np.array([[box_iou(item['bbox'], box) for box in boxes] for item in predictions])
        eligible = ious >= threshold
        reward = eligible * (min(ious.shape) + 1 + ious)
        rows, cols = linear_sum_assignment(-reward)
        for row, col in zip(rows, cols):
            if eligible[row, col]:
                pairs.append((classes[col], predictions[row]['label']))
                matched_predictions.add(row)
                matched_references.add(col)
    pairs.extend((category, 'aucun') for index, category in enumerate(classes) if index not in matched_references)
    pairs.extend(('hors_reference', item['label']) for index, item in enumerate(predictions) if index not in matched_predictions)
    return pairs


def evaluate(root=ROOT, model_path=DEFAULT_MODEL, output_dir=None):
    root = Path(root)
    bundle = load_classifier(model_path)
    external_protocol = bundle.get('training_protocol') == 'external_test_photos'
    metadata = pd.read_csv(root / ('data/recognition/dataset.csv' if external_protocol else 'data/metadata.csv'))
    if external_protocol:
        selected = metadata[(metadata['recognition_split'] == 'test') & metadata['fruit_types'].notna()]
    else:
        selected = metadata[
            (metadata['split'] == 'evaluation')
            & metadata['fruit_types'].map(canonical_class).isin(CLASSES)
        ]
    if selected.empty:
        raise ValueError('Aucune image d’évaluation avec une catégorie exploitable.')
    if set(selected['image_id']) & set(bundle['training_image_ids']):
        raise ValueError('Des images de test sont présentes dans l’apprentissage.')
    if set(selected['scene_id']) & set(bundle['training_scene_ids']):
        raise ValueError('Une scène du test est présente dans l’apprentissage.')

    rows, detection_rows, true_presence, predicted_presence = [], [], [], []
    classification_pairs = []
    external_annotation_path = root / 'data/test/annotations.json'
    external_annotations = json.loads(external_annotation_path.read_text()) if external_protocol and external_annotation_path.exists() else {}
    for row in selected.itertuples(index=False):
        expected_classes = sorted({canonical_class(value) for value in row.fruit_types.split('|')})
        if not set(expected_classes) <= set(CLASSES):
            raise ValueError(f'Catégories de référence inconnues : {row.image_id}')
        photo_path = root / row.filename if external_protocol else root / 'data/processed' / f'{row.image_id}.png'
        result, original, _ = predict_photo(photo_path, bundle)
        main = max(result['detections'], key=lambda item: item['area_processing_pixels'], default=None)
        main_label = main['label'] if main else 'aucun'
        rows.append({
            'image_id': row.image_id, 'split': 'recognition_test' if external_protocol else row.split, 'source': row.source,
            'expected_class': '|'.join(expected_classes), 'main_prediction': main_label,
            'classes_detected': '|'.join(result['classes_presentes']),
            'n_regions': len(result['detections']),
            'exact_presence_match': result['classes_presentes'] == expected_classes,
            'main_class_correct': main_label == expected_classes[0] if len(expected_classes) == 1 else None,
        })
        true_presence.append([category in expected_classes for category in CLASSES])
        predicted_presence.append([category in result['classes_presentes'] for category in CLASSES])

        annotation_path = root / 'data/masks' / f'{row.image_id}.json'
        if not external_protocol and annotation_path.exists():
            annotation = json.loads(annotation_path.read_text())
            if (annotation['imageWidth'], annotation['imageHeight']) != original.size:
                raise ValueError(f'Dimensions de référence incorrectes : {row.image_id}')
            boxes = []
            for shape in annotation['shapes']:
                mask = _polygon_mask(shape, original.size)
                yy, xx = np.nonzero(mask)
                if len(yy):
                    boxes.append([int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)])
            tp, fp, fn = match_detections(result['detections'], boxes, expected_classes[0])
            classification_pairs.extend(object_classification_pairs(result['detections'], boxes, [expected_classes[0]] * len(boxes)))
            detection_rows.append({'image_id': row.image_id, 'reference_class': expected_classes[0],
                                   'n_reference_regions': len(boxes), 'tp': tp, 'fp': fp, 'fn': fn})
        entry = external_annotations.get(photo_path.name)
        if external_protocol and entry and entry.get('boxes'):
            if hashlib.sha256(photo_path.read_bytes()).hexdigest() != entry['image_sha256']:
                raise ValueError(f'La photo a changé depuis son annotation : {photo_path.name}')
            boxes, classes = [], []
            for item in entry['boxes']:
                normalized = item['bbox_normalized']
                boxes.append([normalized[0] * original.width, normalized[1] * original.height,
                              normalized[2] * original.width, normalized[3] * original.height])
                classes.append(item['class_name'])
            tp, fp, fn = match_detections(result['detections'], boxes, classes)
            classification_pairs.extend(object_classification_pairs(result['detections'], boxes, classes))
            detection_rows.append({'image_id': row.image_id, 'reference_class': 'plusieurs',
                                   'n_reference_regions': len(boxes), 'tp': tp, 'fp': fp, 'fn': fn})

    per_image = pd.DataFrame(rows)
    true_presence = np.asarray(true_presence)
    predicted_presence = np.asarray(predicted_presence)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_presence, predicted_presence, average=None, zero_division=0
    )
    per_class = pd.DataFrame({'class_name': CLASSES, 'precision': precision, 'recall': recall,
                              'f1': f1, 'n_images': support})
    simple = per_image[per_image['source'] == 'fruits360_simple']
    confusion_labels = [*CLASSES, 'indetermine', 'aucun']
    confusion = pd.DataFrame(
        confusion_matrix(simple['expected_class'], simple['main_prediction'], labels=confusion_labels)
        if len(simple) else np.zeros((len(confusion_labels), len(confusion_labels)), dtype=int),
        index=confusion_labels, columns=confusion_labels,
    )
    confusion.index.name = 'expected_class'
    detections = pd.DataFrame(detection_rows, columns=['image_id', 'reference_class', 'n_reference_regions', 'tp', 'fp', 'fn'])
    object_confusion = pd.DataFrame(0, index=[*CLASSES, 'hors_reference'], columns=[*CLASSES, 'indetermine', 'aucun'])
    object_confusion.index.name = 'expected_class'
    for expected_class, predicted_class in classification_pairs:
        object_confusion.loc[expected_class, predicted_class] += 1
    summary = {
        'training_images': len(bundle['training_image_ids']), 'training_regions': bundle['n_regions'],
        'training_protocol': bundle.get('training_protocol', 'segmentation_split'),
        'evaluation_status': 'exploratory_development_examples' if external_protocol else 'initial_protocol',
        'external_images_used_for_fitting': False,
        'external_images_used_for_model_selection': bundle.get('external_images_used_for_model_selection', False),
        'fitting_scope': bundle.get('fitting_scope', 'apprentissage supervisé du classifieur'),
        'test_images': len(per_image),
        'photo_presence_exact_match': float(per_image['exact_presence_match'].mean()),
        'photo_presence_macro_f1': float(f1.mean()),
        'simple_test_images': len(simple),
        'simple_main_class_accuracy': float(simple['main_class_correct'].mean()) if len(simple) else None,
        'annotated_test_images': len(detections),
        'annotated_reference_regions': int(object_confusion.loc[list(CLASSES)].to_numpy().sum()),
        'min_score': bundle['min_score'],
        'localisation': bundle.get('localisation', 'adaptive_border_lab'),
        'scope': 'Présence des catégories par photo, distincte de la justesse des étiquettes de chaque objet.',
        'box_evaluation_scope': 'Boîtes de référence approximatives pour les photos externes annotées ; sinon boîtes dérivées des polygones visibles. Bilan exploratoire.',
    }
    if len(detections):
        tp, fp, fn = (int(detections[column].sum()) for column in ('tp', 'fp', 'fn'))
        summary.update({
            'box_iou_threshold': 0.5, 'box_tp': tp, 'box_fp': fp, 'box_fn': fn,
            'box_precision': tp / (tp + fp) if tp + fp else 0.0,
            'box_recall': tp / (tp + fn) if tp + fn else 0.0,
            'box_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        })
    corrected = bundle.get('localisation') == 'kmeans_regions_with_background_rejection_v2'
    output = Path(output_dir) if output_dir else root / ('results/recognition/region_correction' if corrected else 'results/recognition')
    output.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(output / 'test_predictions.csv', index=False)
    per_class.to_csv(output / 'class_presence_metrics.csv', index=False)
    confusion_path = output / 'simple_confusion_matrix.csv'
    if len(simple):
        confusion.to_csv(confusion_path)
    elif confusion_path.exists():
        confusion_path.unlink()
    detections.to_csv(output / 'annotated_detection_metrics.csv', index=False)
    object_confusion.to_csv(output / 'object_confusion_matrix.csv')
    (output / 'evaluation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary, per_image = evaluate(args.root, args.model, args.output_dir)
    print(per_image[['image_id', 'expected_class', 'classes_detected', 'n_regions', 'exact_presence_match']].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
