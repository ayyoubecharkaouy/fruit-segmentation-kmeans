"""Contrats d'apprentissage, localisation multiple et inférence sans annotations."""

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import recognition
from evaluate_recognition import box_iou, match_detections, object_classification_pairs
from scripts.prepare_recognition_dataset import prepare


class FixedEstimator:
    classes_ = np.array(recognition.CLASSES)

    def predict_proba(self, x):
        return np.tile([0.8, 0.1, 0.1], (len(x), 1))


def scene_image(size=(420, 180)):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    sx, sy = size[0] / 420, size[1] / 180
    for x0, y0, x1, y1, colour in (
        (25, 35, 100, 125, '#b02820'),
        (160, 45, 200, 150, '#e0bf20'),
        (280, 30, 370, 135, '#187c36'),
    ):
        draw.ellipse((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)), fill=colour)
    return image


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fruit-recognition-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photo = self.root / 'scene.png'
        scene_image().save(self.photo)
        _, rgb = recognition.load_photo(self.photo)
        masks = recognition.object_masks(recognition.foreground_mask(rgb))
        self.bundle = {
            'estimator': FixedEstimator(), 'classes': list(recognition.CLASSES),
            'feature_version': recognition.FEATURE_VERSION,
            'feature_count': len(recognition.extract_features(rgb, masks[0])),
            'min_score': 0.45,
        }

    def test_three_regions_and_blank_background(self):
        _, rgb = recognition.load_photo(self.photo)
        masks = recognition.object_masks(recognition.foreground_mask(rgb))
        self.assertEqual(len(masks), 3)
        self.assertFalse(np.any(masks[0] & masks[1]))
        self.assertFalse(recognition.foreground_mask(np.ones((80, 120, 3), np.float32)).any())

    def test_features_reject_bad_masks_and_non_normalized_images(self):
        rgb = np.ones((40, 60, 3), np.float32)
        with self.assertRaises(ValueError):
            recognition.extract_features(rgb, np.zeros((40, 60), bool))
        with self.assertRaises(ValueError):
            recognition.extract_features(rgb, np.ones((20, 60), bool))
        with self.assertRaises(ValueError):
            recognition.foreground_mask(rgb * 255)
        with self.assertRaises(ValueError):
            recognition.foreground_mask(rgb, min_area_fraction=0)

    def test_inference_ignores_metadata_and_filename(self):
        renamed = self.root / 'unrelated_name.png'
        renamed.write_bytes(self.photo.read_bytes())
        with patch.object(recognition.pd, 'read_csv', side_effect=AssertionError('Metadata read during inference')):
            result, _, _ = recognition.predict_photo(self.photo, self.bundle)
            other, _, _ = recognition.predict_photo(renamed, self.bundle)
        self.assertEqual(len(result['detections']), 3)
        self.assertEqual(result['detections'], other['detections'])
        self.assertEqual(result['classes_presentes'], ['banane'])

    def test_uncertain_regions_are_not_named_as_a_fruit(self):
        result, _, _ = recognition.predict_photo(self.photo, self.bundle, min_score=0.9)
        self.assertEqual(result['classes_presentes'], [])
        self.assertTrue(all(item['label'] == 'indetermine' for item in result['detections']))
        with self.assertRaises(ValueError):
            recognition.predict_photo(self.photo, self.bundle, min_score=1.1)

    def test_boxes_return_to_original_image_coordinates(self):
        scene_image((1260, 540)).save(self.photo)
        result, original, _ = recognition.predict_photo(self.photo, self.bundle, max_side=420)
        self.assertEqual(original.size, (1260, 540))
        self.assertEqual(result['processing_size'], [420, 180])
        self.assertEqual(len(result['detections']), 3)
        first = result['detections'][0]['bbox']
        self.assertLess(abs(first[0] - 75), 8)
        self.assertLess(abs(first[1] - 105), 8)
        for item in result['detections']:
            x0, y0, x1, y1 = item['bbox']
            self.assertTrue(0 <= x0 < x1 <= 1260 and 0 <= y0 < y1 <= 540)

    def test_transparency_is_composited_on_white(self):
        image = Image.new('RGBA', (100, 80), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse((30, 20, 70, 65), fill=(200, 10, 10, 255))
        path = self.root / 'transparent.png'
        image.save(path)
        original, rgb = recognition.load_photo(path)
        self.assertEqual(original.getpixel((0, 0)), (255, 255, 255))
        np.testing.assert_array_equal(rgb[0, 0], [1, 1, 1])

    def test_saved_outputs_and_model_schema(self):
        result, original, mask = recognition.predict_photo(self.photo, self.bundle)
        image_path, json_path = recognition.save_prediction(result, original, mask, self.root / 'output')
        with Image.open(image_path) as image:
            image.verify()
        saved = json.loads(json_path.read_text())
        self.assertEqual(len(saved['detections']), 3)
        self.assertIn('non calibrée', saved['score_interpretation'])
        model_path = self.root / 'model.joblib'
        joblib.dump(self.bundle, model_path)
        self.assertEqual(recognition.load_classifier(model_path)['feature_version'], recognition.FEATURE_VERSION)
        joblib.dump({'feature_version': 'obsolete'}, model_path)
        with self.assertRaises(ValueError):
            recognition.load_classifier(model_path)

    def _training_fixture(self):
        directory = self.root / 'data/processed'
        directory.mkdir(parents=True)
        rows = []
        for index, category in enumerate(recognition.CLASSES):
            image_id = f'train{index}'
            scene_image().save(directory / f'{image_id}.png')
            rows.append({'image_id': image_id, 'fruit_types': category, 'source': 'fruits360_simple',
                         'scene_id': f'scene{index}', 'split': 'mise_au_point'})
        # Son fichier image est volontairement absent : l'apprentissage ne doit pas le lire.
        rows.append({'image_id': 'reserved', 'fruit_types': 'pomme', 'source': 'fruits360_simple',
                     'scene_id': 'reserved_scene', 'split': 'evaluation'})
        metadata = pd.DataFrame(rows)
        metadata.to_csv(self.root / 'data/metadata.csv', index=False)
        return metadata

    def test_training_does_not_open_evaluation_images(self):
        self._training_fixture()
        x, records = recognition.collect_training_samples(self.root)
        self.assertEqual(len(x), 3)
        self.assertTrue((records['split'] == 'mise_au_point').all())
        self.assertNotIn('reserved', set(records['image_id']))
        self.assertTrue(np.isfinite(x).all())

    def test_training_rejects_shared_scenes_between_partitions(self):
        metadata = self._training_fixture()
        metadata.loc[metadata.image_id == 'reserved', 'scene_id'] = 'scene0'
        metadata.to_csv(self.root / 'data/metadata.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'scène'):
            recognition.collect_training_samples(self.root)

    def test_recognition_protocol_uses_its_own_partition_without_opening_test(self):
        metadata = self._training_fixture()
        metadata['recognition_split'] = ['train', 'train', 'train', 'test']
        metadata['filename'] = [f'data/processed/{image_id}.png' for image_id in metadata.image_id]
        metadata['sha256'] = [f'{index:064x}' for index in range(len(metadata))]
        # Une ancienne image de test de segmentation peut être apprise dans le
        # nouveau protocole ; le test externe reste strictement séparé.
        metadata.loc[0, 'split'] = 'evaluation'
        directory = self.root / 'data/recognition'
        directory.mkdir()
        metadata.to_csv(directory / 'dataset.csv', index=False)
        _, records = recognition.collect_training_samples(self.root)
        self.assertEqual(len(records), 3)
        self.assertTrue((records['split'] == 'recognition_train').all())
        self.assertIn('evaluation', records['segmentation_split'].tolist())
        self.assertNotIn('reserved', records['image_id'].tolist())

    def test_recognition_protocol_rejects_identical_train_and_test(self):
        metadata = self._training_fixture()
        metadata['recognition_split'] = ['train', 'train', 'train', 'test']
        metadata['filename'] = [f'data/processed/{image_id}.png' for image_id in metadata.image_id]
        metadata['sha256'] = ['same', 'another', 'third', 'same']
        directory = self.root / 'data/recognition'
        directory.mkdir()
        metadata.to_csv(directory / 'dataset.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'identique'):
            recognition.collect_training_samples(self.root)

    def test_detection_matching_is_one_to_one_and_class_sensitive(self):
        box = [10, 10, 30, 30]
        predictions = [{'bbox': box, 'label': 'pomme'}, {'bbox': box, 'label': 'pomme'}]
        self.assertEqual(box_iou(box, box), 1)
        self.assertEqual(box_iou(box, [40, 40, 50, 50]), 0)
        self.assertEqual(match_detections(predictions, [box], 'pomme'), (1, 1, 0))
        self.assertEqual(match_detections(predictions, [box], 'banane'), (0, 2, 1))
        self.assertEqual(match_detections([], [box], 'pomme'), (0, 0, 1))
        self.assertEqual(match_detections([{'bbox': box, 'label': 'pomme'},
                                          {'bbox': [40, 40, 60, 60], 'label': 'banane'}],
                                         [box, [40, 40, 60, 60]], ['pomme', 'banane']), (2, 0, 0))

    def test_piment_alias_is_explicit(self):
        self.assertEqual(recognition.canonical_class('piment'), 'poivron')

    def test_dataset_preparation_preserves_original_inventory(self):
        self._training_fixture()
        scene_image().save(self.root / 'data/processed/reserved.png')
        test_directory = self.root / 'data/test'
        test_directory.mkdir()
        photo = scene_image()
        photo.putpixel((0, 0), (240, 240, 240))
        test_path = test_directory / 'new_photo.png'
        photo.save(test_path)
        (test_directory / 'annotations.json').write_text(json.dumps({
            test_path.name: {'image_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(),
                             'expected_classes': ['pomme', 'banane', 'poivron']}
        }))
        original = (self.root / 'data/metadata.csv').read_bytes()
        dataset = prepare(self.root)
        self.assertEqual((self.root / 'data/metadata.csv').read_bytes(), original)
        self.assertEqual((dataset.recognition_split == 'train').sum(), 4)
        test = dataset[dataset.recognition_split == 'test'].iloc[0]
        self.assertEqual(test.fruit_types, 'banane|poivron|pomme')
        self.assertEqual(test.image_id, 'external_new_photo')

    def test_dataset_preparation_rejects_duplicate_external_photo(self):
        self._training_fixture()
        scene_image().save(self.root / 'data/processed/reserved.png')
        test_directory = self.root / 'data/test'
        test_directory.mkdir()
        (test_directory / 'duplicate.png').write_bytes(self.photo.read_bytes())
        with self.assertRaisesRegex(ValueError, 'identique'):
            prepare(self.root)

    def test_confusion_matches_geometry_before_reading_classes(self):
        predictions = [{'bbox': [0, 0, 20, 20], 'label': 'poivron'},
                       {'bbox': [30, 0, 50, 20], 'label': 'indetermine'}]
        pairs = object_classification_pairs(predictions, [[0, 0, 20, 20], [30, 0, 50, 20]], ['pomme', 'banane'])
        self.assertCountEqual(pairs, [('pomme', 'poivron'), ('banane', 'indetermine')])
        self.assertEqual(object_classification_pairs([], [[0, 0, 20, 20]], ['pomme']), [('pomme', 'aucun')])


if __name__ == '__main__':
    unittest.main()
