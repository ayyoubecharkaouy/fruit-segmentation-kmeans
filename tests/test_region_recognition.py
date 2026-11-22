"""Propositions, rejet du fond et absence de références pendant l'inférence."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import joblib
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import recognition
import region_recognition as regions


class Classifier:
    classes_ = np.array(recognition.CLASSES)

    def predict_proba(self, x):
        return np.tile([0.9, 0.05, 0.05], (len(x), 1))


class ForegroundClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, score):
        self.score = score

    def predict_proba(self, x):
        return np.tile([1 - self.score, self.score], (len(x), 1))


def bundle(score=0.9):
    return dict(estimator=Classifier(), background_estimator=ForegroundClassifier(score),
                feature_count=615, min_score=0.45, background_threshold=0.5,
                localisation=regions.METHOD, localisation_parameters=regions.PARAMETERS,
                score_interpretation='non calibré')


class RegionRecognitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='classical-regions-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'photo.png'
        image = Image.new('RGB', (240, 120), 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 20, 90, 100), fill=(190, 20, 20))
        draw.ellipse((140, 30, 215, 100), fill=(220, 190, 10))
        image.save(self.path)

    def test_background_rejection_overrides_high_class_score(self):
        _, rgb = recognition.load_photo(self.path)
        proposals = recognition.object_masks(recognition.foreground_mask(rgb))
        with patch.object(regions, 'region_proposals', return_value=(proposals, 'lab_kmeans_region_proposals')):
            result, _, mask = recognition.predict_photo(self.path, bundle(0.05))
        self.assertEqual(result['detections'], [])
        self.assertEqual(result['classes_presentes'], [])
        self.assertEqual(len(result['rejected_regions']), 2)
        self.assertTrue(all(d['reason'] == 'background' for d in result['rejected_regions']))
        self.assertFalse(mask.any())

    def test_uniform_background_keeps_confirmed_foreground_components(self):
        result, _, mask = recognition.predict_photo(self.path, bundle(0.05))
        self.assertFalse(result['background_rejection_applied'])
        self.assertEqual(len(result['detections']), 2)
        self.assertTrue(mask.any())

    def test_fast_appearance_extraction_matches_complete_descriptor(self):
        _, rgb = recognition.load_photo(self.path)
        mask = recognition.foreground_mask(rgb)
        complete = recognition.extract_features(rgb, mask, correct_alignment=True)
        fast = recognition.extract_features(rgb, mask, appearance_only=True)
        np.testing.assert_array_equal(complete[regions.APPEARANCE_COLUMNS], fast[regions.APPEARANCE_COLUMNS])

    def test_background_threshold_balances_classes_on_held_out_scores(self):
        scores = np.array([0.05, 0.1, 0.45, 0.55, 0.9, 0.95])
        reference = np.array([0, 0, 1, 0, 1, 1])
        selected, rows = regions.select_background_threshold(scores, reference, np.ones(6))
        achieved = next(row for row in rows if row['threshold'] == selected)
        self.assertEqual(achieved['balanced_accuracy'], max(row['balanced_accuracy'] for row in rows))
        selected, rows = regions.select_background_threshold(scores, reference, np.ones(6), min_recall=1)
        self.assertEqual(next(row for row in rows if row['threshold'] == selected)['fruit_recall'], 1)

    def test_corrected_alignment_preserves_elongated_shape_after_rotation(self):
        from skimage import transform
        yy, xx = np.ogrid[-50:51, -50:51]
        mask = (yy / 36)**2 + (xx / 10)**2 <= 1
        descriptors = []
        for angle in (0, 25, 45):
            rotated = transform.rotate(mask.astype(float), angle, resize=True, order=0) > 0.5
            rgb = np.ones((*rotated.shape, 3), dtype=np.float32)
            rgb[rotated] = [0.9, 0.7, 0.05]
            features = recognition.extract_features(rgb, rotated, correct_alignment=True)
            descriptors.append(features[61:317])
            extra = regions.extract_features(rgb, rotated)
            self.assertEqual(extra.shape, (733,))
            self.assertTrue(np.isfinite(extra).all())
        for descriptor in descriptors[1:]:
            similarity = np.dot(descriptors[0], descriptor) / np.linalg.norm(descriptors[0]) / np.linalg.norm(descriptor)
            self.assertGreater(similarity, 0.97)

    def test_no_reference_or_filename_information_is_used(self):
        renamed = self.root / 'not_the_same_name.png'
        renamed.write_bytes(self.path.read_bytes())
        with patch.object(regions.pd, 'read_csv', side_effect=AssertionError('Unexpected metadata read')):
            result, original, mask = recognition.predict_photo(self.path, bundle())
            other, _, _ = recognition.predict_photo(renamed, bundle())
        self.assertEqual(result['detections'], other['detections'])
        self.assertEqual(len(result['detections']), 2)
        self.assertEqual(result['classes_presentes'], ['banane'])
        self.assertEqual(original.size, (240, 120))
        self.assertTrue(mask.any())

    def test_blank_image_has_no_detections(self):
        Image.new('RGB', (120, 80), 'white').save(self.path)
        result, _, mask = recognition.predict_photo(self.path, bundle())
        self.assertEqual(result['candidate_count'], 0)
        self.assertEqual(result['detections'], [])
        self.assertFalse(mask.any())

    def test_rescaling_returns_boxes_in_original_photo_coordinates(self):
        with Image.open(self.path) as image:
            image.resize((720, 360)).save(self.path)
        result, original, _ = recognition.predict_photo(self.path, bundle(), max_side=240)
        self.assertEqual(original.size, (720, 360))
        self.assertEqual(result['processing_size'], [240, 120])
        self.assertEqual(len(result['detections']), 2)
        for item in result['detections']:
            x0, y0, x1, y1 = item['bbox']
            self.assertTrue(0 <= x0 < x1 <= 720 and 0 <= y0 < y1 <= 360)
        first = min(result['detections'], key=lambda item: item['bbox'][0])
        self.assertLess(abs(first['bbox'][0] - 60), 10)

    def test_incomplete_rejection_model_is_not_loaded(self):
        model = bundle()
        model.update(feature_version=recognition.FEATURE_VERSION, classes=list(recognition.CLASSES))
        model_path = self.root / 'model.joblib'
        joblib.dump(model, model_path)
        loaded = recognition.load_classifier(model_path)
        self.assertEqual(loaded['model_sha256'], hashlib.sha256(model_path.read_bytes()).hexdigest())
        del model['background_estimator']
        joblib.dump(model, model_path)
        with self.assertRaisesRegex(ValueError, 'rejet du fond'):
            recognition.load_classifier(model_path)

    def test_duplicate_suppression_compares_competing_classes(self):
        def item(box, category, score):
            return dict(bbox=box, label=category, score=score, foreground_score=0.9)
        first = item([10, 10, 80, 90], 'pomme', 0.9)
        duplicate = item([12, 12, 81, 91], 'poivron', 0.7)
        contained = item([20, 20, 50, 60], 'pomme', 0.8)
        separate = item([100, 20, 140, 100], 'banane', 0.85)
        self.assertEqual(regions.suppress_duplicates([duplicate, separate, contained, first]), [first, separate])

    def test_kmeans_proposals_are_deterministic_on_heterogeneous_background(self):
        rgb = np.empty((100, 160, 3), np.float32)
        rgb[:, :80] = [0.0, 0.7, 0.1]
        rgb[:, 80:] = [0.4, 0.2, 0.0]
        yy, xx = np.ogrid[:100, :160]
        target = (yy - 50)**2 + (xx - 80)**2 < 20**2
        rgb[target] = [0.9, 0.0, 0.0]
        self.assertFalse(regions.uniform_border(rgb))
        first, mode = regions.region_proposals(rgb)
        second, _ = regions.region_proposals(rgb)
        self.assertEqual(mode, 'lab_kmeans_region_proposals')
        self.assertEqual(len(first), len(second))
        self.assertTrue(first)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.bool_)
            self.assertEqual(a.shape, rgb.shape[:2])
        self.assertTrue(any(np.count_nonzero(m & target) / np.count_nonzero(m | target) > 0.9 for m in first))

    def test_watershed_separates_objects_connected_by_a_narrow_bridge(self):
        rgb = np.empty((110, 180, 3), np.float32)
        rgb[:, :90] = [0, 0.6, 0.1]
        rgb[:, 90:] = [0.4, 0.2, 0]
        yy, xx = np.ogrid[:110, :180]
        left = (yy - 55)**2 + (xx - 55)**2 < 22**2
        right = (yy - 55)**2 + (xx - 120)**2 < 22**2
        bridge = np.zeros((110, 180), bool)
        bridge[53:58, 55:121] = True
        rgb[left | right | bridge] = [0.95, 0, 0]
        proposals, _ = regions.region_proposals(rgb)
        for target in (left, right):
            self.assertTrue(any(np.count_nonzero(m & target) / np.count_nonzero(m | target) > 0.75 for m in proposals))

    def test_augmentation_preserves_input_and_mask_alignment(self):
        _, rgb = recognition.load_photo(self.path)
        mask = recognition.foreground_mask(rgb)
        rgb_before, mask_before = rgb.copy(), mask.copy()
        examples = list(regions._augment(rgb, mask))
        self.assertEqual(examples[0][0], 'original')
        self.assertGreaterEqual(len(examples), 7)
        for _, changed, m in examples:
            self.assertEqual(changed.shape[:2], m.shape)
            self.assertEqual(m.dtype, np.bool_)
            self.assertTrue(np.isfinite(changed).all())
            self.assertGreaterEqual(changed.min(), 0)
            self.assertLessEqual(changed.max(), 1)
        np.testing.assert_array_equal(rgb, rgb_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_background_annotations_cannot_use_test_photos(self):
        directory = self.root / 'data/recognition'
        directory.mkdir(parents=True)
        pd.DataFrame([dict(image_id='external', filename='data/test/photo.png',
                           recognition_split='test')]).to_csv(directory / 'dataset.csv', index=False)
        (directory / 'background_regions.json').write_text(json.dumps({'external': {}}))
        with self.assertRaisesRegex(ValueError, 'apprentissage'):
            list(regions._background_examples(self.root))

    def test_background_annotations_reject_changed_image(self):
        directory = self.root / 'data/recognition'
        directory.mkdir(parents=True)
        processed = self.root / 'data/processed'
        processed.mkdir()
        path = processed / 'image.png'
        path.write_bytes(self.path.read_bytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pd.DataFrame([dict(image_id='image', filename='data/processed/image.png', sha256=digest,
                           recognition_split='train', scene_id='scene')]).to_csv(directory / 'dataset.csv', index=False)
        (directory / 'background_regions.json').write_text(json.dumps({'image': {'image_sha256': 'outdated'}}))
        with self.assertRaisesRegex(ValueError, 'modifiée'):
            list(regions._background_examples(self.root))


if __name__ == '__main__':
    unittest.main()
