"""Reconnaissance locale : propositions K-means, rejet du fond et classification.

Le SVM et la forêt de rejet sont appris sur les images du projet. Les photos de
diagnostic et leurs références ne participent pas à l'ajustement des poids.
"""

from collections import Counter
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, measure, morphology
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage.segmentation import watershed
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from recognition import (
    CLASSES, DEFAULT_MODEL, ROOT, _polygon_mask,
    _sample_weights, collect_training_samples, extract_features as legacy_features,
    foreground_mask, load_photo, object_masks,
)

METHOD = 'kmeans_regions_with_background_rejection_v2'
REGION_FEATURE_VERSION = 'color_shape_texture_local_v2'
APPEARANCE_COLUMNS = list(range(13, 61)) + list(range(605, 615))
PARAMETERS = dict(k=6, n_init=5, random_state=42, lab_weights=[0.25, 1, 1],
                  min_saturation=0.15, closing_radius=2, min_area_fraction=0.003,
                  max_area_fraction=0.30, nms_iou=0.45, containment=0.85,
                  split_maxima_fraction=0.30, split_solidity=0.90)


def extract_features(rgb, mask):
    """Alignement corrigé, texture intérieure à taille comparable et contour invariant."""
    base = legacy_features(rgb, mask, correct_alignment=True)
    region = max(measure.regionprops(measure.label(mask)), key=lambda r: r.area)
    y0, x0, y1, x1 = region.bbox
    crop, cmask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ratio = 96 / max(cmask.shape)
    shape = tuple(max(4, round(d * ratio)) for d in cmask.shape)
    scaled = resize(crop, shape, anti_aliasing=True, preserve_range=True)
    smask = resize(cmask.astype(float), shape, order=0, anti_aliasing=False) > 0.5
    inner = morphology.erosion(smask, morphology.disk(2))
    if inner.sum() < 20:
        inner = smask
    gray = np.rint(color.rgb2gray(scaled) * 255).astype(np.uint8)
    extra = []
    for radius, points in ((1, 8), (2, 16)):
        lbp = local_binary_pattern(gray, points, radius, method='uniform')[inner]
        hist, _ = np.histogram(lbp, bins=np.arange(points + 3))
        extra.extend(hist / max(hist.sum(), 1))
    lab = color.rgb2lab(scaled) / 100.0
    for sigma in (1, 2):
        residual = np.abs(lab - ndi.gaussian_filter(lab, sigma=(sigma, sigma, 0)))
        values = residual[inner]
        extra.extend(np.quantile(values, [0.5, 0.9, 0.99], axis=0).ravel())
        extra.extend(values.mean(axis=0))
        extra.extend(values.std(axis=0))
    gradient = np.hypot(ndi.sobel(lab, axis=0), ndi.sobel(lab, axis=1))[inner]
    extra.extend(np.quantile(gradient, [0.5, 0.9, 0.99], axis=0).ravel())
    extra.extend(gradient.mean(axis=0))
    # Le spectre de la distance au centre décrit les ondulations du contour,
    # indépendamment de la rotation et de la position du premier point.
    contours = measure.find_contours(np.pad(cmask.astype(float), 1), 0.5)
    contour = max(contours, key=len)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
    samples = np.linspace(0, distances[-1], 128, endpoint=False)
    boundary = np.c_[np.interp(samples, distances, contour[:, 0]), np.interp(samples, distances, contour[:, 1])]
    center = np.array(region.centroid) - [y0, x0] + 1
    radius = np.linalg.norm(boundary - center, axis=1)
    spectrum = np.abs(np.fft.rfft(radius))
    extra.extend(spectrum[1:13] / max(spectrum[0], 1e-6))
    hsv = color.rgb2hsv(scaled)[inner]
    hist, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=(12, 3), range=((0, 1), (0, 1)))
    extra.extend((hist / max(hist.sum(), 1)).ravel())
    return np.r_[base, extra].astype(np.float32)


def uniform_border(rgb):
    """Le seuil Lab 6 est celui de la localisation historique, sans réglage sur le test."""
    h, w = rgb.shape[:2]
    border = np.ones((h, w), bool)
    thickness = max(1, round(0.02 * min(h, w)))
    border[thickness:-thickness, thickness:-thickness] = False
    lab = color.rgb2lab(rgb)[border]
    distances = np.linalg.norm((lab - np.median(lab, axis=0)) * [0.25, 1, 1], axis=1)
    return bool(np.median(distances) <= 6)


def region_proposals(rgb, parameters=None):
    """Fond uniforme : composantes ; sinon plusieurs groupes de couleur Lab."""
    params = {**PARAMETERS, **(parameters or {})}
    if uniform_border(rgb):
        return object_masks(foreground_mask(rgb)), 'uniform_border_components'
    height, width = rgb.shape[:2]
    features = (color.rgb2lab(rgb) * params['lab_weights']).reshape(-1, 3)
    # Échantillon déterministe pour ajuster les centres, puis affectation de tous
    # les pixels. L'étude comparative K-means conserve son propre calcul.
    rng = np.random.default_rng(params['random_state'])
    indices = rng.choice(len(features), min(20000, len(features)), replace=False)
    k = min(params['k'], len(np.unique(features[indices], axis=0)))
    with threadpool_limits(limits=1):
        model = KMeans(k, n_init=params['n_init'], random_state=params['random_state'])
        model.fit(features[indices])
        labels = model.predict(features).reshape(height, width)
    hsv = color.rgb2hsv(rgb)
    selected = [i for i in range(k) if np.any(labels == i)
                and np.median(hsv[..., 1][labels == i]) >= params['min_saturation']]
    groups = [(i,) for i in selected]
    # Réunir des teintes voisines permet de conserver ombres et reflets d'un
    # même produit. Chaque union fournit une proposition, pas une catégorie.
    for i, first in enumerate(selected):
        for second in selected[i + 1:]:
            a, b = model.cluster_centers_[first, 1:], model.cluster_centers_[second, 1:]
            cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
            if cosine >= 0.85 and np.linalg.norm(a - b) <= 30:
                groups.append((first, second))
    proposals = []
    for group in groups:
        mask = morphology.closing(np.isin(labels, group), morphology.disk(params['closing_radius']))
        components = measure.label(mask)
        for region in measure.regionprops(components):
            component = ndi.binary_fill_holes(components == region.label)
            parts = [component]
            if region.solidity < params['split_solidity']:
                distance = ndi.distance_transform_edt(component)
                maxima = morphology.h_maxima(distance, max(2.0, params['split_maxima_fraction'] * distance.max()))
                markers, count = ndi.label(maxima & component)
                if count > 1:
                    separated = watershed(-distance, markers, mask=component)
                    parts = [separated == i for i in range(1, count + 1)]
            for part in parts:
                area = part.sum()
                if max(30, params['min_area_fraction'] * labels.size) <= area <= params['max_area_fraction'] * labels.size:
                    proposals.append(part)
    return proposals, 'lab_kmeans_region_proposals'


def _augment(rgb, mask):
    """Variations d'éclairage, saturation, netteté et contour ; catégories conservées."""
    yield 'original', rgb, mask
    for factor in (0.8, 1.2):
        yield f'brightness_{factor}', np.clip(rgb * factor, 0, 1), mask
    hsv = color.rgb2hsv(rgb)
    for factor in (0.75, 1.25):
        changed = hsv.copy()
        changed[..., 1] = np.clip(changed[..., 1] * factor, 0, 1)
        yield f'saturation_{factor}', color.hsv2rgb(changed).astype(np.float32), mask
    for name, changed in (('erosion', morphology.erosion(mask, morphology.disk(1))),
                          ('dilation', morphology.dilation(mask, morphology.disk(1)))):
        if changed.sum() >= 20:
            yield name, rgb, changed
    yield 'blur', ndi.gaussian_filter(rgb, sigma=(0.7, 0.7, 0)), mask


def _original_examples(root):
    """Réutiliser la sélection vérifiée des régions et leurs catégories explicites."""
    _, records = collect_training_samples(root)
    manifest = pd.read_csv(root / 'data/recognition/dataset.csv').set_index('image_id')
    for image_id, group in records.groupby('image_id', sort=False):
        row = manifest.loc[image_id]
        if row.recognition_split != 'train':
            raise ValueError('Une image de test a été sélectionnée pour l’apprentissage.')
        path = root / row.filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f'Image modifiée depuis l’inventaire : {image_id}')
        _, rgb = load_photo(path)
        annotation = None
        for record in group.to_dict('records'):
            if pd.isna(record['shape_index']):
                mask = max(object_masks(foreground_mask(rgb)), key=np.count_nonzero)
            else:
                if annotation is None:
                    annotation = json.loads((root / 'data/masks' / f'{image_id}.json').read_text())
                mask = _polygon_mask(annotation['shapes'][int(record['shape_index'])], (rgb.shape[1], rgb.shape[0]))
            yield rgb, mask, record


def _background_examples(root):
    """Régions négatives explicitement annotées ; aucune hypothèse sur les fruits non annotés."""
    manifest = pd.read_csv(root / 'data/recognition/dataset.csv').set_index('image_id')
    entries = json.loads((root / 'data/recognition/background_regions.json').read_text())
    for image_id, entry in entries.items():
        if image_id not in manifest.index or manifest.loc[image_id, 'recognition_split'] != 'train':
            raise ValueError(f'Le fond annoté doit appartenir à l’apprentissage : {image_id}')
        row = manifest.loc[image_id]
        path = root / row.filename
        if not path.resolve().is_relative_to((root / 'data/processed').resolve()):
            raise ValueError('Les annotations négatives doivent provenir des images traitées du projet.')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['image_sha256'] or digest != row.sha256:
            raise ValueError(f'Image de fond modifiée : {image_id}')
        _, rgb = load_photo(path)
        height, width = rgb.shape[:2]
        for index, box in enumerate(entry['boxes_normalized']):
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError(f'Boîte de fond invalide : {image_id}')
            x0, y0, x1, y1 = np.rint(np.array(box) * [width, height, width, height]).astype(int)
            # Fenêtre complète puis quatre sous-fenêtres chevauchantes ; ces
            # variations restent groupées avec leur image et leur scène.
            windows = [(x0, y0, x1, y1)]
            if min(x1 - x0, y1 - y0) >= 30:
                dx, dy = max(1, (x1 - x0) // 3), max(1, (y1 - y0) // 3)
                windows += [(x0, y0, x1 - dx, y1 - dy), (x0 + dx, y0, x1, y1 - dy),
                            (x0, y0 + dy, x1 - dx, y1), (x0 + dx, y0 + dy, x1, y1)]
            for window_index, (a, b, c, d) in enumerate(windows):
                mask = np.zeros((height, width), bool)
                mask[b:d, a:c] = True
                yield rgb, mask, dict(image_id=image_id, scene_id=row.scene_id,
                                     class_name='autre', shape_index=index,
                                     label_source='background_rectangle', image_sha256=digest,
                                     window_index=window_index, split='recognition_train')
    # Les coins des images simples fournissent aussi des fonds blancs/gris.
    # La dilatation du masque exclut les pixels du fruit et son voisinage.
    for rgb, fruit_mask, record in _original_examples(root):
        if not pd.isna(record['shape_index']):
            continue
        height, width = rgb.shape[:2]
        safe = ~morphology.dilation(fruit_mask, morphology.disk(3))
        for index, (top, left) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            y0, y1 = (0, max(1, height // 3)) if not top else (2 * height // 3, height)
            x0, x1 = (0, max(1, width // 3)) if not left else (2 * width // 3, width)
            mask = np.zeros((height, width), bool)
            mask[y0:y1, x0:x1] = safe[y0:y1, x0:x1]
            if mask.sum() >= 30:
                yield rgb, mask, {**record, 'class_name': 'autre',
                                 'label_source': 'simple_background_corner', 'window_index': index}


def _forest(columns, trees=120):
    return Pipeline([
        ('select', ColumnTransformer([('features', 'passthrough', columns)])),
        ('forest', RandomForestClassifier(n_estimators=trees, max_features=1.0,
                                         random_state=42, n_jobs=1, min_samples_leaf=1)),
    ])


def select_background_threshold(probabilities, references, weights, min_recall=None):
    """Choisir le seuil sur les seules prédictions tenues à l'écart par scène."""
    probabilities, references, weights = map(np.asarray, (probabilities, references, weights))
    if not (len(probabilities) == len(references) == len(weights)) or set(references) != {0, 1}:
        raise ValueError('La validation du rejet nécessite les deux catégories et des tableaux alignés.')
    rows = []
    for threshold in sorted(set([0.0, *probabilities.tolist()])):
        keep = probabilities >= threshold
        recall = float(np.average(keep[references == 1], weights=weights[references == 1]))
        specificity = float(np.average(~keep[references == 0], weights=weights[references == 0]))
        rows.append(dict(threshold=threshold, fruit_recall=recall, background_specificity=specificity,
                         balanced_accuracy=(recall + specificity) / 2))
    if min_recall is not None:
        eligible = [row for row in rows if row['fruit_recall'] >= min_recall]
        selected = max(eligible, key=lambda row: (row['background_specificity'], row['threshold']))
    else:
        selected = max(rows, key=lambda row: (row['balanced_accuracy'], -row['threshold']))
    return selected['threshold'], rows


def _make_classifier(name, n_features, labels, groups):
    if name == 'rf_original_compact':
        return _forest(list(range(61)) + list(range(605, 615)), 150)
    columns = list(range(61))
    columns += list(range(605, n_features))
    regularization = float(name.removeprefix('svm_probability_c'))
    return Pipeline([
        ('select', ColumnTransformer([('features', 'passthrough', columns)])),
        ('scale', StandardScaler()),
        ('classifier', SVC(C=regularization, kernel='rbf', probability=True, random_state=42)),
    ])


def _fit_classifier(model, x, labels, records):
    step = 'forest' if 'forest' in model.named_steps else 'classifier'
    # requirements.txt borne la version compatible avec cette API SVC.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated',
                                category=FutureWarning, module='sklearn.svm')
        model.fit(x, labels, **{f'{step}__sample_weight': _sample_weights(records)})
    return model


def _binary_weights(records, labels):
    counts = records.image_id.value_counts()
    weights = np.array([1 / counts[i] for i in records.image_id])
    for label in (0, 1):
        selected = labels == label
        if not selected.any():
            raise ValueError('Les deux catégories fruit/fond sont nécessaires dans chaque apprentissage.')
        weights[selected] /= weights[selected].sum()
    return weights * len(weights) / weights.sum()


@threadpool_limits.wrap(limits=1)
def train_region_classifier(root=ROOT, output=DEFAULT_MODEL, selected_classifier='svm_probability_c1'):
    """Choisir la classification et le seuil de rejet sur des scènes tenues à l'écart."""
    root, output = Path(root), Path(output)
    print('Description des régions avec alignement corrigé et texture normalisée…', flush=True)
    examples = list(_original_examples(root))
    x = np.stack([extract_features(rgb, mask) for rgb, mask, _ in examples])
    records = pd.DataFrame([{**record, 'augmentation': 'original'} for _, _, record in examples])
    labels, groups = records.class_name.to_numpy(), records.scene_id.to_numpy()
    counts = records.image_id.value_counts()
    weights = np.array([1 / counts[i] for i in records.image_id])
    cv_rows, cv_predictions = [], {}
    for name in ('rf_original_compact', 'svm_probability_c0.3', 'svm_probability_c1', 'svm_probability_c3'):
        print(f'Validation entre scènes : {name}…', flush=True)
        predicted = np.empty(len(labels), dtype=object)
        for fit, valid in LeaveOneGroupOut().split(x, labels, groups):
            if set(labels[fit]) != set(CLASSES):
                raise ValueError('Nombre de scènes insuffisant pour valider les trois catégories.')
            model = _make_classifier(name, x.shape[1], labels[fit], groups[fit])
            _fit_classifier(model, x[fit], labels[fit], records.iloc[fit])
            predicted[valid] = model.classes_[model.predict_proba(x[valid]).argmax(axis=1)]
        cv_predictions[name] = predicted
        cv_rows.append(dict(classifier=name,
                            macro_f1_image_weighted=f1_score(labels, predicted, labels=list(CLASSES), average='macro', sample_weight=weights, zero_division=0),
                            accuracy_image_weighted=accuracy_score(labels, predicted, sample_weight=weights),
                            n_scenes=len(np.unique(groups)), n_images=records.image_id.nunique(),
                            n_regions=len(records), augmented_regions=len(records)))
    comparison = pd.DataFrame(cv_rows).sort_values('macro_f1_image_weighted', ascending=False, kind='stable')
    selected = comparison.iloc[0]['classifier'] if selected_classifier == 'auto' else selected_classifier
    if selected not in set(comparison.classifier):
        raise ValueError('Variante de classification inconnue.')
    estimator = _fit_classifier(_make_classifier(selected, x.shape[1], labels, groups), x, labels, records)

    print('Description des exemples de fruit et de fond pour le rejet…', flush=True)
    vectors, binary_rows = [], []
    for source in (iter(examples), _background_examples(root)):
        for rgb, mask, record in source:
            for augmentation, image, changed_mask in _augment(rgb, mask):
                # Le rejet n'utilise que couleur et LBP ; ni HOG ni la silhouette
                # ne sont calculés pour ses milliers de variantes.
                vectors.append(legacy_features(image, changed_mask, appearance_only=True))
                binary_rows.append({**record, 'augmentation': augmentation})
    bx, binary_records = np.stack(vectors), pd.DataFrame(binary_rows)
    by = (binary_records.class_name != 'autre').to_numpy(dtype=int)
    binary_groups = binary_records.scene_id.to_numpy()
    original = binary_records.augmentation.to_numpy() == 'original'
    original_indices = np.flatnonzero(original)
    probabilities = np.empty(len(original_indices))
    bg_model = _forest(APPEARANCE_COLUMNS)
    print('Validation du rejet du fond entre scènes…', flush=True)
    for fit, _ in LeaveOneGroupOut().split(bx, by, binary_groups):
        held_out = ~np.isin(np.arange(len(bx)), fit)
        valid = held_out[original_indices]
        model = clone(bg_model)
        model.fit(bx[fit], by[fit], forest__sample_weight=_binary_weights(binary_records.iloc[fit], by[fit]))
        probabilities[valid] = model.predict_proba(bx[original_indices[valid]])[:, list(model.classes_).index(1)]
    binary_valid = binary_records[original].copy()
    counts = binary_valid.image_id.value_counts()
    weights = np.array([1 / counts[i] for i in binary_valid.image_id])
    threshold, threshold_rows = select_background_threshold(probabilities, by[original], weights)
    bg_model.fit(bx, by, forest__sample_weight=_binary_weights(binary_records, by))
    background_records = binary_records[binary_records.class_name == 'autre']
    classifier_parameters = estimator.named_steps['forest' if selected.startswith('rf_') else 'classifier'].get_params()
    info = dict(feature_version=REGION_FEATURE_VERSION, feature_count=x.shape[1], classes=list(CLASSES),
                class_aliases={'piment': 'poivron'}, classifier=selected,
                localisation=METHOD, localisation_parameters=PARAMETERS,
                classifier_parameters=classifier_parameters,
                training_protocol='external_test_photos', training_split='recognition_train',
                training_image_ids=sorted(binary_records.image_id.unique()),
                training_scene_ids=sorted(binary_records.scene_id.unique()),
                n_regions=len(records), augmented_regions=int((by == 1).sum()),
                classification_uses_augmentation=False, classification_training_regions=len(records),
                background_regions=int(((binary_records.class_name == 'autre') & original).sum()),
                background_classifier_parameters=bg_model.named_steps['forest'].get_params(),
                background_feature_columns=APPEARANCE_COLUMNS,
                min_score=0.45, background_threshold=threshold,
                background_threshold_selection='maximum de l’exactitude équilibrée sur la validation par scènes',
                background_cv=threshold_rows, cv_scores=comparison.to_dict('records'),
                validation='LeaveOneGroupOut par scene_id ; estimation probabiliste interne du SVM dans chaque partition d’apprentissage',
                external_images_used_for_fitting=False,
                external_images_used_for_model_selection=selected_classifier != 'auto',
                model_selection='diagnostic_development_selection' if selected_classifier != 'auto' else 'scene_cross_validation',
                fitting_scope='Apprentissage supervisé des classifieurs ; les centres K-means sont calculés séparément dans chaque photo à l’inférence.',
                versions={name: version(name) for name in ('numpy', 'scikit-learn', 'scikit-image', 'Pillow')},
                dataset_manifest_sha256=hashlib.sha256((root / 'data/recognition/dataset.csv').read_bytes()).hexdigest(),
                background_annotations_sha256=hashlib.sha256((root / 'data/recognition/background_regions.json').read_bytes()).hexdigest(),
                score_interpretation='scores des classifieurs ; estimation probabiliste interne du SVM, fiabilité non établie sur les scènes externes')
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(estimator=estimator, background_estimator=bg_model, **info), output, compress=3)
    output.with_suffix('.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n')
    out = root / 'results/recognition/region_correction'
    out.mkdir(parents=True, exist_ok=True)
    records.to_csv(out / 'training_samples.csv', index=False)
    binary_records[by == 1].to_csv(out / 'augmented_training_samples.csv', index=False)
    background_records.to_csv(out / 'background_training_samples.csv', index=False)
    comparison.to_csv(out / 'model_selection.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(out / 'background_validation.csv', index=False)
    records['predicted_class'] = cv_predictions[selected]
    records.to_csv(out / 'cross_validation_predictions.csv', index=False)
    binary_valid['reference_foreground'] = by[original]
    binary_valid['foreground_score'] = probabilities
    binary_valid.to_csv(out / 'background_cross_validation.csv', index=False)
    return info


def suppress_duplicates(detections, iou_threshold=0.45, containment=0.85):
    """Une boîte par proposition retenue, y compris en cas de catégories concurrentes."""
    kept = []
    ordered = sorted(detections, key=lambda d: d['score'] * d['foreground_score'], reverse=True)
    for item in ordered:
        box = item['bbox']
        area = (box[2] - box[0]) * (box[3] - box[1])
        duplicate = False
        for other in kept:
            b = other['bbox']
            intersection = max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(0, min(box[3], b[3]) - max(box[1], b[1]))
            other_area = (b[2] - b[0]) * (b[3] - b[1])
            union = area + other_area - intersection
            if intersection / max(union, 1) > iou_threshold or intersection / max(min(area, other_area), 1) > containment:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def predict_regions(image_path, bundle, min_score=None, max_side=512):
    threshold = bundle['min_score'] if min_score is None else min_score
    if not 0 <= threshold <= 1:
        raise ValueError('Le seuil de score doit être compris entre 0 et 1.')
    original, rgb = load_photo(image_path, max_side)
    proposals, mode = region_proposals(rgb, bundle['localisation_parameters'])
    if not proposals:
        vectors = np.empty((0, bundle['feature_count']), np.float32)
        probabilities, foreground = np.empty((0, 3)), np.empty(0)
    else:
        extractor = extract_features if bundle.get('feature_version') == REGION_FEATURE_VERSION else legacy_features
        vectors = np.stack([extractor(rgb, mask) for mask in proposals])
        if vectors.shape[1] != bundle['feature_count']:
            raise ValueError('Nombre de caractéristiques incompatible avec le modèle.')
        probabilities = bundle['estimator'].predict_proba(vectors)
        bg_model = bundle['background_estimator']
        foreground = bg_model.predict_proba(vectors[:, :615])[:, list(bg_model.classes_).index(1)]
    classes = bundle['estimator'].classes_
    sx, sy = original.width / rgb.shape[1], original.height / rgb.shape[0]
    candidates, rejected = [], []
    for index, (mask, scores, fg_score) in enumerate(zip(proposals, probabilities, foreground)):
        best = int(scores.argmax())
        yy, xx = np.nonzero(mask)
        item = dict(region_id=index + 1, proposal_index=index, label=str(classes[best]),
                    candidate_label=str(classes[best]), score=float(scores[best]), foreground_score=float(fg_score),
                    class_scores={str(c): float(s) for c, s in zip(classes, scores)},
                    bbox=[int(np.floor(xx.min() * sx)), int(np.floor(yy.min() * sy)),
                          min(original.width, int(np.ceil((xx.max() + 1) * sx))),
                          min(original.height, int(np.ceil((yy.max() + 1) * sy)))],
                    area_processing_pixels=int(mask.sum()))
        reject_background = mode != 'uniform_border_components' and fg_score < bundle['background_threshold']
        reason = 'background' if reject_background else 'uncertain_class' if scores[best] < threshold else None
        if reason:
            rejected.append({**item, 'reason': reason})
        else:
            candidates.append(item)
    params = bundle['localisation_parameters']
    detections = suppress_duplicates(candidates, params['nms_iou'], params['containment'])
    mask = np.zeros(rgb.shape[:2], bool)
    for index, item in enumerate(detections):
        mask |= proposals[item['proposal_index']]
        item['region_id'] = index + 1
    counts = Counter(item['label'] for item in detections)
    result = dict(image=str(Path(image_path)), image_size=list(original.size),
                  processing_size=[rgb.shape[1], rgb.shape[0]], classes_presentes=sorted(counts),
                  regions_par_classe=dict(counts), detections=detections, rejected_regions=rejected,
                  candidate_count=len(proposals), duplicates_removed=len(candidates) - len(detections),
                  min_score=threshold, background_threshold=bundle['background_threshold'],
                  background_rejection_applied=mode != 'uniform_border_components',
                  model_sha256=bundle.get('model_sha256'),
                  localisation=METHOD, localisation_mode=mode,
                  score_interpretation=bundle['score_interpretation'],
                  bbox_convention='[x_min, y_min, x_max, y_max[ dans la photo orientée',
                  scope='méthode classique entraînée localement ; couleurs, contours, texture ; objets inconnus ou cachés non garantis')
    return result, original, mask
