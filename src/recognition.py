"""Chargement, descripteur historique et API de reconnaissance des produits.

Cette extension complète l'étude K-means ; elle ne modifie pas ses expériences.
Les catégories sont apprises à partir de la partition de reconnaissance, avec
des photos de diagnostic séparées des poids d'apprentissage. La version révisée
est définie dans region_recognition.py ; les modèles v1 restent compatibles.
"""

from collections import Counter
from importlib.metadata import version
from pathlib import Path
import hashlib
import io
import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi
from skimage import color, measure, morphology, transform
from skimage.feature import hog, local_binary_pattern
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('banane', 'poivron', 'pomme')
FEATURE_VERSION = 'color_shape_texture_v1'
DEFAULT_MODEL = ROOT / 'models' / 'fruit_classifier.joblib'
DEFAULT_MIN_SCORE = 0.45
SCORE_INTERPRETATION = 'moyenne des scores des arbres, non calibrée'


def canonical_class(value):
    """Les piments appartiennent à la catégorie poivron de cette expérience."""
    return 'poivron' if value == 'piment' else value


def load_photo(path, max_side=512):
    """Retourner la photo sRGB orientée et sa version de calcul, sans agrandissement."""
    if max_side < 1:
        raise ValueError('La taille maximale doit être positive.')
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        icc = image.info.get('icc_profile')
        if 'A' in image.getbands() or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(background, rgba).convert('RGB')
        else:
            image = image.convert('RGB')
        if icc:
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile('sRGB'),
                outputMode='RGB',
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
        original = image.copy()
    prepared = original.copy()
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return original, np.asarray(prepared, dtype=np.float32) / 255.0


def _validate_arrays(rgb, mask=None):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError('Une image RGB non vide de forme H × W × 3 est nécessaire.')
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError('Les valeurs RGB doivent être finies et comprises entre 0 et 1.')
    if mask is not None:
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError('Le masque doit être booléen et aligné sur la photo.')
        if not mask.any():
            raise ValueError('Le masque du produit ne peut pas être vide.')


def foreground_mask(rgb, min_area_fraction=0.002):
    """Localisation adaptée aux fonds unis, avec tolérance aux ombres achromatiques.

    Le fond est estimé par la médiane Lab du contour. Sa variation fixe un seuil
    robuste ; L* reçoit un poids 0,25. Cette règle est propre à la reconnaissance,
    distincte de la règle K-means à un groupe de fond de l'étude comparative.
    """
    _validate_arrays(rgb)
    if not 0 < min_area_fraction < 1:
        raise ValueError('La fraction minimale de surface doit être comprise entre 0 et 1.')
    lab = color.rgb2lab(rgb)
    height, width = rgb.shape[:2]
    border_width = max(1, round(0.02 * min(height, width)))
    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = border[-border_width:, :] = True
    border[:, :border_width] = border[:, -border_width:] = True
    center = np.median(lab[border], axis=0)
    distances = np.linalg.norm((lab - center) * np.array([0.25, 1, 1]), axis=-1)
    samples = distances[border]
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))
    threshold = max(6.0, float(median + 3 * 1.4826 * mad))
    mask = morphology.closing(distances > threshold, morphology.disk(1))
    mask = ndi.binary_fill_holes(mask)
    labels, _ = ndi.label(mask)
    counts = np.bincount(labels.ravel())
    keep = counts >= max(30, int(min_area_fraction * height * width))
    keep[0] = False
    return keep[labels]


def object_masks(mask):
    """Un masque par composante connexe, ordonné de gauche à droite."""
    labels = measure.label(mask, connectivity=2)
    regions = sorted(measure.regionprops(labels), key=lambda region: region.bbox[1])
    return [labels == region.label for region in regions]


def extract_features(rgb, mask, correct_alignment=False, appearance_only=False):
    """Décrire une région ; conserver l'alignement historique pour les modèles v1.

    La reconnaissance révisée utilise correct_alignment=True. Le mode
    appearance_only conserve les positions des 58 caractéristiques couleur/LBP.
    """
    _validate_arrays(rgb, mask)
    region = max(measure.regionprops(measure.label(mask)), key=lambda item: item.area)
    y0, x0, y1, x1 = region.bbox
    crop = rgb[y0:y1, x0:x1]
    small_mask = mask[y0:y1, x0:x1]
    pixels = crop[small_mask]
    hsv = color.rgb2hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = color.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3) / 100.0

    colour = []
    for values in (pixels, lab):
        colour.extend(np.quantile(values, [0.1, 0.5, 0.9], axis=0).ravel())
        colour.extend(values.std(axis=0))
    for values, bins in ((hsv[:, 0], 12), (hsv[:, 1], 6), (hsv[:, 2], 6)):
        histogram, _ = np.histogram(values, bins=bins, range=(0, 1))
        colour.extend(histogram / len(values))

    crop_gray = np.rint(color.rgb2gray(crop) * 255).astype(np.uint8)
    lbp = local_binary_pattern(crop_gray, 8, 1, method='uniform')[small_mask]
    texture, _ = np.histogram(lbp, bins=np.arange(11), density=False)
    texture = texture / max(texture.sum(), 1)
    if appearance_only:
        vector = np.zeros(615, np.float32)
        vector[13:61], vector[605:615] = colour, texture
        return vector

    shape = [
        region.eccentricity, region.solidity, region.extent,
        region.axis_minor_length / max(region.axis_major_length, 1),
        4 * np.pi * region.area / max(region.perimeter_crofton ** 2, 1),
        (region.area_filled - region.area) / region.area,
    ]
    hu = region.moments_hu
    shape.extend(np.sign(hu) * np.log10(np.abs(hu) + 1e-12) / 12)

    # Alignement du grand axe et conservation des proportions dans un carré.
    angle = np.degrees(region.orientation) * (-1 if correct_alignment else 1)
    oriented_mask = transform.rotate(small_mask.astype(float), angle, resize=True, order=0) > 0.5
    oriented_image = transform.rotate(
        np.where(small_mask[..., None], crop, 1), angle, resize=True, cval=1
    )
    yy, xx = np.nonzero(oriented_mask)
    if len(yy) == 0:
        raise ValueError('Région trop petite après alignement.')
    slices = np.s_[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    oriented_mask = oriented_mask[slices]
    oriented_image = oriented_image[slices]
    height, width = oriented_mask.shape
    side = max(height, width) + 4
    square = np.zeros((side, side), dtype=float)
    canvas = np.ones((side, side, 3), dtype=float)
    y, x = (side - height) // 2, (side - width) // 2
    square[y:y + height, x:x + width] = oriented_mask
    canvas[y:y + height, x:x + width] = oriented_image
    silhouette = transform.resize(square, (16, 16), anti_aliasing=True).ravel()
    gray = color.rgb2gray(transform.resize(canvas, (64, 64), anti_aliasing=True))
    gradients = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    vector = np.r_[shape, colour, silhouette, gradients, texture]
    return np.nan_to_num(vector, nan=0, posinf=0, neginf=0).astype(np.float32)


def _polygon_mask(shape, size):
    canvas = Image.new('L', size, 0)
    if shape.get('shape_type', 'polygon') != 'polygon':
        raise ValueError('Les annotations de reconnaissance doivent être des polygones.')
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in shape['points']], fill=255)
    return np.array(canvas) > 0


def collect_training_samples(root=ROOT):
    """Sélectionner les objets d'apprentissage, sans ouvrir les photos du test."""
    root = Path(root)
    recognition_manifest = root / 'data/recognition/dataset.csv'
    external_protocol = recognition_manifest.exists()
    metadata = pd.read_csv(recognition_manifest if external_protocol else root / 'data/metadata.csv')
    if metadata['image_id'].duplicated().any():
        raise ValueError('Les identifiants des métadonnées doivent être uniques.')
    if external_protocol:
        if not set(metadata['recognition_split']) <= {'train', 'test'}:
            raise ValueError('Partition de reconnaissance inconnue.')
        train_hashes = set(metadata.loc[metadata.recognition_split == 'train', 'sha256'].dropna())
        test_hashes = set(metadata.loc[metadata.recognition_split == 'test', 'sha256'].dropna())
        if train_hashes & test_hashes:
            raise ValueError('Une image de test est identique à une image d’apprentissage.')
    split_column = 'recognition_split' if external_protocol else 'split'
    training_value = 'train' if external_protocol else 'mise_au_point'
    crossing = metadata.groupby('scene_id')[split_column].nunique()
    if (crossing > 1).any():
        raise ValueError('Une scène ne peut pas traverser les partitions.')
    extra_path = root / 'data/recognition/instance_classes.json'
    explicit = json.loads(extra_path.read_text()) if extra_path.exists() else {}
    records, vectors = [], []

    for row in metadata[metadata[split_column] == training_value].itertuples(index=False):
        label = canonical_class(row.fruit_types)
        simple = row.source in ('fruits360_simple', 'kaggle_chili') and label in CLASSES
        annotation_path = root / 'data/masks' / f'{row.image_id}.json'
        labelled_polygons = annotation_path.exists() and (label in CLASSES or row.image_id in explicit)
        if not simple and not labelled_polygons:
            continue
        photo_path = root / row.filename if external_protocol else root / 'data/processed' / f'{row.image_id}.png'
        if not photo_path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Le chemin des images doit rester dans le projet.')
        _, rgb = load_photo(photo_path)
        if simple:
            masks = object_masks(foreground_mask(rgb))
            if not masks:
                raise ValueError(f'Aucune région détectée pour {row.image_id}.')
            examples = [(max(masks, key=np.count_nonzero), label, None, 'categorie_image')]
        else:
            annotation = json.loads(annotation_path.read_text())
            if (annotation['imageHeight'], annotation['imageWidth']) != rgb.shape[:2]:
                raise ValueError(f'Annotation non alignée pour {row.image_id}.')
            classes = [label] * len(annotation['shapes'])
            if row.image_id in explicit:
                entry = explicit[row.image_id]
                if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != entry['annotation_sha256']:
                    raise ValueError(f'Annotations modifiées : vérifier les catégories de {row.image_id}.')
                classes = entry['shape_classes']
                if len(classes) != len(annotation['shapes']):
                    raise ValueError(f'Nombre de catégories incorrect pour {row.image_id}.')
            examples = []
            for index, (shape, category) in enumerate(zip(annotation['shapes'], classes)):
                if category not in CLASSES:
                    raise ValueError(f'Catégorie d’instance inconnue : {category}')
                mask = _polygon_mask(shape, (rgb.shape[1], rgb.shape[0]))
                yy, xx = np.nonzero(mask)
                if len(yy) < 150 or np.ptp(yy) < 8 or np.ptp(xx) < 8:
                    continue
                examples.append((mask, category, index, 'polygone_categorie'))
        for mask, category, index, source in examples:
            vectors.append(extract_features(rgb, mask))
            records.append({
                'image_id': row.image_id, 'class_name': category,
                'scene_id': row.scene_id,
                'split': 'recognition_train' if external_protocol else row.split,
                'segmentation_split': row.split,
                'shape_index': index, 'label_source': source,
                'image_sha256': hashlib.sha256(photo_path.read_bytes()).hexdigest(),
            })
    if not records or set(record['class_name'] for record in records) != set(CLASSES):
        raise ValueError('Les données d’apprentissage doivent couvrir les trois catégories.')
    return np.stack(vectors), pd.DataFrame(records)


def _sample_weights(records):
    """Équilibrer les images puis les classes, malgré les nombreux objets d'un étal."""
    counts = records['image_id'].value_counts()
    weights = np.array([1 / counts[image_id] for image_id in records['image_id']])
    classes = records['class_name'].to_numpy()
    for category in CLASSES:
        selected = classes == category
        if selected.any():
            weights[selected] /= weights[selected].sum()
    return weights * len(weights) / weights.sum()


def train_classifier(root=ROOT, output=DEFAULT_MODEL):
    """Choisir le classifieur par validation entre scènes, puis ajuster le modèle."""
    root, output = Path(root), Path(output)
    x, records = collect_training_samples(root)
    y = records['class_name'].to_numpy()
    groups = records['scene_id'].to_numpy()
    # 13 caractéristiques géométriques, 48 de couleur, silhouette/HOG, puis 10 LBP.
    families = {
        'all': list(range(x.shape[1])),
        'compact': list(range(61)) + list(range(x.shape[1] - 10, x.shape[1])),
        'shape_colour': list(range(61)),
        'geometry_texture': list(range(13)) + list(range(x.shape[1] - 10, x.shape[1])),
    }
    configurations = {
        'random_forest': ('all', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
        'extra_trees': ('all', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=1)),
        'random_forest_compact': ('compact', RandomForestClassifier(n_estimators=150, max_features=1.0, random_state=42, n_jobs=1)),
        'random_forest_shape_colour': ('shape_colour', RandomForestClassifier(n_estimators=150, max_features=1.0, random_state=42, n_jobs=1)),
        'random_forest_geometry_texture': ('geometry_texture', RandomForestClassifier(n_estimators=150, max_features=1.0, random_state=42, n_jobs=1)),
    }
    candidates = {
        name: Pipeline([
            ('select', ColumnTransformer([('features', 'passthrough', families[family])])),
            ('forest', classifier),
        ])
        for name, (family, classifier) in configurations.items()
    }
    cv_rows, predictions = [], {}
    image_counts = records['image_id'].value_counts()
    validation_weights = np.array([1 / image_counts[image_id] for image_id in records['image_id']])
    for name, candidate in candidates.items():
        predicted = np.empty(len(y), dtype=object)
        for fit, valid in LeaveOneGroupOut().split(x, y, groups):
            if set(y[fit]) != set(CLASSES):
                raise ValueError('Nombre de scènes insuffisant pour valider les trois catégories.')
            estimator = clone(candidate)
            estimator.fit(x[fit], y[fit], forest__sample_weight=_sample_weights(records.iloc[fit]))
            predicted[valid] = estimator.predict(x[valid])
        predictions[name] = predicted
        cv_rows.append({
            'classifier': name,
            'feature_family': configurations[name][0],
            'macro_f1_image_weighted': f1_score(y, predicted, labels=list(CLASSES), average='macro', sample_weight=validation_weights, zero_division=0),
            'accuracy_image_weighted': accuracy_score(y, predicted, sample_weight=validation_weights),
            'n_scenes': len(np.unique(groups)), 'n_images': records['image_id'].nunique(),
            'n_regions': len(records),
        })
    comparison = pd.DataFrame(cv_rows).sort_values('macro_f1_image_weighted', ascending=False, kind='stable')
    selected = comparison.iloc[0]['classifier']
    estimator = candidates[selected]
    estimator.fit(x, y, forest__sample_weight=_sample_weights(records))
    info = {
        'feature_version': FEATURE_VERSION, 'feature_count': x.shape[1],
        'selected_feature_count': len(families[configurations[selected][0]]),
        'classes': list(CLASSES), 'class_aliases': {'piment': 'poivron'},
        'classifier': selected, 'feature_family': configurations[selected][0], 'random_state': 42,
        'classifier_parameters': estimator.named_steps['forest'].get_params(),
        'localisation_parameters': {'max_side': 512, 'border_fraction': 0.02,
                                    'lab_weights': [0.25, 1, 1], 'min_distance': 6.0,
                                    'mad_factor': 3 * 1.4826, 'min_area_fraction': 0.002},
        'training_split': 'recognition_train' if (root / 'data/recognition/dataset.csv').exists() else 'mise_au_point',
        'training_protocol': 'external_test_photos' if (root / 'data/recognition/dataset.csv').exists() else 'segmentation_split',
        'dataset_manifest_sha256': hashlib.sha256((root / 'data/recognition/dataset.csv').read_bytes()).hexdigest()
        if (root / 'data/recognition/dataset.csv').exists() else None,
        'training_image_ids': sorted(records['image_id'].unique()),
        'training_scene_ids': sorted(records['scene_id'].unique()),
        'n_regions': len(records), 'min_score': DEFAULT_MIN_SCORE,
        'localisation': 'adaptive_border_lab',
        'score_interpretation': SCORE_INTERPRETATION,
        'validation': 'LeaveOneGroupOut par scene_id ; pondération par image',
        'cv_scores': comparison.to_dict(orient='records'),
        'versions': {name: version(name) for name in ('numpy', 'scikit-learn', 'scikit-image', 'Pillow')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'estimator': estimator, **info}, output, compress=3)
    output.with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    report_dir = root / 'results/recognition'
    report_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(report_dir / 'training_samples.csv', index=False)
    comparison.to_csv(report_dir / 'model_selection.csv', index=False)
    cv_predictions = records.copy()
    cv_predictions['predicted_class'] = predictions[selected]
    cv_predictions.to_csv(report_dir / 'cross_validation_predictions.csv', index=False)
    return info


def load_classifier(path=DEFAULT_MODEL):
    if not Path(path).is_file():
        raise FileNotFoundError('Modèle absent. Exécuter : python src/train_recognition.py')
    bundle = joblib.load(path)
    if bundle.get('feature_version') not in (FEATURE_VERSION, 'color_shape_texture_local_v2') or set(bundle.get('classes', [])) != set(CLASSES):
        raise ValueError('Modèle incompatible avec la version des caractéristiques.')
    if bundle.get('localisation') == 'kmeans_regions_with_background_rejection_v2':
        required = {'background_estimator', 'background_threshold', 'localisation_parameters'}
        if not required <= bundle.keys() or not 0 <= bundle['background_threshold'] <= 1:
            raise ValueError('Modèle de rejet du fond absent ou incompatible.')
        bundle['model_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return bundle


def predict_photo(image_path, bundle, min_score=None, max_side=512):
    """Reconnaître chaque région sans consulter les noms de fichiers ou les annotations."""
    if bundle.get('localisation') == 'kmeans_regions_with_background_rejection_v2':
        from region_recognition import predict_regions
        return predict_regions(image_path, bundle, min_score, max_side)
    threshold = bundle.get('min_score', DEFAULT_MIN_SCORE) if min_score is None else min_score
    if not 0 <= threshold <= 1:
        raise ValueError('Le seuil de score doit être compris entre 0 et 1.')
    original, rgb = load_photo(image_path, max_side=max_side)
    mask = foreground_mask(rgb)
    scale_x, scale_y = original.width / rgb.shape[1], original.height / rgb.shape[0]
    detections = []
    for component in object_masks(mask):
        vector = extract_features(rgb, component)
        if len(vector) != bundle['feature_count']:
            raise ValueError('Nombre de caractéristiques incompatible avec le modèle.')
        scores = bundle['estimator'].predict_proba(vector[None, :])[0]
        classes = bundle['estimator'].classes_
        best = int(np.argmax(scores))
        category, score = str(classes[best]), float(scores[best])
        yy, xx = np.nonzero(component)
        detections.append({
            'region_id': len(detections) + 1,
            'label': category if score >= threshold else 'indetermine',
            'candidate_label': category, 'score': score,
            'class_scores': {str(name): float(value) for name, value in zip(classes, scores)},
            'bbox': [int(np.floor(xx.min() * scale_x)), int(np.floor(yy.min() * scale_y)),
                     min(original.width, int(np.ceil((xx.max() + 1) * scale_x))),
                     min(original.height, int(np.ceil((yy.max() + 1) * scale_y)))],
            'area_processing_pixels': int(component.sum()),
        })
    counts = Counter(item['label'] for item in detections if item['label'] in CLASSES)
    result = {
        'image': str(Path(image_path)), 'image_size': list(original.size),
        'processing_size': [rgb.shape[1], rgb.shape[0]],
        'classes_presentes': sorted(counts), 'regions_par_classe': dict(counts),
        'detections': detections, 'min_score': threshold,
        'localisation': 'adaptive_border_lab',
        'score_interpretation': SCORE_INTERPRETATION,
        'bbox_convention': '[x_min, y_min, x_max, y_max[ dans la photo orientée',
        'scope': 'fonds relativement uniformes ; objets séparés ; objets en contact potentiellement fusionnés',
    }
    return result, original, mask


def save_prediction(result, original, mask, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result['image']).stem
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(12, min(22, min(original.size) // 55)))
    except OSError:
        font = ImageFont.load_default()
    colours = {'pomme': '#e34b4b', 'banane': '#e6aa00', 'poivron': '#18a070', 'indetermine': '#666666'}
    label_boxes = []
    # Tracer toutes les boîtes avant les textes pour qu'une boîte suivante
    # ne traverse pas l'étiquette déjà posée d'un autre objet.
    for detection in result['detections']:
        x0, y0, x1, y1 = detection['bbox']
        ink = colours[detection['label']]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=ink, width=max(2, original.width // 350))
    for detection in result['detections']:
        x0, y0, x1, y1 = detection['bbox']
        ink = colours[detection['label']]
        text = f"{detection['label']} ({detection['score']:.2f})"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8
        tx = max(0, min(x0, original.width - text_width))
        candidates = [max(0, min(original.height - text_height, y))
                      for y in (y0 - text_height, y0 + 2, y1 + 2)]
        def overlap(y):
            return sum(max(0, min(tx + text_width, bx1) - max(tx, bx0))
                       * max(0, min(y + text_height, by1) - max(y, by0))
                       for bx0, by0, bx1, by1 in label_boxes)
        ty = min(candidates, key=overlap)
        label_boxes.append((tx, ty, tx + text_width, ty + text_height))
        draw.rectangle((tx, ty, min(original.width - 1, tx + text_width), ty + text_height), fill=ink)
        draw.text((tx + 4, ty + 4 - bounds[1]), text, fill='white', font=font)
    image_path = output_dir / f'{stem}_recognition.png'
    json_path = output_dir / f'{stem}_recognition.json'
    mask_path = output_dir / f'{stem}_foreground.png'
    annotated.save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return image_path, json_path
