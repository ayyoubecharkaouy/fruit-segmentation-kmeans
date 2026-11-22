from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentation import reconstruct_rgb


def create_color_space_comparison(image_id, k=3, seed=42):
    # Comparer les reconstructions sans relancer K-means.
    project_root = Path(__file__).resolve().parents[1]
    with Image.open(project_root / 'data' / 'processed' / f'{image_id}.png') as image:
        rgb = np.array(image.convert('RGB')).astype(np.float32) / 255.0

    spaces = [('RGB', 'RGB'), ('HSV_circulaire', 'HSV circulaire'), ('Lab', 'Lab')]
    reconstructions = []
    for space_name, title in spaces:
        label_path = (
            project_root / 'results' / 'labels'
            / f'{image_id}_{space_name}_k{k}_seed{seed}_labels.npy'
        )
        label_map = np.load(label_path, allow_pickle=False)
        if label_map.shape != rgb.shape[:2]:
            raise ValueError(f'Dimensions des groupes différentes de la photo : {label_path}')
        reconstructed, _ = reconstruct_rgb(rgb, label_map.reshape(-1), k)
        reconstructions.append((title, reconstructed))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    axes[0].imshow(rgb, interpolation='nearest')
    axes[0].set_title('Photo originale')
    for axis, (title, reconstructed) in zip(axes[1:], reconstructions):
        axis.imshow(reconstructed, interpolation='nearest')
        axis.set_title(title)
    for axis in axes:
        axis.axis('off')

    fig.suptitle(f'{image_id} — comparaison des espaces couleur — K = {k}, graine {seed}')
    fig.text(
        0.5, 0.04,
        'Chaque groupe est représenté par la moyenne RGB de ses pixels.',
        ha='center', fontsize=10
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))

    output_path = (
        project_root / 'results' / 'figures'
        / f'{image_id}_comparaison_espaces_k{k}_seed{seed}.png'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_k_comparison(image_id, space_name='Lab', seed=42):
    """Comparer K = 3, 4, 5 et 6 dans un même espace couleur."""
    project_root = Path(__file__).resolve().parents[1]
    with Image.open(project_root / 'data' / 'processed' / f'{image_id}.png') as image:
        rgb = np.array(image.convert('RGB')).astype(np.float32) / 255.0

    reconstructions = []
    for k in (3, 4, 5, 6):
        label_path = (
            project_root / 'results' / 'labels'
            / f'{image_id}_{space_name}_k{k}_seed{seed}_labels.npy'
        )
        label_map = np.load(label_path, allow_pickle=False)
        if label_map.shape != rgb.shape[:2]:
            raise ValueError(f'Dimensions des groupes différentes de la photo : {label_path}')
        reconstructed, _ = reconstruct_rgb(rgb, label_map.reshape(-1), k)
        reconstructions.append((k, reconstructed))

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2))
    axes[0].imshow(rgb, interpolation='nearest')
    axes[0].set_title('Photo originale')
    for axis, (k, reconstructed) in zip(axes[1:], reconstructions):
        axis.imshow(reconstructed, interpolation='nearest')
        axis.set_title(f'K = {k}')
    for axis in axes:
        axis.axis('off')

    space_title = space_name.replace('_', ' ')
    fig.suptitle(f'{image_id} — effet de K en {space_title} — graine {seed}')
    fig.text(
        0.5, 0.04,
        'Chaque groupe est représenté par la moyenne RGB de ses pixels.',
        ha='center', fontsize=10
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))

    output_path = (
        project_root / 'results' / 'figures'
        / f'{image_id}_{space_name}_comparaison_k3_a_k6_seed{seed}.png'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_mask_comparison(image_id, space_name='Lab', k=3, seed=42):
    """Illustrer la référence et le masque issu du groupe de fond majoritaire."""
    project_root = Path(__file__).resolve().parents[1]
    with Image.open(project_root / 'data' / 'processed' / f'{image_id}.png') as image:
        original = np.array(image.convert('RGB'))
    with Image.open(project_root / 'data' / 'masks' / f'{image_id}_mask.png') as image:
        reference_mask = np.array(image) > 0

    label_path = (
        project_root / 'results' / 'labels'
        / f'{image_id}_{space_name}_k{k}_seed{seed}_labels.npy'
    )
    label_map = np.load(label_path, allow_pickle=False)
    if label_map.shape != original.shape[:2] or reference_mask.shape != label_map.shape:
        raise ValueError(f'Dimensions des images et des masques différentes : {image_id}')

    # Même règle que dans evaluation.py : contour de 2 %, un groupe de fond.
    height, width = label_map.shape
    border_width = max(1, round(0.02 * min(height, width)))
    border_mask = np.zeros_like(label_map, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    background_cluster = np.argmax(np.bincount(label_map[border_mask]))
    predicted_mask = label_map != background_cluster

    intersection = (reference_mask & predicted_mask).sum()
    union = (reference_mask | predicted_mask).sum()
    iou = intersection / union if union > 0 else np.nan
    dice = 2 * intersection / (intersection + union) if intersection + union > 0 else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    axes[0].imshow(original, interpolation='nearest')
    axes[0].set_title('Photo originale')
    for axis, mask, title in zip(
        axes[1:],
        (reference_mask, predicted_mask),
        ('Masque de référence', 'Masque prédit')
    ):
        axis.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axis.set_title(title)
    for axis in axes:
        axis.axis('off')

    space_title = space_name.replace('_', ' ')
    fig.suptitle(f'{image_id} — segmentation en {space_title} — K = {k}, graine {seed}')
    fig.text(
        0.5, 0.04,
        f"Blanc : produit d'intérêt ; noir : fond.   IoU = {iou:.4f} ; Dice = {dice:.4f}",
        ha='center', fontsize=10
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))

    output_path = (
        project_root / 'results' / 'figures'
        / f'{image_id}_{space_name}_k{k}_comparaison_masques.png'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    print(create_color_space_comparison('img001', k=3, seed=42))
    print(create_color_space_comparison('img055', k=3, seed=42))
    print(create_k_comparison('img055', space_name='Lab', seed=42))
    print(create_mask_comparison('img001', space_name='RGB', k=3, seed=42))
    print(create_mask_comparison('img032', space_name='Lab', k=3, seed=42))
    print(create_mask_comparison('img055', space_name='Lab', k=3, seed=42))
