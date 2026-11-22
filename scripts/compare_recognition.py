"""Comparer les sorties historiques et révisées sur les deux photos de diagnostic."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    before = ROOT / 'results/recognition/before_region_correction/user_tests'
    after = ROOT / 'results/recognition/user_tests'
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout='constrained')
    for row, (image_id, scene) in enumerate((('test_img001', 'fond simple'), ('test_img002', 'scène complexe'))):
        for column, (directory, name) in enumerate(((before, 'Prototype initial'), (after, 'Version révisée'))):
            with Image.open(directory / f'{image_id}_recognition.png') as image:
                axes[row, column].imshow(image)
            axes[row, column].set_title(f'{name} — {scene}', fontsize=12)
            axes[row, column].axis('off')
    fig.suptitle('Comparaison des méthodes de reconnaissance', fontsize=16)
    output = ROOT / 'results/recognition/region_correction/comparison.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
