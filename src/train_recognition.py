"""Entraîner le classifieur de produits sur sa partition d'apprentissage."""

import argparse
from pathlib import Path

from recognition import DEFAULT_MODEL, ROOT, train_classifier
from region_recognition import train_region_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Racine du projet')
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL, help='Fichier du modèle')
    parser.add_argument('--method', choices=('regions', 'legacy'), default='regions',
                        help='Régions K-means avec rejet du fond, ou ancien prototype')
    parser.add_argument('--classifier', default='svm_probability_c1',
                        choices=('svm_probability_c1', 'svm_probability_c0.3', 'svm_probability_c3', 'rf_original_compact', 'auto'),
                        help='Variante du prototype révisé ; auto choisit le meilleur F1 entre scènes')
    args = parser.parse_args()
    trainer = train_region_classifier if args.method == 'regions' else train_classifier
    info = trainer(args.root, args.output, args.classifier) if args.method == 'regions' else trainer(args.root, args.output)
    print(f"Modèle : {args.output}")
    print(f"Classifieur : {info['classifier']}")
    print(f"Apprentissage : {len(info['training_image_ids'])} images, {info['n_regions']} régions ; protocole = {info['training_protocol']}")
    print('Catégories : pomme, banane, poivron (piments inclus)')
    for row in info['cv_scores']:
        print(f"Validation entre scènes — {row['classifier']} : F1 macro pondéré = {row['macro_f1_image_weighted']:.4f}")


if __name__ == '__main__':
    main()
