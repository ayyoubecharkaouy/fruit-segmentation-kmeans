import json
from PIL import Image, ImageDraw

def create_mask_from_annotation(json_path, mask_path):
    annotation = json.load(open(json_path))
    imageHeight, imageWidth = annotation['imageHeight'], annotation['imageWidth']

    mask = Image.new('L', (imageWidth, imageHeight), 0)
    draw = ImageDraw.Draw(mask)

    for shape in annotation['shapes']:
        points = shape['points']
        draw.polygon(points, outline=255, fill=255)

    mask.save(mask_path)

# recuprer les images depuis fichier data/annotation_selection.txt
with open('data/annotation_selection.txt', 'r') as f:
    annotation_files = [line.strip() for line in f.readlines()]

for image_id in annotation_files:
    json_path = f'data/masks/{image_id}.json'
    mask_path = f'data/masks/{image_id}_mask.png'
    create_mask_from_annotation(json_path, mask_path)
