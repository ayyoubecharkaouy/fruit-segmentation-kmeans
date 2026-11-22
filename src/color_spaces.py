import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2hsv


def prepare_color_features(rgb):
    rgb_features = rgb.reshape(-1, 3)  # Reshape en un tableau 2D de pixels

    hsv = rgb2hsv(rgb)
    h = hsv[:, :, 0].reshape(-1, 1)  # extraire la composante H
    s = hsv[:, :, 1].reshape(-1, 1)  # extraire la composante S
    v = hsv[:, :, 2].reshape(-1, 1)  # extraire la composante V

    # representation de H par angle
    theta = h * 2 * np.pi  # Convertir H en radians
    hsv_x = s * np.cos(theta)  # Composante x de H
    hsv_y = s * np.sin(theta)  # Composante y de H

    hsv_features = np.hstack((hsv_x, hsv_y, v))  # Combiner les composantes H, S et V

    # convertir rgb en lab
    lab = rgb2lab(rgb, illuminant='D65', observer='2')
    lab_scaled = lab / 100.0  # Normaliser les valeurs de L, a et b

    lab_features = lab_scaled.reshape(-1, 3)  # Reshape en un tableau 2D de pixels

    return rgb_features, hsv_features, lab_features

if __name__ == '__main__':
    image = Image.open("data/processed/img001.png")
    rgb = np.array(image).astype(np.float32) / 255.0  # Normaliser les valeurs des pixels entre 0 et 1
    rgb_features, hsv_features, lab_features = prepare_color_features(rgb)
