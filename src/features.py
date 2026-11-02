import numpy as np
from skimage.feature import hog


def extract_hog_features(images):
    """
    Görüntülerden HOG feature çıkarır.
    Pattern Recognition projesinde feature sabit tutulur.
    """
    hog_features = []

    for image in images:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        hog_features.append(features)

    return np.array(hog_features)