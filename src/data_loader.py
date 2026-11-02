import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist, cifar10

from config import (
    TRAIN_LIMIT,
    TEST_LIMIT,
    RAW_DATA_DIR,
    RANDOM_STATE,
)


FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

KTH_TIPS_IMAGE_SIZE = (128, 128)
KTH_TIPS_TEST_SIZE = 0.2
KTH_TIPS_FOLDER_NAME = "KTH_TIPS"


def apply_limit(x_train, y_train, x_test, y_test):
    if TRAIN_LIMIT is not None:
        x_train = x_train[:TRAIN_LIMIT]
        y_train = y_train[:TRAIN_LIMIT]

    if TEST_LIMIT is not None:
        x_test = x_test[:TEST_LIMIT]
        y_test = y_test[:TEST_LIMIT]

    return x_train, y_train, x_test, y_test


def normalize_images(x_train, x_test):
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, x_test


def rgb_to_grayscale(images):
    return (
        0.299 * images[:, :, :, 0]
        + 0.587 * images[:, :, :, 1]
        + 0.114 * images[:, :, :, 2]
    )


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train, y_train, x_test, y_test = apply_limit(
        x_train, y_train, x_test, y_test
    )

    x_train, x_test = normalize_images(x_train, x_test)

    return x_train, y_train, x_test, y_test, FASHION_MNIST_CLASS_NAMES


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train, y_train, x_test, y_test = apply_limit(
        x_train, y_train, x_test, y_test
    )

    x_train, x_test = normalize_images(x_train, x_test)

    x_train = rgb_to_grayscale(x_train)
    x_test = rgb_to_grayscale(x_test)

    return x_train, y_train, x_test, y_test, CIFAR10_CLASS_NAMES


def load_kth_tips():
    kth_dir = RAW_DATA_DIR / KTH_TIPS_FOLDER_NAME

    if not kth_dir.exists():
        raise FileNotFoundError(f"KTH-TIPS folder not found: {kth_dir}")

    class_names = sorted(
        [
            folder.name
            for folder in kth_dir.iterdir()
            if folder.is_dir()
        ]
    )

    images = []
    labels = []

    for label_id, class_name in enumerate(class_names):
        class_dir = kth_dir / class_name

        image_paths = (
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.bmp"))
        )

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, KTH_TIPS_IMAGE_SIZE)
            image = image.astype(np.float32) / 255.0

            images.append(image)
            labels.append(label_id)

    images = np.array(images)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=KTH_TIPS_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return x_train, y_train, x_test, y_test, class_names


def load_dataset(dataset_name):
    if dataset_name == "fashion_mnist":
        return load_fashion_mnist()

    if dataset_name == "cifar10":
        return load_cifar10()

    if dataset_name == "kth_tips":
        return load_kth_tips()

    raise ValueError(f"Unknown dataset name: {dataset_name}")