import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression


PROJECT_FOLDER = Path(__file__).resolve().parent
CLASSES = ("Elbow", "Hand", "Shoulder")
SAMPLES_PER_CLASS = 3000
OUTPUT_PATH = PROJECT_FOLDER / "weights" / "body_part_classifier.pkl"


def image_files(root):
    return [path for path in root.rglob("*") if path.is_file()]


def extract_features(paths):
    values = []
    for path in paths:
        with Image.open(path) as source_image:
            gray = source_image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            pixels = np.asarray(gray, dtype=np.float32) / 255.0
            aspect_ratio = source_image.width / max(source_image.height, 1)
        values.append(np.concatenate((pixels.ravel(), [aspect_ratio])))
    return np.asarray(values)


def main():
    random_state = np.random.default_rng(42)
    paths = []
    labels = []
    for label, body in enumerate(CLASSES):
        body_paths = image_files(PROJECT_FOLDER / "Dataset" / "train_valid" / body)
        sample_size = min(SAMPLES_PER_CLASS, len(body_paths))
        selected_paths = random_state.choice(body_paths, size=sample_size, replace=False)
        paths.extend(selected_paths)
        labels.extend([label] * sample_size)

    classifier = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="multinomial")
    classifier.fit(extract_features(paths), labels)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as classifier_file:
        pickle.dump(classifier, classifier_file)
    print(f"Saved body-part classifier to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
