import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_DIRECTORIES = {
    "Transverse": "Transverse fracture",
    "Oblique": "Oblique fracture",
    "Spiral": "Spiral Fracture",
    "Comminuted": "Comminuted fracture",
    "Greenstick": "Greenstick fracture",
    "Hairline": "Hairline Fracture",
}
CLASS_NAMES = tuple(CLASS_DIRECTORIES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def collect_images(dataset_root):
    records = []
    for class_name, directory_name in CLASS_DIRECTORIES.items():
        class_directory = dataset_root / directory_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")

        for image_path in sorted(class_directory.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as image_file:
                    image_file.verify()
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                print(f"Skipping unreadable image {image_path}: {error}")
                continue
            records.append((digest, class_name, image_path))

    labels_by_hash = {}
    for digest, class_name, _ in records:
        labels_by_hash.setdefault(digest, set()).add(class_name)
    conflicting_hashes = {
        digest for digest, labels in labels_by_hash.items() if len(labels) > 1
    }
    if conflicting_hashes:
        print(
            f"Excluding {len(conflicting_hashes)} exact-image duplicates with conflicting labels."
        )

    rows = []
    seen_hashes = set()
    class_counts = {class_name: 0 for class_name in CLASS_NAMES}
    for digest, class_name, image_path in records:
        if digest in conflicting_hashes or digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        rows.append({"filepath": str(image_path), "label": class_name})
        class_counts[class_name] += 1

    for class_name, class_count in class_counts.items():
        print(f"{class_name}: {class_count} unique, unambiguous images")
        if class_count < 10:
            raise ValueError(f"Only {class_count} unique readable images found for {class_name}")

    return pd.DataFrame(rows)


def make_generator(frame, augment):
    if augment:
        image_data = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
        )
    else:
        image_data = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )

    return image_data.flow_from_dataframe(
        dataframe=frame,
        x_col="filepath",
        y_col="label",
        classes=list(CLASS_NAMES),
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=augment,
        seed=42,
    )


def build_model():
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = backbone(inputs, training=False)
    features = tf.keras.layers.Dropout(0.35)(features)
    features = tf.keras.layers.Dense(128, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.25)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train the six-class fracture-type model using labeled class folders."
    )
    parser.add_argument("dataset", type=Path, help="Root folder containing the 17 fracture class folders")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "weights",
        help="Directory for the trained model and evaluation report",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Evaluate the saved model on the deterministic held-out split without retraining",
    )
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    all_images = collect_images(dataset_root)
    train_and_validation, test_frame = train_test_split(
        all_images,
        test_size=0.15,
        random_state=42,
        stratify=all_images["label"],
    )
    train_frame, validation_frame = train_test_split(
        train_and_validation,
        test_size=0.15 / 0.85,
        random_state=43,
        stratify=train_and_validation["label"],
    )

    print(
        "Dataset has no patient identifiers; these are stratified image-level splits, "
        "not patient-independent validation."
    )
    validation_generator = make_generator(validation_frame, augment=False)
    test_generator = make_generator(test_frame, augment=False)
    model_path = args.output_dir / "ResNet50_FractureType.keras"
    if args.evaluate_existing:
        if not model_path.is_file():
            raise FileNotFoundError(f"No saved model found to evaluate: {model_path}")
        model = tf.keras.models.load_model(model_path)
        if model.output_shape[-1] != len(CLASS_NAMES):
            raise ValueError("The saved model does not have six output classes")
    else:
        train_generator = make_generator(train_frame, augment=True)
        train_labels = np.asarray(
            [CLASS_NAMES.index(label) for label in train_frame["label"]], dtype=np.int32
        )
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(CLASS_NAMES)),
            y=train_labels,
        )
        class_weights = {index: float(weight) for index, weight in enumerate(weights)}

        model = build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
            ),
        ]
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    probabilities = model.predict(test_generator, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    test_labels = np.asarray(test_generator.classes, dtype=np.int32)
    per_class_report = classification_report(
        test_labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        test_labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
    ).tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_existing:
        model.save(model_path)

    metadata = {
        "class_names": list(CLASS_NAMES),
        "image_size": list(IMAGE_SIZE),
        "preprocessing": "tf.keras.applications.resnet50.preprocess_input",
        "dataset_root": str(dataset_root),
        "split": "stratified image-level 70/15/15; no patient IDs available",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": per_class_report,
        "confusion_matrix": confusion,
        "test_predictions": predictions.tolist(),
        "test_labels": test_labels.tolist(),
        "test_probabilities": probabilities.tolist(),
    }
    metadata_path = args.output_dir / "ResNet50_FractureType.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation data: {metadata_path}")
    print(f"Held-out image-level accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()