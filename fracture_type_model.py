import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_FOLDER, "weights", "ResNet50_FractureType.keras")
METADATA_PATH = os.path.join(PROJECT_FOLDER, "weights", "ResNet50_FractureType.json")
CLASS_NAMES = ("Transverse", "Oblique", "Spiral", "Comminuted", "Greenstick", "Hairline")
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            "The trained six-class fracture-type model is not installed. "
            "Run train_fracture_type.py with the labeled dataset first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("The fracture-type model class order does not match the dashboard.")

    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    if loaded_model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("The fracture-type model must have exactly six output classes.")
    _model = loaded_model
    return _model


def predict_fracture_type(image_path):
    """Return the six-class type prediction and its uncalibrated model score."""
    loaded_model = _load_model()
    input_image = image.load_img(image_path, target_size=(224, 224), color_mode="rgb")
    input_array = image.img_to_array(input_image)
    input_array = tf.keras.applications.resnet50.preprocess_input(input_array)
    probabilities = loaded_model.predict(np.expand_dims(input_array, axis=0), verbose=0)[0]
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("The fracture-type model returned invalid probabilities.")

    prediction_index = int(np.argmax(probabilities))
    return CLASS_NAMES[prediction_index], float(probabilities[prediction_index]) * 100