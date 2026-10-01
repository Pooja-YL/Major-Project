import os
import pickle

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def build_fallback_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling='avg',
    )
    base_model.trainable = False
    inputs = base_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_model_or_fallback(model_path, num_classes):
    full_path = os.path.join(THIS_FOLDER, model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Required trained model was not found: {full_path}")

    try:
        return tf.keras.models.load_model(full_path)
    except (OSError, ValueError, IOError) as error:
        raise RuntimeError(f"Trained model is corrupted or unreadable: {full_path}") from error


# load the models when import "predictions.py"
model_elbow_frac = load_model_or_fallback("weights/ResNet50_Elbow_frac.h5", 2)
model_hand_frac = load_model_or_fallback("weights/ResNet50_Hand_frac.h5", 2)
model_shoulder_frac = load_model_or_fallback("weights/ResNet50_Shoulder_frac.h5", 2)
model_parts = load_model_or_fallback("weights/ResNet50_BodyParts.h5", 3)
body_part_classifier = None
body_part_classifier_path = os.path.join(THIS_FOLDER, "weights/body_part_classifier.pkl")
if os.path.exists(body_part_classifier_path):
    with open(body_part_classifier_path, "rb") as classifier_file:
        body_part_classifier = pickle.load(classifier_file)

# categories for each result by index

#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

#   0-fractured     1-normal
categories_fracture = ['fractured', 'normal']

MIN_IMAGE_WIDTH = 128
MIN_IMAGE_HEIGHT = 128
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 235.0
MIN_CONTRAST = 15.0
MIN_SHARPNESS = 10.0


def check_image_quality(img_path):
    """Return whether an image is suitable for analysis and any failed checks."""
    report = get_image_quality_report(img_path)
    return not report["issues"], report["issues"]


def get_image_quality_report(img_path):
    """Return quality metrics and failed checks for an image."""
    with Image.open(img_path) as source_image:
        image_rgb = source_image.convert("RGB")
        width, height = image_rgb.size
        grayscale = np.asarray(image_rgb.convert("L"), dtype=np.float32)

    brightness = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))
    denoised = np.asarray(
        image_rgb.convert("L").filter(ImageFilter.MedianFilter(size=3)),
        dtype=np.float32,
    )
    noise = float(np.std(grayscale - denoised))
    laplacian = (
        -4 * grayscale
        + np.roll(grayscale, 1, axis=0)
        + np.roll(grayscale, -1, axis=0)
        + np.roll(grayscale, 1, axis=1)
        + np.roll(grayscale, -1, axis=1)
    )
    sharpness = float(np.var(laplacian[1:-1, 1:-1]))

    issues = []
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        issues.append(f"resolution too low ({width}x{height}; minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT})")
    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        issues.append(f"brightness unsuitable ({brightness:.1f}; expected {MIN_BRIGHTNESS:.0f}-{MAX_BRIGHTNESS:.0f})")
    if contrast < MIN_CONTRAST:
        issues.append(f"contrast too low ({contrast:.1f}; minimum {MIN_CONTRAST:.0f})")
    if sharpness < MIN_SHARPNESS:
        issues.append(f"image appears blurred (sharpness {sharpness:.1f}; minimum {MIN_SHARPNESS:.0f})")

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise,
        "issues": issues,
    }


def enhance_xray_image(img_path, output_path):
    """Enhance a poor-quality X-ray and save the image used for analysis."""
    with Image.open(img_path) as source_image:
        enhanced = source_image.convert("RGB")

    if min(enhanced.size) < 224:
        scale = 224 / min(enhanced.size)
        enhanced = enhanced.resize(
            (round(enhanced.width * scale), round(enhanced.height * scale)),
            Image.Resampling.LANCZOS,
        )

    enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
    grayscale = np.asarray(enhanced.convert("L"), dtype=np.float32)
    brightness = float(np.mean(grayscale))
    brightness_factor = max(1.02, min(1.08, 128.0 / max(brightness, 1.0)))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness_factor)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
    enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.1)
    enhanced.save(output_path, format="PNG")
    return output_path


def _body_part_features(img_path):
    with Image.open(img_path) as source_image:
        gray = source_image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
        aspect_ratio = source_image.width / max(source_image.height, 1)
    return np.concatenate((pixels.ravel(), [aspect_ratio]))


# get image and model name, the default model is "Parts"
# Parts - bone type predict model of 3 classes
# otherwise - fracture predict for each part
def predict(img, model="Parts", return_confidence=False):
    size = 224

    if model == 'Parts' and body_part_classifier is not None:
        probabilities = body_part_classifier.predict_proba([_body_part_features(img)])[0]
        prediction_index = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction_index]) * 100
        prediction_str = categories_parts[prediction_index]
        if return_confidence:
            return prediction_str, confidence
        return prediction_str

    if model == 'Parts':
        chosen_model = model_parts
    elif model == 'Elbow':
        chosen_model = model_elbow_frac
    elif model == 'Hand':
        chosen_model = model_hand_frac
    elif model == 'Shoulder':
        chosen_model = model_shoulder_frac
    else:
        raise ValueError(f"Unsupported model '{model}'")

    # Load image with 224x224 pixels
    temp_img = image.load_img(img, target_size=(size, size))

    x = image.img_to_array(temp_img)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Get probabilities from the model
    probabilities = chosen_model.predict(images, verbose=0)

    # Get predicted class
    prediction_index = np.argmax(probabilities, axis=1)[0]

    # Get confidence of predicted class
    confidence = float(probabilities[0][prediction_index]) * 100

    # Choose category
    if model == 'Parts':
        prediction_str = categories_parts[prediction_index]
    else:
        prediction_str = categories_fracture[prediction_index]

    # Return prediction + confidence when requested
    if return_confidence:
        return prediction_str, confidence

    # Keep the original behavior
    return prediction_str


def _get_model(model_name):
    if model_name == 'Parts':
        return model_parts
    if model_name == 'Elbow':
        return model_elbow_frac
    if model_name == 'Hand':
        return model_hand_frac
    if model_name == 'Shoulder':
        return model_shoulder_frac
    raise ValueError(f"Unsupported model '{model_name}'")


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            nested_layer = _find_last_conv_layer(layer)
            if nested_layer is not None:
                return nested_layer
    return None


def _generate_gradcam_heatmap(img, model_name, class_index=None):
    chosen_model = _get_model(model_name)
    last_conv_layer = _find_last_conv_layer(chosen_model)
    if last_conv_layer is None:
        raise ValueError(f"No convolutional layer found in '{model_name}' model")

    temp_img = image.load_img(img, target_size=(224, 224))
    input_array = image.img_to_array(temp_img)
    input_array = tf.keras.applications.resnet50.preprocess_input(input_array)
    input_array = np.expand_dims(input_array, axis=0)
    input_tensor = tf.convert_to_tensor(input_array)

    grad_model = tf.keras.models.Model(
        inputs=chosen_model.inputs,
        outputs=[last_conv_layer.output, chosen_model.output],
    )
    with tf.GradientTape() as tape:
        convolution_output, predictions = grad_model(input_tensor)
        predicted_index = tf.argmax(predictions[0]) if class_index is None else class_index
        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, convolution_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    convolution_output = convolution_output[0]
    heatmap = tf.reduce_sum(convolution_output * pooled_gradients[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return np.asarray(heatmap.numpy(), dtype=np.float32)


def generate_gradcam(img, model_name):
    """Return an image overlay showing regions influencing a fracture result."""
    heatmap = _generate_gradcam_heatmap(img, model_name)

    original = Image.open(img).convert("RGB")
    heatmap_image = Image.fromarray(np.asarray(255 * heatmap, dtype=np.uint8), mode="L").resize(original.size, Image.Resampling.BILINEAR)
    colorized_heatmap = ImageOps.colorize(heatmap_image, black=(0, 0, 0), white=(255, 0, 0))
    return Image.blend(original, colorized_heatmap, alpha=0.45)


def generate_fracture_localization(img, model_name, original_img=None):
    """Draw a suspected fracture-region box from the fracture-class Grad-CAM."""
    heatmap = _generate_gradcam_heatmap(img, model_name, class_index=0)
    source_path = original_img or img
    with Image.open(source_path) as source_image:
        localized = source_image.convert("RGB")

    resized_heatmap = Image.fromarray(np.asarray(255 * heatmap, dtype=np.uint8), mode="L").resize(
        localized.size,
        Image.Resampling.BILINEAR,
    )
    activation = np.asarray(resized_heatmap, dtype=np.float32) / 255.0
    peak_activation = float(activation.max())
    if peak_activation <= np.finfo(np.float32).eps:
        raise ValueError("The model did not produce an activation map for localization")

    active_y, active_x = np.where(activation >= peak_activation * 0.6)
    if active_x.size == 0:
        raise ValueError("The model did not produce a clear localization region")

    padding_x = max(4, int(localized.width * 0.025))
    padding_y = max(4, int(localized.height * 0.025))
    left = max(0, int(active_x.min()) - padding_x)
    top = max(0, int(active_y.min()) - padding_y)
    right = min(localized.width - 1, int(active_x.max()) + padding_x)
    bottom = min(localized.height - 1, int(active_y.max()) + padding_y)
    line_width = max(3, min(localized.size) // 120)
    ImageDraw.Draw(localized).rectangle((left, top, right, bottom), outline="#FF2E3F", width=line_width)
    return localized
