
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from pathlib import Path


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_artifacts" / "best_mobilenetv2.weights.h5"
CLASS_PATH = BASE_DIR / "model_artifacts" / "class_names.json"

IMG_SIZE = (224, 224)
NUM_CLASSES = 15


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

    return class_names


# ==========================================
# 3. BUILD MODEL ARCHITECTURE
# ==========================================

@st.cache_resource
def load_model():

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    # Data augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ], name="data_augmentation")

    x = augmentation(inputs)

    # MobileNetV2 preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Load previously trained weights
    model.load_weights(str(MODEL_PATH))

    return model


# ==========================================
# 4. IMAGE PREDICTION FUNCTION
# ==========================================

def predict_disease(image, model, class_names):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    # Top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]

    top_predictions = [
        (
            class_names[i],
            float(predictions[i])
        )
        for i in top_indices
    ]

    return predicted_class, confidence, top_predictions


# ==========================================
# 5. APPLICATION HEADER
# ==========================================

st.title("🌿 Crop Disease Detection")

st.markdown(
    """
    Upload a plant leaf image to identify its
    potential disease using a MobileNetV2
    deep learning model.

    **Supported crops:** Tomato, Potato and Pepper.
    """
)

st.divider()


# ==========================================
# 6. LOAD MODEL
# ==========================================

try:

    if not MODEL_PATH.exists():
        st.error(f"Model weights not found: {MODEL_PATH}")
        st.stop()

    if not CLASS_PATH.exists():
        st.error(f"Class labels not found: {CLASS_PATH}")
        st.stop()

    CLASS_NAMES = load_classes()

    if len(CLASS_NAMES) != NUM_CLASSES:
        st.error("Number of class labels does not match model output.")
        st.stop()

    model = load_model()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error("Unable to load the model.")

    st.exception(e)

    st.stop()


# ==========================================
# 7. IMAGE UPLOAD
# ==========================================

st.subheader("Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

        st.write("")

        # ==================================
        # 8. PREDICTION
        # ==================================

        if st.button(
            "Predict Disease",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Analyzing leaf image..."):

                predicted_class, confidence, top_predictions = (
                    predict_disease(
                        image,
                        model,
                        CLASS_NAMES
                    )
                )

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class.replace('_', ' ')}"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            # Confidence warning
            if confidence < 0.60:

                st.warning(
                    "Low confidence prediction. "
                    "Please upload a clearer image or "
                    "consult an agricultural expert."
                )

            # ==================================
            # 9. TOP 3 PREDICTIONS
            # ==================================

            st.subheader("Top 3 Predictions")

            for class_name, score in top_predictions:

                st.write(
                    f"**{class_name.replace('_', ' ')}**"
                )

                st.progress(score)

                st.caption(
                    f"Confidence: {score * 100:.2f}%"
                )

            # ==================================
            # 10. DISCLAIMER
            # ==================================

            st.divider()

            st.info(
                """
                **Disclaimer:** This application provides
                AI-based preliminary predictions and is
                not a confirmed agricultural diagnosis.

                Consult an agricultural expert before
                making crop protection or treatment decisions.
                """
            )

    except Exception as e:

        st.error("Unable to process the uploaded image.")

        st.exception(e)


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Crop Disease Classification | "
    "MobileNetV2 | TensorFlow/Keras"
)
