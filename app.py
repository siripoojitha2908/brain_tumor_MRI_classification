
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload an MRI image to obtain a model prediction "
    "using a trained MobileNetV2 model."
)

# Load the trained model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenetv2.keras")
    with open("models/class_names.json", "r") as file:
        class_names = json.load(file)
    return model, class_names

model, class_names = load_model()

# Upload an MRI image
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI image",
        use_container_width=True
    )

    if st.button("Analyze MRI Image"):
        # Prepare image for the model
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict class probabilities
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        display_name = predicted_class.replace("_", " ").title()

        st.subheader("Prediction Result")
        st.success(f"Predicted class: {display_name}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.write("Class probabilities:")
        for name, probability in zip(class_names, predictions):
            st.write(
                f"**{name.replace('_', ' ').title()}**: "
                f"{probability * 100:.2f}%"
            )

st.warning(
    "Educational use only. This model is not a medical diagnostic tool. "
    "Do not use its predictions to make medical decisions. "
    "Consult a qualified healthcare professional for diagnosis."
)
