
from importlib.resources import files
import json
import glob
import os
import numpy as np
import tensorflow as tf

# Load the exact model used by the app
model = tf.keras.models.load_model("models/mobilenetv2.keras")

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

print("Class order:", class_names)
print("Model output shape:", model.output_shape)

# Select the first glioma image from the test dataset
files = sorted(glob.glob("dataset/test/meningioma/*"))

if not files:
    raise FileNotFoundError("No images found in dataset/test/meningioma/")

image_path = files[0]
print("\nImage:", image_path)
print("Expected class: meningioma")

# Use TensorFlow/Keras image loading with bilinear resizing
image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224),
    color_mode="rgb",
    interpolation="bilinear"
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

# Predict
probabilities = model.predict(image_array, verbose=0)[0]
predicted_index = int(np.argmax(probabilities))

print("\nPredicted class:", class_names[predicted_index])
print("\nClass probabilities:")

for name, probability in zip(class_names, probabilities):
    print(f"{name}: {probability * 100:.2f}%")
