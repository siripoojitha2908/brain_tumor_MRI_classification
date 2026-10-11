
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

TEST_DIR = "dataset/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

os.makedirs("outputs", exist_ok=True)

# Load the saved class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Load test images without shuffling
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

# Actual labels
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

model_paths = [
    ("Custom CNN", "models/custom_cnn.keras", "cnn"),
    ("MobileNetV2", "models/mobilenetv2.keras", "mobilenetv2")
]

results = []

for model_name, model_path, prefix in model_paths:
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    results.append((model_name, accuracy))

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")

    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        f"outputs/{prefix}_confusion_matrix.png"
    )
    plt.close()

# Compare the test accuracy
print("\nMODEL COMPARISON")
for name, accuracy in results:
    print(f"{name}: {accuracy * 100:.2f}%")

best_model = max(results, key=lambda item: item[1])
print("Higher test accuracy:", best_model[0])
print("Evaluation completed.")
