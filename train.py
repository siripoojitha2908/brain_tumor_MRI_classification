
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------
# 1. SETTINGS
# -------------------------------
DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
train_dir = DATA_DIR / "train"
valid_dir = DATA_DIR / "valid"
test_dir = DATA_DIR / "test"

for folder in [train_dir, valid_dir, test_dir]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

class_names = sorted(
    folder.name for folder in train_dir.iterdir()
    if folder.is_dir()
)

if len(class_names) < 2:
    raise ValueError("At least two class folders are required.")

print("Classes:", class_names)

def load_data(folder, shuffle=False):
    return keras.utils.image_dataset_from_directory(
        str(folder),
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED
    )

train_ds = load_data(train_dir, shuffle=True)
valid_ds = load_data(valid_dir)
test_ds = load_data(test_dir)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

num_classes = len(class_names)

# -------------------------------
# 3. DATA AUGMENTATION
# -------------------------------
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
], name="augmentation")

# -------------------------------
# 4. CUSTOM CNN
# -------------------------------
def build_cnn():
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        num_classes, activation="softmax"
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------------
# 5. TRANSFER LEARNING MODEL
# -------------------------------
def build_mobilenet():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    base.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)

    # Keep pretrained feature extraction in inference mode
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes, activation="softmax"
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------------
# 6. TRAIN AND SAVE EACH MODEL
# -------------------------------
def train_model(model, name):
    print(f"\nTraining {name}...")

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / f"{name}.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.CSVLogger(
            str(RESULT_DIR / f"{name}_history.csv")
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Reload the best checkpoint
    best_model = keras.models.load_model(
        MODEL_DIR / f"{name}.keras"
    )

    plot_history(history, name)
    evaluate_model(best_model, name)

    return best_model

# -------------------------------
# 7. TRAINING GRAPHS
# -------------------------------
def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_history.png")
    plt.close()

# -------------------------------
# 8. TEST EVALUATION
# -------------------------------
def evaluate_model(model, name):
    print(f"\nEvaluating {name} on test data...")

    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    probabilities = model.predict(test_ds)
    y_pred = np.argmax(probabilities, axis=1)

    print("\nTest results:")
    model.evaluate(test_ds, verbose=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification report:")
    print(report)

    with open(RESULT_DIR / f"{name}_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{name}_confusion_matrix.png")
    plt.close()

    return float(np.mean(y_true == y_pred))

# -------------------------------
# 9. RUN BOTH MODELS
# -------------------------------
cnn_model = build_cnn()
cnn_accuracy = train_model(cnn_model, "custom_cnn")

transfer_model = build_mobilenet()
mobilenet_accuracy = train_model(
    transfer_model, "mobilenet_v2"
)

# -------------------------------
# 10. COMPARE MODELS
# -------------------------------
print("\nFINAL TEST ACCURACY COMPARISON")
print(f"Custom CNN:   {cnn_accuracy:.4f}")
print(f"MobileNetV2:  {mobilenet_accuracy:.4f}")

best_name = (
    "custom_cnn"
    if cnn_accuracy >= mobilenet_accuracy
    else "mobilenet_v2"
)

print("Better model by test accuracy:", best_name)
print("Check per-class metrics before selecting a model.")
