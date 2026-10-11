
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=42
)

# Load the validation dataset with the same class order
class_names = train_ds.class_names

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=class_names,
    shuffle=False
)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# Data augmentation
augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05)
])

# Build CNN
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/custom_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds.prefetch(tf.data.AUTOTUNE),
    validation_data=valid_ds.prefetch(tf.data.AUTOTUNE),
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("models/custom_cnn.keras")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/cnn_accuracy.png")
plt.close()

print("Classes:", class_names)
print("CNN training completed!")
