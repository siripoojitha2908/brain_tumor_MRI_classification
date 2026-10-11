
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=42
)

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

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

# MobileNetV2 expects pixel values scaled to [-1, 1]
x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

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
        "models/mobilenetv2.keras",
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

model.save("models/mobilenetv2.keras")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/transfer_accuracy.png")
plt.close()

print("Classes:", class_names)
print("Transfer learning training completed!")
