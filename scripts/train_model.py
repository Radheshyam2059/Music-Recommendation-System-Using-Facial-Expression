"""
Train a modern CNN for facial emotion recognition on FER-2013 data.

Usage:
    python scripts/train_model.py

The dataset must already be prepared by download_dataset.py in:
    data/train/<ClassName>/  and  data/test/<ClassName>/

Output:
    Models/model_new.h5          — best model checkpoint
    Models/training_history.json  — epoch-by-epoch metrics
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ---- Configuration ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "Models", "model_new.h5")
HISTORY_OUTPUT = os.path.join(PROJECT_ROOT, "Models", "training_history.json")

# Must match app.py label_map order
CLASS_NAMES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001


def check_gpu():
    """Report available hardware."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[✓] GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[!] No GPU found — training on CPU (this will be slower).")
    print()


def create_data_generators():
    """Create training (augmented) and validation data generators."""
    # Training with aggressive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
    )

    # Test/validation — only rescale
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"\n[*] Training samples: {train_generator.samples}")
    print(f"[*] Test samples:     {test_generator.samples}")
    print(f"[*] Classes:          {train_generator.class_indices}")
    print()

    return train_generator, test_generator


def build_model(num_classes=6):
    """Build a Mini-VGGNet style CNN with BatchNorm and Dropout."""
    model = keras.Sequential(name="MoodifyEmotionNet")

    # Block 1: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3: 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Fully Connected layers
    model.add(layers.Flatten())

    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def compute_weights(train_generator):
    """Compute class weights to handle imbalanced dataset."""
    class_labels = train_generator.classes
    unique_classes = np.unique(class_labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=class_labels)
    class_weight_dict = dict(zip(unique_classes, weights))

    print("[*] Class weights (to handle imbalance):")
    for cls_idx, weight in class_weight_dict.items():
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
        print(f"    {cls_name}: {weight:.3f}")
    print()

    return class_weight_dict


def train():
    """Main training function."""
    print("=" * 60)
    print("  MOODIFY — Emotion Detection Model Training")
    print("=" * 60)
    print()

    # Check hardware
    check_gpu()

    # Verify data directory exists
    if not os.path.isdir(TRAIN_DIR):
        print(f"[✗] Training directory not found: {TRAIN_DIR}")
        print("    Run 'python scripts/download_dataset.py' first!")
        return

    # Data generators
    train_gen, test_gen = create_data_generators()

    if train_gen.samples == 0:
        print("[✗] No training images found! Check your data directory.")
        return

    # Build model
    model = build_model(num_classes=len(CLASS_NAMES))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print()

    # Compute class weights
    class_weights = compute_weights(train_gen)

    # Callbacks
    callback_list = [
        # Save best model by validation accuracy
        callbacks.ModelCheckpoint(
            MODEL_OUTPUT,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Reduce LR when val_loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # Stop early if no improvement
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Train!
    print("[*] Starting training...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1,
    )

    # Save training history
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(HISTORY_OUTPUT, "w") as f:
        json.dump(history_dict, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\n  Best model saved to: {MODEL_OUTPUT}")
    print(f"  Training history:    {HISTORY_OUTPUT}")
    print(f"\n  Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Final Test Loss:     {test_loss:.4f}")
    print()
    print("Next step: python scripts/evaluate_model.py")


if __name__ == "__main__":
    train()
