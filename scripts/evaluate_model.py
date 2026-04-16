"""
Evaluate the trained emotion detection model on the test set.

Usage:
    python scripts/evaluate_model.py

Output:
    - Test accuracy and loss
    - Per-class precision, recall, F1-score
    - Confusion matrix saved as Models/confusion_matrix.png
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# ---- Configuration ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
NEW_MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "model_new.h5")
OLD_MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "model.h5")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "Models", "training_history.json")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "Models", "confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(PROJECT_ROOT, "Models", "training_curves.png")

CLASS_NAMES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = 48
BATCH_SIZE = 64


def get_test_generator():
    """Create test data generator."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return test_gen


def evaluate_model(model, model_name, test_gen):
    """Evaluate a model and print detailed metrics."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'=' * 60}\n")

    # Overall metrics
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}\n")

    # Per-class metrics
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("  Classification Report:")
    print("  " + "-" * 56)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    for line in report.split("\n"):
        print(f"  {line}")

    return y_true, y_pred, test_acc


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a color-coded confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("True", fontsize=12)
    ax1.set_xticks(range(len(CLASS_NAMES)))
    ax1.set_yticks(range(len(CLASS_NAMES)))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax1.set_yticklabels(CLASS_NAMES, fontsize=10)
    fig.colorbar(im1, ax=ax1)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)

    # Normalized (percentages)
    ax2 = axes[1]
    im2 = ax2.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax2.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("True", fontsize=12)
    ax2.set_xticks(range(len(CLASS_NAMES)))
    ax2.set_yticks(range(len(CLASS_NAMES)))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax2.set_yticklabels(CLASS_NAMES, fontsize=10)
    fig.colorbar(im2, ax=ax2)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center", color=color, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [✓] Confusion matrix saved to: {save_path}")
    plt.close()


def plot_training_curves(history_path, save_path):
    """Plot accuracy and loss curves from training history."""
    if not os.path.exists(history_path):
        print(f"  [!] Training history not found at {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [✓] Training curves saved to: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("  MOODIFY — Model Evaluation")
    print("=" * 60)

    if not os.path.isdir(TEST_DIR):
        print(f"\n[✗] Test directory not found: {TEST_DIR}")
        print("    Run 'python scripts/download_dataset.py' first!")
        return

    test_gen = get_test_generator()
    print(f"\n[*] Test samples: {test_gen.samples}")

    # Evaluate new model
    if os.path.exists(NEW_MODEL_PATH):
        new_model = load_model(NEW_MODEL_PATH, compile=False)
        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        y_true, y_pred, new_acc = evaluate_model(new_model, "NEW Model (model_new.h5)", test_gen)
        plot_confusion_matrix(y_true, y_pred, CONFUSION_MATRIX_PATH)
    else:
        print(f"\n[!] New model not found at {NEW_MODEL_PATH}")
        print("    Run 'python scripts/train_model.py' first!")
        new_acc = None

    # Evaluate old model for comparison
    if os.path.exists(OLD_MODEL_PATH):
        try:
            old_model = load_model(OLD_MODEL_PATH, compile=False)
            old_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            test_gen.reset()
            _, _, old_acc = evaluate_model(old_model, "OLD Model (model.h5)", test_gen)

            if new_acc is not None:
                print(f"\n{'=' * 60}")
                print(f"  COMPARISON")
                print(f"{'=' * 60}")
                improvement = (new_acc - old_acc) * 100
                arrow = "↑" if improvement > 0 else "↓"
                print(f"  Old Model: {old_acc * 100:.2f}%")
                print(f"  New Model: {new_acc * 100:.2f}%")
                print(f"  Change:    {arrow} {abs(improvement):.2f}%")
        except Exception as e:
            print(f"\n[!] Could not load old model for comparison: {e}")

    # Plot training curves
    plot_training_curves(HISTORY_PATH, TRAINING_CURVES_PATH)

    print(f"\n{'=' * 60}")
    print("  Evaluation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
