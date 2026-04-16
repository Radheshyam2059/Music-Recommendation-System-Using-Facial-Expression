"""
Download and prepare the FER-2013 dataset for emotion detection training.

Usage:
    python scripts/download_dataset.py

Prerequisites:
    - Kaggle API key (kaggle.json) placed at ~/.kaggle/kaggle.json
    - pip install kaggle pandas Pillow
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

# ---- Configuration ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# FER-2013 emotion labels (original)
FER_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Merge mapping: merge Disgust → Angry to keep 6 classes matching Moodify's label_map
MERGE_MAP = {
    "Disgust": "Angry",
}

# Final 6 classes (must match app.py label_map order)
FINAL_CLASSES = ["Angry", "Neutral", "Fear", "Happy", "Sad", "Surprise"]


def download_fer2013():
    """Download FER-2013 dataset from Kaggle."""
    kaggle_json_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print("=" * 60)
        print("ERROR: Kaggle API key not found!")
        print(f"Expected location: {kaggle_json_path}")
        print()
        print("Steps to fix:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' under the API section")
        print("3. Save the downloaded kaggle.json to ~/.kaggle/")
        print("=" * 60)
        sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)

    csv_path = os.path.join(RAW_DIR, "fer2013.csv")
    if os.path.exists(csv_path):
        print(f"[✓] FER-2013 CSV already exists at {csv_path}")
        return csv_path

    print("[*] Downloading FER-2013 dataset from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("msambare/fer2013", path=RAW_DIR, unzip=True)
        print("[✓] Download complete!")
    except Exception as e:
        print(f"[!] Kaggle API download failed: {e}")
        print("[*] Trying alternative dataset (deadskull7/fer2013)...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("deadskull7/fer2013", path=RAW_DIR, unzip=True)
            print("[✓] Download complete!")
        except Exception as e2:
            print(f"[✗] Both downloads failed: {e2}")
            print("Please manually download FER-2013 from Kaggle and place fer2013.csv in data/raw/")
            sys.exit(1)

    # Check if we got the CSV or a directory-based dataset
    if os.path.exists(csv_path):
        return csv_path

    # The msambare/fer2013 dataset comes as pre-organized folders (train/test)
    # Check if that structure exists
    msambare_train = os.path.join(RAW_DIR, "train")
    msambare_test = os.path.join(RAW_DIR, "test")
    if os.path.isdir(msambare_train) and os.path.isdir(msambare_test):
        print("[✓] Dataset downloaded as pre-organized image folders (msambare format).")
        return "FOLDER_FORMAT"

    # Look for CSV with different possible names
    for f in os.listdir(RAW_DIR):
        if f.endswith(".csv"):
            csv_path = os.path.join(RAW_DIR, f)
            print(f"[✓] Found CSV: {csv_path}")
            return csv_path

    print("[✗] Could not find dataset files after download.")
    sys.exit(1)


def organize_from_folders(raw_dir):
    """Organize the msambare/fer2013 folder-based dataset into our structure."""
    src_train = os.path.join(raw_dir, "train")
    src_test = os.path.join(raw_dir, "test")

    for split, src_dir in [("train", src_train), ("test", src_test)]:
        for class_name in os.listdir(src_dir):
            src_class_dir = os.path.join(src_dir, class_name)
            if not os.path.isdir(src_class_dir):
                continue

            # Map class name (handle case variations)
            mapped_name = class_name.capitalize()
            if mapped_name in MERGE_MAP:
                mapped_name = MERGE_MAP[mapped_name]
            if mapped_name not in FINAL_CLASSES:
                print(f"  [!] Skipping unknown class: {class_name}")
                continue

            dst_class_dir = os.path.join(DATA_DIR, split, mapped_name)
            os.makedirs(dst_class_dir, exist_ok=True)

            count = 0
            for img_file in os.listdir(src_class_dir):
                src_path = os.path.join(src_class_dir, img_file)
                dst_path = os.path.join(dst_class_dir, f"{class_name}_{img_file}")
                if not os.path.exists(dst_path):
                    import shutil
                    shutil.copy2(src_path, dst_path)
                count += 1

            print(f"  [{split}] {mapped_name}: {count} images")


def organize_from_csv(csv_path):
    """Parse FER-2013 CSV and save images into train/test folder structure."""
    print(f"[*] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[*] Dataset contains {len(df)} samples")
    print(f"[*] Columns: {list(df.columns)}")

    # Create directories
    for split in ["train", "test"]:
        for cls in FINAL_CLASSES:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    # Map FER-2013 'Usage' column to our splits
    usage_map = {
        "Training": "train",
        "PublicTest": "test",
        "PrivateTest": "test",
    }

    counters = {}
    for idx, row in df.iterrows():
        emotion_idx = int(row["emotion"])
        pixels = row["pixels"]
        usage = row.get("Usage", "Training")

        # Get emotion name
        emotion_name = FER_LABELS.get(emotion_idx, None)
        if emotion_name is None:
            continue

        # Apply merge mapping
        if emotion_name in MERGE_MAP:
            emotion_name = MERGE_MAP[emotion_name]

        if emotion_name not in FINAL_CLASSES:
            continue

        # Determine split
        split = usage_map.get(usage, "train")

        # Convert pixel string to image
        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        img_array = pixel_values.reshape(48, 48)
        img = Image.fromarray(img_array, mode="L")

        # Save image
        key = f"{split}_{emotion_name}"
        counters[key] = counters.get(key, 0) + 1
        filename = f"{emotion_name}_{counters[key]:05d}.png"
        img.save(os.path.join(DATA_DIR, split, emotion_name, filename))

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")

    print("\n[✓] Dataset organization complete!")
    print("\nFinal class distribution:")
    for split in ["train", "test"]:
        print(f"\n  {split.upper()}:")
        for cls in FINAL_CLASSES:
            cls_dir = os.path.join(DATA_DIR, split, cls)
            count = len(os.listdir(cls_dir)) if os.path.exists(cls_dir) else 0
            print(f"    {cls}: {count} images")


def main():
    print("=" * 60)
    print("  MOODIFY — FER-2013 Dataset Downloader & Organizer")
    print("=" * 60)
    print()

    result = download_fer2013()

    if result == "FOLDER_FORMAT":
        print("\n[*] Organizing folder-based dataset...")
        organize_from_folders(RAW_DIR)
    else:
        print(f"\n[*] Organizing CSV-based dataset...")
        organize_from_csv(result)

    # Print final summary
    print("\n" + "=" * 60)
    print("  DONE! Dataset is ready at:")
    print(f"  {os.path.join(DATA_DIR, 'train')}  (training set)")
    print(f"  {os.path.join(DATA_DIR, 'test')}   (test set)")
    print("=" * 60)
    print("\nNext step: python scripts/train_model.py")


if __name__ == "__main__":
    main()
