"""
Sit Stay Forever — Image Augmentation
Group A: Image Quality & Composition
Author: Nishant Chaudhari

Expands the 54-image set to 200+ samples for a proper train/val/test split.
Applies: horizontal flip, brightness shifts, small rotation, gaussian noise.

This script operates on the actual image FILES (not the spreadsheet), so run it
locally where your image folder lives.

Run:  python augment_images.py
Edit INPUT_DIR / OUTPUT_DIR below to point at your folders.
Needs: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
from pathlib import Path

INPUT_DIR = "D:\\College NEU\\CV\\product_images"          # <-- folder with original product images
OUTPUT_DIR = "D:\\College NEU\\CV\\images_augmented" # <-- where augmented copies get written
AUG_PER_IMAGE = 3               # 54 originals * (1 + 3) = 216 total

os.makedirs(OUTPUT_DIR, exist_ok=True)


def adjust_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # white border fill so we don't introduce black corners on white-bg images
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def add_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augmentations(img):
    """Return a list of (suffix, image) augmented variants."""
    return [
        ("flip", cv2.flip(img, 1)),
        ("bright", adjust_brightness(img, np.random.uniform(0.8, 1.2))),
        ("rot", rotate(img, np.random.uniform(-8, 8))),
        ("noise", add_noise(img, sigma=8)),
    ]


def main():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in Path(INPUT_DIR).iterdir() if f.suffix.lower() in exts]
    if not files:
        print(f"No images found in '{INPUT_DIR}'. Edit INPUT_DIR at the top of this file.")
        return

    total = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  skipped (unreadable): {f.name}")
            continue
        # keep the original
        cv2.imwrite(os.path.join(OUTPUT_DIR, f.name), img)
        total += 1
        # pick AUG_PER_IMAGE random augmentations
        augs = augmentations(img)
        np.random.shuffle(augs)
        for suffix, aug_img in augs[:AUG_PER_IMAGE]:
            out_name = f"{f.stem}_{suffix}{f.suffix}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), aug_img)
            total += 1

    print(f"Done. {len(files)} originals -> {total} total images in '{OUTPUT_DIR}/'")
    print("Now you have enough for a 70/15/15 train/val/test split.")


if __name__ == "__main__":
    main()
