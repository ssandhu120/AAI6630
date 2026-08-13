from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "validation"
TEST_IMAGE_DIR = VALIDATION_DIR / "synthetic_test_images"
OUTPUT_DIR = VALIDATION_DIR / "outputs"

TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "image_quality"))

from william_image_quality_pipeline import analyze_image


def save_image(name, image):
    path = TEST_IMAGE_DIR / name
    cv2.imwrite(str(path), image)
    return path


def uniform_image(width, height, value):
    return np.full((height, width, 3), value, dtype=np.uint8)


def checkerboard(width, height, block_size=25):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            value = 255 if ((x // block_size) + (y // block_size)) % 2 == 0 else 0
            image[y:y + block_size, x:x + block_size] = value

    return image


def create_test_cases():
    cases = []

    normal = uniform_image(1200, 1200, 200)
    cv2.rectangle(normal, (250, 200), (950, 1000), (90, 90, 90), -1)
    cv2.rectangle(normal, (350, 300), (850, 900), (220, 220, 220), 8)
    cv2.line(normal, (350, 450), (850, 450), (40, 40, 40), 6)
    cv2.line(normal, (350, 600), (850, 600), (40, 40, 40), 6)
    cv2.putText(
        normal,
        "PRODUCT",
        (390, 760),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (30, 30, 30),
        5,
        cv2.LINE_AA,
    )

    cases.append({
        "name": "normal_high_resolution.jpg",
        "image": normal,
        "expected_brightness": "Acceptable",
        "expected_blur": "No",
        "expected_resolution": "High Resolution",
    })

    dark = uniform_image(1200, 1200, 50)

    cases.append({
        "name": "too_dark.jpg",
        "image": dark,
        "expected_brightness": "Too Dark",
        "expected_blur": "Yes",
        "expected_resolution": "High Resolution",
    })

    bright = uniform_image(1200, 1200, 245)

    cases.append({
        "name": "overexposed.jpg",
        "image": bright,
        "expected_brightness": "Overexposed",
        "expected_blur": "Yes",
        "expected_resolution": "High Resolution",
    })

    low_res = uniform_image(300, 300, 170)
    cv2.rectangle(low_res, (50, 50), (250, 250), (70, 70, 70), -1)

    cases.append({
        "name": "low_resolution.jpg",
        "image": low_res,
        "expected_brightness": "Acceptable",
        "expected_blur": "No",
        "expected_resolution": "Low Resolution",
    })

    moderate = uniform_image(700, 700, 170)
    cv2.rectangle(moderate, (100, 100), (600, 600), (60, 60, 60), -1)

    cases.append({
        "name": "moderate_resolution.jpg",
        "image": moderate,
        "expected_brightness": "Acceptable",
        "expected_blur": "No",
        "expected_resolution": "Moderate Resolution",
    })

    sharp = checkerboard(1200, 1200)

    cases.append({
        "name": "sharp_checkerboard.jpg",
        "image": sharp,
        "expected_brightness": "Acceptable",
        "expected_blur": "No",
        "expected_resolution": "High Resolution",
    })

    blurred = cv2.GaussianBlur(sharp, (51, 51), 0)

    cases.append({
        "name": "blurred_checkerboard.jpg",
        "image": blurred,
        "expected_brightness": "Acceptable",
        "expected_blur": "Yes",
        "expected_resolution": "High Resolution",
    })

    return cases


def run_validation():
    cases = create_test_cases()
    results = []

    for case in cases:
        path = save_image(case["name"], case["image"])
        actual = analyze_image(path)

        brightness_pass = actual["brightness_status"] == case["expected_brightness"]
        blur_pass = actual["blur_flag"] == case["expected_blur"]
        resolution_pass = actual["resolution_status"] == case["expected_resolution"]

        results.append({
            "test_image": case["name"],
            "expected_brightness": case["expected_brightness"],
            "actual_brightness": actual["brightness_status"],
            "brightness_test": "PASS" if brightness_pass else "FAIL",
            "expected_blur": case["expected_blur"],
            "actual_blur": actual["blur_flag"],
            "blur_test": "PASS" if blur_pass else "FAIL",
            "expected_resolution": case["expected_resolution"],
            "actual_resolution": actual["resolution_status"],
            "resolution_test": "PASS" if resolution_pass else "FAIL",
            "brightness_mean": actual["brightness_mean"],
            "contrast_std": actual["contrast_std"],
            "sharpness_laplacian_var": actual["sharpness_laplacian_var"],
            "overall_quality_score": actual["overall_quality_score"],
            "quality_status": actual["quality_status"],
            "overall_test_result": (
                "PASS"
                if brightness_pass and blur_pass and resolution_pass
                else "FAIL"
            ),
        })

    df = pd.DataFrame(results)

    output = OUTPUT_DIR / "synthetic_validation_results.csv"
    df.to_csv(output, index=False)

    passed = (df["overall_test_result"] == "PASS").sum()
    total = len(df)

    print("=== Synthetic Pipeline Validation ===")
    print(df.to_string(index=False))
    print(f"\nPassed: {passed}/{total}")
    print(f"Pass rate: {passed / total:.1%}")
    print(f"\nResults written to: {output}")


if __name__ == "__main__":
    run_validation()
