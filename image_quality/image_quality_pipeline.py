import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def score_range(value, low_good, high_good, low_abs, high_abs):
    if value < low_abs or value > high_abs:
        return 0
    if low_good <= value <= high_good:
        return 100
    if value < low_good:
        return max(0, 100 * (value - low_abs) / (low_good - low_abs))
    return max(0, 100 * (high_abs - value) / (high_abs - high_good))


def analyze_image(path):
    image = cv2.imread(str(path))
    if image is None:
        return {"filename": path.name, "error": "Could not read image"}

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    min_dimension = min(width, height)
    megapixels = (width * height) / 1_000_000

    white_mask = cv2.inRange(image, np.array([245, 245, 245]), np.array([255, 255, 255]))
    white_ratio = float(np.mean(white_mask > 0))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    blue, green, red = cv2.split(image.astype("float"))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    saturation = float(np.mean(hsv[:, :, 1]))

    non_white = cv2.inRange(image, np.array([0, 0, 0]), np.array([244, 244, 244]))
    kernel = np.ones((5, 5), np.uint8)
    non_white_clean = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, kernel)
    coords = cv2.findNonZero(non_white_clean)

    if coords is not None:
        x, y, box_width, box_height = cv2.boundingRect(coords)
        bbox_area_ratio = float((box_width * box_height) / (width * height))
        center_x = x + box_width / 2
        center_y = y + box_height / 2
        center_offset = float(
            math.sqrt(
                ((center_x - width / 2) / (width / 2)) ** 2
                + ((center_y - height / 2) / (height / 2)) ** 2
            )
        )
    else:
        bbox_area_ratio = 0.0
        center_offset = 1.0

    aspect_ratio = float(width / height) if height else 0
    asin = path.name.split("_")[0] if "_" in path.name else ""

    resolution_score = 100 if min_dimension >= 1000 else max(0, min_dimension / 1000 * 100)
    sharpness_score = min(100, sharpness / 300 * 100)
    brightness_score = score_range(brightness, 120, 225, 50, 250)
    contrast_score = score_range(contrast, 35, 90, 10, 130)
    white_bg_score = min(100, white_ratio / 0.85 * 100)
    composition_score = max(0, 100 - center_offset * 60)

    overall = (
        0.20 * resolution_score
        + 0.25 * sharpness_score
        + 0.15 * brightness_score
        + 0.15 * contrast_score
        + 0.15 * white_bg_score
        + 0.10 * composition_score
    )

    if overall >= 80:
        status = "Pass"
    elif overall >= 60:
        status = "Review"
    else:
        status = "Needs Improvement"

    recommendations = []
    if min_dimension < 1000:
        recommendations.append("Increase image resolution")
    if sharpness < 100:
        recommendations.append("Retake or sharpen image")
    if brightness < 100:
        recommendations.append("Improve lighting")
    elif brightness > 235:
        recommendations.append("Reduce overexposure")
    if contrast < 25:
        recommendations.append("Increase contrast")
    if white_ratio < 0.75:
        recommendations.append("Review background compliance")
    if center_offset > 0.35:
        recommendations.append("Improve product centering")
    if not recommendations:
        recommendations.append("No major issue detected")

    return {
        "filename": path.name,
        "asin": asin,
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 3),
        "min_dimension": min_dimension,
        "aspect_ratio": round(aspect_ratio, 3),
        "brightness_mean": round(brightness, 2),
        "contrast_std": round(contrast, 2),
        "sharpness_laplacian_var": round(sharpness, 2),
        "blur_flag": "Yes" if sharpness < 100 else "No",
        "white_background_ratio": round(white_ratio, 4),
        "edge_density": round(edge_density, 4),
        "colorfulness": round(colorfulness, 2),
        "saturation_mean": round(saturation, 2),
        "content_bbox_area_ratio": round(bbox_area_ratio, 4),
        "center_offset": round(center_offset, 4),
        "resolution_score": round(resolution_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "brightness_score": round(brightness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "white_bg_score": round(white_bg_score, 2),
        "composition_score": round(composition_score, 2),
        "overall_quality_score": round(overall, 2),
        "status": status,
        "recommendations": "; ".join(recommendations),
    }


def create_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    df["overall_quality_score"].hist(bins=10)
    plt.title("Overall Quality Score Distribution")
    plt.xlabel("Overall Quality Score")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_quality_score_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["sharpness_laplacian_var"].hist(bins=10)
    plt.title("Sharpness Distribution")
    plt.xlabel("Laplacian Variance")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(output_dir / "sharpness_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["white_background_ratio"].hist(bins=10)
    plt.title("White Background Ratio Distribution")
    plt.xlabel("White Background Ratio")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(output_dir / "white_background_ratio_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    df["status"].value_counts().plot(kind="bar")
    plt.title("Image Quality Status Counts")
    plt.xlabel("Status")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_status_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    df.groupby("asin")["overall_quality_score"].mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Average Quality Score by Product ASIN")
    plt.xlabel("Product ASIN")
    plt.ylabel("Average Quality Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "average_score_by_product.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="images")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir.resolve()}")

    records = [analyze_image(path) for path in image_paths]
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "image_quality_results.csv", index=False)

    create_plots(df, output_dir)

    summary = {
        "images_analyzed": int(len(df)),
        "status_counts": df["status"].value_counts().to_dict(),
        "average_overall_quality_score": round(float(df["overall_quality_score"].mean()), 2),
        "average_sharpness": round(float(df["sharpness_laplacian_var"].mean()), 2),
        "average_white_background_ratio": round(float(df["white_background_ratio"].mean()), 4),
        "products_detected": sorted(df["asin"].unique().tolist()),
    }

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as file:
        file.write("OpenCV Image Quality Pipeline Summary\n")
        file.write("=====================================\n")
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print("Analysis completed.")
    print(f"Images analyzed: {len(df)}")
    print(f"Results saved to: {output_dir / 'image_quality_results.csv'}")


if __name__ == "__main__":
    main()
