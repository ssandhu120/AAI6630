from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SPONSOR_ASIN = "B07TGDFPBN"


def read_image(path: Path) -> np.ndarray | None:
    """Read an image robustly, including Windows paths with non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path))


def resize_for_metric(image: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """Resize large images before sharpness measurement for fairer comparison."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image
    scale = max_side / longest_side
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def score_range(
    value: float,
    low_good: float,
    high_good: float,
    low_absolute: float,
    high_absolute: float,
) -> float:
    """Return a 0-100 score with a full-credit interval."""
    if value < low_absolute or value > high_absolute:
        return 0.0
    if low_good <= value <= high_good:
        return 100.0
    if value < low_good:
        denominator = max(1e-9, low_good - low_absolute)
        return max(0.0, 100.0 * (value - low_absolute) / denominator)
    denominator = max(1e-9, high_absolute - high_good)
    return max(0.0, 100.0 * (high_absolute - value) / denominator)


def resolution_score(min_dimension: int) -> float:
    """Score resolution using the shortest image dimension."""
    if min_dimension >= 1000:
        return 100.0
    if min_dimension <= 150:
        return 0.0
    return float(np.interp(min_dimension, [150, 300, 500, 1000], [0, 20, 55, 100]))


def sharpness_score(laplacian_variance: float) -> float:
    """Score sharpness on a logarithmic scale to limit extreme outliers."""
    low, high = 35.0, 600.0
    value = max(0.0, laplacian_variance)
    numerator = math.log1p(value) - math.log1p(low)
    denominator = math.log1p(high) - math.log1p(low)
    return float(np.clip(100.0 * numerator / max(1e-9, denominator), 0.0, 100.0))


def label_brightness(value: float) -> str:
    if value < 100:
        return "Too Dark"
    if value > 235:
        return "Overexposed"
    return "Acceptable"


def label_contrast(value: float) -> str:
    if value < 25:
        return "Low Contrast"
    if value > 110:
        return "Very High Contrast"
    return "Acceptable"


def label_resolution(min_dimension: int) -> str:
    if min_dimension >= 1000:
        return "High Resolution"
    if min_dimension >= 500:
        return "Moderate Resolution"
    return "Low Resolution"


def analyze_image(path: Path) -> dict[str, Any]:
    image = read_image(path)
    if image is None:
        return {"filename": path.name, "error": "Could not read image"}

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metric_image = resize_for_metric(image)
    metric_gray = cv2.cvtColor(metric_image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(metric_gray, cv2.CV_64F).var())
    min_dimension = int(min(width, height))
    megapixels = float((width * height) / 1_000_000)
    aspect_ratio = float(width / height) if height else 0.0
    asin = path.name.split("_")[0] if "_" in path.name else ""

    resolution_component = resolution_score(min_dimension)
    brightness_component = score_range(brightness, 110, 225, 45, 250)
    contrast_component = score_range(contrast, 35, 90, 10, 130)
    sharpness_component = sharpness_score(sharpness)

    overall_score = (
        0.25 * resolution_component
        + 0.25 * brightness_component
        + 0.20 * contrast_component
        + 0.30 * sharpness_component
    )

    if overall_score >= 80:
        quality_status = "Pass"
    elif overall_score >= 60:
        quality_status = "Review"
    else:
        quality_status = "Needs Improvement"

    blur_flag = "Yes" if sharpness < 80 else "No"
    recommendations: list[str] = []
    if min_dimension < 1000:
        recommendations.append("Increase image resolution")
    if sharpness < 80:
        recommendations.append("Retake or sharpen image")
    if brightness < 100:
        recommendations.append("Improve lighting")
    elif brightness > 235:
        recommendations.append("Reduce overexposure")
    if contrast < 25:
        recommendations.append("Increase contrast")
    elif contrast > 110:
        recommendations.append("Reduce overly harsh contrast")
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
        "brightness_status": label_brightness(brightness),
        "contrast_std": round(contrast, 2),
        "contrast_status": label_contrast(contrast),
        "sharpness_laplacian_var": round(sharpness, 2),
        "blur_flag": blur_flag,
        "resolution_status": label_resolution(min_dimension),
        "resolution_score": round(resolution_component, 2),
        "brightness_score": round(brightness_component, 2),
        "contrast_score": round(contrast_component, 2),
        "sharpness_score": round(sharpness_component, 2),
        "overall_quality_score": round(overall_score, 2),
        "quality_status": quality_status,
        "recommendations": "; ".join(recommendations),
    }


def save_chart(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 12) -> None:
    plt.figure(figsize=(8, 5))
    series.dropna().hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def create_plots(df: pd.DataFrame, output_dir: Path, sponsor_asin: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_chart(
        df["overall_quality_score"],
        "Overall Image Quality Score Distribution",
        "Overall Quality Score",
        output_dir / "overall_quality_score_histogram.png",
    )
    save_chart(
        df["brightness_mean"],
        "Brightness Distribution",
        "Mean Grayscale Brightness",
        output_dir / "brightness_histogram.png",
    )
    save_chart(
        df["contrast_std"],
        "Contrast Distribution",
        "Grayscale Standard Deviation",
        output_dir / "contrast_histogram.png",
    )
    save_chart(
        df["sharpness_laplacian_var"],
        "Sharpness Distribution",
        "Laplacian Variance",
        output_dir / "sharpness_histogram.png",
    )
    save_chart(
        df["min_dimension"],
        "Image Resolution Distribution",
        "Shortest Image Dimension (pixels)",
        output_dir / "resolution_histogram.png",
    )

    plt.figure(figsize=(7, 5))
    df["quality_status"].value_counts().reindex(
        ["Pass", "Review", "Needs Improvement"], fill_value=0
    ).plot(kind="bar")
    plt.title("Image Quality Status Counts")
    plt.xlabel("Status")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_status_counts.png", dpi=170)
    plt.close()

    product_scores = (
        df.groupby("asin", dropna=False)["overall_quality_score"]
        .mean()
        .sort_values(ascending=False)
    )
    product_scores.to_csv(output_dir / "average_quality_score_by_product.csv", header=["average_score"])

    plt.figure(figsize=(9, 5))
    product_scores.plot(kind="bar")
    plt.title("Average Image Quality Score by Product")
    plt.xlabel("Product ASIN")
    plt.ylabel("Average Quality Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "average_quality_score_by_product.png", dpi=170)
    plt.close()

    sponsor_df = df[df["asin"] == sponsor_asin].copy()
    if not sponsor_df.empty:
        sponsor_df.to_csv(output_dir / "sponsor_image_quality_results.csv", index=False)
        plt.figure(figsize=(9, 5))
        sponsor_df.set_index("filename")["overall_quality_score"].plot(kind="bar")
        plt.title("Sponsor Image Quality Scores")
        plt.xlabel("Sponsor Image")
        plt.ylabel("Overall Quality Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "sponsor_image_quality_scores.png", dpi=170)
        plt.close()


def make_manual_validation_template(df: pd.DataFrame, output_path: Path) -> None:
    """Create a small worksheet for manual review of the automated metrics."""
    low_examples = df.nsmallest(min(5, len(df)), "overall_quality_score")
    high_examples = df.nlargest(min(5, len(df)), "overall_quality_score")
    sample = pd.concat([low_examples, high_examples]).drop_duplicates("filename")
    template = sample[
        [
            "filename",
            "brightness_status",
            "contrast_status",
            "blur_flag",
            "resolution_status",
            "overall_quality_score",
        ]
    ].copy()
    template["manual_lighting_assessment"] = ""
    template["manual_contrast_assessment"] = ""
    template["manual_sharpness_assessment"] = ""
    template["manual_resolution_assessment"] = ""
    template["manual_notes"] = ""
    template.to_csv(output_path, index=False)


def write_summary(df: pd.DataFrame, output_dir: Path, sponsor_asin: str) -> None:
    summary = {
        "scope": ["brightness", "contrast", "sharpness/blur", "resolution"],
        "images_analyzed": int(len(df)),
        "products_detected": int(df["asin"].nunique()),
        "average_overall_quality_score": round(float(df["overall_quality_score"].mean()), 2),
        "median_overall_quality_score": round(float(df["overall_quality_score"].median()), 2),
        "quality_status_counts": df["quality_status"].value_counts().to_dict(),
        "blurred_images": int((df["blur_flag"] == "Yes").sum()),
        "low_resolution_images": int((df["min_dimension"] < 1000).sum()),
        "lighting_issues": int((df["brightness_status"] != "Acceptable").sum()),
        "contrast_issues": int((df["contrast_status"] != "Acceptable").sum()),
    }

    sponsor_df = df[df["asin"] == sponsor_asin]
    if not sponsor_df.empty:
        summary["sponsor_asin"] = sponsor_asin
        summary["sponsor_images_analyzed"] = int(len(sponsor_df))
        summary["sponsor_average_quality_score"] = round(
            float(sponsor_df["overall_quality_score"].mean()), 2
        )
        summary["sponsor_blurred_images"] = int((sponsor_df["blur_flag"] == "Yes").sum())
        summary["sponsor_low_resolution_images"] = int(
            (sponsor_df["min_dimension"] < 1000).sum()
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("William's OpenCV Image Quality Analysis Summary\n")
        file.write("================================================\n")
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    lowest = df.nsmallest(min(5, len(df)), "overall_quality_score")
    with (output_dir / "presentation_findings.md").open("w", encoding="utf-8") as file:
        file.write("# Presentation Findings — William's Scope\n\n")
        file.write("This analysis is limited to brightness, contrast, sharpness/blur, and resolution.\n\n")
        file.write(f"- Images analyzed: **{len(df)}**\n")
        file.write(
            f"- Average overall image-quality score: **{df['overall_quality_score'].mean():.2f}/100**\n"
        )
        file.write(f"- Images flagged as blurred: **{(df['blur_flag'] == 'Yes').sum()}**\n")
        file.write(f"- Images below 1000 px on the shortest side: **{(df['min_dimension'] < 1000).sum()}**\n")
        file.write(
            f"- Images with lighting concerns: **{(df['brightness_status'] != 'Acceptable').sum()}**\n"
        )
        file.write(
            f"- Images with contrast concerns: **{(df['contrast_status'] != 'Acceptable').sum()}**\n\n"
        )
        if not sponsor_df.empty:
            file.write(
                f"- Sponsor average score ({sponsor_asin}): **{sponsor_df['overall_quality_score'].mean():.2f}/100**\n\n"
            )
        file.write("## Lowest-scoring images for manual review\n\n")
        for row in lowest.itertuples(index=False):
            file.write(
                f"- `{row.filename}` — {row.overall_quality_score:.2f}/100; "
                f"{row.recommendations}\n"
            )
        file.write("\n## Limitation\n\n")
        file.write(
            "These scores are explainable screening metrics, not evidence of Amazon sales performance. "
            "The automated labels should be checked against manual image review.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OpenCV image-quality analysis limited to William's assigned scope: "
            "brightness, contrast, sharpness/blur, and resolution."
        )
    )
    parser.add_argument("--images", default="images", help="Folder containing input images")
    parser.add_argument("--outputs", default="outputs", help="Folder for CSV files and charts")
    parser.add_argument(
        "--sponsor-asin",
        default=DEFAULT_SPONSOR_ASIN,
        help="Sponsor ASIN used only to create a sponsor-only result file and chart",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {image_dir.resolve()}")

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir.resolve()}")

    records = [analyze_image(path) for path in image_paths]
    error_records = [record for record in records if "error" in record]
    valid_records = [record for record in records if "error" not in record]
    if not valid_records:
        raise RuntimeError("No images could be analyzed successfully")

    results_df = pd.DataFrame(valid_records)
    results_df.to_csv(output_dir / "image_quality_results.csv", index=False)

    if error_records:
        pd.DataFrame(error_records).to_csv(output_dir / "unreadable_images.csv", index=False)

    create_plots(results_df, output_dir, args.sponsor_asin)
    make_manual_validation_template(results_df, output_dir / "manual_validation_template.csv")
    write_summary(results_df, output_dir, args.sponsor_asin)

    print("Analysis completed successfully.")
    print(f"Images analyzed: {len(results_df)}")
    print(f"Unreadable images: {len(error_records)}")
    print(f"Results saved to: {(output_dir / 'image_quality_results.csv').resolve()}")


if __name__ == "__main__":
    main()
