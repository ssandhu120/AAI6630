# OpenCV Image Quality Analysis Pipeline

This project analyzes Amazon product images for the XN Project: **Computer Vision for Product Image Optimization**.

## Project Purpose

The pipeline evaluates image quality and composition using OpenCV-based computer vision metrics. The goal is to create an image quality scorecard that can help identify product images that may need improvement for e-commerce use.

## Dataset

The current folder includes 42 unique product images. During upload, 60 product image files were received, but 18 were exact duplicates and were skipped from the final analysis set.

## Metrics Calculated

The pipeline calculates:

- Image width and height
- Megapixels
- Aspect ratio
- Brightness
- Contrast
- Sharpness using Laplacian variance
- Blur flag
- White background ratio
- Edge density
- Colorfulness
- Saturation
- Approximate content bounding box area ratio
- Approximate centering score
- Overall quality score
- Pass / Review / Needs Improvement status
- Improvement recommendations

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python image_quality_pipeline.py
```

Or:

```bash
py image_quality_pipeline.py
```

The output files will be saved in the `outputs` folder.

## Output Files

- `image_quality_results.csv`
- `overall_quality_score_histogram.png`
- `sharpness_histogram.png`
- `white_background_ratio_histogram.png`
- `quality_status_counts.png`
- `average_score_by_product.png`
- `summary.txt`

## Current Results

The first run analyzed 42 unique images.

Status distribution:

- Pass: 15
- Review: 26
- Needs Improvement: 1

Average overall image quality score: 76.38

## Notes

The product occupancy and centering metrics use heuristic OpenCV rules based on non-white pixels. These are useful for initial screening but should be interpreted as approximate metrics rather than perfect object detection results.

For future improvements, this pipeline can be extended with OCR, CLIP, ResNet, or YOLO-based product localization.
