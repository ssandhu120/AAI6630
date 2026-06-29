# Sit Stay Forever — Computer Vision for Product Image Optimization

**Group A: Image Quality & Composition**
Northeastern University — MPS in Applied AI · XN Project

An analysis of Amazon product listing images for Sit Stay Forever, a natural pet products brand, using computer vision to identify the visual characteristics that distinguish high-performing listings and to benchmark the sponsor's own images against them.

## Team

- Nishant Chaudhari
- Chia Yang Hsu (William)
- Sharan Sandhu

## Project Goal

Sit Stay Forever sells a dry powder pet shampoo on Amazon, where image quality affects search ranking, click-through, and customer trust. This project analyzes the brand's product images alongside a set of competitor listings to surface concrete, evidence-based recommendations for improving their visual content.

Because Amazon engagement data (impressions, click-through, conversion) is not available, findings describe correlation between visual characteristics and product success rather than a causal effect on sales.

## Dataset

A structured dataset of 100 Amazon pet grooming products grouped into performance tiers based on real Amazon ratings and review counts (high, medium, low), plus the sponsor's own product.

- `pet_cv_dataset_full.xlsx` — primary working file. Image rows with 20+ computed CV features per image (brightness, contrast, sharpness, white background compliance, product dominance, color warmth, symmetry, text density, dominant colors, OCR text, and more), plus product metadata.
- `SSF_CV_Dataset.xlsx` — supplementary file. 146 keywords across 19 categories mapped to CV features, sponsor and competitor listings, and a data dictionary.

Of the available image rows, 54 currently have a complete set of computed CV features. Augmentation expands this to 216 images for model training.

## Repository Structure

```
.
├── ssf_analysis.py          # Feature analysis, RF feature importance, SSF gap analysis
├── augment_images.py        # Image augmentation pipeline (54 -> 216 images)
├── pet_cv_dataset_full.xlsx # Primary dataset
├── SSF_CV_Dataset.xlsx      # Supplementary keyword + listing dataset
├── outputs/
│   ├── ssf_analysis_charts.png
│   ├── ssf_gap_analysis.csv
│   └── tier_means.csv
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl opencv-python
```

## Usage

**Run the feature analysis** (works directly from the spreadsheet):

```bash
python ssf_analysis.py
```

Produces console output, `ssf_analysis_charts.png`, and the supporting tables `tier_means.csv` and `ssf_gap_analysis.csv`.

**Run the augmentation pipeline** (operates on the actual image files):

1. Place the original images in a folder.
2. Edit `INPUT_DIR` at the top of `augment_images.py` to point at that folder.
3. Run:

```bash
python augment_images.py
```

## Methods

The OpenCV pipeline computes per-image quality metrics (brightness, contrast, blur via Laplacian variance, resolution, white background compliance, and more). The analysis layer then compares mean feature values across performance tiers, trains a Random Forest classifier to rank which features most distinguish high-performing images, and benchmarks the sponsor's images against the high-performer average. Augmentation (horizontal flip, brightness scaling, small-angle rotation, and Gaussian noise) expands the dataset to support a 70/15/15 train, validation, and test split.

Traditional CV methods are used rather than a custom CNN, since the dataset is small and lacks labels indicating which images performed well on Amazon.

## Key Findings (in progress)

- Sponsor images average noticeably lower sharpness than the high-performer group, making image sharpness the clearest improvement opportunity.
- Sponsor white background percentage and product dominance are already at or above the high-performer average.
- A Random Forest ranks saturation, contrast, and color features among the strongest signals separating high-performing images, though reported accuracy is affected by class imbalance and the feature ranking is the more reliable output.

## Limitations

- No ground-truth Amazon engagement data, so findings are correlational.
- Only a subset of rows has complete CV features, with few medium and low-tier examples, so tier comparisons are framed as high performers versus the sponsor.
- Automated metrics can flag usable images as problematic (texture, shadows, reflective packaging), so human review is part of validation.

## Roadmap

- Validate quality metrics against manual review and tune thresholds.
- Add color consistency, product-to-background contrast, and accessibility checks (thumbnail readability, color-vision-deficiency simulation).
- Add OCR-based text and keyword alignment analysis.
- Consolidate findings into a practical image scorecard and photography guidelines for the sponsor.
