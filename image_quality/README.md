# William's OpenCV Image Quality Analysis — Final Version

This folder contains **only William's assigned image-quality work** for the AAI 6630 XN project.

## Included scope

The pipeline measures four explainable image-quality areas:

1. **Brightness** — identifies images that may be too dark or overexposed.
2. **Contrast** — measures grayscale variation and flags unusually low or harsh contrast.
3. **Sharpness / blur** — uses Laplacian variance after size normalization.
4. **Resolution** — evaluates the shortest image dimension and megapixel count.

It also creates:

- component scores and an overall image-quality score;
- basic recommendations for each image;
- CSV outputs and presentation-ready charts;
- a manual-validation worksheet for checking automated labels.

## Intentionally excluded

These items belong to other team modules or the group-level analysis and are **not implemented here**:

- OCR and keyword extraction;
- color-tone or color-consistency analysis;
- accessibility simulation;
- object detection or semantic segmentation;
- Random Forest feature importance;
- performance-tier or sponsor-versus-competitor benchmarking;
- image augmentation.

## Folder structure

```text
william_final_image_quality_project/
├── images/
├── outputs/
├── william_image_quality_pipeline.py
├── requirements.txt
├── run_pipeline.bat
├── METHODS_AND_PRESENTATION_NOTES.md
└── README.md
```

## Installation

Open PowerShell or Command Prompt in this folder and run:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python william_image_quality_pipeline.py
```

Or double-click `run_pipeline.bat` on Windows.

The main output is:

```text
outputs/image_quality_results.csv
```

## Important interpretation note

The thresholds and overall score are explainable screening rules. They help identify images that deserve review, but they do not prove that an image causes stronger Amazon sales or engagement. The `manual_validation_template.csv` file should be used to compare automated labels with human observations.
