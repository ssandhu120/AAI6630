# Sharan Sandhu - QA Validation and Extended Analysis

This folder contains Sharan Sandhu's QA validation work for the AAI 6630 Sit Stay Forever group project.

## Synthetic Pipeline Validation

`validate_pipeline.py` creates controlled synthetic images representing:

- normal high-resolution imagery
- underexposure
- overexposure
- low resolution
- moderate resolution
- high sharpness
- blur

Each case defines an expected result and is passed through the existing OpenCV image-quality pipeline.

Result:

- 7 test cases executed
- 7 passed
- 100% functional pass rate

Output:

`outputs/synthetic_validation_results.csv`

## Existing Output Validation

`validate_existing_results.py` independently validates all 54 existing image-quality output rows against the implemented classification rules.

Checks include:

- brightness labels
- blur classification
- resolution classification
- overall quality status

Result:

- 54 rows validated
- 54 passed
- 0 failed
- 100% internal consistency

Output:

`outputs/existing_results_validation.csv`

## Edge-Case Analysis

`edge_cases.md` documents limitations and edge conditions identified during validation, including:

- flat or uniform image regions
- sharpness sensitivity to edge density
- highly textured imagery
- blur-threshold sensitivity
- component-level warnings versus overall quality score

## Interpretation

The validation confirms that the pipeline behaves consistently with its implemented rules for the tested conditions.

This does not establish that the thresholds are universally correct for all Amazon product images or that any image characteristic causes stronger sales performance.

The original 54 source image files were not available in the shared repository, so direct manual visual validation of those source images was not performed.
