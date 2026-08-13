from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "image_quality" / "outputs" / "image_quality_results.csv"
OUTPUT = ROOT / "validation" / "outputs" / "existing_results_validation.csv"

df = pd.read_csv(SOURCE)

checks = []

for _, row in df.iterrows():
    problems = []

    expected_brightness = (
        "Too Dark"
        if row["brightness_mean"] < 100
        else "Overexposed"
        if row["brightness_mean"] > 235
        else "Acceptable"
    )

    if row["brightness_status"] != expected_brightness:
        problems.append("brightness label mismatch")

    expected_blur = (
        "Yes"
        if row["sharpness_laplacian_var"] < 80
        else "No"
    )

    if row["blur_flag"] != expected_blur:
        problems.append("blur label mismatch")

    min_dim = row["min_dimension"]

    expected_resolution = (
        "High Resolution"
        if min_dim >= 1000
        else "Moderate Resolution"
        if min_dim >= 500
        else "Low Resolution"
    )

    if row["resolution_status"] != expected_resolution:
        problems.append("resolution label mismatch")

    score = row["overall_quality_score"]

    expected_status = (
        "Pass"
        if score >= 80
        else "Review"
        if score >= 60
        else "Needs Improvement"
    )

    if row["quality_status"] != expected_status:
        problems.append("quality status mismatch")

    checks.append({
        "filename": row["filename"],
        "brightness_check": (
            "PASS"
            if row["brightness_status"] == expected_brightness
            else "FAIL"
        ),
        "blur_check": (
            "PASS"
            if row["blur_flag"] == expected_blur
            else "FAIL"
        ),
        "resolution_check": (
            "PASS"
            if row["resolution_status"] == expected_resolution
            else "FAIL"
        ),
        "quality_status_check": (
            "PASS"
            if row["quality_status"] == expected_status
            else "FAIL"
        ),
        "issues": "; ".join(problems) if problems else "None",
        "overall_result": "FAIL" if problems else "PASS",
    })

result = pd.DataFrame(checks)
result.to_csv(OUTPUT, index=False)

passed = (result["overall_result"] == "PASS").sum()
total = len(result)

print("=== Existing Output Validation ===")
print(f"Rows validated: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Pass rate: {passed / total:.1%}")

if total - passed:
    print("\nFailures:")
    print(
        result[result["overall_result"] == "FAIL"]
        .to_string(index=False)
    )

print(f"\nOutput: {OUTPUT}")

