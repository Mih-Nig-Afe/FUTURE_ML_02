#!/usr/bin/env python3
"""Future Interns ML Task 2 requirement verification."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_exists(path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, f"FOUND: {path.relative_to(PROJECT_ROOT)}"
    return False, f"MISSING: {path.relative_to(PROJECT_ROOT)}"


def required_files_check() -> list[tuple[bool, str]]:
    required_files = [
        PROJECT_ROOT / "src" / "run_pipeline.py",
        PROJECT_ROOT / "data" / "processed" / "clean_tickets.csv",
        PROJECT_ROOT / "data" / "processed" / "category_model_comparison.csv",
        PROJECT_ROOT / "data" / "processed" / "priority_model_comparison.csv",
        PROJECT_ROOT / "data" / "processed" / "test_predictions.csv",
        PROJECT_ROOT / "data" / "processed" / "run_metadata.json",
        PROJECT_ROOT / "data" / "processed" / "category_classification_report.csv",
        PROJECT_ROOT / "data" / "processed" / "priority_classification_report.csv",
        PROJECT_ROOT / "figures" / "category_distribution.html",
        PROJECT_ROOT / "figures" / "priority_distribution.html",
        PROJECT_ROOT / "figures" / "category_confusion_matrix.html",
        PROJECT_ROOT / "figures" / "priority_confusion_matrix.html",
        PROJECT_ROOT / "reports" / "final_business_summary.md",
        PROJECT_ROOT / "reports" / "error_analysis.md",
        PROJECT_ROOT / "reports" / "linkedin_post.md",
        PROJECT_ROOT / "reports" / "presentation_talk_track.md",
        PROJECT_ROOT / "reports" / "TASK2_REQUIREMENTS_STATUS.md",
        PROJECT_ROOT / "docs" / "ML_TASK2_PROJECT_PLAN.md",
        PROJECT_ROOT / "models" / "category_model.pkl",
        PROJECT_ROOT / "models" / "priority_model.pkl",
        PROJECT_ROOT / "notebooks" / "support_ticket_classification_walkthrough.ipynb",
    ]
    return [check_exists(path) for path in required_files]


def check_metadata() -> tuple[bool, str]:
    metadata_path = PROJECT_ROOT / "data" / "processed" / "run_metadata.json"
    if not metadata_path.exists():
        return False, "MISSING metadata file"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"FAILED to parse metadata JSON: {exc}"

    selected = metadata.get("selected_models", {})
    test_metrics = metadata.get("test_metrics", {})

    for key in ["category", "priority"]:
        if key not in selected:
            return False, f"MISSING selected model for {key}"
        task_metrics = test_metrics.get(key, {})
        for metric_key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            if metric_key not in task_metrics:
                return False, f"MISSING {metric_key} metric for {key}"

    return True, (
        "METADATA OK: "
        f"category_model={selected['category']}, priority_model={selected['priority']}, "
        f"category_f1={test_metrics['category']['f1_macro']:.4f}, "
        f"priority_f1={test_metrics['priority']['f1_macro']:.4f}"
    )


def check_model_comparison(path: Path, expected_baseline: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING model comparison CSV: {path.name}"

    with path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    if not rows:
        return False, f"EMPTY model comparison CSV: {path.name}"

    model_names = {row.get("model", "") for row in rows}
    if expected_baseline not in model_names:
        return False, f"BASELINE {expected_baseline} missing from {path.name}"
    if len(model_names) < 2:
        return False, f"Need at least 2 models in {path.name}"

    return True, f"MODEL COMPARISON OK ({path.name}): {len(model_names)} models"


def check_summary_content() -> tuple[bool, str]:
    summary_path = PROJECT_ROOT / "reports" / "final_business_summary.md"
    if not summary_path.exists():
        return False, "MISSING final business summary"

    content = summary_path.read_text(encoding="utf-8").lower()
    required_phrases = [
        "how tickets are categorized",
        "how priority is decided",
        "support operations",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in content]
    if missing:
        return False, f"SUMMARY missing required phrases: {missing}"

    return (
        True,
        "BUSINESS SUMMARY includes categorization, priority logic, and operations impact",
    )


def check_visuals() -> tuple[bool, str]:
    visuals = [
        PROJECT_ROOT / "figures" / "category_distribution.html",
        PROJECT_ROOT / "figures" / "priority_distribution.html",
        PROJECT_ROOT / "figures" / "category_confusion_matrix.html",
        PROJECT_ROOT / "figures" / "priority_confusion_matrix.html",
    ]

    for visual in visuals:
        if not visual.exists():
            return False, f"MISSING visual: {visual.name}"
        content = visual.read_text(encoding="utf-8", errors="ignore").lower()
        if "plotly" not in content:
            return False, f"VISUAL is not a Plotly chart: {visual.name}"

    return True, "VISUALS OK: distribution and confusion matrix charts are present"


def main() -> int:
    checks: list[tuple[bool, str]] = []
    checks.extend(required_files_check())
    checks.append(
        check_model_comparison(
            PROJECT_ROOT / "data" / "processed" / "category_model_comparison.csv",
            expected_baseline="dummy_most_frequent",
        )
    )
    checks.append(
        check_model_comparison(
            PROJECT_ROOT / "data" / "processed" / "priority_model_comparison.csv",
            expected_baseline="dummy_most_frequent",
        )
    )
    checks.append(check_metadata())
    checks.append(check_summary_content())
    checks.append(check_visuals())

    print("Future Interns ML Task 2 - Compliance Check")
    print("=" * 46)

    failed = 0
    for ok, message in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {message}")
        if not ok:
            failed += 1

    print("=" * 46)
    if failed:
        print(f"RESULT: {failed} check(s) failed")
        return 1

    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
