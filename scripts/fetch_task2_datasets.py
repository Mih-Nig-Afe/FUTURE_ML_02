#!/usr/bin/env python3
"""Fetch Task 2 datasets from requested Kaggle sources and local provided folder."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

KAGGLE_IT_SERVICE_DIR = RAW_DIR / "kaggle_it_service_ticket_classification"
KAGGLE_CUSTOMER_SUPPORT_DIR = RAW_DIR / "kaggle_customer_support_ticket"
LOCAL_IT_SUPPORT_DIR = (
    BASE_DIR / "Classification of IT Support Tickets Authors:Creators "
)
MERGED_LOCAL_DATASET = RAW_DIR / "support_tickets_from_local_it_support.csv"
CANONICAL_DATASET = RAW_DIR / "support_tickets.csv"

TEXT_COLUMN_CANDIDATES = [
    "ticket_text",
    "text",
    "description",
    "ticket_description",
    "issue_description",
    "message",
    "complaint",
]

CATEGORY_COLUMN_CANDIDATES = [
    "category",
    "category_truth",
    "ticket_category",
    "ticket_type",
    "issue_type",
    "type",
    "topic",
    "label",
]

PRIORITY_COLUMN_CANDIDATES = [
    "priority",
    "urgency",
    "severity",
    "impact",
    "priority_level",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_IT_SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_CUSTOMER_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)


def has_kaggle_auth() -> bool:
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if token:
        return True

    config = Path.home() / ".kaggle" / "kaggle.json"
    return config.exists()


def run_kaggle_download(args: list[str], target_dir: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("[WARN] kaggle CLI is not installed in this environment.")
        return

    if not has_kaggle_auth():
        print(
            "[WARN] Kaggle auth not found. Set KAGGLE_API_TOKEN or provide ~/.kaggle/kaggle.json."
        )
        return

    command = [kaggle, *args, "--path", str(target_dir), "--unzip"]
    print("[INFO] Running:", " ".join(command))
    subprocess.run(command, check=True)


def resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {column.strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def normalize_priority(value: str) -> str:
    key = str(value).strip().lower()
    high_values = {"high", "urgent", "critical", "p1", "sev1", "severity1", "1"}
    medium_values = {"medium", "normal", "moderate", "p2", "sev2", "severity2", "2"}
    low_values = {
        "low",
        "minor",
        "p3",
        "p4",
        "sev3",
        "sev4",
        "severity3",
        "severity4",
        "3",
        "4",
    }

    if key in high_values:
        return "High"
    if key in medium_values:
        return "Medium"
    if key in low_values:
        return "Low"
    return "Medium"


def derive_priority_from_text(text: str) -> str:
    value = str(text).lower()
    high_markers = [
        "urgent",
        "critical",
        "outage",
        "down",
        "immediately",
        "asap",
        "production",
    ]
    low_markers = [
        "no rush",
        "when possible",
        "whenever",
        "low priority",
        "question",
        "guidance",
    ]

    if any(marker in value for marker in high_markers):
        return "High"
    if any(marker in value for marker in low_markers):
        return "Low"
    return "Medium"


def fetch_kaggle_sources() -> None:
    run_kaggle_download(
        ["datasets", "download", "adisongoh/it-service-ticket-classification-dataset"],
        KAGGLE_IT_SERVICE_DIR,
    )
    run_kaggle_download(
        ["datasets", "download", "suraj520/customer-support-ticket-dataset"],
        KAGGLE_CUSTOMER_SUPPORT_DIR,
    )


def build_local_it_support_merged_csv() -> None:
    if not LOCAL_IT_SUPPORT_DIR.exists():
        print(f"[WARN] Local folder not found: {LOCAL_IT_SUPPORT_DIR}")
        return

    x_train_path = LOCAL_IT_SUPPORT_DIR / "X_train.csv"
    y_train_path = LOCAL_IT_SUPPORT_DIR / "y_train.csv"
    x_test_path = LOCAL_IT_SUPPORT_DIR / "X_test.csv"
    y_test_path = LOCAL_IT_SUPPORT_DIR / "y_test.csv"

    if not all(
        path.exists() for path in [x_train_path, y_train_path, x_test_path, y_test_path]
    ):
        print("[WARN] Missing one or more local IT support CSV files; skipping merge.")
        return

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    train = x_train.merge(y_train, on="id", how="inner")
    test = x_test.merge(y_test, on="id", how="inner")
    merged = pd.concat([train, test], ignore_index=True)

    merged = merged.rename(
        columns={"text": "ticket_text", "category_truth": "category"}
    )

    # The local dataset does not include explicit urgency labels.
    # Derive approximate priorities from text so all three classes can appear.
    merged["priority"] = merged["ticket_text"].map(derive_priority_from_text)

    merged = merged[["id", "ticket_text", "category", "priority"]]
    merged.to_csv(MERGED_LOCAL_DATASET, index=False)
    print(f"[INFO] Saved merged local IT support dataset to {MERGED_LOCAL_DATASET}")


def canonicalize_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Skipping unreadable CSV {csv_path}: {exc}")
        return None

    text_col = resolve_column(frame, TEXT_COLUMN_CANDIDATES)
    category_col = resolve_column(frame, CATEGORY_COLUMN_CANDIDATES)
    priority_col = resolve_column(frame, PRIORITY_COLUMN_CANDIDATES)

    if text_col is None or category_col is None:
        return None

    working = frame[[text_col, category_col]].copy()
    working = working.rename(
        columns={text_col: "ticket_text", category_col: "category"}
    )

    if priority_col is not None:
        working["priority"] = frame[priority_col].astype(str).map(normalize_priority)
    else:
        working["priority"] = working["ticket_text"].map(derive_priority_from_text)

    working = working.dropna(subset=["ticket_text", "category", "priority"]).copy()
    working["ticket_text"] = working["ticket_text"].astype(str)
    working["category"] = working["category"].astype(str)

    if working.empty:
        return None

    return working[["ticket_text", "category", "priority"]]


def build_canonical_dataset() -> None:
    candidate_csvs: list[Path] = []

    candidate_csvs.extend(sorted(KAGGLE_IT_SERVICE_DIR.rglob("*.csv")))
    candidate_csvs.extend(sorted(KAGGLE_CUSTOMER_SUPPORT_DIR.rglob("*.csv")))

    if MERGED_LOCAL_DATASET.exists():
        candidate_csvs.append(MERGED_LOCAL_DATASET)

    frames: list[pd.DataFrame] = []
    for csv_path in candidate_csvs:
        canonical = canonicalize_csv(csv_path)
        if canonical is not None:
            frames.append(canonical)
            print(f"[INFO] Added canonical rows from {csv_path}")

    if not frames:
        print("[WARN] No canonical ticket dataset could be built from available files.")
        return

    merged = (
        pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    )
    merged.to_csv(CANONICAL_DATASET, index=False)
    print(
        f"[INFO] Saved canonical support dataset to {CANONICAL_DATASET} with {len(merged)} rows"
    )


def main() -> None:
    ensure_dirs()
    fetch_kaggle_sources()
    build_local_it_support_merged_csv()
    build_canonical_dataset()
    print("[INFO] Task 2 dataset fetch flow completed.")


if __name__ == "__main__":
    main()
