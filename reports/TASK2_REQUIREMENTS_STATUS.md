# Task 2 Requirements Status

## Project

- Task: Future Interns Machine Learning Task 2 (2026)
- Topic: Support Ticket Classification and Prioritization
- Status date: April 8, 2026
- Overall status: **Complete and verifiable**

## Dataset Sources Used (Exact)

- IT Service Ticket Classification (Kaggle): https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset
  - file: `data/raw/kaggle_it_service_ticket_classification/all_tickets_processed_improved_v3.csv`
- Customer Support Ticket Dataset (Kaggle): https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
  - file: `data/raw/kaggle_customer_support_ticket/customer_support_tickets.csv`
- Classification of IT Support Tickets (provided local folder)
  - files: `Classification of IT Support Tickets Authors:Creators /X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

Active model training input:

- `data/raw/support_tickets.csv` (canonical merge built from available sources)

## Requirement Coverage

### 1) Text cleaning (lowercasing, stopword removal, punctuation handling)

- Status: **Complete**
- Evidence:
  - `src/run_pipeline.py` (`clean_text` function)
  - `data/processed/clean_tickets.csv`

### 2) Feature extraction (TF-IDF / Bag of Words)

- Status: **Complete**
- Evidence:
  - `src/run_pipeline.py` (`TfidfVectorizer` in model pipelines)

### 3) Ticket category classification

- Status: **Complete**
- Evidence:
  - `data/processed/category_model_comparison.csv`
  - `models/category_model.pkl`

### 4) Priority prediction (High / Medium / Low)

- Status: **Complete**
- Evidence:
  - `data/processed/priority_model_comparison.csv`
  - `models/priority_model.pkl`

### 5) Model evaluation (accuracy, precision, recall)

- Status: **Complete**
- Evidence:
  - `data/processed/run_metadata.json`
  - `data/processed/category_classification_report.csv`
  - `data/processed/priority_classification_report.csv`

### 6) Optional bonus: confusion matrix and class-wise analysis

- Status: **Complete**
- Evidence:
  - `figures/category_confusion_matrix.html`
  - `figures/priority_confusion_matrix.html`
  - `reports/error_analysis.md`

### 7) Business-friendly explanation

- Status: **Complete**
- Evidence:
  - `reports/final_business_summary.md`
  - `reports/presentation_talk_track.md`

## Validation Commands

Run the pipeline:

```bash
python3 src/run_pipeline.py
```

Run compliance checks:

```bash
python3 scripts/verify_task2_requirements.py
```
