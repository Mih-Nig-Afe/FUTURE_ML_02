# Future Interns ML Task 2 (2026)

## Support Ticket Classification and Prioritization

This repository contains a complete NLP workflow for classifying support tickets and predicting urgency priority.

## Objective

Build a practical decision-support system that can:

- read support ticket text
- classify the ticket category (Billing, Technical Issue, Account, General Query)
- assign a priority level (High, Medium, Low)

## What is implemented

- text cleaning (lowercasing, punctuation handling, stopword removal)
- TF-IDF feature extraction
- category classification model comparison
- priority prediction model comparison
- test evaluation using accuracy, precision, recall, and F1
- confusion matrices and class-wise performance outputs
- business-friendly reports for managers and stakeholders

## Validated result

Latest verified run date: April 8, 2026

- selected category model: multinomial_nb
- selected priority model: linear_svc
- category test accuracy: 0.7736
- category test precision (macro): 0.7330
- category test recall (macro): 0.8079
- category test F1 (macro): 0.7339
- priority test accuracy: 0.8967
- priority test precision (macro): 0.7578
- priority test recall (macro): 0.7012
- priority test F1 (macro): 0.7124

## Project structure

FUTURE_ML_02/
|- data/
|  |- raw/
|  `- processed/
|- docs/
|- figures/
|- models/
|- notebooks/
|- reports/
|- scripts/
|- src/
|  `- run_pipeline.py
|- docker-compose.yml
|- Dockerfile
|- requirements.txt
`- README.md

## Run locally

1. Install dependencies:

   pip install -r requirements.txt

2. Run the pipeline:

   python3 src/run_pipeline.py

3. Validate Task 2 coverage:

   python3 scripts/verify_task2_requirements.py

## Run with Docker

1. Build image:

   docker compose build

2. Run Task 2 pipeline:

   docker compose run --rm task2

2a. Fetch Task 2 datasets from requested sources inside Docker:

   docker compose run --rm datasets

2b. Verify Task 2 requirement coverage inside Docker:

   docker compose run --rm verify

This Docker dataset flow uses:

- `kaggle datasets download adisongoh/it-service-ticket-classification-dataset`
- `kaggle datasets download suraj520/customer-support-ticket-dataset`
- Local folder merge from `Classification of IT Support Tickets Authors:Creators ` into `data/raw/support_tickets_from_local_it_support.csv`
- Canonical merged ticket dataset output: `data/raw/support_tickets.csv`

Exact dataset links and local files:

- IT Service Ticket Classification (Kaggle): https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset
   - Local file: `data/raw/kaggle_it_service_ticket_classification/all_tickets_processed_improved_v3.csv`
- Customer Support Ticket Dataset (Kaggle): https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
   - Local file: `data/raw/kaggle_customer_support_ticket/customer_support_tickets.csv`
- Classification of IT Support Tickets (local provided folder):
   - Local files: `Classification of IT Support Tickets Authors:Creators /X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

Active training dataset used by the NLP pipeline:

- `data/raw/support_tickets.csv` (canonical merge built from all available sources above)

Kaggle auth requirement (optional but required to download Kaggle datasets):

- Recommended one-time setup for persistent local secret:

   ```bash
   cp .env.example .env
   # edit .env and set KAGGLE_API_TOKEN=...
   ```

- Preferred: `KAGGLE_API_TOKEN` in local `.env` before running Docker.
- Alternative: put your Kaggle API token file at `~/.kaggle/kaggle.json` on the host machine.
- If neither is provided, the dataset fetch service runs but skips Kaggle downloads and the strict dataset-source checks will fail.

3. Launch Jupyter:

   docker compose up notebook

4. Open:

   http://localhost:8888/lab?token=tickets

## Main outputs

Processed data:

- data/processed/clean_tickets.csv
- data/processed/category_model_comparison.csv
- data/processed/priority_model_comparison.csv
- data/processed/test_predictions.csv
- data/processed/category_classification_report.csv
- data/processed/priority_classification_report.csv
- data/processed/run_metadata.json

Visualizations:

- figures/category_distribution.html
- figures/priority_distribution.html
- figures/category_confusion_matrix.html
- figures/priority_confusion_matrix.html

Reports:

- reports/final_business_summary.md
- reports/error_analysis.md
- reports/TASK2_REQUIREMENTS_STATUS.md
- reports/presentation_talk_track.md

## Notes

- If data/raw/support_tickets.csv does not exist, the pipeline creates a realistic synthetic dataset automatically.
- To use your own dataset, place a CSV file at data/raw/support_tickets.csv with columns: ticket_text, category, priority.
- The loader also auto-detects common alternatives such as text or description, ticket_type or label, and urgency or severity.
- Priority values like critical, urgent, p1, p2, p3, and sev1-sev4 are normalized to High, Medium, and Low.
- The provided local IT Support folder is supported by `scripts/fetch_task2_datasets.py` and can be merged automatically via Docker.
- All deliverables are generated by src/run_pipeline.py.
