# Future Interns ML Task 2 (2026)

## Support Ticket Classification and Prioritization Project Plan

## 1) Goal

Build a machine learning system that reads ticket text and predicts:

- ticket category: Billing, Technical Issue, Account, General Query
- ticket priority: High, Medium, Low

The final output should help support teams route and escalate tickets faster.

## 2) Scope and Deliverables

Required deliverables:

- category classification model
- priority prediction model
- evaluation metrics (accuracy, precision, recall, F1)
- confusion matrix and class-wise analysis
- business summary for support operations
- clean and reproducible repository structure

Repository deliverables:

- src/run_pipeline.py
- data/processed outputs and metadata
- figures with Plotly visuals
- reports/final_business_summary.md
- reports/error_analysis.md
- reports/presentation_talk_track.md
- scripts/verify_task2_requirements.py

## 3) ML Method

- text preprocessing: lowercase, punctuation cleanup, stopword removal
- feature extraction: TF-IDF with unigram and bigram features
- models compared:
  - Dummy most-frequent baseline
  - Logistic Regression
  - Linear SVC
  - Multinomial Naive Bayes
- model selection metric: validation F1 (macro)
- final evaluation: holdout test split

## 4) Business Use Case

The system is designed for support operations where incoming ticket volume is high.

Expected impact:

- faster triage and assignment
- better SLA adherence
- reduced manual ticket sorting workload
- clearer operational reporting with model metrics

## 5) Execution Checklist

- [x] Build text cleaning function
- [x] Add TF-IDF feature extraction
- [x] Train and compare classification models
- [x] Evaluate and export metrics
- [x] Save model artifacts
- [x] Generate confusion matrices and distributions
- [x] Write business-facing reports
- [x] Provide automated compliance validation script
