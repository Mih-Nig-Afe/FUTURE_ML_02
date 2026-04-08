# Final Business Summary

## Project outcome

A support ticket decision-support system was built to classify ticket category and predict urgency priority from text.

## How tickets are categorized

Tickets are cleaned (lowercasing, punctuation removal, stopword removal), transformed with TF-IDF features, and then classified into:

- Billing
- Technical Issue
- Account
- General Query

Selected category model: **multinomial_nb**

Category test performance:

- Accuracy: 0.7736
- Precision (macro): 0.7330
- Recall (macro): 0.8079
- F1 (macro): 0.7339

## How priority is decided

A second text classification model predicts urgency labels:

- High
- Medium
- Low

Selected priority model: **linear_svc**

Priority test performance:

- Accuracy: 0.8967
- Precision (macro): 0.7578
- Recall (macro): 0.7012
- F1 (macro): 0.7124

## How this improves support operations

- Routes tickets to the right specialist queue faster.
- Escalates high-priority tickets sooner to reduce SLA breaches.
- Reduces manual triage workload so agents spend more time solving issues.
- Provides measurable service quality metrics for support managers.
