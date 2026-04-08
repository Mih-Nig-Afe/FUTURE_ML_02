# Final Business Summary

## Project outcome

A support ticket decision-support system was built to classify ticket category and predict urgency priority from text.

## How tickets are categorized

Tickets are cleaned (lowercasing, punctuation removal, stopword removal), transformed with TF-IDF features, and then classified into:

- Billing
- Technical Issue
- Account
- General Query

Selected category model: **linear_svc**

Category test performance:

- Accuracy: 0.8114
- Precision (macro): 0.7403
- Recall (macro): 0.7244
- F1 (macro): 0.7318

## How priority is decided

A second text classification model predicts urgency labels:

- High
- Medium
- Low

Selected priority model: **linear_svc**

Priority test performance:

- Accuracy: 0.9899
- Precision (macro): 0.9949
- Recall (macro): 0.7500
- F1 (macro): 0.8308

## How this improves support operations

- Routes tickets to the right specialist queue faster.
- Escalates high-priority tickets sooner to reduce SLA breaches.
- Reduces manual triage workload so agents spend more time solving issues.
- Provides measurable service quality metrics for support managers.
