# Final Business Summary

## Project outcome

A support ticket decision-support system was built to classify ticket category and predict urgency priority from text.

## How tickets are categorized

Tickets are cleaned (lowercasing, punctuation removal, stopword removal), transformed with TF-IDF features, and then classified into:

- Billing
- Technical Issue
- Account
- General Query

Selected category model: **logistic_regression**

Category test performance:

- Accuracy: 1.0000
- Precision (macro): 1.0000
- Recall (macro): 1.0000
- F1 (macro): 1.0000

## How priority is decided

A second text classification model predicts urgency labels:

- High
- Medium
- Low

Selected priority model: **logistic_regression**

Priority test performance:

- Accuracy: 1.0000
- Precision (macro): 1.0000
- Recall (macro): 1.0000
- F1 (macro): 1.0000

## How this improves support operations

- Routes tickets to the right specialist queue faster.
- Escalates high-priority tickets sooner to reduce SLA breaches.
- Reduces manual triage workload so agents spend more time solving issues.
- Provides measurable service quality metrics for support managers.
