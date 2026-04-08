# LinkedIn Post Draft

I completed **Future Interns ML Task 2 (2026): Support Ticket Classification and Prioritization**.

I built a machine learning workflow that reads ticket text and predicts both:

- ticket category (Billing, Technical Issue, Account, General Query)
- ticket priority (High, Medium, Low)

What I implemented:

- text cleaning (lowercasing, punctuation handling, stopword removal)
- TF-IDF feature extraction
- category classification model comparison
- priority classification model comparison
- confusion matrices and class-wise performance reports
- business-ready recommendations for support operations

Final selected models:

- Category: multinomial_nb (F1 macro: 0.7339)
- Priority: linear_svc (F1 macro: 0.7124)

Business impact:

- faster ticket routing
- quicker escalation for urgent issues
- lower manual triage effort
- improved support SLA tracking

#MachineLearning #NLP #TextClassification #CustomerSupport #Python #ScikitLearn #FutureInterns
