# Presentation Talk Track

## 60-second version

I built a support ticket classification and prioritization system for Future Interns ML Task 2. The solution reads ticket text, cleans it, creates TF-IDF features, and predicts both category and urgency level. I compared baseline and machine learning classifiers, selected **linear_svc** for category and **linear_svc** for priority, and evaluated them on a holdout test split. The models provide fast triage support so teams can route tickets better and escalate urgent cases earlier.

## 2 to 3 minute walkthrough

### 1. Business problem

Support teams lose time when they manually sort incoming tickets and miss urgent issues. This creates backlog and SLA risk.

### 2. Data preparation

Ticket text was cleaned using lowercasing, punctuation removal, and stopword removal. This created consistent text inputs for modeling.

### 3. Feature extraction and modeling

I used TF-IDF for feature extraction and compared multiple models for both classification tasks.

### 4. Evaluation

Category model (linear_svc) test metrics:

- Accuracy: 0.8114
- Precision (macro): 0.7403
- Recall (macro): 0.7244
- F1 (macro): 0.7318

Priority model (linear_svc) test metrics:

- Accuracy: 0.9899
- Precision (macro): 0.9949
- Recall (macro): 0.7500
- F1 (macro): 0.8308

### 5. Operational value

- automatic category routing to relevant support teams
- automatic urgency labeling to speed up escalation
- better agent productivity because less manual sorting is needed
- measurable quality monitoring through precision and recall by class
