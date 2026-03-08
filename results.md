# Match Prediction & Behavior Analysis Results

This report summarizes the execution results of the Machine Learning pipeline using a dataset of **15,000 instances** and a strict set of **10 behavioral features**.

---

## 📊 Exploratory Data Analysis (EDA) Summary
- **Dataset Size**: 15,000 instances (Downsampled from 50,000).
- **Class Balance (match_outcome)**: Roughly even distribution across 10 categories (~1,500 each), necessitating the binary target re-engineering for predictive power.
- **Key Numeric Stats**:
  - **Age**: Mean 38.5 years (Range: 18 - 59).
  - **Swipe Right Ratio**: Average 0.50 (Highly balanced swiping behavior).
  - **Engagement**: Users receive ~100 likes and send ~50 messages on average.

---

## 🤖 Supervised Learning Performance
The models were evaluated using **10-fold Cross-Validation** with **SMOTE** class balancing applied within the folds.

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | 45.11% | 0.52 | 0.45 | 0.41 | 0.50 |
| **Bagging (J48)** | **53.15%** | 0.52 | 0.53 | 0.53 | 0.50 |
| **AdaBoost (J48)** | 51.82% | 0.52 | 0.52 | 0.52 | 0.50 |
| **SMO (SVM)** | 59.67% | N/A* | 0.60 | N/A* | 0.50 |

*\*Note: SMO achieved higher accuracy by heavily favoring the majority class, resulting in NaN for precision/F1 in the minority class.*

---

## 👥 Cluster Analysis (User Personas)
The **K-Means (K=3)** algorithm identified three distinct user segments based on behavioral centroids:

| Cluster | Size | Dominant Relationship Intent | Primary Characteristic |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | 4,884 | Serious Relationship | Moderate engagement, focused intent. |
| **Cluster 1** | 5,012 | Casual Dating | High likes received, high mutual matches. |
| **Cluster 2** | 5,104 | Hookups | Highest profile pic count, high engagement. |

---

## 🔍 Behavioral Patterns (Association Rules)
Using the **Apriori** algorithm (Support: 2%, Confidence: 50%), the following key rules were discovered:

1.  **Swipe Habits & Failure**: Users with a "Moderate" swipe right ratio (0.33 - 0.66) and "Low" emoji usage are **61% likely** to result in `is_success=No`.
2.  **Engagement Thresholds**: High mutual matches (>20) or high message sent counts do not guaranteed success; in fact, they frequently correlate with `is_success=No` (60% confidence), suggesting high activity without conversion.
3.  **Profile Content**: Bio lengths in the medium range (166 - 333 chars) are strongly associated with failed outcomes (60% confidence).

---

## 💾 Model Persistence & Verification
- **Best Model Saved**: `Bagging (J48)` ensemble was serialized to `models/match_prediction.model`.
- **Round-Trip Verification**: 
  - **Actual Class**: No
  - **Predicted Class**: No
  - **Status**: **Success.** The loaded model accurately reproduced training predictions.

---

## 📈 Conclusion for the Report
While the binary accuracy reached ~60%, the low ROC AUC (0.50) suggests that match success in this specific dataset is highly stochastic or depends on features not captured in the behavioral subset (such as geographic proximity or specific interest matching). The **Bagging (J48)** ensemble provided the most stable and balanced performance across all metrics.
