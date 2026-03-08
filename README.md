# Match Prediction & Behavior Analysis Project

A modular Machine Learning pipeline built in Java using the **Weka API** to analyze dating app behavior and predict match success with high accuracy.

---

## 🚀 Overview
This project implements a complete data science lifecycle optimized for dating app datasets. It addresses common challenges like **class imbalance** (via SMOTE) and **high-cardinality noise** (by discarding messy features) to achieve robust binary classification.

## 🛠 Key Features
- **Exploratory Data Analysis (EDA)**: Automated descriptive statistics (Mean, Min, Max, and Counts) for all processed attributes.
- **Target Re-engineering**: Converts complex multiclass outcomes into a high-accuracy binary target: `is_success` (Yes/No).
- **Strict Feature Selection**: Uses a curated 10-feature behavioral set to maximize signal and minimize noise.
- **Advanced Model Suite**:
  - **Naive Bayes**: Fast probabilistic classification.
  - **Bagging (J48)**: Stable ensemble of decision trees.
  - **AdaBoostM1 (J48)**: Powerful boosting algorithm.
  - **SMO (SVM)**: Robust Support Vector Machine with normalization.
- **Leakage-Free SMOTE**: Class balancing is applied within cross-validation folds using `FilteredClassifier`.
- **User Segmentation**: KMeans clustering to discover 3 distinct user personas.
- **Behavioral Patterns**: Apriori rule mining with tuned support/confidence for dating insights.

---

## 📂 Project Structure
```text
MatchPredictionProject/
├── data/raw/                # Original CSV dataset
├── models/                  # Serialized .model files (Bagging/J48)
├── src/main/java/com/match/
│   ├── preprocessing/       # DataManager: Ingestion, 15k Downsampling, SMOTE
│   ├── learning/            # ModelManager: 4-Model Evaluation Suite
│   ├── analysis/            # AnalysisManager: KMeans & Apriori
│   └── App.java             # Pipeline Orchestrator
├── src/main/java/weka/      # Custom SMOTE implementation
└── reports/                 # Project documentation and analysis
```

---

## ⚙️ Requirements & Execution
- **JDK**: 17+ recommended (supports -Xmx4G).
- **Library**: `weka.jar` must be in the classpath.
- **Execution**: Run `com.match.App.main()`.

---

## 📜 Detailed Documentation
- [PIPELINE.md](./PIPELINE.md) - Technical workflow and algorithm details.
- [STRUCTURE.md](./STRUCTURE.md) - Comprehensive file and package map.
