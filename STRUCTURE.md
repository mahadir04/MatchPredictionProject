# Project Directory Structure

```text
MatchPredictionProject/
├── data/
│   └── raw/                            # Source Data
│       └── dating_app_behavior_dataset_extended1.csv
├── models/                             # Persistent Storage
│   └── match_prediction.model          # Serialized Bagging/J48 Model
├── src/
│   └── main/java/com/match/            # Application Logic
│       ├── App.java                    # Entry point; Orchestrates 15k pipeline
│       ├── preprocessing/
│       │   └── DataManager.java        # Loading, 15k Downsampling, Binary Target
│       ├── learning/
│       │   └── ModelManager.java       # NB, Bagging, AdaBoost, SMO Evaluation
│       └── analysis/
│           └── AnalysisManager.java    # KMeans Evaluation & Apriori Rules
│   └── main/java/weka/                 # Custom Weka Extensions
│       └── filters/supervised/instance/
│           └── SMOTE.java              # Standard Weka-compatible Filter
├── build.xml                           # Ant Build Script
├── README.md                           # General Overview
├── PIPELINE.md                         # Technical Workflow
└── STRUCTURE.md                        # This Map
```

### 📂 Directory Purposes:
- **`com.match.preprocessing`**: Handles the critical 15k downsampling and target re-engineering logic.
- **`com.match.learning`**: Contains the cross-validation logic using `FilteredClassifier` to prevent data leakage.
- **`com.match.analysis`**: Implements descriptive analytics (User Personas and Behavioral Rules).
- **`weka` (Internal)**: Houses the functional SMOTE implementation, allowing the project to run without external oversampling jars.
