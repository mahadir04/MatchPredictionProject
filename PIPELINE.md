# Match Prediction Pipeline (Technical Workflow)

The pipeline is designed for scientific rigor and performance, processing **15,000 instances** through a multi-stage refinement process.

---

## 🏗 Pipeline Stages

### 1. Data Ingestion & Sanitization
- **Manual Discard**: `interest_tags` is removed immediately to prevent memory exhaustion.
- **Downsampling**: The raw 50k dataset is randomly sampled down to **15,000** instances.
- **EDA**: Prints attribute statistics to verify data distribution before processing.

### 2. Preprocessing & Feature Engineering
- **Binary Target**: Multiclass `match_outcome` is mapped to `is_success` (Yes/No).
  - *Success*: Mutual Match, Date Happened, Relationship Formed, Instant Match.
  - *Failure*: Everything else.
- **Strict Selection**: Filters the dataset down to 10 key attributes:
  - `likes_received`, `mutual_matches`, `message_sent_count`, `swipe_right_ratio`, `profile_pics_count`, `bio_length`, `age`, `relationship_intent`, `last_active_hour`, `emoji_usage_rate`.

### 3. Model Training (Balanced Evaluation)
To prevent **Data Leakage**, SMOTE is wrapped inside a `FilteredClassifier`. This ensures the test set remains unseen by the oversampler during 10-fold cross-validation.

| Algorithm | Configuration | Strength |
| :--- | :--- | :--- |
| **Naive Bayes** | Standard | Fast, handles high dimensionality. |
| **Bagging** | 10 Iterations (J48) | Reduces variance and overfitting. |
| **AdaBoost** | 10 Iterations (J48) | Focuses on difficult-to-classify instances. |
| **SMO (SVM)** | PolyKernel + Normalize | Powerful for non-linear boundaries. |

### 4. Clustering Analysis
- **Algorithm**: `SimpleKMeans` (K=3).
- **Data**: Uses normalized numeric attributes (Manual Features).
- **Output**: Identifies user "Personas" (e.g., *Highly Active Engagers* vs *Passive Browsers*).

### 5. Association Rule Mining
- **Algorithm**: `Apriori` (Class Association Rules).
- **Discretization**: Uses a 3-bin strategy (Low, Medium, High).
- **Tuning**: Support = 2%, Confidence = 50% to ensure behavioral rules are discovered.

### 6. Model Persistence
- **Serialization**: The best-performing ensemble (Bagging/J48) is saved to the `/models` directory for deployment verification.
