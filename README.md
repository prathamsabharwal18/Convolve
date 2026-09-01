# Credit Card Default Risk Prediction — Convolve 3.0

A machine learning pipeline for predicting credit-card default risk from high-dimensional customer behavioral, transaction, and credit-bureau data.

The project was developed for **Convolve 3.0**, a Pan-IIT AI/ML Hackathon, and focuses on building an imbalance-aware **Behavior Score Model** capable of identifying customers with an elevated probability of future default.

---

## Project Overview

Credit-risk prediction is an inherently imbalanced classification problem: the number of customers who default is significantly smaller than the number who do not.

A model that simply predicts every customer as a non-defaulter can therefore achieve very high accuracy while providing almost no practical value.

This project addresses that problem through:

- Exploratory Data Analysis
- Missing-value analysis and imputation
- Correlation-based feature reduction
- High-dimensional feature selection
- Categorical feature encoding
- Feature scaling
- Imbalanced-learning techniques
- Random Forest and Gradient Boosted Decision Trees
- Custom imbalance-aware model evaluation

The target variable is:

```text
bad_flag = 0  → Non-defaulter
bad_flag = 1  → Potential defaulter
```

The objective is not only to classify customers but also to estimate their **probability of default**, enabling risk-based customer prioritization.

---

## Dataset

The development dataset contains:

```text
96,806 customers
1,216 columns
```

The validation dataset contains:

```text
41,792 customers
```

The features represent multiple aspects of customer financial behavior, including:

```text
onus_attribute_*
transaction_attribute_*
bureau_*
bureau_enquiry_*
```

along with:

```text
account_number
bad_flag
```

where `bad_flag` is the binary target variable.

The dataset presents three major challenges:

1. **Extreme class imbalance**
2. **Large number of missing values**
3. **Very high dimensionality**

---

## End-to-End Pipeline

```text
Raw Customer Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Missing-Value Analysis
        │
        ▼
Feature Selection
   ├── High-Missing Features
   └── Correlation Filtering
        │
        ▼
Missing-Value Imputation
        │
        ▼
Categorical Encoding
        │
        ▼
Z-Score Normalization
        │
        ▼
Imbalance Handling
   ├── Random Oversampling
   ├── Borderline-SMOTE
   ├── SMOTE
   ├── ADASYN
   ├── SMOTE-Tomek
   └── SMOTE-NC
        │
        ▼
Model Training
   ├── Random Forest
   └── Gradient Boosted Trees
        │
        ▼
Model Evaluation
   ├── ROC-AUC
   ├── F1 Score
   ├── G-Mean
   ├── Log Loss
   └── IAS
```

---

# Exploratory Data Analysis

The initial dataset contains more than a thousand features, making direct modeling computationally expensive and susceptible to redundant information.

The notebook analyzes the major feature groups independently to understand:

- missing-value percentages
- feature distributions
- correlation between attributes
- numerical and categorical columns
- redundant features

---

## Missing-Value Handling

Several columns contain substantial amounts of missing data.

Features with excessive missing values are removed to avoid introducing unreliable information into the model.

For remaining features, missing values are handled through preprocessing and **Iterative Imputation**.

### Iterative Imputer

Instead of replacing missing values using only the mean or median, Iterative Imputer estimates each incomplete feature using the other available features.

Conceptually:

```text
Feature A is missing

        ↓

Use Features B, C, D, ... to estimate A

        ↓

Repeat iteratively
```

The validation data is processed using:

```python
IterativeImputer(
    max_iter=50,
    random_state=42
)
```

This allows relationships between financial attributes to contribute to missing-value estimation.

---

# Correlation-Based Feature Selection

The transaction data contains a large number of highly correlated features.

Keeping strongly correlated variables:

- increases dimensionality
- adds redundant information
- increases computational cost
- may reduce model interpretability

The project calculates the **Pearson correlation matrix** and identifies feature pairs satisfying:

```text
|correlation| > 0.75
```

Redundant transaction attributes are removed.

This step eliminates hundreds of highly correlated features.

---

## Dimensionality Reduction

The original dataset contains:

```text
1,216 columns
```

After missing-value filtering, correlation analysis and feature preprocessing, the modeling dataset is reduced to approximately:

```text
130 features
```

This represents a substantial reduction in dimensionality while retaining useful behavioral information.

---

# Data Type Optimization

The project also reduces memory usage by converting:

```text
float64 → float32
int64   → int32
```

For the validation dataset, this reduced memory usage from approximately:

```text
41.5 MB → 20.7 MB
```

while preserving the information required for model training and inference.

---

# Categorical Feature Encoding

Categorical features are transformed using:

```python
LabelEncoder
```

Each category is represented using an integer.

For example:

```text
Category A → 0
Category B → 1
Category C → 2
```

The same encoding scheme is then applied to the validation data.

---

# Feature Scaling

Numerical features can have very different scales.

For example:

```text
Feature A → 0–10
Feature B → 0–1,000,000
```

The project applies **Z-Score Normalization** using `StandardScaler`.

For every feature:

```text
z = (x - μ) / σ
```

where:

```text
x = original value
μ = feature mean
σ = feature standard deviation
```

After normalization, numerical features are centered approximately around:

```text
mean = 0
standard deviation = 1
```

---

# Handling Class Imbalance

Credit-default datasets contain far fewer defaulters than non-defaulters.

This creates a major problem.

Suppose:

```text
98 customers → non-default
 2 customers → default
```

A classifier predicting:

```text
everyone → non-default
```

would achieve:

```text
98% accuracy
```

despite identifying **zero actual defaulters**.

The baseline Random Forest demonstrates this behavior: it achieves very high accuracy but effectively fails on the minority class.

Therefore, the project evaluates several resampling strategies.

---

## Random Oversampling

Random Oversampling duplicates minority-class samples until the class distribution becomes more balanced.

```text
Before

Class 0: ███████████████████
Class 1: █

After

Class 0: ███████████████████
Class 1: ███████████████████
```

Advantage:

- simple
- preserves original minority examples

Disadvantage:

- can overfit duplicated samples

---

## SMOTE

**Synthetic Minority Over-sampling Technique** generates new minority samples instead of simply duplicating existing ones.

For two minority samples:

```text
x₁
x₂
```

a synthetic point can be generated as:

```text
x_new = x₁ + λ(x₂ - x₁)
```

where:

```text
0 ≤ λ ≤ 1
```

This generates new examples between existing minority-class observations.

---

## Borderline-SMOTE

Borderline-SMOTE focuses mainly on minority samples located near the decision boundary.

These samples are more difficult to classify and therefore more useful for creating synthetic examples.

---

## ADASYN

**Adaptive Synthetic Sampling** generates more synthetic samples in difficult regions of the feature space.

Unlike ordinary SMOTE, ADASYN gives additional attention to minority samples surrounded by majority-class observations.

---

## SMOTE-Tomek

SMOTE-Tomek combines:

```text
SMOTE
  +
Tomek Links
```

SMOTE creates synthetic minority observations.

Tomek Links then identify overlapping majority/minority samples located close to the decision boundary and remove noisy observations.

This produces a cleaner and more balanced training dataset.

---

## SMOTE-NC

SMOTE-NC extends SMOTE to datasets containing both:

```text
Numerical features
+
Categorical features
```

It avoids treating categorical variables as ordinary continuous numbers when constructing synthetic samples.

---

# Models

## 1. Random Forest — Baseline

A Random Forest classifier with:

```python
n_estimators = 100
```

is used as the initial baseline.

### Baseline Results

```text
Accuracy : 98.47%
ROC-AUC  : 0.6881
IAS      : 0.2235
```

Despite the high accuracy, the model predicts almost no minority-class examples correctly.

This demonstrates why **accuracy alone is misleading for highly imbalanced datasets**.

---

# 2. Gradient Boosted Decision Trees

The primary model is implemented using:

```text
TensorFlow Decision Forests
GradientBoostedTreesModel
```

Gradient Boosting constructs decision trees sequentially.

Each new tree attempts to reduce errors made by the existing ensemble.

Conceptually:

```text
Model₁ = Tree₁

Model₂ = Tree₁ + Tree₂

Model₃ = Tree₁ + Tree₂ + Tree₃

...

Final Model = Σ Trees
```

This allows the model to progressively learn difficult customer patterns.

---

# Evaluation Metrics

Because the dataset is highly imbalanced, several metrics are considered.

---

## ROC-AUC

ROC-AUC measures the model's ability to rank positive examples above negative examples across classification thresholds.

```text
AUC = 0.5 → random classifier
AUC = 1.0 → perfect classifier
```

---

## Precision

```text
Precision = TP / (TP + FP)
```

It answers:

> Of all customers predicted as defaulters, how many actually defaulted?

---

## Recall

```text
Recall = TP / (TP + FN)
```

It answers:

> Of all actual defaulters, how many were successfully detected?

For credit-risk applications, recall is especially important because missing a risky customer can be costly.

---

## F1 Score

F1 balances Precision and Recall.

```text
F1 = 2 × Precision × Recall
     ----------------------
      Precision + Recall
```

---

## G-Mean

G-Mean measures balanced performance across both classes.

```text
G-Mean = √(Sensitivity × Specificity)
```

A high G-Mean indicates that the model performs well on both default and non-default customers.

---

## Log Loss

Log Loss evaluates the quality of predicted probabilities rather than only the final class prediction.

Confident but incorrect predictions receive a larger penalty.

---

# Imbalance-Aware Score — IAS

A custom evaluation metric was designed to evaluate models across multiple dimensions.

```text
IAS =
0.35 × ROC-AUC
+ 0.30 × F1
+ 0.25 × G-Mean
- 0.10 × Log Loss
```

The metric rewards:

- strong class separation
- balanced precision and recall
- balanced sensitivity and specificity

while penalizing poorly calibrated probability predictions.

This provides a more meaningful comparison than accuracy alone.

---

# Experimental Results

| Approach | Accuracy | ROC-AUC | F1 Score | IAS |
|----------|----------|---------|----------|-----|
| Random Forest Baseline | 98.47% | 0.6881 | Poor minority detection | 0.2235 |
| GB Trees — No Oversampling | 98.58% | 0.9006 | 0.1404 | 0.4210 |
| Borderline-SMOTE + GBT | 98.58%* | 0.9065 | 0.0651 | 0.2296 |
| Random Oversampling + GBT | 92.07% | 0.9767 | 0.9280 | 0.8245 |
| SMOTE + GBT | 94.33% | 0.9872 | 0.9445 | 0.8479 |
| ADASYN + GBT | 94.32% | **0.9872** | 0.9441 | 0.8478 |
| SMOTE-Tomek + GBT | **94.35%** | 0.9869 | **0.9447** | **0.8480** |

\*Values correspond to the experiments recorded in the notebook.

The experiments demonstrate that optimizing purely for accuracy is inappropriate for this task.

Oversampling-based Gradient Boosted Trees substantially improve:

```text
ROC-AUC
F1 Score
G-Mean
IAS
```

compared with the baseline classifier.

Among the recorded experiments, **SMOTE-Tomek + Gradient Boosted Trees** achieved the strongest overall IAS and F1 score.

---

# Why Accuracy Fell but the Model Improved

The baseline model achieves approximately:

```text
98.5% accuracy
```

while the balanced models achieve around:

```text
94% accuracy
```

This does **not** mean the baseline is better.

The baseline obtains high accuracy primarily by predicting the majority class.

After imbalance handling, the model becomes much better at identifying actual defaulters.

Therefore:

```text
Accuracy ↓

but

Minority Detection ↑
F1 ↑
ROC-AUC ↑
G-Mean ↑
IAS ↑
```

which is much more valuable for credit-risk assessment.

---

# Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- TensorFlow
- TensorFlow Decision Forests

### Imbalanced Learning

- imbalanced-learn
- SMOTE
- Borderline-SMOTE
- ADASYN
- SMOTE-Tomek
- SMOTE-NC
- Random Oversampling

### Preprocessing

- Iterative Imputer
- StandardScaler
- LabelEncoder
- Pearson Correlation Analysis

### Environment

- Jupyter Notebook
- Kaggle
- GPU-enabled Kaggle runtime

---

# Repository Structure

```text
Convolve/
│
├── Final_Notebook.ipynb
│   ├── Data Loading
│   ├── Exploratory Data Analysis
│   ├── Missing-Value Analysis
│   ├── Correlation Analysis
│   ├── Feature Selection
│   ├── Data Imputation
│   ├── Feature Encoding
│   ├── Feature Scaling
│   ├── Random Forest Baseline
│   ├── Gradient Boosted Trees
│   ├── Oversampling Experiments
│   └── Model Evaluation
│
└── README.md
```

---

# Running the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd Convolve
```

## 2. Install dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn tensorflow tensorflow-decision-forests matplotlib seaborn jupyter
```

## 3. Add the dataset

The notebook expects:

```text
Dev_data_to_be_shared.csv
validation_data_to_be_shared.csv
```

The original Kaggle paths used in the notebook are:

```text
/kaggle/input/convolve-3-0/Dev_data_to_be_shared.csv
/kaggle/input/convolve-3-0/validation_data_to_be_shared.csv
```

Update these paths when running locally.

## 4. Run the notebook

```bash
jupyter notebook Final_Notebook.ipynb
```

---

# Key Takeaways

This project demonstrates an end-to-end approach to a real-world financial machine-learning problem involving:

- almost 100K customer records
- more than 1,200 initial features
- extensive missing data
- highly correlated attributes
- extreme target imbalance
- dimensionality reduction
- advanced imputation
- multiple synthetic oversampling strategies
- tree-based ensemble models
- probability-based risk prediction
- imbalance-aware evaluation

A major takeaway from the project is that:

> **High accuracy does not necessarily mean a useful model when the target distribution is highly imbalanced.**

Model selection should instead consider the business objective and metrics such as ROC-AUC, Recall, F1, G-Mean and probability calibration.

---

# Possible Improvements

Several improvements can further strengthen the pipeline:

### 1. Resampling inside training folds

Apply SMOTE/ADASYN only after the train-test split or inside a cross-validation pipeline to prevent synthetic samples derived from evaluation observations from influencing training.

### 2. Stratified Cross-Validation

Replace a single holdout evaluation with:

```text
Stratified K-Fold Cross Validation
```

to obtain more robust performance estimates.

### 3. Threshold Optimization

The default classification threshold is:

```text
0.5
```

For credit risk, an optimal threshold can instead be selected based on:

- business cost
- recall
- F1
- G-Mean
- expected financial loss

### 4. Hyperparameter Optimization

Tune parameters such as:

- number of trees
- tree depth
- learning rate
- minimum samples per node
- regularization

using Random Search, Bayesian optimization or Optuna.

### 5. Probability Calibration

Since the final objective is a Behavior Score / probability of default, probability calibration using:

```text
Platt Scaling
Isotonic Regression
```

could make predicted probabilities more reliable.

### 6. Explainability

Add:

```text
SHAP
Feature Importance
Partial Dependence Plots
```

to identify the financial attributes that contribute most strongly to default risk.

### 7. Cost-Sensitive Learning

False negatives and false positives have very different financial consequences.

A production credit-risk model can incorporate these costs directly into training and threshold selection.

---

# Business Use Case

The resulting probability can be interpreted as a customer-level risk score.

```text
Customer Features
       │
       ▼
Behavior Score Model
       │
       ▼
Probability of Default
       │
       ├── Low Risk
       ├── Medium Risk
       └── High Risk
```

Such predictions can support:

- portfolio risk monitoring
- proactive customer intervention
- credit-limit management
- collections prioritization
- risk-based decision making

---

# Acknowledgement

This project was developed as part of **Convolve 3.0**, a Pan-IIT AI/ML Hackathon focused on solving real-world machine-learning and data-analytics problems.

---

## Author

**Pratham Sabharwal**  
Indian Institute of Technology Guwahati
