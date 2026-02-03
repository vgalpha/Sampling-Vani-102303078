# Sampling Assignment - Credit Card Fraud Detection

**Author:** Vani Goyal  
**Roll No:** 102303078  
**Course:** UCS654 - Predictive Analytics Using Statistics

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Sampling Techniques](#sampling-techniques)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)
- [References](#references)


---

## Introduction

### What are Imbalanced Datasets?

An **imbalanced dataset** is one where the distribution of classes is not equal. One class has significantly more samples than the other. For example, if you have 1000 transactions and only 10 are fraudulent, that's an imbalanced dataset with a 100:1 ratio.

**Real-life examples where imbalanced datasets occur:**
- **Fraud detection:** Most transactions are legitimate, very few are fraudulent
- **Disease diagnosis:** Most patients are healthy, few have a specific disease
- **Spam email filtering:** Most emails are legitimate, fewer are spam
- **Manufacturing defects:** Most products pass quality checks, few are defective
- **Customer churn:** Most customers stay, only a small percentage leave

**The Problem:** If you train a model on imbalanced data, it can just predict the majority class every time and still get high accuracy. A model that says "no fraud" for every transaction would be 99% accurate but completely useless for catching fraud.

### What is Sampling?

**Sampling** is the process of selecting a subset of data from a larger dataset. Instead of using all available data, you pick a representative sample based on some technique or strategy.

**How sampling helps with imbalanced datasets:**
- **Balancing:** You can undersample the majority class or oversample the minority class to create balance
- **Better learning:** Models learn patterns from both classes instead of just memorizing the majority
- **Reduced bias:** Prevents models from being biased toward predicting only the majority class
- **Improved generalization:** Helps models perform better on real-world data where catching the minority class matters

Different sampling techniques select data differently, and this selection strategy can significantly impact how well your model learns.

### Project Objective

This project explores how different **sampling techniques** affect **machine learning model performance** on imbalanced credit card fraud data. The key questions we answer:

1. How do we balance a highly imbalanced dataset?
2. Which sampling technique works best for which machine learning model?
3. What is the optimal combination of sampling technique and model for fraud detection?

---

## Dataset Overview

The dataset comes from a credit card transaction database with the following characteristics:

| Attribute | Value |
|-----------|-------|
| **Total Samples** | 772 |
| **Legitimate Transactions (Class 0)** | 763 (98.83%) |
| **Fraudulent Transactions (Class 1)** | 9 (1.17%) |
| **Imbalance Ratio** | 84.8:1 |
| **Features** | 30 (V1-V28, Time, Amount, Class) |

**Challenge:** With only 9 fraud cases out of 772 transactions, this is a severely imbalanced dataset. A naive model that predicts "no fraud" for every transaction would achieve 98.83% accuracy while being completely useless.

---

## Methodology

### Overview

The complete workflow consists of these steps:

```
Step 1: Load imbalanced dataset (772 samples)
   ↓
Step 2: Balance the dataset
   ↓
Step 3: Create 5 samples using different sampling techniques
   ↓
Step 4: Train 5 ML models on each sample
   ↓
Step 5: Evaluate and compare 25 combinations
```

---

### Step 1: Data Balancing

To handle the severe class imbalance, this project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced dataset:

#### **SMOTE Balancing Approach**

**Method:**
- Use **SMOTE** to generate synthetic fraud cases from the 9 original minority samples
- Oversample minority class: 9 → 100 fraud cases
- Balance majority class: 763 → 100 legitimate cases (randomly sampled)
- **Result:** Balanced dataset with 200 samples (100 fraud + 100 legitimate)

**How SMOTE Works:**

SMOTE creates synthetic samples by:
1. Finding k-nearest neighbors of minority class samples
2. Randomly selecting one of these neighbors
3. Creating a new sample along the line between the original and the neighbor

This generates realistic fraud cases based on existing patterns, rather than simple duplication.

---

### Step 2: Sample Size Calculation

Using **Cochran's formula** for sample size calculation:

```
n₀ = (Z² × p × (1-p)) / E²
```

Where:
- **Z** = 1.96 (for 95% confidence level)
- **p** = 0.5 (balanced dataset proportion)
- **E** = 0.05 (5% margin of error)

**Finite Population Correction:**
```
n = n₀ / (1 + (n₀ - 1) / N)
```

Where N is the balanced population size.

**Result:**
- For N=200 (our SMOTE-balanced dataset): Sample size ≈ 132

Each of the five sampling techniques creates a sample of this calculated size from the balanced dataset.

---

## Sampling Techniques

### Sampling1: Simple Random Sampling

**Method:** Randomly select samples from the balanced dataset without replacement.

**How it works:** Like drawing names from a hat - each sample has an equal probability of being selected.

```python
sample = df.sample(n=sample_size, random_state=42, replace=False)
```

**Characteristics:**
- Unbiased
- Easy to implement
- Baseline for comparison
- No special structure

**Use case:** General-purpose sampling when you have no prior knowledge about data structure.

---

### Sampling2: Stratified Sampling

**Method:** Maintain the same class proportions as the original balanced dataset.

**How it works:** If the balanced dataset is 50% fraud and 50% legitimate, the sample maintains this exact ratio.

```python
# Sample proportionally from each class
fraud_sample = fraud_data.sample(n=n_fraud, random_state=42)
legit_sample = legit_data.sample(n=n_legit, random_state=42)
sample = pd.concat([fraud_sample, legit_sample])
```

**Characteristics:**
- Preserves class distribution
- Ensures minority class representation
- Reduces sampling bias
- Best for larger datasets

**Use case:** When you want to ensure all classes are represented proportionally in your sample.

---

### Sampling3: Systematic Sampling

**Method:** Select every k-th element after a random start.

**How it works:**
1. Calculate interval k = Population_size / Sample_size
2. Choose random starting point (0 to k-1)
3. Select every k-th element thereafter

```python
k = population_size // sample_size
start = random.randint(0, k)
indices = range(start, population_size, k)
sample = df.iloc[indices]
```

**Characteristics:**
- Easy to implement
- Provides good coverage
- Deterministic after start point
- Risk of periodicity bias

**Use case:** When data is randomly ordered and you want evenly distributed samples.

---

### Sampling4: Cluster Sampling

**Method:** Group data into clusters, then randomly select entire clusters.

**How it works:**
1. Use K-Means clustering to group similar transactions
2. Randomly select entire clusters
3. Include all samples from selected clusters

```python
# Create clusters using K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Randomly select clusters
selected_clusters = random.choice(clusters, size=n_clusters_needed)
sample = df[df['Cluster'].isin(selected_clusters)]
```

**Characteristics:**
- Captures natural groupings
- Efficient for large datasets
- May have higher variance
- Good for geographically dispersed data

**Use case:** When data has natural groupings and you want to capture group-level patterns.

---

### Sampling5: Bootstrap Sampling

**Method:** Random sampling **with replacement** - same sample can appear multiple times.

**How it works:** Draw samples randomly, but after each draw, the sample is "put back" so it can be drawn again.

```python
sample = df.sample(n=sample_size, random_state=42, replace=True)
```

**Characteristics:**
- Allows duplicates
- Foundation of ensemble methods
- Reduces variance
- Better for small datasets
- Used internally by Random Forest

**Use case:** When you want to estimate model stability or when working with small datasets.

---

## Machine Learning Models

I selected five diverse models representing different learning paradigms:

### M1: Logistic Regression
**Type:** Linear, Probabilistic

**How it works:** Finds a linear decision boundary to separate classes using the logistic function.

**Characteristics:**
- Fast training and prediction
- Interpretable coefficients
- Works well when classes are linearly separable
- Requires feature scaling for best results

**Parameters used:**
```python
LogisticRegression(max_iter=1000, random_state=42)
```

---

### M2: Decision Tree
**Type:** Tree-based, Non-parametric

**How it works:** Builds a tree of if-then rules by recursively splitting data based on features.

**Characteristics:**
- Easy to interpret and visualize
- Handles non-linear relationships
- No feature scaling needed
- Prone to overfitting if not constrained
- High variance with different samples

**Parameters used:**
```python
DecisionTreeClassifier(max_depth=10, random_state=42)
```

---

### M3: Random Forest
**Type:** Ensemble, Tree-based

**How it works:** Creates multiple decision trees using bootstrap samples and averages their predictions.

**Characteristics:**
- More robust than single trees
- Handles overfitting better
- Internally uses bootstrap sampling
- Feature importance available
- Slower than single trees

**Parameters used:**
```python
RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
```

---

### M4: Support Vector Machine (SVM)
**Type:** Margin-based, Kernel

**How it works:** Finds the optimal hyperplane that maximizes the margin between classes.

**Characteristics:**
- Effective in high-dimensional spaces
- Works well with clear margin of separation
- Memory efficient
- Requires feature scaling
- Sensitive to parameter tuning

**Parameters used:**
```python
SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
```

---

### M5: Naive Bayes
**Type:** Probabilistic, Generative

**How it works:** Applies Bayes' theorem with "naive" independence assumptions between features.

**Characteristics:**
- Very fast training and prediction
- Works well with small datasets
- Assumes feature independence (rarely true)
- Provides probability estimates
- Less accurate but computationally efficient

**Parameters used:**
```python
GaussianNB()
```

---

## Results

### Training & Evaluation Process

For each of the **25 combinations** (5 sampling techniques × 5 models):

1. **Split sample:** 80% training, 20% testing (stratified when possible)
2. **Scale features:** Apply StandardScaler for SVM
3. **Train model:** Fit on training set
4. **Predict:** Make predictions on test set
5. **Calculate accuracy:** Compare predictions with actual labels

### Accuracy Matrix

**Configuration Used:** SMOTE with 200 balanced samples (100 fraud + 100 legitimate)

|                          | Sampling1<br>(Simple Random) | Sampling2<br>(Stratified) | Sampling3<br>(Systematic) | Sampling4<br>(Cluster) | Sampling5<br>(Bootstrap) |
|--------------------------|------------------------------|---------------------------|---------------------------|------------------------|--------------------------|
| M1 (Logistic Regression) |   85.19%  |   88.89%  |   96.30%  |   81.48%  |   88.89%  |
| M2 (Decision Tree)       |   88.89%  |   96.30%  |   92.59%  |   74.07%  |   92.59%  |
| M3 (Random Forest)       |   96.30%  |  100.00%  |  100.00%  |   88.89%  |   96.30%  |
| M4 (SVM)                 |   96.30%  |   96.30%  |  100.00%  |   88.89%  |   88.89%  |
| M5 (Naive Bayes)         |   62.96%  |   81.48%  |   77.78%  |   81.48%  |   81.48%  |

### Best Sampling Technique for Each Model

| Model | Best Sampling Technique | Accuracy |
|-------|------------------------|----------|
| M1 (Logistic Regression) | Sampling3 (Systematic) | 96.30% |
| M2 (Decision Tree) | Sampling2 (Stratified) | 96.30% |
| M3 (Random Forest) | Sampling2 (Stratified) / Sampling3 (Systematic) | 100.00% |
| M4 (SVM) | Sampling3 (Systematic) | 100.00% |
| M5 (Naive Bayes) | Sampling2 (Stratified) / Sampling4 (Cluster) / Sampling5 (Bootstrap) | 81.48% |

### Best Model for Each Sampling Technique

| Sampling Technique | Best Model | Accuracy |
|-------------------|------------|----------|
| Sampling1 (Simple Random) | M3 (Random Forest) | 96.30% |
| Sampling2 (Stratified) | M3 (Random Forest) | 100.00% |
| Sampling3 (Systematic) | M3 (Random Forest) / M4 (SVM) | 100.00% |
| Sampling4 (Cluster) | M3 (Random Forest) | 88.89% |
| Sampling5 (Bootstrap) | M3 (Random Forest) | 96.30% |

### Best Model Overall

**🏆 M3 (Random Forest)**

| Metric | Value |
|--------|-------|
| Average Accuracy | 96.30% |
| Accuracy Range | 88.89% - 100.00% |
| Peak Performance | 100.00% (Sampling2 & Sampling3) |
| Consistency | Highest - performed best in all 5 sampling techniques |

Random Forest demonstrated superior and consistent performance across all sampling techniques, making it the most reliable model for this fraud detection task.

### Best Sampling Technique Overall

**🥇 Sampling3 (Systematic Sampling)**

| Metric | Value |
|--------|-------|
| Average Accuracy | 93.33% |
| Best Model Performance | 100.00% (Random Forest & SVM) |
| Models with 90%+ | 3 out of 5 models |

**🥈 Sampling2 (Stratified Sampling) - Close Second**

| Metric | Value |
|--------|-------|
| Average Accuracy | 92.59% |
| Best Model Performance | 100.00% (Random Forest) |
| Models with 90%+ | 3 out of 5 models |

Both Systematic and Stratified sampling techniques produced excellent results by ensuring good coverage and maintaining class balance in the samples.

### Overall Best Combination

**⭐ Winner:** M3 (Random Forest) + Sampling2 (Stratified) = **100.00%** accuracy

---

## Project Structure

```
sampling-assignment/
│
├── main.py                    # Main implementation
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── results.csv               # Generated results (after running)
└── Creditcard_data.csv       # Downloaded dataset (after running)
```


## How to Run

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Install required packages:**


Use requirements.txt:
```bash
pip install -r requirements.txt
```

### Configuration

Open `main.py` and modify the balanced dataset size if desired:

```python
# Set target balanced dataset size
BALANCED_DATASET_SIZE = 200  # Try 200, 400, etc.
```

### Execution

Run the main script:
```bash
python main.py
```

The script will:
1. Download the dataset automatically from GitHub
2. Display dataset statistics and class distribution
3. Balance the dataset using SMOTE
4. Create 5 samples using different techniques
5. Train 25 model-sample combinations
6. Display results matrix and analysis
7. Save results to `results.csv`

### Output Files

- `results.csv` - Accuracy matrix


---

## Conclusion

This project demonstrates that:

1. **Sampling technique selection significantly impacts model performance**, especially with imbalanced datasets
2. **Different models respond differently to sampling strategies** - there's no one-size-fits-all
3. **SMOTE effectively handles severe class imbalance** by creating synthetic minority samples
4. **Ensemble methods (Random Forest) are generally more robust** across sampling techniques

---

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Imbalanced-learn Documentation: https://imbalanced-learn.org/

