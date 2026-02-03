# Sampling Assignment - Credit Card Fraud Detection

**Author:** Vani Goyal
**Roll No:** 102303078
**Course:** UCS654

## Introduction

### Understanding Imbalanced Datasets

An imbalanced dataset is one where the distribution of classes is not equal. One class has significantly more samples than the other. For example, if you have 1000 transactions and only 10 are fraudulent, that's an imbalanced dataset with a 100:1 ratio.

**Real-life examples where imbalanced datasets occur:**
- **Fraud detection:** Most transactions are legitimate, very few are fraudulent
- **Disease diagnosis:** Most patients are healthy, few have a specific disease
- **Spam email filtering:** Most emails are legitimate, fewer are spam
- **Manufacturing defects:** Most products pass quality checks, few are defective
- **Customer churn:** Most customers stay, only a small percentage leave

The problem? If you train a model on imbalanced data, it can just predict the majority class every time and still get high accuracy. A model that says "no fraud" for every transaction would be 99% accurate but completely useless.

### What is Sampling?

Sampling is the process of selecting a subset of data from a larger dataset. Instead of using all available data, you pick a representative sample based on some technique or strategy.

**How sampling helps with imbalanced datasets:**
- **Balancing:** You can undersample the majority class or oversample the minority class to create balance
- **Better learning:** Models learn patterns from both classes instead of just memorizing the majority
- **Reduced bias:** Prevents models from being biased toward predicting only the majority class
- **Improved generalization:** Helps models perform better on real-world data where catching the minority class matters

Different sampling techniques select data differently, and this selection strategy can significantly impact how well your model learns.

### What This Project Does

This project explores how different sampling techniques affect machine learning model performance. I worked with a credit card fraud detection dataset that's heavily imbalanced (85:1 ratio). Here's what I did:

1. Balanced the dataset using undersampling
2. Applied five different sampling techniques to create five different samples
3. Trained five different ML models on each sample
4. Compared all 25 combinations to see which sampling technique works best with which model

The main question: Does the way we sample our data really matter? Spoiler: Yes, it matters a lot.

## Dataset Overview

The dataset comes from a credit card transaction database with 772 records. Here's what makes it challenging:

- **Total samples:** 772
- **Legitimate transactions (Class 0):** 763 (98.8%)
- **Fraudulent transactions (Class 1):** 9 (1.2%)
- **Imbalance ratio:** About 85:1

The dataset has 30 features: V1-V28 (28 features), Time and Amount.

This kind of extreme imbalance is common in fraud detection. If you train a model on this as-is, it could just predict "legitimate" for everything and still be right 98.8% of the time - but completely useless for catching fraud.

## Methodology

### Step 1: Balancing the Dataset

First, I needed to balance the dataset as per the assignment requirement. I used random undersampling - the simplest approach:

- Minority class (Fraud): 9 samples
- Majority class (Legitimate): 763 samples
- **Solution:** Randomly picked 9 legitimate transactions to match the 9 fraudulent ones
- **Result:** 18 samples total (9 fraud + 9 legitimate)

Yes, this throws away a lot of data. But for this assignment, the goal was to create a balanced base dataset to then apply different sampling techniques on.

### Step 2: Calculating Sample Size

Using the standard statistical formula for sample size:

```
n = (Z² × p × (1-p)) / E²
```

Where:
- Z = 1.96 (for 95% confidence)
- p = 0.5 (we have balanced data now)
- E = 0.05 (5% margin of error)

This gave me a sample size of **17** from my population of 18. So each sampling technique creates a sample of 17 records.

### Step 3: Five Sampling Techniques

Here's what each sampling technique does:

**Sampling1 - Simple Random Sampling**

Just randomly pick 17 samples from the 18 available. Every sample has an equal chance. Think of it like drawing names from a hat.

```python
sample = df.sample(n=17, random_state=42)
```

**Sampling2 - Stratified Sampling**

This tries to maintain the same class ratio as the original. Since we have 9:9 ratio, the sample should have roughly the same proportion. I ended up with 9 fraud cases and 8 legitimate ones.

The idea: If your population has certain groups, your sample should represent those groups proportionally.

**Sampling3 - Systematic Sampling**

Pick every kth element. With 18 samples and needing 17, k=1 (every element). Started at a random position and then selected every 1st element systematically.

Example: If k=2, and you start at position 1, you'd pick positions 1, 3, 5, 7, 9...

**Sampling4 - Cluster Sampling**

First, I grouped the data into clusters using K-Means (based on transaction features). Then randomly selected entire clusters until I had about 17 samples.

Think of it like: instead of picking individual students from a school, you pick entire classrooms.

For this small dataset, I used 6 clusters and selected 6 of them.

**Sampling5 - Bootstrap Sampling**

Sample with replacement - meaning the same transaction can appear multiple times. Drew 17 samples, but because of replacement, only 10 were unique.

This is the foundation of techniques like Random Forest. The duplicates actually help by emphasizing certain patterns.

### Step 4: Five Machine Learning Models

I chose five different types of models to get a diverse comparison:

**M1 - Logistic Regression**
The classic linear classifier. Fast, simple, works great when classes are somewhat linearly separable.

**M2 - Decision Tree**
Builds a tree of if-then rules. Easy to interpret and doesn't need feature scaling.

**M3 - Random Forest**
An ensemble of 100 decision trees. More robust than a single tree, less likely to overfit.

**M4 - Support Vector Machine (SVM)**
Tries to find the best boundary between classes. I used the RBF kernel which can handle non-linear patterns.

**M5 - Naive Bayes**
A probabilistic model based on Bayes theorem. Fast and works well with small datasets.

### Step 5: Training and Evaluation

For each of the 25 combinations (5 samples × 5 models):

1. Split the sample: 80% training, 20% testing (with stratification)
2. Train the model
3. Test and calculate accuracy
4. Store the result

## Results

Here's the accuracy matrix (in percentages):

|                          | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|--------------------------|-----------|-----------|-----------|-----------|-----------|
| M1 (Logistic Regression) | 75        | 25        | 75        | 75        | **100**   |
| M2 (Decision Tree)       | **100**   | 75        | 75        | **100**   | **100**   |
| M3 (Random Forest)       | 25        | 50        | 75        | 25        | **100**   |
| M4 (SVM)                 | 50        | 25        | 75        | 50        | **100**   |
| M5 (Naive Bayes)         | 25        | 0         | 75        | 25        | **100**   |

### What's the best sampling for each model?

- **Logistic Regression:** Sampling5 (100%)
- **Decision Tree:** Sampling1 (100%) - though Sampling4 and 5 also got 100%
- **Random Forest:** Sampling5 (100%)
- **SVM:** Sampling5 (100%)
- **Naive Bayes:** Sampling5 (100%)

Pattern? Sampling5 (Bootstrap) wins for almost everything.

## Analysis and Discussion

### The Bootstrap Winner

Sampling5 achieved perfect 100% accuracy across all five models. Why did bootstrap sampling dominate?

With only 18 samples in the balanced dataset and needing 17 for each sample, bootstrap sampling with replacement does something clever. It can pick the same transaction multiple times, which means:

1. **More diversity in training:** The duplicates act like emphasis on certain patterns
2. **Natural data augmentation:** Seeing the same sample multiple times helps models learn better
3. **Variance reduction:** This is why Random Forest (which uses bootstrapping internally) works so well

Think of it this way: if you're studying for an exam and you have 18 practice problems, seeing some problems twice might actually help you learn those patterns better than seeing each problem exactly once.

### The Systematic Surprise

Sampling3 (Systematic) got exactly 75% across all five models. Not great, not terrible - but remarkably consistent.

This makes sense because systematic sampling provides good coverage. By selecting at regular intervals, you ensure the sample is spread across the entire dataset. No model gets an advantage or disadvantage - everyone gets the same quality of data.

### The Stratified Failure

Sampling2 (Stratified) performed worst. Naive Bayes even scored 0% - a complete failure.

Why? With such a tiny dataset (18 samples), trying to maintain exact proportions (9:8 split) probably created an unfortunate train-test split. When you only have 3-4 test samples, getting them all from one difficult region can tank your accuracy.

Lesson: Stratified sampling is great for larger datasets, but with extremely small samples, it can backfire.

### Model-Specific Observations

**Decision Tree is the most robust:**
It scored 75% or higher on every sampling technique. Trees adapt well to different data distributions.

**Naive Bayes is the most fragile:**
Range from 0% to 100%. Probabilistic models are sensitive to sample quality because they estimate probabilities from the data. Bad sample = bad probability estimates = bad predictions.

**Random Forest's weird behavior:**
You'd expect the ensemble method to be robust, but it ranged from 25% to 100%. However, it did perfect with Bootstrap (its natural habitat), which makes sense since Random Forest internally uses bootstrap sampling.

**SVM and Logistic Regression:**
Both struggled with Simple Random and Stratified sampling but excelled with Bootstrap. These models benefit from seeing repeated patterns in the data.

### The Sample Size Problem

Let's be real: 18 samples is tiny. With test sets of only 3-4 samples, accuracy can only be 0%, 25%, 50%, 75%, or 100%. A single misclassification drops you by 25%.

This amplifies the importance of sampling technique. In normal circumstances with hundreds or thousands of samples, the differences would be smaller. But here, it's the difference between complete failure and perfect accuracy.

### Practical Takeaways

What did I learn from this experiment?

1. **For very small datasets, use bootstrap sampling.** It consistently outperformed other techniques.

2. **Don't blindly trust stratified sampling with tiny samples.** It's great in theory but can fail when you have very few data points.

3. **Systematic sampling is the "safe choice"** if you want consistent, middle-of-the-road performance.

4. **Model choice matters, but sampling matters more** (at least for small datasets). The same model went from 0% to 100% just by changing the sampling technique.

5. **Decision Trees are your robust friend.** They performed well across different sampling strategies.

## Conclusion

This assignment clearly demonstrated that sampling technique selection is critical when working with small, balanced datasets.

Bootstrap sampling emerged as the clear winner, achieving perfect accuracy across all models. This makes sense theoretically - bootstrap sampling's use of replacement provides better training data quality for small samples.

The most surprising finding was how badly stratified sampling performed. This goes against conventional wisdom but highlights an important lesson: techniques that work well on large datasets don't always translate to very small samples.

For anyone working on fraud detection or similar imbalanced problems with limited data: balance your dataset first, then use bootstrap sampling to create your training sets. Pair it with Decision Trees or Random Forest for the most robust results.

## How to Run

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python main.py
```

The script will:
- Download the dataset automatically
- Balance it
- Create all 5 samples
- Train all 25 model-sample combinations
- Print results and analysis
- Save results to `results.csv`

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
requests>=2.28.0
```

## Files in This Repository

- `main.py` - Complete implementation
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `Sampling_Assignment.pdf` - Assignment instructions
- `Creditcard_data.csv` - Dataset (downloaded automatically)
- `results.csv` - Results matrix

---

**Vani Goyal**
**102303078**
**UCS654 - Predictive Analytics Using Statistics**
