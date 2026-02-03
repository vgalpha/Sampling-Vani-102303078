"""
Sampling Assignment - Credit Card Fraud Detection
Author: Vani Goyal (102303078)
Objective: Analyze different sampling techniques on imbalanced dataset
"""

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==================== CONFIGURATION ====================
DATASET_URL = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
DATASET_FILE = "Creditcard_data.csv"
CONFIDENCE_LEVEL = 1.96  # 95% confidence
MARGIN_ERROR = 0.05
PROPORTION = 0.5  # For balanced dataset

# ==================== 1. DATA LOADING ====================

def download_dataset(url, filename):
  """Download dataset from GitHub URL."""
  print(f"Downloading dataset from {url}...")
  try:
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
      f.write(response.content)
    print(f"✓ Dataset downloaded successfully: {filename}")
    return True
  except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    return False

def load_and_explore_data(filename):
  """Load dataset and display basic information."""
  print(f"\n{'='*60}")
  print("DATASET EXPLORATION")
  print(f"{'='*60}")

  df = pd.read_csv(filename)

  print(f"\nDataset Shape: {df.shape}")
  print(f"Columns: {list(df.columns)}")
  print(f"\nData Types:\n{df.dtypes}")
  print(f"\nMissing Values:\n{df.isnull().sum()}")

  # Identify target column (typically 'Class' for fraud detection)
  target_col = 'Class' if 'Class' in df.columns else df.columns[-1]
  print(f"\nTarget Column: {target_col}")

  class_dist = df[target_col].value_counts()
  print(f"\nClass Distribution:\n{class_dist}")
  print(f"Imbalance Ratio: {class_dist.max() / class_dist.min():.2f}:1")

  return df, target_col

# ==================== 2. DATA BALANCING ====================

def balance_dataset(df, target_col):
  """Balance the imbalanced dataset using random undersampling."""
  print(f"\n{'='*60}")
  print("BALANCING DATASET")
  print(f"{'='*60}")

  # Separate majority and minority classes
  class_counts = df[target_col].value_counts()
  minority_class = class_counts.idxmin()
  majority_class = class_counts.idxmax()

  minority_samples = df[df[target_col] == minority_class]
  majority_samples = df[df[target_col] == majority_class]

  # Undersample majority class to match minority class
  majority_downsampled = majority_samples.sample(n=len(minority_samples), random_state=42)

  # Combine and shuffle
  balanced_df = pd.concat([minority_samples, majority_downsampled])
  balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

  print(f"Original Dataset Size: {len(df)}")
  print(f"Balanced Dataset Size: {len(balanced_df)}")
  print(f"\nBalanced Class Distribution:\n{balanced_df[target_col].value_counts()}")

  return balanced_df

# ==================== 3. SAMPLING TECHNIQUES ====================

def calculate_sample_size(population_size, confidence=1.96, margin=0.05, proportion=0.5):
  """Calculate sample size using statistical formula."""
  n = (confidence**2 * proportion * (1 - proportion)) / (margin**2)
  # Apply finite population correction
  n_adjusted = n / (1 + (n - 1) / population_size)
  return int(n_adjusted)

def sampling1_simple_random(df, sample_size):
  """Sampling1: Simple Random Sampling without replacement."""
  print("\n→ Sampling1: Simple Random Sampling")
  sample = df.sample(n=sample_size, random_state=42, replace=False)
  print(f"  Sample Size: {len(sample)}")
  return sample

def sampling2_stratified(df, target_col, sample_size):
  """Sampling2: Stratified Sampling maintaining class proportions."""
  print("\n→ Sampling2: Stratified Sampling")

  # Manual stratified sampling to maintain exact proportions
  classes = df[target_col].unique()
  class_counts = df[target_col].value_counts()

  samples_per_class = {}
  for cls in classes:
    proportion = class_counts[cls] / len(df)
    samples_per_class[cls] = max(1, int(sample_size * proportion))

  # Adjust if total doesn't match sample_size exactly
  total = sum(samples_per_class.values())
  if total < sample_size:
    samples_per_class[classes[0]] += (sample_size - total)
  elif total > sample_size:
    samples_per_class[classes[0]] -= (total - sample_size)

  # Sample from each class
  sample_list = []
  for cls in classes:
    cls_data = df[df[target_col] == cls]
    n_samples = min(samples_per_class[cls], len(cls_data))
    sample_list.append(cls_data.sample(n=n_samples, random_state=42))

  sample = pd.concat(sample_list).sample(frac=1, random_state=42).reset_index(drop=True)

  print(f"  Sample Size: {len(sample)}")
  print(f"  Class Distribution: {sample[target_col].value_counts().to_dict()}")
  return sample

def sampling3_systematic(df, sample_size):
  """Sampling3: Systematic Sampling with regular intervals."""
  print("\n→ Sampling3: Systematic Sampling")

  population_size = len(df)
  k = population_size // sample_size  # Sampling interval

  # Random start point
  start = np.random.randint(0, k)
  indices = list(range(start, population_size, k))[:sample_size]

  sample = df.iloc[indices].reset_index(drop=True)
  print(f"  Sample Size: {len(sample)}")
  print(f"  Sampling Interval (k): {k}")
  return sample

def sampling4_cluster(df, target_col, sample_size, n_clusters=10):
  """Sampling4: Cluster Sampling using K-Means."""
  print("\n→ Sampling4: Cluster Sampling")

  # Separate features for clustering
  X = df.drop(columns=[target_col])

  # Adjust number of clusters for small datasets
  n_clusters = min(n_clusters, len(df) // 2, max(2, len(df) // 3))

  # Apply K-Means clustering
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  df['Cluster'] = kmeans.fit_predict(X)

  # Calculate clusters to select based on sample size
  avg_cluster_size = len(df) // n_clusters
  num_clusters_to_select = max(1, min(n_clusters, sample_size // max(1, avg_cluster_size)))

  # Randomly select clusters
  selected_clusters = np.random.choice(range(n_clusters), size=num_clusters_to_select, replace=False)
  sample = df[df['Cluster'].isin(selected_clusters)].drop(columns=['Cluster']).reset_index(drop=True)

  # If sample is too small, add more clusters
  attempts = 0
  while len(sample) < sample_size and num_clusters_to_select < n_clusters and attempts < 10:
    num_clusters_to_select += 1
    selected_clusters = np.random.choice(range(n_clusters), size=num_clusters_to_select, replace=False)
    sample = df[df['Cluster'].isin(selected_clusters)].drop(columns=['Cluster']).reset_index(drop=True)
    attempts += 1

  # If sample is too large, randomly subsample
  if len(sample) > sample_size:
    sample = sample.sample(n=sample_size, random_state=42)

  print(f"  Sample Size: {len(sample)}")
  print(f"  Clusters Selected: {num_clusters_to_select}/{n_clusters}")
  return sample

def sampling5_bootstrap(df, sample_size):
  """Sampling5: Bootstrap Sampling with replacement."""
  print("\n→ Sampling5: Bootstrap Sampling (with replacement)")
  sample = df.sample(n=sample_size, random_state=42, replace=True)
  print(f"  Sample Size: {len(sample)}")
  print(f"  Unique Samples: {len(sample.drop_duplicates())}")
  return sample

def create_all_samples(balanced_df, target_col):
  """Create all 5 samples using different techniques."""
  print(f"\n{'='*60}")
  print("CREATING SAMPLES")
  print(f"{'='*60}")

  population_size = len(balanced_df)
  sample_size = calculate_sample_size(population_size, CONFIDENCE_LEVEL, MARGIN_ERROR, PROPORTION)

  print(f"\nPopulation Size: {population_size}")
  print(f"Calculated Sample Size: {sample_size}")

  samples = {
    'Sampling1': sampling1_simple_random(balanced_df, sample_size),
    'Sampling2': sampling2_stratified(balanced_df, target_col, sample_size),
    'Sampling3': sampling3_systematic(balanced_df, sample_size),
    'Sampling4': sampling4_cluster(balanced_df.copy(), target_col, sample_size),
    'Sampling5': sampling5_bootstrap(balanced_df, sample_size)
  }

  return samples

# ==================== 4. MACHINE LEARNING MODELS ====================

def get_models():
  """Initialize 5 ML models."""
  models = {
    'M1 (Logistic Regression)': LogisticRegression(max_iter=1000, random_state=42),
    'M2 (Decision Tree)': DecisionTreeClassifier(max_depth=10, random_state=42),
    'M3 (Random Forest)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'M4 (SVM)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'M5 (Naive Bayes)': GaussianNB()
  }
  return models

# ==================== 5. TRAINING & EVALUATION ====================

def train_and_evaluate(sample, target_col, model, model_name, needs_scaling=False):
  """Train model on sample and return accuracy."""

  # Separate features and target
  X = sample.drop(columns=[target_col])
  y = sample[target_col]

  # Split into train and test
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Scale features if needed (for SVM)
  if needs_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

  # Train model
  model.fit(X_train, y_train)

  # Predict and calculate accuracy
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred) * 100

  return accuracy

def evaluate_all_combinations(samples, target_col):
  """Evaluate all model-sampling combinations."""
  print(f"\n{'='*60}")
  print("TRAINING & EVALUATION")
  print(f"{'='*60}")

  models = get_models()
  results = pd.DataFrame(index=models.keys(), columns=samples.keys())

  for sampling_name, sample in samples.items():
    print(f"\n{sampling_name}:")
    for model_name, model in models.items():
      needs_scaling = 'SVM' in model_name
      accuracy = train_and_evaluate(sample, target_col, model, model_name, needs_scaling)
      results.loc[model_name, sampling_name] = accuracy
      print(f"  {model_name}: {accuracy:.2f}%")

  return results

# ==================== 6. RESULTS ANALYSIS ====================

def analyze_results(results):
  """Analyze and display results with best techniques per model."""
  print(f"\n{'='*60}")
  print("RESULTS ANALYSIS")
  print(f"{'='*60}")

  print("\n📊 ACCURACY MATRIX (%):\n")
  print(results.to_string())

  print("\n\n🏆 BEST SAMPLING TECHNIQUE FOR EACH MODEL:\n")
  for model in results.index:
    best_sampling = results.loc[model].astype(float).idxmax()
    best_accuracy = results.loc[model].astype(float).max()
    print(f"{model:30} → {best_sampling} ({best_accuracy:.2f}%)")

  print("\n\n🏆 BEST MODEL FOR EACH SAMPLING TECHNIQUE:\n")
  for sampling in results.columns:
    best_model = results[sampling].astype(float).idxmax()
    best_accuracy = results[sampling].astype(float).max()
    print(f"{sampling:15} → {best_model} ({best_accuracy:.2f}%)")

  # Overall best combination
  max_accuracy = results.astype(float).max().max()
  best_combo = results.astype(float).stack().idxmax()
  print(f"\n\n⭐ OVERALL BEST COMBINATION:")
  print(f"   {best_combo[0]} + {best_combo[1]} = {max_accuracy:.2f}%")

  return results

def save_results_to_csv(results, filename='results.csv'):
  """Save results to CSV file."""
  results.to_csv(filename)
  print(f"\n✓ Results saved to {filename}")

# ==================== 7. MAIN EXECUTION ====================

def main():
  """Main execution function."""
  print("\n" + "="*60)
  print("SAMPLING ASSIGNMENT - CREDIT CARD FRAUD DETECTION")
  print("="*60)

  # Step 1: Download and load dataset
  if not download_dataset(DATASET_URL, DATASET_FILE):
    return

  df, target_col = load_and_explore_data(DATASET_FILE)

  # Step 2: Balance dataset
  balanced_df = balance_dataset(df, target_col)

  # Step 3: Create 5 samples
  samples = create_all_samples(balanced_df, target_col)

  # Step 4 & 5: Train models and evaluate
  results = evaluate_all_combinations(samples, target_col)

  # Step 6: Analyze results
  final_results = analyze_results(results)

  # Save results
  save_results_to_csv(final_results)

  print("\n" + "="*60)
  print("✓ ASSIGNMENT COMPLETED SUCCESSFULLY")
  print("="*60 + "\n")

if __name__ == '__main__':
  main()
