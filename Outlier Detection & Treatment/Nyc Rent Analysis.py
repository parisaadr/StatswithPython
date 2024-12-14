import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, trim_mean
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load dataset (replace with actual file path if needed)
rentals = pd.read_csv("nyc_rentals.csv")

# ------------------ Data Cleaning ------------------ #
# Replace binary columns with more interpretable values
binary_cols = ['no_fee', 'has_roofdeck', 'has_gym', 'has_pool']
for col in binary_cols:
    rentals[col] = rentals[col].replace({1: 'yes', 0: 'no'})

# ------------------ Univariate Analysis ------------------ #
# Rent column statistics
rent_stats = {
    "mean": rentals.rent.mean(),
    "median": rentals.rent.median(),
    "mode": rentals.rent.mode()[0],
    "trimmed_mean": trim_mean(rentals.rent, 0.1),
}

print("Rent Statistics:", rent_stats)

# Visualizing rent distribution
plt.figure(figsize=(8, 6))
plt.hist(rentals.rent, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.title("Distribution of Rent")
plt.xlabel("Rent Price")
plt.ylabel("Frequency")
plt.axvline(rent_stats['mean'], color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(rent_stats['median'], color='green', linestyle='dashed', linewidth=1, label='Median')
plt.legend()
plt.show()

# ------------------ Outlier Detection ------------------ #
# Z-score method
z_scores = zscore(rentals[['rent', 'min_to_subway']].select_dtypes(include=np.number))
outliers_zscore = rentals[(np.abs(z_scores) > 3).any(axis=1)]
non_outliers = rentals[~(np.abs(z_scores) > 3).any(axis=1)]

# IQR method for 'min_to_subway'
Q1 = rentals['min_to_subway'].quantile(0.25)
Q3 = rentals['min_to_subway'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = rentals[(rentals['min_to_subway'] < (Q1 - 1.5 * IQR)) | (rentals['min_to_subway'] > (Q3 + 1.5 * IQR))]

# Adding an outlier flag column
rentals['is_outlier'] = (np.abs(z_scores) > 3).any(axis=1)

# ------------------ Outlier Treatment ------------------ #
# Winsorization: Cap outliers to 1.5*IQR bounds
winsorized_subway = rentals['min_to_subway'].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
rentals['winsorized_min_to_subway'] = winsorized_subway

# Log transformation (with offset for non-negativity)
rentals['log_min_to_subway'] = np.log(rentals['min_to_subway'] + 1)

# Visualization after log transformation
plt.figure(figsize=(8, 6))
plt.hist(rentals['log_min_to_subway'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title("Distribution of Log-Transformed Min to Subway")
plt.xlabel("Log(Min to Subway + 1)")
plt.ylabel("Frequency")
plt.show()

# ------------------ Categorical Insights ------------------ #
# Analyzing borough distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='borough', data=rentals, palette='Set2')
plt.title("Number of Listings by Borough")
plt.xlabel("Borough")
plt.ylabel("Count")
plt.show()

# Borough analysis with 'no_fee'
plt.figure(figsize=(8, 6))
sns.countplot(x='borough', hue='no_fee', data=rentals, palette='pastel')
plt.title("Fee-Free Listings by Borough")
plt.xlabel("Borough")
plt.ylabel("Count")
plt.legend(title="No Fee")
plt.show()

# ------------------ Feature Engineering ------------------ #
# Standardizing 'min_to_subway'
scaler = StandardScaler()
rentals['scaled_min_to_subway'] = scaler.fit_transform(rentals[['min_to_subway']])

# ------------------ Statistical Testing ------------------ #
from scipy.stats import ttest_ind

# T-test to compare outliers vs non-outliers in 'min_to_subway'
outliers_vs_non = ttest_ind(outliers_zscore['min_to_subway'], non_outliers['min_to_subway'], nan_policy='omit')
print("T-test results (Outliers vs Non-Outliers - Min to Subway):", outliers_vs_non)

# ------------------ Summary Visualization ------------------ #
# Boxplot of rent (including outliers)
plt.figure(figsize=(8, 6))
plt.boxplot(rentals.rent, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Rent Prices")
plt.ylabel("Rent Price")
plt.show()
