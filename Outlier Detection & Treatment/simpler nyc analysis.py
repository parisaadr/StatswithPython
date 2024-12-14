import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

rentals = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/streeteasy.csv")
# print(rentals.head())

# DATA CLEANING

# print(rentals.info())
rentals.no_fee = rentals.no_fee.replace({1:'yes', 0:'no'})
rentals.has_roofdeck = rentals.has_roofdeck.replace({1:'yes', 0:'no'})
rentals.has_washer_dryer = rentals.has_washer_dryer.replace({1:'yes', 0:'no'})
rentals.has_doorman = rentals.has_doorman.replace({1:'yes', 0:'no'})
rentals.has_elevator = rentals.has_elevator.replace({1:'yes', 0:'no'})
rentals.has_dishwasher = rentals.has_dishwasher.replace({1:'yes', 0:'no'})
rentals.has_patio = rentals.has_patio.replace({1:'yes', 0:'no'})
rentals.has_gym = rentals.has_gym.replace({1:'yes', 0:'no'})
# print(rentals.describe(include = 'all'))

# UNIVARIATE (quantitative)

rent_mean = rentals.rent.mean()
# print(rent_mean)
rent_median = rentals.rent.median()
# print(rent_median)
rent_mode = rentals.rent.mode()
# print(rent_mode)

# OR

mean_rent = np.mean(rentals.rent)
# print(mean_rent)
median_rent = np.median(rentals.rent)
# print(median_rent)
mode_rent = stats.mode(rentals.rent)
# print(mode_rent[0])
# data excluding the lowest 10% and the highest 10%
trimmed_mean = stats.trim_mean(rentals.rent, proportiontocut=0.1)
# print(trimmed_mean)
standev_rent = np.std(rentals.rent)
small_std = np.random.normal(10, 1, 1000)
large_std = np.random.normal(10, 10, 1000)
plt.hist(large_std, alpha = 0.5)
plt.hist(small_std, alpha = 0.5)
# plt.legend()
# plt.show()

# VISUALIZE
plt.hist(rentals.rent, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(mean_rent, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_rent:.2f}')
plt.axvline(median_rent, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_rent:.2f}')
plt.axvline(trimmed_mean, color='b', linestyle='dashed', linewidth=2, label=f'Trimmed mean: {trimmed_mean:.2f}')
# plt.legend()
# plt.show()

sns.histplot(rentals.rent, kde=False)
# plt.show()

plt.boxplot(rentals.rent)
# plt.show()

sns.displot(rentals.min_to_subway, kde=False)
# plt.show()


# FINDING THE OUTLIERS
subway_outliers = rentals[rentals.min_to_subway > 40]
# print(subway_outliers)
neighborhood_outliers = rentals[rentals.min_to_subway > 40].groupby('neighborhood').size()
# print(neighborhood_outliers)

# dealing with outliers
# Z-Scores
from scipy.stats import zscore
z_scores = zscore(rentals['min_to_subway'])
outliers_zscore = rentals[np.abs(z_scores) > 3]
# print(outliers_zscore)

# Interquartile Range (IQR)
Q1 = rentals['min_to_subway'].quantile(0.25)
Q3 = rentals['min_to_subway'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = rentals[(rentals['min_to_subway'] < (Q1 - 1.5 * IQR)) |
                       (rentals['min_to_subway'] > (Q3 + 1.5 * IQR))]
# print(outliers_iqr)

# Checking Neighborhoods and Their Locations
neighborhood_stats = rentals.groupby('neighborhood')['min_to_subway'].describe()
# print(neighborhood_stats)

# Checking for Data Entry Errors
duplicates = rentals[rentals.duplicated(subset=['neighborhood', 'min_to_subway'], keep=False)]
# print(duplicates)

# Examining Extreme Outliers
# print("Extreme Outliers by Z-scores:")
# print(outliers_zscore[['neighborhood', 'min_to_subway']])
zscore_outlier_count = outliers_zscore.groupby(['neighborhood', 'min_to_subway']).size().reset_index(name='count')
zscore_outlier_count_sorted = zscore_outlier_count.sort_values(by='count', ascending=False)
# print("Z-score Outliers Count:")
# print(zscore_outlier_count_sorted)

# print("Extreme Outliers by IQR:")
# print(outliers_iqr[['neighborhood', 'min_to_subway']])
iqr_outlier_count = outliers_iqr.groupby(['neighborhood', 'min_to_subway']).size().reset_index(name='count')
iqr_outlier_count_sorted = iqr_outlier_count.sort_values(by='count', ascending=False)
# print("IQR Outliers Count:")
# print(iqr_outlier_count_sorted)

# plotting
zscore_outlier_count_sorted.plot(kind='bar', x='neighborhood', y='count', legend=False, figsize=(10, 6))
plt.title('Count of Z-score Outliers by Neighborhood and min_to_subway (Sorted)')
plt.xlabel('Neighborhood and min_to_subway')
plt.ylabel('Count')
plt.xticks(rotation=90)
# plt.show()

iqr_outlier_count_sorted.plot(kind='bar', x='neighborhood', y='count', legend=False, figsize=(10, 6))
plt.title('Count of IQR Outliers by Neighborhood and min_to_subway (Sorted)')
plt.xlabel('Neighborhood and min_to_subway')
plt.ylabel('Count')
plt.xticks(rotation=90)
# plt.show()

# Various Approaches for handling the Outliers

# Removing Common Outliers in both Z-score and IQR
common_outliers = pd.merge(outliers_zscore, outliers_iqr, how='inner', on=['rental_id', 'min_to_subway'])
cleaned_rentals = rentals.drop(common_outliers.index)
# print("Cleaned Dataset (Common Outliers Removed):")
# print(cleaned_rentals)

# Winsorizing (Capping) the Outliers
min_to_subway_lower = rentals['min_to_subway'].quantile(0.01)
min_to_subway_upper = rentals['min_to_subway'].quantile(0.99)
rentals['min_to_subway'] = rentals['min_to_subway'].clip(lower=min_to_subway_lower, upper=min_to_subway_upper)
# print("Dataset after Winsorizing the 'min_to_subway' values:")
# print(rentals.head())

# Imputing outliers by replacing with median
median_value = rentals['min_to_subway'].median()
outliers_zscore_mask = (rentals['min_to_subway'] > 3) | (rentals['min_to_subway'] < -3)
rentals.loc[outliers_zscore_mask, 'min_to_subway'] = median_value
# print("Dataset after imputing outliers with the median:")
# print(rentals.head())

# Segmentation
# Separating the outliers (both Z-score and IQR based)
outliers_zscore_segment = rentals[rentals['min_to_subway'] > 3]
outliers_iqr_segment = rentals[rentals['min_to_subway'] > rentals['min_to_subway'].quantile(0.75) + 1.5 * (rentals['min_to_subway'].quantile(0.75) - rentals['min_to_subway'].quantile(0.25))]
non_outliers = rentals[~rentals.index.isin(outliers_zscore_segment.index)]

# Analyze outliers separately
# print("Outliers based on Z-score:")
# print(outliers_zscore_segment)
# print("\nOutliers based on IQR:")
# print(outliers_iqr_segment)

# Log Transformation (For Skewed Data)
rentals['log_min_to_subway'] = np.log(rentals['min_to_subway'] + 1)

# print("Dataset after Log Transformation:")
# print(rentals.head())


# CATEGORICAL
print(rentals.borough.unique())
print(rentals.borough.value_counts())
print(rentals.borough.value_counts(normalize=True))
print(rentals.borough.value_counts(dropna = False))

sns.countplot(x=rentals.borough)
sns.countplot(x=rentals.bedrooms)
sns.countplot(x=rentals.has_washer_dryer)