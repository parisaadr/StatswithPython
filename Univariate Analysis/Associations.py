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

# ------------------ UNIVARIATE (quantitative) ------------------ #

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

# ------------------ CATEGORICAL ------------------ #

# print(rentals.borough.unique())
# print(rentals.borough.value_counts())
# print(rentals.borough.value_counts(normalize=True))
# print(rentals.borough.value_counts(dropna = False))

sns.countplot(x=rentals.borough)
sns.countplot(x=rentals.bedrooms)
sns.countplot(x=rentals.has_washer_dryer)

print(rentals['rent'].dtype)
print(rentals['rent'].unique())
rentals['rent'] = pd.to_numeric(rentals['rent'], errors='coerce')

# Check if 'mean()' works on the entire DataFrame
# print(rentals.rent.mean())

# # Check if 'mean()' works after grouping
# print(rentals.select_dtypes(include='number').mean())
# print(rentals.dtypes)
# print(rentals.drop(columns=['borough']).mean())

mean_grouped = rentals.groupby("borough")["rent"].mean().reset_index()
median_grouped = rentals.groupby("borough")["rent"].median().reset_index()

plt.hist(rentals.rent[rentals.borough=='Manhattan'], label='Manhattan', bins = 30, density=True, alpha=.5)
plt.hist(rentals.rent[rentals.borough=='Brooklyn'], label='Brooklyn', bins = 30, density=True, alpha=.5)
plt.hist(rentals.rent[rentals.borough=='Queens'], label='Queens', bins = 30, density=True, alpha=.5)
plt.legend()

sns.boxplot(x='borough', y='rent', data = rentals)

sns.boxplot(x='borough', y='rent', data = rentals[rentals.rent<10000])

sns.boxplot(x='no_fee', y='rent', data = rentals[rentals.rent<10000])

sns.boxplot(x='has_roofdeck', y='rent', data = rentals[rentals.rent<10000])

sns.boxplot(x='has_gym', y='rent', data = rentals[rentals.rent<10000])

sns.scatterplot(x='size_sqft', y='rent', data=rentals)
from scipy.stats import pearsonr
print(pearsonr(rentals.rent, rentals.size_sqft))

numeric_data = rentals.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

colors = sns.diverging_palette(150, 275, as_cmap=True)
sns.heatmap(numeric_data.corr(), center=0, cmap=colors)
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
# plt.show()

sns.scatterplot(x='building_age_yrs', y='floor', data=rentals)

sns.scatterplot(x='building_age_yrs', y='min_to_subway', data=rentals)

sns.scatterplot(x='size_sqft', y='rent', data=rentals, hue='borough',palette='bright')

sns.pairplot(rentals)
# plt.show()

elevatgymcross = pd.crosstab(rentals.has_elevator, rentals.has_gym)
print(elevatgymcross)
sns.countplot(x='has_elevator', hue='has_gym', data=rentals)

from scipy.stats import chi2_contingency
Xtab = pd.crosstab(rentals.has_elevator, rentals.has_gym)
chi2_xtab = chi2_contingency(Xtab)
print(chi2_xtab)





