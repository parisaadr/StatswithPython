import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statsmodels

heart = pd.read_csv('heart_disease.csv')
heart.head()
heart.info()

grouped_mean = heart.groupby('heart_disease').mean(numeric_only=True)
print(grouped_mean)

grouped_median = heart.groupby('heart_disease').median(numeric_only=True)
print(grouped_median)

# sns.boxplot(x=heart.heart_disease, y=heart.thalach)
# plt.show()

# sns.boxplot(x=heart.heart_disease, y=heart.trestbps)
# plt.show()

chol_hd = heart.chol[heart.heart_disease == 'presence']
chol_no_hd = heart.chol[heart.heart_disease == 'absence']
tstat, pval = ttest_ind(chol_hd, chol_no_hd)
print('p-value for `chol` two-sample t-test: ', pval)
# sns.boxplot(x=heart.cp, y=heart.thalach)
# plt.show()


thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_nonangin = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']
# ------------------ Two Sample T-Test ------------------ #
# Null hypothesis: there is no difference in mean resting blood pressure among patients
# Alternative: there is a difference in mean resting blood pressure among patients

scramble = np.random.choice(heart.heart_disease, size= len(heart), replace=False)
print(scramble)

sim_trestbps_hd = heart.trestbps[scramble == 'presence']
sim_trestbps_no_hd = heart.trestbps[scramble == 'absence']
sim_mean_diff = np.mean(sim_trestbps_hd) - np.mean(sim_trestbps_no_hd)
print(sim_mean_diff)

null_mean_diff = []
for i in range(1000):
    scramble = np.random.choice(heart.heart_disease, size=len(heart), replace=False)
    sim_trestbps_hd = heart.trestbps[scramble == 'presence']
    sim_trestbps_no_hd = heart.trestbps[scramble == 'absence']
    sim_mean_diff = np.mean(sim_trestbps_hd) - np.mean(sim_trestbps_no_hd)
    null_mean_diff.append(sim_mean_diff)
# plt.hist(null_mean_diff)
# plt.show()

trestbps_hd = heart.trestbps[heart.heart_disease == 'presence']
trestbps_no_hd = heart.trestbps[heart.heart_disease == 'absence']
observed_mean_diff = np.mean(trestbps_hd) - np.mean(trestbps_no_hd)
print(observed_mean_diff)
# plt.hist(null_mean_diff)
# plt.axvline(x=observed_mean_diff, color='red')
# plt.show()

one_sided = np.sum(np.array(null_mean_diff)>observed_mean_diff)/len(null_mean_diff)
print(one_sided)

from scipy.stats import ttest_ind
tstat, pval = ttest_ind(trestbps_hd, trestbps_no_hd)
print(pval)

null_median_diff = []
for i in range(1000):
    scramble = np.random.choice(heart.heart_disease, size=len(heart), replace=False)
    sim_trestbps_hd = heart.trestbps[scramble == 'presence']
    sim_trestbps_no_hd = heart.trestbps[scramble == 'absence']
    sim_median_diff = np.mean(sim_trestbps_hd) - np.median(sim_trestbps_no_hd)
    null_median_diff.append(sim_median_diff)
# plt.hist(null_median_diff)
# plt.show()

# ------------------ ANOVA ------------------ #
from scipy.stats import f_oneway
Fstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print('p-value for ANOVA: ', pval)

# ------------------
# Tukey's range test ------------------ #
from statsmodels.stats.multicomp import pairwise_tukeyhsd
output = pairwise_tukeyhsd(heart.thalach, heart.cp)
print(output)

Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)

# ------------------ Chi-sq Test ------------------ #
from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print('p-value for chi-square test: ', pval)

Xtab = pd.crosstab(heart.cp, heart.heart_disease)
from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print('scipy p-value for chi-square test: ', pval)

probs_cp = heart.cp.value_counts(normalize=True).sort_index()
probs_cp

probs_hd = heart.heart_disease.value_counts(normalize=True)
probs_hd

Xtab = pd.crosstab(heart.heart_disease,heart.cp)
from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print(exp)
print(chi2)

observed = pd.crosstab(heart.heart_disease,heart.cp)
expected = np.array([probs_hd[0] * probs_cp, probs_hd[1] * probs_cp])*len(heart)
print(expected)
obs_chi2 = ((observed-expected)**2/expected).sum().sum()
print(obs_chi2)

null_chisqs = []
expected = np.array([probs_hd[0] * probs_cp, probs_hd[1] * probs_cp])*len(heart)

for i in range(1000):
    scrambled_hd = np.random.choice(heart.heart_disease, size = len(heart), replace = False)
    scrambled_cp = np.random.choice(heart.cp, size = len(heart), replace = False)
    sim_observed = pd.crosstab(scrambled_hd,scrambled_cp)
    sim_chisq = ((sim_observed-expected)**2/expected).sum().sum()
    null_chisqs.append(sim_chisq)

# plt.hist(null_chisqs)
# plt.axvline(x = obs_chi2, color = 'red')
# plt.show()