import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import binomtest

monthly_report = pd.read_csv('monthly_report.csv')
print(monthly_report.head())

sample_size = len(monthly_report)
print('sample size:' + ' ' + str(sample_size))

purchased_num = len(monthly_report[monthly_report.purchase == "y"])
print("number of people who made a purchase" + ": " + str(purchased_num) + " out of " + str(sample_size))

proportion = monthly_report["purchase"].value_counts(normalize=True)
print(proportion)

simulated_monthly_visitors = np.random.choice(['y', 'n'], size=500, p=[0.1, 0.9])
purchases = np.sum(simulated_monthly_visitors == 'y')
print(purchases)
p_value = binomtest(purchases, 500, .1, alternative='less')
print(p_value)
# null hypothesis: purchase rate is 10%
# alternative hypothesis: purchase rate is <10%
# if number of purchases or k is 39 the p_value is 0.055 that means:
# there's a 5.5% chance of seeing 39 or fewer(given the alternative) purchases given that the true purchase rate is 10%

p_vals = []

for i in range(1000):
    simulated_monthly_visitors = np.random.choice(['y', 'n'], size=500, p=[0.1, 0.9])
    purchases = np.sum(simulated_monthly_visitors == 'y')
    p_value = binomtest(purchases, 500, .1).pvalue
    p_vals.append(p_value)

# plt.hist(p_vals)
# plt.xlabel("P-value")
# plt.ylabel("Frequency")
# plt.title("Histogram of P-values")
# plt.show()

type_1_error = 0

for i in range(10000):
    simulated_visitors = np.random.choice(['y', 'n'], size=500, p=[.1, .9])
    purchases = np.sum(simulated_visitors == 'y')
    p_value = binomtest(purchases, 500, .1).pvalue
    if p_value < 0.05:
        type_1_error += 1

print(type_1_error/10000)

false_positives = 0
sig_threshold = 0.05

ss = 500
pp = 0.1

pvals = []

for i in range(1000):
    sim_visitors = np.random.choice(['y', 'n'], size=ss, p=[pp, 1-pp])
    num_purchases = np.sum(sim_visitors == 'y')
    p_val = binomtest(num_purchases, ss, pp).pvalue
    pvals.append(p_val)
    if p_val < sig_threshold:
        false_positives += 1

print(false_positives/1000)
# plt.hist(pvals)
# plt.show()


sig_threshold = 0.05
num_tests = np.array(range(50))
probabilities = 1-((1-sig_threshold)**num_tests)
plt.plot(num_tests, probabilities)

plt.xlabel('Number of Tests')
plt.ylabel('Probability of at Least One Type I Error')
plt.title('Type I Error Rate fpr Multiple Tests')
plt.show()









