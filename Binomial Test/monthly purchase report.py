import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

monthly_report = pd.read_csv('monthly_report.csv')

sample_size = len(monthly_report)
print('sample size:' + ' ' + str(sample_size))

purchased_num = len(monthly_report[monthly_report.purchase == "y"])
proportion = monthly_report["purchase"].value_counts(normalize=True)
print("number of people who made a purchase" + ": " + str(purchased_num) + " out of " + str(sample_size))
print(proportion)

simulated_monthly_visitors = np.random.choice(['y', 'n'], size=10, p=[0.1, 0.9])
ran_num_purchased = np.sum(simulated_monthly_visitors == "y")
print(ran_num_purchased)

simulated_monthly_visitors = np.random.choice(['y', 'n'], size=500, p=[0.1, 0.9])


null_outcomes = []

for i in range(10000):
    simulated_monthly_visitors = np.random.choice(['y', 'n'], size=500, p=[0.1, 0.9])
    num_purchased = np.sum(simulated_monthly_visitors == 'y')
    null_outcomes.append(num_purchased)

null_min = np.min(null_outcomes)
print("the minimum value: " + str(null_min))

null_max = np.max(null_outcomes)
print("the maximum value: " + str(null_max))

# plt.hist(null_outcomes)
# plt.axvline(purchased_num, color = 'r', density=True)
# plt.xlabel('number of purchases')
# plt.ylabel('probability')
# plt.show()

# --------------- 90% confidence interval (CI) --------------- #
null_90CI = np.percentile(null_outcomes, [5, 95])
print(null_90CI)

# --------------- 1-sided p-value --------------- #
null_outcomes = np.array(null_outcomes)
one_p_value = np.sum(null_outcomes <= purchased_num)/len(null_outcomes)
print("our one-sided p_value: " + str(one_p_value))

# --------------- 2-sided p-value --------------- #
two_p_value = np.sum((null_outcomes <= purchased_num) | (null_outcomes >= 59))/len(null_outcomes)
print("our two-sided p_value: " + str(two_p_value))

# --------------- binomial test --------------- #
def simulation_binomial_test(observed_successes, n, p):
    nul_outcomes = []
    for j in range(10000):
        the_simulated_monthly_visitors = np.random.choice(['y', 'n'], size=n, p=[p, 1 - p])
        the_purchased_num = np.sum(simulated_monthly_visitors == 'y')
        nul_outcomes.append(num_purchased)
    nul_outcomes = np.array(nul_outcomes)

    p_value = np.sum(nul_outcomes <= observed_successes) / len(nul_outcomes)

    return p_value

p_value1 = simulation_binomial_test(41, 500, .1)
print("simulation p-value: ", p_value1)


from scipy.stats import binomtest
p_value2 = binomtest(41, 500, .1, alternative='less')
print("binomial test p-value: ", p_value2)

# --------------- alternative hypotheses --------------- #

def simulation_binomial_test(observed_successes, n, p, alternative_hypothesis):
    # initialize null_outcomes
    null_outcomes = []

    # generate the simulated null distribution
    for i in range(10000):
        simulated_monthly_visitors = np.random.choice(['y', 'n'], size=n, p=[p, 1 - p])
        num_purchased = np.sum(simulated_monthly_visitors == 'y')
        null_outcomes.append(num_purchased)

    null_outcomes = np.array(null_outcomes)

    if alternative_hypothesis == 'less':
        p_value = np.sum(null_outcomes <= observed_successes) / len(null_outcomes)
    elif alternative_hypothesis == 'greater':
        p_value = np.sum(null_outcomes >= observed_successes) / len(null_outcomes)
    else:
        difference = np.abs(p * n - observed_successes)
        upper = p * n + difference
        lower = p * n - difference
        p_value = np.sum((null_outcomes >= upper) | (null_outcomes <= lower)) / len(null_outcomes)

    # return the p-value
    return p_value

# Test the function:
print('lower tail one-sided test:')
p_value1 = simulation_binomial_test(45, 500, .1, alternative_hypothesis='less')
print("simulation p-value: ", p_value1)

p_value2 = binomtest(45, 500, .1, alternative='less')
print("binom_test p-value: ", p_value2)

print('upper tail one-sided test:')
p_value1 = simulation_binomial_test(53, 500, .1, alternative_hypothesis='greater')
print("simulation p-value: ", p_value1)

p_value2 = binomtest(53, 500, .1, alternative='greater')
print("binom_test p-value: ", p_value2)

print('two-sided test:')
p_value1 = simulation_binomial_test(42, 500, .1, alternative_hypothesis='not_equal')
print("simulation p-value: ", p_value1)

p_value2 = binomtest(42, 500, .1)
print("binom_test p-value: ", p_value2)



