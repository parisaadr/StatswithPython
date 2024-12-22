import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

# simulate data:
# samp_size = 100
# A_rate = .5
# B_rate = (1 + lift) * A_rate
#
# clicks_A = np.random.choice(['yes', 'no'],
#                             size=int(samp_size/2),
#                             p=[A_rate, 1-A_rate])
# clicks_B = np.random.choice(['yes', 'no'],
#                             size=int(samp_size/2),
#                             p=[B_rate, 1-B_rate])
# outcome = list(clicks_A) + list(clicks_B)
# group = ['Button A']*int(samp_size/2) + ['Button B']*int(samp_size/2)
# sim_data = {"Group": group, "Clicked": outcome}
# sim_data = pd.DataFrame(sim_data)
# print(sim_data)

# OR

A_rate = .5
B_rate = .6

clicks_A = np.random.choice(['yes', 'no'], size=100, p=[.5, .5])
clicks_B = np.random.choice(['yes', 'no'], size=100, p=[.6, .4])

outcome = list(clicks_A) + list(clicks_B)
group = ['A']*100 + ['B']*100

data = pd.DataFrame({'Group': group, 'Clicked': outcome})

# ------------------ run a hypothesis test ------------------ #
ab_contingency = pd.crosstab(data.Group, data.Clicked)
sig_threshold = 0.05
chi2, pval, dof, expected = st.chi2_contingency(ab_contingency)
result = ('significant' if pval < sig_threshold else 'not significant')
print(result)

sample_size = 200
A_rate = .5
B_rate = .7
sig_threshold = 0.05
results = []

for i in range(100):
    clicks_A = np.random.choice(['yes', 'no'],
                            size=int(sample_size/2),
                            p=[A_rate, 1-A_rate])
    clicks_B = np.random.choice(['yes', 'no'],
                            size=int(sample_size/2),
                            p=[B_rate, 1-B_rate])
    outcome = list(clicks_A) + list(clicks_B)
    group = ['Button A']*int(sample_size/2) + ['Button B']*int(sample_size/2)
    sim_data = {"Group": group, "Clicked": outcome}
    sim_data = pd.DataFrame(sim_data)
    ab_contingency = pd.crosstab(sim_data.Group, sim_data.Clicked)
    chi2, pval, dof, expected = st.chi2_contingency(ab_contingency)
    result = ('significant' if pval < sig_threshold else 'not significant')
    results.append(result)

print("Proportion of results that are significant:")
results = np.array(results)
print(np.sum(results == 'significant')/100)

print("Proportion of results that are NOT significant:")
results = np.array(results)
print(np.sum(results == 'not significant')/100)















