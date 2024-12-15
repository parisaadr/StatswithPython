import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

hourly_pay = np.genfromtxt('hourly_pay')
print(hourly_pay)

# plt.hist(hourly_pay, bins=50)
# plt.show()

mean_salary = np.mean(hourly_pay)
print(mean_salary)

population = list(hourly_pay)
sample = random.sample(population, 150)
np.mean(sample)

# plt.hist(hourly_pay, bins=35)
# plt.vlines(np.mean(hourly_pay), 0, 500000, color = 'k', lw=3, linestyles='dashed')
# plt.show()

sampl_means = []

for i in range(10000):
    sampl = random.sample(population, 150)
    sampl_means.append(np.mean(sampl))

# print(sampl_means)

# plt.hist(sampl_means, bins=50)
# plt.show()
print(np.mean(sampl_means))

population_mean = hourly_pay.mean()

samp_means = []

for i in range(10000):
    samp = random.sample(population, 50)
    samp_means.append(np.mean(samp))

# plot the sample mean
# sns.distplot(samp_means)
# plt.xlim(0, 40)

# plot the normal distribution
muu = population_mean
sigmaa = np.std(population)/(50**.5)
xx = np.linspace(muu - 3*sigmaa, muu + 3*sigmaa, 100)
# plt.plot(xx, stats.norm.pdf(xx, muu, sigmaa), color='k')

# plt.show()

# print(np.linspace(0, 1, 10))

sample_mean = np.mean(sample)
sample_sd = np.std(sample)
print(sample_mean)
print(sample_sd)

print(np.mean(population))
print(np.std(population))

mu = sample_mean
sample_size = len(sample)
sigma = sample_sd/(sample_size**.5)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma), color='k')
# plt.show()

samp1 = random.sample(list(hourly_pay), 150)
print(np.mean(samp1))

standard_error = np.std(samp1)/(150**.5)
print(standard_error)

true_se = np.std(population)/(150**.5)
print(true_se)

print(stats.norm.ppf(0.975, np.mean(population), standard_error))
print(stats.norm.ppf(0.025, np.mean(population), standard_error))

lower_ci = stats.norm.ppf(0.975, np.mean(sample), standard_error)
upper_ci = stats.norm.ppf(0.025, np.mean(sample), standard_error)
print(lower_ci)
print(upper_ci)

import scipy.stats as stats
import math
from scipy.stats import t
mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='k', label="Normal")
rv = t(df=5, loc=0, scale=1)
t_val = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = rv.pdf(x)

plt.xlim(-3,3)
plt.plot(t_val,y, label = "Student's t (df=5)")
plt.xlabel('value')
plt.ylabel('density')
plt.legend()
plt.show()




