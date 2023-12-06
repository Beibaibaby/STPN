import argparse
import math
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('e_to_e_means.pkl', 'rb') as f:
    e_to_e_means = pickle.load(f)

with open('e_to_i_means.pkl', 'rb') as f:
    e_to_i_means = pickle.load(f)
    
mean_e_to_e = np.mean(e_to_e_means)
std_e_to_e = np.std(e_to_e_means)

mean_e_to_i = np.mean(e_to_i_means)
std_e_to_i = np.std(e_to_i_means)

for e_to_e, e_to_i in zip(e_to_e_means, e_to_i_means):
    plt.scatter(['E-to-E'], [e_to_e], color='red', alpha=0.15)  # Individual E-to-E means
    plt.scatter(['E-to-I'], [e_to_i], color='blue', alpha=0.15)  # Individual E-to-I means

# Plot the overall mean and standard deviation as error bars
plt.errorbar(['E-to-E'], [mean_e_to_e], yerr=[std_e_to_e], fmt='o', label='E-to-E mean='+str(mean_e_to_e), color='red', capsize=5)
plt.errorbar(['E-to-I'], [mean_e_to_i], yerr=[std_e_to_i], fmt='o', label='E-to-I mean='+str(mean_e_to_i), color='blue', capsize=5)

plt.ylabel('Average Weight')
plt.title('Average Weights in E-I Network Across Runs')
plt.legend()
plt.savefig('average_weights_EI_across_runs_re.png', dpi=300)
plt.show()
    
plt.clf()
    
# Convert e_to_e_means and e_to_i_means to their absolute values
abs_e_to_e_means = [-x for x in e_to_e_means]
abs_e_to_i_means = [x for x in e_to_i_means]

# Compute mean and standard deviation for the absolute values
abs_mean_e_to_e = np.mean(abs_e_to_e_means)
abs_std_e_to_e = np.std(abs_e_to_e_means)

abs_mean_e_to_i = np.mean(abs_e_to_i_means)
abs_std_e_to_i = np.std(abs_e_to_i_means)

# Plotting absolute values
for abs_e_to_e, abs_e_to_i in zip(abs_e_to_e_means, abs_e_to_i_means):
    plt.scatter(['E-to-E'], [abs_e_to_e], color='red', alpha=0.15)  # Individual E-to-E means
    plt.scatter(['E-to-I'], [abs_e_to_i], color='blue', alpha=0.15)  # Individual E-to-I means

# Plot the overall mean and standard deviation of absolute values as error bars
plt.errorbar(['E-to-E'], [abs_mean_e_to_e], yerr=[abs_std_e_to_e], fmt='o', label='E-to-E (Abs) mean='+str(abs_mean_e_to_e), color='red', capsize=5)
plt.errorbar(['E-to-I'], [abs_mean_e_to_i], yerr=[abs_std_e_to_i], fmt='o', label='E-to-I (Abs) mean='+str(abs_mean_e_to_i), color='blue', capsize=5)

plt.ylabel('Depression Magnitude')
plt.title('Depression Magnitude in E-I Network Across Runs')
plt.legend()
plt.savefig('average_abs_weights_EI_across_runs.png', dpi=300)
plt.show()
