import numpy as np
import pandas as pd
from algorithm import run_algorithm

'''
Experiment 1: How does the algorithm behave at different sample sizes? 
We choose a fixed noise level, 0.25. This is a realistic noise level.
'''

# Load the data
n = 6
F = pd.read_csv('Data_generated/DUM_clean_n_{}.csv'.format(n), sep=",", header=None)
F = F.map(lambda s: complex(s.replace('i', 'j'))).values
N = F.shape[0]
d = F.shape[1]

if n % 2 == 1:  # if odd
    size_aug = n
else:
    size_aug = n - 1 # make the window odd

num_components = 3  #for PCA
sigma = 0.1  # fixed noise level
# at which sample sizes:
powers = np.arange(2, n+2)
sample_sizes = 2**powers

number_of_runs = 10
number_of_batches = 10 # how many parts of the data we look at

base_seed = 26

distance_noisy_all = np.zeros((number_of_runs*number_of_batches, len(sample_sizes)))
distance_denoised_all = np.zeros((number_of_runs*number_of_batches, len(sample_sizes)))
distance_aug_denoised_all = np.zeros((number_of_runs*number_of_batches, len(sample_sizes)))

for run in range(number_of_runs):
    print(run)
    # add different noise to the whole data
    rng = np.random.default_rng(seed=base_seed+run)
    noise_real = np.random.normal(loc=0.0, scale=sigma, size=F.shape)
    noise_complex = np.random.normal(loc=0.0, scale=sigma, size=F.shape)
    F_n = F + noise_real + 1j*noise_complex
    for i, N in enumerate(sample_sizes):
        #print("sample size ", N)
        for j in range(number_of_batches):
            #print("batch ", j)
            idx = rng.choice(F.shape[0], size=N, replace=False)
            # cut the noisy data
            F_reduced = F[idx][:]
            F_n_reduced = F_n[idx][:]
            distance_noisy, distance_denoised, distance_aug_denoised = run_algorithm(F_reduced, F_n_reduced, size_aug, int(np.log2(d)), N, num_components)
            distance_noisy_all[number_of_batches*run+j][i] = distance_noisy
            distance_denoised_all[number_of_batches*run+j][i] = distance_denoised
            distance_aug_denoised_all[number_of_batches*run+j][i] = distance_aug_denoised

# Calculate Mean and STD
mean_distance_noisy_all = np.mean(distance_noisy_all, axis=0) # mean over the columns
mean_distance_denoised_all= np.mean(distance_denoised_all, axis=0) # mean over the columns
mean_distance_aug_denoised_all = np.mean(distance_aug_denoised_all, axis=0) # mean over the columns

std_distance_noisy_all = np.std(distance_noisy_all, axis=0)
std_distance_denoised_all = np.std(distance_denoised_all, axis=0)
std_distance_aug_denoised_all = np.std(distance_aug_denoised_all, axis=0)

# Save the data
df = pd.DataFrame({"mean_distance_noisy_all": mean_distance_noisy_all})
df["mean_distance_denoised_all"] = mean_distance_denoised_all
df["mean_distance_aug_denoised_all"] = mean_distance_aug_denoised_all
df["std_distance_noisy_all"] = std_distance_noisy_all
df["std_distance_denoised_all"] = std_distance_denoised_all
df["std_distance_aug_denoised_all"] = std_distance_aug_denoised_all


df.to_csv(
    "Results-experiment-1/Distances_n_{}_size_aug_{}_runs_{}_batches_{}_sigma_{}.csv".format(n, size_aug, number_of_runs, number_of_batches, sigma),
    sep=",",
    header=True,
    index=True
)
