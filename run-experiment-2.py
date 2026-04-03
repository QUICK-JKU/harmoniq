import numpy as np
import pandas as pd
from algorithm import run_algorithm

'''
Experiment 2: How does the algorithm behave at different noise level? 
We choose a fixed sample size N=100 for any n. This connects to the data scarcity regime form experiment 1.
'''
# Load the data
n = 6
F = pd.read_csv('Data_generated/DUM_clean_n_{}.csv'.format(n), sep=",", header=None)  #dataset
F = F.map(lambda s: complex(s.replace('i', 'j'))).values
N = F.shape[0]  # number of samples
d = F.shape[1]  # dimensions

if n % 2 == 1:  # if odd
    size_aug = n
else:
    size_aug = n - 1 # make the window odd

num_components = 3  # for PCA
N_reduced = 100  # how many sampels do we consider N_reduced << N
sigmas = np.linspace(0.1, 1.5, 10)  # noise values

number_of_runs = 10  # how many noise realizations
number_of_batches = 10 # how many parts of the data we look at

base_seed = 26

distance_noisy_all = np.zeros((number_of_runs*number_of_batches, len(sigmas)))
distance_denoised_all = np.zeros((number_of_runs*number_of_batches, len(sigmas)))
distance_aug_denoised_all = np.zeros((number_of_runs*number_of_batches, len(sigmas)))

for run in range(number_of_runs):
    print(run)
    for i, sigma in enumerate(sigmas):
        # add different noise to the whole data
        rng = np.random.default_rng(seed=base_seed + run)
        noise_real = np.random.normal(loc=0.0, scale=sigma, size=F.shape)
        noise_complex = np.random.normal(loc=0.0, scale=sigma, size=F.shape)
        F_n = F + noise_real + 1j * noise_complex
        for j in range(number_of_batches):
            #print("batch ", j)
            idx = rng.choice(F.shape[0], size=N_reduced, replace=False)
            # cut the noisy data
            F_reduced = F[idx][:]
            F_n_reduced = F_n[idx][:]
            distance_noisy, distance_denoised, distance_aug_denoised = run_algorithm(F_reduced, F_n_reduced, size_aug, int(np.log2(d)), N_reduced, num_components)
            #save the intermediate data:
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

# Save the final data
df = pd.DataFrame({"mean_distance_noisy_all": mean_distance_noisy_all})
df["mean_distance_denoised_all"] = mean_distance_denoised_all
df["mean_distance_aug_denoised_all"] = mean_distance_aug_denoised_all
df["std_distance_noisy_all"] = std_distance_noisy_all
df["std_distance_denoised_all"] = std_distance_denoised_all
df["std_distance_aug_denoised_all"] = std_distance_aug_denoised_all


df.to_csv(
    "Results-experiment-2/Distances_n_{}_size_aug_{}_runs_{}_batches_{}_N_{}.csv".format(n, size_aug, number_of_runs, number_of_batches, N_reduced),
    sep=",",
    header=True,
    index=True
)
