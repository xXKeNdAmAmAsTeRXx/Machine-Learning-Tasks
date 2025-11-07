#%%
import matplotlib.pyplot as plt
import numpy as np
#%% md
# # Monte Carlo Test
#%% md
# ## Being Given Measurments
# 
#%%
arr = np.array([6.20, 4.34, 8.14, 6.24, 3.72, 3.54, 4.35, 2.67, 7.16, 6.00])

plt.hist(arr)
plt.title("Histogram of arr values")
plt.show()
#%% md
# ## Mean and Standard deviation of measurments
#%%
mean = np.mean(arr)
std = np.std(arr)

print(f"{mean=} +/- {std}")
#%% md
# ## Sampling from Normal Distribution with given parameters and Calculating mean
#%%
N = int(1e4)
np.random.seed(42)

sampled_means = []


for i in range(N):
    sampled_means.append(np.mean(np.random.normal(loc=mean, scale=std, size=len(arr))))

plt.hist(sampled_means)
plt.title("Histogram of means got from sampling from normal distribution")
plt.show()
#%% md
# ## Confidence intervals and observations greater than mean of arr
#%%
alpha = 0.005

sampled_means = sorted(sampled_means)[int(N * alpha):-int(N*alpha)]
plt.hist(sampled_means)
plt.title("With confidence intervals")
plt.show()

N_above_mean = np.sum(sampled_means >= mean)

print(f"With applied confidence intervals sampled means ranges in [{sampled_means[0]}, {sampled_means[-1]}]")
print(f"There is {N_above_mean} samples greater than {mean=} being {np.ceil(N_above_mean/len(sampled_means)*100)}%")
#%% md
# # Bootstrap Test
#%% md
# ## Being given array
#%%
arr = np.array([6.20, 4.34, 8.14, 6.24, 3.72, 3.54, 4.35, 2.67, 7.16, 6.00])

plt.hist(arr)
plt.title("Histogram of arr values")
plt.show()
#%% md
# ## Testing a hypothesis
# 
# $H_0$: Expected value of population is equal to 0
# $$
# H_0: \space \mu = 0
# $$
#%%
N = int(1e4)
np.random.seed(42)

mean = np.mean(arr)
std = np.std(arr)
print(f"{mean=} +/- {std}")

sampled_means = []
for i in range(N):
    sampled_means.append(np.mean(np.random.choice(arr, size=len(arr), replace=True)))




## Applaing confidence intervals
alpha = 0.005

sampled_means_ci = sorted(sampled_means)[int(N * alpha):-int(N*alpha)]


plt.hist(sampled_means, label="without confidence intervals")
plt.hist(sampled_means_ci, label="with confidence intervals")
plt.title("Histogram of means got from sampling from normal distribution")
plt.legend()
plt.show()

print(f"values with confidence intervals ranges in [{sampled_means[0]}, {sampled_means[-1]}]")
print(f"There is {np.sum([1 if m>mean else 0 for m in sampled_means_ci])} values above {mean=}")
#%% md
# ### Hypothesis $H_0$ is not true
#%% md
# # Permutation Test
#%% md
# ## Data
# We are given measurments of Cortisol level in blood for high or low dose of blude light
#%%
low_dose = np.array([378, 346,245,285,365,245,208,360,296,224,292])
high_dose = np.array([218,264,211,180,256,240, 261, 205, 145, 195, 187,210,378, 204, 232, 237,310])

low_mean = np.mean(low_dose)
low_std = np.std(low_dose)

high_mean = np.mean(high_dose)
high_std = np.std(high_dose)

plt.hist(high_dose, label="high dose")
plt.hist(low_dose, label="low dose", alpha=0.9, color="red")

print(f"{low_mean=} +/- {low_std}\n{high_mean=} +/- {high_std}")
plt.title("Histogram of Cortisol level in blood for blue light")
plt.legend()
plt.show()

#%% md
# ## Null Hypothesis
# $H_0$: blood cortisol level is not related with blue light dose
#%% md
# ## Permutation Test
#%%
N = int(1e4)
np.random.seed(42)

all_doses = np.concatenate((low_dose, high_dose))


sampled_means_low = []
sampled_means_high = []

idx = len(low_dose)
for i in range(N):
    np.random.shuffle(all_doses)
    sampled_means_low.append(np.mean(all_doses[idx:]))
    sampled_means_high.append(np.mean(all_doses[:idx]))


# Apply confidence intervals
alpha = 0.005
sampled_means_low = np.array(sorted(sampled_means_low)[int(N * alpha):-int(N*alpha)])
sampled_means_high = np.array(sorted(sampled_means_high)[int(N * alpha):-int(N*alpha)])

print(f"values with confidence intervals for low doses ranges in [{sampled_means_low[0]}, {sampled_means_low[-1]}]")
print(f"There is {np.sum([1 if m>low_mean else 0 for m in sampled_means_low])} values above {low_mean=}\n")

print(f"values with confidence intervals for high doses ranges in [{sampled_means_high[0]}, {sampled_means_high[-1]}]")
print(f"There is {np.sum([1 if m>high_mean else 0 for m in sampled_means_high])} values above {high_mean=}")

plt.title("Means of sampled doses with applied CI")
plt.hist(sampled_means_high, label="high dose")
plt.hist(sampled_means_low, label="low dose",color="red", alpha=0.9)
plt.legend()
plt.show()

#%%
mean_diff = high_mean - low_mean
sampled_means_diff = sampled_means_high - sampled_means_low

N_above_mean  = np.sum(sampled_means_diff > mean_diff)
print(f"There is {N_above_mean} samples greater than {mean_diff=}")

p_value = (np.sum(np.abs(sampled_means_diff) >= np.abs(mean_diff)) / N_above_mean)


print(f"{p_value=}")
#%%
import numpy as np

# Assuming you have the original data:
# low_dose = ...
# high_dose = ...

# --- Step 1: Calculate Observed Test Statistic ---
obs_diff = np.mean(high_dose) - np.mean(low_dose)
N_low = len(low_dose)
N_high = len(high_dose)
all_doses = np.concatenate((low_dose, high_dose))
N_perms = 10000  # Number of permutations (should be large)

# --- Step 2: Generate Null Distribution (Permutation Test) ---
null_distribution = []
for _ in range(N_perms):
    # Shuffle all data
    np.random.shuffle(all_doses)
    
    # Create permuted samples of original sizes
    perm_low = all_doses[:N_low]
    perm_high = all_doses[N_low:]
    
    # Calculate difference in means for this permutation
    perm_diff = np.mean(perm_high) - np.mean(perm_low)
    null_distribution.append(perm_diff)

# --- Step 3: Calculate P-value (Two-Sided) ---
# Count how many differences in the null distribution are as extreme as the observed difference
# np.abs(null_distribution) >= np.abs(obs_diff)
p_value_count = np.sum(np.abs(null_distribution) >= np.abs(obs_diff))

p_value = (p_value_count + 1) / (N_perms + 1) # Adding 1 to numerator and denominator is a common correction

N_above_mean  = np.sum(null_distribution > obs_diff)
print(f"There is {N_above_mean} samples greater than {obs_diff=}")

print(f"Observed Difference: {obs_diff:.4f}")
print(f"P-value: {p_value:.4f}")