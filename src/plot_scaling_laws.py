import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle

from scipy.optimize import curve_fit

def load_experiment(filename):
    with open(filename, "r") as f:
        return json.load(f)


centralized_experiment_batch_sizes = [32, 64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(10, 6))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
save_dir = os.path.join(project_root, "plots", "centralized")
os.makedirs(save_dir, exist_ok=True)

x_vals = []
y_vals = []
centralized_experiment_results = [
    (batch_size, load_experiment(os.path.join(project_root, "results", f"centralized_experiment_results_{batch_size}.json")))
    for batch_size in centralized_experiment_batch_sizes
]

# Left subplot: Compute Budget vs. Cumulative Training Time for each batch size
fig, ax = plt.subplots(figsize=(10, 6))

accuracy_thresholds = [0.4, 0.5, 0.6]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

for idx, (accuracy_threshold, color) in enumerate(zip(accuracy_thresholds, colors)):
    x_vals = []
    y_vals = []
    for batch_size, results in centralized_experiment_results:
        # Find number of epochs to reach accuracy threshold
        num_epochs = len(results["accuracies"])
        for epoch, acc in enumerate(results["accuracies"]):
            if acc >= accuracy_threshold:
                num_epochs = epoch + 1
                break

        time_per_round = results["training_time"] / len(results["accuracies"])
        cumulative_time = time_per_round * num_epochs
        compute_budget = np.sum(results["compute_cost"][:num_epochs])
        
        x_vals.append(cumulative_time)
        y_vals.append(compute_budget)
        ax.plot(cumulative_time, compute_budget, marker='o', color=color,
                label=f"Batch size: {batch_size}, Acc threshold: {accuracy_threshold}")
        
        print(f"Batch size: {batch_size}, Acc threshold: {accuracy_threshold}")
        print(f"Total Training Time (s): {cumulative_time}")
        print(f"Compute Budget (samples): {compute_budget}")

    ax.plot(x_vals, y_vals, linestyle='--', color=color)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Total Training Time (s)")
ax.set_ylabel("Compute Budget (Total Samples Processed)")
ax.set_title("Compute Budget vs. Total Training Time")
ax.legend()

fig.savefig(os.path.join(save_dir, "compute_budget_centralized.png"))

x_vals = []
y_vals = []

fig, ax = plt.subplots(figsize=(10, 6))
# Right subplot: Noise Scale vs. Cumulative Training Time for each batch size
for batch_size, results in centralized_experiment_results:
    noise_scale = np.mean(results["noise_scales"]) # do we average ??

    x_axis = batch_size / noise_scale
    y_axis = 1 / (1 + (noise_scale / batch_size))
    x_vals.append(x_axis)
    y_vals.append(y_axis)
    ax.plot(x_axis, y_axis, marker='o', label=f"Batch size: {batch_size}")
    
    print(f"Batch size: {batch_size}")
    print(f"Cumulative Training Time (s): {cumulative_time}")
    print(f"Noise Scale: {noise_scale}")


ax.axvline(x=1, color='gray', linestyle='--')
ax.plot(x_vals, y_vals, linestyle='-', color='#1f77b4', linewidth=4, zorder=0)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Batch Size / Noise Scale")
ax.set_ylabel(fr"${{\epsilon_\text{{B}}}} / {{\epsilon_\text{{max}}}}$")
ax.set_title("Predicted Training Speed")
ax.legend()

fig.savefig(os.path.join(save_dir, "lr_scaling_centralized.png"))

plt.tight_layout()
plt.show()


def load_experiment_pickle(file_name):
    """Load experiment results from a pickle file.
    
    Args:
        file_name (str): Path to the results file
        
    Returns:
        tuple: (batch_size, parameters_for_each_round, hist)
    """
    with open(file_name, 'rb') as f:  # Note: 'rb' for binary read mode
        results_dict = pickle.load(f)
    
    return (
        results_dict['batch_size'],
        results_dict['parameters_for_each_round'],
        results_dict['history']
    )


###################################################### LOCAL BATCH SIZE #####################################################

experiment_batch_sizes = [16, 32, 64, 128, 256]
save_dir = os.path.join(project_root, "plots", "federated")
os.makedirs(save_dir, exist_ok=True)

total_batch_results = []
for batch_size in experiment_batch_sizes:
    save_file_name = os.path.join(project_root, "results", f"federated_batch_results_{batch_size}.pkl")
    total_batch_results.append(load_experiment_pickle(save_file_name))

# Create first figure: Compute Budget vs Training Time
fig, ax = plt.subplots(figsize=(10, 6))

accuracy_thresholds = [0.4, 0.5, 0.6]
for accuracy_threshold in accuracy_thresholds:
    x_vals = []
    y_vals = []
    for batch_size, params, hist in total_batch_results:
        times = []
        samples = []
        
        num_rounds = len(hist.metrics_centralized['accuracy']) - 1
        for round_idx, centralized_accuracy in hist.metrics_centralized['accuracy']:
            if centralized_accuracy >= accuracy_threshold:
                num_rounds = round_idx
                break

        for round_idx, round_metrics in hist.metrics_distributed_fit['training_time']:
            if round_idx > num_rounds:
                break
            round_times = [t for _, t in round_metrics['all']]
            times.append(np.mean(round_times))
            
        for round_idx, round_metrics in hist.metrics_distributed_fit['samples_processed']:
            if round_idx > num_rounds:
                break
            round_samples = [s for _, s in round_metrics['all']]
            samples.append(np.sum(round_samples))
        
        cumulative_time = np.sum(times)
        total_samples = np.sum(samples)
        x_vals.append(cumulative_time)
        y_vals.append(total_samples)
        ax.plot(cumulative_time, total_samples, marker='o', 
                label=f"Local Batch size: {batch_size}, Acc threshold: {accuracy_threshold}")

    ax.plot(x_vals, y_vals, linestyle='--', color='black')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Total Training Time (s)")
ax.set_ylabel("Compute Budget (Total Samples Processed)")
ax.set_title("Compute Budget vs. Total Training Time")
ax.legend()

fig.savefig(os.path.join(save_dir, "compute_budget_federated_local_batch_size.png"))

# Create second figure: Noise Scale Analysis
fig, ax = plt.subplots(figsize=(10, 6))

x_vals = []
y_vals = []
for batch_size, params, hist in total_batch_results:
    noise_scales = []
    for round_idx, round_metrics in hist.metrics_distributed_fit['noise_scale']:
        round_noise_scales = [ns for _, ns in round_metrics['all']]
        noise_scale = np.mean(round_noise_scales)
        noise_scales.append(noise_scale)
    
    avg_noise_scale = np.mean(noise_scales)
    x_axis = batch_size / (avg_noise_scale + 1e-10)
    y_axis = 1 / (1 + (avg_noise_scale / batch_size))
    x_vals.append(x_axis)
    y_vals.append(y_axis)
    
    ax.plot(x_axis, y_axis, marker='o', label=f"Local Batch size: {batch_size}")
    
    print(f"Batch size: {batch_size}")
    print(f"Avg Noise Scale: {avg_noise_scale}")

ax.axvline(x=1, color='gray', linestyle='--')
ax.plot(x_vals, y_vals, linestyle='-', color='#1f77b4', linewidth=4, zorder=0)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Batch Size / Noise Scale")
ax.set_ylabel(fr"${{\epsilon_\text{{B}}}} / {{\epsilon_\text{{max}}}}$")
ax.set_title("Predicted Training Speed")
ax.legend()

fig.savefig(os.path.join(save_dir, "lr_scaling_federated_local_batch_size.png"))

plt.tight_layout()
plt.show()


###################################################### COHORT SIZE #####################################################


time_per_round = 0.00427
cohort_sizes =  [5, 10, 20, 50, 75, 100]
total_cohort_results = []
for cohort_size in cohort_sizes:

    total_cohort_results.append(load_experiment_pickle(f"results/federated_cohort_results_{cohort_size}.pkl"))

# Create first figure: Compute Budget vs Training Time
fig, ax = plt.subplots(figsize=(10, 6))

accuracy_thresholds = [0.4, 0.5, 0.6]
for accuracy_threshold in accuracy_thresholds:
    x_vals = []
    y_vals = []
    for cohort_size, params, hist in total_cohort_results:
        times = []
        samples = []

        num_rounds = len(hist.metrics_centralized['accuracy']) - 1
        for round_idx, centralized_accuracy in hist.metrics_centralized['accuracy']:
            if centralized_accuracy >= accuracy_threshold:
                num_rounds = round_idx
                break

        cumulative_time = num_rounds * time_per_round

        for round_idx, round_metrics in hist.metrics_distributed_fit['samples_processed']:
            if round_idx > num_rounds:
                break
            round_samples = [s for _, s in round_metrics['all']]
            samples.append(np.sum(round_samples))
        
        total_samples = np.sum(samples)
        x_vals.append(cumulative_time)
        y_vals.append(total_samples)
        ax.plot(cumulative_time, total_samples, marker='o', label=f"Cohort size: {cohort_size}, Accuracy threshold: {accuracy_threshold}")

    ax.plot(x_vals, y_vals, linestyle='--', color='black')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Total Training Time (s)")
ax.set_ylabel("Compute Budget (Total Samples Processed)")
ax.set_title("Compute Budget vs. Total Training Time")
ax.legend()

fig.savefig(os.path.join(save_dir, "compute_budget_federated_cohort_size.png"))

# Create second figure: Noise Scale Analysis
fig, ax = plt.subplots(figsize=(10, 6))

x_vals = []
y_vals = []
for cohort_size, params, hist in total_cohort_results:
    noise_scales = []
    for round_idx, round_metrics in hist.metrics_distributed_fit['noise_scale']:
        round_noise_scales = [ns for _, ns in round_metrics['all']]
        noise_scale = np.mean(round_noise_scales)
        noise_scales.append(noise_scale)
    
    avg_noise_scale = np.mean(noise_scales)
    x_axis = cohort_size / (avg_noise_scale + 1e-10)
    y_axis = 1 / (1 + (avg_noise_scale / cohort_size))
    x_vals.append(x_axis)
    y_vals.append(y_axis)
    
    ax.plot(x_axis, y_axis, marker='o', label=f"Cohort size: {cohort_size}")

ax.axvline(x=1, color='gray', linestyle='--')
ax.plot(x_vals, y_vals, linestyle='-', color='#1f77b4', linewidth=4, zorder=0)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Cohort Size / Noise Scale")
ax.set_ylabel(fr"${{\epsilon_\text{{B}}}} / {{\epsilon_\text{{max}}}}$")
ax.set_title("Predicted Training Speed")
ax.legend()

fig.savefig(os.path.join(save_dir, "lr_scaling_federated_cohort_size.png"))

plt.tight_layout()
plt.show()



###################################################### GLOBAL BATCH SIZE #####################################################


# global batch size

time_per_round = 0.00427
cs_bs_pairs = [(5, 20), (20, 50), (50, 200), (100, 250), (100, 1000), (100, 2000), (100, 4000), (100, 12000)]
total_global_batch_results = []
for cohort_size, batch_size in cs_bs_pairs:
    global_batch_size = cohort_size * batch_size

    total_global_batch_results.append(load_experiment_pickle(f"results/federated_global_batch_results_{global_batch_size}.pkl"))

# Create first figure: Compute Budget vs Training Time
fig, ax = plt.subplots(figsize=(10, 6))

accuracy_thresholds = [0.4, 0.5, 0.6]
for accuracy_threshold in accuracy_thresholds:
    x_vals = []
    y_vals = []
    for batch_size, params, hist in total_global_batch_results:
        times = []
        samples = []
        
        num_rounds = len(hist.metrics_centralized['accuracy']) - 1
        for round_idx, centralized_accuracy in hist.metrics_centralized['accuracy']:
            if centralized_accuracy >= accuracy_threshold:
                num_rounds = round_idx
                break

        cumulative_time = num_rounds * time_per_round

        for round_idx, round_metrics in hist.metrics_distributed_fit['samples_processed']:
            if round_idx > num_rounds:
                break
            round_samples = [s for _, s in round_metrics['all']]
            samples.append(np.sum(round_samples))
        
        total_samples = np.sum(samples)
        x_vals.append(cumulative_time)
        y_vals.append(total_samples)
        ax.plot(cumulative_time, total_samples, marker='o', 
                label=f"Global batch size: {batch_size}, Acc threshold: {accuracy_threshold}")

    ax.plot(x_vals, y_vals, linestyle='--', color='black')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Total Training Time (s)")
ax.set_ylabel("Compute Budget (Total Samples Processed)")
ax.set_title("Compute Budget vs. Total Training Time")
ax.legend()

fig.savefig(os.path.join(save_dir, "compute_budget_federated_global_batch_size.png"))

# Create second figure: Noise Scale Analysis
fig, ax = plt.subplots(figsize=(10, 6))

x_vals = []
y_vals = []
rel_y = None
for batch_size, params, hist in total_global_batch_results:
    for round_idx, round_metrics in hist.metrics_distributed_fit['noise_scale']:
        round_noise_scales = [ns for _, ns in round_metrics['all']]

    
    avg_noise_scale = 108600.0 # from critical bs plot
    x_axis = batch_size / avg_noise_scale
    y_axis = 1 / (1 + (avg_noise_scale / batch_size))
    x_vals.append(x_axis)
    y_vals.append(y_axis)
    
    ax.plot(x_axis, y_axis, marker='o', label=f"Global batch size: {batch_size}")


ax.axvline(x=1, color='gray', linestyle='--')
# send this to behind the dots
ax.plot(x_vals, y_vals, linestyle='-', color='#1f77b4', linewidth=4, zorder=0)
ax.set_xlabel("Global Batch Size / Noise Scale")
ax.set_ylabel(fr"${{\epsilon_\text{{B}}}} / {{\epsilon_\text{{max}}}}$")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title("Predicted Training Speed")
ax.legend()

fig.savefig(os.path.join(save_dir, "lr_scaling_federated_global_batch_size.png"))

plt.tight_layout()
plt.show()
