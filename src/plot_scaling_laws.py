import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle

from scipy.optimize import curve_fit

def load_experiment(filename):
    with open(filename, "r") as f:
        return json.load(f)

def create_lr_scaling_plot(total_results, save_dir, save_file, analysis_name, avg_noise_scale):
    # Create second figure: Noise Scale Analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = []
    y_vals = []
    rel_y = None
    for batch_size, *_ in total_results:
        
        x_axis = batch_size / avg_noise_scale
        y_axis = 1 / (1 + (avg_noise_scale / batch_size))
        x_vals.append(x_axis)
        y_vals.append(y_axis)
        
        ax.plot(x_axis, y_axis, marker='o', label=f"{analysis_name}: {batch_size}")


    ax.axvline(x=1, color='gray', linestyle='--')
    # send this to behind the dots
    ax.plot(x_vals, y_vals, linestyle='-', color='#1f77b4', linewidth=4, zorder=0)
    ax.set_xlabel(f"{analysis_name} / Noise Scale")
    ax.set_ylabel(fr"${{\epsilon_\text{{B}}}} / {{\epsilon_\text{{max}}}}$")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Predicted Training Speed")
    ax.legend()

    fig.savefig(os.path.join(save_dir, save_file))

    plt.tight_layout()
    plt.show()


def create_compute_budget_plot(total_results, save_dir, save_file, analysis_name, accuracy_thresholds, accuracy_colors, mode, universal_round_time=None):
    # Create first figure: Compute Budget vs Training Time
    fig, ax = plt.subplots(figsize=(10, 6))

    for accuracy_threshold in accuracy_thresholds:
        x_vals = []
        y_vals = []
        for batch_size, params, hist in total_results:
            times = []
            samples = []
            
            num_rounds = len(hist.metrics_centralized['accuracy']) - 1
            for round_idx, centralized_accuracy in hist.metrics_centralized['accuracy']:
                if centralized_accuracy >= accuracy_threshold:
                    num_rounds = round_idx
                    break

            if mode == "1":
                for round_idx, round_metrics in hist.metrics_distributed_fit['training_time']:
                    if round_idx > num_rounds:
                        break
                    round_times = [t for _, t in round_metrics['all']]
                    times.append(np.mean(round_times))
                cumulative_time = np.sum(times)
            elif mode == "2":
                assert universal_round_time is not None, "Universal round time must be provided for mode 2"
                cumulative_time = universal_round_time * num_rounds

                
            for round_idx, round_metrics in hist.metrics_distributed_fit['samples_processed']:
                if round_idx > num_rounds:
                    break
                round_samples = [s for _, s in round_metrics['all']]
                samples.append(np.sum(round_samples))
            
            total_samples = np.sum(samples)
            x_vals.append(cumulative_time)
            y_vals.append(total_samples)
            ax.plot(cumulative_time, total_samples, marker='o', color=accuracy_colors[accuracy_threshold])

        ax.plot(x_vals, y_vals, linestyle='--', color=accuracy_colors[accuracy_threshold], 
                label=f"Accuracy threshold: {accuracy_threshold}")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Compute Budget (Total Samples Processed)")
    ax.set_title(f"Compute Budget vs. Total Training Time - Federated {analysis_name}")
    ax.legend()

    fig.savefig(os.path.join(save_dir, save_file))
    


def create_compute_budget_plot_centralized(centralized_experiment_results, save_dir, save_file, accuracy_thresholds, accuracy_colors):
    # Create first figure: Compute Budget vs Training Time
    fig, ax = plt.subplots(figsize=(10, 6))

    for accuracy_threshold in accuracy_thresholds:
        x_vals = []
        y_vals = []
        color = accuracy_colors[accuracy_threshold]
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
            ax.plot(cumulative_time, compute_budget, marker='o', color=color)
            
            print(f"Batch size: {batch_size}, Acc threshold: {accuracy_threshold}")
            print(f"Total Training Time (s): {cumulative_time}")
            print(f"Compute Budget (samples): {compute_budget}")
        
        ax.plot(x_vals, y_vals, linestyle='--', color=color, label=f"Accuracy threshold: {accuracy_threshold}")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Total Training Time (s)")
    ax.set_ylabel("Compute Budget (Total Samples Processed)")
    ax.set_title("Compute Budget vs. Total Training Time - Centralized")
    ax.legend()

    fig.savefig(os.path.join(save_dir, save_file))


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


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ###################################################### CENTRALIZED ######################################################

    save_dir = os.path.join(project_root, "plots", "centralized")
    os.makedirs(save_dir, exist_ok=True)

    centralized_experiment_batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    centralized_experiment_results = [
        (batch_size, load_experiment(os.path.join(project_root, "results", "centralized", f"centralized_experiment_results_{batch_size}.json")))
        for batch_size in centralized_experiment_batch_sizes
    ]

    accuracy_thresholds = [0.4, 0.5, 0.6] #, 0.7]
    accuracy_colors = {
        0.4: '#1f77b4',  # Blue
        0.5: '#ff7f0e',  # Orange
        0.6: '#2ca02c',  # Green
        0.7: '#d62728'   # Red
    }


    create_compute_budget_plot_centralized(
        centralized_experiment_results=centralized_experiment_results,
        save_dir=save_dir,
        save_file="compute_budget_centralized.png",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
    )


    create_lr_scaling_plot(
        total_results=centralized_experiment_results, 
        save_dir=save_dir,
        save_file="lr_scaling_centralized.png",
        analysis_name="Batch Size",
        avg_noise_scale=60426, # TODO: COMPUTE THE CORRECT ONE
    )


    ###################################################### LOCAL BATCH SIZE #####################################################
    save_dir = os.path.join(project_root, "plots", "federated")
    os.makedirs(save_dir, exist_ok=True)

    experiment_batch_sizes = [16, 32, 64, 128, 256]
    time_per_round = 0.00427

    ################################# NON-IID #################################
    
    total_batch_results = []
    for batch_size in experiment_batch_sizes:
        save_file_name = os.path.join(project_root, "results", f"federated_batch_results_{batch_size}.pkl")
        total_batch_results.append(load_experiment_pickle(save_file_name))

    create_compute_budget_plot(
        total_results=total_batch_results,
        save_dir=save_dir,
        save_file="compute_budget_federated_local_batch_size.png",
        analysis_name="Local Batch Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )


    ################################# IID #################################

    total_batch_results = []
    for batch_size in experiment_batch_sizes:
        save_file_name = os.path.join(project_root, "results", f"IID_federated_local_batch_results_{batch_size}.pkl")
        total_batch_results.append(load_experiment_pickle(save_file_name))

    create_compute_budget_plot(
        total_results=total_batch_results,
        save_dir=save_dir,
        save_file="iid_compute_budget_federated_local_batch_size.png",
        analysis_name="Local Batch Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )



    ###################################################### COHORT SIZE #####################################################

    cohort_sizes =  [5, 10, 20, 50, 75, 100]

    ################################# NON-IID #################################
    total_cohort_results = []
    for cohort_size in cohort_sizes:
        total_cohort_results.append(load_experiment_pickle(f"results/federated_cohort_results_{cohort_size}.pkl"))

    create_compute_budget_plot(
        total_results=total_cohort_results,
        save_dir=save_dir,
        save_file="compute_budget_federated_cohort_size.png",
        analysis_name="Cohort Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )


    ################################# IID #################################

    total_cohort_results = []
    for cohort_size in cohort_sizes:
        total_cohort_results.append(load_experiment_pickle(f"results/IID_federated_cohort_results_{cohort_size}.pkl"))


    create_compute_budget_plot(
        total_results=total_cohort_results,
        save_dir=save_dir,
        save_file="iid_compute_budget_federated_cohort_size.png",
        analysis_name="Cohort Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )



    ###################################################### GLOBAL BATCH SIZE #####################################################

    cs_bs_pairs = [(5, 20), (20, 50), (50, 200), (100, 250), (100, 1000), (100, 2000), (100, 4000), (100, 12000)]

    ################################# NON-IID #################################
    total_global_batch_results = []
    for cohort_size, batch_size in cs_bs_pairs:
        global_batch_size = cohort_size * batch_size
        total_global_batch_results.append(load_experiment_pickle(f"results/federated_global_batch_results_{global_batch_size}.pkl"))


    create_compute_budget_plot(
        total_results=total_global_batch_results,
        save_dir=save_dir,
        save_file="compute_budget_federated_global_batch_size.png",
        analysis_name="Global Batch Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )

    create_lr_scaling_plot(
        total_results=total_global_batch_results, 
        save_dir=save_dir,
        save_file="lr_scaling_federated_global_batch_size.png",
        analysis_name="Global Batch Size",
        avg_noise_scale=108600.0 # from critical bs plot
    )



    ################################# IID ##########################################
    total_global_batch_results = []
    for cohort_size, batch_size in cs_bs_pairs:
        global_batch_size = cohort_size * batch_size
        total_global_batch_results.append(load_experiment_pickle(f"results/IID_federated_global_batch_results_{global_batch_size}.pkl"))

    create_compute_budget_plot(
        total_results=total_global_batch_results,
        save_dir=save_dir,
        save_file="iid_compute_budget_federated_global_batch_size.png",
        analysis_name="Global Batch Size",
        accuracy_thresholds=accuracy_thresholds,
        accuracy_colors=accuracy_colors,
        mode="2",
        universal_round_time=time_per_round,
    )

    create_lr_scaling_plot(
        total_results=total_global_batch_results, 
        save_dir=save_dir,
        save_file="iid_lr_scaling_federated_global_batch_size.png",
        analysis_name="Global Batch Size",
        avg_noise_scale=10000000000000000 # IDK this yet
    )


if __name__ == "__main__":
    main()