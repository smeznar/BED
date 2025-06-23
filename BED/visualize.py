import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Function to expand the run to all evaluation steps with best-so-far error
def expand_run(run):
    run = sorted(run, key=lambda x: x[1])
    max_eval = run[-1][1]
    errors = np.full(max_eval + 1, np.inf)
    best_so_far = float('inf')
    last_index = 0
    for error, evals in run:
        for i in range(last_index, evals + 1):
            errors[i] = max(best_so_far, 1e-8)
        best_so_far = min(best_so_far, error)
        last_index = evals + 1
    return errors

# Aggregate and collect all expanded runs
def aggregate_curves(runs):
    expanded = [expand_run(run) for run in runs]
    max_len = 601
    padded_runs = [np.pad(r, (0, max_len - len(r)), constant_values=r[-1]) for r in expanded]
    stacked = np.stack(padded_runs)
    mean_curve = np.mean(stacked, axis=0)
    lower = np.min(stacked, axis=0)
    upper = np.max(stacked, axis=0)
    return np.arange(max_len), mean_curve, lower, upper

def plot_dataset(dataset_path):
    dataset_name = ".".join(dataset_path.split("/")[-1].split("_")[-1].split(".")[:-1])
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Plotting
    plt.figure(figsize=(10, 6))

    for measure, runs in reversed(data.items()):
        evals, avg_errors, lower, upper = aggregate_curves(runs)
        print("Measure", avg_errors[-1])
        plt.plot(evals[100:], avg_errors[100:], label=measure)
        plt.fill_between(evals[100:], lower[100:], upper[100:], alpha=0.2)
    print("------------------------------------")
    plt.yscale('log')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Error (Averaged)')
    plt.title(f'Convergence Curves by Distance Measure: {dataset_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    all_datasets = glob("../results/simple_discovery/results_exploration*")
    datasets = np.random.choice(all_datasets, 5)
    for dataset in datasets:
        plot_dataset(dataset)
