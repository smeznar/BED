from argparse import ArgumentParser
from pathlib import Path

import editdistance
from SRToolkit.dataset import SRBenchmark
from SRToolkit.utils import generate_n_expressions
from SRToolkit.utils import tokens_to_tree as ttt
import numpy as np
import pandas as pd
import seaborn as sns
import zss
import matplotlib.pyplot as plt

from bed import BED
from utils import expr_to_zss


# sns.set_theme(font_scale=1, palette="Set2", rc={'figure.figsize': (10, 8), 'text.usetex': True,
#                                                 'axes.titlesize': 30, "axes.labelsize": 25})
np.random.seed()

grammar = """
E -> E '+' F [0.2004]
E -> E '-' F [0.1108]
E -> F [0.6888]
F -> F '*' T [0.3349]
F -> F '/' T [0.1098]
F -> T [0.5553]
T -> 'C' [0.1174]
T -> R [0.1746]
T -> V [0.708]
R -> '(' E ')' [0.6841]
R -> E '^2' [0.00234]
R -> E '^3' [0.00126]
R -> 'sin' '(' E ')' [0.028]
R -> 'cos' '(' E ')' [0.049]
R -> 'sqrt' '(' E ')' [0.0936]
R -> 'exp' '(' E ')' [0.0878]
R -> 'ln' '(' E ')' [0.0539]
"""

# Use LaTeX fonts for publication figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 400
})

# --- Aggregation ---
def aggregate_results(losses, aggregator="mean"):
    losses = np.asarray(losses)
    if aggregator == "mean":
        return np.cumsum(losses) / np.arange(1, len(losses) + 1)
    elif aggregator == "median":
        return np.array([np.median(losses[:i+1]) for i in range(len(losses))])
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")


# --- Build plotting data ---
def build_df(optimal, edit, tree, main_curve, main_name, y_text, min_val=1.1e-4):
    dfs = []
    dfs.append(pd.DataFrame({
        "Number of Closest Expressions": np.arange(1, len(optimal) + 1),
        y_text: np.clip(optimal, min_val, None),
        "Metric": "Optimal"
    }))
    dfs.append(pd.DataFrame({
        "Number of Closest Expressions": np.arange(1, len(edit) + 1),
        y_text: np.clip(edit, min_val, None),
        "Metric": "Edit"
    }))
    dfs.append(pd.DataFrame({
        "Number of Closest Expressions": np.arange(1, len(tree) + 1),
        y_text: np.clip(tree, min_val, None),
        "Metric": "TreeEdit"
    }))
    for row in main_curve:
        dfs.append(pd.DataFrame({
            "Number of Closest Expressions": np.arange(1, len(row) + 1),
            y_text: np.clip(row, min_val, None),
            "Metric": main_name
        }))
    return pd.concat(dfs, ignore_index=True)

def plot_metric(df, main_name, ds, dataset, palette, y_text, save_name):
    plt.figure(figsize=(8, 6))
    # Baselines as fixed lines
    for metric in ["Optimal", "Edit", "TreeEdit"]:
        subset = df[df["Metric"] == metric]
        plt.plot(
            subset["Number of Closest Expressions"],
            subset[y_text],
            label=metric,
            color=palette[metric],
            linewidth=1.5
        )
    # Main metric with shaded CI
    main_subset = df[df["Metric"] == main_name]
    mean_vals = main_subset.groupby("Number of Closest Expressions")[y_text].mean()
    ci_lower = main_subset.groupby("Number of Closest Expressions")[y_text].quantile(0.025)
    ci_upper = main_subset.groupby("Number of Closest Expressions")[y_text].quantile(0.975)
    plt.plot(mean_vals.index, mean_vals.values, color=palette[main_name], label=main_name, linewidth=2)
    plt.fill_between(mean_vals.index, ci_lower.values, ci_upper.values, color=palette[main_name], alpha=0.25)

    plt.yscale("log")
    plt.ylim(max(1e-4, df[df["Metric"] == "Optimal"][y_text].min() * 0.95))
    plt.xlabel("Number of Closest Expressions")
    plt.ylabel(y_text)
    expr_tex = ttt(dataset.ground_truth, dataset.symbols).to_latex(dataset.symbols)
    plt.title(f"Expression ({ds}): {expr_tex}", fontsize=14)
    plt.legend(frameon=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name, dpi=400, bbox_inches="tight")
    plt.close()


def plot_ranking_baselines(x_data, y_data, metric_names, baseline_labels, palette, y_axis_label, plot_title):
    plt.figure(figsize=(8, 6))

    # Create a DataFrame from the raw lists for easier plotting
    df = pd.DataFrame({
        "Number of Closest Expressions": x_data,
        "Rank Value": y_data,
        "Metric": metric_names
    })

    # Plot each baseline
    for metric in baseline_labels:
        subset = df[df["Metric"] == metric]
        plt.plot(
            subset["Number of Closest Expressions"],
            subset["Rank Value"],
            label=metric,
            color=palette[metric],
            linewidth=2,
            # marker='o', # Add markers to distinguish points
            # markersize=5
        )

    # Set plot labels, title, and legend
    plt.xlabel("Number of Closest Expressions", fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.legend(frameon=True, fontsize=10) # Display legend with a frame
    plt.grid(True, linestyle=':', alpha=0.7) # Add a subtle grid for readability

    # Set y-axis limits dynamically based on data, ensuring it starts from 0 or slightly below min
    min_rank = df["Rank Value"].min()
    max_rank = df["Rank Value"].max()
    plt.ylim(max(0, min_rank - (max_rank - min_rank) * 0.1), max_rank + (max_rank - min_rank) * 0.1)

    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    # plt.show()
    plt.savefig("../results/figures/average_rank_mean.png", dpi=400, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser(prog='Smoothness to true equation',
                            description='A script for testing the smoothness of the error landscape')
    parser.add_argument("-dataset", type=str)
    parser.add_argument("-compute", action="store_true")
    parser.add_argument("-aggregator", type=str, default="mean")
    args = parser.parse_args()
    top_n = 250

    if args.compute:
        number_of_runs = 10
        number_of_expressions = 100000
        num_points = 64
        threshold = 1e20
        dataset = SRBenchmark.feynman("../data/feynman").create_dataset(args.dataset)
        num_variables = dataset.X.shape[1]
        evaluator = dataset.create_evaluator()
        for i in range(num_variables):
            grammar += f"\nV -> 'X_{i}' [{1 / num_variables}]"
        expressions = generate_n_expressions(grammar, number_of_expressions, max_expression_length=40, verbose=False)
        losses = np.array([evaluator.evaluate_expr(expr) for expr in expressions])
        losses = np.nan_to_num(losses, nan=np.inf, posinf=np.inf, neginf=np.inf)
        invalid_indices = np.where(losses > 1e20)[0]

        # Optimal
        distance = np.sort(losses)[:top_n]
        np.savez(f"../results/smoothness/Optimal/{args.dataset}.npz", rmse=distance,
                 invalid_count=0)

        # Edit
        edit = []
        for expr in expressions:
            edit.append(editdistance.eval(dataset.ground_truth, expr))
        edit_ind = np.argsort(edit)
        invalid_edit = np.isin(edit_ind[:top_n], invalid_indices).sum()
        result_edit = losses[edit_ind[~np.isin(edit_ind, invalid_indices)][:top_n]]
        np.savez(f"../results/smoothness/Edit/{args.dataset}.npz", rmse=result_edit, invalid_count=invalid_edit)

        # Tree edit
        tree = []
        zss_ground_truth = expr_to_zss(ttt(dataset.ground_truth, dataset.symbols))
        for expr in expressions:
            tree.append(zss.simple_distance(zss_ground_truth, expr_to_zss(ttt(expr, dataset.symbols))))
        tree_ind = np.argsort(tree)
        invalid_tree = np.isin(tree_ind[:top_n], invalid_indices).sum()
        result_tree = losses[tree_ind[~np.isin(tree_ind, invalid_indices)][:top_n]]
        np.savez(f"../results/smoothness/TreeEdit/{args.dataset}.npz", rmse=result_tree, invalid_count=invalid_tree)

        # BED
        results_bed = np.zeros((number_of_runs, top_n))
        invalid_bed = np.zeros((number_of_runs))
        for i in range(number_of_runs):
            x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
            distances = BED([dataset.ground_truth], expressions2=expressions, normalize=False,
                            x=dataset.X[x_indices], const_bounds=(-5, 5)).calculate_distances()
            bed_ind = np.argsort(distances[0])
            invalid_bed[i] = np.isin(bed_ind[:top_n], invalid_indices).sum()
            results_bed[i, :] = losses[bed_ind[~np.isin(bed_ind, invalid_indices)][:top_n]]
        np.savez(f"../results/smoothness/BED/{args.dataset}.npz", rmse=results_bed, invalid_count=invalid_bed)

        # NBED
        results_nbed = np.zeros((number_of_runs, top_n))
        invalid_nbed = np.zeros((number_of_runs))
        for i in range(number_of_runs):
            x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
            distances = BED([dataset.ground_truth], expressions2=expressions, normalize=True,
                            x=dataset.X[x_indices], const_bounds=(-5, 5)).calculate_distances()
            bed_ind = np.argsort(distances[0])
            invalid_nbed[i] = np.isin(bed_ind[:top_n], invalid_indices).sum()
            results_nbed[i, :] = losses[bed_ind[~np.isin(bed_ind, invalid_indices)][:top_n]]
        np.savez(f"../results/smoothness/NBED/{args.dataset}.npz", rmse=results_nbed, invalid_count=invalid_nbed)
    else:
        if args.dataset == "all":
            datasets = SRBenchmark.feynman("../data/feynman").list_datasets(verbose=False)
            ranks = np.zeros((3, top_n))
            invalid = np.zeros(3)
            for dataset in datasets:
                base_path = Path("../results/smoothness")
                # Load results
                edit_obj = np.load(base_path / "Edit" / f"{dataset}.npz")
                edit, edit_invalid = edit_obj["rmse"], edit_obj["invalid_count"]
                edit = aggregate_results(edit, args.aggregator)
                tree_obj = np.load(base_path / "TreeEdit" / f"{dataset}.npz")
                tree, tree_invalid = tree_obj["rmse"], tree_obj["invalid_count"]
                tree = aggregate_results(tree, args.aggregator)
                bed_obj = np.load(base_path / "BED" / f"{dataset}.npz")
                bed, bed_invalid = bed_obj["rmse"], bed_obj["invalid_count"]
                bed = np.array([aggregate_results(bed[i], args.aggregator) for i in range(len(bed))])
                bed = np.mean(bed, axis=0) if args.aggregator == "mean" else np.median(bed, axis=0)

                invalid[0] += edit_invalid
                invalid[1] += tree_invalid
                invalid[2] += np.mean(bed_invalid)

                sorted_indices = np.argsort(np.stack([edit, tree, bed]), axis=0)
                for i in range(top_n):
                    for j in range(3):
                        ranks[sorted_indices[j, i], i] += j + 1
            ranks = ranks / len(datasets)
            invalid = invalid / len(datasets)
            print("Edit invalid:", invalid[0])
            print("Tree invalid:", invalid[1])
            print("BED invalid:", invalid[2])

            x = []
            y = []
            baseline = []
            baseline_names = ["Edit", "TreeEdit", "BED"]
            for i in range(3):
                for j in range(top_n):
                    x.append(j+1)
                    y.append(ranks[i, j])
                    baseline.append(baseline_names[i])

            custom_palette = {
                "Edit": "#D55E00",  # Vermillion
                "TreeEdit": "#0072B2",  # Blue
                "BED": "#009E73",  # Bluish Green
            }

            plot_ranking_baselines(
                x_data=x,
                y_data=y,
                metric_names=baseline,
                baseline_labels=baseline_names,
                palette=custom_palette,
                y_axis_label="Average Rank",
                plot_title="Average Rank of Baselines for Closest Expressions"
            )


        else:
            # --- Dataset selection ---
            if args.dataset == "none":
                all_datasets = SRBenchmark.feynman("../data/feynman").list_datasets(verbose=False)
                ds = np.random.choice(all_datasets)
            else:
                ds = args.dataset

            dataset = SRBenchmark.feynman("../data/feynman").create_dataset(ds)
            y_text = "Mean Aggregated RMSE" if args.aggregator == "mean" else "Median Aggregated RMSE"
            # Load data
            base_path = Path("../results/smoothness")
            optimal_obj = np.load(base_path / "Optimal" / f"{ds}.npz")
            optimal, optimal_invalid = optimal_obj["rmse"], optimal_obj["invalid_count"]
            optimal = aggregate_results(optimal, args.aggregator)

            edit_obj = np.load(base_path / "Edit" / f"{ds}.npz")
            edit, edit_invalid = edit_obj["rmse"], edit_obj["invalid_count"]
            edit = aggregate_results(edit, args.aggregator)

            tree_obj = np.load(base_path / "TreeEdit" / f"{ds}.npz")
            tree, tree_invalid = tree_obj["rmse"], tree_obj["invalid_count"]
            tree = aggregate_results(tree, args.aggregator)

            bed_obj = np.load(base_path / "BED" / f"{ds}.npz")
            bed, bed_invalid = bed_obj["rmse"], bed_obj["invalid_count"]
            bed = np.array([aggregate_results(bed[i], args.aggregator) for i in range(len(bed))])

            nbed_obj = np.load(base_path / "NBED" / f"{ds}.npz")
            nbed, nbed_invalid = nbed_obj["rmse"], nbed_obj["invalid_count"]
            nbed = np.array([aggregate_results(nbed[i], args.aggregator) for i in range(len(nbed))])

            # Palette
            palette = {
                "Optimal": "#000000",  # Black
                "Edit": "#D55E00",  # Vermillion
                "TreeEdit": "#0072B2",  # Blue
                "BED": "#009E73",  # Bluish Green
                "NBED": "#009E73"  # Bluish Green
            }

            # Build and plot BED
            df_bed = build_df(optimal, edit, tree, bed, "BED", y_text)
            plot_metric(df_bed, "BED", ds, dataset, palette, y_text, save_name=f"plot_{ds}_BED.png")

            # Build and plot NBED
            # df_nbed = build_df(optimal, edit, tree, nbed, "NBED", y_text)
            # plot_metric(df_nbed, "NBED", ds, dataset, palette, y_text, save_name=f"plot_{ds}_NBED.png")

            print("Edit invalid:", edit_invalid)
            print("Tree invalid:", tree_invalid)
            print("BED invalid:", np.mean(bed_invalid), np.std(bed_invalid))
            print("NBED invalid:", np.mean(nbed_invalid), np.std(nbed_invalid))

    # name = []
    # y = []
    # x = []
    #
    #
    # x_i, y_i, names_i = rank_distance(edit, losses, "Edit")
    # x += x_i
    # y += y_i
    # name += names_i
    #
    # x_i, y_i, names_i = rank_distance(tree, losses, "Tree")
    # x += x_i
    # y += y_i
    # name += names_i
    #
    #
    # bed_min = np.zeros((10, 500))
    # for i, dm in enumerate(bed_distances):
    #     x_i, y_i, names_i = rank_distance(dm[0], losses, "BED")
    #     bed_min[i, :] = np.array(y_i)
    #     x += x_i
    #     y += y_i
    #     name += names_i
    #
    # bed_min = np.min(bed_min, axis=0)
    #
    # x += [i+1 for i in range(500)]
    # y += [v for v in bed_min]
    # name += ["BED min" for i in range(500)]
    #
    # x += [i+1 for i in range(500)]
    # y += [v for v in np.sort(losses)[:500]]
    # name += ["Optimal" for i in range(500)]
    #
    # data = pd.DataFrame(
    #     {"Number of Closest Expressions": x, f"Aggregated RMSE": y,
    #      "Metric": name})
    # sns_plot = sns.lineplot(x="Number of Closest Expressions", y=f"Aggregated RMSE",
    #                         hue="Metric",
    #                         data=data)
    # sns_plot.axes.set_title(f"Expression ({args.dataset}): {ttt(dataset.ground_truth, sl).to_latex(sl)}", fontsize=10)
    # # sns_plot.axes.yscale("log")
    # plt.yscale("log")
    # # plt.show()
    # plt.savefig(f"../results/smoothness_true/{args.dataset}.png", dpi=400)