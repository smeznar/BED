import os
import time
from argparse import ArgumentParser
from pathlib import Path

import editdistance
import torch
from SRToolkit.approaches.EDHiE import TreeDataset, HVAE, train_hvae, create_batch
from SRToolkit.dataset import SR_benchmark
from SRToolkit.utils import generate_n_expressions, SymbolLibrary
from SRToolkit.utils import tokens_to_tree as ttt
import numpy as np
import pandas as pd
import zss
import matplotlib.pyplot as plt
from rapidfuzz.distance import Jaro

from bed import BED, expr_to_zss, maximal_hausdorff
from finetune_snip import load_snip_encoder

# ---------------------------------------------------------------------------
# Symbol set that covers the Feynman grammar and the finetuned SNIP/HVAE model
# ---------------------------------------------------------------------------
_SMOOTHNESS_SYMBOLS = [
    "+", "-", "*", "/", "u-",
    "sqrt", "sin", "cos", "exp", "arcsin", "tanh",
    "ln", "^2", "^3", "pi", "C",
]

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 400
})

# ---------------------------------------------------------------------------
# Baseline registry — each entry: (display_name, result_dir, color)
# ---------------------------------------------------------------------------
_BASELINE_META = {
    "edit":     ("Edit",      "Edit",      "#E69F00"),
    "tree":     ("Tree edit", "TreeEdit",  "#D55E00"),
    "jaro":     ("Jaro",      "Jaro",      "#009E73"),
    "bed":      ("BED",       "BED",       "#CC79A7"),
    "maximal":  ("Maximal",   "Maximal",   "#56B4E9"),
    "hausdorff":("Hausdorff", "Hausdorff", "#0072B2"),
    "snip":     ("SNIP",      "SNIP",      "#F0E442"),
    "hvae":     ("HVAE",      "HVAE",      "#000000"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def aggregate_results(losses, aggregator="mean"):
    losses = np.asarray(losses)
    if aggregator == "mean":
        return np.cumsum(losses) / np.arange(1, len(losses) + 1)
    elif aggregator == "median":
        return np.array([np.median(losses[:i + 1]) for i in range(len(losses))])
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")


def _load_result(base_path, result_dir, dataset_name):
    """Load a .npz result file; return None if it doesn't exist."""
    p = base_path / result_dir / f"{dataset_name}.npz"
    if not p.exists():
        return None
    return np.load(p)


def plot_metric(det_curves, stoch_curves, ds, dataset, palette, y_text, save_name, aggregator, top_n):
    """
    det_curves:   {display_name: 1-D aggregated array}   (Edit, Tree, Jaro, SNIP, HVAE …)
    stoch_curves: {display_name: 2-D runs×top_n array}   (BED)
    """
    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    xs = np.arange(1, top_n + 1)

    for name, data in det_curves.items():
        plt.plot(xs[:len(data)], data, label=name, color=palette[name], linewidth=1.5)

    for name, runs in stoch_curves.items():
        agg = np.array([aggregate_results(runs[i], aggregator) for i in range(len(runs))])
        mean_vals = agg.mean(axis=0)
        ci_lower = np.percentile(agg, 2.5, axis=0)
        ci_upper = np.percentile(agg, 97.5, axis=0)
        plt.plot(xs[:len(mean_vals)], mean_vals, label=name, color=palette[name], linewidth=2)
        plt.fill_between(xs[:len(mean_vals)], ci_lower, ci_upper, color=palette[name], alpha=0.25)

    plt.yscale("log")
    plt.xlabel("Number of Closest Expressions")
    plt.ylabel(y_text)
    expr_tex = (ttt(dataset.ground_truth, dataset.symbol_library)
                .to_latex(dataset.symbol_library)
                .replace("C", "c").replace("X", "x"))
    plt.title(f"\\textbf{{Expression ({ds}): }}\\boldmath {expr_tex}", fontsize=14)
    plt.legend(title="\\textbf{Measure}", frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_name, dpi=400, bbox_inches="tight")
    plt.close()


def plot_ranking_baselines(baseline_curves, palette, y_axis_label, plot_title, aggregator):
    """
    baseline_curves: {display_name: 1-D aggregated array}
    """
    names = list(baseline_curves.keys())
    top_n = max(len(v) for v in baseline_curves.values())
    xs = np.arange(1, top_n + 1)

    stacked = np.stack([baseline_curves[n] for n in names])  # (n_baselines, top_n)
    sorted_indices = np.argsort(stacked, axis=0)  # rank 0 = best

    x_list, y_list, label_list = [], [], []
    for i in range(top_n):
        for rank in range(len(names)):
            x_list.append(i + 1)
            y_list.append(rank + 1)
            label_list.append(names[sorted_indices[rank, i]])

    df = pd.DataFrame({"Number of Closest Expressions": x_list, "Rank Value": y_list, "Metric": label_list})

    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    for name in names:
        subset = df[df["Metric"] == name]
        mean_vals = subset.groupby("Number of Closest Expressions")["Rank Value"].mean()
        std_vals = subset.groupby("Number of Closest Expressions")["Rank Value"].std().fillna(0)
        plt.plot(mean_vals.index, mean_vals.values, color=palette[name], label=name, linewidth=2)
        plt.fill_between(
            mean_vals.index,
            np.clip(mean_vals - std_vals, 1, len(names)),
            np.clip(mean_vals + std_vals, 1, len(names)),
            color=palette[name], alpha=0.25,
        )

    plt.xlabel("Number of Closest Expressions", fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.title("\\textbf{" + plot_title + "}", fontsize=14)
    plt.legend(title="\\textbf{Measure}", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.ylim(0.9, len(names) + 0.1)
    plt.tight_layout()
    plt.savefig(f"../results/smoothness/figures/average_rank_{aggregator}.png", dpi=400, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# HVAE helpers
# ---------------------------------------------------------------------------

def _build_hvae_sl(base_symbols, num_variables):
    sl = SymbolLibrary.from_symbol_list(base_symbols, num_variables=num_variables)
    return sl


def _get_hvae_model(hvae_sl, latent_size, model_path):
    if model_path and Path(model_path).exists():
        model = HVAE(len(hvae_sl), latent_size, hvae_sl)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    # Train fresh
    expressions = generate_n_expressions(hvae_sl, 50000, max_expression_length=40)
    expr_trees = [ttt(expr, hvae_sl) for expr in expressions]
    trainset = TreeDataset(expr_trees)
    model = HVAE(len(hvae_sl), latent_size, hvae_sl)
    train_hvae(model, trainset, hvae_sl, epochs=20, max_beta=0.03)
    if model_path:
        os.makedirs(Path(model_path).parent, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Smoothness to true equation",
        description="Tests how well each distance measure ranks candidate expressions by closeness to the ground truth.",
    )
    parser.add_argument("-dataset",    type=str)
    parser.add_argument("-compute",    action="store_true")
    parser.add_argument("-aggregator", type=str, default="median")
    parser.add_argument(
        "--baselines", type=str, default="edit,bed",
        help="Comma-separated baselines to compute/plot. "
             "Choices: edit, tree, jaro, bed, maximal, hausdorff, snip, hvae. "
             "Example: --baselines edit,bed,maximal,hausdorff,snip,hvae",
    )
    # SNIP
    parser.add_argument("--snip_checkpoint",      type=str, default="Multimodal-Math-Pretraining-main/weights/snip-10dmax-finetuned-smoothness/best_checkpoint.pth",
                        help="Path to finetuned SNIP checkpoint (required when snip in --baselines).")
    parser.add_argument("--snip_base_checkpoint", type=str, default=None,
                        help="Path to the original pretrained SNIP checkpoint for architecture params "
                             "(needed for checkpoints saved before params were embedded).")
    # HVAE
    parser.add_argument("--hvae_model",       type=str, default=None,
                        help="Path to load (or save after training) HVAE weights.")
    parser.add_argument("--hvae_latent_size", type=int, default=32)

    args = parser.parse_args()
    baselines = [b.strip().lower() for b in args.baselines.split(",")]
    top_n = 250

    # -----------------------------------------------------------------------
    # Compute mode
    # -----------------------------------------------------------------------
    if args.compute:
        number_of_runs = 10
        number_of_expressions = 1000
        num_points = 64

        dataset = SR_benchmark.feynman("../data/feynman").create_dataset(args.dataset)
        num_variables = dataset.X.shape[1]
        evaluator = dataset.create_evaluator()

        full_grammar = grammar
        for i in range(num_variables):
            full_grammar += f"\nV -> 'X_{i}' [{1 / num_variables}]"

        expressions = generate_n_expressions(
            full_grammar, number_of_expressions, max_expression_length=40, verbose=False
        )

        start_time = time.time()
        losses = np.array([evaluator.evaluate_expr(expr) for expr in expressions])
        time_optimal = time.time() - start_time
        losses = np.nan_to_num(losses, nan=np.inf, posinf=np.inf, neginf=np.inf)
        invalid_indices = np.where(losses > 1e20)[0]

        # Optimal (always saved — needed for plotting)
        os.makedirs("../results/smoothness/Optimal", exist_ok=True)
        np.savez(
            f"../results/smoothness/Optimal/{args.dataset}.npz",
            rmse=np.sort(losses)[:top_n], invalid_count=0, time=time_optimal,
        )

        # ---- Edit distance ------------------------------------------------
        if "edit" in baselines:
            edit = []
            start_time = time.time()
            for expr in expressions:
                edit.append(editdistance.eval(dataset.ground_truth, expr))
            time_edit = time.time() - start_time
            edit_ind = np.argsort(edit)
            os.makedirs("../results/smoothness/Edit", exist_ok=True)
            np.savez(
                f"../results/smoothness/Edit/{args.dataset}.npz",
                rmse=losses[edit_ind[~np.isin(edit_ind, invalid_indices)][:top_n]],
                invalid_count=np.isin(edit_ind[:top_n], invalid_indices).sum(),
                time=time_edit,
            )

        # ---- Tree edit distance -------------------------------------------
        if "tree" in baselines:
            tree = []
            zss_gt = expr_to_zss(ttt(dataset.ground_truth, dataset.symbol_library))
            start_time = time.time()
            for expr in expressions:
                tree.append(zss.simple_distance(zss_gt, expr_to_zss(ttt(expr, dataset.symbol_library))))
            time_tree = time.time() - start_time
            tree_ind = np.argsort(tree)
            os.makedirs("../results/smoothness/TreeEdit", exist_ok=True)
            np.savez(
                f"../results/smoothness/TreeEdit/{args.dataset}.npz",
                rmse=losses[tree_ind[~np.isin(tree_ind, invalid_indices)][:top_n]],
                invalid_count=np.isin(tree_ind[:top_n], invalid_indices).sum(),
                time=time_tree,
            )

        # ---- Jaro distance ------------------------------------------------
        if "jaro" in baselines:
            jaro = []
            start_time = time.time()
            for expr in expressions:
                jaro.append(Jaro.distance(dataset.ground_truth, expr))
            time_jaro = time.time() - start_time
            jaro_ind = np.argsort(jaro)
            os.makedirs("../results/smoothness/Jaro", exist_ok=True)
            np.savez(
                f"../results/smoothness/Jaro/{args.dataset}.npz",
                rmse=losses[jaro_ind[~np.isin(jaro_ind, invalid_indices)][:top_n]],
                invalid_count=np.isin(jaro_ind[:top_n], invalid_indices).sum(),
                time=time_jaro,
            )

        # ---- BED ----------------------------------------------------------
        if "bed" in baselines:
            results_bed = np.zeros((number_of_runs, top_n))
            invalid_bed = np.zeros(number_of_runs)
            times_bed = []
            for i in range(number_of_runs):
                x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
                start_time = time.time()
                distances = BED(
                    [dataset.ground_truth], expressions2=expressions,
                    x=dataset.X[x_indices], const_params=(-5, 5),
                ).calculate_distances()
                times_bed.append(time.time() - start_time)
                bed_ind = np.argsort(distances[0])
                invalid_bed[i] = np.isin(bed_ind[:top_n], invalid_indices).sum()
                results_bed[i] = losses[bed_ind[~np.isin(bed_ind, invalid_indices)][:top_n]]
            os.makedirs("../results/smoothness/BED", exist_ok=True)
            np.savez(
                f"../results/smoothness/BED/{args.dataset}.npz",
                rmse=results_bed, invalid_count=invalid_bed, time=np.array(times_bed),
            )

        # ---- Maximal ------------------------------------------------------
        if "maximal" in baselines:
            results_maximal = np.zeros((number_of_runs, top_n))
            invalid_maximal = np.zeros(number_of_runs)
            times_maximal = []
            for i in range(number_of_runs):
                x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
                start_time = time.time()
                distances = maximal_hausdorff(
                    [dataset.ground_truth], expressions2=expressions,
                    x=dataset.X[x_indices], const_params=(-5, 5),
                ).calculate_distances()
                times_maximal.append(time.time() - start_time)
                maximal_ind = np.argsort(distances[0])
                invalid_maximal[i] = np.isin(maximal_ind[:top_n], invalid_indices).sum()
                results_maximal[i] = losses[maximal_ind[~np.isin(maximal_ind, invalid_indices)][:top_n]]
            os.makedirs("../results/smoothness/Maximal", exist_ok=True)
            np.savez(
                f"../results/smoothness/Maximal/{args.dataset}.npz",
                rmse=results_maximal, invalid_count=invalid_maximal, time=np.array(times_maximal),
            )

        # ---- Hausdorff ----------------------------------------------------
        if "hausdorff" in baselines:
            results_hausdorff = np.zeros((number_of_runs, top_n))
            invalid_hausdorff = np.zeros(number_of_runs)
            times_hausdorff = []
            for i in range(number_of_runs):
                x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
                start_time = time.time()
                distances = maximal_hausdorff(
                    [dataset.ground_truth], expressions2=expressions,
                    x=dataset.X[x_indices], const_params=(-5, 5), hausdorff=True,
                ).calculate_distances()
                times_hausdorff.append(time.time() - start_time)
                hausdorff_ind = np.argsort(distances[0])
                invalid_hausdorff[i] = np.isin(hausdorff_ind[:top_n], invalid_indices).sum()
                results_hausdorff[i] = losses[hausdorff_ind[~np.isin(hausdorff_ind, invalid_indices)][:top_n]]
            os.makedirs("../results/smoothness/Hausdorff", exist_ok=True)
            np.savez(
                f"../results/smoothness/Hausdorff/{args.dataset}.npz",
                rmse=results_hausdorff, invalid_count=invalid_hausdorff, time=np.array(times_hausdorff),
            )

        # ---- SNIP ---------------------------------------------------------
        if "snip" in baselines:
            if not args.snip_checkpoint:
                raise ValueError("--snip_checkpoint is required when 'snip' is in --baselines")
            snip_sl = SymbolLibrary.from_symbol_list(_SMOOTHNESS_SYMBOLS, num_variables=num_variables)
            snip_encoder = load_snip_encoder(
                args.snip_checkpoint, snip_sl,
                base_checkpoint=args.snip_base_checkpoint,
            )
            all_snip = [dataset.ground_truth] + expressions
            start_time = time.time()
            snip_emb = snip_encoder.encode(all_snip, dataset.symbol_library, allow_ood=True)
            time_snip = time.time() - start_time
            snip_dists = torch.cdist(snip_emb[0:1], snip_emb[1:], p=2)[0].numpy()
            snip_ind = np.argsort(snip_dists)
            os.makedirs("../results/smoothness/SNIP", exist_ok=True)
            np.savez(
                f"../results/smoothness/SNIP/{args.dataset}.npz",
                rmse=losses[snip_ind[~np.isin(snip_ind, invalid_indices)][:top_n]],
                invalid_count=np.isin(snip_ind[:top_n], invalid_indices).sum(),
                time=time_snip,
            )

        # ---- HVAE ---------------------------------------------------------
        if "hvae" in baselines:
            hvae_sl = _build_hvae_sl(_SMOOTHNESS_SYMBOLS, 9)
            hvae_model = _get_hvae_model(hvae_sl, args.hvae_latent_size, args.hvae_model)
            s2i = hvae_sl.symbols2index()
            all_hvae = [dataset.ground_truth] + expressions
            try:
                neuro_exprs = [ttt(expr, hvae_sl) for expr in all_hvae]
                start_time = time.time()
                hvae_emb = hvae_model.encode(create_batch(neuro_exprs, s2i))[0]
                time_hvae = time.time() - start_time
                hvae_dists = torch.cdist(hvae_emb[0:1], hvae_emb[1:], p=2)[0].detach().numpy()
                hvae_ind = np.argsort(hvae_dists)
                os.makedirs("../results/smoothness/HVAE", exist_ok=True)
                np.savez(
                    f"../results/smoothness/HVAE/{args.dataset}.npz",
                    rmse=losses[hvae_ind[~np.isin(hvae_ind, invalid_indices)][:top_n]],
                    invalid_count=np.isin(hvae_ind[:top_n], invalid_indices).sum(),
                    time=time_hvae,
                )
            except Exception as exc:
                print(f"HVAE encoding failed for {args.dataset}: {exc}")

    # -----------------------------------------------------------------------
    # Plot mode
    # -----------------------------------------------------------------------
    else:
        base_path = Path("../results/smoothness")

        if args.dataset == "all":
            all_datasets = SR_benchmark.feynman("../data/feynman").list_datasets(verbose=False)

            # Collect per-baseline aggregated curves across all datasets
            agg_curves: dict[str, list] = {}  # display_name -> list of 1-D agg arrays
            totals: dict[str, dict] = {}       # display_name -> {invalid, time, count}

            for ds_name in all_datasets:
                for key, (display, result_dir, _) in _BASELINE_META.items():
                    obj = _load_result(base_path, result_dir, ds_name)
                    if obj is None:
                        continue
                    rmse = obj["rmse"]
                    # BED has shape (runs, top_n); others are 1-D
                    if rmse.ndim == 2:
                        agg = np.mean(
                            [aggregate_results(rmse[i], args.aggregator) for i in range(len(rmse))],
                            axis=0,
                        )
                        t = np.mean(obj["time"])
                        inv = np.mean(obj["invalid_count"])
                    else:
                        agg = aggregate_results(rmse, args.aggregator)
                        t = float(obj["time"])
                        inv = float(obj["invalid_count"])

                    if display not in agg_curves:
                        agg_curves[display] = []
                        totals[display] = {"invalid": 0.0, "time": 0.0, "count": 0}
                    agg_curves[display].append(agg)
                    totals[display]["invalid"] += inv
                    totals[display]["time"] += t
                    totals[display]["count"] += 1

            if not agg_curves:
                print("No result files found. Run with -compute first.")
            else:
                # Average across datasets
                mean_curves = {
                    name: np.mean(curves, axis=0)
                    for name, curves in agg_curves.items()
                }
                palette = {name: _BASELINE_META[k][2]
                           for k, (name, _, _) in _BASELINE_META.items()
                           if name in mean_curves}

                for name, tot in totals.items():
                    n = tot["count"]
                    print(f"{name}: invalid {tot['invalid']/n:.2f}, time {tot['time']/n:.4f}")

                plot_ranking_baselines(
                    mean_curves, palette,
                    y_axis_label="Average Rank",
                    plot_title="Average Rank of Baselines for Closest Expressions",
                    aggregator=args.aggregator,
                )

        else:
            if args.dataset == "none":
                all_datasets = SR_benchmark.feynman("../data/feynman").list_datasets(verbose=False)
                ds = np.random.choice(all_datasets)
            else:
                ds = args.dataset

            dataset = SR_benchmark.feynman("../data/feynman").create_dataset(ds)
            y_text = "Mean Aggregated RMSE" if args.aggregator == "mean" else "Median Aggregated RMSE"

            # Optimal (always loaded for the plot baseline)
            opt_obj = _load_result(base_path, "Optimal", ds)
            if opt_obj is not None:
                optimal_agg = aggregate_results(opt_obj["rmse"], args.aggregator)
                print(f"Optimal: time {float(opt_obj['time']):.4f}")
            else:
                optimal_agg = None
                print("Optimal result not found — re-run with -compute")

            # Collect deterministic and stochastic curves
            det_curves: dict[str, np.ndarray] = {}
            stoch_curves: dict[str, np.ndarray] = {}
            if optimal_agg is not None:
                det_curves["Optimal"] = optimal_agg
            palette = {"Optimal": "#000000"}

            for key, (display, result_dir, color) in _BASELINE_META.items():
                obj = _load_result(base_path, result_dir, ds)
                if obj is None:
                    continue
                palette[display] = color
                rmse = obj["rmse"]
                if rmse.ndim == 2:
                    stoch_curves[display] = rmse
                else:
                    det_curves[display] = aggregate_results(rmse, args.aggregator)
                t = float(np.mean(obj["time"]))
                inv = float(np.mean(obj["invalid_count"]))
                print(f"{display}: invalid {inv:.2f}, time {t:.4f}")

            os.makedirs("../results/smoothness/figures", exist_ok=True)
            plot_metric(
                det_curves, stoch_curves,
                ds, dataset, palette, y_text,
                save_name=f"../results/smoothness/figures/plot_{ds}_{args.aggregator}.png",
                aggregator=args.aggregator,
                top_n=top_n,
            )
