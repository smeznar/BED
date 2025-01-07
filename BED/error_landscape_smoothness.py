import itertools
from argparse import ArgumentParser
import time

import numpy as np
import editdistance
import zss
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import read_expressions_json, read_expressions_zss
from bed import BED
from evaluation import RustEval


sns.set_theme(font_scale=0.8, palette="Set2", rc={'figure.figsize': (6, 4), 'text.usetex': True,
                                                'axes.titlesize': 80, "axes.labelsize": 12})


def ranking_call(diffs, agf1, agf2, name):
    aggr1 = np.zeros(diffs.shape)
    for i in range(diffs.shape[1]):
        aggr1[:, i] = agf1(diffs[:, :i + 1], axis=1)
    aggr2 = agf2(aggr1, axis=0).tolist()
    names = [name for i in range(diffs.shape[1])]
    x = list(range(1, diffs.shape[1]+1))
    return aggr2, names, x


def tree_ed_matrix(ys, ys2=None, same=False):
    if ys2 is None:
        dm = np.zeros((len(ys), len(ys)))
        for i, j in itertools.combinations(range(len(ys)), 2):
            dm[i, j] = dm[j, i] = zss.simple_distance(ys[i], ys[j])
    else:
        dm = np.zeros((len(ys), len(ys2)))
        for i in range(len(ys)):
            if same:
                start = i+1
            else:
                start = 0
            for j in range(start, len(ys2)):
                dm[i, j] = zss.simple_distance(ys[i], ys2[j])
                if same:
                    dm[j, i] = dm[i, j]
    return dm


if __name__ == '__main__':
    parser = ArgumentParser(prog='Smoothness of the error landscape',
                            description='A script for testing the smoothness of the error landscape')
    parser.add_argument("-calculate_loss", action="store_true")
    parser.add_argument("-dataset_num", default=0, type=int)
    parser.add_argument("-calculate_distance", action="store_true")
    parser.add_argument("-baseline", default="BED")
    parser.add_argument("-expr_path", default="../data/expression_sets/metric_exprs_20k_2v.json", type=str)
    parser.add_argument("-subset1_low", default=0, type=int)
    parser.add_argument("-subset2_low", default=0, type=int)
    parser.add_argument("-subset_len", type=int)
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-var_domain_low", default=1.0, type=float)
    parser.add_argument("-var_domain_high", default=5.0, type=float)
    parser.add_argument("-precompute_ranks", action="store_true")
    parser.add_argument("-top_n", default=200, type=int)
    parser.add_argument("-plot_results", action="store_true")
    parser.add_argument("-aggr1", default="max")
    parser.add_argument("-aggr2", default="mean")

    args = parser.parse_args()
    feynman_ids = ["I.6.2", "I.12.1", "I.14.4", "I.25.13", "I.26.2", "I.34.27",
                   "I.39.1", "II.3.24", "II.11.28", "II.27.18", "II.38.14"]
    fexpr = ["$(\sqrt{2\pi}\cdot y)^{-1} e^{-\\frac{(x/y)^2}{2}}$", "$xy$", "$0.5\, x y^2$", "$x/y$",
             "$\\arcsin (x\sin{y})$", "$(2\pi)^{-1}xy$", "$1.5\, xy$", "$\\frac{x}{4\pi y^2}$",
             "$\\frac{1+xy}{1-(0.\overline{3}\, xy)}$", "$xy^2$", "$\\frac{x}{2\cdot(1 + y)}$"]

    # Calculate the loss
    if args.calculate_loss:
        expressions = read_expressions_json(args.expr_path)
        losses = np.zeros(len(expressions))
        data = np.load(f"../data/feynman_data/{feynman_ids[args.dataset_num]}.npy")
        ys = np.zeros((len(expressions), data.shape[0]))
        re_train = RustEval(data, default_value=1e10, seed=args.seed)
        for i, expr in enumerate(expressions):
            print(f"[{feynman_ids[args.dataset_num]}] {i} expressions done")
            rmse, x = re_train.fit_and_evaluate(expr)
            ys[i, :] = np.array(re_train.evaluate(expr, constants=[x])[0])
            losses[i] = rmse

        np.save(f"../results/smoothness/losses_20k_{args.dataset_num}.npy", losses)
        np.save(f"../results/smoothness/evaluated_values_20k_{args.dataset_num}.npy", ys)

    # Calculate the distance between expressions
    elif args.calculate_distance:
        # BED
        # Use arguments to compute matrices in parallel
        if args.baseline == "BED":
            expressions = read_expressions_json(args.expr_path)
            st = time.time()
            if args.subset_len is not None:
                exprs1 = expressions[args.subset1_low: args.subset1_low + args.subset_len]
                exprs2 = expressions[args.subset2_low: args.subset2_low + args.subset_len]

                if args.subset1_low == args.subset2_low:
                    bed = BED(exprs1, [(args.var_domain_low, args.var_domain_high) for i in range(2)],
                              (0.2, 5), points_sampled=64, consts_sampled=16, seed=args.seed)
                    dm = bed.calculate_distances()
                else:
                    bed = BED(exprs1, [(args.var_domain_low, args.var_domain_high) for i in range(2)],
                              (0.2, 5), expressions2=exprs2, points_sampled=64, consts_sampled=16,
                              seed=args.seed)
                    dm = bed.calculate_distances()

                # np.save(f"../results/smoothness/bed_submatrices/time_bed_{args.var_domain_low}-{args.var_domain_high}"
                #         f"_{args.subset1_low}_{args.subset2_low}_{args.seed}.npy", np.array([time.time() - st]))
                np.save(f"../results/BEDHIE/dm_bed_{args.var_domain_low}-{args.var_domain_high}"
                        f"_{args.subset1_low}_{args.subset2_low}_{args.seed}.npy", dm)
            else:
                bed = BED(expressions, [(args.var_domain_low, args.var_domain_high) for i in range(2)],
                          (0.2, 5), points_sampled=64, consts_sampled=16, seed=args.seed)
                dm = bed.calculate_distances()
                np.save(f"../results/smoothness/time_bed_.npy", np.array([time.time() - st]))
                np.save(f"../results/smoothness/dm_bed.npy", dm)

        # Edit distance
        elif args.baseline == "edit":
            expressions = read_expressions_json(args.expr_path)
            st = time.time()
            dm = np.zeros((len(expressions), len(expressions)))
            for i, j in itertools.combinations(range(len(expressions)), 2):
                dm[i, j] = dm[j, i] = editdistance.eval(expressions[i], expressions[j])
            np.save("../results/smoothness/time_edit.npy", np.array(time.time()-st))
            np.save("../results/smoothness/dm_edit.npy", dm)

        # Tree edit distance
        # Use arguments "subset1_low", "subset2_low", and "subset_len" to compute matrices in parallel
        elif args.baseline == "tree-edit":
            expressions = read_expressions_zss(args.expr_path)
            st = time.time()
            if args.subset_len is not None:
                exprs1 = expressions[args.subset1_low: args.subset1_low + args.subset_len]
                exprs2 = expressions[args.subset2_low: args.subset2_low + args.subset_len]
                same_subsets = args.subset1_low == args.subset2_low
                dm = tree_ed_matrix(exprs1, exprs2, same_subsets)
                np.save(f"../results/smoothness/tree_submatrices/time_tree-edit_{args.subset1_low}_{args.subset2_low}.npy", np.array([time.time() - st]))
                np.save(f"../results/smoothness/tree_submatrices/dm_tree-edit_{args.subset1_low}_{args.subset2_low}.npy", dm)
            else:
                dm = tree_ed_matrix(expressions, expressions, True)
                np.save(f"../results/smoothness/time_tree_.npy", np.array([time.time() - st]))
                np.save(f"../results/smoothness/dm_tree.npy", dm)

        # Optimal distance
        # Calculate losses using arguments "calculate_loss" and "dataset_num" before calculating this distance
        else:
            ys = np.load(f"../results/smoothness/evaluated_values_20k_{args.dataset_num}.npy")
            st = time.time()
            dm = np.zeros((ys.shape[0], ys.shape[0]))
            for i, j in itertools.combinations(range(ys.shape[0]), 2):
                dm[i, j] = dm[j, i] = np.sqrt(((ys[i, :] - ys[j, :]) ** 2).mean())
            np.save(f"../results/smoothness/time_optimal_{args.dataset_num}.npy", np.array(time.time() - st))
            np.save(f"../results/smoothness/dm_optimal_{args.dataset_num}.npy", dm)

    # Precompute top ranks to accelerate generation of figures (modify or other/specific data sets)
    elif args.precompute_ranks:
        for j in range(0, 11):
            print(f"Precomputing data set {j}")
            losses = np.load(f"../results/smoothness/losses_20k_{j}.npy")
            losses[losses > 9e9] = np.nan
            finite_losses = np.isfinite(losses)
            floss = losses[finite_losses]

            # BED
            for i in range(5):
                time_needed = 0
                dm = np.zeros((20000, 20000))
                for k in range(20):
                    for l in range(k, 20):
                        small_dm = np.load(f"../results/smoothness/bed_submatrices/dm_bed_{args.var_domain_low}-{args.var_domain_high}"
                                           f"_{k * 1000}_{l * 1000}_{i}.npy")
                        dm[(k * 1000):((k + 1) * 1000), (l * 1000):((l + 1) * 1000)] = small_dm
                        dm[(l * 1000):((l + 1) * 1000), (k * 1000):((k + 1) * 1000)] = small_dm.T
                        time_needed += np.load(f"../results/smoothness/bed_submatrices/time_bed_{args.var_domain_low}"
                                               f"-{args.var_domain_high}_{k*1000}_{l*1000}_{i}.npy")[0]
                print(f"BED: time needed: {time_needed}")
                dm = dm[finite_losses]
                dm = dm[:, finite_losses]
                sdm = dm.argsort(axis=1)[:, :args.top_n]
                slm = floss[sdm]
                diffs = abs(slm - floss[:, None])
                np.save(f"../results/smoothness/precomputed_bed_{args.var_domain_low}-{args.var_domain_high}_{i}_{j}.npy", diffs)

            # Tree
            time_needed = 0
            dm = np.zeros((20000, 20000))
            for k in range(20):
                for l in range(k, 20):
                    small_dm = np.load(
                        f"../results/smoothness/tree_submatrices/dm_tree-edit_{k * 1000}_{l * 1000}.npy")
                    dm[(k * 1000):((k + 1) * 1000), (l * 1000):((l + 1) * 1000)] = small_dm
                    dm[(l * 1000):((l + 1) * 1000), (k * 1000):((k + 1) * 1000)] = small_dm.T
                    time_needed += np.load(f"../results/smoothness/tree_submatrices/time_tree-edit_{k * 1000}_{l * 1000}.npy")[0]
            print(f"Tree-edit distance: time needed: {time_needed}")
            dm = dm[finite_losses]
            dm = dm[:, finite_losses]
            sdm = dm.argsort(axis=1)[:, :args.top_n]
            slm = floss[sdm]
            diffs = abs(slm - floss[:, None])
            np.save(f"../results/smoothness/precomputed_tree_{j}.npy", diffs)

            # Edit
            dm = np.load("../results/smoothness/dm_edit.npy")[finite_losses]
            dm = dm[:, finite_losses]
            sdm = dm.argsort(axis=1)[:, :args.top_n]
            slm = floss[sdm]
            diffs = abs(slm - floss[:, None])
            np.save(f"../results/smoothness/precomputed_edit_{j}.npy", diffs)
            print(f"Edit distance: time needed: {np.load('../results/smoothness/time_edit.npy')[None][0]}")

            # Optimal
            dm = np.load(f"../results/smoothness/dm_optimal_{j}.npy")[finite_losses]
            dm = dm[:, finite_losses]
            sdm = dm.argsort(axis=1)[:, :args.top_n]
            slm = floss[sdm]
            diffs = abs(slm - floss[:, None])
            np.save(f"../results/smoothness/precomputed_optimal_{j}.npy", diffs)
            time_optimal = np.load(f'../results/smoothness/time_optimal_{j}.npy')[None][0]
            print(f"Optimal distance: time needed: {time_optimal}")

    elif args.plot_results:
        if args.aggr1 == "mean":
            agf1 = np.nanmean
        elif args.aggr1 == "median":
            agf1 = np.nanmedian
        else:
            agf1 = np.nanmax

        if args.aggr2 == "mean":
            agf2 = np.nanmean
        else:
            agf2 = np.nanmedian

        names = []
        aggr2 = []
        xs = []

        diffs = np.load(f"../results/smoothness/precomputed_edit_{args.dataset_num}.npy")
        a, n, x = ranking_call(diffs, agf1, agf2, "Edit distance")
        names += n
        aggr2 += a
        xs += x

        diffs = np.load(f"../results/smoothness/precomputed_tree_{args.dataset_num}.npy")
        a, n, x = ranking_call(diffs, agf1, agf2, "Tree-edit distance")
        names += n
        aggr2 += a
        xs += x

        diffs = np.load(f'../results/smoothness/precomputed_optimal_{args.dataset_num}.npy')
        a, n, x = ranking_call(diffs, agf1, agf2, "Optimal distance")
        names += n
        aggr2 += a
        xs += x

        for i in range(5):
            diffs = np.load(f"../results/smoothness/precomputed_bed_{args.var_domain_low}-{args.var_domain_high}_{i}_{args.dataset_num}.npy")
            a, n, x = ranking_call(diffs, agf1, agf2, "BED (Our)")
            names += n
            aggr2 += a
            xs += x

        data = pd.DataFrame(
            {"Number of neighbours": xs, f"{args.aggr2.capitalize()} {args.aggr1} difference (RMSE)": aggr2, "Metric": names})
        sns_plot = sns.lineplot(x="Number of neighbours", y=f"{args.aggr2.capitalize()} {args.aggr1} difference (RMSE)", hue="Metric",
                                data=data)
        sns_plot.axes.set_title(f"Expression ({feynman_ids[args.dataset_num]}): {fexpr[args.dataset_num]}", fontsize=15)

        plt.savefig(f"../results/figures/smoothness_{args.aggr2}_{args.aggr1}_{args.var_domain_low}-{args.var_domain_high}_{args.dataset_num}.png", dpi=400)

    else:
        print("Add one of the following arguments: '-calculate_losses', '-calculate_distance',"
              " '-precompute_ranks', '-plot_results'")