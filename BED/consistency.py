import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(font_scale=2, palette="Set2", rc={'figure.figsize': (10, 8), 'text.usetex': True,
                                                'axes.titlesize': 35, "axes.labelsize": 26})

from argparse import ArgumentParser
import itertools

from utils import read_expressions_json
from bed import BED
from ebed import EBED

def sd(matrices):
    distances = np.stack(matrices)
    stds = np.std(distances, axis=0)
    stds = stds[np.triu_indices(len(matrices))]
    return np.median(stds)


def spearman(matrices):
    dm = np.zeros((len(matrices), len(matrices)))
    sorted_matrices = []
    for i, mat in enumerate(matrices):
        sorted_matrices.append(mat.argsort(axis=-1).argsort() + 1)
        dm[i, i] = 1.0
    for i, j in itertools.combinations(range(len(matrices)), 2):
        di = np.power(sorted_matrices[i] - sorted_matrices[j], 2)
        n = di.shape[0]
        dm[i, j] = dm[j, i] = np.mean(1 - (6*np.sum(di, axis=1)/(n**3 - n)))
    sm = dm[np.triu_indices(len(matrices))]
    return np.mean(sm)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Symbolic regression', description='Run a symbolic regression benchmark')
    parser.add_argument("-seed", type=int)
    parser.add_argument("-num_points", type=int)
    parser.add_argument("-num_const", type=int)
    parser.add_argument("-num_vars", type=int)
    parser.add_argument("-expr_path", type=str)
    parser.add_argument("-figure_path", type=str)
    parser.add_argument("-stability_type", default="spearman", type=str)
    parser.add_argument("-random", action="store_true")
    parser.add_argument("-baseline", default="BED", type=str)
    args = parser.parse_args()

    if (args.seed is not None and args.num_points is not None and args.num_const is not None
            and args.expr_path is not None and args.num_vars is not None):
        expressions = read_expressions_json(args.expr_path)
        if args.baseline == "BED":
            bed = BED(expressions, [(1, 5) for i in range(args.num_vars)], (0.2, 5),
                    points_sampled=args.num_points, consts_sampled=args.num_const, seed=args.seed)
        else:
            bed = EBED(expressions, [(1, 5) for i in range(args.num_vars)], (0.2, 5),
                    points_sampled=args.num_points, consts_sampled=args.num_const, seed=args.seed, randomized=True)
        dm = bed.calculate_distances()
        np.save(f"../results/consistency_{args.baseline}/dm_ablation_{args.num_points}_{args.num_const}_{args.seed}_{args.num_vars}.npy", dm)

    elif args.figure_path is not None and args.num_vars is not None and args.stability_type is not None:
        if args.stability_type == "std":
            stability_metric = sd
        else:
            stability_metric = spearman

        x = []
        y = []
        value = []
        for num_points in [16, 32, 64, 128, 256, 512]:
            for num_const in [2, 4, 8, 16, 32, 64]:
                files = glob.glob(f"../results/consistency_{args.baseline}/dm_ablation_{num_points}_{num_const}_*_{args.num_vars}.npy")
                matrices = []
                for file in files:
                    if args.random:
                        matrices.append(np.random.random((100, 100)))
                    else:
                        matrices.append(np.load(file))
                value.append(stability_metric(matrices))
                x.append(num_points)
                y.append(num_const)

        sd_mat = (pd.DataFrame(data={"\#VS": x, "\#CS": y, "SD": value})
                  .pivot(index="\#VS", columns="\#CS", values="SD"))
        sd_mat.sort_index(level=0, ascending=False, inplace=True)

        sns.heatmap(sd_mat, annot=True, cmap="coolwarm", linewidth=.5, vmax=1.0, vmin=0.0, fmt=".3g")
        plt.title(f"Number of variables: {args.num_vars}")
        plt.savefig(args.figure_path)
