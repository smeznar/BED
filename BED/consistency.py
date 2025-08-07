import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SRToolkit.dataset import SRBenchmark
from SRToolkit.utils import generate_n_expressions

sns.set_theme(font_scale=2, palette="Set2", rc={'figure.figsize': (10, 8), 'text.usetex': True,
                                                'axes.titlesize': 35, "axes.labelsize": 26})

from argparse import ArgumentParser
import itertools

from bed import BED

def read_expressions_txt(path):
    expressions = []
    with open(path, "r") as f:
        for line in f:
            expressions.append(line.strip().split(" "))
    return expressions

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


def generate_expressions(num_expressions, num_vars):
    grammar_string = """
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
    for v in range(num_vars):
        grammar_string += f"\nV -> 'X_{v}' [{1 / num_vars}]"

    benchmark = SRBenchmark.feynman("../data/feynman")
    id = benchmark.list_datasets(num_variables=num_vars)[0]
    eval = benchmark.create_dataset(id).create_evaluator()
    expressions = generate_n_expressions(grammar_string, num_expressions*10, unique=True, max_expression_length=50, verbose=True)
    non_equivalent = []
    for expr in expressions:
        loss = eval.evaluate_expr(expr)
        if not np.isfinite(loss) or loss > 1e9:
            continue
        is_equivalent = False
        for eq, eq_loss in non_equivalent:
            if np.abs(loss - eq_loss)<1e-5:
                is_equivalent = True
                break

        if not is_equivalent:
            print(loss)
            non_equivalent.append((expr, loss))
            if len(non_equivalent) >= num_expressions:
                break

    with open(f"../results/consistency/expressions/expressions_{num_vars}.txt", "w") as f:
        for expr, expr_loss in non_equivalent:
            f.write(' '.join(expr) + "\n")


if __name__ == '__main__':
    # generate_expressions(200, 1)
    # for num_vars in [1, 4]:
    #     for num_points in [4, 8, 16, 32, 64, 128]:
    #         for num_consts in [4, 8, 16, 32, 64, 128]:
    #             for seed in range(100):
    #                 print(
    #                     f"python consistency.py -num_vars {num_vars} -expr_path ../results/consistency/expressions/expressions_{num_vars}.txt -seed {seed} -num_points {num_points} -num_const {num_consts}")
    # print("finished")
    parser = ArgumentParser(prog='Symbolic regression', description='Run a symbolic regression benchmark')
    parser.add_argument("-num_vars", type=int)
    parser.add_argument("-expr_path", type=str)
    parser.add_argument("-seed", type=int)
    parser.add_argument("-num_points", type=int)
    parser.add_argument("-num_const", type=int)
    parser.add_argument("-figure_path", type=str)
    parser.add_argument("-random", action="store_true")
    args = parser.parse_args()

    if (args.seed is not None and args.num_points is not None and args.num_const is not None
            and args.expr_path is not None and args.num_vars is not None):
        expressions = read_expressions_txt(args.expr_path)
        bed = BED(expressions, [(1, 5) for i in range(args.num_vars)], (1, 5), normalize=True,
                  points_sampled=args.num_points, consts_sampled=args.num_const, seed=args.seed)
        dm = bed.calculate_distances()
        np.save(f"../results/consistency/distance_matrices_smape_{args.num_vars}/dm_{args.num_points}_{args.num_const}_{args.seed}_{args.num_vars}.npy", dm)

    elif args.figure_path is not None and args.num_vars is not None:
        x = []
        y = []
        value = []
        for num_points in [4, 8, 16, 32, 64, 128]:
            for num_const in [4, 8, 16, 32, 64, 128]:
                files = glob.glob(f"../results/consistency/distance_matrices_smape_{args.num_vars}/dm_{num_points}_{num_const}_*_{args.num_vars}.npy")
                matrices = []
                for file in files:
                    if args.random:
                        matrices.append(np.random.random((200, 200)))
                    else:
                        matrices.append(np.load(file))
                value.append(spearman(matrices))
                x.append(num_points)
                y.append(num_const)

        sd_mat = (pd.DataFrame(data={"\#VS": x, "\#CS": y, "SD": value})
                  .pivot(index="\#VS", columns="\#CS", values="SD"))
        sd_mat.sort_index(level=0, ascending=False, inplace=True)

        sns.heatmap(sd_mat, annot=True, cmap="coolwarm", linewidth=.5, vmax=1.0, vmin=0.0, fmt=".3g")
        plt.title(f"Number of variables: {args.num_vars}")
        plt.savefig(args.figure_path)
