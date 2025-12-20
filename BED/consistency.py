import glob
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SRToolkit.dataset import SRBenchmark
from SRToolkit.utils import generate_n_expressions
from scipy.stats.qmc import LatinHypercube


plt.rcParams.update({
    "text.usetex": True,
    "figure.figsize": (6, 5),
    "font.size": 12,
    "figure.dpi": 400,
    "axes.labelsize": 14
})
plt.rcParams['text.latex.preamble']="\\usepackage{amsmath}"

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
    parser = ArgumentParser(prog='Symbolic regression', description='Run a symbolic regression benchmark')
    parser.add_argument("-num_vars", type=int)
    parser.add_argument("-expr_path", type=str)
    parser.add_argument("-seed", type=int)
    parser.add_argument("-num_points", type=int)
    parser.add_argument("-num_const", type=int)
    parser.add_argument("-figure_path", type=str)
    parser.add_argument("-random", action="store_true")
    parser.add_argument("-stringency", default="run") # all: Same X across all runs; run: same X for a run; pair: new X for each pair
    args = parser.parse_args()

    if (args.seed is not None and args.num_points is not None and args.num_const is not None
            and args.expr_path is not None and args.num_vars is not None):
        expressions = read_expressions_txt(args.expr_path)
        if args.stringency == "all":
            np.random.seed(0)
            x_bounds = [(1, 5) for i in range(args.num_vars)]
            interval_length = np.array([ub - lb for (lb, ub) in x_bounds])
            lower_bound = np.array([lb for (lb, ub) in x_bounds])
            lho = LatinHypercube(len(x_bounds), optimization="random-cd", seed=0)
            X = lho.random(args.num_points) * interval_length + lower_bound
            start_time = time.time()
            bed = BED(expressions, x=X, const_bounds=(1, 5), points_sampled=args.num_points,
                      consts_sampled=args.num_const, seed=args.seed)
        elif args.stringency == "run":
            start_time = time.time()
            bed = BED(expressions, [(1, 5) for i in range(args.num_vars)], (1, 5),
                      points_sampled=args.num_points, consts_sampled=args.num_const, seed=args.seed)
        else:
            start_time = time.time()
            bed = BED(expressions, [(1, 5) for i in range(args.num_vars)], (1, 5),
                      points_sampled=args.num_points, consts_sampled=args.num_const, seed=args.seed,
                      randomized=True)
        dm = bed.calculate_distances()
        time_taken = time.time() - start_time
        np.savez(f"../results/consistency/distance_matrices_{args.num_vars}_{args.stringency}/dm_{args.num_points}_{args.num_const}_{args.seed}_{args.num_vars}.npz", dm=dm, time=time_taken)

    elif args.figure_path is not None and args.num_vars is not None:
        x = []
        y = []
        value = []
        t = []
        for num_points in [4, 8, 16, 32, 64, 128]:
            for num_const in [4, 8, 16, 32, 64, 128]:
                files = glob.glob(f"../results/consistency/distance_matrices_{args.num_vars}_{args.stringency}/dm_{num_points}_{num_const}_*_{args.num_vars}.npz")
                matrices = []
                time_taken = []
                if len(files) == 0:
                    time_taken.append(0)
                for file in files:
                    if args.random:
                        matrices.append(np.random.random((200, 200)))
                    else:
                        data = np.load(file)
                        matrices.append(data["dm"])
                        time_taken.append(data["time"])
                value.append(spearman(matrices))
                x.append(num_points)
                y.append(num_const)
                t.append(np.mean(time_taken))
                print(num_points, num_const, np.mean(time_taken), np.std(time_taken), np.std(time_taken)/np.mean(time_taken))
        sd_mat = (pd.DataFrame(data={"$|\\boldsymbol{X}|$": x, "$|\\mathbf{C}|$": y, "SD": value})
                  .pivot(index="$|\\boldsymbol{X}|$", columns="$|\\mathbf{C}|$", values="SD"))
        sd_mat.sort_index(level=0, ascending=False, inplace=True)

        sns.heatmap(sd_mat, annot=True, cmap="coolwarm", linewidth=.5, vmax=1.0, vmin=0.0, fmt=".3g")
        if args.num_vars == 1:
            text = f"\\textbf{{Consistency: 1 Variable}}"
        elif args.stringency == "all":
            text = f"\\textbf{{Consistency: $\\boldsymbol{{X}}$ is sampled once}}\n"
        elif args.stringency == "pair":
            text = f"\\textbf{{Consistency: $\\boldsymbol{{X}}$ is sampled for each calculation}}\n"
        else:
            text = f"\\textbf{{Consistency: {args.num_vars} Variables}}\n"

        plt.title(text)
        plt.tight_layout()
        plt.savefig(args.figure_path)

        plt.clf()
        sd_mat = (pd.DataFrame(data={"$|\\boldsymbol{X}|$": x, "$|\\mathbf{C}|$": y, "time": t})
                  .pivot(index="$|\\boldsymbol{X}|$", columns="$|\\mathbf{C}|$", values="time", ))
        sd_mat.sort_index(level=0, ascending=False, inplace=True)

        sns.heatmap(sd_mat, annot=True, cmap="coolwarm", linewidth=.5, vmin=0.0, fmt=".1f")

        plt.title("\\textbf{Time taken to calculate a single run}")
        plt.tight_layout()
        plt.savefig(args.figure_path.replace(".png", "_time.png"))
