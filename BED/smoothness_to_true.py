from argparse import ArgumentParser

import editdistance
from SRToolkit.dataset import SRBenchmark
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils import tokens_to_tree as ttt
from ProGED.generators import GeneratorGrammar
import numpy as np
import pandas as pd
import seaborn as sns
import zss
import matplotlib.pyplot as plt

from bed import BED
from utils import expr_to_zss


sns.set_theme(font_scale=0.8, palette="Set2", rc={'figure.figsize': (6, 4), 'text.usetex': True,
                                                'axes.titlesize': 80, "axes.labelsize": 12})
np.random.seed()

grammar = """
E -> E '+' F [0.2]
E -> E '-' F [0.2]
E -> F [0.6]
F -> F '*' T [0.2]
F -> F '/' T [0.2]
F -> T [0.6]
T -> R [0.2]
T -> 'C' [0.2]
T -> V [0.6]
R -> '(' E ')' [0.6]
R -> 'ln' '(' E ')' [0.1]
R -> 'sqrt' '(' E ')' [0.07]
R -> '(' E ')' '^2' [0.07]
R -> '(' E ')' '^3' [0.06]
R -> 'sin' '(' E ')' [0.05]
R -> 'cos' '(' E ')' [0.05]
"""


def rank_distance(distances, losses, name, n_best=500):
    indices = np.argsort(distances)[:n_best]
    l = losses[indices]
    x = []
    y = []
    names = []
    running_sum = 0
    for i in range(0, indices.shape[0]):
        running_sum += l[i]
        x.append(i+1)
        y.append(running_sum/(i+1))
        names.append(name)
    return x, y, names


if __name__ == '__main__':
    parser = ArgumentParser(prog='Smoothness to true equation',
                            description='A script for testing the smoothness of the error landscape')
    parser.add_argument("-dataset", type=str)
    args = parser.parse_args()

    number_of_runs = 10
    number_of_expressions = 40000
    num_points = 64
    dataset = SRBenchmark.feynman("../data/feynman").create_dataset(args.dataset)
    num_variables = dataset.X.shape[1]
    evaluator = dataset.create_evaluator()
    # evaluator.parameter_estimator.estimation_settings['bounds'] = (0.2, 5)
    for i in range(num_variables):
        grammar += f"\nV -> 'X_{i}' [{1 / num_variables}]"
    generator = GeneratorGrammar(grammar)
    expressions = []
    expr_strings_set = set()
    losses = []
    while len(expr_strings_set) < number_of_expressions:
        new_expression = generator.generate_one()[0]
        if "".join(new_expression) not in expr_strings_set:
            error = evaluator.evaluate_expr(new_expression)
            if np.isfinite(error):
                expressions.append(new_expression)
                expr_strings_set.add("".join(new_expression))
                losses.append(np.max([error, 1e-8]))

    sl = SymbolLibrary.default_symbols(10)

    # Calculate distance using edit distance
    print("Edit")
    edit = []
    for expr in expressions:
        edit.append(editdistance.eval(dataset.ground_truth, expr))
    edit = np.array(edit)

    print("Tree Edit")
    # Calculate distance using tree edit distance
    tree = []
    zss_ground_truth = expr_to_zss(ttt(dataset.ground_truth, sl))
    for expr in expressions:
        tree.append(zss.simple_distance(zss_ground_truth, expr_to_zss(ttt(expr, sl))))

    bed_distances = []
    print("BED")
    for i in range(number_of_runs):
        print(f"Iteration {i}")
        # Calculate distance using BED
        x_indices = np.random.permutation(dataset.X.shape[0])[:num_points]
        bed_distances.append(BED([dataset.ground_truth], expressions2=expressions, x=dataset.X[x_indices], const_bounds=(-5,5)).calculate_distances())

    name = []
    y = []
    x = []

    losses = np.nan_to_num(losses, nan=1e10, posinf=1e10, neginf=1e10)

    x_i, y_i, names_i = rank_distance(edit, losses, "Edit")
    x += x_i
    y += y_i
    name += names_i

    x_i, y_i, names_i = rank_distance(tree, losses, "Tree")
    x += x_i
    y += y_i
    name += names_i


    bed_min = np.zeros((10, 500))
    for i, dm in enumerate(bed_distances):
        x_i, y_i, names_i = rank_distance(dm[0], losses, "BED")
        bed_min[i, :] = np.array(y_i)
        x += x_i
        y += y_i
        name += names_i

    bed_min = np.min(bed_min, axis=0)

    x += [i+1 for i in range(500)]
    y += [v for v in bed_min]
    name += ["BED min" for i in range(500)]

    x += [i+1 for i in range(500)]
    y += [v for v in np.sort(losses)[:500]]
    name += ["Optimal" for i in range(500)]

    data = pd.DataFrame(
        {"Number of Closest Expressions": x, f"Aggregated RMSE": y,
         "Metric": name})
    sns_plot = sns.lineplot(x="Number of Closest Expressions", y=f"Aggregated RMSE",
                            hue="Metric",
                            data=data)
    sns_plot.axes.set_title(f"Expression ({args.dataset}): {ttt(dataset.ground_truth, sl).to_latex(sl)}", fontsize=10)
    # sns_plot.axes.yscale("log")
    plt.yscale("log")
    # plt.show()
    plt.savefig(f"../results/smoothness_true/{args.dataset}.png", dpi=400)