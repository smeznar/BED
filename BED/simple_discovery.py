import numpy as np
from ProGED.generators import GeneratorGrammar

def generate_batch(num_expressions, generator):
    expressions = []
    for i in range(num_expressions):
        expressions.append(generator.generate_one()[0])
    return expressions


def calculate_distances(best_exprs, new_exprs, distance):
    distance_matrix = np.zeros((len(best_exprs), len(new_exprs)))
    for i in range(len(best_exprs)):
        for j in range(i+1, len(new_exprs)):
            d = distance(best_exprs[i], new_exprs[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
    return distance_matrix


def select_best(distance_matrix, num_selected):
    avg_distance = np.mean(distance_matrix, axis=0)
    return np.argsort(avg_distance)[:num_selected]


def calculate_error(expr):
    pass


if __name__ == '__main__':
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
    R -> 'sqrt' '(' E ')' [0.1]
    R -> '(' E ')' '^2' [0.1]
    R -> '(' E ')' '^3' [0.1]
    R -> 'sin' '(' E ')' [0.1]
    V -> 'A' [0.3]
    V -> 'B' [0.3]
    V -> 'D' [0.4]
    """
    generator = GeneratorGrammar(grammar)
    expr = generator.generate_one()[0]

