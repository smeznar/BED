import numpy.random
from ProGED.generators import GeneratorGrammar
from SRToolkit.utils import tokens_to_tree, SymbolLibrary, Node
import numpy as np
import matplotlib.pyplot as plt
import editdistance
from sklearn.manifold import TSNE, MDS

from bed import BED

def same_tree(expr_tree1: Node, expr_tree2: Node) -> bool:
    if expr_tree1.symbol == expr_tree2.symbol and expr_tree1.symbol != "C":
        if expr_tree1.left is not None and not same_tree(expr_tree1.left, expr_tree2.left):
            return False
        if expr_tree1.right is not None and not same_tree(expr_tree1.right, expr_tree2.right):
            return False
        return True
    return False


def commutativity(expr_tree: Node) -> Node:
    left = expr_tree.left
    expr_tree.left = expr_tree.right
    expr_tree.right = left
    return expr_tree


def add_identity(expr_tree: Node) -> Node:
    if np.random.choice([True, False]):
        if np.random.choice([True, False]):
            return Node("+", expr_tree, Node("0"))
        else:
            return Node("+", Node("0"), expr_tree)
    else:
        if np.random.choice([True, False]):
            return Node("*", expr_tree, Node("1"))
        else:
            return Node("*", Node("1"), expr_tree)


def pow_expansion(expr_tree: Node, power: int) -> Node:
    if power < 2:
        raise Exception("Only integer powers of 2 or more are allowed")
    elif power == 2:
        return Node("*", expr_tree.left, expr_tree.left)
    else:
        if np.random.choice([True, False]):
            if np.random.choice([True, False]):
                return Node("*", pow_expansion(expr_tree, power-1), expr_tree.left)
            else:
                return Node("*", expr_tree.left, pow_expansion(expr_tree, power-1))
        else:
            return Node(f"^{power}", left=expr_tree.left)


def sine_cosine(expr_tree: Node)-> Node:
    if expr_tree.symbol == "sin":
        symbol = "cos"
    else:
        symbol = "sin"

    return Node(symbol, left=Node("-", expr_tree.left, Node("/", Node("2"), Node("pi"))))


def divison_removal(expr_tree: Node) -> Node:
    return Node("*", Node("^-1", left=expr_tree.right), expr_tree.left)


def minus_removal(expr_tree: Node) -> Node:
    return Node("+", Node("*", expr_tree.right, Node("-1")), expr_tree.left)


def combine_log(expr_tree: Node) -> Node:
    if expr_tree.left.symbol in ["log", "ln"] and expr_tree.right.symbol in ["log", "ln"] and expr_tree.left.symbol == expr_tree.right.symbol:
        return Node(expr_tree.left.symbol, left=Node("*", expr_tree.left.left, expr_tree.right.left))
    return expr_tree


def distributivity_mul(expr_tree: Node) -> Node:
    if expr_tree.left.symbol in ["+", "-"] and expr_tree.right.symbol in ["+", "-"]:
        rn = numpy.random.random()
        if rn < 0.4:
            last_plus = expr_tree.left.symbol == expr_tree.right.symbol
            c1 = Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left.left)
            c2 = Node(expr_tree.symbol, expr_tree.right.right, expr_tree.left.left)
            c3 = Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left.right)
            c4 = Node(expr_tree.symbol, expr_tree.right.right, expr_tree.left.right)
            return Node("+" if last_plus else "-", Node(expr_tree.left.symbol, Node(expr_tree.right.symbol, c1, c2), c3), c4)
        elif rn < 0.7:
            return Node(expr_tree.left.symbol, left=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.left),
                        right=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.right))
        else:
            return Node(expr_tree.right.symbol, left=Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left),
                        right=Node(expr_tree.symbol, expr_tree.right.right, expr_tree.left))
    if expr_tree.left.symbol in ["+", "-"]:
        return Node(expr_tree.left.symbol, left=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.left),
                    right=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.right))
    if expr_tree.right.symbol in ["+", "-"]:
        return Node(expr_tree.right.symbol, left=Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left),
                    right=Node(expr_tree.symbol, expr_tree.right.right, expr_tree.left))
    return expr_tree


def associativity(expr_tree: Node) -> Node:
    if expr_tree.left.symbol == expr_tree.symbol and expr_tree.right.symbol == expr_tree.symbol:
        if np.random.choice([True, False]):
            return Node(expr_tree.symbol, Node(expr_tree.symbol, expr_tree.right, expr_tree.left.right),
                        expr_tree.left.left)
        else:
            return Node(expr_tree.symbol, expr_tree.right.right,
                        Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left))
    if expr_tree.left.symbol == expr_tree.symbol:
        return Node(expr_tree.symbol, Node(expr_tree.symbol, expr_tree.right, expr_tree.left.right), expr_tree.left.left)
    if expr_tree.right.symbol == expr_tree.symbol:
        return Node(expr_tree.symbol, expr_tree.right.right, Node(expr_tree.symbol, expr_tree.right.left, expr_tree.left))
    return expr_tree


def sin_cos_identity(expr_tree: Node) -> Node:
    if (expr_tree.left.symbol == "^2"
            and expr_tree.right.symbol == "^2"
            and expr_tree.left.left.symbol in ["sin", "cos"]
            and expr_tree.right.left.symbol in ["sin", "cos"]
            and expr_tree.right.left.symbol != expr_tree.left.left.symbol
            and same_tree(expr_tree.left.left.left, expr_tree.right.left.left)):
        return Node("1")
    return expr_tree


def factor(expr_tree: Node) -> Node:
    if same_tree(expr_tree.left, expr_tree.right):
        if np.random.choice([True, False]):
            return Node("*", Node("2"), expr_tree.left)
        else:
            return Node("*", expr_tree.left, Node("2"))
    return expr_tree


def generate_subtrees(generator, sl):
    while True:
        expr_candidate = generator.generate_one(depth_limit=7)[0]
        if "C" in expr_candidate:
            continue
        subtree1 = tokens_to_tree(expr_candidate, sl)
        if len(subtree1) > 5:
            continue
        subtree2 = tokens_to_tree(expr_candidate, sl)
        return subtree1, subtree2


def expand_identity_1(expr_tree: Node, generator: GeneratorGrammar, sl: SymbolLibrary) -> Node:
    expansion_type = np.random.choice(["no", "cos", "sin", "cos_sin", "div", "log"], p=[0.6, 0.08, 0.08, 0.08, 0.08, 0.08])
    if expansion_type == "cos":
        return Node("cos", left=Node("0"))
    elif expansion_type == "sin":
        return Node("sin", left=Node("*", Node("pi"), Node("0.5")))
    elif expansion_type == "cos_sin":
        subtree1, subtree2 = generate_subtrees(generator, sl)
        return Node("+", Node("^2", left=Node("cos", left=subtree2)), Node("^2", left=Node("sin", left=subtree1)))
    elif expansion_type == "log":
        return Node("log", left=Node("10"))
    elif expansion_type == "div":
        subtree1, subtree2 = generate_subtrees(generator, sl)
        return Node("/", subtree2, subtree1)
    return expr_tree


def expand_identity_0(expr_tree: Node, generator: GeneratorGrammar, sl: SymbolLibrary) -> Node:
    expansion_type = np.random.choice(["no", "cos", "sin", "sqrt", "sub", "log"], p=[0.6, 0.08, 0.08, 0.08, 0.08, 0.08])
    if expansion_type == "cos":
        return Node("cos", left=Node("*", Node("pi"), Node("0.5")))
    elif expansion_type == "sin":
        return Node("sin", left=Node("0"))
    elif expansion_type == "log":
        return Node("log", left=Node("1"))
    elif expansion_type == "sqrt":
        return Node("sqrt", left=Node("0"))
    elif expansion_type == "sub":
        subtree1, subtree2 = generate_subtrees(generator, sl)
        return Node("-", subtree2, subtree1)
    return expr_tree


def transform_expression(expr_tree: Node, generator: GeneratorGrammar, sl: SymbolLibrary) -> Node:
    transformations_per_symbol = {
        "+": [commutativity, combine_log, associativity, sin_cos_identity, factor],
        "*": [commutativity, distributivity_mul, associativity],
        "-": [minus_removal],
        "/": [divison_removal],
        "sin": [sine_cosine],
        "cos": [sine_cosine],
        "^2": [lambda et: pow_expansion(et, 2)],
        "^3": [lambda et: pow_expansion(et, 3)],
        "1": [lambda et: expand_identity_1(et, generator, sl)],
        "0": [lambda et: expand_identity_0(et, generator, sl)]
    }

    if expr_tree.left is not None:
        expr_tree.left = transform_expression(expr_tree.left, generator, sl)
    if expr_tree.right is not None:
        expr_tree.right = transform_expression(expr_tree.right, generator, sl)

    if np.random.random() < 0.04:
        expr_tree = add_identity(expr_tree)

    if expr_tree.symbol in transformations_per_symbol:
        if np.random.choice([True, False]):
            fn_ = np.random.choice(transformations_per_symbol[expr_tree.symbol])
            expr_tree = fn_(expr_tree)

    return expr_tree


def generate_equivalent(expr, num_equivalent, symbol_library, generator, length_limit=40):
    equivalent_expressions = [expr]
    eq_strings_set = {''.join(expr)}
    while len(eq_strings_set) < num_equivalent:
        new_expression = []
        expr_tree = tokens_to_tree(expr, symbol_library)
        for i in range(np.random.randint(4)+1):
            new_expression = transform_expression(expr_tree, generator, symbol_library).to_list(symbol_library=symbol_library)
            # print("".join(new_expression))
            expr_tree = tokens_to_tree(new_expression, symbol_library)

        if 0 < length_limit < len(new_expression):
            continue
        if "".join(new_expression) not in eq_strings_set:
            equivalent_expressions.append(new_expression)
            eq_strings_set.add("".join(new_expression))
            # print(len(eq_strings_set))
    return equivalent_expressions


def show_TSNE_clusters(distance_matrix, colors, markers, exprs, num):
    tsne = TSNE(n_components=2, metric='precomputed', init="random")
    embedding = tsne.fit_transform(distance_matrix)  # shape: (N, 2)
    # Step 3: Visualize
    plt.figure(figsize=(8, 6))
    for i in range(distance_matrix.shape[0]//num):
        plt.scatter(embedding[(i*num):((i+1)*num), 0], embedding[(i*num):((i+1)*num), 1], color=colors[(i*num)], marker=markers[(i*num)], label=exprs[(i*num)])
    plt.title("t-SNE with Precomputed Distances")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def show_MDS_clusters(distance_matrix, colors, markers, exprs, num):
    mds = MDS(n_components=2, dissimilarity='precomputed')
    embedding = mds.fit_transform(distance_matrix)  # shape: (N, 2)
    # Step 3: Visualize
    plt.figure(figsize=(8, 6))
    for i in range(distance_matrix.shape[0]//num):
        plt.scatter(embedding[(i*num):((i+1)*num), 0], embedding[(i*num):((i+1)*num), 1], color=colors[(i*num)], marker=markers[(i*num)], label=exprs[(i*num)])
    plt.title("MDS with Precomputed Distances")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


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
    R -> 'log' '(' E ')' [0.1]
    R -> 'sqrt' '(' E ')' [0.07]
    R -> '(' E ')' '^2' [0.07]
    R -> '(' E ')' '^3' [0.06]
    R -> 'sin' '(' E ')' [0.05]
    R -> 'cos' '(' E ')' [0.05]
    V -> 'X_0' [0.5]
    V -> 'X_1' [0.5]
    """
    num_equivalent = 10

    expressions = [
        ("C", "blue", "o"),  # constant

        ("C + C * X_0", "green", "o"),  # linear
        ("C + C * X_1", "green", "s"),
        ("C + C * X_0 + C * X_1", "green", "^"),

        ("C + C * X_0 + C * X_0 ^2", "red", "o"),  # polynomial
        ("C + C * X_1 + C * X_1 ^2", "red", "s"),
        ("C + C * X_0 ^2", "red", "*"),
        ("C + C * X_1 ^2", "red", "P"),
        ("C + C * X_0 + C * X_1 ^2", "red", "v"),
        ("C + C * X_1 + C * X_0 ^2", "red", "^"),
        ("C + C * X_0 + C * X_1 + C * X_0 * X_1 + C * X_0 ^2 + C * X_1 ^2", "red", "x"),

        ("sin ( C * X_0 )", "purple", "o"),  # trigonometric
        ("cos ( C * X_0 )", "purple", "s"),
        ("sin ( C * X_1 )", "purple", "^"),
        ("cos ( C * X_1 )", "purple", "v"),

        ("C + sqrt ( X_0 )", "orange", "o"),  # root
        ("C + sqrt ( X_1 )", "orange", "s"),

        ("log ( C * X_0 ^2 + 1 )", "brown", "o"),  # logarithmic
        ("log ( C * X_1 ^2 + 1 )", "brown", "s"),
    ]
    expressions = [(expression[0].split(" "), expression[1], expression[2]) for expression in expressions]
    colors = []
    markers =  []
    labels = []
    np.random.seed()
    sl = SymbolLibrary.default_symbols(3)
    generator = GeneratorGrammar(grammar)
    all_expressions = []
    for expr in expressions:
        equivalent = generate_equivalent(expr[0], num_equivalent, sl, generator)
        print("--------------------------------")
        print(f"       {''.join(expr[0])}")
        print("--------------------------------")
        for e in equivalent:
            print(''.join(e))
        all_expressions += equivalent
        colors += [expr[1] for i in range(num_equivalent)]
        markers += [expr[2] for i in range(num_equivalent)]
        labels += ["".join(expr[0]) for i in range(num_equivalent)]

    bed = BED(all_expressions, [[1,5],[10,15]]).calculate_distances()
    bed = np.log10(bed+1)
    show_TSNE_clusters(bed, colors, markers, labels, num_equivalent)
    show_MDS_clusters(bed, colors, markers, labels, num_equivalent)


    edit = np.zeros((len(all_expressions), len(all_expressions)))
    for i in range(len(all_expressions)):
        for j in range(i+1, len(all_expressions)):
            edit[i, j] = edit[j, i] = editdistance.eval(all_expressions[i], all_expressions[j])

    show_TSNE_clusters(edit, colors, markers, labels, num_equivalent)
    show_MDS_clusters(edit, colors, markers, labels, num_equivalent)
