import numpy.random
from ProGED.generators import GeneratorGrammar
from SRToolkit.utils import tokens_to_tree, SymbolLibrary, Node
import numpy as np


def same_tree(expr_tree1: Node, expr_tree2: Node) -> bool:
    if expr_tree1.symbol == expr_tree2.symbol:
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


def transform_expression(expr_tree: Node, level) -> Node:
    # print(f"Level: {level}, expression in: {str(expr_tree)}")
    transformations_per_symbol = {
        "+": [commutativity, combine_log, associativity, sin_cos_identity, factor],
        "*": [commutativity, distributivity_mul, associativity],
        "-": [minus_removal],
        "/": [divison_removal],
        "sin": [sine_cosine],
        "cos": [sine_cosine],
        "^2": [lambda et: pow_expansion(et, 2)],
        "^3": [lambda et: pow_expansion(et, 3)]
    }

    if expr_tree.left is not None:
        expr_tree.left = transform_expression(expr_tree.left, level+1)
    if expr_tree.right is not None:
        expr_tree.right = transform_expression(expr_tree.right, level+1)

    if np.random.random() < 0.02:
        expr_tree = add_identity(expr_tree)

    if expr_tree.symbol in transformations_per_symbol:
        if np.random.choice([True, False]):
            fn_ = np.random.choice(transformations_per_symbol[expr_tree.symbol])
            before = str(expr_tree)
            expr_tree = fn_(expr_tree)
            # print(f"{str(fn_)}, Before: {before}, After: {str(expr_tree)}")

    # print(f"Level: {level}, expression out: {str(expr_tree)}")
    return expr_tree


def generate_equivalent(expr, num_equivalent, symbol_library):
    equivalent_expressions = [expr]
    eq_strings_set = {''.join(expr)}
    while len(eq_strings_set) < num_equivalent:
        expr_tree = tokens_to_tree(expr, symbol_library)
        for i in range(np.random.randint(4)+1):
            new_expression = transform_expression(expr_tree, 0).to_list(symbol_library=symbol_library)
            # print("".join(new_expression))
            expr_tree = tokens_to_tree(new_expression, symbol_library)

        if "".join(new_expression) not in eq_strings_set:
            equivalent_expressions.append(new_expression)
            eq_strings_set.add("".join(new_expression))
            # print(len(eq_strings_set))
    return equivalent_expressions


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
    R -> '(' E ')' '**2' [0.07]
    R -> '(' E ')' '**3' [0.06]
    R -> 'sin' '(' E ')' [0.05]
    R -> 'cos' '(' E ')' [0.05]
    V -> 'X_0' [0.34]
    V -> 'X_1' [0.33]
    V -> 'X_2' [0.33]
    """
    expressions = [["sin", "(", "X_0", ")", "^2", "+", "cos", "(", "X_0", ")", "^2"],
                   ["log", "(", "X_0", ")", "+", "log", "(", "X_1", "^2", ")"]]
    np.random.seed()
    sl = SymbolLibrary.default_symbols(3)
    generator = GeneratorGrammar(grammar)
    num_exprs = 0
    for expr in expressions:
        equivalent = generate_equivalent(expr, 29, sl)
        print("--------------------------------")
        print(f"       {''.join(expr)}")
        print("--------------------------------")
        for e in equivalent:
            print(''.join(e))
        num_exprs += 1

