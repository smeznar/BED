import json

import zss


SYMBOLS = {
    "+": {"symbol": '+', "type": "op", "precedence": 0},
    "-": {"symbol": '-', "type": "op", "precedence": 0},
    "*": {"symbol": '*', "type": "op", "precedence": 1},
    "/": {"symbol": '/', "type": "op", "precedence": 1},
    "^": {"symbol": "^", "type": "op", "precedence": 2},
    "u-": {"symbol": "u-", "type": "fn", "precedence": 5},
    "sqrt": {"symbol": 'sqrt', "type": "fn", "precedence": 5},
    "sin": {"symbol": 'sin', "type": "fn", "precedence": 5},
    "cos": {"symbol": 'cos', "type": "fn", "precedence": 5},
    "exp": {"symbol": 'exp', "type": "fn", "precedence": 5},
    "log": {"symbol": 'log', "type": "fn", "precedence": 5},
    "^-1": {"symbol": "^-1", "type": "fn", "precedence": -1},
    "^2": {"symbol": '^2', "type": "fn", "precedence": -1},
    "^3": {"symbol": '^3', "type": "fn", "precedence": -1},
    "^4": {"symbol": '^4', "type": "fn", "precedence": -1},
    "^5": {"symbol": '^5', "type": "fn", "precedence": -1},
    "1": {"symbol": '1', "type": "lit", "precedence": 5},
    "pi": {"symbol": 'pi', "type": "lit", "precedence": 5},
    "e": {"symbol": 'e', "type": "lit", "precedence": 5},
    "C": {"symbol": 'C', "type": "const", "precedence": 5}
}
for char in range(25):
    SYMBOLS[f"X_{char}"] = {"symbol": f"X_{char}", "type": "var", "precedence": 5}


class Node:
    def __init__(self, symbol=None, right=None, left=None):
        self.symbol = symbol
        self.right = right
        self.left = left

    def to_postfix(self) -> list[str]:
        if self.left is None and self.right is None:
            return [self.symbol]
        elif self.right is None:
            return self.left.to_postfix() + [self.symbol]
        else:
            return self.left.to_postfix() + self.right.to_postfix() + [self.symbol]

    def to_dict(self) -> dict:
        d = {'s': self.symbol}
        if self.left is not None:
            d['l'] = self.left.to_dict()
        if self.right is not None:
            d['r'] = self.right.to_dict()
        return d

    @staticmethod
    def from_dict(d):
        left = None
        right = None
        if "l" in d:
            left = Node.from_dict(d["l"])
        if 'r' in d:
            right = Node.from_dict(d["r"])
        return Node(d["s"], right=right, left=left)

    def __len__(self):
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens: list[str]) -> Node:
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    expr_str = ''.join(tokens)
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in SYMBOLS and SYMBOLS[token]["type"] in ["var", "const", "lit"] or is_float(token):
            out_stack.append(Node(token))
        elif token in SYMBOLS and SYMBOLS[token]["type"] == "fn":
            if token[0] == "^":
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif token in SYMBOLS and SYMBOLS[token]["type"] == "op":
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and SYMBOLS[operator_stack[-1]]["precedence"] > SYMBOLS[token]["precedence"]:
                if SYMBOLS[operator_stack[-1]]["type"] == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if SYMBOLS[operator_stack[-1]]["type"] == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in SYMBOLS \
                    and SYMBOLS[operator_stack[-1]]["type"] == "fn":
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")


def infix_to_postfix(exprs: list[list[str]]) -> list[list[str]]:
    postfix = []
    for expr in exprs:
        tree = tokens_to_tree(expr)
        postfix.append(tree.to_postfix())
    return postfix


def read_expressions_json(filepath):
    with open(filepath, "r") as file:
        return [Node.from_dict(d).to_postfix() for d in json.load(file)]


def expr_to_zss(expr):
    zexpr = zss.Node(expr.symbol)
    if expr.left is not None:
        zexpr.addkid(expr_to_zss(expr.left))
    if expr.right is not None:
        zexpr.addkid(expr_to_zss(expr.right))

    return zexpr


def read_expressions_zss(filepath):
    with open(filepath, "r") as file:
        return [expr_to_zss(Node.from_dict(d)) for d in json.load(file)]
