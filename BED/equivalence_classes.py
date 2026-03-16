import numpy.random
import sklearn
from SRToolkit.utils import tokens_to_tree, SymbolLibrary, Node, generate_n_expressions, expr_to_executable_function
from SRToolkit.approaches.EDHiE import TreeDataset, HVAE, train_hvae, create_batch
import numpy as np
import matplotlib.pyplot as plt
import editdistance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, v_measure_score, fowlkes_mallows_score
from sklearn.manifold import MDS
import zss
import matplotlib as mpl
from rapidfuzz.distance import Jaro
from matplotlib.patches import Polygon
import torch

from finetune_snip import load_snip_encoder
from bed import BED, expr_to_zss, maximal_hausdorff

def same_tree(expr_tree1: Node, expr_tree2: Node) -> bool:
    if expr_tree1.symbol == expr_tree2.symbol and expr_tree1.symbol != "C":
        if expr_tree1.left is not None and not same_tree(expr_tree1.left, expr_tree2.left):
            return False
        if expr_tree1.right is not None and not same_tree(expr_tree1.right, expr_tree2.right):
            return False
        return True
    return False

def contains_free_parameters(expr_tree: Node) -> bool:
    if expr_tree.symbol == "C":
        return True
    if expr_tree.left is not None and contains_free_parameters(expr_tree.left):
        return True
    if expr_tree.right is not None and contains_free_parameters(expr_tree.right):
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
        if contains_free_parameters(expr_tree.left):
            return expr_tree
        return Node("*", expr_tree.left, expr_tree.left)
    else:
        if contains_free_parameters(expr_tree.left):
            return expr_tree
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
        if contains_free_parameters(expr_tree):
            return expr_tree
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
        if contains_free_parameters(expr_tree.right):
            return expr_tree
        return Node(expr_tree.left.symbol, left=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.left),
                    right=Node(expr_tree.symbol, expr_tree.right, expr_tree.left.right))
    if expr_tree.right.symbol in ["+", "-"]:
        if contains_free_parameters(expr_tree.left):
            return expr_tree
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


def generate_subtrees(sl):
    while True:
        expr_candidate = generate_n_expressions(sl, 1, verbose=False)[0]
        if "C" in expr_candidate:
            continue
        subtree1 = tokens_to_tree(expr_candidate, sl)
        if len(subtree1) > 5:
            continue
        if np.all(np.isclose(expr_to_executable_function(expr_candidate, sl)(np.random.random(size=(20, 2))*4+1, None), np.zeros(20))):
            continue
        subtree2 = tokens_to_tree(expr_candidate, sl)
        return subtree1, subtree2


def expand_identity_1(expr_tree: Node, sl: SymbolLibrary) -> Node:
    expansion_type = np.random.choice(["no", "cos", "sin", "cos_sin", "div", "log"], p=[0.6, 0.08, 0.08, 0.08, 0.08, 0.08])
    if expansion_type == "cos":
        return Node("cos", left=Node("0"))
    elif expansion_type == "sin":
        return Node("sin", left=Node("*", Node("pi"), Node("0.5")))
    elif expansion_type == "cos_sin":
        subtree1, subtree2 = generate_subtrees(sl)
        return Node("+", Node("^2", left=Node("cos", left=subtree2)), Node("^2", left=Node("sin", left=subtree1)))
    elif expansion_type == "log":
        return Node("log", left=Node("10"))
    elif expansion_type == "div":
        subtree1, subtree2 = generate_subtrees(sl)
        return Node("/", subtree2, subtree1)
    return expr_tree


def expand_identity_0(expr_tree: Node, sl: SymbolLibrary) -> Node:
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
        subtree1, subtree2 = generate_subtrees(sl)
        return Node("-", subtree2, subtree1)
    return expr_tree


def transform_expression(expr_tree: Node, sl: SymbolLibrary) -> Node:
    transformations_per_symbol = {
        "+": [commutativity, combine_log, associativity, sin_cos_identity, factor],
        "*": [commutativity, distributivity_mul, associativity],
        "-": [minus_removal],
        "/": [divison_removal],
        "sin": [sine_cosine],
        "cos": [sine_cosine],
        "^2": [lambda et: pow_expansion(et, 2)],
        "^3": [lambda et: pow_expansion(et, 3)],
        "1": [lambda et: expand_identity_1(et, sl)],
        "0": [lambda et: expand_identity_0(et, sl)]
    }

    if expr_tree.left is not None:
        expr_tree.left = transform_expression(expr_tree.left, sl)
    if expr_tree.right is not None:
        expr_tree.right = transform_expression(expr_tree.right, sl)

    if np.random.random() < 0.04:
        expr_tree = add_identity(expr_tree)

    if expr_tree.symbol in transformations_per_symbol:
        if np.random.choice([True, False]):
            fn_ = np.random.choice(transformations_per_symbol[expr_tree.symbol])
            expr_tree = fn_(expr_tree)

    return expr_tree


def generate_equivalent(expr, num_equivalent, symbol_library, length_limit=40):
    equivalent_expressions = [expr]
    eq_strings_set = {''.join(expr)}
    while len(eq_strings_set) < num_equivalent:
        new_expression = []
        expr_tree = tokens_to_tree(expr, symbol_library)
        for i in range(np.random.randint(4)+1):
            new_expression = transform_expression(expr_tree, symbol_library).to_list(symbol_library=symbol_library)
            expr_tree = tokens_to_tree(new_expression, symbol_library)

        if 0 < length_limit < len(new_expression):
            continue
        if "".join(new_expression) not in eq_strings_set:
            equivalent_expressions.append(new_expression)
            eq_strings_set.add("".join(new_expression))
    return equivalent_expressions


def show_MDS_clusters(distance_matrix, colors, markers, exprs, num, baseline, precomputed=True,
                      equivalence_groups_for_ga=False):
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    if precomputed:
        mds = MDS(n_components=2, dissimilarity='precomputed', n_init=10, random_state=42)
    else:
        mds = MDS(n_components=2, n_init=10, random_state=42)
    embedding = mds.fit_transform(distance_matrix)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    if equivalence_groups_for_ga:
        for group, c in [([1,2,3,4], "green"), ([5,6,7], "red"), ([8,9,10], "purple"), ([11,12], "orange"), ([13,14,15], "brown")]:
            patch = Polygon(embedding[group], facecolor=c, edgecolor=c, linewidth=0.6, alpha=0.4)
            ax.add_patch(patch)

    seen_labels = set()
    for i in range(0, distance_matrix.shape[0], num):
        group_label = exprs[i]
        if group_label not in seen_labels:
            ax.scatter(
                embedding[i:i+num, 0],
                embedding[i:i+num, 1],
                color=colors[i],
                marker=markers[i],
                s=60,
                edgecolors='black',
                linewidths=0.6,
                alpha=0.85,
                label=group_label.replace("C", "c").replace("X", "x")
            )
            seen_labels.add(group_label)

    if baseline == "BED" and equivalence_groups_for_ga:
        ax.set_title(f"Expression Space Structured by Behavior ({baseline})", weight='bold')
    elif equivalence_groups_for_ga:
        ax.set_title(f"Expression Space Structured by Syntax ({baseline})", weight='bold')
    else:
        ax.set_title(f"MDS Visualization of Dissimilarities Computed Using {baseline}", weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    legend = ax.legend(
        title="Expressions",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
        ncol=1,
        borderaxespad=0.
    )
    legend._legend_box.align = "left"
    plt.setp(legend.get_title(), fontsize=13, weight='bold')

    ax.grid(True, linestyle='', alpha=0.9)
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.show()


def graphical_abstract_figure(eq_classes):
    expressions = [(expression[0].split(" "), expression[1], expression[2]) for expression in eq_classes]
    colors = []
    markers = []
    labels = []
    np.random.seed()
    all_expressions = []
    for expr in expressions:
        equivalent = [expr[0]]
        all_expressions += equivalent
        colors += [expr[1]]
        markers += [expr[2]]
        labels += [tokens_to_tree(expr[0], sl).to_latex(sl)]

    bed = BED(all_expressions, [[1, 5], [1, 5]]).calculate_distances()
    bed = np.log10(bed + 1)
    show_MDS_clusters(bed, colors, markers, labels, 1, "BED", precomputed=True,
                      equivalence_groups_for_ga=True)

    zss_exprs = []
    for expr in all_expressions:
        zss_exprs.append(expr_to_zss(tokens_to_tree(expr, sl)))
    tree_edit = np.zeros((len(zss_exprs), len(zss_exprs)))
    for i in range(len(zss_exprs)):
        for j in range(i + 1, len(zss_exprs)):
            tree_edit[i, j] = tree_edit[j, i] = zss.simple_distance(zss_exprs[i], zss_exprs[j])
    show_MDS_clusters(tree_edit, colors, markers, labels, 1, "Tree edit distance",
                      precomputed=True, equivalence_groups_for_ga=True)


def print_results(baseline, results):
    res = np.array(results)
    mean = np.mean(res, axis=0)
    std = np.std(res, axis=0)
    print()
    print(baseline)
    print(f"ARI: {mean[0]} (+- {std[0]})")
    print(f"Silhouette: {mean[1]} (+- {std[1]})")
    print(f"V-measure: {mean[2]} (+- {std[2]})")
    print(f"Fowlkes-Mallows: {mean[3]} (+- {std[3]})")
    print("All: ", results)


def train_hvae_model(sl: SymbolLibrary):
    latent_size = 32
    num_expressions = 50000
    max_beta = 0.03
    max_expression_length = 30
    sl.add_symbol("0", "lit", 5, "np.full(X.shape[0], 0)", "0")
    sl.add_symbol("1", "lit", 5, "np.full(X.shape[0], 1)", "1")
    sl.add_symbol("2", "lit", 5, "np.full(X.shape[0], 2)", "2")
    sl.add_symbol("-1", "lit", 5, "np.full(X.shape[0], -1)", "-1")
    sl.add_symbol("0.5", "lit", 5, "np.full(X.shape[0], 0.5)", "0.5")
    sl.add_symbol("10", "lit", 5, "np.full(X.shape[0], 10)", "10")

    if True:
        model = HVAE(len(sl), latent_size, sl)
        model.load_state_dict(torch.load("eq_model.pt"))
        model.eval()
        # model = torch.load("eq_model.pt")
        return model, sl.symbols2index()
    # Possibly create a training set or load expressions
    expressions = generate_n_expressions(sl, num_expressions, max_expression_length=max_expression_length)
    expr_tree = [tokens_to_tree(expr, sl) for expr in expressions]
    # Create a training set
    trainset = TreeDataset(expr_tree)

    # Train the model
    model = HVAE(len(sl), latent_size, sl)
    train_hvae(model, trainset, sl, epochs=20, max_beta=max_beta)
    torch.save(model.state_dict(), f"eq_model.pt")
    return model, sl.symbols2index()


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
    num_equivalent = 10 # 1 for graphical abstract
    sl = SymbolLibrary.default_symbols(2)
    num_runs = 1
    show_MDS_plots = True # For MDS plots this should be True and num_runs = 1
    verbose = True

    eq_classes = [
        # constant
        ("C", "blue", "o"),

        # linear
        ("C * X_0", "green", "o"),
        ("C * X_1", "green", "s"),
        ("C + C * X_1", "green", "^"),
        ("C + C * X_0 + C * X_1", "green", "v"),

        # quadratic
        ("C * X_0 ^ 2", "red", "o"),
        ("C + C * X_0 * X_1", "red", "s"),
        ("C * X_0 ^ 2 + C * X_1 ^ 2", "red", "^"),

        # trigonometric
        ("C * sin ( X_0 )", "purple", "o"),
        ("cos ( C * X_1 )", "purple", "s"),
        ("C * sin ( X_0 ) + C * cos ( X_1 )", "purple", "^"),

        # containing square root
        ("sqrt ( C * X_0 )", "orange", "o"),
        ("C + sqrt ( X_0 + X_1 )", "orange", "s"),

        # logarithmic
        ("C * log ( X_0 )", "brown", "o"),
        ("log ( X_1 + C )", "brown", "s"),
        ("C * log ( X_0 * X_1 )", "brown", "^"),
    ]

    # Uncomment the next line for clustering plots in the graphical abstract
    # graphical_abstract_figure(eq_classes)

    results = {"BED": [], "BED features": [], "Edit distance": [], "Edit features": [], "Jaro distance": [],
               "JARO features": [], "Tree edit distance": [], "Tree edit features": [],"maximal distance": [],
               "maximal features": [], "Hausdorff distance": [], "Hausdorff features": [], "HVAE": [],
               "HVAE features": [], "SNIP": [], "SNIP features": [],}

    model, s2i = train_hvae_model(sl)
    snip_encoder = load_snip_encoder(
        "../Multimodal-Math-Pretraining-main/weights/snip-10dmax-finetuned-equivalence-classes/best_checkpoint.pth",
        sl,
        base_checkpoint="../Multimodal-Math-Pretraining-main/weights/snip-10dmax.pth",
    )

    for run_seed in range(num_runs):
        print(f"RUN {run_seed+1}/{num_runs}")
        np.random.seed(run_seed)
        expressions = [(expression[0].split(" "), expression[1], expression[2]) for expression in eq_classes]
        ground_truth = []
        for i in range(len(expressions)):
            for j in range(num_equivalent):
                ground_truth.append(i)

        colors = []
        markers =  []
        labels = []
        all_expressions = []
        for expr in expressions:
            equivalent = generate_equivalent(expr[0], num_equivalent, sl)
            if verbose:
                print("--------------------------------")
                print(f"       {''.join(expr[0])}")
                print("--------------------------------")
                for e in equivalent:
                    print(''.join(e))
            all_expressions += equivalent
            colors += [expr[1] for i in range(num_equivalent)]
            markers += [expr[2] for i in range(num_equivalent)]
            labels += [tokens_to_tree(expr[0], sl).to_latex(sl) for i in range(num_equivalent)]

        neuro_exprs = [tokens_to_tree(expr, sl) for expr in all_expressions]
        embeddings = model.encode(create_batch(neuro_exprs, s2i))[0]
        hvae_dist = torch.cdist(embeddings, embeddings, 2).detach().numpy()
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(hvae_dist)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(hvae_dist, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["HVAE"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("HVAE")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(hvae_dist, colors, markers, labels, num_equivalent, "HVAE",
                              precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        hvae_dist = normalizer.fit_transform(hvae_dist)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(hvae_dist)
        np.fill_diagonal(hvae_dist, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(hvae_dist, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["HVAE features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized HVAE as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        snip_embeddings = snip_encoder.encode(all_expressions, sl)
        snip_dist = torch.cdist(snip_embeddings, snip_embeddings, p=2).numpy()
        np.fill_diagonal(snip_dist, 0)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), metric="precomputed", linkage="single").fit_predict(snip_dist)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(snip_dist, clusters, metric="precomputed")
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["SNIP"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("SNIP")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(snip_dist, colors, markers, labels, num_equivalent, "SNIP",
                              precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        snip_dist = normalizer.fit_transform(snip_dist)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(snip_dist)
        np.fill_diagonal(snip_dist, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(snip_dist, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["SNIP features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized SNIP as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)


        bed = BED(all_expressions, [[1,5],[1,5]], seed=run_seed).calculate_distances()
        bed = np.log10(bed+1)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), metric="precomputed", linkage="single").fit_predict(bed)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(bed, clusters, metric="precomputed")
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["BED"].append([ari, silhouette, v_measure, fowlkes_mallows])

        if verbose:
            print()
            print("BED")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(bed, colors, markers, labels, num_equivalent, "BED", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        bed = normalizer.fit_transform(bed)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(bed)
        np.fill_diagonal(bed, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(bed, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["BED features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized BED as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)


        maximal = maximal_hausdorff(all_expressions, [[1, 5], [1, 5]], seed=run_seed).calculate_distances()
        maximal = np.log10(maximal+1)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), metric="precomputed", linkage="single").fit_predict(maximal)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(maximal, clusters, metric="precomputed")
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["maximal distance"].append([ari, silhouette, v_measure, fowlkes_mallows])

        if verbose:
            print()
            print("maximal")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(maximal, colors, markers, labels, num_equivalent, "Maximal", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        maximal = normalizer.fit_transform(maximal)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(maximal)
        np.fill_diagonal(maximal, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(maximal, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["maximal features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized maximal as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        hausdorf = maximal_hausdorff(all_expressions, [[1, 5], [1, 5]], seed=run_seed, hausdorff=True).calculate_distances()
        hausdorf = np.log10(hausdorf+1)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), metric="precomputed", linkage="single").fit_predict(hausdorf)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(hausdorf, clusters, metric="precomputed")
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Hausdorff distance"].append([ari, silhouette, v_measure, fowlkes_mallows])

        if verbose:
            print()
            print("hausdorf")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(hausdorf, colors, markers, labels, num_equivalent, "Hausdorff", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        hausdorf = normalizer.fit_transform(hausdorf)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(hausdorf)
        np.fill_diagonal(hausdorf, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(hausdorf, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Hausdorff features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized hausdorf as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)


        edit = np.zeros((len(all_expressions), len(all_expressions)))
        for i in range(len(all_expressions)):
            for j in range(i+1, len(all_expressions)):
                edit[i, j] = edit[j, i] = editdistance.eval(all_expressions[i], all_expressions[j])
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(edit)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(edit, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Edit distance"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Edit distance")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(edit, colors, markers, labels, num_equivalent, "Edit distance", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        edit = normalizer.fit_transform(edit)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(edit)
        np.fill_diagonal(edit, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(edit, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Edit features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized Edit distance as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)


        jaro = np.zeros((len(all_expressions), len(all_expressions)))
        for i in range(len(all_expressions)):
            for j in range(i+1, len(all_expressions)):
                jaro[i, j] = jaro[j, i] = Jaro.distance(all_expressions[i], all_expressions[j])
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(jaro)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(edit, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Jaro distance"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("JARO distance")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(jaro, colors, markers, labels, num_equivalent, "JARO distance", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        jaro = normalizer.fit_transform(jaro)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(jaro)
        np.fill_diagonal(jaro, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(jaro, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["JARO features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized JARO as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        zss_exprs = []
        for expr in all_expressions:
            zss_exprs.append(expr_to_zss(tokens_to_tree(expr, sl)))
        tree_edit = np.zeros((len(zss_exprs), len(zss_exprs)))
        for i in range(len(zss_exprs)):
            for j in range(i+1, len(zss_exprs)):
                tree_edit[i, j] = tree_edit[j, i] = zss.simple_distance(zss_exprs[i], zss_exprs[j])
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(tree_edit)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(tree_edit, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Tree edit distance"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Tree edit distance")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

        if show_MDS_plots:
            show_MDS_clusters(tree_edit, colors, markers, labels, num_equivalent, "Tree edit distance", precomputed=True)

        normalizer = sklearn.preprocessing.Normalizer()
        tree_edit = normalizer.fit_transform(tree_edit)
        clusters = AgglomerativeClustering(n_clusters=len(expressions), linkage="single").fit_predict(tree_edit)
        np.fill_diagonal(tree_edit, 0)
        ari = adjusted_rand_score(clusters, np.array(ground_truth))
        silhouette = silhouette_score(tree_edit, clusters)
        v_measure = v_measure_score(clusters, np.array(ground_truth))
        fowlkes_mallows = fowlkes_mallows_score(clusters, np.array(ground_truth))
        results["Tree edit features"].append([ari, silhouette, v_measure, fowlkes_mallows])
        if verbose:
            print()
            print("Normalized tree edit as features")
            print("ARI: ", ari)
            print("Silhouette: ", silhouette)
            print("V-measure: ", v_measure)
            print("Fowlkes-Mallows: ", fowlkes_mallows)

    print()
    print("----------------------------------------------------------------")
    print()
    print_results("HVAE", results["HVAE"])
    print_results("HVAE-cf", results["HVAE features"])
    print_results("BED", results["BED"])
    print_results("BED-cf", results["BED features"])
    print_results("Edit distance", results["Edit distance"])
    print_results("Edit-cf", results["Edit features"])
    print_results("JARO distance", results["Jaro distance"])
    print_results("JARO-cf", results["JARO features"])
    print_results("Tree edit distance", results["Tree edit distance"])
    print_results("Tree-cf", results["Tree edit features"])
    print_results("maximal", results["maximal distance"])
    print_results("maximal-cf", results["maximal features"])
    print_results("Hausdorff", results["Hausdorff distance"])
    print_results("Hausdorff-cf", results["Hausdorff features"])
    print_results("SNIP", results["SNIP"])
    print_results("SNIP-cf", results["SNIP features"])

# HVAE
# ARI: 0.021410323648827365 (+- 0.027117858674885618)
# Silhouette: -0.09336840258911253 (+- 0.062912134282581)
# V-measure: 0.26674686039782947 (+- 0.07264707694398553)
# Fowlkes-Mallows: 0.2212366778763745 (+- 0.009401440632700417)
# All:  [[0.08499637738631638, 0.015651483088731766, 0.42798515652043534, 0.244101155959686], [0.012033870564825263, -0.07807040959596634, 0.2542061890136456, 0.22151055602924974], [0.02315037020614111, 0.012741106562316418, 0.3332052885677644, 0.22528175503649567], [0.005256078634247284, -0.12726588547229767, 0.21597128778452507, 0.2144602408010174], [0.0031250819073887015, -0.1424402892589569, 0.20113512023248076, 0.21372159728624868], [0.003825772051835941, -0.13570915162563324, 0.21605125134350422, 0.21339469429116545], [0.009011190700757093, -0.12113277614116669, 0.24414493390039418, 0.21792562893779996], [0.005672988366085413, -0.15359224379062653, 0.21354068655849023, 0.2167278447176676], [0.005049637522945843, -0.1580866128206253, 0.2134401819900422, 0.21408857989902647], [0.06198186914773066, -0.04577924683690071, 0.34778850806701284, 0.231154725805388]]
#
# HVAE-cf
# ARI: 0.1314184089053694 (+- 0.024170747072340844)
# Silhouette: -0.015406004711985588 (+- 0.04591794584664286)
# V-measure: 0.4590116480695713 (+- 0.034046263071111214)
# Fowlkes-Mallows: 0.2621680048504914 (+- 0.017898662950903498)
# All:  [[0.10640372839714557, -0.04139013960957527, 0.4661171444713401, 0.25947842703074947], [0.15154331435605478, -0.008747830986976624, 0.48678471697185066, 0.2768115942028986], [0.10696334724023522, -0.03940325230360031, 0.43837013207602593, 0.24688798723887415], [0.16679195455644819, 0.03287452459335327, 0.4966108725633143, 0.28576644245824073], [0.11091375567396881, -0.01797371730208397, 0.43247599088434785, 0.24860679188346244], [0.14439897655579353, 0.0663650780916214, 0.46076492273554714, 0.23401066865335277], [0.11163211712636238, -0.043534956872463226, 0.412791888164919, 0.2511100223748868], [0.1661960467930617, 0.04243091121315956, 0.5170388572197511, 0.28782025952339546], [0.10621536618241725, -0.06380486488342285, 0.4082430230605721, 0.25125770591687074], [0.14312548217220675, -0.08087579905986786, 0.47091893254804434, 0.27993014922218323]]
#
# BED
# ARI: 0.19236237726443975 (+- 2.7755575615628914e-17)
# Silhouette: 0.5437290672821106 (+- 0.0017618435152506424)
# V-measure: 0.6914860120802387 (+- 1.530336829712622e-16)
# Fowlkes-Mallows: 0.38526681219424835 (+- 0.0)
# All:  [[0.19236237726443972, 0.5423157075273891, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5424000467102608, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5400976686032012, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5444146561201098, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.542731195134311, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5460431583725057, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5457044001372612, 0.6914860120802385, 0.38526681219424835], [0.19236237726443972, 0.5435579164950687, 0.6914860120802389, 0.38526681219424835], [0.19236237726443972, 0.5448694586021805, 0.6914860120802385, 0.38526681219424835], [0.19236237726443972, 0.5451564651188168, 0.6914860120802385, 0.38526681219424835]]
#
# BED-cf
# ARI: 1.0 (+- 0.0)
# Silhouette: 0.9388862016393272 (+- 0.0012683937380655952)
# V-measure: 1.0 (+- 0.0)
# Fowlkes-Mallows: 1.0 (+- 0.0)
# All:  [[1.0, 0.9384575073657933, 1.0, 1.0], [1.0, 0.9387425317153454, 1.0, 1.0], [1.0, 0.9358509561728319, 1.0, 1.0], [1.0, 0.9392878804328808, 1.0, 1.0], [1.0, 0.9385568317614051, 1.0, 1.0], [1.0, 0.9399173627909783, 1.0, 1.0], [1.0, 0.9399512735975598, 1.0, 1.0], [1.0, 0.9383134219752712, 1.0, 1.0], [1.0, 0.9389299016720966, 1.0, 1.0], [1.0, 0.9408543489091119, 1.0, 1.0]]
#
# Edit distance
# ARI: 0.003588324015489057 (+- 0.0017858057811085079)
# Silhouette: 0.4094120454565374 (+- 0.0815303650346733)
# V-measure: 0.18728668003165436 (+- 0.01135964222973798)
# Fowlkes-Mallows: 0.2148265615006386 (+- 0.002994228398888602)
# All:  [[0.005733695708421909, 0.3765932426523172, 0.19884844641253666, 0.2152602690687491], [0.003123514632987031, 0.48065422238758637, 0.17408268429083884, 0.2181955050107931], [0.0016721888991487743, 0.20306345122937586, 0.17327096817818866, 0.2155961297733631], [0.001272686707602169, 0.45556086584160693, 0.1741592434213867, 0.2148670990021539], [0.004683793509658055, 0.3868592115107315, 0.20028064306342, 0.21642034805009266], [0.005283852792737191, 0.3542583109991158, 0.19619061210014418, 0.219017111380266], [0.006190241241178145, 0.48097525622642817, 0.1968152832794013, 0.21043691247477797], [0.0010670391603371844, 0.4764804007988932, 0.17342485630139812, 0.21449727611230507], [0.0028013965045449304, 0.456992307695093, 0.18976610596954407, 0.2086766498117006], [0.004054830998275184, 0.4226831852242263, 0.19602795729968522, 0.21529831432218488]]
#
# Edit-cf
# ARI: 0.03553198394614314 (+- 0.007143225561883319)
# Silhouette: 0.03706794268753694 (+- 0.036903928993469295)
# V-measure: 0.32241475588645635 (+- 0.020776807392130725)
# Fowlkes-Mallows: 0.21116955299903672 (+- 0.008708131229854546)
# All:  [[0.030785658086995616, 0.014684057113792282, 0.3270175512065152, 0.2202909587542706], [0.02301585548979772, -0.011456757301317694, 0.28542888145908984, 0.21047861052498412], [0.039540572157821974, 0.05610936744304766, 0.3165841737818791, 0.2064063093686968], [0.04636634135206462, 0.11020012385646719, 0.36461281659273065, 0.2021997568305824], [0.02483262917438504, 9.616120605545342e-05, 0.2933960341695515, 0.2023773960185522], [0.036099344811823864, 0.005723253215475908, 0.3287298543402775, 0.21872181200392724], [0.03804874243478148, 0.044146674059891165, 0.3346833149538856, 0.21408890674909817], [0.03550349440022768, 0.049701308865742616, 0.3209905513719842, 0.21102175551286698], [0.0369680311387408, 0.08329211547755819, 0.3214826296365099, 0.19853779312181122], [0.0441591704147926, 0.018183122938656517, 0.3312217513521399, 0.2275722311055775]]
#
# JARO distance
# ARI: 0.003774747577136588 (+- 0.0012890978955601533)
# Silhouette: -0.28248825262169197 (+- 0.06322482698982426)
# V-measure: 0.18843097210854784 (+- 0.01140824955482561)
# Fowlkes-Mallows: 0.21482391308023957 (+- 0.0019993064693500946)
# All:  [[0.003384532685626686, -0.3455793475345029, 0.18479372228616223, 0.21564184767748792], [0.0033799247997384338, -0.15656421361931036, 0.18921695116495865, 0.21262285392040822], [0.005629957496096053, -0.36173610440608406, 0.19653828643813656, 0.21516937483095078], [0.006172520073109795, -0.23747737708408928, 0.2007248144548229, 0.2160263198483532], [0.002130464042943784, -0.3671209380131643, 0.17932568723549377, 0.21340247733792478], [0.0018578140232399128, -0.2823677954785837, 0.16684897071805485, 0.21742030574858204], [0.003376048208516354, -0.2681402054200509, 0.1855674102610446, 0.21413217366937107], [0.003413320422428662, -0.3156774961359557, 0.19118132237404498, 0.21416516529943655], [0.003993569409887229, -0.2619291517306218, 0.18059038100050676, 0.21823659361751777], [0.004409324609778968, -0.22828989679455597, 0.20952217515225324, 0.21142201885236347]]
#
# JARO-cf
# ARI: 0.0036016556287345033 (+- 0.002119025851784117)
# Silhouette: -0.037398492922658365 (+- 0.030172305164976706)
# V-measure: 0.19487603514808288 (+- 0.019616291082795306)
# Fowlkes-Mallows: 0.21208901991369541 (+- 0.0034135366839964714)
# All:  [[0.006180647637657896, -0.009295692163662736, 0.22640411038667327, 0.21003664162920094], [0.0033362989323843417, -0.02746420962548619, 0.19443653801152544, 0.20952667089120489], [0.008585959246925828, 0.02251613290164111, 0.23109937085656876, 0.21109734720940743], [0.0044249143515539295, -0.03315943490988558, 0.19399573160308708, 0.2144911567308421], [0.002552574675751337, -0.0784209983993654, 0.18018268217418812, 0.2141525739190774], [0.0016547951712157574, -0.08313352064938806, 0.16643552721796182, 0.2170530417186013], [0.002283887175973507, -0.026572953851613214, 0.20240306311073494, 0.2047384686576046], [0.0014765447239748907, -0.0517515085024657, 0.18398862312072428, 0.2107548939126557], [0.002535287666709904, -0.028537915529997394, 0.17647607687693445, 0.2156296718507556], [0.002985646705197648, -0.05816482849636051, 0.19333862812243105, 0.21340973261760432]]
#
# Tree edit distance
# ARI: 0.007155122729525816 (+- 0.003713533710899685)
# Silhouette: 0.29729193000222814 (+- 0.05483827927943328)
# V-measure: 0.21757200694710707 (+- 0.026978972095659904)
# Fowlkes-Mallows: 0.2176816461592374 (+- 0.0029135124308489694)
# All:  [[0.005283852792737191, 0.3815226744298829, 0.19619061210014418, 0.219017111380266], [0.007088168807969489, 0.3594800525229583, 0.21140072535934273, 0.2191118483418171], [0.013603229355145486, 0.2847403497077964, 0.24611068948506246, 0.21852782839650028], [0.002284397239925052, 0.35853041918929324, 0.19438382493960607, 0.21071006528783826], [0.008879506263261635, 0.2811193738167422, 0.23876226318872676, 0.21606724045846956], [0.008079121340093497, 0.20407001638382566, 0.22316550953948902, 0.22244065307100452], [0.00021178428258074276, 0.2396466689046822, 0.1589179923502115, 0.2159261340359882], [0.009543728983603492, 0.3111893646037931, 0.24643321022492493, 0.2187738673270805], [0.010384238645511325, 0.24586323018109776, 0.2435916562034017, 0.21716623556206358], [0.006193199584430256, 0.3067571502822095, 0.2167635860801611, 0.21907547773134603]]
#
# Tree-cf
# ARI: 0.060001033471575906 (+- 0.030962068930701282)
# Silhouette: 0.018211577712126767 (+- 0.05521617169350134)
# V-measure: 0.4226007199279862 (+- 0.07981198279279456)
# Fowlkes-Mallows: 0.25055865747403044 (+- 0.02642911177930626)
# All:  [[0.07797463167263205, 0.006328413251237297, 0.49317510238959, 0.2732364073626719], [0.022460190345083193, 0.12881961384221247, 0.3178921989236792, 0.21439945901909746], [0.11992619926199262, 0.06544949978998087, 0.5299426378184421, 0.2904249795989265], [0.026195608018070932, -0.033108527134026206, 0.31710162882890786, 0.22294748326356503], [0.03791578247641449, -0.08556656609215571, 0.36179354225373894, 0.2429753582316466], [0.08155323438301221, 0.009362868973574417, 0.49217901027160577, 0.265502925179156], [0.03743659144010353, 0.03826928727538348, 0.35956093689181695, 0.22407996600034935], [0.033445060363501505, 0.05201266806499112, 0.37338657904577693, 0.22435382734907064], [0.07802848859195764, -0.010002206339342956, 0.4686868744152553, 0.26838050436787886], [0.08507454816299088, 0.0105507254894129, 0.5122886884410485, 0.2792856643679418]]