import numpy as np
import matplotlib.pyplot as plt
from SRToolkit.utils import generate_n_expressions, expr_to_executable_function, tokens_to_tree, SymbolLibrary
from SRToolkit.dataset import SRBenchmark
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter
from scipy.stats import wasserstein_distance
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 400
})
plt.style.use('seaborn-v0_8-whitegrid')


def plot_cdf_comparison(expr1_tokens, expr2_tokens, x_fixed, num_samples = 5000, param_range = (0.1, 5.0),
                        symbols=SymbolLibrary.default_symbols(1)):
    num_constants1 = expr1_tokens.count("C")
    num_constants2 = expr2_tokens.count("C")
    func1 = expr_to_executable_function(expr1_tokens)
    func2 = expr_to_executable_function(expr2_tokens)
    params1 = np.random.uniform(param_range[0], param_range[1], size=(num_samples, num_constants1))
    params2 = np.random.uniform(param_range[0], param_range[1], size=(num_samples, num_constants2))
    expression_1 = tokens_to_tree(expr1_tokens, symbols).to_latex(symbols).replace("X_{0}", "x")
    expression_2 = tokens_to_tree(expr2_tokens, symbols).to_latex(symbols).replace("X_{0}", "x")
    outputs1 = np.array([func1(np.array([x_fixed])[:, None], p)[0] for p in params1])
    outputs2 = np.array([func2(np.array([x_fixed])[:, None], p)[0] for p in params2])
    outputs1.sort()
    outputs2.sort()

    cdf1 = np.concatenate([np.array([0]), np.linspace(0, 1, num_samples), np.array([1])])
    cdf2 = np.concatenate([np.array([0]), np.linspace(0, 1, num_samples), np.array([1])])
    minval = min(min(outputs1), min(outputs2))
    maxval = max(max(outputs1), max(outputs2))
    outputs1 = np.concatenate([np.array([minval]), outputs1, np.array([maxval])])
    outputs2 = np.concatenate([np.array([minval]), outputs2, np.array([maxval])])
    # Plot the CDFs
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(outputs1, cdf1, label="CDF of $\\mathit{E}$", linewidth=2, color= "#D55E00") # :="+expression_1[1:], linewidth=2, color= "#D55E00")
    ax.plot(outputs2, cdf2, label="CDF of $\\mathit{F}$", linewidth=2, color="#0072B2") # :="+expression_2[1:], linewidth=2, color="#0072B2")

    # Interpolate to find a common range for a shaded area
    all_outputs = np.unique(np.concatenate([outputs1, outputs2]))
    interp_cdf1 = np.interp(all_outputs, outputs1, cdf1)
    interp_cdf2 = np.interp(all_outputs, outputs2, cdf2)
    wasserstein_dist = wasserstein_distance(outputs1, outputs2)

    # Shade the area between the two CDFs
    ax.fill_between(all_outputs, interp_cdf1, interp_cdf2, alpha=0.2, color="grey", label=f"$\\tilde{{\\mathrm{{BED}}}}^{{(\\mathbf{{x}})}}(\\mathit{{E}}, \mathit{{F}}) = {wasserstein_dist:.2f}$")

    # Add labels and styling
    ax.set_title(f"\\boldmath $\\textbf{{CDF Comparison of Expressions }} \\mathit{{E}} \\textbf{{ and }} \\mathit{{F}} \\textbf{{ at }}\\mathbf{{x}} = {x_fixed}$",
                 fontsize=14)
    ax.set_xlabel(r"Output ($y$)", fontsize=12)
    ax.set_ylabel("Cumulative Proportion", fontsize=12)
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


def plot_bed_across_whole_domain(expr1_tokens, expr2_tokens, param_range=(0.1, 5.0), num_samples=5000,
                                 symbols=SymbolLibrary.default_symbols(1), num_points=1000):
    num_constants1 = expr1_tokens.count("C")
    num_constants2 = expr2_tokens.count("C")
    func1 = expr_to_executable_function(expr1_tokens)
    func2 = expr_to_executable_function(expr2_tokens)
    expression_1 = tokens_to_tree(expr1_tokens, symbols).to_latex(symbols).replace("X_{0}", "x")
    expression_2 = tokens_to_tree(expr2_tokens, symbols).to_latex(symbols).replace("X_{0}", "x")

    domain = np.linspace(1, 5, num_points)
    output = []
    points = []
    for x in domain:
        points.append(x)
        params1 = np.random.uniform(param_range[0], param_range[1], size=(num_samples, num_constants1))
        params2 = np.random.uniform(param_range[0], param_range[1], size=(num_samples, num_constants2))
        outputs1 = np.array([func1(np.array([x])[:, None], p)[0] for p in params1])
        outputs2 = np.array([func2(np.array([x])[:, None], p)[0] for p in params2])
        outputs1.sort()
        outputs2.sort()
        output.append(wasserstein_distance(outputs1, outputs2))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(points, output, linewidth=2, color="#009E73")
    ax.scatter([3.0], [2.26], color="red", s=50, zorder=3)
    ax.text(3.0, 1.2, f"$\\tilde{{\\textrm{{BED}}}}^{{(X)}}(\\mathit{{E}}, \\mathit{{F}})={np.mean(output):.2f}$", fontsize=16)
    ax.set_title(f"\\boldmath $\\textbf{{Value of }}\\tilde{{\\textbf{{BED}}}}^{{(\\mathbf{{x}})}}(\\mathit{{E}}, \\mathit{{F}}) \\textbf{{ for }} x\in[1, 5]$",
        fontsize=14)
    ax.set_ylabel(f"$\\tilde{{\\textrm{{BED}}}}^{{(\\mathbf{{x}})}}(\\mathit{{E}}, \\mathit{{F}})$", fontsize=12)
    # ax.set_xlabel("Domain ($x$)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    # ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

def create_density_plot(domain, expr, expr_str, num_constants, param_range=(0.2, 2), num_samples=2048,
                        fixed_constants=None, num_bins=200):
    # Generate random parameter samples
    # param_samples = []
    # for _ in range(num_samples):
    #     val = np.random.uniform(param_range[0], param_range[1], size=1)
    #     param_samples.append([val,val])
    param_samples = [
        np.random.uniform(param_range[0], param_range[1], size=num_constants)
        for _ in range(num_samples)
    ]
    print(f"First 5 parameter samples: {param_samples[:5]}")

    # Evaluate function for all parameter samples
    outputs = np.array([expr(domain, params) for params in param_samples])
    print(f"Outputs shape: {outputs.shape}")

    plt.figure(figsize=(7, 5))

    # Define a global y-range for histograms to ensure consistent bins
    y_min, y_max = outputs.min(), outputs.max()
    bin_edges = np.linspace(y_min, y_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    all_percentages = []
    x_coords = []
    y_coords = []

    for i, x in enumerate(domain):
        values = outputs[:, i]
        percentages, _ = np.histogram(values, bins=bin_edges, density=False)
        percentages = (percentages / np.sum(percentages)) * 100
        x_coords.extend([x] * len(bin_centers))
        y_coords.extend(bin_centers)
        all_percentages.extend(percentages)

    print(max(all_percentages))
    print(f"Built density grid with {len(x_coords)} points, y-range fixed from {y_min:.2f} to {y_max:.2f}")

    # Scatter plot colored by local percentage
    log_norm = LogNorm(vmin=np.min(all_percentages)+0.1, vmax=np.max(all_percentages)+2.5, clip=True)
    sc = plt.scatter(x_coords, y_coords, c=np.array(all_percentages), cmap="YlGn", s=5, norm=log_norm)

    if fixed_constants is not None:
        fixed_output = expr(domain, fixed_constants)
        plt.plot(domain, fixed_output, color="red", linewidth=2, label="Standard normal distribution")
        expr_str = "$\\frac{1}{\\sqrt{2\\cdot\\pi\\cdot C}}\\cdot e^{-\\frac{x^2}{2\\cdot C^2}}$"

        # Styling
    plt.ylim(y_min, y_max)
    plt.xlim(1, 5)
    plt.xlabel("Input domain ($x$)", fontsize=12)
    plt.ylabel("Output ($y$)", fontsize=12)
    plt.title(f"\\textbf {{Behavior of Expression}} \\boldmath {expr_str}", fontsize=14)
    formatter = ScalarFormatter()
    cbar = plt.colorbar(sc, label="$P(y|x)[\\%]$", format=formatter, ticks=[0.1, 1, 2, 5])
    cbar.ax.set_yticklabels(['$0$', '$1$', '$2$', '$5$'])


    if fixed_constants is not None:
        plt.legend(frameon=True, loc="upper right", fontsize=12)
    plt.tight_layout()

    print("Plotting complete. Displaying figure...")
    plt.show()


def probability_distribution_at_a_point(expr, num_samples=50000, param_range=(0.2, 2), point=1.0):
    param_samples = []
    for _ in range(num_samples):
        val = np.random.uniform(param_range[0], param_range[1], size=1)
        param_samples.append([val, val])
    outputs = np.array([expr(np.array([[point]]), params) for params in param_samples])
    distribution_data = outputs[:, 0]
    # --------------------------------------------------------

    # Colors from 'YlGn'
    LINE_COLOR = '#006837'
    FILL_COLOR = '#a1d99b'

    plt.figure(figsize=(6, 4))  # Adjust size for horizontal orientation

    # Create the HORIZONTAL, less-smoothed KDE plot
    sns.kdeplot(
        x=distribution_data,  # Plot along the x-axis
        bw_adjust=0.3,  # REDUCE BANDWIDTH for LESS SMOOTHING (default is 1.0)
        fill=True,
        color=LINE_COLOR,
        facecolor=FILL_COLOR,
        linewidth=2,
        alpha=0.8
    )

    # Set the x-limits to match the y-axis of the main plot
    # This means the distribution's range (the y-values) will be consistent
    plt.xlim(-0.05, 0.91)
    plt.ylim(0, 45)  # Ensure the density (y-axis) starts at 0

    # Styling
    plt.title(f"\\textbf {{Distribution at }}\\boldmath $x={point:.1f}$", fontsize=14)
    plt.xlabel("Output ($y$)", fontsize=12)
    plt.ylabel("Density [\%]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # TODO: Clean up this script. Currently plots can be created but some manual tunning is needed
    # The dataset doesn't matter, we're only interested in the SymbolLibrary with all the symbols
    symbols = SRBenchmark.feynman("../data/feynman").create_dataset("I.6.2a").symbols
    # expr1 = "C * X_0 + C * X_0".split(" ")
    # expr2 = "2 * C * X_0".split(" ")
    expr1 = "C * sin ( X_0 )".split(" ")
    expr2 = "sqrt ( C * X_0 )".split(" ")
    # expr0 = "( 1 / sqrt ( 2 * pi * C ) ) * exp ( ( u- X_0 ^2 ) / ( 2 * C ^2 ) )".split(" ")
    expr = expr2
    num_constants = sum([1 for s in expr if s == "C"])
    expr_str = tokens_to_tree(expr, symbols).to_latex(symbols)
    expr_str = "$\\mathit{F}:=" + expr_str[1:]
    print(f"Generated expression: {expr_str}")
    expr = expr_to_executable_function(expr, symbols)
    # expr_str = "$F:=" + expr_str[1:]
    # plot_cdf_comparison(expr1, expr2, 3.0)
    # plot_bed_across_whole_domain(expr1, expr2)
    # probability_distribution_at_a_point(expr, point=1.0)
    # probability_distribution_at_a_point(expr, point=0.0)
    create_density_plot(np.linspace(1, 5, 1000)[:, None], expr, expr_str, num_constants)
