import numpy as np
from scipy.stats.qmc import LatinHypercube

from utils import infix_to_postfix
from evaluation import RustEval


class BED:
    def __init__(self, expressions, x_bounds, const_bounds, points_sampled=64, consts_sampled=16, x=None, infix=False,
                 randomized=False, cutoff_threshold=1e20, seed=None):
        self.points_sampled = points_sampled
        self.consts_sampled = consts_sampled
        self.cutoff_threshold = cutoff_threshold
        if seed is None:
            self.seed = np.random.randint(-10000000, 10000000)
        else:
            self.seed = seed
        np.random.seed(self.seed)

        # Transform expressions to the postfix notation, if they are in the infix notation
        if infix:
            expressions = infix_to_postfix(expressions)

        # Prepare data points on which the distance will be calculated
        self.randomized = randomized
        self.x_bounds = x_bounds
        self.const_bounds = const_bounds
        if x is not None:
            self.x = x
        elif randomized:
            self.x = None
        else:
            self.x = self.sample_x()

    def sample_x(self):
        interval_length = np.array([ub - lb for (lb, ub) in self.x_bounds])
        lower_bound = np.array([lb for (lb, ub) in self.x_bounds])
        lho = LatinHypercube(len(self.x_bounds), optimization="random-cd", seed=self.seed)
        return lho.random(self.points_sampled) * interval_length + lower_bound

    def evaluate_expressions_on_points(self, expressions, points):
        seed = self.seed
        cbounds_len = self.const_bounds[1] - self.const_bounds[0]
        ys = []
        rev = RustEval(points, no_target=True)
        for expr in expressions:
            seed += 1
            num_constants = expr.count("C")
            if num_constants > 0:
                lho = LatinHypercube(num_constants, seed=seed)
                constants = lho.random(self.consts_sampled) * cbounds_len + self.const_bounds[0]
            else:
                constants = None
            y = np.array(rev.evaluate(expr, constants)).T.tolist()
            y = [[v for v in point if np.isfinite(v) and v < self.cutoff_threshold] for point in y]
            ys.append(y)
        return ys

    def calculate_distances(self):
        if self.randomized:
            self.seed += 1
            x = self.sample_x()
        else:
            x = self.x


