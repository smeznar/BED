import itertools

import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import wasserstein_distance

from SRToolkit.utils.expression_compiler import expr_to_executable_function
# from SRToolkit.utils........

class BED:
    def __init__(self, expressions, x_bounds, const_bounds=(0.2, 5), points_sampled=64, consts_sampled=16, expressions2=None,
                 x=None, randomized=False, cutoff_threshold=1e20, default_distance=1e10, seed=None):
        self.points_sampled = points_sampled
        self.consts_sampled = consts_sampled
        self.cutoff_threshold = cutoff_threshold
        self.default_distance = default_distance

        if seed is None:
            self.seed = np.random.randint(0, 2**32-1)
        else:
            self.seed = seed
        np.random.seed(self.seed)

        self.expressions = expressions
        self.expressions2 = expressions2

        # Prepare data points on which the distance will be calculated
        self.randomized = randomized
        self.x_bounds = x_bounds
        self.const_bounds = const_bounds
        if x is not None:
            self.x = x
            self.y = self.evaluate_expressions_on_points(self.expressions, self.x)
            if self.expressions2 is not None:
                self.y2 = self.evaluate_expressions_on_points(self.expressions2, self.x)
            else:
                self.y2 = None
        elif randomized:
            self.x = None
            self.y = None
            self.y2 = None
        else:
            self.x = self.sample_x()
            self.y = self.evaluate_expressions_on_points(self.expressions, self.x)
            if self.expressions2 is not None:
                self.y2 = self.evaluate_expressions_on_points(self.expressions2, self.x)
            else:
                self.y2 = None

    def sample_x(self):
        interval_length = np.array([ub - lb for (lb, ub) in self.x_bounds])
        lower_bound = np.array([lb for (lb, ub) in self.x_bounds])
        lho = LatinHypercube(len(self.x_bounds), optimization="random-cd", seed=self.seed)
        return lho.random(self.points_sampled) * interval_length + lower_bound

    def evaluate_expressions_on_points(self, expressions, points):
        seed = self.seed
        cbounds_len = self.const_bounds[1] - self.const_bounds[0]
        ys = []
        for expr in expressions:
            seed += 1
            num_constants = expr.count("C")
            if num_constants > 0:
                lho = LatinHypercube(num_constants, seed=seed)
                constants = lho.random(self.consts_sampled) * cbounds_len + self.const_bounds[0]
                exec_expr = expr_to_executable_function(expr)
                values = []
                for i in range(constants.shape[0]):
                    values.append(exec_expr(points, constants[i]))
                y = np.array(values)
            else:
                y = expr_to_executable_function(expr)(points, None)[None, :]

            y[(y > self.cutoff_threshold) | (y < -self.cutoff_threshold)] = np.nan
            y_points = []
            for i in range(y.shape[1]):
                all_points = y[:, i]
                y_points.append(all_points[~np.isnan(all_points)])
            ys.append(y_points)
        return ys

    def bed(self, y1, y2):
        cube_distance = []
        for p1, p2 in zip(y1, y2):
            if len(p1)==0 and len(p2)==0:
                cube_distance.append(0.0)
            elif len(p1)==0 or len(p2)==0:
                cube_distance.append(self.default_distance)
            else:
                cube_distance.append(wasserstein_distance(p1, p2))

        return np.mean(cube_distance)

    def calculate_distances(self):
        # Case where we try to calculate the distance between all expressions
        if self.expressions2 is None:
            # We only have two expression
            if len(self.expressions) == 2:
                if self.randomized:
                    x = self.sample_x()
                    y = self.evaluate_expressions_on_points(self.expressions, x)
                    return self.bed(y[0], y[1])
                else:
                    return self.bed(self.y[0], self.y[1])

            # We have more expressions
            dm = np.zeros((len(self.expressions), len(self.expressions)))
            for i, j in itertools.combinations(range(len(self.expressions)), 2):
                if self.randomized:
                    self.seed += 1
                    x = self.sample_x()
                    y = self.evaluate_expressions_on_points([self.expressions[i], self.expressions[j]], x)
                    dm[i, j] = dm[j, i] = self.bed(y[0], y[1])
                else:
                    dm[i, j] = dm[j, i] = self.bed(self.y[i], self.y[j])

        # Case where we try to calculate the distance between two sets of expressions
        else:
            # We only have two expressions
            if len(self.expressions) == 1 and len(self.expressions2) == 1:
                if self.randomized:
                    x = self.sample_x()
                    y = self.evaluate_expressions_on_points([self.expressions[0], self.expressions2[0]], x)
                    return self.bed(y[0], y[1])
                else:
                    return self.bed(self.y[0], self.y2[0])

            # We have more expressions
            dm = np.zeros((len(self.expressions), len(self.expressions2)))
            for i in range(len(self.expressions)):
                for j in range(len(self.expressions2)):
                    if self.randomized:
                        self.seed += 1
                        x = self.sample_x()
                        y = self.evaluate_expressions_on_points([self.expressions[i], self.expressions2[j]], x)
                        dm[i, j] = self.bed(y[0], y[1])
                    else:
                        dm[i, j] = self.bed(self.y[i], self.y2[j])

        return dm
