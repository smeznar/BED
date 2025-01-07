

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination


class PymooProblem(Problem):
    def __init__(self, model, constants, evaluator, estimation_settings, lower=-5, upper=5, default_error=1e10):
        xl = np.full(constants, lower)
        xu = np.full(constants, upper)
        super().__init__(n_var=constants, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.model = model
        self.evaluator = evaluator
        self.default_error = default_error
        self.estimation_settings = estimation_settings

    def _evaluate(self, x, out, *args, **kwargs):
        rmse = self.evaluator.get_error(self.model, x.tolist())
        out["F"] = np.array([error for error in rmse])


def DE_pymoo(model, constants, evaluator, verbose=False, seed=None, **estimation_settings):
    pymoo_problem = PymooProblem(model, constants, evaluator, estimation_settings)
    strategy = "DE/best/1/bin"
    algorithm = DE(
        pop_size=20,
        sampling=LHS(),
        variant=strategy,
        CR=0.7,
        dither="vector",
        jitter=False
    )

    termination = DefaultSingleObjectiveTermination(
        xtol=0.1,
        cvtol=1e-6,
        ftol=1e-6,
        period=20,
        n_max_gen=1000,
    )

    output = minimize(pymoo_problem,
                      algorithm,
                      termination,
                      seed=seed,
                      verbose=verbose,
                      save_history=False)

    return output.X, output.F


class RustEval:
    variable_names = 'ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'

    def __init__(self, data, default_value=1e10, no_target=False, seed=None, verbose=False):
        self.seed = seed
        self.data = data
        self.default_value = default_value
        d = data.T
        columns = []
        names = []
        num_columns = d.shape[0] if no_target else d.shape[0] - 1
        for i in range(num_columns):
            columns.append([float(v) for v in d[i]])
            names.append(RustEval.variable_names[i])
        if no_target:
            target = [0.0 for _ in d[-1]]
        else:
            target = [float(v) for v in d[-1]]

        self.no_target = no_target
        self.verbose = verbose
        self.evaluator = Evaluator(columns, names, target)

    def evaluate(self, expression, constants=None):
        if constants is None:
            constants = [[]]
        try:
            return self.evaluator.eval_expr(expression, constants, False)
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def get_error(self, expression, constants=None):
        if self.no_target:
            raise Exception('Initialize RustEval without the "no_target" flag to use the "get_error" function')
        else:
            if constants is None:
                constants = [[]]
            try:
                return self.evaluator.get_rmse(expression, constants, self.default_value, self.verbose)
            except Exception as e:
                if self.verbose:
                    print(e)
                return None

    def fit_and_evaluate(self, expr):
        if self.no_target:
            raise Exception('Initialize RustEval without the "no_target" flag to use the "get_error" function')
        else:
            num_constants = sum([1 for t in expr if t == "C"])
            if num_constants > 0:
                x, rmse = DE_pymoo(expr, num_constants, self, seed=self.seed)
                return rmse, x
            else:
                error = self.get_error(expr)
                if error is not None:
                    return error[0], []
                else:
                    return None, []


if __name__ == '__main__':
    data = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]).T
    rev = RustEval(data)
    print(rev.fit_and_evaluate(["A", "1", "/", "C", "+"]))
# names = ["X", "Y"]
# evaluator = Evaluator(data, names)
# print(evaluator.eval_expr(["X", "Y", "+"]))