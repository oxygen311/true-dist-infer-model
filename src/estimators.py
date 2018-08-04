import numpy as np
from bisect import bisect
from scipy import optimize
from src.dirichlet_graph import DirichletDnkGraph
import random
from scipy.special import hyp2f1
import math
import json


# predict value of x or y relying on certain data
class DataValuer:
    xs = np.arange(0, 3, 0.01)

    def __init__(self, data, data_function):
        actual_func = lambda p: data_function(*p)
        self.ys = [(np.mean([actual_func(d) for d in ds])) for ds in data]

    def predict_arg(self, y):
        i = bisect(self.ys, y)
        return self.xs[min(i, len(self.xs) - 1)]

    def predict_val(self, x):
        i = bisect(self.xs, x)
        return self.ys[min(i, len(self.xs) - 1)]


get_b_from_cycles = lambda c: sum([int(k) * v for k, v in c.items()])
get_d_from_cycles = lambda c: sum([(int(k) - 1) * v for k, v in c.items()])


class DataEstimator:
    def __init__(self, data):
        self.b_over_n_valuer = DataValuer(data, lambda n, _, c: (n - c['1']) / n)
        self.d_over_b_valuer = DataValuer(data, lambda n, _, c: (n - sum(c.values())) / (max(1e-6, n - c['1'])))

    def predict(self, cycles):
        d = get_d_from_cycles(cycles)
        b = get_b_from_cycles(cycles)
        x = self.d_over_b_valuer.predict_arg(d / b)
        print(x)

        b_over_n = self.b_over_n_valuer.predict_val(x)
        n = b / b_over_n

        # n = b / (1 - math.exp(-x))
        return n, max(d, n * x / 2)
        # return n * x / 2


class TannierEstimator:
    @staticmethod
    def predict(cycles):
        def e_c1(n, k):
            return n * sum(
                np.prod([(- 2 * k / (n + u)) for u in range(0, l)])
                for l in range(50))

        def dc1_dn(n, k):
            return e_c1(n, k) / n - n * sum(np.prod([(- 2 * k / (n + u))
                                                     for u in range(l)])
                                            * sum(1 / (n + u) for u in range(l))
                                            for l in range(50))

        def dc1_dk(n, k):
            return n * sum(
                l * np.prod([(- 2 * k / (n + u)) for u in range(l)]) / k
                for l in range(50))

        def e_c2(n, k):
            return k * n * n * sum(
                (l + 1) * (m + 1) /
                (4 * (k - 1) ** 2 * np.prod([(n + u) / (- 2 * (k - 1)) for u in range(l + m + 2)]))
                for l, m in np.ndindex((50, 50)))

        def dc2_dn(n, k):
            return e_c2(n, k) / n * 2 - k * n * n * sum(
                (l + 1) * (m + 1) /
                (4 * (k - 1) ** 2 * np.prod([(n + u) / (- 2 * (k - 1)) for u in range(l + m + 2)])) *
                sum(1 / (n + u) for u in range(l + m + 2))
                for l, m in np.ndindex((50, 50)))

        def dc2_dk(n, k):
            return e_c2(n, k) / k + k * n * n * sum(
                (l + 1) * (m + 1) * (l + m) /
                (4 * (k - 1) ** 3 * np.prod([(n + u) / (- 2 * (k - 1)) for u in range(l + m + 2)]))
                for l, m in np.ndindex((50, 50)))

        def create_fun(real_b, real_c2):
            def fun(x):
                return [x[0] - e_c1(x[0], x[1]) - real_b,
                        e_c2(x[0], x[1]) - real_c2]

            return fun

        def jac(x):
            return np.array([[1 - dc1_dn(x[0], x[1]),
                              - dc1_dk(x[0], x[1])],
                             [dc2_dn(x[0], x[1]),
                              dc2_dk(x[0], x[1])]])

        b = get_b_from_cycles(cycles)
        d = get_d_from_cycles(cycles)
        c2 = cycles.get('2', 0)

        fun = create_fun(b, c2)
        prediction = optimize.root(fun, np.array([3 * b, d]), jac=jac, method='hybr')

        # print(prediction)
        return prediction.x[0], prediction.x[1]


    def predict_k(self, cycles):
        return self.predict(cycles)[1]


class DBFunctionEstimator:
    def d_over_n(self, x):
        raise NotImplementedError('subclasses must override d_over_n!')

    def b_over_n(self, x):
        raise NotImplementedError('subclasses must override b_over_n!')

    def predict(self, cycles):
        d_over_b = lambda r: lambda x: self.d_over_n(x) / self.b_over_n(x) - r

        d = get_d_from_cycles(cycles)
        b = get_b_from_cycles(cycles)

        x = optimize.bisect(d_over_b(d / b), 1e-6, 3, xtol=1e-4)

        b_n = self.b_over_n(x)
        n = b / b_n
        return n, n * x / 2


class DirichletEstimator(DBFunctionEstimator):
    def __init__(self):
        # super.__init__()
        self.err = json.loads(open("data/dirichlet_d_error.txt", 'r').read())

    def d_over_n(self, x):
        return 1 - (1 + x) ** 2 * (hyp2f1(-2 / 3, -1 / 3, 1 / 2, 27 * x / (4 * (1 + x) ** 3)) - 1) / (3 * x)

    def b_over_n(self, x):
        return x / (1 + x)

    def predict(self, cycles):
        def d_over_n_with_n(n):
            def ret_func(x):
                ret = 1 - (1 + x) ** 2 * (hyp2f1(-2 / 3, -1 / 3, 1 / 2, 27 * x / (4 * (1 + x) ** 3)) - 1) / (3 * x)
                ret -= self.err[int(x * 100)] / math.sqrt(n) / 2
                return ret

            return ret_func

        d_over_b = lambda r: lambda x: self.d_over_n(x) / self.b_over_n(x) - r

        d = get_d_from_cycles(cycles)
        b = get_b_from_cycles(cycles)

        x = optimize.bisect(d_over_b(d / b), 1e-6, 3, xtol=1e-4)
        n = b / self.b_over_n(x)

        new_d_over_n_func = d_over_n_with_n(n)
        new_d_over_b_func = lambda r: lambda x: new_d_over_n_func(x) / self.b_over_n(x) - r

        x2 = optimize.bisect(new_d_over_b_func(d / b), 1e-6, 3, xtol=1e-4)

        # return n * (x + x2) / 2 / 2
        return n, n * x / 2


class UniformEstimator(DBFunctionEstimator):
    def d_over_n(self, x):
        c_m = lambda m: lambda x: math.exp(- m * x) * (x ** (m - 1)) * (m ** (m - 2)) / math.factorial(m)
        return 1 - sum(c_m(m)(x) for m in range(1, 100))

    def b_over_n(self, x):
        return 1 - math.exp(- x)


class FirstCmsDirEstimator:
    def __init__(self, max_m):
        self.max_m = max_m + 1
        self.cm_err = [1] + [1] + [1] + [1] + [1] * 30
        # self.cm_err = json.loads(open("data/cms_error_abs.txt", 'r').read())
        # self.cm_err = json.loads(open("data/cms_error_squared.txt", 'r').read())

    def predict(self, cycles):
        def e_c_m(n, k, m):
            x = 2 * k / n
            return n * math.factorial(3 * m - 3) * x ** (m - 1) \
                   / (math.factorial(m) * math.factorial(2 * m - 1) * (x + 1) ** (3 * m - 2))

        def dc_m_dk(n, k, m):
            return - 2 ** m * (k / n) ** m * (2 * k + n) * (k * (-2 + 4 * m) + n - m * n) * math.factorial(
                3 * m - 3) \
                   / ((1 + 2 * k / n) ** (3 * m) * k ** 2 * math.factorial(m - 1) * math.factorial(2 * m))

        def dc_m_dn(n, k, m):
            return 2 ** (m - 1) * (k / n) ** (m - 1) * (2 * k + n) * (4 * k * m - n * (m - 2)) * math.factorial(
                3 * m - 3) \
                   / ((1 + 2 * k / n) ** (3 * m) * n ** 2 * math.factorial(m) * math.factorial(2 * m - 1))

        def create_fun(real_b, cycles):
            # x[0] - n, x[1] - k
            def fun(x):
                return [x[0] - e_c_m(x[0], x[1], 1) - real_b,
                        sum(
                            ((e_c_m(x[0], x[1], m) - cycles[str(m)]) if str(m) in cycles else 0) / self.cm_err[m]
                            for m in range(2, self. max_m)
                        )]

            return fun

        def jac(x):
            return np.array([
                [1 - dc_m_dn(x[0], x[1], 1),
                 - dc_m_dk(x[0], x[1], 1)],
                [sum(dc_m_dn(x[0], x[1], m) / self.cm_err[m] for m in range(2, self.max_m)),
                 sum(dc_m_dk(x[0], x[1], m) / self.cm_err[m] for m in range(2, self.max_m))]
            ])

        b = get_b_from_cycles(cycles)
        d = get_d_from_cycles(cycles)

        fun = create_fun(b, cycles)
        prediction = optimize.root(fun, np.array([3 * b, d]), jac=jac, method='hybr')

        # print(prediction)
        return prediction.x[0], prediction.x[1]

    def predict_k(self, cycles):
        return self.predict(cycles)[1]


without_c1 = lambda g: {str(k): v for k, v in g.items() if k != 1}

if __name__ == "__main__":
    # dirichlet_data = json.loads(open("data/dirichlet_data_randomN_200.txt", 'r').read())
    # dataEst = DataEstimator(dirichlet_data)
    tanEst = TannierEstimator()
    dirEst = DirichletEstimator()
    # testDirEst = TestDirEstimator(2)
    random.seed(42)
    np.random.seed(42)

    n = 1000
    k = 500

    g = DirichletDnkGraph(n)
    for k in range(k):
        g.do_k_break()

    print("breaks done")

    cycles = without_c1(g.count_cycles())
    # print(testDirEst.predict_k(cycles))
    # print(tanEst.predict_k(cycles))
    # print(dataEst.predict_k(cycles))
