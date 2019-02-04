from sklearn import linear_model, ensemble, neighbors
import json
import random
import numpy as np
from src.dirichlet_bp_graph import DirichletBPGraph
from src.estimators import get_b_from_cycles, get_d_from_cycles


def collect_cm_n_data(data, split_start=0, split_end=1, cms=500, with_ans=True):
    ln = len(data[0])
    str = int(ln * split_start)
    end = int(ln * split_end)

    xs = []
    ys = []
    for ds in data:
        for d in ds[str:end]:
            n, k, c = d

            x = collect_cm_n_from_cycles(c, cms)
            xs.append(x)

            if with_ans:
                ys.append(k)

    return xs, ys


def collect_cm_n_from_cycles(cycles, cms=500):
    b = get_b_from_cycles(cycles)
    d = get_d_from_cycles(cycles)
    # return [cycles["%d" % m] / b if "%d" % m in cycles else 0 for m in range(2, cms)]
    return [b, d]


class SGDRegressorXEstimator:
    def __init__(self, x, y):
        # self.reg = ensemble.AdaBoostRegressor()
        self.reg = ensemble.RandomForestRegressor()
        # self.reg = ensemble.GradientBoostingRegressor()
        self.reg.fit(x, y)

    def predict_k(self, c):
        xs = [collect_cm_n_from_cycles(c)]
        # print(xs)
        return self.reg.predict(xs)[0]


without_c1 = lambda g: {str(k): v for k, v in g.items() if k != 1}


def main():
    dirichlet_file = "data/dirichlet_data_randomN_2000.txt"
    lite_dirichlet_file = "data/dirichlet_data_randomN_200.txt"

    lite_dirichlet_data = json.loads(open(lite_dirichlet_file, 'r').read())
    dirichlet_data = json.loads(open(dirichlet_file, 'r').read())
    print("<read>")

    x, y = collect_cm_n_data(dirichlet_data)
    # x, y = collect_cm_n_data(lite_dirichlet_data)
    print("<data collected>")
    est = SGDRegressorXEstimator(x, y)
    print("<regressor fitted>")

    random.seed(42)
    np.random.seed(42)

    n = 3000
    k = 1000

    g = DirichletBPGraph(n)
    for k in range(k):
        g.do_k_break()

    print("<breaks done>")
    cycles = without_c1(g.count_cycles())
    print(est.predict_k(cycles))


main()
