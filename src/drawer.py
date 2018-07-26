import matplotlib.pyplot as plt
import math
import numpy as np
import json
from bisect import bisect_left
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator
# from estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator
# from src.regression_estimators import SGDRegressorXEstimator, collect_cm_n_from_cycles, collect_cm_n_data
from time import time
from scipy.special import hyp2f1

# plotly
classic_file = "data/classic_data_N2000_2000.txt"
dirichlet_file = "data/dirichlet_data_randomN_2000.txt"
lite_classic_file = "data/classic_data_N2000_200.txt"
lite_dirichlet_file = "data/dirichlet_data_randomN_200.txt"
lite_dirichlet_file_smallN = "data/dirichlet_data_random_smallN_200.txt"
slite_dirichlet_file = "data/dirichlet_data_randomN_20.txt"

cycles_breakdowns_lite_file = "data/dirichlet_breakdown_of_a_cycle20.txt"
cycles_breakdowns_file = "data/dirichlet_breakdown_of_a_cycle200.txt"

test_file = "data/dirichlet_1000_20.txt"


class Drawer:
    colors = ['#2c8298', '#e0552e', '#90af3d', '#df9b34', '#8779b1', '#c36e27', '#609ec6', '#edb126']
    current_color = 0
    xs = np.arange(0, 3, 0.01)

    def __init__(self, x_min=0.0, x_max=3.0, usetex=False):
        self.x_min = max(x_min, 0.0)
        self.x_max = min(x_max, 3.0)
        self.min_index = bisect_left(self.xs, x_min)
        self.max_index = bisect_left(self.xs, x_max)
        self.draw_xs = self.xs[self.min_index:self.max_index]

        if usetex:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=12)
        plt.grid(True)

    def increase_color(self):
        self.current_color = (self.current_color + 1) % len(self.colors)

    def draw_data(self, data, data_function, with_interval=True, label=None, linewidth = 1.0):
        actual_func = lambda p: data_function(*p)
        ys = []
        for ds in data[self.min_index:self.max_index]:
            tmp = list([actual_func(d) for d in ds])[0:100]
            # print(ds[10:11])
            # print(tmp)
            ys.append(sorted(tmp))

        mean = list(map(lambda y: np.mean(y), ys))
        # open("data/dirichlet_d_error.txt", 'w').write(json.dumps(mean))
        plt.plot(self.draw_xs, mean, color=self.colors[self.current_color], label=label, linewidth=linewidth)
        print("sum of ys ** 2:", sum(map(lambda x: x ** 2, mean)))

        if with_interval:
            for interval in np.arange(0.5, 1, 0.05):
                low = list(map(lambda y: y[-int(len(y) * interval)], ys))
                upper = list(map(lambda y: y[int(len(y) * interval)], ys))
                plt.fill_between(self.draw_xs, low, upper, color=self.colors[self.current_color], alpha=0.125)

        self.increase_color()

    def draw_function(self, func, label=None, linewidth=1.0, linestyle='-'):
        vectorized = np.vectorize(func)
        plt.plot(self.draw_xs, vectorized(self.draw_xs), color=self.colors[self.current_color], label=label,
                 linewidth=linewidth, linestyle=linestyle)
        self.increase_color()

    def draw_boxplot_data(self, data, data_function):
        actual_func = lambda p: data_function(*p)
        ys, all_ys = [], []
        start_time = time()
        for ds in data[self.min_index:self.max_index:10]:
            tmp = list(filter(lambda el: el, [actual_func(d) for d in ds])) or [0]
            ys.append(sorted(tmp))
        # ys = json.loads(open("data/tannier_est_error.txt", 'r').read())

        print(time() - start_time, "seconds")
        print("mean squared error", np.mean(list(map(lambda xs: list(map(lambda x: x ** 2, xs)), ys))))

        real_xs = list(map(lambda x: round(x, 1), self.xs[self.min_index:self.max_index:10]))

        # dmp = {}
        # for x, y in zip(real_xs, ys):
        #     dmp[x] = y
        # open("data/est.txt", 'w').write(json.dumps(dmp))

        plt.boxplot(ys, labels=real_xs, whis=[50, 50], showfliers=False)
        # plt.ylim([-0.11, 0.11])

    @staticmethod
    def show_and_save(title=None, xlabel=None, ylabel=None, legend_loc=None):
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel, fontsize=14)
        if ylabel: plt.ylabel(ylabel)

        if legend_loc: plt.legend(loc=legend_loc)
        # plt.savefig('cms.pdf')
        plt.show()


# delete c1
without_c1 = lambda g: {k: v for k, v in g.items() if k != '1'}

# data functions
d_distance = lambda n, _, c: (n - sum(c.values())) / n
b_distance = lambda n, _, c: (n - c['1']) / n
d_over_b = lambda n, _, c: d_distance(n, _, c) / b_distance(n, _, c)
b_over_d = lambda n, _, c: b_distance(n, _, c) / d_distance(n, _, c)
data_c_m = lambda m: lambda n, _, c: c[str(m)] / n if str(m) in c.keys() else 0
est_error = lambda est: lambda _, k, c: (est.predict_k(without_c1(c)) - k) / k

# x functions
line = lambda x: x / 2
c_m = lambda m: lambda x: math.exp(- m * x) * (x ** (m - 1)) * (m ** (m - 2)) / math.factorial(m)
c_m_dir = lambda m: lambda x: math.factorial(3 * m - 3) * x ** (m - 1) \
                              / (math.factorial(m) * math.factorial(2 * m - 1) * (x + 1) ** (3 * m - 2))

d_over_n_dir = lambda x: 1 - (1 + x) ** 2 * (hyp2f1(-2 / 3, -1 / 3, 1 / 2, 27 * x / (4 * (1 + x) ** 3)) - 1) / (3 * x)
b_over_n_dir = lambda x: x / (1 + x)
d_over_b_dir = lambda x: d_over_n_dir(x) / b_over_n_dir(x)
b_over_d_dir = lambda x: b_over_n_dir(x) / d_over_n_dir(x)

if __name__ == "__main__":
    # classic_data = json.loads(open(classic_file, 'r').read())
    lite_classic_data = json.loads(open(lite_classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)

    # dirichlet_data = json.loads(open(dirichlet_file, 'r').read())
    lite_dirichlet_data = json.loads(open(lite_dirichlet_file, 'r').read())
    # slite_dirichlet_data = json.loads(open(slite_dirichlet_file, 'r').read())
    # slite_dirichlet_data_round = json.loads(open("data/dirichlet_data_random_round_10.txt", 'r').read())
    # slite_classic_data_round = json.loads(open("data/classic_data_random_round_10.txt", 'r').read())
    # lite_dirichlet_data_smallN = json.loads(open(lite_dirichlet_file_smallN, 'r').read())
    test_data = json.loads(open(test_file, 'r').read())

    # data_dirichlet_est = DataEstimator(dirichlet_data)
    # data_classic_est = DataEstimator(classic_data)
    dirichlet_est = DirichletEstimator()
    test_dirichlet_est = FirstCmsDirEstimator(6)
    uniform_est = UniformEstimator()
    tannier_est = TannierEstimator()
    print("<read>")

    # drawer = Drawer()
    # drawer.draw_function(d_over_b_dir, "d/b analytical")
    # drawer.draw_data(dirichlet_data, d_over_b, False, "d/b empirical")
    drawer = Drawer(0, 2)
    # drawer = Drawer(0.3, 1)

    # drawer.draw_data(dirichlet_data, data_c_m(2))

    # drawer.draw_boxplot_data(lite_dirichlet_data, est_error(test_dirichlet_est))

    # drawer.draw_data(lite_dirichlet_data, d_over_b, False, "Empirical value of $d/b$")
    # drawer.draw_function(d_over_b_dir, "Analytical value of $d/b$", linestyle='--')
    #
    # drawer.draw_data(lite_dirichlet_data, d_distance, False, "Empirical value of $d/n$")
    # drawer.draw_function(d_over_n_dir, "Analytical value of $d/n$", linestyle='--')
    #
    # drawer.draw_data(lite_dirichlet_data, b_distance, False, "Empirical value of $b/n$")
    # drawer.draw_function(b_over_n_dir, "Analytical value of $b/n$", linestyle='--')
    #
    # drawer.draw_data(lite_classic_data, d_distance, False, "Empirical value of $d/n$")
    # drawer.draw_function(lambda x: 1 - sum(c_m(m)(x) for m in range(1, 30)), "Analytical value of $d/n$", linestyle='--')

    # drawer.draw_data(dirichlet_data, data_c_m(2), False, "Empirical value of $c_2/n$", linewidth=0.8)
    # drawer.draw_function(c_m_dir(2), "Analytical value of $c_2/n$", linestyle='--', linewidth=0.8)
    #
    # drawer.draw_data(dirichlet_data, data_c_m(3), False, "Empirical value of $c_3/n$", linewidth=0.8)
    # drawer.draw_function(c_m_dir(3), "Analytical value of $c_3/n$", linestyle='--', linewidth=0.8)
    #
    # drawer.draw_data(dirichlet_data, data_c_m(4), False, "Empirical value of $c_4/n$", linewidth=0.8)
    # drawer.draw_function(c_m_dir(4), "Analytical value of $c_4/n$", linestyle='--', linewidth=0.8)

    # drawer.draw_function(line, "real distance")
    # drawer.draw_data(test_data, d_distance, False, "minimal distance")

    drawer.draw_data(test_data, data_c_m(2), True, "Empirical value of $c_2/n$", linewidth=0.8)
    # drawer.draw_function(c_m_dir(2), "Analytical value of $c_2/n$", linestyle='--', linewidth=0.8)

    plt.subplots_adjust(bottom=0.11, top=0.98, right=0.99, left=0.07)
    drawer.show_and_save(xlabel=r'\gamma', legend_loc=1)
