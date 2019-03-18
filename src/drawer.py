import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import json
from bisect import bisect_left
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, GammaEstimator, CorrectedGammaEstimator
from src.cmrecur import cm4, b_gamma, d_gamma, d_gamma_b
# from estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator
# from src.regression_estimators import SGDRegressorXEstimator, collect_cm_n_from_cycles, collect_cm_n_data
from time import time
from scipy.special import hyp2f1, gamma
import seaborn as sns
import matplotlib
from scipy.signal import savgol_filter, medfilt
from collections import defaultdict

# plotly
classic_file = "sim_data/classic_data_N2000_2000.txt"
dirichlet_file = "sim_data/dirichlet_data_randomN_2000.txt"
lite_classic_file = "sim_data/classic_data_N2000_200.txt"
lite_dirichlet_file = "sim_data/dirichlet_data_randomN_200.txt"
lite_genome_file = "sim_data/genome_data_N1000_200_%dс.txt"
slite_genome_file_100 = "sim_data/genome_data_N1000_20_100с.txt"
lite_dirichlet_file_smallN = "sim_data/dirichlet_data_random_smallN_200.txt"
slite_dirichlet_file = "sim_data/dirichlet_data_randomN_20.txt"
lite_gamma09_file = "sim_data/gamma09_data_N1000_200.txt"
lite_gamma03_file = "sim_data/gamma03_data_N1000_200.txt"
lite_gamma_file = "sim_data/gamma0%d_data_N1000_200.txt"
db_genome_file = "sim_data/dir_genome_data_N1000_200_%dс.txt"
db_genome_file2 = "sim_data/dir_genome_data_N1000_104_%dс.txt"
like_real_file = "sim_data/lin_0303_chr20_data_N1000_100.txt"

cycles_breakdowns_lite_file = "sim_data/dirichlet_breakdown_of_a_cycle20.txt"
cycles_breakdowns_file = "sim_data/dirichlet_breakdown_of_a_cycle200.txt"

test_file = "sim_data/dirichlet_1000_20.txt"


class Drawer:
    # colors = ['#2c8298', '#e0552e', '#90af3d', '#df9b34', '#8779b1', '#c36e27', '#609ec6', '#edb126']
    colors = ['#2e4d84', '#23781f', '#df9b34', '#8779b1', '#c36e27', '#609ec6', '#edb126']
    current_color = 0
    xs = np.arange(0, 3, 0.01)

    def __init__(self, x_min=0.0, x_max=3.0, usetex=False):
        self.x_min = max(x_min, 0.0)
        self.x_max = min(x_max, 3.0)
        self.min_index = bisect_left(self.xs, x_min)
        self.max_index = bisect_left(self.xs, x_max)
        self.draw_xs = self.xs[self.min_index:self.max_index]
        # mpl.rcParams['figure.dpi'] = 300

        if usetex:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=12)
        plt.grid(True)

    def increase_color(self):
        self.current_color = (self.current_color + 1) % len(self.colors)

    def reset_color(self):
        self.current_color = 0

    def draw_data(self, data, data_function, with_interval=False, label=None, linewidth=1.0, inc_color=True):
        actual_func = lambda p: data_function(*p)
        ys = []
        for ds in data[self.min_index:self.max_index]:
            tmp = list([actual_func(d) for d in ds])  # [10:11]
            # print(ds[10:11])
            # print(tmp)
            ys.append(sorted(tmp))

        mean = list(map(lambda y: np.mean(y), ys))
        # print("\n".join(map(lambda x: ("%.10f" % x).replace(".", ","), mean)))
        # open("data/dirichlet_d_error.txt", 'w').write(json.dumps(mean))
        plt.plot(self.draw_xs, mean, color=self.colors[self.current_color], label=label, linewidth=linewidth)
        print("sum of ys ** 2:", sum(map(lambda x: x ** 2, mean)))

        if with_interval:
            for interval in np.arange(0.5, 1, 0.05):
                low = list(map(lambda y: y[-int(len(y) * interval)], ys))
                upper = list(map(lambda y: y[int(len(y) * interval)], ys))
                plt.fill_between(self.draw_xs, low, upper, color=self.colors[self.current_color], alpha=0.125)
        if inc_color:
            self.increase_color()

    def draw_interval(self, data, data_function):
        actual_func = lambda p: data_function(*p)
        ys = []
        for ds in data[self.min_index:self.max_index]:
            tmp = list([actual_func(d) for d in ds])
            ys.append(sorted(tmp))

        low = list(map(lambda y: y[int(len(y) * 0.025)], ys))
        lowhat = savgol_filter(low, 31, 3)

        upper = list(map(lambda y: y[int(len(y) * 0.975)], ys))
        upperhat = savgol_filter(upper, 31, 3)
        plt.fill_between(self.draw_xs, lowhat, upperhat, color=self.colors[self.current_color], alpha=0.35)

    def draw_dots(self, xs, ys, legend, marker):
        plt.scatter(xs, ys, color=self.colors[self.current_color], label=legend, marker=marker, s=25)

    def draw_function(self, func, label=None, linewidth=1.0, linestyle='-'):
        plt.plot(self.draw_xs, list(map(func, self.draw_xs)), color=self.colors[self.current_color], label=label,
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
        print("mean abs error", np.mean(list(map(lambda xs: list(map(lambda x: abs(x), xs)), ys))) * 100)

        real_xs = list(map(lambda x: round(x, 1), self.xs[self.min_index:self.max_index:10]))

        # plt.boxplot(ys, labels=real_xs, whis=[5, 95], showfliers=False)
        ax = sns.boxplot(x=real_xs, y=ys, whis=[5, 95], showfliers=False, color="#ebeae7", linewidth=1.2)
        # plt.ylim([-0.11, 0.11])

    @staticmethod
    def show_and_save(title=None, xlabel=None, ylabel=None, legend_loc=None):
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel, fontsize=14)
        if ylabel: plt.ylabel(ylabel)

        if legend_loc: plt.legend(loc=legend_loc, frameon=True, prop={'size': 12})
        # plt.savefig('test_03.svg')
        plt.show()


# delete c1
without_c1 = lambda g: {k: v for k, v in g.items() if k != '1'}

# data functions
d_distance = lambda n, _, c: (n - sum(c.values())) / n
b_distance = lambda n, _, c: (n - c['1']) / n
d_over_b = lambda n, _, c: d_distance(n, _, c) / b_distance(n, _, c)
b_over_d = lambda n, _, c: b_distance(n, _, c) / d_distance(n, _, c)
# data_c_m = lambda m: lambda n, _, c: c[str(m)] / n if str(m) in c.keys() else 0
data_c_m = lambda m: lambda n, k, c: c[str(m)] / n if str(m) in c.keys() else 0
est_error = lambda est: lambda _, k, c: (est.predict_k(without_c1(c)) - k) / k
db_est_error = lambda est: lambda _, k, d, b: (est.predict_k_by_db(d, b) - k) / k


def convert_dict(c):
    ans = defaultdict(lambda: 0)
    for k, v in c.items():
        ans[str(math.ceil(int(k) / 2))] += v
    return ans


gdata_c_m = lambda m: lambda n, k, c: data_c_m(m)(n, k, convert_dict(c))
gd_distance = lambda n, k, c: d_distance(n, k, convert_dict(c))
gb_distance = lambda n, k, c: b_distance(n, k, convert_dict(c))
gd_over_b = lambda n, k, c: d_over_b(n, k, convert_dict(c))
gest_error = lambda est: lambda n, k, c: est_error(est)(n, k, convert_dict(c))

# def convert_func(f):

# x functions
line = lambda x: x / 2
c_m = lambda m: lambda x: math.exp(- m * x) * (x ** (m - 1)) * (m ** (m - 2)) / math.factorial(m)
c_m_dir = lambda m: lambda x: math.factorial(3 * m - 3) * x ** (m - 1) \
                              / (math.factorial(m) * math.factorial(2 * m - 1) * (x + 1) ** (3 * m - 2))

d_over_n_dir = lambda x: 1 - (1 + x) ** 2 * (hyp2f1(-2 / 3, -1 / 3, 1 / 2, 27 * x / (4 * (1 + x) ** 3)) - 1) / (3 * x)
b_over_n_dir = lambda x: x / (1 + x)
d_over_b_dir = lambda x: d_over_n_dir(x) / b_over_n_dir(x)
b_over_d_dir = lambda x: b_over_n_dir(x) / d_over_n_dir(x)

# t^m*Г(mt+2m-2)/Г(mt+m)/Г(m+1) * x^(m-1)/(1+x)^(3m-2)
c_m_gamma = lambda t, m: lambda x: x ** (m - 1) * (t ** (t + 1)) ** m / (x + t) ** (m * t + 2 * m - 2) * gamma(
    m * t + 2 * m - 2) / gamma(m * t + m) / gamma(m + 1)
b_over_n = lambda t: lambda x: 1 - c_m_gamma(t, 1)(x)
cms_sum = lambda t, cms: lambda x: np.sum([c_m_gamma(t, m)(x) * m for m in range(2, cms)]) / b_over_n(t)(x)
get_cms_from_cycles = lambda cms: lambda n, k, c: np.sum([c.get(str(m), 0) / n * m for m in range(2, cms)])

# baseline = lambda x: 0 if x <= 0.5 else -0.008 + x / 1000
if __name__ == "__main__":
    # classic_data = json.loads(open(classic_file, 'r').read())
    # lite_classic_data = json.loads(open(lite_classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)
    sns.set(style="whitegrid", font="serif", font_scale=1.2)

    # dirichlet_data = json.loads(open(dirichlet_file, 'r').read())
    lite_dirichlet_data = json.loads(open(lite_dirichlet_file, 'r').read())
    lite_genome_data = json.loads(open(lite_genome_file % 100, 'r').read())
    slite_genome_data_100 = json.loads(open(slite_genome_file_100, 'r').read())
    # lite_gamma09_data = json.loads(open(lite_gamma09_file, 'r').read())
    lite_gamma03_data = json.loads(open(lite_gamma03_file, 'r').read())
    lite_gamma09_data = json.loads(open(lite_gamma09_file, 'r').read())
    # slite_dirichlet_data = json.loads(open(slite_dirichlet_file, 'r').read())
    # slite_dirichlet_data_round = json.loads(open("data/dirichlet_data_random_round_10.txt", 'r').read())
    # slite_classic_data_round = json.loads(open("data/classic_data_random_round_10.txt", 'r').read())
    # lite_dirichlet_data_smallN = json.loads(open(lite_dirichlet_file_smallN, 'r').read())
    # test_data = json.loads(open(test_file, 'r').read())
    db_genone_data1 = json.loads(open(db_genome_file % 1, 'r').read())
    db_genone_data1_2 = json.loads(open(db_genome_file2 % 1, 'r').read())
    db_genone_data5 = json.loads(open(db_genome_file % 5, 'r').read())
    db_genone_data10 = json.loads(open(db_genome_file % 10, 'r').read())
    db_genone_data20 = json.loads(open(db_genome_file % 20, 'r').read())
    db_genone_data30 = json.loads(open(db_genome_file % 30, 'r').read())
    db_genone_data50 = json.loads(open(db_genome_file % 50, 'r').read())
    db_genone_data100 = json.loads(open(db_genome_file % 100, 'r').read())

    like_real_data = json.loads(open(like_real_file, 'r').read())

    # data_dirichlet_est = DataEstimator(dirichlet_data)
    # data_classic_est = DataEstimator(classic_data)
    dirichlet_est = DirichletEstimator()
    test_est = CorrectedGammaEstimator(0.3, 20/1000)
    uniform_est = UniformEstimator()
    tannier_est = TannierEstimator()
    gamma_est = GammaEstimator(0.3)
    print("<read>")
    plt.rc('text', usetex=True)

    # drawer.draw_function(d_over_b_dir, "d/b analytical")
    # drawer.draw_data(dirichlet_data, d_over_b, False, "d/b empirical")
    # drawer = Drawer(0.3, 1.5)
    drawer = Drawer(0.5, 1.5)

    # drawer.draw_boxplot_data(db_genone_data1, db_est_error(test_est))
    gamma_data = json.loads(open(lite_gamma_file % 7, 'r').read())
    drawer.draw_boxplot_data(like_real_data, db_est_error(test_est))

    # drawer.draw_function(d_over_n_dir, label="analytcs")

    # drawer.draw_data(lite_gamma03_data,
    #                  lambda n, k, c: d_distance(n, k, c) / b_distance(n, k, c) - d_gamma_b(0.3, 50)(2 * k / n),
    #                  with_interval=False)

    # drawer.draw_data(dirichlet_data,
    #                  lambda n, k, c: d_distance(n, k, c) / b_distance(n, k, c) - d_over_b_dir(2 * k / n), label="dir data")

    #
    # drawer.draw_data(db_genone_data1, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)) / 1, with_interval=False,
    #                  label="1 chr")
    #

    # baseline = lambda x: 0 if x <= 0.5 else -0.009 + x / 1000
    # for i in range(1, 10):
    #     print(("0.%d" % i))
    #     gamma_data = json.loads(open(lite_gamma_file % i, 'r').read())
    #     drawer.draw_data(gamma_data,lambda n, k, c: d_distance(n, k, c) / b_distance(n, k, c),
    #                      label=("0.%d" % i), inc_color=False)
    #     drawer.draw_function(lambda x: d_gamma_b(i/10, 50)(x), linestyle="--")


    # drawer.draw_data(db_genone_data5, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- baseline(2 * k / n)) / 5,
    #                  label="5 chr")
    # drawer.draw_data(db_genone_data10, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- baseline(2 * k / n)) / 10,
    #                  label="10 chr")
    # drawer.draw_data(db_genone_data30, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n) - baseline(2 * k / n)) / 30,
    #                  label="30 chr")
    # drawer.draw_data(db_genone_data45, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n) - baseline(2 * k / n)) / 45,
    #                  label="45 chr")
    # drawer.draw_data(db_genone_data100, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- baseline(2 * k / n)) / 100,
    #                  label="100 chr")
    #
    # drawer.draw_data(db_genone_data10, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)),
    #                  with_interval=False, label="10 chr")
    #
    # drawer.draw_data(db_genone_data50, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)),
    #                  with_interval=False, label="50 chr")

    # drawer.draw_data(db_genone_data1, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)) / 1,
    #                  label="1 chr d/b")
    #
    # drawer.draw_data(db_genone_data5, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)) / 5,
    #                  with_interval=False, label="5 chr d/b")
    #
    # drawer.draw_data(db_genone_data10, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)),
    #                  with_interval=False, label="10 chr d/b", inc_color=False)
    # drawer.draw_function(addings(10/1000), linestyle="--")
    #
    # drawer.draw_data(db_genone_data20, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)),
    #                  with_interval=False, label="20 chr d/b", inc_color=False)
    # drawer.draw_function(addings(20/1000), linestyle="--")
    #
    # drawer.draw_data(db_genone_data30, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)),
    #                  with_interval=False, label="30 chr d/b", inc_color=False)
    # drawer.draw_function(addings(30/1000), linestyle="--")
    #
    # drawer.draw_data(db_genone_data50, lambda n, k, d, b: (d / b - d_over_b_dir(2 * k / n) - baseline(2 * k / n)),
    #                  with_interval=False, label="50 chr d/b", inc_color=False)
    # drawer.draw_function(addings(50/1000), linestyle="--")

    # drawer.draw_boxplot_data(db_genone_data, db_est_error(dirichlet_est))

    # drawer.draw_function(b_gamma(0.3), "b-gamma")
    # drawer.draw_function(d_gamma(0.3, 50), "d-gamma")
    # drawer.draw_data(lite_gamma03_data, lambda n, k, c: get_cms_from_cycles(6)(n, k, c) / b_distance(n, k, c), False,
    #                  "data")
    # drawer.draw_boxplot_data(lite_gamma03_data, est_error(gamma_est))
    # drawer.draw_data(lite_gamma03_data, b_distance, False, "b/n empirical (gamma)")
    # drawer.draw_data(lite_gamma03_data, d_distance, False, "d/n empirical (gamma)")

    # drawer.draw_data(lite_gamma03_data, data_c_m(5), True, "Empirical value of $c_2$/n", linewidth=0.8)
    # drawer.draw_function(cm4(5, 0.3), "Analytical value of $c_2$/n", linestyle='--', linewidth=0.8)

    # drawer.draw_data(lite_gamma03_data, gdata_c_m(3), True, "Empirical value of $c_3$/n", linewidth=0.8)
    # drawer.draw_function(cm4(3, 0.3), "Analytical value of $c_3$/n", linestyle='--', linewidth=0.8)

    # drawer.draw_function(d_over_b_dir, "d/b analytical (dirichlet)")
    # drawer.draw_data(lite_gamma03_data, d_over_b, False, "d/b empirical (gamma)")
    #
    # drawer.draw_function(d_over_n_dir, "d/n analytical (dirichlet)")
    # drawer.draw_data(lite_gamma03_data, d_distance, False, "d/n empirical (gamma)")
    #
    # drawer.draw_function(b_over_n_dir, "b/n analytical (dirichlet)")
    # drawer.draw_data(lite_gamma03_data, b_distance, False, "b/n empirical (gamma)")

    # drawer.draw_data(lite_genome_data, gdata_c_m(2), True, "Empirical value of $c_2$/n", linewidth=0.8)
    # drawer.draw_function(c_m_dir(2), "Analytical value of $c_2$/n", linestyle='--', linewidth=0.8)
    #
    # drawer.draw_data(lite_genome_data, gdata_c_m(3), True, "Empirical value of $c_3$/n", linewidth=0.8)
    # drawer.draw_function(c_m_dir(3), "Analytical value of $c_3$/n", linestyle='--', linewidth=0.8)
    # drawer.draw_function(d_over_b_dir, "d/b analytical")
    # drawer.draw_data(lite_genome_data, gd_over_b, False, "d/b empirical")

    # drawer.draw_data(dirichlet_data, data_c_m(2))

    # drawer.draw_boxplot_data(lite_genome_data, gest_error(test_dirichlet_est))

    # drawer.draw_data(lite_dirichlet_data, d_over_b, False, "Empirical value of d/b")
    # drawer.draw_function(d_over_b_dir, "Analytical value of d/b", linestyle='--')
    #
    # drawer.draw_data(lite_dirichlet_data, d_distance, False, "Empirical value of d/n")
    # drawer.draw_function(d_over_n_dir, "Analytical value of d/n", linestyle='--')
    #
    # drawer.draw_data(lite_dirichlet_data, b_distance, False, "Empirical value of b/n")
    # drawer.draw_function(b_over_n_dir, "Analytical value of b/n", linestyle='--')

    # drawer.draw_data(lite_classic_data, d_distance, False, "Empirical value of $d/n$")
    # drawer.draw_function(lambda x: 1 - sum(c_m(m)(x) for m in range(1, 30)), "Analytical value of $d/n$", linestyle='--')

    # drawer.draw_data(lite_gamma03_data, data_c_m(2), False, "Empirical value of $c_2$/n", linewidth=0.8)
    # drawer.draw_function(c_m_gamma(0.3, 2), "Analytical value of $c_2$/n t = 0.3", linestyle='--', linewidth=0.8)
    # drawer.draw_function(c_m_gamma(1, 2), "Analytical value of $c_2$/n t = 1", linestyle='--', linewidth=0.8)

    # drawer.draw_data(lite_gamma03_data, data_c_m(2), False, "Empirical value of $c_3$/n", linewidth=0.8)
    # drawer.draw_function(c_m_gamma(0.3, 2), "Analytical value of $c_2$/n", linestyle='--', linewidth=0.8)
    # # drawer.draw_function(c_m_gamma(1, 1), "Analytical value of $c_3$/n t = 1", linestyle='--', linewidth=0.8)
    # # print(c_m_gamma(0.3, 2)(1))
    #
    # drawer.draw_data(lite_gamma03_data, data_c_m(3), False, "Empirical value of $c_3$/n", linewidth=0.8)
    # drawer.draw_function(c_m_gamma(0.3, 3), "Analytical value of $c_3$/n", linestyle='--', linewidth=0.8)
    # #
    # drawer.draw_data(lite_gamma03_data, data_c_m(4), False, "Empirical value of $c_4$/n", linewidth=0.8)
    # drawer.draw_function(c_m_gamma(0.3, 4), "Analytical value of $c_4$/n", linestyle='--', linewidth=0.8)

    # drawer.draw_function(line, "$k/n$")
    # drawer.draw_data(lite_dirichlet_data, d_distance, False, "$\hat{d}/n$")

    # drawer.draw_data(dirichlet_data, data_c_m(2), True, "Empirical value of $c_2/n$", linewidth=0.8)
    # drawer.draw_function(c_m_dir(2), "Analytical value of $c_2/n$", linestyle='--', linewidth=0.8)

    plt.subplots_adjust(bottom=0.11, top=0.98, right=0.99, left=0.1)
    # plt.xlim(xmax=2.05, xmin=-0.05)
    # plt.ylim(ymin=0)
    drawer.show_and_save(xlabel=r'$\gamma$', legend_loc=1)
