import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import json
from bisect import bisect_left
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, GammaEstimator
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
db_genome_file = "sim_data/dir_genome_data_N1000_104_%dс.txt"

cycles_breakdowns_lite_file = "sim_data/dirichlet_breakdown_of_a_cycle20.txt"
cycles_breakdowns_file = "sim_data/dirichlet_breakdown_of_a_cycle200.txt"

test_file = "sim_data/dirichlet_1000_20.txt"

bs = [-3.2530401983255075e-14, -1.5595164049031496e-15, -3.6741443221188774e-15, -2.5673907444456745e-15,
      -2.237793284010081e-15, -2.1337098754514727e-15, -4.0245584642661925e-15, -5.898059818321144e-16,
      -1.1449174941446927e-15, 5.134781488891349e-16, -2.0469737016526324e-15, 1.949829186997931e-15,
      2.275957200481571e-15, -2.0539125955565396e-15, 3.7470027081099033e-16, -1.923076923150477e-05,
      -3.8461538462496064e-05, -5.769230769405581e-05, -7.692307692404844e-05, -0.00010576923077097947,
      -0.00010576923077097947, -0.00010576923077032721, -0.00010576923076999415, -0.00012500000000020828,
      -0.000163461538463537, -0.00021153846154068218, -0.0002115384615427916, -0.0002307692307723396,
      -0.00023076923076947704, -0.000269230769234572, -0.0003076923076982199, -0.00031730769231271634,
      -0.00035576923077536505, -0.00038461538462007563, -0.0004038461538527323, -0.0004807692307718402,
      -0.0005192307692404841, -0.0005480769230868601, -0.0005865384615470662, -0.000605769230779945,
      -0.0006442307692506983, -0.0006634615384841583, -0.0007211538461765787, -0.0007692307692631606,
      -0.0008365384615819552, -0.0009326923077676649, -0.0009519230770280771, -0.0011057692307695557,
      -0.001192307692308026, -0.0012980769230771462, -0.0014326828890968498, -0.0016614929937688412,
      -0.0018247941595849653, -0.0019169116409855412, -0.0021251969446780036, -0.0021953278256884217,
      -0.0024120035166312454, -0.0025315722023798166, -0.0025896664247795236, -0.0027473069019856087,
      -0.0029254738606629585, -0.002997375877238758, -0.0030288001795809017, -0.003085774606389877,
      -0.0033210791922656194, -0.00344537618230441, -0.0034194965678408453, -0.0034448058990550084,
      -0.003347802941037528, -0.003195489391350666, -0.0033818950617587706, -0.0034069121842636185,
      -0.0033378232939065187, -0.0034439071606780546, -0.0032733534141485138, -0.0033936395963520957,
      -0.0032857551608007395, -0.003180733870073911, -0.003098109900538129, -0.003317066817602347, -0.00322258705428118,
      -0.0032958326092217475, -0.003306449225044136, -0.0031779478511974445, -0.0032953941234422075,
      -0.003293867365184418, -0.003394996773411834, -0.003493497839696377, -0.0034744782134141627,
      -0.003626897601421265, -0.003643566300615658, -0.003601913115675313, -0.0037332149543766735,
      -0.0037206727022684313, -0.0039013333643764803, -0.003891089232071131, -0.004142369449897935,
      -0.004136447050185037, -0.004104592216684543, -0.004210764077627157, -0.00412853332931511, -0.004300697222794422,
      -0.004362356139003132, -0.0042178363629417895, -0.004406074437122355, -0.004312155397990865,
      -0.004205774122542862, -0.0039335428587817765, -0.0041112987832372305, -0.004297180813665905,
      -0.004164706443729739, -0.004406617828209647, -0.004119497116841296, -0.003938381804105019,
      -0.0038636877857081376, -0.0037131324279481848, -0.003689042263980217, -0.0036668145465574833,
      -0.0033006865022673264, -0.0034179661324171915, -0.003307493790414472, -0.003469642227154097,
      -0.0036163166420477085, -0.003661339354472543, -0.0037396806334295293, -0.0036978433771641898,
      -0.003766940103794746, -0.0038223084063109835, -0.0038354341023016454, -0.003893182078018009,
      -0.0038708732879904712, -0.0037303617560234784, -0.0038469577311214063, -0.003865196998132343,
      -0.003602687112049441, -0.003607799786611395, -0.003582747898466054, -0.003585508645258692,
      -0.0035298236267038057, -0.0034544296946439164, -0.003609597495083858, -0.003389823855890187,
      -0.0033819087893705215, -0.003403417109547472, -0.0033872938949152972, -0.0034010953350068294,
      -0.003281604191521401, -0.0034425221816416847, -0.0033552392834990544, -0.0033084491945109947,
      -0.0033793032502572274, -0.003500718188497368, -0.00352868391212622, -0.0034730327895330115,
      -0.003497439723101468, -0.003659806832009197, -0.0038064942877819455, -0.003880012687187526,
      -0.003938253885581531, -0.004173721829119491, -0.004057763385908783, -0.0036097991760025735,
      -0.0035992467080385704, -0.003651289668614372, -0.00341033951826722, -0.003338112471015534,
      -0.0031367064728277457, -0.0032582166405794384, -0.0032316583922856693, -0.0031148905763002293,
      -0.0032350001379735788, -0.0030536867853061395, -0.0031288014997638953, -0.0033066542769056542,
      -0.003318168019907002, -0.003201955509034779, -0.0033620117548576437, -0.0035677140428156404,
      -0.0036557450538373284, -0.003770477522548139, -0.0037101281253562354, -0.0037729113677305763,
      -0.004007039624455823, -0.004028030871403689, -0.004239862570338756, -0.004036894783714587,
      -0.0040634855963851075, -0.00393514499810404, -0.004007765688353981, -0.0038007000343130777,
      -0.003939067797316734, -0.003999910012491676, -0.003973727483077301, -0.003966403889470504,
      -0.0037665135129971994, -0.003691475112889961, -0.0036548596487935518, -0.0035990826180342427,
      -0.0038800194692195675, -0.0039208517844731006, -0.003779374999553458, -0.0036672292003366025,
      -0.0037287453033641917, -0.0038005604666048274, -0.003940464268937367, -0.003956244889101217,
      -0.003674919899532539, -0.003702351675088671, -0.0036828628012861788, -0.003193466866673223,
      -0.0034554069470626456, -0.0035072325508333354, -0.0035221073330202754, -0.0034135783474424685,
      -0.003729806836847412, -0.0035189528688170904, -0.0036176368943254267, -0.003843247459746333,
      -0.0038304796877163816, -0.004021719911440725, -0.003888199539415427, -0.004035764304387137,
      -0.003897181973497735, -0.004059065442705649, -0.004050334292380006, -0.004274907111816263, -0.004242470746491369,
      -0.003962711083448714, -0.004147236144089623, -0.004161499176631141, -0.004399798671356129, -0.004573739914028088,
      -0.004260311923631915, -0.0042768874667625904, -0.004258146148658649, -0.004156074427035005,
      -0.0041438117795182345, -0.004192573793880327, -0.004254344488806007, -0.004281107095943218,
      -0.004292151764786654, -0.0040856140365619084, -0.004267321010188707, -0.004337329815693867,
      -0.004343773779733231, -0.004132862437026404, -0.0038873430800630774, -0.0038861161547324857,
      -0.0038215429634697764, -0.003953291829705876, -0.003973722723695791, -0.003844425734281855,
      -0.0037673749251065478, -0.0036464668060048463, -0.0038182895734709195, -0.003927123120167862,
      -0.003905708275023343, -0.003898323735486747, -0.0040204014611502446, -0.004252757451811939,
      -0.0042108225253995706, -0.004298488480343183, -0.004246569642412518, -0.003930110565246798,
      -0.004185693730763381, -0.004292208787836004, -0.004326621790892186, -0.004394744438320577, -0.004342772541552612,
      -0.004295747570909022, -0.004196018201329568, -0.00449554801182423, -0.004261684723249623, -0.004388698666746708,
      -0.004251628944362528, -0.0041101295895480455, -0.004137315727545726, -0.004015917427984009,
      -0.0041305874034999555, -0.003990977939022752, -0.004260587051738313, -0.00398752772784838,
      -0.0042679894669504425, -0.004409699826283423, -0.003989616734135413, -0.004046236187755998,
      -0.004089207643481784, -0.003945487868319792, -0.00399972556065937, -0.003944261355513438, -0.00368297398352297,
      -0.0038216648914093237, -0.003956519785425793, -0.003962570174719003, -0.003936001068465629,
      -0.004126843288306021, -0.004131281165323052, -0.004083960236933916, -0.004138756482181637, -0.004113007095412163]


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

    def draw_data(self, data, data_function, with_interval=True, label=None, linewidth=1.0):
        actual_func = lambda p: data_function(*p)
        ys = []
        for ds in data[self.min_index:self.max_index]:
            tmp = list([actual_func(d) for d in ds])  # [10:11]
            # print(ds[10:11])
            # print(tmp)
            ys.append(sorted(tmp))

        mean = list(map(lambda y: np.mean(y), ys))
        print("\n".join(map(lambda x: ("%.10f" % x).replace(".", ","), mean)))
        # open("data/dirichlet_d_error.txt", 'w').write(json.dumps(mean))
        plt.plot(self.draw_xs, mean, color=self.colors[self.current_color], label=label, linewidth=linewidth)
        print("sum of ys ** 2:", sum(map(lambda x: x ** 2, mean)))

        if with_interval:
            for interval in np.arange(0.5, 1, 0.05):
                low = list(map(lambda y: y[-int(len(y) * interval)], ys))
                upper = list(map(lambda y: y[int(len(y) * interval)], ys))
                plt.fill_between(self.draw_xs, low, upper, color=self.colors[self.current_color], alpha=0.125)

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


def smth(x):
    return bs[int((x) * 100)]


if __name__ == "__main__":
    # classic_data = json.loads(open(classic_file, 'r').read())
    # lite_classic_data = json.loads(open(lite_classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)
    sns.set(style="whitegrid", font="serif", font_scale=1.2)

    dirichlet_data = json.loads(open(dirichlet_file, 'r').read())
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
    db_genone_data5 = json.loads(open(db_genome_file % 5, 'r').read())
    db_genone_data10 = json.loads(open(db_genome_file % 10, 'r').read())
    db_genone_data30 = json.loads(open(db_genome_file % 30, 'r').read())
    db_genone_data45 = json.loads(open(db_genome_file % 45, 'r').read())
    db_genone_data100 = json.loads(open(db_genome_file % 100, 'r').read())

    # data_dirichlet_est = DataEstimator(dirichlet_data)
    # data_classic_est = DataEstimator(classic_data)
    dirichlet_est = DirichletEstimator()
    test_dirichlet_est = FirstCmsDirEstimator(10)
    uniform_est = UniformEstimator()
    tannier_est = TannierEstimator()
    gamma_est = GammaEstimator(0.3, 30)
    print("<read>")
    plt.rc('text', usetex=True)

    # drawer = Drawer()
    # drawer.draw_function(d_over_b_dir, "d/b analytical")
    # drawer.draw_data(dirichlet_data, d_over_b, False, "d/b empirical")
    # drawer = Drawer(0.3, 1.5)
    drawer = Drawer(0.1, 3)

    # drawer.draw_function(d_over_n_dir, label="analytcs")

    # drawer.draw_data(lite_gamma03_data,
    #                  lambda n, k, c: d_distance(n, k, c) / b_distance(n, k, c) - d_gamma_b(0.3, 50)(2 * k / n),
    #                  with_interval=False)

    drawer.draw_data(dirichlet_data,
                     lambda n, k, c: d_distance(n, k, c) / b_distance(n, k, c) - d_over_b_dir(2 * k / n),
                     with_interval=False)

    drawer.draw_data(db_genone_data1, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)) / 1, with_interval=False,
                     label="1 chr")

    # drawer.draw_data(db_genone_data5, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- smth(2 * k / n)) / 5, with_interval=False,
    #                  label="5 chr")
    # drawer.draw_data(db_genone_data10, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- smth(2 * k / n)) / 10, with_interval=False,
    #                  label="10 chr")
    # drawer.draw_data(db_genone_data30, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n) - smth(2 * k / n)) / 30,
    #                  with_interval=False, label="30 chr")
    # drawer.draw_data(db_genone_data45, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n) - smth(2 * k / n)) / 45,
    #                  with_interval=False, label="45 chr")
    # drawer.draw_data(db_genone_data100, lambda n, k, d, b: (d / n - d_over_n_dir(2 * k / n)- smth(2 * k / n)) / 100,
    #                  with_interval=False, label="100 chr")

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
