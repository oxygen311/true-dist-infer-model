import networkx as nx
from collections import defaultdict
from src.estimators import DirichletEstimator, TannierEstimator, get_d_from_cycles, get_b_from_cycles, DataEstimator, \
    UniformEstimator, FirstCmsDirEstimator
from itertools import combinations
from os import listdir
import numpy as np
import sys
from scipy import stats
import random
from src.convertors import get_cms_dist
from src.drawer import Drawer, data_c_m, c_m_dir, d_distance, b_distance, d_over_n_dir, b_over_n_dir, d_over_b, \
    d_over_b_dir
import json
from src.real_data import RealDataGraph
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


without_c1 = lambda g: {k: v for k, v in g.items() if k != '1'}
# est = FirstCmsDirEstimator(3)
# est = DirichletEstimator()
est = TannierEstimator()
cs = range(2, 4)
lite_dirichlet_data = json.loads(open("data/dirichlet_data_randomN_200.txt", 'r').read())
dirichlet_data = json.loads(open("data/dirichlet_data_randomN_2000.txt", 'r').read())


def process_real_data(species, marker='o'):
    print(species)
    folder_path = "real_data/" + species + "/"
    files = sorted(listdir(folder_path))

    cs_dots = defaultdict(lambda: defaultdict(lambda: []))
    # cs_dots = defaultdict(lambda: [])
    for f1, f2 in combinations(files, 2):
        print(f1, "-", f2)
        graph = RealDataGraph()
        graph.add_edges_from_file(folder_path + f1, "red")
        graph.add_edges_from_file(folder_path + f2, "black")
        graph.add_missing_edges_minimizing_d()

        cycles = without_c1(graph.count_cycles())

        for k in sorted(map(lambda x: int(x), cycles.keys())):
            # print("   ", k, ":", cycles[str(k)])
            n, k = est.predict(cycles)
            x = 2 * k / n
            for c in cs:
                real_cm = cycles.get(str(c), 0) / n
                cs_dots[c]["xs"].append(x)
                cs_dots[c]["ys"].append(real_cm)
            # real_dn = get_d_from_cycles(cycles) / get_b_from_cycles(cycles)
            # if (x >= 0.1):
            #     cs_dots["xs"].append(x)
            #     cs_dots["ys"].append(real_dn)

    drawer.reset_color()
    for c in cs:
        drawer.draw_dots(cs_dots[c]["xs"], cs_dots[c]["ys"], ("real value of $c_{%d}$/n in %s" % (c, species)),
                         marker=marker)
        drawer.increase_color()
    # drawer.draw_dots(cs_dots["xs"], cs_dots["ys"], ("real value of d/b in %s" % species), marker=marker)
    # drawer.increase_color()


if __name__ == "__main__":
    sns.set(style="whitegrid", font="serif", font_scale=1.2)
    plt.rc('text', usetex=True)
    mpl.rcParams['figure.dpi'] = 300
    plt.axvline(x=0.5, linestyle='dashed', color='grey', linewidth=1, label="parsimony phase bound")

    drawer = Drawer(0, 1.4)

    for c in cs:
        drawer.draw_function(c_m_dir(c))
        drawer.draw_interval(dirichlet_data, data_c_m(c))
        drawer.increase_color()
    # drawer.draw_function(d_over_b_dir)
    # drawer.draw_interval(dirichlet_data, d_over_b)
    # drawer.increase_color()

    process_real_data("mammals")
    process_real_data("plants", marker='^')

    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1.4)

    plt.subplots_adjust(bottom=0.1, top=0.98, right=0.98, left=0.07)
    drawer.show_and_save(legend_loc=1, xlabel=r'$\gamma$')
