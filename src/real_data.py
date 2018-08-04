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
from src.cms_dist_from_data import get_cms_dist
from src.drawer import Drawer, data_c_m
import json

species = "mammalian"
folder_path = "real_data/" + species + "/"


class RealDataGraph(nx.Graph):
    def add_edges_from_file_with_rnd_ends(self, file):
        with open(folder_path + file, "r") as f:
            ends = []
            for line in filter(lambda l: len(l) > 1 and (l[-2] == "$" or l[-2] == "@"), f):
                split = list(map(lambda x: int(x), line.split()[:-1]))
                for i, j in zip(split[:-1], split[1:]):
                    self.add_edge(2 * abs(i) + (1 if i > 0 else 0), 2 * abs(j) + (0 if j > 0 else 1))
                ends.append(split[0])
                ends.append(split[-1])

            random.shuffle(ends)
            for k in range(0, len(ends), 2):
                i = ends[k]
                j = ends[k + 1]
                self.add_edge(2 * abs(i) + (1 if i > 0 else 0), 2 * abs(j) + (0 if j > 0 else 1))

    def add_edges_from_file(self, file, label="red", cyclic=False):
        def add_block_edge(i, j):
            self.add_edge(2 * abs(i) + (1 if i > 0 else 0), 2 * abs(j) + (0 if j > 0 else 1), label=label)

        with open(file, "r") as f:
            for ch_index, line in enumerate(filter(lambda l: len(l) > 1 and (l[-2] == "$" or l[-2] == "@"), f)):
                split = list(map(lambda x: int(x), line.split()[:-1]))
                for i, j in zip(split, split[1:] + split[:1]) if cyclic else zip(split[:-1], split[1:]):
                    add_block_edge(i, j)

    def add_missing_edges_minimizing_d(self):
        components_without_ends = []
        rb_edges, bb_edges = defaultdict(lambda: 0), defaultdict(lambda: 0)

        for component in nx.connected_components(self):
            dfsed = list(nx.dfs_edges(self, next(iter(component))))
            dfsed_2 = list(nx.dfs_edges(self, dfsed[-1][1]))

            if self.has_edge(dfsed_2[-1][1], dfsed_2[0][0]):
                continue

            for i, j in zip(dfsed_2[:-1], dfsed_2[1:]):
                if i[1] != j[0]:
                    print("SUPER OOOOOOOOOOOOOOOOOOOOOOPS", i, j)

            label1 = self.get_edge_data(dfsed_2[0][0], dfsed_2[0][1])['label']
            label2 = self.get_edge_data(dfsed_2[-1][0], dfsed_2[-1][1])['label']

            if label1 == label2:
                self.add_edge(dfsed_2[0][0], dfsed_2[-1][1])
                bb_edges[int(len(component) / 2)] += 1
            else:
                components_without_ends.append((dfsed_2[0][0], dfsed_2[-1][1]))

        random.shuffle(components_without_ends)
        for i in range(0, len(components_without_ends) - 1, 2):
            comp1 = components_without_ends[i]
            comp2 = components_without_ends[i + 1]
            self.add_edge(comp1[0], comp2[0])
            self.add_edge(comp1[1], comp2[1])
            rb_edges[int(len(comp1) + len(comp2))] += 1

        if len(components_without_ends) % 2 == 1:
            self.add_edge(components_without_ends[-1][0], components_without_ends[-1][1])

    def count_cycles(self):
        counter = defaultdict(lambda: 0)
        for component in nx.connected_components(self):
            counter[str(int(len(component) / 2))] += 1
        return counter


without_c1 = lambda g: {k: v for k, v in g.items() if k != '1'}

if __name__ == "__main__":
    sys.stdout = open(species + '.txt', 'w')
    # classic_file = "data/classic_data_N2000_2000.txt"
    # classic_data = json.loads(open(classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)

    # dirEst = DirichletEstimator()
    tanEst = TannierEstimator()
    uniEst = UniformEstimator()
    dirEst = FirstCmsDirEstimator(4)
    files = sorted(listdir(folder_path))
    max_cms = 10
    cms_dist = get_cms_dist("data/dirichlet_data_randomN_2000.txt", max_cms)

    for f1, f2 in combinations(files, 2):
        print(f1, "-", f2)
        graph = RealDataGraph()
        graph.add_edges_from_file(f1, "red")
        graph.add_edges_from_file(f2, "black")
        graph.add_missing_edges_minimizing_d()

        cycles = without_c1(graph.count_cycles())
        for k in sorted(map(lambda x: int(x), cycles.keys())):
            print("   ", k, ":", cycles[str(k)])

        n, k = dirEst.predict(cycles)
        x = 2 * k / n
        i_cur_dist = int(round(x * 100))
        cur_dist = cms_dist[i_cur_dist]
        for cm in range(2, max_cms):
            real_cm = cycles.get(str(cm), 0) / n
            dist_cm = cur_dist[cm]
            print("m = %d, stats.percentileofscore = %.2f, real c%d/n = %.4f, range of dist is [%.4f, %.4f]" % (cm, stats.percentileofscore(
                dist_cm,
                real_cm
            ), cm, real_cm, np.min(dist_cm), np.max(dist_cm)))
