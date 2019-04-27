import networkx as nx
from collections import defaultdict
from src.estimators import DirichletEstimator, TannierEstimator, get_d_from_cycles, get_b_from_cycles, DataEstimator, \
    UniformEstimator, FirstCmsDirEstimator, CmBFunctionGammaEstimator, GammaEstimator, CorrectedGammaEstimator
from itertools import combinations
from os import listdir
import numpy as np
import sys
from scipy import stats
import random
from src.convertors import get_cms_dist
from src.drawer import Drawer, data_c_m
import json
import math
import matplotlib.pyplot as plt

species = "plants"
folder_path = "real_data/" + species + "/"


class RealDataGraph(nx.MultiGraph):
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

    def add_edges_from_list(self, ls, label, cyclic=False):
        def add_block_edge(i, j):
            self.add_edge(str(abs(i)) + ("h" if i > 0 else "t"), str(abs(j)) + ("t" if j > 0 else "h"), label=label)
            # self.add_edge(2 * abs(i) + (1 if i > 0 else 0), 2 * abs(j) + (0 if j > 0 else 1), label=label)

        self.add_nodes_from(map(lambda x: str(abs(x)) + "h", ls))
        self.add_nodes_from(map(lambda x: str(abs(x)) + "t", ls))
        for i, j in zip(ls, ls[1:] + ls[:1]) if cyclic else zip(ls[:-1], ls[1:]):
            add_block_edge(i, j)

    @staticmethod
    def get_blocks_from_file(file):
        lls = []
        with open(file, "r") as f:
            for ch_index, line in enumerate(filter(lambda l: len(l) > 1 and (l[-2] == "$" or l[-2] == "@"), f)):
                split = list(map(lambda x: int(x), line.split()[:-1]))
                lls.append(split)
        return lls

    def add_edges_from_file(self, file, label="red", cyclic=False):
        for split in self.get_blocks_from_file(file):
            # print(split)
            self.add_edges_from_list(split, label, cyclic)

    def add_missing_edges_minimizing_d(self):
        components_without_ends = []
        rb_edges, bb_edges = defaultdict(lambda: 0), defaultdict(lambda: 0)

        for component in nx.connected_components(self):
            dfsed = list(nx.dfs_edges(self, next(iter(component))))

            # Kostil'
            if len(dfsed) == 0:
                continue

            dfsed_2 = list(nx.dfs_edges(self, dfsed[-1][1]))

            if self.has_edge(dfsed_2[-1][1], dfsed_2[0][0]):
                continue

            for i, j in zip(dfsed_2[:-1], dfsed_2[1:]):
                if i[1] != j[0]:
                    print("SUPER OOOOOOOOOOOOOOOOOOOOOOPS", i, j)

            label1 = self.get_edge_data(dfsed_2[0][0], dfsed_2[0][1])[0]['label']
            label2 = self.get_edge_data(dfsed_2[-1][0], dfsed_2[-1][1])[0]['label']

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
        for component in nx.connected_component_subgraphs(self):
            # print(len(component), len(component.edges))
            # counter[str(int(len(component) / 2))] += 1
            counter[int(len(component) / 2)] += 1
        return counter

    # get_b_from_cycles = lambda c: sum([int(k) * v for k, v in c.items()])
    # get_d_from_cycles = lambda c: sum([(int(k) - 1) * v for k, v in c.items()])

    def check(self):
        assert all(map(lambda v: v[1] <= 2, self.degree))

    def b(self):
        b = 0.
        for component in nx.connected_component_subgraphs(self):
            if len(component) > 2 or (len(component) == 2 and len(component.edges) == 1):
                b += len(component) / 2
                # b += math.floor((len(component)) / 2 + 1)
                # b += math.ceil((len(component)) / 2 - 1)
            # print(len(component), len(component.edges))
        return b

    def non_trivial_blocks(self):
        dct = defaultdict(lambda: 0)
        for component in nx.connected_component_subgraphs(self):
            if not (len(component) == len(component.edges) == 2 or len(component) == 1):
                for v in component:
                    dct[int(v[:-1])] += 1
        return dct

    def b_v2(self):
        b = 0
        for component in nx.connected_component_subgraphs(self):
            if len(component) == 1:
                continue
            if len(component) == len(component.edges) and len(component) == 2:
                continue
            if len(component) == len(component.edges):
                b += (len(component) / 2)
            else:
                b += len(component) / 2  # \
                # - (1 / 2 if len(component) % 2 == 1 else 0)
        return b

    def d_v2(self):
        d = 0
        for component in nx.connected_component_subgraphs(self):
            if len(component) == len(component.edges):
                d += (len(component) / 2) - 1
            else:
                d += len(component) / 2 \
                     - (1 / 2 if len(component) % 2 == 1 else 0)
        return d

    def count_components(self, predicate):
        return len(list(filter(predicate, nx.connected_component_subgraphs(self))))

    def p_even(self):
        return self.count_components(lambda c: len(c) != len(c.edges) and len(c.edges) % 2 == 0)

    def p_odd(self):
        return self.count_components(lambda c: len(c) != len(c.edges) and len(c.edges) % 2 == 1)

    def p_m(self, m):
        return self.count_components(lambda c: len(c) != len(c.edges) and len(c.edges) == m)

    def c(self):
        return self.count_components(lambda c: len(c) == len(c.edges))

    def c_m(self, m):
        return self.count_components(lambda c: len(c) == len(c.edges) and len(c) == m * 2)

    def c_not_2(self):
        return self.count_components(lambda c: len(c) == len(c.edges) and len(c) != 2)

    def n(self):
        return len(self) // 2

    def d(self):
        return self.n() - self.c() - self.p_even() // 2

    def draw(self):
        red_edges = [(u, v) for (u, v, d) in self.edges(data=True) if d['label'] == 'red']
        black_edges = [(u, v) for (u, v, d) in self.edges(data=True) if d['label'] == 'black']

        # nodes
        pos = nx.circular_layout(sorted(self.nodes))
        print("in draw", sorted(self.nodes))
        nx.draw_networkx_nodes(self, pos, nodelist=sorted(self.nodes))

        # edges
        nx.draw_networkx_edges(self, pos, edgelist=red_edges, width=2, style=':', edge_color='r')
        nx.draw_networkx_edges(self, pos, edgelist=black_edges, width=1, style='-', edge_color='k')

        # labels
        nx.draw_networkx_labels(self, pos)

        plt.axis('off')
        plt.show()


without_c1 = lambda g: {k: v for k, v in g.items() if k != 1}

if __name__ == "__main__":
    # sys.stdout = open(species + '.txt', 'w')
    # classic_file = "data/classic_data_N2000_2000.txt"
    # classic_data = json.loads(open(classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)

    dirEst = DirichletEstimator()
    gammaEst = GammaEstimator(0.3)
    corGammaEst = CorrectedGammaEstimator(0.3, 20/1000)

    # tanEst = TannierEstimator()
    # uniEst = UniformEstimator()
    # dirEst = FirstCmsDirEstimator(4)
    files = sorted(listdir(folder_path))
    print(files)
    # max_cms = 10
    # cms_dist = get_cms_dist("data/dirichlet_data_randomN_2000.txt", max_cms)

    for f1, f2 in combinations(files, 2):
        print(f1, "-", f2)
        graph = RealDataGraph()
        # print("red")
        graph.add_edges_from_file(folder_path + f1, "red")
        # print("black")
        graph.add_edges_from_file(folder_path + f2, "black")

        # for k, v in sorted(graph.count_cycles().items()):
        #     print(k,":", v)
        # graph.add_missing_edges_minimizing_d()

        # cycles = without_c1(graph.count_cycles())
        # for k in sorted(map(lambda x: int(x), cycles.keys())):
        #     print("   ", k, ":", cycles[str(k)])
        #
        n, k = gammaEst.predict_by_db(graph.d(), graph.b())
        n2, k2 = corGammaEst.predict_by_db(graph.d(), graph.b())

        print(graph.d(), graph.b(), graph.d() / graph.b(), k, k2)
        # x = 2 * k / n
        # i_cur_dist = int(round(x * 100))
        # cur_dist = cms_dist[i_cur_dist]
        # for cm in range(2, max_cms):
        #     real_cm = cycles.get(str(cm), 0) / n
        #     dist_cm = cur_dist[cm]
        #     print("m = %d, stats.percentileofscore = %.2f, real c%d/n = %.4f, range of dist is [%.4f, %.4f]" % (cm, stats.percentileofscore(
        #         dist_cm,
        #         real_cm
        #     ), cm, real_cm, np.min(dist_cm), np.max(dist_cm)))
