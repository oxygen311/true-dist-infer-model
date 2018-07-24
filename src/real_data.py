import networkx as nx
from collections import defaultdict
from src.estimators import DirichletEstimator, TannierEstimator, get_d_from_cycles, get_b_from_cycles, DataEstimator, \
    UniformEstimator, FirstCmsDirEstimator
from itertools import combinations
from os import listdir
from time import time
import numpy as np
import random

folder_path = "real_data/rasacae/"


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

        with open(folder_path + file, "r") as f:
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

        # print("Components lengths by adding edge between edges same colors")
        # print(bb_edges)
        # print("Components lengths by adding edges between black and red edges")
        # print(rb_edges)

    def count_cycles(self):
        counter = defaultdict(lambda: 0)
        for component in nx.connected_components(self):
            counter[str(int(len(component) / 2))] += 1
        return counter


def print_array_stats(arr, parament_name):
    arr.sort()
    ln = len(arr)
    print("Value:", parament_name)
    print("    Min: ", round(np.min(arr), 2), "    Mean:", round(np.mean(arr), 2), "    Max: ", round(np.max(arr), 2))
    print("    80% of results are between", round(arr[int(ln * 0.1)], 2), "and", round(arr[int(ln * 0.9)], 2))


without_c1 = lambda g: {k: v for k, v in g.items() if k != '1'}

if __name__ == "__main__":
    # classic_file = "data/classic_data_N2000_2000.txt"
    # classic_data = json.loads(open(classic_file, 'r').read())
    # classic_est = DataEstimator(classic_data)

    # dirEst = DirichletEstimator()
    tanEst = TannierEstimator()
    uniEst = UniformEstimator()
    dirEst = FirstCmsDirEstimator(6)
    files = sorted(listdir(folder_path))

    for f1, f2 in combinations(files, 2):
        # print(f1, "-", f2)
        graph = RealDataGraph()
        graph.add_edges_from_file(f1, "red")
        graph.add_edges_from_file(f2, "black")
        graph.add_missing_edges_minimizing_d()

        cycles = without_c1(graph.count_cycles())

        some_dict = {}
        some_dict["P.gen"] = "Prunus"
        some_dict["S.gen"] = "Fragaria"
        some_dict["V.gen"] = "Malus"

        # get_d_from_cycles(cycles), "b:", get_b_from_cycles(cycles), "estimated k:",
        #               dirEst.predict_k(cycles))

        print("\emph{%s} --- \emph{%s}    & %d & %d & %d & %d     \\\hline" % (some_dict[f1], some_dict[f2],
                                                                               get_d_from_cycles(cycles),
                                                                               dirEst.predict_k(cycles),
                                                                               tanEst.predict_k(cycles),
                                                                               uniEst.predict_k(cycles)))
