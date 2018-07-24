import networkx as nx
import random
import numpy as np
import operator
from collections import defaultdict


class DnkGraph(nx.MultiGraph):
    def __init__(self, n):
        super().__init__()
        for i in range(n):
            self.add_edge(i, i)

    def check_correctness(self):
        for deg in list(self.degree):
            assert deg[1] == 2

    def do_k_break(self, k=2):
        def generate_new_permutation(n):
            def check_permutation(p):
                for i in range(0, len(permutation), 2):
                    if (p[i] % 2 == 0 and p[i] == p[i + 1] - 1) or (p[i + 1] % 2 == 0 and p[i + 1] == p[i] - 1):
                        return False
                return True

            permutation = np.random.permutation(n)
            while not check_permutation(permutation):
                permutation = np.random.permutation(n)
            return permutation

        edges_indexes = np.random.choice(len(self.edges), k, replace=False)
        edges = operator.itemgetter(*edges_indexes)(list(self.edges(data=False)))
        new_order = generate_new_permutation(k * 2)

        self.remove_edges_from(edges)
        vertexes_in_edges = [item for sublist in list(edges) for item in sublist]

        for i in range(0, len(new_order), 2):
            self.add_edge(vertexes_in_edges[new_order[i]], vertexes_in_edges[new_order[i + 1]])

    def count_cycles(self):
        counter = defaultdict(lambda: 0)
        for component in nx.connected_components(self):
            counter[len(component)] += 1
        return counter