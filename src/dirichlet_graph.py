import networkx as nx
import random
import numpy as np
from operator import itemgetter
from collections import defaultdict


class DirichletDnkGraph(nx.MultiGraph):
    def __init__(self, n, initial_distribution=None):
        self.n = n
        super().__init__()
        if not initial_distribution:
            initial_distribution = sorted(
                [random.expovariate(1) for _ in range(n)], reverse=True
            )
            initial_distribution = initial_distribution / np.sum(initial_distribution)
        for i in range(n):
            self.add_edge(i, i, weight=initial_distribution[i])

    def check_correctness(self):
        for deg in list(self.degree):
            assert deg[1] == 2
        assert abs(self.sum_on_edges() - 1) < 1e-6

    def do_k_break(self, k=2):
        def generate_new_permutation(n):
            def check_permutation(p):
                return not any(
                    (p[i] % 2 == 0 and p[i] == p[i + 1] - 1)
                    or (p[i + 1] % 2 == 0 and p[i + 1] == p[i] - 1)
                    for i in range(0, len(p), 2)
                )

            permutation = np.random.permutation(n)
            while not check_permutation(permutation):
                permutation = np.random.permutation(n)
            return permutation

        def generate_new_weights(ws):
            def generate_normed_exp(n):
                xs = [random.expovariate(1) for _ in range(n)]
                return xs / np.sum(xs)

            if len(ws) == 2:
                r1 = random.random()
                r2 = random.random()
                return [ws[0] * r1 + ws[1] * r2, (1 - r1) * ws[0] + (1 - r2) * ws[1]]

            r = np.array([generate_normed_exp(len(ws)) for _ in range(len(ws))])
            return np.dot(r.transpose(), ws)

        all_old_ws = list(map(lambda x: x[2]["weight"], self.edges.data()))
        edges_indexes = np.random.choice(self.n, k, replace=False, p=all_old_ws)

        edges = itemgetter(*edges_indexes)(list(self.edges(keys=True)))
        self.remove_edges_from(edges)

        old_ws = itemgetter(*edges_indexes)(all_old_ws)
        new_ws = generate_new_weights(old_ws)
        new_order = generate_new_permutation(k * 2)

        vertexes_in_edges = [
            item for sublist in map(lambda x: x[:2], edges) for item in sublist
        ]
        for i, new_w in zip(range(0, len(new_order), 2), new_ws):
            self.add_edge(
                vertexes_in_edges[new_order[i]],
                vertexes_in_edges[new_order[i + 1]],
                weight=new_w,
            )

    def count_cycles(self):
        counter = defaultdict(lambda: 0)
        for component in nx.connected_components(self):
            counter[len(component)] += 1
        return counter

    def sum_on_edges(self):
        return sum(map(lambda x: x[2]["weight"], self.edges(data=True)))
