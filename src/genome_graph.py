import networkx as nx
import random
import numpy as np
from collections import defaultdict
from src.real_data import RealDataGraph
import matplotlib.pyplot as plt


class GenomeGraph(nx.Graph):
    def __init__(self, n, chrs):
        super().__init__()
        self.n, self.chrs = n, chrs
        self.ns = list(range(n))
        self.add_nodes_from([str(i) + 'h' for i in self.ns])
        self.add_nodes_from([str(i) + 't' for i in self.ns])
        [self.add_edge(str(i) + "h", str(i) + "t", label='block') for i in self.ns]

        ws = [random.expovariate(1) for _ in range(n + chrs)]
        i = 0
        for xs in np.array_split(self.ns, chrs):
            self.add_edge(str(xs[0]) + 'h', 'telomere', label='adj-red', weight=ws[i] / sum(ws))
            self.add_edge(str(xs[-1]) + 't', 'telomere', label='adj-red', weight=ws[i + 1] / sum(ws))
            i += 2
            for x, y in zip(xs[:-1], xs[1:]):
                self.add_edge(str(x) + 't', str(y) + 'h', label='adj-red', weight=ws[i] / sum(ws))
                i += 1

    def count_cycles(self):
        g = nx.MultiGraph(self)
        # removing block edges
        for edge in self.edges(data='label'):
            if edge[2] == 'block':
                g.remove_edge(edge[0], edge[1])
        # adding bp edges
        for x, y in zip(self.ns, self.ns[1:] + self.ns[:1]):
            g.add_edge(str(x) + 't', str(y) + 'h', label='adj-black')
        # counting
        counter = defaultdict(lambda: 0)
        for component in nx.connected_component_subgraphs(g):
            counter[len(component.edges)] += 1
        return counter

    def make_bg_real_data_graph(self):
        g = RealDataGraph()

        # black edges are the same as red at the begin of the process
        g.add_nodes_from(map(lambda x: str(x) + "h", self.ns))
        g.add_nodes_from(map(lambda x: str(x) + "t", self.ns))

        for xs in np.array_split(self.ns, self.chrs):
            for x, y in zip(xs[:-1], xs[1:]):
                g.add_edge(str(x) + 't', str(y) + 'h', label='black')

        # adding bp edges
        for e in self.label_edges_with_weight('adj-red'):
            if e[1] != 'telomere':
                g.add_edge(e[0], e[1], label='red')

        return g

    def label_edges_with_weight(self, label):
        # return list(self.edges(data=True))
        return list(map(lambda e: (e[0], e[1], e[2]),
                        filter(lambda e: e[2]['label'] == label, self.edges(data=True))))

    def do_k2_break(self):
        def insert_edge(e, label='adj-red'):
            self.add_edge(e[0], e[1], weight=e[2], label=label)

        def delete_edge(e):
            self.remove_edge(e[0], e[1])

        def new_weights(w1, w2=0):
            r1, r2 = random.random(), random.random()
            return w1 * r1 + w2 * r2, w1 * (1 - r1) + w2 * (1 - r2)

        break_is_done = False
        while not break_is_done:
            es = self.label_edges_with_weight('adj-red')
            all_ws = list(map(lambda e: e[2]['weight'], es))
            i1, i2 = np.random.choice(len(es), size=2, replace=True, p=all_ws)
            e1, e2 = es[i1], es[i2]

            if e1[0] == 'telomere' or e2[0] == 'telomere':
                print("WARNING")

            # Breaking chromosome
            if i1 == i2:
                if e1[1] != 'telomere':
                    print("________Breaking_chromosome")
                    delete_edge(e1)
                    w1, w2 = new_weights(e1[2]['weight'])
                    insert_edge((e1[0], 'telomere', w1), label='adj-red')
                    insert_edge((e1[1], 'telomere', w2), label='adj-red')
                    break
                else:
                    continue

            # Merging chromosome
            if e1[1] == 'telomere' and e2[1] == 'telomere':
                print("________Merging_chromosome")
                delete_edge(e1)
                delete_edge(e2)
                insert_edge((e1[0], e2[0], e1[2]['weight'] + e2[2]['weight']), label='adj-red')

                if len(nx.cycle_basis(self.without_telomere())) > 0:
                    delete_edge((e1[0], e2[0]))
                    insert_edge((e1[0], e1[1], e1[2]['weight']))
                    insert_edge((e2[0], e2[1], e2[2]['weight']))
                    continue
                break

            # Translocation
            # print("____Translocation_started")
            if i1 == i2:
                continue

            e1 = (e1[0], e1[1], e1[2]['weight'])
            e2 = (e2[0], e2[1], e2[2]['weight'])
            w1, w2 = new_weights(e1[2], e2[2])


            delete_edge(e1)
            delete_edge(e2)

            if random.random() > 0.5:
                ne1 = (e1[0], e2[1], w1)
                ne2 = (e1[1], e2[0], w2)
            else:
                ne1 = (e1[0], e2[0], w1)
                ne2 = (e1[1], e2[1], w2)

            if self.has_edge(ne1[0], ne1[1]) or self.has_edge(ne2[0], ne2[1]):
                insert_edge(e1)
                insert_edge(e2)
                # print("possible!")
                # print("____Translocation_failed_becouse_of_edge_existence")
                continue

            insert_edge(ne1)
            insert_edge(ne2)

            if len(nx.cycle_basis(self.without_telomere())) > 0:
                delete_edge(ne1)
                delete_edge(ne2)
                insert_edge(e1)
                insert_edge(e2)
                # print("____Translocation_failed_becouse_of_cycles")
                continue

            # print("____Translocation_ended")
            break_is_done = True

    def sum_ws_on_edges(self):
        # print(self.edges.data('weight', default=0))
        now_ws = list(filter(lambda x: x != 0, map(lambda x: x[2], self.edges.data('weight', default=0))))
        return sum(now_ws)

    def without_telomere(self):
        g = nx.Graph(self)
        g.remove_node('telomere')
        return g


# random.seed(42)
# np.random.seed(42)
# g = GenomeGraph(1000, 50)  # 1000, 10
#
# for i in range(3000):
#     bpg = g.make_bg_real_data_graph()
#
#     bpg.check()
#
#     b, bv2 = bpg.b(), bpg.b_v2()
#     # bpg.draw()
#     print(b, bv2)
#     # print(bpg.d(), bpg.d_v2())
#     assert b == bv2
#
#     print("____K =", i + 1)
#     g.do_k2_break()
#
#
# print("sum", g.sum_ws_on_edges())
# print(g.count_cycles())
