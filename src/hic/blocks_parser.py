import numpy as np
import re
import pandas as pd
import networkx as nx
from collections import defaultdict
from src.real_data import RealDataGraph


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


p = "([A-Za-z0-9]+)\.([A-Za-z0-9_]+):(\d+)-(\d+) ([+|-]).*"
pattern = re.compile(p)
columns = ["block", "species", "chr", "chr_beg", "chr_end", "orientation"]


def parse_to_df(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    bs = np.split(lines, find_indices(lines, lambda x: x[0] == ">"))
    temp = []

    for i, b in enumerate(bs):
        b = b[1:-1]
        for oc in b:
            m = pattern.match(oc)
            temp.append([i, m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)])

    return pd.DataFrame(temp, columns=columns)


def create_bg(df, sp1, sp2, allowed_blocks=None, cyclic=False, add_missing_edges=False):
    g = RealDataGraph()

    def add_from_list_only_allowed(bs, label, cyclic):
        g.add_edges_from_list(
            bs if allowed_blocks is None else list(filter(lambda b: abs(b) in allowed_blocks, bs)),
            label, cyclic)

    def add_from_bss(label, bss):
        # print("<species>: APCF")
        for bs in bss:
            # print("   ", bs)
            add_from_list_only_allowed(bs, label, cyclic)
        # print("</species>")

    def add_one_sp(label, sp):
        # print("<species>:", sp)
        df_sp = df.loc[df['species'] == sp].sort_values(by=['chr', 'chr_beg'])

        for chr in df_sp['chr'].unique():
            df_sp_chr = df_sp.loc[df_sp['chr'] == chr]
            bs = [row['block'] if row['orientation'] == "+" else -row['block'] for _, row in df_sp_chr.iterrows()]
            add_from_list_only_allowed(bs, label, cyclic)
            # print("   ", chr, bs)

        # print("</species>")

    def add_any(label, a):
        if isinstance(a, str):
            add_one_sp(label, a)
        else:
            add_from_bss(label, a)

    add_any("black", sp1)
    add_any("red", sp2)
    # print()

    if add_missing_edges:
        g.add_missing_edges_minimizing_d()
    return g


# def remove_trivial_cycles_df_old(df, sp1, sp2):
#     g, coords = parse_to_bg(df, sp1, sp2)
#     to_delete = []
#     for component in nx.connected_component_subgraphs(g):
#         if len(component) == 2:
#             for edge in component.edges(data=True):
#                 if edge[2]['label'] == 'red':
#                     chr_i_beg, chr_i_end, chr_i = coords[int(edge[0] / 2)]
#                     chr_j_beg, chr_j_end, chr_j = coords[int(edge[1] / 2)]
#
#                     assert chr_i == chr_j
#
#                     if chr_i_beg > chr_j_beg:
#                         chr_i_beg, chr_i_end, chr_j_beg, chr_j_end = chr_j_beg, chr_j_end, chr_i_beg, chr_i_end
#
#                     to_delete.append((chr_i, chr_i_beg, chr_i_end, chr_j_beg, chr_j_end))
#
#     to_delete.sort(key=operator.itemgetter(1))
#     to_delete.sort(key=operator.itemgetter(0))
#
#     inds = []
#     for d in to_delete:
#         inds.extend([df.loc[(df['chr'] == d[0]) & (df['chr_beg'] == d[1])].index[0], df.loc[(df['chr'] == d[0]) & (df['chr_beg'] == d[3])].index[0]])
#     while len(inds) > 2:
#         i = 0
#         while i + 1 < len(inds) and inds[i] != inds[i + 1]:
#             i += 1
#
#         si = i
#         duplicates = False
#         while i + 1 < len(inds) and inds[i] == inds[i + 1]:
#             df.drop(inds[i], inplace=True)
#             i += 2
#             duplicates = True
#
#         if duplicates:
#             df.loc[inds[0]]['chr_end'] = df.loc[inds[i]]['chr_end']
#             df.drop(inds[i], inplace=True)
#             inds = inds[i + 1:]
#         else:
#             inds = inds[i - 1:]


def remove_trivial_cycles_df(df, df_sp, sp1, sp2):
    def merge_blocks(cmp):
        df_cmp = pd.DataFrame(columns=columns)
        for c in cmp:
            df_cmp = df_cmp.append(df_sp.loc[df_sp["block"] == int(c)], ignore_index=True)
        df_cmp = df_cmp.sort_values(by=['chr_beg'])

        assert df_cmp.iloc[0]['chr'] == df_cmp.iloc[-1]['chr']
        assert len(df_cmp) == len(cmp)
        df_sp.loc[df_sp['block'] == df_cmp.iloc[0]['block'], 'chr_end'] = df_cmp.iloc[1]['chr_end']

        for i in range(1, len(df_cmp)):
            df_sp.drop(df_sp.loc[df_sp['block'] == df_cmp.iloc[i]['block']].index, inplace=True)

    g = create_bg(df, sp1, sp2, allowed_blocks=df_sp['block'].unique().tolist())
    connections = nx.Graph()
    for component in nx.connected_component_subgraphs(g):
        if len(component) == 2:
            for edge in component.edges(data=True):
                if edge[2]['label'] == 'red':
                    connections.add_edge(edge[0][:-1], edge[1][:-1])

    for component in nx.connected_components(connections):
        merge_blocks(component)


def remove_copies(df):
    for sp in df['species'].unique():
        print(sp)
        for block in df['block'].unique():
            l = len(df.loc[(df['species'] == sp) & (df['block'] == block)])
            if l != 1:
                print(l)
