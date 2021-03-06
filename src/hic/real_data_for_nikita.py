from src.hic.blocks_parser import parse_to_df, create_bg
import pandas as pd
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, get_d_from_cycles, get_b_from_cycles, GammaEstimator, MoretEstimator, CorrectedGammaEstimator
from src.real_data import without_c1
import time
from src.real_data import RealDataGraph
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

sp_blocks_from_df = lambda df, sp: df.loc[df['species'] == sp]['block'].tolist()
uniq_predicate = lambda ls: lambda b: ls.count(b) == 1

# df_file = "real_data/%s/300Kbp/SFs/Orthology.Blocks"
df_file = "real_data/%s/%s/SFs/Conserved.Segments"
df_file_way2 = "real_data/%s/%s/APCF_%s.merged.map"

apcf_file = "real_data/%s/%s/Ancestor.APCF"
output_file = "output/corrected_estimations_alpha1_withoot_hg19.csv"

dir_est = DirichletEstimator()
uni_est = UniformEstimator()
gamma_est = GammaEstimator(1 / 3)
mor_est = MoretEstimator()
species = ['hg19', 'panTro4', 'ponAbe2', 'rheMac3', 'calJac3', 'mm10', 'rn5', 'bosTau7', 'capHir1', 'susScr3',
           'equCab2', 'canFam3', 'monDom5', 'galGal4']
apcfs = ['HUC', 'GAP', 'CAT', 'SIM', 'EUA', 'BOR', 'EUT']
bounds = [1, 2, 3, 4, 5, 7, 12, 14]
res = "500Kbp"
output_file_name = "graph_stats_%s" % res

rbs = {
    'HUC': {'hg19': 13, 'panTro4': 41},
    'GAP': {'hg19': 46, 'panTro4': 74, 'ponAbe2': 17},
    'CAT': {'hg19': 68, 'panTro4': 96, 'ponAbe2': 39, 'rheMac3': 122},
    'SIM': {'hg19': 100, 'panTro4': 128, 'ponAbe2': 71, 'rheMac3': 154, 'calJac3': 110},
    'EUA': 47,
    'BOR': 9,
    'EUT': 6
}

def process_rbs():
    def get_dict(old_d, new_d):
        dict = {}
        diff = rbs[new_d]
        for k, v in rbs[old_d].items():
            dict[k] = v + diff
        return dict

    rbs['EUA'] = get_dict('SIM', 'EUA')
    rbs['BOR'] = get_dict('EUA', 'BOR')
    rbs['EUT'] = get_dict('BOR', 'EUT')


def collect_graph_stats(g):
    cms = defaultdict(lambda: 0)
    pms = defaultdict(lambda: 0)
    for c in nx.connected_component_subgraphs(g):
        if len(c) == len(c.edges):
            cms[len(c) // 2] += 1
        else:
            pms[len(c.edges)] += 1

    return (g.n(), cms, pms)


def main():
    resutls = []
    bs_us, bs_real = [], []
    process_rbs()

    for apcf, b in zip(apcfs, bounds[1:]):
        group = species[:b]
        df = parse_to_df(df_file % (apcf, res))

        print(apcf, group)
        for sp in group:
            real_b = rbs[apcf].get(sp, -1)
            if real_b == -1:
                continue
            allowed_blocks_sp_apcf = df['block'].unique().tolist()
            allowed_blocks_sp_apcf = list(filter(uniq_predicate((sp_blocks_from_df(df, sp))), allowed_blocks_sp_apcf))

            bss = RealDataGraph.get_blocks_from_file(apcf_file % (apcf, res))
            bss_flat = [abs(b) for bs in bss for b in bs]
            allowed_blocks_sp_apcf = list(filter(lambda b: bss_flat.count(b) == 1, allowed_blocks_sp_apcf))

            print(sp, apcf)
            # df_apcf_sp = parse_to_df(df_file_way2 % (apcf, res, sp))

            g_apcf_sp = create_bg(df, bss, sp, allowed_blocks_sp_apcf)
            # g_apcf_sp1_way2 = create_bg(df_apcf_sp, "APCF", sp)

            print(g_apcf_sp.b(), real_b)
            bs_us.append(g_apcf_sp.b())
            bs_real.append(real_b)

            # resutls.append((apcf, sp, "way_1", real_b) + collect_graph_stats(g_apcf_sp))
            # resutls.append((apcf, sp, "way_2", real_b) + collect_graph_stats(g_apcf_sp1_way2))

    plt.scatter(bs_us, bs_real)
    plt.plot(range(int(max(bs_us+bs_real))))
    plt.show()

    with open(output_file_name, 'w+') as f:
        f.write(json.dumps(resutls))


if __name__ == "__main__":
    main()
