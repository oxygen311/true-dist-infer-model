from src.hic.blocks_parser import parse_to_df, create_bg
import pandas as pd
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, get_d_from_cycles, get_b_from_cycles, GammaEstimator, MoretEstimator, \
    CorrectedGammaEstimator, RangeGammaEstimator
from src.real_data import without_c1
import time
from src.real_data import RealDataGraph
import sys
import json
import numpy as np

sp_blocks_from_df = lambda df, sp: df.loc[df['species'] == sp]['block'].tolist()
uniq_predicate = lambda ls: lambda b: ls.count(b) == 1

# df_file = "real_data/%s/300Kbp/SFs/Orthology.Blocks"
df_file = "real_data/%s/%s/SFs/Conserved.Segments"
df_file_way2 = "real_data/%s/%s/APCF_%s.merged.map"

apcf_file = "real_data/%s/%s/Ancestor.APCF"
output_file = "output/range_estimations_2_5_97_5_2.csv"

dir_est = DirichletEstimator()
uni_est = UniformEstimator()
gamma_est = GammaEstimator(1 / 3)
mor_est = MoretEstimator()

# gamma_data_file = "sim_data/gamma03_data_N1000_200.txt"
linear_gamma_data_file = "sim_data/lin_0303_chr20_data_N1000_100.txt"
# gamma_data = json.loads(open(gamma_data_file, 'r').read())
linear_gamma_data = json.loads(open(linear_gamma_data_file, 'r').read())
range_gamma_est = RangeGammaEstimator(0.3, linear_gamma_data, 2.5, mode="linear")

dir_data_file = lite_dirichlet_file = "sim_data/dirichlet_data_randomN_200.txt"
dir_data = json.loads(open(dir_data_file, 'r').read())
range_dir_est = RangeGammaEstimator(1, dir_data, 10)

species = ['hg19', 'panTro4', 'ponAbe2', 'rheMac3', 'calJac3', 'mm10', 'rn5', 'bosTau7', 'capHir1', 'susScr3',
           'equCab2', 'canFam3', 'monDom5', 'galGal4']
apcfs = ['HUC', 'GAP', 'CAT', 'SIM', 'EUA', 'BOR', 'EUT']
bounds = [1, 2, 3, 4, 5, 7, 12, 14]
res = "300Kbp"


def print_graph_stats(name, g):
    print(name + ":")
    print(int(g.n()), end=";")
    cms = []
    for m in range(int(g.n())):
        if g.c_m(m) != 0:
            cms.append((m, g.c_m(m)))
    print(", ".join([f"{k}: {v}" for k, v in cms]), end=";")

    pms = []
    for m in range(int(g.n())):
        if g.p_m(m) != 0:
            pms.append((m, g.p_m(m)))
    print(", ".join([f"{k}: {v}" for k, v in pms]))


def main():
    with open(output_file, 'w+') as f:
        # print("species1", "species2", "apcf", "parsimony",
        #       "n",
        #       "d: s1 - apcf", "d: s2 - apcf", "d: s1 - s2",
        #       "b: s1 - apcf", "b: s2 - apcf", "b: s1 - s2",
        #       "unif: s1 - apcf", "unif: s2 - apcf", "unif: s1 - s2",
        #       "dir: s1 - apcf", "dir: s2 - apcf", "dir: s1 - s2",
        #       "gamma(1/3): s1 - apcf", "gamma(1/3): s2 - apcf", "gamma(1/3): s1 - s2",
        #       "moret: s1 - apcf", "moret: s2 - apcf", "moret: s1 - s2",
        #       "cor_gamma: s1 - apcf", "cor_gamma: s2 - apcf", "cor_gamma: s1 - s2",
        #       sep=",", file=f
        #       )
        print("species1", "species2", "apcf", "parsimony",
              "n",
              "d: s1 - apcf", "d: s2 - apcf", "d: s1 - s2",
              "b: s1 - apcf", "b: s2 - apcf", "b: s1 - s2",
              "dir_min: s1 - apcf", "dir_min: s2 - apcf", "dir_min: s1 - s2",
              "dir_max: s1 - apcf", "dir_max: s2 - apcf", "dir_max: s1 - s2",
              "gamma0.3_min: s1 - apcf", "gamma0.3_min: s2 - apcf", "gamma0.3_min: s1 - s2",
              "gamma0.3_max: s1 - apcf", "gamma0.3_max: s2 - apcf", "gamma0.3_max: s1 - s2",
              sep=",", file=f
              )
        xs = []
        ys = []
        unique_blocks_part = []
        for apcf, b1, b2 in zip(apcfs, bounds[:-1], bounds[1:]):
            group1 = species[:b1]
            group2 = species[b1:b2]
            df = parse_to_df(df_file % (apcf, res))

            print(apcf, group1, group2)

            for sp1 in group1:
                for sp2 in group2:
                    # if sp1 == "hg19" or sp2 == "hg19":
                    #     continue
                    allowed_blocks = df['block'].unique().tolist()
                    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp1))), allowed_blocks))
                    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp2))), allowed_blocks))

                    bss = RealDataGraph.get_blocks_from_file(apcf_file % (apcf, res))
                    bss_flat = [abs(b) for bs in bss for b in bs]

                    allowed_blocks = list(filter(lambda b: bss_flat.count(b) == 1, allowed_blocks))

                    # if sp1 == "hg19":
                    #     continue
                    # print(sp1, sp2, apcf, "Allowed blocks:", len(allowed_blocks))
                    # print(sp1, sp2, apcf)
                    # sp1l = len(sp_blocks_from_df(df, sp1))
                    # sp2l = len(sp_blocks_from_df(df, sp2))
                    # apcfl = len(bss_flat)
                    # abl = len(allowed_blocks)
                    # unique_blocks_part.append((sp1l - abl) * 100 / sp1l)
                    # unique_blocks_part.append((sp2l - abl) * 100 / sp2l)
                    # unique_blocks_part.append((apcfl - abl) * 100 / apcfl)
                    # print(f"    Blocks in sp1: {(sp1l - abl) / sp1l},\n"
                    #       f"    Blocks in sp2: {(sp2l - abl) / sp2l},\n"
                    #       f"    Blocks in APCF: {(apcfl - abl) / apcfl}")

                    # df_apcf_sp1 = parse_to_df(df_file_way2 % (apcf, res, sp1))
                    # df_apcf_sp2 = parse_to_df(df_file_way2 % (apcf, res, sp2))

                    g_apcf_sp1 = create_bg(df, bss, sp1, allowed_blocks)  # , allowed_blocks
                    g_apcf_sp2 = create_bg(df, bss, sp2, allowed_blocks)  # , allowed_blocks
                    g_sp1_sp2 = create_bg(df, sp1, sp2, allowed_blocks)  # , allowed_blocks

                    # g_apcf_sp1_way2 = create_bg(df_apcf_sp1, "APCF", sp2)
                    # g_apcf_sp2_way2 = create_bg(df_apcf_sp2, "APCF", sp2)

                    cor_gamma_est = CorrectedGammaEstimator(1, 23 / len(allowed_blocks))

                    n, k = cor_gamma_est.predict_by_db(g_sp1_sp2.d(), g_sp1_sp2.b())

                    x = 2 * k / n
                    ys.append((g_apcf_sp1.d() + g_apcf_sp2.d()) / n)
                    xs.append(x)

                    # print_graph_stats(f"{apcf} - {sp1}", g_apcf_sp1)
                    # print_graph_stats(f"{apcf} - {sp1} (way2)", g_apcf_sp1_way2)
                    #
                    # print_graph_stats(f"{apcf} - {sp2}", g_apcf_sp2)
                    # print_graph_stats(f"{apcf} - {sp2} (way2)", g_apcf_sp2_way2)

                    # print(sp1, sp2, apcf, (g_sp1_sp2.d() / g_sp1_sp2.b()) <= 0.75,
                    #       len(allowed_blocks), "\n",
                    #       g_apcf_sp1.d(), g_apcf_sp2.d(), g_sp1_sp2.d(), "\n",
                    #       g_apcf_sp1.b(), g_apcf_sp2.b(), g_sp1_sp2.b(),
                    #       # round(uni_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                    #       # round(uni_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                    #       # round(uni_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                    #       round(dir_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                    #       round(dir_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                    #       round(dir_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                    #       round(gamma_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                    #       round(gamma_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                    #       round(gamma_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                    #       round(mor_est.predict_k_by_bn(g_apcf_sp1.b(), len(allowed_blocks)), 2),
                    #       round(mor_est.predict_k_by_bn(g_apcf_sp2.b(), len(allowed_blocks)), 2),
                    #       round(mor_est.predict_k_by_bn(g_sp1_sp2.b(), len(allowed_blocks)), 2),
                    #       round(cor_gamma_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                    #       round(cor_gamma_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                    #       round(cor_gamma_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                    #       sep=",") #, file=f)

                    print(sp1, sp2, apcf, (g_sp1_sp2.d() / g_sp1_sp2.b()) <= 0.75,
                          len(allowed_blocks),
                          g_apcf_sp1.d(), g_apcf_sp2.d(), g_sp1_sp2.d(),
                          g_apcf_sp1.b(), g_apcf_sp2.b(), g_sp1_sp2.b(),

                          round(range_dir_est.predict_k_min_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(range_dir_est.predict_k_min_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(range_dir_est.predict_k_min_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),

                          round(range_dir_est.predict_k_max_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(range_dir_est.predict_k_max_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(range_dir_est.predict_k_max_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),

                          round(range_gamma_est.predict_k_min_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(range_gamma_est.predict_k_min_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(range_gamma_est.predict_k_min_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),

                          round(range_gamma_est.predict_k_max_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(range_gamma_est.predict_k_max_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(range_gamma_est.predict_k_max_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),

                          # round(gamma_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          # round(gamma_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          # round(gamma_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),

                          sep=",", file=f)

        # print(np.min(unique_blocks_part),
        #       np.percentile(unique_blocks_part, 25),
        #       np.mean(unique_blocks_part),
        #       np.percentile(unique_blocks_part, 75),
        #       np.max(unique_blocks_part))
        print(xs)
        print(ys)


if __name__ == "__main__":
    main()
