from src.hic.blocks_parser import parse_to_df, create_bg
import pandas as pd
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, get_d_from_cycles, get_b_from_cycles, GammaEstimator, MoretEstimator
from src.real_data import without_c1
import time
from src.real_data import RealDataGraph
import sys

sp_blocks_from_df = lambda df, sp: df.loc[df['species'] == sp]['block'].tolist()
uniq_predicate = lambda ls: lambda b: ls.count(b) == 1

# df_file = "real_data/%s/300Kbp/SFs/Orthology.Blocks"
df_file = "real_data/%s/%s/SFs/Conserved.Segments"

apcf_file = "real_data/%s/%s/Ancestor.APCF"
output_file = "output/real_data_estimations_moret.csv"

dir_est = DirichletEstimator()
uni_est = UniformEstimator()
gamma_est = GammaEstimator(1 / 3, 50)
mor_est = MoretEstimator()
species = ['hg19', 'panTro4', 'ponAbe2', 'rheMac3', 'calJac3', 'mm10', 'rn5', 'bosTau7', 'capHir1', 'susScr3',
           'equCab2', 'canFam3', 'monDom5', 'galGal4']
apcfs = ['HUC', 'GAP', 'CAT', 'SIM', 'EUA', 'BOR', 'EUT']
bounds = [1, 2, 3, 4, 5, 7, 12, 14]
res = "300Kbp"


def main():
    with open(output_file, 'w+') as f:
        print("species1", "species2", "apcf", "parsimony",
              "n",
              "d: s1 - apcf", "d: s2 - apcf", "d: s1 - s2",
              "b: s1 - apcf", "b: s2 - apcf", "b: s1 - s2",
              "unif: s1 - apcf", "unif: s2 - apcf", "unif: s1 - s2",
              "dir: s1 - apcf", "dir: s2 - apcf", "dir: s1 - s2",
              "gamma(1/3): s1 - apcf", "gamma(1/3): s2 - apcf", "gamma(1/3): s1 - s2",
              "moret: s1 - apcf", "moret: s2 - apcf", "moret: s1 - s2",
              sep=",", file=f
              )
        for apcf, b1, b2 in zip(apcfs, bounds[:-1], bounds[1:]):
            group1 = species[:b1]
            group2 = species[b1:b2]
            df = parse_to_df(df_file % (apcf, res))

            print(apcf, group1, group2)

            for sp1 in group1:
                for sp2 in group2:
                    allowed_blocks = df['block'].unique().tolist()
                    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp1))), allowed_blocks))
                    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp2))), allowed_blocks))

                    bss = RealDataGraph.get_blocks_from_file(apcf_file % (apcf, res))
                    bss_flat = [abs(b) for bs in bss for b in bs]
                    allowed_blocks = list(filter(lambda b: bss_flat.count(b) == 1, allowed_blocks))

                    print(sp1, sp2, "Allowed blocks:", len(allowed_blocks))

                    g_apcf_sp1 = create_bg(df, bss, sp1, allowed_blocks)
                    g_apcf_sp2 = create_bg(df, bss, sp2, allowed_blocks)
                    g_sp1_sp2 = create_bg(df, sp1, sp2, allowed_blocks)

                    print(sp1, sp2, apcf, (g_sp1_sp2.d() / g_sp1_sp2.b()) <= 0.75,
                          len(allowed_blocks),
                          g_apcf_sp1.d(), g_apcf_sp2.d(), g_sp1_sp2.d(),
                          g_apcf_sp1.b(), g_apcf_sp2.b(), g_sp1_sp2.b(),
                          round(uni_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(uni_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(uni_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                          round(dir_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(dir_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(dir_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                          round(gamma_est.predict_k_by_db(g_apcf_sp1.d(), g_apcf_sp1.b()), 2),
                          round(gamma_est.predict_k_by_db(g_apcf_sp2.d(), g_apcf_sp2.b()), 2),
                          round(gamma_est.predict_k_by_db(g_sp1_sp2.d(), g_sp1_sp2.b()), 2),
                          round(mor_est.predict_k_by_bn(g_apcf_sp1.b(), len(allowed_blocks)), 2),
                          round(mor_est.predict_k_by_bn(g_apcf_sp2.b(), len(allowed_blocks)), 2),
                          round(mor_est.predict_k_by_bn(g_sp1_sp2.b(), len(allowed_blocks)), 2),
                          sep=",", file=f)


if __name__ == "__main__":
    main()
