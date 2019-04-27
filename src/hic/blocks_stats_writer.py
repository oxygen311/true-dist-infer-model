import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left, bisect_right
import seaborn as sns
from scipy.stats import ks_2samp, kstest, gamma
from src.hic.blocks_parser import remove_trivial_cycles_df, parse_to_df
import scipy
from src.hic.blocks_stats import every_chr, blocks_lengths, dist_between_blocks, exponential_dist, norm_and_sort, blocks_unique
import sys
from scipy.stats import chisquare



def main():
    df = parse_to_df("real_data/EUT/300Kbp/SFs/Conserved.Segments")
    # sys.stdout = open("dist_fitting_500.txt", 'w')
    or_sp = "hg19"
    sps = ['hg19', 'panTro4', 'ponAbe2', 'rheMac3', 'calJac3', 'mm10', 'rn5', 'bosTau7', 'capHir1', 'susScr3', 'equCab2', 'canFam3', 'monDom5', 'galGal4']
    # sps = ['monDom5']

    for sp in sps:
        # if or_sp == sp:
        #     continue
        df_sp = df.loc[df['species'] == sp].sort_values(by=['chr', 'chr_beg'])
        blocks_unique(df_sp)

        remove_trivial_cycles_df(df, df_sp, or_sp if or_sp != sp else 'ponAbe2', sp)

        ls = dist_between_blocks(df_sp, with_begs=False)
        ls = np.array(list(sorted(filter(lambda l: l > 1, ls))))
        ls = ls / sum(ls)
        # ls = list(sorted(filter(lambda l: l != 0, map(lambda l: l / sum(ls), ls))))

        dist = getattr(scipy.stats, "gamma")
        param = dist.fit(ls, floc=0)
        a = 1/3

        # chisquare
        # h = plt.hist(ls, bins=100)
        # cs_exp = [dist.cdf(e, a, loc=param[-2], scale=param[-1]) - dist.cdf(s, a, loc=param[-2], scale=param[-1]) for s, e in zip(h[1][:-1], h[1][1:])]
        # cs_exp = list(map(lambda c: c / sum(cs_exp) * sum(h[0]), cs_exp))

        # print([dist.cdf(e, param[0]) - dist.cdf(s, param[0]) for s, e in zip(h[1][:-1], h[1][1:])])
        # print(h[0])
        # param[0] = 1/3
        # print(param)
        # param = dist.fit(ls, floc=0, fscale=1)
        # print(sum(ls))
        # ds = dist.rvs(*param[:-2], size=len(ls))
        ds = dist.rvs(a, scale=3 / len(ls), size=len(ls))
        # print("sum ds", sum(ds), sum(ls))


        print("%s: a_fitted=%.3f, a_used=%.3f, blocks=%d, ks_2samp=%.3f, kstest=%.3f" % (sp, param[0], a, len(ls), ks_2samp(ds, ls)[1], kstest(ls, "gamma", (a, 0, 3 / len(ls)))[1]))
        # print(chisquare(h[0], cs_exp))
        # print()
        # print(param)
        # break

main()
