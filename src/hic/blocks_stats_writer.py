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


def main():
    df = parse_to_df("real_data/EUT/300Kbp/SFs/Conserved.Segments")
    # sys.stdout = open("dist_fitting_500.txt", 'w')
    or_sp = "hg19"
    sps = ['panTro4', 'ponAbe2', 'rheMac3', 'calJac3', 'mm10', 'rn5', 'bosTau7', 'capHir1', 'susScr3', 'equCab2', 'canFam3', 'monDom5', 'galGal4']
    # sps = ['monDom5']

    for sp in sps:
        df_sp = df.loc[df['species'] == sp].sort_values(by=['chr', 'chr_beg'])
        blocks_unique(df_sp)

        # remove_trivial_cycles_df(df, df_sp, or_sp, sp)

        ls = dist_between_blocks(df_sp, with_begs=False)
        ls = list(sorted(filter(lambda l: l != 0, map(lambda l: l, ls))))

        dist = getattr(scipy.stats, "gamma")
        param = dist.fit(ls, floc=0)
        # param = dist.fit(ls, floc=0, fscale=1)
        # print(param)
        # ds = dist.rvs(*param[:-2], size=len(ls))
        ds = dist.rvs(*param, size=len(ls))

        print("%s: a=%.3f, blocks=%d, ks_2samp=%.3f, kstest=%.3f" % (sp, param[0], len(ls), ks_2samp(ds, ls)[1], kstest(ls, "gamma", param)[1]))
        # print(param)

main()
