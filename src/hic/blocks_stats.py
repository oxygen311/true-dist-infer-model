import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left, bisect_right
import seaborn as sns
from scipy.stats import ks_2samp, kstest, gamma
from src.hic.blocks_parser import remove_trivial_cycles_df, parse_to_df
import scipy
from scipy.stats.stats import pearsonr
import sys



blocks_lengths = lambda df: every_chr(df, lambda df_chr: [e - s for s, e in zip(df_chr['chr_beg'], df_chr['chr_end'])])


every_chr = lambda df, chr_f: [r for chr in df['chr'].unique() for r in chr_f(df.loc[df['chr'] == chr])]

def dist_between_blocks(df, with_begs=False):
    def chr_f(df):
        ss = sorted(df['chr_beg'].tolist())
        es = sorted(df['chr_end'].tolist())
        ans = [ss[0]] if with_begs else []
        for e in es:
            l = bisect_left(ss, e)
            r = bisect_right(ss, e)
            if r != len(ss):  # and ss[l] - e != 0:
                ans.append(ss[r] - e)
        return ans
    return every_chr(df, chr_f)


def exponential_dist(n, lambd=1):
    return [random.expovariate(lambd) for _ in range(n)]


def blocks_unique(df_sp):
    bs = df_sp['block'].tolist()
    bsu = df_sp['block'].unique().tolist()

    for b in bsu:
        if bs.count(b) > 1:
            df_sp.drop(df_sp.loc[df_sp['block'] == b].index, inplace=True)


norm_and_sort = lambda xs: sorted(list(filter(lambda x: x != 0, map(lambda x: x * len(xs) * 0.3 / sum(xs), xs))))


def main():
    sns.set(style="whitegrid")
    df = parse_to_df("real_data/EUT/300Kbp/SFs/Conserved.Segments")

    or_sp = "hg19"
    sp = "mm10"
    df_sp = df.loc[df['species'] == sp].sort_values(by=['chr', 'chr_beg'])
    blocks_unique(df_sp)

    print("Before removing trivial cycles:", len(df_sp))
    ls = dist_between_blocks(df_sp, True)
    print("len =", len(ls), "ls = ", sorted(ls))
    # remove_trivial_cycles_df(df, df_sp, or_sp, sp)
    print("After removing trivial cycles:", len(df_sp))

    ls = dist_between_blocks(df_sp, True)
    print("len =", len(ls), "ls = ", sorted(ls))
    ls = norm_and_sort(ls)
    # h = plt.hist(ls, bins=70, alpha=0.2, edgecolor='k', cumulative=False, normed=True)

    ls_shifted = ls[1:] + ls[:1]
    print(np.corrcoef(ls, ls_shifted))
    print(pearsonr(ls, ls_shifted))

    dist = getattr(scipy.stats, "gamma")
    param = dist.fit(ls)
    print("<params>", param)

    # ds = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=len(ls))
    ds = dist.rvs(*param[:-2], size=len(ls), loc=param[-2])
    # ds = norm_and_sort(ds)

    print(ks_2samp(ds, ls))
    print(kstest(ls, "gamma", [*param[:-2], param[-2]]))

    ds = dist.rvs(0.3, size=len(ls), loc=param[-2])
    print(ks_2samp(ds, ls))
    print(kstest(ls, "gamma", [0.3, param[-2]]))

    ds = dist.rvs(0.4, size=len(ls), loc=param[-2])
    print(ks_2samp(ds, ls))
    print(kstest(ls, "gamma", [0.4, param[-2]]))

    # plt.plot(sorted(ls), range(len(ls)), label="blocks")

    xs = np.arange(0.01, 4.4, 0.01)
    plt.plot(xs, dist.pdf(xs, *param[:-2], loc=param[-2]), label=("pdf a=%f" % param[0]))
    plt.plot(xs, dist.pdf(xs, 1/3, loc=param[-2]), label=("pdf a=%f" % 0.33))
    plt.legend(frameon=True)
    # plt.savefig("mm8_pdf.pdf")
    # plt.show(legend_loc=4)


# main()
