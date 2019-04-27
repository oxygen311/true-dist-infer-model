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
from scipy.stats import chisquare
import pylab
import scipy.stats as stats

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

if __name__ == "__main__":
    sns.set(style="whitegrid", font="serif", font_scale=1.2)
    plt.rc('text', usetex=True)

    df = parse_to_df("real_data/EUT/500Kbp/SFs/Conserved.Segments")

    or_sp = "hg19"
    sp = "mm10"
    df_sp = df.loc[df['species'] == sp].sort_values(by=['chr', 'chr_beg'])
    blocks_unique(df_sp)

    print("Before removing trivial cycles:", len(df_sp))
    ls = dist_between_blocks(df_sp, True)
    print("len =", len(ls), "ls = ", sorted(ls))
    remove_trivial_cycles_df(df, df_sp, or_sp, sp)
    print("After removing trivial cycles:", len(df_sp))

    ls = dist_between_blocks(df_sp, True)
    print("len =", len(ls), "ls = ", sorted(ls))

    ls = np.array(list(sorted(filter(lambda l: l > 1, ls))))
    ls = ls / sum(ls)

    # fig1, ax1 = plt.subplots()
    # h = plt.hist(ls, bins=70, alpha=0.2, edgecolor='k', cumulative=False, normed=True, label="hist")

    # print(h)

    dist = stats.gamma(a=1 / 3, loc=0, scale=3 / len(ls))

    # hf = (np.array(h[1][1:]) + np.array(h[1][:-1])) / 2
    # print(h[0])
    # print(dist.cdf(hf, 1 / 3, scale=1 / len(ls)))

    # xs = np.arange(h[1][0] + 1.5e-4, h[1][-1], 0.0001)
    # plt.plot(xs, dist.pdf(xs), label=("pdf a=%f" % 0.33333333333))

    # plt.xlabel("Adjacency length", fontsize=14)
    # plt.ylabel("Frequency", fontsize=14)

    # measurements = h[0]
    stats.probplot(ls, dist=dist, fit=False, plot=plt)
    # plt.show()
    plt.xlim(xmin=0, xmax=0.009)
    # plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=0.009)
    plt.subplots_adjust(left=0.13, bottom=0.11, top=0.98, right=0.96)

    # plt.legend(loc=1, frameon=True, prop={'size': 14})

    plt.title("")
    plt.savefig('mm10-fitting-qqplot.pdf')
    plt.show()

    # fig2, ax2 = plt.subplots()
    # ax2.scatter(h[0], dist.cdf(hf, 1 / 3, scale=1 / len(ls)))
    # ax2.plot([0, 1], [0, 1])

    # plt.plot(xs, dist.pdf(xs, 1 / 3, scale=1 / len(ls)), label=("pdf a=%f" % 0.33333333333))
    # plt.plot(xs, dist.pdf(xs, *param), label=("pdf a=%f" % 0.33333333333))
    # fig2.legend(frameon=True)
