from src.dirichlet_bp_graph import DirichletBPGraph
from src.classic_bp_graph import BPGraph
import numpy as np
import json
from datetime import datetime
from src.estimators import get_d_from_cycles
import random
import multiprocessing as mp
from src.genome_graph import GenomeGraph
from collections import defaultdict
from scipy.stats import gamma

X_MIN = 0
X_MAX = 1.2
X_STEP = 0.01
N = 1000
xs = np.arange(X_MIN, X_MAX, X_STEP)
x = 0.75
p_m_max = 11
c_m_max = 9


def run_fix_x_diff_n(tries, workers):
    ns = defaultdict(lambda: [])

    for i in range(int(tries / workers)):
        if i != 0:
            print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%")
        for n in range(1000, 1001):
            g = DirichletBPGraph(n)
            breaks = 0

            while breaks <= int(round(x * n / 2)):
                g.do_k_break()
                breaks += 1
            ns[n].append([n, breaks, dict(g.count_cycles())])
    return dict(ns)


def run(tries, workers, chr):
    # ns = defaultdict(lambda: [])
    ys = [[] for _ in xs]

    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%", datetime.now())
        # chrs = random.randint(10, 30)
        # g = GenomeGraph(N, chr)
        a = 0.3
        dist = gamma.rvs(a, scale=1 / a, size=N)
        print(sum(dist))
        dist /= sum(dist)
        print(sum(dist))
        g = DirichletBPGraph(N, dist)
        # print(g.count_cycles())
        breaks = 0

        for x_index, x in enumerate(xs):
            # if x_index % 50 == 0:
            #     print("cur x:", x)
            while breaks <= int(round(x * N / 2)):
                # g.do_k2_break()
                g.do_k_break()
                breaks += 1
            ys[x_index].append([N, breaks, dict(g.count_cycles())])
    print(ys)
    return ys


def run_line(tries, workers, chr):
    ys = [[] for _ in xs]

    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%", datetime.now())
        g = GenomeGraph(N, chr)
        breaks = 0

        for x_index, x in enumerate(xs):
            # if breaks % 300 == 0:
            #     print("   ", breaks, " breaks are done")
            while breaks <= int(round(x * N / 2)):
                # print("____K =", breaks + 1)
                g.do_k2_break()
                breaks += 1
            ys[x_index].append([N, breaks, g.d(), g.b()])
    # print(ys)
    return ys


def run_line_nikita(tries, workers, chr, n):
    res = []

    def append_to_res():
        bpg = g.make_bg_real_data_graph()
        res.append([n, breaks, chr, bpg.d(), bpg.b(), bpg.p_odd(), bpg.p_even()] +
                   [bpg.p_m(m) for m in range(p_m_max)] + [bpg.c()] +
                   [bpg.c_m(m) for m in range(1, c_m_max)])

    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%", datetime.now())
        g = GenomeGraph(n, chr)
        breaks = 0

        for x_index, x in enumerate(xs):
            append_to_res()
            while breaks <= int(round(x * n / 2)):
                # print("____K =", breaks + 1, "of", int(round(X_MAX * n / 2)))
                g.do_k2_break()
                breaks += 1
            append_to_res()
    # print(ys)
    return res


def measure_c(tries, workers, chr, n):
    pool = mp.Pool()
    results = [pool.apply_async(run_line_nikita, [tries, workers, chr, n]) for _ in range(workers)]

    res = []
    for r in results:
        res.extend(r.get())

    print("res", res)

    return res


if __name__ == "__main__":
    TRIES = 100
    WORKERS = 4

    chrs = [10, 20]
    ns = [1000]
    file_name = "sim_data2/dir_dist_n%d_runs%d_chr%d.csv"
    columns = ["n", "k", "chr", "d", "b", "p_odd", "p_even"] + ["p_" + str(m) for m in range(p_m_max)] + ["c"] + [
        "c_" + str(m) for m in range(1, c_m_max)]

    for n in ns:
        for chr in chrs:
            print("<Run with n=%d and chr=%d>" % (n, chr))
            measured_c = measure_c(TRIES, WORKERS, chr, n)
            with open(file_name % (n, TRIES, chr), 'w+') as f:
                print(','.join(columns), file=f)
                for r in measured_c:
                    print(','.join(map(str, r)), file=f)
