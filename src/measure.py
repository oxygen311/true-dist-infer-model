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
X_MAX = 3
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


def run(tries, workers, a):
    # ns = defaultdict(lambda: [])
    ys = [[] for _ in xs]

    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%", datetime.now())
        # chrs = random.randint(10, 30)
        # g = GenomeGraph(N, chr)
        chr = 20
        dist = gamma.rvs(a, size=(N + chr))
        dist /= sum(dist)
        # g = GenomeGraph(N, chr, dist)
        g = DirichletBPGraph(N, dist)
        # print(g.count_cycles())
        breaks = 0

        for x_index, x in enumerate(xs):
            # if x_index % 50 == 0:
            #     print("cur x:", x)
            while breaks <= int(round(x * N / 2)):
                # g.do_k2_break()
                g.do_k2_break()
                breaks += 1
            # ys[x_index].append([N, breaks, dict(g.count_cycles())])
            ys[x_index].append([N, breaks, g.make_bg_real_data_graph().d(), g.make_bg_real_data_graph().b()])
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


def run_line_nikita(tries, workers, chr, n, a):
    res = []

    def append_to_res():
        bpg = g.make_bg_real_data_graph()
        res.append([n, breaks, chr, bpg.d(), bpg.b(), bpg.p_odd(), bpg.p_even()] +
                   [bpg.p_m(m) for m in range(p_m_max)] + [bpg.c()] +
                   [bpg.c_m(m) for m in range(1, c_m_max)])

    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%", datetime.now())
        if chr == 0:
            dist = gamma.rvs(a, size=n)
            dist /= sum(dist)
            g = DirichletBPGraph(n, dist)
        else:
            dist = gamma.rvs(a, size=(n + chrs))
            dist /= sum(dist)
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


def measure_c_nikita(tries, workers, chr, n, a=1):
    pool = mp.Pool()
    results = [pool.apply_async(run_line_nikita, [tries, workers, chr, n, a]) for _ in range(workers)]

    res = []
    for r in results:
        res.extend(r.get())

    return res


def measure_c(tries, workers, a=1):
    pool = mp.Pool()
    results = [pool.apply_async(run, [tries, workers, a]) for _ in range(workers)]

    res = [[] for _ in range(len(xs))]
    for r in results:
        for i, r in enumerate(r.get()):
            res[i].extend(r)

    print("res", res)

    return res


if __name__ == "__main__":
    TRIES = 100
    WORKERS = 4

    chrs = [1]
    ns = [1000]
    as_ = [1/3]
    # file_name = "sim_data2/diff_a_n%d_runs%d_chr%d_a%s.csv"
    file_name = "sim_data/gamma06_data_N%d_%d.txt"
    # columns = ["n", "k", "chr", "d", "b", "p_odd", "p_even"] + ["p_" + str(m) for m in range(p_m_max)] + ["c"] + [
    #     "c_" + str(m) for m in range(1, c_m_max)]

    for n, chr in zip(ns, chrs):
        # for chr in chrs:
        for a in as_:
            print("<Run with n=%d and chr=%d and a=%s>" % (n, chr, "0" + str(round(a, 1))[-1:]))
            measured_c = measure_c(TRIES, WORKERS, a)
            # with open(file_name % (n, TRIES, chr, "0" + str(round(a,1))[-1:]), 'w+') as f:
            print("WRITE")
            with open(file_name % (n, TRIES), 'w+') as f:
                f.write(json.dumps(measured_c))
            #     print(','.join(columns), file=f)
            #     for r in measured_c:
            #         print(','.join(map(str, r)), file=f)
